"""Read-only snapshot geometry registered into the fixed human task frame.

Each saved snapshot receives one fixed rigid SE(2) placement. Its true head-point
velocity rotates as a vector; no pelvis velocity is subtracted and no head height
is fitted. These are target candidates, not references or physical admissions.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from itertools import product
import json
import math
from pathlib import Path
import resource
import time

import numpy as np

from .phase2_domain import file_hash, fingerprint
from .wbc.splits import assign_trajectory_groups

CALIBRATION_SLOT = 52912
CALIBRATION_COMMAND = 3307
MAX_OFFSET_M = .05
MAX_YAW_RAD = math.radians(5.)


def _finite(values,size,name):
    result=tuple(float(x) for x in values)
    if len(result) != size or not all(map(math.isfinite,result)):
        raise ValueError(f"{name} requires {size} finite values")
    return result


def yaw_matrix(angle):
    if not math.isfinite(angle):raise ValueError("Finite yaw required")
    c,s=math.cos(angle),math.sin(angle)
    return np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])


def pelvis_yaw(quaternion):
    q=np.asarray(_finite(quaternion,4,"root quaternion"))
    norm=float(np.linalg.norm(q))
    if abs(norm-1.) > 1.e-6:raise ValueError("Unit wxyz root quaternion required")
    w,x,y,z=q/norm
    # Normalize only for robust heading extraction; placement left-multiplies
    # the original saved quaternion and preserves its native numerical norm.
    return math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))


def _yaw_multiply(angle,quaternion):
    w,x,y,z=np.asarray(quaternion,float)
    c,s=math.cos(angle/2),math.sin(angle/2)
    return np.array([c*w-s*z,c*x-s*y,c*y+s*x,c*z+s*w])


@dataclass(frozen=True)
class Placement:
    forward_m: float = 0.
    left_m: float = 0.
    yaw_rad: float = 0.

    def __post_init__(self):
        if (not all(map(math.isfinite,(self.forward_m,self.left_m,self.yaw_rad)))
                or abs(self.forward_m) > MAX_OFFSET_M or abs(self.left_m) > MAX_OFFSET_M
                or abs(self.yaw_rad) > MAX_YAW_RAD):
            raise ValueError("Placement outside initial +/-0.05 m and +/-5 degree candidate bounds")


def placement_candidates():
    """Nominal first, then all remaining corners/edges of the declared 3^3 grid."""
    rows=[Placement(x,y,yaw) for x,y,yaw in product((-.05,0.,.05),(-.05,0.,.05),(-MAX_YAW_RAD,0.,MAX_YAW_RAD))]
    return tuple(sorted(rows,key=lambda p:(p != Placement(),p.forward_m,p.left_m,p.yaw_rad)))


@dataclass(frozen=True)
class Registration:
    calibration_slot_id: int
    calibration_command_id: int
    calibration_replica_id: int
    calibration_payload_sha256: str
    calibration_root_qpos: tuple[float, ...]
    calibration_head_w: tuple[float,float,float]
    original_target_w: tuple[float,float,float]
    opponent_heading_rad: float
    root_xy_anchor: tuple[float,float]

    def __post_init__(self):
        object.__setattr__(self,"calibration_root_qpos",_finite(self.calibration_root_qpos,7,"calibration root"))
        for key in ("calibration_head_w","original_target_w"):
            object.__setattr__(self,key,_finite(getattr(self,key),3,key))
        object.__setattr__(self,"root_xy_anchor",_finite(self.root_xy_anchor,2,"anchor"))
        if ((self.calibration_slot_id,self.calibration_command_id,self.calibration_replica_id) != (52912,3307,0)
                or len(self.calibration_payload_sha256) != 64
                or any(c not in "0123456789abcdef" for c in self.calibration_payload_sha256)
                or self.opponent_heading_rad != math.pi):
            raise ValueError("Use the fixed reviewed standing calibration and opponent heading pi")
        pelvis_yaw(self.calibration_root_qpos[3:7])
        angle=self.opponent_heading_rad-pelvis_yaw(self.calibration_root_qpos[3:7])
        root_plane=np.r_[self.calibration_root_qpos[:2],0.]
        offset=yaw_matrix(angle)@(np.asarray(self.calibration_head_w)-root_plane)
        expected=np.asarray(self.original_target_w)[:2]-offset[:2]
        if not np.allclose(expected,self.root_xy_anchor,atol=1.e-12,rtol=0):
            raise ValueError("Horizontal anchor must derive from the fixed calibration")

    @property
    def identity(self):return fingerprint(asdict(self))


def calibrate_registration(*, calibration_qpos, calibration_head_w, calibration_payload_sha256,
                           original_target_w):
    q=_finite(calibration_qpos,36,"G1 qpos")
    head=np.asarray(_finite(calibration_head_w,3,"measured head center"))
    target=np.asarray(_finite(original_target_w,3,"original frozen target"))
    rotation=yaw_matrix(math.pi-pelvis_yaw(q[3:7]))
    anchor=target[:2]-(rotation@(head-np.r_[q[:2],0.]))[:2]
    return Registration(CALIBRATION_SLOT,CALIBRATION_COMMAND,0,calibration_payload_sha256,
                        q[:7],tuple(head),tuple(target),math.pi,tuple(anchor))


def register_snapshot(registration: Registration, *, qpos, qvel, head_position_w,
                      head_velocity_w, placement: Placement = Placement()) -> dict:
    """Apply a fixed yaw/translation to a complete instantaneous snapshot.

    Placement uses human task axes +x forward,+y left,+z up. Yaw offsets rotate
    the whole opponent about its own registered pelvis, not about the head. The
    transform is fixed for this frozen snapshot; it is not a moving body frame.
    """
    if not isinstance(registration,Registration) or not isinstance(placement,Placement):
        raise ValueError("Explicit immutable registration and bounded placement required")
    q=np.asarray(_finite(qpos,36,"G1 qpos"));v=np.asarray(_finite(qvel,35,"G1 qvel"))
    head=np.asarray(_finite(head_position_w,3,"actual head center"))
    point_velocity=np.asarray(_finite(head_velocity_w,3,"actual Jp qvel head velocity"))
    source_yaw=pelvis_yaw(q[3:7])
    angle=registration.opponent_heading_rad+placement.yaw_rad-source_yaw
    rotation=yaw_matrix(angle)
    root_target=np.r_[np.asarray(registration.root_xy_anchor)+[placement.forward_m,placement.left_m],0.]
    translation=root_target-rotation@np.r_[q[:2],0.]
    registered_q=q.copy();registered_v=v.copy()
    registered_q[:3]=rotation@q[:3]+translation
    registered_q[3:7]=_yaw_multiply(angle,q[3:7])
    registered_v[:3]=rotation@v[:3]
    # Native free-root angular qvel is body-local; a global rigid yaw does not
    # rotate these three coordinates. Joint velocities also remain unchanged.
    target=rotation@head+translation
    velocity=rotation@point_velocity
    return {"registration_identity":registration.identity,"placement":asdict(placement),
        "source_pelvis_yaw_rad":source_yaw,"world_yaw_rotation_rad":angle,
        "rotation_world_to_registered":rotation.tolist(),"translation_world_to_registered":translation.tolist(),
        "qpos_registered":registered_q.tolist(),"qvel_registered":registered_v.tolist(),
        "head_target_w":target.tolist(),"head_point_velocity_w":velocity.tolist(),
        "source_head_height_m":float(head[2]),"registered_head_height_m":float(target[2]),
        "root_translation_z_m":float(translation[2]),
        "physics_executed":False,"physical_qualification":False,"live_g1_actor":False}


def build_target_geometry(*, project_root: Path, domain_path: Path, output: Path):
    from .wbc.snapshots import SnapshotBank,head_point_state,load_verified_g1_model

    root,domain_path,output=Path(project_root).resolve(),Path(domain_path).resolve(strict=True),Path(output).resolve()
    if not output.is_relative_to(root/"outputs/task10b") or output.exists():
        raise ValueError("Fresh human-owned Task10B output required")
    if not domain_path.is_relative_to(root/"outputs/task10b"):
        raise ValueError("Bind the predeclared Phase2 domain")
    output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    domain=json.loads(domain_path.read_text())
    if domain.get("schema") != "Task10BPhase2Domain/v1" or domain.get("physical_domain_qualified") is not False:
        raise ValueError("Expected predeclared geometry-only snapshot domain")
    bank=SnapshotBank(root.parent/"robust_sav/constants/boxing/snapshots/v1_20260921")
    model,asset_provenance=load_verified_g1_model(root/".cache/deps/mjlab-8ee51fbc/src/mjlab/asset_zoo/robots/unitree_g1",
        bank,root.parent/"robust_sav/constants/boxing/stance/v1_20260921/nominal_plant.json")
    calibration=bank.state(CALIBRATION_SLOT)
    if (calibration.command_id,calibration.replica_id,calibration.time_index) != (3307,0,1):
        raise ValueError("Calibration snapshot identity differs")
    cal_head,cal_velocity=head_point_state(model,calibration.qpos,calibration.qvel)
    reference=root/"outputs/four-punch-milestone/jab/reference.npz"
    release=json.loads((root/"outputs/four-punch-milestone/manifest.json").read_text())
    if file_hash(reference) != release["artifacts"]["jab/reference.npz"]["sha256"]:
        raise ValueError("Accepted original target source changed")
    with np.load(reference,allow_pickle=False) as archive:original_target=archive["head_center"].copy()
    registration=calibrate_registration(calibration_qpos=calibration.qpos,calibration_head_w=cal_head,
        calibration_payload_sha256=calibration.payload_sha256,original_target_w=original_target)
    registered_cal=register_snapshot(registration,qpos=calibration.qpos,qvel=calibration.qvel,
        head_position_w=cal_head,head_velocity_w=cal_velocity)
    rows=[];maximum_position_error=maximum_velocity_error=0.;latencies=[]
    selected=domain["selected_command_groups"]
    if (Counter(group["split"] for group in selected) != {"train":32,"validation":8,"test":16}
            or len({group["snapshot"]["command_id"] for group in selected}) != 56):
        raise ValueError("Preserve all 32/8/16 distinct predeclared command groups")
    domain_hash=file_hash(domain_path)
    counts={"train":0,"validation":0,"test":0}
    for group in selected:
        snapshot=group["snapshot"];state=bank.state(snapshot["slot_id"])
        if (state.command_id == 3307 or state.command_id != snapshot["command_id"]
                or state.original_split != group["split"]
                or assign_trajectory_groups([state.command_id])[state.command_id] != group["split"]
                or state.payload_sha256 != snapshot["payload_sha256"]
                or not np.array_equal(state.qpos,snapshot["qpos"]) or not np.array_equal(state.qvel,snapshot["qvel"])):
            raise ValueError("Frozen domain grouping/payload changed")
        head,velocity=head_point_state(model,state.qpos,state.qvel)
        if (not np.allclose(head,group["head_position_w"],atol=1.e-10,rtol=0)
                or not np.allclose(velocity,group["head_velocity_w"],atol=1.e-10,rtol=0)):
            raise ValueError("Snapshot head/Jacobian geometry differs from declared domain")
        for placement in placement_candidates():
            start=time.perf_counter()
            transformed=register_snapshot(registration,qpos=state.qpos,qvel=state.qvel,
                head_position_w=head,head_velocity_w=velocity,placement=placement)
            latencies.append(time.perf_counter()-start)
            check_head,check_velocity=head_point_state(model,transformed["qpos_registered"],transformed["qvel_registered"])
            position_error=float(np.max(abs(check_head-transformed["head_target_w"])))
            velocity_error=float(np.max(abs(check_velocity-transformed["head_point_velocity_w"])))
            if position_error > 1.e-8 or velocity_error > 1.e-8:
                raise ValueError("Registered full-state FK/Jacobian does not match rigid target placement")
            maximum_position_error=max(maximum_position_error,position_error)
            maximum_velocity_error=max(maximum_velocity_error,velocity_error)
            record={"source_domain_sha256":domain_hash,"split":group["split"],
                "domain_stage":group["domain_stage"],"command_id":state.command_id,
                "source_group_id":state.source_group_id,"snapshot_slot_id":state.slot_id,
                "original_replica_id":state.replica_id,"time_index":state.time_index,
                "payload_sha256":state.payload_sha256,"measured_motion":group["measured_velocity_body_vx_vy_native_gyro_z"],
                **transformed,"candidate_status":"target_geometry_only_unqualified",
                "full_state_fk_max_error_m":position_error,"full_state_jacobian_velocity_max_error_m_s":velocity_error}
            record["candidate_identity"]=fingerprint(record);rows.append(record)
            counts[group["split"]]+=1
    artifact={"schema":"Task10BPhase2RegisteredTargets/v1","registration":asdict(registration),
        "registration_identity":registration.identity,"geometry_candidates":rows,"candidate_count":len(rows),
        "selected_command_group_count":len(selected),"placements_per_group":27,"split_candidate_counts":counts,
        "all_command_variants_remain_in_source_split":True,"physical_qualification":False,
        "g1_actor_executed":False,"physics_integrated":False,"reference_generation_executed":False,
        "cached_preparation_real_time_qualified":False}
    target_path=output/"target_geometry.json"
    with target_path.open("x") as stream:json.dump(artifact,stream,indent=2,allow_nan=False);stream.write("\n")
    receipt={"schema":"Task10BPhase2TargetRegistrationDesign/v1","source_sha256":file_hash(Path(__file__)),
        "domain_path":str(domain_path),"domain_sha256":domain_hash,
        "target_artifact_sha256":file_hash(target_path),"target_artifact_bytes":target_path.stat().st_size,
        "calibration_snapshot":asdict(calibration),"calibration_head_w":cal_head.tolist(),
        "calibration_head_velocity_w":cal_velocity.tolist(),"registered_calibration":registered_cal,
        "accepted_original_target_source":str(reference),"accepted_original_target_sha256":file_hash(reference),
        "calibration_vertical_difference_from_original_target_m":float(cal_head[2]-original_target[2]),
        "g1_asset_provenance":asset_provenance,"placement_axes":"fixed human task +x forward,+y left,+z up",
        "calibration_semantics":"Command3307 replica0 slot52912 at0.5s is calibration only, with nonzero measured motion; no equilibrium or independent split example claim",
        "registration_semantics":"Remove each snapshot's arbitrary world rootxy and heading, then place at common calibrated rootxy anchor with opponent heading pi; yaw/xy perturbations are rigid whole-opponent placements about its pelvis",
        "height_semantics":"No vertical translation, head-height fit, pelvis-height assumption, human root translation or pose swap",
        "velocity_semantics":"Apply the same fixed rotation to true head-point Jp(q)qvel and world linear root velocity; native local angular qvel and joint qvel unchanged. No root-velocity subtraction, heading-rate subtraction or predictive lead",
        "evaluation_semantics":"Static frozen target geometry for future specialist qualification; no moving G1 execution, scene integration or SAV outcomes",
        "maximum_full_state_fk_error_m":maximum_position_error,
        "maximum_full_state_jacobian_velocity_error_m_s":maximum_velocity_error,
        "helper_latency_excludes_fk_and_io_s":{"median":float(np.median(latencies)),"maximum":max(latencies)},
        "generation_wall_s":time.perf_counter()-started,
        "process_high_water_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        "simulation_pauses":"not applicable; no integration executed",
        "physical_qualification":False,"scheduler_or_live_g1_implementation":False,
        "limitations":["Originaltargetxy anchors a candidate placement; newheadheight/geometry still require physical qualification.",
            "Helper computation timing does not establish20ms final-reference preparation or delayed-execution real-time behavior.",
            "No family eligibility, reference, connector, or shared-start claim follows from these target geometries."]}
    with (output/"design_provenance.json").open("x") as stream:json.dump(receipt,stream,indent=2,allow_nan=False);stream.write("\n")
    return {"output":str(output),"candidate_count":len(rows),"splits":counts,
            "artifact_sha256":file_hash(target_path),"physical_qualification":False}


def main():
    root=Path(__file__).resolve().parents[2]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain",type=Path,default=root/"outputs/task10b/p2domain01/domain.json")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(build_target_geometry(project_root=root,domain_path=args.domain,output=args.output),indent=2))


if __name__ == "__main__":main()
