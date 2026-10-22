"""Compare original DuoBox source with H2 references using actual collision geoms.

Diagnostics use FK and static collision detection only, never physics stepping.
Rendering requires an allocated GPU. Source sticks/axes are measured skeleton
frames, not a reconstruction of human gloves or physical head volume.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from human_punching.motion import JOINT_NAMES as SOURCE_NAMES, PARENTS, load_crop
from human_punching.task10c.robot import H2_XML, ROOT, get_h2_robot_cfg


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Comparison outputs must remain inside human-punching")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def interpolate(t, values, query):
    flat = values.reshape(len(t), -1)
    return np.array([np.interp(query, t, flat[:, i]) for i in range(flat.shape[1])]).reshape(values.shape[1:])


def angle_deg(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / max(denom, 1e-12), -1, 1))))


@dataclass
class Reference:
    path: Path
    model_path: Path
    config_path: Path

    def __post_init__(self):
        with np.load(self.path, allow_pickle=False) as source:
            self.arrays = {key: source[key].copy() for key in source.files}
        self.metadata = json.loads(self.path.with_suffix(".json").read_text())
        self.config = json.loads(self.config_path.read_text())
        self.frames = self.arrays["source_frames"]
        if np.any(np.diff(self.frames) < 0):
            raise ValueError("Comparison requires nondecreasing recorded source frames")
        # Same controller/collision configuration as training, explicit variant XML.
        cfg = get_h2_robot_cfg()
        cfg.spec_fn = lambda: mujoco.MjSpec.from_file(str(self.model_path.resolve()))
        self.model = cfg.build().compile()
        self.data = mujoco.MjData(self.model)
        if tuple(self.model.joint(i).name for i in range(1, self.model.njnt)) != tuple(self.arrays["joint_names"]):
            raise ValueError("Reference and compiled model named joint order differ")
        self.body_ids = {self.model.body(i).name: i for i in range(self.model.nbody)}
        self.geom_ids = {self.model.geom(i).name: i for i in range(self.model.ngeom)}
        self.colliding_geoms = [i for i in range(self.model.ngeom)
                                if self.model.geom_contype[i] or self.model.geom_conaffinity[i]]
        self.rotation = np.asarray(self.metadata["adapter"]["source_to_reference_rotation"])
        self.origin = np.asarray(self.metadata["adapter"]["source_ground_origin"])

    def qpos_at_source(self, frame):
        # Duplicate source frames denote a hold. First occurrence defines the
        # matched source pose; hold-time behavior is separately identifiable.
        unique, indices = np.unique(self.frames, return_index=True)
        frame = float(np.clip(frame, unique[0], unique[-1]))
        q = interpolate(unique, self.arrays["qpos"][indices], frame)
        q[3:7] = Slerp(unique, Rotation.from_quat(self.arrays["qpos"][indices, 3:7], scalar_first=True))([frame]).as_quat(scalar_first=True)[0]
        return q

    def set_frame(self, frame):
        self.data.qpos[:] = self.qpos_at_source(frame)
        self.data.qvel[:] = 0
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_collision(self.model, self.data)

    def qpos_at_time(self, time_s):
        times = self.arrays["time_s"]
        time_s = float(np.clip(time_s, times[0], times[-1]))
        q = interpolate(times, self.arrays["qpos"], time_s)
        q[3:7] = Slerp(times, Rotation.from_quat(self.arrays["qpos"][:, 3:7], scalar_first=True))([time_s]).as_quat(scalar_first=True)[0]
        return q

    def set_time(self, time_s):
        self.data.qpos[:] = self.qpos_at_time(time_s)
        self.data.qvel[:] = 0
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_collision(self.model, self.data)

    def identity(self):
        return {"reference": str(self.path.resolve()), "reference_sha256": sha(self.path),
                "model": str(self.model_path.resolve()), "model_sha256": sha(self.model_path),
                "mapping": str(self.config_path.resolve()), "mapping_sha256": sha(self.config_path),
                "nq_nv_nu": [self.model.nq, self.model.nv, self.model.nu],
                "controller_collision_config": "current Task10C get_h2_robot_cfg with explicit variant XML",
                "source_to_reference_rotation": self.rotation.tolist(), "source_ground_origin": self.origin.tolist()}


class Source:
    def __init__(self, family, dataset_root):
        # Extra recorded context permits candidate guard selection nearby, while
        # retaining the original data and hashes. No poses are synthesized.
        self.clip = load_crop(dataset_root, family, padding_s=0.5)
        self.frames = self.clip.source_frames
        self.rotations = [Slerp(self.frames, Rotation.from_matrix(self.clip.global_rotations[:, i]))
                          for i in range(len(SOURCE_NAMES))]

    def sample(self, frame, reference):
        if not self.frames[0]-1e-8 <= frame <= self.frames[-1]+1e-8:
            raise ValueError("Requested reference frame lies outside loaded original source context")
        frame = float(np.clip(frame, self.frames[0], self.frames[-1]))
        raw = interpolate(self.frames, self.clip.positions, frame)
        positions = (raw - reference.origin) @ reference.rotation.T
        rotations = np.stack([reference.rotation @ r([frame]).as_matrix()[0] for r in self.rotations])
        return positions, rotations


def frame_metrics(reference, source, frame, *, time_s=None):
    if time_s is None:
        reference.set_frame(frame)
    else:
        reference.set_time(time_s)
    model, data = reference.model, reference.data
    positions, rotations = source.sample(frame, reference)
    head = reference.geom_ids["head_collision"]
    head_pos = data.geom_xpos[head]
    torso = reference.body_ids["torso_link"]
    source_head = positions[SOURCE_NAMES.index("Head")]
    reference_time = float(np.interp(frame, reference.frames, reference.arrays["time_s"])) if time_s is None else float(time_s)
    correction = reference.metadata.get("guard_correction")
    unchanged = correction["central_punch_unchanged_interval_s"] if correction else None
    hook_correction = reference.metadata.get("hook_final_correction", {})
    bypass = hook_correction.get("bypass", {})
    row = {"source_frame": float(frame), "reference_time_s": reference_time,
           "intentional_endpoint_correction": bool(unchanged and not unchanged[0] <= reference_time <= unchanged[1]),
           "intentional_hook_recovery_correction": bool(bypass and reference_time > bypass["start_time_s"]),
           "initial_guard_return": bool(reference.metadata.get("hook_guard_return")),
           "source_head_joint_xyz": source_head.tolist(),
           "h2_head_collision_center_xyz": head_pos.tolist(),
           "h2_head_minus_source_head_xyz": (head_pos-source_head).tolist(), "hands": {}}
    for side in ("left", "right"):
        title = side.title()
        hand, elbow = SOURCE_NAMES.index(title+"Hand"), SOURCE_NAMES.index(title+"ForeArm")
        arm = SOURCE_NAMES.index(title+"Arm")
        wrist_id = reference.body_ids[side+"_wrist_yaw_link"]
        elbow_id = reference.body_ids[side+"_elbow_link"]
        shoulder_id = reference.body_ids[side+"_shoulder_yaw_link"]
        glove = reference.geom_ids[side+"_glove_collision"]
        source_forearm = positions[hand]-positions[elbow]
        h2_forearm = data.xpos[wrist_id]-data.xpos[elbow_id]
        actual_rotation = data.xmat[wrist_id].reshape(3, 3)
        source_rotation = rotations[hand]
        offset = Rotation.from_quat(reference.config["ik_match_table1"][side+"_wrist_yaw_link"][4], scalar_first=True).as_matrix()
        target_rotation = source_rotation @ offset
        orientation_error = float(np.degrees(Rotation.from_matrix(target_rotation.T@actual_rotation).magnitude()))
        source_continuation = source_rotation @ np.array([1. if side=="left" else -1., 0, 0])
        glove_vector = data.geom_xpos[glove]-data.xpos[wrist_id]
        distances = {}
        for other in reference.colliding_geoms:
            if other == glove:
                continue
            name = model.geom(other).name
            if name == "head_collision" or name == "torso_collision" or "glove" in name or "elbow" in name or "wrist" in name:
                distances[name] = float(mujoco.mj_geomDistance(model, data, glove, other, 10.0, None))
        row["hands"][side] = {
            "source_wrist_to_source_head_joint_m": float(np.linalg.norm(positions[hand]-source_head)),
            "h2_wrist_to_h2_head_center_m": float(np.linalg.norm(data.xpos[wrist_id]-head_pos)),
            "glove_to_h2_head_center_m": float(np.linalg.norm(data.geom_xpos[glove]-head_pos)),
            "glove_head_surface_clearance_m": distances["head_collision"],
            "glove_radius_m": float(model.geom_size[glove, 0]),
            "head_collision_radius_m": float(model.geom_size[head, 0]),
            "glove_center_xyz": data.geom_xpos[glove].tolist(), "h2_wrist_xyz": data.xpos[wrist_id].tolist(),
            "source_wrist_xyz": positions[hand].tolist(),
            "source_elbow_angle_deg": angle_deg(positions[arm]-positions[elbow], source_forearm),
            "h2_elbow_angle_deg": angle_deg(data.xpos[shoulder_id]-data.xpos[elbow_id], h2_forearm),
            "source_forearm_direction": (source_forearm/np.linalg.norm(source_forearm)).tolist(),
            "h2_forearm_direction": (h2_forearm/np.linalg.norm(h2_forearm)).tolist(),
            "source_vs_h2_forearm_direction_deg": angle_deg(source_forearm,h2_forearm),
            "mapped_source_vs_h2_hand_rotation_deg": orientation_error,
            "source_continuation_vs_h2_glove_direction_deg": angle_deg(source_continuation,glove_vector),
            "source_hand_bend_from_forearm_deg": angle_deg(source_continuation,source_forearm),
            "h2_glove_bend_from_forearm_deg": angle_deg(glove_vector,h2_forearm),
            "geom_surface_distances_m": distances,
            "source_hand_axes_world_columns": source_rotation.tolist(),
            "mapped_source_hand_axes_world_columns": target_rotation.tolist(),
            "h2_hand_axes_world_columns": actual_rotation.tolist(),
        }
    row["penetrating_collision_contacts"] = [
        {"geom1": model.geom(int(data.contact[i].geom1)).name,
         "geom2": model.geom(int(data.contact[i].geom2)).name,
         "distance_m": float(data.contact[i].dist)}
        for i in range(data.ncon) if data.contact[i].dist < -1e-5]
    # Pelvis/torso source bind +y is upright; H2 local +z is upright.
    row["source_torso_up_tilt_deg"] = angle_deg(rotations[SOURCE_NAMES.index("Spine1")][:,1],np.array([0,0,1.]))
    row["h2_torso_up_tilt_deg"] = angle_deg(data.xmat[torso].reshape(3,3)[:,2],np.array([0,0,1.]))
    return row


def diagnose(reference, source):
    frames = reference.frames
    times = reference.arrays["time_s"]
    strike = reference.metadata["strike_time_s"]
    selected = {"guard": float(times[0]), "strike": float(strike),
                "withdrawal": float(strike + .65*(times[-1]-strike)), "terminal": float(times[-1])}
    def at_time(t):
        return frame_metrics(reference, source, float(np.interp(t,times,frames)), time_s=t)
    rows = [at_time(float(t)) for t in times]
    summary = {"identity": reference.identity(), "sampling_axis": "explicit reference seconds; repeated source frames are lineage only",
               "phases": {phase: at_time(t) for phase,t in selected.items()},
               "all_frame_count":len(rows), "minimum_glove_head_clearance_m": {}, "max_penetration_m": 0.0}
    for side in ("left","right"):
        distances = [r["hands"][side]["glove_head_surface_clearance_m"] for r in rows]
        idx = int(np.argmin(distances))
        summary["minimum_glove_head_clearance_m"][side] = {"value":distances[idx],"source_frame":float(frames[idx]),"reference_time_s":float(times[idx])}
    contacts = [dict(c,source_frame=r["source_frame"],reference_time_s=r["reference_time_s"]) for r in rows for c in r["penetrating_collision_contacts"]]
    contacts.sort(key=lambda x:x["distance_m"])
    summary["max_penetration_m"] = -min([0.0]+[c["distance_m"] for c in contacts])
    summary["deepest_static_contacts"] = contacts[:20]
    pairs = {}
    for contact in contacts:
        key = " / ".join(sorted((contact["geom1"], contact["geom2"])))
        if key not in pairs:
            pairs[key] = dict(contact, contact_count=0)
        pairs[key]["contact_count"] += 1
    summary["penetrating_contact_pairs"] = pairs
    summary["frames_with_penetration"] = sum(bool(r["penetrating_collision_contacts"]) for r in rows)
    return summary


AXIS_COLORS = ((.95,.1,.1,1),(.1,.7,.15,1),(.15,.35,1,1))


def add_shape(scene, kind, size, position, rgba, matrix=None):
    geom=scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom,kind,np.asarray(size,dtype=float),np.asarray(position,dtype=float),
                       np.eye(3).ravel() if matrix is None else matrix.ravel(),np.asarray(rgba,dtype=np.float32))
    scene.ngeom += 1
    return geom


def add_segment(scene, start, end, color, radius=.012, kind=mujoco.mjtGeom.mjGEOM_CAPSULE):
    geom=add_shape(scene,kind,[radius,radius,radius],start,color)
    mujoco.mjv_connector(geom,kind,radius,np.asarray(start),np.asarray(end))


def add_axes(scene, position, rotation, scale=.12):
    for axis,color in enumerate(AXIS_COLORS):
        add_segment(scene,position,position+scale*rotation[:,axis],color,.006,mujoco.mjtGeom.mjGEOM_ARROW)


def render_comparison(family,before,after,source,output,fps=25):
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("Render comparisons inside an allocated GPU job; diagnostics need no allocation")
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw
    panel=400
    blank=mujoco.MjModel.from_xml_string('<mujoco><visual><global offwidth="400" offheight="400"/></visual><worldbody><light pos="2 -2 4"/><geom type="plane" size="6 6 .1" rgba=".88 .89 .91 1"/></worldbody></mujoco>')
    blank_data=mujoco.MjData(blank)
    mujoco.mj_forward(blank,blank_data)
    models=[blank,before.model,after.model]
    for model in models:
        model.vis.global_.offwidth=panel; model.vis.global_.offheight=panel
    for ref in (before,after):
        for i in ref.colliding_geoms:
            name=ref.model.geom(i).name
            ref.model.geom_rgba[i,3]=.2 if name=="head_collision" else (1 if "glove" in name else 0)
    renderers=[mujoco.Renderer(m,height=panel,width=panel) for m in models]
    cameras=[]
    for azimuth in (135,90):
        cam=mujoco.MjvCamera(); cam.distance=3.6; cam.azimuth=azimuth; cam.elevation=-10
        cam.lookat[:]=[.15,0,1.0]; cameras.append(cam)
    options=mujoco.MjvOption(); options.geomgroup[2]=1; options.geomgroup[3]=1
    end=float(max(before.arrays["time_s"][-1],after.arrays["time_s"][-1]))
    times=np.arange(0,end+1e-6,1/fps)
    strike=float(after.metadata["strike_time_s"])
    phases=[0.,strike,strike+.65*(float(after.arrays["time_s"][-1])-strike),float(after.arrays["time_s"][-1])]
    output.mkdir(parents=True,exist_ok=True)
    def paint(time_s):
        before.set_time(time_s); after.set_time(time_s)
        frame=float(np.interp(time_s,after.arrays["time_s"],after.frames))
        positions,rotations=source.sample(frame,after)
        canvas=Image.new("RGB",(3*panel,2*panel+90),"white")
        draw=ImageDraw.Draw(canvas)
        draw.text((12,7),f"{family} | t={time_s:.2f}s | source lineage {frame:.2f} | source / before H2 / after H2 (end poses held)",fill="black")
        draw.text((12,25),"RGB = local X/Y/Z hand+forearm axes; yellow = original human Head joint (not a head sphere estimate)",fill="black")
        draw.text((12,43),"Source: raw unscaled sticks, wrist endpoints, no gloves. H2: actual 0.08m collision gloves; translucent H2 head collision sphere.",fill="black")
        correction=after.metadata.get("guard_correction")
        note="After H2 includes intentional guard corrections near endpoints; source frame indicates provenance." if correction else "H2 is retargeted; original skeleton and H2 body proportions differ. Top: oblique. Bottom: side."
        if after.metadata.get("hook_final_correction", {}).get("bypass"):
            note="After H2: corrected withdrawal hand path + nearby recovery guard; source frame is lineage, not an exact pose target."
        if after.metadata.get("hook_guard_return"):
            note="After H2 returns to its own initial upper-body guard on the recovered stance; appended source lineage is held."
        if after.metadata.get("right_hook_compact_recovery"):
            note="After H2: replaced right-arm recovery segment; original attack, torso/stance and final guard retained. Playback: 1x."
        draw.text((12,61),note,fill="black")
        for row,camera in enumerate(cameras):
            for col,(renderer,ref) in enumerate(zip(renderers,(None,before,after))):
                renderer.update_scene(blank_data if ref is None else ref.data,camera,scene_option=options)
                if ref is None:
                    for joint,parent in enumerate(PARENTS):
                        if parent>=0: add_segment(renderer.scene,positions[parent],positions[joint],(.25,.25,.3,1))
                    for name in ("LeftForeArm","RightForeArm","LeftHand","RightHand"):
                        i=SOURCE_NAMES.index(name); add_axes(renderer.scene,positions[i],rotations[i])
                    head=positions[SOURCE_NAMES.index("Head")]
                else:
                    add_shape(renderer.scene,mujoco.mjtGeom.mjGEOM_PLANE,[6,6,.1],[0,0,0],(.88,.89,.91,1))
                    ref_frame=float(np.interp(time_s,ref.arrays["time_s"],ref.frames))
                    p,_=source.sample(ref_frame,ref); head=p[SOURCE_NAMES.index("Head")]
                    for name in ("left_elbow_link","right_elbow_link","left_wrist_yaw_link","right_wrist_yaw_link"):
                        i=ref.body_ids[name]; add_axes(renderer.scene,ref.data.xpos[i],ref.data.xmat[i].reshape(3,3))
                add_shape(renderer.scene,mujoco.mjtGeom.mjGEOM_SPHERE,[.035]*3,head,(1,.75,0,.9))
                canvas.paste(Image.fromarray(renderer.render().copy()),(col*panel,row*panel+90))
        return canvas
    with imageio.get_writer(output/f"{family}_source_before_after.mp4",fps=fps,codec="libx264",quality=7,macro_block_size=2) as writer:
        for time_s in times:
            image=paint(float(time_s)); writer.append_data(np.asarray(image))
    for time_s,name in zip(phases,("guard","strike","withdrawal","terminal")):
        paint(time_s).save(output/f"{family}_{name}.png")
    # Show the selected initial and terminal poses in matched torso-heading views.
    # Only the camera follows the torso: root, feet and world heading are not reset.
    if after.metadata.get("hook_guard_return"):
        canvas=Image.new("RGB",(2*panel,2*panel+70),"white")
        draw=ImageDraw.Draw(canvas)
        draw.text((12,7),f"{family} | selected INITIAL guard / selected TERMINAL guard",fill="black")
        draw.text((12,25),"Cameras follow torso yaw; actual recovered support stance retained. Top: oblique. Bottom: side.",fill="black")
        draw.text((12,43),"Red = left glove, blue = right glove. Kinematic reference poses, not a PD balance test.",fill="black")
        for col,time_s in enumerate((0.,float(after.arrays["time_s"][-1]))):
            after.set_time(time_s)
            torso=after.body_ids["torso_link"]
            rotation=after.data.xmat[torso].reshape(3,3)
            yaw=float(np.degrees(np.arctan2(rotation[1,0],rotation[0,0])))
            for row,azimuth in enumerate((135,90)):
                camera=mujoco.MjvCamera(); camera.distance=3.4
                camera.azimuth=azimuth+yaw; camera.elevation=-10
                camera.lookat[:]=[after.data.xpos[torso,0],after.data.xpos[torso,1],1.0]
                renderer=renderers[2]
                renderer.update_scene(after.data,camera,scene_option=options)
                add_shape(renderer.scene,mujoco.mjtGeom.mjGEOM_PLANE,[6,6,.1],[0,0,0],(.88,.89,.91,1))
                canvas.paste(Image.fromarray(renderer.render().copy()),(col*panel,row*panel+70))
        canvas.save(output/f"{family}_initial_vs_terminal.png")
    for renderer in renderers: renderer.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families",nargs="+",default=["jab","left_hook","right_hook"])
    parser.add_argument("--dataset-root",type=Path,default=ROOT.parent/"DuoBox")
    parser.add_argument("--before-dir",type=Path,default=ROOT/"outputs/task10c/reference_correction/before")
    parser.add_argument("--after-dir",type=Path,default=ROOT/"outputs/task10c/references")
    parser.add_argument("--before-model",type=Path)
    parser.add_argument("--after-model",type=Path,default=H2_XML)
    parser.add_argument("--before-config",type=Path)
    parser.add_argument("--after-config",type=Path,default=ROOT/"configs/task10c/duobox_to_h2_31.json")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--render",action="store_true")
    args=parser.parse_args(); output=args.output.resolve()
    if not output.is_relative_to(ROOT): raise ValueError("Output escapes human-punching")
    for family in args.families:
        before=Reference(args.before_dir/f"{family}.npz",args.before_model or args.before_dir/"h2.xml",args.before_config or args.before_dir/"duobox_to_h2_31.json")
        after=Reference(args.after_dir/f"{family}.npz",args.after_model,args.after_config)
        source=Source(family,args.dataset_root)
        report={"family":family,"source_sha256":source.clip.metadata["source_sha256"],
                "scope":"Robot FK and static self-collision geometry only; standalone model omits training ground. Ground shown in renders is a visual overlay. No integration or solver/contact-force validation.",
                "source_definition":"Original recorded skeleton, scene yaw/ground XY removal only; Head is a skeleton joint, Hand is wrist origin; no invented physical source glove/head radius.",
                "distance_definition":"mj_geomDistance signed separation of compiled training collision shapes; negative means geometric penetration. Contact records from mj_collision include actual masks/exclusions.",
                "before":diagnose(before,source),"after":diagnose(after,source)}
        write_json(output/f"{family}_comparison.json",report)
        print(json.dumps({"family":family,"before_min_glove_head":report["before"]["minimum_glove_head_clearance_m"],
                          "after_min_glove_head":report["after"]["minimum_glove_head_clearance_m"]}),flush=True)
        if args.render: render_comparison(family,before,after,source,output)


if __name__=="__main__": main()
