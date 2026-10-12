#!/usr/bin/env python3
"""Two-world Warp standing/jab replay with the existing CPU feedback controller.

Physics advances only in Warp. CPU mj_forward materializes mass/Jacobians from
observed GPU states for bounded_pd; it never advances or resets those states.
Host transfers deliberately favor a transparent parity check over throughput.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
import warp as wp
from mjlab.sim import MujocoCfg, Simulation, SimulationCfg

from human_punching.contact import evaluate_clearance, first_contact_time
from human_punching.physical import bounded_pd, controller_arrays, interpolate_pose
from human_punching.physics_cli import rollout as native_rollout


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--speed', type=float, default=.6)
    parser.add_argument('--gain-scale', type=float, default=1.)
    parser.add_argument('--punch', choices=('jab','cross','left_hook','right_hook'), default='jab')
    parser.add_argument('--head-center', type=float, nargs=3,
                        default=[.8539635, .2, 1.2314907452582464])
    args = parser.parse_args()
    if args.speed<=0 or args.gain_scale<=0:
        raise ValueError('Playback speed and gain scale must be positive.')
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Requires an allocated Slurm CUDA GPU.')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Freeze these generated model assets and reference for the experiment.
    shutil.copytree(args.model.parent, output/'model', dirs_exist_ok=True)
    shutil.copy2(args.reference, output/'reference.npz')
    controller_source=Path(__file__).resolve().parents[1]/'src/human_punching/physical.py'
    shutil.copy2(controller_source,output/'controller_source.py')
    project=Path(__file__).resolve().parents[1]
    for name in ('physics_cli.py','contact.py'):
        shutil.copy2(project/'src/human_punching'/name,output/name)
    shutil.copy2(Path(__file__),output/'warp_tracking_probe.py')
    shutil.copy2(project/'scripts/warp_tracking_volta.sbatch',output/'warp_tracking_volta.sbatch')
    arguments={key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()}
    (output/'arguments.json').write_text(json.dumps(arguments,indent=2)+'\n')
    model_path = output/'model'/args.model.name
    model = mujoco.MjModel.from_xml_path(str(model_path))
    source = np.load(output/'reference.npz')
    times, reference = source['time_s'], source['qpos']
    dt = float(model.opt.timestep)
    assert (model.nq, model.nv, model.nu) == (87,66,60)
    assert dt == .002
    head = np.asarray(args.head_center)
    durations = [5., float(times[-1]/args.speed + 2.)]
    steps = [round(duration/dt) for duration in durations]
    sample_interval = max(1,round(1/(120*dt)))
    cfg = SimulationCfg(nconmax=128, njmax=512,
                        mujoco=MujocoCfg(timestep=dt,cone='elliptic',iterations=60,
                                        tolerance=1e-9,ls_iterations=50))
    wp.init()
    wp.config.kernel_cache_dir = str(Path('.cache/warp').resolve())
    sim = Simulation(2,cfg,model,'cuda:0')
    helpers = [mujoco.MjData(model),mujoco.MjData(model)]
    feet_ids = [model.site(f'{side}Foot_sole').id for side in ('Left','Right')]
    hands_ids = [model.site(f'{side}Hand_center').id for side in ('Left','Right')]
    foot_geoms = [model.geom(f'physical/{side}Foot').id for side in ('Left','Right')]
    hips, floor = model.body('Hips').id,model.geom('floor').id
    limits = controller_arrays(model)[2]
    error = np.zeros(model.nv)
    buckets = [{key:[] for key in ('qpos','qvel','time_s','desired_qpos','foot_centers_w',
                                 'glove_centers_w','center_of_mass_w','foot_contact_counts',
                                 'root_up','floor_depth','root_actuator_force','joint_error',
                                 'saturated_fraction','torque_fraction','substep_time_s','substep_glove_centers_w')} for _ in range(2)]
    with wp.ScopedDevice(sim.wp_device):
        stream = torch.cuda.ExternalStream(wp.get_stream().cuda_stream)
    stream.wait_stream(torch.cuda.current_stream())
    # These are the only qpos/qvel writes to GPU state in the entire replay.
    with torch.cuda.stream(stream):
        sim.data.qpos[:] = torch.as_tensor(reference[0],device='cuda',dtype=torch.float32)
        sim.data.qvel.zero_()
        sim.data.ctrl.zero_()
        sim.forward()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    started = time.perf_counter()
    for step in range(max(steps)+1):
        now = step*dt
        qpos = sim.data.qpos.cpu().numpy().copy()
        qvel = sim.data.qvel.cpu().numpy().copy()
        if not np.isfinite(qpos).all() or not np.isfinite(qvel).all() or np.max(abs(qvel))>1000:
            raise RuntimeError(f'Warp dynamics diverged at {now:.4f}s')
        sample = step%sample_interval == 0
        if sample:
            # Contact data really comes from Warp, not the CPU helper collision pass.
            count = int(sim.wp_data.nacon.numpy()[0])
            if count>sim.wp_data.naconmax:
                raise RuntimeError('Warp contact buffer overflow')
            wc = sim.wp_data.contact
            contact_world = wc.worldid.numpy()[:count]
            contact_geom = wc.geom.numpy()[:count]
            contact_dist = wc.dist.numpy()[:count]
            root_force = sim.data.qfrc_actuator.cpu().numpy()[:,:6]
        controls = []
        for world,data in enumerate(helpers):
            data.qpos[:], data.qvel[:] = qpos[world],qvel[world]
            data.time = now
            mujoco.mj_forward(model,data)
            desired,speed = ((reference[0],np.zeros(model.nv)) if world==0 else
                             interpolate_pose(model,times,reference,now*args.speed))
            speed = speed*args.speed
            saturated,_ = bounded_pd(model,data,desired,speed,balance=True,gain_scale=args.gain_scale)
            controls.append(data.ctrl.copy())
            if step<=steps[world]:
                b=buckets[world]
                b['substep_time_s'].append(now)
                b['substep_glove_centers_w'].append(data.site_xpos[hands_ids].copy())
                mujoco.mj_differentiatePos(model,error,1.,data.qpos,desired)
                b['joint_error'].append(float(np.sqrt(np.mean(error[6:]**2))))
                b['saturated_fraction'].append(saturated)
                b['torque_fraction'].append(float(np.max(np.linalg.norm(data.ctrl.reshape(-1,3),axis=1)/limits)))
                if sample:
                    world_contacts=contact_geom[contact_world==world]
                    counts=[int(np.sum(np.any(world_contacts==floor,axis=1)&np.any(world_contacts==g,axis=1))) for g in foot_geoms]
                    for key,value in (
                        ('qpos',qpos[world]),('qvel',qvel[world]),('time_s',now),
                        ('desired_qpos',desired),('foot_centers_w',data.site_xpos[feet_ids].copy()),
                        ('glove_centers_w',data.site_xpos[hands_ids].copy()),
                        ('center_of_mass_w',data.subtree_com[hips].copy()),
                        ('foot_contact_counts',counts),('root_up',float(data.xmat[hips].reshape(3,3)[2,1])),
                        ('floor_depth',float(np.min(contact_dist[contact_world==world],initial=0.))),
                        ('root_actuator_force',float(np.max(abs(root_force[world]))))):
                        b[key].append(value)
        if step<max(steps):
            with torch.cuda.stream(stream):
                sim.data.ctrl[:] = torch.as_tensor(np.array(controls),device='cuda',dtype=torch.float32)
                sim.step()
            torch.cuda.current_stream().wait_stream(stream)
            # State transfer on the next iteration synchronizes the produced state.
        if step%500==0:
            print(f'Warp controlled replay t={now:.3f}s/{max(durations):.3f}s',flush=True)
    elapsed=time.perf_counter()-started
    results={}
    for world,title in enumerate(('standing',args.punch)):
        arrays={key:np.asarray(values) for key,values in buckets[world].items()}
        poses,feet,contacts=arrays['qpos'],arrays['foot_centers_w'],arrays['foot_contact_counts']
        upright=bool(poses[:,2].min()>.7*poses[0,2] and arrays['root_up'].min()>.7)
        labels=evaluate_clearance(arrays['substep_glove_centers_w'],head)
        hit=first_contact_time(arrays['substep_time_s'],arrays['substep_glove_centers_w'],head)
        metrics={
            'duration_s':durations[world],'standing_only':world==0,'physical_stepping':True,
            'backend':'mjlab Simulation / MuJoCo-Warp CUDA',
            'controller':'CPU bounded_pd(balance=True), feedback from live Warp states',
            'external_root_assistance':False,'gpu_state_overwrites_after_reset':0,
            'initial_pelvis_height_m':float(poses[0,2]),'min_pelvis_height_m':float(poses[:,2].min()),
            'final_pelvis_height_m':float(poses[-1,2]),'root_up_min':float(arrays['root_up'].min()),
            'remained_upright':upright,'root_actuator_force_max':float(arrays['root_actuator_force'].max()),
            'joint_error_rms_rad':float(np.sqrt(np.mean(arrays['joint_error']**2))),
            'max_torque_norm_fraction':float(arrays['torque_fraction'].max()),
            'mean_joint_saturation_fraction':float(arrays['saturated_fraction'].mean()),
            'floor_penetration_max_m':float(-arrays['floor_depth'].min()),
            'each_foot_contact_fraction':np.mean(contacts>0,axis=0).tolist(),
            'both_feet_contact_fraction':float(np.mean(np.all(contacts>0,axis=1))),
            'max_sole_horizontal_drift_m':float(np.linalg.norm(feet[:,:,:2]-feet[0,:,:2],axis=-1).max()),
            'max_sole_height_m':float(feet[:,:,2].max()),'timestep_s':dt,'playback_speed':args.speed,
            'minimum_glove_head_distance_m':labels.distances_m.min(axis=0).tolist(),
            'any_proxy_contact':bool(labels.any_contact.any()),'first_contact_time_s':float(hit) if np.isfinite(hit) else None,
            'contact_counts_source':'actual Warp contact arrays; CPU FK only for output body/site coordinates',
            'head_center_w':head.tolist(),'impact_response_g1':False,
            'event_sampling_s':dt,'gain_scale':args.gain_scale,'punch_label':args.punch,
            'fixed_g1_in_scene':any((model.body(i).name or '').startswith('g1/') for i in range(model.nbody))}
        metrics['support_gate']=bool(metrics['remained_upright'] and metrics['max_sole_horizontal_drift_m']<=.03
            and metrics['max_sole_height_m']<=.03 and metrics['floor_penetration_max_m']<=.01
            and metrics['both_feet_contact_fraction']>=.95)
        metrics['isolated_punch_gate']=bool(metrics['support_gate'] and metrics['any_proxy_contact']
            and metrics['mean_joint_saturation_fraction']<=.05)
        np.savez_compressed(output/f'{title}_warp_rollout.npz',**arrays)
        native,native_metrics=native_rollout(model,times,reference,standing=world==0,speed=args.speed,balance=True,gain_scale=args.gain_scale)
        np.savez_compressed(output/f'{title}_native_rollout.npz',**native)
        # Same sampled times and root/joint state representation; no tolerances silently widened.
        assert native['qpos'].shape==arrays['qpos'].shape
        max_joint,max_root=0.,0.
        for cpu,gpu in zip(native['qpos'],arrays['qpos'],strict=True):
            mujoco.mj_differentiatePos(model,error,1.,cpu,gpu)
            max_joint=max(max_joint,float(np.max(abs(error[6:]))))
            max_root=max(max_root,float(np.linalg.norm(cpu[:3]-gpu[:3])))
        native_labels=evaluate_clearance(native['substep_glove_centers_w'],head)
        native_metrics['minimum_glove_head_distance_m']=native_labels.distances_m.min(axis=0).tolist()
        native_metrics['any_proxy_contact']=bool(native_labels.any_contact.any())
        native_metrics['event_sampling_s']=dt
        native_hit=first_contact_time(native['substep_time_s'],native['substep_glove_centers_w'],head)
        native_metrics['first_contact_time_s']=float(native_hit) if np.isfinite(native_hit) else None
        native_metrics['support_gate']=bool(native_metrics['remained_upright'] and native_metrics['max_sole_horizontal_drift_m']<=.03
            and native_metrics['max_sole_height_m']<=.03 and native_metrics['floor_penetration_max_m']<=.01
            and native_metrics['both_feet_contact_fraction']>=.95)
        native_metrics['isolated_punch_gate']=bool(native_metrics['support_gate'] and native_metrics['any_proxy_contact']
            and native_metrics['mean_joint_saturation_fraction']<=.05)
        result={'warp':metrics,'native':native_metrics,'comparison':{
            'same_reference_same_controller':True,'same_upright_outcome':upright==native_metrics['remained_upright'],
            'maximum_root_position_difference_m':max_root,'maximum_joint_geodesic_component_difference_rad':max_joint,
            'note':'Warp float32 dynamics vs native float64. Both backends recompute FK at the current state and record glove contact at every 2 ms physics step.'}}
        (output/f'{title}_metrics.json').write_text(json.dumps(result,indent=2)+'\n')
        results[title]=result
        print(title,json.dumps(result),flush=True)
    report={'job_id':os.environ['SLURM_JOB_ID'],'status':'passed' if all(r['warp']['remained_upright'] and r['warp']['root_actuator_force_max']==0 and r['warp']['max_torque_norm_fraction']<=1.000001 for r in results.values()) else 'control_gate_failed',
            'status_scope':'upright/no-root-assistance/torque-bound control gate; contact reported separately',
            'contact_goal_achieved':results[args.punch]['warp']['any_proxy_contact'],
            'isolated_punch_gate':results[args.punch]['warp']['isolated_punch_gate'],
            'gpu':torch.cuda.get_device_name(0),'worlds':2,'warp_replay_elapsed_s':elapsed,
            'scope':'Physical control parity, deliberately CPU controller + synchronous GPU transfers; not a throughput benchmark or learned tracker.',
            'model_sha256':sha256(model_path),'reference_sha256':sha256(output/'reference.npz'),
            'controller_source_sha256':sha256(output/'controller_source.py'),
            'native_cli_source_sha256':sha256(output/'physics_cli.py'),
            'event_source_sha256':sha256(output/'contact.py'),
            'probe_source_sha256':sha256(output/'warp_tracking_probe.py'),
            'arguments':arguments,
            'runs':results}
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return 0 if report['status']=='passed' else 1


if __name__=='__main__':
    raise SystemExit(main())
