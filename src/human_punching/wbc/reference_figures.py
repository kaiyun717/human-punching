"""Portable scientific figures from actual reference arrays, with no rendering.

These are kinematic diagnostics and skeleton plots, not physical rollout videos.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import mujoco
import numpy as np

from .contracts import LaunchSnapshot
from .provenance import PROJECT_ROOT, writable_path
from .references import sha256
from .runtime.bundle import load_runtime_bundle


def plot_reference_bundle(bundle_path: Path, output_root: Path, *,
                          phase_sidecar: Path | None = None,
                          effort_directory: Path | None = None) -> dict:
    output=writable_path(output_root)
    if output.exists():
        raise FileExistsError('Choose a new figure directory')
    os.environ.setdefault('MPLCONFIGDIR',str(PROJECT_ROOT/'.cache/matplotlib-task10'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    bundle=load_runtime_bundle(bundle_path)
    phases={}
    if phase_sidecar is not None:
        payload=json.loads(Path(phase_sidecar).read_text())
        if payload['bundle_manifest_sha256']!=sha256(bundle.root/'manifest.json'):
            raise ValueError('Phase sidecar belongs to another bundle')
        phases={row['reference_id']:row for row in payload['references']}
    effort={}
    if effort_directory is not None:
        report=json.loads((Path(effort_directory)/'report.json').read_text())
        if report['bundle_manifest_sha256']!=sha256(bundle.root/'manifest.json'):
            raise ValueError('Effort diagnostics belong to another bundle')
        effort={row['reference_id']:row for row in report['families']}
    output.mkdir(parents=True)
    model=mujoco.MjModel.from_xml_path(str(bundle.model_path))
    data=mujoco.MjData(model)
    parents=[int(model.body_parentid[i])-1 for i in range(1,model.nbody)]
    files=[]
    for clip in bundle.clips:
        with np.load(bundle.path(clip['master_path']),allow_pickle=False) as archive:
            arrays={key:archive[key].copy() for key in ('time_s','qpos','qvel','glove_pos_w','body_pos_w')}
        times=arrays['time_s']
        phase=phases.get(clip['reference_id'],clip)
        if phase.get('master_sha256')!=clip['master_sha256']:
            raise ValueError('Phase source master hash differs')
        outward_end=phase.get('outward_end_s')
        if outward_end is None:
            raise ValueError('An exact outward phase is required for labeled plots')
        snapshot=LaunchSnapshot(**clip['launch_snapshot'])
        def local(points):
            flat=np.asarray(points).reshape(-1,3)
            return np.asarray([snapshot.world_to_human_point(point) for point in flat]).reshape(np.asarray(points).shape)
        body,gloves=local(arrays['body_pos_w']),local(arrays['glove_pos_w'])
        target=np.array(snapshot.world_to_human_point(snapshot.head_position_w))
        figure=plt.figure(figsize=(13,8),layout='constrained')
        grid=figure.add_gridspec(3,2,width_ratios=(1.,1.2))
        ax=figure.add_subplot(grid[:,0],projection='3d')
        frame_times=[0.,clip['requested_ttc_s'],outward_end,clip['guard_time_s']]
        labels=['launch guard','requested contact','outward end','recovered reference']
        colors=['#606060','#0077bb','#ee7733','#009988']
        for time,label,color in zip(frame_times,labels,colors,strict=True):
            index=int(np.argmin(abs(times-time)))
            points=body[index]
            for child,parent in enumerate(parents):
                if parent>=0:
                    segment=points[[parent,child]]
                    ax.plot(segment[:,0],segment[:,1],segment[:,2],color=color,alpha=.65,lw=1.5)
            ax.scatter(*points[0],color=color,label=f'{label}: {times[index]:.3f} s',s=15)
        ax.plot(gloves[:,0,0],gloves[:,0,1],gloves[:,0,2],color='#0077bb',lw=1.2,label='left glove path')
        ax.plot(gloves[:,1,0],gloves[:,1,1],gloves[:,1,2],color='#cc3311',lw=1.2,ls='--',label='right glove path')
        ax.scatter(*target,color='black',marker='x',s=70,label='frozen head center')
        data.qpos[:]=arrays['qpos'][0]
        mujoco.mj_forward(model,data)
        for side in ('Left','Right'):
            geom=model.geom('physical/'+side+'Foot').id
            size=model.geom_size[geom]
            corners=np.array([[sx*size[0],sy*size[1],-size[2]] for sx,sy in ((-1,-1),(1,-1),(1,1),(-1,1),(-1,-1))])
            corners=local(corners@data.geom_xmat[geom].reshape(3,3).T+data.geom_xpos[geom])
            ax.plot(corners[:,0],corners[:,1],corners[:,2],color='black',lw=2)
        ax.set(xlabel='forward (m)',ylabel='left (m)',zlabel='height (m)',
               title='Actual reference poses and glove paths')
        ax.set_box_aspect((1.2,1.,1.8));ax.view_init(elev=15,azim=-60)
        ax.legend(fontsize=7,loc='upper left')
        axes=[figure.add_subplot(grid[i,1]) for i in range(3)]
        distances=np.linalg.norm(arrays['glove_pos_w']-snapshot.head_position_w,axis=-1)
        axes[0].plot(times,distances[:,0],label='left glove',color='#0077bb')
        axes[0].plot(times,distances[:,1],label='right glove',color='#cc3311')
        axes[0].axhline(.14,color='black',ls='--',label='analytic contact: 0.14 m')
        axes[0].axhline(.16,color='grey',ls=':',label='clearance: 0.16 m')
        axes[0].set(ylabel='glove / target distance (m)');axes[0].legend(fontsize=7)
        axes[1].plot(times,np.max(abs(arrays['qvel'][:,6:]),axis=1),color='#663399')
        axes[1].set(ylabel='max joint speed (rad/s)')
        if clip['reference_id'] in effort:
            row=effort[clip['reference_id']]
            artifact=Path(effort_directory)/row['artifact']
            if sha256(artifact)!=row['artifact_sha256']:
                raise ValueError('Effort diagnostic artifact hash mismatch')
            with np.load(artifact,allow_pickle=False) as source:
                ratio=source['minimum_peak_effort_ratio'].copy()
                feasible=source['support_feasible'].copy()
            axes[2].plot(times,ratio,color='#228833',label='minimum peak torque / cap')
            axes[2].axhline(1.,color='black',ls='--')
            if np.any(~feasible):
                axes[2].scatter(times[~feasible],np.zeros((~feasible).sum()),color='red',s=8,label='support infeasible')
            axes[2].legend(fontsize=7)
        else:
            acceleration=np.gradient(arrays['qvel'],times,axis=0,edge_order=2)
            axes[2].plot(times,np.max(abs(acceleration[:,6:]),axis=1),color='#228833')
        axes[2].set(xlabel='time since launch (s)',ylabel='effort ratio' if effort else 'max acceleration (rad/s²)')
        for graph in axes:
            graph.axvline(clip['requested_ttc_s'],color='#0077bb',alpha=.6)
            graph.axvline(outward_end,color='#ee7733',alpha=.6)
            graph.axvline(clip['guard_time_s'],color='#009988',alpha=.6)
            graph.grid(alpha=.2);graph.set_xlim(0,times[-1])
        figure.suptitle(f"{clip['family'].replace('_',' ').title()} — offline reference, not a physical rollout",fontsize=15)
        for extension in ('png','pdf'):
            path=output/f"{clip['family']}.{extension}"
            figure.savefig(path,dpi=170)
            files.append({'path':path.name,'sha256':sha256(path),'family':clip['family']})
        plt.close(figure)
    report={'schema':'human_wbc_reference_figures_v1','physics_executed':False,
            'bundle_manifest_sha256':sha256(bundle.root/'manifest.json'),
            'source_sha256':sha256(Path(__file__)),'figures':files,
            'description':'Actual reference arrays, FK skeleton poses and geometric target metrics. No simulated execution or motion-quality acceptance is claimed.'}
    (output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
