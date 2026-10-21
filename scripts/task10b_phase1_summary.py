#!/usr/bin/env python3
"""Summarize every Phase 1 attempt without changing its classification."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import shutil
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from human_punching.wbc.provenance import sha256_file, writable_path, write_json


def summarize(packet_path, output):
    packet_path = Path(packet_path).resolve(strict=True)
    packet = json.loads(packet_path.read_text())
    run = ROOT / packet['output']
    audit = json.loads((run/'audit.json').read_text())
    destination = writable_path(output)
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, destination/'summary_source.py')
    rows, families = [], []
    for condition in audit['conditions']:
        folder = run/condition['name']
        functional = json.loads((folder/'functional_evaluation/summary.json').read_text())
        historical = json.loads((folder/'historical_evaluation/summary.json').read_text())
        old = {r['trial_id']: r for r in historical['trials']}
        for trial in functional['trials']:
            h = old[trial['trial_id']]
            final = trial['diagnostic_final_values']
            rows.append(dict(family=condition['family'], trial_id=trial['trial_id'],
                functional_complete=trial['functional_complete'],
                nominal_complete_cycle=trial.get('nominal_complete_cycle'),
                historical_exact_pose_cycle=h['strict_historical_release_pass'],
                contact_outcome=trial['contact_outcome'],
                commanded_hit_success=trial.get('commanded_hit_success'),
                launch_rejected=trial['launch_rejected'],
                readiness_s=trial['readiness_time_s'], launch_s=trial['launch_time_s'],
                outward_complete_s=trial['outward_completion_time_s'], recovery_s=trial['recovery_time_s'],
                first_active_contact_s=trial['first_active_hand_outward_contact_s'],
                stability_observed_s=trial['diagnostic_stability_observed_s'],
                stability_pass=trial['diagnostic_stability_pass'],
                final_hand_error_max_m=max(final['hand_position_error_m']),
                final_root_linear_m_s=final['root_linear_speed_m_s'],
                final_root_angular_rad_s=final['root_angular_speed_rad_s'],
                final_hand_speed_max_m_s=max(final['hand_relative_speed_m_s']),
                final_hand_orientation_max_rad=max(final['hand_orientation_error_rad']),
                final_torso_orientation_rad=final['torso_orientation_error_rad'],
                final_head_orientation_rad=final['head_orientation_error_rad'],
                final_com_margin_m=final['com_support_margin_m'],
                final_root_height_fraction=final['root_height_fraction'],
                final_joint_return_max_rad=max(final['joint_return_error_rad']),
                final_joint_speed_max_rad_s=max(final['joint_speed_rad_s']),
                final_glove_clearance_min_m=min(final['frozen_target_glove_distance_m']),
                tracking_full_rms_m=h['reference_tracking']['active_glove_position_rms_m'],
                tracking_outward_rms_m=h['reference_tracking']['attack_active_glove_position_rms_m'],
                historical_timing_error_s=h['contact_timing']['absolute_interpolated_contact_error_s']))
        selected = [r for r in rows if r['family']==condition['family']]
        recoveries = [r['recovery_s'] for r in selected if r['recovery_s'] is not None]
        families.append(dict(family=condition['family'], attempts=len(selected),
            functional_complete=sum(r['functional_complete'] for r in selected),
            nominal_complete_cycles=sum(bool(r['nominal_complete_cycle']) for r in selected),
            historical_exact_pose_cycles=sum(r['historical_exact_pose_cycle'] for r in selected),
            launches=sum(not r['launch_rejected'] for r in selected),
            hits=sum(r['contact_outcome']=='hit' for r in selected),
            stable_additional_half_second=sum(bool(r['stability_pass']) for r in selected),
            recovery_range_s=[min(recoveries), max(recoveries)] if recoveries else None,
            diagnostic_final_pass_counts={k:sum(t['diagnostic_final_checks'][k] for t in functional['trials'])
                for k in functional['trials'][0]['diagnostic_final_checks']}))
    with (destination/'all_trials.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,1,figsize=(10,11),sharex=True,constrained_layout=True)
    for ax,condition in zip(axes,audit['conditions'],strict=True):
        folder=run/condition['name']/'functional_evaluation'
        with np.load(folder/'functional_00.npz') as a:
            t=a['time_s']; ax.plot(t,a['hand_position_error_m'].max(-1)/.05,label='Hand error / 0.05 m')
            ax.plot(t,a['root_linear_speed_m_s']/.05,label='Root linear / 0.05 m/s')
            ax.plot(t,a['root_angular_speed_rad_s']/.12,label='Root angular / 0.12 rad/s')
            ax.plot(t,a['hand_relative_speed_m_s'].max(-1)/.15,label='Hand speed / 0.15 m/s')
        r=next(r for r in rows if r['family']==condition['family'] and r['trial_id']==0)
        for key,color in [('launch_s','gray'),('outward_complete_s','orange'),('recovery_s','green')]:
            if r[key] is not None: ax.axvline(r[key],color=color,ls='--',alpha=.7,label=key)
        ax.axhline(1,color='black',lw=.8); ax.set_ylim(0,4); ax.set_title(condition['family'])
        ax.set_ylabel('Threshold ratio (clipped at 4)')
    axes[0].legend(ncol=3,fontsize=7);axes[-1].set_xlabel('Original reset-relative time (s)')
    fig.savefig(destination/'nominal_guard_diagnostics.png',dpi=140)
    fig.savefig(destination/'nominal_guard_diagnostics.svg');plt.close(fig)
    report=dict(schema='task10b_phase1_summary_v1',packet_sha256=sha256_file(packet_path),
        audit_sha256=sha256_file(run/'audit.json'),families=families,attempts=len(rows),
        planned_attempts=packet['requested_full_cycle_attempts'],
        unattempted_conditions=audit['unattempted_conditions'],
        actual_world_physics_transitions=audit['actual_world_physics_transitions'],
        learning_updates=0,weights_changed=False,
        limitations='Original fixed references and banks only. No moving-G1 dodge, expanded target/TTC/start or scheduler qualification.',
        artifacts={p.name:dict(sha256=sha256_file(p),bytes=p.stat().st_size)
                   for p in destination.iterdir() if p.is_file()})
    write_json(destination/'summary.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--packet',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();summarize(a.packet,a.output)
