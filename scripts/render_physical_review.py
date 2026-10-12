"""Collect final accepted native physics cases and render measured state videos."""
import json
import shutil
from pathlib import Path

import mujoco
import numpy as np

from human_punching.physical_audit import audit_reference
from human_punching.physics_cli import render
from human_punching.render import make_overview

project = Path(__file__).resolve().parents[1]
output = project / 'outputs/physical-review'
output.mkdir(parents=True, exist_ok=True)
selected = {'jab': '5890952', 'cross': '5890924', 'left_hook': '5890924', 'right_hook': '5891006'}
report = {'description': 'Actual native MuJoCo stepped states; G1 fixed, analytic head events; source speed explicitly shown.', 'cases': {}}
stills = []
for punch, job in selected.items():
    source = project / f'outputs/physical-{job}'
    model = mujoco.MjModel.from_xml_path(str(source / 'scene/physical_scene.xml'))
    trajectory = dict(np.load(source / punch / f'{punch}_rollout.npz'))
    metrics = json.loads((source / punch / f'{punch}_metrics.json').read_text())
    assert metrics['isolated_punch_gate']
    destination = output / punch
    destination.mkdir(exist_ok=True)
    reference = dict(np.load(source / punch / 'reference.npz'))
    audit = audit_reference(model, reference['qpos'], reference['time_s'] / metrics['playback_speed'], metrics['head_center_w'])
    executed_audit = audit_reference(model, trajectory['qpos'], trajectory['time_s'], metrics['head_center_w'])
    assert audit['cone_violation_max_rad'] < 1e-8
    assert executed_audit['cone_violation_max_rad'] < 1e-3
    (destination / 'reference-audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    (destination / 'executed-fk-audit.json').write_text(json.dumps(executed_audit, indent=2)+'\n')
    shutil.copy2(source / punch / f'{punch}_metrics.json', destination / 'metrics.json')
    render(model, trajectory, metrics, destination, punch)
    report['cases'][punch] = {'job': job, 'source_directory': str(source), 'metrics': metrics,
                            'video': str(destination / f'{punch}.mp4'), 'reference_audit': audit,
                            'executed_cone_violation_max_rad': executed_audit['cone_violation_max_rad']}
    stills.append(destination / f'{punch}.png')
make_overview(stills, output / 'overview.png')
(output / 'manifest.json').write_text(json.dumps(report, indent=2)+'\n')
sequence = project / 'outputs/physical-sequence-5890969'
model = mujoco.MjModel.from_xml_path(str(project / 'outputs/physical-5890912/scene/physical_scene.xml'))
trajectory = dict(np.load(sequence / 'two_jabs_rollout.npz'))
metrics = json.loads((sequence / 'metrics.json').read_text())
metrics['playback_speed'] = .6
metrics['head_center_w'] = [.8539635, .2, 1.2314907452582464]
render(model, trajectory, metrics, output / 'two_jabs', 'two_jabs')
print(json.dumps({'status':'passed','cases':list(selected),'output':str(output)}))
