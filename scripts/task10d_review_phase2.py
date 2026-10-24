"""Focused event/media consistency review for Phase 2 preview outputs.

Run after generation. This is an evidence reader, not a motion qualification
system; numerical diagnostics do not establish biomechanics or avoidance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from human_punching.task10c.robot import ROOT


def first(times, mask):
    indices = np.flatnonzero(mask)
    return None if not len(indices) else float(times[indices[0]])


def review(run: Path):
    checks, videos = [], []
    summaries = []
    for trace_path in sorted(run.glob('*_trace.npz')):
        label = trace_path.name.removesuffix('_trace.npz')
        summary = json.loads((run / f'{label}.json').read_text())
        summaries.append(summary)
        with np.load(trace_path) as trace:
            t = trace['time_s']
            frozen = np.linalg.norm(trace['gloves'] - summary['frozen_launch_head'], axis=-1)
            moving = np.linalg.norm(trace['gloves'] - trace['head'][:, None, :], axis=-1)
            checks.extend([
                (label + ':physics_grid', bool(np.allclose(np.diff(t), .005))),
                (label + ':frozen_geometry', bool(np.allclose(frozen, trace['frozen_distances'], atol=2e-7))),
                (label + ':moving_geometry', bool(np.allclose(moving, trace['moving_distances'], atol=2e-7))),
                # Use recorded native precision for event reconstruction; the
                # independent float64 geometry comparison above is separate.
                (label + ':frozen_event', first(t, trace['frozen_distances'].min(axis=1) <= .14) == summary['replay_frozen_target_first_entry_s']),
                (label + ':moving_event', first(t, trace['moving_distances'].min(axis=1) <= .14) == summary['moving_head_first_encounter_s']),
                (label + ':clearance_event', first(t, trace['moving_distances'].min(axis=1) < .16) == summary['first_glove_clearance_violation_s']),
                (label + ':mechanical_event', first(t, ~trace['mechanically_safe']) == summary['first_mechanical_safety_violation_s']),
                (label + ':complete_hold', t[-1] >= summary['recovery_complete_s'] + summary['terminal_hold_s'] - 1e-9),
                (label + ':monotone_clock', bool(np.all(np.diff(trace['source_s']) >= 0))),
                (label + ':no_arrival_claim', summary['safe_arrival_evaluated'] is False),
                (label + ':both_hazards', frozen.shape == (len(t), 2)),
            ])
        for kind in ('external', 'perception'):
            path = run / label / f'{kind}.mp4'
            reader = imageio.get_reader(path)
            meta = reader.get_meta_data()
            selected = set(np.linspace(0, summary['video_frames'] - 1, 24).round().astype(int))
            tiles, count = [], 0
            for i, frame in enumerate(reader):
                count += 1
                if kind == 'external' and i in selected:
                    tile = Image.fromarray(frame).resize((672, 216))
                    ImageDraw.Draw(tile).text((4, 198), f'frame {i} / t={i/50:.2f}s', fill='yellow')
                    tiles.append(tile)
            reader.close()
            checks.extend([(label + ':' + kind + ':all_frames', count == summary['video_frames']),
                           (label + ':' + kind + ':normal_fps', meta['fps'] == 50)])
            videos.append({'path': str(path.relative_to(ROOT)), 'frames': count,
                           'fps': meta['fps'], 'size': meta['size'],
                           'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
            if tiles:
                sheet = Image.new('RGB', (1344, 216 * ((len(tiles)+1)//2)), '#151b23')
                for i, tile in enumerate(tiles):
                    sheet.paste(tile, ((i % 2)*672, (i//2)*216))
                sheet.save(run / label / 'sequence_review.jpg', quality=90)
    result = {'scope': 'media decode and independent event consistency; no dynamic H2 or avoidance claim',
              'run': str(run.relative_to(ROOT)), 'scenarios': len(summaries),
              'checks': {name: bool(ok) for name, ok in checks}, 'passed': int(sum(bool(ok) for _, ok in checks)),
              'failed': [name for name, ok in checks if not ok], 'videos': videos}
    (run / 'focused_review.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('run','scenarios','passed','failed')}, indent=2))
    if not summaries or result['failed']:
        raise SystemExit('Missing scenarios or failed review checks')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path, nargs='+')
    args = parser.parse_args()
    for path in args.runs:
        path = path.resolve(strict=True)
        path.relative_to((ROOT / 'outputs/task10d/phase2').resolve())
        review(path)
