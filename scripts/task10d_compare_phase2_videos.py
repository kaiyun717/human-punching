"""Compare two complete same-clock Phase 2 external videos at their native rate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from human_punching.task10c.robot import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--before-label', default='BEFORE: target adaptation with curled wrist')
    parser.add_argument('--after-label', default='AFTER: H2 reset 12 cm farther back; fist aligned with forearm')
    parser.add_argument('--key-times', nargs='+', type=float)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to((ROOT/'outputs/task10d/phase2').resolve())
    output.mkdir(parents=True, exist_ok=False)
    paths = [args.before.resolve(), args.after.resolve()]
    readers = [imageio.get_reader(str(p)) for p in paths]
    counts = [r.count_frames() for r in readers]
    rates = [r.get_meta_data()['fps'] for r in readers]
    if counts[0] != counts[1] or rates != [50., 50.]:
        raise ValueError('Comparison requires matching complete 50 fps source-clock videos')
    selected = {0, 25, 34, 37, 40, 43, 50, 70, 98, counts[0]-1}
    if args.key_times is not None:
        selected = {min(counts[0]-1, max(0, round(t*50))) for t in args.key_times}
    tiles = []
    try:
        with imageio.get_writer(output/'before_after.mp4', fps=50, codec='libx264',
                                quality=8, macro_block_size=2) as writer:
            for i in range(counts[0]):
                frames = [Image.fromarray(r.get_data(i)) for r in readers]
                if frames[0].size != frames[1].size:
                    raise ValueError('Video geometry differs')
                width, height = frames[0].size
                canvas = Image.new('RGB', (width, 2*(height+28)), 'white')
                draw = ImageDraw.Draw(canvas)
                labels = [args.before_label, args.after_label]
                for row, (frame, label) in enumerate(zip(frames, labels)):
                    y = row*(height+28)
                    draw.text((12, y+7), f'{label} | t={i/50:.2f}s | normal speed', fill='black')
                    canvas.paste(frame, (0, y+28))
                writer.append_data(np.asarray(canvas))
                if i in selected:
                    canvas.save(output/f'frame_{i:04d}.jpg', quality=95)
                    tiles.append(canvas.resize((672, 460)))
        sheet = Image.new('RGB', (1344, 460*((len(tiles)+1)//2)), 'white')
        for i, tile in enumerate(tiles):
            sheet.paste(tile, ((i%2)*672, (i//2)*460))
        sheet.save(output/'comparison_sheet.jpg', quality=93)
    finally:
        for r in readers:
            r.close()
    record = {'before':str(paths[0]), 'after':str(paths[1]), 'frames':counts[0],
              'labels':[args.before_label, args.after_label],
              'fps':50, 'matched_source_clock':True,
              'source_sha256':[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths],
              'comparison_sha256':hashlib.sha256((output/'before_after.mp4').read_bytes()).hexdigest(),
              'scope':'Visual comparison only; runtime and actual mounted camera unchanged.'}
    (output/'comparison.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
