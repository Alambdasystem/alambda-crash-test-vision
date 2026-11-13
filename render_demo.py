#!/usr/bin/env python3
"""Render a real crash-test clip with observed vehicle boxes and a source-time replay.

No generated/interpolated video frames and no physical crash measurements.
The reviewed contact annotation is an uncertain visual reference, not a sensor trigger.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
WIDTH, HEIGHT = 1080, 1350
BG = (9, 16, 27)
CARD = (17, 29, 44)
EDGE = (41, 58, 77)
INK = (237, 244, 250)
MUTED = (158, 179, 198)
TEAL = (81, 229, 205)
ORANGE = (255, 184, 104)
FONT_DIR = Path('/usr/share/fonts/truetype/dejavu')
FONTS = {}


def font(size, bold=False):
    key = (size, bold)
    if key not in FONTS:
        name = 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
        candidates = [FONT_DIR / name, Path(name)]
        for candidate in candidates:
            try:
                FONTS[key] = ImageFont.truetype(str(candidate), size)
                break
            except OSError:
                continue
        else:
            FONTS[key] = ImageFont.load_default(size=size)
    return FONTS[key]


def label(draw, xy, text, size=22, color=INK, bold=False):
    draw.text(xy, text, font=font(size, bold), fill=color)


def fit_label(draw, xy, text, max_width, size=22, color=INK, bold=False):
    while size > 12 and draw.textlength(text, font=font(size, bold)) > max_width:
        size -= 1
    label(draw, xy, text, size, color, bold)


def dashed_line(draw, x, y1, y2, color, width=2, dash=9):
    for y in range(int(y1), int(y2), dash * 2):
        draw.line((x, y, x, min(y + dash, y2)), fill=color, width=width)


def read_frames(path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f'Cannot open video: {path}')
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames or fps <= 0:
        raise RuntimeError('Video has no decodable frames or invalid frame rate')
    return frames, fps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'assets/source.mp4')
    parser.add_argument('--tracks', type=Path, default=ROOT / 'output/tracks.json')
    parser.add_argument('--provenance', type=Path, default=ROOT / 'assets/source.json')
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'output/Alambda_Crash_Test_Slow_Motion.mp4')
    parser.add_argument('--preview', type=Path,
                        default=ROOT / 'output/Alambda_Crash_Test_Preview.jpg')
    parser.add_argument('--contact-frame', type=int)
    parser.add_argument('--contact-uncertainty', type=int)
    parser.add_argument('--replay-after-frames', type=int, default=60)
    parser.add_argument('--replay-repeat', type=int, default=4,
                        help='Repeat each source frame N times; 4 means 0.25x source playback')
    parser.add_argument('--hold-seconds', type=float, default=1.2)
    parser.add_argument('--end-hold-seconds', type=float, default=1.5)
    parser.add_argument('--display-crop', type=int, nargs=4, default=[100, 0, 1180, 720],
                        metavar=('X1', 'Y1', 'X2', 'Y2'),
                        help='Reviewed crop for this supplied clip; use full frame for other clips')
    parser.add_argument('--detail-crop', type=int, nargs=4, default=[670, 240, 970, 440],
                        metavar=('X1', 'Y1', 'X2', 'Y2'),
                        help='Reviewed fixed impact area in source image pixels')
    args = parser.parse_args()
    cv2.setNumThreads(4)
    video, fps = read_frames(args.source)
    track_data = json.loads(args.tracks.read_text())
    provenance = json.loads(args.provenance.read_text())
    review = provenance.get('contact_review', {})
    contact = args.contact_frame
    if contact is None:
        contact = review.get('first_contact_frame')
    if contact is None:
        raise ValueError('Supply a reviewed contact frame in source.json or --contact-frame')
    contact = int(contact)
    uncertainty = args.contact_uncertainty
    if uncertainty is None:
        uncertainty = int(review.get('uncertainty_frames', 0))
    count = len(video)
    if not (0 <= contact < count) or uncertainty < 0:
        raise ValueError('Invalid contact annotation')
    if args.replay_repeat < 1 or args.replay_after_frames < 1:
        raise ValueError('Replay repeat and replay frame count must be positive')
    frame_map = {int(f['frame']): f for f in track_data['frames']}
    if set(frame_map) != set(range(count)):
        raise ValueError('Tracks must contain one record for every decoded video frame')
    h, w = video[0].shape[:2]
    if abs(float(track_data['video']['fps']) - fps) > 0.01:
        raise ValueError('Source and tracks frame rates disagree')
    for crop in [args.display_crop, args.detail_crop]:
        x1, y1, x2, y2 = crop
        if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
            raise ValueError(f'Crop {crop} is outside the {w}x{h} source')

    objects = {i: frame_map[i].get('objects', []) for i in range(count)}
    if any(obj.get('observation') != 'detected' for row in objects.values() for obj in row):
        raise ValueError('This renderer accepts observed detector boxes only')
    replay_start = max(0, contact - 16)
    replay_end = min(count - 1, contact + args.replay_after_frames)
    hold_frames = max(0, round(args.hold_seconds * fps))
    end_hold_frames = max(0, round(args.end_hold_seconds * fps))
    sequence = [(i, 'overview') for i in range(count)]
    for i in range(replay_start, replay_end + 1):
        sequence.extend([(i, 'replay')] * args.replay_repeat)
        if i == contact:
            sequence.extend([(i, 'hold')] * hold_frames)
    sequence.extend([(replay_end, 'end')] * end_hold_frames)

    # Source banner and camera timing are preserved; only outer black bars are cropped.
    vx, vy, vw, vh = 32, 140, 1016, 678
    crop_x1, crop_y1, crop_x2, crop_y2 = args.display_crop
    sx = vw / (crop_x2 - crop_x1)
    sy = vh / (crop_y2 - crop_y1)

    def point(x, y):
        return (vx + (x - crop_x1) * sx, vy + (y - crop_y1) * sy)

    def render(source_index, mode):
        frame = video[source_index]
        canvas = Image.new('RGB', (WIDTH, HEIGHT), BG)
        draw = ImageDraw.Draw(canvas)
        label(draw, (34, 23), 'ALAMBDA SYSTEMS', 20, TEAL, True)
        label(draw, (772, 25), 'COMPUTER VISION / 02', 17, MUTED)
        label(draw, (32, 60), 'Crash test, frame by frame', 49, INK, True)
        crop = Image.fromarray(frame[crop_y1:crop_y2, crop_x1:crop_x2])
        canvas.paste(crop.resize((vw, vh), Image.Resampling.LANCZOS), (vx, vy))
        draw.rectangle((vx, vy, vx + vw, vy + vh), outline=EDGE, width=2)

        # Bbox-center trail is a sequence of actual observations, not a physical trajectory.
        trail = []
        for j in range(max(0, source_index - 50), source_index + 1):
            found = next((o for o in objects[j] if o['id'] == 'V1'), None)
            if found:
                p = point(*found['point'])
                if vx <= p[0] <= vx + vw and vy <= p[1] <= vy + vh:
                    trail.append(p)
            else:
                if len(trail) > 1:
                    draw.line(trail, fill=TEAL, width=3)
                trail = []
        if len(trail) > 1:
            draw.line(trail, fill=TEAL, width=3)
        for obj in objects[source_index]:
            x1, y1, x2, y2 = obj['bbox']
            left, top = point(x1, y1)
            right, bottom = point(x2, y2)
            left, right = max(vx + 2, left), min(vx + vw - 2, right)
            top, bottom = max(vy + 2, top), min(vy + vh - 2, bottom)
            if right <= left or bottom <= top:
                continue
            # Quiet full box plus stronger corner brackets.
            draw.rectangle((left, top, right, bottom), outline=TEAL, width=2)
            length = min(24, (right - left) / 6, (bottom - top) / 6)
            for cx, cy, dx, dy in [(left, top, 1, 1), (right, top, -1, 1),
                                   (left, bottom, 1, -1), (right, bottom, -1, -1)]:
                draw.line((cx, cy + dy * length, cx, cy, cx + dx * length, cy),
                          fill=TEAL, width=4)
            tag = 'V1  /  VEHICLE'
            tag_w = int(draw.textlength(tag, font=font(19, True))) + 22
            tag_top = max(vy + 54, top - 34)
            draw.rounded_rectangle((left, tag_top, left + tag_w, tag_top + 31),
                                   radius=5, fill=BG)
            label(draw, (left + 10, tag_top + 3), tag, 19, TEAL, True)
            cx, cy = point(*obj['point'])
            draw.ellipse((cx - 5, cy - 5, cx + 5, cy + 5), fill=TEAL)

        stage_titles = {
            'overview': '01 / OVERVIEW',
            'replay': '02 / IMPACT REPLAY',
            'hold': '02 / CONTACT REVIEW',
            'end': '02 / REPLAY COMPLETE',
        }
        playback = {
            'overview': '1x source playback',
            'replay': f'{1 / args.replay_repeat:.2f}x source playback',
            'hold': 'Freeze frame',
            'end': 'Final frame held',
        }
        draw.rounded_rectangle((32, 836, 1048, 902), radius=12, fill=CARD, outline=EDGE)
        label(draw, (52, 851), stage_titles[mode], 24, TEAL if mode != 'hold' else ORANGE, True)
        text = playback[mode]
        text_w = draw.textlength(text, font=font(24))
        label(draw, (1026 - text_w, 851), text, 24, INK)

        # An enlarged crop of the same decoded source frame, without synthetic in-between frames.
        dx1, dy1, dx2, dy2 = args.detail_crop
        detail = Image.fromarray(frame[dy1:dy2, dx1:dx2])
        draw.rounded_rectangle((32, 920, 510, 1184), radius=12, fill=CARD, outline=EDGE)
        label(draw, (49, 930), 'IMPACT DETAIL', 18, MUTED, True)
        detail.thumbnail((444, 208), Image.Resampling.LANCZOS)
        # thumbnail does not enlarge; resize explicitly to a fitted display rectangle.
        factor = min(444 / (dx2 - dx1), 208 / (dy2 - dy1))
        detail = Image.fromarray(frame[dy1:dy2, dx1:dx2]).resize(
            (round((dx2 - dx1) * factor), round((dy2 - dy1) * factor)), Image.Resampling.LANCZOS)
        canvas.paste(detail, (49 + (444 - detail.width) // 2, 965 + (208 - detail.height) // 2))

        draw.rounded_rectangle((528, 920, 1048, 1184), radius=12, fill=CARD, outline=EDGE)
        lo, hi = max(0, contact - uncertainty), min(count - 1, contact + uncertainty)
        phase = ('APPROACH' if source_index < lo else
                 'CONTACT WINDOW' if source_index <= hi else 'AFTER CONTACT')
        phase_color = ORANGE if lo <= source_index <= hi else TEAL
        label(draw, (550, 935), phase, 25, phase_color, True)
        label(draw, (550, 986), 'CLIP FRAME', 17, MUTED)
        label(draw, (800, 986), 'CLIP TIME', 17, MUTED)
        label(draw, (550, 1014), f'{source_index:03d} / {count - 1:03d}', 30, INK, True)
        label(draw, (800, 1014), f'{source_index / fps:.3f}s', 30, INK, True)
        draw.line((550, 1065, 1027, 1065), fill=EDGE, width=1)
        if objects[source_index]:
            confidence = objects[source_index][0]['confidence']
            label(draw, (550, 1083), f'Detection score  {confidence:.2f}', 22, INK)
        else:
            label(draw, (550, 1083), 'No vehicle detection this frame', 20, MUTED)
        label(draw, (550, 1122), 'Trail = detected box centers', 20, MUTED)
        label(draw, (550, 1151), 'Clock = source-clip playback time', 17, MUTED)

        # Full-clip timeline: shaded interval conveys manual contact uncertainty.
        tx1, tx2, ty = 48, 1032, 1240
        scale = (tx2 - tx1) / max(1, count - 1)
        label(draw, (34, 1202), 'SOURCE FRAME', 16, MUTED, True)
        note = f'Contact review: frame {contact} +/- {uncertainty}'
        note_w = draw.textlength(note, font=font(16))
        label(draw, (1046 - note_w, 1202), note, 16, ORANGE)
        draw.line((tx1, ty, tx2, ty), fill=EDGE, width=6)
        draw.line((tx1, ty, tx1 + source_index * scale, ty), fill=TEAL, width=6)
        bx1, bx2 = tx1 + lo * scale, tx1 + hi * scale
        draw.rounded_rectangle((bx1 - 2, ty - 12, bx2 + 2, ty + 12), radius=3,
                               fill=(130, 86, 46))
        dashed_line(draw, tx1 + contact * scale, ty - 15, ty + 15, ORANGE, width=2, dash=4)
        px = tx1 + source_index * scale
        draw.ellipse((px - 7, ty - 7, px + 7, ty + 7), fill=INK)
        label(draw, (tx1 - 6, 1256), '0', 15, MUTED)
        label(draw, (tx2 - 28, 1256), str(count - 1), 15, MUTED)

        label(draw, (34, 1289), 'NHTSA test footage via Wikimedia Commons', 17, MUTED)
        label(draw, (34, 1316), 'Source is already slow motion. OpenCV + YOLOX | Assisted visual review', 15, MUTED)
        return np.asarray(canvas)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.preview.parent.mkdir(parents=True, exist_ok=True)
    rate = Fraction(fps).limit_denominator(1001)
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
               '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{WIDTH}x{HEIGHT}',
               '-r', f'{rate.numerator}/{rate.denominator}', '-i', '-', '-an',
               '-c:v', 'libx264', '-preset', 'fast', '-crf', '20',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(args.output)]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        last_key, last_rgb = None, None
        for output_index, key in enumerate(sequence):
            if key != last_key:
                last_rgb = render(*key)
                last_key = key
            encoder.stdin.write(last_rgb.tobytes())
            if output_index % 120 == 0:
                print(f'Rendered {output_index}/{len(sequence)} output frames', flush=True)
        encoder.stdin.close()
        error = encoder.stderr.read().decode(errors='replace')
        returncode = encoder.wait()
        if returncode:
            raise RuntimeError(f'ffmpeg failed ({returncode}): {error}')
    except BaseException:
        encoder.kill()
        encoder.wait()
        raise
    preview_frame = min(count - 1, contact + 27)
    Image.fromarray(render(preview_frame, 'replay')).save(args.preview, quality=94)
    manifest = {
        'source_frames': count,
        'source_fps': fps,
        'source_clip_duration_s': count / fps,
        'output_frames': len(sequence),
        'output_fps': float(rate),
        'output_duration_s': len(sequence) / float(rate),
        'dimensions': [WIDTH, HEIGHT],
        'contact_review': {'frame': contact, 'uncertainty_frames': uncertainty,
                           'method': 'manual visual review'},
        'replay': {'start_frame': replay_start, 'end_frame_inclusive': replay_end,
                   'source_playback_rate': 1 / args.replay_repeat,
                   'frame_repeat_count': args.replay_repeat,
                   'contact_hold_output_frames': hold_frames,
                   'end_hold_output_frames': end_hold_frames},
        'frames_are_generated_or_interpolated': False,
        'audio': False,
        'video_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
        'sequence': [{'output_frame': i, 'source_frame': f, 'mode': mode}
                     for i, (f, mode) in enumerate(sequence)],
    }
    summary_path = args.output.parent / 'render_summary.json'
    summary_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'sequence'}, indent=2))


if __name__ == '__main__':
    main()
