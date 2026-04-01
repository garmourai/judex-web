#!/usr/bin/env python3
"""Frame extractor for pickleball calibration.

This script creates a calibration folder (default `calibration_1512`) inside a
lens directory (default `tools/pickleball_calib/3.6mm_lens`) and makes
`<calib_folder>/source` and `<calib_folder>/sink` subfolders.

It writes per-folder calibration JSON files and extracts frames from videos
found in `/mnt/data/source` and `/mnt/data/sink` into those subfolders.
"""
import os
import cv2
import glob
import json
import argparse
from datetime import datetime
import fnmatch


SCRIPT_DIR = os.path.dirname(__file__)
DEFAULT_LENS_DIR = os.path.join(SCRIPT_DIR, '3.6mm_lens')
DEFAULT_SOURCE_INPUT = '/mnt/data/source'
DEFAULT_SINK_INPUT = '/mnt/data/sink'
DEFAULT_CALIB_NAME = 'calibration_1512'
DEFAULT_INTERVAL_SEC = 5


DEFAULT_CALIB_JSON = {
    "is_fisheye": False,
    "chessboard_dims": [6, 8],
    "square_size_mm": 25,
    "camera_name": "3.6mm_lens"
}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def write_json(path, data, force=False):
    if os.path.exists(path) and not force:
        print(f"Skipping existing JSON: {path}")
        return path
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Wrote JSON: {path}")
    return path


def find_videos(input_dir):
    if not os.path.isdir(input_dir):
        return []
    patterns = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.h264']
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(input_dir, p)))
    return sorted(files)


def extract_frames_from_video(video_path, out_dir, interval_sec=DEFAULT_INTERVAL_SEC):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
    frame_step = max(1, int(round(fps * interval_sec)))

    base = os.path.splitext(os.path.basename(video_path))[0]
    saved = 0
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_step == 0:
            ts = datetime.utcnow().strftime('%Y%m%dT%H%M%S')
            name = f"{base}_frame_{idx}_{ts}.jpg"
            out_path = os.path.join(out_dir, name)
            cv2.imwrite(out_path, frame)
            saved += 1
        idx += 1

    cap.release()
    print(f"Extracted {saved} frames from {video_path} -> {out_dir}")
    return saved


def extract_folder(input_dir, out_dir, interval_sec, force_json):
    videos = find_videos(input_dir)
    if not videos:
        print(f"No videos found in {input_dir}")
        return 0
    total = 0
    for v in videos:
        total += extract_frames_from_video(v, out_dir, interval_sec)
    return total


def find_latest_media(base_dir, role_keywords=('source', 'sink')):
    """Search recursively under base_dir for media files (.ts, .mp4, .mkv, .avi).
    Return a dict mapping role -> latest_filepath (or None).
    We match files whose path or filename contains the role keyword first; if none found, return newest media file overall for that role position.
    """
    media_exts = ('.ts', '.mp4', '.mkv', '.avi', '.mov')
    latest = {k: None for k in role_keywords}
    latest_mtime = {k: 0 for k in role_keywords}

    for root, dirs, files in os.walk(base_dir):
        for f in files:
            lf = f.lower()
            if not any(lf.endswith(ext) for ext in media_exts):
                continue
            full = os.path.join(root, f)
            mtime = os.path.getmtime(full)
            for role in role_keywords:
                if role in lf or role in root.lower():
                    if mtime > latest_mtime[role]:
                        latest_mtime[role] = mtime
                        latest[role] = full
    # If some roles not found with keyword, pick newest overall file for them
    all_media = []
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if any(f.lower().endswith(ext) for ext in media_exts):
                all_media.append(os.path.join(root, f))
    if all_media:
        newest = sorted(all_media, key=lambda p: os.path.getmtime(p), reverse=True)
        for role in role_keywords:
            if latest[role] is None and newest:
                latest[role] = newest[0]
    return latest


def extract_single_frame_from_media(media_path, out_path):
    """Extract a single representative frame (middle) from a media file and save to out_path"""
    cap = cv2.VideoCapture(media_path)
    if not cap.isOpened():
        print(f"Cannot open media: {media_path}")
        return False
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count <= 0:
        # fallback: grab first frame
        target_idx = 0
    else:
        target_idx = frame_count // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
    ret, frame = cap.read()
    if not ret:
        # fallback to first frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        print(f"Failed to read frame from {media_path}")
        return False
    ensure_dir(os.path.dirname(out_path))
    cv2.imwrite(out_path, frame)
    print(f"Saved single frame {out_path} from {media_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description='Extract frames into calibration folder')
    parser.add_argument('--lens-dir', default=DEFAULT_LENS_DIR, help='Lens base directory')
    parser.add_argument('--calib-name', default=DEFAULT_CALIB_NAME, help='Calibration folder name')
    parser.add_argument('--source-input', default=DEFAULT_SOURCE_INPUT, help='Source videos directory')
    parser.add_argument('--sink-input', default=DEFAULT_SINK_INPUT, help='Sink videos directory')
    parser.add_argument('--interval-sec', type=int, default=DEFAULT_INTERVAL_SEC, help='Seconds between extracted frames')
    parser.add_argument('--single-base', default=None, help='Base directory to search for latest media files (source/sink)')
    parser.add_argument('--force', action='store_true', help='Overwrite JSON files if present')
    args = parser.parse_args()

    lens_dir = os.path.abspath(args.lens_dir)
    calib_dir = os.path.join(lens_dir, args.calib_name)
    src_out = os.path.join(calib_dir, 'source')
    sink_out = os.path.join(calib_dir, 'sink')

    ensure_dir(src_out)
    ensure_dir(sink_out)

    # Write per-folder JSONs
    write_json(os.path.join(src_out, 'source_calibration.json'), DEFAULT_CALIB_JSON, force=args.force)
    write_json(os.path.join(sink_out, 'sink_calibration.json'), DEFAULT_CALIB_JSON, force=args.force)

    print(f"Extracting frames to: {calib_dir}")
    # New: if single-base provided, extract one frame from latest media under that base for source and sink
    if getattr(args, 'single_base', None):
        base = args.single_base
        print(f"Looking for latest media under: {base}")
        latest = find_latest_media(base, role_keywords=('source', 'sink'))
        # save as source.jpg and sink.jpg
        if latest.get('source'):
            extract_single_frame_from_media(latest['source'], os.path.join(src_out, 'source.jpg'))
        else:
            print('No source media found under base')
        if latest.get('sink'):
            extract_single_frame_from_media(latest['sink'], os.path.join(sink_out, 'sink.jpg'))
        else:
            print('No sink media found under base')
        src_count = 1 if latest.get('source') else 0
        sink_count = 1 if latest.get('sink') else 0
    else:
        src_count = extract_folder(args.source_input, src_out, args.interval_sec, args.force)
        sink_count = extract_folder(args.sink_input, sink_out, args.interval_sec, args.force)

    print('\nDone.')
    print(f'  Source frames: {src_count}')
    print(f'  Sink frames:   {sink_count}')


if __name__ == '__main__':
    main()
