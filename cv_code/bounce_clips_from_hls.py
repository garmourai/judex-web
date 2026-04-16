#!/usr/bin/env python3
"""
Generate short MP4 clips (300×300) around bounce points from HLS (playlist.m3u8 + seg_*.ts).

Uses M3U8SegmentReader for frame access and bounce_events.csv (bbox_*),
using bbox center for crop anchor. Source: frame_id == global source index.
Sink: requires triplet CSV mapping Source_Index -> Sink_Index.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import threading
from typing import Dict, List, Optional, Tuple

import cv2

# Load deps without importing cv_code/__init__.py (avoids torch via triplet_pipeline_runner).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_module(mod_name: str, rel_path: str):
    path = os.path.join(_REPO_ROOT, rel_path)
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_m3u8 = _load_module("m3u8_reader_bounce_clips", "cv_code/m3u8_reader.py")
M3U8SegmentReader = _m3u8.M3U8SegmentReader
BBox2D = Dict[str, float]

CROP_MAX = 300  # max side length; actual crop may be smaller near borders

# Default runnable paths (source camera).
DEFAULT_BOUNCE_CSV = "/mnt/data/cv_output/correlation/bounce_events.csv"
DEFAULT_CAMERA = "source"
DEFAULT_SEGMENTS_DIR = "/mnt/data/mar30_test/sync_reports/ts_segments_source/1547"
DEFAULT_TRIPLET_CSV = "/mnt/data/mar30_test/sync_reports/segments_1547/sync/hls_sync_1547_triple.csv"
DEFAULT_OUTPUT_DIR = "/mnt/data/cv_output/bounce_clips"


def _bbox_from_bounce_row(row: Dict[str, str], camera: str) -> Optional[BBox2D]:
    prefix = "bbox_source_" if camera == "source" else "bbox_sink_"
    sx = row.get(prefix + "x", "")
    sy = row.get(prefix + "y", "")
    sw = row.get(prefix + "w", "")
    sh = row.get(prefix + "h", "")
    if not sx or not sy or not sw or not sh:
        return None
    return {
        "x": float(sx),
        "y": float(sy),
        "w": float(sw),
        "h": float(sh),
    }


def _load_triplet_source_to_sink(path: str) -> Dict[int, int]:
    """Build Source_Index -> Sink_Index for rows where both are present."""
    mapping: Dict[int, int] = {}
    if not path or not os.path.exists(path):
        return mapping
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                si_s = row.get("Source_Index", "").strip()
                sk_s = row.get("Sink_Index", "").strip()
                if not si_s:
                    continue
                si = int(si_s)
                if not sk_s or sk_s.lower() in ("none", "null", "nan"):
                    continue
                mapping[si] = int(sk_s)
            except (ValueError, TypeError):
                continue
    return mapping


def _square_crop_rect(cx: float, cy: float, w: int, h: int, max_side: int = CROP_MAX) -> Tuple[int, int, int, int]:
    """
    Largest axis-aligned square centered on (cx, cy), side at most max_side, clipped to image.
    Returns (x0, y0, x1, y1) with x1, y1 exclusive for numpy slicing.
    """
    max_half = max_side // 2
    xi = int(round(cx))
    yi = int(round(cy))
    xi = max(0, min(w - 1, xi))
    yi = max(0, min(h - 1, yi))
    ext_l = xi
    ext_r = w - 1 - xi
    ext_t = yi
    ext_b = h - 1 - yi
    half = min(max_half, ext_l, ext_r, ext_t, ext_b)
    half = max(0, half)
    x0 = xi - half
    x1 = xi + half + 1
    y0 = yi - half
    y1 = yi + half + 1
    return (x0, y0, x1, y1)


def _crop_with_rect(img, rect: Tuple[int, int, int, int]):
    x0, y0, x1, y1 = rect
    return img[y0:y1, x0:x1]


def _draw_landing_bbox_from_xywh(
    patch,
    rect: Tuple[int, int, int, int],
    bbox_xywh: Optional[BBox2D],
):
    """
    Draw bbox from full-image xywh mapped into crop-local coords.
    If bbox is missing/invalid, return patch unchanged.
    """
    if not bbox_xywh:
        return patch

    x0, y0, _, _ = rect
    bw = float(bbox_xywh["w"])
    bh = float(bbox_xywh["h"])
    print(f"[bounce_clips_from_hls] landing bbox size: w={bw:.2f}px h={bh:.2f}px")
    bx = float(bbox_xywh["x"]) - float(x0)
    by = float(bbox_xywh["y"]) - float(y0)

    out = patch.copy()
    ph, pw = out.shape[:2]
    rx1 = max(0, int(round(bx)))
    ry1 = max(0, int(round(by)))
    rx2 = min(pw - 1, int(round(bx + bw)))
    ry2 = min(ph - 1, int(round(by + bh)))
    if rx2 <= rx1 or ry2 <= ry1:
        return patch
    cv2.rectangle(out, (rx1, ry1), (rx2, ry2), (0, 255, 0), thickness=2)
    return out


def _crop_landing_bbox_region(
    patch,
    rect: Tuple[int, int, int, int],
    bbox_xywh: Optional[BBox2D],
) -> Optional[object]:
    """Return exact landing bbox pixel crop (patch-local), or None if unavailable/invalid."""
    if not bbox_xywh:
        return None
    x0, y0, _, _ = rect
    bx = float(bbox_xywh["x"]) - float(x0)
    by = float(bbox_xywh["y"]) - float(y0)
    bw = float(bbox_xywh["w"])
    bh = float(bbox_xywh["h"])

    ph, pw = patch.shape[:2]
    rx1 = max(0, int(round(bx)))
    ry1 = max(0, int(round(by)))
    rx2 = min(pw, int(round(bx + bw)))
    ry2 = min(ph, int(round(by + bh)))
    if rx2 <= rx1 or ry2 <= ry1:
        return None
    return patch[ry1:ry2, rx1:rx2]


def _fps_from_segments_dir(segments_dir: str) -> float:
    ts0 = os.path.join(segments_dir, "seg_00000.ts")
    if os.path.exists(ts0):
        cap = cv2.VideoCapture(ts0)
        if cap.isOpened():
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            cap.release()
            if fps > 0:
                return fps
    return 30.0


def _resolve_read_indices(
    camera: str,
    source_frame_ids: List[int],
    triplet_map: Dict[int, int],
) -> Optional[List[int]]:
    """Return list of global indices for M3U8SegmentReader.read_frame_at, or None if invalid."""
    if camera == "source":
        return list(source_frame_ids)
    out: List[int] = []
    for fid in source_frame_ids:
        sk = triplet_map.get(fid)
        if sk is None:
            return None
        out.append(sk)
    # Reader requires monotonic non-decreasing read indices in time order
    for i in range(len(out) - 1):
        if out[i] > out[i + 1]:
            return None
    return out


def _make_clip(
    reader: M3U8SegmentReader,
    bounce_frame: int,
    source_frame_ids: List[int],
    read_indices: List[int],
    out_path: str,
    source_fps: float,
    playback_speed: float,
    pause_frames: int,
    landing_bbox: Optional[BBox2D],
    landing_bbox_image_path: str,
) -> bool:
    """
    One crop rect from bounce_frame anchor; same rect applied to every frame.
    On the landing frame (bounce_frame), draw a bbox around the bounce center, then
    duplicate that frame `pause_frames` times for a hold.
    output_fps = source_fps * playback_speed (e.g. 0.5 = half-speed playback).
    """
    assert len(source_frame_ids) == len(read_indices)
    frames: List[object] = []
    for sfid, ridx in zip(source_frame_ids, read_indices):
        frame = reader.read_frame_at(ridx)
        if frame is None:
            print(f"[bounce_clips_from_hls] decode failed at read_index={ridx} (source_frame={sfid})")
            return False
        frames.append(frame)

    ih, iw = frames[0].shape[:2]
    if not landing_bbox:
        print(f"[bounce_clips_from_hls] skip bounce_frame={bounce_frame}: missing bbox for camera")
        return False
    u = float(landing_bbox["x"]) + float(landing_bbox["w"]) / 2.0
    v = float(landing_bbox["y"]) + float(landing_bbox["h"]) / 2.0
    rect = _square_crop_rect(u, v, iw, ih, CROP_MAX)

    patches: List[object] = []
    for frame in frames:
        patches.append(_crop_with_rect(frame, rect))

    if not patches:
        return False

    try:
        landing_idx = source_frame_ids.index(bounce_frame)
    except ValueError:
        landing_idx = -1
        print(f"[bounce_clips_from_hls] warning: bounce_frame {bounce_frame} not in window, no bbox/pause")

    out_h, out_w = patches[0].shape[:2]
    out_fps = max(1e-6, float(source_fps) * float(playback_speed))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, out_fps, (out_w, out_h))
    if not writer.isOpened():
        print(f"[bounce_clips_from_hls] could not open VideoWriter for {out_path}")
        return False

    n_hold = max(1, int(pause_frames)) if landing_idx >= 0 else 0

    for i, p in enumerate(patches):
        if i == landing_idx and landing_idx >= 0:
            p_marked = _draw_landing_bbox_from_xywh(
                p,
                rect,
                landing_bbox,
            )
            bbox_crop = _crop_landing_bbox_region(p, rect, landing_bbox)
            if bbox_crop is not None:
                cv2.imwrite(landing_bbox_image_path, bbox_crop)
            for _ in range(n_hold):
                writer.write(p_marked)
        else:
            writer.write(p)

    writer.release()
    return True


def main() -> None:
    p = argparse.ArgumentParser(
        description="Bounce clips from HLS: crop up to 300×300 centered on bounce_frame; optional slow-mo."
    )
    p.add_argument(
        "--bounce-csv",
        default=DEFAULT_BOUNCE_CSV,
        help=f"bounce_events.csv from net_crossings_from_jsonl (default: {DEFAULT_BOUNCE_CSV})",
    )
    p.add_argument("--camera", choices=("source", "sink"), default=DEFAULT_CAMERA)
    p.add_argument(
        "--segments-dir",
        default=DEFAULT_SEGMENTS_DIR,
        help=f"HLS directory with playlist.m3u8 and seg_*.ts (default: {DEFAULT_SEGMENTS_DIR})",
    )
    p.add_argument(
        "--triplet-csv",
        default=DEFAULT_TRIPLET_CSV,
        help=f"hls_sync_*_triple.csv (required for --camera sink, default: {DEFAULT_TRIPLET_CSV})",
    )
    p.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"directory for output MP4s (default: {DEFAULT_OUTPUT_DIR})",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=None,
        help="override source video FPS (default: from first .ts or 30); output FPS = this × playback-speed",
    )
    p.add_argument(
        "--playback-speed",
        type=float,
        default=0.5,
        help="multiply source FPS for output (default 0.5 = half-speed playback)",
    )
    p.add_argument("--frames-before", type=int, default=4, help="frames before bounce_frame (default 4)")
    p.add_argument("--frames-after", type=int, default=5, help="frames after bounce_frame (default 5)")
    p.add_argument(
        "--pause-frames",
        type=int,
        default=5,
        help="how many frames to hold at landing (same frame + bbox repeated; default 5)",
    )
    p.add_argument("--limit", type=int, default=None, help="max number of bounce rows to process")
    args = p.parse_args()

    if args.camera == "sink" and not args.triplet_csv:
        raise SystemExit("--triplet-csv is required when --camera sink")
    if args.frames_before < 0 or args.frames_after < 0:
        raise SystemExit("--frames-before and --frames-after must be non-negative")
    if args.playback_speed <= 0:
        raise SystemExit("--playback-speed must be positive")

    os.makedirs(args.output_dir, exist_ok=True)

    triplet_map = _load_triplet_source_to_sink(args.triplet_csv or "")
    source_fps = args.fps if args.fps is not None else _fps_from_segments_dir(args.segments_dir)

    rows: List[dict] = []
    with open(args.bounce_csv, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    if args.limit is not None:
        rows = rows[: args.limit]

    n_ok = 0
    n_skip = 0

    for idx, row in enumerate(rows):
        try:
            bounce_frame = int(row["bounce_frame"])
        except (KeyError, ValueError) as e:
            print(f"[bounce_clips_from_hls] skip row {idx}: {e}")
            n_skip += 1
            continue

        landing_bbox = _bbox_from_bounce_row(row, args.camera)
        if landing_bbox is not None:
            print(
                f"[bounce_clips_from_hls] bounce_frame={bounce_frame} "
                f"bbox x={landing_bbox['x']:.2f} y={landing_bbox['y']:.2f} "
                f"w={landing_bbox['w']:.2f} h={landing_bbox['h']:.2f}"
            )
        else:
            print(f"[bounce_clips_from_hls] bounce_frame={bounce_frame} bbox missing for camera={args.camera}")
        start_f = bounce_frame - args.frames_before
        end_f = bounce_frame + args.frames_after
        source_frame_ids = list(range(start_f, end_f + 1))

        read_indices = _resolve_read_indices(args.camera, source_frame_ids, triplet_map)
        if read_indices is None:
            print(
                f"[bounce_clips_from_hls] skip bounce_frame={bounce_frame}: "
                f"missing or non-monotonic sink_index for frames in [{start_f}, {end_f}]"
            )
            n_skip += 1
            continue

        out_name = f"bounce_{bounce_frame}_{idx:04d}.mp4"
        out_path = os.path.join(args.output_dir, out_name)
        bbox_img_path = os.path.join(args.output_dir, f"bounce_{bounce_frame}_{idx:04d}_bbox.png")

        stop_event = threading.Event()
        reader = M3U8SegmentReader(args.segments_dir, stop_event, poll_interval=0.05)

        ok = _make_clip(
            reader=reader,
            bounce_frame=bounce_frame,
            source_frame_ids=source_frame_ids,
            read_indices=read_indices,
            out_path=out_path,
            source_fps=source_fps,
            playback_speed=args.playback_speed,
            pause_frames=args.pause_frames,
            landing_bbox=landing_bbox,
            landing_bbox_image_path=bbox_img_path,
        )
        reader.close()

        if ok:
            if bounce_frame in source_frame_ids:
                n_out = len(source_frame_ids) - 1 + max(1, args.pause_frames)
            else:
                n_out = len(source_frame_ids)
            print(f"[bounce_clips_from_hls] wrote {out_path} ({n_out} output frames)")
            n_ok += 1
        else:
            n_skip += 1

    print(f"[bounce_clips_from_hls] done: ok={n_ok} skipped={n_skip}")


if __name__ == "__main__":
    main()
