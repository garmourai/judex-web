#!/usr/bin/env python3
"""
Generate short MP4 clips (300×300) around bounce points from HLS (playlist.m3u8 + seg_*.ts).

Uses M3U8SegmentReader for frame access, trajectory_selection.jsonl for per-frame 3D,
camera pickle + reproject_point for the crop center. Source: frame_id == global source index.
Sink: requires triplet CSV mapping Source_Index -> Sink_Index.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import pickle
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

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


def _coerce_pickled_camera_payload(payload, pkl_path: str):
    """
    Same behavior as data_loader._coerce_pickled_camera_payload (no pandas import).
    """
    if not isinstance(payload, dict):
        return payload

    d = payload
    p = Path(pkl_path)
    K = np.asarray(d["camera_matrix"], dtype=np.float64)
    dist = np.asarray(d["dist_coeffs"], dtype=np.float64).reshape(-1, 1)
    wh = (int(d["image_size"][0]), int(d["image_size"][1]))
    ctype = d.get("calibration_type") or "pinhole"

    if ctype == "fisheye":
        D = dist.astype(np.float64).reshape(4, 1)
        newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D, wh, np.eye(3), balance=1.0
        )
        new_scaled = newK
    else:
        newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, wh, 1.0, wh)
        new_scaled = None

    R = None
    tvec = None
    P = None
    ext_path = p.resolve().parent / "extrinsic_pose_undistorted.json"
    if ext_path.is_file():
        extr = json.loads(ext_path.read_text(encoding="utf-8"))
        if extr.get("solve_space") == "distorted":
            rvec = np.asarray(extr["rvec"], dtype=np.float64).reshape(3, 1)
            tvec = np.asarray(extr["tvec"], dtype=np.float64).reshape(3, 1)
            R, _ = cv2.Rodrigues(rvec)
            P = newK @ np.hstack((R, tvec))

    ns = SimpleNamespace()
    ns.camera_matrix = K
    ns.distortion_coefficients = dist
    ns.new_camera_matrix = newK
    ns.new_scaled_camera_matrix = new_scaled
    ns.calibration_type = ctype
    ns.image_size = wh
    ns.rotation_matrix = R
    ns.translation_vectors = tvec
    ns.projection_matrix = P
    ns.calibration_rotation_vectors = d.get("rvecs")
    ns.calibration_translation_vectors = d.get("tvecs")
    ns.final_dimensions = None
    return ns


_m3u8 = _load_module("m3u8_reader_bounce_clips", "cv_code/m3u8_reader.py")
M3U8SegmentReader = _m3u8.M3U8SegmentReader

_utils = _load_module("viz_utils_bounce_clips", "cv_code/correlation/visualization/utils.py")
reproject_point = _utils.reproject_point

Point3D = Tuple[float, float, float]

CROP_MAX = 300  # max side length; actual crop may be smaller near borders


def _load_trajectory_jsonl(path: str) -> Dict[int, Point3D]:
    out: Dict[int, Point3D] = {}
    if not path or not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            fid = obj.get("frame_id")
            sel = obj.get("current_selected_point")
            if not isinstance(fid, int) or not isinstance(sel, dict):
                continue
            try:
                out[fid] = (float(sel["x"]), float(sel["y"]), float(sel["z"]))
            except (KeyError, TypeError, ValueError):
                continue
    return out


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


def _load_camera(pkl_path: str):
    with open(pkl_path, "rb") as f:
        cam = pickle.load(f)
    cam = _coerce_pickled_camera_payload(cam, pkl_path)
    if getattr(cam, "projection_matrix", None) is None:
        raise ValueError(
            f"Camera at {pkl_path} has no projection_matrix after load "
            "(needs extrinsic_pose_undistorted.json next to pickle for dict payloads)."
        )
    return cam


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


def _crop_with_rect(img: np.ndarray, rect: Tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = rect
    return img[y0:y1, x0:x1]


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
    cam,
    trajectory: Dict[int, Point3D],
    fallback_xyz: Point3D,
    bounce_frame: int,
    source_frame_ids: List[int],
    read_indices: List[int],
    out_path: str,
    source_fps: float,
    playback_speed: float,
) -> bool:
    """
    One crop rect from bounce_frame anchor; same rect applied to every frame.
    output_fps = source_fps * playback_speed (e.g. 0.5 = half-speed playback).
    """
    assert len(source_frame_ids) == len(read_indices)
    anchor_xyz = trajectory.get(bounce_frame, fallback_xyz)

    frames: List[np.ndarray] = []
    for sfid, ridx in zip(source_frame_ids, read_indices):
        frame = reader.read_frame_at(ridx)
        if frame is None:
            print(f"[bounce_clips_from_hls] decode failed at read_index={ridx} (source_frame={sfid})")
            return False
        frames.append(frame)

    ih, iw = frames[0].shape[:2]
    u, v = reproject_point(cam, anchor_xyz)
    rect = _square_crop_rect(u, v, iw, ih, CROP_MAX)

    patches: List[np.ndarray] = []
    for frame in frames:
        patches.append(_crop_with_rect(frame, rect))

    if not patches:
        return False

    out_h, out_w = patches[0].shape[:2]
    out_fps = max(1e-6, float(source_fps) * float(playback_speed))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, out_fps, (out_w, out_h))
    if not writer.isOpened():
        print(f"[bounce_clips_from_hls] could not open VideoWriter for {out_path}")
        return False
    for p in patches:
        writer.write(p)
    writer.release()
    return True


def main() -> None:
    p = argparse.ArgumentParser(
        description="Bounce clips from HLS: crop up to 300×300 centered on bounce_frame; optional slow-mo."
    )
    p.add_argument("--bounce-csv", required=True, help="bounce_events.csv from net_crossings_from_jsonl")
    p.add_argument("--trajectory-jsonl", required=True, help="trajectory_selection.jsonl")
    p.add_argument("--camera", choices=("source", "sink"), required=True)
    p.add_argument("--segments-dir", required=True, help="HLS directory with playlist.m3u8 and seg_*.ts")
    p.add_argument("--camera-pkl", required=True, help="camera_object.pkl for reprojection")
    p.add_argument(
        "--triplet-csv",
        default=None,
        help="hls_sync_*_triple.csv (required for --camera sink)",
    )
    p.add_argument("--output-dir", required=True, help="directory for output MP4s")
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
    trajectory = _load_trajectory_jsonl(args.trajectory_jsonl)
    cam = _load_camera(args.camera_pkl)

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
            bx = float(row["x"])
            by = float(row["y"])
            bz = float(row["z"])
        except (KeyError, ValueError) as e:
            print(f"[bounce_clips_from_hls] skip row {idx}: {e}")
            n_skip += 1
            continue

        fallback_xyz: Point3D = (bx, by, bz)
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

        stop_event = threading.Event()
        reader = M3U8SegmentReader(args.segments_dir, stop_event, poll_interval=0.05)

        ok = _make_clip(
            reader=reader,
            cam=cam,
            trajectory=trajectory,
            fallback_xyz=fallback_xyz,
            bounce_frame=bounce_frame,
            source_frame_ids=source_frame_ids,
            read_indices=read_indices,
            out_path=out_path,
            source_fps=source_fps,
            playback_speed=args.playback_speed,
        )
        reader.close()

        if ok:
            print(f"[bounce_clips_from_hls] wrote {out_path} ({len(source_frame_ids)} frames)")
            n_ok += 1
        else:
            n_skip += 1

    print(f"[bounce_clips_from_hls] done: ok={n_ok} skipped={n_skip}")


if __name__ == "__main__":
    main()
