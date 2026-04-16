"""
Incremental bounce clip generation: only process CSV rows with index >= start_row_index.
Loads bounce_clips_from_hls by file path to avoid importing cv_code/__init__.py (torch).
"""

from __future__ import annotations

import csv
import importlib.util
import os
import threading
from typing import List, Optional

_bc = None


def _bounce_module():
    global _bc
    if _bc is None:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        path = os.path.join(repo_root, "cv_code", "bounce_clips_from_hls.py")
        spec = importlib.util.spec_from_file_location("bounce_clips_from_hls_impl", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load {path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _bc = mod
    return _bc


def run_bounce_clips_incremental(
    bounce_csv_path: str,
    camera: str,
    segments_dir: str,
    output_dir: str,
    triplet_csv_path: str,
    start_row_index: int,
    frames_before: int = 8,
    frames_after: int = 8,
    pause_frames: int = 4,
    fps: Optional[float] = None,
    playback_speed: float = 0.5,
    limit: Optional[int] = None,
) -> Tuple[bool, int]:
    """
    Process bounce CSV rows with global index >= start_row_index only.
    Returns (ok, next_start_row_index).
    """
    bc = _bounce_module()
    if not bounce_csv_path or not os.path.exists(bounce_csv_path):
        return True, start_row_index
    if not segments_dir or not os.path.exists(segments_dir):
        return False, start_row_index

    if camera == "sink" and not triplet_csv_path:
        return False, start_row_index

    os.makedirs(output_dir, exist_ok=True)

    rows: List[dict] = []
    with open(bounce_csv_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    if start_row_index < 0:
        start_row_index = 0
    if start_row_index >= len(rows):
        return True, start_row_index

    tail = rows[start_row_index:]
    if limit is not None:
        tail = tail[:limit]

    triplet_map = bc._load_triplet_source_to_sink(triplet_csv_path or "")
    source_fps = fps if fps is not None else bc._fps_from_segments_dir(segments_dir)

    n_ok = 0
    n_skip = 0

    for offset, row in enumerate(tail):
        idx = start_row_index + offset
        bounce_frame = int(row["bounce_frame"])

        landing_bbox = bc._bbox_from_bounce_row(row, camera)
        if landing_bbox is not None:
            print(
                f"[bounce_clips incremental] bounce_frame={bounce_frame} "
                f"bbox x={landing_bbox['x']:.2f} y={landing_bbox['y']:.2f}"
            )
        else:
            print(
                f"[bounce_clips incremental] bounce_frame={bounce_frame} bbox missing camera={camera}"
            )

        start_f = bounce_frame - frames_before
        end_f = bounce_frame + frames_after
        source_frame_ids = list(range(start_f, end_f + 1))

        read_indices = bc._resolve_read_indices(camera, source_frame_ids, triplet_map)
        if read_indices is None:
            print(
                f"[bounce_clips incremental] skip bounce_frame={bounce_frame}: "
                f"missing or non-monotonic sink_index"
            )
            n_skip += 1
            continue

        out_name = f"bounce_{bounce_frame}_{idx:04d}.mp4"
        out_path = os.path.join(output_dir, out_name)
        bbox_img_path = os.path.join(output_dir, f"bounce_{bounce_frame}_{idx:04d}_bbox.png")

        stop_event = threading.Event()
        reader = bc.M3U8SegmentReader(segments_dir, stop_event, poll_interval=0.05)

        ok = bc._make_clip(
            reader=reader,
            bounce_frame=bounce_frame,
            source_frame_ids=source_frame_ids,
            read_indices=read_indices,
            out_path=out_path,
            source_fps=source_fps,
            playback_speed=playback_speed,
            pause_frames=pause_frames,
            landing_bbox=landing_bbox,
            landing_bbox_image_path=bbox_img_path,
        )
        reader.close()

        if ok:
            n_ok += 1
            print(f"[bounce_clips incremental] wrote {out_path}")
        else:
            n_skip += 1

    next_index = start_row_index + len(tail)
    print(
        f"[bounce_clips incremental] camera={camera} "
        f"rows [{start_row_index}, {next_index}) ok={n_ok} skipped={n_skip}"
    )
    return True, next_index
