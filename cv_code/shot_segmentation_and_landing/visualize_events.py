"""
Visualization for postprocessed rally/net/shot/bounce events.

Creates an overlay video from:
- merged per-frame picked 3D points (from trajectory_filter_stats CSVs),
- postprocess JSON output (coarse rallies, net crossings, shots, bounce/landing),
- camera projection matrix (for 3D -> 2D reprojection),
- triplet CSV Source_Index sequence (sync frame reference).
"""

from __future__ import annotations

import os
import pickle
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .filter_stats_merger import merge_filter_stats
from .coarse_rally import merged_rows_to_best_per_frame

# Marker colors (BGR)
BALL_COLOR = (0, 255, 255)  # yellow
BALL_TRAIL_COLOR = (0, 255, 0)  # green
NET_CROSS_COLOR = (0, 0, 255)  # red
SHOT_START_COLOR = (0, 255, 0)  # green
SHOT_END_COLOR = (255, 0, 0)  # blue
BOUNCE_COLOR = (0, 255, 255)  # yellow
LANDING_TEXT_COLOR = (255, 255, 255)  # white


def load_camera(camera_pkl_path: str):
    with open(camera_pkl_path, "rb") as f:
        return pickle.load(f)


def reproject_point(cam, point3d: Tuple[float, float, float]) -> Tuple[int, int]:
    point3d_arr = np.asarray(point3d, dtype=float).reshape(3, 1)
    x_h = np.vstack((point3d_arr, [1.0]))
    p = np.asarray(cam.projection_matrix, dtype=float)
    x_proj = p @ x_h
    x_proj /= x_proj[2, 0]
    return int(x_proj[0, 0]), int(x_proj[1, 0])


def load_triplet_source_indices(triplet_csv_path: str) -> List[int]:
    """
    Build the source-frame sequence from triplet CSV rows.
    Uses Source_Index from each non-empty row, in file order.
    """
    if not triplet_csv_path or not os.path.exists(triplet_csv_path):
        return []

    out: List[int] = []
    with open(triplet_csv_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        try:
            src_idx_col = header.index("Source_Index")
        except ValueError:
            return []

        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if src_idx_col >= len(parts):
                continue
            token = parts[src_idx_col].strip()
            if not token:
                continue
            try:
                out.append(int(token))
            except ValueError:
                continue
    return out


def _build_event_indices(payload: Dict[str, Any]) -> Tuple[set, Dict[int, Dict[str, Any]]]:
    net_cross_frames: set = set()
    shot_events: Dict[int, Dict[str, Any]] = {}

    for rally in payload.get("coarse_rallies", []):
        for f in rally.get("net_crossings", []):
            net_cross_frames.add(int(f))

        for shot in rally.get("shots", []):
            s = shot.get("start_shot_frame")
            e = shot.get("end_shot_frame")
            b = shot.get("bounce_frame")
            n = shot.get("net_crossing_frame")

            if s is not None:
                shot_events.setdefault(int(s), {})["start"] = True
            if e is not None:
                shot_events.setdefault(int(e), {})["end"] = True
            if b is not None:
                shot_events.setdefault(int(b), {})["bounce"] = True
            if n is not None:
                shot_events.setdefault(int(n), {})["net_cross"] = True
    return net_cross_frames, shot_events


def create_event_overlay_video(
    trajectory_output_dir: str,
    postprocess_json_path: str,
    video_path: str,
    camera_pkl_path: str,
    output_video_path: str,
    triplet_csv_path: Optional[str] = None,
    start_sync_frame: Optional[int] = None,
    end_sync_frame: Optional[int] = None,
    trail_length: int = 10,
    fps_override: Optional[float] = None,
    verbose: bool = True,
) -> str:
    """
    Render postprocess events on top of source video.
    """
    # Lazy import so JSON-only flows can run without OpenCV installed.
    import cv2

    with open(postprocess_json_path, "r", encoding="utf-8") as f:
        payload = __import__("json").load(f)

    merged_rows, _ = merge_filter_stats(trajectory_output_dir, verbose=False)
    if start_sync_frame is not None or end_sync_frame is not None:
        lo = start_sync_frame if start_sync_frame is not None else -10**18
        hi = end_sync_frame if end_sync_frame is not None else 10**18
        merged_rows = [
            r for r in merged_rows if lo <= int(r["frame_number_after_sync"]) <= hi
        ]
    best_per_frame = merged_rows_to_best_per_frame(merged_rows)
    if not best_per_frame and verbose:
        print(
            "[visualize] warning: no trajectory points in this sync range; "
            "writing every video frame with event overlays only (no ball trail)."
        )

    cam = load_camera(camera_pkl_path)
    triplet_source_indices = load_triplet_source_indices(triplet_csv_path) if triplet_csv_path else []

    net_cross_frames, shot_events = _build_event_indices(payload)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    fps_use = fps_override if fps_override and fps_override > 0 else fps

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps_use, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create video writer: {output_video_path}")

    trail: deque = deque(maxlen=trail_length)
    # Determine decode window in video-frame space (similar to original code style:
    # seek to actual start frame, then write forward).
    decode_start_idx = 0
    decode_end_idx: Optional[int] = None
    if triplet_source_indices:
        if start_sync_frame is not None:
            for i, sidx in enumerate(triplet_source_indices):
                if sidx >= start_sync_frame:
                    decode_start_idx = i
                    break
            else:
                cap.release()
                writer.release()
                raise RuntimeError("start_sync_frame is beyond available triplet Source_Index range.")
        if end_sync_frame is not None:
            last_i = None
            for i, sidx in enumerate(triplet_source_indices):
                if sidx <= end_sync_frame:
                    last_i = i
                else:
                    break
            decode_end_idx = last_i
            if decode_end_idx is None:
                cap.release()
                writer.release()
                raise RuntimeError("No frames in requested sync range for provided triplet CSV.")
    else:
        decode_start_idx = start_sync_frame or 0
        decode_end_idx = end_sync_frame

    cap.set(cv2.CAP_PROP_POS_FRAMES, decode_start_idx)
    frame_idx = decode_start_idx
    drawn_events = 0

    while True:
        if decode_end_idx is not None and frame_idx > decode_end_idx:
            break
        ret, frame = cap.read()
        if not ret:
            break

        # Determine sync frame id for this video frame.
        if triplet_source_indices:
            if frame_idx >= len(triplet_source_indices):
                break
            sync_frame = triplet_source_indices[frame_idx]
        else:
            sync_frame = frame_idx
        # Decode window already limits which frames we read; never skip writing a frame
        # (older code used continue here and dropped frames from the output MP4).

        if best_per_frame and sync_frame in best_per_frame:
            _, point3d = best_per_frame[sync_frame]
            try:
                u, v = reproject_point(cam, point3d)
                trail.append((u, v))

                # trail
                points = list(trail)
                for i in range(1, len(points)):
                    cv2.line(frame, points[i - 1], points[i], BALL_TRAIL_COLOR, 2)

                # current ball point
                cv2.circle(frame, (u, v), 7, BALL_COLOR, -1)
                cv2.circle(frame, (u, v), 9, (255, 255, 255), 1)
            except Exception:
                pass

        # draw event markers (text badge + anchor marker around current point if available)
        events = shot_events.get(sync_frame, {})
        if sync_frame in net_cross_frames:
            events = dict(events)
            events["net_cross"] = True

        if events:
            y = 30
            if events.get("net_cross"):
                cv2.putText(frame, "NET CROSS", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, NET_CROSS_COLOR, 2)
                y += 25
            if events.get("start"):
                cv2.putText(frame, "SHOT START", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, SHOT_START_COLOR, 2)
                y += 25
            if events.get("end"):
                cv2.putText(frame, "SHOT END", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, SHOT_END_COLOR, 2)
                y += 25
            if events.get("bounce"):
                cv2.putText(frame, "BOUNCE / LANDING", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, BOUNCE_COLOR, 2)
                y += 25
            drawn_events += 1

            if trail:
                u, v = trail[-1]
                if events.get("net_cross"):
                    size = 10
                    cv2.line(frame, (u - size, v - size), (u + size, v + size), NET_CROSS_COLOR, 2)
                    cv2.line(frame, (u - size, v + size), (u + size, v - size), NET_CROSS_COLOR, 2)
                if events.get("start"):
                    cv2.circle(frame, (u, v), 12, SHOT_START_COLOR, 2)
                if events.get("end"):
                    cv2.rectangle(frame, (u - 10, v - 10), (u + 10, v + 10), SHOT_END_COLOR, 2)
                if events.get("bounce"):
                    pts = np.array([[u, v - 10], [u - 10, v], [u, v + 10], [u + 10, v]], np.int32)
                    cv2.polylines(frame, [pts], True, BOUNCE_COLOR, 2)
                    cv2.putText(frame, "LND", (u + 12, v - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, LANDING_TEXT_COLOR, 1)

        cv2.putText(
            frame,
            f"orig_frame={frame_idx} sync_frame={sync_frame}",
            (20, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    if verbose:
        frames_written = frame_idx - decode_start_idx
        print(f"[visualize] wrote: {output_video_path}")
        print(
            f"[visualize] frames_written: {frames_written} "
            f"(decode_idx [{decode_start_idx}, {decode_end_idx if decode_end_idx is not None else '…'}]), "
            f"event-frames: {drawn_events}"
        )
    return output_video_path

