"""Cam1 overlay MP4s from triangulated 3D points (reprojection onto OriginalFrameBuffer frames)."""

import json
import os
import pickle
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ...inference.inference import get_trajectory_color
from ..pairwise_correlation.data.data_loader import _coerce_pickled_camera_payload
from ...triplet_csv_reader import OriginalFrameBuffer as _OriginalFrameBuffer
from .utils import reproject_point

# Overlay text: white on dark panels. HUD + trajectory labels +50% vs prior; cost panel matches HUD.
_VIZ_TEXT_COLOR = (255, 255, 255)
_VIZ_FONT = cv2.FONT_HERSHEY_SIMPLEX
_VIZ_THICKNESS = 1
_VIZ_SCALE_HUD = 0.5625
_VIZ_SCALE_LABEL = 0.5625
_VIZ_SCALE_COST_TITLE = 0.61875
_VIZ_SCALE_COST_LINE = 0.50625


def _load_trajectory_selection_jsonl(
    jsonl_path: str, frame_ranges: List[Tuple[int, int]]
) -> Dict[int, Dict[str, Any]]:
    """Load post-select_best records; last line wins if frame_id repeats."""
    by_frame: Dict[int, Dict[str, Any]] = {}
    if not os.path.exists(jsonl_path):
        return by_frame
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            fid = int(obj.get("frame_id", -1))
            if frame_ranges and not any(s <= fid <= e for s, e in frame_ranges):
                continue
            by_frame[fid] = obj
    return by_frame


def _build_traj_maps_from_selection(
    selection_by_frame: Dict[int, Dict[str, Any]],
) -> Tuple[Dict, Dict, Dict]:
    frame_trajectory_map: Dict[int, List[Tuple[int, Tuple[float, float, float]]]] = {}
    trajectory_history: Dict[int, List[Tuple[int, Tuple[float, float, float]]]] = {}
    frame_removed_trajectory_map: Dict[int, List[Tuple[float, float, float]]] = {}
    for fid, obj in selection_by_frame.items():
        for cand in obj.get("active_trajectories", []):
            if cand.get("is_selected"):
                tid = int(cand["trajectory_id"])
                coords = (float(cand["x"]), float(cand["y"]), float(cand["z"]))
                frame_trajectory_map.setdefault(fid, []).append((tid, coords))
                trajectory_history.setdefault(tid, []).append((fid, coords))
        for ign in obj.get("current_ignored_points", []):
            coords = (float(ign["x"]), float(ign["y"]), float(ign["z"]))
            frame_removed_trajectory_map.setdefault(fid, []).append(coords)
    return frame_trajectory_map, trajectory_history, frame_removed_trajectory_map


def _build_frame_data_and_costs_from_selection(
    selection_by_frame: Dict[int, Dict[str, Any]],
) -> Tuple[Dict[int, Tuple], Dict[int, Dict[str, List[float]]]]:
    """
    Build per-frame point lists and cost component lists from trajectory_selection.jsonl only.
    Each active_trajectories entry may include a 'costs' dict (f1, f2, epi, reproj, temp).
    """
    all_frame_data: Dict[int, Tuple] = {}
    frame_costs_map: Dict[int, Dict[str, List[float]]] = {}
    for fid, obj in selection_by_frame.items():
        coords_list: List[Tuple[float, float, float]] = []
        lists: Dict[str, List[float]] = {"f1": [], "f2": [], "epi": [], "reproj": [], "temp": []}
        for c in obj.get("active_trajectories", []):
            coords_list.append((float(c["x"]), float(c["y"]), float(c["z"])))
            co = c.get("costs") or {}
            for key in ("f1", "f2", "epi", "reproj", "temp"):
                if key in co:
                    lists[key].append(float(co[key]))
        if coords_list:
            all_frame_data[fid] = (None, coords_list)
        frame_costs_map[fid] = lists
    return all_frame_data, frame_costs_map


def _left_hud_dimensions() -> Tuple[int, int, int]:
    """Width, total height (px), and vertical step between HUD text baselines."""
    hud_w = 400
    sample = "FPS: 30  |  frame_id: 999999  |  camera_stream_index: 999999"
    (_, th), bl = cv2.getTextSize(sample, _VIZ_FONT, _VIZ_SCALE_HUD, _VIZ_THICKNESS)
    line_step = th + bl + 6
    overlay_height = 10 + 4 * line_step + 10
    return hud_w, overlay_height, line_step


def _draw_propagated_cost_panel(
    frame_img: np.ndarray,
    frame_idx: int,
    frame_costs_map: Dict[int, Dict[str, List[float]]],
    left_hud_right_x: int,
    left_hud_bottom_y: int,
    gap: int = 8,
) -> None:
    """Top-right cost summary; box size from measured text; avoids overlap with left HUD."""
    default_lists = {"f1": [], "f2": [], "epi": [], "reproj": [], "temp": []}
    cost_info = frame_costs_map.get(frame_idx, default_lists)
    cost_lines: List[str] = []
    for key, label in (("f1", "F1"), ("f2", "F2"), ("epi", "Epi"), ("reproj", "Rho"), ("temp", "Temp")):
        vals = cost_info.get(key, []) or []
        if vals:
            shown = ", ".join(f"{float(v):.2f}" for v in vals[:3])
            suffix = "..." if len(vals) > 3 else ""
            cost_lines.append(f"{label}: [{shown}{suffix}]")
        else:
            cost_lines.append(f"{label}: []")

    lines_to_draw: List[Tuple[str, float]] = [("Propagated Costs", _VIZ_SCALE_COST_TITLE)]
    lines_to_draw.extend((ln, _VIZ_SCALE_COST_LINE) for ln in cost_lines)

    margin = 6
    line_gap = 4
    lines_meta: List[Tuple[str, float, int, int, int]] = []
    for text, scale in lines_to_draw:
        (tw, th), bl = cv2.getTextSize(text, _VIZ_FONT, scale, _VIZ_THICKNESS)
        lines_meta.append((text, scale, tw, th, bl))

    box_w = margin * 2 + max(m[2] for m in lines_meta)
    inner_h = sum(m[3] + m[4] + line_gap for m in lines_meta) - line_gap
    box_h = margin * 2 + inner_h

    right_pad = 15
    x2 = frame_img.shape[1] - right_pad
    x1 = max(0, x2 - box_w)
    y1 = 10
    if x1 < left_hud_right_x + gap:
        y1 = left_hud_bottom_y + gap
    y2 = y1 + box_h

    overlay_r = frame_img.copy()
    cv2.rectangle(overlay_r, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay_r, 0.55, frame_img, 0.45, 0, frame_img)

    y_cursor = y1 + margin
    for text, scale, _tw, th, bl in lines_meta:
        y_cursor += th + bl
        cv2.putText(
            frame_img,
            text,
            (x1 + margin, y_cursor),
            _VIZ_FONT,
            scale,
            _VIZ_TEXT_COLOR,
            _VIZ_THICKNESS,
        )
        y_cursor += line_gap


def create_visualization_from_triangulation(
    frame_segments: List[Tuple[int, int]],
    output_dir: str,
    original_buffer,
    camera_1_cam_path: str,
    camera_1_id: str,
    video_chunk_size: int = 1000,
    profiler=None,
    trajectory_data: Optional[Dict] = None,
    removed_trajectory_data: Optional[Dict] = None,
    trail_length: int = 10,
    show_trajectory_labels: bool = True,
    tracker_base_dir: Optional[str] = None,
    camera_1_csv_path: Optional[str] = None,
):
    """
    Create visualization videos by reprojecting 3D coordinates to camera 1 frames.

    Args:
        frame_segments: List of (start_frame, end_frame) tuples that were triangulated
        output_dir: Output directory for videos; also used to locate trajectory_selection.jsonl if tracker_base_dir is not set
        original_buffer: OriginalFrameBuffer for camera 1 frames (source_index == global frame id)
        camera_1_cam_path: Path to camera 1 calibration pickle file
        camera_1_id: Camera 1 identifier
        video_chunk_size: Number of frames per video chunk (default: 1000)
        profiler: Optional TimeProfiler instance for recording timing metrics
        trajectory_data: Dict mapping segment tuple to list of trajectories (kept trajectories)
        removed_trajectory_data: Dict mapping segment tuple to list of removed trajectories (drawn as big black circles)
        trail_length: Number of past frames to show in trajectory trail (default: 10)
        show_trajectory_labels: Whether to show trajectory ID text labels (default: True)
        tracker_base_dir: If set, read trajectory_selection.jsonl from this directory (e.g. correlation output read path)
        camera_1_csv_path: Unused (kept for call-site compatibility; OFB uses identity frame mapping)
    """
    visualization_start_time = time.time()
    if not frame_segments:
        print("[CorrelationWorker]    ❌ create_visualization_from_triangulation: frame_segments is empty")
        return
    if not isinstance(original_buffer, _OriginalFrameBuffer):
        print("[CorrelationWorker]    ❌ Camera 1 visualization requires OriginalFrameBuffer")
        return

    segment_ranges = [(int(s), int(e)) for s, e in frame_segments]
    f_min = min(s for s, _ in segment_ranges)
    f_max = max(e for _, e in segment_ranges)

    with open(camera_1_cam_path, "rb") as f:
        camera_1 = pickle.load(f)
    camera_1 = _coerce_pickled_camera_payload(camera_1, camera_1_cam_path)
    if getattr(camera_1, "projection_matrix", None) is None:
        raise ValueError(
            f"Camera 1 calibration at {camera_1_cam_path} has no projection_matrix after load. "
            "For calibration_testing dict PKLs, keep extrinsic_pose_undistorted.json in the same folder."
        )
    print(f"[CorrelationWorker]    📷 Loaded camera 1 calibration from {camera_1_cam_path}")

    frame_trajectory_map: Dict[int, List] = {}
    trajectory_history: Dict[int, List] = {}
    frame_removed_trajectory_map: Dict[int, List] = {}
    _data_dir = tracker_base_dir if tracker_base_dir is not None else output_dir
    jsonl_path = os.path.join(_data_dir, "trajectory_selection.jsonl")
    selection_by_frame = _load_trajectory_selection_jsonl(jsonl_path, segment_ranges)
    no_jsonl_for_range = not selection_by_frame
    if not selection_by_frame:
        print(
            f"[CorrelationWorker]    ⚠️  trajectory_selection.jsonl missing or empty for range "
            f"[{f_min}, {f_max}]: {jsonl_path} — writing plain video (no trajectory overlays)"
        )

    frame_trajectory_map, trajectory_history, frame_removed_trajectory_map = _build_traj_maps_from_selection(
        selection_by_frame
    )
    all_frame_data, frame_costs_map = _build_frame_data_and_costs_from_selection(selection_by_frame)
    if selection_by_frame:
        print(
            f"[CorrelationWorker]    ✅ Loaded trajectory_selection.jsonl: {jsonl_path} "
            f"({len(selection_by_frame)} frame records, {len(trajectory_history)} trajectories)"
        )

    for traj_id in trajectory_history:
        trajectory_history[traj_id].sort(key=lambda x: x[0])

    render_frame_ids = list(range(f_min, f_max + 1))
    frame_id_set = set(render_frame_ids)
    frame_dict_cam1 = original_buffer.get_frames_for_sync_frames(frame_id_set)
    print(
        f"[CorrelationWorker]    🗺️  OriginalFrameBuffer JSONL viz: frame_id ∈ "
        f"[{render_frame_ids[0]}, {render_frame_ids[-1]}], "
        f"retrieved {len(frame_dict_cam1)}/{len(render_frame_ids)} cam1 frames from buffer"
    )
    fw, fh = original_buffer.get_original_frame_size()
    if fw is not None and fh is not None and int(fw) > 0 and int(fh) > 0:
        black_placeholder = np.zeros((int(fh), int(fw), 3), dtype=np.uint8)
    else:
        sample = next(iter(frame_dict_cam1.values()), None)
        if sample is not None:
            black_placeholder = np.zeros_like(sample)
        else:
            black_placeholder = np.zeros((1080, 1920, 3), dtype=np.uint8)

    total_raw_points = sum(len(coords_list) for _, coords_list in all_frame_data.values())
    print(
        f"[CorrelationWorker]    📊 JSONL keys with point payloads: {len(all_frame_data)}, "
        f"raw 3D positions: {total_raw_points}"
    )

    hud_w, overlay_height, hud_line_step = _left_hud_dimensions()
    left_hud_right_x = 10 + hud_w
    left_hud_bottom_y = 10 + overlay_height

    print(
        "[CorrelationWorker]    🔄 Reprojecting 3D coordinates to camera 1 frames "
        f"({'(continuous plain frames; no JSONL)' if no_jsonl_for_range else '(continuous frames + JSONL overlays)'})..."
    )

    # Helper function to get trail points for a trajectory up to current frame
    def get_trail_points(traj_id: int, current_frame: int, n: int) -> List[Tuple[float, float, float]]:
        """Get last n 3D points from trajectory history up to current_frame."""
        if traj_id not in trajectory_history:
            return []

        history = trajectory_history[traj_id]
        # Get points up to and including current frame
        relevant_points = [(f, coords) for f, coords in history if f <= current_frame]
        # Return last n points (coords only)
        return [coords for _, coords in relevant_points[-n:]]

    print(
        f"[CorrelationWorker]    📊 Total points in trajectories: "
        f"{sum(len(v) for v in frame_trajectory_map.values())}"
    )

    # Color for removed/filtered points (black with red outline)
    REMOVED_POINT_COLOR = (0, 0, 0)  # Black

    # Video FPS
    VIDEO_FPS = 30.0

    # Prepare frames with reprojected points
    frames_with_points: List[Tuple[int, np.ndarray]] = []

    use_trajectory_viz = len(frame_trajectory_map) > 0

    removed_points_count = 0
    trajectory_points_count = 0
    total_raw_points_drawn = 0

    for frame_idx in render_frame_ids:
        cam_stream_idx = frame_idx
        entry = all_frame_data.get(frame_idx)
        if entry is not None:
            _, coords_list = entry
        else:
            coords_list = []

        pref = frame_dict_cam1.get(frame_idx)
        if pref is None:
            frame_img = black_placeholder.copy()
        else:
            frame_img = pref.copy()

        frame_raw_points = len(coords_list) if coords_list else 0
        frame_traj_points = 0
        frame_removed_points = 0

        if frame_idx in frame_removed_trajectory_map:
            for coords in frame_removed_trajectory_map[frame_idx]:
                pt_2d = reproject_point(camera_1, coords)
                pt_int = (int(pt_2d[0]), int(pt_2d[1]))
                cv2.circle(frame_img, pt_int, 15, REMOVED_POINT_COLOR, -1)
                removed_points_count += 1
                frame_removed_points += 1

        total_raw_points_drawn += frame_raw_points

        if use_trajectory_viz and frame_idx in frame_trajectory_map:
            traj_points_in_frame = frame_trajectory_map[frame_idx]
            active_traj_ids = set(traj_id for traj_id, _ in traj_points_in_frame)

            for traj_id in active_traj_ids:
                trail_coords = get_trail_points(traj_id, frame_idx, trail_length)
                if len(trail_coords) >= 2:
                    color = get_trajectory_color(traj_id)
                    trail_2d = []
                    for coords in trail_coords:
                        pt_2d = reproject_point(camera_1, coords)
                        trail_2d.append((int(pt_2d[0]), int(pt_2d[1])))
                    for i in range(len(trail_2d) - 1):
                        fade = (i + 1) / len(trail_2d)
                        thickness = max(1, int(3 * fade))
                        cv2.line(frame_img, trail_2d[i], trail_2d[i + 1], color, thickness)

            for traj_id, coords in traj_points_in_frame:
                pt_2d = reproject_point(camera_1, coords)
                pt_int = (int(pt_2d[0]), int(pt_2d[1]))
                color = get_trajectory_color(traj_id)
                cv2.circle(frame_img, pt_int, 8, color, -1)
                cv2.circle(frame_img, pt_int, 10, (0, 0, 0), 2)
                if show_trajectory_labels:
                    label = f"T{traj_id}"
                    label_pos = (pt_int[0] + 18, pt_int[1] - 8)
                    (text_w, text_h), _ = cv2.getTextSize(
                        label, _VIZ_FONT, _VIZ_SCALE_LABEL, _VIZ_THICKNESS
                    )
                    cv2.rectangle(
                        frame_img,
                        (label_pos[0] - 4, label_pos[1] - text_h - 4),
                        (label_pos[0] + text_w + 4, label_pos[1] + 4),
                        (0, 0, 0),
                        -1,
                    )
                    cv2.putText(
                        frame_img,
                        label,
                        label_pos,
                        _VIZ_FONT,
                        _VIZ_SCALE_LABEL,
                        _VIZ_TEXT_COLOR,
                        _VIZ_THICKNESS,
                    )
                trajectory_points_count += 1
                frame_traj_points += 1

        elif coords_list:
            for coords in coords_list:
                pt_2d = reproject_point(camera_1, coords)
                pt_int = (int(pt_2d[0]), int(pt_2d[1]))
                cv2.circle(frame_img, pt_int, 5, (0, 255, 0), -1)
                cv2.circle(frame_img, pt_int, 8, (0, 255, 0), 2)

        overlay = frame_img.copy()
        cv2.rectangle(overlay, (10, 10), (10 + hud_w, 10 + overlay_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame_img, 0.45, 0, frame_img)

        y_text = 10 + hud_line_step
        hud_lines = [
            f"FPS: {VIDEO_FPS:.0f}  |  frame_id: {frame_idx}  |  camera_stream_index: {cam_stream_idx}",
            f"Raw 3D Points: {frame_raw_points}",
            f"Trajectory Points: {frame_traj_points}",
            f"Removed Points: {frame_removed_points}",
        ]
        for line in hud_lines:
            cv2.putText(
                frame_img,
                line,
                (20, y_text),
                _VIZ_FONT,
                _VIZ_SCALE_HUD,
                _VIZ_TEXT_COLOR,
                _VIZ_THICKNESS,
            )
            y_text += hud_line_step

        _draw_propagated_cost_panel(
            frame_img,
            frame_idx,
            frame_costs_map,
            left_hud_right_x,
            left_hud_bottom_y,
        )

        frames_with_points.append((frame_idx, frame_img))


    print(
        f"[CorrelationWorker]    📊 Visualization stats: {total_raw_points_drawn} total raw 3D points drawn ({trajectory_points_count} in trajectories, {removed_points_count} removed/filtered)"
    )

    if not frames_with_points:
        print("[CorrelationWorker]    ⚠️  No frames in visualization range (empty segment)")
        return

    # Create output video directory
    video_output_dir = os.path.join(output_dir, camera_1_id, "visualization_videos")
    os.makedirs(video_output_dir, exist_ok=True)

    # Video writer settings
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Segment frames into chunks for video writing
    chunks = []
    current_chunk = []

    for _, (frame_idx, frame_img) in enumerate(frames_with_points):
        current_chunk.append((frame_idx, frame_img))
        if len(current_chunk) >= video_chunk_size:
            chunks.append(current_chunk)
            current_chunk = []

    if current_chunk:
        chunks.append(current_chunk)

    total_frames_visualized = len(frames_with_points)
    num_chunks = len(chunks)

    print(f"[CorrelationWorker]    🎬 Writing {total_frames_visualized} frames into {num_chunks} video chunk(s)...")

    for chunk_idx, chunk in enumerate(chunks):
        if not chunk:
            continue

        start_frame_idx = chunk[0][0]
        end_frame_idx = chunk[-1][0]

        # Always use segment-based filename so multiple batches don't overwrite the same file
        video_path = os.path.join(
            video_output_dir,
            f"trajectory_visualization_{start_frame_idx}_to_{end_frame_idx}.mp4",
        )

        # Get frame size from first frame in chunk
        first_frame = chunk[0][1]
        height, width = first_frame.shape[:2]

        # Initialize video writer
        out = cv2.VideoWriter(video_path, fourcc, VIDEO_FPS, (width, height))

        # Write frames to video
        for frame_idx, frame_img in chunk:
            out.write(frame_img)

        out.release()
        print(f"[CorrelationWorker]    ✅ Video chunk {chunk_idx + 1}/{num_chunks} written: {video_path}")

    visualization_duration = time.time() - visualization_start_time
    print(
        f"[CorrelationWorker]    ✅ Visualization chunks written in {visualization_duration:.2f}s "
        f"({total_frames_visualized} frames, {num_chunks} chunk(s))"
    )
    # Per-batch timing is recorded in correlation_worker (correlation_segment_viz_overlay_s).
