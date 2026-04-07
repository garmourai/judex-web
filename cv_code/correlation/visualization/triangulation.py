"""Cam1 overlay MP4s from triangulated 3D points (reprojection onto staging-buffer frames)."""

import json
import os
import pickle
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2

from ...inference.inference import get_trajectory_color
from ..correlation_worker_utils import load_fid_to_stream_from_dist_tracker_csv
from .utils import reproject_point

# Overlay text: white on dark panels. Scales/layout are 25% of prior (75% smaller than last size).
_VIZ_TEXT_COLOR = (255, 255, 255)
_VIZ_FONT = cv2.FONT_HERSHEY_SIMPLEX
_VIZ_THICKNESS = 1
_VIZ_SCALE_HUD = 0.375
_VIZ_SCALE_LABEL = 0.375
_VIZ_SCALE_COST_TITLE = 0.4125
_VIZ_SCALE_COST_LINE = 0.3375
_VIZ_LINE_HEIGHT = 15


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


def create_visualization_from_triangulation(
    frame_segments: List[Tuple[int, int]],
    output_dir: str,
    staging_buffer_1,
    camera_1_cam_path: str,
    camera_1_id: str,
    video_chunk_size: int = 1000,
    staging_buffer_2=None,
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
        staging_buffer_1: StagingBuffer for camera 1 to get frames (or VideoFrameProvider for offline flow)
        camera_1_cam_path: Path to camera 1 calibration pickle file
        camera_1_id: Camera 1 identifier
        video_chunk_size: Number of frames per video chunk (default: 1000)
        staging_buffer_2: Optional StagingBuffer for camera 2 (to clear frames)
        profiler: Optional TimeProfiler instance for recording timing metrics
        trajectory_data: Dict mapping segment tuple to list of trajectories (kept trajectories)
        removed_trajectory_data: Dict mapping segment tuple to list of removed trajectories (drawn as big black circles)
        trail_length: Number of past frames to show in trajectory trail (default: 10)
        show_trajectory_labels: Whether to show trajectory ID text labels (default: True)
        tracker_base_dir: If set, read trajectory_selection.jsonl from this directory (e.g. correlation output read path)
        camera_1_csv_path: Path to camera 1 dist_tracker.csv (StagingBuffer frame→stream index mapping; not used with OriginalFrameBuffer)
    """
    visualization_start_time = time.time()
    # Load camera calibration
    with open(camera_1_cam_path, "rb") as f:
        camera_1 = pickle.load(f)
    print(f"[CorrelationWorker]    📷 Loaded camera 1 calibration from {camera_1_cam_path}")

    frame_trajectory_map: Dict[int, List] = {}
    trajectory_history: Dict[int, List] = {}
    frame_removed_trajectory_map: Dict[int, List] = {}
    _data_dir = tracker_base_dir if tracker_base_dir is not None else output_dir
    frame_ranges = frame_segments or []

    jsonl_path = os.path.join(_data_dir, "trajectory_selection.jsonl")
    selection_by_frame = _load_trajectory_selection_jsonl(jsonl_path, frame_ranges)
    if not selection_by_frame:
        print(
            f"[CorrelationWorker]    ❌ trajectory_selection.jsonl missing or empty (required for visualization): "
            f"{jsonl_path}"
        )
        return

    frame_trajectory_map, trajectory_history, frame_removed_trajectory_map = _build_traj_maps_from_selection(
        selection_by_frame
    )
    all_frame_data, frame_costs_map = _build_frame_data_and_costs_from_selection(selection_by_frame)
    print(
        f"[CorrelationWorker]    ✅ Loaded trajectory_selection.jsonl: {jsonl_path} "
        f"({len(selection_by_frame)} frame records, {len(trajectory_history)} trajectories)"
    )

    for traj_id in trajectory_history:
        trajectory_history[traj_id].sort(key=lambda x: x[0])

    # frame_id -> per-camera stream index (StagingBuffer / overlay labels)
    from ...triplet_csv_reader import OriginalFrameBuffer as _OriginalFrameBuffer

    _use_original_buffer = isinstance(staging_buffer_1, _OriginalFrameBuffer)

    if _use_original_buffer:
        frame_id_to_cam_stream = {fn: fn for fn in all_frame_data.keys()}
        print(
            "[CorrelationWorker]    🗺️  OriginalFrameBuffer: using frame_id as camera_stream_index "
            "(no dist_tracker mapping needed)"
        )
    else:
        if not camera_1_csv_path:
            print(
                "[CorrelationWorker]    ❌ camera_1_csv_path (dist_tracker.csv) required for "
                "non–OriginalFrameBuffer visualization"
            )
            return
        frame_id_to_cam_stream = load_fid_to_stream_from_dist_tracker_csv(camera_1_csv_path)
        if len(frame_id_to_cam_stream) == 0:
            print(
                f"[CorrelationWorker]    ❌ No frame rows in dist_tracker for visualization: {camera_1_csv_path}"
            )
            return
        print(
            f"[CorrelationWorker]    🗺️  Loaded {len(frame_id_to_cam_stream)} frame_id→camera_stream_index "
            f"from dist_tracker"
        )

    # Filter to only frames that have a valid mapping entry
    original_frame_count = len(all_frame_data)
    all_frame_data = {
        fnwd: value for fnwd, value in all_frame_data.items() if fnwd in frame_id_to_cam_stream
    }
    frame_costs_map = {k: frame_costs_map[k] for k in all_frame_data.keys() if k in frame_costs_map}
    if original_frame_count != len(all_frame_data):
        print(
            f"[CorrelationWorker]    ℹ️  Filtered frames without mapping: "
            f"{original_frame_count} → {len(all_frame_data)} usable frames for visualization"
        )

    # Count total raw points
    total_raw_points = sum(len(coords_list) for _, coords_list in all_frame_data.values())
    print(
        f"[CorrelationWorker]    📊 Loaded {total_raw_points} raw 3D point positions from trajectory_selection.jsonl"
    )

    if len(all_frame_data) == 0:
        print(
            f"[CorrelationWorker]    ⚠️  No frame data after applying frame_id→camera_stream mapping "
            f"(check trajectory_selection.jsonl vs dist_tracker)"
        )
        return

    global_frame_indices = set(all_frame_data.keys())

    # --- judex pipeline: OriginalFrameBuffer direct lookup (no mapping file needed) ---
    if _use_original_buffer:
        # source_index == global frame_id — direct dict lookup, no mapping file
        frame_dict_cam1 = staging_buffer_1.get_frames_for_sync_frames(global_frame_indices)
        print(f"[CorrelationWorker]    📦 OriginalFrameBuffer: retrieved {len(frame_dict_cam1)} cam1 frames directly")
        for fnwd in global_frame_indices:
            if fnwd not in all_frame_data:
                continue
            frame = frame_dict_cam1.get(fnwd)
            if frame is not None:
                _, coords_list = all_frame_data[fnwd]
                all_frame_data[fnwd] = (frame, coords_list)
    else:
        # --- Original StagingBuffer path (mapping file required) ---
        frame_indices_cam1: set[int] = set()
        for fnwd in global_frame_indices:
            frame_indices_cam1.add(frame_id_to_cam_stream[fnwd])

        print(
            f"[CorrelationWorker]    📦 Peeking at {len(frame_indices_cam1)} frames from staging buffer for visualization..."
        )
        frames_from_buffer = staging_buffer_1.peek_frames_by_indices(frame_indices_cam1)
        print(f"[CorrelationWorker]    ✅ Retrieved {len(frames_from_buffer)} frames from camera 1 staging buffer")

        frame_dict_cam1 = {
            frame_data.camera_stream_index: frame_data.frame for frame_data in frames_from_buffer
        }

        for fnwd in global_frame_indices:
            if fnwd not in all_frame_data:
                continue
            cam_stream_idx = frame_id_to_cam_stream.get(fnwd)
            if cam_stream_idx is None:
                continue
            if cam_stream_idx in frame_dict_cam1:
                _, coords_list = all_frame_data[fnwd]
                all_frame_data[fnwd] = (frame_dict_cam1[cam_stream_idx], coords_list)

    # Reproject 3D coordinates to camera 1 frames
    print(f"[CorrelationWorker]    🔄 Reprojecting 3D coordinates to camera 1 frames...")

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
    REMOVED_POINT_OUTLINE_COLOR = (0, 0, 255)  # Red (BGR format)

    # Video FPS
    VIDEO_FPS = 30.0

    # Prepare frames with reprojected points
    frames_with_points = []
    sorted_frame_indices = sorted(all_frame_data.keys())

    use_trajectory_viz = len(frame_trajectory_map) > 0

    removed_points_count = 0
    trajectory_points_count = 0
    total_raw_points_drawn = 0

    for frame_idx in sorted_frame_indices:
        frame_img, coords_list = all_frame_data[frame_idx]
        cam_stream_idx = frame_id_to_cam_stream.get(frame_idx)
        if cam_stream_idx is None:
            continue

        if frame_img is None:
            continue

        # Make a copy to avoid modifying original
        frame_img = frame_img.copy()

        # Count raw points in this frame
        frame_raw_points = len(coords_list) if coords_list else 0
        frame_traj_points = 0
        frame_removed_points = 0

        # ===== Step 0: Draw REMOVED TRAJECTORY points first (so they appear behind trajectory points) =====
        # Draw removed trajectories from get_best_point_each_frame as big black circles
        if frame_idx in frame_removed_trajectory_map:
            for coords in frame_removed_trajectory_map[frame_idx]:
                pt_2d = reproject_point(camera_1, coords)
                pt_int = (int(pt_2d[0]), int(pt_2d[1]))

                # Draw big black circle for removed trajectory points
                cv2.circle(frame_img, pt_int, 15, REMOVED_POINT_COLOR, -1)  # Big black filled circle

                removed_points_count += 1
                frame_removed_points += 1

        total_raw_points_drawn += frame_raw_points

        if use_trajectory_viz and frame_idx in frame_trajectory_map:
            # ===== TRAJECTORY-AWARE VISUALIZATION =====
            traj_points_in_frame = frame_trajectory_map[frame_idx]

            # Collect all trajectory IDs active in this frame
            active_traj_ids = set(traj_id for traj_id, _ in traj_points_in_frame)

            # Step 1: Draw trajectory trails (lines connecting past points)
            for traj_id in active_traj_ids:
                trail_coords = get_trail_points(traj_id, frame_idx, trail_length)
                if len(trail_coords) >= 2:
                    color = get_trajectory_color(traj_id)

                    # Reproject all trail points
                    trail_2d = []
                    for coords in trail_coords:
                        pt_2d = reproject_point(camera_1, coords)
                        trail_2d.append((int(pt_2d[0]), int(pt_2d[1])))

                    # Draw lines connecting trail points with fading opacity
                    for i in range(len(trail_2d) - 1):
                        # Fade factor: older points are more transparent
                        fade = (i + 1) / len(trail_2d)
                        # Use thinner lines for older parts of trail
                        thickness = max(1, int(3 * fade))
                        cv2.line(frame_img, trail_2d[i], trail_2d[i + 1], color, thickness)

            # Step 2: Draw current detection points on top
            for traj_id, coords in traj_points_in_frame:
                pt_2d = reproject_point(camera_1, coords)
                pt_int = (int(pt_2d[0]), int(pt_2d[1]))
                color = get_trajectory_color(traj_id)

                # Draw filled circle with black outline
                cv2.circle(frame_img, pt_int, 8, color, -1)  # Filled circle
                cv2.circle(frame_img, pt_int, 10, (0, 0, 0), 2)  # Black outline

                # Step 3: Draw trajectory ID label (if enabled)
                if show_trajectory_labels:
                    label = f"T{traj_id}"
                    label_pos = (pt_int[0] + 14, pt_int[1] - 6)
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
            # ===== FALLBACK: Simple green point visualization (no trajectory data) =====
            for coords in coords_list:
                pt_2d = reproject_point(camera_1, coords)
                pt_int = (int(pt_2d[0]), int(pt_2d[1]))
                cv2.circle(frame_img, pt_int, 5, (0, 255, 0), -1)  # Green filled circle
                cv2.circle(frame_img, pt_int, 8, (0, 255, 0), 2)  # Green outline

        # ===== Draw FPS and frame info overlay =====
        # Dark semi-transparent patch so white HUD text stays readable
        overlay_height = 78
        overlay = frame_img.copy()
        hud_w = 400
        cv2.rectangle(overlay, (10, 10), (10 + hud_w, 10 + overlay_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame_img, 0.45, 0, frame_img)

        y_offset = 23
        line_height = _VIZ_LINE_HEIGHT

        # Line 1: FPS and Frame number
        cv2.putText(
            frame_img,
            f"FPS: {VIDEO_FPS:.0f}  |  frame_id: {frame_idx}  |  camera_stream_index: {cam_stream_idx}",
            (20, y_offset),
            _VIZ_FONT,
            _VIZ_SCALE_HUD,
            _VIZ_TEXT_COLOR,
            _VIZ_THICKNESS,
        )
        y_offset += line_height

        # Line 2: Raw points in this frame
        cv2.putText(
            frame_img,
            f"Raw 3D Points: {frame_raw_points}",
            (20, y_offset),
            _VIZ_FONT,
            _VIZ_SCALE_HUD,
            _VIZ_TEXT_COLOR,
            _VIZ_THICKNESS,
        )
        y_offset += line_height

        # Line 3: Trajectory points
        cv2.putText(
            frame_img,
            f"Trajectory Points: {frame_traj_points}",
            (20, y_offset),
            _VIZ_FONT,
            _VIZ_SCALE_HUD,
            _VIZ_TEXT_COLOR,
            _VIZ_THICKNESS,
        )
        y_offset += line_height

        # Line 4: Removed points
        cv2.putText(
            frame_img,
            f"Removed Points: {frame_removed_points}",
            (20, y_offset),
            _VIZ_FONT,
            _VIZ_SCALE_HUD,
            _VIZ_TEXT_COLOR,
            _VIZ_THICKNESS,
        )

        # ===== Draw propagated cost summary (top-right) =====
        cost_info = frame_costs_map.get(frame_idx, {"f1": [], "f2": [], "epi": [], "reproj": [], "temp": []})
        cost_lines = []
        for key, label in (("f1", "F1"), ("f2", "F2"), ("epi", "Epi"), ("reproj", "Rho"), ("temp", "Temp")):
            vals = cost_info.get(key, []) or []
            if vals:
                shown = ", ".join(f"{float(v):.2f}" for v in vals[:3])
                suffix = "..." if len(vals) > 3 else ""
                cost_lines.append(f"{label}: [{shown}{suffix}]")
            else:
                cost_lines.append(f"{label}: []")
        right_pad = 15
        box_w = 280
        cost_line_spacing = 5
        box_h = 6 + 5 * cost_line_spacing
        x2 = frame_img.shape[1] - right_pad
        x1 = max(0, x2 - box_w)
        y1 = 10
        y2 = y1 + box_h
        overlay_r = frame_img.copy()
        cv2.rectangle(overlay_r, (x1, y1), (x2, y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay_r, 0.55, frame_img, 0.45, 0, frame_img)
        cv2.putText(
            frame_img,
            "Propagated Costs",
            (x1 + 3, y1 + 6),
            _VIZ_FONT,
            _VIZ_SCALE_COST_TITLE,
            _VIZ_TEXT_COLOR,
            _VIZ_THICKNESS,
        )
        for i, line in enumerate(cost_lines):
            cv2.putText(
                frame_img,
                line,
                (x1 + 3, y1 + 6 + cost_line_spacing * (i + 1)),
                _VIZ_FONT,
                _VIZ_SCALE_COST_LINE,
                _VIZ_TEXT_COLOR,
                _VIZ_THICKNESS,
            )

        frames_with_points.append((frame_idx, frame_img))

    print(
        f"[CorrelationWorker]    📊 Visualization stats: {total_raw_points_drawn} total raw 3D points drawn ({trajectory_points_count} in trajectories, {removed_points_count} removed/filtered)"
    )

    if len(frames_with_points) == 0:
        print(f"[CorrelationWorker]    ⚠️  No frames available for visualization")
        return

    # Sort frames for video writing
    frames_with_points.sort(key=lambda x: x[0])

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
