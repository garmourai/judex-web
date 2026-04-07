"""Cam1 overlay MP4s from triangulated 3D points (reprojection onto staging-buffer frames)."""

import os
import pickle
import time
import csv
from typing import Dict, List, Optional, Tuple

import cv2

from ...inference.inference import get_trajectory_color
from ..correlation_worker_utils import load_fid_to_stream_from_dist_tracker_csv
from .utils import reproject_point

# Overlay text: white on dark panels; scales are 3× prior defaults (0.5→1.5, etc.)
_VIZ_TEXT_COLOR = (255, 255, 255)
_VIZ_FONT = cv2.FONT_HERSHEY_SIMPLEX
_VIZ_THICKNESS = 3
_VIZ_SCALE_HUD = 1.5
_VIZ_SCALE_LABEL = 1.5
_VIZ_SCALE_COST_TITLE = 1.65
_VIZ_SCALE_COST_LINE = 1.35
_VIZ_LINE_HEIGHT = 60


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
        output_dir: Output directory for videos; also used for tracker.csv if tracker_base_dir is not set
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
        tracker_base_dir: If set, read tracker.csv files from this directory (e.g. read dir when using separate write dir)
        camera_1_csv_path: Path to camera 1 dist_tracker.csv (for legacy StagingBuffer frame→stream lookup)
    """
    visualization_start_time = time.time()
    # Load camera calibration
    with open(camera_1_cam_path, "rb") as f:
        camera_1 = pickle.load(f)
    print(f"[CorrelationWorker]    📷 Loaded camera 1 calibration from {camera_1_cam_path}")

    # Build frame-to-trajectory mapping from persisted mapping CSV
    # frame_trajectory_map: frame_idx -> [(traj_id, (x, y, z)), ...]
    frame_trajectory_map = {}
    # trajectory_history: traj_id -> [(frame_num, (x, y, z)), ...] sorted by frame
    trajectory_history = {}
    frame_removed_trajectory_map = {}

    # Collect all frames/coords/trajectory IDs and costs from mapping CSV
    all_frame_data = {}  # frame_id -> (frame_image, [list of 3d_coords])
    frame_costs_map = {}  # frame_id -> cost lists from persisted mapping columns
    _tracker_dir = tracker_base_dir if tracker_base_dir is not None else output_dir

    mapping_csv_path = os.path.join(_tracker_dir, "points_trajectory_mapping.csv")
    if not os.path.exists(mapping_csv_path):
        print(f"[CorrelationWorker]    ⚠️  Mapping CSV not found: {mapping_csv_path}")
    else:
        frame_ranges = frame_segments or []
        with open(mapping_csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    frame_num = int(row["Frame"])
                    if frame_ranges and not any(s <= frame_num <= e for s, e in frame_ranges):
                        continue
                    x_3d = float(row["X"])
                    y_3d = float(row["Y"])
                    z_3d = float(row["Z"])
                    traj_id = int(float(str(row["Trajectory_ID"]).strip()))
                    kept_cell = str(row["Kept_After_Filter"]).strip()
                except (ValueError, KeyError, TypeError):
                    continue
                coords = (x_3d, y_3d, z_3d)

                if frame_num not in all_frame_data:
                    all_frame_data[frame_num] = (None, [])
                _, coords_list = all_frame_data[frame_num]
                coords_list.append(coords)

                # points_trajectory_mapping.csv: Trajectory_ID -1 = unmatched; Kept_After_Filter 1/0 for matched
                if traj_id < 0 or kept_cell == "0":
                    frame_removed_trajectory_map.setdefault(frame_num, []).append(coords)
                elif kept_cell == "1":
                    frame_trajectory_map.setdefault(frame_num, []).append((traj_id, coords))
                    trajectory_history.setdefault(traj_id, []).append((frame_num, coords))
                else:
                    continue

                frame_costs_map.setdefault(
                    frame_num, {"f1": [], "f2": [], "epi": [], "reproj": [], "temp": []}
                )
                for col, key in (
                    ("Cost_Formula1", "f1"),
                    ("Cost_Formula2", "f2"),
                    ("Cost_Epipolar", "epi"),
                    ("Cost_Reprojection", "reproj"),
                    ("Cost_Temporal", "temp"),
                ):
                    raw = row.get(col, "")
                    if raw is None or str(raw).strip() == "":
                        continue
                    try:
                        frame_costs_map[frame_num][key].append(float(raw))
                    except ValueError:
                        continue

    for traj_id in trajectory_history:
        trajectory_history[traj_id].sort(key=lambda x: x[0])
    print(
        f"[CorrelationWorker]    ✅ Built trajectory map from mapping CSV: "
        f"{len(trajectory_history)} trajectories"
    )

    # frame_id -> per-camera stream index (legacy StagingBuffer / overlay labels)
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
    if original_frame_count != len(all_frame_data):
        print(
            f"[CorrelationWorker]    ℹ️  Filtered frames without mapping: "
            f"{original_frame_count} → {len(all_frame_data)} usable frames for visualization"
        )

    # Count total raw points
    total_raw_points = sum(len(coords_list) for _, coords_list in all_frame_data.values())
    print(
        f"[CorrelationWorker]    📊 Loaded {total_raw_points} raw 3D points from points_trajectory_mapping.csv (after mapping filter)"
    )

    if len(all_frame_data) == 0:
        print(f"[CorrelationWorker]    ⚠️  No frame data found in tracker CSVs")
        return

    # Get per-camera frame indices (using mapping) and take frames from staging buffers
    if len(all_frame_data) == 0:
        print(f"[CorrelationWorker]    ⚠️  No frame data found in tracker CSVs after applying mapping")
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
        overlay_height = 310
        overlay = frame_img.copy()
        hud_w = 900
        cv2.rectangle(overlay, (10, 10), (10 + hud_w, 10 + overlay_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame_img, 0.45, 0, frame_img)

        y_offset = 90
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
        box_w = 432
        cost_line_spacing = 18
        box_h = 24 + 5 * cost_line_spacing
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
            (x1 + 12, y1 + 24),
            _VIZ_FONT,
            _VIZ_SCALE_COST_TITLE,
            _VIZ_TEXT_COLOR,
            _VIZ_THICKNESS,
        )
        for i, line in enumerate(cost_lines):
            cv2.putText(
                frame_img,
                line,
                (x1 + 12, y1 + 24 + cost_line_spacing * (i + 1)),
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
