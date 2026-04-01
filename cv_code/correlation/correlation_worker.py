import os
import time
import threading
import pickle
import re
import csv
from typing import Optional, Tuple, List, Dict
from ast import literal_eval

import cv2

from ..inference.inference import get_trajectory_color
from ..trajectory.trajectory_realtime import create_trajectories_realtime
from ..trajectory.merge_traj_realtime import (
    merge_trajectories,
    merge_overlapping_trajectories,
)
from ..trajectory.trajectory_filter_realtime import get_best_point_each_frame
from .stitched_correlation_visualization import (
    append_stitched_segment_to_video,
    load_correlated_pairs_from_tracker_csvs_realtime,
)
from .correlation_worker_utils import load_fid_to_stream_from_dist_tracker_csv

_trajectory_frame_size_fallback_logged = False


def _resolve_trajectory_original_frame_size(
    staging_buffer_1,
    explicit_width: Optional[int],
    explicit_height: Optional[int],
) -> Tuple[int, int]:
    """
    Prefer dimensions from OriginalFrameBuffer (set when the reader decodes the first frame),
    then optional PipelineConfig overrides, else 1920x1080 (logged once).
    """
    global _trajectory_frame_size_fallback_logged
    if staging_buffer_1 is not None:
        from ..triplet_csv_reader import OriginalFrameBuffer as _OFB

        if isinstance(staging_buffer_1, _OFB):
            w, h = staging_buffer_1.get_original_frame_size()
            if w is not None and h is not None:
                return w, h
    if explicit_width is not None and explicit_height is not None:
        return explicit_width, explicit_height
    if not _trajectory_frame_size_fallback_logged:
        print(
            "[CorrelationWorker]   Trajectory frame size: using default 1920x1080 "
            "(set PipelineConfig.camera_1_original_frame_width/height or wait for first decoded frame)",
            flush=True,
        )
        _trajectory_frame_size_fallback_logged = True
    return 1920, 1080


def _create_visualization_from_triangulation(
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
    with open(camera_1_cam_path, 'rb') as f:
        camera_1 = pickle.load(f)
    print(f"[CorrelationWorker]    📷 Loaded camera 1 calibration from {camera_1_cam_path}")

    # Build frame-to-trajectory mapping if trajectory data is provided
    # frame_trajectory_map: frame_idx -> [(traj_id, (x, y, z)), ...]
    frame_trajectory_map = {}
    # trajectory_history: traj_id -> [(frame_num, (x, y, z)), ...] sorted by frame
    trajectory_history = {}
    
    if trajectory_data is not None:
        print(f"[CorrelationWorker]    🎯 Building trajectory visualization data...")
        global_traj_offset = 0  # Offset trajectory IDs across segments
        
        for segment, trajectories in trajectory_data.items():
            if trajectories is None:
                continue
            
            for local_traj_id, trajectory in enumerate(trajectories):
                global_traj_id = global_traj_offset + local_traj_id
                
                # Initialize history for this trajectory
                if global_traj_id not in trajectory_history:
                    trajectory_history[global_traj_id] = []
                
                # Add all detections from this trajectory
                for frame_num, detection in trajectory.detections.items():
                    coords = (detection.x, detection.y, detection.z)
                    
                    # Add to frame map
                    if frame_num not in frame_trajectory_map:
                        frame_trajectory_map[frame_num] = []
                    frame_trajectory_map[frame_num].append((global_traj_id, coords))
                    
                    # Add to history
                    trajectory_history[global_traj_id].append((frame_num, coords))
            
            # Update offset for next segment
            global_traj_offset += len(trajectories) if trajectories else 0
        
        # Sort trajectory histories by frame number
        for traj_id in trajectory_history:
            trajectory_history[traj_id].sort(key=lambda x: x[0])
        
        total_traj = len(trajectory_history)
        total_points = sum(len(h) for h in trajectory_history.values())
        print(f"[CorrelationWorker]    ✅ Built trajectory map: {total_traj} trajectories, {total_points} total points")
    
    # Build frame-to-removed-trajectory mapping if removed trajectory data is provided
    # frame_removed_trajectory_map: frame_idx -> [(coords), ...] (no traj_id needed, just points)
    frame_removed_trajectory_map = {}
    
    if removed_trajectory_data is not None:
        print(f"[CorrelationWorker]    🎯 Building removed trajectory visualization data...")
        removed_points_count = 0
        
        for segment, removed_trajectories in removed_trajectory_data.items():
            if removed_trajectories is None:
                continue
            
            for trajectory in removed_trajectories:
                # Add all detections from removed trajectory
                for frame_num, detection in trajectory.detections.items():
                    coords = (detection.x, detection.y, detection.z)
                    
                    # Add to frame map (no traj_id needed, just coordinates)
                    if frame_num not in frame_removed_trajectory_map:
                        frame_removed_trajectory_map[frame_num] = []
                    frame_removed_trajectory_map[frame_num].append(coords)
                    removed_points_count += 1
        
        print(f"[CorrelationWorker]    ✅ Built removed trajectory map: {removed_points_count} points from {sum(len(trajs) for trajs in removed_trajectory_data.values() if trajs)} trajectories")
    
    # Collect all frames and ALL 3D coordinates from all segments
    # Keyed by global frame id
    all_frame_data = {}  # frame_id -> (frame_image, [list of 3d_coords])
    _tracker_dir = tracker_base_dir if tracker_base_dir is not None else output_dir

    for segment in frame_segments:
        start_frame, end_frame = segment
        segment_output = os.path.join(_tracker_dir, 'perform_correlation_output', f'{start_frame}_to_{end_frame}', 'tracker.csv')
        
        if not os.path.exists(segment_output):
            print(f"[CorrelationWorker]    ⚠️  Tracker CSV not found: {segment_output}")
            continue
        
        # Read tracker.csv to get frame indices and 3D coordinates
        # Format: Frame, Visibility, X, Y, Z, Original_Frame, Point_Coordinates_Camera1, Point_Coordinates_Camera2
        with open(segment_output, 'r') as f:
            lines = f.readlines()
        if len(lines) <= 1:  # Only header or empty
            continue

        for line in lines[1:]:  # Skip header
            parts = line.strip().split(',')
            if len(parts) < 5:
                continue

            frame_num = int(parts[0])
            visibility = int(parts[1])

            # Parse X, Y, Z coordinates (they are in format "[x1, x2, ...]")
            x_str = parts[2].strip()
            y_str = parts[3].strip()
            z_str = parts[4].strip()

            # Extract ALL numbers from string like "[1.234, 2.345, 3.456]"
            x_match = re.findall(r'[-+]?\d*\.?\d+', x_str)
            y_match = re.findall(r'[-+]?\d*\.?\d+', y_str)
            z_match = re.findall(r'[-+]?\d*\.?\d+', z_str)

            # Initialize frame entry if not exists
            if frame_num not in all_frame_data:
                all_frame_data[frame_num] = (None, [])

            # Extract ALL 3D coordinates (there can be multiple per frame)
            if visibility == 1 and len(x_match) > 0 and len(y_match) > 0 and len(z_match) > 0:
                num_points = min(len(x_match), len(y_match), len(z_match))
                for i in range(num_points):
                    x_3d = float(x_match[i])
                    y_3d = float(y_match[i])
                    z_3d = float(z_match[i])
                    # Add to list of coordinates for this frame
                    _, coords_list = all_frame_data[frame_num]
                    coords_list.append((x_3d, y_3d, z_3d))

    # frame_id -> per-camera stream index (legacy StagingBuffer / overlay labels)
    from ..triplet_csv_reader import OriginalFrameBuffer as _OriginalFrameBuffer

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
    print(f"[CorrelationWorker]    📊 Loaded {total_raw_points} raw 3D points from tracker.csv (after mapping filter)")
    
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

        print(f"[CorrelationWorker]    📦 Peeking at {len(frame_indices_cam1)} frames from staging buffer for visualization...")
        frames_from_buffer = staging_buffer_1.peek_frames_by_indices(frame_indices_cam1)
        print(f"[CorrelationWorker]    ✅ Retrieved {len(frames_from_buffer)} frames from camera 1 staging buffer")

        frame_dict_cam1 = {
            frame_data.camera_stream_index: frame_data.frame
            for frame_data in frames_from_buffer
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
    
    # Import reproject function (realtime package copy)
    from .visualisation import reproject_point
    
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
    
    # Build set of trajectory points for comparison (to identify removed points)
    trajectory_points_set = set()  # (frame, rounded_x, rounded_y, rounded_z)
    if trajectory_data is not None:
        for segment, trajectories in trajectory_data.items():
            if trajectories is None:
                continue
            for trajectory in trajectories:
                for frame_num, detection in trajectory.detections.items():
                    # Round to avoid floating point comparison issues
                    key = (frame_num, round(detection.x, 4), round(detection.y, 4), round(detection.z, 4))
                    trajectory_points_set.add(key)
    
    print(f"[CorrelationWorker]    📊 Total points in trajectories: {len(trajectory_points_set)}")
    
    # Color for removed/filtered points (black with red outline)
    REMOVED_POINT_COLOR = (0, 0, 0)  # Black
    REMOVED_POINT_OUTLINE_COLOR = (0, 0, 255)  # Red (BGR format)
    
    # Video FPS
    VIDEO_FPS = 30.0
    
    # Prepare frames with reprojected points
    frames_with_points = []
    sorted_frame_indices = sorted(all_frame_data.keys())
    
    use_trajectory_viz = trajectory_data is not None and len(frame_trajectory_map) > 0
    
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
        
        # Get frame dimensions for overlay positioning
        frame_height, frame_width = frame_img.shape[:2]
        
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
        
        # ===== Step 0.5: Draw REMOVED raw points (fallback if removed_trajectory_data not provided) =====
        # This is for backward compatibility - if removed_trajectory_data is not provided,
        # fall back to comparing raw points with trajectory points
        if removed_trajectory_data is None and coords_list:
            for coords in coords_list:
                total_raw_points_drawn += 1
                
                # Check if this point is in a trajectory
                key = (frame_idx, round(coords[0], 4), round(coords[1], 4), round(coords[2], 4))
                
                if key not in trajectory_points_set:
                    # This is a removed/filtered point - draw it as big black dot with red outline
                    pt_2d = reproject_point(camera_1, coords)
                    pt_int = (int(pt_2d[0]), int(pt_2d[1]))

                    # Draw big black circle with red outline for removed points
                    cv2.circle(frame_img, pt_int, 10, REMOVED_POINT_COLOR, -1)  # Black filled (big dot)
                    cv2.circle(frame_img, pt_int, 12, REMOVED_POINT_OUTLINE_COLOR, 3)  # Red outline (thick)

                    removed_points_count += 1
                    frame_removed_points += 1
        
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
                    label_pos = (pt_int[0] + 12, pt_int[1] - 5)
                    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame_img, 
                                 (label_pos[0] - 2, label_pos[1] - text_h - 2),
                                 (label_pos[0] + text_w + 2, label_pos[1] + 2),
                                 (255, 255, 255), -1)
                    cv2.putText(frame_img, label, label_pos, 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

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
        # Light semi-transparent patch so black HUD text stays readable
        overlay_height = 90
        overlay = frame_img.copy()
        cv2.rectangle(overlay, (10, 10), (280, 10 + overlay_height), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.55, frame_img, 0.45, 0, frame_img)
        
        text_color = (0, 0, 0)
        traj_color = (0, 0, 0)
        removed_color = (0, 0, 0)
        
        y_offset = 30
        line_height = 20
        
        # Line 1: FPS and Frame number
        cv2.putText(
            frame_img,
            f"FPS: {VIDEO_FPS:.0f}  |  frame_id: {frame_idx}  |  camera_stream_index: {cam_stream_idx}",
            (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            text_color,
            1,
        )
        y_offset += line_height
        
        # Line 2: Raw points in this frame
        cv2.putText(frame_img, f"Raw 3D Points: {frame_raw_points}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
        y_offset += line_height
        
        # Line 3: Trajectory points
        cv2.putText(frame_img, f"Trajectory Points: {frame_traj_points}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, traj_color, 1)
        y_offset += line_height
        
        # Line 4: Removed points
        cv2.putText(frame_img, f"Removed Points: {frame_removed_points}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, removed_color, 1)
        
        frames_with_points.append((frame_idx, frame_img))
    
    print(f"[CorrelationWorker]    📊 Visualization stats: {total_raw_points_drawn} total raw 3D points drawn ({trajectory_points_count} in trajectories, {removed_points_count} removed/filtered)")
    
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
    
    for idx, (frame_idx, frame_img) in enumerate(frames_with_points):
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


def _concatenate_visualization_videos(video_output_dir: str) -> Optional[str]:
    """
    Concatenate all segment visualization videos (trajectory_visualization_*_to_*.mp4)
    into a single trajectory_visualization.mp4 in frame order.
    Call this when the correlation worker has finished all segments.
    """
    import glob
    pattern = os.path.join(video_output_dir, "trajectory_visualization_*_to_*.mp4")
    segment_files = glob.glob(pattern)
    if not segment_files:
        return None

    # Parse start frame from filename to sort (e.g. trajectory_visualization_0_to_200.mp4 -> 0)
    def start_frame_from_path(path: str) -> int:
        basename = os.path.basename(path)
        # trajectory_visualization_0_to_200.mp4
        match = re.match(r"trajectory_visualization_(\d+)_to_\d+\.mp4", basename)
        return int(match.group(1)) if match else 0

    segment_files.sort(key=start_frame_from_path)
    final_path = os.path.join(video_output_dir, "trajectory_visualization.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = None
    total_frames = 0
    for seg_path in segment_files:
        cap = cv2.VideoCapture(seg_path)
        if not cap.isOpened():
            print(f"[CorrelationWorker]    ⚠️  Could not open segment video: {seg_path}")
            cap.release()
            continue
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if writer is None:
            writer = cv2.VideoWriter(final_path, fourcc, fps, (width, height))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
            total_frames += 1
        cap.release()
    if writer is not None:
        writer.release()
        print(f"[CorrelationWorker]    ✅ Concatenated {len(segment_files)} segment video(s) into {final_path} ({total_frames} frames)")
        return final_path
    return None


def _create_point_trajectory_mapping_csv(
    segment: Tuple[int, int],
    output_dir: str,
    stored_trajectories: List,
    global_traj_offset: int = 0,
) -> Optional[str]:
    """
    Create a CSV file mapping each 3D point from triangulation to its trajectory ID.
    
    Args:
        segment: Tuple of (start_frame, end_frame)
        output_dir: Output directory where tracker.csv files are stored
        stored_trajectories: List of merged trajectories for this segment
        global_traj_offset: Offset to add to trajectory IDs (for multi-segment continuity)
    
    Returns:
        Path to the created CSV file
    """
    start_frame, end_frame = segment
    
    # Path to triangulated 3D points
    tracker_csv_path = os.path.join(
        output_dir, "perform_correlation_output", 
        f"{start_frame}_to_{end_frame}", "tracker.csv"
    )
    
    if not os.path.exists(tracker_csv_path):
        print(f"[CorrelationWorker]    ⚠️  Tracker CSV not found: {tracker_csv_path}")
        return None
    
    # Build a mapping from (frame, x, y, z) to trajectory ID
    # Use rounded coordinates for matching (tolerance for floating point)
    trajectory_point_map = {}  # (frame, rounded_x, rounded_y, rounded_z) -> traj_id
    duplicate_keys = []  # Track duplicate keys (shouldn't happen after merging)
    
    for local_traj_id, trajectory in enumerate(stored_trajectories):
        global_traj_id = global_traj_offset + local_traj_id
        for frame_num, detection in trajectory.detections.items():
            # Round to 4 decimal places for matching
            key = (frame_num, round(detection.x, 4), round(detection.y, 4), round(detection.z, 4))
            if key in trajectory_point_map:
                duplicate_keys.append((key, trajectory_point_map[key], global_traj_id))
            trajectory_point_map[key] = global_traj_id
    
    if duplicate_keys:
        print(f"[CorrelationWorker]    ⚠️  Warning: Found {len(duplicate_keys)} duplicate point mappings (point in multiple trajectories)")
    
    # Output CSV path
    output_csv_path = os.path.join(
        output_dir, "perform_correlation_output",
        f"{start_frame}_to_{end_frame}", "points_trajectory_mapping.csv"
    )
    
    # Read tracker.csv and create mapping CSV
    all_points = []  # List of (frame, x, y, z) tuples
    frame_point_counts = {}  # Track point counts per frame for debugging
    
    with open(tracker_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_num = int(row['Frame'])
            visibility = int(row['Visibility'])

            if visibility != 1:
                continue

            # Parse X, Y, Z coordinates (they are in format "[x1, x2, ...]")
            x_values = literal_eval(row['X'])
            y_values = literal_eval(row['Y'])
            z_values = literal_eval(row['Z'])

            # Extract all 3D coordinates
            num_points = min(len(x_values), len(y_values), len(z_values))
            points_added_this_frame = 0

            for i in range(num_points):
                x_3d = float(x_values[i])
                y_3d = float(y_values[i])
                z_3d = float(z_values[i])

                # Skip (0, 0, 0) points (invisible detections)
                if x_3d == 0 and y_3d == 0 and z_3d == 0:
                    continue

                all_points.append((frame_num, x_3d, y_3d, z_3d))
                points_added_this_frame += 1

            # Track point count for this frame
            if points_added_this_frame > 0:
                if frame_num not in frame_point_counts:
                    frame_point_counts[frame_num] = 0
                frame_point_counts[frame_num] += points_added_this_frame

    # Print info for frames with multiple detections
    frames_with_multiple = {frame: count for frame, count in frame_point_counts.items() if count > 1}
    if frames_with_multiple:
        print(f"[CorrelationWorker]    📊 Found {len(frames_with_multiple)} frame(s) with multiple 3D detections:")
        for frame_num, count in sorted(frames_with_multiple.items())[:10]:  # Show first 10
            print(f"[CorrelationWorker]       Frame {frame_num}: {count} detections")
        if len(frames_with_multiple) > 10:
            print(f"[CorrelationWorker]       ... and {len(frames_with_multiple) - 10} more frame(s)")
    
    # Write mapping CSV
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Frame', 'X', 'Y', 'Z', 'Trajectory_ID'])

        trajectory_points_count = 0
        removed_points_count = 0

        for frame_num, x, y, z in all_points:
            # Try to match point to trajectory
            key = (frame_num, round(x, 4), round(y, 4), round(z, 4))
            traj_id = trajectory_point_map.get(key)

            if traj_id is not None:
                writer.writerow([frame_num, f"{x:.6f}", f"{y:.6f}", f"{z:.6f}", traj_id])
                trajectory_points_count += 1
            else:
                writer.writerow([frame_num, f"{x:.6f}", f"{y:.6f}", f"{z:.6f}", "removed"])
                removed_points_count += 1

        print(f"[CorrelationWorker]    📊 Point mapping CSV created: {output_csv_path}")
        print(f"[CorrelationWorker]       Trajectory points: {trajectory_points_count}, Removed points: {removed_points_count}")

    return output_csv_path


def _cleanup_staging_buffers_from_triangulation(
    max_common_frame: int,
    output_dir: str,
    camera_1_id: str,
    camera_2_id: str,
    staging_buffer_1,
    staging_buffer_2=None,
    camera_1_csv_path: Optional[str] = None,
    camera_2_csv_path: Optional[str] = None,
) -> Tuple[int, int]:
    """
    Remove frames from staging buffers based on max_common_frame timestamp.
    
    For each camera independently:
    1. Maps max_common_frame (global frame id) to per-camera stream index using mapping files
    2. Finds that frame in the staging buffer to get its sensor_timestamp
    3. Removes all frames with sensor_timestamp <= that timestamp from the staging buffer
    
    Args:
        max_common_frame: Maximum global frame index that both cameras have processed
        output_dir: Output directory where mapping files are stored
        camera_1_id: Camera 1 identifier
        camera_2_id: Camera 2 identifier
        staging_buffer_1: StagingBuffer for camera 1
        staging_buffer_2: Optional StagingBuffer for camera 2
    
    Returns:
        (removed_cam1, removed_cam2): counts removed from each staging buffer.
    """
    if staging_buffer_1 is None:
        return (0, 0)
    
    removed_cam1 = 0
    removed_cam2 = 0
    
    fid_to_stream_cam1 = (
        load_fid_to_stream_from_dist_tracker_csv(camera_1_csv_path) if camera_1_csv_path else {}
    )
    fid_to_stream_cam2 = (
        load_fid_to_stream_from_dist_tracker_csv(camera_2_csv_path) if camera_2_csv_path else {}
    )
    
    stream_idx_cam1 = fid_to_stream_cam1.get(max_common_frame)
    stream_idx_cam2 = fid_to_stream_cam2.get(max_common_frame) if staging_buffer_2 is not None else None
    
    if stream_idx_cam1 is None:
        print(f"[CorrelationWorker]    ⚠️  Could not map max_common_frame={max_common_frame} to Camera 1 camera_stream_index")
        return (0, removed_cam2)
    
    # Find timestamps for these frames in staging buffers
    ts_cut_cam1 = None
    ts_cut_cam2 = None
    
    # Camera 1: peek at frames to find the one with matching camera_stream_index
    with staging_buffer_1._condition:
        for frame_data in staging_buffer_1.buffer:
            if frame_data.camera_stream_index == stream_idx_cam1:
                frame_ts = getattr(frame_data, "sensor_timestamp", None)
                if frame_ts is not None:
                    ts_cut_cam1 = int(frame_ts)
                    break
    
    # Camera 2: peek at frames to find the one with matching camera_stream_index
    if staging_buffer_2 is not None and stream_idx_cam2 is not None:
        with staging_buffer_2._condition:
            for frame_data in staging_buffer_2.buffer:
                if frame_data.camera_stream_index == stream_idx_cam2:
                    frame_ts = getattr(frame_data, "sensor_timestamp", None)
                    if frame_ts is not None:
                        ts_cut_cam2 = int(frame_ts)
                        break
    
    # Remove frames with timestamp <= ts_cut for each camera
    # First, collect timestamps that will be removed (before removal)
    removed_ts_cam1 = []
    removed_ts_cam2 = []
    
    if ts_cut_cam1 is not None:
        # Collect timestamps that will be removed
        with staging_buffer_1._condition:
            for frame_data in staging_buffer_1.buffer:
                frame_ts = getattr(frame_data, "sensor_timestamp", None)
                if frame_ts is not None:
                    frame_ts_int = int(frame_ts)
                    if frame_ts_int <= ts_cut_cam1:
                        removed_ts_cam1.append(frame_ts_int)
        
        removed_cam1 = staging_buffer_1.remove_frames_by_timestamp_threshold(ts_cut_cam1)
        print(
            f"[CorrelationWorker]    🧹 [StagingCleanup] Camera 1: removed {removed_cam1} frames "
            f"(max_common_frame={max_common_frame}, camera_stream_index_cam1={stream_idx_cam1}, ts_cut={ts_cut_cam1})"
        )
        print(f"[CorrelationWorker]    🧹 [StagingCleanup] Camera 1: removed timestamps={removed_ts_cam1}")
    else:
        print(f"[CorrelationWorker]    ⚠️  Camera 1: Could not find timestamp for camera_stream_index={stream_idx_cam1}")
    
    if staging_buffer_2 is not None and ts_cut_cam2 is not None:
        # Collect timestamps that will be removed
        with staging_buffer_2._condition:
            for frame_data in staging_buffer_2.buffer:
                frame_ts = getattr(frame_data, "sensor_timestamp", None)
                if frame_ts is not None:
                    frame_ts_int = int(frame_ts)
                    if frame_ts_int <= ts_cut_cam2:
                        removed_ts_cam2.append(frame_ts_int)
        
        removed_cam2 = staging_buffer_2.remove_frames_by_timestamp_threshold(ts_cut_cam2)
        print(
            f"[CorrelationWorker]    🧹 [StagingCleanup] Camera 2: removed {removed_cam2} frames "
            f"(max_common_frame={max_common_frame}, camera_stream_index_cam2={stream_idx_cam2}, ts_cut={ts_cut_cam2})"
        )
        print(f"[CorrelationWorker]    🧹 [StagingCleanup] Camera 2: removed timestamps={removed_ts_cam2}")
    elif staging_buffer_2 is not None:
        print(f"[CorrelationWorker]    ⚠️  Camera 2: Could not find timestamp for camera_stream_index={stream_idx_cam2}")
    
    return (removed_cam1, removed_cam2)


def correlation_worker_wrapper(*args, **kwargs):
    """Wrapper to call correlation worker."""
    # Extract profiler from kwargs if present, or from args if it's the last positional arg
    correlation_worker(*args, **kwargs)


def correlation_worker(
    camera_1_csv_path: str,
    camera_2_csv_path: str,
    camera_1_id: str,
    camera_2_id: str,
    camera_1_cam_path: str,
    camera_2_cam_path: str,
    camera_1_video_path: str,
    camera_2_video_path: str,
        output_dir: str,
        frame_sync_info_path: str,  # Unused in realtime flow (shared global frame id in CSVs)
        stop_event: threading.Event,
        profiler=None,
        check_interval: float = 0.05,  # Check for new frames every 0.05 seconds
        staging_buffer_1=None,  # Staging buffer for camera 1 (to get frames for visualization)
        staging_buffer_2=None,  # Staging buffer for camera 2 (to get frames for visualization)
        enable_visualization: bool = False,  # Only create videos when True
        enable_stitched_visualization: bool = False,  # Only create stitched correlation video when True
        inference_done_event: Optional[threading.Event] = None,  # Set when inference is finished
        chunk_size: int = 96,  # Batch size for OriginalFrameBuffer clear_batch (judex pipeline)
        original_frame_width: Optional[int] = None,
        original_frame_height: Optional[int] = None,
        force_stop_event: Optional[threading.Event] = None,
):
    """
    Worker thread that continuously monitors dist_tracker.csv files and performs
    pairwise correlation for ready frame segments.
    
    Args:
        camera_1_csv_path: Path to camera 1 dist_tracker.csv
        camera_2_csv_path: Path to camera 2 dist_tracker.csv
        camera_1_id: Camera 1 identifier
        camera_2_id: Camera 2 identifier
        camera_1_cam_path: Path to camera 1 calibration pickle file
        camera_2_cam_path: Path to camera 2 calibration pickle file
        camera_1_video_path: Path to camera 1 video file
        camera_2_video_path: Path to camera 2 video file
        output_dir: Output directory for correlation results
        frame_sync_info_path: Path to frame synchronization info JSON (unused in realtime flow)
        stop_event: Event to signal thread to stop
        profiler: TimeProfiler instance for recording timing metrics
        check_interval: Time in seconds between checks for new frames
        original_frame_width: Optional override for camera-1 original frame width (else from OriginalFrameBuffer)
        original_frame_height: Optional override for camera-1 original frame height
    """
    import sys
    sys.stdout.flush()
    print("[CorrelationWorker] " + "=" * 60, flush=True)
    print("[CorrelationWorker] 🔗 Correlation worker thread started", flush=True)
    print("[CorrelationWorker] " + "=" * 60, flush=True)
    print(f"[CorrelationWorker]    📁 Camera 1 CSV: {camera_1_csv_path}", flush=True)
    print(f"[CorrelationWorker]    📁 Camera 2 CSV: {camera_2_csv_path}", flush=True)
    print(f"[CorrelationWorker]    📁 Output dir: {output_dir}", flush=True)
    print(f"[CorrelationWorker]    📁 Camera 1 object: {camera_1_cam_path}", flush=True)
    print(f"[CorrelationWorker]    📁 Camera 2 object: {camera_2_cam_path}", flush=True)
    print(f"[CorrelationWorker]    📁 Camera 1 video: {camera_1_video_path}", flush=True)
    print(f"[CorrelationWorker]    📁 Camera 2 video: {camera_2_video_path}", flush=True)
    sys.stdout.flush()

    cam1_csv_dir = os.path.dirname(camera_1_csv_path)
    cam2_csv_dir = os.path.dirname(camera_2_csv_path)

    # Import here to avoid circular imports
    import sys
    print("[CorrelationWorker]    🔄 Importing do_pairwise_correlation_realtime...", flush=True)
    sys.stdout.flush()
    from .realtime_pairwise_correlation import do_pairwise_correlation_realtime
    print("[CorrelationWorker]    ✅ Successfully imported do_pairwise_correlation_realtime", flush=True)
    sys.stdout.flush()
    
    # Track processed segments to avoid duplicates
    processed_segments = set()
    
    # Single stitched correlation video: append per segment, finalize on exit
    stitched_writer_state = {}
    
    # Track last processed frame (common for both cameras)
    last_processed_frame = -1
    
    # Track last segment formation time for profiling
    last_segment_formation_time = None
    
    # Track trajectory handoff context between segments
    trajectory_context = None
    
    # World points file for trajectory bounds checking
    world_points_file = 'shot_detection_and_trajectory_analysis/world_points_3D.txt'
    
    check_count = 0
    last_progress_time = time.time()
    correlation_batch = 0  # increments each time we successfully process a set of segments
    last_busy_end_time = time.time()  # for correlation_wait_for_ready_s (first segment t_wait_ready ≈ 0)
    while not stop_event.is_set():
        check_count += 1
        if check_count % 10 == 0:  # Print status every 10 checks (every 0.5 seconds now)
            print(f"[CorrelationWorker]    🔍 Correlation worker: Check #{check_count} (waiting for CSV files or new frames...)")
        
        # Check if CSV files exist
        csv1_exists = os.path.exists(camera_1_csv_path)
        csv2_exists = os.path.exists(camera_2_csv_path)
        if not csv1_exists or not csv2_exists:
            if check_count <= 3:  # Print first few times
                print(f"[CorrelationWorker]    ⏳ Waiting for CSV files... (cam1: {csv1_exists}, cam2: {csv2_exists})")
            time.sleep(check_interval)
            continue

        # Read CSV files once for frame numbers (Frame column)
        # Note: CSV has format: Frame,X,Y,Visibility where X and Y can be lists like "[300, 500, 0]"
        # The CSV is written without quotes, so pandas will fail to parse due to commas in lists
        # We'll read just the Frame column by parsing line by line
        frames_cam1 = []
        frames_cam2 = []
        
        # Read Camera 1 CSV once
        with open(camera_1_csv_path, 'r') as f:
            lines_cam1 = f.readlines()

        # Extract frame numbers from Camera 1 CSV
        if len(lines_cam1) > 1:  # Skip header
            for line in lines_cam1[1:]:  # Skip header line
                # Frame is the first field before the first comma
                frame_num = int(line.split(',')[0].strip())
                frames_cam1.append(frame_num)

        # Read Camera 2 CSV once
        with open(camera_2_csv_path, 'r') as f:
            lines_cam2 = f.readlines()

        # Extract frame numbers from Camera 2 CSV
        if len(lines_cam2) > 1:  # Skip header
            for line in lines_cam2[1:]:  # Skip header line
                # Frame is the first field before the first comma
                frame_num = int(line.split(',')[0].strip())
                frames_cam2.append(frame_num)
        
        # Get frame ranges from both cameras
        if len(frames_cam1) == 0 or len(frames_cam2) == 0:
            if check_count <= 3:
                print(f"[CorrelationWorker]    ⏳ CSV files are empty (cam1: {len(frames_cam1)} frames, cam2: {len(frames_cam2)} frames)", flush=True)
                import sys
                sys.stdout.flush()
            time.sleep(check_interval)
            continue
        
        frames_cam1 = sorted(set(frames_cam1))  # Remove duplicates and sort
        frames_cam2 = sorted(set(frames_cam2))  # Remove duplicates and sort
        
        if len(frames_cam1) == 0 or len(frames_cam2) == 0:
            if check_count <= 3:
                print(f"[CorrelationWorker]    ⏳ No frames found in CSV (cam1: {len(frames_cam1)}, cam2: {len(frames_cam2)})")
            time.sleep(check_interval)
            continue
        
        max_frame_cam1 = max(frames_cam1)
        max_frame_cam2 = max(frames_cam2)
        
        # Determine the maximum frame that both cameras have processed
        max_common_frame = min(max_frame_cam1, max_frame_cam2)
        
        if check_count % 10 == 0:  # Print status periodically
            print(f"[CorrelationWorker]    📊 CSV status: cam1 max={max_frame_cam1}, cam2 max={max_frame_cam2}, common={max_common_frame}, last_processed={last_processed_frame}")
        
        # Check if we have new frames to process
        if max_common_frame <= last_processed_frame:
            # No new frames available - check if inference is done
            if inference_done_event is not None and inference_done_event.is_set():
                # Inference is done and no new frames - all work is complete
                # Give a short grace period in case CSV flushes lag behind inference shutdown
                if (time.time() - last_progress_time) > 2.0:
                    print("[CorrelationWorker] ✅ Correlation worker: inference done and no new frames — exiting")
                    break
                else:
                    # Grace period not elapsed yet, wait a bit more
                    if stop_event.is_set():
                        break
                    time.sleep(check_interval)
                    continue
            else:
                # Inference still active, but if we haven't seen any new frames for a while,
                # print a periodic warning (but keep waiting - no timeout/exit)
                if stop_event.is_set():
                    print("[CorrelationWorker] 🛑 Stop event set — exiting correlation worker")
                    break
                idle_duration = time.time() - last_progress_time
                if idle_duration >= 20.0 and int(idle_duration) % 20 == 0:
                    print(f"[CorrelationWorker] ⚠️ Correlation worker: waiting for new frames (idle for {int(idle_duration):.0f}s, cam1 max={max_frame_cam1}, cam2 max={max_frame_cam2}, last_processed={last_processed_frame})...")

                # Keep waiting for more frames.
                time.sleep(check_interval)
                continue
        
        # Determine which frames are ready for correlation.
        # We take ALL accumulated new frames since last processing:
        # start from the last processed frame + 1 up to max_common_frame.
        start_frame = max(0, last_processed_frame + 1)
        end_frame = max_common_frame
        
        if end_frame <= start_frame:
            time.sleep(check_interval)
            continue
        
        # Form exactly one segment spanning all currently-available common frames.
        # This processes everything from the first unprocessed frame up to max_common_frame.
        segment = (start_frame, end_frame)
        segment_key = f"{start_frame}_{end_frame}"
        if segment_key in processed_segments:
            # No new work for this exact range; wait for more frames.
            time.sleep(check_interval)
            continue
        
        # Process ready segment — timing model (seconds; accounted + overhead ≈ wall):
        #   correlation_segment_pairwise_s   — do_pairwise_correlation_realtime
        #   correlation_segment_trajectory_s   — create/merge/filter + mapping CSV
        #   correlation_segment_viz_overlay_s — cam1 overlay MP4 chunks (if enabled)
        #   correlation_segment_viz_stitched_s — side-by-side stitched MP4 (if enabled)
        #   correlation_segment_accounted_s    — sum of the four above
        #   correlation_segment_overhead_s     — wall − accounted (cleanup, I/O, etc.)
        #   correlation_complete_segment_time  — wall time for full batch
        #   correlation_wait_for_ready_s     — idle before this segment (poll/sleep since last segment end)
        t_wait_ready = time.time() - last_busy_end_time
        if profiler:
            profiler.record(
                "correlation_wait_for_ready_s",
                t_wait_ready,
                write_immediately=True,
                batch=correlation_batch,
                metadata=f"segment={start_frame}-{end_frame}",
            )
        complete_segment_start_time = time.time()
        t_pairwise = 0.0
        t_trajectory = 0.0
        t_viz_overlay = 0.0
        t_viz_stitched = 0.0
        
        # Time profiling: time since last segment formation
        current_time = time.time()
        if last_segment_formation_time is not None and profiler:
            time_since_last_segment = current_time - last_segment_formation_time
            profiler.record(
                "correlation_time_between_segments",
                time_since_last_segment,
                write_immediately=False,
                batch=correlation_batch,
            )
        
        # Record segment size (inclusive range: end_frame - start_frame + 1)
        segment_size_frames = end_frame - start_frame + 1
        if profiler:
            profiler.record(
                "correlation_segment_size",
                segment_size_frames,
                write_immediately=False,
                metadata=f"segment={start_frame}-{end_frame}",
                batch=correlation_batch,
            )
        
        # Update last segment formation time
        last_segment_formation_time = current_time
        
        seg_meta = f"segment={start_frame}-{end_frame},frames={segment_size_frames}"
        print(
            f"[CorrelationWorker] Batch {correlation_batch} — start {seg_meta}",
            flush=True,
        )
        if last_segment_formation_time is not None:
            if stop_event.is_set() or (
                force_stop_event is not None and force_stop_event.is_set()
            ):
                print(
                    "[CorrelationWorker] 🛑 Stop before segment processing — exiting",
                    flush=True,
                )
                break
            correlation_start_time = time.time()

            do_pairwise_correlation_realtime(
                camera_1_cam_path=camera_1_cam_path,
                camera_2_cam_path=camera_2_cam_path,
                camera_1_id=camera_1_id,
                camera_2_id=camera_2_id,
                camera_1_video_path=camera_1_video_path,
                camera_2_video_path=camera_2_video_path,
                output_dir=output_dir,
                frame_segments=[segment],
                camera_1_tracker_csv=camera_1_csv_path,
                camera_2_tracker_csv=camera_2_csv_path,
                create_video=False,
            )
            
            correlation_duration = time.time() - correlation_start_time
            t_pairwise = correlation_duration
            total_frames_processed = segment_size_frames

            if profiler:
                profiler.record(
                    "correlation_segment_pairwise_s",
                    t_pairwise,
                    write_immediately=True,
                    batch=correlation_batch,
                    metadata=seg_meta,
                )
                if total_frames_processed >= 96:
                    time_per_96_frames = (t_pairwise / total_frames_processed) * 96
                    profiler.record(
                        "correlation_and_triangulation_96_frames",
                        time_per_96_frames,
                        write_immediately=True,
                        batch=correlation_batch,
                        metadata=seg_meta,
                    )
            
            # Mark segment as processed
            processed_segments.add(segment_key)
            
            # Update last processed frame
            last_processed_frame = max(end_frame, last_processed_frame)
            last_progress_time = time.time()
            
            # Output files are stored in:
            # {output_dir}/perform_correlation_output/{start_frame}_to_{end_frame}/tracker.csv
            # This CSV contains 3D coordinates (X, Y, Z) after triangulation
            # Format: Frame, Visibility, X, Y, Z, Original_Frame, Point_Coordinates_Camera1, Point_Coordinates_Camera2
            segment_output = os.path.join(
                output_dir,
                "perform_correlation_output",
                f"{segment[0]}_to_{segment[1]}",
                "tracker.csv",
            )
            print(
                f"[CorrelationWorker]    ├─ pairwise+triangulation: {t_pairwise:.3f}s  "
                f"tracker_csv={segment_output}",
                flush=True,
            )
            
            # ============ TRAJECTORY CREATION ============
            # After triangulation, create trajectories from 3D points
            all_trajectory_data = {}  # Collect trajectory data for visualization
            
            trajectory_start_time = time.time()
            
            # Track global trajectory offset across segments for consistent IDs
            global_traj_offset = 0
            
            tw, th = _resolve_trajectory_original_frame_size(
                staging_buffer_1,
                original_frame_width,
                original_frame_height,
            )
            stored_trajectories, detections_per_frame, traj_folder, trajectory_context = create_trajectories_realtime(
                segment=segment,
                output_dir=output_dir,
                camera_1_id=camera_1_id,
                world_points_file=world_points_file,
                previous_context=trajectory_context,
                frame_width=tw,
                frame_height=th,
            )
            
            original_count = len(stored_trajectories)
            print(
                f"[CorrelationWorker] Batch {correlation_batch}: trajectories raw={original_count} "
                f"(frame_size={tw}x{th})"
            )
            
            # ============ TRAJECTORY MERGING ============
            removed_trajectories = []
            if len(stored_trajectories) > 0:
                merge_start = time.time()

                # Step 1: Merge consecutive trajectories (gap 1–3 frames)
                stored_trajectories = merge_trajectories(stored_trajectories)
                after_merge = len(stored_trajectories)

                # Step 2: Merge overlapping trajectories (1–4 overlapping frames)
                stored_trajectories = merge_overlapping_trajectories(stored_trajectories)
                after_overlap = len(stored_trajectories)

                # Step 3: Filter to keep only one point per frame (select best trajectory)
                before_filter_count = len(stored_trajectories)
                (
                    stored_trajectories,
                    removed_trajectories,
                    filter_stats,
                ) = get_best_point_each_frame(stored_trajectories, segment)
                after_filter_count = len(stored_trajectories)

                # Write per-frame filter statistics CSV for this segment
                if traj_folder:
                    filter_csv_path = os.path.join(
                        traj_folder, "trajectory_filter_stats.csv"
                    )
                else:
                    stats_dir = os.path.join(output_dir, "trajectory_filter_stats")
                    os.makedirs(stats_dir, exist_ok=True)
                    filter_csv_path = os.path.join(
                        stats_dir, f"{segment[0]}_to_{segment[1]}.csv"
                    )

                # Always create the CSV (even if no frames were filtered) so the
                # file is present per segment; rows are only written when stats exist.
                with open(filter_csv_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            "frame_id",
                            "picked_x",
                            "picked_y",
                            "picked_z",
                            "num_points_dropped",
                            "num_points_before",
                        ]
                    )
                    if filter_stats:
                        for row in sorted(filter_stats, key=lambda r: r["frame"]):
                            writer.writerow(
                                [
                                    row["frame"],
                                    row["picked_x"],
                                    row["picked_y"],
                                    row["picked_z"],
                                    row["points_dropped"],
                                    row["points_before"],
                                ]
                            )

                merge_duration = time.time() - merge_start
                if profiler:
                    profiler.record(
                        "trajectory_merging",
                        merge_duration,
                        write_immediately=False,
                        batch=correlation_batch,
                        metadata=seg_meta,
                    )
            # ============ END TRAJECTORY MERGING ============
            
            # ============ CREATE POINT-TO-TRAJECTORY MAPPING CSV ============
            mapping_csv_path = _create_point_trajectory_mapping_csv(
                segment=segment,
                output_dir=output_dir,
                stored_trajectories=stored_trajectories,
                global_traj_offset=global_traj_offset,
            )
            # ============ END POINT-TO-TRAJECTORY MAPPING ============
            
            # Store merged & filtered trajectory data for visualization
            all_trajectory_data[segment] = stored_trajectories
            
            # Update global trajectory offset
            global_traj_offset += len(stored_trajectories) if stored_trajectories else 0
            
            trajectory_duration = time.time() - trajectory_start_time
            t_trajectory = trajectory_duration
            if profiler:
                profiler.record(
                    "trajectory_creation_batch",
                    t_trajectory,
                    write_immediately=True,
                    batch=correlation_batch,
                    metadata=seg_meta,
                )
                profiler.record(
                    "correlation_segment_trajectory_s",
                    t_trajectory,
                    write_immediately=True,
                    batch=correlation_batch,
                    metadata=seg_meta,
                )
            print(
                f"[CorrelationWorker]    ├─ trajectory (create/merge/filter/map): {t_trajectory:.3f}s  "
                f"raw={original_count} final={len(stored_trajectories)} removed={len(removed_trajectories)} "
                f"{tw}x{th} mapping={'yes' if mapping_csv_path else 'no'}",
                flush=True,
            )
            # ============ END TRAJECTORY CREATION ============
            
            # Optional: cam1 overlay videos + stitched side-by-side (timed separately)
            if enable_visualization and staging_buffer_1 is not None:
                viz_overlay_t0 = time.time()
                _create_visualization_from_triangulation(
                    frame_segments=[segment],
                    output_dir=output_dir,
                    staging_buffer_1=staging_buffer_1,
                    camera_1_cam_path=camera_1_cam_path,
                    camera_1_id=camera_1_id,
                    camera_1_csv_path=camera_1_csv_path,
                    staging_buffer_2=staging_buffer_2,
                    profiler=profiler,
                    trajectory_data=all_trajectory_data,  # Kept trajectories
                    removed_trajectory_data={segment: removed_trajectories}
                    if removed_trajectories
                    else None,
                    trail_length=10,
                    show_trajectory_labels=True,
                )
                t_viz_overlay = time.time() - viz_overlay_t0
                if profiler:
                    profiler.record(
                        "correlation_segment_viz_overlay_s",
                        t_viz_overlay,
                        write_immediately=True,
                        batch=correlation_batch,
                        metadata=seg_meta,
                    )
                print(
                    f"[CorrelationWorker]    ├─ viz cam1 overlay (chunks): {t_viz_overlay:.3f}s",
                    flush=True,
                )

                if enable_stitched_visualization and staging_buffer_2 is not None:
                    stitched_t0 = time.time()
                    fnwd_to_original_cam1 = load_fid_to_stream_from_dist_tracker_csv(
                        camera_1_csv_path
                    )
                    fnwd_to_original_cam2 = load_fid_to_stream_from_dist_tracker_csv(
                        camera_2_csv_path
                    )
                    s0, s1 = segment[0], segment[1]
                    if not fnwd_to_original_cam1:
                        fnwd_to_original_cam1 = {f: f for f in range(s0, s1 + 1)}
                    if not fnwd_to_original_cam2:
                        fnwd_to_original_cam2 = {f: f for f in range(s0, s1 + 1)}

                    if fnwd_to_original_cam1 and fnwd_to_original_cam2:
                        correlated_pairs = load_correlated_pairs_from_tracker_csvs_realtime(
                            output_dir=output_dir,
                            frame_segments=[segment],
                        )
                        stitched_suffix = (
                            "_nodetections" if not correlated_pairs else ""
                        )
                        append_stitched_segment_to_video(
                            frame_segments=[segment],
                            output_dir=output_dir,
                            camera_1_id=camera_1_id,
                            camera_2_id=camera_2_id,
                            staging_buffer_1=staging_buffer_1,
                            staging_buffer_2=staging_buffer_2,
                            fnwd_to_original_cam1=fnwd_to_original_cam1,
                            fnwd_to_original_cam2=fnwd_to_original_cam2,
                            correlated_pairs_per_frame=correlated_pairs,
                            filename_suffix=stitched_suffix,
                        )
                        if not correlated_pairs:
                            print(
                                "[CorrelationWorker]    ℹ️  Stitched viz (no detections): "
                                f"filename suffix _nodetections, segment {segment[0]}-{segment[1]}",
                                flush=True,
                            )
                    else:
                        print(
                            "[CorrelationWorker]    ⚠️  Missing FID→stream maps for stitched viz "
                            f"(cam1={len(fnwd_to_original_cam1)}, cam2={len(fnwd_to_original_cam2)})",
                            flush=True,
                        )
                    t_viz_stitched = time.time() - stitched_t0
                    if profiler:
                        profiler.record(
                            "correlation_segment_viz_stitched_s",
                            t_viz_stitched,
                            write_immediately=True,
                            batch=correlation_batch,
                            metadata=seg_meta,
                        )
                    print(
                        f"[CorrelationWorker]    ├─ viz stitched (side-by-side): {t_viz_stitched:.3f}s",
                        flush=True,
                    )
            
            # Cleanup staging buffers after visualization (or if visualization disabled)
            if staging_buffer_1 is not None:
                # judex pipeline: OriginalFrameBuffer cleared by batch number
                from ..triplet_csv_reader import OriginalFrameBuffer as _OriginalFrameBuffer
                _use_original_buffer = isinstance(staging_buffer_1, _OriginalFrameBuffer)

                if _use_original_buffer:
                    # Reader batch_num is 0,1,2,... per triplet chunk — NOT max_frame // chunk_size.
                    # Clear every reader batch whose max Source_Index is within this segment so
                    # partial trailing chunks are kept until a later segment.
                    end_frame_processed = segment[1]
                    orig_size_before = len(staging_buffer_1)
                    clear_infos: List[Tuple[int, Dict]] = []
                    for bn in sorted(staging_buffer_1.get_batch_info().keys()):
                        rng = staging_buffer_1.peek_batch_source_index_range(bn)
                        if rng is None:
                            continue
                        _min_src, _max_src = rng
                        if _max_src <= end_frame_processed:
                            clear_infos.append((bn, staging_buffer_1.clear_batch(bn)))
                    orig_size_after = len(staging_buffer_1)
                    if clear_infos:
                        parts = []
                        total_frames = 0
                        for bn, ci in clear_infos:
                            total_frames += int(ci.get("count") or 0)
                            parts.append(
                                f"reader_batch={bn}:src={ci.get('min_source_index')}..{ci.get('max_source_index')}"
                                f",csv_rows={ci.get('min_csv_row_id')}..{ci.get('max_csv_row_id')},n={ci.get('count')}"
                            )
                        print(
                            f"[CorrelationWorker] Batch {correlation_batch}: original_buffer cleared "
                            f"{len(clear_infos)} reader batch(es) (segment end_frame={end_frame_processed}); "
                            f"total_frames={total_frames}; " + "; ".join(parts) + f"; "
                            f"total_size {orig_size_before}->{orig_size_after}",
                            flush=True,
                        )
                        if profiler:
                            profiler.record(
                                "correlation_orig_buffer_cleared",
                                float(orig_size_before - orig_size_after),
                                write_immediately=True,
                                batch=correlation_batch,
                                metadata=(
                                    f"segment_end_frame={end_frame_processed},reader_batches_cleared="
                                    f"{[b for b, _ in clear_infos]},total_frames={total_frames},"
                                    f"details={'; '.join(parts)},before={orig_size_before},after={orig_size_after}"
                                ),
                            )
                    else:
                        print(
                            f"[CorrelationWorker] Batch {correlation_batch}: original_buffer — no reader batch "
                            f"fully cleared (max Source_Index per batch still > segment end_frame={end_frame_processed}, "
                            f"or buffer empty); size={orig_size_after}",
                            flush=True,
                        )
                else:
                    removed1, removed2 = _cleanup_staging_buffers_from_triangulation(
                        max_common_frame=max_common_frame,
                        output_dir=output_dir,
                        camera_1_id=camera_1_id,
                        camera_2_id=camera_2_id,
                        staging_buffer_1=staging_buffer_1,
                        staging_buffer_2=staging_buffer_2,
                        camera_1_csv_path=camera_1_csv_path,
                        camera_2_csv_path=camera_2_csv_path,
                    )
                    if enable_visualization:
                        print(
                            f"[CorrelationWorker]    🧹 Cleaned staging buffers after visualization "
                            f"(cam1 removed={removed1}, cam2 removed={removed2})"
                        )
                    else:
                        pass

            complete_segment_duration = time.time() - complete_segment_start_time
            t_accounted = (
                t_pairwise + t_trajectory + t_viz_overlay + t_viz_stitched
            )
            t_overhead = complete_segment_duration - t_accounted
            if profiler:
                profiler.record(
                    "correlation_segment_accounted_s",
                    t_accounted,
                    write_immediately=True,
                    batch=correlation_batch,
                    metadata=seg_meta,
                )
                profiler.record(
                    "correlation_segment_overhead_s",
                    t_overhead,
                    write_immediately=True,
                    batch=correlation_batch,
                    metadata=seg_meta,
                )
                profiler.record(
                    "correlation_complete_segment_time",
                    complete_segment_duration,
                    write_immediately=True,
                    batch=correlation_batch,
                    metadata=(
                        f"{seg_meta},wait_ready={t_wait_ready:.4f},pairwise={t_pairwise:.4f},"
                        f"trajectory={t_trajectory:.4f},viz_overlay={t_viz_overlay:.4f},"
                        f"viz_stitched={t_viz_stitched:.4f},accounted={t_accounted:.4f},"
                        f"overhead={t_overhead:.4f}"
                    ),
                )
                profiler.write_correlation_segment_breakdown(
                    batch=correlation_batch,
                    seg_meta=seg_meta,
                    t_wait_ready=t_wait_ready,
                    t_pairwise=t_pairwise,
                    t_trajectory=t_trajectory,
                    t_viz_overlay=t_viz_overlay,
                    t_viz_stitched=t_viz_stitched,
                    t_accounted=t_accounted,
                    t_overhead=t_overhead,
                    t_wall=complete_segment_duration,
                )
            print(
                f"[CorrelationWorker] Batch {correlation_batch} — timing (s)  "
                f"wait_before={t_wait_ready:.3f} | "
                f"pairwise={t_pairwise:.3f} + trajectory={t_trajectory:.3f} + "
                f"viz_cam1={t_viz_overlay:.3f} + viz_stitched={t_viz_stitched:.3f} "
                f"= accounted {t_accounted:.3f} | overhead {t_overhead:.3f} | "
                f"wall {complete_segment_duration:.3f}",
                flush=True,
            )

            # Mark one correlation batch complete (segments + trajectory + optional viz/cleanup)
            correlation_batch += 1
            last_busy_end_time = time.time()
        
        # Wait before next check
        time.sleep(check_interval)

    print("[CorrelationWorker] 🛑 Correlation worker thread stopped")

