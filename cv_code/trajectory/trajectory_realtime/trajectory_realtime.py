"""
Real-Time Trajectory Creator Module

This module provides functions for creating trajectories from 3D points
in a real-time streaming fashion, with support for handoff context
to maintain trajectory continuity across segment boundaries.

This is the realtime-specific version, separated from the original
shot_detection_and_trajectory_analysis module.
"""

import os
import csv
import numpy as np
from ast import literal_eval
from typing import Dict, List, Tuple, Optional

from .trajectory_tree import DetectionTree
from .trajectory_context import TrajectoryHandoffContext


def process_csv_realtime(file_path: str, result_dict: Dict):
    """
    Process CSV file containing trajectory data and populate result_dict.

    Args:
        file_path: Path to the CSV file
        result_dict: Dictionary to store processed trajectory data
    """
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            original_frame = int(row['Original_Frame'])
            x_values = literal_eval(row['X'])
            y_values = literal_eval(row['Y'])
            z_values = literal_eval(row['Z'])
            coordinates = list(zip(x_values, y_values, z_values))
            result_dict[original_frame] = coordinates


def point_in_bounds(point: np.ndarray, min_bound: np.ndarray, max_bound: np.ndarray) -> bool:
    """
    Check if a point lies within the bounding box.
    
    Args:
        point (np.array): 3D point coordinates
        min_bound (np.array): Minimum bounds [min_x, min_y, min_z]
        max_bound (np.array): Maximum bounds [max_x, max_y, max_z]
    
    Returns:
        bool: True if point is within bounds, False otherwise
    """
    return np.all(point >= min_bound) and np.all(point <= max_bound)


def create_trajectories_realtime(
    segment: Tuple[int, int],
    output_dir: str,
    camera_1_id: str,
    world_points_file: str,
    previous_context: Optional[TrajectoryHandoffContext] = None,
    frame_width: Optional[int] = None,
    frame_height: Optional[int] = None,
) -> Tuple[List, Dict, str, Optional[TrajectoryHandoffContext]]:
    """
    Create trajectories for a segment with optional context from previous segment.

    This function processes 3D points from triangulation and builds trajectories
    using DetectionTree. When previous_context is provided, it continues
    trajectories from the previous segment.

    Args:
        segment: Tuple of (start_frame, end_frame)
        output_dir: Base directory to read from (perform_correlation_output)
        camera_1_id: Camera 1 identifier
        world_points_file: Path to world points file for bounds checking
        previous_context: Optional handoff context from previous segment
        frame_width: Original-resolution frame width for camera 1 (from pipeline / OriginalFrameBuffer)
        frame_height: Original-resolution frame height for camera 1

    Returns:
        Tuple of:
            - stored_trajectories: List of completed trajectories
            - detections_per_frame: Dict mapping frame to detections
            - frame_segment_folder: Output folder path
            - next_context: Handoff context for next segment
    """
    start_frame = segment[0]
    end_frame = segment[1]

    if start_frame == end_frame:
        print(f"[CorrelationWorker]   ⚠️  Empty segment, skipping", flush=True)
        return [], {}, None, previous_context

    # Where to write trajectory_output and handoff context
    trajectory_output_dir = os.path.join(output_dir, "trajectory_output")
    if not os.path.exists(trajectory_output_dir):
        os.makedirs(trajectory_output_dir)
    
    frame_segment_folder = os.path.join(trajectory_output_dir, f"{start_frame}_to_{end_frame}")
    if not os.path.exists(frame_segment_folder):
        os.makedirs(frame_segment_folder)
    
    # Load world points for bounds checking
    if os.path.exists(world_points_file):
        world_points = np.loadtxt(world_points_file)
        min_bound = world_points.min(axis=0)
        max_bound = world_points.max(axis=0)
    else:
        # Default bounds if file doesn't exist
        print(f"[CorrelationWorker]   ⚠️  World points file not found: {world_points_file}, using default bounds", flush=True)
        min_bound = np.array([-10, -10, 0])
        max_bound = np.array([20, 20, 10])
    
    # Path to triangulated 3D points
    tracker_csv_path = os.path.join(
        output_dir, "perform_correlation_output", 
        f"{start_frame}_to_{end_frame}", "tracker.csv"
    )
    
    if not os.path.exists(tracker_csv_path):
        print(f"[CorrelationWorker]   ⚠️  Tracker CSV not found: {tracker_csv_path}", flush=True)
        return [], {}, frame_segment_folder, previous_context
    
    # Process CSV data
    result_dict = {}
    process_csv_realtime(tracker_csv_path, result_dict)
    if len(result_dict) == 0:
        print(f"[CorrelationWorker]   ⚠️  No data in tracker CSV", flush=True)
        return [], {}, frame_segment_folder, previous_context
    
    # Sort data by frame number
    data_dict = dict(sorted(result_dict.items()))

    tree = DetectionTree(start_frame, end_frame, frame_width, frame_height)
    
    # Restore context from previous segment if available
    if previous_context is not None and not previous_context.is_empty():
        tree.restore_from_context(previous_context)
    
    # Add detections frame by frame
    total_detections = 0
    frames_with_detections = 0
    
    for frame in range(start_frame, end_frame + 1):
        if frame not in data_dict:
            continue
        
        coords_all = data_dict[frame]
        # Filter coordinates within bounds
        coords = [pt for pt in coords_all if point_in_bounds(np.array(pt), min_bound, max_bound)]
        
        if len(coords) > 0:
            frames_with_detections += 1
            total_detections += len(coords)
        
        tree.add_detections(coords, frame)
    
    # Finalize for handoff - store finished trajectories, keep active ones
    newly_stored = tree.finalize_for_handoff(end_frame, frame_gap_threshold=10)
    
    # Get handoff context for next segment
    next_context = tree.get_handoff_context(frame_gap_threshold=10)
    
    # Get all stored trajectories
    stored_trajectories = tree.get_stored_trajectories()
    
    # Build detections_per_frame from trajectories
    detections_per_frame = {}
    for traj_id, trajectory in enumerate(stored_trajectories):
        for frame, detection in trajectory.detections.items():
            if frame not in detections_per_frame:
                detections_per_frame[frame] = []
            detections_per_frame[frame].append({
                'x': detection.x,
                'y': detection.y,
                'z': detection.z,
                'traj_id': traj_id
            })
    
    # Save context to file for debugging/recovery
    context_file = os.path.join(frame_segment_folder, "handoff_context.json")
    next_context.save_to_file(context_file)
    print(
        f"[CorrelationWorker]   Trajectory details: frames_with_detections={frames_with_detections}, "
        f"points={total_detections}, stored_now={len(newly_stored)}, context_saved={context_file}",
        flush=True,
    )
    
    return stored_trajectories, detections_per_frame, frame_segment_folder, next_context
