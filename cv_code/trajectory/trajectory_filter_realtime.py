"""
Realtime trajectory filtering utilities.

This module provides a realtime-compatible version of the
`get_best_point_each_frame` logic originally implemented in
`shot_detection_and_trajectory_analysis.utility.trajectory_filter`.

It operates on `Trajectory` objects from `realtime.trajectory_realtime`
and selects a single "best" trajectory when multiple trajectories
have detections in the same frame.
"""

from typing import List, Tuple, Optional, Dict, Any

import numpy as np

from .trajectory_realtime.detection import Trajectory, Detection


# Net line coordinates (start and end points).
# These are copied from the original trajectory_filter implementation.
NET_LINE_START = np.array([0.0, 6.7, 0.0])
NET_LINE_END = np.array([6.1, 6.7, 0.0])
NET_LINE_CENTER = (NET_LINE_START + NET_LINE_END) / 2.0  # (3.05, 6.7, 0.0)

# Frame difference threshold for Case 2 vs Case 3
FRAME_DIFF_THRESHOLD = 3

# Number of last frames to skip (don't filter)
LAST_FRAMES_TO_SKIP = 10


def euclidean_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Calculate Euclidean distance between two 3D points."""
    return np.linalg.norm(p1 - p2)


def get_trajectory_length(trajectory: Trajectory) -> int:
    """
    Calculate trajectory length as span of frames (last_frame - first_frame).

    Args:
        trajectory: Trajectory object

    Returns:
        Length as integer (0 if empty trajectory)
    """
    if len(trajectory.detections) == 0:
        return 0

    frame_numbers = sorted(trajectory.detections.keys())
    return frame_numbers[-1] - frame_numbers[0]


def get_trajectory_centroid(trajectory: Trajectory) -> np.ndarray:
    """
    Calculate centroid of all points in a trajectory.

    Args:
        trajectory: Trajectory object

    Returns:
        Centroid as numpy array (x, y, z)
    """
    if len(trajectory.detections) == 0:
        return np.array([0.0, 0.0, 0.0])

    points = []
    for detection in trajectory.detections.values():
        points.append(np.array([detection.x, detection.y, detection.z]))

    return np.mean(points, axis=0)


def find_previous_trajectory(
    current_frame: int,
    trajectories: List[Trajectory],
    exclude_trajectories: set,
) -> Optional[Tuple[Trajectory, int]]:
    """
    Find the closest previous trajectory (one with most recent detection before current_frame).

    Args:
        current_frame: Current frame number
        trajectories: List of all trajectories
        exclude_trajectories: Set of trajectory indices to exclude

    Returns:
        Tuple of (trajectory, last_frame) or None if no previous trajectory found
    """
    best_trajectory: Optional[Trajectory] = None
    best_last_frame = -1

    for idx, trajectory in enumerate(trajectories):
        if idx in exclude_trajectories:
            continue

        if len(trajectory.detections) == 0:
            continue

        # Find last frame before current_frame
        frames_before = [f for f in trajectory.detections.keys() if f < current_frame]
        if not frames_before:
            continue

        last_frame = max(frames_before)
        if last_frame > best_last_frame:
            best_last_frame = last_frame
            best_trajectory = trajectory

    if best_trajectory is None:
        return None

    return best_trajectory, best_last_frame


def get_best_point_each_frame(
    trajectories: List[Trajectory],
    segment: Tuple[int, int],
) -> Tuple[List[Trajectory], List[Trajectory], List[Dict[str, Any]]]:
    """
    Filter trajectories to keep only one point per frame by selecting best trajectory.

    Selection logic:
        1. Case 1: If trajectories have different lengths, pick longest
        2. Case 2: If equal lengths and frame_diff <= 3, use centroid of previous trajectory
        3. Case 3: If frame_diff > 3 or no previous trajectory, use net line center

    Args:
        trajectories: List of Trajectory objects (after merging)
        segment: Tuple of (start_frame, end_frame) for the segment

    Returns:
        Tuple of (filtered_trajectories, removed_trajectories)
        - filtered_trajectories: List of trajectories kept after filtering
        - removed_trajectories: List of trajectories removed during filtering
    """
    start_frame, end_frame = segment

    # Build frame-to-trajectories mapping
    frame_to_trajectories: Dict[int, List[Tuple[int, Detection]]] = {}
    # Map: frame_number -> [(trajectory_index, detection), ...]

    for traj_idx, trajectory in enumerate(trajectories):
        for frame_num, detection in trajectory.detections.items():
            if frame_num not in frame_to_trajectories:
                frame_to_trajectories[frame_num] = []
            frame_to_trajectories[frame_num].append((traj_idx, detection))

    # Determine which frames to process (exclude last 10 frames)
    frames_to_process = sorted(
        [f for f in frame_to_trajectories.keys() if f <= end_frame - LAST_FRAMES_TO_SKIP]
    )

    # Track which trajectories to remove
    trajectories_to_remove = set()

    # Collect per-frame filter statistics for debugging/analysis
    per_frame_stats: List[Dict[str, Any]] = []

    # Process each frame
    for frame_num in frames_to_process:
        traj_detections = frame_to_trajectories.get(frame_num, [])

        # Skip frames with no detections at all
        if not traj_detections:
            continue

        # Get trajectory indices for this frame (excluding already removed ones)
        active_traj_indices = [
            idx for idx, _ in traj_detections if idx not in trajectories_to_remove
        ]

        if not active_traj_indices:
            continue

        # Get trajectories for these indices
        active_trajectories = [(idx, trajectories[idx]) for idx in active_traj_indices]

        points_before = len(active_trajectories)

        # If only one active trajectory for this frame, keep it but still record stats
        if points_before == 1:
            selected_traj_idx = active_trajectories[0][0]
            selected_traj = trajectories[selected_traj_idx]
            if frame_num in selected_traj.detections:
                det = selected_traj.detections[frame_num]
                per_frame_stats.append(
                    {
                        "frame": frame_num,
                        "picked_traj_idx": selected_traj_idx,
                        "picked_x": det.x,
                        "picked_y": det.y,
                        "picked_z": det.z,
                        "points_before": points_before,
                        "points_dropped": 0,
                    }
                )
            continue

        # ========== CASE 1: Compare trajectory lengths ==========
        traj_lengths: Dict[int, int] = {}
        for idx, traj in active_trajectories:
            length = get_trajectory_length(traj)
            traj_lengths[idx] = length

        # Find longest trajectory
        max_length = max(traj_lengths.values())
        longest_trajectories = [
            idx for idx, length in traj_lengths.items() if length == max_length
        ]

        # If one trajectory is longest, pick it and remove others
        if len(longest_trajectories) == 1:
            selected_traj_idx = longest_trajectories[0]

            # Record stats for this frame
            selected_traj = trajectories[selected_traj_idx]
            if frame_num in selected_traj.detections:
                det = selected_traj.detections[frame_num]
                per_frame_stats.append(
                    {
                        "frame": frame_num,
                        "picked_traj_idx": selected_traj_idx,
                        "picked_x": det.x,
                        "picked_y": det.y,
                        "picked_z": det.z,
                        "points_before": points_before,
                        "points_dropped": max(points_before - 1, 0),
                    }
                )

            # Remove all other trajectories for this frame
            for idx, _ in active_trajectories:
                if idx != selected_traj_idx:
                    trajectories_to_remove.add(idx)
            continue

        # If multiple trajectories have same length, proceed to Case 2/3
        # ========== CASE 2/3: Use centroid or net line center ==========

        # Find previous trajectory (closest one with detection before current frame)
        # This could be one of the competing trajectories or a different trajectory
        # We exclude trajectories already marked for removal
        prev_result = find_previous_trajectory(
            frame_num, trajectories, trajectories_to_remove
        )

        if prev_result is None:
            # No previous trajectory -> Case 3: Use net line center
            selected_traj_idx = _select_by_net_line_center(
                active_trajectories, frame_num
            )
        else:
            prev_trajectory, prev_last_frame = prev_result
            frame_diff = frame_num - prev_last_frame

            if frame_diff <= FRAME_DIFF_THRESHOLD:
                # Case 2: Use centroid of previous trajectory
                selected_traj_idx = _select_by_centroid(
                    active_trajectories, prev_trajectory, frame_num
                )
            else:
                # Case 3: Use net line center
                selected_traj_idx = _select_by_net_line_center(
                    active_trajectories, frame_num
                )

        # Record stats for this frame
        selected_traj = trajectories[selected_traj_idx]
        if frame_num in selected_traj.detections:
            det = selected_traj.detections[frame_num]
            per_frame_stats.append(
                {
                    "frame": frame_num,
                    "picked_traj_idx": selected_traj_idx,
                    "picked_x": det.x,
                    "picked_y": det.y,
                    "picked_z": det.z,
                    "points_before": points_before,
                    "points_dropped": max(points_before - 1, 0),
                }
            )

        # Remove all other trajectories
        for idx, _ in active_trajectories:
            if idx != selected_traj_idx:
                trajectories_to_remove.add(idx)

    # Separate filtered and removed trajectories
    filtered_trajectories = [
        traj
        for idx, traj in enumerate(trajectories)
        if idx not in trajectories_to_remove
    ]
    removed_trajectories = [
        traj
        for idx, traj in enumerate(trajectories)
        if idx in trajectories_to_remove
    ]

    return filtered_trajectories, removed_trajectories, per_frame_stats


def _select_by_centroid(
    active_trajectories: List[Tuple[int, Trajectory]],
    previous_trajectory: Trajectory,
    current_frame: int,
) -> int:
    """
    Select trajectory based on closest point to centroid of previous trajectory.

    Args:
        active_trajectories: List of (trajectory_index, trajectory) tuples
        previous_trajectory: Previous trajectory to compute centroid from
        current_frame: Current frame number

    Returns:
        Selected trajectory index
    """
    centroid = get_trajectory_centroid(previous_trajectory)

    min_distance = float("inf")
    selected_idx: Optional[int] = None

    for traj_idx, trajectory in active_trajectories:
        if current_frame not in trajectory.detections:
            continue

        detection = trajectory.detections[current_frame]
        point = np.array([detection.x, detection.y, detection.z])
        distance = euclidean_distance(centroid, point)

        if distance < min_distance:
            min_distance = distance
            selected_idx = traj_idx

    # Fallback: return first trajectory if something went wrong
    return selected_idx if selected_idx is not None else active_trajectories[0][0]


def _select_by_net_line_center(
    active_trajectories: List[Tuple[int, Trajectory]],
    current_frame: int,
) -> int:
    """
    Select trajectory based on closest point to net line center.

    Args:
        active_trajectories: List of (trajectory_index, trajectory) tuples
        current_frame: Current frame number

    Returns:
        Selected trajectory index
    """
    min_distance = float("inf")
    selected_idx: Optional[int] = None

    for traj_idx, trajectory in active_trajectories:
        if current_frame not in trajectory.detections:
            continue

        detection = trajectory.detections[current_frame]
        point = np.array([detection.x, detection.y, detection.z])
        distance = euclidean_distance(NET_LINE_CENTER, point)

        if distance < min_distance:
            min_distance = distance
            selected_idx = traj_idx

    # Fallback: return first trajectory if something went wrong
    return selected_idx if selected_idx is not None else active_trajectories[0][0]

