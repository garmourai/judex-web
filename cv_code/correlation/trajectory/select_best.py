"""
Realtime trajectory filtering utilities.
"""

from typing import List, Tuple, Optional, Dict, Any, Set

import numpy as np

from .realtime.models import Trajectory, Detection


NET_LINE_START = np.array([0.0, 6.7, 0.0])
NET_LINE_END = np.array([6.1, 6.7, 0.0])
NET_LINE_CENTER = (NET_LINE_START + NET_LINE_END) / 2.0
FRAME_DIFF_THRESHOLD = 3
LAST_FRAMES_TO_SKIP = 10


def euclidean_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    return np.linalg.norm(p1 - p2)


def get_trajectory_length(trajectory: Trajectory) -> int:
    if len(trajectory.detections) == 0:
        return 0
    frame_numbers = sorted(trajectory.detections.keys())
    return frame_numbers[-1] - frame_numbers[0]


def get_trajectory_centroid(trajectory: Trajectory) -> np.ndarray:
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
    best_trajectory: Optional[Trajectory] = None
    best_last_frame = -1
    for idx, trajectory in enumerate(trajectories):
        if idx in exclude_trajectories or len(trajectory.detections) == 0:
            continue
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


def _active_candidates_payload(
    active_trajectories: List[Tuple[int, Trajectory]],
    frame_num: int,
    selected_traj_idx: int,
) -> List[Dict[str, Any]]:
    """One entry per active trajectory at frame_num with track_id (post-merge) and is_selected."""
    out: List[Dict[str, Any]] = []
    for idx, traj in active_trajectories:
        if frame_num not in traj.detections:
            continue
        det = traj.detections[frame_num]
        tid = traj.track_id
        if tid is None:
            raise ValueError("Trajectory.track_id must be set after merge before get_best_point_each_frame")
        out.append(
            {
                "local_traj_idx": idx,
                "track_id": int(tid),
                "x": float(det.x),
                "y": float(det.y),
                "z": float(det.z),
                "is_selected": idx == selected_traj_idx,
            }
        )
    return out


def get_best_point_each_frame(
    trajectories: List[Trajectory],
    segment: Tuple[int, int],
    last_frames_to_skip: int = LAST_FRAMES_TO_SKIP,
    handoff_frames: Optional[Set[int]] = None,
) -> Tuple[List[Trajectory], List[Trajectory], List[Dict[str, Any]], List[Dict[str, Any]]]:
    start_frame, end_frame = segment
    per_frame_stats: List[Dict[str, Any]] = []
    per_frame_decisions: List[Dict[str, Any]] = []
    frame_to_trajectories: Dict[int, List[Tuple[int, Detection]]] = {}
    for traj_idx, trajectory in enumerate(trajectories):
        for frame_num, detection in trajectory.detections.items():
            frame_to_trajectories.setdefault(frame_num, []).append((traj_idx, detection))

    tail_skip = max(0, int(last_frames_to_skip))
    low = max(0, start_frame - LAST_FRAMES_TO_SKIP)
    high = end_frame - tail_skip

    def _should_process(f: int) -> bool:
        if low <= f <= high:
            return True
        if tail_skip > 0 and high < f <= end_frame and handoff_frames is not None:
            return f not in handoff_frames
        return False

    frames_to_process = sorted(f for f in frame_to_trajectories.keys() if _should_process(f))
    trajectories_to_remove = set()

    for frame_num in frames_to_process:
        traj_detections = frame_to_trajectories.get(frame_num, [])
        if not traj_detections:
            continue
        active_traj_indices = [idx for idx, _ in traj_detections if idx not in trajectories_to_remove]
        if not active_traj_indices:
            continue
        active_trajectories = [(idx, trajectories[idx]) for idx in active_traj_indices]
        points_before = len(active_trajectories)

        if points_before == 1:
            selected_traj_idx = active_trajectories[0][0]
            selected_traj = trajectories[selected_traj_idx]
            if frame_num in selected_traj.detections:
                det = selected_traj.detections[frame_num]
                per_frame_stats.append({"frame": frame_num, "picked_traj_idx": selected_traj_idx, "picked_x": det.x, "picked_y": det.y, "picked_z": det.z, "points_before": points_before, "points_dropped": 0})
                stid = selected_traj.track_id
                if stid is None:
                    raise ValueError("Trajectory.track_id must be set after merge before get_best_point_each_frame")
                per_frame_decisions.append(
                    {
                        "frame": frame_num,
                        "local_selected_traj_idx": selected_traj_idx,
                        "selected_track_id": int(stid),
                        "num_points_before": points_before,
                        "num_points_dropped": 0,
                        "active_candidates": _active_candidates_payload(
                            active_trajectories, frame_num, selected_traj_idx
                        ),
                    }
                )
            continue

        traj_lengths: Dict[int, int] = {idx: get_trajectory_length(traj) for idx, traj in active_trajectories}
        max_length = max(traj_lengths.values())
        longest_trajectories = [idx for idx, length in traj_lengths.items() if length == max_length]

        if len(longest_trajectories) == 1:
            selected_traj_idx = longest_trajectories[0]
        else:
            prev_result = find_previous_trajectory(frame_num, trajectories, trajectories_to_remove)
            if prev_result is None:
                selected_traj_idx = _select_by_net_line_center(active_trajectories, frame_num)
            else:
                prev_trajectory, prev_last_frame = prev_result
                frame_diff = frame_num - prev_last_frame
                if frame_diff <= FRAME_DIFF_THRESHOLD:
                    selected_traj_idx = _select_by_centroid(active_trajectories, prev_trajectory, frame_num)
                else:
                    selected_traj_idx = _select_by_net_line_center(active_trajectories, frame_num)

        selected_traj = trajectories[selected_traj_idx]
        if frame_num in selected_traj.detections:
            det = selected_traj.detections[frame_num]
            dropped = max(points_before - 1, 0)
            per_frame_stats.append(
                {
                    "frame": frame_num,
                    "picked_traj_idx": selected_traj_idx,
                    "picked_x": det.x,
                    "picked_y": det.y,
                    "picked_z": det.z,
                    "points_before": points_before,
                    "points_dropped": dropped,
                }
            )
            stid = selected_traj.track_id
            if stid is None:
                raise ValueError("Trajectory.track_id must be set after merge before get_best_point_each_frame")
            per_frame_decisions.append(
                {
                    "frame": frame_num,
                    "local_selected_traj_idx": selected_traj_idx,
                    "selected_track_id": int(stid),
                    "num_points_before": points_before,
                    "num_points_dropped": dropped,
                    "active_candidates": _active_candidates_payload(
                        active_trajectories, frame_num, selected_traj_idx
                    ),
                }
            )

        for idx, _ in active_trajectories:
            if idx != selected_traj_idx:
                trajectories_to_remove.add(idx)

    filtered_trajectories = [traj for idx, traj in enumerate(trajectories) if idx not in trajectories_to_remove]
    removed_trajectories = [traj for idx, traj in enumerate(trajectories) if idx in trajectories_to_remove]
    return filtered_trajectories, removed_trajectories, per_frame_stats, per_frame_decisions


def _select_by_centroid(
    active_trajectories: List[Tuple[int, Trajectory]],
    previous_trajectory: Trajectory,
    current_frame: int,
) -> int:
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
    return selected_idx if selected_idx is not None else active_trajectories[0][0]


def _select_by_net_line_center(
    active_trajectories: List[Tuple[int, Trajectory]],
    current_frame: int,
) -> int:
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
    return selected_idx if selected_idx is not None else active_trajectories[0][0]
