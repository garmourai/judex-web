"""
Shot start/end and bounce (landing proxy) from best-per-frame 3D points.

Extracted from old pipeline rally_segmentation.py — only the functions needed for
detect_shot_points → detect_bounce_point (no cv2, no rally visualization).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

# Type alias: frame -> (traj_id, (x, y, z))
BestPerFrame = Dict[int, Tuple[int, Tuple[float, float, float]]]


def detect_bounce_point(
    start_frame: int,
    end_frame: int,
    best_per_frame: BestPerFrame,
    window_size: int = 7,
    z_threshold: float = 0.3,
) -> Optional[int]:
    """
    First bounce candidate between start and end using dz pattern + low z (ground proxy).
    """
    if start_frame is None or end_frame is None:
        return None

    shot_frames = [f for f in best_per_frame.keys() if start_frame <= f <= end_frame]
    shot_frames.sort()

    if len(shot_frames) < window_size:
        return None

    frame_to_z: Dict[int, float] = {}
    for frame in shot_frames:
        if frame in best_per_frame:
            _, (_, _, z) = best_per_frame[frame]
            frame_to_z[frame] = float(z)

    dz_values: Dict[int, float] = {}
    for i in range(1, len(shot_frames)):
        curr_frame = shot_frames[i]
        prev_frame = shot_frames[i - 1]
        if curr_frame in frame_to_z and prev_frame in frame_to_z:
            dz_values[curr_frame] = frame_to_z[curr_frame] - frame_to_z[prev_frame]

    for i in range(len(shot_frames) - window_size + 1):
        window_frames = shot_frames[i : i + window_size]
        if not all(f in dz_values for f in window_frames[1:]):
            continue

        window_dz = [dz_values[f] for f in window_frames[1:]]
        mid_point = len(window_dz) // 2
        first_half = window_dz[:mid_point]
        second_half = window_dz[mid_point:]

        first_decreasing = sum(1 for dz in first_half if dz < 0) >= len(first_half) * 0.6
        second_increasing = sum(1 for dz in second_half if dz > 0) >= len(second_half) * 0.6

        if first_decreasing and second_increasing:
            transition_frame = window_frames[mid_point]
            if transition_frame in frame_to_z and frame_to_z[transition_frame] < z_threshold:
                return transition_frame

    return None


def calculate_shot_speed(
    start_shot_frame: int,
    best_per_frame: BestPerFrame,
    fps: float = 25.0,
    consecutive_points: int = 4,
) -> float:
    if start_shot_frame not in best_per_frame:
        return 0.0

    available_frames = sorted(f for f in best_per_frame.keys() if f > start_shot_frame)
    if not available_frames:
        return 0.0

    end_frames = available_frames[: min(consecutive_points, len(available_frames))]
    if len(end_frames) < 2:
        return 0.0

    _, (x1, y1, z1) = best_per_frame[start_shot_frame]
    end_frame = end_frames[-1]
    _, (x2, y2, z2) = best_per_frame[end_frame]

    distance_meters = float(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2))
    time_seconds = (end_frame - start_shot_frame) / fps
    if time_seconds <= 0:
        return 0.0

    speed_ms = distance_meters / time_seconds
    return round(float(speed_ms * 3.6), 2)


def calculate_confidence(
    frame: int,
    detection_frames: List[int],
    dy_values: Dict[int, float],
    window_size: int,
) -> float:
    if frame not in detection_frames:
        return 0.0

    window_frames = [
        f for f in detection_frames if frame - window_size // 2 <= f <= frame + window_size // 2
    ]
    if len(window_frames) < 3:
        return 0.0

    positive_count = 0
    negative_count = 0
    for f in window_frames:
        if f in dy_values:
            if abs(dy_values[f]) < 0.1:
                continue
            if dy_values[f] > 0:
                positive_count += 1
            elif dy_values[f] < 0:
                negative_count += 1

    total_votes = positive_count + negative_count
    if total_votes == 0:
        return 0.0

    vote_margin = abs(positive_count - negative_count) / total_votes
    return min(1.0, vote_margin * 2)


def get_window_direction(
    frame: int,
    detection_frames: List[int],
    dy_values: Dict[int, float],
    window_size: int,
    direction: str,
) -> int:
    if direction == "backward":
        window_frames = [f for f in detection_frames if frame - window_size + 1 <= f <= frame]
    else:
        window_frames = [f for f in detection_frames if frame <= f <= frame + window_size - 1]

    if len(window_frames) < 3:
        return 0

    positive_count = 0
    negative_count = 0
    for f in window_frames:
        if f in dy_values:
            if abs(dy_values[f]) < 0.1:
                continue
            if dy_values[f] > 0:
                positive_count += 1
            elif dy_values[f] < 0:
                negative_count += 1

    if positive_count > negative_count:
        return 1
    if negative_count > positive_count:
        return -1
    return 0


def determine_shot_baseline(
    net_cross_frame: int,
    detection_frames: List[int],
    frame_to_y: Dict[int, float],
    net_y: float,
    baseline_top_y: float,
    baseline_bottom_y: float,
    points_each_side: int = 2,
) -> Optional[float]:
    if net_cross_frame not in detection_frames:
        return None
    df_sorted = sorted(detection_frames)
    idx = df_sorted.index(net_cross_frame)
    before_idxs = [j for j in range(max(0, idx - points_each_side), idx)]
    after_idxs = [j for j in range(idx + 1, min(len(df_sorted), idx + 1 + points_each_side))]
    if not before_idxs or not after_idxs:
        return None

    before_vals = [frame_to_y[df_sorted[j]] for j in before_idxs if df_sorted[j] in frame_to_y]
    after_vals = [frame_to_y[df_sorted[j]] for j in after_idxs if df_sorted[j] in frame_to_y]
    if not before_vals or not after_vals:
        return None

    before_avg = float(np.mean(before_vals))
    after_avg = float(np.mean(after_vals))

    if before_avg < net_y and after_avg > net_y:
        return baseline_bottom_y
    if before_avg > net_y and after_avg < net_y:
        return baseline_top_y

    dist_bottom = abs(before_avg - baseline_bottom_y)
    dist_top = abs(before_avg - baseline_top_y)
    return baseline_bottom_y if dist_bottom <= dist_top else baseline_top_y


def find_start_near_baseline_backward(
    net_cross_frame: int,
    search_start: int,
    detection_frames: List[int],
    dy_values: Dict[int, float],
    frame_to_y: Dict[int, float],
    baseline_y: float,
    proximity_m: float = 1.0,
) -> Optional[int]:
    window_frames = [f for f in detection_frames if search_start <= f <= net_cross_frame]
    window_frames.sort()
    if len(window_frames) < 3:
        return None

    best_frame: Optional[int] = None
    best_avg: Optional[float] = None

    for i in range(0, len(window_frames) - 2):
        f1 = window_frames[i]
        f2 = window_frames[i + 1]
        f3 = window_frames[i + 2]
        if f1 not in dy_values or f2 not in dy_values or f3 not in dy_values:
            continue
        if f1 not in frame_to_y:
            continue

        y1 = frame_to_y[f1]
        if abs(y1 - baseline_y) > proximity_m:
            continue

        dy1 = abs(dy_values[f1])
        dy2 = abs(dy_values[f2])
        dy3 = abs(dy_values[f3])
        if dy1 < 0.1 or dy2 < 0.1 or dy3 < 0.1:
            continue

        tri_avg = (dy1 + dy2 + dy3) / 3.0
        if tri_avg < 0.15:
            continue

        signed_avg = (dy_values[f1] + dy_values[f2] + dy_values[f3]) / 3.0
        moving_away = (baseline_y == 0.0 and signed_avg > 0) or (baseline_y == 13.4 and signed_avg < 0)
        if not moving_away:
            continue

        if best_avg is None or tri_avg > best_avg or (
            abs(tri_avg - best_avg) < 1e-6 and (best_frame is None or f1 < best_frame)
        ):
            best_avg = tri_avg
            best_frame = f1

    return best_frame


def find_sudden_increase_start_backward(
    net_cross_frame: int,
    search_start: int,
    detection_frames: List[int],
    dy_values: Dict[int, float],
) -> Optional[int]:
    window_frames = [f for f in detection_frames if search_start <= f <= net_cross_frame]
    window_frames.sort()
    if len(window_frames) < 4:
        return None

    for i in range(1, len(window_frames) - 2):
        prev_f = window_frames[i - 1]
        f1 = window_frames[i]
        f2 = window_frames[i + 1]
        f3 = window_frames[i + 2]
        if prev_f not in dy_values or f1 not in dy_values or f2 not in dy_values or f3 not in dy_values:
            continue

        prev_dy = abs(dy_values[prev_f])
        dy1 = abs(dy_values[f1])
        dy2 = abs(dy_values[f2])
        dy3 = abs(dy_values[f3])
        if dy1 < 0.1 or dy2 < 0.1 or dy3 < 0.1:
            continue

        tri_avg = (dy1 + dy2 + dy3) / 3.0
        baseline = max(prev_dy, 1e-6)
        if tri_avg >= 1.5 * baseline:
            return f1

    return None


def _find_steepest_pair_frame(
    search_frames: List[int],
    detection_frames: List[int],
    dy_values: Dict[int, float],
    default_frame: int,
) -> int:
    if not search_frames:
        return default_frame

    df_set = set(detection_frames)
    best_score = -1.0
    best_frame = search_frames[0]
    ordered = sorted(search_frames)

    for idx in range(len(ordered) - 1):
        f = ordered[idx]
        f_next = ordered[idx + 1]
        if f not in df_set or f_next not in df_set:
            continue
        if f not in dy_values or f_next not in dy_values:
            continue

        dy1 = abs(dy_values[f])
        dy2 = abs(dy_values[f_next])
        if dy1 < 0.1 or dy2 < 0.1:
            continue

        score = min(dy1, dy2)
        if score > best_score:
            best_score = score
            best_frame = f_next

    return best_frame if best_score >= 0 else default_frame


def find_steepest_shot_point_backward(
    net_cross_frame: int,
    search_start: int,
    detection_frames: List[int],
    dy_values: Dict[int, float],
) -> int:
    search_frames = [f for f in detection_frames if search_start <= f <= net_cross_frame]
    return _find_steepest_pair_frame(search_frames, detection_frames, dy_values, net_cross_frame)


def find_shot_point_backward(
    net_cross_frame: int,
    search_start: int,
    detection_frames: List[int],
    dy_values: Dict[int, float],
    window_size: int,
) -> int:
    search_frames = [f for f in detection_frames if search_start <= f <= net_cross_frame]
    search_frames.sort(reverse=True)

    if len(search_frames) < window_size:
        return search_frames[-1] if search_frames else net_cross_frame

    initial_direction = get_window_direction(
        net_cross_frame, detection_frames, dy_values, window_size, "backward"
    )

    for i in range(len(search_frames) - window_size + 1):
        current_frame = search_frames[i]
        current_direction = get_window_direction(
            current_frame, detection_frames, dy_values, window_size, "backward"
        )
        if current_direction != 0 and current_direction != initial_direction:
            return current_frame

    return _find_steepest_pair_frame(search_frames, detection_frames, dy_values, net_cross_frame)


def find_shot_point_forward(
    net_cross_frame: int,
    search_end: int,
    detection_frames: List[int],
    dy_values: Dict[int, float],
    window_size: int,
) -> int:
    search_frames = [f for f in detection_frames if net_cross_frame <= f <= search_end]
    search_frames.sort()

    if len(search_frames) < window_size:
        return search_frames[-1] if search_frames else net_cross_frame

    initial_direction = get_window_direction(
        net_cross_frame, detection_frames, dy_values, window_size, "forward"
    )

    for i in range(len(search_frames) - window_size + 1):
        current_frame = search_frames[i]
        current_direction = get_window_direction(
            current_frame, detection_frames, dy_values, window_size, "forward"
        )
        if current_direction != 0 and current_direction != initial_direction:
            return current_frame

    return _find_steepest_pair_frame(search_frames, detection_frames, dy_values, net_cross_frame)


def detect_shot_points(
    best_per_frame: BestPerFrame,
    net_crossings: List[int],
    net_y: float,
    baseline_top_y: float = 13.4,
    baseline_bottom_y: float = 0.0,
    window_size: int = 6,
    max_backward_frames: int = 70,
    max_forward_frames: int = 70,
    min_net_crossing_separation: int = 6,
    bounce_window_size: int = 7,
    bounce_z_threshold: float = 0.3,
) -> List[Dict]:
    """
    For each net crossing frame, estimate shot start (backward) and end (forward), speed, bounce frame.
    """
    if not net_crossings or not best_per_frame:
        return []

    filtered_crossings = [net_crossings[0]]
    for i in range(1, len(net_crossings)):
        if net_crossings[i] - net_crossings[i - 1] >= min_net_crossing_separation:
            filtered_crossings.append(net_crossings[i])

    if not filtered_crossings:
        return []

    detection_frames = sorted(best_per_frame.keys())
    if len(detection_frames) < 2:
        return []

    frame_to_y: Dict[int, float] = {}
    for frame in detection_frames:
        _, (_, y, _) = best_per_frame[frame]
        frame_to_y[frame] = y

    dy_values: Dict[int, float] = {}
    for i in range(1, len(detection_frames)):
        frame_curr = detection_frames[i]
        frame_prev = detection_frames[i - 1]
        if frame_curr in frame_to_y and frame_prev in frame_to_y:
            dy_values[frame_curr] = frame_to_y[frame_curr] - frame_to_y[frame_prev]

    shot_points: List[Dict] = []

    for i, net_cross_frame in enumerate(filtered_crossings):
        if i == 0:
            search_start = max(0, net_cross_frame - max_backward_frames)
        else:
            search_start = filtered_crossings[i - 1]

        if i == len(filtered_crossings) - 1:
            search_end = min(detection_frames[-1], net_cross_frame + max_forward_frames)
        else:
            search_end = filtered_crossings[i + 1]

        if i == 0:
            baseline_y = determine_shot_baseline(
                net_cross_frame,
                detection_frames,
                frame_to_y,
                net_y,
                baseline_top_y,
                baseline_bottom_y,
                points_each_side=2,
            )
            start_shot_frame: Optional[int] = None
            if baseline_y is not None:
                start_shot_frame = find_start_near_baseline_backward(
                    net_cross_frame,
                    search_start,
                    detection_frames,
                    dy_values,
                    frame_to_y,
                    baseline_y,
                    proximity_m=1.0,
                )
                if start_shot_frame is None:
                    start_shot_frame = find_start_near_baseline_backward(
                        net_cross_frame,
                        search_start,
                        detection_frames,
                        dy_values,
                        frame_to_y,
                        baseline_y,
                        proximity_m=2.0,
                    )

            if start_shot_frame is None:
                start_shot_frame = find_sudden_increase_start_backward(
                    net_cross_frame, search_start, detection_frames, dy_values
                )
                if start_shot_frame is None:
                    start_shot_frame = find_steepest_shot_point_backward(
                        net_cross_frame, search_start, detection_frames, dy_values
                    )
        else:
            start_shot_frame = find_shot_point_backward(
                net_cross_frame, search_start, detection_frames, dy_values, window_size
            )

        end_shot_frame = find_shot_point_forward(
            net_cross_frame, search_end, detection_frames, dy_values, window_size
        )

        start_confidence = calculate_confidence(start_shot_frame, detection_frames, dy_values, window_size)
        end_confidence = calculate_confidence(end_shot_frame, detection_frames, dy_values, window_size)
        shot_speed = calculate_shot_speed(start_shot_frame, best_per_frame)
        bounce_frame = detect_bounce_point(
            start_shot_frame, end_shot_frame, best_per_frame, bounce_window_size, bounce_z_threshold
        )

        shot_points.append(
            {
                "net_crossing_frame": net_cross_frame,
                "start_shot_frame": start_shot_frame,
                "end_shot_frame": end_shot_frame,
                "start_confidence": start_confidence,
                "end_confidence": end_confidence,
                "shot_speed_kmh": shot_speed,
                "bounce_frame": bounce_frame,
            }
        )

    return shot_points
