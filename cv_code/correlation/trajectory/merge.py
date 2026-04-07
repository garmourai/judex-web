"""
Realtime-specific trajectory merging module.

Provides merge_trajectories and merge_overlapping_trajectories for the realtime
flow, separated from shot_detection_and_trajectory_analysis.merge_traj.
Expects trajectory objects with .detections as {frame: Detection} (e.g. from
realtime.trajectory_realtime).
"""

import math
import random
import numpy as np


def check_points_following_trend(vec_a, proj_first_frame_b, proj_last_frame_a,
                                  proj_first_point_b, proj_last_point_a):
    """
    Check if the projected points follow the directional trend of vec_a (3D).
    The trend is checked along x or y — whichever has the larger magnitude in vec_a.

    Returns True if the direction of motion (based on frame order) aligns with the
    increase/decrease in the dominant axis (x or y).
    """
    dominant_axis = 'y' if abs(vec_a[1]) >= abs(vec_a[0]) else 'x'
    val_first = proj_first_point_b[1] if dominant_axis == 'y' else proj_first_point_b[0]
    val_last = proj_last_point_a[1] if dominant_axis == 'y' else proj_last_point_a[0]
    direction = vec_a[1] if dominant_axis == 'y' else vec_a[0]

    if proj_first_frame_b > proj_last_frame_a:
        if direction > 0:
            return val_first > val_last
        else:
            return val_first < val_last
    else:
        if direction > 0:
            return val_first < val_last
        else:
            return val_first > val_last


def euclidean_distance(point_a, point_b):
    point_a = np.array(point_a)
    point_b = np.array(point_b)
    return np.linalg.norm(point_a - point_b)


def correct_pca_direction(direction, inliers, frame_indices):
    projections = [(np.dot(pt, direction), frame) for pt, frame in zip(inliers, frame_indices)]
    projections.sort(key=lambda x: x[1])
    values = [p[0] for p in projections]
    diffs = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    score = sum(1 if d > 0 else -1 for d in diffs)
    if score < 0:
        direction *= -1
    return direction


def fit_ransac_line(trajectory, max_iter=100, threshold=0.3):
    detections = trajectory.detections
    if len(detections) < 3:
        return None, None, None, None, None

    sorted_items = sorted(detections.items())
    frame_indices = [f for f, _ in sorted_items]
    points = np.array([[d.x, d.y, d.z] for _, d in sorted_items])
    is_short = len(points) <= 4

    def ransac(points, threshold):
        best_inliers = []
        for _ in range(max_iter):
            if len(points) < 2:
                break
            p1, p2 = random.sample(points.tolist(), 2)
            p1, p2 = np.array(p1), np.array(p2)
            if np.allclose(p1, p2):
                continue
            line_vec = p2 - p1
            line_vec /= np.linalg.norm(line_vec)
            inliers = []
            for pt in points:
                proj_len = np.dot(pt - p1, line_vec)
                proj_pt = p1 + proj_len * line_vec
                dist = np.linalg.norm(pt - proj_pt)
                if dist < threshold:
                    inliers.append(pt)
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
        return best_inliers

    inliers = ransac(points, threshold)
    if len(inliers) < 3 and not is_short:
        inliers = ransac(points, threshold * 1.5)

    if len(inliers) < 2:
        return None, None, None, None, None

    inliers = np.array(inliers)
    center = np.mean(inliers, axis=0)
    cov = np.cov(inliers - center, rowvar=False)
    eig_vals, eig_vecs = np.linalg.eig(cov)
    direction = eig_vecs[:, np.argmax(eig_vals)]
    direction = correct_pca_direction(direction, inliers, frame_indices)

    projections = np.dot(inliers - center, direction)
    min_idx = np.argmin(projections)
    max_idx = np.argmax(projections)
    first_point_proj = center + projections[min_idx] * direction
    last_point_proj = center + projections[max_idx] * direction

    sorted_points = np.array([[d.x, d.y, d.z] for _, d in sorted_items])
    sorted_frames = [f for f, _ in sorted_items]

    def find_closest_frame(projected_point):
        distances = np.linalg.norm(sorted_points - projected_point, axis=1)
        min_idx = np.argmin(distances)
        return sorted_frames[min_idx]

    first_frame = find_closest_frame(first_point_proj)
    last_frame = find_closest_frame(last_point_proj)
    return direction, first_point_proj, last_point_proj, first_frame, last_frame


def fit_ransac_line_overlapping_function(trajectory, max_iter=100, threshold=0.3):
    detections = trajectory.detections
    if len(detections) < 3:
        return None, None, None

    sorted_items = sorted(detections.items())
    frame_indices = [f for f, _ in sorted_items]
    points = np.array([[d.x, d.y, d.z] for _, d in sorted_items])
    is_short = len(points) <= 4

    def ransac(points, threshold):
        best_inliers = []
        for _ in range(max_iter):
            if len(points) < 2:
                break
            p1, p2 = random.sample(points.tolist(), 2)
            p1, p2 = np.array(p1), np.array(p2)
            if np.allclose(p1, p2):
                continue
            line_vec = p2 - p1
            line_vec /= np.linalg.norm(line_vec)
            inliers = []
            for pt in points:
                proj_len = np.dot(pt - p1, line_vec)
                proj_pt = p1 + proj_len * line_vec
                dist = np.linalg.norm(pt - proj_pt)
                if dist < threshold:
                    inliers.append(pt)
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
        return best_inliers

    inliers = ransac(points, threshold)
    if len(inliers) < 3 and not is_short:
        inliers = ransac(points, threshold * 1.5)

    if len(inliers) < 2:
        return None, None, None

    inliers = np.array(inliers)
    center = np.mean(inliers, axis=0)
    cov = np.cov(inliers - center, rowvar=False)
    eig_vals, eig_vecs = np.linalg.eig(cov)
    direction = eig_vecs[:, np.argmax(eig_vals)]
    direction = correct_pca_direction(direction, inliers, frame_indices)

    projections = np.dot(inliers - center, direction)
    min_idx = np.argmin(projections)
    max_idx = np.argmax(projections)
    first_point_proj = center + projections[min_idx] * direction
    last_point_proj = center + projections[max_idx] * direction

    sorted_points = np.array([[d.x, d.y, d.z] for _, d in sorted_items])
    sorted_frames = [f for f, _ in sorted_items]

    def find_closest_frame(projected_point):
        distances = np.linalg.norm(sorted_points - projected_point, axis=1)
        min_idx = np.argmin(distances)
        return sorted_frames[min_idx]

    projected_points = center + np.outer(projections, direction)
    projected_frames = []
    for proj_pt in projected_points:
        distances = np.linalg.norm(sorted_points - proj_pt, axis=1)
        min_idx = np.argmin(distances)
        projected_frames.append(sorted_frames[min_idx])

    return direction, projected_points, projected_frames


def merge_trajectories(stored_trajectories):
    merged = []
    used = set()

    def angle_between(v1, v2):
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        return math.degrees(math.acos(cos_theta))

    trajectories = stored_trajectories[:]

    for i, traj_a in enumerate(trajectories):
        if i in used:
            continue
        if len(traj_a.detections) < 2:
            if i not in used:
                merged.append(traj_a)
            continue

        vec_a, proj_first_point_a, proj_last_point_a, proj_first_frame_a, proj_last_frame_a = fit_ransac_line(traj_a)
        if vec_a is None:
            merged.append(traj_a)
            used.add(i)
            continue

        merged_traj = traj_a
        last_frame_a = max(traj_a.detections.keys())
        first_frame_a = min(traj_a.detections.keys())
        end_a = traj_a.detections[last_frame_a]
        used.add(i)

        for j, traj_b in enumerate(trajectories):
            if j in used or i == j:
                continue
            if len(traj_b.detections) < 2:
                continue

            first_frame_b = min(traj_b.detections.keys())
            last_frame_b = max(traj_b.detections.keys())
            if 0 < (first_frame_b - last_frame_a) <= 3:
                vec_b, proj_first_point_b, proj_last_point_b, proj_first_frame_b, proj_last_frame_b = fit_ransac_line(traj_b)
                if vec_b is None:
                    continue

                angle = angle_between(vec_a, vec_b)
                dist = euclidean_distance(proj_first_point_b, proj_last_point_a) / (proj_first_frame_b - proj_last_frame_a)
                trend_followed = check_points_following_trend(vec_a, proj_first_frame_b, proj_last_frame_a,
                                                              proj_first_point_b, proj_last_point_a)

                if angle < 60 and dist <= 1.5 and trend_followed:
                    merged_traj.detections.update(traj_b.detections)
                    used.add(j)
                    last_frame_a = max(merged_traj.detections.keys())
                    first_frame_a = min(merged_traj.detections.keys())
                    end_a = traj_b.detections[max(traj_b.detections)]
                    vec_a, proj_first_point_a, proj_last_point_a, proj_first_frame_a, proj_last_frame_a = fit_ransac_line(merged_traj)

        merged.append(merged_traj)

    return merged


def merge_overlapping_trajectories(stored_trajectories):
    merged = []
    used = set()

    def angle_between(v1, v2):
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        return math.degrees(math.acos(cos_theta))

    trajectories = stored_trajectories[:]

    for i, traj_a in enumerate(trajectories):
        if i in used:
            continue
        if len(traj_a.detections) < 3:
            if i not in used:
                merged.append(traj_a)
            continue

        vec_a, projected_points_vec_a, projected_frames_vec_a = fit_ransac_line_overlapping_function(traj_a)
        if vec_a is None:
            merged.append(traj_a)
            used.add(i)
            continue
        proj_first_frame_a = projected_frames_vec_a[0]
        proj_last_frame_a = projected_frames_vec_a[-1]
        proj_first_point_a = projected_points_vec_a[0]
        proj_last_point_a = projected_points_vec_a[-1]

        merged_traj = traj_a
        frames_a = set(traj_a.detections.keys())
        last_frame_a = max(traj_a.detections.keys())
        first_frame_a = min(traj_a.detections.keys())
        start_a, end_a = min(frames_a), max(frames_a)
        used.add(i)

        for j, traj_b in enumerate(trajectories):
            if j in used or i == j:
                continue
            if len(traj_b.detections) < 3:
                continue

            frames_b = set(traj_b.detections.keys())
            last_frame_b = max(traj_b.detections.keys())
            first_frame_b = min(traj_b.detections.keys())
            start_b, end_b = min(frames_b), max(frames_b)

            frames_a = sorted(traj_a.detections.keys())
            frames_b = sorted(traj_b.detections.keys())
            start_a, end_a = frames_a[0], frames_a[-1]
            overlap_frames = [f for f in frames_b if start_a <= f <= end_a]

            if 1 <= len(overlap_frames) <= 4:
                vec_b, projected_points_vec_b, projected_frames_vec_b = fit_ransac_line_overlapping_function(traj_b)
                if vec_b is None:
                    continue
                proj_first_frame_b = projected_frames_vec_b[0]
                proj_last_frame_b = projected_frames_vec_b[-1]
                proj_first_point_b = projected_points_vec_b[0]
                proj_last_point_b = projected_points_vec_b[-1]

                total_dist_per_frame = 0
                valid_pair_count = 0
                for ii, frame_a in enumerate(projected_frames_vec_a):
                    for jj, frame_b in enumerate(projected_frames_vec_b):
                        if valid_pair_count > 3:
                            break
                        frame_diff = abs(frame_a - frame_b)
                        if frame_diff in [1, 2]:
                            point_a = projected_points_vec_a[ii]
                            point_b = projected_points_vec_b[jj]
                            dist = euclidean_distance(point_a, point_b)
                            total_dist_per_frame += dist / frame_diff
                            valid_pair_count += 1

                if valid_pair_count > 0:
                    avg_dist_per_frame = total_dist_per_frame / valid_pair_count
                else:
                    avg_dist_per_frame = None

                angle = angle_between(vec_a, vec_b)
                trend_followed = check_points_following_trend(vec_a, proj_first_frame_b, proj_last_frame_a,
                                                             proj_first_point_b, proj_last_point_a)

                if angle < 60 and avg_dist_per_frame is not None and avg_dist_per_frame <= 1.5 and trend_followed:
                    merged_traj.detections.update(traj_b.detections)
                    used.add(j)
                    last_frame_a = max(merged_traj.detections.keys())
                    first_frame_a = min(merged_traj.detections.keys())
                    end_a = traj_b.detections[max(traj_b.detections)]
                    vec_a, projected_points_vec_a, projected_frames_vec_a = fit_ransac_line_overlapping_function(merged_traj)
                    if vec_a is None:
                        continue
                    proj_first_frame_a = projected_frames_vec_a[0]
                    proj_last_frame_a = projected_frames_vec_a[-1]
                    proj_first_point_a = projected_points_vec_a[0]
                    proj_last_point_a = projected_points_vec_a[-1]
                    frames_a = set(merged_traj.detections.keys())
                    start_a, end_a = min(frames_a), max(frames_a)

        merged.append(merged_traj)

    return merged
