"""CSV mapping triangulated 3D points to trajectory IDs."""

import csv
import os
from ast import literal_eval
from typing import List, Optional, Set, Tuple


def create_point_trajectory_mapping_csv(
    segment: Tuple[int, int],
    output_dir: str,
    stored_trajectories: List,
    global_traj_offset: int = 0,
    kept_global_traj_ids: Optional[Set[int]] = None,
) -> Optional[str]:
    """
    Create a CSV file mapping each 3D point from triangulation to its trajectory ID.

    Args:
        segment: Tuple of (start_frame, end_frame)
        output_dir: Output directory where tracker.csv files are stored
        stored_trajectories: Merged trajectories **before** ``get_best_point_each_frame``
            so every hypothesis that survived merging gets a stable numeric ID.
        global_traj_offset: Offset to add to trajectory IDs (for multi-segment continuity)
        kept_global_traj_ids: Global IDs still present after ``get_best_point_each_frame``;
            used to fill ``Kept_After_Filter`` (1 = kept, 0 = dropped by that filter).
            None if there were no merged trajectories this segment.

    Returns:
        Path to the created CSV file
    """
    start_frame, end_frame = segment

    # Path to triangulated 3D points (single growing file)
    tracker_csv_path = os.path.join(output_dir, "tracker.csv")

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
        print(
            f"[CorrelationWorker]    ⚠️  Warning: Found {len(duplicate_keys)} duplicate point mappings (point in multiple trajectories)"
        )

    # Output CSV path (single growing file)
    output_csv_path = os.path.join(output_dir, "points_trajectory_mapping.csv")

    # Read tracker.csv and create mapping CSV
    # List of tuples:
    # (frame, x, y, z, f1, f2, epi, reproj, temporal)
    all_points = []
    frame_point_counts = {}  # Track point counts per frame for debugging

    with open(tracker_csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_num = int(row["Frame"])
            if frame_num < start_frame or frame_num > end_frame:
                continue
            visibility = int(row["Visibility"])

            if visibility != 1:
                continue

            # Parse X, Y, Z coordinates (they are in format "[x1, x2, ...]")
            x_values = literal_eval(row["X"])
            y_values = literal_eval(row["Y"])
            z_values = literal_eval(row["Z"])

            # Extract all 3D coordinates
            num_points = min(len(x_values), len(y_values), len(z_values))
            f1_values = literal_eval(row.get("Point_Costs_Formula1", "[]"))
            f2_values = literal_eval(row.get("Point_Costs_Formula2", "[]"))
            epi_values = literal_eval(row.get("Point_Costs_Epipolar", "[]"))
            reproj_values = literal_eval(row.get("Point_Costs_Reprojection", "[]"))
            temporal_values = literal_eval(row.get("Point_Costs_Temporal", "[]"))
            points_added_this_frame = 0

            for i in range(num_points):
                x_3d = float(x_values[i])
                y_3d = float(y_values[i])
                z_3d = float(z_values[i])

                # Skip (0, 0, 0) points (invisible detections)
                if x_3d == 0 and y_3d == 0 and z_3d == 0:
                    continue

                cost_f1 = float(f1_values[i]) if i < len(f1_values) else None
                cost_f2 = float(f2_values[i]) if i < len(f2_values) else None
                cost_epi = float(epi_values[i]) if i < len(epi_values) else None
                cost_reproj = float(reproj_values[i]) if i < len(reproj_values) else None
                cost_temporal = float(temporal_values[i]) if i < len(temporal_values) else None
                all_points.append(
                    (
                        frame_num,
                        x_3d,
                        y_3d,
                        z_3d,
                        cost_f1,
                        cost_f2,
                        cost_epi,
                        cost_reproj,
                        cost_temporal,
                    )
                )
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

    # Append mapping rows to global CSV
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    file_exists = os.path.exists(output_csv_path) and os.path.getsize(output_csv_path) > 0
    with open(output_csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                [
                    "Frame",
                    "X",
                    "Y",
                    "Z",
                    "Trajectory_ID",
                    "Kept_After_Filter",
                    "Cost_Formula1",
                    "Cost_Formula2",
                    "Cost_Epipolar",
                    "Cost_Reprojection",
                    "Cost_Temporal",
                ]
            )

        trajectory_points_count = 0
        removed_points_count = 0

        for frame_num, x, y, z, cost_f1, cost_f2, cost_epi, cost_reproj, cost_temporal in all_points:
            # Try to match point to trajectory
            key = (frame_num, round(x, 4), round(y, 4), round(z, 4))
            traj_id = trajectory_point_map.get(key)
            f1_str = f"{cost_f1:.6f}" if cost_f1 is not None else ""
            f2_str = f"{cost_f2:.6f}" if cost_f2 is not None else ""
            epi_str = f"{cost_epi:.6f}" if cost_epi is not None else ""
            reproj_str = f"{cost_reproj:.6f}" if cost_reproj is not None else ""
            temporal_str = f"{cost_temporal:.6f}" if cost_temporal is not None else ""

            if traj_id is not None:
                if kept_global_traj_ids is None:
                    kept_str = ""
                else:
                    kept_str = "1" if traj_id in kept_global_traj_ids else "0"
                writer.writerow(
                    [
                        frame_num,
                        f"{x:.6f}",
                        f"{y:.6f}",
                        f"{z:.6f}",
                        traj_id,
                        kept_str,
                        f1_str,
                        f2_str,
                        epi_str,
                        reproj_str,
                        temporal_str,
                    ]
                )
                trajectory_points_count += 1
            else:
                kept_str = "" if kept_global_traj_ids is None else ""
                writer.writerow(
                    [
                        frame_num,
                        f"{x:.6f}",
                        f"{y:.6f}",
                        f"{z:.6f}",
                        -1,
                        kept_str,
                        f1_str,
                        f2_str,
                        epi_str,
                        reproj_str,
                        temporal_str,
                    ]
                )
                removed_points_count += 1

        print(f"[CorrelationWorker]    📊 Point mapping CSV created: {output_csv_path}")
        print(
            f"[CorrelationWorker]       Matched to trajectory (numeric ID): {trajectory_points_count}, "
            f"Unmatched (Trajectory_ID=-1): {removed_points_count}"
        )

    return output_csv_path
