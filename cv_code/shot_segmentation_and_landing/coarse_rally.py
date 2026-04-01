"""
Coarse rally windows: split the global frame timeline when gaps exceed a threshold.
Used before per-window net crossing + shot detection.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, TypeVar

T = TypeVar("T")


def segment_coarse_rallies_by_frame_gap(
    sorted_frame_ids: List[int],
    max_frame_gap: int,
) -> List[Tuple[int, int]]:
    """
    Build contiguous frame ranges. When consecutive frames differ by more than
    ``max_frame_gap``, start a new range.

    Returns:
        List of (start_frame_inclusive, end_frame_inclusive).
    """
    if not sorted_frame_ids:
        return []

    out: List[Tuple[int, int]] = []
    start = sorted_frame_ids[0]
    prev = sorted_frame_ids[0]

    for f in sorted_frame_ids[1:]:
        if f - prev > max_frame_gap:
            out.append((start, prev))
            start = f
        prev = f

    out.append((start, prev))
    return out


def subset_best_per_frame(
    best_per_frame: Dict[int, T],
    frame_start: int,
    frame_end: int,
) -> Dict[int, T]:
    """Restrict mapping to frames in [frame_start, frame_end]."""
    return {f: best_per_frame[f] for f in best_per_frame if frame_start <= f <= frame_end}


def merged_rows_to_segment_data(merged_rows: List[dict]) -> List[dict]:
    """
    Rows from merge_filter_stats / read_filter_stats_csv → NetCrossingDetector.process_segment format.
    """
    segment_data = []
    for row in merged_rows:
        segment_data.append(
            {
                "frame_number_after_sync": row["frame_number_after_sync"],
                "picked_x": row["picked_x"],
                "picked_y": row["picked_y"],
                "picked_z": row["picked_z"],
            }
        )
    segment_data.sort(key=lambda r: r["frame_number_after_sync"])
    return segment_data


def merged_rows_to_best_per_frame(merged_rows: List[dict]) -> Dict[int, Tuple[int, Tuple[float, float, float]]]:
    """Single synthetic traj_id 0 for every frame."""
    bpf: Dict[int, Tuple[int, Tuple[float, float, float]]] = {}
    for row in merged_rows:
        f = int(row["frame_number_after_sync"])
        bpf[f] = (
            0,
            (
                float(row["picked_x"]),
                float(row["picked_y"]),
                float(row["picked_z"]),
            ),
        )
    return bpf
