"""
Append-only segment index written next to HLS segments (same dir as playlist.m3u8).

Written when each seg_*.ts frame count is first known (triplet reader path).
Loaded by M3U8SegmentReader on startup so bounce / later reads jump to segments
without re-probing the whole playlist history.
"""

from __future__ import annotations

import csv
import os
from typing import List, Optional, Tuple

MANIFEST_FILENAME = "hls_segment_frame_index.csv"


def manifest_path(segments_dir: str) -> str:
    return os.path.join(segments_dir, MANIFEST_FILENAME)


def append_segment_manifest_row(
    segments_dir: str,
    segment_index: int,
    cumulative_start_frame: int,
    frame_count: int,
) -> None:
    """Append one row when segment `segment_index` is finalized (thread: reader thread)."""
    os.makedirs(segments_dir, exist_ok=True)
    path = manifest_path(segments_dir)
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    seg_basename = f"seg_{segment_index:05d}.ts"
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(
                [
                    "segment_index",
                    "seg_basename",
                    "cumulative_start_frame",
                    "frame_count",
                ]
            )
        w.writerow(
            [segment_index, seg_basename, cumulative_start_frame, frame_count]
        )
        f.flush()


def load_segment_manifest(
    segments_dir: str,
) -> Optional[Tuple[List[int], List[int]]]:
    """
    Load cumulative_offsets and frame_counts for segments 0..N-1 if the file
    is present, contiguous, and self-consistent. Otherwise return None.
    """
    path = manifest_path(segments_dir)
    if not os.path.exists(path):
        return None
    by_seg: dict[int, Tuple[int, int]] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return None
        for row in reader:
            try:
                si = int(row["segment_index"])
                cs = int(row["cumulative_start_frame"])
                fc = int(row["frame_count"])
            except (KeyError, ValueError, TypeError):
                continue
            by_seg[si] = (cs, fc)
    if not by_seg:
        return None
    max_seg = max(by_seg.keys())
    cumulative_offsets: List[int] = []
    frame_counts: List[int] = []
    for i in range(max_seg + 1):
        if i not in by_seg:
            return None
        cs, fc = by_seg[i]
        cumulative_offsets.append(cs)
        frame_counts.append(fc)
    if cumulative_offsets[0] != 0:
        return None
    for i in range(len(cumulative_offsets) - 1):
        if cumulative_offsets[i] + frame_counts[i] != cumulative_offsets[i + 1]:
            return None
    return cumulative_offsets, frame_counts


def global_frame_to_segment(
    cumulative_offsets: List[int],
    frame_counts: List[int],
    global_frame_index: int,
) -> Optional[Tuple[int, int]]:
    """
    Return (segment_index, local_frame_offset) for global_frame_index, or None if OOB.
    """
    for seg_idx in range(len(cumulative_offsets)):
        start = cumulative_offsets[seg_idx]
        end = start + frame_counts[seg_idx] - 1
        if start <= global_frame_index <= end:
            return seg_idx, global_frame_index - start
    return None
