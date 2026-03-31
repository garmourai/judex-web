"""
Utility functions for correlation worker.

This module contains helper functions used by correlation_worker.py
to keep the main worker file cleaner and more maintainable.
"""

import os
from typing import Optional, Tuple, List, Dict


def parse_first_field_int(line: str) -> Optional[int]:
    """First CSV field as int (stops at first comma outside brackets)."""
    line = line.strip()
    if not line:
        return None
    bracket_depth = 0
    for i, ch in enumerate(line):
        if ch == "[":
            bracket_depth += 1
        elif ch == "]":
            bracket_depth -= 1
        elif ch == "," and bracket_depth == 0:
            try:
                return int(line[:i].strip())
            except ValueError:
                return None
    try:
        return int(line.strip())
    except ValueError:
        return None


def parse_first_two_fields_ints(line: str) -> Optional[Tuple[int, int]]:
    """
    Parse first two integer fields from a CSV row (handles extra trailing columns).
    We only need the first two integer fields and we must tolerate commas
    inside bracketed X/Y list fields later in the row.
    
    Args:
        line: CSV line string
        
    Returns:
        Tuple of two leading ints or None if parsing fails
    """
    line = line.strip()
    if not line:
        return None
    fields: List[str] = []
    current = ""
    bracket_depth = 0
    for ch in line:
        if ch == "[":
            bracket_depth += 1
            current += ch
        elif ch == "]":
            bracket_depth -= 1
            current += ch
        elif ch == "," and bracket_depth == 0:
            fields.append(current.strip())
            current = ""
            if len(fields) >= 2:
                break
        else:
            current += ch
    if len(fields) < 2:
        # If we broke early after 2 commas, fields is already populated.
        # Otherwise, append what's left and validate length.
        if current:
            fields.append(current.strip())
    if len(fields) < 2:
        return None
    try:
        return int(fields[0]), int(fields[1])
    except ValueError:
        return None


def parse_frame_pair_from_dist_row(line: str) -> Optional[Tuple[int, int]]:
    """
    From a dist_tracker CSV row: (frame_id, camera_stream_index).
    Four-column rows use identity (frame_id, frame_id).
    """
    two = parse_first_two_fields_ints(line)
    if two is not None:
        return two
    one = parse_first_field_int(line)
    if one is None:
        return None
    return (one, one)


def load_fid_to_stream_from_dist_tracker_csv(csv_path: str) -> Dict[int, int]:
    """Build frame_id -> camera_stream_index from dist_tracker CSV (no sidecar file)."""
    out: Dict[int, int] = {}
    if not os.path.exists(csv_path):
        return out
    try:
        with open(csv_path, "r") as f:
            lines = f.readlines()
    except OSError:
        return out
    if len(lines) <= 1:
        return out
    for line in lines[1:]:
        p = parse_frame_pair_from_dist_row(line)
        if p is None:
            continue
        fid, stream = p
        out[fid] = stream
    return out
