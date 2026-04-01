"""
Filter Stats Merger Module

Functions for reading, merging, and deduplicating trajectory filter stats CSVs
from multiple segment folders.
"""

import os
import csv
import re
from typing import List, Dict, Tuple, Optional


def get_segment_folders_sorted(trajectory_output_dir: str) -> List[Tuple[str, int, int]]:
    """
    Get all segment folders sorted by start frame number.
    
    Args:
        trajectory_output_dir: Path to trajectory_output directory
        
    Returns:
        List of tuples: (folder_path, start_frame, end_frame) sorted by start_frame
    """
    if not os.path.exists(trajectory_output_dir):
        return []
    
    segment_folders = []
    pattern = re.compile(r'^(\d+)_to_(\d+)$')
    
    for folder_name in os.listdir(trajectory_output_dir):
        folder_path = os.path.join(trajectory_output_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
            
        match = pattern.match(folder_name)
        if match:
            start_frame = int(match.group(1))
            end_frame = int(match.group(2))
            segment_folders.append((folder_path, start_frame, end_frame))
    
    # Sort by start frame
    segment_folders.sort(key=lambda x: x[1])
    return segment_folders


def read_filter_stats_csv(csv_path: str) -> List[Dict]:
    """
    Read a single trajectory_filter_stats.csv file.

    Judex correlation CSVs use ``frame_id``; older pipelines used
    ``frame_number_after_sync``. Output rows always use
    ``frame_number_after_sync`` for NetCrossingDetector and merge logic.

    Args:
        csv_path: Path to the CSV file

    Returns:
        List of dictionaries with keys matching CSV columns
    """
    if not os.path.exists(csv_path):
        return []

    rows = []
    try:
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('frame_number_after_sync'):
                    frame = int(row['frame_number_after_sync'])
                elif row.get('frame_id'):
                    frame = int(row['frame_id'])
                else:
                    continue
                rows.append({
                    'frame_number_after_sync': frame,
                    'picked_x': float(row['picked_x']),
                    'picked_y': float(row['picked_y']),
                    'picked_z': float(row['picked_z']),
                    'num_points_dropped': int(row['num_points_dropped']),
                    'num_points_before': int(row['num_points_before']),
                })
    except Exception as e:
        print(f"  ⚠️  Error reading {csv_path}: {e}")
        return []

    return rows


def merge_filter_stats(
    trajectory_output_dir: str,
    verbose: bool = True
) -> Tuple[List[Dict], List[Dict]]:
    """
    Merge all trajectory_filter_stats.csv files from segment folders.
    
    Processes folders in order (by start frame). If the same frame_number_after_sync
    appears in multiple segments, keeps the last occurrence and reports duplicates.
    
    Args:
        trajectory_output_dir: Path to trajectory_output directory
        verbose: Whether to print progress messages
        
    Returns:
        Tuple of:
            - merged_rows: List of unique rows (by frame_number_after_sync), sorted by frame
            - duplicates: List of duplicate entries that were overwritten
    """
    segment_folders = get_segment_folders_sorted(trajectory_output_dir)
    
    if not segment_folders:
        if verbose:
            print(f"  ⚠️  No segment folders found in {trajectory_output_dir}")
        return [], []
    
    if verbose:
        print(f"  📂 Found {len(segment_folders)} segment folders")
    
    # Use dict to track entries by frame (keeps last occurrence)
    merged_dict: Dict[int, Dict] = {}
    duplicates: List[Dict] = []
    
    files_read = 0
    files_missing = 0
    total_rows = 0
    
    for folder_path, start_frame, end_frame in segment_folders:
        csv_path = os.path.join(folder_path, 'trajectory_filter_stats.csv')
        
        if not os.path.exists(csv_path):
            files_missing += 1
            if verbose:
                print(f"    ⏭️  Segment [{start_frame}-{end_frame}]: no filter stats CSV (likely no trajectories)")
            continue
        
        rows = read_filter_stats_csv(csv_path)
        files_read += 1
        total_rows += len(rows)
        
        if verbose:
            print(f"    ✅ Segment [{start_frame}-{end_frame}]: {len(rows)} rows")
        
        for row in rows:
            frame = row['frame_number_after_sync']
            
            if frame in merged_dict:
                # Duplicate found - report and keep the new one (last wins)
                old_row = merged_dict[frame].copy()
                old_row['replaced_by_segment'] = f"{start_frame}_to_{end_frame}"
                duplicates.append(old_row)
                
                if verbose:
                    print(f"      ⚠️  Duplicate frame {frame}: replacing with entry from segment [{start_frame}-{end_frame}]")
            
            # Add source segment info
            row['source_segment'] = f"{start_frame}_to_{end_frame}"
            merged_dict[frame] = row
    
    # Sort by frame number
    merged_rows = [merged_dict[frame] for frame in sorted(merged_dict.keys())]
    
    if verbose:
        print(f"\n  📊 Summary:")
        print(f"      Files read: {files_read}")
        print(f"      Files missing: {files_missing}")
        print(f"      Total rows processed: {total_rows}")
        print(f"      Unique frames: {len(merged_rows)}")
        print(f"      Duplicates replaced: {len(duplicates)}")
    
    return merged_rows, duplicates


def write_merged_csv(
    merged_rows: List[Dict],
    output_path: str,
    include_source_segment: bool = True
) -> None:
    """
    Write merged filter stats to a CSV file.
    
    Args:
        merged_rows: List of merged row dictionaries
        output_path: Path to output CSV file
        include_source_segment: Whether to include source_segment column
    """
    if not merged_rows:
        print(f"  ⚠️  No rows to write")
        return
    
    fieldnames = [
        'frame_number_after_sync',
        'picked_x',
        'picked_y',
        'picked_z',
        'num_points_dropped',
        'num_points_before',
    ]
    
    if include_source_segment:
        fieldnames.append('source_segment')
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(merged_rows)
    
    print(f"  ✅ Wrote {len(merged_rows)} rows to {output_path}")


def write_duplicates_report(
    duplicates: List[Dict],
    output_path: str
) -> None:
    """
    Write a report of duplicate entries that were replaced.
    
    Args:
        duplicates: List of duplicate row dictionaries
        output_path: Path to output CSV file
    """
    if not duplicates:
        return
    
    fieldnames = [
        'frame_number_after_sync',
        'picked_x',
        'picked_y',
        'picked_z',
        'num_points_dropped',
        'num_points_before',
        'source_segment',
        'replaced_by_segment',
    ]
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(duplicates)
    
    print(f"  📝 Wrote {len(duplicates)} duplicate entries to {output_path}")
