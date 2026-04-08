import json
import os
import sys
import time
import threading
import csv
from ast import literal_eval
from typing import Optional, Tuple, List, Dict, Any

from .visualization import (
    append_stitched_segment_to_video,
    load_correlated_pairs_from_tracker_csvs_realtime,
    load_selected_pair_costs_from_match_decisions,
    create_visualization_from_triangulation,
)
from .trajectory import (
    create_trajectories_realtime,
    merge_trajectories,
    merge_overlapping_trajectories,
    get_best_point_each_frame,
    LAST_FRAMES_TO_SKIP,
)


def _load_tracker_points_costs_by_frame(
    tracker_path: str,
) -> Dict[int, List[Tuple[Tuple[float, float, float], Dict[str, float]]]]:
    """
    Map Original_Frame -> list of (xyz, cost dict) for each valid triangulated point (Visibility==1, non-null).
    Used only when writing trajectory_selection.jsonl so downstream consumers need not read tracker.csv.
    """
    result: Dict[int, List[Tuple[Tuple[float, float, float], Dict[str, float]]]] = {}
    if not os.path.exists(tracker_path):
        return result
    with open(tracker_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                of = int(row["Original_Frame"])
            except (KeyError, ValueError, TypeError):
                continue
            try:
                visibility = int(row["Visibility"])
            except (ValueError, TypeError):
                continue
            if visibility != 1:
                continue
            try:
                x_values = literal_eval(row["X"])
                y_values = literal_eval(row["Y"])
                z_values = literal_eval(row["Z"])
            except (ValueError, SyntaxError):
                continue
            f1_values = literal_eval(row.get("Point_Costs_Formula1", "[]"))
            f2_values = literal_eval(row.get("Point_Costs_Formula2", "[]"))
            epi_values = literal_eval(row.get("Point_Costs_Epipolar", "[]"))
            reproj_values = literal_eval(row.get("Point_Costs_Reprojection", "[]"))
            temporal_values = literal_eval(row.get("Point_Costs_Temporal", "[]"))
            n = min(len(x_values), len(y_values), len(z_values))
            for i in range(n):
                x_3d = float(x_values[i])
                y_3d = float(y_values[i])
                z_3d = float(z_values[i])
                if x_3d == 0 and y_3d == 0 and z_3d == 0:
                    continue
                costs: Dict[str, float] = {}
                if i < len(f1_values):
                    costs["f1"] = float(f1_values[i])
                if i < len(f2_values):
                    costs["f2"] = float(f2_values[i])
                if i < len(epi_values):
                    costs["epi"] = float(epi_values[i])
                if i < len(reproj_values):
                    costs["reproj"] = float(reproj_values[i])
                if i < len(temporal_values):
                    costs["temp"] = float(temporal_values[i])
                result.setdefault(of, []).append(((x_3d, y_3d, z_3d), costs))
    return result


def _nearest_tracker_costs(
    frame_points: List[Tuple[Tuple[float, float, float], Dict[str, float]]],
    x: float,
    y: float,
    z: float,
) -> Optional[Dict[str, float]]:
    """Match a trajectory point to the closest tracker row point (same segment / float noise)."""
    if not frame_points:
        return None
    best_costs: Optional[Dict[str, float]] = None
    best_d = float("inf")
    for (cx, cy, cz), costs in frame_points:
        d = (cx - x) ** 2 + (cy - y) ** 2 + (cz - z) ** 2
        if d < best_d:
            best_d = d
            best_costs = costs
    if best_d > 1e-6:
        return None
    return best_costs


def _append_trajectory_selection_jsonl(
    correlation_output_dir: str,
    segment: Tuple[int, int],
    frame_decisions: List[Dict[str, Any]],
) -> Optional[str]:
    """
    Append one JSON object per line to trajectory_selection.jsonl (authoritative post-select_best output).

    Each line includes frame_id, selected_trajectory_id (incremental track_id from Trajectory after merge),
    current_selected_point, current_ignored_points, active_trajectories (with is_selected and optional
    costs copied from tracker.csv at write time), and counters. Visualization reads only this file (not tracker.csv).
    """
    if not frame_decisions:
        return None
    tracker_path = os.path.join(correlation_output_dir, "tracker.csv")
    costs_by_frame = _load_tracker_points_costs_by_frame(tracker_path)
    path = os.path.join(correlation_output_dir, "trajectory_selection.jsonl")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for rec in frame_decisions:
            selected_tid = int(rec["selected_track_id"])
            frame_num = int(rec["frame"])
            frame_pts = costs_by_frame.get(frame_num, [])
            active = []
            for c in rec["active_candidates"]:
                gid = int(c["track_id"])
                xf, yf, zf = float(c["x"]), float(c["y"]), float(c["z"])
                matched = _nearest_tracker_costs(frame_pts, xf, yf, zf) or {}
                active.append(
                    {
                        "trajectory_id": gid,
                        "x": c["x"],
                        "y": c["y"],
                        "z": c["z"],
                        "is_selected": c["is_selected"],
                        "costs": matched,
                    }
                )
            selected = next((a for a in active if a["is_selected"]), None)
            ignored = [a for a in active if not a["is_selected"]]
            out = {
                "segment": [segment[0], segment[1]],
                "frame_id": frame_num,
                "selected_trajectory_id": selected_tid,
                "current_selected_point": (
                    {
                        "x": selected["x"],
                        "y": selected["y"],
                        "z": selected["z"],
                        "costs": selected.get("costs", {}),
                    }
                    if selected
                    else None
                ),
                "current_ignored_points": [
                    {
                        "trajectory_id": x["trajectory_id"],
                        "x": x["x"],
                        "y": x["y"],
                        "z": x["z"],
                        "costs": x.get("costs", {}),
                    }
                    for x in ignored
                ],
                "active_trajectories": active,
                "num_points_before": rec["num_points_before"],
                "num_points_dropped": rec["num_points_dropped"],
            }
            f.write(json.dumps(out) + "\n")
    return path


def correlation_worker_wrapper(*args, **kwargs):
    """Wrapper to call correlation worker."""
    # Extract profiler from kwargs if present, or from args if it's the last positional arg
    correlation_worker(*args, **kwargs)


def correlation_worker(
    camera_1_csv_path: str,
    camera_2_csv_path: str,
    camera_1_id: str,
    camera_2_id: str,
    camera_1_cam_path: str,
    camera_2_cam_path: str,
    camera_1_video_path: str,
    camera_2_video_path: str,
        output_dir: str,
        frame_sync_info_path: str,  # Unused in realtime flow (shared global frame id in CSVs)
        stop_event: threading.Event,
        profiler=None,
        check_interval: float = 0.05,  # Check for new frames every 0.05 seconds
        original_buffer=None,  # OriginalFrameBuffer: shared cam1+cam2 frames for viz (triplet pipeline)
        enable_visualization: bool = False,  # Only create videos when True
        enable_stitched_visualization: bool = False,  # Only create stitched correlation video when True
        inference_done_event: Optional[threading.Event] = None,  # Set when inference is finished
        chunk_size: int = 96,  # Batch size for OriginalFrameBuffer clear_batch (judex pipeline)
        original_frame_width: Optional[int] = None,
        original_frame_height: Optional[int] = None,
        force_stop_event: Optional[threading.Event] = None,
):
    """
    Worker thread that continuously monitors dist_tracker.csv files and performs
    pairwise correlation for ready frame segments.
    
    Args:
        camera_1_csv_path: Path to camera 1 dist_tracker.csv
        camera_2_csv_path: Path to camera 2 dist_tracker.csv
        camera_1_id: Camera 1 identifier
        camera_2_id: Camera 2 identifier
        camera_1_cam_path: Path to camera 1 calibration pickle file
        camera_2_cam_path: Path to camera 2 calibration pickle file
        camera_1_video_path: Path to camera 1 video file
        camera_2_video_path: Path to camera 2 video file
        output_dir: Output directory for correlation results
        frame_sync_info_path: Path to frame synchronization info JSON (unused in realtime flow)
        stop_event: Event to signal thread to stop
        profiler: TimeProfiler instance for recording timing metrics
        check_interval: Time in seconds between checks for new frames
        original_frame_width: Camera-1 original frame width from PipelineConfig
        original_frame_height: Camera-1 original frame height from PipelineConfig
    """
    sys.stdout.flush()
    print("[CorrelationWorker] " + "=" * 60, flush=True)
    print("[CorrelationWorker] 🔗 Correlation worker thread started", flush=True)
    print("[CorrelationWorker] " + "=" * 60, flush=True)
    print(f"[CorrelationWorker]    📁 Camera 1 CSV: {camera_1_csv_path}", flush=True)
    print(f"[CorrelationWorker]    📁 Camera 2 CSV: {camera_2_csv_path}", flush=True)
    print(f"[CorrelationWorker]    📁 Output dir: {output_dir}", flush=True)
    print(f"[CorrelationWorker]    📁 Camera 1 object: {camera_1_cam_path}", flush=True)
    print(f"[CorrelationWorker]    📁 Camera 2 object: {camera_2_cam_path}", flush=True)
    print(f"[CorrelationWorker]    📁 Camera 1 video: {camera_1_video_path}", flush=True)
    print(f"[CorrelationWorker]    📁 Camera 2 video: {camera_2_video_path}", flush=True)
    sys.stdout.flush()
    if original_frame_width is None or original_frame_height is None:
        print(
            "[CorrelationWorker]    ℹ️  camera_1_original_frame_width/height not set in pipeline; "
            "will read from OriginalFrameBuffer once first batch is decoded.",
            flush=True,
        )

    # Import here to avoid circular imports
    print("[CorrelationWorker]    🔄 Importing do_pairwise_correlation_realtime...", flush=True)
    sys.stdout.flush()
    from .realtime_pairwise_correlation import do_pairwise_correlation_realtime
    print("[CorrelationWorker]    ✅ Successfully imported do_pairwise_correlation_realtime", flush=True)
    sys.stdout.flush()
    
    # Track processed segments to avoid duplicates
    processed_segments = set()

    # Track last processed frame (common for both cameras)
    last_processed_frame = -1
    
    # Track last segment formation time for profiling
    last_segment_formation_time = None
    
    # Track trajectory handoff context between segments
    trajectory_context = None
    # Monotonic track_id across segments (assigned after merge); must init before any += in loop.
    next_track_id = 0

    # World points file for trajectory bounds checking (repo pickleball court outline)
    _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    world_points_file = os.path.join(_repo_root, "tools", "pickleball_calib", "worldpickleball.txt")
    correlation_output_dir = os.path.join(output_dir, "correlation")
    os.makedirs(correlation_output_dir, exist_ok=True)
    
    check_count = 0
    last_progress_time = time.time()
    correlation_batch = 0  # increments each time we successfully process a set of segments
    last_busy_end_time = time.time()  # for correlation_wait_for_ready_s (first segment t_wait_ready ≈ 0)
    while not stop_event.is_set():
        check_count += 1
        if check_count % 10 == 0:  # Print status every 10 checks (every 0.5 seconds now)
            print(f"[CorrelationWorker]    🔍 Correlation worker: Check #{check_count} (waiting for CSV files or new frames...)")
        
        # Check if CSV files exist
        csv1_exists = os.path.exists(camera_1_csv_path)
        csv2_exists = os.path.exists(camera_2_csv_path)
        if not csv1_exists or not csv2_exists:
            if check_count <= 3:  # Print first few times
                print(f"[CorrelationWorker]    ⏳ Waiting for CSV files... (cam1: {csv1_exists}, cam2: {csv2_exists})")
            time.sleep(check_interval)
            continue

        # Read CSV files once for frame numbers (Frame column)
        # Note: CSV has format: Frame,X,Y,Visibility where X and Y can be lists like "[300, 500, 0]"
        # The CSV is written without quotes, so pandas will fail to parse due to commas in lists
        # We'll read just the Frame column by parsing line by line
        frames_cam1 = []
        frames_cam2 = []
        
        # Read Camera 1 CSV once
        with open(camera_1_csv_path, 'r') as f:
            lines_cam1 = f.readlines()

        # Extract frame numbers from Camera 1 CSV
        if len(lines_cam1) > 1:  # Skip header
            for line in lines_cam1[1:]:  # Skip header line
                # Frame is the first field before the first comma
                frame_num = int(line.split(',')[0].strip())
                frames_cam1.append(frame_num)

        # Read Camera 2 CSV once
        with open(camera_2_csv_path, 'r') as f:
            lines_cam2 = f.readlines()

        # Extract frame numbers from Camera 2 CSV
        if len(lines_cam2) > 1:  # Skip header
            for line in lines_cam2[1:]:  # Skip header line
                # Frame is the first field before the first comma
                frame_num = int(line.split(',')[0].strip())
                frames_cam2.append(frame_num)
        
        # Get frame ranges from both cameras
        if len(frames_cam1) == 0 or len(frames_cam2) == 0:
            if check_count <= 3:
                print(f"[CorrelationWorker]    ⏳ CSV files are empty (cam1: {len(frames_cam1)} frames, cam2: {len(frames_cam2)} frames)", flush=True)
                sys.stdout.flush()
            time.sleep(check_interval)
            continue
        
        frames_cam1 = sorted(set(frames_cam1))  # Remove duplicates and sort
        frames_cam2 = sorted(set(frames_cam2))  # Remove duplicates and sort
        
        if len(frames_cam1) == 0 or len(frames_cam2) == 0:
            if check_count <= 3:
                print(f"[CorrelationWorker]    ⏳ No frames found in CSV (cam1: {len(frames_cam1)}, cam2: {len(frames_cam2)})")
            time.sleep(check_interval)
            continue
        
        max_frame_cam1 = max(frames_cam1)
        max_frame_cam2 = max(frames_cam2)
        
        # Determine the maximum frame that both cameras have processed
        max_common_frame = min(max_frame_cam1, max_frame_cam2)
        
        if check_count % 10 == 0:  # Print status periodically
            print(f"[CorrelationWorker]    📊 CSV status: cam1 max={max_frame_cam1}, cam2 max={max_frame_cam2}, common={max_common_frame}, last_processed={last_processed_frame}")
        
        # Check if we have new frames to process
        if max_common_frame <= last_processed_frame:
            # No new frames available - check if inference is done
            if inference_done_event is not None and inference_done_event.is_set():
                # Inference is done and no new frames - all work is complete
                # Give a short grace period in case CSV flushes lag behind inference shutdown
                if (time.time() - last_progress_time) > 2.0:
                    print("[CorrelationWorker] ✅ Correlation worker: inference done and no new frames — exiting")
                    break
                else:
                    # Grace period not elapsed yet, wait a bit more
                    if stop_event.is_set():
                        break
                    time.sleep(check_interval)
                    continue
            else:
                # Inference still active, but if we haven't seen any new frames for a while,
                # print a periodic warning (but keep waiting - no timeout/exit)
                if stop_event.is_set():
                    print("[CorrelationWorker] 🛑 Stop event set — exiting correlation worker")
                    break
                idle_duration = time.time() - last_progress_time
                if idle_duration >= 20.0 and int(idle_duration) % 20 == 0:
                    print(f"[CorrelationWorker] ⚠️ Correlation worker: waiting for new frames (idle for {int(idle_duration):.0f}s, cam1 max={max_frame_cam1}, cam2 max={max_frame_cam2}, last_processed={last_processed_frame})...")

                # Keep waiting for more frames.
                time.sleep(check_interval)
                continue
        
        # Determine which frames are ready for correlation.
        # We take ALL accumulated new frames since last processing:
        # start from the last processed frame + 1 up to max_common_frame.
        start_frame = max(0, last_processed_frame + 1)
        end_frame = max_common_frame
        
        if end_frame <= start_frame:
            time.sleep(check_interval)
            continue
        
        # Form exactly one segment spanning all currently-available common frames.
        # This processes everything from the first unprocessed frame up to max_common_frame.
        segment = (start_frame, end_frame)
        segment_key = f"{start_frame}_{end_frame}"
        if segment_key in processed_segments:
            # No new work for this exact range; wait for more frames.
            time.sleep(check_interval)
            continue
        
        # Process ready segment — timing model (seconds; accounted + overhead ≈ wall):
        #   correlation_segment_pairwise_s   — do_pairwise_correlation_realtime
        #   correlation_segment_trajectory_s   — create/merge/filter + mapping CSV
        #   correlation_segment_viz_overlay_s — cam1 overlay MP4 chunks (if enabled)
        #   correlation_segment_viz_stitched_s — side-by-side stitched MP4 (if enabled)
        #   correlation_segment_accounted_s    — sum of the four above
        #   correlation_segment_overhead_s     — wall − accounted (cleanup, I/O, etc.)
        #   correlation_complete_segment_time  — wall time for full batch
        #   correlation_wait_for_ready_s     — idle before this segment (poll/sleep since last segment end)
        t_wait_ready = time.time() - last_busy_end_time
        if profiler:
            profiler.record(
                "correlation_wait_for_ready_s",
                t_wait_ready,
                write_immediately=True,
                batch=correlation_batch,
                metadata=f"segment={start_frame}-{end_frame}",
            )
        complete_segment_start_time = time.time()
        t_pairwise = 0.0
        t_trajectory = 0.0
        t_viz_overlay = 0.0
        t_viz_stitched = 0.0
        
        # Time profiling: time since last segment formation
        current_time = time.time()
        if last_segment_formation_time is not None and profiler:
            time_since_last_segment = current_time - last_segment_formation_time
            profiler.record(
                "correlation_time_between_segments",
                time_since_last_segment,
                write_immediately=False,
                batch=correlation_batch,
            )
        
        # Record segment size (inclusive range: end_frame - start_frame + 1)
        segment_size_frames = end_frame - start_frame + 1
        if profiler:
            profiler.record(
                "correlation_segment_size",
                segment_size_frames,
                write_immediately=False,
                metadata=f"segment={start_frame}-{end_frame}",
                batch=correlation_batch,
            )
        
        # Update last segment formation time
        last_segment_formation_time = current_time
        
        seg_meta = f"segment={start_frame}-{end_frame},frames={segment_size_frames}"
        print(
            f"[CorrelationWorker] Batch {correlation_batch} — start {seg_meta}",
            flush=True,
        )
        if last_segment_formation_time is not None:
            if stop_event.is_set() or (
                force_stop_event is not None and force_stop_event.is_set()
            ):
                print(
                    "[CorrelationWorker] 🛑 Stop before segment processing — exiting",
                    flush=True,
                )
                break
            correlation_start_time = time.time()

            do_pairwise_correlation_realtime(
                camera_1_cam_path=camera_1_cam_path,
                camera_2_cam_path=camera_2_cam_path,
                camera_1_id=camera_1_id,
                camera_2_id=camera_2_id,
                camera_1_video_path=camera_1_video_path,
                camera_2_video_path=camera_2_video_path,
                output_dir=correlation_output_dir,
                frame_segments=[segment],
                camera_1_tracker_csv=camera_1_csv_path,
                camera_2_tracker_csv=camera_2_csv_path,
            )
            
            correlation_duration = time.time() - correlation_start_time
            t_pairwise = correlation_duration
            total_frames_processed = segment_size_frames

            if profiler:
                profiler.record(
                    "correlation_segment_pairwise_s",
                    t_pairwise,
                    write_immediately=True,
                    batch=correlation_batch,
                    metadata=seg_meta,
                )
                if total_frames_processed >= 96:
                    time_per_96_frames = (t_pairwise / total_frames_processed) * 96
                    profiler.record(
                        "correlation_and_triangulation_96_frames",
                        time_per_96_frames,
                        write_immediately=True,
                        batch=correlation_batch,
                        metadata=seg_meta,
                    )
            
            # Mark segment as processed
            processed_segments.add(segment_key)
            
            # Update last processed frame
            last_processed_frame = max(end_frame, last_processed_frame)
            last_progress_time = time.time()
            
            # Output files are stored in:
            # {output_dir}/correlation/tracker.csv
            # This CSV contains 3D coordinates (X, Y, Z) after triangulation
            # Format: Frame, Visibility, X, Y, Z, Original_Frame, Point_Coordinates_Camera1, Point_Coordinates_Camera2
            segment_output = os.path.join(
                correlation_output_dir,
                "tracker.csv",
            )
            print(
                f"[CorrelationWorker]    ├─ pairwise+triangulation: {t_pairwise:.3f}s  "
                f"tracker_csv={segment_output}",
                flush=True,
            )

            # Optional: stitched side-by-side correlation video (shared OriginalFrameBuffer only)
            if enable_stitched_visualization and original_buffer is not None:
                stitched_t0 = time.time()
                from ..triplet_csv_reader import OriginalFrameBuffer as _OFB
                if not isinstance(original_buffer, _OFB):
                    print(
                        "[CorrelationWorker]    ⚠️  Stitched viz skipped: requires OriginalFrameBuffer",
                        flush=True,
                    )
                else:
                    s0, s1 = segment[0], segment[1]
                    fnwd_to_original_cam1 = {f: f for f in range(s0, s1 + 1)}
                    fnwd_to_original_cam2 = fnwd_to_original_cam1
                    correlated_pairs = load_correlated_pairs_from_tracker_csvs_realtime(
                        output_dir=correlation_output_dir,
                        frame_segments=[segment],
                    )
                    selected_pair_costs = load_selected_pair_costs_from_match_decisions(
                        output_dir=correlation_output_dir,
                        frame_segments=[segment],
                    )
                    stitched_suffix = (
                        "_nodetections" if not correlated_pairs else ""
                    )
                    append_stitched_segment_to_video(
                        frame_segments=[segment],
                        output_dir=correlation_output_dir,
                        camera_1_id=camera_1_id,
                        camera_2_id=camera_2_id,
                        original_buffer=original_buffer,
                        fnwd_to_original_cam1=fnwd_to_original_cam1,
                        fnwd_to_original_cam2=fnwd_to_original_cam2,
                        correlated_pairs_per_frame=correlated_pairs,
                        pair_costs_per_frame=selected_pair_costs,
                        filename_suffix=stitched_suffix,
                    )
                    if not correlated_pairs:
                        print(
                            "[CorrelationWorker]    ℹ️  Stitched viz (no detections): "
                            f"filename suffix _nodetections, segment {segment[0]}-{segment[1]}",
                            flush=True,
                        )
                t_viz_stitched = time.time() - stitched_t0
                if profiler:
                    profiler.record(
                        "correlation_segment_viz_stitched_s",
                        t_viz_stitched,
                        write_immediately=True,
                        batch=correlation_batch,
                        metadata=seg_meta,
                    )
                print(
                    f"[CorrelationWorker]    ├─ viz stitched (side-by-side): {t_viz_stitched:.3f}s",
                    flush=True,
                )
            
            # ============ TRAJECTORY CREATION ============
            # After triangulation, create trajectories from 3D points
            all_trajectory_data = {}  # Collect trajectory data for visualization
            
            trajectory_start_time = time.time()
            
            tw, th = original_frame_width, original_frame_height
            if tw is None or th is None:
                from ..triplet_csv_reader import OriginalFrameBuffer as _OriginalFrameBuffer
                if isinstance(original_buffer, _OriginalFrameBuffer):
                    bw, bh = original_buffer.get_original_frame_size()
                    if bw is not None and bh is not None:
                        tw, th = int(bw), int(bh)
            if tw is None or th is None:
                raise ValueError(
                    "Unable to determine camera-1 original frame size. "
                    "Set PipelineConfig.camera_1_original_frame_width/height "
                    "or ensure OriginalFrameBuffer has decoded at least one frame."
                )
            tw, th = int(tw), int(th)
            stored_trajectories, _, traj_folder, trajectory_context = create_trajectories_realtime(
                segment=segment,
                output_dir=correlation_output_dir,
                camera_1_id=camera_1_id,
                world_points_file=world_points_file,
                previous_context=trajectory_context,
                frame_width=tw,
                frame_height=th,
            )
            
            original_count = len(stored_trajectories)
            print(
                f"[CorrelationWorker] Batch {correlation_batch}: trajectories raw={original_count} "
                f"(frame_size={tw}x{th})"
            )
            
            # ============ TRAJECTORY MERGING ============
            removed_trajectories = []
            frame_decisions: List[Dict[str, Any]] = []
            trajectory_jsonl_path: Optional[str] = None
            if len(stored_trajectories) > 0:
                merge_start = time.time()

                # Step 1: Merge consecutive trajectories (gap 1–3 frames)
                stored_trajectories = merge_trajectories(stored_trajectories)

                # Step 2: Merge overlapping trajectories (1–4 overlapping frames)
                stored_trajectories = merge_overlapping_trajectories(stored_trajectories)

                for traj in stored_trajectories:
                    traj.track_id = next_track_id
                    next_track_id += 1

                # Step 3: Filter to keep only one point per frame (select best trajectory)
                (
                    stored_trajectories,
                    removed_trajectories,
                    filter_stats,
                    frame_decisions,
                ) = get_best_point_each_frame(stored_trajectories, segment)

                trajectory_jsonl_path = _append_trajectory_selection_jsonl(
                    correlation_output_dir,
                    segment,
                    frame_decisions,
                )

                # Write per-frame filter statistics CSV for this segment
                # Keep a single canonical location under trajectory_output/<segment>/.
                filter_csv_path = os.path.join(
                    traj_folder, "trajectory_filter_stats.csv"
                )

                # Always create the CSV (even if no frames were filtered) so the
                # file is present per segment; rows are only written when stats exist.
                with open(filter_csv_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            "frame_id",
                            "picked_x",
                            "picked_y",
                            "picked_z",
                            "num_points_dropped",
                            "num_points_before",
                        ]
                    )
                    if filter_stats:
                        for row in sorted(filter_stats, key=lambda r: r["frame"]):
                            writer.writerow(
                                [
                                    row["frame"],
                                    row["picked_x"],
                                    row["picked_y"],
                                    row["picked_z"],
                                    row["points_dropped"],
                                    row["points_before"],
                                ]
                            )

                merge_duration = time.time() - merge_start
                if profiler:
                    profiler.record(
                        "trajectory_merging",
                        merge_duration,
                        write_immediately=False,
                        batch=correlation_batch,
                        metadata=seg_meta,
                    )
            # ============ END TRAJECTORY MERGING ============
            
            # Store merged & filtered trajectory data for visualization
            all_trajectory_data[segment] = stored_trajectories
            
            trajectory_duration = time.time() - trajectory_start_time
            t_trajectory = trajectory_duration
            if profiler:
                profiler.record(
                    "trajectory_creation_batch",
                    t_trajectory,
                    write_immediately=True,
                    batch=correlation_batch,
                    metadata=seg_meta,
                )
                profiler.record(
                    "correlation_segment_trajectory_s",
                    t_trajectory,
                    write_immediately=True,
                    batch=correlation_batch,
                    metadata=seg_meta,
                )
            print(
                f"[CorrelationWorker]    ├─ trajectory (create/merge/filter/json): {t_trajectory:.3f}s  "
                f"raw={original_count} final={len(stored_trajectories)} removed={len(removed_trajectories)} "
                f"{tw}x{th} selection_jsonl={'yes' if trajectory_jsonl_path else 'no'}",
                flush=True,
            )
            # ============ END TRAJECTORY CREATION ============
            
            # Optional: cam1 overlay videos (timed separately)
            if enable_visualization and original_buffer is not None:
                viz_overlay_t0 = time.time()
                viz_segment = (
                    max(0, segment[0] - LAST_FRAMES_TO_SKIP),
                    segment[1],
                )
                create_visualization_from_triangulation(
                    frame_segments=[viz_segment],
                    output_dir=correlation_output_dir,
                    original_buffer=original_buffer,
                    camera_1_cam_path=camera_1_cam_path,
                    camera_1_id=camera_1_id,
                    camera_1_csv_path=camera_1_csv_path,
                    profiler=profiler,
                    trajectory_data=all_trajectory_data,  # Kept trajectories
                    removed_trajectory_data={segment: removed_trajectories}
                    if removed_trajectories
                    else None,
                    trail_length=10,
                    show_trajectory_labels=True,
                )
                t_viz_overlay = time.time() - viz_overlay_t0
                if profiler:
                    profiler.record(
                        "correlation_segment_viz_overlay_s",
                        t_viz_overlay,
                        write_immediately=True,
                        batch=correlation_batch,
                        metadata=seg_meta,
                    )
                print(
                    f"[CorrelationWorker]    ├─ viz cam1 overlay (chunks): {t_viz_overlay:.3f}s",
                    flush=True,
                )
            
            # Cleanup OriginalFrameBuffer after visualization (or when viz disabled but buffer present)
            if original_buffer is not None:
                from ..triplet_csv_reader import OriginalFrameBuffer as _OriginalFrameBuffer
                if not isinstance(original_buffer, _OriginalFrameBuffer):
                    print(
                        "[CorrelationWorker]    ⚠️  Buffer cleanup skipped: expected OriginalFrameBuffer",
                        flush=True,
                    )
                else:
                    end_frame_processed = segment[1]
                    clear_upto = end_frame_processed - LAST_FRAMES_TO_SKIP
                    if inference_done_event is not None and inference_done_event.is_set():
                        clear_upto = end_frame_processed
                    orig_size_before = len(original_buffer)
                    clear_infos: List[Tuple[int, Dict]] = []
                    for bn in sorted(original_buffer.get_batch_info().keys()):
                        rng = original_buffer.peek_batch_source_index_range(bn)
                        if rng is None:
                            continue
                        _, _max_src = rng
                        if _max_src <= clear_upto:
                            clear_infos.append((bn, original_buffer.clear_batch(bn)))
                    orig_size_after = len(original_buffer)
                    if clear_infos:
                        parts = []
                        total_frames = 0
                        for bn, ci in clear_infos:
                            total_frames += int(ci.get("count") or 0)
                            parts.append(
                                f"reader_batch={bn}:src={ci.get('min_source_index')}..{ci.get('max_source_index')}"
                                f",csv_rows={ci.get('min_csv_row_id')}..{ci.get('max_csv_row_id')},n={ci.get('count')}"
                            )
                        print(
                            f"[CorrelationWorker] Batch {correlation_batch}: original_buffer cleared "
                            f"{len(clear_infos)} reader batch(es) (clear_upto={clear_upto}, "
                            f"segment_end={end_frame_processed}); "
                            f"total_frames={total_frames}; " + "; ".join(parts) + f"; "
                            f"total_size {orig_size_before}->{orig_size_after}",
                            flush=True,
                        )
                        if profiler:
                            profiler.record(
                                "correlation_orig_buffer_cleared",
                                float(orig_size_before - orig_size_after),
                                write_immediately=True,
                                batch=correlation_batch,
                                metadata=(
                                    f"clear_upto={clear_upto},segment_end_frame={end_frame_processed},"
                                    f"reader_batches_cleared={[b for b, _ in clear_infos]},total_frames={total_frames},"
                                    f"details={'; '.join(parts)},before={orig_size_before},after={orig_size_after}"
                                ),
                            )
                    else:
                        print(
                            f"[CorrelationWorker] Batch {correlation_batch}: original_buffer — no reader batch "
                            f"fully cleared (max Source_Index per batch still > clear_upto={clear_upto}, "
                            f"or buffer empty); size={orig_size_after}",
                            flush=True,
                        )

            complete_segment_duration = time.time() - complete_segment_start_time
            t_accounted = (
                t_pairwise + t_trajectory + t_viz_overlay + t_viz_stitched
            )
            t_overhead = complete_segment_duration - t_accounted
            if profiler:
                profiler.record(
                    "correlation_segment_accounted_s",
                    t_accounted,
                    write_immediately=True,
                    batch=correlation_batch,
                    metadata=seg_meta,
                )
                profiler.record(
                    "correlation_segment_overhead_s",
                    t_overhead,
                    write_immediately=True,
                    batch=correlation_batch,
                    metadata=seg_meta,
                )
                profiler.record(
                    "correlation_complete_segment_time",
                    complete_segment_duration,
                    write_immediately=True,
                    batch=correlation_batch,
                    metadata=(
                        f"{seg_meta},wait_ready={t_wait_ready:.4f},pairwise={t_pairwise:.4f},"
                        f"trajectory={t_trajectory:.4f},viz_overlay={t_viz_overlay:.4f},"
                        f"viz_stitched={t_viz_stitched:.4f},accounted={t_accounted:.4f},"
                        f"overhead={t_overhead:.4f}"
                    ),
                )
                profiler.write_correlation_segment_breakdown(
                    batch=correlation_batch,
                    seg_meta=seg_meta,
                    t_wait_ready=t_wait_ready,
                    t_pairwise=t_pairwise,
                    t_trajectory=t_trajectory,
                    t_viz_overlay=t_viz_overlay,
                    t_viz_stitched=t_viz_stitched,
                    t_accounted=t_accounted,
                    t_overhead=t_overhead,
                    t_wall=complete_segment_duration,
                )
            print(
                f"[CorrelationWorker] Batch {correlation_batch} — timing (s)  "
                f"wait_before={t_wait_ready:.3f} | "
                f"pairwise={t_pairwise:.3f} + trajectory={t_trajectory:.3f} + "
                f"viz_cam1={t_viz_overlay:.3f} + viz_stitched={t_viz_stitched:.3f} "
                f"= accounted {t_accounted:.3f} | overhead {t_overhead:.3f} | "
                f"wall {complete_segment_duration:.3f}",
                flush=True,
            )

            # Mark one correlation batch complete (segments + trajectory + optional viz/cleanup)
            correlation_batch += 1
            last_busy_end_time = time.time()
        
        # Wait before next check
        time.sleep(check_interval)

    print("[CorrelationWorker] 🛑 Correlation worker thread stopped")

