"""Remove frames from staging buffers using global frame id and sensor timestamps."""

from typing import Optional, Tuple

from ..correlation_worker_utils import load_fid_to_stream_from_dist_tracker_csv


def cleanup_staging_buffers_from_triangulation(
    max_common_frame: int,
    output_dir: str,
    camera_1_id: str,
    camera_2_id: str,
    staging_buffer_1,
    staging_buffer_2=None,
    camera_1_csv_path: Optional[str] = None,
    camera_2_csv_path: Optional[str] = None,
) -> Tuple[int, int]:
    """
    Remove frames from staging buffers based on max_common_frame timestamp.

    For each camera independently:
    1. Maps max_common_frame (global frame id) to per-camera stream index using mapping files
    2. Finds that frame in the staging buffer to get its sensor_timestamp
    3. Removes all frames with sensor_timestamp <= that timestamp from the staging buffer

    Args:
        max_common_frame: Maximum global frame index that both cameras have processed
        output_dir: Output directory where mapping files are stored
        camera_1_id: Camera 1 identifier
        camera_2_id: Camera 2 identifier
        staging_buffer_1: StagingBuffer for camera 1
        staging_buffer_2: Optional StagingBuffer for camera 2

    Returns:
        (removed_cam1, removed_cam2): counts removed from each staging buffer.
    """
    if staging_buffer_1 is None:
        return (0, 0)

    removed_cam1 = 0
    removed_cam2 = 0

    fid_to_stream_cam1 = (
        load_fid_to_stream_from_dist_tracker_csv(camera_1_csv_path) if camera_1_csv_path else {}
    )
    fid_to_stream_cam2 = (
        load_fid_to_stream_from_dist_tracker_csv(camera_2_csv_path) if camera_2_csv_path else {}
    )

    stream_idx_cam1 = fid_to_stream_cam1.get(max_common_frame)
    stream_idx_cam2 = fid_to_stream_cam2.get(max_common_frame) if staging_buffer_2 is not None else None

    if stream_idx_cam1 is None:
        print(
            f"[CorrelationWorker]    ⚠️  Could not map max_common_frame={max_common_frame} to Camera 1 camera_stream_index"
        )
        return (0, removed_cam2)

    # Find timestamps for these frames in staging buffers
    ts_cut_cam1 = None
    ts_cut_cam2 = None

    # Camera 1: peek at frames to find the one with matching camera_stream_index
    with staging_buffer_1._condition:
        for frame_data in staging_buffer_1.buffer:
            if frame_data.camera_stream_index == stream_idx_cam1:
                frame_ts = getattr(frame_data, "sensor_timestamp", None)
                if frame_ts is not None:
                    ts_cut_cam1 = int(frame_ts)
                    break

    # Camera 2: peek at frames to find the one with matching camera_stream_index
    if staging_buffer_2 is not None and stream_idx_cam2 is not None:
        with staging_buffer_2._condition:
            for frame_data in staging_buffer_2.buffer:
                if frame_data.camera_stream_index == stream_idx_cam2:
                    frame_ts = getattr(frame_data, "sensor_timestamp", None)
                    if frame_ts is not None:
                        ts_cut_cam2 = int(frame_ts)
                        break

    # Remove frames with timestamp <= ts_cut for each camera
    # First, collect timestamps that will be removed (before removal)
    removed_ts_cam1 = []
    removed_ts_cam2 = []

    if ts_cut_cam1 is not None:
        # Collect timestamps that will be removed
        with staging_buffer_1._condition:
            for frame_data in staging_buffer_1.buffer:
                frame_ts = getattr(frame_data, "sensor_timestamp", None)
                if frame_ts is not None:
                    frame_ts_int = int(frame_ts)
                    if frame_ts_int <= ts_cut_cam1:
                        removed_ts_cam1.append(frame_ts_int)

        removed_cam1 = staging_buffer_1.remove_frames_by_timestamp_threshold(ts_cut_cam1)
        print(
            f"[CorrelationWorker]    🧹 [StagingCleanup] Camera 1: removed {removed_cam1} frames "
            f"(max_common_frame={max_common_frame}, camera_stream_index_cam1={stream_idx_cam1}, ts_cut={ts_cut_cam1})"
        )
        print(f"[CorrelationWorker]    🧹 [StagingCleanup] Camera 1: removed timestamps={removed_ts_cam1}")
    else:
        print(f"[CorrelationWorker]    ⚠️  Camera 1: Could not find timestamp for camera_stream_index={stream_idx_cam1}")

    if staging_buffer_2 is not None and ts_cut_cam2 is not None:
        # Collect timestamps that will be removed
        with staging_buffer_2._condition:
            for frame_data in staging_buffer_2.buffer:
                frame_ts = getattr(frame_data, "sensor_timestamp", None)
                if frame_ts is not None:
                    frame_ts_int = int(frame_ts)
                    if frame_ts_int <= ts_cut_cam2:
                        removed_ts_cam2.append(frame_ts_int)

        removed_cam2 = staging_buffer_2.remove_frames_by_timestamp_threshold(ts_cut_cam2)
        print(
            f"[CorrelationWorker]    🧹 [StagingCleanup] Camera 2: removed {removed_cam2} frames "
            f"(max_common_frame={max_common_frame}, camera_stream_index_cam2={stream_idx_cam2}, ts_cut={ts_cut_cam2})"
        )
        print(f"[CorrelationWorker]    🧹 [StagingCleanup] Camera 2: removed timestamps={removed_ts_cam2}")
    elif staging_buffer_2 is not None:
        print(f"[CorrelationWorker]    ⚠️  Camera 2: Could not find timestamp for camera_stream_index={stream_idx_cam2}")

    return (removed_cam1, removed_cam2)
