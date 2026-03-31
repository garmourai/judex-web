"""
Realtime-specific stitched correlation visualization module.

Creates a stitched video showing both camera views side-by-side with correlation
points labeled (A, B, C...). Uses staging buffers instead of video files for
the realtime flow.
"""

import os
import csv
import cv2
import numpy as np
from ast import literal_eval
from typing import Dict, List, Optional, Tuple, Any


# Correlation labels for stitched visualization
_CORRELATION_LABELS = ["A", "B", "C", "D", "E", "F", "G"]


def _draw_circles_with_labels(
    img,
    points: List[Tuple[float, float]],
    labels: List[str],
    color_bgr=(0, 255, 0),
    radius=12,
    font_scale=0.8,
):
    """Draw circles and label text on image. points and labels must have same length."""
    for (x, y), label in zip(points, labels):
        ix, iy = int(round(x)), int(round(y))
        cv2.circle(img, (ix, iy), radius, color_bgr, 2)
        cv2.putText(
            img, label, (ix + radius + 2, iy - radius - 2),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_bgr, 2, cv2.LINE_AA,
        )


def load_correlated_pairs_from_tracker_csvs_realtime(
    output_dir: str,
    frame_segments: List[Tuple[int, int]],
) -> Dict[int, List[Tuple[Tuple[float, float], Tuple[float, float]]]]:
    """
    Load per-frame correlated (cam1, cam2) 2D point pairs from all segment tracker CSVs.
    tracker.csv format: Frame, Visibility, X, Y, Z, Original_Frame, Point_Coordinates_Camera1, Point_Coordinates_Camera2
    Point_Coordinates columns are like "[(x1, y1), (x2, y2), ...]"; same index = same correlation (label A, B, C...).

    Args:
        output_dir: Directory containing perform_correlation_output
        frame_segments: List of (start, end) segment tuples

    Returns:
        Dict mapping global frame_id -> list of ((x_cam1, y_cam1), (x_cam2, y_cam2))
    """
    result: Dict[int, List[Tuple[Tuple[float, float], Tuple[float, float]]]] = {}

    for segment in frame_segments:
        segment_path = os.path.join(
            output_dir,
            "perform_correlation_output",
            f"{segment[0]}_to_{segment[1]}",
            "tracker.csv",
        )
        if not os.path.isfile(segment_path):
            continue
        try:
            with open(segment_path, "r", newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header is None or len(header) < 8:
                    continue
                for row in reader:
                    if len(row) < 8:
                        continue
                    try:
                        frame_num = int(row[0])
                        cam1_str = row[6].strip()
                        cam2_str = row[7].strip()
                        if not cam1_str.startswith("[") or not cam2_str.startswith("["):
                            continue
                        # Parse "[(x1, y1), (x2, y2), ...]"
                        list_cam1 = literal_eval(cam1_str)
                        list_cam2 = literal_eval(cam2_str)
                        if not isinstance(list_cam1, list) or not isinstance(list_cam2, list):
                            continue
                        pairs = []
                        for p1, p2 in zip(list_cam1, list_cam2):
                            if isinstance(p1, (list, tuple)) and len(p1) >= 2 and isinstance(p2, (list, tuple)) and len(p2) >= 2:
                                pairs.append(((float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1]))))
                        if pairs:
                            if frame_num not in result:
                                result[frame_num] = []
                            result[frame_num].extend(pairs)
                    except (ValueError, SyntaxError, TypeError):
                        continue
        except Exception as e:
            print(f"[CorrelationWorker]    ⚠️  Error reading tracker CSV {segment_path}: {e}")
            continue

    return result


def _write_stitched_frame(
    writer: cv2.VideoWriter,
    img1: np.ndarray,
    img2: np.ndarray,
    out_h: int,
    fnwd: int,
    camera_1_id: str,
    camera_2_id: str,
    correlated_pairs_per_frame: Dict[int, List[Tuple[Tuple[float, float], Tuple[float, float]]]],
) -> None:
    """Draw labels and write one stitched frame to the writer."""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    if h1 != out_h:
        img1 = cv2.resize(img1, (w1, out_h))
    if h2 != out_h:
        img2 = cv2.resize(img2, (w2, out_h))
    pairs = correlated_pairs_per_frame.get(fnwd, [])
    pts_cam1 = [p[0] for p in pairs]
    pts_cam2 = [p[1] for p in pairs]
    labels = [_CORRELATION_LABELS[i % len(_CORRELATION_LABELS)] for i in range(len(pairs))]
    _draw_circles_with_labels(img1, pts_cam1, labels, color_bgr=(0, 255, 0))
    _draw_circles_with_labels(img2, pts_cam2, labels, color_bgr=(0, 255, 0))
    cv2.putText(img1, f"{camera_1_id} | fnwd={fnwd}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img2, f"{camera_2_id} | fnwd={fnwd}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    stitched = np.hstack((img1, img2))
    writer.write(stitched)


def append_stitched_segment_to_video(
    frame_segments: List[Tuple[int, int]],
    output_dir: str,
    camera_1_id: str,
    camera_2_id: str,
    staging_buffer_1,
    staging_buffer_2,
    fnwd_to_original_cam1: Dict[int, int],
    fnwd_to_original_cam2: Dict[int, int],
    correlated_pairs_per_frame: Dict[int, List[Tuple[Tuple[float, float], Tuple[float, float]]]],
    writer_state: Optional[Dict[str, Any]] = None,
) -> int:
    """
    Write one stitched MP4 per segment (batch): ``stitched_correlation_{s}_to_{e}.mp4``
    under ``output_dir/stitched_correlation_videos/``. Each file is opened, filled, and
    closed in this call (no single rolling output file).

    writer_state: optional, ignored (kept for API compatibility).

    Returns: total number of frames written across all segment files this call.
    """
    del writer_state  # per-batch files; shared state no longer used
    if staging_buffer_1 is None or staging_buffer_2 is None:
        return 0

    try:
        from ..triplet_csv_reader import OriginalFrameBuffer as _OFB
        _use_ofb_stitched = (
            isinstance(staging_buffer_1, _OFB)
            and staging_buffer_1 is staging_buffer_2
        )
    except ImportError:
        _use_ofb_stitched = False

    video_subdir = os.path.join(output_dir, "stitched_correlation_videos")
    os.makedirs(video_subdir, exist_ok=True)
    out_fps = 15.0
    total_frames_written = 0

    for (seg_s, seg_e) in frame_segments:
        frame_numbers = sorted(
            fnwd
            for fnwd in range(seg_s, seg_e + 1)
            if fnwd in fnwd_to_original_cam1 and fnwd in fnwd_to_original_cam2
        )
        if not frame_numbers:
            continue

        frame_indices_cam1 = set(fnwd_to_original_cam1[fnwd] for fnwd in frame_numbers)
        frame_indices_cam2 = set(fnwd_to_original_cam2[fnwd] for fnwd in frame_numbers)

        if _use_ofb_stitched:
            frame_dict_cam1, frame_dict_cam2 = staging_buffer_1.get_stitched_frame_dicts(
                frame_indices_cam1, frame_indices_cam2
            )
        else:
            try:
                frames_from_buffer_cam1 = staging_buffer_1.peek_frames_by_indices(
                    frame_indices_cam1
                )
                frames_from_buffer_cam2 = staging_buffer_2.peek_frames_by_indices(
                    frame_indices_cam2
                )
            except Exception as e:
                print(f"[CorrelationWorker]    ❌ Error peeking frames from staging buffers: {e}")
                import traceback
                traceback.print_exc()
                continue
            frame_dict_cam1 = {
                frame_data.camera_stream_index: frame_data.frame
                for frame_data in frames_from_buffer_cam1
            }
            frame_dict_cam2 = {
                frame_data.camera_stream_index: frame_data.frame
                for frame_data in frames_from_buffer_cam2
            }
        if not frame_dict_cam1 or not frame_dict_cam2:
            continue

        first_frame_cam1 = next(iter(frame_dict_cam1.values()))
        first_frame_cam2 = next(iter(frame_dict_cam2.values()))
        h1, w1 = first_frame_cam1.shape[:2]
        h2, w2 = first_frame_cam2.shape[:2]
        out_w = w1 + w2
        out_h = max(h1, h2)
        out_path = os.path.join(
            video_subdir, f"stitched_correlation_{seg_s}_to_{seg_e}.mp4"
        )
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, out_fps, (out_w, out_h))
        if not writer.isOpened():
            print(f"[CorrelationWorker]    ❌ Could not create video writer: {out_path}")
            continue

        segment_frames = 0
        try:
            for fnwd in frame_numbers:
                orig1 = fnwd_to_original_cam1.get(fnwd)
                orig2 = fnwd_to_original_cam2.get(fnwd)
                if orig1 is None or orig2 is None:
                    continue
                img1 = frame_dict_cam1.get(orig1)
                img2 = frame_dict_cam2.get(orig2)
                if img1 is None or img2 is None:
                    continue
                _write_stitched_frame(
                    writer,
                    img1,
                    img2,
                    out_h,
                    fnwd,
                    camera_1_id,
                    camera_2_id,
                    correlated_pairs_per_frame,
                )
                segment_frames += 1
        finally:
            writer.release()

        total_frames_written += segment_frames
        if segment_frames:
            print(
                f"[CorrelationWorker]    📹 Stitched correlation segment video: {out_path} "
                f"({segment_frames} frames)"
            )

    return total_frames_written


def finalize_stitched_video(writer_state: Dict[str, Any]) -> None:
    """No-op: per-segment stitched videos are closed inside append_stitched_segment_to_video."""
    writer = writer_state.get("writer")
    if writer is not None:
        writer.release()
        writer_state["writer"] = None
        out_path = writer_state.get("out_path", "")
        print(
            f"[CorrelationWorker]    ✅ Stitched writer finalized (legacy path): {out_path}",
            flush=True,
        )
