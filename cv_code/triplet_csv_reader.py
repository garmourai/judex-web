"""
Triplet CSV Reader and Original Frame Buffer for Judex Pipeline

Provides:
  - TripletRow: dataclass for one row of the triplet.csv
  - OriginalFrameBuffer: bounded thread-safe buffer storing original-resolution
    frames keyed by (batch_num, source_index)
  - TripletCSVReaderWorker: reads triplet.csv incrementally, reads .ts segment
    files via M3U8SegmentReader, and pushes frames to TracknetBuffers (resized)
    and OriginalFrameBuffer (original res)
"""

import io
import os
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from .inference.frame_reader import TracknetBuffer
from .inference.utils_general import WIDTH, HEIGHT

from .m3u8_reader import M3U8SegmentReader


# ---------------------------------------------------------------------------
# TripletRow dataclass
# ---------------------------------------------------------------------------

@dataclass
class TripletRow:
    """One row from the triplet.csv (hls_sync_*_triple.csv)."""
    source_index: int
    sink_index: Optional[int]        # None → skip this row (missing sink)
    hq_index: Optional[int]          # parsed but unused by this pipeline
    source_sensor_ns: Optional[int]
    sink_sensor_ns: Optional[int]
    source_wall_ns: Optional[int]    # parsed but unused
    hq_wall_ns: Optional[int]        # parsed but unused
    status: str


def _parse_optional_int(value: str) -> Optional[int]:
    v = value.strip()
    if v == "" or v.lower() in ("none", "null", "nan"):
        return None
    return int(v)


# ---------------------------------------------------------------------------
# OriginalFrameBuffer
# ---------------------------------------------------------------------------

class OriginalFrameBuffer:
    """
    Bounded thread-safe buffer for original-resolution frames.

    Organised by batch_num (outer key) and source_index (inner key).
    Stores both camera frames plus sink_index and csv_row_id for
    post-run verification.

    Layout:
        _store[batch_num][source_index] = (cam1_frame, cam2_frame,
                                           sink_index, csv_row_id)

    The total number of frames across all live batches is capped at
    max_size (default 196).  add_frame() blocks when the buffer is full
    and unblocks when clear_batch() removes entries.
    """

    def __init__(self, max_size: int = 196):
        self._condition = threading.Condition()
        self.max_size = max_size
        # batch_num → { source_index → (cam1_frame, cam2_frame, sink_index, csv_row_id) }
        self._store: Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray, int, int]]] = {}
        # (height, width) of cam1 original frames, set on first add_frame
        self._cam1_shape_hw: Optional[Tuple[int, int]] = None

    # ------------------------------------------------------------------
    # Write side (reader thread)
    # ------------------------------------------------------------------

    def add_frame(
        self,
        batch_num: int,
        csv_row_id: int,
        source_index: int,
        sink_index: int,
        cam1_frame: np.ndarray,
        cam2_frame: np.ndarray,
        stop_event: Optional[threading.Event] = None,
    ) -> bool:
        """
        Store a pair of original-res frames.

        Blocks if total frame count >= max_size.
        Returns False if stop_event is set while waiting.
        """
        with self._condition:
            while self._total_count() >= self.max_size:
                if stop_event is not None and stop_event.is_set():
                    return False
                self._condition.wait(timeout=0.1)

            if batch_num not in self._store:
                self._store[batch_num] = {}
            if self._cam1_shape_hw is None and cam1_frame is not None and cam1_frame.size:
                self._cam1_shape_hw = (int(cam1_frame.shape[0]), int(cam1_frame.shape[1]))
            self._store[batch_num][source_index] = (
                cam1_frame, cam2_frame, sink_index, csv_row_id
            )
            self._condition.notify_all()
            return True

    # ------------------------------------------------------------------
    # Read side (correlation worker)
    # ------------------------------------------------------------------

    def get_frames_for_sync_frames(
        self, sync_frame_indices: set
    ) -> Dict[int, np.ndarray]:
        """
        Return {source_index → cam1_frame} for the given set of source indices.
        Non-blocking read across all live batches.
        """
        result: Dict[int, np.ndarray] = {}
        with self._condition:
            for batch in self._store.values():
                for src_idx, (cam1, _, _, _) in batch.items():
                    if src_idx in sync_frame_indices:
                        result[src_idx] = cam1
        return result

    def get_cam2_frames_for_sync_frames(
        self, sync_frame_indices: set
    ) -> Dict[int, np.ndarray]:
        """
        Return {source_index → cam2_frame} for the given set of source indices.
        Non-blocking read across all live batches.
        """
        result: Dict[int, np.ndarray] = {}
        with self._condition:
            for batch in self._store.values():
                for src_idx, (_, cam2, _, _) in batch.items():
                    if src_idx in sync_frame_indices:
                        result[src_idx] = cam2
        return result

    def get_stitched_frame_dicts(
        self,
        source_indices: Set[int],
        sink_indices: Set[int],
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
        """
        For side-by-side stitched correlation: cam1 keyed by source_index,
        cam2 keyed by sink_index (matches dist_tracker per-camera stream ids).
        Scans all live batches.
        """
        by_source: Dict[int, np.ndarray] = {}
        by_sink: Dict[int, np.ndarray] = {}
        with self._condition:
            for batch in self._store.values():
                for src_idx, (cam1, cam2, sink_idx, _) in batch.items():
                    if src_idx in source_indices:
                        by_source[src_idx] = cam1
                    if sink_idx in sink_indices:
                        by_sink[sink_idx] = cam2
        return by_source, by_sink

    # ------------------------------------------------------------------
    # Cleanup (correlation worker, after batch is fully processed)
    # ------------------------------------------------------------------

    def clear_batch(self, batch_num: int) -> Dict[str, Optional[int]]:
        """
        Remove all entries for batch_num and notify blocked adders.
        Called by CorrelationWorker after it has finished triangulating
        and drawing viz for that batch.
        """
        with self._condition:
            batch = self._store.get(batch_num)
            info: Dict[str, Optional[int]] = {
                "batch_num": batch_num,
                "count": 0,
                "min_csv_row_id": None,
                "max_csv_row_id": None,
                "min_source_index": None,
                "max_source_index": None,
                "min_sink_index": None,
                "max_sink_index": None,
            }
            if batch:
                src_idxs = list(batch.keys())
                rows = [v[3] for v in batch.values()]
                sink_idxs = [v[2] for v in batch.values()]
                info["count"] = len(batch)
                info["min_csv_row_id"] = min(rows)
                info["max_csv_row_id"] = max(rows)
                info["min_source_index"] = min(src_idxs)
                info["max_source_index"] = max(src_idxs)
                info["min_sink_index"] = min(sink_idxs)
                info["max_sink_index"] = max(sink_idxs)
                del self._store[batch_num]
            self._condition.notify_all()
            return info

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def _total_count(self) -> int:
        """Total number of frames currently stored (must hold _condition)."""
        return sum(len(b) for b in self._store.values())

    def __len__(self) -> int:
        with self._condition:
            return self._total_count()

    def get_batch_info(self) -> Dict[int, int]:
        """Return {batch_num → frame_count} for all live batches."""
        with self._condition:
            return {k: len(v) for k, v in self._store.items()}

    def get_original_frame_size(self) -> Tuple[Optional[int], Optional[int]]:
        """Return (width, height) of camera-1 original frames; set after first add_frame."""
        with self._condition:
            if self._cam1_shape_hw is None:
                return None, None
            h, w = self._cam1_shape_hw
            return w, h

    def clear_all(self) -> None:
        """Drop all stored frames (e.g. force-stop)."""
        with self._condition:
            self._store.clear()
            self._condition.notify_all()


# ---------------------------------------------------------------------------
# TripletCSVReaderWorker
# ---------------------------------------------------------------------------

class TripletCSVReaderWorker:
    """
    Reads triplet.csv incrementally (polling for new rows) and pushes frames
    to both TracknetBuffers (resized 288×512) and OriginalFrameBuffer
    (original resolution).

    One batch = up to chunk_size (default 96) CSV rows.  Only matched rows
    (both Source_Index and Sink_Index non-empty) produce frame pushes.
    Unmatched rows are silently skipped; they do NOT count toward the batch.

    After prepare_batch() reads the video frames, the worker waits
    pacing_seconds before pushing so that inference has time to drain the
    previous batch.  Batch preparation runs concurrently during that wait.
    """

    def __init__(
        self,
        triplet_csv_path: str,
        source_segments_dir: str,       # e.g. .../ts_segments_source/1541/
        source_segment_csvs_dir: str,   # e.g. .../segments_1541/source/
        sink_segments_dir: str,
        sink_segment_csvs_dir: str,
        camera_1_id: str,
        camera_2_id: str,
        tracknet_buffer_1: TracknetBuffer,
        tracknet_buffer_2: TracknetBuffer,
        original_buffer: OriginalFrameBuffer,
        stop_event: threading.Event,
        triplet_csv_reader_done_event: threading.Event,
        force_stop_event: Optional[threading.Event] = None,
        camera_1_output_dir: str = "",
        camera_2_output_dir: str = "",
        chunk_size: int = 96,
        pacing_seconds: float = 3.2,
        poll_interval: float = 0.5,
        csv_idle_timeout_seconds: float = 15.0,
        profiler=None,
    ):
        self._triplet_csv_path = triplet_csv_path
        self._source_segments_dir = source_segments_dir
        self._source_segment_csvs_dir = source_segment_csvs_dir
        self._sink_segments_dir = sink_segments_dir
        self._sink_segment_csvs_dir = sink_segment_csvs_dir
        self._camera_1_id = camera_1_id
        self._camera_2_id = camera_2_id
        self._tracknet_buffer_1 = tracknet_buffer_1
        self._tracknet_buffer_2 = tracknet_buffer_2
        self._original_buffer = original_buffer
        self._stop_event = stop_event
        self._triplet_csv_reader_done_event = triplet_csv_reader_done_event
        self._force_stop_event = force_stop_event
        self._camera_1_output_dir = camera_1_output_dir
        self._camera_2_output_dir = camera_2_output_dir
        self._chunk_size = chunk_size
        self._pacing_seconds = pacing_seconds
        self._poll_interval = poll_interval
        self._csv_idle_timeout_seconds = csv_idle_timeout_seconds
        self._profiler = profiler

        # M3U8 segment readers (created in run())
        self._source_reader: Optional[M3U8SegmentReader] = None
        self._sink_reader: Optional[M3U8SegmentReader] = None

        # CSV file state for incremental reading
        self._csv_file: Optional[io.TextIOWrapper] = None
        self._csv_header: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        print(f"[TripletCSVReader] Starting — csv={self._triplet_csv_path}")

        # Register cameras in TracknetBuffers
        self._tracknet_buffer_1.register_camera(self._camera_1_id)
        self._tracknet_buffer_2.register_camera(self._camera_2_id)

        # Create segment readers
        self._source_reader = M3U8SegmentReader(
            segments_dir=self._source_segments_dir,
            segment_csvs_dir=self._source_segment_csvs_dir,
            stop_event=self._stop_event,
            poll_interval=self._poll_interval,
        )
        self._sink_reader = M3U8SegmentReader(
            segments_dir=self._sink_segments_dir,
            segment_csvs_dir=self._sink_segment_csvs_dir,
            stop_event=self._stop_event,
            poll_interval=self._poll_interval,
        )

        try:
            batch_num = 0

            while not self._stop_event.is_set() and not (
                self._force_stop_event is not None and self._force_stop_event.is_set()
            ):
                batch_start = time.time()

                tb1_size = len(self._tracknet_buffer_1)
                tb2_size = len(self._tracknet_buffer_2)
                orig_size = len(self._original_buffer)

                # --- Read next chunk_size rows from triplet.csv ---
                csv_poll_start = time.time()
                rows = self._read_next_rows(self._chunk_size)
                csv_poll_elapsed = time.time() - csv_poll_start

                if rows is None:
                    if self._force_stop_event is not None and self._force_stop_event.is_set():
                        print(f"[TripletCSVReader] Batch {batch_num}: exiting — force_stop (csv read)")
                    else:
                        print(f"[TripletCSVReader] Batch {batch_num}: exiting — interrupted during csv poll")
                    break
                if len(rows) == 0:
                    # CSV signalled end-of-stream
                    print(f"[TripletCSVReader] CSV end-of-stream reached after batch {batch_num - 1}")
                    break

                is_partial_final = len(rows) < self._chunk_size

                # --- Prepare batch (read frames from .ts segments) ---
                prepare_start = time.time()
                matched = self._prepare_batch(rows, batch_num)
                prepare_elapsed = time.time() - prepare_start

                tb1_after = len(self._tracknet_buffer_1)
                tb2_after = len(self._tracknet_buffer_2)
                orig_after = len(self._original_buffer)
                stop_after_prepare = self._stop_event.is_set() or (
                    self._force_stop_event is not None and self._force_stop_event.is_set()
                )
                if self._profiler:
                    self._profiler.record("reader_prepare_batch_time", prepare_elapsed,
                                          write_immediately=False, batch=batch_num, thread="TripletCSVReader",
                                          metadata=f"matched={len(matched)},stop={stop_after_prepare},"
                                                   f"tb1={tb1_after},tb2={tb2_after},orig={orig_after}")

                if stop_after_prepare:
                    print(f"[TripletCSVReader] Batch {batch_num}: exiting — stop after prepare")
                    break

                # --- Pacing wait (remaining time after prepare) ---
                wait_time = max(0.0, self._pacing_seconds - prepare_elapsed)
                pacing_wait_elapsed = 0.0
                if wait_time > 0:
                    pacing_start = time.time()
                    deadline = time.time() + wait_time
                    while time.time() < deadline and not self._stop_event.is_set() and not (
                        self._force_stop_event is not None and self._force_stop_event.is_set()
                    ):
                        time.sleep(min(0.1, deadline - time.time()))
                    pacing_wait_elapsed = time.time() - pacing_start

                if self._stop_event.is_set() or (
                    self._force_stop_event is not None and self._force_stop_event.is_set()
                ):
                    print(f"[TripletCSVReader] Batch {batch_num}: exiting — stop after pacing")
                    break

                # --- Push batch to both buffers ---
                push_start = time.time()
                self._push_batch(batch_num, matched)
                push_elapsed = time.time() - push_start

                # Log buffer sizes after push
                tb1_post = len(self._tracknet_buffer_1)
                tb2_post = len(self._tracknet_buffer_2)
                orig_post = len(self._original_buffer)
                if self._profiler:
                    self._profiler.record("reader_buffer_push_time", push_elapsed,
                                          write_immediately=False, batch=batch_num, thread="TripletCSVReader",
                                          metadata=f"tb1={tb1_post},tb2={tb2_post},orig={orig_post}")

                batch_elapsed = time.time() - batch_start
                if self._profiler:
                    self._profiler.record("reader_complete_batch_time", batch_elapsed,
                                          write_immediately=True, batch=batch_num, thread="TripletCSVReader")

                partial_tag = " [final partial]" if is_partial_final else ""
                print(
                    f"[TripletCSVReader] Batch {batch_num}: rows={len(rows)} matched={len(matched)}{partial_tag} "
                    f"(csv_poll={csv_poll_elapsed:.2f}s, prepare={prepare_elapsed:.2f}s, "
                    f"pace={pacing_wait_elapsed:.2f}s, push={push_elapsed:.2f}s, total={batch_elapsed:.2f}s; "
                    f"tb1 {tb1_size}->{tb1_post}, tb2 {tb2_size}->{tb2_post}, orig {orig_size}->{orig_post})"
                )
                batch_num += 1
                if is_partial_final:
                    print("[TripletCSVReader] Final partial batch pushed — exiting reader loop")
                    break

        finally:
            # Always signal downstream that no more frames are coming
            print(f"[TripletCSVReader] Signalling triplet_csv_reader_done_event")
            self._triplet_csv_reader_done_event.set()
            if self._csv_file is not None:
                self._csv_file.close()
                self._csv_file = None
            if self._source_reader is not None:
                self._source_reader.close()
            if self._sink_reader is not None:
                self._sink_reader.close()
            print(f"[TripletCSVReader] Thread exiting")

    # ------------------------------------------------------------------
    # Incremental CSV reading
    # ------------------------------------------------------------------

    def _open_csv(self) -> bool:
        """Open triplet.csv and skip the header row."""
        while not os.path.exists(self._triplet_csv_path):
            if self._stop_event.is_set():
                return False
            print(f"[TripletCSVReader] Waiting for triplet CSV: {self._triplet_csv_path}")
            time.sleep(self._poll_interval)

        self._csv_file = open(self._triplet_csv_path, "r", newline="")
        # Read and parse header
        header_line = self._csv_file.readline()
        self._csv_header = [h.strip() for h in header_line.strip().split(",")]
        return True

    def _read_next_rows(self, n: int) -> Optional[List[TripletRow]]:
        """
        Read up to n rows from triplet.csv, polling for new content.

        Returns:
            List[TripletRow] — up to n rows; may be shorter on EOF idle timeout or graceful stop.
            None only if force_stop_event is set (abandon read).
            Empty list — no more rows (end of stream / idle timeout with nothing read this call).
        """
        if self._csv_file is None:
            if not self._open_csv():
                return None

        rows: List[TripletRow] = []
        last_non_empty_line_time = time.time()

        while len(rows) < n:
            if self._force_stop_event is not None and self._force_stop_event.is_set():
                return None
            if self._stop_event.is_set():
                return rows

            line = self._csv_file.readline()
            if not line:
                now = time.time()
                if self._stop_event.is_set():
                    return rows
                if self._force_stop_event is not None and self._force_stop_event.is_set():
                    return None
                if now - last_non_empty_line_time >= self._csv_idle_timeout_seconds:
                    return rows
                time.sleep(self._poll_interval)
                continue

            line = line.strip()
            if not line:
                continue

            parts = line.split(",")
            if len(parts) < len(self._csv_header):
                continue

            row_dict = dict(zip(self._csv_header, parts))
            row = self._parse_row(row_dict)
            if row is not None:
                rows.append(row)
                last_non_empty_line_time = time.time()

        return rows

    def _parse_row(self, row_dict: dict) -> Optional[TripletRow]:
        """Parse one CSV row dict into a TripletRow. Returns None if source_index is missing."""
        source_index = _parse_optional_int(row_dict.get("Source_Index", ""))
        if source_index is None:
            return None  # source missing — definitely skip

        return TripletRow(
            source_index=source_index,
            sink_index=_parse_optional_int(row_dict.get("Sink_Index", "")),
            hq_index=_parse_optional_int(row_dict.get("HQ_Index", "")),
            source_sensor_ns=_parse_optional_int(row_dict.get("Source_Sensor_ns", "")),
            sink_sensor_ns=_parse_optional_int(row_dict.get("Sink_Sensor_ns", "")),
            source_wall_ns=_parse_optional_int(row_dict.get("Source_Wall_ns", "")),
            hq_wall_ns=_parse_optional_int(row_dict.get("HQ_Wall_ns", "")),
            status=row_dict.get("TripleStatus", "").strip(),
        )

    # ------------------------------------------------------------------
    # Batch preparation (read frames from .ts segments)
    # ------------------------------------------------------------------

    def _prepare_batch(
        self,
        rows: List[TripletRow],
        batch_num: int,
    ) -> List[Tuple[int, TripletRow, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        For each matched row (both source_index AND sink_index present),
        read original-res frames from the segment readers and produce
        resized copies.

        Returns list of:
            (csv_row_id, row, cam1_orig, cam2_orig, cam1_resized, cam2_resized)

        Rows with missing sink are silently skipped.
        If a frame cannot be read (segment unavailable after stop_event),
        the row is also skipped.
        """
        matched = []

        for csv_row_id, row in enumerate(rows):
            if self._stop_event.is_set():
                break

            # Skip unmatched rows
            if row.sink_index is None:
                continue

            # Read source frame
            cam1_orig = self._source_reader.read_frame_at(row.source_index)
            if cam1_orig is None:
                print(f"[TripletCSVReader] ⚠️  DECODE FAILED: batch={batch_num}, "
                      f"csv_row={csv_row_id}, source_index={row.source_index} — skipping row")
                if self._profiler:
                    self._profiler.record(
                        "reader_decode_failed_source", 1.0,
                        write_immediately=True, batch=batch_num, thread="TripletCSVReader",
                        metadata=f"source_index={row.source_index},csv_row={csv_row_id}",
                    )
                continue

            # Read sink frame
            cam2_orig = self._sink_reader.read_frame_at(row.sink_index)
            if cam2_orig is None:
                print(f"[TripletCSVReader] ⚠️  DECODE FAILED: batch={batch_num}, "
                      f"csv_row={csv_row_id}, sink_index={row.sink_index} — skipping row")
                if self._profiler:
                    self._profiler.record(
                        "reader_decode_failed_sink", 1.0,
                        write_immediately=True, batch=batch_num, thread="TripletCSVReader",
                        metadata=f"sink_index={row.sink_index},csv_row={csv_row_id}",
                    )
                continue

            # Resize to TrackNet input dimensions (WIDTH=512, HEIGHT=288)
            cam1_resized = cv2.resize(cam1_orig, (WIDTH, HEIGHT))
            cam2_resized = cv2.resize(cam2_orig, (WIDTH, HEIGHT))

            matched.append((csv_row_id, row, cam1_orig, cam2_orig, cam1_resized, cam2_resized))

        return matched

    # ------------------------------------------------------------------
    # Batch push (to TracknetBuffers + OriginalFrameBuffer)
    # ------------------------------------------------------------------

    def _push_batch(
        self,
        batch_num: int,
        matched: List[Tuple[int, TripletRow, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    ) -> None:
        """
        Push resized frames to TracknetBuffer_1/2 and original-res frames to
        OriginalFrameBuffer.

        TracknetBuffer.add_frame() is non-blocking (returns False if full).
        We call wait_for_space() first to block until there is room.

        After all frames are pushed, call add_batch_info() for both cameras
        so InferenceWorker knows how many frames are in this batch.
        """
        orig_width = self._source_reader.original_width
        orig_height = self._source_reader.original_height

        for csv_row_id, row, cam1_orig, cam2_orig, cam1_resized, cam2_resized in matched:
            if self._stop_event.is_set():
                break

            source_ts_ms = (row.source_sensor_ns / 1e6) if row.source_sensor_ns is not None else 0.0
            sink_ts_ms = (row.sink_sensor_ns / 1e6) if row.sink_sensor_ns is not None else 0.0

            # --- Push cam1 (source) to TracknetBuffer_1 ---
            while not self._stop_event.is_set():
                if self._tracknet_buffer_1.wait_for_space(timeout=0.5):
                    self._tracknet_buffer_1.add_frame(
                        frame=cam1_resized,
                        frame_index=row.source_index,
                        timestamp_ms=source_ts_ms,
                        camera_id=self._camera_1_id,
                        original_width=orig_width,
                        original_height=orig_height,
                        sensor_timestamp=row.source_sensor_ns,
                        tracknet_index=row.source_index,
                        is_placeholder=False,
                    )
                    break

            if self._stop_event.is_set():
                break

            # --- Push cam2 (sink) to TracknetBuffer_2 ---
            while not self._stop_event.is_set():
                if self._tracknet_buffer_2.wait_for_space(timeout=0.5):
                    self._tracknet_buffer_2.add_frame(
                        frame=cam2_resized,
                        frame_index=row.sink_index,
                        timestamp_ms=sink_ts_ms,
                        camera_id=self._camera_2_id,
                        original_width=orig_width,
                        original_height=orig_height,
                        sensor_timestamp=row.sink_sensor_ns,
                        tracknet_index=row.source_index,  # frame_id (global) = source_index
                        is_placeholder=False,
                    )
                    break

            if self._stop_event.is_set():
                break

            # --- Push original frames to OriginalFrameBuffer ---
            ok = self._original_buffer.add_frame(
                batch_num=batch_num,
                csv_row_id=csv_row_id,
                source_index=row.source_index,
                sink_index=row.sink_index,
                cam1_frame=cam1_orig,
                cam2_frame=cam2_orig,
                stop_event=self._stop_event,
            )
            if not ok:
                break

        # Signal InferenceWorker that a full batch has been pushed
        matched_count = len(matched)
        if matched_count > 0:
            self._tracknet_buffer_1.add_batch_info(self._camera_1_id, matched_count)
            self._tracknet_buffer_2.add_batch_info(self._camera_2_id, matched_count)
