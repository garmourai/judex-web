"""
M3U8 Segment Reader for Judex Pipeline

Reads frames sequentially from HLS .ts segment files.
Segment boundaries are derived from the playlist.m3u8 file (EXTINF durations),
not from the segment CSV files.  The reader polls the m3u8 file until a
segment's entry appears, then uses OpenCV's CAP_PROP_FRAME_COUNT from the
.ts file as the authoritative frame count.
"""

import importlib.util
import os
import threading
import time
from typing import Dict, List, Optional

import cv2
import numpy as np


_here = os.path.dirname(os.path.abspath(__file__))
_spec_hls = importlib.util.spec_from_file_location(
    "hls_segment_manifest",
    os.path.join(_here, "hls_segment_manifest.py"),
)
if _spec_hls is None or _spec_hls.loader is None:
    raise ImportError("cannot load hls_segment_manifest.py next to m3u8_reader")
_hls_manifest = importlib.util.module_from_spec(_spec_hls)
_spec_hls.loader.exec_module(_hls_manifest)
append_segment_manifest_row = _hls_manifest.append_segment_manifest_row
load_segment_manifest = _hls_manifest.load_segment_manifest
manifest_path = _hls_manifest.manifest_path


def assert_hls_segment_inputs_or_raise(
    segments_dir: str,
    *,
    manifest_dir: Optional[str] = None,
    m3u8_filename: str = "playlist.m3u8",
) -> None:
    """
    Require segments_dir and playlist.m3u8 to exist.
    If hls_segment_frame_index.csv exists under manifest_dir (default: segments_dir),
    it must pass load_segment_manifest (otherwise raise — no silent repair).
    """
    if not os.path.isdir(segments_dir):
        raise FileNotFoundError(f"HLS segments directory does not exist: {segments_dir!r}")
    m3u8_path = os.path.join(segments_dir, m3u8_filename)
    if not os.path.isfile(m3u8_path):
        raise FileNotFoundError(f"HLS playlist not found: {m3u8_path!r}")
    md = manifest_dir if manifest_dir is not None else segments_dir
    mp = manifest_path(md)
    if os.path.isfile(mp):
        loaded = load_segment_manifest(md)
        if loaded is None:
            raise ValueError(
                f"Invalid HLS segment manifest (fix or remove): {mp!r} — "
                "expected contiguous segment_index rows and consistent cumulative_start_frame / frame_count."
            )


class M3U8SegmentReader:
    """
    Sequential frame reader backed by HLS .ts segment files.

    Video frames are stored across numbered segment files (seg_00000.ts,
    seg_00001.ts, ...).  A parallel playlist.m3u8 (in segments_dir) records
    EXTINF durations and is used to detect when each segment is finalised.

    Because both files are written incrementally, this reader polls the
    playlist.m3u8 until the required segment entry appears, then polls the
    .ts file until it exists, then reads its frame count via OpenCV.

    Source_Index values in triplet.csv are 0-based global frame indices that
    map directly to physical frames:
        segment 0 → global frames  [0, N0-1]
        segment 1 → global frames  [N0, N0+N1-1]
        ...

    read_frame_at() is forward-only — Source_Index values must be
    monotonically non-decreasing across calls.
    """

    def __init__(
        self,
        segments_dir: str,
        stop_event: threading.Event,
        poll_interval: float = 0.5,
        m3u8_filename: str = "playlist.m3u8",
        manifest_dir: Optional[str] = None,
    ):
        """
        Args:
            segments_dir: Directory containing seg_XXXXX.ts files and playlist.m3u8
                          (e.g. /mnt/data/mar30_test/ts_segments_source/1541/)
            stop_event: Checked during polling — returns None if set.
            poll_interval: Seconds to sleep between existence checks.
            m3u8_filename: Name of the m3u8 playlist file inside segments_dir.
            manifest_dir: Directory for hls_segment_frame_index.csv (default: same as segments_dir).
                          Triplet pipeline uses unique_output_dir/reader/source|sink.
        """
        self._segments_dir = segments_dir
        self._manifest_dir = manifest_dir if manifest_dir is not None else segments_dir
        self._stop_event = stop_event
        self._poll_interval = poll_interval
        self._m3u8_path = os.path.join(segments_dir, m3u8_filename)

        # cumulative_offsets[i] = first global frame index in segment i
        # Built lazily as segment entries appear in the m3u8, or loaded from
        # hls_segment_frame_index.csv when present and valid.
        self._cumulative_offsets: List[int] = []  # len = number of known segments
        self._frame_counts: List[int] = []          # frame count per segment

        mp = manifest_path(self._manifest_dir)
        if os.path.isfile(mp):
            loaded = load_segment_manifest(self._manifest_dir)
            if loaded is None:
                raise ValueError(
                    f"Invalid HLS segment manifest: {mp!r} — "
                    "file exists but failed validation (contiguous segments, offset chain)."
                )
            self._cumulative_offsets = list(loaded[0])
            self._frame_counts = list(loaded[1])
            print(
                f"[M3U8SegmentReader] Loaded {len(self._frame_counts)} segments from {mp!r}"
            )

        # Currently open VideoCapture
        self._cap: Optional[cv2.VideoCapture] = None
        self._current_seg_idx: int = -1
        self._current_seg_pos: int = 0  # local frame position within current segment

        # Original frame dimensions (read from first opened segment)
        self._orig_width: Optional[int] = None
        self._orig_height: Optional[int] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def original_width(self) -> Optional[int]:
        return self._orig_width

    @property
    def original_height(self) -> Optional[int]:
        return self._orig_height

    def read_frame_at(self, global_frame_index: int) -> Optional[np.ndarray]:
        """
        Return the frame at global_frame_index (0-based).

        Waits (polls) until the required segment appears in playlist.m3u8
        and the .ts file is available.
        Returns None if stop_event is set while waiting.

        Calls must have monotonically non-decreasing global_frame_index.
        """
        # Ensure we know which segment this frame is in
        seg_idx = self._find_segment(global_frame_index)
        if seg_idx is None:
            return None  # stop_event was set

        local_offset = global_frame_index - self._cumulative_offsets[seg_idx]

        # Open (or advance to) the correct segment
        if self._current_seg_idx != seg_idx:
            if not self._open_segment(seg_idx):
                return None
            # Seek to local_offset from the beginning of this segment
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, local_offset)
            self._current_seg_pos = local_offset
        else:
            # Same segment — advance forward if needed
            if local_offset > self._current_seg_pos:
                skip = local_offset - self._current_seg_pos
                for _ in range(skip):
                    self._cap.grab()
                self._current_seg_pos = local_offset
            elif local_offset < self._current_seg_pos:
                # Backward seek within same segment (shouldn't happen, but handle it)
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, local_offset)
                self._current_seg_pos = local_offset

        ret, frame = self._cap.read()
        if not ret:
            total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"[M3U8SegmentReader] ⚠️  DECODE FAILED: global_frame={global_frame_index}, "
                  f"seg={seg_idx}, local_offset={local_offset}, "
                  f"seg_total_frames={total_frames}, ts={self._segment_ts_path(seg_idx)}")
            return None

        self._current_seg_pos = local_offset + 1
        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            self._current_seg_idx = -1

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _segment_ts_path(self, seg_idx: int) -> str:
        return os.path.join(self._segments_dir, f"seg_{seg_idx:05d}.ts")

    # ------------------------------------------------------------------
    # M3U8 parsing
    # ------------------------------------------------------------------

    def _parse_m3u8_segments(self) -> Dict[int, float]:
        """
        Read and parse the m3u8 playlist file.

        Returns {seg_idx: extinf_duration_seconds} for every segment entry
        present so far.  Reads the file in one shot to avoid partial-read
        artefacts.
        """
        if not os.path.exists(self._m3u8_path):
            return {}

        with open(self._m3u8_path, "r") as f:
            content = f.read()

        durations: Dict[int, float] = {}
        pending_duration: Optional[float] = None

        for line in content.splitlines():
            line = line.strip()
            if line.startswith("#EXTINF:"):
                # e.g.  #EXTINF:5.999944,
                try:
                    duration_str = line[len("#EXTINF:"):].split(",")[0]
                    pending_duration = float(duration_str)
                except ValueError:
                    pending_duration = None
            elif line.startswith("seg_") and line.endswith(".ts"):
                # e.g.  seg_00000.ts
                try:
                    seg_name = os.path.splitext(line)[0]  # "seg_00000"
                    seg_idx = int(seg_name.split("_")[1])
                    if pending_duration is not None:
                        durations[seg_idx] = pending_duration
                        pending_duration = None
                except (ValueError, IndexError):
                    pass

        return durations

    # ------------------------------------------------------------------
    # Segment loading (replaces CSV-based loading)
    # ------------------------------------------------------------------

    def _load_segment_frame_count(self, seg_idx: int) -> Optional[int]:
        """
        Wait until segment seg_idx appears in playlist.m3u8, then return the
        actual frame count from the .ts file via OpenCV.

        Returns frame count, or None if stop_event is set while waiting.
        """
        # 1. Poll m3u8 until the segment entry appears
        while not self._stop_event.is_set():
            durations = self._parse_m3u8_segments()
            if seg_idx in durations:
                break
            time.sleep(self._poll_interval)
        else:
            return None  # stop_event set

        # 2. The .ts file should be present once listed in the m3u8;
        #    poll briefly in case of filesystem propagation lag.
        ts_path = self._segment_ts_path(seg_idx)
        while not os.path.exists(ts_path):
            if self._stop_event.is_set():
                return None
            time.sleep(self._poll_interval)

        # 3. Use OpenCV as the authoritative frame count
        cap = cv2.VideoCapture(ts_path)
        if cap.isOpened():
            cv_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if cv_count > 0:
                return cv_count
            cap.release()

        # Fallback: estimate from EXTINF duration × FPS
        # (Only reached if OpenCV can't determine frame count)
        durations = self._parse_m3u8_segments()
        duration = durations.get(seg_idx, 0.0)
        if duration > 0:
            # Try to get FPS from an already-opened segment
            fps = self._get_fps_estimate()
            if fps > 0:
                estimated = round(duration * fps)
                print(f"[M3U8SegmentReader] seg {seg_idx}: OpenCV returned 0, "
                      f"estimating {estimated} frames from EXTINF {duration:.4f}s × {fps:.2f}fps")
                return estimated

        print(f"[M3U8SegmentReader] WARNING: seg {seg_idx}: could not determine frame count")
        return None

    def _get_fps_estimate(self) -> float:
        """Return FPS from an already-open segment, or 0 if unknown."""
        if self._cap is not None and self._cap.isOpened():
            fps = self._cap.get(cv2.CAP_PROP_FPS)
            if fps > 0:
                return fps
        # Try the first segment if it exists
        ts0 = self._segment_ts_path(0)
        if os.path.exists(ts0):
            cap = cv2.VideoCapture(ts0)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                if fps > 0:
                    return fps
        return 0.0

    def _ensure_segment_known(self, seg_idx: int) -> bool:
        """
        Ensure cumulative_offsets is populated up to and including seg_idx.
        Waits for segment entries to appear in the m3u8 as needed.
        Returns False if stop_event is set.
        """
        while len(self._cumulative_offsets) <= seg_idx:
            next_idx = len(self._frame_counts)
            frame_count = self._load_segment_frame_count(next_idx)
            if frame_count is None:
                return False
            if self._frame_counts:
                offset = self._cumulative_offsets[-1] + self._frame_counts[-1]
            else:
                offset = 0
            self._cumulative_offsets.append(offset)
            self._frame_counts.append(frame_count)
            append_segment_manifest_row(
                self._manifest_dir, next_idx, offset, frame_count
            )
            print(f"[M3U8SegmentReader] seg {next_idx}: {frame_count} frames "
                  f"(cumulative offset {offset})")
        return True

    def _find_segment(self, global_frame_index: int) -> Optional[int]:
        """
        Find which segment contains global_frame_index.
        Loads more segment entries from m3u8 as needed.
        Returns segment index, or None if stop_event is set.
        """
        seg_idx = 0
        while True:
            if not self._ensure_segment_known(seg_idx):
                return None

            start = self._cumulative_offsets[seg_idx]
            end = start + self._frame_counts[seg_idx] - 1

            if start <= global_frame_index <= end:
                return seg_idx

            if global_frame_index < start:
                print(f"[M3U8SegmentReader] WARNING: global_frame_index={global_frame_index} "
                      f"is before segment {seg_idx} start={start}")
                return seg_idx  # Best effort

            seg_idx += 1

    def _open_segment(self, seg_idx: int) -> bool:
        """
        Close current segment (if open) and open seg_XXXXX.ts.
        The .ts file is guaranteed to exist (checked in _load_segment_frame_count).
        Returns False if stop_event is set.
        """
        ts_path = self._segment_ts_path(seg_idx)
        while not os.path.exists(ts_path):
            if self._stop_event.is_set():
                return False
            print(f"[M3U8SegmentReader] Waiting for segment file: {ts_path}")
            time.sleep(self._poll_interval)

        if self._cap is not None:
            self._cap.release()
            self._cap = None

        cap = cv2.VideoCapture(ts_path)
        if not cap.isOpened():
            print(f"[M3U8SegmentReader] Failed to open: {ts_path}")
            return False

        if self._orig_width is None:
            self._orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self._orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self._cap = cap
        self._current_seg_idx = seg_idx
        self._current_seg_pos = 0
        print(f"[M3U8SegmentReader] Opened segment {seg_idx}: {ts_path}")
        return True
