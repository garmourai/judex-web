#!/usr/bin/env python3
"""
Build side-by-side triplet verification video for stream 1542.

Panel layout (per stream):
  ┌─────────────────┐  ← LABEL_H (30px) — label strip, no video overlap
  │  SOURCE idx=N   │
  ├─────────────────┤  ← VIDEO_H (360px) — full letterboxed video, never cropped
  │   video frame   │
  └─────────────────┘

Combined frame: [ SOURCE panel | SINK panel | HQ panel ] + bottom status bar

Usage:
    python3 make_triplet_video.py            # all chunks, 1000 frames each
    python3 make_triplet_video.py --chunk 3  # single chunk by index
    python3 make_triplet_video.py --start 0 --end 999  # explicit row range
"""

import argparse
import bisect
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────
BASE         = Path("/mnt/data/mar30_test")
TRIPLE_CSV   = BASE / "segments_1541/sync/hls_sync_1541_triple.csv"
SOURCE_M3U8  = BASE / "ts_segments_source/1541/playlist.m3u8"
SINK_M3U8    = BASE / "ts_segments_sink/1541/playlist.m3u8"
HQ_M3U8      = BASE / "ts_segments_hq/1541/playlist.m3u8"
OUT_DIR      = BASE / "segments_1541/sync/triplet_videos"

# ── output settings ────────────────────────────────────────────────────────
PANEL_W    = 640
VIDEO_H    = 360   # video area — full frame, never cropped
LABEL_H    = 30    # label strip above video
STATUS_H   = 28    # status bar below the combined frame
PANEL_H    = LABEL_H + VIDEO_H          # per-panel height
FRAME_H    = PANEL_H + STATUS_H         # total output frame height
FPS        = 30.0
CHUNK_SIZE = 1000


# ── frame helpers ──────────────────────────────────────────────────────────

def _letterbox(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Fit img into target_w x target_h preserving aspect ratio, pad black."""
    src_h, src_w = img.shape[:2]
    scale  = min(target_w / src_w, target_h / src_h)
    new_w  = int(src_w * scale)
    new_h  = int(src_h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas  = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_off   = (target_w - new_w) // 2
    y_off   = (target_h - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def _make_panel(video: np.ndarray, label: str, ok: bool) -> np.ndarray:
    """
    Build one panel: 30px label strip on top, then the full video below.
    Video is never overlaid — it occupies its own rows.
    """
    panel = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
    # label strip
    color = (0, 220, 0) if ok else (0, 0, 220)
    cv2.putText(panel, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    # video below label strip
    panel[LABEL_H:LABEL_H + VIDEO_H, :] = video
    return panel


def _status_bar(text: str, total_w: int) -> np.ndarray:
    bar = np.full((STATUS_H, total_w, 3), 20, dtype=np.uint8)
    cv2.putText(bar, text, (8, STATUS_H - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (220, 220, 220), 1, cv2.LINE_AA)
    return bar


# ── HLS helpers ────────────────────────────────────────────────────────────

def parse_playlist(m3u8: Path) -> list[tuple[float, Path]]:
    base = m3u8.parent
    out, pending = [], None
    for raw in m3u8.read_text().splitlines():
        line = raw.strip()
        if line.startswith("#EXTINF:"):
            try:
                pending = float(line.split(":")[1].split(",")[0])
            except Exception:
                pending = 4.0
        elif line and not line.startswith("#"):
            out.append((pending or 4.0, base / line))
            pending = None
    return out


class HLSProvider:
    """Sequential frame provider over HLS TS segments (no backwards seek)."""

    def __init__(self, m3u8: Path, fps: float):
        self.fps   = fps
        self.black = np.zeros((VIDEO_H, PANEL_W, 3), dtype=np.uint8)
        self.segs  = parse_playlist(m3u8)

        self.cum_starts:  list[int] = []
        self.seg_lengths: list[int] = []
        cum = 0
        for dur, _ in self.segs:
            n = max(1, int(round(dur * fps)))
            self.cum_starts.append(cum)
            self.seg_lengths.append(n)
            cum += n
        self.total = cum

        self.seg_i    = 0
        self.seg_pos  = 0
        self.next_gi  = 0
        self.cap: cv2.VideoCapture | None = None
        self.seg_dead = False

        self._last_gi:    int | None       = None
        self._last_frame: np.ndarray | None = None
        self._last_ok:    bool             = False

    def _open_seg(self):
        if self.cap:
            self.cap.release()
        self.cap = None
        if self.seg_i >= len(self.segs):
            return
        _, path = self.segs[self.seg_i]
        self.cap = cv2.VideoCapture(str(path))
        self.seg_dead = not self.cap.isOpened()
        self.seg_pos  = 0

    def _next_seg(self):
        if self.cap:
            self.cap.release()
        self.cap = None
        self.seg_i += 1
        if self.seg_i < len(self.segs):
            self._open_seg()

    def get(self, target: int | None) -> tuple[np.ndarray, bool]:
        if target is None:
            return self.black.copy(), False
        if target == self._last_gi and self._last_frame is not None:
            return self._last_frame.copy(), self._last_ok
        if target < self.next_gi or target >= self.total:
            return self.black.copy(), False

        # jump to the right segment
        desired_seg = bisect.bisect_right(self.cum_starts, target) - 1
        if desired_seg > self.seg_i:
            if self.cap:
                self.cap.release()
            self.cap     = None
            self.seg_i   = desired_seg
            self.next_gi = self.cum_starts[desired_seg]
            self._open_seg()

        frame_img, frame_ok = self.black, False
        while self.next_gi <= target:
            if self.seg_i >= len(self.segs):
                break
            if self.cap is None:
                self._open_seg()
            if self.seg_pos >= self.seg_lengths[self.seg_i]:
                self._next_seg()
                continue

            gi = self.next_gi
            if self.seg_dead:
                ok, img = False, self.black
            else:
                ok, raw = self.cap.read()
                if not ok or raw is None:
                    self.seg_dead = True
                    ok, img = False, self.black
                else:
                    img = _letterbox(raw, PANEL_W, VIDEO_H)

            self.next_gi += 1
            self.seg_pos += 1

            if gi == target:
                frame_img, frame_ok = img, ok
                break

        self._last_gi, self._last_frame, self._last_ok = target, frame_img.copy(), frame_ok
        return frame_img.copy(), frame_ok

    def close(self):
        if self.cap:
            self.cap.release()
        self.cap = None


# ── chunk builder ────────────────────────────────────────────────────────────

def build_chunk(rows: list[dict], chunk_idx: int, start_row: int,
                src_p: HLSProvider, snk_p: HLSProvider, hq_p: HLSProvider,
                out_dir: Path) -> Path:

    total_w  = PANEL_W * 3
    out_path = out_dir / f"triplet_chunk_{chunk_idx:04d}_{len(rows)}f.mp4"
    writer   = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (total_w, FRAME_H),
    )

    for i, row in enumerate(rows):
        ts  = row.get("TripleStatus", "").strip()
        si  = int(row["Source_Index"].strip())
        ski = row.get("Sink_Index", "").strip()
        hqi = row.get("HQ_Index",   "").strip()
        ski = int(ski) if ski else None
        hqi = int(hqi) if hqi else None

        fs, oks = src_p.get(si)
        fk, okk = snk_p.get(ski)
        fh, okh = hq_p.get(hqi)

        ps = _make_panel(fs, f"SOURCE {si}", oks)
        pk = _make_panel(fk, f"SINK   {ski if ski is not None else '-'}", okk)
        ph = _make_panel(fh, f"HQ     {hqi if hqi is not None else '-'}", okh)

        panels  = np.hstack([ps, pk, ph])
        status  = (f"row={start_row+i}  src={si}  snk={ski}  hq={hqi}  "
                   f"status={ts}  "
                   f"src={'OK' if oks else 'MISS'}  "
                   f"snk={'OK' if okk else 'MISS'}  "
                   f"hq={'OK'  if okh else 'MISS'}")
        bar     = _status_bar(status, total_w)
        frame   = np.vstack([panels, bar])
        writer.write(frame)

    writer.release()
    return out_path


# ── main ─────────────────────────────────────────────────────────────────────

def load_rows() -> list[dict]:
    rows = []
    with TRIPLE_CSV.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("Source_Index", "").strip():
                rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk",      type=int, default=None, help="Single chunk index to generate")
    parser.add_argument("--start",      type=int, default=None, help="Start row")
    parser.add_argument("--end",        type=int, default=None, help="End row inclusive")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = load_rows()
    total    = len(all_rows)
    cs       = args.chunk_size

    if args.start is not None:
        start = args.start
        end   = args.end if args.end is not None else min(start + cs - 1, total - 1)
        chunks_to_run = [(0, start, end)]
    elif args.chunk is not None:
        ci    = args.chunk
        start = ci * cs
        end   = min(start + cs - 1, total - 1)
        if start >= total:
            print(f"Chunk {ci} out of range (total rows: {total})")
            sys.exit(1)
        chunks_to_run = [(ci, start, end)]
    else:
        # default: all chunks
        chunks_to_run = []
        for ci_start in range(0, total, cs):
            ci = ci_start // cs
            chunks_to_run.append((ci, ci_start, min(ci_start + cs - 1, total - 1)))

    print(f"Total rows : {total}")
    print(f"Panel      : {PANEL_W}x{PANEL_H}  (video area {PANEL_W}x{VIDEO_H} + {LABEL_H}px label)")
    print(f"Output     : {PANEL_W*3}x{FRAME_H}  @{FPS}fps")
    print(f"Chunks     : {len(chunks_to_run)}  ({cs} frames each)")
    print(f"Output dir : {OUT_DIR}")
    print()

    src_p = HLSProvider(SOURCE_M3U8, FPS)
    snk_p = HLSProvider(SINK_M3U8,   FPS)
    hq_p  = HLSProvider(HQ_M3U8,     FPS)

    for ci, start, end in chunks_to_run:
        chunk_rows = all_rows[start:end + 1]
        print(f"  chunk {ci:04d}  rows {start}..{end}  ({len(chunk_rows)} frames) ...", end="", flush=True)
        out = build_chunk(chunk_rows, ci, start, src_p, snk_p, hq_p, OUT_DIR)
        print(f"  -> {out.name}")

    src_p.close()
    snk_p.close()
    hq_p.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
