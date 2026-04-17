#!/usr/bin/env python3
"""
hls_segment_watcher.py

Single-script replacement for rsync_from_other_servers.sh.
On each poll cycle (default 2s):
  1. rsync *.ts + *.m3u8 only from each HQ/Sink Pi (no packet dirs)
  2. Split pass: for each stream (source/hq/sink), slice the growing frame log
     into per-segment CSVs using that stream's own m3u8 as the time reference
  3. Sync pass: full match on accumulated data each cycle; emit only newly
     committed source frames (hold back last 1 frame for look-ahead).
     Source is anchor. Output appended to growing sync CSVs.

Frame log CSV columns: frame_num, sensor_ts_ns, wall_ts_ns, is_keyframe, size_bytes, packet_count
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import subprocess
import sys
import time
from typing import Optional

# ── Config ────────────────────────────────────────────────────────────────────

def load_config(variables_dir: str, track_id_override: Optional[str] = None) -> dict:
    """Load config.yaml + track_video_index.json → {track_id, hq_ips, sink_ips}."""
    try:
        import yaml
    except ImportError:
        print("PyYAML required: pip install pyyaml", file=sys.stderr)
        sys.exit(1)

    cfg_path = os.path.join(variables_dir, "config.yaml")
    idx_path = os.path.join(variables_dir, "track_video_index.json")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    with open(idx_path, "r", encoding="utf-8") as f:
        tj = json.load(f)

    track_id = int(track_id_override) if track_id_override else int(tj.get("counter", 0))

    def ips(key: str) -> list[str]:
        v = cfg.get(key) or []
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        return [str(v).strip()] if v else []

    return {
        "track_id": str(track_id),
        "hq_ips": ips("SOURCE_IPS_CHECK"),
        "sink_ips": ips("SINK_IPS"),
    }

# ── SSH helpers ───────────────────────────────────────────────────────────────

_SSH_OPTS = [
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=10",
    "-o", "StrictHostKeyChecking=accept-new",
]


def ssh_run(ip: str, cmd: str, ssh_user: str, timeout: int = 30) -> Optional[str]:
    """Run a shell command on a remote host. Returns stdout or None on failure."""
    try:
        r = subprocess.run(
            ["ssh"] + _SSH_OPTS + [f"{ssh_user}@{ip}", cmd],
            capture_output=True, text=True, timeout=timeout,
        )
        return r.stdout if r.returncode == 0 else None
    except (subprocess.TimeoutExpired, OSError):
        return None


def fetch_remote_text(ip: str, remote_path: str, ssh_user: str) -> Optional[str]:
    """Fetch a complete remote file (e.g. m3u8 playlist)."""
    return ssh_run(ip, f"cat {remote_path}", ssh_user)


def fetch_first_data_row(ip: Optional[str], remote_path: str, ssh_user: str) -> Optional[dict]:
    """Fetch the first data row (row 0) of a frame log CSV to determine epoch_ns."""
    if ip is None:
        # Local file
        if not os.path.isfile(remote_path):
            return None
        with open(remote_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    return _parse_row(row)
                except (ValueError, KeyError):
                    pass
        return None
    else:
        text = ssh_run(ip, f"head -n 2 {remote_path}", ssh_user)
        if not text:
            return None
        rows = parse_frame_log_with_header(text)
        return rows[0] if rows else None


def fetch_remote_csv_tail(
    ip: str, remote_path: str, ssh_user: str, rows_consumed: int
) -> str:
    """
    Fetch only new rows from a remote CSV, skipping the header and
    already-consumed rows.
    tail -n +N where N = rows_consumed + 2  (1-indexed; +1 skips header)
    Returns raw CSV text (no header) or "" on failure.
    """
    skip = rows_consumed + 2
    text = ssh_run(ip, f"tail -n +{skip} {remote_path}", ssh_user)
    return text or ""

# ── Frame log I/O ─────────────────────────────────────────────────────────────

_FRAME_LOG_FIELDS = [
    "frame_num", "sensor_ts_ns", "wall_ts_ns",
    "is_keyframe", "size_bytes", "packet_count",
]


def _parse_row(row: dict) -> dict:
    return {
        "frame_num": int(row["frame_num"]),
        "SensorTimestamp": int(row["sensor_ts_ns"]),
        "WallClockTimestamp": int(row["wall_ts_ns"]),
        "is_keyframe": int(row["is_keyframe"]),
        "size_bytes": int(row["size_bytes"]),
        "packet_count": int(row["packet_count"]),
    }


def _parse_rows_reader(reader) -> list[dict]:
    rows = []
    for row in reader:
        try:
            rows.append(_parse_row(row))
        except (ValueError, KeyError):
            pass
    return rows


def parse_frame_log_with_header(text: str) -> list[dict]:
    return _parse_rows_reader(csv.DictReader(io.StringIO(text)))


def parse_frame_log_no_header(text: str) -> list[dict]:
    """Parse CSV rows that have NO header line (tail-fetched incremental data)."""
    return _parse_rows_reader(
        csv.DictReader(io.StringIO(text), fieldnames=_FRAME_LOG_FIELDS)
    )


def load_local_rows_incremental(path: str, rows_consumed: int) -> list[dict]:
    """Load only rows beyond rows_consumed from a local frame log CSV."""
    if not os.path.isfile(path):
        return []
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i < rows_consumed:
                continue
            try:
                rows.append(_parse_row(row))
            except (ValueError, KeyError):
                pass
    return rows


def write_frame_log_csv(path: str, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_FRAME_LOG_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({
                "frame_num": r["frame_num"],
                "sensor_ts_ns": r["SensorTimestamp"],
                "wall_ts_ns": r["WallClockTimestamp"],
                "is_keyframe": r["is_keyframe"],
                "size_bytes": r["size_bytes"],
                "packet_count": r["packet_count"],
            })

# ── M3U8 helpers ──────────────────────────────────────────────────────────────

def parse_m3u8_extinf_durations(text: str) -> list[float]:
    durations: list[float] = []
    for line in text.splitlines():
        m = re.match(r"#EXTINF:([\d.]+)", line.strip())
        if m:
            durations.append(float(m.group(1)))
    return durations


def cumulative_segment_intervals(durations: list[float]) -> list[tuple[float, float]]:
    cum, out = 0.0, []
    for d in durations:
        out.append((cum, cum + d))
        cum += d
    return out

# ── TS rsync ──────────────────────────────────────────────────────────────────

def rsync_ts_files(
    ip: str,
    is_sink: bool,
    track_id: str,
    dest_hq: str,
    dest_sink: str,
    ssh_user: str,
    dry_run: bool = False,
) -> bool:
    """rsync *.ts + *.m3u8 only from one remote Pi. Returns True on success."""
    if is_sink:
        remote = f"/home/pi/sink_code/ts_segments/{track_id}/"
        local = f"{dest_sink.rstrip('/')}/{track_id}/"
    else:
        remote = f"/home/pi/source_code/ts_segments/{track_id}/"
        local = f"{dest_hq.rstrip('/')}/{track_id}/"

    os.makedirs(local, exist_ok=True)

    rsync_args = [
        "rsync", "-a", "--itemize-changes",
        "--partial", "--partial-dir=.rsync-partial",
        "--include=*/", "--include=*.ts", "--include=*.m3u8", "--exclude=*",
        "--timeout=30",
        "-e", "ssh " + " ".join(_SSH_OPTS),
    ]
    if dry_run:
        rsync_args.append("--dry-run")
    rsync_args += [f"{ssh_user}@{ip}:{remote}", local]

    try:
        r = subprocess.run(rsync_args, capture_output=True, text=True, timeout=120)
        new_files = [l.split()[-1] for l in r.stdout.splitlines() if l.startswith(">f")]
        label = "Sink" if is_sink else "HQ"
        if new_files:
            print(f"  rsync {label} {ip}: +{len(new_files)} file(s)  [{new_files[0]} .. {new_files[-1]}]")
        else:
            print(f"  rsync {label} {ip}: everything synced, nothing new")
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"  Warn: rsync timeout {ip}", file=sys.stderr)
        return False

# ── State helpers ─────────────────────────────────────────────────────────────

def load_state(path: str) -> dict:
    if os.path.isfile(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def save_state(path: str, state: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f, indent=2)

# ── Split pass ────────────────────────────────────────────────────────────────

def _stream_remote_paths(stream: str, track_id: str) -> tuple[str, str]:
    """Return (m3u8_remote_path, frame_log_remote_path) for hq or sink."""
    if stream == "hq":
        return (
            f"/home/pi/source_code/ts_segments/{track_id}/playlist.m3u8",
            f"/home/pi/source_code/streamed_packets/{track_id}/hls_frame_log.csv",
        )
    else:  # sink
        return (
            f"/home/pi/sink_code/ts_segments/{track_id}/playlist.m3u8",
            f"/home/pi/sink_code/streamed_packets/{track_id}/hls_frame_log.csv",
        )


def run_split_pass(
    stream: str,
    ip: Optional[str],
    track_id: str,
    out_dir: str,
    state: dict,
    stream_rows: list[dict],
    stream_intervals: list[tuple[float, float]],
    ssh_user: str,
    dry_run: bool = False,
) -> int:
    """
    Fetch new frame log rows + any new m3u8 segments for this stream, write
    per-segment CSVs for newly finalized segments.
    stream_rows and stream_intervals are in-memory lists updated in place.
    Returns number of new segment CSVs written.
    """
    rows_key = f"{stream}_rows_consumed"
    seg_key = f"{stream}_last_segment"
    rows_consumed = int(state.get(rows_key, 0))

    # ── Fetch m3u8 (only re-read when more segments may have arrived) ─────────
    if stream == "source":
        m3u8_local = f"/home/pi/source_code/ts_segments/{track_id}/playlist.m3u8"
        if not os.path.isfile(m3u8_local):
            return 0, None, None
        with open(m3u8_local, "r", errors="replace") as f:
            m3u8_text = f.read()
    else:
        m3u8_path, _ = _stream_remote_paths(stream, track_id)
        m3u8_text = fetch_remote_text(ip, m3u8_path, ssh_user)
        if not m3u8_text:
            print(f"  Warn: could not fetch m3u8 for {stream} {ip}", file=sys.stderr)
            return 0, None, None

    # Update in-memory intervals only with newly added segments
    durations = parse_m3u8_extinf_durations(m3u8_text)
    if len(durations) > len(stream_intervals):
        stream_intervals.clear()
        stream_intervals.extend(cumulative_segment_intervals(durations))

    # ── Fetch new frame log rows ───────────────────────────────────────────────
    if stream == "source":
        log_local = f"/home/pi/source_code/streamed_packets/{track_id}/hls_frame_log.csv"
        new_rows = load_local_rows_incremental(log_local, rows_consumed)
    else:
        _, log_path = _stream_remote_paths(stream, track_id)
        raw = fetch_remote_csv_tail(ip, log_path, ssh_user, rows_consumed)
        new_rows = parse_frame_log_no_header(raw) if raw.strip() else []

    stream_rows.extend(new_rows)
    state[rows_key] = rows_consumed + len(new_rows)

    if not stream_rows:
        return 0, None, None

    # ── Slice into segments ───────────────────────────────────────────────────
    intervals = stream_intervals
    if not intervals:
        return 0, None, None
    epoch_key = f"{stream}_epoch_ns"
    if epoch_key not in state:
        # Fetch the very first data row to anchor the epoch correctly,
        # even if we've already consumed many rows this session.
        if stream == "source":
            log_path = f"/home/pi/source_code/streamed_packets/{track_id}/hls_frame_log.csv"
            first = fetch_first_data_row(None, log_path, ssh_user)
        else:
            _, log_path = _stream_remote_paths(stream, track_id)
            first = fetch_first_data_row(ip, log_path, ssh_user)
        state[epoch_key] = first["SensorTimestamp"] if first else stream_rows[0]["SensorTimestamp"]
    epoch_ns = state[epoch_key]
    # Hold back last 1 frame so partial segments at the live edge aren't written prematurely
    if len(stream_rows) < 2:
        return 0, None, None
    horizon = (stream_rows[-1]["SensorTimestamp"] - epoch_ns) / 1e9
    last_seg = int(state.get(seg_key, -1))

    stream_out = os.path.join(out_dir, stream)
    os.makedirs(stream_out, exist_ok=True)

    written = 0
    first_frame: Optional[int] = None
    last_frame: Optional[int] = None
    for k, (t_lo, t_hi) in enumerate(intervals):
        if k <= last_seg:
            continue
        if t_hi > horizon:
            break
        seg_rows = [
            r for r in stream_rows
            if t_lo <= (r["SensorTimestamp"] - epoch_ns) / 1e9 < t_hi
        ]
        if seg_rows:
            if first_frame is None:
                first_frame = seg_rows[0]["frame_num"]
            last_frame = seg_rows[-1]["frame_num"]
        seg_path = os.path.join(stream_out, f"segment_{k:05d}.csv")
        if dry_run:
            print(f"  [dry] {stream}/segment_{k:05d}.csv: {len(seg_rows)} rows")
        else:
            write_frame_log_csv(seg_path, seg_rows)
        state[seg_key] = k
        written += 1

    return written, first_frame, last_frame

# ── Frame matching (from hls_stream_sync.py) ──────────────────────────────────

_WALL_MATCH_THRESHOLD_NS = 33_000_000


def _match_frames_sensor(
    sink_meta: list[dict],
    source_meta: list[dict],
) -> tuple[list[tuple[int, int]], Optional[int]]:
    """Align sink to source by SensorTimestamp (greedy, drift-corrected)."""
    pairs: list[tuple[int, int]] = []
    time_diff: Optional[int] = None
    src_times = [f["SensorTimestamp"] for f in source_meta]
    src_idx = 0

    for sk_idx, sk_frame in enumerate(sink_meta):
        if sk_idx == len(sink_meta) - 1 or src_idx == len(src_times) - 1:
            break
        if time_diff is None:
            time_diff = src_times[src_idx] - sk_frame["SensorTimestamp"]
        adj = sk_frame["SensorTimestamp"] + time_diff
        while (
            src_idx < len(src_times) - 1
            and abs(adj - src_times[src_idx + 1]) < abs(adj - src_times[src_idx])
        ):
            src_idx += 1
        if abs(adj - src_times[src_idx]) < 33_000_000:
            residual = adj - src_times[src_idx]
            time_diff -= residual
            pairs.append((sk_idx, src_idx))

    return pairs, time_diff


def _match_frames_wall_nearest(
    source_meta: list[dict],
    hq_meta: list[dict],
    threshold_ns: int = _WALL_MATCH_THRESHOLD_NS,
) -> list[tuple[int, int]]:
    """Align source to HQ by WallClockTimestamp (monotonic nearest-neighbor)."""
    hq_times = [f["WallClockTimestamp"] for f in hq_meta]
    pairs: list[tuple[int, int]] = []
    hq_idx = 0
    n_hq = len(hq_times)

    for src_idx, src_frame in enumerate(source_meta):
        if hq_idx >= n_hq:
            break
        src_t = src_frame["WallClockTimestamp"]
        while (
            hq_idx < n_hq - 1
            and abs(src_t - hq_times[hq_idx + 1]) < abs(src_t - hq_times[hq_idx])
        ):
            hq_idx += 1
        if abs(src_t - hq_times[hq_idx]) < threshold_ns:
            pairs.append((src_idx, hq_idx))

    return pairs

# ── Sync output helpers ───────────────────────────────────────────────────────

_TRIPLE_FIELDS = [
    "Source_Index", "Sink_Index", "HQ_Index",
    "Source_Sensor_ns", "Sink_Sensor_ns", "Source_Wall_ns", "HQ_Wall_ns",
    "TripleStatus",
]
_SS_FIELDS = [
    "EventSensor_ns", "RowType", "Source_Index", "Source_SensorTimestamp_ns",
    "Sink_Index", "Sink_SensorTimestamp_ns", "Sink_SensorTimestamp_Corrected_ns",
    "TimeDifference_ns", "TimeDifference_ms",
]
_SH_FIELDS = [
    "EventWall_ns", "RowType", "Source_Index", "Source_WallClock_ns",
    "HQ_Index", "HQ_WallClock_ns", "WallTimeDifference_ns", "WallTimeDifference_ms",
]
_SINK_HQ_FIELDS = [
    "EventWall_ns", "RowType", "Sink_Index", "Sink_WallClock_ns",
    "HQ_Index", "HQ_WallClock_ns", "WallTimeDifference_ns", "WallTimeDifference_ms",
]


def _ensure_header(path: str, fieldnames: list[str]) -> None:
    if not os.path.isfile(path):
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()


def _append_rows(path: str, fieldnames: list[str], rows: list[dict]) -> None:
    with open(path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writerows(rows)

# ── Sync pass ─────────────────────────────────────────────────────────────────

def run_sync_pass(
    track_id: str,
    source_meta: list[dict],
    sink_meta: list[dict],
    hq_meta: list[dict],
    state: dict,
    out_dir: str,
    dry_run: bool = False,
) -> int:
    """
    Run full matching on accumulated data. Emit source frames [source_committed,
    len(source)-1) — the last frame is held back until the next cycle confirms
    its match status.

    Appends newly committed rows to growing sync CSVs. Returns rows emitted.
    """
    if not source_meta or not sink_meta or not hq_meta:
        return 0, None, None
    if len(source_meta) < 2:
        return 0, None, None

    source_committed = int(state.get("sync_source_committed", 0))
    commit_up_to = len(source_meta) - 1  # hold back last frame

    # Guard: if state carries a committed count larger than what's in memory
    # (e.g. after restart), clamp so we don't skip newly loaded rows.
    if source_committed > commit_up_to:
        source_committed = 0
        state["sync_source_committed"] = 0
        state["sync_sink_hq_committed"] = 0

    if source_committed >= commit_up_to:
        return 0, None, None

    # ── Full matching ─────────────────────────────────────────────────────────
    sink_source_pairs, _ = _match_frames_sensor(sink_meta, source_meta)
    source_hq_pairs = _match_frames_wall_nearest(source_meta, hq_meta)
    sink_hq_pairs = _match_frames_wall_nearest(sink_meta, hq_meta)

    src_to_sink: dict[int, int] = {s: k for k, s in sink_source_pairs}
    src_to_hq: dict[int, int] = {s: h for s, h in source_hq_pairs}
    sink_to_hq: dict[int, int] = {s: h for s, h in sink_hq_pairs}

    sink_initial_offset_ns = (
        source_meta[0]["SensorTimestamp"] - sink_meta[0]["SensorTimestamp"]
    )

    sync_dir = os.path.join(out_dir, "sync")
    os.makedirs(sync_dir, exist_ok=True)
    prefix = f"hls_sync_{track_id}"

    triple_path = os.path.join(sync_dir, f"{prefix}_triple.csv")
    ss_path = os.path.join(sync_dir, f"{prefix}_source_sink_interleaved.csv")
    sh_path = os.path.join(sync_dir, f"{prefix}_source_hq_interleaved.csv")
    sink_hq_path = os.path.join(sync_dir, f"{prefix}_sink_hq_interleaved.csv")

    triple_rows: list[dict] = []
    ss_rows: list[dict] = []
    sh_rows: list[dict] = []
    sink_hq_rows: list[dict] = []

    # ── Emit newly committed source frames ────────────────────────────────────
    for k in range(source_committed, commit_up_to):
        src = source_meta[k]
        sk = src_to_sink.get(k)
        hq = src_to_hq.get(k)

        if sk is not None and hq is not None:
            status = "FULL"
        elif sk is not None:
            status = "MISSING_HQ"
        elif hq is not None:
            status = "MISSING_SINK"
        else:
            status = "SOURCE_ONLY"

        if dry_run:
            print(f"  [dry] sync src={k} {status}")
            continue

        # Triple CSV
        triple_rows.append({
            "Source_Index": k,
            "Sink_Index": "" if sk is None else sk,
            "HQ_Index": "" if hq is None else hq,
            "Source_Sensor_ns": src["SensorTimestamp"],
            "Sink_Sensor_ns": "" if sk is None else sink_meta[sk]["SensorTimestamp"],
            "Source_Wall_ns": src["WallClockTimestamp"],
            "HQ_Wall_ns": "" if hq is None else hq_meta[hq]["WallClockTimestamp"],
            "TripleStatus": status,
        })

        # Source–Sink interleaved
        src_ts = src["SensorTimestamp"]
        if sk is not None:
            sink_ts = sink_meta[sk]["SensorTimestamp"]
            corr = sink_ts + sink_initial_offset_ns
            diff = src_ts - corr
            ss_rows.append({
                "EventSensor_ns": src_ts, "RowType": "MATCHED",
                "Source_Index": k, "Source_SensorTimestamp_ns": src_ts,
                "Sink_Index": sk, "Sink_SensorTimestamp_ns": sink_ts,
                "Sink_SensorTimestamp_Corrected_ns": corr,
                "TimeDifference_ns": int(diff),
                "TimeDifference_ms": f"{diff / 1_000_000:.6f}",
            })
        else:
            ss_rows.append({
                "EventSensor_ns": src_ts, "RowType": "SOURCE_ONLY",
                "Source_Index": k, "Source_SensorTimestamp_ns": src_ts,
                "Sink_Index": "", "Sink_SensorTimestamp_ns": "",
                "Sink_SensorTimestamp_Corrected_ns": "",
                "TimeDifference_ns": "", "TimeDifference_ms": "",
            })

        # Source–HQ interleaved
        src_wall = src["WallClockTimestamp"]
        if hq is not None:
            hq_wall = hq_meta[hq]["WallClockTimestamp"]
            diff_w = src_wall - hq_wall
            sh_rows.append({
                "EventWall_ns": src_wall, "RowType": "MATCHED",
                "Source_Index": k, "Source_WallClock_ns": src_wall,
                "HQ_Index": hq, "HQ_WallClock_ns": hq_wall,
                "WallTimeDifference_ns": int(diff_w),
                "WallTimeDifference_ms": f"{diff_w / 1_000_000:.6f}",
            })
        else:
            sh_rows.append({
                "EventWall_ns": src_wall, "RowType": "SOURCE_ONLY",
                "Source_Index": k, "Source_WallClock_ns": src_wall,
                "HQ_Index": "", "HQ_WallClock_ns": "",
                "WallTimeDifference_ns": "", "WallTimeDifference_ms": "",
            })

    # ── Sink–HQ rows (sink-indexed, emit newly covered sink indices) ──────────
    sink_hq_committed = int(state.get("sync_sink_hq_committed", 0))
    if not dry_run:
        for si in range(sink_hq_committed, len(sink_meta) - 1):
            hi = sink_to_hq.get(si)
            sink_wall = sink_meta[si]["WallClockTimestamp"]
            if hi is not None:
                hq_wall = hq_meta[hi]["WallClockTimestamp"]
                diff_w = sink_wall - hq_wall
                sink_hq_rows.append({
                    "EventWall_ns": sink_wall, "RowType": "MATCHED",
                    "Sink_Index": si, "Sink_WallClock_ns": sink_wall,
                    "HQ_Index": hi, "HQ_WallClock_ns": hq_wall,
                    "WallTimeDifference_ns": int(diff_w),
                    "WallTimeDifference_ms": f"{diff_w / 1_000_000:.6f}",
                })
            else:
                sink_hq_rows.append({
                    "EventWall_ns": sink_wall, "RowType": "SINK_ONLY",
                    "Sink_Index": si, "Sink_WallClock_ns": sink_wall,
                    "HQ_Index": "", "HQ_WallClock_ns": "",
                    "WallTimeDifference_ns": "", "WallTimeDifference_ms": "",
                })
        state["sync_sink_hq_committed"] = len(sink_meta) - 1

    # ── Write to CSVs ─────────────────────────────────────────────────────────
    if not dry_run:
        if triple_rows:
            _ensure_header(triple_path, _TRIPLE_FIELDS)
            _append_rows(triple_path, _TRIPLE_FIELDS, triple_rows)
        if ss_rows:
            _ensure_header(ss_path, _SS_FIELDS)
            _append_rows(ss_path, _SS_FIELDS, ss_rows)
        if sh_rows:
            _ensure_header(sh_path, _SH_FIELDS)
            _append_rows(sh_path, _SH_FIELDS, sh_rows)
        if sink_hq_rows:
            _ensure_header(sink_hq_path, _SINK_HQ_FIELDS)
            _append_rows(sink_hq_path, _SINK_HQ_FIELDS, sink_hq_rows)

    written = commit_up_to - source_committed
    src_start = source_meta[source_committed]["frame_num"] if written > 0 else None
    src_end = source_meta[commit_up_to - 1]["frame_num"] if written > 0 else None
    state["sync_source_committed"] = commit_up_to
    return written, src_start, src_end

# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "HLS segment watcher: TS rsync + per-stream frame log split "
            "+ incremental source-anchored sync"
        )
    )
    p.add_argument("--track-id", default=None, help="Override track ID")
    p.add_argument("--work-root", default="/home/pi/source_code")
    p.add_argument(
        "--variables-dir", default=None,
        help="Directory with config.yaml + track_video_index.json "
             "(default: <work-root>/variable_files)",
    )
    p.add_argument(
        "--out-dir", default=None,
        help="Output root (default: /home/pi/judex-web/sync_reports/segments_<id>)",
    )
    p.add_argument("--dest-hq", default="/home/pi/judex-web/sync_reports/ts_segments_hq")
    p.add_argument("--dest-sink", default="/home/pi/judex-web/sync_reports/ts_segments_sink")
    p.add_argument("--ssh-user", default="pi")
    p.add_argument("--poll-seconds", type=float, default=4.0)
    p.add_argument("--reset-state", action="store_true", help="Clear state before run")
    p.add_argument("--dry-run", action="store_true", help="Print actions, no writes")
    args = p.parse_args()

    work_root = os.path.abspath(args.work_root)
    variables_dir = args.variables_dir or os.path.join(work_root, "variable_files")

    try:
        cfg = load_config(variables_dir, args.track_id)
    except Exception as e:
        print(f"Config load failed: {e}", file=sys.stderr)
        sys.exit(1)

    track_id = cfg["track_id"]
    hq_ips = cfg["hq_ips"]
    sink_ips = cfg["sink_ips"]

    # Use first IP from each list for frame log + sync (split handles all IPs)
    hq_ip = hq_ips[0] if hq_ips else None
    sink_ip = sink_ips[0] if sink_ips else None

    out_dir = args.out_dir or os.path.join(
        "/home/pi/judex-web/sync_reports", f"segments_{track_id}"
    )
    os.makedirs(out_dir, exist_ok=True)
    state_path = os.path.join(out_dir, "watcher_state.json")

    if args.reset_state and os.path.isfile(state_path):
        os.remove(state_path)

    state = load_state(state_path)

    # Always reload all frame log rows from scratch on startup.
    # last_segment and sync_source_committed are kept so existing files aren't rewritten.
    for stream in ("source", "hq", "sink"):
        state[f"{stream}_rows_consumed"] = 0

    print(f"Track:   {track_id}")
    print(f"HQ IPs:  {hq_ips}")
    print(f"Sink IPs:{sink_ips}")
    print(f"Out:     {out_dir}")
    print(f"Poll:    {args.poll_seconds}s")

    # In-memory accumulated frame rows and m3u8 intervals (read fully on first cycle, incremental after)
    source_rows: list[dict] = []
    hq_rows: list[dict] = []
    sink_rows: list[dict] = []
    source_intervals: list[tuple] = []
    hq_intervals: list[tuple] = []
    sink_intervals: list[tuple] = []

    iteration = 0
    try:
        while True:
            import datetime
            iteration += 1
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            print(f"\n=== Iter {iteration} {ts} ===")

            # ── 1. rsync TS files ─────────────────────────────────────────────
            for ip in hq_ips:
                rsync_ts_files(
                    ip, False, track_id,
                    args.dest_hq, args.dest_sink, args.ssh_user, args.dry_run,
                )
            for ip in sink_ips:
                rsync_ts_files(
                    ip, True, track_id,
                    args.dest_hq, args.dest_sink, args.ssh_user, args.dry_run,
                )

            # ── 2. Split pass ─────────────────────────────────────────────────
            n, f0, f1 = run_split_pass(
                "source", None, track_id, out_dir, state, source_rows,
                source_intervals, args.ssh_user, args.dry_run,
            )
            src_seg = int(state.get("source_last_segment", -1))
            print(f"  Split source: {len(source_rows)} rows in mem, {len(source_intervals)} segs in m3u8, up to seg {src_seg}"
                  + (f", +{n} new  frames [{f0} .. {f1}]" if n else ", no new segs"))

            if hq_ip:
                n, f0, f1 = run_split_pass(
                    "hq", hq_ip, track_id, out_dir, state, hq_rows,
                    hq_intervals, args.ssh_user, args.dry_run,
                )
                hq_seg = int(state.get("hq_last_segment", -1))
                print(f"  Split hq:     {len(hq_rows)} rows in mem, {len(hq_intervals)} segs in m3u8, up to seg {hq_seg}"
                      + (f", +{n} new  frames [{f0} .. {f1}]" if n else ", no new segs"))

            if sink_ip:
                n, f0, f1 = run_split_pass(
                    "sink", sink_ip, track_id, out_dir, state, sink_rows,
                    sink_intervals, args.ssh_user, args.dry_run,
                )
                sk_seg = int(state.get("sink_last_segment", -1))
                print(f"  Split sink:   {len(sink_rows)} rows in mem, {len(sink_intervals)} segs in m3u8, up to seg {sk_seg}"
                      + (f", +{n} new  frames [{f0} .. {f1}]" if n else ", no new segs"))

            # ── 3. Sync pass ──────────────────────────────────────────────────
            if source_rows and sink_rows and hq_rows:
                n, f0, f1 = run_sync_pass(
                    track_id, source_rows, sink_rows, hq_rows,
                    state, out_dir, args.dry_run,
                )
                committed = int(state.get("sync_source_committed", 0))
                print(f"  Sync:         committed {committed} src frames"
                      + (f", +{n} new  [{f0} .. {f1}]" if n else ", nothing new"))
            else:
                missing = []
                if not source_rows:
                    missing.append("source")
                if not sink_rows:
                    missing.append("sink")
                if not hq_rows:
                    missing.append("hq")
                print(f"  Sync:         waiting for {', '.join(missing)} frame log(s)")

            if not args.dry_run:
                save_state(state_path, state)

            print(f"--- sleeping {args.poll_seconds}s ---")
            time.sleep(args.poll_seconds)

    except KeyboardInterrupt:
        print("\nStopped.")
        if not args.dry_run:
            save_state(state_path, state)


if __name__ == "__main__":
    main()
