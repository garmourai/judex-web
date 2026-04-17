#!/usr/bin/env python3
"""
Path/config defaults for hls_segment_watcher.py.
"""

from __future__ import annotations

WORK_ROOT_DEFAULT = "/home/pi/source_code"
SYNC_REPORTS_ROOT_DEFAULT = "/home/pi/judex-web/sync_reports"

DEST_HQ_DEFAULT = f"{SYNC_REPORTS_ROOT_DEFAULT}/ts_segments_hq"
DEST_SINK_DEFAULT = f"{SYNC_REPORTS_ROOT_DEFAULT}/ts_segments_sink"

REMOTE_TS_ROOT_SOURCE = "/home/pi/source_code/ts_segments"
REMOTE_TS_ROOT_SINK = "/home/pi/sink_code/ts_segments"
REMOTE_FRAME_LOG_ROOT_SOURCE = "/home/pi/source_code/streamed_packets"
REMOTE_FRAME_LOG_ROOT_SINK = "/home/pi/sink_code/streamed_packets"


def default_out_dir(track_id: str) -> str:
    return f"{SYNC_REPORTS_ROOT_DEFAULT}/segments_{track_id}"


def remote_ts_dir(is_sink: bool, track_id: str) -> str:
    root = REMOTE_TS_ROOT_SINK if is_sink else REMOTE_TS_ROOT_SOURCE
    return f"{root}/{track_id}"


def remote_playlist_path(stream: str, track_id: str) -> str:
    if stream in ("source", "hq"):
        return f"{REMOTE_TS_ROOT_SOURCE}/{track_id}/playlist.m3u8"
    return f"{REMOTE_TS_ROOT_SINK}/{track_id}/playlist.m3u8"


def remote_frame_log_path(stream: str, track_id: str) -> str:
    if stream in ("source", "hq"):
        return f"{REMOTE_FRAME_LOG_ROOT_SOURCE}/{track_id}/hls_frame_log.csv"
    return f"{REMOTE_FRAME_LOG_ROOT_SINK}/{track_id}/hls_frame_log.csv"
