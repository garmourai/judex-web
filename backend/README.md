# Judex stream backend

Serves **live** HLS (`.m3u8` + `.ts`) from a directory. The frontend uses **hls.js** in low-latency live mode.

## Run locally

```bash
npm install
npm run dev
```

- **`STREAM_DIR`** – Directory where ffmpeg writes `playlist.m3u8` and segments (default: `./stream-output`).
- **`PORT`** – Server port (default `3014`; match `frontend/vite.config.ts` proxy).

## Live low-latency pipeline

1. **Start this backend** so `/stream` is available.
2. **Publish HLS with ffmpeg** into `STREAM_DIR` (or `./stream-output`). The playlist must **update over time** (sliding window); static VOD-style files are not the target use case.

Example (see `scripts/live-hls-example.sh`):

```bash
./scripts/live-hls-example.sh              # lavfi test pattern (simulated live)
./scripts/live-hls-example.sh /path/to.mp4 # file as simulated live (-re)
```

Environment tuning:

- `HLS_TIME` – Segment length in seconds (default `1`; lower ≈ lower latency).
- `HLS_DVR_MINUTES` – How many minutes of history to keep in the sliding playlist (default `10`). Roughly **DVR window ≈ `HLS_DVR_MINUTES × 60` seconds** (also `HLS_LIST_SIZE × HLS_TIME`).
- `HLS_LIST_SIZE` – Override segment count in the playlist (if unset, derived from `HLS_DVR_MINUTES`).

### Live + DVR (stay live, scrub back e.g. to 2:15)

You can watch at the **live edge** and **seek backward** within the time range still listed in the playlist (and still on disk). The frontend is configured for this (`maxBufferLength`, finite live duration for a proper seek bar).

1. **Encoder** must keep a **large enough** sliding window — e.g. default `HLS_DVR_MINUTES=10` (~10 minutes of rewind). If you only keep 6 one-second segments, you cannot jump to 2:15.
2. **Not infinite**: anything older than the window falls off the playlist (and with `delete_segments`, is removed from disk). For archive of the full day, use recording + VOD or a proper DVR server.

For **RTSP / SRT / real cameras**, replace the ffmpeg input with your source and keep the same `-f hls ...` block.

### Why cache headers matter

Live playlists **change** every segment. This server sends **`Cache-Control: no-cache`** on `.m3u8` so clients always fetch the latest playlist. `.ts` files are immutable and may be cached.

### Going lower latency

- Shorter segments (`HLS_TIME=1` or `0.5` if your encoder keeps up).
- Tune **hls.js** in the frontend (`liveSyncDurationCount`, `liveMaxLatencyDurationCount`).
- **LL-HLS** (partial segments, fMP4) needs ffmpeg features and a player that supports it; this stack uses classic TS HLS by default.

## API

- `GET /api/stream-url` – JSON `{ url }` for the default playlist.
- `GET /api/health` – Health check.

## Test VOD files (optional)

`./scripts/generate-sample-stream.sh` creates a **fixed** VOD playlist for UI testing only—not for live latency.
