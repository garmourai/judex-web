#!/usr/bin/env bash
# Example: publish **live** low-latency HLS into stream-output/ for the Judex backend to serve.
#
# Requirements: ffmpeg. Run the backend first (`npm run dev`). Then run this script.
#
# Replace INPUT with your source:
#   - Webcam (Linux): -f v4l2 -i /dev/video0 -f alsa -i default
#   - Test pattern (no camera): uses lavfi below
#   - File (simulated live): -re -i /path/to/video.mp4
#
# Tune latency: smaller -hls_time (e.g. 1) = lower latency, more requests.
#
# DVR (rewind / scrub within recent history):
#   Approx. window (seconds) ≈ HLS_LIST_SIZE × HLS_TIME
#   Example: HLS_TIME=1, HLS_LIST_SIZE=600 → ~10 minutes of seekable history.
#   Set HLS_DVR_MINUTES instead of HLS_LIST_SIZE to size the window.
#   Smaller list = less latency overhead, less rewind depth.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${STREAM_DIR:-$ROOT/stream-output}"
mkdir -p "$OUT"

INPUT="${1:-}"

# Low-latency-ish x264: frequent keyframes align with segment length
HLS_TIME="${HLS_TIME:-1}"
HLS_DVR_MINUTES="${HLS_DVR_MINUTES:-10}"
if [[ -n "${HLS_LIST_SIZE:-}" ]]; then
  :
else
  # Segments in sliding playlist ≈ minutes of DVR at this segment length
  HLS_LIST_SIZE=$(( HLS_DVR_MINUTES * 60 / HLS_TIME ))
  [[ "$HLS_LIST_SIZE" -lt 6 ]] && HLS_LIST_SIZE=6
fi
GOP=$(( HLS_TIME * 30 ))  # ~30 fps; adjust if your input fps differs

common_hls=(
  -f hls
  -hls_time "$HLS_TIME"
  -hls_list_size "$HLS_LIST_SIZE"
  -hls_flags delete_segments+append_list+program_date_time
  -hls_segment_filename "$OUT/seg_%05d.ts"
  "$OUT/playlist.m3u8"
)

if [[ -z "$INPUT" ]]; then
  echo "No input file: streaming test pattern (10s loop) as simulated live."
  ffmpeg -re -f lavfi -i "testsrc=size=1280x720:rate=30" \
    -f lavfi -i "sine=frequency=440" \
    -c:v libx264 -preset veryfast -tune zerolatency \
    -g "$GOP" -keyint_min "$GOP" -sc_threshold 0 \
    -c:a aac -ar 44100 \
    "${common_hls[@]}"
elif [[ -f "$INPUT" ]]; then
  echo "Simulated live from file: $INPUT (-re = realtime pace)"
  ffmpeg -re -i "$INPUT" \
    -c:v libx264 -preset veryfast -tune zerolatency \
    -g "$GOP" -keyint_min "$GOP" -sc_threshold 0 \
    -c:a aac -ar 44100 \
    "${common_hls[@]}"
else
  echo "Usage: $0 [input.mp4|mkv|...]"
  echo "  With no args, uses lavfi test pattern."
  exit 1
fi
