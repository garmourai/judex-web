#!/usr/bin/env bash
# Generate sample HLS (.ts) stream for local testing.
# Requires ffmpeg. Usage: ./scripts/generate-sample-stream.sh [input.mp4]

set -e
INPUT="${1:-}"
OUTPUT_DIR="$(dirname "$0")/../stream-output"
mkdir -p "$OUTPUT_DIR"

if [ -z "$INPUT" ] || [ ! -f "$INPUT" ]; then
  echo "No input file. Generating a short test pattern with ffmpeg..."
  ffmpeg -f lavfi -i "testsrc=duration=10:size=1280x720:rate=30" \
    -f lavfi -i "sine=frequency=1000:duration=10" \
    -c:v libx264 -preset fast -tune zerolatency -c:a aac \
    -hls_time 2 -hls_playlist_type vod \
    -hls_segment_filename "$OUTPUT_DIR/segment%03d.ts" \
    "$OUTPUT_DIR/playlist.m3u8" -y
  echo "Done. Playlist: $OUTPUT_DIR/playlist.m3u8"
  exit 0
fi

echo "Converting $INPUT to HLS in $OUTPUT_DIR..."
ffmpeg -i "$INPUT" -c:v libx264 -preset fast -c:a aac \
  -hls_time 2 -hls_playlist_type vod \
  -hls_segment_filename "$OUTPUT_DIR/segment%03d.ts" \
  "$OUTPUT_DIR/playlist.m3u8" -y
echo "Done. Playlist: $OUTPUT_DIR/playlist.m3u8"
