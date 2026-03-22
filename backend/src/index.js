import express from 'express';
import cors from 'cors';
import path from 'path';
import { fileURLToPath } from 'url';
import { promises as fs } from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT ?? 3014;
const DEFAULT_SNAPSHOT_MINUTES = Number(process.env.SNAPSHOT_DEFAULT_MINUTES ?? 5);

// Stream directory: set STREAM_DIR to serve your own HLS files (e.g. /path/to/folder with playlist.m3u8 + .ts)
const defaultStreamDir = path.join(__dirname, '..', 'stream-output');
const streamDir = process.env.STREAM_DIR
  ? path.resolve(process.env.STREAM_DIR)
  : defaultStreamDir;

// CORS for frontend (adjust origin in production)
app.use(cors());
app.use(express.json());

function parseMediaPlaylist(playlistContent) {
  const lines = playlistContent.split(/\r?\n/);
  const segments = [];
  let mediaSequence = 0;
  let currentDuration = null;
  let currentProgramDateTime = null;

  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line) continue;
    if (line.startsWith('#EXT-X-MEDIA-SEQUENCE:')) {
      mediaSequence = Number(line.slice('#EXT-X-MEDIA-SEQUENCE:'.length)) || 0;
      continue;
    }
    if (line.startsWith('#EXTINF:')) {
      const value = Number(line.slice('#EXTINF:'.length).split(',')[0]);
      currentDuration = Number.isFinite(value) ? value : 0;
      continue;
    }
    if (line.startsWith('#EXT-X-PROGRAM-DATE-TIME:')) {
      currentProgramDateTime = line.slice('#EXT-X-PROGRAM-DATE-TIME:'.length).trim();
      continue;
    }
    if (line.startsWith('#')) continue;
    if (currentDuration == null) continue;
    segments.push({
      uri: line,
      duration: currentDuration,
      programDateTime: currentProgramDateTime,
      sequence: mediaSequence + segments.length,
    });
    currentDuration = null;
    currentProgramDateTime = null;
  }

  return { mediaSequence, segments };
}

function mapSegmentUriForSnapshot(uri) {
  // Snapshot playlists live in /stream/snapshots, but segment files stay in /stream root.
  if (/^(https?:)?\/\//i.test(uri)) return uri;
  if (uri.startsWith('/')) return uri;
  if (uri.startsWith('../')) return uri;
  return `../${uri}`;
}

function buildVodSnapshotPlaylist(segments, startSequence) {
  const targetDuration = Math.max(1, Math.ceil(Math.max(...segments.map((s) => s.duration))));
  const body = [
    '#EXTM3U',
    '#EXT-X-VERSION:3',
    '#EXT-X-PLAYLIST-TYPE:VOD',
    `#EXT-X-TARGETDURATION:${targetDuration}`,
    `#EXT-X-MEDIA-SEQUENCE:${startSequence}`,
    '',
  ];

  for (const segment of segments) {
    if (segment.programDateTime) {
      body.push(`#EXT-X-PROGRAM-DATE-TIME:${segment.programDateTime}`);
    }
    body.push(`#EXTINF:${segment.duration.toFixed(3)},`);
    body.push(mapSegmentUriForSnapshot(segment.uri));
    body.push('');
  }

  body.push('#EXT-X-ENDLIST');
  body.push('');
  return body.join('\n');
}

// Snapshot folder serves playlist files only.
app.use('/stream/snapshots', (req, res, next) => {
  if (!req.path.endsWith('.m3u8')) {
    return res.status(403).json({ error: 'Only .m3u8 files are allowed in snapshots.' });
  }
  return next();
});

// Serve HLS (live: playlist must be re-fetched often; segments can be cached briefly)
app.use('/stream', express.static(streamDir, {
  setHeaders: (res, filePath) => {
    if (filePath.endsWith('.m3u8')) {
      res.set('Content-Type', 'application/vnd.apple.mpegurl');
      // Critical for live: browsers/CDNs must not serve a stale playlist
      res.set('Cache-Control', 'no-cache, no-store, must-revalidate');
      res.set('Pragma', 'no-cache');
    }
    if (filePath.endsWith('.ts')) {
      res.set('Content-Type', 'video/MP2T');
      // Each segment file is immutable once written
      res.set('Cache-Control', 'public, max-age=3600, immutable');
    }
  },
}));

// HLS playlist URL (frontend can use: /stream/playlist.m3u8)
app.get('/api/stream-url', (_req, res) => {
  const baseUrl = process.env.PUBLIC_URL ?? `http://localhost:${PORT}`;
  res.json({ url: `${baseUrl}/stream/playlist.m3u8` });
});

// Create a VOD snapshot from the last N minutes of the current live playlist.
app.post('/api/snapshots', async (req, res) => {
  try {
    const requested = Number(req.body?.minutes);
    const minutes = Number.isFinite(requested) && requested > 0
      ? requested
      : DEFAULT_SNAPSHOT_MINUTES;
    const clampedMinutes = Math.min(Math.max(minutes, 1), 120);
    const targetSeconds = clampedMinutes * 60;

    const livePlaylistPath = path.join(streamDir, 'playlist.m3u8');
    const content = await fs.readFile(livePlaylistPath, 'utf8');
    const { segments } = parseMediaPlaylist(content);
    if (!segments.length) {
      return res.status(409).json({ error: 'No segments available in live playlist yet.' });
    }

    let total = 0;
    const selected = [];
    for (let i = segments.length - 1; i >= 0; i -= 1) {
      selected.push(segments[i]);
      total += segments[i].duration;
      if (total >= targetSeconds) break;
    }
    selected.reverse();
    const firstSequence = selected[0].sequence;

    const id = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    const snapshotsDir = path.join(streamDir, 'snapshots');
    await fs.mkdir(snapshotsDir, { recursive: true });

    const snapshotFilename = `${id}.m3u8`;
    const snapshotPath = path.join(snapshotsDir, snapshotFilename);
    const snapshotContent = buildVodSnapshotPlaylist(selected, firstSequence);
    await fs.writeFile(snapshotPath, snapshotContent, 'utf8');

    return res.json({
      snapshotId: id,
      minutes: clampedMinutes,
      durationSec: Number(total.toFixed(3)),
      segmentCount: selected.length,
      url: `/stream/snapshots/${snapshotFilename}`,
    });
  } catch (error) {
    return res.status(500).json({
      error: 'Failed to create snapshot playlist.',
      details: error instanceof Error ? error.message : 'Unknown error',
    });
  }
});

// Load snapshot metadata + playlist URL (for deep links /replay/:id after refresh).
app.get('/api/snapshots/:id', async (req, res) => {
  try {
    const rawId = String(req.params.id ?? '').trim();
    const id = rawId.replace(/\.m3u8$/i, '');
    if (!id || !/^[\w.-]+$/.test(id) || id.includes('..')) {
      return res.status(400).json({ error: 'Invalid snapshot id' });
    }
    const snapshotPath = path.join(streamDir, 'snapshots', `${id}.m3u8`);
    const content = await fs.readFile(snapshotPath, 'utf8');
    const { segments } = parseMediaPlaylist(content);
    const durationSec = segments.reduce((sum, s) => sum + s.duration, 0);
    return res.json({
      snapshotId: id,
      url: `/stream/snapshots/${id}.m3u8`,
      durationSec: Number(durationSec.toFixed(3)),
      segmentCount: segments.length,
    });
  } catch (error) {
    if (error && error.code === 'ENOENT') {
      return res.status(404).json({ error: 'Snapshot not found' });
    }
    return res.status(500).json({
      error: 'Failed to load snapshot.',
      details: error instanceof Error ? error.message : 'Unknown error',
    });
  }
});

app.get('/api/health', (_req, res) => {
  res.json({ status: 'ok', service: 'judex-stream-backend' });
});

app.listen(PORT, () => {
  console.log(`Stream backend running at http://localhost:${PORT}`);
  console.log(`Serving HLS from: ${streamDir}`);
  console.log(`Playlist URL: http://localhost:${PORT}/stream/playlist.m3u8`);
});
