import express from 'express';
import http2Express from 'http2-express-bridge';
import cors from 'cors';
import path from 'path';
import { fileURLToPath } from 'url';
import { promises as fs } from 'fs';
import { exec, spawn } from 'child_process';
import { promisify } from 'util';
import http2 from 'node:http2';
import os from 'node:os';
import selfsigned from 'selfsigned';

const execAsync = promisify(exec);

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// http2-express-bridge wraps Express so its req/res work over both HTTP/1.1
// and HTTP/2 streams. Without it Node's native http2 compat layer trips on
// `IncomingMessage._read` because `req.socket._readableState` isn't always
// populated for HTTP/2 streams when middleware (body-parser etc.) tries to
// read the body. The wrapper installs the missing shims.
const app = http2Express(express);
const PORT = process.env.PORT ?? 3014;
const DEFAULT_SNAPSHOT_MINUTES = Number(process.env.SNAPSHOT_DEFAULT_MINUTES ?? 5);

const defaultStreamDir = path.join(__dirname, '..', 'stream-output');
const streamDir = process.env.STREAM_DIR
  ? path.resolve(process.env.STREAM_DIR)
  : defaultStreamDir;

const SOURCE_CODE_DIR = '/home/pi/source_code';
const TS_SEGMENTS_DIR = path.join(SOURCE_CODE_DIR, 'ts_segments');
const TRACK_INDEX_PATH = path.join(SOURCE_CODE_DIR, 'variable_files', 'track_video_index.json');

const SYNC_REPORTS_DIR = path.resolve(__dirname, '..', '..', 'sync_reports');
const WATCHER_SCRIPT = path.resolve(__dirname, '..', '..', 'hls_segment_watcher.py');
const EVENTS_ROOT = path.resolve(__dirname, '..', '..', 'events');

// Jetson SSH config
const JETSON_HOST = 'jetson@192.168.0.148';
const JETSON_SSH_OPTS = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-o', 'StrictHostKeyChecking=accept-new'];
const JETSON_EVENTS_CSV = '/home/jetson/Desktop/cv_output/correlation/bounce_events.csv';
const JETSON_CLIPS_CSV = '/home/jetson/Desktop/cv_output/correlation/bounce_events_clips.csv';
const JETSON_CLIPS_DIR = '/home/jetson/Desktop/cv_output/bounce_clips';

async function sshCatText(remotePath) {
  const { stdout } = await execAsync(
    `ssh ${JETSON_SSH_OPTS.join(' ')} ${JETSON_HOST} cat ${remotePath}`,
    { maxBuffer: 32 * 1024 * 1024 },
  );
  return stdout;
}

// SSE: live bounce events streaming
const sseClients = new Set();
let lastSeenEventRowCount = 0;
let ssePolling = false;

function parseBounceEventRows(content) {
  const lines = content.split(/\r?\n/);
  if (lines.length < 2) return { header: [], events: [] };
  const header = lines[0].split(',');
  const bounceFrameIdx = header.indexOf('bounce_frame');
  const dirIdx = header.indexOf('crossing_direction');
  const sideIdx = header.indexOf('side');
  const scoreIdx = header.indexOf('score');
  const events = [];
  for (let i = 1; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;
    const cols = line.split(',');
    events.push({
      frame: Number(cols[bounceFrameIdx]) || 0,
      direction: cols[dirIdx] || '',
      side: cols[sideIdx] || '',
      score: Number(cols[scoreIdx]) || 0,
    });
  }
  return events;
}

function sseWrite(res, eventName, data) {
  res.write(`event: ${eventName}\ndata: ${JSON.stringify(data)}\n\n`);
}

async function runSsePoller() {
  if (ssePolling) return;
  ssePolling = true;
  while (sseClients.size > 0) {
    await new Promise((r) => setTimeout(r, 4000));
    if (sseClients.size === 0) break;
    try {
      const content = await sshCatText(JETSON_EVENTS_CSV);
      const events = parseBounceEventRows(content);
      if (events.length > lastSeenEventRowCount) {
        const newEvents = events.slice(lastSeenEventRowCount);
        lastSeenEventRowCount = events.length;
        for (const client of sseClients) {
          sseWrite(client, 'new-events', newEvents);
        }
      }
    } catch (e) {
      console.error('[jetson-sse] poll error:', e.message);
    }
  }
  ssePolling = false;
}

const CAMERA_DIRS = {
  source: TS_SEGMENTS_DIR,
  hq: path.join(SYNC_REPORTS_DIR, 'ts_segments_hq'),
  sink: path.join(SYNC_REPORTS_DIR, 'ts_segments_sink'),
};

const FPS_ESTIMATE = 30;

let activeStreamDir = streamDir;

const cleanSession = {
  cameraPrepared: false,
  preparingCamera: false,
  captureActive: false,
  startingCapture: false,
  stoppingCapture: false,
  trackId: null,
  startedAt: null,
  error: null,
};

let session = { ...cleanSession };

let watcherProcess = null;

function startWatcher(trackId) {
  stopWatcher();
  console.log(`[watcher] Starting hls_segment_watcher.py for track ${trackId}`);
  const proc = spawn('python3', [WATCHER_SCRIPT, '--track-id', String(trackId), '--reset-state'], {
    cwd: path.dirname(WATCHER_SCRIPT),
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  proc.stdout.on('data', (d) => process.stdout.write(`[watcher] ${d}`));
  proc.stderr.on('data', (d) => process.stderr.write(`[watcher-err] ${d}`));
  proc.on('exit', (code) => {
    console.log(`[watcher] exited with code ${code}`);
    if (watcherProcess === proc) watcherProcess = null;
  });
  watcherProcess = proc;
}

function stopWatcher() {
  if (watcherProcess) {
    console.log('[watcher] Stopping watcher process');
    watcherProcess.kill('SIGTERM');
    watcherProcess = null;
  }
}

app.use(cors());
app.use(express.json());

// Express adds weak ETags to JSON by default; clients revalidate with If-None-Match → 304 with an
// empty body, which breaks fetch().json() and shows stale data. API routes must not be cached here.
app.use('/api', (_req, res, next) => {
  res.set('Cache-Control', 'no-store, no-cache, must-revalidate');
  res.set('Pragma', 'no-cache');
  res.set('etag', false);
  next();
});

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

function buildVodPlaylist(segments, startSequence) {
  const targetDuration = Math.max(1, Math.ceil(Math.max(...segments.map((s) => s.duration))));
  const body = [
    '#EXTM3U',
    '#EXT-X-VERSION:3',
    '#EXT-X-PLAYLIST-TYPE:VOD',
    `#EXT-X-TARGETDURATION:${targetDuration}`,
    `#EXT-X-MEDIA-SEQUENCE:${startSequence}`,
    '',
  ];
  for (const seg of segments) {
    body.push(`#EXTINF:${seg.duration.toFixed(3)},`);
    body.push(seg.uri);
    body.push('');
  }
  body.push('#EXT-X-ENDLIST');
  body.push('');
  return body.join('\n');
}

const DEFAULT_SEGMENT_DURATION = 4.0;
const FIRST_SEGMENT_DURATION = 6.0;

async function discoverSegments(segmentDir) {
  const files = await fs.readdir(segmentDir);
  const tsFiles = files
    .filter((f) => /^seg_\d+\.ts$/.test(f))
    .sort();
  if (!tsFiles.length) return [];
  return tsFiles.map((f, i) => {
    const seq = parseInt(f.replace('seg_', '').replace('.ts', ''), 10);
    return {
      uri: f,
      duration: i === 0 ? FIRST_SEGMENT_DURATION : DEFAULT_SEGMENT_DURATION,
      programDateTime: null,
      sequence: seq,
    };
  });
}

function findSegmentAtPlaylistTime(segments, targetTime) {
  let elapsed = 0;
  for (const seg of segments) {
    if (elapsed + seg.duration > targetTime) {
      return { segNum: seg.sequence, offsetSec: targetTime - elapsed };
    }
    elapsed += seg.duration;
  }
  const last = segments[segments.length - 1];
  return { segNum: last.sequence, offsetSec: 0 };
}

async function computeSyncInfo(segmentId, minutes) {
  const syncCsvPath = path.join(SYNC_REPORTS_DIR, `segments_${segmentId}`, 'sync', `hls_sync_${segmentId}_triple.csv`);
  let csvContent;
  try {
    csvContent = await fs.readFile(syncCsvPath, 'utf8');
  } catch {
    return null;
  }

  const lines = csvContent.split(/\r?\n/);
  if (lines.length < 2) return null;

  // Discover all segments on disk for each camera
  const camPlaylists = {};
  for (const [camKey, baseDir] of Object.entries(CAMERA_DIRS)) {
    const segDir = path.join(baseDir, segmentId);
    try {
      const segments = await discoverSegments(segDir);
      if (!segments.length) continue;
      const totalDur = segments.reduce((s, seg) => s + seg.duration, 0);
      camPlaylists[camKey] = { segments, totalDur };
    } catch {
      // directory not available for this camera
    }
  }

  if (!camPlaylists.source) return null;

  // Derive each camera's recording start wall time from the sync CSV
  let firstSourceIdx = null, firstSinkIdx = null, firstHqIdx = null;
  let firstSourceWall = null;
  for (let i = 1; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;
    const cols = line.split(',');
    if (firstSourceIdx === null) {
      firstSourceIdx = Number(cols[0]) || 0;
      firstSinkIdx = Number(cols[1]) || 0;
      firstHqIdx = cols[2] ? Number(cols[2]) : null;
      firstSourceWall = BigInt(cols[5]);
      break;
    }
  }

  if (firstSourceWall === null) return null;

  const camRecordingStartNs = {};
  camRecordingStartNs.source = firstSourceWall - BigInt(Math.round((firstSourceIdx / FPS_ESTIMATE) * 1e9));
  if (firstSinkIdx !== null) {
    camRecordingStartNs.sink = firstSourceWall - BigInt(Math.round((firstSinkIdx / FPS_ESTIMATE) * 1e9));
  }
  if (firstHqIdx !== null) {
    camRecordingStartNs.hq = firstSourceWall - BigInt(Math.round((firstHqIdx / FPS_ESTIMATE) * 1e9));
  }

  // Attach wall-clock start time and playlist end time to each camera
  for (const [camKey, cam] of Object.entries(camPlaylists)) {
    const startNs = camRecordingStartNs[camKey];
    if (!startNs) continue;
    cam.startNs = startNs;
    cam.endNs = startNs + BigInt(Math.round(cam.totalDur * 1e9));
  }

  // Determine the common end wall-clock time (earliest end across all cameras)
  let commonEndWallNs = null;
  for (const cam of Object.values(camPlaylists)) {
    if (!cam.endNs) continue;
    if (commonEndWallNs === null || cam.endNs < commonEndWallNs) {
      commonEndWallNs = cam.endNs;
    }
  }
  if (!commonEndWallNs) return null;

  const targetNs = BigInt(minutes) * 60n * 1000000000n;
  const actualCutoffNs = commonEndWallNs - targetNs;

  // Find last source/sink/hq frame indices
  let lastSourceIdx = 0, lastSinkIdx = 0, lastHqIdx = 0;
  for (let i = lines.length - 1; i >= 1; i--) {
    const line = lines[i].trim();
    if (!line) continue;
    const cols = line.split(',');
    lastSourceIdx = Number(cols[0]) || 0;
    lastSinkIdx = Number(cols[1]) || 0;
    lastHqIdx = cols[2] ? Number(cols[2]) : 0;
    break;
  }

  // Compute segment ranges for each camera
  const camInfo = {};
  const syncOffsets = {};

  for (const [camKey, cam] of Object.entries(camPlaylists)) {
    if (!cam.startNs) {
      // Duration-based fallback
      let total = 0;
      const selected = [];
      const targetSec = minutes * 60;
      for (let i = cam.segments.length - 1; i >= 0; i--) {
        selected.push(cam.segments[i]);
        total += cam.segments[i].duration;
        if (total >= targetSec) break;
      }
      selected.reverse();
      camInfo[camKey] = {
        startSegNum: selected[0].sequence,
        endSegNum: selected[selected.length - 1].sequence,
        startOffsetSec: 0,
        segmentCount: selected.length,
        durationSec: Number(total.toFixed(3)),
      };
      continue;
    }

    const cutoffPlaylistTime = Number(actualCutoffNs - cam.startNs) / 1e9;
    const endPlaylistTime = Number(commonEndWallNs - cam.startNs) / 1e9;

    const clampedStart = Math.max(0, cutoffPlaylistTime);
    const clampedEnd = Math.min(endPlaylistTime, cam.totalDur);

    const startInfo = findSegmentAtPlaylistTime(cam.segments, clampedStart);
    const endInfo = findSegmentAtPlaylistTime(cam.segments, clampedEnd);

    const selected = cam.segments.filter(
      (s) => s.sequence >= startInfo.segNum && s.sequence <= endInfo.segNum,
    );
    const durationSec = selected.reduce((s, seg) => s + seg.duration, 0);

    camInfo[camKey] = {
      startSegNum: startInfo.segNum,
      endSegNum: endInfo.segNum,
      startOffsetSec: Number(startInfo.offsetSec.toFixed(4)),
      segmentCount: selected.length,
      durationSec: Number(durationSec.toFixed(3)),
    };
  }

  // Compute sync offsets relative to source
  const sourceOffset = camInfo.source?.startOffsetSec ?? 0;
  for (const camKey of Object.keys(camInfo)) {
    const offset = camInfo[camKey].startOffsetSec ?? 0;
    syncOffsets[camKey] = {
      startOffsetSec: offset,
      deltaSec: Number((offset - sourceOffset).toFixed(4)),
    };
  }

  // Build per-frame sync mapping for the replay window
  const syncMap = { source: [], sink: [], hq: [] };
  let cutoffSourceIdx = 0, cutoffSinkIdx = 0, cutoffHqIdx = 0;
  let foundCutoff = false;

  for (let i = 1; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;
    const cols = line.split(',');
    const wallNs = BigInt(cols[5]);
    if (wallNs < actualCutoffNs) continue;
    if (wallNs > commonEndWallNs) break;

    const srcIdx = Number(cols[0]) || 0;
    const sinkIdx = Number(cols[1]) || 0;
    const hqIdx = cols[2] ? Number(cols[2]) : -1;

    if (!foundCutoff) {
      cutoffSourceIdx = srcIdx;
      cutoffSinkIdx = sinkIdx;
      cutoffHqIdx = hqIdx >= 0 ? hqIdx : 0;
      foundCutoff = true;
    }

    syncMap.source.push(srcIdx);
    syncMap.sink.push(sinkIdx);
    syncMap.hq.push(hqIdx);
  }

  const frameInfo = {
    source: { startFrame: cutoffSourceIdx, endFrame: lastSourceIdx },
    hq: { startFrame: cutoffHqIdx, endFrame: lastHqIdx },
    sink: { startFrame: cutoffSinkIdx, endFrame: lastSinkIdx },
  };

  const syncDurationSec = Number((commonEndWallNs - actualCutoffNs)) / 1e9;
  if (syncDurationSec < 1) return null;

  return { camInfo, syncOffsets, syncDurationSec: Math.round(syncDurationSec), frameInfo, syncMap };
}

// --- Camera segment routes for multi-replay ---

// Serve TS segments from camera directories
for (const [camKey, baseDir] of Object.entries(CAMERA_DIRS)) {
  app.use(`/cam/${camKey}`, (req, res, next) => {
    const segmentId = req.path.split('/')[1];
    if (!segmentId || !/^\d+$/.test(segmentId)) return next();
    const dir = path.join(baseDir, segmentId);
    // express.static(root) joins root + req.url. Here req.url is still /1565/seg_….ts,
    // so without rewriting we would resolve …/1565/1565/seg_….ts (404) and hls.js never loads .ts.
    const queryIndex = req.url.indexOf('?');
    const query = queryIndex >= 0 ? req.url.slice(queryIndex) : '';
    const pathAfterSeg = req.path.slice(`/${segmentId}`.length) || '/';
    const prevUrl = req.url;
    req.url = pathAfterSeg + query;
    express.static(dir, {
      setHeaders: (res, filePath) => {
        if (filePath.endsWith('.ts')) {
          res.set('Content-Type', 'video/MP2T');
          res.set('Cache-Control', 'public, max-age=3600, immutable');
        }
        if (filePath.endsWith('.m3u8')) {
          res.set('Content-Type', 'application/vnd.apple.mpegurl');
          res.set('Cache-Control', 'no-cache');
        }
      },
    })(req, res, () => {
      req.url = prevUrl;
      next();
    });
  });
}

// VOD sub-playlist for a camera's replay window
app.get('/cam/:camera/:segmentId/replay.m3u8', async (req, res) => {
  try {
    const { camera, segmentId } = req.params;
    const baseDir = CAMERA_DIRS[camera];
    if (!baseDir) return res.status(404).json({ error: `Unknown camera: ${camera}` });

    const segDir = path.join(baseDir, segmentId);
    const segments = await discoverSegments(segDir);
    if (!segments.length) return res.status(409).json({ error: 'No segments found on disk' });

    const startSeg = Number(req.query.startSeg);
    const endSeg = Number(req.query.endSeg);

    let selected;
    if (Number.isFinite(startSeg) && Number.isFinite(endSeg)) {
      selected = segments.filter((s) => s.sequence >= startSeg && s.sequence <= endSeg);
    } else {
      const minutesParam = Number(req.query.minutes) || 5;
      const targetSec = minutesParam * 60;
      selected = [];
      let total = 0;
      for (let i = segments.length - 1; i >= 0; i--) {
        selected.push(segments[i]);
        total += segments[i].duration;
        if (total >= targetSec) break;
      }
      selected.reverse();
    }

    if (!selected.length) return res.status(409).json({ error: 'No matching segments' });
    const playlist = buildVodPlaylist(selected, selected[0].sequence);
    res.set('Content-Type', 'application/vnd.apple.mpegurl');
    res.set('Cache-Control', 'no-cache');
    return res.send(playlist);
  } catch (err) {
    if (err?.code === 'ENOENT') return res.status(404).json({ error: 'Segment directory not found' });
    return res.status(500).json({ error: err.message });
  }
});

// SSE stream: sends all current events on connect, then pushes new rows as they appear on Jetson
app.get('/api/events/live/stream', async (req, res) => {
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.flushHeaders();

  // Send current events as the initial payload
  try {
    const content = await sshCatText(JETSON_EVENTS_CSV);
    const events = parseBounceEventRows(content);
    // Sync lastSeenEventRowCount to at least what we just read (first client wins)
    if (events.length > lastSeenEventRowCount) lastSeenEventRowCount = events.length;
    sseWrite(res, 'init', events);
  } catch (e) {
    console.error('[jetson-sse] init fetch error:', e.message);
    sseWrite(res, 'init', []);
  }

  sseClients.add(res);
  runSsePoller();

  req.on('close', () => {
    sseClients.delete(res);
  });
});

// Bounce events for a segment — reads live from Jetson
app.get('/api/events/:segmentId', async (req, res) => {
  try {
    const content = await sshCatText(JETSON_EVENTS_CSV);
    return res.json(parseBounceEventRows(content));
  } catch (err) {
    console.error('[jetson] /api/events error:', err.message);
    return res.json([]);
  }
});

// Bounce event clip files (camera + clip_name) for a given bounce_frame — reads live from Jetson
app.get('/api/event-clips/:bounceFrame', async (req, res) => {
  try {
    const bounceFrame = Number(req.params.bounceFrame);
    if (!Number.isFinite(bounceFrame)) {
      return res.status(400).json({ error: 'Invalid bounce_frame' });
    }

    const content = await sshCatText(JETSON_CLIPS_CSV);
    const lines = content.split(/\r?\n/).filter((l) => l.trim());
    if (lines.length < 2) return res.json({ bounceFrame, clips: [] });

    const header = lines[0].split(',').map((h) => h.trim());
    const camIdx = header.indexOf('camera');
    const frameIdx = header.indexOf('bounce_frame');
    const nameIdx = header.indexOf('clip_name');
    const statusIdx = header.indexOf('status');
    if (camIdx < 0 || frameIdx < 0 || nameIdx < 0) {
      return res.status(500).json({ error: 'bounce_events_clips.csv missing required columns' });
    }

    const clips = [];
    for (let i = 1; i < lines.length; i++) {
      const cols = lines[i].split(',');
      if (cols.length <= Math.max(camIdx, frameIdx, nameIdx)) continue;
      const rowFrame = Number(cols[frameIdx]);
      if (rowFrame !== bounceFrame) continue;
      if (statusIdx >= 0 && (cols[statusIdx] || '').trim().toLowerCase() !== 'ok') continue;
      const camera = (cols[camIdx] || '').trim();
      const clipName = (cols[nameIdx] || '').trim();
      if (!camera || !clipName) continue;
      const url = `/events/bounce_clips/${encodeURIComponent(camera)}/${encodeURIComponent(clipName)}`;
      clips.push({ camera, clipName, url });
    }

    return res.json({ bounceFrame, clips });
  } catch (err) {
    console.error('[jetson] /api/event-clips error:', err.message);
    return res.json({ bounceFrame: Number(req.params.bounceFrame), clips: [] });
  }
});

// Multi-camera replay metadata
app.get('/api/replay/multi/:segmentId', async (req, res) => {
  try {
    const { segmentId } = req.params;
    const minutes = Number(req.query.minutes) || 3;

    const syncInfo = await computeSyncInfo(segmentId, minutes);

    const cameras = {};
    for (const camKey of ['source', 'hq', 'sink']) {
      const ci = syncInfo?.camInfo?.[camKey];
      if (ci && ci.segmentCount > 0) {
        const url = `/cam/${camKey}/${segmentId}/replay.m3u8?startSeg=${ci.startSegNum}&endSeg=${ci.endSegNum}`;
        cameras[camKey] = {
          url,
          durationSec: ci.durationSec,
          segmentCount: ci.segmentCount,
        };
      } else {
        // Try duration-based fallback
        const baseDir = CAMERA_DIRS[camKey];
        try {
          const playlistPath = path.join(baseDir, segmentId, 'playlist.m3u8');
          await fs.access(playlistPath);
          cameras[camKey] = {
            url: `/cam/${camKey}/${segmentId}/replay.m3u8?minutes=${minutes}`,
            durationSec: 0,
            segmentCount: 0,
          };
        } catch {
          cameras[camKey] = { url: null, durationSec: 0, segmentCount: 0, error: 'Unavailable' };
        }
      }
    }

    if (syncInfo) {
      return res.json({
        segmentId,
        minutes,
        syncMethod: 'csv',
        syncDurationSec: syncInfo.syncDurationSec,
        cameras,
        syncOffsets: syncInfo.syncOffsets,
        frameInfo: syncInfo.frameInfo,
        syncMap: syncInfo.syncMap,
      });
    }

    return res.json({
      segmentId,
      minutes,
      syncMethod: 'duration',
      cameras,
    });
  } catch (err) {
    return res.status(500).json({ error: err.message });
  }
});

// Snapshot folder serves playlist files only.
app.use('/stream/snapshots', (req, res, next) => {
  if (!req.path.endsWith('.m3u8')) {
    return res.status(403).json({ error: 'Only .m3u8 files are allowed in snapshots.' });
  }
  return next();
});

// Serve HLS from the active stream directory
app.use('/stream', (req, res, next) => {
  express.static(activeStreamDir, {
    setHeaders: (res, filePath) => {
      if (filePath.endsWith('.m3u8')) {
        res.set('Content-Type', 'application/vnd.apple.mpegurl');
        res.set('Cache-Control', 'no-cache, no-store, must-revalidate');
        res.set('Pragma', 'no-cache');
      }
      if (filePath.endsWith('.ts')) {
        res.set('Content-Type', 'video/MP2T');
        res.set('Cache-Control', 'public, max-age=3600, immutable');
      }
    },
  })(req, res, next);
});

app.get('/api/stream-url', (_req, res) => {
  const baseUrl = process.env.PUBLIC_URL ?? `http://localhost:${PORT}`;
  res.json({ url: `${baseUrl}/stream/playlist.m3u8` });
});

app.post('/api/snapshots', async (req, res) => {
  try {
    const requested = Number(req.body?.minutes);
    const minutes = Number.isFinite(requested) && requested > 0
      ? requested
      : DEFAULT_SNAPSHOT_MINUTES;
    const clampedMinutes = Math.min(Math.max(minutes, 1), 120);
    const targetSeconds = clampedMinutes * 60;

    const livePlaylistPath = path.join(activeStreamDir, 'playlist.m3u8');
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
    const snapshotsDir = path.join(activeStreamDir, 'snapshots');
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

app.get('/api/snapshots/:id', async (req, res) => {
  try {
    const rawId = String(req.params.id ?? '').trim();
    const id = rawId.replace(/\.m3u8$/i, '');
    if (!id || !/^[\w.-]+$/.test(id) || id.includes('..')) {
      return res.status(400).json({ error: 'Invalid snapshot id' });
    }
    const snapshotPath = path.join(activeStreamDir, 'snapshots', `${id}.m3u8`);
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

async function readTrackIndex() {
  const content = await fs.readFile(TRACK_INDEX_PATH, 'utf8');
  return JSON.parse(content);
}

async function runScript(scriptPath) {
  return execAsync(`bash "${scriptPath}"`, {
    cwd: SOURCE_CODE_DIR,
    timeout: 180_000,
    env: { ...process.env, HOME: '/home/pi' },
  });
}

app.post('/api/session/prepare-camera', async (_req, res) => {
  if (session.preparingCamera) {
    return res.status(409).json({ error: 'Camera is already being prepared.' });
  }
  if (session.cameraPrepared) {
    return res.status(409).json({ error: 'Camera is already prepared.' });
  }

  session = { ...session, preparingCamera: true, error: null };

  try {
    await runScript(path.join(SOURCE_CODE_DIR, 'prepare_cameras.sh'));
    await new Promise((r) => setTimeout(r, 2000));

    const trackIndex = await readTrackIndex();
    const trackId = String(trackIndex.counter);
    const trackStreamDir = path.join(TS_SEGMENTS_DIR, trackId);

    activeStreamDir = trackStreamDir;

    session = {
      ...session,
      cameraPrepared: true,
      preparingCamera: false,
      trackId,
      error: null,
    };

    return res.json({ trackId, streamUrl: '/stream/playlist.m3u8' });
  } catch (error) {
    const msg = error instanceof Error ? error.message : 'Unknown error';
    session = { ...session, preparingCamera: false, error: msg };
    return res.status(500).json({ error: 'Failed to prepare camera.', details: msg });
  }
});

app.post('/api/session/start-capture', async (_req, res) => {
  if (!session.cameraPrepared) {
    return res.status(400).json({ error: 'Camera must be prepared first.' });
  }
  if (session.startingCapture) {
    return res.status(409).json({ error: 'Capture is already starting.' });
  }
  if (session.captureActive) {
    return res.status(409).json({ error: 'Capture is already active.' });
  }

  session = { ...session, startingCapture: true, error: null };

  try {
    await runScript(path.join(SOURCE_CODE_DIR, 'start_capture.sh'));

    session = {
      ...session,
      captureActive: true,
      startingCapture: false,
      startedAt: new Date().toISOString(),
      error: null,
    };

    if (session.trackId) {
      try { startWatcher(session.trackId); } catch (e) { console.error('[watcher] Failed to start:', e); }
    }

    return res.json({
      trackId: session.trackId,
      streamUrl: '/stream/playlist.m3u8',
      startedAt: session.startedAt,
    });
  } catch (error) {
    const msg = error instanceof Error ? error.message : 'Unknown error';
    session = { ...session, startingCapture: false, error: msg };
    return res.status(500).json({ error: 'Failed to start capture.', details: msg });
  }
});

app.post('/api/session/stop-capture', async (_req, res) => {
  if (session.stoppingCapture) {
    return res.status(409).json({ error: 'Capture is already stopping.' });
  }

  session = { ...session, stoppingCapture: true, error: null };

  try { stopWatcher(); } catch { /* best-effort */ }

  try {
    await runScript(path.join(SOURCE_CODE_DIR, 'stop_capture.sh'));
  } catch (error) {
    console.warn('[stop-capture] stop_capture.sh failed (resetting session anyway):', error.message);
  }

  activeStreamDir = streamDir;
  session = { ...cleanSession };

  return res.json({ stopped: true });
});

app.post('/api/session/reset', (_req, res) => {
  try { stopWatcher(); } catch { /* best-effort */ }
  activeStreamDir = streamDir;
  session = { ...cleanSession };
  return res.json({ reset: true });
});

app.get('/api/session/status', (_req, res) => {
  res.json(session);
});

app.get('/api/health', (_req, res) => {
  res.json({ status: 'ok', service: 'judex-stream-backend' });
});

function isSafeBounceClipCamera(s) {
  return typeof s === 'string' && /^[a-z0-9_-]+$/i.test(s);
}
function isSafeBounceClipFilename(s) {
  return typeof s === 'string' && /^[a-z0-9._-]+\.mp4$/i.test(s);
}

// Stream bounce clip MP4 from Jetson via ssh cat — BounceClipVideo fetches full body as blob so no Range needed.
app.get('/events/bounce_clips/:camera/:filename', (req, res) => {
  const { camera, filename } = req.params;
  if (!isSafeBounceClipCamera(camera) || !isSafeBounceClipFilename(filename)) {
    return res.status(400).send('Invalid path');
  }
  // Try camera subdirectory first; Jetson may have flat or nested layout
  const remotePath = `${JETSON_CLIPS_DIR}/${camera}/${filename}`;
  res.setHeader('Content-Type', 'video/mp4');
  res.setHeader('Cache-Control', 'no-store');

  const ssh = spawn('ssh', [...JETSON_SSH_OPTS, JETSON_HOST, 'cat', remotePath]);
  let headersSent = false;

  ssh.stdout.once('data', () => {
    headersSent = true;
    res.status(200);
  });

  ssh.stdout.pipe(res);

  ssh.stderr.on('data', (d) => {
    console.error('[jetson-clip] ssh stderr:', d.toString().trim());
  });

  ssh.on('exit', (code) => {
    if (code !== 0 && !headersSent) {
      res.status(404).send('Clip not found on Jetson');
    }
  });

  req.on('close', () => ssh.kill());
});

// Other files under events/ (if any)
app.use('/events', express.static(EVENTS_ROOT));

// Serve frontend build
const frontendDist = path.join(__dirname, '..', '..', 'frontend', 'dist');
app.use(express.static(frontendDist));
app.get('*', (req, res) => {
  // Do not serve SPA shell for media paths (breaks <video> / HLS if a file is missing).
  if (/\.(mp4|m3u8|ts|webm|vtt)$/i.test(req.path)) {
    return res.status(404).type('text/plain').send('Not found');
  }
  return res.sendFile(path.join(frontendDist, 'index.html'));
});

// HTTPS + HTTP/2 listener.
// Browsers only negotiate HTTP/2 over TLS, so we need a cert. We auto-generate
// a self-signed one on first start and reuse it across restarts. `allowHTTP1:
// true` keeps any HTTP/1.1 clients (curl, scripts, the Pi's own tools) working;
// browsers will pick HTTP/2 via ALPN.
const TLS_DIR = path.join(__dirname, '..', '.tls');
const TLS_KEY_PATH = path.join(TLS_DIR, 'key.pem');
const TLS_CERT_PATH = path.join(TLS_DIR, 'cert.pem');

function collectAltNames() {
  // Subject Alternative Names: every hostname/IP a browser might use to reach
  // this backend must be listed, otherwise the TLS handshake fails with
  // ERR_CERT_COMMON_NAME_INVALID even after the user accepts the warning.
  const altNames = [
    { type: 2, value: 'localhost' },
    { type: 2, value: os.hostname() },
    { type: 7, value: '127.0.0.1' },
    { type: 7, value: '::1' },
  ];
  for (const list of Object.values(os.networkInterfaces())) {
    for (const iface of list ?? []) {
      if (!iface.address) continue;
      altNames.push({ type: 7, value: iface.address });
    }
  }
  // De-duplicate (same address can appear multiple times across interfaces).
  const seen = new Set();
  return altNames.filter((a) => {
    const k = `${a.type}:${a.value}`;
    if (seen.has(k)) return false;
    seen.add(k);
    return true;
  });
}

async function ensureTlsCert() {
  try {
    const [key, cert] = await Promise.all([
      fs.readFile(TLS_KEY_PATH),
      fs.readFile(TLS_CERT_PATH),
    ]);
    return { key, cert };
  } catch {
    await fs.mkdir(TLS_DIR, { recursive: true });
    const altNames = collectAltNames();
    console.log(`[tls] Generating self-signed cert for: ${altNames.map((a) => a.value).join(', ')}`);
    const pems = await selfsigned.generate(
      [{ name: 'commonName', value: 'judex-web-local' }],
      {
        keySize: 2048,
        days: 3650,
        algorithm: 'sha256',
        extensions: [{ name: 'subjectAltName', altNames }],
      },
    );
    await fs.writeFile(TLS_KEY_PATH, pems.private);
    await fs.writeFile(TLS_CERT_PATH, pems.cert);
    console.log(`[tls] Cert written to ${TLS_DIR}`);
    return { key: Buffer.from(pems.private), cert: Buffer.from(pems.cert) };
  }
}

const { key: tlsKey, cert: tlsCert } = await ensureTlsCert();
const server = http2.createSecureServer(
  { key: tlsKey, cert: tlsCert, allowHTTP1: true },
  app,
);
server.listen(PORT, () => {
  console.log(`Stream backend running at https://localhost:${PORT} (HTTP/2 + HTTP/1.1)`);
  console.log(`Serving HLS from: ${streamDir}`);
  console.log(`Playlist URL: https://localhost:${PORT}/stream/playlist.m3u8`);
});
