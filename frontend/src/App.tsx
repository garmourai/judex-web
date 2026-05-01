import { useCallback, useEffect, useState } from 'react';
import { StreamPlayer } from './components/StreamPlayer';
import { ReplayScreen } from './components/ReplayScreen';
import { MultiReplayScreen } from './components/MultiReplayScreen';
import './App.css';

const STREAM_URL =
  import.meta.env.VITE_STREAM_URL ?? '/stream/playlist.m3u8';
const API_BASE = import.meta.env.VITE_API_BASE ?? '';

function getPathname() {
  return window.location.pathname;
}

function parseReplayId(pathname: string): string | null {
  const m = pathname.match(/^\/replay\/([^/]+)\/?$/);
  if (!m) return null;
  try {
    return decodeURIComponent(m[1]);
  } catch {
    return m[1];
  }
}

function parseMultiReplayId(pathname: string): string | null {
  const m = pathname.match(/^\/multi-replay\/([^/]+)\/?$/);
  if (!m) return null;
  return m[1];
}

function replayPath(snapshotId: string) {
  return `/replay/${encodeURIComponent(snapshotId)}`;
}

type SessionStatus = {
  cameraPrepared: boolean;
  preparingCamera: boolean;
  captureActive: boolean;
  startingCapture: boolean;
  stoppingCapture: boolean;
  trackId: string | null;
  startedAt: string | null;
  error: string | null;
};

export default function App() {
  const [pathname, setPathname] = useState(getPathname);
  const replayId = parseReplayId(pathname);
  const multiReplayId = parseMultiReplayId(pathname);
  const route = multiReplayId ? 'multi-replay' : replayId ? 'replay' : 'live';

  const [streamUrl, setStreamUrl] = useState(STREAM_URL);
  const [inputUrl, setInputUrl] = useState(STREAM_URL);
  const [replayUrl, setReplayUrl] = useState<string>('');
  const [snapshotMinutes, setSnapshotMinutes] = useState('5');
  const [multiSegmentId, setMultiSegmentId] = useState('');
  const [creatingSnapshot, setCreatingSnapshot] = useState(false);
  const [replayBootstrapLoading, setReplayBootstrapLoading] = useState(false);
  const [replayError, setReplayError] = useState<string | null>(null);
  const [replayMeta, setReplayMeta] = useState<{
    minutes?: number | null;
    durationSec: number;
    segmentCount?: number;
  } | null>(null);

  const [sessionStatus, setSessionStatus] = useState<SessionStatus>({
    cameraPrepared: false, preparingCamera: false,
    captureActive: false, startingCapture: false, stoppingCapture: false,
    trackId: null, startedAt: null, error: null,
  });
  const [sessionError, setSessionError] = useState<string | null>(null);
  const [streamKey, setStreamKey] = useState(0);

  const [cacheStatus, setCacheStatus] = useState({ source: 0, hq: 0, sink: 0 });

  useEffect(() => {
    const onPopState = () => setPathname(getPathname());
    window.addEventListener('popstate', onPopState);
    return () => window.removeEventListener('popstate', onPopState);
  }, []);

  /** When URL is /replay/:id (including full page load / refresh), fetch snapshot info and playlist URL. */
  useEffect(() => {
    if (!replayId) {
      setReplayUrl('');
      setReplayMeta(null);
      setReplayError(null);
      setReplayBootstrapLoading(false);
      return;
    }

    let cancelled = false;
    setReplayBootstrapLoading(true);
    setReplayError(null);

    (async () => {
      try {
        const res = await fetch(`${API_BASE}/api/snapshots/${encodeURIComponent(replayId)}`);
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
          throw new Error(data?.error ?? 'Failed to load replay');
        }
        if (cancelled) return;
        const url = typeof data.url === 'string' ? data.url : `/stream/snapshots/${replayId}.m3u8`;
        setReplayUrl(url.startsWith('http') ? url : `${API_BASE}${url}`);
        setReplayMeta({
          minutes: data.minutes ?? null,
          durationSec: Number(data.durationSec) || 0,
          segmentCount: data.segmentCount,
        });
      } catch (e) {
        if (!cancelled) {
          setReplayUrl('');
          setReplayMeta(null);
          setReplayError(e instanceof Error ? e.message : 'Unknown error');
        }
      } finally {
        if (!cancelled) setReplayBootstrapLoading(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [replayId, API_BASE]);

  useEffect(() => {
    fetch(`${API_BASE}/api/session/status`)
      .then((r) => r.json())
      .then((data) => setSessionStatus(data))
      .catch(() => {});
  }, []);

  // Background segment pre-fetcher: polls all 3 camera live playlists every 4 s while capture is
  // active, fetching each new .ts segment so the browser caches it. When the user clicks "Replay",
  // the VOD playlists reference the same URLs that are already in the browser cache → instant load.
  // We poll /cam/source/{trackId}/ (not /stream/) so URLs match what the VOD m3u8 will reference.
  useEffect(() => {
    if (!sessionStatus.captureActive || !sessionStatus.trackId) return;
    const trackId = sessionStatus.trackId;
    const MAX_CACHED = 45;
    let cancelled = false;

    const cameras: Record<string, string> = {
      source: `/cam/source/${trackId}/playlist.m3u8`,
      hq: `/cam/hq/${trackId}/playlist.m3u8`,
      sink: `/cam/sink/${trackId}/playlist.m3u8`,
    };

    // Fresh seen-sets for this capture session (captured by closures below)
    const seen: Record<string, Set<string>> = { source: new Set(), hq: new Set(), sink: new Set() };

    async function pollCamera(cam: string, m3u8Url: string) {
      try {
        const res = await fetch(m3u8Url, { cache: 'no-store' });
        if (cancelled || !res.ok) return;
        const text = await res.text();
        if (cancelled) return;
        const base = m3u8Url.slice(0, m3u8Url.lastIndexOf('/') + 1);
        const segUrls: string[] = [];
        for (const line of text.split('\n')) {
          const t = line.trim();
          if (!t || t.startsWith('#')) continue;
          segUrls.push(t.startsWith('http') || t.startsWith('/') ? t : base + t);
        }
        const recent = segUrls.slice(-MAX_CACHED);
        const camSeen = seen[cam];
        for (const url of recent) {
          if (!camSeen.has(url)) {
            camSeen.add(url);
            fetch(url).catch(() => {});
          }
        }
        if (camSeen.size > MAX_CACHED) {
          const arr = Array.from(camSeen);
          arr.slice(0, arr.length - MAX_CACHED).forEach((u) => camSeen.delete(u));
        }
        if (!cancelled) {
          setCacheStatus((prev) => ({ ...prev, [cam]: Math.min(camSeen.size, MAX_CACHED) }));
        }
      } catch { /* polling errors are normal during live capture */ }
    }

    function poll() {
      Object.entries(cameras).forEach(([c, u]) => { void pollCamera(c, u); });
    }

    poll();
    const id = setInterval(poll, 4000);

    return () => {
      cancelled = true;
      clearInterval(id);
      setCacheStatus({ source: 0, hq: 0, sink: 0 });
    };
  }, [sessionStatus.captureActive, sessionStatus.trackId]);

  const prepareCamera = useCallback(async () => {
    setSessionError(null);
    setSessionStatus((s) => ({ ...s, preparingCamera: true, error: null }));
    try {
      const res = await fetch(`${API_BASE}/api/session/prepare-camera`, { method: 'POST' });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.error ?? 'Failed to prepare camera');
      setSessionStatus((s) => ({ ...s, cameraPrepared: true, preparingCamera: false, trackId: data.trackId, error: null }));
      setStreamUrl(data.streamUrl ?? STREAM_URL);
      setInputUrl(data.streamUrl ?? STREAM_URL);
      setStreamKey((k) => k + 1);
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Unknown error';
      setSessionError(msg);
      setSessionStatus((s) => ({ ...s, preparingCamera: false, error: msg }));
    }
  }, []);

  const startCapture = useCallback(async () => {
    setSessionError(null);
    setSessionStatus((s) => ({ ...s, startingCapture: true, error: null }));
    try {
      const res = await fetch(`${API_BASE}/api/session/start-capture`, { method: 'POST' });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.error ?? 'Failed to start capture');
      setSessionStatus((s) => ({ ...s, captureActive: true, startingCapture: false, startedAt: data.startedAt, error: null }));
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Unknown error';
      setSessionError(msg);
      setSessionStatus((s) => ({ ...s, startingCapture: false, error: msg }));
    }
  }, []);

  const stopCapture = useCallback(async () => {
    setSessionError(null);
    setSessionStatus((s) => ({ ...s, stoppingCapture: true, error: null }));
    try {
      const res = await fetch(`${API_BASE}/api/session/stop-capture`, { method: 'POST' });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.error ?? 'Failed to stop capture');
      setSessionStatus({
        cameraPrepared: false, preparingCamera: false,
        captureActive: false, startingCapture: false, stoppingCapture: false,
        trackId: null, startedAt: null, error: null,
      });
      setStreamKey((k) => k + 1);
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Unknown error';
      setSessionError(msg);
      setSessionStatus((s) => ({ ...s, stoppingCapture: false, error: msg }));
    }
  }, []);

  const loadStream = useCallback(() => {
    setStreamUrl(inputUrl.trim() || STREAM_URL);
    setStreamKey((k) => k + 1);
  }, [inputUrl]);

  const startReplay = useCallback(async () => {
    setCreatingSnapshot(true);
    setReplayError(null);
    try {
      const parsed = Number(snapshotMinutes);
      const minutes = Number.isFinite(parsed) ? parsed : 5;
      const res = await fetch(`${API_BASE}/api/snapshots`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ minutes }),
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data?.error ?? 'Failed to create snapshot');
      }
      const id = data.snapshotId as string;
      if (!id) {
        throw new Error('Server did not return snapshotId');
      }
      const url = typeof data.url === 'string' ? data.url : `/stream/snapshots/${id}.m3u8`;
      setReplayUrl(url.startsWith('http') ? url : `${API_BASE}${url}`);
      setReplayMeta({
        minutes: data.minutes,
        durationSec: Number(data.durationSec) || 0,
        segmentCount: data.segmentCount,
      });
      window.history.pushState({}, '', replayPath(id));
      setPathname(getPathname());
    } catch (error) {
      setReplayError(error instanceof Error ? error.message : 'Unknown error');
    } finally {
      setCreatingSnapshot(false);
    }
  }, [snapshotMinutes, API_BASE]);

  const goLive = useCallback(() => {
    window.history.pushState({}, '', '/');
    setPathname(getPathname());
  }, []);

  const startMultiReplay = useCallback(() => {
    const segId = multiSegmentId.trim() || sessionStatus.trackId;
    if (!segId) return;
    const parsed = Number(snapshotMinutes);
    const mins = Number.isFinite(parsed) && parsed > 0 ? parsed : 3;
    const url = `/multi-replay/${segId}?minutes=${mins}`;
    window.history.pushState({}, '', url);
    setPathname(getPathname());
  }, [multiSegmentId, sessionStatus.trackId, snapshotMinutes]);

  if (route === 'multi-replay' && multiReplayId) {
    const params = new URLSearchParams(window.location.search);
    const mins = Number(params.get('minutes')) || 5;
    return (
      <MultiReplayScreen
        segmentId={multiReplayId}
        minutes={mins}
        onGoLive={goLive}
      />
    );
  }

  if (route === 'replay') {
    return (
      <ReplayScreen
        replayUrl={replayUrl}
        loading={replayBootstrapLoading}
        error={replayError}
        meta={replayMeta}
        onGoLive={goLive}
      />
    );
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>Judex Stream</h1>
        <p className="subtitle">HLS · .ts transport stream playback</p>
      </header>

      <div className="session-controls">
        {sessionStatus.trackId && (
          <div className="session-active">
            <span className="session-indicator" />
            <span>Track #{sessionStatus.trackId}{sessionStatus.captureActive ? ' · Capturing' : ' · Camera ready'}</span>
          </div>
        )}

        <div className="session-buttons">
          <button
            type="button"
            className="session-btn session-btn--prepare"
            onClick={prepareCamera}
            disabled={sessionStatus.preparingCamera || sessionStatus.cameraPrepared}
          >
            {sessionStatus.preparingCamera ? 'Preparing…' : sessionStatus.cameraPrepared ? 'Camera Ready' : 'Prepare Camera'}
          </button>

          <button
            type="button"
            className="session-btn session-btn--start"
            onClick={startCapture}
            disabled={!sessionStatus.cameraPrepared || sessionStatus.startingCapture || sessionStatus.captureActive}
          >
            {sessionStatus.startingCapture ? 'Starting…' : sessionStatus.captureActive ? 'Capture Running' : 'Start Capture'}
          </button>

          <button
            type="button"
            className="session-btn session-btn--stop"
            onClick={stopCapture}
            disabled={(!sessionStatus.captureActive && !sessionStatus.cameraPrepared) || sessionStatus.stoppingCapture}
          >
            {sessionStatus.stoppingCapture ? 'Stopping…' : 'Stop Capture'}
          </button>
        </div>

        {sessionStatus.captureActive && sessionStatus.trackId && (
          <div className="live-replay-bar">
            <span className="cache-status-text">
              Buffered — Source: {cacheStatus.source}/45 · HQ: {cacheStatus.hq}/45 · Sink: {cacheStatus.sink}/45
            </span>
            <button
              type="button"
              className="session-btn session-btn--replay"
              onClick={() => {
                const url = `/multi-replay/${sessionStatus.trackId}?minutes=3`;
                window.history.pushState({}, '', url);
                setPathname(getPathname());
              }}
            >
              Replay Last 3 Min
            </button>
          </div>
        )}

        {(sessionError || sessionStatus.error) && (
          <p className="error-text">{sessionError || sessionStatus.error}</p>
        )}
      </div>

      <div className="stream-url-bar">
        <input
          type="text"
          value={inputUrl}
          onChange={(e) => setInputUrl(e.target.value)}
          placeholder="Stream URL (e.g. /stream/playlist.m3u8)"
          aria-label="Stream URL"
        />
        <button type="button" onClick={loadStream}>
          Load
        </button>
      </div>

      <div className="snapshot-controls">
        <label htmlFor="snapshotMinutes">Replay last</label>
        <input
          id="snapshotMinutes"
          type="number"
          min={1}
          max={120}
          value={snapshotMinutes}
          onChange={(e) => setSnapshotMinutes(e.target.value)}
        />
        <span>min</span>
        <button type="button" onClick={startReplay} disabled={creatingSnapshot}>
          {creatingSnapshot ? 'Creating snapshot…' : 'Open replay page'}
        </button>
      </div>

      <div className="snapshot-controls">
        <label htmlFor="multiSegmentId">Segment ID</label>
        <input
          id="multiSegmentId"
          type="text"
          value={multiSegmentId}
          onChange={(e) => setMultiSegmentId(e.target.value)}
          placeholder={sessionStatus.trackId || 'e.g. 1563'}
          style={{ width: '6rem' }}
        />
        <button
          type="button"
          onClick={startMultiReplay}
          disabled={!multiSegmentId.trim() && !sessionStatus.trackId}
        >
          3-Camera Replay
        </button>
      </div>
      {replayError && <p className="error-text">{replayError}</p>}

      <main className="player-container">
        <StreamPlayer key={streamKey} src={streamUrl} playbackMode="live" />
      </main>
    </div>
  );
}
