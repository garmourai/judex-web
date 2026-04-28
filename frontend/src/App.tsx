import { useCallback, useEffect, useState } from 'react';
import { StreamPlayer } from './components/StreamPlayer';
import { ReplayScreen } from './components/ReplayScreen';
import { MultiReplayScreen } from './components/MultiReplayScreen';
import { API_BASE, resolveApi } from './api-base';
import './App.css';

const STREAM_URL =
  import.meta.env.VITE_STREAM_URL ?? resolveApi('/stream/playlist.m3u8');

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
    // 3-camera replay window is fixed at 3 minutes.
    const mins = 3;
    const url = `/multi-replay/${segId}?minutes=${mins}`;
    window.history.pushState({}, '', url);
    setPathname(getPathname());
  }, [multiSegmentId, sessionStatus.trackId]);

  if (route === 'multi-replay' && multiReplayId) {
    // 3-camera replay window is fixed at 3 minutes regardless of any
    // stale `?minutes=...` carried over from older URLs/bookmarks.
    const mins = 3;
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
