import { useCallback, useEffect, useState } from 'react';
import { StreamPlayer } from './components/StreamPlayer';
import { ReplayScreen } from './components/ReplayScreen';
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

function replayPath(snapshotId: string) {
  return `/replay/${encodeURIComponent(snapshotId)}`;
}

export default function App() {
  const [pathname, setPathname] = useState(getPathname);
  const replayId = parseReplayId(pathname);
  const route = replayId ? 'replay' : 'live';

  const [streamUrl, setStreamUrl] = useState(STREAM_URL);
  const [inputUrl, setInputUrl] = useState(STREAM_URL);
  const [replayUrl, setReplayUrl] = useState<string>('');
  const [snapshotMinutes, setSnapshotMinutes] = useState('5');
  const [creatingSnapshot, setCreatingSnapshot] = useState(false);
  const [replayBootstrapLoading, setReplayBootstrapLoading] = useState(false);
  const [replayError, setReplayError] = useState<string | null>(null);
  const [replayMeta, setReplayMeta] = useState<{
    minutes?: number | null;
    durationSec: number;
    segmentCount?: number;
  } | null>(null);

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

  const loadStream = useCallback(() => {
    setStreamUrl(inputUrl.trim() || STREAM_URL);
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
      {replayError && <p className="error-text">{replayError}</p>}

      <main className="player-container">
        <StreamPlayer src={streamUrl} playbackMode="live" />
      </main>
    </div>
  );
}
