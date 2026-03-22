import { useCallback, useEffect, useRef, useState } from 'react';
import { StreamPlayer } from './StreamPlayer';
import type { StreamPlayerHandle } from './StreamPlayer';
import './ReplayScreen.css';

const SPEEDS = [0.25, 0.5, 1, 1.25, 1.5] as const;
const SKIP_SECONDS = 5;

type ReplayMeta = {
  minutes?: number | null;
  durationSec: number;
  segmentCount?: number;
};

type ReplayScreenProps = {
  replayUrl: string;
  loading: boolean;
  error: string | null;
  meta: ReplayMeta | null;
  onGoLive: () => void;
};

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

export function ReplayScreen({ replayUrl, loading, error, meta, onGoLive }: ReplayScreenProps) {
  const playerRef = useRef<StreamPlayerHandle>(null);
  const wrapperRef = useRef<HTMLDivElement>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [speedMenuOpen, setSpeedMenuOpen] = useState(false);
  const speedMenuRef = useRef<HTMLDivElement>(null);
  const seekingRef = useRef(false);

  useEffect(() => {
    const video = playerRef.current?.getVideo();
    if (!video) return;

    const onTime = () => {
      if (!seekingRef.current) setCurrentTime(video.currentTime);
    };
    const onDuration = () => {
      if (Number.isFinite(video.duration)) setDuration(video.duration);
    };
    const onPlay = () => setPlaying(true);
    const onPause = () => setPlaying(false);

    video.addEventListener('timeupdate', onTime);
    video.addEventListener('durationchange', onDuration);
    video.addEventListener('play', onPlay);
    video.addEventListener('pause', onPause);

    if (Number.isFinite(video.duration)) setDuration(video.duration);
    setPlaying(!video.paused);

    return () => {
      video.removeEventListener('timeupdate', onTime);
      video.removeEventListener('durationchange', onDuration);
      video.removeEventListener('play', onPlay);
      video.removeEventListener('pause', onPause);
    };
  }, [replayUrl]);

  // Close speed menu on outside click
  useEffect(() => {
    if (!speedMenuOpen) return;
    const onClick = (e: MouseEvent) => {
      if (speedMenuRef.current && !speedMenuRef.current.contains(e.target as Node)) {
        setSpeedMenuOpen(false);
      }
    };
    document.addEventListener('mousedown', onClick);
    return () => document.removeEventListener('mousedown', onClick);
  }, [speedMenuOpen]);

  const togglePlay = useCallback(() => {
    const video = playerRef.current?.getVideo();
    if (!video) return;
    if (video.paused) {
      void video.play();
    } else {
      video.pause();
    }
  }, []);

  const skip = useCallback((delta: number) => {
    const video = playerRef.current?.getVideo();
    if (!video) return;
    video.currentTime = Math.min(Math.max(0, video.currentTime + delta), video.duration || Infinity);
  }, []);

  const setSpeedTo = useCallback((s: number) => {
    setSpeed(s);
    const video = playerRef.current?.getVideo();
    if (video) video.playbackRate = s;
    setSpeedMenuOpen(false);
  }, []);

  const seekValueRef = useRef(0);

  const onSeekStart = useCallback(() => {
    seekingRef.current = true;
  }, []);

  const onSeekInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const val = Number(e.target.value);
    seekValueRef.current = val;
    setCurrentTime(val);
  }, []);

  const onSeekCommit = useCallback(() => {
    seekingRef.current = false;
    const video = playerRef.current?.getVideo();
    if (!video) return;
    video.currentTime = seekValueRef.current;
  }, []);

  const toggleFullscreen = useCallback(() => {
    const el = wrapperRef.current;
    if (!el) return;
    if (document.fullscreenElement) {
      void document.exitFullscreen();
    } else {
      void el.requestFullscreen();
    }
  }, []);

  useEffect(() => {
    const video = playerRef.current?.getVideo();
    if (video) video.playbackRate = speed;
  }, [replayUrl, speed]);

  const subtitle = loading
    ? 'Loading replay…'
    : meta?.minutes != null
      ? `Last ${meta.minutes} min snapshot (non-live)`
      : meta
        ? 'Snapshot replay (non-live)'
        : '';

  const seekFraction = duration > 0 ? currentTime / duration : 0;

  return (
    <div className="app">
      <header className="app-header">
        <h1>Judex Replay</h1>
        <p className="subtitle">{subtitle}</p>
      </header>

      <div className="top-actions">
        <button type="button" onClick={onGoLive}>
          Back to live
        </button>
        {meta && !loading && (
          <span className="meta-pill">
            Snapshot length: {Math.round(meta.durationSec)}s
          </span>
        )}
      </div>

      <main className="player-container">
        {loading && <div className="empty-state">Loading replay…</div>}
        {!loading && error && <p className="error-text">{error}</p>}
        {!loading && !error && !replayUrl && (
          <div className="empty-state">No replay snapshot loaded.</div>
        )}
        {!loading && !error && replayUrl && (
          <div className="replay-wrapper" ref={wrapperRef}>
            <StreamPlayer
              ref={playerRef}
              src={replayUrl}
              playbackMode="vod"
              hideNativeControls
            />

            <div className="replay-fs-overlay">
            {/* Seek bar */}
            <div className="replay-seek-bar">
              <input
                type="range"
                className="replay-seeker"
                min={0}
                max={duration || 0}
                step={0.1}
                value={currentTime}
                onMouseDown={onSeekStart}
                onTouchStart={onSeekStart}
                onChange={onSeekInput}
                onMouseUp={onSeekCommit}
                onTouchEnd={onSeekCommit}
                style={{ '--seek-fraction': seekFraction } as React.CSSProperties}
                aria-label="Seek"
              />
            </div>

            {/* Controls row */}
            <div className="replay-controls">
              <div className="replay-controls-left">
                <button
                  type="button"
                  className="replay-ctrl-btn"
                  onClick={togglePlay}
                  aria-label={playing ? 'Pause' : 'Play'}
                >
                  {playing ? '⏸' : '▶'}
                </button>
                <button
                  type="button"
                  className="replay-ctrl-btn"
                  onClick={() => skip(-SKIP_SECONDS)}
                  aria-label={`Rewind ${SKIP_SECONDS}s`}
                >
                  ⏪ {SKIP_SECONDS}s
                </button>
                <button
                  type="button"
                  className="replay-ctrl-btn"
                  onClick={() => skip(SKIP_SECONDS)}
                  aria-label={`Forward ${SKIP_SECONDS}s`}
                >
                  {SKIP_SECONDS}s ⏩
                </button>

                {/* Speed picker */}
                <div className="replay-speed-wrapper" ref={speedMenuRef}>
                  <button
                    type="button"
                    className="replay-ctrl-btn replay-speed-btn"
                    onClick={() => setSpeedMenuOpen((o) => !o)}
                    aria-label="Change speed"
                    aria-expanded={speedMenuOpen}
                  >
                    {speed}x
                  </button>
                  {speedMenuOpen && (
                    <div className="replay-speed-menu" role="menu">
                      {SPEEDS.map((s) => (
                        <button
                          key={s}
                          type="button"
                          role="menuitem"
                          className={`replay-speed-option${s === speed ? ' active' : ''}`}
                          onClick={() => setSpeedTo(s)}
                        >
                          {s}x
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              </div>

              <div className="replay-controls-right">
                <span className="replay-time">
                  {formatTime(currentTime)} / {formatTime(duration)}
                </span>
                <button
                  type="button"
                  className="replay-ctrl-btn replay-fullscreen-btn"
                  onClick={toggleFullscreen}
                  aria-label="Fullscreen"
                >
                  ⛶
                </button>
              </div>
            </div>
            </div>{/* end replay-fs-overlay */}
          </div>
        )}
      </main>
    </div>
  );
}
