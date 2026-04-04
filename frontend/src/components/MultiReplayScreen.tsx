import { useCallback, useEffect, useRef, useState } from 'react';
import { StreamPlayer } from './StreamPlayer';
import type { StreamPlayerHandle } from './StreamPlayer';
import './MultiReplayScreen.css';

const SPEEDS = [0.25, 0.5, 1, 1.25, 1.5] as const;
const SKIP_SECONDS = 5;
const SYNC_THRESHOLD_SEC = 0.15;
const FPS = 30;

const CAMERA_LABELS = { source: 'Source', hq: 'HQ', sink: 'Sink' } as const;
type CameraKey = keyof typeof CAMERA_LABELS;
const CAMERAS: CameraKey[] = ['source', 'hq', 'sink'];

type CameraInfo = {
  url: string | null;
  durationSec: number;
  segmentCount: number;
  error?: string;
};

type SyncOffset = {
  startOffsetSec: number;
  deltaSec: number;
};

type FrameRange = {
  startFrame: number;
  endFrame: number;
};

type SyncMap = {
  source: number[];
  sink: number[];
  hq: number[];
};

type MultiReplayMeta = {
  segmentId: string;
  minutes: number;
  syncMethod?: 'csv' | 'duration';
  syncDurationSec?: number;
  cameras: Record<CameraKey, CameraInfo>;
  syncOffsets?: Partial<Record<CameraKey, SyncOffset>>;
  frameInfo?: Partial<Record<CameraKey, FrameRange>>;
  syncMap?: SyncMap;
};

type MultiReplayScreenProps = {
  segmentId: string;
  minutes: number;
  onGoLive: () => void;
};

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

export function MultiReplayScreen({ segmentId, minutes, onGoLive }: MultiReplayScreenProps) {
  const [meta, setMeta] = useState<MultiReplayMeta | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeCam, setActiveCam] = useState<CameraKey>('source');

  const sourceRef = useRef<StreamPlayerHandle | null>(null);
  const hqRef = useRef<StreamPlayerHandle | null>(null);
  const sinkRef = useRef<StreamPlayerHandle | null>(null);
  const refs = { source: sourceRef, hq: hqRef, sink: sinkRef };

  const sourceFrameRef = useRef<HTMLSpanElement>(null);
  const hqFrameRef = useRef<HTMLSpanElement>(null);
  const sinkFrameRef = useRef<HTMLSpanElement>(null);
  const frameRefs = { source: sourceFrameRef, hq: hqFrameRef, sink: sinkFrameRef };

  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [speedMenuOpen, setSpeedMenuOpen] = useState(false);
  const speedMenuRef = useRef<HTMLDivElement>(null);
  const seekingRef = useRef(false);
  const wrapperRef = useRef<HTMLDivElement>(null);

  // Fetch multi-camera replay metadata
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    (async () => {
      try {
        const res = await fetch(`/api/replay/multi/${segmentId}?minutes=${minutes}`);
        const data = await res.json();
        if (!res.ok) throw new Error(data?.error ?? 'Failed to load');
        if (!cancelled) setMeta(data);
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : 'Unknown error');
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();

    return () => { cancelled = true; };
  }, [segmentId, minutes]);

  // Set a stable duration from the API response instead of reading
  // the video element's duration (which fluctuates as hls.js loads segments).
  useEffect(() => {
    if (!meta) return;
    if (meta.syncDurationSec && meta.syncDurationSec > 0) {
      setDuration(meta.syncDurationSec);
    } else {
      const totalDur = meta.cameras.source?.durationSec ?? 0;
      if (totalDur > 0) setDuration(totalDur);
    }
  }, [meta]);

  const getAllVideos = useCallback((): HTMLVideoElement[] => {
    return CAMERAS
      .map((c) => refs[c].current?.getVideo())
      .filter((v): v is HTMLVideoElement => v != null);
  }, []);

  const sourceStartOffset = meta?.syncOffsets?.source?.startOffsetSec ?? 0;

  // Convert a source playlist time to the sync map index
  const sourceTimeToMapIdx = useCallback(
    (sourcePlaylistTime: number): number => {
      if (!meta?.syncMap?.source?.length) return -1;
      const frameOffset = Math.round((sourcePlaylistTime - sourceStartOffset) * FPS);
      return Math.max(0, Math.min(frameOffset, meta.syncMap.source.length - 1));
    },
    [meta, sourceStartOffset],
  );

  // Convert a target camera's frame index to its playlist time
  const frameToPlaylistTime = useCallback(
    (cam: CameraKey, frame: number): number => {
      const fi = meta?.frameInfo?.[cam];
      const offset = meta?.syncOffsets?.[cam]?.startOffsetSec ?? 0;
      if (!fi || (fi.startFrame === 0 && fi.endFrame === 0)) return -1;
      return offset + (frame - fi.startFrame) / FPS;
    },
    [meta],
  );

  // Keep follower videos in sync with the source (leader).
  // Uses per-frame sync mapping from the CSV: each source frame maps
  // to the exact corresponding sink/hq frame.
  useEffect(() => {
    if (!meta || loading) return;

    const hasSyncMap = meta.syncMap && meta.syncMap.source.length > 0;
    const getDelta = (cam: CameraKey) => meta.syncOffsets?.[cam]?.deltaSec ?? 0;

    let rafId: number;
    const tick = () => {
      const leader = sourceRef.current?.getVideo();
      if (leader && !seekingRef.current && !leader.seeking) {
        const t = leader.currentTime;
        const syncT = Math.max(0, t - sourceStartOffset);
        setCurrentTime(syncT);
        const leaderPlaying = !leader.paused;
        setPlaying(leaderPlaying);

        const mapIdx = hasSyncMap ? sourceTimeToMapIdx(t) : -1;

        for (const cam of ['hq', 'sink'] as const) {
          const follower = refs[cam].current?.getVideo();
          if (!follower || !Number.isFinite(follower.duration)) continue;

          if (follower.seeking) continue;

          if (leaderPlaying && follower.paused && follower.readyState >= 3) {
            follower.play().catch(() => {});
          }

          let target: number;
          if (hasSyncMap && mapIdx >= 0) {
            const targetFrame = meta.syncMap![cam][mapIdx];
            if (targetFrame < 0) { target = t + getDelta(cam); }
            else { target = frameToPlaylistTime(cam, targetFrame); }
          } else {
            target = t + getDelta(cam);
          }

          if (target >= 0 && Math.abs(follower.currentTime - target) > SYNC_THRESHOLD_SEC) {
            follower.currentTime = target;
          }
        }

        // Update frame counters from the exact sync map
        if (hasSyncMap && mapIdx >= 0) {
          for (const cam of CAMERAS) {
            const el = frameRefs[cam].current;
            if (el) {
              const frame = meta.syncMap![cam][mapIdx];
              el.textContent = frame >= 0 ? `F ${frame}` : '';
            }
          }
        } else if (meta.frameInfo) {
          for (const cam of CAMERAS) {
            const fi = meta.frameInfo[cam];
            const el = frameRefs[cam].current;
            if (fi && el) {
              const totalFrames = fi.endFrame - fi.startFrame;
              const syncDur = meta.syncDurationSec ?? (duration > 0 ? duration : 1);
              const frame = fi.startFrame + Math.round((syncT / syncDur) * totalFrames);
              el.textContent = `F ${frame}`;
            }
          }
        }
      }
      rafId = requestAnimationFrame(tick);
    };
    rafId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafId);
  }, [meta, loading, sourceStartOffset, sourceTimeToMapIdx, frameToPlaylistTime]);

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
    const videos = getAllVideos();
    if (!videos.length) return;
    const shouldPlay = videos[0].paused;
    videos.forEach((v) => {
      if (shouldPlay) {
        v.play().catch(() => {});
      } else {
        v.pause();
      }
    });
  }, [getAllVideos]);

  const seekFollowersToSourceTime = useCallback(
    (sourceTime: number) => {
      const mapIdx = sourceTimeToMapIdx(sourceTime);
      for (const cam of ['hq', 'sink'] as const) {
        const v = refs[cam].current?.getVideo();
        if (!v || !Number.isFinite(v.duration)) continue;
        if (meta?.syncMap && mapIdx >= 0) {
          const frame = meta.syncMap[cam][mapIdx];
          if (frame >= 0) {
            v.currentTime = frameToPlaylistTime(cam, frame);
            continue;
          }
        }
        v.currentTime = sourceTime + (meta?.syncOffsets?.[cam]?.deltaSec ?? 0);
      }
    },
    [meta, sourceTimeToMapIdx, frameToPlaylistTime],
  );

  const skip = useCallback(
    (delta: number) => {
      const leader = sourceRef.current?.getVideo();
      if (!leader) return;
      const t = Math.min(
        Math.max(sourceStartOffset, leader.currentTime + delta),
        leader.duration || Infinity,
      );
      leader.currentTime = t;
      seekFollowersToSourceTime(t);
    },
    [sourceStartOffset, seekFollowersToSourceTime],
  );

  const setSpeedTo = useCallback(
    (s: number) => {
      setSpeed(s);
      getAllVideos().forEach((v) => {
        v.playbackRate = s;
      });
      setSpeedMenuOpen(false);
    },
    [getAllVideos],
  );

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
    const syncTime = seekValueRef.current;
    const sourceTime = syncTime + sourceStartOffset;
    const leader = sourceRef.current?.getVideo();
    if (leader) leader.currentTime = sourceTime;
    seekFollowersToSourceTime(sourceTime);
  }, [sourceStartOffset, seekFollowersToSourceTime]);

  const toggleFullscreen = useCallback(() => {
    const el = wrapperRef.current;
    if (!el) return;
    if (document.fullscreenElement) {
      void document.exitFullscreen();
    } else {
      void el.requestFullscreen();
    }
  }, []);

  // Apply speed to all videos when it changes
  useEffect(() => {
    getAllVideos().forEach((v) => { v.playbackRate = speed; });
  }, [speed, getAllVideos]);

  // Initial seek to startOffsetSec for each camera
  useEffect(() => {
    if (!meta || loading) return;

    const applyInitialSeek = () => {
      for (const cam of CAMERAS) {
        const v = refs[cam].current?.getVideo();
        const offset = meta.syncOffsets?.[cam]?.startOffsetSec ?? 0;
        if (v && Number.isFinite(v.duration) && offset > 0) {
          v.currentTime = offset;
        }
      }
    };

    const timerId = setTimeout(applyInitialSeek, 500);
    return () => clearTimeout(timerId);
  }, [meta, loading]);

  const seekFraction = duration > 0 ? currentTime / duration : 0;

  if (loading) {
    return (
      <div className="app">
        <header className="app-header">
          <h1>Judex Multi-Camera Replay</h1>
          <p className="subtitle">Loading segment {segmentId}…</p>
        </header>
        <div className="top-actions">
          <button type="button" onClick={onGoLive}>Back to live</button>
        </div>
        <main className="player-container">
          <div className="empty-state">Loading replay…</div>
        </main>
      </div>
    );
  }

  if (error || !meta) {
    return (
      <div className="app">
        <header className="app-header">
          <h1>Judex Multi-Camera Replay</h1>
        </header>
        <div className="top-actions">
          <button type="button" onClick={onGoLive}>Back to live</button>
        </div>
        <p className="error-text">{error || 'Failed to load replay data'}</p>
      </div>
    );
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>Judex Multi-Camera Replay</h1>
        <p className="subtitle">
          Segment {segmentId} · Last {minutes} min · {meta.syncMethod === 'csv' ? 'CSV-synced' : 'Duration-synced'}
        </p>
      </header>

      <div className="top-actions">
        <button type="button" onClick={onGoLive}>Back to live</button>
        <span className="meta-pill">
          Viewing: {CAMERA_LABELS[activeCam]} · {meta.cameras[activeCam].segmentCount} segments
        </span>
      </div>

      <div className="multi-replay-wrapper" ref={wrapperRef}>
        <div className="multi-replay-layout">
          {/* Main player area */}
          <div className="multi-replay-main">
            {CAMERAS.map((cam) => {
              const info = meta.cameras[cam];
              const ref = refs[cam];
              const isActive = cam === activeCam;
              return (
                <div
                  key={cam}
                  className={`multi-replay-main-slot${isActive ? ' active' : ''}`}
                  onClick={isActive ? togglePlay : undefined}
                >
                  {info.url ? (
                    <StreamPlayer
                      key={info.url}
                      ref={ref}
                      src={info.url}
                      playbackMode="vod"
                      hideNativeControls
                    />
                  ) : (
                    <div className="multi-replay-unavailable">
                      {info.error || 'Unavailable'}
                    </div>
                  )}
                  <span className="multi-replay-frame" ref={frameRefs[cam]} />
                </div>
              );
            })}
            <div className="multi-replay-main-label">{CAMERA_LABELS[activeCam]}</div>
          </div>

          {/* Right sidebar with camera thumbnails */}
          <div className="multi-replay-sidebar">
            {CAMERAS.map((cam) => {
              const info = meta.cameras[cam];
              const isActive = cam === activeCam;
              return (
                <button
                  key={cam}
                  type="button"
                  className={`multi-replay-thumb${isActive ? ' active' : ''}`}
                  onClick={() => setActiveCam(cam)}
                  aria-label={`Switch to ${CAMERA_LABELS[cam]}`}
                  aria-pressed={isActive}
                >
                  <span className="multi-replay-thumb-label">{CAMERA_LABELS[cam]}</span>
                  {!info.url && (
                    <span className="multi-replay-thumb-unavail">N/A</span>
                  )}
                </button>
              );
            })}
          </div>
        </div>

        <div className="multi-replay-controls-overlay">
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
        </div>
      </div>
    </div>
  );
}
