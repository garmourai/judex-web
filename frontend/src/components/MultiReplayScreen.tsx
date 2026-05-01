import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { StreamPlayer } from './StreamPlayer';
import type { StreamPlayerHandle } from './StreamPlayer';
import { BounceClipVideo } from './BounceClipVideo';
import './MultiReplayScreen.css';

const SPEEDS = [0.25, 0.5, 1, 1.25, 1.5] as const;
const SKIP_SECONDS = 5;
const SYNC_THRESHOLD_SEC = 0.15;
const FPS = 30;

// On open, jump to the last N seconds of the synced window so the most
// recent action plays immediately.
const TAIL_OFFSET_SEC = 10;
// Tail "priority" batch — last N segments per camera. Segments are 4s
// each (first is 6s) per backend/src/index.js, so 10 ≈ 40s of tail —
// well past the 10s seek target, giving the user ~30s of cache-warm
// room to immediately scrub backward into without waiting on Phase B.
const TAIL_PRIORITY_BATCH = 10;
// Global concurrency cap for the prefetch worker pool. Keeps the
// browser's HTTP/1.1 6-connection-per-origin pool from being saturated
// (1 SSE + 3 prefetch + ~2 hls.js = ~6) so hls.js can still load the
// segments it actively needs without queueing minutes behind backfill.
const PREFETCH_CONCURRENCY = 3;

const CAMERA_LABELS = { source: 'Source', hq: 'HQ', sink: 'Sink' } as const;
type CameraKey = keyof typeof CAMERA_LABELS;
const CAMERAS: CameraKey[] = ['source', 'hq', 'sink'];

// Persists across React StrictMode tear-down/remount and Vite HMR so we
// don't fire the same prefetch twice. The entry is removed in `finally`
// once the prefetch completes, so navigating away and re-opening the
// same window after completion triggers a fresh prefetch.
const inFlightPrefetches = new Map<string, AbortController>();

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

type BounceEvent = {
  frame: number;
  direction: string;
  side: string;
  score: number;
};

type EventMarker = {
  fraction: number;
  timeSec: number;
  frame: number;
  direction: string;
  side: string;
  score: number;
  label: string;
};

type EventClipRow = {
  camera: string;
  clipName: string;
  url: string;
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

  const [events, setEvents] = useState<BounceEvent[]>([]);
  const [selectedEventClips, setSelectedEventClips] = useState<{
    marker: EventMarker;
    clips: EventClipRow[];
    loading: boolean;
    error: string | null;
  } | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [speedMenuOpen, setSpeedMenuOpen] = useState(false);
  const speedMenuRef = useRef<HTMLDivElement>(null);
  const seekingRef = useRef(false);
  const seekCooldownRef = useRef(0);
  const wrapperRef = useRef<HTMLDivElement>(null);
  // Tracks whether the initial seek-to-tail has been applied to the leader
  // video element. Used to gate the rAF tracking loop so it doesn't snap
  // the seeker UI back to 0:00 in the brief window between meta-loaded
  // and the actual seek landing.
  const didInitialTailSeekRef = useRef(false);
  // Set true once Phase A of the prefetch (last 5 segments per camera)
  // has finished. Tail-seek waits for this so the segment containing the
  // seek target is already in the browser HTTP cache when hls.js asks
  // for it — otherwise hls.js can stall mid-seek and the playhead can
  // silently revert toward t=0 (the "snap to 0:03" bug).
  const phaseAReadyRef = useRef(false);
  // Buffering overlay — true whenever the leader is mid-seek or its
  // readyState is below HAVE_FUTURE_DATA *after* the initial first-load
  // has completed (we don't want to overlap with StreamPlayer's own
  // first-load spinner).
  const [isBuffering, setIsBuffering] = useState(false);
  const isBufferingRef = useRef(false);
  const hasInitiallyLoadedRef = useRef(false);
  // Optional sync-time at which the prefetch should refocus. `null` =
  // tail (initial behavior). A number means "the user just scrubbed
  // here, drop the in-flight backward walk and start over from this
  // point". Updated by `onSeekCommit` together with `prefetchGen`.
  const prefetchFocusRef = useRef<number | null>(null);
  const [prefetchGen, setPrefetchGen] = useState(0);

  // Fetch multi-camera replay metadata.
  //
  // The URL param `segmentId` is either a persisted replayId (prefixed
  // `r_…`) — in which case we read the saved meta.json — or a raw
  // numeric segmentId (legacy / typed-in URLs) which we resolve via the
  // dynamic /api/replay/multi endpoint as before.
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    const isPersistedReplay = /^r_/.test(segmentId);

    (async () => {
      try {
        let res = isPersistedReplay
          ? await fetch(`/api/replays/${encodeURIComponent(segmentId)}`)
          : await fetch(`/api/replay/multi/${segmentId}?minutes=${minutes}`);

        if (!res.ok && isPersistedReplay && res.status === 404) {
          // Replay folder was deleted but URL is still bookmarked —
          // fall back to a fresh dynamic resolution so the user at
          // least sees something.
          res = await fetch(`/api/replay/multi/${segmentId}?minutes=${minutes}`);
        }

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

  // Live bounce events via SSE — receives all current events on 'init', new rows on 'new-events'
  useEffect(() => {
    const es = new EventSource('/api/events/live/stream');

    es.addEventListener('init', (e) => {
      try {
        const data = JSON.parse(e.data);
        if (Array.isArray(data)) setEvents(data);
      } catch {}
    });

    es.addEventListener('new-events', (e) => {
      try {
        const incoming = JSON.parse(e.data);
        if (Array.isArray(incoming) && incoming.length > 0) {
          setEvents((prev) => [...prev, ...incoming]);
        }
      } catch {}
    });

    es.onerror = () => {
      // SSE will auto-reconnect; no user-visible error needed
    };

    return () => es.close();
  }, []);

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

  // Map bounce_frame (global source frame index, same space as sync CSV Source_Index) onto the
  // current replay window [startFrame..endFrame]. Prefer syncMap lookup so markers match playback.
  const eventMarkers = useMemo(() => {
    if (!events.length || !meta?.frameInfo?.source || !meta.syncDurationSec) return [];
    const { startFrame, endFrame } = meta.frameInfo.source;
    const syncDur = meta.syncDurationSec;
    const sm = meta.syncMap?.source;

    const frameToSyncTime = (bounceFrame: number): number | null => {
      if (sm?.length) {
        let bestIdx = 0;
        let bestDiff = Infinity;
        for (let i = 0; i < sm.length; i++) {
          const d = Math.abs(sm[i] - bounceFrame);
          if (d < bestDiff) {
            bestDiff = d;
            bestIdx = i;
          }
        }
        if (bestDiff > 90) return null;
        const matched = sm[bestIdx];
        if (matched < startFrame - 60 || matched > endFrame + 60) return null;
        const t = (matched - startFrame) / FPS;
        if (t < 0 || t > syncDur) return null;
        return t;
      }
      const t = (bounceFrame - startFrame) / FPS;
      if (t < 0 || t > syncDur) return null;
      return t;
    };

    return events
      .map((ev) => {
        const timeSec = frameToSyncTime(ev.frame);
        if (timeSec == null) return null;
        const marker: EventMarker = {
          fraction: timeSec / syncDur,
          timeSec,
          frame: ev.frame,
          direction: ev.direction,
          side: ev.side,
          score: ev.score,
          label: ev.direction === 'left_to_right' ? 'L→R' : 'R→L',
        };
        return marker;
      })
      .filter((m): m is EventMarker => m !== null);
  }, [events, meta]);

  const eventsTimeline = useMemo(
    (): EventMarker[] => [...eventMarkers].sort((a, b) => a.timeSec - b.timeSec),
    [eventMarkers],
  );

  const { canPrevEvent, canNextEvent } = useMemo(() => {
    if (!eventsTimeline.length) return { canPrevEvent: false, canNextEvent: false };
    const t = currentTime;
    const prevOk = eventsTimeline.some((ev) => ev.timeSec < t - 0.02);
    const nextOk = eventsTimeline.some((ev) => ev.timeSec > t + 0.02);
    return { canPrevEvent: prevOk, canNextEvent: nextOk };
  }, [eventsTimeline, currentTime]);

  const getAllVideos = useCallback((): HTMLVideoElement[] => {
    return CAMERAS
      .map((c) => refs[c].current?.getVideo())
      .filter((v): v is HTMLVideoElement => v != null);
  }, []);

  const sourceStartOffset = meta?.syncOffsets?.source?.startOffsetSec ?? 0;

  // Per-camera playlist time corresponding to the tail-seek target. We
  // pass these into hls.js as `startPosition` so each player begins
  // loading the *tail* fragment (instead of seg_0). This avoids the
  // race where autoplay-from-zero loads seg_0 in parallel with our
  // programmatic seek to the tail — that race was the cause of the
  // "snap back to 0:08" symptom in the logs.
  const camInitialPositions = useMemo<Partial<Record<CameraKey, number>>>(() => {
    if (!meta) return {};
    const syncDur = meta.syncDurationSec ?? 0;
    if (syncDur <= 0) return {};
    const target = Math.max(0, syncDur - TAIL_OFFSET_SEC);
    const sourceTime = target + sourceStartOffset;
    const result: Partial<Record<CameraKey, number>> = { source: sourceTime };

    const hasSyncMap = !!meta.syncMap && meta.syncMap.source.length > 0;
    const mapIdx = hasSyncMap
      ? Math.max(
          0,
          Math.min(
            Math.round((sourceTime - sourceStartOffset) * FPS),
            (meta.syncMap?.source.length ?? 1) - 1,
          ),
        )
      : -1;

    for (const cam of ['hq', 'sink'] as const) {
      const startOffset = meta.syncOffsets?.[cam]?.startOffsetSec ?? 0;
      const delta = meta.syncOffsets?.[cam]?.deltaSec ?? 0;
      let pos = sourceTime + delta;
      if (hasSyncMap && mapIdx >= 0) {
        const frame = meta.syncMap![cam][mapIdx];
        const fi = meta.frameInfo?.[cam];
        const hasFrameInfo = !!fi && !(fi.startFrame === 0 && fi.endFrame === 0);
        if (frame >= 0 && hasFrameInfo) {
          // Mirror frameToPlaylistTime: convert the camera's absolute
          // frame index into its playlist time, anchored at the camera's
          // own startOffsetSec.
          pos = startOffset + (frame - fi!.startFrame) / FPS;
        }
      }
      result[cam] = Math.max(0, pos);
    }
    return result;
  }, [meta, sourceStartOffset]);

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
      const now = Date.now();
      const inCooldown = now < seekCooldownRef.current;

      if (leader) {
        // Once the video has reached HAVE_FUTURE_DATA at least once, we
        // know the initial first-load is done and any subsequent drop
        // into seeking / low-readyState is genuine buffering caused by
        // a user seek or a SourceBuffer gap.
        if (leader.readyState >= 3) hasInitiallyLoadedRef.current = true;
        const buf =
          hasInitiallyLoadedRef.current && (leader.seeking || leader.readyState < 3);
        if (buf !== isBufferingRef.current) {
          isBufferingRef.current = buf;
          setIsBuffering(buf);
        }
      }

      if (leader && !seekingRef.current && !leader.seeking) {
        const t = leader.currentTime;
        const syncT = Math.max(0, t - sourceStartOffset);
        // Don't snap the seeker UI back to 0 before the initial tail seek
        // has applied. The pre-position effect below sets currentTime to
        // syncDur - 10 the moment meta loads; this gate keeps that visible
        // until either the seek lands (didInitialTailSeekRef = true) or
        // the leader has any meaningful playhead (syncT > 0.1).
        if (didInitialTailSeekRef.current || syncT > 0.1) {
          setCurrentTime(syncT);
        }
        const leaderPlaying = !leader.paused;
        setPlaying(leaderPlaying);

        const mapIdx = hasSyncMap ? sourceTimeToMapIdx(t) : -1;

        if (!inCooldown) {
          for (const cam of ['hq', 'sink'] as const) {
            const follower = refs[cam].current?.getVideo();
            if (!follower || !Number.isFinite(follower.duration)) continue;

            if (follower.seeking || follower.readyState < 2) continue;

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
      seekCooldownRef.current = Date.now() + 2000;
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

  const seekToSyncTime = useCallback(
    (syncT: number) => {
      const maxSync = meta?.syncDurationSec ?? duration;
      const clamped = maxSync > 0 ? Math.max(0, Math.min(syncT, maxSync)) : Math.max(0, syncT);
      const sourceTime = clamped + sourceStartOffset;
      const leader = sourceRef.current?.getVideo();
      if (leader) leader.currentTime = sourceTime;
      seekFollowersToSourceTime(sourceTime);
      setCurrentTime(clamped);
      seekValueRef.current = clamped;
    },
    [sourceStartOffset, seekFollowersToSourceTime, duration, meta?.syncDurationSec],
  );

  const openEventClips = useCallback(
    async (ev: EventMarker) => {
      seekToSyncTime(ev.timeSec);
      setSelectedEventClips({ marker: ev, clips: [], loading: true, error: null });
      try {
        const r = await fetch(`/api/event-clips/${ev.frame}`);
        const data = (await r.json()) as { clips?: EventClipRow[]; error?: string };
        if (!r.ok) throw new Error(data?.error ?? 'Failed to load clip list');
        const clips = Array.isArray(data?.clips) ? data.clips : [];
        setSelectedEventClips({ marker: ev, clips, loading: false, error: null });
      } catch (e) {
        setSelectedEventClips({
          marker: ev,
          clips: [],
          loading: false,
          error: e instanceof Error ? e.message : 'Failed to load clip list',
        });
      }
    },
    [seekToSyncTime],
  );

  const jumpToPrevEvent = useCallback(() => {
    const leader = sourceRef.current?.getVideo();
    if (!leader || !eventsTimeline.length) return;
    const t = Math.max(0, leader.currentTime - sourceStartOffset);
    for (let i = eventsTimeline.length - 1; i >= 0; i--) {
      const ev = eventsTimeline[i];
      if (ev.timeSec < t - 0.02) {
        void openEventClips(ev);
        return;
      }
    }
  }, [eventsTimeline, openEventClips, sourceStartOffset]);

  const jumpToNextEvent = useCallback(() => {
    const leader = sourceRef.current?.getVideo();
    if (!leader || !eventsTimeline.length) return;
    const t = Math.max(0, leader.currentTime - sourceStartOffset);
    for (const ev of eventsTimeline) {
      if (ev.timeSec > t + 0.02) {
        void openEventClips(ev);
        return;
      }
    }
  }, [eventsTimeline, openEventClips, sourceStartOffset]);

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
    // Refocus the prefetch around the user's new playback position:
    // cancel the in-flight backward walk and start a fresh run that
    // forward-buffers from here, then walks backward from here to t=0.
    prefetchFocusRef.current = syncTime;
    setPrefetchGen((g) => g + 1);
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

  // Reset the "did initial tail seek" + "phase-A ready" guards whenever
  // the user opens a new segment / changes the replay length. Without
  // this the guards would remain true after the previous open and we'd
  // never seek-to-tail again on the next open.
  useEffect(() => {
    didInitialTailSeekRef.current = false;
    phaseAReadyRef.current = false;
  }, [segmentId, minutes]);

  // Pre-position the seeker UI to syncDur - TAIL_OFFSET_SEC the moment
  // meta arrives, so the user never sees a 0:00 → tail flash. The actual
  // video seek lands a beat later (Effect below, gated on duration).
  useEffect(() => {
    if (!meta || loading) return;
    if (didInitialTailSeekRef.current) return;
    const syncDur = meta.syncDurationSec ?? duration;
    if (!syncDur || syncDur <= 0) return;
    setCurrentTime(Math.max(0, syncDur - TAIL_OFFSET_SEC));
  }, [meta, loading, duration]);

  // Tail seek with poll-for-duration + verify-and-retry. The leader's
  // duration isn't known until hls.js has finished parsing the playlist,
  // which can take a few hundred ms; polling avoids racing it. The
  // verify-after-1.2s catches the rare case where the seek was issued
  // before hls.js was ready to honor it and silently snapped back.
  useEffect(() => {
    if (!meta || loading || didInitialTailSeekRef.current) return;
    const syncDur = meta.syncDurationSec ?? duration;
    if (!syncDur || syncDur <= 0) return;
    const target = Math.max(0, syncDur - TAIL_OFFSET_SEC);
    const targetSourceTime = target + sourceStartOffset;

    let cancelled = false;
    let attempts = 0;
    let pollId: ReturnType<typeof setTimeout> | undefined;
    const verifyIds: ReturnType<typeof setTimeout>[] = [];

    const trySeek = () => {
      if (cancelled || didInitialTailSeekRef.current) return;
      const leader = sourceRef.current?.getVideo();
      const ready =
        phaseAReadyRef.current &&
        leader &&
        Number.isFinite(leader.duration) &&
        leader.duration > 0;
      if (ready) {
        console.log(`[MultiReplay tail-seek] applying seek to ${target.toFixed(2)}s (source=${targetSourceTime.toFixed(2)}s)`);
        seekToSyncTime(target);
        didInitialTailSeekRef.current = true;
        // Verify a few times: if hls.js stalls or some other code path
        // resets currentTime back toward 0, re-issue the seek. Once the
        // tail segment is in cache (Phase A is done before we get here),
        // a single retry is usually enough — but stagger 3 checks to be
        // robust against a slow first decode.
        for (const ms of [800, 1600, 3000]) {
          verifyIds.push(
            setTimeout(() => {
              if (cancelled) return;
              const l = sourceRef.current?.getVideo();
              if (l && l.currentTime < targetSourceTime - 5) {
                console.warn(`[MultiReplay tail-seek] playhead drifted to ${l.currentTime.toFixed(2)}s, re-seeking`);
                seekToSyncTime(target);
              }
            }, ms),
          );
        }
        return;
      }
      // ~30 s upper bound (300 × 100 ms) — long enough to cover Phase A
      // even on slow Pi network. After that we give up and fall back to
      // playing from t=0.
      if (attempts++ < 300) {
        pollId = setTimeout(trySeek, 100);
      } else {
        console.warn('[MultiReplay tail-seek] timed out waiting for Phase A / leader.duration');
      }
    };

    pollId = setTimeout(trySeek, 100);
    return () => {
      cancelled = true;
      if (pollId) clearTimeout(pollId);
      for (const id of verifyIds) clearTimeout(id);
    };
  }, [meta, loading, duration, seekToSyncTime, sourceStartOffset, segmentId, minutes]);

  // Smart prefetch:
  //   - Initial run (focus=null): Phase A buffers TAIL_PRIORITY_BATCH
  //     tail segments per camera at high priority, Phase B walks
  //     strictly backward to segment 0 with default priority.
  //   - On user scrub (focus=syncTime): cancel any in-flight prefetch,
  //     buffer ~TAIL_PRIORITY_BATCH segments forward of the seek point
  //     at high priority, then walk backward from there to segment 0.
  //
  // Re-runs whenever `prefetchGen` changes (incremented in onSeekCommit)
  // and reads the focus point from `prefetchFocusRef` so the effect
  // doesn't have to depend on a fast-changing scrub state.
  useEffect(() => {
    if (!meta || loading) return;
    const key = `${segmentId}:${minutes}`;

    // Abort any in-flight prefetch for this replay before starting a new
    // one. This is the mechanism that "stops the backward ts file
    // loading" when the user manually seeks.
    inFlightPrefetches.get(key)?.abort();

    const ac = new AbortController();
    inFlightPrefetches.set(key, ac);

    const focusTime = prefetchFocusRef.current;
    const isInitial = focusTime === null;
    const syncDur = meta.syncDurationSec ?? duration;

    type PriorityInit = RequestInit & { priority?: 'high' | 'low' | 'auto' };
    const fetchUrl = (u: string, priority?: 'high') =>
      fetch(u, { signal: ac.signal, ...(priority ? { priority } : {}) } as PriorityInit)
        .catch(() => {});

    const fetchPlaylistTsUrls = async (playlistUrl: string): Promise<string[]> => {
      try {
        const res = await fetch(playlistUrl, { signal: ac.signal });
        if (!res.ok) return [];
        const text = await res.text();
        const base = new URL(playlistUrl, window.location.origin);
        return text
          .split(/\r?\n/)
          .map((l) => l.trim())
          .filter((l) => l && !l.startsWith('#'))
          .map((l) => new URL(l, base).toString());
      } catch {
        return [];
      }
    };

    const runPool = async <T,>(
      items: T[],
      worker: (it: T) => Promise<unknown>,
      concurrency: number,
    ) => {
      let i = 0;
      const runners = Array.from(
        { length: Math.min(concurrency, items.length) },
        async () => {
          while (i < items.length) {
            const idx = i++;
            if (ac.signal.aborted) return;
            await worker(items[idx]);
          }
        },
      );
      await Promise.all(runners);
    };

    const interleave = <T,>(arrays: T[][]): T[] => {
      const out: T[] = [];
      const max = Math.max(0, ...arrays.map((a) => a.length));
      for (let i = 0; i < max; i++) {
        for (const a of arrays) {
          if (i < a.length) out.push(a[i]);
        }
      }
      return out;
    };

    // Approximate sync-time → segment-index in a per-camera playlist.
    // All three cameras share the same synced-window duration so a
    // proportional mapping is good enough for prefetch ordering.
    const timeToIdx = (t: number, count: number) => {
      if (!syncDur || syncDur <= 0 || count <= 0) return 0;
      return Math.max(0, Math.min(count - 1, Math.floor((t / syncDur) * count)));
    };

    void (async () => {
      try {
        const perCam = await Promise.all(
          CAMERAS.map(async (cam) => {
            const u = meta.cameras[cam]?.url;
            if (!u) return [] as string[];
            return fetchPlaylistTsUrls(u);
          }),
        );
        if (ac.signal.aborted) return;

        // Phase A
        let phaseAByCam: string[][];
        let phaseBByCam: string[][];
        if (isInitial) {
          // Tail behavior.
          phaseAByCam = perCam.map((urls) =>
            urls.slice(Math.max(0, urls.length - TAIL_PRIORITY_BATCH)),
          );
          phaseBByCam = perCam.map((urls) =>
            urls.slice(0, Math.max(0, urls.length - TAIL_PRIORITY_BATCH)).reverse(),
          );
        } else {
          // Focus behavior: forward `TAIL_PRIORITY_BATCH` from focus,
          // then backward walk from focus−1 to 0. Anything *after* the
          // forward window is left alone — it was probably already
          // cache-warm from the initial Phase A/B run.
          phaseAByCam = perCam.map((urls) => {
            const idx = timeToIdx(focusTime, urls.length);
            return urls.slice(idx, Math.min(urls.length, idx + TAIL_PRIORITY_BATCH));
          });
          phaseBByCam = perCam.map((urls) => {
            const idx = timeToIdx(focusTime, urls.length);
            return urls.slice(0, idx).reverse();
          });
        }

        const phaseA = interleave(phaseAByCam);
        console.log(
          `[MultiReplay prefetch] Phase A: ${phaseA.length} ${isInitial ? 'tail' : `forward-from-${focusTime?.toFixed(1)}s`} segments`,
        );
        await runPool(phaseA, (u) => fetchUrl(u, 'high'), PREFETCH_CONCURRENCY);
        if (ac.signal.aborted) return;

        // Tail segments are now in the browser HTTP cache — release the
        // tail-seek gate (only relevant on the initial run; re-focused
        // prefetches always run after the tail-seek has applied so the
        // gate is already true).
        if (isInitial) {
          phaseAReadyRef.current = true;
          console.log('[MultiReplay prefetch] Phase A complete — tail-seek can now land');
        }

        const phaseB = interleave(phaseBByCam);
        console.log(
          `[MultiReplay prefetch] Phase B: ${phaseB.length} backward segments`,
        );
        await runPool(phaseB, (u) => fetchUrl(u), PREFETCH_CONCURRENCY);
        if (ac.signal.aborted) return;
        console.log('[MultiReplay prefetch] complete');
      } finally {
        if (inFlightPrefetches.get(key) === ac) {
          inFlightPrefetches.delete(key);
        }
      }
    })();

    return () => {
      // Abort on cleanup so a manual seek (which bumps prefetchGen and
      // thus re-runs this effect) reliably stops the in-flight Phase B
      // walk before the new run starts. In dev StrictMode this also
      // aborts the first mount's Phase A — wasteful but the second
      // mount's re-run picks up where it left off via the browser HTTP
      // cache, so no user-visible delay.
      ac.abort();
    };
  }, [meta, loading, segmentId, minutes, duration, prefetchGen]);

  // Initial seek to startOffsetSec for each camera. Only runs as a
  // fallback when the tail-seek hasn't applied — otherwise it would
  // override the leader's tail position back to its small startOffsetSec
  // (~3 s) and snap the seeker UI back to ~0:03 a few hundred ms after
  // the tail seek lands.
  useEffect(() => {
    if (!meta || loading) return;

    const applyInitialSeek = () => {
      if (didInitialTailSeekRef.current) return;
      for (const cam of CAMERAS) {
        const v = refs[cam].current?.getVideo();
        const offset = meta.syncOffsets?.[cam]?.startOffsetSec ?? 0;
        if (v && Number.isFinite(v.duration) && offset > 0) {
          v.currentTime = offset;
        }
      }
    };

    const timerId = setTimeout(applyInitialSeek, 500);
    return () => { clearTimeout(timerId); };
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
                      initialPosition={camInitialPositions[cam]}
                    />
                  ) : (
                    <div className="multi-replay-unavailable">
                      {info.error || 'Unavailable'}
                    </div>
                  )}
                  {isActive && isBuffering && (
                    <div className="multi-replay-buffering" aria-live="polite">
                      <div className="multi-replay-buffering-spinner" />
                      <span>Buffering…</span>
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
            <div className="replay-seek-track">
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
              {eventMarkers.map((ev, i) => (
                <button
                  key={i}
                  type="button"
                  className={`event-marker${ev.side === 'source_side' ? ' event-marker--source' : ' event-marker--sink'}`}
                  style={{ left: `${ev.fraction * 100}%` }}
                  onClick={() => void openEventClips(ev)}
                  title={`${ev.label} · F${ev.frame} · ${formatTime(ev.timeSec)} · Score ${ev.score.toFixed(2)} — show clip`}
                  aria-label={`Event at ${formatTime(ev.timeSec)}, show bounce clip`}
                />
              ))}
            </div>
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

              <span className="replay-event-nav-sep" aria-hidden="true" />

              <button
                type="button"
                className="replay-ctrl-btn replay-event-nav-btn"
                onClick={jumpToPrevEvent}
                disabled={!canPrevEvent}
                title="Jump to previous bounce event"
                aria-label="Previous bounce event"
              >
                ◀ Event
              </button>
              <button
                type="button"
                className="replay-ctrl-btn replay-event-nav-btn"
                onClick={jumpToNextEvent}
                disabled={!canNextEvent}
                title="Jump to next bounce event"
                aria-label="Next bounce event"
              >
                Event ▶
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

        {selectedEventClips && (
          <div className="event-clip-panel" role="region" aria-label="Bounce event clip">
            <div className="event-clip-panel-header">
              <h2 className="event-clip-panel-title">
                Event clip · Frame {selectedEventClips.marker.frame} · {selectedEventClips.marker.label}{' '}
                · {selectedEventClips.marker.side.replace(/_/g, ' ')}
              </h2>
              <button
                type="button"
                className="replay-ctrl-btn event-clip-panel-close"
                onClick={() => setSelectedEventClips(null)}
                aria-label="Close event clip panel"
              >
                Close
              </button>
            </div>
            {selectedEventClips.loading && (
              <p className="event-clip-panel-status">Loading clip list…</p>
            )}
            {selectedEventClips.error && (
              <p className="event-clip-panel-error">{selectedEventClips.error}</p>
            )}
            {!selectedEventClips.loading && !selectedEventClips.error && selectedEventClips.clips.length === 0 && (
              <p className="event-clip-panel-status">
                No clip files listed for this frame (see events/bounce_events_clips.csv and events/bounce_clips/).
              </p>
            )}
            {selectedEventClips.clips.length > 0 && (
              <>
                <p className="event-clip-hint">
                  Clips are loaded fully then played (avoids Range/proxy glitches). They must be{' '}
                  <strong>H.264</strong> in MP4 for Chrome; old <code>mp4v</code> (MPEG-4 Part 2) files will not play.
                </p>
                <div className="event-clip-grid">
                  {selectedEventClips.clips.map((c) => (
                    <div key={`${c.camera}-${c.clipName}`} className="event-clip-cell">
                      <div className="event-clip-label">
                        {CAMERA_LABELS[c.camera as CameraKey] ?? c.camera}
                      </div>
                      <BounceClipVideo url={c.url} label={`${c.camera} ${c.clipName}`} />
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
