import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { StreamPlayer } from './StreamPlayer';
import type { StreamPlayerHandle } from './StreamPlayer';
import { BounceClipVideo } from './BounceClipVideo';
import { resolveApi } from '../api-base';
import './MultiReplayScreen.css';

const SPEEDS = [0.25, 0.5, 1, 1.25, 1.5] as const;
const SKIP_SECONDS = 5;
const SYNC_THRESHOLD_SEC = 0.15;
const FPS = 30;
// On open, jump to the last N seconds of the synced window so the most recent
// action plays immediately. Backward scrubbing relies on the parallel .ts prefetch.
const TAIL_OFFSET_SEC = 10;
// Number of trailing segments per camera to prefetch with priority before
// firing the rest of the timeline. Awaiting these helps hls.js's first
// segment fetch on the tail seek hit the cache.
const TAIL_PRIORITY_BATCH = 3;

// Map keyed by `${segmentId}:${minutes}` so the same replay window only
// prefetches once across React StrictMode tear-down/remount and Vite HMR
// remounts. Lives at module scope so it's not destroyed by component
// lifecycle. The AbortController is kept around in case a future caller
// wants to cancel, but we never abort on tear-down (that was the bug).
const inFlightPrefetches = new Map<string, AbortController>();

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

  // Fetch multi-camera replay metadata
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    (async () => {
      try {
        const res = await fetch(resolveApi(`/api/replay/multi/${segmentId}?minutes=${minutes}`));
        const data = await res.json();
        if (!res.ok) throw new Error(data?.error ?? 'Failed to load');
        // Resolve camera playlist URLs against API_BASE so hls.js (and our
        // prefetch) hit the HTTPS+HTTP/2 backend directly. Any .ts URIs in
        // the playlist are relative, so they'll be resolved against this
        // absolute playlist URL automatically by both hls.js and the URL
        // constructor we use during prefetch.
        if (data && data.cameras) {
          for (const cam of Object.keys(data.cameras)) {
            const ci = data.cameras[cam];
            if (ci && typeof ci.url === 'string') {
              ci.url = resolveApi(ci.url);
            }
          }
        }
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
    const es = new EventSource(resolveApi('/api/events/live/stream'));

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

      if (leader && !seekingRef.current && !leader.seeking) {
        const t = leader.currentTime;
        const syncT = Math.max(0, t - sourceStartOffset);
        // Don't let the rAF loop snap the seeker back to 0 before the
        // initial tail seek has actually been applied. Without this gate
        // the seeker briefly shows the pre-set tail position then flashes
        // to 0:00 while hls.js is still loading the manifest, then jumps
        // back once the seek fires. After the seek (or once leader has
        // any non-zero currentTime), we resume normal tracking.
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
        const r = await fetch(resolveApi(`/api/event-clips/${ev.frame}`));
        const data = (await r.json()) as { clips?: EventClipRow[]; error?: string };
        if (!r.ok) throw new Error(data?.error ?? 'Failed to load clip list');
        const clips = (Array.isArray(data?.clips) ? data.clips : []).map((c) => ({
          ...c,
          url: resolveApi(c.url),
        }));
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

  const didInitialTailSeekRef = useRef(false);

  // Reset the tail-seek guard whenever the replay window changes so a new
  // (segmentId, minutes) load triggers a fresh seek attempt.
  useEffect(() => {
    didInitialTailSeekRef.current = false;
  }, [segmentId, minutes]);

  // Pre-position the seeker UI at the tail target as soon as we know the
  // replay duration — i.e. immediately after meta loads, well before
  // hls.js has fetched enough segments for the actual seek to land. The
  // rAF loop above is gated to NOT overwrite this until the real seek
  // applies, so the user sees the playhead at "last 10 s" from the
  // moment the player appears instead of at 0:00 → flash → 0:00 → jump.
  useEffect(() => {
    if (!meta || loading) return;
    if (didInitialTailSeekRef.current) return;
    const syncDur = meta.syncDurationSec ?? duration;
    if (!syncDur || syncDur <= 0) return;
    setCurrentTime(Math.max(0, syncDur - TAIL_OFFSET_SEC));
  }, [meta, loading, duration]);

  // Prefetch all .ts segments for all 3 cameras into the browser HTTP cache.
  //
  // State lives at module level (`inFlightPrefetches`) so React StrictMode's
  // intentional tear-down + remount doesn't cancel the work. The previous
  // implementation kept a `prefetchKickedOffRef` and returned an
  // `() => ac.abort()` cleanup; in StrictMode that cleanup ran between the
  // two mounts, killing every in-flight fetch, and the second mount bailed
  // at the guard — so the prefetch never actually completed.
  //
  // We still keep an AbortController in the Map for symmetry with future
  // cancellation callers, but we never abort on cleanup.
  useEffect(() => {
    if (!meta || loading) return;
    const key = `${segmentId}:${minutes}`;
    if (inFlightPrefetches.has(key)) return;

    const ac = new AbortController();
    inFlightPrefetches.set(key, ac);

    // `priority: 'high'` is a Chromium-only Fetch hint (TS doesn't ship it
    // in lib.dom yet); other browsers ignore the field.
    //
    // We only bump the *tail* batch to 'high' so it lands in cache before
    // the upcoming seek. Backfill is left at default ('auto') because
    // Chrome's resource scheduler aggressively *throttles* fetches tagged
    // 'low' (it holds them back when other resources are in flight), which
    // would actively reduce parallelism rather than improve it.
    type PriorityInit = RequestInit & { priority?: 'high' | 'low' | 'auto' };
    const fetchUrl = (u: string, priority?: 'high') =>
      fetch(u, {
        signal: ac.signal,
        ...(priority ? { priority } : {}),
      } as PriorityInit).catch(() => {});

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

    // Strict tail-first ordering:
    //   Phase 1: fetch all 3 playlists in parallel.
    //   Phase 2: fetch the tail batch (last TAIL_PRIORITY_BATCH segments
    //            per camera, ~12 s of video covering the last 10 s) in
    //            parallel and AWAIT it. Under HTTP/1.1 the browser's
    //            6-connection pool processes requests in dispatch order,
    //            so awaiting here is what actually guarantees these
    //            segments arrive before the backfill starts competing
    //            for the same connections. Under HTTP/2 the streams
    //            multiplex anyway and the await is essentially free.
    //   Phase 3: fan out the rest of each camera's timeline in parallel.
    //            By the time we reach this phase the upcoming tail seek
    //            has either already landed in cache (Phase 2 segments)
    //            or is about to.
    void (async () => {
      const reversedByCam = await Promise.all(
        CAMERAS.map(async (cam) => {
          const u = meta.cameras[cam]?.url;
          if (!u) return [] as string[];
          return (await fetchPlaylistTsUrls(u)).reverse();
        }),
      );
      if (ac.signal.aborted) return;

      const tailUrls = reversedByCam.flatMap((urls) => urls.slice(0, TAIL_PRIORITY_BATCH));
      if (import.meta.env.DEV) {
        console.log(`[MultiReplay prefetch] tail-first: awaiting ${tailUrls.length} segments`);
      }
      await Promise.all(tailUrls.map((u) => fetchUrl(u, 'high')));
      if (ac.signal.aborted) return;
      if (import.meta.env.DEV) {
        console.log('[MultiReplay prefetch] tail done, firing backfill');
      }

      const backfillUrls = reversedByCam.flatMap((urls) => urls.slice(TAIL_PRIORITY_BATCH));
      await Promise.all(backfillUrls.map((u) => fetchUrl(u)));
      if (import.meta.env.DEV) {
        console.log(`[MultiReplay prefetch] backfill done (${backfillUrls.length} segments)`);
      }
    })();

    // No cleanup that aborts. StrictMode tear-down/remount won't kill the
    // work; a true unmount lets the in-flight fetches finish (the browser
    // caches them anyway). If the user re-opens the same window, the Map
    // entry is still present and we skip.
  }, [meta, loading, segmentId, minutes]);

  // Tail-seek effect, independent of the prefetch.
  //
  // Polls for the source leader video to have a finite, positive `duration`
  // (Chrome silently drops `video.currentTime = X` while duration is NaN),
  // then seeks to `syncDurationSec - TAIL_OFFSET_SEC`. After 1.2 s it
  // verifies the seek stuck — if hls.js MANIFEST_PARSED snapped the
  // playhead back to 0 in between, we issue the seek a second time.
  useEffect(() => {
    if (!meta || loading) return;
    if (didInitialTailSeekRef.current) return;

    const syncDur = meta.syncDurationSec ?? duration;
    if (!syncDur || syncDur <= 0) return;

    const target = Math.max(0, syncDur - TAIL_OFFSET_SEC);
    const targetSourceTime = target + sourceStartOffset;

    let cancelled = false;
    let pollId: ReturnType<typeof setTimeout> | undefined;
    let verifyId: ReturnType<typeof setTimeout> | undefined;
    let attempts = 0;
    const MAX_ATTEMPTS = 80; // ~8s @ 100ms

    const trySeek = () => {
      if (cancelled || didInitialTailSeekRef.current) return;
      const leader = sourceRef.current?.getVideo();
      const ready = leader && Number.isFinite(leader.duration) && leader.duration > 0;
      if (ready) {
        seekToSyncTime(target);
        didInitialTailSeekRef.current = true;
        verifyId = setTimeout(() => {
          if (cancelled) return;
          const l = sourceRef.current?.getVideo();
          if (l && l.currentTime < targetSourceTime - 5) {
            seekToSyncTime(target);
          }
        }, 1200);
        return;
      }
      if (attempts++ < MAX_ATTEMPTS) pollId = setTimeout(trySeek, 100);
    };

    pollId = setTimeout(trySeek, 100);
    return () => {
      cancelled = true;
      if (pollId) clearTimeout(pollId);
      if (verifyId) clearTimeout(verifyId);
    };
  }, [meta, loading, duration, seekToSyncTime, sourceStartOffset, segmentId, minutes]);

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
