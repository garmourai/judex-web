import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from 'react';
import Hls from 'hls.js';
import type { ComponentProps } from 'react';
import './StreamPlayer.css';

export type StreamPlayerHandle = {
  getVideo: () => HTMLVideoElement | null;
};

type StreamPlayerProps = {
  src: string;
  /**
   * `vod` = full timeline seekable (5‑min replay, etc.). Do not clamp seeks to `video.seekable`:
   * with MSE, seekable often tracks buffered range first (~30s), so clamping breaks seeks and
   * prevents hls.js from loading distant .ts segments.
   * `live` = clamp seeks into the valid DVR/live window.
   * `auto` = detect from playlist (#EXT-X-ENDLIST / VOD).
   */
  playbackMode?: 'auto' | 'vod' | 'live';
  /** Hide the native <video> controls (e.g. when the parent provides custom controls). */
  hideNativeControls?: boolean;
} & Omit<ComponentProps<'div'>, 'children'>;

/** VOD playlists must include #EXT-X-ENDLIST (or PLAYLIST-TYPE:VOD) or hls.js treats them as live and only shows the "live edge" (~last few segments). */
async function playlistLooksLikeVod(playlistUrl: string): Promise<boolean> {
  if (playlistUrl.startsWith('blob:')) return false;
  try {
    const res = await fetch(playlistUrl);
    const text = await res.text();
    if (/#EXT-X-PLAYLIST-TYPE:\s*VOD/i.test(text)) return true;
    if (/#EXT-X-ENDLIST/i.test(text)) return true;
    return false;
  } catch {
    return false;
  }
}

function createHlsOptions(isVod: boolean): Partial<Hls['config']> {
  if (isVod) {
    return {
      enableWorker: true,
      lowLatencyMode: false,
      // Buffer the entire VOD so every seek lands on already-loaded data.
      maxBufferLength: 600,
      maxMaxBufferLength: 600,
      maxBufferSize: 300 * 1000 * 1000,
      // Never evict old segments — the file is short enough to keep everything in memory.
      backBufferLength: -1,
    };
  }
  // Live + DVR: seek back within the playlist window (e.g. 2:15). Requires ffmpeg to keep a long enough
  // sliding playlist (HLS_LIST_SIZE × segment duration). Default hls.js maxBufferLength is 30s — too small.
  return {
    enableWorker: true,
    lowLatencyMode: true,
    liveSyncDurationCount: 2,
    liveMaxLatencyDurationCount: 5,
    // Finite duration = seek bar matches the DVR window (playlist length), not Infinity.
    liveDurationInfinity: false,
    // How far ahead to buffer from playhead (seeking far behind live needs room up to the live edge).
    maxBufferLength: 600,
    maxMaxBufferLength: 600,
    maxBufferSize: 120 * 1000 * 1000,
    backBufferLength: 120,
  };
}

function clampToSeekable(video: HTMLVideoElement) {
  if (!video.seekable.length) return;
  const start = video.seekable.start(0);
  const end = video.seekable.end(video.seekable.length - 1);
  // Keep playhead in valid range to avoid browser snapping to a broken position.
  if (video.currentTime < start) video.currentTime = start + 0.05;
  if (video.currentTime > end) video.currentTime = Math.max(start, end - 0.05);
}

export const StreamPlayer = forwardRef<StreamPlayerHandle, StreamPlayerProps>(function StreamPlayer(
  { src, className = '', playbackMode = 'auto', hideNativeControls = false, ...rest },
  ref,
) {
  const videoRef = useRef<HTMLVideoElement>(null);

  useImperativeHandle(ref, () => ({ getVideo: () => videoRef.current }), []);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<'idle' | 'loading' | 'playing' | 'error'>('idle');

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !src) return;

    setError(null);
    setStatus('loading');

    const state = {
      cancelled: false,
      hls: null as Hls | null,
      onNativeMeta: null as (() => void) | null,
      onNativeErr: null as (() => void) | null,
    };

    let removeSeekClamp: (() => void) | undefined;

    const run = async () => {
      const isVod =
        playbackMode === 'vod'
          ? true
          : playbackMode === 'live'
            ? false
            : await playlistLooksLikeVod(src);
      if (state.cancelled) return;

      const clampSeeks = !isVod;
      const onSeeking = () => clampToSeekable(video);
      if (clampSeeks) {
        video.addEventListener('seeking', onSeeking);
        removeSeekClamp = () => video.removeEventListener('seeking', onSeeking);
      }

      if (Hls.isSupported()) {
        const instance = new Hls({
          ...createHlsOptions(isVod),
        });
        state.hls = instance;
        if (state.cancelled) {
          instance.destroy();
          state.hls = null;
          removeSeekClamp?.();
          return;
        }

        instance.loadSource(src);
        instance.attachMedia(video);

        instance.on(Hls.Events.MANIFEST_PARSED, () => {
          if (!state.cancelled) setStatus('playing');
        });

        if (isVod) {
          const onVodSeeking = () => {
            const t = video.currentTime;
            const buf = video.buffered;
            let covered = false;
            for (let i = 0; i < buf.length; i++) {
              if (t >= buf.start(i) - 0.5 && t <= buf.end(i) + 0.5) {
                covered = true;
                break;
              }
            }
            if (!covered) instance.startLoad(t);
          };
          video.addEventListener('seeking', onVodSeeking);
          removeSeekClamp = () => video.removeEventListener('seeking', onVodSeeking);
        }

        instance.on(Hls.Events.ERROR, (_e, data) => {
          if (!data.fatal) return;
          // Recover from transient live errors instead of immediately failing.
          if (data.type === Hls.ErrorTypes.NETWORK_ERROR) {
            instance.startLoad();
            return;
          }
          if (data.type === Hls.ErrorTypes.MEDIA_ERROR) {
            instance.recoverMediaError();
            return;
          }
          setError(data.type + ': ' + (data.details ?? 'Unknown'));
          setStatus('error');
        });
        return;
      }

      if (video.canPlayType('application/vnd.apple.mpegurl')) {
        video.src = src;
        const onMeta = () => {
          if (!state.cancelled) setStatus('playing');
        };
        const onErr = () => {
          setError('Playback failed');
          setStatus('error');
        };
        state.onNativeMeta = onMeta;
        state.onNativeErr = onErr;
        video.addEventListener('loadedmetadata', onMeta);
        video.addEventListener('error', onErr);
        return;
      }

      removeSeekClamp?.();
      removeSeekClamp = undefined;
      setError('HLS not supported in this browser');
      setStatus('error');
    };

    void run();

    return () => {
      state.cancelled = true;
      removeSeekClamp?.();
      state.hls?.destroy();
      state.hls = null;
      if (state.onNativeMeta) video.removeEventListener('loadedmetadata', state.onNativeMeta);
      if (state.onNativeErr) video.removeEventListener('error', state.onNativeErr);
    };
  }, [src, playbackMode]);

  return (
    <div className={`stream-player ${className}`.trim()} {...rest}>
      <div className="stream-player-video-wrap">
        <video
          ref={videoRef}
          className="stream-player-video"
          controls={!hideNativeControls}
          muted
          playsInline
          aria-label="HLS stream"
        />
        {status === 'loading' && (
          <div className="stream-player-overlay" aria-hidden>
            <span>Loading stream…</span>
          </div>
        )}
        {status === 'error' && error && (
          <div className="stream-player-overlay stream-player-overlay--error" aria-live="polite">
            <span>{error}</span>
          </div>
        )}
      </div>
    </div>
  );
});
