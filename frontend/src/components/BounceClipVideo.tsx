import { useEffect, useRef, useState } from 'react';

/** Bounce clips are short; buffer fully so Range/proxy issues do not break <video>. */
const MAX_CLIP_BYTES = 80 * 1024 * 1024;

type BounceClipVideoProps = {
  url: string;
  /** Shown in aria-label / title */
  label: string;
};

function decodeErrorMessage(code: number | undefined): string {
  if (code === MediaError.MEDIA_ERR_SRC_NOT_SUPPORTED || code === MediaError.MEDIA_ERR_DECODE) {
    return (
      'This browser cannot decode this file. Clips must be H.264 (AVC) in MP4 for Chrome — not MPEG-4 Part 2 (mp4v). ' +
      'From the judex-web repo run: ./scripts/reencode-bounce-clips-h264.sh — or manually: ' +
      'ffmpeg -i in.mp4 -c:v libx264 -crf 23 -preset fast -movflags +faststart -c:a aac out.mp4'
    );
  }
  if (code === MediaError.MEDIA_ERR_NETWORK) {
    return 'Network error while loading the video.';
  }
  return 'Playback failed.';
}

/**
 * Loads the clip with fetch (full body) then plays from a blob URL.
 * Avoids progressive Range requests that some proxies mishandle; same codec rules as a normal URL.
 */
export function BounceClipVideo({ url, label }: BounceClipVideoProps) {
  const [blobUrl, setBlobUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [fetchError, setFetchError] = useState<string | null>(null);
  const [playError, setPlayError] = useState<string | null>(null);
  const blobUrlRef = useRef<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setFetchError(null);
    setPlayError(null);
    setBlobUrl(null);

    if (blobUrlRef.current) {
      URL.revokeObjectURL(blobUrlRef.current);
      blobUrlRef.current = null;
    }

    (async () => {
      try {
        const res = await fetch(url, { credentials: 'same-origin' });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const cl = res.headers.get('content-length');
        if (cl != null) {
          const n = Number(cl);
          if (Number.isFinite(n) && n > MAX_CLIP_BYTES) {
            throw new Error('Clip is too large to load in-page; open the URL in a new tab.');
          }
        }
        const buf = await res.arrayBuffer();
        if (cancelled) return;
        if (buf.byteLength > MAX_CLIP_BYTES) {
          throw new Error('Clip is too large to load in-page.');
        }
        const blob = new Blob([buf], { type: 'video/mp4' });
        const objectUrl = URL.createObjectURL(blob);
        if (cancelled) {
          URL.revokeObjectURL(objectUrl);
          return;
        }
        blobUrlRef.current = objectUrl;
        setBlobUrl(objectUrl);
      } catch (e) {
        if (!cancelled) {
          setFetchError(e instanceof Error ? e.message : 'Failed to load clip');
        }
      } finally {
        setLoading(false);
      }
    })();

    return () => {
      cancelled = true;
      if (blobUrlRef.current) {
        URL.revokeObjectURL(blobUrlRef.current);
        blobUrlRef.current = null;
      }
    };
  }, [url]);

  if (loading) {
    return <p className="event-clip-panel-status">Loading clip…</p>;
  }
  if (fetchError) {
    return (
      <p className="event-clip-panel-error" role="alert">
        {fetchError}{' '}
        <a href={url} className="event-clip-open-link" target="_blank" rel="noreferrer">
          Open file
        </a>
      </p>
    );
  }
  if (!blobUrl) {
    return <p className="event-clip-panel-error">No clip data.</p>;
  }

  return (
    <>
      <video
        className="event-clip-video"
        src={blobUrl}
        controls
        playsInline
        muted
        preload="auto"
        aria-label={label}
        title={label}
        onError={(e) => {
          const code = e.currentTarget.error?.code;
          setPlayError(decodeErrorMessage(code));
        }}
      />
      {playError && (
        <p className="event-clip-video-error" role="alert">
          {playError}{' '}
          <a href={url} className="event-clip-open-link" target="_blank" rel="noreferrer">
            Try opening file
          </a>
        </p>
      )}
    </>
  );
}
