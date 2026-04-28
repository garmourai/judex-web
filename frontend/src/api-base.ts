// Backend base URL.
//
// Production (frontend dist served by the backend itself): same origin, so an
// empty string keeps requests relative.
//
// Dev (Vite at :5173): we prefer the browser to hit the HTTPS+HTTP/2 backend
// at :3014 directly so segment fetches multiplex over a single TLS connection
// instead of being squeezed through the Vite proxy's HTTP/1.1 6-connection
// cap. We derive the host from `window.location` so this also works when
// the page is opened from another LAN device (e.g. http://192.168.0.10:5173).
//
// Override in any env via `VITE_API_BASE` (set to a non-empty string to pin
// to a specific host, or leave unset for the auto behavior).
export const API_BASE: string =
  (import.meta.env.VITE_API_BASE as string | undefined) ||
  (import.meta.env.DEV ? `https://${window.location.hostname}:3014` : '');

/**
 * Resolve a path-or-URL against {@link API_BASE}. Pass-through for already
 * absolute URLs (those starting with `http://` or `https://`).
 */
export function resolveApi(pathOrUrl: string): string {
  if (/^https?:\/\//i.test(pathOrUrl)) return pathOrUrl;
  if (!API_BASE) return pathOrUrl;
  if (pathOrUrl.startsWith('/')) return `${API_BASE}${pathOrUrl}`;
  return `${API_BASE}/${pathOrUrl}`;
}
