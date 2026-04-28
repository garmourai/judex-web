// Backend base URL.
//
// Default behavior: empty string → all `resolveApi()` calls return RELATIVE
// paths, which the browser resolves against the page origin. In dev that
// means requests go through the Vite proxy (see vite.config.ts); in
// production the backend serves the bundle itself, so same-origin trivially
// reaches the API. This works in every environment, including SSH port-
// forwarded dev (`localhost:33051 → Pi's Vite at 5173`), where the backend
// port (3014) is typically NOT forwarded and a direct `https://localhost:3014`
// fetch from the user's laptop fails with ERR_CONNECTION_REFUSED.
//
// Opt-in HTTP/2 direct mode: set `VITE_API_BASE` to the absolute backend
// URL (e.g. `https://localhost:3014` or `https://192.168.0.10:3014`). This
// bypasses the Vite proxy and lets segment fetches multiplex over a single
// HTTP/2 TLS connection, lifting the browser's HTTP/1.1 6-connection cap.
// Requirements for that to work:
//   1. The backend must be reachable from the browser at that URL (no
//      firewall, port not blocked, port-forwarded if remote).
//   2. The backend's self-signed cert must be trusted (visit the URL once
//      and click through the warning, or install the cert).
//   3. If running through SSH port forwarding, forward 3014 as well.
export const API_BASE: string =
  (import.meta.env.VITE_API_BASE as string | undefined) ?? '';

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
