import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

// Reuse the backend's self-signed TLS cert (generated on first run into
// backend/.tls/) so the Vite dev server can serve HTTPS without the user
// having to trust a second cert. Both the page (this server) and direct
// backend API calls (https://<host>:3014) end up under one cert chain.
//
// HTTPS on Vite is required for two reasons:
//   1. The browser refuses to negotiate HTTP/2 over plaintext, and we want
//      HTTP/2 on direct-backend calls (VITE_API_BASE) without falling foul
//      of mixed-content blocking when the page itself is HTTP.
//   2. Some browser features (Service Workers, certain Permissions APIs)
//      require a secure context.
//
// If the cert hasn't been generated yet (first checkout, backend not run),
// we transparently fall back to HTTP and print a hint.
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const TLS_DIR = path.resolve(__dirname, '..', 'backend', '.tls');
const TLS_KEY = path.join(TLS_DIR, 'key.pem');
const TLS_CERT = path.join(TLS_DIR, 'cert.pem');

let httpsOpts: { key: Buffer; cert: Buffer } | undefined;
try {
  httpsOpts = {
    key: fs.readFileSync(TLS_KEY),
    cert: fs.readFileSync(TLS_CERT),
  };
} catch {
  console.warn(
    `[vite] Backend TLS cert not found at ${TLS_DIR} — serving plain HTTP.\n` +
      `       Start the backend once to generate it, then restart Vite to enable HTTPS+HTTP/2.`,
  );
}

// Vite proxy is now a fallback path. When the frontend's API_BASE resolves
// to an absolute URL (opt-in via VITE_API_BASE — see src/api-base.ts), the
// browser hits the backend directly over HTTPS+HTTP/2, bypassing this proxy
// entirely. The proxy is the default path otherwise. `secure: false` is
// required because the backend serves a self-signed cert.
const BACKEND = 'https://localhost:3014';
const proxyOpts = { target: BACKEND, changeOrigin: true, secure: false } as const;

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    https: httpsOpts,
    proxy: {
      '/api': proxyOpts,
      '/stream': proxyOpts,
      '/cam': proxyOpts,
      '/events': {
        ...proxyOpts,
        configure(proxy) {
          proxy.on('proxyReq', (proxyReq, req) => {
            const r = req.headers.range;
            if (r) proxyReq.setHeader('Range', r);
          });
        },
      },
    },
  },
});
