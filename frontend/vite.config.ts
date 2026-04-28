import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Vite proxy is now a fallback path. When the frontend's API_BASE resolves
// to an absolute URL (default in dev — see src/api-base.ts), the browser
// hits the backend directly over HTTPS+HTTP/2, bypassing this proxy entirely.
// The proxy is kept so explicitly setting `VITE_API_BASE=` (empty) still
// works for anyone who wants relative URLs through Vite. `secure: false`
// is required because the backend now serves a self-signed cert.
const BACKEND = 'https://localhost:3014';
const proxyOpts = { target: BACKEND, changeOrigin: true, secure: false } as const;

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
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
