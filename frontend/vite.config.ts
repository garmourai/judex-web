import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': { target: 'http://localhost:3014', changeOrigin: true },
      '/stream': { target: 'http://localhost:3014', changeOrigin: true },
      '/cam': { target: 'http://localhost:3014', changeOrigin: true },
      '/events': {
        target: 'http://localhost:3014',
        changeOrigin: true,
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
