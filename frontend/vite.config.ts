import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': { target: 'http://localhost:3014', changeOrigin: true },
      '/stream': { target: 'http://localhost:3014', changeOrigin: true },
    },
  },
});
