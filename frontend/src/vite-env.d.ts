/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_STREAM_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
