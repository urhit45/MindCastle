import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: true,  // expose on 0.0.0.0 — lets phones on the same WiFi connect
    port: 5173,
  },
})
