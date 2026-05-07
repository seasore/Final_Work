import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        /** 节假日多节假日回测等可能接近 10 分钟，需与 axios HOLIDAY_ACCURACY_TIMEOUT_MS 一致或更大 */
        timeout: 660000,
        proxyTimeout: 660000,
      },
    },
  },
})
