import path from "path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
// import { visualizer } from "rollup-plugin-visualizer";
// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
    // visualizer({
    //   //번들 시각화를 위함
    //   open: true, // 빌드 후 자동으로 시각화 파일 열기
    //   filename: "stats.html", // 출력 파일명
    //   gzipSize: true, // gzip 크기 표시
    //   brotliSize: true, // brotli 크기 표시
    // }),
  ],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "https://k12b105.p.ssafy.io",
        secure: true,
        changeOrigin: true,
        configure: (proxy, _options) => {
          proxy.on("proxyReq", function (proxyReq, req) {
            // 쿠키를 포함하여 요청을 전송
            proxyReq.setHeader("Cookie", req.headers.cookie || "");
          });
        },
        cookieDomainRewrite: "localhost", // 쿠키 도메인 재작성
        cookiePathRewrite: "/", // 쿠키 경로 재작성
      },
    },
    host: true,
    allowedHosts: true,
  },
  build: {
    outDir: "dist",
    terserOptions: {
      compress: {
        drop_console: true,
      },
    },
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
});
