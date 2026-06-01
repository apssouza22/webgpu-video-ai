# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
npm run dev      # Dev server at http://localhost:5180
npm run build    # Type-check (tsc) + Vite bundle → docs/
npm run preview  # Serve the production build locally
```

No test suite exists — this is a visual prototype, tested manually in the browser.

## Architecture

This is a **browser-based GPU video compositor** with real-time AI object detection. It demonstrates WebGPU composition, MediaBunny decoding, Transformers.js inference, and Web Audio playback working together in a single-page app.

### Data flow

```
Composition (timeline definition)
  → CompositionPlayer (playback state + UI)
    → VideoPlayer (per-frame decode + GPU submit)
      → VideoFrameSource / ImageLoader  (MediaBunny / HTML Image)
      → GpuCompositor
          GpuVideoRenderer   — external texture from VideoFrame
          GpuImageRenderer   — cached rgba8unorm 2D textures
          GpuDetectionRenderer — bounding boxes via uniform buffer
    → AudioPlayer (Web Audio API, pre-decoded buffers)
    → VideoObjectDetection → ObjectDetector (Web Worker, RF-DETR via Transformers.js)
```

### Key modules

| Path | Role |
|------|------|
| `src/composition.ts` | Fluent timeline builder; derives duration from clips |
| `src/types.ts` | `Clip` / `VideoClip` / `ImageClip` / `AudioClip` base classes; `VideoFrameContext` |
| `src/main.ts` | Wires everything together; warms up RF-DETR on first frame |
| `src/gpu/GpuCompositor.ts` | Renders video → images → detection boxes in order; alpha blending |
| `src/gpu/AbstractGpuRenderer.ts` | Base class — pipeline creation, shader loading with compilation error checks |
| `src/detection/detection.worker.ts` | RF-DETR inference on `OffscreenCanvas`; model: `onnx-community/rfdetr_medium-ONNX` |
| `src/player/VideoObjectDetection.ts` | Detection scheduler — one frame in flight at a time, throttled |
| `src/media/VideoFrameSource.ts` | MediaBunny wrapper; `frameAtTime()` returns a `VideoFrame` that must be `close()`d |

### WebGPU shaders

WGSL shaders live in `src/shaders/` and are imported as raw strings (`?raw`). Each shader uses a full-screen triangle with a rect-bounds uniform (normalized 0–1) for positioning. The detection shader iterates up to **32 boxes** stored in a uniform buffer.

### Web Worker and CORS requirements

The detection worker (`src/detection/detection.worker.ts`) uses Transformers.js with `device: 'webgpu'`. Vite is configured to send `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp` headers, which are required for `SharedArrayBuffer` / WASM. These headers are in `vite.config.ts` — keep them when modifying the dev server config.

### Build output

Production build targets `docs/` with base path `/webgpu-video-encoding/` for GitHub Pages deployment. Transformers.js is excluded from `optimizeDeps` because it downloads models dynamically at runtime.

### Clip coordinate system

All clip positions and sizes (`x`, `y`, `width`, `height`) are normalized **0–1** relative to the composition canvas. Conversions happen inside the GPU shaders.
