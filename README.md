# WebGPU Video Composition + Real-Time AI

An experimental browser project for **WebGPU video composition** with **real-time object detection** — all in the browser, without a backend.

The goal is to explore how far a native GPU pipeline can go when decode, composition, inference, and overlay drawing stay on the GPU (or in a worker with WebGPU), while the main thread keeps the preview responsive.

For a longer write-up on browser-native video decode, processing, and encoding with WebCodecs, WebGPU, and MediaBunny, see [ARTICLE.md](./ARTICLE.md).

**Live demo:** [apssouza22.github.io/webgpu-video-ai](https://apssouza22.github.io/webgpu-video-ai/)

![Real-time AI architecture](./assets/diagrams/realtime-ai-architecture.drawio.png)

This is not a full editor. It is a focused playground for the building blocks of a GPU-powered compositor with on-device vision.

## What it does

In a supported desktop browser (Chrome or Edge recommended), the app:

1. Loads a timeline-style composition with **video**, **image overlay**, and **audio** layers
2. Decodes video and audio with **MediaBunny**
3. Composes video and image layers with **WebGPU** shaders (`GpuCompositor`: video → images → detection boxes)
4. Runs **RF-DETR** object detection on each decoded video frame via **Transformers.js** (`device: 'webgpu'`) in a **dedicated Web Worker**
5. Draws bounding boxes in a **second WebGPU pass** (not a 2D canvas overlay)
6. Shows an interactive preview with play/pause, scrubber, and live render/detection FPS

The demo composition is **1280×720 at 30 fps**: one `video.mp4` clip (duration from the file), explicit audio from the same file, and two transparent PNG overlays from **1s to 3s** (right and left).

## Architecture

```
Composition (timeline)
  → CompositionPlayer
    → VideoPlayer — MediaBunny decode, GpuCompositor render
    → VideoObjectDetection — clones VideoFrame, transfers to worker
    → ObjectDetector / detection.worker — RF-DETR on WebGPU
    → GpuDetectionRenderer — box overlay in WebGPU
```

Additional diagrams live under [`assets/diagrams/`](./assets/diagrams/) (compositing order, frame transfer, detection throttle, ONNX graph, and more).

## Why WebGPU?

Video composition is naturally GPU-shaped work: each frame is textures (base video, overlays, transforms, opacity). WebGPU gives the browser a modern graphics pipeline so pixels are not bounced through the CPU for every layer.

This project uses WebGPU in **two** places:

| Stage | WebGPU role |
|-------|-------------|
| **Composition** | Sample `VideoFrame` as external textures; blend image overlays; draw up to 32 detection boxes from a uniform buffer |
| **Inference** | Transformers.js runs the RF-DETR ONNX model with `device: 'webgpu'` inside the worker |

Keeping composition and box drawing on the GPU avoids readback for preview. Detection still preprocesses the frame once in the worker (`OffscreenCanvas` at 576×576 for the model), but the main thread never blocks on inference.

## Composition API

Compositions are built from ordered clip layers. A frame context exposes active clips at a timeline time; video clips decode their next source frame from that context.

```ts
import { AudioClip, Composition, ImageClip, VideoClip } from './src/composition';

const composition = new Composition(30, 1280, 720);

composition
  .addLayer(new VideoClip('/samples/video.mp4', 0))
  .addLayer(new AudioClip('/samples/video.mp4', 0))
  .addLayer(new ImageClip('/samples/overlay.png', 1, 3, 0.62, 0.08, 0.32, 0.32, 0.92))
  .addLayer(new ImageClip('/samples/overlay-2.png', 1, 3, 0, 0.08, 0.32, 0.32, 0.92));

const frame = composition.getFrameContextAtTime(2.5);
const sourceFrame = await frame.videos[0]?.nextSourceFrame();
```

Clip positions and sizes (`x`, `y`, `width`, `height`) are normalized **0–1** relative to the composition canvas.

## Object detection

Detection follows the same stack as the [RF-DETR WebGPU demo](https://huggingface.co/spaces/webml-community/RF-DETR-Medium-WebGPU):

- **Model:** `onnx-community/rfdetr_medium-ONNX` (`@huggingface/transformers`, `device: 'webgpu'`, `dtype: 'fp32'`)
- **Worker:** `src/detection/detection.worker.ts` — loads the model once, runs inference off the main thread
- **Input:** cloned **`VideoFrame`** from the compositor’s base video layer, **transferred** to the worker (no main-thread canvas readback)
- **Preprocess:** worker draws the frame once into a 576×576 `OffscreenCanvas` for the model
- **Output:** boxes converted to GPU uniforms and rendered by `GpuDetectionRenderer` after video and image layers
- **Scheduling:** at most one detection in flight; if a new frame arrives while busy, only the latest pending frame is kept

Detection runs on the **decoded video frame** (before image overlays are composited), matching the base layer the compositor samples.

## Requirements

- Browser with **WebGPU** (composition and RF-DETR inference)
- **Chrome or Edge (desktop)** recommended
- Dev/preview server sends **COOP/COEP** headers (see `vite.config.ts`) for Transformers.js / WASM
- Sample media in `public/samples/`:
  - `video.mp4` — main video clip, ideally with an audio track
  - `overlay.png` — transparent PNG, right side (demo: 1s–3s)
  - `overlay-2.png` — transparent PNG, left side (demo: 1s–3s)

## Quick start

```bash
npm install
# add your files under public/samples/:
#   video.mp4
#   overlay.png
#   overlay-2.png
npm run dev
```

Open http://localhost:5180. The app checks sample media, downloads the RF-DETR weights from Hugging Face on first load, warms up WebGPU shaders, then you can play and scrub the timeline.

Production build: `npm run build` → `docs/` with base path `/webgpu-video-ai/` for GitHub Pages.

## License

MIT (demo code)
