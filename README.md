# WebGPU Video Composition Preview

An experimental browser project for exploring high-performance video processing with **WebGPU**.

The goal is to understand how far a browser-native video pipeline can go when
composition work stays on the GPU: decode media frames, combine multiple visual
layers, and preview the timeline interactively.

For a longer write-up on browser-native video decode, processing, and encoding
with WebCodecs, WebGPU, and MediaBunny, read [ARTICLE.md](./ARTICLE.md).

Try the live demo at [apssouza22.github.io/webgpu-video-encoding](https://apssouza22.github.io/webgpu-video-encoding/).

![Media pipeline](./assets/diagrams/media-pipeline.drawio.png)

This is not a full editor. It is a focused playground for testing the building
blocks of a GPU-powered video compositor.

## What it does

In a supported desktop browser (Chrome/Edge recommended), the app:

1. Loads a timeline-style composition with ordered **video**, **image overlay**, and **audio** layers
2. Decodes video and audio with **MediaBunny**
3. Composes video and image layers with a **WebGPU shader**
4. Shows an interactive WebGPU preview player with play/pause controls, audio playback, and a scrubber
5. Runs **RF-DETR** object detection on each composed frame via **Transformers.js** (WebGPU), drawing bounding boxes on an overlay

The demo composition is 1280x720 at 30 fps. It plays `video.mp4` for the first 5 seconds, switches to `video-2.mp4`,
schedules explicit audio layers from the same files for preview playback, and displays two transparent image overlays
from 1s to 4s.

## Why WebGPU?

Video composition is naturally GPU-shaped work. Each frame can be treated as a
set of textures: a base video frame plus overlays, transforms, opacity, and later
effects. WebGPU gives the browser direct access to a modern graphics pipeline, so
this project keeps rendering on the GPU instead of copying pixels back through the CPU.

The current compositor is intentionally small: it composites one active video
layer with any active image overlays. The structure is meant to grow toward more
video layers, transitions, effects, and timeline behavior.

## Composition API

Compositions are built from ordered clip layers. A frame context exposes the
active clips at a timeline time, and active video clips can decode their next
source frame from that context.

```ts
import { AudioClip, Composition, ImageClip, VideoClip } from './src/composition';

const composition = new Composition(30, 1280, 720);

composition
  .addLayer(new VideoClip('/samples/video.mp4', 0, 5))
  .addLayer(new VideoClip('/samples/video-2.mp4', 5))
  .addLayer(new AudioClip('/samples/video.mp4', 0, 5))
  .addLayer(new AudioClip('/samples/video-2.mp4', 5))
  .addLayer(new ImageClip('/samples/overlay.png', 1, 3, 0.62, 0.08, 0.32, 0.32, 0.92));

const frame = composition.getFrameContextAtTime(2.5);
const sourceFrame = await frame.videos[0]?.nextSourceFrame();
```

## Object detection

Detection uses the same setup as the [RF-DETR WebGPU demo](https://huggingface.co/spaces/webml-community/RF-DETR-Medium-WebGPU): `onnx-community/rfdetr_medium-ONNX` via `@huggingface/transformers` with `device: 'webgpu'`. The model downloads from Hugging Face on first load (~tens of MB).

Use the detection panel to toggle inference, adjust the score threshold, filter COCO labels, and view detection FPS. Inference runs on the **composed** WebGPU canvas (video + image overlays), not raw source files alone.

## Requirements

- Browser with **WebGPU** (used for both composition and RF-DETR inference)
- **Chrome or Edge (desktop)** recommended
- Dev server serves `Cross-Origin-Opener-Policy` / `Cross-Origin-Embedder-Policy` headers (configured in `vite.config.ts`) for Transformers.js
- Sample media in `public/samples/`:
  - `video.mp4` — first video clip, ideally with an audio track
  - `video-2.mp4` — second video clip, ideally with an audio track
  - `overlay.png` — transparent PNG shown on the right side from 1s to 4s
  - `overlay-2.png` — transparent PNG shown on the left side from 1s to 4s
- MediaBunny dependency is currently resolved from `../MasterSelects/node_modules/mediabunny`; adjust `package.json` if you want to install it from npm or another local path

## Quick start

```bash
npm install
# copy your files:
#   public/samples/video.mp4
#   public/samples/video-2.mp4
#   public/samples/overlay.png
#   public/samples/overlay-2.png
npm run dev
```

Open http://localhost:5180. The app checks sample media, loads the preview player, then you can play and scrub the timeline.

## License

MIT (demo code)
