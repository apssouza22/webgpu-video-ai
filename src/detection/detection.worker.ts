/// <reference lib="webworker" />

import {pipeline, type ObjectDetectionOutput} from '@huggingface/transformers';
import type {WorkerRequest, WorkerResponse} from './workerMessages';

const MODEL_ID = 'onnx-community/rfdetr_medium-ONNX';

type DetectFn = (
  input: OffscreenCanvas,
  options?: {threshold?: number; percentage?: boolean},
) => Promise<ObjectDetectionOutput>;

let detector: DetectFn | null = null;
let loadPromise: Promise<void> | null = null;

function post(message: WorkerResponse): void {
  self.postMessage(message);
}

async function loadDetector(): Promise<DetectFn> {
  if (detector) {
    return detector;
  }

  if (!loadPromise) {
    loadPromise = (async () => {
      post({type: 'status', message: 'Loading RF-DETR model (WebGPU)…'});
      detector = (await pipeline('object-detection', MODEL_ID, {
        device: 'webgpu',
        dtype: 'fp32',
        progress_callback: (progress) => {
          if (progress.status === 'progress' && progress.file) {
            const percent =
              progress.total && progress.total > 0
                ? Math.round((progress.loaded / progress.total) * 100)
                : null;
            post({
              type: 'status',
              message: percent
                ? `Downloading ${progress.file} (${percent}%)…`
                : `Downloading ${progress.file}…`,
            });
          }
        },
      })) as DetectFn;
      post({type: 'status', message: 'Compiling RF-DETR shaders…'});
    })();
  }

  await loadPromise;
  return detector!;
}

async function runDetection(
  frame: VideoFrame,
  threshold: number,
): Promise<ObjectDetectionOutput> {
  const detect = await loadDetector();
  const width = frame.displayWidth;
  const height = frame.displayHeight;
  const canvas = new OffscreenCanvas(width, height);
  const ctx = canvas.getContext('2d', {willReadFrequently: true});
  if (!ctx) {
    frame.close();
    throw new Error('OffscreenCanvas 2D context not available in worker');
  }

  try {
    ctx.drawImage(frame, 0, 0, width, height);
  } finally {
    frame.close();
  }

  return detect(canvas, {threshold, percentage: true});
}

self.onmessage = async (event: MessageEvent<WorkerRequest>) => {
  const message = event.data;

  try {
    if (message.type === 'init') {
      await loadDetector();
      post({type: 'ready'});
      return;
    }

    if (message.type === 'detect') {
      const results = await runDetection(message.frame, message.threshold);
      post({type: 'detect-result', id: message.id, results});
      return;
    }
  } catch (error) {
    if (message.type === 'detect') {
      message.frame.close();
    }

    post({
      type: 'error',
      error: error instanceof Error ? error.message : String(error),
      id: message.type === 'detect' ? message.id : undefined,
    });
  }
};
