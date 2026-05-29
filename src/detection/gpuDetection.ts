import {colorForLabel} from './detectionColors';
import type {DetectionResult} from './workerMessages';

export interface GpuDetectionBox {
  xmin: number;
  ymin: number;
  xmax: number;
  ymax: number;
  r: number;
  g: number;
  b: number;
}

export const MAX_GPU_DETECTIONS = 32;

export function toGpuDetections(results: DetectionResult[]): GpuDetectionBox[] {
  return results.slice(0, MAX_GPU_DETECTIONS).map((result) => {
    const [r, g, b] = colorForLabel(result.label);
    return {
      xmin: result.box.xmin,
      ymin: result.box.ymin,
      xmax: result.box.xmax,
      ymax: result.box.ymax,
      r,
      g,
      b,
    };
  });
}
