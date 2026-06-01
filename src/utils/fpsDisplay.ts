const DEFAULT_ALPHA = 0.12;
const DEFAULT_UPDATE_INTERVAL_MS = 500;

function smoothSample(previous: number | null, sample: number, alpha: number): number {
  return previous === null ? sample : alpha * sample + (1 - alpha) * previous;
}

export type FpsDisplayValues = {
  render: number | null;
  detection: number | null;
};

export type FpsDisplaySink = {
  addRenderSample(instantFps: number): void;
  addDetectionSample(instantFps: number): void;
};

export function createFpsDisplay(
  onUpdate: (values: FpsDisplayValues) => void,
  options?: {alpha?: number; updateIntervalMs?: number},
): FpsDisplaySink {
  const alpha = options?.alpha ?? DEFAULT_ALPHA;
  const updateIntervalMs = options?.updateIntervalMs ?? DEFAULT_UPDATE_INTERVAL_MS;

  let render: number | null = null;
  let detection: number | null = null;
  let lastPaintAt = 0;
  let timer: ReturnType<typeof setTimeout> | null = null;

  const paint = (): void => {
    timer = null;
    lastPaintAt = performance.now();
    onUpdate({render, detection});
  };

  const schedulePaint = (): void => {
    const now = performance.now();
    if (lastPaintAt === 0 || now - lastPaintAt >= updateIntervalMs) {
      if (timer !== null) {
        clearTimeout(timer);
        timer = null;
      }
      paint();
      return;
    }

    if (timer !== null) {
      return;
    }

    timer = setTimeout(paint, updateIntervalMs - (now - lastPaintAt));
  };

  return {
    addRenderSample(instantFps) {
      render = smoothSample(render, instantFps, alpha);
      schedulePaint();
    },
    addDetectionSample(instantFps) {
      detection = smoothSample(detection, instantFps, alpha);
      schedulePaint();
    },
  };
}
