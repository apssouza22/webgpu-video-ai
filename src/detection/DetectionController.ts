import type {ObjectDetector} from './ObjectDetector';
import {DetectionOverlay} from './DetectionOverlay';

export interface DetectionControllerOptions {
  detector: ObjectDetector;
  threshold?: number;
  onFpsUpdate?: (fps: number) => void;
}

export class DetectionController {
  private readonly detector: ObjectDetector;
  private readonly overlay = new DetectionOverlay();
  private enabled = true;
  private threshold: number;
  private allowedLabels: Set<string> | null = null;
  private busy = false;
  private frameVersion = 0;
  private lastDetectionAt = 0;
  private readonly onFpsUpdate?: (fps: number) => void;

  constructor(options: DetectionControllerOptions) {
    this.detector = options.detector;
    this.threshold = options.threshold ?? 0.5;
    this.onFpsUpdate = options.onFpsUpdate;
  }

  get overlayCanvas(): HTMLCanvasElement {
    return this.overlay.canvas;
  }

  setEnabled(enabled: boolean): void {
    this.enabled = enabled;
    if (!enabled) {
      this.overlay.clear();
    }
  }

  isEnabled(): boolean {
    return this.enabled;
  }

  getThreshold(): number {
    return this.threshold;
  }

  setThreshold(threshold: number): void {
    this.threshold = threshold;
  }

  setAllowedLabels(value: string): void {
    const trimmed = value.trim();
    this.allowedLabels = trimmed
      ? new Set(trimmed.split(',').map((label) => label.trim().toLowerCase()).filter(Boolean))
      : null;
  }

  resize(sourceCanvas: HTMLCanvasElement): void {
    this.overlay.resizeToMatch(sourceCanvas);
  }

  async processFrame(sourceCanvas: HTMLCanvasElement): Promise<void> {
    if (!this.enabled || this.busy) {
      return;
    }

    const version = ++this.frameVersion;
    this.busy = true;

    try {
      this.resize(sourceCanvas);

      const startedAt = performance.now();
      let results = await this.detector.detect(sourceCanvas, {threshold: this.threshold});

      if (this.allowedLabels) {
        results = results.filter((result) =>
          this.allowedLabels!.has(result.label.toLowerCase()),
        );
      }

      if (version !== this.frameVersion) {
        return;
      }

      const displayWidth = sourceCanvas.clientWidth;
      const displayHeight = sourceCanvas.clientHeight;
      this.overlay.draw(results, displayWidth, displayHeight);

      if (this.onFpsUpdate && this.lastDetectionAt > 0) {
        const elapsed = startedAt - this.lastDetectionAt;
        if (elapsed > 0) {
          this.onFpsUpdate(1000 / elapsed);
        }
      }
      this.lastDetectionAt = startedAt;
    } finally {
      this.busy = false;
    }
  }
}
