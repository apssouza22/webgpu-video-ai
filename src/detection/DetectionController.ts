import type { ObjectDetector } from './ObjectDetector';
import { DetectionOverlay } from './DetectionOverlay';

export interface DetectionControllerOptions {
  detector: ObjectDetector;
  threshold?: number;
  onFpsUpdate?: (fps: number) => void;
}

export class DetectionController {
  private readonly detector: ObjectDetector;
  private readonly overlay = new DetectionOverlay();
  private readonly inputCanvas = document.createElement('canvas');
  private readonly inputCtx: CanvasRenderingContext2D;
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

    const ctx = this.inputCanvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) {
      throw new Error('Failed to create detection input canvas');
    }
    this.inputCtx = ctx;
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
      this.copyFrame(sourceCanvas);

      const startedAt = performance.now();
      let results = await this.detector.detect(this.inputCanvas, { threshold: this.threshold });

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

  private copyFrame(sourceCanvas: HTMLCanvasElement): void {
    this.inputCanvas.width = sourceCanvas.width;
    this.inputCanvas.height = sourceCanvas.height;
    this.inputCtx.drawImage(sourceCanvas, 0, 0);
  }
}
