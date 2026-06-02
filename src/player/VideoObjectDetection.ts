import type {GpuDetectionBox} from '../detection/gpuDetection';
import {toGpuDetections} from '../detection/gpuDetection';
import type {ObjectDetectorService} from '../detection/ObjectDetectorService';

export interface ObjectDetectionOptions {
  threshold?: number;
  enabled?: boolean;
  onFpsUpdate?: (fps: number) => void;
  onDetectionsUpdated?: () => void;
}

export class VideoObjectDetection {
  private detectionVersion = 0;
  private detectionBusy = false;
  private pendingDetectionFrame: VideoFrame | null = null;
  private enabled: boolean;
  private threshold: number;
  private detections: GpuDetectionBox[] = [];
  private lastDetectionAt = 0;
  private onDetectionsUpdated?: () => void;

  constructor(
    private readonly detector: ObjectDetectorService,
    private readonly options: ObjectDetectionOptions,
  ) {
    this.enabled = options.enabled ?? true;
    this.threshold = options.threshold ?? 0.5;
    this.onDetectionsUpdated = options.onDetectionsUpdated;
  }

  setEnabled(enabled: boolean): void {
    this.enabled = enabled;
    if (!enabled) {
      this.detections = [];
      this.onDetectionsUpdated?.();
    }
  }

  isEnabled(): boolean {
    return this.enabled;
  }

  setThreshold(threshold: number): void {
    this.threshold = threshold;
  }

  getThreshold(): number {
    return this.threshold;
  }

  getDetections(): GpuDetectionBox[] {
    return this.enabled ? this.detections : [];
  }

  async warmup(frame: VideoFrame, threshold = this.threshold): Promise<void> {
    await this.detector.warmup(frame, threshold);
  }

  schedule(videoFrame: VideoFrame): void {
    if (!this.enabled) {
      return;
    }

    if (this.detectionBusy) {
      this.pendingDetectionFrame?.close();
      this.pendingDetectionFrame = new VideoFrame(videoFrame);
      return;
    }

    void this.runDetection(new VideoFrame(videoFrame));
  }

  destroy(): void {
    this.pendingDetectionFrame?.close();
    this.pendingDetectionFrame = null;
  }

  private async runDetection(detectionFrame: VideoFrame): Promise<void> {
    const detectionVersion = ++this.detectionVersion;
    this.detectionBusy = true;
    const startedAt = performance.now();

    try {
      const results = await this.detector.detect(detectionFrame, {
        threshold: this.threshold,
      });

      if (detectionVersion !== this.detectionVersion) {
        return;
      }

      this.detections = toGpuDetections(results);

      const objectNames = [...new Set(results.map((r) => r.label))];
      if (objectNames.length > 0) {
        console.log('Detected objects:', objectNames);
      }

      this.onDetectionsUpdated?.();

      if (this.options.onFpsUpdate && this.lastDetectionAt > 0) {
        const elapsed = startedAt - this.lastDetectionAt;
        if (elapsed > 0) {
          this.options.onFpsUpdate(1000 / elapsed);
        }
      }
      this.lastDetectionAt = startedAt;
    } catch (error) {
      console.warn('Object detection failed', error);
    } finally {
      this.detectionBusy = false;

      if (this.pendingDetectionFrame) {
        const nextFrame = this.pendingDetectionFrame;
        this.pendingDetectionFrame = null;
        void this.runDetection(nextFrame);
      }
    }
  }
}
