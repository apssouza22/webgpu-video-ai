import type {Composition} from '../composition';
import {GpuCompositor} from '../gpu/GpuCompositor';
import {PlayerCanvas} from '../gpu/PlayerCanvas';
import type {ImageClip} from '../types';
import type {VideoObjectDetection} from './VideoObjectDetection';

export class VideoPlayer {
  private readonly playerCanvas: PlayerCanvas;
  private readonly gpuCompositor: GpuCompositor;
  private readonly imageLayers: readonly ImageClip[];
  private readonly detection: VideoObjectDetection;
  private renderVersion = 0;

  static async create(
    composition: Composition,
    detection: VideoObjectDetection,
  ): Promise<VideoPlayer> {
    if (!navigator.gpu) {
      throw new Error('WebGPU is not available');
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      throw new Error('Failed to acquire GPU adapter');
    }

    const device = await adapter.requestDevice();
    const playerCanvas = new PlayerCanvas();
    playerCanvas.init(device, composition.width, composition.height);
    const compositor = await GpuCompositor.create(device, playerCanvas.getFormat());

    return new VideoPlayer(composition, playerCanvas, compositor, detection);
  }

  private constructor(
    private readonly composition: Composition,
    playerCanvas: PlayerCanvas,
    compositor: GpuCompositor,
    detection: VideoObjectDetection,
  ) {
    this.playerCanvas = playerCanvas;
    this.gpuCompositor = compositor;
    this.imageLayers = composition.imageLayers;
    this.detection = detection;
  }

  getCanvas(): HTMLCanvasElement {
    return this.playerCanvas.getCanvas();
  }

  async warmupDetection(time = 0, duration?: number): Promise<void> {
    if (!this.detection) {
      return;
    }

    const renderDuration = duration ?? this.composition.duration;
    const frameContext = this.composition.getFrameContextAtTime(time);
    const videoLayer = frameContext.videos[0];
    if (!videoLayer) {
      return;
    }

    const sourceFrame = await videoLayer.clip.nextSourceFrame(
      videoLayer.sourceTime,
      frameContext.frame,
    );

    try {
      const detectionFrame = new VideoFrame(sourceFrame.frame);
      await this.detection.warmup(detectionFrame, this.detection.getThreshold());
    } finally {
      sourceFrame.close();
    }

    await this.render(time, renderDuration);
  }

  async render(
    time: number,
    duration: number,
    options: {skipDetection?: boolean} = {},
  ): Promise<void> {
    const renderVersion = ++this.renderVersion;
    const renderTime = Math.min(time, Math.max(0, duration - 0.001));
    const frameContext = this.composition.getFrameContextAtTime(renderTime);
    const videoLayer = frameContext.videos[0];
    if (!videoLayer) {
      return;
    }

    const sourceFrame = await videoLayer.clip.nextSourceFrame(
      videoLayer.sourceTime,
      frameContext.frame,
    );
    const overlays = await Promise.all(
      this.currentImageLayers(renderTime).map(async (imageClip) => ({
        image: await imageClip.loadImageElement(),
        imageClip,
      })),
    );

    try {
      if (renderVersion !== this.renderVersion) {
        return;
      }

      const videoFrame = sourceFrame.frame;
      if (!options.skipDetection) {
        this.detection?.schedule(videoFrame);
      }

      await this.gpuCompositor.renderFrame(this.playerCanvas.getContext(), {
        time: renderTime,
        videoFrame,
        overlays,
        detections: this.detection?.getDetections() ?? [],
      });
    } finally {
      sourceFrame.close();
    }
  }

  destroy(): void {
    this.gpuCompositor.destroy();
    this.playerCanvas.destroy();
  }

  private currentImageLayers(time: number): readonly ImageClip[] {
    return this.imageLayers.filter((clip) => clip.containsTime(time));
  }
}
