import type {Composition} from '../composition';
import {GpuCompositor, type VideoLayerInput} from '../gpu/GpuCompositor';
import {PlayerCanvas} from '../gpu/PlayerCanvas';
import type {DecodedVideoFrame} from '../media/VideoFrameSource';
import type {ImageClip, VideoFrameContext, VideoLayerClip} from '../types';
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
    const decodedVideos = await this.decodeVideoLayers(frameContext);
    if (decodedVideos.length === 0) {
      return;
    }

    try {
      const detectionFrame = new VideoFrame(decodedVideos[0].sourceFrame.frame);
      await this.detection.warmup(detectionFrame, this.detection.getThreshold());
    } finally {
      this.closeDecodedVideos(decodedVideos);
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
    const decodedVideos = await this.decodeVideoLayers(frameContext);
    const imageLayers = await Promise.all(
      this.currentImageLayers(renderTime).map(async (imageClip) => ({
        image: await imageClip.loadImageElement(),
        imageClip,
      })),
    );

    if (decodedVideos.length === 0 && imageLayers.length === 0) {
      return;
    }

    try {
      if (renderVersion !== this.renderVersion) {
        return;
      }

      if (!options.skipDetection && decodedVideos.length > 0) {
        this.detection?.schedule(decodedVideos[0].sourceFrame.frame);
      }

      await this.gpuCompositor.renderFrame(this.playerCanvas.getContext(), {
        time: renderTime,
        videoLayers: this.toVideoLayerInputs(decodedVideos),
        imageLayers,
        detections: this.detection?.getDetections() ?? [],
      });
    } finally {
      this.closeDecodedVideos(decodedVideos);
    }
  }

  destroy(): void {
    this.gpuCompositor.destroy();
    this.playerCanvas.destroy();
  }

  private currentImageLayers(time: number): readonly ImageClip[] {
    return this.imageLayers.filter((clip) => clip.containsTime(time));
  }

  private decodeVideoLayers(
    frameContext: VideoFrameContext,
  ): Promise<Array<{videoLayer: VideoLayerClip; sourceFrame: DecodedVideoFrame}>> {
    return Promise.all(
      frameContext.videos.map(async (videoLayer) => ({
        videoLayer,
        sourceFrame: await videoLayer.clip.nextSourceFrame(
          videoLayer.sourceTime,
          frameContext.frame,
        ),
      })),
    );
  }

  private toVideoLayerInputs(
    decodedVideos: Array<{videoLayer: VideoLayerClip; sourceFrame: DecodedVideoFrame}>,
  ): VideoLayerInput[] {
    return decodedVideos.map(({videoLayer, sourceFrame}) => ({
      videoFrame: sourceFrame.frame,
      videoClip: videoLayer.clip,
    }));
  }

  private closeDecodedVideos(
    decodedVideos: Array<{sourceFrame: DecodedVideoFrame}>,
  ): void {
    for (const {sourceFrame} of decodedVideos) {
      sourceFrame.close();
    }
  }
}
