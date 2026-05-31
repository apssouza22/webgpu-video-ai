import type {GpuDetectionBox} from '../detection/gpuDetection';
import type {ImageClip, VideoClip} from '../types';
import {GpuDetectionRenderer} from './GpuDetectionRenderer';
import {GpuImageRenderer} from './GpuImageRenderer';
import {GpuVideoRenderer} from './GpuVideoRenderer';

export interface VideoLayerInput {
  videoFrame: VideoFrame;
  videoClip: VideoClip;
}

export interface ImageLayerInput {
  image: HTMLImageElement;
  imageClip: ImageClip;
}

export interface CompositorFrameInput {
  time: number;
  videoLayers: VideoLayerInput[];
  imageLayers: ImageLayerInput[];
  detections?: GpuDetectionBox[];
}

export class GpuCompositor {
  private constructor(
    private readonly videoRenderer: GpuVideoRenderer,
    private readonly imageRenderer: GpuImageRenderer,
    private readonly detectionRenderer: GpuDetectionRenderer,
  ) {}

  static async create(
    device: GPUDevice,
    canvasFormat: GPUTextureFormat,
  ): Promise<GpuCompositor> {
    const [videoRenderer, imageRenderer, detectionRenderer] = await Promise.all([
      GpuVideoRenderer.create(device, canvasFormat),
      GpuImageRenderer.create(device, canvasFormat),
      GpuDetectionRenderer.create(device, canvasFormat),
    ]);

    return new GpuCompositor(videoRenderer, imageRenderer, detectionRenderer);
  }

  async renderFrame(
    canvasContext: GPUCanvasContext,
    input: CompositorFrameInput,
  ): Promise<void> {
    const {videoLayers, imageLayers, detections = []} = input;
    const outputView = canvasContext.getCurrentTexture().createView();

    for (const [index, videoInput] of videoLayers.entries()) {
      this.videoRenderer.render(outputView, videoInput, index === 0 ? 'clear' : 'load');
    }

    for (const {image, imageClip} of imageLayers) {
      this.imageRenderer.render(
        outputView,
        image,
        imageClip,
        videoLayers.length === 0 ? 'clear' : 'load',
      );
    }

    if (detections.length > 0) {
      this.detectionRenderer.render(outputView, detections);
    }
  }

  destroy(): void {
    this.videoRenderer.destroy();
    this.imageRenderer.destroy();
    this.detectionRenderer.destroy();
  }
}
