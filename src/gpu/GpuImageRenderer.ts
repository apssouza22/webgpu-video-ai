import imageShader from '../shaders/image.wgsl?raw';
import type {ImageClip} from '../types';
import {AbstractGpuRenderer, type FragmentTargetOptions} from './AbstractGpuRenderer';

const TEXTURE_USAGE =
  GPUTextureUsage.TEXTURE_BINDING |
  GPUTextureUsage.COPY_DST |
  GPUTextureUsage.RENDER_ATTACHMENT;

const UNIFORM_SIZE = 20;

const ALPHA_BLEND_TARGET: FragmentTargetOptions = {
  blend: {
    color: {
      operation: 'add',
      srcFactor: 'src-alpha',
      dstFactor: 'one-minus-src-alpha',
    },
    alpha: {
      operation: 'add',
      srcFactor: 'one',
      dstFactor: 'one-minus-src-alpha',
    },
  },
};

export class GpuImageRenderer extends AbstractGpuRenderer {
  private readonly uniformBuffer: GPUBuffer;
  private readonly imageTextureMap = new Map<HTMLImageElement, GPUTexture>();

  private constructor(
    device: GPUDevice,
    pipeline: GPURenderPipeline,
    sampler: GPUSampler,
    bindGroupLayout: GPUBindGroupLayout,
    uniformBuffer: GPUBuffer,
  ) {
    super(device, pipeline, sampler, bindGroupLayout);
    this.uniformBuffer = uniformBuffer;
  }

  static async create(
    device: GPUDevice,
    canvasFormat: GPUTextureFormat,
  ): Promise<GpuImageRenderer> {
    const shaderModule = await AbstractGpuRenderer.loadShader(device, imageShader, 'image-shader');

    const bindGroupLayout = device.createBindGroupLayout({
      entries: [
        {binding: 0, visibility: GPUShaderStage.FRAGMENT, sampler: {}},
        {binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: {}},
        {binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: {type: 'uniform'}},
      ],
    });

    const pipeline = AbstractGpuRenderer.createPipeline(
      device,
      shaderModule,
      bindGroupLayout,
      canvasFormat,
      [ALPHA_BLEND_TARGET],
    );

    const uniformBuffer = device.createBuffer({
      size: UNIFORM_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    return new GpuImageRenderer(
      device,
      pipeline,
      AbstractGpuRenderer.createLinearSampler(device),
      bindGroupLayout,
      uniformBuffer,
    );
  }

  render(
    outputView: GPUTextureView,
    image: HTMLImageElement,
    imageClip: ImageClip,
    loadOp: GPULoadOp = 'load',
  ): boolean {
    const texture = this.ensureImageTexture(image);
    if (!texture) {
      return false;
    }

    const uniformData = new Float32Array([
      imageClip.opacity,
      imageClip.x,
      imageClip.y,
      imageClip.x + imageClip.width,
      imageClip.y + imageClip.height,
    ]);
    this.device.queue.writeBuffer(this.uniformBuffer, 0, uniformData);

    const bindGroup = this.device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        {binding: 0, resource: this.sampler},
        {binding: 1, resource: texture.createView()},
        {binding: 2, resource: {buffer: this.uniformBuffer}},
      ],
    });

    this.drawPass(outputView, bindGroup, loadOp);
    return true;
  }

  private ensureImageTexture(image: HTMLImageElement): GPUTexture | null {
    const width = image.naturalWidth || image.width;
    const height = image.naturalHeight || image.height;
    if (width === 0 || height === 0) {
      return null;
    }

    const existingTexture = this.imageTextureMap.get(image);
    if (
      existingTexture &&
      existingTexture.width === width &&
      existingTexture.height === height
    ) {
      return existingTexture;
    }

    existingTexture?.destroy();
    const texture = this.device.createTexture({
      size: {width, height},
      format: 'rgba8unorm',
      usage: TEXTURE_USAGE,
    });

    this.device.queue.copyExternalImageToTexture(
      {source: image},
      {texture},
      {width, height},
    );
    this.imageTextureMap.set(image, texture);

    return texture;
  }

  destroy(): void {
    for (const texture of this.imageTextureMap.values()) {
      texture.destroy();
    }
    this.imageTextureMap.clear();
    this.uniformBuffer.destroy();
  }
}
