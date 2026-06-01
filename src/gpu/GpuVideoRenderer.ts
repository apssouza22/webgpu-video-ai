import videoShader from '../shaders/video.wgsl?raw';
import {AbstractGpuRenderer, type FragmentTargetOptions} from './AbstractGpuRenderer';
import type {VideoLayerInput} from './GpuCompositor';

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

export class GpuVideoRenderer extends AbstractGpuRenderer {
  private readonly uniformBuffer: GPUBuffer;

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
  ): Promise<GpuVideoRenderer> {
    const shaderModule = await AbstractGpuRenderer.loadShader(device, videoShader, 'video-shader');

    const bindGroupLayout = device.createBindGroupLayout({
      entries: [
        {binding: 0, visibility: GPUShaderStage.FRAGMENT, sampler: {}},
        {binding: 1, visibility: GPUShaderStage.FRAGMENT, externalTexture: {}},
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

    return new GpuVideoRenderer(
      device,
      pipeline,
      AbstractGpuRenderer.createLinearSampler(device),
      bindGroupLayout,
      uniformBuffer,
    );
  }

  render(
    outputView: GPUTextureView,
    videoLayerInput: VideoLayerInput,
    loadOp: GPULoadOp = 'clear',
  ): void {
    const videoClip = videoLayerInput.videoClip;
    const videoFrame = videoLayerInput.videoFrame;
    const uniformData = new Float32Array([
      1,
      videoClip.x,
      videoClip.y,
      videoClip.x + videoClip.width,
      videoClip.y + videoClip.height,
    ]);
    this.device.queue.writeBuffer(this.uniformBuffer, 0, uniformData);

    const videoTexture = this.device.importExternalTexture({source: videoFrame});
    const bindGroup = this.device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        {binding: 0, resource: this.sampler},
        {binding: 1, resource: videoTexture},
        {binding: 2, resource: {buffer: this.uniformBuffer}},
      ],
    });

    this.drawPass(outputView, bindGroup, loadOp);
  }

  destroy(): void {
    this.uniformBuffer.destroy();
  }
}
