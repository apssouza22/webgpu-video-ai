import compositeShader from '../shaders/composite.wgsl?raw';
import detectionShader from '../shaders/detection.wgsl?raw';
import {
  MAX_GPU_DETECTIONS,
  type GpuDetectionBox,
} from '../detection/gpuDetection';
import type {ImageClip} from '../types';

const TEXTURE_USAGE =
  GPUTextureUsage.TEXTURE_BINDING |
  GPUTextureUsage.COPY_DST |
  GPUTextureUsage.RENDER_ATTACHMENT;

const DETECTION_UNIFORM_SIZE = 16 + MAX_GPU_DETECTIONS * 32;

export interface CompositorFrameInput {
  time: number;
  videoFrame: VideoFrame;
  overlays: CompositorOverlayInput[];
  detections?: GpuDetectionBox[];
}

export interface CompositorOverlayInput {
  image: HTMLImageElement;
  imageClip: ImageClip;
}

export class GpuCompositor {
  private device: GPUDevice;
  private compositePipeline: GPURenderPipeline;
  private detectionPipeline: GPURenderPipeline;
  private sampler: GPUSampler;
  private compositeUniformBuffer: GPUBuffer;
  private detectionUniformBuffer: GPUBuffer;
  private compositeBindGroupLayout: GPUBindGroupLayout;
  private detectionBindGroupLayout: GPUBindGroupLayout;
  private dummyOverlayTexture: GPUTexture;
  private overlayTextures = new Map<HTMLImageElement, GPUTexture>();

  private constructor(
    device: GPUDevice,
    compositePipeline: GPURenderPipeline,
    detectionPipeline: GPURenderPipeline,
    sampler: GPUSampler,
    compositeUniformBuffer: GPUBuffer,
    detectionUniformBuffer: GPUBuffer,
    compositeBindGroupLayout: GPUBindGroupLayout,
    detectionBindGroupLayout: GPUBindGroupLayout,
    dummyOverlayTexture: GPUTexture,
  ) {
    this.device = device;
    this.compositePipeline = compositePipeline;
    this.detectionPipeline = detectionPipeline;
    this.sampler = sampler;
    this.compositeUniformBuffer = compositeUniformBuffer;
    this.detectionUniformBuffer = detectionUniformBuffer;
    this.compositeBindGroupLayout = compositeBindGroupLayout;
    this.detectionBindGroupLayout = detectionBindGroupLayout;
    this.dummyOverlayTexture = dummyOverlayTexture;
  }

  static async create(
    device: GPUDevice,
    canvasFormat: GPUTextureFormat,
  ): Promise<GpuCompositor> {
    const compositeModule = await compileShader(device, compositeShader, 'composite-shader');
    const detectionModule = await compileShader(device, detectionShader, 'detection-shader');

    const compositeBindGroupLayout = device.createBindGroupLayout({
      entries: [
        {binding: 0, visibility: GPUShaderStage.FRAGMENT, sampler: {}},
        {binding: 1, visibility: GPUShaderStage.FRAGMENT, externalTexture: {}},
        {binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: {}},
        {binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: {type: 'uniform'}},
      ],
    });

    const detectionBindGroupLayout = device.createBindGroupLayout({
      entries: [
        {binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: {type: 'uniform'}},
      ],
    });

    const compositePipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({bindGroupLayouts: [compositeBindGroupLayout]}),
      vertex: {module: compositeModule, entryPoint: 'vertexMain'},
      fragment: {
        module: compositeModule,
        entryPoint: 'fragmentMain',
        targets: [
          {
            format: canvasFormat,
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
          },
        ],
      },
      primitive: {topology: 'triangle-list'},
    });

    const detectionPipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({bindGroupLayouts: [detectionBindGroupLayout]}),
      vertex: {module: detectionModule, entryPoint: 'vertexMain'},
      fragment: {
        module: detectionModule,
        entryPoint: 'fragmentMain',
        targets: [
          {
            format: canvasFormat,
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
          },
        ],
      },
      primitive: {topology: 'triangle-list'},
    });

    const sampler = device.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
    });

    const compositeUniformBuffer = device.createBuffer({
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const detectionUniformBuffer = device.createBuffer({
      size: DETECTION_UNIFORM_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const dummyOverlayTexture = device.createTexture({
      size: {width: 1, height: 1},
      format: 'rgba8unorm',
      usage: TEXTURE_USAGE,
    });

    return new GpuCompositor(
      device,
      compositePipeline,
      detectionPipeline,
      sampler,
      compositeUniformBuffer,
      detectionUniformBuffer,
      compositeBindGroupLayout,
      detectionBindGroupLayout,
      dummyOverlayTexture,
    );
  }

  private ensureOverlayTexture(image: HTMLImageElement): GPUTexture | null {
    const width = image.naturalWidth || image.width;
    const height = image.naturalHeight || image.height;
    if (width === 0 || height === 0) {
      return null;
    }

    const existingTexture = this.overlayTextures.get(image);
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
    this.overlayTextures.set(image, texture);

    return texture;
  }

  async renderFrame(
    canvasContext: GPUCanvasContext,
    input: CompositorFrameInput,
  ): Promise<void> {
    const {videoFrame, overlays, detections = []} = input;
    const externalVideoTexture = this.device.importExternalTexture({source: videoFrame});
    const textureView = canvasContext.getCurrentTexture().createView();
    const encoder = this.device.createCommandEncoder();

    const baseUniforms = new Float32Array([0, 0, 0, 0, 0, 0, 0, 0]);
    this.encodeCompositePass(
      encoder,
      textureView,
      externalVideoTexture,
      this.dummyOverlayTexture,
      baseUniforms,
      'clear',
    );

    for (const overlay of overlays) {
      const overlayTexture = this.ensureOverlayTexture(overlay.image);
      if (!overlayTexture) {
        continue;
      }

      const {imageClip} = overlay;
      const overlayUniforms = new Float32Array([
        imageClip.opacity,
        imageClip.x,
        imageClip.y,
        imageClip.x + imageClip.width,
        imageClip.y + imageClip.height,
        1,
        1,
        0,
      ]);
      this.encodeCompositePass(
        encoder,
        textureView,
        externalVideoTexture,
        overlayTexture,
        overlayUniforms,
        'load',
      );
    }

    if (detections.length > 0) {
      this.encodeDetectionPass(encoder, textureView, detections);
    }

    this.device.queue.submit([encoder.finish()]);
  }

  private encodeCompositePass(
    encoder: GPUCommandEncoder,
    textureView: GPUTextureView,
    externalVideoTexture: GPUExternalTexture,
    overlayTexture: GPUTexture,
    uniformData: Float32Array,
    loadOp: GPULoadOp,
  ): void {
    this.device.queue.writeBuffer(
      this.compositeUniformBuffer,
      0,
      uniformData.buffer,
      uniformData.byteOffset,
      uniformData.byteLength,
    );

    const bindGroup = this.device.createBindGroup({
      layout: this.compositeBindGroupLayout,
      entries: [
        {binding: 0, resource: this.sampler},
        {binding: 1, resource: externalVideoTexture},
        {binding: 2, resource: overlayTexture.createView()},
        {binding: 3, resource: {buffer: this.compositeUniformBuffer}},
      ],
    });

    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: textureView,
          clearValue: {r: 0, g: 0, b: 0, a: 1},
          loadOp,
          storeOp: 'store',
        },
      ],
    });

    pass.setPipeline(this.compositePipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(6);
    pass.end();
  }

  private encodeDetectionPass(
    encoder: GPUCommandEncoder,
    textureView: GPUTextureView,
    detections: GpuDetectionBox[],
  ): void {
    const uniformData = new ArrayBuffer(DETECTION_UNIFORM_SIZE);
    const view = new DataView(uniformData);
    const count = Math.min(detections.length, MAX_GPU_DETECTIONS);
    view.setUint32(0, count, true);
    view.setFloat32(4, 0.0025, true);

    const floats = new Float32Array(uniformData, 16);
    for (let i = 0; i < count; i++) {
      const box = detections[i];
      const offset = i * 8;
      floats[offset] = box.xmin;
      floats[offset + 1] = box.ymin;
      floats[offset + 2] = box.xmax;
      floats[offset + 3] = box.ymax;
      floats[offset + 4] = box.r;
      floats[offset + 5] = box.g;
      floats[offset + 6] = box.b;
      floats[offset + 7] = 1;
    }

    this.device.queue.writeBuffer(this.detectionUniformBuffer, 0, uniformData);

    const bindGroup = this.device.createBindGroup({
      layout: this.detectionBindGroupLayout,
      entries: [{binding: 0, resource: {buffer: this.detectionUniformBuffer}}],
    });

    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: textureView,
          loadOp: 'load',
          storeOp: 'store',
        },
      ],
    });

    pass.setPipeline(this.detectionPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(6);
    pass.end();
  }

  destroy(): void {
    for (const texture of this.overlayTextures.values()) {
      texture.destroy();
    }
    this.overlayTextures.clear();
    this.dummyOverlayTexture.destroy();
    this.compositeUniformBuffer.destroy();
    this.detectionUniformBuffer.destroy();
  }
}

async function compileShader(
  device: GPUDevice,
  code: string,
  label: string,
): Promise<GPUShaderModule> {
  const shaderModule = device.createShaderModule({code, label});

  if (shaderModule.getCompilationInfo) {
    const info = await shaderModule.getCompilationInfo();
    for (const message of info.messages) {
      if (message.type === 'error') {
        throw new Error(`WGSL (${label}): ${message.message}`);
      }
    }
  }

  return shaderModule;
}
