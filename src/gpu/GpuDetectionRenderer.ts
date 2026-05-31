import detectionShader from '../shaders/detection.wgsl?raw';
import {
  MAX_GPU_DETECTIONS,
  type GpuDetectionBox,
} from '../detection/gpuDetection';
import {createShaderModule} from './gpuShader';

// 16 bytes for the header, 32 bytes per box
const DETECTION_UNIFORM_SIZE = 16 + MAX_GPU_DETECTIONS * 32;

const ALPHA_BLEND_TARGET = {
  blend: {
    color: {
      operation: 'add' as const,
      srcFactor: 'src-alpha' as const,
      dstFactor: 'one-minus-src-alpha' as const,
    },
    alpha: {
      operation: 'add' as const,
      srcFactor: 'one' as const,
      dstFactor: 'one-minus-src-alpha' as const,
    },
  },
};

export class GpuDetectionRenderer {
  private readonly pipeline: GPURenderPipeline;
  private readonly bindGroupLayout: GPUBindGroupLayout;
  private readonly uniformBuffer: GPUBuffer;

  private constructor(
    private readonly device: GPUDevice,
    pipeline: GPURenderPipeline,
    bindGroupLayout: GPUBindGroupLayout,
    uniformBuffer: GPUBuffer,
  ) {
    this.pipeline = pipeline;
    this.bindGroupLayout = bindGroupLayout;
    this.uniformBuffer = uniformBuffer;
  }

  static async create(
    device: GPUDevice,
    canvasFormat: GPUTextureFormat,
  ): Promise<GpuDetectionRenderer> {
    const shaderModule = await createShaderModule(device, detectionShader, 'detection-shader');

    const bindGroupLayout = device.createBindGroupLayout({
      entries: [
        {binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: {type: 'uniform'}},
      ],
    });

    const pipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({bindGroupLayouts: [bindGroupLayout]}),
      vertex: {module: shaderModule, entryPoint: 'vertexMain'},
      fragment: {
        module: shaderModule,
        entryPoint: 'fragmentMain',
        targets: [{format: canvasFormat, ...ALPHA_BLEND_TARGET}],
      },
      primitive: {topology: 'triangle-list'},
    });

    const uniformBuffer = device.createBuffer({
      size: DETECTION_UNIFORM_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    return new GpuDetectionRenderer(device, pipeline, bindGroupLayout, uniformBuffer);
  }

  render(
    outputView: GPUTextureView,
    detections: GpuDetectionBox[],
    loadOp: GPULoadOp = 'load',
  ): void {
    const count = Math.min(detections.length, MAX_GPU_DETECTIONS);
    const uniformData = new ArrayBuffer(DETECTION_UNIFORM_SIZE);
    const view = new DataView(uniformData);
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

    this.device.queue.writeBuffer(this.uniformBuffer, 0, uniformData);

    const bindGroup = this.device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [{binding: 0, resource: {buffer: this.uniformBuffer}}],
    });

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: outputView,
          clearValue: {r: 0, g: 0, b: 0, a: 1},
          loadOp,
          storeOp: 'store',
        },
      ],
    });

    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(6);
    pass.end();

    this.device.queue.submit([encoder.finish()]);
  }

  destroy(): void {
    this.uniformBuffer.destroy();
  }
}
