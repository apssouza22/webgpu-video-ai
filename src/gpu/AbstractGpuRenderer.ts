import {createShaderModule} from './gpuShader';

export type FragmentTargetOptions = Omit<GPUColorTargetState, 'format'>;

export abstract class AbstractGpuRenderer {
  protected constructor(
    protected readonly device: GPUDevice,
    protected readonly pipeline: GPURenderPipeline,
    protected readonly sampler: GPUSampler,
    protected readonly bindGroupLayout: GPUBindGroupLayout,
  ) {}

  protected static async loadShader(
    device: GPUDevice,
    code: string,
    label: string,
  ): Promise<GPUShaderModule> {
    return createShaderModule(device, code, label);
  }

  protected static createLinearSampler(device: GPUDevice): GPUSampler {
    return device.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
    });
  }

  protected static createPipeline(
    device: GPUDevice,
    shaderModule: GPUShaderModule,
    bindGroupLayout: GPUBindGroupLayout,
    canvasFormat: GPUTextureFormat,
    fragmentTargets: FragmentTargetOptions[],
  ): GPURenderPipeline {
    return device.createRenderPipeline({
      layout: device.createPipelineLayout({bindGroupLayouts: [bindGroupLayout]}),
      vertex: {module: shaderModule, entryPoint: 'vertexMain'},
      fragment: {
        module: shaderModule,
        entryPoint: 'fragmentMain',
        targets: fragmentTargets.map((target) => ({
          ...target,
          format: canvasFormat,
        })),
      },
      primitive: {topology: 'triangle-list'},
    });
  }

  protected drawPass(
    outputView: GPUTextureView,
    bindGroup: GPUBindGroup,
    loadOp: GPULoadOp,
  ): void {
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
}
