export async function createShaderModule(
  device: GPUDevice,
  code: string,
  label: string,
): Promise<GPUShaderModule> {
  const module = device.createShaderModule({code, label});

  if (module.getCompilationInfo) {
    const info = await module.getCompilationInfo();
    for (const message of info.messages) {
      if (message.type === 'error') {
        throw new Error(`WGSL (${label}): ${message.message}`);
      }
    }
  }

  return module;
}
