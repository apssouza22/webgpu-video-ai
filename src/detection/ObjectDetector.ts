import { pipeline, type ObjectDetectionOutput } from '@huggingface/transformers';

const MODEL_ID = 'onnx-community/rfdetr_medium-ONNX';

export type DetectionResult = ObjectDetectionOutput[number];

type DetectFn = (
  input: HTMLCanvasElement,
  options?: { threshold?: number; percentage?: boolean },
) => Promise<ObjectDetectionOutput>;

export class ObjectDetector {
  private constructor(private readonly detector: DetectFn) {}

  static async create(onStatus?: (message: string) => void): Promise<ObjectDetector> {
    onStatus?.('Loading RF-DETR model (WebGPU)…');
    const detector = (await pipeline('object-detection', MODEL_ID, {
      device: 'webgpu',
      dtype: 'fp32',
    })) as DetectFn;
    return new ObjectDetector(detector);
  }

  async warmup(canvas: HTMLCanvasElement, threshold = 0.5): Promise<void> {
    await this.detect(canvas, { threshold });
  }

  async detect(
    canvas: HTMLCanvasElement,
    options: { threshold: number },
  ): Promise<DetectionResult[]> {
    const results = await this.detector(canvas, {
      threshold: options.threshold,
      percentage: true,
    });
    return results as DetectionResult[];
  }
}
