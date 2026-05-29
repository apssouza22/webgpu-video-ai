import type {ObjectDetectionOutput} from '@huggingface/transformers';

export type DetectionResult = ObjectDetectionOutput[number];

export type WorkerRequest =
  | {type: 'init'}
  | {type: 'detect'; id: number; threshold: number; bitmap: ImageBitmap};

export type WorkerResponse =
  | {type: 'status'; message: string}
  | {type: 'ready'}
  | {type: 'detect-result'; id: number; results: DetectionResult[]}
  | {type: 'error'; error: string; id?: number};
