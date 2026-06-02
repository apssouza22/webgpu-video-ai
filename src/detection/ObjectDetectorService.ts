import type {DetectionResult, WorkerRequest, WorkerResponse} from './workerMessages';

export type {DetectionResult} from './workerMessages';

type PendingRequest = {
  resolve: (results: DetectionResult[]) => void;
  reject: (error: Error) => void;
};

export class ObjectDetectorService {
  private readonly worker: Worker;
  private readonly pending = new Map<number, PendingRequest>();
  private nextId = 0;
  private readonly ready: Promise<void>;
  private readySettled = false;
  private resolveReady!: () => void;
  private rejectReady!: (error: Error) => void;
  private onStatus?: (message: string) => void;

  private constructor(worker: Worker, onStatus?: (message: string) => void) {
    this.worker = worker;
    this.onStatus = onStatus;

    this.ready = new Promise<void>((resolve, reject) => {
      this.resolveReady = resolve;
      this.rejectReady = reject;
    });

    this.worker.addEventListener('message', (event: MessageEvent<WorkerResponse>) => {
      this.handleMessage(event.data);
    });
    this.worker.addEventListener('error', (event) => {
      this.failReady(new Error(event.message || 'Detection worker failed'));
    });

    this.worker.postMessage({type: 'init'} satisfies WorkerRequest);
  }

  static async create(onStatus?: (message: string) => void): Promise<ObjectDetectorService> {
    const worker = new Worker(new URL('./detection.worker.ts', import.meta.url), {
      type: 'module',
    });
    const detector = new ObjectDetectorService(worker, onStatus);
    await detector.ready;
    return detector;
  }

  async warmup(frame: VideoFrame, threshold = 0.5): Promise<void> {
    await this.detect(frame, {threshold});
  }

  async detect(
    frame: VideoFrame,
    options: {threshold: number},
  ): Promise<DetectionResult[]> {
    await this.ready;

    const id = this.nextId++;

    return new Promise<DetectionResult[]>((resolve, reject) => {
      this.pending.set(id, {resolve, reject});
      this.worker.postMessage(
        {type: 'detect', id, threshold: options.threshold, frame} satisfies WorkerRequest,
        [frame],
      );
    });
  }

  dispose(): void {
    for (const {reject} of this.pending.values()) {
      reject(new Error('Detection worker terminated'));
    }
    this.pending.clear();
    this.worker.terminate();
  }

  private handleMessage(message: WorkerResponse): void {
    if (message.type === 'status') {
      this.onStatus?.(message.message);
      return;
    }

    if (message.type === 'ready') {
      if (!this.readySettled) {
        this.readySettled = true;
        this.resolveReady();
      }
      return;
    }

    if (message.type === 'detect-result') {
      const pending = this.pending.get(message.id);
      if (pending) {
        this.pending.delete(message.id);
        pending.resolve(message.results);
      }
      return;
    }

    if (message.type === 'error') {
      const error = new Error(message.error);

      if (message.id !== undefined) {
        const pending = this.pending.get(message.id);
        if (pending) {
          this.pending.delete(message.id);
          pending.reject(error);
        }
        return;
      }

      this.failReady(error);
    }
  }

  private failReady(error: Error): void {
    if (!this.readySettled) {
      this.readySettled = true;
      this.rejectReady(error);
    }
  }
}
