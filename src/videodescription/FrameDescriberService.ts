import type {WorkerRequest, WorkerResponse} from './workerMessages';

type PendingRequest = {
  resolve: (description: string) => void;
  reject: (error: Error) => void;
};

export class FrameDescriberService {
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
      this.failReady(new Error(event.message || 'Video description worker failed'));
    });

    this.worker.postMessage({type: 'init'} satisfies WorkerRequest);
  }

  static async create(onStatus?: (message: string) => void): Promise<FrameDescriberService> {
    const worker = new Worker(
      new URL('./videodescription.worker.ts', import.meta.url),
      {type: 'module'},
    );
    const describer = new FrameDescriberService(worker, onStatus);
    await describer.ready;
    return describer;
  }

  async warmup(frame: VideoFrame, instruction = 'What do you see?'): Promise<void> {
    await this.describe(frame, {instruction});
  }

  async describe(
    frame: VideoFrame,
    options: {instruction: string; maxNewTokens?: number},
  ): Promise<string> {
    await this.ready;

    const id = this.nextId++;

    return new Promise<string>((resolve, reject) => {
      this.pending.set(id, {resolve, reject});
      this.worker.postMessage(
        {
          type: 'describe',
          id,
          instruction: options.instruction,
          maxNewTokens: options.maxNewTokens ?? 100,
          frame,
        } satisfies WorkerRequest,
        [frame],
      );
    });
  }

  dispose(): void {
    for (const {reject} of this.pending.values()) {
      reject(new Error('Video description worker terminated'));
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

    if (message.type === 'describe-result') {
      const pending = this.pending.get(message.id);
      if (pending) {
        this.pending.delete(message.id);
        pending.resolve(message.description);
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
