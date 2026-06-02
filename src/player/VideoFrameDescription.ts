import type {FrameDescriberService} from '../videodescription/FrameDescriberService';

export interface FrameDescriptionOptions {
  instruction?: string;
  maxNewTokens?: number;
  enabled?: boolean;
  onFpsUpdate?: (fps: number) => void;
  onDescriptionUpdated?: (description: string) => void;
}

export class VideoFrameDescription {
  private descriptionVersion = 0;
  private descriptionBusy = false;
  private pendingDescriptionFrame: VideoFrame | null = null;
  private enabled: boolean;
  private instruction: string;
  private maxNewTokens: number;
  private description = '';
  private lastDescriptionAt = 0;
  private onDescriptionUpdated?: (description: string) => void;

  constructor(
    private readonly describer: FrameDescriberService,
    private readonly options: FrameDescriptionOptions,
  ) {
    this.enabled = options.enabled ?? true;
    this.instruction = options.instruction ?? 'What do you see in this video frame?';
    this.maxNewTokens = options.maxNewTokens ?? 100;
    this.onDescriptionUpdated = options.onDescriptionUpdated;
  }

  setEnabled(enabled: boolean): void {
    this.enabled = enabled;
    if (!enabled) {
      this.description = '';
      this.onDescriptionUpdated?.('');
    }
  }

  isEnabled(): boolean {
    return this.enabled;
  }

  setInstruction(instruction: string): void {
    this.instruction = instruction;
  }

  getInstruction(): string {
    return this.instruction;
  }

  getDescription(): string {
    return this.enabled ? this.description : '';
  }

  async warmup(frame: VideoFrame, instruction = this.instruction): Promise<void> {
    await this.describer.warmup(frame, instruction);
  }

  schedule(videoFrame: VideoFrame): void {
    if (!this.enabled) {
      return;
    }

    if (this.descriptionBusy) {
      this.pendingDescriptionFrame?.close();
      this.pendingDescriptionFrame = new VideoFrame(videoFrame);
      return;
    }

    void this.runDescription(new VideoFrame(videoFrame));
  }

  destroy(): void {
    this.pendingDescriptionFrame?.close();
    this.pendingDescriptionFrame = null;
  }

  private async runDescription(descriptionFrame: VideoFrame): Promise<void> {
    const descriptionVersion = ++this.descriptionVersion;
    this.descriptionBusy = true;
    const startedAt = performance.now();

    try {
      const text = await this.describer.describe(descriptionFrame, {
        instruction: this.instruction,
        maxNewTokens: this.maxNewTokens,
      });

      if (descriptionVersion !== this.descriptionVersion) {
        return;
      }

      this.description = text;
      this.onDescriptionUpdated?.(text);

      if (this.options.onFpsUpdate && this.lastDescriptionAt > 0) {
        const elapsed = startedAt - this.lastDescriptionAt;
        if (elapsed > 0) {
          this.options.onFpsUpdate(1000 / elapsed);
        }
      }
      this.lastDescriptionAt = startedAt;
    } catch (error) {
      console.warn('Video description failed', error);
    } finally {
      this.descriptionBusy = false;

      if (this.pendingDescriptionFrame) {
        const nextFrame = this.pendingDescriptionFrame;
        this.pendingDescriptionFrame = null;
        void this.runDescription(nextFrame);
      }
    }
  }
}
