import type {Composition} from '../composition';
import {AudioPlayer} from './AudioPlayer';
import {VideoPlayer} from './VideoPlayer';
import type {VideoObjectDetection} from './VideoObjectDetection';

export interface CompositionPlayerOptions {
  onRenderFpsUpdate?: (fps: number) => void;
}

export class CompositionPlayer {
  private readonly root: HTMLElement;
  private readonly videoPlayer: VideoPlayer;
  private readonly audioPlayer: AudioPlayer;
  private readonly detection: VideoObjectDetection;
  private readonly onRenderFpsUpdate?: (fps: number) => void;
  private playButton!: HTMLButtonElement;
  private scrubber!: HTMLInputElement;
  private timeLabel!: HTMLSpanElement;
  private animationFrame: number | null = null;
  private currentTime = 0;
  private isPlaying = false;
  private playStartedAt = 0;
  private playStartedTime = 0;
  private lastRenderAt = 0;

  static async create(
    composition: Composition,
    container: HTMLElement,
    detection: VideoObjectDetection,
    options: CompositionPlayerOptions = {},
  ): Promise<CompositionPlayer> {
    const videoPlayer = await VideoPlayer.create(composition, detection);

    try {
      const audioPlayer = await AudioPlayer.create(composition.audioLayers);
      return new CompositionPlayer(
        composition,
        container,
        videoPlayer,
        audioPlayer,
        detection,
        options,
      );
    } catch (error) {
      videoPlayer.destroy();
      throw error;
    }
  }

  private constructor(
    private readonly composition: Composition,
    container: HTMLElement,
    videoPlayer: VideoPlayer,
    audioPlayer: AudioPlayer,
    detection: VideoObjectDetection,
    options: CompositionPlayerOptions,
  ) {
    this.videoPlayer = videoPlayer;
    this.audioPlayer = audioPlayer;
    this.detection = detection;
    this.onRenderFpsUpdate = options.onRenderFpsUpdate;
    this.root = document.createElement('div');
    this.root.className = 'composition-player';

    const canvas = this.videoPlayer.getCanvas();
    canvas.className = 'composition-player__canvas';
    this.root.appendChild(canvas);
    this.root.appendChild(this.createControls());

    container.replaceChildren(this.root);
    this.updateControls();
    void this.refreshFrame();
  }

  getVideoPlayer(): VideoPlayer {
    return this.videoPlayer;
  }

  getDetection(): VideoObjectDetection  {
    return this.detection;
  }

  getVideoCanvas(): HTMLCanvasElement {
    return this.videoPlayer.getCanvas();
  }

  async refreshFrame(options: {skipDetection?: boolean} = {}): Promise<void> {
    const startedAt = performance.now();
    this.updateControls();
    await this.videoPlayer.render(this.currentTime, this.duration, options);

    if (this.onRenderFpsUpdate && this.lastRenderAt > 0) {
      const elapsed = startedAt - this.lastRenderAt;
      if (elapsed > 0) {
        this.onRenderFpsUpdate(1000 / elapsed);
      }
    }
    this.lastRenderAt = startedAt;
  }

  pause(): void {
    this.pausePlayback();
  }

  destroy(): void {
    this.pausePlayback();
    this.audioPlayer.destroy();
    this.videoPlayer.destroy();
    this.detection?.destroy();
  }

  private createControls(): HTMLElement {
    const controls = document.createElement('div');
    controls.className = 'composition-player__controls';

    this.playButton = document.createElement('button');
    this.playButton.type = 'button';
    this.playButton.textContent = 'Play';

    this.scrubber = document.createElement('input');
    this.scrubber.type = 'range';
    this.scrubber.min = '0';
    this.scrubber.max = `${Math.max(this.composition.duration, 0)}`;
    this.scrubber.step = '0.001';
    this.scrubber.value = '0';

    this.timeLabel = document.createElement('span');
    this.timeLabel.className = 'composition-player__time';

    this.playButton.addEventListener('click', () => {
      if (this.isPlaying) {
        this.pausePlayback();
      } else {
        void this.startPlayback();
      }
    });

    this.scrubber.addEventListener('input', () => {
      this.currentTime = Number(this.scrubber.value);
      if (this.isPlaying) {
        this.playStartedAt = performance.now();
        this.playStartedTime = this.currentTime;
        this.audioPlayer.seek(this.currentTime);
      }
      void this.refreshFrame();
    });

    controls.append(this.playButton, this.scrubber, this.timeLabel);
    return controls;
  }

  private async startPlayback(): Promise<void> {
    if (this.currentTime >= this.duration) {
      this.currentTime = 0;
    }

    this.isPlaying = true;
    this.playStartedAt = performance.now();
    this.playStartedTime = this.currentTime;
    this.playButton.textContent = 'Pause';

    try {
      await this.audioPlayer.play(this.currentTime);
    } catch (error) {
      console.warn('Audio preview playback failed', error);
      this.pausePlayback();
      return;
    }

    this.schedulePlaybackFrame();
  }

  private pausePlayback(): void {
    if (this.isPlaying) {
      this.currentTime = this.playbackTime();
    }

    this.isPlaying = false;
    this.cancelPlaybackFrame();
    this.audioPlayer.pause();
    this.playButton.textContent = 'Play';
    this.updateControls();
  }

  private updateControls(): void {
    this.scrubber.max = `${this.duration}`;
    this.scrubber.value = `${this.currentTime}`;
    this.timeLabel.textContent = `${this.formatTime(this.currentTime)} / ${this.formatTime(this.duration)}`;
  }

  private schedulePlaybackFrame(): void {
    if (!this.isPlaying || this.animationFrame !== null) {
      return;
    }

    this.animationFrame = requestAnimationFrame(async () => {
      this.animationFrame = null;
      if (!this.isPlaying) {
        return;
      }

      this.currentTime = this.playbackTime();
      if (this.currentTime >= this.duration) {
        this.currentTime = this.duration;
        this.pausePlayback();
        return;
      }

      await this.refreshFrame();
      this.schedulePlaybackFrame();
    });
  }

  private cancelPlaybackFrame(): void {
    if (this.animationFrame !== null) {
      cancelAnimationFrame(this.animationFrame);
      this.animationFrame = null;
    }
  }

  private playbackTime(): number {
    return Math.min(
      this.duration,
      this.playStartedTime + (performance.now() - this.playStartedAt) / 1000,
    );
  }

  private get duration(): number {
    return Math.max(this.composition.duration, 0);
  }

  private formatTime(time: number): string {
    if (!Number.isFinite(time)) {
      return '0:00';
    }

    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60).toString().padStart(2, '0');
    return `${minutes}:${seconds}`;
  }
}
