import type { DetectionResult } from './ObjectDetector';

const COLORS = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899'];

export class DetectionOverlay {
  readonly canvas: HTMLCanvasElement;
  private readonly ctx: CanvasRenderingContext2D;
  private readonly labelColorMap = new Map<string, string>();
  private nextColorIndex = 0;

  constructor() {
    this.canvas = document.createElement('canvas');
    this.canvas.className = 'composition-player__detection-overlay';
    const ctx = this.canvas.getContext('2d');
    if (!ctx) {
      throw new Error('Failed to create detection overlay context');
    }
    this.ctx = ctx;
  }

  resizeToMatch(sourceCanvas: HTMLCanvasElement): void {
    const width = sourceCanvas.clientWidth;
    const height = sourceCanvas.clientHeight;
    const dpr = window.devicePixelRatio || 1;

    this.canvas.style.width = `${width}px`;
    this.canvas.style.height = `${height}px`;
    this.canvas.width = Math.round(width * dpr);
    this.canvas.height = Math.round(height * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  clear(): void {
    const dpr = window.devicePixelRatio || 1;
    this.ctx.setTransform(1, 0, 0, 1, 0, 0);
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  draw(results: DetectionResult[], displayWidth: number, displayHeight: number): void {
    this.clear();

    this.ctx.font = '600 13px system-ui, sans-serif';
    this.ctx.lineWidth = 2.5;

    for (const { box, label, score } of results) {
      const color = this.colorForLabel(label);
      const x1 = box.xmin * displayWidth;
      const y1 = box.ymin * displayHeight;
      const boxWidth = (box.xmax - box.xmin) * displayWidth;
      const boxHeight = (box.ymax - box.ymin) * displayHeight;

      this.ctx.strokeStyle = color;
      this.ctx.beginPath();
      this.ctx.roundRect(x1, y1, boxWidth, boxHeight, 6);
      this.ctx.stroke();

      const text = `${label} ${(score * 100).toFixed(0)}%`;
      const textWidth = this.ctx.measureText(text).width;

      this.ctx.fillStyle = color;
      this.ctx.beginPath();
      this.ctx.roundRect(x1, y1 - 26, textWidth + 12, 22, 4);
      this.ctx.fill();

      this.ctx.fillStyle = 'white';
      this.ctx.fillText(text, x1 + 6, y1 - 9);
    }
  }

  private colorForLabel(label: string): string {
    if (!this.labelColorMap.has(label)) {
      this.labelColorMap.set(label, COLORS[this.nextColorIndex % COLORS.length]);
      this.nextColorIndex++;
    }
    return this.labelColorMap.get(label)!;
  }
}
