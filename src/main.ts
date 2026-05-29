import {DetectionController} from './detection/DetectionController';
import {ObjectDetector} from './detection/ObjectDetector';
import {CompositionPlayer} from './player/CompositionPlayer';
import {DEMO_COMPOSITION} from './composition';

const statusEl = document.getElementById('status');
const playerEl = document.getElementById('player');
const detectionToggle = document.getElementById('detection-enabled') as HTMLInputElement | null;
const thresholdSlider = document.getElementById('detection-threshold') as HTMLInputElement | null;
const thresholdValue = document.getElementById('detection-threshold-value');
const allowedLabelsInput = document.getElementById('detection-labels') as HTMLInputElement | null;
const detectionFps = document.getElementById('detection-fps');

function setStatus(message: string): void {
  if (statusEl) {
    statusEl.textContent = message;
  }
  console.log(message);
}

async function verifySamples(): Promise<void> {
  const urls = DEMO_COMPOSITION.layers.map((clip) => clip.url);

  for (const url of urls) {
    const response = await fetch(url, { method: 'HEAD' });
    if (!response.ok) {
      throw new Error(
        `Missing sample media: ${url}\n\n` +
          'Add files under public/samples/:\n' +
          '  - video.mp4 (with audio track)\n' +
          '  - overlay.png\n\n' +
          'See README.md for details.',
      );
    }
  }
}

function wireDetectionControls(
  player: CompositionPlayer,
  detection: DetectionController,
): void {
  detectionToggle?.addEventListener('change', () => {
    detection.setEnabled(detectionToggle.checked);
    if (detectionToggle.checked) {
      void detection.processFrame(player.getVideoCanvas());
    }
  });

  thresholdSlider?.addEventListener('input', () => {
    const threshold = Number(thresholdSlider.value);
    detection.setThreshold(threshold);
    if (thresholdValue) {
      thresholdValue.textContent = threshold.toFixed(2);
    }
  });

  allowedLabelsInput?.addEventListener('input', () => {
    detection.setAllowedLabels(allowedLabelsInput.value);
  });
}

async function main(): Promise<void> {
  setStatus('Checking sample media…');
  await verifySamples();

  if (!playerEl) {
    throw new Error('Missing player markup');
  }

  setStatus('Loading RF-DETR object detection model (WebGPU)…');
  const detector = await ObjectDetector.create(setStatus);

  const detection = new DetectionController({
    detector,
    threshold: thresholdSlider ? Number(thresholdSlider.value) : 0.5,
    onFpsUpdate: (fps) => {
      if (detectionFps) {
        detectionFps.textContent = `Detection FPS: ${fps.toFixed(1)}`;
      }
    },
  });

  if (detectionToggle) {
    detection.setEnabled(detectionToggle.checked);
  }

  setStatus('Loading preview…');
  await DEMO_COMPOSITION.loadLayerSources();

  const player = await CompositionPlayer.create(DEMO_COMPOSITION, playerEl, {
    detection,
  });

  wireDetectionControls(player, detection);

  setStatus('Warming up RF-DETR shaders (first inference)…');
  const canvas = player.getVideoCanvas();
  await detector.warmup(canvas, detection.getThreshold());
  await detection.processFrame(canvas);

  setStatus(
    'Preview ready. Play or scrub the timeline — object detection runs on each composed frame.',
  );
}

main().catch((error) => {
  const message = error instanceof Error ? error.message : String(error);
  setStatus(`Initialization failed:\n${message}`);
  console.error(error);
});
