import {ObjectDetector} from './detection/ObjectDetector';
import {CompositionPlayer} from './player/CompositionPlayer';
import {VideoObjectDetection} from './player/VideoObjectDetection';
import {DEMO_COMPOSITION} from './composition';

const statusEl = document.getElementById('status');
const playerEl = document.getElementById('player');
const detectionToggle = document.getElementById('detection-enabled') as HTMLInputElement | null;
const thresholdSlider = document.getElementById('detection-threshold') as HTMLInputElement | null;
const thresholdValue = document.getElementById('detection-threshold-value');
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
          '  - video.mp4, video-2.mp4 (with audio tracks)\n' +
          '  - overlay.png, overlay-2.png\n\n' +
          'See README.md for details.',
      );
    }
  }
}

function wireDetectionControls(player: CompositionPlayer): void {
  const detection = player.getDetection();
  if (!detection) {
    return;
  }

  detectionToggle?.addEventListener('change', () => {
    detection.setEnabled(detectionToggle.checked);
    if (detectionToggle.checked) {
      void player.refreshFrame();
    }
  });

  thresholdSlider?.addEventListener('input', () => {
    const threshold = Number(thresholdSlider.value);
    detection.setThreshold(threshold);
    if (thresholdValue) {
      thresholdValue.textContent = threshold.toFixed(2);
    }
  });
}

async function main(): Promise<void> {
  setStatus('Checking sample media…');
  await verifySamples();

  if (!playerEl) {
    throw new Error('Missing player markup');
  }

  setStatus('Loading RF-DETR object detection model (WebGPU worker)…');
  const detector = await ObjectDetector.create(setStatus);
  const threshold = thresholdSlider ? Number(thresholdSlider.value) : 0.5;

  const detection = new VideoObjectDetection(detector, {
    threshold,
    enabled: detectionToggle?.checked ?? true,
    onFpsUpdate: (fps) => {
      if (detectionFps) {
        detectionFps.textContent = `Detection FPS: ${fps.toFixed(1)}`;
      }
    },
  });

  setStatus('Loading preview…');
  await DEMO_COMPOSITION.loadLayerSources();

  const player = await CompositionPlayer.create(DEMO_COMPOSITION, playerEl, detection);

  wireDetectionControls(player);

  setStatus('Warming up RF-DETR shaders (first inference)…');
  await player.getVideoPlayer().warmupDetection(0);

  setStatus(
    'Preview ready. Detection runs on the decoded VideoFrame (transferred to a worker) and boxes are drawn on the GPU.',
  );
}

main().catch((error) => {
  const message = error instanceof Error ? error.message : String(error);
  setStatus(`Initialization failed:\n${message}`);
  console.error(error);
});
