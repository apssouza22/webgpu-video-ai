import {
  AI_MODELS,
  DEFAULT_AI_MODEL_ID,
  isDescriptionModel,
  isDetectionModel,
} from './aiModels';
import {ObjectDetectorService} from './detection/ObjectDetectorService';
import {CompositionPlayer} from './player/CompositionPlayer';
import {VideoFrameDescription} from './player/VideoFrameDescription';
import {VideoObjectDetection} from './player/VideoObjectDetection';
import {FrameDescriberService} from './videodescription/FrameDescriberService';
import {createFpsDisplay} from './utils/fpsDisplay';
import {DEMO_COMPOSITION} from './composition';

const statusEl = document.getElementById('status');
const playerEl = document.getElementById('player');
const modelSelect = document.getElementById('model-select') as HTMLSelectElement | null;
const detectionPanel = document.getElementById('detection-panel');
const descriptionPanel = document.getElementById('description-panel');
const detectionToggle = document.getElementById('detection-enabled') as HTMLInputElement | null;
const thresholdSlider = document.getElementById('detection-threshold') as HTMLInputElement | null;
const thresholdValue = document.getElementById('detection-threshold-value');
const detectionFps = document.getElementById('detection-fps');
const descriptionFps = document.getElementById('description-fps');
const descriptionToggle = document.getElementById('description-enabled') as HTMLInputElement | null;
const descriptionInstruction = document.getElementById(
  'description-instruction',
) as HTMLInputElement | null;
const descriptionOutput = document.getElementById('description-output');

const fpsDisplay = createFpsDisplay(({render, detection}) => {
  const inferenceFpsEl = isDetectionModel(selectedModelId) ? detectionFps : descriptionFps;
  if (!inferenceFpsEl) {
    return;
  }

  const renderText = render !== null ? render.toFixed(1) : '—';
  const inferenceLabel = isDetectionModel(selectedModelId) ? 'Detection' : 'Description';
  const inferenceText = detection !== null ? detection.toFixed(1) : '—';
  inferenceFpsEl.textContent = `Render FPS: ${renderText} · ${inferenceLabel} FPS: ${inferenceText}`;
});

let selectedModelId = DEFAULT_AI_MODEL_ID;
let detector: ObjectDetectorService | null = null;
let describer: FrameDescriberService | null = null;
let detection: VideoObjectDetection | null = null;
let description: VideoFrameDescription | null = null;
let player: CompositionPlayer | null = null;
let controlsAbort: AbortController | null = null;

function setStatus(message: string): void {
  if (statusEl) {
    statusEl.textContent = message;
  }
  console.log(message);
}

function updateDescriptionOutput(text: string): void {
  if (!descriptionOutput) {
    return;
  }

  descriptionOutput.textContent = text || '—';
}

function populateModelSelect(): void {
  if (!modelSelect) {
    return;
  }

  modelSelect.replaceChildren(
    ...AI_MODELS.map((model) => {
      const option = document.createElement('option');
      option.value = model.id;
      option.textContent = model.label;
      option.title = model.description;
      return option;
    }),
  );
  modelSelect.value = DEFAULT_AI_MODEL_ID;
}

function updateModelPanels(): void {
  const isDetection = isDetectionModel(selectedModelId);
  if (detectionPanel) {
    detectionPanel.hidden = !isDetection;
  }
  if (descriptionPanel) {
    descriptionPanel.hidden = isDetection;
  }
}

async function verifySamples(): Promise<void> {
  const urls = DEMO_COMPOSITION.layers.map((clip) => clip.url);

  for (const url of urls) {
    const response = await fetch(url, {method: 'HEAD'});
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

async function ensureDetector(): Promise<ObjectDetectorService> {
  if (!detector) {
    setStatus('Loading RF-DETR object detection model (WebGPU worker)…');
    detector = await ObjectDetectorService.create(setStatus);
  }
  return detector;
}

async function ensureDescriber(): Promise<FrameDescriberService> {
  if (!describer) {
    setStatus('Loading SmolVLM-500M (WebGPU worker)…');
    describer = await FrameDescriberService.create(setStatus);
  }
  return describer;
}

function wireDetectionControls(activePlayer: CompositionPlayer): void {
  const activeDetection = activePlayer.getDetection();
  if (!activeDetection) {
    return;
  }

  controlsAbort?.abort();
  controlsAbort = new AbortController();
  const {signal} = controlsAbort;

  detectionToggle?.addEventListener(
    'change',
    () => {
      activeDetection.setEnabled(detectionToggle.checked);
      if (detectionToggle.checked) {
        void activePlayer.refreshFrame();
      }
    },
    {signal},
  );

  thresholdSlider?.addEventListener(
    'input',
    () => {
      const threshold = Number(thresholdSlider.value);
      activeDetection.setThreshold(threshold);
      if (thresholdValue) {
        thresholdValue.textContent = threshold.toFixed(2);
      }
    },
    {signal},
  );
}

function wireDescriptionControls(activePlayer: CompositionPlayer): void {
  const activeDescription = activePlayer.getDescription();
  if (!activeDescription) {
    return;
  }

  controlsAbort?.abort();
  controlsAbort = new AbortController();
  const {signal} = controlsAbort;

  descriptionToggle?.addEventListener(
    'change',
    () => {
      activeDescription.setEnabled(descriptionToggle.checked);
      if (!descriptionToggle.checked) {
        updateDescriptionOutput('');
      } else {
        void activePlayer.refreshFrame();
      }
    },
    {signal},
  );

  descriptionInstruction?.addEventListener(
    'change',
    () => {
      activeDescription.setInstruction(
        descriptionInstruction.value.trim() || 'What do you see in this video frame?',
      );
    },
    {signal},
  );
}

async function createPlayerForModel(modelId: string): Promise<CompositionPlayer> {
  const threshold = thresholdSlider ? Number(thresholdSlider.value) : 0.5;
  detection = null;
  description = null;

  if (isDetectionModel(modelId)) {
    await ensureDetector();
    detection = new VideoObjectDetection(detector!, {
      threshold,
      enabled: detectionToggle?.checked ?? true,
      onFpsUpdate: (fps) => fpsDisplay.addDetectionSample(fps),
    });
  }

  if (isDescriptionModel(modelId)) {
    await ensureDescriber();
    const instruction =
      descriptionInstruction?.value.trim() || 'What do you see in this video frame?';
    description = new VideoFrameDescription(describer!, {
      instruction,
      enabled: descriptionToggle?.checked ?? true,
      onFpsUpdate: (fps) => fpsDisplay.addDetectionSample(fps),
      onDescriptionUpdated: updateDescriptionOutput,
    });
    updateDescriptionOutput('');
  }

  setStatus('Loading preview…');
  await DEMO_COMPOSITION.loadLayerSources();

  const activePlayer = await CompositionPlayer.create(
    DEMO_COMPOSITION,
    playerEl!,
    detection,
    description,
    {
      onRenderFpsUpdate: (fps) => fpsDisplay.addRenderSample(fps),
    },
  );

  if (isDetectionModel(modelId)) {
    setStatus('Warming up RF-DETR shaders (first inference)…');
    await activePlayer.getVideoPlayer().warmupDetection(0);
    setStatus(
      'Preview ready. Detection runs on the decoded VideoFrame and boxes are drawn on the GPU.',
    );
    return activePlayer;
  }

  if (isDescriptionModel(modelId)) {
    setStatus('Warming up SmolVLM shaders (first inference)…');
    await activePlayer.getVideoPlayer().warmupDescription(0);
    setStatus(
      'Preview ready. Description runs on each decoded frame in a worker; text updates below the player.',
    );
  }

  return activePlayer;
}

async function switchModel(modelId: string): Promise<void> {
  player?.destroy();
  player = null;
  detection = null;
  description = null;
  selectedModelId = modelId;
  updateModelPanels();

  if (modelSelect) {
    modelSelect.disabled = true;
  }

  try {
    player = await createPlayerForModel(modelId);

    if (isDetectionModel(modelId)) {
      wireDetectionControls(player);
    } else {
      wireDescriptionControls(player);
    }
  } finally {
    if (modelSelect) {
      modelSelect.disabled = false;
    }
  }
}

async function main(): Promise<void> {
  populateModelSelect();
  updateModelPanels();

  setStatus('Checking sample media…');
  await verifySamples();

  if (!playerEl) {
    throw new Error('Missing player markup');
  }

  modelSelect?.addEventListener('change', () => {
    const modelId = modelSelect.value;
    if (modelId === selectedModelId) {
      return;
    }
    void switchModel(modelId);
  });

  await switchModel(DEFAULT_AI_MODEL_ID);
}

main().catch((error) => {
  const message = error instanceof Error ? error.message : String(error);
  setStatus(`Initialization failed:\n${message}`);
  console.error(error);
});
