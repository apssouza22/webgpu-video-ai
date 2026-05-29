import {CompositionPlayer} from './player/CompositionPlayer';
import {DEMO_COMPOSITION} from './composition';

const statusEl = document.getElementById('status');
const playerEl = document.getElementById('player');

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

async function main(): Promise<void> {
  setStatus('Checking sample media…');
  await verifySamples();

  if (!playerEl) {
    throw new Error('Missing player markup');
  }

  setStatus('Loading preview…');
  await DEMO_COMPOSITION.loadLayerSources();
  await CompositionPlayer.create(DEMO_COMPOSITION, playerEl);
  setStatus('Preview ready. Use play, pause, and the scrubber to explore the timeline.');
}

main().catch((error) => {
  const message = error instanceof Error ? error.message : String(error);
  setStatus(`Initialization failed:\n${message}`);
  console.error(error);
});
