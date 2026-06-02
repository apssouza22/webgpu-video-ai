/// <reference lib="webworker" />

import {
  AutoModelForVision2Seq,
  AutoProcessor,
  RawImage,
  Tensor,
  type PreTrainedModel,
  type Processor,
} from '@huggingface/transformers';
import type {WorkerRequest, WorkerResponse} from './workerMessages';

const MODEL_ID = 'HuggingFaceTB/SmolVLM-500M-Instruct';

const DTYPE = {
  embed_tokens: 'fp16',
  vision_encoder: 'q4',
  decoder_model_merged: 'q4',
} as const;

let processor: Processor | null = null;
let model: PreTrainedModel | null = null;
let loadPromise: Promise<void> | null = null;

function post(message: WorkerResponse): void {
  self.postMessage(message);
}

function reportDownloadProgress(progress: {
  status: string;
  file?: string;
  loaded?: number;
  total?: number;
}): void {
  if (progress.status === 'progress' && progress.file) {
    const percent =
      progress.total && progress.total > 0
        ? Math.round(((progress.loaded ?? 0) / progress.total) * 100)
        : null;
    post({
      type: 'status',
      message: percent
        ? `Downloading ${progress.file} (${percent}%)…`
        : `Downloading ${progress.file}…`,
    });
  }
}

async function loadPipeline(): Promise<{processor: Processor; model: PreTrainedModel}> {
  if (processor && model) {
    return {processor, model};
  }

  if (!loadPromise) {
    loadPromise = (async () => {
      post({type: 'status', message: 'Loading SmolVLM processor…'});
      processor = await AutoProcessor.from_pretrained(MODEL_ID, {
        progress_callback: reportDownloadProgress,
      });

      post({type: 'status', message: 'Loading SmolVLM model (WebGPU)…'});
      model = await AutoModelForVision2Seq.from_pretrained(MODEL_ID, {
        dtype: DTYPE,
        device: 'webgpu',
        progress_callback: reportDownloadProgress,
      });

      post({type: 'status', message: 'Compiling SmolVLM shaders…'});
    })();
  }

  await loadPromise;
  return {processor: processor!, model: model!};
}

async function frameToRawImage(frame: VideoFrame): Promise<RawImage> {
  const bitmap = await createImageBitmap(frame);
  try {
    const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
    const ctx = canvas.getContext('2d', {willReadFrequently: true});
    if (!ctx) {
      throw new Error('OffscreenCanvas 2D context not available in worker');
    }
    ctx.drawImage(bitmap, 0, 0);
    return RawImage.fromCanvas(canvas);
  } finally {
    bitmap.close();
    frame.close();
  }
}

async function runDescription(
  frame: VideoFrame,
  instruction: string,
  maxNewTokens: number,
): Promise<string> {
  const {processor: proc, model: vlm} = await loadPipeline();
  const image = await frameToRawImage(frame);

  const messages = [
    {
      role: 'user',
      content: [{type: 'image'}, {type: 'text', text: instruction}],
    },
  ];

  const text = proc.apply_chat_template(messages, {
    add_generation_prompt: true,
  });

  const inputs = await proc(text, [image], {
    do_image_splitting: false,
  });

  const generatedIds = (await vlm.generate({
    ...inputs,
    max_new_tokens: maxNewTokens,
    do_sample: false,
    repetition_penalty: 1.1,
  })) as Tensor;

  const decoded = proc.batch_decode(generatedIds, {
    skip_special_tokens: true,
  });

  const promptDecoded = proc.batch_decode(inputs.input_ids, {
    skip_special_tokens: true,
  });

  let description = decoded[0] ?? '';
  const promptText = promptDecoded[0] ?? '';
  if (promptText && description.startsWith(promptText)) {
    description = description.slice(promptText.length);
  }

  return description.trim();
}

self.onmessage = async (event: MessageEvent<WorkerRequest>) => {
  const message = event.data;

  try {
    if (message.type === 'init') {
      await loadPipeline();
      post({type: 'ready'});
      return;
    }

    if (message.type === 'describe') {
      const description = await runDescription(
        message.frame,
        message.instruction,
        message.maxNewTokens,
      );
      post({type: 'describe-result', id: message.id, description});
      return;
    }
  } catch (error) {
    if (message.type === 'describe') {
      message.frame.close();
    }

    post({
      type: 'error',
      error: error instanceof Error ? error.message : String(error),
      id: message.type === 'describe' ? message.id : undefined,
    });
  }
};
