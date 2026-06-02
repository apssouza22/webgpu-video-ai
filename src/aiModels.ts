/** AI models available in the demo. Mirrors gpu-video-cluster/shared/clusterModels.ts */
export interface AiModelDefinition {
  id: string;
  label: string;
  description: string;
  kind: 'detection' | 'description';
}

export const AI_MODELS: readonly AiModelDefinition[] = [
  {
    id: 'rfdetr-medium',
    label: 'Object detection (RF-DETR)',
    description: 'RF-DETR Medium (COCO) on WebGPU — bounding boxes on video',
    kind: 'detection',
  },
  {
    id: 'smolvlm-500m',
    label: 'Video description (SmolVLM)',
    description: 'SmolVLM-500M-Instruct vision-language model on WebGPU',
    kind: 'description',
  },
] as const;

export const OBJECT_DETECTION_MODEL_ID = 'rfdetr-medium';
export const VIDEO_DESCRIPTION_MODEL_ID = 'smolvlm-500m';

export const DEFAULT_AI_MODEL_ID = OBJECT_DETECTION_MODEL_ID;

export function getAiModel(id: string): AiModelDefinition | undefined {
  return AI_MODELS.find((model) => model.id === id);
}

export function isDetectionModel(id: string): boolean {
  return getAiModel(id)?.kind === 'detection';
}

export function isDescriptionModel(id: string): boolean {
  return getAiModel(id)?.kind === 'description';
}
