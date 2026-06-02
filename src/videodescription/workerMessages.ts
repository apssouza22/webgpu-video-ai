export type WorkerRequest =
  | {type: 'init'}
  | {
      type: 'describe';
      id: number;
      frame: VideoFrame;
      instruction: string;
      maxNewTokens: number;
    };

export type WorkerResponse =
  | {type: 'status'; message: string}
  | {type: 'ready'}
  | {type: 'describe-result'; id: number; description: string}
  | {type: 'error'; error: string; id?: number};
