import { ElectronAPI } from '@electron-toolkit/preload';
import { Message } from '../messages';

interface API {
  isDev: boolean;
  logger: {
    debug: (message: string, ...args: unknown[]) => void;
    info: (message: string, ...args: unknown[]) => void;
    warn: (message: string, ...args: unknown[]) => void;
    error: (message: string, ...args: unknown[]) => void;
  };
}

interface MockAPI {
  _isPaused: boolean;
  ping: () => Promise<string>;
  setString: (target_mode: 'TRANSCRIBE' | 'COMMAND', content: string) => void;
  appendSegments: (
    target_mode: 'TRANSCRIBE' | 'COMMAND',
    completedSegments: Segment[],
    inProgressSegment: Segment,
  ) => void;
  changeMode: (target_mode: 'TRANSCRIBE' | 'COMMAND') => void;
  commitOperation: (cancelled?: boolean) => void;
  commandExecuting: (waitingMessages?: string[]) => void;
  togglePauseScenario: () => void;
  runScenario: (scenarioGenerator: Generator<{ delay: number; message: Message }>) => Promise<void>;
  runHappyPath: () => Promise<void>;
  runPerfectionistSpiral: () => Promise<void>;
}

declare global {
  interface Window {
    electron: ElectronAPI;
    api: API;
    _mock: MockAPI;
  }
}
