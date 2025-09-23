import { contextBridge, ipcRenderer } from 'electron';
import { electronAPI } from '@electron-toolkit/preload';
import { Message, Mode } from '../messages';
import { happyPathScenario, perfectionistSpiralScenario } from './mock-scenarios';
import { Segment } from '../transcription';

// Custom APIs for renderer
const api = {
  isDev: process.env.NODE_ENV === 'development',
  logger: {
    debug: (message: string, ...args: unknown[]) => {
      console.debug(message, ...args);
      ipcRenderer.send('logger', 'debug', message, ...args);
    },
    info: (message: string, ...args: unknown[]) => {
      console.info(message, ...args);
      ipcRenderer.send('logger', 'info', message, ...args);
    },
    warn: (message: string, ...args: unknown[]) => {
      console.warn(message, ...args);
      ipcRenderer.send('logger', 'warn', message, ...args);
    },
    error: (message: string, ...args: unknown[]) => {
      console.error(message, ...args);
      ipcRenderer.send('logger', 'error', message, ...args);
    },
  },
};

// Dev-only APIs for testing and mocking

let _mock;
// eslint-disable-next-line prefer-const
_mock =
  process.env.NODE_ENV === 'development'
    ? {
        // Pause state for scenario execution
        _isPaused: false,

        ping: () => ipcRenderer.invoke('mock.ping'),
        setString: (target_mode: 'TRANSCRIBE' | 'COMMAND', content: string) => {
          const mappedMode = target_mode === 'TRANSCRIBE' ? Mode.TRANSCRIBE : Mode.COMMAND;
          ipcRenderer.send('mock.python-data', {
            type: 'set_string',
            target_mode: mappedMode,
            content,
          });
        },
        appendSegments: (
          target_mode: 'TRANSCRIBE' | 'COMMAND',
          completedSegments: Segment[],
          inProgressSegment: Segment,
        ) => {
          const mappedMode = target_mode === 'TRANSCRIBE' ? Mode.TRANSCRIBE : Mode.COMMAND;
          ipcRenderer.send('mock.python-data', {
            type: 'append_segments',
            target_mode: mappedMode,
            completed_segments: completedSegments,
            in_progress_segment: inProgressSegment,
          });
        },
        changeMode: (target_mode: 'TRANSCRIBE' | 'COMMAND') => {
          const mappedMode = target_mode === 'TRANSCRIBE' ? Mode.TRANSCRIBE : Mode.COMMAND;
          ipcRenderer.send('mock.python-data', {
            type: 'change_mode',
            target_mode: mappedMode,
          });
        },
        commitOperation: (cancelled: boolean = false) => {
          ipcRenderer.send('mock.python-data', {
            type: 'commit_operation',
            cancelled,
          });
        },
        commandExecuting: (waitingMessages: string[] = []) => {
          ipcRenderer.send('mock.python-data', {
            type: 'command_executing',
            waiting_messages: waitingMessages,
          });
        },

        /**
         * Toggles pause state for currently running scenario.
         * When paused, execution stops between steps allowing DOM inspection.
         */
        togglePauseScenario: () => {
          _mock._isPaused = !_mock._isPaused;
          console.log(
            _mock._isPaused
              ? 'Scenario paused - call togglePauseScenario() to resume'
              : 'Scenario resumed',
          );
        },

        /**
         * Shared execution logic for all scenario generators.
         * Handles timing, pause state, and message sending.
         */
        runScenario: async (scenarioGenerator: Generator<{ delay: number; message: Message }>) => {
          let step = scenarioGenerator.next();

          while (!step.done) {
            const { delay, message } = step.value;

            // Wait for the specified delay
            await new Promise((resolve) => setTimeout(resolve, delay));

            // Check for pause state before sending message
            while (_mock._isPaused) {
              await new Promise((resolve) => setTimeout(resolve, 100)); // Check every 100ms
            }

            // Send the message
            ipcRenderer.send('mock.python-data', message);

            step = scenarioGenerator.next();
          }
        },

        /**
         * Runs a complete happy path scenario: transcribe → command → refine → commit
         * Demonstrates normal user workflow with realistic timing and content
         */
        runHappyPath: async () => {
          return _mock.runScenario(happyPathScenario());
        },

        /**
         * Runs the perfectionist spiral scenario: multiple refinements with undo operations
         * Stress tests rapid mode switching and version history management
         */
        runPerfectionistSpiral: async () => {
          return _mock.runScenario(perfectionistSpiralScenario());
        },
      }
    : undefined;

// Use `contextBridge` APIs to expose Electron APIs to
// renderer only if context isolation is enabled, otherwise
// just add to the DOM global.
if (process.contextIsolated) {
  try {
    contextBridge.exposeInMainWorld('electron', electronAPI);
    contextBridge.exposeInMainWorld('api', api);
    if (_mock) {
      contextBridge.exposeInMainWorld('_mock', _mock);
    }
  } catch (error) {
    console.error(error);
  }
} else {
  // @ts-ignore (define in dts)
  window.electron = electronAPI;
  // @ts-ignore (define in dts)
  window.api = api;
  if (_mock) {
    // @ts-ignore (define in dts)
    window._mock = _mock;
  }
}
