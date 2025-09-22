import { contextBridge, ipcRenderer } from 'electron'
import { electronAPI } from '@electron-toolkit/preload'
import { Mode } from '../messages'
import { happyPathScenario, perfectionistSpiralScenario } from './mock-scenarios'

// Custom APIs for renderer
const api = {
  isDev: process.env.NODE_ENV === 'development',
  logger: {
    debug: (message: string, ...args: unknown[]) => ipcRenderer.send('logger', 'debug', message, ...args),
    info: (message: string, ...args: unknown[]) => ipcRenderer.send('logger', 'info', message, ...args),
    warn: (message: string, ...args: unknown[]) => ipcRenderer.send('logger', 'warn', message, ...args),
    error: (message: string, ...args: unknown[]) => ipcRenderer.send('logger', 'error', message, ...args)
  }
}

// Dev-only APIs for testing and mocking
const _mock = process.env.NODE_ENV === 'development' ? {
  ping: () => ipcRenderer.invoke('mock.ping'),
  setString: (target_mode: 'TRANSCRIBE' | 'COMMAND', content: string) => {
    const mappedMode = target_mode === 'TRANSCRIBE' ? Mode.TRANSCRIBE : Mode.COMMAND;
    ipcRenderer.send('mock.python-data', {
      type: 'set_string',
      target_mode: mappedMode,
      content
    });
  },
  appendSegments: (target_mode: 'TRANSCRIBE' | 'COMMAND', completedSegments: any[], inProgressSegment: any) => {
    const mappedMode = target_mode === 'TRANSCRIBE' ? Mode.TRANSCRIBE : Mode.COMMAND;
    ipcRenderer.send('mock.python-data', {
      type: 'append_segments',
      target_mode: mappedMode,
      completed_segments: completedSegments,
      in_progress_segment: inProgressSegment
    });
  },
  changeMode: (target_mode: 'TRANSCRIBE' | 'COMMAND') => {
    const mappedMode = target_mode === 'TRANSCRIBE' ? Mode.TRANSCRIBE : Mode.COMMAND;
    ipcRenderer.send('mock.python-data', {
      type: 'change_mode',
      target_mode: mappedMode
    });
  },
  commitOperation: (cancelled: boolean = false) => {
    ipcRenderer.send('mock.python-data', {
      type: 'commit_operation',
      cancelled
    });
  },
  commandExecuting: (waitingMessages: string[] = []) => {
    ipcRenderer.send('mock.python-data', {
      type: 'command_executing',
      waiting_messages: waitingMessages
    });
  },

  /**
   * Runs a complete happy path scenario: transcribe → command → refine → commit
   * Demonstrates normal user workflow with realistic timing and content
   */
  runHappyPath: async () => {
    const scenario = happyPathScenario();
    let step = scenario.next();

    while (!step.done) {
      const { delay, message } = step.value;

      // Wait for the specified delay
      await new Promise(resolve => setTimeout(resolve, delay));

      // Send the message
      ipcRenderer.send('mock.python-data', message);

      step = scenario.next();
    }
  },

  /**
   * Runs the perfectionist spiral scenario: multiple refinements with undo operations
   * Stress tests rapid mode switching and version history management
   */
  runPerfectionistSpiral: async () => {
    const scenario = perfectionistSpiralScenario();
    let step = scenario.next();

    while (!step.done) {
      const { delay, message } = step.value;

      // Wait for the specified delay
      await new Promise(resolve => setTimeout(resolve, delay));

      // Send the message
      ipcRenderer.send('mock.python-data', message);

      step = scenario.next();
    }
  }
} : undefined

// Use `contextBridge` APIs to expose Electron APIs to
// renderer only if context isolation is enabled, otherwise
// just add to the DOM global.
if (process.contextIsolated) {
  try {
    contextBridge.exposeInMainWorld('electron', electronAPI)
    contextBridge.exposeInMainWorld('api', api)
    if (_mock) {
      contextBridge.exposeInMainWorld('_mock', _mock)
    }
  } catch (error) {
    console.error(error)
  }
} else {
  // @ts-ignore (define in dts)
  window.electron = electronAPI
  // @ts-ignore (define in dts)
  window.api = api
  if (_mock) {
    // @ts-ignore (define in dts)
    window._mock = _mock
  }
}
