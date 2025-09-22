import { ElectronAPI } from '@electron-toolkit/preload'

interface API {
  isDev: boolean
  logger: {
    debug: (message: string, ...args: unknown[]) => void
    info: (message: string, ...args: unknown[]) => void
    warn: (message: string, ...args: unknown[]) => void
    error: (message: string, ...args: unknown[]) => void
  }
}

interface MockAPI {
  ping: () => Promise<string>
  setString: (target_mode: 'TRANSCRIBE' | 'COMMAND', content: string) => void
  appendSegments: (target_mode: 'TRANSCRIBE' | 'COMMAND', completedSegments: any[], inProgressSegment: any) => void
  changeMode: (target_mode: 'TRANSCRIBE' | 'COMMAND') => void
}

declare global {
  interface Window {
    electron: ElectronAPI
    api: API
    _mock?: MockAPI
  }
}
