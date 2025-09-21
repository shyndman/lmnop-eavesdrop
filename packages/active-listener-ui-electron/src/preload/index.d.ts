import { ElectronAPI } from '@electron-toolkit/preload'

interface MockAPI {
  ping: () => Promise<string>
}

declare global {
  interface Window {
    electron: ElectronAPI
    api: unknown
    _mock?: MockAPI
  }
}
