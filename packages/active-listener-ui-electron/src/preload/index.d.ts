import { ElectronAPI } from '@electron-toolkit/preload'

interface API {
  isDev: boolean
}

interface MockAPI {
  ping: () => Promise<string>
}

declare global {
  interface Window {
    electron: ElectronAPI
    api: API
    _mock?: MockAPI
  }
}
