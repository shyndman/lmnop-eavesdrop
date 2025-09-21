import { contextBridge, ipcRenderer } from 'electron'
import { electronAPI } from '@electron-toolkit/preload'

// Custom APIs for renderer
const api = {
  isDev: process.env.NODE_ENV === 'development'
}

// Dev-only APIs for testing and mocking
const _mock = process.env.NODE_ENV === 'development' ? {
  ping: () => ipcRenderer.invoke('mock.ping')
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
