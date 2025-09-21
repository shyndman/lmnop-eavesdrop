import { app, shell, BrowserWindow, ipcMain, Display } from 'electron';
import { screen } from 'electron';
import { join } from 'path';
import { electronApp, optimizer, is } from '@electron-toolkit/utils';
import icon from '../../resources/icon.png?asset';

const WINDOW_WIDTH = 360;
const WINDOW_H_INSET = 20

function createWindow(screen: Display): BrowserWindow {
  // Create the browser window.
  const { height } = screen.workAreaSize

  const mainWindow = new BrowserWindow({
    width: WINDOW_WIDTH,
    height: height - 200,
    show: false,
    autoHideMenuBar: true,
    transparent: true,
    frame: false,
    center: false,
    ...(process.platform === 'linux' ? { icon } : {}),
    webPreferences: {
      preload: join(__dirname, '../preload/index.js'),
      sandbox: false
    },
    titleBarStyle: 'hidden'
  });

  if (!is.dev) {
    mainWindow.setIgnoreMouseEvents(true, { forward: true });
  }

  mainWindow.setAlwaysOnTop(true, 'status');

  mainWindow.on('ready-to-show', () => {
    mainWindow.show();
  });

  mainWindow.webContents.setWindowOpenHandler((details) => {
    shell.openExternal(details.url);
    return { action: 'deny' };
  });

  // HMR for renderer base on electron-vite cli.
  // Load the remote URL for development or the local html file for production.
  if (is.dev && process.env['ELECTRON_RENDERER_URL']) {
    mainWindow.loadURL(process.env['ELECTRON_RENDERER_URL']);
  } else {
    mainWindow.loadFile(join(__dirname, '../renderer/index.html'));
  }

  return mainWindow;
}

// This method will be called when Electron has finished
// initialization and is ready to create browser windows.
// Some APIs can only be used after this event occurs.
app.whenReady().then(() => {
  // Set app user model id for windows
  electronApp.setAppUserModelId('com.electron');

  // Default open or close DevTools by F12 in development
  // and ignore CommandOrControl + R in production.
  // see https://github.com/alex8088/electron-toolkit/tree/master/packages/utils
  app.on('browser-window-created', (_, window) => {
    optimizer.watchWindowShortcuts(window);
  });

  // IPC test
  ipcMain.on('ping', () => console.log('pong'));

  // Dev-only mock handlers
  if (is.dev) {
    ipcMain.handle('mock.ping', () => {
      console.log('ed: pong');
      return 'pong';
    });
  }

  // Set up stdin communication from Python
  if (process.stdin.isTTY === false) {
    process.stdin.setEncoding('utf8');

    let buffer = '';
    process.stdin.on('data', (chunk) => {
      buffer += chunk;
      const lines = buffer.split('\n');
      buffer = lines.pop() || ''; // Keep incomplete line in buffer

      lines.forEach(line => {
        if (line.trim()) {
          try {
            const data = JSON.parse(line);
            mainWindow.webContents.send('python-data', data);
          } catch (error) {
            console.error('Failed to parse JSON from Python:', error, 'Line:', line);
          }
        }
      });
    });
  }

  // Create a window that fills the screen's available work area.
  const primaryDisplay = screen.getPrimaryDisplay();
  console.log('Primary display:', primaryDisplay.bounds);
  console.log('All displays:', screen.getAllDisplays().map(d => ({ id: d.id, bounds: d.bounds })));

  const mainWindow = createWindow(primaryDisplay);

  // Force the window to appear on the primary display
  const { x: displayX, y: displayY } = primaryDisplay.bounds;
  const { width } = primaryDisplay.workAreaSize;
  const x = displayX + width - WINDOW_WIDTH - WINDOW_H_INSET;
  const y = displayY + 100;

  mainWindow.setPosition(x, y);
});

// Quit when all windows are closed, except on macOS. There, it's common
// for applications and their menu bar to stay active until the user quits
// explicitly with Cmd + Q.
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

// In this file you can include the rest of your app's specific main process
// code. You can also put them in separate files and require them here.
