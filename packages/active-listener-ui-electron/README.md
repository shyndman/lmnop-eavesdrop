# Active Listener UI (Electron)

Transparent floating overlay that displays real-time voice transcription. Designed to sit unobtrusively on the right edge of the screen while the user speaks.

## Architecture

This is the UI component of the Active Listener system. It's spawned by the Python `active-listener` package and communicates via stdin/stderr.

### IPC Protocol

```
Python ──stdin JSON messages──► Electron (this package)
       ◄──stderr logs/signals──
```

**Startup sequence:**
1. Python spawns this Electron app
2. Renderer initializes and calls `window.api.logger.info("ACTIVE_LISTENER_UI_READY")`
3. Main process writes this to stderr
4. Python detects the signal and begins sending transcription updates via stdin

**Message flow:**
1. Python writes JSON to Electron's stdin
2. Main process parses JSON, forwards via IPC (`python-data` channel)
3. Renderer's `MessageHandler` processes messages and updates UI state

### Key Files

- **`src/main/index.ts`** — Window management, stdin parsing, IPC forwarding
- **`src/preload/index.ts`** — Secure bridge exposing `electron` and `api` to renderer
- **`src/renderer/src/message-handler.ts`** — Processes incoming messages
- **`src/renderer/src/state-manager.ts`** — UI state and DOM updates
- **`src/messages.ts`** — TypeScript message type definitions (must match Python's `ui_messages.py`)

### Window Behavior

- Transparent, frameless overlay (390px wide)
- Always-on-top with `skipTaskbar: true`
- Mouse events ignored (click-through)
- Positioned on right edge of primary display
- Automatically repositions on display changes

## Development

```bash
# Install dependencies
pnpm install

# Start development server with hot reload
pnpm dev

# Type check
pnpm typecheck

# Build for production
pnpm build

# Build unpacked app (for testing with Python)
pnpm build:unpack
```

### Testing Without Python

The app includes mock functions for development:

```javascript
// In DevTools console
await window._mock.runHappyPath()        // Complete usage scenario
await window._mock.appendSegments(...)   // Individual message testing
```

## Recommended IDE Setup

- [VSCode](https://code.visualstudio.com/)
- [ESLint](https://marketplace.visualstudio.com/items?itemName=dbaeumer.vscode-eslint)
- [Prettier](https://marketplace.visualstudio.com/items?itemName=esbenp.prettier-vscode)
