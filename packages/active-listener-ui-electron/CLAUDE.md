# Transcription Electron App - CLAUDE.md

## Overview

This is an Electron-based real-time transcription application that receives audio transcription data from a Python client via stdin and displays it with animated visual effects. The app is designed as an always-on-top overlay with transparent background for unobtrusive voice input visualization.

## Architecture

### Multi-Process Electron Structure

- **Main Process** (`src/main/index.ts`)
  - Creates and manages the application window with specific positioning and overlay behavior
  - Handles stdin communication from Python client (JSON message parsing)
  - Forwards parsed messages to renderer via IPC (`python-data` channel)
  - Dev-only mock handlers for testing

- **Preload Script** (`src/preload/index.ts`)
  - Secure IPC bridge using `contextBridge`
  - Exposes electron APIs and custom APIs to renderer
  - Dev-only `window._mock` functions for testing

- **Renderer Process** (`src/renderer/`)
  - UI rendering and real-time transcription display
  - Canvas-based animated background with smooth height transitions
  - Two-mode system: TRANSCRIBE vs COMMAND targeting different DOM elements

### Communication Protocol

**Python → Electron via stdin:**
The Python client spawns Electron and communicates via stdin using JSON messages. All message types are defined in `src/messages.ts`:

- `append_segments`: Real-time transcription updates with completed + in-progress segments
- `change_mode`: Switch between TRANSCRIBE and COMMAND modes
- `set_segments`: Complete replacement of transcription content
- `set_string`: Raw string content with markdown-style preprocessing
- `command_executed`: Visual feedback when command recognition begins
- `commit_operation`: End session and reset to TRANSCRIBE mode

**Message Processing Flow:**

1. Python writes JSON messages to Electron's stdin
2. Main process parses JSON and forwards via `python-data` IPC channel
3. Renderer receives messages and updates appropriate DOM elements
4. Animation system smoothly transitions visual changes

### Real-time Transcription System

**Two-Mode Operation:**

- **TRANSCRIBE mode**: Updates `#transcription` element (primary speech-to-text)
- **COMMAND mode**: Updates `#commands` element (voice command recognition)

**Segment Handling:**

- `completed_segments`: Finalized transcription text that won't change
- `in_progress_segment`: Current partial text that updates in real-time
- Processing removes previous in-progress elements before appending new content

**TypeScript Data Models:**

- `Segment` interface (`src/transcription.ts`): Complete transcription metadata including timestamps, confidence, tokens
- `Word` interface: Word-level timing for detailed analysis
- `Message` union type: Type-safe discriminated union for all communication

### Visual System

**Canvas Animation** (`src/renderer/src/animation.ts`, `src/renderer/src/renderer.ts`):

- Animated background rounded rectangle that smoothly follows content height changes
- `AnimatedValue` class with configurable easing functions (easeOut, easeIn, easeInOut, linear)
- 60fps render loop with `requestAnimationFrame`
- High-DPI display support with device pixel ratio scaling

**Window Positioning:**

- Always-on-top overlay positioned on right edge of primary display
- Transparent, frameless window (360px wide, screen height - 200px)
- Mouse events ignored in production for non-intrusive overlay behavior
- Development mode allows mouse interaction for debugging

**DOM Structure:**

```html
#result-layer (content) #asr-state (container) #transcription.has-focus (speech-to-text content)
#commands (voice command content) #frame-layer (animated background canvas)
```

## Development Workflow

### Package Manager & Build System

Uses **pnpm** for package management and **electron-vite** for build tooling.

### Key Commands

```bash
# Development
pnpm dev                    # Start development server with hot reload
pnpm build                  # Type check + build for production

# Quality Assurance
pnpm typecheck              # Check TypeScript types (both node + web)
pnpm typecheck:node         # Check main/preload process types
pnpm typecheck:web          # Check renderer process types
pnpm lint                   # ESLint with caching
pnpm format                 # Prettier formatting

# Distribution
pnpm build:unpack           # Build + create unpacked app bundle
pnpm build:win              # Build Windows executable
pnpm build:mac              # Build macOS app bundle
pnpm build:linux            # Build Linux package
```

### TypeScript Configuration

Dual TypeScript configs for different execution environments:

- `tsconfig.node.json`: Main process + preload (Node.js environment)
- `tsconfig.web.json`: Renderer process (browser environment)
- Both extend `@electron-toolkit` base configs for best practices

### Development Tools

- **ESLint**: `@electron-toolkit/eslint-config-ts` + Prettier integration
- **VSCode**: Preconfigured launch targets for debugging main + renderer processes
- **Hot Reload**: Electron-vite provides instant refresh during development
- **Mock System**: Dev-only `window._mock` functions for testing without Python client. Includes both individual message functions and complete scenario generators:
  - **Individual functions**: `setString()`, `appendSegments()`, `changeMode()`, etc. for manual testing
  - **Scenario generators**: `runHappyPath()` and `runPerfectionistSpiral()` - complete workflows with realistic timing and content that demonstrate normal usage patterns and stress test the system

## Key Files Reference

### Core Application Logic

- `src/messages.ts` - Message type definitions and communication protocol
- `src/transcription.ts` - Transcription data models (Segment, Word interfaces)
- `src/main/index.ts` - Main process window management + stdin handling
- `src/preload/index.ts` - IPC bridge and API exposure
- `src/mock-scenarios.ts` - Generator functions for realistic testing scenarios with proper Whisper-style timing and segment completion behavior

### UI & Rendering

- `src/renderer/index.html` - Main UI structure
- `src/renderer/assets/main.css` - Styling for overlay appearance
- `src/renderer/src/renderer.ts` - Canvas rendering + animation orchestration
- `src/renderer/src/animation.ts` - Animation value system with easing

### Configuration

- `package.json` - Dependencies + build scripts
- `electron.vite.config.ts` - Build configuration
- `eslint.config.mjs` - Code quality rules

## Current Implementation Status

### ✅ Implemented

- Window creation and positioning system
- Stdin JSON message parsing and IPC forwarding
- Type-safe message definitions for Python communication
- Canvas animation system with smooth height transitions
- Basic UI structure for transcription + command display
- Development tooling and build pipeline

### ⚠️ Missing Implementation

- **Message Handler in Renderer**: No IPC listener for `python-data` messages in renderer process
- **DOM Update Logic**: No implementation of segment appending/replacement based on message types
- **Mode Switching**: Visual transitions between TRANSCRIBE/COMMAND modes not implemented
- **Content Preprocessing**: String preprocessing for markdown-style formatting not implemented
- **Error Handling**: No error boundaries or communication failure handling

### Development Notes

- The app currently shows static placeholder content for both transcription and command elements
- Python client integration depends on implementing the missing renderer message handlers
- Canvas animation works independently and smoothly follows content height changes
- Dev mock system in place for testing without Python dependency

## Testing Strategy

### Development Testing

- Use `window._mock.ping()` to verify preload IPC bridge functionality
- **Scenario Testing**: Run complete workflows with `await window._mock.runHappyPath()` (normal usage) or `await window._mock.runPerfectionistSpiral()` (stress testing with rapid edits and undo operations)
- **Individual Testing**: Use specific mock functions like `_mock.setString()`, `_mock.appendSegments()` for targeted feature testing
- Verify canvas animations with content height changes via dev tools
- Test window positioning across multiple displays

### Integration Testing

- Validate JSON message parsing with malformed input
- Test real-time performance with high-frequency segment updates
- Verify mode switching doesn't break animation states
- Check overlay behavior with different desktop environments

## Architectural Decisions

### Why Canvas Animation vs CSS Transitions?

Canvas provides frame-perfect control over the background animation and integrates smoothly with the transparent overlay requirements. CSS transitions could interfere with the precise positioning needed for the overlay.

### Why stdin Communication vs HTTP/WebSocket?

stdin/stdout provides the simplest integration pattern for a Python client spawning Electron. No port management, authentication, or network error handling required.

### Why Dual TypeScript Configs?

Main and renderer processes operate in fundamentally different environments (Node.js vs Chromium). Separate configs ensure proper type checking for each context's available APIs.

### Why Two-Mode DOM Structure?

Separate elements allow independent styling and transitions between transcription and command modes without complex state management in a single container.
