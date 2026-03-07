# Active Listener - CLAUDE.md

## Overview

Python orchestrator for real-time voice transcription with a floating overlay UI. Connects to an Eavesdrop transcription server and displays results through a companion Electron application.

## Architecture

Active Listener is a **two-package system**:

- **`active-listener`** (this package) — Python process that:
  - Connects to Eavesdrop server via WebSocket
  - Spawns and manages the Electron UI subprocess
  - Maintains transcription workspace state
  - Handles dual-mode operation (TRANSCRIBE vs COMMAND)

- **`active-listener-ui-electron`** — Electron overlay that displays transcription

### IPC Protocol

Communication uses stdin/stderr between Python and Electron:

```
Python ──stdin JSON messages──► Electron
       ◄──stderr logs/signals──
```

**Startup sequence:**
1. Python spawns the Electron binary via `ui_channel.py`
2. Electron renderer emits `ACTIVE_LISTENER_UI_READY` to stderr
3. Python detects signal and begins sending transcription updates
4. JSON messages flow via stdin; Electron logs return via stderr

**Important:** stdout is unused and set to DEVNULL.

### Key Components

| File | Purpose |
|------|---------|
| `app.py` | Main application orchestrator, lifecycle management |
| `workspace.py` | Transcription state with per-mode segment tracking |
| `ui_channel.py` | Subprocess lifecycle, IPC, ready signal detection |
| `ui_messages.py` | Pydantic message types for Python→Electron |
| `client.py` | WebSocket client wrapper for Eavesdrop server |
| `output.py` | Clipboard-based text output (copy + paste via ydotool) |

### Message Types

Defined in `ui_messages.py`, must stay in sync with `active-listener-ui-electron/src/messages.ts`:

- `append_segments` — Real-time transcription updates (completed + in-progress)
- `change_mode` — Switch between TRANSCRIBE/COMMAND modes
- `set_string` — Replace content with transformed text
- `command_executing` — Visual feedback for command recognition
- `commit_operation` — End session, reset to TRANSCRIBE

### Dual-Mode Operation

- **TRANSCRIBE mode** — Voice builds the text buffer
- **COMMAND mode** — Voice edits/transforms the text buffer

Each mode maintains independent segment tracking via `_completed_by_mode` dict.

## Development

```bash
# Install dependencies
uv sync

# Type checking
uv run basedpyright

# Linting
uv run ruff check

# Run (requires Electron app built)
python -m eavesdrop.active_listener --ui-bin /path/to/electron --server localhost:9090
```

## Known Issues

See `/ISSUES.md` for tracked issues.
