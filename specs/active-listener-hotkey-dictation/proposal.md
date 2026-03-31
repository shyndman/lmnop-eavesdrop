## Why

The repo no longer has a usable end-user transcription flow now that the old active-listener path is gone. The immediate need is a keyboard-driven dictation tool for one known Linux workstation: press a hotkey to begin dictation, press it again to finish and type the result into the currently focused app, and press Escape to cancel.

## What Changes

- Add a new `active-listener` package that runs as a long-lived CLI/service for keyboard-driven dictation.
- Implement `active-listener` as a bare Clypi command rather than a subcommand tree or UI shell.
- Support one configured keyboard device, exact-name matched at startup, with fail-fast startup if the keyboard or server connection is unavailable.
- Use `Caps Lock` as the start/finish hotkey and `Escape` as the cancel hotkey.
- Grab the keyboard only while a recording is active so hotkeys do not leak into the focused application during dictation.
- Keep the client WebSocket connection alive across recordings, with automatic reconnect attempts every 10 seconds after disconnect.
- Ignore start hotkey actions while disconnected/reconnecting, but log the suppressed attempt. Future user-facing feedback for this state is explicitly deferred.
- Emit finalized transcription as typed text through `ydotool`; no clipboard path, overlay UI, or file-mode entrypoint is included in this feature.
- Modify the client live-stream lifecycle so repeated start/stop cycles on one persistent connection are first-class and truthful.

## Capabilities

### New Capabilities
- `active-listener-hotkey-dictation`: A foreground/service CLI that listens for global hotkeys on one configured keyboard, controls dictation state, cancels or finalizes recordings, and types completed transcription into the currently focused application.

### Modified Capabilities
- `live-transcriber-client-lifecycle`: The client live transcriber connection must support long-lived connected operation with reconnect events, reconnect attempts, and repeated recording cycles on a single WebSocket without overlapping stale audio-loop tasks.

## Impact

- **Active listener package**: new long-running CLI package, logging setup, keyboard device selection, evdev input loop, dictation state machine, and text emission through `python-ydotool`.
- **Client package**: reconnect-aware live connection management, connection-state event stream, and corrected audio streaming task lifecycle for repeated `start_streaming()` / `stop_streaming()` cycles.
- **Dependencies**: `clypi` for the CLI entrypoint, `evdev` for keyboard events/grabs, and `python-ydotool` for text emission in `active-listener`. `python-ydotool` also implies an external `ydotoold`/`uinput` runtime requirement on the workstation. Existing `sounddevice` + NumPy microphone capture remains in the client path, which continues to rely on the existing PortAudio-backed audio stack.
- **Operations**: intended to run as a systemd service; logs are the only operator feedback surface in the MVP.
- **Runtime prerequisites**: Linux input-device access for `evdev`, working `ydotoold` connectivity for text emission, and the already-required live audio stack for the client microphone path.
- **Out of scope for this feature**: overlay UI, clipboard/paste output, file transcription entrypoints in `active-listener`, RTSP changes, raw key emission APIs, portability beyond the user’s current workstation.
