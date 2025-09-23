/**
 * Message types for Python <-> Electron communication via stdin/stdout.
 *
 * These messages handle real-time transcription display, mode switching between
 * transcription and command recognition, and operation lifecycle management.
 */

import { Segment } from './transcription';

/**
 * Application modes that determine which DOM element receives transcription updates.
 *
 * - TRANSCRIBE: Normal speech-to-text mode, updates #transcription element
 * - COMMAND: Voice command recognition mode, updates #command element
 */
export enum Mode {
  TRANSCRIBE = 'TRANSCRIBE',
  COMMAND = 'COMMAND',
}

/**
 * Message type discriminator enum for type-safe message handling.
 */
export enum MessageType {
  APPEND_SEGMENTS = 'append_segments',
  CHANGE_MODE = 'change_mode',
  SET_STRING = 'set_string',
  COMMAND_EXECUTING = 'command_executing',
  COMMIT_OPERATION = 'commit_operation',
}

/**
 * Incrementally updates transcription content with new completed segments and
 * current in-progress text.
 *
 * Processing order:
 * 1. Remove previous in-progress segment elements from DOM
 * 2. Append new completed segments to target element
 * 3. Append new in-progress segment (if provided) to target element
 *
 * This message type handles the high-frequency real-time transcription updates
 * as speech is being processed.
 */
export interface AppendSegmentsMessage {
  readonly type: MessageType.APPEND_SEGMENTS;

  /** Which mode/DOM element to update (transcription vs command) */
  readonly target_mode: Mode;

  /** Newly finalized segments to permanently append */
  readonly completed_segments: readonly Segment[];

  /** Current partial transcription text that may change on next update */
  readonly in_progress_segment: Segment;
}

/**
 * Switches the application between transcription and command recognition modes.
 *
 * This triggers:
 * - Visual transitions between #transcription and #command elements
 * - Mode-specific UI state changes
 * - Redirection of subsequent transcription messages to the target element
 *
 * All operations begin in TRANSCRIBE mode.
 */
export interface ChangeModeMessage {
  readonly type: MessageType.CHANGE_MODE;

  /** The mode to switch to */
  readonly target_mode: Mode;
}

/**
 * Completely replaces all existing content with a preprocessed string.
 *
 * The content undergoes preprocessing before display:
 * - Markdown-style paragraphs converted to <p> tags
 * - Future: syntax highlighting, command formatting, etc.
 *
 * This is useful for displaying formatted text that doesn't follow the
 * standard segment-based transcription structure.
 */
export interface SetStringMessage {
  readonly type: MessageType.SET_STRING;

  /** Which mode/DOM element to update */
  readonly target_mode: Mode;

  /** Raw content string to preprocess and display */
  readonly content: string;
}

/**
 * Indicates that a voice command has been recognized and execution has begun.
 *
 * This triggers visual feedback to show the user that their command is being
 * processed. Fired when Python begins executing the recognized command, not
 * when the command completes.
 *
 * The waiting_messages provide user feedback during command processing,
 * displayed one at a time in a loop until command results arrive.
 */
export interface CommandExecutingMessage {
  readonly type: MessageType.COMMAND_EXECUTING;
  // TODO: Document
  readonly waiting_messages: string[];
}

/**
 * Signals the end of a transcription operation and resets application state.
 *
 * This message indicates that the current transcription session is complete
 * and the application should prepare for the next operation. Always results
 * in a reset to TRANSCRIBE mode.
 *
 * The cancelled flag indicates whether the operation was completed normally
 * or was interrupted/cancelled by the user or system.
 */
export interface CommitOperationMessage {
  readonly type: MessageType.COMMIT_OPERATION;

  /** True if the operation was cancelled, false if completed normally */
  readonly cancelled: boolean;
}

/**
 * Union type of all possible messages sent from Python to Electron.
 *
 * Use TypeScript's discriminated union pattern with the `type` field
 * for type-safe message handling:
 *
 * ```typescript
 * function handleMessage(message: Message) {
 *   switch (message.type) {
 *     case MessageType.APPEND_SEGMENTS:
 *       // message is now typed as AppendSegmentsMessage
 *       break;
 *     case MessageType.CHANGE_MODE:
 *       // message is now typed as ChangeModeMessage
 *       break;
 *     // ... etc
 *   }
 * }
 * ```
 */
export type Message =
  | AppendSegmentsMessage
  | ChangeModeMessage
  | SetStringMessage
  | CommandExecutingMessage
  | CommitOperationMessage;
