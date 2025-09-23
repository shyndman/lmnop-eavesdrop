import { Mode } from '../../messages';
import { Segment } from '../../transcription';
import { DomManager } from './dom';
import { WaitingMessageManager } from './waiting-message-manager';

// Timing constants from spec
export const TRANSITION_DURATION_MS = 240;
export const COMMIT_FEEDBACK_DURATION_MS = 1000;

export class StateManager {
  // asrState's bounding box most closely resembles the shape of the window the user sees on screen
  private asrState: HTMLElement;

  // Content state tracking - stored outside DOM for quick checks
  private isTranscriptionEmpty: boolean = true;
  private isCommandEmpty: boolean = true;

  // Mode state tracking - null when in default state (no content)
  private currentMode: Mode = Mode.TRANSCRIBE;

  // Command execution state
  private waitingMessageManager: WaitingMessageManager;

  // Tracks which modes are currently having their content set
  private contentSettingInProgress = new Set<Mode>();

  constructor(private dom: DomManager) {
    const asrState = document.getElementById('asr-state');
    if (!asrState) {
      throw new Error('asr-state element not found');
    }

    this.asrState = asrState;

    // Initialize waiting message manager
    const waitingMessageContainer = document.getElementById('command-waiting-messages');
    if (!waitingMessageContainer) {
      throw new Error('command-waiting-messages element not found');
    }
    this.waitingMessageManager = new WaitingMessageManager(waitingMessageContainer);

    // Add dev mode indicator to body
    if (window.api.isDev) {
      // document.body.classList.add('dev-mode');
    }

    this.setupMouseHover();
  }

  /**
   * Set content for a specific mode and handle visibility transitions
   *
   * FATAL ERROR if called concurrently for the same mode. This method assumes
   * serialized calls from the message handler.
   */
  async setString(mode: Mode, content: string): Promise<void> {
    // Exit command execution state if currently active
    this.stopCommandExecution();

    if (this.contentSettingInProgress.has(mode)) {
      throw new Error(
        `FATAL: setString called concurrently for mode ${mode}. This violates the serialization assumption.`,
      );
    }

    this.contentSettingInProgress.add(mode);
    try {
      const hasNewContent = content.trim() !== '';
      const wasActive = this.isActive();
      const hasExistingContent = this.hasExistingContent(mode);

      // Determine animation strategy
      if (hasExistingContent) {
        if (!hasNewContent) {
          // Clearing content - fade out if there was content
          await this.dom.clearModeContent(mode);
        } else {
          // Replacing existing content - smooth transition
          await this.dom.replaceModeContent(mode, content);
        }
      } else {
        // Adding content to empty mode - direct fade in
        await this.dom.addModeContent(mode, content);
      }

      // Update content state tracking
      if (mode === Mode.TRANSCRIBE) {
        this.isTranscriptionEmpty = !hasNewContent;
      } else {
        this.isCommandEmpty = !hasNewContent;
      }

      // Handle mode state: if transitioning from default to active, set the target mode
      const hasTransitionedToActive = !wasActive && this.isActive();
      if (hasTransitionedToActive && hasNewContent) {
        this.currentMode = mode;
      } else if (!this.isActive()) {
        // If no content anywhere, clear mode state
        this.currentMode = Mode.TRANSCRIBE;
      }

      // Handle visibility transitions
      this.applyStateToDom();
    } finally {
      this.contentSettingInProgress.delete(mode);
    }
  }

  /**
   * Check if UI should be active (has content in any mode)
   */
  private isActive(): boolean {
    return !this.isTranscriptionEmpty || !this.isCommandEmpty;
  }

  /**
   * Check if a specific mode has content
   */
  private hasExistingContent(mode: Mode): boolean {
    switch (mode) {
      case Mode.TRANSCRIBE:
        return !this.isTranscriptionEmpty;
      case Mode.COMMAND:
        return !this.isCommandEmpty;
      default:
        const _exhaustive: never = mode;
        throw new Error(`Unknown mode: ${_exhaustive}`);
    }
  }

  /**
   * Evaluates the the receiver's state, and adds or removes the corresponding CSS classes to the
   * body element.
   */
  private applyStateToDom(): void {
    this.dom.commitBodyClasses({
      isCommandExecuting: this.waitingMessageManager.isRunning(),
      currentMode: this.currentMode,
      isActive: this.isActive(),
      commandElementVisible: this.isCommandElementVisible(),
    });
  }

  /**
   * Check if command element should be visible based on mode and content
   */
  private isCommandElementVisible(): boolean {
    return this.currentMode === Mode.COMMAND || !this.isCommandEmpty;
  }

  /**
   * Append segments to a specific mode with staggered animations
   *
   * FATAL ERROR if called concurrently for the same mode. This method assumes
   * serialized calls from the message handler.
   */
  async appendSegments(
    mode: Mode,
    completedSegments: readonly Segment[],
    inProgressSegment: Segment,
  ): Promise<void> {
    if (this.contentSettingInProgress.has(mode)) {
      throw new Error(
        `FATAL: appendSegments called concurrently for mode ${mode}. This violates the serialization assumption.`,
      );
    }

    this.contentSettingInProgress.add(mode);
    try {
      const wasActive = this.isActive();
      const hasNewContent = completedSegments.length > 0 || inProgressSegment.text.trim() !== '';

      // Remove existing in-progress segment with animation
      await this.dom.updateModeDomSegments(this.currentMode!, completedSegments, inProgressSegment);

      // Update content state tracking
      if (mode === Mode.TRANSCRIBE) {
        this.isTranscriptionEmpty = !hasNewContent;
      } else {
        this.isCommandEmpty = !hasNewContent;
      }

      // Handle mode state: if transitioning from default to active, set the target mode
      const hasTransitionedToActive = !wasActive && this.isActive();
      if (hasTransitionedToActive && hasNewContent) {
        this.currentMode = mode;
      } else if (!this.isActive()) {
        // If no content anywhere, clear mode state
        this.currentMode = Mode.TRANSCRIBE;
      }

      // Handle visibility transitions
      this.applyStateToDom();
    } finally {
      this.contentSettingInProgress.delete(mode);
    }
  }

  /**
   * Change the active mode and handle visual transitions
   */
  changeMode(mode: Mode): void {
    this.currentMode = mode;
    this.applyStateToDom();
  }

  /**
   * Start command execution with waiting message cycling
   */
  startCommandExecution(waitingMessages: string[]): void {
    this.waitingMessageManager.start(waitingMessages);
    this.applyStateToDom();
  }

  /**
   * Stop command execution and hide overlay
   */
  stopCommandExecution(): void {
    this.waitingMessageManager.stop();
    this.dom.clearModeContent(Mode.COMMAND);
    this.applyStateToDom();
  }

  /**
   * Handle commit operation with visual feedback and session reset
   */
  async commitOperation(_cancelled: boolean): Promise<void> {
    await this.dom.whileCommitActive(async () => {
      // Phase 1: Show commit feedback for 1 second
      await new Promise((resolve) => setTimeout(resolve, COMMIT_FEEDBACK_DURATION_MS));

      // Phase 2: Update content state tracking
      this.isTranscriptionEmpty = true;
      this.isCommandEmpty = true;

      // Phase 3: Remove commit feedback class and trigger fade-out
      await this.dom.clearAllContent();
    });

    // Since both modes are now empty, this will trigger fade-out
    this.currentMode = Mode.TRANSCRIBE;
    this.applyStateToDom();
  }

  private setupMouseHover(): void {
    // Mouse hover affordance - allows seeing content below window during active states
    this.asrState.addEventListener('mouseenter', () => {
      document.body.classList.add('mouse-is-over');
    });

    this.asrState.addEventListener('mouseleave', () => {
      document.body.classList.remove('mouse-is-over');
    });
  }
}
