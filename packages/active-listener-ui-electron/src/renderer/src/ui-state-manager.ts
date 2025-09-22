import { Mode } from '../../messages';
import { Animation, AnimatedValue, Easing } from './animation';

// Timing constants from spec
const TRANSITION_DURATION_MS = 240;
const COMMIT_FEEDBACK_DURATION_MS = 1000;
const WAITING_MESSAGE_DURATION_MS = 2000;
const SEGMENT_STAGGER_DELAY_MS = 50;

// Easing function constants
const FADE_OUT_EASING = Easing.easeOut;
const FADE_IN_EASING = Easing.easeIn;

export class UIStateManager {
  // asrState's bounding box most closely resembles the shape of the window the user sees on screen
  private asrState: HTMLElement;

  // Content state tracking - stored outside DOM for quick checks
  private isTranscriptionEmpty: boolean = true;
  private isCommandEmpty: boolean = true;

  // Mode state tracking - null when in default state (no content)
  private currentMode: Mode | null = null;

  // Animation state tracking
  // Maps each mode to its active opacity animation (fade-in/fade-out transitions)
  private elementAnimations = new Map<Mode, Animation<{ opacity: AnimatedValue }>>();
  // Queues content changes that should be applied after fade-out completes
  private pendingContentChanges = new Map<Mode, string>();
  // Tracks which modes are currently having their content set
  private contentSettingInProgress = new Set<Mode>();

  constructor() {
    const asrState = document.getElementById('asr-state');
    if (!asrState) {
      throw new Error('asr-state element not found');
    }

    this.asrState = asrState;

    // Add dev mode indicator to body
    if (window.api.isDev) {
      document.body.classList.add('dev-mode');
    }

    this.setupMouseHover();
  }

  /**
   * Set content for a specific mode and handle visibility transitions
   *
   * FATAL ERROR if called concurrently for the same mode. This method assumes
   * serialized calls from the message handler.
   */
  async setContent(mode: Mode, content: string): Promise<void> {
    if (this.contentSettingInProgress.has(mode)) {
      throw new Error(
        `FATAL: setContent called concurrently for mode ${mode}. This violates the serialization assumption.`
      );
    }

    this.contentSettingInProgress.add(mode);
    try {
      const container = this.getElementForMode(mode);
      const contentIsEmpty = content.trim() === '';
      const wasActive = this.isActive();
      const hadContent = this.hasExistingContent(mode);

      // Determine animation strategy
      if (contentIsEmpty) {
        // Clearing content - fade out if there was content
        if (hadContent) {
          await this.fadeOutContent(mode);
        }
        container.innerHTML = '<p>&nbsp;</p>';
      } else if (hadContent) {
        // Replacing existing content - smooth transition
        await this.replaceContent(mode, content);
      } else {
        // Adding content to empty mode - direct fade in
        await this.fadeInContent(mode, content);
      }

      // Update content state tracking
      if (mode === Mode.TRANSCRIBE) {
        this.isTranscriptionEmpty = contentIsEmpty;
      } else {
        this.isCommandEmpty = contentIsEmpty;
      }

      // Handle mode state: if transitioning from default to active, set the target mode
      const hasTransitionedToActive = !wasActive && this.isActive();
      if (hasTransitionedToActive && !contentIsEmpty) {
        this.setMode(mode);
      } else if (!this.isActive()) {
        // If no content anywhere, clear mode state
        this.setMode(null);
      }

      // Handle visibility transitions
      this.updateVisibility();
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
   * Set the current active mode and update body classes
   */
  private setMode(mode: Mode | null): void {
    // Remove existing mode classes
    document.body.classList.remove('transcribe-active', 'command-active', 'command-executing');

    this.currentMode = mode;

    if (mode === Mode.TRANSCRIBE) {
      document.body.classList.add('transcribe-active');
    } else if (mode === Mode.COMMAND) {
      document.body.classList.add('command-active');
    }
  }

  /**
   * Update overall UI visibility based on content state
   */
  private updateVisibility(): void {
    if (this.isActive()) {
      document.body.classList.add('active');
    } else {
      document.body.classList.remove('active');
    }
  }

  /**
   * Animate opacity of paragraphs in the specified mode
   */
  private async animateOpacity(
    mode: Mode,
    fromOpacity: number,
    toOpacity: number,
    easing: (t: number) => number
  ): Promise<void> {
    const container = this.getElementForMode(mode);
    const paragraphs = container.querySelectorAll('p');

    if (paragraphs.length === 0) {
      return;
    }

    // Assert no animation is already running - our concurrency protection should prevent this
    const existingAnimation = this.elementAnimations.get(mode);
    if (existingAnimation) {
      throw new Error(
        `FATAL: Animation already running for mode ${mode}. Concurrency protection failed.`
      );
    }

    // Set initial opacity
    paragraphs.forEach((p) => {
      p.style.opacity = fromOpacity.toString();
    });

    const opacityValue = new AnimatedValue(fromOpacity, TRANSITION_DURATION_MS, easing);
    const animation = new Animation({ opacity: opacityValue }, ({ opacity }) => {
      paragraphs.forEach((p) => {
        p.style.opacity = opacity.toString();
      });
    });

    this.elementAnimations.set(mode, animation);
    opacityValue.setTarget(toOpacity);

    await animation.getCompletionPromise();
    this.elementAnimations.delete(mode);
  }

  /**
   * Fade out content in the specified mode
   */
  private fadeOutContent(mode: Mode): Promise<void> {
    return this.animateOpacity(mode, 1, 0, FADE_OUT_EASING);
  }

  /**
   * Fade in content in the specified mode
   */
  private fadeInContent(mode: Mode, content: string): Promise<void> {
    const container = this.getElementForMode(mode);

    // Set the new content, ensuring it's wrapped in paragraph tags
    const processedContent = this.ensureParagraphTags(content);
    container.innerHTML = processedContent;

    return this.animateOpacity(mode, 0, 1, FADE_IN_EASING);
  }

  /**
   * Replace content with smooth fade-out → fade-in transition
   */
  private async replaceContent(mode: Mode, newContent: string): Promise<void> {
    await this.fadeOutContent(mode);
    //
    await this.fadeInContent(mode, newContent);
  }

  /**
   * Ensure content is wrapped in paragraph tags
   */
  private ensureParagraphTags(content: string): string {
    const trimmed = content.trim();

    if (trimmed === '') {
      return '<p>&nbsp;</p>';
    }

    // If content already has paragraph tags, return as-is
    if (trimmed.startsWith('<p>') && trimmed.endsWith('</p>')) {
      return trimmed;
    }

    // Wrap content in paragraph tags
    return `<p>${trimmed}</p>`;
  }

  /**
   * Get the DOM element for a specific mode
   */
  private getElementForMode(mode: Mode): HTMLElement {
    let elementId: string;
    switch (mode) {
      case Mode.TRANSCRIBE:
        elementId = 'transcription';
        break;
      case Mode.COMMAND:
        elementId = 'command';
        break;
      default:
        const _exhaustive: never = mode;
        throw new Error(`Unknown mode: ${_exhaustive}`);
    }

    const element = document.getElementById(elementId);
    if (!element) {
      throw new Error(`${elementId} element not found`);
    }
    return element;
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
