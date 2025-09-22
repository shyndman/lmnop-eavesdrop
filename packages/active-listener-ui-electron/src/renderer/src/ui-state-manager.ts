import { Mode } from '../../messages';
import { Segment } from '../../transcription';
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
  private elementAnimations = new Map<
    Mode,
    Animation<{ opacity: AnimatedValue }>
  >();
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
        `FATAL: setContent called concurrently for mode ${mode}. This violates the serialization assumption.`,
      );
    }

    this.contentSettingInProgress.add(mode);
    try {
      const container = this.getElementForMode(mode);
      const hasNewContent = content.trim() !== '';
      const wasActive = this.isActive();
      const hasExistingContent = this.hasExistingContent(mode);

      // Determine animation strategy
      if (hasExistingContent) {
        if (!hasNewContent) {
          // Clearing content - fade out if there was content
          await this.fadeOutContent(mode);
          container.innerHTML = '<p>&nbsp;</p>';
        } else {
          // Replacing existing content - smooth transition
          await this.replaceContent(mode, content);
        }
      } else {
        // Adding content to empty mode - direct fade in
        await this.fadeInContent(mode, content);
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
        this.currentMode = null;
      }

      // Handle visibility transitions
      this.commitBodyClasses();
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
  private commitBodyClasses(): void {
    document.body.classList.remove(
      'transcribe-active',
      'command-active',
      'command-executing',
    );
    if (this.currentMode === Mode.TRANSCRIBE) {
      document.body.classList.add('transcribe-active');
    } else if (this.currentMode === Mode.COMMAND) {
      document.body.classList.add('command-active');
    }

    if (this.isActive()) {
      document.body.classList.add('active');
    } else {
      document.body.classList.remove('active');
    }

    if (this.isCommandElementVisible()) {
      document.body.classList.add('command-visible');
    } else {
      document.body.classList.remove('command-visible');
    }
  }

  /**
   * Check if command element should be visible based on mode and content
   */
  private isCommandElementVisible(): boolean {
    return this.currentMode === Mode.COMMAND || !this.isCommandEmpty;
  }

  /**
   * Animate opacity of any set of elements with optional staggered delays
   */
  private async animateElementsOpacity(
    elements: NodeListOf<Element> | HTMLElement[],
    fromOpacity: number,
    toOpacity: number,
    easing: (t: number) => number,
    interElementDelay: number = 0,
  ): Promise<void> {
    if (elements.length === 0) {
      return;
    }

    // Set initial opacity
    const elementsArray = Array.from(elements) as HTMLElement[];
    elementsArray.forEach((element) => {
      element.style.opacity = fromOpacity.toString();
    });

    // Staggered animation - create separate AnimatedValue for each element
    const animatedValues: Record<string, AnimatedValue> = {};
    elementsArray.forEach((element, index) => {
      const delay = index * interElementDelay;
      animatedValues[`element${index}`] = new AnimatedValue(
        fromOpacity,
        TRANSITION_DURATION_MS,
        easing,
        delay,
      );
    });

    const animation = new Animation(animatedValues, (values) => {
      elementsArray.forEach((element, index) => {
        element.style.opacity = values[`element${index}`].toString();
      });
    });

    // Start all animations
    Object.values(animatedValues).forEach((value) => {
      value.setTarget(toOpacity);
    });

    await animation.getCompletionPromise();
  }

  /**
   * Animate opacity of paragraphs in the specified mode
   */
  private async animateOpacity(
    mode: Mode,
    fromOpacity: number,
    toOpacity: number,
    easing: (t: number) => number,
  ): Promise<void> {
    const container = this.getElementForMode(mode);
    const paragraphs = container.querySelectorAll('p');

    // Assert no animation is already running - our concurrency protection should prevent this
    const existingAnimation = this.elementAnimations.get(mode);
    if (existingAnimation) {
      throw new Error(
        `FATAL: Animation already running for mode ${mode}. Concurrency protection failed.`,
      );
    }

    this.elementAnimations.set(
      mode,
      {} as Animation<{ opacity: AnimatedValue }>,
    );
    try {
      await this.animateElementsOpacity(
        paragraphs,
        fromOpacity,
        toOpacity,
        easing,
      );
    } finally {
      this.elementAnimations.delete(mode);
    }
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

  /**
   * Calculate CSS probability class for a segment
   */
  private getSegmentProbabilityClass(segment: Segment): string {
    const rounded = Math.round((segment.avg_probability * 100) / 5) * 5;
    return `segment-prob-${rounded}`;
  }

  /**
   * Remove the existing in-progress segment with animation (single segment invariant)
   */
  private async removeInProgressSegment(container: HTMLElement): Promise<void> {
    const inProgressSpan = container.querySelector(
      '.in-progress-segment',
    ) as HTMLElement;
    if (inProgressSpan) {
      await this.animateElementsOpacity(
        [inProgressSpan],
        1,
        0,
        FADE_OUT_EASING,
      );
      inProgressSpan.remove();
    }
  }

  /**
   * Create a paragraph containing segment spans
   */
  private createSegmentParagraph(
    completedSegments: readonly Segment[],
    inProgressSegment: Segment,
  ): HTMLParagraphElement {
    const paragraph = document.createElement('p');

    // Create spans for completed segments
    completedSegments.forEach((segment) => {
      const span = document.createElement('span');
      span.id = `segment-${segment.id}`;
      span.className = this.getSegmentProbabilityClass(segment);
      span.textContent = segment.text;
      paragraph.appendChild(span);
    });

    // Create span for in-progress segment if it has text
    if (inProgressSegment.text.trim() !== '') {
      const span = document.createElement('span');
      span.id = `segment-${inProgressSegment.id}`;
      span.className = `in-progress-segment ${this.getSegmentProbabilityClass(inProgressSegment)}`;
      span.textContent = inProgressSegment.text;
      paragraph.appendChild(span);
    }

    return paragraph;
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
      const container = this.getElementForMode(mode);
      const wasActive = this.isActive();
      const hasExistingContent = this.hasExistingContent(mode);

      // Remove existing in-progress segment with animation
      await this.removeInProgressSegment(container);

      // Create new paragraph with all segments
      const paragraph = this.createSegmentParagraph(completedSegments, inProgressSegment);
      const allSpans = Array.from(paragraph.querySelectorAll('span')) as HTMLSpanElement[];

      // Determine if we have meaningful content
      const hasNewContent = completedSegments.length > 0 || inProgressSegment.text.trim() !== '';

      if (hasNewContent) {
        // Add new paragraph alongside existing content
        container.appendChild(paragraph);
        await this.animateElementsOpacity(allSpans, 0, 1, FADE_IN_EASING, SEGMENT_STAGGER_DELAY_MS);
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
        this.currentMode = null;
      }

      // Handle visibility transitions
      this.commitBodyClasses();
    } finally {
      this.contentSettingInProgress.delete(mode);
    }
  }

  /**
   * Change the active mode and handle visual transitions
   */
  changeMode(mode: Mode): void {
    this.currentMode = mode;
    this.commitBodyClasses();
  }

  /**
   * Handle commit operation with visual feedback and session reset
   */
  async commitOperation(cancelled: boolean): Promise<void> {
    // Phase 1: Show commit feedback for 1 second
    document.body.classList.add('commit-active');
    await new Promise(resolve => setTimeout(resolve, COMMIT_FEEDBACK_DURATION_MS));

    // Phase 2: Clear content from both modes
    const transcriptionElement = this.getElementForMode(Mode.TRANSCRIBE);
    const commandElement = this.getElementForMode(Mode.COMMAND);

    // Update content state tracking
    this.isTranscriptionEmpty = true;
    this.isCommandEmpty = true;

    // Phase 3: Remove commit feedback class and trigger fade-out
    document.body.classList.remove('commit-active');

    // Since both modes are now empty, this will trigger fade-out
    this.commitBodyClasses();

    // Phase 4: After fade-out completes, reset to transcribe mode
    setTimeout(() => {
      transcriptionElement.innerHTML = '<p>&nbsp;</p>';
      commandElement.innerHTML = '<p>&nbsp;</p>';
      this.currentMode = Mode.TRANSCRIBE;
      this.commitBodyClasses();
    }, TRANSITION_DURATION_MS);
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
