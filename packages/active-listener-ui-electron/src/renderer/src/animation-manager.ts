import { Animation, AnimatedValue, Easing } from './animation';
import { Mode } from '../../messages';

// Easing function constants
const FADE_OUT_EASING = Easing.easeOut;
const FADE_IN_EASING = Easing.easeIn;
const FADE_IN_DURATION = 200;
const FADE_OUT_DURATION = 150;

/**
 * Manages element animations with support for opacity transitions and staggered timing.
 *
 * Provides semantic animation methods that work with arrays of elements,
 * abstracting the complexity of coordinated transitions from the UI state layer.
 */
export class AnimationManager {
  // Mode-based animation tracking for concurrency protection
  private modeAnimations = new Map<Mode, Animation<{ opacity: AnimatedValue }>>();

  /**
   * Fade elements in (opacity 0 → 1) with optional staggered timing
   */
  async fadeIn(elements: HTMLElement[], stagger: number = 0): Promise<void> {
    return this.fadeElements(elements, 0, 1, FADE_IN_DURATION, FADE_IN_EASING, stagger);
  }

  /**
   * Fade elements out (opacity 1 → 0)
   */
  async fadeOut(elements: HTMLElement[]): Promise<void> {
    return this.fadeElements(elements, 1, 0, FADE_OUT_DURATION, FADE_OUT_EASING);
  }

  /**
   * Animate element opacity with full control over direction, easing, and stagger timing
   */
  async fadeElements(
    elements: HTMLElement[],
    fromOpacity: number,
    toOpacity: number,
    duration: number,
    easing: (t: number) => number,
    interElementDelay: number = 0,
  ): Promise<void> {
    if (elements.length === 0) {
      return;
    }

    // Set initial opacity
    elements.forEach((el) => {
      el.style.opacity = fromOpacity.toString();
    });

    // Staggered animation - create separate AnimatedValue for each element
    const animatedValues: Record<string, AnimatedValue> = {};
    elements.forEach((_element, index) => {
      const delay = index * interElementDelay;
      animatedValues[`element${index}`] = new AnimatedValue(fromOpacity, duration, easing, delay);
    });

    const animation = new Animation(animatedValues, (values) => {
      elements.forEach((element, index) => {
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
   * Fade out content in a specific mode with concurrency protection
   */
  async fadeOutModeContent(mode: Mode, container: HTMLElement): Promise<void> {
    const paragraphs = Array.from(container.querySelectorAll('p')) as HTMLElement[];

    // Assert no animation is already running - concurrency protection
    const existingAnimation = this.modeAnimations.get(mode);
    if (existingAnimation) {
      throw new Error(
        `FATAL: Animation already running for mode ${mode}. Concurrency protection failed.`,
      );
    }

    this.modeAnimations.set(mode, {} as Animation<{ opacity: AnimatedValue }>);
    try {
      await this.fadeElements(paragraphs, 1, 0, FADE_OUT_DURATION, FADE_OUT_EASING);
    } finally {
      this.modeAnimations.delete(mode);
    }
  }

  /**
   * Fade in content in a specific mode with concurrency protection
   */
  async fadeInModeContent(
    mode: Mode,
    container: HTMLElement,
    processedContent: string,
  ): Promise<void> {
    // Set the new content first
    container.innerHTML = processedContent;

    const paragraphs = Array.from(container.querySelectorAll('p')) as HTMLElement[];

    // Assert no animation is already running - concurrency protection
    const existingAnimation = this.modeAnimations.get(mode);
    if (existingAnimation) {
      throw new Error(
        `FATAL: Animation already running for mode ${mode}. Concurrency protection failed.`,
      );
    }

    this.modeAnimations.set(mode, {} as Animation<{ opacity: AnimatedValue }>);
    try {
      await this.fadeElements(paragraphs, 0, 1, FADE_IN_DURATION, FADE_IN_EASING);
    } finally {
      this.modeAnimations.delete(mode);
    }
  }

  /**
   * Replace content with smooth fade-out → fade-in transition
   */
  async replaceModeContent(
    mode: Mode,
    container: HTMLElement,
    processedContent: string,
  ): Promise<void> {
    await this.fadeOutModeContent(mode, container);
    await this.fadeInModeContent(mode, container, processedContent);
  }
}
