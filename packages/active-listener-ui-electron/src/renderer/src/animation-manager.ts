import { Animation, AnimatedValue, Easing } from './animation';

/**
 * Manages element animations with support for opacity transitions and staggered timing.
 *
 * Provides semantic animation methods that work with arrays of elements,
 * abstracting the complexity of coordinated transitions from the UI state layer.
 */
export class AnimationManager {
  constructor(
    private transitionDuration: number,
    private fadeOutEasing: (t: number) => number = Easing.easeOut,
    private fadeInEasing: (t: number) => number = Easing.easeIn,
  ) {}

  /**
   * Fade elements out (opacity 1 → 0)
   */
  async fadeOut(elements: HTMLElement[]): Promise<void> {
    return this.fadeElements(elements, 1, 0, this.fadeOutEasing);
  }

  /**
   * Fade elements in (opacity 0 → 1) with optional staggered timing
   */
  async fadeIn(elements: HTMLElement[], stagger: number = 0): Promise<void> {
    return this.fadeElements(elements, 0, 1, this.fadeInEasing, stagger);
  }

  /**
   * Animate element opacity with full control over direction, easing, and stagger timing
   */
  async fadeElements(
    elements: HTMLElement[],
    fromOpacity: number,
    toOpacity: number,
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
      animatedValues[`element${index}`] = new AnimatedValue(
        fromOpacity,
        this.transitionDuration,
        easing,
        delay,
      );
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
}