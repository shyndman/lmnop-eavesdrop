export const Easing = {
  easeOut: (t: number): number => 1 - Math.pow(1 - t, 3),
  easeIn: (t: number): number => Math.pow(t, 3),
  easeInOut: (t: number): number => t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2,
  linear: (t: number): number => t
};

export class AnimatedValue {
  private current: number;
  private target: number;
  private startValue: number;
  private isAnimating: boolean = false;
  private animationStart: number = 0;
  private duration: number;
  private easingFn: (t: number) => number;
  private onComplete?: () => void;

  constructor(
    initialValue: number,
    duration: number = 300,
    easing: (t: number) => number = Easing.easeOut,
    onComplete?: () => void
  ) {
    this.current = initialValue;
    this.target = initialValue;
    this.startValue = initialValue;
    this.duration = duration;
    this.easingFn = easing;
    this.onComplete = onComplete;
  }

  setTarget(newTarget: number): void {
    if (newTarget !== this.target) {
      this.target = newTarget;
      this.startValue = this.current;
      this.isAnimating = true;
      this.animationStart = performance.now();
    }
  }

  update(): number {
    if (!this.isAnimating) {
      return this.current;
    }

    const elapsed = performance.now() - this.animationStart;
    const progress = Math.min(elapsed / this.duration, 1);
    const easedProgress = this.easingFn(progress);

    this.current = this.startValue + (this.target - this.startValue) * easedProgress;

    if (progress >= 1) {
      this.isAnimating = false;
      this.current = this.target;

      if (this.onComplete) {
        this.onComplete();
      }
    }

    return this.current;
  }

  getValue(): number {
    return this.current;
  }

  isRunning(): boolean {
    return this.isAnimating;
  }

  setOnComplete(callback: () => void): void {
    this.onComplete = callback;
  }
}