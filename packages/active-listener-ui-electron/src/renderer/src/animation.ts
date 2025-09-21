export const Easing = {
  easeOut: (t: number): number => 1 - Math.pow(1 - t, 3),
  easeIn: (t: number): number => Math.pow(t, 3),
  easeInOut: (t: number): number => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2),
  linear: (t: number): number => t
};

export class AnimatedValue {
  private current: number;
  private target: number;
  private startValue: number;
  private isAnimating: boolean = false;
  private animationStart: number = 0;
  private duration: number;
  private delay: number;
  private easingFn: (t: number) => number;
  private onComplete?: () => void;
  private animation?: Animation<any>;

  constructor(
    initialValue: number,
    duration: number = 300,
    easing: (t: number) => number = Easing.easeOut,
    onComplete?: () => void,
    delay: number = 0
  ) {
    this.current = initialValue;
    this.target = initialValue;
    this.startValue = initialValue;
    this.duration = duration;
    this.delay = delay;
    this.easingFn = easing;
    this.onComplete = onComplete;
  }

  setTarget(newTarget: number): void {
    if (newTarget !== this.target) {
      this.target = newTarget;
      this.startValue = this.current;
      const wasAnimating = this.isAnimating;
      this.isAnimating = true;
      this.animationStart = performance.now();

      if (!wasAnimating && this.animation) {
        this.animation.onAnimatedValueChanged(this);
      }
    }
  }

  update(now: DOMHighResTimeStamp): number {
    if (!this.isAnimating) {
      return this.current;
    }

    const elapsed = now - this.animationStart;

    // Still in delay period
    if (elapsed < this.delay) {
      return this.current;
    }

    // Animation phase
    const animationElapsed = elapsed - this.delay;
    const progress = Math.min(animationElapsed / this.duration, 1);
    const easedProgress = this.easingFn(progress);

    this.current = this.startValue + (this.target - this.startValue) * easedProgress;

    if (progress >= 1) {
      this.isAnimating = false;
      this.current = this.target;

      if (this.animation) {
        this.animation.onAnimatedValueSettled(this);
      }

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

  setAnimation(animation: Animation<any>): void {
    this.animation = animation;
  }
}

export class Animation<T extends Record<string, AnimatedValue>> {
  private isRunning: boolean = false;
  private animationId: number | null = null;
  private callback: (values: { [K in keyof T]: number }) => void;
  private animatedValues: T;

  constructor(animatedValues: T, callback: (values: { [K in keyof T]: number }) => void) {
    this.animatedValues = animatedValues;
    this.callback = callback;

    for (const value of Object.values(animatedValues)) {
      value.setAnimation(this);
      const originalOnComplete = value['onComplete'];
      value.setOnComplete(() => {
        if (originalOnComplete) {
          originalOnComplete();
        }
      });
    }
  }

  getIsRunning(): boolean {
    return this.isRunning;
  }

  getAnimatedValue<K extends keyof T>(name: K): T[K] {
    return this.animatedValues[name];
  }

  onAnimatedValueChanged(_value: AnimatedValue): void {
    if (!this.isRunning) {
      this.start();
    }
  }

  onAnimatedValueSettled(_value: AnimatedValue): void {
    this.checkAndStopIfComplete();
  }

  private start(): void {
    if (this.isRunning) {
      return;
    }

    this.isRunning = true;
    this.tick(performance.now());
  }

  private stop(): void {
    if (!this.isRunning) {
      return;
    }

    this.isRunning = false;
    if (this.animationId !== null) {
      cancelAnimationFrame(this.animationId);
      this.animationId = null;
    }
  }

  private tick = (timestamp: DOMHighResTimeStamp): void => {
    if (!this.isRunning || !this.callback) {
      return;
    }

    const values = {} as { [K in keyof T]: number };
    for (const [name, animatedValue] of Object.entries(this.animatedValues)) {
      (values as Record<string, number>)[name] = animatedValue.update(timestamp);
    }

    this.callback(values);
    this.animationId = requestAnimationFrame(this.tick);
  };

  private checkAndStopIfComplete(): void {
    const hasRunningAnimations = Object.values(this.animatedValues).some((value) =>
      value.isRunning()
    );

    if (!hasRunningAnimations && this.isRunning) {
      this.stop();
    }
  }
}
