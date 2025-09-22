import { AnimatedValue, Animation, Easing } from './animation';

export class FrameRenderer {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private canvasWidth: number = 0;
  private canvasHeight: number = 0;

  private asrState: HTMLElement;
  private frameLayer: HTMLElement;
  private animation: Animation<{ height: AnimatedValue }>;
  private resizeObserver: ResizeObserver;

  constructor() {
    const canvas = document.querySelector<HTMLCanvasElement>(
      '#frame-layer canvas',
    );
    const frameLayer = document.getElementById('frame-layer');
    const asrState = document.getElementById('asr-state');

    if (!canvas || !frameLayer || !asrState) {
      throw new Error('Canvas, frame-layer, or asr-state not found');
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) {
      throw new Error('Could not get 2D context');
    }

    this.canvas = canvas;
    this.ctx = ctx;
    this.frameLayer = frameLayer;
    this.asrState = asrState;

    const dpr = window.devicePixelRatio || 1;
    const rect = frameLayer.getBoundingClientRect();

    this.canvasWidth = rect.width;
    this.canvasHeight = rect.height;

    this.canvas.width = this.canvasWidth * dpr;
    this.canvas.height = this.canvasHeight * dpr;

    this.canvas.style.width = `${this.canvasWidth}px`;
    this.canvas.style.height = `${this.canvasHeight}px`;

    this.ctx.scale(dpr, dpr);

    // Initialize animation system
    const initialHeight = asrState.getBoundingClientRect().height;
    const heightAnimation = new AnimatedValue(
      initialHeight,
      300,
      Easing.easeOut,
    );

    this.animation = new Animation(
      { height: heightAnimation },
      ({ height: height }) => this.draw(height),
    );

    // Set up ResizeObserver to detect height changes
    this.resizeObserver = new ResizeObserver(() => {
      this.handleResize();
    });
    this.resizeObserver.observe(asrState);

    // Initial render
    this.draw(initialHeight);
  }

  private handleResize(): void {
    const actualHeight = this.asrState.getBoundingClientRect().height;
    this.animation.getAnimatedValue('height').setTarget(actualHeight);
  }

  private draw(height: number): void {
    if (!this.canvas || !this.ctx) return;

    this.ctx.clearRect(0, 0, this.canvasWidth, this.canvasHeight);

    const asrRect = this.asrState.getBoundingClientRect();
    const frameRect = this.frameLayer.getBoundingClientRect();

    const x = asrRect.left - frameRect.left;
    const y = asrRect.top - frameRect.top;
    const width = asrRect.width;

    this.ctx.fillStyle = 'rgba(32, 32, 48, 0.7)';
    this.ctx.beginPath();
    this.ctx.roundRect(x, y, width, height, 12);
    this.ctx.fill();
  }
}
