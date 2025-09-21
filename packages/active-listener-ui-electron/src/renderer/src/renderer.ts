import { AnimatedValue, Easing } from './animation';

class Renderer {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private canvasWidth: number = 0;
  private canvasHeight: number = 0;

  private heightAnimation: AnimatedValue;

  constructor(onAnimationComplete?: () => void) {
    const canvas = document.querySelector<HTMLCanvasElement>('#frame-layer canvas');
    const frameLayer = document.getElementById('frame-layer');

    if (!canvas || !frameLayer) {
      throw new Error('Canvas or frame-layer not found');
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) {
      throw new Error('Could not get 2D context');
    }

    this.canvas = canvas;
    this.ctx = ctx;

    const dpr = window.devicePixelRatio || 1;
    const rect = frameLayer.getBoundingClientRect();

    this.canvasWidth = rect.width;
    this.canvasHeight = rect.height;

    this.canvas.width = this.canvasWidth * dpr;
    this.canvas.height = this.canvasHeight * dpr;

    this.canvas.style.width = `${this.canvasWidth}px`;
    this.canvas.style.height = `${this.canvasHeight}px`;

    this.ctx.scale(dpr, dpr);

    // Initialize height animation
    const asrState = document.getElementById('asr-state');
    const initialHeight = asrState ? asrState.getBoundingClientRect().height : 0;

    this.heightAnimation = new AnimatedValue(
      initialHeight,
      300,
      Easing.easeOut,
      onAnimationComplete
    );

    this.startRenderLoop();
  }

  private drawFrame(): void {
    if (!this.canvas || !this.ctx) return;

    const asrState = document.getElementById('asr-state');
    const frameLayer = document.getElementById('frame-layer');

    if (!asrState || !frameLayer) return;

    // Update height animation
    const actualHeight = asrState.getBoundingClientRect().height;
    this.heightAnimation.setTarget(actualHeight);
    const currentHeight = this.heightAnimation.update();

    this.ctx.clearRect(0, 0, this.canvasWidth, this.canvasHeight);

    const asrRect = asrState.getBoundingClientRect();
    const frameRect = frameLayer.getBoundingClientRect();

    const x = asrRect.left - frameRect.left;
    const y = asrRect.top - frameRect.top;
    const width = asrRect.width;
    const height = currentHeight;

    this.ctx.fillStyle = 'rgba(32, 32, 48, 0.8)';
    this.ctx.beginPath();
    this.ctx.roundRect(x, y, width, height, 12);
    this.ctx.fill();
  }

  private startRenderLoop(): void {
    const render = (): void => {
      this.drawFrame();
      requestAnimationFrame(render);
    };

    requestAnimationFrame(render);
  }
}

new Renderer();
