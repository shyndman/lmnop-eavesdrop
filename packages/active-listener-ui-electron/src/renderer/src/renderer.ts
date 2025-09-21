class Renderer {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private canvasWidth: number = 0;
  private canvasHeight: number = 0;

  constructor() {
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

    this.startRenderLoop();
  }

  private drawFrame(): void {
    if (!this.canvas || !this.ctx) return;

    const asrState = document.getElementById('asr-state');
    const frameLayer = document.getElementById('frame-layer');

    if (!asrState || !frameLayer) return;

    this.ctx.clearRect(0, 0, this.canvasWidth, this.canvasHeight);

    const asrRect = asrState.getBoundingClientRect();
    const frameRect = frameLayer.getBoundingClientRect();

    const x = asrRect.left - frameRect.left;
    const y = asrRect.top - frameRect.top;
    const width = asrRect.width;
    const height = asrRect.height;

    this.ctx.fillStyle = 'rgba(22, 25, 27, 0.87)';
    this.ctx.beginPath();
    this.ctx.roundRect(x, y, width, height, 8);
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
