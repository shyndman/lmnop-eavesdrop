import { AnimatedValue, Animation, Easing } from './animation';

const FRAME_BACKGROUND_COLOR = 'rgba(26, 26, 40, 0.87)';
const FRAME_BORDER_COLOR = 'rgba(50, 50, 70, 0.87)';
const COMMAND_BACKGROUND_COLOR = 'rgba(38, 08, 08, 0.2)';
const COMMAND_BORDER_COLOR = 'rgba(120, 30, 30, 0.6)';

export class FrameRenderer {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private canvasWidth: number = 0;
  private canvasHeight: number = 0;

  private asrState: HTMLElement;
  private commandPrompt: HTMLElement;
  private overlayLayer: HTMLElement;
  private frameLayer: HTMLElement;
  private transcription: HTMLElement;
  private transcriptionOverlay: HTMLElement;
  private animation: Animation<{ frameHeight: AnimatedValue; commandPromptHeight: AnimatedValue }>;
  private resizeObserver: ResizeObserver;

  constructor() {
    const canvas = document.querySelector<HTMLCanvasElement>('#frame-layer canvas');
    const frameLayer = document.getElementById('frame-layer');
    const asrState = document.getElementById('asr-state');
    const commandPrompt = document.getElementById('command');
    const overlayLayer = document.getElementById('overlay-layer');
    const transcriptionOverlay = document.getElementById('transcription-overlay');
    const transcription = document.getElementById('transcription');

    if (
      !canvas ||
      !commandPrompt ||
      !frameLayer ||
      !asrState ||
      !overlayLayer ||
      !transcription ||
      !transcriptionOverlay
    ) {
      throw new Error('Canvas, frame-layer, overlay-layer, or asr-state not found');
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) {
      throw new Error('Could not get 2D context');
    }

    this.canvas = canvas;
    this.ctx = ctx;
    this.frameLayer = frameLayer;
    this.overlayLayer = overlayLayer;
    this.asrState = asrState;
    this.commandPrompt = commandPrompt;
    this.transcriptionOverlay = transcriptionOverlay;
    this.transcription = transcription;

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
    const initialFrameHeight = asrState.getBoundingClientRect().height;
    const frameHeightAnimation = new AnimatedValue(initialFrameHeight, 300, Easing.easeOut);
    const initialCommandPromptHeight = commandPrompt.getBoundingClientRect().height;
    const commandPromptAnimation = new AnimatedValue(
      initialCommandPromptHeight,
      300,
      Easing.easeOut,
    );

    this.animation = new Animation(
      { frameHeight: frameHeightAnimation, commandPromptHeight: commandPromptAnimation },
      ({ frameHeight, commandPromptHeight }) => this.draw(frameHeight, commandPromptHeight),
    );

    // Set up ResizeObserver to detect height changes
    this.resizeObserver = new ResizeObserver((entries) => {
      this.handleResize(entries.map((el) => el.target));
    });
    this.resizeObserver.observe(asrState);
    this.resizeObserver.observe(commandPrompt);

    // Initial render
    this.draw(initialFrameHeight, initialCommandPromptHeight);
  }

  private handleResize(resizedElements: Element[]): void {
    const asrResized = resizedElements.some((el) => el === this.asrState);
    const commandResized = resizedElements.some((el) => el === this.commandPrompt);

    if (asrResized) {
      const measuredFrameHeight = this.asrState.getBoundingClientRect().height;
      const transcriptionHeight = this.transcription.getBoundingClientRect().height;
      this.animation.getAnimatedValue('frameHeight').setTarget(measuredFrameHeight);
      this.overlayLayer.style.height = `${measuredFrameHeight}px`;
      this.transcriptionOverlay.style.height = `${transcriptionHeight}px`;
    }

    if (commandResized) {
      const measuredCommandPromptHeight = this.commandPrompt.getBoundingClientRect().height;
      this.animation.getAnimatedValue('commandPromptHeight').setTarget(measuredCommandPromptHeight);
    }
  }

  private draw(frameHeight: number, commandPromptHeight: number): void {
    if (!this.canvas || !this.ctx) return;

    this.ctx.clearRect(0, 0, this.canvasWidth, this.canvasHeight);

    const asrRect = this.asrState.getBoundingClientRect();
    const commandRect = this.commandPrompt.getBoundingClientRect();
    const frameRect = this.frameLayer.getBoundingClientRect();

    const x = asrRect.left - frameRect.left;
    const y = asrRect.top - frameRect.top;
    const width = asrRect.width;

    this.ctx.fillStyle = FRAME_BACKGROUND_COLOR;
    this.ctx.strokeStyle = FRAME_BORDER_COLOR;
    this.ctx.lineWidth = 2;
    this.ctx.beginPath();
    this.ctx.roundRect(x, y, width, frameHeight, 12);
    this.ctx.fill();
    this.ctx.stroke();

    const promptY = y + frameHeight - (asrRect.bottom - commandRect.bottom) - commandPromptHeight;

    this.ctx.fillStyle = COMMAND_BACKGROUND_COLOR;
    this.ctx.strokeStyle = COMMAND_BORDER_COLOR;
    this.ctx.lineWidth = 2;
    this.ctx.beginPath();
    this.ctx.roundRect(commandRect.x, promptY, commandRect.width, commandPromptHeight, 12);
    this.ctx.fill();
    this.ctx.stroke();
  }
}
