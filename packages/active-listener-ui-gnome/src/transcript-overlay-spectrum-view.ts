import Clutter from 'gi://Clutter';
import St from 'gi://St';

export type TranscriptOverlaySpectrumViewOptions = {
  frameOffsetX: number;
  frameWidth: number;
  frameHeight: number;
  framePaddingVertical: number;
  framePaddingHorizontal: number;
  cornerRadius: number;
  barCount: number;
  barColor: string;
  barWidth: number;
  barGap: number;
  barMinHeight: number;
  barMaxHeight: number;
  barCornerRadius: number;
};

export class TranscriptOverlaySpectrumView {
  readonly actor: St.BoxLayout;
  readonly barCount: number;

  private readonly barMinimumHeight: number;
  private readonly barMaximumHeight: number;
  private readonly spectrumBars: St.Widget[] = [];
  private spectrumLevels: Uint8Array<ArrayBufferLike>;

  constructor(options: TranscriptOverlaySpectrumViewOptions) {
    this.barCount = options.barCount;
    this.barMinimumHeight = options.barMinHeight;
    this.barMaximumHeight = options.barMaxHeight;
    this.spectrumLevels = new Uint8Array(options.barCount);

    const spectrumBarStyle =
      `background-color: ${options.barColor};` +
      `border-radius: ${options.barCornerRadius}px;`;

    this.actor = new St.BoxLayout({
      x: options.frameOffsetX,
      width: options.frameWidth,
      height: options.frameHeight,
      clip_to_allocation: true,
      style:
        `border-radius: ${options.cornerRadius}px;` +
        `spacing: ${options.barGap}px;` +
        `padding: ${options.framePaddingVertical}px ${options.framePaddingHorizontal}px;`,
    });

    for (let index = 0; index < options.barCount; index += 1) {
      const bar = new St.Widget({
        reactive: false,
        can_focus: false,
        width: options.barWidth,
        height: options.barMinHeight,
        y_align: Clutter.ActorAlign.END,
        style: spectrumBarStyle,
      });
      this.spectrumBars.push(bar);
      this.actor.add_child(bar);
    }
  }

  clear(): void {
    this.spectrumLevels = new Uint8Array(this.barCount);
    this.renderBars();
  }

  setLevels(levels: Uint8Array): boolean {
    if (levels.length !== this.barCount) {
      return false;
    }

    this.spectrumLevels = Uint8Array.from(levels);
    this.renderBars();
    return true;
  }

  private renderBars(): void {
    for (let index = 0; index < this.barCount; index += 1) {
      const level = this.spectrumLevels[index] ?? 0;
      const normalizedLevel = level / 255;
      const height =
        this.barMinimumHeight +
        Math.round(normalizedLevel * (this.barMaximumHeight - this.barMinimumHeight));
      const bar = this.spectrumBars[index];
      bar.height = height;
    }
  }
}
