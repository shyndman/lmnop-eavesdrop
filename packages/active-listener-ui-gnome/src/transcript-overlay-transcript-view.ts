import Clutter from 'gi://Clutter';
import Pango from 'gi://Pango';
import St from 'gi://St';

import { buildTranscriptAttributeSpecs, type TranscriptDisplay } from './transcript-attributes.js';

type OverlayClutterText = Clutter.Text & {
  set_line_wrap(lineWrap: boolean): void;
  set_text(text: string): void;
  set_attributes(attrs: Pango.AttrList | null): void;
};

export type TranscriptOverlayTranscriptViewOptions = {
  offsetX: number;
  offsetY: number;
  width: number;
  minHeight: number;
  textColor: string;
  commandTextColor: string;
  fontFamily: string;
  fontSize: number;
  lineHeight: number;
};

export class TranscriptOverlayTranscriptView {
  readonly actor: St.Widget;

  private readonly overlayLabel: St.Label;
  private readonly minimumHeight: number;
  private readonly textWidth: number;
  private readonly textColor: string;
  private readonly commandTextColor: string;
  private installedTranscriptDisplay: TranscriptDisplay = {
    text: '',
    runs: [],
  };

  constructor(options: TranscriptOverlayTranscriptViewOptions) {
    this.minimumHeight = options.minHeight;
    this.textWidth = options.width;
    this.textColor = options.textColor;
    this.commandTextColor = options.commandTextColor;

    this.overlayLabel = new St.Label({
      text: '',
      width: options.width,
      style:
        `color: ${options.textColor};` +
        `font-family: ${options.fontFamily};` +
        `font-size: ${options.fontSize}px;` +
        `line-height: ${options.lineHeight};` +
        'font-weight: 500;' +
        'text-align: center;',
    });
    this.getOverlayClutterText()?.set_line_wrap(true);

    this.actor = new St.Widget({
      x: options.offsetX,
      y: options.offsetY,
      width: options.width,
      height: options.minHeight,
      layout_manager: new Clutter.FixedLayout(),
    });
    this.actor.set_clip_to_allocation(true);
    this.actor.add_child(this.overlayLabel);
  }

  setDisplay(transcriptDisplay: TranscriptDisplay): void {
    if (this.isInstalledTranscriptDisplay(transcriptDisplay)) {
      return;
    }

    const overlayClutterText = this.getOverlayClutterText();
    if (overlayClutterText === null) {
      return;
    }

    overlayClutterText.set_text(transcriptDisplay.text);
    this.applyTranscriptDisplayAttributes(transcriptDisplay);
    this.installedTranscriptDisplay = {
      text: transcriptDisplay.text,
      runs: transcriptDisplay.runs.map((run) => ({ ...run })),
    };
  }

  setHeight(height: number): void {
    this.actor.set_height(height);
  }

  measureHeight(): number {
    const [, naturalHeight] = this.overlayLabel.get_preferred_height(this.textWidth);
    return Math.max(this.minimumHeight, Math.ceil(naturalHeight));
  }

  private getOverlayClutterText(): OverlayClutterText | null {
    return this.overlayLabel.get_clutter_text() as OverlayClutterText | null;
  }

  private isInstalledTranscriptDisplay(transcriptDisplay: TranscriptDisplay): boolean {
    return (
      transcriptDisplay.text === this.installedTranscriptDisplay.text &&
      transcriptDisplay.runs.length === this.installedTranscriptDisplay.runs.length &&
      transcriptDisplay.runs.every((run, index) => {
        const installedRun = this.installedTranscriptDisplay.runs[index];
        return installedRun !== undefined &&
          run.text === installedRun.text &&
          run.isCommand === installedRun.isCommand &&
          run.isComplete === installedRun.isComplete &&
          run.startByte === installedRun.startByte &&
          run.endByte === installedRun.endByte;
      })
    );
  }

  private clearTranscriptAttributes(): void {
    this.getOverlayClutterText()?.set_attributes(null);
  }

  private applyTranscriptDisplayAttributes(transcriptDisplay: TranscriptDisplay): void {
    const overlayClutterText = this.getOverlayClutterText();
    if (overlayClutterText === null) {
      return;
    }

    const attributeSpecs = buildTranscriptAttributeSpecs(
      transcriptDisplay,
      this.textColor,
      this.commandTextColor,
    );
    if (attributeSpecs.length === 0) {
      this.clearTranscriptAttributes();
      return;
    }

    const attrs = Pango.AttrList.new();
    for (const spec of attributeSpecs) {
      const attr = spec.kind === 'foreground-color'
        ? Pango.attr_foreground_new(spec.red, spec.green, spec.blue)
        : Pango.attr_foreground_alpha_new(spec.alpha);
      attr.start_index = spec.startByte;
      attr.end_index = spec.endByte;
      attrs.insert(attr);
    }

    overlayClutterText.set_attributes(attrs);
  }
}
