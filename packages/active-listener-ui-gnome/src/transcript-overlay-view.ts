import Clutter from 'gi://Clutter';
import GLib from 'gi://GLib';
import St from 'gi://St';

import * as Main from 'resource:///org/gnome/shell/ui/main.js';

import type { TranscriptDisplay } from './transcript-attributes.js';
import { TranscriptOverlaySpectrumView } from './transcript-overlay-spectrum-view.js';
import { TranscriptOverlayTranscriptView } from './transcript-overlay-transcript-view.js';

type ActorEaseOptions = {
  duration: number;
  mode: Clutter.AnimationMode;
  opacity?: number;
  y?: number;
  onComplete?: () => void;
};

type OverlayActor = St.Widget & {
  ease(options: ActorEaseOptions): void;
  remove_all_transitions(): void;
};

type OverlayMonitor = NonNullable<typeof Main.layoutManager.primaryMonitor>;

const OVERLAY_DISPLAY_DURATION_MS = 2000;
const OVERLAY_BOTTOM_MARGIN_PX = 96;
const OVERLAY_ANIMATION_DURATION_MS = 180;
const OVERLAY_WIDTH_PX = 720;
const OVERLAY_HEIGHT_PX = 72;
const OVERLAY_CORNER_RADIUS_PX = 20;
const OVERLAY_BACKGROUND_COLOR = 'rgba(11, 13, 16, 0.82)';
const OVERLAY_TEXT_OFFSET_X_PX = 24;
const OVERLAY_TEXT_OFFSET_Y_PX = 16;
const OVERLAY_TEXT_WIDTH_PX = OVERLAY_WIDTH_PX - OVERLAY_TEXT_OFFSET_X_PX * 2;
const OVERLAY_TEXT_HEIGHT_PX = 40;
const OVERLAY_TEXT_COLOR = '#F5F7FA';
const OVERLAY_COMMAND_TEXT_COLOR = '#C4B5FD';
const OVERLAY_FONT_FAMILY = 'Inter';
const OVERLAY_FONT_SIZE_PX = 18;
const OVERLAY_LINE_HEIGHT = 1.3;
const SPECTRUM_BAR_COUNT = 50;
const SPECTRUM_FRAME_OFFSET_X_PX = 8;
const SPECTRUM_FRAME_WIDTH_PX = 704;
const SPECTRUM_FRAME_PADDING_VERTICAL_PX = 8;
const SPECTRUM_FRAME_PADDING_HORIZONTAL_PX = 12;
const SPECTRUM_BAR_COLOR = '#2B1D3D';
const SPECTRUM_BAR_WIDTH_PX = 4;
const SPECTRUM_BAR_GAP_PX = 7;
const SPECTRUM_BAR_MAX_HEIGHT_PX = 56;
const SPECTRUM_BAR_MIN_HEIGHT_PX = SPECTRUM_BAR_WIDTH_PX;
const SPECTRUM_BAR_CORNER_RADIUS_PX = SPECTRUM_BAR_WIDTH_PX / 2;

export class TranscriptOverlayView {
  readonly spectrumBarCount = SPECTRUM_BAR_COUNT;

  private readonly transcriptView: TranscriptOverlayTranscriptView;
  private readonly spectrumView: TranscriptOverlaySpectrumView;
  private readonly overlay: St.Widget;
  private overlayTimeoutId: number | null = null;

  constructor() {
    this.transcriptView = new TranscriptOverlayTranscriptView({
      offsetX: OVERLAY_TEXT_OFFSET_X_PX,
      offsetY: OVERLAY_TEXT_OFFSET_Y_PX,
      width: OVERLAY_TEXT_WIDTH_PX,
      minHeight: OVERLAY_TEXT_HEIGHT_PX,
      textColor: OVERLAY_TEXT_COLOR,
      commandTextColor: OVERLAY_COMMAND_TEXT_COLOR,
      fontFamily: OVERLAY_FONT_FAMILY,
      fontSize: OVERLAY_FONT_SIZE_PX,
      lineHeight: OVERLAY_LINE_HEIGHT,
    });

    this.spectrumView = new TranscriptOverlaySpectrumView({
      frameOffsetX: SPECTRUM_FRAME_OFFSET_X_PX,
      frameWidth: SPECTRUM_FRAME_WIDTH_PX,
      frameHeight: OVERLAY_HEIGHT_PX,
      framePaddingVertical: SPECTRUM_FRAME_PADDING_VERTICAL_PX,
      framePaddingHorizontal: SPECTRUM_FRAME_PADDING_HORIZONTAL_PX,
      cornerRadius: OVERLAY_CORNER_RADIUS_PX,
      barCount: SPECTRUM_BAR_COUNT,
      barColor: SPECTRUM_BAR_COLOR,
      barWidth: SPECTRUM_BAR_WIDTH_PX,
      barGap: SPECTRUM_BAR_GAP_PX,
      barMinHeight: SPECTRUM_BAR_MIN_HEIGHT_PX,
      barMaxHeight: SPECTRUM_BAR_MAX_HEIGHT_PX,
      barCornerRadius: SPECTRUM_BAR_CORNER_RADIUS_PX,
    });

    const overlayContent = new St.Widget({
      width: OVERLAY_WIDTH_PX,
      layout_manager: new Clutter.FixedLayout(),
    });
    overlayContent.add_child(this.spectrumView.actor);
    overlayContent.add_child(this.transcriptView.actor);

    this.overlay = new St.Widget({
      visible: false,
      reactive: false,
      can_focus: false,
      opacity: 0,
      width: OVERLAY_WIDTH_PX,
      height: OVERLAY_HEIGHT_PX,
      clip_to_allocation: true,
      layout_manager: new Clutter.FixedLayout(),
      style:
        `background-color: ${OVERLAY_BACKGROUND_COLOR};` +
        `border-radius: ${OVERLAY_CORNER_RADIUS_PX}px;`,
    });
    this.overlay.add_child(overlayContent);

    Main.layoutManager.addChrome(this.overlay, { trackFullscreen: true });
    this.overlay.set_position(-10000, -10000);

    this.clearSpectrum();
  }

  destroy(): void {
    this.clearOverlayTimeout();
    Main.layoutManager.removeChrome(this.overlay);
    this.overlay.destroy();
  }

  setTranscriptDisplay(transcriptDisplay: TranscriptDisplay): void {
    this.transcriptView.setDisplay(transcriptDisplay);
    this.syncOverlayHeightToInstalledText();
  }

  clearSpectrum(): void {
    this.spectrumView.clear();
  }

  setSpectrum(levels: Uint8Array): boolean {
    return this.spectrumView.setLevels(levels);
  }

  show(autoHide: boolean = true): boolean {
    this.clearOverlayTimeout();

    const monitor = this.getOverlayMonitor();
    if (monitor === null) {
      return false;
    }

    const x = Math.floor(monitor.x + (monitor.width - OVERLAY_WIDTH_PX) / 2);
    const y = this.getAnchoredOverlayY(this.overlay.height, monitor);

    if (this.overlay.visible) {
      this.overlay.set_position(x, y);
      this.overlay.opacity = 255;
      if (autoHide) {
        this.scheduleOverlayAutoHide();
      }
      return true;
    }

    const overlayActor = this.overlayActor(this.overlay);
    overlayActor.remove_all_transitions();
    this.overlay.set_position(x, y);
    this.overlay.opacity = 0;
    this.overlay.show();
    overlayActor.ease({
      opacity: 255,
      duration: OVERLAY_ANIMATION_DURATION_MS,
      mode: Clutter.AnimationMode.EASE_OUT_QUAD,
    });

    if (autoHide) {
      this.scheduleOverlayAutoHide();
    }

    return true;
  }

  hide(): void {
    const overlayActor = this.overlayActor(this.overlay);
    overlayActor.remove_all_transitions();
    overlayActor.ease({
      opacity: 0,
      duration: OVERLAY_ANIMATION_DURATION_MS,
      mode: Clutter.AnimationMode.EASE_OUT_QUAD,
      onComplete: () => {
        this.overlay.hide();
      },
    });
  }

  private overlayActor(actor: St.Widget): OverlayActor {
    return actor as OverlayActor;
  }

  private scheduleOverlayAutoHide(): void {
    this.overlayTimeoutId = GLib.timeout_add(GLib.PRIORITY_DEFAULT, OVERLAY_DISPLAY_DURATION_MS, () => {
      this.hide();
      this.overlayTimeoutId = null;
      return GLib.SOURCE_REMOVE;
    });
  }

  private clearOverlayTimeout(): void {
    if (this.overlayTimeoutId === null) {
      return;
    }

    GLib.Source.remove(this.overlayTimeoutId);
    this.overlayTimeoutId = null;
  }

  private syncOverlayHeightToInstalledText(): void {
    const transcriptHeight = this.transcriptView.measureHeight();
    const shellHeight = Math.max(OVERLAY_HEIGHT_PX, transcriptHeight + OVERLAY_TEXT_OFFSET_Y_PX * 2);

    this.overlayActor(this.transcriptView.actor).remove_all_transitions();
    this.transcriptView.setHeight(transcriptHeight);

    this.overlayActor(this.overlay).remove_all_transitions();
    this.overlay.set_height(shellHeight);

    if (this.overlay.visible) {
      this.overlay.set_y(this.getAnchoredOverlayY(shellHeight));
    }
  }

  private getOverlayMonitor(): OverlayMonitor | null {
    const focusedWindow = global.display.get_focus_window();
    const focusedMonitor = focusedWindow === null
      ? undefined
      : Main.layoutManager.monitors[focusedWindow.get_monitor()];

    return focusedMonitor ?? Main.layoutManager.primaryMonitor;
  }

  private getAnchoredOverlayY(height: number, monitor: OverlayMonitor | null = this.getOverlayMonitor()): number {
    if (monitor === null) {
      return -10000;
    }

    return Math.floor(monitor.y + monitor.height - height - OVERLAY_BOTTOM_MARGIN_PX);
  }
}
