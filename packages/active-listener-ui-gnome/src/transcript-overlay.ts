import Clutter from 'gi://Clutter';
import GLib from 'gi://GLib';
import Pango from 'gi://Pango';
import St from 'gi://St';

import * as Main from 'resource:///org/gnome/shell/ui/main.js';

import {
  isTranscriptFrameLoggingEnabled,
  PANGO_ALPHA_MAX,
  setTranscriptFrameLoggingEnabled,
  TRANSCRIPT_FRAME_INTERVAL_MS,
  TRANSCRIPT_LOG_SAMPLE_INTERVAL,
  TranscriptAnimationController,
  type ByteAlphaRun,
} from './transcript-animation.js';
import { buildTranscriptAttributeSpecs } from './transcript-attributes.js';
import type {
  ActiveListenerServiceState,
  TranscriptionUpdate,
} from './active-listener-service-client.js';

const OVERLAY_MESSAGE = 'Overlay PoC';
const OVERLAY_DISPLAY_DURATION_MS = 2000;
const OVERLAY_BOTTOM_MARGIN_PX = 96;
const OVERLAY_ANIMATION_DURATION_MS = 180;
const OVERLAY_WIDTH_PX = 1000;
const OVERLAY_HEIGHT_PX = 120;
const OVERLAY_CORNER_RADIUS_PX = 32;
const OVERLAY_BACKGROUND_COLOR = 'rgba(11, 13, 16, 0.82)';
const OVERLAY_TEXT_OFFSET_X_PX = 36;
const OVERLAY_TEXT_OFFSET_Y_PX = 28;
const OVERLAY_TEXT_WIDTH_PX = OVERLAY_WIDTH_PX - OVERLAY_TEXT_OFFSET_X_PX * 2;
const OVERLAY_TEXT_HEIGHT_PX = 64;
const OVERLAY_TEXT_COLOR = '#F5F7FA';
const OVERLAY_FONT_FAMILY = 'Inter';
const OVERLAY_FONT_SIZE_PX = 24;
const OVERLAY_LINE_HEIGHT = 1.35;
const SPECTRUM_BAR_COUNT = 50;
const SPECTRUM_FRAME_OFFSET_X_PX = 11;
const SPECTRUM_FRAME_WIDTH_PX = 977;
const SPECTRUM_FRAME_PADDING_VERTICAL_PX = 12;
const SPECTRUM_FRAME_PADDING_HORIZONTAL_PX = 16;
const SPECTRUM_BAR_COLOR = '#241732';
const SPECTRUM_BAR_WIDTH_PX = 5;
const SPECTRUM_BAR_GAP_PX = 14;
const SPECTRUM_BAR_MAX_HEIGHT_PX = 97;
const SPECTRUM_BAR_MIN_HEIGHT_PX = 5;
const SPECTRUM_BAR_CORNER_RADIUS_PX = 7;
const SPECTRUM_BAR_STYLE =
  `background-color: ${SPECTRUM_BAR_COLOR};` +
  `border-radius: ${SPECTRUM_BAR_CORNER_RADIUS_PX}px;`;

type ActorEaseOptions = {
  duration: number;
  mode: Clutter.AnimationMode;
  opacity?: number;
  height?: number;
  y?: number;
  onComplete?: () => void;
};

type OverlayActor = St.Widget & {
  ease(options: ActorEaseOptions): void;
  remove_all_transitions(): void;
};

type OverlayMonitor = NonNullable<typeof Main.layoutManager.primaryMonitor>;

type OverlayClutterText = Clutter.Text & {
  set_line_wrap(lineWrap: boolean): void;
  set_text(text: string): void;
  set_attributes(attrs: Pango.AttrList | null): void;
};

type TranscriptSwapMeasurement = {
  transcriptHeight: number;
  shellHeight: number;
};

export class TranscriptOverlayController {
  private overlay: St.Widget | null = null;
  private overlayContent: St.Widget | null = null;
  private overlayLabel: St.Label | null = null;
  private transcriptClip: St.Widget | null = null;
  private spectrumFrame: St.BoxLayout | null = null;
  private spectrumBars: St.Widget[] = [];
  private spectrumLevels: Uint8Array<ArrayBufferLike> = new Uint8Array(SPECTRUM_BAR_COUNT);
  private overlayTimeoutId: number | null = null;
  private transcriptAnimationController: TranscriptAnimationController | null = null;
  private pendingTranscriptSwapMeasurement: TranscriptSwapMeasurement | null = null;
  private readonly overlayActor = (actor: St.Widget): OverlayActor => actor as OverlayActor;
  private serviceState: ActiveListenerServiceState = {
    indicatorState: 'absent',
    servicePresent: false,
    phase: null,
  };
  private completedTranscriptParts: string[] = [];
  private incompleteTranscriptText = '';
  private transcriptEventCounter = 0;
  private transcriptFrameCounter = 0;

  constructor() {
    this.createOverlay();
  }

  destroy(): void {
    this.clearOverlayTimeout();
    this.stopTranscriptAnimation();
    this.resetTranscriptOverlay();

    if (this.overlay !== null) {
      Main.layoutManager.removeChrome(this.overlay);
      this.overlay.destroy();
      this.overlay = null;
    }

    this.overlayContent = null;
    this.overlayLabel = null;
    this.transcriptClip = null;
    this.spectrumFrame = null;
    this.spectrumBars = [];
    this.spectrumLevels = new Uint8Array(SPECTRUM_BAR_COUNT);
    this.transcriptAnimationController = null;
    this.pendingTranscriptSwapMeasurement = null;
  }

  setServiceState(serviceState: ActiveListenerServiceState): void {
    this.logTranscriptOverlayEvent('updated service state', {
      servicePresent: serviceState.servicePresent,
      phase: serviceState.phase,
    });

    const previousState = this.serviceState.indicatorState;
    this.serviceState = serviceState;

    if (previousState === 'recording' && serviceState.indicatorState !== 'recording') {
      this.clearSpectrumBars();
      this.resetTranscriptOverlay();
      return;
    }

    if (previousState !== 'recording' && serviceState.indicatorState === 'recording') {
      this.clearSpectrumBars();
      this.resetTranscriptOverlay();
    }
  }

  showPreview(text: string = OVERLAY_MESSAGE): void {
    this.installTranscriptTextImmediately(text);
    this.showOverlay();
  }

  applyTranscriptionUpdate(update: TranscriptionUpdate): void {
    this.logTranscriptOverlayEvent('received transcription update', {
      completedSegments: update.completedSegments,
      incompleteSegment: update.incompleteSegment,
      completedTranscriptPartsBefore: [...this.completedTranscriptParts],
      incompleteTranscriptTextBefore: this.incompleteTranscriptText,
    });

    for (const { text } of update.completedSegments) {
      const normalizedText = text.trim();
      if (normalizedText.length > 0) {
        this.completedTranscriptParts.push(normalizedText);
      }
    }

    this.incompleteTranscriptText = update.incompleteSegment.text.trim();
    this.logTranscriptOverlayEvent('applied transcription update', {
      completedTranscriptPartsAfter: [...this.completedTranscriptParts],
      incompleteTranscriptTextAfter: this.incompleteTranscriptText,
    });
    this.renderOverlay();
  }

  applySpectrum(levels: Uint8Array): void {
    if (this.serviceState.indicatorState !== 'recording') {
      this.logTranscriptOverlayEvent('ignored spectrum update because indicator is not recording', {
        indicatorState: this.serviceState.indicatorState,
      });
      return;
    }

    this.spectrumLevels = levels;
    this.logTranscriptOverlayEvent('received spectrum update', {
      barCount: this.spectrumLevels.length,
      leadingBars: Array.from(this.spectrumLevels.slice(0, 8)),
    });

    if (this.spectrumLevels.length !== SPECTRUM_BAR_COUNT) {
      this.logTranscriptOverlayEvent('ignored spectrum update because bar count did not match', {
        expectedBarCount: SPECTRUM_BAR_COUNT,
        actualBarCount: this.spectrumLevels.length,
      });
      return;
    }

    this.renderSpectrumBars();
    this.renderOverlay();
  }

  private createOverlay(): void {
    if (this.overlay !== null) {
      return;
    }

    this.overlayLabel = new St.Label({
      text: OVERLAY_MESSAGE,
      width: OVERLAY_TEXT_WIDTH_PX,
      style:
        `color: ${OVERLAY_TEXT_COLOR};` +
        `font-family: ${OVERLAY_FONT_FAMILY};` +
        `font-size: ${OVERLAY_FONT_SIZE_PX}px;` +
        `line-height: ${OVERLAY_LINE_HEIGHT};` +
        'font-weight: 500;' +
        'text-align: center;',
    });
    this.getOverlayClutterText()?.set_line_wrap(true);

    this.transcriptClip = new St.Widget({
      x: OVERLAY_TEXT_OFFSET_X_PX,
      y: OVERLAY_TEXT_OFFSET_Y_PX,
      width: OVERLAY_TEXT_WIDTH_PX,
      height: OVERLAY_TEXT_HEIGHT_PX,
      layout_manager: new Clutter.FixedLayout(),
    });
    this.transcriptClip.set_clip_to_allocation(true);
    this.transcriptClip.add_child(this.overlayLabel);

    this.spectrumFrame = new St.BoxLayout({
      x: SPECTRUM_FRAME_OFFSET_X_PX,
      width: SPECTRUM_FRAME_WIDTH_PX,
      height: OVERLAY_HEIGHT_PX,
      clip_to_allocation: true,
      style:
        `border-radius: ${OVERLAY_CORNER_RADIUS_PX}px;` +
        `spacing: ${SPECTRUM_BAR_GAP_PX}px;` +
        `padding: ${SPECTRUM_FRAME_PADDING_VERTICAL_PX}px ${SPECTRUM_FRAME_PADDING_HORIZONTAL_PX}px;`,
    });
    this.spectrumBars = [];
    for (let index = 0; index < SPECTRUM_BAR_COUNT; index += 1) {
      const bar = new St.Widget({
        reactive: false,
        can_focus: false,
        width: SPECTRUM_BAR_WIDTH_PX,
        height: SPECTRUM_BAR_MIN_HEIGHT_PX,
        y_align: Clutter.ActorAlign.END,
        style: SPECTRUM_BAR_STYLE,
      });
      this.spectrumBars.push(bar);
      this.spectrumFrame.add_child(bar);
    }

    this.overlayContent = new St.Widget({
      width: OVERLAY_WIDTH_PX,
      layout_manager: new Clutter.FixedLayout(),
    });
    this.overlayContent.add_child(this.spectrumFrame);
    this.overlayContent.add_child(this.transcriptClip);

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
    this.overlay.add_child(this.overlayContent);

    this.transcriptAnimationController = new TranscriptAnimationController({
      installText: (text) => {
        this.installTranscriptText(text);
      },
      applyAlphaRuns: (runs) => {
        this.applyTranscriptAlphaRuns(runs);
      },
      clearAlphaRuns: () => {
        this.clearTranscriptAttributes();
      },
      beforeSwap: () => {
        this.captureTranscriptSwapMeasurement();
      },
      afterSwap: () => {
        this.startTranscriptSwapHeightTween();
      },
    });

    Main.layoutManager.addChrome(this.overlay, { trackFullscreen: true });
    this.overlay.set_position(-10000, -10000);

    this.clearSpectrumBars();
    this.installTranscriptTextImmediately(OVERLAY_MESSAGE);
  }

  private renderOverlay(): void {
    const completedTranscript = this.completedTranscriptParts.join(' ');
    const transcriptParts = [completedTranscript];
    if (this.incompleteTranscriptText.length > 0) {
      transcriptParts.push(this.incompleteTranscriptText);
    }

    const transcript = transcriptParts.filter((part) => part.length > 0).join(' ');
    this.logTranscriptOverlayEvent('render overlay requested', {
      indicatorState: this.serviceState.indicatorState,
      servicePresent: this.serviceState.servicePresent,
      servicePhase: this.serviceState.phase,
      completedTranscript: this.describeTranscriptTextForLogging(completedTranscript),
      incompleteTranscript: this.describeTranscriptTextForLogging(this.incompleteTranscriptText),
      combinedTranscript: this.describeTranscriptTextForLogging(transcript),
      controllerSnapshot: this.transcriptAnimationController?.getSnapshot() ?? null,
    });

    if (transcript.length === 0 && this.serviceState.indicatorState !== 'recording') {
      this.logTranscriptOverlayEvent('render overlay hiding because transcript is empty and indicator is not recording');
      this.installTranscriptTextImmediately('');
      this.hideOverlay();
      return;
    }

    const transitionController = this.transcriptAnimationController;
    if (transitionController !== null) {
      const nowMilliseconds = this.getNowMs();
      const isAnimating = transitionController.setCanonicalText(transcript, nowMilliseconds);
      this.logTranscriptOverlayEvent('submitted transcript to animation controller', {
        nowMilliseconds,
        isAnimating,
        controllerSnapshot: transitionController.getSnapshot(),
      });
      this.refreshTranscriptAnimation();
    }

    this.showOverlay(false);
  }

  private showOverlay(autoHide: boolean = true): void {
    if (this.overlay === null) {
      return;
    }

    const monitor = this.getOverlayMonitor();
    if (monitor === null) {
      this.logTranscriptOverlayEvent('show overlay skipped because no monitor is available');
      return;
    }

    this.clearOverlayTimeout();

    const x = Math.floor(monitor.x + (monitor.width - OVERLAY_WIDTH_PX) / 2);
    const y = this.getAnchoredOverlayY(this.overlay.height, monitor);
    this.logTranscriptOverlayEvent('show overlay requested', {
      autoHide,
      visible: this.overlay.visible,
      overlayHeight: this.overlay.height,
      x,
      y,
      monitor,
    });

    if (this.overlay.visible) {
      this.overlay.set_position(x, y);
      this.overlay.opacity = 255;
      this.logTranscriptOverlayEvent('updated visible overlay position', {
        autoHide,
        x,
        y,
        overlayHeight: this.overlay.height,
      });
      if (autoHide) {
        this.scheduleOverlayAutoHide();
      }
      return;
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
  }

  private hideOverlay(): void {
    if (this.overlay === null) {
      return;
    }

    this.logTranscriptOverlayEvent('hide overlay requested', {
      visible: this.overlay.visible,
      overlayHeight: this.overlay.height,
    });
    const overlayActor = this.overlayActor(this.overlay);
    overlayActor.remove_all_transitions();
    overlayActor.ease({
      opacity: 0,
      duration: OVERLAY_ANIMATION_DURATION_MS,
      mode: Clutter.AnimationMode.EASE_OUT_QUAD,
      onComplete: () => {
        this.logTranscriptOverlayEvent('hide overlay animation completed');
        this.overlay?.hide();
      },
    });
  }

  private scheduleOverlayAutoHide(): void {
    this.overlayTimeoutId = GLib.timeout_add(GLib.PRIORITY_DEFAULT, OVERLAY_DISPLAY_DURATION_MS, () => {
      this.hideOverlay();
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

  private resetTranscriptOverlay(): void {
    this.logTranscriptOverlayEvent('reset transcript overlay requested', {
      indicatorState: this.serviceState.indicatorState,
      completedTranscriptPartsBefore: [...this.completedTranscriptParts],
      incompleteTranscriptTextBefore: this.incompleteTranscriptText,
    });
    this.completedTranscriptParts = [];
    this.incompleteTranscriptText = '';
    this.installTranscriptTextImmediately('');

    if (this.serviceState.indicatorState === 'recording') {
      this.showOverlay(false);
      return;
    }

    this.hideOverlay();
  }

  private clearSpectrumBars(): void {
    this.spectrumLevels = new Uint8Array(SPECTRUM_BAR_COUNT);
    this.renderSpectrumBars();
  }

  private renderSpectrumBars(): void {
    if (this.spectrumBars.length !== SPECTRUM_BAR_COUNT) {
      return;
    }

    for (let index = 0; index < SPECTRUM_BAR_COUNT; index += 1) {
      const level = this.spectrumLevels[index] ?? 0;
      const normalizedLevel = level / 255;
      const height =
        SPECTRUM_BAR_MIN_HEIGHT_PX +
        Math.round(normalizedLevel * (SPECTRUM_BAR_MAX_HEIGHT_PX - SPECTRUM_BAR_MIN_HEIGHT_PX));
      const bar = this.spectrumBars[index];
      bar.height = height;
    }
  }

  private getOverlayClutterText(): OverlayClutterText | null {
    return this.overlayLabel?.get_clutter_text() as OverlayClutterText | null;
  }

  private installTranscriptText(text: string): void {
    const overlayClutterText = this.getOverlayClutterText();
    if (overlayClutterText === null) {
      this.logTranscriptOverlayEvent('install transcript text skipped because overlay text actor is unavailable', {
        text: this.describeTranscriptTextForLogging(text),
      });
      return;
    }

    this.logTranscriptOverlayEvent('installing transcript text', {
      text: this.describeTranscriptTextForLogging(text),
    });
    overlayClutterText.set_text(text);
  }

  private installTranscriptTextImmediately(text: string): void {
    this.logTranscriptOverlayEvent('install transcript text immediately requested', {
      text: this.describeTranscriptTextForLogging(text),
      controllerSnapshot: this.transcriptAnimationController?.getSnapshot() ?? null,
    });
    this.stopTranscriptAnimation();
    this.pendingTranscriptSwapMeasurement = null;

    if (this.transcriptAnimationController !== null) {
      this.transcriptAnimationController.installImmediate(text);
    } else {
      this.clearTranscriptAttributes();
      this.installTranscriptText(text);
    }

    this.syncOverlayHeightToInstalledText();
  }

  private refreshTranscriptAnimation(): void {
    const transitionController = this.transcriptAnimationController;
    if (transitionController === null) {
      this.logTranscriptOverlayEvent('refresh transcript animation skipped because controller is unavailable');
      return;
    }

    const currentFrameNumber = this.transcriptFrameCounter;
    const shouldContinue = this.runWithTranscriptFrameLogging(currentFrameNumber, () =>
      transitionController.tick(this.getNowMs()),
    );
    this.logTranscriptOverlayFrameEvent('refreshed transcript animation', {
      frameNumber: currentFrameNumber,
      shouldContinue,
      controllerSnapshot: transitionController.getSnapshot(),
    });
    this.transcriptFrameCounter += 1;
    if (!shouldContinue) {
      this.stopTranscriptAnimation();
      return;
    }

    if (transitionController.getSnapshot().frameSourceId !== null) {
      this.logTranscriptOverlayFrameEvent('transcript animation frame source already exists', {
        frameNumber: currentFrameNumber,
        frameSourceId: transitionController.getSnapshot().frameSourceId,
      });
      return;
    }

    const sourceId = GLib.timeout_add(GLib.PRIORITY_DEFAULT, TRANSCRIPT_FRAME_INTERVAL_MS, () => {
      const controller = this.transcriptAnimationController;
      if (controller === null) {
        this.logTranscriptOverlayFrameEvent('transcript animation frame source stopped because controller disappeared');
        return GLib.SOURCE_REMOVE;
      }

      const frameNumber = this.transcriptFrameCounter;
      const keepRunning = this.runWithTranscriptFrameLogging(frameNumber, () => controller.tick(this.getNowMs()));
      this.logTranscriptOverlayFrameEvent('transcript animation frame source ticked', {
        frameNumber,
        keepRunning,
        controllerSnapshot: controller.getSnapshot(),
      });
      this.transcriptFrameCounter += 1;
      if (!keepRunning) {
        controller.setFrameSourceId(null);
        this.logTranscriptOverlayFrameEvent('transcript animation frame source completed', {
          frameNumber,
        });
        return GLib.SOURCE_REMOVE;
      }

      return GLib.SOURCE_CONTINUE;
    });
    transitionController.setFrameSourceId(sourceId);
    this.logTranscriptOverlayFrameEvent('scheduled transcript animation frame source', {
      frameNumber: currentFrameNumber,
      sourceId,
    });
  }

  private stopTranscriptAnimation(): void {
    const transitionController = this.transcriptAnimationController;
    if (transitionController === null) {
      return;
    }

    const frameSourceId = transitionController.getSnapshot().frameSourceId;
    if (frameSourceId !== null) {
      this.logTranscriptOverlayFrameEvent('stopping transcript animation frame source', {
        frameSourceId,
      });
      GLib.Source.remove(frameSourceId);
      transitionController.setFrameSourceId(null);
    }
  }

  private clearTranscriptAttributes(): void {
    this.logTranscriptOverlayFrameEvent('clearing transcript attributes');
    this.getOverlayClutterText()?.set_attributes(null);
  }

  private applyTranscriptAlphaRuns(runs: ByteAlphaRun[]): void {
    const overlayClutterText = this.getOverlayClutterText();
    if (overlayClutterText === null) {
      this.logTranscriptOverlayFrameEvent('skipped transcript attribute application because overlay text actor is unavailable', {
        runs,
      });
      return;
    }

    if (runs.length === 0) {
      this.logTranscriptOverlayFrameEvent('clearing transcript attributes because there are no alpha runs');
      this.clearTranscriptAttributes();
      return;
    }

    const attrs = Pango.AttrList.new();
    const attributeSpecs = buildTranscriptAttributeSpecs(overlayClutterText.get_text(), runs, OVERLAY_TEXT_COLOR);
    this.logTranscriptOverlayFrameEvent('applying transcript attributes', {
      text: this.describeTranscriptTextForLogging(overlayClutterText.get_text()),
      runs,
      attributeSpecs,
    });
    for (const spec of attributeSpecs) {
      const attr = spec.kind === 'foreground-color'
        ? Pango.attr_foreground_new(spec.red, spec.green, spec.blue)
        : Pango.attr_foreground_alpha_new(Math.max(0, Math.min(spec.alpha, PANGO_ALPHA_MAX)));
      attr.start_index = spec.startByte;
      attr.end_index = spec.endByte;
      attrs.insert(attr);
    }

    overlayClutterText.set_attributes(attrs);
  }

  private captureTranscriptSwapMeasurement(): void {
    if (this.transcriptClip === null || this.overlay === null) {
      this.logTranscriptOverlayFrameEvent('capture transcript swap measurement skipped because overlay actors are unavailable');
      return;
    }

    const transcriptHeight = this.measureTranscriptHeight();
    const shellHeight = this.measureOverlayShellHeight(transcriptHeight);
    this.pendingTranscriptSwapMeasurement = {
      transcriptHeight,
      shellHeight,
    };
    this.logTranscriptOverlayFrameEvent('captured transcript swap measurement', {
      transcriptHeight,
      shellHeight,
    });

    const clipActor = this.overlayActor(this.transcriptClip);
    clipActor.remove_all_transitions();
    this.transcriptClip.set_height(transcriptHeight);

    const overlayActor = this.overlayActor(this.overlay);
    overlayActor.remove_all_transitions();
    this.overlay.set_height(shellHeight);
    this.overlay.set_y(this.getAnchoredOverlayY(shellHeight));
  }

  private startTranscriptSwapHeightTween(): void {
    if (this.transcriptClip === null || this.overlay === null) {
      this.logTranscriptOverlayFrameEvent('start transcript swap height tween skipped because overlay actors are unavailable');
      return;
    }

    const previousMeasurement = this.pendingTranscriptSwapMeasurement;
    this.pendingTranscriptSwapMeasurement = null;
    if (previousMeasurement === null) {
      this.logTranscriptOverlayFrameEvent('start transcript swap height tween fell back to direct sync because there was no pending measurement');
      this.syncOverlayHeightToInstalledText();
      return;
    }

    const nextTranscriptHeight = this.measureTranscriptHeight();
    const nextShellHeight = this.measureOverlayShellHeight(nextTranscriptHeight);
    this.logTranscriptOverlayFrameEvent('starting transcript swap height tween', {
      previousMeasurement,
      nextTranscriptHeight,
      nextShellHeight,
    });

    const clipActor = this.overlayActor(this.transcriptClip);
    clipActor.remove_all_transitions();
    this.transcriptClip.set_height(previousMeasurement.transcriptHeight);
    clipActor.ease({
      height: nextTranscriptHeight,
      duration: OVERLAY_ANIMATION_DURATION_MS,
      mode: Clutter.AnimationMode.EASE_OUT_QUAD,
    });

    const overlayActor = this.overlayActor(this.overlay);
    overlayActor.remove_all_transitions();
    this.overlay.set_height(previousMeasurement.shellHeight);
    this.overlay.set_y(this.getAnchoredOverlayY(previousMeasurement.shellHeight));
    overlayActor.ease({
      height: nextShellHeight,
      y: this.getAnchoredOverlayY(nextShellHeight),
      duration: OVERLAY_ANIMATION_DURATION_MS,
      mode: Clutter.AnimationMode.EASE_OUT_QUAD,
    });
  }

  private syncOverlayHeightToInstalledText(): void {
    if (this.transcriptClip === null || this.overlay === null) {
      this.logTranscriptOverlayFrameEvent('sync overlay height skipped because overlay actors are unavailable');
      return;
    }

    const transcriptHeight = this.measureTranscriptHeight();
    const shellHeight = this.measureOverlayShellHeight(transcriptHeight);
    this.logTranscriptOverlayFrameEvent('syncing overlay height to installed text', {
      transcriptHeight,
      shellHeight,
      overlayVisible: this.overlay.visible,
    });

    this.overlayActor(this.transcriptClip).remove_all_transitions();
    this.transcriptClip.set_height(transcriptHeight);

    this.overlayActor(this.overlay).remove_all_transitions();
    this.overlay.set_height(shellHeight);

    if (this.overlay.visible) {
      this.overlay.set_y(this.getAnchoredOverlayY(shellHeight));
    }
  }

  private measureTranscriptHeight(): number {
    if (this.overlayLabel === null) {
      this.logTranscriptOverlayFrameEvent('measure transcript height used fallback because overlay label is unavailable', {
        fallbackHeight: OVERLAY_TEXT_HEIGHT_PX,
      });
      return OVERLAY_TEXT_HEIGHT_PX;
    }

    const [, naturalHeight] = this.overlayLabel.get_preferred_height(OVERLAY_TEXT_WIDTH_PX);
    const measuredHeight = Math.max(OVERLAY_TEXT_HEIGHT_PX, Math.ceil(naturalHeight));
    this.logTranscriptOverlayFrameEvent('measured transcript height', {
      naturalHeight,
      measuredHeight,
      installedText: this.describeTranscriptTextForLogging(this.overlayLabel.get_text()),
    });
    return measuredHeight;
  }

  private measureOverlayShellHeight(transcriptHeight: number): number {
    return Math.max(OVERLAY_HEIGHT_PX, transcriptHeight + OVERLAY_TEXT_OFFSET_Y_PX * 2);
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

  private getNowMs(): number {
    return GLib.get_monotonic_time() / 1000;
  }

  private logTranscriptOverlayEvent(message: string, details: Record<string, unknown> = {}): void {
    if (this.transcriptEventCounter % TRANSCRIPT_LOG_SAMPLE_INTERVAL !== 0) {
      this.transcriptEventCounter += 1;
      return;
    }

    this.transcriptEventCounter += 1;
    console.error(`Active Listener transcript overlay ${message}`, details);
  }

  private logTranscriptOverlayFrameEvent(message: string, details: Record<string, unknown> = {}): void {
    if (!isTranscriptFrameLoggingEnabled()) {
      return;
    }

    console.error(`Active Listener transcript overlay ${message}`, details);
  }

  private runWithTranscriptFrameLogging<T>(frameNumber: number, callback: () => T): T {
    const shouldEnableFrameLogging = frameNumber % TRANSCRIPT_LOG_SAMPLE_INTERVAL === 0;
    setTranscriptFrameLoggingEnabled(shouldEnableFrameLogging);
    try {
      return callback();
    } finally {
      setTranscriptFrameLoggingEnabled(false);
    }
  }

  private describeTranscriptTextForLogging(text: string): Record<string, unknown> {
    const trimmedText = text.trim();
    return {
      text,
      characterCount: text.length,
      wordCount: trimmedText.length === 0 ? 0 : trimmedText.split(/\s+/u).length,
    };
  }
}
