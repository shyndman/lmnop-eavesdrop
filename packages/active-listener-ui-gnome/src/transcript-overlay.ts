import {
  buildTranscriptDisplay,
  normalizeTranscriptRuns,
  type TranscriptRun,
  type TranscriptDisplay,
} from './transcript-attributes.js';
import type {
  ActiveListenerServiceState,
  TranscriptionUpdate,
} from './active-listener-service-client.js';
import { TranscriptOverlayView } from './transcript-overlay-view.js';

const OVERLAY_MESSAGE = 'Overlay PoC';
const TRANSCRIPT_LOG_SAMPLE_INTERVAL = 180;

export class TranscriptOverlayController {
  private readonly view: TranscriptOverlayView;
  private serviceState: ActiveListenerServiceState = {
    indicatorState: 'absent',
    servicePresent: false,
    phase: null,
  };
  private transcriptRuns: TranscriptRun[] = [];
  private transcriptDisplay: TranscriptDisplay = {
    text: '',
    runs: [],
  };
  private transcriptEventCounter = 0;

  constructor() {
    this.view = new TranscriptOverlayView();
    this.view.setTranscriptDisplay(
      buildTranscriptDisplay([{ text: OVERLAY_MESSAGE, isCommand: false, isComplete: true }]),
    );
  }

  destroy(): void {
    this.resetTranscriptState();
    this.view.destroy();
  }

  setServiceState(serviceState: ActiveListenerServiceState): void {
    this.logTranscriptOverlayEvent('updated service state', {
      servicePresent: serviceState.servicePresent,
      phase: serviceState.phase,
    });

    const previousState = this.serviceState.indicatorState;
    this.serviceState = serviceState;

    if (previousState === 'recording' && serviceState.indicatorState !== 'recording') {
      this.view.clearSpectrum();
      this.resetTranscriptOverlay();
      return;
    }

    if (previousState !== 'recording' && serviceState.indicatorState === 'recording') {
      this.view.clearSpectrum();
      this.resetTranscriptOverlay();
    }
  }

  showPreview(text: string = OVERLAY_MESSAGE): void {
    this.view.setTranscriptDisplay(
      buildTranscriptDisplay([{ text, isCommand: false, isComplete: true }]),
    );
    this.view.show();
  }

  applyTranscriptionUpdate(update: TranscriptionUpdate): void {
    this.logTranscriptOverlayEvent('received transcription update', () => ({
      runs: update.runs,
      transcriptRunsBefore: this.describeTranscriptRunsForLogging(this.transcriptRuns),
    }));

    this.transcriptRuns = normalizeTranscriptRuns(update.runs);
    this.transcriptDisplay = buildTranscriptDisplay(this.transcriptRuns);
    this.logTranscriptOverlayEvent('applied transcription update', () => ({
      transcriptRunsAfter: this.describeTranscriptRunsForLogging(this.transcriptRuns),
      transcriptDisplay: this.transcriptDisplay,
    }));
    this.renderOverlay();
  }

  applySpectrum(levels: Uint8Array): void {
    if (this.serviceState.indicatorState !== 'recording') {
      this.logTranscriptOverlayEvent('ignored spectrum update because indicator is not recording', {
        indicatorState: this.serviceState.indicatorState,
      });
      return;
    }

    this.logTranscriptOverlayEvent('received spectrum update', () => ({
      barCount: levels.length,
      leadingBars: Array.from(levels.slice(0, 8)),
    }));

    if (!this.view.setSpectrum(levels)) {
      this.logTranscriptOverlayEvent('ignored spectrum update because bar count did not match', {
        expectedBarCount: this.view.spectrumBarCount,
        actualBarCount: levels.length,
      });
      return;
    }

    this.renderOverlay();
  }

  private renderOverlay(): void {
    const transcriptDisplay = this.transcriptDisplay;
    this.logTranscriptOverlayEvent('render overlay requested', () => ({
      indicatorState: this.serviceState.indicatorState,
      servicePresent: this.serviceState.servicePresent,
      servicePhase: this.serviceState.phase,
      transcriptRuns: this.describeTranscriptRunsForLogging(this.transcriptRuns),
      combinedTranscript: this.describeTranscriptTextForLogging(transcriptDisplay.text),
      transcriptDisplay,
    }));

    if (transcriptDisplay.text.length === 0 && this.serviceState.indicatorState !== 'recording') {
      this.logTranscriptOverlayEvent('render overlay hiding because transcript is empty and indicator is not recording');
      this.view.setTranscriptDisplay({ text: '', runs: [] });
      this.view.hide();
      return;
    }

    this.view.setTranscriptDisplay(transcriptDisplay);
    this.view.show(false);
  }

  private resetTranscriptOverlay(): void {
    this.logTranscriptOverlayEvent('reset transcript overlay requested', () => ({
      indicatorState: this.serviceState.indicatorState,
      transcriptRunsBefore: this.describeTranscriptRunsForLogging(this.transcriptRuns),
    }));
    this.resetTranscriptState();
    this.view.setTranscriptDisplay(this.transcriptDisplay);

    if (this.serviceState.indicatorState === 'recording') {
      this.view.show(false);
      return;
    }

    this.view.hide();
  }

  private resetTranscriptState(): void {
    this.transcriptRuns = [];
    this.transcriptDisplay = { text: '', runs: [] };
  }

  private logTranscriptOverlayEvent(
    message: string,
    details: Record<string, unknown> | (() => Record<string, unknown>) = {},
  ): void {
    if (this.transcriptEventCounter % TRANSCRIPT_LOG_SAMPLE_INTERVAL !== 0) {
      this.transcriptEventCounter += 1;
      return;
    }

    this.transcriptEventCounter += 1;
    console.error(
      `Active Listener transcript overlay ${message}`,
      typeof details === 'function' ? details() : details,
    );
  }

  private describeTranscriptTextForLogging(text: string): Record<string, unknown> {
    const trimmedText = text.trim();
    return {
      text,
      characterCount: text.length,
      wordCount: trimmedText.length === 0 ? 0 : trimmedText.split(/\s+/u).length,
    };
  }

  private describeTranscriptRunsForLogging(runs: TranscriptRun[]): Record<string, unknown> {
    return {
      runCount: runs.length,
      runs,
      combinedText: this.describeTranscriptTextForLogging(
        runs.map((run) => run.text.trim()).filter((text) => text.length > 0).join(' '),
      ),
    };
  }
}
