import { Mode } from '../../messages';
import { Segment } from '../../transcription';
import { AnimationManager } from './animation-manager';
import { diffStringsAsChunks, ChunkDiff, segmentAdditionDiff } from './segment-chunks';

const SEGMENT_STAGGER_DELAY_MS = 100;

type SegmentChunkDiff = [Segment, ChunkDiff];

export class DomManager {
  private animationManager: AnimationManager;

  constructor(animationManager: AnimationManager) {
    this.animationManager = animationManager;
  }

  commitBodyClasses({
    isCommandExecuting,
    currentMode,
    isActive,
    commandElementVisible,
  }: {
    isCommandExecuting: boolean;
    currentMode: Mode | null;
    isActive: boolean;
    commandElementVisible: boolean;
  }): void {
    document.body.classList.toggle('command-executing', isCommandExecuting);
    document.body.classList.toggle(
      'transcribe-active',
      !isCommandExecuting && currentMode === Mode.TRANSCRIBE,
    );
    document.body.classList.toggle(
      'command-active',
      !isCommandExecuting && currentMode === Mode.COMMAND,
    );
    document.body.classList.toggle('active', isActive);
    document.body.classList.toggle('command-visible', commandElementVisible);
  }

  async whileCommitActive(runDuringCommit: () => Promise<void>): Promise<void> {
    try {
      document.body.classList.add('commit-active');
      await runDuringCommit();
    } finally {
      document.body.classList.remove('commit-active');
    }
  }

  /**
   * For a specific mode, gets the containing DOM element and its last paragraph element
   */
  private getElementsForMode(mode: Mode): [HTMLElement, HTMLParagraphElement] {
    let elementId: string;
    switch (mode) {
      case Mode.TRANSCRIBE:
        elementId = 'transcription';
        break;
      case Mode.COMMAND:
        elementId = 'command';
        break;
      default:
        const _exhaustive: never = mode;
        throw new Error(`Unknown mode: ${_exhaustive}`);
    }

    const element = document.getElementById(elementId);
    if (!element) {
      throw new Error(`${elementId} element not found`);
    }

    let lastParagraph = element.querySelector('p:last-of-type') as HTMLParagraphElement | null;
    if (!lastParagraph) {
      // If no paragraph exists, create one
      const newParagraph = document.createElement('p') as HTMLParagraphElement;
      element.appendChild(newParagraph);
      lastParagraph = newParagraph;
    }

    return [element, lastParagraph];
  }

  /**
   * Compares the newly completed and in-progress segments with the existing state, mutates the
   * DOM to reflect the new state, then fades _out_ only the words that have been removed, and
   * fades in _only_ the words that are new.
   */
  async updateModeDomSegments(
    mode: Mode,
    completedSegments: readonly Segment[],
    inProgressSegment: Segment,
  ): Promise<void> {
    // First things first, we're going to cheat a little for an edge case. We rarely receive
    // multiple completed segments, although it is technically possible. To simplify the logic
    // here, we're going to combine them, and average their probabilities.
    let completedSegment: Segment | null = null;
    if (completedSegments.length > 1) {
      completedSegment = this.mergeCompletedSegments(completedSegments, completedSegment);
    } else if (completedSegments.length === 1) {
      completedSegment = completedSegments[0];
    }

    // We remove the entire current in-progress segment, but we're going to recreate the parts of it
    // that we need to animate.
    const [, modeContainer] = this.getElementsForMode(mode);
    const currentInProgressText = await this.removeInProgressSegment(modeContainer);

    // If there are completed segments, they are guaranteed to have completed from the in-progress
    // element in the DOM.
    const chunksBySegment: SegmentChunkDiff[] = [];
    if (completedSegment) {
      chunksBySegment.push([
        completedSegment,
        diffStringsAsChunks(currentInProgressText, completedSegment.text),
      ]);
      chunksBySegment.push([inProgressSegment, segmentAdditionDiff(inProgressSegment)]);
    } else {
      chunksBySegment.push([
        inProgressSegment,
        diffStringsAsChunks(currentInProgressText, inProgressSegment.text),
      ]);
    }

    // Create new paragraph with all segments
    const { segmentSpans, fadeOutChunks, fadeInChunks } =
      this.createChunkedSegmentSpans(chunksBySegment);

    // Add new paragraph alongside existing content
    for (const s of segmentSpans) {
      modeContainer.appendChild(s);
    }

    await this.animationManager.fadeOut(fadeOutChunks);
    for (const c of fadeOutChunks) {
      c.remove();
    }
    await this.animationManager.fadeIn(fadeInChunks, SEGMENT_STAGGER_DELAY_MS);
  }

  createChunkedSegmentSpans(chunksBySegment: SegmentChunkDiff[]): {
    segmentSpans: HTMLSpanElement[];
    fadeOutChunks: HTMLSpanElement[];
    fadeInChunks: HTMLSpanElement[];
  } {
    const segmentSpans: HTMLSpanElement[] = [];
    const fadeOutChunks: HTMLSpanElement[] = [];
    const fadeInChunks: HTMLSpanElement[] = [];

    for (const [segment, chunks] of chunksBySegment) {
      const segmentSpan = this.createSegmentSpan(segment);
      segmentSpans.push(segmentSpan);

      for (const chunk of chunks.static) {
        const span = document.createElement('span');
        span.textContent = chunk;
        segmentSpan.appendChild(span);
      }

      for (const chunk of chunks.removed) {
        const span = document.createElement('span');
        span.textContent = chunk;
        segmentSpan.appendChild(span);
        fadeOutChunks.push(span);
      }

      for (const chunk of chunks.added) {
        const span = document.createElement('span');
        span.textContent = chunk;
        segmentSpan.appendChild(span);
        fadeInChunks.push(span);
      }
    }

    return {
      segmentSpans,
      fadeOutChunks,
      fadeInChunks,
    };
  }

  private mergeCompletedSegments(
    completedSegments: readonly Segment[],
    completedSegment: Segment | null,
  ): Segment {
    const combinedText = completedSegments.map((s) => s.text).join(' ');
    const avgProb =
      completedSegments.reduce((sum, s) => sum + s.avg_probability, 0) / completedSegments.length;
    completedSegment = {
      id: completedSegments[completedSegments.length - 1].id,
      text: combinedText,
      avg_probability: avgProb,
      absolute_start_time: completedSegments[0].absolute_start_time,
      absolute_end_time: completedSegments[completedSegments.length - 1].absolute_end_time,
      duration: completedSegments.reduce((sum, s) => sum + s.duration, 0),
      completed: true,
    };
    return completedSegment;
  }

  private createSegmentSpan(segment: Segment): HTMLSpanElement {
    const span = document.createElement('span') as HTMLSpanElement;
    span.id = `segment-${segment.id}`;
    span.classList.add('segment', this.getSegmentProbabilityClass(segment));
    span.classList.toggle('in-progress-segment', !segment.completed);
    return span;
  }

  /**
   * Calculate CSS probability class for a segment
   */
  private getSegmentProbabilityClass(segment: Segment): string {
    const rounded = Math.round((segment.avg_probability * 100) / 5) * 5;
    return `segment-prob-${rounded}`;
  }

  /**
   * Remove the existing in-progress segment with animation (single segment invariant)
   */
  private async removeInProgressSegment(container: HTMLElement): Promise<string> {
    const inProgressSpan = container.querySelector('.in-progress-segment') as HTMLElement;
    if (inProgressSpan) {
      const spanText = inProgressSpan.textContent;
      inProgressSpan.remove();
      return spanText;
    }
    return '';
  }

  async clearAllContent(): Promise<void> {
    await Promise.all([
      this.clearModeContent(Mode.TRANSCRIBE),
      this.clearModeContent(Mode.COMMAND),
    ]);
  }

  async clearModeContent(mode: Mode): Promise<void> {
    const [container, paragraph] = this.getElementsForMode(mode);
    await this.animationManager.fadeOutModeContent(mode, paragraph);
    container.innerHTML = '<p></p>';
  }

  async replaceModeContent(mode: Mode, content: string): Promise<void> {
    const [container] = this.getElementsForMode(mode);
    const processedContent = this.ensureParagraphTags(content);
    await this.animationManager.replaceModeContent(mode, container, processedContent);
  }

  async addModeContent(mode: Mode, content: string): Promise<void> {
    const [container] = this.getElementsForMode(mode);
    const processedContent = this.ensureParagraphTags(content);
    await this.animationManager.fadeInModeContent(mode, container, processedContent);
  }

  setEmptyContent(mode: Mode): void {
    const [container] = this.getElementsForMode(mode);
    container.innerHTML = '<p></p>';
  }

  private ensureParagraphTags(content: string): string {
    const trimmed = content.trim();

    if (trimmed === '') {
      return '<p></p>';
    }

    if (trimmed.startsWith('<p>') && trimmed.endsWith('</p>')) {
      return trimmed;
    }

    return `<p>${trimmed}</p>`;
  }
}
