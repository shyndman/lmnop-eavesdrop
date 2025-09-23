/**
 * Lists of these objects are sent accompanying segments if word-level timestamps are
 * requested.
 */
export interface Word {
  /** Start timestamp of the word in seconds within the audio segment. */
  readonly start: number;

  /** End timestamp of the word in seconds within the audio segment. */
  readonly end: number;

  /** The transcribed text content of the word. */
  readonly word: string;

  /** Confidence score from forced alignment, ranging from 0.0 to 1.0. */
  readonly probability: number;
}

export interface Segment {
  /** Unique segment identifier computed using chain-based CRC64 hash. */
  readonly id: number;

  /** Transcribed text content of the audio segment. */
  readonly text: string;

  /** Return the segment probability by exponentiating the average log probability. */
  readonly avg_probability: number;

  /** Whether the segment transcription has been finalized and assigned a chain ID. */
  readonly completed: boolean;

  /** Return the absolute start time in the audio stream. */
  readonly absolute_start_time: number;

  /** Return the absolute end time in the audio stream. */
  readonly absolute_end_time: number;

  /** Return the duration of this segment in seconds. */
  readonly duration: number;
}
