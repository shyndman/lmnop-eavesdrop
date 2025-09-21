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

  /** Frame position in audio features where segment processing started. */
  readonly seek: number;

  /** Segment start time in seconds relative to the current audio window. */
  readonly start: number;

  /** Segment end time in seconds relative to the current audio window. */
  readonly end: number;

  /** Transcribed text content of the audio segment. */
  readonly text: string;

  /** List of token IDs from the model's vocabulary used to generate this segment. */
  readonly tokens: readonly number[];

  /** Average log probability across all tokens, indicating generation confidence. */
  readonly avg_logprob: number;

  /** Ratio of text length to token count, used for hallucination detection. */
  readonly compression_ratio: number;

  /** Word-level timing breakdown when word timestamps are enabled, null otherwise. */
  readonly words: readonly Word[] | null;

  /** Generation temperature used when creating this segment, null if not tracked. */
  readonly temperature: number | null;

  /** Whether the segment transcription has been finalized and assigned a chain ID. */
  readonly completed: boolean;

  /** Absolute time offset to convert relative segment times to stream timestamps. */
  readonly time_offset: number;

  /** Return the segment probability by exponentiating the average log probability. */
  readonly probability: number;

  /** Return the absolute start time in the audio stream. */
  readonly absolute_start_time: number;

  /** Return the absolute end time in the audio stream. */
  readonly absolute_end_time: number;

  /** Return the duration of this segment in seconds. */
  readonly duration: number;
}