import { Segment } from '../../transcription';

export interface ChunkDiff {
  static: string[];
  added: string[];
  removed: string[];
}

/**
 * Diff that represents adding a new segment (all chunks are "added")
 */
export function segmentAdditionDiff(segment: Segment): ChunkDiff {
  const added_chunks = chunkForAnimation(segment.text);
  return {
    static: [],
    added: added_chunks,
    removed: [],
  };
}

/**
 * Find the common prefix between two arrays of chunks, as defined by chunkForAnimation.
 */
export function diffStringsAsChunks(current: string, next: string): ChunkDiff {
  const current_chunks = chunkForAnimation(current);
  const next_chunks = chunkForAnimation(next);
  const commonPrefix = findCommonPrefix(current_chunks, next_chunks);
  return {
    static: commonPrefix,
    added: next_chunks.slice(commonPrefix.length),
    removed: current_chunks.slice(commonPrefix.length),
  };
}

function findCommonPrefix(s1: string[], s2: string[]): string[] {
  const minLength = Math.min(s1.length, s2.length);
  const common: string[] = [];
  for (let i = 0; i < minLength; i++) {
    if (s1[i] !== s2[i]) {
      break;
    }
    common.push(s1[i]);
  }
  return common;
}

function chunkForAnimation(text: string): string[] {
  return text.match(/(^|\s+)\S+/g) || [];
}
