/**
 * Mock scenarios for testing the transcription UI with realistic message sequences.
 * These generators produce timed message sequences that simulate real user workflows.
 */

import { Mode, MessageType, type Message } from '../messages';
import { type Segment } from '../transcription';

// Timing constants to match real transcription behavior
const TRANSCRIPTION_INTERVAL_MS = 2000; // Whisper segments arrive every 2 seconds
const COMMAND_PROCESSING_MS = 3000; // Typical LLM response time
const UNDO_PROCESSING_MS = 500; // Instant revert
const USER_PAUSE_MS = 1000; // User thinking time

interface TimedMessage {
  delay: number;
  message: Message;
}

/**
 * Creates a realistic Segment object with proper Whisper-style metadata
 */
function createSegment(
  id: number,
  text: string,
  startTime: number,
  endTime: number,
  confidence: number = 0.85,
  completed: boolean = false
): Segment {
  const tokens = Array.from({ length: Math.ceil(text.length / 4) }, (_, i) => 1000 + i);

  return {
    id,
    seek: Math.floor(startTime * 100), // Frame position
    start: startTime,
    end: endTime,
    text,
    tokens,
    avg_logprob: Math.log(confidence),
    avg_probability: confidence,
    compression_ratio: text.length / tokens.length,
    words: null, // Not using word-level timing for these scenarios
    temperature: null,
    completed,
    time_offset: 0,
    absolute_start_time: startTime,
    absolute_end_time: endTime,
    duration: endTime - startTime
  };
}

/**
 * Happy Path Scenario: User transcribes thoughtful content, then refines it with commands
 *
 * Flow: Transcribe → Transform → Small adjustment → Commit
 * Demonstrates normal workflow with realistic transcription artifacts
 */
export function* happyPathScenario(): Generator<TimedMessage> {
  let segmentId = 1;
  let currentTime = 0;

  // Phase 1: Initial transcription - first continuous burst
  yield {
    delay: 500,
    message: {
      type: MessageType.APPEND_SEGMENTS,
      target_mode: Mode.TRANSCRIBE,
      completed_segments: [],
      in_progress_segment: createSegment(segmentId, "So I've been thinking about", currentTime, currentTime + 1.2, 0.9, false)
    }
  };

  currentTime += 2;
  yield {
    delay: TRANSCRIPTION_INTERVAL_MS,
    message: {
      type: MessageType.APPEND_SEGMENTS,
      target_mode: Mode.TRANSCRIBE,
      completed_segments: [],
      in_progress_segment: createSegment(segmentId, "So I've been thinking about the caching strategy we discussed, um,", currentTime, currentTime + 3.1, 0.87, false)
    }
  };

  currentTime += 2;
  yield {
    delay: TRANSCRIPTION_INTERVAL_MS,
    message: {
      type: MessageType.APPEND_SEGMENTS,
      target_mode: Mode.TRANSCRIBE,
      completed_segments: [],
      in_progress_segment: createSegment(segmentId, "So I've been thinking about the caching strategy we discussed, um, and I think there's a few different", currentTime, currentTime + 4.8, 0.89, false)
    }
  };

  currentTime += 2;
  yield {
    delay: TRANSCRIPTION_INTERVAL_MS,
    message: {
      type: MessageType.APPEND_SEGMENTS,
      target_mode: Mode.TRANSCRIBE,
      completed_segments: [],
      in_progress_segment: createSegment(segmentId, "So I've been thinking about the caching strategy we discussed, um, and I think there's a few different approaches we could take here.", currentTime, currentTime + 6.2, 0.88, false)
    }
  };

  const completedSegment1 = createSegment(segmentId - 1, "So I've been thinking about the caching strategy we discussed, um, and I think there's a few different approaches we could take here.", 0, 6.2, 0.88, true);

  // Natural pause - segment completes
  currentTime += 3; // Longer pause
  segmentId++;
  yield {
    delay: 3000, // Pause for thinking
    message: {
      type: MessageType.APPEND_SEGMENTS,
      target_mode: Mode.TRANSCRIBE,
      completed_segments: [completedSegment1],
      in_progress_segment: createSegment(segmentId, "Like, we could do the simple approach", currentTime, currentTime + 1.7, 0.91)
    }
  };

  currentTime += 2;
  yield {
    delay: TRANSCRIPTION_INTERVAL_MS,
    message: {
      type: MessageType.APPEND_SEGMENTS,
      target_mode: Mode.TRANSCRIBE,
      completed_segments: [completedSegment1],
      in_progress_segment: createSegment(segmentId, "Like, we could do the simple approach where we just cache everything for", currentTime, currentTime + 3.1, 0.92)
    }
  };

  currentTime += 2;
  yield {
    delay: TRANSCRIPTION_INTERVAL_MS,
    message: {
      type: MessageType.APPEND_SEGMENTS,
      target_mode: Mode.TRANSCRIBE,
      completed_segments: [completedSegment1],
      in_progress_segment: createSegment(segmentId, "Like, we could do the simple approach where we just cache everything for like 5 minutes or", currentTime, currentTime + 4.5, 0.91, false)
    }
  };

  const completedSegment2 = createSegment(segmentId, "Like, we could do the simple approach where we just cache everything for like 5 minutes or", 6.5, 11.0, 0.91, true)

  // Pause after "or" - natural thinking moment, segment completes
  currentTime += 2.5; // Longer pause after "or"
  segmentId++;
  yield {
    delay: 2500, // Extended pause for thinking after "or"
    message: {
      type: MessageType.APPEND_SEGMENTS,
      target_mode: Mode.TRANSCRIBE,
      completed_segments: [completedSegment1,
      completedSegment2],
      in_progress_segment: createSegment(segmentId, "however, but that might not be optimal", currentTime, currentTime + 1.8, 0.76, false) // Lower confidence on "however"
    }
  };

  currentTime += 2;
  yield {
    delay: TRANSCRIPTION_INTERVAL_MS,
    message: {
      type: MessageType.APPEND_SEGMENTS,
      target_mode: Mode.TRANSCRIBE,
      completed_segments: [completedSegment1,
      completedSegment2],
      in_progress_segment: createSegment(segmentId, "however, but that might not be optimal because some data changes more frequently", currentTime, currentTime + 4.2, 0.89, false)
    }
  };

  currentTime += 2;
  yield {
    delay: TRANSCRIPTION_INTERVAL_MS,
    message: {
      type: MessageType.APPEND_SEGMENTS,
      target_mode: Mode.TRANSCRIBE,
      completed_segments: [completedSegment1,
      completedSegment2],
      in_progress_segment: createSegment(segmentId, "however, but that might not be optimal because some data changes more frequently than others.", currentTime, currentTime + 5.1, 0.92, false)
    }
  };

  // Final pause - last segment completes
  currentTime += 2;
  segmentId++;
  yield {
    delay: TRANSCRIPTION_INTERVAL_MS,
    message: {
      type: MessageType.APPEND_SEGMENTS,
      target_mode: Mode.TRANSCRIBE,
      completed_segments: [completedSegment1,
      completedSegment2,
      createSegment(segmentId - 1, "however, but that might not be optimal because some data changes more frequently than others.", 13.5, 18.6, 0.92, true)],
      in_progress_segment: createSegment(segmentId, "", currentTime, currentTime, 0.0, false) // Empty to signal end
    }
  };

  // Phase 2: User switches to command mode to clean up the text
  yield {
    delay: USER_PAUSE_MS,
    message: {
      type: MessageType.CHANGE_MODE,
      target_mode: Mode.COMMAND
    }
  };

  // User dictates cleanup command in one continuous utterance
  yield {
    delay: 800,
    message: {
      type: MessageType.APPEND_SEGMENTS,
      target_mode: Mode.COMMAND,
      completed_segments: [],
      in_progress_segment: createSegment(segmentId, "Clean up the filler words and", 0, 1.2, 0.91)
    }
  };

  yield {
    delay: TRANSCRIPTION_INTERVAL_MS,
    message: {
      type: MessageType.APPEND_SEGMENTS,
      target_mode: Mode.COMMAND,
      completed_segments: [],
      in_progress_segment: createSegment(segmentId, "Clean up the filler words and improve the grammar, but keep", 0, 2.4, 0.89)
    }
  };

  yield {
    delay: TRANSCRIPTION_INTERVAL_MS,
    message: {
      type: MessageType.APPEND_SEGMENTS,
      target_mode: Mode.COMMAND,
      completed_segments: [],
      in_progress_segment: createSegment(segmentId, "Clean up the filler words and improve the grammar, but keep all the original ideas intact.", 0, 3.6, 0.91)
    }
  };

  // Natural pause - command segment completes
  yield {
    delay: TRANSCRIPTION_INTERVAL_MS,
    message: {
      type: MessageType.APPEND_SEGMENTS,
      target_mode: Mode.COMMAND,
      completed_segments: [createSegment(segmentId++, "Clean up the filler words and improve the grammar, but keep all the original ideas intact.", 0, 3.6, 0.91, true)],
      in_progress_segment: createSegment(segmentId, "", 3.6, 3.6, 0.0)
    }
  };

  // Command execution begins
  yield {
    delay: 500,
    message: {
      type: MessageType.COMMAND_EXECUTING,
      waiting_messages: [
        "Analyzing text structure...",
        "Identifying filler words...",
        "Improving grammar...",
        "Preserving original meaning..."
      ]
    }
  };

  // Command completes with cleaned up result
  yield {
    delay: COMMAND_PROCESSING_MS,
    message: {
      type: MessageType.SET_STRING,
      target_mode: Mode.TRANSCRIBE,
      content: "I've been thinking about the caching strategy we discussed, and I think there are several different approaches we could take. We could use a simple approach where we cache everything for 5 minutes, but that might not be optimal because some data changes more frequently than others."
    }
  };

  // User reviews, decides to make one more small adjustment
  yield {
    delay: 2000, // User reads and thinks
    message: {
      type: MessageType.CHANGE_MODE,
      target_mode: Mode.COMMAND
    }
  };

  yield {
    delay: 800,
    message: {
      type: MessageType.APPEND_SEGMENTS,
      target_mode: Mode.COMMAND,
      completed_segments: [],
      in_progress_segment: createSegment(segmentId, "Make the last part more", 0, 1.0, 0.88)
    }
  };

  yield {
    delay: TRANSCRIPTION_INTERVAL_MS,
    message: {
      type: MessageType.APPEND_SEGMENTS,
      target_mode: Mode.COMMAND,
      completed_segments: [],
      in_progress_segment: createSegment(segmentId, "Make the last part more decisive and suggest a solution.", 0, 2.3, 0.90)
    }
  };

  // Natural pause - command segment completes
  yield {
    delay: TRANSCRIPTION_INTERVAL_MS,
    message: {
      type: MessageType.APPEND_SEGMENTS,
      target_mode: Mode.COMMAND,
      completed_segments: [createSegment(segmentId++, "Make the last part more decisive and suggest a solution.", 0, 2.3, 0.90, true)],
      in_progress_segment: createSegment(segmentId, "", 2.3, 2.3, 0.0)
    }
  };

  yield {
    delay: 500,
    message: {
      type: MessageType.COMMAND_EXECUTING,
      waiting_messages: [
        "Strengthening conclusion...",
        "Adding solution...",
        "Making it more decisive..."
      ]
    }
  };

  yield {
    delay: COMMAND_PROCESSING_MS,
    message: {
      type: MessageType.SET_STRING,
      target_mode: Mode.TRANSCRIBE,
      content: "I've been thinking about the caching strategy we discussed, and I think there are several different approaches we could take. We could use a simple approach where we cache everything for 5 minutes, but that might not be optimal because some data changes more frequently than others. Instead, we should implement a tiered TTL system with different cache durations based on data volatility."
    }
  };

  // User is satisfied and commits
  yield {
    delay: 3000, // User reviews final result
    message: {
      type: MessageType.COMMIT_OPERATION,
      cancelled: false
    }
  };
}

/**
 * Perfectionist Spiral Scenario: User gets caught in endless refinement loop
 *
 * Flow: Transcribe → Command → Undo → Command → Undo → Command → Finally commit
 * Demonstrates stress testing with rapid mode switching and version history
 */
export function* perfectionistSpiralScenario(): Generator<TimedMessage> {
  let segmentId = 1;
  let currentTime = 0;

  // Phase 1: Initial transcription of a message about a technical decision - one continuous utterance
  yield {
    delay: 500,
    message: {
      type: MessageType.APPEND_SEGMENTS,
      target_mode: Mode.TRANSCRIBE,
      completed_segments: [],
      in_progress_segment: createSegment(segmentId, "I think we should probably", currentTime, currentTime + 1.1, 0.88)
    }
  };

  currentTime += 2;
  yield {
    delay: TRANSCRIPTION_INTERVAL_MS,
    message: {
      type: MessageType.APPEND_SEGMENTS,
      target_mode: Mode.TRANSCRIBE,
      completed_segments: [],
      in_progress_segment: createSegment(segmentId, "I think we should probably go with the microservices approach", currentTime, currentTime + 2.7, 0.89)
    }
  };

  currentTime += 2;
  yield {
    delay: TRANSCRIPTION_INTERVAL_MS,
    message: {
      type: MessageType.APPEND_SEGMENTS,
      target_mode: Mode.TRANSCRIBE,
      completed_segments: [],
      in_progress_segment: createSegment(segmentId, "I think we should probably go with the microservices approach for this project, even though", currentTime, currentTime + 4.0, 0.88)
    }
  };

  currentTime += 2;
  yield {
    delay: TRANSCRIPTION_INTERVAL_MS,
    message: {
      type: MessageType.APPEND_SEGMENTS,
      target_mode: Mode.TRANSCRIBE,
      completed_segments: [],
      in_progress_segment: createSegment(segmentId, "I think we should probably go with the microservices approach for this project, even though it might be overkill for our", currentTime, currentTime + 5.4, 0.90)
    }
  };

  currentTime += 2;
  yield {
    delay: TRANSCRIPTION_INTERVAL_MS,
    message: {
      type: MessageType.APPEND_SEGMENTS,
      target_mode: Mode.TRANSCRIBE,
      completed_segments: [],
      in_progress_segment: createSegment(segmentId, "I think we should probably go with the microservices approach for this project, even though it might be overkill for our current scale, but I think", currentTime, currentTime + 6.6, 0.89)
    }
  };

  currentTime += 2;
  yield {
    delay: TRANSCRIPTION_INTERVAL_MS,
    message: {
      type: MessageType.APPEND_SEGMENTS,
      target_mode: Mode.TRANSCRIBE,
      completed_segments: [],
      in_progress_segment: createSegment(segmentId, "I think we should probably go with the microservices approach for this project, even though it might be overkill for our current scale, but I think it'll help us in the long run.", currentTime, currentTime + 8.1, 0.87)
    }
  };

  // Natural pause - segment completes
  currentTime += 2;
  segmentId++;
  yield {
    delay: TRANSCRIPTION_INTERVAL_MS,
    message: {
      type: MessageType.APPEND_SEGMENTS,
      target_mode: Mode.TRANSCRIBE,
      completed_segments: [createSegment(segmentId - 1, "I think we should probably go with the microservices approach for this project, even though it might be overkill for our current scale, but I think it'll help us in the long run.", 0, 8.1, 0.87, true)],
      in_progress_segment: createSegment(segmentId, "", currentTime, currentTime, 0.0)
    }
  };

  // Store the original text for undo operations
  const originalText = "I think we should probably go with the microservices approach for this project, even though it might be overkill for our current scale, but I think it'll help us in the long run.";

  // Phase 2: First refinement attempt - make it more concise
  yield {
    delay: USER_PAUSE_MS,
    message: {
      type: MessageType.CHANGE_MODE,
      target_mode: Mode.COMMAND
    }
  };

  yield {
    delay: 600,
    message: {
      type: MessageType.APPEND_SEGMENTS,
      target_mode: Mode.COMMAND,
      completed_segments: [],
      in_progress_segment: createSegment(segmentId++, "Make this more concise", 0, 1.0, 0.94, false)
    }
  };

  yield {
    delay: TRANSCRIPTION_INTERVAL_MS,
    message: {
      type: MessageType.APPEND_SEGMENTS,
      target_mode: Mode.COMMAND,
      completed_segments: [createSegment(segmentId - 1, "Make this more concise", 0, 1.0, 0.94, true)],
      in_progress_segment: createSegment(segmentId++, "", 1.0, 1.0, 0.0, false)
    }
  };

  yield {
    delay: 500,
    message: {
      type: MessageType.COMMAND_EXECUTING,
      waiting_messages: ["Condensing text...", "Removing redundancy...", "Shortening sentences..."]
    }
  };

  yield {
    delay: COMMAND_PROCESSING_MS,
    message: {
      type: MessageType.SET_STRING,
      target_mode: Mode.TRANSCRIBE,
      content: "We should use microservices for long-term scalability."
    }
  };

  // Phase 3: User doesn't like it, undoes
  yield {
    delay: 1500, // User reads and dislikes
    message: {
      type: MessageType.CHANGE_MODE,
      target_mode: Mode.COMMAND
    }
  };

  yield {
    delay: 400,
    message: {
      type: MessageType.APPEND_SEGMENTS,
      target_mode: Mode.COMMAND,
      completed_segments: [],
      in_progress_segment: createSegment(segmentId++, "Undo", 0, 0.3, 0.97, false)
    }
  };

  yield {
    delay: 800,
    message: {
      type: MessageType.APPEND_SEGMENTS,
      target_mode: Mode.COMMAND,
      completed_segments: [createSegment(segmentId - 1, "Undo", 0, 0.3, 0.97, true)],
      in_progress_segment: createSegment(segmentId++, "", 0.3, 0.3, 0.0, false)
    }
  };

  // Instant undo - revert to original
  yield {
    delay: UNDO_PROCESSING_MS,
    message: {
      type: MessageType.SET_STRING,
      target_mode: Mode.TRANSCRIBE,
      content: originalText
    }
  };

  // Phase 4: Try making it more detailed instead
  yield {
    delay: 800,
    message: {
      type: MessageType.CHANGE_MODE,
      target_mode: Mode.COMMAND
    }
  };

  yield {
    delay: 600,
    message: {
      type: MessageType.APPEND_SEGMENTS,
      target_mode: Mode.COMMAND,
      completed_segments: [],
      in_progress_segment: createSegment(segmentId, "Make it more detailed and", 0, 1.1, 0.90)
    }
  };

  yield {
    delay: TRANSCRIPTION_INTERVAL_MS,
    message: {
      type: MessageType.APPEND_SEGMENTS,
      target_mode: Mode.COMMAND,
      completed_segments: [],
      in_progress_segment: createSegment(segmentId, "Make it more detailed and add specific technical benefits.", 0, 2.4, 0.89)
    }
  };

  // Natural pause - command segment completes
  yield {
    delay: TRANSCRIPTION_INTERVAL_MS,
    message: {
      type: MessageType.APPEND_SEGMENTS,
      target_mode: Mode.COMMAND,
      completed_segments: [createSegment(segmentId++, "Make it more detailed and add specific technical benefits.", 0, 2.4, 0.89, true)],
      in_progress_segment: createSegment(segmentId, "", 2.4, 2.4, 0.0)
    }
  };

  yield {
    delay: 500,
    message: {
      type: MessageType.COMMAND_EXECUTING,
      waiting_messages: ["Adding technical details...", "Expanding benefits...", "Providing specifics..."]
    }
  };

  const detailedText = "I think we should adopt a microservices architecture for this project. While it may seem like overkill for our current scale, microservices will provide us with better fault isolation, independent deployment capabilities, technology diversity, and improved scalability. The initial complexity will pay dividends as our system grows and our team expands.";

  yield {
    delay: COMMAND_PROCESSING_MS,
    message: {
      type: MessageType.SET_STRING,
      target_mode: Mode.TRANSCRIBE,
      content: detailedText
    }
  };

  // Phase 5: Still not satisfied, undo again
  yield {
    delay: 2000,
    message: {
      type: MessageType.CHANGE_MODE,
      target_mode: Mode.COMMAND
    }
  };

  yield {
    delay: 400,
    message: {
      type: MessageType.APPEND_SEGMENTS,
      target_mode: Mode.COMMAND,
      completed_segments: [],
      in_progress_segment: createSegment(segmentId++, "Undo", 0, 0.3, 0.97, false)
    }
  };

  yield {
    delay: 800,
    message: {
      type: MessageType.APPEND_SEGMENTS,
      target_mode: Mode.COMMAND,
      completed_segments: [createSegment(segmentId - 1, "Undo", 0, 0.3, 0.97, true)],
      in_progress_segment: createSegment(segmentId++, "", 0.3, 0.3, 0.0, false)
    }
  };

  yield {
    delay: UNDO_PROCESSING_MS,
    message: {
      type: MessageType.SET_STRING,
      target_mode: Mode.TRANSCRIBE,
      content: originalText
    }
  };

  // Phase 6: Try removing jargon
  yield {
    delay: 1000,
    message: {
      type: MessageType.CHANGE_MODE,
      target_mode: Mode.COMMAND
    }
  };

  yield {
    delay: 700,
    message: {
      type: MessageType.APPEND_SEGMENTS,
      target_mode: Mode.COMMAND,
      completed_segments: [],
      in_progress_segment: createSegment(segmentId++, "Remove the jargon and make", 0, 1.2, 0.89, false)
    }
  };

  yield {
    delay: TRANSCRIPTION_INTERVAL_MS,
    message: {
      type: MessageType.APPEND_SEGMENTS,
      target_mode: Mode.COMMAND,
      completed_segments: [createSegment(segmentId - 1, "Remove the jargon and make", 0, 1.2, 0.89, true)],
      in_progress_segment: createSegment(segmentId++, "it accessible to non-engineers.", 1.2, 2.5, 0.91, false)
    }
  };

  yield {
    delay: TRANSCRIPTION_INTERVAL_MS,
    message: {
      type: MessageType.APPEND_SEGMENTS,
      target_mode: Mode.COMMAND,
      completed_segments: [createSegment(segmentId - 1, "it accessible to non-engineers.", 1.2, 2.5, 0.91, true)],
      in_progress_segment: createSegment(segmentId++, "", 2.5, 2.5, 0.0, false)
    }
  };

  yield {
    delay: 500,
    message: {
      type: MessageType.COMMAND_EXECUTING,
      waiting_messages: ["Simplifying language...", "Removing technical terms...", "Making it accessible..."]
    }
  };

  yield {
    delay: COMMAND_PROCESSING_MS,
    message: {
      type: MessageType.SET_STRING,
      target_mode: Mode.TRANSCRIBE,
      content: "I think we should break our application into smaller, independent services. It might seem like too much for what we have now, but it will make things easier to manage and scale as we grow."
    }
  };

  // Phase 7: Getting frustrated, wants the tone more confident
  yield {
    delay: 1800,
    message: {
      type: MessageType.CHANGE_MODE,
      target_mode: Mode.COMMAND
    }
  };

  yield {
    delay: 600,
    message: {
      type: MessageType.APPEND_SEGMENTS,
      target_mode: Mode.COMMAND,
      completed_segments: [],
      in_progress_segment: createSegment(segmentId, "Make the tone more confident", 0, 1.3, 0.92)
    }
  };

  yield {
    delay: TRANSCRIPTION_INTERVAL_MS,
    message: {
      type: MessageType.APPEND_SEGMENTS,
      target_mode: Mode.COMMAND,
      completed_segments: [],
      in_progress_segment: createSegment(segmentId, "Make the tone more confident and decisive.", 0, 1.9, 0.93)
    }
  };

  // Natural pause - command segment completes
  yield {
    delay: TRANSCRIPTION_INTERVAL_MS,
    message: {
      type: MessageType.APPEND_SEGMENTS,
      target_mode: Mode.COMMAND,
      completed_segments: [createSegment(segmentId++, "Make the tone more confident and decisive.", 0, 1.9, 0.93, true)],
      in_progress_segment: createSegment(segmentId, "", 1.9, 1.9, 0.0)
    }
  };

  yield {
    delay: 500,
    message: {
      type: MessageType.COMMAND_EXECUTING,
      waiting_messages: ["Strengthening tone...", "Making it decisive...", "Adding confidence..."]
    }
  };

  yield {
    delay: COMMAND_PROCESSING_MS,
    message: {
      type: MessageType.SET_STRING,
      target_mode: Mode.TRANSCRIBE,
      content: "We should definitely break our application into smaller, independent services. This approach will provide superior scalability and maintainability as we grow."
    }
  };

  // Phase 8: Second thoughts, wants to soften it
  yield {
    delay: 1500,
    message: {
      type: MessageType.CHANGE_MODE,
      target_mode: Mode.COMMAND
    }
  };

  yield {
    delay: 700,
    message: {
      type: MessageType.APPEND_SEGMENTS,
      target_mode: Mode.COMMAND,
      completed_segments: [],
      in_progress_segment: createSegment(segmentId, "Actually, soften the tone", 0, 1.1, 0.88)
    }
  };

  yield {
    delay: TRANSCRIPTION_INTERVAL_MS,
    message: {
      type: MessageType.APPEND_SEGMENTS,
      target_mode: Mode.COMMAND,
      completed_segments: [],
      in_progress_segment: createSegment(segmentId, "Actually, soften the tone and add some uncertainty.", 0, 2.2, 0.89)
    }
  };

  // Natural pause - command segment completes
  yield {
    delay: TRANSCRIPTION_INTERVAL_MS,
    message: {
      type: MessageType.APPEND_SEGMENTS,
      target_mode: Mode.COMMAND,
      completed_segments: [createSegment(segmentId++, "Actually, soften the tone and add some uncertainty.", 0, 2.2, 0.89, true)],
      in_progress_segment: createSegment(segmentId, "", 2.2, 2.2, 0.0)
    }
  };

  yield {
    delay: 500,
    message: {
      type: MessageType.COMMAND_EXECUTING,
      waiting_messages: ["Softening tone...", "Adding nuance...", "Balancing confidence..."]
    }
  };

  yield {
    delay: COMMAND_PROCESSING_MS,
    message: {
      type: MessageType.SET_STRING,
      target_mode: Mode.TRANSCRIBE,
      content: "I'm leaning towards breaking our application into smaller, independent services. While there might be some added complexity initially, it could provide better scalability and maintainability as we grow."
    }
  };

  // Phase 9: User finally gives up and commits whatever is there
  yield {
    delay: 4000, // Long pause - user is exhausted
    message: {
      type: MessageType.COMMIT_OPERATION,
      cancelled: false
    }
  };
}
