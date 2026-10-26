# Wire

Shared language for transcription requests and results exchanged
between the server and its clients.

## Language

### Recordings

**Recording**:
Audio captured between a recording start and stop, with its own transcript and
recording-relative timeline.

**Recording identifier**:
The identity associating audio, controls, and transcript updates with one recording.
_Avoid_: Recording epoch.

**Recording boundary**:
The declared start of a recording, establishing its identifier and timestamp zero.

**Recording stop**:
The end of audio capture for a recording.

**Utterance**:
Speech treated as one unit for transcription and cancellation within a recording.

### Transcription

**Transcription**:
The recognized text with segment and individual word timestamps produced from recorded speech.

**Transcript**:
The accumulated recognized text for a recording.

**Transcript update**:
A result containing the current segments for a recording, including revisions
and completion states.

**Pending transcription**:
Accepted audio and recognized text still awaiting explicit or automatic completion.

**Segment**:
An identified, timed portion of a transcript, carrying recognized text
and a completion state.

**Word**:
Recognized word text with start and end times on the recording timeline.

**Word timestamps**:
The start and end times of individual recognized words on the recording timeline.

**Completed segment**:
A segment whose recognized text has been finalized.

**Incomplete segment**:
A segment whose recognized text remains open to revision.
_Avoid_: Final segment.

**Segment completion**:
The transition that finalizes a segment's recognized text.

**Explicit completion**:
Segment completion requested by the client through a force-completing flush.

**Recording-relative timestamp**:
A time measured from the beginning of the original recording,
including intervening silence.

### Live controls

**Flush**:
A request for a transcript update covering audio accepted through a specified
boundary, with a choice of whether to complete that transcription.

**Flush boundary**:
The position in a recording through which an accepted flush must process audio.

**Force completion**:
The flush option requesting completion of all pending segments through its boundary.

**Flush completion**:
The acknowledgement that an accepted flush has been processed and its requested
transcript update delivered.

**Pending flush**:
An accepted flush whose requested transcript update has not yet been delivered.

**Cancellation**:
An explicit request to discard the current utterance and its outstanding
transcription work.

