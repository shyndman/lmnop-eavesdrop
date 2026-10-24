# Wire

Shared language for transcription requests and results exchanged
between the server and its clients.

## Language

**Segment**:
An identified, timed portion of a transcript, carrying recognized text
and a completion state.

**Word**:
Recognized word text with start and end times on the recording timeline.

**Completed segment**:
A segment whose recognition is finalized.

**Incomplete segment**:
The trailing segment whose recognition can still change.
_Avoid_: Final segment.

**Recording-relative timestamp**:
A time measured from the beginning of the original recording,
including intervening silence.

**Source mode**:
The distinction between live audio and a finite audio file.
