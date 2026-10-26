# Eavesdrop Client API

`EavesdropClient` streams microphone audio over WebSocket to the English-only
Orukeet CPU (INT8 ONNX) or Vulkan (Q8 GGUF) backend. One connection can carry
multiple recordings. Every transcription contains timed segments and word
timestamps.

## Live transcription

```python
import asyncio
import uuid

from eavesdrop.client import EavesdropClient
from eavesdrop.wire import TranscriptionMessage

RECORDING_SECONDS = 5


def show_transcription(message: TranscriptionMessage) -> None:
  for segment in message.segments:
    print(f"{segment.start:.2f}-{segment.end:.2f}: {segment.text}")
    for word in segment.words:
      print(f"  {word.start:.2f}-{word.end:.2f}: {word.word}")


async def transcribe_microphone() -> None:
  async with EavesdropClient.transcriber(
    host="localhost",
    port=9090,
    audio_device="default",
    on_transcription=show_transcription,
  ) as client:
    recording_id = uuid.uuid4().hex
    await client.start_streaming(recording_id)
    try:
      await asyncio.sleep(RECORDING_SECONDS)
    finally:
      await client.stop_streaming()
      await client.flush(recording_id, force_complete=True)
```

The transcriber factory accepts:

- `host: str = "localhost"` and `port: int = 9090`
- `audio_device: str = "default"`
- `on_capture: Callable[[bytes], None] | None = None`, called with PCM chunks
  after sending; the active listener uses this to archive recording audio
- `send_last_n_segments: int | None = None`, a positive segment-window limit
- `language: Literal["en"] | None = None`; both values select English
- The synchronous callbacks listed below

The server selects the model.

## Callbacks

Callbacks can be supplied at construction or assigned before connecting:

- `on_transcription: Callable[[TranscriptionMessage], None] | None`
- `on_connected: Callable[[str], None] | None`
- `on_disconnected: Callable[[str, str | None], None] | None`
- `on_reconnecting: Callable[[str, int, float], None] | None`
- `on_reconnected: Callable[[str], None] | None`

Connection callbacks receive the session identifier. Disconnection adds the
reason; reconnection adds the attempt number and delay in seconds. The client
pushes the wire `TranscriptionMessage` directly to `on_transcription`.

## Session options

Use the strict wire model for direct construction or a connection override:

```python
from eavesdrop.wire import UserTranscriptionOptions

options = UserTranscriptionOptions(send_last_n_segments=3, language="en")
```

Pass it as `transcription_options` to the constructor or `setup_options` to
`connect()`. Unknown fields fail validation.

## Recordings, flush, and cancellation

1. `connect()` establishes the connection and initializes capture resources.
2. `start_streaming(recording_id)` begins a recording and starts capture.
3. `stop_streaming()` stops capture and sends all already captured audio.
4. `flush(recording_id, force_complete=True)` awaits the matching
   `flush_complete` transcription response.
5. `disconnect()` closes the connection and releases resources.

```python
await client.stop_streaming()
final_a = await client.flush(recording_a, force_complete=True)
# on_transcription has already received final_a.
await client.start_streaming(recording_b)
```

The callback receives the flush response before `flush()` returns the same
message. Apply that response once. `start_streaming()` waits for a preceding
pending flush. Concurrent flush requests are rejected.

The async context manager connects and disconnects automatically. A recording
identifier associates audio, controls, and results with that recording. Each new
recording starts with fresh transcript state and timestamp zero on the same
socket. Starting B preserves A's accepted audio, recognition, transcript, and
pending flush. The server queues accepted B audio while A finishes.

A flush requests prompt processing through its accepted audio boundary. A
non-force-completing flush delivers an update with `flush_complete=True` without
explicitly completing pending text. An empty recording or an already processed
buffer also receives an acknowledgement. Flush completion acknowledges the
request; segment completion is expressed separately by `completed`.

`cancel_utterance(recording_id)` stops capture and sends cancellation without
awaiting an acknowledgement. It discards the recording's unfinished work,
suppresses its outstanding results, and immediately fails its pending flush.
Disconnection also fails the pending flush. Request-local cleanup prevents a
failed request from blocking the next recording, including after reconnection.
Controls for an earlier recording leave the current recording unchanged.

Live transcribers automatically retry unexpected disconnects. The client owns
connection, capture, serialized recording transitions, and one pending flush
future. It delivers incoming messages directly through callbacks.

## Transcription data

A `TranscriptionMessage` contains its stream, message timestamp, language, and
segments. Recording results carry their `recording_id`; a flush response sets
`flush_complete=True`.

Each `Segment` contains:

- `id`: the chained CRC64 identifier assigned on completion
- `start` and `end`: seconds on the recording timeline
- `text`: recognized text
- `words`: recognized words with timestamps
- `completed`: whether this timed segment is finalized
- `duration`: `end - start`

Each `Word` contains `start`, `end`, and `word`. CPU boundaries derive from token
onsets and durations at 80 ms granularity; Vulkan word timestamps are converted
from milliseconds. Segment and word timestamps share the original recording
timeline, including silent gaps skipped by VAD.

At most one timed segment is incomplete and open for revision. With VAD enabled,
silence can complete timed segments, including after a non-force-completing flush.
With VAD disabled, recognized portions accumulate in one incomplete timed segment,
preserving all text and word timestamps until an explicit force-completing flush.
Time, silence, and new audio leave that timed segment incomplete.
