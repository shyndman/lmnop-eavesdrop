# Eavesdrop Client API

`EavesdropClient` supplies live microphone audio over WebSocket to the English-only
Orukeet CPU (INT8 ONNX) or Vulkan (Q8 GGUF) backend. One connection can carry
multiple recordings. Every transcription includes native segment and individual
word timestamps.

## Live transcription

```python
import uuid

from eavesdrop.client import EavesdropClient


async def transcribe_microphone() -> None:
  async with EavesdropClient.transcriber(
    host="localhost",
    port=9090,
    audio_device="default",
    language="en",
  ) as client:
    await client.start_streaming(uuid.uuid4().hex)
    async for event in client:
      if event.family != "transcription":
        continue
      for segment in event.message.segments:
        print(f"{segment.start:.2f}-{segment.end:.2f}: {segment.text}")
        for word in segment.words:
          print(f"  {word.start:.2f}-{word.end:.2f}: {word.word}")
```

The transcriber factory accepts:

- `host: str = "localhost"` and `port: int = 9090`
- `audio_device: str = "default"`
- `on_capture: Callable[[bytes], None] | None = None`, for captured PCM chunks;
  the active listener uses this for its historical audio archive
- `send_last_n_segments: int | None = None`, a positive segment-window limit
- `language: Literal["en"] | None = None`, an English-only compatibility hint

`None` for language also means English; there is no language auto-detection.
Model selection is controlled by the server. Prompting, hotwords, beam-search
overrides, per-session models, and translation are unsupported.

## Session options

For direct construction or a connection-specific override, use the strict wire
model:

```python
from eavesdrop.wire import UserTranscriptionOptions

options = UserTranscriptionOptions(
  send_last_n_segments=3,
  language="en",
)
```

Pass it as `transcription_options` to the constructor or `setup_options` to
`connect()`. Unknown fields fail validation. Live audio is the sole ingestion
workflow; setup does not select an audio source or an output timing format.

## Recordings, flush, and cancellation

1. `connect()` establishes the connection and initializes capture resources.
2. `start_streaming(recording_id)` declares a recording boundary and starts capture.
3. `stop_streaming()` stops capture and waits for the audio loop to finish.
4. `flush(recording_id, force_complete=True)` awaits and returns the matching
   `flush_complete` transcription response; deliver it before the next recording.
5. `disconnect()` closes the connection and releases resources.

```python
# After capturing recording A:
await client.stop_streaming()
final_a = await client.flush(recording_a, force_complete=True)
for segment in final_a.segments:
  print(segment.start, segment.end, segment.text)
await client.start_streaming(recording_b)
```

`flush()` also queues its result as a transcription event, so consumers should not
apply the returned result and its queued event as separate transcript additions.
`start_streaming()` waits for a preceding pending flush.

The async context manager connects and disconnects automatically. A recording
identifier associates audio, controls, and results with that recording. Each new
recording starts with fresh transcript state and recording-relative timestamp
zero; it does not require a new socket. Starting recording B must not invalidate
recording A's accepted audio, recognition, transcript, or pending flush. If B's
audio is accepted while A finishes, it queues until A's final result is delivered.

A flush requests prompt processing through its accepted audio boundary. Its
`force_complete` choice determines whether pending segments are finalized. A
non-force-completing flush still delivers an update and `flush_complete=True`,
but does not itself finalize pending text. An empty recording or an already
consumed buffer also receives the acknowledgement without blocking other work.
Flush completion acknowledges the request, not necessarily segment completion.

`cancel_utterance(recording_id)` stops capture, sends cancellation, and clears the
current recording identifier and queued transcription events without awaiting an
acknowledgement. Cancellation discards the current utterance, clears its
utterance-local transcript history, and suppresses outstanding results from that
work. Start recording again on the same connection afterward.
Controls for an earlier recording must not change the current recording.

Clients iterate over events including `connected`, `transcription`,
`disconnected`, `reconnecting`, and `reconnected`. Live transcribers automatically
retry unexpected disconnects. Legacy `language_detection` events remain decodable,
but the English-only backend does not perform language detection.

## Transcription data

A `TranscriptionMessage` contains its stream, message timestamp, language, and
segments. Recording results carry their `recording_id`; a flush response sets
`flush_complete=True`.

Each `Segment` contains:

- `id`: the existing chained CRC64 identifier assigned on completion
- `start` and `end`: seconds on the recording timeline
- `text`: recognized text
- `words`: recognized words with native timestamps
- `completed`: whether this segment is finalized
- `duration`: `end - start`

Each `Word` contains `start`, `end`, and `word`. CPU boundaries derive from native
token onsets and durations at 80 ms granularity; Vulkan uses native word timestamps
converted from milliseconds. Neither backend uses proportional duration estimates.
Segment and word timestamps share the original recording timeline, including
silent gaps skipped by VAD.

With VAD enabled, ordinary silence-based completion and the trailing incomplete
segment convention remain. With VAD disabled, **all pending segments remain
incomplete until an explicit force-completing flush**. Time, silence, new audio,
and becoming non-trailing do not finalize them. Consumers must support revisions
to multiple incomplete segments, not just the last one.

Segments do not contain `seek`, `tokens`, `compression_ratio`, `temperature`,
`avg_logprob`, or `avg_probability`. Words do not contain `probability`. No
replacement confidence values are fabricated.
