# Eavesdrop Client API

`EavesdropClient` connects to an Eavesdrop transcription server using the
English-only Orukeet CPU (INT8 ONNX) or Vulkan (Q8 GGUF) backend. It provides live transcription,
finite-file transcription, and RTSP subscriptions.

## Live transcription

```python
import uuid

from eavesdrop.client import EavesdropClient


async def transcribe_microphone() -> None:
  async with EavesdropClient.transcriber(
    host="localhost",
    port=9090,
    audio_device="default",
    word_timestamps=True,
    language="en",
  ) as client:
    await client.start_streaming(uuid.uuid4().hex)
    async for event in client:
      if event.family != "transcription":
        continue
      for segment in event.message.segments:
        print(f"{segment.start:.2f}-{segment.end:.2f}: {segment.text}")
        if segment.words is not None:
          for word in segment.words:
            print(f"  {word.start:.2f}-{word.end:.2f}: {word.word}")
```

The transcriber factory accepts:

- `host: str = "localhost"` and `port: int = 9090`
- `audio_device: str = "default"`
- `on_capture: Callable[[bytes], None] | None = None`, for captured PCM chunks
- `word_timestamps: bool = True`, to include native word timings
- `send_last_n_segments: int | None = None`, a positive segment-window limit
- `language: Literal["en"] | None = None`, an English-only compatibility hint

`None` for language also means English; there is no language auto-detection.
Word timestamps require an actual boolean. They are enabled by default, and
`False` omits words from the result.

Model selection is controlled by the server. The client no longer accepts
`initial_prompt`, `hotwords`, `beam_size`, `model`, or `task`. Translation is
unsupported. Old fields in `UserTranscriptionOptions` or a setup JSON message
raise validation errors rather than being silently ignored.

## File transcription

Call `transcribe_file()` on an idle transcriber client. It manages its own
connection and uses the finite-file source mode automatically.

```python
from eavesdrop.client import EavesdropClient


async def transcribe_recording() -> None:
  client = EavesdropClient.transcriber(word_timestamps=True)
  result = await client.transcribe_file("recording.wav", timeout_s=120.0)
  print(result.text)
  for segment in result.segments:
    print(segment.start, segment.end, segment.text)
  for warning in result.warnings:
    print(f"Warning: {warning}")
```

`FileTranscriptionResult` contains `segments`, `text`, `language`, and `warnings`.
A file operation cannot run alongside live streaming or another file operation
on the same client.

## RTSP subscriptions

```python
from eavesdrop.client import EavesdropClient


async def subscribe() -> None:
  async with EavesdropClient.subscriber(
    stream_names=["office_audio", "meeting_room_1"],
    host="localhost",
    port=9090,
  ) as client:
    async for event in client:
      if event.family == "transcription":
        for segment in event.message.segments:
          print(f"[{event.stream}] {segment.text}")
```

Subscribers receive transcription messages from server-managed streams and do
not send microphone audio or transcription setup options.

## Session options

For direct construction or a connection-specific override, use the strict wire
model:

```python
from eavesdrop.wire import TranscriptionSourceMode, UserTranscriptionOptions

options = UserTranscriptionOptions(
  source_mode=TranscriptionSourceMode.LIVE,
  send_last_n_segments=3,
  word_timestamps=True,
  language="en",
)
```

Pass it as `transcription_options` to the constructor or `setup_options` to
`connect()`. `source_mode` preserves live versus finite-file routing intent.
Use `transcribe_file()` for normal file uploads.

## Lifecycle and events

1. `connect()` establishes the connection and initializes capture resources.
2. `start_streaming(recording_id)` declares a recording epoch and starts capture.
3. `stop_streaming()` stops capture and waits for the audio loop to finish.
4. `flush(recording_id)` requests a final response for the active recording.
5. `disconnect()` closes the connection and releases resources.

The async context manager connects and disconnects automatically. Live clients
iterate over events, including `connected`, `transcription`, `disconnected`,
`reconnecting`, and `reconnected`. Live transcribers automatically retry
unexpected disconnects. File operations do not reconnect automatically.

Legacy `language_detection` events remain decodable for compatibility, but the
English-only backend does not perform language detection.

## Transcription data

A `TranscriptionMessage` contains its stream, message timestamp, language, and
segments. Live messages may include a `recording_id`; a flush response sets
`flush_complete=True`.

Each `Segment` contains:

- `id`: the existing chained CRC64 identifier assigned on completion
- `start` and `end`: seconds on the recording timeline
- `text`: recognized text
- `words`: words grouped from native token timings, or `None` when omitted
- `completed`: whether this segment is finalized
- `duration`: `end - start`

Each `Word` contains `start`, `end`, and `word`. CPU boundaries derive from native
token onsets and durations at 80 ms granularity; Vulkan uses native word timestamps
converted from milliseconds. Neither backend uses proportional duration estimates.
Segment and word timestamps share the original recording timeline, including
silent gaps skipped by VAD. Completed chain IDs and the trailing incomplete
segment convention are unchanged.

Segments no longer contain `seek`, `tokens`, `compression_ratio`, `temperature`,
`avg_logprob`, or `avg_probability`. Words no longer contain `probability`. No
replacement confidence values are fabricated.
