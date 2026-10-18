# Eavesdrop Wire

Shared Pydantic messages and JSON codecs for Eavesdrop's English-only NVIDIA
Parakeet TDT v2 transcription service.

## Session setup

```python
from eavesdrop.wire import (
  TranscriptionSetupMessage,
  TranscriptionSourceMode,
  UserTranscriptionOptions,
  serialize_message,
)

message = TranscriptionSetupMessage(
  stream="microphone",
  options=UserTranscriptionOptions(
    source_mode=TranscriptionSourceMode.LIVE,
    send_last_n_segments=3,
    word_timestamps=True,
    language="en",
  ),
)
print(serialize_message(message))
```

`UserTranscriptionOptions` is strict and forbids unknown fields:

- `source_mode`: `TranscriptionSourceMode.LIVE` (default) or `.FILE`
- `send_last_n_segments`: a positive integer or `None` (default)
- `word_timestamps`: a boolean, default `True` for native word timings
- `language`: `"en"` or `None` (default); both select English

Old decoder fields (`initial_prompt`, `hotwords`, `beam_size`, `model`, and
`task`) fail validation. There is no translation task or per-session model
selection. JSON source modes remain `"live"` and `"file"`.

## Transcript results

`Segment` carries `id`, recording-relative `start` and `end`, `text`, optional
`words`, `completed`, and computed `duration`. `Word` carries `start`, `end`, and
`word`. Word and segment timings use the same recording timeline.

Whisper-specific segment metadata (`seek`, `tokens`, `compression_ratio`,
`temperature`, `avg_logprob`, and `avg_probability`) and word `probability` have
been removed. No dummy scores are emitted.

Chained CRC64 segment IDs, recording IDs, flush controls, and recording-relative
timestamps retain their existing semantics. The internal `time_offset`
compatibility helper is excluded from JSON. The language-detection message type
remains decodable for legacy compatibility, but the Parakeet backend does not
perform language detection.
