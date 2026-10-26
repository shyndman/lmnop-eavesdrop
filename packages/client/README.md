# Eavesdrop Client

Python client for English speech transcription with the Eavesdrop server's
Orukeet CPU (INT8 ONNX) or Vulkan (Q8 GGUF) backend. Streams microphone audio over
WebSocket, with multiple recordings on one connection.

```python
import asyncio
import uuid

from eavesdrop.client import EavesdropClient
from eavesdrop.wire import TranscriptionMessage

RECORDING_SECONDS = 5


def show_transcription(message: TranscriptionMessage) -> None:
  for segment in message.segments:
    print(segment.start, segment.end, segment.text, segment.completed)


async def main() -> None:
  async with EavesdropClient.transcriber(
    audio_device="default", on_transcription=show_transcription
  ) as client:
    recording_id = uuid.uuid4().hex
    await client.start_streaming(recording_id)
    try:
      await asyncio.sleep(RECORDING_SECONDS)
    finally:
      await client.stop_streaming()
      await client.flush(recording_id, force_complete=True)


asyncio.run(main())
```

Every transcription contains timed segments and word timestamps on the original
recording timeline, including silence. The optional language hint accepts only
`"en"` or `None`; both select English. The server selects the model.

For normal completion, stop recording A, flush it, and await its final response
before starting recording B. Stopping capture drains already captured audio before
the flush is sent. The transcription callback receives the flush response before
`flush()` returns that same response; apply it once. Each recording has a fresh
identifier, transcript, and timestamp zero. Starting B preserves A's accepted work.

A non-force-completing flush requests prompt processing without explicitly
completing pending text. With VAD enabled, silence can still complete timed
segments. With VAD disabled, one incomplete timed segment accumulates all text and
word timestamps until an explicit force-completing flush.

Cancellation discards the current recording's unfinished work and immediately
fails its pending flush. Disconnection also fails a pending flush. Neither leaves
a waiter blocking the next recording.

See [the API guide](docs/api-client.md) and the
[microphone example](examples/simple_transcriber.py) for the supported interface.
