# Eavesdrop Client

Python client for English speech transcription with the Eavesdrop server's
Orukeet CPU (INT8 ONNX) or Vulkan (Q8 GGUF) backend. Supports live microphone
capture over WebSocket, with multiple recordings on one connection.

```python
import asyncio
import uuid

from eavesdrop.client import EavesdropClient


async def main() -> None:
  async with EavesdropClient.transcriber(audio_device="default") as client:
    await client.start_streaming(uuid.uuid4().hex)
    async for event in client:
      if event.family == "transcription":
        for segment in event.message.segments:
          print(segment.start, segment.end, segment.text)


asyncio.run(main())
```

Every transcription includes native segment and individual word timestamps on
the original recording timeline, including silence. The optional language hint
accepts only `"en"` or `None`; both select English. Model selection is server-side.
Prompting, hotwords, beam search overrides, per-session models, and translation
are unsupported.

For normal completion, stop recording A, send its force-completing flush, and
await and deliver the `flush_complete` result before starting recording B. Each
recording has a fresh identifier, transcript, and timestamp zero. Accepted B audio
queues if A is still finishing; starting B must not discard A's accepted work.
Flush also works with no pending audio. A non-force-completing flush acknowledges
processing without explicitly finalizing pending text. Cancellation instead
discards the current utterance and suppresses its outstanding results; recording
can restart on the same connection.

With VAD enabled, silence can complete segments normally. With VAD disabled,
every pending segment remains incomplete until an explicit force-completing flush,
including segments that are no longer trailing.

See [the API guide](docs/api-client.md) and the
[microphone example](examples/simple_transcriber.py) for the supported interface.
