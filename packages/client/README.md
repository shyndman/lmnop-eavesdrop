# Eavesdrop Client

Python client for English speech transcription with the Eavesdrop server's
Orukeet CPU (INT8 ONNX) or Vulkan (Q8 GGUF) backend. Supports live microphone capture, finite audio files, and
RTSP stream subscriptions.

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

Native word timestamps are included by default. Pass `word_timestamps=False` to
omit word-level output. The optional language hint accepts only `"en"` or `None`;
both select English. Model selection is server-side. Prompting, hotwords, beam
search overrides, per-session models, and translation are unsupported.

See [the API guide](docs/api-client.md) and the
[microphone example](examples/simple_transcriber.py) for the supported interface.
