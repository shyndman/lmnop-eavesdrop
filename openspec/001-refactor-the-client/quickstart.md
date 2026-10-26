# Quickstart Guide: EavesdropClient Library

## Overview
This guide demonstrates the essential usage patterns for the live-only EavesdropClient library. All examples assume a running eavesdrop server. Every transcription includes native segment and individual word timestamps; no option is needed to enable them.

Recording lifetime is independent of connection lifetime. Complete recording A with a force-completing flush and await its final transcript and flush-complete response before starting recording B. Each recording starts a fresh transcript and recording-relative timeline. A non-force-completing flush requests an update without completing pending segments. With VAD disabled, all pending segments remain provisional until explicit force completion; with VAD enabled, silence-based completion remains available. Cancellation discards the active utterance rather than completing it.

## Basic Transcriber Mode Usage

### Simple Audio Transcription
```python
from eavesdrop.client import EavesdropClient

# Basic transcriber usage with context manager
async def basic_transcription():
    async with EavesdropClient.transcriber(host="localhost", port=9090) as client:
        await client.start_streaming()
        
        async for transcription in client:
            print(f"Language: {transcription.language}")
            for segment in transcription.segments:
                print(f"  {segment.start:.2f}s: {segment.text}")
                
            # Stop on specific command
            if "stop recording" in segment.text.lower():
                await client.stop_streaming()
                break
```

### Advanced Transcriber Configuration
```python
async def advanced_transcription():
    client = EavesdropClient.transcriber(
        host="home-brainbox",
        port=9090,
        audio_device="USB Audio",  # Specific device
        on_ready=lambda backend: print(f"Ready with {backend}"),
        on_error=lambda msg: print(f"Error: {msg}"),
        on_streaming_started=lambda: print("Started streaming"),
        on_streaming_stopped=lambda: print("Stopped streaming")
    )
    
    await client.connect()
    await client.start_streaming()
    
    try:
        async for transcription in client:
            # Process rich transcription data
            for segment in transcription.segments:
                print(f"{segment.start:.2f}-{segment.end:.2f}: {segment.text}")
                if segment.confidence:
                    print(f"  Confidence: {segment.confidence:.2f}")
    finally:
        await client.disconnect()
```

## Audio Device Selection

### List Available Devices
```python
import sounddevice as sd

# List all available audio devices
print("Available audio devices:")
devices = sd.query_devices()
for i, device in enumerate(devices):
    if device['max_input_channels'] > 0:
        print(f"  {i}: {device['name']}")
```

### Device Selection Methods
```python
# By index
client1 = EavesdropClient.transcriber(audio_device=1)

# By exact name
client2 = EavesdropClient.transcriber(audio_device="USB Audio Device")

# By substring match
client3 = EavesdropClient.transcriber(audio_device="USB")

# Use default device
client4 = EavesdropClient.transcriber(audio_device=None)
```

## Error Handling Patterns

### Connection Error Handling
```python
async def robust_connection():
    try:
        async with EavesdropClient.transcriber() as client:
            await client.start_streaming()
            async for transcription in client:
                # Process transcriptions
                pass
    except ConnectionError as e:
        print(f"Failed to connect: {e}")
    except InvalidAudioDeviceError as e:
        print(f"Audio device issue: {e}")
```

### Runtime Error Handling via Callbacks
```python
def error_handler(error_msg: str):
    if "device" in error_msg.lower():
        print(f"Device error occurred: {error_msg}")
        # Implement recovery logic
    else:
        print(f"General error: {error_msg}")

client = EavesdropClient.transcriber(on_error=error_handler)
```

## Lifecycle Management Patterns

### Manual Connection Management
```python
async def manual_lifecycle():
    client = EavesdropClient.transcriber()
    
    # Connect manually
    await client.connect()
    assert client.is_connected()
    
    # Start/stop streaming cycles
    await client.start_streaming()
    assert client.is_streaming()
    
    # Process some transcriptions...
    
    await client.stop_streaming()
    assert not client.is_streaming()
    assert client.is_connected()  # Still connected
    
    # Can restart without reconnecting
    await client.start_streaming()
    
    # Clean up
    await client.disconnect()
    assert not client.is_connected()
```

### Context Manager (Recommended)
```python
async def context_managed():
    # Automatic connection/disconnection
    async with EavesdropClient.transcriber() as client:
        # Connection established automatically
        await client.start_streaming()
        
        async for transcription in client:
            # Process transcriptions
            pass
        # Disconnection handled automatically on exit
```

## Testing Scenarios

### Transcriber Mode Integration Test
```python
import pytest

@pytest.mark.asyncio
async def test_transcriber_basic_flow():
    """Test basic transcriber mode functionality."""
    client = EavesdropClient.transcriber(
        host="localhost",
        port=9090,
        audio_device=None  # Use default
    )
    
    # Test connection
    await client.connect()
    assert client.is_connected()
    assert not client.is_streaming()
    
    # Test streaming control
    await client.start_streaming()
    assert client.is_streaming()
    
    # Should receive at least one transcription
    transcription = await client.__anext__()
    assert isinstance(transcription.stream, str)
    assert isinstance(transcription.segments, list)
    
    # Test stop streaming
    await client.stop_streaming()
    assert not client.is_streaming()
    assert client.is_connected()
    
    # Cleanup
    await client.disconnect()
    assert not client.is_connected()
```

### Error Condition Tests
```python
@pytest.mark.asyncio
async def test_invalid_audio_device():
    """Test invalid audio device handling."""
    with pytest.raises(InvalidAudioDeviceError):
        EavesdropClient.transcriber(audio_device="nonexistent_device")

@pytest.mark.asyncio
async def test_connection_failure():
    """Test connection failure handling."""
    client = EavesdropClient.transcriber(host="invalid.host", port=99999)
    
    with pytest.raises(ConnectionError):
        await client.connect()
```

## Performance Guidelines

### Optimal Usage Patterns
```python
# ✅ Good: Process transcriptions as they arrive
async with EavesdropClient.transcriber() as client:
    await client.start_streaming()
    async for transcription in client:
        process_immediately(transcription)

# ❌ Avoid: Accumulating large batches
async with EavesdropClient.transcriber() as client:
    await client.start_streaming()
    batch = []
    async for transcription in client:
        batch.append(transcription)  # Memory leak risk
        if len(batch) > 1000:
            process_batch(batch)
            batch.clear()
```

### Resource Management
```python
# ✅ Good: Use context managers for automatic cleanup
async with EavesdropClient.transcriber() as client:
    # Resources automatically cleaned up

# ✅ Good: Manual cleanup when context manager not suitable  
client = EavesdropClient.transcriber()
try:
    await client.connect()
    # Use client...
finally:
    await client.disconnect()  # Always cleanup
```

This quickstart guide covers the essential patterns for using the EavesdropClient library effectively while following best practices for async programming and resource management.