#!/usr/bin/env python3
"""
Test script for EavesdropClient transcriber mode.

Usage:
    python test_transcriber.py --host localhost --port 8080 [--audio-device default]
"""

import argparse
import asyncio
import sys

from eavesdrop.client import EavesdropClient


async def test_transcriber(host: str, port: int, audio_device: str = "default"):
  """Test the transcriber client with a real server."""
  print("Creating transcriber client...")
  print(f"  Host: {host}")
  print(f"  Port: {port}")
  print(f"  Audio device: {audio_device}")

  try:
    # Create transcriber client
    client = EavesdropClient.transcriber(
      host=host,
      port=port,
      audio_device=audio_device,
      beam_size=5,
      word_timestamps=True,
      initial_prompt="Test transcription session",
    )
    print("✅ Client created successfully")

    # Test connection
    print("\nConnecting to server...")
    await client.connect()
    print("✅ Connected successfully")
    print(f"   Connection status: {client.is_connected()}")

    # Start streaming
    print("\nStarting audio streaming...")
    await client.start_streaming()
    print("✅ Streaming started")
    print(f"   Streaming status: {client.is_streaming()}")

    # Listen for transcriptions
    print("\nListening for transcriptions (press Ctrl+C to stop)...")
    message_count = 0

    async for message in client:
      message_count += 1
      print(f"\n[Message {message_count}]")
      print(f"  Stream: {message.stream}")
      print(f"  Language: {message.language}")
      print(f"  Segments: {len(message.segments)}")

      if message.segments:
        for i, segment in enumerate(message.segments):
          print(
            f"    Segment {i}: '{segment.text}' ({segment.start:.2f}s - {segment.end:.2f}s) "
            f"[{segment.avg_logprob:.2f}]"
          )

  except KeyboardInterrupt:
    print("\n\nInterrupted by user")
  except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback

    traceback.print_exc()
    return 1
  finally:
    # Cleanup
    if "client" in locals():
      print("\nCleaning up...")
      if client.is_streaming():
        await client.stop_streaming()
        print("✅ Stopped streaming")

      if client.is_connected():
        await client.disconnect()
        print("✅ Disconnected")

  print("\n🎉 Test completed successfully!")
  return 0


def main():
  """Main entry point."""
  parser = argparse.ArgumentParser(description="Test EavesdropClient transcriber mode")
  parser.add_argument(
    "--server",
    default="localhost:8080",
    help="Server address as host:port (default: localhost:8080)",
  )
  parser.add_argument(
    "--audio-device", default="default", help="Audio device to use (default: default)"
  )

  args = parser.parse_args()

  # Parse server address
  try:
    if ":" in args.server:
      host, port_str = args.server.rsplit(":", 1)
      port = int(port_str)
    else:
      host = args.server
      port = 8080
  except ValueError:
    print(f"Error: Invalid server address '{args.server}'. Use format 'host:port'")
    return 1

  try:
    return asyncio.run(test_transcriber(host, port, args.audio_device))
  except KeyboardInterrupt:
    print("\nExiting...")
    return 0


if __name__ == "__main__":
  sys.exit(main())
