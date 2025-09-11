#!/usr/bin/env python3
"""
Test script for EavesdropClient subscriber mode.

Usage:
    python simple_subscriber.py --host localhost --port 8080 --streams stream1,stream2
"""

import argparse
import asyncio
import sys

from eavesdrop.client import EavesdropClient


async def test_subscriber(host: str, port: int, stream_names: list[str]):
  """Test the subscriber client with a real server."""
  print("Creating subscriber client...")
  print(f"  Host: {host}")
  print(f"  Port: {port}")
  print(f"  Streams: {stream_names}")

  try:
    # Create subscriber client
    client = EavesdropClient.subscriber(stream_names=stream_names, host=host, port=port)
    print("✅ Client created successfully")

    # Test connection
    print("\nConnecting to server...")
    await client.connect()
    print("✅ Connected successfully")
    print(f"   Connection status: {client.is_connected()}")
    print(f"   Streaming status: {client.is_streaming()} (should always be False for subscribers)")

    # Listen for transcriptions from subscribed streams
    print(
      f"\nListening for transcriptions from {len(stream_names)} streams (press Ctrl+C to stop)..."
    )
    message_count = 0
    streams_seen = set()

    async for message in client:
      message_count += 1
      streams_seen.add(message.stream)

      print(f"\n[Message {message_count}]")
      print(f"  Stream: {message.stream}")
      print(f"  Language: {message.language}")
      print(f"  Segments: {len(message.segments)}")

      if message.segments:
        # Show just the first segment for brevity
        segment = message.segments[0]
        print(f"    Text: '{segment.text}' ({segment.start:.2f}s - {segment.end:.2f}s)")
        if len(message.segments) > 1:
          print(f"    ... and {len(message.segments) - 1} more segments")

      print(f"  Streams seen so far: {sorted(streams_seen)}")

      if message_count >= 20:  # Stop after 20 messages for testing
        print("\nReceived 20 messages, stopping test...")
        break

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
      if client.is_connected():
        await client.disconnect()
        print("✅ Disconnected")

  print("\n🎉 Test completed successfully!")
  print(f"   Messages received: {message_count}")
  print(f"   Streams with messages: {sorted(streams_seen)}")
  return 0


def main():
  """Main entry point."""
  parser = argparse.ArgumentParser(description="Test EavesdropClient subscriber mode")
  parser.add_argument(
    "--server",
    default="localhost:8080",
    help="Server address as host:port (default: localhost:8080)",
  )
  parser.add_argument(
    "--streams", required=True, help="Comma-separated list of stream names to subscribe to"
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

  # Parse stream names
  stream_names = [s.strip() for s in args.streams.split(",")]
  if not stream_names or not all(stream_names):
    print("Error: --streams must contain at least one non-empty stream name")
    return 1

  try:
    return asyncio.run(test_subscriber(host, port, stream_names))
  except KeyboardInterrupt:
    print("\nExiting...")
    return 0


if __name__ == "__main__":
  sys.exit(main())
