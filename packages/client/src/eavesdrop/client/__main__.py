#!/usr/bin/env python3
"""
Main entry point for eavesdrop client.
"""

import argparse

from .microphone_client import MicrophoneClient, parse_host_port


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Eavesdrop microphone client for real-time transcription"
  )
  parser.add_argument(
    "host_port",
    nargs="?",
    default="home-brainbox:9090",
    help="Server host:port (default: home-brainbox:9090)",
  )

  args = parser.parse_args()
  host, port = parse_host_port(args.host_port)

  client = MicrophoneClient(host, port)
  client.run()


if __name__ == "__main__":
  main()
