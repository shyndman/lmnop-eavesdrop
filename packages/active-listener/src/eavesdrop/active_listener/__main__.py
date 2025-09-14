"""Main entry point for eavesdrop active listener application."""

import sys

from eavesdrop.active_listener.cli import ActiveListener


def main() -> None:
  """Main entry point for active-listener command."""
  try:
    cli = ActiveListener.parse()
    cli.start()
  except KeyboardInterrupt:
    print("\nReceived interrupt signal, shutting down...")
    sys.exit(0)
  except Exception as e:
    print(f"Fatal error: {e}")
    sys.exit(1)


if __name__ == "__main__":
  main()
