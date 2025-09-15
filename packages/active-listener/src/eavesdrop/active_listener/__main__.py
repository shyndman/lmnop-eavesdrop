"""Main entry point for eavesdrop active listener application."""

import sys

from eavesdrop.active_listener.cli import ActiveListener
from eavesdrop.common import get_logger, setup_logging_from_env

logger = get_logger("main")


def main() -> None:
  setup_logging_from_env()

  """Main entry point for active-listener command."""
  try:
    cli = ActiveListener.parse()
    cli.start()
  except KeyboardInterrupt:
    logger.info("\nReceived interrupt signal, shutting down...")
    sys.exit(0)
  except Exception:
    logger.exception("Fatal error")
    sys.exit(1)


if __name__ == "__main__":
  main()
