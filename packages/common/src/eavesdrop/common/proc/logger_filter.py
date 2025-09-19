from structlog import DropEvent
from structlog.typing import EventDict, WrappedLogger


class LoggerFilterProcessor:
  """
  A structlog processor that filters log events based on the logger name.

  This processor allows you to specify a logger name, and only log events
  originating from that logger (or its children) will be processed. All other
  log events will be ignored.

  Example usage:
      import structlog

      structlog.configure(
          processors=[
              LoggerFilterProcessor("my_logger"),
              structlog.processors.JSONRenderer()
          ]
      )
      logger = structlog.get_logger("my_logger")
      logger.info("This will be logged")
  """

  def __init__(self, logger_name: str):
    """
    Initialize the LoggerFilterProcessor.

    :param logger_name: The name of the logger to filter on. Only log events
                        from this logger or its children will be processed.
    """
    self.logger_name = logger_name

  def __call__(self, logger: WrappedLogger, __: str, event_dict: EventDict):
    """
    Process a log event.

    :param logger: The logger instance that generated the event.
    :param method_name: The logging method name (e.g., "info", "error").
    :param event_dict: The log event dictionary.
    :return: The event_dict if the event should be processed, or None to ignore it.
    """
    if logger.name == self.logger_name or logger.name.startswith(f"{self.logger_name}."):
      return event_dict
    else:
      raise DropEvent(f"Filtered out by LoggerFilterProcessor, logger={logger.name}")
