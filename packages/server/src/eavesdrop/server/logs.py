"""Centralized logging configuration for Eavesdrop using structlog."""

import logging
import os
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from io import StringIO
from typing import Any

import structlog
from structlog.dev import (
  BLUE,
  BRIGHT,
  CYAN,
  DIM,
  GREEN,
  MAGENTA,
  RED,
  RED_BACK,
  RESET_ALL,
  YELLOW,
  Column,
  ConsoleRenderer,
  KeyValueColumnFormatter,
  _pad,
)
from structlog.typing import EventDict, Processor, WrappedLogger

try:
  import numpy as np

  numpy_installed = True
except ImportError:
  numpy_installed = False

# Store program start time for relative timestamps
_PROGRAM_START_TIME = time.time()


def hex_to_ansi_fg(hex_color: int) -> str:
  """Convert hex color (e.g., 0xad8a89) to ANSI 24-bit foreground escape code."""
  r = (hex_color >> 16) & 0xFF
  g = (hex_color >> 8) & 0xFF
  b = hex_color & 0xFF
  return f"\x1b[38;2;{r};{g};{b}m"


@dataclass
class RegexValueColumnFormatter:
  """
  Format a key-value pair with regex-based value styling.

  Like KeyValueColumnFormatter, but allows mapping values to styles based on
  regular expression patterns.

  :param key_style: The style to apply to the key. If None, the key is omitted.
  :param value_style_map: A list of (regex_pattern, style) tuples. The first
      pattern that matches the value will determine its style.
  :param default_value_style: The style to use if no regex patterns match.
  :param reset_style: The style to apply whenever a style is no longer needed.
  :param value_repr: A callable that returns the string representation of the value.
  :param width: The width to pad the value to. If 0, no padding is done.
  :param prefix: A string to prepend to the formatted key-value pair. May contain
      styles.
  :param postfix: A string to append to the formatted key-value pair. May contain
      styles.
  """

  key_style: str | None
  value_style_map: list[tuple[str, str]]
  default_value_style: str
  reset_style: str
  value_repr: Callable[[object], str]
  width: int = 0
  prefix: str = ""
  postfix: str = ""

  def __post_init__(self) -> None:
    """Compile regex patterns for efficiency."""
    self._compiled_patterns = [
      (re.compile(pattern), style) for pattern, style in self.value_style_map
    ]

  def __call__(self, key: str, value: object) -> str:
    sio = StringIO()

    if self.prefix:
      sio.write(self.prefix)
      sio.write(self.reset_style)

    if self.key_style is not None:
      sio.write(self.key_style)
      sio.write(key)
      sio.write(self.reset_style)
      sio.write("=")

    # Determine value style based on regex matching
    value_str = self.value_repr(value)
    value_style = self.default_value_style

    for pattern, style in self._compiled_patterns:
      if pattern.search(value_str):
        value_style = style
        break

    sio.write(value_style)
    sio.write(_pad(value_str, self.width))
    sio.write(self.reset_style)

    if self.postfix:
      sio.write(self.postfix)
      sio.write(self.reset_style)

    return sio.getvalue()


class NerdStyles:
  reset = RESET_ALL
  bright = BRIGHT

  level_critical = RED
  level_exception = RED
  level_error = RED
  level_warn = YELLOW
  level_info = GREEN
  level_debug = GREEN
  level_notset = RED_BACK

  timestamp = DIM
  logger_name = BLUE
  kv_key = CYAN
  kv_value = MAGENTA


class _FloatPrecisionProcessor:
  """
  A structlog processor for rounding floats. Both as single numbers or in data structures like
  (nested) lists, dicts, or numpy arrays.

  Inspired by https://github.com/underyx/structlog-pretty/blob/master/structlog_pretty/processors.py
  """

  def __init__(
    self,
    digits: int = 3,
    only_fields: set[str] = set(),
    not_fields: set[str] = set(),
    np_array_to_list: bool = True,
  ):
    """
    Create a FloatRounder processor. That rounds floats to the given number of digits.

    :param digits: The number of digits to round to
    :param only_fields: A set specifying the fields to round (None = round all fields except
      not_fields)
    :param not_fields: A set specifying fields not to round
    :param np_array_to_list: Whether to cast np.array to list for nicer printing
    """
    self.digits = digits
    self.np_array_to_list = np_array_to_list
    self.only_fields = only_fields
    self.not_fields = not_fields

  def _round(self, value: Any):
    """
    Round floats, unpack lists, convert np.arrays to lists

    :param value: The value/data structure to round
    :returns: The rounded value
    """
    # round floats
    if isinstance(value, float):
      return round(value, self.digits)
    # convert np.array to list
    if self.np_array_to_list:
      if isinstance(value, np.ndarray):
        return self._round(list(value))
    # round values in lists recursively (to handle lists of lists)
    if isinstance(value, list):
      for idx, item in enumerate(value):
        value[idx] = self._round(item)
      return value
    # similarly, round values in dicts recursively
    if isinstance(value, dict):
      for k, v in value.items():
        value[k] = self._round(v)
      return value
    # return any other values as they are
    return value

  def __call__(self, _: WrappedLogger, __: str, event_dict: EventDict):
    for key, value in event_dict.items():
      if not len(self.only_fields) and key not in self.only_fields:
        continue
      if not len(self.not_fields) and key in self.not_fields:
        continue
      if isinstance(value, bool):
        continue  # don't convert True to 1.0

      event_dict[key] = self._round(value)
      if isinstance(value, float):
        print(f"!!! Rounded {key}: {value} -> {event_dict[key]}", file=sys.stderr)
    return event_dict


def _relative_time_processor(
  _logger: structlog.stdlib.BoundLogger, _method_name: str, event_dict: EventDict
) -> EventDict:
  """Add relative timestamp since program start with hours:minutes:seconds.milliseconds format."""
  elapsed = time.time() - _PROGRAM_START_TIME

  # Calculate hours, minutes, seconds
  hours = int(elapsed // 3600)
  minutes = int((elapsed % 3600) // 60)
  seconds = elapsed % 60

  # ANSI color codes: \x1b[2m for dim/gray, \x1b[90m for darker gray, \x1b[0m to reset
  gray = "\x1b[2m"  # Normal gray (like original timestamp)
  dark_gray = "\x1b[90m"  # Darker gray for zeros
  reset = "\x1b[0m"

  separator = f"{gray}:{reset}"

  # Format hours with darker gray if zero, normal gray otherwise
  hours_str = f"{gray}{hours:02d}{reset}{separator}" if hours != 0 else ""

  # Format minutes with darker gray if zero, normal gray otherwise
  minutes_str = f"{gray}{minutes:02d}{reset}{separator}" if minutes != 0 and hours != 0 else ""

  # Always show seconds in normal gray
  seconds_str = f"{gray}{seconds:06.3f}{reset}"

  # Colons in normal gray too
  time_str = f"{dark_gray}+{reset}{hours_str}{minutes_str}{seconds_str}"

  event_dict["timestamp"] = time_str
  return event_dict


def _compact_level_processor(
  _logger: structlog.stdlib.BoundLogger, _method_name: str, event_dict: EventDict
) -> EventDict:
  """Convert log levels to compact 4-character format with 24-bit colors and darker brackets."""
  # 24-bit color codes: \x1b[38;2;R;G;Bm for foreground, \x1b[48;2;R;G;Bm for background
  reset = "\x1b[0m"

  # Original colors and their 10% darkened versions for brackets
  level_mapping = {
    "debug": {
      "text": f"{hex_to_ansi_fg(0x908CAA)}dbug{reset}",  # fg #908caa
      "bracket": hex_to_ansi_fg(0x827E99),  # 10% darker
    },
    "info": {
      "text": f"{hex_to_ansi_fg(0x9CCFD8)}info{reset}",  # fg #9ccfd8
      "bracket": hex_to_ansi_fg(0x8CBAC2),  # 10% darker
    },
    "warning": {
      "text": f"{hex_to_ansi_fg(0xF6C177)}warn{reset}",  # fg #f6c177
      "bracket": hex_to_ansi_fg(0xDDAE6B),  # 10% darker
    },
    "error": {
      "text": f"{hex_to_ansi_fg(0xEB6F92)}eror{reset}",  # fg #eb6f92
      "bracket": hex_to_ansi_fg(0xD46483),  # 10% darker
    },
    "exception": {
      "text": f"{hex_to_ansi_fg(0xEB6F92)}exc!{reset}",  # fg #eb6f92
      "bracket": hex_to_ansi_fg(0xD46483),  # 10% darker
    },
    "critical": {
      "text": f"\x1b[48;2;235;111;146;38;2;33;32;46mcrit{reset}",  # bg #eb6f92, fg #21202e
      "bracket": hex_to_ansi_fg(0xD46483),  # 10% darker
    },
  }

  if "level" in event_dict:
    original_level = event_dict["level"]
    if original_level in level_mapping:
      colors = level_mapping[original_level]
      event_dict["level"] = (
        f"{colors['bracket']}[{reset}{colors['text']}{colors['bracket']}]{reset}"
      )
    else:
      event_dict["level"] = original_level

  return event_dict


def _debug_event_colorer(
  _logger: structlog.stdlib.BoundLogger, _method_name: str, event_dict: EventDict
) -> EventDict:
  """Color the event text purple for debug level messages."""
  # Check both the original level and processed level
  level = event_dict.get("level", "")
  if ("debug" in str(level).lower() or level == "debug") and "event" in event_dict:
    # Color debug event text with #908caa
    debug_color = hex_to_ansi_fg(0x908CAA)
    reset = "\x1b[0m"
    event_dict["event"] = f"{debug_color}{event_dict['event']}{reset}"

  return event_dict


def setup_logging(
  level: str = "INFO", json_output: bool = False, correlation_id: str | None = None
) -> None:
  """Configure structured logging for the application."""

  # Configure processors
  shared_processors: list[Processor] = [
    structlog.stdlib.filter_by_level,
    structlog.stdlib.add_logger_name,
    structlog.stdlib.add_log_level,
    _debug_event_colorer,  # Run before level processing
    _compact_level_processor,
    structlog.stdlib.PositionalArgumentsFormatter(),
    structlog.stdlib.ExtraAdder(),
    _FloatPrecisionProcessor(digits=3),
    _relative_time_processor,
    structlog.processors.StackInfoRenderer(),
    structlog.processors.format_exc_info,
  ]

  # Add correlation ID if provided
  if correlation_id:
    shared_processors.insert(0, structlog.contextvars.merge_contextvars)
    structlog.contextvars.bind_contextvars(correlation_id=correlation_id)

  # Configure output format
  if json_output:
    log_renderer = structlog.processors.JSONRenderer()
  else:
    event_key: str = "event"
    timestamp_key: str = "timestamp"
    logger_name_formatter = KeyValueColumnFormatter(
      key_style=None,
      value_style=hex_to_ansi_fg(0x7D6B95),
      reset_style=RESET_ALL,
      value_repr=str,
      prefix="[",
      postfix="]",
    )

    log_renderer = ConsoleRenderer(
      colors=True,
      columns=[
        # Default formatter
        Column(
          "",
          RegexValueColumnFormatter(
            key_style=hex_to_ansi_fg(0x6E6A86),
            value_style_map=[
              (
                # Integers (including negative) get bright green styling
                r"^True|False$",
                hex_to_ansi_fg(0x6E6A86),
              ),
              (
                # Integers (including negative) get bright green styling
                r"^-?\d+$",
                hex_to_ansi_fg(0xF6C177),
              ),
              (
                # Floats (including negative) get bright yellow styling
                r"^-?\d*\.\d+$",
                hex_to_ansi_fg(0xF6C177),
              ),
              (
                # Numeric durations (123ms, 45.67s, 12h, etc.)
                r"^-?\d*\.?\d+(?:h|m|s|ms|us|µs)$",
                hex_to_ansi_fg(0x9CCFD8),
              ),
            ],
            default_value_style="",
            reset_style=RESET_ALL,
            value_repr=str,
          ),
        ),
        Column(
          timestamp_key,
          KeyValueColumnFormatter(
            key_style=None,
            value_style=DIM,
            reset_style=RESET_ALL,
            value_repr=str,
          ),
        ),
        Column(
          "level",
          KeyValueColumnFormatter(
            key_style=None,
            value_style="",
            reset_style=RESET_ALL,
            value_repr=str,
          ),
        ),
        Column("logger_name", logger_name_formatter),
        Column("logger", logger_name_formatter),
        Column(
          event_key,
          KeyValueColumnFormatter(
            key_style=None,
            value_style=BRIGHT,
            reset_style=RESET_ALL,
            value_repr=str,
            width=30,
          ),
        ),
      ],
    )

  # Configure structlog
  structlog.configure(
    processors=shared_processors + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
  )

  formatter = structlog.stdlib.ProcessorFormatter(
    foreign_pre_chain=shared_processors,
    processors=[
      structlog.stdlib.ProcessorFormatter.remove_processors_meta,
      log_renderer,
    ],
  )

  # setup logging
  handler = logging.StreamHandler()
  handler.setFormatter(formatter)
  root_logger = logging.getLogger()
  root_logger.addHandler(handler)
  root_logger.setLevel(level)

  # Propogate the logs of some libraries
  for liblog in [logging.getLogger(_liblog) for _liblog in ["websockets"]]:
    liblog.handlers.clear()
    liblog.setLevel(logging.WARNING)
    liblog.propagate = True

  # And suppress the logs of others
  # for liblog in [logging.getLogger(_liblog) for _liblog in ["websockets"]]:
  #   liblog.handlers.clear()
  #   liblog.propagate = False


def get_logger(
  name: str | None = None, *args: list[Any], **initial_values: Any
) -> structlog.stdlib.BoundLogger:
  """Get a structured logger instance."""
  return structlog.get_logger(*([name] + list(args)), **initial_values)


def setup_logging_from_env() -> None:
  """Setup logging using environment variables."""
  log_level = os.getenv("LOG_LEVEL", "INFO").upper()
  json_output = os.getenv("JSON_LOGS", "false").lower() in ("true", "1", "yes", "on")
  correlation_id = os.getenv("CORRELATION_ID")

  setup_logging(level=log_level, json_output=json_output, correlation_id=correlation_id)
