"""Regression tests for shared logging configuration."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import pytest
import structlog

from eavesdrop.common import logs as logs_module
from eavesdrop.common.logs import _make_relative_time_processor, setup_logging
from eavesdrop.common.proc import LoggerFilterProcessor


@dataclass
class _NamedLogger:
  name: str


def test_setup_logging_formats_foreign_stdlib_records() -> None:
  root_logger = logging.getLogger()
  original_handlers = list(root_logger.handlers)
  original_level = root_logger.level
  root_logger.handlers.clear()

  try:
    setup_logging(level="INFO")

    handler = root_logger.handlers[-1]
    formatter = handler.formatter
    assert formatter is not None

    record = logging.getLogger("httpx").makeRecord(
      "httpx",
      logging.INFO,
      __file__,
      0,
      "hello %s",
      ("world",),
      None,
    )

    rendered = formatter.format(record)

    assert "hello world" in rendered
    assert "httpx" in rendered
  finally:
    root_logger.handlers.clear()
    root_logger.handlers.extend(original_handlers)
    root_logger.setLevel(original_level)


def test_setup_logging_raises_httpcore_http11_to_warning() -> None:
  root_logger = logging.getLogger()
  original_handlers = list(root_logger.handlers)
  original_level = root_logger.level
  httpcore_logger = logging.getLogger("httpcore.http11")
  original_httpcore_handlers = list(httpcore_logger.handlers)
  original_httpcore_level = httpcore_logger.level
  original_httpcore_propagate = httpcore_logger.propagate
  root_logger.handlers.clear()
  httpcore_logger.handlers.clear()
  httpcore_logger.setLevel(logging.NOTSET)
  httpcore_logger.propagate = False

  try:
    setup_logging(level="INFO")

    assert httpcore_logger.level == logging.WARNING
    assert httpcore_logger.propagate is True
  finally:
    root_logger.handlers.clear()
    root_logger.handlers.extend(original_handlers)
    root_logger.setLevel(original_level)
    httpcore_logger.handlers.clear()
    httpcore_logger.handlers.extend(original_httpcore_handlers)
    httpcore_logger.setLevel(original_httpcore_level)
    httpcore_logger.propagate = original_httpcore_propagate


def test_relative_time_processor_includes_minutes_during_first_hour(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr(logs_module, "_PROGRAM_START_TIME", 1000.0)
  plain_processor = _make_relative_time_processor(use_color=False)
  colored_processor = _make_relative_time_processor(use_color=True)
  bound_logger = logging.getLogger("test")

  monkeypatch.setattr(time, "time", lambda: 1125.5)
  plain_event = cast(Mapping[str, object], plain_processor(bound_logger, "info", {}))
  colored_event = cast(Mapping[str, object], colored_processor(bound_logger, "info", {}))
  assert plain_event["timestamp"] == "+02:05.500"
  assert "\x1b[" in str(colored_event["timestamp"])
  assert "02" in str(colored_event["timestamp"])

  monkeypatch.setattr(time, "time", lambda: 4605.25)
  hour_event = cast(Mapping[str, object], plain_processor(bound_logger, "info", {}))
  assert hour_event["timestamp"] == "+01:00:05.250"


def test_setup_logging_json_output_has_no_ansi_escape_codes() -> None:
  root_logger = logging.getLogger()
  original_handlers = list(root_logger.handlers)
  original_level = root_logger.level
  root_logger.handlers.clear()

  try:
    setup_logging(level="DEBUG", json_output=True)

    handler = root_logger.handlers[-1]
    formatter = handler.formatter
    assert formatter is not None

    record = logging.getLogger("al/app").makeRecord(
      "al/app",
      logging.DEBUG,
      __file__,
      0,
      "json log %s",
      ("ok",),
      None,
    )

    rendered = formatter.format(record)
    assert "\x1b" not in rendered
    decoded = cast(dict[str, object], json.loads(rendered))
    assert decoded["event"] == "json log ok"
    assert decoded["level"] == "debug"
    assert decoded["logger"] == "al/app"
    assert str(decoded["timestamp"]).startswith("+")
  finally:
    root_logger.handlers.clear()
    root_logger.handlers.extend(original_handlers)
    root_logger.setLevel(original_level)


def test_logger_filter_processor_matches_slash_and_dot_hierarchies() -> None:
  processor = LoggerFilterProcessor("al")

  assert processor(_NamedLogger("al"), "info", {"event": "ok"}) == {"event": "ok"}
  assert processor(_NamedLogger("al/app"), "info", {"event": "ok"}) == {"event": "ok"}
  assert processor(_NamedLogger("al.app"), "info", {"event": "ok"}) == {"event": "ok"}

  with pytest.raises(structlog.DropEvent):
    _ = processor(_NamedLogger("other"), "info", {"event": "drop"})

  with pytest.raises(structlog.DropEvent):
    _ = processor(_NamedLogger("alpha"), "info", {"event": "drop"})
