"""Regression tests for shared logging configuration."""

from __future__ import annotations

import logging

from eavesdrop.common.logs import setup_logging


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
