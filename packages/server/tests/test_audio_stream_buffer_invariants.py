"""Deterministic contract tests for :class:`AudioStreamBuffer` timeline invariants."""

from typing import Callable, cast

import numpy as np
from numpy.typing import NDArray

from eavesdrop.server.config import BufferConfig
from eavesdrop.server.streaming.buffer import AudioStreamBuffer

Float32Audio = NDArray[np.float32]


def _make_config(*, clip_audio: bool = False) -> BufferConfig:
  """Build a small integer-friendly config so time/sample math stays exact in tests."""
  return BufferConfig(
    sample_rate=10,
    max_buffer_duration=4.0,
    cleanup_duration=2.0,
    min_chunk_duration=1.5,
    transcription_interval=0.2,
    clip_audio=clip_audio,
    max_stall_duration=3.0,
  )


def _frames(seconds: float, *, sample_rate: int) -> NDArray[np.float32]:
  """Create deterministic audio frames with exact sample counts for boundary assertions."""
  sample_count = int(seconds * sample_rate)
  return np.linspace(-0.5, 0.5, num=sample_count, dtype=np.float32)


def _add_frames(buffer: AudioStreamBuffer, frames: Float32Audio) -> None:
  cast(Callable[[Float32Audio], None], buffer.add_frames)(frames)


def _get_chunk(buffer: AudioStreamBuffer) -> tuple[Float32Audio, float, float]:
  return buffer.get_chunk_for_processing()


def test_chunk_extraction_boundaries_around_min_chunk_duration() -> None:
  """Chunk extraction must reflect below-threshold and exact-threshold sample boundaries."""
  config = _make_config()

  below_threshold = AudioStreamBuffer(config)
  _add_frames(below_threshold, _frames(1.4, sample_rate=config.sample_rate))

  below_chunk, below_duration, below_start_time = _get_chunk(below_threshold)

  assert below_start_time == 0.0
  assert below_chunk.shape[0] == 14
  assert below_duration == 1.4
  assert below_duration < config.min_chunk_duration

  at_threshold = AudioStreamBuffer(config)
  _add_frames(at_threshold, _frames(config.min_chunk_duration, sample_rate=config.sample_rate))

  exact_chunk, exact_duration, exact_start_time = _get_chunk(at_threshold)

  assert exact_start_time == 0.0
  assert exact_chunk.shape[0] == 15
  assert exact_duration == config.min_chunk_duration


def test_processed_boundary_advances_monotonically_with_repeated_advancement() -> None:
  """Repeated positive advances must move processed time strictly forward."""
  config = _make_config()
  buffer = AudioStreamBuffer(config)
  _add_frames(buffer, _frames(4.0, sample_rate=config.sample_rate))

  previous = buffer.processed_up_to_time
  expected = 0.0
  for advance_by in (0.3, 0.4, 0.9, 0.2):
    buffer.advance_processed_boundary(advance_by)
    expected += advance_by

    assert buffer.processed_up_to_time > previous
    assert buffer.processed_up_to_time == expected

    _, _, chunk_start_time = _get_chunk(buffer)
    assert chunk_start_time == buffer.processed_up_to_time

    previous = buffer.processed_up_to_time


def test_cleanup_keeps_processed_boundary_safe_when_buffer_is_trimmed() -> None:
  """Cleanup must not leave processed time behind the trimmed buffer start."""
  config = _make_config()
  buffer = AudioStreamBuffer(config)

  _add_frames(buffer, _frames(4.5, sample_rate=config.sample_rate))
  buffer.advance_processed_boundary(1.0)

  # Cleanup runs at the beginning of add_frames when the existing buffer already exceeds max.
  _add_frames(buffer, _frames(0.1, sample_rate=config.sample_rate))

  assert buffer.buffer_start_time == config.cleanup_duration
  assert buffer.processed_up_to_time == buffer.buffer_start_time

  chunk, duration, start_time = _get_chunk(buffer)
  assert start_time == buffer.processed_up_to_time
  assert duration == chunk.shape[0] / config.sample_rate
  assert start_time <= buffer.buffer_start_time + buffer.total_duration


def test_discard_unprocessed_audio_preserves_processed_cursor_for_next_utterance() -> None:
  """Dropping a live utterance tail must preserve the processed cursor for future audio."""
  config = _make_config()
  buffer = AudioStreamBuffer(config)
  _add_frames(buffer, _frames(3.0, sample_rate=config.sample_rate))
  buffer.advance_processed_boundary(1.2)

  discarded_duration = buffer.discard_unprocessed_audio()

  assert discarded_duration == 1.8
  assert buffer.processed_up_to_time == 1.2
  assert buffer.buffer_start_time == 1.2
  assert buffer.total_duration == 0.0
  assert buffer.available_duration == 0.0

  _add_frames(buffer, _frames(0.5, sample_rate=config.sample_rate))
  chunk, duration, start_time = _get_chunk(buffer)

  assert chunk.shape[0] == 5
  assert duration == 0.5
  assert start_time == 1.2


def test_clip_if_stalled_respects_enabled_disabled_and_threshold_boundaries() -> None:
  """Clip behavior must be gated and trigger only above the stall threshold."""
  disabled_config = _make_config(clip_audio=False)
  disabled_buffer = AudioStreamBuffer(disabled_config)
  _add_frames(disabled_buffer, _frames(9.0, sample_rate=disabled_config.sample_rate))
  disabled_buffer.clip_if_stalled()
  assert disabled_buffer.processed_up_to_time == 0.0

  enabled_config = _make_config(clip_audio=True)

  exact_threshold = AudioStreamBuffer(enabled_config)
  _add_frames(exact_threshold, _frames(3.0, sample_rate=enabled_config.sample_rate))
  exact_threshold.clip_if_stalled()
  assert exact_threshold.processed_up_to_time == 0.0

  over_threshold = AudioStreamBuffer(enabled_config)
  _add_frames(over_threshold, _frames(9.0, sample_rate=enabled_config.sample_rate))
  over_threshold.clip_if_stalled()

  expected_processed_time = over_threshold.buffer_start_time + over_threshold.total_duration - 5.0
  buffer_end_time = over_threshold.buffer_start_time + over_threshold.total_duration

  assert over_threshold.processed_up_to_time == expected_processed_time
  assert over_threshold.processed_up_to_time <= buffer_end_time


def test_processed_position_beyond_buffer_end_never_exposes_invalid_chunk_domain() -> None:
  """Overshoot in processed time must still yield bounded chunk extraction."""
  config = _make_config()
  buffer = AudioStreamBuffer(config)
  _add_frames(buffer, _frames(2.0, sample_rate=config.sample_rate))

  buffer.advance_processed_boundary(5.0)

  chunk, duration, start_time = _get_chunk(buffer)

  assert start_time == buffer.processed_up_to_time
  assert chunk.size == 0
  assert duration == 0.0
  assert buffer.available_duration == 0.0
