"""Shared runtime ports for the active-listener service."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol

from active_listener.recording.reducer import RecordingReducerState
from eavesdrop.client import (
  ConnectedEvent,
  DisconnectedEvent,
  ReconnectedEvent,
  ReconnectingEvent,
  TranscriptionEvent,
)
from eavesdrop.wire import TranscriptionMessage


@dataclass(frozen=True)
class RewriteResult:
  """Structured rewrite output plus optional provider metadata.

  :param text: Final rewritten transcript text.
  :type text: str
  :param model: Model identifier used for the rewrite request.
  :type model: str
  :param input_tokens: Prompt token count reported by the provider.
  :type input_tokens: int | None
  :param output_tokens: Completion token count reported by the provider.
  :type output_tokens: int | None
  :param cost: Provider-reported request cost.
  :type cost: Decimal | None
  :param word_count: Word count of the emitted transcript text.
  :type word_count: int
  :param duration_seconds: Duration of the committed audio backing the transcript.
  :type duration_seconds: float | None
  """

  text: str
  model: str
  input_tokens: int | None
  output_tokens: int | None
  cost: Decimal | None


@dataclass(frozen=True)
class FinalizedTranscriptRecord:
  """Persistable transcript payload produced after successful emission.

  :param pre_finalization_text: Transcript text before the local finalization pipeline.
  :type pre_finalization_text: str
  :param post_finalization_text: Transcript text emitted to the target application.
  :type post_finalization_text: str
  :param llm_model: Model identifier used during rewrite, when present.
  :type llm_model: str | None
  :param tokens_in: Prompt token count reported by the rewrite provider.
  :type tokens_in: int | None
  :param tokens_out: Completion token count reported by the rewrite provider.
  :type tokens_out: int | None
  :param cost: Provider-reported request cost.
  :type cost: Decimal | None
  """

  pre_finalization_text: str
  post_finalization_text: str
  llm_model: str | None
  tokens_in: int | None
  tokens_out: int | None
  cost: Decimal | None
  word_count: int
  duration_seconds: float | None


@dataclass(frozen=True)
class CapturedRecordingAudio:
  """Immutable raw audio snapshot captured for one finished recording.

  :param pcm_f32le: Raw mono float32 PCM bytes captured during the recording.
  :type pcm_f32le: bytes
  :param sample_rate_hz: Capture sample rate in Hz.
  :type sample_rate_hz: int
  :param channels: Capture channel count.
  :type channels: int
  """

  pcm_f32le: bytes
  sample_rate_hz: int
  channels: int


@dataclass(frozen=True)
class FinishedRecording:
  """Immutable recording snapshot handed from session teardown to finalization.

  :param reducer_state: Final reducer state captured at recording finish.
  :type reducer_state: RecordingReducerState
  :param captured_audio: Immutable microphone capture snapshot for the recording.
  :type captured_audio: CapturedRecordingAudio
  """

  reducer_state: RecordingReducerState
  captured_audio: CapturedRecordingAudio


class ActiveListenerClient(Protocol):
  """Protocol for the live transcription client dependency."""

  async def connect(self) -> None:
    """Establish the live connection."""
    ...

  async def disconnect(self) -> None:
    """Close the live connection."""
    ...

  async def start_streaming(self) -> None:
    """Begin microphone capture and upstream streaming."""
    ...

  async def stop_streaming(self) -> None:
    """Stop microphone capture for the active recording."""
    ...

  async def cancel_utterance(self) -> None:
    """Discard the current live utterance without closing the session."""
    ...

  async def flush(self, *, force_complete: bool = True) -> TranscriptionMessage:
    """Request a committed transcription flush from the server."""
    ...

  def __aiter__(
    self,
  ) -> AsyncIterator[
    ConnectedEvent | DisconnectedEvent | ReconnectingEvent | ReconnectedEvent | TranscriptionEvent
  ]:
    """Iterate ordered client lifecycle events."""
    ...


class ActiveListenerRewriteClient(Protocol):
  async def rewrite_text(
    self,
    *,
    instructions: str,
    transcript: str,
  ) -> RewriteResult: ...

  async def close(self) -> None: ...


class ActiveListenerTranscriptHistoryStore(Protocol):
  """Persistence boundary for successfully emitted transcript records."""

  def record_finalized_recording(
    self,
    record: FinalizedTranscriptRecord,
    captured_audio: CapturedRecordingAudio,
  ) -> None: ...


class MediaPlaybackController(Protocol):
  async def pause_if_playing(self) -> bool: ...

  async def resume(self) -> None: ...


class ActiveListenerRuntimeError(RuntimeError):
  """Raised when the service cannot satisfy runtime prerequisites."""
