"""Model-independent contracts for Parakeet's native timestamp adapter."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from threading import Event, Lock
from typing import Literal, override

import numpy as np
import pytest

from eavesdrop.server.config import TranscriptionConfig
from eavesdrop.server.transcription.models import SpeechChunk, VadParameters
from eavesdrop.server.transcription.parakeet import (
  SAMPLE_RATE,
  CudaRuntime,
  FloatAudio,
  Hypothesis,
  NativeTimestamps,
  ParakeetTranscriber,
  native_segments,
  resolve_device,
)
from eavesdrop.server.transcription.session import create_session
from eavesdrop.wire import Segment


@dataclass
class NativeHypothesis:
  """Provide a typed native result without importing NeMo.

  :param text: Exact recognized text.
  :param timestamp: Native word and segment timestamps in seconds.
  """

  text: str
  timestamp: NativeTimestamps


@dataclass
class InferenceCall:
  """Record the inputs and decoding options passed to the fake model."""

  audio: list[FloatAudio]
  batch_size: int
  return_hypotheses: bool
  timestamps: bool
  num_workers: int
  verbose: bool


@dataclass
class FakeNemoModel:
  """Return prepared native results and record inference calls.

  :param results: One result batch for each inference invocation.
  """

  results: list[list[Hypothesis]]
  calls: list[InferenceCall] = field(default_factory=list)

  def eval(self) -> object:
    """Keep the fake model unchanged and return it for API compatibility."""
    return self

  def transcribe(
    self,
    audio: list[FloatAudio],
    *,
    batch_size: int,
    return_hypotheses: bool,
    timestamps: bool,
    num_workers: int,
    verbose: bool,
  ) -> list[Hypothesis]:
    """Record a decoding call and return its prepared native hypotheses.

    :param audio: Input waveforms.
    :param batch_size: Requested batch size.
    :param return_hypotheses: Whether full hypotheses are requested.
    :param timestamps: Whether native timestamps are requested.
    :param num_workers: Requested data-loader workers.
    :param verbose: Whether model progress output is enabled.
    :returns: Prepared hypothesis batch for this call.
    """
    self.calls.append(
      InferenceCall(audio, batch_size, return_hypotheses, timestamps, num_workers, verbose)
    )
    return self.results[len(self.calls) - 1]


@dataclass
class FakeSpeechDetector:
  """Return sample-relative speech spans and record detector inputs.

  :param speech: Original-input speech intervals in samples.
  """

  speech: list[SpeechChunk]
  audio_inputs: list[FloatAudio] = field(default_factory=list)
  parameters: list[VadParameters] = field(default_factory=list)

  def detect(self, audio: FloatAudio, parameters: VadParameters) -> list[SpeechChunk]:
    """Record a VAD call and return the supplied speech intervals.

    :param audio: Original input waveform.
    :param parameters: Session VAD configuration.
    :returns: Speech intervals without collapsing silence gaps.
    """
    self.audio_inputs.append(audio)
    self.parameters.append(parameters)
    return self.speech.copy()


@dataclass
class FakeCudaRuntime:
  """Provide deterministic accelerator availability without importing Torch.

  :param available: Whether an accelerator runtime is available.
  :param count: Number of available accelerators.
  """

  available: bool
  count: int

  def is_available(self) -> bool:
    """Return the configured accelerator availability."""
    return self.available

  def device_count(self) -> int:
    """Return the configured accelerator count."""
    return self.count


@dataclass
class FakeTorchRuntime:
  """Supply device discovery and a lightweight inference context.

  :param cuda: Fake accelerator runtime.
  """

  cuda: CudaRuntime
  inference_calls: int = 0

  def inference_mode(self) -> AbstractContextManager[object]:
    """Record a requested inference context and return a no-op manager."""
    self.inference_calls += 1
    return nullcontext()


@dataclass
class BlockingNemoModel(FakeNemoModel):
  """Hold the first inference open to observe shared-adapter serialization."""

  first_entered: Event = field(default_factory=Event)
  second_entered: Event = field(default_factory=Event)
  release_first: Event = field(default_factory=Event)
  entry_lock: Lock = field(default_factory=Lock)
  entries: int = 0

  @override
  def transcribe(
    self,
    audio: list[FloatAudio],
    *,
    batch_size: int,
    return_hypotheses: bool,
    timestamps: bool,
    num_workers: int,
    verbose: bool,
  ) -> list[Hypothesis]:
    """Block the first model call and expose any overlapping second call.

    :param audio: Input waveforms.
    :param batch_size: Requested batch size.
    :param return_hypotheses: Whether full hypotheses are requested.
    :param timestamps: Whether native timestamps are requested.
    :param num_workers: Requested data-loader workers.
    :param verbose: Whether model progress output is enabled.
    :returns: The prepared result after the first call is released.
    :raises TimeoutError: If the test fails to release the first call.
    """
    with self.entry_lock:
      self.entries += 1
      entry = self.entries
    if entry == 1:
      self.first_entered.set()
      if not self.release_first.wait(timeout=5.0):
        raise TimeoutError("The test did not release the first inference")
    else:
      self.second_entered.set()
    return super().transcribe(
      audio,
      batch_size=batch_size,
      return_hypotheses=return_hypotheses,
      timestamps=timestamps,
      num_workers=num_workers,
      verbose=verbose,
    )


def _hypothesis(text: str = "Hello.") -> NativeHypothesis:
  return NativeHypothesis(
    text=text,
    timestamp={
      "word": [{"word": text, "start": 0.2, "end": 0.7}],
      "segment": [{"segment": text, "start": 0.1, "end": 0.8}],
    },
  )


def _audio(seconds: float = 1.0) -> FloatAudio:
  return np.zeros(round(seconds * SAMPLE_RATE), dtype=np.float32)


def test_native_words_preserve_punctuation_whitespace_and_segment_bounds() -> None:
  """Every native word, segment bound, and source character survives conversion."""
  hypothesis = NativeHypothesis(
    text="  Hello,\tworld!\n Again.  ",
    timestamp={
      "word": [
        {"word": "Hello,", "start": 0.2, "end": 0.5},
        {"word": "world!", "start": 0.6, "end": 0.95},
        {"word": "Again.", "start": 1.3, "end": 1.65},
      ],
      "segment": [
        {"segment": "Hello,\tworld!", "start": 0.12, "end": 1.04},
        {"segment": "Again.", "start": 1.2, "end": 1.8},
      ],
    },
  )

  segments = native_segments(hypothesis, duration=2.0)

  assert [segment.text for segment in segments] == ["  Hello,\tworld!", "\n Again.  "]
  assert "".join(segment.text for segment in segments) == hypothesis.text
  assert [(segment.start, segment.end) for segment in segments] == [(0.12, 1.04), (1.2, 1.8)]
  words = [word for segment in segments for word in segment.words or []]
  assert [word.word for word in words] == ["  Hello,", "\tworld!", "\n Again.  "]
  assert [(word.start, word.end) for word in words] == [(0.2, 0.5), (0.6, 0.95), (1.3, 1.65)]


def test_word_only_native_output_keeps_native_word_boundaries() -> None:
  """Absent sentence metadata falls back to native words without estimated times."""
  hypothesis = _hypothesis()
  hypothesis.timestamp["segment"] = []

  segments = native_segments(hypothesis, duration=1.0)

  assert len(segments) == 1
  assert (segments[0].start, segments[0].end, segments[0].text) == (0.2, 0.7, "Hello.")


def test_zero_duration_boundary_word_belongs_to_its_native_sentence() -> None:
  """A tied TDT word timestamp cannot move a word into the previous sentence."""
  hypothesis = NativeHypothesis(
    text="Hello. I agree.",
    timestamp={
      "word": [
        {"word": "Hello.", "start": 0.0, "end": 0.4},
        {"word": "I", "start": 0.4, "end": 0.4},
        {"word": "agree.", "start": 0.4, "end": 0.8},
      ],
      "segment": [
        {"segment": "Hello.", "start": 0.0, "end": 0.4},
        {"segment": "I agree.", "start": 0.4, "end": 0.8},
      ],
    },
  )

  segments = native_segments(hypothesis, duration=1.0)

  assert [segment.text for segment in segments] == ["Hello.", " I agree."]
  assert [(segment.start, segment.end) for segment in segments] == [(0.0, 0.4), (0.4, 0.8)]
  first_words, second_words = segments[0].words, segments[1].words
  assert first_words is not None
  assert second_words is not None
  assert [word.word for word in first_words] == ["Hello."]
  assert [word.word for word in second_words] == [" I", " agree."]
  assert [(word.start, word.end) for word in second_words] == [(0.4, 0.4), (0.4, 0.8)]


@pytest.mark.parametrize("native_end", [1.28, 1.32])
@pytest.mark.parametrize("include_segments", [False, True])
def test_final_tdt_token_overhang_is_clipped_to_original_audio(
  native_end: float, include_segments: bool
) -> None:
  """Valid TDT duration overhang is bounded to audio without shifting starts.

  :param native_end: Native final-token end within the 0.32-second TDT allowance.
  :param include_segments: Whether native sentence metadata is available.
  """
  hypothesis = NativeHypothesis(
    text="Hello.",
    timestamp={
      "word": [{"word": "Hello.", "start": 0.96, "end": native_end}],
      "segment": (
        [{"segment": "Hello.", "start": 0.88, "end": native_end}] if include_segments else []
      ),
    },
  )
  model = FakeNemoModel([[hypothesis]])

  segments, info = ParakeetTranscriber(model, detector=None).transcribe(
    _audio(),
    config=TranscriptionConfig(use_vad=False),
    absolute_stream_start=40.0,
    session=create_session("tdt-final-token-contract"),
  )

  assert len(segments) == 1
  segment = segments[0]
  assert segment.start == pytest.approx(40.88 if include_segments else 40.96)
  assert segment.end == 41.0
  assert segment.text == "Hello."
  assert segment.words is not None
  assert [(word.start, word.end) for word in segment.words] == [(40.96, 41.0)]
  assert all(40.0 <= word.start <= word.end <= 40.0 + info.duration for word in segment.words)
  assert 40.0 <= segment.start <= segment.end <= 40.0 + info.duration
  assert hypothesis.timestamp["word"][0]["end"] == native_end


def test_recording_offset_is_applied_once_beyond_thirty_seconds() -> None:
  """Repeated decoding of a native result never compounds recording offsets."""
  hypothesis = _hypothesis()
  model = FakeNemoModel([[hypothesis], [hypothesis]])
  transcriber = ParakeetTranscriber(model, detector=None)
  session = create_session("offset-contract")

  for offset in (31.25, 61.5):
    segments, info = transcriber.transcribe(
      _audio(),
      config=TranscriptionConfig(use_vad=False),
      absolute_stream_start=offset,
      session=session,
    )

    assert len(segments) == 1
    segment = segments[0]
    assert segment.start == pytest.approx(offset + 0.1)
    assert segment.end == pytest.approx(offset + 0.8)
    assert segment.time_offset == offset
    assert segment.words is not None
    assert segment.words[0].start == pytest.approx(offset + 0.2)
    assert segment.words[0].end == pytest.approx(offset + 0.7)
    assert session.get_recording_time_range() == (offset, offset + 1.0)
    assert info.duration == 1.0
  assert hypothesis.timestamp["word"][0]["start"] == 0.2
  assert hypothesis.timestamp["segment"][0]["start"] == 0.1


def test_vad_spans_keep_original_gap_and_recording_timeline() -> None:
  """Separately decoded speech retains silence and each span's original offset."""
  speech: list[SpeechChunk] = [
    {"start": SAMPLE_RATE, "end": 2 * SAMPLE_RATE},
    {"start": 5 * SAMPLE_RATE, "end": 6 * SAMPLE_RATE},
  ]
  detector = FakeSpeechDetector(speech)
  model = FakeNemoModel([[_hypothesis("First.")], [_hypothesis("Second.")]])
  transcriber = ParakeetTranscriber(model, detector)
  audio = np.arange(7 * SAMPLE_RATE, dtype=np.float32)
  config = TranscriptionConfig(use_vad=True, silence_completion_threshold=0.65)

  segments, info = transcriber.transcribe(
    audio,
    config=config,
    absolute_stream_start=40.0,
    session=create_session("vad-gap-contract"),
  )

  assert [segment.text for segment in segments] == ["First.", "Second."]
  assert [segment.start for segment in segments] == pytest.approx([41.1, 45.1])
  assert [segment.end for segment in segments] == pytest.approx([41.8, 45.8])
  assert [segment.time_offset for segment in segments] == [41.0, 45.0]
  words = [word for segment in segments for word in segment.words or []]
  assert [word.start for word in words] == pytest.approx([41.2, 45.2])
  assert [word.end for word in words] == pytest.approx([41.7, 45.7])
  assert len(model.calls) == 2
  np.testing.assert_array_equal(model.calls[0].audio[0], audio[SAMPLE_RATE : 2 * SAMPLE_RATE])
  np.testing.assert_array_equal(model.calls[1].audio[0], audio[5 * SAMPLE_RATE : 6 * SAMPLE_RATE])
  assert len(detector.audio_inputs) == 1
  np.testing.assert_array_equal(detector.audio_inputs[0], audio)
  assert detector.parameters == [config.vad_parameters]
  assert detector.parameters[0].min_silence_duration_ms == 650
  assert info.duration == 7.0
  assert info.duration_after_vad == 2.0
  assert info.speech_chunks == speech
  assert info.language == "en"


@pytest.mark.parametrize("use_runtime", [False, True])
def test_short_chunk_is_right_padded_without_shifting_native_timestamps(
  use_runtime: bool,
) -> None:
  """Subsecond chunks keep their zero point and original duration after padding.

  :param use_runtime: Whether inference uses the optional Torch context.
  """
  audio = np.linspace(0.25, 0.75, num=6400, dtype=np.float32)
  hypothesis = NativeHypothesis(
    text="Hello.",
    timestamp={
      "word": [{"word": "Hello.", "start": 0.08, "end": 0.48}],
      "segment": [{"segment": "Hello.", "start": 0.0, "end": 0.48}],
    },
  )
  model = FakeNemoModel([[hypothesis]])
  runtime = FakeTorchRuntime(FakeCudaRuntime(False, 0)) if use_runtime else None
  session = create_session("short-chunk-contract")

  segments, info = ParakeetTranscriber(model, detector=None, torch=runtime).transcribe(
    audio,
    config=TranscriptionConfig(use_vad=False),
    absolute_stream_start=31.2,
    session=session,
  )

  assert len(model.calls) == 1
  waveform = model.calls[0].audio[0]
  assert len(waveform) == SAMPLE_RATE
  np.testing.assert_array_equal(waveform[:6400], audio)
  np.testing.assert_array_equal(waveform[6400:], np.zeros(9600, dtype=np.float32))
  assert len(audio) == 6400
  assert len(segments) == 1
  segment = segments[0]
  assert (segment.start, segment.end) == pytest.approx((31.2, 31.6))
  assert segment.words is not None
  assert len(segment.words) == 1
  assert (segment.words[0].start, segment.words[0].end) == pytest.approx((31.28, 31.6))
  assert segment.time_offset == 31.2
  assert info.duration == 0.4
  assert info.duration_after_vad == 0.4
  assert session.get_recording_time_range() == pytest.approx((31.2, 31.6))


def test_short_vad_span_is_right_padded_on_its_original_recording_timeline() -> None:
  """Short VAD spans retain original samples, silence offsets, and unclipped duration."""
  audio = np.linspace(0.25, 0.75, num=2 * SAMPLE_RATE, dtype=np.float32)
  speech: list[SpeechChunk] = [{"start": 12000, "end": 18400}]
  detector = FakeSpeechDetector(speech)
  hypothesis = NativeHypothesis(
    text="Hello.",
    timestamp={
      "word": [{"word": "Hello.", "start": 0.08, "end": 0.48}],
      "segment": [{"segment": "Hello.", "start": 0.0, "end": 0.48}],
    },
  )
  model = FakeNemoModel([[hypothesis]])
  session = create_session("short-vad-span-contract")

  segments, info = ParakeetTranscriber(model, detector).transcribe(
    audio,
    config=TranscriptionConfig(use_vad=True),
    absolute_stream_start=50.0,
    session=session,
  )

  assert len(model.calls) == 1
  waveform = model.calls[0].audio[0]
  assert len(waveform) == SAMPLE_RATE
  np.testing.assert_array_equal(waveform[:6400], audio[12000:18400])
  np.testing.assert_array_equal(waveform[6400:], np.zeros(9600, dtype=np.float32))
  assert len(detector.audio_inputs) == 1
  np.testing.assert_array_equal(detector.audio_inputs[0], audio)
  assert len(segments) == 1
  segment = segments[0]
  assert (segment.start, segment.end) == pytest.approx((50.75, 51.15))
  assert segment.words is not None
  assert len(segment.words) == 1
  assert (segment.words[0].start, segment.words[0].end) == pytest.approx((50.83, 51.15))
  assert segment.time_offset == 50.75
  assert info.duration == 2.0
  assert info.duration_after_vad == 0.4
  assert info.speech_chunks == speech
  assert session.get_recording_time_range() == (50.0, 52.0)


def test_no_speech_skips_model_inference() -> None:
  """Silent VAD results produce empty output without calling the recognizer."""
  detector = FakeSpeechDetector([])
  model = FakeNemoModel([])

  segments, info = ParakeetTranscriber(model, detector).transcribe(
    _audio(2.0),
    config=TranscriptionConfig(use_vad=True),
    absolute_stream_start=32.0,
    session=create_session("no-speech-contract"),
  )

  assert segments == []
  assert model.calls == []
  assert len(detector.audio_inputs) == 1
  assert info.duration == 2.0
  assert info.duration_after_vad == 0.0
  assert info.speech_chunks == []


def test_disabled_vad_decodes_the_full_waveform_once_with_native_timestamps() -> None:
  """Disabling VAD bypasses the detector and requests native timestamps once."""
  detector = FakeSpeechDetector([])
  model = FakeNemoModel([[_hypothesis()]])
  runtime = FakeTorchRuntime(FakeCudaRuntime(False, 0))
  audio = np.arange(2 * SAMPLE_RATE, dtype=np.float32)[::2]

  segments, info = ParakeetTranscriber(model, detector, runtime).transcribe(
    audio,
    config=TranscriptionConfig(use_vad=False),
    absolute_stream_start=0.0,
    session=create_session("no-vad-contract"),
  )

  assert [segment.text for segment in segments] == ["Hello."]
  assert detector.audio_inputs == []
  assert len(model.calls) == 1
  call = model.calls[0]
  assert len(call.audio) == 1
  np.testing.assert_array_equal(call.audio[0], audio)
  assert call.audio[0].flags.c_contiguous
  assert call.batch_size == 1
  assert call.return_hypotheses is True
  assert call.timestamps is True
  assert call.num_workers == 0
  assert call.verbose is False
  assert runtime.inference_calls == 1
  assert info.duration == 1.0
  assert info.duration_after_vad == 1.0
  assert info.speech_chunks == []


def test_hiding_words_preserves_native_segment_text_and_bounds() -> None:
  """Word visibility does not change the native segment's recording timestamps."""
  hypothesis = _hypothesis()
  model = FakeNemoModel([[hypothesis], [hypothesis]])
  transcriber = ParakeetTranscriber(model, detector=None)
  results: list[list[Segment]] = []

  for include_words in (True, False):
    segments, _ = transcriber.transcribe(
      _audio(),
      config=TranscriptionConfig(use_vad=False, word_timestamps=include_words),
      absolute_stream_start=50.0,
      session=create_session(f"word-visibility-{include_words}"),
    )
    results.append(segments)

  visible, hidden = results[0][0], results[1][0]
  assert visible.words is not None
  assert hidden.words is None
  assert (hidden.text, hidden.start, hidden.end) == (visible.text, visible.start, visible.end)
  assert (hidden.start, hidden.end) == pytest.approx((50.1, 50.8))
  assert all(call.timestamps for call in model.calls)


@pytest.mark.parametrize("use_vad", [False, True])
def test_empty_audio_returns_empty_output_without_inference(use_vad: bool) -> None:
  """An empty input cannot produce model work or positive audio duration.

  :param use_vad: Whether the session requests VAD.
  """
  model = FakeNemoModel([])
  detector = FakeSpeechDetector([])
  session = create_session("empty-audio-contract")

  segments, info = ParakeetTranscriber(model, detector).transcribe(
    _audio(0.0),
    config=TranscriptionConfig(use_vad=use_vad),
    absolute_stream_start=35.0,
    session=session,
  )

  assert segments == []
  assert model.calls == []
  assert info.duration == 0.0
  assert info.duration_after_vad == 0.0
  assert info.speech_chunks == []
  assert session.get_recording_time_range() == (35.0, 35.0)


@pytest.mark.parametrize(
  "audio",
  [
    np.array([0.0, np.nan], dtype=np.float32),
    np.array([np.inf], dtype=np.float32),
    np.array([-np.inf], dtype=np.float32),
    np.zeros((2, SAMPLE_RATE), dtype=np.float32),
    np.array(0.0, dtype=np.float32),
  ],
  ids=["nan", "positive-infinity", "negative-infinity", "stereo", "scalar"],
)
def test_invalid_audio_fails_before_detector_or_model(audio: FloatAudio) -> None:
  """Non-finite or non-mono audio is rejected before any inference work.

  :param audio: An invalid input waveform.
  """
  model = FakeNemoModel([])
  detector = FakeSpeechDetector([])

  with pytest.raises(ValueError, match="finite mono audio"):
    ParakeetTranscriber(model, detector).transcribe(
      audio,
      config=TranscriptionConfig(use_vad=True),
      absolute_stream_start=0.0,
      session=create_session("invalid-audio-contract"),
    )

  assert model.calls == []
  assert detector.audio_inputs == []


def test_nonempty_text_requires_native_words() -> None:
  """Text without native word boundaries fails instead of inventing alignment."""
  hypothesis = _hypothesis()
  hypothesis.timestamp["word"] = []

  with pytest.raises(ValueError, match="word timestamps"):
    native_segments(hypothesis, duration=1.0)


def test_empty_native_hypothesis_produces_no_segments() -> None:
  """A model result containing only whitespace does not invent transcript text."""
  hypothesis = NativeHypothesis(text=" \t\n", timestamp={"word": [], "segment": []})

  assert native_segments(hypothesis, duration=1.0) == []


def test_empty_native_word_is_rejected() -> None:
  """An empty timestamp token cannot stand in for recognized text."""
  hypothesis = _hypothesis()
  hypothesis.timestamp["word"][0]["word"] = ""

  with pytest.raises(ValueError, match="text"):
    native_segments(hypothesis, duration=1.0)


@pytest.mark.parametrize("text", ["Wrong.", "Unexpected Hello.", "Hello. unexpected"])
def test_native_words_must_account_for_all_recognized_text(text: str) -> None:
  """Unmatched text before, after, or instead of native words is rejected.

  :param text: Recognized text inconsistent with the native word list.
  """
  hypothesis = _hypothesis()
  hypothesis.text = text

  with pytest.raises(ValueError, match="text"):
    native_segments(hypothesis, duration=1.0)


@pytest.mark.parametrize(
  ("start", "end"),
  [
    (float("nan"), 0.7),
    (0.2, float("inf")),
    (-0.1, 0.7),
    (0.7, 0.2),
    (0.2, 1.33),
    (1.01, 1.28),
  ],
  ids=["nan", "infinite", "negative", "reversed", "excessive-overhang", "start-past-audio"],
)
def test_invalid_native_word_bounds_are_rejected(start: float, end: float) -> None:
  """Malformed native timing fails instead of producing an invented boundary.

  :param start: Invalid native word start.
  :param end: Invalid native word end.
  """
  hypothesis = _hypothesis()
  hypothesis.timestamp["word"][0]["start"] = start
  hypothesis.timestamp["word"][0]["end"] = end

  with pytest.raises(ValueError, match="timestamp"):
    native_segments(hypothesis, duration=1.0)


def test_out_of_order_native_words_are_rejected() -> None:
  """Native words must follow their textual order on the audio timeline."""
  hypothesis = NativeHypothesis(
    text="First second",
    timestamp={
      "word": [
        {"word": "First", "start": 0.6, "end": 0.8},
        {"word": "second", "start": 0.1, "end": 0.3},
      ],
      "segment": [],
    },
  )

  with pytest.raises(ValueError, match="timestamp"):
    native_segments(hypothesis, duration=1.0)


def test_native_word_ends_cannot_move_backward() -> None:
  """Increasing word starts cannot conceal a reversed end-timestamp sequence."""
  hypothesis = NativeHypothesis(
    text="First second",
    timestamp={
      "word": [
        {"word": "First", "start": 0.1, "end": 0.8},
        {"word": "second", "start": 0.3, "end": 0.6},
      ],
      "segment": [],
    },
  )

  with pytest.raises(ValueError, match="timestamp"):
    native_segments(hypothesis, duration=1.0)


@pytest.mark.parametrize(
  ("start", "end"),
  [
    (float("nan"), 0.8),
    (0.1, float("inf")),
    (-0.1, 0.8),
    (0.8, 0.1),
    (0.1, 1.33),
    (1.01, 1.28),
  ],
  ids=["nan", "infinite", "negative", "reversed", "excessive-overhang", "start-past-audio"],
)
def test_invalid_native_segment_bounds_are_rejected(start: float, end: float) -> None:
  """Native sentence bounds are validated independently of valid word bounds.

  :param start: Invalid native segment start.
  :param end: Invalid native segment end.
  """
  hypothesis = _hypothesis()
  hypothesis.timestamp["segment"][0]["start"] = start
  hypothesis.timestamp["segment"][0]["end"] = end

  with pytest.raises(ValueError, match="timestamp"):
    native_segments(hypothesis, duration=1.0)


def test_out_of_order_native_segments_are_rejected() -> None:
  """Native segments cannot move backward even after all words are consumed."""
  hypothesis = _hypothesis()
  hypothesis.timestamp["segment"].append({"segment": "Hello.", "start": 0.0, "end": 0.9})

  with pytest.raises(ValueError, match="timestamp"):
    native_segments(hypothesis, duration=1.0)


def test_native_segment_text_must_match_its_words() -> None:
  """Inconsistent native sentence text cannot silently receive unrelated words."""
  hypothesis = _hypothesis()
  hypothesis.timestamp["segment"][0]["segment"] = "Different."

  with pytest.raises(ValueError, match="text"):
    native_segments(hypothesis, duration=1.0)


@pytest.mark.parametrize(("start", "end"), [(0.3, 0.8), (0.1, 0.6)])
def test_words_outside_native_segment_bounds_are_rejected(start: float, end: float) -> None:
  """Present sentence metadata must enclose the words assigned to that sentence.

  :param start: Native segment start, possibly after the word's start.
  :param end: Native segment end, possibly before the word's end.
  """
  hypothesis = _hypothesis()
  hypothesis.timestamp["segment"][0]["start"] = start
  hypothesis.timestamp["segment"][0]["end"] = end

  with pytest.raises(ValueError):
    native_segments(hypothesis, duration=1.0)


def test_requested_vad_requires_a_speech_detector() -> None:
  """A requested detector cannot be silently bypassed during transcription."""
  model = FakeNemoModel([])

  with pytest.raises(RuntimeError, match="speech detector"):
    ParakeetTranscriber(model, detector=None).transcribe(
      _audio(),
      config=TranscriptionConfig(use_vad=True),
      absolute_stream_start=0.0,
      session=create_session("missing-detector-contract"),
    )

  assert model.calls == []


@pytest.mark.parametrize("count", [0, 2])
def test_unexpected_hypothesis_cardinality_is_rejected(count: int) -> None:
  """Each submitted waveform must produce exactly one native hypothesis.

  :param count: Invalid number of results for a single input waveform.
  """
  model = FakeNemoModel([[_hypothesis() for _ in range(count)]])

  with pytest.raises(ValueError, match="one hypothesis"):
    ParakeetTranscriber(model, detector=None).transcribe(
      _audio(),
      config=TranscriptionConfig(use_vad=False),
      absolute_stream_start=0.0,
      session=create_session("cardinality-contract"),
    )

  assert len(model.calls) == 1


@pytest.mark.parametrize(
  ("device", "available", "count", "index", "expected"),
  [
    ("cpu", False, 0, 0, "cpu"),
    ("cpu", True, 2, 1, "cpu"),
    ("auto", False, 0, 0, "cpu"),
    ("auto", True, 2, 1, "cuda:1"),
    ("cuda", True, 1, 0, "cuda:0"),
  ],
)
def test_device_resolution_respects_cpu_and_automatic_selection(
  device: Literal["auto", "cpu", "cuda"], available: bool, count: int, index: int, expected: str
) -> None:
  """Device selection behaves deterministically without a real accelerator.

  :param device: Requested device mode.
  :param available: Runtime accelerator availability.
  :param count: Runtime accelerator count.
  :param index: Requested accelerator index.
  :param expected: Expected resolved Torch device string.
  """
  runtime = FakeTorchRuntime(FakeCudaRuntime(available, count))

  assert resolve_device(runtime, device, index) == expected


@pytest.mark.parametrize(
  ("device", "available", "count", "index"),
  [("cuda", False, 0, 0), ("cuda", True, 1, 1), ("auto", True, 1, 1)],
)
def test_unavailable_requested_gpu_fails_without_cpu_fallback(
  device: Literal["auto", "cpu", "cuda"], available: bool, count: int, index: int
) -> None:
  """Explicit GPU requests and invalid selected GPU indexes fail fast.

  :param device: Requested accelerator mode.
  :param available: Runtime accelerator availability.
  :param count: Runtime accelerator count.
  :param index: Unavailable accelerator index.
  """
  runtime = FakeTorchRuntime(FakeCudaRuntime(available, count))

  with pytest.raises(RuntimeError, match=f"cuda:{index} is unavailable"):
    resolve_device(runtime, device, index)


def test_shared_adapter_serializes_concurrent_inference() -> None:
  """Two sessions cannot enter a shared mutable NeMo decoder concurrently."""
  model = BlockingNemoModel([[_hypothesis()], [_hypothesis()]])
  transcriber = ParakeetTranscriber(model, detector=None)
  second_started = Event()

  def transcribe_at(offset: float) -> list[Segment]:
    if offset > 0.0:
      second_started.set()
    segments, _ = transcriber.transcribe(
      _audio(),
      config=TranscriptionConfig(use_vad=False),
      absolute_stream_start=offset,
      session=create_session(f"concurrent-{offset}"),
    )
    return segments

  with ThreadPoolExecutor(max_workers=2) as executor:
    first = executor.submit(transcribe_at, 0.0)
    try:
      assert model.first_entered.wait(timeout=2.0)
      second = executor.submit(transcribe_at, 40.0)
      assert second_started.wait(timeout=2.0)
      assert not model.second_entered.wait(timeout=0.1)
    finally:
      model.release_first.set()
    first_segments = first.result(timeout=2.0)
    second_segments = second.result(timeout=2.0)

  assert model.second_entered.is_set()
  assert len(model.calls) == 2
  assert first_segments[0].start == pytest.approx(0.1)
  assert second_segments[0].start == pytest.approx(40.1)
