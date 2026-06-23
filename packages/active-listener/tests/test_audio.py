from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from active_listener.app.ports import ActiveListenerRuntimeError, CapturedRecordingAudio
from active_listener.infra.audio import encode_recording_audio, resolve_ffmpeg_executable


def test_resolve_ffmpeg_executable_prefers_configured_without_probe(tmp_path: Path) -> None:
  fake_ffmpeg = tmp_path / "ffmpeg"
  _ = fake_ffmpeg.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
  fake_ffmpeg.chmod(0o755)

  assert resolve_ffmpeg_executable(str(fake_ffmpeg)) == str(fake_ffmpeg)


def test_resolve_ffmpeg_executable_does_not_flatten_symlink(tmp_path: Path) -> None:
  real_ffmpeg = tmp_path / "real-ffmpeg"
  _ = real_ffmpeg.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
  real_ffmpeg.chmod(0o755)
  link_ffmpeg = tmp_path / "linked-ffmpeg"
  link_ffmpeg.symlink_to(real_ffmpeg)

  assert resolve_ffmpeg_executable(str(link_ffmpeg)) == str(link_ffmpeg)


def test_encode_recording_audio_returns_none_for_empty_pcm() -> None:
  captured_audio = CapturedRecordingAudio(pcm_f32le=b"", sample_rate_hz=16000, channels=1)

  assert encode_recording_audio(captured_audio, ffmpeg_path=None) is None


def test_resolve_ffmpeg_executable_raises_when_absent(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(shutil, "which", lambda _name: None)

  with pytest.raises(ActiveListenerRuntimeError):
    _ = resolve_ffmpeg_executable(None)
