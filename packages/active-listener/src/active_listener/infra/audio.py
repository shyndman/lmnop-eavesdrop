"""FFmpeg-backed audio encoding helpers for active-listener."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from active_listener.app.ports import ActiveListenerRuntimeError, CapturedRecordingAudio


class AudioEncodeError(RuntimeError):
  """Raised when FFmpeg cannot encode captured recording audio."""


def encode_m4a(
  ffmpeg_path: str,
  pcm_f32le: bytes,
  *,
  sample_rate_hz: int = 16_000,
  channels: int = 1,
) -> bytes:
  """Encode raw mono float32 PCM bytes into an M4A payload.

  :param ffmpeg_path: Resolved FFmpeg executable path.
  :type ffmpeg_path: str
  :param pcm_f32le: Raw mono float32 PCM bytes.
  :type pcm_f32le: bytes
  :param sample_rate_hz: Capture sample rate in Hz.
  :type sample_rate_hz: int
  :param channels: Capture channel count.
  :type channels: int
  :returns: Encoded M4A payload.
  :rtype: bytes
  :raises AudioEncodeError: If FFmpeg fails to encode the audio.
  """

  return _encode_audio(
    ffmpeg_path,
    pcm_f32le,
    output_filename="recording.m4a",
    output_format="ipod",
    codec_args=["-c:a", "aac", "-b:a", "128k", "-profile:a", "aac_low"],
    sample_rate_hz=sample_rate_hz,
    channels=channels,
  )


def resolve_ffmpeg_executable(configured_path: str | None) -> str:
  """Resolve a currently-usable ffmpeg path at call time.

  Cheap by design: no ``-version`` probe and no symlink flattening, so the
  result follows binary relocations across the service lifetime. Existence +
  executability is sufficient signal; a broken binary surfaces as an encode
  failure handled by the caller.

  :param configured_path: Optional explicit override (already absolute; the
      config loader normalizes relative values against the config dir).
  :type configured_path: str | None
  :returns: Path to an executable ffmpeg binary.
  :rtype: str
  :raises ActiveListenerRuntimeError: If no executable ffmpeg is found.
  """
  if configured_path is not None:
    candidate = Path(configured_path).expanduser()
    if candidate.is_file() and os.access(candidate, os.X_OK):
      return str(candidate)
  resolved = shutil.which("ffmpeg")
  if resolved is not None:
    return resolved
  raise ActiveListenerRuntimeError(
    "ffmpeg executable not found; set ffmpeg_path or add ffmpeg to PATH"
  )


def encode_recording_audio(
  captured_audio: CapturedRecordingAudio,
  *,
  ffmpeg_path: str | None,
) -> bytes | None:
  """Encode captured PCM to m4a once, resolving ffmpeg at call time.

  :param captured_audio: Raw mono float32 PCM snapshot for the recording.
  :type captured_audio: CapturedRecordingAudio
  :param ffmpeg_path: Configured override, or None to resolve from PATH.
  :type ffmpeg_path: str | None
  :returns: Encoded m4a bytes, or None when there is no audio to encode.
  :rtype: bytes | None
  :raises ActiveListenerRuntimeError: If ffmpeg cannot be resolved.
  :raises AudioEncodeError: If ffmpeg fails to encode.
  """
  if captured_audio.pcm_f32le == b"":
    return None
  ffmpeg = resolve_ffmpeg_executable(ffmpeg_path)
  return encode_m4a(
    ffmpeg,
    captured_audio.pcm_f32le,
    sample_rate_hz=captured_audio.sample_rate_hz,
    channels=captured_audio.channels,
  )


def _encode_audio(
  ffmpeg_path: str,
  pcm_f32le: bytes,
  *,
  output_filename: str,
  output_format: str,
  codec_args: list[str],
  sample_rate_hz: int,
  channels: int,
) -> bytes:
  with tempfile.TemporaryDirectory(prefix="active-listener-audio-") as temp_dir:
    output_path = Path(temp_dir) / output_filename
    result = subprocess.run(
      [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "f32le",
        "-ar",
        str(sample_rate_hz),
        "-ac",
        str(channels),
        "-i",
        "pipe:0",
        *codec_args,
        "-f",
        output_format,
        str(output_path),
      ],
      input=pcm_f32le,
      capture_output=True,
      check=False,
    )
    if result.returncode != 0:
      stderr_text = result.stderr.decode("utf-8", errors="replace").strip()
      raise AudioEncodeError(stderr_text or "ffmpeg audio encode failed")

    if not output_path.exists():
      raise AudioEncodeError("ffmpeg completed without creating an output file")

    encoded_audio = output_path.read_bytes()
    if encoded_audio == b"":
      raise AudioEncodeError(f"ffmpeg created an empty {output_format} file")

  return encoded_audio
