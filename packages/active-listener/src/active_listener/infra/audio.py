"""FFmpeg-backed audio encoding helpers for active-listener."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


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


def encode_mp3(
  ffmpeg_path: str,
  pcm_f32le: bytes,
  *,
  sample_rate_hz: int = 16_000,
  channels: int = 1,
) -> bytes:
  """Encode raw mono float32 PCM bytes into an MP3 payload.

  :param ffmpeg_path: Resolved FFmpeg executable path.
  :type ffmpeg_path: str
  :param pcm_f32le: Raw mono float32 PCM bytes.
  :type pcm_f32le: bytes
  :param sample_rate_hz: Capture sample rate in Hz.
  :type sample_rate_hz: int
  :param channels: Capture channel count.
  :type channels: int
  :returns: Encoded MP3 payload.
  :rtype: bytes
  :raises AudioEncodeError: If FFmpeg fails to encode the audio.
  """

  return _encode_audio(
    ffmpeg_path,
    pcm_f32le,
    output_filename="recording.mp3",
    output_format="mp3",
    codec_args=["-c:a", "libmp3lame", "-b:a", "128k"],
    sample_rate_hz=sample_rate_hz,
    channels=channels,
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
