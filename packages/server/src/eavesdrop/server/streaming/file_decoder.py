"""Finite-file decoding helpers for canonical transcription ingest audio.

This module converts uploaded file bytes into the exact canonical audio format
used by the streaming pipeline: mono, 16kHz, float32 PCM.
"""

import asyncio

import numpy as np

CANONICAL_SAMPLE_RATE_HZ = 16_000
CANONICAL_CHANNELS = 1
CANONICAL_DTYPE = np.float32


class FileDecodeError(RuntimeError):
  """Raised when finite-file audio cannot be decoded into canonical format."""


async def decode_file_bytes_to_canonical_audio(file_bytes: bytes) -> np.ndarray:
  """Decode uploaded file bytes to canonical mono 16kHz float32 audio.

  :param file_bytes: Raw uploaded file bytes (WAV/MP3/AAC).
  :type file_bytes: bytes
  :returns: Canonical audio samples as float32 mono waveform.
  :rtype: np.ndarray
  :raises FileDecodeError: If ffmpeg is unavailable or decoding fails.
  """
  ffmpeg_command = [
    "ffmpeg",
    "-hide_banner",
    "-loglevel",
    "error",
    "-i",
    "pipe:0",
    "-f",
    "f32le",
    "-ac",
    str(CANONICAL_CHANNELS),
    "-ar",
    str(CANONICAL_SAMPLE_RATE_HZ),
    "pipe:1",
  ]

  try:
    process = await asyncio.create_subprocess_exec(
      *ffmpeg_command,
      stdin=asyncio.subprocess.PIPE,
      stdout=asyncio.subprocess.PIPE,
      stderr=asyncio.subprocess.PIPE,
    )
  except FileNotFoundError as exc:
    raise FileDecodeError("ffmpeg executable was not found in PATH") from exc

  decoded_stdout, decoded_stderr = await process.communicate(file_bytes)
  if process.returncode != 0:
    stderr_text = decoded_stderr.decode("utf-8", errors="replace").strip()
    raise FileDecodeError(
      f"ffmpeg decode failed with exit code {process.returncode}: {stderr_text}"
    )

  if not decoded_stdout:
    return np.array([], dtype=CANONICAL_DTYPE)

  decoded_audio = np.frombuffer(decoded_stdout, dtype=CANONICAL_DTYPE)
  return decoded_audio.copy()
