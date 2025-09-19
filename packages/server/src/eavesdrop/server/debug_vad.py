#!/usr/bin/env python3
"""
VAD Debug Tool - Isolated testing of Voice Activity Detection

This script tests VAD in isolation to debug why it's not detecting speech
that Whisper can successfully transcribe. It takes a WAV file, processes it
through the same VAD pipeline used by the server, and provides detailed output.

Usage:
    eavesdrop-debug-vad input.wav
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from faster_whisper.vad import VadOptions, get_speech_timestamps

from eavesdrop.common import get_logger

logger = get_logger("vad-debug")


def load_and_prepare_audio(wav_path: Path) -> tuple[np.ndarray, int]:
  """Load WAV file and convert to format expected by VAD (16kHz mono float32)."""
  logger.info(f"Loading audio file: {wav_path}")

  # Load audio file
  audio, sample_rate = sf.read(wav_path, dtype="float32")

  logger.info(f"Original audio: {audio.shape} samples at {sample_rate}Hz")

  # Convert stereo to mono if needed
  if len(audio.shape) > 1:
    audio = np.mean(audio, axis=1)
    logger.info("Converted stereo to mono")

  # Resample to 16kHz if needed (simple approach)
  if sample_rate != 16000:
    logger.warning(f"Sample rate is {sample_rate}Hz, but VAD expects 16kHz")
    logger.warning("Consider resampling the file externally for accurate testing")

  logger.info(f"Final audio: {audio.shape} samples, {len(audio) / sample_rate:.2f}s duration")
  logger.info(f"Audio range: min={audio.min():.3f}, max={audio.max():.3f}")

  return audio, sample_rate


def create_vad_options(silence_completion_threshold: float = 0.8) -> VadOptions:
  """Create VAD options matching the server configuration."""
  vad_options = VadOptions()

  # Set the min_silence_duration_ms to match our server logic
  vad_options.min_silence_duration_ms = int(silence_completion_threshold * 1000)

  return vad_options


def analyze_with_vad(audio: np.ndarray, sample_rate: int, vad_options: VadOptions):
  """Run VAD analysis and print detailed results."""
  logger.info("Running VAD analysis...")
  logger.info("VAD Parameters:")
  logger.info(f"  onset: {vad_options.onset}")
  logger.info(f"  offset: {vad_options.offset}")
  logger.info(f"  min_speech_duration_ms: {vad_options.min_speech_duration_ms}")
  logger.info(f"  max_speech_duration_s: {vad_options.max_speech_duration_s}")
  logger.info(f"  min_silence_duration_ms: {vad_options.min_silence_duration_ms}")
  logger.info(f"  speech_pad_ms: {vad_options.speech_pad_ms}")

  # Run VAD
  speech_timestamps = get_speech_timestamps(audio, vad_options)

  logger.info("VAD Results:")
  logger.info(f"  Speech chunks detected: {len(speech_timestamps)}")

  if speech_timestamps:
    total_speech_samples = 0
    for i, chunk in enumerate(speech_timestamps):
      start_time = chunk["start"] / sample_rate
      end_time = chunk["end"] / sample_rate
      duration = end_time - start_time
      total_speech_samples += chunk["end"] - chunk["start"]

      logger.info(f"    Chunk {i + 1}: {start_time:.3f}s - {end_time:.3f}s ({duration:.3f}s)")
      logger.info(f"      Samples: {chunk['start']} - {chunk['end']}")

    total_speech_time = total_speech_samples / sample_rate
    total_audio_time = len(audio) / sample_rate
    speech_ratio = total_speech_time / total_audio_time

    logger.info(f"  Total speech time: {total_speech_time:.3f}s")
    logger.info(f"  Total audio time: {total_audio_time:.3f}s")
    logger.info(f"  Speech ratio: {speech_ratio:.1%}")

    # Create timeline visualization
    timeline_length = min(100, int(total_audio_time * 10))  # 100ms resolution, max 100 chars
    timeline = ["~"] * timeline_length

    for chunk in speech_timestamps:
      start_idx = int((chunk["start"] / sample_rate) * 10)
      end_idx = int((chunk["end"] / sample_rate) * 10)
      for i in range(start_idx, min(end_idx, len(timeline))):
        timeline[i] = "S"

    logger.info(f"  Timeline (S=speech, ~=silence): {''.join(timeline)}")

  else:
    logger.warning("  NO SPEECH DETECTED!")
    total_audio_time = len(audio) / sample_rate
    logger.info(f"  Total audio time: {total_audio_time:.3f}s")
    logger.info("  Speech ratio: 0.0%")
    timeline_length = min(100, int(total_audio_time * 10))
    logger.info(f"  Timeline: {'~' * timeline_length}")

  return speech_timestamps


def analyze_audio_characteristics(audio: np.ndarray, sample_rate: int):
  """Analyze audio characteristics that might affect VAD."""
  logger.info("Audio Analysis:")

  # RMS energy
  rms = np.sqrt(np.mean(audio**2))
  logger.info(f"  RMS Energy: {rms:.6f}")

  # Peak amplitude
  peak = np.max(np.abs(audio))
  logger.info(f"  Peak Amplitude: {peak:.6f}")

  # Dynamic range
  if peak > 0:
    dynamic_range_db = 20 * np.log10(peak / (rms + 1e-10))
    logger.info(f"  Dynamic Range: {dynamic_range_db:.1f} dB")

  # Zero crossing rate (rough measure of speech characteristics)
  zero_crossings = np.sum(np.diff(np.signbit(audio)))
  zcr = zero_crossings / len(audio) * sample_rate
  logger.info(f"  Zero Crossing Rate: {zcr:.1f} Hz")

  # Detect potential issues
  if rms < 0.001:
    logger.warning("  ⚠️  Very low RMS energy - audio might be too quiet for VAD")
  if peak < 0.01:
    logger.warning("  ⚠️  Very low peak amplitude - audio might need amplification")
  if zcr < 50:
    logger.warning("  ⚠️  Very low zero crossing rate - might not be speech-like")


def test_different_vad_settings(audio: np.ndarray, sample_rate: int):
  """Test different VAD parameter combinations."""
  logger.info("Testing different VAD parameter combinations...")

  test_configs = [
    ("Default", VadOptions()),
    ("More Sensitive", VadOptions(onset=0.3, offset=0.25)),
    ("Less Sensitive", VadOptions(onset=0.7, offset=0.45)),
    ("Very Sensitive", VadOptions(onset=0.2, offset=0.15)),
    ("Short Silence", VadOptions(min_silence_duration_ms=100)),
    ("No Min Speech", VadOptions(min_speech_duration_ms=0)),
  ]

  for name, vad_options in test_configs:
    logger.info(f"\n--- Testing: {name} ---")
    logger.info(f"onset={vad_options.onset}, offset={vad_options.offset}")
    logger.info(f"min_speech_duration_ms={vad_options.min_speech_duration_ms}")
    logger.info(f"min_silence_duration_ms={vad_options.min_silence_duration_ms}")

    try:
      speech_chunks = get_speech_timestamps(audio, vad_options)
      if speech_chunks:
        logger.info(f"✅ Detected {len(speech_chunks)} speech chunk(s)")
        total_speech = sum((chunk["end"] - chunk["start"]) / sample_rate for chunk in speech_chunks)
        logger.info(f"   Total speech: {total_speech:.3f}s")
      else:
        logger.info("❌ No speech detected")
    except Exception as e:
      logger.error(f"Error with {name}: {e}")


def main():
  """Main entry point for VAD debug tool."""
  parser = argparse.ArgumentParser(
    description="Debug VAD (Voice Activity Detection) with audio files",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
    eavesdrop-debug-vad recording.wav
    eavesdrop-debug-vad --silence-threshold 0.5 speech_sample.wav
    eavesdrop-debug-vad --test-all-params noisy_audio.wav
        """,
  )
  parser.add_argument("wav_file", type=Path, help="Input WAV file to analyze")
  parser.add_argument(
    "--silence-threshold",
    type=float,
    default=0.8,
    help="Silence completion threshold in seconds (default: 0.8)",
  )
  parser.add_argument(
    "--test-all-params", action="store_true", help="Test multiple VAD parameter combinations"
  )

  args = parser.parse_args()

  # Validate input file
  if not args.wav_file.exists():
    logger.error(f"File not found: {args.wav_file}")
    sys.exit(1)

  if args.wav_file.suffix.lower() not in [".wav", ".wave"]:
    logger.error(f"File must be a WAV file, got: {args.wav_file.suffix}")
    sys.exit(1)

  logger.info("=" * 60)
  logger.info("EAVESDROP VAD DEBUG TOOL")
  logger.info("=" * 60)

  try:
    # Load and prepare audio
    audio, sample_rate = load_and_prepare_audio(args.wav_file)

    # Analyze audio characteristics
    analyze_audio_characteristics(audio, sample_rate)

    # Create VAD options matching server config
    vad_options = create_vad_options(args.silence_threshold)

    # Run primary VAD analysis
    logger.info("\n" + "=" * 40)
    logger.info("PRIMARY VAD ANALYSIS")
    logger.info("=" * 40)
    speech_chunks = analyze_with_vad(audio, sample_rate, vad_options)

    # Test different parameters if requested
    if args.test_all_params:
      logger.info("\n" + "=" * 40)
      logger.info("TESTING DIFFERENT VAD PARAMETERS")
      logger.info("=" * 40)
      test_different_vad_settings(audio, sample_rate)

    # Summary
    logger.info("\n" + "=" * 40)
    logger.info("SUMMARY")
    logger.info("=" * 40)
    if speech_chunks:
      logger.info(f"✅ VAD detected {len(speech_chunks)} speech chunk(s)")
      logger.info("   This should work properly with the transcription pipeline")
    else:
      logger.error("❌ VAD detected NO SPEECH")
      logger.error("   This explains why transcription is being run on silence!")
      logger.error("   Consider:")
      logger.error("   - Checking if audio is too quiet (low RMS/peak)")
      logger.error("   - Testing with --test-all-params to find better settings")
      logger.error("   - Amplifying the audio file")
      logger.error("   - Using different VAD parameters")

  except Exception as e:
    logger.exception(f"Error processing {args.wav_file}: {e}")
    sys.exit(1)


if __name__ == "__main__":
  main()
