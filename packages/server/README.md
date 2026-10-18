# Eavesdrop server

English speech recognition using NVIDIA's [`nvidia/parakeet-tdt-0.6b-v2`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) checkpoint through NeMo. The server retains WebSocket and RTSP streaming, silence-based completion, and recording-relative word timestamps.

The model supplies native word start/end times. Silero VAD runs separately through its ONNX implementation to detect speech and silence. The model is downloaded on first use; first startup needs network access and sufficient disk space for the checkpoint and dependencies.

## Model and runtime choice

The official English-only v2 checkpoint is the baseline. Its [model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) documents native word and segment timing; no separate alignment model is needed. We did not find evidence sufficient to prefer a general-purpose English fine-tune over NVIDIA's checkpoint.

The popular [istupakov ONNX conversion](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v2) is an alternate runtime representation, not a demonstrated accuracy improvement. The evaluated [onnx-asr result API](https://github.com/istupakov/onnx-asr/blob/main/src/onnx_asr/asr.py) exposes token onsets rather than the native word start/end contract required by active-listener. NeMo 2.7.3 preserves that contract directly, at the cost of a substantially larger dependency footprint. PyArrow is constrained below 26 because its 26 wheel requires NumPy 2 while this workspace remains on NumPy 1.

The Whisper engine, CTranslate2, and separate forced-alignment stage are removed. NeMo's ASR extra still declares the transitive `whisper-normalizer` text utility; it is not a retained Whisper recognizer or alignment stage.

Streaming remains rolling offline chunk re-decoding, with the existing buffer, cancellation, flush, and silence-completion behavior. Native boundaries are shifted to recording time exactly once. Short chunks are right-padded to avoid NeMo's default centered padding; final token durations are clipped at the original audio boundary. Those are timeline conversions, not forced alignment or phrase filtering.

## Native CPU setup

Run these commands from the repository root with Python 3.12 and `uv` installed. FFmpeg is required for RTSP ingestion and file decoding; NeMo also uses libsndfile and SoX. On Debian/Ubuntu, install the system packages first:

```bash
sudo apt-get install build-essential ffmpeg libsndfile1 sox
uv sync --package eavesdrop-server --extra transcription
cp packages/server/config.sample.yaml packages/server/config.yaml
# Set transcription.device: cpu to force CPU inference.
uv run --package eavesdrop-server --extra transcription eavesdrop-server \
  --config packages/server/config.yaml
```

ONNX Runtime and Hugging Face usage telemetry are disabled before runtime imports. The Docker image also sets `ORT_DISABLE_TELEMETRY=1` before Python starts, following Microsoft’s [process-lifetime opt-out](https://github.com/microsoft/onnxruntime/blob/main/docs/Privacy.md#disabling-telemetry).

The heavy runtime is an opt-in `transcription` extra. Unit tests and normal workspace development can omit it. A native GPU installation needs a compatible, matching vendor PyTorch/torchaudio stack; the container paths below protect these wheels from replacement by the portable host dependency lock.

Allow several GB each for installed packages, the roughly 2.5 GB checkpoint, and temporary checkpoint extraction, plus enough RAM to construct the model and load its weights. If `/tmp` is small or RAM-backed, point `TMPDIR` at a disk-backed directory with free space before starting the server.

## Configuration and migration

```yaml
transcription:
  model: nvidia/parakeet-tdt-0.6b-v2
  device: auto
  device_index: 0
  language: en
  word_timestamps: true
  use_vad: true
  silence_completion_threshold: 0.8
```

- `model` is fixed to the NVIDIA v2 checkpoint; other models and local custom checkpoints are rejected.
- `device` accepts `auto`, `cpu`, or `cuda`. `auto` chooses an available GPU, otherwise CPU. PyTorch uses `cuda` for both NVIDIA CUDA and AMD ROCm devices.
- `device_index` selects the zero-based visible GPU index; the default is `0`.
- Input audio remains mono float32 at 16 kHz. Timing stays relative to the original recording, including when VAD removes silent regions.
- `word_timestamps: false` hides word details from clients; native timings are still used for segmentation and completion.
- English transcription is the only supported task. Translation, non-English language selection, prompts, hotwords, custom checkpoints, beam search, and the previous model-selection options are removed.

Start from `config.sample.yaml` when upgrading. Old options such as `custom_model`, `initial_prompt`, `hotwords`, `beam_size`, `num_workers`, `same_output_threshold`, and `gpu_name` are rejected. Use `device` and `device_index` instead of GPU-name matching. Clients must use the matching updated protocol and must not send removed task/model options.

## Docker builds

All Docker commands use the repository root as their build context. The default task target is now **CPU**; select GPU support explicitly:

```bash
# CPU (default)
task build-server-image

task build-server-image ACCELERATOR=cuda

# Experimental AMD path; not a claim of GPU inference validation.
task build-server-image ACCELERATOR=rocm
```

The three base images share `/opt/venv`:

- CPU: Python 3.12 and official PyTorch 2.9.1 CPU wheels
- CUDA: NVIDIA CUDA 12.8.1 / Ubuntu 24.04, Python 3.12, and official PyTorch 2.9.1 CUDA 12.8 wheels
- ROCm: AMD's `rocm/pytorch:rocm7.2.3_ubuntu24.04_py3.12_pytorch_release_2.9.1` image and its existing vendor stack

The final image installs the server's `transcription` extra using the workspace-root `uv.lock`. It explicitly skips the base's Torch, torchaudio, torchvision, Triton, and vendor GPU libraries, then checks that the Torch build is unchanged and NeMo/Silero import. Those build checks do not execute the model or prove GPU compatibility.

You can override the image names with `CPU_BASE_IMAGE`, `CUDA_BASE_IMAGE`, `ROCM_BASE_IMAGE`, and `SERVER_IMAGE`. `SKIP_CACHE=true` disables caching for the final image build. Equivalent manual CPU commands are:

```bash
docker build -f packages/server/docker/Dockerfile.cpu-base -t eavesdrop-cpu-base .
docker build -f packages/server/docker/Dockerfile \
  --build-arg BASE_IMAGE=eavesdrop-cpu-base -t eavesdrop-server .
```

AMD's [7.2.3 PyTorch installation guide](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.2.3/install/3rd-party/pytorch-install.html) lists the selected image. The [compatibility matrix](https://rocm.docs.amd.com/en/docs-7.2.3/compatibility/compatibility-matrix.html) lists gfx1030 and gfx1100. Neither W6800 Pro nor W7800 Pro inference has been validated here. The integrated gfx1103 target is not verified by this base and may need a different compatible vendor build; it is not ruled out. Test the actual GPU and driver combination.

The old `GFX_ARCH` build option is gone: it controlled compilation of the previous backend and cannot change the architectures supported by a prebuilt ROCm PyTorch wheel. ROCm remains experimental for NeMo/Parakeet; validate on your actual hardware before relying on it. The adapter disables NeMo's optional CUDA graph decoder before model construction, avoiding its NVIDIA-driver probe on CPU/ROCm hosts. No HSA override is applied automatically. Supply `HSA_OVERRIDE_GFX_VERSION` only if your hardware requires one and you have verified the value.

## Docker runs

Create `packages/server/config.yaml` from the sample first. Use the same accelerator selection for build and run:

```bash
task run-server-image                         # CPU
task run-server-image ACCELERATOR=cuda        # NVIDIA Container Toolkit required
task run-server-image ACCELERATOR=rocm        # Experimental; requires /dev/kfd and /dev/dri

# Background container with restart policy
task run-server-image-persistent ACCELERATOR=cuda
task stop-server-image
```

The tasks expose port `9090`, mount the config read-only, and persist model downloads in `~/.cache/lmnop/eavesdrop-models`. The container sets `HF_HOME` and `NEMO_CACHE_DIR` inside that mounted cache. CUDA runs pass `--gpus all`. ROCm runs pass the two device mappings and `--security-opt seccomp=unconfined`; this relaxes container isolation and is unnecessary for CPU/CUDA.

For an end-to-end check, start the selected image and transcribe a short English WAV:

```bash
task transcribe-file audio_file=/absolute/path/to/speech.wav
```

The CLI prints final text. To inspect word start/end times, use `EavesdropClient.transcribe_file()` and inspect `result.segments` and each segment's `words`. Also test a recording with leading/trailing silence. CPU, CUDA, and ROCm must each be checked on their intended hosts; a successful dependency import or unit-test run is not an inference benchmark.

## Langfuse tracing

Tracing is env-driven. `task run-server`, `task run-server-image`, and `task run-server-image-persistent` load the repository-root `.env`. Docker tasks pass through `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_BASE_URL` when set. For manual Docker runs, provide the same environment variables. `LOG_LEVEL` controls server verbosity (default `INFO`).

## Real-model smoke test

Use an English 16 kHz mono WAV (the [NVIDIA model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) links a public example):

```bash
uv run --package eavesdrop-server --extra transcription \
  python -m eavesdrop.server.smoke_parakeet /path/to/speech.wav --device cpu
```

This verifies nonempty speech output, native word boundaries at recording offsets 0 and 40 seconds, a sub-second speech excerpt at offset 60, and VAD silence bypass, and reports load/inference timings. Use `--device cuda --device-index 0` for an explicitly requested NVIDIA or HIP GPU; unavailable GPUs fail rather than silently falling back. Model-independent tests use injected typed doubles and never download weights.

### Migration validation (2026-10-18)

A real CPU adapter smoke passed with NeMo 2.7.3, Torch 2.9.1+cpu, the official model-card sample `2086-149220-0033.wav` (7.435 seconds), and `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4`. It produced 23 native-timed words at offsets 0 and 40, passed the 0.8-second speech excerpt at offset 60, and bypassed silence through VAD. First inference took 4.61 seconds; the warm repeat took 2.92 seconds (real-time factor 0.39). Cached model initialization took 78.2 seconds. This is one sample on a shared CPU host, not an accuracy evaluation or an end-to-end WebSocket benchmark.

The downloaded model snapshot was `ae9ad07059c7c739ffaf932226a8fe64ae2620b0`. Initial attempts exposed small `/tmp` extraction capacity and memory pressure from a RAM-backed dependency cache; moving the cache and extraction directory to disk allowed the run to complete on the 9.8 GiB host. Docker builds and CUDA/ROCm inference were not executed here.

A separate loopback WebSocket run passed with the actual server and file client: the file produced 23 timed words and terminated. Paced live PCM produced four updates and exactly one flush completion; a second 0.8-second recording on the same socket reset timestamps and completed exactly once. Segment completion, native word containment, and the single incomplete-tail invariant held. This uncovered and fixed finite EOF re-decoding: an EOF-covering snapshot now completes the last sentence and drains residual audio even without trailing silence; older snapshots cannot discard later audio.

The five Python package suites passed 499 tests with 7 optional DBus skips. Workspace/server type checks and lint/format checks on every changed Python file passed. Repository-wide Ruff still reports 20 pre-existing issues in the historical `openspec` client contract and untouched active-listener CLI. The lockfile consistency check passed with uv 0.12.23.
