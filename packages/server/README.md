# Eavesdrop server

English speech recognition using the fixed [`oruk/orukeet`](https://huggingface.co/oruk/orukeet) model through standard PyPI `sherpa-onnx` on CPU (INT8 ONNX) or pinned NeMo-Speech.cpp on Vulkan (Q8 GGUF). Run directly on the host. The server retains WebSocket and RTSP streaming, silence-based completion, and recording-relative word timestamps.

## Runtime and timing

The CPU backend's released INT8 encoder, decoder, and joiner use sherpa-onnx's offline `nemo_transducer` recognizer with 128-dimensional features, 16 kHz input, the CPU provider, and greedy decoding. The Vulkan backend uses pinned NeMo-Speech.cpp with a Q8 GGUF model on Linux x86_64 and selects GPU 0. Streaming remains rolling offline chunk re-decoding with the existing buffering, cancellation, flush, and silence-completion behavior; it is not a stateful streaming recognizer.

CPU native token onsets and durations have 80 ms granularity. Subword tokens are grouped into words without proportional duration estimates or a separate alignment stage; punctuation and exact recognized text are preserved. Vulkan native word timestamps are converted from milliseconds. Standalone Silero VAD remains ONNX CPU to detect speech and silence, retaining PyTorch/torchaudio as its dependencies. For both backends, each speech span's original sample offset and the stream chunk's absolute offset are added once. Word times therefore remain on the original recording timeline across VAD gaps.

For CPU, the adapter downloads `onnx/sherpa-onnx-orukeet-v0.1.0-int8.tar.bz2` from `oruk/orukeet` at revision `55a984d46f68323301837194ce647c702f55facc` through `huggingface_hub.hf_hub_download`. The archive SHA256 is `f9191f30178cc9122ce2f023bf9fefafc822028307b0efa4caff645ba3fe8d0a`. Downloads use the standard Hugging Face cache; extracted files are reused after safe tar data-filter extraction. First startup requires network access and disk space for both archive and extracted model. Vulkan uses separately pinned NeMo-Speech.cpp and Q8 GGUF assets. No runtime benchmark or accuracy improvement is implied by either model packaging.

## Native host setup

Use Python 3.12 and `uv`, from the repository root. FFmpeg supports RTSP ingestion and file decoding; libsndfile supports WAV loading:

```bash
sudo apt-get install build-essential ffmpeg libsndfile1
uv sync --package eavesdrop-server --extra transcription
cp packages/server/config.sample.yaml packages/server/config.yaml
uv run --package eavesdrop-server --extra transcription eavesdrop-server \
  --config packages/server/config.yaml
```

The `transcription` extra supplies sherpa-onnx, huggingface-hub, Silero, and its Torch/audio dependencies. Lightweight development can omit the extra. ONNX Runtime and Hugging Face usage telemetry are disabled before runtime imports. Docker also sets `ORT_DISABLE_TELEMETRY=1` before Python starts, following Microsoft's [process-lifetime opt-out](https://github.com/microsoft/onnxruntime/blob/main/docs/Privacy.md#disabling-telemetry).

The runtime pins both `sherpa-onnx` and `sherpa-onnx-core` to `1.13.4`. The explicit core dependency is required because uv's package metadata omitted that transitive dependency even though the installed sherpa-onnx wheel declares it; both packages are needed for `import sherpa_onnx`.

The asset loader selects Hub's standard HTTP transport before importing the Hub library. Xet's reconstruction endpoint returned HTTP 416 for this pinned archive during host verification. The archive is still SHA256-verified before extraction.

The Vulkan SDK bundles an older C++ runtime that cannot load current Mesa drivers. The adapter loads the host `libstdc++.so.6` before the SDK, allowing both the SDK and Vulkan driver to use the host runtime. Vulkan requires the host Vulkan loader, GPU driver, and C++ runtime; startup fails if GPU 0 is unavailable.

## Configuration and migration

```yaml
transcription:
  model: oruk/orukeet
  device: cpu
  language: en
  word_timestamps: true
  use_vad: true
```

- `model` is fixed; alternate IDs and local custom checkpoints are rejected.
- `device` accepts `cpu` (default) or `vulkan`. Vulkan requires a Linux x86_64 host with a Vulkan-capable GPU and driver, and uses GPU 0; configurable GPU selection via `device_index` is unsupported. Docker remains CPU-only.
- The client contract remains English-only (`language: en`), without a new language-selection feature or translation.
- Input audio remains mono float32 at 16 kHz, with original recording offsets preserved.
- `word_timestamps: false` hides client word details; native timing still drives segmentation and completion.

Start from `config.sample.yaml` when upgrading. Removed options such as `custom_model`, `initial_prompt`, `hotwords`, `beam_size`, `num_workers`, `same_output_threshold`, and `gpu_name` remain unsupported. Clients must not send removed task/model options. The former Python NeMo runtime and its caches are not used; the Vulkan backend uses NeMo-Speech.cpp instead.

## Optional CPU Docker packaging

Host deployment is preferred. Docker packaging is CPU-only; the former CUDA and ROCm base images, build tasks, and GPU device mappings have been removed. PyTorch/torchaudio CPU wheels remain solely for standalone Silero VAD, not recognition inference.

```bash
task build-server-image
task run-server-image
# Or run in the background:
task run-server-image-persistent
# Stop and remove the container:
task stop-server-image
```

The final image installs the `transcription` extra using the workspace-root `uv.lock` and `/opt/venv`, preserving the base's CPU Torch/audio wheels. Build-time dependency import checks are not inference tests. Override names with `CPU_BASE_IMAGE` and `SERVER_IMAGE`; `SKIP_CACHE=true` disables caching for the final image build.

Equivalent manual builds, also available through `packages/server/scripts/build.sh`:

```bash
docker build -f packages/server/docker/Dockerfile.cpu-base -t eavesdrop-cpu-base .
docker build -f packages/server/docker/Dockerfile \
  --build-arg BASE_IMAGE=eavesdrop-cpu-base \
  -t ghcr.io/shyndman/lmnop-eavesdrop:latest .
```

Container tasks expose port `9090`, mount `packages/server/config.yaml` read-only, and persist downloads in `~/.cache/lmnop/eavesdrop-models`. `HF_HOME` points inside that mounted cache. Keep `transcription.device: cpu` for Docker; no GPU permissions or relaxed container security are needed.

## Langfuse tracing

Tracing is env-driven. `task run-server`, `task run-server-image`, and `task run-server-image-persistent` load the repository-root `.env`. Docker tasks pass through `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_BASE_URL` when set. For manual Docker runs, provide the same environment variables. `LOG_LEVEL` controls server verbosity (default `INFO`).

## Real-model smoke test

From the repository root, use an English 16 kHz mono WAV and a configuration selecting the desired backend:

```bash
uv run --package eavesdrop-server --extra transcription \
  python -m eavesdrop.server.smoke_orukeet /path/to/speech.wav \
  --config packages/server/config.yaml
```

The smoke module exercises real recognition, recording-offset word timing, short audio, and VAD silence bypass, and reports load/inference timings. Model-independent unit tests use typed doubles and do not download weights. For end-to-end validation, run the server and transcribe a file through `EavesdropClient.transcribe_file()`, inspecting each segment's words as well as final text. Include recordings with leading, trailing, and internal silence; dependency import checks alone do not establish recognition or streaming correctness.

### Previously verified CPU host run

A real CPU smoke run with sherpa-onnx/core 1.13.4 transcribed a 10.435-second mono 16 kHz WAV in 1.27 seconds (real-time factor 0.122), producing 28 timed words. Re-decoding at a 40-second recording offset preserved the word timeline. A 0.8-second speech excerpt at offset 60 and VAD silence bypass also passed. First load including archive extraction took 49.5 seconds. These are observations from one local sample, not an accuracy benchmark.

The same CPU sample also passed a real WebSocket upload through `EavesdropClient.transcribe_file()`, including final completion and all 28 word boundaries. Concatenating two copies with a two-second silence gap produced two VAD spans and retained their recording-relative offsets within the CPU model's 80 ms timing grid. These CPU observations do not establish Vulkan correctness or performance.

### Verified Vulkan host run

A real Vulkan smoke run selected `Vulkan0` on an AMD Radeon 780M (Mesa RADV 26.0.8). A 12-second excerpt produced 21 timed words across three VAD spans, retaining leading and internal silence. The same excerpt passed at a 40-second recording offset; a 0.8-second speech excerpt at offset 60 and VAD silence bypass also passed. With cached assets, model/VAD load took 3.33 seconds, the first full pass took 0.93 seconds, and the second took 0.49 seconds. These are single-sample host observations.

A real Vulkan server also passed WebSocket upload through `EavesdropClient.transcribe_file()`. A 14-second recording containing two speech copies, leading/trailing silence, and a two-second inserted gap yielded two completed segments and 12 timed words without warnings. Every delivered word boundary matched direct inference of the same waveform exactly. Independent VAD crops can change native word alignments, so repeated speech copies are checked against their own native boundaries rather than an assumed constant word-time shift.
