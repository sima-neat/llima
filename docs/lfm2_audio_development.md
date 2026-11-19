# LFM2.5 Audio development target

The explicit compiler covers the language backbone, FastConformer encoder,
adapter, depth transformer and default LFM2 audio detokenizer. A private C++
runtime implements persistent caches, the eight-codebook loop and sequential/
interleaved generation. This is an experimental contributor workflow; the
ordinary `llima-compile`, `llima run` and GenAI server paths do not register it.

## Source and precision

- Checkpoint: `LiquidAI/LFM2.5-Audio-1.5B`, revision
  `95d7e5a0ca706151aa7b7989275a902007a04cbe`.
- Reference: [liquid-audio](https://github.com/Liquid4All/liquid-audio/tree/19e65845923a7f136442c95137884ec61eb386aa).
- MLA compute: BF16 activations and INT8 quantizable weights, `A_BF16_W_INT8`.
- KV state is BF16. CPU embedding tables retain the original checkpoint dtype;
  normalization parameters, constants and scales follow compiler precision rules.
  This does not mean every stored tensor occupies eight bits.

Use an unquantized single-file snapshot containing the root `config.json`,
`model.safetensors`, tokenizer JSON assets, `chat_template.jinja`, and
`audio_detokenizer/{config.json,model.safetensors}`. Sharded, pre-quantized, GGUF
audio checkpoints and the optional older Mimi decoder are outside this target.
The adapter exposes separate component views without relabeling the whole audio
architecture as a text LLM. Depth Q/K weights and norm scales are permuted
together for rotary ordering. Decoder embeddings are averaged; generation
embeddings are summed over eight codebooks.

## Compile and package

From the checkout in the Model Compiler environment:

```bash
source /sdk-extensions/model-compiler/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
python scripts/gen_models--liquidai--lfm2_audio.py /path/to/snapshot \
  -o /path/to/audio-build
```

The default plan contains 104 units. The output includes quantized SDK graphs,
MPK archives and `audio_compile_manifest.json`, with source hashes, compiler
versions, graph signatures, tensor interfaces and archive hashes.
`complete_compute_build` becomes true after every planned unit compiles;
compilation alone does not establish runtime or speech quality.

Use `--resume` after interruption. Matching completed archives are reused.
Floating SDK intermediates are removed after successful compilation unless
`--keep-fp32` is set. Diagnostics include `--component encoder`,
`--unit 'encoder_n32_layer*'`, `--list` and `--stage export|quantize|compile|all`.

Defaults: 1,024-token language context, 16-token prefill, 256 input mel frames,
12 output code frames per decoder chunk. The encoder bucket accepts about
2.55 seconds of 16 kHz PCM. Longer utterances require a larger `--audio-frames`
build; splitting a bidirectional encoder input is not an equivalent workaround.
Other sizing flags are `--context-length`, `--group-size`, `--decoder-frames`.

The encoder executes its full compiled bucket even for a shorter utterance,
then returns only valid audio embeddings to the backbone. A larger bucket can
therefore increase encoder time for short clips without adding padding tokens
to the language prompt. An actual longer utterance adds valid prompt tokens
and prefill work, and leaves less room in the language context. It does not
change depth/decoder graph shapes. In the current fixed 1,024-token attention
build, generation already scans the compiled cache shape. Exactly four seconds
of 16 kHz PCM produces 401 mel frames, so a 400-frame bucket is insufficient;
the larger encoder still requires compilation and device validation.

For per-frame PCM output, compile with `--decoder-frames 1`. The tested
lower-latency profile additionally uses `--group-size 32`:

```bash
python scripts/gen_models--liquidai--lfm2_audio.py /path/to/snapshot \
  -o /path/to/audio-streaming-build --decoder-frames 1 --group-size 32
```

Package that build with `prepare_lfm2_audio_runtime` as below. The default
12-frame decoder waits for 12 generated audio-code frames before emitting PCM;
changing only the server cannot remove that buffering. The one-frame plan still
contains 104 units; its 14 detokenizer units process six neural timesteps per
audio-code frame. The group size selects the backbone prefill shape; the
default remains 16. Use a fresh output directory when changing build options.

After the complete build:

```bash
python -m sima_lmm.host.prepare_lfm2_audio_runtime \
  /path/to/snapshot /path/to/audio-build
```

Packaging verifies source/graph/configuration identity, archive hashes and one
MLA segment per unit without A65 fallback, then writes `audio-build/runtime/`.
It extracts MLA ELFs and exports host tables and tokenizer assets. The runtime
uses recorded raw BF16 interfaces, bypassing MPK cast wrappers.

The encoder preserves physical input boundaries through explicit stage masks.
Its fixed relative-position projection is folded offline, and relative shift
uses static row selection. This avoids the compiler's constant-only projection
layout failure while retaining the tested floating computation. Inference batch
normalization uses a depthwise 1x1 affine convolution: the current compiler's
constant-vector multiplication produced incorrect channel scales on Modalix.

## Native execution

Build in the Neat SDK:

```bash
./build.sh --core --dev --no-dist --jobs 1 --build-dir /path/to/runtime-build \
  -- -DSIMA_LMM_BUILD_PYTHON=OFF -DSIMA_LMM_BUILD_TESTS=ON
```

On Modalix, use the isolated build library if necessary instead of replacing
the installed runtime:

```bash
LD_LIBRARY_PATH=/path/to/runtime-build:$LD_LIBRARY_PATH \
  /path/to/runtime-build/tests/runtime/sima_lmm_lfm2_audio_smoke \
  /path/to/audio-build/runtime "Say hello." /tmp/audio-output.f32 32
```

Output is raw mono float32 PCM at 24 kHz. The optional next argument supplies
raw mono float32 input PCM at 16 kHz; use `-` when no input is supplied. A
task argument accepts `interleaved` (default), `asr` or `tts`. ASR and TTS use
sequential generation and the pinned source's fixed system prompts. TTS selects
the UK male voice, temperature 0.8 and top-k 64; other tasks use greedy sampling.
The runner uses the saved Jinja template through Minja and the saved tokenizer.
It reports text, code frames, samples and elapsed time. The step limit is a
generation limit, not an audio-duration limit.

For example, append `64 /tmp/clip-16k.f32 asr` after the output filename for
transcription, or `128 - tts` for text-to-speech. ASR prints its text and leaves
the output PCM file empty.

An optional repeat count follows the task, for example `128 - tts 6`. Each
request resets session state and the sampling seed while retaining loaded
weights. Only the last request's PCM file is retained. `metrics_json` records
first output event, first visible text, first complete audio-code frame, PCM
availability and per-stage times. Requests start before audio preprocessing and
prompt preparation; constructor, input-file reads, HTTP and network time are
excluded. The legacy `elapsed_seconds` covers generation plus waveform decoding
and excludes audio preprocessing/encoding and prompt preparation.

Temporal, depth and decoder caches are separate. Depth resets for every frame.
Partial prompt tails use a zero-padded prefill group, with attention masks,
cache positions and convolution tails advanced only through real tokens. The
last real row supplies the returned hidden state. Single-token generation stays
on the single-token graph; near cache capacity, a partial tail falls back to
single-token calls when a full physical group would not fit.
The decoder keeps 29 previous attention entries, absolute rotary positions,
convolution tails and waveform overlap state. A response reset clears session
state. This is a single-session contributor runtime; it does not add an
audio-output GenAI task or real-time guarantees. The optional example below
provides HTTP streaming and browser-side input resampling.

## Browser microphone and speaker server

Build the optional example in the Neat SDK by adding
`-DSIMA_LMM_BUILD_AUDIO_SERVER=ON` to the build command above. On Modalix:

```bash
LD_LIBRARY_PATH=/path/to/runtime-build:$LD_LIBRARY_PATH \
  /path/to/runtime-build/examples/lfm2_audio/sima_lmm_lfm2_audio_server \
  /path/to/audio-build/runtime /path/to/llima/examples/lfm2_audio/web \
  8096 127.0.0.1
```

The `extras` install component includes the executable in `bin/` and web assets
in `share/sima-lmm/lfm2_audio/`. It uses the native audio backend directly;
the normal `llima run` and Neat GenAIServer remain separate interfaces.

On the computer with your microphone and speakers, keep this tunnel running:

```bash
ssh -N -L 8096:127.0.0.1:8096 sima@DEVKIT
```

Open `http://localhost:8096`. Allow microphone access, click **Record question**,
say a short question, and click **Send recording**. Recording stops automatically
at 2.5 seconds for the default bucket. Browser input is resampled to mono 16 kHz;
the reply plays at 24 kHz. You can use the computer's built-in microphone and
speakers or a headset; USB audio devices on Modalix are unnecessary. Browser
microphone access requires a secure context, which localhost provides; opening
the DevKit's plain HTTP LAN address does not provide that context. See
[the browser microphone contract](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia).

Weights remain loaded between requests; conversation state resets for each
request. One generation runs at a time, with HTTP 409 for a concurrent request.
**Stop reply** cancels the current request and playback. Playback starts when a
decoder chunk is ready; codes alone are not playable audio. The page reports
server text/PCM/completion times and browser PCM arrival/scheduled playback
times. Recording time is excluded, and playback may continue after generation
finishes. The first request includes lazy model loading.

The experimental HTTP contract is:

- `GET /api/health`: model, input limit, sample rates and busy state.
- `POST /api/speech?id=REQUEST_ID`: raw mono little-endian float32 PCM at 16 kHz,
  `Content-Type: application/octet-stream`; optional `task=asr` selects transcription.
- `POST /api/text?id=REQUEST_ID`: JSON with `text`, optional `task` (`interleaved`
  or `tts`) and `max_steps` (1–512, default 320).
- `POST /api/stop?id=REQUEST_ID`: cancels only the matching active request.

Generation responses are newline-delimited JSON events: `start`, cumulative
`text`, `audio` (base64 `pcm_s16le`, 24 kHz mono), and `done` with timings. Errors
after streaming starts use an `error` event. Input lengths and values are
validated; oversized clips are rejected rather than split or truncated. The
server binds localhost by default and disables cross-origin browser requests.
It is an experimental single-user example without authentication or TLS.

The final event also separates `audio_encode_ms`, `prompt_prepare_ms`, backend
prefill/backbone/text-head/depth timings, `decoder_network_ms` and
`inverse_fft_overlap_cpu_ms`. The bounded `backend_profile.first_steps` trace
records at most 64 backend calls. Decoder/FFT timings accumulate across the
request; first-PCM time is measured separately. TCP_NODELAY is enabled for
streamed responses.

Set `SIMA_LLIMA_RUN_PROFILE=1` when starting the server to log each ELF's MLA
tile time and submit-to-reap time. The audio runtime honors the same native
profiling environment variables as the ordinary runtime. Disable verbose
profiling for final latency comparisons because logging adds overhead.

For repeatable protocol and device checks, provide a short speech fixture and
write reports outside the checkout:

```bash
python3 examples/lfm2_audio/validate_server.py http://localhost:8096 \
  /path/to/mono-16k.f32 --output /path/to/server-validation
```

This checks invalid input, contention, matching-request cancellation, subsequent
requests and deterministic state reset. It saves streamed replies as WAV files.
Run it while the demo is idle. Actual microphone and speaker quality should also
be checked interactively with the browser.

Resume fingerprints include module helpers and tensor aliases/rotary settings.
Archives made with the earlier development signature are intentionally not
reused by the current compiler. This does not change their existing executables;
fresh packaging with current source requires a current compile manifest.

## Validation and limits

The fast compiler suite has 490 cases. Added hermetic cases cover ingestion,
BF16 loading, depth rotary conversion, projection bias/slicing and subsampling
boundaries. Four checkpoint-backed integration cases cover conv-cache updates,
attention prefill, final normalization and the depth-head epsilon. Additional
pinned-source checks:

| Floating check | Maximum absolute difference |
| --- | --- |
| Selected language blocks | 1.04e-5 |
| Conformer block, full/padded lengths | 5.73e-5 |
| Encoder subsampling | 1.84e-4 |
| Adapter | 3.58e-7 |
| Depth block, eight cached positions and resets | 2.69e-7 |
| Complete eight-layer decoder, two 72-step chunks | 8.70e-5 |
| CPU inverse FFT/overlap-add, 1/6/17 frames | 3.52e-8 |

These selected FP32 comparisons do not establish full-model W8 speech quality.
The native frontend reproduces the Slaney filterbank exactly; tested noise/tone
features differed by at most 1.06e-4. Exact silence differs by up to 0.161 because
the source's FP32 reduction roundoff is amplified by its normalization epsilon.
Do not claim bitwise frontend equivalence. Reference-generated mel features can
be supplied to `encode_mel`.

`sima_lmm_audio_generation_test` checks sampling, generation modes and waveform
chunking. `sima_lmm_audio_component_test` accepts a generated fixture JSON to
save actual MLA outputs for comparison with quantized SDK outputs. Full-model
ASR/TTS/speech-to-speech quality, long sessions and sustained throughput still
require explicit evaluation before ready-to-use support is advertised.

For a selected-unit benchmark on Modalix, append an iteration count (1–10,000):

```bash
LD_LIBRARY_PATH=/path/to/candidate/lib:$LD_LIBRARY_PATH \
  /path/to/sima_lmm_audio_component_test /path/to/fixture.json 200
```

The helper warms the model once, reports median/p95/minimum wall time for
repeated execution with the same fixture inputs, and saves the final outputs.
It does not advance a session cache between iterations or measure complete
speech generation. Compare the saved tensors separately against the reference.

## Local compile and Modalix validation, 2026-11-18

The default 104-unit build completed with `A_BF16_W_INT8`: 49 backbone,
19 encoder, 22 depth and 14 detokenizer units. Every archive contained one MLA
segment and no A65 neural fallback. All 104 executable paths were observed in
successful native runs across the following smoke tests:

- Interleaved text/audio: returned `Hello! How can I help you today?` and
  42,240 waveform samples at 24 kHz from a bounded 32-step run.
- Sequential ASR: consumed 1.76 seconds of resampled generated speech and
  returned `Hello! How are`.
- Sequential TTS: `Hello from Modalix.` generated 38,400 samples. An existing
  Whisper service transcribed the result as `Hello from Modellix.`. This is a
  small intelligibility check, not a speech-quality or WER benchmark.

Selected backbone, subsampling, rewritten Conformer and depth interfaces matched
the quantized SDK outputs bit for bit on Modalix, including physical K/V strides
and a partial-length subsampling input. The compiler wheel, native core/dev
packages and runtime test targets built; packaged audio-generation tests passed
on the board. The normal CLI/server registration boundary above still applies.

## Exploratory native latency, 2026-11-18

Local NVMe execution used one initial request followed by five requests with
weights resident in the same process. Existing model services remained running.
The following are medians of the five resident requests, not sustained or
isolated-service benchmarks:

| Task | Prompt tokens | First visible text | First audio-code frame | Full request |
| --- | --- | --- | --- | --- |
| ASR, 1.76-second input | 42 | 451 ms | N/A | 605 ms |
| Interleaved text/audio | 28 | 449 ms | 708 ms | 3,436 ms |
| TTS, UK male voice | 31 | N/A | 618 ms | 2,003 ms |

TTS's first output was an audio-start control token at 551 ms. The initial
requests had first-visible-text times of 3,201 ms for ASR and 2,563 ms for
interleaved generation; initial TTS audio codes arrived at 2,722 ms. Those
requests included lazy MLA model loading. Constructor time was an additional
826–841 ms, and filesystem caches may have been warm.

The legacy smoke runner buffers all audio codes before waveform decoding. First PCM
therefore arrived at 2,002 ms for the 1.6-second TTS output and 3,435 ms for the
3.04-second interleaved output. Audio codes are not playable PCM. These timings
do not include transport or playback. These measurements predate padded prompt
tails and the one-frame streaming decoder described below.

Speech-in/speech-out was also measured directly, without a text prompt or
separate ASR/TTS pipeline. A complete 1.44-second synthetic question, `Two plus
two?`, produced `Two plus two equals four. Anything else I can help with?` and
10.96 seconds of output audio. Five resident requests had a median complete
request time of 10,215 ms (range 10,205–10,218 ms), including 28.5 ms for input
preprocessing/encoding and 138.8 ms for waveform decoding. First visible text
arrived at 481 ms, first audio codes at 740 ms, and buffered PCM at 10,214 ms.
The initial request took 13,629 ms, plus 825 ms for construction. Recording,
transport and playback are excluded. A separate 1.76-second partial greeting
produced a clarification request and 9.92 seconds of audio in 9,604 ms; both
cases are timing checks, not dataset-level quality results.

## Streaming latency optimization, 2026-11-18

The same complete 1.44-second question was measured through the native HTTP
server with padded prefill and a one-frame decoder. The median of three warm
requests was:

| Event | Server time |
| --- | --- |
| First visible text | 146 ms |
| First complete audio-code frame | 406 ms |
| First playable PCM payload | 410 ms |

Client PCM arrival was about 430 ms. A complete response produced 10.96 seconds
of speech in about 10.31 seconds. Before these changes, streaming with the
12-frame decoder took about 1.48 seconds to first PCM. Padding the partial
43-token prompt reduced prefill from about 444 ms to 108 ms; compiling the
one-frame decoder removed the remaining multi-frame buffering. All figures
exclude recording, upload/network and playback from server timings, and the
first request can be slower due to lazy loading.

The first payload contains 1,440 samples/60 ms because inverse STFT trims the
initial 20 ms. Subsequent payloads contain 1,920 samples/80 ms; final flush
returns 480 samples/20 ms. Neural waveform decoding takes about 3.06 ms per
frame and CPU FFT/overlap about 0.56 ms. The source's six-text/twelve-audio
schedule is retained. First audio under 200 ms and a hard 80 ms deadline for
every frame have not been achieved; early text bursts can cause audio gaps of
about 302 ms. Input still requires the complete utterance.

The prefill regression helper saves seven prefix lengths (2, 7, 15, 16, 17,
31, 43), each followed by three continuation tokens. Run it on Modalix against
the same bundle with the baseline and candidate private libraries, then compare
the two output files:

```bash
LD_LIBRARY_PATH=/path/to/candidate/lib:$LD_LIBRARY_PATH \
  /path/to/sima_lmm_lfm2_audio_prefill_test \
  /path/to/audio-build/runtime /tmp/candidate-prefill.f32
cmp /tmp/baseline-prefill.f32 /tmp/candidate-prefill.f32
```

All 28 states, each 2,048 dimensions, matched bit for bit on SDK 3.0.0 Modalix.
One-frame versus 12-frame decoder waveforms were not bit-identical (selected
PCM comparison: SNR 30.57 dB); independent Whisper transcribed the one-frame
reply correctly. This does not establish broad speech-quality equivalence.
The optimized runtime passed invalid-input, contention, cancellation,
disconnect cleanup and deterministic repeated speech checks. Matching SDK
3.0.1 release validation remains open.

## Further prefill tuning, 2026-11-19

The same 1.44-second synthetic question was compared with 16-token and 32-token
prefill while retaining the one-frame decoder, BF16 activations, INT8 compute
weights and trained interleave schedule. Four isolated server processes ran
in A/B/B/A order, each with six requests; discarding the first request per
process left ten resident samples per profile. Existing services stayed up.

| Prefill group | Median first visible text | Median first PCM |
| --- | --- | --- |
| 16 tokens | 145.626 ms | 409.392 ms |
| 32 tokens | 111.428 ms | 374.747 ms |

This reduces first-PCM time by about 8.5% for this scenario. The 43-token prompt
uses two prefill passes instead of three. The 32-token candidate comprised
24 freshly compiled backbone units and 80 reused units after source,
configuration, topology, interface and archive/ELF identity checks. Its host
tables were checksum-verified. The selected 28 prefix/continuation states
(2,048 dimensions each) and the full 263,040-sample waveform matched the
baseline bit for bit. Protocol, cancellation, disconnect and deterministic
repeated speech checks passed using the extracted candidate core and packaged
server/tests. Independent Whisper recognized the complete expected reply.
These are selected-case results, not full-model or dataset-level parity.

A separate complete speech request took about 10.33 seconds for 10.96 seconds
of audio; the initial lazy-loading request took about 910 ms to first PCM.
Server timings exclude recording, upload/network and playback. The larger
prefill group does not accelerate single-token generation or remove text-burst
audio gaps. The under-200 ms first-audio target and sustained frame deadlines
remain open, as does matching SDK 3.0.1 validation.

Faster-looking alternatives were not adopted. A 64-token profile measured
341.373 ms first PCM, but its 28-state relative L2 difference was 0.14027 and
the test reply continued to the 320-step limit (24.48 seconds of audio).
It is not recommended as an equivalent replacement. A selected gate/up
projection fusion matched floating, quantized SDK and hardware outputs, but
improved that unit only from 2.003 ms to 1.992 ms. Sharing rotary input buffers
passed continuation checks but made no meaningful end-to-end timing difference.
Neither source rewrite was retained. Per-ELF profiling showed about 31.7 ms of
MLA tile execution within a roughly 34 ms backbone step, limiting the saving
available from launch overhead alone.
