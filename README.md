# audiocpp-go

Go bindings for [audio.cpp](https://github.com/0xShug0/audio.cpp) — text-to-speech and audio inference on [ggml](https://github.com/ggml-org/ggml) — via [purego](https://github.com/ebitengine/purego) (no cgo).

audio.cpp exposes no C API of its own, so this project provides one: a thin `extern "C"` shim over its C++ `engine::runtime` facade, compiled into a self-contained shared library that Go loads at runtime.

## How it works

- **`shim/audiocpp_c.{h,cpp}`** — the `extern "C"` surface (`audiocpp_load` / `audiocpp_synthesize` / `audiocpp_free_samples` / `audiocpp_free` / `audiocpp_version`). It is pointer-only, so it binds cleanly with purego. The shim wraps `make_default_registry → load → create_task_session → prepare → run → AudioBuffer`; the session is created per synthesis and keyed on task (`tts` / `voice_design` / `voice_clone`), so preset voices, voice design and cloning share one loaded model. C++ exceptions never cross the boundary — failures return a status code plus a message string.
- **`CMakeLists.txt`** — builds audio.cpp's static `engine_runtime` and the shim into a single SHARED `libaudiocpp` with global PIC and hidden visibility, exporting only the five `audiocpp_*` symbols (`-exported_symbols_list` on macOS, `--exclude-libs,ALL` + a version script on Linux) so nothing from ggml/sentencepiece/etc. leaks.
- **`audiocpp.go` + `binding.go` + `library_{unix,windows}.go`** — the Go binding (package `audiocpp`): `Load(libDir)`, `New(ModelParams)`, `(*Model).Synthesize(SynthParams) → *Audio`, `Close()`, `Version()`, `EncodeWAV()`, and a typed `Error{Code, Msg}`. purego `RegisterLibFunc` binds the five symbols; native PCM is copied into Go memory and then freed, so callers never hold C-owned buffers.

## Usage

```go
import audiocpp "github.com/pendra-ai/audiocpp-go"

// Point Load at a directory containing the prebuilt libaudiocpp for your platform.
if err := audiocpp.Load(libDir); err != nil { log.Fatal(err) }

m, err := audiocpp.New(audiocpp.ModelParams{
    ModelPath:  "/path/to/model.gguf",
    FamilyHint: "qwen3_tts",
    Backend:    "metal", // "cpu" | "cuda" | "vulkan" | "hip" | "metal" | "best"
})
if err != nil { log.Fatal(err) }
defer m.Close()

audio, err := m.Synthesize(audiocpp.SynthParams{
    Text:    "Hello from audio dot cpp, bound to Go.",
    VoiceID: "ryan",
})
if err != nil { log.Fatal(err) }

os.WriteFile("out.wav", audiocpp.EncodeWAV(audio), 0o644)
```

A `Model` is not safe for concurrent use — synthesise serially.

See [`examples/tts`](examples/tts) for a complete runnable example.

## Building the shared library

```bash
# Clone the pinned upstream (into upstream/audio.cpp):
scripts/clone-upstream.sh

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DAUDIOCPP_SRC=upstream/audio.cpp -DAUDIOCPP_BACKEND=metal
cmake --build build --target audiocpp --parallel

# Verify only the shim's symbols are exported (fail-closed):
scripts/check-symbols.sh build/libaudiocpp.dylib

# Go end-to-end against the local build (needs a model GGUF):
AUDIOCPP_GO_E2E=1 go test ./...
```

`AUDIOCPP_BACKEND` selects the ggml backend (`cpu` / `cuda` / `vulkan` / `hip` / `metal`). `AUDIOCPP_NATIVE_CPU` (default `ON`) tunes the CPU kernels for the build machine; pass `-DAUDIOCPP_NATIVE_CPU=OFF` for a library that must run on other CPUs. `AUDIOCPP_MODEL_SET` defaults to a small set for fast local builds; release builds pass `-DAUDIOCPP_MODEL_SET=full`.

## Prebuilt libraries & releases

`.github/workflows/build-libs.yml` builds a self-contained `libaudiocpp` for each variant — linux amd64 (cpu / cuda / vulkan), linux arm64 (cpu / cuda), darwin arm64 (metal), windows amd64 (cpu) — runs the fail-closed symbol gate on each, and publishes one `vX.Y.Z` release carrying the module tag (so `go get github.com/pendra-ai/audiocpp-go@vX.Y.Z` resolves) alongside per-variant `audiocpp-libs-<os>-<arch>-<backend>.tar.gz` archives and `checksums.txt`. GPU and Windows legs are build/link/symbol-check only (no GPU CI runners). The amd64 CUDA and Windows legs are best-effort, so a toolchain mismatch never blocks a release; the arm64 CUDA leg gates the release. The linux arm64 archives are built with generic CPU flags (`-DAUDIOCPP_NATIVE_CPU=OFF`) so they run on Arm cores other than the CI runner's. The CUDA archives need `libcudart.so.13`, `libcublas.so.13` and `libcufft.so.12` from the host's CUDA 13 runtime. Consumers extract an archive and pass its directory to `Load(libDir)`.

## Pin

audio.cpp is pinned in `lib/version.txt`. The shim compiles against audio.cpp's internal C++ headers, which move quickly, so the pin is intentionally strict.

## Licensing

This project (the Go code and the `extern "C"` shim) is MIT-licensed — see [`LICENSE`](LICENSE).

The prebuilt shared libraries statically link **audio.cpp** (Apache-2.0), **ggml** (MIT), and **sentencepiece** (Apache-2.0). Their licence and NOTICE files are bundled inside each released `audiocpp-libs-*.tar.gz`.
