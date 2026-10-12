# audiocpp-go

Go bindings for [audio.cpp](https://github.com/0xShug0/audio.cpp) — text-to-speech and audio inference on [ggml](https://github.com/ggml-org/ggml) — via [purego](https://github.com/ebitengine/purego) (no cgo).

audio.cpp exposes no C API of its own, so this project provides one: a thin `extern "C"` shim over its C++ `engine::runtime` facade, compiled into a self-contained shared library that Go loads at runtime.

## How it works

- **`shim/audiocpp_c.{h,cpp}`** — the `extern "C"` surface (`audiocpp_load` / `audiocpp_synthesize` / `audiocpp_free_samples` / `audiocpp_free` / `audiocpp_version`, plus the device registry: `audiocpp_device_count` / `audiocpp_device_info` / `audiocpp_device_memory` / `audiocpp_resolve_device`). It is pointer-only, so it binds cleanly with purego. The shim wraps `make_default_registry → load → create_task_session → prepare → run → AudioBuffer`; the session is created per synthesis and keyed on task (`tts` / `voice_design` / `voice_clone`), so preset voices, voice design and cloning share one loaded model. C++ exceptions never cross the boundary — failures return a status code plus a message string.
- **`CMakeLists.txt`** — builds audio.cpp's static `engine_runtime` and the shim into a single SHARED `libaudiocpp` with global PIC and hidden visibility, exporting only the nine `audiocpp_*` symbols (`-exported_symbols_list` on macOS, `--exclude-libs,ALL` + a version script on Linux) so nothing from ggml/sentencepiece/etc. leaks.
- **`audiocpp.go` + `binding.go` + `library_{unix,windows}.go`** — the Go binding (package `audiocpp`): `Load(libDir)`, `New(ModelParams)`, `(*Model).Synthesize(SynthParams) → *Audio`, `Close()`, `Version()`, `EncodeWAV()`, the device registry (`Devices()`, `DeviceMemory()`, `DeviceID()`, `ResolveDevice()`, `(*Model).Device()`), and a typed `Error{Code, Msg}`. purego binds the nine symbols (the four device ones are optional, so an older library still loads); native PCM is copied into Go memory and then freed, so callers never hold C-owned buffers.

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

## Devices and device memory

The library links ggml statically and hides it, so nothing outside can see which devices it uses or how it numbers them. The device registry exposes that: the library's own device list, each device's free and total memory, and the device a given model runs on. It is what lets a caller attribute a loaded model's memory to the right GPU on a host with several, without `nvidia-smi` (whose GPU indices are not CUDA's).

```go
devs, err := audiocpp.Devices() // every device, in the library's own order
if errors.Is(err, audiocpp.ErrDeviceAPIUnavailable) {
    // libaudiocpp older than v0.2.0: it loads and synthesises, but has no registry.
}

dev, _ := m.Device()                              // the device this model runs on
before, _ := audiocpp.DeviceMemory(dev.Index)
audio, _ := m.Synthesize(params)                  // the first synthesis allocates on the device
after, _ := audiocpp.DeviceMemory(dev.Index)
footprint := before.FreeBytes - after.FreeBytes   // guard against after > before
```

| Call | Returns |
| --- | --- |
| `HasDeviceAPI() bool` | Whether the loaded library has the registry (v0.2.0 or later). |
| `Devices() ([]Device, error)` | Every device the library can run on. Fixed for the life of the process. |
| `DeviceMemory(index int) (Memory, error)` | `FreeBytes` and `TotalBytes` of `Devices()[index]`, read now. |
| `DeviceID(index int) (string, error)` | The device's PCI bus id (`0000:01:00.0`) on CUDA, ROCm and Vulkan, else `""`. |
| `ResolveDevice(backend string, device int) (Device, error)` | The device `ModelParams{Backend, Device}` resolves to, without loading a model. `cpu` and `best` ignore `device`. |
| `(*Model).Device() (Device, error)` | The same, for a loaded model. |

A `Device` carries `Index` (its position in `Devices()`), `Backend` (ggml's registry name: `CUDA`, `Vulkan`, `MTL`, `ROCm`, `CPU`), `BackendIndex` (the index within that backend, which is what `ModelParams.Device` takes), `Name` (`CUDA0`), `Description` (the hardware name) and `Type` (`DeviceTypeCPU` / `GPU` / `IGPU` / `Accel` / `Meta`).

Things to know when measuring with it:

- **Numbering.** For CUDA, `BackendIndex` is CUDA's own device ordinal (fastest card first unless `CUDA_DEVICE_ORDER` is set; `CUDA_VISIBLE_DEVICES` renumbers), not `nvidia-smi`'s index. Use `DeviceID` to match a device against another tool's list.
- **It is a device reading, not a per-model sum.** A model's share is the drop in `FreeBytes` across its load and its first synthesis (the engine allocates on the device at the first synthesis, not at load), so it covers what the model keeps afterwards, not the peak during synthesis. audio.cpp does not account its buffers per model, so the library cannot report an exact per-model byte count. `(*Model).Device()` is the device the model was loaded for; a model may still keep some parts on the CPU.
- **What each backend reports.**

  | Backend | `DeviceMemory` |
  | --- | --- |
  | CUDA | The whole card, so other processes move it too. On a shared-memory host (DGX Spark, Jetson) free is the system's available memory. Every device is typed `GPU`, shared-memory ones included. |
  | Metal | This process's recommended working set less what this process has allocated. |
  | Vulkan | This process's heap budget less its use; `FreeBytes == TotalBytes` on a driver without `VK_EXT_memory_budget`. |
  | CPU (in every build) | System memory, with `FreeBytes == TotalBytes` except on Windows, so it cannot show a drop. Measure a CPU model by process RSS. |

- **Cost.** `Devices`, `ResolveDevice` and `(*Model).Device()` only read the registry. `DeviceMemory` and `DeviceID` open the device they are asked about: on CUDA that creates this process's context on that GPU if it has none (a few hundred MB of that card), so read the model's device, not every device. Take the first reading before the window you want to measure, so the context is not counted as the model. A CUDA driver error during a read aborts the process, as it does inside the engine.
- **Threading.** All of these are stateless and safe to call from any goroutine, including while a `Synthesize` is running.

## Building the shared library

```bash
# Clone the pinned upstream (into upstream/audio.cpp):
scripts/clone-upstream.sh

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DAUDIOCPP_SRC=upstream/audio.cpp -DAUDIOCPP_BACKEND=metal
cmake --build build --target audiocpp --parallel

# Verify only the shim's symbols are exported (fail-closed):
scripts/check-symbols.sh build/libaudiocpp.dylib

# Read the built library's device registry from Go (no model needed):
AUDIOCPP_GO_DEVICES=1 AUDIOCPP_GO_EXPECT_BACKEND=MTL AUDIOCPP_GO_RESOLVE_BACKEND=metal \
  go test -run TestDeviceRegistryIntegration .

# Go end-to-end against the local build (needs a model GGUF):
AUDIOCPP_GO_E2E=1 go test ./...
```

`AUDIOCPP_BACKEND` selects the ggml backend (`cpu` / `cuda` / `vulkan` / `hip` / `metal`). `AUDIOCPP_NATIVE_CPU` (default `ON`) tunes the CPU kernels for the build machine; pass `-DAUDIOCPP_NATIVE_CPU=OFF` for a library that must run on other CPUs. `AUDIOCPP_MODEL_SET` defaults to a small set for fast local builds; release builds pass `-DAUDIOCPP_MODEL_SET=full`.

## Prebuilt libraries & releases

`.github/workflows/build-libs.yml` builds a self-contained `libaudiocpp` for each variant — linux amd64 (cpu / cuda / vulkan), linux arm64 (cpu / cuda), darwin arm64 (metal), windows amd64 (cpu) — runs the fail-closed symbol gate on each, and publishes one `vX.Y.Z` release carrying the module tag (so `go get github.com/pendra-ai/audiocpp-go@vX.Y.Z` resolves) alongside per-variant `audiocpp-libs-<os>-<arch>-<backend>.tar.gz` archives and `checksums.txt`. Each leg whose library can be opened on its runner (CPU, Vulkan, Metal, Windows) also reads the device registry through the Go binding; the CUDA legs are build/link/symbol-check only (no GPU CI runners). Pull requests that touch the native build run the same matrix as a check, without publishing. The amd64 CUDA and Windows legs are best-effort, so a toolchain mismatch never blocks a release; the arm64 CUDA leg gates the release. The linux arm64 archives are built with generic CPU flags (`-DAUDIOCPP_NATIVE_CPU=OFF`) so they run on Arm cores other than the CI runner's. The CUDA archives need `libcudart.so.13`, `libcublas.so.13` and `libcufft.so.12` from the host's CUDA 13 runtime. Consumers extract an archive and pass its directory to `Load(libDir)`.

## Pin

audio.cpp is pinned in `lib/version.txt`. The shim compiles against audio.cpp's internal C++ headers, which move quickly, so the pin is intentionally strict.

## Licensing

This project (the Go code and the `extern "C"` shim) is MIT-licensed — see [`LICENSE`](LICENSE).

The prebuilt shared libraries statically link **audio.cpp** (Apache-2.0), **ggml** (MIT), and **sentencepiece** (Apache-2.0). Their licence and NOTICE files are bundled inside each released `audiocpp-libs-*.tar.gz`.
