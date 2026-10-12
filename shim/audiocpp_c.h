/*
 * audiocpp_c.h — a thin extern "C" shim over audio.cpp's C++ engine
 * (namespace engine::runtime). This is the ONLY surface exported by the
 * shared library; everything else (ggml, sentencepiece, engine::*) is hidden.
 *
 * Designed pointer-only (no by-value structs, no callbacks, no float value
 * args) so it can be bound from Go with purego, exactly like
 * purego binds a C shared library.
 */
#ifndef AUDIOCPP_C_H
#define AUDIOCPP_C_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  if defined(AUDIOCPP_BUILD)
#    define AUDIOCPP_API __declspec(dllexport)
#  else
#    define AUDIOCPP_API __declspec(dllimport)
#  endif
#else
#  define AUDIOCPP_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Coarse error codes. audio.cpp throws untyped std::exception with free-text
 * messages, so callers that need finer detail must read the `err` string.
 * The real value of this surface is that transport is gone: an engine failure
 * is a synchronous return code + message, not an HTTP status to re-parse. */
typedef enum {
    AUDIOCPP_OK = 0,
    AUDIOCPP_ERR_BAD_ARG = 1,
    AUDIOCPP_ERR_LOAD_FAILED = 2,
    AUDIOCPP_ERR_SYNTH_FAILED = 3,
    AUDIOCPP_ERR_DEVICE_QUERY_FAILED = 4,
} audiocpp_status;

/* Device kinds, mirroring ggml's ggml_backend_dev_type. */
typedef enum {
    AUDIOCPP_DEVICE_CPU = 0,   /* host CPU, system memory */
    AUDIOCPP_DEVICE_GPU = 1,   /* GPU with its own memory */
    AUDIOCPP_DEVICE_IGPU = 2,  /* integrated GPU sharing host memory */
    AUDIOCPP_DEVICE_ACCEL = 3, /* accelerator used alongside the CPU backend */
    AUDIOCPP_DEVICE_META = 4,  /* wrapper over several devices */
} audiocpp_device_type;

/* Opaque handle: owns a loaded ILoadedVoiceModel plus the backend/load config
 * needed to create per-synthesis sessions. */
typedef struct audiocpp_ctx audiocpp_ctx;

/*
 * Load a TTS model. Returns NULL on failure, writing a message into `err`
 * (bounded by `errlen`). `family_hint` selects the audio.cpp model family
 * (e.g. "qwen3_tts"); `backend` is one of "cpu"/"metal"/"cuda"/"vulkan"/
 * "hip"/"best". `load_options_json` is a flat JSON object of string values
 * (may be NULL/empty) merged into the load request options.
 */
AUDIOCPP_API audiocpp_ctx * audiocpp_load(const char * model_path,
                                          const char * family_hint,
                                          const char * backend,
                                          int device,
                                          int threads,
                                          const char * load_options_json,
                                          char * err,
                                          size_t errlen);

/*
 * Synthesise speech. Returns AUDIOCPP_OK (0) on success, else a nonzero
 * audiocpp_status with a message in `err`.
 *
 *   task        "tts" | "voice_design" | "voice_clone" (maps to VoiceTaskKind)
 *   text        UTF-8 text to speak (required)
 *   voice_id    preset speaker id (e.g. "ryan"); NULL when cloning from audio
 *   ref_pcm/ref_n/ref_sample_rate
 *               reference-audio voice (mono f32 PCM); pass NULL/0 when using voice_id
 *   options_json flat JSON object of string values (instruct, seed, temperature,
 *               reference_text, ...); may be NULL/empty
 *
 * On success the interleaved f32 PCM is returned via *out_samples (owned by the
 * caller — free with audiocpp_free_samples), with *out_n samples,
 * *out_sample_rate Hz, *out_channels channels.
 */
AUDIOCPP_API int audiocpp_synthesize(audiocpp_ctx * ctx,
                                     const char * task,
                                     const char * text,
                                     const char * voice_id,
                                     const float * ref_pcm,
                                     int ref_n,
                                     int ref_sample_rate,
                                     const char * options_json,
                                     float ** out_samples,
                                     int * out_n,
                                     int * out_sample_rate,
                                     int * out_channels,
                                     char * err,
                                     size_t errlen);

/* Free a sample buffer returned by audiocpp_synthesize (same-CRT free — safe
 * to call from any loader on any platform). */
AUDIOCPP_API void audiocpp_free_samples(float * samples);

/* Destroy a context (unloads the model). */
AUDIOCPP_API void audiocpp_free(audiocpp_ctx * ctx);

/* Shim version string (static storage; do not free). */
AUDIOCPP_API const char * audiocpp_version(void);

/*
 * Device registry (shim 0.2.0+). The library links ggml statically and hides
 * it, so these four calls are the only way to read the devices and device
 * memory THIS library sees. A device is addressed by its position in the
 * library's ggml device list, 0 .. audiocpp_device_count()-1; the list is fixed
 * for the life of the process. All four are stateless, need no context, and are
 * safe to call from any thread, including while a synthesis is running.
 *
 * The first call initialises the ggml backends if no model has done so yet
 * (the CUDA backend creates its context on every visible GPU at that point).
 */

/* Number of devices, or 0 when the registry cannot be read. */
AUDIOCPP_API int audiocpp_device_count(void);

/*
 * Static description of device `index`. Every out parameter is optional (NULL /
 * zero length skips it); strings are NUL-terminated and truncated to fit.
 *
 *   backend        ggml registry name: "CUDA", "Vulkan", "MTL", "ROCm", "CPU"
 *   backend_index  index within that backend, i.e. the `device` audiocpp_load
 *                  takes. For CUDA this is CUDA's own device ordinal, which is
 *                  NOT nvidia-smi's index.
 *   name           ggml device handle, e.g. "CUDA0"
 *   description    hardware name, e.g. "NVIDIA GeForce RTX 4090"
 *   id             lower-case PCI bus id "domain:bus:device.function" where the
 *                  backend knows it (CUDA, Vulkan), else ""
 *   type           an audiocpp_device_type
 *
 * Returns AUDIOCPP_OK, or AUDIOCPP_ERR_BAD_ARG for an index out of range.
 */
AUDIOCPP_API int audiocpp_device_info(int index,
                                      char * backend,
                                      size_t backend_len,
                                      int * backend_index,
                                      char * name,
                                      size_t name_len,
                                      char * description,
                                      size_t description_len,
                                      char * id,
                                      size_t id_len,
                                      int * type);

/*
 * Free and total memory of device `index`, in bytes, as the backend reports it
 * now (device-wide, not per process). Both are 0 when the backend has no
 * reading. Returns AUDIOCPP_OK, AUDIOCPP_ERR_BAD_ARG for a bad index or NULL
 * pointer, or AUDIOCPP_ERR_DEVICE_QUERY_FAILED.
 */
AUDIOCPP_API int audiocpp_device_memory(int index, uint64_t * free_bytes, uint64_t * total_bytes);

/*
 * The device a model loaded with (`backend`, `device`) runs on: the same
 * arguments audiocpp_load takes, resolved the way the engine resolves them.
 * Returns the device's index in the list above, or -1 when this library has no
 * such device (the load, or its first synthesis, would fail the same way).
 */
AUDIOCPP_API int audiocpp_resolve_device(const char * backend, int device);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AUDIOCPP_C_H
