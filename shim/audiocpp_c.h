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
} audiocpp_status;

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

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AUDIOCPP_C_H
