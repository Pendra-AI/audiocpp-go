// audiocpp_c.cpp — implementation of the extern "C" shim over engine::runtime.
//
// Mirrors the call sequence proven by audio.cpp's own
// examples/xcode/.../MiniTTSDemoBridge.mm:
//   make_default_registry -> ModelRegistry::load -> create_task_session
//   -> prepare(build_preparation_request(req)) -> IOfflineVoiceTaskSession::run
//   -> TaskResult.audio_output (AudioBuffer{sample_rate, channels, vector<float>})
//
// The model is loaded once,
// but the *session* is created per synthesis and keyed on `task`, so preset
// TTS, VoiceDesign (instruct) and voice cloning share one loaded model without
// baking the task into an ABI.

#include "audiocpp_c.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/module.h"
#include "engine/framework/runtime/registry.h"
#include "engine/framework/runtime/session.h"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace {

void set_err(char * err, size_t errlen, const char * msg) {
    if (err == nullptr || errlen == 0) {
        return;
    }
    std::snprintf(err, errlen, "%s", msg != nullptr ? msg : "unknown error");
}

engine::core::BackendType parse_backend(const std::string & backend) {
    if (backend.empty() || backend == "cpu") return engine::core::BackendType::Cpu;
    if (backend == "metal") return engine::core::BackendType::Metal;
    if (backend == "cuda") return engine::core::BackendType::Cuda;
    if (backend == "hip" || backend == "rocm") return engine::core::BackendType::Hip;
    if (backend == "vulkan") return engine::core::BackendType::Vulkan;
    if (backend == "best") return engine::core::BackendType::BestAvailable;
    throw std::invalid_argument("unsupported backend: " + backend);
}

// Minimal, tolerant parser for a FLAT JSON object of scalar values:
//   {"seed":"1234","instruct":"Calm.","temperature":0.9}
// String values are unquoted (with basic \" \\ \n escapes); bare number/bool/
// null tokens are captured as their literal text. Nested objects/arrays are not
// supported (the shim's options are all scalar). Empty / NULL input -> no keys.
// Kept dependency-free on purpose so the shim never links audio.cpp's own
// (unstable, hidden) JSON internals.
void parse_flat_json(const char * json, std::unordered_map<std::string, std::string> & out) {
    if (json == nullptr) return;
    const std::string s(json);
    size_t i = 0;
    const size_t n = s.size();
    auto skip_ws = [&]() { while (i < n && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i; };
    auto parse_string = [&](std::string & dst) -> bool {
        if (i >= n || s[i] != '"') return false;
        ++i;
        while (i < n) {
            char c = s[i++];
            if (c == '\\' && i < n) {
                char e = s[i++];
                switch (e) {
                    case 'n': dst.push_back('\n'); break;
                    case 't': dst.push_back('\t'); break;
                    case 'r': dst.push_back('\r'); break;
                    case '"': dst.push_back('"'); break;
                    case '\\': dst.push_back('\\'); break;
                    case '/': dst.push_back('/'); break;
                    default: dst.push_back(e); break;
                }
            } else if (c == '"') {
                return true;
            } else {
                dst.push_back(c);
            }
        }
        return false;  // unterminated
    };
    skip_ws();
    if (i >= n || s[i] != '{') return;  // not an object -> ignore
    ++i;
    while (true) {
        skip_ws();
        if (i >= n || s[i] == '}') break;
        std::string key;
        if (!parse_string(key)) break;
        skip_ws();
        if (i >= n || s[i] != ':') break;
        ++i;
        skip_ws();
        std::string val;
        if (i < n && s[i] == '"') {
            if (!parse_string(val)) break;
        } else {
            // bare token: number / true / false / null
            size_t start = i;
            while (i < n && s[i] != ',' && s[i] != '}' && s[i] != ' ' && s[i] != '\t' &&
                   s[i] != '\n' && s[i] != '\r') {
                ++i;
            }
            val = s.substr(start, i - start);
        }
        out[key] = val;
        skip_ws();
        if (i < n && s[i] == ',') { ++i; continue; }
        break;
    }
}

}  // namespace

struct audiocpp_ctx {
    std::unique_ptr<engine::runtime::ILoadedVoiceModel> model;
    engine::core::BackendConfig backend;
    std::unordered_map<std::string, std::string> session_options;
};

extern "C" {

AUDIOCPP_API audiocpp_ctx * audiocpp_load(const char * model_path,
                                          const char * family_hint,
                                          const char * backend,
                                          int device,
                                          int threads,
                                          const char * load_options_json,
                                          char * err,
                                          size_t errlen) {
    if (model_path == nullptr || *model_path == '\0') {
        set_err(err, errlen, "model_path is required");
        return nullptr;
    }
    try {
        auto ctx = std::make_unique<audiocpp_ctx>();

        engine::runtime::ModelLoadRequest req;
        req.model_path = std::filesystem::path(model_path);
        if (family_hint != nullptr && *family_hint != '\0') {
            req.family_hint = std::string(family_hint);
        }
        parse_flat_json(load_options_json, req.options);

        ctx->backend.type = parse_backend(backend != nullptr ? backend : "cpu");
        ctx->backend.device = device;
        ctx->backend.threads = threads > 0 ? threads : 1;
        // Session options carried per-synthesis; seeded from load options so a
        // caller can set session-scoped knobs once at load if desired.
        parse_flat_json(load_options_json, ctx->session_options);

        auto registry = engine::runtime::make_default_registry();
        ctx->model = registry.load(req);
        if (!ctx->model) {
            set_err(err, errlen, "registry.load returned null");
            return nullptr;
        }
        return ctx.release();
    } catch (const std::exception & ex) {
        set_err(err, errlen, ex.what());
        return nullptr;
    } catch (...) {
        set_err(err, errlen, "unknown load failure");
        return nullptr;
    }
}

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
                                     size_t errlen) {
    if (ctx == nullptr || ctx->model == nullptr) {
        set_err(err, errlen, "null context");
        return AUDIOCPP_ERR_BAD_ARG;
    }
    if (text == nullptr || *text == '\0') {
        set_err(err, errlen, "text is required");
        return AUDIOCPP_ERR_BAD_ARG;
    }
    if (out_samples == nullptr || out_n == nullptr || out_sample_rate == nullptr || out_channels == nullptr) {
        set_err(err, errlen, "null output pointer");
        return AUDIOCPP_ERR_BAD_ARG;
    }
    *out_samples = nullptr;
    *out_n = 0;
    *out_sample_rate = 0;
    *out_channels = 0;

    try {
        engine::runtime::TaskSpec spec;
        spec.mode = engine::runtime::RunMode::Offline;
        spec.task = (task != nullptr && *task != '\0')
                        ? engine::runtime::parse_voice_task_kind(std::string(task))
                        : engine::runtime::VoiceTaskKind::Tts;

        engine::runtime::SessionOptions sopts;
        sopts.backend = ctx->backend;
        sopts.options = ctx->session_options;

        auto session = ctx->model->create_task_session(spec, sopts);
        auto * offline = dynamic_cast<engine::runtime::IOfflineVoiceTaskSession *>(session.get());
        if (offline == nullptr) {
            set_err(err, errlen, "model does not support offline synthesis for this task");
            return AUDIOCPP_ERR_SYNTH_FAILED;
        }

        engine::runtime::TaskRequest request;
        request.text_input = engine::runtime::Transcript{std::string(text), std::string()};

        const bool have_voice_id = voice_id != nullptr && *voice_id != '\0';
        const bool have_ref = ref_pcm != nullptr && ref_n > 0;
        if (have_voice_id || have_ref) {
            engine::runtime::VoiceReference speaker;
            if (have_voice_id) {
                speaker.cached_voice_id = std::string(voice_id);
            }
            if (have_ref) {
                engine::runtime::AudioBuffer ref;
                ref.sample_rate = ref_sample_rate;
                ref.channels = 1;
                ref.samples.assign(ref_pcm, ref_pcm + ref_n);
                speaker.audio = std::move(ref);
            }
            engine::runtime::VoiceCondition voice;
            voice.speaker = std::move(speaker);
            request.voice = std::move(voice);
        }

        parse_flat_json(options_json, request.options);

        session->prepare(engine::runtime::build_preparation_request(request));
        engine::runtime::TaskResult result = offline->run(request);
        if (!result.audio_output.has_value() || result.audio_output->samples.empty()) {
            set_err(err, errlen, "model returned no audio");
            return AUDIOCPP_ERR_SYNTH_FAILED;
        }

        const engine::runtime::AudioBuffer & audio = *result.audio_output;
        const size_t count = audio.samples.size();
        float * buf = static_cast<float *>(std::malloc(count * sizeof(float)));
        if (buf == nullptr) {
            set_err(err, errlen, "out of memory copying audio");
            return AUDIOCPP_ERR_SYNTH_FAILED;
        }
        std::memcpy(buf, audio.samples.data(), count * sizeof(float));
        *out_samples = buf;
        *out_n = static_cast<int>(count);
        *out_sample_rate = audio.sample_rate;
        *out_channels = audio.channels;
        return AUDIOCPP_OK;
    } catch (const std::invalid_argument & ex) {
        set_err(err, errlen, ex.what());
        return AUDIOCPP_ERR_BAD_ARG;
    } catch (const std::exception & ex) {
        set_err(err, errlen, ex.what());
        return AUDIOCPP_ERR_SYNTH_FAILED;
    } catch (...) {
        set_err(err, errlen, "unknown synthesis failure");
        return AUDIOCPP_ERR_SYNTH_FAILED;
    }
}

AUDIOCPP_API void audiocpp_free_samples(float * samples) {
    std::free(samples);
}

AUDIOCPP_API void audiocpp_free(audiocpp_ctx * ctx) {
    delete ctx;
}

AUDIOCPP_API const char * audiocpp_version(void) {
    return "audiocpp-go-shim 0.1.0 (audio.cpp pin 545e29a6)";
}

}  // extern "C"
