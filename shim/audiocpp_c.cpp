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

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
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

// --- Device registry -------------------------------------------------------

// Same guard the engine's init_backend runs before it touches the registry.
void ensure_backends_loaded() {
    if (ggml_backend_reg_count() == 0) {
        ggml_backend_load_all();
    }
}

ggml_backend_dev_t device_at(int index) {
    ensure_backends_loaded();
    if (index < 0 || static_cast<size_t>(index) >= ggml_backend_dev_count()) {
        return nullptr;
    }
    return ggml_backend_dev_get(static_cast<size_t>(index));
}

int device_index_of(ggml_backend_dev_t device) {
    if (device == nullptr) {
        return -1;
    }
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        if (ggml_backend_dev_get(i) == device) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

void copy_str(char * dst, size_t dstlen, const char * src) {
    if (dst == nullptr || dstlen == 0) {
        return;
    }
    std::snprintf(dst, dstlen, "%s", src != nullptr ? src : "");
}

// The ggml registry names behind each engine backend type. Mirrors
// k_backend_reg_names in audio.cpp's src/framework/core/backend.cpp, which is
// file-local there; keep the two in step when moving the pin.
bool reg_name_matches(engine::core::BackendType type, const char * reg_name) {
    if (reg_name == nullptr) {
        return false;
    }
    const std::string name(reg_name);
    switch (type) {
        case engine::core::BackendType::Cuda: return name == "CUDA" || name == "MUSA";
        case engine::core::BackendType::Hip: return name == "ROCm";
        case engine::core::BackendType::Vulkan: return name == "Vulkan";
        case engine::core::BackendType::Metal: return name == "MTL";
        default: return false;
    }
}

// Resolves a backend config to the device init_backend would open for it.
ggml_backend_dev_t resolve_device(const engine::core::BackendConfig & config) {
    ensure_backends_loaded();
    switch (config.type) {
        case engine::core::BackendType::Cpu:
            return ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        case engine::core::BackendType::BestAvailable: {
            // ggml_backend_init_best's own order.
            ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
            dev = dev != nullptr ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
            return dev != nullptr ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        }
        default:
            break;
    }
    if (config.device < 0) {
        return nullptr;
    }
    for (size_t i = 0; i < ggml_backend_reg_count(); ++i) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(i);
        if (reg == nullptr || !reg_name_matches(config.type, ggml_backend_reg_name(reg))) {
            continue;
        }
        if (static_cast<size_t>(config.device) >= ggml_backend_reg_dev_count(reg)) {
            return nullptr;
        }
        return ggml_backend_reg_dev_get(reg, static_cast<size_t>(config.device));
    }
    return nullptr;
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
    return "audiocpp-go-shim 0.2.0 (audio.cpp pin 545e29a6)";
}

AUDIOCPP_API int audiocpp_device_count(void) {
    try {
        ensure_backends_loaded();
        return static_cast<int>(ggml_backend_dev_count());
    } catch (...) {
        return 0;
    }
}

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
                                      int * type) {
    try {
        ggml_backend_dev_t device = device_at(index);
        if (device == nullptr) {
            return AUDIOCPP_ERR_BAD_ARG;
        }
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);
        copy_str(backend, backend_len, reg != nullptr ? ggml_backend_reg_name(reg) : "");
        if (backend_index != nullptr) {
            *backend_index = -1;
            for (size_t i = 0; reg != nullptr && i < ggml_backend_reg_dev_count(reg); ++i) {
                if (ggml_backend_reg_dev_get(reg, i) == device) {
                    *backend_index = static_cast<int>(i);
                    break;
                }
            }
        }
        copy_str(name, name_len, ggml_backend_dev_name(device));
        copy_str(description, description_len, ggml_backend_dev_description(device));
        if (id != nullptr && id_len > 0) {
            // The id is only reachable through the props; zeroed first because
            // backends without one (Metal, CPU) leave the field untouched.
            ggml_backend_dev_props props = {};
            ggml_backend_dev_get_props(device, &props);
            copy_str(id, id_len, props.device_id);
        }
        if (type != nullptr) {
            *type = static_cast<int>(ggml_backend_dev_type(device));
        }
        return AUDIOCPP_OK;
    } catch (...) {
        return AUDIOCPP_ERR_DEVICE_QUERY_FAILED;
    }
}

AUDIOCPP_API int audiocpp_device_memory(int index, uint64_t * free_bytes, uint64_t * total_bytes) {
    if (free_bytes == nullptr || total_bytes == nullptr) {
        return AUDIOCPP_ERR_BAD_ARG;
    }
    *free_bytes = 0;
    *total_bytes = 0;
    try {
        ggml_backend_dev_t device = device_at(index);
        if (device == nullptr) {
            return AUDIOCPP_ERR_BAD_ARG;
        }
        size_t free = 0;
        size_t total = 0;
        ggml_backend_dev_memory(device, &free, &total);
        *free_bytes = static_cast<uint64_t>(free);
        *total_bytes = static_cast<uint64_t>(total);
        return AUDIOCPP_OK;
    } catch (...) {
        return AUDIOCPP_ERR_DEVICE_QUERY_FAILED;
    }
}

AUDIOCPP_API int audiocpp_resolve_device(const char * backend, int device) {
    try {
        engine::core::BackendConfig config;
        config.type = parse_backend(backend != nullptr ? backend : "cpu");
        config.device = device;
        return device_index_of(resolve_device(config));
    } catch (...) {
        return -1;
    }
}

}  // extern "C"
