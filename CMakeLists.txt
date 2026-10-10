cmake_minimum_required(VERSION 3.20)
project(audiocpp_go LANGUAGES C CXX)

# --- Where audio.cpp's source lives (a pinned checkout; see lib/version.txt) ---
# CI's clone-upstream.sh populates ./upstream; override AUDIOCPP_SRC for a local
# build against an existing checkout.
set(AUDIOCPP_SRC "${CMAKE_CURRENT_SOURCE_DIR}/upstream/audio.cpp"
    CACHE PATH "Path to the audio.cpp source tree")

# --- Backend selection (default CPU; a build ships one backend variant) ---
set(AUDIOCPP_BACKEND "cpu" CACHE STRING "cpu|metal|cuda|vulkan|hip")

# ---------------------------------------------------------------------------
# Export control — the load-bearing requirement.
# audio.cpp has no shared-lib product, no export macro, and force-caches
# BUILD_SHARED_LIBS OFF. To wrap its STATIC engine_runtime in a self-contained
# SHARED lib that leaks NONE of ggml/sentencepiece/absl/protobuf, we must set
# all of the following BEFORE add_subdirectory(audio.cpp):
# ---------------------------------------------------------------------------
set(CMAKE_POSITION_INDEPENDENT_CODE ON)                 # static-into-shared needs PIC everywhere
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
# Pin OFF: this option force-flips BUILD_SHARED_LIBS ON + GGML_BACKEND_DL ON
# (backends as separate .so files), which would break the one-self-contained-lib
# packaging.
set(ENGINE_ENABLE_CPU_ALL_VARIANTS OFF CACHE BOOL "" FORCE)

# Model set baked at configure time. The RELEASE build (build-libs.yml) passes
# -DAUDIOCPP_MODEL_SET=full; this local default compiles only qwen3_tts
# (+ the always-on VADs) to build fast. Deliberately NOT FORCE-d so a -D override
# on the configure line (CI) wins — these run before add_subdirectory(audio.cpp),
# so audio.cpp's own cache defaults never clobber the value chosen here.
set(AUDIOCPP_MODEL_SET "custom" CACHE STRING "audio.cpp model set: full|core|custom")
set(AUDIOCPP_MODELS "qwen3_tts" CACHE STRING "models when AUDIOCPP_MODEL_SET=custom")

# Embed the model-spec JSONs (model_specs/*.json) into the shim so the lib is
# self-contained. audio.cpp separates a family's compiled kernel from its
# runtime model-spec descriptor; without this the engine looks for
# model_specs/<family>.json on disk and aborts with "model spec not found for
# family 'qwen3_tts'" (there is no such directory beside the shipped lib). ON
# matches audio.cpp's own `build_*.sh --deployment` and the binding's
# one-self-contained-lib packaging. FORCE: every build this repo produces is a
# shipped lib, so it must always embed the specs.
set(AUDIOCPP_DEPLOYMENT_BUILD ON CACHE BOOL "" FORCE)

# CPU kernel tuning. ON compiles ggml's CPU kernels for the BUILD machine's CPU
# (-march=native / -mcpu=native), so the lib can hit an illegal instruction on a
# host whose CPU lacks a feature the build machine has. OFF builds the generic
# baseline for the target arch. Deliberately NOT FORCE-d so -DAUDIOCPP_NATIVE_CPU
# on the configure line wins; build-libs.yml passes OFF for the linux arm64 legs,
# whose archives run on other Arm cores (e.g. the DGX Spark's GB10) than the CI
# runner's.
set(AUDIOCPP_NATIVE_CPU ON CACHE BOOL "Tune ggml CPU kernels for the build machine's CPU")

# Backend + CPU kernel options, mirroring audio.cpp's own scripts/build_metal.sh.
set(ENGINE_ENABLE_CUDA OFF CACHE BOOL "" FORCE)
set(ENGINE_ENABLE_VULKAN OFF CACHE BOOL "" FORCE)
set(ENGINE_ENABLE_METAL OFF CACHE BOOL "" FORCE)
set(ENGINE_ENABLE_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(ENGINE_ENABLE_NATIVE_CPU ${AUDIOCPP_NATIVE_CPU} CACHE BOOL "" FORCE)
set(ENGINE_ENABLE_LLAMAFILE ON CACHE BOOL "" FORCE)
if(AUDIOCPP_BACKEND STREQUAL "metal")
  set(ENGINE_ENABLE_METAL ON CACHE BOOL "" FORCE)
  set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "" FORCE)
elseif(AUDIOCPP_BACKEND STREQUAL "cuda")
  set(ENGINE_ENABLE_CUDA ON CACHE BOOL "" FORCE)
elseif(AUDIOCPP_BACKEND STREQUAL "vulkan")
  set(ENGINE_ENABLE_VULKAN ON CACHE BOOL "" FORCE)
elseif(AUDIOCPP_BACKEND STREQUAL "hip")
  set(ENGINE_ENABLE_HIP ON CACHE BOOL "" FORCE)
endif()

add_subdirectory(${AUDIOCPP_SRC} audiocpp_build EXCLUDE_FROM_ALL)

# --- The shim: the only thing this repo ships as a linkable product ---
add_library(audiocpp SHARED shim/audiocpp_c.cpp)
target_compile_features(audiocpp PRIVATE cxx_std_17)
target_compile_definitions(audiocpp PRIVATE AUDIOCPP_BUILD)
target_include_directories(audiocpp
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/shim"
    "${AUDIOCPP_SRC}/include")
# engine_runtime is PUBLIC-linked to ggml; linking it PRIVATE pulls ggml in
# statically. Visibility + the link controls below keep those symbols hidden.
target_link_libraries(audiocpp PRIVATE engine_runtime)

set_target_properties(audiocpp PROPERTIES
  C_VISIBILITY_PRESET hidden
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
  OUTPUT_NAME audiocpp)

# Belt-and-braces symbol hiding of everything pulled from the static libs.
if(APPLE)
  target_link_options(audiocpp PRIVATE
    "-Wl,-exported_symbols_list,${CMAKE_CURRENT_SOURCE_DIR}/shim/exports_macos.txt")
elseif(UNIX)
  target_link_options(audiocpp PRIVATE
    "-Wl,--exclude-libs,ALL"
    "-Wl,--version-script,${CMAKE_CURRENT_SOURCE_DIR}/shim/exports_linux.map")
endif()
