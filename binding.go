// Package audiocpp is a Go binding for audio.cpp's TTS engine, loaded in-process
// via purego (no cgo) over the extern "C" shim in shim/audiocpp_c.h. It mirrors
// a common purego binding approach.
//
// The surface is deliberately small (five C symbols), so — unlike sd-go, which
// splits a ~50-symbol API into pkg/sd + a root layer — the raw bindings and the
// ergonomic API live in one package here.
package audiocpp

import (
	"fmt"
	"path/filepath"
	"runtime"
	"sync"

	"github.com/ebitengine/purego"
)

// Status codes returned by audiocpp_synthesize (mirrors audiocpp_status in the
// shim). Load failure is signalled by a NULL context rather than a status.
const (
	statusOK          = 0
	statusBadArg      = 1
	statusLoadFailed  = 2
	statusSynthFailed = 3
)

// errBufLen is the size of the caller-provided buffer the shim snprintf's its
// error message into.
const errBufLen = 512

// Raw C symbol bindings, populated by Load. All C-string IN parameters are typed
// `string` so purego marshals them to null-terminated char* for the call; the
// shim treats an empty string the same as NULL for optional args.
var (
	cLoad func(modelPath, familyHint, backend string, device, threads int32,
		loadOptionsJSON string, err *byte, errlen uintptr) uintptr // returns audiocpp_ctx*

	cSynthesize func(ctx uintptr, task, text, voiceID string,
		refPCM *float32, refN, refSampleRate int32, optionsJSON string,
		outSamples **float32, outN, outSampleRate, outChannels *int32,
		err *byte, errlen uintptr) int32

	cFreeSamples func(samples *float32)
	cFree        func(ctx uintptr)
	cVersion     func() *byte
)

// Dynamic library handle, populated by Load.
var libHandle uintptr

var (
	loadMu sync.Mutex
	loaded bool
)

// libFileName is the platform-specific shared library filename.
func libFileName() string {
	switch runtime.GOOS {
	case "windows":
		return "audiocpp.dll"
	case "darwin":
		return "libaudiocpp.dylib"
	default:
		return "libaudiocpp.so"
	}
}

// libCandidates is the ordered list of paths to try when opening the library
// from libDir. An empty libDir yields a bare filename so the OS loader search
// path is used.
func libCandidates(libDir string) []string {
	name := libFileName()
	if libDir == "" {
		return []string{name}
	}
	return []string{filepath.Join(libDir, name)}
}

// Load resolves and dlopens the audiocpp shared library from libDir (an empty
// libDir falls back to the OS default search path). It is idempotent, safe for
// concurrent use, and returns an error — never panics — when the library is
// absent or a symbol is missing, so importing this package or calling Load with
// no library present keeps the caller healthy.
func Load(libDir string) (err error) {
	loadMu.Lock()
	defer loadMu.Unlock()

	if loaded {
		return nil
	}

	// purego.RegisterLibFunc panics when a symbol is missing; convert any panic
	// across the FFI boundary into an error and release the handle so a partial
	// registration never leaks (or locks the DLL on Windows).
	defer func() {
		if r := recover(); r != nil {
			if libHandle != 0 {
				_ = closeLibrary(libHandle)
			}
			libHandle = 0
			err = fmt.Errorf("audiocpp: failed to load library: %v", r)
		}
	}()

	var (
		handle  uintptr
		lastErr error
	)
	for _, path := range libCandidates(libDir) {
		handle, lastErr = openLibrary(path)
		if lastErr == nil && handle != 0 {
			break
		}
	}
	if handle == 0 {
		if lastErr == nil {
			lastErr = fmt.Errorf("no library candidates for GOOS %q", runtime.GOOS)
		}
		return fmt.Errorf("audiocpp: failed to load library from %q: %w", libDir, lastErr)
	}

	libHandle = handle
	registerFunctions()
	loaded = true
	return nil
}

// registerFunctions binds the five C symbols. It panics (via RegisterLibFunc) on
// a missing symbol; Load recovers.
func registerFunctions() {
	purego.RegisterLibFunc(&cLoad, libHandle, "audiocpp_load")
	purego.RegisterLibFunc(&cSynthesize, libHandle, "audiocpp_synthesize")
	purego.RegisterLibFunc(&cFreeSamples, libHandle, "audiocpp_free_samples")
	purego.RegisterLibFunc(&cFree, libHandle, "audiocpp_free")
	purego.RegisterLibFunc(&cVersion, libHandle, "audiocpp_version")
}
