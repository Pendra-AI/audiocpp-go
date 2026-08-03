package audiocpp

import (
	"encoding/json"
	"errors"
	"fmt"
	"sync"
	"unsafe"
)

// Error is a synthesis or load failure from the engine. Code is the coarse shim
// status (see the shim's audiocpp_status); Msg is the engine's free-text
// message. The whole point of the in-process binding over the old HTTP path is
// that this arrives synchronously — no status code to re-parse off a response.
type Error struct {
	Code int
	Msg  string
}

func (e *Error) Error() string {
	if e.Msg == "" {
		return fmt.Sprintf("audiocpp: engine error (code %d)", e.Code)
	}
	return fmt.Sprintf("audiocpp: %s (code %d)", e.Msg, e.Code)
}

// ModelParams configures a model load.
type ModelParams struct {
	ModelPath   string            // path to the GGUF (required)
	FamilyHint  string            // audio.cpp family, e.g. "qwen3_tts"
	Backend     string            // "cpu"|"metal"|"cuda"|"vulkan"|"hip"|"best" ("" == cpu)
	Device      int               // GPU device index
	Threads     int               // CPU threads (<=0 -> engine default)
	LoadOptions map[string]string // extra load-time options
}

// SynthParams configures one synthesis call.
type SynthParams struct {
	Task          string            // "tts" (default) | "voice_design" | "voice_clone"
	Text          string            // text to speak (required)
	VoiceID       string            // preset speaker id, e.g. "ryan"
	RefPCM        []float32         // reference-audio voice (mono f32); alternative to VoiceID
	RefSampleRate int               // sample rate of RefPCM
	Options       map[string]string // per-request options (instruct, seed, temperature, ...)
}

// Audio is a synthesis result: interleaved f32 PCM plus format.
type Audio struct {
	Samples    []float32
	SampleRate int
	Channels   int
}

// Model is a loaded TTS model. It is NOT safe for concurrent use — synthesise
// serially — hold one model behind a single-slot mutex. An
// internal mutex is kept only to prevent a use-after-free race between
// Synthesize and Close, not to enable parallelism.
type Model struct {
	mu  sync.Mutex
	ctx uintptr // audiocpp_ctx*; 0 after Close
}

// New loads a model. The returned Model must be Close()d to free native memory.
func New(p ModelParams) (m *Model, err error) {
	if !loaded {
		return nil, errors.New("audiocpp: library not loaded — call Load first")
	}
	if p.ModelPath == "" {
		return nil, &Error{Code: statusBadArg, Msg: "model path is required"}
	}
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("audiocpp: load panicked across FFI: %v", r)
		}
	}()

	errBuf := make([]byte, errBufLen)
	ctx := cLoad(
		p.ModelPath, p.FamilyHint, p.Backend,
		int32(p.Device), int32(p.Threads), flatJSON(p.LoadOptions),
		&errBuf[0], uintptr(len(errBuf)),
	)
	if ctx == 0 {
		return nil, &Error{Code: statusLoadFailed, Msg: cBufString(errBuf)}
	}
	return &Model{ctx: ctx}, nil
}

// Synthesize turns text into audio. Safe against a concurrent Close (returns an
// error if the model is closed); concurrent Synthesize calls are serialized.
func (m *Model) Synthesize(p SynthParams) (a *Audio, err error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.ctx == 0 {
		return nil, errors.New("audiocpp: model is closed")
	}
	if p.Text == "" {
		return nil, &Error{Code: statusBadArg, Msg: "text is required"}
	}
	task := p.Task
	if task == "" {
		task = "tts"
	}

	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("audiocpp: synthesize panicked across FFI: %v", r)
		}
	}()

	var refPtr *float32
	var refN int32
	if len(p.RefPCM) > 0 {
		refPtr = &p.RefPCM[0]
		refN = int32(len(p.RefPCM))
	}

	var (
		outSamples                       *float32
		outN, outSampleRate, outChannels int32
	)
	errBuf := make([]byte, errBufLen)

	status := cSynthesize(
		m.ctx, task, p.Text, p.VoiceID,
		refPtr, refN, int32(p.RefSampleRate), flatJSON(p.Options),
		&outSamples, &outN, &outSampleRate, &outChannels,
		&errBuf[0], uintptr(len(errBuf)),
	)
	if status != statusOK {
		return nil, &Error{Code: int(status), Msg: cBufString(errBuf)}
	}
	if outSamples == nil || outN <= 0 {
		return nil, &Error{Code: statusSynthFailed, Msg: "engine returned no audio"}
	}

	// Copy the native buffer into Go-owned memory, then free the native side —
	// never hand the caller a pointer the C library owns.
	src := unsafe.Slice(outSamples, int(outN))
	samples := make([]float32, int(outN))
	copy(samples, src)
	cFreeSamples(outSamples)

	return &Audio{
		Samples:    samples,
		SampleRate: int(outSampleRate),
		Channels:   int(outChannels),
	}, nil
}

// Close unloads the model and frees native memory. Idempotent.
func (m *Model) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.ctx == 0 {
		return nil
	}
	cFree(m.ctx)
	m.ctx = 0
	return nil
}

// Version returns the shim's version string. Requires Load to have succeeded.
func Version() string {
	if !loaded {
		return ""
	}
	return cBytePtrString(cVersion())
}

// flatJSON marshals a string map to the flat JSON object the shim parses
// ({"k":"v",...}); an empty/nil map yields "" (the shim treats it as no options).
func flatJSON(m map[string]string) string {
	if len(m) == 0 {
		return ""
	}
	b, err := json.Marshal(m)
	if err != nil {
		return ""
	}
	return string(b)
}

// cBufString reads a NUL-terminated string out of a fixed buffer the shim wrote
// into (used for the err out-parameter).
func cBufString(buf []byte) string {
	for i, c := range buf {
		if c == 0 {
			return string(buf[:i])
		}
	}
	return string(buf)
}

// cBytePtrString reads a NUL-terminated C string returned as a *byte.
func cBytePtrString(p *byte) string {
	if p == nil {
		return ""
	}
	var n int
	for q := p; *q != 0; q = (*byte)(unsafe.Add(unsafe.Pointer(q), 1)) {
		n++
	}
	return string(unsafe.Slice(p, n))
}
