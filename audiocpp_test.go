package audiocpp

import (
	"os"
	"testing"
)

// TestSynthesizeE2E exercises the full Go -> shim -> engine path. It is gated
// behind AUDIOCPP_GO_E2E=1 because it needs the built shared lib and a real
// GGUF, so `go test` is hermetic by default.
//
// Env:
//
//	AUDIOCPP_GO_E2E=1              enable
//	AUDIOCPP_GO_LIB_DIR=<dir>      dir holding libaudiocpp.dylib (default ./build)
//	AUDIOCPP_GO_MODEL=<gguf>       model path (default ~/.pendra-tts-demo/models/qwen3-tts.gguf)
//	AUDIOCPP_GO_BACKEND=<name>     backend (default metal)
func TestSynthesizeE2E(t *testing.T) {
	if os.Getenv("AUDIOCPP_GO_E2E") != "1" {
		t.Skip("set AUDIOCPP_GO_E2E=1 (and build the lib) to run the real-engine smoke test")
	}

	libDir := envOr("AUDIOCPP_GO_LIB_DIR", "build")
	if err := Load(libDir); err != nil {
		t.Fatalf("Load(%q): %v", libDir, err)
	}
	t.Logf("shim version: %s", Version())

	home, _ := os.UserHomeDir()
	model := envOr("AUDIOCPP_GO_MODEL", home+"/.pendra-tts-demo/models/qwen3-tts.gguf")
	m, err := New(ModelParams{
		ModelPath:  model,
		FamilyHint: "qwen3_tts",
		Backend:    envOr("AUDIOCPP_GO_BACKEND", "metal"),
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer m.Close()

	audio, err := m.Synthesize(SynthParams{
		Text:    "Binding audio dot cpp from Go with purego.",
		VoiceID: "ryan",
	})
	if err != nil {
		t.Fatalf("Synthesize: %v", err)
	}
	if len(audio.Samples) == 0 {
		t.Fatal("no samples returned")
	}
	if audio.SampleRate != 24000 {
		t.Errorf("SampleRate = %d, want 24000", audio.SampleRate)
	}
	t.Logf("synthesised %d samples, %d Hz, %d ch (%.2fs)",
		len(audio.Samples), audio.SampleRate, audio.Channels,
		float64(len(audio.Samples))/float64(audio.SampleRate*maxInt(audio.Channels, 1)))
}

func envOr(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
