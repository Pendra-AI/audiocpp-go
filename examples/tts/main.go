// Command tts is a minimal end-to-end example: load a model and synthesise a
// WAV, exercising the Go -> shim -> engine path end to end.
//
// Usage:
//
//	go run ./examples/tts \
//	  -lib build \
//	  -model ~/.pendra-tts-demo/models/qwen3-tts.gguf \
//	  -family qwen3_tts -backend metal -voice ryan \
//	  -text "Hello from audio dot cpp." -out out.wav
package main

import (
	"flag"
	"fmt"
	"os"

	audiocpp "github.com/Pendra-Cloud/audiocpp-go"
)

func main() {
	lib := flag.String("lib", "build", "dir holding the audiocpp shared library")
	model := flag.String("model", "", "path to the GGUF model (required)")
	family := flag.String("family", "qwen3_tts", "audio.cpp model family")
	backend := flag.String("backend", "metal", "backend: cpu|metal|cuda|vulkan|hip|best")
	voice := flag.String("voice", "ryan", "preset voice id")
	text := flag.String("text", "Hello from audio dot cpp, bound in Go.", "text to speak")
	out := flag.String("out", "out.wav", "output WAV path")
	flag.Parse()

	if *model == "" {
		fmt.Fprintln(os.Stderr, "error: -model is required")
		os.Exit(2)
	}

	if err := audiocpp.Load(*lib); err != nil {
		fatal(err)
	}
	fmt.Println("shim:", audiocpp.Version())

	m, err := audiocpp.New(audiocpp.ModelParams{
		ModelPath:  *model,
		FamilyHint: *family,
		Backend:    *backend,
	})
	if err != nil {
		fatal(err)
	}
	defer m.Close()

	audio, err := m.Synthesize(audiocpp.SynthParams{Text: *text, VoiceID: *voice})
	if err != nil {
		fatal(err)
	}

	if err := os.WriteFile(*out, audiocpp.EncodeWAV(audio), 0o644); err != nil {
		fatal(err)
	}
	fmt.Printf("wrote %s — %d samples, %d Hz, %d ch\n",
		*out, len(audio.Samples), audio.SampleRate, audio.Channels)
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "error:", err)
	os.Exit(1)
}
