package audiocpp

import (
	"bytes"
	"encoding/binary"
	"math"
)

// EncodeWAV encodes an Audio result as a 16-bit PCM WAV file (little-endian).
// A convenience for callers/examples; the worker ships base64 WAV to the API.
func EncodeWAV(a *Audio) []byte {
	sampleRate := a.SampleRate
	if sampleRate <= 0 {
		sampleRate = 24000
	}
	channels := a.Channels
	if channels <= 0 {
		channels = 1
	}

	dataLen := len(a.Samples) * 2 // 16-bit
	byteRate := sampleRate * channels * 2
	blockAlign := channels * 2

	var b bytes.Buffer
	b.WriteString("RIFF")
	binary.Write(&b, binary.LittleEndian, uint32(36+dataLen))
	b.WriteString("WAVE")
	b.WriteString("fmt ")
	binary.Write(&b, binary.LittleEndian, uint32(16))         // PCM fmt chunk size
	binary.Write(&b, binary.LittleEndian, uint16(1))          // PCM
	binary.Write(&b, binary.LittleEndian, uint16(channels))   //nolint:gosec
	binary.Write(&b, binary.LittleEndian, uint32(sampleRate)) //nolint:gosec
	binary.Write(&b, binary.LittleEndian, uint32(byteRate))   //nolint:gosec
	binary.Write(&b, binary.LittleEndian, uint16(blockAlign)) //nolint:gosec
	binary.Write(&b, binary.LittleEndian, uint16(16))         // bits per sample
	b.WriteString("data")
	binary.Write(&b, binary.LittleEndian, uint32(dataLen)) //nolint:gosec

	for _, s := range a.Samples {
		v := s
		if v > 1 {
			v = 1
		} else if v < -1 {
			v = -1
		}
		binary.Write(&b, binary.LittleEndian, int16(math.Round(float64(v)*32767)))
	}
	return b.Bytes()
}
