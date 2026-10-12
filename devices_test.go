package audiocpp

import (
	"errors"
	"os"
	"sync"
	"testing"
	"unsafe"
)

// fakeDevice is one entry of the fake registry installed by fakeDeviceAPI.
type fakeDevice struct {
	backend, name, description, id string
	deviceType                     DeviceType
	free, total                    uint64
}

// fakeDeviceAPI swaps the device bindings for an in-memory registry shaped like
// the shim's (global list order, per-backend indices) and marks the library
// loaded. Everything is restored when the test ends.
func fakeDeviceAPI(t *testing.T, devs []fakeDevice) {
	t.Helper()
	prevLoaded, prevAPI := loaded, deviceAPI
	prevCount, prevInfo, prevMem, prevResolve := cDeviceCount, cDeviceInfo, cDeviceMemory, cResolveDevice
	t.Cleanup(func() {
		loaded, deviceAPI = prevLoaded, prevAPI
		cDeviceCount, cDeviceInfo, cDeviceMemory, cResolveDevice = prevCount, prevInfo, prevMem, prevResolve
	})

	put := func(dst *byte, n uintptr, s string) {
		if dst == nil || n == 0 {
			return
		}
		buf := unsafe.Slice(dst, int(n))
		c := copy(buf[:len(buf)-1], s)
		buf[c] = 0
	}
	backendIndex := func(index int) int32 {
		var n int32
		for i := 0; i < index; i++ {
			if devs[i].backend == devs[index].backend {
				n++
			}
		}
		return n
	}

	cDeviceCount = func() int32 { return int32(len(devs)) }
	cDeviceInfo = func(index int32, backend *byte, backendLen uintptr, bi *int32,
		name *byte, nameLen uintptr, description *byte, descriptionLen uintptr,
		id *byte, idLen uintptr, deviceType *int32) int32 {
		if index < 0 || int(index) >= len(devs) {
			return statusBadArg
		}
		d := devs[index]
		put(backend, backendLen, d.backend)
		put(name, nameLen, d.name)
		put(description, descriptionLen, d.description)
		put(id, idLen, d.id)
		*bi = backendIndex(int(index))
		*deviceType = int32(d.deviceType)
		return statusOK
	}
	cDeviceMemory = func(index int32, free, total *uint64) int32 {
		if index < 0 || int(index) >= len(devs) {
			return statusBadArg
		}
		*free, *total = devs[index].free, devs[index].total
		return statusOK
	}
	cResolveDevice = func(backend string, device int32) int32 {
		want := map[string]string{"": "CPU", "cpu": "CPU", "cuda": "CUDA", "vulkan": "Vulkan", "metal": "MTL"}[backend]
		if want == "CPU" {
			device = 0
		}
		for i := range devs {
			if devs[i].backend == want && backendIndex(i) == device {
				return int32(i)
			}
		}
		return -1
	}
	loaded, deviceAPI = true, true
}

// twoGPUHost is a two-card CUDA host as the library sees it: CUDA orders the
// faster card first, so CUDA0 is the card on the higher PCI bus.
func twoGPUHost() []fakeDevice {
	return []fakeDevice{
		{backend: "CUDA", name: "CUDA0", description: "NVIDIA GeForce RTX 4090", id: "0000:41:00.0", deviceType: DeviceTypeGPU, free: 20 << 30, total: 24 << 30},
		{backend: "CUDA", name: "CUDA1", description: "NVIDIA GeForce RTX 3060", id: "0000:01:00.0", deviceType: DeviceTypeGPU, free: 11 << 30, total: 12 << 30},
		{backend: "CPU", name: "CPU", description: "AMD Ryzen 9", deviceType: DeviceTypeCPU, free: 64 << 30, total: 64 << 30},
	}
}

func TestDevices_ListsTheRegistryInOrder(t *testing.T) {
	fakeDeviceAPI(t, twoGPUHost())

	if !HasDeviceAPI() {
		t.Fatal("HasDeviceAPI() = false with the device symbols bound")
	}
	got, err := Devices()
	if err != nil {
		t.Fatalf("Devices: %v", err)
	}
	want := []Device{
		{Index: 0, Backend: "CUDA", BackendIndex: 0, Name: "CUDA0", Description: "NVIDIA GeForce RTX 4090", ID: "0000:41:00.0", Type: DeviceTypeGPU},
		{Index: 1, Backend: "CUDA", BackendIndex: 1, Name: "CUDA1", Description: "NVIDIA GeForce RTX 3060", ID: "0000:01:00.0", Type: DeviceTypeGPU},
		{Index: 2, Backend: "CPU", BackendIndex: 0, Name: "CPU", Description: "AMD Ryzen 9", ID: "", Type: DeviceTypeCPU},
	}
	if len(got) != len(want) {
		t.Fatalf("Devices() returned %d devices, want %d: %+v", len(got), len(want), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("Devices()[%d] = %+v, want %+v", i, got[i], want[i])
		}
	}
}

func TestDeviceMemory_ReadsTheIndexedDevice(t *testing.T) {
	fakeDeviceAPI(t, twoGPUHost())

	mem, err := DeviceMemory(1)
	if err != nil {
		t.Fatalf("DeviceMemory(1): %v", err)
	}
	if want := (Memory{FreeBytes: 11 << 30, TotalBytes: 12 << 30}); mem != want {
		t.Errorf("DeviceMemory(1) = %+v, want %+v", mem, want)
	}

	for _, index := range []int{-1, 3} {
		_, err := DeviceMemory(index)
		var e *Error
		if !errors.As(err, &e) || e.Code != statusBadArg {
			t.Errorf("DeviceMemory(%d) error = %v, want a bad-argument *Error", index, err)
		}
	}
}

// The point of the API: a model on CUDA device 1 is attributed to the card CUDA
// calls 1, whatever any other tool numbers it.
func TestModelDevice_ResolvesTheLoadedBackendAndDevice(t *testing.T) {
	fakeDeviceAPI(t, twoGPUHost())

	tests := []struct {
		backend string
		device  int
		want    string
	}{
		{"cuda", 0, "CUDA0"},
		{"cuda", 1, "CUDA1"},
		{"cpu", 0, "CPU"},
		{"", 0, "CPU"},
	}
	for _, tc := range tests {
		m := &Model{backend: tc.backend, device: tc.device}
		got, err := m.Device()
		if err != nil {
			t.Errorf("Model{%q,%d}.Device(): %v", tc.backend, tc.device, err)
			continue
		}
		if got.Name != tc.want {
			t.Errorf("Model{%q,%d}.Device().Name = %q, want %q", tc.backend, tc.device, got.Name, tc.want)
		}
		if tc.backend == "cuda" && got.BackendIndex != tc.device {
			t.Errorf("Model{%q,%d}.Device().BackendIndex = %d", tc.backend, tc.device, got.BackendIndex)
		}
	}
}

func TestResolveDevice_UnknownDeviceIsABadArgument(t *testing.T) {
	fakeDeviceAPI(t, twoGPUHost())

	for _, tc := range []struct {
		backend string
		device  int
	}{{"cuda", 2}, {"vulkan", 0}, {"metal", 0}} {
		_, err := ResolveDevice(tc.backend, tc.device)
		var e *Error
		if !errors.As(err, &e) || e.Code != statusBadArg {
			t.Errorf("ResolveDevice(%q, %d) error = %v, want a bad-argument *Error", tc.backend, tc.device, err)
		}
	}
}

// A library built before the device registry still loads; the device calls
// report ErrDeviceAPIUnavailable instead of failing the load or panicking.
func TestDeviceAPI_UnavailableOnAnOlderLibrary(t *testing.T) {
	fakeDeviceAPI(t, nil)
	deviceAPI = false
	cDeviceCount, cDeviceInfo, cDeviceMemory, cResolveDevice = nil, nil, nil, nil

	if HasDeviceAPI() {
		t.Error("HasDeviceAPI() = true without the device symbols")
	}
	if _, err := Devices(); !errors.Is(err, ErrDeviceAPIUnavailable) {
		t.Errorf("Devices() error = %v, want ErrDeviceAPIUnavailable", err)
	}
	if _, err := DeviceMemory(0); !errors.Is(err, ErrDeviceAPIUnavailable) {
		t.Errorf("DeviceMemory() error = %v, want ErrDeviceAPIUnavailable", err)
	}
	if _, err := ResolveDevice("cuda", 0); !errors.Is(err, ErrDeviceAPIUnavailable) {
		t.Errorf("ResolveDevice() error = %v, want ErrDeviceAPIUnavailable", err)
	}
	if _, err := (&Model{backend: "cuda"}).Device(); !errors.Is(err, ErrDeviceAPIUnavailable) {
		t.Errorf("Model.Device() error = %v, want ErrDeviceAPIUnavailable", err)
	}
}

func TestDeviceAPI_UnavailableBeforeLoad(t *testing.T) {
	fakeDeviceAPI(t, twoGPUHost())
	loaded = false

	if HasDeviceAPI() {
		t.Error("HasDeviceAPI() = true before Load")
	}
	if _, err := Devices(); !errors.Is(err, ErrDeviceAPIUnavailable) {
		t.Errorf("Devices() error = %v, want ErrDeviceAPIUnavailable", err)
	}
}

func TestDevices_QueryFailureIsAnError(t *testing.T) {
	fakeDeviceAPI(t, twoGPUHost())
	cDeviceMemory = func(int32, *uint64, *uint64) int32 { return statusDeviceQuery }
	cDeviceInfo = func(int32, *byte, uintptr, *int32, *byte, uintptr, *byte, uintptr, *byte, uintptr, *int32) int32 {
		return statusDeviceQuery
	}

	var e *Error
	if _, err := DeviceMemory(0); !errors.As(err, &e) || e.Code != statusDeviceQuery {
		t.Errorf("DeviceMemory() error = %v, want a device-query *Error", err)
	}
	if devs, err := Devices(); !errors.As(err, &e) || devs != nil {
		t.Errorf("Devices() = %v, %v; want nil and a device-query *Error", devs, err)
	}
}

// Device readings are taken while a synthesis runs on another goroutine, so
// they must not share state. Run with -race.
func TestDeviceAPI_ConcurrentReads(t *testing.T) {
	fakeDeviceAPI(t, twoGPUHost())
	m := &Model{backend: "cuda", device: 1}

	var wg sync.WaitGroup
	for i := 0; i < 16; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 50; j++ {
				if devs, err := Devices(); err != nil || len(devs) != 3 {
					t.Errorf("Devices() = %v, %v", devs, err)
					return
				}
				if _, err := DeviceMemory(j % 3); err != nil {
					t.Errorf("DeviceMemory: %v", err)
					return
				}
				if d, err := m.Device(); err != nil || d.Name != "CUDA1" {
					t.Errorf("Model.Device() = %+v, %v", d, err)
					return
				}
			}
		}()
	}
	wg.Wait()
}

func TestDeviceType_String(t *testing.T) {
	for typ, want := range map[DeviceType]string{
		DeviceTypeCPU: "cpu", DeviceTypeGPU: "gpu", DeviceTypeIGPU: "igpu",
		DeviceTypeAccel: "accel", DeviceTypeMeta: "meta", DeviceType(9): "unknown(9)",
	} {
		if got := typ.String(); got != want {
			t.Errorf("DeviceType(%d).String() = %q, want %q", int(typ), got, want)
		}
	}
}

// TestDeviceRegistryIntegration reads the device registry of a real built
// library. It needs no model, so build-libs.yml runs it on every leg whose
// library can be opened on the runner (the CUDA legs cannot: no driver).
//
// Env:
//
//	AUDIOCPP_GO_DEVICES=1              enable
//	AUDIOCPP_GO_LIB_DIR=<dir>          dir holding the built library (default ./build)
//	AUDIOCPP_GO_EXPECT_BACKEND=<name>  ggml registry name that must have a device,
//	                                   e.g. MTL on the Metal leg (default CPU)
//	AUDIOCPP_GO_RESOLVE_BACKEND=<name> backend ResolveDevice must resolve to it,
//	                                   e.g. metal (default cpu)
func TestDeviceRegistryIntegration(t *testing.T) {
	if os.Getenv("AUDIOCPP_GO_DEVICES") != "1" {
		t.Skip("set AUDIOCPP_GO_DEVICES=1 (and build the lib) to read a real device registry")
	}
	libDir := envOr("AUDIOCPP_GO_LIB_DIR", "build")
	if err := Load(libDir); err != nil {
		t.Fatalf("Load(%q): %v", libDir, err)
	}
	t.Logf("shim version: %s", Version())
	if !HasDeviceAPI() {
		t.Fatal("HasDeviceAPI() = false on a freshly built library")
	}

	devs, err := Devices()
	if err != nil {
		t.Fatalf("Devices: %v", err)
	}
	if len(devs) == 0 {
		t.Fatal("Devices() is empty; every build has at least the CPU device")
	}
	expect := envOr("AUDIOCPP_GO_EXPECT_BACKEND", "CPU")
	var expected *Device
	for i, d := range devs {
		mem, err := DeviceMemory(d.Index)
		if err != nil {
			t.Fatalf("DeviceMemory(%d): %v", d.Index, err)
		}
		t.Logf("device %d: backend=%s[%d] name=%q description=%q id=%q type=%s free=%d total=%d",
			d.Index, d.Backend, d.BackendIndex, d.Name, d.Description, d.ID, d.Type, mem.FreeBytes, mem.TotalBytes)
		if d.Index != i {
			t.Errorf("Devices()[%d].Index = %d", i, d.Index)
		}
		if d.Backend == "" || d.Name == "" || d.BackendIndex < 0 {
			t.Errorf("device %d is missing its backend, name or backend index: %+v", i, d)
		}
		if mem.FreeBytes > mem.TotalBytes {
			t.Errorf("device %d reports free %d above total %d", i, mem.FreeBytes, mem.TotalBytes)
		}
		if d.Backend == expect && expected == nil {
			expected = &devs[i]
		}
	}
	if expected == nil {
		t.Fatalf("no %s device in %+v", expect, devs)
	}
	if mem, _ := DeviceMemory(expected.Index); mem.TotalBytes == 0 {
		t.Errorf("%s device reports no total memory", expect)
	}

	resolve := envOr("AUDIOCPP_GO_RESOLVE_BACKEND", "cpu")
	got, err := ResolveDevice(resolve, 0)
	if err != nil {
		t.Fatalf("ResolveDevice(%q, 0): %v", resolve, err)
	}
	if got != *expected {
		t.Errorf("ResolveDevice(%q, 0) = %+v, want %+v", resolve, got, *expected)
	}
	if _, err := ResolveDevice(resolve, 99); err == nil {
		t.Errorf("ResolveDevice(%q, 99) succeeded", resolve)
	}
	if _, err := DeviceMemory(len(devs)); err == nil {
		t.Error("DeviceMemory past the end of the list succeeded")
	}
}
