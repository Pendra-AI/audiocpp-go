package audiocpp

import (
	"errors"
	"fmt"
)

// ErrDeviceAPIUnavailable is returned by the device functions when the loaded
// library predates the device registry (shim 0.2.0, release v0.2.0). Such a
// library still loads and synthesises; only these readings are missing.
var ErrDeviceAPIUnavailable = errors.New("audiocpp: the loaded library has no device registry (needs v0.2.0 or later)")

// DeviceType is the kind of a device, mirroring ggml's ggml_backend_dev_type.
type DeviceType int

const (
	DeviceTypeCPU   DeviceType = 0 // host CPU, system memory
	DeviceTypeGPU   DeviceType = 1 // GPU with its own memory
	DeviceTypeIGPU  DeviceType = 2 // integrated GPU sharing host memory
	DeviceTypeAccel DeviceType = 3 // accelerator used alongside the CPU backend
	DeviceTypeMeta  DeviceType = 4 // wrapper over several devices
)

func (t DeviceType) String() string {
	switch t {
	case DeviceTypeCPU:
		return "cpu"
	case DeviceTypeGPU:
		return "gpu"
	case DeviceTypeIGPU:
		return "igpu"
	case DeviceTypeAccel:
		return "accel"
	case DeviceTypeMeta:
		return "meta"
	}
	return fmt.Sprintf("unknown(%d)", int(t))
}

// Device describes one device in the library's own ggml registry. The library
// links ggml statically and hides it, so this is the only view of the devices —
// and the device numbering — a loaded speech model actually uses.
type Device struct {
	// Index is the device's position in the library's device list, as returned
	// by Devices. It is what DeviceMemory takes and is fixed for the process.
	Index int
	// Backend is the ggml registry name: "CUDA", "Vulkan", "MTL", "ROCm", "CPU".
	Backend string
	// BackendIndex is the index within Backend — the value ModelParams.Device
	// takes. For CUDA it is CUDA's own device ordinal, which is not nvidia-smi's
	// index (CUDA orders fastest first unless CUDA_DEVICE_ORDER says otherwise).
	BackendIndex int
	// Name is ggml's device handle, e.g. "CUDA0".
	Name string
	// Description is the hardware name, e.g. "NVIDIA GeForce RTX 4090".
	Description string
	// ID is the lower-case PCI bus id ("0000:01:00.0") where the backend knows
	// it (CUDA, Vulkan), else "". It names the same card across numberings.
	ID string
	// Type is the device kind. CUDA reports DeviceTypeGPU for every device,
	// shared-memory ones (DGX Spark, Jetson) included.
	Type DeviceType
}

// Memory is a device's free and total memory in bytes, device-wide (every
// process's use counts), as the backend reports it. Both are zero when the
// backend has no reading.
type Memory struct {
	FreeBytes  uint64
	TotalBytes uint64
}

// deviceStringLen bounds each string audiocpp_device_info writes.
const deviceStringLen = 256

// HasDeviceAPI reports whether the loaded library has the device registry.
// False before Load, and for a library older than v0.2.0.
func HasDeviceAPI() bool {
	loadMu.Lock()
	defer loadMu.Unlock()
	return loaded && deviceAPI
}

// Devices lists the devices the library can run a model on, in the library's
// own order (Devices()[i].Index == i). The list is fixed for the life of the
// process, so callers may cache it. Safe for concurrent use, including during a
// Synthesize.
//
// The first device call initialises the engine's backends if no synthesis has
// yet; the CUDA backend creates its context on every visible GPU at that point.
func Devices() (devs []Device, err error) {
	if !HasDeviceAPI() {
		return nil, ErrDeviceAPIUnavailable
	}
	defer func() {
		if r := recover(); r != nil {
			devs, err = nil, fmt.Errorf("audiocpp: device query panicked across FFI: %v", r)
		}
	}()
	n := int(cDeviceCount())
	devs = make([]Device, 0, n)
	for i := 0; i < n; i++ {
		d, err := deviceAt(i)
		if err != nil {
			return nil, err
		}
		devs = append(devs, d)
	}
	return devs, nil
}

// DeviceMemory reads the free and total memory of the device at index (a
// Device.Index) now. It is a device-wide reading, so a model's share is the
// drop in FreeBytes across its load and its first synthesis — the engine
// allocates on the device at the first synthesis, not at load. The CPU device
// reports FreeBytes == TotalBytes except on Windows, so it cannot show a drop.
// Safe for concurrent use, including during a Synthesize.
func DeviceMemory(index int) (mem Memory, err error) {
	if !HasDeviceAPI() {
		return Memory{}, ErrDeviceAPIUnavailable
	}
	defer func() {
		if r := recover(); r != nil {
			mem, err = Memory{}, fmt.Errorf("audiocpp: device query panicked across FFI: %v", r)
		}
	}()
	var free, total uint64
	if status := cDeviceMemory(int32(index), &free, &total); status != statusOK {
		return Memory{}, deviceError(status, index)
	}
	return Memory{FreeBytes: free, TotalBytes: total}, nil
}

// ResolveDevice returns the device a model loaded with ModelParams.Backend
// backend and ModelParams.Device device runs on, resolved the way the engine
// resolves them ("" is "cpu"; "best" is the first GPU, else the first
// integrated GPU, else the CPU). It needs no loaded model, so a caller can pick
// the device to watch before New. An unknown backend, or a device this library
// does not have, is an *Error with the bad-argument code.
func ResolveDevice(backend string, device int) (dev Device, err error) {
	if !HasDeviceAPI() {
		return Device{}, ErrDeviceAPIUnavailable
	}
	defer func() {
		if r := recover(); r != nil {
			dev, err = Device{}, fmt.Errorf("audiocpp: device query panicked across FFI: %v", r)
		}
	}()
	index := int(cResolveDevice(backend, int32(device)))
	if index < 0 {
		return Device{}, &Error{Code: statusBadArg, Msg: fmt.Sprintf("no device %d for backend %q in this library", device, backend)}
	}
	return deviceAt(index)
}

// Device returns the device the model runs on: ResolveDevice for the backend
// and device it was loaded with. It does not touch the native model, so it is
// safe during a Synthesize and after Close.
func (m *Model) Device() (Device, error) {
	return ResolveDevice(m.backend, m.device)
}

func deviceAt(index int) (Device, error) {
	var (
		backend, name, description, id [deviceStringLen]byte
		backendIndex, deviceType       int32
	)
	status := cDeviceInfo(int32(index),
		&backend[0], deviceStringLen, &backendIndex,
		&name[0], deviceStringLen,
		&description[0], deviceStringLen,
		&id[0], deviceStringLen,
		&deviceType)
	if status != statusOK {
		return Device{}, deviceError(status, index)
	}
	return Device{
		Index:        index,
		Backend:      cBufString(backend[:]),
		BackendIndex: int(backendIndex),
		Name:         cBufString(name[:]),
		Description:  cBufString(description[:]),
		ID:           cBufString(id[:]),
		Type:         DeviceType(deviceType),
	}, nil
}

func deviceError(status int32, index int) error {
	if status == statusBadArg {
		return &Error{Code: int(status), Msg: fmt.Sprintf("no device at index %d", index)}
	}
	return &Error{Code: int(status), Msg: fmt.Sprintf("device %d could not be read", index)}
}
