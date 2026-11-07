"""Set UVC camera controls (exposure mode and time, gain) directly over USB.

OpenCV's AVFoundation backend ignores CAP_PROP_EXPOSURE on macOS, so the eye camera was left on auto exposure.
In dim IR the auto exposure lengthens the exposure past one frame (300 = 30 ms seen on the U20CAM-9281M), and
the camera then slows itself from 120 to 30 fps. Manual exposure at <= 8 ms keeps it at 120.

The requests go to the camera's VideoControl interface through libusb (`brew install libusb`). Reading and
setting controls doesn't need the interface claimed, so it works while AVFoundation streams.
Values follow the UVC 1.5 spec: exposure in units of 100 us; ae_mode 1 = manual, 8 = aperture priority (auto).

Two eye cameras of the same model share a vendor and product ID, so `apply` sets every matching device, not
just the first one libusb finds.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import struct
import threading

# Two camera threads (two eye cameras of one model) must not talk to the same devices at once: overlapping
# control transfers failed and read back None (dashboard, 2026-11-07).
_LOCK = threading.Lock()

CAMERA_TERMINAL = {"ae_mode": (0x02, 1), "ae_priority": (0x03, 1), "exposure_abs": (0x04, 4),
                   "focus_auto": (0x08, 1)}   # focus_auto = external trigger mode on the U20CAM-9281M
PROCESSING_UNIT = {"backlight_comp": (0x01, 2), "brightness": (0x02, 2), "gain": (0x04, 2)}
GET_CUR, SET_CUR = 0x81, 0x01


def _libusb():
    for name in (ctypes.util.find_library("usb-1.0"), "/opt/homebrew/lib/libusb-1.0.dylib",
                 "/usr/local/lib/libusb-1.0.dylib"):
        if name:
            try:
                lib = ctypes.CDLL(name)
                break
            except OSError:
                continue
    else:
        raise RuntimeError("libusb not found: brew install libusb")
    lib.libusb_open_device_with_vid_pid.restype = ctypes.c_void_p
    lib.libusb_open_device_with_vid_pid.argtypes = [ctypes.c_void_p, ctypes.c_uint16, ctypes.c_uint16]
    lib.libusb_control_transfer.argtypes = [ctypes.c_void_p, ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint16,
                                            ctypes.c_uint16, ctypes.c_char_p, ctypes.c_uint16, ctypes.c_uint]
    lib.libusb_close.argtypes = [ctypes.c_void_p]
    lib.libusb_exit.argtypes = [ctypes.c_void_p]
    lib.libusb_get_device_list.restype = ctypes.c_ssize_t
    lib.libusb_get_device_list.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))]
    lib.libusb_free_device_list.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
    lib.libusb_get_device_descriptor.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.libusb_open.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    lib.libusb_get_bus_number.argtypes = [ctypes.c_void_p]
    lib.libusb_get_port_number.argtypes = [ctypes.c_void_p]
    return lib


def _open_all(lib, ctx, vendor_id: int, product_id: int) -> list[tuple[ctypes.c_void_p, str]]:
    """Open every device with this vendor/product ID; returns (handle, "bus-port") pairs."""
    devs = ctypes.POINTER(ctypes.c_void_p)()
    n = lib.libusb_get_device_list(ctx, ctypes.byref(devs))
    found = []
    try:
        for i in range(max(n, 0)):
            dev = devs[i]
            desc = ctypes.create_string_buffer(18)
            if lib.libusb_get_device_descriptor(dev, desc) != 0:
                continue
            vid, pid = struct.unpack_from("<HH", desc.raw, 8)
            if (vid, pid) != (vendor_id, product_id):
                continue
            h = ctypes.c_void_p()
            if lib.libusb_open(dev, ctypes.byref(h)) == 0 and h:
                found.append((h, f"{lib.libusb_get_bus_number(dev)}-{lib.libusb_get_port_number(dev)}"))
    finally:
        if n > 0:
            lib.libusb_free_device_list(devs, 1)
    return found


def find_units(config_descriptor: bytes) -> tuple[int, int | None, int | None]:
    """(VideoControl interface number, camera terminal ID, processing unit ID) from a configuration descriptor."""
    i, iface, vc_if, ct, pu = 0, None, 0, None, None
    while i + 2 < len(config_descriptor):
        length, kind = config_descriptor[i], config_descriptor[i + 1]
        if kind == 0x04:                                   # INTERFACE: number, class, subclass
            iface = (config_descriptor[i + 2], config_descriptor[i + 5], config_descriptor[i + 6])
        elif kind == 0x24 and iface and iface[1:] == (0x0E, 0x01):   # class-specific, VideoControl
            vc_if, sub = iface[0], config_descriptor[i + 2]
            if sub == 0x02 and struct.unpack_from("<H", config_descriptor, i + 4)[0] == 0x0201:
                ct = config_descriptor[i + 3]              # input terminal of type camera
            elif sub == 0x05:
                pu = config_descriptor[i + 3]
        i += max(length, 1)
    return vc_if, ct, pu


class UVCControls:
    def __init__(self, vendor_id: int, product_id: int, handle=None, lib=None, ctx=None, where: str = ""):
        """Open the first matching camera, or wrap an already opened handle (from `all_cameras`)."""
        self.where = where
        self.lib = lib or _libusb()
        self._owns_ctx = ctx is None
        self.ctx = ctx or ctypes.c_void_p()
        if self._owns_ctx and self.lib.libusb_init(ctypes.byref(self.ctx)) != 0:
            raise RuntimeError("libusb_init failed")
        self.h = handle or self.lib.libusb_open_device_with_vid_pid(self.ctx, vendor_id, product_id)
        if not self.h:
            self.lib.libusb_exit(self.ctx)
            raise RuntimeError(f"USB camera {vendor_id:04x}:{product_id:04x} not found")
        buf = ctypes.create_string_buffer(4096)
        n = self.lib.libusb_control_transfer(self.h, 0x80, 0x06, 0x0200, 0, buf, 4096, 1000)  # GET_DESCRIPTOR
        self.interface, self.ct, self.pu = find_units(buf.raw[:max(n, 0)])

    def _where(self, name: str) -> tuple[int, int, int]:
        if name in CAMERA_TERMINAL:
            return (self.ct, *CAMERA_TERMINAL[name])
        if name in PROCESSING_UNIT:
            return (self.pu, *PROCESSING_UNIT[name])
        raise KeyError(name)

    def get(self, name: str) -> int | None:
        unit, sel, size = self._where(name)
        buf = ctypes.create_string_buffer(size)
        r = self.lib.libusb_control_transfer(self.h, 0xA1, GET_CUR, sel << 8, (unit << 8) | self.interface,
                                             buf, size, 1000)
        return int.from_bytes(buf.raw[:size], "little", signed=True) if r == size else None

    def set(self, name: str, value: int) -> bool:
        unit, sel, size = self._where(name)
        buf = ctypes.create_string_buffer(int(value).to_bytes(size, "little", signed=True), size)
        r = self.lib.libusb_control_transfer(self.h, 0x21, SET_CUR, sel << 8, (unit << 8) | self.interface,
                                             buf, size, 1000)
        return r == size

    def close(self):
        if self.h:
            self.lib.libusb_close(self.h)
            if self._owns_ctx:
                self.lib.libusb_exit(self.ctx)
            self.h = None


def _set_controls(c: UVCControls, uvc_cfg: dict) -> dict:
    """Set only controls that differ from what the camera already has.

    Writing a control mid-stream can make the camera stall briefly. With two eye cameras sharing a USB ID, each
    camera's watchdog re-sent the settings to both, the other camera stalled, its watchdog fired, and the two kept
    re-triggering each other (2026-11-07). Skipping unchanged values breaks that loop.
    """
    def put(name, value):
        if c.get(name) != value:
            c.set(name, value)
    if uvc_cfg.get("exposure_ms") is not None:
        put("ae_mode", 1)
        put("exposure_abs", round(float(uvc_cfg["exposure_ms"]) * 10))
    elif uvc_cfg.get("auto_exposure"):
        put("ae_mode", 8)
    if uvc_cfg.get("gain") is not None:
        put("gain", int(uvc_cfg["gain"]))
    return {name: c.get(name) for name in ("ae_mode", "exposure_abs", "gain", "focus_auto")}


def apply(uvc_cfg: dict) -> dict:
    """Apply the eye_camera.uvc config to every connected camera with that vendor/product ID.

    Returns the values read back (for meta.json). With one camera that's the old flat dict; with several
    (two eye cameras of the same model) it's {"bus-port": values} per camera.
    """
    with _LOCK:
        return _apply_all(uvc_cfg)


def _apply_all(uvc_cfg: dict) -> dict:
    vid, pid = int(uvc_cfg["vendor_id"]), int(uvc_cfg["product_id"])
    lib = _libusb()
    ctx = ctypes.c_void_p()
    if lib.libusb_init(ctypes.byref(ctx)) != 0:
        raise RuntimeError("libusb_init failed")
    try:
        handles = _open_all(lib, ctx, vid, pid)
        if not handles:
            raise RuntimeError(f"USB camera {vid:04x}:{pid:04x} not found")
        results = {}
        for h, where in handles:
            c = UVCControls(vid, pid, handle=h, lib=lib, ctx=ctx, where=where)
            try:
                results[where] = _set_controls(c, uvc_cfg)
            finally:
                c.close()
        return next(iter(results.values())) if len(results) == 1 else results
    finally:
        lib.libusb_exit(ctx)
