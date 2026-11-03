"""Set UVC camera controls (exposure mode and time, gain) directly over USB.

OpenCV's AVFoundation backend ignores CAP_PROP_EXPOSURE on macOS, so the eye camera was left on auto exposure.
In dim IR the auto exposure lengthens the exposure past one frame (300 = 30 ms seen on the U20CAM-9281M), and
the camera then slows itself from 120 to 30 fps. Manual exposure at <= 8 ms keeps it at 120.

The requests go to the camera's VideoControl interface through libusb (`brew install libusb`). Reading and
setting controls doesn't need the interface claimed, so it works while AVFoundation streams.
Values follow the UVC 1.5 spec: exposure in units of 100 us; ae_mode 1 = manual, 8 = aperture priority (auto).
"""

from __future__ import annotations

import ctypes
import ctypes.util
import struct

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
    return lib


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
    def __init__(self, vendor_id: int, product_id: int):
        self.lib = _libusb()
        self.ctx = ctypes.c_void_p()
        if self.lib.libusb_init(ctypes.byref(self.ctx)) != 0:
            raise RuntimeError("libusb_init failed")
        self.h = self.lib.libusb_open_device_with_vid_pid(self.ctx, vendor_id, product_id)
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
            self.lib.libusb_exit(self.ctx)
            self.h = None


def apply(uvc_cfg: dict) -> dict:
    """Apply the eye_camera.uvc config; returns the values read back (for meta.json)."""
    c = UVCControls(int(uvc_cfg["vendor_id"]), int(uvc_cfg["product_id"]))
    try:
        if uvc_cfg.get("exposure_ms") is not None:
            c.set("ae_mode", 1)
            c.set("exposure_abs", round(float(uvc_cfg["exposure_ms"]) * 10))
        elif uvc_cfg.get("auto_exposure"):
            c.set("ae_mode", 8)
        if uvc_cfg.get("gain") is not None:
            c.set("gain", int(uvc_cfg["gain"]))
        return {name: c.get(name) for name in ("ae_mode", "exposure_abs", "gain", "focus_auto")}
    finally:
        c.close()
