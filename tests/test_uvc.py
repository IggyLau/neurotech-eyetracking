import struct

from eyetrack.capture.uvc import find_units


def test_find_units_reads_camera_terminal_and_processing_unit():
    iface_vc = bytes([9, 0x04, 0, 0, 1, 0x0E, 0x01, 0, 0])
    header = bytes([13, 0x24, 0x01, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1])
    camera = bytes([18, 0x24, 0x02, 1]) + struct.pack("<H", 0x0201) + bytes(12)
    unit = bytes([11, 0x24, 0x05, 2, 1, 0, 0, 2, 0, 0, 0])
    iface_vs = bytes([9, 0x04, 1, 0, 1, 0x0E, 0x02, 0, 0])
    config = bytes([9, 0x02, 0, 0, 2, 1, 0, 0x80, 250]) + iface_vc + header + camera + unit + iface_vs
    assert find_units(config) == (0, 1, 2)


def test_controls_already_set_are_not_written_again():
    from eyetrack.capture.uvc import _set_controls

    class Fake:
        def __init__(self, vals):
            self.vals, self.writes = dict(vals), []

        def get(self, name):
            return self.vals.get(name)

        def set(self, name, value):
            self.writes.append(name)
            self.vals[name] = value
            return True

    done = Fake({"ae_mode": 1, "exposure_abs": 80, "gain": 15, "focus_auto": 0})
    _set_controls(done, {"exposure_ms": 8})
    assert done.writes == []            # a re-send from the other camera's watchdog changes nothing
    reset = Fake({"ae_mode": 8, "exposure_abs": 300, "gain": 15, "focus_auto": 0})
    out = _set_controls(reset, {"exposure_ms": 8})
    assert reset.writes == ["ae_mode", "exposure_abs"] and out["exposure_abs"] == 80
