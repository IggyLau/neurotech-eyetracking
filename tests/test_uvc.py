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
