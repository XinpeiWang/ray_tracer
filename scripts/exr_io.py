"""Minimal OpenEXR reader for the renderer's own output (scanline, float32, ZIP/ZIPS/NONE), no third-party EXR library needed.

Only what ray_tracer.exe --output foo.exr writes is supported: a single-part scanline file whose channels are 32-bit float.
Anything else (tiled, half, PIZ, multipart) raises ValueError instead of returning wrong numbers.
"""
import struct
import zlib

import numpy as np

_COMPRESSION_LINES = {0: 1, 2: 1, 3: 16}  # NONE, ZIPS, ZIP -> scanlines per block


def _read_header(b):
    if b[:4] != b'v/1\x01':
        raise ValueError('not an OpenEXR file')
    flags = struct.unpack('<I', b[4:8])[0]
    if flags & 0x1a00:  # tiled / long names / non-image / multipart
        raise ValueError('unsupported EXR flags: 0x%x' % flags)
    attrs = {}
    i = 8
    while b[i] != 0:
        j = b.index(b'\0', i)
        name = b[i:j].decode()
        i = j + 1
        j = b.index(b'\0', i)
        typ = b[i:j].decode()
        i = j + 1
        size = struct.unpack('<I', b[i:i + 4])[0]
        i += 4
        attrs[name] = (typ, b[i:i + size])
        i += size
    return attrs, i + 1


def read_exr(path):
    """Return (height, width, {channel_name: float32 array [height, width]})."""
    with open(path, 'rb') as f:
        b = f.read()
    attrs, pos = _read_header(b)

    channels = []  # (name, pixel_type) in file order
    ch = attrs['channels'][1]
    i = 0
    while ch[i] != 0:
        j = ch.index(b'\0', i)
        name = ch[i:j].decode()
        ptype = struct.unpack('<I', ch[j + 1:j + 5])[0]
        channels.append((name, ptype))
        i = j + 1 + 16
    if any(t != 2 for _, t in channels):
        raise ValueError('only 32-bit float channels are supported')

    comp = attrs['compression'][1][0]
    if comp not in _COMPRESSION_LINES:
        raise ValueError('unsupported EXR compression %d' % comp)
    lines_per_block = _COMPRESSION_LINES[comp]

    x0, y0, x1, y1 = struct.unpack('<4i', attrs['dataWindow'][1])
    width, height = x1 - x0 + 1, y1 - y0 + 1
    n_blocks = (height + lines_per_block - 1) // lines_per_block
    offsets = struct.unpack('<%dQ' % n_blocks, b[pos:pos + 8 * n_blocks])

    out = {name: np.empty((height, width), np.float32) for name, _ in channels}
    for off in offsets:
        y, size = struct.unpack('<ii', b[off:off + 8])
        data = b[off + 8:off + 8 + size]
        rows = min(lines_per_block, height - (y - y0))
        raw_size = rows * len(channels) * width * 4
        if comp != 0 and size < raw_size:  # stored compressed (ZIP falls back to raw when it does not shrink)
            data = zlib.decompress(data)
            a = np.frombuffer(data, np.uint8).astype(np.int32)
            a = np.concatenate(([a[0]], a[1:]))
            a = (np.cumsum(a - np.concatenate(([0], np.full(len(a) - 1, 128)))) & 0xFF).astype(np.uint8)
            half = (len(a) + 1) // 2
            data = np.empty(len(a), np.uint8)
            data[0::2] = a[:half]
            data[1::2] = a[half:]
            data = data.tobytes()
        px = np.frombuffer(data, '<f4').reshape(rows, len(channels), width)
        for k, (name, _) in enumerate(channels):
            out[name][y - y0:y - y0 + rows] = px[:, k, :]
    return height, width, out
