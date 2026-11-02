"""Bitwise check the full raw-FP4 -> TMA -> RS A-register layout."""

import random
import re
from pathlib import Path

r = Path(__file__).resolve().parents[3] / "DeepGEMM"
lut = [
    int(x, 16)
    for x in re.findall(
        r"0x([0-9a-f]+)ull",
        (r / "deep_gemm/include/deep_gemm/common/fp4_decode_detail.cuh").read_text(),
    )
]
g = random.Random(6318)
n, k = 128, 128
raw = [bytes(g.randrange(256) for _ in range(k // 2)) for _ in range(n)]
sf = [[g.randrange(256) for _ in range(k // 32)] for _ in range(n)]
fragment = (0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15)
permutation = (0, 4, 1, 5, 2, 6, 3, 7)
packed = []
for row in raw:
    b = bytearray()
    for start in range(0, k // 2, 16):
        b.extend(row[start + i] for i in fragment)
    for j in range(0, k // 2, 4):
        x = int.from_bytes(b[j : j + 4], "little")
        y = x & 0x77777777
        for dest, source in enumerate(permutation):
            y |= ((x >> (source * 4)) & 8) << (dest * 4)
        b[j : j + 4] = y.to_bytes(4, "little")
    packed.append(b)
smem = bytearray(n * k // 2)
for row in range(n):
    for col in range(k // 2):
        prow = row // 2
        pcol = (row % 2) * 64 + col
        smem[prow * 128 + (pcol ^ ((prow & 7) * 16))] = packed[row][col]
checks = 0
for wg in range(2):
    for warp in range(4):
        for lane in range(32):
            for ko in range(0, k, 32):
                for half in range(2):
                    row = wg * 64 + warp * 16 + lane // 4 + half * 8
                    prow = row // 2
                    pcol = (row % 2) * 64 + ko // 2 + (lane % 4) * 4
                    off = prow * 128 + (pcol ^ ((prow & 7) * 16))
                    x = int.from_bytes(smem[off : off + 4], "little")
                    for i in range(8):
                        byte = i % 4
                        sign = ((x << (4 if i < 4 else 0)) >> (byte * 8)) & 128
                        actual = (
                            (lut[sf[row][ko // 32]] >> (((x >> (i * 4)) & 7) * 8)) & 255
                        ) | sign
                        logical_k = ko + (lane % 4) * 4 + (i // 4) * 16 + i % 4
                        nibble = (raw[row][logical_k // 2] >> (logical_k % 2 * 4)) & 15
                        expected = (
                            (lut[sf[row][ko // 32]] >> ((nibble & 7) * 8)) & 255
                        ) | ((nibble & 8) << 4)
                        assert actual == expected, (row, logical_k, actual, expected)
                        checks += 1
print(f"PASS: {checks} register FP8 bytes match raw FP4+UE8M0 exactly")
