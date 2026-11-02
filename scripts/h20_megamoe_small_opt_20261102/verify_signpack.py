"""Exhaustively check every nibble/position/UE8M0 against the existing LUT."""

import re
from pathlib import Path

p = (
    Path(__file__).resolve().parents[3]
    / "DeepGEMM/deep_gemm/include/deep_gemm/common/fp4_decode_detail.cuh"
)
lut = [int(s, 16) for s in re.findall(r"0x([0-9a-fA-F]+)ull", p.read_text())]
assert len(lut) == 256
permutation = (0, 4, 1, 5, 2, 6, 3, 7)
checks = 0
for e, table in enumerate(lut):
    for position in range(8):
        for code in range(16):
            src = code << (position * 4)
            packed = src & 0x77777777
            for dest, source in enumerate(permutation):
                packed |= ((src >> (source * 4)) & 8) << (dest * 4)
            expected = []
            actual = []
            for i in range(8):
                nibble = (src >> (i * 4)) & 15
                expected.append(
                    ((table >> ((nibble & 7) * 8)) & 255) | ((nibble & 8) << 4)
                )
                mag = (packed >> (i * 4)) & 7
                out_byte = i % 4
                sign = ((packed << (4 if i < 4 else 0)) >> (out_byte * 8)) & 128
                actual.append(((table >> (mag * 8)) & 255) | sign)
            assert actual == expected, (e, position, code, expected, actual)
            checks += 1
print(f"PASS: {checks} payload/scale cases, all 8 output bytes equal")
