"""Check TMA byte permutation and shared-bank mapping without CUDA."""

from collections import defaultdict

for kbytes in (1152, 2560):
    original = [
        bytes((row * 37 + k) % 256 for k in range(kbytes)) for row in range(128)
    ]
    paired = []
    for row in range(0, 128, 2):
        paired.append(
            b"".join(
                original[row][k : k + 64] + original[row + 1][k : k + 64]
                for k in range(0, kbytes, 64)
            )
        )
    for row in range(128):
        for k in range(kbytes):
            assert (
                paired[row // 2][(k // 64) * 128 + (row % 2) * 64 + k % 64]
                == original[row][k]
            )
for base in (0, 32, 64, 96):
    for half in (0, 8):
        bank_old = defaultdict(set)
        bank_new = defaultdict(set)
        for lane in range(32):
            row = lane // 4 + half
            k = base + (lane % 4) * 4
            old = row * 64 + k // 2
            prow = row // 2
            pcol = (row % 2) * 64 + k // 2
            new = prow * 128 + (pcol ^ ((prow & 7) * 16))
            bank_old[old // 4 % 32].add(old // 4)
            bank_new[new // 4 % 32].add(new // 4)
        assert max(map(len, bank_old.values())) == 4
        assert max(map(len, bank_new.values())) == 1
print("PASS: paired TMA bytes reversible; 4-way -> 1-way bank mapping for all K slices")
