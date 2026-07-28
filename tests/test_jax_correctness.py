"""Does JAX's jax.pmap + jax.lax.pmean actually reproduce single-device
training, the same invariant tests/test_correctness.py proves for PyTorch
DDP -- or does it just run?

Trains the same synthetic regression task (examples_jax/model.py) single-
device and via a 4-way pmap (simulated CPU devices, see
examples_jax/train.py's docstring), then compares final parameters.
Unlike the PyTorch tests, this needs no subprocess/multi-process
launching at all: pmap's "devices" are simulated within this one process
via XLA_FLAGS, so the whole comparison runs in a single Python call.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "examples_jax"))

from train import train_pmap, train_single_device

STEPS = 50
SEED = 0
TOLERANCE = 1e-4


def main():
    import jax

    print(f"jax devices: {jax.devices()}")

    print(f"Single-device reference: {STEPS} steps, batch=32...")
    single = train_single_device(seed=SEED, steps=STEPS, batch_size=32)

    print(f"4-device pmap: {STEPS} steps, 4x8=32 effective batch...")
    multi = train_pmap(seed=SEED, steps=STEPS, n_devices=4, per_device_batch=8)

    assert single.keys() == multi.keys(), "parameter names differ"

    max_diff = 0.0
    for name in single:
        diff = float(abs(single[name] - multi[name]).max())
        max_diff = max(max_diff, diff)
        print(f"  {name}: max abs diff = {diff:.2e}")

    print(f"\nmax abs param difference: {max_diff:.2e}")
    assert max_diff < TOLERANCE, (
        f"4-device pmap training diverged from the single-device reference by {max_diff} "
        f"(tolerance {TOLERANCE})"
    )
    print(
        f"PASS: 4-device jax.pmap training matches single-device reference within {TOLERANCE}"
    )


if __name__ == "__main__":
    main()
