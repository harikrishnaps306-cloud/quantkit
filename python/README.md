# quantkit (Python)

Python bindings for the Rust [`quantkit`](../README.md) library (not published to PyPI).

```python
import numpy as np
import quantkit as qk

qk.bs_price(42, 40, 0.10, 0.20, 0.5)                          # 4.759...
qk.bs_greeks(42, 40, 0.10, 0.20, 0.5)                         # {'delta': ..., 'gamma': ..., ...}
qk.implied_vol(1.875, 21, 20, 0.10, 0.25)                     # ~0.235

spots = np.linspace(80, 120, 5)
qk.bs_price_batch(spots, 100.0, 0.03, 0.2, 1.0)               # same-shape arrays; scalars broadcast

d = qk.implied_vol_detailed(prices, spots, 100.0, 0.03, 1.0)  # per-row status, residual, iterations, vega, messages
ds = qk.make_dataset(rows=10_000, seed=7)                     # reproducible dataset + audit metadata
```

Build locally with `maturin develop --release`. The wheel uses the Python 3.9 stable ABI and
declares `Requires-Python >=3.9`, `numpy>=1.21`. It passed the test-suite on CPython 3.9-3.13 with
NumPy 1.21.6-2.5.4 (ten combinations, Windows x86-64 only); Linux and macOS are untested. See `docs/NUMERICS.md` for shape, error and
threading semantics and `docs/BENCHMARKS.md` for measured performance.
