# quantkit (Python)

Python bindings for the Rust [`quantkit`](../README.md) crate.

```python
import numpy as np
import quantkit as qk

qk.bs_price(42, 40, 0.10, 0.20, 0.5)                    # 4.759...
qk.bs_greeks(42, 40, 0.10, 0.20, 0.5)                   # {'delta': ..., 'gamma': ..., ...}
qk.implied_vol(1.875, 21, 20, 0.10, 0.25)               # ~0.235

# Vectorised, GIL released, scalars broadcast:
spots = np.linspace(80, 120, 5)
qk.bs_price_batch(spots, 100.0, 0.03, 0.2, 1.0)
```

Build locally with `maturin develop` (see the repository README).
