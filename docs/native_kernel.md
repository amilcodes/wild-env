# Native kernel boundary

The Python simulator defines canonical semantics. `native/` is a C++20 parity
surface for kernels that have been measured as rollout bottlenecks.

The current native primitive computes scalar or batched ignition hazards
through a stable C ABI. It has a standalone smoke test and CMake shared-library
target:

```bash
cmake -S native -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native --parallel
ctest --test-dir build/native --output-on-failure
```

The batch API is intentionally small enough to compare bit-for-bit against the
Python probability calculation. It is not currently called from the Python
environment, because crossing the ABI once per cell would cost more than the
arithmetic saved.

A production native port should batch complete active-front updates across
many environments, keep state in structure-of-arrays form, and expose one
operation per simulated minute. CPU SIMD and CUDA implementations must run the
same recorded parity fixtures before either becomes a training backend.
Stochastic parity requires counter-based random streams keyed by episode,
minute, cell and event type; sharing a mutable generator between host and
device would make policy comparisons irreproducible.
