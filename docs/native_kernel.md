# Accelerator and native kernel boundary

The Python simulator defines canonical semantics. `native/` is a C++20 parity
surface for kernels that have been measured as rollout bottlenecks.

The C++20 primitive computes the legacy scalar or batched ignition hazard
through a stable C ABI. It remains a build/parity fixture:

```bash
cmake -S native -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native --parallel
ctest --test-dir build/native --output-on-failure
```

Version 0.3 adds the production-oriented `TensorFireKernel`. Complete batched
landscapes remain accelerator-resident across local-behavior interpolation,
moisture response, crown transition, propagation and spotting. It can run on
CUDA, ROCm, MPS or CPU and is benchmarked with `aeolus-fire benchmark`.
NumPy/PyTorch local-behavior parity is part of the test suite.

A future C++/CUDA port is justified only if it exceeds this tensor backend on a
measured cluster workload. It must batch complete active-front updates across
many environments and run the retained Pyretechnics point fixtures plus
NumPy/tensor parity cases. Strict cross-device stochastic parity still requires
counter-based random streams keyed by episode, minute, cell and event type;
the tensor backend currently guarantees seeded repeatability on a given device,
not identical ember draws across device families.
