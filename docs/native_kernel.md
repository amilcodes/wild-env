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

Version 0.4 uses the production-oriented `TensorFireKernel`. Complete batched
landscapes remain accelerator-resident across local-behavior interpolation,
moisture response, crown transition, WENO5/RK3 level-set propagation,
reinitialization and spotting. It can run on CUDA, ROCm, MPS or CPU and is
benchmarked with `aeolus-fire benchmark`. NumPy/PyTorch local-behavior parity
and idealized front checks are part of the test suite.

A C++/CUDA port is justified only if it exceeds this tensor backend on a
measured cluster workload. It must batch complete active-front updates across
many environments and run the retained Pyretechnics point fixtures plus
NumPy/tensor parity cases. Strict cross-device stochastic parity still requires
counter-based random streams keyed by episode, minute, cell and event type;
the tensor backend currently guarantees seeded repeatability on a given device,
not identical ember draws across device families.

Version 0.6 also provides `TensorOperationsEnv`, a PyTorch tensor kernel for
fleet, service nodes, queues, payload/endurance, and attack segments. It shares
task/resource feature semantics with the canonical simulator and is used for
operations pretraining. It is not yet a replacement for fire-coupled incident
rollout: advancing-front belief/task extraction and suppression rasters must be
joined to `TensorFireKernel` before that claim is warranted.
