# Project Aether: Engineering Rules

These rules are non-negotiable. They ensure that the engine remains deterministic, hyper-optimized, and compatible with the target hardware (Intel Kaby Lake / 2-core AVX2).

## 🚫 The "NEVER" List
- **NO GPUs**: Do not include CUDA, ROCm, or Metal code. Aether is CPU-First.
- **NO RUNTIME MALLOCS**: Once the `UnifiedArena` is initialized, no heap allocations (`Box`, `Vec`, `HashMap`) are allowed during the inference loop. Use the arena or stack-allocated arrays.
- **NO GLOBAL TENSOR EXPANSION**: Never expand quantized tensors to FP32 globally. Dequantize only in registers during fused execution.
- **NO AVX-512**: Only AVX2 and FMA3 are allowed to ensure compatibility with the target hardware baseline.

## ✅ The "ALWAYS" List
- **SUREFIRE ALIGNMENT**: Every single major buffer must be 64-byte aligned. No exceptions.
- **BENCHMARK BEFORE COMMIT**: Every performance change must include a before/after report from the `aether-benchmark` crate.
- **VALIDATE CORRECTNESS**: New kernels must pass triple-loop reference tests before optimization.
- **MEMORY FIRST**: Before optimizing for CPU cycles, optimize for memory locality. Minimize DRAM trips.
- **CONTIGUOUS DATA**: Keep data contiguous. Avoid pointer chasing and fragmentation.

## 🛠 Implementation Standard
- **Idiomatic Rust**: Use the type system to enforce invariants (e.g., using the `Tensor` view instead of raw pointers).
- **Explicit Layouts**: If you are doing pointer arithmetic, document the exact memory layout in the comments.
- **Zero-Cost Abstractions**: Prefer generics and inlining over virtual dispatch (dyn).
