# Project Aether: Contribution Guidelines

Welcome to the frontier of CPU inference. To keep the engine lean and fast, all contributions must follow this workflow.

## 🔄 The Development Cycle

### 1. Design & Consultation
Before writing a complex kernel or runtime change, open an issue or start a discussion about the memory implications.
**Question to answer**: "Does this change reduce memory movement?"

### 2. Implementation
- Implement a **naive reference** version first for correctness.
- Implement the **optimized** version (Tiling \(\rightarrow\) SIMD \(\rightarrow\) Quant Fusion).
- Ensure all memory is sourced from the `UnifiedArena`.

### 3. Verification
A contribution is not complete without:
- **Correctness Test**: A test case comparing the optimized kernel output against the naive reference.
- **Benchmark**: A `aether-benchmark` report showing the impact on `tokens/sec`.
- **Profiling**: A `perf` or `cachegrind` report showing L1/L2 cache miss rates.

## 📁 Project Structure
- `crates/arena`: Memory management.
- `crates/tensor`: Data layout and views.
- `crates/kernels`: The performance heart (GEMM/GEMV).
- `crates/runtime`: The execution loop.

## 📝 Commit Messages
Use a clear prefix:
- `feat(kernel):` for new operations.
- `perf(tensor):` for memory locality improvements.
- `fix(arena):` for memory bug fixes.
- `docs(readme):` for documentation updates.
