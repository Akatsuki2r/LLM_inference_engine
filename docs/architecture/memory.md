# Memory Tape Architecture

Quantumn utilizes a **Unified Contiguous Memory Arena** (`aether-arena`). Instead of per-layer allocations, the engine pre-allocates a monolithic, 64-byte aligned memory block at startup.

- **Tape Layout**: Weights, KV Cache, and Scratchpad are arranged sequentially in the exact order of execution.
- **Cache-Friendly**: This linear layout ensures that the CPU prefetcher remains saturated, eliminating DRAM stalls.
- **Alignment**: Every tensor offset is aligned to 64 bytes (cache line width) to prevent cache-line splitting during AVX2 loads.
