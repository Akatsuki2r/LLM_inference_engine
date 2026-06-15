# Project Aether 🌌

**CPU-First Frontier-Class LLM Inference Engine**

Project Aether is a clean-room, ultra-optimized LLM inference engine written in Rust. We aren't just "running models on CPU"—we are pushing the boundaries of CPU inference toward frontier-class reasoning quality.

## 🎯 Mission
To maximize intelligence per watt, per byte, and per core on mid-to-low-end hardware (x86_64 AVX2/FMA3).

## 🛠 Core Philosophy
- **Memory is the Bottleneck**: Minimize DRAM trips, cache misses, and tensor movement.
- **Quantization is Mandatory**: Weights stay compressed until the moment of computation.
- **CPUs are Cache Machines**: Design for cache locality and vectorized pipelines.
- **Inference-Time Scaling**: Implement "slow smart modes" (Best-of-N, recursive refinement).

## 🏗 Architecture
Project Aether is structured into 5 layers:
1. **Storage**: GGUF/SafeTensors loading, mmap streaming.
2. **Tensor**: Shape, stride, and view logic (Zero-allocation).
3. **Kernel**: Performance heart (GEMM, GEMV, Softmax, RMSNorm).
4. **Runtime**: Transformer execution and KV cache management.
5. **Intelligence**: Inference-time scaling and reasoning loops.

## 🚀 Getting Started

### Prerequisites
- Rust (Latest Stable)
- x86_64 CPU with AVX2 and FMA3 support.

### Quick Start
Run the tests to verify the memory foundations:
```bash
cargo test -p aether-arena
cargo test -p aether-tensor
```

### Basic Usage (Example)
```rust
use aether_arena::{UnifiedArena, MemoryCategory};
use aether_tensor::Tensor;

// 1. Initialize the giant contiguous memory block (e.g., 1GB)
let mut arena = UnifiedArena::new(1024 * 1024 * 1024).unwrap();

// 2. Allocate memory for weights (64-byte aligned)
let weight_ptr = arena.alloc(1024 * 4, MemoryCategory::Weights).unwrap();

// 3. Wrap that memory in a Tensor for structured access
let weight_tensor = Tensor::new(weight_ptr as *const f32, &[32, 32]);

// 4. Create a zero-copy view (reshape)
let view = weight_tensor.view(&[1024]);
```

## 📊 Benchmarking
Every optimization must be accompanied by a benchmark. Use the `aether-benchmark` crate to track tokens/sec and cache miss rates.
