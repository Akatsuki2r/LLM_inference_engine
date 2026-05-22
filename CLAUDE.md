# `MASTER_CONTEXT.md`

## Project: Project Aether — CPU-First Frontier-Class LLM Inference Engine

---

# 0. Mission Statement

Project Aether is a clean-room, CPU-first, ultra-optimized LLM inference engine written in Rust.

The goal is not merely “running models on CPU.”

The goal is:

* pushing CPU inference toward frontier-class reasoning quality,
* minimizing memory bandwidth waste,
* maximizing cache residency,
* implementing aggressive quantization,
* integrating inference-time scaling systems,
* and eventually producing novel research contributions.

This engine prioritizes:

* x86_64 AVX2/FMA3 systems first,
* low-end and mid-range hardware,
* deterministic execution,
* and local-first intelligence.

Initial hardware target:

* Intel Kaby Lake (ThinkPad X270 class systems)
* 2 physical cores
* AVX2 + FMA3
* single-channel DDR4 memory

This hardware limitation is treated as a design advantage:
constraints force efficiency.

---

# 1. Core Philosophy

The engine follows five principles:

## 1.1 Memory Is The Real Bottleneck

LLM inference is primarily memory-bandwidth bound.

The enemy is:

* cache misses,
* DRAM trips,
* pointer chasing,
* fragmentation,
* and redundant tensor movement.

The engine must minimize:

* allocations,
* copies,
* tensor reshaping,
* and scattered access patterns.

Every design decision must answer:

> “Does this reduce memory movement?”

---

## 1.2 Quantization Is Mandatory

FP16 is not enough.

Core formats:

* Q4_0
* Q4_K
* Q5
* INT2 experimental
* ternary experimental

Weights should remain compressed until the exact moment computation occurs.

Dequantization should happen:

* in registers,
* during fused execution,
* never into large temporary buffers.

---

## 1.3 GEMM/GEMV Dominates Everything

Most runtime cost comes from:

* matrix multiplication,
* matrix-vector multiplication,
* attention projections,
* FFN projections.

Kernel optimization is the heart of the project.

Everything else is secondary.

---

## 1.4 CPUs Are Cache Machines

GPUs thrive on massive parallelism.

CPUs thrive on:

* cache locality,
* branch prediction,
* low latency,
* vectorized pipelines.

The architecture must align with CPU strengths.

---

## 1.5 Intelligence Can Be Scaled At Inference Time

Model weights are only one axis of intelligence.

Additional compute at inference can improve reasoning:

* self-consistency,
* verification,
* entropy-triggered reruns,
* latent recursive refinement,
* best-of-N decoding,
* speculative reasoning loops.

The engine must support “slow smart modes.”

---

# 2. Repository Architecture

```txt
aether/
├── crates/
│   ├── core/
│   ├── tensor/
│   ├── arena/
│   ├── formats/
│   ├── kernels/
│   ├── quant/
│   ├── runtime/
│   ├── transformer/
│   ├── scheduler/
│   ├── kv_cache/
│   ├── sampling/
│   ├── tts/
│   ├── benchmark/
│   ├── profiling/
│   ├── cli/
│   └── api/
│
├── models/
├── benches/
├── tests/
├── scripts/
├── docs/
└── research/
```

---

# 3. System Architecture

The system consists of 5 layers:

---

# Layer 1 — Storage Layer

Responsibilities:

* GGUF loading
* SafeTensors loading
* mmap file streaming
* metadata parsing
* quant block interpretation

Requirements:

* zero-copy where possible
* aligned memory reads
* sequential access

---

# Layer 2 — Tensor Layer

Responsibilities:

* tensor abstraction
* shape metadata
* stride logic
* views/slices
* quantized tensor representations

Constraints:

* no hidden allocations
* no dynamic reshaping during inference

---

# Layer 3 — Kernel Layer

Responsibilities:

* GEMM
* GEMV
* softmax
* RMSNorm
* rotary embeddings
* quant dequant
* fused operations

This is the performance heart.

---

# Layer 4 — Runtime Layer

Responsibilities:

* transformer execution
* KV cache
* scheduler
* thread pool
* token generation

---

# Layer 5 — Intelligence Layer

Responsibilities:

* inference-time scaling
* self-consistency
* recursive reasoning
* verifier loops
* best-of-N
* confidence routing

---

# 4. Memory Architecture

## 4.1 Unified Arena

The engine allocates:

* one giant contiguous arena.

Everything lives inside:

* weights,
* activations,
* scratch buffers,
* KV cache.

No runtime mallocs allowed during inference.

---

## 4.2 Alignment Rules

All major buffers:

* 64-byte aligned.

Reasons:

* cache-line alignment,
* SIMD loading,
* predictable prefetch behavior.

---

## 4.3 Tape Execution Model

Layer outputs directly feed:
next layer inputs.

No copies.

Memory should behave like:
a conveyor belt.

---

## 4.4 Huge Pages

Experimental support:

* 2MB huge pages,
* TLB optimization.

Only enabled after correctness stabilization.

---

# 5. Quantization System

---

# Supported Formats

## Initial

* FP32
* FP16
* Q4_0

## Mid-Term

* Q4_K
* GPTQ-style blocks
* NF4

## Experimental

* INT2
* ternary
* binary

---

# Quantization Design Rules

## DO:

* dequantize inside registers
* fuse dequant + multiply
* use block scales

## NEVER:

* expand entire tensors to FP32
* allocate dequant buffers

---

# 6. SIMD Rules

---

# x86_64

Allowed:

* AVX2
* FMA3

Forbidden:

* AVX512

---

# ARM64

Future:

* NEON
* SVE/SVE2

---

# SIMD Philosophy

SIMD code must:

* minimize shuffle operations,
* maximize FMA density,
* avoid branch divergence.

---

# 7. Threading Model

---

# Hard Rules

Initial target:

* exactly 2 physical worker threads.

Pinned:

* core 0,
* core 1.

No oversubscription.

---

# Scheduler Philosophy

Goal:

* avoid synchronization overhead.

Preferred:

* lock-free queues,
* atomic coordination,
* static partitioning.

---

# False Sharing Prevention

Thread-local buffers:

* 64-byte aligned,
* separated by cache-line padding.

---

# 8. Kernel Development Roadmap

---

# Phase 1 — Correctness

Implement:

* naive GEMM
* naive GEMV

Triple-loop reference implementations.

Priority:

* correctness only.

---

# Phase 2 — Memory Locality

Implement:

* transposed B layout
* row-major optimization

Goal:

* sequential memory access.

---

# Phase 3 — Blocking/Tiling

Implement:

* cache-aware tiles
* L1-sized blocks
* L2-sized macro tiles

Initial targets:

* 32x32
* 64x64

---

# Phase 4 — SIMD

Implement:

* AVX2 vector paths
* fused multiply-add loops
* manual unrolling

---

# Phase 5 — Quantized Kernels

Implement:

* fused dequant + GEMV
* Q4 block decode
* scale application in registers

---

# Phase 6 — Parallelism

Implement:

* thread pool
* work partitioning
* pinned execution

---

# 9. Transformer Runtime Roadmap

---

# Stage 1

Implement:

* embeddings
* RMSNorm
* linear layers

---

# Stage 2

Implement:

* attention
* rotary embeddings
* KV cache

---

# Stage 3

Implement:

* FFN
* SwiGLU
* multi-layer execution

---

# Stage 4

Implement:

* streaming generation
* token sampling

---

# Stage 5

Implement:

* speculative decoding
* paged KV
* prefix caching

---

# 10. Inference-Time Scaling System (TTS)

This is a core research pillar.

---

# Modes

## Fast Mode

Single-pass generation.

Lowest latency.

---

## Smart Mode

Best-of-N generation.

---

## Deep Reasoning Mode

Entropy-triggered recursive refinement.

---

# Entropy Logic

At token generation:

Compute:

H(T) = -Σ P(x) log P(x)

If entropy is high:

* do not emit token immediately,
* reprocess hidden states,
* refine reasoning.

---

# Recursive Refinement

Potential implementations:

* replay upper transformer layers,
* hidden-state recurrence,
* verifier loops,
* self-consistency voting.

---

# Constraints

Recursive loops must:

* have hard compute budgets,
* avoid infinite loops,
* maintain deterministic state tracking.

---

# 11. Benchmarking Philosophy

Benchmarks are mandatory.

Every optimization requires:

* before/after metrics,
* correctness validation,
* cache analysis.

---

# Metrics

## Core

* tokens/sec
* latency/token
* RAM usage
* cache misses
* branch mispredicts

## Quality

* perplexity
* MMLU
* GSM8K
* HumanEval

---

# Benchmark Targets

Compare against:

* llama.cpp
* MLX
* Ollama
* vLLM CPU paths

---

# 12. Profiling Stack

Linux tools:

* perf
* flamegraph
* valgrind
* cachegrind

Key measurements:

* L1 misses
* L2 misses
* DRAM stalls
* branch misses

---

# 13. Research Targets

Potential publishable areas:

---

## 13.1 CPU Recursive Inference Scaling

Can recursive latent refinement improve small-model reasoning?

---

## 13.2 Register-Level Quant Fusion

Can fused INT4 dequant outperform existing llama.cpp kernels?

---

## 13.3 Ultra-Low-Core Optimization

Can 2-core systems remain competitive through cache-first design?

---

## 13.4 Entropy-Aware Dynamic Compute

Can token uncertainty dynamically allocate compute budgets?

---

# 14. Engineering Rules

---

# NEVER

* use GPU code
* depend on CUDA
* use PyTorch runtime
* allocate tensors during inference
* expand quant tensors globally

---

# ALWAYS

* benchmark changes
* profile kernels
* validate correctness
* optimize memory locality first
* keep data contiguous

---

# 15. Initial Development Sequence

DO NOT SKIP STEPS.

---

# Step 1

Build:

* Tensor struct
* Arena allocator

---

# Step 2

Build:

* GGUF parser

---

# Step 3

Build:

* naive GEMM

---

# Step 4

Build:

* benchmark harness

---

# Step 5

Optimize:

* transpose
* tiling
* SIMD

---

# Step 6

Build:

* quantized kernels

---

# Step 7

Build:

* minimal transformer block

---

# Step 8

Build:

* token generation loop

---

# Step 9

Build:

* inference-time scaling system

---

# 16. AI Agent Instructions

When generating code:

## Priorities

1. correctness
2. memory locality
3. deterministic execution
4. SIMD optimization
5. threading

---

# Required Coding Style

* idiomatic Rust
* minimal dependencies
* explicit memory layouts
* no hidden allocations
* no macro-heavy abstractions

---

# SIMD Rules

Allowed:

* `core::arch::x86_64`
* `__m256`
* `_mm256_fmadd_ps`

Forbidden:

* AVX512 intrinsics

---

# Benchmarking Mandate

Every kernel modification:
must include:

* correctness test
* benchmark
* performance comparison

---

# 17. Long-Term Vision

Project Aether aims to become:

* the fastest CPU-native inference runtime for low/mid hardware,
* a research platform for inference-time intelligence scaling,
* and a fully local frontier-grade reasoning engine.

The final objective is not merely speed.

It is:
maximizing intelligence per watt,
per byte,
and per core.

    