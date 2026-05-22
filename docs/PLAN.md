# Quantumn Implementation Plan

This document outlines the technical roadmap for building **Quantumn**, a frontier-class CPU LLM inference engine.

## 🎯 Core Objective
Maximize intelligence per watt, per byte, and per core on x86_64 AVX2/FMA3 systems, prioritizing memory locality and deterministic execution.

## 🗺️ The Master Roadmap

### Phase 1: Foundation & Correctness (The Baseline)
- [ ] **Step 1: Memory & Tensors**
    - Implement `Tensor` struct with explicit layout.
    - Implement `Arena` allocator (one giant contiguous block).
    - Ensure 64-byte alignment for all buffers.
- [ ] **Step 2: Storage Layer**
    - Implement GGUF parser.
    - Implement mmap file streaming for zero-copy weight loading.
- [ ] **Step 3: Naive Kernels**
    - Implement triple-loop naive GEMM.
    - Implement triple-loop naive GEMV.
    - **Verification**: Compare against reference Python/PyTorch outputs.

### Phase 2: Performance Optimization (The Speed)
- [ ] **Step 4: Memory Locality**
    - Implement Transposed B layout.
    - Implement row-major optimizations to minimize cache misses.
- [ ] **Step 5: Blocking & Tiling**
    - Implement L1/L2 cache-aware tiles (32x32, 64x64).
    - Implement macro-tiling strategy.
- [ ] **Step 6: SIMD Acceleration**
    - Implement AVX2 vector paths.
    - Implement fused multiply-add (FMA) loops.
    - Manual loop unrolling.

### Phase 3: Quantization & Precision (The Efficiency)
- [ ] **Step 7: Quantized Kernels**
    - Implement fused dequant + GEMV.
    - Implement Q4_0 block decoding.
    - Register-level scale application.
- [ ] **Step 8: Precision Support**
    - Support FP32 and FP16.
    - Implement Q4_K and Q5 formats.

### Phase 4: Runtime & Transformer (The Engine)
- [ ] **Step 9: Basic Transformer Block**
    - Implement Embeddings.
    - Implement RMSNorm.
    - Implement Linear Layers.
- [ ] **Step 10: Attention Mechanism**
    - Implement Rotary Positional Embeddings (RoPE).
    - Implement KV Cache (K-V stores).
- [ ] **Step 11: Feed-Forward Network (FFN)**
    - Implement SwiGLU.
    - Implement multi-layer execution flow.

### Phase 5: Generation & Intelligence (The Reasoning)
- [ ] **Step 12: Token Generation Loop**
    - Implement streaming generation.
    - Implement token sampling (greedy, temperature, top-p).
- [ ] **Step 13: Inference-Time Scaling (TTS)**
    - Implement "Fast Mode" (single-pass).
    - Implement "Smart Mode" (Best-of-N).
    - Implement "Deep Reasoning Mode" (Entropy-triggered recursive refinement).
- [ ] **Step 14: Entropy Logic**
    - Implement $H(T) = -\sum P(x) \log P(x)$ computation.
    - Implement recursive reasoning loops based on entropy thresholds.

## 📈 Verification Gates
Every step must pass these gates before progressing:
1. **Correctness**: Bit-perfect match with reference implementations.
2. **Performance**: Benchmark against `llama.cpp` CPU paths.
3. **Profiling**: Verify L1/L2 cache miss reduction via `perf`/`cachegrind`.
4. **Alignment**: Verify 64-byte alignment via memory dumps.
