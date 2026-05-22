CPU-First LLM Inference: Research Survey and Roadmap
Building a CPU-optimized LLM engine from scratch demands understanding memory hierarchies, quantization, and CPU instructions. Recent research shows CPUs can match or exceed GPUs for LLM inference if carefully tuned: for example, on a 1B parameter model a 2-thread CPU (FP16) ran 17 tokens/sec versus 12.8 tps on GPU
. Key factors are memory bandwidth and cache use – the CPU avoids costly GPU transfers and with proper threading can win.

Below is a detailed outline of best practices and findings, drawn from the latest academic papers and open-source engines, to guide a ground-up CPU LLM engine design. Citations back up the crucial points.

1. Memory & Data Layout
Unified Contiguous Buffer: Allocate one large memory block for all weights, activations and scratch space, rather than many small allocations. Place data in the exact order of execution (layer0 weights, layer0 outputs, layer1 weights, …). This “tape” layout makes access predictable and cache-friendly. (The advice from recent work is to fuse all model data into one contiguous region so the CPU prefetcher hits every time.)

Alignment & Pages: Align data on 64-byte boundaries (cache lines) and use huge pages (e.g. 2 MB) if possible. With 64-byte alignment, loading one float brings ~16 floats “for free” in L1 cache. Huge pages minimize TLB misses for large models.

Cache-Chain Execution: Ensure that Layer N’s output is exactly at the address that Layer N+1 expects as input. In practice this means no extra copying – the result buffer of one layer seamlessly becomes the input buffer of the next. This creates a “perfect cache chain,” so data stays in L1/L2 cache across layers
.

Avoid Fragmentation: Do not scatter tensors with independent allocations. For example, llama.cpp already contiguously stores KV-caches to avoid page faults. Similarly, pre-allocate space for even the KV cache to avoid dynamic allocation mid-inference. A linear arena or custom allocator that never splits blocks is ideal.

Design note: Think of memory like a conveyor belt in a factory. Everything moves forward in a straight line; nothing loops back. This is the opposite of letting code “malloc” arbitrarily during inference, which creates cache misses.

2. Quantization & Numerical Precision
Weight-Only Quantization (WOQ): Reduce weight precision to 4-bit (or even lower) to cut memory bandwidth. Recent Intel research shows that converting weights to INT4 (with FP16 activations) drastically reduces memory load
. In practice, use k-quants (LLama.cpp’s 4-bit scheme) or GPTQ to quantize weights with minimal accuracy loss.

4-bit Q4 Gains: Studies report 1.5×–2.5× speedups when moving from 16-bit to 4-bit weights on CPU
. Quantizing from FP16 to Q4 roughly doubles inference throughput on CPU (and narrows the gap to GPU)
. (The GPU also speeds up with Q4, but CPU multi-threading benefits more.)

Tradeoffs: Lower bits (2-bit, ternary, etc.) offer even more memory savings, but with greater accuracy loss and engineering complexity. Some recent work explores extreme quantization (1-bit weights with retention of key distribution via outlier quantization)
. That’s a possible future enhancement if accuracy drop is tolerable.

Activation Quantization: Optionally quantize activations (GPTQ-lite or quant-aware scaling). However, weight quantization alone yields major gains since weights dominate memory use. For now, focus on weight quant; keep activations in FP16/32 or use FP16/Mixed as needed.

Insight: LLM inference is memory-bandwidth-bound. If you can cut memory traffic by half (4-bit weights), you literally double the data that fits in cache and reduce DRAM fetches
.

3. Compute Kernels and GEMM Optimization
GEMM Dominates: Profile analysis (e.g. in LLAMA-3.2) shows matrix multiplications (GEMM) take ~80–87% of CPU time during inference
. Concretely, Llama-3.2 1B spent ~87.6% of prefill time in GEMM. Thus, GEMM is the number-one target for speed.

Naive GEMM Baseline: Start with a correct triple-loop implementation:

cpp
Copy
for i in rows(A):
  for j in cols(B):
    acc = 0
    for k in range(common):
       acc += A[i][k] * B[k][j]
    C[i][j] = acc
This verifies correctness and serves as a baseline.

Memory Access: In row-major layout, reading A’s row is cache-friendly, but reading B’s column is not. A common trick is transpose B beforehand, so both A-row and B-row (originally columns) are sequential in memory. This single change yields a noticeable speedup.

Blocking/Tiling: Instead of full matrices, operate on small blocks that fit in L1/L2 cache. For example, divide matrices into 64×64 tiles and multiply tile-by-tile. Each tile multiply fits in L2, vastly reducing L1 misses. This “cache tiling” often yields order-of-magnitude improvements in practice. (All high-performance BLAS and inference engines use tiling.)

SIMD Vectorization: Leverage SIMD instructions:

On x86: AVX2 (8 floats per 256-bit op) or AVX-512 (16 floats per 512-bit op). Use std::simd or intrinsics (_mm256_fmadd_ps, etc.) to multiply-accumulate multiple elements at once.
On ARM: NEON or SVE, similarly processing 4–8 floats per vector. After tiling is correct, rewrite the inner loop to handle a SIMD-width chunk at a time. Intel’s WOQ article describes INT4 kernel logic: dequantize to FP16 in registers and perform FMA
.
Loop Unrolling & FMA: Unroll loops to keep pipelines full and use fused multiply-add (FMA) operations. E.g., process 4 elements of k per iteration. Many compilers will auto-vectorize a well-structured loop; sometimes writing intrinsics or compiler pragmas is needed for maximum effect.

Pruning and Sparsity (Advanced): If acceptable, exploit any sparse structure (pruned weights or block-sparse patterns) to skip zeros. E.g. FlashAttention and SparseGPT show large pruning with little accuracy loss. But sparse GEMM usually benefits large GPUs more. Still, you could consider block-sparse ffn layers if privacy or offline pruning is viable.

Key fact: In top-tier optimization papers, once memory is optimal, further speedups come from GEMM: using AVX512 and careful tiling. For instance, LLaMA inference was sped up significantly by replacing the naive GGML matmul with a cache-blocked, SIMD FMA kernel
.

4. Threading and Parallelism
CPU Thread Scaling: Modern CPUs have many cores (x86 with 8–64, ARM big.LITTLE). Use a thread-pool to execute layer operations in parallel. Benchmarks show speed gains up to a point (often 4–8 threads), then memory bandwidth saturates
. In the 1B model, performance peaked around 4–5 threads.

Avoid Oversubscription: On mobile/ARM, 2–4 threads was optimal. On desktop, 8–16 threads might help, but watch memory contention. Oversubscribing (more threads than logical cores) wastes time.

Work Scheduling: Split work at a coarse level (e.g., different output rows or attention heads per thread) to minimize synchronization. Use lock-free queues or barriers only when necessary. Consider a work-stealing queue for dynamic balancing if layer sizes vary.

False Sharing: Ensure different threads work on separate memory regions. For example, each thread writes to a distinct output block so they don’t clobber each others’ cache lines. Align per-thread stacks and buffers to 64B.

NUMA Awareness: On multi-socket systems, allocate memory local to threads. Pin threads to sockets/cores using sched_setaffinity or similar. (This is advanced; start simple unless your target is server NUMA.)

Insight: Threads are helpful, but only until cache/bandwidth limits. Profiling shows LLM CPU inference becomes memory-bound beyond a few cores
. So multi-threading must be paired with good data locality (blocking) to be effective.

5. Architecture-Specific Optimizations
Instruction Sets: Optimize for your target ISA:

x86_64: Use AVX2 on older chips, AVX-512 on newer (or FMA3). Consider Intel AMX or AVX-512-VNNI if available for quantized workloads.
ARM64: Use NEON intrinsics; if available, SVE (Scalable Vector Extension) or SVE2 for larger vectors. (The iPhone paper specifically suggests using ARM SVE for LLM GEMM
.)
Prefetching: Manually prefetch the next block of data using _mm_prefetch or compiler intrinsics, if hardware prefetch isn’t enough. E.g., prefetch the next row of A or B ahead of time to hide DRAM latency.

Cache-Level Blocking: Tune tile sizes to match L1/L2 sizes. E.g., if L2 is 512KB, a 64×64 float tile (~64KB) fits in L2. Experiment for best performance.

Memory Mapping (MMAP): For loading large weights, consider mmap to avoid copies. Use madvise with WILLNEED to prefetch initial layers if startup latency matters.

6. Inference Pipeline and Additional Strategies
KV-Cache Management: For autoregressive models, manage the KV cache in a contiguous, aligned buffer. Compress or quantize the KV cache if context is long (some engines quantize past keys/values to int4 to save memory). Also consider sliding window or paged attention (like vLLM’s PagedKV) if context > available memory.

Flash Attention / Efficiency Tricks: While full FlashAttention is GPU-centric, the idea of reducing redundant memory operations carries over. For example, avoid writing out attention scores if not needed. Use fused QKV compute if possible (e.g., single large GEMM for QKV instead of 3 separate).

Beam Search / Sampling Techniques: Generally outside “engine core,” but if implementing decoding strategies:

Beam search uses more compute but could improve output quality if beam width >1.
Stochastic sampling (top-p, temperature) is just logic around the final softmax. This part is light compute.
Inference-Time Scaling (Self-Consistency, Best-of-N): If quality is paramount and latency can be sacrificed, you can sample multiple outputs and choose the best (e.g. using a simple scoring or human-in-the-loop). This is known to improve answers: spending more compute at inference (ensemble of samples) often yields higher accuracy
. For example, running N independent generations of the same prompt and taking a majority vote or scoring with another model is a known technique (often called self-consistency). Note: This is not a CPU-engine optimization per se, but rather a mode (“slow-but-smart mode”) you could support.

Trade-off: We can add a “power” mode that reruns the model multiple times to boost answer quality. A recent survey notes major LLM providers use such inference-time scaling (e.g. run multiple chains-of-thought) to improve correctness
. But each extra sample costs N× latency.

7. Implementation Roadmap
Given the above, a phased approach is prudent:

Language & Formats:
We recommend Rust (or C++ if more comfortable) for full control and performance. Rust has std::simd and safe memory handling. Support GGUF or SafeTensors weights (both are contiguous format on disk). Starting with GGUF (used by llama.cpp) is fine.

Core Infrastructure:

Write a Tensor struct: holds a pointer (or Vec<f32>) plus shape (rows,cols). Provide get(i,j)/set(i,j) via i*cols+j.
Implement a MemoryArena: a single Vec<u8> with 64B alignment. It dishes out offsets for each Tensor, with no ability to free.
Parser for GGUF/Safetensors: load weights into the arena sequentially by layers.
Naive GEMM:

Code a straightforward matrix multiply (FP32 or FP16) using the Tensor struct.
Write unit tests (e.g. 2×2, 3×5 mats) to verify correctness.
Optimization Steps:

Transpose Trick: After naive version, implement Bᵀ approach and measure. Expect ~2× speed-up for large mats due to cache.
Blocking: Implement blocked GEMM (tile matrices by fixed block sizes). Benchmark vs naive.
SIMD: Rewrite innermost loops with SIMD (e.g. Rust std::simd or C++ intrinsics). Profile again – should see 4×–8× boost per core (depending on width).
Parallelize: Use a thread pool to divide outer loops across cores. Test scaling up threads.
Minimal Model Inference:

Build a tiny model runner: e.g., a 2-layer MLP (Linear -> ReLU -> Linear). Allocate weights and biases in the arena in execution order.
Run inference on random input; validate vs a known result.
Profile time: GEMM should be dominant. Optimize until overhead is minimal.
Transformer Block (LLaMA-like):

Implement (iteratively): token embedding lookup, single-head attention (QKV gemms + softmax + weighted sum + out gemm), FFN (two GEMMs with activation).
Optimize each part. Attention usually less time than FFN, but attention has memory motion for K/V cache.
Manage a rolling KV cache buffer to avoid recomputing past tokens.
Benchmark and Compare:

Test against PyTorch/llama.cpp with same model (on same CPU).
Measure tokens/sec and memory. Aim for measurable gains from each optimization stage.
Use real CPU profilers (e.g., Linux perf) to spot stalls or cache-misses.
Advanced Modes:

Re-run Mode: Implement an option to sample multiple times or run chains-of-thought. Useful for “power mode” at cost of speed.
Quantized GEMM: Add INT4 weight support. Pre-load int4 blocks, dequantize in registers per tile (like Intel’s scheme
).
Huge Pages: For Linux, experiment with mmap and MAP_HUGETLB. (Ineffectual on all OS, but can cut TLB misses.)
Publication and Tuning:

If results are exceptional (say >2× better than standard engines on CPU), consider writing up.
Throughout, compare to known limits: e.g. SNLI benchmarks or inference challenges.
8. Key Takeaways from Research
Cache-Mindset: The biggest leaps come from data locality. Arrange weights/activations so that sequential memory reads are maximized
.
Quantization is Critical: 4-bit (and weight-only) quantization is a proven high-gain strategy
. Almost all modern CPU inference engines use it.
Parallel CPU Can Win: With careful threading, moderate CPU models (0.5–3B) can surpass GPUs on low batches
. This breaks the “GPU always wins” myth.
GEMM is King: Over 80% of time is in matmuls, so invest engineering there. Use blocking+SIMD+FMA to extract every cycle
.
Progressive Scaling: Don’t attempt full 50B model at once. Validate each optimization on small models first.
Inference vs Accuracy: Don’t ignore inference-time scaling (self-consistency, best-of-N) if your goal is ultimate output quality. It’s computationally expensive but can turn a small model into a “stronger” one in terms of answers
.
By following this blueprint, you’ll build a purpose-built CPU inference engine. It will take time, but each step is verifiable. In the end you’ll know exactly how data flows through the chip, giving maximal efficiency – and possibly results worthy of publication.

Sources: Academic benchmarks and engineering reports (Texas A&M et al. 2025
, Intel AI Developer blog 2024
) underline the importance of memory-locality, quantization, and SIMD for CPU LLMs. These findings directly shaped the plan above.


CPU-First LLM Inference: Research Survey and Roadmap
Building a CPU-optimized LLM engine from scratch demands understanding memory hierarchies, quantization, and CPU instructions. Recent research shows CPUs can match or exceed GPUs for LLM inference if carefully tuned: for example, on a 1B parameter model a 2-thread CPU (FP16) ran 17 tokens/sec versus 12.8 tps on GPU
. Key factors are memory bandwidth and cache use – the CPU avoids costly GPU transfers and with proper threading can win.

Below is a detailed outline of best practices and findings, drawn from the latest academic papers and open-source engines, to guide a ground-up CPU LLM engine design. Citations back up the crucial points.

1. Memory & Data Layout
Unified Contiguous Buffer: Allocate one large memory block for all weights, activations and scratch space, rather than many small allocations. Place data in the exact order of execution (layer0 weights, layer0 outputs, layer1 weights, …). This “tape” layout makes access predictable and cache-friendly. (The advice from recent work is to fuse all model data into one contiguous region so the CPU prefetcher hits every time.)

Alignment & Pages: Align data on 64-byte boundaries (cache lines) and use huge pages (e.g. 2 MB) if possible. With 64-byte alignment, loading one float brings ~16 floats “for free” in L1 cache. Huge pages minimize TLB misses for large models.

Cache-Chain Execution: Ensure that Layer N’s output is exactly at the address that Layer N+1 expects as input. In practice this means no extra copying – the result buffer of one layer seamlessly becomes the input buffer of the next. This creates a “perfect cache chain,” so data stays in L1/L2 cache across layers
.

Avoid Fragmentation: Do not scatter tensors with independent allocations. For example, llama.cpp already contiguously stores KV-caches to avoid page faults. Similarly, pre-allocate space for even the KV cache to avoid dynamic allocation mid-inference. A linear arena or custom allocator that never splits blocks is ideal.

Design note: Think of memory like a conveyor belt in a factory. Everything moves forward in a straight line; nothing loops back. This is the opposite of letting code “malloc” arbitrarily during inference, which creates cache misses.

2. Quantization & Numerical Precision
Weight-Only Quantization (WOQ): Reduce weight precision to 4-bit (or even lower) to cut memory bandwidth. Recent Intel research shows that converting weights to INT4 (with FP16 activations) drastically reduces memory load
. In practice, use k-quants (LLama.cpp’s 4-bit scheme) or GPTQ to quantize weights with minimal accuracy loss.

4-bit Q4 Gains: Studies report 1.5×–2.5× speedups when moving from 16-bit to 4-bit weights on CPU
. Quantizing from FP16 to Q4 roughly doubles inference throughput on CPU (and narrows the gap to GPU)
. (The GPU also speeds up with Q4, but CPU multi-threading benefits more.)

Tradeoffs: Lower bits (2-bit, ternary, etc.) offer even more memory savings, but with greater accuracy loss and engineering complexity. Some recent work explores extreme quantization (1-bit weights with retention of key distribution via outlier quantization)
. That’s a possible future enhancement if accuracy drop is tolerable.

Activation Quantization: Optionally quantize activations (GPTQ-lite or quant-aware scaling). However, weight quantization alone yields major gains since weights dominate memory use. For now, focus on weight quant; keep activations in FP16/32 or use FP16/Mixed as needed.

Insight: LLM inference is memory-bandwidth-bound. If you can cut memory traffic by half (4-bit weights), you literally double the data that fits in cache and reduce DRAM fetches
.

3. Compute Kernels and GEMM Optimization
GEMM Dominates: Profile analysis (e.g. in LLAMA-3.2) shows matrix multiplications (GEMM) take ~80–87% of CPU time during inference
. Concretely, Llama-3.2 1B spent ~87.6% of prefill time in GEMM. Thus, GEMM is the number-one target for speed.

Naive GEMM Baseline: Start with a correct triple-loop implementation:

cpp
Copy
for i in rows(A):
  for j in cols(B):
    acc = 0
    for k in range(common):
       acc += A[i][k] * B[k][j]
    C[i][j] = acc
This verifies correctness and serves as a baseline.

Memory Access: In row-major layout, reading A’s row is cache-friendly, but reading B’s column is not. A common trick is transpose B beforehand, so both A-row and B-row (originally columns) are sequential in memory. This single change yields a noticeable speedup.

Blocking/Tiling: Instead of full matrices, operate on small blocks that fit in L1/L2 cache. For example, divide matrices into 64×64 tiles and multiply tile-by-tile. Each tile multiply fits in L2, vastly reducing L1 misses. This “cache tiling” often yields order-of-magnitude improvements in practice. (All high-performance BLAS and inference engines use tiling.)

SIMD Vectorization: Leverage SIMD instructions:

On x86: AVX2 (8 floats per 256-bit op) or AVX-512 (16 floats per 512-bit op). Use std::simd or intrinsics (_mm256_fmadd_ps, etc.) to multiply-accumulate multiple elements at once.
On ARM: NEON or SVE, similarly processing 4–8 floats per vector. After tiling is correct, rewrite the inner loop to handle a SIMD-width chunk at a time. Intel’s WOQ article describes INT4 kernel logic: dequantize to FP16 in registers and perform FMA
.
Loop Unrolling & FMA: Unroll loops to keep pipelines full and use fused multiply-add (FMA) operations. E.g., process 4 elements of k per iteration. Many compilers will auto-vectorize a well-structured loop; sometimes writing intrinsics or compiler pragmas is needed for maximum effect.

Pruning and Sparsity (Advanced): If acceptable, exploit any sparse structure (pruned weights or block-sparse patterns) to skip zeros. E.g. FlashAttention and SparseGPT show large pruning with little accuracy loss. But sparse GEMM usually benefits large GPUs more. Still, you could consider block-sparse ffn layers if privacy or offline pruning is viable.

Key fact: In top-tier optimization papers, once memory is optimal, further speedups come from GEMM: using AVX512 and careful tiling. For instance, LLaMA inference was sped up significantly by replacing the naive GGML matmul with a cache-blocked, SIMD FMA kernel
.

4. Threading and Parallelism
CPU Thread Scaling: Modern CPUs have many cores (x86 with 8–64, ARM big.LITTLE). Use a thread-pool to execute layer operations in parallel. Benchmarks show speed gains up to a point (often 4–8 threads), then memory bandwidth saturates
. In the 1B model, performance peaked around 4–5 threads.

Avoid Oversubscription: On mobile/ARM, 2–4 threads was optimal. On desktop, 8–16 threads might help, but watch memory contention. Oversubscribing (more threads than logical cores) wastes time.

Work Scheduling: Split work at a coarse level (e.g., different output rows or attention heads per thread) to minimize synchronization. Use lock-free queues or barriers only when necessary. Consider a work-stealing queue for dynamic balancing if layer sizes vary.

False Sharing: Ensure different threads work on separate memory regions. For example, each thread writes to a distinct output block so they don’t clobber each others’ cache lines. Align per-thread stacks and buffers to 64B.

NUMA Awareness: On multi-socket systems, allocate memory local to threads. Pin threads to sockets/cores using sched_setaffinity or similar. (This is advanced; start simple unless your target is server NUMA.)

Insight: Threads are helpful, but only until cache/bandwidth limits. Profiling shows LLM CPU inference becomes memory-bound beyond a few cores
. So multi-threading must be paired with good data locality (blocking) to be effective.

5. Architecture-Specific Optimizations
Instruction Sets: Optimize for your target ISA:

x86_64: Use AVX2 on older chips, AVX-512 on newer (or FMA3). Consider Intel AMX or AVX-512-VNNI if available for quantized workloads.
ARM64: Use NEON intrinsics; if available, SVE (Scalable Vector Extension) or SVE2 for larger vectors. (The iPhone paper specifically suggests using ARM SVE for LLM GEMM
.)
Prefetching: Manually prefetch the next block of data using _mm_prefetch or compiler intrinsics, if hardware prefetch isn’t enough. E.g., prefetch the next row of A or B ahead of time to hide DRAM latency.

Cache-Level Blocking: Tune tile sizes to match L1/L2 sizes. E.g., if L2 is 512KB, a 64×64 float tile (~64KB) fits in L2. Experiment for best performance.

Memory Mapping (MMAP): For loading large weights, consider mmap to avoid copies. Use madvise with WILLNEED to prefetch initial layers if startup latency matters.

6. Inference Pipeline and Additional Strategies
KV-Cache Management: For autoregressive models, manage the KV cache in a contiguous, aligned buffer. Compress or quantize the KV cache if context is long (some engines quantize past keys/values to int4 to save memory). Also consider sliding window or paged attention (like vLLM’s PagedKV) if context > available memory.

Flash Attention / Efficiency Tricks: While full FlashAttention is GPU-centric, the idea of reducing redundant memory operations carries over. For example, avoid writing out attention scores if not needed. Use fused QKV compute if possible (e.g., single large GEMM for QKV instead of 3 separate).

Beam Search / Sampling Techniques: Generally outside “engine core,” but if implementing decoding strategies:

Beam search uses more compute but could improve output quality if beam width >1.
Stochastic sampling (top-p, temperature) is just logic around the final softmax. This part is light compute.
Inference-Time Scaling (Self-Consistency, Best-of-N): If quality is paramount and latency can be sacrificed, you can sample multiple outputs and choose the best (e.g. using a simple scoring or human-in-the-loop). This is known to improve answers: spending more compute at inference (ensemble of samples) often yields higher accuracy
. For example, running N independent generations of the same prompt and taking a majority vote or scoring with another model is a known technique (often called self-consistency). Note: This is not a CPU-engine optimization per se, but rather a mode (“slow-but-smart mode”) you could support.

Trade-off: We can add a “power” mode that reruns the model multiple times to boost answer quality. A recent survey notes major LLM providers use such inference-time scaling (e.g. run multiple chains-of-thought) to improve correctness
. But each extra sample costs N× latency.

7. Implementation Roadmap
Given the above, a phased approach is prudent:

Language & Formats:
We recommend Rust (or C++ if more comfortable) for full control and performance. Rust has std::simd and safe memory handling. Support GGUF or SafeTensors weights (both are contiguous format on disk). Starting with GGUF (used by llama.cpp) is fine.

Core Infrastructure:

Write a Tensor struct: holds a pointer (or Vec<f32>) plus shape (rows,cols). Provide get(i,j)/set(i,j) via i*cols+j.
Implement a MemoryArena: a single Vec<u8> with 64B alignment. It dishes out offsets for each Tensor, with no ability to free.
Parser for GGUF/Safetensors: load weights into the arena sequentially by layers.
Naive GEMM:

Code a straightforward matrix multiply (FP32 or FP16) using the Tensor struct.
Write unit tests (e.g. 2×2, 3×5 mats) to verify correctness.
Optimization Steps:

Transpose Trick: After naive version, implement Bᵀ approach and measure. Expect ~2× speed-up for large mats due to cache.
Blocking: Implement blocked GEMM (tile matrices by fixed block sizes). Benchmark vs naive.
SIMD: Rewrite innermost loops with SIMD (e.g. Rust std::simd or C++ intrinsics). Profile again – should see 4×–8× boost per core (depending on width).
Parallelize: Use a thread pool to divide outer loops across cores. Test scaling up threads.
Minimal Model Inference:

Build a tiny model runner: e.g., a 2-layer MLP (Linear -> ReLU -> Linear). Allocate weights and biases in the arena in execution order.
Run inference on random input; validate vs a known result.
Profile time: GEMM should be dominant. Optimize until overhead is minimal.
Transformer Block (LLaMA-like):

Implement (iteratively): token embedding lookup, single-head attention (QKV gemms + softmax + weighted sum + out gemm), FFN (two GEMMs with activation).
Optimize each part. Attention usually less time than FFN, but attention has memory motion for K/V cache.
Manage a rolling KV cache buffer to avoid recomputing past tokens.
Benchmark and Compare:

Test against PyTorch/llama.cpp with same model (on same CPU).
Measure tokens/sec and memory. Aim for measurable gains from each optimization stage.
Use real CPU profilers (e.g., Linux perf) to spot stalls or cache-misses.
Advanced Modes:

Re-run Mode: Implement an option to sample multiple times or run chains-of-thought. Useful for “power mode” at cost of speed.
Quantized GEMM: Add INT4 weight support. Pre-load int4 blocks, dequantize in registers per tile (like Intel’s scheme
).
Huge Pages: For Linux, experiment with mmap and MAP_HUGETLB. (Ineffectual on all OS, but can cut TLB misses.)
Publication and Tuning:

If results are exceptional (say >2× better than standard engines on CPU), consider writing up.
Throughout, compare to known limits: e.g. SNLI benchmarks or inference challenges.
8. Key Takeaways from Research
Cache-Mindset: The biggest leaps come from data locality. Arrange weights/activations so that sequential memory reads are maximized
.
Quantization is Critical: 4-bit (and weight-only) quantization is a proven high-gain strategy
. Almost all modern CPU inference engines use it.
Parallel CPU Can Win: With careful threading, moderate CPU models (0.5–3B) can surpass GPUs on low batches
. This breaks the “GPU always wins” myth.
GEMM is King: Over 80% of time is in matmuls, so invest engineering there. Use blocking+SIMD+FMA to extract every cycle
.
Progressive Scaling: Don’t attempt full 50B model at once. Validate each optimization on small models first.
Inference vs Accuracy: Don’t ignore inference-time scaling (self-consistency, best-of-N) if your goal is ultimate output quality. It’s computationally expensive but can turn a small model into a “stronger” one in terms of answers
.
By following this blueprint, you’ll build a purpose-built CPU inference engine. It will take time, but each step is verifiable. In the end you’ll know exactly how data flows through the chip, giving maximal efficiency – and possibly results worthy of publication.

Sources: Academic benchmarks and engineering reports (Texas A&M et al. 2025
, Intel AI Developer blog 2024
) underline the importance of memory-locality, quantization, and SIMD for CPU LLMs. These findings directly shaped the plan above.

# CLAUDE.md — System-Level Context & AI Instructions (X270 Optimized)

## 1. System Vision & Architecture
Clean-room, zero-dependency CPU-first LLM inference engine optimized specifically for the Intel Kaby Lake architecture (Dual-Core, Single-Channel Memory, AVX2/FMA3 enabled).

### Ground Rules:
1. Hard target: x86_64 with AVX2 and FMA3 extensions. Eliminate all AVX-512 logic.
2. Threading: Exactly 2 persistent worker threads pinned to physical cores. Zero lock overhead.
3. Quantization focus: Aggressive 4-bit (Q4_0) and 2-bit weight-only schemes to counter single-channel RAM bottlenecks.

## 2. Kernel Design Constraints
* Vector Width: 256-bit using `__m256` data types.
* Tiling Strategy: Block sizes must strictly align with the 32 KB L1 Data Cache. Max tile footprint should not exceed 16 KB to leave room for intermediate activations.
* Fused Unpacking: Read bit-packed INT4 arrays straight into YMM registers, apply bitwise masks/shifts to convert to floating-point representation, apply scale factors, and execute `_mm256_fmadd_ps` instantly.
# CLAUDE.md — System-Level Context & AI Instructions (Rust / X270 Optimized)

## 1. System Vision & Technical Architecture

This file serves as the core blueprint, operational guardrails, and persistent context for building a zero-dependency, clean-room CPU-first LLM inference engine from scratch in **Rust**. The engine is strictly optimized for the unique constraints of the **Intel Kaby Lake (ThinkPad X270 / i7)** architecture, leveraging hardware boundaries as design mechanics to achieve record-level inference speeds and low-latency token execution.

### Target Hardware Profile (Intel Core i7 Kaby Lake)

* **Physical Compute Core Structure:** 2 Physical Cores / 4 Execution Threads.
* **SIMD Instruction Support:** x86_64 with AVX2 and FMA3 extensions. **Strictly NO AVX-512.**
* **Cache Hierarchy:** L1 Data Cache: 32 KB per core | L2 Cache: 256 KB per core | L3 Cache: 4 MB shared.
* **Memory Architecture Limitation:** Single-Channel DDR4 RAM configuration (~19.2 – 21.3 GB/s theoretical max bandwidth limit).

### Core Engine Design Pillars

1. **Register-Level Dequantization Fusion:** Counteract the single-channel RAM bottleneck by streaming compressed INT4/INT2/Ternary blocks into CPU registers and unpacking them directly inside 256-bit vector registers (`__m256`). Unquantized weights must never touch main memory or L2/L3 caches.
2. **Zero-Allocation Contiguous Memory Tape:** Allocate a single, monolithic, 64-byte aligned memory arena at startup. Layout weights, activation buffers, and the KV cache sequentially in exact order of execution to create a perfect cache chain.
3. **Hard Affinity Physical Threading:** Execute using exactly 2 worker threads pinned directly to physical CPU core 0 and core 1. Avoid hyper-threaded logical paths to prevent pipeline thrashing and cache evictions.
4. **System-2 Test-Time Scaling (TTS) Loop:** Build a native execution state machine that intercepts activation tensors, evaluates token entropy, and recursively loops hidden states through the transformer stack to scale compute dynamically for complex reasoning tasks.

---

## 2. Low-Level Execution Constraints & Mathematics

### Memory Hierarchy & Contiguous Tape Mechanics

* **Conveyor Belt Allocations:** All layer execution resources are arranged as a linear tape. The output pointer of Layer $N$ serves exactly as the input pointer of Layer $N+1$. No runtime allocations (`alloc`/`malloc`) or data-copying operations are permitted during the inference lifecycle.
* **Data Alignment:** All pointer offsets must be aligned to 64-byte boundaries (matching CPU cache line widths) using custom Rust arena offsets.
* **Memory Mapping:** Use the `memmap2` crate to memory-map GGUF or Safetensors files directly into the execution arena. Utilize `madvise` with `WILLNEED` configurations to minimize kernel page-fault penalties during execution.

### Vector Math & Intrinsic Register Packing

* **Target Vector Width:** 256-bit wide registers using Rust's `core::arch::x86_64` safe intrinsic blocks.
* **GEMV Dominance:** Autoregressive decode phase execution is entirely matrix-vector multiplication ($GEMV$). Weights are parsed directly from memory as packed 4-bit sequences (`Q4_0` structure style: 32 blocks of weights packed with an initial 32-bit float scale factor).
* **Fused Register Unpacking Equation:** For a 4-bit packed weight vector $W_q$, dequantize on-the-fly inside the register lane using bitwise logical shifts and masks before blending with scale factors $\gamma$ and biases $\beta$:

$$W = \gamma \cdot (W_q) + \beta$$


* **FMA3 Optimization Loop:** Process unrolled compute loops inside 256-bit blocks utilizing `_mm256_fmadd_ps(a, b, c)` executing 8 single-precision floating-point operations in a single clock cycle.
* **L1 Cache Line Tiling Constraints:** Matrix tiles must be sized to prevent cache pollution. Limit active math tiles to a maximum footprint of **16 KB** to leave the remaining 16 KB of the 32 KB L1 data cache open for input tracking vectors and quantization scale metadata.

### Memory Layout Matrix (X270 Tailored)

```
[ DDR4 RAM Single-Channel ] ── (Streams Packed INT4 Weights once per token) ──► [ CPU Registers (YMM0 - YMM15) ]
                                                                                         │
   ┌─────────────────────────────────────────────────────────────────────────────────────┘
   ▼
[ Register Operations ] ──► Unpack 4-bit to FP32 natively via shifts
                        ──► Load FP16/FP32 Activations from L1 Cache (16KB Tiles)
                        ──► Execute Fused Multiply-Add (_mm256_fmadd_ps)
                        ──► Stream out directly to next layer's execution head

```

---

## 3. The System-2 Test-Time Scaling (TTS) Reasoning Loop

To bridge the gap between small models (0.5B – 3B parameters) and massive frontier systems on your i7 processor, the engine implements an automated internal feedback routine that Trades Latency for Reasoning Prowess.

### Token Entropy Evaluation & Verification State Machine

At the final layer before projecting logits to the output sequence, the engine interceptor calculates the Shannon entropy $H(T)$ of the predicted probability distribution:


$$H(T) = -\sum_{i=1}^{V} P(x_i) \log_2 P(x_i)$$

* **Low Entropy ($H(T) < \tau_{\text{low}}$):** High token confidence. The engine instantly emits the token via the API stream and moves to the next autoregressive step.
* **High Entropy ($H(T) \ge \tau_{\text{high}}$):** High token ambiguity. The engine stalls sequence emissions and transitions to **Latent Recurrent Refinement Mode**.

### Latent Recurrent Refinement Protocol

Instead of sampling an ambiguous token, the activation hidden state $H_{\text{state}}$ is routed recursively backwards into a subset of intermediate Transformer layers (e.g., repeating layers $L/2$ to $3L/4$).

1. **Context Injection:** The activation sequence is combined with an implicit "thinking block" state token structure allocated in the scratchpad area of the contiguous memory buffer.
2. **State Evolution Loop:** The hidden states are processed recursively through the execution network up to a hard configuration limit $N_{\text{loops}}$ or until token prediction entropy drops below the acceptance threshold $\tau_{\text{low}}$.
3. **KV-Cache Decoupling:** Thinking tokens generated during internal refinement loops are retained inside a dedicated, isolated partition of the contiguous KV arena to prevent corruption of the global sequence token history.

---

## 4. Rust Phased Implementation Roadmap

### Phase 1: Bare-Metal Architecture & Unified Arena

* Define the rigid `Tensor` struct holding shapes, strides, data pointers, and enum formats (`FP32`, `FP16`, `Q4_0`).
* Implement a zero-allocation `MemoryArena` backed by a singular `Vec<u8>` or raw memory block wrapped in an alignment structure ensuring strict 64-byte alignment boundaries.
* Integrate `memmap2` to implement zero-copy file parsing for GGUF metadata headers and sequential weight offsets.

### Phase 2: Fused AVX2/FMA3 Processing Kernels

* Develop baseline native Rust triple-loop GEMM/GEMV functions to validate calculation accuracy via comprehensive test suites.
* Implement matrix transposition routines to force weight layouts into row-major alignment matching sequential reading pointers.
* Write the optimized fused `Q4_0` execution kernels utilizing `core::arch::x86_64` intrinsics. Block loops into 64x64 or 32x32 tiles matching the 16KB L1 cache capacity rules. Use loop unrolling by unrolling depth vectors 4x per iteration step.

### Phase 3: Lock-Free Bi-Core Thread Pool

* Incorporate thread-pinning using platform-specific hooks or native abstractions (`core_affinity` crate structure logic) to explicitly pin exactly 2 worker threads to CPU Core 0 and Core 1.
* Eliminate traditional mutex patterns. Construct thread synchronization workflows around lock-free atomic atomic boundaries (`AtomicU8` state ping-pong flags) to assign processing rows across attention heads or output channels without cache-line bouncing.

### Phase 4: Transformer Execution Graph & Pipeline Synthesis

* Implement sequential network operations: Token embedding lookups, Single-Head/Multi-Query Attention blocks, Silu/GELU activation primitives, and Feed-Forward Networks (FFN).
* Incorporate an isolated, pre-allocated, continuous KV-cache buffer system that avoids all runtime fragmentation or page-fault spikes.

### Phase 5: Test-Time Scaling (TTS) Routing Logic

* Build out the token entropy tracking block and the feedback loops in the central inference runtime loop.
* Incorporate configurable configuration knobs for execution depth limits, self-correction triggers, and conditional layer skipping.

---

## 5. Agent Behavioral Guardrails & Instructions

When executing commands or writing files inside this workspace using Claude Code, you must adhere strictly to the following execution constraints:

* **Language Locking:** Write 100% standard idiomatic Rust. Use `#![no_std]` style layouts or avoid external dependency bloat. Do not import heavy runtimes like PyTorch bindings, ONNX, or external BLAS abstractions.
* **No Code Walls / Hallucinations:** When producing kernel variations, write the absolute concrete loop implementations. Do not use pseudo-code blocks or leave operational loops as comment stubs.
* **SIMD Compliance Check:** Double-check every single vector intrinsic choice. Never generate or suggest AVX-512 types (`__m512`, `_mm512_add_ps`, etc.). Force all SIMD vectors to 256-bit operations (`__m256`, `_mm256_fmadd_ps`).
* **TDD Micro-benchmarking Mandate:** Prior to editing critical compute kernels or updating file streaming components, you must craft an isolated TDD benchmarking unit test to evaluate computation correctness, memory leaks, and core runtime cycle counts. Validate changes natively by reading performance profiles via standard console tooling.

This is the right direction now.

You are not building “a model.” You are building a **CPU inference system** with three separable layers: a kernel core, a model runtime, and a test-time quality layer. The research points in the same direction: CPU inference is usually bottlenecked by memory bandwidth and GEMM, not raw compute; one recent CPU-only study found matrix multiplication took about 87.6% of prefill time and 76.2% of decode time, while another showed 4-bit quantization could give roughly 1.5×–2.5× speedups and that, for small models, well-threaded CPU execution can beat GPU due to overhead and transfer costs. ([arXiv][1])

So the master plan is this:

First, build a **clean-room Rust inference core** in a separate repo. Do not couple it to Quantumn Code yet. Start with a narrow target: x86_64 AVX2/FMA3 first, then ARM64 NEON/SVE later. The runtime should use one contiguous arena, 64-byte alignment, `mmap`-backed loading, and a strict execution-order memory layout so tensors, scratch buffers, and KV cache all live in a single “tape.” That is the foundational win. The research and Intel’s WOQ guidance both point to the same thing: reduce memory traffic first, then accelerate math inside cache-sized tiles. ([arXiv][2])

Second, build the compute stack in the only sensible order: naive GEMM, then transpose-aware GEMM, then blocked/tiled GEMM, then SIMD, then fused dequant + GEMM for INT4 weights, then thread-pool parallelism. Don’t jump ahead. The reason is simple: if the baseline is wrong, every later optimization becomes noise. The paper trail says this is where the gains are: GEMM dominates, weight-only quantization cuts bandwidth, and small-block int4 dequantization fused with FMA is the CPU-friendly path. ([arXiv][1])

Third, the model runtime should start small and ruthless: embeddings, RMSNorm, attention, MLP, KV cache. Keep the KV cache contiguous and aligned. Later, if you need larger-context support, move to a paged or block-managed KV design; the general idea of paged KV management is established in systems like PagedAttention, which store KV in blocks instead of one huge flat region. That matters once contexts get large enough that naive cache growth hurts memory behavior. ([Wikipedia][3])

Fourth, add a separate **quality mode**. This is not a kernel optimization; it is test-time scaling. The research says inference-time scaling is one of the most effective ways to improve answer quality, and newer work shows self-consistency-style methods can improve reasoning accuracy, sometimes with meaningful latency/token tradeoffs. So your engine should support modes like single-pass, best-of-N, self-consistency, and budgeted rerun. Do not make it an infinite loop. Make it a bounded controller: entropy or confidence decides whether to emit, resample, or verify. ([arXiv][2])

Fifth, make benchmarking a first-class module from day one. Every kernel change gets a microbenchmark. Every model-layer change gets a token/sec benchmark. Every quality-mode change gets an accuracy-vs-latency benchmark. Your success metrics should be explicit: throughput, latency, peak RAM, KV growth, cache-miss behavior, and answer quality on a fixed eval set. Without that, you will “optimize” in circles.

Here is the build order I would lock in for the agent:

1. **Repository skeleton**: Rust workspace, `core`, `kernels`, `runtime`, `formats`, `bench`, `cli`.
2. **Tensor + Arena**: flat tensor abstraction, aligned allocator, offset-based memory plan, no per-layer allocations.
3. **Format loader**: GGUF first, because it is practical for CPU inference; add Safetensors later if needed.
4. **Naive kernels**: GEMM, GEMV, softmax, layernorm, RMSNorm, activation functions.
5. **Optimized kernels**: blocked GEMM, cache-aware layouts, SIMD, fused ops, quantized weight paths.
6. **Transformer runtime**: attention + KV cache + sampling.
7. **Quality controller**: self-consistency, best-of-N, entropy-triggered reruns, budget caps.
8. **Profiling and eval**: perf counters, cache misses, regression tests, quality evals.
9. **Hardware specialization**: AVX2/FMA3 path first, then ARM64 NEON/SVE, then huge pages, thread pinning, NUMA handling.
10. **Publication-grade experiments**: compare against llama.cpp baselines and publish only if the gains are real and repeatable.

The big strategic idea is this: the “greatest” CPU engine will not be the one that does the most things. It will be the one that makes **data movement cheap**, **math dense**, and **quality adjustable by budget**. That is how you chase 40%+ gains without fooling yourself.

One caution: the “run the model in a loop until it becomes frontier” instinct is useful only when you turn it into a bounded inference-time-scaling mode. It can improve answers, but every extra sample costs latency and compute, so the controller has to decide when the extra budget is worth it. The research supports this direction, but not as a free lunch. ([arXiv][2])

Next step: I can turn this into a single agent-ready context file spec for the new repo, with the exact modules, constraints, and acceptance tests.

[1]: https://arxiv.org/abs/2509.13990?utm_source=chatgpt.com "Slim-SC: Thought Pruning for Efficient Scaling with Self-Consistency"
[2]: https://arxiv.org/abs/2603.21301?utm_source=chatgpt.com "enhancing reasoning accuracy in large language models during inference time"
[3]: https://en.wikipedia.org/wiki/PagedAttention?utm_source=chatgpt.com "PagedAttention"
