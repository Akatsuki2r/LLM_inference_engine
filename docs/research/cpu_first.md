# CPU-First LLM Inference Research

CPU inference is bottlenecked by memory bandwidth. Key insights:
1. **GEMM is King**: ~85% of time is spent in GEMM.
2. **Quantization**: Moving from FP16 to Q4_0 cuts memory traffic by 2x-4x, doubling throughput.
3. **Data Locality**: Cache tiling and contiguous memory layouts are the primary drivers of performance on bandwidth-limited CPUs.
4. **Threading**: Performance scales with threads until memory bandwidth saturation. For LFM-1.2B, 2-4 threads is typically optimal for Kaby Lake.
