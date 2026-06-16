# GEMM Benchmarks

The core performance of Quantumn is bottlenecked by Matrix Multiplication (GEMM). We use tiled, AVX2-accelerated kernels.

## Performance Metrics

| Operation | Latency (avg) | Notes |
| :--- | :--- | :--- |
| **64x64 GEMM (AVX2)** | ~837 µs | 4x Unrolled, 32x32 Tiling |

Measured on: Intel Core i7 Kaby Lake.
