#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m256, _mm256_load_ps, _mm256_store_ps, _mm256_setzero_ps,
    _mm256_fmadd_ps, _mm256_broadcast_ss
};
use aether_tensor::{TensorView, TensorMut, TensorError};
use crate::KernelError;

/// High-performance AVX2 + FMA3 accelerated GEMM engine.
pub struct AvxEngine;
impl AvxEngine {
    /// High-performance Vectorized GEMM utilizing AVX2 + Fused Multiply-Add (FMA3).
    /// Assumes strict 64-byte pointer boundaries verified from the UnifiedArena.
    ///
    /// # Safety
    /// This function uses unsafe AVX2 intrinsics and requires that the input tensors
    /// are properly aligned and within bounds. The caller must ensure that the
    /// hardware supports AVX2 and FMA3 instructions.
    pub unsafe fn gemm_avx2(a: &TensorView, b: &TensorView, c: &mut TensorMut) -> Result<(), KernelError> {
        let shape_a = a.shape();
        let shape_b = b.shape();

        let m = shape_a[0];
        let k = shape_a[1];
        let n = shape_b[1];

        // Check tensor dimensions
        if shape_a.len() != 2 || shape_b.len() != 2 {
            return Err(KernelError::DimensionMismatch);
        }
        if k != shape_b[0] {
            return Err(KernelError::DimensionMismatch);
        }

        // Ensure the hardware explicitly supports AVX2 and FMA instructions before executing raw assembly
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                return Self::execute_core(a, b, c, m, k, n);
            }
        }

        Err(KernelError::DimensionMismatch) // Fallback or architecture unsupported error code
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn execute_core(
        a: &TensorView,
        b: &TensorView,
        c: &mut TensorMut,
        m: usize,
        k: usize,
        n: usize
    ) -> Result<(), KernelError> {
        let a_ptr = a.data().as_ptr();
        let b_ptr = b.data().as_ptr();

        // Step 4 Tiling Panels: process output in chunks of 8 columns (matching the 256-bit vector width)
        for row in 0..m {
            let mut col = 0;
            while col < n {
                if col + 8 <= n {
                    // Accumulate into a 256-bit register (8 packed floats initialized to 0.0)
                    let mut c_vector: __m256 = _mm256_setzero_ps();

                    for i in 0..k {
                        // 1. Broadcast a single scalar element of matrix A across all 8 vector lanes
                        let val_a = *a_ptr.add(row * k + i);
                        let a_broadcast = _mm256_broadcast_ss(&val_a);

                        // 2. Load 8 contiguous elements from Matrix B (utilizing aligned memory access)
                        let b_idx = i * n + col;
                        let b_vector = _mm256_load_ps(b_ptr.add(b_idx));

                        // 3. Fused Multiply-Add: c_vector = (a_broadcast * b_vector) + c_vector
                        c_vector = _mm256_fmadd_ps(a_broadcast, b_vector, c_vector);
                    }

                    // 4. Stream the compiled vector directly back to the mutable workspace
                    let out_slice = c.get_mut_2d_ptr(row, col)?;
                    _mm256_store_ps(out_slice, c_vector);

                    col += 8;
                } else {
                    // Scalar fallback loop handling remaining boundary columns (< 8 columns)
                    while col < n {
                        let mut accum = 0.0;
                        for i in 0..k {
                            accum += a.get_2d(row, i)? * b.get_2d(i, col)?;
                        }
                        c.set_2d(row, col, accum)?;
                        col += 1;
                    }
                }
            }
        }
        Ok(())
    }
}