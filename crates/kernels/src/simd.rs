#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m256, _mm256_loadu_ps, _mm256_storeu_ps, _mm256_setzero_ps,
    _mm256_fmadd_ps, _mm256_broadcast_ss
};
use aether_tensor::{TensorView, TensorMut};
use crate::KernelError;

pub struct AvxEngine;

impl AvxEngine {
    pub unsafe fn gemm_avx2(a: &TensorView, b: &TensorView, c: &mut TensorMut) -> Result<(), KernelError> {
        let shape_a = a.shape();
        let shape_b = b.shape();

        if shape_a.len() != 2 || shape_b.len() != 2 {
            return Err(KernelError::DimensionMismatch);
        }

        let m = shape_a[0];
        let k = shape_a[1];
        let n = shape_b[1];

        if k != shape_b[0] {
            return Err(KernelError::DimensionMismatch);
        }

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                return Self::execute_core(a, b, c, m, k, n);
            }
        }

        Err(KernelError::UnsupportedHardware)
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
        let a_ptr = a.data().as_ptr() as *const f32;
        let b_ptr = b.data().as_ptr() as *const f32;

        for row in 0..m {
            let mut col = 0;
            while col < n {
                if col + 8 <= n {
                    let mut c_vector: __m256 = _mm256_setzero_ps();
                    for i in 0..k {
                        let val_a = *a_ptr.add(row * k + i);
                        let a_broadcast = _mm256_broadcast_ss(&val_a);
                        let b_vector = _mm256_loadu_ps(b_ptr.add(i * n + col));
                        c_vector = _mm256_fmadd_ps(a_broadcast, b_vector, c_vector);
                    }
                    let out_ptr = c.get_mut_2d_ptr(row, col)?;
                    _mm256_storeu_ps(out_ptr, c_vector);
                    col += 8;
                } else {
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
