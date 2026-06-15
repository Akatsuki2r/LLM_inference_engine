pub mod simd;
pub mod quant;

use aether_tensor::{TensorView, TensorMut, TensorError};
use std::ops::{AddAssign, Mul};

/// Error types for kernel operations.
#[derive(Debug, PartialEq, Eq)]
pub enum KernelError {
    /// Tensor dimensions do not match for the operation.
    DimensionMismatch,
    /// Error originating from tensor operations.
    TensorError(TensorError),
}

impl From<TensorError> for KernelError {
    fn from(e: TensorError) -> Self {
        KernelError::TensorError(e)
    }
}

/// Naive General Matrix Multiplication (GEMM)
/// Computes C = A * B
///
/// A: (M x K)
/// B: (K x N)
/// C: (M x N)
pub struct NaiveEngine;
impl NaiveEngine {
    pub fn gemm(a: &TensorView, b: &TensorView, c: &mut TensorMut) -> Result<(), KernelError> {
        let shape_a = a.shape();
        let shape_b = b.shape();
        let shape_c = c.shape();

        if shape_a.len() != 2 || shape_b.len() != 2 || shape_c.len() != 2 {
            return Err(KernelError::DimensionMismatch);
        }

        let m = shape_a[0];
        let k = shape_a[1];
        let n = shape_b[1];

        if k != shape_b[0] || m != shape_c[0] || n != shape_c[1] {
            return Err(KernelError::DimensionMismatch);
        }

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for kk in 0..k {
                    // a[i, kk] * b[kk, j]
                    sum += a.get_2d(i, kk)? * b.get_2d(kk, j)?;
                }
                c.set_2d(i, j, sum)?;
            }
        }
        Ok(())
    }
}

/// Cache-aware tiled GEMM (blocked matrix multiplication)
/// Improves memory locality by operating on submatrices that fit in cache.
///
/// A: (M x K)
/// B: (K x N)
/// C: (M x N)
///
/// Tile size should be chosen so that a tile of A, a tile of B, and a tile of C
/// fit in L1 cache. Typical values: 32, 64, 128 depending on data type and
/// cache size.
pub struct TiledEngine;
impl TiledEngine {
    pub fn gemm(a: &TensorView, b: &TensorView, c: &mut TensorMut, tile: usize) -> Result<(), KernelError> {
        let shape_a = a.shape();
        let shape_b = b.shape();
        let shape_c = c.shape();

        if shape_a.len() != 2 || shape_b.len() != 2 || shape_c.len() != 2 {
            return Err(KernelError::DimensionMismatch);
        }

        let m = shape_a[0];
        let k = shape_a[1];
        let n = shape_b[1];

        if k != shape_b[0] || m != shape_c[0] || n != shape_c[1] {
            return Err(KernelError::DimensionMismatch);
        }

        // Process in tiles
        for i0 in (0..m).step_by(tile) {
            let i1 = std::cmp::min(i0 + tile, m);
            for j0 in (0..n).step_by(tile) {
                let j1 = std::cmp::min(j0 + tile, n);
                for k0 in (0..k).step_by(tile) {
                    let k1 = std::cmp::min(k0 + tile, k);

                    // Compute C[i0:i1, j0:j1] += A[i0:i1, k0:k1] * B[k0:k1, j0:j1]
                    for i in i0..i1 {
                        for j in j0..j1 {
                            let mut sum = c.get_2d(i, j)?; // accumulate into existing C
                            for kk in k0..k1 {
                                sum += a.get_2d(i, kk)? * b.get_2d(kk, j)?;
                            }
                            c.set_2d(i, j, sum)?;
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::alloc::{alloc, Layout};

    #[test]
    fn test_gemm_naive_correctness() {
        // A: 2x3
        // [ 1, 2, 3 ]
        // [ 4, 5, 6 ]
        let a_len = 2 * 3;
        let b_len = 3 * 2;
        let c_len = 2 * 2;

        let layout_a = Layout::from_size_align(a_len * 4, 64).expect("Layout A failed");
        let layout_b = Layout::from_size_align(b_len * 4, 64).expect("Layout B failed");
        let layout_c = Layout::from_size_align(c_len * 4, 64).expect("Layout C failed");

        let ptr_a = unsafe { alloc(layout_a) as *mut u8 };
        let ptr_b = unsafe { alloc(layout_b) as *mut u8 };
        let ptr_c = unsafe { alloc(layout_c) as *mut u8 };

        // Initialize A as f32 values
        unsafe {
            let ptr_a_f32 = ptr_a as *mut f32;
            let a_vals = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
            for (i, &val) in a_vals.iter().enumerate() {
                ptr_a_f32.add(i).write(val);
            }
        }
        // Initialize B
        unsafe {
            let ptr_b_f32 = ptr_b as *mut f32;
            let b_vals = [7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];
            for (i, &val) in b_vals.iter().enumerate() {
                ptr_b_f32.add(i).write(val);
            }
        }
        // Zero C
        unsafe {
            let ptr_c_f32 = ptr_c as *mut f32;
            for i in 0..c_len {
                ptr_c_f32.add(i).write(0.0);
            }
        }

        let a = TensorView::from_raw_parts(unsafe { std::slice::from_raw_parts(ptr_a, a_len * 4) }, vec![2, 3]).unwrap();
        let b = TensorView::from_raw_parts(unsafe { std::slice::from_raw_parts(ptr_b, b_len * 4) }, vec![3, 2]).unwrap();
        let mut c = TensorMut::from_raw_parts(unsafe { std::slice::from_raw_parts_mut(ptr_c, c_len * 4) }, vec![2, 2]).unwrap();

        NaiveEngine::gemm(&a, &b, &mut c).unwrap();

        let expected = [
            [58.0f32, 64.0],
            [139.0, 154.0],
        ];

        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(c.get_2d(i, j).unwrap(), expected[i][j]);
            }
        }

        unsafe {
            std::alloc::dealloc(ptr_a, layout_a);
            std::alloc::dealloc(ptr_b, layout_b);
            std::alloc::dealloc(ptr_c, layout_c);
        }
    }

    #[test]
    fn test_gemm_tiled_matches_naive() {
        // Test with various sizes and tile dimensions
        let test_cases = vec![
            (2, 3, 2, 1), // tiny
            (4, 5, 3, 2),
            (10, 10, 10, 3),
            (16, 16, 16, 4),
            (32, 32, 32, 8),
        ];

        for (m, k, n, tile) in test_cases {
            // Allocate A (m x k), B (k x n), C (m x n)
            let a_len = m * k;
            let b_len = k * n;
            let c_len = m * n;

            let layout_a = Layout::from_size_align(a_len * 4, 64).unwrap();
            let layout_b = Layout::from_size_align(b_len * 4, 64).unwrap();
            let layout_c = Layout::from_size_align(c_len * 4, 64).unwrap();

            let ptr_a = unsafe { alloc(layout_a) as *mut u8 };
            let ptr_b = unsafe { alloc(layout_b) as *mut u8 };
            let ptr_c_naive = unsafe { alloc(layout_c) as *mut u8 };
            let ptr_c_tiled = unsafe { alloc(layout_c) as *mut u8 };

            // Initialize with deterministic values
            unsafe {
                let ptr_a_f32 = ptr_a as *mut f32;
                for i in 0..a_len {
                    ptr_a_f32.add(i).write((i as f32 + 1.0) / 10.0);
                }
            }
            unsafe {
                let ptr_b_f32 = ptr_b as *mut f32;
                for i in 0..b_len {
                    ptr_b_f32.add(i).write((i as f32 + 2.0) / 10.0);
                }
            }
            unsafe {
                let ptr_c_naive_f32 = ptr_c_naive as *mut f32;
                let ptr_c_tiled_f32 = ptr_c_tiled as *mut f32;
                for i in 0..c_len {
                    ptr_c_naive_f32.add(i).write(0.0);
                    ptr_c_tiled_f32.add(i).write(0.0);
                }
            }

            let a = TensorView::from_raw_parts(unsafe { std::slice::from_raw_parts(ptr_a, a_len * 4) }, vec![m, k]).unwrap();
            let b = TensorView::from_raw_parts(unsafe { std::slice::from_raw_parts(ptr_b, b_len * 4) }, vec![k, n]).unwrap();
            let mut c_naive = TensorMut::from_raw_parts(unsafe { std::slice::from_raw_parts_mut(ptr_c_naive, c_len * 4) }, vec![m, n]).unwrap();
            let mut c_tiled = TensorMut::from_raw_parts(unsafe { std::slice::from_raw_parts_mut(ptr_c_tiled, c_len * 4) }, vec![m, n]).unwrap();

            NaiveEngine::gemm(&a, &b, &mut c_naive).unwrap();
            TiledEngine::gemm(&a, &b, &mut c_tiled, tile).unwrap();

            // Compare results
            for i in 0..m {
                for j in 0..n {
                    let naive = c_naive.get_2d(i, j).unwrap();
                    let tiled = c_tiled.get_2d(i, j).unwrap();
                    assert!(
                        (naive - tiled).abs() < 1e-5,
                        "Mismatch at ({}, {}): naive={}, tiled={}",
                        i, j, naive, tiled
                    );
                }
            }

            unsafe {
                std::alloc::dealloc(ptr_a, layout_a);
                std::alloc::dealloc(ptr_b, layout_b);
                std::alloc::dealloc(ptr_c_naive, layout_c);
                std::alloc::dealloc(ptr_c_tiled, layout_c);
            }
        }
    }
}