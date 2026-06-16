pub mod simd;
pub mod quant;

pub use simd::AvxEngine;

use aether_tensor::{TensorView, TensorMut, TensorType, TensorError};

/// Error types for kernel operations.
#[derive(Debug, PartialEq, Eq)]
pub enum KernelError {
    /// Tensor dimensions do not match for the operation.
    DimensionMismatch,
    /// Error originating from tensor operations.
    TensorError(TensorError),
    /// Quantization not supported for this operation.
    QuantizationNotSupported,
    /// Hardware feature required for this kernel is missing.
    UnsupportedHardware,
}

impl From<TensorError> for KernelError {
    fn from(e: TensorError) -> Self {
        KernelError::TensorError(e)
    }
}

/// Naive General Matrix Multiplication (GEMM)
pub struct NaiveEngine;
impl NaiveEngine {
    pub fn gemm(a: &TensorView, b: &TensorView, c: &mut TensorMut) -> Result<(), KernelError> {
        match (a.tensor_type(), b.tensor_type(), c.tensor_type()) {
            (TensorType::F32, TensorType::F32, TensorType::F32) => {
                Self::gemm_f32_f32_f32(a, b, c)
            }
            _ => Err(KernelError::QuantizationNotSupported),
        }
    }

    fn gemm_f32_f32_f32(a: &TensorView, b: &TensorView, c: &mut TensorMut) -> Result<(), KernelError> {
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
                    sum += a.get_2d(i, kk)? * b.get_2d(kk, j)?;
                }
                c.set_2d(i, j, sum)?;
            }
        }
        Ok(())
    }
}

/// Cache-aware tiled GEMM
pub struct TiledEngine;
impl TiledEngine {
    pub fn gemm(a: &TensorView, b: &TensorView, c: &mut TensorMut, tile: usize) -> Result<(), KernelError> {
        match (a.tensor_type(), b.tensor_type(), c.tensor_type()) {
            (TensorType::F32, TensorType::F32, TensorType::F32) => {
                Self::gemm_f32_f32_f32(a, b, c, tile)
            }
            _ => Err(KernelError::QuantizationNotSupported),
        }
    }

    fn gemm_f32_f32_f32(a: &TensorView, b: &TensorView, c: &mut TensorMut, tile: usize) -> Result<(), KernelError> {
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

        for i0 in (0..m).step_by(tile) {
            let i1 = std::cmp::min(i0 + tile, m);
            for j0 in (0..n).step_by(tile) {
                let j1 = std::cmp::min(j0 + tile, n);
                for k0 in (0..k).step_by(tile) {
                    let k1 = std::cmp::min(k0 + tile, k);

                    for i in i0..i1 {
                        for j in j0..j1 {
                            let mut sum = if k0 == 0 { 0.0f32 } else { c.get_2d(i, j)? };
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
        let a_len = 2 * 3;
        let b_len = 3 * 2;
        let c_len = 2 * 2;

        let layout_a = Layout::from_size_align(a_len * 4, 64).unwrap();
        let layout_b = Layout::from_size_align(b_len * 4, 64).unwrap();
        let layout_c = Layout::from_size_align(c_len * 4, 64).unwrap();

        let ptr_a = unsafe { alloc(layout_a) };
        let ptr_b = unsafe { alloc(layout_b) };
        let ptr_c = unsafe { alloc(layout_c) };

        unsafe {
            let a_f32 = ptr_a as *mut f32;
            let b_f32 = ptr_b as *mut f32;
            let c_f32 = ptr_c as *mut f32;
            
            let a_vals = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
            for i in 0..6 { a_f32.add(i).write(a_vals[i]); }
            let b_vals = [7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];
            for i in 0..6 { b_f32.add(i).write(b_vals[i]); }
            for i in 0..4 { c_f32.add(i).write(0.0); }
        }

        let a = TensorView::from_raw_parts(unsafe { std::slice::from_raw_parts(ptr_a, a_len * 4) }, &[2, 3], TensorType::F32).unwrap();
        let b = TensorView::from_raw_parts(unsafe { std::slice::from_raw_parts(ptr_b, b_len * 4) }, &[3, 2], TensorType::F32).unwrap();
        let mut c = TensorMut::from_raw_parts(unsafe { std::slice::from_raw_parts_mut(ptr_c, c_len * 4) }, &[2, 2], TensorType::F32).unwrap();

        NaiveEngine::gemm(&a, &b, &mut c).unwrap();

        let expected = [[58.0, 64.0], [139.0, 154.0]];
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
        let test_cases = vec![(2, 3, 2, 1), (4, 5, 3, 2), (10, 10, 10, 3)];

        for (m, k, n, tile) in test_cases {
            let layout_a = Layout::from_size_align(m * k * 4, 64).unwrap();
            let layout_b = Layout::from_size_align(k * n * 4, 64).unwrap();
            let layout_c = Layout::from_size_align(m * n * 4, 64).unwrap();

            let ptr_a = unsafe { alloc(layout_a) };
            let ptr_b = unsafe { alloc(layout_b) };
            let ptr_cn = unsafe { alloc(layout_c) };
            let ptr_ct = unsafe { alloc(layout_c) };

            unsafe {
                for i in 0..(m*k) { (ptr_a as *mut f32).add(i).write((i as f32 + 1.0) / 10.0); }
                for i in 0..(k*n) { (ptr_b as *mut f32).add(i).write((i as f32 + 2.0) / 10.0); }
                for i in 0..(m*n) { (ptr_cn as *mut f32).add(i).write(0.0); (ptr_ct as *mut f32).add(i).write(0.0); }
            }

            let a = TensorView::from_raw_parts(unsafe { std::slice::from_raw_parts(ptr_a, m * k * 4) }, &[m, k], TensorType::F32).unwrap();
            let b = TensorView::from_raw_parts(unsafe { std::slice::from_raw_parts(ptr_b, k * n * 4) }, &[k, n], TensorType::F32).unwrap();
            let mut cn = TensorMut::from_raw_parts(unsafe { std::slice::from_raw_parts_mut(ptr_cn, m * n * 4) }, &[m, n], TensorType::F32).unwrap();
            let mut ct = TensorMut::from_raw_parts(unsafe { std::slice::from_raw_parts_mut(ptr_ct, m * n * 4) }, &[m, n], TensorType::F32).unwrap();

            NaiveEngine::gemm(&a, &b, &mut cn).unwrap();
            TiledEngine::gemm(&a, &b, &mut ct, tile).unwrap();

            for i in 0..m {
                for j in 0..n {
                    assert!((cn.get_2d(i, j).unwrap() - ct.get_2d(i, j).unwrap()).abs() < 1e-5);
                }
            }

            unsafe {
                std::alloc::dealloc(ptr_a, layout_a);
                std::alloc::dealloc(ptr_b, layout_b);
                std::alloc::dealloc(ptr_cn, layout_c);
                std::alloc::dealloc(ptr_ct, layout_c);
            }
        }
    }
}
