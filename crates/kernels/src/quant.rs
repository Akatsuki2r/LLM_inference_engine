use aether_tensor::{TensorView, TensorMut, TensorType};
use crate::KernelError;

/// Quantized GEMM engine for Q4_0 format.
/// Performs fused dequantization and matrix multiplication.
/// Assumes weight matrix is stored in Q4_0 format with block-wise scaling.
pub struct Q4Engine;
impl Q4Engine {
    /// Fused Q4_0 dequantization and GEMM: C = A * B, where A is Q4_0 packed and B is f32.
    ///
    /// # Arguments
    /// * `a` - Packed Q4_0 weight matrix (M x K)
    /// * `b` - f32 matrix (K x N)
    /// * `c` - f32 output matrix (M x N) to accumulate into
    /// * `a_scales` - Scale factors for A tensor, one per block
    /// * `block_size` - Block size for quantization (typically 32 for Q4_0)
    ///
    /// # Safety
    /// This function uses unsafe pointer arithmetic and assumes that the input pointers are valid
    /// and the tensors are properly aligned and within bounds.
    #[inline(always)]
    pub unsafe fn gemm_q4_0_f32_f32(
        a: &TensorView,
        b: &TensorView,
        c: &mut TensorMut,
        a_scales: &[f32],
        block_size: usize, // typically 32 for Q4_0
    ) -> Result<(), KernelError> {
        // Validate tensor types
        if a.tensor_type() != TensorType::Q4_0 {
            return Err(KernelError::QuantizationNotSupported);
        }
        if b.tensor_type() != TensorType::F32 {
            return Err(KernelError::QuantizationNotSupported);
        }
        if c.tensor_type() != TensorType::F32 {
            return Err(KernelError::QuantizationNotSupported);
        }

        let shape_a = a.shape();
        let shape_b = b.shape();
        let shape_c = c.shape();

        // Validate dimensions
        if shape_a.len() != 2 || shape_b.len() != 2 || shape_c.len() != 2 {
            return Err(KernelError::DimensionMismatch);
        }

        let m = shape_a[0]; // rows of A
        let k = shape_a[1]; // cols of A, rows of B
        let n = shape_b[1]; // cols of B

        if k != shape_b[0] {
            return Err(KernelError::DimensionMismatch);
        }
        if m != shape_c[0] || n != shape_c[1] {
            return Err(KernelError::DimensionMismatch);
        }

        let num_blocks_per_row = (k + block_size - 1) / block_size;
        let expected_scales_len = m * num_blocks_per_row;
        if a_scales.len() != expected_scales_len {
            return Err(KernelError::DimensionMismatch);
        }

        // Process each row of A
        for i in 0..m {
            // Process each column of B (each output column)
            for j in 0..n {
                let mut acc = 0.0f32;

                // Process each block in the row
                for block_idx in 0..num_blocks_per_row {
                    let block_start = block_idx * block_size;
                    let block_end = core::cmp::min(block_start + block_size, k);
                    let actual_block_len = block_end - block_start;

                    // Pointer to the packed weights for this block in the row
                    let weights_ptr = a.data().as_ptr().add(
                        i * (num_blocks_per_row * (block_size / 2)) +
                        block_idx * (block_size / 2)
                    );
                    // Pointer to the scale for this block
                    let scale_ptr = a_scales.as_ptr().add(i * num_blocks_per_row + block_idx);
                    let scale = *scale_ptr;

                    // Process the block in chunks of 2 elements (since each byte holds 2 weights)
                    let mut k_in_block = 0;
                    while k_in_block < actual_block_len {
                        // Load one byte containing two 4-bit weights
                        let packed_byte = *weights_ptr.add(k_in_block / 2);
                        // Extract the two 4-bit values (little-endian nibble order: lower 4 bits first)
                        let w0 = (packed_byte & 0x0F) as f32;
                        let w1 = ((packed_byte >> 4) & 0x0F) as f32;

                        // Dequantize: (weight - 8) * scale
                        let dq0 = (w0 - 8.0) * scale;
                        let dq1 = (w1 - 8.0) * scale;

                        // Global column index
                        let global_k0 = block_start + k_in_block;
                        let global_k1 = block_start + k_in_block + 1;

                        // Accumulate with input vector elements
                        if global_k0 < k {
                            acc += dq0 * b.get_2d(global_k0, j)?;
                        }
                        if global_k1 < k {
                            acc += dq1 * b.get_2d(global_k1, j)?;
                        }

                        k_in_block += 2;
                    }
                }

                // Store the accumulated result
                c.set_2d(i, j, acc)?;
            }
        }

        Ok(())
    }
}

/// Quantized GEMV engine for Q4_0 format.
/// Performs fused dequantization and matrix-vector multiplication.
/// Assumes weight matrix is stored in Q4_0 format with block-wise scaling.
impl Q4Engine {
    /// Fused Q4_0 dequantization and GEMV: y = A * x, where A is Q4_0 packed.
    ///
    /// # Arguments
    /// * `a` - Packed Q4_0 weight matrix (M x K)
    /// * `x` - Input vector (f32) of length K.
    /// * `y` - Output vector (f32) of length M (to be accumulated into).
    /// * `a_scales` - Scale factors for A tensor, one per block
    /// * `block_size` - Block size for quantization (typically 32 for Q4_0)
    ///
    /// # Safety
    /// This function uses unsafe pointer arithmetic and assumes that the input pointers are valid
    /// and the tensors are properly aligned and within bounds.
    #[inline(always)]
    pub unsafe fn gemv_q4_0_f32(
        a: &TensorView,
        x: &TensorView,
        y: &mut TensorMut,
        a_scales: &[f32],
        block_size: usize, // typically 32 for Q4_0
    ) -> Result<(), KernelError> {
        // Validate tensor types
        if a.tensor_type() != TensorType::Q4_0 {
            return Err(KernelError::QuantizationNotSupported);
        }
        if x.tensor_type() != TensorType::F32 {
            return Err(KernelError::QuantizationNotSupported);
        }
        if y.tensor_type() != TensorType::F32 {
            return Err(KernelError::QuantizationNotSupported);
        }

        let shape_a = a.shape();
        let shape_x = x.shape();
        let shape_y = y.shape();

        // Validate dimensions
        if shape_a.len() != 2 || shape_x.len() != 1 || shape_y.len() != 1 {
            return Err(KernelError::DimensionMismatch);
        }

        let m = shape_a[0]; // rows of A
        let k = shape_a[1]; // cols of A, size of x
        let x_len = shape_x[0]; // size of x
        let y_len = shape_y[0]; // size of y

        if k != x_len {
            return Err(KernelError::DimensionMismatch);
        }
        if m != y_len {
            return Err(KernelError::DimensionMismatch);
        }

        let num_blocks_per_row = (k + block_size - 1) / block_size;
        let expected_scales_len = m * num_blocks_per_row;
        if a_scales.len() != expected_scales_len {
            return Err(KernelError::DimensionMismatch);
        }

        // Process each row of A
        for i in 0..m {
            let mut acc = 0.0f32;

            // Process each block in the row
            for block_idx in 0..num_blocks_per_row {
                let block_start = block_idx * block_size;
                let block_end = core::cmp::min(block_start + block_size, k);
                let actual_block_len = block_end - block_start;

                // Pointer to the packed weights for this block in the row
                let weights_ptr = a.data().as_ptr().add(
                    i * (num_blocks_per_row * (block_size / 2)) +
                    block_idx * (block_size / 2)
                );
                // Pointer to the scale for this block
                let scale_ptr = a_scales.as_ptr().add(i * num_blocks_per_row + block_idx);
                let scale = *scale_ptr;

                // Process the block in chunks of 2 elements (since each byte holds 2 weights)
                let mut k_in_block = 0;
                while k_in_block < actual_block_len {
                    // Load one byte containing two 4-bit weights
                    let packed_byte = *weights_ptr.add(k_in_block / 2);
                    // Extract the two 4-bit values (little-endian nibble order: lower 4 bits first)
                    let w0 = (packed_byte & 0x0F) as f32;
                    let w1 = ((packed_byte >> 4) & 0x0F) as f32;

                    // Dequantize: (weight - 8) * scale
                    let dq0 = (w0 - 8.0) * scale;
                    let dq1 = (w1 - 8.0) * scale;

                    // Global column index
                    let global_k0 = block_start + k_in_block;
                    let global_k1 = block_start + k_in_block + 1;

                    // Accumulate with input vector elements
                    if global_k0 < k {
                        acc += dq0 * x.get(&[global_k0])?;
                    }
                    if global_k1 < k {
                        acc += dq1 * x.get(&[global_k1])?;
                    }

                    k_in_block += 2;
                }
            }

            // Store the accumulated result
            *y.get_mut(&[i])? = acc;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::alloc::{alloc, Layout};

    #[test]
    fn test_q4_0_gemv_small() {
        // Simple test: 1x2 matrix, block_size=2
        // Weights: [1.0, 2.0] -> quantize to Q4_0
        // For simplicity, we'll use a scale that makes the dequantization exact.
        // Let's choose scale = 1.0, then the quantized weight should be (weight + 8) as u4.
        // So 1.0 -> 9, 2.0 -> 10.
        // Packed: 9 in lower 4 bits, 10 in upper 4 bits -> 0b1010_1001 = 0xA9.

        let m = 1;
        let k = 2;
        let block_size = 2;
        let num_blocks_per_row = (k + block_size - 1) / block_size; // 1

        // Allocate space for packed weights: 1 byte per block (since block_size/2 = 1)
        let packed_layout = Layout::from_size_align(num_blocks_per_row, 64).unwrap();
        let packed_ptr = unsafe { alloc(packed_layout) as *mut u8 };
        unsafe { *packed_ptr = 0xA9u8; } // 1010 1001

        // Scale for the block
        let scale_layout = Layout::from_size_align(core::mem::size_of::<f32>(), 64).unwrap();
        let scale_ptr = unsafe { alloc(scale_layout) as *mut f32 };
        unsafe { *scale_ptr = 1.0f32; }

        // Input vector: [3.0, 4.0]
        let x_layout = Layout::from_size_align(k * core::mem::size_of::<f32>(), 64).unwrap();
        let x_ptr = unsafe { alloc(x_layout) as *mut f32 };
        unsafe {
            *x_ptr.add(0) = 3.0f32;
            *x_ptr.add(1) = 4.0f32;
        }

        // Expected output vector: [11.0] (since 1.0*3.0 + 2.0*4.0 = 3.0 + 8.0 = 11.0)
        let y_layout = Layout::from_size_align(m * core::mem::size_of::<f32>(), 64).unwrap();
        let y_ptr = unsafe { alloc(y_layout) as *mut f32 };
        unsafe { *y_ptr = 0.0f32; }

        // Create tensors
        let a = unsafe {
            TensorView::from_raw_parts(
                std::slice::from_raw_parts(packed_ptr, num_blocks_per_row),
                &[m, k],
                TensorType::Q4_0
            ).unwrap()
        };
        let x = unsafe {
            TensorView::from_raw_parts(
                std::slice::from_raw_parts(x_ptr as *const u8, k * core::mem::size_of::<f32>()),
                &[k],
                TensorType::F32
            ).unwrap()
        };
        let mut y = unsafe {
            TensorMut::from_raw_parts(
                std::slice::from_raw_parts_mut(y_ptr as *mut u8, m * core::mem::size_of::<f32>()),
                &[m],
                TensorType::F32
            ).unwrap()
        };

        let scales = unsafe {
            std::slice::from_raw_parts(scale_ptr, 1)
        };

        // Safety: we are using raw pointers, but the data is valid for this test.
        unsafe {
            Q4Engine::gemv_q4_0_f32(&a, &x, &mut y, scales, block_size).unwrap();
        }

        let result = unsafe { *y_ptr };
        assert_eq!(result, 11.0);

        // Cleanup
        unsafe {
            std::alloc::dealloc(packed_ptr, packed_layout);
            std::alloc::dealloc(scale_ptr as *mut u8, scale_layout);
            std::alloc::dealloc(x_ptr as *mut u8, x_layout);
            std::alloc::dealloc(y_ptr as *mut u8, y_layout);
        }
    }

    #[test]
    fn test_q4_0_gemv_zero() {
        // Test with zero weights and zero input -> zero output.
        let m = 1;
        let k = 4;
        let block_size = 2;
        let num_blocks_per_row = (k + block_size - 1) / block_size; // 2

        // All weights zero -> quantized value 8 (since (0-8)*scale = 0 => weight=8)
        // Packed: two 8's per byte: 0x88 (1000_1000)
        let packed_layout = Layout::from_size_align(num_blocks_per_row, 64).unwrap();
        let packed_ptr = unsafe { alloc(packed_layout) as *mut u8 };
        unsafe {
            *packed_ptr.add(0) = 0x88u8;
            *packed_ptr.add(1) = 0x88u8;
        }

        let scale_layout = Layout::from_size_align(num_blocks_per_row * core::mem::size_of::<f32>(), 64).unwrap();
        let scale_ptr = unsafe { alloc(scale_layout) as *mut f32 };
        unsafe {
            *scale_ptr.add(0) = 1.0f32;
            *scale_ptr.add(1) = 1.0f32;
        }

        // Input vector: [0.0, 0.0, 0.0, 0.0]
        let x_layout = Layout::from_size_align(k * core::mem::size_of::<f32>(), 64).unwrap();
        let x_ptr = unsafe { alloc(x_layout) as *mut f32 };
        unsafe {
            *x_ptr.add(0) = 0.0f32;
            *x_ptr.add(1) = 0.0f32;
            *x_ptr.add(2) = 0.0f32;
            *x_ptr.add(3) = 0.0f32;
        }

        // Expected output vector: [0.0]
        let y_layout = Layout::from_size_align(m * core::mem::size_of::<f32>(), 64).unwrap();
        let y_ptr = unsafe { alloc(y_layout) as *mut f32 };
        unsafe { *y_ptr = 0.0f32; }

        // Create tensors
        let a = unsafe {
            TensorView::from_raw_parts(
                std::slice::from_raw_parts(packed_ptr, num_blocks_per_row),
                &[m, k],
                TensorType::Q4_0
            ).unwrap()
        };
        let x = unsafe {
            TensorView::from_raw_parts(
                std::slice::from_raw_parts(x_ptr as *const u8, k * core::mem::size_of::<f32>()),
                &[k],
                TensorType::F32
            ).unwrap()
        };
        let mut y = unsafe {
            TensorMut::from_raw_parts(
                std::slice::from_raw_parts_mut(y_ptr as *mut u8, m * core::mem::size_of::<f32>()),
                &[m],
                TensorType::F32
            ).unwrap()
        };

        let scales = unsafe {
            std::slice::from_raw_parts(scale_ptr, num_blocks_per_row)
        };

        unsafe {
            Q4Engine::gemv_q4_0_f32(&a, &x, &mut y, scales, block_size).unwrap();
        }

        let result = unsafe { *y_ptr };
        assert_eq!(result, 0.0);

        // Cleanup
        unsafe {
            std::alloc::dealloc(packed_ptr, packed_layout);
            std::alloc::dealloc(scale_ptr as *mut u8, scale_layout);
            std::alloc::dealloc(x_ptr as *mut u8, x_layout);
            std::alloc::dealloc(y_ptr as *mut u8, y_layout);
        }
    }

    #[test]
    fn test_q4_0_gemv_multi_row() {
        let m = 2;
        let k = 4;
        let block_size = 32; // Standard Q4_0 block size
        let num_blocks_per_row = (k + block_size - 1) / block_size; // 1

        // Allocate space for packed weights: m * num_blocks_per_row * (block_size/2)
        let packed_size = m * num_blocks_per_row * (block_size / 2);
        let packed_layout = Layout::from_size_align(packed_size, 64).unwrap();
        let packed_ptr = unsafe { alloc(packed_layout) as *mut u8 };
        unsafe {
            // Row 0: all weights 1.0 (quantized 9) -> 0x99
            for i in 0..(block_size / 2) {
                *packed_ptr.add(i) = 0x99u8;
            }
            // Row 1: all weights 2.0 (quantized 10) -> 0xAA
            for i in 0..(block_size / 2) {
                *packed_ptr.add(num_blocks_per_row * (block_size / 2) + i) = 0xAAu8;
            }
        }

        let scale_size = m * num_blocks_per_row;
        let scale_layout = Layout::from_size_align(scale_size * core::mem::size_of::<f32>(), 64).unwrap();
        let scale_ptr = unsafe { alloc(scale_layout) as *mut f32 };
        unsafe {
            *scale_ptr.add(0) = 1.0f32; // Row 0 scale
            *scale_ptr.add(1) = 1.0f32; // Row 1 scale
        }

        let x_layout = Layout::from_size_align(k * core::mem::size_of::<f32>(), 64).unwrap();
        let x_ptr = unsafe { alloc(x_layout) as *mut f32 };
        unsafe {
            for i in 0..k { *x_ptr.add(i) = 1.0f32; }
        }

        let y_layout = Layout::from_size_align(m * core::mem::size_of::<f32>(), 64).unwrap();
        let y_ptr = unsafe { alloc(y_layout) as *mut f32 };
        unsafe {
            *y_ptr.add(0) = 0.0f32;
            *y_ptr.add(1) = 0.0f32;
        }

        let a = unsafe {
            TensorView::from_raw_parts(
                std::slice::from_raw_parts(packed_ptr, packed_size),
                &[m, k],
                TensorType::Q4_0
            ).unwrap()
        };
        let x = unsafe {
            TensorView::from_raw_parts(
                std::slice::from_raw_parts(x_ptr as *const u8, k * core::mem::size_of::<f32>()),
                &[k],
                TensorType::F32
            ).unwrap()
        };
        let mut y = unsafe {
            TensorMut::from_raw_parts(
                std::slice::from_raw_parts_mut(y_ptr as *mut u8, m * core::mem::size_of::<f32>()),
                &[m],
                TensorType::F32
            ).unwrap()
        };

        let scales = unsafe { std::slice::from_raw_parts(scale_ptr, scale_size) };

        unsafe {
            Q4Engine::gemv_q4_0_f32(&a, &x, &mut y, scales, block_size).unwrap();
        }

        assert_eq!(unsafe { *y_ptr.add(0) }, 4.0); // 1.0 * 4 elements
        assert_eq!(unsafe { *y_ptr.add(1) }, 8.0); // 2.0 * 4 elements

        unsafe {
            std::alloc::dealloc(packed_ptr, packed_layout);
            std::alloc::dealloc(scale_ptr as *mut u8, scale_layout);
            std::alloc::dealloc(x_ptr as *mut u8, x_layout);
            std::alloc::dealloc(y_ptr as *mut u8, y_layout);
        }
    }
}