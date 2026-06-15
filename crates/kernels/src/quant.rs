use aether_tensor::{TensorView, TensorMut, TensorError};
use crate::KernelError;

/// Quantized GEMV engine for Q4_0 format.
/// Performs fused dequantization and matrix-vector multiplication.
/// Assumes weight matrix is stored in Q4_0 format with block-wise scaling.
pub struct Q4Engine;
impl Q4Engine {
    /// Fused Q4_0 dequantization and GEMV: y = A * x, where A is Q4_0 packed.
    ///
    /// # Arguments
    /// * `a_packed` - Packed Q4_0 weight matrix (M x K) as [u8; (K * 4 + 7) / 8 * M?]
    ///   Actually, we expect the weight matrix to be stored in row-major order with blocks.
    ///   Each row is divided into blocks of BLOCK_SIZE elements (typically 32).
    ///   For each block, we have BLOCK_SIZE/2 bytes of packed weights (4 bits each) followed by a scale (f32).
    ///   So the layout per row: [block0_weights (BLOCK_SIZE/2 bytes), scale0, block1_weights, scale1, ...]
    /// * `scales` - Pointer to the scales for each block (f32). Length: (K + BLOCK_SIZE - 1) / BLOCK_SIZE per row.
    ///   But note: we assume the scales are stored contiguously per row? Actually, in GGUF, the scales are stored
    ///   in a separate array per tensor. We'll assume the caller has arranged the scales appropriately.
    ///   For simplicity, we'll assume the scales are provided as a slice of f32 with length equal to
    ///   number of blocks in the weight matrix (M * num_blocks_per_row).
    /// * `x` - Input vector (f32) of length K.
    /// * `y` - Output vector (f32) of length M (to be accumulated into).
    ///
    /// # Safety
    /// This function uses unsafe pointer arithmetic and assumes that the input pointers are valid
    /// and the tensors are properly aligned and within bounds.
    #[inline(always)]
    pub unsafe fn gemv_q4_0(
        a_packed: *const u8,
        scales: *const f32,
        x: *const f32,
        y: *mut f32,
        m: usize, // number of rows
        k: usize, // number of columns
        block_size: usize, // typically 32 for Q4_0
    ) {
        let num_blocks_per_row = (k + block_size - 1) / block_size;

        for i in 0..m {
            let mut acc = 0.0f32;
            let row_base = i * (num_blocks_per_row * (block_size / 2 + core::mem::size_of::<f32>()));
            let scale_base = i * num_blocks_per_row;

            for b in 0..num_blocks_per_row {
                let block_start = b * block_size;
                let block_end = core::cmp::min(block_start + block_size, k);
                let actual_block_len = block_end - block_start;

                // Pointer to the packed weights for this block in the row
                let mut weights_ptr = a_packed.add(row_base + b * (block_size / 2));
                // Pointer to the scale for this block
                let scale_ptr = scales.add(scale_base + b);
                let scale = *scale_ptr;

                // Process the block in chunks of 2 elements (since each byte holds 2 weights)
                let mut j = 0;
                while j < actual_block_len {
                    // Load one byte containing two 4-bit weights
                    let packed_byte = *weights_ptr.add(j / 2);
                    // Extract the two 4-bit values (assuming little-endian nibble order: lower 4 bits first?)
                    // In GGUF Q4_0, the packing is: two 4-bit values per byte, with the least significant 4 bits being the first weight.
                    let w0 = (packed_byte & 0x0F) as f32;
                    let w1 = ((packed_byte >> 4) & 0x0F) as f32;

                    // Dequantize: (weight - 8) * scale
                    let dq0 = (w0 - 8.0) * scale;
                    let dq1 = (w1 - 8.0) * scale;

                    // Accumulate with input vector elements
                    if j < actual_block_len {
                        acc += dq0 * *x.add(block_start + j);
                    }
                    if j + 1 < actual_block_len {
                        acc += dq1 * *x.add(block_start + j + 1);
                    }

                    j += 2;
                    weights_ptr = weights_ptr.add(1); // move to next byte
                }
            }

            *y.add(i) = acc;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let mut packed = vec![0u8; num_blocks_per_row];
        packed[0] = 0xA9; // 1010 1001

        // Scale for the block
        let scale = 1.0f32;
        let scales = vec![scale];

        // Input vector
        let x = [3.0f32, 4.0f32];

        // Expected output: (1.0*3.0 + 2.0*4.0) = 3.0 + 8.0 = 11.0
        let mut y = vec![0.0f32; m];

        // Safety: we are using raw pointers, but the data is valid for this test.
        unsafe {
            Q4Engine::gemv_q4_0(
                packed.as_ptr(),
                scales.as_ptr(),
                x.as_ptr(),
                y.as_mut_ptr(),
                m,
                k,
                block_size,
            );
        }

        assert_eq!(y[0], 11.0);
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
        let mut packed = vec![0x88u8; num_blocks_per_row]; // 2 bytes

        let scale = 1.0f32;
        let scales = vec![scale; num_blocks_per_row];

        let x = vec![0.0f32; k];
        let mut y = vec![0.0f32; m];

        unsafe {
            Q4Engine::gemv_q4_0(
                packed.as_ptr(),
                scales.as_ptr(),
                x.as_ptr(),
                y.as_mut_ptr(),
                m,
                k,
                block_size,
            );
        }

        assert_eq!(y[0], 0.0);
    }
}