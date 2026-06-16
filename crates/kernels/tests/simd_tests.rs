#[cfg(test)]
mod tests {
    use aether_arena::{UnifiedArena, MemoryCategory};
    use aether_tensor::{TensorView, TensorMut, TensorType};
    use aether_kernels::{NaiveEngine, simd::AvxEngine};
    use std::vec;

    #[test]
    fn test_simd_avx2_mathematical_parity() {
        // Core validation requirement: ensure SIMD outputs align perfectly with standard baseline arithmetic
        let mut arena = UnifiedArena::new(2048).unwrap();

        // Allocate a single buffer for all allocations to avoid multiple mutable borrows
        let total_size = 32 + 256 + 32 + 32; // 352 bytes
        let mut big_buf = arena.alloc_slice(total_size, MemoryCategory::Scratch).unwrap();

        // Split the buffer into four parts: 32, 256, 32, 32
        let (buf_a, rest) = big_buf.split_at_mut(32);
        let (buf_b, rest) = rest.split_at_mut(256);
        let (buf_c_naive, buf_c_simd) = rest.split_at_mut(32);

        let a_floats = unsafe { core::slice::from_raw_parts_mut(buf_a.as_mut_ptr() as *mut f32, 8) };
        for i in 0..8 { a_floats[i] = i as f32; }

        let b_floats = unsafe { core::slice::from_raw_parts_mut(buf_b.as_mut_ptr() as *mut f32, 64) };
        for i in 0..64 { b_floats[i] = (i * 2) as f32; }

        let view_a = TensorView::from_raw_parts(buf_a, &[1, 8], TensorType::F32).unwrap();
        let view_b = TensorView::from_raw_parts(buf_b, &[8, 8], TensorType::F32).unwrap();

        let mut mut_c_naive = TensorMut::from_raw_parts(buf_c_naive, &[1, 8], TensorType::F32).unwrap();
        let mut mut_c_simd = TensorMut::from_raw_parts(buf_c_simd, &[1, 8], TensorType::F32).unwrap();

        // 1. Calculate naive reference baseline
        NaiveEngine::gemm(&view_a, &view_b, &mut mut_c_naive).unwrap();

        // 2. Calculate hardware vector path
        unsafe {
            AvxEngine::gemm_avx2(&view_a, &view_b, &mut mut_c_simd).unwrap();
        }

        // 3. Verify vector registers generated identical outputs to the reference engine
        for i in 0..8 {
            assert_eq!(
                mut_c_naive.get_2d(0, i).unwrap(),
                mut_c_simd.get_2d(0, i).unwrap(),
                "SIMD precision failure at lane coordinate index [{}]", i
            );
        }
    }
}