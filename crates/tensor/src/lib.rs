use std::marker::PhantomData;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum TensorError {
    DimensionMismatch,
    IndexOutOfBounds,
}

/// Centralized coordinate-to-offset computer.
/// Eliminates DRY violations and isolates unsafe raw boundary calculations.
#[inline(always)]
fn calculate_flat_offset(indices: &[usize], shape: &[usize], strides: &[usize]) -> Result<usize, TensorError> {
    if indices.len() != shape.len() {
        return Err(TensorError::DimensionMismatch);
    }
    let mut flat_index = 0;
    for i in 0..indices.len() {
        if indices[i] >= shape[i] {
            return Err(TensorError::IndexOutOfBounds);
        }
        flat_index += indices[i] * strides[i];
    }
    Ok(flat_index)
}

/// Fast 2D matrix offset extractor that completely avoids slice allocation overhead.
#[inline(always)]
fn calculate_2d_offset(row: usize, col: usize, shape: &[usize], strides: &[usize]) -> Result<usize, TensorError> {
    if shape.len() != 2 {
        return Err(TensorError::DimensionMismatch);
    }
    if row >= shape[0] || col >= shape[1] {
        return Err(TensorError::IndexOutOfBounds);
    }
    Ok(row * strides[0] + col * strides[1])
}

/// Read-Only view designed specifically for memory-mapped weights.
pub struct TensorView<'a> {
    data: &'a [f32],
    shape: Vec<usize>,
    strides: Vec<usize>,
}

impl<'a> TensorView<'a> {
    pub fn from_raw_parts(raw_bytes: &'a [u8], shape: Vec<usize>) -> Result<Self, TensorError> {
        let float_count = raw_bytes.len() / core::mem::size_of::<f32>();
        let data = unsafe { core::slice::from_raw_parts(raw_bytes.as_ptr() as *const f32, float_count) };

        if shape.iter().product::<usize>() > data.len() {
            return Err(TensorError::DimensionMismatch);
        }

        let mut strides = vec![0; shape.len()];
        let mut current_stride = 1;
        for i in (0..shape.len()).rev() {
            strides[i] = current_stride;
            current_stride *= shape[i];
        }

        Ok(Self { data, shape, strides })
    }

    #[inline]
    pub fn get(&self, indices: &[usize]) -> Result<f32, TensorError> {
        let offset = calculate_flat_offset(indices, &self.shape, &self.strides)?;
        Ok(self.data[offset])
    }

    /// High-performance 2D lookup that replaces array slice bounds checking in inner loops.
    #[inline(always)]
    pub fn get_2d(&self, row: usize, col: usize) -> Result<f32, TensorError> {
        let offset = calculate_2d_offset(row, col, &self.shape, &self.strides)?;
        Ok(self.data[offset])
    }

    pub fn transpose(&self) -> Result<Self, TensorError> {
        if self.shape.len() != 2 { return Err(TensorError::DimensionMismatch); }
        Ok(Self {
            data: self.data,
            shape: vec![self.shape[1], self.shape[0]],
            strides: vec![self.strides[1], self.strides[0]],
        })
    }

    pub fn shape(&self) -> &[usize] { &self.shape }

    /// Returns a slice of the underlying data.
    #[inline(always)]
    pub fn data(&self) -> &[f32] {
        self.data
    }
}

/// Write-enabled mutable workspace tensor wrapper.
pub struct TensorMut<'a> {
    data: &'a mut [f32],
    shape: Vec<usize>,
    strides: Vec<usize>,
}

impl<'a> TensorMut<'a> {
    pub fn from_raw_parts(raw_bytes: &'a mut [u8], shape: Vec<usize>) -> Result<Self, TensorError> {
        let float_count = raw_bytes.len() / core::mem::size_of::<f32>();
        let data = unsafe { core::slice::from_raw_parts_mut(raw_bytes.as_mut_ptr() as *mut f32, float_count) };

        if shape.iter().product::<usize>() > data.len() {
            return Err(TensorError::DimensionMismatch);
        }

        let mut strides = vec![0; shape.len()];
        let mut current_stride = 1;
        for i in (0..shape.len()).rev() {
            strides[i] = current_stride;
            current_stride *= shape[i];
        }

        Ok(Self { data, shape, strides })
    }

    #[inline]
    pub fn get(&self, indices: &[usize]) -> Result<f32, TensorError> {
        let offset = calculate_flat_offset(indices, &self.shape, &self.strides)?;
        Ok(self.data[offset])
    }

    #[inline]
    pub fn get_mut(&mut self, indices: &[usize]) -> Result<&mut f32, TensorError> {
        let offset = calculate_flat_offset(indices, &self.shape, &self.strides)?;
        Ok(&mut self.data[offset])
    }

    #[inline(always)]
    pub fn get_2d(&self, row: usize, col: usize) -> Result<f32, TensorError> {
        let offset = calculate_2d_offset(row, col, &self.shape, &self.strides)?;
        Ok(self.data[offset])
    }

    #[inline(always)]
    pub fn set_2d(&mut self, row: usize, col: usize, val: f32) -> Result<(), TensorError> {
        let offset = calculate_2d_offset(row, col, &self.shape, &self.strides)?;
        self.data[offset] = val;
        Ok(())
    }

    /// Returns a direct mutable raw pointer to a 2D coordinate for SIMD streaming.
    #[inline(always)]
    pub unsafe fn get_mut_2d_ptr(&mut self, row: usize, col: usize) -> Result<*mut f32, TensorError> {
        let offset = calculate_2d_offset(row, col, &self.shape, &self.strides)?;
        Ok(self.data.as_mut_ptr().add(offset))
    }

    pub fn shape(&self) -> &[usize] { &self.shape }

    /// Returns a slice of the underlying data.
    #[inline(always)]
    pub fn data(&self) -> &[f32] {
        self.data
    }

    /// Returns a mutable slice of the underlying data.
    #[inline(always)]
    pub fn data_mut(&mut self) -> &mut [f32] {
        self.data
    }
}