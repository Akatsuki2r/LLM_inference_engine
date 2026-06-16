use thiserror::Error;

#[derive(Error, Debug, PartialEq, Eq, Clone)]
pub enum TensorError {
    #[error("Dimension mismatch: expected {expected:?}, got {actual:?}")]
    DimensionMismatch { expected: Vec<usize>, actual: Vec<usize> },
    #[error("Dimension mismatch")]
    SimpleDimensionMismatch,
    #[error("Index out of bounds")]
    IndexOutOfBounds,
    #[error("Invalid shape: product of dimensions does not match buffer size")]
    InvalidShape,
    #[error("Alignment mismatch: pointer is not properly aligned for the data type")]
    AlignmentMismatch,
    #[error("Integer overflow during offset calculation")]
    Overflow,
}

/// Supported data types for tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorType {
    F32,
    F16,
    Q4_0,
}

const MAX_DIMS: usize = 8;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    dims: [usize; MAX_DIMS],
    len: usize,
}

impl Shape {
    pub fn new(slice: &[usize]) -> Result<Self, TensorError> {
        if slice.len() > MAX_DIMS {
            return Err(TensorError::SimpleDimensionMismatch);
        }
        let mut dims = [0; MAX_DIMS];
        dims[..slice.len()].copy_from_slice(slice);
        Ok(Self { dims, len: slice.len() })
    }

    pub fn as_slice(&self) -> &[usize] {
        &self.dims[..self.len]
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn product(&self) -> Result<usize, TensorError> {
        let mut p = 1usize;
        for &d in self.as_slice() {
            p = p.checked_mul(d).ok_or(TensorError::Overflow)?;
        }
        Ok(p)
    }
}

/// Centralized coordinate-to-offset computer.
#[inline(always)]
fn calculate_flat_offset(indices: &[usize], shape: &Shape, strides: &Shape) -> Result<usize, TensorError> {
    if indices.len() != shape.len() {
        return Err(TensorError::SimpleDimensionMismatch);
    }
    let mut flat_index = 0usize;
    for i in 0..indices.len() {
        if indices[i] >= shape.as_slice()[i] {
            return Err(TensorError::IndexOutOfBounds);
        }
        let term = indices[i].checked_mul(strides.as_slice()[i]).ok_or(TensorError::Overflow)?;
        flat_index = flat_index.checked_add(term).ok_or(TensorError::Overflow)?;
    }
    Ok(flat_index)
}

/// Fast 2D matrix offset extractor.
#[inline(always)]
fn calculate_2d_offset(row: usize, col: usize, shape: &Shape, strides: &Shape) -> Result<usize, TensorError> {
    if shape.len() != 2 {
        return Err(TensorError::SimpleDimensionMismatch);
    }
    if row >= shape.as_slice()[0] || col >= shape.as_slice()[1] {
        return Err(TensorError::IndexOutOfBounds);
    }
    let r_term = row.checked_mul(strides.as_slice()[0]).ok_or(TensorError::Overflow)?;
    let c_term = col.checked_mul(strides.as_slice()[1]).ok_or(TensorError::Overflow)?;
    r_term.checked_add(c_term).ok_or(TensorError::Overflow)
}

/// Read-Only view designed specifically for memory-mapped weights.
pub struct TensorView<'a> {
    data: &'a [u8],
    shape: Shape,
    strides: Shape,
    tensor_type: TensorType,
}

impl<'a> TensorView<'a> {
    pub fn from_raw_parts(raw_bytes: &'a [u8], shape_slice: &[usize], tensor_type: TensorType) -> Result<Self, TensorError> {
        let element_size = tensor_type.size();
        
        // Alignment and size validation
        if tensor_type == TensorType::F32 {
            if raw_bytes.as_ptr() as usize % 4 != 0 {
                return Err(TensorError::AlignmentMismatch);
            }
            if raw_bytes.len() % 4 != 0 {
                return Err(TensorError::InvalidShape);
            }
        }

        let element_count = if element_size > 0 {
            raw_bytes.len() / element_size
        } else {
            // Q4_0: 2 elements per byte
            raw_bytes.len().checked_mul(2).ok_or(TensorError::Overflow)?
        };

        let shape = Shape::new(shape_slice)?;
        if shape.product()? > element_count {
            return Err(TensorError::InvalidShape);
        }

        let mut strides_arr = [0; MAX_DIMS];
        let mut current_stride = 1usize;
        for i in (0..shape.len()).rev() {
            strides_arr[i] = current_stride;
            current_stride = current_stride.checked_mul(shape.as_slice()[i]).ok_or(TensorError::Overflow)?;
        }
        let strides = Shape::new(&strides_arr[..shape.len()])?;

        Ok(Self { data: raw_bytes, shape, strides, tensor_type })
    }

    #[inline]
    pub fn get(&self, indices: &[usize]) -> Result<f32, TensorError> {
        let offset = calculate_flat_offset(indices, &self.shape, &self.strides)?;
        Ok(self.get_element_as_f32(offset)?)
    }

    #[inline(always)]
    pub fn get_2d(&self, row: usize, col: usize) -> Result<f32, TensorError> {
        let offset = calculate_2d_offset(row, col, &self.shape, &self.strides)?;
        Ok(self.get_element_as_f32(offset)?)
    }

    fn get_element_as_f32(&self, offset: usize) -> Result<f32, TensorError> {
        match self.tensor_type {
            TensorType::F32 => {
                let byte_offset = offset.checked_mul(4).ok_or(TensorError::Overflow)?;
                if byte_offset.checked_add(4).ok_or(TensorError::Overflow)? > self.data.len() {
                    return Err(TensorError::IndexOutOfBounds);
                }
                unsafe {
                    let ptr = self.data.as_ptr().add(byte_offset) as *const f32;
                    Ok(core::ptr::read_unaligned(ptr))
                }
            }
            TensorType::F16 => {
                let byte_offset = offset.checked_mul(2).ok_or(TensorError::Overflow)?;
                if byte_offset.checked_add(2).ok_or(TensorError::Overflow)? > self.data.len() {
                    return Err(TensorError::IndexOutOfBounds);
                }
                unsafe {
                    let ptr = self.data.as_ptr().add(byte_offset) as *const F16Half;
                    Ok(core::ptr::read_unaligned(ptr).to_f32())
                }
            }
            TensorType::Q4_0 => Err(TensorError::IndexOutOfBounds),
        }
    }

    pub fn transpose(&self) -> Result<Self, TensorError> {
        if self.shape.len() != 2 { return Err(TensorError::SimpleDimensionMismatch); }
        let new_shape = Shape::new(&[self.shape.as_slice()[1], self.shape.as_slice()[0]])?;
        let new_strides = Shape::new(&[self.strides.as_slice()[1], self.strides.as_slice()[0]])?;
        Ok(Self {
            data: self.data,
            shape: new_shape,
            strides: new_strides,
            tensor_type: self.tensor_type,
        })
    }

    pub fn shape(&self) -> &[usize] { self.shape.as_slice() }
    pub fn data(&self) -> &[u8] { self.data }
    pub fn tensor_type(&self) -> TensorType { self.tensor_type }
}

pub struct TensorMut<'a> {
    data: &'a mut [u8],
    shape: Shape,
    strides: Shape,
    tensor_type: TensorType,
}

impl<'a> TensorMut<'a> {
    pub fn from_raw_parts(raw_bytes: &'a mut [u8], shape_slice: &[usize], tensor_type: TensorType) -> Result<Self, TensorError> {
        let element_size = tensor_type.size();
        
        // Alignment and size validation
        if tensor_type == TensorType::F32 {
            if raw_bytes.as_ptr() as usize % 4 != 0 {
                return Err(TensorError::AlignmentMismatch);
            }
            if raw_bytes.len() % 4 != 0 {
                return Err(TensorError::InvalidShape);
            }
        }

        let element_count = if element_size > 0 {
            raw_bytes.len() / element_size
        } else {
            raw_bytes.len().checked_mul(2).ok_or(TensorError::Overflow)?
        };

        let shape = Shape::new(shape_slice)?;
        if shape.product()? > element_count {
            return Err(TensorError::InvalidShape);
        }

        let mut strides_arr = [0; MAX_DIMS];
        let mut current_stride = 1usize;
        for i in (0..shape.len()).rev() {
            strides_arr[i] = current_stride;
            current_stride = current_stride.checked_mul(shape.as_slice()[i]).ok_or(TensorError::Overflow)?;
        }
        let strides = Shape::new(&strides_arr[..shape.len()])?;

        Ok(Self { data: raw_bytes, shape, strides, tensor_type })
    }

    #[inline]
    pub fn get(&self, indices: &[usize]) -> Result<f32, TensorError> {
        let offset = calculate_flat_offset(indices, &self.shape, &self.strides)?;
        Ok(self.get_element_as_f32(offset)?)
    }

    #[inline]
    pub fn get_mut(&mut self, indices: &[usize]) -> Result<&mut f32, TensorError> {
        let offset = calculate_flat_offset(indices, &self.shape, &self.strides)?;
        self.get_element_as_f32_mut(offset)
    }

    #[inline(always)]
    pub fn get_2d(&self, row: usize, col: usize) -> Result<f32, TensorError> {
        let offset = calculate_2d_offset(row, col, &self.shape, &self.strides)?;
        Ok(self.get_element_as_f32(offset)?)
    }

    #[inline(always)]
    pub fn set_2d(&mut self, row: usize, col: usize, val: f32) -> Result<(), TensorError> {
        let offset = calculate_2d_offset(row, col, &self.shape, &self.strides)?;
        self.set_element_from_f32(offset, val)
    }

    fn get_element_as_f32(&self, offset: usize) -> Result<f32, TensorError> {
        match self.tensor_type {
            TensorType::F32 => {
                let byte_offset = offset.checked_mul(4).ok_or(TensorError::Overflow)?;
                if byte_offset.checked_add(4).ok_or(TensorError::Overflow)? > self.data.len() {
                    return Err(TensorError::IndexOutOfBounds);
                }
                unsafe {
                    let ptr = self.data.as_ptr().add(byte_offset) as *const f32;
                    Ok(core::ptr::read_unaligned(ptr))
                }
            }
            TensorType::F16 => {
                let byte_offset = offset.checked_mul(2).ok_or(TensorError::Overflow)?;
                if byte_offset.checked_add(2).ok_or(TensorError::Overflow)? > self.data.len() {
                    return Err(TensorError::IndexOutOfBounds);
                }
                unsafe {
                    let ptr = self.data.as_ptr().add(byte_offset) as *const F16Half;
                    Ok(core::ptr::read_unaligned(ptr).to_f32())
                }
            }
            TensorType::Q4_0 => Err(TensorError::IndexOutOfBounds),
        }
    }

    fn get_element_as_f32_mut(&mut self, offset: usize) -> Result<&mut f32, TensorError> {
        match self.tensor_type {
            TensorType::F32 => {
                let byte_offset = offset.checked_mul(4).ok_or(TensorError::Overflow)?;
                if byte_offset.checked_add(4).ok_or(TensorError::Overflow)? > self.data.len() {
                    return Err(TensorError::IndexOutOfBounds);
                }
                Ok(unsafe { &mut *(self.data.as_mut_ptr().add(byte_offset) as *mut f32) })
            }
            TensorType::F16 => {
                let byte_offset = offset.checked_mul(2).ok_or(TensorError::Overflow)?;
                if byte_offset.checked_add(2).ok_or(TensorError::Overflow)? > self.data.len() {
                    return Err(TensorError::IndexOutOfBounds);
                }
                Ok(unsafe { F16Half::to_f32_mut(&mut *(self.data.as_mut_ptr().add(byte_offset) as *mut F16Half)) })
            }
            TensorType::Q4_0 => Err(TensorError::IndexOutOfBounds),
        }
    }

    fn set_element_from_f32(&mut self, offset: usize, val: f32) -> Result<(), TensorError> {
        match self.tensor_type {
            TensorType::F32 => {
                let byte_offset = offset.checked_mul(4).ok_or(TensorError::Overflow)?;
                if byte_offset.checked_add(4).ok_or(TensorError::Overflow)? > self.data.len() {
                    return Err(TensorError::IndexOutOfBounds);
                }
                unsafe {
                    let ptr = self.data.as_mut_ptr().add(byte_offset) as *mut f32;
                    core::ptr::write_unaligned(ptr, val);
                }
                Ok(())
            }
            TensorType::F16 => {
                let byte_offset = offset.checked_mul(2).ok_or(TensorError::Overflow)?;
                if byte_offset.checked_add(2).ok_or(TensorError::Overflow)? > self.data.len() {
                    return Err(TensorError::IndexOutOfBounds);
                }
                unsafe {
                    let ptr = self.data.as_mut_ptr().add(byte_offset) as *mut F16Half;
                    core::ptr::write_unaligned(ptr, F16Half::from_f32(val));
                }
                Ok(())
            }
            TensorType::Q4_0 => Err(TensorError::IndexOutOfBounds),
        }
    }

    #[inline(always)]
    pub unsafe fn get_mut_2d_ptr(&mut self, row: usize, col: usize) -> Result<*mut f32, TensorError> {
        let offset = calculate_2d_offset(row, col, &self.shape, &self.strides)?;
        match self.tensor_type {
            TensorType::F32 => {
                let byte_offset = offset.checked_mul(4).ok_or(TensorError::Overflow)?;
                if byte_offset.checked_add(4).ok_or(TensorError::Overflow)? > self.data.len() {
                    return Err(TensorError::IndexOutOfBounds);
                }
                Ok(self.data.as_mut_ptr().add(byte_offset) as *mut f32)
            }
            _ => Err(TensorError::IndexOutOfBounds),
        }
    }

    pub fn shape(&self) -> &[usize] { self.shape.as_slice() }
    pub fn data(&self) -> &[u8] { self.data }
    pub fn data_mut(&mut self) -> &mut [u8] { self.data }
    pub fn tensor_type(&self) -> TensorType { self.tensor_type }
    
    pub fn copy_from_slice(&mut self, other: &TensorView) -> Result<(), TensorError> {
        if self.shape != other.shape || self.tensor_type != other.tensor_type {
            return Err(TensorError::SimpleDimensionMismatch);
        }
        self.data.copy_from_slice(other.data());
        Ok(())
    }
}

impl TensorType {
    pub fn size(&self) -> usize {
        match self {
            TensorType::F32 => 4,
            TensorType::F16 => 2,
            TensorType::Q4_0 => 0,
        }
    }
}

#[allow(non_camel_case_types)]
#[derive(Clone, Copy)]
pub struct F16Half(u16);

impl F16Half {
    #[inline(always)]
    pub const fn from_le_bytes(bytes: [u8; 2]) -> Self {
        Self(u16::from_le_bytes(bytes))
    }

    #[inline(always)]
    pub const fn to_le_bytes(self) -> [u8; 2] {
        self.0.to_le_bytes()
    }

    #[inline(always)]
    pub fn to_f32(self) -> f32 {
        let bits = self.0;
        let sign = if (bits >> 15) != 0 { -1.0 } else { 1.0 };
        let exponent = ((bits >> 10) & 0x1F) as i32;
        let mantissa = bits & 0x3FF;

        if exponent == 0 {
            if mantissa == 0 { sign * 0.0 }
            else { sign * (mantissa as f32) * 2.0f32.powi(-14 - 10) }
        } else if exponent == 0x1F {
            if mantissa == 0 { sign * f32::INFINITY }
            else { f32::NAN }
        } else {
            sign * (1.0 + (mantissa as f32) / 1024.0) * 2.0f32.powi(exponent - 15)
        }
    }

    #[inline(always)]
    pub fn from_f32(val: f32) -> Self {
        if val.is_nan() { return Self(0x7E00); }
        if val.is_infinite() {
            return if val.is_sign_positive() { Self(0x7C00) } else { Self(0xFC00) };
        }
        if val == 0.0 { return Self(if val.is_sign_positive() { 0x0000 } else { 0x8000 }); }

        let sign = if val.is_sign_positive() { 0u16 } else { 0x8000 };
        let abs_val = val.abs();
        let mut exponent = (abs_val.log2().floor() as i32) + 15;
        let mut mantissa;

        if exponent <= 0 {
            exponent = 0;
            mantissa = (abs_val * 2.0f32.powi(14 + 10)).round() as u16;
        } else if exponent >= 31 {
            return Self(sign | 0x7C00);
        } else {
            mantissa = ((abs_val / 2.0f32.powi(exponent - 15) - 1.0) * 1024.0).round() as u16;
            if mantissa >= 1024 {
                mantissa = 0;
                exponent += 1;
                if exponent >= 31 { return Self(sign | 0x7C00); }
            }
        }
        Self(sign | ((exponent as u16) << 10) | (mantissa & 0x3FF))
    }

    #[inline(always)]
    pub unsafe fn to_f32_mut<'a>(ptr: &'a mut Self) -> &'a mut f32 {
        &mut *(ptr as *mut Self as *mut f32)
    }
}
