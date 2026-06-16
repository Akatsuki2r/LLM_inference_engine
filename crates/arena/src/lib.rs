use std::alloc::{alloc, Layout};
use std::ptr::NonNull;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ArenaError {
    #[error("Arena overflow: requested {requested} bytes, but only {available} bytes remain")]
    Overflow { requested: usize, available: usize },
    #[error("Allocation failed due to system memory limit")]
    AllocationFailed,
}

/// The memory categories for Project Aether.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryCategory {
    Weights,
    Activations,
    KvCache,
    Scratch,
}

/// A UnifiedArena manages one giant contiguous block of memory.
/// It enforces 64-byte alignment for all allocations to satisfy AVX2/FMA3 requirements.
pub struct UnifiedArena {
    base_ptr: NonNull<u8>,
    capacity: usize,
    offset: usize,
    layout: Layout,
}

impl UnifiedArena {
    /// Create a new Arena with a total capacity in bytes.
    pub fn new(capacity: usize) -> Result<Self, ArenaError> {
        // Ensure 64-byte alignment for the base pointer
        let layout = Layout::from_size_align(capacity, 64)
            .map_err(|_| ArenaError::AllocationFailed)?;

        let ptr = unsafe { alloc(layout) };
        let base_ptr = NonNull::new(ptr).ok_or(ArenaError::AllocationFailed)?;

        Ok(Self {
            base_ptr,
            capacity,
            offset: 0,
            layout,
        })
    }

    /// Allocate a chunk of memory from the arena.
    /// Enforces 64-byte alignment for every allocation.
    pub fn alloc(&mut self, size: usize, _category: MemoryCategory) -> Result<*mut u8, ArenaError> {
        // Calculate alignment padding to ensure the NEXT pointer is also 64-byte aligned
        let align = 64;
        let current_ptr = self.base_ptr.as_ptr() as usize;
        let current_offset_ptr = current_ptr.checked_add(self.offset).ok_or(ArenaError::AllocationFailed)?;
        
        let padding = (align - (current_offset_ptr % align)) % align;

        let total_needed = size.checked_add(padding).ok_or(ArenaError::AllocationFailed)?;

        let next_offset = self.offset.checked_add(total_needed).ok_or(ArenaError::Overflow {
            requested: total_needed,
            available: self.capacity.saturating_sub(self.offset),
        })?;

        if next_offset > self.capacity {
            return Err(ArenaError::Overflow {
                requested: total_needed,
                available: self.capacity.saturating_sub(self.offset),
            });
        }

        let alloc_ptr = unsafe { self.base_ptr.as_ptr().add(self.offset + padding) };
        self.offset = next_offset;

        Ok(alloc_ptr)
    }

    /// Allocate a chunk of memory and return it as a byte slice.
    pub fn alloc_slice(&mut self, size: usize, _category: MemoryCategory) -> Result<&mut [u8], ArenaError> {
        let ptr = self.alloc(size, _category)?;
        Ok(unsafe { std::slice::from_raw_parts_mut(ptr, size) })
    }

    pub fn current_offset(&self) -> usize {
        self.offset
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn usage_fraction(&self) -> f64 {
        self.offset as f64 / self.capacity as f64
    }
}

impl Drop for UnifiedArena {
    fn drop(&mut self) {
        unsafe {
            std::alloc::dealloc(self.base_ptr.as_ptr(), self.layout);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_alignment() {
        let mut arena = UnifiedArena::new(1024).unwrap();

        // Allocate various sizes
        for size in [1, 7, 13, 63, 64, 65] {
            let ptr = arena.alloc(size, MemoryCategory::Weights).unwrap();
            assert_eq!(ptr as usize % 64, 0, "Pointer {:p} must be 64-byte aligned", ptr);
        }
    }

    #[test]
    fn test_arena_overflow() {
        let mut arena = UnifiedArena::new(64).unwrap();
        assert!(arena.alloc(64, MemoryCategory::Weights).is_ok());
        assert!(arena.alloc(1, MemoryCategory::Weights).is_err());
    }
}
