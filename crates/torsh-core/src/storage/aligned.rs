//! Aligned storage for SIMD-optimized tensor operations
//!
//! This module provides memory-aligned storage allocations that are optimized
//! for SIMD operations, ensuring proper alignment for AVX2, NEON, and other
//! SIMD instruction sets.
//!
//! # SciRS2 POLICY COMPLIANCE
//!
//! ## Which AlignedVec to Use?
//!
//! There are two AlignedVec implementations available:
//!
//! 1. **`torsh_core::simd::AlignedVec`** (RECOMMENDED for new code)
//!    - From scirs2-core::simd_aligned
//!    - Production-ready with 2-4x SIMD speedup guarantees
//!    - Automatically selects optimal alignment for platform
//!    - Integrates with scirs2-core SIMD operations
//!    - USE THIS for high-performance SIMD operations
//!
//! 2. **`torsh_core::storage::aligned::AlignedVec`** (Legacy/fallback)
//!    - Custom implementation in this module
//!    - Used when simd feature is disabled
//!    - Provides basic aligned allocation
//!    - USE THIS only for storage internals or when simd feature unavailable
//!
//! ## Migration Path
//!
//! For optimal performance with SciRS2 POLICY compliance:
//!
//! ```rust,ignore
//! // ❌ OLD (still works but not optimal)
//! use torsh_core::storage::aligned::AlignedVec;
//! let data = AlignedVec::<f32>::new();
//!
//! // ✅ NEW (recommended for SIMD operations)
//! #[cfg(feature = "simd")]
//! use torsh_core::simd::AlignedVec;
//! #[cfg(feature = "simd")]
//! let data = AlignedVec::<f32>::new()?;  // Note: returns Result
//! ```
//!
//! This custom implementation is maintained for:
//! - Storage layer internals that don't require SIMD
//! - Fallback when simd feature is disabled
//! - Custom alignment requirements not covered by scirs2-core

use std::alloc::{alloc, dealloc, Layout};
use std::fmt;
use std::marker::PhantomData;
use std::ptr::NonNull;

/// Platform-specific alignment requirements for SIMD
pub mod alignment {
    /// Alignment for AVX-512 (64 bytes)
    pub const AVX512: usize = 64;
    /// Alignment for AVX/AVX2 (32 bytes)
    pub const AVX2: usize = 32;
    /// Alignment for SSE (16 bytes)
    pub const SSE: usize = 16;
    /// Alignment for ARM NEON (16 bytes)
    pub const NEON: usize = 16;

    /// Get recommended alignment for current platform
    #[cfg(target_arch = "x86_64")]
    pub const fn recommended() -> usize {
        #[cfg(target_feature = "avx512f")]
        {
            AVX512
        }
        #[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
        {
            AVX2
        }
        #[cfg(all(not(target_feature = "avx2"), not(target_feature = "avx512f")))]
        {
            SSE
        }
    }

    #[cfg(target_arch = "aarch64")]
    pub const fn recommended() -> usize {
        NEON
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    pub const fn recommended() -> usize {
        16 // Conservative default
    }
}

/// Aligned vector for SIMD operations
///
/// This type provides a memory-aligned container that's optimized for SIMD
/// operations. The alignment is automatically selected based on the platform's
/// SIMD capabilities.
///
/// # Examples
///
/// ```ignore
/// use torsh_core::storage::aligned::AlignedVec;
///
/// // Create an aligned vector for f32 values
/// let mut vec: AlignedVec<f32> = AlignedVec::with_capacity(1024);
///
/// // Fill with data
/// for i in 0..1024 {
///     vec.push(i as f32);
/// }
///
/// // Use in SIMD operations
/// let data = vec.as_slice();
/// // ... SIMD operations on data ...
/// ```
pub struct AlignedVec<T> {
    ptr: NonNull<T>,
    len: usize,
    capacity: usize,
    alignment: usize,
    _marker: PhantomData<T>,
}

unsafe impl<T: Send> Send for AlignedVec<T> {}
unsafe impl<T: Sync> Sync for AlignedVec<T> {}

impl<T> AlignedVec<T> {
    /// Create a new empty aligned vector with default alignment
    pub fn new() -> Self {
        Self::with_alignment(alignment::recommended())
    }

    /// Create a new aligned vector with specified alignment
    pub fn with_alignment(alignment: usize) -> Self {
        assert!(alignment.is_power_of_two(), "Alignment must be power of 2");
        assert!(
            alignment >= std::mem::align_of::<T>(),
            "Alignment must be at least type alignment"
        );

        Self {
            ptr: NonNull::dangling(),
            len: 0,
            capacity: 0,
            alignment,
            _marker: PhantomData,
        }
    }

    /// Create a new aligned vector with specified capacity
    pub fn with_capacity(capacity: usize) -> Self {
        let mut vec = Self::new();
        vec.reserve(capacity);
        vec
    }

    /// Create a new aligned vector with specified capacity and alignment
    pub fn with_capacity_and_alignment(capacity: usize, alignment: usize) -> Self {
        let mut vec = Self::with_alignment(alignment);
        vec.reserve(capacity);
        vec
    }

    /// Create from a standard Vec (will reallocate with alignment)
    pub fn from_vec(vec: Vec<T>) -> Self {
        let mut aligned = Self::with_capacity(vec.len());
        for item in vec {
            aligned.push(item);
        }
        aligned
    }

    /// Reserve additional capacity
    pub fn reserve(&mut self, additional: usize) {
        let new_capacity = self.len + additional;
        if new_capacity <= self.capacity {
            return;
        }

        // Calculate new capacity (grow by 2x)
        let new_capacity = new_capacity.max(self.capacity * 2).max(4);

        unsafe {
            let layout = Layout::from_size_align_unchecked(
                new_capacity * std::mem::size_of::<T>(),
                self.alignment,
            );

            let new_ptr = if self.capacity == 0 {
                alloc(layout) as *mut T
            } else {
                let old_layout = Layout::from_size_align_unchecked(
                    self.capacity * std::mem::size_of::<T>(),
                    self.alignment,
                );

                // Allocate new memory and copy
                let new_ptr = alloc(layout) as *mut T;
                if !new_ptr.is_null() {
                    std::ptr::copy_nonoverlapping(self.ptr.as_ptr(), new_ptr, self.len);
                    dealloc(self.ptr.as_ptr() as *mut u8, old_layout);
                }
                new_ptr
            };

            if new_ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }

            self.ptr = NonNull::new_unchecked(new_ptr);
            self.capacity = new_capacity;
        }
    }

    /// Push a value to the end
    pub fn push(&mut self, value: T) {
        if self.len == self.capacity {
            self.reserve(1);
        }

        unsafe {
            std::ptr::write(self.ptr.as_ptr().add(self.len), value);
        }
        self.len += 1;
    }

    /// Pop a value from the end
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            None
        } else {
            self.len -= 1;
            unsafe { Some(std::ptr::read(self.ptr.as_ptr().add(self.len))) }
        }
    }

    /// Get length
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get alignment
    pub fn alignment(&self) -> usize {
        self.alignment
    }

    /// Get as slice
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Get as mutable slice
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// Get pointer
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    /// Get mutable pointer
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    /// Check if pointer is properly aligned
    pub fn is_aligned(&self) -> bool {
        // Empty vecs with dangling pointers are considered aligned
        if self.capacity == 0 {
            return true;
        }
        (self.ptr.as_ptr() as usize) % self.alignment == 0
    }

    /// Clear all elements
    pub fn clear(&mut self) {
        unsafe {
            std::ptr::drop_in_place(std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len));
        }
        self.len = 0;
    }

    /// Shrink capacity to fit length
    pub fn shrink_to_fit(&mut self) {
        if self.capacity > self.len {
            if self.len == 0 {
                if self.capacity > 0 {
                    unsafe {
                        let layout = Layout::from_size_align_unchecked(
                            self.capacity * std::mem::size_of::<T>(),
                            self.alignment,
                        );
                        dealloc(self.ptr.as_ptr() as *mut u8, layout);
                    }
                    self.ptr = NonNull::dangling();
                    self.capacity = 0;
                }
            } else {
                unsafe {
                    let old_layout = Layout::from_size_align_unchecked(
                        self.capacity * std::mem::size_of::<T>(),
                        self.alignment,
                    );
                    let new_layout = Layout::from_size_align_unchecked(
                        self.len * std::mem::size_of::<T>(),
                        self.alignment,
                    );

                    let new_ptr = alloc(new_layout) as *mut T;
                    if !new_ptr.is_null() {
                        std::ptr::copy_nonoverlapping(self.ptr.as_ptr(), new_ptr, self.len);
                        dealloc(self.ptr.as_ptr() as *mut u8, old_layout);
                        self.ptr = NonNull::new_unchecked(new_ptr);
                        self.capacity = self.len;
                    }
                }
            }
        }
    }
}

impl<T> Default for AlignedVec<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Drop for AlignedVec<T> {
    fn drop(&mut self) {
        if self.capacity > 0 {
            unsafe {
                // Drop all elements
                std::ptr::drop_in_place(std::slice::from_raw_parts_mut(
                    self.ptr.as_ptr(),
                    self.len,
                ));

                // Deallocate memory
                let layout = Layout::from_size_align_unchecked(
                    self.capacity * std::mem::size_of::<T>(),
                    self.alignment,
                );
                dealloc(self.ptr.as_ptr() as *mut u8, layout);
            }
        }
    }
}

impl<T: Clone> Clone for AlignedVec<T> {
    fn clone(&self) -> Self {
        let mut new_vec = Self::with_capacity_and_alignment(self.len, self.alignment);
        for item in self.as_slice() {
            new_vec.push(item.clone());
        }
        new_vec
    }
}

impl<T: fmt::Debug> fmt::Debug for AlignedVec<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AlignedVec")
            .field("len", &self.len)
            .field("capacity", &self.capacity)
            .field("alignment", &self.alignment)
            .field("data", &self.as_slice())
            .finish()
    }
}

impl<T> std::ops::Deref for AlignedVec<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T> std::ops::DerefMut for AlignedVec<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<T> std::ops::Index<usize> for AlignedVec<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.as_slice()[index]
    }
}

impl<T> std::ops::IndexMut<usize> for AlignedVec<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.as_mut_slice()[index]
    }
}

/// Alignment check utilities
pub struct AlignmentChecker;

impl AlignmentChecker {
    /// Check if a pointer is aligned to the specified alignment
    pub fn is_aligned<T>(ptr: *const T, alignment: usize) -> bool {
        (ptr as usize) % alignment == 0
    }

    /// Check if a slice is properly aligned for SIMD operations
    pub fn is_simd_aligned<T>(slice: &[T]) -> bool {
        let alignment = alignment::recommended();
        Self::is_aligned(slice.as_ptr(), alignment)
    }

    /// Get misalignment offset (how many bytes to skip to reach alignment)
    pub fn misalignment_offset<T>(ptr: *const T, alignment: usize) -> usize {
        let addr = ptr as usize;
        let remainder = addr % alignment;
        if remainder == 0 {
            0
        } else {
            alignment - remainder
        }
    }

    /// Calculate optimal alignment for a given element size and count
    pub fn optimal_alignment(element_size: usize, count: usize) -> usize {
        let total_size = element_size * count;

        // Choose alignment based on size
        if total_size >= 64 * 1024 {
            // Large tensors: use maximum alignment
            alignment::recommended()
        } else if total_size >= 4 * 1024 {
            // Medium tensors: use AVX2/NEON alignment
            32.min(alignment::recommended())
        } else {
            // Small tensors: use SSE/NEON alignment
            16.min(alignment::recommended())
        }
    }
}

/// SIMD-friendly memory layout analyzer
pub struct SimdLayoutAnalyzer;

impl SimdLayoutAnalyzer {
    /// Analyze if a memory layout is SIMD-friendly
    pub fn analyze<T>(data: &[T]) -> SimdLayoutAnalysis {
        let ptr = data.as_ptr();
        let alignment = alignment::recommended();
        let is_aligned = AlignmentChecker::is_aligned(ptr, alignment);

        let element_size = std::mem::size_of::<T>();
        let _total_size = element_size * data.len();

        // Calculate SIMD width (elements per SIMD register)
        let simd_width = alignment / element_size;
        let aligned_elements = (data.len() / simd_width) * simd_width;
        let remainder_elements = data.len() % simd_width;

        SimdLayoutAnalysis {
            is_aligned,
            alignment,
            element_size,
            total_elements: data.len(),
            aligned_elements,
            remainder_elements,
            simd_width,
            expected_speedup: if is_aligned && remainder_elements < simd_width / 2 {
                simd_width as f64 * 0.9 // Expect ~90% of theoretical speedup
            } else if is_aligned {
                simd_width as f64 * 0.7 // Expect ~70% with remainder
            } else {
                1.0 // No speedup if unaligned
            },
        }
    }
}

/// SIMD layout analysis result
#[derive(Debug, Clone)]
pub struct SimdLayoutAnalysis {
    /// Whether the data is properly aligned
    pub is_aligned: bool,
    /// Alignment in bytes
    pub alignment: usize,
    /// Size of each element in bytes
    pub element_size: usize,
    /// Total number of elements
    pub total_elements: usize,
    /// Number of elements that can be processed with SIMD
    pub aligned_elements: usize,
    /// Number of remainder elements (scalar processing)
    pub remainder_elements: usize,
    /// SIMD width (elements per vector)
    pub simd_width: usize,
    /// Expected speedup with SIMD
    pub expected_speedup: f64,
}

impl SimdLayoutAnalysis {
    /// Get a recommendation for this layout
    pub fn recommendation(&self) -> String {
        if !self.is_aligned {
            format!(
                "❌ Unaligned memory: no SIMD acceleration possible. \
                 Use AlignedVec for {:.1}x speedup.",
                self.simd_width as f64
            )
        } else if self.remainder_elements > self.simd_width / 2 {
            format!(
                "⚠️  Aligned but {} remainder elements. Consider padding to multiple of {} \
                 for {:.1}x → {:.1}x speedup.",
                self.remainder_elements,
                self.simd_width,
                self.expected_speedup,
                self.simd_width as f64
            )
        } else {
            format!(
                "✓ Optimal SIMD layout: {}/{} elements aligned ({:.1}% coverage), \
                 expected {:.1}x speedup.",
                self.aligned_elements,
                self.total_elements,
                (self.aligned_elements as f64 / self.total_elements as f64) * 100.0,
                self.expected_speedup
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aligned_vec_creation() {
        let vec: AlignedVec<f32> = AlignedVec::new();
        assert_eq!(vec.len(), 0);
        assert_eq!(vec.capacity(), 0);
        assert!(vec.is_aligned());
    }

    #[test]
    fn test_aligned_vec_push() {
        let mut vec = AlignedVec::new();
        for i in 0..100 {
            vec.push(i as f32);
        }
        assert_eq!(vec.len(), 100);
        assert!(vec.is_aligned());
        assert_eq!(vec[50], 50.0);
    }

    #[test]
    fn test_aligned_vec_from_vec() {
        let std_vec: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let aligned_vec = AlignedVec::from_vec(std_vec);

        assert_eq!(aligned_vec.len(), 100);
        assert!(aligned_vec.is_aligned());
        assert_eq!(aligned_vec[75], 75.0);
    }

    #[test]
    fn test_alignment_checker() {
        let vec: AlignedVec<f32> = AlignedVec::with_capacity(128);
        assert!(AlignmentChecker::is_simd_aligned(vec.as_slice()));

        let alignment = alignment::recommended();
        assert!(AlignmentChecker::is_aligned(vec.as_ptr(), alignment));
    }

    #[test]
    fn test_simd_layout_analyzer() {
        let vec: AlignedVec<f32> = AlignedVec::from_vec((0..1000).map(|i| i as f32).collect());
        let analysis = SimdLayoutAnalyzer::analyze(vec.as_slice());

        assert!(analysis.is_aligned);
        assert!(analysis.aligned_elements > 0);
        assert!(analysis.expected_speedup > 1.0);
        assert!(analysis.recommendation().contains("✓"));
    }

    #[test]
    fn test_aligned_vec_clear() {
        let mut vec: AlignedVec<i32> = AlignedVec::from_vec(vec![1, 2, 3, 4, 5]);
        assert_eq!(vec.len(), 5);

        vec.clear();
        assert_eq!(vec.len(), 0);
        assert!(vec.is_empty());
    }

    #[test]
    fn test_aligned_vec_pop() {
        let mut vec: AlignedVec<i32> = AlignedVec::from_vec(vec![1, 2, 3]);
        assert_eq!(vec.pop(), Some(3));
        assert_eq!(vec.pop(), Some(2));
        assert_eq!(vec.len(), 1);
    }

    #[test]
    fn test_optimal_alignment() {
        let alignment = AlignmentChecker::optimal_alignment(4, 1000);
        assert!(alignment >= 16);
        assert!(alignment.is_power_of_two());
    }
}
