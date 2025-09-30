//! Core SIMD infrastructure and feature detection
//!
//! This module provides CPU feature detection and SIMD capabilities checking
//! for optimal performance on different architectures.

#[cfg(feature = "simd")]
pub use wide::*;

// CPU feature detection
#[cfg(not(feature = "std"))]
use spin::Once;
#[cfg(feature = "std")]
use std::sync::Once;

static INIT: Once = Once::new();
static mut HAS_AVX2: bool = false;
static mut HAS_AVX512: bool = false;
static mut HAS_NEON: bool = false;

/// Initialize SIMD feature detection
fn init_simd_features() {
    INIT.call_once(|| {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    HAS_AVX2 = true;
                }
            }
            if is_x86_feature_detected!("avx512f") {
                unsafe {
                    HAS_AVX512 = true;
                }
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            // NEON is standard on aarch64
            unsafe {
                HAS_NEON = true;
            }
        }
    });
}

/// Check if AVX2 is available
pub fn has_avx2() -> bool {
    init_simd_features();
    unsafe { HAS_AVX2 }
}

/// Check if AVX-512 is available
pub fn has_avx512() -> bool {
    init_simd_features();
    unsafe { HAS_AVX512 }
}

/// Check if ARM NEON is available
pub fn has_neon() -> bool {
    init_simd_features();
    unsafe { HAS_NEON }
}

/// Determine if SIMD should be used for a given array size
pub fn should_use_simd(len: usize) -> bool {
    // SIMD is beneficial for arrays larger than the vector width
    // f32x8 = 8 elements minimum
    len >= 16 && (has_avx2() || has_avx512() || has_neon())
}

/// Get optimal SIMD chunk size for given type
pub fn optimal_simd_chunk_size<T>() -> usize {
    let type_size = std::mem::size_of::<T>();

    if has_avx512() {
        // AVX-512 can handle 512 bits = 64 bytes
        64 / type_size
    } else if has_avx2() {
        // AVX2 can handle 256 bits = 32 bytes
        32 / type_size
    } else if has_neon() {
        // NEON can handle 128 bits = 16 bytes
        16 / type_size
    } else {
        // Fallback to scalar processing
        1
    }
}

/// SIMD vector width constants
pub mod vector_width {
    pub const F32_AVX512: usize = 16; // 512 bits / 32 bits = 16 f32s
    pub const F32_AVX2: usize = 8; // 256 bits / 32 bits = 8 f32s
    pub const F32_NEON: usize = 4; // 128 bits / 32 bits = 4 f32s

    pub const F64_AVX512: usize = 8; // 512 bits / 64 bits = 8 f64s
    pub const F64_AVX2: usize = 4; // 256 bits / 64 bits = 4 f64s
    pub const F64_NEON: usize = 2; // 128 bits / 64 bits = 2 f64s

    pub const I32_AVX512: usize = 16; // 512 bits / 32 bits = 16 i32s
    pub const I32_AVX2: usize = 8; // 256 bits / 32 bits = 8 i32s
    pub const I32_NEON: usize = 4; // 128 bits / 32 bits = 4 i32s
}

/// Get the current SIMD vector width for f32
pub fn f32_vector_width() -> usize {
    if has_avx512() {
        vector_width::F32_AVX512
    } else if has_avx2() {
        vector_width::F32_AVX2
    } else if has_neon() {
        vector_width::F32_NEON
    } else {
        1
    }
}

/// Get the current SIMD vector width for f64
pub fn f64_vector_width() -> usize {
    if has_avx512() {
        vector_width::F64_AVX512
    } else if has_avx2() {
        vector_width::F64_AVX2
    } else if has_neon() {
        vector_width::F64_NEON
    } else {
        1
    }
}

/// Get the current SIMD vector width for i32
pub fn i32_vector_width() -> usize {
    if has_avx512() {
        vector_width::I32_AVX512
    } else if has_avx2() {
        vector_width::I32_AVX2
    } else if has_neon() {
        vector_width::I32_NEON
    } else {
        1
    }
}

/// SIMD alignment helpers
pub mod alignment {
    /// Check if pointer is properly aligned for SIMD operations
    pub fn is_aligned<T>(ptr: *const T, alignment: usize) -> bool {
        (ptr as usize).is_multiple_of(alignment)
    }

    /// Get the alignment requirement for current SIMD instruction set
    pub fn simd_alignment() -> usize {
        if crate::cpu::simd::core::has_avx512() {
            64 // AVX-512 prefers 64-byte alignment
        } else if crate::cpu::simd::core::has_avx2() {
            32 // AVX2 prefers 32-byte alignment
        } else {
            16 // NEON and SSE prefer 16-byte alignment
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_detection() {
        // Just verify these don't panic
        let _avx2 = has_avx2();
        let _avx512 = has_avx512();
        let _neon = has_neon();
    }

    #[test]
    fn test_should_use_simd() {
        assert!(!should_use_simd(4)); // Too small
        assert!(!should_use_simd(8)); // Still too small
                                      // Length 16+ should use SIMD if features are available
                                      // Result depends on actual CPU features
        let _result = should_use_simd(16);
    }

    #[test]
    fn test_optimal_chunk_size() {
        let f32_chunk = optimal_simd_chunk_size::<f32>();
        let f64_chunk = optimal_simd_chunk_size::<f64>();

        // Should return reasonable chunk sizes
        assert!(f32_chunk >= 1);
        assert!(f64_chunk >= 1);
        assert!(f32_chunk >= f64_chunk); // f32 chunks should be larger than f64
    }

    #[test]
    fn test_vector_widths() {
        let f32_width = f32_vector_width();
        let f64_width = f64_vector_width();
        let i32_width = i32_vector_width();

        assert!(f32_width >= 1);
        assert!(f64_width >= 1);
        assert!(i32_width >= 1);
    }

    #[test]
    fn test_alignment() {
        let test_array = [1.0f32; 16];
        let ptr = test_array.as_ptr();

        // Test alignment check
        let is_16_aligned = alignment::is_aligned(ptr, 16);
        let _is_32_aligned = alignment::is_aligned(ptr, 32);

        // At least 16-byte alignment should be common
        // (though not guaranteed in all contexts)
        println!("16-byte aligned: {}", is_16_aligned);

        // Verify alignment requirement is reasonable
        let req = alignment::simd_alignment();
        assert!(req == 16 || req == 32 || req == 64);
    }
}
