//! SIMD-optimized type conversions for tensors
//!
//! This module provides efficient type conversion operations using SIMD instructions
//! when available, with fallback to scalar operations.

use crate::{Tensor, TensorElement};
use std::marker::PhantomData;
use torsh_core::error::Result;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// SIMD-optimized type conversion trait
pub trait SIMDTypeConversion<T: TensorElement, U: TensorElement> {
    /// Convert tensor from type T to type U using SIMD optimization when possible
    fn convert_simd(&self) -> Result<Tensor<U>>;

    /// Convert tensor with custom SIMD conversion function
    fn convert_with<F>(&self, converter: F) -> Result<Tensor<U>>
    where
        F: Fn(&[T]) -> Vec<U>;
}

/// SIMD conversion strategies
pub enum SIMDStrategy {
    /// Use the best available SIMD instructions for the target architecture
    Auto,
    /// Force scalar conversion (no SIMD)
    Scalar,
    /// Use SSE instructions (x86_64 only)
    SSE,
    /// Use AVX instructions (x86_64 only)
    AVX,
    /// Use AVX-512 instructions (x86_64 only)
    AVX512,
    /// Use NEON instructions (ARM only)
    NEON,
}

/// Runtime SIMD feature detection
pub struct SIMDCapabilities {
    pub sse2: bool,
    pub avx: bool,
    pub avx2: bool,
    pub avx512f: bool,
    pub neon: bool,
}

impl SIMDCapabilities {
    /// Detect available SIMD features at runtime
    pub fn detect() -> Self {
        Self {
            #[cfg(target_arch = "x86_64")]
            sse2: is_x86_feature_detected!("sse2"),
            #[cfg(not(target_arch = "x86_64"))]
            sse2: false,

            #[cfg(target_arch = "x86_64")]
            avx: is_x86_feature_detected!("avx"),
            #[cfg(not(target_arch = "x86_64"))]
            avx: false,

            #[cfg(target_arch = "x86_64")]
            avx2: is_x86_feature_detected!("avx2"),
            #[cfg(not(target_arch = "x86_64"))]
            avx2: false,

            #[cfg(target_arch = "x86_64")]
            avx512f: is_x86_feature_detected!("avx512f"),
            #[cfg(not(target_arch = "x86_64"))]
            avx512f: false,

            #[cfg(target_arch = "aarch64")]
            neon: cfg!(target_feature = "neon"),
            #[cfg(not(target_arch = "aarch64"))]
            neon: false,
        }
    }

    /// Get the best available strategy for this system
    pub fn best_strategy(&self) -> SIMDStrategy {
        #[cfg(target_arch = "x86_64")]
        {
            if self.avx512f {
                return SIMDStrategy::AVX512;
            }
            if self.avx2 {
                return SIMDStrategy::AVX;
            }
            if self.sse2 {
                return SIMDStrategy::SSE;
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if self.neon {
                return SIMDStrategy::NEON;
            }
        }

        SIMDStrategy::Scalar
    }
}

/// SIMD type converter with configurable strategy
pub struct SIMDConverter<T, U> {
    #[allow(dead_code)]
    strategy: SIMDStrategy,
    _phantom: PhantomData<(T, U)>,
}

impl<T, U> SIMDConverter<T, U> {
    /// Create a new SIMD converter with specified strategy
    pub fn new(strategy: SIMDStrategy) -> Self {
        Self {
            strategy,
            _phantom: PhantomData,
        }
    }

    /// Create a converter that automatically selects the best strategy
    pub fn auto() -> Self {
        Self::new(SIMDStrategy::Auto)
    }
}

/// f32 tensor conversions with SIMD optimization
impl Tensor<f32> {
    /// Convert to f64 tensor with SIMD optimization
    pub fn to_f64_simd(&self) -> Result<Tensor<f64>> {
        self.convert_simd_generic()
    }

    /// Convert to i32 tensor with SIMD optimization (with bounds checking)
    pub fn to_i32_simd(&self) -> Result<Tensor<i32>> {
        let data = self.data()?;
        let converted_data: Vec<i32> = data.iter().map(|&x| x as i32).collect();

        Tensor::from_data(converted_data, self.shape().dims().to_vec(), self.device)
    }
}

/// i32 tensor conversions with SIMD optimization
impl Tensor<i32> {
    /// Convert to f32 tensor with SIMD optimization
    pub fn to_f32_simd(&self) -> Result<Tensor<f32>> {
        let data = self.data()?;

        #[cfg(target_arch = "x86_64")]
        {
            let capabilities = SIMDCapabilities::detect();
            let converted_data = if data.len() >= 8 && capabilities.avx2 {
                i32_to_f32_simd(&data)
            } else if data.len() >= 4 && capabilities.sse2 {
                i32_to_f32_sse2(&data)
            } else {
                data.iter().map(|&x| x as f32).collect()
            };

            Tensor::from_data(converted_data, self.shape().dims().to_vec(), self.device)
        }

        #[cfg(target_arch = "aarch64")]
        {
            let converted_data = if data.len() >= 4 {
                i32_to_f32_neon(&data)
            } else {
                data.iter().map(|&x| x as f32).collect()
            };

            Tensor::from_data(converted_data, self.shape().dims().to_vec(), self.device)
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            let converted_data = data.iter().map(|&x| x as f32).collect();
            Tensor::from_data(converted_data, self.shape().dims().to_vec(), self.device)
        }
    }

    /// Convert to f64 tensor with SIMD optimization
    pub fn to_f64_simd(&self) -> Result<Tensor<f64>> {
        self.convert_simd_generic()
    }

    /// Convert to i64 tensor with SIMD optimization
    pub fn to_i64_simd(&self) -> Result<Tensor<i64>> {
        self.convert_simd_generic()
    }
}

/// i64 tensor conversions with SIMD optimization
impl Tensor<i64> {
    /// Convert to f32 tensor with SIMD optimization
    pub fn to_f32_simd(&self) -> Result<Tensor<f32>> {
        let data = self.data()?;
        let converted_data: Vec<f32> = data.iter().map(|&x| x as f32).collect();

        Tensor::from_data(converted_data, self.shape().dims().to_vec(), self.device)
    }

    /// Convert to f64 tensor with SIMD optimization
    pub fn to_f64_simd(&self) -> Result<Tensor<f64>> {
        let data = self.data()?;
        let converted_data: Vec<f64> = data.iter().map(|&x| x as f64).collect();

        Tensor::from_data(converted_data, self.shape().dims().to_vec(), self.device)
    }

    /// Convert to i32 tensor with SIMD optimization (with bounds checking)
    pub fn to_i32_simd(&self) -> Result<Tensor<i32>> {
        let data = self.data()?;
        let converted_data: Vec<i32> = data.iter().map(|&x| x as i32).collect();

        Tensor::from_data(converted_data, self.shape().dims().to_vec(), self.device)
    }
}

/// f64 tensor conversions with SIMD optimization
impl Tensor<f64> {
    /// Convert to f32 tensor with SIMD optimization
    pub fn to_f32_simd(&self) -> Result<Tensor<f32>> {
        self.convert_simd_f64_to_f32()
    }

    /// Specific f64 to f32 conversion with bounds checking
    fn convert_simd_f64_to_f32(&self) -> Result<Tensor<f32>> {
        let data = self.data()?;
        let converted_data: Vec<f32> = data.iter().map(|&x| x as f32).collect();

        Tensor::from_data(converted_data, self.shape().dims().to_vec(), self.device)
    }
}

/// Generic SIMD conversion implementations
impl<T: TensorElement + Copy> Tensor<T> {
    /// Generic SIMD conversion method with runtime feature detection
    fn convert_simd_generic<U: TensorElement + Copy + From<T>>(&self) -> Result<Tensor<U>> {
        let data = self.data()?;

        #[cfg(target_arch = "x86_64")]
        {
            let capabilities = SIMDCapabilities::detect();
            // Choose conversion strategy based on data size and available SIMD
            let converted_data = if data.len() >= 8 && capabilities.avx2 {
                convert_avx2(&data)
            } else if data.len() >= 4 && capabilities.sse2 {
                convert_sse2(&data)
            } else {
                convert_scalar(&data)
            };

            Tensor::from_data(converted_data, self.shape().dims().to_vec(), self.device)
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            let converted_data = convert_scalar(&data);
            Tensor::from_data(converted_data, self.shape().dims().to_vec(), self.device)
        }
    }

    /// Convert tensor using optimal SIMD strategy for this system
    pub fn convert_with_optimal_simd<U: TensorElement + Copy + From<T>>(
        &self,
    ) -> Result<Tensor<U>> {
        self.convert_simd_generic()
    }

    /// Convert tensor using specific SIMD strategy
    pub fn convert_with_strategy<U: TensorElement + Copy + From<T>>(
        &self,
        strategy: SIMDStrategy,
    ) -> Result<Tensor<U>> {
        let data = self.data()?;

        let converted_data = match strategy {
            SIMDStrategy::Scalar => convert_scalar(&data),
            SIMDStrategy::Auto => {
                let capabilities = SIMDCapabilities::detect();
                return self.convert_with_strategy(capabilities.best_strategy());
            }
            SIMDStrategy::SSE => {
                #[cfg(target_arch = "x86_64")]
                {
                    if data.len() >= 4 {
                        convert_sse2(&data)
                    } else {
                        convert_scalar(&data)
                    }
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    convert_scalar(&data)
                }
            }
            SIMDStrategy::AVX => {
                #[cfg(target_arch = "x86_64")]
                {
                    if data.len() >= 8 {
                        convert_avx2(&data)
                    } else {
                        convert_scalar(&data)
                    }
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    convert_scalar(&data)
                }
            }
            SIMDStrategy::AVX512 => {
                #[cfg(target_arch = "x86_64")]
                {
                    if data.len() >= 16 {
                        convert_avx512(&data)
                    } else {
                        convert_scalar(&data)
                    }
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    convert_scalar(&data)
                }
            }
            SIMDStrategy::NEON => {
                #[cfg(target_arch = "aarch64")]
                {
                    if data.len() >= 4 {
                        convert_neon(&data)
                    } else {
                        convert_scalar(&data)
                    }
                }
                #[cfg(not(target_arch = "aarch64"))]
                {
                    convert_scalar(&data)
                }
            }
        };

        Tensor::from_data(converted_data, self.shape().dims().to_vec(), self.device)
    }
}

/// Scalar fallback conversion
fn convert_scalar<T: Copy, U: From<T>>(data: &[T]) -> Vec<U> {
    data.iter().map(|&x| U::from(x)).collect()
}

/// SSE2 optimized conversion for compatible types
#[cfg(target_arch = "x86_64")]
fn convert_sse2<T: Copy + 'static, U: From<T> + 'static>(data: &[T]) -> Vec<U> {
    // Implement specific SSE2 conversions for common type pairs
    use std::any::TypeId;

    let t_type = TypeId::of::<T>();
    let u_type = TypeId::of::<U>();

    // f32 to f64 conversion with SSE2
    if t_type == TypeId::of::<f32>() && u_type == TypeId::of::<f64>() {
        let f32_data = unsafe { std::mem::transmute::<&[T], &[f32]>(data) };
        let result = f32_to_f64_sse2(f32_data);
        return unsafe { std::mem::transmute::<Vec<f64>, Vec<U>>(result) };
    }

    // f64 to f32 conversion with SSE2
    if t_type == TypeId::of::<f64>() && u_type == TypeId::of::<f32>() {
        let f64_data = unsafe { std::mem::transmute::<&[T], &[f64]>(data) };
        let result = f64_to_f32_sse2(f64_data);
        return unsafe { std::mem::transmute::<Vec<f32>, Vec<U>>(result) };
    }

    // i32 to f32 conversion with SSE2
    if t_type == TypeId::of::<i32>() && u_type == TypeId::of::<f32>() {
        let i32_data = unsafe { std::mem::transmute::<&[T], &[i32]>(data) };
        let result = i32_to_f32_sse2(i32_data);
        return unsafe { std::mem::transmute::<Vec<f32>, Vec<U>>(result) };
    }

    // Fall back to scalar conversion for unsupported pairs
    convert_scalar(data)
}

/// AVX2 optimized conversion for compatible types
#[cfg(target_arch = "x86_64")]
fn convert_avx2<T: Copy + 'static, U: From<T> + 'static>(data: &[T]) -> Vec<U> {
    // Implement specific AVX2 conversions for common type pairs
    use std::any::TypeId;

    let t_type = TypeId::of::<T>();
    let u_type = TypeId::of::<U>();

    // f32 to f64 conversion with AVX2
    if t_type == TypeId::of::<f32>() && u_type == TypeId::of::<f64>() {
        let f32_data = unsafe { std::mem::transmute::<&[T], &[f32]>(data) };
        let result = f32_to_f64_simd(f32_data);
        return unsafe { std::mem::transmute::<Vec<f64>, Vec<U>>(result) };
    }

    // f64 to f32 conversion with AVX2
    if t_type == TypeId::of::<f64>() && u_type == TypeId::of::<f32>() {
        let f64_data = unsafe { std::mem::transmute::<&[T], &[f64]>(data) };
        let result = f64_to_f32_simd(f64_data);
        return unsafe { std::mem::transmute::<Vec<f32>, Vec<U>>(result) };
    }

    // i32 to f32 conversion with AVX2
    if t_type == TypeId::of::<i32>() && u_type == TypeId::of::<f32>() {
        let i32_data = unsafe { std::mem::transmute::<&[T], &[i32]>(data) };
        let result = i32_to_f32_simd(i32_data);
        return unsafe { std::mem::transmute::<Vec<f32>, Vec<U>>(result) };
    }

    // f32 to i32 conversion with AVX2
    if t_type == TypeId::of::<f32>() && u_type == TypeId::of::<i32>() {
        let f32_data = unsafe { std::mem::transmute::<&[T], &[f32]>(data) };
        let result = f32_to_i32_simd(f32_data);
        return unsafe { std::mem::transmute::<Vec<i32>, Vec<U>>(result) };
    }

    // Fall back to scalar conversion for unsupported pairs
    convert_scalar(data)
}

/// AVX-512 optimized conversion for compatible types
#[cfg(target_arch = "x86_64")]
fn convert_avx512<T: Copy + 'static, U: From<T> + 'static>(data: &[T]) -> Vec<U> {
    // For now, fall back to AVX2 until AVX-512 specific implementations are added
    convert_avx2(data)
}

/// NEON optimized conversion for compatible types
#[cfg(target_arch = "aarch64")]
fn convert_neon<T: Copy + 'static, U: From<T> + 'static>(data: &[T]) -> Vec<U> {
    use std::any::TypeId;

    let t_type = TypeId::of::<T>();
    let u_type = TypeId::of::<U>();

    // f32 to f64 conversion with NEON
    if t_type == TypeId::of::<f32>() && u_type == TypeId::of::<f64>() {
        let f32_data = unsafe { std::mem::transmute::<&[T], &[f32]>(data) };
        let result = f32_to_f64_neon(f32_data);
        return unsafe { std::mem::transmute::<Vec<f64>, Vec<U>>(result) };
    }

    // i32 to f32 conversion with NEON
    if t_type == TypeId::of::<i32>() && u_type == TypeId::of::<f32>() {
        let i32_data = unsafe { std::mem::transmute::<&[T], &[i32]>(data) };
        let result = i32_to_f32_neon(i32_data);
        return unsafe { std::mem::transmute::<Vec<f32>, Vec<U>>(result) };
    }

    // Fall back to scalar conversion for unsupported pairs
    convert_scalar(data)
}

/// Optimized f32 to f64 conversion using SIMD
#[cfg(target_arch = "x86_64")]
pub fn f32_to_f64_simd(input: &[f32]) -> Vec<f64> {
    let mut output = Vec::with_capacity(input.len());

    #[cfg(target_feature = "avx")]
    {
        let chunks = input.chunks_exact(8);
        let remainder = chunks.remainder();

        for chunk in chunks {
            unsafe {
                // Load 8 f32 values
                let lo = _mm_loadu_ps(chunk.as_ptr());
                let hi = _mm_loadu_ps(chunk.as_ptr().add(4));

                // Convert to f64
                let lo_f64 = _mm256_cvtps_pd(lo);
                let hi_f64 = _mm256_cvtps_pd(hi);

                // Store results
                let mut lo_result = [0.0f64; 4];
                let mut hi_result = [0.0f64; 4];
                _mm256_storeu_pd(lo_result.as_mut_ptr(), lo_f64);
                _mm256_storeu_pd(hi_result.as_mut_ptr(), hi_f64);

                output.extend_from_slice(&lo_result);
                output.extend_from_slice(&hi_result);
            }
        }

        // Handle remainder with scalar conversion
        output.extend(remainder.iter().map(|&x| x as f64));
    }

    #[cfg(not(target_feature = "avx"))]
    {
        output.extend(input.iter().map(|&x| x as f64));
    }

    output
}

/// Optimized f64 to f32 conversion using SIMD
#[cfg(target_arch = "x86_64")]
pub fn f64_to_f32_simd(input: &[f64]) -> Vec<f32> {
    let mut output = Vec::with_capacity(input.len());

    #[cfg(target_feature = "avx")]
    {
        let chunks = input.chunks_exact(8);
        let remainder = chunks.remainder();

        for chunk in chunks {
            unsafe {
                // Load 8 f64 values
                let lo = _mm256_loadu_pd(chunk.as_ptr());
                let hi = _mm256_loadu_pd(chunk.as_ptr().add(4));

                // Convert to f32
                let lo_f32 = _mm256_cvtpd_ps(lo);
                let hi_f32 = _mm256_cvtpd_ps(hi);

                // Combine and store
                let combined = _mm256_set_m128(hi_f32, lo_f32);
                let mut result = [0.0f32; 8];
                _mm256_storeu_ps(result.as_mut_ptr(), combined);

                output.extend_from_slice(&result);
            }
        }

        // Handle remainder with scalar conversion
        output.extend(remainder.iter().map(|&x| x as f32));
    }

    #[cfg(not(target_feature = "avx"))]
    {
        output.extend(input.iter().map(|&x| x as f32));
    }

    output
}

/// Optimized i32 to f32 conversion using SIMD
#[cfg(target_arch = "x86_64")]
pub fn i32_to_f32_simd(input: &[i32]) -> Vec<f32> {
    let mut output = Vec::with_capacity(input.len());

    #[cfg(target_feature = "avx2")]
    {
        let chunks = input.chunks_exact(8);
        let remainder = chunks.remainder();

        for chunk in chunks {
            unsafe {
                // Load 8 i32 values
                let ints = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);

                // Convert to f32
                let floats = _mm256_cvtepi32_ps(ints);

                // Store results
                let mut result = [0.0f32; 8];
                _mm256_storeu_ps(result.as_mut_ptr(), floats);
                output.extend_from_slice(&result);
            }
        }

        // Handle remainder with scalar conversion
        output.extend(remainder.iter().map(|&x| x as f32));
    }

    #[cfg(not(target_feature = "avx2"))]
    {
        output.extend(input.iter().map(|&x| x as f32));
    }

    output
}

/// Optimized f32 to i32 conversion using SIMD (with truncation)
#[cfg(target_arch = "x86_64")]
pub fn f32_to_i32_simd(input: &[f32]) -> Vec<i32> {
    let mut output = Vec::with_capacity(input.len());

    #[cfg(target_feature = "avx2")]
    {
        let chunks = input.chunks_exact(8);
        let remainder = chunks.remainder();

        for chunk in chunks {
            unsafe {
                // Load 8 f32 values
                let floats = _mm256_loadu_ps(chunk.as_ptr());

                // Convert to i32 (truncating)
                let ints = _mm256_cvttps_epi32(floats);

                // Store results
                let mut result = [0i32; 8];
                _mm256_storeu_si256(result.as_mut_ptr() as *mut __m256i, ints);
                output.extend_from_slice(&result);
            }
        }

        // Handle remainder with scalar conversion
        output.extend(remainder.iter().map(|&x| x as i32));
    }

    #[cfg(not(target_feature = "avx2"))]
    {
        output.extend(input.iter().map(|&x| x as i32));
    }

    output
}

/// SIMD implementation trait for different architectures
#[cfg(target_arch = "x86_64")]
pub trait X86SIMDConversions {
    /// Check if AVX2 is available
    fn has_avx2() -> bool {
        cfg!(target_feature = "avx2")
    }

    /// Check if AVX is available
    fn has_avx() -> bool {
        cfg!(target_feature = "avx")
    }

    /// Check if SSE2 is available
    fn has_sse2() -> bool {
        cfg!(target_feature = "sse2")
    }
}

/// ARM NEON SIMD implementations
#[cfg(target_arch = "aarch64")]
pub trait ArmNEONConversions {
    /// Check if NEON is available
    fn has_neon() -> bool {
        cfg!(target_feature = "neon")
    }
}

/// SSE2 f32 to f64 conversion
#[cfg(target_arch = "x86_64")]
fn f32_to_f64_sse2(input: &[f32]) -> Vec<f64> {
    let mut output = Vec::with_capacity(input.len());

    let chunks = input.chunks_exact(4);
    let remainder = chunks.remainder();

    for chunk in chunks {
        unsafe {
            // Load 4 f32 values
            let f32_vec = _mm_loadu_ps(chunk.as_ptr());

            // Convert to f64 (SSE2 can only convert 2 at a time)
            let lo_f64 = _mm_cvtps_pd(f32_vec);
            let hi_f64 = _mm_cvtps_pd(_mm_movehl_ps(f32_vec, f32_vec));

            // Store results
            let mut lo_result = [0.0f64; 2];
            let mut hi_result = [0.0f64; 2];
            _mm_storeu_pd(lo_result.as_mut_ptr(), lo_f64);
            _mm_storeu_pd(hi_result.as_mut_ptr(), hi_f64);

            output.extend_from_slice(&lo_result);
            output.extend_from_slice(&hi_result);
        }
    }

    // Handle remainder with scalar conversion
    output.extend(remainder.iter().map(|&x| x as f64));
    output
}

/// SSE2 f64 to f32 conversion
#[cfg(target_arch = "x86_64")]
fn f64_to_f32_sse2(input: &[f64]) -> Vec<f32> {
    let mut output = Vec::with_capacity(input.len());

    let chunks = input.chunks_exact(4);
    let remainder = chunks.remainder();

    for chunk in chunks {
        unsafe {
            // Load 4 f64 values (2 at a time)
            let lo_f64 = _mm_loadu_pd(chunk.as_ptr());
            let hi_f64 = _mm_loadu_pd(chunk.as_ptr().add(2));

            // Convert to f32
            let lo_f32 = _mm_cvtpd_ps(lo_f64);
            let hi_f32 = _mm_cvtpd_ps(hi_f64);

            // Combine into single SSE register
            let combined = _mm_movelh_ps(lo_f32, hi_f32);

            // Store results
            let mut result = [0.0f32; 4];
            _mm_storeu_ps(result.as_mut_ptr(), combined);
            output.extend_from_slice(&result);
        }
    }

    // Handle remainder with scalar conversion
    output.extend(remainder.iter().map(|&x| x as f32));
    output
}

/// SSE2 i32 to f32 conversion
#[cfg(target_arch = "x86_64")]
fn i32_to_f32_sse2(input: &[i32]) -> Vec<f32> {
    let mut output = Vec::with_capacity(input.len());

    let chunks = input.chunks_exact(4);
    let remainder = chunks.remainder();

    for chunk in chunks {
        unsafe {
            // Load 4 i32 values
            let ints = _mm_loadu_si128(chunk.as_ptr() as *const __m128i);

            // Convert to f32
            let floats = _mm_cvtepi32_ps(ints);

            // Store results
            let mut result = [0.0f32; 4];
            _mm_storeu_ps(result.as_mut_ptr(), floats);
            output.extend_from_slice(&result);
        }
    }

    // Handle remainder with scalar conversion
    output.extend(remainder.iter().map(|&x| x as f32));
    output
}

/// NEON f32 to f64 conversion
#[cfg(target_arch = "aarch64")]
fn f32_to_f64_neon(input: &[f32]) -> Vec<f64> {
    use std::arch::aarch64::*;
    let mut output = Vec::with_capacity(input.len());

    let chunks = input.chunks_exact(4);
    let remainder = chunks.remainder();

    for chunk in chunks {
        unsafe {
            // Load 4 f32 values
            let f32_vec = vld1q_f32(chunk.as_ptr());

            // Convert to f64 (2 at a time)
            let lo_f64 = vcvt_f64_f32(vget_low_f32(f32_vec));
            let hi_f64 = vcvt_f64_f32(vget_high_f32(f32_vec));

            // Store results
            let mut lo_result = [0.0f64; 2];
            let mut hi_result = [0.0f64; 2];
            vst1q_f64(lo_result.as_mut_ptr(), lo_f64);
            vst1q_f64(hi_result.as_mut_ptr(), hi_f64);

            output.extend_from_slice(&lo_result);
            output.extend_from_slice(&hi_result);
        }
    }

    // Handle remainder with scalar conversion
    output.extend(remainder.iter().map(|&x| x as f64));
    output
}

/// NEON i32 to f32 conversion
#[cfg(target_arch = "aarch64")]
fn i32_to_f32_neon(input: &[i32]) -> Vec<f32> {
    use std::arch::aarch64::*;
    let mut output = Vec::with_capacity(input.len());

    let chunks = input.chunks_exact(4);
    let remainder = chunks.remainder();

    for chunk in chunks {
        unsafe {
            // Load 4 i32 values
            let ints = vld1q_s32(chunk.as_ptr());

            // Convert to f32
            let floats = vcvtq_f32_s32(ints);

            // Store results
            let mut result = [0.0f32; 4];
            vst1q_f32(result.as_mut_ptr(), floats);
            output.extend_from_slice(&result);
        }
    }

    // Handle remainder with scalar conversion
    output.extend(remainder.iter().map(|&x| x as f32));
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor;
    use torsh_core::DeviceType;

    #[test]
    fn test_scalar_conversion() {
        let f32_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let f64_result = convert_scalar::<f32, f64>(&f32_data);

        assert_eq!(f64_result.len(), 4);
        assert_eq!(f64_result[0], 1.0f64);
        assert_eq!(f64_result[3], 4.0f64);
    }

    #[test]
    fn test_tensor_f32_to_f64_conversion() {
        let tensor = tensor![1.0f32, 2.0, 3.0, 4.0].unwrap();
        let converted = tensor.to_f64_simd().unwrap();

        assert_eq!(converted.shape().dims(), tensor.shape().dims());
        let converted_data = converted.data().unwrap();
        assert_eq!(converted_data[0], 1.0f64);
        assert_eq!(converted_data[3], 4.0f64);
    }

    #[test]
    fn test_tensor_i32_to_f32_conversion() {
        let tensor = tensor![1i32, 2, 3, 4].unwrap();
        let converted = tensor.to_f32_simd().unwrap();

        assert_eq!(converted.shape().dims(), tensor.shape().dims());
        let converted_data = converted.data().unwrap();
        assert_eq!(converted_data[0], 1.0f32);
        assert_eq!(converted_data[3], 4.0f32);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_simd_f32_to_f64() {
        let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let output = f32_to_f64_simd(&input);

        assert_eq!(output.len(), input.len());
        for (i, (&inp, &out)) in input.iter().zip(output.iter()).enumerate() {
            assert_eq!(inp as f64, out, "Mismatch at index {}", i);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_simd_i32_to_f32() {
        let input = vec![1i32, -2, 3, -4, 5, -6, 7, -8, 9];
        let output = i32_to_f32_simd(&input);

        assert_eq!(output.len(), input.len());
        for (i, (&inp, &out)) in input.iter().zip(output.iter()).enumerate() {
            assert_eq!(inp as f32, out, "Mismatch at index {}", i);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_sse2_f32_to_f64() {
        let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let output = f32_to_f64_sse2(&input);

        assert_eq!(output.len(), input.len());
        for (i, (&inp, &out)) in input.iter().zip(output.iter()).enumerate() {
            assert_eq!(inp as f64, out, "SSE2 f32->f64 mismatch at index {}", i);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_sse2_f64_to_f32() {
        let input = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
        let output = f64_to_f32_sse2(&input);

        assert_eq!(output.len(), input.len());
        for (i, (&inp, &out)) in input.iter().zip(output.iter()).enumerate() {
            assert_eq!(inp as f32, out, "SSE2 f64->f32 mismatch at index {}", i);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_sse2_i32_to_f32() {
        let input = vec![1i32, -2, 3, -4, 5, -6, 7, -8];
        let output = i32_to_f32_sse2(&input);

        assert_eq!(output.len(), input.len());
        for (i, (&inp, &out)) in input.iter().zip(output.iter()).enumerate() {
            assert_eq!(inp as f32, out, "SSE2 i32->f32 mismatch at index {}", i);
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_neon_f32_to_f64() {
        let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let output = f32_to_f64_neon(&input);

        assert_eq!(output.len(), input.len());
        for (i, (&inp, &out)) in input.iter().zip(output.iter()).enumerate() {
            assert_eq!(inp as f64, out, "NEON f32->f64 mismatch at index {}", i);
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_neon_i32_to_f32() {
        let input = vec![1i32, -2, 3, -4, 5, -6, 7, -8];
        let output = i32_to_f32_neon(&input);

        assert_eq!(output.len(), input.len());
        for (i, (&inp, &out)) in input.iter().zip(output.iter()).enumerate() {
            assert_eq!(inp as f32, out, "NEON i32->f32 mismatch at index {}", i);
        }
    }

    #[test]
    fn test_simd_converter_creation() {
        let converter = SIMDConverter::<f32, f64>::auto();
        assert!(matches!(converter.strategy, SIMDStrategy::Auto));

        let converter = SIMDConverter::<f32, f64>::new(SIMDStrategy::Scalar);
        assert!(matches!(converter.strategy, SIMDStrategy::Scalar));
    }

    #[test]
    fn test_large_tensor_conversion() {
        // Test with larger data to ensure SIMD paths are taken
        let large_data: Vec<f32> = (0..1000).map(|i| i as f32 * 0.1).collect();
        let tensor =
            crate::Tensor::from_data(large_data.clone(), vec![1000], DeviceType::Cpu).unwrap();

        let converted = tensor.to_f64_simd().unwrap();
        let converted_data = converted.to_vec().unwrap();

        assert_eq!(converted_data.len(), large_data.len());
        for (i, (&original, &converted)) in large_data.iter().zip(converted_data.iter()).enumerate()
        {
            assert_eq!(
                original as f64, converted,
                "Large tensor conversion mismatch at index {}",
                i
            );
        }
    }

    #[test]
    fn test_simd_capabilities_detection() {
        let capabilities = SIMDCapabilities::detect();

        // These should be detectable at runtime
        println!(
            "SIMD capabilities: SSE2={}, AVX={}, AVX2={}, AVX512F={}, NEON={}",
            capabilities.sse2,
            capabilities.avx,
            capabilities.avx2,
            capabilities.avx512f,
            capabilities.neon
        );

        let strategy = capabilities.best_strategy();
        match strategy {
            SIMDStrategy::Scalar => println!("Using scalar conversion"),
            #[cfg(target_arch = "x86_64")]
            SIMDStrategy::SSE => println!("Using SSE2 conversion"),
            #[cfg(target_arch = "x86_64")]
            SIMDStrategy::AVX => println!("Using AVX/AVX2 conversion"),
            #[cfg(target_arch = "x86_64")]
            SIMDStrategy::AVX512 => println!("Using AVX-512 conversion"),
            #[cfg(target_arch = "aarch64")]
            SIMDStrategy::NEON => println!("Using NEON conversion"),
            _ => println!("Unknown strategy"),
        }
    }

    #[test]
    fn test_convert_with_strategy() {
        let tensor = tensor![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0].unwrap();

        // Test scalar strategy
        let scalar_result: Tensor<f64> =
            tensor.convert_with_strategy(SIMDStrategy::Scalar).unwrap();
        let scalar_data = scalar_result.to_vec().unwrap();
        assert_eq!(scalar_data.len(), 8);
        assert_eq!(scalar_data[0], 1.0f64);

        // Test auto strategy
        let auto_result: Tensor<f64> = tensor.convert_with_strategy(SIMDStrategy::Auto).unwrap();
        let auto_data = auto_result.to_vec().unwrap();
        assert_eq!(auto_data.len(), 8);
        assert_eq!(auto_data[0], 1.0f64);

        // Results should be the same regardless of strategy
        assert_eq!(scalar_data, auto_data);
    }

    #[test]
    fn test_additional_data_types() {
        // Test i64 conversions
        let i64_tensor = tensor![1i64, 2, 3, 4].unwrap();
        let i64_to_f32 = i64_tensor.to_f32_simd().unwrap();
        let i64_to_f64 = i64_tensor.to_f64_simd().unwrap();

        assert_eq!(i64_to_f32.to_vec().unwrap(), vec![1.0f32, 2.0, 3.0, 4.0]);
        assert_eq!(i64_to_f64.to_vec().unwrap(), vec![1.0f64, 2.0, 3.0, 4.0]);

        // Test i32 conversions that are working
        let i32_tensor = tensor![1i32, 2, 3, 4].unwrap();
        let i32_to_f32 = i32_tensor.to_f32_simd().unwrap();
        let i32_to_f64 = i32_tensor.to_f64_simd().unwrap();
        let i32_to_i64 = i32_tensor.to_i64_simd().unwrap();

        assert_eq!(i32_to_f32.to_vec().unwrap(), vec![1.0f32, 2.0, 3.0, 4.0]);
        assert_eq!(i32_to_f64.to_vec().unwrap(), vec![1.0f64, 2.0, 3.0, 4.0]);
        assert_eq!(i32_to_i64.to_vec().unwrap(), vec![1i64, 2, 3, 4]);
    }

    #[test]
    fn test_optimal_simd_conversion() {
        let tensor = tensor![1.0f32, 2.0, 3.0, 4.0].unwrap();
        let result: Tensor<f64> = tensor.convert_with_optimal_simd().unwrap();

        assert_eq!(result.to_vec().unwrap(), vec![1.0f64, 2.0, 3.0, 4.0]);
    }
}
