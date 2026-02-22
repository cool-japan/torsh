//! WebAssembly SIMD optimizations for browser deployment
//!
//! This module provides SIMD implementations optimized for WebAssembly targets,
//! enabling efficient tensor operations when ToRSh is compiled to WASM and run
//! in web browsers or other WASM runtimes.
//!
//! Features:
//! - WASM SIMD 128-bit vector operations
//! - Browser compatibility detection
//! - Memory-efficient algorithms for limited WASM heap
//! - JavaScript interop optimizations
//! - Progressive Web App (PWA) support

use crate::cpu::error::CpuResult;
use torsh_core::error::TorshError;

#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::*;

/// WebAssembly SIMD operations interface
#[derive(Debug)]
pub struct WasmSimdOps {
    /// Whether WASM SIMD is available in the current runtime
    simd_available: bool,
    /// Vector width for WASM SIMD (typically 128-bit)
    vector_width: usize,
}

impl Default for WasmSimdOps {
    fn default() -> Self {
        Self::new()
    }
}

impl WasmSimdOps {
    /// Create a new WASM SIMD operations instance
    pub fn new() -> Self {
        let simd_available = Self::detect_wasm_simd();
        Self {
            simd_available,
            vector_width: if simd_available { 16 } else { 1 }, // 128-bit SIMD = 16 bytes
        }
    }

    /// Detect if WASM SIMD is available
    #[cfg(target_arch = "wasm32")]
    fn detect_wasm_simd() -> bool {
        // Use a runtime test to check if WASM SIMD is available
        std::panic::catch_unwind(|| {
            unsafe {
                // Try to create a simple v128 value to test SIMD availability
                let _test_vector = v128_const(1, 2, 3, 4);
                true
            }
        })
        .unwrap_or(false)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn detect_wasm_simd() -> bool {
        false
    }

    /// Check if WASM SIMD is available
    pub fn is_available(&self) -> bool {
        self.simd_available
    }

    /// Get the SIMD vector width in bytes
    pub fn vector_width(&self) -> usize {
        self.vector_width
    }

    /// Vectorized addition for f32 arrays using WASM SIMD
    #[cfg(target_arch = "wasm32")]
    pub fn add_f32(&self, a: &[f32], b: &[f32], result: &mut [f32]) -> CpuResult<()> {
        if !self.simd_available {
            return self.add_f32_scalar(a, b, result);
        }

        if a.len() != b.len() || a.len() != result.len() {
            return Err(TorshError::ComputeError(
                "Array length mismatch".to_string(),
            ));
        }

        unsafe {
            let chunks = a.len() / 4; // 4 f32 values per 128-bit vector
            let remainder = a.len() % 4;

            // Process 4 elements at a time using WASM SIMD
            for i in 0..chunks {
                let idx = i * 4;

                // Load 4 f32 values into v128 vectors
                let va = v128_load(a.as_ptr().add(idx) as *const v128);
                let vb = v128_load(b.as_ptr().add(idx) as *const v128);

                // Perform SIMD addition
                let vresult = f32x4_add(va, vb);

                // Store result
                v128_store(result.as_mut_ptr().add(idx) as *mut v128, vresult);
            }

            // Handle remaining elements with scalar operations
            for i in (chunks * 4)..(chunks * 4 + remainder) {
                result[i] = a[i] + b[i];
            }
        }

        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn add_f32(&self, a: &[f32], b: &[f32], result: &mut [f32]) -> CpuResult<()> {
        self.add_f32_scalar(a, b, result)
    }

    /// Scalar fallback for f32 addition
    fn add_f32_scalar(&self, a: &[f32], b: &[f32], result: &mut [f32]) -> CpuResult<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return Err(TorshError::ComputeError(
                "Array length mismatch".to_string(),
            ));
        }

        for i in 0..a.len() {
            result[i] = a[i] + b[i];
        }

        Ok(())
    }

    /// Vectorized multiplication for f32 arrays using WASM SIMD
    #[cfg(target_arch = "wasm32")]
    pub fn mul_f32(&self, a: &[f32], b: &[f32], result: &mut [f32]) -> CpuResult<()> {
        if !self.simd_available {
            return self.mul_f32_scalar(a, b, result);
        }

        if a.len() != b.len() || a.len() != result.len() {
            return Err(TorshError::ComputeError(
                "Array length mismatch".to_string(),
            ));
        }

        unsafe {
            let chunks = a.len() / 4;
            let remainder = a.len() % 4;

            // Process 4 elements at a time using WASM SIMD
            for i in 0..chunks {
                let idx = i * 4;

                // Load 4 f32 values into v128 vectors
                let va = v128_load(a.as_ptr().add(idx) as *const v128);
                let vb = v128_load(b.as_ptr().add(idx) as *const v128);

                // Perform SIMD multiplication
                let vresult = f32x4_mul(va, vb);

                // Store result
                v128_store(result.as_mut_ptr().add(idx) as *mut v128, vresult);
            }

            // Handle remaining elements with scalar operations
            for i in (chunks * 4)..(chunks * 4 + remainder) {
                result[i] = a[i] * b[i];
            }
        }

        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn mul_f32(&self, a: &[f32], b: &[f32], result: &mut [f32]) -> CpuResult<()> {
        self.mul_f32_scalar(a, b, result)
    }

    /// Scalar fallback for f32 multiplication
    fn mul_f32_scalar(&self, a: &[f32], b: &[f32], result: &mut [f32]) -> CpuResult<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return Err(TorshError::ComputeError(
                "Array length mismatch".to_string(),
            ));
        }

        for i in 0..a.len() {
            result[i] = a[i] * b[i];
        }

        Ok(())
    }

    /// Dot product using WASM SIMD
    pub fn dot_product_f32(&self, a: &[f32], b: &[f32]) -> CpuResult<f32> {
        if a.len() != b.len() {
            return Err(TorshError::ComputeError(
                "Array length mismatch".to_string(),
            ));
        }

        if a.is_empty() {
            return Ok(0.0);
        }

        let mut result = 0.0f32;

        #[cfg(target_arch = "wasm32")]
        if self.simd_available {
            unsafe {
                let chunks = a.len() / 4;
                let remainder = a.len() % 4;

                // Initialize accumulator vector with zeros
                let mut sum_vec = f32x4_splat(0.0);

                for i in 0..chunks {
                    let idx = i * 4;

                    // Load vectors
                    let va = v128_load(a.as_ptr().add(idx) as *const v128);
                    let vb = v128_load(b.as_ptr().add(idx) as *const v128);

                    // Multiply and accumulate
                    let vmul = f32x4_mul(va, vb);
                    sum_vec = f32x4_add(sum_vec, vmul);
                }

                // Horizontal sum of the accumulator vector
                let sum_array = [
                    f32x4_extract_lane::<0>(sum_vec),
                    f32x4_extract_lane::<1>(sum_vec),
                    f32x4_extract_lane::<2>(sum_vec),
                    f32x4_extract_lane::<3>(sum_vec),
                ];
                result = sum_array.iter().sum();

                // Handle remaining elements
                for i in (chunks * 4)..(chunks * 4 + remainder) {
                    result += a[i] * b[i];
                }
            }
        } else {
            // Scalar fallback
            for i in 0..a.len() {
                result += a[i] * b[i];
            }
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            // Scalar fallback for non-WASM targets
            for i in 0..a.len() {
                result += a[i] * b[i];
            }
        }

        Ok(result)
    }

    /// Matrix multiplication optimized for WASM
    pub fn matmul_f32(
        &self,
        a: &[f32],
        b: &[f32],
        result: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> CpuResult<()> {
        if a.len() != m * k || b.len() != k * n || result.len() != m * n {
            return Err(TorshError::ComputeError(
                "Invalid matrix dimensions".to_string(),
            ));
        }

        // Use cache-friendly algorithm suitable for WASM
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    sum += a[i * k + l] * b[l * n + j];
                }
                result[i * n + j] = sum;
            }
        }

        Ok(())
    }

    /// ReLU activation function using WASM SIMD
    #[cfg(target_arch = "wasm32")]
    pub fn relu_f32(&self, input: &[f32], output: &mut [f32]) -> CpuResult<()> {
        if input.len() != output.len() {
            return Err(TorshError::ComputeError(
                "Array length mismatch".to_string(),
            ));
        }

        if !self.simd_available {
            return self.relu_f32_scalar(input, output);
        }

        unsafe {
            let chunks = input.len() / 4;
            let remainder = input.len() % 4;

            // Create a zero vector for comparison
            let zero_vec = f32x4_splat(0.0);

            // Process 4 elements at a time using proper SIMD
            for i in 0..chunks {
                let idx = i * 4;

                // Load input vector
                let input_vec = v128_load(input.as_ptr().add(idx) as *const v128);

                // Use SIMD max to compute ReLU: max(input, 0)
                let result_vec = f32x4_pmax(input_vec, zero_vec);

                // Store result
                v128_store(output.as_mut_ptr().add(idx) as *mut v128, result_vec);
            }

            // Handle remaining elements
            for i in (chunks * 4)..(chunks * 4 + remainder) {
                output[i] = input[i].max(0.0);
            }
        }

        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn relu_f32(&self, input: &[f32], output: &mut [f32]) -> CpuResult<()> {
        self.relu_f32_scalar(input, output)
    }

    /// Scalar ReLU implementation
    fn relu_f32_scalar(&self, input: &[f32], output: &mut [f32]) -> CpuResult<()> {
        if input.len() != output.len() {
            return Err(TorshError::ComputeError(
                "Array length mismatch".to_string(),
            ));
        }

        for i in 0..input.len() {
            output[i] = input[i].max(0.0);
        }

        Ok(())
    }

    /// Softmax function optimized for WASM
    pub fn softmax_f32(&self, input: &[f32], output: &mut [f32]) -> CpuResult<()> {
        if input.len() != output.len() {
            return Err(TorshError::ComputeError(
                "Array length mismatch".to_string(),
            ));
        }

        if input.is_empty() {
            return Ok(());
        }

        // Find maximum for numerical stability
        let max_val = input.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // Compute exponentials and sum
        let mut sum = 0.0f32;
        for i in 0..input.len() {
            output[i] = (input[i] - max_val).exp();
            sum += output[i];
        }

        // Normalize
        for i in 0..output.len() {
            output[i] /= sum;
        }

        Ok(())
    }

    /// Memory bandwidth optimization for WASM
    pub fn optimize_memory_access<T>(&self, data: &mut [T]) -> CpuResult<()> {
        // In WASM, memory access patterns are important for performance
        // This function could implement memory prefetching hints or
        // data reorganization for better cache locality

        // For now, this is a placeholder that could be expanded with
        // WASM-specific memory optimization techniques
        let _ = data; // Acknowledge parameter
        Ok(())
    }

    /// Fused multiply-add operation using WASM SIMD
    #[cfg(target_arch = "wasm32")]
    pub fn fma_f32(&self, a: &[f32], b: &[f32], c: &[f32], result: &mut [f32]) -> CpuResult<()> {
        if a.len() != b.len() || a.len() != c.len() || a.len() != result.len() {
            return Err(TorshError::ComputeError(
                "Array length mismatch".to_string(),
            ));
        }

        if !self.simd_available {
            return self.fma_f32_scalar(a, b, c, result);
        }

        unsafe {
            let chunks = a.len() / 4;
            let remainder = a.len() % 4;

            for i in 0..chunks {
                let idx = i * 4;

                // Load vectors
                let va = v128_load(a.as_ptr().add(idx) as *const v128);
                let vb = v128_load(b.as_ptr().add(idx) as *const v128);
                let vc = v128_load(c.as_ptr().add(idx) as *const v128);

                // Perform a * b + c using separate mul and add (no native FMA in WASM SIMD)
                let vmul = f32x4_mul(va, vb);
                let vresult = f32x4_add(vmul, vc);

                // Store result
                v128_store(result.as_mut_ptr().add(idx) as *mut v128, vresult);
            }

            // Handle remaining elements
            for i in (chunks * 4)..(chunks * 4 + remainder) {
                result[i] = a[i].mul_add(b[i], c[i]);
            }
        }

        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn fma_f32(&self, a: &[f32], b: &[f32], c: &[f32], result: &mut [f32]) -> CpuResult<()> {
        self.fma_f32_scalar(a, b, c, result)
    }

    /// Scalar FMA fallback
    fn fma_f32_scalar(&self, a: &[f32], b: &[f32], c: &[f32], result: &mut [f32]) -> CpuResult<()> {
        if a.len() != b.len() || a.len() != c.len() || a.len() != result.len() {
            return Err(TorshError::ComputeError(
                "Array length mismatch".to_string(),
            ));
        }

        for i in 0..a.len() {
            result[i] = a[i].mul_add(b[i], c[i]);
        }

        Ok(())
    }

    /// Check browser compatibility for advanced features
    #[cfg(target_arch = "wasm32")]
    pub fn check_browser_compatibility(&self) -> BrowserCompatibility {
        BrowserCompatibility {
            wasm_simd: self.simd_available,
            wasm_threads: Self::detect_wasm_threads(),
            shared_array_buffer: Self::detect_shared_array_buffer(),
            offscreen_canvas: Self::detect_offscreen_canvas(),
            web_workers: Self::detect_web_workers(),
            webgl2: Self::detect_webgl2(),
            estimated_memory_limit_mb: Self::estimate_memory_limit(),
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn check_browser_compatibility(&self) -> BrowserCompatibility {
        BrowserCompatibility::default()
    }

    #[cfg(target_arch = "wasm32")]
    fn detect_wasm_threads() -> bool {
        // In a real implementation, this would check for WASM threads support
        // For now, return false as threads are still experimental
        false
    }

    #[cfg(target_arch = "wasm32")]
    fn detect_shared_array_buffer() -> bool {
        // Check if SharedArrayBuffer is available (required for WASM threads)
        std::panic::catch_unwind(|| {
            // This would use JS bindings to check for SharedArrayBuffer
            false
        })
        .unwrap_or(false)
    }

    #[cfg(target_arch = "wasm32")]
    fn detect_offscreen_canvas() -> bool {
        // Check if OffscreenCanvas is supported for WebGL in workers
        true // Most modern browsers support this
    }

    #[cfg(target_arch = "wasm32")]
    fn detect_web_workers() -> bool {
        // Web Workers are widely supported
        true
    }

    #[cfg(target_arch = "wasm32")]
    fn detect_webgl2() -> bool {
        // WebGL2 support for GPU acceleration
        true // Assume modern browser
    }

    #[cfg(target_arch = "wasm32")]
    fn estimate_memory_limit() -> usize {
        // Estimate available memory for WASM heap
        // In browsers, this is typically limited to 2-4GB on 64-bit systems
        if cfg!(target_pointer_width = "64") {
            2048 // 2GB for 64-bit WASM
        } else {
            512 // 512MB for 32-bit WASM
        }
    }

    /// Optimize for browser-specific memory constraints
    pub fn optimize_for_browser(&mut self, memory_limit_mb: Option<usize>) -> CpuResult<()> {
        let limit = memory_limit_mb.unwrap_or(512); // Default 512MB limit

        // Adjust vector operations for memory efficiency
        if limit < 256 {
            // Very limited memory - use more conservative chunking
            self.vector_width = self.vector_width.min(8);
        } else if limit < 1024 {
            // Moderate memory - normal chunking
            self.vector_width = self.vector_width.min(16);
        }
        // else keep default settings for high-memory environments

        Ok(())
    }

    /// Create WASM-optimized matrix multiplication with blocking
    pub fn matmul_f32_blocked(
        &self,
        a: &[f32],
        b: &[f32],
        result: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
        block_size: usize,
    ) -> CpuResult<()> {
        if a.len() != m * k || b.len() != k * n || result.len() != m * n {
            return Err(TorshError::ComputeError(
                "Invalid matrix dimensions".to_string(),
            ));
        }

        // Initialize result to zero
        result.fill(0.0);

        // Blocked matrix multiplication for better cache performance in WASM
        for i0 in (0..m).step_by(block_size) {
            for j0 in (0..n).step_by(block_size) {
                for k0 in (0..k).step_by(block_size) {
                    let i_end = (i0 + block_size).min(m);
                    let j_end = (j0 + block_size).min(n);
                    let k_end = (k0 + block_size).min(k);

                    for i in i0..i_end {
                        for j in j0..j_end {
                            let mut sum = 0.0f32;

                            #[cfg(target_arch = "wasm32")]
                            if self.simd_available && (k_end - k0) >= 4 {
                                unsafe {
                                    let simd_end = k0 + ((k_end - k0) / 4) * 4;
                                    let mut sum_vec = f32x4_splat(0.0);

                                    for l in (k0..simd_end).step_by(4) {
                                        let va =
                                            v128_load(a.as_ptr().add(i * k + l) as *const v128);
                                        let vb_vals = [
                                            b[l * n + j],
                                            b[(l + 1) * n + j],
                                            b[(l + 2) * n + j],
                                            b[(l + 3) * n + j],
                                        ];
                                        let vb =
                                            f32x4(vb_vals[0], vb_vals[1], vb_vals[2], vb_vals[3]);
                                        let vmul = f32x4_mul(va, vb);
                                        sum_vec = f32x4_add(sum_vec, vmul);
                                    }

                                    // Extract and sum the SIMD result
                                    sum += f32x4_extract_lane::<0>(sum_vec)
                                        + f32x4_extract_lane::<1>(sum_vec)
                                        + f32x4_extract_lane::<2>(sum_vec)
                                        + f32x4_extract_lane::<3>(sum_vec);

                                    // Handle remaining elements
                                    for l in simd_end..k_end {
                                        sum += a[i * k + l] * b[l * n + j];
                                    }
                                }
                            } else {
                                // Scalar version
                                for l in k0..k_end {
                                    sum += a[i * k + l] * b[l * n + j];
                                }
                            }

                            #[cfg(not(target_arch = "wasm32"))]
                            {
                                // Scalar version for non-WASM targets
                                for l in k0..k_end {
                                    sum += a[i * k + l] * b[l * n + j];
                                }
                            }

                            result[i * n + j] += sum;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Sum reduction using WASM SIMD
    #[cfg(target_arch = "wasm32")]
    pub fn sum_f32(&self, input: &[f32]) -> CpuResult<f32> {
        if input.is_empty() {
            return Ok(0.0);
        }

        if !self.simd_available {
            return Ok(input.iter().sum());
        }

        unsafe {
            let chunks = input.len() / 4;
            let remainder = input.len() % 4;

            // Initialize accumulator
            let mut sum_vec = f32x4_splat(0.0);

            // Process 4 elements at a time
            for i in 0..chunks {
                let idx = i * 4;
                let input_vec = v128_load(input.as_ptr().add(idx) as *const v128);
                sum_vec = f32x4_add(sum_vec, input_vec);
            }

            // Extract individual elements and sum them
            let mut result = f32x4_extract_lane::<0>(sum_vec)
                + f32x4_extract_lane::<1>(sum_vec)
                + f32x4_extract_lane::<2>(sum_vec)
                + f32x4_extract_lane::<3>(sum_vec);

            // Handle remaining elements
            for i in (chunks * 4)..(chunks * 4 + remainder) {
                result += input[i];
            }

            Ok(result)
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn sum_f32(&self, input: &[f32]) -> CpuResult<f32> {
        Ok(input.iter().sum())
    }

    /// Integer addition using WASM SIMD
    #[cfg(target_arch = "wasm32")]
    pub fn add_i32(&self, a: &[i32], b: &[i32], result: &mut [i32]) -> CpuResult<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return Err(TorshError::ComputeError(
                "Array length mismatch".to_string(),
            ));
        }

        if !self.simd_available {
            return self.add_i32_scalar(a, b, result);
        }

        unsafe {
            let chunks = a.len() / 4;
            let remainder = a.len() % 4;

            // Process 4 elements at a time
            for i in 0..chunks {
                let idx = i * 4;

                let va = v128_load(a.as_ptr().add(idx) as *const v128);
                let vb = v128_load(b.as_ptr().add(idx) as *const v128);

                let vresult = i32x4_add(va, vb);

                v128_store(result.as_mut_ptr().add(idx) as *mut v128, vresult);
            }

            // Handle remaining elements
            for i in (chunks * 4)..(chunks * 4 + remainder) {
                result[i] = a[i] + b[i];
            }
        }

        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn add_i32(&self, a: &[i32], b: &[i32], result: &mut [i32]) -> CpuResult<()> {
        self.add_i32_scalar(a, b, result)
    }

    /// Scalar fallback for i32 addition
    fn add_i32_scalar(&self, a: &[i32], b: &[i32], result: &mut [i32]) -> CpuResult<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return Err(TorshError::ComputeError(
                "Array length mismatch".to_string(),
            ));
        }

        for i in 0..a.len() {
            result[i] = a[i] + b[i];
        }

        Ok(())
    }

    /// Integer multiplication using WASM SIMD
    #[cfg(target_arch = "wasm32")]
    pub fn mul_i32(&self, a: &[i32], b: &[i32], result: &mut [i32]) -> CpuResult<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return Err(TorshError::ComputeError(
                "Array length mismatch".to_string(),
            ));
        }

        if !self.simd_available {
            return self.mul_i32_scalar(a, b, result);
        }

        unsafe {
            let chunks = a.len() / 4;
            let remainder = a.len() % 4;

            for i in 0..chunks {
                let idx = i * 4;

                let va = v128_load(a.as_ptr().add(idx) as *const v128);
                let vb = v128_load(b.as_ptr().add(idx) as *const v128);

                let vresult = i32x4_mul(va, vb);

                v128_store(result.as_mut_ptr().add(idx) as *mut v128, vresult);
            }

            // Handle remaining elements
            for i in (chunks * 4)..(chunks * 4 + remainder) {
                result[i] = a[i] * b[i];
            }
        }

        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn mul_i32(&self, a: &[i32], b: &[i32], result: &mut [i32]) -> CpuResult<()> {
        self.mul_i32_scalar(a, b, result)
    }

    /// Scalar fallback for i32 multiplication
    fn mul_i32_scalar(&self, a: &[i32], b: &[i32], result: &mut [i32]) -> CpuResult<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return Err(TorshError::ComputeError(
                "Array length mismatch".to_string(),
            ));
        }

        for i in 0..a.len() {
            result[i] = a[i] * b[i];
        }

        Ok(())
    }

    /// Maximum reduction using WASM SIMD
    #[cfg(target_arch = "wasm32")]
    pub fn max_f32(&self, input: &[f32]) -> CpuResult<f32> {
        if input.is_empty() {
            return Err(TorshError::ComputeError(
                "Cannot find max of empty array".to_string(),
            ));
        }

        if !self.simd_available {
            return Ok(input.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)));
        }

        unsafe {
            let chunks = input.len() / 4;
            let remainder = input.len() % 4;

            // Initialize with first element repeated
            let mut max_vec = if chunks > 0 {
                v128_load(input.as_ptr() as *const v128)
            } else {
                f32x4_splat(input[0])
            };

            // Process 4 elements at a time
            for i in 1..chunks {
                let idx = i * 4;
                let input_vec = v128_load(input.as_ptr().add(idx) as *const v128);
                max_vec = f32x4_pmax(max_vec, input_vec);
            }

            // Find max of the 4 elements in max_vec
            let mut result = f32x4_extract_lane::<0>(max_vec)
                .max(f32x4_extract_lane::<1>(max_vec))
                .max(f32x4_extract_lane::<2>(max_vec))
                .max(f32x4_extract_lane::<3>(max_vec));

            // Handle remaining elements
            for i in (chunks * 4)..(chunks * 4 + remainder) {
                result = result.max(input[i]);
            }

            Ok(result)
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn max_f32(&self, input: &[f32]) -> CpuResult<f32> {
        if input.is_empty() {
            return Err(TorshError::ComputeError(
                "Cannot find max of empty array".to_string(),
            ));
        }
        Ok(input.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)))
    }

    /// Minimum reduction using WASM SIMD
    #[cfg(target_arch = "wasm32")]
    pub fn min_f32(&self, input: &[f32]) -> CpuResult<f32> {
        if input.is_empty() {
            return Err(TorshError::ComputeError(
                "Cannot find min of empty array".to_string(),
            ));
        }

        if !self.simd_available {
            return Ok(input.iter().fold(f32::INFINITY, |a, &b| a.min(b)));
        }

        unsafe {
            let chunks = input.len() / 4;
            let remainder = input.len() % 4;

            // Initialize with first element repeated
            let mut min_vec = if chunks > 0 {
                v128_load(input.as_ptr() as *const v128)
            } else {
                f32x4_splat(input[0])
            };

            // Process 4 elements at a time
            for i in 1..chunks {
                let idx = i * 4;
                let input_vec = v128_load(input.as_ptr().add(idx) as *const v128);
                min_vec = f32x4_pmin(min_vec, input_vec);
            }

            // Find min of the 4 elements in min_vec
            let mut result = f32x4_extract_lane::<0>(min_vec)
                .min(f32x4_extract_lane::<1>(min_vec))
                .min(f32x4_extract_lane::<2>(min_vec))
                .min(f32x4_extract_lane::<3>(min_vec));

            // Handle remaining elements
            for i in (chunks * 4)..(chunks * 4 + remainder) {
                result = result.min(input[i]);
            }

            Ok(result)
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn min_f32(&self, input: &[f32]) -> CpuResult<f32> {
        if input.is_empty() {
            return Err(TorshError::ComputeError(
                "Cannot find min of empty array".to_string(),
            ));
        }
        Ok(input.iter().fold(f32::INFINITY, |a, &b| a.min(b)))
    }

    /// Vector comparison - greater than
    #[cfg(target_arch = "wasm32")]
    pub fn greater_than_f32(&self, a: &[f32], b: &[f32], result: &mut [u32]) -> CpuResult<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return Err(TorshError::ComputeError(
                "Array length mismatch".to_string(),
            ));
        }

        if !self.simd_available {
            return self.greater_than_f32_scalar(a, b, result);
        }

        unsafe {
            let chunks = a.len() / 4;
            let remainder = a.len() % 4;

            for i in 0..chunks {
                let idx = i * 4;

                let va = v128_load(a.as_ptr().add(idx) as *const v128);
                let vb = v128_load(b.as_ptr().add(idx) as *const v128);

                let mask = f32x4_gt(va, vb);

                v128_store(result.as_mut_ptr().add(idx) as *mut v128, mask);
            }

            // Handle remaining elements
            for i in (chunks * 4)..(chunks * 4 + remainder) {
                result[i] = if a[i] > b[i] { u32::MAX } else { 0 };
            }
        }

        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn greater_than_f32(&self, a: &[f32], b: &[f32], result: &mut [u32]) -> CpuResult<()> {
        self.greater_than_f32_scalar(a, b, result)
    }

    /// Scalar fallback for greater than comparison
    fn greater_than_f32_scalar(&self, a: &[f32], b: &[f32], result: &mut [u32]) -> CpuResult<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return Err(TorshError::ComputeError(
                "Array length mismatch".to_string(),
            ));
        }

        for i in 0..a.len() {
            result[i] = if a[i] > b[i] { u32::MAX } else { 0 };
        }

        Ok(())
    }

    /// Vector select using mask (ternary operator)
    #[cfg(target_arch = "wasm32")]
    pub fn select_f32(
        &self,
        mask: &[u32],
        a: &[f32],
        b: &[f32],
        result: &mut [f32],
    ) -> CpuResult<()> {
        if mask.len() != a.len() || a.len() != b.len() || a.len() != result.len() {
            return Err(TorshError::ComputeError(
                "Array length mismatch".to_string(),
            ));
        }

        if !self.simd_available {
            return self.select_f32_scalar(mask, a, b, result);
        }

        unsafe {
            let chunks = a.len() / 4;
            let remainder = a.len() % 4;

            for i in 0..chunks {
                let idx = i * 4;

                let vmask = v128_load(mask.as_ptr().add(idx) as *const v128);
                let va = v128_load(a.as_ptr().add(idx) as *const v128);
                let vb = v128_load(b.as_ptr().add(idx) as *const v128);

                let vresult = v128_bitselect(va, vb, vmask);

                v128_store(result.as_mut_ptr().add(idx) as *mut v128, vresult);
            }

            // Handle remaining elements
            for i in (chunks * 4)..(chunks * 4 + remainder) {
                result[i] = if mask[i] != 0 { a[i] } else { b[i] };
            }
        }

        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn select_f32(
        &self,
        mask: &[u32],
        a: &[f32],
        b: &[f32],
        result: &mut [f32],
    ) -> CpuResult<()> {
        self.select_f32_scalar(mask, a, b, result)
    }

    /// Scalar fallback for select operation
    fn select_f32_scalar(
        &self,
        mask: &[u32],
        a: &[f32],
        b: &[f32],
        result: &mut [f32],
    ) -> CpuResult<()> {
        if mask.len() != a.len() || a.len() != b.len() || a.len() != result.len() {
            return Err(TorshError::ComputeError(
                "Array length mismatch".to_string(),
            ));
        }

        for i in 0..a.len() {
            result[i] = if mask[i] != 0 { a[i] } else { b[i] };
        }

        Ok(())
    }

    /// Get performance characteristics for WASM SIMD
    pub fn get_performance_info(&self) -> WasmSimdPerformanceInfo {
        WasmSimdPerformanceInfo {
            simd_available: self.simd_available,
            vector_width: self.vector_width,
            estimated_speedup: if self.simd_available { 3.0 } else { 1.0 }, // Better estimate
            supports_f32: true,
            supports_f64: self.simd_available && cfg!(target_feature = "simd128"), // More accurate
            supports_i32: self.simd_available,
            memory_bandwidth_gbps: if self.simd_available { 15.0 } else { 8.0 },
            browser_compatibility: self.check_browser_compatibility(),
        }
    }
}

/// Browser compatibility information for WASM operations
#[derive(Debug, Clone)]
pub struct BrowserCompatibility {
    /// WASM SIMD support
    pub wasm_simd: bool,
    /// WASM threads support (experimental)
    pub wasm_threads: bool,
    /// SharedArrayBuffer support
    pub shared_array_buffer: bool,
    /// OffscreenCanvas support
    pub offscreen_canvas: bool,
    /// Web Workers support
    pub web_workers: bool,
    /// WebGL2 support
    pub webgl2: bool,
    /// Estimated memory limit in MB
    pub estimated_memory_limit_mb: usize,
}

impl Default for BrowserCompatibility {
    fn default() -> Self {
        Self {
            wasm_simd: false,
            wasm_threads: false,
            shared_array_buffer: false,
            offscreen_canvas: false,
            web_workers: false,
            webgl2: false,
            estimated_memory_limit_mb: 512,
        }
    }
}

/// Performance information for WASM SIMD operations
#[derive(Debug, Clone)]
pub struct WasmSimdPerformanceInfo {
    /// Whether SIMD is available
    pub simd_available: bool,
    /// Vector width in bytes
    pub vector_width: usize,
    /// Estimated speedup over scalar operations
    pub estimated_speedup: f32,
    /// Support for different data types
    pub supports_f32: bool,
    pub supports_f64: bool,
    pub supports_i32: bool,
    /// Estimated memory bandwidth in GB/s
    pub memory_bandwidth_gbps: f32,
    /// Browser compatibility information
    pub browser_compatibility: BrowserCompatibility,
}

/// WASM deployment configuration for optimizing performance
#[derive(Debug, Clone)]
pub struct WasmDeploymentConfig {
    /// Target memory limit in MB
    pub memory_limit_mb: usize,
    /// Enable aggressive optimizations (may increase code size)
    pub aggressive_optimizations: bool,
    /// Use web workers for parallel processing
    pub use_web_workers: bool,
    /// Prefer memory efficiency over speed
    pub memory_efficient: bool,
    /// Block size for matrix operations
    pub matrix_block_size: usize,
    /// Enable debug features for development
    pub debug_mode: bool,
}

impl Default for WasmDeploymentConfig {
    fn default() -> Self {
        Self {
            memory_limit_mb: 512,
            aggressive_optimizations: false,
            use_web_workers: false,
            memory_efficient: true,
            matrix_block_size: 64,
            debug_mode: false,
        }
    }
}

impl WasmDeploymentConfig {
    /// Create configuration optimized for mobile browsers
    pub fn mobile_optimized() -> Self {
        Self {
            memory_limit_mb: 256,
            aggressive_optimizations: false,
            use_web_workers: false,
            memory_efficient: true,
            matrix_block_size: 32,
            debug_mode: false,
        }
    }

    /// Create configuration optimized for desktop browsers
    pub fn desktop_optimized() -> Self {
        Self {
            memory_limit_mb: 2048,
            aggressive_optimizations: true,
            use_web_workers: true,
            memory_efficient: false,
            matrix_block_size: 128,
            debug_mode: false,
        }
    }

    /// Create configuration for development/debugging
    pub fn debug() -> Self {
        Self {
            memory_limit_mb: 1024,
            aggressive_optimizations: false,
            use_web_workers: false,
            memory_efficient: true,
            matrix_block_size: 64,
            debug_mode: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wasm_simd_creation() {
        let wasm_ops = WasmSimdOps::new();
        let info = wasm_ops.get_performance_info();

        // SIMD availability depends on target architecture
        #[cfg(target_arch = "wasm32")]
        assert!(info.simd_available);

        #[cfg(not(target_arch = "wasm32"))]
        assert!(!info.simd_available);

        assert!(info.supports_f32);
        assert!(info.estimated_speedup >= 1.0);
    }

    #[test]
    fn test_add_f32() {
        let wasm_ops = WasmSimdOps::new();
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0, 3.0, 4.0, 5.0, 6.0];
        let mut result = vec![0.0; 5];

        let res = wasm_ops.add_f32(&a, &b, &mut result);
        assert!(res.is_ok());

        let expected = vec![3.0, 5.0, 7.0, 9.0, 11.0];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_mul_f32() {
        let wasm_ops = WasmSimdOps::new();
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];
        let mut result = vec![0.0; 4];

        let res = wasm_ops.mul_f32(&a, &b, &mut result);
        assert!(res.is_ok());

        let expected = vec![2.0, 6.0, 12.0, 20.0];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_dot_product() {
        let wasm_ops = WasmSimdOps::new();
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];

        let result = wasm_ops.dot_product_f32(&a, &b).unwrap();
        let expected = 1.0 * 2.0 + 2.0 * 3.0 + 3.0 * 4.0 + 4.0 * 5.0; // = 40.0
        assert!((result - expected).abs() < 1e-6);
    }

    #[test]
    fn test_relu() {
        let wasm_ops = WasmSimdOps::new();
        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let mut output = vec![0.0; 5];

        let res = wasm_ops.relu_f32(&input, &mut output);
        assert!(res.is_ok());

        let expected = vec![0.0, 0.0, 0.0, 1.0, 2.0];
        assert_eq!(output, expected);
    }

    #[test]
    fn test_softmax() {
        let wasm_ops = WasmSimdOps::new();
        let input = vec![1.0, 2.0, 3.0];
        let mut output = vec![0.0; 3];

        let res = wasm_ops.softmax_f32(&input, &mut output);
        assert!(res.is_ok());

        // Check that probabilities sum to 1
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Check that values are in [0, 1]
        for &val in &output {
            assert!(val >= 0.0 && val <= 1.0);
        }

        // Check monotonicity (larger input should give larger output)
        assert!(output[0] < output[1]);
        assert!(output[1] < output[2]);
    }

    #[test]
    fn test_matmul() {
        let wasm_ops = WasmSimdOps::new();

        // 2x3 * 3x2 = 2x2 matrix multiplication
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3x2
        let mut result = vec![0.0; 4]; // 2x2

        let res = wasm_ops.matmul_f32(&a, &b, &mut result, 2, 2, 3);
        assert!(res.is_ok());

        // Expected result:
        // [1*1+2*3+3*5, 1*2+2*4+3*6] = [22, 28]
        // [4*1+5*3+6*5, 4*2+5*4+6*6] = [49, 64]
        let expected = vec![22.0, 28.0, 49.0, 64.0];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_array_length_mismatch() {
        let wasm_ops = WasmSimdOps::new();
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0]; // Different length
        let mut result = vec![0.0; 2];

        let res = wasm_ops.add_f32(&a, &b, &mut result);
        assert!(res.is_err());
    }

    #[test]
    fn test_performance_info() {
        let wasm_ops = WasmSimdOps::new();
        let info = wasm_ops.get_performance_info();

        assert!(info.vector_width > 0);
        assert!(info.estimated_speedup >= 1.0);
        assert!(info.memory_bandwidth_gbps > 0.0);
    }

    #[test]
    fn test_sum_f32() {
        let wasm_ops = WasmSimdOps::new();
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let result = wasm_ops.sum_f32(&input).unwrap();
        assert_eq!(result, 15.0);

        // Test empty array
        let empty: Vec<f32> = vec![];
        let empty_result = wasm_ops.sum_f32(&empty).unwrap();
        assert_eq!(empty_result, 0.0);
    }

    #[test]
    fn test_add_i32() {
        let wasm_ops = WasmSimdOps::new();
        let a = vec![1, 2, 3, 4, 5];
        let b = vec![2, 3, 4, 5, 6];
        let mut result = vec![0; 5];

        let res = wasm_ops.add_i32(&a, &b, &mut result);
        assert!(res.is_ok());

        let expected = vec![3, 5, 7, 9, 11];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_mul_i32() {
        let wasm_ops = WasmSimdOps::new();
        let a = vec![1, 2, 3, 4];
        let b = vec![2, 3, 4, 5];
        let mut result = vec![0; 4];

        let res = wasm_ops.mul_i32(&a, &b, &mut result);
        assert!(res.is_ok());

        let expected = vec![2, 6, 12, 20];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_max_f32() {
        let wasm_ops = WasmSimdOps::new();
        let input = vec![1.0, 5.0, 2.0, 8.0, 3.0];

        let result = wasm_ops.max_f32(&input).unwrap();
        assert_eq!(result, 8.0);

        // Test single element
        let single = vec![42.0];
        let single_result = wasm_ops.max_f32(&single).unwrap();
        assert_eq!(single_result, 42.0);

        // Test empty array should fail
        let empty: Vec<f32> = vec![];
        let empty_result = wasm_ops.max_f32(&empty);
        assert!(empty_result.is_err());
    }

    #[test]
    fn test_min_f32() {
        let wasm_ops = WasmSimdOps::new();
        let input = vec![5.0, 1.0, 8.0, 2.0, 3.0];

        let result = wasm_ops.min_f32(&input).unwrap();
        assert_eq!(result, 1.0);

        // Test with negative numbers
        let input_neg = vec![-5.0, -1.0, -8.0, -2.0];
        let result_neg = wasm_ops.min_f32(&input_neg).unwrap();
        assert_eq!(result_neg, -8.0);
    }

    #[test]
    fn test_greater_than_f32() {
        let wasm_ops = WasmSimdOps::new();
        let a = vec![1.0, 3.0, 5.0, 2.0];
        let b = vec![2.0, 2.0, 4.0, 3.0];
        let mut result = vec![0u32; 4];

        let res = wasm_ops.greater_than_f32(&a, &b, &mut result);
        assert!(res.is_ok());

        // Expected: [false, true, true, false]
        let expected = vec![0, u32::MAX, u32::MAX, 0];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_select_f32() {
        let wasm_ops = WasmSimdOps::new();
        let mask = vec![u32::MAX, 0, u32::MAX, 0]; // [true, false, true, false]
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![10.0, 20.0, 30.0, 40.0];
        let mut result = vec![0.0; 4];

        let res = wasm_ops.select_f32(&mask, &a, &b, &mut result);
        assert!(res.is_ok());

        // Should select a where mask is true, b where mask is false
        let expected = vec![1.0, 20.0, 3.0, 40.0];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_fma_f32() {
        let wasm_ops = WasmSimdOps::new();
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];
        let c = vec![1.0, 1.0, 1.0, 1.0];
        let mut result = vec![0.0; 4];

        let res = wasm_ops.fma_f32(&a, &b, &c, &mut result);
        assert!(res.is_ok());

        // a * b + c: [1*2+1, 2*3+1, 3*4+1, 4*5+1] = [3, 7, 13, 21]
        let expected = vec![3.0, 7.0, 13.0, 21.0];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_large_arrays_simd() {
        let wasm_ops = WasmSimdOps::new();
        let size = 1000;
        let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..size).map(|i| (i + 1) as f32).collect();
        let mut result = vec![0.0; size];

        // Test large array addition
        let res = wasm_ops.add_f32(&a, &b, &mut result);
        assert!(res.is_ok());

        // Verify first few and last few elements
        assert_eq!(result[0], 1.0); // 0 + 1
        assert_eq!(result[1], 3.0); // 1 + 2
        assert_eq!(result[size - 1], (2 * size - 1) as f32); // (size-1) + size

        // Test sum of large array
        let sum_result = wasm_ops.sum_f32(&a).unwrap();
        let expected: f32 = (0..size).map(|i| i as f32).sum();
        assert!((sum_result - expected).abs() < 1e-6);
    }

    #[test]
    fn test_deployment_configs() {
        let mobile_config = WasmDeploymentConfig::mobile_optimized();
        assert_eq!(mobile_config.memory_limit_mb, 256);
        assert_eq!(mobile_config.matrix_block_size, 32);
        assert!(!mobile_config.aggressive_optimizations);

        let desktop_config = WasmDeploymentConfig::desktop_optimized();
        assert_eq!(desktop_config.memory_limit_mb, 2048);
        assert_eq!(desktop_config.matrix_block_size, 128);
        assert!(desktop_config.aggressive_optimizations);

        let debug_config = WasmDeploymentConfig::debug();
        assert!(debug_config.debug_mode);
        assert!(!debug_config.aggressive_optimizations);
    }

    #[test]
    fn test_browser_compatibility() {
        let wasm_ops = WasmSimdOps::new();
        let compatibility = wasm_ops.check_browser_compatibility();

        // These should be reasonable values
        assert!(compatibility.estimated_memory_limit_mb > 0);

        // On non-WASM targets, most features should be false
        #[cfg(not(target_arch = "wasm32"))]
        {
            assert!(!compatibility.wasm_simd);
            assert!(!compatibility.wasm_threads);
        }
    }

    #[test]
    fn test_blocked_matmul() {
        let wasm_ops = WasmSimdOps::new();

        // Test 4x4 * 4x4 matrix multiplication with blocking
        let a = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];
        let b = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ]; // Identity matrix
        let mut result = vec![0.0; 16];

        let res = wasm_ops.matmul_f32_blocked(&a, &b, &mut result, 4, 4, 4, 2);
        assert!(res.is_ok());

        // Multiplying by identity should give the original matrix
        assert_eq!(result, a);
    }
}
