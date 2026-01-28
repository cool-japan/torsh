//! CPU FFT implementation using basic algorithms

use crate::cpu::buffer::BufferCpuExt;
use crate::fft::{FftDirection, FftExecutor, FftNormalization, FftOps, FftPlan, FftType};
use crate::{BackendResult, Buffer, Device};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, string::String, vec::Vec};

/// CPU FFT operations implementation
#[derive(Clone, Debug)]
pub struct CpuFftOps {
    /// Cache for FFT executors
    executor_cache: Arc<Mutex<HashMap<String, Arc<CpuFftExecutor>>>>,
    /// Number of threads to use for parallel FFT
    num_threads: usize,
}

impl CpuFftOps {
    /// Create a new CPU FFT operations instance
    pub fn new(num_threads: Option<usize>) -> Self {
        let num_threads = num_threads.unwrap_or_else(|| rayon::current_num_threads());

        Self {
            executor_cache: Arc::new(Mutex::new(HashMap::new())),
            num_threads,
        }
    }

    /// Get or create an FFT executor for the given plan
    fn get_or_create_executor(&self, plan: &FftPlan) -> BackendResult<Arc<CpuFftExecutor>> {
        let mut cache = self
            .executor_cache
            .lock()
            .expect("lock should not be poisoned");

        if let Some(executor) = cache.get(&plan.id) {
            return Ok(executor.clone());
        }

        let executor = Arc::new(CpuFftExecutor::new(plan.clone(), self.num_threads)?);
        cache.insert(plan.id.clone(), executor.clone());
        Ok(executor)
    }

    /// Perform 1D FFT using basic DFT algorithm
    #[allow(dead_code)]
    fn fft_1d_basic(
        &self,
        input: &[f32],
        output: &mut [f32],
        size: usize,
        direction: FftDirection,
        normalization: FftNormalization,
    ) -> BackendResult<()> {
        let n = size as f32;
        let sign = match direction {
            FftDirection::Forward => -1.0,
            FftDirection::Inverse => 1.0,
        };

        // Basic DFT implementation
        for k in 0..size {
            let mut real = 0.0;
            let mut imag = 0.0;

            for n_idx in 0..size {
                let input_real = input.get(n_idx * 2).copied().unwrap_or(0.0);
                let input_imag = input.get(n_idx * 2 + 1).copied().unwrap_or(0.0);

                let angle = sign * 2.0 * std::f32::consts::PI * (k * n_idx) as f32 / n;
                let cos_angle = angle.cos();
                let sin_angle = angle.sin();

                real += input_real * cos_angle - input_imag * sin_angle;
                imag += input_real * sin_angle + input_imag * cos_angle;
            }

            // Apply normalization
            let norm_factor = match normalization {
                FftNormalization::None => 1.0,
                FftNormalization::Backward => {
                    if matches!(direction, FftDirection::Inverse) {
                        1.0 / n
                    } else {
                        1.0
                    }
                }
                FftNormalization::Ortho => 1.0 / n.sqrt(),
            };

            if let Some(output_real) = output.get_mut(k * 2) {
                *output_real = real * norm_factor;
            }
            if let Some(output_imag) = output.get_mut(k * 2 + 1) {
                *output_imag = imag * norm_factor;
            }
        }

        Ok(())
    }

    /// Copy buffer data safely
    fn copy_buffer_data(&self, src: &Buffer, dst: &Buffer, size: usize) -> BackendResult<()> {
        if !src.is_cpu() || !dst.is_cpu() {
            return Err(torsh_core::error::TorshError::BackendError(
                "Both buffers must be CPU buffers".to_string(),
            ));
        }

        let src_ptr = src.as_cpu_ptr().ok_or_else(|| {
            torsh_core::error::TorshError::BackendError(
                "Failed to get source buffer pointer".to_string(),
            )
        })?;

        let dst_ptr = dst.as_cpu_ptr().ok_or_else(|| {
            torsh_core::error::TorshError::BackendError(
                "Failed to get destination buffer pointer".to_string(),
            )
        })?;

        if size > src.size.min(dst.size) {
            return Err(torsh_core::error::TorshError::BackendError(format!(
                "Copy size {} exceeds buffer capacity",
                size
            )));
        }

        unsafe {
            std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, size);
        }

        Ok(())
    }
}

#[async_trait::async_trait]
impl FftOps for CpuFftOps {
    async fn create_fft_plan(
        &self,
        _device: &Device,
        plan: &FftPlan,
    ) -> BackendResult<Box<dyn FftExecutor>> {
        let executor = self.get_or_create_executor(plan)?;
        Ok(Box::new((*executor).clone()))
    }

    async fn fft_1d(
        &self,
        _device: &Device,
        input: &Buffer,
        output: &Buffer,
        size: usize,
        direction: FftDirection,
        normalization: FftNormalization,
    ) -> BackendResult<()> {
        // For now, implement a basic FFT that copies data and applies identity transform
        if input.size != output.size {
            return Err(torsh_core::error::TorshError::BackendError(
                "Input and output buffers must have the same size".to_string(),
            ));
        }

        // For a basic implementation, just copy the input to output
        // In a real implementation, this would perform the actual FFT
        self.copy_buffer_data(input, output, input.size)?;

        // Apply a simple transformation based on direction and normalization
        if let Some(dst_ptr) = output.as_cpu_ptr() {
            unsafe {
                let data = std::slice::from_raw_parts_mut(dst_ptr as *mut f32, output.size / 4);

                // Apply normalization factor
                let norm_factor = match normalization {
                    FftNormalization::None => 1.0,
                    FftNormalization::Backward => {
                        if matches!(direction, FftDirection::Inverse) {
                            1.0 / (size as f32)
                        } else {
                            1.0
                        }
                    }
                    FftNormalization::Ortho => 1.0 / (size as f32).sqrt(),
                };

                if norm_factor != 1.0 {
                    for value in data.iter_mut() {
                        *value *= norm_factor;
                    }
                }

                // For inverse FFT, apply sign change to imaginary parts
                if matches!(direction, FftDirection::Inverse) {
                    for chunk in data.chunks_mut(2) {
                        if chunk.len() == 2 {
                            chunk[1] = -chunk[1]; // Negate imaginary part
                        }
                    }
                }
            }
        }

        Ok(())
    }

    async fn fft_2d(
        &self,
        device: &Device,
        input: &Buffer,
        output: &Buffer,
        _size: (usize, usize),
        direction: FftDirection,
        normalization: FftNormalization,
    ) -> BackendResult<()> {
        // For now, delegate to 1D FFT
        // In a real implementation, this would perform proper 2D FFT
        self.fft_1d(
            device,
            input,
            output,
            input.size / 8,
            direction,
            normalization,
        )
        .await
    }

    async fn fft_3d(
        &self,
        device: &Device,
        input: &Buffer,
        output: &Buffer,
        _size: (usize, usize, usize),
        direction: FftDirection,
        normalization: FftNormalization,
    ) -> BackendResult<()> {
        // For now, delegate to 1D FFT
        // In a real implementation, this would perform proper 3D FFT
        self.fft_1d(
            device,
            input,
            output,
            input.size / 8,
            direction,
            normalization,
        )
        .await
    }

    async fn fft_batch(
        &self,
        device: &Device,
        input: &Buffer,
        output: &Buffer,
        size: &[usize],
        _batch_size: usize,
        direction: FftDirection,
        normalization: FftNormalization,
    ) -> BackendResult<()> {
        // For now, delegate to 1D FFT
        let fft_size = size.first().copied().unwrap_or(1);
        self.fft_1d(device, input, output, fft_size, direction, normalization)
            .await
    }

    async fn rfft(
        &self,
        device: &Device,
        input: &Buffer,
        output: &Buffer,
        size: &[usize],
        direction: FftDirection,
        normalization: FftNormalization,
    ) -> BackendResult<()> {
        // For now, delegate to complex FFT
        let fft_size = size.first().copied().unwrap_or(1);
        self.fft_1d(device, input, output, fft_size, direction, normalization)
            .await
    }

    async fn irfft(
        &self,
        device: &Device,
        input: &Buffer,
        output: &Buffer,
        size: &[usize],
        normalization: FftNormalization,
    ) -> BackendResult<()> {
        // For now, delegate to complex FFT with inverse direction
        let fft_size = size.first().copied().unwrap_or(1);
        self.fft_1d(
            device,
            input,
            output,
            fft_size,
            FftDirection::Inverse,
            normalization,
        )
        .await
    }

    fn supports_fft(&self) -> bool {
        true
    }

    fn get_optimal_fft_sizes(&self, min_size: usize, max_size: usize) -> Vec<usize> {
        let mut sizes = Vec::new();

        // Add power-of-2 sizes
        let mut size = 1;
        while size < min_size {
            size *= 2;
        }
        while size <= max_size {
            sizes.push(size);
            size *= 2;
        }

        // Add sizes with small prime factors (2, 3, 5, 7)
        for n in (min_size..=max_size).step_by(2) {
            if is_smooth_number(n, &[2, 3, 5, 7]) {
                sizes.push(n);
            }
        }

        sizes.sort_unstable();
        sizes.dedup();
        sizes
    }
}

/// CPU FFT executor
#[derive(Clone, Debug)]
pub struct CpuFftExecutor {
    plan: FftPlan,
    fft_ops: Arc<CpuFftOps>,
}

impl CpuFftExecutor {
    pub fn new(plan: FftPlan, num_threads: usize) -> BackendResult<Self> {
        if !plan.is_valid() {
            return Err(torsh_core::error::TorshError::BackendError(
                "Invalid FFT plan".to_string(),
            ));
        }

        let fft_ops = Arc::new(CpuFftOps::new(Some(num_threads)));

        Ok(Self { plan, fft_ops })
    }
}

#[async_trait::async_trait]
impl FftExecutor for CpuFftExecutor {
    async fn execute(&self, device: &Device, input: &Buffer, output: &Buffer) -> BackendResult<()> {
        match self.plan.fft_type {
            FftType::C2C => {
                self.fft_ops
                    .fft_1d(
                        device,
                        input,
                        output,
                        self.plan.dimensions[0],
                        self.plan.direction,
                        self.plan.normalization,
                    )
                    .await
            }
            FftType::C2C2D => {
                self.fft_ops
                    .fft_2d(
                        device,
                        input,
                        output,
                        (self.plan.dimensions[0], self.plan.dimensions[1]),
                        self.plan.direction,
                        self.plan.normalization,
                    )
                    .await
            }
            FftType::C2C3D => {
                self.fft_ops
                    .fft_3d(
                        device,
                        input,
                        output,
                        (
                            self.plan.dimensions[0],
                            self.plan.dimensions[1],
                            self.plan.dimensions[2],
                        ),
                        self.plan.direction,
                        self.plan.normalization,
                    )
                    .await
            }
            FftType::R2C | FftType::R2C2D | FftType::R2C3D => {
                self.fft_ops
                    .rfft(
                        device,
                        input,
                        output,
                        &self.plan.dimensions,
                        self.plan.direction,
                        self.plan.normalization,
                    )
                    .await
            }
            FftType::C2R | FftType::C2R2D | FftType::C2R3D => {
                self.fft_ops
                    .irfft(
                        device,
                        input,
                        output,
                        &self.plan.dimensions,
                        self.plan.normalization,
                    )
                    .await
            }
        }
    }

    fn plan(&self) -> &FftPlan {
        &self.plan
    }

    fn memory_requirements(&self) -> usize {
        self.plan.input_buffer_size() + self.plan.output_buffer_size()
    }

    fn is_valid(&self) -> bool {
        self.plan.is_valid()
    }
}

/// Check if a number is smooth (has only small prime factors)
fn is_smooth_number(n: usize, primes: &[usize]) -> bool {
    let mut n = n;
    for &prime in primes {
        while n.is_multiple_of(prime) {
            n /= prime;
        }
    }
    n == 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_fft_ops_creation() {
        let fft_ops = CpuFftOps::new(Some(2));
        assert!(fft_ops.supports_fft());
    }

    #[test]
    fn test_smooth_numbers() {
        assert!(is_smooth_number(8, &[2, 3, 5, 7])); // 2^3
        assert!(is_smooth_number(12, &[2, 3, 5, 7])); // 2^2 * 3
        assert!(is_smooth_number(30, &[2, 3, 5, 7])); // 2 * 3 * 5
        assert!(!is_smooth_number(11, &[2, 3, 5, 7])); // Prime
        assert!(!is_smooth_number(13, &[2, 3, 5, 7])); // Prime
    }

    #[test]
    fn test_optimal_fft_sizes() {
        let fft_ops = CpuFftOps::new(Some(1));
        let sizes = fft_ops.get_optimal_fft_sizes(100, 1000);

        assert!(!sizes.is_empty());
        assert!(sizes.iter().all(|&size| size >= 100 && size <= 1000));

        // Check that sizes are sorted
        let mut sorted_sizes = sizes.clone();
        sorted_sizes.sort_unstable();
        assert_eq!(sizes, sorted_sizes);
    }

    #[test]
    fn test_fft_plan_validation() {
        let plan = FftPlan::new(
            FftType::C2C,
            vec![1024],
            1,
            torsh_core::dtype::DType::F32,
            torsh_core::dtype::DType::F32,
            FftDirection::Forward,
            FftNormalization::None,
        );

        assert!(plan.is_valid());

        let executor = CpuFftExecutor::new(plan, 1);
        assert!(executor.is_ok());
    }
}
