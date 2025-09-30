//! Fast Fourier Transform operations for all backends
//!
//! This module provides a unified interface for FFT operations across all backends,
//! with optimized implementations for each platform.

use crate::{BackendResult, Buffer, Device};
use torsh_core::dtype::DType;

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, string::String, vec::Vec};

/// FFT direction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FftDirection {
    /// Forward FFT
    Forward,
    /// Inverse FFT
    Inverse,
}

/// FFT normalization mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FftNormalization {
    /// No normalization
    None,
    /// Normalize by 1/N
    Backward,
    /// Normalize by 1/sqrt(N)
    Ortho,
}

/// FFT operation type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FftType {
    /// 1D Complex-to-Complex FFT
    C2C,
    /// 1D Real-to-Complex FFT
    R2C,
    /// 1D Complex-to-Real FFT
    C2R,
    /// 2D Complex-to-Complex FFT
    C2C2D,
    /// 2D Real-to-Complex FFT
    R2C2D,
    /// 2D Complex-to-Real FFT
    C2R2D,
    /// 3D Complex-to-Complex FFT
    C2C3D,
    /// 3D Real-to-Complex FFT
    R2C3D,
    /// 3D Complex-to-Real FFT
    C2R3D,
}

/// FFT execution plan
#[derive(Debug, Clone)]
pub struct FftPlan {
    /// Plan ID for caching
    pub id: String,
    /// FFT type
    pub fft_type: FftType,
    /// Transform dimensions
    pub dimensions: Vec<usize>,
    /// Batch size
    pub batch_size: usize,
    /// Input data type
    pub input_dtype: DType,
    /// Output data type
    pub output_dtype: DType,
    /// Direction
    pub direction: FftDirection,
    /// Normalization mode
    pub normalization: FftNormalization,
    /// Backend-specific plan data
    pub backend_data: Vec<u8>,
}

impl FftPlan {
    /// Create a new FFT plan
    pub fn new(
        fft_type: FftType,
        dimensions: Vec<usize>,
        batch_size: usize,
        input_dtype: DType,
        output_dtype: DType,
        direction: FftDirection,
        normalization: FftNormalization,
    ) -> Self {
        let id = format!(
            "{:?}_{:?}_{}_{}_{:?}_{:?}_{:?}",
            fft_type, dimensions, batch_size, input_dtype, output_dtype, direction, normalization
        );

        Self {
            id,
            fft_type,
            dimensions,
            batch_size,
            input_dtype,
            output_dtype,
            direction,
            normalization,
            backend_data: Vec::new(),
        }
    }

    /// Create a new 1D FFT plan with default parameters
    ///
    /// This is a convenience function for creating 1D FFT plans commonly used in benchmarks.
    ///
    /// # Arguments
    ///
    /// * `size` - Size of the 1D FFT
    /// * `direction` - Forward or inverse transform
    ///
    /// # Returns
    ///
    /// A new FftPlan configured for 1D transforms
    pub fn new_1d(size: usize, direction: FftDirection) -> Self {
        Self::new(
            FftType::C2C,
            vec![size],
            1, // Single batch
            DType::C64, // Complex64 input
            DType::C64, // Complex64 output
            direction,
            FftNormalization::None,
        )
    }

    /// Get the total number of elements in the transform
    pub fn total_elements(&self) -> usize {
        self.dimensions.iter().product::<usize>() * self.batch_size
    }

    /// Get input buffer size in bytes
    pub fn input_buffer_size(&self) -> usize {
        let element_size = match self.input_dtype {
            DType::F32 => 4,
            DType::F64 => 8,
            DType::C64 => 8,
            DType::C128 => 16,
            _ => 4, // Default to f32
        };

        self.total_elements() * element_size
    }

    /// Get output buffer size in bytes
    pub fn output_buffer_size(&self) -> usize {
        let element_size = match self.output_dtype {
            DType::F32 => 4,
            DType::F64 => 8,
            DType::C64 => 8,
            DType::C128 => 16,
            _ => 8, // Default to c32
        };

        match self.fft_type {
            FftType::R2C | FftType::R2C2D | FftType::R2C3D => {
                // Real-to-complex transforms have reduced output size
                let mut output_elements = self.batch_size;
                for (i, &dim) in self.dimensions.iter().enumerate() {
                    if i == self.dimensions.len() - 1 {
                        // Last dimension is halved + 1 for R2C
                        output_elements *= (dim / 2) + 1;
                    } else {
                        output_elements *= dim;
                    }
                }
                output_elements * element_size
            }
            _ => self.total_elements() * element_size,
        }
    }

    /// Check if the plan is valid
    pub fn is_valid(&self) -> bool {
        !self.dimensions.is_empty() && self.batch_size > 0 && self.dimensions.iter().all(|&d| d > 0)
    }
}

/// FFT operations trait
#[async_trait::async_trait]
pub trait FftOps: Send + Sync {
    /// Create an FFT plan
    async fn create_fft_plan(
        &self,
        device: &Device,
        plan: &FftPlan,
    ) -> BackendResult<Box<dyn FftExecutor>>;

    /// Execute a 1D FFT
    async fn fft_1d(
        &self,
        device: &Device,
        input: &Buffer,
        output: &Buffer,
        size: usize,
        direction: FftDirection,
        normalization: FftNormalization,
    ) -> BackendResult<()>;

    /// Execute a 2D FFT
    async fn fft_2d(
        &self,
        device: &Device,
        input: &Buffer,
        output: &Buffer,
        size: (usize, usize),
        direction: FftDirection,
        normalization: FftNormalization,
    ) -> BackendResult<()>;

    /// Execute a 3D FFT
    async fn fft_3d(
        &self,
        device: &Device,
        input: &Buffer,
        output: &Buffer,
        size: (usize, usize, usize),
        direction: FftDirection,
        normalization: FftNormalization,
    ) -> BackendResult<()>;

    /// Execute a batched FFT
    async fn fft_batch(
        &self,
        device: &Device,
        input: &Buffer,
        output: &Buffer,
        size: &[usize],
        batch_size: usize,
        direction: FftDirection,
        normalization: FftNormalization,
    ) -> BackendResult<()>;

    /// Execute a real-to-complex FFT
    async fn rfft(
        &self,
        device: &Device,
        input: &Buffer,
        output: &Buffer,
        size: &[usize],
        direction: FftDirection,
        normalization: FftNormalization,
    ) -> BackendResult<()>;

    /// Execute a complex-to-real FFT
    async fn irfft(
        &self,
        device: &Device,
        input: &Buffer,
        output: &Buffer,
        size: &[usize],
        normalization: FftNormalization,
    ) -> BackendResult<()>;

    /// Check if FFT operations are supported
    fn supports_fft(&self) -> bool;

    /// Get optimal FFT sizes for performance
    fn get_optimal_fft_sizes(&self, min_size: usize, max_size: usize) -> Vec<usize>;
}

/// FFT executor for cached plans
#[async_trait::async_trait]
pub trait FftExecutor: Send + Sync {
    /// Execute the FFT plan
    async fn execute(&self, device: &Device, input: &Buffer, output: &Buffer) -> BackendResult<()>;

    /// Get the plan this executor was created for
    fn plan(&self) -> &FftPlan;

    /// Get memory requirements for execution
    fn memory_requirements(&self) -> usize;

    /// Check if the executor is valid
    fn is_valid(&self) -> bool;
}

/// Default FFT operations implementation
pub struct DefaultFftOps;

#[async_trait::async_trait]
impl FftOps for DefaultFftOps {
    async fn create_fft_plan(
        &self,
        _device: &Device,
        plan: &FftPlan,
    ) -> BackendResult<Box<dyn FftExecutor>> {
        Ok(Box::new(DefaultFftExecutor { plan: plan.clone() }))
    }

    async fn fft_1d(
        &self,
        _device: &Device,
        _input: &Buffer,
        _output: &Buffer,
        _size: usize,
        _direction: FftDirection,
        _normalization: FftNormalization,
    ) -> BackendResult<()> {
        Err(torsh_core::error::TorshError::BackendError(
            "FFT operations not implemented for this backend".to_string(),
        ))
    }

    async fn fft_2d(
        &self,
        _device: &Device,
        _input: &Buffer,
        _output: &Buffer,
        _size: (usize, usize),
        _direction: FftDirection,
        _normalization: FftNormalization,
    ) -> BackendResult<()> {
        Err(torsh_core::error::TorshError::BackendError(
            "FFT operations not implemented for this backend".to_string(),
        ))
    }

    async fn fft_3d(
        &self,
        _device: &Device,
        _input: &Buffer,
        _output: &Buffer,
        _size: (usize, usize, usize),
        _direction: FftDirection,
        _normalization: FftNormalization,
    ) -> BackendResult<()> {
        Err(torsh_core::error::TorshError::BackendError(
            "FFT operations not implemented for this backend".to_string(),
        ))
    }

    async fn fft_batch(
        &self,
        _device: &Device,
        _input: &Buffer,
        _output: &Buffer,
        _size: &[usize],
        _batch_size: usize,
        _direction: FftDirection,
        _normalization: FftNormalization,
    ) -> BackendResult<()> {
        Err(torsh_core::error::TorshError::BackendError(
            "FFT operations not implemented for this backend".to_string(),
        ))
    }

    async fn rfft(
        &self,
        _device: &Device,
        _input: &Buffer,
        _output: &Buffer,
        _size: &[usize],
        _direction: FftDirection,
        _normalization: FftNormalization,
    ) -> BackendResult<()> {
        Err(torsh_core::error::TorshError::BackendError(
            "FFT operations not implemented for this backend".to_string(),
        ))
    }

    async fn irfft(
        &self,
        _device: &Device,
        _input: &Buffer,
        _output: &Buffer,
        _size: &[usize],
        _normalization: FftNormalization,
    ) -> BackendResult<()> {
        Err(torsh_core::error::TorshError::BackendError(
            "FFT operations not implemented for this backend".to_string(),
        ))
    }

    fn supports_fft(&self) -> bool {
        false
    }

    fn get_optimal_fft_sizes(&self, min_size: usize, max_size: usize) -> Vec<usize> {
        // Return power-of-2 sizes as default
        let mut sizes = Vec::new();
        let mut size = 1;
        while size < min_size {
            size *= 2;
        }
        while size <= max_size {
            sizes.push(size);
            size *= 2;
        }
        sizes
    }
}

/// Default FFT executor implementation
pub struct DefaultFftExecutor {
    plan: FftPlan,
}

#[async_trait::async_trait]
impl FftExecutor for DefaultFftExecutor {
    async fn execute(
        &self,
        _device: &Device,
        _input: &Buffer,
        _output: &Buffer,
    ) -> BackendResult<()> {
        Err(torsh_core::error::TorshError::BackendError(
            "FFT execution not implemented for this backend".to_string(),
        ))
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

/// Convenience functions for common FFT operations
pub mod convenience {
    use super::*;

    /// Create a 1D complex-to-complex FFT plan
    pub fn create_c2c_1d_plan(
        size: usize,
        batch_size: usize,
        direction: FftDirection,
        normalization: FftNormalization,
    ) -> FftPlan {
        FftPlan::new(
            FftType::C2C,
            vec![size],
            batch_size,
            DType::C64,
            DType::C64,
            direction,
            normalization,
        )
    }

    /// Create a 1D real-to-complex FFT plan
    pub fn create_r2c_1d_plan(
        size: usize,
        batch_size: usize,
        normalization: FftNormalization,
    ) -> FftPlan {
        FftPlan::new(
            FftType::R2C,
            vec![size],
            batch_size,
            DType::F32,
            DType::C64,
            FftDirection::Forward,
            normalization,
        )
    }

    /// Create a 2D complex-to-complex FFT plan
    pub fn create_c2c_2d_plan(
        size: (usize, usize),
        batch_size: usize,
        direction: FftDirection,
        normalization: FftNormalization,
    ) -> FftPlan {
        FftPlan::new(
            FftType::C2C2D,
            vec![size.0, size.1],
            batch_size,
            DType::C64,
            DType::C64,
            direction,
            normalization,
        )
    }

    /// Create a 3D complex-to-complex FFT plan
    pub fn create_c2c_3d_plan(
        size: (usize, usize, usize),
        batch_size: usize,
        direction: FftDirection,
        normalization: FftNormalization,
    ) -> FftPlan {
        FftPlan::new(
            FftType::C2C3D,
            vec![size.0, size.1, size.2],
            batch_size,
            DType::C64,
            DType::C64,
            direction,
            normalization,
        )
    }

    /// Get the next power of 2 greater than or equal to n
    pub fn next_power_of_2(n: usize) -> usize {
        if n == 0 {
            return 1;
        }
        let mut power = 1;
        while power < n {
            power *= 2;
        }
        power
    }

    /// Check if a size is optimal for FFT (power of 2, 3, 5, 7)
    pub fn is_optimal_fft_size(size: usize) -> bool {
        if size == 0 {
            return false;
        }

        let mut n = size;
        for prime in &[2, 3, 5, 7] {
            while n % prime == 0 {
                n /= prime;
            }
        }

        n == 1
    }

    /// Find the next optimal FFT size
    pub fn next_optimal_fft_size(size: usize) -> usize {
        let mut candidate = size;
        while !is_optimal_fft_size(candidate) {
            candidate += 1;
        }
        candidate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fft_plan_creation() {
        let plan = FftPlan::new(
            FftType::C2C,
            vec![1024],
            1,
            DType::C64,
            DType::C64,
            FftDirection::Forward,
            FftNormalization::None,
        );

        assert_eq!(plan.fft_type, FftType::C2C);
        assert_eq!(plan.dimensions, vec![1024]);
        assert_eq!(plan.batch_size, 1);
        assert_eq!(plan.input_dtype, DType::C64);
        assert_eq!(plan.output_dtype, DType::C64);
        assert_eq!(plan.direction, FftDirection::Forward);
        assert_eq!(plan.normalization, FftNormalization::None);
        assert!(plan.is_valid());
    }

    #[test]
    fn test_fft_plan_buffer_sizes() {
        let plan = FftPlan::new(
            FftType::C2C,
            vec![1024],
            1,
            DType::C64,
            DType::C64,
            FftDirection::Forward,
            FftNormalization::None,
        );

        assert_eq!(plan.input_buffer_size(), 1024 * 8); // C32 is 8 bytes
        assert_eq!(plan.output_buffer_size(), 1024 * 8);
    }

    #[test]
    fn test_r2c_plan_buffer_sizes() {
        let plan = FftPlan::new(
            FftType::R2C,
            vec![1024],
            1,
            DType::F32,
            DType::C64,
            FftDirection::Forward,
            FftNormalization::None,
        );

        assert_eq!(plan.input_buffer_size(), 1024 * 4); // F32 is 4 bytes
        assert_eq!(plan.output_buffer_size(), (1024 / 2 + 1) * 8); // C32 is 8 bytes, output is N/2+1
    }

    #[test]
    fn test_convenience_functions() {
        let plan =
            convenience::create_c2c_1d_plan(1024, 1, FftDirection::Forward, FftNormalization::None);

        assert_eq!(plan.fft_type, FftType::C2C);
        assert_eq!(plan.dimensions, vec![1024]);
        assert!(plan.is_valid());
    }

    #[test]
    fn test_optimal_fft_sizes() {
        assert!(convenience::is_optimal_fft_size(1024)); // 2^10
        assert!(convenience::is_optimal_fft_size(1080)); // 2^3 * 3^3 * 5
        assert!(!convenience::is_optimal_fft_size(1023)); // Prime

        assert_eq!(convenience::next_power_of_2(1000), 1024);
        assert_eq!(convenience::next_power_of_2(1024), 1024);

        assert_eq!(convenience::next_optimal_fft_size(1023), 1024);
        assert_eq!(convenience::next_optimal_fft_size(1024), 1024);
    }

    #[test]
    fn test_default_fft_ops() {
        let ops = DefaultFftOps;
        assert!(!ops.supports_fft());

        let sizes = ops.get_optimal_fft_sizes(100, 2000);
        assert!(!sizes.is_empty());
        assert!(sizes.iter().all(|&size| size >= 100 && size <= 2000));
        assert!(sizes.iter().all(|&size| size.is_power_of_two()));
    }
}
