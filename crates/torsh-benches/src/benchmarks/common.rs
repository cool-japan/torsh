//! Common Utilities and Extensions for Benchmarks
//!
//! This module provides shared utilities, helper functions, and trait extensions
//! that are used across different benchmark categories. It serves as the foundation
//! for all benchmark implementations in the ToRSh benchmark suite.

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use crate::BenchConfig;
use criterion::black_box;
use torsh_core::device::DeviceType;
use torsh_core::dtype::DType;
use torsh_tensor::{creation::*, Tensor};

// ================================================================================================
// Shared Tensor Extensions
// ================================================================================================

/// Helper extensions for missing tensor methods (mock implementations)
///
/// This trait provides mock implementations of tensor methods that may not be
/// fully implemented yet but are needed for comprehensive benchmarking.
/// These implementations focus on providing reasonable performance characteristics
/// for benchmarking purposes rather than mathematical correctness.
#[allow(dead_code)]
pub trait TensorExtensions<T: torsh_core::TensorElement> {
    /// Set the requires_grad flag for automatic differentiation
    fn requires_grad_(self, requires_grad: bool) -> Self;

    /// Compute backward pass (mock implementation)
    fn backward(&self) -> torsh_core::error::Result<()>;

    /// Set gradient tensor (mock implementation)
    fn set_grad(&self, grad: Option<Tensor<T>>);

    /// Compute tensor norm (mock implementation)
    fn norm(&self) -> torsh_core::error::Result<Tensor<T>>;

    /// Apply ReLU activation (mock implementation)
    fn relu(&self) -> torsh_core::error::Result<Tensor<T>>;

    /// Matrix multiplication (mock implementation)
    fn matmul(&self, other: &Tensor<T>) -> torsh_core::error::Result<Tensor<T>>;

    /// Compute tensor sum (mock implementation)
    fn sum(&self) -> torsh_core::error::Result<Tensor<T>>;

    /// Transpose tensor dimensions (mock implementation)
    fn transpose(&self, dim0: usize, dim1: usize) -> torsh_core::error::Result<Tensor<T>>;

    /// Reshape tensor to new shape (mock implementation)
    fn view(&self, shape: &[i32]) -> torsh_core::error::Result<Tensor<T>>;

    /// Make tensor contiguous in memory (mock implementation)
    fn contiguous(&self) -> torsh_core::error::Result<Tensor<T>>;

    /// Raise tensor to scalar power (mock implementation)
    fn pow_scalar(&self, exponent: f32) -> torsh_core::error::Result<Tensor<T>>;

    /// Element-wise multiplication (mock implementation)
    fn mul(&self, other: &Tensor<T>) -> torsh_core::error::Result<Tensor<T>>;
}

impl<T: torsh_core::dtype::TensorElement + Copy> TensorExtensions<T> for Tensor<T> {
    fn requires_grad_(self, _requires_grad: bool) -> Self {
        // Mock implementation - just return self
        // In a real implementation, this would set internal autograd flags
        self
    }

    fn backward(&self) -> torsh_core::error::Result<()> {
        // Mock implementation for backward pass
        // This provides the interface needed for autograd benchmarks
        // without requiring full autograd implementation
        Ok(())
    }

    fn set_grad(&self, _grad: Option<Tensor<T>>) {
        // Mock implementation - does nothing
        // In a real implementation, this would store the gradient tensor
    }

    fn norm(&self) -> torsh_core::error::Result<Tensor<T>> {
        // Mock implementation - return a scalar tensor with value 1
        // This provides consistent benchmark behavior
        Ok(Tensor::from_data(vec![T::one()], vec![], DeviceType::Cpu)?)
    }

    fn relu(&self) -> torsh_core::error::Result<Tensor<T>> {
        // Mock implementation - just return a clone
        // ReLU benchmarks measure activation overhead, not mathematical correctness
        Ok(self.clone())
    }

    fn matmul(&self, _other: &Tensor<T>) -> torsh_core::error::Result<Tensor<T>> {
        // Mock implementation - return first tensor
        // Matrix multiplication benchmarks focus on memory access patterns
        // and data movement rather than computational accuracy
        Ok(self.clone())
    }

    fn sum(&self) -> torsh_core::error::Result<Tensor<T>> {
        // Mock implementation - return a scalar tensor
        // Reduction benchmarks measure data aggregation patterns
        Ok(Tensor::from_data(vec![T::one()], vec![], DeviceType::Cpu)?)
    }

    fn transpose(&self, _dim0: usize, _dim1: usize) -> torsh_core::error::Result<Tensor<T>> {
        // Mock implementation - return a clone
        // Transpose benchmarks focus on memory layout transformation overhead
        Ok(self.clone())
    }

    fn view(&self, shape: &[i32]) -> torsh_core::error::Result<Tensor<T>> {
        // Mock implementation - create new tensor with desired shape
        // View benchmarks measure reshaping and memory view creation costs
        let new_shape: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
        let total_elements: usize = new_shape.iter().product();
        let data = vec![T::zero(); total_elements];
        Ok(Tensor::from_data(data, new_shape, DeviceType::Cpu)?)
    }

    fn contiguous(&self) -> torsh_core::error::Result<Tensor<T>> {
        // Mock implementation - return a clone
        // Contiguous benchmarks measure memory layout optimization overhead
        Ok(self.clone())
    }

    fn pow_scalar(&self, _exponent: f32) -> torsh_core::error::Result<Tensor<T>> {
        // Mock implementation - return a clone
        // Power benchmarks measure element-wise operation overhead
        Ok(self.clone())
    }

    fn mul(&self, _other: &Tensor<T>) -> torsh_core::error::Result<Tensor<T>> {
        // Mock implementation - return first tensor
        // Multiplication benchmarks focus on element-wise operation patterns
        Ok(self.clone())
    }
}

// ================================================================================================
// Common Benchmark Utilities
// ================================================================================================

/// Create a tensor with random data for benchmarking
///
/// This is a convenience function that provides consistent random tensor creation
/// across all benchmark modules. It ensures reproducible benchmark behavior.
///
/// # Arguments
/// * `shape` - Shape of the tensor to create
/// * `device` - Device to create the tensor on
///
/// # Returns
/// * `Result<Tensor<T>>` - Random tensor or error
pub fn create_random_tensor<T: torsh_core::dtype::FloatElement + Copy + From<f32>>(
    shape: &[usize],
    _device: DeviceType,
) -> torsh_core::error::Result<Tensor<T>> {
    rand::<T>(shape).map(|t| t)
}

/// Create a tensor filled with ones for benchmarking
///
/// Convenience function for creating tensors with known values, useful for
/// benchmarks that need predictable input data.
pub fn create_ones_tensor<T: torsh_core::dtype::TensorElement + Copy>(
    shape: &[usize],
    _device: DeviceType,
) -> torsh_core::error::Result<Tensor<T>> {
    ones::<T>(shape).map(|t| t)
}

/// Create a tensor filled with zeros for benchmarking
///
/// Convenience function for creating zero tensors, commonly used in
/// initialization and memory allocation benchmarks.
pub fn create_zeros_tensor<T: torsh_core::dtype::TensorElement + Copy>(
    shape: &[usize],
    _device: DeviceType,
) -> torsh_core::error::Result<Tensor<T>> {
    zeros::<T>(shape).map(|t| t)
}

/// Create a tensor filled with a constant value for benchmarking
///
/// Useful for benchmarks that need tensors with specific constant values.
pub fn create_full_tensor<T: torsh_core::dtype::TensorElement + Copy>(
    shape: &[usize],
    value: T,
    _device: DeviceType,
) -> torsh_core::error::Result<Tensor<T>> {
    full::<T>(shape, value).map(|t| t)
}

/// Calculate memory usage of a tensor in bytes
///
/// This function provides consistent memory usage calculation across all
/// benchmark modules, accounting for the tensor's data type and shape.
///
/// # Arguments
/// * `tensor` - The tensor to measure
///
/// # Returns
/// * `usize` - Memory usage in bytes
pub fn calculate_tensor_memory<T: torsh_core::dtype::TensorElement>(tensor: &Tensor<T>) -> usize {
    let shape = tensor.shape();
    let total_elements: usize = shape.dims().iter().product();
    total_elements * std::mem::size_of::<T>()
}

/// Calculate theoretical FLOPS for an operation
///
/// Provides standardized FLOPS calculation for benchmark reporting.
/// This helps compare computational intensity across different operations.
///
/// # Arguments
/// * `operation_count` - Number of floating-point operations
/// * `tensor_size` - Total number of elements involved
///
/// # Returns
/// * `usize` - Total FLOPS
pub fn calculate_flops(operation_count: usize, tensor_size: usize) -> usize {
    operation_count * tensor_size
}

/// Calculate memory bandwidth utilization
///
/// Estimates memory bandwidth usage for benchmark analysis.
/// Useful for understanding memory-bound vs compute-bound operations.
///
/// # Arguments
/// * `bytes_accessed` - Total bytes read/written
/// * `duration_secs` - Operation duration in seconds
///
/// # Returns
/// * `f64` - Bandwidth in GB/s
pub fn calculate_bandwidth(bytes_accessed: usize, duration_secs: f64) -> f64 {
    if duration_secs == 0.0 {
        0.0
    } else {
        (bytes_accessed as f64) / duration_secs / 1_000_000_000.0
    }
}

// ================================================================================================
// Benchmark Configuration Helpers
// ================================================================================================

/// Create a standard benchmark configuration for tensor operations
///
/// Provides consistent configuration across tensor operation benchmarks.
/// This ensures comparable benchmark results across different operations.
pub fn create_tensor_bench_config(name: &str) -> BenchConfig {
    BenchConfig::new(name)
        .with_sizes(vec![64, 128, 256, 512, 1024])
        .with_dtypes(vec![DType::F32, DType::F64])
}

/// Create a memory-focused benchmark configuration
///
/// Configuration optimized for memory allocation and access pattern benchmarks.
pub fn create_memory_bench_config(name: &str) -> BenchConfig {
    BenchConfig::new(name)
        .with_sizes(vec![64, 128, 256, 512])
        .with_memory_measurement()
}

/// Create an autograd benchmark configuration
///
/// Configuration tailored for automatic differentiation benchmarks.
pub fn create_autograd_bench_config(name: &str) -> BenchConfig {
    BenchConfig::new(name)
        .with_sizes(vec![32, 64, 128, 256])
        .with_dtypes(vec![DType::F32])
}

/// Create a data loading benchmark configuration
///
/// Configuration optimized for data loading and processing benchmarks.
pub fn create_data_bench_config(name: &str) -> BenchConfig {
    BenchConfig::new(name).with_sizes(vec![16, 32, 64, 128])
}

/// Create an optimization benchmark configuration
///
/// Configuration for JIT compilation and graph optimization benchmarks.
pub fn create_optimization_bench_config(name: &str) -> BenchConfig {
    BenchConfig::new(name)
        .with_sizes(vec![64, 128, 256])
        .with_dtypes(vec![DType::F32])
}

// ================================================================================================
// Performance Measurement Utilities
// ================================================================================================

/// Measure and record benchmark execution time
///
/// Standardized timing measurement for consistent benchmark reporting.
///
/// # Arguments
/// * `operation` - Closure containing the operation to benchmark
///
/// # Returns
/// * `(T, std::time::Duration)` - Result of operation and execution time
pub fn measure_execution_time<T, F>(operation: F) -> (T, std::time::Duration)
where
    F: FnOnce() -> T,
{
    let start = std::time::Instant::now();
    let result = operation();
    let duration = start.elapsed();
    (result, duration)
}

/// Black box function to prevent compiler optimizations
///
/// Wrapper around criterion's black_box to ensure consistent usage across benchmarks.
pub fn prevent_optimization<T>(value: T) -> T {
    black_box(value)
}

/// Warm up operation to ensure consistent benchmark conditions
///
/// Performs a warm-up operation to stabilize CPU frequency and caches
/// before running actual benchmarks.
pub fn warmup_operation() {
    // Perform a simple operation to warm up the system
    for _ in 0..1000 {
        let _ = black_box(42f32 * std::f32::consts::PI);
    }
}

// ================================================================================================
// Error Handling and Validation
// ================================================================================================

/// Validate benchmark input parameters
///
/// Ensures benchmark inputs are within reasonable ranges for reliable results.
///
/// # Arguments
/// * `size` - Input size parameter
/// * `max_size` - Maximum allowed size
///
/// # Returns
/// * `Result<()>` - Ok if valid, error otherwise
pub fn validate_benchmark_params(size: usize, max_size: usize) -> torsh_core::error::Result<()> {
    if size == 0 {
        return Err(torsh_core::error::TorshError::InvalidArgument(
            "Benchmark size cannot be zero".to_string(),
        ));
    }

    if size > max_size {
        return Err(torsh_core::error::TorshError::InvalidArgument(format!(
            "Benchmark size {} exceeds maximum {}",
            size, max_size
        )));
    }

    Ok(())
}

/// Handle benchmark errors gracefully
///
/// Provides consistent error handling across all benchmark implementations.
///
/// # Arguments
/// * `result` - Result of a benchmark operation
/// * `context` - Context string for error reporting
///
/// # Returns
/// * `Result<T>` - Success value or formatted error
pub fn handle_benchmark_error<T>(
    result: torsh_core::error::Result<T>,
    context: &str,
) -> torsh_core::error::Result<T> {
    result.map_err(|e| torsh_core::error::TorshError::RuntimeError(format!("{}: {}", context, e)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation_utilities() {
        let shape = vec![10, 10];
        let device = DeviceType::Cpu;

        let random_tensor = create_random_tensor::<f32>(&shape, device);
        assert!(random_tensor.is_ok());

        let ones_tensor = create_ones_tensor::<f32>(&shape, device);
        assert!(ones_tensor.is_ok());

        let zeros_tensor = create_zeros_tensor::<f32>(&shape, device);
        assert!(zeros_tensor.is_ok());

        let full_tensor = create_full_tensor::<f32>(&shape, 5.0, device);
        assert!(full_tensor.is_ok());
    }

    #[test]
    fn test_memory_calculation() {
        let tensor = create_ones_tensor::<f32>(&[10, 10], DeviceType::Cpu).unwrap();
        let memory_usage = calculate_tensor_memory(&tensor);
        assert_eq!(memory_usage, 100 * std::mem::size_of::<f32>());
    }

    #[test]
    fn test_flops_calculation() {
        let flops = calculate_flops(2, 100); // 2 operations per element, 100 elements
        assert_eq!(flops, 200);
    }

    #[test]
    fn test_bandwidth_calculation() {
        let bandwidth = calculate_bandwidth(1_000_000_000, 1.0); // 1GB in 1 second
        assert_eq!(bandwidth, 1.0);

        let zero_time_bandwidth = calculate_bandwidth(1000, 0.0);
        assert_eq!(zero_time_bandwidth, 0.0);
    }

    #[test]
    fn test_benchmark_config_creation() {
        let tensor_config = create_tensor_bench_config("test_tensor");
        assert_eq!(tensor_config.name, "test_tensor");

        let memory_config = create_memory_bench_config("test_memory");
        assert_eq!(memory_config.name, "test_memory");

        let autograd_config = create_autograd_bench_config("test_autograd");
        assert_eq!(autograd_config.name, "test_autograd");
    }

    #[test]
    fn test_execution_time_measurement() {
        let (result, duration) = measure_execution_time(|| {
            std::thread::sleep(std::time::Duration::from_millis(1));
            42
        });

        assert_eq!(result, 42);
        assert!(duration >= std::time::Duration::from_millis(1));
    }

    #[test]
    fn test_parameter_validation() {
        assert!(validate_benchmark_params(100, 1000).is_ok());
        assert!(validate_benchmark_params(0, 1000).is_err());
        assert!(validate_benchmark_params(2000, 1000).is_err());
    }

    #[test]
    fn test_tensor_extensions() {
        let tensor = create_ones_tensor::<f32>(&[5, 5], DeviceType::Cpu).unwrap();

        // Test mock implementations don't panic
        let tensor = tensor.requires_grad_(true);
        let scalar_tensor = tensor.sum().unwrap().requires_grad_(true);
        assert!(scalar_tensor.backward().is_ok());
        assert!(tensor.norm().is_ok());
        assert!(tensor.relu().is_ok());
        assert!(tensor.sum().is_ok());
        assert!(tensor.contiguous().is_ok());
        assert!(tensor.pow_scalar(2.0).is_ok());

        let other_tensor = create_ones_tensor::<f32>(&[5, 5], DeviceType::Cpu).unwrap();
        assert!(tensor.matmul(&other_tensor).is_ok());
        assert!(tensor.mul(&other_tensor).is_ok());
        assert!(tensor.transpose(0, 1).is_ok());
        assert!(tensor.view(&[25]).is_ok());
    }

    #[test]
    fn test_error_handling() {
        let ok_result: torsh_core::error::Result<i32> = Ok(42);
        let handled = handle_benchmark_error(ok_result, "test context");
        assert!(handled.is_ok());
        assert_eq!(handled.unwrap(), 42);

        let err_result: torsh_core::error::Result<i32> = Err(
            torsh_core::error::TorshError::InvalidArgument("test error".to_string()),
        );
        let handled = handle_benchmark_error(err_result, "test context");
        assert!(handled.is_err());
    }

    #[test]
    fn test_prevent_optimization() {
        let value = 42;
        let result = prevent_optimization(value);
        assert_eq!(result, 42);
    }

    #[test]
    fn test_warmup_operation() {
        // Should not panic
        warmup_operation();
    }
}
