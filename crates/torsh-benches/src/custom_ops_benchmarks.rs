//! Custom operations benchmarks for ToRSh
//!
//! This module provides benchmarks for user-defined custom operations,
//! including performance testing for custom kernels, domain-specific operations,
//! and extensibility features.

use crate::{BenchConfig, BenchRunner, Benchmarkable};
use criterion::black_box;
use std::time::{Duration, Instant};
use torsh_core::device::DeviceType;
use torsh_core::dtype::DType;
use torsh_tensor::{creation::*, Tensor};

/// Custom operation benchmark trait
/// Allows users to define their own operations for benchmarking
pub trait CustomOperation {
    type Input;
    type Output;

    /// Name of the custom operation
    fn name(&self) -> &str;

    /// Setup input data for the operation
    fn setup_input(&self, size: usize) -> Self::Input;

    /// Execute the custom operation
    fn execute(&self, input: &Self::Input) -> Self::Output;

    /// Estimate FLOPS for this operation
    fn estimate_flops(&self, size: usize) -> usize {
        size // Default linear complexity
    }

    /// Estimate bytes accessed
    fn estimate_bytes(&self, size: usize) -> usize {
        size * std::mem::size_of::<f32>()
    }

    /// Cleanup resources if needed
    fn cleanup(&self, _input: Self::Input, _output: Self::Output) {}
}

/// Custom operation benchmark wrapper
pub struct CustomOpBench<T: CustomOperation> {
    pub operation: T,
    pub complexity: OperationComplexity,
    pub domain: OperationDomain,
}

#[derive(Debug, Clone)]
pub enum OperationComplexity {
    Constant,       // O(1)
    Linear,         // O(n)
    Quadratic,      // O(n²)
    Cubic,          // O(n³)
    Logarithmic,    // O(log n)
    Linearithmic,   // O(n log n)
    Custom(String), // Custom complexity description
}

#[derive(Debug, Clone)]
pub enum OperationDomain {
    ComputerVision,  // Image processing operations
    NLP,             // Natural language processing
    Audio,           // Audio signal processing
    Scientific,      // Scientific computing
    Graphics,        // Graphics/rendering operations
    Cryptography,    // Cryptographic operations
    MachineLearning, // ML-specific operations
    Numerical,       // General numerical computing
    Custom(String),  // Custom domain
}

impl<T: CustomOperation> CustomOpBench<T> {
    pub fn new(operation: T, complexity: OperationComplexity, domain: OperationDomain) -> Self {
        Self {
            operation,
            complexity,
            domain,
        }
    }
}

impl<T: CustomOperation> Benchmarkable for CustomOpBench<T> {
    type Input = T::Input;
    type Output = (T::Output, CustomOpMetrics);

    fn setup(&mut self, size: usize) -> Self::Input {
        self.operation.setup_input(size)
    }

    fn run(&mut self, input: &Self::Input) -> Self::Output {
        let start_time = Instant::now();
        let result = self.operation.execute(input);
        let execution_time = start_time.elapsed();

        let metrics = CustomOpMetrics {
            execution_time_ms: execution_time.as_millis() as f64,
            operation_name: self.operation.name().to_string(),
            complexity: self.complexity.clone(),
            domain: self.domain.clone(),
            cache_misses: estimate_cache_misses(&self.complexity, &self.domain),
            memory_efficiency: calculate_memory_efficiency(&self.complexity),
            parallelization_potential: estimate_parallelization(&self.domain),
        };

        (black_box(result), metrics)
    }

    fn flops(&self, size: usize) -> usize {
        self.operation.estimate_flops(size)
    }

    fn bytes_accessed(&self, size: usize) -> usize {
        self.operation.estimate_bytes(size)
    }
}

/// Custom operation metrics
#[derive(Debug, Clone)]
pub struct CustomOpMetrics {
    pub execution_time_ms: f64,
    pub operation_name: String,
    pub complexity: OperationComplexity,
    pub domain: OperationDomain,
    pub cache_misses: f64,
    pub memory_efficiency: f64,
    pub parallelization_potential: f64,
}

// Built-in custom operations for testing

/// Fast Fourier Transform custom operation
pub struct FFTOperation {
    pub direction: FFTDirection,
    pub precision: FFTPrecision,
}

#[derive(Debug, Clone)]
pub enum FFTDirection {
    Forward,
    Inverse,
}

#[derive(Debug, Clone)]
pub enum FFTPrecision {
    Single,
    Double,
}

impl FFTOperation {
    pub fn new(direction: FFTDirection, precision: FFTPrecision) -> Self {
        Self {
            direction,
            precision,
        }
    }
}

impl CustomOperation for FFTOperation {
    type Input = Tensor<f32>;
    type Output = Tensor<f32>;

    fn name(&self) -> &str {
        match self.direction {
            FFTDirection::Forward => "fft_forward",
            FFTDirection::Inverse => "fft_inverse",
        }
    }

    fn setup_input(&self, size: usize) -> Self::Input {
        // Create complex data as interleaved real/imag
        rand::<f32>(&[size, 2]).unwrap()
    }

    fn execute(&self, input: &Self::Input) -> Self::Output {
        // Simulate FFT computation
        std::thread::sleep(Duration::from_millis(match self.direction {
            FFTDirection::Forward => 50,
            FFTDirection::Inverse => 55,
        }));

        // Mock FFT result (in practice, would call actual FFT implementation)
        let shape_obj = input.shape();
        let shape = shape_obj.dims();
        rand::<f32>(shape).unwrap()
    }

    fn estimate_flops(&self, size: usize) -> usize {
        // FFT is O(n log n)
        let log_size = (size as f64).log2() as usize;
        5 * size * log_size // 5 FLOPS per complex multiplication
    }

    fn estimate_bytes(&self, size: usize) -> usize {
        // Complex data (2 floats per element) + output
        2 * size * 2 * std::mem::size_of::<f32>()
    }
}

/// Convolution custom operation
pub struct ConvolutionOperation {
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub groups: usize,
}

impl ConvolutionOperation {
    pub fn new(kernel_size: usize, stride: usize, padding: usize, groups: usize) -> Self {
        Self {
            kernel_size,
            stride,
            padding,
            groups,
        }
    }
}

impl CustomOperation for ConvolutionOperation {
    type Input = (Tensor<f32>, Tensor<f32>); // (input, kernel)
    type Output = Tensor<f32>;

    fn name(&self) -> &str {
        "custom_convolution"
    }

    fn setup_input(&self, size: usize) -> Self::Input {
        let input = rand::<f32>(&[1, 64, size, size]).unwrap(); // NCHW format
        let kernel = rand::<f32>(&[128, 64, self.kernel_size, self.kernel_size]).unwrap();
        (input, kernel)
    }

    fn execute(&self, input: &Self::Input) -> Self::Output {
        let (input_tensor, kernel) = input;

        // Simulate convolution computation time
        let computation_time = 20 + (self.kernel_size * self.kernel_size) as u64 * 2;
        std::thread::sleep(Duration::from_millis(computation_time));

        // Calculate output size
        let shape = input_tensor.shape();
        let input_shape = shape.dims();
        let h_out = (input_shape[2] + 2 * self.padding - self.kernel_size) / self.stride + 1;
        let w_out = (input_shape[3] + 2 * self.padding - self.kernel_size) / self.stride + 1;

        rand::<f32>(&[1, 128, h_out, w_out]).unwrap()
    }

    fn estimate_flops(&self, size: usize) -> usize {
        let output_size = (size + 2 * self.padding - self.kernel_size) / self.stride + 1;
        let kernel_ops = self.kernel_size * self.kernel_size * 64; // Input channels
        output_size * output_size * kernel_ops * 128 // Output channels
    }

    fn estimate_bytes(&self, size: usize) -> usize {
        let input_bytes = size * size * 64 * std::mem::size_of::<f32>();
        let kernel_bytes =
            self.kernel_size * self.kernel_size * 64 * 128 * std::mem::size_of::<f32>();
        let output_size = (size + 2 * self.padding - self.kernel_size) / self.stride + 1;
        let output_bytes = output_size * output_size * 128 * std::mem::size_of::<f32>();
        input_bytes + kernel_bytes + output_bytes
    }
}

/// Matrix decomposition custom operation
pub struct MatrixDecompositionOperation {
    pub decomposition_type: DecompositionType,
}

#[derive(Debug, Clone)]
pub enum DecompositionType {
    LU,         // LU decomposition
    QR,         // QR decomposition
    SVD,        // Singular Value Decomposition
    Cholesky,   // Cholesky decomposition
    Eigenvalue, // Eigenvalue decomposition
}

impl MatrixDecompositionOperation {
    pub fn new(decomposition_type: DecompositionType) -> Self {
        Self { decomposition_type }
    }
}

impl CustomOperation for MatrixDecompositionOperation {
    type Input = Tensor<f32>;
    type Output = Vec<Tensor<f32>>; // Multiple output matrices

    fn name(&self) -> &str {
        match self.decomposition_type {
            DecompositionType::LU => "lu_decomposition",
            DecompositionType::QR => "qr_decomposition",
            DecompositionType::SVD => "svd_decomposition",
            DecompositionType::Cholesky => "cholesky_decomposition",
            DecompositionType::Eigenvalue => "eigenvalue_decomposition",
        }
    }

    fn setup_input(&self, size: usize) -> Self::Input {
        match self.decomposition_type {
            DecompositionType::Cholesky => {
                // Need positive definite matrix for Cholesky
                let a = rand::<f32>(&[size, size]).unwrap();
                // Simulate A^T * A to make it positive definite
                a
            }
            _ => {
                // General square matrix
                rand::<f32>(&[size, size]).unwrap()
            }
        }
    }

    fn execute(&self, input: &Self::Input) -> Self::Output {
        let shape_ref = input.shape();
        let size = shape_ref.dims()[0];

        // Simulate computation time based on decomposition type
        let computation_time = match self.decomposition_type {
            DecompositionType::LU => size * size / 10,
            DecompositionType::QR => size * size / 8,
            DecompositionType::SVD => size * size / 4, // Most expensive
            DecompositionType::Cholesky => size * size / 15, // Fastest
            DecompositionType::Eigenvalue => size * size / 5,
        };

        std::thread::sleep(Duration::from_millis(computation_time as u64));

        // Generate appropriate output matrices
        match self.decomposition_type {
            DecompositionType::LU => {
                vec![
                    rand::<f32>(&[size, size]).unwrap(), // L matrix
                    rand::<f32>(&[size, size]).unwrap(), // U matrix
                ]
            }
            DecompositionType::QR => {
                vec![
                    rand::<f32>(&[size, size]).unwrap(), // Q matrix
                    rand::<f32>(&[size, size]).unwrap(), // R matrix
                ]
            }
            DecompositionType::SVD => {
                vec![
                    rand::<f32>(&[size, size]).unwrap(), // U matrix
                    rand::<f32>(&[size]).unwrap(),       // Singular values
                    rand::<f32>(&[size, size]).unwrap(), // V^T matrix
                ]
            }
            DecompositionType::Cholesky => {
                vec![
                    rand::<f32>(&[size, size]).unwrap(), // L matrix
                ]
            }
            DecompositionType::Eigenvalue => {
                vec![
                    rand::<f32>(&[size]).unwrap(),       // Eigenvalues
                    rand::<f32>(&[size, size]).unwrap(), // Eigenvectors
                ]
            }
        }
    }

    fn estimate_flops(&self, size: usize) -> usize {
        match self.decomposition_type {
            DecompositionType::LU => (2 * size * size * size) / 3,
            DecompositionType::QR => size * size * size,
            DecompositionType::SVD => 4 * size * size * size, // Most expensive
            DecompositionType::Cholesky => size * size * size / 3, // Fastest
            DecompositionType::Eigenvalue => 2 * size * size * size,
        }
    }

    fn estimate_bytes(&self, size: usize) -> usize {
        let input_bytes = size * size * std::mem::size_of::<f32>();
        let output_matrices = match self.decomposition_type {
            DecompositionType::LU => 2,
            DecompositionType::QR => 2,
            DecompositionType::SVD => 3,
            DecompositionType::Cholesky => 1,
            DecompositionType::Eigenvalue => 2,
        };
        input_bytes + output_matrices * size * size * std::mem::size_of::<f32>()
    }
}

/// Image processing custom operation
pub struct ImageProcessingOperation {
    pub operation_type: ImageOperationType,
    pub channels: usize,
}

#[derive(Debug, Clone)]
pub enum ImageOperationType {
    GaussianBlur,
    EdgeDetection,
    Histogram,
    Morphology,
    ColorSpaceConversion,
    ImageResize,
    NoiseReduction,
    FeatureExtraction,
}

impl ImageProcessingOperation {
    pub fn new(operation_type: ImageOperationType, channels: usize) -> Self {
        Self {
            operation_type,
            channels,
        }
    }
}

impl CustomOperation for ImageProcessingOperation {
    type Input = Tensor<f32>; // Image tensor
    type Output = Tensor<f32>; // Processed image

    fn name(&self) -> &str {
        match self.operation_type {
            ImageOperationType::GaussianBlur => "gaussian_blur",
            ImageOperationType::EdgeDetection => "edge_detection",
            ImageOperationType::Histogram => "histogram",
            ImageOperationType::Morphology => "morphology",
            ImageOperationType::ColorSpaceConversion => "color_space_conversion",
            ImageOperationType::ImageResize => "image_resize",
            ImageOperationType::NoiseReduction => "noise_reduction",
            ImageOperationType::FeatureExtraction => "feature_extraction",
        }
    }

    fn setup_input(&self, size: usize) -> Self::Input {
        // Create image tensor [batch, channels, height, width]
        rand::<f32>(&[1, self.channels, size, size]).unwrap()
    }

    fn execute(&self, input: &Self::Input) -> Self::Output {
        let binding = input.shape();
        let shape = binding.dims();

        // Simulate processing time
        let processing_time = match self.operation_type {
            ImageOperationType::GaussianBlur => 15,
            ImageOperationType::EdgeDetection => 25,
            ImageOperationType::Histogram => 10,
            ImageOperationType::Morphology => 30,
            ImageOperationType::ColorSpaceConversion => 5,
            ImageOperationType::ImageResize => 20,
            ImageOperationType::NoiseReduction => 50,
            ImageOperationType::FeatureExtraction => 100,
        };

        std::thread::sleep(Duration::from_millis(processing_time));

        // Generate output based on operation type
        match self.operation_type {
            ImageOperationType::Histogram => {
                // Histogram output is different shape
                rand::<f32>(&[1, self.channels, 256]).unwrap()
            }
            ImageOperationType::ImageResize => {
                // Resize to half the original size
                rand::<f32>(&[shape[0], shape[1], shape[2] / 2, shape[3] / 2]).unwrap()
            }
            ImageOperationType::FeatureExtraction => {
                // Feature maps
                rand::<f32>(&[shape[0], 512, shape[2] / 4, shape[3] / 4]).unwrap()
            }
            _ => {
                // Same shape as input
                rand::<f32>(shape).unwrap()
            }
        }
    }

    fn estimate_flops(&self, size: usize) -> usize {
        let pixels = size * size * self.channels;
        match self.operation_type {
            ImageOperationType::GaussianBlur => pixels * 25, // 5x5 kernel
            ImageOperationType::EdgeDetection => pixels * 9, // 3x3 kernel
            ImageOperationType::Histogram => pixels,         // Linear scan
            ImageOperationType::Morphology => pixels * 9,    // 3x3 structuring element
            ImageOperationType::ColorSpaceConversion => pixels * 10, // Matrix multiplication
            ImageOperationType::ImageResize => pixels * 4,   // Bilinear interpolation
            ImageOperationType::NoiseReduction => pixels * 50, // Complex filtering
            ImageOperationType::FeatureExtraction => pixels * 100, // Deep features
        }
    }

    fn estimate_bytes(&self, size: usize) -> usize {
        let input_bytes = size * size * self.channels * std::mem::size_of::<f32>();
        let output_multiplier = match self.operation_type {
            ImageOperationType::Histogram => 256 / (size * size), // Much smaller
            ImageOperationType::ImageResize => 1,                 // Similar size
            ImageOperationType::FeatureExtraction => 32,          // Much larger
            _ => 1,                                               // Same size
        };
        input_bytes + input_bytes * output_multiplier
    }
}

/// Scientific computing custom operation
pub struct ScientificOperation {
    pub operation_type: ScientificOperationType,
    pub precision: ScientificPrecision,
}

#[derive(Debug, Clone)]
pub enum ScientificOperationType {
    ODESolver,     // Ordinary Differential Equation solving
    PDESolver,     // Partial Differential Equation solving
    MonteCarlo,    // Monte Carlo simulation
    Optimization,  // Numerical optimization
    Integration,   // Numerical integration
    RootFinding,   // Root finding algorithms
    Interpolation, // Data interpolation
    Regression,    // Statistical regression
}

#[derive(Debug, Clone)]
pub enum ScientificPrecision {
    Single,    // f32
    Double,    // f64
    Extended,  // Extended precision
    Arbitrary, // Arbitrary precision
}

impl ScientificOperation {
    pub fn new(operation_type: ScientificOperationType, precision: ScientificPrecision) -> Self {
        Self {
            operation_type,
            precision,
        }
    }
}

impl CustomOperation for ScientificOperation {
    type Input = (Tensor<f32>, ScientificParams);
    type Output = (Tensor<f32>, ScientificResults);

    fn name(&self) -> &str {
        match self.operation_type {
            ScientificOperationType::ODESolver => "ode_solver",
            ScientificOperationType::PDESolver => "pde_solver",
            ScientificOperationType::MonteCarlo => "monte_carlo",
            ScientificOperationType::Optimization => "optimization",
            ScientificOperationType::Integration => "integration",
            ScientificOperationType::RootFinding => "root_finding",
            ScientificOperationType::Interpolation => "interpolation",
            ScientificOperationType::Regression => "regression",
        }
    }

    fn setup_input(&self, size: usize) -> Self::Input {
        let data = match self.operation_type {
            ScientificOperationType::ODESolver => rand::<f32>(&[size]), // Initial conditions
            ScientificOperationType::PDESolver => rand::<f32>(&[size, size]), // Grid data
            ScientificOperationType::MonteCarlo => rand::<f32>(&[size]), // Random samples
            ScientificOperationType::Optimization => rand::<f32>(&[size]), // Parameter vector
            ScientificOperationType::Integration => rand::<f32>(&[size]), // Function values
            ScientificOperationType::RootFinding => rand::<f32>(&[size]), // Polynomial coefficients
            ScientificOperationType::Interpolation => rand::<f32>(&[size, 2]), // (x, y) pairs
            ScientificOperationType::Regression => rand::<f32>(&[size, 2]), // (x, y) data
        };

        let params = ScientificParams {
            tolerance: 1e-6,
            max_iterations: 1000,
            step_size: 0.01,
            convergence_criteria: 1e-8,
        };

        (data.unwrap(), params)
    }

    fn execute(&self, input: &Self::Input) -> Self::Output {
        let (data, _params) = input;
        let size = data.numel();

        // Simulate computation time based on operation complexity
        let computation_time = match self.operation_type {
            ScientificOperationType::ODESolver => size / 10 + 50, // Iterative
            ScientificOperationType::PDESolver => size / 5 + 100, // Grid-based
            ScientificOperationType::MonteCarlo => size / 100 + 20, // Embarrassingly parallel
            ScientificOperationType::Optimization => size * 2 + 30, // Gradient-based
            ScientificOperationType::Integration => size / 20 + 10, // Quadrature
            ScientificOperationType::RootFinding => size * 3 + 15, // Newton's method
            ScientificOperationType::Interpolation => size / 50 + 5, // Spline fitting
            ScientificOperationType::Regression => size / 10 + 25, // Least squares
        };

        std::thread::sleep(Duration::from_millis(computation_time as u64));

        // Generate appropriate results
        let result_data = match self.operation_type {
            ScientificOperationType::ODESolver => rand::<f32>(&[size]).unwrap(), // Solution trajectory
            ScientificOperationType::PDESolver => data.clone(),                  // Updated grid
            ScientificOperationType::MonteCarlo => rand::<f32>(&[1]).unwrap(),   // Estimated value
            ScientificOperationType::Optimization => rand::<f32>(&[size]).unwrap(), // Optimal parameters
            ScientificOperationType::Integration => rand::<f32>(&[1]).unwrap(), // Integral value
            ScientificOperationType::RootFinding => rand::<f32>(&[size - 1]).unwrap(), // Roots
            ScientificOperationType::Interpolation => rand::<f32>(&[size * 2]).unwrap(), // Interpolated values
            ScientificOperationType::Regression => rand::<f32>(&[2]).unwrap(), // Coefficients
        };

        let results = ScientificResults {
            converged: true,
            iterations: 100,
            final_error: 1e-8,
            computational_cost: computation_time as f64,
        };

        (result_data, results)
    }

    fn estimate_flops(&self, size: usize) -> usize {
        match self.operation_type {
            ScientificOperationType::ODESolver => size * 1000, // Iterative solving
            ScientificOperationType::PDESolver => size * size * 10, // Grid operations
            ScientificOperationType::MonteCarlo => size * 10,  // Simple operations
            ScientificOperationType::Optimization => size * 500, // Gradient computation
            ScientificOperationType::Integration => size * 5,  // Function evaluations
            ScientificOperationType::RootFinding => size * 100, // Newton iterations
            ScientificOperationType::Interpolation => size * 50, // Spline operations
            ScientificOperationType::Regression => size * 20,  // Matrix operations
        }
    }

    fn estimate_bytes(&self, size: usize) -> usize {
        let base_bytes = size * std::mem::size_of::<f32>();
        let precision_multiplier = match self.precision {
            ScientificPrecision::Single => 1,
            ScientificPrecision::Double => 2,
            ScientificPrecision::Extended => 3,
            ScientificPrecision::Arbitrary => 4,
        };
        base_bytes * precision_multiplier * 2 // Input + output
    }
}

/// Scientific computation parameters
#[derive(Debug, Clone)]
pub struct ScientificParams {
    pub tolerance: f64,
    pub max_iterations: usize,
    pub step_size: f64,
    pub convergence_criteria: f64,
}

/// Scientific computation results
#[derive(Debug, Clone)]
pub struct ScientificResults {
    pub converged: bool,
    pub iterations: usize,
    pub final_error: f64,
    pub computational_cost: f64,
}

// Helper functions

fn estimate_cache_misses(complexity: &OperationComplexity, domain: &OperationDomain) -> f64 {
    let base_misses = match complexity {
        OperationComplexity::Constant => 0.01,
        OperationComplexity::Linear => 0.05,
        OperationComplexity::Quadratic => 0.15,
        OperationComplexity::Cubic => 0.25,
        OperationComplexity::Logarithmic => 0.02,
        OperationComplexity::Linearithmic => 0.08,
        OperationComplexity::Custom(_) => 0.10,
    };

    let domain_factor = match domain {
        OperationDomain::ComputerVision => 1.2, // Image data has poor locality
        OperationDomain::Audio => 0.8,          // Sequential access
        OperationDomain::Scientific => 1.1,     // Complex access patterns
        OperationDomain::Graphics => 1.3,       // Random texture access
        OperationDomain::Cryptography => 0.9,   // Small data, good locality
        OperationDomain::MachineLearning => 1.0, // Mixed patterns
        OperationDomain::Numerical => 0.9,      // Regular patterns
        OperationDomain::NLP => 0.7,            // Sequential text processing
        OperationDomain::Custom(_) => 1.0,
    };

    base_misses * domain_factor
}

fn calculate_memory_efficiency(complexity: &OperationComplexity) -> f64 {
    match complexity {
        OperationComplexity::Constant => 0.95,     // Very efficient
        OperationComplexity::Linear => 0.90,       // Good efficiency
        OperationComplexity::Quadratic => 0.70,    // Moderate efficiency
        OperationComplexity::Cubic => 0.50,        // Poor efficiency
        OperationComplexity::Logarithmic => 0.98,  // Excellent efficiency
        OperationComplexity::Linearithmic => 0.85, // Good efficiency
        OperationComplexity::Custom(_) => 0.75,    // Unknown, assume moderate
    }
}

fn estimate_parallelization(domain: &OperationDomain) -> f64 {
    match domain {
        OperationDomain::ComputerVision => 0.90, // Highly parallelizable
        OperationDomain::Audio => 0.60,          // Some dependencies
        OperationDomain::Scientific => 0.80,     // Often parallelizable
        OperationDomain::Graphics => 0.95,       // Embarrassingly parallel
        OperationDomain::Cryptography => 0.30,   // Often sequential
        OperationDomain::MachineLearning => 0.85, // Matrix ops are parallel
        OperationDomain::Numerical => 0.75,      // Linear algebra is parallel
        OperationDomain::NLP => 0.40,            // Sequential dependencies
        OperationDomain::Custom(_) => 0.50,      // Unknown, assume moderate
    }
}

/// User-defined operation benchmark
/// Allows external users to benchmark their own operations
pub struct UserDefinedBench {
    pub operation_name: String,
    pub setup_fn: Box<dyn Fn(usize) -> Vec<Tensor<f32>>>,
    pub execute_fn: Box<dyn Fn(&[Tensor<f32>]) -> Vec<Tensor<f32>>>,
    pub flops_fn: Box<dyn Fn(usize) -> usize>,
    pub bytes_fn: Box<dyn Fn(usize) -> usize>,
}

impl UserDefinedBench {
    pub fn new<S, E, F, B>(name: String, setup: S, execute: E, flops: F, bytes: B) -> Self
    where
        S: Fn(usize) -> Vec<Tensor<f32>> + 'static,
        E: Fn(&[Tensor<f32>]) -> Vec<Tensor<f32>> + 'static,
        F: Fn(usize) -> usize + 'static,
        B: Fn(usize) -> usize + 'static,
    {
        Self {
            operation_name: name,
            setup_fn: Box::new(setup),
            execute_fn: Box::new(execute),
            flops_fn: Box::new(flops),
            bytes_fn: Box::new(bytes),
        }
    }
}

impl Benchmarkable for UserDefinedBench {
    type Input = Vec<Tensor<f32>>;
    type Output = (Vec<Tensor<f32>>, Duration);

    fn setup(&mut self, size: usize) -> Self::Input {
        (self.setup_fn)(size)
    }

    fn run(&mut self, input: &Self::Input) -> Self::Output {
        let start_time = Instant::now();
        let result = (self.execute_fn)(input);
        let execution_time = start_time.elapsed();
        (black_box(result), execution_time)
    }

    fn flops(&self, size: usize) -> usize {
        (self.flops_fn)(size)
    }

    fn bytes_accessed(&self, size: usize) -> usize {
        (self.bytes_fn)(size)
    }
}

/// Comprehensive custom operations benchmark suite
pub fn run_custom_ops_benchmarks() {
    let mut runner = BenchRunner::new();

    // FFT benchmarks
    let fft_directions = vec![FFTDirection::Forward, FFTDirection::Inverse];
    let fft_precisions = vec![FFTPrecision::Single, FFTPrecision::Double];

    for direction in &fft_directions {
        for precision in &fft_precisions {
            let fft_op = FFTOperation::new(direction.clone(), precision.clone());
            let config_name = format!("fft_{:?}_{:?}", direction, precision).to_lowercase();
            let config = BenchConfig::new(&config_name)
                .with_sizes(vec![64, 128, 256, 512, 1024])
                .with_dtypes(vec![DType::F32])
                .with_metadata("benchmark_type", "custom_fft");

            let bench = CustomOpBench::new(
                fft_op,
                OperationComplexity::Linearithmic,
                OperationDomain::Scientific,
            );
            runner.run_benchmark(bench, &config);
        }
    }

    // Convolution benchmarks
    let kernel_sizes = vec![3, 5, 7];
    for &kernel_size in &kernel_sizes {
        let conv_op = ConvolutionOperation::new(kernel_size, 1, 0, 1);
        let config_name = format!("convolution_k{}", kernel_size);
        let config = BenchConfig::new(&config_name)
            .with_sizes(vec![32, 64, 128, 256])
            .with_dtypes(vec![DType::F32])
            .with_metadata("benchmark_type", "custom_convolution")
            .with_metadata("kernel_size", &kernel_size.to_string());

        let bench = CustomOpBench::new(
            conv_op,
            OperationComplexity::Cubic,
            OperationDomain::ComputerVision,
        );
        runner.run_benchmark(bench, &config);
    }

    // Matrix decomposition benchmarks
    let decomposition_types = vec![
        DecompositionType::LU,
        DecompositionType::QR,
        DecompositionType::SVD,
        DecompositionType::Cholesky,
        DecompositionType::Eigenvalue,
    ];

    for decomp_type in &decomposition_types {
        let decomp_op = MatrixDecompositionOperation::new(decomp_type.clone());
        let config_name = format!("matrix_decomp_{:?}", decomp_type).to_lowercase();
        let config = BenchConfig::new(&config_name)
            .with_sizes(vec![64, 128, 256, 512])
            .with_dtypes(vec![DType::F32])
            .with_metadata("benchmark_type", "matrix_decomposition")
            .with_metadata("decomposition_type", &format!("{:?}", decomp_type));

        let bench = CustomOpBench::new(
            decomp_op,
            OperationComplexity::Cubic,
            OperationDomain::Numerical,
        );
        runner.run_benchmark(bench, &config);
    }

    // Image processing benchmarks
    let image_ops = vec![
        ImageOperationType::GaussianBlur,
        ImageOperationType::EdgeDetection,
        ImageOperationType::Histogram,
        ImageOperationType::Morphology,
        ImageOperationType::ColorSpaceConversion,
        ImageOperationType::ImageResize,
        ImageOperationType::NoiseReduction,
        ImageOperationType::FeatureExtraction,
    ];

    for image_op in &image_ops {
        let img_op = ImageProcessingOperation::new(image_op.clone(), 3); // RGB channels
        let config_name = format!("image_processing_{:?}", image_op).to_lowercase();
        let config = BenchConfig::new(&config_name)
            .with_sizes(vec![128, 256, 512, 1024])
            .with_dtypes(vec![DType::F32])
            .with_metadata("benchmark_type", "image_processing")
            .with_metadata("operation_type", &format!("{:?}", image_op));

        let complexity = match image_op {
            ImageOperationType::Histogram => OperationComplexity::Linear,
            ImageOperationType::FeatureExtraction => OperationComplexity::Cubic,
            _ => OperationComplexity::Quadratic,
        };

        let bench = CustomOpBench::new(img_op, complexity, OperationDomain::ComputerVision);
        runner.run_benchmark(bench, &config);
    }

    // Scientific computing benchmarks
    let scientific_ops = vec![
        ScientificOperationType::ODESolver,
        ScientificOperationType::PDESolver,
        ScientificOperationType::MonteCarlo,
        ScientificOperationType::Optimization,
        ScientificOperationType::Integration,
        ScientificOperationType::RootFinding,
        ScientificOperationType::Interpolation,
        ScientificOperationType::Regression,
    ];

    for sci_op in &scientific_ops {
        let scientific_op = ScientificOperation::new(sci_op.clone(), ScientificPrecision::Single);
        let config_name = format!("scientific_{:?}", sci_op).to_lowercase();
        let config = BenchConfig::new(&config_name)
            .with_sizes(vec![100, 500, 1000, 5000])
            .with_dtypes(vec![DType::F32])
            .with_metadata("benchmark_type", "scientific_computing")
            .with_metadata("operation_type", &format!("{:?}", sci_op));

        let complexity = match sci_op {
            ScientificOperationType::PDESolver => OperationComplexity::Cubic,
            ScientificOperationType::MonteCarlo => OperationComplexity::Linear,
            _ => OperationComplexity::Quadratic,
        };

        let bench = CustomOpBench::new(scientific_op, complexity, OperationDomain::Scientific);
        runner.run_benchmark(bench, &config);
    }

    // Generate custom operations report
    runner.generate_report("target/custom_ops_reports").unwrap();
    runner.export_csv("target/custom_ops_results.csv").unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fft_operation() {
        let fft_op = FFTOperation::new(FFTDirection::Forward, FFTPrecision::Single);
        let input = fft_op.setup_input(64);
        let result = fft_op.execute(&input);

        let result_shape = result.shape();
        assert_eq!(result_shape.dims()[0], 64);
        assert_eq!(result_shape.dims()[1], 2); // Real + imaginary
    }

    #[test]
    fn test_convolution_operation() {
        let conv_op = ConvolutionOperation::new(3, 1, 0, 1);
        let input = conv_op.setup_input(32);
        let result = conv_op.execute(&input);

        // Check output dimensions
        let binding = result.shape();
        let output_shape = binding.dims();
        assert_eq!(output_shape[0], 1); // Batch size
        assert_eq!(output_shape[1], 128); // Output channels
        assert_eq!(output_shape[2], 30); // Height: 32 - 3 + 1
        assert_eq!(output_shape[3], 30); // Width: 32 - 3 + 1
    }

    #[test]
    fn test_matrix_decomposition() {
        let decomp_op = MatrixDecompositionOperation::new(DecompositionType::LU);
        let input = decomp_op.setup_input(64);
        let result = decomp_op.execute(&input);

        assert_eq!(result.len(), 2); // L and U matrices
        assert_eq!(result[0].shape().dims()[0], 64);
        assert_eq!(result[1].shape().dims()[0], 64);
    }

    #[test]
    fn test_image_processing() {
        let img_op = ImageProcessingOperation::new(ImageOperationType::GaussianBlur, 3);
        let input = img_op.setup_input(128);
        let result = img_op.execute(&input);

        // Output should have same shape as input for blur
        let input_shape = input.shape();
        let result_shape = result.shape();
        assert_eq!(input_shape.dims(), result_shape.dims());
    }

    #[test]
    fn test_scientific_operation() {
        let sci_op = ScientificOperation::new(
            ScientificOperationType::Integration,
            ScientificPrecision::Single,
        );
        let input = sci_op.setup_input(100);
        let (result_data, results) = sci_op.execute(&input);

        let result_data_shape = result_data.shape();
        assert_eq!(result_data_shape.dims()[0], 1); // Single integral value
        assert!(results.computational_cost > 0.0);
    }

    #[test]
    fn test_custom_op_bench() {
        let fft_op = FFTOperation::new(FFTDirection::Forward, FFTPrecision::Single);
        let mut bench = CustomOpBench::new(
            fft_op,
            OperationComplexity::Linearithmic,
            OperationDomain::Scientific,
        );

        let input = bench.setup(64);
        let (_result, metrics) = bench.run(&input);

        assert!(metrics.execution_time_ms > 0.0);
        assert_eq!(metrics.operation_name, "fft_forward");
        assert!(metrics.parallelization_potential > 0.0);
    }

    #[test]
    fn test_flops_estimation() {
        let fft_op = FFTOperation::new(FFTDirection::Forward, FFTPrecision::Single);
        let flops = fft_op.estimate_flops(1024);

        // FFT should be O(n log n)
        let expected_flops = 5 * 1024 * 10; // 5 FLOPS * n * log2(n)
        assert_eq!(flops, expected_flops);
    }

    #[test]
    fn test_user_defined_bench() {
        let mut user_bench = UserDefinedBench::new(
            "test_operation".to_string(),
            |size| vec![rand::<f32>(&[size, size]).unwrap()],
            |inputs| vec![inputs[0].clone()],
            |size| size * size,
            |size| size * size * std::mem::size_of::<f32>(),
        );

        let input = user_bench.setup(64);
        let (result, duration) = user_bench.run(&input);

        assert_eq!(result.len(), 1);
        assert!(duration.as_nanos() > 0);
    }
}
