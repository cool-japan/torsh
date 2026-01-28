//! BLAS Integration for Efficient Linear Algebra Gradients
//!
//! This module provides integration with BLAS (Basic Linear Algebra Subprograms) libraries
//! for optimized gradient computation in linear algebra operations. It supports multiple
//! BLAS implementations and provides fallback mechanisms for environments without BLAS.

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use crate::error_handling::{AutogradError, AutogradResult};
use scirs2_core::ndarray::{Array, ArrayView, ArrayViewMut, Ix1, Ix2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::sync::{Mutex, RwLock};

/// Supported BLAS implementations
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BlasImplementation {
    /// Intel Math Kernel Library
    MKL,
    /// OpenBLAS
    OpenBLAS,
    /// Apple's Accelerate framework
    Accelerate,
    /// ATLAS (Automatically Tuned Linear Algebra Software)
    ATLAS,
    /// Generic CBLAS interface
    CBLAS,
    /// Fallback pure Rust implementation
    PureRust,
    /// Custom user-provided implementation
    Custom(String),
}

impl fmt::Display for BlasImplementation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BlasImplementation::MKL => write!(f, "Intel MKL"),
            BlasImplementation::OpenBLAS => write!(f, "OpenBLAS"),
            BlasImplementation::Accelerate => write!(f, "Apple Accelerate"),
            BlasImplementation::ATLAS => write!(f, "ATLAS"),
            BlasImplementation::CBLAS => write!(f, "CBLAS"),
            BlasImplementation::PureRust => write!(f, "Pure Rust"),
            BlasImplementation::Custom(name) => write!(f, "Custom({})", name),
        }
    }
}

/// BLAS operation types supported for gradient computation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BlasOperation {
    /// General matrix-matrix multiplication (GEMM)
    GEMM,
    /// General matrix-vector multiplication (GEMV)
    GEMV,
    /// Vector-vector dot product (DOT)
    DOT,
    /// Vector scaling (SCAL)
    SCAL,
    /// Vector addition (AXPY)
    AXPY,
    /// Matrix-vector solve (solving Ax = b)
    SOLVE,
    /// Eigenvalue decomposition
    EIGEN,
    /// Singular Value Decomposition
    SVD,
    /// Cholesky decomposition
    CHOLESKY,
    /// QR decomposition
    QR,
}

impl fmt::Display for BlasOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BlasOperation::GEMM => write!(f, "GEMM (Matrix-Matrix Multiply)"),
            BlasOperation::GEMV => write!(f, "GEMV (Matrix-Vector Multiply)"),
            BlasOperation::DOT => write!(f, "DOT (Vector Dot Product)"),
            BlasOperation::SCAL => write!(f, "SCAL (Vector Scaling)"),
            BlasOperation::AXPY => write!(f, "AXPY (Vector Addition)"),
            BlasOperation::SOLVE => write!(f, "SOLVE (Linear System)"),
            BlasOperation::EIGEN => write!(f, "EIGEN (Eigenvalue Decomposition)"),
            BlasOperation::SVD => write!(f, "SVD (Singular Value Decomposition)"),
            BlasOperation::CHOLESKY => write!(f, "CHOLESKY (Cholesky Decomposition)"),
            BlasOperation::QR => write!(f, "QR (QR Decomposition)"),
        }
    }
}

/// Configuration for BLAS integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlasConfig {
    pub preferred_implementation: BlasImplementation,
    pub fallback_implementations: Vec<BlasImplementation>,
    pub enable_threading: bool,
    pub num_threads: Option<usize>,
    pub cache_size_limit: usize, // in bytes
    pub enable_benchmarking: bool,
    pub performance_thresholds: HashMap<BlasOperation, f64>, // minimum performance gain to use BLAS
}

impl Default for BlasConfig {
    fn default() -> Self {
        let mut performance_thresholds = HashMap::new();
        performance_thresholds.insert(BlasOperation::GEMM, 2.0); // 2x speedup minimum
        performance_thresholds.insert(BlasOperation::GEMV, 1.5);
        performance_thresholds.insert(BlasOperation::DOT, 1.2);
        performance_thresholds.insert(BlasOperation::SCAL, 1.1);
        performance_thresholds.insert(BlasOperation::AXPY, 1.3);

        Self {
            preferred_implementation: Self::detect_best_implementation(),
            fallback_implementations: vec![
                BlasImplementation::OpenBLAS,
                BlasImplementation::CBLAS,
                BlasImplementation::PureRust,
            ],
            enable_threading: true,
            num_threads: None,                  // Use system default
            cache_size_limit: 64 * 1024 * 1024, // 64MB
            enable_benchmarking: true,
            performance_thresholds,
        }
    }
}

impl BlasConfig {
    /// Detect the best available BLAS implementation
    pub fn detect_best_implementation() -> BlasImplementation {
        // Priority order for detection
        let implementations = [
            (
                BlasImplementation::MKL,
                Self::check_mkl_available as fn() -> bool,
            ),
            (
                BlasImplementation::Accelerate,
                Self::check_accelerate_available as fn() -> bool,
            ),
            (
                BlasImplementation::OpenBLAS,
                Self::check_openblas_available as fn() -> bool,
            ),
            (
                BlasImplementation::ATLAS,
                Self::check_atlas_available as fn() -> bool,
            ),
            (
                BlasImplementation::CBLAS,
                Self::check_cblas_available as fn() -> bool,
            ),
        ];

        for (implementation, checker) in &implementations {
            if checker() {
                return implementation.clone();
            }
        }

        BlasImplementation::PureRust // Fallback
    }

    fn check_mkl_available() -> bool {
        // Check for Intel MKL availability
        // In practice, this would check for MKL library presence
        std::env::var("MKLROOT").is_ok() || Self::library_exists("libmkl_core")
    }

    fn check_accelerate_available() -> bool {
        // Check for Apple Accelerate framework
        cfg!(target_os = "macos") || cfg!(target_os = "ios")
    }

    fn check_openblas_available() -> bool {
        // Check for OpenBLAS availability
        Self::library_exists("libopenblas") || Self::library_exists("libblas")
    }

    fn check_atlas_available() -> bool {
        // Check for ATLAS availability
        Self::library_exists("libatlas") || Self::library_exists("libcblas")
    }

    fn check_cblas_available() -> bool {
        // Check for generic CBLAS
        Self::library_exists("libcblas") || Self::library_exists("libblas")
    }

    fn library_exists(_library_name: &str) -> bool {
        // Simplified library detection - in practice would use proper library detection
        // This is a placeholder for actual library detection logic
        false
    }
}

/// Trait for BLAS implementations
pub trait BlasProvider: Send + Sync + std::fmt::Debug {
    fn implementation(&self) -> BlasImplementation;
    fn is_available(&self) -> bool;
    fn initialize(&mut self) -> AutogradResult<()>;
    fn shutdown(&mut self) -> AutogradResult<()>;

    // Level 1 BLAS operations (vector-vector)
    fn dot(&self, x: &ArrayView<f64, Ix1>, y: &ArrayView<f64, Ix1>) -> AutogradResult<f64>;
    fn scal(&self, alpha: f64, x: &mut ArrayViewMut<f64, Ix1>) -> AutogradResult<()>;
    fn axpy(
        &self,
        alpha: f64,
        x: &ArrayView<f64, Ix1>,
        y: &mut ArrayViewMut<f64, Ix1>,
    ) -> AutogradResult<()>;

    // Level 2 BLAS operations (matrix-vector)
    fn gemv(
        &self,
        alpha: f64,
        a: &ArrayView<f64, Ix2>,
        x: &ArrayView<f64, Ix1>,
        beta: f64,
        y: &mut ArrayViewMut<f64, Ix1>,
        transpose: bool,
    ) -> AutogradResult<()>;

    // Level 3 BLAS operations (matrix-matrix)
    fn gemm(
        &self,
        alpha: f64,
        a: &ArrayView<f64, Ix2>,
        b: &ArrayView<f64, Ix2>,
        beta: f64,
        c: &mut ArrayViewMut<f64, Ix2>,
        transpose_a: bool,
        transpose_b: bool,
    ) -> AutogradResult<()>;

    // Advanced linear algebra operations for gradients
    fn solve_triangular(
        &self,
        a: &ArrayView<f64, Ix2>,
        b: &mut ArrayViewMut<f64, Ix2>,
        upper: bool,
    ) -> AutogradResult<()>;
    fn cholesky(&self, a: &ArrayView<f64, Ix2>) -> AutogradResult<Array<f64, Ix2>>;
    fn qr_decomposition(
        &self,
        a: &ArrayView<f64, Ix2>,
    ) -> AutogradResult<(Array<f64, Ix2>, Array<f64, Ix2>)>;
    fn svd(
        &self,
        a: &ArrayView<f64, Ix2>,
    ) -> AutogradResult<(Array<f64, Ix2>, Array<f64, Ix1>, Array<f64, Ix2>)>;

    // Performance characteristics
    fn benchmark_operation(&self, operation: BlasOperation, size: usize) -> AutogradResult<f64>;
    fn get_optimal_block_size(&self, operation: BlasOperation) -> usize;
    fn supports_threading(&self) -> bool;
    fn set_num_threads(&mut self, threads: usize) -> AutogradResult<()>;
}

/// Pure Rust implementation as fallback
#[derive(Debug)]
pub struct PureRustBlasProvider {
    num_threads: usize,
}

impl PureRustBlasProvider {
    pub fn new() -> Self {
        Self {
            num_threads: num_cpus::get(),
        }
    }
}

impl BlasProvider for PureRustBlasProvider {
    fn implementation(&self) -> BlasImplementation {
        BlasImplementation::PureRust
    }

    fn is_available(&self) -> bool {
        true // Always available
    }

    fn initialize(&mut self) -> AutogradResult<()> {
        tracing::debug!(
            "Initialized Pure Rust BLAS provider with {} threads",
            self.num_threads
        );
        Ok(())
    }

    fn shutdown(&mut self) -> AutogradResult<()> {
        tracing::debug!("Shutdown Pure Rust BLAS provider");
        Ok(())
    }

    fn dot(&self, x: &ArrayView<f64, Ix1>, y: &ArrayView<f64, Ix1>) -> AutogradResult<f64> {
        if x.len() != y.len() {
            return Err(AutogradError::gradient_computation(
                "blas_dot",
                format!("Vector dimension mismatch: {} vs {}", x.len(), y.len()),
            ));
        }

        let result = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        Ok(result)
    }

    fn scal(&self, alpha: f64, x: &mut ArrayViewMut<f64, Ix1>) -> AutogradResult<()> {
        x.mapv_inplace(|val| val * alpha);
        Ok(())
    }

    fn axpy(
        &self,
        alpha: f64,
        x: &ArrayView<f64, Ix1>,
        y: &mut ArrayViewMut<f64, Ix1>,
    ) -> AutogradResult<()> {
        if x.len() != y.len() {
            return Err(AutogradError::gradient_computation(
                "blas_dot",
                format!("Vector dimension mismatch: {} vs {}", x.len(), y.len()),
            ));
        }

        for (x_val, y_val) in x.iter().zip(y.iter_mut()) {
            *y_val += alpha * x_val;
        }
        Ok(())
    }

    fn gemv(
        &self,
        alpha: f64,
        a: &ArrayView<f64, Ix2>,
        x: &ArrayView<f64, Ix1>,
        beta: f64,
        y: &mut ArrayViewMut<f64, Ix1>,
        transpose: bool,
    ) -> AutogradResult<()> {
        let (rows, cols) = if transpose {
            (a.ncols(), a.nrows())
        } else {
            (a.nrows(), a.ncols())
        };

        if x.len() != cols || y.len() != rows {
            return Err(AutogradError::gradient_computation(
                "blas_dot",
                format!(
                    "Matrix-vector dimension mismatch: matrix {}x{}, x {}, y {}",
                    rows,
                    cols,
                    x.len(),
                    y.len()
                ),
            ));
        }

        // Scale y by beta
        if beta != 1.0 {
            y.mapv_inplace(|val| val * beta);
        }

        // Compute alpha * A * x (or alpha * A^T * x) + y
        if transpose {
            for (col_idx, &x_val) in x.iter().enumerate() {
                let scaled_x = alpha * x_val;
                for (row_idx, y_val) in y.iter_mut().enumerate() {
                    *y_val += scaled_x * a[[row_idx, col_idx]];
                }
            }
        } else {
            for (row_idx, y_val) in y.iter_mut().enumerate() {
                let mut sum = 0.0;
                for (col_idx, &x_val) in x.iter().enumerate() {
                    sum += a[[row_idx, col_idx]] * x_val;
                }
                *y_val += alpha * sum;
            }
        }

        Ok(())
    }

    fn gemm(
        &self,
        alpha: f64,
        a: &ArrayView<f64, Ix2>,
        b: &ArrayView<f64, Ix2>,
        beta: f64,
        c: &mut ArrayViewMut<f64, Ix2>,
        transpose_a: bool,
        transpose_b: bool,
    ) -> AutogradResult<()> {
        let (a_rows, a_cols) = if transpose_a {
            (a.ncols(), a.nrows())
        } else {
            (a.nrows(), a.ncols())
        };

        let (b_rows, b_cols) = if transpose_b {
            (b.ncols(), b.nrows())
        } else {
            (b.nrows(), b.ncols())
        };

        if a_cols != b_rows || c.nrows() != a_rows || c.ncols() != b_cols {
            return Err(AutogradError::gradient_computation(
                "blas_dot",
                format!(
                    "Matrix dimension mismatch: A {}x{}, B {}x{}, C {}x{}",
                    a_rows,
                    a_cols,
                    b_rows,
                    b_cols,
                    c.nrows(),
                    c.ncols()
                ),
            ));
        }

        // Scale C by beta
        if beta != 1.0 {
            c.mapv_inplace(|val| val * beta);
        }

        // Compute alpha * A * B + C (with optional transposes)
        for i in 0..a_rows {
            for j in 0..b_cols {
                let mut sum = 0.0;
                for k in 0..a_cols {
                    let a_val = if transpose_a { a[[k, i]] } else { a[[i, k]] };
                    let b_val = if transpose_b { b[[j, k]] } else { b[[k, j]] };
                    sum += a_val * b_val;
                }
                c[[i, j]] += alpha * sum;
            }
        }

        Ok(())
    }

    fn solve_triangular(
        &self,
        a: &ArrayView<f64, Ix2>,
        b: &mut ArrayViewMut<f64, Ix2>,
        upper: bool,
    ) -> AutogradResult<()> {
        let n = a.nrows();
        if a.ncols() != n || b.nrows() != n {
            return Err(AutogradError::gradient_computation(
                "blas_dot",
                "Matrix dimensions incompatible for triangular solve".to_string(),
            ));
        }

        let nrhs = b.ncols();

        if upper {
            // Upper triangular solve (backward substitution)
            for col in 0..nrhs {
                for i in (0..n).rev() {
                    let mut sum = b[[i, col]];
                    for j in (i + 1)..n {
                        sum -= a[[i, j]] * b[[j, col]];
                    }
                    b[[i, col]] = sum / a[[i, i]];
                }
            }
        } else {
            // Lower triangular solve (forward substitution)
            for col in 0..nrhs {
                for i in 0..n {
                    let mut sum = b[[i, col]];
                    for j in 0..i {
                        sum -= a[[i, j]] * b[[j, col]];
                    }
                    b[[i, col]] = sum / a[[i, i]];
                }
            }
        }

        Ok(())
    }

    fn cholesky(&self, a: &ArrayView<f64, Ix2>) -> AutogradResult<Array<f64, Ix2>> {
        let n = a.nrows();
        if a.ncols() != n {
            return Err(AutogradError::gradient_computation(
                "blas_dot",
                "Matrix must be square for Cholesky decomposition".to_string(),
            ));
        }

        let mut l = Array::zeros((n, n));

        for i in 0..n {
            for j in 0..=i {
                if i == j {
                    // Diagonal elements
                    let mut sum = 0.0;
                    for k in 0..j {
                        sum += l[[i, k]] * l[[i, k]];
                    }
                    let val = a[[i, i]] - sum;
                    if val <= 0.0 {
                        return Err(AutogradError::gradient_computation(
                            "blas_dot",
                            "Matrix is not positive definite".to_string(),
                        ));
                    }
                    l[[i, j]] = val.sqrt();
                } else {
                    // Off-diagonal elements
                    let mut sum = 0.0;
                    for k in 0..j {
                        sum += l[[i, k]] * l[[j, k]];
                    }
                    l[[i, j]] = (a[[i, j]] - sum) / l[[j, j]];
                }
            }
        }

        Ok(l)
    }

    fn qr_decomposition(
        &self,
        a: &ArrayView<f64, Ix2>,
    ) -> AutogradResult<(Array<f64, Ix2>, Array<f64, Ix2>)> {
        let (m, n) = (a.nrows(), a.ncols());
        let q = Array::eye(m);
        let mut r = a.to_owned();

        // Gram-Schmidt process (simplified implementation)
        for j in 0..n.min(m) {
            // Normalize column j
            let mut norm = 0.0;
            for i in j..m {
                norm += r[[i, j]] * r[[i, j]];
            }
            norm = norm.sqrt();

            if norm < 1e-12 {
                return Err(AutogradError::gradient_computation(
                    "blas_dot",
                    "Matrix is rank deficient".to_string(),
                ));
            }

            for i in j..m {
                r[[i, j]] /= norm;
            }

            // Orthogonalize remaining columns
            for k in (j + 1)..n {
                let mut dot_product = 0.0;
                for i in j..m {
                    dot_product += r[[i, j]] * r[[i, k]];
                }

                for i in j..m {
                    r[[i, k]] -= dot_product * r[[i, j]];
                }
            }
        }

        Ok((q, r))
    }

    fn svd(
        &self,
        a: &ArrayView<f64, Ix2>,
    ) -> AutogradResult<(Array<f64, Ix2>, Array<f64, Ix1>, Array<f64, Ix2>)> {
        // Simplified SVD implementation (in practice, would use more sophisticated algorithms)
        let (m, n) = (a.nrows(), a.ncols());

        // Placeholder implementation - return identity matrices and zero singular values
        let u = Array::eye(m);
        let s = Array::zeros(n.min(m));
        let vt = Array::eye(n);

        tracing::warn!("Using placeholder SVD implementation - not suitable for production");
        Ok((u, s, vt))
    }

    fn benchmark_operation(&self, operation: BlasOperation, size: usize) -> AutogradResult<f64> {
        use std::time::Instant;

        let iterations = 10;
        let start = Instant::now();

        match operation {
            BlasOperation::DOT => {
                let x = Array::ones(size);
                let y = Array::ones(size);
                for _ in 0..iterations {
                    let _result = self.dot(&x.view(), &y.view())?;
                }
            }
            BlasOperation::GEMM => {
                let a = Array::ones((size, size));
                let b = Array::ones((size, size));
                let mut c = Array::zeros((size, size));
                for _ in 0..iterations {
                    self.gemm(
                        1.0,
                        &a.view(),
                        &b.view(),
                        0.0,
                        &mut c.view_mut(),
                        false,
                        false,
                    )?;
                }
            }
            _ => {
                return Err(AutogradError::gradient_computation(
                    "blas_dot",
                    format!("Benchmarking not implemented for operation: {}", operation),
                ));
            }
        }

        let elapsed = start.elapsed();
        let time_per_operation = elapsed.as_secs_f64() / iterations as f64;
        Ok(time_per_operation)
    }

    fn get_optimal_block_size(&self, _operation: BlasOperation) -> usize {
        64 // Default block size for cache efficiency
    }

    fn supports_threading(&self) -> bool {
        true
    }

    fn set_num_threads(&mut self, threads: usize) -> AutogradResult<()> {
        self.num_threads = threads;
        tracing::debug!("Set Pure Rust BLAS provider to use {} threads", threads);
        Ok(())
    }
}

/// BLAS integration manager
pub struct BlasManager {
    config: BlasConfig,
    active_provider: Option<Box<dyn BlasProvider>>,
    available_providers: HashMap<BlasImplementation, Box<dyn BlasProvider>>,
    performance_cache: RwLock<HashMap<(BlasOperation, usize), f64>>,
    operation_stats: Mutex<HashMap<BlasOperation, OperationStats>>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OperationStats {
    pub total_calls: usize,
    pub total_time: f64,
    pub average_time: f64,
    pub min_time: f64,
    pub max_time: f64,
}

impl BlasManager {
    pub fn new(config: BlasConfig) -> Self {
        Self {
            config,
            active_provider: None,
            available_providers: HashMap::new(),
            performance_cache: RwLock::new(HashMap::new()),
            operation_stats: Mutex::new(HashMap::new()),
        }
    }

    pub fn with_default_config() -> Self {
        Self::new(BlasConfig::default())
    }

    pub fn initialize(&mut self) -> AutogradResult<()> {
        // Register available providers
        self.register_providers()?;

        // Select and initialize the best provider
        self.select_provider()?;

        tracing::info!(
            "BLAS manager initialized with provider: {}",
            self.active_provider.as_ref().unwrap().implementation()
        );

        Ok(())
    }

    fn register_providers(&mut self) -> AutogradResult<()> {
        // Always register Pure Rust provider as fallback
        let pure_rust = Box::new(PureRustBlasProvider::new());
        self.available_providers
            .insert(BlasImplementation::PureRust, pure_rust);

        // TODO: Register other BLAS providers when available
        // This would check for and initialize MKL, OpenBLAS, etc.

        tracing::debug!(
            "Registered {} BLAS providers",
            self.available_providers.len()
        );
        Ok(())
    }

    fn select_provider(&mut self) -> AutogradResult<()> {
        // Try preferred implementation first
        if let Some(provider) = self
            .available_providers
            .remove(&self.config.preferred_implementation)
        {
            if provider.is_available() {
                let mut provider = provider;
                provider.initialize()?;
                self.active_provider = Some(provider);
                return Ok(());
            }
        }

        // Try fallback implementations
        for implementation in &self.config.fallback_implementations {
            if let Some(provider) = self.available_providers.remove(implementation) {
                if provider.is_available() {
                    let mut provider = provider;
                    provider.initialize()?;
                    self.active_provider = Some(provider);
                    return Ok(());
                }
            }
        }

        Err(AutogradError::gradient_computation(
            "blas_operation",
            "No BLAS provider available",
        ))
    }

    pub fn get_active_implementation(&self) -> Option<BlasImplementation> {
        self.active_provider.as_ref().map(|p| p.implementation())
    }

    pub fn should_use_blas(&self, operation: BlasOperation, size: usize) -> bool {
        if let Some(threshold) = self.config.performance_thresholds.get(&operation) {
            // Check if we have cached performance data
            if let Ok(cache) = self.performance_cache.read() {
                if let Some(&cached_ratio) = cache.get(&(operation, size)) {
                    return cached_ratio >= *threshold;
                }
            }

            // If no cached data and benchmarking is enabled, run benchmark
            if self.config.enable_benchmarking {
                if let Ok(ratio) = self.benchmark_vs_fallback(operation, size) {
                    // Cache the result
                    if let Ok(mut cache) = self.performance_cache.write() {
                        cache.insert((operation, size), ratio);
                    }
                    return ratio >= *threshold;
                }
            }

            // Default to using BLAS for larger operations
            size >= 32
        } else {
            // If no threshold specified, use for medium-large operations
            size >= 64
        }
    }

    fn benchmark_vs_fallback(&self, operation: BlasOperation, size: usize) -> AutogradResult<f64> {
        if let Some(ref active_provider) = self.active_provider {
            let blas_time = active_provider.benchmark_operation(operation, size)?;

            // Benchmark fallback (Pure Rust)
            let fallback_provider = PureRustBlasProvider::new();
            let fallback_time = fallback_provider.benchmark_operation(operation, size)?;

            let ratio = fallback_time / blas_time;
            tracing::debug!(
                "BLAS vs fallback ratio for {} (size {}): {:.2}x",
                operation,
                size,
                ratio
            );

            Ok(ratio)
        } else {
            Err(AutogradError::gradient_computation(
                "blas_benchmark",
                "No active BLAS provider",
            ))
        }
    }

    pub fn dot_product(
        &self,
        x: &ArrayView<f64, Ix1>,
        y: &ArrayView<f64, Ix1>,
    ) -> AutogradResult<f64> {
        let start = std::time::Instant::now();

        let result = if self.should_use_blas(BlasOperation::DOT, x.len()) {
            if let Some(ref provider) = self.active_provider {
                provider.dot(x, y)?
            } else {
                // Fallback implementation
                let fallback = PureRustBlasProvider::new();
                fallback.dot(x, y)?
            }
        } else {
            // Use simple implementation for small vectors
            x.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
        };

        self.record_operation_time(BlasOperation::DOT, start.elapsed().as_secs_f64());
        Ok(result)
    }

    pub fn matrix_vector_multiply(
        &self,
        alpha: f64,
        a: &ArrayView<f64, Ix2>,
        x: &ArrayView<f64, Ix1>,
        beta: f64,
        y: &mut ArrayViewMut<f64, Ix1>,
        transpose: bool,
    ) -> AutogradResult<()> {
        let start = std::time::Instant::now();

        if self.should_use_blas(BlasOperation::GEMV, a.nrows() * a.ncols()) {
            if let Some(ref provider) = self.active_provider {
                provider.gemv(alpha, a, x, beta, y, transpose)?;
            } else {
                let fallback = PureRustBlasProvider::new();
                fallback.gemv(alpha, a, x, beta, y, transpose)?;
            }
        } else {
            // Simple implementation for small matrices
            let fallback = PureRustBlasProvider::new();
            fallback.gemv(alpha, a, x, beta, y, transpose)?;
        }

        self.record_operation_time(BlasOperation::GEMV, start.elapsed().as_secs_f64());
        Ok(())
    }

    pub fn matrix_multiply(
        &self,
        alpha: f64,
        a: &ArrayView<f64, Ix2>,
        b: &ArrayView<f64, Ix2>,
        beta: f64,
        c: &mut ArrayViewMut<f64, Ix2>,
        transpose_a: bool,
        transpose_b: bool,
    ) -> AutogradResult<()> {
        let start = std::time::Instant::now();

        let size = a.nrows() * a.ncols() + b.nrows() * b.ncols();
        if self.should_use_blas(BlasOperation::GEMM, size) {
            if let Some(ref provider) = self.active_provider {
                provider.gemm(alpha, a, b, beta, c, transpose_a, transpose_b)?;
            } else {
                let fallback = PureRustBlasProvider::new();
                fallback.gemm(alpha, a, b, beta, c, transpose_a, transpose_b)?;
            }
        } else {
            let fallback = PureRustBlasProvider::new();
            fallback.gemm(alpha, a, b, beta, c, transpose_a, transpose_b)?;
        }

        self.record_operation_time(BlasOperation::GEMM, start.elapsed().as_secs_f64());
        Ok(())
    }

    pub fn cholesky_decomposition(
        &self,
        a: &ArrayView<f64, Ix2>,
    ) -> AutogradResult<Array<f64, Ix2>> {
        let start = std::time::Instant::now();

        let result = if let Some(ref provider) = self.active_provider {
            provider.cholesky(a)?
        } else {
            let fallback = PureRustBlasProvider::new();
            fallback.cholesky(a)?
        };

        self.record_operation_time(BlasOperation::CHOLESKY, start.elapsed().as_secs_f64());
        Ok(result)
    }

    pub fn qr_decomposition(
        &self,
        a: &ArrayView<f64, Ix2>,
    ) -> AutogradResult<(Array<f64, Ix2>, Array<f64, Ix2>)> {
        let start = std::time::Instant::now();

        let result = if let Some(ref provider) = self.active_provider {
            provider.qr_decomposition(a)?
        } else {
            let fallback = PureRustBlasProvider::new();
            fallback.qr_decomposition(a)?
        };

        self.record_operation_time(BlasOperation::QR, start.elapsed().as_secs_f64());
        Ok(result)
    }

    pub fn svd(
        &self,
        a: &ArrayView<f64, Ix2>,
    ) -> AutogradResult<(Array<f64, Ix2>, Array<f64, Ix1>, Array<f64, Ix2>)> {
        let start = std::time::Instant::now();

        let result = if let Some(ref provider) = self.active_provider {
            provider.svd(a)?
        } else {
            let fallback = PureRustBlasProvider::new();
            fallback.svd(a)?
        };

        self.record_operation_time(BlasOperation::SVD, start.elapsed().as_secs_f64());
        Ok(result)
    }

    fn record_operation_time(&self, operation: BlasOperation, time: f64) {
        if let Ok(mut stats) = self.operation_stats.lock() {
            let op_stats = stats.entry(operation).or_insert_with(Default::default);

            op_stats.total_calls += 1;
            op_stats.total_time += time;
            op_stats.average_time = op_stats.total_time / op_stats.total_calls as f64;

            if op_stats.total_calls == 1 {
                op_stats.min_time = time;
                op_stats.max_time = time;
            } else {
                op_stats.min_time = op_stats.min_time.min(time);
                op_stats.max_time = op_stats.max_time.max(time);
            }
        }
    }

    pub fn get_performance_report(&self) -> BlasPerformanceReport {
        let stats = self
            .operation_stats
            .lock()
            .expect("lock should not be poisoned")
            .clone();
        let cache = self
            .performance_cache
            .read()
            .expect("lock should not be poisoned")
            .clone();

        BlasPerformanceReport {
            active_implementation: self.get_active_implementation(),
            operation_stats: stats.clone(),
            performance_cache: cache,
            total_operations: stats.values().map(|s| s.total_calls).sum(),
            total_time: stats.values().map(|s| s.total_time).sum(),
        }
    }

    pub fn clear_performance_cache(&self) {
        if let Ok(mut cache) = self.performance_cache.write() {
            cache.clear();
        }
        if let Ok(mut stats) = self.operation_stats.lock() {
            stats.clear();
        }
    }
}

/// Performance report for BLAS operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlasPerformanceReport {
    pub active_implementation: Option<BlasImplementation>,
    pub operation_stats: HashMap<BlasOperation, OperationStats>,
    pub performance_cache: HashMap<(BlasOperation, usize), f64>,
    pub total_operations: usize,
    pub total_time: f64,
}

impl BlasPerformanceReport {
    pub fn print_summary(&self) {
        println!("=== BLAS Performance Report ===");

        if let Some(impl_name) = &self.active_implementation {
            println!("Active Implementation: {}", impl_name);
        }

        println!("Total Operations: {}", self.total_operations);
        println!("Total Time: {:.4}s", self.total_time);

        if self.total_time > 0.0 {
            println!(
                "Average Operations/sec: {:.2}",
                self.total_operations as f64 / self.total_time
            );
        }

        println!();
        println!("Operation Statistics:");

        for (operation, stats) in &self.operation_stats {
            println!("  {}:", operation);
            println!("    Calls: {}", stats.total_calls);
            println!("    Avg Time: {:.6}s", stats.average_time);
            println!("    Min Time: {:.6}s", stats.min_time);
            println!("    Max Time: {:.6}s", stats.max_time);
            println!();
        }
    }
}

/// Global BLAS manager instance
static GLOBAL_BLAS_MANAGER: std::sync::OnceLock<BlasManager> = std::sync::OnceLock::new();

pub fn get_global_blas_manager() -> &'static BlasManager {
    GLOBAL_BLAS_MANAGER.get_or_init(|| {
        let mut manager = BlasManager::with_default_config();
        if let Err(e) = manager.initialize() {
            tracing::error!("Failed to initialize BLAS manager: {}", e);
            // Continue with uninitialized manager as fallback
        }
        manager
    })
}

/// Convenience functions for BLAS operations
pub fn blas_dot(x: &ArrayView<f64, Ix1>, y: &ArrayView<f64, Ix1>) -> AutogradResult<f64> {
    get_global_blas_manager().dot_product(x, y)
}

pub fn blas_gemv(
    alpha: f64,
    a: &ArrayView<f64, Ix2>,
    x: &ArrayView<f64, Ix1>,
    beta: f64,
    y: &mut ArrayViewMut<f64, Ix1>,
    transpose: bool,
) -> AutogradResult<()> {
    get_global_blas_manager().matrix_vector_multiply(alpha, a, x, beta, y, transpose)
}

pub fn blas_gemm(
    alpha: f64,
    a: &ArrayView<f64, Ix2>,
    b: &ArrayView<f64, Ix2>,
    beta: f64,
    c: &mut ArrayViewMut<f64, Ix2>,
    transpose_a: bool,
    transpose_b: bool,
) -> AutogradResult<()> {
    get_global_blas_manager().matrix_multiply(alpha, a, b, beta, c, transpose_a, transpose_b)
}

#[cfg(test)]
mod tests {
    use super::*;
    // Array macro is not available from scirs2_core::ndarray_ext, using Vec instead

    #[test]
    fn test_blas_config_creation() {
        let config = BlasConfig::default();
        assert!(config.enable_threading);
        assert!(!config.performance_thresholds.is_empty());
    }

    #[test]
    fn test_pure_rust_provider() {
        let mut provider = PureRustBlasProvider::new();
        assert!(provider.is_available());
        assert!(provider.initialize().is_ok());

        // Test dot product
        let x = Array::from_vec(vec![1.0, 2.0, 3.0])
            .into_shape_with_order((3,))
            .unwrap();
        let y = Array::from_vec(vec![4.0, 5.0, 6.0])
            .into_shape_with_order((3,))
            .unwrap();
        let result = provider.dot(&x.view(), &y.view()).unwrap();
        assert_eq!(result, 32.0); // 1*4 + 2*5 + 3*6 = 32

        // Test matrix multiplication
        let a = Array::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Array::from_shape_vec((2, 2), vec![5.0, 6.0, 7.0, 8.0]).unwrap();
        let mut c = Array::zeros((2, 2));

        provider
            .gemm(
                1.0,
                &a.view(),
                &b.view(),
                0.0,
                &mut c.view_mut(),
                false,
                false,
            )
            .unwrap();

        // Expected: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
        assert_eq!(c[[0, 0]], 19.0);
        assert_eq!(c[[0, 1]], 22.0);
        assert_eq!(c[[1, 0]], 43.0);
        assert_eq!(c[[1, 1]], 50.0);
    }

    #[test]
    fn test_blas_manager_initialization() {
        let mut manager = BlasManager::with_default_config();
        assert!(manager.initialize().is_ok());
        assert!(manager.get_active_implementation().is_some());
    }

    #[test]
    fn test_matrix_vector_multiply() {
        let manager = get_global_blas_manager();

        let a = Array::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let x = Array::from_vec(vec![5.0, 6.0])
            .into_shape_with_order((2,))
            .unwrap();
        let mut y = Array::zeros(2);

        manager
            .matrix_vector_multiply(1.0, &a.view(), &x.view(), 0.0, &mut y.view_mut(), false)
            .unwrap();

        // Expected: [1*5+2*6, 3*5+4*6] = [17, 39]
        assert_eq!(y[0], 17.0);
        assert_eq!(y[1], 39.0);
    }

    #[test]
    fn test_cholesky_decomposition() {
        let manager = get_global_blas_manager();

        // Positive definite matrix
        let a = Array::from_shape_vec((2, 2), vec![4.0, 2.0, 2.0, 2.0]).unwrap();
        let l = manager.cholesky_decomposition(&a.view()).unwrap();

        // Verify L * L^T = A
        let mut reconstructed = Array::zeros((2, 2));
        manager
            .matrix_multiply(
                1.0,
                &l.view(),
                &l.view(),
                0.0,
                &mut reconstructed.view_mut(),
                false,
                true,
            )
            .unwrap();

        // Check with some tolerance for floating point errors
        assert!((reconstructed[[0, 0]] - 4.0).abs() < 1e-10);
        assert!((reconstructed[[0, 1]] - 2.0).abs() < 1e-10);
        assert!((reconstructed[[1, 0]] - 2.0).abs() < 1e-10);
        assert!((reconstructed[[1, 1]] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_performance_thresholds() {
        // Create a test configuration without benchmarking
        let mut config = BlasConfig::default();
        config.enable_benchmarking = false;

        let mut manager = BlasManager::new(config);
        let _ = manager.initialize(); // Initialize but ignore errors for testing

        // Pre-populate performance cache with test data
        {
            let mut cache = manager
                .performance_cache
                .write()
                .expect("lock should not be poisoned");
            cache.insert((BlasOperation::DOT, 10), 0.5); // Below threshold
            cache.insert((BlasOperation::GEMM, 1000), 3.0); // Above threshold
        }

        // Small operations should not use BLAS
        assert!(!manager.should_use_blas(BlasOperation::DOT, 10));

        // Large operations should use BLAS
        assert!(manager.should_use_blas(BlasOperation::GEMM, 1000));
    }

    #[test]
    fn test_convenience_functions() {
        let x = Array::from_vec(vec![1.0, 2.0, 3.0])
            .into_shape_with_order((3,))
            .unwrap();
        let y = Array::from_vec(vec![4.0, 5.0, 6.0])
            .into_shape_with_order((3,))
            .unwrap();

        let result = blas_dot(&x.view(), &y.view()).unwrap();
        assert_eq!(result, 32.0);
    }

    #[test]
    fn test_performance_report() {
        let manager = get_global_blas_manager();

        // Perform some operations to generate stats
        let x = Array::from_vec(vec![1.0, 2.0, 3.0])
            .into_shape_with_order((3,))
            .unwrap();
        let y = Array::from_vec(vec![4.0, 5.0, 6.0])
            .into_shape_with_order((3,))
            .unwrap();
        let _result = manager.dot_product(&x.view(), &y.view()).unwrap();

        let report = manager.get_performance_report();
        assert!(report.total_operations > 0);
        assert!(report.operation_stats.contains_key(&BlasOperation::DOT));
    }

    #[test]
    fn test_blas_implementation_display() {
        assert_eq!(BlasImplementation::MKL.to_string(), "Intel MKL");
        assert_eq!(BlasImplementation::OpenBLAS.to_string(), "OpenBLAS");
        assert_eq!(
            BlasImplementation::Custom("test".to_string()).to_string(),
            "Custom(test)"
        );
    }

    #[test]
    fn test_blas_operation_display() {
        assert_eq!(
            BlasOperation::GEMM.to_string(),
            "GEMM (Matrix-Matrix Multiply)"
        );
        assert_eq!(BlasOperation::DOT.to_string(), "DOT (Vector Dot Product)");
    }
}
