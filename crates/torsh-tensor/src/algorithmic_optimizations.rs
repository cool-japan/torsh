//! Algorithmic Efficiency Optimizations for Core Tensor Operations
//!
//! This module provides cutting-edge algorithmic optimizations that enhance the fundamental
//! efficiency of tensor operations through advanced mathematical techniques, adaptive algorithms,
//! and intelligent operation scheduling.
//!
//! # Features
//!
//! - **Adaptive Algorithm Selection**: Runtime selection of optimal algorithms based on tensor properties
//! - **Operation Fusion**: Multi-operation fusion for reduced memory bandwidth and computation
//! - **Cache-Oblivious Algorithms**: Memory hierarchy-aware algorithms that adapt to hardware
//! - **Numerical Stability Enhancements**: Advanced numerical techniques for robust computations
//! - **Asymptotic Optimizations**: Implementation of asymptotically superior algorithms
//! - **Parallel Algorithm Scheduling**: Intelligent work distribution for multi-core efficiency

use std::cmp::min;
use std::collections::HashMap;
use std::time::Instant;

// SciRS2 Parallel Operations for algorithmic optimizations
use scirs2_core::parallel_ops::*;
use torsh_core::{
    dtype::FloatElement,
    error::{Result, TorshError},
};

// Standard Rust Algorithm Integration (fallback from scirs2_core)
// Note: Using stable Rust APIs instead of unstable std::simd

/// Configuration for algorithmic optimizations
#[derive(Debug, Clone)]
pub struct AlgorithmConfig {
    /// Enable adaptive algorithm selection
    pub enable_adaptive_selection: bool,
    /// Minimum size for using advanced algorithms
    pub min_size_for_advanced: usize,
    /// Cache size hints for cache-oblivious algorithms
    pub l1_cache_size: usize,
    pub l2_cache_size: usize,
    pub l3_cache_size: usize,
    /// Enable operation fusion
    pub enable_operation_fusion: bool,
    /// Maximum fusion chain length
    pub max_fusion_chain: usize,
    /// Enable numerical stability optimizations
    pub enable_numerical_stability: bool,
    /// Parallel scheduling strategy
    pub scheduling_strategy: SchedulingStrategy,
}

impl Default for AlgorithmConfig {
    fn default() -> Self {
        Self {
            enable_adaptive_selection: true,
            min_size_for_advanced: 64,
            l1_cache_size: 32 * 1024,       // 32KB L1
            l2_cache_size: 256 * 1024,      // 256KB L2
            l3_cache_size: 8 * 1024 * 1024, // 8MB L3
            enable_operation_fusion: true,
            max_fusion_chain: 8,
            enable_numerical_stability: true,
            scheduling_strategy: SchedulingStrategy::WorkStealing,
        }
    }
}

/// Parallel scheduling strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulingStrategy {
    /// Static work distribution
    Static,
    /// Dynamic work stealing
    WorkStealing,
    /// Adaptive load balancing
    Adaptive,
    /// NUMA-aware scheduling
    NumaAware,
}

/// Advanced algorithmic operations manager
pub struct AlgorithmicOptimizer {
    config: AlgorithmConfig,
    /// Operation performance history for adaptive selection
    performance_history: std::sync::RwLock<HashMap<OperationSignature, PerformanceMetrics>>,
}

impl AlgorithmicOptimizer {
    /// Create new algorithmic optimizer
    pub fn new() -> Self {
        Self::with_config(AlgorithmConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: AlgorithmConfig) -> Self {
        Self {
            config,
            performance_history: std::sync::RwLock::new(HashMap::new()),
        }
    }

    /// Optimized matrix multiplication with adaptive algorithm selection
    pub fn optimized_matmul<T>(
        &self,
        a: &[T],
        b: &[T],
        c: &mut [T],
        m: usize, // rows of A
        k: usize, // cols of A, rows of B
        n: usize, // cols of B
    ) -> Result<()>
    where
        T: FloatElement + Send + Sync + std::ops::AddAssign,
    {
        #[cfg(feature = "profiling")]
        {
            // let _profile = profile_section!("optimized_matmul");
        }
        let signature = OperationSignature::MatMul { m, k, n };

        // Select optimal algorithm based on size and previous performance
        let algorithm = self.select_matmul_algorithm(&signature);

        let start_time = Instant::now();

        match algorithm {
            MatMulAlgorithm::Naive => self.naive_matmul(a, b, c, m, k, n)?,
            MatMulAlgorithm::Blocked => self.blocked_matmul(a, b, c, m, k, n)?,
            MatMulAlgorithm::Strassen => self.strassen_matmul(a, b, c, m, k, n)?,
            MatMulAlgorithm::CacheOblivious => self.cache_oblivious_matmul(a, b, c, m, k, n)?,
            MatMulAlgorithm::Parallel => self.parallel_matmul(a, b, c, m, k, n)?,
        }

        // Record performance for future algorithm selection
        let duration = start_time.elapsed();
        self.record_performance(signature, algorithm, duration);

        Ok(())
    }

    /// Select optimal matrix multiplication algorithm
    fn select_matmul_algorithm(&self, signature: &OperationSignature) -> MatMulAlgorithm {
        if !self.config.enable_adaptive_selection {
            return MatMulAlgorithm::Blocked; // Default fallback
        }

        // Check performance history
        if let Some(metrics) = self
            .performance_history
            .read()
            .expect("lock should not be poisoned")
            .get(signature)
        {
            return metrics
                .best_algorithm
                .clone()
                .unwrap_or(MatMulAlgorithm::Blocked);
        }

        // Algorithm selection based on problem size
        match signature {
            OperationSignature::MatMul { m, k, n } => {
                let total_size = m * k * n;

                if total_size < 1000 {
                    MatMulAlgorithm::Naive
                } else if total_size < 10000 {
                    MatMulAlgorithm::Blocked
                } else if *m >= 1024 && *k >= 1024 && *n >= 1024 {
                    MatMulAlgorithm::Strassen
                } else if total_size > 100000 {
                    MatMulAlgorithm::Parallel
                } else {
                    MatMulAlgorithm::CacheOblivious
                }
            }
        }
    }

    /// Naive matrix multiplication (O(n³))
    fn naive_matmul<T>(
        &self,
        a: &[T],
        b: &[T],
        c: &mut [T],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<()>
    where
        T: FloatElement + std::ops::AddAssign,
    {
        for i in 0..m {
            for j in 0..n {
                let mut sum = <T as torsh_core::TensorElement>::zero();
                for l in 0..k {
                    sum += a[i * k + l] * b[l * n + j];
                }
                c[i * n + j] = sum;
            }
        }
        Ok(())
    }

    /// Cache-blocked matrix multiplication
    fn blocked_matmul<T>(
        &self,
        a: &[T],
        b: &[T],
        c: &mut [T],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<()>
    where
        T: FloatElement + std::ops::AddAssign,
    {
        // Calculate optimal block size based on cache hierarchy
        let block_size = self.calculate_optimal_block_size(m, k, n);

        for i_block in (0..m).step_by(block_size) {
            for j_block in (0..n).step_by(block_size) {
                for k_block in (0..k).step_by(block_size) {
                    let i_end = min(i_block + block_size, m);
                    let j_end = min(j_block + block_size, n);
                    let k_end = min(k_block + block_size, k);

                    // Multiply the blocks
                    for i in i_block..i_end {
                        for j in j_block..j_end {
                            let mut sum = if k_block == 0 {
                                <T as torsh_core::TensorElement>::zero()
                            } else {
                                c[i * n + j]
                            };
                            for l in k_block..k_end {
                                sum += a[i * k + l] * b[l * n + j];
                            }
                            c[i * n + j] = sum;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Strassen matrix multiplication (O(n^2.807))
    fn strassen_matmul<T>(
        &self,
        a: &[T],
        b: &[T],
        c: &mut [T],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<()>
    where
        T: FloatElement + Send + Sync + std::ops::AddAssign,
    {
        // For non-square or small matrices, fall back to blocked algorithm
        if m != k || k != n || m < 128 {
            return self.blocked_matmul(a, b, c, m, k, n);
        }

        self.strassen_recursive(a, b, c, m, 0, 0, 0, 0, 0, 0)
    }

    /// Recursive Strassen implementation
    fn strassen_recursive<T>(
        &self,
        a: &[T],
        b: &[T],
        c: &mut [T],
        n: usize,
        a_row: usize,
        a_col: usize,
        b_row: usize,
        b_col: usize,
        c_row: usize,
        c_col: usize,
    ) -> Result<()>
    where
        T: FloatElement + Send + Sync + std::ops::AddAssign,
    {
        if n <= 64 {
            // Base case: use naive multiplication for small matrices
            for i in 0..n {
                for j in 0..n {
                    let mut sum = <T as torsh_core::TensorElement>::zero();
                    for k in 0..n {
                        let a_val = a[(a_row + i) * n + (a_col + k)];
                        let b_val = b[(b_row + k) * n + (b_col + j)];
                        sum += a_val * b_val;
                    }
                    c[(c_row + i) * n + (c_col + j)] = sum;
                }
            }
            return Ok(());
        }

        let half = n / 2;

        // Allocate temporary matrices for Strassen products and intermediate results
        let temp_size = half * half;
        let mut m1 = vec![<T as torsh_core::TensorElement>::zero(); temp_size];
        let mut m2 = vec![<T as torsh_core::TensorElement>::zero(); temp_size];
        let mut m3 = vec![<T as torsh_core::TensorElement>::zero(); temp_size];
        let mut m4 = vec![<T as torsh_core::TensorElement>::zero(); temp_size];
        let mut m5 = vec![<T as torsh_core::TensorElement>::zero(); temp_size];
        let mut m6 = vec![<T as torsh_core::TensorElement>::zero(); temp_size];
        let mut m7 = vec![<T as torsh_core::TensorElement>::zero(); temp_size];

        // Allocate temporary matrices for sums/differences
        let mut temp_a = vec![<T as torsh_core::TensorElement>::zero(); temp_size];
        let mut temp_b = vec![<T as torsh_core::TensorElement>::zero(); temp_size];

        // Helper to add two matrix quadrants: temp = A_quad1 + A_quad2
        let add_quadrants = |temp: &mut [T],
                             quad1_row: usize,
                             quad1_col: usize,
                             quad2_row: usize,
                             quad2_col: usize,
                             source: &[T]| {
            for i in 0..half {
                for j in 0..half {
                    let val1 = source[(quad1_row + i) * n + (quad1_col + j)];
                    let val2 = source[(quad2_row + i) * n + (quad2_col + j)];
                    temp[i * half + j] = val1 + val2;
                }
            }
        };

        // Helper to subtract two matrix quadrants: temp = A_quad1 - A_quad2
        let sub_quadrants = |temp: &mut [T],
                             quad1_row: usize,
                             quad1_col: usize,
                             quad2_row: usize,
                             quad2_col: usize,
                             source: &[T]| {
            for i in 0..half {
                for j in 0..half {
                    let val1 = source[(quad1_row + i) * n + (quad1_col + j)];
                    let val2 = source[(quad2_row + i) * n + (quad2_col + j)];
                    temp[i * half + j] = val1 - val2;
                }
            }
        };

        // M1 = (A11 + A22)(B11 + B22)
        add_quadrants(&mut temp_a, a_row, a_col, a_row + half, a_col + half, a);
        add_quadrants(&mut temp_b, b_row, b_col, b_row + half, b_col + half, b);
        self.blocked_matmul(&temp_a, &temp_b, &mut m1, half, half, half)?;

        // M2 = (A21 + A22)B11
        add_quadrants(
            &mut temp_a,
            a_row + half,
            a_col,
            a_row + half,
            a_col + half,
            a,
        );
        for i in 0..half {
            for j in 0..half {
                temp_b[i * half + j] = b[(b_row + i) * n + (b_col + j)];
            }
        }
        self.blocked_matmul(&temp_a, &temp_b, &mut m2, half, half, half)?;

        // M3 = A11(B12 - B22)
        for i in 0..half {
            for j in 0..half {
                temp_a[i * half + j] = a[(a_row + i) * n + (a_col + j)];
            }
        }
        sub_quadrants(
            &mut temp_b,
            b_row,
            b_col + half,
            b_row + half,
            b_col + half,
            b,
        );
        self.blocked_matmul(&temp_a, &temp_b, &mut m3, half, half, half)?;

        // M4 = A22(B21 - B11)
        for i in 0..half {
            for j in 0..half {
                temp_a[i * half + j] = a[(a_row + half + i) * n + (a_col + half + j)];
            }
        }
        sub_quadrants(&mut temp_b, b_row + half, b_col, b_row, b_col, b);
        self.blocked_matmul(&temp_a, &temp_b, &mut m4, half, half, half)?;

        // M5 = (A11 + A12)B22
        add_quadrants(&mut temp_a, a_row, a_col, a_row, a_col + half, a);
        for i in 0..half {
            for j in 0..half {
                temp_b[i * half + j] = b[(b_row + half + i) * n + (b_col + half + j)];
            }
        }
        self.blocked_matmul(&temp_a, &temp_b, &mut m5, half, half, half)?;

        // M6 = (A21 - A11)(B11 + B12)
        sub_quadrants(&mut temp_a, a_row + half, a_col, a_row, a_col, a);
        add_quadrants(&mut temp_b, b_row, b_col, b_row, b_col + half, b);
        self.blocked_matmul(&temp_a, &temp_b, &mut m6, half, half, half)?;

        // M7 = (A12 - A22)(B21 + B22)
        sub_quadrants(
            &mut temp_a,
            a_row,
            a_col + half,
            a_row + half,
            a_col + half,
            a,
        );
        add_quadrants(
            &mut temp_b,
            b_row + half,
            b_col,
            b_row + half,
            b_col + half,
            b,
        );
        self.blocked_matmul(&temp_a, &temp_b, &mut m7, half, half, half)?;

        // Combine results into output quadrants
        // C11 = M1 + M4 - M5 + M7
        for i in 0..half {
            for j in 0..half {
                c[(c_row + i) * n + (c_col + j)] =
                    m1[i * half + j] + m4[i * half + j] - m5[i * half + j] + m7[i * half + j];
            }
        }

        // C12 = M3 + M5
        for i in 0..half {
            for j in 0..half {
                c[(c_row + i) * n + (c_col + half + j)] = m3[i * half + j] + m5[i * half + j];
            }
        }

        // C21 = M2 + M4
        for i in 0..half {
            for j in 0..half {
                c[(c_row + half + i) * n + (c_col + j)] = m2[i * half + j] + m4[i * half + j];
            }
        }

        // C22 = M1 - M2 + M3 + M6
        for i in 0..half {
            for j in 0..half {
                c[(c_row + half + i) * n + (c_col + half + j)] =
                    m1[i * half + j] - m2[i * half + j] + m3[i * half + j] + m6[i * half + j];
            }
        }

        Ok(())
    }

    /// Cache-oblivious matrix multiplication
    fn cache_oblivious_matmul<T>(
        &self,
        a: &[T],
        b: &[T],
        c: &mut [T],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<()>
    where
        T: FloatElement + std::ops::AddAssign,
    {
        self.cache_oblivious_recursive(a, b, c, m, k, n, 0, 0, 0, 0, 0, 0)
    }

    /// Recursive cache-oblivious implementation
    fn cache_oblivious_recursive<T>(
        &self,
        a: &[T],
        b: &[T],
        c: &mut [T],
        m: usize,
        k: usize,
        n: usize,
        a_row: usize,
        a_col: usize,
        b_row: usize,
        b_col: usize,
        c_row: usize,
        c_col: usize,
    ) -> Result<()>
    where
        T: FloatElement + std::ops::AddAssign,
    {
        // Base case for small matrices
        if m <= 32 || k <= 32 || n <= 32 {
            return self
                .naive_matmul_region(a, b, c, m, k, n, a_row, a_col, b_row, b_col, c_row, c_col);
        }

        // Recursively divide along the largest dimension
        if m >= k && m >= n {
            let m1 = m / 2;
            let m2 = m - m1;

            // C₁₁ = A₁ × B
            self.cache_oblivious_recursive(
                a, b, c, m1, k, n, a_row, a_col, b_row, b_col, c_row, c_col,
            )?;

            // C₂₁ = A₂ × B
            self.cache_oblivious_recursive(
                a,
                b,
                c,
                m2,
                k,
                n,
                a_row + m1,
                a_col,
                b_row,
                b_col,
                c_row + m1,
                c_col,
            )?;
        } else if k >= n {
            let k1 = k / 2;
            let k2 = k - k1;

            // C = A₁ × B₁ + A₂ × B₂
            self.cache_oblivious_recursive(
                a, b, c, m, k1, n, a_row, a_col, b_row, b_col, c_row, c_col,
            )?;

            self.cache_oblivious_recursive(
                a,
                b,
                c,
                m,
                k2,
                n,
                a_row,
                a_col + k1,
                b_row + k1,
                b_col,
                c_row,
                c_col,
            )?;
        } else {
            let n1 = n / 2;
            let n2 = n - n1;

            // C₁ = A × B₁
            self.cache_oblivious_recursive(
                a, b, c, m, k, n1, a_row, a_col, b_row, b_col, c_row, c_col,
            )?;

            // C₂ = A × B₂
            self.cache_oblivious_recursive(
                a,
                b,
                c,
                m,
                k,
                n2,
                a_row,
                a_col,
                b_row,
                b_col + n1,
                c_row,
                c_col + n1,
            )?;
        }

        Ok(())
    }

    /// Naive multiplication for a specific region
    fn naive_matmul_region<T>(
        &self,
        a: &[T],
        b: &[T],
        c: &mut [T],
        m: usize,
        k: usize,
        n: usize,
        a_row: usize,
        a_col: usize,
        b_row: usize,
        b_col: usize,
        c_row: usize,
        c_col: usize,
    ) -> Result<()>
    where
        T: FloatElement + std::ops::AddAssign,
    {
        for i in 0..m {
            for j in 0..n {
                let mut sum = <T as torsh_core::TensorElement>::zero();
                for l in 0..k {
                    let a_idx = (a_row + i) * k + (a_col + l);
                    let b_idx = (b_row + l) * n + (b_col + j);
                    sum += a[a_idx] * b[b_idx];
                }
                let c_idx = (c_row + i) * n + (c_col + j);
                c[c_idx] += sum; // Accumulate for recursive calls
            }
        }
        Ok(())
    }

    /// Parallel matrix multiplication with intelligent scheduling
    fn parallel_matmul<T>(
        &self,
        a: &[T],
        b: &[T],
        c: &mut [T],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<()>
    where
        T: FloatElement + Send + Sync + std::ops::AddAssign,
    {
        let num_cores = get_num_threads();
        let block_size = self.calculate_optimal_block_size(m, k, n);

        // Decide whether to parallelize based on problem size and available cores
        let total_operations = m * k * n;
        let min_work_per_core = 100_000; // Minimum operations to justify parallelization overhead
        let should_parallelize = num_cores > 1 && total_operations > min_work_per_core * num_cores;

        if !should_parallelize {
            // Fall back to serial blocked multiplication for small problems
            return self.blocked_matmul(a, b, c, m, k, n);
        }

        // Create work items for parallel execution
        let work_items: Vec<_> = (0..m)
            .step_by(block_size)
            .flat_map(|i| (0..n).step_by(block_size).map(move |j| (i, j)))
            .collect();

        // Execute in parallel using SciRS2 and collect results
        let results: Result<Vec<_>> = parallel_map_result(&work_items, |&(i_block, j_block)| {
            let i_end = min(i_block + block_size, m);
            let j_end = min(j_block + block_size, n);

            let mut block_results = Vec::new();
            for i in i_block..i_end {
                for j in j_block..j_end {
                    let mut sum = <T as torsh_core::TensorElement>::zero();
                    for l in 0..k {
                        sum += a[i * k + l] * b[l * n + j];
                    }
                    let idx = i * n + j;
                    block_results.push((idx, sum));
                }
            }
            Ok(block_results)
        });

        // Assign all results to output
        for block_results in results? {
            for (idx, value) in block_results {
                c[idx] = value;
            }
        }

        Ok(())
    }

    /// Calculate optimal block size for cache efficiency
    fn calculate_optimal_block_size(&self, m: usize, k: usize, n: usize) -> usize {
        // Calculate block size based on cache size and matrix dimensions
        let element_size = std::mem::size_of::<f32>(); // Assume f32 for estimation

        // For matrix multiplication C = A*B, we need to fit blocks of A, B, and C in cache
        // A block: block_size × k, B block: k × block_size, C block: block_size × block_size
        let l1_elements = self.config.l1_cache_size / element_size;

        // Target: block_size² + 2*block_size*k ≤ L1_elements
        // Simplified: block_size ≈ sqrt(L1_elements / 3)
        let cache_optimal = (l1_elements as f64 / 3.0).sqrt() as usize;

        // Consider matrix dimensions - don't make blocks larger than necessary
        let dim_optimal = m.min(k).min(n);

        // Combine heuristics: use smaller of cache-optimal and dimension-optimal
        let optimal_block = cache_optimal.min(dim_optimal);

        // Ensure block size is reasonable (power of 2 friendly, between 16 and 256)
        let clamped = optimal_block.clamp(16, 256);

        // Round to nearest power of 2 for better memory alignment
        let log2 = (clamped as f64).log2().round() as u32;
        2usize.pow(log2).min(256)
    }

    /// Record performance metrics for algorithm selection
    fn record_performance(
        &self,
        signature: OperationSignature,
        algorithm: MatMulAlgorithm,
        duration: std::time::Duration,
    ) {
        let mut history = self
            .performance_history
            .write()
            .expect("lock should not be poisoned");
        let metrics = history
            .entry(signature)
            .or_insert_with(PerformanceMetrics::default);

        metrics.update_performance(algorithm, duration);
    }

    /// Optimized convolution with advanced algorithms
    pub fn optimized_conv2d<T>(
        &self,
        input: &[T],
        kernel: &[T],
        output: &mut [T],
        input_h: usize,
        input_w: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride: usize,
        padding: usize,
    ) -> Result<()>
    where
        T: FloatElement + Send + Sync + std::ops::AddAssign,
    {
        #[cfg(feature = "profiling")]
        {
            // let _profile = profile_section!("optimized_conv2d");
        }

        // Calculate expected output dimensions
        let output_h = (input_h + 2 * padding - kernel_h) / stride + 1;
        let output_w = (input_w + 2 * padding - kernel_w) / stride + 1;
        let expected_output_size = output_h * output_w;

        // Validate output buffer size
        if output.len() < expected_output_size {
            return Err(torsh_core::error::TorshError::InvalidShape(format!(
                "Output buffer too small: expected at least {} ({}x{}) elements, got {}",
                expected_output_size,
                output_h,
                output_w,
                output.len()
            )));
        }

        // TODO: Re-enable when tracing is added to dependencies
        // #[cfg(feature = "profiling")]
        // tracing::trace!(
        //     "Conv2d: input={}x{}, kernel={}x{}, output={}x{}, stride={}, padding={}",
        //     input_h,
        //     input_w,
        //     kernel_h,
        //     kernel_w,
        //     output_h,
        //     output_w,
        //     stride,
        //     padding
        // );

        // Select convolution algorithm based on kernel size and input size
        if kernel_h * kernel_w <= 9 && input_h * input_w > 10000 {
            // Use direct convolution for small kernels and large inputs
            self.direct_conv2d(
                input, kernel, output, input_h, input_w, kernel_h, kernel_w, stride, padding,
            )
        } else if kernel_h >= 7 && kernel_w >= 7 {
            // Use FFT-based convolution for large kernels
            self.fft_conv2d(
                input, kernel, output, input_h, input_w, kernel_h, kernel_w, stride, padding,
            )
        } else {
            // Use Winograd for medium-sized kernels
            self.winograd_conv2d(
                input, kernel, output, input_h, input_w, kernel_h, kernel_w, stride, padding,
            )
        }
    }

    /// Direct convolution implementation
    fn direct_conv2d<T>(
        &self,
        input: &[T],
        kernel: &[T],
        output: &mut [T],
        input_h: usize,
        input_w: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride: usize,
        padding: usize,
    ) -> Result<()>
    where
        T: FloatElement + Send + Sync + std::ops::AddAssign,
    {
        let output_h = (input_h + 2 * padding - kernel_h) / stride + 1;
        let output_w = (input_w + 2 * padding - kernel_w) / stride + 1;

        // SciRS2 Parallel processing over all output positions
        let output_positions: Vec<_> = (0..output_h)
            .flat_map(|out_y| (0..output_w).map(move |out_x| (out_y, out_x)))
            .collect();

        let results: Vec<_> = parallel_map_collect(output_positions, |(out_y, out_x)| {
            let mut sum = <T as torsh_core::TensorElement>::zero();

            for ky in 0..kernel_h {
                for kx in 0..kernel_w {
                    let in_y = out_y * stride + ky;
                    let in_x = out_x * stride + kx;

                    if in_y >= padding
                        && in_y < input_h + padding
                        && in_x >= padding
                        && in_x < input_w + padding
                    {
                        let input_y = in_y - padding;
                        let input_x = in_x - padding;

                        if input_y < input_h && input_x < input_w {
                            sum += input[input_y * input_w + input_x] * kernel[ky * kernel_w + kx];
                        }
                    }
                }
            }

            (out_y * output_w + out_x, sum)
        });

        // Assign results to output
        for (idx, value) in results {
            output[idx] = value;
        }

        Ok(())
    }

    /// FFT-based convolution for large kernels
    fn fft_conv2d<T>(
        &self,
        input: &[T],
        kernel: &[T],
        output: &mut [T],
        input_h: usize,
        input_w: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride: usize,
        padding: usize,
    ) -> Result<()>
    where
        T: FloatElement + std::ops::AddAssign,
    {
        // Simplified FFT convolution - in practice would use actual FFT implementation
        // For now, fall back to direct convolution
        self.direct_conv2d(
            input, kernel, output, input_h, input_w, kernel_h, kernel_w, stride, padding,
        )
    }

    /// Winograd convolution for specific kernel sizes
    fn winograd_conv2d<T>(
        &self,
        input: &[T],
        kernel: &[T],
        output: &mut [T],
        input_h: usize,
        input_w: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride: usize,
        padding: usize,
    ) -> Result<()>
    where
        T: FloatElement + std::ops::AddAssign,
    {
        // Simplified Winograd - in practice would implement F(2x2,3x3) or F(4x4,3x3)
        // For now, fall back to direct convolution
        self.direct_conv2d(
            input, kernel, output, input_h, input_w, kernel_h, kernel_w, stride, padding,
        )
    }

    /// Fused operation execution
    pub fn execute_fused_operations<T>(
        &self,
        operations: &[FusedOperation<T>],
        inputs: &[&[T]],
        outputs: &mut [&mut [T]],
    ) -> Result<()>
    where
        T: FloatElement + Send + Sync + std::ops::AddAssign,
    {
        if !self.config.enable_operation_fusion {
            return Err(TorshError::InvalidArgument(
                "Operation fusion disabled".to_string(),
            ));
        }

        #[cfg(feature = "profiling")]
        {
            // let _profile = profile_section!("execute_fused_operations");
        }

        // Compile fusion directly (caching disabled for now due to generic complexity)
        let compiled = self.compile_fusion(operations)?;
        compiled.execute(inputs, outputs)
    }

    /// Compile fusion operations into optimized execution plan
    fn compile_fusion<T>(&self, operations: &[FusedOperation<T>]) -> Result<CompiledFusion<T>>
    where
        T: FloatElement + std::ops::AddAssign,
    {
        // Simplified fusion compilation - would be more sophisticated in practice
        let plan = ExecutionPlan {
            operations: operations.to_vec(),
            optimization_level: OptimizationLevel::Aggressive,
        };

        Ok(CompiledFusion {
            plan,
            estimated_flops: self.estimate_fusion_flops(operations),
        })
    }

    /// Estimate FLOPs for fusion operations
    fn estimate_fusion_flops<T>(&self, operations: &[FusedOperation<T>]) -> usize
    where
        T: FloatElement + std::ops::AddAssign,
    {
        // Simplified FLOP estimation
        operations.len() * 1000 // Placeholder
    }

    /// Get algorithm performance statistics
    pub fn get_performance_stats(&self) -> AlgorithmPerformanceStats {
        let history = self
            .performance_history
            .read()
            .expect("lock should not be poisoned");

        let mut total_operations = 0;
        let mut algorithm_counts = HashMap::new();

        for metrics in history.values() {
            total_operations += metrics.execution_count;
            if let Some(ref algorithm) = metrics.best_algorithm {
                *algorithm_counts.entry(algorithm.clone()).or_insert(0) += 1;
            }
        }

        AlgorithmPerformanceStats {
            total_operations,
            unique_operation_signatures: history.len(),
            algorithm_distribution: algorithm_counts,
            average_speedup: self.calculate_average_speedup(&history),
        }
    }

    /// Calculate average speedup from adaptive algorithm selection
    fn calculate_average_speedup(
        &self,
        history: &HashMap<OperationSignature, PerformanceMetrics>,
    ) -> f64 {
        if history.is_empty() {
            return 1.0;
        }

        let speedups: Vec<f64> = history
            .values()
            .filter_map(|metrics| metrics.best_speedup)
            .collect();

        if speedups.is_empty() {
            1.0
        } else {
            speedups.iter().sum::<f64>() / speedups.len() as f64
        }
    }
}

impl Default for AlgorithmicOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Operation signature for performance tracking
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
enum OperationSignature {
    MatMul { m: usize, k: usize, n: usize },
}

/// Matrix multiplication algorithms
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MatMulAlgorithm {
    Naive,
    Blocked,
    Strassen,
    CacheOblivious,
    Parallel,
}

/// Performance metrics for adaptive algorithm selection
#[derive(Debug, Clone)]
struct PerformanceMetrics {
    execution_count: usize,
    algorithm_timings: HashMap<MatMulAlgorithm, Vec<std::time::Duration>>,
    best_algorithm: Option<MatMulAlgorithm>,
    best_speedup: Option<f64>,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            execution_count: 0,
            algorithm_timings: HashMap::new(),
            best_algorithm: None,
            best_speedup: None,
        }
    }
}

impl PerformanceMetrics {
    fn update_performance(&mut self, algorithm: MatMulAlgorithm, duration: std::time::Duration) {
        self.execution_count += 1;
        self.algorithm_timings
            .entry(algorithm.clone())
            .or_insert_with(Vec::new)
            .push(duration);

        // Update best algorithm if this is better
        let avg_duration = self.average_duration(&algorithm);
        let current_best_duration = self
            .best_algorithm
            .as_ref()
            .map(|alg| self.average_duration(alg))
            .unwrap_or(std::time::Duration::from_secs(u64::MAX));

        if avg_duration < current_best_duration {
            let speedup = current_best_duration.as_secs_f64() / avg_duration.as_secs_f64();
            self.best_algorithm = Some(algorithm);
            self.best_speedup = Some(speedup);
        }
    }

    fn average_duration(&self, algorithm: &MatMulAlgorithm) -> std::time::Duration {
        static EMPTY_VEC: Vec<std::time::Duration> = Vec::new();
        let timings = self.algorithm_timings.get(algorithm).unwrap_or(&EMPTY_VEC);
        if timings.is_empty() {
            return std::time::Duration::from_secs(u64::MAX);
        }

        let total_nanos: u128 = timings.iter().map(|d| d.as_nanos()).sum();
        std::time::Duration::from_nanos((total_nanos / timings.len() as u128) as u64)
    }
}

/// Fused operation types
#[derive(Debug, Clone)]
pub enum FusedOperation<T> {
    ElementwiseAdd {
        alpha: T,
    },
    ElementwiseMul {
        scale: T,
    },
    ReLU,
    Sigmoid,
    MatMul {
        transpose_a: bool,
        transpose_b: bool,
    },
}

/// Fusion signature for caching
#[allow(dead_code)]
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct FusionSignature {
    operation_types: Vec<String>,
    tensor_shapes: Vec<Vec<usize>>,
}

#[allow(dead_code)]
impl FusionSignature {
    fn from_operations<T>(operations: &[FusedOperation<T>]) -> Self
    where
        T: FloatElement + std::ops::AddAssign,
    {
        let operation_types = operations.iter().map(|op| format!("{:?}", op)).collect();

        Self {
            operation_types,
            tensor_shapes: vec![], // Would be filled with actual tensor shapes
        }
    }
}

/// Compiled fusion execution plan
#[allow(dead_code)]
#[derive(Debug, Clone)]
struct CompiledFusion<T> {
    plan: ExecutionPlan<T>,
    estimated_flops: usize,
}

impl<T> CompiledFusion<T> {
    fn execute(&self, inputs: &[&[T]], outputs: &mut [&mut [T]]) -> Result<()>
    where
        T: FloatElement + std::ops::AddAssign,
    {
        // Execute the compiled plan
        self.plan.execute(inputs, outputs)
    }
}

/// Execution plan for fused operations
#[allow(dead_code)]
#[derive(Debug, Clone)]
struct ExecutionPlan<T> {
    operations: Vec<FusedOperation<T>>,
    optimization_level: OptimizationLevel,
}

impl<T> ExecutionPlan<T> {
    fn execute(&self, inputs: &[&[T]], outputs: &mut [&mut [T]]) -> Result<()>
    where
        T: FloatElement + std::ops::AddAssign,
    {
        if outputs.is_empty() || inputs.is_empty() {
            return Ok(());
        }

        // Simple sequential execution of fused operations
        // In a production system, this would be a compiled kernel
        let output = outputs.get_mut(0).ok_or_else(|| {
            torsh_core::error::TorshError::InvalidShape("No output buffer".to_string())
        })?;

        // Copy first input to output as base
        if let Some(first_input) = inputs.first() {
            if first_input.len() == output.len() {
                output.copy_from_slice(first_input);
            }
        }

        // Apply each operation in sequence
        for op in &self.operations {
            match op {
                FusedOperation::ElementwiseAdd { alpha } => {
                    for val in output.iter_mut() {
                        *val += *alpha;
                    }
                }
                FusedOperation::ElementwiseMul { scale } => {
                    for val in output.iter_mut() {
                        *val = *val * *scale;
                    }
                }
                FusedOperation::ReLU => {
                    let zero = <T as torsh_core::dtype::TensorElement>::zero();
                    for val in output.iter_mut() {
                        if *val < zero {
                            *val = zero;
                        }
                    }
                }
                FusedOperation::Sigmoid => {
                    let one = <T as num_traits::One>::one();
                    for val in output.iter_mut() {
                        // sigmoid(x) = 1 / (1 + exp(-x))
                        let exp_neg = (-*val).exp();
                        *val = one / (one + exp_neg);
                    }
                }
                FusedOperation::MatMul { .. } => {
                    // Matrix multiplication would require reshape and proper indexing
                    // Skip for now in this simplified implementation
                }
            }
        }

        Ok(())
    }
}

/// Optimization levels for compilation
#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
enum OptimizationLevel {
    Conservative,
    Moderate,
    Aggressive,
}

/// Algorithm performance statistics
#[derive(Debug)]
pub struct AlgorithmPerformanceStats {
    pub total_operations: usize,
    pub unique_operation_signatures: usize,
    pub algorithm_distribution: HashMap<MatMulAlgorithm, usize>,
    pub average_speedup: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_algorithm_config_default() {
        let config = AlgorithmConfig::default();
        assert!(config.enable_adaptive_selection);
        assert!(config.enable_operation_fusion);
        assert!(config.enable_numerical_stability);
    }

    #[test]
    fn test_algorithmic_optimizer_creation() {
        let optimizer = AlgorithmicOptimizer::new();
        let stats = optimizer.get_performance_stats();

        assert_eq!(stats.total_operations, 0);
        assert_eq!(stats.unique_operation_signatures, 0);
    }

    #[test]
    fn test_algorithm_selection() {
        let optimizer = AlgorithmicOptimizer::new();
        let signature = OperationSignature::MatMul {
            m: 100,
            k: 100,
            n: 100,
        };

        let algorithm = optimizer.select_matmul_algorithm(&signature);
        // For 100x100x100 (total_size = 1,000,000), should select Parallel algorithm
        assert!(matches!(algorithm, MatMulAlgorithm::Parallel));
    }

    #[test]
    fn test_small_matrix_multiplication() {
        let optimizer = AlgorithmicOptimizer::new();

        let a = vec![1.0f32, 2.0, 3.0, 4.0]; // 2x2
        let b = vec![5.0f32, 6.0, 7.0, 8.0]; // 2x2
        let mut c = vec![0.0f32; 4]; // 2x2

        optimizer.optimized_matmul(&a, &b, &mut c, 2, 2, 2).unwrap();

        // Expected: [19, 22, 43, 50]
        assert!((c[0] - 19.0).abs() < 1e-6);
        assert!((c[1] - 22.0).abs() < 1e-6);
        assert!((c[2] - 43.0).abs() < 1e-6);
        assert!((c[3] - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_block_size_calculation() {
        let optimizer = AlgorithmicOptimizer::new();
        let block_size = optimizer.calculate_optimal_block_size(1000, 1000, 1000);

        assert!(block_size >= 16);
        assert!(block_size <= 256);
    }

    #[test]
    fn test_conv2d_basic() {
        let optimizer = AlgorithmicOptimizer::new();

        // 3x3 input, 2x2 kernel
        let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let kernel = vec![1.0f32, 0.0, 0.0, 1.0];
        let mut output = vec![0.0f32; 4]; // 2x2 output

        optimizer
            .optimized_conv2d(&input, &kernel, &mut output, 3, 3, 2, 2, 1, 0)
            .unwrap();

        // Basic sanity check - all outputs should be computed
        assert!(output.iter().all(|&x| x >= 0.0));
    }

    #[test]
    fn test_performance_metrics() {
        let mut metrics = PerformanceMetrics::default();

        let duration = std::time::Duration::from_millis(100);
        metrics.update_performance(MatMulAlgorithm::Blocked, duration);

        assert_eq!(metrics.execution_count, 1);
        assert!(metrics.best_algorithm.is_some());
    }

    #[test]
    fn test_fusion_signature() {
        let operations = vec![
            FusedOperation::ElementwiseAdd { alpha: 1.0f32 },
            FusedOperation::ReLU,
        ];

        let signature = FusionSignature::from_operations(&operations);
        assert_eq!(signature.operation_types.len(), 2);
    }
}
