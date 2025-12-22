//! # Memory Analysis and Compression Tracking
//!
//! This module provides comprehensive memory analysis capabilities for sparse tensor operations,
//! including memory usage tracking, compression ratio analysis, and cache performance evaluation.
//!
//! ## Key Components
//!
//! - **MemoryAnalysis**: Detailed memory usage analysis with compression metrics
//! - **CachePerformanceResult**: Cache efficiency analysis and recommendations
//! - **Memory tracking utilities**: Tools for monitoring memory usage patterns
//!
//! ## Usage Example
//!
//! ```rust
//! use torsh_sparse::performance_tools::memory::{analyze_sparse_memory, track_memory_usage};
//!
//! // Analyze memory characteristics of a sparse tensor
//! let analysis = analyze_sparse_memory(&sparse_tensor)?;
//! println!("Compression ratio: {:.2}x", analysis.compression_ratio);
//!
//! // Track memory usage during operations
//! let tracker = MemoryTracker::new();
//! let usage = tracker.track_operation(|| perform_sparse_operation())?;
//! ```

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use crate::{SparseFormat, SparseTensor, TorshResult};
use std::collections::HashMap;
use std::time::{Duration, Instant};

use super::core::PerformanceMeasurement;

/// Comprehensive memory analysis result for sparse tensors
///
/// This struct provides detailed information about memory usage patterns,
/// compression characteristics, and efficiency metrics for sparse tensor storage.
#[derive(Debug, Clone)]
pub struct MemoryAnalysis {
    /// Sparse format being analyzed
    pub format: SparseFormat,
    /// Number of non-zero elements
    pub nnz: usize,
    /// Estimated memory usage for sparse representation (bytes)
    pub estimated_memory: usize,
    /// Memory usage if stored as dense tensor (bytes)
    pub dense_memory: usize,
    /// Compression ratio (dense_memory / sparse_memory)
    pub compression_ratio: f32,
    /// Memory overhead per non-zero element (bytes)
    pub overhead_per_nnz: f32,
    /// Matrix dimensions (rows, columns)
    pub matrix_dimensions: (usize, usize),
    /// Memory breakdown by component
    pub memory_breakdown: MemoryBreakdown,
    /// Additional memory metrics
    pub metrics: HashMap<String, f64>,
}

/// Detailed breakdown of memory usage by component
#[derive(Debug, Clone)]
pub struct MemoryBreakdown {
    /// Memory used by value storage
    pub values_memory: usize,
    /// Memory used by index storage
    pub indices_memory: usize,
    /// Memory used by structural information (row pointers, etc.)
    pub structure_memory: usize,
    /// Additional metadata memory
    pub metadata_memory: usize,
}

impl MemoryAnalysis {
    /// Create a new memory analysis result
    pub fn new(format: SparseFormat, nnz: usize, matrix_dimensions: (usize, usize)) -> Self {
        Self {
            format,
            nnz,
            estimated_memory: 0,
            dense_memory: 0,
            compression_ratio: 1.0,
            overhead_per_nnz: 0.0,
            matrix_dimensions,
            memory_breakdown: MemoryBreakdown::default(),
            metrics: HashMap::new(),
        }
    }

    /// Calculate compression effectiveness score (0-1, higher is better)
    pub fn compression_effectiveness(&self) -> f32 {
        if self.dense_memory == 0 {
            return 0.0;
        }
        1.0 - (self.estimated_memory as f32 / self.dense_memory as f32)
    }

    /// Get sparsity ratio (percentage of zero elements)
    pub fn sparsity_ratio(&self) -> f32 {
        let total_elements = self.matrix_dimensions.0 * self.matrix_dimensions.1;
        if total_elements == 0 {
            return 0.0;
        }
        1.0 - (self.nnz as f32 / total_elements as f32)
    }

    /// Check if sparse representation is memory efficient
    pub fn is_memory_efficient(&self) -> bool {
        self.compression_ratio > 2.0 // At least 2x compression
    }

    /// Get memory efficiency rating (Poor, Fair, Good, Excellent)
    pub fn memory_efficiency_rating(&self) -> String {
        match self.compression_ratio {
            r if r >= 10.0 => "Excellent".to_string(),
            r if r >= 5.0 => "Good".to_string(),
            r if r >= 2.0 => "Fair".to_string(),
            _ => "Poor".to_string(),
        }
    }

    /// Add a custom memory metric
    pub fn add_metric(&mut self, key: String, value: f64) {
        self.metrics.insert(key, value);
    }
}

impl Default for MemoryBreakdown {
    fn default() -> Self {
        Self {
            values_memory: 0,
            indices_memory: 0,
            structure_memory: 0,
            metadata_memory: 0,
        }
    }
}

impl MemoryBreakdown {
    /// Calculate total memory usage
    pub fn total_memory(&self) -> usize {
        self.values_memory + self.indices_memory + self.structure_memory + self.metadata_memory
    }

    /// Get memory distribution as percentages
    pub fn memory_distribution(&self) -> HashMap<String, f64> {
        let total = self.total_memory() as f64;
        if total == 0.0 {
            return HashMap::new();
        }

        let mut distribution = HashMap::new();
        distribution.insert(
            "values".to_string(),
            (self.values_memory as f64 / total) * 100.0,
        );
        distribution.insert(
            "indices".to_string(),
            (self.indices_memory as f64 / total) * 100.0,
        );
        distribution.insert(
            "structure".to_string(),
            (self.structure_memory as f64 / total) * 100.0,
        );
        distribution.insert(
            "metadata".to_string(),
            (self.metadata_memory as f64 / total) * 100.0,
        );
        distribution
    }
}

/// Cache performance analysis result
#[derive(Debug, Clone)]
pub struct CachePerformanceResult {
    /// Operation being analyzed
    pub operation: String,
    /// Performance measurements during cache analysis
    pub measurements: Vec<PerformanceMeasurement>,
    /// Cache efficiency score (0-1, higher is better)
    pub cache_efficiency_score: f64,
    /// Optimization recommendations
    pub recommendations: Vec<String>,
    /// Cache miss ratio estimation
    pub cache_miss_ratio: f64,
    /// Memory access pattern analysis
    pub access_pattern: MemoryAccessPattern,
}

/// Memory access pattern classification
#[derive(Debug, Clone)]
pub enum MemoryAccessPattern {
    Sequential,
    Random,
    Strided { stride: usize },
    Blocked { block_size: usize },
    Mixed,
}

impl std::fmt::Display for MemoryAccessPattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemoryAccessPattern::Sequential => write!(f, "Sequential"),
            MemoryAccessPattern::Random => write!(f, "Random"),
            MemoryAccessPattern::Strided { stride } => write!(f, "Strided (stride: {})", stride),
            MemoryAccessPattern::Blocked { block_size } => {
                write!(f, "Blocked (block size: {})", block_size)
            }
            MemoryAccessPattern::Mixed => write!(f, "Mixed"),
        }
    }
}

/// Memory usage tracker for monitoring operations
#[derive(Debug)]
pub struct MemoryTracker {
    /// Initial memory usage
    baseline_memory: usize,
    /// Peak memory usage
    peak_memory: usize,
    /// Current memory usage
    current_memory: usize,
    /// Memory samples taken during tracking
    samples: Vec<(Instant, usize)>,
    /// Sampling interval
    sampling_interval: Duration,
}

impl Default for MemoryTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryTracker {
    /// Create a new memory tracker
    pub fn new() -> Self {
        let current_memory = get_current_memory_usage();
        Self {
            baseline_memory: current_memory,
            peak_memory: current_memory,
            current_memory,
            samples: Vec::new(),
            sampling_interval: Duration::from_millis(10),
        }
    }

    /// Track memory usage during operation execution
    pub fn track_operation<F, R>(&mut self, operation: F) -> TorshResult<(R, MemoryUsageResult)>
    where
        F: FnOnce() -> TorshResult<R>,
    {
        // Reset tracking state
        self.reset();

        // Start tracking
        let start_time = Instant::now();
        let start_memory = get_current_memory_usage();
        self.baseline_memory = start_memory;
        self.current_memory = start_memory;
        self.peak_memory = start_memory;

        // Add initial sample
        self.add_sample(start_time, start_memory);

        // Execute operation with periodic memory sampling
        let result = operation()?;

        // Final memory reading
        let end_memory = get_current_memory_usage();
        self.current_memory = end_memory;
        self.add_sample(Instant::now(), end_memory);

        // Analyze results
        let usage_result = MemoryUsageResult {
            baseline_memory: self.baseline_memory,
            peak_memory: self.peak_memory,
            final_memory: end_memory,
            memory_delta: end_memory as i64 - start_memory as i64,
            peak_delta: self.peak_memory.saturating_sub(start_memory),
            samples: self.samples.clone(),
        };

        Ok((result, usage_result))
    }

    /// Reset tracking state
    pub fn reset(&mut self) {
        self.samples.clear();
        self.current_memory = get_current_memory_usage();
        self.baseline_memory = self.current_memory;
        self.peak_memory = self.current_memory;
    }

    /// Add a memory sample
    fn add_sample(&mut self, timestamp: Instant, memory_usage: usize) {
        self.samples.push((timestamp, memory_usage));
        if memory_usage > self.peak_memory {
            self.peak_memory = memory_usage;
        }
    }

    /// Get current memory growth rate (bytes per second)
    pub fn memory_growth_rate(&self) -> f64 {
        if self.samples.len() < 2 {
            return 0.0;
        }

        let first = &self.samples[0];
        let last = &self.samples[self.samples.len() - 1];

        let time_diff = last.0.duration_since(first.0).as_secs_f64();
        let memory_diff = last.1 as i64 - first.1 as i64;

        if time_diff > 0.0 {
            memory_diff as f64 / time_diff
        } else {
            0.0
        }
    }
}

/// Result of memory usage tracking
#[derive(Debug, Clone)]
pub struct MemoryUsageResult {
    /// Memory usage at start of operation
    pub baseline_memory: usize,
    /// Peak memory usage during operation
    pub peak_memory: usize,
    /// Memory usage at end of operation
    pub final_memory: usize,
    /// Net memory change (can be negative)
    pub memory_delta: i64,
    /// Peak memory increase from baseline
    pub peak_delta: usize,
    /// Time-series memory samples
    pub samples: Vec<(Instant, usize)>,
}

impl MemoryUsageResult {
    /// Check if there was a memory leak (final > baseline by significant amount)
    pub fn has_potential_leak(&self, threshold_bytes: usize) -> bool {
        self.memory_delta > threshold_bytes as i64
    }

    /// Get memory efficiency score (0-1, lower peak usage is better)
    pub fn efficiency_score(&self) -> f64 {
        if self.peak_delta == 0 {
            return 1.0;
        }

        // Score based on how much peak exceeded the final delta
        let final_delta = self.memory_delta.max(0) as usize;
        if self.peak_delta <= final_delta {
            1.0
        } else {
            final_delta as f64 / self.peak_delta as f64
        }
    }
}

/// Analyze memory characteristics of a sparse tensor
pub fn analyze_sparse_memory(sparse: &dyn SparseTensor) -> TorshResult<MemoryAnalysis> {
    let format = sparse.format();
    let shape = sparse.shape();
    let nnz = sparse.nnz();

    let mut analysis = MemoryAnalysis::new(format, nnz, (shape.dims()[0], shape.dims()[1]));

    // Calculate memory usage based on format
    analysis.memory_breakdown = calculate_memory_breakdown(sparse)?;
    analysis.estimated_memory = analysis.memory_breakdown.total_memory();

    // Calculate dense memory equivalent
    let element_size = match sparse.dtype() {
        torsh_core::DType::F32 => 4,
        torsh_core::DType::F64 => 8,
        torsh_core::DType::I32 => 4,
        torsh_core::DType::I64 => 8,
        _ => 4, // Default to 4 bytes
    };
    analysis.dense_memory = shape.dims()[0] * shape.dims()[1] * element_size;

    // Calculate derived metrics
    if analysis.estimated_memory > 0 {
        analysis.compression_ratio =
            analysis.dense_memory as f32 / analysis.estimated_memory as f32;
    }

    if nnz > 0 {
        analysis.overhead_per_nnz = analysis.estimated_memory as f32 / nnz as f32;
    }

    // Add format-specific metrics
    analysis.add_metric(
        "sparsity_ratio".to_string(),
        analysis.sparsity_ratio() as f64,
    );
    analysis.add_metric(
        "compression_effectiveness".to_string(),
        analysis.compression_effectiveness() as f64,
    );

    Ok(analysis)
}

/// Calculate memory breakdown for different sparse formats
fn calculate_memory_breakdown(sparse: &dyn SparseTensor) -> TorshResult<MemoryBreakdown> {
    let nnz = sparse.nnz();
    let shape = sparse.shape();

    let element_size = match sparse.dtype() {
        torsh_core::DType::F32 => 4,
        torsh_core::DType::F64 => 8,
        torsh_core::DType::I32 => 4,
        torsh_core::DType::I64 => 8,
        _ => 4,
    };

    let index_size = 4; // Assume 32-bit indices

    let mut breakdown = MemoryBreakdown::default();

    match sparse.format() {
        SparseFormat::Coo => {
            // COO format: values + row_indices + col_indices
            breakdown.values_memory = nnz * element_size;
            breakdown.indices_memory = nnz * 2 * index_size; // row and col indices
            breakdown.structure_memory = 0;
            breakdown.metadata_memory = 32; // Shape and other metadata
        }
        SparseFormat::Csr => {
            // CSR format: values + col_indices + row_ptrs
            breakdown.values_memory = nnz * element_size;
            breakdown.indices_memory = nnz * index_size; // Column indices
            breakdown.structure_memory = (shape.dims()[0] + 1) * index_size; // Row pointers
            breakdown.metadata_memory = 32;
        }
        SparseFormat::Csc => {
            // CSC format: values + row_indices + col_ptrs
            breakdown.values_memory = nnz * element_size;
            breakdown.indices_memory = nnz * index_size; // Row indices
            breakdown.structure_memory = (shape.dims()[1] + 1) * index_size; // Column pointers
            breakdown.metadata_memory = 32;
        }
        SparseFormat::Bsr => {
            // BSR format: blocks + block_indices + block_ptrs
            breakdown.values_memory = nnz * element_size;
            breakdown.indices_memory = nnz * index_size; // Block indices
            breakdown.structure_memory = nnz * index_size; // Block pointers
            breakdown.metadata_memory = 64; // Block size + metadata
        }
        SparseFormat::Dia => {
            // DIA format: data + offsets
            breakdown.values_memory = nnz * element_size;
            breakdown.indices_memory = 0; // Implicit indexing
            breakdown.structure_memory = nnz * index_size; // Diagonal offsets
            breakdown.metadata_memory = 32;
        }
        SparseFormat::Dsr => {
            // DSR format: dynamic sparse row
            breakdown.values_memory = nnz * element_size;
            breakdown.indices_memory = nnz * index_size;
            breakdown.structure_memory = nnz * index_size;
            breakdown.metadata_memory = 32;
        }
        SparseFormat::Ell => {
            // ELL format: column indices + values in fixed-width format
            breakdown.values_memory = nnz * element_size;
            breakdown.indices_memory = nnz * index_size;
            breakdown.structure_memory = 0;
            breakdown.metadata_memory = 32;
        }
        SparseFormat::Rle => {
            // RLE format: run-length encoded
            breakdown.values_memory = nnz * element_size;
            breakdown.indices_memory = nnz * index_size;
            breakdown.structure_memory = nnz * index_size; // Run lengths
            breakdown.metadata_memory = 32;
        }
        SparseFormat::Symmetric => {
            // Symmetric format: stores only upper or lower triangle
            breakdown.values_memory = nnz * element_size;
            breakdown.indices_memory = nnz * index_size;
            breakdown.structure_memory = nnz * index_size; // Triangle structure
            breakdown.metadata_memory = 64; // Symmetry mode + metadata
        }
    }

    Ok(breakdown)
}

/// Benchmark cache performance for sparse operations
pub fn benchmark_cache_performance(
    sparse: &dyn SparseTensor,
    operation_name: String,
) -> TorshResult<CachePerformanceResult> {
    let mut measurements = Vec::new();
    let _cache_metrics: HashMap<String, f64> = HashMap::new();

    // Simulate cache analysis through repeated operations
    for iteration in 0..5 {
        let measurement_name = format!("{}_cache_iteration_{}", operation_name, iteration);
        let mut measurement = PerformanceMeasurement::new(measurement_name);

        let start = Instant::now();

        // Simulate cache-sensitive operation
        simulate_cache_sensitive_operation(sparse)?;

        measurement.duration = start.elapsed();
        measurements.push(measurement);
    }

    // Analyze cache performance
    let cache_efficiency_score = calculate_cache_efficiency(&measurements);
    let cache_miss_ratio = estimate_cache_miss_ratio(sparse);
    let access_pattern = analyze_access_pattern(sparse);
    let recommendations = generate_cache_recommendations(cache_efficiency_score, &access_pattern);

    Ok(CachePerformanceResult {
        operation: operation_name,
        measurements,
        cache_efficiency_score,
        recommendations,
        cache_miss_ratio,
        access_pattern,
    })
}

/// Simulate a cache-sensitive operation for benchmarking
fn simulate_cache_sensitive_operation(sparse: &dyn SparseTensor) -> TorshResult<()> {
    // This is a simplified simulation - in practice, this would perform
    // actual sparse operations with known cache access patterns
    let nnz = sparse.nnz();
    let mut sum = 0.0f64;

    // Simulate memory access patterns based on format
    match sparse.format() {
        SparseFormat::Csr => {
            // CSR has good row-wise cache locality
            for i in 0..nnz.min(1000) {
                sum += (i as f64).sin(); // Simulate some computation
            }
        }
        SparseFormat::Csc => {
            // CSC has good column-wise cache locality
            for i in 0..nnz.min(1000) {
                sum += (i as f64).cos(); // Simulate some computation
            }
        }
        SparseFormat::Coo => {
            // COO has less predictable cache behavior
            for i in 0..nnz.min(1000) {
                sum += (i as f64).tan(); // Simulate some computation
            }
        }
        _ => {
            // Default case for other formats
            for i in 0..nnz.min(1000) {
                sum += (i as f64).sqrt(); // Simulate some computation
            }
        }
    }

    // Prevent optimization
    std::hint::black_box(sum);
    Ok(())
}

/// Calculate cache efficiency score from measurements
fn calculate_cache_efficiency(measurements: &[PerformanceMeasurement]) -> f64 {
    if measurements.len() < 2 {
        return 0.5; // Default middle score
    }

    // Lower variance in execution times suggests better cache performance
    let times: Vec<f64> = measurements
        .iter()
        .map(|m| m.duration.as_secs_f64())
        .collect();
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let variance = times.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / times.len() as f64;
    let coefficient_of_variation = if mean > 0.0 {
        variance.sqrt() / mean
    } else {
        1.0
    };

    // Lower coefficient of variation = higher cache efficiency
    (1.0 - coefficient_of_variation.min(1.0)).max(0.0)
}

/// Estimate cache miss ratio based on sparse tensor characteristics
fn estimate_cache_miss_ratio(sparse: &dyn SparseTensor) -> f64 {
    let shape = sparse.shape();
    let nnz = sparse.nnz();
    let sparsity = 1.0 - (nnz as f64 / (shape.dims()[0] * shape.dims()[1]) as f64);

    // Higher sparsity typically leads to more cache misses
    match sparse.format() {
        SparseFormat::Csr => 0.1 + sparsity * 0.3, // CSR generally has better cache behavior
        SparseFormat::Csc => 0.1 + sparsity * 0.3,
        SparseFormat::Coo => 0.2 + sparsity * 0.5, // COO has less predictable access
        _ => 0.15 + sparsity * 0.4,                // Default case for other formats
    }
}

/// Analyze memory access pattern based on sparse format and structure
fn analyze_access_pattern(sparse: &dyn SparseTensor) -> MemoryAccessPattern {
    let shape = sparse.shape();
    let nnz = sparse.nnz();

    match sparse.format() {
        SparseFormat::Csr => {
            // CSR provides sequential access within rows
            if nnz < shape.dims()[0] * 2 {
                MemoryAccessPattern::Random
            } else {
                MemoryAccessPattern::Sequential
            }
        }
        SparseFormat::Csc => {
            // CSC provides sequential access within columns
            if nnz < shape.dims()[1] * 2 {
                MemoryAccessPattern::Random
            } else {
                MemoryAccessPattern::Sequential
            }
        }
        SparseFormat::Coo => {
            // COO access pattern depends on sorting
            MemoryAccessPattern::Random
        }
        _ => {
            // Default case for other formats
            MemoryAccessPattern::Random
        }
    }
}

/// Generate cache optimization recommendations
fn generate_cache_recommendations(
    efficiency_score: f64,
    access_pattern: &MemoryAccessPattern,
) -> Vec<String> {
    let mut recommendations = Vec::new();

    if efficiency_score < 0.5 {
        recommendations.push("Consider reordering data to improve cache locality".to_string());
    }

    match access_pattern {
        MemoryAccessPattern::Random => {
            recommendations.push(
                "Random access pattern detected - consider data layout optimization".to_string(),
            );
            recommendations.push("Try blocking algorithms to improve spatial locality".to_string());
        }
        MemoryAccessPattern::Sequential => {
            recommendations
                .push("Good sequential access pattern - consider prefetching".to_string());
        }
        MemoryAccessPattern::Strided { stride } => {
            if *stride > 8 {
                recommendations.push(format!(
                    "Large stride ({}) detected - consider data reordering",
                    stride
                ));
            }
        }
        MemoryAccessPattern::Blocked { .. } => {
            recommendations
                .push("Blocked access pattern - ensure block size matches cache line".to_string());
        }
        MemoryAccessPattern::Mixed => {
            recommendations
                .push("Mixed access pattern - consider hybrid optimization strategies".to_string());
        }
    }

    if recommendations.is_empty() {
        recommendations.push("Cache performance appears optimal".to_string());
    }

    recommendations
}

/// Generate comprehensive memory optimization recommendations
///
/// Analyzes memory usage patterns and provides specific recommendations for optimization
/// based on tensor characteristics, access patterns, and system constraints.
pub fn generate_memory_optimization_recommendations(
    memory_analyses: &[MemoryAnalysis],
    total_memory_budget: Option<usize>,
    target_operations: &[String],
) -> Vec<String> {
    let mut recommendations = Vec::new();

    if memory_analyses.is_empty() {
        recommendations.push("No memory analysis data available for recommendations".to_string());
        return recommendations;
    }

    // Analyze compression ratios across all tensors
    let avg_compression_ratio: f32 = memory_analyses
        .iter()
        .map(|analysis| analysis.compression_ratio)
        .sum::<f32>()
        / memory_analyses.len() as f32;

    // Format-specific recommendations
    let format_distribution = get_format_distribution(memory_analyses);

    // Compression ratio analysis
    if avg_compression_ratio < 2.0 {
        recommendations.push(
            "Low compression ratios detected: Consider switching to more memory-efficient sparse formats".to_string()
        );

        // Specific format recommendations
        if format_distribution.get(&SparseFormat::Coo).unwrap_or(&0) > &0 {
            recommendations.push(
                "COO format detected with low compression: Consider CSR/CSC for better memory efficiency".to_string()
            );
        }
    } else if avg_compression_ratio > 10.0 {
        recommendations.push(
            "Excellent compression ratios achieved: Current format choices are optimal".to_string(),
        );
    }

    // Memory overhead analysis
    let high_overhead_count = memory_analyses
        .iter()
        .filter(|analysis| analysis.overhead_per_nnz > 16.0)
        .count();

    if high_overhead_count > 0 {
        let percentage = (high_overhead_count * 100) / memory_analyses.len();
        recommendations.push(format!(
            "High memory overhead detected in {}% of tensors: Consider hybrid sparse formats",
            percentage
        ));

        // Suggest specific optimizations
        recommendations.push(
            "For matrices with mixed sparsity patterns, consider HybridTensor format".to_string(),
        );
    }

    // Memory budget analysis
    if let Some(budget) = total_memory_budget {
        let total_estimated_memory: usize = memory_analyses
            .iter()
            .map(|analysis| analysis.estimated_memory)
            .sum();

        let memory_utilization = (total_estimated_memory as f64) / (budget as f64);

        if memory_utilization > 0.8 {
            recommendations.push(
                "Memory utilization approaching budget limit: Consider more aggressive compression"
                    .to_string(),
            );
            recommendations.push(
                "Recommend enabling memory-efficient storage options or reducing precision"
                    .to_string(),
            );
        } else if memory_utilization < 0.3 {
            recommendations.push(
                "Memory utilization is low: Consider using less compressed formats for better performance".to_string()
            );
        }
    }

    // Operation-specific recommendations
    for operation in target_operations {
        match operation.as_str() {
            "matmul" | "matrix_multiplication" => {
                recommendations.push(
                    "For matrix multiplication: CSR format recommended for sparse-dense operations"
                        .to_string(),
                );
            }
            "transpose" => {
                recommendations.push(
                    "For transpose operations: Consider maintaining both CSR and CSC representations".to_string()
                );
            }
            "element_access" => {
                recommendations.push(
                    "For element access: COO format provides fastest random access".to_string(),
                );
            }
            "reduction" => {
                recommendations.push(
                    "For reduction operations: CSR/CSC formats optimize sequential access patterns"
                        .to_string(),
                );
            }
            _ => {}
        }
    }

    // Advanced optimization recommendations
    let sparsity_variance = calculate_sparsity_variance(memory_analyses);
    if sparsity_variance > 0.1 {
        recommendations.push(
            "High sparsity variance detected: Consider adaptive format selection per tensor"
                .to_string(),
        );
    }

    // Memory fragmentation recommendations
    let total_memory_usage: usize = memory_analyses.iter().map(|a| a.estimated_memory).sum();
    if total_memory_usage > 100 * 1024 * 1024 {
        // > 100MB
        recommendations.push(
            "Large memory usage detected: Consider memory pooling for better allocation efficiency"
                .to_string(),
        );
    }

    recommendations
}

/// Calculate distribution of sparse formats in the analysis
fn get_format_distribution(memory_analyses: &[MemoryAnalysis]) -> HashMap<SparseFormat, usize> {
    let mut distribution = HashMap::new();
    for analysis in memory_analyses {
        *distribution.entry(analysis.format).or_insert(0) += 1;
    }
    distribution
}

/// Calculate variance in sparsity levels across tensors
fn calculate_sparsity_variance(memory_analyses: &[MemoryAnalysis]) -> f32 {
    if memory_analyses.len() < 2 {
        return 0.0;
    }

    let sparsity_levels: Vec<f32> = memory_analyses
        .iter()
        .map(|analysis| {
            let total_elements = analysis.matrix_dimensions.0 * analysis.matrix_dimensions.1;
            1.0 - (analysis.nnz as f32 / total_elements as f32)
        })
        .collect();

    let mean_sparsity = sparsity_levels.iter().sum::<f32>() / sparsity_levels.len() as f32;

    let variance = sparsity_levels
        .iter()
        .map(|&sparsity| (sparsity - mean_sparsity).powi(2))
        .sum::<f32>()
        / sparsity_levels.len() as f32;

    variance.sqrt()
}

/// Get current memory usage (platform-specific implementation)
fn get_current_memory_usage() -> usize {
    // This is a simplified implementation - in practice, this would use
    // platform-specific APIs to get actual memory usage
    #[cfg(target_os = "linux")]
    {
        // Could parse /proc/self/statm or use libc calls
        64 * 1024 * 1024 // 64MB placeholder
    }
    #[cfg(target_os = "macos")]
    {
        // Could use mach_task_basic_info
        64 * 1024 * 1024 // 64MB placeholder
    }
    #[cfg(target_os = "windows")]
    {
        // Could use GetProcessMemoryInfo
        64 * 1024 * 1024 // 64MB placeholder
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        64 * 1024 * 1024 // 64MB placeholder
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CooTensor;
    use torsh_core::Shape;

    fn create_test_sparse_tensor() -> CooTensor {
        // Create a larger, more sparse matrix (100x100 with only 10 non-zero elements)
        // This will have a good compression ratio
        let row_indices = vec![0, 1, 2, 10, 20, 30, 40, 50, 60, 70];
        let col_indices = vec![0, 1, 2, 10, 20, 30, 40, 50, 60, 70];
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let shape = Shape::new(vec![100, 100]);
        CooTensor::new(row_indices, col_indices, values, shape).unwrap()
    }

    #[test]
    fn test_memory_analysis_creation() {
        let analysis = MemoryAnalysis::new(SparseFormat::Coo, 100, (1000, 1000));

        assert_eq!(analysis.format, SparseFormat::Coo);
        assert_eq!(analysis.nnz, 100);
        assert_eq!(analysis.matrix_dimensions, (1000, 1000));
        assert_eq!(analysis.estimated_memory, 0);
        assert_eq!(analysis.compression_ratio, 1.0);
    }

    #[test]
    fn test_memory_analysis_metrics() {
        let mut analysis = MemoryAnalysis::new(SparseFormat::Coo, 100, (1000, 1000));
        analysis.estimated_memory = 1000;
        analysis.dense_memory = 4000000; // 1000*1000*4 bytes
        analysis.compression_ratio =
            analysis.dense_memory as f32 / analysis.estimated_memory as f32;

        assert_eq!(analysis.compression_ratio, 4000.0);
        assert!(analysis.is_memory_efficient());
        assert_eq!(analysis.memory_efficiency_rating(), "Excellent");
        assert_eq!(analysis.sparsity_ratio(), 0.9999); // 99.99% sparse
    }

    #[test]
    fn test_memory_breakdown() {
        let mut breakdown = MemoryBreakdown::default();
        breakdown.values_memory = 400;
        breakdown.indices_memory = 800;
        breakdown.structure_memory = 100;
        breakdown.metadata_memory = 32;

        assert_eq!(breakdown.total_memory(), 1332);

        let distribution = breakdown.memory_distribution();
        assert!((distribution["values"] - 30.03).abs() < 0.1); // 400/1332 ≈ 30.03%
        assert!((distribution["indices"] - 60.06).abs() < 0.1); // 800/1332 ≈ 60.06%
    }

    #[test]
    fn test_memory_tracker() {
        let mut tracker = MemoryTracker::new();

        // Test tracking a simple operation
        let result = tracker.track_operation(|| -> TorshResult<i32> {
            // Simulate some work
            std::thread::sleep(Duration::from_millis(1));
            Ok(42)
        });

        assert!(result.is_ok());
        let (value, usage_result) = result.unwrap();
        assert_eq!(value, 42);
        assert!(usage_result.samples.len() >= 2);
    }

    #[test]
    fn test_memory_usage_result() {
        let result = MemoryUsageResult {
            baseline_memory: 1000,
            peak_memory: 1500,
            final_memory: 1200,
            memory_delta: 200,
            peak_delta: 500,
            samples: Vec::new(),
        };

        assert!(result.has_potential_leak(100)); // 200 > 100, so leak detected
        assert!(!result.has_potential_leak(300)); // 200 < 300, so no leak

        let efficiency = result.efficiency_score();
        assert!((efficiency - 0.4).abs() < 0.01); // 200/500 = 0.4
    }

    #[test]
    fn test_analyze_sparse_memory() {
        let sparse_tensor = create_test_sparse_tensor();
        let analysis = analyze_sparse_memory(&sparse_tensor);

        assert!(analysis.is_ok());
        let analysis = analysis.unwrap();

        assert_eq!(analysis.format, SparseFormat::Coo);
        assert_eq!(analysis.nnz, 10);
        assert_eq!(analysis.matrix_dimensions, (100, 100));
        assert!(analysis.compression_ratio > 1.0);
    }

    #[test]
    fn test_calculate_memory_breakdown_coo() {
        let sparse_tensor = create_test_sparse_tensor();
        let breakdown = calculate_memory_breakdown(&sparse_tensor);

        assert!(breakdown.is_ok());
        let breakdown = breakdown.unwrap();

        // COO format: 10 values * 4 bytes + 10*2 indices * 4 bytes + metadata
        assert_eq!(breakdown.values_memory, 40); // 10 * 4
        assert_eq!(breakdown.indices_memory, 80); // 10 * 2 * 4 (32-bit indices)
        assert_eq!(breakdown.structure_memory, 0);
        assert_eq!(breakdown.metadata_memory, 32);
    }

    #[test]
    fn test_cache_performance_analysis() {
        let sparse_tensor = create_test_sparse_tensor();
        let result = benchmark_cache_performance(&sparse_tensor, "test_operation".to_string());

        assert!(result.is_ok());
        let result = result.unwrap();

        assert_eq!(result.operation, "test_operation");
        assert_eq!(result.measurements.len(), 5);
        assert!(result.cache_efficiency_score >= 0.0 && result.cache_efficiency_score <= 1.0);
        assert!(!result.recommendations.is_empty());
    }

    #[test]
    fn test_memory_access_patterns() {
        use MemoryAccessPattern::*;

        let sequential = Sequential;
        let random = Random;
        let strided = Strided { stride: 4 };
        let blocked = Blocked { block_size: 64 };
        let mixed = Mixed;

        assert_eq!(format!("{}", sequential), "Sequential");
        assert_eq!(format!("{}", random), "Random");
        assert_eq!(format!("{}", strided), "Strided (stride: 4)");
        assert_eq!(format!("{}", blocked), "Blocked (block size: 64)");
        assert_eq!(format!("{}", mixed), "Mixed");
    }

    #[test]
    fn test_cache_efficiency_calculation() {
        let mut measurements = Vec::new();

        // Add measurements with consistent timing (good cache performance)
        for i in 0..5 {
            let mut measurement = PerformanceMeasurement::new(format!("test_{}", i));
            measurement.duration = Duration::from_millis(100); // Consistent timing
            measurements.push(measurement);
        }

        let efficiency = calculate_cache_efficiency(&measurements);
        assert!(efficiency > 0.8); // Should be high due to consistent timing

        // Add measurement with very different timing (poor cache performance)
        let mut outlier = PerformanceMeasurement::new("outlier".to_string());
        outlier.duration = Duration::from_millis(500); // Much slower
        measurements.push(outlier);

        let efficiency_with_outlier = calculate_cache_efficiency(&measurements);
        assert!(efficiency_with_outlier < efficiency); // Should be lower due to variance
    }

    #[test]
    fn test_generate_cache_recommendations() {
        // Test recommendations for poor cache performance
        let recommendations = generate_cache_recommendations(0.3, &MemoryAccessPattern::Random);
        assert!(recommendations.len() >= 2);
        assert!(recommendations.iter().any(|r| r.contains("cache locality")));
        assert!(recommendations
            .iter()
            .any(|r| r.contains("Random access pattern")));

        // Test recommendations for good cache performance
        let recommendations = generate_cache_recommendations(0.8, &MemoryAccessPattern::Sequential);
        assert!(recommendations
            .iter()
            .any(|r| r.contains("prefetching") || r.contains("optimal")));
    }
}
