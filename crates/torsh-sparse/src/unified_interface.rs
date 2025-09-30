//! Unified sparse tensor interfaces to reduce technical debt
//!
//! This module provides a consolidated interface for sparse tensor operations,
//! reducing code duplication and improving maintainability across different formats.

use crate::*;
use std::collections::HashMap;
use std::sync::Arc;
use torsh_core::{DType, DeviceType, Result as TorshResult, Shape};

/// Unified sparse tensor that can hold any sparse format
pub struct UnifiedSparseTensor {
    /// The underlying sparse tensor implementation
    inner: Box<dyn SparseTensor + Send + Sync>,
    /// Cached dense representation for performance
    dense_cache: Option<Arc<torsh_tensor::Tensor>>,
    /// Operation history for debugging and optimization
    operation_history: Vec<OperationRecord>,
    /// Metadata for optimization hints
    metadata: TensorMetadata,
}

/// Record of operations performed on the tensor
#[derive(Debug, Clone)]
pub struct OperationRecord {
    /// Operation name
    pub operation: String,
    /// Input shapes involved
    pub input_shapes: Vec<Shape>,
    /// Output shape
    pub output_shape: Shape,
    /// Execution time in nanoseconds
    pub execution_time_ns: u64,
    /// Memory usage delta in bytes
    pub memory_delta: i64,
    /// Timestamp of operation
    pub timestamp: std::time::Instant,
}

/// Metadata for tensor optimization
#[derive(Debug, Clone)]
pub struct TensorMetadata {
    /// Access pattern statistics
    pub access_patterns: AccessPatterns,
    /// Performance characteristics
    pub performance_hints: PerformanceHints,
    /// Format suitability scores
    pub format_scores: HashMap<SparseFormat, f32>,
    /// Optimization flags
    pub optimization_flags: OptimizationFlags,
}

/// Access pattern statistics for optimization
#[derive(Debug, Clone)]
pub struct AccessPatterns {
    /// Number of random access operations
    pub random_access_count: u64,
    /// Number of sequential access operations
    pub sequential_access_count: u64,
    /// Number of transpose operations
    pub transpose_count: u64,
    /// Number of format conversions
    pub conversion_count: u64,
    /// Most frequently accessed regions
    pub hot_regions: Vec<(usize, usize, usize, usize)>, // (row_start, row_end, col_start, col_end)
}

/// Performance hints for optimization
#[derive(Debug, Clone)]
pub struct PerformanceHints {
    /// Prefer memory-efficient operations
    pub prefer_memory_efficiency: bool,
    /// Prefer computation speed over memory
    pub prefer_speed: bool,
    /// Target device for optimization
    pub target_device: DeviceType,
    /// Expected operation types
    pub expected_operations: Vec<String>,
    /// Sparsity pattern stability
    pub pattern_stable: bool,
}

/// Optimization flags
#[derive(Debug, Clone)]
pub struct OptimizationFlags {
    /// Enable automatic format conversion
    pub auto_format_conversion: bool,
    /// Enable dense caching for frequently accessed tensors
    pub enable_dense_cache: bool,
    /// Enable pattern analysis
    pub enable_pattern_analysis: bool,
    /// Enable operation fusion
    pub enable_operation_fusion: bool,
    /// Maximum memory overhead for optimization (as fraction)
    pub max_memory_overhead: f32,
}

impl Default for TensorMetadata {
    fn default() -> Self {
        Self {
            access_patterns: AccessPatterns {
                random_access_count: 0,
                sequential_access_count: 0,
                transpose_count: 0,
                conversion_count: 0,
                hot_regions: Vec::new(),
            },
            performance_hints: PerformanceHints {
                prefer_memory_efficiency: true,
                prefer_speed: false,
                target_device: DeviceType::Cpu,
                expected_operations: Vec::new(),
                pattern_stable: true,
            },
            format_scores: HashMap::new(),
            optimization_flags: OptimizationFlags {
                auto_format_conversion: true,
                enable_dense_cache: false,
                enable_pattern_analysis: true,
                enable_operation_fusion: false,
                max_memory_overhead: 0.1,
            },
        }
    }
}

impl UnifiedSparseTensor {
    /// Create a new unified sparse tensor
    pub fn new(inner: Box<dyn SparseTensor + Send + Sync>) -> Self {
        let mut metadata = TensorMetadata::default();
        metadata.format_scores.insert(inner.format(), 1.0);

        Self {
            inner,
            dense_cache: None,
            operation_history: Vec::new(),
            metadata,
        }
    }

    /// Create from any sparse tensor type
    pub fn from_sparse<T: SparseTensor + Send + Sync + 'static>(tensor: T) -> Self {
        Self::new(Box::new(tensor))
    }

    /// Get the underlying sparse tensor
    pub fn inner(&self) -> &dyn SparseTensor {
        self.inner.as_ref()
    }

    /// Get mutable access to metadata
    pub fn metadata_mut(&mut self) -> &mut TensorMetadata {
        &mut self.metadata
    }

    /// Get operation history
    pub fn operation_history(&self) -> &[OperationRecord] {
        &self.operation_history
    }

    /// Record an operation
    fn record_operation(
        &mut self,
        operation: String,
        input_shapes: Vec<Shape>,
        output_shape: Shape,
        execution_time: std::time::Duration,
        memory_delta: i64,
    ) {
        let record = OperationRecord {
            operation,
            input_shapes,
            output_shape,
            execution_time_ns: execution_time.as_nanos() as u64,
            memory_delta,
            timestamp: std::time::Instant::now(),
        };
        self.operation_history.push(record);

        // Limit history size to prevent memory bloat
        if self.operation_history.len() > 1000 {
            self.operation_history.drain(0..500);
        }
    }

    /// Get or compute dense representation with caching
    pub fn to_dense_cached(&mut self) -> TorshResult<Arc<torsh_tensor::Tensor>> {
        if let Some(cached) = &self.dense_cache {
            return Ok(cached.clone());
        }

        let start_time = std::time::Instant::now();
        let dense = self.inner.to_dense()?;
        let execution_time = start_time.elapsed();

        let dense_arc = Arc::new(dense);
        if self.metadata.optimization_flags.enable_dense_cache {
            self.dense_cache = Some(dense_arc.clone());
        }

        self.record_operation(
            "to_dense".to_string(),
            vec![self.inner.shape().clone()],
            self.inner.shape().clone(),
            execution_time,
            (dense_arc.shape().numel() * std::mem::size_of::<f32>()) as i64,
        );

        Ok(dense_arc)
    }

    /// Clear dense cache to free memory
    pub fn clear_dense_cache(&mut self) {
        if let Some(cached) = self.dense_cache.take() {
            let memory_freed = cached.shape().numel() * std::mem::size_of::<f32>();
            self.record_operation(
                "clear_cache".to_string(),
                vec![],
                Shape::new(vec![]),
                std::time::Duration::from_nanos(0),
                -(memory_freed as i64),
            );
        }
    }

    /// Convert to optimal format based on access patterns and performance hints
    pub fn optimize_format(&mut self) -> TorshResult<bool> {
        let current_format = self.inner.format();
        let optimal_format = self.suggest_optimal_format()?;

        if current_format == optimal_format {
            return Ok(false); // No conversion needed
        }

        let start_time = std::time::Instant::now();
        let converted = convert_sparse_format(self.inner.as_ref(), optimal_format)?;
        let execution_time = start_time.elapsed();

        // Update metadata
        self.metadata.access_patterns.conversion_count += 1;
        self.metadata.format_scores.insert(optimal_format, 1.0);
        self.metadata.format_scores.insert(current_format, 0.8);

        // Clear dense cache as it's no longer valid
        self.clear_dense_cache();

        // Replace inner tensor
        self.inner = converted;

        self.record_operation(
            format!("format_conversion_{current_format:?}_to_{optimal_format:?}"),
            vec![self.inner.shape().clone()],
            self.inner.shape().clone(),
            execution_time,
            0, // Conversion typically doesn't change memory usage significantly
        );

        Ok(true)
    }

    /// Suggest optimal format based on current metadata
    pub fn suggest_optimal_format(&self) -> TorshResult<SparseFormat> {
        let patterns = &self.metadata.access_patterns;
        let hints = &self.metadata.performance_hints;
        let sparsity = self.inner.sparsity();
        let shape = self.inner.shape();

        // Simple heuristic-based format selection
        let format = if patterns.transpose_count > patterns.random_access_count {
            // Frequent transpose operations favor CSC
            SparseFormat::Csc
        } else if patterns.sequential_access_count > patterns.random_access_count {
            // Sequential access favors CSR
            SparseFormat::Csr
        } else if sparsity > 0.95 {
            // Very sparse matrices work well with COO
            SparseFormat::Coo
        } else if hints.prefer_memory_efficiency {
            // Memory-efficient choice based on shape
            if shape.dims()[0] > shape.dims()[1] {
                SparseFormat::Csr
            } else {
                SparseFormat::Csc
            }
        } else {
            // Default to CSR for general computation
            SparseFormat::Csr
        };

        Ok(format)
    }

    /// Update access patterns based on operation
    pub fn update_access_pattern(
        &mut self,
        operation: &str,
        region: Option<(usize, usize, usize, usize)>,
    ) {
        let patterns = &mut self.metadata.access_patterns;

        match operation {
            "transpose" => patterns.transpose_count += 1,
            "sequential_access" => patterns.sequential_access_count += 1,
            "random_access" => patterns.random_access_count += 1,
            _ => {}
        }

        if let Some(region) = region {
            patterns.hot_regions.push(region);
            // Keep only recent hot regions
            if patterns.hot_regions.len() > 100 {
                patterns.hot_regions.drain(0..50);
            }
        }
    }

    /// Perform matrix multiplication with optimization
    pub fn matmul_optimized(
        &mut self,
        other: &mut UnifiedSparseTensor,
    ) -> TorshResult<UnifiedSparseTensor> {
        let start_time = std::time::Instant::now();

        // Check if format conversion would be beneficial
        let should_convert_self = self
            .metadata
            .format_scores
            .get(&SparseFormat::Csr)
            .unwrap_or(&0.0)
            > &0.8;
        let should_convert_other = other
            .metadata
            .format_scores
            .get(&SparseFormat::Csc)
            .unwrap_or(&0.0)
            > &0.8;

        if should_convert_self && self.inner.format() != SparseFormat::Csr {
            self.optimize_format()?;
        }

        if should_convert_other && other.inner.format() != SparseFormat::Csc {
            other.optimize_format()?;
        }

        // Perform multiplication using ops module
        let result = crate::ops::sparse_matmul(self.inner.as_ref(), other.inner.as_ref())?;
        let execution_time = start_time.elapsed();

        let unified_result = UnifiedSparseTensor::new(Box::new(result));

        // Update metadata based on operation
        self.update_access_pattern("matmul", None);
        other.update_access_pattern("matmul", None);

        // Record operation
        self.record_operation(
            "matmul_optimized".to_string(),
            vec![self.inner.shape().clone(), other.inner.shape().clone()],
            unified_result.inner.shape().clone(),
            execution_time,
            0, // Memory usage would need more sophisticated tracking
        );

        Ok(unified_result)
    }

    /// Get memory usage statistics
    pub fn memory_stats(&self) -> MemoryStats {
        let sparse_memory = self.estimate_sparse_memory();
        let dense_cache_memory = self
            .dense_cache
            .as_ref()
            .map(|cache| cache.shape().numel() * std::mem::size_of::<f32>())
            .unwrap_or(0);
        let metadata_memory = std::mem::size_of::<TensorMetadata>()
            + self.operation_history.len() * std::mem::size_of::<OperationRecord>();

        MemoryStats {
            sparse_memory_bytes: sparse_memory,
            dense_cache_bytes: dense_cache_memory,
            metadata_bytes: metadata_memory,
            total_bytes: sparse_memory + dense_cache_memory + metadata_memory,
            compression_ratio: self.calculate_compression_ratio(),
        }
    }

    /// Estimate memory usage of sparse representation
    fn estimate_sparse_memory(&self) -> usize {
        let nnz = self.inner.nnz();
        match self.inner.format() {
            SparseFormat::Coo => {
                nnz * (2 * std::mem::size_of::<usize>() + std::mem::size_of::<f32>())
            }
            SparseFormat::Csr => {
                let rows = self.inner.shape().dims()[0];
                nnz * (std::mem::size_of::<usize>() + std::mem::size_of::<f32>())
                    + rows * std::mem::size_of::<usize>()
            }
            SparseFormat::Csc => {
                let cols = self.inner.shape().dims()[1];
                nnz * (std::mem::size_of::<usize>() + std::mem::size_of::<f32>())
                    + cols * std::mem::size_of::<usize>()
            }
            _ => nnz * 3 * std::mem::size_of::<f32>(), // Rough estimate for other formats
        }
    }

    /// Calculate compression ratio compared to dense storage
    fn calculate_compression_ratio(&self) -> f32 {
        let dense_memory = self.inner.shape().numel() * std::mem::size_of::<f32>();
        let sparse_memory = self.estimate_sparse_memory();

        if sparse_memory == 0 {
            return f32::INFINITY;
        }

        dense_memory as f32 / sparse_memory as f32
    }

    /// Generate optimization report
    pub fn optimization_report(&self) -> OptimizationReport {
        let memory_stats = self.memory_stats();
        let performance_summary = self.analyze_performance();

        OptimizationReport {
            current_format: self.inner.format(),
            suggested_format: self.suggest_optimal_format().unwrap_or(self.inner.format()),
            memory_stats,
            performance_summary,
            access_pattern_summary: self.metadata.access_patterns.clone(),
            optimization_recommendations: self.generate_recommendations(),
        }
    }

    /// Analyze performance from operation history
    fn analyze_performance(&self) -> PerformanceSummary {
        if self.operation_history.is_empty() {
            return PerformanceSummary::default();
        }

        let total_ops = self.operation_history.len();
        let total_time_ns: u64 = self
            .operation_history
            .iter()
            .map(|op| op.execution_time_ns)
            .sum();
        let avg_time_ns = total_time_ns / total_ops as u64;

        let conversion_ops = self
            .operation_history
            .iter()
            .filter(|op| op.operation.contains("conversion"))
            .count();

        let conversion_ratio = conversion_ops as f32 / total_ops as f32;

        PerformanceSummary {
            total_operations: total_ops,
            average_operation_time_ns: avg_time_ns,
            total_execution_time_ns: total_time_ns,
            format_conversion_ratio: conversion_ratio,
            memory_efficiency_score: self.calculate_memory_efficiency_score(),
        }
    }

    /// Calculate memory efficiency score (0.0 to 1.0)
    fn calculate_memory_efficiency_score(&self) -> f32 {
        let compression_ratio = self.calculate_compression_ratio();
        let cache_overhead = if self.dense_cache.is_some() { 0.5 } else { 0.0 };

        // Score based on compression ratio and cache usage
        let base_score = (compression_ratio.log10() / 3.0).clamp(0.0, 1.0);
        (base_score - cache_overhead).max(0.0)
    }

    /// Generate optimization recommendations
    fn generate_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        let memory_stats = self.memory_stats();
        let patterns = &self.metadata.access_patterns;

        // Format recommendations
        if patterns.conversion_count > 5 {
            recommendations.push(
                "Consider using auto format conversion to reduce conversion overhead".to_string(),
            );
        }

        if patterns.transpose_count > patterns.random_access_count {
            recommendations
                .push("Consider using CSC format for frequent transpose operations".to_string());
        }

        // Memory recommendations
        if memory_stats.compression_ratio < 2.0 {
            recommendations.push(
                "Sparse representation may not be memory efficient for this matrix".to_string(),
            );
        }

        if self.dense_cache.is_some()
            && memory_stats.dense_cache_bytes > memory_stats.sparse_memory_bytes * 2
        {
            recommendations
                .push("Dense cache is using significant memory; consider disabling it".to_string());
        }

        // Performance recommendations
        let performance = self.analyze_performance();
        if performance.format_conversion_ratio > 0.3 {
            recommendations.push(
                "High format conversion ratio detected; consider optimizing format selection"
                    .to_string(),
            );
        }

        if recommendations.is_empty() {
            recommendations.push("Tensor appears to be well-optimized".to_string());
        }

        recommendations
    }
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub sparse_memory_bytes: usize,
    pub dense_cache_bytes: usize,
    pub metadata_bytes: usize,
    pub total_bytes: usize,
    pub compression_ratio: f32,
}

/// Performance analysis summary
#[derive(Debug, Clone)]
pub struct PerformanceSummary {
    pub total_operations: usize,
    pub average_operation_time_ns: u64,
    pub total_execution_time_ns: u64,
    pub format_conversion_ratio: f32,
    pub memory_efficiency_score: f32,
}

impl Default for PerformanceSummary {
    fn default() -> Self {
        Self {
            total_operations: 0,
            average_operation_time_ns: 0,
            total_execution_time_ns: 0,
            format_conversion_ratio: 0.0,
            memory_efficiency_score: 1.0,
        }
    }
}

/// Comprehensive optimization report
#[derive(Debug, Clone)]
pub struct OptimizationReport {
    pub current_format: SparseFormat,
    pub suggested_format: SparseFormat,
    pub memory_stats: MemoryStats,
    pub performance_summary: PerformanceSummary,
    pub access_pattern_summary: AccessPatterns,
    pub optimization_recommendations: Vec<String>,
}

/// Implement SparseTensor trait for UnifiedSparseTensor
impl SparseTensor for UnifiedSparseTensor {
    fn format(&self) -> SparseFormat {
        self.inner.format()
    }

    fn shape(&self) -> &Shape {
        self.inner.shape()
    }

    fn dtype(&self) -> DType {
        self.inner.dtype()
    }

    fn device(&self) -> DeviceType {
        self.inner.device()
    }

    fn nnz(&self) -> usize {
        self.inner.nnz()
    }

    fn to_dense(&self) -> TorshResult<torsh_tensor::Tensor> {
        self.inner.to_dense()
    }

    fn to_coo(&self) -> TorshResult<CooTensor> {
        self.inner.to_coo()
    }

    fn to_csr(&self) -> TorshResult<CsrTensor> {
        self.inner.to_csr()
    }

    fn to_csc(&self) -> TorshResult<CscTensor> {
        self.inner.to_csc()
    }

    fn sparsity(&self) -> f32 {
        self.inner.sparsity()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Factory for creating unified sparse tensors with optimal configuration
pub struct UnifiedSparseTensorFactory;

impl UnifiedSparseTensorFactory {
    /// Create optimized unified tensor from dense tensor
    pub fn from_dense_optimized(
        dense: &torsh_tensor::Tensor,
        config: OptimizationConfig,
    ) -> TorshResult<UnifiedSparseTensor> {
        let optimal_format = Self::analyze_dense_for_format(dense, &config)?;
        let sparse = sparse_from_dense(dense, optimal_format, Some(config.sparsity_threshold))?;

        let mut unified = UnifiedSparseTensor::new(sparse);

        // Configure based on config
        unified
            .metadata_mut()
            .optimization_flags
            .auto_format_conversion = config.enable_auto_conversion;
        unified.metadata_mut().optimization_flags.enable_dense_cache = config.enable_dense_cache;
        unified
            .metadata_mut()
            .optimization_flags
            .enable_pattern_analysis = config.enable_pattern_analysis;
        unified
            .metadata_mut()
            .optimization_flags
            .max_memory_overhead = config.max_memory_overhead;

        unified
            .metadata_mut()
            .performance_hints
            .prefer_memory_efficiency = config.prefer_memory_efficiency;
        unified.metadata_mut().performance_hints.prefer_speed = config.prefer_speed;
        unified.metadata_mut().performance_hints.target_device = config.target_device;

        Ok(unified)
    }

    /// Analyze dense tensor to suggest optimal sparse format
    fn analyze_dense_for_format(
        dense: &torsh_tensor::Tensor,
        config: &OptimizationConfig,
    ) -> TorshResult<SparseFormat> {
        let shape = dense.shape();
        let sparsity = Self::calculate_sparsity(dense, config.sparsity_threshold)?;

        // Simple heuristic-based selection
        let format = if sparsity > 0.95 {
            SparseFormat::Coo
        } else if config.prefer_memory_efficiency {
            if shape.dims()[0] > shape.dims()[1] {
                SparseFormat::Csr
            } else {
                SparseFormat::Csc
            }
        } else {
            SparseFormat::Csr // Default for performance
        };

        Ok(format)
    }

    /// Calculate sparsity of dense tensor
    fn calculate_sparsity(_dense: &torsh_tensor::Tensor, _threshold: f32) -> TorshResult<f32> {
        // This would need implementation based on torsh_tensor's actual API
        // For now, return a placeholder
        Ok(0.8) // Placeholder
    }
}

/// Configuration for optimization
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    pub sparsity_threshold: f32,
    pub enable_auto_conversion: bool,
    pub enable_dense_cache: bool,
    pub enable_pattern_analysis: bool,
    pub max_memory_overhead: f32,
    pub prefer_memory_efficiency: bool,
    pub prefer_speed: bool,
    pub target_device: DeviceType,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            sparsity_threshold: 0.0,
            enable_auto_conversion: true,
            enable_dense_cache: false,
            enable_pattern_analysis: true,
            max_memory_overhead: 0.1,
            prefer_memory_efficiency: true,
            prefer_speed: false,
            target_device: DeviceType::Cpu,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coo::CooTensor;
    use torsh_core::{DType, Shape};

    #[test]
    fn test_unified_tensor_creation() {
        let shape = Shape::new(vec![3, 3]);
        let mut coo = CooTensor::empty(shape, DType::F32).unwrap();
        coo.insert(0, 0, 1.0).unwrap();
        coo.insert(1, 1, 2.0).unwrap();
        coo.insert(2, 2, 3.0).unwrap();

        let unified = UnifiedSparseTensor::from_sparse(coo);
        assert_eq!(unified.format(), SparseFormat::Coo);
        assert_eq!(unified.nnz(), 3);
    }

    #[test]
    fn test_format_optimization() {
        let shape = Shape::new(vec![3, 3]);
        let mut coo = CooTensor::empty(shape, DType::F32).unwrap();
        coo.insert(0, 0, 1.0).unwrap();
        coo.insert(1, 1, 2.0).unwrap();

        let mut unified = UnifiedSparseTensor::from_sparse(coo);

        // Simulate access patterns that would favor CSR
        unified.update_access_pattern("sequential_access", None);
        unified.update_access_pattern("sequential_access", None);

        // Check optimization suggestion
        let suggested = unified.suggest_optimal_format().unwrap();
        assert_eq!(suggested, SparseFormat::Csr);
    }

    #[test]
    fn test_memory_stats() {
        // Create a larger sparse matrix for better compression ratio
        let shape = Shape::new(vec![10, 10]);
        let mut coo = CooTensor::empty(shape, DType::F32).unwrap();
        coo.insert(0, 0, 1.0).unwrap();
        coo.insert(9, 9, 2.0).unwrap();

        let unified = UnifiedSparseTensor::from_sparse(coo);
        let stats = unified.memory_stats();

        assert!(stats.sparse_memory_bytes > 0);
        assert_eq!(stats.dense_cache_bytes, 0); // No cache by default
        assert!(stats.compression_ratio > 1.0); // 400 bytes dense vs ~24 bytes sparse
    }

    #[test]
    fn test_optimization_report() {
        let shape = Shape::new(vec![2, 2]);
        let mut coo = CooTensor::empty(shape, DType::F32).unwrap();
        coo.insert(0, 0, 1.0).unwrap();

        let unified = UnifiedSparseTensor::from_sparse(coo);
        let report = unified.optimization_report();

        assert_eq!(report.current_format, SparseFormat::Coo);
        assert!(!report.optimization_recommendations.is_empty());
    }

    #[test]
    fn test_operation_recording() {
        let shape = Shape::new(vec![2, 2]);
        let mut coo = CooTensor::empty(shape.clone(), DType::F32).unwrap();
        coo.insert(0, 0, 1.0).unwrap();

        let mut unified = UnifiedSparseTensor::from_sparse(coo);

        // This would normally be called internally during operations
        unified.record_operation(
            "test_op".to_string(),
            vec![shape.clone()],
            shape.clone(),
            std::time::Duration::from_millis(1),
            100,
        );

        assert_eq!(unified.operation_history().len(), 1);
        assert_eq!(unified.operation_history()[0].operation, "test_op");
    }

    #[test]
    fn test_optimization_config() {
        let config = OptimizationConfig::default();
        assert!(config.enable_auto_conversion);
        assert!(config.prefer_memory_efficiency);
        assert_eq!(config.target_device, DeviceType::Cpu);
    }
}
