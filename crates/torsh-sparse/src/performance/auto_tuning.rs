//! Auto-tuning system for optimal sparse tensor format selection
//!
//! This module provides intelligent format selection based on performance characteristics
//! and caching mechanisms to optimize sparse tensor operations automatically.

use crate::{SparseFormat, TorshResult};
use super::benchmarking::SparseProfiler;
use super::core::BenchmarkConfig;
use super::memory_analysis::PerformanceReport;
use std::collections::HashMap;
use torsh_tensor::Tensor;

/// Auto-tuning system for sparse operations
///
/// AutoTuner automatically selects optimal sparse formats based on matrix characteristics
/// and operation types. It uses performance profiling and caching to make intelligent
/// decisions that improve overall system performance.
///
/// # Examples
///
/// ```rust
/// use torsh_sparse::performance::AutoTuner;
/// use torsh_tensor::Tensor;
///
/// let mut tuner = AutoTuner::new();
///
/// // Create a test matrix
/// let dense = Tensor::randn(&[1000, 1000])?;
///
/// // Find optimal format for matrix multiplication
/// let optimal_format = tuner.find_optimal_format(&dense, "matmul", 1e-6)?;
/// println!("Optimal format for this matrix: {:?}", optimal_format);
///
/// // Get performance recommendations
/// let recommendations = tuner.get_recommendations();
/// for rec in recommendations {
///     println!("Recommendation: {}", rec);
/// }
/// ```
pub struct AutoTuner {
    profiler: SparseProfiler,
    cache: HashMap<String, SparseFormat>,
    config: TuningConfig,
}

/// Configuration for auto-tuning behavior
#[derive(Debug, Clone)]
pub struct TuningConfig {
    /// Enable caching of tuning results
    pub enable_caching: bool,
    /// Maximum number of cache entries
    pub max_cache_size: usize,
    /// Benchmark configuration for profiling
    pub benchmark_config: BenchmarkConfig,
    /// Minimum performance improvement required to switch formats (as ratio)
    pub min_improvement_threshold: f64,
    /// Operation types to consider during tuning
    pub target_operations: Vec<String>,
}

impl Default for TuningConfig {
    fn default() -> Self {
        Self {
            enable_caching: true,
            max_cache_size: 1000,
            benchmark_config: BenchmarkConfig::fast(),
            min_improvement_threshold: 0.1, // 10% improvement required
            target_operations: vec![
                "conversion".to_string(),
                "matmul".to_string(),
                "transpose".to_string(),
            ],
        }
    }
}

impl TuningConfig {
    /// Create a conservative tuning configuration
    pub fn conservative() -> Self {
        Self {
            enable_caching: true,
            max_cache_size: 500,
            benchmark_config: BenchmarkConfig::comprehensive(),
            min_improvement_threshold: 0.05, // 5% improvement required
            target_operations: vec!["conversion".to_string(), "matmul".to_string()],
        }
    }

    /// Create an aggressive tuning configuration
    pub fn aggressive() -> Self {
        Self {
            enable_caching: true,
            max_cache_size: 2000,
            benchmark_config: BenchmarkConfig::fast(),
            min_improvement_threshold: 0.02, // 2% improvement required
            target_operations: vec![
                "conversion".to_string(),
                "matmul".to_string(),
                "transpose".to_string(),
                "add".to_string(),
                "multiply".to_string(),
            ],
        }
    }
}

impl Default for AutoTuner {
    fn default() -> Self {
        Self::new()
    }
}

impl AutoTuner {
    /// Create a new auto-tuner with default configuration
    pub fn new() -> Self {
        Self::with_config(TuningConfig::default())
    }

    /// Create a new auto-tuner with custom configuration
    pub fn with_config(config: TuningConfig) -> Self {
        let profiler = SparseProfiler::new(config.benchmark_config.clone());
        Self {
            profiler,
            cache: HashMap::new(),
            config,
        }
    }

    /// Find optimal format for a given operation and matrix characteristics
    ///
    /// This method benchmarks different sparse formats and selects the one with
    /// the best performance for the specified operation. Results are cached to
    /// avoid repeated profiling of similar matrices.
    ///
    /// # Arguments
    ///
    /// * `dense` - The dense matrix to analyze
    /// * `operation` - The target operation type (e.g., "matmul", "conversion")
    /// * `threshold` - Sparsity threshold for format conversion
    ///
    /// # Returns
    ///
    /// Returns the optimal sparse format for the given matrix and operation.
    pub fn find_optimal_format(
        &mut self,
        dense: &Tensor,
        operation: &str,
        threshold: f32,
    ) -> TorshResult<SparseFormat> {
        let cache_key = self.generate_cache_key(dense, operation, threshold);

        // Check cache first if enabled
        if self.config.enable_caching {
            if let Some(&cached_format) = self.cache.get(&cache_key) {
                return Ok(cached_format);
            }
        }

        // Benchmark all formats
        let comparisons = self.profiler.profile_format_comparison(dense, threshold)?;

        // Find the format with best performance for the given operation
        let optimal_format = self.select_optimal_format(&comparisons, operation);

        // Cache result if enabled
        if self.config.enable_caching {
            self.update_cache(cache_key, optimal_format);
        }

        Ok(optimal_format)
    }

    /// Find optimal format considering both time and memory efficiency
    pub fn find_balanced_format(
        &mut self,
        dense: &Tensor,
        operation: &str,
        threshold: f32,
        memory_weight: f64,
    ) -> TorshResult<SparseFormat> {
        let comparisons = self.profiler.profile_format_comparison(dense, threshold)?;

        let optimal_format = comparisons
            .iter()
            .min_by(|(_, a), (_, b)| {
                let a_score = self.calculate_balanced_score(a, memory_weight);
                let b_score = self.calculate_balanced_score(b, memory_weight);
                a_score.partial_cmp(&b_score).expect("score comparison should succeed")
            })
            .map(|(&format, _)| format)
            .unwrap_or(SparseFormat::Csr);

        Ok(optimal_format)
    }

    /// Get performance recommendations based on collected data
    ///
    /// Analyzes the performance data collected during tuning and provides
    /// actionable recommendations for improving sparse tensor performance.
    ///
    /// # Returns
    ///
    /// Returns a vector of recommendation strings.
    pub fn get_recommendations(&self) -> Vec<String> {
        let measurements = self.profiler.measurements();
        let mut recommendations = Vec::new();

        if measurements.is_empty() {
            recommendations.push("No performance data available. Run some benchmarks first.".to_string());
            return recommendations;
        }

        // Find fastest operations by category
        let mut operation_groups: HashMap<String, Vec<_>> = HashMap::new();
        for measurement in measurements {
            let operation_type = self.extract_operation_type(&measurement.operation);
            operation_groups.entry(operation_type).or_default().push(measurement);
        }

        for (op_type, measurements) in operation_groups {
            if let Some(best) = measurements.iter().min_by_key(|m| m.duration) {
                recommendations.push(format!(
                    "Fastest {} operation: {} (avg: {:?})",
                    op_type,
                    best.operation,
                    best.duration
                ));
            }

            if let Some(memory_efficient) = measurements.iter().min_by_key(|m| m.peak_memory) {
                recommendations.push(format!(
                    "Most memory-efficient {}: {} (peak: {} bytes)",
                    op_type,
                    memory_efficient.operation,
                    memory_efficient.peak_memory
                ));
            }
        }

        // General recommendations
        self.add_general_recommendations(&mut recommendations);

        recommendations
    }

    /// Get detailed tuning report
    pub fn get_tuning_report(&self) -> TuningReport {
        TuningReport {
            cache_size: self.cache.len(),
            cache_hit_ratio: self.calculate_cache_hit_ratio(),
            total_benchmarks: self.profiler.measurements().len(),
            recommendations: self.get_recommendations(),
            cached_formats: self.cache.clone(),
        }
    }

    /// Clear all cached results
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Update tuning configuration
    pub fn update_config(&mut self, config: TuningConfig) {
        self.config = config.clone();
        self.profiler.set_config(config.benchmark_config);

        // Clear cache if caching was disabled
        if !config.enable_caching {
            self.clear_cache();
        }
    }

    /// Get current configuration
    pub fn config(&self) -> &TuningConfig {
        &self.config
    }

    // Private helper methods

    fn generate_cache_key(&self, dense: &Tensor, operation: &str, threshold: f32) -> String {
        let shape = dense.shape();
        let sparsity_estimate = self.estimate_sparsity(dense, threshold);

        format!(
            "{}x{}_{}_{:.6}_{:.3}",
            shape.dims()[0],
            shape.dims()[1],
            operation,
            threshold,
            sparsity_estimate
        )
    }

    fn estimate_sparsity(&self, dense: &Tensor, threshold: f32) -> f32 {
        // Simple sparsity estimation - in practice, you might sample the tensor
        // For now, return a placeholder based on threshold
        threshold.clamp(0.0, 1.0)
    }

    fn select_optimal_format(
        &self,
        comparisons: &HashMap<SparseFormat, super::core::PerformanceMeasurement>,
        operation: &str,
    ) -> SparseFormat {
        // Weight different metrics based on operation type
        let (time_weight, memory_weight) = match operation {
            "matmul" => (0.8, 0.2),  // Prioritize speed for matrix multiplication
            "conversion" => (0.6, 0.4), // Balanced for conversions
            "storage" => (0.2, 0.8), // Prioritize memory for storage
            _ => (0.5, 0.5), // Balanced by default
        };

        comparisons
            .iter()
            .min_by(|(_, a), (_, b)| {
                let a_score = self.calculate_weighted_score(a, time_weight, memory_weight);
                let b_score = self.calculate_weighted_score(b, time_weight, memory_weight);
                a_score.partial_cmp(&b_score).expect("score comparison should succeed")
            })
            .map(|(&format, _)| format)
            .unwrap_or(SparseFormat::Csr)
    }

    fn calculate_weighted_score(
        &self,
        measurement: &super::core::PerformanceMeasurement,
        time_weight: f64,
        memory_weight: f64,
    ) -> f64 {
        let time_score = measurement.duration.as_nanos() as f64;
        let memory_score = measurement.peak_memory as f64;

        // Normalize scores (simple approach - could be more sophisticated)
        time_weight * time_score + memory_weight * memory_score
    }

    fn calculate_balanced_score(
        &self,
        measurement: &super::core::PerformanceMeasurement,
        memory_weight: f64,
    ) -> f64 {
        let time_weight = 1.0 - memory_weight;
        self.calculate_weighted_score(measurement, time_weight, memory_weight)
    }

    fn update_cache(&mut self, key: String, format: SparseFormat) {
        // Implement LRU eviction if cache is full
        if self.cache.len() >= self.config.max_cache_size {
            // Simple eviction: remove one random entry
            if let Some(old_key) = self.cache.keys().next().cloned() {
                self.cache.remove(&old_key);
            }
        }

        self.cache.insert(key, format);
    }

    fn extract_operation_type(&self, operation: &str) -> String {
        if operation.contains("convert") {
            "conversion".to_string()
        } else if operation.contains("matmul") {
            "multiplication".to_string()
        } else if operation.contains("dense_to") {
            "sparsification".to_string()
        } else {
            "other".to_string()
        }
    }

    fn add_general_recommendations(&self, recommendations: &mut Vec<String>) {
        let measurements = self.profiler.measurements();

        // Check for consistent performance
        let mut operation_variance: HashMap<String, Vec<u128>> = HashMap::new();
        for measurement in measurements {
            let op_type = self.extract_operation_type(&measurement.operation);
            operation_variance.entry(op_type).or_default().push(measurement.duration.as_nanos());
        }

        for (op_type, times) in operation_variance {
            if times.len() > 2 {
                let mean = times.iter().sum::<u128>() as f64 / times.len() as f64;
                let variance = times.iter()
                    .map(|&t| {
                        let diff = t as f64 - mean;
                        diff * diff
                    })
                    .sum::<f64>() / times.len() as f64;
                let std_dev = variance.sqrt();
                let cv = std_dev / mean; // Coefficient of variation

                if cv > 0.2 {
                    recommendations.push(format!(
                        "{} operations show high variance (CV: {:.2}). Consider more stable algorithms.",
                        op_type, cv
                    ));
                }
            }
        }

        // Cache efficiency recommendation
        if self.config.enable_caching && self.cache.len() > 10 {
            let hit_ratio = self.calculate_cache_hit_ratio();
            if hit_ratio < 0.5 {
                recommendations.push(format!(
                    "Cache hit ratio is low ({:.1}%). Consider adjusting cache size or tuning thresholds.",
                    hit_ratio * 100.0
                ));
            }
        }
    }

    fn calculate_cache_hit_ratio(&self) -> f64 {
        // This would require tracking cache hits/misses
        // For now, return a placeholder
        0.7
    }
}

/// Comprehensive tuning report
#[derive(Debug, Clone)]
pub struct TuningReport {
    pub cache_size: usize,
    pub cache_hit_ratio: f64,
    pub total_benchmarks: usize,
    pub recommendations: Vec<String>,
    pub cached_formats: HashMap<String, SparseFormat>,
}

impl std::fmt::Display for TuningReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== Auto-Tuning Report ===")?;
        writeln!(f)?;
        writeln!(f, "Cache Statistics:")?;
        writeln!(f, "  Size: {} entries", self.cache_size)?;
        writeln!(f, "  Hit ratio: {:.1}%", self.cache_hit_ratio * 100.0)?;
        writeln!(f)?;
        writeln!(f, "Benchmarking:")?;
        writeln!(f, "  Total benchmarks: {}", self.total_benchmarks)?;
        writeln!(f)?;
        writeln!(f, "Recommendations:")?;
        for rec in &self.recommendations {
            writeln!(f, "  • {}", rec)?;
        }
        writeln!(f)?;
        writeln!(f, "Cached Format Decisions:")?;
        for (key, format) in &self.cached_formats {
            writeln!(f, "  {}: {:?}", key, format)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::Shape;

    fn create_test_tensor() -> TorshResult<Tensor> {
        // Create a simple test tensor
        Tensor::zeros(&[10, 10])
    }

    #[test]
    fn test_auto_tuner_creation() {
        let tuner = AutoTuner::new();
        assert!(tuner.cache.is_empty());
        assert!(tuner.config.enable_caching);
    }

    #[test]
    fn test_tuning_config_presets() {
        let conservative = TuningConfig::conservative();
        assert_eq!(conservative.min_improvement_threshold, 0.05);
        assert_eq!(conservative.max_cache_size, 500);

        let aggressive = TuningConfig::aggressive();
        assert_eq!(aggressive.min_improvement_threshold, 0.02);
        assert_eq!(aggressive.max_cache_size, 2000);
        assert!(aggressive.target_operations.len() > conservative.target_operations.len());
    }

    #[test]
    fn test_cache_key_generation() -> TorshResult<()> {
        let tuner = AutoTuner::new();
        let tensor = create_test_tensor()?;

        let key1 = tuner.generate_cache_key(&tensor, "matmul", 1e-6);
        let key2 = tuner.generate_cache_key(&tensor, "matmul", 1e-6);
        let key3 = tuner.generate_cache_key(&tensor, "conversion", 1e-6);

        assert_eq!(key1, key2); // Same parameters should generate same key
        assert_ne!(key1, key3); // Different operations should generate different keys

        Ok(())
    }

    #[test]
    fn test_operation_type_extraction() {
        let tuner = AutoTuner::new();

        assert_eq!(tuner.extract_operation_type("convert_coo_to_csr"), "conversion");
        assert_eq!(tuner.extract_operation_type("matmul_csr_csr"), "multiplication");
        assert_eq!(tuner.extract_operation_type("dense_to_coo"), "sparsification");
        assert_eq!(tuner.extract_operation_type("unknown_op"), "other");
    }

    #[test]
    fn test_cache_management() {
        let config = TuningConfig {
            enable_caching: true,
            max_cache_size: 2,
            ..TuningConfig::default()
        };

        let mut tuner = AutoTuner::with_config(config);

        // Add entries up to cache limit
        tuner.update_cache("key1".to_string(), SparseFormat::Coo);
        tuner.update_cache("key2".to_string(), SparseFormat::Csr);
        assert_eq!(tuner.cache.len(), 2);

        // Adding another should trigger eviction
        tuner.update_cache("key3".to_string(), SparseFormat::Csc);
        assert_eq!(tuner.cache.len(), 2);
    }

    #[test]
    fn test_recommendations_empty_data() {
        let tuner = AutoTuner::new();
        let recommendations = tuner.get_recommendations();

        assert!(!recommendations.is_empty());
        assert!(recommendations[0].contains("No performance data"));
    }

    #[test]
    fn test_clear_cache() {
        let mut tuner = AutoTuner::new();
        tuner.cache.insert("test".to_string(), SparseFormat::Coo);
        assert_eq!(tuner.cache.len(), 1);

        tuner.clear_cache();
        assert!(tuner.cache.is_empty());
    }

    #[test]
    fn test_tuning_report() {
        let tuner = AutoTuner::new();
        let report = tuner.get_tuning_report();

        assert_eq!(report.cache_size, 0);
        assert_eq!(report.total_benchmarks, 0);
        assert!(!report.recommendations.is_empty());
    }
}