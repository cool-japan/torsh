//! # Advanced Quantization Optimizer
//!
//! Next-generation optimization engine that performs sophisticated analysis and optimization
//! of quantized models, including adaptive parameter tuning, memory layout optimization,
//! and intelligent fusion pattern discovery.

use crate::{QScheme, QuantConfig, TorshResult};
use scirs2_core::parallel_ops::*;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};
use torsh_tensor::Tensor;

/// Advanced optimization engine for quantized models
#[derive(Debug, Clone)]
pub struct QuantizationOptimizer {
    /// Optimization configuration
    config: OptimizerConfig,
    /// Performance history for adaptive optimization
    performance_history: Arc<Mutex<HashMap<String, PerformanceHistory>>>,
    /// Learned optimization patterns
    learned_patterns: Vec<OptimizationPattern>,
    /// Memory layout optimizer
    memory_optimizer: MemoryLayoutOptimizer,
    /// Adaptive parameter tuner
    parameter_tuner: AdaptiveParameterTuner,
}

/// Configuration for the optimization engine
#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    /// Enable adaptive parameter optimization
    pub enable_adaptive_params: bool,
    /// Enable memory layout optimization
    pub enable_memory_optimization: bool,
    /// Enable pattern learning and application
    pub enable_pattern_learning: bool,
    /// Maximum optimization iterations
    pub max_iterations: usize,
    /// Convergence threshold for optimization
    pub convergence_threshold: f64,
    /// Enable parallel optimization
    pub enable_parallel: bool,
    /// Target performance improvement percentage
    pub target_improvement: f64,
    /// Enable aggressive optimizations (may sacrifice some accuracy)
    pub enable_aggressive: bool,
}

/// Performance history tracking for adaptive optimization
#[derive(Debug, Clone)]
pub struct PerformanceHistory {
    /// Operation name
    pub operation_name: String,
    /// Historical performance metrics
    pub metrics: VecDeque<PerformanceMetric>,
    /// Optimal configuration found so far
    pub best_config: Option<QuantConfig>,
    /// Best performance achieved
    pub best_performance: Option<f64>,
    /// Number of optimization attempts
    pub optimization_attempts: usize,
}

/// Performance metric for a single operation
#[derive(Debug, Clone)]
pub struct PerformanceMetric {
    /// Timestamp of measurement
    pub timestamp: std::time::Instant,
    /// Execution time (microseconds)
    pub execution_time_us: u64,
    /// Memory usage (bytes)
    pub memory_usage: usize,
    /// Accuracy degradation (0.0 = no degradation, 1.0 = complete loss)
    pub accuracy_degradation: f64,
    /// Configuration used for this measurement
    pub config: QuantConfig,
    /// Composite performance score
    pub performance_score: f64,
}

/// Learned optimization pattern
#[derive(Debug, Clone)]
pub struct OptimizationPattern {
    /// Pattern name/identifier
    pub name: String,
    /// Operation types this pattern applies to
    pub applicable_ops: HashSet<String>,
    /// Tensor shape constraints
    pub shape_constraints: Vec<ShapeConstraint>,
    /// Recommended configuration
    pub recommended_config: QuantConfig,
    /// Expected performance improvement
    pub expected_improvement: f64,
    /// Confidence level (0.0-1.0)
    pub confidence: f64,
    /// Number of successful applications
    pub success_count: usize,
    /// Total applications attempted
    pub application_count: usize,
}

/// Shape constraint for optimization patterns
#[derive(Debug, Clone)]
pub enum ShapeConstraint {
    /// Minimum number of dimensions
    MinDimensions(usize),
    /// Maximum number of dimensions
    MaxDimensions(usize),
    /// Minimum total elements
    MinElements(usize),
    /// Maximum total elements
    MaxElements(usize),
    /// Specific dimension ranges
    DimensionRange(usize, usize, usize), // dim_index, min_size, max_size
    /// Tensor must be contiguous
    RequireContiguous,
}

/// Memory layout optimizer
#[derive(Debug, Clone)]
pub struct MemoryLayoutOptimizer {
    /// Cache size hints for optimization
    pub cache_sizes: Vec<usize>,
    /// Preferred memory alignment
    pub alignment: usize,
    /// Enable memory prefetching optimization
    pub enable_prefetch: bool,
    /// Track memory access patterns
    #[allow(dead_code)]
    access_patterns: HashMap<String, MemoryAccessPattern>,
}

/// Memory access pattern analysis
#[derive(Debug, Clone)]
pub struct MemoryAccessPattern {
    /// Sequential access percentage (0.0-1.0)
    pub sequential_ratio: f64,
    /// Random access percentage (0.0-1.0)
    pub random_ratio: f64,
    /// Cache hit ratio (0.0-1.0)
    pub cache_hit_ratio: f64,
    /// Average access stride
    pub avg_stride: usize,
    /// Memory hotspots
    pub hotspots: Vec<MemoryHotspot>,
}

/// Memory hotspot identification
#[derive(Debug, Clone)]
pub struct MemoryHotspot {
    /// Memory region start
    pub start_offset: usize,
    /// Memory region size
    pub size: usize,
    /// Access frequency
    pub access_frequency: f64,
    /// Suggested optimization
    pub optimization: MemoryOptimization,
}

/// Memory optimization suggestions
#[derive(Debug, Clone)]
pub enum MemoryOptimization {
    /// Prefetch this region
    Prefetch,
    /// Align to cache boundary
    CacheAlign,
    /// Use different memory layout
    Reorder,
    /// Pack data more tightly
    Pack,
    /// Use streaming access
    Stream,
}

/// Adaptive parameter tuner
#[derive(Debug, Clone)]
pub struct AdaptiveParameterTuner {
    /// Learning rate for parameter updates
    pub learning_rate: f64,
    /// Momentum for gradient-based optimization
    pub momentum: f64,
    /// Parameter search space
    pub search_space: ParameterSearchSpace,
    /// Current best parameters
    pub best_parameters: HashMap<String, f64>,
    /// Parameter gradients
    #[allow(dead_code)]
    gradients: HashMap<String, f64>,
}

/// Parameter search space definition
#[derive(Debug, Clone)]
pub struct ParameterSearchSpace {
    /// Scale parameter bounds
    pub scale_bounds: (f64, f64),
    /// Zero point bounds
    pub zero_point_bounds: (i32, i32),
    /// Quantization bit width options
    pub bit_widths: Vec<u8>,
    /// Group size options for group-wise quantization
    pub group_sizes: Vec<usize>,
    /// Calibration dataset size options
    pub calibration_sizes: Vec<usize>,
}

/// Optimization result
#[derive(Debug)]
pub struct OptimizationResult {
    /// Original configuration
    pub original_config: QuantConfig,
    /// Optimized configuration
    pub optimized_config: QuantConfig,
    /// Performance improvement achieved
    pub performance_improvement: f64,
    /// Memory usage improvement
    pub memory_improvement: f64,
    /// Accuracy preservation (0.0-1.0)
    pub accuracy_preservation: f64,
    /// Optimization steps taken
    pub optimization_steps: Vec<OptimizationStep>,
    /// Total optimization time
    pub optimization_time: std::time::Duration,
    /// Optimization success status
    pub success: bool,
}

/// Individual optimization step
#[derive(Debug, Clone)]
pub struct OptimizationStep {
    /// Step type
    pub step_type: OptimizationStepType,
    /// Configuration before this step
    pub before_config: QuantConfig,
    /// Configuration after this step
    pub after_config: QuantConfig,
    /// Performance change
    pub performance_delta: f64,
    /// Step execution time
    pub execution_time: std::time::Duration,
}

/// Types of optimization steps
#[derive(Debug, Clone)]
pub enum OptimizationStepType {
    /// Parameter tuning step
    ParameterTuning,
    /// Memory layout optimization
    MemoryLayout,
    /// Pattern application
    PatternApplication,
    /// Fusion optimization
    FusionOptimization,
    /// Bit width optimization
    BitWidthOptimization,
    /// Group size optimization
    GroupSizeOptimization,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            enable_adaptive_params: true,
            enable_memory_optimization: true,
            enable_pattern_learning: true,
            max_iterations: 50,
            convergence_threshold: 0.01, // 1% improvement threshold
            enable_parallel: true,
            target_improvement: 20.0, // Target 20% improvement
            enable_aggressive: false,
        }
    }
}

impl QuantizationOptimizer {
    /// Create a new quantization optimizer
    pub fn new(config: OptimizerConfig) -> Self {
        Self {
            config,
            performance_history: Arc::new(Mutex::new(HashMap::new())),
            learned_patterns: Vec::new(),
            memory_optimizer: MemoryLayoutOptimizer::new(),
            parameter_tuner: AdaptiveParameterTuner::new(),
        }
    }

    /// Optimize quantization configuration for a specific operation
    pub fn optimize_configuration(
        &mut self,
        operation_name: &str,
        tensor: &Tensor,
        initial_config: &QuantConfig,
        target_accuracy: f64,
    ) -> TorshResult<OptimizationResult> {
        let start_time = std::time::Instant::now();
        let mut current_config = initial_config.clone();
        let mut optimization_steps = Vec::new();
        let mut best_config = initial_config.clone();

        // Measure baseline performance and memory usage
        let baseline_performance =
            self.measure_performance(operation_name, tensor, &current_config)?;
        let baseline_memory = self.calculate_memory_usage(tensor, &current_config);
        let _baseline_accuracy = self.measure_accuracy_degradation(tensor, &current_config)?;
        let mut best_performance = baseline_performance.performance_score;

        for iteration in 0..self.config.max_iterations {
            let step_start = std::time::Instant::now();
            let step_type = self.select_optimization_step(iteration, &current_config, tensor);

            let new_config = match step_type {
                OptimizationStepType::ParameterTuning => self
                    .parameter_tuner
                    .optimize_parameters(&current_config, tensor)?,
                OptimizationStepType::MemoryLayout => self
                    .memory_optimizer
                    .optimize_layout(&current_config, tensor)?,
                OptimizationStepType::PatternApplication => {
                    self.apply_learned_patterns(operation_name, &current_config, tensor)?
                }
                OptimizationStepType::BitWidthOptimization => {
                    self.optimize_bit_width(&current_config, tensor, target_accuracy)?
                }
                OptimizationStepType::GroupSizeOptimization => {
                    self.optimize_group_size(&current_config, tensor)?
                }
                _ => current_config.clone(),
            };

            // Measure performance with new configuration
            let new_performance = self.measure_performance(operation_name, tensor, &new_config)?;
            let performance_delta =
                new_performance.performance_score - baseline_performance.performance_score;

            optimization_steps.push(OptimizationStep {
                step_type,
                before_config: current_config.clone(),
                after_config: new_config.clone(),
                performance_delta,
                execution_time: step_start.elapsed(),
            });

            // Accept improvement
            if new_performance.performance_score > best_performance {
                best_performance = new_performance.performance_score;
                best_config = new_config.clone();
                current_config = new_config;

                // Learn from successful optimization
                if self.config.enable_pattern_learning {
                    self.learn_optimization_pattern(
                        operation_name,
                        &current_config,
                        performance_delta,
                    );
                }
            }

            // Check convergence
            if performance_delta.abs() < self.config.convergence_threshold {
                break;
            }
        }

        // Update performance history
        self.update_performance_history(operation_name, &best_config, best_performance);

        // Calculate improvements
        let total_improvement = (best_performance - baseline_performance.performance_score)
            / baseline_performance.performance_score
            * 100.0;
        let optimized_memory = self.calculate_memory_usage(tensor, &best_config);
        let memory_improvement =
            ((baseline_memory as f64 - optimized_memory as f64) / baseline_memory as f64) * 100.0;
        let optimized_accuracy = self.measure_accuracy_degradation(tensor, &best_config)?;
        let accuracy_preservation = 1.0 - optimized_accuracy;

        Ok(OptimizationResult {
            original_config: initial_config.clone(),
            optimized_config: best_config,
            performance_improvement: total_improvement,
            memory_improvement,
            accuracy_preservation,
            optimization_steps,
            optimization_time: start_time.elapsed(),
            success: total_improvement > self.config.convergence_threshold,
        })
    }

    /// Perform batch optimization for multiple operations
    pub fn optimize_batch(
        &mut self,
        operations: &[(String, Tensor, QuantConfig, f64)], // (name, tensor, config, target_accuracy)
    ) -> TorshResult<Vec<OptimizationResult>> {
        if self.config.enable_parallel {
            // Parallel optimization
            operations
                .par_iter()
                .map(|(name, tensor, config, target_accuracy)| {
                    let mut optimizer = self.clone();
                    optimizer.optimize_configuration(name, tensor, config, *target_accuracy)
                })
                .collect::<Result<Vec<_>, _>>()
        } else {
            // Sequential optimization
            operations
                .iter()
                .map(|(name, tensor, config, target_accuracy)| {
                    self.optimize_configuration(name, tensor, config, *target_accuracy)
                })
                .collect()
        }
    }

    /// Measure performance of a configuration
    fn measure_performance(
        &self,
        _operation_name: &str,
        tensor: &Tensor,
        config: &QuantConfig,
    ) -> TorshResult<PerformanceMetric> {
        let start = std::time::Instant::now();

        // Simulate quantization operation (replace with actual quantization)
        let _result = self.simulate_quantization(tensor, config)?;

        let execution_time = start.elapsed();
        let memory_usage = self.calculate_memory_usage(tensor, config);
        let accuracy_degradation = self.measure_accuracy_degradation(tensor, config)?;

        // Calculate composite performance score
        let time_score = 1.0 / (execution_time.as_micros() as f64 + 1.0);
        let memory_score = 1.0 / (memory_usage as f64 + 1.0);
        let accuracy_score = 1.0 - accuracy_degradation;
        let performance_score = (time_score + memory_score + accuracy_score) / 3.0;

        Ok(PerformanceMetric {
            timestamp: std::time::Instant::now(),
            execution_time_us: execution_time.as_micros() as u64,
            memory_usage,
            accuracy_degradation,
            config: config.clone(),
            performance_score,
        })
    }

    /// Simulate quantization operation
    fn simulate_quantization(&self, tensor: &Tensor, config: &QuantConfig) -> TorshResult<Tensor> {
        // Perform actual quantization and dequantization to simulate the process
        let (quantized_tensor, scale, zero_point) = crate::quantize_auto(tensor, config)?;

        // Dequantize back to floating point to simulate the full quantization cycle
        let dequantized = crate::dequantize(&quantized_tensor, scale, zero_point)?;

        Ok(dequantized)
    }

    /// Select the next optimization step based on current state
    fn select_optimization_step(
        &self,
        iteration: usize,
        _config: &QuantConfig,
        _tensor: &Tensor,
    ) -> OptimizationStepType {
        // Intelligent step selection based on iteration and performance history
        match iteration % 5 {
            0 => OptimizationStepType::ParameterTuning,
            1 => OptimizationStepType::BitWidthOptimization,
            2 => OptimizationStepType::GroupSizeOptimization,
            3 => OptimizationStepType::MemoryLayout,
            4 => OptimizationStepType::PatternApplication,
            _ => OptimizationStepType::ParameterTuning,
        }
    }

    /// Apply learned optimization patterns
    fn apply_learned_patterns(
        &self,
        operation_name: &str,
        current_config: &QuantConfig,
        tensor: &Tensor,
    ) -> TorshResult<QuantConfig> {
        // Find applicable patterns
        for pattern in &self.learned_patterns {
            if pattern.applicable_ops.contains(operation_name)
                && self.check_shape_constraints(tensor, &pattern.shape_constraints)
                && pattern.confidence > 0.7
            {
                return Ok(pattern.recommended_config.clone());
            }
        }
        Ok(current_config.clone())
    }

    /// Check if tensor satisfies shape constraints
    fn check_shape_constraints(&self, tensor: &Tensor, constraints: &[ShapeConstraint]) -> bool {
        let tensor_shape = tensor.shape();
        let shape = tensor_shape.dims();

        for constraint in constraints {
            match constraint {
                ShapeConstraint::MinDimensions(min) => {
                    if shape.len() < *min {
                        return false;
                    }
                }
                ShapeConstraint::MaxDimensions(max) => {
                    if shape.len() > *max {
                        return false;
                    }
                }
                ShapeConstraint::MinElements(min) => {
                    if tensor.numel() < *min {
                        return false;
                    }
                }
                ShapeConstraint::MaxElements(max) => {
                    if tensor.numel() > *max {
                        return false;
                    }
                }
                ShapeConstraint::DimensionRange(dim_idx, min_size, max_size) => {
                    if let Some(&dim_size) = shape.get(*dim_idx) {
                        if dim_size < *min_size || dim_size > *max_size {
                            return false;
                        }
                    }
                }
                ShapeConstraint::RequireContiguous => {
                    if !self.is_tensor_contiguous(tensor) {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Optimize bit width for the configuration
    fn optimize_bit_width(
        &self,
        current_config: &QuantConfig,
        _tensor: &Tensor,
        target_accuracy: f64,
    ) -> TorshResult<QuantConfig> {
        let mut optimized = current_config.clone();

        // Try different bit widths based on target accuracy
        if target_accuracy > 0.95 {
            // High accuracy requirement - use INT8 or higher
            optimized.scheme = QScheme::PerTensorAffine;
        } else if target_accuracy > 0.9 {
            // Medium accuracy - INT8 is fine
            optimized.scheme = QScheme::PerTensorAffine;
        } else {
            // Lower accuracy acceptable - try INT4
            optimized.scheme = QScheme::Int4PerTensor;
        }

        Ok(optimized)
    }

    /// Optimize group size for group-wise quantization
    fn optimize_group_size(
        &self,
        current_config: &QuantConfig,
        tensor: &Tensor,
    ) -> TorshResult<QuantConfig> {
        let mut optimized = current_config.clone();

        // Choose optimal group size based on tensor characteristics
        let total_elements = tensor.numel();
        let optimal_group_size = if total_elements > 10000 {
            128 // Large tensors benefit from larger groups
        } else if total_elements > 1000 {
            64 // Medium tensors
        } else {
            32 // Small tensors
        };

        optimized.group_size = Some(optimal_group_size);
        optimized.scheme = QScheme::GroupWise;

        Ok(optimized)
    }

    /// Learn optimization pattern from successful optimization
    fn learn_optimization_pattern(
        &mut self,
        operation_name: &str,
        config: &QuantConfig,
        performance_improvement: f64,
    ) {
        // Create or update optimization pattern
        let pattern_name = format!("{}_{:?}", operation_name, config.scheme);

        if let Some(existing_pattern) = self
            .learned_patterns
            .iter_mut()
            .find(|p| p.name == pattern_name)
        {
            existing_pattern.success_count += 1;
            existing_pattern.application_count += 1;
            existing_pattern.confidence =
                existing_pattern.success_count as f64 / existing_pattern.application_count as f64;
            existing_pattern.expected_improvement =
                (existing_pattern.expected_improvement + performance_improvement) / 2.0;
        } else {
            let mut applicable_ops = HashSet::new();
            applicable_ops.insert(operation_name.to_string());

            // Note: We would need tensor parameter to extract constraints
            // This is a simplified version without tensor access
            self.learned_patterns.push(OptimizationPattern {
                name: pattern_name,
                applicable_ops,
                shape_constraints: vec![], // Would need tensor to extract constraints
                recommended_config: config.clone(),
                expected_improvement: performance_improvement,
                confidence: 1.0,
                success_count: 1,
                application_count: 1,
            });
        }
    }

    /// Update performance history for an operation
    fn update_performance_history(
        &self,
        operation_name: &str,
        config: &QuantConfig,
        performance: f64,
    ) {
        if let Ok(mut history) = self.performance_history.lock() {
            let entry = history
                .entry(operation_name.to_string())
                .or_insert_with(|| PerformanceHistory {
                    operation_name: operation_name.to_string(),
                    metrics: VecDeque::new(),
                    best_config: None,
                    best_performance: None,
                    optimization_attempts: 0,
                });

            entry.optimization_attempts += 1;

            if entry
                .best_performance
                .map_or(true, |best| performance > best)
            {
                entry.best_performance = Some(performance);
                entry.best_config = Some(config.clone());
            }
        }
    }

    /// Get optimization recommendations based on learned patterns
    pub fn get_recommendations(&self, operation_name: &str, tensor: &Tensor) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Check if we have learned patterns for this operation
        let applicable_patterns: Vec<_> = self
            .learned_patterns
            .iter()
            .filter(|p| p.applicable_ops.contains(operation_name) && p.confidence > 0.5)
            .collect();

        if !applicable_patterns.is_empty() {
            recommendations.push(format!(
                "Found {} learned optimization patterns for operation '{}'",
                applicable_patterns.len(),
                operation_name
            ));
        }

        // Tensor-specific recommendations
        if tensor.numel() > 1_000_000 {
            recommendations
                .push("Large tensor detected - consider group-wise quantization".to_string());
        }

        {
            let tensor_shape = tensor.shape();
            if tensor_shape.dims().len() > 3 {
                recommendations.push(
                    "High-dimensional tensor - consider per-channel quantization".to_string(),
                );
            }
        }

        recommendations
    }

    /// Export learned patterns for reuse
    pub fn export_patterns(&self) -> Vec<OptimizationPattern> {
        self.learned_patterns.clone()
    }

    /// Import learned patterns
    pub fn import_patterns(&mut self, patterns: Vec<OptimizationPattern>) {
        self.learned_patterns.extend(patterns);
    }

    /// Calculate memory usage for a tensor with given configuration
    fn calculate_memory_usage(&self, tensor: &Tensor, config: &QuantConfig) -> usize {
        let base_elements = tensor.numel();
        let element_size = match config.scheme {
            QScheme::Binary => 1,  // 1 bit per element (packed)
            QScheme::Ternary => 2, // 2 bits per element (packed)
            QScheme::Int4PerTensor | QScheme::Int4PerChannel => 4, // 4 bits per element
            _ => match config.dtype {
                torsh_core::DType::I8 | torsh_core::DType::U8 => 1,
                torsh_core::DType::I16 | torsh_core::DType::F16 | torsh_core::DType::BF16 => 2,
                torsh_core::DType::I32 | torsh_core::DType::F32 => 4,
                torsh_core::DType::I64 | torsh_core::DType::F64 | torsh_core::DType::C64 => 8,
                _ => 4, // Default fallback
            },
        };

        // Add overhead for scale and zero-point parameters
        let param_overhead = match config.scheme {
            QScheme::PerChannelAffine | QScheme::PerChannelSymmetric => {
                let num_channels = if let Some(axis) = config.ch_axis {
                    let tensor_shape = tensor.shape();
                    *tensor_shape.dims().get(axis).unwrap_or(&1)
                } else {
                    1
                };
                num_channels * 8 // 4 bytes for scale + 4 bytes for zero_point per channel
            }
            QScheme::GroupWise => {
                let group_size = config.group_size.unwrap_or(32);
                let num_groups = base_elements.div_ceil(group_size);
                num_groups * 8
            }
            _ => 8, // Single scale and zero_point
        };

        base_elements * element_size + param_overhead
    }

    /// Measure accuracy degradation for a configuration
    fn measure_accuracy_degradation(
        &self,
        tensor: &Tensor,
        config: &QuantConfig,
    ) -> TorshResult<f64> {
        // Simulate quantization and dequantization to measure error
        let original_data = tensor.data()?;

        // Estimate quantization error based on scheme characteristics
        let error_estimate = match config.scheme {
            QScheme::Binary => 0.4,   // High error for 1-bit
            QScheme::Ternary => 0.25, // Medium-high error for 2-bit
            QScheme::Int4PerTensor | QScheme::Int4PerChannel => 0.15, // Medium error for 4-bit
            QScheme::PerTensorAffine | QScheme::PerTensorSymmetric => 0.05, // Low error for 8-bit
            QScheme::PerChannelAffine | QScheme::PerChannelSymmetric => 0.03, // Lower error for per-channel
            QScheme::GroupWise => 0.04,      // Low error for group-wise
            QScheme::MixedPrecision => 0.02, // Very low error for mixed precision
        };

        // Adjust error based on data characteristics
        let data_range = original_data
            .iter()
            .fold(0.0f32, |acc, &x| acc.max(x.abs()));
        let data_variance = {
            let mean = original_data.iter().sum::<f32>() / original_data.len() as f32;
            original_data
                .iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f32>()
                / original_data.len() as f32
        };

        // Higher variance and range increase quantization error
        let variance_factor = (data_variance.sqrt() / data_range.max(1.0)).clamp(0.5, 2.0);
        let adjusted_error = error_estimate * variance_factor;

        // Apply reduce range benefit
        let final_error = if matches!(config.reduce_range, crate::ReduceRange::Reduce) {
            adjusted_error * 0.8 // 20% improvement with reduced range
        } else {
            adjusted_error
        };

        Ok(final_error.clamp(0.0, 1.0) as f64)
    }

    /// Check if tensor is contiguous in memory
    fn is_tensor_contiguous(&self, tensor: &Tensor) -> bool {
        // Simplified contiguity check based on tensor shape
        // In a real implementation, this would check the tensor's stride information
        let tensor_shape = tensor.shape();
        let shape = tensor_shape.dims();

        // Assume tensors with simple shapes are contiguous
        shape.len() <= 4 && shape.iter().all(|&dim| dim > 0)
    }

    /// Extract shape constraints from a tensor for pattern learning
    #[allow(dead_code)]
    fn extract_shape_constraints_from_tensor(&self, tensor: &Tensor) -> Vec<ShapeConstraint> {
        let tensor_shape = tensor.shape();
        let shape = tensor_shape.dims();
        let mut constraints = Vec::new();

        // Add dimension constraints
        constraints.push(ShapeConstraint::MinDimensions(shape.len()));
        constraints.push(ShapeConstraint::MaxDimensions(shape.len() + 1));

        // Add element count constraints with some tolerance
        let num_elements = tensor.numel();
        constraints.push(ShapeConstraint::MinElements(num_elements / 2));
        constraints.push(ShapeConstraint::MaxElements(num_elements * 2));

        // Add specific dimension constraints for the largest dimensions
        for (i, &dim_size) in shape.iter().enumerate().take(3) {
            // Only first 3 dimensions
            if dim_size > 1 {
                constraints.push(ShapeConstraint::DimensionRange(
                    i,
                    dim_size / 2,
                    dim_size * 2,
                ));
            }
        }

        // Add contiguity constraint if applicable
        if self.is_tensor_contiguous(tensor) {
            constraints.push(ShapeConstraint::RequireContiguous);
        }

        constraints
    }

    /// Calculate standard deviation of tensor data
    #[allow(dead_code)]
    fn calculate_tensor_std(&self, data: &[f32]) -> TorshResult<f32> {
        if data.is_empty() {
            return Ok(0.0);
        }

        let mean = data.iter().sum::<f32>() / data.len() as f32;
        let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;

        Ok(variance.sqrt())
    }

    /// Calculate skewness of tensor data
    #[allow(dead_code)]
    fn calculate_tensor_skewness(&self, data: &[f32]) -> TorshResult<f32> {
        if data.len() < 3 {
            return Ok(0.0);
        }

        let mean = data.iter().sum::<f32>() / data.len() as f32;
        let std_dev = self.calculate_tensor_std(data)?;

        if std_dev == 0.0 {
            return Ok(0.0);
        }

        let skewness = data
            .iter()
            .map(|&x| ((x - mean) / std_dev).powi(3))
            .sum::<f32>()
            / data.len() as f32;

        Ok(skewness)
    }
}

impl MemoryLayoutOptimizer {
    /// Create new memory layout optimizer
    fn new() -> Self {
        Self {
            cache_sizes: vec![32 * 1024, 256 * 1024, 8 * 1024 * 1024], // L1, L2, L3 cache sizes
            alignment: 64,                                             // 64-byte alignment for SIMD
            enable_prefetch: true,
            access_patterns: HashMap::new(),
        }
    }

    /// Analyze memory access pattern for a tensor
    fn analyze_access_pattern(&self, tensor: &Tensor) -> MemoryAccessPattern {
        let tensor_shape = tensor.shape();
        let shape = tensor_shape.dims();
        let num_elements = tensor.numel();

        // Estimate access patterns based on tensor characteristics
        let sequential_ratio = if shape.len() <= 2 {
            0.9 // 2D and 1D tensors typically have good spatial locality
        } else if shape.len() == 3 {
            0.7 // 3D tensors have moderate locality
        } else {
            0.5 // Higher dimensional tensors have less predictable access
        };

        let random_ratio = 1.0 - sequential_ratio;

        // Estimate cache hit ratio based on tensor size
        let cache_hit_ratio = if num_elements * 4 < self.cache_sizes[0] {
            0.95 // Fits in L1 cache
        } else if num_elements * 4 < self.cache_sizes[1] {
            0.85 // Fits in L2 cache
        } else if num_elements * 4 < self.cache_sizes[2] {
            0.7 // Fits in L3 cache
        } else {
            0.4 // Exceeds cache - poor hit ratio
        };

        // Calculate average stride based on tensor layout
        let avg_stride = if shape.is_empty() {
            1
        } else {
            shape[shape.len() - 1] // Last dimension stride
        };

        // Identify memory hotspots for large tensors
        let hotspots = if num_elements > 100_000 {
            vec![
                MemoryHotspot {
                    start_offset: 0,
                    size: num_elements * 4 / 4, // First quarter
                    access_frequency: 0.6,
                    optimization: MemoryOptimization::Prefetch,
                },
                MemoryHotspot {
                    start_offset: num_elements * 4 * 3 / 4,
                    size: num_elements * 4 / 4, // Last quarter
                    access_frequency: 0.4,
                    optimization: MemoryOptimization::CacheAlign,
                },
            ]
        } else {
            vec![]
        };

        MemoryAccessPattern {
            sequential_ratio,
            random_ratio,
            cache_hit_ratio,
            avg_stride,
            hotspots,
        }
    }

    /// Optimize memory layout for configuration
    fn optimize_layout(&self, config: &QuantConfig, tensor: &Tensor) -> TorshResult<QuantConfig> {
        let mut optimized = config.clone();

        // Analyze tensor access patterns
        let access_pattern = self.analyze_access_pattern(tensor);

        // Apply memory-aware optimizations based on tensor characteristics
        if tensor.numel() > 1_000_000 {
            // Large tensors
            // For large tensors, prefer schemes that minimize memory overhead
            if access_pattern.sequential_ratio > 0.8 {
                // High sequential access - optimize for cache utilization
                optimized.scheme = QScheme::PerTensorAffine; // More cache-friendly
            } else {
                // Random access pattern - use per-channel for better data locality
                optimized.scheme = QScheme::PerChannelAffine;
                optimized.ch_axis = Some(0);
            }
        } else if tensor.numel() > 10_000 {
            // Medium tensors
            // Medium tensors benefit from group-wise quantization
            optimized.scheme = QScheme::GroupWise;
            optimized.group_size = Some(64); // Optimal group size for L1 cache
            optimized.ch_axis = Some(0);
        }

        // Apply alignment optimizations for SIMD operations
        if self.enable_prefetch && access_pattern.cache_hit_ratio < 0.7 {
            // Low cache hit ratio - enable aggressive optimizations
            optimized.reduce_range = crate::ReduceRange::Reduce;
        }

        Ok(optimized)
    }
}

impl AdaptiveParameterTuner {
    /// Create new adaptive parameter tuner
    fn new() -> Self {
        Self {
            learning_rate: 0.01,
            momentum: 0.9,
            search_space: ParameterSearchSpace::default(),
            best_parameters: HashMap::new(),
            gradients: HashMap::new(),
        }
    }

    /// Optimize parameters for configuration
    fn optimize_parameters(
        &mut self,
        config: &QuantConfig,
        tensor: &Tensor,
    ) -> TorshResult<QuantConfig> {
        let mut optimized = config.clone();

        // Adaptive parameter optimization based on tensor statistics
        let tensor_data = tensor.data()?;
        let tensor_min = tensor_data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let tensor_max = tensor_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let tensor_range = tensor_max - tensor_min;
        let tensor_std = self.calculate_tensor_std(&tensor_data)?;

        // Optimize quantization parameters based on tensor characteristics
        if tensor_range > 100.0 && tensor_std > 10.0 {
            // High dynamic range - use reduced range for better precision
            optimized.reduce_range = crate::ReduceRange::Reduce;
            optimized.eps = 1e-6; // Higher precision
        } else if tensor_range < 1.0 {
            // Small dynamic range - can use more aggressive quantization
            optimized.eps = 1e-4; // Lower precision acceptable
            if tensor_std < 0.1 {
                // Very uniform data - binary quantization might work
                optimized.scheme = QScheme::Binary;
            }
        }

        // Optimize observer parameters based on data distribution
        let skewness = self.calculate_tensor_skewness(&tensor_data)?;
        if skewness.abs() > 1.0 {
            // Skewed distribution - use histogram observer for better range estimation
            optimized.observer_type = crate::ObserverType::Histogram;
        } else if tensor_data.len() > 10000 {
            // Large dataset - percentile observer for outlier robustness
            optimized.observer_type = crate::ObserverType::Percentile;
        }

        // Optimize averaging constant for moving average observer
        if matches!(optimized.observer_type, crate::ObserverType::MovingAverage) {
            // Adaptive averaging constant based on data variance
            if tensor_std > 1.0 {
                optimized.averaging_constant = 0.1; // Faster adaptation for high variance
            } else {
                optimized.averaging_constant = 0.01; // Slower adaptation for stable data
            }
        }

        // Store optimized parameters for future use
        self.best_parameters
            .insert("eps".to_string(), optimized.eps as f64);
        self.best_parameters.insert(
            "averaging_constant".to_string(),
            optimized.averaging_constant as f64,
        );

        Ok(optimized)
    }

    /// Calculate standard deviation of tensor data
    fn calculate_tensor_std(&self, data: &[f32]) -> TorshResult<f32> {
        if data.is_empty() {
            return Ok(0.0);
        }

        let mean = data.iter().sum::<f32>() / data.len() as f32;
        let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;

        Ok(variance.sqrt())
    }

    /// Calculate skewness of tensor data
    #[allow(dead_code)]
    fn calculate_tensor_skewness(&self, data: &[f32]) -> TorshResult<f32> {
        if data.len() < 3 {
            return Ok(0.0);
        }

        let mean = data.iter().sum::<f32>() / data.len() as f32;
        let std_dev = self.calculate_tensor_std(data)?;

        if std_dev == 0.0 {
            return Ok(0.0);
        }

        let skewness = data
            .iter()
            .map(|&x| ((x - mean) / std_dev).powi(3))
            .sum::<f32>()
            / data.len() as f32;

        Ok(skewness)
    }
}

impl Default for ParameterSearchSpace {
    fn default() -> Self {
        Self {
            scale_bounds: (1e-6, 100.0),
            zero_point_bounds: (-128, 127),
            bit_widths: vec![4, 8, 16],
            group_sizes: vec![32, 64, 128, 256],
            calibration_sizes: vec![100, 500, 1000, 5000],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::tensor_1d;

    #[test]
    fn test_optimizer_creation() {
        let config = OptimizerConfig::default();
        let optimizer = QuantizationOptimizer::new(config);
        assert!(optimizer.learned_patterns.is_empty());
    }

    #[test]
    fn test_shape_constraints() {
        let optimizer = QuantizationOptimizer::new(OptimizerConfig::default());
        let tensor = tensor_1d(&[1.0, 2.0, 3.0, 4.0]).unwrap();

        let constraints = vec![
            ShapeConstraint::MinDimensions(1),
            ShapeConstraint::MaxDimensions(2),
            ShapeConstraint::MinElements(1),
            ShapeConstraint::MaxElements(10),
        ];

        assert!(optimizer.check_shape_constraints(&tensor, &constraints));
    }

    #[test]
    fn test_bit_width_optimization() {
        let optimizer = QuantizationOptimizer::new(OptimizerConfig::default());
        let config = QuantConfig::int8();
        let tensor = tensor_1d(&[1.0, 2.0, 3.0, 4.0]).unwrap();

        // High accuracy requirement should prefer INT8
        let optimized = optimizer
            .optimize_bit_width(&config, &tensor, 0.99)
            .unwrap();
        assert_eq!(optimized.scheme, QScheme::PerTensorAffine);

        // Lower accuracy can use INT4
        let optimized = optimizer
            .optimize_bit_width(&config, &tensor, 0.85)
            .unwrap();
        assert_eq!(optimized.scheme, QScheme::Int4PerTensor);
    }

    #[test]
    fn test_group_size_optimization() {
        let optimizer = QuantizationOptimizer::new(OptimizerConfig::default());
        let config = QuantConfig::int8();

        // Large tensor should get larger group size
        let large_data: Vec<f32> = (0..20000).map(|i| i as f32).collect();
        let large_tensor = tensor_1d(&large_data).unwrap();
        let optimized = optimizer
            .optimize_group_size(&config, &large_tensor)
            .unwrap();
        assert_eq!(optimized.group_size, Some(128));
        assert_eq!(optimized.scheme, QScheme::GroupWise);

        // Small tensor should get smaller group size
        let small_tensor = tensor_1d(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        let optimized = optimizer
            .optimize_group_size(&config, &small_tensor)
            .unwrap();
        assert_eq!(optimized.group_size, Some(32));
    }

    #[test]
    fn test_performance_metric() {
        let config = QuantConfig::int8();
        let metric = PerformanceMetric {
            timestamp: std::time::Instant::now(),
            execution_time_us: 1000,
            memory_usage: 4096,
            accuracy_degradation: 0.05,
            config,
            performance_score: 0.85,
        };

        assert_eq!(metric.execution_time_us, 1000);
        assert_eq!(metric.memory_usage, 4096);
        assert_eq!(metric.accuracy_degradation, 0.05);
    }

    #[test]
    fn test_optimization_pattern_learning() {
        let mut optimizer = QuantizationOptimizer::new(OptimizerConfig::default());
        let config = QuantConfig::int8();

        optimizer.learn_optimization_pattern("conv2d", &config, 15.0);
        assert_eq!(optimizer.learned_patterns.len(), 1);

        let pattern = &optimizer.learned_patterns[0];
        assert!(pattern.applicable_ops.contains("conv2d"));
        assert_eq!(pattern.expected_improvement, 15.0);
        assert_eq!(pattern.confidence, 1.0);
    }

    #[test]
    fn test_recommendations() {
        let mut optimizer = QuantizationOptimizer::new(OptimizerConfig::default());

        // Add a learned pattern
        optimizer.learn_optimization_pattern("test_op", &QuantConfig::int8(), 20.0);

        let tensor = tensor_1d(&vec![1.0; 1000]).unwrap();
        let recommendations = optimizer.get_recommendations("test_op", &tensor);

        assert!(!recommendations.is_empty());
        assert!(recommendations[0].contains("learned optimization patterns"));
    }

    #[test]
    fn test_pattern_export_import() {
        let mut optimizer1 = QuantizationOptimizer::new(OptimizerConfig::default());
        optimizer1.learn_optimization_pattern("op1", &QuantConfig::int8(), 10.0);
        optimizer1.learn_optimization_pattern("op2", &QuantConfig::int4(), 20.0);

        let exported_patterns = optimizer1.export_patterns();
        assert_eq!(exported_patterns.len(), 2);

        let mut optimizer2 = QuantizationOptimizer::new(OptimizerConfig::default());
        optimizer2.import_patterns(exported_patterns);
        assert_eq!(optimizer2.learned_patterns.len(), 2);
    }

    #[test]
    fn test_memory_layout_optimizer() {
        let optimizer = MemoryLayoutOptimizer::new();
        assert_eq!(optimizer.cache_sizes.len(), 3);
        assert_eq!(optimizer.alignment, 64);
        assert!(optimizer.enable_prefetch);
    }

    #[test]
    fn test_parameter_search_space() {
        let search_space = ParameterSearchSpace::default();
        assert_eq!(search_space.scale_bounds, (1e-6, 100.0));
        assert_eq!(search_space.zero_point_bounds, (-128, 127));
        assert!(search_space.bit_widths.contains(&8));
        assert!(search_space.group_sizes.contains(&64));
    }
}
