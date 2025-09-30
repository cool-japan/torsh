//! Adaptive kernel selection based on input characteristics
//!
//! This module provides intelligent kernel selection that adapts to input characteristics,
//! system state, and historical performance data to automatically choose optimal kernels
//! for different operations.

use crate::error::BackendResult as Result;
use crate::performance_modeling::{
    EnvironmentalFactors, PerformanceMeasurement, RuntimePerformanceModeler,
};
use crate::performance_tuning::{
    AccessPattern, ActualPerformance, DataType, OperationType, PerformancePrediction, SystemState,
    TuningParameters, WorkloadCharacteristics,
};
use crate::{BackendType, Device};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use torsh_core::error::TorshError;

#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, format, string::String, vec::Vec};

/// Adaptive kernel selection coordinator
pub struct AdaptiveKernelSelector {
    /// Kernel registry for different backends
    kernel_registry: Arc<RwLock<KernelRegistry>>,
    /// Performance modeler for prediction
    performance_modeler: Arc<RuntimePerformanceModeler>,
    /// Selection algorithm
    selection_algorithm: SelectionAlgorithm,
    /// Performance tracker for learning
    performance_tracker: Arc<Mutex<PerformanceTracker>>,
    /// Configuration parameters
    config: AdaptiveSelectionConfig,
}

/// Registry of available kernels organized by operation type and backend
#[derive(Debug)]
pub struct KernelRegistry {
    /// Kernels by operation type and backend
    kernels: HashMap<(OperationType, BackendType), Vec<KernelImplementation>>,
    /// Custom kernel implementations
    custom_kernels: HashMap<String, Box<dyn CustomKernel + Send + Sync>>,
    /// Kernel performance characteristics
    kernel_characteristics: HashMap<String, KernelCharacteristics>,
    /// Default kernel fallbacks
    #[allow(dead_code)]
    default_kernels: HashMap<(OperationType, BackendType), String>,
}

/// Kernel implementation metadata
#[derive(Debug, Clone)]
pub struct KernelImplementation {
    /// Unique kernel identifier
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Operation type this kernel implements
    pub operation_type: OperationType,
    /// Backend type this kernel runs on
    pub backend_type: BackendType,
    /// Kernel variant (e.g., "naive", "optimized", "tiled")
    pub variant: KernelVariant,
    /// Performance characteristics
    pub characteristics: KernelCharacteristics,
    /// Supported input constraints
    pub constraints: KernelConstraints,
    /// Kernel implementation
    pub implementation: Arc<dyn KernelExecutor + Send + Sync>,
}

/// Kernel variant types
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub enum KernelVariant {
    /// Naive implementation (simple, works for all inputs)
    Naive,
    /// Optimized implementation (tuned for specific characteristics)
    Optimized,
    /// Tiled implementation (memory hierarchy optimized)
    Tiled,
    /// Vectorized implementation (SIMD optimized)
    Vectorized,
    /// Parallel implementation (multi-threaded)
    Parallel,
    /// Fused implementation (multiple operations combined)
    Fused,
    /// Hardware-specific implementation
    HardwareSpecific(String),
    /// Custom implementation
    Custom(String),
}

/// Kernel performance characteristics
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct KernelCharacteristics {
    /// Optimal input size range
    pub optimal_size_range: (usize, usize),
    /// Memory access pattern
    pub memory_pattern: AccessPattern,
    /// Compute intensity (operations per byte)
    pub compute_intensity: f64,
    /// Parallelization efficiency
    pub parallelization_efficiency: f64,
    /// Cache efficiency
    pub cache_efficiency: f64,
    /// Memory bandwidth utilization
    pub memory_bandwidth_utilization: f64,
    /// Initialization overhead
    pub initialization_overhead: Duration,
    /// Scalability characteristics
    pub scalability: ScalabilityCharacteristics,
}

/// Kernel scalability characteristics
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct ScalabilityCharacteristics {
    /// How performance scales with input size
    pub size_scaling: ScalingBehavior,
    /// How performance scales with thread count
    pub thread_scaling: ScalingBehavior,
    /// How performance scales with memory hierarchy
    pub memory_hierarchy_scaling: ScalingBehavior,
}

/// Scaling behavior patterns
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub enum ScalingBehavior {
    /// Linear scaling
    Linear,
    /// Logarithmic scaling
    Logarithmic,
    /// Exponential scaling
    Exponential,
    /// Constant (no scaling)
    Constant,
    /// Custom scaling function
    Custom(String),
}

/// Kernel input constraints
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct KernelConstraints {
    /// Minimum input size
    pub min_size: usize,
    /// Maximum input size
    pub max_size: Option<usize>,
    /// Supported data types
    pub supported_dtypes: Vec<DataType>,
    /// Required memory alignment
    pub required_alignment: usize,
    /// Supported tensor shapes (None means any shape)
    pub supported_shapes: Option<Vec<Vec<usize>>>,
    /// Required hardware features
    pub required_features: Vec<String>,
}

/// Kernel executor interface
pub trait KernelExecutor: std::fmt::Debug + Send + Sync {
    /// Execute the kernel with given inputs
    fn execute(&self, inputs: &KernelInputs) -> Result<KernelOutputs>;

    /// Get estimated execution time
    fn estimate_execution_time(&self, inputs: &KernelInputs) -> Duration;

    /// Check if kernel can handle given inputs
    fn can_handle(&self, inputs: &KernelInputs) -> bool;

    /// Get kernel resource requirements
    fn get_resource_requirements(&self, inputs: &KernelInputs) -> ResourceRequirements;
}

/// Kernel input specification
#[derive(Debug, Clone)]
pub struct KernelInputs {
    /// Input tensor dimensions
    pub input_shapes: Vec<Vec<usize>>,
    /// Data types
    pub data_types: Vec<DataType>,
    /// Total data size in bytes
    pub total_size: usize,
    /// Operation parameters
    pub operation_params: HashMap<String, KernelParameter>,
    /// Device information
    pub device: Device,
}

/// Kernel output specification
#[derive(Debug, Clone)]
pub struct KernelOutputs {
    /// Output tensor dimensions
    pub output_shapes: Vec<Vec<usize>>,
    /// Execution time
    pub execution_time: Duration,
    /// Memory usage
    pub memory_usage: usize,
    /// Success flag
    pub success: bool,
    /// Error message (if any)
    pub error_message: Option<String>,
}

/// Kernel parameter values
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub enum KernelParameter {
    Integer(i64),
    Float(f64),
    String(String),
    Boolean(bool),
    IntegerArray(Vec<i64>),
    FloatArray(Vec<f64>),
}

/// Resource requirements for kernel execution
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct ResourceRequirements {
    /// Memory requirement in bytes
    pub memory: usize,
    /// Compute units required
    pub compute_units: usize,
    /// Bandwidth requirement in bytes/second
    pub bandwidth: usize,
    /// Temporary storage requirement
    pub temporary_storage: usize,
}

/// Custom kernel trait for user-defined kernels
pub trait CustomKernel: std::fmt::Debug + Send + Sync {
    /// Get kernel name
    fn name(&self) -> &str;

    /// Get operation type
    fn operation_type(&self) -> OperationType;

    /// Get backend type
    fn backend_type(&self) -> BackendType;

    /// Get kernel characteristics
    fn characteristics(&self) -> KernelCharacteristics;

    /// Get kernel constraints
    fn constraints(&self) -> KernelConstraints;

    /// Execute the kernel
    fn execute(&self, inputs: &KernelInputs) -> Result<KernelOutputs>;

    /// Benchmark the kernel
    fn benchmark(&self, inputs: &KernelInputs, iterations: usize) -> Result<BenchmarkResult>;
}

/// Benchmark result for kernel performance
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct BenchmarkResult {
    /// Average execution time
    pub avg_execution_time: Duration,
    /// Minimum execution time
    pub min_execution_time: Duration,
    /// Maximum execution time
    pub max_execution_time: Duration,
    /// Standard deviation
    pub std_deviation: Duration,
    /// Throughput (operations per second)
    pub throughput: f64,
    /// Memory bandwidth utilization
    pub memory_bandwidth: f64,
    /// Cache hit rate
    pub cache_hit_rate: f64,
}

/// Kernel selection algorithm
#[derive(Debug, Clone)]
pub enum SelectionAlgorithm {
    /// Score-based selection
    ScoreBased(ScoreBasedConfig),
    /// Machine learning-based selection
    MachineLearning(MLBasedConfig),
    /// Hybrid approach
    Hybrid(HybridConfig),
}

/// Configuration for score-based selection
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct ScoreBasedConfig {
    /// Weight for execution time
    pub execution_time_weight: f64,
    /// Weight for memory usage
    pub memory_usage_weight: f64,
    /// Weight for cache efficiency
    pub cache_efficiency_weight: f64,
    /// Weight for historical performance
    pub historical_weight: f64,
    /// Penalty for kernel switching
    pub switching_penalty: f64,
}

/// Configuration for ML-based selection
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct MLBasedConfig {
    /// Model type
    pub model_type: MLModelType,
    /// Training parameters
    pub training_params: MLTrainingParams,
    /// Feature weights
    pub feature_weights: HashMap<String, f64>,
}

/// Machine learning model types
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub enum MLModelType {
    DecisionTree,
    RandomForest,
    NeuralNetwork,
    SupportVectorMachine,
    LinearRegression,
    Custom(String),
}

/// ML training parameters
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct MLTrainingParams {
    /// Learning rate
    pub learning_rate: f64,
    /// Number of epochs
    pub epochs: usize,
    /// Batch size
    pub batch_size: usize,
    /// Regularization parameter
    pub regularization: f64,
}

/// Configuration for hybrid selection
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct HybridConfig {
    /// Score-based configuration
    pub score_based: ScoreBasedConfig,
    /// ML-based configuration
    pub ml_based: MLBasedConfig,
    /// Threshold for switching to ML
    pub ml_threshold: f64,
}

/// Performance tracker for kernel learning
#[derive(Debug)]
pub struct PerformanceTracker {
    /// Historical performance data
    #[allow(dead_code)]
    performance_history: HashMap<String, Vec<KernelPerformanceRecord>>,
    /// Kernel usage statistics
    usage_stats: HashMap<String, KernelUsageStats>,
    /// Selection accuracy tracking
    selection_accuracy: SelectionAccuracyTracker,
}

/// Kernel performance record
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct KernelPerformanceRecord {
    /// Timestamp
    pub timestamp: std::time::SystemTime,
    /// Input characteristics
    pub input_characteristics: WorkloadCharacteristics,
    /// System state
    pub system_state: SystemState,
    /// Actual performance
    pub actual_performance: ActualPerformance,
    /// Predicted performance
    pub predicted_performance: Option<PerformancePrediction>,
    /// Selection confidence
    pub selection_confidence: f64,
}

/// Kernel usage statistics
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct KernelUsageStats {
    /// Total executions
    pub total_executions: usize,
    /// Successful executions
    pub successful_executions: usize,
    /// Average execution time
    pub avg_execution_time: Duration,
    /// Last used timestamp
    pub last_used: std::time::SystemTime,
    /// Selection frequency
    pub selection_frequency: f64,
}

/// Selection accuracy tracker
#[derive(Debug)]
pub struct SelectionAccuracyTracker {
    /// Total selections made
    total_selections: usize,
    /// Optimal selections (in hindsight)
    optimal_selections: usize,
    /// Selection accuracy by operation type
    accuracy_by_operation: HashMap<OperationType, f64>,
    /// Selection accuracy by backend
    accuracy_by_backend: HashMap<BackendType, f64>,
}

/// Adaptive selection configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct AdaptiveSelectionConfig {
    /// Enable learning from performance feedback
    pub enable_learning: bool,
    /// Exploration vs exploitation trade-off
    pub exploration_factor: f64,
    /// Minimum confidence threshold for selections
    pub min_confidence_threshold: f64,
    /// Maximum number of concurrent benchmarks
    pub max_concurrent_benchmarks: usize,
    /// Benchmark timeout
    pub benchmark_timeout: Duration,
    /// History retention period
    pub history_retention: Duration,
}

impl Default for AdaptiveSelectionConfig {
    fn default() -> Self {
        Self {
            enable_learning: true,
            exploration_factor: 0.1,
            min_confidence_threshold: 0.8,
            max_concurrent_benchmarks: 4,
            benchmark_timeout: Duration::from_secs(30),
            history_retention: Duration::from_secs(7 * 24 * 3600), // 7 days
        }
    }
}

impl AdaptiveKernelSelector {
    /// Create a new adaptive kernel selector
    pub fn new(performance_modeler: Arc<RuntimePerformanceModeler>) -> Self {
        Self {
            kernel_registry: Arc::new(RwLock::new(KernelRegistry::new())),
            performance_modeler,
            selection_algorithm: SelectionAlgorithm::ScoreBased(ScoreBasedConfig::default()),
            performance_tracker: Arc::new(Mutex::new(PerformanceTracker::new())),
            config: AdaptiveSelectionConfig::default(),
        }
    }

    /// Register a kernel implementation
    pub fn register_kernel(&self, kernel: KernelImplementation) -> Result<()> {
        let mut registry = self.kernel_registry.write().unwrap();
        registry.register_kernel(kernel)
    }

    /// Register a custom kernel
    pub fn register_custom_kernel(
        &self,
        kernel: Box<dyn CustomKernel + Send + Sync>,
    ) -> Result<()> {
        let mut registry = self.kernel_registry.write().unwrap();
        registry.register_custom_kernel(kernel)
    }

    /// Select optimal kernel for given inputs
    pub fn select_kernel(
        &self,
        operation_type: OperationType,
        backend_type: BackendType,
        inputs: &KernelInputs,
        workload: &WorkloadCharacteristics,
        system_state: &SystemState,
    ) -> Result<KernelSelection> {
        let registry = self.kernel_registry.read().unwrap();

        // Get candidate kernels
        let candidates = registry.get_candidates(operation_type, backend_type, inputs)?;

        if candidates.is_empty() {
            return Err(TorshError::BackendError(format!(
                "No suitable kernels found for operation {:?} on backend {:?}",
                operation_type, backend_type
            )));
        }

        // Apply selection algorithm
        let selection = match &self.selection_algorithm {
            SelectionAlgorithm::ScoreBased(config) => {
                self.score_based_selection(&candidates, inputs, workload, system_state, config)?
            }
            SelectionAlgorithm::MachineLearning(config) => {
                self.ml_based_selection(&candidates, inputs, workload, system_state, config)?
            }
            SelectionAlgorithm::Hybrid(config) => {
                self.hybrid_selection(&candidates, inputs, workload, system_state, config)?
            }
        };

        // Track selection for learning
        if self.config.enable_learning {
            self.track_selection(&selection, workload, system_state)?;
        }

        Ok(selection)
    }

    /// Score-based kernel selection
    fn score_based_selection(
        &self,
        candidates: &[KernelImplementation],
        inputs: &KernelInputs,
        workload: &WorkloadCharacteristics,
        system_state: &SystemState,
        config: &ScoreBasedConfig,
    ) -> Result<KernelSelection> {
        let mut best_kernel = None;
        let mut best_score = f64::NEG_INFINITY;

        for kernel in candidates {
            let score =
                self.calculate_kernel_score(kernel, inputs, workload, system_state, config)?;

            if score > best_score {
                best_score = score;
                best_kernel = Some(kernel);
            }
        }

        let selected_kernel = best_kernel
            .ok_or_else(|| TorshError::BackendError("No suitable kernel found".to_string()))?;

        Ok(KernelSelection {
            kernel: selected_kernel.clone(),
            confidence: (best_score + 1.0) / 2.0, // Normalize to [0, 1]
            selection_reason: SelectionReason::ScoreBased(best_score),
            alternatives: candidates
                .iter()
                .filter(|k| k.id != selected_kernel.id)
                .cloned()
                .collect(),
        })
    }

    /// Machine learning-based kernel selection
    fn ml_based_selection(
        &self,
        candidates: &[KernelImplementation],
        inputs: &KernelInputs,
        workload: &WorkloadCharacteristics,
        system_state: &SystemState,
        _config: &MLBasedConfig,
    ) -> Result<KernelSelection> {
        // In a real implementation, this would use a trained ML model
        // For now, fall back to score-based selection
        let score_config = ScoreBasedConfig::default();
        self.score_based_selection(candidates, inputs, workload, system_state, &score_config)
    }

    /// Hybrid kernel selection
    fn hybrid_selection(
        &self,
        candidates: &[KernelImplementation],
        inputs: &KernelInputs,
        workload: &WorkloadCharacteristics,
        system_state: &SystemState,
        config: &HybridConfig,
    ) -> Result<KernelSelection> {
        // Use ML if confidence is above threshold, otherwise use score-based
        let ml_confidence = self.get_ml_confidence(inputs, workload, system_state)?;

        if ml_confidence > config.ml_threshold {
            self.ml_based_selection(candidates, inputs, workload, system_state, &config.ml_based)
        } else {
            self.score_based_selection(
                candidates,
                inputs,
                workload,
                system_state,
                &config.score_based,
            )
        }
    }

    /// Calculate kernel score for score-based selection
    fn calculate_kernel_score(
        &self,
        kernel: &KernelImplementation,
        inputs: &KernelInputs,
        workload: &WorkloadCharacteristics,
        system_state: &SystemState,
        config: &ScoreBasedConfig,
    ) -> Result<f64> {
        let mut score = 0.0;

        // Execution time score
        let predicted_time = self.predict_execution_time(kernel, inputs, workload, system_state)?;
        let time_score = 1.0 / (1.0 + predicted_time.as_secs_f64());
        score += config.execution_time_weight * time_score;

        // Memory usage score
        let memory_requirements = kernel.implementation.get_resource_requirements(inputs);
        let memory_score = 1.0 / (1.0 + memory_requirements.memory as f64 / 1024.0 / 1024.0);
        score += config.memory_usage_weight * memory_score;

        // Cache efficiency score
        let cache_score = kernel.characteristics.cache_efficiency;
        score += config.cache_efficiency_weight * cache_score;

        // Historical performance score
        let historical_score = self.get_historical_performance_score(&kernel.id)?;
        score += config.historical_weight * historical_score;

        // Switching penalty (if currently using a different kernel)
        if let Some(current_kernel) = self.get_current_kernel(workload)? {
            if current_kernel != kernel.id {
                score -= config.switching_penalty;
            }
        }

        Ok(score)
    }

    /// Predict execution time for a kernel
    fn predict_execution_time(
        &self,
        kernel: &KernelImplementation,
        _inputs: &KernelInputs,
        workload: &WorkloadCharacteristics,
        system_state: &SystemState,
    ) -> Result<Duration> {
        // Use the performance modeler to predict execution time
        let _measurement = PerformanceMeasurement {
            id: 0, // Placeholder
            timestamp: std::time::SystemTime::now(),
            backend_type: kernel.backend_type,
            device_id: 0, // Placeholder
            workload: workload.clone(),
            parameters: TuningParameters::default(),
            system_state: system_state.clone(),
            actual_performance: ActualPerformance::default(),
            predicted_performance: None,
            prediction_accuracy: None,
            environment: crate::performance_modeling::EnvironmentalFactors::default(),
        };

        // Get prediction from performance modeler
        let default_params = TuningParameters::default();
        let default_env = EnvironmentalFactors::default();
        let prediction = self.performance_modeler.predict_performance(
            kernel.backend_type,
            workload,
            &default_params,
            system_state,
            &default_env,
        )?;

        Ok(prediction.execution_time)
    }

    /// Get historical performance score for a kernel
    fn get_historical_performance_score(&self, kernel_id: &str) -> Result<f64> {
        let tracker = self.performance_tracker.lock().unwrap();

        if let Some(stats) = tracker.usage_stats.get(kernel_id) {
            let success_rate = stats.successful_executions as f64 / stats.total_executions as f64;
            let recency_factor = self.calculate_recency_factor(stats.last_used);
            Ok(success_rate * recency_factor)
        } else {
            Ok(0.5) // Neutral score for new kernels
        }
    }

    /// Calculate recency factor for historical performance
    fn calculate_recency_factor(&self, last_used: std::time::SystemTime) -> f64 {
        let now = std::time::SystemTime::now();
        let elapsed = now
            .duration_since(last_used)
            .unwrap_or(Duration::from_secs(0));
        let days_elapsed = elapsed.as_secs() as f64 / (24.0 * 3600.0);

        // Exponential decay with half-life of 7 days
        (-days_elapsed / 7.0).exp()
    }

    /// Get current kernel for workload (if any)
    fn get_current_kernel(&self, _workload: &WorkloadCharacteristics) -> Result<Option<String>> {
        // In a real implementation, this would track the currently selected kernel
        // For now, return None
        Ok(None)
    }

    /// Get ML confidence for hybrid selection
    fn get_ml_confidence(
        &self,
        _inputs: &KernelInputs,
        _workload: &WorkloadCharacteristics,
        _system_state: &SystemState,
    ) -> Result<f64> {
        // Placeholder implementation
        Ok(0.5)
    }

    /// Track kernel selection for learning
    fn track_selection(
        &self,
        selection: &KernelSelection,
        workload: &WorkloadCharacteristics,
        system_state: &SystemState,
    ) -> Result<()> {
        let mut tracker = self.performance_tracker.lock().unwrap();
        tracker.track_selection(selection, workload, system_state)
    }

    /// Update performance feedback
    pub fn update_performance_feedback(
        &self,
        kernel_id: &str,
        actual_performance: ActualPerformance,
        predicted_performance: Option<PerformancePrediction>,
    ) -> Result<()> {
        let mut tracker = self.performance_tracker.lock().unwrap();
        tracker.update_performance_feedback(kernel_id, actual_performance, predicted_performance)
    }

    /// Get selection statistics
    pub fn get_selection_statistics(&self) -> Result<SelectionStatistics> {
        let tracker = self.performance_tracker.lock().unwrap();
        Ok(tracker.get_statistics())
    }

    /// Benchmark kernels for calibration
    pub fn benchmark_kernels(
        &self,
        operation_type: OperationType,
        backend_type: BackendType,
        test_inputs: &[KernelInputs],
    ) -> Result<BenchmarkResults> {
        let registry = self.kernel_registry.read().unwrap();
        let kernels = registry.get_kernels_for_operation(operation_type, backend_type);

        let mut results = BenchmarkResults::new();

        for kernel in kernels {
            for inputs in test_inputs {
                if kernel.implementation.can_handle(inputs) {
                    let benchmark = self.benchmark_kernel(&kernel, inputs)?;
                    results.add_result(kernel.id.clone(), benchmark);
                }
            }
        }

        Ok(results)
    }

    /// Benchmark a single kernel
    fn benchmark_kernel(
        &self,
        kernel: &KernelImplementation,
        inputs: &KernelInputs,
    ) -> Result<BenchmarkResult> {
        let iterations = 10;
        let mut execution_times = Vec::new();

        for _ in 0..iterations {
            let start = Instant::now();
            let result = kernel.implementation.execute(inputs)?;
            let execution_time = start.elapsed();

            if result.success {
                execution_times.push(execution_time);
            }
        }

        if execution_times.is_empty() {
            return Err(TorshError::BackendError(
                "All benchmark iterations failed".to_string(),
            ));
        }

        let avg_time = execution_times.iter().sum::<Duration>() / execution_times.len() as u32;
        let min_time = *execution_times.iter().min().unwrap();
        let max_time = *execution_times.iter().max().unwrap();

        // Calculate standard deviation
        let variance = execution_times
            .iter()
            .map(|t| {
                let diff = t.as_secs_f64() - avg_time.as_secs_f64();
                diff * diff
            })
            .sum::<f64>()
            / execution_times.len() as f64;
        let std_dev = Duration::from_secs_f64(variance.sqrt());

        Ok(BenchmarkResult {
            avg_execution_time: avg_time,
            min_execution_time: min_time,
            max_execution_time: max_time,
            std_deviation: std_dev,
            throughput: 1.0 / avg_time.as_secs_f64(),
            memory_bandwidth: 0.0, // Placeholder
            cache_hit_rate: 0.0,   // Placeholder
        })
    }
}

/// Kernel selection result
#[derive(Debug, Clone)]
pub struct KernelSelection {
    /// Selected kernel
    pub kernel: KernelImplementation,
    /// Selection confidence (0.0 to 1.0)
    pub confidence: f64,
    /// Reason for selection
    pub selection_reason: SelectionReason,
    /// Alternative kernels considered
    pub alternatives: Vec<KernelImplementation>,
}

/// Reason for kernel selection
#[derive(Debug, Clone)]
pub enum SelectionReason {
    /// Score-based selection with score
    ScoreBased(f64),
    /// Machine learning prediction
    MachineLearning(f64),
    /// Hybrid selection
    Hybrid(f64),
    /// Fallback to default
    Default,
}

/// Benchmark results for multiple kernels
#[derive(Debug)]
pub struct BenchmarkResults {
    /// Results by kernel ID
    results: HashMap<String, BenchmarkResult>,
}

impl BenchmarkResults {
    pub fn new() -> Self {
        Self {
            results: HashMap::new(),
        }
    }

    pub fn add_result(&mut self, kernel_id: String, result: BenchmarkResult) {
        self.results.insert(kernel_id, result);
    }

    pub fn get_result(&self, kernel_id: &str) -> Option<&BenchmarkResult> {
        self.results.get(kernel_id)
    }

    pub fn get_best_kernel(&self) -> Option<(&String, &BenchmarkResult)> {
        self.results
            .iter()
            .min_by(|a, b| a.1.avg_execution_time.cmp(&b.1.avg_execution_time))
    }
}

/// Selection statistics
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct SelectionStatistics {
    /// Total selections made
    pub total_selections: usize,
    /// Selection accuracy
    pub overall_accuracy: f64,
    /// Accuracy by operation type
    pub accuracy_by_operation: HashMap<OperationType, f64>,
    /// Accuracy by backend
    pub accuracy_by_backend: HashMap<BackendType, f64>,
    /// Most frequently selected kernels
    pub popular_kernels: Vec<(String, usize)>,
}

impl KernelRegistry {
    pub fn new() -> Self {
        Self {
            kernels: HashMap::new(),
            custom_kernels: HashMap::new(),
            kernel_characteristics: HashMap::new(),
            default_kernels: HashMap::new(),
        }
    }

    pub fn register_kernel(&mut self, kernel: KernelImplementation) -> Result<()> {
        let key = (kernel.operation_type, kernel.backend_type);
        self.kernels
            .entry(key)
            .or_insert_with(Vec::new)
            .push(kernel.clone());
        self.kernel_characteristics
            .insert(kernel.id.clone(), kernel.characteristics);
        Ok(())
    }

    pub fn register_custom_kernel(
        &mut self,
        kernel: Box<dyn CustomKernel + Send + Sync>,
    ) -> Result<()> {
        let name = kernel.name().to_string();
        self.custom_kernels.insert(name, kernel);
        Ok(())
    }

    pub fn get_candidates(
        &self,
        operation_type: OperationType,
        backend_type: BackendType,
        inputs: &KernelInputs,
    ) -> Result<Vec<KernelImplementation>> {
        let key = (operation_type, backend_type);

        if let Some(kernels) = self.kernels.get(&key) {
            let candidates = kernels
                .iter()
                .filter(|k| k.implementation.can_handle(inputs))
                .cloned()
                .collect();
            Ok(candidates)
        } else {
            Ok(Vec::new())
        }
    }

    pub fn get_kernels_for_operation(
        &self,
        operation_type: OperationType,
        backend_type: BackendType,
    ) -> Vec<KernelImplementation> {
        let key = (operation_type, backend_type);
        self.kernels.get(&key).cloned().unwrap_or_default()
    }
}

impl PerformanceTracker {
    pub fn new() -> Self {
        Self {
            performance_history: HashMap::new(),
            usage_stats: HashMap::new(),
            selection_accuracy: SelectionAccuracyTracker::new(),
        }
    }

    pub fn track_selection(
        &mut self,
        selection: &KernelSelection,
        workload: &WorkloadCharacteristics,
        system_state: &SystemState,
    ) -> Result<()> {
        let kernel_id = &selection.kernel.id;

        // Update usage stats
        let stats = self
            .usage_stats
            .entry(kernel_id.clone())
            .or_insert_with(KernelUsageStats::default);
        stats.total_executions += 1;
        stats.last_used = std::time::SystemTime::now();

        // Update selection accuracy tracker
        self.selection_accuracy
            .track_selection(selection, workload, system_state);

        Ok(())
    }

    pub fn update_performance_feedback(
        &mut self,
        kernel_id: &str,
        actual_performance: ActualPerformance,
        _predicted_performance: Option<PerformancePrediction>,
    ) -> Result<()> {
        // Update usage stats
        if let Some(stats) = self.usage_stats.get_mut(kernel_id) {
            stats.successful_executions += 1;
            stats.avg_execution_time = actual_performance.execution_time;
        }

        Ok(())
    }

    pub fn get_statistics(&self) -> SelectionStatistics {
        SelectionStatistics {
            total_selections: self.selection_accuracy.total_selections,
            overall_accuracy: self.selection_accuracy.get_overall_accuracy(),
            accuracy_by_operation: self.selection_accuracy.accuracy_by_operation.clone(),
            accuracy_by_backend: self.selection_accuracy.accuracy_by_backend.clone(),
            popular_kernels: self.get_popular_kernels(),
        }
    }

    fn get_popular_kernels(&self) -> Vec<(String, usize)> {
        let mut kernels: Vec<_> = self
            .usage_stats
            .iter()
            .map(|(id, stats)| (id.clone(), stats.total_executions))
            .collect();
        kernels.sort_by(|a, b| b.1.cmp(&a.1));
        kernels.into_iter().take(10).collect()
    }
}

impl SelectionAccuracyTracker {
    pub fn new() -> Self {
        Self {
            total_selections: 0,
            optimal_selections: 0,
            accuracy_by_operation: HashMap::new(),
            accuracy_by_backend: HashMap::new(),
        }
    }

    pub fn track_selection(
        &mut self,
        selection: &KernelSelection,
        _workload: &WorkloadCharacteristics,
        _system_state: &SystemState,
    ) {
        self.total_selections += 1;

        // In a real implementation, this would determine if the selection was optimal
        // For now, assume high-confidence selections are more likely to be optimal
        if selection.confidence > 0.8 {
            self.optimal_selections += 1;
        }
    }

    pub fn get_overall_accuracy(&self) -> f64 {
        if self.total_selections == 0 {
            0.0
        } else {
            self.optimal_selections as f64 / self.total_selections as f64
        }
    }
}

// Default implementations
impl Default for ScoreBasedConfig {
    fn default() -> Self {
        Self {
            execution_time_weight: 0.4,
            memory_usage_weight: 0.2,
            cache_efficiency_weight: 0.2,
            historical_weight: 0.15,
            switching_penalty: 0.05,
        }
    }
}

impl Default for KernelUsageStats {
    fn default() -> Self {
        Self {
            total_executions: 0,
            successful_executions: 0,
            avg_execution_time: Duration::from_secs(0),
            last_used: std::time::SystemTime::now(),
            selection_frequency: 0.0,
        }
    }
}

impl Default for crate::performance_modeling::EnvironmentalFactors {
    fn default() -> Self {
        Self {
            ambient_temperature: None,
            system_load: 0.0,
            background_processes: 0,
            network_activity: 0.0,
            storage_io: 0.0,
            available_memory: 0,
            cpu_frequency: None,
            gpu_frequency: None,
        }
    }
}

impl Default for TuningParameters {
    fn default() -> Self {
        Self {
            thread_count: 1,
            vector_width: 1,
            block_size: Some(1024),
            tile_size: None,
            unroll_factor: 1,
            scheduling_strategy: crate::performance_tuning::SchedulingStrategy::Static,
            memory_allocation_strategy:
                crate::performance_tuning::MemoryAllocationStrategy::Default,
            optimization_level: crate::performance_tuning::OptimizationLevel::Default,
            backend_specific: HashMap::new(),
        }
    }
}

impl Default for ActualPerformance {
    fn default() -> Self {
        Self {
            execution_time: Duration::from_secs(0),
            throughput: 0.0,
            memory_usage_peak: 0,
            power_consumption_avg: 0.0,
            cache_hit_ratio: 0.0,
            thermal_increase: 0.0,
            cpu_utilization: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_registry() {
        let mut registry = KernelRegistry::new();

        let kernel = KernelImplementation {
            id: "test_kernel".to_string(),
            name: "Test Kernel".to_string(),
            operation_type: OperationType::MatrixMultiply,
            backend_type: BackendType::Cpu,
            variant: KernelVariant::Naive,
            characteristics: KernelCharacteristics {
                optimal_size_range: (1, 1000),
                memory_pattern: AccessPattern::Sequential,
                compute_intensity: 1.0,
                parallelization_efficiency: 0.8,
                cache_efficiency: 0.7,
                memory_bandwidth_utilization: 0.6,
                initialization_overhead: Duration::from_millis(1),
                scalability: ScalabilityCharacteristics {
                    size_scaling: ScalingBehavior::Linear,
                    thread_scaling: ScalingBehavior::Linear,
                    memory_hierarchy_scaling: ScalingBehavior::Constant,
                },
            },
            constraints: KernelConstraints {
                min_size: 1,
                max_size: Some(1000),
                supported_dtypes: vec![DataType::F32],
                required_alignment: 4,
                supported_shapes: None,
                required_features: vec![],
            },
            implementation: std::sync::Arc::new(MockKernelExecutor),
        };

        assert!(registry.register_kernel(kernel).is_ok());

        let inputs = KernelInputs {
            input_shapes: vec![vec![10, 10]],
            data_types: vec![DataType::F32],
            total_size: 400,
            operation_params: HashMap::new(),
            device: Device::cpu().unwrap(),
        };

        let candidates = registry
            .get_candidates(OperationType::MatrixMultiply, BackendType::Cpu, &inputs)
            .unwrap();
        assert_eq!(candidates.len(), 1);
    }

    #[test]
    fn test_performance_tracker() {
        let mut tracker = PerformanceTracker::new();

        let selection = KernelSelection {
            kernel: create_test_kernel(),
            confidence: 0.9,
            selection_reason: SelectionReason::ScoreBased(0.8),
            alternatives: vec![],
        };

        let workload = WorkloadCharacteristics {
            operation_type: OperationType::MatrixMultiply,
            data_size: 1000,
            data_shape: vec![10, 10],
            data_type: DataType::F32,
            access_pattern: AccessPattern::Sequential,
            compute_intensity: 1.0,
            memory_bandwidth_requirement: 0.5,
            parallelization_potential: 0.8,
            cache_locality: 0.7,
            branch_predictability: 0.9,
            vectorization_potential: 0.8,
        };

        let system_state = SystemState {
            cpu_utilization: 0.5,
            memory_utilization: 0.6,
            thermal_state: crate::performance_tuning::ThermalState {
                cpu_temperature: 65.0,
                gpu_temperature: Some(70.0),
                thermal_throttling_active: false,
                cooling_efficiency: 0.85,
            },
            power_state: crate::performance_tuning::PowerState {
                power_limit: Some(100.0),
                current_power_draw: 75.0,
                battery_level: Some(0.8),
                power_efficiency_mode: crate::performance_tuning::PowerEfficiencyMode::Balanced,
            },
            concurrent_workloads: 2,
            available_memory_bandwidth: 0.7,
            cache_pressure: 0.4,
            numa_topology: crate::performance_tuning::NumaTopologyState {
                node_count: 1,
                current_node: 0,
                memory_distribution: vec![0.6],
                cross_node_traffic: 0.0,
            },
        };

        assert!(tracker
            .track_selection(&selection, &workload, &system_state)
            .is_ok());

        let stats = tracker.get_statistics();
        assert_eq!(stats.total_selections, 1);
    }

    fn create_test_kernel() -> KernelImplementation {
        KernelImplementation {
            id: "test_kernel".to_string(),
            name: "Test Kernel".to_string(),
            operation_type: OperationType::MatrixMultiply,
            backend_type: BackendType::Cpu,
            variant: KernelVariant::Naive,
            characteristics: KernelCharacteristics {
                optimal_size_range: (1, 1000),
                memory_pattern: AccessPattern::Sequential,
                compute_intensity: 1.0,
                parallelization_efficiency: 0.8,
                cache_efficiency: 0.7,
                memory_bandwidth_utilization: 0.6,
                initialization_overhead: Duration::from_millis(1),
                scalability: ScalabilityCharacteristics {
                    size_scaling: ScalingBehavior::Linear,
                    thread_scaling: ScalingBehavior::Linear,
                    memory_hierarchy_scaling: ScalingBehavior::Constant,
                },
            },
            constraints: KernelConstraints {
                min_size: 1,
                max_size: Some(1000),
                supported_dtypes: vec![DataType::F32],
                required_alignment: 4,
                supported_shapes: None,
                required_features: vec![],
            },
            implementation: std::sync::Arc::new(MockKernelExecutor),
        }
    }

    #[derive(Debug)]
    struct MockKernelExecutor;

    impl KernelExecutor for MockKernelExecutor {
        fn execute(&self, inputs: &KernelInputs) -> Result<KernelOutputs> {
            Ok(KernelOutputs {
                output_shapes: inputs.input_shapes.clone(),
                execution_time: Duration::from_millis(10),
                memory_usage: inputs.total_size,
                success: true,
                error_message: None,
            })
        }

        fn estimate_execution_time(&self, _inputs: &KernelInputs) -> Duration {
            Duration::from_millis(10)
        }

        fn can_handle(&self, _inputs: &KernelInputs) -> bool {
            true
        }

        fn get_resource_requirements(&self, inputs: &KernelInputs) -> ResourceRequirements {
            ResourceRequirements {
                memory: inputs.total_size,
                compute_units: 1,
                bandwidth: 1000,
                temporary_storage: 0,
            }
        }
    }
}
