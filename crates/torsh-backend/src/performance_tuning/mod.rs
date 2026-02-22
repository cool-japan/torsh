//! Backend-specific performance tuning strategies
//!
//! This module provides comprehensive performance tuning strategies that are tailored
//! to specific backend characteristics, workload patterns, and system conditions.
//!
//! # Architecture
//!
//! The performance tuning system is organized into specialized modules:
//!
//! - **`types`**: Core type definitions for all performance tuning components
//! - **`coordination`**: Main coordinator implementation and supporting logic
//! - **`strategies`**: Backend-specific tuning strategy implementations
//!
//! # Usage
//!
//! ## Basic Performance Tuning
//!
//! ```rust,ignore
//! use torsh_backend::performance_tuning::{
//!     PerformanceTuningCoordinator, WorkloadCharacteristics, OperationType, DataType
//! };
//!
//! // Create coordinator
//! let coordinator = PerformanceTuningCoordinator::new()?;
//!
//! // Define workload
//! let workload = WorkloadCharacteristics {
//!     operation_type: OperationType::MatrixMultiply,
//!     data_size: 1024 * 1024,
//!     data_type: DataType::F32,
//!     // ... other fields
//! };
//!
//! // Get tuning recommendation
//! let recommendation = coordinator.get_tuning_recommendation(
//!     BackendType::Cpu,
//!     &workload,
//!     &system_state,
//!     &constraints
//! )?;
//! ```
//!
//! ## Advanced Configuration
//!
//! ```rust,ignore
//! use torsh_backend::performance_tuning::{
//!     create_sample_workload, create_default_system_state, create_default_constraints,
//!     OperationType, PowerEfficiencyMode
//! };
//!
//! // Create sample configurations
//! let workload = create_sample_workload(OperationType::Convolution2D, 512 * 512);
//! let mut system_state = create_default_system_state();
//! system_state.power_state.power_efficiency_mode = PowerEfficiencyMode::MaxPerformance;
//!
//! // Get backend-specific metrics
//! let metrics = coordinator.get_strategy_metrics(BackendType::Cuda)?;
//! println!("CUDA strategy accuracy: {:.2}%", metrics.prediction_accuracy * 100.0);
//! ```

// Core modules
pub mod coordination;
pub mod strategies;
pub mod types;

// Re-export all public types and functionality
pub use types::*;

// Re-export key functions and utilities
pub use coordination::{
    create_default_constraints, create_default_system_state, create_sample_workload,
};

use crate::backend::BackendType;
use crate::error::BackendResult;

// ================================================================================================
// Public API and Convenience Functions
// ================================================================================================

/// Create a new PerformanceTuningCoordinator with default configuration
///
/// This is a convenience function that creates a coordinator with all backends enabled.
///
/// # Returns
/// * `BackendResult<PerformanceTuningCoordinator>` - Configured coordinator
///
/// # Example
/// ```rust,ignore
/// let coordinator = torsh_backend::performance_tuning::new_coordinator()?;
/// ```
pub fn new_coordinator() -> BackendResult<PerformanceTuningCoordinator> {
    PerformanceTuningCoordinator::new()
}

/// Create a performance-optimized system configuration
///
/// Returns a SystemState configured for maximum performance scenarios.
///
/// # Returns
/// * `SystemState` - Performance-optimized system state
pub fn create_performance_optimized_system_state() -> SystemState {
    let mut state = create_default_system_state();
    state.power_state.power_efficiency_mode = PowerEfficiencyMode::MaxPerformance;
    state.thermal_state.cooling_efficiency = 1.0;
    state.cpu_utilization = 0.3; // Leave headroom for performance
    state.memory_utilization = 0.4;
    state.cache_pressure = 0.2;
    state
}

/// Create a power-efficient system configuration
///
/// Returns a SystemState configured for power-saving scenarios.
///
/// # Returns
/// * `SystemState` - Power-efficient system state
pub fn create_power_efficient_system_state() -> SystemState {
    let mut state = create_default_system_state();
    state.power_state.power_efficiency_mode = PowerEfficiencyMode::PowerSaver;
    state.cpu_utilization = 0.7; // Higher utilization acceptable for power saving
    state.memory_utilization = 0.8;
    state.thermal_state.cooling_efficiency = 0.6;
    state
}

/// Create strict performance constraints for real-time applications
///
/// Returns TuningConstraints configured for latency-sensitive workloads.
///
/// # Arguments
/// * `max_latency_ms` - Maximum acceptable latency in milliseconds
///
/// # Returns
/// * `TuningConstraints` - Real-time optimized constraints
pub fn create_realtime_constraints(max_latency_ms: u64) -> TuningConstraints {
    TuningConstraints {
        max_memory_usage: Some(512 * 1024 * 1024), // 512MB limit for predictability
        max_power_draw: None,
        max_temperature: None,
        latency_requirement: Some(std::time::Duration::from_millis(max_latency_ms)),
        throughput_requirement: None,
        energy_budget: None,
        real_time_constraints: true,
    }
}

/// Create throughput-optimized constraints for batch processing
///
/// Returns TuningConstraints configured for maximum throughput scenarios.
///
/// # Arguments
/// * `min_throughput` - Minimum required throughput (operations per second)
///
/// # Returns
/// * `TuningConstraints` - Throughput-optimized constraints
pub fn create_throughput_constraints(min_throughput: f64) -> TuningConstraints {
    TuningConstraints {
        max_memory_usage: None, // No memory limit for throughput
        max_power_draw: None,
        max_temperature: None,
        latency_requirement: None,
        throughput_requirement: Some(min_throughput),
        energy_budget: None,
        real_time_constraints: false,
    }
}

/// Create energy-budget constraints for mobile/battery scenarios
///
/// Returns TuningConstraints configured for energy-efficient operation.
///
/// # Arguments
/// * `energy_budget_joules` - Energy budget in joules
/// * `max_power_watts` - Maximum power consumption in watts
///
/// # Returns
/// * `TuningConstraints` - Energy-budget optimized constraints
pub fn create_energy_budget_constraints(
    energy_budget_joules: f64,
    max_power_watts: f32,
) -> TuningConstraints {
    TuningConstraints {
        max_memory_usage: Some(256 * 1024 * 1024), // Conservative memory usage
        max_power_draw: Some(max_power_watts),
        max_temperature: Some(70.0), // Conservative temperature limit
        latency_requirement: None,
        throughput_requirement: None,
        energy_budget: Some(energy_budget_joules),
        real_time_constraints: false,
    }
}

/// Create a workload profile for machine learning training
///
/// Returns WorkloadCharacteristics optimized for ML training patterns.
///
/// # Arguments
/// * `batch_size` - Training batch size
/// * `model_params` - Number of model parameters
/// * `precision` - Data type precision
///
/// # Returns
/// * `WorkloadCharacteristics` - ML training workload profile
pub fn create_ml_training_workload(
    batch_size: usize,
    model_params: usize,
    precision: DataType,
) -> WorkloadCharacteristics {
    WorkloadCharacteristics {
        operation_type: OperationType::MatrixMultiply, // Training is dominated by matrix ops
        data_size: batch_size * model_params,
        data_shape: vec![batch_size, model_params],
        data_type: precision,
        access_pattern: AccessPattern::Sequential, // Training typically has good locality
        compute_intensity: 0.9,                    // Training is very compute-intensive
        memory_bandwidth_requirement: 0.8,         // High memory bandwidth needed
        parallelization_potential: 0.95,           // Training parallelizes very well
        cache_locality: 0.8,                       // Good cache locality in training
        branch_predictability: 0.9,                // Training loops are predictable
        vectorization_potential: 0.95,             // Matrix operations vectorize well
    }
}

/// Create a workload profile for machine learning inference
///
/// Returns WorkloadCharacteristics optimized for ML inference patterns.
///
/// # Arguments
/// * `input_size` - Size of input data
/// * `model_params` - Number of model parameters
/// * `precision` - Data type precision
///
/// # Returns
/// * `WorkloadCharacteristics` - ML inference workload profile
pub fn create_ml_inference_workload(
    input_size: usize,
    model_params: usize,
    precision: DataType,
) -> WorkloadCharacteristics {
    WorkloadCharacteristics {
        operation_type: OperationType::MatrixMultiply,
        data_size: input_size * model_params,
        data_shape: vec![1, input_size], // Typically single batch for inference
        data_type: precision,
        access_pattern: AccessPattern::Sequential,
        compute_intensity: 0.8, // Inference is compute-intensive but less than training
        memory_bandwidth_requirement: 0.6, // Lower memory bandwidth than training
        parallelization_potential: 0.7, // Less parallelization than training
        cache_locality: 0.9,    // Better cache locality than training
        branch_predictability: 0.95, // Very predictable for inference
        vectorization_potential: 0.9, // Good vectorization potential
    }
}

/// Create a workload profile for image processing operations
///
/// Returns WorkloadCharacteristics optimized for image processing patterns.
///
/// # Arguments
/// * `image_width` - Image width in pixels
/// * `image_height` - Image height in pixels
/// * `channels` - Number of color channels
/// * `operation` - Type of image operation
///
/// # Returns
/// * `WorkloadCharacteristics` - Image processing workload profile
pub fn create_image_processing_workload(
    image_width: usize,
    image_height: usize,
    channels: usize,
    operation: OperationType,
) -> WorkloadCharacteristics {
    let data_size = image_width * image_height * channels;

    WorkloadCharacteristics {
        operation_type: operation,
        data_size,
        data_shape: vec![image_height, image_width, channels],
        data_type: DataType::U8, // Most image data is 8-bit
        access_pattern: AccessPattern::Blocked { block_size: 64 }, // Images often processed in blocks
        compute_intensity: match operation {
            OperationType::Convolution2D => 0.8,
            OperationType::ElementWise => 0.3,
            OperationType::Pooling => 0.4,
            _ => 0.5,
        },
        memory_bandwidth_requirement: 0.7, // Image processing is often memory-bound
        parallelization_potential: 0.9,    // Images parallelize very well
        cache_locality: 0.6,               // Depends on access pattern
        branch_predictability: 0.8,        // Image operations are fairly predictable
        vectorization_potential: 0.95,     // Image operations vectorize excellently
    }
}

/// Get recommended backend for a specific workload
///
/// Analyzes workload characteristics and returns the most suitable backend.
///
/// # Arguments
/// * `workload` - Workload characteristics to analyze
/// * `available_backends` - List of available backends
///
/// # Returns
/// * `BackendType` - Recommended backend for the workload
pub fn recommend_backend(
    workload: &WorkloadCharacteristics,
    available_backends: &[BackendType],
) -> BackendType {
    // Simple heuristic-based backend selection
    // In practice, this would use ML models and performance history

    for &backend in available_backends {
        match backend {
            BackendType::Cuda
                if workload.compute_intensity > 0.8 && workload.parallelization_potential > 0.8 =>
            {
                return BackendType::Cuda; // CUDA for highly parallel compute workloads
            }
            BackendType::Metal
                if workload.compute_intensity > 0.7 && workload.data_type == DataType::F16 =>
            {
                return BackendType::Metal; // Metal for Apple Silicon with FP16
            }
            BackendType::WebGpu if workload.data_size < 1024 * 1024 => {
                return BackendType::WebGpu; // WebGPU for smaller workloads
            }
            _ => {}
        }
    }

    // Default to CPU if available
    if available_backends.contains(&BackendType::Cpu) {
        BackendType::Cpu
    } else {
        // Return first available backend
        available_backends
            .first()
            .copied()
            .unwrap_or(BackendType::Cpu)
    }
}

/// Analyze workload and suggest optimization opportunities
///
/// Examines workload characteristics and provides optimization suggestions.
///
/// # Arguments
/// * `workload` - Workload to analyze
///
/// # Returns
/// * `Vec<String>` - List of optimization suggestions
pub fn analyze_workload_optimization_opportunities(
    workload: &WorkloadCharacteristics,
) -> Vec<String> {
    let mut suggestions = Vec::new();

    // Analyze parallelization potential
    if workload.parallelization_potential > 0.8 && workload.compute_intensity > 0.7 {
        suggestions.push("Consider GPU acceleration for this highly parallel workload".to_string());
    }

    // Analyze memory access patterns
    match workload.access_pattern {
        AccessPattern::Random => {
            suggestions.push("Random memory access detected - consider data restructuring for better cache locality".to_string());
        }
        AccessPattern::Strided { stride } if stride > 4 => {
            suggestions.push(format!(
                "Large stride ({}) detected - consider memory layout optimization",
                stride
            ));
        }
        _ => {}
    }

    // Analyze vectorization potential
    if workload.vectorization_potential > 0.8
        && workload.operation_type == OperationType::ElementWise
    {
        suggestions.push(
            "High vectorization potential - ensure SIMD optimizations are enabled".to_string(),
        );
    }

    // Analyze compute intensity
    if workload.compute_intensity < 0.3 {
        suggestions.push(
            "Memory-bound workload detected - focus on memory bandwidth optimization".to_string(),
        );
    }

    // Analyze data size
    if workload.data_size > 100 * 1024 * 1024 {
        suggestions
            .push("Large dataset detected - consider chunking or streaming approaches".to_string());
    }

    // Data type specific suggestions
    match workload.data_type {
        DataType::F64 => {
            suggestions.push(
                "Using FP64 precision - consider FP32 if acceptable for performance gains"
                    .to_string(),
            );
        }
        DataType::F32 if workload.operation_type == OperationType::MatrixMultiply => {
            suggestions.push(
                "Consider FP16 precision for matrix multiplication if accuracy allows".to_string(),
            );
        }
        _ => {}
    }

    suggestions
}

// ================================================================================================
// Prelude Module for Convenient Imports
// ================================================================================================

/// Prelude module for convenient imports
pub mod prelude {
    pub use super::{
        analyze_workload_optimization_opportunities,
        create_default_constraints,
        create_default_system_state,
        create_energy_budget_constraints,
        create_image_processing_workload,
        create_ml_inference_workload,
        create_ml_training_workload,
        create_performance_optimized_system_state,
        create_power_efficient_system_state,
        create_realtime_constraints,
        create_sample_workload,
        create_throughput_constraints,
        // Convenience functions
        new_coordinator,
        recommend_backend,
        AccessPattern,
        ActualPerformance,
        BackendTuningStrategy,
        DataType,
        GlobalPerformanceStats,

        MemoryAllocationStrategy,
        NumaTopologyState,

        // Enums
        OperationType,
        OptimizationLevel,
        PerformanceFeedback,
        PerformancePrediction,
        // Core types
        PerformanceTuningCoordinator,
        PowerEfficiencyMode,
        PowerState,
        SchedulingStrategy,
        StrategyMetrics,
        SystemState,
        // System monitoring types
        ThermalState,
        TuningConstraints,
        TuningParameters,
        TuningRecommendation,
        TuningValue,

        WorkloadCharacteristics,
    };
}

// ================================================================================================
// Tests
// ================================================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::BackendType;

    #[test]
    fn test_coordinator_creation() {
        let coordinator = PerformanceTuningCoordinator::new();
        assert!(coordinator.is_ok());
    }

    #[test]
    fn test_workload_characteristics_creation() {
        let workload = WorkloadCharacteristics {
            operation_type: OperationType::MatrixMultiply,
            data_size: 1024 * 1024,
            data_shape: vec![1024, 1024],
            data_type: DataType::F32,
            access_pattern: AccessPattern::Sequential,
            compute_intensity: 0.8,
            memory_bandwidth_requirement: 0.6,
            parallelization_potential: 0.9,
            cache_locality: 0.7,
            branch_predictability: 0.95,
            vectorization_potential: 0.85,
        };

        assert_eq!(workload.operation_type, OperationType::MatrixMultiply);
        assert_eq!(workload.data_size, 1024 * 1024);
    }

    #[test]
    fn test_cache_key_computation() {
        let coordinator = PerformanceTuningCoordinator::new().unwrap();

        let workload = WorkloadCharacteristics {
            operation_type: OperationType::ElementWise,
            data_size: 1000,
            data_shape: vec![100, 10],
            data_type: DataType::F32,
            access_pattern: AccessPattern::Sequential,
            compute_intensity: 0.5,
            memory_bandwidth_requirement: 0.3,
            parallelization_potential: 0.7,
            cache_locality: 0.8,
            branch_predictability: 0.9,
            vectorization_potential: 0.6,
        };

        let system_state = create_default_system_state();
        let key1 = coordinator.compute_cache_key(BackendType::Cpu, &workload, &system_state);
        let key2 = coordinator.compute_cache_key(BackendType::Cpu, &workload, &system_state);

        assert_eq!(key1, key2);
    }

    #[test]
    fn test_convenience_functions() {
        // Test system state creation
        let default_state = create_default_system_state();
        assert!(default_state.cpu_utilization >= 0.0 && default_state.cpu_utilization <= 1.0);

        let perf_state = create_performance_optimized_system_state();
        assert_eq!(
            perf_state.power_state.power_efficiency_mode,
            PowerEfficiencyMode::MaxPerformance
        );

        let power_state = create_power_efficient_system_state();
        assert_eq!(
            power_state.power_state.power_efficiency_mode,
            PowerEfficiencyMode::PowerSaver
        );

        // Test constraint creation
        let realtime_constraints = create_realtime_constraints(10);
        assert!(realtime_constraints.real_time_constraints);
        assert_eq!(
            realtime_constraints.latency_requirement,
            Some(std::time::Duration::from_millis(10))
        );

        let throughput_constraints = create_throughput_constraints(1000.0);
        assert_eq!(throughput_constraints.throughput_requirement, Some(1000.0));

        let energy_constraints = create_energy_budget_constraints(100.0, 50.0);
        assert_eq!(energy_constraints.energy_budget, Some(100.0));
        assert_eq!(energy_constraints.max_power_draw, Some(50.0));
    }

    #[test]
    fn test_ml_workload_creation() {
        let training_workload = create_ml_training_workload(32, 1000, DataType::F32);
        assert_eq!(
            training_workload.operation_type,
            OperationType::MatrixMultiply
        );
        assert_eq!(training_workload.data_size, 32 * 1000);
        assert!(training_workload.compute_intensity > 0.8);

        let inference_workload = create_ml_inference_workload(1, 1000, DataType::F16);
        assert_eq!(inference_workload.data_shape, vec![1, 1]);
        assert!(inference_workload.cache_locality > training_workload.cache_locality);
    }

    #[test]
    fn test_image_processing_workload() {
        let image_workload =
            create_image_processing_workload(1920, 1080, 3, OperationType::Convolution2D);
        assert_eq!(image_workload.data_size, 1920 * 1080 * 3);
        assert_eq!(image_workload.data_type, DataType::U8);
        assert_eq!(image_workload.operation_type, OperationType::Convolution2D);

        if let AccessPattern::Blocked { block_size } = image_workload.access_pattern {
            assert_eq!(block_size, 64);
        } else {
            panic!("Expected blocked access pattern");
        }
    }

    #[test]
    fn test_backend_recommendation() {
        let available_backends = vec![BackendType::Cpu, BackendType::Cuda, BackendType::Metal];

        // Test CUDA recommendation for compute-intensive workload
        let compute_workload = WorkloadCharacteristics {
            operation_type: OperationType::MatrixMultiply,
            data_size: 1024 * 1024,
            data_shape: vec![1024, 1024],
            data_type: DataType::F32,
            access_pattern: AccessPattern::Sequential,
            compute_intensity: 0.9,
            memory_bandwidth_requirement: 0.6,
            parallelization_potential: 0.95,
            cache_locality: 0.7,
            branch_predictability: 0.95,
            vectorization_potential: 0.85,
        };

        let recommended = recommend_backend(&compute_workload, &available_backends);
        assert_eq!(recommended, BackendType::Cuda);

        // Test Metal recommendation for FP16 workload
        let fp16_workload = WorkloadCharacteristics {
            operation_type: OperationType::MatrixMultiply,
            data_size: 512 * 512,
            data_shape: vec![512, 512],
            data_type: DataType::F16,
            access_pattern: AccessPattern::Sequential,
            compute_intensity: 0.8,
            memory_bandwidth_requirement: 0.6,
            parallelization_potential: 0.8,
            cache_locality: 0.7,
            branch_predictability: 0.95,
            vectorization_potential: 0.85,
        };

        let recommended = recommend_backend(&fp16_workload, &available_backends);
        assert_eq!(recommended, BackendType::Metal);
    }

    #[test]
    fn test_optimization_analysis() {
        let workload = WorkloadCharacteristics {
            operation_type: OperationType::ElementWise,
            data_size: 200 * 1024 * 1024, // Large dataset
            data_shape: vec![200 * 1024 * 1024],
            data_type: DataType::F64,              // High precision
            access_pattern: AccessPattern::Random, // Poor access pattern
            compute_intensity: 0.2,                // Memory-bound
            memory_bandwidth_requirement: 0.9,
            parallelization_potential: 0.9,
            cache_locality: 0.3,
            branch_predictability: 0.9,
            vectorization_potential: 0.9,
        };

        let suggestions = analyze_workload_optimization_opportunities(&workload);

        // Should suggest optimizations for random access, large dataset, FP64, and memory-bound nature
        assert!(suggestions.len() > 0);
        assert!(suggestions
            .iter()
            .any(|s| s.contains("Random memory access")));
        assert!(suggestions.iter().any(|s| s.contains("Large dataset")));
        assert!(suggestions.iter().any(|s| s.contains("FP64")));
        assert!(suggestions.iter().any(|s| s.contains("Memory-bound")));
    }

    #[test]
    fn test_tuning_parameters_equality() {
        let params1 = TuningParameters {
            thread_count: 8,
            vector_width: 256,
            block_size: Some(1024),
            tile_size: Some((16, 16)),
            unroll_factor: 4,
            scheduling_strategy: SchedulingStrategy::Dynamic,
            memory_allocation_strategy: MemoryAllocationStrategy::NumaLocal,
            optimization_level: OptimizationLevel::Optimized,
            backend_specific: std::collections::HashMap::new(),
        };

        let params2 = TuningParameters {
            thread_count: 8,
            vector_width: 256,
            block_size: Some(1024),
            tile_size: Some((16, 16)),
            unroll_factor: 4,
            scheduling_strategy: SchedulingStrategy::Dynamic,
            memory_allocation_strategy: MemoryAllocationStrategy::NumaLocal,
            optimization_level: OptimizationLevel::Optimized,
            backend_specific: std::collections::HashMap::new(),
        };

        assert_eq!(params1, params2);
    }

    #[test]
    fn test_new_coordinator_convenience() {
        let coordinator = new_coordinator();
        assert!(coordinator.is_ok());
    }

    #[test]
    fn test_sample_workload_creation() {
        let workload = create_sample_workload(OperationType::Convolution2D, 1024);
        assert_eq!(workload.operation_type, OperationType::Convolution2D);
        assert_eq!(workload.data_size, 1024);
        assert_eq!(workload.data_type, DataType::F32);
    }
}
