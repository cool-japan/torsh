//! Communication-Efficient Distributed Training Framework
//!
//! This module provides a comprehensive framework for communication-efficient distributed
//! deep learning training. It includes gradient compression, aggregation algorithms,
//! transmission scheduling, and configuration management for optimizing distributed training performance.
//!
//! # Architecture
//!
//! The communication-efficient framework is built around several key components:
//!
//! - **Configuration Management**: Comprehensive configuration for compression strategies,
//!   communication topologies, and aggregation methods
//! - **Gradient Compression**: Advanced compression algorithms including quantization,
//!   sparsification, low-rank approximation, and sketching
//! - **Gradient Aggregation**: Robust aggregation engine with Byzantine fault tolerance,
//!   staleness compensation, and quality-weighted aggregation
//! - **Transmission Scheduling**: Adaptive scheduling, protocol optimization, and QoS-aware
//!   transmission management for efficient gradient communication
//! - **Bandwidth Management**: Dynamic bandwidth allocation, congestion control, and network
//!   topology optimization for optimal distributed communication
//!
//! # Key Features
//!
//! - **Multiple Compression Strategies**: Support for various compression techniques
//!   with adaptive selection based on network conditions
//! - **Byzantine Fault Tolerance**: Robust aggregation against malicious or faulty workers
//! - **Quality-Aware Aggregation**: Dynamic weighting based on gradient quality metrics
//! - **Adaptive Transmission Scheduling**: Intelligent scheduling algorithms that adapt
//!   to network conditions, deadlines, and QoS requirements
//! - **Protocol Optimization**: Dynamic protocol switching for optimal performance
//! - **Staleness Compensation**: Handling of delayed updates in asynchronous scenarios
//! - **Performance Monitoring**: Comprehensive metrics collection and analysis
//! - **Bandwidth Management**: Dynamic bandwidth allocation with congestion control
//! - **Topology Optimization**: Latency-aware topology reconfiguration and load balancing
//! - **Fault Tolerance**: Comprehensive fault detection, recovery, and distributed coordination
//!
//! # Module Organization
//!
//! The framework is organized into specialized modules for different aspects of communication-efficient
//! distributed training:
//!
//! - [`config`]: Configuration structures and enums (~550 lines)
//!   - Core configuration types and strategy enums
//!   - Communication metadata and quality-of-service specifications
//!   - Security requirements and privacy levels
//!
//! - [`compression`]: Gradient compression engine with multiple algorithms (~600 lines)
//!   - Multi-strategy compression engine with adaptive selection
//!   - Quantization, sparsification, low-rank, and sketching algorithms
//!   - Error feedback mechanisms and compression statistics
//!
//! - [`aggregation`]: Gradient aggregation engine with robustness features (~878 lines)
//!   - Byzantine fault-tolerant aggregation algorithms
//!   - Quality-weighted aggregation with staleness compensation
//!   - Comprehensive aggregation metrics and monitoring
//!
//! - [`transmission`]: Transmission scheduling and protocol optimization (~1,200 lines)
//!   - Adaptive scheduling algorithms with QoS awareness
//!   - Protocol stack optimization and dynamic selection
//!   - Performance monitoring and network metrics collection
//!
//! - [`management`]: Bandwidth management and topology optimization (~400 lines)
//!   - Dynamic bandwidth allocation and congestion control
//!   - Topology reconfiguration and load balancing
//!   - Shared resource management across distributed nodes
//!
//! - [`fault_tolerance`]: Fault detection and recovery mechanisms (~485 lines)
//!   - Comprehensive fault detection with multiple methods
//!   - Byzantine detection and distributed coordination
//!   - Recovery strategies and resilient communication patterns
//!
//! - [`adaptation`]: Adaptive control and performance monitoring (~691 lines)
//!   - Performance-based adaptation controllers
//!   - Dynamic strategy selection based on runtime conditions
//!   - Comprehensive communication metrics and profiling
//!
//! # Examples
//!
//! ## Basic Setup with Unified Interface
//! ```rust,ignore
//! use torsh_autograd::communication_efficient::{
//!     // Use the unified interface with organized re-exports
//!     Config, Strategy, Method, Compressor, Aggregator,
//!     CommunicationConfig, CompressionStrategy, AggregationMethod,
//!     CompressionEngine, AggregationEngine,
//! };
//!
//! // Create configuration using unified types
//! let config = Config {
//!     compression_strategy: Strategy::AdaptiveCompression,
//!     aggregation_method: Method::WeightedAverage,
//!     ..Default::default()
//! };
//!
//! // Create engines using type aliases
//! let compressor = Compressor::new(Strategy::AdaptiveCompression);
//! let mut aggregator = Aggregator::with_config(Method::WeightedAverage, config.clone());
//! ```
//!
//! ## Module-Specific Usage
//! ```rust,ignore
//! use torsh_autograd::communication_efficient::{
//!     // Configuration module
//!     config::{CommunicationConfig, AggregationMethod, CompressionStrategy},
//!
//!     // Compression module
//!     compression::{CompressionEngine, QuantizationMethod, SparsificationMethod},
//!
//!     // Aggregation module
//!     aggregation::AggregationEngine,
//!
//!     // Management module
//!     management::{BandwidthManager, TopologyManager},
//!
//!     // Fault tolerance module
//!     fault_tolerance::CommunicationFaultDetector,
//! };
//!
//! // Use specific modules directly
//! let config = CommunicationConfig::default();
//! let mut compression_engine = CompressionEngine::new(CompressionStrategy::Quantization);
//! let mut aggregation_engine = AggregationEngine::new(AggregationMethod::WeightedAverage);
//! let bandwidth_manager = BandwidthManager::new();
//! let fault_detector = CommunicationFaultDetector::new();
//! ```
//!
//! ## Advanced Byzantine-Resilient Configuration
//! ```rust,ignore
//! use torsh_autograd::communication_efficient::{
//!     Config, Method, Aggregator, FaultDetector, Controller,
//!     ByzantineDetectionConfig, AdaptationController, AdaptationStrategy,
//! };
//! use std::time::Duration;
//!
//! // Configure for Byzantine-resilient training with adaptation
//! let config = Config {
//!     aggregation_method: Method::Median, // Byzantine-resilient aggregation
//!     ..Default::default()
//! };
//!
//! let mut aggregator = Aggregator::new(Method::Median);
//! aggregator.enable_byzantine_resilience(true);
//! aggregator.set_byzantine_threshold(0.33); // Tolerate up to 33% malicious workers
//! aggregator.set_max_staleness(Duration::from_secs(30));
//! aggregator.set_quality_threshold(0.7);
//!
//! // Add fault detection and adaptive control
//! let fault_detector = FaultDetector::new();
//! let adaptation_controller = Controller::new(AdaptationStrategy::PerformanceBased);
//! ```
//!
//! ## Comprehensive Distributed Training Setup
//! ```rust,ignore
//! use torsh_autograd::communication_efficient::{
//!     Config, Strategy, Method, Compressor, Aggregator, Scheduler,
//!     BandwidthMgr, TopologyMgr, FaultDetector, Controller,
//!     SchedulingStrategy, OptimizationObjective, AdaptationStrategy,
//! };
//!
//! // Complete communication-efficient setup
//! let config = Config {
//!     compression_strategy: Strategy::AdaptiveCompression,
//!     aggregation_method: Method::WeightedAverage,
//!     ..Default::default()
//! };
//!
//! // Initialize all components
//! let compressor = Compressor::new(Strategy::AdaptiveCompression);
//! let aggregator = Aggregator::with_config(Method::WeightedAverage, config.clone());
//! let scheduler = Scheduler::new(SchedulingStrategy::AdaptiveScheduling);
//! let bandwidth_mgr = BandwidthMgr::new();
//! let topology_mgr = TopologyMgr::new(OptimizationObjective::MinimizeLatency);
//! let fault_detector = FaultDetector::new();
//! let controller = Controller::new(AdaptationStrategy::PerformanceBased);
//! ```
//!
//! # Feature Overview by Module
//!
//! ## Configuration (`config`) - ~550 lines
//! - **Core Types**: `CommunicationConfig`, `CommunicationEfficientGradient`, `CompressedGradient`
//! - **Strategies**: `CompressionStrategy`, `AggregationMethod`, `CommunicationTopology`
//! - **QoS**: Quality of service specifications, security requirements, privacy levels
//!
//! ## Compression (`compression`) - ~600 lines
//! - **Multi-Algorithm Engine**: Adaptive compression with strategy selection
//! - **Quantization**: Multiple quantization methods with adaptive precision
//! - **Sparsification**: Top-k, random, and structured sparsification
//! - **Low-Rank**: SVD, PCA, and rank adaptation strategies
//! - **Sketching**: Count-sketch and other probabilistic methods
//! - **Error Feedback**: Gradient error accumulation and compensation
//!
//! ## Aggregation (`aggregation`) - ~878 lines
//! - **Byzantine Tolerance**: Robust aggregation against malicious workers
//! - **Quality Weighting**: Dynamic weighting based on gradient quality
//! - **Staleness Compensation**: Handling delayed updates in async scenarios
//! - **Multiple Methods**: Median, trimmed mean, weighted average, etc.
//!
//! ## Transmission (`transmission`) - ~1,200 lines
//! - **Adaptive Scheduling**: QoS-aware scheduling with deadline management
//! - **Protocol Optimization**: Dynamic protocol selection (TCP/UDP/RDMA)
//! - **Batch Management**: Intelligent gradient batching and prioritization
//! - **Performance Monitoring**: Comprehensive network metrics collection
//!
//! ## Management (`management`) - ~400 lines
//! - **Bandwidth Management**: Dynamic allocation with congestion control
//! - **Topology Optimization**: Latency-aware reconfiguration
//! - **Load Balancing**: Distributed load balancing across nodes
//! - **RTT Estimation**: Real-time network performance monitoring
//!
//! ## Fault Tolerance (`fault_tolerance`) - ~485 lines
//! - **Multi-Method Detection**: Heartbeat, statistical, and pattern-based
//! - **Byzantine Detection**: Advanced Byzantine fault detection
//! - **Recovery Strategies**: Graceful degradation and recovery mechanisms
//! - **Distributed Coordination**: Consensus-based fault coordination
//!
//! ## Adaptation (`adaptation`) - ~691 lines
//! - **Performance Monitoring**: Real-time performance snapshots
//! - **Adaptive Control**: Dynamic strategy adaptation based on conditions
//! - **Trigger System**: Configurable adaptation triggers
//! - **Metrics Collection**: Comprehensive communication metrics

//! ===========================================================================================
//! # MODULE DECLARATIONS
//! ===========================================================================================

// Core configuration module
pub mod config;

// Compression algorithms and engine
pub mod compression;

// Gradient aggregation engine
pub mod aggregation;

// Transmission scheduling and protocol optimization
pub mod transmission;

// Bandwidth management and topology optimization
pub mod management;

// Fault detection and recovery mechanisms
pub mod fault_tolerance;

// Adaptive control and performance monitoring
pub mod adaptation;

// ===========================================================================================
// # RE-EXPORTS FOR UNIFIED INTERFACE
// ===========================================================================================
//
// This section provides organized re-exports from all specialized modules to create
// a unified interface while maintaining backward compatibility.

// Configuration and Core Types
pub use config::{
    // Core configuration structures
    AggregationMethod,
    CommunicationConfig,
    CommunicationEfficientGradient,
    CommunicationMetadata,
    CommunicationPriority,
    CommunicationTopology,
    CompressedGradient,
    CompressionInfo,
    // Strategy and topology enums
    CompressionStrategy,
    DecompressionHints,

    LatencySensitivity,

    PrivacyLevel,
    ProtocolOptimization,
    // Quality of service and security
    QualityOfService,
    ReconstructionMethod,
    ReliabilityLevel,

    SecurityRequirements,
    // Compression-specific types
    SketchParameters,
    SketchType,
    SparsityPattern,
    SynchronizationPattern,
};

// Compression Engine and Algorithms
pub use compression::{
    AdaptiveCompressionParameters,

    // Core compression engine and error handling
    CompressionEngine,
    CompressionError,
    CompressionStatistics,
    DecompositionMethod,
    // Low-rank compression
    LowRankEngine,
    // Quantization algorithms
    QuantizationEngine,
    QuantizationMethod,

    RankAdaptationStrategy,

    // Sketching algorithms
    SketchingEngine,
    // Sparsification algorithms
    SparsificationEngine,
    SparsificationMethod,
};

// Gradient Aggregation Engine
pub use aggregation::{
    // Core aggregation engine and error handling
    AggregationEngine,
    AggregationError,

    AggregationMetrics,
    // Worker and metric types
    WorkerContribution,
};

// Transmission Scheduling and Protocol Optimization
pub use transmission::{
    CommunicationProtocol,

    NetworkMetrics,
    PriorityMetrics,
    // Performance metrics and monitoring
    ProtocolMetrics,
    ProtocolSelection,

    // Protocol stack and communication
    ProtocolStack,
    SchedulingKey,
    // Scheduling algorithms and strategies
    SchedulingStrategy,
    TransmissionBatchManager,
    TransmissionPriorityManager,

    // Core transmission components
    TransmissionScheduler,
};

// Bandwidth Management and Topology Optimization
pub use management::{
    // Bandwidth management components
    BandwidthManager,
    BandwidthMeasurement,
    // Congestion control algorithms
    CongestionControl,
    CongestionControlAlgorithm,
    CongestionLevel,
    CongestionState,

    OptimizationObjective,

    // Network performance monitoring
    RTTEstimator,
    SharedBandwidthManager,

    SharedTopologyManager,
    // Network topology management
    TopologyManager,
    TopologyOptimization,
};

// Fault Tolerance and Recovery Mechanisms
pub use fault_tolerance::{
    AnomalyThresholds,

    // Byzantine fault tolerance
    ByzantineDetectionConfig,
    ByzantineVerificationMethod,

    // Core fault detection and recovery
    CommunicationFaultDetector,
    CommunicationPattern,
    ConnectionStatus,
    ConsensusConfig,
    ConsensusRequest,
    CoordinationProtocol,
    // Distributed coordination
    DistributedFaultCoordinator,
    FailureEvent,
    FaultDetectionMethod,
    FaultSeverity,
    FaultType,
    // Monitoring and analysis components
    HeartbeatTracker,
    QuorumRequirements,
    RecoveryStrategy,

    StatisticalAnalyzer,
};

// Adaptive Control and Performance Monitoring
pub use adaptation::{
    // Core adaptation controller and strategies
    AdaptationController,
    AdaptationStrategy,
    AdaptationTrigger,

    CommunicationMetrics,
    ControlParameters,
    // Performance monitoring and control
    PerformanceSnapshot,
};

// ===========================================================================================
// # CONVENIENCE TYPE ALIASES AND COMMON PATTERNS
// ===========================================================================================

// Convenience type aliases for commonly used types
/// Type alias for communication configuration
pub type Config = CommunicationConfig;

/// Type alias for efficient gradient
pub type EfficientGradient = CommunicationEfficientGradient;

/// Type alias for aggregation method
pub type Method = AggregationMethod;

/// Type alias for compression strategy
pub type Strategy = CompressionStrategy;

/// Type alias for compression engine
pub type Compressor = CompressionEngine;

/// Type alias for aggregation engine
pub type Aggregator = AggregationEngine;

/// Type alias for transmission scheduler
pub type Scheduler = TransmissionScheduler;

/// Type alias for bandwidth manager
pub type BandwidthMgr = BandwidthManager;

/// Type alias for topology manager
pub type TopologyMgr = TopologyManager;

/// Type alias for fault detector
pub type FaultDetector = CommunicationFaultDetector;

/// Type alias for adaptation controller
pub type Controller = AdaptationController;

// ===========================================================================================
// # UNIFIED ERROR HANDLING
// ===========================================================================================

/// Common error type for communication-efficient operations
///
/// This unified error type provides comprehensive error handling across all modules
/// in the communication-efficient framework, allowing for easy error propagation
/// and consistent error handling patterns.
#[derive(Debug, Clone)]
pub enum CommunicationError {
    /// Compression operation failed
    CompressionFailed,
    /// Decompression operation failed
    DecompressionFailed,
    /// Transmission operation failed
    TransmissionFailed,
    /// Scheduling operation failed
    SchedulingFailed,
    /// Available bandwidth exhausted
    BandwidthExhausted,
    /// Topology configuration error
    TopologyError,
    /// Aggregation operation failed
    AggregationFailed,
    /// Fault detection failed
    FaultDetectionFailed,
    /// Adaptation operation failed
    AdaptationFailed,
    /// Security-related error
    SecurityError,
    /// Protocol-level error
    ProtocolError,
    /// Configuration error
    ConfigurationError,
}

impl std::fmt::Display for CommunicationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CommunicationError::CompressionFailed => write!(f, "Compression operation failed"),
            CommunicationError::DecompressionFailed => write!(f, "Decompression operation failed"),
            CommunicationError::TransmissionFailed => write!(f, "Transmission operation failed"),
            CommunicationError::SchedulingFailed => write!(f, "Scheduling operation failed"),
            CommunicationError::BandwidthExhausted => write!(f, "Available bandwidth exhausted"),
            CommunicationError::TopologyError => write!(f, "Topology configuration error"),
            CommunicationError::AggregationFailed => write!(f, "Aggregation operation failed"),
            CommunicationError::FaultDetectionFailed => write!(f, "Fault detection failed"),
            CommunicationError::AdaptationFailed => write!(f, "Adaptation operation failed"),
            CommunicationError::SecurityError => write!(f, "Security-related error"),
            CommunicationError::ProtocolError => write!(f, "Protocol-level error"),
            CommunicationError::ConfigurationError => write!(f, "Configuration error"),
        }
    }
}

impl std::error::Error for CommunicationError {}

// Error conversions from module-specific errors to unified communication error
impl From<aggregation::AggregationError> for CommunicationError {
    fn from(_: aggregation::AggregationError) -> Self {
        CommunicationError::AggregationFailed
    }
}

impl From<compression::CompressionError> for CommunicationError {
    fn from(_: compression::CompressionError) -> Self {
        CommunicationError::CompressionFailed
    }
}

// ===========================================================================================
// # UTILITY FUNCTIONS AND HELPER METHODS
// ===========================================================================================

/// Utility functions for the communication-efficient framework
///
/// This module provides common utility functions used across the communication-efficient
/// distributed training framework, including gradient analysis, bandwidth estimation,
/// compatibility checking, and validation functions.
pub mod utils {
    use super::*;
    use std::collections::HashMap;

    /// Calculate the compression ratio for a gradient
    pub fn calculate_compression_ratio(original_size: usize, compressed_size: usize) -> f64 {
        if original_size == 0 {
            1.0
        } else {
            compressed_size as f64 / original_size as f64
        }
    }

    /// Estimate bandwidth requirement for a gradient
    pub fn estimate_bandwidth_requirement(
        gradient: &HashMap<String, Vec<f32>>,
        compression_ratio: f64,
    ) -> u64 {
        let original_size: usize = gradient
            .values()
            .map(|v| v.len() * std::mem::size_of::<f32>())
            .sum();

        (original_size as f64 * compression_ratio) as u64
    }

    /// Check if gradients are compatible for aggregation
    pub fn check_gradient_compatibility(
        gradients: &[HashMap<String, Vec<f32>>],
    ) -> Result<(), CommunicationError> {
        if gradients.is_empty() {
            return Ok(());
        }

        let reference = &gradients[0];
        for gradient in gradients.iter().skip(1) {
            if gradient.len() != reference.len() {
                return Err(CommunicationError::AggregationFailed);
            }

            for (param_name, values) in gradient {
                if let Some(ref_values) = reference.get(param_name) {
                    if values.len() != ref_values.len() {
                        return Err(CommunicationError::AggregationFailed);
                    }
                } else {
                    return Err(CommunicationError::AggregationFailed);
                }
            }
        }
        Ok(())
    }

    /// Compute L2 norm of a gradient
    pub fn compute_gradient_norm(gradient: &HashMap<String, Vec<f32>>) -> f64 {
        let mut norm_squared = 0.0;
        for values in gradient.values() {
            for &value in values {
                norm_squared += (value as f64).powi(2);
            }
        }
        norm_squared.sqrt()
    }

    /// Check if all values in gradient are finite
    pub fn validate_gradient_finite(gradient: &HashMap<String, Vec<f32>>) -> bool {
        gradient
            .values()
            .all(|values| values.iter().all(|&v| v.is_finite()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_compression_ratio_calculation() {
        assert_eq!(utils::calculate_compression_ratio(1000, 500), 0.5);
        assert_eq!(utils::calculate_compression_ratio(0, 0), 1.0);
        assert_eq!(utils::calculate_compression_ratio(100, 100), 1.0);
    }

    #[test]
    fn test_bandwidth_estimation() {
        let mut gradient = HashMap::new();
        gradient.insert("param".to_string(), vec![1.0; 1000]);

        let bandwidth = utils::estimate_bandwidth_requirement(&gradient, 0.5);
        let expected = 1000 * std::mem::size_of::<f32>() / 2;
        assert_eq!(bandwidth, expected as u64);
    }

    #[test]
    fn test_gradient_compatibility() {
        let mut grad1 = HashMap::new();
        grad1.insert("param1".to_string(), vec![1.0, 2.0]);
        grad1.insert("param2".to_string(), vec![3.0]);

        let mut grad2 = HashMap::new();
        grad2.insert("param1".to_string(), vec![4.0, 5.0]);
        grad2.insert("param2".to_string(), vec![6.0]);

        assert!(utils::check_gradient_compatibility(&[grad1, grad2]).is_ok());
    }

    #[test]
    fn test_gradient_incompatibility() {
        let mut grad1 = HashMap::new();
        grad1.insert("param1".to_string(), vec![1.0, 2.0]);

        let mut grad2 = HashMap::new();
        grad2.insert("param1".to_string(), vec![4.0, 5.0, 6.0]); // Different size

        assert!(utils::check_gradient_compatibility(&[grad1, grad2]).is_err());
    }

    #[test]
    fn test_gradient_norm() {
        let mut gradient = HashMap::new();
        gradient.insert("param".to_string(), vec![3.0, 4.0]); // 3-4-5 triangle

        let norm = utils::compute_gradient_norm(&gradient);
        assert!((norm - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_gradient_finite_validation() {
        let mut gradient_finite = HashMap::new();
        gradient_finite.insert("param".to_string(), vec![1.0, 2.0, 3.0]);

        let mut gradient_infinite = HashMap::new();
        gradient_infinite.insert("param".to_string(), vec![1.0, f32::INFINITY, 3.0]);

        assert!(utils::validate_gradient_finite(&gradient_finite));
        assert!(!utils::validate_gradient_finite(&gradient_infinite));
    }

    #[test]
    fn test_communication_error_display() {
        let error = CommunicationError::CompressionFailed;
        assert_eq!(format!("{}", error), "Compression operation failed");

        let error = CommunicationError::BandwidthExhausted;
        assert_eq!(format!("{}", error), "Available bandwidth exhausted");
    }

    #[test]
    fn test_error_conversion() {
        let agg_error = aggregation::AggregationError::EmptyGradientSet;
        let comm_error: CommunicationError = agg_error.into();
        assert!(matches!(comm_error, CommunicationError::AggregationFailed));
    }
}
