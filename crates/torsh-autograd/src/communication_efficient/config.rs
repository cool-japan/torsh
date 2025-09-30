//! Configuration structures and enums for communication-efficient distributed training.
//!
//! This module provides comprehensive configuration options for distributed communication
//! in deep learning training, including compression strategies, communication topologies,
//! aggregation methods, and various quality-of-service parameters.
//!
//! # Examples
//!
//! ## Basic Configuration
//! ```rust
//! use torsh_autograd::communication_efficient::config::*;
//!
//! let config = CommunicationConfig::default();
//! println!("Using {} compression with {} topology",
//!          config.compression_strategy, config.topology);
//! ```
//!
//! ## Custom Configuration
//! ```rust
//! use torsh_autograd::communication_efficient::config::*;
//! use std::time::Duration;
//!
//! let config = CommunicationConfig {
//!     compression_strategy: CompressionStrategy::Quantization,
//!     quantization_bits: 4,
//!     sparsification_ratio: 0.05,
//!     topology: CommunicationTopology::Ring,
//!     aggregation_method: AggregationMethod::Median,
//!     communication_budget: Some(500_000),
//!     ..Default::default()
//! };
//! ```

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Main configuration structure for communication-efficient distributed training.
///
/// This structure encapsulates all configuration options for optimizing communication
/// in distributed machine learning scenarios, including compression settings,
/// network topology choices, and various optimization flags.
#[derive(Debug, Clone)]
pub struct CommunicationConfig {
    /// Strategy used for gradient compression
    pub compression_strategy: CompressionStrategy,
    /// Number of bits used for quantization (when quantization is enabled)
    pub quantization_bits: u8,
    /// Ratio of gradient values to keep during sparsification (0.0 to 1.0)
    pub sparsification_ratio: f64,
    /// Whether to use error feedback to compensate for compression losses
    pub error_feedback: bool,
    /// Whether to enable gradient dropping for extremely slow workers
    pub gradient_dropping: bool,
    /// Whether to accumulate gradients locally before communication
    pub local_accumulation: bool,
    /// Whether to adapt communication patterns based on available bandwidth
    pub bandwidth_adaptation: bool,
    /// Network protocol optimization strategy
    pub protocol_optimization: ProtocolOptimization,
    /// Communication topology for distributed workers
    pub topology: CommunicationTopology,
    /// Method used for aggregating gradients across workers
    pub aggregation_method: AggregationMethod,
    /// Synchronization pattern for distributed training
    pub synchronization_pattern: SynchronizationPattern,
    /// Whether to enable fault tolerance mechanisms
    pub fault_tolerance: bool,
    /// Whether to enable adaptive compression based on network conditions
    pub adaptive_compression: bool,
    /// Optional budget for communication bandwidth (bytes per round)
    pub communication_budget: Option<u64>,
    /// Whether to optimize for low latency
    pub latency_optimization: bool,
    /// Whether to optimize for energy efficiency
    pub energy_efficiency: bool,
}

/// Compression strategies for reducing communication overhead.
///
/// Different compression methods provide various trade-offs between
/// compression ratio, computational overhead, and convergence quality.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CompressionStrategy {
    /// No compression applied
    None,
    /// Quantization-based compression (reduces precision)
    Quantization,
    /// Sparsification-based compression (keeps only top-k gradients)
    Sparsification,
    /// Low-rank approximation compression
    LowRank,
    /// Sketching-based compression using random projections
    Sketching,
    /// Combination of multiple compression techniques
    HybridCompression,
    /// Adaptive compression that changes based on conditions
    AdaptiveCompression,
    /// Context-aware compression using historical data
    ContextAwareCompression,
    /// Learning-based compression using neural networks
    LearningBasedCompression,
}

/// Network protocol optimization strategies.
///
/// Different protocols offer various characteristics in terms of
/// reliability, latency, and throughput.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProtocolOptimization {
    /// Traditional TCP protocol
    TCP,
    /// User Datagram Protocol for low latency
    UDP,
    /// QUIC protocol for improved performance over unreliable networks
    QUIC,
    /// WebRTC for real-time communication
    WebRTC,
    /// Custom protocol designed for ML workloads
    CustomProtocol,
    /// Adaptive protocol selection based on network conditions
    AdaptiveProtocol,
    /// Hierarchical protocol optimization
    HierarchicalProtocol,
}

/// Communication topology patterns for distributed training.
///
/// Different topologies provide various trade-offs between
/// scalability, fault tolerance, and communication efficiency.
#[derive(Debug, Clone, PartialEq)]
pub enum CommunicationTopology {
    /// All-reduce pattern where all workers communicate with all others
    AllReduce,
    /// Centralized parameter server architecture
    ParameterServer,
    /// Hierarchical topology with multiple levels
    Hierarchical,
    /// Ring topology for sequential communication
    Ring,
    /// Tree topology for logarithmic scaling
    Tree,
    /// Butterfly topology for parallel algorithms
    Butterfly,
    /// Gossip protocol for decentralized communication
    Gossip,
    /// Hybrid topology combining multiple patterns
    Hybrid,
    /// Adaptive topology that changes based on conditions
    AdaptiveTopology,
}

/// Methods for aggregating gradients from multiple workers.
///
/// Different aggregation methods provide various robustness
/// characteristics against Byzantine failures and statistical outliers.
#[derive(Debug, Clone, PartialEq)]
pub enum AggregationMethod {
    /// Simple sum aggregation
    Sum,
    /// Average aggregation (sum divided by count)
    Average,
    /// Median aggregation for robustness against outliers
    Median,
    /// Weighted average based on worker reliability or data size
    WeightedAverage,
    /// Trimmed mean excluding extreme values
    TrimmedMean,
    /// Adaptive weighting based on gradient quality metrics
    AdaptiveWeighting,
    /// Hierarchical aggregation with multiple stages
    HierarchicalAggregation,
}

/// Synchronization patterns for coordinating distributed training.
///
/// Different patterns provide trade-offs between convergence speed,
/// fault tolerance, and computational efficiency.
#[derive(Debug, Clone, PartialEq)]
pub enum SynchronizationPattern {
    /// Synchronous training where all workers wait for each other
    Synchronous,
    /// Asynchronous training with no synchronization barriers
    Asynchronous,
    /// Bulk Synchronous Parallel with periodic synchronization
    BulkSynchronousParallel,
    /// Bounded staleness allowing some workers to lag
    StaleSync,
    /// Local SGD with periodic averaging
    LocalSGD,
    /// Federated averaging for privacy-preserving training
    FederatedAveraging,
    /// Elastic averaging with dynamic worker participation
    ElasticAveraging,
}

/// Communication-efficient gradient container with compression and metadata.
///
/// This structure represents a gradient that has been prepared for
/// efficient transmission across the network, including compression
/// information and communication metadata.
#[derive(Debug, Clone)]
pub struct CommunicationEfficientGradient {
    /// Original uncompressed gradient data
    pub original_gradient: HashMap<String, Vec<f32>>,
    /// Compressed representation of the gradient
    pub compressed_gradient: CompressedGradient,
    /// Achieved compression ratio (compressed_size / original_size)
    pub compression_ratio: f64,
    /// ID of the worker that produced this gradient
    pub worker_id: u32,
    /// Training round number
    pub round: u64,
    /// Timestamp when the gradient was created
    pub timestamp: Instant,
    /// Communication priority level
    pub priority: CommunicationPriority,
    /// Estimated bandwidth requirement for transmission
    pub bandwidth_requirement: u64,
    /// Latency sensitivity for this gradient
    pub latency_sensitivity: LatencySensitivity,
    /// Error feedback information for compression correction
    pub error_feedback: Option<HashMap<String, Vec<f32>>>,
    /// Local accumulation buffer for gradient aggregation
    pub accumulation_buffer: Option<HashMap<String, Vec<f32>>>,
    /// Additional communication metadata
    pub metadata: CommunicationMetadata,
}

/// Compressed gradient representation with decompression information.
///
/// Contains the actual compressed data along with all information
/// needed to reconstruct the original gradient.
#[derive(Debug, Clone)]
pub struct CompressedGradient {
    /// Compression method used for this gradient
    pub compression_method: CompressionStrategy,
    /// Actual compressed gradient data
    pub compressed_data: Vec<u8>,
    /// Information about the compression process
    pub compression_info: CompressionInfo,
    /// Hints for efficient decompression
    pub decompression_hints: DecompressionHints,
}

/// Detailed information about the compression process and parameters.
///
/// Provides comprehensive statistics and parameters used during
/// compression for analysis and decompression.
#[derive(Debug, Clone)]
pub struct CompressionInfo {
    /// Size of the original gradient in bytes
    pub original_size: usize,
    /// Size of the compressed gradient in bytes
    pub compressed_size: usize,
    /// Quantization scale factor (for quantization methods)
    pub quantization_scale: Option<f32>,
    /// Indices of non-zero elements (for sparsification methods)
    pub sparse_indices: Option<Vec<usize>>,
    /// Low-rank decomposition factors (for low-rank methods)
    pub low_rank_factors: Option<(Vec<f32>, Vec<f32>)>,
    /// Parameters for sketching-based compression
    pub sketch_parameters: Option<SketchParameters>,
    /// Time taken for compression
    pub compression_time: Duration,
}

/// Hints and metadata for efficient gradient decompression.
///
/// Provides shape information and reconstruction parameters
/// to enable accurate and efficient decompression.
#[derive(Debug, Clone)]
pub struct DecompressionHints {
    /// Shape information for each parameter tensor
    pub shape_information: HashMap<String, Vec<usize>>,
    /// Sparsity pattern for sparse gradients
    pub sparsity_pattern: Option<SparsityPattern>,
    /// Quantization bounds for dequantization
    pub quantization_bounds: Option<(f32, f32)>,
    /// Method to use for gradient reconstruction
    pub reconstruction_method: ReconstructionMethod,
}

/// Parameters for sketching-based compression methods.
///
/// Contains all necessary information to perform and invert
/// sketching operations on gradients.
#[derive(Debug, Clone)]
pub struct SketchParameters {
    /// Size of the sketch representation
    pub sketch_size: usize,
    /// Hash functions used for sketching
    pub hash_functions: Vec<u32>,
    /// Random matrix for projection-based sketching
    pub random_matrix: Vec<f32>,
    /// Type of sketching algorithm used
    pub sketch_type: SketchType,
}

/// Types of sketching algorithms for compression.
///
/// Different sketching methods provide various trade-offs
/// between compression ratio and reconstruction accuracy.
#[derive(Debug, Clone, PartialEq)]
pub enum SketchType {
    /// Count sketch for frequency estimation
    CountSketch,
    /// Johnson-Lindenstrauss embedding
    JohnsonLindenstrauss,
    /// Fast Johnson-Lindenstrauss transform
    FastJohnsonLindenstrauss,
    /// Sparse random projection
    SparseRandomProjection,
}

/// Sparsity pattern information for sparse gradient representations.
///
/// Describes the structure of sparse gradients including
/// locations and values of non-zero elements.
#[derive(Debug, Clone)]
pub struct SparsityPattern {
    /// Indices of non-zero elements
    pub indices: Vec<usize>,
    /// Values at non-zero positions
    pub values: Vec<f32>,
    /// Shape of the original dense tensor
    pub shape: Vec<usize>,
    /// Density (ratio of non-zero elements)
    pub density: f64,
}

/// Methods for reconstructing compressed gradients.
///
/// Different reconstruction methods provide various trade-offs
/// between accuracy and computational cost.
#[derive(Debug, Clone, PartialEq)]
pub enum ReconstructionMethod {
    /// Direct reconstruction without approximation
    Direct,
    /// Iterative reconstruction for better accuracy
    Iterative,
    /// Approximate reconstruction for speed
    ApproximateReconstruction,
    /// Learning-based reconstruction using neural networks
    LearningBasedReconstruction,
}

/// Priority levels for communication operations.
///
/// Higher priority gradients are transmitted first and
/// may receive additional quality-of-service guarantees.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CommunicationPriority {
    /// Low priority for background operations
    Low,
    /// Normal priority for standard gradients
    Normal,
    /// High priority for important gradients
    High,
    /// Critical priority for essential communications
    Critical,
    /// Real-time priority for time-sensitive operations
    RealTime,
}

/// Latency sensitivity levels for communication operations.
///
/// Indicates how sensitive a gradient transmission is to
/// network latency and helps with scheduling decisions.
#[derive(Debug, Clone, PartialEq)]
pub enum LatencySensitivity {
    /// Very low latency sensitivity
    VeryLow,
    /// Low latency sensitivity
    Low,
    /// Moderate latency sensitivity
    Moderate,
    /// High latency sensitivity
    High,
    /// Very high latency sensitivity (real-time)
    VeryHigh,
}

/// Metadata for communication operations including routing and QoS.
///
/// Provides comprehensive information about communication
/// requirements and constraints for gradient transmission.
#[derive(Debug, Clone)]
pub struct CommunicationMetadata {
    /// ID of the source worker
    pub source_worker: u32,
    /// IDs of destination workers
    pub destination_workers: Vec<u32>,
    /// Routing path through the network
    pub routing_path: Vec<u32>,
    /// Quality of service requirements
    pub quality_of_service: QualityOfService,
    /// Security requirements for this communication
    pub security_requirements: SecurityRequirements,
    /// Required reliability level
    pub reliability_level: ReliabilityLevel,
}

/// Quality of Service (QoS) requirements for communication.
///
/// Specifies performance requirements and constraints
/// for gradient transmission operations.
#[derive(Debug, Clone)]
pub struct QualityOfService {
    /// Maximum acceptable latency
    pub max_latency: Duration,
    /// Minimum required bandwidth
    pub min_bandwidth: u64,
    /// Maximum acceptable jitter
    pub max_jitter: Duration,
    /// Tolerance for packet loss (0.0 to 1.0)
    pub packet_loss_tolerance: f64,
    /// Optional energy budget for transmission
    pub energy_budget: Option<f64>,
}

/// Security requirements for communication operations.
///
/// Specifies security and privacy constraints for
/// gradient transmission and aggregation.
#[derive(Debug, Clone)]
pub struct SecurityRequirements {
    /// Whether encryption is required
    pub encryption_required: bool,
    /// Whether integrity checking is required
    pub integrity_check: bool,
    /// Whether authentication is required
    pub authentication_required: bool,
    /// Required privacy level
    pub privacy_level: PrivacyLevel,
}

/// Privacy levels for secure communication.
///
/// Different privacy levels provide various guarantees
/// about information leakage during training.
#[derive(Debug, Clone, PartialEq)]
pub enum PrivacyLevel {
    /// No privacy protection
    None,
    /// Basic privacy protection
    Basic,
    /// Differential privacy protection
    Differential,
    /// Homomorphic encryption protection
    Homomorphic,
    /// Secure multiparty computation protection
    SecureMultiparty,
}

/// Reliability levels for communication operations.
///
/// Different reliability levels provide various guarantees
/// about message delivery and processing.
#[derive(Debug, Clone, PartialEq)]
pub enum ReliabilityLevel {
    /// Best effort delivery with no guarantees
    BestEffort,
    /// At least once delivery guarantee
    AtLeastOnce,
    /// At most once delivery guarantee
    AtMostOnce,
    /// Exactly once delivery guarantee
    ExactlyOnce,
    /// High reliability with additional redundancy
    HighReliability,
}

impl Default for CommunicationConfig {
    fn default() -> Self {
        Self {
            compression_strategy: CompressionStrategy::AdaptiveCompression,
            quantization_bits: 8,
            sparsification_ratio: 0.1,
            error_feedback: true,
            gradient_dropping: false,
            local_accumulation: true,
            bandwidth_adaptation: true,
            protocol_optimization: ProtocolOptimization::AdaptiveProtocol,
            topology: CommunicationTopology::Hierarchical,
            aggregation_method: AggregationMethod::WeightedAverage,
            synchronization_pattern: SynchronizationPattern::LocalSGD,
            fault_tolerance: true,
            adaptive_compression: true,
            communication_budget: Some(1_000_000),
            latency_optimization: true,
            energy_efficiency: true,
        }
    }
}

impl CommunicationEfficientGradient {
    /// Creates a new communication-efficient gradient with default settings.
    ///
    /// # Arguments
    ///
    /// * `gradient` - The original gradient data as a HashMap of parameter names to values
    /// * `worker_id` - ID of the worker that produced this gradient
    /// * `round` - Training round number
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::collections::HashMap;
    /// use torsh_autograd::communication_efficient::config::CommunicationEfficientGradient;
    ///
    /// let mut gradient_data = HashMap::new();
    /// gradient_data.insert("layer1.weight".to_string(), vec![0.1, 0.2, 0.3]);
    /// gradient_data.insert("layer1.bias".to_string(), vec![0.01, 0.02]);
    ///
    /// let gradient = CommunicationEfficientGradient::new(gradient_data, 1, 100);
    /// assert_eq!(gradient.worker_id, 1);
    /// assert_eq!(gradient.round, 100);
    /// ```
    pub fn new(gradient: HashMap<String, Vec<f32>>, worker_id: u32, round: u64) -> Self {
        Self {
            original_gradient: gradient,
            compressed_gradient: CompressedGradient {
                compression_method: CompressionStrategy::None,
                compressed_data: Vec::new(),
                compression_info: CompressionInfo {
                    original_size: 0,
                    compressed_size: 0,
                    quantization_scale: None,
                    sparse_indices: None,
                    low_rank_factors: None,
                    sketch_parameters: None,
                    compression_time: Duration::from_millis(0),
                },
                decompression_hints: DecompressionHints {
                    shape_information: HashMap::new(),
                    sparsity_pattern: None,
                    quantization_bounds: None,
                    reconstruction_method: ReconstructionMethod::Direct,
                },
            },
            compression_ratio: 1.0,
            worker_id,
            round,
            timestamp: Instant::now(),
            priority: CommunicationPriority::Normal,
            bandwidth_requirement: 0,
            latency_sensitivity: LatencySensitivity::Moderate,
            error_feedback: None,
            accumulation_buffer: None,
            metadata: CommunicationMetadata {
                source_worker: worker_id,
                destination_workers: Vec::new(),
                routing_path: Vec::new(),
                quality_of_service: QualityOfService {
                    max_latency: Duration::from_secs(1),
                    min_bandwidth: 1000,
                    max_jitter: Duration::from_millis(100),
                    packet_loss_tolerance: 0.01,
                    energy_budget: None,
                },
                security_requirements: SecurityRequirements {
                    encryption_required: false,
                    integrity_check: true,
                    authentication_required: false,
                    privacy_level: PrivacyLevel::None,
                },
                reliability_level: ReliabilityLevel::AtLeastOnce,
            },
        }
    }

    /// Computes and updates the bandwidth requirement for this gradient.
    ///
    /// The bandwidth requirement is calculated based on the original gradient size
    /// and the compression ratio achieved.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::collections::HashMap;
    /// use torsh_autograd::communication_efficient::config::CommunicationEfficientGradient;
    ///
    /// let mut gradient_data = HashMap::new();
    /// gradient_data.insert("param".to_string(), vec![1.0; 1000]);
    ///
    /// let mut gradient = CommunicationEfficientGradient::new(gradient_data, 1, 1);
    /// gradient.compression_ratio = 0.5;  // 50% compression
    /// gradient.compute_bandwidth_requirement();
    ///
    /// assert!(gradient.bandwidth_requirement > 0);
    /// ```
    pub fn compute_bandwidth_requirement(&mut self) {
        let original_size = self.compute_original_size();
        self.bandwidth_requirement = (original_size as f64 * self.compression_ratio) as u64;
    }

    /// Computes the total size of the original gradient in bytes.
    ///
    /// # Returns
    ///
    /// The total size of all gradient parameters in bytes.
    fn compute_original_size(&self) -> usize {
        self.original_gradient
            .values()
            .map(|v| v.len() * std::mem::size_of::<f32>())
            .sum()
    }

    /// Applies local accumulation by combining with previous gradients.
    ///
    /// This method accumulates the current gradient with previously received
    /// gradients, which can improve communication efficiency by reducing
    /// the frequency of transmissions.
    ///
    /// # Arguments
    ///
    /// * `previous_gradients` - Previous gradients to accumulate with
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::collections::HashMap;
    /// use torsh_autograd::communication_efficient::config::CommunicationEfficientGradient;
    ///
    /// let mut gradient_data = HashMap::new();
    /// gradient_data.insert("param".to_string(), vec![1.0, 2.0]);
    ///
    /// let mut previous = HashMap::new();
    /// previous.insert("param".to_string(), vec![0.5, 1.0]);
    ///
    /// let mut gradient = CommunicationEfficientGradient::new(gradient_data, 1, 1);
    /// gradient.apply_local_accumulation(&previous);
    ///
    /// assert!(gradient.accumulation_buffer.is_some());
    /// ```
    pub fn apply_local_accumulation(&mut self, previous_gradients: &HashMap<String, Vec<f32>>) {
        if self.accumulation_buffer.is_none() {
            self.accumulation_buffer = Some(HashMap::new());
        }

        let buffer = self.accumulation_buffer.as_mut().unwrap();

        for (param_name, gradient) in &self.original_gradient {
            let accumulated = buffer
                .entry(param_name.clone())
                .or_insert_with(|| vec![0.0; gradient.len()]);

            for (i, &grad_value) in gradient.iter().enumerate() {
                accumulated[i] += grad_value;
            }

            if let Some(prev_grad) = previous_gradients.get(param_name) {
                for (i, &prev_value) in prev_grad.iter().enumerate() {
                    accumulated[i] += prev_value;
                }
            }
        }
    }
}

impl Default for QualityOfService {
    fn default() -> Self {
        Self {
            max_latency: Duration::from_millis(100),
            min_bandwidth: 1_000_000, // 1 MB/s
            max_jitter: Duration::from_millis(10),
            packet_loss_tolerance: 0.01, // 1%
            energy_budget: None,
        }
    }
}

impl Default for CommunicationMetadata {
    fn default() -> Self {
        Self {
            source_worker: 0,
            destination_workers: Vec::new(),
            routing_path: Vec::new(),
            quality_of_service: QualityOfService::default(),
            security_requirements: SecurityRequirements::default(),
            reliability_level: ReliabilityLevel::BestEffort,
        }
    }
}

impl Default for SecurityRequirements {
    fn default() -> Self {
        Self {
            encryption_required: false,
            integrity_check: true,
            authentication_required: false,
            privacy_level: PrivacyLevel::None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_communication_config_default() {
        let config = CommunicationConfig::default();
        assert_eq!(
            config.compression_strategy,
            CompressionStrategy::AdaptiveCompression
        );
        assert_eq!(config.quantization_bits, 8);
        assert_eq!(config.sparsification_ratio, 0.1);
        assert!(config.error_feedback);
        assert!(!config.gradient_dropping);
        assert!(config.local_accumulation);
    }

    #[test]
    fn test_communication_efficient_gradient_creation() {
        let mut gradient_data = HashMap::new();
        gradient_data.insert("param_1".to_string(), vec![1.0, 2.0, 3.0]);
        gradient_data.insert("param_2".to_string(), vec![0.1, 0.2]);

        let gradient = CommunicationEfficientGradient::new(gradient_data, 42, 100);

        assert_eq!(gradient.worker_id, 42);
        assert_eq!(gradient.round, 100);
        assert_eq!(gradient.priority, CommunicationPriority::Normal);
        assert_eq!(gradient.latency_sensitivity, LatencySensitivity::Moderate);
        assert_eq!(gradient.compression_ratio, 1.0);
        assert!(gradient.error_feedback.is_none());
        assert!(gradient.accumulation_buffer.is_none());
    }

    #[test]
    fn test_bandwidth_requirement_computation() {
        let mut gradient_data = HashMap::new();
        gradient_data.insert("param".to_string(), vec![1.0; 1000]);

        let mut gradient = CommunicationEfficientGradient::new(gradient_data, 1, 1);
        gradient.compression_ratio = 0.5;
        gradient.compute_bandwidth_requirement();

        let expected_size = 1000 * std::mem::size_of::<f32>();
        let expected_bandwidth = (expected_size as f64 * 0.5) as u64;
        assert_eq!(gradient.bandwidth_requirement, expected_bandwidth);
    }

    #[test]
    fn test_local_accumulation() {
        let mut gradient_data = HashMap::new();
        gradient_data.insert("param".to_string(), vec![1.0, 2.0, 3.0]);

        let mut previous = HashMap::new();
        previous.insert("param".to_string(), vec![0.5, 1.0, 1.5]);

        let mut gradient = CommunicationEfficientGradient::new(gradient_data, 1, 1);
        gradient.apply_local_accumulation(&previous);

        assert!(gradient.accumulation_buffer.is_some());
        let buffer = gradient.accumulation_buffer.unwrap();
        let accumulated = &buffer["param"];
        assert_eq!(accumulated, &vec![1.5, 3.0, 4.5]);
    }

    #[test]
    fn test_communication_priority_ordering() {
        assert!(CommunicationPriority::RealTime > CommunicationPriority::Critical);
        assert!(CommunicationPriority::Critical > CommunicationPriority::High);
        assert!(CommunicationPriority::High > CommunicationPriority::Normal);
        assert!(CommunicationPriority::Normal > CommunicationPriority::Low);
    }

    #[test]
    fn test_compression_strategy_equality() {
        assert_eq!(
            CompressionStrategy::Quantization,
            CompressionStrategy::Quantization
        );
        assert_ne!(
            CompressionStrategy::Quantization,
            CompressionStrategy::Sparsification
        );
        assert_ne!(
            CompressionStrategy::None,
            CompressionStrategy::AdaptiveCompression
        );
    }

    #[test]
    fn test_quality_of_service_defaults() {
        let qos = QualityOfService {
            max_latency: Duration::from_millis(500),
            min_bandwidth: 10_000,
            max_jitter: Duration::from_millis(50),
            packet_loss_tolerance: 0.001,
            energy_budget: Some(100.0),
        };

        assert_eq!(qos.max_latency, Duration::from_millis(500));
        assert_eq!(qos.min_bandwidth, 10_000);
        assert_eq!(qos.packet_loss_tolerance, 0.001);
        assert_eq!(qos.energy_budget, Some(100.0));
    }

    #[test]
    fn test_sketch_parameters() {
        let sketch_params = SketchParameters {
            sketch_size: 1024,
            hash_functions: vec![1, 3, 7, 11],
            random_matrix: vec![0.5; 100],
            sketch_type: SketchType::CountSketch,
        };

        assert_eq!(sketch_params.sketch_size, 1024);
        assert_eq!(sketch_params.hash_functions.len(), 4);
        assert_eq!(sketch_params.random_matrix.len(), 100);
        assert_eq!(sketch_params.sketch_type, SketchType::CountSketch);
    }

    #[test]
    fn test_sparsity_pattern() {
        let pattern = SparsityPattern {
            indices: vec![0, 2, 5, 7],
            values: vec![1.0, 2.0, 3.0, 4.0],
            shape: vec![10],
            density: 0.4,
        };

        assert_eq!(pattern.indices.len(), pattern.values.len());
        assert_eq!(pattern.density, 0.4);
        assert_eq!(pattern.shape, vec![10]);
    }
}
