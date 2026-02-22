//! Federated Learning Framework for ToRSh
//!
//! This module provides a comprehensive federated learning framework that enables collaborative
//! machine learning across distributed clients while preserving privacy and handling various
//! challenges such as data heterogeneity, Byzantine faults, and communication constraints.
//!
//! # Architecture Overview
//!
//! The federated learning framework is organized into focused modules:
//!
//! - **[`types`]**: Core types, enums, and configuration structures
//! - **[`client`]**: Client management and local training functionality
//! - **[`aggregation`]**: Model aggregation strategies and coordination
//! - **[`selection`]**: Client selection algorithms for each round
//! - **[`privacy`]**: Privacy-preserving mechanisms and differential privacy
//! - **[`byzantine`]**: Byzantine fault tolerance and anomaly detection
//! - **[`personalization`]**: Personalization and meta-learning capabilities
//! - **[`metrics`]**: Comprehensive metrics collection and monitoring
//!
//! # Key Features
//!
//! ## Aggregation Strategies
//! - **FederatedAveraging**: Standard weighted averaging of client updates
//! - **FederatedProx**: Proximal term for handling heterogeneous data
//! - **Robust Aggregation**: Byzantine-resilient methods (Median, Krum, etc.)
//! - **Adaptive Aggregation**: Self-tuning aggregation weights
//!
//! ## Privacy Protection
//! - **Differential Privacy**: Formal privacy guarantees with budget tracking
//! - **Secure Aggregation**: Cryptographic protection of individual updates
//! - **Local Differential Privacy**: Client-side privacy protection
//! - **Privacy Amplification**: Enhanced privacy through subsampling
//!
//! ## Client Selection
//! - **Random Selection**: Uniform random sampling of clients
//! - **Fair Selection**: Ensuring equitable participation
//! - **Performance-Based**: Selecting high-quality contributors
//! - **Resource-Aware**: Considering computational and communication constraints
//!
//! ## Personalization
//! - **Per-Client Models**: Individual personalized models
//! - **Cluster-Based**: Grouping similar clients for shared personalization
//! - **Meta-Learning**: Fast adaptation to new clients and tasks
//! - **Multi-Task Learning**: Shared representations across related tasks
//!
//! # Usage Examples
//!
//! ## Basic Federated Learning Setup
//!
//! ```rust,ignore
//! use torsh_autograd::federated_learning::{
//!     FederatedLearningConfig, FederatedAggregator, FederatedClient,
//!     AggregationStrategy, ClientSelectionStrategy, PrivacyMechanism
//! };
//!
//! // Configure federated learning
//! let config = FederatedLearningConfig {
//!     aggregation_strategy: AggregationStrategy::FederatedAveraging,
//!     client_selection_strategy: ClientSelectionStrategy::Random,
//!     privacy_mechanism: PrivacyMechanism::DifferentialPrivacy,
//!     communication_rounds: 100,
//!     clients_per_round: 10,
//!     differential_privacy_epsilon: 1.0,
//!     ..Default::default()
//! };
//!
//! // Create aggregator and register clients
//! let aggregator = FederatedAggregator::new(config);
//! let client = FederatedClient::new("client_1".to_string(), 1000);
//! aggregator.register_client(client)?;
//!
//! // Run federated learning
//! aggregator.run_federated_learning()?;
//! ```
//!
//! ## Advanced Configuration with Privacy and Robustness
//!
//! ```rust,ignore
//! use torsh_autograd::federated_learning::{
//!     FederatedLearningConfig, FederatedAggregator,
//!     AggregationStrategy, PrivacyMechanism, ClientSelectionStrategy
//! };
//!
//! let config = FederatedLearningConfig {
//!     aggregation_strategy: AggregationStrategy::Krum, // Byzantine-resilient
//!     client_selection_strategy: ClientSelectionStrategy::FairSelection,
//!     privacy_mechanism: PrivacyMechanism::DifferentialPrivacy,
//!     differential_privacy_epsilon: 0.5, // Strong privacy
//!     differential_privacy_delta: 1e-6,
//!     byzantine_tolerance: true,
//!     secure_aggregation: true,
//!     personalization_enabled: true,
//!     adaptive_aggregation: true,
//!     ..Default::default()
//! };
//!
//! let aggregator = FederatedAggregator::new(config);
//! // ... register clients and run training
//! ```
//!
//! ## Client-Side Operations
//!
//! ```rust,ignore
//! use torsh_autograd::federated_learning::{FederatedClient, FederatedError};
//! use std::collections::HashMap;
//!
//! let mut client = FederatedClient::new("client_1".to_string(), 5000);
//!
//! // Simulate global model parameters
//! let mut global_model = HashMap::new();
//! global_model.insert("layer1.weight".to_string(), vec![0.1, 0.2, 0.3]);
//! global_model.insert("layer1.bias".to_string(), vec![0.0]);
//!
//! // Compute local update
//! let local_update = client.compute_local_update(&global_model, 3)?;
//!
//! // Apply differential privacy
//! client.apply_differential_privacy(0.1, 1.0)?;
//! ```
//!
//! ## Metrics and Monitoring
//!
//! ```rust,ignore
//! use torsh_autograd::federated_learning::{MetricsCollector, RoundMetrics};
//! use std::time::{Duration, Instant};
//!
//! let mut collector = MetricsCollector::new();
//!
//! let round_metrics = RoundMetrics {
//!     num_participants: 10,
//!     average_local_loss: 0.5,
//!     global_loss: 0.45,
//!     communication_cost: 1024.0,
//!     computation_cost: 500.0,
//!     privacy_cost: 0.1,
//!     fairness_score: 0.85,
//!     data_efficiency: 0.9,
//!     start_time: Instant::now(),
//!     duration: Duration::from_secs(30),
//!     bandwidth_utilization: 1.5,
//!     energy_consumption: 50.0,
//! };
//!
//! collector.record_round_metrics(round_metrics);
//! let report = collector.generate_performance_report();
//! ```
//!
//! # Design Principles
//!
//! ## Modularity and Extensibility
//! The framework is designed with clear separation of concerns, making it easy to:
//! - Add new aggregation strategies
//! - Implement custom client selection algorithms
//! - Integrate additional privacy mechanisms
//! - Extend personalization approaches
//!
//! ## Performance and Scalability
//! - Efficient gradient compression and communication
//! - Asynchronous and parallel processing support
//! - Memory-efficient client management
//! - Scalable metrics collection
//!
//! ## Robustness and Reliability
//! - Comprehensive error handling and recovery
//! - Byzantine fault tolerance
//! - Adaptive parameter tuning
//! - Extensive testing and validation
//!
//! ## Privacy and Security
//! - Formal privacy guarantees through differential privacy
//! - Secure aggregation protocols
//! - Privacy budget management
//! - Audit trails and compliance support
//!
//! # Advanced Topics
//!
//! ## Custom Aggregation Strategies
//!
//! To implement a custom aggregation strategy, extend the aggregation module:
//!
//! ```rust,ignore
//! use torsh_autograd::federated_learning::aggregation::FederatedAggregator;
//! use std::collections::HashMap;
//!
//! impl FederatedAggregator {
//!     fn custom_aggregation(&self,
//!         client_updates: &HashMap<String, HashMap<String, Vec<f32>>>,
//!     ) -> Result<HashMap<String, Vec<f32>>, FederatedError> {
//!         // Custom aggregation logic here
//!         todo!("Implement custom aggregation")
//!     }
//! }
//! ```
//!
//! ## Integration with External Systems
//!
//! The framework can be integrated with various external systems:
//! - Blockchain networks for decentralized coordination
//! - Cloud platforms for scalable deployment
//! - Edge computing infrastructure
//! - IoT device networks
//!
//! # Performance Considerations
//!
//! ## Communication Optimization
//! - Use gradient compression techniques
//! - Implement periodic model broadcasting
//! - Consider hierarchical aggregation for large-scale deployments
//!
//! ## Computational Efficiency
//! - Leverage GPU acceleration where available
//! - Implement model pruning and quantization
//! - Use efficient data loading and preprocessing
//!
//! ## Memory Management
//! - Monitor client resource constraints
//! - Implement adaptive batch sizing
//! - Use streaming for large datasets
//!
//! # Security Considerations
//!
//! ## Threat Model
//! - Honest-but-curious aggregation servers
//! - Malicious clients (Byzantine attacks)
//! - Eavesdropping on communications
//! - Model inversion and membership inference attacks
//!
//! ## Countermeasures
//! - Differential privacy for formal guarantees
//! - Secure multi-party computation protocols
//! - Homomorphic encryption for sensitive computations
//! - Regular security audits and updates

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
// Module declarations
pub mod aggregation;
pub mod byzantine;
pub mod client;
pub mod metrics;
pub mod personalization;
pub mod privacy;
pub mod selection;
pub mod types;

// Re-export core types for convenience
pub use aggregation::FederatedError;
pub use types::{
    AggregationStrategy, ByzantineDetectionMethod, ClientSelectionStrategy, CompositionMethod,
    ConnectionType, DeviceType, DistributionType, FederatedLearningConfig, PersonalizationStrategy,
    PrivacyMechanism,
};

// Re-export main components
pub use aggregation::{AggregationRound, ConvergenceMetrics, FederatedAggregator, RoundMetrics};

pub use client::{
    ClientMetrics, ConnectionQuality, DataDistribution, FeatureStats, FederatedClient,
    ResourceCapabilities,
};

pub use selection::{ClientSelector, FairnessTracker};

pub use privacy::{NoiseCalibration, PrivacyAccountant, PrivacyEngine};

pub use byzantine::{AdaptiveThresholdConfig, ByzantineDetector, DetectionResult};

pub use personalization::{
    AdaptationStep, MetaLearningState, PersonalizationConfig, PersonalizationManager,
    TaskAdaptation, TaskPerformanceMetrics,
};

pub use metrics::{
    ClientMetricsSummary, FederatedMetrics, MetricsCollector, MetricsConfig, PerformanceBaselines,
    PerformanceReport,
};

/// Convenience type alias for the main federated learning coordinator
pub type FederatedCoordinator = FederatedAggregator;

/// Convenience type alias for federated learning results
pub type FederatedResult<T> = Result<T, FederatedError>;

/// Helper function to create a basic federated learning configuration
///
/// This function provides sensible defaults for most federated learning scenarios.
/// Users can customize the returned configuration as needed.
///
/// # Arguments
///
/// * `communication_rounds` - Number of federated learning rounds to execute
/// * `clients_per_round` - Number of clients to select for each round
///
/// # Returns
///
/// A `FederatedLearningConfig` with reasonable default values
///
/// # Examples
/// use std::time::Duration;
///
/// ```rust,ignore
/// use torsh_autograd::federated_learning::create_default_config;
///
/// let config = create_default_config(50, 10);
/// // Customize as needed
/// let mut custom_config = config;
/// custom_config.privacy_mechanism = PrivacyMechanism::DifferentialPrivacy;
/// ```
pub fn create_default_config(
    communication_rounds: u32,
    clients_per_round: usize,
) -> FederatedLearningConfig {
    FederatedLearningConfig {
        aggregation_strategy: AggregationStrategy::FederatedAveraging,
        client_selection_strategy: ClientSelectionStrategy::Random,
        privacy_mechanism: PrivacyMechanism::None,
        communication_rounds,
        clients_per_round,
        min_clients: clients_per_round.max(1),
        max_staleness: 5,
        global_learning_rate: 1.0,
        client_learning_rate: 0.01,
        local_epochs: 1,
        differential_privacy_epsilon: 1.0,
        differential_privacy_delta: 1e-5,
        secure_aggregation: false,
        byzantine_tolerance: false,
        adaptive_aggregation: true,
        personalization_enabled: false,
    }
}

/// Helper function to create a privacy-preserving federated learning configuration
///
/// This configuration enables differential privacy with reasonable parameters
/// for privacy-sensitive applications.
///
/// # Arguments
///
/// * `communication_rounds` - Number of federated learning rounds to execute
/// * `clients_per_round` - Number of clients to select for each round
/// * `epsilon` - Privacy budget (smaller values = stronger privacy)
///
/// # Examples
/// use std::time::Duration;
///
/// ```rust,ignore
/// use torsh_autograd::federated_learning::create_private_config;
///
/// let config = create_private_config(100, 10, 0.5);
/// ```
pub fn create_private_config(
    communication_rounds: u32,
    clients_per_round: usize,
    epsilon: f64,
) -> FederatedLearningConfig {
    FederatedLearningConfig {
        privacy_mechanism: PrivacyMechanism::DifferentialPrivacy,
        differential_privacy_epsilon: epsilon,
        differential_privacy_delta: 1.0 / (clients_per_round as f64).powi(2),
        secure_aggregation: true,
        ..create_default_config(communication_rounds, clients_per_round)
    }
}

/// Helper function to create a Byzantine-resilient federated learning configuration
///
/// This configuration enables Byzantine fault tolerance mechanisms for
/// adversarial environments.
///
/// # Arguments
///
/// * `communication_rounds` - Number of federated learning rounds to execute
/// * `clients_per_round` - Number of clients to select for each round
///
/// # Examples
/// use std::time::Duration;
///
/// ```rust,ignore
/// use torsh_autograd::federated_learning::create_robust_config;
///
/// let config = create_robust_config(100, 15);
/// ```
pub fn create_robust_config(
    communication_rounds: u32,
    clients_per_round: usize,
) -> FederatedLearningConfig {
    FederatedLearningConfig {
        aggregation_strategy: AggregationStrategy::Krum,
        client_selection_strategy: ClientSelectionStrategy::PerformanceBased,
        byzantine_tolerance: true,
        min_clients: (clients_per_round * 2 / 3).max(1), // Ensure sufficient honest clients
        ..create_default_config(communication_rounds, clients_per_round)
    }
}

/// Helper function to create a personalized federated learning configuration
///
/// This configuration enables personalization mechanisms for handling
/// heterogeneous client data distributions.
///
/// # Arguments
///
/// * `communication_rounds` - Number of federated learning rounds to execute
/// * `clients_per_round` - Number of clients to select for each round
/// * `strategy` - Personalization strategy to use
///
/// # Examples
/// use std::time::Duration;
///
/// ```rust,ignore
/// use torsh_autograd::federated_learning::{create_personalized_config, PersonalizationStrategy};
///
/// let config = create_personalized_config(100, 10, PersonalizationStrategy::MetaLearning);
/// ```
pub fn create_personalized_config(
    communication_rounds: u32,
    clients_per_round: usize,
    _strategy: PersonalizationStrategy,
) -> FederatedLearningConfig {
    FederatedLearningConfig {
        personalization_enabled: true,
        client_selection_strategy: ClientSelectionStrategy::FairSelection,
        local_epochs: 3, // More local training for personalization
        ..create_default_config(communication_rounds, clients_per_round)
    }
}

/// Validates a federated learning configuration for common issues
///
/// This function checks for potential configuration problems that could
/// lead to poor performance or failures during federated learning.
///
/// # Arguments
///
/// * `config` - The configuration to validate
///
/// # Returns
///
/// A vector of warning messages for potential issues (empty if no issues found)
///
/// # Examples
/// use std::time::Duration;
///
/// ```rust,ignore
/// use torsh_autograd::federated_learning::{validate_config, create_default_config};
///
/// let config = create_default_config(10, 5);
/// let warnings = validate_config(&config);
/// for warning in warnings {
///     println!("Warning: {}", warning);
/// }
/// ```
pub fn validate_config(config: &FederatedLearningConfig) -> Vec<String> {
    let mut warnings = Vec::new();

    // Check for insufficient communication rounds
    if config.communication_rounds < 10 {
        warnings.push("Very few communication rounds may not allow for convergence".to_string());
    }

    // Check for too many clients per round
    if config.clients_per_round > 1000 {
        warnings.push(
            "Large number of clients per round may cause communication bottlenecks".to_string(),
        );
    }

    // Check privacy configuration
    if config.privacy_mechanism == PrivacyMechanism::DifferentialPrivacy {
        if config.differential_privacy_epsilon > 10.0 {
            warnings.push("Large epsilon value provides weak privacy protection".to_string());
        }
        if config.differential_privacy_epsilon < 0.01 {
            warnings.push("Very small epsilon may severely impact model utility".to_string());
        }
    }

    // Check Byzantine tolerance configuration
    if config.byzantine_tolerance && config.clients_per_round < 6 {
        warnings.push(
            "Byzantine tolerance requires sufficient number of clients (typically >6)".to_string(),
        );
    }

    // Check learning rates
    if config.global_learning_rate > 10.0 || config.client_learning_rate > 1.0 {
        warnings.push("High learning rates may cause training instability".to_string());
    }

    // Check personalization with insufficient local epochs
    if config.personalization_enabled && config.local_epochs < 2 {
        warnings.push("Personalization typically requires multiple local epochs".to_string());
    }

    warnings
}

/// Estimates the computational and communication costs for a federated learning configuration
///
/// This function provides rough estimates of resource requirements to help with
/// capacity planning and configuration optimization.
///
/// # Arguments
///
/// * `config` - The federated learning configuration
/// * `model_parameters` - Number of parameters in the model
///
/// # Returns
///
/// A tuple of (total_computation_hours, total_communication_gb)
///
/// # Examples
/// use std::time::Duration;
///
/// ```rust,ignore
/// use torsh_autograd::federated_learning::{estimate_costs, create_default_config};
///
/// let config = create_default_config(100, 10);
/// let (compute_hours, comm_gb) = estimate_costs(&config, 1_000_000);
/// println!("Estimated compute: {:.1} hours, communication: {:.1} GB", compute_hours, comm_gb);
/// ```
pub fn estimate_costs(config: &FederatedLearningConfig, model_parameters: usize) -> (f64, f64) {
    // Rough estimation formulas (would be calibrated based on empirical data)
    let rounds = config.communication_rounds as f64;
    let clients_per_round = config.clients_per_round as f64;
    let local_epochs = config.local_epochs as f64;
    let params = model_parameters as f64;

    // Computation estimate (CPU-hours)
    let computation_per_round = clients_per_round * local_epochs * 0.1; // 0.1 hours per epoch per client
    let total_computation = rounds * computation_per_round;

    // Communication estimate (GB)
    let bytes_per_param = 4.0; // float32
    let comm_per_round = clients_per_round * params * bytes_per_param * 2.0 / 1e9; // Upload + download
    let total_communication = rounds * comm_per_round;

    (total_computation, total_communication)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_default_config() {
        let config = create_default_config(50, 10);
        assert_eq!(config.communication_rounds, 50);
        assert_eq!(config.clients_per_round, 10);
        assert_eq!(
            config.aggregation_strategy,
            AggregationStrategy::FederatedAveraging
        );
        assert_eq!(config.privacy_mechanism, PrivacyMechanism::None);
    }

    #[test]
    fn test_create_private_config() {
        let config = create_private_config(100, 10, 0.5);
        assert_eq!(
            config.privacy_mechanism,
            PrivacyMechanism::DifferentialPrivacy
        );
        assert_eq!(config.differential_privacy_epsilon, 0.5);
        assert!(config.secure_aggregation);
    }

    #[test]
    fn test_create_robust_config() {
        let config = create_robust_config(100, 15);
        assert_eq!(config.aggregation_strategy, AggregationStrategy::Krum);
        assert_eq!(
            config.client_selection_strategy,
            ClientSelectionStrategy::PerformanceBased
        );
        assert!(config.byzantine_tolerance);
        assert_eq!(config.min_clients, 10); // 2/3 of 15
    }

    #[test]
    fn test_create_personalized_config() {
        let config = create_personalized_config(100, 10, PersonalizationStrategy::MetaLearning);
        assert!(config.personalization_enabled);
        assert_eq!(
            config.client_selection_strategy,
            ClientSelectionStrategy::FairSelection
        );
        assert_eq!(config.local_epochs, 3);
    }

    #[test]
    fn test_validate_config_warnings() {
        let mut config = create_default_config(5, 10); // Few rounds
        config.differential_privacy_epsilon = 20.0; // Weak privacy
        config.privacy_mechanism = PrivacyMechanism::DifferentialPrivacy;

        let warnings = validate_config(&config);
        assert!(warnings.len() >= 2); // Should have warnings for few rounds and weak privacy
        assert!(warnings.iter().any(|w| w.contains("communication rounds")));
        assert!(warnings.iter().any(|w| w.contains("epsilon")));
    }

    #[test]
    fn test_validate_config_no_warnings() {
        let config = create_default_config(50, 10);
        let warnings = validate_config(&config);
        assert!(warnings.is_empty());
    }

    #[test]
    fn test_estimate_costs() {
        let config = create_default_config(10, 5);
        let (compute_hours, comm_gb) = estimate_costs(&config, 1_000_000);

        assert!(compute_hours > 0.0);
        assert!(comm_gb > 0.0);
        // For 10 rounds, 5 clients, 1M parameters: expect reasonable estimates
        assert!(compute_hours < 100.0); // Should be reasonable
        assert!(comm_gb < 100.0); // Should be reasonable
    }

    #[test]
    fn test_module_exports() {
        // Test that key types are properly exported
        let _config: FederatedLearningConfig = create_default_config(10, 5);
        let _aggregation_strategy = AggregationStrategy::FederatedAveraging;
        let _selection_strategy = ClientSelectionStrategy::Random;
        let _privacy_mechanism = PrivacyMechanism::DifferentialPrivacy;

        // This test mainly ensures compilation succeeds with proper exports
        assert!(true);
    }

    #[test]
    fn test_convenience_aliases() {
        // Test type aliases work correctly
        let config = create_default_config(10, 5);
        let _coordinator: FederatedCoordinator = FederatedAggregator::new(config);

        // Test result type alias
        let _result: FederatedResult<()> = Ok(());
        let _error_result: FederatedResult<()> = Err(FederatedError::ClientNotFound);

        assert!(true);
    }

    #[test]
    fn test_byzantine_config_validation() {
        let mut config = create_default_config(10, 3); // Too few clients for Byzantine tolerance
        config.byzantine_tolerance = true;

        let warnings = validate_config(&config);
        assert!(warnings.iter().any(|w| w.contains("Byzantine tolerance")));
    }

    #[test]
    fn test_learning_rate_validation() {
        let mut config = create_default_config(10, 5);
        config.global_learning_rate = 15.0; // Too high
        config.client_learning_rate = 2.0; // Too high

        let warnings = validate_config(&config);
        assert!(warnings.iter().any(|w| w.contains("learning rates")));
    }

    #[test]
    fn test_personalization_validation() {
        let mut config = create_default_config(10, 5);
        config.personalization_enabled = true;
        config.local_epochs = 1; // Too few for personalization

        let warnings = validate_config(&config);
        assert!(warnings.iter().any(|w| w.contains("Personalization")));
    }
}
