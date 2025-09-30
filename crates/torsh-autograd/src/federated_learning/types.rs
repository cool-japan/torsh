//! Core types and configuration for federated learning
//!
//! This module defines the fundamental types, enums, and configuration structures
//! used throughout the federated learning system. It provides the building blocks
//! for federated learning operations including aggregation strategies, client
//! selection methods, privacy mechanisms, and system configuration.

use std::time::Instant;

/// Configuration for federated learning system
///
/// This structure defines all the hyperparameters and settings for a federated
/// learning experiment, including aggregation strategy, privacy settings,
/// and system constraints.
///
/// # Examples
///
/// ```rust,ignore
/// use torsh_autograd::federated_learning::types::*;
///
/// let config = FederatedLearningConfig {
///     aggregation_strategy: AggregationStrategy::FederatedAveraging,
///     client_selection_strategy: ClientSelectionStrategy::Random,
///     privacy_mechanism: PrivacyMechanism::DifferentialPrivacy,
///     communication_rounds: 100,
///     clients_per_round: 10,
///     min_clients: 5,
///     max_staleness: 2,
///     global_learning_rate: 0.01,
///     client_learning_rate: 0.1,
///     local_epochs: 1,
///     differential_privacy_epsilon: 1.0,
///     differential_privacy_delta: 1e-5,
///     secure_aggregation: true,
///     byzantine_tolerance: true,
///     adaptive_aggregation: false,
///     personalization_enabled: false,
/// };
/// ```
#[derive(Debug, Clone)]
pub struct FederatedLearningConfig {
    /// Strategy for aggregating client updates
    pub aggregation_strategy: AggregationStrategy,
    /// Strategy for selecting clients each round
    pub client_selection_strategy: ClientSelectionStrategy,
    /// Privacy preservation mechanism
    pub privacy_mechanism: PrivacyMechanism,
    /// Total number of communication rounds
    pub communication_rounds: u32,
    /// Number of clients to select per round
    pub clients_per_round: usize,
    /// Minimum number of clients required for aggregation
    pub min_clients: usize,
    /// Maximum allowed staleness for asynchronous updates
    pub max_staleness: u32,
    /// Learning rate for global model updates
    pub global_learning_rate: f64,
    /// Learning rate for client local training
    pub client_learning_rate: f64,
    /// Number of local epochs per client
    pub local_epochs: u32,
    /// Epsilon parameter for differential privacy
    pub differential_privacy_epsilon: f64,
    /// Delta parameter for differential privacy
    pub differential_privacy_delta: f64,
    /// Whether to use secure aggregation protocols
    pub secure_aggregation: bool,
    /// Whether to enable Byzantine fault tolerance
    pub byzantine_tolerance: bool,
    /// Whether to use adaptive aggregation weights
    pub adaptive_aggregation: bool,
    /// Whether to enable personalization features
    pub personalization_enabled: bool,
}

impl Default for FederatedLearningConfig {
    fn default() -> Self {
        Self {
            aggregation_strategy: AggregationStrategy::FederatedAveraging,
            client_selection_strategy: ClientSelectionStrategy::Random,
            privacy_mechanism: PrivacyMechanism::None,
            communication_rounds: 100,
            clients_per_round: 10,
            min_clients: 2,
            max_staleness: 0,
            global_learning_rate: 1.0,
            client_learning_rate: 0.01,
            local_epochs: 1,
            differential_privacy_epsilon: 1.0,
            differential_privacy_delta: 1e-5,
            secure_aggregation: false,
            byzantine_tolerance: false,
            adaptive_aggregation: false,
            personalization_enabled: false,
        }
    }
}

/// Strategies for aggregating client updates in federated learning
///
/// Different aggregation strategies provide various trade-offs between
/// convergence speed, robustness to heterogeneity, and Byzantine resilience.
#[derive(Debug, Clone, PartialEq)]
pub enum AggregationStrategy {
    /// Standard federated averaging (FedAvg)
    FederatedAveraging,
    /// Federated averaging with proximal term (FedProx)
    FederatedProx,
    /// Federated Adam optimization
    FederatedAdam,
    /// Federated Yogi optimization
    FederatedYogi,
    /// Median-based aggregation for Byzantine robustness
    MedianAggregation,
    /// Trimmed mean aggregation
    TrimmedMean,
    /// Krum aggregation for Byzantine tolerance
    Krum,
    /// Bulyan aggregation combining multiple robust methods
    BulyanAggregation,
    /// Adaptive weighted aggregation based on client performance
    AdaptiveWeighted,
    /// Hierarchical aggregation for large-scale systems
    HierarchicalAggregation,
}

impl AggregationStrategy {
    /// Check if the strategy is Byzantine-robust
    pub fn is_byzantine_robust(&self) -> bool {
        matches!(
            self,
            AggregationStrategy::MedianAggregation
                | AggregationStrategy::TrimmedMean
                | AggregationStrategy::Krum
                | AggregationStrategy::BulyanAggregation
        )
    }

    /// Check if the strategy supports adaptive weighting
    pub fn supports_adaptive_weighting(&self) -> bool {
        matches!(
            self,
            AggregationStrategy::AdaptiveWeighted | AggregationStrategy::HierarchicalAggregation
        )
    }
}

/// Strategies for selecting which clients participate in each round
///
/// Client selection affects convergence, fairness, and system efficiency.
/// Different strategies optimize for different objectives.
#[derive(Debug, Clone, PartialEq)]
pub enum ClientSelectionStrategy {
    /// Random selection of clients
    Random,
    /// Round-robin selection ensuring all clients participate equally
    RoundRobin,
    /// Select clients proportional to their data size
    ProportionalToData,
    /// Select clients with higher loss (inverse proportional to performance)
    InverseProportionalToLoss,
    /// Diversity-based selection to maximize data heterogeneity
    DiversityBased,
    /// Resource-aware selection based on computation/communication capabilities
    ResourceAware,
    /// Geographic-based selection for edge computing scenarios
    GeographicBased,
    /// Performance-based selection favoring high-performing clients
    PerformanceBased,
    /// Fair selection ensuring equitable participation
    FairSelection,
    /// Adaptive selection that changes strategy based on system state
    Adaptive,
}

impl ClientSelectionStrategy {
    /// Check if the strategy considers fairness
    pub fn considers_fairness(&self) -> bool {
        matches!(
            self,
            ClientSelectionStrategy::RoundRobin
                | ClientSelectionStrategy::FairSelection
                | ClientSelectionStrategy::DiversityBased
        )
    }

    /// Check if the strategy requires performance metrics
    pub fn requires_performance_metrics(&self) -> bool {
        matches!(
            self,
            ClientSelectionStrategy::InverseProportionalToLoss
                | ClientSelectionStrategy::PerformanceBased
                | ClientSelectionStrategy::Adaptive
        )
    }
}

/// Privacy preservation mechanisms for federated learning
///
/// Different privacy mechanisms provide various levels of privacy protection
/// with different computational and communication overhead.
#[derive(Debug, Clone, PartialEq)]
pub enum PrivacyMechanism {
    /// No privacy protection
    None,
    /// Differential privacy with Gaussian noise
    DifferentialPrivacy,
    /// Gaussian noise addition
    GaussianNoise,
    /// Laplace noise addition
    LaplaceNoise,
    /// Secure aggregation protocols
    SecureAggregation,
    /// Homomorphic encryption
    HomomorphicEncryption,
    /// Multi-party computation
    MultiPartyComputation,
    /// Local differential privacy
    LocalDifferentialPrivacy,
    /// Privacy amplification through subsampling
    PrivacyAmplification,
}

impl PrivacyMechanism {
    /// Check if the mechanism provides formal privacy guarantees
    pub fn provides_formal_guarantees(&self) -> bool {
        matches!(
            self,
            PrivacyMechanism::DifferentialPrivacy | PrivacyMechanism::LocalDifferentialPrivacy
        )
    }

    /// Check if the mechanism requires cryptographic protocols
    pub fn requires_cryptography(&self) -> bool {
        matches!(
            self,
            PrivacyMechanism::SecureAggregation
                | PrivacyMechanism::HomomorphicEncryption
                | PrivacyMechanism::MultiPartyComputation
        )
    }

    /// Get the computational overhead level
    pub fn computational_overhead(&self) -> OverheadLevel {
        match self {
            PrivacyMechanism::None => OverheadLevel::None,
            PrivacyMechanism::GaussianNoise | PrivacyMechanism::LaplaceNoise => OverheadLevel::Low,
            PrivacyMechanism::DifferentialPrivacy
            | PrivacyMechanism::LocalDifferentialPrivacy
            | PrivacyMechanism::PrivacyAmplification => OverheadLevel::Medium,
            PrivacyMechanism::SecureAggregation => OverheadLevel::High,
            PrivacyMechanism::HomomorphicEncryption | PrivacyMechanism::MultiPartyComputation => {
                OverheadLevel::VeryHigh
            }
        }
    }
}

/// Overhead levels for different mechanisms
#[derive(Debug, Clone, PartialEq)]
pub enum OverheadLevel {
    None,
    Low,
    Medium,
    High,
    VeryHigh,
}

/// Device types for federated learning clients
///
/// Different device types have different computational capabilities
/// and resource constraints.
#[derive(Debug, Clone, PartialEq)]
pub enum DeviceType {
    /// Mobile devices (smartphones, tablets)
    Mobile,
    /// Desktop computers
    Desktop,
    /// Server-class machines
    Server,
    /// Edge computing devices
    Edge,
    /// IoT devices with limited resources
    IoT,
    /// Unknown or unspecified device type
    Unknown,
}

impl DeviceType {
    /// Get the relative computational capability
    pub fn computational_capability(&self) -> f64 {
        match self {
            DeviceType::IoT => 0.1,
            DeviceType::Mobile => 0.3,
            DeviceType::Edge => 0.5,
            DeviceType::Desktop => 0.8,
            DeviceType::Server => 1.0,
            DeviceType::Unknown => 0.5,
        }
    }

    /// Check if the device is resource-constrained
    pub fn is_resource_constrained(&self) -> bool {
        matches!(
            self,
            DeviceType::Mobile | DeviceType::IoT | DeviceType::Edge
        )
    }
}

/// Data distribution types for analyzing client data heterogeneity
///
/// Understanding data distribution helps in designing appropriate
/// aggregation and personalization strategies.
#[derive(Debug, Clone, PartialEq)]
pub enum DistributionType {
    /// Independent and identically distributed data
    IID,
    /// Non-IID data with feature skew
    NonIIDFeature,
    /// Non-IID data with label skew
    NonIIDLabel,
    /// Non-IID data with both feature and label skew
    NonIIDMixed,
    /// Dirichlet distribution
    Dirichlet,
    /// Pathological non-IID (extreme heterogeneity)
    Pathological,
    /// Unknown distribution pattern
    Unknown,
}

impl DistributionType {
    /// Check if the distribution is non-IID
    pub fn is_non_iid(&self) -> bool {
        !matches!(self, DistributionType::IID)
    }

    /// Get the heterogeneity level
    pub fn heterogeneity_level(&self) -> f64 {
        match self {
            DistributionType::IID => 0.0,
            DistributionType::NonIIDFeature => 0.3,
            DistributionType::NonIIDLabel => 0.5,
            DistributionType::Dirichlet => 0.7,
            DistributionType::NonIIDMixed => 0.8,
            DistributionType::Pathological => 1.0,
            DistributionType::Unknown => 0.5,
        }
    }
}

/// Connection quality and type information for clients
///
/// Network characteristics affect communication efficiency and
/// client selection decisions.
#[derive(Debug, Clone, PartialEq)]
pub enum ConnectionType {
    /// Wi-Fi connection
    WiFi,
    /// Cellular data connection
    Cellular,
    /// Ethernet connection
    Ethernet,
    /// Satellite connection
    Satellite,
    /// Unknown connection type
    Unknown,
}

impl ConnectionType {
    /// Get the expected bandwidth (Mbps)
    pub fn expected_bandwidth(&self) -> f64 {
        match self {
            ConnectionType::Ethernet => 1000.0,
            ConnectionType::WiFi => 100.0,
            ConnectionType::Cellular => 50.0,
            ConnectionType::Satellite => 25.0,
            ConnectionType::Unknown => 50.0,
        }
    }

    /// Get the expected latency (ms)
    pub fn expected_latency(&self) -> f64 {
        match self {
            ConnectionType::Ethernet => 1.0,
            ConnectionType::WiFi => 5.0,
            ConnectionType::Cellular => 50.0,
            ConnectionType::Satellite => 600.0,
            ConnectionType::Unknown => 50.0,
        }
    }

    /// Check if the connection is reliable
    pub fn is_reliable(&self) -> bool {
        matches!(self, ConnectionType::Ethernet | ConnectionType::WiFi)
    }
}

/// Composition methods for differential privacy
///
/// Different composition methods provide different privacy guarantees
/// when multiple mechanisms are applied.
#[derive(Debug, Clone, PartialEq)]
pub enum CompositionMethod {
    /// Basic composition
    Basic,
    /// Advanced composition
    Advanced,
    /// Moments accountant
    MomentsAccountant,
    /// Renyi differential privacy
    RenyiDP,
}

impl CompositionMethod {
    /// Check if the method provides tight bounds
    pub fn provides_tight_bounds(&self) -> bool {
        matches!(
            self,
            CompositionMethod::MomentsAccountant | CompositionMethod::RenyiDP
        )
    }
}

/// Personalization strategies for federated learning
///
/// Different strategies for creating personalized models while
/// maintaining privacy and leveraging collective knowledge.
#[derive(Debug, Clone, PartialEq)]
pub enum PersonalizationStrategy {
    /// No personalization - use global model
    None,
    /// Per-client personalization
    PerClient,
    /// Clustered personalization
    Clustered,
    /// Meta-learning based personalization
    MetaLearning,
    /// Multi-task learning personalization
    MultiTask,
    /// Fine-tuning based personalization
    FineTuning,
}

impl PersonalizationStrategy {
    /// Check if the strategy requires clustering
    pub fn requires_clustering(&self) -> bool {
        matches!(self, PersonalizationStrategy::Clustered)
    }

    /// Check if the strategy uses meta-learning
    pub fn uses_meta_learning(&self) -> bool {
        matches!(self, PersonalizationStrategy::MetaLearning)
    }
}

/// Byzantine detection methods for identifying malicious clients
///
/// Different methods for detecting and handling Byzantine (malicious)
/// clients in federated learning systems.
#[derive(Debug, Clone, PartialEq)]
pub enum ByzantineDetectionMethod {
    /// No detection
    None,
    /// Statistical anomaly detection
    Statistical,
    /// Gradient similarity analysis
    GradientSimilarity,
    /// Loss-based detection
    LossBased,
    /// Reputation-based system
    ReputationBased,
    /// Multi-modal detection combining multiple methods
    MultiModal,
}

impl ByzantineDetectionMethod {
    /// Check if the method requires historical data
    pub fn requires_history(&self) -> bool {
        matches!(
            self,
            ByzantineDetectionMethod::Statistical | ByzantineDetectionMethod::ReputationBased
        )
    }

    /// Get the detection accuracy (estimated)
    pub fn detection_accuracy(&self) -> f64 {
        match self {
            ByzantineDetectionMethod::None => 0.0,
            ByzantineDetectionMethod::Statistical => 0.7,
            ByzantineDetectionMethod::GradientSimilarity => 0.8,
            ByzantineDetectionMethod::LossBased => 0.6,
            ByzantineDetectionMethod::ReputationBased => 0.85,
            ByzantineDetectionMethod::MultiModal => 0.9,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = FederatedLearningConfig::default();
        assert_eq!(
            config.aggregation_strategy,
            AggregationStrategy::FederatedAveraging
        );
        assert_eq!(
            config.client_selection_strategy,
            ClientSelectionStrategy::Random
        );
        assert_eq!(config.privacy_mechanism, PrivacyMechanism::None);
    }

    #[test]
    fn test_aggregation_strategy_properties() {
        assert!(AggregationStrategy::MedianAggregation.is_byzantine_robust());
        assert!(!AggregationStrategy::FederatedAveraging.is_byzantine_robust());
        assert!(AggregationStrategy::AdaptiveWeighted.supports_adaptive_weighting());
    }

    #[test]
    fn test_client_selection_properties() {
        assert!(ClientSelectionStrategy::FairSelection.considers_fairness());
        assert!(ClientSelectionStrategy::PerformanceBased.requires_performance_metrics());
        assert!(!ClientSelectionStrategy::Random.requires_performance_metrics());
    }

    #[test]
    fn test_privacy_mechanism_properties() {
        assert!(PrivacyMechanism::DifferentialPrivacy.provides_formal_guarantees());
        assert!(!PrivacyMechanism::GaussianNoise.provides_formal_guarantees());
        assert!(PrivacyMechanism::SecureAggregation.requires_cryptography());
        assert_eq!(
            PrivacyMechanism::None.computational_overhead(),
            OverheadLevel::None
        );
    }

    #[test]
    fn test_device_type_capabilities() {
        assert_eq!(DeviceType::Server.computational_capability(), 1.0);
        assert_eq!(DeviceType::IoT.computational_capability(), 0.1);
        assert!(DeviceType::Mobile.is_resource_constrained());
        assert!(!DeviceType::Server.is_resource_constrained());
    }

    #[test]
    fn test_distribution_type_properties() {
        assert!(!DistributionType::IID.is_non_iid());
        assert!(DistributionType::NonIIDFeature.is_non_iid());
        assert_eq!(DistributionType::IID.heterogeneity_level(), 0.0);
        assert_eq!(DistributionType::Pathological.heterogeneity_level(), 1.0);
    }

    #[test]
    fn test_connection_type_characteristics() {
        assert_eq!(ConnectionType::Ethernet.expected_bandwidth(), 1000.0);
        assert_eq!(ConnectionType::Satellite.expected_latency(), 600.0);
        assert!(ConnectionType::WiFi.is_reliable());
        assert!(!ConnectionType::Satellite.is_reliable());
    }

    #[test]
    fn test_byzantine_detection_properties() {
        assert!(ByzantineDetectionMethod::ReputationBased.requires_history());
        assert!(!ByzantineDetectionMethod::GradientSimilarity.requires_history());
        assert_eq!(ByzantineDetectionMethod::None.detection_accuracy(), 0.0);
        assert_eq!(
            ByzantineDetectionMethod::MultiModal.detection_accuracy(),
            0.9
        );
    }
}
