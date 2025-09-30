//! Client management for federated learning
//!
//! This module provides structures and functionality for managing federated learning
//! clients, including their capabilities, metrics, data distributions, and
//! connection quality. It supports comprehensive client profiling for optimal
//! selection and resource management.

use std::collections::{HashMap, HashSet};
use std::time::Duration;

use super::types::{ConnectionType, DeviceType, DistributionType};

/// A federated learning client with comprehensive metadata
///
/// This structure represents a client in the federated learning system,
/// containing all necessary information for client selection, aggregation
/// weighting, and personalization decisions.
///
/// # Examples
///
/// ```rust,ignore
/// use torsh_autograd::federated_learning::client::*;
/// use std::collections::HashMap;
///
/// let mut client = FederatedClient {
///     client_id: "client_001".to_string(),
///     data_size: 1000,
///     model_parameters: HashMap::new(),
///     local_gradients: HashMap::new(),
///     client_metrics: ClientMetrics::default(),
///     privacy_budget: 1.0,
///     resource_capabilities: ResourceCapabilities::default(),
///     data_distribution: DataDistribution::default(),
///     connection_quality: ConnectionQuality::default(),
///     trust_score: 1.0,
///     last_participation_round: 0,
///     personalized_layers: HashSet::new(),
/// };
/// ```
#[derive(Debug, Clone)]
pub struct FederatedClient {
    /// Unique identifier for the client
    pub client_id: String,
    /// Size of the client's local dataset
    pub data_size: usize,
    /// Current model parameters for the client
    pub model_parameters: HashMap<String, Vec<f32>>,
    /// Local gradients computed by the client
    pub local_gradients: HashMap<String, Vec<f32>>,
    /// Performance and quality metrics for the client
    pub client_metrics: ClientMetrics,
    /// Remaining privacy budget for differential privacy
    pub privacy_budget: f64,
    /// Client's computational and communication capabilities
    pub resource_capabilities: ResourceCapabilities,
    /// Statistical information about client's data distribution
    pub data_distribution: DataDistribution,
    /// Network connection quality and characteristics
    pub connection_quality: ConnectionQuality,
    /// Trust score based on historical behavior
    pub trust_score: f64,
    /// Last round in which the client participated
    pub last_participation_round: u32,
    /// Set of layers that are personalized for this client
    pub personalized_layers: HashSet<String>,
}

impl FederatedClient {
    /// Create a new federated client with the given ID
    ///
    /// # Arguments
    ///
    /// * `client_id` - Unique identifier for the client
    ///
    /// # Returns
    ///
    /// A new `FederatedClient` instance with default values
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let client = FederatedClient::new("client_001".to_string());
    /// assert_eq!(client.client_id, "client_001");
    /// ```
    pub fn new(client_id: String) -> Self {
        Self {
            client_id,
            data_size: 0,
            model_parameters: HashMap::new(),
            local_gradients: HashMap::new(),
            client_metrics: ClientMetrics::default(),
            privacy_budget: 1.0,
            resource_capabilities: ResourceCapabilities::default(),
            data_distribution: DataDistribution::default(),
            connection_quality: ConnectionQuality::default(),
            trust_score: 1.0,
            last_participation_round: 0,
            personalized_layers: HashSet::new(),
        }
    }

    /// Check if the client is eligible for participation
    ///
    /// Considers privacy budget, trust score, and resource availability
    ///
    /// # Returns
    ///
    /// `true` if the client can participate, `false` otherwise
    pub fn is_eligible(&self) -> bool {
        self.privacy_budget > 0.0
            && self.trust_score > 0.5
            && self.resource_capabilities.is_available()
    }

    /// Calculate the client's selection priority score
    ///
    /// Higher scores indicate higher priority for selection
    ///
    /// # Returns
    ///
    /// A priority score between 0.0 and 1.0
    pub fn selection_priority(&self) -> f64 {
        let data_weight = (self.data_size as f64).ln() / 10.0; // Log scale for data size
        let resource_weight = self.resource_capabilities.compute_power;
        let quality_weight = self.client_metrics.data_quality_score;
        let trust_weight = self.trust_score;
        let connection_weight = self.connection_quality.stability_score;

        (data_weight + resource_weight + quality_weight + trust_weight + connection_weight) / 5.0
    }

    /// Update the client's trust score based on behavior
    ///
    /// # Arguments
    ///
    /// * `behavior_score` - Score from 0.0 to 1.0 representing recent behavior
    /// * `decay_factor` - Factor for exponential moving average (0.0 to 1.0)
    pub fn update_trust_score(&mut self, behavior_score: f64, decay_factor: f64) {
        self.trust_score = decay_factor * self.trust_score + (1.0 - decay_factor) * behavior_score;
        // Clamp to valid range
        self.trust_score = self.trust_score.clamp(0.0, 1.0);
    }

    /// Check if the client has sufficient privacy budget
    ///
    /// # Arguments
    ///
    /// * `required_budget` - Required privacy budget for the operation
    ///
    /// # Returns
    ///
    /// `true` if the client has sufficient budget
    pub fn has_privacy_budget(&self, required_budget: f64) -> bool {
        self.privacy_budget >= required_budget
    }

    /// Consume privacy budget for an operation
    ///
    /// # Arguments
    ///
    /// * `consumed_budget` - Amount of privacy budget to consume
    ///
    /// # Returns
    ///
    /// `true` if the budget was successfully consumed, `false` if insufficient
    pub fn consume_privacy_budget(&mut self, consumed_budget: f64) -> bool {
        if self.privacy_budget >= consumed_budget {
            self.privacy_budget -= consumed_budget;
            true
        } else {
            false
        }
    }

    /// Add a layer to the personalized layers set
    ///
    /// # Arguments
    ///
    /// * `layer_name` - Name of the layer to personalize
    pub fn add_personalized_layer(&mut self, layer_name: String) {
        self.personalized_layers.insert(layer_name);
    }

    /// Check if a layer is personalized for this client
    ///
    /// # Arguments
    ///
    /// * `layer_name` - Name of the layer to check
    ///
    /// # Returns
    ///
    /// `true` if the layer is personalized
    pub fn is_layer_personalized(&self, layer_name: &str) -> bool {
        self.personalized_layers.contains(layer_name)
    }

    /// Compute local model updates based on global model and local training
    ///
    /// # Arguments
    ///
    /// * `global_model` - The current global model parameters
    /// * `local_epochs` - Number of local training epochs to perform
    ///
    /// # Returns
    ///
    /// Result containing the local model updates or federated learning error
    pub fn compute_local_update(
        &mut self,
        global_model: &HashMap<String, Vec<f32>>,
        local_epochs: u32,
    ) -> Result<HashMap<String, Vec<f32>>, super::aggregation::FederatedError> {
        // Initialize local gradients based on global model
        self.model_parameters = global_model.clone();

        // Simulate local training by computing gradients
        // In a real implementation, this would perform actual model training
        for (param_name, global_params) in global_model.iter() {
            let local_gradients: Vec<f32> = global_params
                .iter()
                .enumerate()
                .map(|(i, &param)| {
                    // Simulate gradient computation with some client-specific variation
                    let gradient_noise = (i as f32 * 0.01) * self.trust_score as f32;
                    param * 0.01 + gradient_noise // Simple gradient simulation
                })
                .collect();

            self.local_gradients
                .insert(param_name.clone(), local_gradients);
        }

        // Apply local epochs factor
        if local_epochs > 1 {
            for gradients in self.local_gradients.values_mut() {
                for gradient in gradients.iter_mut() {
                    *gradient *= local_epochs as f32;
                }
            }
        }

        Ok(self.local_gradients.clone())
    }

    /// Apply differential privacy to the client's local gradients
    ///
    /// # Arguments
    ///
    /// * `epsilon` - Privacy budget parameter (smaller = more private)
    /// * `sensitivity` - Sensitivity of the gradient computation
    ///
    /// # Returns
    ///
    /// Result indicating success or failure of privacy application
    pub fn apply_differential_privacy(
        &mut self,
        epsilon: f64,
        sensitivity: f64,
    ) -> Result<(), super::aggregation::FederatedError> {
        // Check if client has sufficient privacy budget
        let required_budget = epsilon;
        if !self.has_privacy_budget(required_budget) {
            return Err(super::aggregation::FederatedError::PrivacyBudgetExceeded);
        }

        // Apply Gaussian noise to gradients based on differential privacy
        // Noise scale = (sensitivity * sqrt(2 * ln(1.25/delta))) / epsilon
        let delta = 1e-5_f64; // Standard delta value
        let noise_scale = (sensitivity * (2.0_f64 * (1.25_f64 / delta).ln()).sqrt()) / epsilon;

        // Add noise to each gradient component
        for gradients in self.local_gradients.values_mut() {
            for gradient in gradients.iter_mut() {
                // Use simple pseudo-random noise for simulation
                // In production, this would use cryptographically secure randomness
                let noise = Self::sample_gaussian_noise(0.0, noise_scale);
                *gradient += noise as f32;
            }
        }

        // Consume privacy budget
        self.consume_privacy_budget(required_budget);

        Ok(())
    }

    /// Sample Gaussian noise for differential privacy
    ///
    /// # Arguments
    ///
    /// * `mean` - Mean of the Gaussian distribution
    /// * `std_dev` - Standard deviation of the Gaussian distribution
    ///
    /// # Returns
    ///
    /// A sample from the Gaussian distribution
    fn sample_gaussian_noise(mean: f64, std_dev: f64) -> f64 {
        // Simple Box-Muller transform for Gaussian noise
        // In production, use a proper cryptographic random number generator
        use std::f64::consts::PI;

        static mut U1: Option<f64> = None;
        static mut U2: Option<f64> = None;

        unsafe {
            if U1.is_none() {
                // Generate two uniform random numbers
                let u1 = 0.5 + (std::ptr::addr_of!(mean) as usize % 1000) as f64 / 2000.0;
                let u2 = 0.5 + (std::ptr::addr_of!(std_dev) as usize % 1000) as f64 / 2000.0;

                // Box-Muller transform
                let z1 = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
                let z2 = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).sin();

                U1 = Some(z1);
                U2 = Some(z2);
            }

            let noise = U1.take().unwrap_or(0.0);
            mean + std_dev * noise
        }
    }
}

/// Performance and quality metrics for a federated learning client
///
/// These metrics are used for client selection, aggregation weighting,
/// and system monitoring.
#[derive(Debug, Clone)]
pub struct ClientMetrics {
    /// Local loss achieved by the client
    pub local_loss: f64,
    /// Local accuracy achieved by the client
    pub local_accuracy: f64,
    /// Time spent on local computation
    pub computation_time: Duration,
    /// Time spent on communication
    pub communication_time: Duration,
    /// Quality score of the client's data (0.0 to 1.0)
    pub data_quality_score: f64,
    /// Norm of the client's gradient updates
    pub gradient_norm: f64,
    /// Staleness of the client's updates
    pub staleness: u32,
    /// Rate of participation in recent rounds
    pub participation_rate: f64,
    /// Contribution score based on impact on global model
    pub contribution_score: f64,
}

impl Default for ClientMetrics {
    fn default() -> Self {
        Self {
            local_loss: 1.0,
            local_accuracy: 0.0,
            computation_time: Duration::from_secs(0),
            communication_time: Duration::from_secs(0),
            data_quality_score: 1.0,
            gradient_norm: 0.0,
            staleness: 0,
            participation_rate: 0.0,
            contribution_score: 0.0,
        }
    }
}

impl ClientMetrics {
    /// Update metrics after a training round
    ///
    /// # Arguments
    ///
    /// * `loss` - New local loss value
    /// * `accuracy` - New local accuracy value
    /// * `comp_time` - Computation time for this round
    /// * `comm_time` - Communication time for this round
    pub fn update_round_metrics(
        &mut self,
        loss: f64,
        accuracy: f64,
        comp_time: Duration,
        comm_time: Duration,
    ) {
        self.local_loss = loss;
        self.local_accuracy = accuracy;
        self.computation_time = comp_time;
        self.communication_time = comm_time;
    }

    /// Calculate overall client quality score
    ///
    /// # Returns
    ///
    /// A composite quality score between 0.0 and 1.0
    pub fn overall_quality_score(&self) -> f64 {
        let accuracy_weight = self.local_accuracy;
        let data_weight = self.data_quality_score;
        let contribution_weight = self.contribution_score;
        let participation_weight = self.participation_rate;

        (accuracy_weight + data_weight + contribution_weight + participation_weight) / 4.0
    }

    /// Check if the client shows signs of improvement
    ///
    /// # Returns
    ///
    /// `true` if metrics indicate improvement
    pub fn is_improving(&self) -> bool {
        self.local_accuracy > 0.1 && self.contribution_score > 0.0
    }
}

/// Computational and communication capabilities of a client
///
/// Used for resource-aware client selection and load balancing.
#[derive(Debug, Clone)]
pub struct ResourceCapabilities {
    /// Relative compute power (0.0 to 1.0)
    pub compute_power: f64,
    /// Memory capacity in bytes
    pub memory_capacity: usize,
    /// Network bandwidth in Mbps
    pub bandwidth: f64,
    /// Battery level for mobile devices (0.0 to 1.0)
    pub battery_level: Option<f64>,
    /// Storage capacity in bytes
    pub storage_capacity: usize,
    /// Whether the device is mobile
    pub is_mobile: bool,
    /// Type of device
    pub device_type: DeviceType,
}

impl Default for ResourceCapabilities {
    fn default() -> Self {
        Self {
            compute_power: 0.5,
            memory_capacity: 1_073_741_824, // 1 GB
            bandwidth: 10.0,                // 10 Mbps
            battery_level: Some(1.0),
            storage_capacity: 10_737_418_240, // 10 GB
            is_mobile: false,
            device_type: DeviceType::Unknown,
        }
    }
}

impl ResourceCapabilities {
    /// Check if the client is available for participation
    ///
    /// Considers battery level, memory, and other constraints
    ///
    /// # Returns
    ///
    /// `true` if the client is available
    pub fn is_available(&self) -> bool {
        if let Some(battery) = self.battery_level {
            if self.is_mobile && battery < 0.2 {
                return false; // Low battery on mobile device
            }
        }

        self.memory_capacity > 536_870_912 // At least 512 MB
            && self.bandwidth > 1.0 // At least 1 Mbps
    }

    /// Calculate resource utilization score
    ///
    /// # Returns
    ///
    /// A score indicating how well the client can handle the workload
    pub fn utilization_score(&self) -> f64 {
        let compute_score = self.compute_power;
        let memory_score = (self.memory_capacity as f64 / 8_589_934_592.0).min(1.0); // 8 GB reference
        let bandwidth_score = (self.bandwidth / 100.0).min(1.0); // 100 Mbps reference

        let battery_score = if let Some(battery) = self.battery_level {
            if self.is_mobile {
                battery
            } else {
                1.0 // Desktop devices don't have battery constraints
            }
        } else {
            1.0
        };

        (compute_score + memory_score + bandwidth_score + battery_score) / 4.0
    }

    /// Get expected training time based on capabilities
    ///
    /// # Arguments
    ///
    /// * `base_time` - Base training time for reference device
    ///
    /// # Returns
    ///
    /// Expected training time for this client
    pub fn expected_training_time(&self, base_time: Duration) -> Duration {
        let time_factor = 1.0 / self.compute_power.max(0.1);
        Duration::from_secs_f64(base_time.as_secs_f64() * time_factor)
    }
}

/// Statistical information about a client's data distribution
///
/// Used for understanding data heterogeneity and designing appropriate
/// aggregation strategies.
#[derive(Debug, Clone)]
pub struct DataDistribution {
    /// Type of data distribution (IID, Non-IID, etc.)
    pub distribution_type: DistributionType,
    /// Distribution of classes/labels in the data
    pub class_distribution: HashMap<String, f64>,
    /// Statistical features of the data
    pub feature_statistics: HashMap<String, FeatureStats>,
    /// How recent/fresh the data is (0.0 to 1.0)
    pub data_freshness: f64,
    /// Level of noise in the labels (0.0 to 1.0)
    pub label_noise_level: f64,
}

impl Default for DataDistribution {
    fn default() -> Self {
        Self {
            distribution_type: DistributionType::Unknown,
            class_distribution: HashMap::new(),
            feature_statistics: HashMap::new(),
            data_freshness: 1.0,
            label_noise_level: 0.0,
        }
    }
}

impl DataDistribution {
    /// Calculate the diversity score of this client's data
    ///
    /// # Returns
    ///
    /// A diversity score between 0.0 and 1.0
    pub fn diversity_score(&self) -> f64 {
        if self.class_distribution.is_empty() {
            return 0.5; // Unknown diversity
        }

        // Calculate entropy of class distribution
        let entropy: f64 = self
            .class_distribution
            .values()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.ln())
            .sum();

        // Normalize by maximum possible entropy
        let max_entropy = (self.class_distribution.len() as f64).ln();
        if max_entropy > 0.0 {
            entropy / max_entropy
        } else {
            0.0
        }
    }

    /// Check if the data distribution is non-IID
    ///
    /// # Returns
    ///
    /// `true` if the distribution is non-IID
    pub fn is_non_iid(&self) -> bool {
        match self.distribution_type {
            DistributionType::IID => false,
            _ => true,
        }
    }

    /// Get the quality score of the data
    ///
    /// # Returns
    ///
    /// A quality score between 0.0 and 1.0
    pub fn quality_score(&self) -> f64 {
        let freshness_weight = self.data_freshness;
        let noise_weight = 1.0 - self.label_noise_level;
        let diversity_weight = self.diversity_score();

        (freshness_weight + noise_weight + diversity_weight) / 3.0
    }
}

/// Statistical features of data dimensions
///
/// Provides detailed statistics for understanding data characteristics.
#[derive(Debug, Clone)]
pub struct FeatureStats {
    /// Mean value of the feature
    pub mean: f64,
    /// Variance of the feature
    pub variance: f64,
    /// Minimum value observed
    pub min: f64,
    /// Maximum value observed
    pub max: f64,
    /// Skewness of the distribution
    pub skewness: f64,
    /// Kurtosis of the distribution
    pub kurtosis: f64,
}

impl Default for FeatureStats {
    fn default() -> Self {
        Self {
            mean: 0.0,
            variance: 1.0,
            min: 0.0,
            max: 1.0,
            skewness: 0.0,
            kurtosis: 0.0,
        }
    }
}

impl FeatureStats {
    /// Calculate the standard deviation
    ///
    /// # Returns
    ///
    /// The standard deviation of the feature
    pub fn std_dev(&self) -> f64 {
        self.variance.sqrt()
    }

    /// Check if the feature distribution is approximately normal
    ///
    /// # Returns
    ///
    /// `true` if the distribution appears normal
    pub fn is_approximately_normal(&self) -> bool {
        self.skewness.abs() < 2.0 && self.kurtosis.abs() < 7.0
    }
}

/// Network connection quality and characteristics
///
/// Used for communication-aware client selection and scheduling.
#[derive(Debug, Clone)]
pub struct ConnectionQuality {
    /// Network latency
    pub latency: Duration,
    /// Available bandwidth in Mbps
    pub bandwidth: f64,
    /// Packet loss rate (0.0 to 1.0)
    pub packet_loss_rate: f64,
    /// Connection stability score (0.0 to 1.0)
    pub stability_score: f64,
    /// Type of network connection
    pub connection_type: ConnectionType,
}

impl Default for ConnectionQuality {
    fn default() -> Self {
        Self {
            latency: Duration::from_millis(50),
            bandwidth: 10.0,
            packet_loss_rate: 0.01,
            stability_score: 0.8,
            connection_type: ConnectionType::Unknown,
        }
    }
}

impl ConnectionQuality {
    /// Calculate overall connection quality score
    ///
    /// # Returns
    ///
    /// A quality score between 0.0 and 1.0
    pub fn quality_score(&self) -> f64 {
        let latency_score = 1.0 - (self.latency.as_millis() as f64 / 1000.0).min(1.0);
        let bandwidth_score = (self.bandwidth / 100.0).min(1.0);
        let loss_score = 1.0 - self.packet_loss_rate;
        let stability_score = self.stability_score;

        (latency_score + bandwidth_score + loss_score + stability_score) / 4.0
    }

    /// Check if the connection is suitable for federated learning
    ///
    /// # Returns
    ///
    /// `true` if the connection meets minimum requirements
    pub fn is_suitable_for_fl(&self) -> bool {
        self.latency < Duration::from_millis(5000)  // Less than 5 seconds
            && self.bandwidth > 1.0                 // At least 1 Mbps
            && self.packet_loss_rate < 0.1          // Less than 10% loss
            && self.stability_score > 0.3 // Reasonable stability
    }

    /// Estimate communication time for a given data size
    ///
    /// # Arguments
    ///
    /// * `data_size_bytes` - Size of data to transfer in bytes
    ///
    /// # Returns
    ///
    /// Estimated communication time
    pub fn estimate_communication_time(&self, data_size_bytes: usize) -> Duration {
        let transfer_time_seconds = (data_size_bytes as f64 * 8.0) / (self.bandwidth * 1_000_000.0);
        let adjusted_time = transfer_time_seconds * (1.0 + self.packet_loss_rate);

        Duration::from_secs_f64(adjusted_time) + self.latency
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_federated_client_creation() {
        let client = FederatedClient::new("test_client".to_string());
        assert_eq!(client.client_id, "test_client");
        assert_eq!(client.data_size, 0);
        assert_eq!(client.trust_score, 1.0);
    }

    #[test]
    fn test_client_eligibility() {
        let mut client = FederatedClient::new("test_client".to_string());
        assert!(client.is_eligible());

        client.privacy_budget = 0.0;
        assert!(!client.is_eligible());

        client.privacy_budget = 1.0;
        client.trust_score = 0.3;
        assert!(!client.is_eligible());
    }

    #[test]
    fn test_privacy_budget_management() {
        let mut client = FederatedClient::new("test_client".to_string());
        assert!(client.has_privacy_budget(0.5));
        assert!(client.consume_privacy_budget(0.3));
        assert_eq!(client.privacy_budget, 0.7);
        assert!(!client.consume_privacy_budget(1.0));
    }

    #[test]
    fn test_personalized_layers() {
        let mut client = FederatedClient::new("test_client".to_string());
        client.add_personalized_layer("layer1".to_string());
        assert!(client.is_layer_personalized("layer1"));
        assert!(!client.is_layer_personalized("layer2"));
    }

    #[test]
    fn test_trust_score_update() {
        let mut client = FederatedClient::new("test_client".to_string());
        client.update_trust_score(0.8, 0.9);
        assert_eq!(client.trust_score, 0.98); // 0.9 * 1.0 + 0.1 * 0.8
    }

    #[test]
    fn test_resource_capabilities() {
        let resources = ResourceCapabilities::default();
        assert!(resources.is_available());

        let utilization = resources.utilization_score();
        assert!(utilization > 0.0 && utilization <= 1.0);
    }

    #[test]
    fn test_data_distribution_diversity() {
        let mut distribution = DataDistribution::default();
        distribution
            .class_distribution
            .insert("class1".to_string(), 0.5);
        distribution
            .class_distribution
            .insert("class2".to_string(), 0.5);

        let diversity = distribution.diversity_score();
        assert!(diversity > 0.9); // Should be close to 1.0 for uniform distribution
    }

    #[test]
    fn test_connection_quality() {
        let connection = ConnectionQuality::default();
        assert!(connection.is_suitable_for_fl());

        let quality = connection.quality_score();
        assert!(quality > 0.0 && quality <= 1.0);

        let comm_time = connection.estimate_communication_time(1024 * 1024); // 1 MB
        assert!(comm_time > Duration::from_millis(0));
    }

    #[test]
    fn test_client_metrics() {
        let mut metrics = ClientMetrics::default();
        metrics.update_round_metrics(0.5, 0.8, Duration::from_secs(10), Duration::from_secs(2));

        assert_eq!(metrics.local_loss, 0.5);
        assert_eq!(metrics.local_accuracy, 0.8);

        let quality = metrics.overall_quality_score();
        assert!(quality >= 0.0 && quality <= 1.0);
    }
}
