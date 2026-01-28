//! Edge Computing for Distributed Training
//!
//! This module provides comprehensive edge computing capabilities for distributed
//! deep learning, including:
//! - Heterogeneous device management and coordination
//! - Adaptive communication for limited bandwidth scenarios
//! - Federated learning protocols and aggregation strategies
//! - Edge-specific optimizations (model compression, quantization)
//! - Dynamic topology management for mobile/intermittent devices
//! - Hierarchical training architectures (edge-fog-cloud)
//! - Privacy-preserving distributed training

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use crate::{TorshDistributedError, TorshResult};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, SystemTime};
use tokio::time::interval;

/// Edge computing configuration for distributed training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeComputingConfig {
    /// Enable heterogeneous device support
    pub heterogeneous_devices: bool,
    /// Enable adaptive communication
    pub adaptive_communication: bool,
    /// Enable federated learning
    pub federated_learning: bool,
    /// Device discovery and management
    pub device_discovery: DeviceDiscoveryConfig,
    /// Bandwidth adaptation configuration
    pub bandwidth_adaptation: BandwidthAdaptationConfig,
    /// Federated learning configuration
    pub federated_config: FederatedLearningConfig,
    /// Edge-specific optimizations
    pub edge_optimizations: EdgeOptimizationConfig,
    /// Hierarchical training configuration
    pub hierarchical_training: HierarchicalTrainingConfig,
    /// Privacy configuration
    pub privacy_config: PrivacyConfig,
}

impl Default for EdgeComputingConfig {
    fn default() -> Self {
        Self {
            heterogeneous_devices: true,
            adaptive_communication: true,
            federated_learning: true,
            device_discovery: DeviceDiscoveryConfig::default(),
            bandwidth_adaptation: BandwidthAdaptationConfig::default(),
            federated_config: FederatedLearningConfig::default(),
            edge_optimizations: EdgeOptimizationConfig::default(),
            hierarchical_training: HierarchicalTrainingConfig::default(),
            privacy_config: PrivacyConfig::default(),
        }
    }
}

/// Device discovery and management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceDiscoveryConfig {
    /// Enable automatic device discovery
    pub auto_discovery: bool,
    /// Discovery protocol
    pub discovery_protocol: DiscoveryProtocol,
    /// Discovery interval in seconds
    pub discovery_interval: u64,
    /// Maximum devices to manage
    pub max_devices: usize,
    /// Device heartbeat interval
    pub heartbeat_interval: u64,
    /// Device timeout threshold
    pub device_timeout: u64,
}

impl Default for DeviceDiscoveryConfig {
    fn default() -> Self {
        Self {
            auto_discovery: true,
            discovery_protocol: DiscoveryProtocol::Mdns,
            discovery_interval: 30,
            max_devices: 100,
            heartbeat_interval: 10,
            device_timeout: 60,
        }
    }
}

/// Device discovery protocols
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiscoveryProtocol {
    /// Multicast DNS (mDNS)
    Mdns,
    /// Universal Plug and Play
    Upnp,
    /// Bluetooth Low Energy
    Ble,
    /// Network broadcast
    Broadcast,
    /// Manual registration
    Manual,
}

/// Bandwidth adaptation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BandwidthAdaptationConfig {
    /// Enable automatic bandwidth detection
    pub auto_detection: bool,
    /// Minimum bandwidth threshold (Mbps)
    pub min_bandwidth: f64,
    /// Bandwidth measurement interval
    pub measurement_interval: u64,
    /// Compression threshold (compress if bandwidth < threshold)
    pub compression_threshold: f64,
    /// Adaptive batch size based on bandwidth
    pub adaptive_batch_size: bool,
    /// Maximum communication timeout
    pub max_timeout: u64,
}

impl Default for BandwidthAdaptationConfig {
    fn default() -> Self {
        Self {
            auto_detection: true,
            min_bandwidth: 1.0, // 1 Mbps minimum
            measurement_interval: 30,
            compression_threshold: 10.0, // 10 Mbps
            adaptive_batch_size: true,
            max_timeout: 300, // 5 minutes
        }
    }
}

/// Federated learning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedLearningConfig {
    /// Federated learning algorithm
    pub algorithm: FederatedAlgorithm,
    /// Number of local training rounds
    pub local_rounds: usize,
    /// Client selection strategy
    pub client_selection: ClientSelectionStrategy,
    /// Minimum clients per round
    pub min_clients_per_round: usize,
    /// Maximum clients per round
    pub max_clients_per_round: usize,
    /// Aggregation strategy
    pub aggregation: AggregationStrategy,
    /// Privacy mechanism
    pub privacy_mechanism: PrivacyMechanism,
    /// Communication rounds
    pub communication_rounds: usize,
}

impl Default for FederatedLearningConfig {
    fn default() -> Self {
        Self {
            algorithm: FederatedAlgorithm::FedAvg,
            local_rounds: 5,
            client_selection: ClientSelectionStrategy::Random,
            min_clients_per_round: 10,
            max_clients_per_round: 100,
            aggregation: AggregationStrategy::FedAvg,
            privacy_mechanism: PrivacyMechanism::DifferentialPrivacy,
            communication_rounds: 100,
        }
    }
}

/// Federated learning algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FederatedAlgorithm {
    /// Federated Averaging
    FedAvg,
    /// Federated Proximal
    FedProx,
    /// Federated NOVA
    FedNova,
    /// Federated Learning with Momentum
    FedMom,
    /// Federated Learning with Adaptive Optimization
    FedAdam,
}

/// Client selection strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClientSelectionStrategy {
    /// Random selection
    Random,
    /// Round-robin selection
    RoundRobin,
    /// Based on data quality/quantity
    DataBased,
    /// Based on computational capability
    ComputeBased,
    /// Based on network quality
    NetworkBased,
    /// Adaptive selection
    Adaptive,
}

/// Aggregation strategies for federated learning
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AggregationStrategy {
    /// Simple averaging
    FedAvg,
    /// Weighted averaging by data size
    WeightedAvg,
    /// Median aggregation
    Median,
    /// Trimmed mean
    TrimmedMean,
    /// Krum aggregation
    Krum,
    /// Byzantine-robust aggregation
    ByzantineRobust,
}

/// Privacy mechanisms for federated learning
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrivacyMechanism {
    /// No privacy protection
    None,
    /// Differential privacy
    DifferentialPrivacy,
    /// Homomorphic encryption
    HomomorphicEncryption,
    /// Secure multiparty computation
    SecureMultipartyComputation,
    /// Federated learning with secure aggregation
    SecureAggregation,
}

/// Edge-specific optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeOptimizationConfig {
    /// Enable model compression
    pub model_compression: bool,
    /// Enable gradient compression
    pub gradient_compression: bool,
    /// Enable quantization
    pub quantization: bool,
    /// Enable pruning
    pub pruning: bool,
    /// Enable knowledge distillation
    pub knowledge_distillation: bool,
    /// Compression ratio target
    pub compression_ratio: f64,
    /// Quantization bits
    pub quantization_bits: u8,
    /// Pruning sparsity target
    pub pruning_sparsity: f64,
}

impl Default for EdgeOptimizationConfig {
    fn default() -> Self {
        Self {
            model_compression: true,
            gradient_compression: true,
            quantization: true,
            pruning: false,
            knowledge_distillation: false,
            compression_ratio: 0.1, // 10x compression
            quantization_bits: 8,
            pruning_sparsity: 0.5, // 50% sparsity
        }
    }
}

/// Hierarchical training configuration (edge-fog-cloud)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalTrainingConfig {
    /// Enable hierarchical training
    pub enable_hierarchical: bool,
    /// Training tiers
    pub tiers: Vec<TrainingTier>,
    /// Aggregation schedule between tiers
    pub aggregation_schedule: AggregationSchedule,
    /// Load balancing between tiers
    pub load_balancing: bool,
}

impl Default for HierarchicalTrainingConfig {
    fn default() -> Self {
        Self {
            enable_hierarchical: true,
            tiers: vec![TrainingTier::Edge, TrainingTier::Fog, TrainingTier::Cloud],
            aggregation_schedule: AggregationSchedule::default(),
            load_balancing: true,
        }
    }
}

/// Training tiers in hierarchical architecture
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrainingTier {
    /// Edge devices (smartphones, IoT devices)
    Edge,
    /// Fog nodes (edge servers, gateways)
    Fog,
    /// Cloud data centers
    Cloud,
}

/// Aggregation schedule for hierarchical training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregationSchedule {
    /// Edge to fog aggregation frequency
    pub edge_to_fog_frequency: u64,
    /// Fog to cloud aggregation frequency
    pub fog_to_cloud_frequency: u64,
    /// Global aggregation frequency
    pub global_aggregation_frequency: u64,
}

impl Default for AggregationSchedule {
    fn default() -> Self {
        Self {
            edge_to_fog_frequency: 5,         // Every 5 rounds
            fog_to_cloud_frequency: 10,       // Every 10 rounds
            global_aggregation_frequency: 20, // Every 20 rounds
        }
    }
}

/// Privacy configuration for edge computing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyConfig {
    /// Enable differential privacy
    pub differential_privacy: bool,
    /// Privacy budget (epsilon)
    pub privacy_budget: f64,
    /// Enable secure aggregation
    pub secure_aggregation: bool,
    /// Enable data anonymization
    pub data_anonymization: bool,
    /// Local training only (no raw data sharing)
    pub local_training_only: bool,
}

impl Default for PrivacyConfig {
    fn default() -> Self {
        Self {
            differential_privacy: true,
            privacy_budget: 1.0,
            secure_aggregation: true,
            data_anonymization: true,
            local_training_only: true,
        }
    }
}

/// Edge device representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeDevice {
    /// Unique device identifier
    pub device_id: String,
    /// Device type
    pub device_type: DeviceType,
    /// Computational capabilities
    pub compute_capability: ComputeCapability,
    /// Network characteristics
    pub network_info: NetworkInfo,
    /// Device status
    pub status: DeviceStatus,
    /// Available resources
    pub resources: DeviceResources,
    /// Data characteristics
    pub data_info: DataInfo,
    /// Last seen timestamp
    pub last_seen: SystemTime,
    /// Device location (if available)
    pub location: Option<DeviceLocation>,
}

/// Types of edge devices
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceType {
    /// Smartphone
    Smartphone,
    /// Tablet
    Tablet,
    /// Laptop
    Laptop,
    /// IoT sensor
    IoTSensor,
    /// Edge server
    EdgeServer,
    /// Fog node
    FogNode,
    /// Embedded device
    Embedded,
    /// Automotive ECU
    Automotive,
}

/// Computational capability of a device
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeCapability {
    /// CPU cores
    pub cpu_cores: u32,
    /// CPU frequency (MHz)
    pub cpu_frequency: u32,
    /// RAM size (MB)
    pub ram_mb: u32,
    /// GPU availability
    pub has_gpu: bool,
    /// GPU memory (MB)
    pub gpu_memory_mb: u32,
    /// NPU/TPU availability
    pub has_accelerator: bool,
    /// Estimated FLOPS
    pub estimated_flops: f64,
    /// Power consumption (watts)
    pub power_consumption: f64,
}

/// Network information for a device
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkInfo {
    /// Connection type
    pub connection_type: ConnectionType,
    /// Bandwidth (Mbps)
    pub bandwidth: f64,
    /// Latency (ms)
    pub latency: f64,
    /// Packet loss rate
    pub packet_loss: f64,
    /// Is connection stable
    pub is_stable: bool,
    /// Data usage limits
    pub data_limits: Option<DataLimits>,
}

/// Types of network connections
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConnectionType {
    /// WiFi connection
    WiFi,
    /// Cellular 4G
    Cellular4G,
    /// Cellular 5G
    Cellular5G,
    /// Ethernet
    Ethernet,
    /// Bluetooth
    Bluetooth,
    /// Satellite
    Satellite,
    /// LoRa/LoRaWAN
    LoRa,
}

/// Data usage limits for mobile connections
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataLimits {
    /// Monthly data limit (MB)
    pub monthly_limit_mb: u64,
    /// Used data this month (MB)
    pub used_data_mb: u64,
    /// Is on unlimited plan
    pub unlimited: bool,
}

/// Current status of an edge device
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceStatus {
    /// Available for training
    Available,
    /// Currently training
    Training,
    /// Temporarily unavailable
    Unavailable,
    /// Disconnected
    Disconnected,
    /// In sleep/power saving mode
    Sleeping,
    /// Under maintenance
    Maintenance,
}

/// Available resources on a device
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceResources {
    /// Available CPU percentage
    pub cpu_available: f64,
    /// Available memory (MB)
    pub memory_available_mb: u32,
    /// Available storage (MB)
    pub storage_available_mb: u64,
    /// Battery level (0-100, None for plugged devices)
    pub battery_level: Option<f64>,
    /// Is device charging
    pub is_charging: Option<bool>,
    /// Thermal state
    pub thermal_state: ThermalState,
}

/// Thermal state of a device
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThermalState {
    /// Normal temperature
    Normal,
    /// Slightly warm
    Warm,
    /// Hot - may throttle
    Hot,
    /// Critical - will throttle
    Critical,
}

/// Data characteristics on a device
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataInfo {
    /// Number of data samples
    pub sample_count: usize,
    /// Data quality score (0-1)
    pub quality_score: f64,
    /// Data diversity score (0-1)
    pub diversity_score: f64,
    /// Label distribution
    pub label_distribution: HashMap<String, f64>,
    /// Data freshness (how recent)
    pub freshness_hours: f64,
    /// Privacy sensitivity level
    pub privacy_level: PrivacyLevel,
}

/// Privacy sensitivity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrivacyLevel {
    /// Public data
    Public,
    /// Internal use
    Internal,
    /// Confidential
    Confidential,
    /// Highly sensitive
    HighlySensitive,
}

/// Device location information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceLocation {
    /// Latitude
    pub latitude: f64,
    /// Longitude  
    pub longitude: f64,
    /// Country code
    pub country: String,
    /// Region/state
    pub region: String,
    /// Time zone
    pub timezone: String,
}

/// Edge computing manager for distributed training
pub struct EdgeComputingManager {
    config: EdgeComputingConfig,
    devices: Arc<RwLock<HashMap<String, EdgeDevice>>>,
    device_groups: Arc<RwLock<HashMap<String, Vec<String>>>>,
    communication_manager: Arc<Mutex<CommunicationManager>>,
    federated_coordinator: Option<FederatedLearningCoordinator>,
    bandwidth_monitor: Arc<Mutex<BandwidthMonitor>>,
    privacy_manager: Arc<Mutex<PrivacyManager>>,
    hierarchical_coordinator: Option<HierarchicalTrainingCoordinator>,
}

impl EdgeComputingManager {
    /// Create a new edge computing manager
    pub fn new(config: EdgeComputingConfig) -> TorshResult<Self> {
        let federated_coordinator = if config.federated_learning {
            Some(FederatedLearningCoordinator::new(&config.federated_config)?)
        } else {
            None
        };

        let hierarchical_coordinator = if config.hierarchical_training.enable_hierarchical {
            Some(HierarchicalTrainingCoordinator::new(
                &config.hierarchical_training,
            )?)
        } else {
            None
        };

        Ok(Self {
            config: config.clone(),
            devices: Arc::new(RwLock::new(HashMap::new())),
            device_groups: Arc::new(RwLock::new(HashMap::new())),
            communication_manager: Arc::new(Mutex::new(CommunicationManager::new(
                &config.bandwidth_adaptation,
            )?)),
            federated_coordinator,
            bandwidth_monitor: Arc::new(Mutex::new(BandwidthMonitor::new(
                &config.bandwidth_adaptation,
            )?)),
            privacy_manager: Arc::new(Mutex::new(PrivacyManager::new(&config.privacy_config)?)),
            hierarchical_coordinator,
        })
    }

    /// Register a new edge device
    pub fn register_device(&self, device: EdgeDevice) -> TorshResult<()> {
        let mut devices = self.devices.write().map_err(|_| {
            TorshDistributedError::InternalError("Failed to acquire devices lock".to_string())
        })?;

        tracing::info!(
            "Registering edge device: {} (type: {:?})",
            device.device_id,
            device.device_type
        );
        devices.insert(device.device_id.clone(), device);

        Ok(())
    }

    /// Discover available devices
    pub async fn discover_devices(&self) -> TorshResult<Vec<EdgeDevice>> {
        if !self.config.device_discovery.auto_discovery {
            return Ok(Vec::new());
        }

        // Simulate device discovery based on protocol
        let discovered = match self.config.device_discovery.discovery_protocol {
            DiscoveryProtocol::Mdns => self.discover_mdns_devices().await?,
            DiscoveryProtocol::Upnp => self.discover_upnp_devices().await?,
            DiscoveryProtocol::Ble => self.discover_ble_devices().await?,
            DiscoveryProtocol::Broadcast => self.discover_broadcast_devices().await?,
            DiscoveryProtocol::Manual => Vec::new(),
        };

        // Register discovered devices
        for device in &discovered {
            self.register_device(device.clone())?;
        }

        tracing::info!("Discovered {} edge devices", discovered.len());
        Ok(discovered)
    }

    /// Simulate mDNS device discovery
    async fn discover_mdns_devices(&self) -> TorshResult<Vec<EdgeDevice>> {
        // In a real implementation, this would use mDNS to discover devices
        let mock_devices = vec![
            self.create_mock_device("edge-phone-1", DeviceType::Smartphone),
            self.create_mock_device("edge-tablet-1", DeviceType::Tablet),
            self.create_mock_device("fog-server-1", DeviceType::FogNode),
        ];

        Ok(mock_devices)
    }

    /// Simulate UPnP device discovery
    async fn discover_upnp_devices(&self) -> TorshResult<Vec<EdgeDevice>> {
        let mock_devices = vec![
            self.create_mock_device("edge-laptop-1", DeviceType::Laptop),
            self.create_mock_device("edge-server-1", DeviceType::EdgeServer),
        ];

        Ok(mock_devices)
    }

    /// Simulate BLE device discovery
    async fn discover_ble_devices(&self) -> TorshResult<Vec<EdgeDevice>> {
        let mock_devices = vec![
            self.create_mock_device("iot-sensor-1", DeviceType::IoTSensor),
            self.create_mock_device("embedded-1", DeviceType::Embedded),
        ];

        Ok(mock_devices)
    }

    /// Simulate broadcast device discovery
    async fn discover_broadcast_devices(&self) -> TorshResult<Vec<EdgeDevice>> {
        let mock_devices = vec![self.create_mock_device("auto-ecu-1", DeviceType::Automotive)];

        Ok(mock_devices)
    }

    /// Create a mock device for testing
    fn create_mock_device(&self, device_id: &str, device_type: DeviceType) -> EdgeDevice {
        EdgeDevice {
            device_id: device_id.to_string(),
            device_type,
            compute_capability: match device_type {
                DeviceType::Smartphone => ComputeCapability {
                    cpu_cores: 8,
                    cpu_frequency: 2400,
                    ram_mb: 6144,
                    has_gpu: true,
                    gpu_memory_mb: 1024,
                    has_accelerator: false,
                    estimated_flops: 1e9,
                    power_consumption: 5.0,
                },
                DeviceType::FogNode => ComputeCapability {
                    cpu_cores: 16,
                    cpu_frequency: 3200,
                    ram_mb: 32768,
                    has_gpu: true,
                    gpu_memory_mb: 8192,
                    has_accelerator: true,
                    estimated_flops: 1e12,
                    power_consumption: 200.0,
                },
                _ => ComputeCapability {
                    cpu_cores: 4,
                    cpu_frequency: 1800,
                    ram_mb: 4096,
                    has_gpu: false,
                    gpu_memory_mb: 0,
                    has_accelerator: false,
                    estimated_flops: 1e8,
                    power_consumption: 10.0,
                },
            },
            network_info: NetworkInfo {
                connection_type: match device_type {
                    DeviceType::Smartphone => ConnectionType::Cellular5G,
                    DeviceType::FogNode => ConnectionType::Ethernet,
                    _ => ConnectionType::WiFi,
                },
                bandwidth: match device_type {
                    DeviceType::Smartphone => 50.0,
                    DeviceType::FogNode => 1000.0,
                    _ => 100.0,
                },
                latency: 20.0,
                packet_loss: 0.01,
                is_stable: true,
                data_limits: if device_type == DeviceType::Smartphone {
                    Some(DataLimits {
                        monthly_limit_mb: 10240,
                        used_data_mb: 2048,
                        unlimited: false,
                    })
                } else {
                    None
                },
            },
            status: DeviceStatus::Available,
            resources: DeviceResources {
                cpu_available: 80.0,
                memory_available_mb: 2048,
                storage_available_mb: 5120,
                battery_level: if device_type == DeviceType::Smartphone {
                    Some(85.0)
                } else {
                    None
                },
                is_charging: if device_type == DeviceType::Smartphone {
                    Some(false)
                } else {
                    None
                },
                thermal_state: ThermalState::Normal,
            },
            data_info: DataInfo {
                sample_count: 1000,
                quality_score: 0.8,
                diversity_score: 0.6,
                label_distribution: HashMap::new(),
                freshness_hours: 2.0,
                privacy_level: PrivacyLevel::Internal,
            },
            last_seen: SystemTime::now(),
            location: Some(DeviceLocation {
                latitude: 37.7749,
                longitude: -122.4194,
                country: "US".to_string(),
                region: "CA".to_string(),
                timezone: "UTC-8".to_string(),
            }),
        }
    }

    /// Select clients for federated learning round
    pub fn select_clients(&self, round: usize) -> TorshResult<Vec<String>> {
        let devices = self.devices.read().map_err(|_| {
            TorshDistributedError::InternalError("Failed to acquire devices lock".to_string())
        })?;

        let available_devices: Vec<&EdgeDevice> = devices
            .values()
            .filter(|d| d.status == DeviceStatus::Available)
            .collect();

        let selection_strategy = self.config.federated_config.client_selection;
        let min_clients = self.config.federated_config.min_clients_per_round;
        let max_clients = self.config.federated_config.max_clients_per_round;

        let selected = match selection_strategy {
            ClientSelectionStrategy::Random => {
                let mut selected = Vec::new();
                let count = (available_devices.len()).min(max_clients).max(min_clients);

                // Simple random selection (in practice, use proper randomization)
                for (i, device) in available_devices.iter().enumerate() {
                    if i < count {
                        selected.push(device.device_id.clone());
                    }
                }
                selected
            }
            ClientSelectionStrategy::ComputeBased => {
                // Select devices with highest computational capability
                let mut sorted_devices = available_devices.clone();
                sorted_devices.sort_by(|a, b| {
                    b.compute_capability
                        .estimated_flops
                        .partial_cmp(&a.compute_capability.estimated_flops)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                sorted_devices
                    .iter()
                    .take(max_clients.min(available_devices.len()))
                    .map(|d| d.device_id.clone())
                    .collect()
            }
            ClientSelectionStrategy::NetworkBased => {
                // Select devices with best network characteristics
                let mut sorted_devices = available_devices.clone();
                sorted_devices.sort_by(|a, b| {
                    let score_a = a.network_info.bandwidth / (a.network_info.latency + 1.0);
                    let score_b = b.network_info.bandwidth / (b.network_info.latency + 1.0);
                    score_b
                        .partial_cmp(&score_a)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                sorted_devices
                    .iter()
                    .take(max_clients.min(available_devices.len()))
                    .map(|d| d.device_id.clone())
                    .collect()
            }
            _ => {
                // Default to first available devices
                available_devices
                    .iter()
                    .take(max_clients.min(available_devices.len()))
                    .map(|d| d.device_id.clone())
                    .collect()
            }
        };

        tracing::info!(
            "Selected {} clients for federated learning round {}",
            selected.len(),
            round
        );
        Ok(selected)
    }

    /// Get device information
    pub fn get_device(&self, device_id: &str) -> TorshResult<Option<EdgeDevice>> {
        let devices = self.devices.read().map_err(|_| {
            TorshDistributedError::InternalError("Failed to acquire devices lock".to_string())
        })?;

        Ok(devices.get(device_id).cloned())
    }

    /// Get all devices
    pub fn get_all_devices(&self) -> TorshResult<Vec<EdgeDevice>> {
        let devices = self.devices.read().map_err(|_| {
            TorshDistributedError::InternalError("Failed to acquire devices lock".to_string())
        })?;

        Ok(devices.values().cloned().collect())
    }

    /// Update device status
    pub fn update_device_status(&self, device_id: &str, status: DeviceStatus) -> TorshResult<()> {
        let mut devices = self.devices.write().map_err(|_| {
            TorshDistributedError::InternalError("Failed to acquire devices lock".to_string())
        })?;

        if let Some(device) = devices.get_mut(device_id) {
            device.status = status;
            device.last_seen = SystemTime::now();
            tracing::debug!("Updated device {} status to {:?}", device_id, status);
        }

        Ok(())
    }

    /// Start device monitoring
    pub async fn start_device_monitoring(&self) -> TorshResult<()> {
        let heartbeat_interval =
            Duration::from_secs(self.config.device_discovery.heartbeat_interval);
        let mut interval_timer = interval(heartbeat_interval);

        loop {
            interval_timer.tick().await;

            // Check device health and update status
            if let Err(e) = self.check_device_health().await {
                tracing::error!("Device health check failed: {}", e);
            }
        }
    }

    /// Check health of all devices
    async fn check_device_health(&self) -> TorshResult<()> {
        let device_timeout = Duration::from_secs(self.config.device_discovery.device_timeout);
        let now = SystemTime::now();

        let mut devices = self.devices.write().map_err(|_| {
            TorshDistributedError::InternalError("Failed to acquire devices lock".to_string())
        })?;

        for device in devices.values_mut() {
            if let Ok(elapsed) = now.duration_since(device.last_seen) {
                if elapsed > device_timeout && device.status != DeviceStatus::Disconnected {
                    device.status = DeviceStatus::Disconnected;
                    tracing::warn!(
                        "Device {} marked as disconnected due to timeout",
                        device.device_id
                    );
                }
            }
        }

        Ok(())
    }
}

/// Communication manager for adaptive bandwidth
pub struct CommunicationManager {
    config: BandwidthAdaptationConfig,
    bandwidth_history: VecDeque<(SystemTime, f64)>,
}

impl CommunicationManager {
    pub fn new(config: &BandwidthAdaptationConfig) -> TorshResult<Self> {
        Ok(Self {
            config: config.clone(),
            bandwidth_history: VecDeque::with_capacity(100),
        })
    }

    /// Measure current bandwidth
    pub async fn measure_bandwidth(&mut self, device_id: &str) -> TorshResult<f64> {
        // Simulate bandwidth measurement
        let bandwidth = 50.0 + (device_id.len() as f64 * 10.0) % 100.0; // Mock measurement

        self.bandwidth_history
            .push_back((SystemTime::now(), bandwidth));
        if self.bandwidth_history.len() > 100 {
            self.bandwidth_history.pop_front();
        }

        Ok(bandwidth)
    }

    /// Get adaptive communication parameters
    pub fn get_adaptive_params(&self, current_bandwidth: f64) -> AdaptiveCommunicationParams {
        let should_compress = current_bandwidth < self.config.compression_threshold;
        let timeout_multiplier = if current_bandwidth < self.config.min_bandwidth {
            3.0
        } else if current_bandwidth < self.config.compression_threshold {
            2.0
        } else {
            1.0
        };

        AdaptiveCommunicationParams {
            use_compression: should_compress,
            compression_ratio: if should_compress { 0.1 } else { 1.0 },
            timeout_multiplier,
            max_batch_size: if self.config.adaptive_batch_size {
                ((current_bandwidth / 10.0) as usize).clamp(1, 64)
            } else {
                32
            },
        }
    }
}

/// Adaptive communication parameters
#[derive(Debug, Clone)]
pub struct AdaptiveCommunicationParams {
    pub use_compression: bool,
    pub compression_ratio: f64,
    pub timeout_multiplier: f64,
    pub max_batch_size: usize,
}

/// Bandwidth monitor for edge devices
pub struct BandwidthMonitor {
    config: BandwidthAdaptationConfig,
    measurements: HashMap<String, VecDeque<(SystemTime, f64)>>,
}

impl BandwidthMonitor {
    pub fn new(config: &BandwidthAdaptationConfig) -> TorshResult<Self> {
        Ok(Self {
            config: config.clone(),
            measurements: HashMap::new(),
        })
    }

    /// Record bandwidth measurement for a device
    pub fn record_measurement(&mut self, device_id: String, bandwidth: f64) {
        let measurements = self
            .measurements
            .entry(device_id)
            .or_insert_with(|| VecDeque::with_capacity(100));
        measurements.push_back((SystemTime::now(), bandwidth));

        if measurements.len() > 100 {
            measurements.pop_front();
        }
    }

    /// Get average bandwidth for a device
    pub fn get_average_bandwidth(&self, device_id: &str, window_minutes: u64) -> Option<f64> {
        let measurements = self.measurements.get(device_id)?;
        let window = Duration::from_secs(window_minutes * 60);
        let now = SystemTime::now();

        let recent_measurements: Vec<f64> = measurements
            .iter()
            .filter_map(|(time, bandwidth)| {
                if now.duration_since(*time).unwrap_or(Duration::MAX) <= window {
                    Some(*bandwidth)
                } else {
                    None
                }
            })
            .collect();

        if recent_measurements.is_empty() {
            None
        } else {
            Some(recent_measurements.iter().sum::<f64>() / recent_measurements.len() as f64)
        }
    }
}

/// Privacy manager for edge computing
pub struct PrivacyManager {
    config: PrivacyConfig,
}

impl PrivacyManager {
    pub fn new(config: &PrivacyConfig) -> TorshResult<Self> {
        Ok(Self {
            config: config.clone(),
        })
    }

    /// Apply differential privacy to gradients
    pub fn apply_differential_privacy(
        &self,
        gradients: &[f32],
        sensitivity: f64,
    ) -> TorshResult<Vec<f32>> {
        if !self.config.differential_privacy {
            return Ok(gradients.to_vec());
        }

        // Simple Gaussian mechanism for differential privacy
        let noise_scale = sensitivity / self.config.privacy_budget;
        let mut private_gradients = Vec::with_capacity(gradients.len());

        for &gradient in gradients {
            // Add Gaussian noise (simplified - use proper crypto RNG in production)
            let noise = (gradient.abs() * 0.01) * (2.0 * std::f32::consts::PI).sin(); // Mock noise
            private_gradients.push(gradient + noise * noise_scale as f32);
        }

        Ok(private_gradients)
    }
}

/// Federated learning coordinator
pub struct FederatedLearningCoordinator {
    config: FederatedLearningConfig,
    current_round: Arc<std::sync::atomic::AtomicUsize>,
    aggregation_buffer: Arc<Mutex<HashMap<String, Vec<f32>>>>,
}

impl FederatedLearningCoordinator {
    pub fn new(config: &FederatedLearningConfig) -> TorshResult<Self> {
        Ok(Self {
            config: config.clone(),
            current_round: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            aggregation_buffer: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// Aggregate model updates from clients
    pub fn aggregate_updates(
        &self,
        client_updates: HashMap<String, Vec<f32>>,
    ) -> TorshResult<Vec<f32>> {
        if client_updates.is_empty() {
            return Err(TorshDistributedError::InternalError(
                "No client updates to aggregate".to_string(),
            ));
        }

        match self.config.aggregation {
            AggregationStrategy::FedAvg => self.federated_averaging(client_updates),
            AggregationStrategy::WeightedAvg => self.weighted_averaging(client_updates),
            AggregationStrategy::Median => self.median_aggregation(client_updates),
            _ => self.federated_averaging(client_updates), // Default to FedAvg
        }
    }

    /// Federated averaging aggregation
    fn federated_averaging(
        &self,
        client_updates: HashMap<String, Vec<f32>>,
    ) -> TorshResult<Vec<f32>> {
        if client_updates.is_empty() {
            return Err(TorshDistributedError::InternalError(
                "No updates to aggregate".to_string(),
            ));
        }

        let num_clients = client_updates.len() as f32;
        let update_size = client_updates
            .values()
            .next()
            .expect("client_updates should not be empty")
            .len();
        let mut aggregated = vec![0.0; update_size];

        for updates in client_updates.values() {
            for (i, &update) in updates.iter().enumerate() {
                aggregated[i] += update / num_clients;
            }
        }

        Ok(aggregated)
    }

    /// Weighted averaging aggregation
    fn weighted_averaging(
        &self,
        client_updates: HashMap<String, Vec<f32>>,
    ) -> TorshResult<Vec<f32>> {
        // In practice, weights would be based on data size or quality
        // For now, use equal weights (same as FedAvg)
        self.federated_averaging(client_updates)
    }

    /// Median aggregation (Byzantine-robust)
    fn median_aggregation(
        &self,
        client_updates: HashMap<String, Vec<f32>>,
    ) -> TorshResult<Vec<f32>> {
        if client_updates.is_empty() {
            return Err(TorshDistributedError::InternalError(
                "No updates to aggregate".to_string(),
            ));
        }

        let update_size = client_updates
            .values()
            .next()
            .expect("client_updates should not be empty")
            .len();
        let mut aggregated = vec![0.0; update_size];

        for i in 0..update_size {
            let mut values: Vec<f32> = client_updates.values().map(|updates| updates[i]).collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            aggregated[i] = if values.len() % 2 == 0 {
                (values[values.len() / 2 - 1] + values[values.len() / 2]) / 2.0
            } else {
                values[values.len() / 2]
            };
        }

        Ok(aggregated)
    }
}

/// Hierarchical training coordinator
pub struct HierarchicalTrainingCoordinator {
    config: HierarchicalTrainingConfig,
    tier_assignments: HashMap<String, TrainingTier>,
}

impl HierarchicalTrainingCoordinator {
    pub fn new(config: &HierarchicalTrainingConfig) -> TorshResult<Self> {
        Ok(Self {
            config: config.clone(),
            tier_assignments: HashMap::new(),
        })
    }

    /// Assign device to training tier
    pub fn assign_device_tier(&mut self, device_id: String, device: &EdgeDevice) -> TrainingTier {
        let tier = match device.device_type {
            DeviceType::Smartphone
            | DeviceType::Tablet
            | DeviceType::IoTSensor
            | DeviceType::Embedded => TrainingTier::Edge,
            DeviceType::Laptop | DeviceType::EdgeServer | DeviceType::Automotive => {
                TrainingTier::Fog
            }
            DeviceType::FogNode => TrainingTier::Cloud,
        };

        self.tier_assignments.insert(device_id, tier);
        tier
    }

    /// Get devices in a specific tier
    pub fn get_tier_devices(&self, tier: TrainingTier) -> Vec<String> {
        self.tier_assignments
            .iter()
            .filter_map(|(device_id, &device_tier)| {
                if device_tier == tier {
                    Some(device_id.clone())
                } else {
                    None
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edge_computing_config_default() {
        let config = EdgeComputingConfig::default();
        assert!(config.heterogeneous_devices);
        assert!(config.adaptive_communication);
        assert!(config.federated_learning);
    }

    #[test]
    fn test_device_creation() {
        let device = EdgeDevice {
            device_id: "test-device".to_string(),
            device_type: DeviceType::Smartphone,
            compute_capability: ComputeCapability {
                cpu_cores: 8,
                cpu_frequency: 2400,
                ram_mb: 6144,
                has_gpu: true,
                gpu_memory_mb: 1024,
                has_accelerator: false,
                estimated_flops: 1e9,
                power_consumption: 5.0,
            },
            network_info: NetworkInfo {
                connection_type: ConnectionType::Cellular5G,
                bandwidth: 50.0,
                latency: 20.0,
                packet_loss: 0.01,
                is_stable: true,
                data_limits: None,
            },
            status: DeviceStatus::Available,
            resources: DeviceResources {
                cpu_available: 80.0,
                memory_available_mb: 2048,
                storage_available_mb: 5120,
                battery_level: Some(85.0),
                is_charging: Some(false),
                thermal_state: ThermalState::Normal,
            },
            data_info: DataInfo {
                sample_count: 1000,
                quality_score: 0.8,
                diversity_score: 0.6,
                label_distribution: HashMap::new(),
                freshness_hours: 2.0,
                privacy_level: PrivacyLevel::Internal,
            },
            last_seen: SystemTime::now(),
            location: None,
        };

        assert_eq!(device.device_type, DeviceType::Smartphone);
        assert_eq!(device.status, DeviceStatus::Available);
    }

    #[tokio::test]
    async fn test_edge_computing_manager_creation() {
        let config = EdgeComputingConfig::default();
        let manager = EdgeComputingManager::new(config).unwrap();

        // Test device registration
        let device = manager.create_mock_device("test-device", DeviceType::Smartphone);
        manager.register_device(device).unwrap();

        // Test device retrieval
        let retrieved = manager.get_device("test-device").unwrap();
        assert!(retrieved.is_some());
    }

    #[tokio::test]
    async fn test_device_discovery() {
        let config = EdgeComputingConfig::default();
        let manager = EdgeComputingManager::new(config).unwrap();

        let discovered = manager.discover_devices().await.unwrap();
        assert!(!discovered.is_empty());
    }

    #[test]
    fn test_client_selection() {
        let config = EdgeComputingConfig::default();
        let manager = EdgeComputingManager::new(config).unwrap();

        // Register some devices
        for i in 0..5 {
            let device =
                manager.create_mock_device(&format!("device-{}", i), DeviceType::Smartphone);
            manager.register_device(device).unwrap();
        }

        let selected = manager.select_clients(1).unwrap();
        assert!(!selected.is_empty());
        assert!(selected.len() <= 5);
    }

    #[test]
    fn test_federated_aggregation() {
        let config = FederatedLearningConfig::default();
        let coordinator = FederatedLearningCoordinator::new(&config).unwrap();

        let mut client_updates = HashMap::new();
        client_updates.insert("client1".to_string(), vec![1.0, 2.0, 3.0]);
        client_updates.insert("client2".to_string(), vec![2.0, 3.0, 4.0]);
        client_updates.insert("client3".to_string(), vec![3.0, 4.0, 5.0]);

        let aggregated = coordinator.aggregate_updates(client_updates).unwrap();
        let expected = [2.0, 3.0, 4.0]; // Average

        // Use approximate equality for floating-point comparison
        assert_eq!(aggregated.len(), expected.len());
        for (i, (&actual, &exp)) in aggregated.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - exp).abs() < 1e-6,
                "Element {} mismatch: expected {}, got {}",
                i,
                exp,
                actual
            );
        }
    }

    #[test]
    fn test_bandwidth_adaptation() {
        let config = BandwidthAdaptationConfig::default();
        let comm_manager = CommunicationManager::new(&config).unwrap();

        // Test high bandwidth
        let high_bw_params = comm_manager.get_adaptive_params(100.0);
        assert!(!high_bw_params.use_compression);

        // Test low bandwidth
        let low_bw_params = comm_manager.get_adaptive_params(5.0);
        assert!(low_bw_params.use_compression);
        assert!(low_bw_params.timeout_multiplier > 1.0);
    }

    #[test]
    fn test_privacy_mechanism() {
        let config = PrivacyConfig::default();
        let privacy_manager = PrivacyManager::new(&config).unwrap();

        let gradients = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let private_gradients = privacy_manager
            .apply_differential_privacy(&gradients, 1.0)
            .unwrap();

        assert_eq!(private_gradients.len(), gradients.len());
        // Gradients should be different due to noise (in most cases)
    }

    #[test]
    fn test_hierarchical_training() {
        let config = HierarchicalTrainingConfig::default();
        let mut coordinator = HierarchicalTrainingCoordinator::new(&config).unwrap();

        let phone_device = EdgeDevice {
            device_id: "phone".to_string(),
            device_type: DeviceType::Smartphone,
            compute_capability: ComputeCapability {
                cpu_cores: 8,
                cpu_frequency: 2400,
                ram_mb: 6144,
                has_gpu: true,
                gpu_memory_mb: 1024,
                has_accelerator: false,
                estimated_flops: 1e9,
                power_consumption: 5.0,
            },
            network_info: NetworkInfo {
                connection_type: ConnectionType::Cellular5G,
                bandwidth: 50.0,
                latency: 20.0,
                packet_loss: 0.01,
                is_stable: true,
                data_limits: None,
            },
            status: DeviceStatus::Available,
            resources: DeviceResources {
                cpu_available: 80.0,
                memory_available_mb: 2048,
                storage_available_mb: 5120,
                battery_level: Some(85.0),
                is_charging: Some(false),
                thermal_state: ThermalState::Normal,
            },
            data_info: DataInfo {
                sample_count: 1000,
                quality_score: 0.8,
                diversity_score: 0.6,
                label_distribution: HashMap::new(),
                freshness_hours: 2.0,
                privacy_level: PrivacyLevel::Internal,
            },
            last_seen: SystemTime::now(),
            location: None,
        };

        let tier = coordinator.assign_device_tier("phone".to_string(), &phone_device);
        assert_eq!(tier, TrainingTier::Edge);

        let edge_devices = coordinator.get_tier_devices(TrainingTier::Edge);
        assert!(edge_devices.contains(&"phone".to_string()));
    }
}
