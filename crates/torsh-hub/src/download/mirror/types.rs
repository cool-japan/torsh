//! Core Type Definitions for Mirror Management System
//!
//! This module contains all the type definitions, configuration structures, enums,
//! and result types used throughout the mirror management system. These types
//! provide the foundation for mirror server management, geographic optimization,
//! performance tracking, and download coordination.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

// ================================================================================================
// Core Configuration Types
// ================================================================================================

/// Advanced mirror configuration and management for redundancy and performance
///
/// This configuration provides sophisticated mirror management with multiple selection
/// strategies, automatic discovery, performance benchmarking, and geographic optimization.
///
/// # Features
/// - Multiple mirror selection strategies (latency, reliability, geographic, weighted)
/// - Automatic mirror discovery and health monitoring
/// - Performance benchmarking with configurable intervals
/// - Geographic proximity optimization with coordinate-based calculations
/// - Load balancing and capacity management
/// - Reliability tracking with failure recovery
///
/// # Examples
/// ```rust
/// use torsh_hub::download::mirror::{MirrorConfig, MirrorSelectionStrategy};
/// use std::time::Duration;
///
/// // Create configuration with geographic optimization
/// let config = MirrorConfig {
///     selection_strategy: MirrorSelectionStrategy::Geographic,
///     max_mirror_attempts: 5,
///     connection_timeout: Duration::from_secs(15),
///     enable_auto_discovery: true,
///     benchmark_interval: 1800, // 30 minutes
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MirrorConfig {
    /// List of mirror servers with detailed configuration
    pub mirrors: Vec<MirrorServer>,
    /// Selection strategy for choosing optimal mirrors
    pub selection_strategy: MirrorSelectionStrategy,
    /// Maximum number of mirrors to try per download operation
    pub max_mirror_attempts: usize,
    /// Connection timeout for mirror testing and benchmarking
    pub connection_timeout: Duration,
    /// Enable automatic mirror discovery and registration
    pub enable_auto_discovery: bool,
    /// Benchmark interval for mirror performance testing (seconds)
    pub benchmark_interval: u64,
    /// Enable geographic proximity optimization
    pub enable_geographic_optimization: bool,
    /// Minimum reliability score for active mirrors (0.0 to 1.0)
    pub min_reliability_score: f64,
    /// Maximum acceptable response time in milliseconds
    pub max_response_time: u64,
    /// Load balancing configuration
    pub load_balancing: LoadBalancingConfig,
}

/// Load balancing configuration for mirror management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadBalancingConfig {
    /// Enable intelligent load balancing
    pub enabled: bool,
    /// Target load percentage (0-100) for optimal performance
    pub target_load_percentage: f32,
    /// Rebalancing interval in seconds
    pub rebalancing_interval: u64,
    /// Enable adaptive load balancing based on real-time metrics
    pub adaptive_balancing: bool,
}

impl Default for LoadBalancingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            target_load_percentage: 70.0,
            rebalancing_interval: 300, // 5 minutes
            adaptive_balancing: true,
        }
    }
}

// ================================================================================================
// Mirror Server Types
// ================================================================================================

/// Advanced mirror server configuration with comprehensive metadata
///
/// This structure contains detailed information about each mirror server including
/// performance metrics, geographic location, capacity information, and health status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MirrorServer {
    /// Unique mirror identifier
    pub id: String,
    /// Base URL of the mirror server
    pub base_url: String,
    /// Geographic location information
    pub location: MirrorLocation,
    /// Mirror reliability score (0.0 to 1.0, higher is better)
    pub reliability_score: f64,
    /// Average response time in milliseconds
    pub avg_response_time: Option<u64>,
    /// Last successful connection timestamp (Unix timestamp)
    pub last_successful_connection: Option<u64>,
    /// Number of consecutive failures
    pub consecutive_failures: u32,
    /// Mirror capacity and performance information
    pub capacity: MirrorCapacity,
    /// Whether this mirror is currently active and available
    pub active: bool,
    /// Mirror-specific metadata and configuration
    pub metadata: HashMap<String, String>,
    /// Priority weight for weighted selection (higher = more preferred)
    pub priority_weight: f64,
    /// Network provider and infrastructure information
    pub provider_info: ProviderInfo,
    /// Performance history for trend analysis
    pub performance_history: Vec<PerformanceSnapshot>,
}

/// Geographic location information for mirrors with coordinate precision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MirrorLocation {
    /// Country code (ISO 3166-1 alpha-2)
    pub country: String,
    /// Region/state/province
    pub region: String,
    /// City name
    pub city: String,
    /// Latitude coordinate for geographic calculations
    pub latitude: Option<f64>,
    /// Longitude coordinate for geographic calculations
    pub longitude: Option<f64>,
    /// Network provider/hosting company
    pub provider: String,
    /// Timezone identifier (e.g., "America/New_York")
    pub timezone: Option<String>,
    /// Data center or facility identifier
    pub datacenter: Option<String>,
}

/// Mirror capacity and performance information with real-time metrics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MirrorCapacity {
    /// Maximum bandwidth capacity in Mbps
    pub max_bandwidth: Option<u64>,
    /// Current load percentage (0-100)
    pub current_load: Option<f32>,
    /// Maximum concurrent connections supported
    pub max_connections: Option<u32>,
    /// Current active connections
    pub current_connections: Option<u32>,
    /// Available storage capacity in GB
    pub storage_capacity: Option<u64>,
    /// Used storage in GB
    pub storage_used: Option<u64>,
    /// CPU utilization percentage (0-100)
    pub cpu_utilization: Option<f32>,
    /// Memory utilization percentage (0-100)
    pub memory_utilization: Option<f32>,
}

/// Network provider and infrastructure information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderInfo {
    /// Provider name (e.g., "AWS", "Google Cloud", "Azure")
    pub name: String,
    /// Network tier or performance level
    pub network_tier: Option<String>,
    /// Content delivery network integration
    pub cdn_integration: bool,
    /// Edge location or point of presence
    pub edge_location: Option<String>,
    /// Peering arrangements and network quality
    pub network_quality: NetworkQuality,
}

/// Network quality metrics and classifications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkQuality {
    /// Overall network quality score (0.0 to 1.0)
    pub quality_score: f64,
    /// Internet exchange point connections
    pub ixp_connections: u32,
    /// Transit provider diversity
    pub transit_diversity: u32,
    /// Peering relationships count
    pub peering_count: u32,
}

impl Default for NetworkQuality {
    fn default() -> Self {
        Self {
            quality_score: 0.8,
            ixp_connections: 3,
            transit_diversity: 2,
            peering_count: 50,
        }
    }
}

/// Performance snapshot for historical tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSnapshot {
    /// Timestamp of the snapshot (Unix timestamp)
    pub timestamp: u64,
    /// Response time in milliseconds
    pub response_time: u64,
    /// Throughput in Mbps
    pub throughput: Option<f64>,
    /// Error rate percentage (0-100)
    pub error_rate: f32,
    /// Load percentage (0-100)
    pub load_percentage: f32,
}

// ================================================================================================
// Selection Strategy Types
// ================================================================================================

/// Advanced mirror selection strategies with sophisticated algorithms
///
/// These strategies provide different approaches to mirror selection based on
/// various performance and geographic criteria.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MirrorSelectionStrategy {
    /// Select mirror with lowest average latency
    LowestLatency,
    /// Select mirror with highest reliability score
    HighestReliability,
    /// Select mirrors based on geographic proximity
    Geographic,
    /// Use weighted scoring with customizable factors
    Weighted(MirrorWeights),
    /// Round-robin selection across available mirrors
    RoundRobin,
    /// Adaptive selection based on historical performance
    Adaptive,
    /// Machine learning-based selection with advanced models
    MachineLearning(MLConfig),
}

/// Configurable weights for weighted mirror selection strategy
///
/// All weights should sum to approximately 1.0 for balanced scoring.
/// Individual weights can be adjusted based on specific requirements.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MirrorWeights {
    /// Weight for latency consideration (0.0 to 1.0)
    pub latency: f64,
    /// Weight for reliability score (0.0 to 1.0)
    pub reliability: f64,
    /// Weight for current load considerations (0.0 to 1.0)
    pub load: f64,
    /// Weight for geographic proximity (0.0 to 1.0)
    pub geographic: f64,
    /// Weight for bandwidth capacity (0.0 to 1.0)
    pub bandwidth: f64,
    /// Weight for provider quality score (0.0 to 1.0)
    pub provider_quality: f64,
}

impl Default for MirrorWeights {
    fn default() -> Self {
        Self {
            latency: 0.25,
            reliability: 0.25,
            load: 0.15,
            geographic: 0.15,
            bandwidth: 0.1,
            provider_quality: 0.1,
        }
    }
}

/// Machine learning configuration for advanced mirror selection
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MLConfig {
    /// Type of ML model to use
    pub model_type: MLModelType,
    /// Whether to enable online learning
    pub online_learning: bool,
    /// Learning rate for adaptive algorithms
    pub learning_rate: f64,
    /// Minimum samples required for training
    pub min_samples: usize,
}

impl Default for MLConfig {
    fn default() -> Self {
        Self {
            model_type: MLModelType::DecisionTree,
            online_learning: true,
            learning_rate: 0.01,
            min_samples: 10,
        }
    }
}

/// Types of machine learning models for mirror selection
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MLModelType {
    /// Decision tree-based selection
    DecisionTree,
    /// Neural network-based selection
    NeuralNetwork,
    /// Random forest ensemble method
    RandomForest,
    /// Support vector machine
    SupportVectorMachine,
}

// ================================================================================================
// Internal State Types
// ================================================================================================

/// Internal state for mirror selection algorithms with advanced tracking
#[derive(Debug, Default)]
pub struct MirrorSelectionState {
    pub round_robin_index: usize,
    pub last_benchmark: u64,
    pub adaptive_weights: MirrorWeights,
    pub selection_history: Vec<SelectionRecord>,
    pub ml_model_state: Option<MLModelState>,
}

/// Performance analyzer for sophisticated performance tracking and prediction
#[derive(Debug)]
pub struct PerformanceAnalyzer {
    pub enabled: bool,
    pub trend_analysis_window: Duration,
    pub prediction_accuracy: f64,
    pub performance_cache: HashMap<String, Vec<PerformanceSnapshot>>,
}

impl Default for PerformanceAnalyzer {
    fn default() -> Self {
        Self {
            enabled: true,
            trend_analysis_window: Duration::from_secs(24 * 60 * 60),
            prediction_accuracy: 0.8,
            performance_cache: HashMap::new(),
        }
    }
}

/// Geographic calculator for proximity and latency estimation
#[derive(Debug)]
pub struct GeographicCalculator {
    pub enabled: bool,
    pub user_location: Option<UserLocation>,
    pub distance_cache: HashMap<String, f64>,
}

impl Default for GeographicCalculator {
    fn default() -> Self {
        Self {
            enabled: true,
            user_location: None,
            distance_cache: HashMap::new(),
        }
    }
}

/// User location for geographic optimization
#[derive(Debug, Clone)]
pub struct UserLocation {
    pub latitude: f64,
    pub longitude: f64,
    pub estimated: bool, // Whether location was estimated vs. explicitly provided
}

/// Selection record for tracking mirror selection history
#[derive(Debug, Clone)]
pub struct SelectionRecord {
    pub timestamp: u64,
    pub mirror_id: String,
    pub strategy_used: MirrorSelectionStrategy,
    pub response_time: Option<u64>,
    pub success: bool,
    pub performance_score: Option<f64>,
}

/// Machine learning model state
#[derive(Debug)]
pub struct MLModelState {
    pub model_accuracy: f64,
    pub training_samples: usize,
    pub last_training: u64,
    pub feature_importance: HashMap<String, f64>,
}

// ================================================================================================
// Result and Metrics Types
// ================================================================================================

/// Result of mirror download operation with comprehensive metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MirrorDownloadResult {
    /// Whether the download was successful
    pub success: bool,
    /// Total time taken for the entire operation
    pub total_duration: Duration,
    /// ID of the successful mirror (if any)
    pub successful_mirror: Option<String>,
    /// List of all attempts made
    pub attempts: Vec<MirrorAttempt>,
    /// Total bytes downloaded
    pub total_bytes_downloaded: u64,
    /// Average throughput achieved (MB/s)
    pub average_throughput: Option<f64>,
    /// Mirror selection strategy used
    pub mirror_selection_strategy: MirrorSelectionStrategy,
}

/// Individual mirror download attempt with detailed metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MirrorAttempt {
    /// Mirror ID that was attempted
    pub mirror_id: String,
    /// Mirror URL used
    pub mirror_url: String,
    /// Start time of the attempt
    pub start_time: SystemTime,
    /// Duration of the attempt
    pub duration: Duration,
    /// Number of bytes downloaded (may be partial)
    pub bytes_downloaded: u64,
    /// Whether this attempt was successful
    pub success: bool,
    /// Error message if the attempt failed
    pub error_message: Option<String>,
    /// Response time to first byte (milliseconds)
    pub response_time: Option<u64>,
    /// Average throughput during download (MB/s)
    pub throughput: Option<f64>,
}

/// Mirror benchmark result for performance evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MirrorBenchmarkResult {
    /// Mirror ID that was benchmarked
    pub mirror_id: String,
    /// Response time in milliseconds
    pub response_time: u64,
    /// Available bandwidth (Mbps)
    pub bandwidth: Option<f64>,
    /// Current load percentage
    pub load_percentage: Option<f32>,
    /// Whether the benchmark was successful
    pub success: bool,
    /// Benchmark timestamp
    pub timestamp: u64,
    /// Additional benchmark metrics
    pub additional_metrics: HashMap<String, f64>,
}

/// Comprehensive mirror statistics for monitoring and analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MirrorStatistics {
    /// Mirror ID
    pub mirror_id: String,
    /// Mirror base URL
    pub mirror_url: String,
    /// Total number of successful downloads
    pub successful_downloads: u64,
    /// Total number of failed downloads
    pub failed_downloads: u64,
    /// Success rate percentage (0-100)
    pub success_rate: f64,
    /// Average response time in milliseconds
    pub avg_response_time: f64,
    /// Total bytes served
    pub total_bytes_served: u64,
    /// Current reliability score
    pub reliability_score: f64,
    /// Geographic location
    pub location: MirrorLocation,
    /// Performance trend
    pub performance_trend: PerformanceTrend,
    /// Last update timestamp
    pub last_updated: u64,
}

/// Performance trend analysis results
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PerformanceTrend {
    /// Performance is improving over time
    Improving,
    /// Performance is stable
    Stable,
    /// Performance is degrading over time
    Degrading,
    /// Insufficient data for trend analysis
    InsufficientData,
}

/// Download metrics for performance tracking
#[derive(Debug, Clone)]
pub struct DownloadMetrics {
    pub bytes_downloaded: u64,
    pub throughput: Option<f64>, // MB/s
    pub duration: Duration,
}

// ================================================================================================
// Default Implementation for MirrorConfig
// ================================================================================================

impl Default for MirrorConfig {
    fn default() -> Self {
        Self {
            mirrors: vec![
                MirrorServer {
                    id: "us-east-1".to_string(),
                    base_url: "https://mirrors.torsh.ai/us-east-1".to_string(),
                    location: MirrorLocation {
                        country: "US".to_string(),
                        region: "Virginia".to_string(),
                        city: "Ashburn".to_string(),
                        latitude: Some(39.0438),
                        longitude: Some(-77.4874),
                        provider: "AWS".to_string(),
                        timezone: Some("America/New_York".to_string()),
                        datacenter: Some("us-east-1a".to_string()),
                    },
                    reliability_score: 0.95,
                    avg_response_time: Some(50),
                    last_successful_connection: None,
                    consecutive_failures: 0,
                    capacity: MirrorCapacity {
                        max_bandwidth: Some(10000), // 10 Gbps
                        current_load: Some(30.0),
                        max_connections: Some(1000),
                        current_connections: Some(150),
                        storage_capacity: Some(10000), // 10 TB
                        storage_used: Some(3000),      // 3 TB
                        cpu_utilization: Some(25.0),
                        memory_utilization: Some(40.0),
                    },
                    active: true,
                    metadata: HashMap::new(),
                    priority_weight: 1.0,
                    provider_info: ProviderInfo {
                        name: "AWS".to_string(),
                        network_tier: Some("Premium".to_string()),
                        cdn_integration: true,
                        edge_location: Some("IAD".to_string()),
                        network_quality: NetworkQuality::default(),
                    },
                    performance_history: Vec::new(),
                },
                MirrorServer {
                    id: "eu-west-1".to_string(),
                    base_url: "https://mirrors.torsh.ai/eu-west-1".to_string(),
                    location: MirrorLocation {
                        country: "DE".to_string(),
                        region: "Hesse".to_string(),
                        city: "Frankfurt".to_string(),
                        latitude: Some(50.1109),
                        longitude: Some(8.6821),
                        provider: "AWS".to_string(),
                        timezone: Some("Europe/Berlin".to_string()),
                        datacenter: Some("eu-west-1a".to_string()),
                    },
                    reliability_score: 0.92,
                    avg_response_time: Some(75),
                    last_successful_connection: None,
                    consecutive_failures: 0,
                    capacity: MirrorCapacity {
                        max_bandwidth: Some(10000),
                        current_load: Some(25.0),
                        max_connections: Some(800),
                        current_connections: Some(120),
                        storage_capacity: Some(8000),
                        storage_used: Some(2500),
                        cpu_utilization: Some(20.0),
                        memory_utilization: Some(35.0),
                    },
                    active: true,
                    metadata: HashMap::new(),
                    priority_weight: 0.9,
                    provider_info: ProviderInfo {
                        name: "AWS".to_string(),
                        network_tier: Some("Premium".to_string()),
                        cdn_integration: true,
                        edge_location: Some("FRA".to_string()),
                        network_quality: NetworkQuality::default(),
                    },
                    performance_history: Vec::new(),
                },
            ],
            selection_strategy: MirrorSelectionStrategy::Weighted(MirrorWeights::default()),
            max_mirror_attempts: 3,
            connection_timeout: Duration::from_secs(10),
            enable_auto_discovery: true,
            benchmark_interval: 3600, // 1 hour
            enable_geographic_optimization: true,
            min_reliability_score: 0.7,
            max_response_time: 5000, // 5 seconds
            load_balancing: LoadBalancingConfig::default(),
        }
    }
}

// ================================================================================================
// Helper Functions for Type Creation
// ================================================================================================

/// Create a basic mirror server configuration
pub fn create_mirror_server(
    id: &str,
    base_url: &str,
    country: &str,
    city: &str,
    provider: &str,
) -> MirrorServer {
    MirrorServer {
        id: id.to_string(),
        base_url: base_url.to_string(),
        location: MirrorLocation {
            country: country.to_string(),
            region: "".to_string(),
            city: city.to_string(),
            latitude: None,
            longitude: None,
            provider: provider.to_string(),
            timezone: None,
            datacenter: None,
        },
        reliability_score: 0.8,
        avg_response_time: None,
        last_successful_connection: None,
        consecutive_failures: 0,
        capacity: MirrorCapacity {
            max_bandwidth: None,
            current_load: None,
            max_connections: None,
            current_connections: None,
            storage_capacity: None,
            storage_used: None,
            cpu_utilization: None,
            memory_utilization: None,
        },
        active: true,
        metadata: HashMap::new(),
        priority_weight: 1.0,
        provider_info: ProviderInfo {
            name: provider.to_string(),
            network_tier: None,
            cdn_integration: false,
            edge_location: None,
            network_quality: NetworkQuality::default(),
        },
        performance_history: Vec::new(),
    }
}

/// Create a performance snapshot with current timestamp
pub fn create_performance_snapshot(
    response_time: u64,
    throughput: Option<f64>,
    error_rate: f32,
    load_percentage: f32,
) -> PerformanceSnapshot {
    PerformanceSnapshot {
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        response_time,
        throughput,
        error_rate,
        load_percentage,
    }
}

/// Create a user location from coordinates
pub fn create_user_location(latitude: f64, longitude: f64, estimated: bool) -> UserLocation {
    UserLocation {
        latitude,
        longitude,
        estimated,
    }
}

// ================================================================================================
// Missing Types for Module Exports
// ================================================================================================

/// Mirror health status and monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MirrorHealth {
    pub id: String,
    pub status: MirrorHealthStatus,
    pub last_check: std::time::SystemTime,
    pub response_time: Option<Duration>,
    pub error_rate: f32,
    pub uptime_percentage: f32,
}

/// Health status enumeration for mirrors
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MirrorHealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
    Offline,
    Unknown,
}

/// Performance analysis results
#[derive(Debug, Clone)]
pub struct PerformanceAnalysis {
    pub mirror_id: String,
    pub average_response_time: Duration,
    pub throughput_mbps: f64,
    pub reliability_score: f64,
    pub trend: PerformanceTrend,
    pub bottlenecks: Vec<String>,
    pub recommendations: Vec<String>,
}

/// Performance prediction for future usage
#[derive(Debug, Clone)]
pub struct PerformancePrediction {
    pub mirror_id: String,
    pub predicted_response_time: Duration,
    pub predicted_throughput: f64,
    pub confidence_level: f32,
    pub time_horizon: Duration,
}

/// Provider tier enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ProviderTier {
    Premium,
    Standard,
    Basic,
    Free,
}

/// Selection statistics for mirror performance tracking
#[derive(Debug, Clone)]
pub struct SelectionStatistics {
    pub total_selections: u64,
    pub successful_selections: u64,
    pub average_selection_time: Duration,
    pub selection_distribution: HashMap<String, u64>,
    pub performance_metrics: HashMap<String, f64>,
    pub success_rate: f64,
    pub avg_performance_score: f64,
    pub strategy_usage: HashMap<String, u64>,
}
