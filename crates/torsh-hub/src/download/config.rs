//! Configuration structures and settings for the ToRSh Hub download system.
//!
//! This module contains all configuration-related structures and functions for managing
//! downloads, CDN settings, mirror configurations, and regional setups. It provides
//! a centralized location for all download system configuration with support for
//! serialization, validation, and builder patterns.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use torsh_core::error::{Result, TorshError};

/// Configuration for parallel downloads
///
/// This structure controls various aspects of parallel download behavior including
/// concurrency limits, chunk sizes, timeouts, and retry policies.
///
/// # Examples
///
/// ```rust
/// use torsh_hub::download::config::ParallelDownloadConfig;
///
/// // Use default configuration
/// let config = ParallelDownloadConfig::default();
///
/// // Create custom configuration
/// let config = ParallelDownloadConfig {
///     max_concurrent_downloads: 8,
///     chunk_size: 2 * 1024 * 1024, // 2MB chunks
///     max_concurrent_chunks: 16,
///     timeout_seconds: 600,
///     max_retries: 5,
///     enable_resume: true,
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelDownloadConfig {
    /// Maximum number of concurrent downloads
    pub max_concurrent_downloads: usize,
    /// Size of each download chunk in bytes
    pub chunk_size: usize,
    /// Maximum number of concurrent chunks per download
    pub max_concurrent_chunks: usize,
    /// Timeout for individual operations in seconds
    pub timeout_seconds: u64,
    /// Maximum number of retry attempts
    pub max_retries: usize,
    /// Enable resume capability for interrupted downloads
    pub enable_resume: bool,
}

impl Default for ParallelDownloadConfig {
    fn default() -> Self {
        Self {
            max_concurrent_downloads: 4,
            chunk_size: 1024 * 1024, // 1MB chunks
            max_concurrent_chunks: 8,
            timeout_seconds: 300,
            max_retries: 3,
            enable_resume: true,
        }
    }
}

impl ParallelDownloadConfig {
    /// Create a new configuration builder
    pub fn builder() -> ParallelDownloadConfigBuilder {
        ParallelDownloadConfigBuilder::new()
    }

    /// Validate the configuration settings
    pub fn validate(&self) -> Result<()> {
        if self.max_concurrent_downloads == 0 {
            return Err(TorshError::config_error_with_context(
                "max_concurrent_downloads must be greater than 0",
                "ParallelDownloadConfig validation",
            ));
        }

        if self.chunk_size == 0 {
            return Err(TorshError::config_error_with_context(
                "chunk_size must be greater than 0",
                "ParallelDownloadConfig validation",
            ));
        }

        if self.max_concurrent_chunks == 0 {
            return Err(TorshError::config_error_with_context(
                "max_concurrent_chunks must be greater than 0",
                "ParallelDownloadConfig validation",
            ));
        }

        if self.timeout_seconds == 0 {
            return Err(TorshError::config_error_with_context(
                "timeout_seconds must be greater than 0",
                "ParallelDownloadConfig validation",
            ));
        }

        Ok(())
    }
}

/// Builder for ParallelDownloadConfig
#[derive(Debug)]
pub struct ParallelDownloadConfigBuilder {
    config: ParallelDownloadConfig,
}

impl ParallelDownloadConfigBuilder {
    /// Create a new builder with default values
    pub fn new() -> Self {
        Self {
            config: ParallelDownloadConfig::default(),
        }
    }

    /// Set maximum concurrent downloads
    pub fn max_concurrent_downloads(mut self, max: usize) -> Self {
        self.config.max_concurrent_downloads = max;
        self
    }

    /// Set chunk size in bytes
    pub fn chunk_size(mut self, size: usize) -> Self {
        self.config.chunk_size = size;
        self
    }

    /// Set maximum concurrent chunks
    pub fn max_concurrent_chunks(mut self, max: usize) -> Self {
        self.config.max_concurrent_chunks = max;
        self
    }

    /// Set timeout in seconds
    pub fn timeout_seconds(mut self, timeout: u64) -> Self {
        self.config.timeout_seconds = timeout;
        self
    }

    /// Set maximum retries
    pub fn max_retries(mut self, retries: usize) -> Self {
        self.config.max_retries = retries;
        self
    }

    /// Enable or disable resume capability
    pub fn enable_resume(mut self, enable: bool) -> Self {
        self.config.enable_resume = enable;
        self
    }

    /// Build the configuration
    pub fn build(self) -> Result<ParallelDownloadConfig> {
        self.config.validate()?;
        Ok(self.config)
    }
}

/// CDN (Content Delivery Network) configuration and management
///
/// This structure manages multiple CDN endpoints with automatic failover,
/// health checking, and performance monitoring capabilities.
///
/// # Examples
///
/// ```rust
/// use torsh_hub::download::config::{CdnConfig, FailoverStrategy};
/// use std::time::Duration;
///
/// let config = CdnConfig {
///     endpoints: vec![],
///     max_retries: 5,
///     endpoint_timeout: Duration::from_secs(60),
///     enable_health_check: true,
///     health_check_interval: 600,
///     failover_strategy: FailoverStrategy::Fastest,
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CdnConfig {
    /// List of CDN endpoints in priority order
    pub endpoints: Vec<CdnEndpoint>,
    /// Maximum number of CDNs to try before failing
    pub max_retries: usize,
    /// Timeout for each CDN endpoint
    #[serde(with = "duration_serde")]
    pub endpoint_timeout: Duration,
    /// Enable health checking of CDN endpoints
    pub enable_health_check: bool,
    /// Health check interval in seconds
    pub health_check_interval: u64,
    /// Failover strategy
    pub failover_strategy: FailoverStrategy,
}

/// CDN endpoint configuration
///
/// Represents a single CDN endpoint with its configuration, health status,
/// and performance metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CdnEndpoint {
    /// Endpoint name/identifier
    pub name: String,
    /// Base URL of the CDN
    pub base_url: String,
    /// Geographic region (for optimization)
    pub region: String,
    /// Priority (lower numbers = higher priority)
    pub priority: u32,
    /// Whether this endpoint is currently healthy
    pub healthy: bool,
    /// Last health check timestamp
    pub last_health_check: Option<u64>,
    /// Average response time in milliseconds
    pub avg_response_time: Option<u64>,
    /// Additional headers to send with requests
    pub headers: HashMap<String, String>,
}

/// Failover strategies for CDN usage
///
/// Defines how the system should select CDN endpoints when multiple
/// options are available.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum FailoverStrategy {
    /// Try endpoints in priority order
    Priority,
    /// Try the fastest responding endpoint first
    Fastest,
    /// Round-robin between healthy endpoints
    RoundRobin,
    /// Random selection from healthy endpoints
    Random,
}

impl Default for CdnConfig {
    fn default() -> Self {
        Self {
            endpoints: vec![
                CdnEndpoint {
                    name: "primary".to_string(),
                    base_url: "https://cdn.torsh.rs".to_string(),
                    region: "global".to_string(),
                    priority: 1,
                    healthy: true,
                    last_health_check: None,
                    avg_response_time: None,
                    headers: HashMap::new(),
                },
                CdnEndpoint {
                    name: "fallback".to_string(),
                    base_url: "https://backup.torsh.rs".to_string(),
                    region: "us-east-1".to_string(),
                    priority: 2,
                    healthy: true,
                    last_health_check: None,
                    avg_response_time: None,
                    headers: HashMap::new(),
                },
            ],
            max_retries: 3,
            endpoint_timeout: Duration::from_secs(30),
            enable_health_check: true,
            health_check_interval: 300, // 5 minutes
            failover_strategy: FailoverStrategy::Priority,
        }
    }
}

/// Mirror configuration and management for redundancy
///
/// Manages multiple mirror servers with sophisticated selection strategies,
/// performance monitoring, and automatic discovery capabilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MirrorConfig {
    /// List of mirror servers
    pub mirrors: Vec<MirrorServer>,
    /// Selection strategy for choosing mirrors
    pub selection_strategy: MirrorSelectionStrategy,
    /// Maximum number of mirrors to try per download
    pub max_mirror_attempts: usize,
    /// Connection timeout for mirror testing
    #[serde(with = "duration_serde")]
    pub connection_timeout: Duration,
    /// Enable automatic mirror discovery
    pub enable_auto_discovery: bool,
    /// Benchmark interval for mirror performance testing
    pub benchmark_interval: u64,
}

/// Mirror server configuration
///
/// Comprehensive configuration for a single mirror server including
/// location, capacity, performance metrics, and metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MirrorServer {
    /// Mirror identifier
    pub id: String,
    /// Base URL of the mirror
    pub base_url: String,
    /// Geographic location
    pub location: MirrorLocation,
    /// Mirror reliability score (0.0 to 1.0)
    pub reliability_score: f64,
    /// Average response time in milliseconds
    pub avg_response_time: Option<u64>,
    /// Last successful connection timestamp
    pub last_successful_connection: Option<u64>,
    /// Number of consecutive failures
    pub consecutive_failures: u32,
    /// Mirror capacity information
    pub capacity: MirrorCapacity,
    /// Whether this mirror is currently active
    pub active: bool,
    /// Mirror-specific metadata
    pub metadata: HashMap<String, String>,
}

/// Geographic location information for mirrors
///
/// Stores detailed geographic and provider information for mirror selection
/// and optimization based on proximity and network characteristics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MirrorLocation {
    /// Country code (ISO 3166-1 alpha-2)
    pub country: String,
    /// Region/state
    pub region: String,
    /// City
    pub city: String,
    /// Latitude
    pub latitude: Option<f64>,
    /// Longitude
    pub longitude: Option<f64>,
    /// Network provider
    pub provider: String,
}

/// Mirror capacity and performance information
///
/// Tracks current and maximum capacity metrics for load balancing
/// and performance optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MirrorCapacity {
    /// Maximum bandwidth in Mbps
    pub max_bandwidth: Option<u64>,
    /// Current load percentage (0-100)
    pub current_load: Option<f32>,
    /// Maximum concurrent connections
    pub max_connections: Option<u32>,
    /// Current active connections
    pub current_connections: Option<u32>,
}

/// Mirror selection strategies
///
/// Defines various algorithms for selecting the best mirror based on
/// different criteria and optimization goals.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MirrorSelectionStrategy {
    /// Select by lowest latency
    LowestLatency,
    /// Select by highest reliability
    HighestReliability,
    /// Select by geographic proximity
    Geographic,
    /// Select by lowest load
    LowestLoad,
    /// Weighted combination of factors
    Weighted(MirrorWeights),
    /// Random selection from available mirrors
    Random,
    /// Round-robin selection
    RoundRobin,
}

/// Weights for different mirror selection factors
///
/// Used with the Weighted selection strategy to combine multiple
/// factors with configurable importance.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MirrorWeights {
    /// Weight for latency factor (0.0 to 1.0)
    pub latency: f64,
    /// Weight for reliability factor (0.0 to 1.0)
    pub reliability: f64,
    /// Weight for load factor (0.0 to 1.0)
    pub load: f64,
    /// Weight for geographic factor (0.0 to 1.0)
    pub geographic: f64,
}

impl Default for MirrorWeights {
    fn default() -> Self {
        Self {
            latency: 0.4,
            reliability: 0.3,
            load: 0.2,
            geographic: 0.1,
        }
    }
}

impl Default for MirrorConfig {
    fn default() -> Self {
        Self {
            mirrors: vec![
                MirrorServer {
                    id: "primary-us".to_string(),
                    base_url: "https://mirror1.torsh.rs".to_string(),
                    location: MirrorLocation {
                        country: "US".to_string(),
                        region: "Virginia".to_string(),
                        city: "Ashburn".to_string(),
                        latitude: Some(39.0458),
                        longitude: Some(-77.5073),
                        provider: "AWS".to_string(),
                    },
                    reliability_score: 0.98,
                    avg_response_time: Some(50),
                    last_successful_connection: None,
                    consecutive_failures: 0,
                    capacity: MirrorCapacity {
                        max_bandwidth: Some(10000), // 10 Gbps
                        current_load: Some(25.0),
                        max_connections: Some(1000),
                        current_connections: Some(250),
                    },
                    active: true,
                    metadata: HashMap::new(),
                },
                MirrorServer {
                    id: "primary-eu".to_string(),
                    base_url: "https://mirror2.torsh.rs".to_string(),
                    location: MirrorLocation {
                        country: "DE".to_string(),
                        region: "Frankfurt".to_string(),
                        city: "Frankfurt".to_string(),
                        latitude: Some(50.1109),
                        longitude: Some(8.6821),
                        provider: "Google Cloud".to_string(),
                    },
                    reliability_score: 0.96,
                    avg_response_time: Some(30),
                    last_successful_connection: None,
                    consecutive_failures: 0,
                    capacity: MirrorCapacity {
                        max_bandwidth: Some(5000), // 5 Gbps
                        current_load: Some(15.0),
                        max_connections: Some(800),
                        current_connections: Some(120),
                    },
                    active: true,
                    metadata: HashMap::new(),
                },
            ],
            selection_strategy: MirrorSelectionStrategy::Weighted(MirrorWeights::default()),
            max_mirror_attempts: 3,
            connection_timeout: Duration::from_secs(10),
            enable_auto_discovery: true,
            benchmark_interval: 3600, // 1 hour
        }
    }
}

/// Create a CDN configuration for a specific region
///
/// This function generates optimized CDN configurations for different
/// geographic regions to improve download performance for users in those areas.
///
/// # Arguments
/// * `region` - The target region ("us-east", "eu-west", "asia-pacific", etc.)
///
/// # Returns
/// A configured CdnConfig optimized for the specified region
///
/// # Examples
/// ```rust
/// use torsh_hub::download::config::create_regional_cdn_config;
///
/// let us_config = create_regional_cdn_config("us-east");
/// let eu_config = create_regional_cdn_config("eu-west");
/// ```
pub fn create_regional_cdn_config(region: &str) -> CdnConfig {
    let mut config = CdnConfig::default();

    match region {
        "us-east" => {
            config.endpoints = vec![
                CdnEndpoint {
                    name: "us-east-primary".to_string(),
                    base_url: "https://us-east.cdn.torsh.rs".to_string(),
                    region: "us-east-1".to_string(),
                    priority: 1,
                    healthy: true,
                    last_health_check: None,
                    avg_response_time: None,
                    headers: HashMap::new(),
                },
                CdnEndpoint {
                    name: "us-east-backup".to_string(),
                    base_url: "https://us-east-backup.cdn.torsh.rs".to_string(),
                    region: "us-east-2".to_string(),
                    priority: 2,
                    healthy: true,
                    last_health_check: None,
                    avg_response_time: None,
                    headers: HashMap::new(),
                },
            ];
        }
        "eu-west" => {
            config.endpoints = vec![
                CdnEndpoint {
                    name: "eu-west-primary".to_string(),
                    base_url: "https://eu-west.cdn.torsh.rs".to_string(),
                    region: "eu-west-1".to_string(),
                    priority: 1,
                    healthy: true,
                    last_health_check: None,
                    avg_response_time: None,
                    headers: HashMap::new(),
                },
                CdnEndpoint {
                    name: "eu-west-backup".to_string(),
                    base_url: "https://eu-west-backup.cdn.torsh.rs".to_string(),
                    region: "eu-west-2".to_string(),
                    priority: 2,
                    healthy: true,
                    last_health_check: None,
                    avg_response_time: None,
                    headers: HashMap::new(),
                },
            ];
        }
        "asia-pacific" => {
            config.endpoints = vec![
                CdnEndpoint {
                    name: "ap-primary".to_string(),
                    base_url: "https://ap.cdn.torsh.rs".to_string(),
                    region: "ap-southeast-1".to_string(),
                    priority: 1,
                    healthy: true,
                    last_health_check: None,
                    avg_response_time: None,
                    headers: HashMap::new(),
                },
                CdnEndpoint {
                    name: "ap-backup".to_string(),
                    base_url: "https://ap-backup.cdn.torsh.rs".to_string(),
                    region: "ap-northeast-1".to_string(),
                    priority: 2,
                    healthy: true,
                    last_health_check: None,
                    avg_response_time: None,
                    headers: HashMap::new(),
                },
            ];
        }
        _ => {
            // Default global configuration
        }
    }

    config
}

/// Create mirror configuration for a specific region
///
/// Generates mirror configurations optimized for specific geographic regions
/// with appropriate selection strategies and regional preferences.
///
/// # Arguments
/// * `region` - The target region ("us", "eu", etc.)
///
/// # Returns
/// A configured MirrorConfig optimized for the specified region
pub fn create_regional_mirror_config(region: &str) -> MirrorConfig {
    let mut config = MirrorConfig::default();

    match region {
        "us" => {
            config.mirrors = vec![
                MirrorServer {
                    id: "us-east-1".to_string(),
                    base_url: "https://us-east-1.mirror.torsh.rs".to_string(),
                    location: MirrorLocation {
                        country: "US".to_string(),
                        region: "Virginia".to_string(),
                        city: "Ashburn".to_string(),
                        latitude: Some(39.0458),
                        longitude: Some(-77.5073),
                        provider: "AWS".to_string(),
                    },
                    reliability_score: 0.99,
                    avg_response_time: Some(20),
                    last_successful_connection: None,
                    consecutive_failures: 0,
                    capacity: MirrorCapacity {
                        max_bandwidth: Some(10000),
                        current_load: Some(15.0),
                        max_connections: Some(1000),
                        current_connections: Some(150),
                    },
                    active: true,
                    metadata: HashMap::new(),
                },
                MirrorServer {
                    id: "us-west-1".to_string(),
                    base_url: "https://us-west-1.mirror.torsh.rs".to_string(),
                    location: MirrorLocation {
                        country: "US".to_string(),
                        region: "California".to_string(),
                        city: "San Francisco".to_string(),
                        latitude: Some(37.7749),
                        longitude: Some(-122.4194),
                        provider: "Google Cloud".to_string(),
                    },
                    reliability_score: 0.97,
                    avg_response_time: Some(25),
                    last_successful_connection: None,
                    consecutive_failures: 0,
                    capacity: MirrorCapacity {
                        max_bandwidth: Some(8000),
                        current_load: Some(20.0),
                        max_connections: Some(800),
                        current_connections: Some(160),
                    },
                    active: true,
                    metadata: HashMap::new(),
                },
            ];
        }
        "eu" => {
            config.mirrors = vec![
                MirrorServer {
                    id: "eu-central-1".to_string(),
                    base_url: "https://eu-central-1.mirror.torsh.rs".to_string(),
                    location: MirrorLocation {
                        country: "DE".to_string(),
                        region: "Frankfurt".to_string(),
                        city: "Frankfurt".to_string(),
                        latitude: Some(50.1109),
                        longitude: Some(8.6821),
                        provider: "AWS".to_string(),
                    },
                    reliability_score: 0.98,
                    avg_response_time: Some(15),
                    last_successful_connection: None,
                    consecutive_failures: 0,
                    capacity: MirrorCapacity {
                        max_bandwidth: Some(5000),
                        current_load: Some(10.0),
                        max_connections: Some(600),
                        current_connections: Some(60),
                    },
                    active: true,
                    metadata: HashMap::new(),
                },
                MirrorServer {
                    id: "eu-west-1".to_string(),
                    base_url: "https://eu-west-1.mirror.torsh.rs".to_string(),
                    location: MirrorLocation {
                        country: "GB".to_string(),
                        region: "London".to_string(),
                        city: "London".to_string(),
                        latitude: Some(51.5074),
                        longitude: Some(-0.1278),
                        provider: "Azure".to_string(),
                    },
                    reliability_score: 0.96,
                    avg_response_time: Some(18),
                    last_successful_connection: None,
                    consecutive_failures: 0,
                    capacity: MirrorCapacity {
                        max_bandwidth: Some(4000),
                        current_load: Some(25.0),
                        max_connections: Some(500),
                        current_connections: Some(125),
                    },
                    active: true,
                    metadata: HashMap::new(),
                },
            ];
        }
        _ => {
            // Default global configuration
        }
    }

    config
}

/// Validate a URL before attempting to download from it
///
/// This function performs basic URL validation to provide better error messages
/// to users before attempting downloads.
///
/// # Arguments
/// * `url` - The URL to validate
///
/// # Returns
/// * `Ok(())` if the URL appears valid
/// * `Err(TorshError)` with a descriptive message if invalid
///
/// # Examples
/// ```rust
/// use torsh_hub::download::config::validate_url;
///
/// // Valid URL
/// assert!(validate_url("https://example.com/model.torsh").is_ok());
///
/// // Invalid URL
/// assert!(validate_url("not-a-url").is_err());
/// ```
pub fn validate_url(url: &str) -> Result<()> {
    // Check if URL is empty
    if url.trim().is_empty() {
        return Err(TorshError::config_error_with_context(
            "URL cannot be empty",
            "URL validation",
        ));
    }

    // Basic URL format validation
    if !url.starts_with("http://") && !url.starts_with("https://") && !url.starts_with("ftp://") {
        return Err(TorshError::config_error_with_context(
            &format!(
                "URL must start with http://, https://, or ftp://, got: {}",
                url
            ),
            "URL validation",
        ));
    }

    // Check for obviously invalid characters
    if url.contains(' ') {
        return Err(TorshError::config_error_with_context(
            &format!("URL contains spaces: {}", url),
            "URL validation",
        ));
    }

    // Check if URL has a reasonable length (prevent extremely long URLs)
    if url.len() > 2048 {
        return Err(TorshError::config_error_with_context(
            &format!("URL is too long ({} characters, max 2048)", url.len()),
            "URL validation",
        ));
    }

    Ok(())
}

/// Validate multiple URLs at once
///
/// This is a convenience function for validating multiple URLs and collecting
/// all validation errors.
///
/// # Arguments
/// * `urls` - A slice of URLs to validate
///
/// # Returns
/// * `Ok(())` if all URLs are valid
/// * `Err(TorshError)` with details about all invalid URLs
pub fn validate_urls(urls: &[&str]) -> Result<()> {
    let mut errors = Vec::new();

    for (i, url) in urls.iter().enumerate() {
        if let Err(e) = validate_url(url) {
            errors.push(format!("URL {}: {}", i + 1, e));
        }
    }

    if !errors.is_empty() {
        return Err(TorshError::config_error_with_context(
            &format!("URL validation failed:\n{}", errors.join("\n")),
            "Multiple URL validation",
        ));
    }

    Ok(())
}

/// Serde module for Duration serialization/deserialization
mod duration_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        duration.as_secs().serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let secs = u64::deserialize(deserializer)?;
        Ok(Duration::from_secs(secs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_download_config_default() {
        let config = ParallelDownloadConfig::default();
        assert_eq!(config.max_concurrent_downloads, 4);
        assert_eq!(config.chunk_size, 1024 * 1024);
        assert_eq!(config.max_concurrent_chunks, 8);
        assert_eq!(config.timeout_seconds, 300);
        assert_eq!(config.max_retries, 3);
        assert!(config.enable_resume);
    }

    #[test]
    fn test_parallel_download_config_builder() {
        let config = ParallelDownloadConfig::builder()
            .max_concurrent_downloads(8)
            .chunk_size(2 * 1024 * 1024)
            .max_concurrent_chunks(16)
            .timeout_seconds(600)
            .max_retries(5)
            .enable_resume(false)
            .build()
            .unwrap();

        assert_eq!(config.max_concurrent_downloads, 8);
        assert_eq!(config.chunk_size, 2 * 1024 * 1024);
        assert_eq!(config.max_concurrent_chunks, 16);
        assert_eq!(config.timeout_seconds, 600);
        assert_eq!(config.max_retries, 5);
        assert!(!config.enable_resume);
    }

    #[test]
    fn test_parallel_download_config_validation() {
        let mut config = ParallelDownloadConfig::default();

        // Valid config should pass
        assert!(config.validate().is_ok());

        // Invalid max_concurrent_downloads
        config.max_concurrent_downloads = 0;
        assert!(config.validate().is_err());

        config = ParallelDownloadConfig::default();
        config.chunk_size = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_cdn_config_default() {
        let config = CdnConfig::default();
        assert_eq!(config.endpoints.len(), 2);
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.endpoint_timeout, Duration::from_secs(30));
        assert!(config.enable_health_check);
        assert_eq!(config.health_check_interval, 300);
        assert_eq!(config.failover_strategy, FailoverStrategy::Priority);
    }

    #[test]
    fn test_regional_cdn_config() {
        let us_config = create_regional_cdn_config("us-east");
        assert_eq!(us_config.endpoints.len(), 2);
        assert!(us_config.endpoints[0].name.contains("us-east"));

        let eu_config = create_regional_cdn_config("eu-west");
        assert_eq!(eu_config.endpoints.len(), 2);
        assert!(eu_config.endpoints[0].name.contains("eu-west"));

        let default_config = create_regional_cdn_config("unknown");
        assert_eq!(default_config.endpoints.len(), 2); // Should use default
    }

    #[test]
    fn test_mirror_config_default() {
        let config = MirrorConfig::default();
        assert_eq!(config.mirrors.len(), 2);
        assert_eq!(config.max_mirror_attempts, 3);
        assert_eq!(config.connection_timeout, Duration::from_secs(10));
        assert!(config.enable_auto_discovery);
        assert_eq!(config.benchmark_interval, 3600);
    }

    #[test]
    fn test_mirror_weights_default() {
        let weights = MirrorWeights::default();
        assert_eq!(weights.latency, 0.4);
        assert_eq!(weights.reliability, 0.3);
        assert_eq!(weights.load, 0.2);
        assert_eq!(weights.geographic, 0.1);
    }

    #[test]
    fn test_regional_mirror_config() {
        let us_config = create_regional_mirror_config("us");
        assert_eq!(us_config.mirrors.len(), 2);
        assert!(us_config.mirrors[0].id.contains("us-"));
        assert_eq!(us_config.mirrors[0].location.country, "US");

        let eu_config = create_regional_mirror_config("eu");
        assert_eq!(eu_config.mirrors.len(), 2);
        assert!(eu_config.mirrors[0].id.contains("eu-"));
    }

    #[test]
    fn test_url_validation() {
        // Valid URLs
        assert!(validate_url("https://example.com/file.txt").is_ok());
        assert!(validate_url("http://example.com/file.txt").is_ok());
        assert!(validate_url("ftp://example.com/file.txt").is_ok());

        // Invalid URLs
        assert!(validate_url("").is_err());
        assert!(validate_url("   ").is_err());
        assert!(validate_url("not-a-url").is_err());
        assert!(validate_url("https://example.com/file with spaces.txt").is_err());

        // Too long URL
        let long_url = format!("https://example.com/{}", "a".repeat(2048));
        assert!(validate_url(&long_url).is_err());
    }

    #[test]
    fn test_multiple_url_validation() {
        let valid_urls = vec!["https://example.com/1", "https://example.com/2"];
        assert!(validate_urls(&valid_urls).is_ok());

        let mixed_urls = vec!["https://example.com/1", "invalid-url"];
        assert!(validate_urls(&mixed_urls).is_err());

        let empty_urls: Vec<&str> = vec![];
        assert!(validate_urls(&empty_urls).is_ok());
    }

    #[test]
    fn test_failover_strategy_equality() {
        assert_eq!(FailoverStrategy::Priority, FailoverStrategy::Priority);
        assert_ne!(FailoverStrategy::Priority, FailoverStrategy::Fastest);
        assert_ne!(FailoverStrategy::RoundRobin, FailoverStrategy::Random);
    }

    #[test]
    fn test_mirror_selection_strategy_equality() {
        assert_eq!(
            MirrorSelectionStrategy::LowestLatency,
            MirrorSelectionStrategy::LowestLatency
        );
        assert_ne!(
            MirrorSelectionStrategy::LowestLatency,
            MirrorSelectionStrategy::HighestReliability
        );

        let weights1 = MirrorWeights::default();
        let weights2 = MirrorWeights {
            latency: 0.5,
            reliability: 0.3,
            load: 0.1,
            geographic: 0.1,
        };
        assert_ne!(
            MirrorSelectionStrategy::Weighted(weights1),
            MirrorSelectionStrategy::Weighted(weights2)
        );
    }
}
