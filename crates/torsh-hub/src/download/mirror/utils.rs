//! Mirror Management Utilities and Helper Functions
//!
//! This module provides utility functions for mirror management including
//! configuration creation, validation helpers, test utilities, and common
//! operations that support the mirror management system.

use super::types::*;
use std::collections::HashMap;
use torsh_core::error::{Result, TorshError};

// ================================================================================================
// Configuration Creation Utilities
// ================================================================================================

/// Create optimized mirror configuration for a specific region
///
/// This function creates a pre-configured mirror setup optimized for different
/// geographic regions with appropriate mirror servers and settings.
///
/// # Arguments
/// * `region` - Region identifier ("us", "eu", "asia", "global", etc.)
///
/// # Returns
/// * `MirrorConfig` - Optimized configuration for the region
///
/// # Examples
/// ```rust
/// use torsh_hub::download::mirror::utils::create_regional_mirror_config;
///
/// let us_config = create_regional_mirror_config("us");
/// let eu_config = create_regional_mirror_config("eu");
/// ```
pub fn create_regional_mirror_config(region: &str) -> MirrorConfig {
    let mut config = MirrorConfig::default();

    match region {
        "us" => {
            config.mirrors = vec![create_us_east_mirror(), create_us_west_mirror()];
            config.selection_strategy = MirrorSelectionStrategy::Geographic;
            config.enable_geographic_optimization = true;
        }
        "eu" => {
            config.mirrors = vec![create_eu_west_mirror(), create_eu_central_mirror()];
            config.selection_strategy = MirrorSelectionStrategy::Geographic;
            config.enable_geographic_optimization = true;
        }
        "asia" => {
            config.mirrors = vec![create_asia_pacific_mirror(), create_asia_east_mirror()];
            config.selection_strategy = MirrorSelectionStrategy::Geographic;
            config.enable_geographic_optimization = true;
        }
        "global" => {
            config.mirrors = vec![
                create_us_east_mirror(),
                create_eu_west_mirror(),
                create_asia_pacific_mirror(),
            ];
            config.selection_strategy = MirrorSelectionStrategy::Adaptive;
            config.enable_geographic_optimization = true;
        }
        _ => {
            // Default configuration with basic mirrors
            config = MirrorConfig::default();
        }
    }

    config
}

/// Create a US East Coast mirror server configuration
fn create_us_east_mirror() -> MirrorServer {
    MirrorServer {
        id: "us-east-primary".to_string(),
        base_url: "https://us-east.torsh-mirrors.io".to_string(),
        reliability_score: 0.98,
        avg_response_time: Some(50),
        consecutive_failures: 0,
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
        capacity: MirrorCapacity {
            max_bandwidth: Some(10000), // 10 Gbps
            current_load: Some(45.0),
            max_connections: Some(2000),
            current_connections: Some(1000),
            storage_capacity: Some(50_000_000), // 50TB
            storage_used: Some(20_000_000),     // 20TB used
            cpu_utilization: Some(35.0),
            memory_utilization: Some(60.0),
        },
        active: true,
        metadata: HashMap::new(),
        priority_weight: 1.0,
        provider_info: ProviderInfo {
            name: "AWS".to_string(),
            network_tier: Some("Premium".to_string()),
            cdn_integration: true,
            edge_location: Some("IAD".to_string()),
            network_quality: NetworkQuality {
                quality_score: 0.95,
                ixp_connections: 5,
                transit_diversity: 4,
                peering_count: 100,
            },
        },
        performance_history: Vec::new(),
        last_successful_connection: None,
    }
}

/// Create a US West Coast mirror server configuration
fn create_us_west_mirror() -> MirrorServer {
    MirrorServer {
        id: "us-west-primary".to_string(),
        base_url: "https://us-west.torsh-mirrors.io".to_string(),
        reliability_score: 0.96,
        avg_response_time: Some(60),
        consecutive_failures: 0,
        location: MirrorLocation {
            country: "US".to_string(),
            region: "California".to_string(),
            city: "San Francisco".to_string(),
            latitude: Some(37.7749),
            longitude: Some(-122.4194),
            provider: "GCP".to_string(),
            timezone: Some("America/Los_Angeles".to_string()),
            datacenter: Some("us-west1-a".to_string()),
        },
        capacity: MirrorCapacity {
            max_bandwidth: Some(10000), // 10 Gbps
            current_load: Some(50.0),
            max_connections: Some(1500),
            current_connections: Some(800),
            storage_capacity: Some(40_000_000), // 40TB
            storage_used: Some(15_000_000),     // 15TB used
            cpu_utilization: Some(40.0),
            memory_utilization: Some(55.0),
        },
        active: true,
        metadata: HashMap::new(),
        priority_weight: 1.0,
        provider_info: ProviderInfo {
            name: "Google Cloud".to_string(),
            network_tier: Some("Premium".to_string()),
            cdn_integration: true,
            edge_location: Some("SFO".to_string()),
            network_quality: NetworkQuality {
                quality_score: 0.92,
                ixp_connections: 4,
                transit_diversity: 3,
                peering_count: 80,
            },
        },
        performance_history: Vec::new(),
        last_successful_connection: None,
    }
}

/// Create a European West mirror server configuration
fn create_eu_west_mirror() -> MirrorServer {
    MirrorServer {
        id: "eu-west-primary".to_string(),
        base_url: "https://eu-west.torsh-mirrors.io".to_string(),
        reliability_score: 0.94,
        avg_response_time: Some(80),
        consecutive_failures: 0,
        location: MirrorLocation {
            country: "GB".to_string(),
            region: "England".to_string(),
            city: "London".to_string(),
            latitude: Some(51.5074),
            longitude: Some(-0.1278),
            provider: "Microsoft Azure".to_string(),
            timezone: Some("Europe/London".to_string()),
            datacenter: Some("uk-south-1a".to_string()),
        },
        capacity: MirrorCapacity {
            max_bandwidth: Some(8000), // 8 Gbps
            current_load: Some(55.0),
            max_connections: Some(1200),
            current_connections: Some(600),
            storage_capacity: Some(35_000_000), // 35TB
            storage_used: Some(12_000_000),     // 12TB used
            cpu_utilization: Some(45.0),
            memory_utilization: Some(65.0),
        },
        active: true,
        metadata: HashMap::new(),
        priority_weight: 1.0,
        provider_info: ProviderInfo {
            name: "Microsoft Azure".to_string(),
            network_tier: Some("Premium".to_string()),
            cdn_integration: true,
            edge_location: Some("LHR".to_string()),
            network_quality: NetworkQuality {
                quality_score: 0.89,
                ixp_connections: 6,
                transit_diversity: 5,
                peering_count: 90,
            },
        },
        performance_history: Vec::new(),
        last_successful_connection: None,
    }
}

/// Create a European Central mirror server configuration
fn create_eu_central_mirror() -> MirrorServer {
    MirrorServer {
        id: "eu-central-primary".to_string(),
        base_url: "https://eu-central.torsh-mirrors.io".to_string(),
        reliability_score: 0.93,
        avg_response_time: Some(70),
        consecutive_failures: 0,
        location: MirrorLocation {
            country: "DE".to_string(),
            region: "Hessen".to_string(),
            city: "Frankfurt".to_string(),
            latitude: Some(50.1109),
            longitude: Some(8.6821),
            provider: "Hetzner".to_string(),
            timezone: Some("Europe/Berlin".to_string()),
            datacenter: Some("fsn1-dc14".to_string()),
        },
        capacity: MirrorCapacity {
            max_bandwidth: Some(8000), // 8 Gbps
            current_load: Some(40.0),
            max_connections: Some(1000),
            current_connections: Some(700),
            storage_capacity: Some(45_000_000), // 45TB
            storage_used: Some(20_000_000),     // 20TB used
            cpu_utilization: Some(30.0),
            memory_utilization: Some(50.0),
        },
        active: true,
        metadata: HashMap::new(),
        priority_weight: 1.0,
        provider_info: ProviderInfo {
            name: "Hetzner".to_string(),
            network_tier: Some("Standard".to_string()),
            cdn_integration: false,
            edge_location: Some("FRA".to_string()),
            network_quality: NetworkQuality {
                quality_score: 0.87,
                ixp_connections: 4,
                transit_diversity: 3,
                peering_count: 60,
            },
        },
        performance_history: Vec::new(),
        last_successful_connection: None,
    }
}

/// Create an Asia Pacific mirror server configuration
fn create_asia_pacific_mirror() -> MirrorServer {
    MirrorServer {
        id: "ap-southeast-primary".to_string(),
        base_url: "https://ap-southeast.torsh-mirrors.io".to_string(),
        reliability_score: 0.91,
        avg_response_time: Some(90),
        consecutive_failures: 0,
        location: MirrorLocation {
            country: "SG".to_string(),
            region: "Singapore".to_string(),
            city: "Singapore".to_string(),
            latitude: Some(1.3521),
            longitude: Some(103.8198),
            provider: "DigitalOcean".to_string(),
            timezone: Some("Asia/Singapore".to_string()),
            datacenter: Some("sgp1".to_string()),
        },
        capacity: MirrorCapacity {
            max_bandwidth: Some(6000), // 6 Gbps
            current_load: Some(60.0),
            max_connections: Some(800),
            current_connections: Some(500),
            storage_capacity: Some(30_000_000), // 30TB
            storage_used: Some(12_000_000),     // 12TB used
            cpu_utilization: Some(50.0),
            memory_utilization: Some(70.0),
        },
        active: true,
        metadata: HashMap::new(),
        priority_weight: 1.0,
        provider_info: ProviderInfo {
            name: "DigitalOcean".to_string(),
            network_tier: Some("Standard".to_string()),
            cdn_integration: false,
            edge_location: Some("SIN".to_string()),
            network_quality: NetworkQuality {
                quality_score: 0.84,
                ixp_connections: 3,
                transit_diversity: 2,
                peering_count: 40,
            },
        },
        performance_history: Vec::new(),
        last_successful_connection: None,
    }
}

/// Create an Asia East mirror server configuration
fn create_asia_east_mirror() -> MirrorServer {
    MirrorServer {
        id: "ap-east-primary".to_string(),
        base_url: "https://ap-east.torsh-mirrors.io".to_string(),
        reliability_score: 0.88,
        avg_response_time: Some(110),
        consecutive_failures: 0,
        location: MirrorLocation {
            country: "JP".to_string(),
            region: "Tokyo".to_string(),
            city: "Tokyo".to_string(),
            latitude: Some(35.6762),
            longitude: Some(139.6503),
            provider: "Linode".to_string(),
            timezone: Some("Asia/Tokyo".to_string()),
            datacenter: Some("jp-osa-1".to_string()),
        },
        capacity: MirrorCapacity {
            max_bandwidth: Some(5000), // 5 Gbps
            current_load: Some(70.0),
            max_connections: Some(600),
            current_connections: Some(400),
            storage_capacity: Some(25_000_000), // 25TB
            storage_used: Some(10_000_000),     // 10TB used (instead of available_storage)
            cpu_utilization: Some(60.0),
            memory_utilization: Some(75.0),
        },
        active: true,
        metadata: HashMap::new(),
        priority_weight: 1.0,
        provider_info: ProviderInfo {
            name: "Linode".to_string(),
            network_tier: Some("Standard".to_string()),
            cdn_integration: false,
            edge_location: Some("NRT".to_string()),
            network_quality: NetworkQuality {
                quality_score: 0.81,
                ixp_connections: 2,
                transit_diversity: 2,
                peering_count: 30,
            },
        },
        performance_history: Vec::new(),
        last_successful_connection: None,
    }
}

// ================================================================================================
// Validation and Helper Utilities
// ================================================================================================

/// Validate mirror configuration for consistency and correctness
pub fn validate_mirror_config(config: &MirrorConfig) -> Result<()> {
    if config.mirrors.is_empty() {
        return Err(TorshError::IoError(
            "Mirror configuration must contain at least one mirror".to_string(),
        ));
    }

    if config.max_mirror_attempts == 0 {
        return Err(TorshError::IoError(
            "Max mirror attempts must be greater than 0".to_string(),
        ));
    }

    if config.min_reliability_score < 0.0 || config.min_reliability_score > 1.0 {
        return Err(TorshError::IoError(
            "Min reliability score must be between 0.0 and 1.0".to_string(),
        ));
    }

    if config.max_response_time == 0 {
        return Err(TorshError::IoError(
            "Max response time must be greater than 0".to_string(),
        ));
    }

    // Validate individual mirrors
    for (i, mirror) in config.mirrors.iter().enumerate() {
        validate_mirror_server(mirror)
            .map_err(|e| TorshError::IoError(format!("Mirror {} validation failed: {}", i, e)))?;
    }

    // Check for duplicate mirror IDs
    let mut seen_ids = std::collections::HashSet::new();
    for mirror in &config.mirrors {
        if !seen_ids.insert(&mirror.id) {
            return Err(TorshError::IoError(format!(
                "Duplicate mirror ID: {}",
                mirror.id
            )));
        }
    }

    Ok(())
}

/// Validate individual mirror server configuration
pub fn validate_mirror_server(mirror: &MirrorServer) -> Result<()> {
    if mirror.id.is_empty() {
        return Err(TorshError::IoError("Mirror ID cannot be empty".to_string()));
    }

    if mirror.base_url.is_empty() {
        return Err(TorshError::IoError(
            "Mirror base URL cannot be empty".to_string(),
        ));
    }

    if !mirror.base_url.starts_with("http://") && !mirror.base_url.starts_with("https://") {
        return Err(TorshError::IoError(
            "Mirror base URL must start with http:// or https://".to_string(),
        ));
    }

    if mirror.reliability_score < 0.0 || mirror.reliability_score > 1.0 {
        return Err(TorshError::IoError(
            "Mirror reliability score must be between 0.0 and 1.0".to_string(),
        ));
    }

    if mirror.priority_weight < 0.0 {
        return Err(TorshError::IoError(
            "Mirror priority weight cannot be negative".to_string(),
        ));
    }

    // Validate geographic coordinates if provided
    if let (Some(lat), Some(lon)) = (mirror.location.latitude, mirror.location.longitude) {
        if lat < -90.0 || lat > 90.0 {
            return Err(TorshError::IoError(
                "Mirror latitude must be between -90.0 and 90.0".to_string(),
            ));
        }
        if lon < -180.0 || lon > 180.0 {
            return Err(TorshError::IoError(
                "Mirror longitude must be between -180.0 and 180.0".to_string(),
            ));
        }
    }

    // Validate capacity information if provided
    if let Some(load) = mirror.capacity.current_load {
        if load < 0.0 || load > 100.0 {
            return Err(TorshError::IoError(
                "Mirror current load must be between 0.0 and 100.0".to_string(),
            ));
        }
    }

    Ok(())
}

/// Create a mirror server with safe defaults for testing
pub fn create_test_mirror(id: &str, base_url: &str, country: &str, city: &str) -> MirrorServer {
    MirrorServer {
        id: id.to_string(),
        base_url: base_url.to_string(),
        reliability_score: 0.9,
        avg_response_time: Some(100),
        consecutive_failures: 0,
        location: MirrorLocation {
            country: country.to_string(),
            region: "Test Region".to_string(),
            city: city.to_string(),
            latitude: Some(40.0),
            longitude: Some(-74.0),
            provider: "Test Provider".to_string(),
            timezone: Some("UTC".to_string()),
            datacenter: Some("test-dc-1".to_string()),
        },
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
            name: "Test Provider".to_string(),
            network_tier: Some("Premium".to_string()),
            cdn_integration: true,
            edge_location: Some("TEST".to_string()),
            network_quality: NetworkQuality::default(),
        },
        performance_history: Vec::new(),
        last_successful_connection: None,
    }
}

/// Create a mirror server with custom parameters for testing
pub fn create_custom_test_mirror(
    id: &str,
    base_url: &str,
    reliability: f64,
    latency: Option<u64>,
    active: bool,
) -> MirrorServer {
    MirrorServer {
        id: id.to_string(),
        base_url: base_url.to_string(),
        reliability_score: reliability,
        avg_response_time: latency,
        consecutive_failures: if active { 0 } else { 5 },
        location: MirrorLocation {
            country: "US".to_string(),
            region: "Test Region".to_string(),
            city: "Test City".to_string(),
            latitude: Some(40.0),
            longitude: Some(-74.0),
            provider: "Test Provider".to_string(),
            timezone: Some("America/New_York".to_string()),
            datacenter: Some("test-dc-1".to_string()),
        },
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
        active,
        metadata: HashMap::new(),
        priority_weight: 1.0,
        provider_info: ProviderInfo {
            name: "Test Provider".to_string(),
            network_tier: Some("Standard".to_string()),
            cdn_integration: false,
            edge_location: None,
            network_quality: NetworkQuality::default(),
        },
        performance_history: Vec::new(),
        last_successful_connection: None,
    }
}

// ================================================================================================
// Mirror Selection Helpers
// ================================================================================================

/// Filter mirrors by health status
pub fn filter_healthy_mirrors(mirrors: &[MirrorServer]) -> Vec<&MirrorServer> {
    mirrors
        .iter()
        .filter(|m| {
            m.active
                && m.consecutive_failures < 3
                && m.reliability_score >= 0.7
                && m.avg_response_time.map_or(true, |latency| latency < 300) // Exclude mirrors with latency >= 300ms
        })
        .collect()
}

/// Filter mirrors by geographic region
pub fn filter_mirrors_by_region<'a>(
    mirrors: &'a [MirrorServer],
    countries: &[&str],
) -> Vec<&'a MirrorServer> {
    mirrors
        .iter()
        .filter(|m| countries.contains(&m.location.country.as_str()))
        .collect()
}

/// Filter mirrors by network tier (string-based matching)
pub fn filter_mirrors_by_network_tier<'a>(
    mirrors: &'a [MirrorServer],
    target_tier: &str,
) -> Vec<&'a MirrorServer> {
    mirrors
        .iter()
        .filter(|m| {
            m.provider_info
                .network_tier
                .as_ref()
                .map_or(false, |tier| tier == target_tier)
        })
        .collect()
}

/// Get mirrors with low latency (below threshold)
pub fn get_low_latency_mirrors(
    mirrors: &[MirrorServer],
    max_latency_ms: u64,
) -> Vec<&MirrorServer> {
    mirrors
        .iter()
        .filter(|m| {
            m.avg_response_time
                .map_or(false, |latency| latency <= max_latency_ms)
        })
        .collect()
}

/// Get mirrors with high reliability (above threshold)
pub fn get_high_reliability_mirrors(
    mirrors: &[MirrorServer],
    min_reliability: f64,
) -> Vec<&MirrorServer> {
    mirrors
        .iter()
        .filter(|m| m.reliability_score >= min_reliability)
        .collect()
}

/// Get mirrors with low load (below threshold)
pub fn get_low_load_mirrors(mirrors: &[MirrorServer], max_load: f32) -> Vec<&MirrorServer> {
    mirrors
        .iter()
        .filter(|m| {
            m.capacity
                .current_load
                .map_or(true, |load| load <= max_load)
        })
        .collect()
}

// ================================================================================================
// Performance Analysis Helpers
// ================================================================================================

/// Calculate average response time for a set of mirrors
pub fn calculate_average_response_time(mirrors: &[MirrorServer]) -> Option<f64> {
    let response_times: Vec<u64> = mirrors.iter().filter_map(|m| m.avg_response_time).collect();

    if response_times.is_empty() {
        None
    } else {
        let sum: u64 = response_times.iter().sum();
        Some(sum as f64 / response_times.len() as f64)
    }
}

/// Calculate average reliability score for a set of mirrors
pub fn calculate_average_reliability(mirrors: &[MirrorServer]) -> f64 {
    if mirrors.is_empty() {
        return 0.0;
    }

    let sum: f64 = mirrors.iter().map(|m| m.reliability_score).sum();
    sum / mirrors.len() as f64
}

/// Calculate load distribution statistics for a set of mirrors
pub fn calculate_load_statistics(mirrors: &[MirrorServer]) -> LoadStatistics {
    let loads: Vec<f32> = mirrors
        .iter()
        .filter_map(|m| m.capacity.current_load)
        .collect();

    if loads.is_empty() {
        return LoadStatistics::default();
    }

    let total: f32 = loads.iter().sum();
    let average = total / loads.len() as f32;

    let mut sorted_loads = loads.clone();
    sorted_loads.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let median = if sorted_loads.len() % 2 == 0 {
        let mid = sorted_loads.len() / 2;
        (sorted_loads[mid - 1] + sorted_loads[mid]) / 2.0
    } else {
        sorted_loads[sorted_loads.len() / 2]
    };

    let min = sorted_loads.first().copied().unwrap_or(0.0);
    let max = sorted_loads.last().copied().unwrap_or(0.0);

    LoadStatistics {
        average,
        median,
        min,
        max,
        total_mirrors: mirrors.len(),
        mirrors_with_load_data: loads.len(),
    }
}

/// Mirror load distribution statistics
#[derive(Debug, Clone, Default)]
pub struct LoadStatistics {
    pub average: f32,
    pub median: f32,
    pub min: f32,
    pub max: f32,
    pub total_mirrors: usize,
    pub mirrors_with_load_data: usize,
}

// ================================================================================================
// URL and Network Utilities
// ================================================================================================

/// Extract hostname from a mirror URL
pub fn extract_hostname(url: &str) -> Result<String> {
    let url = url.trim();
    if !url.starts_with("http://") && !url.starts_with("https://") {
        return Err(TorshError::IoError(
            "URL must start with http:// or https://".to_string(),
        ));
    }

    let without_protocol = url
        .split("://")
        .nth(1)
        .ok_or_else(|| TorshError::IoError("Invalid URL format".to_string()))?;

    let hostname = without_protocol
        .split('/')
        .next()
        .ok_or_else(|| TorshError::IoError("Could not extract hostname".to_string()))?
        .split(':')
        .next()
        .ok_or_else(|| TorshError::IoError("Could not extract hostname".to_string()))?;

    Ok(hostname.to_string())
}

/// Normalize a mirror URL by removing trailing slashes and fragments
pub fn normalize_mirror_url(url: &str) -> String {
    let mut normalized = url.trim().to_string();

    // Remove fragment identifier
    if let Some(fragment_pos) = normalized.find('#') {
        normalized.truncate(fragment_pos);
    }

    // Remove query parameters
    if let Some(query_pos) = normalized.find('?') {
        normalized.truncate(query_pos);
    }

    // Remove trailing slash
    while normalized.ends_with('/') {
        normalized.pop();
    }

    normalized
}

/// Check if a URL uses HTTPS
pub fn is_secure_url(url: &str) -> bool {
    url.trim().starts_with("https://")
}

/// Convert HTTP URL to HTTPS if possible
pub fn secure_url(url: &str) -> String {
    if url.starts_with("http://") {
        url.replace("http://", "https://")
    } else {
        url.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_regional_config() {
        let us_config = create_regional_mirror_config("us");
        assert_eq!(us_config.mirrors.len(), 2);
        assert!(us_config.mirrors.iter().all(|m| m.location.country == "US"));
        assert_eq!(
            us_config.selection_strategy,
            MirrorSelectionStrategy::Geographic
        );

        let eu_config = create_regional_mirror_config("eu");
        assert_eq!(eu_config.mirrors.len(), 2);
        assert!(eu_config
            .mirrors
            .iter()
            .any(|m| m.location.country == "GB" || m.location.country == "DE"));

        let asia_config = create_regional_mirror_config("asia");
        assert_eq!(asia_config.mirrors.len(), 2);
        assert!(asia_config
            .mirrors
            .iter()
            .any(|m| m.location.country == "SG" || m.location.country == "JP"));

        let global_config = create_regional_mirror_config("global");
        assert_eq!(global_config.mirrors.len(), 3);
        assert_eq!(
            global_config.selection_strategy,
            MirrorSelectionStrategy::Adaptive
        );
    }

    #[test]
    fn test_mirror_config_validation() {
        let valid_config = create_regional_mirror_config("us");
        assert!(validate_mirror_config(&valid_config).is_ok());

        // Test empty mirrors
        let mut invalid_config = valid_config.clone();
        invalid_config.mirrors.clear();
        assert!(validate_mirror_config(&invalid_config).is_err());

        // Test invalid max attempts
        let mut invalid_config = valid_config.clone();
        invalid_config.max_mirror_attempts = 0;
        assert!(validate_mirror_config(&invalid_config).is_err());

        // Test invalid reliability score
        let mut invalid_config = valid_config.clone();
        invalid_config.min_reliability_score = 1.5;
        assert!(validate_mirror_config(&invalid_config).is_err());

        // Test duplicate mirror IDs
        let mut invalid_config = valid_config.clone();
        invalid_config.mirrors[1].id = invalid_config.mirrors[0].id.clone();
        assert!(validate_mirror_config(&invalid_config).is_err());
    }

    #[test]
    fn test_mirror_server_validation() {
        let valid_mirror =
            create_test_mirror("test", "https://test.example.com", "US", "Test City");
        assert!(validate_mirror_server(&valid_mirror).is_ok());

        // Test empty ID
        let mut invalid_mirror = valid_mirror.clone();
        invalid_mirror.id = "".to_string();
        assert!(validate_mirror_server(&invalid_mirror).is_err());

        // Test empty URL
        let mut invalid_mirror = valid_mirror.clone();
        invalid_mirror.base_url = "".to_string();
        assert!(validate_mirror_server(&invalid_mirror).is_err());

        // Test invalid URL protocol
        let mut invalid_mirror = valid_mirror.clone();
        invalid_mirror.base_url = "ftp://test.example.com".to_string();
        assert!(validate_mirror_server(&invalid_mirror).is_err());

        // Test invalid reliability score
        let mut invalid_mirror = valid_mirror.clone();
        invalid_mirror.reliability_score = 1.5;
        assert!(validate_mirror_server(&invalid_mirror).is_err());

        // Test invalid coordinates
        let mut invalid_mirror = valid_mirror.clone();
        invalid_mirror.location.latitude = Some(91.0);
        assert!(validate_mirror_server(&invalid_mirror).is_err());
    }

    #[test]
    fn test_filter_functions() {
        let mirrors = vec![
            create_custom_test_mirror(
                "healthy",
                "https://healthy.example.com",
                0.9,
                Some(50),
                true,
            ),
            create_custom_test_mirror(
                "unhealthy",
                "https://unhealthy.example.com",
                0.5,
                Some(200),
                false,
            ),
            create_custom_test_mirror("slow", "https://slow.example.com", 0.8, Some(500), true),
        ];

        let healthy = filter_healthy_mirrors(&mirrors);
        assert_eq!(healthy.len(), 1);
        assert_eq!(healthy[0].id, "healthy");

        let low_latency = get_low_latency_mirrors(&mirrors, 100);
        assert_eq!(low_latency.len(), 1);
        assert_eq!(low_latency[0].id, "healthy");

        let high_reliability = get_high_reliability_mirrors(&mirrors, 0.85);
        assert_eq!(high_reliability.len(), 1);
        assert_eq!(high_reliability[0].id, "healthy");
    }

    #[test]
    fn test_performance_calculations() {
        let mirrors = vec![
            create_custom_test_mirror(
                "mirror1",
                "https://mirror1.example.com",
                0.9,
                Some(100),
                true,
            ),
            create_custom_test_mirror(
                "mirror2",
                "https://mirror2.example.com",
                0.8,
                Some(200),
                true,
            ),
            create_custom_test_mirror("mirror3", "https://mirror3.example.com", 0.7, None, true),
        ];

        let avg_response_time = calculate_average_response_time(&mirrors);
        assert!(avg_response_time.is_some());
        assert_eq!(avg_response_time.unwrap(), 150.0);

        let avg_reliability = calculate_average_reliability(&mirrors);
        assert!((avg_reliability - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_url_utilities() {
        assert_eq!(
            extract_hostname("https://example.com/path").unwrap(),
            "example.com"
        );
        assert_eq!(
            extract_hostname("http://sub.example.com:8080/path").unwrap(),
            "sub.example.com"
        );
        assert!(extract_hostname("invalid-url").is_err());

        assert_eq!(
            normalize_mirror_url("https://example.com/path/"),
            "https://example.com/path"
        );
        assert_eq!(
            normalize_mirror_url("https://example.com/path?query=1#frag"),
            "https://example.com/path"
        );

        assert!(is_secure_url("https://example.com"));
        assert!(!is_secure_url("http://example.com"));

        assert_eq!(secure_url("http://example.com"), "https://example.com");
        assert_eq!(secure_url("https://example.com"), "https://example.com");
    }

    #[test]
    fn test_load_statistics() {
        let mut mirrors = vec![
            create_test_mirror("mirror1", "https://mirror1.example.com", "US", "City1"),
            create_test_mirror("mirror2", "https://mirror2.example.com", "US", "City2"),
            create_test_mirror("mirror3", "https://mirror3.example.com", "US", "City3"),
        ];

        // Set load values
        mirrors[0].capacity.current_load = Some(30.0);
        mirrors[1].capacity.current_load = Some(50.0);
        mirrors[2].capacity.current_load = Some(70.0);

        let stats = calculate_load_statistics(&mirrors);
        assert_eq!(stats.total_mirrors, 3);
        assert_eq!(stats.mirrors_with_load_data, 3);
        assert_eq!(stats.average, 50.0);
        assert_eq!(stats.median, 50.0);
        assert_eq!(stats.min, 30.0);
        assert_eq!(stats.max, 70.0);
    }

    #[test]
    fn test_create_test_mirrors() {
        let mirror = create_test_mirror("test", "https://test.example.com", "US", "Test City");
        assert_eq!(mirror.id, "test");
        assert_eq!(mirror.base_url, "https://test.example.com");
        assert_eq!(mirror.location.country, "US");
        assert_eq!(mirror.location.city, "Test City");
        assert_eq!(mirror.reliability_score, 0.9);
        assert!(mirror.active);

        let custom_mirror = create_custom_test_mirror(
            "custom",
            "https://custom.example.com",
            0.75,
            Some(150),
            false,
        );
        assert_eq!(custom_mirror.id, "custom");
        assert_eq!(custom_mirror.reliability_score, 0.75);
        assert_eq!(custom_mirror.avg_response_time, Some(150));
        assert!(!custom_mirror.active);
        assert_eq!(custom_mirror.consecutive_failures, 5);
    }
}
