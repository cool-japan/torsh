//! Content Delivery Network (CDN) support for fast package distribution
//!
//! This module provides CDN integration for efficient package distribution
//! with geographic load balancing, caching, and edge node management.

use std::collections::HashMap;
use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};
use torsh_core::error::{Result, TorshError};

/// CDN provider type
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CdnProvider {
    /// Cloudflare CDN
    Cloudflare,
    /// AWS CloudFront
    CloudFront,
    /// Google Cloud CDN
    GoogleCdn,
    /// Azure CDN
    AzureCdn,
    /// Fastly CDN
    Fastly,
    /// Custom CDN provider
    Custom(String),
}

/// CDN configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CdnConfig {
    /// CDN provider
    pub provider: CdnProvider,
    /// CDN endpoint URL
    pub endpoint: String,
    /// API key for CDN management
    pub api_key: Option<String>,
    /// Cache TTL in seconds
    pub cache_ttl: u64,
    /// Enable compression at edge
    pub edge_compression: bool,
    /// Geographic regions to use
    pub regions: Vec<CdnRegion>,
    /// Custom headers to add
    pub custom_headers: HashMap<String, String>,
}

/// CDN geographic region
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CdnRegion {
    /// North America
    NorthAmerica,
    /// Europe
    Europe,
    /// Asia Pacific
    AsiaPacific,
    /// South America
    SouthAmerica,
    /// Africa
    Africa,
    /// Middle East
    MiddleEast,
    /// Oceania
    Oceania,
}

/// CDN cache control settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheControl {
    /// Maximum age in seconds
    pub max_age: u64,
    /// Enable public caching
    pub public: bool,
    /// Enable private caching
    pub private: bool,
    /// No cache directive
    pub no_cache: bool,
    /// No store directive
    pub no_store: bool,
    /// Must revalidate directive
    pub must_revalidate: bool,
}

/// CDN edge node information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeNode {
    /// Node ID
    pub id: String,
    /// Node location
    pub location: String,
    /// Region
    pub region: CdnRegion,
    /// Node status
    pub status: EdgeNodeStatus,
    /// Current load percentage (0-100)
    pub load: u8,
    /// Latency in milliseconds
    pub latency_ms: u64,
    /// Bandwidth capacity in Mbps
    pub bandwidth_mbps: u64,
}

/// Edge node status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EdgeNodeStatus {
    /// Node is active and healthy
    Active,
    /// Node is degraded (partial functionality)
    Degraded,
    /// Node is offline
    Offline,
    /// Node is under maintenance
    Maintenance,
}

/// CDN statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CdnStatistics {
    /// Total requests served
    pub total_requests: u64,
    /// Cache hit rate (0.0-1.0)
    pub cache_hit_rate: f64,
    /// Total bytes transferred
    pub bytes_transferred: u64,
    /// Average response time in milliseconds
    pub avg_response_ms: f64,
    /// Requests by region
    pub requests_by_region: HashMap<String, u64>,
    /// Error rate (0.0-1.0)
    pub error_rate: f64,
}

/// CDN manager for package distribution
pub struct CdnManager {
    /// CDN configuration
    config: CdnConfig,
    /// Available edge nodes
    edge_nodes: Vec<EdgeNode>,
    /// CDN statistics
    statistics: CdnStatistics,
    /// Cache entries
    cache: HashMap<String, CachedItem>,
}

/// Cached item metadata
#[derive(Debug, Clone)]
struct CachedItem {
    /// Cache key
    _key: String,
    /// URL to cached content
    url: String,
    /// Cache expiration time
    expires_at: SystemTime,
    /// Content size in bytes
    _size: u64,
    /// Number of hits
    hits: u64,
}

impl Default for CdnConfig {
    fn default() -> Self {
        Self {
            provider: CdnProvider::Cloudflare,
            endpoint: "https://cdn.torsh.rs".to_string(),
            api_key: None,
            cache_ttl: 86400, // 24 hours
            edge_compression: true,
            regions: vec![
                CdnRegion::NorthAmerica,
                CdnRegion::Europe,
                CdnRegion::AsiaPacific,
            ],
            custom_headers: HashMap::new(),
        }
    }
}

impl CdnConfig {
    /// Create a new CDN configuration
    pub fn new(provider: CdnProvider, endpoint: String) -> Self {
        Self {
            provider,
            endpoint,
            ..Default::default()
        }
    }

    /// Set API key
    pub fn with_api_key(mut self, api_key: String) -> Self {
        self.api_key = Some(api_key);
        self
    }

    /// Set cache TTL
    pub fn with_cache_ttl(mut self, ttl: u64) -> Self {
        self.cache_ttl = ttl;
        self
    }

    /// Enable edge compression
    pub fn with_edge_compression(mut self, enabled: bool) -> Self {
        self.edge_compression = enabled;
        self
    }

    /// Add region
    pub fn add_region(mut self, region: CdnRegion) -> Self {
        if !self.regions.contains(&region) {
            self.regions.push(region);
        }
        self
    }

    /// Add custom header
    pub fn add_header(mut self, key: String, value: String) -> Self {
        self.custom_headers.insert(key, value);
        self
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if self.endpoint.is_empty() {
            return Err(TorshError::InvalidArgument(
                "CDN endpoint cannot be empty".to_string(),
            ));
        }

        if self.regions.is_empty() {
            return Err(TorshError::InvalidArgument(
                "At least one region must be configured".to_string(),
            ));
        }

        if self.cache_ttl == 0 {
            return Err(TorshError::InvalidArgument(
                "Cache TTL must be greater than zero".to_string(),
            ));
        }

        Ok(())
    }
}

impl Default for CacheControl {
    fn default() -> Self {
        Self {
            max_age: 86400, // 24 hours
            public: true,
            private: false,
            no_cache: false,
            no_store: false,
            must_revalidate: false,
        }
    }
}

impl CacheControl {
    /// Create cache control for immutable content
    pub fn immutable() -> Self {
        Self {
            max_age: 31536000, // 1 year
            public: true,
            private: false,
            no_cache: false,
            no_store: false,
            must_revalidate: false,
        }
    }

    /// Create cache control for no caching
    pub fn no_cache() -> Self {
        Self {
            max_age: 0,
            public: false,
            private: false,
            no_cache: true,
            no_store: true,
            must_revalidate: true,
        }
    }

    /// Generate Cache-Control header value
    pub fn to_header(&self) -> String {
        let mut parts = Vec::new();

        if self.public {
            parts.push("public".to_string());
        }
        if self.private {
            parts.push("private".to_string());
        }
        if self.no_cache {
            parts.push("no-cache".to_string());
        }
        if self.no_store {
            parts.push("no-store".to_string());
        }
        if self.must_revalidate {
            parts.push("must-revalidate".to_string());
        }
        if self.max_age > 0 {
            parts.push(format!("max-age={}", self.max_age));
        }

        parts.join(", ")
    }
}

impl EdgeNode {
    /// Create a new edge node
    pub fn new(id: String, location: String, region: CdnRegion) -> Self {
        Self {
            id,
            location,
            region,
            status: EdgeNodeStatus::Active,
            load: 0,
            latency_ms: 0,
            bandwidth_mbps: 1000, // Default 1 Gbps
        }
    }

    /// Check if node is healthy
    pub fn is_healthy(&self) -> bool {
        matches!(self.status, EdgeNodeStatus::Active) && self.load < 90
    }

    /// Check if node is available
    pub fn is_available(&self) -> bool {
        matches!(
            self.status,
            EdgeNodeStatus::Active | EdgeNodeStatus::Degraded
        )
    }

    /// Calculate node score for selection
    pub fn calculate_score(&self) -> f64 {
        if !self.is_available() {
            return 0.0;
        }

        // Lower is better for latency and load
        let latency_score = 1.0 / (1.0 + self.latency_ms as f64 / 100.0);
        let load_score = 1.0 - (self.load as f64 / 100.0);
        let bandwidth_score = (self.bandwidth_mbps as f64).min(10000.0) / 10000.0;

        // Weighted average: latency 40%, load 40%, bandwidth 20%
        (latency_score * 0.4) + (load_score * 0.4) + (bandwidth_score * 0.2)
    }
}

impl Default for CdnStatistics {
    fn default() -> Self {
        Self::new()
    }
}

impl CdnStatistics {
    /// Create new CDN statistics
    pub fn new() -> Self {
        Self {
            total_requests: 0,
            cache_hit_rate: 0.0,
            bytes_transferred: 0,
            avg_response_ms: 0.0,
            requests_by_region: HashMap::new(),
            error_rate: 0.0,
        }
    }

    /// Record a request
    pub fn record_request(&mut self, region: &str, bytes: u64, response_ms: u64, cache_hit: bool) {
        self.total_requests += 1;
        self.bytes_transferred += bytes;

        // Update cache hit rate (moving average)
        let hit_value = if cache_hit { 1.0 } else { 0.0 };
        self.cache_hit_rate = (self.cache_hit_rate * (self.total_requests - 1) as f64 + hit_value)
            / self.total_requests as f64;

        // Update average response time
        self.avg_response_ms = (self.avg_response_ms * (self.total_requests - 1) as f64
            + response_ms as f64)
            / self.total_requests as f64;

        // Update region statistics
        *self
            .requests_by_region
            .entry(region.to_string())
            .or_insert(0) += 1;
    }

    /// Record an error
    pub fn record_error(&mut self) {
        self.total_requests += 1;
        self.error_rate =
            (self.error_rate * (self.total_requests - 1) as f64 + 1.0) / self.total_requests as f64;
    }
}

impl Default for CdnManager {
    fn default() -> Self {
        Self::new(CdnConfig::default())
    }
}

impl CdnManager {
    /// Create a new CDN manager
    pub fn new(config: CdnConfig) -> Self {
        Self {
            config,
            edge_nodes: Vec::new(),
            statistics: CdnStatistics::new(),
            cache: HashMap::new(),
        }
    }

    /// Add an edge node
    pub fn add_edge_node(&mut self, node: EdgeNode) {
        self.edge_nodes.push(node);
    }

    /// Get best edge node for a region
    pub fn get_best_node(&self, region: &CdnRegion) -> Option<&EdgeNode> {
        let mut candidates: Vec<_> = self
            .edge_nodes
            .iter()
            .filter(|n| n.is_available() && &n.region == region)
            .collect();

        if candidates.is_empty() {
            // Fall back to any available node
            candidates = self
                .edge_nodes
                .iter()
                .filter(|n| n.is_available())
                .collect();
        }

        candidates
            .iter()
            .max_by(|a, b| {
                a.calculate_score()
                    .partial_cmp(&b.calculate_score())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .copied()
    }

    /// Upload package to CDN
    pub fn upload_package(
        &mut self,
        package_name: &str,
        version: &str,
        _data: &[u8],
    ) -> Result<String> {
        let cache_key = format!("{}/{}", package_name, version);

        // Generate CDN URL
        let url = format!(
            "{}/packages/{}/{}",
            self.config.endpoint, package_name, version
        );

        // In production, would upload to CDN
        // For now, create cache entry
        let cache_item = CachedItem {
            _key: cache_key.clone(),
            url: url.clone(),
            expires_at: SystemTime::now() + Duration::from_secs(self.config.cache_ttl),
            _size: _data.len() as u64,
            hits: 0,
        };

        self.cache.insert(cache_key, cache_item);

        Ok(url)
    }

    /// Get package URL from CDN
    pub fn get_package_url(&mut self, package_name: &str, version: &str) -> Option<String> {
        let cache_key = format!("{}/{}", package_name, version);

        if let Some(item) = self.cache.get_mut(&cache_key) {
            // Check if cache entry is still valid
            if SystemTime::now() < item.expires_at {
                item.hits += 1;
                return Some(item.url.clone());
            } else {
                // Cache expired
                self.cache.remove(&cache_key);
            }
        }

        None
    }

    /// Purge cache for a package
    pub fn purge_cache(&mut self, package_name: &str, version: Option<&str>) -> Result<()> {
        if let Some(ver) = version {
            // Purge specific version
            let cache_key = format!("{}/{}", package_name, ver);
            self.cache.remove(&cache_key);
        } else {
            // Purge all versions
            let prefix = format!("{}/", package_name);
            self.cache.retain(|k, _| !k.starts_with(&prefix));
        }

        Ok(())
    }

    /// Get CDN statistics
    pub fn get_statistics(&self) -> &CdnStatistics {
        &self.statistics
    }

    /// Get cache hit rate
    pub fn get_cache_hit_rate(&self) -> f64 {
        self.statistics.cache_hit_rate
    }

    /// Get healthy edge nodes
    pub fn get_healthy_nodes(&self) -> Vec<&EdgeNode> {
        self.edge_nodes.iter().filter(|n| n.is_healthy()).collect()
    }

    /// Get edge nodes by region
    pub fn get_nodes_by_region(&self, region: &CdnRegion) -> Vec<&EdgeNode> {
        self.edge_nodes
            .iter()
            .filter(|n| &n.region == region)
            .collect()
    }

    /// Generate cache control header
    pub fn generate_cache_control(&self, package_version: &str) -> String {
        // Use immutable cache for versioned packages
        if !package_version.is_empty() {
            CacheControl::immutable().to_header()
        } else {
            CacheControl::default().to_header()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cdn_config() {
        let config = CdnConfig::new(
            CdnProvider::Cloudflare,
            "https://cdn.example.com".to_string(),
        )
        .with_cache_ttl(3600)
        .with_edge_compression(true)
        .add_region(CdnRegion::NorthAmerica);

        assert_eq!(config.provider, CdnProvider::Cloudflare);
        assert_eq!(config.cache_ttl, 3600);
        assert!(config.edge_compression);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_cache_control_headers() {
        let immutable = CacheControl::immutable();
        assert!(immutable.to_header().contains("public"));
        assert!(immutable.to_header().contains("max-age=31536000"));

        let no_cache = CacheControl::no_cache();
        assert!(no_cache.to_header().contains("no-cache"));
        assert!(no_cache.to_header().contains("no-store"));
    }

    #[test]
    fn test_edge_node_scoring() {
        let node = EdgeNode {
            id: "edge1".to_string(),
            location: "New York".to_string(),
            region: CdnRegion::NorthAmerica,
            status: EdgeNodeStatus::Active,
            load: 50,
            latency_ms: 50,
            bandwidth_mbps: 1000,
        };

        let score = node.calculate_score();
        assert!(score > 0.0 && score <= 1.0);
        assert!(node.is_healthy());
        assert!(node.is_available());
    }

    #[test]
    fn test_cdn_manager() {
        let mut manager = CdnManager::new(CdnConfig::default());

        let node = EdgeNode::new("edge1".to_string(), "London".to_string(), CdnRegion::Europe);
        manager.add_edge_node(node);

        let best = manager.get_best_node(&CdnRegion::Europe);
        assert!(best.is_some());
        assert_eq!(best.unwrap().id, "edge1");
    }

    #[test]
    fn test_package_upload() {
        let mut manager = CdnManager::new(CdnConfig::default());

        let data = b"package data";
        let url = manager
            .upload_package("test-package", "1.0.0", data)
            .unwrap();

        assert!(url.contains("test-package"));
        assert!(url.contains("1.0.0"));

        let retrieved_url = manager.get_package_url("test-package", "1.0.0");
        assert_eq!(retrieved_url, Some(url));
    }

    #[test]
    fn test_cache_purge() {
        let mut manager = CdnManager::new(CdnConfig::default());

        manager.upload_package("pkg1", "1.0.0", b"data1").unwrap();
        manager.upload_package("pkg1", "2.0.0", b"data2").unwrap();

        // Purge specific version
        manager.purge_cache("pkg1", Some("1.0.0")).unwrap();
        assert!(manager.get_package_url("pkg1", "1.0.0").is_none());
        assert!(manager.get_package_url("pkg1", "2.0.0").is_some());

        // Purge all versions
        manager.purge_cache("pkg1", None).unwrap();
        assert!(manager.get_package_url("pkg1", "2.0.0").is_none());
    }

    #[test]
    fn test_cdn_statistics() {
        let mut stats = CdnStatistics::new();

        stats.record_request("us-east", 1000, 50, true);
        stats.record_request("us-east", 2000, 100, false);

        assert_eq!(stats.total_requests, 2);
        assert_eq!(stats.cache_hit_rate, 0.5);
        assert_eq!(stats.avg_response_ms, 75.0);
        assert_eq!(stats.bytes_transferred, 3000);
    }
}
