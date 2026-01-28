//! Mirror management for high availability package distribution
//!
//! This module provides mirror server management with automatic failover,
//! health checking, and geographic distribution for package availability.

use std::collections::HashMap;
use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};
use torsh_core::error::{Result, TorshError};

/// Mirror server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MirrorConfig {
    /// Mirror ID
    pub id: String,
    /// Mirror URL
    pub url: String,
    /// Geographic region
    pub region: String,
    /// Priority (lower is higher priority)
    pub priority: u32,
    /// Weight for load balancing (higher gets more traffic)
    pub weight: u32,
    /// Health check interval in seconds
    pub health_check_interval: u64,
    /// Timeout for health checks in seconds
    pub health_check_timeout: u64,
    /// Maximum concurrent connections
    pub max_connections: u32,
}

/// Mirror health status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MirrorHealth {
    /// Mirror is healthy and responding
    Healthy,
    /// Mirror is degraded but usable
    Degraded,
    /// Mirror is unhealthy
    Unhealthy,
    /// Mirror status is unknown
    Unknown,
}

/// Mirror synchronization status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncStatus {
    /// Last successful sync time
    pub last_sync: Option<SystemTime>,
    /// Packages synced
    pub packages_synced: u64,
    /// Bytes synced
    pub bytes_synced: u64,
    /// Sync in progress
    pub syncing: bool,
    /// Sync errors
    pub sync_errors: u32,
}

/// Mirror selection strategy
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SelectionStrategy {
    /// Use closest mirror by geographic region
    Geographic,
    /// Use least loaded mirror
    LeastLoaded,
    /// Use round-robin selection
    RoundRobin,
    /// Use weighted random selection
    WeightedRandom,
    /// Use priority-based selection
    Priority,
}

/// Mirror metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mirror {
    /// Mirror configuration
    pub config: MirrorConfig,
    /// Current health status
    pub health: MirrorHealth,
    /// Last health check time
    pub last_health_check: Option<SystemTime>,
    /// Current load (0-100)
    pub load: u8,
    /// Synchronization status
    pub sync_status: SyncStatus,
    /// Available packages count
    pub package_count: u64,
    /// Total storage used in bytes
    pub storage_used: u64,
}

/// Mirror manager for coordinating multiple mirrors
pub struct MirrorManager {
    /// All configured mirrors
    mirrors: HashMap<String, Mirror>,
    /// Selection strategy
    strategy: SelectionStrategy,
    /// Round-robin counter
    round_robin_index: usize,
    /// Failover configuration
    failover_config: FailoverConfig,
    /// Mirror statistics
    statistics: MirrorStatistics,
}

/// Failover configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailoverConfig {
    /// Enable automatic failover
    pub enabled: bool,
    /// Number of retries before failover
    pub max_retries: u32,
    /// Retry delay in milliseconds
    pub retry_delay_ms: u64,
    /// Failback after successful health check
    pub auto_failback: bool,
    /// Minimum mirrors that must be healthy
    pub min_healthy_mirrors: usize,
}

/// Mirror statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MirrorStatistics {
    /// Total requests handled
    pub total_requests: u64,
    /// Requests by mirror
    pub requests_by_mirror: HashMap<String, u64>,
    /// Failed requests
    pub failed_requests: u64,
    /// Failover count
    pub failover_count: u64,
    /// Average response time by mirror (ms)
    pub avg_response_time: HashMap<String, f64>,
}

/// Mirror selection result
#[derive(Debug)]
pub struct MirrorSelection {
    /// Selected mirror
    pub mirror: Mirror,
    /// Fallback mirrors (in order of preference)
    pub fallbacks: Vec<Mirror>,
}

impl MirrorConfig {
    /// Create a new mirror configuration
    pub fn new(id: String, url: String, region: String) -> Self {
        Self {
            id,
            url,
            region,
            priority: 100,
            weight: 100,
            health_check_interval: 60,
            health_check_timeout: 10,
            max_connections: 100,
        }
    }

    /// Set priority
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    /// Set weight
    pub fn with_weight(mut self, weight: u32) -> Self {
        self.weight = weight;
        self
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if self.id.is_empty() {
            return Err(TorshError::InvalidArgument(
                "Mirror ID cannot be empty".to_string(),
            ));
        }

        if self.url.is_empty() {
            return Err(TorshError::InvalidArgument(
                "Mirror URL cannot be empty".to_string(),
            ));
        }

        if self.max_connections == 0 {
            return Err(TorshError::InvalidArgument(
                "Max connections must be greater than zero".to_string(),
            ));
        }

        Ok(())
    }
}

impl Default for SyncStatus {
    fn default() -> Self {
        Self::new()
    }
}

impl SyncStatus {
    /// Create new sync status
    pub fn new() -> Self {
        Self {
            last_sync: None,
            packages_synced: 0,
            bytes_synced: 0,
            syncing: false,
            sync_errors: 0,
        }
    }

    /// Start sync
    pub fn start_sync(&mut self) {
        self.syncing = true;
    }

    /// Complete sync
    pub fn complete_sync(&mut self, packages: u64, bytes: u64) {
        self.last_sync = Some(SystemTime::now());
        self.packages_synced = packages;
        self.bytes_synced = bytes;
        self.syncing = false;
    }

    /// Record sync error
    pub fn record_error(&mut self) {
        self.sync_errors += 1;
        self.syncing = false;
    }

    /// Check if sync is outdated (older than 24 hours)
    pub fn is_outdated(&self) -> bool {
        match self.last_sync {
            Some(last_sync) => {
                SystemTime::now()
                    .duration_since(last_sync)
                    .unwrap_or(Duration::from_secs(0))
                    > Duration::from_secs(86400)
            }
            None => true,
        }
    }
}

impl Mirror {
    /// Create a new mirror
    pub fn new(config: MirrorConfig) -> Self {
        Self {
            config,
            health: MirrorHealth::Unknown,
            last_health_check: None,
            load: 0,
            sync_status: SyncStatus::new(),
            package_count: 0,
            storage_used: 0,
        }
    }

    /// Check if mirror is available
    pub fn is_available(&self) -> bool {
        matches!(self.health, MirrorHealth::Healthy | MirrorHealth::Degraded)
    }

    /// Check if mirror is healthy
    pub fn is_healthy(&self) -> bool {
        self.health == MirrorHealth::Healthy
    }

    /// Calculate mirror score for selection
    pub fn calculate_score(&self) -> f64 {
        if !self.is_available() {
            return 0.0;
        }

        // Lower priority is better (lower number = higher priority)
        let priority_score = 1.0 / (1.0 + self.config.priority as f64 / 100.0);

        // Lower load is better
        let load_score = 1.0 - (self.load as f64 / 100.0);

        // Higher weight is better
        let weight_score = self.config.weight as f64 / 100.0;

        // Degraded health reduces score
        let health_score = match self.health {
            MirrorHealth::Healthy => 1.0,
            MirrorHealth::Degraded => 0.5,
            _ => 0.0,
        };

        // Weighted average
        priority_score * 0.3 + load_score * 0.3 + weight_score * 0.2 + health_score * 0.2
    }

    /// Update health status
    pub fn update_health(&mut self, health: MirrorHealth) {
        self.health = health;
        self.last_health_check = Some(SystemTime::now());
    }

    /// Update load
    pub fn update_load(&mut self, load: u8) {
        self.load = load.min(100);
    }
}

impl Default for FailoverConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_retries: 3,
            retry_delay_ms: 1000,
            auto_failback: true,
            min_healthy_mirrors: 1,
        }
    }
}

impl Default for MirrorStatistics {
    fn default() -> Self {
        Self::new()
    }
}

impl MirrorStatistics {
    /// Create new statistics
    pub fn new() -> Self {
        Self {
            total_requests: 0,
            requests_by_mirror: HashMap::new(),
            failed_requests: 0,
            failover_count: 0,
            avg_response_time: HashMap::new(),
        }
    }

    /// Record a request
    pub fn record_request(&mut self, mirror_id: &str, response_time_ms: u64, success: bool) {
        self.total_requests += 1;

        if success {
            *self
                .requests_by_mirror
                .entry(mirror_id.to_string())
                .or_insert(0) += 1;

            // Update average response time
            let current_avg = self
                .avg_response_time
                .get(mirror_id)
                .copied()
                .unwrap_or(0.0);
            let count = self.requests_by_mirror.get(mirror_id).copied().unwrap_or(0) as f64;

            let new_avg = if count > 0.0 {
                (current_avg * (count - 1.0) + response_time_ms as f64) / count
            } else {
                response_time_ms as f64
            };

            self.avg_response_time
                .insert(mirror_id.to_string(), new_avg);
        } else {
            self.failed_requests += 1;
        }
    }

    /// Record a failover
    pub fn record_failover(&mut self) {
        self.failover_count += 1;
    }
}

impl Default for MirrorManager {
    fn default() -> Self {
        Self::new(SelectionStrategy::Geographic)
    }
}

impl MirrorManager {
    /// Create a new mirror manager
    pub fn new(strategy: SelectionStrategy) -> Self {
        Self {
            mirrors: HashMap::new(),
            strategy,
            round_robin_index: 0,
            failover_config: FailoverConfig::default(),
            statistics: MirrorStatistics::new(),
        }
    }

    /// Add a mirror
    pub fn add_mirror(&mut self, mirror: Mirror) -> Result<()> {
        mirror.config.validate()?;
        self.mirrors.insert(mirror.config.id.clone(), mirror);
        Ok(())
    }

    /// Remove a mirror
    pub fn remove_mirror(&mut self, mirror_id: &str) -> bool {
        self.mirrors.remove(mirror_id).is_some()
    }

    /// Get a mirror by ID
    pub fn get_mirror(&self, mirror_id: &str) -> Option<&Mirror> {
        self.mirrors.get(mirror_id)
    }

    /// Get all healthy mirrors
    pub fn get_healthy_mirrors(&self) -> Vec<&Mirror> {
        self.mirrors.values().filter(|m| m.is_healthy()).collect()
    }

    /// Get all available mirrors
    pub fn get_available_mirrors(&self) -> Vec<&Mirror> {
        self.mirrors.values().filter(|m| m.is_available()).collect()
    }

    /// Select best mirror based on strategy
    pub fn select_mirror(&mut self, region: Option<&str>) -> Option<MirrorSelection> {
        // Clone available mirrors to avoid borrow checker issues
        let available: Vec<Mirror> = self
            .get_available_mirrors()
            .iter()
            .map(|&m| m.clone())
            .collect();

        if available.is_empty() {
            return None;
        }

        // Create a temporary slice of references for selection
        let available_refs: Vec<&Mirror> = available.iter().collect();

        let selected = match self.strategy {
            SelectionStrategy::Geographic => self.select_by_geography(region, &available_refs),
            SelectionStrategy::LeastLoaded => self.select_least_loaded(&available_refs),
            SelectionStrategy::RoundRobin => self.select_round_robin(&available_refs),
            SelectionStrategy::WeightedRandom => self.select_weighted_random(&available_refs),
            SelectionStrategy::Priority => self.select_by_priority(&available_refs),
        }?;

        // Get fallback mirrors
        let mut fallbacks: Vec<_> = available
            .iter()
            .filter(|m| m.config.id != selected.config.id)
            .cloned()
            .collect();

        // Sort fallbacks by score
        fallbacks.sort_by(|a, b| {
            b.calculate_score()
                .partial_cmp(&a.calculate_score())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Some(MirrorSelection {
            mirror: selected.clone(),
            fallbacks,
        })
    }

    /// Select mirror by geographic proximity
    fn select_by_geography<'a>(
        &self,
        region: Option<&str>,
        mirrors: &[&'a Mirror],
    ) -> Option<&'a Mirror> {
        if let Some(r) = region {
            // Try to find mirror in same region
            let regional_mirrors: Vec<_> = mirrors
                .iter()
                .filter(|m| m.config.region == r)
                .copied()
                .collect();

            if !regional_mirrors.is_empty() {
                return regional_mirrors
                    .iter()
                    .max_by(|a, b| {
                        a.calculate_score()
                            .partial_cmp(&b.calculate_score())
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .copied();
            }
        }

        // Fall back to best overall mirror
        mirrors
            .iter()
            .max_by(|a, b| {
                a.calculate_score()
                    .partial_cmp(&b.calculate_score())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .copied()
    }

    /// Select least loaded mirror
    fn select_least_loaded<'a>(&self, mirrors: &[&'a Mirror]) -> Option<&'a Mirror> {
        mirrors.iter().min_by_key(|m| m.load).copied()
    }

    /// Select mirror using round-robin
    fn select_round_robin<'a>(&mut self, mirrors: &[&'a Mirror]) -> Option<&'a Mirror> {
        if mirrors.is_empty() {
            return None;
        }

        let selected = mirrors[self.round_robin_index % mirrors.len()];
        self.round_robin_index = (self.round_robin_index + 1) % mirrors.len();
        Some(selected)
    }

    /// Select mirror using weighted random
    fn select_weighted_random<'a>(&self, mirrors: &[&'a Mirror]) -> Option<&'a Mirror> {
        let total_weight: u32 = mirrors.iter().map(|m| m.config.weight).sum();
        if total_weight == 0 {
            return mirrors.first().copied();
        }

        // Simplified weighted random (in production, use proper RNG)
        let random_weight = (SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .expect("system time should be after UNIX_EPOCH")
            .as_millis() as u32)
            % total_weight;

        let mut cumulative = 0;
        for mirror in mirrors {
            cumulative += mirror.config.weight;
            if random_weight < cumulative {
                return Some(mirror);
            }
        }

        mirrors.last().copied()
    }

    /// Select mirror by priority
    fn select_by_priority<'a>(&self, mirrors: &[&'a Mirror]) -> Option<&'a Mirror> {
        mirrors.iter().min_by_key(|m| m.config.priority).copied()
    }

    /// Update mirror health
    pub fn update_mirror_health(&mut self, mirror_id: &str, health: MirrorHealth) -> Result<()> {
        let mirror = self
            .mirrors
            .get_mut(mirror_id)
            .ok_or_else(|| TorshError::InvalidArgument("Mirror not found".to_string()))?;

        mirror.update_health(health);
        Ok(())
    }

    /// Get statistics
    pub fn get_statistics(&self) -> &MirrorStatistics {
        &self.statistics
    }

    /// Check if enough healthy mirrors are available
    pub fn has_sufficient_mirrors(&self) -> bool {
        let healthy_count = self.get_healthy_mirrors().len();
        healthy_count >= self.failover_config.min_healthy_mirrors
    }

    /// Get failover configuration
    pub fn get_failover_config(&self) -> &FailoverConfig {
        &self.failover_config
    }

    /// Set failover configuration
    pub fn set_failover_config(&mut self, config: FailoverConfig) {
        self.failover_config = config;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_mirror(id: &str, region: &str, priority: u32) -> Mirror {
        let config = MirrorConfig::new(
            id.to_string(),
            format!("https://{}.example.com", id),
            region.to_string(),
        )
        .with_priority(priority);

        let mut mirror = Mirror::new(config);
        mirror.update_health(MirrorHealth::Healthy);
        mirror
    }

    #[test]
    fn test_mirror_config() {
        let config = MirrorConfig::new(
            "mirror1".to_string(),
            "https://mirror1.example.com".to_string(),
            "us-east".to_string(),
        )
        .with_priority(10)
        .with_weight(200);

        assert_eq!(config.priority, 10);
        assert_eq!(config.weight, 200);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_mirror_health() {
        let mut mirror = create_test_mirror("mirror1", "us-east", 10);

        assert!(mirror.is_healthy());
        assert!(mirror.is_available());

        mirror.update_health(MirrorHealth::Degraded);
        assert!(!mirror.is_healthy());
        assert!(mirror.is_available());

        mirror.update_health(MirrorHealth::Unhealthy);
        assert!(!mirror.is_available());
    }

    #[test]
    fn test_mirror_scoring() {
        let mirror = create_test_mirror("mirror1", "us-east", 10);
        let score = mirror.calculate_score();
        assert!(score > 0.0 && score <= 1.0);
    }

    #[test]
    fn test_sync_status() {
        let mut status = SyncStatus::new();
        assert!(status.is_outdated());

        status.start_sync();
        assert!(status.syncing);

        status.complete_sync(100, 1024 * 1024);
        assert!(!status.syncing);
        assert_eq!(status.packages_synced, 100);
        assert!(!status.is_outdated());
    }

    #[test]
    fn test_mirror_manager() {
        let mut manager = MirrorManager::new(SelectionStrategy::Priority);

        let mirror1 = create_test_mirror("mirror1", "us-east", 10);
        let mirror2 = create_test_mirror("mirror2", "us-west", 20);

        manager.add_mirror(mirror1).unwrap();
        manager.add_mirror(mirror2).unwrap();

        assert_eq!(manager.mirrors.len(), 2);
        assert!(manager.has_sufficient_mirrors());
    }

    #[test]
    fn test_geographic_selection() {
        let mut manager = MirrorManager::new(SelectionStrategy::Geographic);

        manager
            .add_mirror(create_test_mirror("us-mirror", "us-east", 10))
            .unwrap();
        manager
            .add_mirror(create_test_mirror("eu-mirror", "europe", 10))
            .unwrap();

        let selection = manager.select_mirror(Some("us-east"));
        assert!(selection.is_some());
        assert_eq!(selection.unwrap().mirror.config.id, "us-mirror");
    }

    #[test]
    fn test_priority_selection() {
        let mut manager = MirrorManager::new(SelectionStrategy::Priority);

        manager
            .add_mirror(create_test_mirror("high-priority", "us-east", 5))
            .unwrap();
        manager
            .add_mirror(create_test_mirror("low-priority", "us-east", 20))
            .unwrap();

        let selection = manager.select_mirror(None);
        assert!(selection.is_some());
        assert_eq!(selection.unwrap().mirror.config.id, "high-priority");
    }

    #[test]
    fn test_round_robin_selection() {
        let mut manager = MirrorManager::new(SelectionStrategy::RoundRobin);

        manager
            .add_mirror(create_test_mirror("mirror1", "us-east", 10))
            .unwrap();
        manager
            .add_mirror(create_test_mirror("mirror2", "us-east", 10))
            .unwrap();

        let sel1 = manager.select_mirror(None).unwrap();
        let sel2 = manager.select_mirror(None).unwrap();

        // Should select different mirrors in round-robin
        assert_ne!(sel1.mirror.config.id, sel2.mirror.config.id);
    }

    #[test]
    fn test_mirror_statistics() {
        let mut stats = MirrorStatistics::new();

        stats.record_request("mirror1", 100, true);
        stats.record_request("mirror1", 200, true);
        stats.record_request("mirror2", 150, false);

        assert_eq!(stats.total_requests, 3);
        assert_eq!(*stats.requests_by_mirror.get("mirror1").unwrap(), 2);
        assert_eq!(stats.failed_requests, 1);
        assert_eq!(*stats.avg_response_time.get("mirror1").unwrap(), 150.0);
    }
}
