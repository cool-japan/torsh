//! Mirror Manager Orchestration
//!
//! This module provides the main MirrorManager implementation that orchestrates
//! all mirror management components including selection algorithms, performance
//! analysis, geographic optimization, and download coordination.

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use super::selection::{validate_selection_strategy, MirrorSelector};
use super::types::*;
use super::types::{GeographicCalculator, PerformanceAnalyzer};
use crate::download::validation::validate_url;
use reqwest::Client;
use std::path::Path;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::io::AsyncWriteExt;
use torsh_core::error::{Result, TorshError};

// ================================================================================================
// Mirror Manager Implementation
// ================================================================================================

/// Advanced mirror manager for handling sophisticated mirror selection and failover
///
/// This manager provides enterprise-grade mirror functionality with intelligent
/// selection algorithms, real-time health monitoring, geographic optimization,
/// and comprehensive performance analytics.
///
/// # Features
/// - Sophisticated mirror selection with multiple strategies
/// - Real-time health monitoring and benchmarking
/// - Geographic proximity calculations and optimization
/// - Load balancing and capacity management
/// - Performance analytics and trend analysis
/// - Automatic failover with intelligent retry logic
/// - Statistics collection and monitoring
///
/// # Examples
/// ```rust
/// use torsh_hub::download::mirror::{MirrorManager, MirrorConfig};
/// use std::path::Path;
///
/// # tokio_test::block_on(async {
/// let config = MirrorConfig::default();
/// let mut manager = MirrorManager::new(config).unwrap();
///
/// // Download with automatic mirror selection and failover
/// let result = manager.download_with_mirrors(
///     "models/bert-base-uncased.torsh",
///     Path::new("/tmp/model.torsh"),
///     true
/// ).await;
/// # });
/// ```
pub struct MirrorManager {
    config: MirrorConfig,
    client: Client,
    selection_state: MirrorSelectionState,
    performance_analyzer: PerformanceAnalyzer,
    geographic_calculator: GeographicCalculator,
}

impl MirrorManager {
    /// Create a new advanced mirror manager with comprehensive configuration
    ///
    /// # Arguments
    /// * `config` - Mirror configuration with selection strategies and settings
    ///
    /// # Returns
    /// * `Ok(MirrorManager)` - Successfully created manager
    /// * `Err(TorshError)` - Configuration error or initialization failure
    pub fn new(config: MirrorConfig) -> Result<Self> {
        // Validate the configuration
        if config.mirrors.is_empty() {
            return Err(TorshError::config_error_with_context(
                "Mirror configuration must contain at least one mirror",
                "Mirror validation",
            ));
        }
        validate_selection_strategy(&config.selection_strategy)?;

        let client = Client::builder()
            .user_agent("torsh-hub/0.1.0-alpha.2")
            .timeout(config.connection_timeout)
            .build()
            .map_err(|e| TorshError::IoError(e.to_string()))?;

        let mut geographic_calculator = GeographicCalculator::new();
        if !config.enable_geographic_optimization {
            geographic_calculator.set_enabled(false);
        }

        Ok(Self {
            config,
            client,
            selection_state: MirrorSelectionState::new(),
            performance_analyzer: PerformanceAnalyzer::new(),
            geographic_calculator,
        })
    }

    /// Create a new mirror manager with custom components
    ///
    /// This allows for dependency injection and custom configuration of individual components.
    pub fn with_components(
        config: MirrorConfig,
        client: Client,
        selection_state: MirrorSelectionState,
        performance_analyzer: PerformanceAnalyzer,
        geographic_calculator: GeographicCalculator,
    ) -> Result<Self> {
        validate_selection_strategy(&config.selection_strategy)?;

        Ok(Self {
            config,
            client,
            selection_state,
            performance_analyzer,
            geographic_calculator,
        })
    }

    /// Download a file using sophisticated mirror selection with automatic failover
    ///
    /// This function implements advanced mirror selection algorithms with intelligent
    /// failover, performance tracking, and geographic optimization.
    ///
    /// # Arguments
    /// * `file_path` - Relative path to the file on mirror servers
    /// * `dest_path` - Local destination path for the downloaded file
    /// * `progress` - Whether to display download progress and mirror information
    ///
    /// # Returns
    /// * `Ok(MirrorDownloadResult)` - Download result with comprehensive metrics
    /// * `Err(TorshError)` - Download failure after all mirrors exhausted
    pub async fn download_with_mirrors(
        &mut self,
        file_path: &str,
        dest_path: &Path,
        progress: bool,
    ) -> Result<MirrorDownloadResult> {
        // Validate inputs
        if file_path.trim().is_empty() {
            return Err(TorshError::IoError("File path cannot be empty".to_string()));
        }

        // Store config values before mutable operations
        let max_mirror_attempts = self.config.max_mirror_attempts;

        // Benchmark mirrors if needed for optimal selection
        if self.should_benchmark().await {
            if progress {
                println!("Benchmarking mirrors for optimal selection...");
            }
            self.benchmark_mirrors().await?;
        }

        // Update user location for geographic optimization if enabled
        if self.config.enable_geographic_optimization {
            self.update_user_location().await;
        }

        let selected_mirrors = self.select_mirrors().await?;
        let mut attempts = Vec::new();
        let start_time = Instant::now();

        if progress {
            println!(
                "Starting download with {} mirror(s) available, strategy: {:?}",
                selected_mirrors.len(),
                self.config.selection_strategy
            );
        }

        for (attempt_num, mirror) in selected_mirrors
            .iter()
            .take(max_mirror_attempts)
            .enumerate()
        {
            let attempt_start = SystemTime::now();
            let url = format!(
                "{}/{}",
                mirror.base_url.trim_end_matches('/'),
                file_path.trim_start_matches('/')
            );

            // Validate URL before attempting download
            if let Err(e) = validate_url(&url) {
                if progress {
                    println!("Invalid mirror URL: {} - {}", url, e);
                }
                continue;
            }

            if progress {
                println!(
                    "Attempt {}/{}: Trying mirror {} ({}, {}) - {}ms avg",
                    attempt_num + 1,
                    max_mirror_attempts,
                    mirror.id,
                    mirror.location.city,
                    mirror.location.country,
                    mirror.avg_response_time.unwrap_or(0)
                );
            }

            match self
                .download_from_mirror(&url, dest_path, mirror, progress)
                .await
            {
                Ok(download_metrics) => {
                    let elapsed = SystemTime::now()
                        .duration_since(attempt_start)
                        .unwrap_or_default();
                    let attempt = MirrorAttempt {
                        mirror_id: mirror.id.clone(),
                        mirror_url: mirror.base_url.clone(),
                        start_time: attempt_start,
                        success: true,
                        duration: elapsed,
                        error_message: None,
                        response_time: None, // Will be filled if available from download_metrics
                        throughput: download_metrics.throughput,
                        bytes_downloaded: download_metrics.bytes_downloaded,
                    };
                    attempts.push(attempt);

                    // Update mirror statistics and performance history
                    self.update_mirror_success(&mirror.id, elapsed, &download_metrics)
                        .await;

                    // Record successful selection for adaptive learning
                    self.record_selection(&mirror.id, true, elapsed).await;

                    if progress {
                        println!(
                            "✓ Successfully downloaded from mirror: {} ({:.2}s, {:.1} MB/s)",
                            mirror.id,
                            elapsed.as_secs_f64(),
                            download_metrics.throughput.unwrap_or(0.0)
                        );
                    }

                    return Ok(MirrorDownloadResult {
                        success: true,
                        total_duration: start_time.elapsed(),
                        successful_mirror: Some(mirror.id.clone()),
                        attempts,
                        total_bytes_downloaded: download_metrics.bytes_downloaded,
                        average_throughput: download_metrics.throughput,
                        mirror_selection_strategy: self.config.selection_strategy.clone(),
                    });
                }
                Err(e) => {
                    let elapsed = SystemTime::now()
                        .duration_since(attempt_start)
                        .unwrap_or_default();
                    let attempt = MirrorAttempt {
                        mirror_id: mirror.id.clone(),
                        mirror_url: mirror.base_url.clone(),
                        start_time: attempt_start,
                        success: false,
                        duration: elapsed,
                        error_message: Some(e.to_string()),
                        response_time: None,
                        throughput: None,
                        bytes_downloaded: 0,
                    };
                    attempts.push(attempt);

                    // Update mirror failure statistics
                    self.update_mirror_failure(&mirror.id).await;

                    // Record failed selection for adaptive learning
                    self.record_selection(&mirror.id, false, elapsed).await;

                    if progress {
                        println!(
                            "✗ Failed to download from mirror: {} ({:.2}s) - {}",
                            mirror.id,
                            elapsed.as_secs_f64(),
                            e
                        );
                    }
                }
            }
        }

        Ok(MirrorDownloadResult {
            success: false,
            total_duration: start_time.elapsed(),
            successful_mirror: None,
            attempts,
            total_bytes_downloaded: 0,
            average_throughput: None,
            mirror_selection_strategy: self.config.selection_strategy.clone(),
        })
    }

    /// Select mirrors using the configured strategy
    async fn select_mirrors(&mut self) -> Result<Vec<MirrorServer>> {
        let mut selector = MirrorSelector::new(
            &self.config,
            &mut self.selection_state,
            &self.geographic_calculator,
            &self.performance_analyzer,
        );

        selector.select_mirrors()
    }

    /// Download from a specific mirror with advanced metrics collection
    async fn download_from_mirror(
        &self,
        url: &str,
        dest_path: &Path,
        _mirror: &MirrorServer,
        progress: bool,
    ) -> Result<DownloadMetrics> {
        let start_time = Instant::now();

        // Make HTTP request
        let response = self
            .client
            .get(url)
            .send()
            .await
            .map_err(|e| TorshError::IoError(e.to_string()))?;

        if !response.status().is_success() {
            return Err(TorshError::IoError(format!(
                "HTTP error: {}",
                response.status()
            )));
        }

        let total_size = response
            .content_length()
            .ok_or_else(|| TorshError::IoError("No content length header".to_string()))?;

        // Create destination file
        let mut file = tokio::fs::File::create(dest_path)
            .await
            .map_err(|e| TorshError::IoError(e.to_string()))?;

        // Download with progress tracking
        let mut downloaded = 0u64;
        let mut stream = response.bytes_stream();
        let mut throughput_samples = Vec::new();
        let mut last_progress_time = start_time;

        use futures_util::StreamExt;

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| TorshError::IoError(e.to_string()))?;

            file.write_all(&chunk)
                .await
                .map_err(|e| TorshError::IoError(e.to_string()))?;

            downloaded += chunk.len() as u64;

            // Update progress and calculate throughput
            let now = Instant::now();
            if progress && now.duration_since(last_progress_time) >= Duration::from_millis(500) {
                let elapsed = start_time.elapsed().as_secs_f64();
                if elapsed > 0.0 {
                    let throughput = (downloaded as f64) / elapsed / 1_048_576.0; // MB/s
                    throughput_samples.push(throughput);

                    let percentage = (downloaded as f64 / total_size as f64) * 100.0;
                    println!(
                        "Download progress: {:.1}% ({:.1} MB/s)",
                        percentage, throughput
                    );
                }
                last_progress_time = now;
            }
        }

        file.flush()
            .await
            .map_err(|e| TorshError::IoError(e.to_string()))?;

        let total_duration = start_time.elapsed();

        // Calculate average throughput
        let average_throughput = if !throughput_samples.is_empty() {
            Some(throughput_samples.iter().sum::<f64>() / throughput_samples.len() as f64)
        } else if total_duration.as_secs_f64() > 0.0 {
            Some((downloaded as f64) / total_duration.as_secs_f64() / 1_048_576.0)
        } else {
            None
        };

        Ok(DownloadMetrics {
            bytes_downloaded: downloaded,
            throughput: average_throughput,
            duration: total_duration,
        })
    }

    /// Comprehensive mirror benchmarking with detailed performance analysis
    pub async fn benchmark_mirrors(&mut self) -> Result<Vec<MirrorBenchmarkResult>> {
        let mut results = Vec::new();
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be after UNIX epoch")
            .as_secs();

        for mirror in &mut self.config.mirrors {
            if !mirror.active {
                continue;
            }

            let benchmark_result = self
                .performance_analyzer
                .benchmark_single_mirror(mirror, &self.client, current_time)
                .await;

            results.push(benchmark_result.clone());
        }

        self.selection_state.last_benchmark = current_time;
        Ok(results)
    }

    /// Check if mirrors should be benchmarked
    async fn should_benchmark(&self) -> bool {
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be after UNIX epoch")
            .as_secs();

        current_time - self.selection_state.last_benchmark > self.config.benchmark_interval
    }

    /// Update user location for geographic optimization
    async fn update_user_location(&mut self) {
        if self.config.enable_geographic_optimization {
            self.geographic_calculator.estimate_user_location();
        }
    }

    /// Update mirror statistics after successful download with comprehensive metrics
    async fn update_mirror_success(
        &mut self,
        mirror_id: &str,
        duration: Duration,
        metrics: &DownloadMetrics,
    ) {
        if let Some(mirror) = self.config.mirrors.iter_mut().find(|m| m.id == mirror_id) {
            let response_time = duration.as_millis() as u64;

            // Update average response time with exponential moving average
            match mirror.avg_response_time {
                Some(avg) => {
                    mirror.avg_response_time = Some((avg * 7 + response_time) / 8);
                }
                None => {
                    mirror.avg_response_time = Some(response_time);
                }
            }

            mirror.consecutive_failures = 0;

            // Update last successful connection timestamp
            let current_time = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system time should be after UNIX epoch")
                .as_secs();
            mirror.last_successful_connection = Some(current_time);

            // Increase reliability score for successful downloads
            mirror.reliability_score = (mirror.reliability_score + 0.01).min(1.0);

            // Add performance snapshot
            let snapshot = PerformanceSnapshot {
                timestamp: current_time,
                response_time,
                throughput: metrics.throughput,
                error_rate: 0.0,
                load_percentage: mirror.capacity.current_load.unwrap_or(0.0),
            };
            mirror.performance_history.push(snapshot);

            // Limit performance history size
            if mirror.performance_history.len() > 1000 {
                mirror.performance_history.drain(0..100); // Remove oldest 100 entries
            }
        }
    }

    /// Update mirror statistics after failed download
    async fn update_mirror_failure(&mut self, mirror_id: &str) {
        if let Some(mirror) = self.config.mirrors.iter_mut().find(|m| m.id == mirror_id) {
            mirror.consecutive_failures += 1;

            // Decrease reliability score for failures
            mirror.reliability_score = (mirror.reliability_score - 0.05).max(0.0);

            // Deactivate mirror if too many consecutive failures
            if mirror.consecutive_failures >= 5 {
                mirror.active = false;
            }

            // Add failure snapshot
            let current_time = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system time should be after UNIX epoch")
                .as_secs();
            let snapshot = PerformanceSnapshot {
                timestamp: current_time,
                response_time: 0, // Failed request
                throughput: None,
                error_rate: 100.0,
                load_percentage: mirror.capacity.current_load.unwrap_or(0.0),
            };
            mirror.performance_history.push(snapshot);

            // Limit performance history size
            if mirror.performance_history.len() > 1000 {
                mirror.performance_history.drain(0..100);
            }
        }
    }

    /// Record selection decision for adaptive learning
    async fn record_selection(&mut self, mirror_id: &str, success: bool, duration: Duration) {
        let performance_score = if success {
            duration.as_millis() as f64
        } else {
            f64::INFINITY
        };

        self.selection_state.record_selection(
            mirror_id,
            self.config.selection_strategy.clone(),
            success,
            performance_score,
        );
    }

    /// Get comprehensive mirror statistics
    pub fn get_mirror_statistics(&self) -> Vec<MirrorStatistics> {
        self.config
            .mirrors
            .iter()
            .map(|mirror| {
                // Calculate average recent throughput from last 10 performance snapshots
                let _avg_recent_throughput = if mirror.performance_history.len() >= 3 {
                    let recent_throughputs: Vec<f64> = mirror
                        .performance_history
                        .iter()
                        .rev()
                        .take(10)
                        .filter_map(|s| s.throughput)
                        .collect();
                    if !recent_throughputs.is_empty() {
                        Some(
                            recent_throughputs.iter().sum::<f64>()
                                / recent_throughputs.len() as f64,
                        )
                    } else {
                        None
                    }
                } else {
                    None
                };

                MirrorStatistics {
                    mirror_id: mirror.id.clone(),
                    mirror_url: mirror.base_url.clone(),
                    successful_downloads: 0, // Placeholder - no direct stats available
                    failed_downloads: mirror.consecutive_failures as u64,
                    success_rate: mirror.reliability_score * 100.0, // Convert to percentage
                    avg_response_time: mirror.avg_response_time.unwrap_or(0) as f64,
                    total_bytes_served: 0, // Placeholder - no direct stats available
                    reliability_score: mirror.reliability_score,
                    location: mirror.location.clone(),
                    performance_trend: self
                        .performance_analyzer
                        .calculate_performance_trend(mirror),
                    last_updated: chrono::Utc::now().timestamp() as u64,
                }
            })
            .collect()
    }

    /// Get selection statistics and performance metrics
    pub fn get_selection_statistics(&self) -> SelectionStatistics {
        self.selection_state.get_selection_statistics()
    }

    /// Add a new mirror to the configuration
    pub fn add_mirror(&mut self, mirror: MirrorServer) {
        self.config.mirrors.push(mirror);
    }

    /// Remove a mirror by ID
    pub fn remove_mirror(&mut self, mirror_id: &str) -> bool {
        let initial_len = self.config.mirrors.len();
        self.config.mirrors.retain(|m| m.id != mirror_id);
        self.config.mirrors.len() < initial_len
    }

    /// Update mirror configuration
    pub fn update_config(&mut self, config: MirrorConfig) -> Result<()> {
        validate_selection_strategy(&config.selection_strategy)?;
        self.config = config;
        Ok(())
    }

    /// Get current mirror selection strategy
    pub fn get_selection_strategy(&self) -> &MirrorSelectionStrategy {
        &self.config.selection_strategy
    }

    /// Update mirror selection strategy
    pub fn set_selection_strategy(&mut self, strategy: MirrorSelectionStrategy) -> Result<()> {
        validate_selection_strategy(&strategy)?;
        self.config.selection_strategy = strategy;
        Ok(())
    }

    /// Get reference to the configuration
    pub fn get_config(&self) -> &MirrorConfig {
        &self.config
    }

    /// Get mutable reference to performance analyzer
    pub fn get_performance_analyzer_mut(&mut self) -> &mut PerformanceAnalyzer {
        &mut self.performance_analyzer
    }

    /// Get reference to performance analyzer
    pub fn get_performance_analyzer(&self) -> &PerformanceAnalyzer {
        &self.performance_analyzer
    }

    /// Get mutable reference to geographic calculator
    pub fn get_geographic_calculator_mut(&mut self) -> &mut GeographicCalculator {
        &mut self.geographic_calculator
    }

    /// Get reference to geographic calculator
    pub fn get_geographic_calculator(&self) -> &GeographicCalculator {
        &self.geographic_calculator
    }

    /// Clear all cached data and reset state
    pub fn clear_cache(&mut self) {
        self.performance_analyzer.clear_cache();
        self.geographic_calculator.clear_cache();
        self.selection_state.clear_history();
    }

    /// Get comprehensive health status of all mirrors
    pub async fn get_mirror_health_status(&mut self) -> Vec<MirrorHealth> {
        let mut health_statuses = Vec::new();
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be after UNIX epoch")
            .as_secs();

        for mirror in &self.config.mirrors {
            let last_check_age = mirror
                .last_successful_connection
                .map(|time| current_time - time)
                .unwrap_or(u64::MAX);

            let health = if !mirror.active {
                MirrorHealthStatus::Offline
            } else if mirror.consecutive_failures >= 5 {
                MirrorHealthStatus::Unhealthy
            } else if mirror.consecutive_failures >= 2 || last_check_age > 3600 {
                MirrorHealthStatus::Degraded
            } else {
                MirrorHealthStatus::Healthy
            };

            health_statuses.push(MirrorHealth {
                id: mirror.id.clone(),
                status: health,
                last_check: mirror
                    .last_successful_connection
                    .map(|timestamp| std::time::UNIX_EPOCH + Duration::from_secs(timestamp))
                    .unwrap_or(std::time::SystemTime::now()),
                response_time: mirror.avg_response_time.map(Duration::from_millis),
                error_rate: (mirror.consecutive_failures as f32) / 10.0, // Convert to a reasonable error rate
                uptime_percentage: (mirror.reliability_score * 100.0) as f32,
            });
        }

        health_statuses
    }

    /// Detect specific issues with a mirror
    fn detect_mirror_issues(&self, mirror: &MirrorServer) -> Vec<String> {
        let mut issues = Vec::new();

        if !mirror.active {
            issues.push("Mirror is inactive".to_string());
        }

        if mirror.consecutive_failures >= 3 {
            issues.push(format!(
                "High failure rate: {} consecutive failures",
                mirror.consecutive_failures
            ));
        }

        if mirror.reliability_score < 0.7 {
            issues.push(format!(
                "Low reliability score: {:.2}",
                mirror.reliability_score
            ));
        }

        if let Some(response_time) = mirror.avg_response_time {
            if response_time > self.config.max_response_time {
                issues.push(format!("High latency: {}ms", response_time));
            }
        }

        if let Some(load) = mirror.capacity.current_load {
            if load > 90.0 {
                issues.push(format!("High load: {:.1}%", load));
            }
        }

        // Check for stale performance data
        if let Some(last_snapshot) = mirror.performance_history.last() {
            let current_time = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system time should be after UNIX epoch")
                .as_secs();

            if current_time - last_snapshot.timestamp > 7200 {
                // 2 hours
                issues.push("Stale performance data".to_string());
            }
        }

        issues
    }
}

/// Download metrics for performance tracking
#[derive(Debug, Clone)]
pub struct DownloadMetrics {
    pub bytes_downloaded: u64,
    pub throughput: Option<f64>, // MB/s
    pub duration: Duration,
}

// ================================================================================================
// Utility Functions
// ================================================================================================

/// Download a file using default mirror configuration
///
/// This is a convenience function for simple downloads without custom configuration.
pub async fn download_with_default_mirrors(
    file_path: &str,
    dest_path: &Path,
    progress: bool,
) -> Result<MirrorDownloadResult> {
    let config = MirrorConfig::default();
    let mut manager = MirrorManager::new(config)?;
    manager
        .download_with_mirrors(file_path, dest_path, progress)
        .await
}

/// Create optimized mirror configuration for a specific region
pub fn create_regional_mirror_config(region: &str) -> MirrorConfig {
    let mut config = MirrorConfig::default();

    match region {
        "us-east" => {
            config.selection_strategy = MirrorSelectionStrategy::Geographic;
            config.enable_geographic_optimization = true;
        }
        "eu-west" => {
            config.selection_strategy = MirrorSelectionStrategy::Geographic;
            config.enable_geographic_optimization = true;
        }
        "asia-pacific" => {
            config.selection_strategy = MirrorSelectionStrategy::Geographic;
            config.enable_geographic_optimization = true;
        }
        "global" => {
            config.selection_strategy = MirrorSelectionStrategy::Adaptive;
            config.enable_geographic_optimization = true;
        }
        _ => {
            // Default balanced configuration
            config.selection_strategy = MirrorSelectionStrategy::Weighted(MirrorWeights::default());
        }
    }

    config
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn create_test_config() -> MirrorConfig {
        MirrorConfig {
            mirrors: vec![MirrorServer {
                id: "test-mirror-1".to_string(),
                base_url: "https://test1.example.com".to_string(),
                reliability_score: 0.95,
                avg_response_time: Some(100),
                consecutive_failures: 0,
                location: MirrorLocation {
                    country: "US".to_string(),
                    region: "California".to_string(),
                    city: "San Francisco".to_string(),
                    latitude: Some(37.7749),
                    longitude: Some(-122.4194),
                    provider: "TestProvider1".to_string(),
                    timezone: Some("America/Los_Angeles".to_string()),
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
                    name: "TestProvider1".to_string(),
                    network_tier: Some("Premium".to_string()),
                    cdn_integration: true,
                    edge_location: Some("SFO".to_string()),
                    network_quality: NetworkQuality::default(),
                },
                performance_history: Vec::new(),
                last_successful_connection: None,
            }],
            selection_strategy: MirrorSelectionStrategy::LowestLatency,
            max_mirror_attempts: 3,
            connection_timeout: Duration::from_secs(10),
            enable_auto_discovery: false,
            benchmark_interval: 3600,
            enable_geographic_optimization: true,
            min_reliability_score: 0.7,
            max_response_time: 5000,
            load_balancing: LoadBalancingConfig::default(),
        }
    }

    #[test]
    fn test_mirror_manager_creation() {
        let config = create_test_config();
        let manager = MirrorManager::new(config);
        assert!(manager.is_ok());
    }

    #[test]
    fn test_mirror_manager_with_invalid_config() {
        let mut config = create_test_config();
        config.selection_strategy = MirrorSelectionStrategy::Weighted(MirrorWeights {
            latency: 0.5,
            reliability: 0.5,
            load: 0.5, // Sum > 1.0, should fail validation
            geographic: 0.0,
            bandwidth: 0.0,
            provider_quality: 0.0,
        });

        let manager = MirrorManager::new(config);
        assert!(manager.is_err());
    }

    #[test]
    fn test_add_remove_mirror() {
        let config = create_test_config();
        let mut manager = MirrorManager::new(config).unwrap();

        assert_eq!(manager.config.mirrors.len(), 1);

        // Add a new mirror
        let new_mirror = MirrorServer {
            id: "test-mirror-2".to_string(),
            base_url: "https://test2.example.com".to_string(),
            reliability_score: 0.85,
            avg_response_time: Some(150),
            consecutive_failures: 0,
            location: MirrorLocation {
                country: "US".to_string(),
                region: "New York".to_string(),
                city: "New York".to_string(),
                latitude: Some(40.7128),
                longitude: Some(-74.0060),
                provider: "TestProvider2".to_string(),
                timezone: Some("America/New_York".to_string()),
                datacenter: Some("test-dc-2".to_string()),
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
                name: "TestProvider2".to_string(),
                network_tier: Some("Standard".to_string()),
                cdn_integration: false,
                edge_location: Some("NYC".to_string()),
                network_quality: NetworkQuality::default(),
            },
            performance_history: Vec::new(),
            last_successful_connection: None,
        };

        manager.add_mirror(new_mirror);
        assert_eq!(manager.config.mirrors.len(), 2);

        // Remove a mirror
        assert!(manager.remove_mirror("test-mirror-2"));
        assert_eq!(manager.config.mirrors.len(), 1);

        // Try to remove non-existent mirror
        assert!(!manager.remove_mirror("non-existent"));
    }

    #[test]
    fn test_get_mirror_statistics() {
        let config = create_test_config();
        let manager = MirrorManager::new(config).unwrap();
        let stats = manager.get_mirror_statistics();

        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].mirror_id, "test-mirror-1");
        assert_eq!(stats[0].reliability_score, 0.95);
        assert_eq!(stats[0].avg_response_time, 100.0);
    }

    #[test]
    fn test_selection_strategy_update() {
        let config = create_test_config();
        let mut manager = MirrorManager::new(config).unwrap();

        assert_eq!(
            manager.get_selection_strategy(),
            &MirrorSelectionStrategy::LowestLatency
        );

        let new_strategy = MirrorSelectionStrategy::HighestReliability;
        assert!(manager.set_selection_strategy(new_strategy.clone()).is_ok());
        assert_eq!(manager.get_selection_strategy(), &new_strategy);
    }

    #[test]
    fn test_config_update() {
        let config = create_test_config();
        let mut manager = MirrorManager::new(config).unwrap();

        let mut new_config = create_test_config();
        new_config.max_mirror_attempts = 5;
        new_config.selection_strategy = MirrorSelectionStrategy::HighestReliability;

        assert!(manager.update_config(new_config).is_ok());
        assert_eq!(manager.config.max_mirror_attempts, 5);
        assert_eq!(
            manager.config.selection_strategy,
            MirrorSelectionStrategy::HighestReliability
        );
    }

    #[tokio::test]
    async fn test_should_benchmark() {
        let config = create_test_config();
        let manager = MirrorManager::new(config).unwrap();

        // Should benchmark initially (last_benchmark = 0)
        assert!(manager.should_benchmark().await);
    }

    #[test]
    fn test_clear_cache() {
        let config = create_test_config();
        let mut manager = MirrorManager::new(config).unwrap();

        // Add some test data to selection history
        manager.selection_state.record_selection(
            "test-mirror",
            MirrorSelectionStrategy::LowestLatency,
            true,
            100.0,
        );

        assert!(!manager
            .selection_state
            .get_selection_statistics()
            .strategy_usage
            .is_empty());

        manager.clear_cache();

        // History should be cleared
        let stats = manager.selection_state.get_selection_statistics();
        assert_eq!(stats.total_selections, 0);
    }

    #[test]
    fn test_detect_mirror_issues() {
        let config = create_test_config();
        let manager = MirrorManager::new(config).unwrap();

        // Test healthy mirror
        let healthy_mirror = &manager.config.mirrors[0];
        let issues = manager.detect_mirror_issues(healthy_mirror);
        assert!(issues.is_empty());

        // Test mirror with issues
        let mut unhealthy_mirror = manager.config.mirrors[0].clone();
        unhealthy_mirror.consecutive_failures = 4;
        unhealthy_mirror.reliability_score = 0.6;
        unhealthy_mirror.avg_response_time = Some(6000);

        let issues = manager.detect_mirror_issues(&unhealthy_mirror);
        assert!(issues.len() >= 3); // Should detect multiple issues
    }

    #[test]
    fn test_regional_config_creation() {
        let us_config = create_regional_mirror_config("us-east");
        assert_eq!(
            us_config.selection_strategy,
            MirrorSelectionStrategy::Geographic
        );
        assert!(us_config.enable_geographic_optimization);

        let global_config = create_regional_mirror_config("global");
        assert_eq!(
            global_config.selection_strategy,
            MirrorSelectionStrategy::Adaptive
        );

        let default_config = create_regional_mirror_config("unknown");
        assert!(
            matches!(
                default_config.selection_strategy,
                MirrorSelectionStrategy::Weighted(_)
            ),
            "Expected weighted strategy for unknown region"
        );
    }
}
