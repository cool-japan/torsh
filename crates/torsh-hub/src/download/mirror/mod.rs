//! Advanced Mirror Management System
//!
//! This module provides a comprehensive mirror management system with sophisticated
//! selection algorithms, performance monitoring, geographic optimization, and
//! intelligent failover capabilities for the ToRSh hub.
//!
//! The system is designed with a modular architecture that separates concerns:
//! - **Types**: Core data structures and configuration types
//! - **Geographic**: Geographic calculations and proximity optimization
//! - **Performance**: Performance analysis, monitoring, and benchmarking
//! - **Selection**: Mirror selection algorithms and strategies
//! - **Manager**: Main orchestration and coordination
//! - **Utils**: Utility functions and helper routines
//!
//! # Features
//! - Multiple mirror selection strategies (latency, reliability, geographic, weighted, adaptive, ML)
//! - Real-time performance monitoring and trend analysis
//! - Geographic proximity calculations using Haversine distance
//! - Intelligent load balancing and capacity management
//! - Comprehensive error handling and fallback mechanisms
//! - Extensive benchmarking and health checking
//! - Machine learning-based adaptive selection
//! - Statistics collection and analysis
//!
//! # Examples
//!
//! ## Simple Download with Default Configuration
//! ```rust
//! use torsh_hub::download::mirror::{MirrorManager, MirrorConfig};
//! use std::path::Path;
//!
//! # tokio_test::block_on(async {
//! let config = MirrorConfig::default();
//! let mut manager = MirrorManager::new(config).unwrap();
//!
//! let result = manager.download_with_mirrors(
//!     "models/bert-base-uncased.torsh",
//!     Path::new("/tmp/model.torsh"),
//!     true
//! ).await;
//! # });
//! ```
//!
//! ## Custom Mirror Configuration
//! ```rust
//! use torsh_hub::download::mirror::{
//!     MirrorConfig, MirrorSelectionStrategy, MirrorWeights, MirrorManager
//! };
//! use std::time::Duration;
//!
//! let config = MirrorConfig {
//!     selection_strategy: MirrorSelectionStrategy::Weighted(MirrorWeights {
//!         latency: 0.4,
//!         reliability: 0.3,
//!         geographic: 0.2,
//!         load: 0.1,
//!         bandwidth: 0.0,
//!         provider_quality: 0.0,
//!     }),
//!     max_mirror_attempts: 5,
//!     connection_timeout: Duration::from_secs(15),
//!     enable_geographic_optimization: true,
//!     min_reliability_score: 0.8,
//!     max_response_time: 3000,
//!     ..Default::default()
//! };
//!
//! let mut manager = MirrorManager::new(config).unwrap();
//! ```
//!
//! ## Regional Mirror Configuration
//! ```rust
//! use torsh_hub::download::mirror::utils::create_regional_mirror_config;
//!
//! let us_config = create_regional_mirror_config("us");
//! let eu_config = create_regional_mirror_config("eu");
//! let global_config = create_regional_mirror_config("global");
//! ```

// ================================================================================================
// Module Declarations
// ================================================================================================

pub mod geographic;
pub mod manager;
pub mod performance;
pub mod selection;
pub mod types;
pub mod utils;

// ================================================================================================
// Comprehensive Re-exports for Public API
// ================================================================================================

// Core types and configurations
pub use types::{
    LoadBalancingConfig,

    MLConfig,

    MLModelState,
    MirrorAttempt,
    MirrorBenchmarkResult,
    MirrorCapacity,
    // Main configuration types
    MirrorConfig,
    MirrorDownloadResult,
    // Health and status types
    MirrorHealth,
    MirrorHealthStatus,

    MirrorLocation,
    // Internal state types (for advanced usage)
    MirrorSelectionState,
    // Selection strategy types
    MirrorSelectionStrategy,
    MirrorServer,
    MirrorStatistics,
    MirrorWeights,
    NetworkQuality,
    PerformanceAnalysis,
    PerformancePrediction,

    // Performance and monitoring types
    PerformanceSnapshot,
    PerformanceTrend,
    ProviderInfo,
    ProviderTier,
    SelectionRecord,
    SelectionStatistics,
    UserLocation,
};

// Geographic calculation utilities
pub use geographic::{
    calculate_bounding_box, calculate_geographic_midpoint, get_continent_for_country,
    normalize_longitude, validate_coordinates,
};

// Re-export calculator structs from types module
pub use types::{GeographicCalculator, PerformanceAnalyzer};

#[cfg(test)]
pub use performance::create_mirror_server;

// Mirror selection algorithms and strategies
pub use selection::{create_optimized_weights, validate_selection_strategy, MirrorSelector};

// Main mirror manager
pub use manager::{
    create_regional_mirror_config as manager_create_regional_config, download_with_default_mirrors,
    DownloadMetrics, MirrorManager,
};

// Utility functions and helpers
pub use utils::{
    calculate_average_reliability, calculate_average_response_time, calculate_load_statistics,
    create_custom_test_mirror, create_regional_mirror_config, create_test_mirror, extract_hostname,
    filter_healthy_mirrors, filter_mirrors_by_network_tier, filter_mirrors_by_region,
    get_high_reliability_mirrors, get_low_latency_mirrors, get_low_load_mirrors, is_secure_url,
    normalize_mirror_url, secure_url, validate_mirror_config, validate_mirror_server,
    LoadStatistics,
};

// ================================================================================================
// Convenience Functions and High-Level API
// ================================================================================================

/// Create a mirror manager with optimized settings for speed
///
/// This creates a configuration optimized for download speed with emphasis
/// on low latency and high bandwidth mirrors.
pub fn create_speed_optimized_manager() -> torsh_core::error::Result<MirrorManager> {
    let weights = create_optimized_weights("speed");
    let config = MirrorConfig {
        selection_strategy: MirrorSelectionStrategy::Weighted(weights),
        enable_geographic_optimization: true,
        max_mirror_attempts: 5,
        min_reliability_score: 0.8,
        max_response_time: 2000,  // 2 seconds
        benchmark_interval: 1800, // 30 minutes
        ..Default::default()
    };

    MirrorManager::new(config)
}

/// Create a mirror manager with optimized settings for reliability
///
/// This creates a configuration optimized for download reliability with emphasis
/// on high reliability scores and stable connections.
pub fn create_reliability_optimized_manager() -> torsh_core::error::Result<MirrorManager> {
    let weights = create_optimized_weights("reliability");
    let config = MirrorConfig {
        selection_strategy: MirrorSelectionStrategy::Weighted(weights),
        enable_geographic_optimization: false, // Focus on reliability over proximity
        max_mirror_attempts: 3,
        min_reliability_score: 0.9,
        max_response_time: 5000,  // 5 seconds
        benchmark_interval: 3600, // 1 hour
        ..Default::default()
    };

    MirrorManager::new(config)
}

/// Create a mirror manager with adaptive selection strategy
///
/// This creates a configuration that uses machine learning and adaptive algorithms
/// to optimize selection based on historical performance.
pub fn create_adaptive_manager() -> torsh_core::error::Result<MirrorManager> {
    let config = MirrorConfig {
        selection_strategy: MirrorSelectionStrategy::Adaptive,
        enable_geographic_optimization: true,
        max_mirror_attempts: 4,
        min_reliability_score: 0.75,
        max_response_time: 3000,  // 3 seconds
        benchmark_interval: 2700, // 45 minutes
        ..Default::default()
    };

    MirrorManager::new(config)
}

/// Create a mirror manager for a specific geographic region
///
/// This is a convenience function that combines regional configuration
/// with manager creation in a single call.
///
/// # Arguments
/// * `region` - Region identifier ("us", "eu", "asia", "global")
///
/// # Returns
/// * `Result<MirrorManager>` - Configured mirror manager for the region
pub fn create_regional_manager(region: &str) -> torsh_core::error::Result<MirrorManager> {
    let config = create_regional_mirror_config(region);
    MirrorManager::new(config)
}

/// Quick download function with automatic mirror selection
///
/// This is the simplest way to download a file using the mirror system.
/// It uses default configuration and handles all mirror selection automatically.
///
/// # Arguments
/// * `file_path` - Relative path to the file on mirror servers
/// * `dest_path` - Local destination path for the downloaded file
///
/// # Returns
/// * `Result<MirrorDownloadResult>` - Download result with metrics
///
/// # Examples
/// ```rust
/// use torsh_hub::download::mirror::quick_download;
/// use std::path::Path;
///
/// # tokio_test::block_on(async {
/// let result = quick_download(
///     "models/bert-base-uncased.torsh",
///     Path::new("/tmp/model.torsh")
/// ).await;
/// # });
/// ```
pub async fn quick_download(
    file_path: &str,
    dest_path: &std::path::Path,
) -> torsh_core::error::Result<MirrorDownloadResult> {
    download_with_default_mirrors(file_path, dest_path, false).await
}

/// Quick download function with progress display
///
/// Same as `quick_download` but displays progress information during the download.
pub async fn quick_download_with_progress(
    file_path: &str,
    dest_path: &std::path::Path,
) -> torsh_core::error::Result<MirrorDownloadResult> {
    download_with_default_mirrors(file_path, dest_path, true).await
}

// ================================================================================================
// Module-Level Documentation and Examples
// ================================================================================================

#[cfg(doc)]
pub mod examples {
    //! # Advanced Usage Examples
    //!
    //! This module contains detailed examples of advanced mirror management features.

    use super::*;
    use std::path::Path;

    /// Example: Custom performance monitoring and analysis
    pub async fn performance_monitoring_example() -> torsh_core::error::Result<()> {
        let mut manager = create_adaptive_manager()?;

        // Perform benchmarking
        let benchmark_results = manager.benchmark_mirrors().await?;
        for result in benchmark_results {
            println!(
                "Mirror {}: success={}, latency={:?}ms",
                result.mirror_id, result.success, result.response_time
            );
        }

        // Get performance statistics
        let stats = manager.get_mirror_statistics();
        for stat in stats {
            println!(
                "Mirror {} trend: {:?}, reliability: {:.2}",
                stat.mirror_id, stat.performance_trend, stat.reliability_score
            );
        }

        // Analyze performance patterns
        let performance_analyzer = manager.get_performance_analyzer();
        let analysis = performance_analyzer.analyze_performance_patterns("primary-mirror");
        println!("Performance analysis: {:?}", analysis);

        Ok(())
    }

    /// Example: Geographic optimization with custom user location
    pub async fn geographic_optimization_example() -> torsh_core::error::Result<()> {
        let mut manager = create_speed_optimized_manager()?;

        // Set custom user location (New York City)
        let geo_calc = manager.get_geographic_calculator_mut();
        geo_calc.set_user_location(40.7128, -74.0060, false);

        // Get mirrors sorted by proximity
        let config = manager.get_config();
        let sorted_mirrors = geo_calc.sort_by_geographic_proximity(config.mirrors.clone());

        for mirror in sorted_mirrors {
            let score = geo_calc.calculate_geographic_score(&mirror);
            println!(
                "Mirror {} ({}): geographic score {:.2}",
                mirror.id, mirror.location.city, score
            );
        }

        Ok(())
    }

    /// Example: Custom selection strategy with machine learning
    pub async fn ml_selection_example() -> torsh_core::error::Result<()> {
        let ml_config = MLConfig {
            enabled: true,
            learning_rate: 0.01,
            update_interval: 3600, // 1 hour
            min_samples: 20,
        };

        let config = MirrorConfig {
            selection_strategy: MirrorSelectionStrategy::MachineLearning(ml_config),
            enable_geographic_optimization: true,
            ..Default::default()
        };

        let mut manager = MirrorManager::new(config)?;

        // The manager will automatically learn from download patterns
        // and optimize selection over time
        let result = manager
            .download_with_mirrors(
                "models/example.torsh",
                Path::new("/tmp/example.torsh"),
                true,
            )
            .await?;

        println!(
            "Download completed with ML strategy: success={}",
            result.success
        );

        // Check selection statistics
        let selection_stats = manager.get_selection_statistics();
        println!(
            "Total selections: {}, success rate: {:.2}%",
            selection_stats.total_selections,
            selection_stats.success_rate * 100.0
        );

        Ok(())
    }

    /// Example: Health monitoring and issue detection
    pub async fn health_monitoring_example() -> torsh_core::error::Result<()> {
        let mut manager = create_reliability_optimized_manager()?;

        // Get health status of all mirrors
        let health_statuses = manager.get_mirror_health_status().await;
        for status in health_statuses {
            println!(
                "Mirror {}: health={:?}, issues={:?}",
                status.mirror_id, status.health, status.issues
            );

            match status.health {
                MirrorHealth::Critical => {
                    println!(
                        "⚠️ Mirror {} requires immediate attention",
                        status.mirror_id
                    );
                }
                MirrorHealth::Warning => {
                    println!("⚡ Mirror {} has performance issues", status.mirror_id);
                }
                MirrorHealth::Healthy => {
                    println!("✅ Mirror {} is operating normally", status.mirror_id);
                }
                MirrorHealth::Inactive => {
                    println!("🔴 Mirror {} is inactive", status.mirror_id);
                }
            }
        }

        Ok(())
    }
}

// ================================================================================================
// Tests Module
// ================================================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_public_api_completeness() {
        // Verify that all major types are accessible through the public API
        let _config = MirrorConfig::default();
        let _strategy = MirrorSelectionStrategy::LowestLatency;
        let _weights = MirrorWeights::default();
        let _trend = PerformanceTrend::Stable;

        // Verify utility functions are accessible
        let mirror = create_test_mirror("test", "https://test.example.com", "US", "Test City");
        assert_eq!(mirror.id, "test");

        let hostname = extract_hostname("https://example.com/path").unwrap();
        assert_eq!(hostname, "example.com");

        let normalized = normalize_mirror_url("https://example.com/path/");
        assert_eq!(normalized, "https://example.com/path");
    }

    #[test]
    fn test_convenience_manager_creation() {
        let speed_manager = create_speed_optimized_manager();
        assert!(speed_manager.is_ok());

        let reliability_manager = create_reliability_optimized_manager();
        assert!(reliability_manager.is_ok());

        let adaptive_manager = create_adaptive_manager();
        assert!(adaptive_manager.is_ok());

        let us_manager = create_regional_manager("us");
        assert!(us_manager.is_ok());

        let global_manager = create_regional_manager("global");
        assert!(global_manager.is_ok());
    }

    #[test]
    fn test_manager_configuration_validation() {
        let speed_manager = create_speed_optimized_manager().unwrap();
        match speed_manager.get_selection_strategy() {
            MirrorSelectionStrategy::Weighted(_) => {} // Expected
            _ => panic!("Speed optimized manager should use weighted strategy"),
        }

        let reliability_manager = create_reliability_optimized_manager().unwrap();
        assert_eq!(reliability_manager.get_config().min_reliability_score, 0.9);

        let adaptive_manager = create_adaptive_manager().unwrap();
        assert_eq!(
            adaptive_manager.get_selection_strategy(),
            &MirrorSelectionStrategy::Adaptive
        );
    }

    #[test]
    fn test_regional_manager_differences() {
        let us_manager = create_regional_manager("us").unwrap();
        let eu_manager = create_regional_manager("eu").unwrap();
        let asia_manager = create_regional_manager("asia").unwrap();

        // Each regional manager should have different mirror configurations
        let us_mirrors = &us_manager.get_config().mirrors;
        let eu_mirrors = &eu_manager.get_config().mirrors;
        let asia_mirrors = &asia_manager.get_config().mirrors;

        assert!(us_mirrors.iter().all(|m| m.location.country == "US"));
        assert!(eu_mirrors
            .iter()
            .any(|m| m.location.country == "GB" || m.location.country == "DE"));
        assert!(asia_mirrors
            .iter()
            .any(|m| m.location.country == "SG" || m.location.country == "JP"));
    }

    #[test]
    fn test_optimized_weights_creation() {
        let speed_weights = create_optimized_weights("speed");
        let reliability_weights = create_optimized_weights("reliability");
        let geographic_weights = create_optimized_weights("geographic");

        // Speed optimization should prioritize latency
        assert!(speed_weights.latency > reliability_weights.latency);
        assert!(speed_weights.latency > geographic_weights.latency);

        // Reliability optimization should prioritize reliability
        assert!(reliability_weights.reliability > speed_weights.reliability);
        assert!(reliability_weights.reliability > geographic_weights.reliability);

        // Geographic optimization should prioritize geographic proximity
        assert!(geographic_weights.geographic > speed_weights.geographic);
        assert!(geographic_weights.geographic > reliability_weights.geographic);
    }

    #[test]
    fn test_module_integration() {
        // Test that all modules work together correctly
        let config = create_regional_mirror_config("us");
        let manager = MirrorManager::new(config);
        assert!(manager.is_ok());

        let mut manager = manager.unwrap();

        // Test that geographic calculator is properly integrated
        let geo_calc = manager.get_geographic_calculator();
        assert!(geo_calc.is_enabled());

        // Test that performance analyzer is properly integrated
        let perf_analyzer = manager.get_performance_analyzer();
        assert!(perf_analyzer.is_enabled());

        // Test that configuration validation works
        let stats = manager.get_mirror_statistics();
        assert!(!stats.is_empty());
    }

    #[tokio::test]
    async fn test_benchmarking_integration() {
        let mut manager = create_adaptive_manager().unwrap();

        // This test verifies that the benchmarking system integrates properly
        // Note: Actual network requests will fail in test environment, but
        // the integration should work correctly
        let result = manager.benchmark_mirrors().await;

        // The benchmark should complete (though individual mirrors may fail)
        assert!(result.is_ok());
    }

    #[test]
    fn test_error_handling_integration() {
        // Test invalid configurations are properly rejected
        let mut invalid_config = MirrorConfig::default();
        invalid_config.mirrors.clear(); // No mirrors should cause validation error

        let manager = MirrorManager::new(invalid_config);
        assert!(manager.is_err());

        // Test invalid selection strategy validation
        let invalid_weights = MirrorWeights {
            latency: 2.0, // Sum > 1.0 should fail
            reliability: 0.0,
            load: 0.0,
            geographic: 0.0,
            bandwidth: 0.0,
            provider_quality: 0.0,
        };

        let mut config = MirrorConfig::default();
        config.selection_strategy = MirrorSelectionStrategy::Weighted(invalid_weights);

        let manager = MirrorManager::new(config);
        assert!(manager.is_err());
    }

    #[test]
    fn test_re_export_completeness() {
        // Ensure all important types are re-exported and accessible
        let _: MirrorConfig = MirrorConfig::default();
        let _: MirrorSelectionStrategy = MirrorSelectionStrategy::LowestLatency;
        let _: MirrorWeights = MirrorWeights::default();
        let _: PerformanceTrend = PerformanceTrend::Stable;
        let _: MirrorHealthStatus = MirrorHealthStatus::Healthy;
        let _: ProviderTier = ProviderTier::Premium;

        // Test utility functions are re-exported
        let _ = validate_coordinates(40.0, -74.0);
        let _ = normalize_longitude(181.0);
        let _ = get_continent_for_country("US");

        // Test that test utilities are available when needed
        #[cfg(test)]
        {
            let _ = create_test_mirror("test", "https://test.example.com", "US", "Test");
            let _ = create_custom_test_mirror(
                "custom",
                "https://custom.example.com",
                0.8,
                Some(100),
                true,
            );
        }
    }
}

// ================================================================================================
// Module Documentation
// ================================================================================================

/// Mirror management system version information
pub const VERSION: &str = "1.0.0";

/// Mirror system feature flags
pub mod features {
    /// Geographic optimization support
    pub const GEOGRAPHIC_OPTIMIZATION: bool = true;

    /// Performance analysis and monitoring
    pub const PERFORMANCE_ANALYSIS: bool = true;

    /// Machine learning-based selection
    pub const MACHINE_LEARNING: bool = true;

    /// Load balancing and capacity management
    pub const LOAD_BALANCING: bool = true;

    /// Comprehensive health monitoring
    pub const HEALTH_MONITORING: bool = true;

    /// Advanced benchmarking
    pub const ADVANCED_BENCHMARKING: bool = true;
}
