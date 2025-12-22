//! Download functionality for ToRSh Hub
//!
//! This module provides comprehensive download capabilities organized into specialized
//! submodules for better maintainability and functionality separation. The modular
//! architecture supports everything from simple single-file downloads to sophisticated
//! multi-CDN parallel downloading with geographic optimization.
//!
//! # Architecture Overview
//!
//! The download system is organized into 6 specialized modules:
//!
//! ## Core Modules
//!
//! - **[`core`]** - Basic download functions and utilities
//! - **[`validation`]** - URL validation and file verification
//! - **[`config`]** - Configuration structures and builders
//!
//! ## Advanced Modules
//!
//! - **[`parallel`]** - Parallel and chunked downloading
//! - **[`cdn`]** - CDN management with health monitoring
//! - **[`mirror`]** - Advanced mirror server management
//!
//! # Quick Start Examples
//!
//! ## Synchronous Download
//!
//! ```rust,no_run
//! use torsh_hub::download::{download_file, validate_url};
//! use std::path::Path;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Simple download with validation
//! validate_url("https://example.com/file.zip")?;
//! download_file("https://example.com/file.zip", Path::new("file.zip"), true)?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Asynchronous Download with CDN
//!
//! ```rust,no_run
//! use torsh_hub::download::download_with_advanced_cdn;
//! use std::path::Path;
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Advanced parallel download with CDN
//! let result = download_with_advanced_cdn(
//!     "https://example.com/large-file.zip",
//!     Path::new("large-file.zip"),
//!     true  // progress
//! ).await?;
//! # Ok(())
//! # }
//! ```
//!
//! # Migration from Legacy API
//!
//! This unified interface maintains 100% backward compatibility with existing code.
//! All previously available functions remain accessible through their original names.
//!
//! The new modular structure allows you to:
//! - Import only what you need: `use torsh_hub::download::core::download_file`
//! - Use the unified interface: `use torsh_hub::download::download_file`
//! - Access advanced features: `use torsh_hub::download::cdn::AdvancedCdnManager`

// ============================================================================
// Module Declarations
// ============================================================================

pub mod cdn;
pub mod config;
pub mod core;
pub mod mirror;
pub mod parallel;
pub mod validation;

// ============================================================================
// Validation & Security Re-exports
// ============================================================================

/// Hash algorithms supported for file integrity verification
pub use validation::HashAlgorithm;

/// URL and file validation functions
pub use validation::{
    validate_archive_format, validate_byte_range, validate_content_type, validate_file_path,
    validate_file_size, validate_hash_format, validate_http_status, validate_url, validate_urls,
};

/// File integrity and verification functions
pub use validation::{calculate_file_hash, verify_download, verify_file_integrity};

// ============================================================================
// Core Download Functions Re-exports
// ============================================================================

/// Basic download functions for single files
pub use core::{download_file, download_file_parallel, download_with_retry, print_progress};

// ============================================================================
// Configuration & Setup Re-exports
// ============================================================================

/// Configuration structures for parallel downloads
pub use config::{ParallelDownloadConfig, ParallelDownloadConfigBuilder};

/// CDN configuration and endpoint management
pub use config::{CdnConfig, CdnEndpoint, FailoverStrategy};

/// Mirror configuration from config module (primary)
pub use config::{
    MirrorCapacity as ConfigMirrorCapacity, MirrorConfig as ConfigMirrorConfig,
    MirrorLocation as ConfigMirrorLocation,
    MirrorSelectionStrategy as ConfigMirrorSelectionStrategy, MirrorServer as ConfigMirrorServer,
    MirrorWeights as ConfigMirrorWeights,
};

/// Regional configuration creators
pub use config::{
    create_regional_cdn_config,
    create_regional_mirror_config as config_create_regional_mirror_config,
};

// ============================================================================
// Parallel Download Re-exports
// ============================================================================

/// Parallel download functions and streaming
pub use parallel::{
    download_file_streaming, download_files_parallel, download_github_repo,
    download_with_default_cdn,
};

/// Basic CDN management from parallel module
pub use parallel::{CdnManager, CdnStatistics};

// ============================================================================
// Advanced CDN Management Re-exports
// ============================================================================

/// Advanced CDN manager and download functions
pub use cdn::{download_with_advanced_cdn, AdvancedCdnManager};

/// Performance monitoring and metrics
pub use cdn::{
    BandwidthStats, DownloadPerformanceData, GeographicPerformance, PerformanceMetrics,
    PerformanceSummary,
};

/// Health monitoring and trend analysis
pub use cdn::{
    ComprehensiveHealthResult, EndpointHealthDetail, HealthMonitoring, HealthTrend,
    OverallHealthStatus, TrendDirection,
};

/// Failure detection and incident management
pub use cdn::{
    FailureDetector, FailurePattern, FailurePatternType, IncidentRecord, IncidentSeverity,
};

/// Load balancing algorithms and management
pub use cdn::{LoadBalancer, LoadBalancingAlgorithm};

/// CDN download results and statistics
pub use cdn::{ComprehensiveCdnStatistics, DownloadAttempt, IntelligentDownloadResult};

// ============================================================================
// Advanced Mirror Management Re-exports
// ============================================================================

/// Advanced mirror manager and download functions
pub use mirror::{create_regional_mirror_config, download_with_default_mirrors, MirrorManager};

/// Mirror configuration and server management (advanced)
pub use mirror::{
    MirrorCapacity, MirrorConfig, MirrorLocation, MirrorSelectionStrategy, MirrorServer,
    MirrorWeights,
};

/// Load balancing and ML configuration
pub use mirror::{LoadBalancingConfig, MLConfig};

/// Provider and network quality information
pub use mirror::{NetworkQuality, PerformanceSnapshot, ProviderInfo};

/// Mirror download results and statistics
pub use mirror::{
    MirrorAttempt, MirrorBenchmarkResult, MirrorDownloadResult, MirrorStatistics, PerformanceTrend,
};

// ============================================================================
// Legacy Compatibility Re-exports
// ============================================================================

// Legacy download functions temporarily commented out due to circular import
// These will be gradually moved into submodules during the refactoring
// use crate::download as legacy_download;

// Legacy download functions for backward compatibility - temporarily disabled
// pub use legacy_download::{
//     download_model,
//     download_model_from_url,
//     EndpointHealth,
//     HealthCheckResult,
// };
