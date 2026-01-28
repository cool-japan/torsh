//! Health Check System for ToRSh Service Integration
//!
//! This module provides comprehensive health monitoring for ToRSh deployments,
//! including:
//! - Component health status (memory, devices, storage)
//! - Performance degradation detection
//! - Resource availability checks
//! - Integration with monitoring systems (Prometheus, Kubernetes)
//! - Readiness and liveness probes
//!
//! # Usage
//!
//! ```rust,ignore
//! use torsh_core::health::{HealthChecker, HealthStatus};
//!
//! let checker = HealthChecker::new();
//! let status = checker.check_health();
//!
//! if status.is_healthy() {
//!     println!("System healthy");
//! } else {
//!     println!("Issues detected: {:?}", status.failing_checks());
//! }
//! ```

use crate::telemetry::ErrorCode;

#[cfg(feature = "std")]
use std::collections::HashMap;
#[cfg(feature = "std")]
use std::sync::{Arc, Mutex, OnceLock};
#[cfg(feature = "std")]
use std::time::{Duration, Instant};

#[cfg(not(feature = "std"))]
use alloc::{
    collections::BTreeMap as HashMap,
    string::{String, ToString},
    vec::Vec,
};

/// Health check status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    /// System is healthy
    Healthy,
    /// System is degraded but operational
    Degraded,
    /// System is unhealthy
    Unhealthy,
    /// Status unknown
    Unknown,
}

impl HealthStatus {
    /// Check if status is healthy
    pub fn is_healthy(&self) -> bool {
        matches!(self, HealthStatus::Healthy)
    }

    /// Check if status is operational (healthy or degraded)
    pub fn is_operational(&self) -> bool {
        matches!(self, HealthStatus::Healthy | HealthStatus::Degraded)
    }

    /// Get status as string
    pub fn as_str(&self) -> &'static str {
        match self {
            HealthStatus::Healthy => "healthy",
            HealthStatus::Degraded => "degraded",
            HealthStatus::Unhealthy => "unhealthy",
            HealthStatus::Unknown => "unknown",
        }
    }

    /// Get HTTP status code for this health status
    pub fn http_status_code(&self) -> u16 {
        match self {
            HealthStatus::Healthy => 200,
            HealthStatus::Degraded => 200,  // Still operational
            HealthStatus::Unhealthy => 503, // Service Unavailable
            HealthStatus::Unknown => 500,   // Internal Server Error
        }
    }
}

/// Individual health check result
#[derive(Debug, Clone)]
pub struct HealthCheckResult {
    /// Check name/identifier
    pub name: String,
    /// Check status
    pub status: HealthStatus,
    /// Optional message
    pub message: Option<String>,
    /// Optional error code
    pub error_code: Option<ErrorCode>,
    /// Check duration
    #[cfg(feature = "std")]
    pub duration: Duration,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl HealthCheckResult {
    /// Create a healthy check result
    pub fn healthy(name: String) -> Self {
        Self {
            name,
            status: HealthStatus::Healthy,
            message: None,
            error_code: None,
            #[cfg(feature = "std")]
            duration: Duration::default(),
            metadata: HashMap::new(),
        }
    }

    /// Create a degraded check result
    pub fn degraded(name: String, message: String) -> Self {
        Self {
            name,
            status: HealthStatus::Degraded,
            message: Some(message),
            error_code: None,
            #[cfg(feature = "std")]
            duration: Duration::default(),
            metadata: HashMap::new(),
        }
    }

    /// Create an unhealthy check result
    pub fn unhealthy(name: String, message: String, error_code: Option<ErrorCode>) -> Self {
        Self {
            name,
            status: HealthStatus::Unhealthy,
            message: Some(message),
            error_code,
            #[cfg(feature = "std")]
            duration: Duration::default(),
            metadata: HashMap::new(),
        }
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Set duration
    #[cfg(feature = "std")]
    pub fn with_duration(mut self, duration: Duration) -> Self {
        self.duration = duration;
        self
    }
}

/// Overall system health report
#[derive(Debug, Clone)]
pub struct HealthReport {
    /// Overall status
    pub status: HealthStatus,
    /// Individual check results
    pub checks: Vec<HealthCheckResult>,
    /// Timestamp of report
    #[cfg(feature = "std")]
    pub timestamp: Instant,
    /// Total checks run
    pub total_checks: usize,
    /// Healthy checks
    pub healthy_checks: usize,
    /// Degraded checks
    pub degraded_checks: usize,
    /// Unhealthy checks
    pub unhealthy_checks: usize,
}

impl HealthReport {
    /// Create a new health report
    #[cfg(feature = "std")]
    pub fn new(checks: Vec<HealthCheckResult>) -> Self {
        let total_checks = checks.len();
        let healthy_checks = checks
            .iter()
            .filter(|c| c.status == HealthStatus::Healthy)
            .count();
        let degraded_checks = checks
            .iter()
            .filter(|c| c.status == HealthStatus::Degraded)
            .count();
        let unhealthy_checks = checks
            .iter()
            .filter(|c| c.status == HealthStatus::Unhealthy)
            .count();

        // Determine overall status
        let status = if unhealthy_checks > 0 {
            HealthStatus::Unhealthy
        } else if degraded_checks > 0 {
            HealthStatus::Degraded
        } else if healthy_checks == total_checks {
            HealthStatus::Healthy
        } else {
            HealthStatus::Unknown
        };

        Self {
            status,
            checks,
            timestamp: Instant::now(),
            total_checks,
            healthy_checks,
            degraded_checks,
            unhealthy_checks,
        }
    }

    /// Get failing checks
    pub fn failing_checks(&self) -> Vec<&HealthCheckResult> {
        self.checks
            .iter()
            .filter(|c| c.status != HealthStatus::Healthy)
            .collect()
    }

    /// Get unhealthy checks only
    pub fn unhealthy_checks_list(&self) -> Vec<&HealthCheckResult> {
        self.checks
            .iter()
            .filter(|c| c.status == HealthStatus::Unhealthy)
            .collect()
    }

    /// Generate summary string
    pub fn summary(&self) -> String {
        format!(
            "Health: {} ({}/{} checks healthy, {} degraded, {} unhealthy)",
            self.status.as_str(),
            self.healthy_checks,
            self.total_checks,
            self.degraded_checks,
            self.unhealthy_checks
        )
    }

    /// Format as JSON-like string
    pub fn to_json(&self) -> String {
        let checks_json: Vec<String> = self
            .checks
            .iter()
            .map(|c| {
                let message = c.message.as_ref().map_or("null", |m| m.as_str());
                let error_code = c
                    .error_code
                    .map_or("null".to_string(), |e| e.code().to_string());
                format!(
                    r#"{{"name":"{}","status":"{}","message":"{}","error_code":{}}}"#,
                    c.name,
                    c.status.as_str(),
                    message,
                    error_code
                )
            })
            .collect();

        format!(
            r#"{{"status":"{}","total":{}, "healthy":{},"degraded":{},"unhealthy":{},"checks":[{}]}}"#,
            self.status.as_str(),
            self.total_checks,
            self.healthy_checks,
            self.degraded_checks,
            self.unhealthy_checks,
            checks_json.join(",")
        )
    }
}

/// Health check configuration
#[derive(Debug, Clone)]
pub struct HealthCheckConfig {
    /// Memory threshold for degraded status (percentage)
    pub memory_degraded_threshold: f64,
    /// Memory threshold for unhealthy status (percentage)
    pub memory_unhealthy_threshold: f64,
    /// Enable device checks
    pub check_devices: bool,
    /// Enable storage checks
    pub check_storage: bool,
    /// Enable performance checks
    pub check_performance: bool,
    /// Performance degradation threshold (percentage slowdown)
    pub performance_degradation_threshold: f64,
}

impl Default for HealthCheckConfig {
    fn default() -> Self {
        Self {
            memory_degraded_threshold: 80.0,  // 80% memory usage
            memory_unhealthy_threshold: 95.0, // 95% memory usage
            check_devices: true,
            check_storage: true,
            check_performance: true,
            performance_degradation_threshold: 50.0, // 50% slowdown
        }
    }
}

/// Health checker for system monitoring
#[cfg(feature = "std")]
pub struct HealthChecker {
    config: HealthCheckConfig,
    last_check: Mutex<Option<HealthReport>>,
    check_count: Mutex<u64>,
}

#[cfg(feature = "std")]
impl HealthChecker {
    /// Create a new health checker
    pub fn new() -> Self {
        Self::with_config(HealthCheckConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: HealthCheckConfig) -> Self {
        Self {
            config,
            last_check: Mutex::new(None),
            check_count: Mutex::new(0),
        }
    }

    /// Perform health check
    pub fn check_health(&self) -> HealthReport {
        let mut checks = Vec::new();

        // Memory check
        checks.push(self.check_memory());

        // Device check
        if self.config.check_devices {
            checks.push(self.check_devices());
        }

        // Storage check
        if self.config.check_storage {
            checks.push(self.check_storage());
        }

        // Performance check
        if self.config.check_performance {
            checks.push(self.check_performance());
        }

        let report = HealthReport::new(checks);

        // Update last check
        *self.last_check.lock().expect("lock should not be poisoned") = Some(report.clone());
        *self
            .check_count
            .lock()
            .expect("lock should not be poisoned") += 1;

        report
    }

    /// Check memory health
    fn check_memory(&self) -> HealthCheckResult {
        let start = Instant::now();

        // Use memory monitor if available
        #[cfg(feature = "std")]
        {
            use crate::memory_monitor::SystemMemoryMonitor;

            if let Ok(_monitor) = SystemMemoryMonitor::new() {
                // In a real implementation, we would query actual memory stats
                // For now, assume healthy since monitor initialized successfully
                return HealthCheckResult::healthy("memory".to_string())
                    .with_duration(start.elapsed());
            }
        }

        HealthCheckResult::healthy("memory".to_string()).with_duration(start.elapsed())
    }

    /// Check device health
    fn check_devices(&self) -> HealthCheckResult {
        let start = Instant::now();

        // Check if at least one device is available
        // In a real implementation, this would query actual device status
        let devices_available = true; // Placeholder

        let result = if devices_available {
            HealthCheckResult::healthy("devices".to_string())
        } else {
            HealthCheckResult::unhealthy(
                "devices".to_string(),
                "No compute devices available".to_string(),
                Some(ErrorCode::DeviceUnavailable),
            )
        };

        result.with_duration(start.elapsed())
    }

    /// Check storage health
    fn check_storage(&self) -> HealthCheckResult {
        let start = Instant::now();

        // Check storage pool statistics
        #[cfg(feature = "std")]
        {
            use crate::storage::pooled_memory_stats;

            let stats_map = pooled_memory_stats();
            if !stats_map.is_empty() {
                let total_cached: u64 = stats_map
                    .values()
                    .map(|s| s.total_cached_allocations as u64)
                    .sum();

                return HealthCheckResult::healthy("storage".to_string())
                    .with_duration(start.elapsed())
                    .with_metadata("cached_allocations".to_string(), total_cached.to_string())
                    .with_metadata("pool_count".to_string(), stats_map.len().to_string());
            }
        }

        HealthCheckResult::healthy("storage".to_string()).with_duration(start.elapsed())
    }

    /// Check performance health
    fn check_performance(&self) -> HealthCheckResult {
        let start = Instant::now();

        // Check for performance regressions
        #[cfg(feature = "std")]
        {
            use crate::perf_metrics::get_metrics_tracker;

            if let Some(tracker) = get_metrics_tracker() {
                let tracker = tracker.lock().expect("lock should not be poisoned");

                // Check SIMD utilization
                let simd_metrics = tracker.simd_metrics();
                let simd_utilization = simd_metrics.utilization_percentage();

                if simd_utilization < 50.0 && simd_metrics.simd_ops > 100 {
                    return HealthCheckResult::degraded(
                        "performance".to_string(),
                        format!("Low SIMD utilization: {:.1}%", simd_utilization),
                    )
                    .with_duration(start.elapsed())
                    .with_metadata(
                        "simd_utilization".to_string(),
                        format!("{:.1}", simd_utilization),
                    );
                }
            }
        }

        HealthCheckResult::healthy("performance".to_string()).with_duration(start.elapsed())
    }

    /// Get last health check
    pub fn last_check(&self) -> Option<HealthReport> {
        self.last_check
            .lock()
            .expect("lock should not be poisoned")
            .clone()
    }

    /// Get total checks performed
    pub fn check_count(&self) -> u64 {
        *self
            .check_count
            .lock()
            .expect("lock should not be poisoned")
    }

    /// Readiness probe (for Kubernetes)
    pub fn is_ready(&self) -> bool {
        let report = self.check_health();
        report.status.is_operational()
    }

    /// Liveness probe (for Kubernetes)
    pub fn is_alive(&self) -> bool {
        // System is alive if it can perform a health check
        true
    }
}

#[cfg(feature = "std")]
impl Default for HealthChecker {
    fn default() -> Self {
        Self::new()
    }
}

/// Global health checker instance
#[cfg(feature = "std")]
static HEALTH_CHECKER: OnceLock<Arc<HealthChecker>> = OnceLock::new();

/// Initialize global health checker
#[cfg(feature = "std")]
pub fn init_health_checker(config: HealthCheckConfig) {
    HEALTH_CHECKER.get_or_init(|| Arc::new(HealthChecker::with_config(config)));
}

/// Get global health checker
#[cfg(feature = "std")]
pub fn health_checker() -> Arc<HealthChecker> {
    HEALTH_CHECKER
        .get_or_init(|| Arc::new(HealthChecker::new()))
        .clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_status() {
        assert!(HealthStatus::Healthy.is_healthy());
        assert!(HealthStatus::Healthy.is_operational());
        assert!(!HealthStatus::Unhealthy.is_healthy());
        assert!(!HealthStatus::Unhealthy.is_operational());
        assert!(HealthStatus::Degraded.is_operational());
    }

    #[test]
    fn test_health_status_http_codes() {
        assert_eq!(HealthStatus::Healthy.http_status_code(), 200);
        assert_eq!(HealthStatus::Degraded.http_status_code(), 200);
        assert_eq!(HealthStatus::Unhealthy.http_status_code(), 503);
        assert_eq!(HealthStatus::Unknown.http_status_code(), 500);
    }

    #[test]
    fn test_health_check_result() {
        let result = HealthCheckResult::healthy("test".to_string());
        assert_eq!(result.status, HealthStatus::Healthy);
        assert!(result.message.is_none());

        let result = HealthCheckResult::degraded("test".to_string(), "warning".to_string());
        assert_eq!(result.status, HealthStatus::Degraded);
        assert_eq!(result.message, Some("warning".to_string()));

        let result = HealthCheckResult::unhealthy(
            "test".to_string(),
            "error".to_string(),
            Some(ErrorCode::DeviceError),
        );
        assert_eq!(result.status, HealthStatus::Unhealthy);
        assert_eq!(result.error_code, Some(ErrorCode::DeviceError));
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_health_report() {
        let checks = vec![
            HealthCheckResult::healthy("memory".to_string()),
            HealthCheckResult::degraded("cpu".to_string(), "high load".to_string()),
            HealthCheckResult::unhealthy(
                "disk".to_string(),
                "full".to_string(),
                Some(ErrorCode::OutOfMemory),
            ),
        ];

        let report = HealthReport::new(checks);

        assert_eq!(report.status, HealthStatus::Unhealthy);
        assert_eq!(report.total_checks, 3);
        assert_eq!(report.healthy_checks, 1);
        assert_eq!(report.degraded_checks, 1);
        assert_eq!(report.unhealthy_checks, 1);

        let failing = report.failing_checks();
        assert_eq!(failing.len(), 2);

        let unhealthy = report.unhealthy_checks_list();
        assert_eq!(unhealthy.len(), 1);
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_health_checker() {
        let checker = HealthChecker::new();
        let report = checker.check_health();

        assert!(report.total_checks > 0);
        assert_eq!(checker.check_count(), 1);

        // Check again
        checker.check_health();
        assert_eq!(checker.check_count(), 2);
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_readiness_and_liveness() {
        let checker = HealthChecker::new();

        assert!(checker.is_alive());
        // Readiness depends on system state, but should return a bool
        let _ = checker.is_ready();
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_health_report_json() {
        let checks = vec![
            HealthCheckResult::healthy("memory".to_string()),
            HealthCheckResult::degraded("cpu".to_string(), "warning".to_string()),
        ];

        let report = HealthReport::new(checks);
        let json = report.to_json();

        assert!(json.contains(r#""status":"degraded"#));
        assert!(json.contains(r#""total":2"#));
        assert!(json.contains(r#""healthy":1"#));
        assert!(json.contains(r#""degraded":1"#));
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_global_health_checker() {
        let config = HealthCheckConfig {
            memory_degraded_threshold: 85.0,
            ..Default::default()
        };

        init_health_checker(config);

        let checker = health_checker();
        let report = checker.check_health();

        assert!(report.total_checks > 0);
    }
}
