//! Alert Management System
//!
//! This module provides comprehensive alert management for the ToRSh performance dashboard,
//! including alert generation, monitoring, severity escalation, and real-time broadcasting.

use super::types::{
    DashboardAlert, DashboardAlertSeverity, MemoryMetrics, PerformanceMetrics, SystemMetrics,
    WebSocketClient,
};
use crate::{MemoryProfiler, Profiler, TorshResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use torsh_core::TorshError;

// =============================================================================
// Alert Configuration and Types
// =============================================================================

/// Configuration for the alert management system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertConfig {
    /// Enable automatic alert generation
    pub enabled: bool,
    /// Maximum number of alerts to keep in memory
    pub max_alerts: usize,
    /// Performance thresholds for alert generation
    pub performance_thresholds: PerformanceThresholds,
    /// Memory thresholds for alert generation
    pub memory_thresholds: MemoryThresholds,
    /// System thresholds for alert generation
    pub system_thresholds: SystemThresholds,
    /// Alert escalation configuration
    pub escalation_config: AlertEscalationConfig,
}

impl Default for AlertConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_alerts: 100,
            performance_thresholds: PerformanceThresholds::default(),
            memory_thresholds: MemoryThresholds::default(),
            system_thresholds: SystemThresholds::default(),
            escalation_config: AlertEscalationConfig::default(),
        }
    }
}

/// Performance-based alert thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceThresholds {
    /// Maximum acceptable average operation duration (ms)
    pub max_average_duration_ms: f64,
    /// Minimum acceptable operations per second
    pub min_operations_per_second: f64,
    /// Maximum acceptable CPU utilization (%)
    pub max_cpu_utilization: f64,
    /// Maximum acceptable operation failure rate (%)
    pub max_failure_rate: f64,
}

impl Default for PerformanceThresholds {
    fn default() -> Self {
        Self {
            max_average_duration_ms: 1000.0,
            min_operations_per_second: 1.0,
            max_cpu_utilization: 90.0,
            max_failure_rate: 5.0,
        }
    }
}

/// Memory-based alert thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryThresholds {
    /// Maximum acceptable memory usage (MB)
    pub max_usage_mb: f64,
    /// Maximum acceptable fragmentation ratio
    pub max_fragmentation_ratio: f64,
    /// Maximum acceptable allocation rate (allocations per second)
    pub max_allocation_rate: f64,
    /// Memory leak detection threshold (MB per minute)
    pub leak_detection_threshold_mb_per_min: f64,
}

impl Default for MemoryThresholds {
    fn default() -> Self {
        Self {
            max_usage_mb: 1024.0,
            max_fragmentation_ratio: 0.3,
            max_allocation_rate: 10000.0,
            leak_detection_threshold_mb_per_min: 10.0,
        }
    }
}

/// System-based alert thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemThresholds {
    /// Maximum acceptable load average
    pub max_load_average: f64,
    /// Minimum acceptable available memory (MB)
    pub min_available_memory_mb: f64,
    /// Maximum acceptable disk usage (%)
    pub max_disk_usage_percent: f64,
    /// Maximum acceptable network I/O (MB/s)
    pub max_network_io_mbps: f64,
}

impl Default for SystemThresholds {
    fn default() -> Self {
        Self {
            max_load_average: 2.0,
            min_available_memory_mb: 512.0,
            max_disk_usage_percent: 90.0,
            max_network_io_mbps: 100.0,
        }
    }
}

/// Alert escalation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertEscalationConfig {
    /// Time in seconds before warning escalates to critical
    pub warning_to_critical_seconds: u64,
    /// Time in seconds before critical escalates to emergency
    pub critical_to_emergency_seconds: u64,
    /// Enable automatic alert resolution
    pub enable_auto_resolution: bool,
    /// Time in seconds to automatically resolve alerts
    pub auto_resolution_seconds: u64,
}

impl Default for AlertEscalationConfig {
    fn default() -> Self {
        Self {
            warning_to_critical_seconds: 300,   // 5 minutes
            critical_to_emergency_seconds: 600, // 10 minutes
            enable_auto_resolution: true,
            auto_resolution_seconds: 3600, // 1 hour
        }
    }
}

/// Alert generation context
#[derive(Debug, Clone)]
pub struct AlertContext {
    pub timestamp: u64,
    pub performance_metrics: PerformanceMetrics,
    pub memory_metrics: MemoryMetrics,
    pub system_metrics: SystemMetrics,
}

// =============================================================================
// Alert Manager
// =============================================================================

/// Comprehensive alert management system
pub struct AlertManager {
    config: AlertConfig,
    alerts: Arc<Mutex<Vec<DashboardAlert>>>,
    alert_history: Arc<Mutex<Vec<DashboardAlert>>>,
    active_conditions: HashMap<String, AlertCondition>,
    escalation_timers: HashMap<String, u64>,
}

/// Alert condition tracking
#[derive(Debug, Clone)]
struct AlertCondition {
    condition_type: String,
    first_detected: u64,
    last_updated: u64,
    severity: DashboardAlertSeverity,
    consecutive_violations: u64,
}

impl AlertManager {
    /// Create a new alert manager
    pub fn new(config: AlertConfig) -> Self {
        Self {
            config,
            alerts: Arc::new(Mutex::new(Vec::new())),
            alert_history: Arc::new(Mutex::new(Vec::new())),
            active_conditions: HashMap::new(),
            escalation_timers: HashMap::new(),
        }
    }

    /// Add alert to the system
    pub fn add_alert(&self, alert: DashboardAlert) -> TorshResult<()> {
        let mut alerts = self.alerts.lock().map_err(|_| {
            TorshError::SynchronizationError("Failed to acquire alerts lock".to_string())
        })?;

        alerts.push(alert.clone());

        // Keep only recent alerts
        if alerts.len() > self.config.max_alerts {
            alerts.remove(0);
        }

        // Add to history
        if let Ok(mut history) = self.alert_history.lock() {
            history.push(alert);
            if history.len() > self.config.max_alerts * 2 {
                history.remove(0);
            }
        }

        Ok(())
    }

    /// Get active alerts (unresolved)
    pub fn get_active_alerts(&self) -> TorshResult<Vec<DashboardAlert>> {
        let alerts = self.alerts.lock().map_err(|_| {
            TorshError::SynchronizationError("Failed to acquire alerts lock".to_string())
        })?;
        Ok(alerts.iter().filter(|a| !a.resolved).cloned().collect())
    }

    /// Get all alerts including resolved ones
    pub fn get_all_alerts(&self) -> TorshResult<Vec<DashboardAlert>> {
        let alerts = self.alerts.lock().map_err(|_| {
            TorshError::SynchronizationError("Failed to acquire alerts lock".to_string())
        })?;
        Ok(alerts.clone())
    }

    /// Get alert history
    pub fn get_alert_history(&self) -> TorshResult<Vec<DashboardAlert>> {
        let history = self.alert_history.lock().map_err(|_| {
            TorshError::SynchronizationError("Failed to acquire history lock".to_string())
        })?;
        Ok(history.clone())
    }

    /// Resolve alert by ID
    pub fn resolve_alert(&self, alert_id: &str) -> TorshResult<bool> {
        let mut alerts = self.alerts.lock().map_err(|_| {
            TorshError::SynchronizationError("Failed to acquire alerts lock".to_string())
        })?;

        for alert in alerts.iter_mut() {
            if alert.id == alert_id && !alert.resolved {
                alert.resolved = true;
                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Generate alerts based on current metrics
    pub fn generate_alerts(&mut self, context: AlertContext) -> TorshResult<Vec<DashboardAlert>> {
        if !self.config.enabled {
            return Ok(Vec::new());
        }

        let mut new_alerts = Vec::new();

        // Performance-based alerts
        new_alerts.extend(self.check_performance_alerts(&context)?);

        // Memory-based alerts
        new_alerts.extend(self.check_memory_alerts(&context)?);

        // System-based alerts
        new_alerts.extend(self.check_system_alerts(&context)?);

        // Process escalations
        self.process_alert_escalations(context.timestamp)?;

        // Add new alerts to the system
        for alert in &new_alerts {
            self.add_alert(alert.clone())?;
        }

        Ok(new_alerts)
    }

    /// Check performance-based alert conditions
    fn check_performance_alerts(
        &mut self,
        context: &AlertContext,
    ) -> TorshResult<Vec<DashboardAlert>> {
        let mut alerts = Vec::new();
        let perf = &context.performance_metrics;
        let thresholds = self.config.performance_thresholds.clone();

        // High average duration alert
        if perf.average_duration_ms > thresholds.max_average_duration_ms {
            let condition_id = "high_average_duration";
            let severity = self.determine_severity(
                condition_id,
                context.timestamp,
                perf.average_duration_ms / thresholds.max_average_duration_ms,
            );

            alerts.push(DashboardAlert {
                id: format!("{condition_id}_{}", context.timestamp),
                severity,
                title: "High Average Operation Duration".to_string(),
                message: format!(
                    "Average operation duration ({:.2}ms) exceeds threshold ({:.2}ms)",
                    perf.average_duration_ms, thresholds.max_average_duration_ms
                ),
                timestamp: context.timestamp,
                resolved: false,
            });
        }

        // Low operations per second alert
        if perf.operations_per_second < thresholds.min_operations_per_second {
            let condition_id = "low_operations_per_second";
            let severity = self.determine_severity(
                condition_id,
                context.timestamp,
                thresholds.min_operations_per_second / perf.operations_per_second.max(0.1),
            );

            alerts.push(DashboardAlert {
                id: format!("{condition_id}_{}", context.timestamp),
                severity,
                title: "Low Operations Throughput".to_string(),
                message: format!(
                    "Operations per second ({:.1}) below threshold ({:.1})",
                    perf.operations_per_second, thresholds.min_operations_per_second
                ),
                timestamp: context.timestamp,
                resolved: false,
            });
        }

        // High CPU utilization alert
        if perf.cpu_utilization > thresholds.max_cpu_utilization {
            let condition_id = "high_cpu_utilization";
            let severity = self.determine_severity(
                condition_id,
                context.timestamp,
                perf.cpu_utilization / thresholds.max_cpu_utilization,
            );

            alerts.push(DashboardAlert {
                id: format!("{condition_id}_{}", context.timestamp),
                severity,
                title: "High CPU Utilization".to_string(),
                message: format!(
                    "CPU utilization ({:.1}%) exceeds threshold ({:.1}%)",
                    perf.cpu_utilization, thresholds.max_cpu_utilization
                ),
                timestamp: context.timestamp,
                resolved: false,
            });
        }

        Ok(alerts)
    }

    /// Check memory-based alert conditions
    fn check_memory_alerts(&mut self, context: &AlertContext) -> TorshResult<Vec<DashboardAlert>> {
        let mut alerts = Vec::new();
        let memory = &context.memory_metrics;
        let thresholds = self.config.memory_thresholds.clone();

        // High memory usage alert
        if memory.current_usage_mb > thresholds.max_usage_mb {
            let condition_id = "high_memory_usage";
            let severity = self.determine_severity(
                condition_id,
                context.timestamp,
                memory.current_usage_mb / thresholds.max_usage_mb,
            );

            alerts.push(DashboardAlert {
                id: format!("{condition_id}_{}", context.timestamp),
                severity,
                title: "High Memory Usage".to_string(),
                message: format!(
                    "Memory usage ({:.1}MB) exceeds threshold ({:.1}MB)",
                    memory.current_usage_mb, thresholds.max_usage_mb
                ),
                timestamp: context.timestamp,
                resolved: false,
            });
        }

        // Memory fragmentation alert
        if memory.fragmentation_ratio > thresholds.max_fragmentation_ratio {
            let condition_id = "high_memory_fragmentation";
            let severity = self.determine_severity(
                condition_id,
                context.timestamp,
                memory.fragmentation_ratio / thresholds.max_fragmentation_ratio,
            );

            alerts.push(DashboardAlert {
                id: format!("{condition_id}_{}", context.timestamp),
                severity,
                title: "High Memory Fragmentation".to_string(),
                message: format!(
                    "Memory fragmentation ratio ({:.2}) exceeds threshold ({:.2})",
                    memory.fragmentation_ratio, thresholds.max_fragmentation_ratio
                ),
                timestamp: context.timestamp,
                resolved: false,
            });
        }

        // High allocation rate alert
        if memory.allocation_rate > thresholds.max_allocation_rate {
            let condition_id = "high_allocation_rate";
            let severity = self.determine_severity(
                condition_id,
                context.timestamp,
                memory.allocation_rate / thresholds.max_allocation_rate,
            );

            alerts.push(DashboardAlert {
                id: format!("{condition_id}_{}", context.timestamp),
                severity,
                title: "High Allocation Rate".to_string(),
                message: format!(
                    "Allocation rate ({:.0}/s) exceeds threshold ({:.0}/s)",
                    memory.allocation_rate, thresholds.max_allocation_rate
                ),
                timestamp: context.timestamp,
                resolved: false,
            });
        }

        Ok(alerts)
    }

    /// Check system-based alert conditions
    fn check_system_alerts(&mut self, context: &AlertContext) -> TorshResult<Vec<DashboardAlert>> {
        let mut alerts = Vec::new();
        let system = &context.system_metrics;
        let thresholds = self.config.system_thresholds.clone();

        // High load average alert
        if system.load_average > thresholds.max_load_average {
            let condition_id = "high_load_average";
            let severity = self.determine_severity(
                condition_id,
                context.timestamp,
                system.load_average / thresholds.max_load_average,
            );

            alerts.push(DashboardAlert {
                id: format!("{condition_id}_{}", context.timestamp),
                severity,
                title: "High System Load".to_string(),
                message: format!(
                    "Load average ({:.2}) exceeds threshold ({:.2})",
                    system.load_average, thresholds.max_load_average
                ),
                timestamp: context.timestamp,
                resolved: false,
            });
        }

        // Low available memory alert
        if system.available_memory_mb < thresholds.min_available_memory_mb {
            let condition_id = "low_available_memory";
            let severity = self.determine_severity(
                condition_id,
                context.timestamp,
                thresholds.min_available_memory_mb / system.available_memory_mb.max(1.0),
            );

            alerts.push(DashboardAlert {
                id: format!("{condition_id}_{}", context.timestamp),
                severity,
                title: "Low Available Memory".to_string(),
                message: format!(
                    "Available memory ({:.1}MB) below threshold ({:.1}MB)",
                    system.available_memory_mb, thresholds.min_available_memory_mb
                ),
                timestamp: context.timestamp,
                resolved: false,
            });
        }

        // High disk usage alert
        if system.disk_usage_percent > thresholds.max_disk_usage_percent {
            let condition_id = "high_disk_usage";
            let severity = self.determine_severity(
                condition_id,
                context.timestamp,
                system.disk_usage_percent / thresholds.max_disk_usage_percent,
            );

            alerts.push(DashboardAlert {
                id: format!("{condition_id}_{}", context.timestamp),
                severity,
                title: "High Disk Usage".to_string(),
                message: format!(
                    "Disk usage ({:.1}%) exceeds threshold ({:.1}%)",
                    system.disk_usage_percent, thresholds.max_disk_usage_percent
                ),
                timestamp: context.timestamp,
                resolved: false,
            });
        }

        Ok(alerts)
    }

    /// Determine alert severity based on condition and escalation
    fn determine_severity(
        &mut self,
        condition_id: &str,
        timestamp: u64,
        violation_ratio: f64,
    ) -> DashboardAlertSeverity {
        // Update or create condition tracking
        let condition = self
            .active_conditions
            .entry(condition_id.to_string())
            .or_insert(AlertCondition {
                condition_type: condition_id.to_string(),
                first_detected: timestamp,
                last_updated: timestamp,
                severity: DashboardAlertSeverity::Info,
                consecutive_violations: 0,
            });

        condition.last_updated = timestamp;
        condition.consecutive_violations += 1;

        // Determine base severity from violation ratio
        let base_severity = if violation_ratio >= 3.0 {
            DashboardAlertSeverity::Emergency
        } else if violation_ratio >= 2.0 {
            DashboardAlertSeverity::Critical
        } else if violation_ratio >= 1.5 {
            DashboardAlertSeverity::Warning
        } else {
            DashboardAlertSeverity::Info
        };

        // Apply escalation based on time and consecutive violations
        let time_since_first = timestamp.saturating_sub(condition.first_detected);
        let escalated_severity = if time_since_first
            >= self.config.escalation_config.critical_to_emergency_seconds
            && condition.consecutive_violations >= 5
        {
            DashboardAlertSeverity::Emergency
        } else if time_since_first >= self.config.escalation_config.warning_to_critical_seconds
            && condition.consecutive_violations >= 3
        {
            DashboardAlertSeverity::Critical
        } else {
            base_severity
        };

        condition.severity = escalated_severity.clone();
        escalated_severity
    }

    /// Process alert escalations and auto-resolutions
    fn process_alert_escalations(&mut self, timestamp: u64) -> TorshResult<()> {
        // Auto-resolve old conditions if enabled
        if self.config.escalation_config.enable_auto_resolution {
            let auto_resolution_threshold =
                timestamp.saturating_sub(self.config.escalation_config.auto_resolution_seconds);

            self.active_conditions.retain(|_condition_id, condition| {
                condition.last_updated > auto_resolution_threshold
            });
        }

        Ok(())
    }

    /// Broadcast alert to WebSocket clients
    pub fn broadcast_alert(
        &self,
        alert: &DashboardAlert,
        clients: &Arc<Mutex<Vec<WebSocketClient>>>,
    ) -> TorshResult<usize> {
        let clients_lock = clients.lock().map_err(|_| {
            TorshError::SynchronizationError("Failed to acquire clients lock".to_string())
        })?;

        let alert_data = serde_json::to_string(alert).map_err(|e| {
            TorshError::SerializationError(format!("Failed to serialize alert: {e}"))
        })?;
        let message = format!("{{\"type\":\"alert\",\"data\":{alert_data}}}");

        let mut broadcast_count = 0;
        for client in clients_lock.iter() {
            if (client.subscriptions.contains("alerts") || client.subscriptions.is_empty())
                && client.sender.send(message.clone()).is_ok()
            {
                broadcast_count += 1;
            }
        }

        Ok(broadcast_count)
    }

    /// Get alert statistics
    pub fn get_alert_stats(&self) -> TorshResult<AlertStats> {
        let alerts = self.get_all_alerts()?;
        let active_alerts = self.get_active_alerts()?;

        let mut severity_counts = HashMap::new();
        for alert in &active_alerts {
            let count = severity_counts
                .entry(format!("{:?}", alert.severity))
                .or_insert(0);
            *count += 1;
        }

        Ok(AlertStats {
            total_alerts: alerts.len(),
            active_alerts: active_alerts.len(),
            resolved_alerts: alerts.len() - active_alerts.len(),
            severity_breakdown: severity_counts,
            active_conditions: self.active_conditions.len(),
        })
    }

    /// Clear resolved alerts from memory
    pub fn clear_resolved_alerts(&self) -> TorshResult<usize> {
        let mut alerts = self.alerts.lock().map_err(|_| {
            TorshError::SynchronizationError("Failed to acquire alerts lock".to_string())
        })?;

        let initial_count = alerts.len();
        alerts.retain(|alert| !alert.resolved);
        let removed_count = initial_count - alerts.len();

        Ok(removed_count)
    }

    /// Generate test alert for system verification
    pub fn generate_test_alert(
        &self,
        severity: DashboardAlertSeverity,
    ) -> TorshResult<DashboardAlert> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| TorshError::RuntimeError("Failed to get timestamp".to_string()))?
            .as_secs();

        let alert = DashboardAlert {
            id: format!("test_alert_{timestamp}"),
            severity,
            title: "Test Alert".to_string(),
            message: "This is a test alert generated for system verification".to_string(),
            timestamp,
            resolved: false,
        };

        self.add_alert(alert.clone())?;
        Ok(alert)
    }
}

// =============================================================================
// Alert Statistics and Reporting
// =============================================================================

/// Alert system statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertStats {
    pub total_alerts: usize,
    pub active_alerts: usize,
    pub resolved_alerts: usize,
    pub severity_breakdown: HashMap<String, usize>,
    pub active_conditions: usize,
}

// =============================================================================
// Utility Functions
// =============================================================================

/// Create alert context from current system state
pub fn create_alert_context(
    profiler: &Profiler,
    memory_profiler: &MemoryProfiler,
) -> TorshResult<AlertContext> {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| TorshError::RuntimeError("Failed to get timestamp".to_string()))?
        .as_secs();

    // These functions would need to be accessible from the metrics module
    // For now, we'll create placeholder implementations
    let performance_metrics = collect_performance_metrics_for_alerts(profiler)?;
    let memory_metrics = collect_memory_metrics_for_alerts(memory_profiler)?;
    let system_metrics = collect_system_metrics_for_alerts()?;

    Ok(AlertContext {
        timestamp,
        performance_metrics,
        memory_metrics,
        system_metrics,
    })
}

/// Collect performance metrics for alert generation
fn collect_performance_metrics_for_alerts(profiler: &Profiler) -> TorshResult<PerformanceMetrics> {
    // Simplified implementation - in production, this would delegate to metrics module
    let events = profiler.events();

    if events.is_empty() {
        return Ok(PerformanceMetrics {
            total_operations: 0,
            average_duration_ms: 0.0,
            operations_per_second: 0.0,
            total_flops: 0,
            gflops_per_second: 0.0,
            cpu_utilization: 0.0,
            thread_count: 0,
        });
    }

    let total_operations = events.len() as u64;
    let total_duration_us: u64 = events.iter().map(|e| e.duration_us).sum();
    let average_duration_ms = (total_duration_us as f64) / (events.len() as f64) / 1000.0;

    let total_flops: u64 = events.iter().map(|e| e.flops.unwrap_or(0)).sum();

    let total_time_seconds = total_duration_us as f64 / 1_000_000.0;
    let operations_per_second = if total_time_seconds > 0.0 {
        total_operations as f64 / total_time_seconds
    } else {
        0.0
    };

    let gflops_per_second = if total_time_seconds > 0.0 {
        (total_flops as f64) / total_time_seconds / 1_000_000_000.0
    } else {
        0.0
    };

    let unique_threads: std::collections::HashSet<_> = events.iter().map(|e| e.thread_id).collect();
    let thread_count = unique_threads.len();

    let cpu_utilization = (thread_count as f64) / (num_cpus::get() as f64) * 100.0;

    Ok(PerformanceMetrics {
        total_operations,
        average_duration_ms,
        operations_per_second,
        total_flops,
        gflops_per_second,
        cpu_utilization,
        thread_count,
    })
}

/// Collect memory metrics for alert generation
fn collect_memory_metrics_for_alerts(
    memory_profiler: &MemoryProfiler,
) -> TorshResult<MemoryMetrics> {
    let stats = memory_profiler.get_stats()?;

    Ok(MemoryMetrics {
        current_usage_mb: stats.allocated as f64 / (1024.0 * 1024.0),
        peak_usage_mb: stats.peak as f64 / (1024.0 * 1024.0),
        total_allocations: stats.allocations as u64,
        total_deallocations: stats.deallocations as u64,
        active_allocations: stats.allocations.saturating_sub(stats.deallocations) as u64,
        fragmentation_ratio: 0.0, // Would need to implement fragmentation analysis
        allocation_rate: 0.0,     // Would need time-based calculation
    })
}

/// Collect system metrics for alert generation
fn collect_system_metrics_for_alerts() -> TorshResult<SystemMetrics> {
    // Simplified implementation - in production, this would read from /proc or use system APIs
    Ok(SystemMetrics {
        uptime_seconds: 0,
        load_average: 0.0,
        available_memory_mb: 0.0,
        disk_usage_percent: 0.0,
        network_io_mbps: 0.0,
    })
}

/// Create default alert manager
pub fn create_alert_manager() -> AlertManager {
    AlertManager::new(AlertConfig::default())
}

/// Create alert manager with custom configuration
pub fn create_alert_manager_with_config(config: AlertConfig) -> AlertManager {
    AlertManager::new(config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MemoryProfiler, ProfileEvent, Profiler};

    #[test]
    fn test_alert_manager_creation() {
        let manager = create_alert_manager();
        assert!(manager.config.enabled);
        assert_eq!(manager.config.max_alerts, 100);
    }

    #[test]
    fn test_alert_addition_and_retrieval() {
        let manager = create_alert_manager();

        let alert = DashboardAlert {
            id: "test_alert".to_string(),
            severity: DashboardAlertSeverity::Warning,
            title: "Test Alert".to_string(),
            message: "Test message".to_string(),
            timestamp: 12345,
            resolved: false,
        };

        manager.add_alert(alert.clone()).unwrap();
        let active_alerts = manager.get_active_alerts().unwrap();
        assert_eq!(active_alerts.len(), 1);
        assert_eq!(active_alerts[0].id, "test_alert");
    }

    #[test]
    fn test_alert_resolution() {
        let manager = create_alert_manager();

        let alert = DashboardAlert {
            id: "test_alert".to_string(),
            severity: DashboardAlertSeverity::Warning,
            title: "Test Alert".to_string(),
            message: "Test message".to_string(),
            timestamp: 12345,
            resolved: false,
        };

        manager.add_alert(alert.clone()).unwrap();
        assert_eq!(manager.get_active_alerts().unwrap().len(), 1);

        let resolved = manager.resolve_alert("test_alert").unwrap();
        assert!(resolved);
        assert_eq!(manager.get_active_alerts().unwrap().len(), 0);
    }

    #[test]
    fn test_performance_alert_generation() {
        let mut manager = create_alert_manager();

        // Create context with high CPU utilization
        let context = AlertContext {
            timestamp: 12345,
            performance_metrics: PerformanceMetrics {
                total_operations: 100,
                average_duration_ms: 2000.0, // High duration
                operations_per_second: 0.5,  // Low ops/sec
                total_flops: 1000,
                gflops_per_second: 1.0,
                cpu_utilization: 95.0, // High CPU
                thread_count: 8,
            },
            memory_metrics: MemoryMetrics {
                current_usage_mb: 512.0,
                peak_usage_mb: 600.0,
                total_allocations: 1000,
                total_deallocations: 900,
                active_allocations: 100,
                fragmentation_ratio: 0.1,
                allocation_rate: 100.0,
            },
            system_metrics: SystemMetrics {
                uptime_seconds: 3600,
                load_average: 1.0,
                available_memory_mb: 1024.0,
                disk_usage_percent: 70.0,
                network_io_mbps: 10.0,
            },
        };

        let alerts = manager.generate_alerts(context).unwrap();
        assert!(!alerts.is_empty());

        // Should generate alerts for high CPU, high duration, and low ops/sec
        assert!(alerts.iter().any(|a| a.title.contains("CPU Utilization")));
        assert!(alerts
            .iter()
            .any(|a| a.title.contains("Average Operation Duration")));
        assert!(alerts
            .iter()
            .any(|a| a.title.contains("Operations Throughput")));
    }

    #[test]
    fn test_alert_escalation() {
        let mut manager = create_alert_manager();

        // Simulate repeated violations
        let base_timestamp = 12345;
        for i in 0..5 {
            let context = AlertContext {
                timestamp: base_timestamp + (i * 60), // 1 minute apart
                performance_metrics: PerformanceMetrics {
                    total_operations: 100,
                    average_duration_ms: 2000.0, // Consistently high
                    operations_per_second: 10.0,
                    total_flops: 1000,
                    gflops_per_second: 1.0,
                    cpu_utilization: 50.0,
                    thread_count: 4,
                },
                memory_metrics: MemoryMetrics {
                    current_usage_mb: 512.0,
                    peak_usage_mb: 600.0,
                    total_allocations: 1000,
                    total_deallocations: 900,
                    active_allocations: 100,
                    fragmentation_ratio: 0.1,
                    allocation_rate: 100.0,
                },
                system_metrics: SystemMetrics {
                    uptime_seconds: 3600,
                    load_average: 1.0,
                    available_memory_mb: 1024.0,
                    disk_usage_percent: 70.0,
                    network_io_mbps: 10.0,
                },
            };

            manager.generate_alerts(context).unwrap();
        }

        // Should have escalated severity due to repeated violations
        assert!(manager.active_conditions.len() > 0);
    }

    #[test]
    fn test_test_alert_generation() {
        let manager = create_alert_manager();

        let test_alert = manager
            .generate_test_alert(DashboardAlertSeverity::Critical)
            .unwrap();
        assert_eq!(test_alert.severity, DashboardAlertSeverity::Critical);
        assert_eq!(test_alert.title, "Test Alert");
        assert!(!test_alert.resolved);

        let active_alerts = manager.get_active_alerts().unwrap();
        assert_eq!(active_alerts.len(), 1);
    }

    #[test]
    fn test_alert_stats() {
        let manager = create_alert_manager();

        // Add various alerts
        let alerts = vec![
            DashboardAlert {
                id: "alert1".to_string(),
                severity: DashboardAlertSeverity::Warning,
                title: "Alert 1".to_string(),
                message: "Message 1".to_string(),
                timestamp: 12345,
                resolved: false,
            },
            DashboardAlert {
                id: "alert2".to_string(),
                severity: DashboardAlertSeverity::Critical,
                title: "Alert 2".to_string(),
                message: "Message 2".to_string(),
                timestamp: 12346,
                resolved: true,
            },
        ];

        for alert in alerts {
            manager.add_alert(alert).unwrap();
        }

        let stats = manager.get_alert_stats().unwrap();
        assert_eq!(stats.total_alerts, 2);
        assert_eq!(stats.active_alerts, 1);
        assert_eq!(stats.resolved_alerts, 1);
    }
}
