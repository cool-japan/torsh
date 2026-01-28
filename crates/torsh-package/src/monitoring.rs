//! Package Monitoring and Analytics
//!
//! This module provides comprehensive monitoring and analytics capabilities for
//! package operations, including usage metrics, performance analytics, resource
//! monitoring, and real-time alerting for production observability.
//!
//! # Features
//!
//! - **Usage Metrics**: Track package downloads, uploads, and access patterns
//! - **Performance Analytics**: Monitor operation durations, compression ratios, throughput
//! - **Resource Monitoring**: Track memory, disk, bandwidth, and CPU usage
//! - **Time-Series Data**: Collect and aggregate metrics over time
//! - **Real-time Alerting**: Trigger alerts based on configurable thresholds
//! - **Analytics Reports**: Generate comprehensive analytics reports
//! - **User Activity**: Track user-level and organization-level statistics
//! - **Geographic Analytics**: Monitor usage patterns by region
//!
//! # Examples
//!
//! ```rust
//! use torsh_package::monitoring::{MetricsCollector, MetricType, AlertThreshold};
//! use std::time::Duration;
//!
//! // Create a metrics collector
//! let mut collector = MetricsCollector::new();
//!
//! // Record package operations
//! collector.record_download("my-package", "1.0.0", Duration::from_secs(2));
//! collector.record_upload("other-package", "2.0.0", 1024 * 1024 * 50); // 50 MB
//!
//! // Configure alerting
//! collector.set_alert_threshold(
//!     MetricType::DownloadTime,
//!     AlertThreshold::Maximum(Duration::from_secs(10)),
//! );
//!
//! // Generate analytics report
//! let report = collector.generate_report();
//! println!("Total downloads: {}", report.total_downloads);
//! ```

use chrono::{DateTime, Duration as ChronoDuration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::Duration;

/// Metric types tracked by the monitoring system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MetricType {
    /// Package download operation
    Download,
    /// Package upload operation
    Upload,
    /// Package access/read operation
    Access,
    /// Package creation operation
    Creation,
    /// Package deletion operation
    Deletion,
    /// Download duration
    DownloadTime,
    /// Upload duration
    UploadTime,
    /// Compression operation duration
    CompressionTime,
    /// Decompression operation duration
    DecompressionTime,
    /// Bandwidth usage in bytes
    BandwidthUsage,
    /// Storage usage in bytes
    StorageUsage,
    /// Memory usage in bytes
    MemoryUsage,
    /// CPU usage percentage
    CpuUsage,
    /// Error count
    ErrorCount,
}

/// Alert threshold configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertThreshold {
    /// Maximum value threshold
    Maximum(Duration),
    /// Maximum count within time window
    MaxCount {
        /// Maximum number of occurrences
        count: usize,
        /// Time window duration
        window: ChronoDuration,
    },
    /// Maximum bytes threshold
    MaxBytes(u64),
    /// Maximum percentage threshold
    MaxPercentage(f64),
    /// Minimum value threshold (e.g., for uptime)
    Minimum(Duration),
}

/// Alert severity level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AlertSeverity {
    /// Informational alert
    Info,
    /// Warning level alert
    Warning,
    /// Error level alert
    Error,
    /// Critical alert requiring immediate attention
    Critical,
}

/// Alert triggered by threshold violation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    /// Alert severity
    pub severity: AlertSeverity,
    /// Metric type that triggered the alert
    pub metric_type: MetricType,
    /// Alert message
    pub message: String,
    /// Current value
    pub current_value: String,
    /// Threshold that was violated
    pub threshold: String,
    /// Timestamp when alert was triggered
    pub timestamp: DateTime<Utc>,
    /// Package ID if relevant
    pub package_id: Option<String>,
}

/// Metric data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricPoint {
    /// Metric type
    pub metric_type: MetricType,
    /// Timestamp of measurement
    pub timestamp: DateTime<Utc>,
    /// Numeric value
    pub value: f64,
    /// Package ID if relevant
    pub package_id: Option<String>,
    /// User ID if relevant
    pub user_id: Option<String>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Time-series data for a specific metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeries {
    /// Metric type
    pub metric_type: MetricType,
    /// Data points sorted by timestamp
    pub points: Vec<MetricPoint>,
    /// Aggregated statistics
    pub stats: TimeSeriesStats,
}

/// Statistical summary of time-series data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesStats {
    /// Total number of data points
    pub count: usize,
    /// Sum of all values
    pub sum: f64,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// Average value
    pub mean: f64,
    /// Median value (50th percentile)
    pub median: f64,
    /// 95th percentile value
    pub p95: f64,
    /// 99th percentile value
    pub p99: f64,
}

/// Package usage statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PackageStats {
    /// Package ID
    pub package_id: String,
    /// Package version
    pub version: String,
    /// Total downloads
    pub downloads: u64,
    /// Total uploads
    pub uploads: u64,
    /// Total accesses
    pub accesses: u64,
    /// Total bandwidth (bytes)
    pub bandwidth_bytes: u64,
    /// Average download time (seconds)
    pub avg_download_time: f64,
    /// Total storage used (bytes)
    pub storage_bytes: u64,
    /// Unique users
    pub unique_users: usize,
    /// Error count
    pub errors: u64,
    /// Last access timestamp
    pub last_access: Option<DateTime<Utc>>,
}

/// User activity statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UserStats {
    /// User ID
    pub user_id: String,
    /// Total downloads
    pub downloads: u64,
    /// Total uploads
    pub uploads: u64,
    /// Unique packages accessed
    pub unique_packages: usize,
    /// Total bandwidth used (bytes)
    pub bandwidth_bytes: u64,
    /// First activity timestamp
    pub first_activity: Option<DateTime<Utc>>,
    /// Last activity timestamp
    pub last_activity: Option<DateTime<Utc>>,
}

/// Geographic region statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RegionStats {
    /// Region name
    pub region: String,
    /// Total requests
    pub requests: u64,
    /// Total bandwidth (bytes)
    pub bandwidth_bytes: u64,
    /// Average latency (milliseconds)
    pub avg_latency_ms: f64,
    /// Error rate percentage
    pub error_rate: f64,
}

/// Comprehensive analytics report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsReport {
    /// Report generation timestamp
    pub generated_at: DateTime<Utc>,
    /// Report time range start
    pub time_range_start: DateTime<Utc>,
    /// Report time range end
    pub time_range_end: DateTime<Utc>,
    /// Total downloads across all packages
    pub total_downloads: u64,
    /// Total uploads across all packages
    pub total_uploads: u64,
    /// Total bandwidth used (bytes)
    pub total_bandwidth_bytes: u64,
    /// Total storage used (bytes)
    pub total_storage_bytes: u64,
    /// Total errors
    pub total_errors: u64,
    /// Per-package statistics
    pub package_stats: Vec<PackageStats>,
    /// Per-user statistics
    pub user_stats: Vec<UserStats>,
    /// Per-region statistics
    pub region_stats: Vec<RegionStats>,
    /// Active alerts
    pub active_alerts: Vec<Alert>,
    /// Top packages by downloads
    pub top_packages: Vec<(String, u64)>,
    /// Top users by activity
    pub top_users: Vec<(String, u64)>,
}

/// Metrics collector for package monitoring
///
/// Collects, aggregates, and analyzes package operation metrics in real-time.
/// Supports alerting, time-series data, and comprehensive analytics reporting.
pub struct MetricsCollector {
    /// Time-series data by metric type
    time_series: HashMap<MetricType, Vec<MetricPoint>>,
    /// Package statistics
    package_stats: HashMap<String, PackageStats>,
    /// User statistics
    user_stats: HashMap<String, UserStats>,
    /// Region statistics
    region_stats: HashMap<String, RegionStats>,
    /// Alert thresholds
    alert_thresholds: HashMap<MetricType, AlertThreshold>,
    /// Active alerts
    active_alerts: VecDeque<Alert>,
    /// Maximum alerts to keep
    max_alerts: usize,
    /// Maximum time-series points per metric
    max_points_per_metric: usize,
}

impl MetricsCollector {
    /// Create a new metrics collector
    pub fn new() -> Self {
        Self {
            time_series: HashMap::new(),
            package_stats: HashMap::new(),
            user_stats: HashMap::new(),
            region_stats: HashMap::new(),
            alert_thresholds: HashMap::new(),
            active_alerts: VecDeque::new(),
            max_alerts: 1000,
            max_points_per_metric: 10000,
        }
    }

    /// Set maximum number of alerts to retain
    pub fn set_max_alerts(&mut self, max: usize) {
        self.max_alerts = max;
    }

    /// Set maximum number of time-series points per metric
    pub fn set_max_points_per_metric(&mut self, max: usize) {
        self.max_points_per_metric = max;
    }

    /// Record a package download
    pub fn record_download(&mut self, package_id: &str, version: &str, duration: Duration) {
        // Record metric point
        self.record_metric(
            MetricType::Download,
            1.0,
            Some(package_id.to_string()),
            None,
            HashMap::new(),
        );

        self.record_metric(
            MetricType::DownloadTime,
            duration.as_secs_f64(),
            Some(package_id.to_string()),
            None,
            HashMap::new(),
        );

        // Update package stats
        let key = format!("{}:{}", package_id, version);
        let stats = self
            .package_stats
            .entry(key.clone())
            .or_insert_with(|| PackageStats {
                package_id: package_id.to_string(),
                version: version.to_string(),
                ..Default::default()
            });

        stats.downloads += 1;
        stats.last_access = Some(Utc::now());

        // Update average download time
        stats.avg_download_time = (stats.avg_download_time * (stats.downloads - 1) as f64
            + duration.as_secs_f64())
            / stats.downloads as f64;

        // Check for alerts
        self.check_alert(
            MetricType::DownloadTime,
            duration.as_secs_f64(),
            Some(package_id),
        );
    }

    /// Record a package upload
    pub fn record_upload(&mut self, package_id: &str, version: &str, size_bytes: u64) {
        self.record_metric(
            MetricType::Upload,
            1.0,
            Some(package_id.to_string()),
            None,
            HashMap::new(),
        );

        self.record_metric(
            MetricType::BandwidthUsage,
            size_bytes as f64,
            Some(package_id.to_string()),
            None,
            HashMap::new(),
        );

        // Update package stats
        let key = format!("{}:{}", package_id, version);
        let stats = self
            .package_stats
            .entry(key)
            .or_insert_with(|| PackageStats {
                package_id: package_id.to_string(),
                version: version.to_string(),
                ..Default::default()
            });

        stats.uploads += 1;
        stats.bandwidth_bytes += size_bytes;
        stats.storage_bytes += size_bytes;
        stats.last_access = Some(Utc::now());

        // Check for alerts
        self.check_alert(
            MetricType::BandwidthUsage,
            size_bytes as f64,
            Some(package_id),
        );
    }

    /// Record a package access
    pub fn record_access(&mut self, package_id: &str, version: &str, user_id: Option<&str>) {
        self.record_metric(
            MetricType::Access,
            1.0,
            Some(package_id.to_string()),
            user_id.map(|s| s.to_string()),
            HashMap::new(),
        );

        // Update package stats
        let key = format!("{}:{}", package_id, version);
        let stats = self
            .package_stats
            .entry(key)
            .or_insert_with(|| PackageStats {
                package_id: package_id.to_string(),
                version: version.to_string(),
                ..Default::default()
            });

        stats.accesses += 1;
        stats.last_access = Some(Utc::now());

        // Update user stats if user_id provided
        if let Some(uid) = user_id {
            let user_stats = self
                .user_stats
                .entry(uid.to_string())
                .or_insert_with(|| UserStats {
                    user_id: uid.to_string(),
                    first_activity: Some(Utc::now()),
                    ..Default::default()
                });

            user_stats.last_activity = Some(Utc::now());
        }
    }

    /// Record an error
    pub fn record_error(&mut self, package_id: Option<&str>, error_type: &str) {
        let mut metadata = HashMap::new();
        metadata.insert("error_type".to_string(), error_type.to_string());

        self.record_metric(
            MetricType::ErrorCount,
            1.0,
            package_id.map(|s| s.to_string()),
            None,
            metadata,
        );

        // Update package stats if package_id provided
        if let Some(pid) = package_id {
            for stats in self.package_stats.values_mut() {
                if stats.package_id == pid {
                    stats.errors += 1;
                }
            }
        }

        // Check for alerts
        self.check_alert(MetricType::ErrorCount, 1.0, package_id);
    }

    /// Record resource usage
    pub fn record_resource_usage(
        &mut self,
        memory_bytes: u64,
        storage_bytes: u64,
        cpu_percent: f64,
    ) {
        self.record_metric(
            MetricType::MemoryUsage,
            memory_bytes as f64,
            None,
            None,
            HashMap::new(),
        );

        self.record_metric(
            MetricType::StorageUsage,
            storage_bytes as f64,
            None,
            None,
            HashMap::new(),
        );

        self.record_metric(
            MetricType::CpuUsage,
            cpu_percent,
            None,
            None,
            HashMap::new(),
        );

        // Check for alerts
        self.check_alert(MetricType::MemoryUsage, memory_bytes as f64, None);
        self.check_alert(MetricType::StorageUsage, storage_bytes as f64, None);
        self.check_alert(MetricType::CpuUsage, cpu_percent, None);
    }

    /// Record a generic metric
    pub fn record_metric(
        &mut self,
        metric_type: MetricType,
        value: f64,
        package_id: Option<String>,
        user_id: Option<String>,
        metadata: HashMap<String, String>,
    ) {
        let point = MetricPoint {
            metric_type,
            timestamp: Utc::now(),
            value,
            package_id,
            user_id,
            metadata,
        };

        let points = self.time_series.entry(metric_type).or_insert_with(Vec::new);
        points.push(point);

        // Trim old points if exceeding limit
        if points.len() > self.max_points_per_metric {
            let excess = points.len() - self.max_points_per_metric;
            points.drain(0..excess);
        }
    }

    /// Set an alert threshold for a metric type
    pub fn set_alert_threshold(&mut self, metric_type: MetricType, threshold: AlertThreshold) {
        self.alert_thresholds.insert(metric_type, threshold);
    }

    /// Get time-series data for a metric type
    pub fn get_time_series(&self, metric_type: MetricType) -> Option<TimeSeries> {
        self.time_series.get(&metric_type).map(|points| {
            let stats = self.calculate_stats(points);
            TimeSeries {
                metric_type,
                points: points.clone(),
                stats,
            }
        })
    }

    /// Get package statistics
    pub fn get_package_stats(&self, package_id: &str) -> Vec<&PackageStats> {
        self.package_stats
            .values()
            .filter(|stats| stats.package_id == package_id)
            .collect()
    }

    /// Get user statistics
    pub fn get_user_stats(&self, user_id: &str) -> Option<&UserStats> {
        self.user_stats.get(user_id)
    }

    /// Get active alerts
    pub fn get_active_alerts(&self) -> Vec<&Alert> {
        self.active_alerts.iter().collect()
    }

    /// Clear all alerts
    pub fn clear_alerts(&mut self) {
        self.active_alerts.clear();
    }

    /// Generate comprehensive analytics report
    pub fn generate_report(&self) -> AnalyticsReport {
        let now = Utc::now();
        let time_range_start = now - ChronoDuration::days(30); // Last 30 days

        let total_downloads = self.package_stats.values().map(|s| s.downloads).sum();

        let total_uploads = self.package_stats.values().map(|s| s.uploads).sum();

        let total_bandwidth_bytes = self.package_stats.values().map(|s| s.bandwidth_bytes).sum();

        let total_storage_bytes = self.package_stats.values().map(|s| s.storage_bytes).sum();

        let total_errors = self.package_stats.values().map(|s| s.errors).sum();

        // Get top packages by downloads
        let mut top_packages: Vec<(String, u64)> = self
            .package_stats
            .values()
            .map(|s| (s.package_id.clone(), s.downloads))
            .collect();
        top_packages.sort_by(|a, b| b.1.cmp(&a.1));
        top_packages.truncate(10);

        // Get top users by activity
        let mut top_users: Vec<(String, u64)> = self
            .user_stats
            .values()
            .map(|s| (s.user_id.clone(), s.downloads + s.uploads))
            .collect();
        top_users.sort_by(|a, b| b.1.cmp(&a.1));
        top_users.truncate(10);

        AnalyticsReport {
            generated_at: now,
            time_range_start,
            time_range_end: now,
            total_downloads,
            total_uploads,
            total_bandwidth_bytes,
            total_storage_bytes,
            total_errors,
            package_stats: self.package_stats.values().cloned().collect(),
            user_stats: self.user_stats.values().cloned().collect(),
            region_stats: self.region_stats.values().cloned().collect(),
            active_alerts: self.active_alerts.iter().cloned().collect(),
            top_packages,
            top_users,
        }
    }

    /// Export metrics to JSON
    pub fn export_to_json(&self) -> Result<String, String> {
        let report = self.generate_report();
        serde_json::to_string_pretty(&report).map_err(|e| e.to_string())
    }

    // Private helper methods

    fn calculate_stats(&self, points: &[MetricPoint]) -> TimeSeriesStats {
        if points.is_empty() {
            return TimeSeriesStats {
                count: 0,
                sum: 0.0,
                min: 0.0,
                max: 0.0,
                mean: 0.0,
                median: 0.0,
                p95: 0.0,
                p99: 0.0,
            };
        }

        let mut values: Vec<f64> = points.iter().map(|p| p.value).collect();
        values.sort_by(|a, b| {
            a.partial_cmp(b)
                .expect("metric values should be comparable")
        });

        let count = values.len();
        let sum: f64 = values.iter().sum();
        let min = values[0];
        let max = values[count - 1];
        let mean = sum / count as f64;

        let median = if count % 2 == 0 {
            (values[count / 2 - 1] + values[count / 2]) / 2.0
        } else {
            values[count / 2]
        };

        let p95_idx = ((count as f64) * 0.95) as usize;
        let p99_idx = ((count as f64) * 0.99) as usize;
        let p95 = values[p95_idx.min(count - 1)];
        let p99 = values[p99_idx.min(count - 1)];

        TimeSeriesStats {
            count,
            sum,
            min,
            max,
            mean,
            median,
            p95,
            p99,
        }
    }

    fn check_alert(&mut self, metric_type: MetricType, value: f64, package_id: Option<&str>) {
        if let Some(threshold) = self.alert_thresholds.get(&metric_type) {
            let (should_alert, alert_value) = match threshold {
                AlertThreshold::Maximum(max_duration) => {
                    (value > max_duration.as_secs_f64(), value)
                }
                AlertThreshold::MaxBytes(max_bytes) => (value > *max_bytes as f64, value),
                AlertThreshold::MaxPercentage(max_percent) => (value > *max_percent, value),
                AlertThreshold::Minimum(min_duration) => {
                    (value < min_duration.as_secs_f64(), value)
                }
                AlertThreshold::MaxCount { count, window } => {
                    // Count occurrences within time window
                    let cutoff = Utc::now() - *window;
                    if let Some(points) = self.time_series.get(&metric_type) {
                        let recent_count = points.iter().filter(|p| p.timestamp > cutoff).count();
                        (recent_count > *count, recent_count as f64)
                    } else {
                        (false, 0.0)
                    }
                }
            };

            if should_alert {
                let alert = Alert {
                    severity: self.determine_severity(metric_type, alert_value),
                    metric_type,
                    message: format!("Threshold exceeded for {:?}", metric_type),
                    current_value: format!("{:.2}", alert_value),
                    threshold: format!("{:?}", threshold),
                    timestamp: Utc::now(),
                    package_id: package_id.map(|s| s.to_string()),
                };

                self.active_alerts.push_back(alert);

                // Trim old alerts
                if self.active_alerts.len() > self.max_alerts {
                    self.active_alerts.pop_front();
                }
            }
        }
    }

    fn determine_severity(&self, metric_type: MetricType, value: f64) -> AlertSeverity {
        match metric_type {
            MetricType::ErrorCount => {
                if value > 100.0 {
                    AlertSeverity::Critical
                } else if value > 50.0 {
                    AlertSeverity::Error
                } else if value > 10.0 {
                    AlertSeverity::Warning
                } else {
                    AlertSeverity::Info
                }
            }
            MetricType::DownloadTime | MetricType::UploadTime => {
                if value > 60.0 {
                    AlertSeverity::Critical
                } else if value > 30.0 {
                    AlertSeverity::Error
                } else if value > 10.0 {
                    AlertSeverity::Warning
                } else {
                    AlertSeverity::Info
                }
            }
            MetricType::CpuUsage | MetricType::MemoryUsage => {
                if value > 90.0 {
                    AlertSeverity::Critical
                } else if value > 80.0 {
                    AlertSeverity::Error
                } else if value > 70.0 {
                    AlertSeverity::Warning
                } else {
                    AlertSeverity::Info
                }
            }
            _ => AlertSeverity::Info,
        }
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_collector_creation() {
        let collector = MetricsCollector::new();
        let report = collector.generate_report();
        assert_eq!(report.total_downloads, 0);
        assert_eq!(report.total_uploads, 0);
    }

    #[test]
    fn test_record_download() {
        let mut collector = MetricsCollector::new();
        collector.record_download("test-pkg", "1.0.0", Duration::from_secs(2));

        let stats = collector.get_package_stats("test-pkg");
        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].downloads, 1);
        assert_eq!(stats[0].avg_download_time, 2.0);
    }

    #[test]
    fn test_record_upload() {
        let mut collector = MetricsCollector::new();
        collector.record_upload("test-pkg", "1.0.0", 1024 * 1024);

        let stats = collector.get_package_stats("test-pkg");
        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].uploads, 1);
        assert_eq!(stats[0].bandwidth_bytes, 1024 * 1024);
    }

    #[test]
    fn test_record_access() {
        let mut collector = MetricsCollector::new();
        collector.record_access("test-pkg", "1.0.0", Some("alice"));

        let stats = collector.get_package_stats("test-pkg");
        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].accesses, 1);

        let user_stats = collector.get_user_stats("alice");
        assert!(user_stats.is_some());
    }

    #[test]
    fn test_record_error() {
        let mut collector = MetricsCollector::new();
        collector.record_upload("test-pkg", "1.0.0", 1024);
        collector.record_error(Some("test-pkg"), "download_failed");

        let stats = collector.get_package_stats("test-pkg");
        assert_eq!(stats[0].errors, 1);
    }

    #[test]
    fn test_alert_threshold() {
        let mut collector = MetricsCollector::new();
        collector.set_alert_threshold(
            MetricType::DownloadTime,
            AlertThreshold::Maximum(Duration::from_secs(5)),
        );

        // This should trigger an alert
        collector.record_download("test-pkg", "1.0.0", Duration::from_secs(10));

        let alerts = collector.get_active_alerts();
        assert!(!alerts.is_empty());
        assert_eq!(alerts[0].metric_type, MetricType::DownloadTime);
    }

    #[test]
    fn test_time_series() {
        let mut collector = MetricsCollector::new();

        for i in 1..=10 {
            collector.record_metric(MetricType::Download, i as f64, None, None, HashMap::new());
        }

        let ts = collector.get_time_series(MetricType::Download);
        assert!(ts.is_some());

        let ts = ts.unwrap();
        assert_eq!(ts.stats.count, 10);
        assert_eq!(ts.stats.min, 1.0);
        assert_eq!(ts.stats.max, 10.0);
        assert_eq!(ts.stats.mean, 5.5);
    }

    #[test]
    fn test_generate_report() {
        let mut collector = MetricsCollector::new();

        collector.record_download("pkg1", "1.0.0", Duration::from_secs(2));
        collector.record_download("pkg2", "1.0.0", Duration::from_secs(3));
        collector.record_upload("pkg3", "1.0.0", 1024 * 1024);

        let report = collector.generate_report();
        assert_eq!(report.total_downloads, 2);
        assert_eq!(report.total_uploads, 1);
        assert!(report.total_bandwidth_bytes > 0);
    }

    #[test]
    fn test_export_to_json() {
        let mut collector = MetricsCollector::new();
        collector.record_download("test-pkg", "1.0.0", Duration::from_secs(2));

        let json = collector.export_to_json();
        assert!(json.is_ok());
        let json_str = json.unwrap();
        assert!(json_str.contains("total_downloads"));
        assert!(json_str.contains("test-pkg"));
    }

    #[test]
    fn test_max_points_limit() {
        let mut collector = MetricsCollector::new();
        collector.set_max_points_per_metric(100);

        // Record more than max
        for i in 0..200 {
            collector.record_metric(MetricType::Download, i as f64, None, None, HashMap::new());
        }

        let ts = collector.get_time_series(MetricType::Download);
        assert!(ts.is_some());
        assert_eq!(ts.unwrap().points.len(), 100);
    }

    #[test]
    fn test_alert_severity() {
        let mut collector = MetricsCollector::new();
        collector.set_alert_threshold(
            MetricType::ErrorCount,
            AlertThreshold::MaxCount {
                count: 10,
                window: ChronoDuration::minutes(5),
            },
        );

        // Record many errors
        for _ in 0..15 {
            collector.record_error(Some("test-pkg"), "error");
        }

        let alerts = collector.get_active_alerts();
        assert!(!alerts.is_empty());

        // Check severity increases with error count
        let severities: Vec<AlertSeverity> = alerts.iter().map(|a| a.severity).collect();
        assert!(severities.iter().any(|s| *s >= AlertSeverity::Warning));
    }
}
