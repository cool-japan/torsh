//! Prometheus Metrics Exporter for Distributed Training
//!
//! This module provides Prometheus-compatible metrics export for distributed training,
//! enabling integration with Prometheus/Grafana dashboards for real-time monitoring.
//!
//! ## Features
//!
//! - **Standard Prometheus Format**: Exports metrics in Prometheus text exposition format
//! - **Comprehensive Metrics**: Includes compute, communication, memory, and I/O metrics
//! - **Multi-Rank Support**: Aggregates metrics across all distributed training ranks
//! - **HTTP Server**: Built-in HTTP server for Prometheus scraping
//! - **Custom Labels**: Support for custom labels and dimensions
//! - **Histogram Support**: Includes distribution metrics for latency analysis
//!
//! ## Usage Example
//!
//! ```rust,no_run
//! use torsh_distributed::prometheus_exporter::{PrometheusExporter, PrometheusConfig};
//! use torsh_distributed::advanced_monitoring::AdvancedMonitor;
//! use std::sync::Arc;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let monitor = Arc::new(AdvancedMonitor::new(process_group));
//!
//!     let config = PrometheusConfig::builder()
//!         .port(9090)
//!         .path("/metrics")
//!         .namespace("torsh")
//!         .build();
//!
//!     let exporter = PrometheusExporter::new(monitor, config)?;
//!     exporter.start().await?;
//!
//!     Ok(())
//! }
//! ```

use crate::advanced_monitoring::{AdvancedMetrics, AdvancedMonitor};
use crate::{TorshDistributedError, TorshResult};
use std::collections::HashMap;
use std::fmt::Write as FmtWrite;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Prometheus exporter configuration
#[derive(Debug, Clone)]
pub struct PrometheusConfig {
    /// HTTP server port for Prometheus scraping
    pub port: u16,

    /// Metrics endpoint path (default: "/metrics")
    pub path: String,

    /// Metrics namespace prefix (default: "torsh")
    pub namespace: String,

    /// Additional static labels to add to all metrics
    pub labels: HashMap<String, String>,

    /// Enable histogram metrics (may increase memory usage)
    pub enable_histograms: bool,

    /// Histogram bucket boundaries for latency metrics (in milliseconds)
    pub histogram_buckets: Vec<f64>,
}

impl Default for PrometheusConfig {
    fn default() -> Self {
        Self {
            port: 9090,
            path: "/metrics".to_string(),
            namespace: "torsh".to_string(),
            labels: HashMap::new(),
            enable_histograms: true,
            histogram_buckets: vec![
                0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0,
            ],
        }
    }
}

impl PrometheusConfig {
    /// Create a new configuration builder
    pub fn builder() -> PrometheusConfigBuilder {
        PrometheusConfigBuilder::default()
    }
}

/// Builder for PrometheusConfig
#[derive(Default)]
pub struct PrometheusConfigBuilder {
    port: Option<u16>,
    path: Option<String>,
    namespace: Option<String>,
    labels: HashMap<String, String>,
    enable_histograms: Option<bool>,
    histogram_buckets: Option<Vec<f64>>,
}

impl PrometheusConfigBuilder {
    /// Set the HTTP server port
    pub fn port(mut self, port: u16) -> Self {
        self.port = Some(port);
        self
    }

    /// Set the metrics endpoint path
    pub fn path(mut self, path: impl Into<String>) -> Self {
        self.path = Some(path.into());
        self
    }

    /// Set the metrics namespace prefix
    pub fn namespace(mut self, namespace: impl Into<String>) -> Self {
        self.namespace = Some(namespace.into());
        self
    }

    /// Add a static label
    pub fn label(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.labels.insert(key.into(), value.into());
        self
    }

    /// Enable or disable histogram metrics
    pub fn enable_histograms(mut self, enable: bool) -> Self {
        self.enable_histograms = Some(enable);
        self
    }

    /// Set custom histogram bucket boundaries
    pub fn histogram_buckets(mut self, buckets: Vec<f64>) -> Self {
        self.histogram_buckets = Some(buckets);
        self
    }

    /// Build the configuration
    pub fn build(self) -> PrometheusConfig {
        let default = PrometheusConfig::default();
        PrometheusConfig {
            port: self.port.unwrap_or(default.port),
            path: self.path.unwrap_or(default.path),
            namespace: self.namespace.unwrap_or(default.namespace),
            labels: self.labels,
            enable_histograms: self.enable_histograms.unwrap_or(default.enable_histograms),
            histogram_buckets: self.histogram_buckets.unwrap_or(default.histogram_buckets),
        }
    }
}

/// Prometheus metrics exporter for distributed training
pub struct PrometheusExporter {
    monitor: Arc<AdvancedMonitor>,
    config: PrometheusConfig,
    histogram_data: Arc<RwLock<HistogramData>>,
}

#[derive(Default)]
struct HistogramData {
    compute_forward_buckets: Vec<(f64, u64)>,
    compute_backward_buckets: Vec<(f64, u64)>,
    #[allow(dead_code)]
    communication_allreduce_buckets: Vec<(f64, u64)>,
    #[allow(dead_code)]
    communication_broadcast_buckets: Vec<(f64, u64)>,
}

impl PrometheusExporter {
    /// Create a new Prometheus exporter
    pub fn new(monitor: Arc<AdvancedMonitor>, config: PrometheusConfig) -> TorshResult<Self> {
        Ok(Self {
            monitor,
            config,
            histogram_data: Arc::new(RwLock::new(HistogramData::default())),
        })
    }

    /// Start the HTTP server for Prometheus scraping
    pub async fn start(&self) -> TorshResult<()> {
        let port = self.config.port;
        let path = self.config.path.clone();
        let path_for_log = path.clone();
        let exporter = self.clone_for_handler();

        tokio::spawn(async move {
            if let Err(e) = exporter.run_server(port, &path).await {
                log::error!("Prometheus exporter server error: {}", e);
            }
        });

        log::info!(
            "Prometheus exporter started on port {} at {}",
            port,
            path_for_log
        );
        Ok(())
    }

    /// Clone for use in async handler
    fn clone_for_handler(&self) -> Self {
        Self {
            monitor: Arc::clone(&self.monitor),
            config: self.config.clone(),
            histogram_data: Arc::clone(&self.histogram_data),
        }
    }

    /// Run the HTTP server (internal)
    async fn run_server(&self, port: u16, path: &str) -> TorshResult<()> {
        use std::net::SocketAddr;

        let addr = SocketAddr::from(([0, 0, 0, 0], port));
        let path = path.to_string();

        // Simple HTTP server implementation
        let listener = tokio::net::TcpListener::bind(addr).await.map_err(|e| {
            TorshDistributedError::io_error(format!("Failed to bind to {}: {}", addr, e))
        })?;

        log::info!("Prometheus metrics available at http://{}{}", addr, path);

        loop {
            match listener.accept().await {
                Ok((stream, _)) => {
                    let exporter = self.clone_for_handler();
                    let path = path.clone();
                    tokio::spawn(async move {
                        if let Err(e) = exporter.handle_connection(stream, &path).await {
                            log::warn!("Error handling metrics request: {}", e);
                        }
                    });
                }
                Err(e) => {
                    log::error!("Error accepting connection: {}", e);
                }
            }
        }
    }

    /// Handle a single HTTP connection
    async fn handle_connection(
        &self,
        mut stream: tokio::net::TcpStream,
        expected_path: &str,
    ) -> TorshResult<()> {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let mut buffer = [0u8; 1024];
        let n = stream.read(&mut buffer).await.map_err(|e| {
            TorshDistributedError::io_error(format!("Failed to read request: {}", e))
        })?;

        let request = String::from_utf8_lossy(&buffer[..n]);

        // Parse request line
        if let Some(first_line) = request.lines().next() {
            let parts: Vec<&str> = first_line.split_whitespace().collect();
            if parts.len() >= 2 && parts[0] == "GET" {
                let requested_path = parts[1];

                if requested_path == expected_path {
                    // Generate metrics
                    let metrics = self.export_metrics().await?;

                    // Send response
                    let response = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: text/plain; version=0.0.4\r\nContent-Length: {}\r\n\r\n{}",
                        metrics.len(),
                        metrics
                    );

                    stream.write_all(response.as_bytes()).await.map_err(|e| {
                        TorshDistributedError::io_error(format!("Failed to write response: {}", e))
                    })?;

                    return Ok(());
                }
            }
        }

        // Send 404 for invalid paths
        let not_found = "HTTP/1.1 404 Not Found\r\nContent-Length: 9\r\n\r\nNot Found";
        stream.write_all(not_found.as_bytes()).await.map_err(|e| {
            TorshDistributedError::io_error(format!("Failed to write 404 response: {}", e))
        })?;

        Ok(())
    }

    /// Export metrics in Prometheus format
    pub async fn export_metrics(&self) -> TorshResult<String> {
        let mut output = String::with_capacity(8192);

        // Get current metrics from monitor
        let metrics = self.monitor.get_latest_metrics().await?;
        let namespace = &self.config.namespace;

        // Helper to format labels
        let format_labels = |rank: u32| -> String {
            let mut labels = vec![format!("rank=\"{}\"", rank)];
            for (key, value) in &self.config.labels {
                labels.push(format!("{}=\"{}\"", key, value));
            }
            labels.join(",")
        };

        // Export compute metrics
        writeln!(
            output,
            "# HELP {}_compute_forward_time_ms Forward pass computation time in milliseconds",
            namespace
        )
        .unwrap();
        writeln!(output, "# TYPE {}_compute_forward_time_ms gauge", namespace).unwrap();

        for (rank, metric) in &metrics {
            writeln!(
                output,
                "{}_compute_forward_time_ms{{{}}} {}",
                namespace,
                format_labels(*rank),
                metric.compute.forward_time_ms
            )
            .unwrap();
        }

        writeln!(
            output,
            "# HELP {}_compute_backward_time_ms Backward pass computation time in milliseconds",
            namespace
        )
        .unwrap();
        writeln!(
            output,
            "# TYPE {}_compute_backward_time_ms gauge",
            namespace
        )
        .unwrap();

        for (rank, metric) in &metrics {
            writeln!(
                output,
                "{}_compute_backward_time_ms{{{}}} {}",
                namespace,
                format_labels(*rank),
                metric.compute.backward_time_ms
            )
            .unwrap();
        }

        // Export communication metrics
        writeln!(
            output,
            "# HELP {}_communication_allreduce_time_ms All-reduce operation time in milliseconds",
            namespace
        )
        .unwrap();
        writeln!(
            output,
            "# TYPE {}_communication_allreduce_time_ms gauge",
            namespace
        )
        .unwrap();

        for (rank, metric) in &metrics {
            writeln!(
                output,
                "{}_communication_allreduce_time_ms{{{}}} {}",
                namespace,
                format_labels(*rank),
                metric.communication.all_reduce_time_ms
            )
            .unwrap();
        }

        writeln!(
            output,
            "# HELP {}_communication_broadcast_time_ms Broadcast operation time in milliseconds",
            namespace
        )
        .unwrap();
        writeln!(
            output,
            "# TYPE {}_communication_broadcast_time_ms gauge",
            namespace
        )
        .unwrap();

        for (rank, metric) in &metrics {
            writeln!(
                output,
                "{}_communication_broadcast_time_ms{{{}}} {}",
                namespace,
                format_labels(*rank),
                metric.communication.broadcast_time_ms
            )
            .unwrap();
        }

        // Export memory metrics
        writeln!(
            output,
            "# HELP {}_memory_gpu_used_mb GPU memory used in megabytes",
            namespace
        )
        .unwrap();
        writeln!(output, "# TYPE {}_memory_gpu_used_mb gauge", namespace).unwrap();

        for (rank, metric) in &metrics {
            writeln!(
                output,
                "{}_memory_gpu_used_mb{{{}}} {}",
                namespace,
                format_labels(*rank),
                metric.memory.gpu_memory_used_mb
            )
            .unwrap();
        }

        writeln!(
            output,
            "# HELP {}_memory_peak_mb Peak memory usage in megabytes",
            namespace
        )
        .unwrap();
        writeln!(output, "# TYPE {}_memory_peak_mb gauge", namespace).unwrap();

        for (rank, metric) in &metrics {
            writeln!(
                output,
                "{}_memory_peak_mb{{{}}} {}",
                namespace,
                format_labels(*rank),
                metric.memory.peak_memory_mb
            )
            .unwrap();
        }

        // Export I/O metrics
        writeln!(
            output,
            "# HELP {}_io_data_load_time_ms Data loading time in milliseconds",
            namespace
        )
        .unwrap();
        writeln!(output, "# TYPE {}_io_data_load_time_ms gauge", namespace).unwrap();

        for (rank, metric) in &metrics {
            writeln!(
                output,
                "{}_io_data_load_time_ms{{{}}} {}",
                namespace,
                format_labels(*rank),
                metric.io.data_load_time_ms
            )
            .unwrap();
        }

        writeln!(
            output,
            "# HELP {}_io_disk_read_mbps Disk read throughput in MB/s",
            namespace
        )
        .unwrap();
        writeln!(output, "# TYPE {}_io_disk_read_mbps gauge", namespace).unwrap();

        for (rank, metric) in &metrics {
            writeln!(
                output,
                "{}_io_disk_read_mbps{{{}}} {}",
                namespace,
                format_labels(*rank),
                metric.io.disk_read_mbps
            )
            .unwrap();
        }

        writeln!(
            output,
            "# HELP {}_io_disk_write_mbps Disk write throughput in MB/s",
            namespace
        )
        .unwrap();
        writeln!(output, "# TYPE {}_io_disk_write_mbps gauge", namespace).unwrap();

        for (rank, metric) in &metrics {
            writeln!(
                output,
                "{}_io_disk_write_mbps{{{}}} {}",
                namespace,
                format_labels(*rank),
                metric.io.disk_write_mbps
            )
            .unwrap();
        }

        // Export histograms if enabled
        if self.config.enable_histograms {
            self.export_histograms(&mut output, &metrics, namespace)
                .await?;
        }

        Ok(output)
    }

    /// Export histogram metrics
    async fn export_histograms(
        &self,
        output: &mut String,
        metrics: &HashMap<u32, AdvancedMetrics>,
        namespace: &str,
    ) -> TorshResult<()> {
        // Update histogram data from metrics
        self.update_histogram_data(metrics).await;

        let histogram_data = self.histogram_data.read().await;
        let _buckets = &self.config.histogram_buckets;

        // Export forward pass histogram
        writeln!(
            output,
            "# HELP {}_compute_forward_time_histogram_ms Forward pass time distribution",
            namespace
        )
        .unwrap();
        writeln!(
            output,
            "# TYPE {}_compute_forward_time_histogram_ms histogram",
            namespace
        )
        .unwrap();

        for (bucket, count) in &histogram_data.compute_forward_buckets {
            writeln!(
                output,
                "{}_compute_forward_time_histogram_ms_bucket{{le=\"{}\"}} {}",
                namespace, bucket, count
            )
            .unwrap();
        }

        // Export backward pass histogram
        writeln!(
            output,
            "# HELP {}_compute_backward_time_histogram_ms Backward pass time distribution",
            namespace
        )
        .unwrap();
        writeln!(
            output,
            "# TYPE {}_compute_backward_time_histogram_ms histogram",
            namespace
        )
        .unwrap();

        for (bucket, count) in &histogram_data.compute_backward_buckets {
            writeln!(
                output,
                "{}_compute_backward_time_histogram_ms_bucket{{le=\"{}\"}} {}",
                namespace, bucket, count
            )
            .unwrap();
        }

        Ok(())
    }

    /// Update histogram data from current metrics
    async fn update_histogram_data(&self, metrics: &HashMap<u32, AdvancedMetrics>) {
        let mut histogram_data = self.histogram_data.write().await;
        let buckets = &self.config.histogram_buckets;

        // Initialize buckets
        histogram_data.compute_forward_buckets.clear();
        histogram_data.compute_backward_buckets.clear();

        for &boundary in buckets {
            histogram_data.compute_forward_buckets.push((boundary, 0));
            histogram_data.compute_backward_buckets.push((boundary, 0));
        }

        // Count samples in each bucket
        for metric in metrics.values() {
            // Forward pass
            for (boundary, count) in &mut histogram_data.compute_forward_buckets {
                if metric.compute.forward_time_ms <= *boundary {
                    *count += 1;
                }
            }

            // Backward pass
            for (boundary, count) in &mut histogram_data.compute_backward_buckets {
                if metric.compute.backward_time_ms <= *boundary {
                    *count += 1;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::advanced_monitoring::{
        AdvancedMetrics, CommunicationMetrics, ComputeMetrics, IoMetrics, MemoryMetrics,
    };
    use crate::backend::BackendType;
    use crate::init_process_group;

    #[tokio::test]
    async fn test_prometheus_config_builder() {
        let config = PrometheusConfig::builder()
            .port(9091)
            .path("/custom_metrics")
            .namespace("test")
            .label("env", "dev")
            .label("cluster", "test-cluster")
            .enable_histograms(false)
            .build();

        assert_eq!(config.port, 9091);
        assert_eq!(config.path, "/custom_metrics");
        assert_eq!(config.namespace, "test");
        assert_eq!(config.labels.get("env"), Some(&"dev".to_string()));
        assert_eq!(
            config.labels.get("cluster"),
            Some(&"test-cluster".to_string())
        );
        assert!(!config.enable_histograms);
    }

    #[tokio::test]
    async fn test_prometheus_exporter_creation() {
        let pg = init_process_group(BackendType::Gloo, 0, 1, "127.0.0.1", 29500)
            .await
            .unwrap();
        let monitor = Arc::new(AdvancedMonitor::new(Arc::new(pg)));

        let config = PrometheusConfig::default();
        let exporter = PrometheusExporter::new(monitor, config);

        assert!(exporter.is_ok());
    }

    #[tokio::test]
    async fn test_metrics_export_format() {
        let pg = init_process_group(BackendType::Gloo, 0, 2, "127.0.0.1", 29500)
            .await
            .unwrap();
        let monitor = Arc::new(AdvancedMonitor::new(Arc::new(pg)));

        // Record some test metrics
        let test_metrics = AdvancedMetrics {
            timestamp: std::time::Duration::from_secs(0),
            compute: ComputeMetrics {
                forward_time_ms: 10.5,
                backward_time_ms: 15.2,
                optimizer_time_ms: 2.3,
                gpu_utilization: 85.0,
                cpu_utilization: 60.0,
                tensor_core_utilization: 75.0,
                gflops: 100.5,
            },
            communication: CommunicationMetrics {
                all_reduce_time_ms: 8.7,
                broadcast_time_ms: 3.2,
                all_gather_time_ms: 1.5,
                bandwidth_mbps: 1024.0,
                comm_comp_ratio: 0.3,
                num_operations: 100,
                avg_message_size: 10240,
            },
            memory: MemoryMetrics {
                gpu_memory_used_mb: 512.0,
                gpu_memory_total_mb: 1024.0,
                cpu_memory_used_mb: 2048.0,
                memory_bandwidth_gbps: 10.0,
                num_allocations: 50,
                peak_memory_mb: 768.0,
            },
            io: IoMetrics {
                data_load_time_ms: 20.0,
                disk_read_mbps: 100.0,
                disk_write_mbps: 50.0,
                preprocessing_time_ms: 5.0,
            },
            custom: HashMap::new(),
        };

        monitor.record_metrics(test_metrics).unwrap();

        let config = PrometheusConfig::builder()
            .namespace("test")
            .enable_histograms(false)
            .build();

        let exporter = PrometheusExporter::new(monitor, config).unwrap();
        let output = exporter.export_metrics().await.unwrap();

        // Verify output format
        assert!(output.contains("# HELP test_compute_forward_time_ms"));
        assert!(output.contains("# TYPE test_compute_forward_time_ms gauge"));
        assert!(output.contains("test_compute_forward_time_ms{rank=\"0\"} 10.5"));
        assert!(output.contains("test_compute_backward_time_ms{rank=\"0\"} 15.2"));
        assert!(output.contains("test_communication_allreduce_time_ms{rank=\"0\"} 8.7"));
        assert!(output.contains("test_memory_gpu_used_mb{rank=\"0\"} 512"));
    }

    #[tokio::test]
    async fn test_custom_labels() {
        let pg = init_process_group(BackendType::Gloo, 0, 1, "127.0.0.1", 29500)
            .await
            .unwrap();
        let monitor = Arc::new(AdvancedMonitor::new(Arc::new(pg)));

        // Record some test metrics
        let test_metrics = AdvancedMetrics {
            timestamp: std::time::Duration::from_secs(0),
            compute: ComputeMetrics {
                forward_time_ms: 10.0,
                backward_time_ms: 15.0,
                optimizer_time_ms: 2.0,
                gpu_utilization: 85.0,
                cpu_utilization: 60.0,
                tensor_core_utilization: 75.0,
                gflops: 100.0,
            },
            communication: CommunicationMetrics {
                all_reduce_time_ms: 8.0,
                broadcast_time_ms: 3.0,
                all_gather_time_ms: 1.0,
                bandwidth_mbps: 1024.0,
                comm_comp_ratio: 0.3,
                num_operations: 100,
                avg_message_size: 10240,
            },
            memory: MemoryMetrics {
                gpu_memory_used_mb: 512.0,
                gpu_memory_total_mb: 1024.0,
                cpu_memory_used_mb: 2048.0,
                memory_bandwidth_gbps: 10.0,
                num_allocations: 50,
                peak_memory_mb: 768.0,
            },
            io: IoMetrics {
                data_load_time_ms: 20.0,
                disk_read_mbps: 100.0,
                disk_write_mbps: 50.0,
                preprocessing_time_ms: 5.0,
            },
            custom: HashMap::new(),
        };
        monitor.record_metrics(test_metrics).unwrap();

        let config = PrometheusConfig::builder()
            .label("environment", "production")
            .label("cluster", "gpu-cluster-1")
            .enable_histograms(false)
            .build();

        let exporter = PrometheusExporter::new(monitor, config).unwrap();
        let output = exporter.export_metrics().await.unwrap();

        // Verify custom labels are present
        assert!(output.contains("environment=\"production\""));
        assert!(output.contains("cluster=\"gpu-cluster-1\""));
    }
}
