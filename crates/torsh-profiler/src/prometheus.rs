//! Prometheus metrics integration for torsh-profiler
//!
//! This module provides comprehensive Prometheus metrics export functionality,
//! allowing profiling data to be exposed in Prometheus format for monitoring
//! and alerting.

use crate::{ProfileEvent, Profiler, TorshError, TorshResult};
use prometheus::{
    Counter, CounterVec, Encoder, Gauge, GaugeVec, Histogram, HistogramOpts, HistogramVec, Opts,
    Registry, TextEncoder,
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Prometheus metrics exporter for profiling data
pub struct PrometheusExporter {
    registry: Registry,
    operation_duration: HistogramVec,
    operation_count: CounterVec,
    memory_allocated: GaugeVec,
    memory_deallocated: GaugeVec,
    flops_total: CounterVec,
    bytes_transferred: CounterVec,
    active_operations: GaugeVec,
    profiling_overhead: Histogram,
    thread_activity: GaugeVec,
    custom_metrics: Arc<Mutex<HashMap<String, Box<dyn std::any::Any + Send + Sync>>>>,
}

impl PrometheusExporter {
    /// Create a new Prometheus exporter with default metrics
    pub fn new() -> TorshResult<Self> {
        Self::with_registry(Registry::new())
    }

    /// Create a new Prometheus exporter with a custom registry
    pub fn with_registry(registry: Registry) -> TorshResult<Self> {
        // Operation duration histogram
        let operation_duration = HistogramVec::new(
            HistogramOpts::new(
                "torsh_operation_duration_microseconds",
                "Operation execution duration in microseconds",
            )
            .buckets(vec![
                10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0, 50000.0, 100000.0,
            ]),
            &["operation", "thread_id"],
        )
        .map_err(|e| TorshError::operation_error(&format!("Failed to create histogram: {}", e)))?;

        // Operation count counter
        let operation_count = CounterVec::new(
            Opts::new("torsh_operation_total", "Total number of operations"),
            &["operation"],
        )
        .map_err(|e| TorshError::operation_error(&format!("Failed to create counter: {}", e)))?;

        // Memory metrics
        let memory_allocated = GaugeVec::new(
            Opts::new("torsh_memory_allocated_bytes", "Memory allocated in bytes"),
            &["operation"],
        )
        .map_err(|e| TorshError::operation_error(&format!("Failed to create gauge: {}", e)))?;

        let memory_deallocated = GaugeVec::new(
            Opts::new(
                "torsh_memory_deallocated_bytes",
                "Memory deallocated in bytes",
            ),
            &["operation"],
        )
        .map_err(|e| TorshError::operation_error(&format!("Failed to create gauge: {}", e)))?;

        // FLOPS counter
        let flops_total = CounterVec::new(
            Opts::new("torsh_flops_total", "Total floating point operations"),
            &["operation"],
        )
        .map_err(|e| TorshError::operation_error(&format!("Failed to create counter: {}", e)))?;

        // Bytes transferred counter
        let bytes_transferred = CounterVec::new(
            Opts::new("torsh_bytes_transferred_total", "Total bytes transferred"),
            &["operation", "direction"],
        )
        .map_err(|e| TorshError::operation_error(&format!("Failed to create counter: {}", e)))?;

        // Active operations gauge
        let active_operations = GaugeVec::new(
            Opts::new("torsh_active_operations", "Number of active operations"),
            &["operation"],
        )
        .map_err(|e| TorshError::operation_error(&format!("Failed to create gauge: {}", e)))?;

        // Profiling overhead histogram
        let profiling_overhead = Histogram::with_opts(
            HistogramOpts::new(
                "torsh_profiling_overhead_microseconds",
                "Profiling overhead in microseconds",
            )
            .buckets(vec![1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]),
        )
        .map_err(|e| TorshError::operation_error(&format!("Failed to create histogram: {}", e)))?;

        // Thread activity gauge
        let thread_activity = GaugeVec::new(
            Opts::new("torsh_thread_activity", "Thread activity metrics"),
            &["thread_id", "metric"],
        )
        .map_err(|e| TorshError::operation_error(&format!("Failed to create gauge: {}", e)))?;

        // Register all metrics
        registry
            .register(Box::new(operation_duration.clone()))
            .map_err(|e| {
                TorshError::operation_error(&format!("Failed to register metric: {}", e))
            })?;
        registry
            .register(Box::new(operation_count.clone()))
            .map_err(|e| {
                TorshError::operation_error(&format!("Failed to register metric: {}", e))
            })?;
        registry
            .register(Box::new(memory_allocated.clone()))
            .map_err(|e| {
                TorshError::operation_error(&format!("Failed to register metric: {}", e))
            })?;
        registry
            .register(Box::new(memory_deallocated.clone()))
            .map_err(|e| {
                TorshError::operation_error(&format!("Failed to register metric: {}", e))
            })?;
        registry
            .register(Box::new(flops_total.clone()))
            .map_err(|e| {
                TorshError::operation_error(&format!("Failed to register metric: {}", e))
            })?;
        registry
            .register(Box::new(bytes_transferred.clone()))
            .map_err(|e| {
                TorshError::operation_error(&format!("Failed to register metric: {}", e))
            })?;
        registry
            .register(Box::new(active_operations.clone()))
            .map_err(|e| {
                TorshError::operation_error(&format!("Failed to register metric: {}", e))
            })?;
        registry
            .register(Box::new(profiling_overhead.clone()))
            .map_err(|e| {
                TorshError::operation_error(&format!("Failed to register metric: {}", e))
            })?;
        registry
            .register(Box::new(thread_activity.clone()))
            .map_err(|e| {
                TorshError::operation_error(&format!("Failed to register metric: {}", e))
            })?;

        Ok(Self {
            registry,
            operation_duration,
            operation_count,
            memory_allocated,
            memory_deallocated,
            flops_total,
            bytes_transferred,
            active_operations,
            profiling_overhead,
            thread_activity,
            custom_metrics: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// Update metrics from profiling events
    pub fn update_from_events(&self, events: &[ProfileEvent]) -> TorshResult<()> {
        for event in events {
            let thread_id = event.thread_id.to_string();

            // Update operation duration
            self.operation_duration
                .with_label_values(&[&event.name, &thread_id])
                .observe(event.duration_us as f64);

            // Update operation count
            self.operation_count.with_label_values(&[&event.name]).inc();

            // Update FLOPS if available
            if let Some(flops) = event.flops {
                self.flops_total
                    .with_label_values(&[&event.name])
                    .inc_by(flops as f64);
            }

            // Update bytes transferred if available
            if let Some(bytes) = event.bytes_transferred {
                // Assume "data" direction for now - could be enhanced with direction metadata
                self.bytes_transferred
                    .with_label_values(&[&event.name, "data"])
                    .inc_by(bytes as f64);
            }

            // Update thread activity
            self.thread_activity
                .with_label_values(&[&thread_id, "operations"])
                .inc();
        }

        Ok(())
    }

    /// Update metrics from the global profiler
    pub fn update_from_profiler(&self, profiler: &Profiler) -> TorshResult<()> {
        let events = profiler.events();
        self.update_from_events(&events)
    }

    /// Export metrics in Prometheus text format
    pub fn export_text(&self) -> TorshResult<String> {
        let encoder = TextEncoder::new();
        let metric_families = self.registry.gather();

        let mut buffer = Vec::new();
        encoder.encode(&metric_families, &mut buffer).map_err(|e| {
            TorshError::operation_error(&format!("Failed to encode metrics: {}", e))
        })?;

        String::from_utf8(buffer).map_err(|e| {
            TorshError::operation_error(&format!("Failed to convert to string: {}", e))
        })
    }

    /// Export metrics as bytes (for HTTP response)
    pub fn export_bytes(&self) -> TorshResult<Vec<u8>> {
        let encoder = TextEncoder::new();
        let metric_families = self.registry.gather();

        let mut buffer = Vec::new();
        encoder.encode(&metric_families, &mut buffer).map_err(|e| {
            TorshError::operation_error(&format!("Failed to encode metrics: {}", e))
        })?;

        Ok(buffer)
    }

    /// Get the underlying registry for custom metrics
    pub fn registry(&self) -> &Registry {
        &self.registry
    }

    /// Record profiling overhead
    pub fn record_overhead(&self, overhead_us: f64) {
        self.profiling_overhead.observe(overhead_us);
    }

    /// Set memory allocation for an operation
    pub fn set_memory_allocated(&self, operation: &str, bytes: f64) {
        self.memory_allocated
            .with_label_values(&[operation])
            .set(bytes);
    }

    /// Set memory deallocation for an operation
    pub fn set_memory_deallocated(&self, operation: &str, bytes: f64) {
        self.memory_deallocated
            .with_label_values(&[operation])
            .set(bytes);
    }

    /// Set active operations count
    pub fn set_active_operations(&self, operation: &str, count: f64) {
        self.active_operations
            .with_label_values(&[operation])
            .set(count);
    }

    /// Create a custom counter metric
    pub fn create_counter(
        &self,
        name: &str,
        help: &str,
        labels: &[&str],
    ) -> TorshResult<CounterVec> {
        let counter = CounterVec::new(Opts::new(name, help), labels).map_err(|e| {
            TorshError::operation_error(&format!("Failed to create counter: {}", e))
        })?;

        self.registry
            .register(Box::new(counter.clone()))
            .map_err(|e| {
                TorshError::operation_error(&format!("Failed to register counter: {}", e))
            })?;

        Ok(counter)
    }

    /// Create a custom gauge metric
    pub fn create_gauge(&self, name: &str, help: &str, labels: &[&str]) -> TorshResult<GaugeVec> {
        let gauge = GaugeVec::new(Opts::new(name, help), labels)
            .map_err(|e| TorshError::operation_error(&format!("Failed to create gauge: {}", e)))?;

        self.registry
            .register(Box::new(gauge.clone()))
            .map_err(|e| {
                TorshError::operation_error(&format!("Failed to register gauge: {}", e))
            })?;

        Ok(gauge)
    }

    /// Create a custom histogram metric
    pub fn create_histogram(
        &self,
        name: &str,
        help: &str,
        labels: &[&str],
        buckets: Vec<f64>,
    ) -> TorshResult<HistogramVec> {
        let histogram = HistogramVec::new(HistogramOpts::new(name, help).buckets(buckets), labels)
            .map_err(|e| {
                TorshError::operation_error(&format!("Failed to create histogram: {}", e))
            })?;

        self.registry
            .register(Box::new(histogram.clone()))
            .map_err(|e| {
                TorshError::operation_error(&format!("Failed to register histogram: {}", e))
            })?;

        Ok(histogram)
    }
}

impl Default for PrometheusExporter {
    fn default() -> Self {
        Self::new().expect("Failed to create default PrometheusExporter")
    }
}

/// Builder for Prometheus exporter configuration
pub struct PrometheusExporterBuilder {
    registry: Option<Registry>,
    custom_buckets: Option<Vec<f64>>,
    namespace: Option<String>,
}

impl PrometheusExporterBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            registry: None,
            custom_buckets: None,
            namespace: None,
        }
    }

    /// Set a custom registry
    pub fn with_registry(mut self, registry: Registry) -> Self {
        self.registry = Some(registry);
        self
    }

    /// Set custom histogram buckets
    pub fn with_buckets(mut self, buckets: Vec<f64>) -> Self {
        self.custom_buckets = Some(buckets);
        self
    }

    /// Set namespace for metrics
    pub fn with_namespace(mut self, namespace: String) -> Self {
        self.namespace = Some(namespace);
        self
    }

    /// Build the exporter
    pub fn build(self) -> TorshResult<PrometheusExporter> {
        let registry = self.registry.unwrap_or_else(Registry::new);
        PrometheusExporter::with_registry(registry)
    }
}

impl Default for PrometheusExporterBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ProfileEvent;

    #[test]
    fn test_prometheus_exporter_creation() {
        let exporter = PrometheusExporter::new();
        assert!(exporter.is_ok());
    }

    #[test]
    fn test_update_from_events() {
        let exporter = PrometheusExporter::new().unwrap();

        let events = vec![
            ProfileEvent {
                name: "test_op".to_string(),
                category: "compute".to_string(),
                start_us: 1000,
                duration_us: 500,
                thread_id: 1,
                operation_count: Some(100),
                flops: Some(1000),
                bytes_transferred: Some(2048),
                stack_trace: None,
            },
            ProfileEvent {
                name: "test_op".to_string(),
                category: "compute".to_string(),
                start_us: 2000,
                duration_us: 300,
                thread_id: 1,
                operation_count: Some(50),
                flops: Some(500),
                bytes_transferred: Some(1024),
                stack_trace: None,
            },
        ];

        let result = exporter.update_from_events(&events);
        assert!(result.is_ok());
    }

    #[test]
    fn test_export_text() {
        let exporter = PrometheusExporter::new().unwrap();

        let events = vec![ProfileEvent {
            name: "matrix_multiply".to_string(),
            category: "compute".to_string(),
            start_us: 1000,
            duration_us: 1500,
            thread_id: 1,
            operation_count: Some(1000),
            flops: Some(1000000),
            bytes_transferred: Some(8192),
            stack_trace: None,
        }];

        exporter.update_from_events(&events).unwrap();

        let text = exporter.export_text().unwrap();
        assert!(text.contains("torsh_operation_duration_microseconds"));
        assert!(text.contains("torsh_operation_total"));
        assert!(text.contains("matrix_multiply"));
    }

    #[test]
    fn test_custom_metrics() {
        let exporter = PrometheusExporter::new().unwrap();

        let counter = exporter
            .create_counter("custom_counter", "Custom counter metric", &["label1"])
            .unwrap();

        counter.with_label_values(&["value1"]).inc();

        let text = exporter.export_text().unwrap();
        assert!(text.contains("custom_counter"));
    }

    #[test]
    fn test_memory_metrics() {
        let exporter = PrometheusExporter::new().unwrap();

        exporter.set_memory_allocated("test_op", 1024.0);
        exporter.set_memory_deallocated("test_op", 512.0);

        let text = exporter.export_text().unwrap();
        assert!(text.contains("torsh_memory_allocated_bytes"));
        assert!(text.contains("torsh_memory_deallocated_bytes"));
    }

    #[test]
    fn test_builder_pattern() {
        let exporter = PrometheusExporterBuilder::new()
            .with_buckets(vec![1.0, 10.0, 100.0])
            .build();

        assert!(exporter.is_ok());
    }
}
