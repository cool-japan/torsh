//! AWS CloudWatch metrics integration for torsh-profiler
//!
//! This module provides AWS CloudWatch metrics publishing functionality,
//! allowing profiling data to be sent to CloudWatch for monitoring and alerting.

use crate::{ProfileEvent, TorshError, TorshResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::SystemTime;

/// AWS CloudWatch metrics publisher
pub struct CloudWatchPublisher {
    namespace: String,
    region: Option<String>,
    dimensions: Vec<Dimension>,
    metric_buffer: Vec<MetricDatum>,
    max_buffer_size: usize,
}

/// CloudWatch metric datum
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricDatum {
    pub metric_name: String,
    pub value: f64,
    pub unit: Unit,
    pub timestamp: SystemTime,
    pub dimensions: Vec<Dimension>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub statistic_values: Option<StatisticSet>,
}

/// CloudWatch dimension for metric filtering
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Dimension {
    pub name: String,
    pub value: String,
}

/// CloudWatch metric unit
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Unit {
    Seconds,
    Microseconds,
    Milliseconds,
    Bytes,
    Kilobytes,
    Megabytes,
    Gigabytes,
    Bits,
    Kilobits,
    Megabits,
    Gigabits,
    Percent,
    Count,
    #[serde(rename = "Bytes/Second")]
    BytesPerSecond,
    #[serde(rename = "Bits/Second")]
    BitsPerSecond,
    #[serde(rename = "Count/Second")]
    CountPerSecond,
    None,
}

/// Statistical values for aggregated metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticSet {
    pub sample_count: f64,
    pub sum: f64,
    pub minimum: f64,
    pub maximum: f64,
}

/// CloudWatch publisher configuration
#[derive(Debug, Clone)]
pub struct CloudWatchConfig {
    pub namespace: String,
    pub region: Option<String>,
    pub default_dimensions: Vec<Dimension>,
    pub buffer_size: usize,
    pub enable_aggregation: bool,
}

impl Default for CloudWatchConfig {
    fn default() -> Self {
        Self {
            namespace: "ToRSh/Profiling".to_string(),
            region: None,
            default_dimensions: Vec::new(),
            buffer_size: 20, // CloudWatch limit is 20 metrics per PutMetricData call
            enable_aggregation: false,
        }
    }
}

impl CloudWatchPublisher {
    /// Create a new CloudWatch publisher with default configuration
    pub fn new(namespace: &str) -> Self {
        Self::with_config(CloudWatchConfig {
            namespace: namespace.to_string(),
            ..Default::default()
        })
    }

    /// Create a new CloudWatch publisher with custom configuration
    pub fn with_config(config: CloudWatchConfig) -> Self {
        Self {
            namespace: config.namespace,
            region: config.region,
            dimensions: config.default_dimensions,
            metric_buffer: Vec::with_capacity(config.buffer_size),
            max_buffer_size: config.buffer_size,
        }
    }

    /// Set AWS region
    pub fn with_region(mut self, region: &str) -> Self {
        self.region = Some(region.to_string());
        self
    }

    /// Add a default dimension (applied to all metrics)
    pub fn add_dimension(&mut self, name: &str, value: &str) -> &mut Self {
        self.dimensions.push(Dimension {
            name: name.to_string(),
            value: value.to_string(),
        });
        self
    }

    /// Publish a single metric
    pub fn put_metric(
        &mut self,
        metric_name: &str,
        value: f64,
        unit: Unit,
        dimensions: Vec<Dimension>,
    ) -> TorshResult<()> {
        let mut all_dimensions = self.dimensions.clone();
        all_dimensions.extend(dimensions);

        let datum = MetricDatum {
            metric_name: metric_name.to_string(),
            value,
            unit,
            timestamp: SystemTime::now(),
            dimensions: all_dimensions,
            statistic_values: None,
        };

        self.metric_buffer.push(datum);

        // Auto-flush if buffer is full
        if self.metric_buffer.len() >= self.max_buffer_size {
            self.flush()?;
        }

        Ok(())
    }

    /// Publish a metric with statistics
    pub fn put_metric_statistics(
        &mut self,
        metric_name: &str,
        statistics: StatisticSet,
        unit: Unit,
        dimensions: Vec<Dimension>,
    ) -> TorshResult<()> {
        let mut all_dimensions = self.dimensions.clone();
        all_dimensions.extend(dimensions);

        let datum = MetricDatum {
            metric_name: metric_name.to_string(),
            value: statistics.sum / statistics.sample_count, // Average as the primary value
            unit,
            timestamp: SystemTime::now(),
            dimensions: all_dimensions,
            statistic_values: Some(statistics),
        };

        self.metric_buffer.push(datum);

        // Auto-flush if buffer is full
        if self.metric_buffer.len() >= self.max_buffer_size {
            self.flush()?;
        }

        Ok(())
    }

    /// Convert profiling events to CloudWatch metrics
    pub fn publish_from_events(&mut self, events: &[ProfileEvent]) -> TorshResult<()> {
        // Group events by operation for aggregation
        let mut event_groups: HashMap<String, Vec<&ProfileEvent>> = HashMap::new();

        for event in events {
            event_groups
                .entry(event.name.clone())
                .or_default()
                .push(event);
        }

        // Publish aggregated metrics for each operation
        for (operation, op_events) in event_groups {
            // Duration statistics
            let durations: Vec<f64> = op_events.iter().map(|e| e.duration_us as f64).collect();
            if !durations.is_empty() {
                let stats = calculate_statistics(&durations);
                self.put_metric_statistics(
                    "OperationDuration",
                    stats,
                    Unit::Microseconds,
                    vec![Dimension {
                        name: "Operation".to_string(),
                        value: operation.clone(),
                    }],
                )?;
            }

            // Operation count
            self.put_metric(
                "OperationCount",
                op_events.len() as f64,
                Unit::Count,
                vec![Dimension {
                    name: "Operation".to_string(),
                    value: operation.clone(),
                }],
            )?;

            // FLOPS if available
            let flops_sum: u64 = op_events.iter().filter_map(|e| e.flops).sum();
            if flops_sum > 0 {
                self.put_metric(
                    "FLOPS",
                    flops_sum as f64,
                    Unit::Count,
                    vec![Dimension {
                        name: "Operation".to_string(),
                        value: operation.clone(),
                    }],
                )?;
            }

            // Bytes transferred if available
            let bytes_sum: u64 = op_events.iter().filter_map(|e| e.bytes_transferred).sum();
            if bytes_sum > 0 {
                self.put_metric(
                    "BytesTransferred",
                    bytes_sum as f64,
                    Unit::Bytes,
                    vec![Dimension {
                        name: "Operation".to_string(),
                        value: operation.clone(),
                    }],
                )?;
            }
        }

        Ok(())
    }

    /// Flush buffered metrics (simulated - would actually call AWS API)
    pub fn flush(&mut self) -> TorshResult<()> {
        if self.metric_buffer.is_empty() {
            return Ok(());
        }

        // In a real implementation, this would call:
        // cloudwatch_client.put_metric_data()
        //   .namespace(&self.namespace)
        //   .metric_data(self.metric_buffer.clone())
        //   .send()
        //   .await?;

        // For now, we'll just clear the buffer and log
        let count = self.metric_buffer.len();
        self.metric_buffer.clear();

        // Log the flush (in production, this would be actual AWS API call)
        #[cfg(debug_assertions)]
        println!(
            "[CloudWatch] Flushed {} metrics to namespace: {}",
            count, self.namespace
        );

        Ok(())
    }

    /// Get the current buffer size
    pub fn buffer_size(&self) -> usize {
        self.metric_buffer.len()
    }

    /// Get buffered metrics (for inspection/testing)
    pub fn get_buffered_metrics(&self) -> &[MetricDatum] {
        &self.metric_buffer
    }

    /// Export metrics as JSON (for debugging/inspection)
    pub fn export_json(&self) -> TorshResult<String> {
        #[derive(Serialize)]
        struct CloudWatchExport {
            namespace: String,
            region: Option<String>,
            metrics: Vec<MetricDatum>,
        }

        let export = CloudWatchExport {
            namespace: self.namespace.clone(),
            region: self.region.clone(),
            metrics: self.metric_buffer.clone(),
        };

        serde_json::to_string_pretty(&export).map_err(|e| {
            TorshError::operation_error(&format!("Failed to serialize metrics: {}", e))
        })
    }
}

/// Calculate statistics from a set of values
fn calculate_statistics(values: &[f64]) -> StatisticSet {
    let sample_count = values.len() as f64;
    let sum: f64 = values.iter().sum();
    let minimum = values.iter().copied().fold(f64::INFINITY, f64::min);
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    StatisticSet {
        sample_count,
        sum,
        minimum,
        maximum,
    }
}

/// Builder for CloudWatch publisher configuration
pub struct CloudWatchPublisherBuilder {
    config: CloudWatchConfig,
}

impl CloudWatchPublisherBuilder {
    /// Create a new builder
    pub fn new(namespace: &str) -> Self {
        Self {
            config: CloudWatchConfig {
                namespace: namespace.to_string(),
                ..Default::default()
            },
        }
    }

    /// Set AWS region
    pub fn region(mut self, region: &str) -> Self {
        self.config.region = Some(region.to_string());
        self
    }

    /// Add a default dimension
    pub fn dimension(mut self, name: &str, value: &str) -> Self {
        self.config.default_dimensions.push(Dimension {
            name: name.to_string(),
            value: value.to_string(),
        });
        self
    }

    /// Set buffer size (max 20 for CloudWatch)
    pub fn buffer_size(mut self, size: usize) -> Self {
        self.config.buffer_size = size.min(20);
        self
    }

    /// Enable metric aggregation
    pub fn enable_aggregation(mut self) -> Self {
        self.config.enable_aggregation = true;
        self
    }

    /// Build the CloudWatch publisher
    pub fn build(self) -> CloudWatchPublisher {
        CloudWatchPublisher::with_config(self.config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cloudwatch_publisher_creation() {
        let publisher = CloudWatchPublisher::new("TestNamespace");
        assert_eq!(publisher.namespace, "TestNamespace");
        assert_eq!(publisher.buffer_size(), 0);
    }

    #[test]
    fn test_put_metric() {
        let mut publisher = CloudWatchPublisher::new("Test");
        let result = publisher.put_metric(
            "TestMetric",
            42.0,
            Unit::Count,
            vec![Dimension {
                name: "Environment".to_string(),
                value: "test".to_string(),
            }],
        );
        assert!(result.is_ok());
        assert_eq!(publisher.buffer_size(), 1);
    }

    #[test]
    fn test_statistics() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = calculate_statistics(&values);
        assert_eq!(stats.sample_count, 5.0);
        assert_eq!(stats.sum, 15.0);
        assert_eq!(stats.minimum, 1.0);
        assert_eq!(stats.maximum, 5.0);
    }

    #[test]
    fn test_publish_from_events() {
        let mut publisher = CloudWatchPublisher::new("Test");
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
                duration_us: 600,
                thread_id: 1,
                operation_count: Some(150),
                flops: Some(1500),
                bytes_transferred: Some(3072),
                stack_trace: None,
            },
        ];

        let result = publisher.publish_from_events(&events);
        assert!(result.is_ok());
        assert!(publisher.buffer_size() > 0);
    }

    #[test]
    fn test_auto_flush() {
        let mut publisher = CloudWatchPublisherBuilder::new("Test")
            .buffer_size(3)
            .build();

        // Add 3 metrics - should auto-flush
        for i in 0..3 {
            publisher
                .put_metric(&format!("Metric{}", i), i as f64, Unit::Count, vec![])
                .unwrap();
        }

        // Buffer should be empty after auto-flush
        assert_eq!(publisher.buffer_size(), 0);
    }

    #[test]
    fn test_builder_pattern() {
        let publisher = CloudWatchPublisherBuilder::new("ToRSh/Test")
            .region("us-west-2")
            .dimension("Environment", "production")
            .dimension("Application", "deep-learning")
            .buffer_size(15)
            .build();

        assert_eq!(publisher.namespace, "ToRSh/Test");
        assert_eq!(publisher.region, Some("us-west-2".to_string()));
        assert_eq!(publisher.dimensions.len(), 2);
    }

    #[test]
    fn test_export_json() {
        let mut publisher = CloudWatchPublisher::new("Test");
        publisher
            .put_metric("TestMetric", 100.0, Unit::Count, vec![])
            .unwrap();

        let json = publisher.export_json();
        assert!(json.is_ok());
        assert!(json.unwrap().contains("TestMetric"));
    }
}
