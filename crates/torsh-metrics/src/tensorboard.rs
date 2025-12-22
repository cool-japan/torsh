//! TensorBoard integration for metric logging
//!
//! This module provides utilities for logging metrics to TensorBoard-compatible
//! event files for visualization and tracking during training.

use crate::{Metric, MetricCollection};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{create_dir_all, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use torsh_core::error::TorshError;
use torsh_tensor::Tensor;

/// TensorBoard event writer for logging metrics
pub struct TensorBoardWriter {
    log_dir: PathBuf,
    run_name: String,
    global_step: u64,
    events: Vec<Event>,
    file_handle: Option<File>,
}

impl TensorBoardWriter {
    /// Create a new TensorBoard writer
    pub fn new(log_dir: impl AsRef<Path>, run_name: impl Into<String>) -> Result<Self, TorshError> {
        let log_dir = log_dir.as_ref().to_path_buf();
        let run_name = run_name.into();

        // Create log directory if it doesn't exist
        create_dir_all(&log_dir).map_err(|e| {
            TorshError::InvalidArgument(format!("Failed to create log directory: {}", e))
        })?;

        Ok(Self {
            log_dir,
            run_name,
            global_step: 0,
            events: Vec::new(),
            file_handle: None,
        })
    }

    /// Log a scalar metric
    pub fn add_scalar(
        &mut self,
        tag: impl Into<String>,
        value: f64,
        step: Option<u64>,
    ) -> Result<(), TorshError> {
        let step = step.unwrap_or(self.global_step);
        let event = Event::Scalar {
            tag: tag.into(),
            value,
            step,
            wall_time: current_timestamp(),
        };

        self.events.push(event.clone());
        self.write_event_to_file(&event)?;

        Ok(())
    }

    /// Log multiple scalars at once
    pub fn add_scalars(
        &mut self,
        main_tag: impl Into<String>,
        tag_scalar_dict: HashMap<String, f64>,
        step: Option<u64>,
    ) -> Result<(), TorshError> {
        let step = step.unwrap_or(self.global_step);
        let main_tag = main_tag.into();

        for (tag, value) in tag_scalar_dict {
            self.add_scalar(format!("{}/{}", main_tag, tag), value, Some(step))?;
        }

        Ok(())
    }

    /// Log a histogram of values
    pub fn add_histogram(
        &mut self,
        tag: impl Into<String>,
        values: &[f64],
        step: Option<u64>,
    ) -> Result<(), TorshError> {
        let step = step.unwrap_or(self.global_step);
        let histogram = compute_histogram(values);

        let event = Event::Histogram {
            tag: tag.into(),
            histogram,
            step,
            wall_time: current_timestamp(),
        };

        self.events.push(event.clone());
        self.write_event_to_file(&event)?;

        Ok(())
    }

    /// Log metrics from a MetricCollection
    pub fn add_metric_collection(
        &mut self,
        collection: &mut MetricCollection,
        predictions: &Tensor,
        targets: &Tensor,
        step: Option<u64>,
    ) -> Result<(), TorshError> {
        let results = collection.compute(predictions, targets);

        for (name, value) in results {
            self.add_scalar(name, value, step)?;
        }

        Ok(())
    }

    /// Log a single metric
    pub fn add_metric<M: Metric>(
        &mut self,
        metric: &M,
        predictions: &Tensor,
        targets: &Tensor,
        step: Option<u64>,
    ) -> Result<(), TorshError> {
        let value = metric.compute(predictions, targets);
        self.add_scalar(metric.name(), value, step)
    }

    /// Add a text summary
    pub fn add_text(
        &mut self,
        tag: impl Into<String>,
        text: impl Into<String>,
        step: Option<u64>,
    ) -> Result<(), TorshError> {
        let step = step.unwrap_or(self.global_step);
        let event = Event::Text {
            tag: tag.into(),
            text: text.into(),
            step,
            wall_time: current_timestamp(),
        };

        self.events.push(event.clone());
        self.write_event_to_file(&event)?;

        Ok(())
    }

    /// Increment the global step counter
    pub fn increment_step(&mut self) {
        self.global_step += 1;
    }

    /// Set the global step counter
    pub fn set_step(&mut self, step: u64) {
        self.global_step = step;
    }

    /// Get current global step
    pub fn get_step(&self) -> u64 {
        self.global_step
    }

    /// Flush all pending events to disk
    pub fn flush(&mut self) -> Result<(), TorshError> {
        if let Some(ref mut file) = self.file_handle {
            file.flush().map_err(|e| {
                TorshError::InvalidArgument(format!("Failed to flush events: {}", e))
            })?;
        }
        Ok(())
    }

    /// Close the writer and flush remaining events
    pub fn close(&mut self) -> Result<(), TorshError> {
        self.flush()?;
        self.file_handle = None;
        Ok(())
    }

    /// Write event to file (JSON lines format for simplicity)
    fn write_event_to_file(&mut self, event: &Event) -> Result<(), TorshError> {
        if self.file_handle.is_none() {
            let filename = format!("events_{}.jsonl", self.run_name);
            let path = self.log_dir.join(filename);
            let file = File::create(path).map_err(|e| {
                TorshError::InvalidArgument(format!("Failed to create event file: {}", e))
            })?;
            self.file_handle = Some(file);
        }

        if let Some(ref mut file) = self.file_handle {
            let json = serde_json::to_string(event).map_err(|e| {
                TorshError::InvalidArgument(format!("Failed to serialize event: {}", e))
            })?;
            writeln!(file, "{}", json).map_err(|e| {
                TorshError::InvalidArgument(format!("Failed to write event: {}", e))
            })?;
        }

        Ok(())
    }
}

impl Drop for TensorBoardWriter {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

/// TensorBoard event types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
enum Event {
    Scalar {
        tag: String,
        value: f64,
        step: u64,
        wall_time: f64,
    },
    Histogram {
        tag: String,
        histogram: HistogramData,
        step: u64,
        wall_time: f64,
    },
    Text {
        tag: String,
        text: String,
        step: u64,
        wall_time: f64,
    },
}

/// Histogram data
#[derive(Debug, Clone, Serialize, Deserialize)]
struct HistogramData {
    min: f64,
    max: f64,
    mean: f64,
    std: f64,
    count: usize,
    buckets: Vec<HistogramBucket>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HistogramBucket {
    left: f64,
    right: f64,
    count: usize,
}

/// Compute histogram statistics
fn compute_histogram(values: &[f64]) -> HistogramData {
    if values.is_empty() {
        return HistogramData {
            min: 0.0,
            max: 0.0,
            mean: 0.0,
            std: 0.0,
            count: 0,
            buckets: Vec::new(),
        };
    }

    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let min = sorted[0];
    let max = sorted[sorted.len() - 1];
    let mean = sorted.iter().sum::<f64>() / sorted.len() as f64;
    let variance = sorted.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / sorted.len() as f64;
    let std = variance.sqrt();

    // Create 30 buckets
    let num_buckets = 30.min(sorted.len());
    let mut buckets = Vec::with_capacity(num_buckets);

    if max > min {
        let bucket_width = (max - min) / num_buckets as f64;

        for i in 0..num_buckets {
            let left = min + i as f64 * bucket_width;
            let right = min + (i + 1) as f64 * bucket_width;

            let count = sorted
                .iter()
                .filter(|&&x| x >= left && (i == num_buckets - 1 || x < right))
                .count();

            buckets.push(HistogramBucket { left, right, count });
        }
    }

    HistogramData {
        min,
        max,
        mean,
        std,
        count: values.len(),
        buckets,
    }
}

/// Get current Unix timestamp
fn current_timestamp() -> f64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs_f64()
}

/// Metric logger that automatically logs to TensorBoard
pub struct MetricLogger {
    writer: TensorBoardWriter,
    prefix: String,
}

impl MetricLogger {
    /// Create a new metric logger
    pub fn new(
        log_dir: impl AsRef<Path>,
        run_name: impl Into<String>,
        prefix: impl Into<String>,
    ) -> Result<Self, TorshError> {
        Ok(Self {
            writer: TensorBoardWriter::new(log_dir, run_name)?,
            prefix: prefix.into(),
        })
    }

    /// Log training metrics
    pub fn log_train_metrics(
        &mut self,
        metrics: HashMap<String, f64>,
        epoch: u64,
    ) -> Result<(), TorshError> {
        for (name, value) in metrics {
            let tag = format!("{}/train/{}", self.prefix, name);
            self.writer.add_scalar(tag, value, Some(epoch))?;
        }
        Ok(())
    }

    /// Log validation metrics
    pub fn log_val_metrics(
        &mut self,
        metrics: HashMap<String, f64>,
        epoch: u64,
    ) -> Result<(), TorshError> {
        for (name, value) in metrics {
            let tag = format!("{}/val/{}", self.prefix, name);
            self.writer.add_scalar(tag, value, Some(epoch))?;
        }
        Ok(())
    }

    /// Log test metrics
    pub fn log_test_metrics(&mut self, metrics: HashMap<String, f64>) -> Result<(), TorshError> {
        for (name, value) in metrics {
            let tag = format!("{}/test/{}", self.prefix, name);
            self.writer.add_scalar(tag, value, None)?;
        }
        Ok(())
    }

    /// Log learning rate
    pub fn log_lr(&mut self, lr: f64, step: u64) -> Result<(), TorshError> {
        self.writer
            .add_scalar(format!("{}/learning_rate", self.prefix), lr, Some(step))
    }

    /// Log loss
    pub fn log_loss(&mut self, loss: f64, step: u64) -> Result<(), TorshError> {
        self.writer
            .add_scalar(format!("{}/loss", self.prefix), loss, Some(step))
    }

    /// Flush pending events
    pub fn flush(&mut self) -> Result<(), TorshError> {
        self.writer.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env::temp_dir;

    #[test]
    fn test_tensorboard_writer_creation() {
        let log_dir = temp_dir().join("torsh_metrics_tb_test");
        let writer = TensorBoardWriter::new(&log_dir, "test_run");
        assert!(writer.is_ok());
    }

    #[test]
    fn test_add_scalar() {
        let log_dir = temp_dir().join("torsh_metrics_tb_scalar");
        let mut writer = TensorBoardWriter::new(&log_dir, "test_run").unwrap();

        let result = writer.add_scalar("accuracy", 0.95, Some(0));
        assert!(result.is_ok());

        let result = writer.add_scalar("loss", 0.05, Some(1));
        assert!(result.is_ok());
    }

    #[test]
    fn test_add_scalars() {
        let log_dir = temp_dir().join("torsh_metrics_tb_scalars");
        let mut writer = TensorBoardWriter::new(&log_dir, "test_run").unwrap();

        let mut metrics = HashMap::new();
        metrics.insert("accuracy".to_string(), 0.95);
        metrics.insert("precision".to_string(), 0.92);

        let result = writer.add_scalars("metrics", metrics, Some(0));
        assert!(result.is_ok());
    }

    #[test]
    fn test_add_histogram() {
        let log_dir = temp_dir().join("torsh_metrics_tb_histogram");
        let mut writer = TensorBoardWriter::new(&log_dir, "test_run").unwrap();

        let values: Vec<f64> = (0..100).map(|i| i as f64 * 0.01).collect();
        let result = writer.add_histogram("weights", &values, Some(0));
        assert!(result.is_ok());
    }

    #[test]
    fn test_histogram_computation() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let histogram = compute_histogram(&values);

        assert_eq!(histogram.min, 1.0);
        assert_eq!(histogram.max, 5.0);
        assert_eq!(histogram.mean, 3.0);
        assert_eq!(histogram.count, 5);
        assert!(!histogram.buckets.is_empty());
    }

    #[test]
    fn test_metric_logger() {
        let log_dir = temp_dir().join("torsh_metrics_logger");
        let mut logger = MetricLogger::new(&log_dir, "test_run", "model").unwrap();

        let mut metrics = HashMap::new();
        metrics.insert("accuracy".to_string(), 0.95);

        let result = logger.log_train_metrics(metrics.clone(), 0);
        assert!(result.is_ok());

        let result = logger.log_val_metrics(metrics, 0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_step_management() {
        let log_dir = temp_dir().join("torsh_metrics_steps");
        let mut writer = TensorBoardWriter::new(&log_dir, "test_run").unwrap();

        assert_eq!(writer.get_step(), 0);

        writer.increment_step();
        assert_eq!(writer.get_step(), 1);

        writer.set_step(10);
        assert_eq!(writer.get_step(), 10);
    }

    #[test]
    fn test_empty_histogram() {
        let values: Vec<f64> = vec![];
        let histogram = compute_histogram(&values);

        assert_eq!(histogram.count, 0);
        assert_eq!(histogram.min, 0.0);
        assert_eq!(histogram.max, 0.0);
    }
}
