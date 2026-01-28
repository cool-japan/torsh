//! Weights & Biases (W&B) integration for experiment tracking
//!
//! This module provides utilities for logging experiments, metrics, parameters,
//! and artifacts to W&B for visualization and comparison.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{create_dir_all, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use torsh_core::error::TorshError;

/// Weights & Biases client for experiment tracking
pub struct WandbClient {
    #[allow(dead_code)] // Stored for reference and future HTTP client implementation
    api_key: Option<String>,
    project_name: String,
    entity: Option<String>,
    run_id: Option<String>,
    run_name: Option<String>,
    config: HashMap<String, serde_json::Value>,
    summary: HashMap<String, f64>,
    logged_metrics: Vec<LogEntry>,
    artifact_dir: PathBuf,
    tags: Vec<String>,
    notes: Option<String>,
}

impl WandbClient {
    /// Create a new W&B client
    pub fn new(project_name: impl Into<String>) -> Self {
        let project_name = project_name.into();
        let artifact_dir = PathBuf::from("/tmp/wandb-artifacts").join(&project_name);

        Self {
            api_key: None,
            project_name,
            entity: None,
            run_id: None,
            run_name: None,
            config: HashMap::new(),
            summary: HashMap::new(),
            logged_metrics: Vec::new(),
            artifact_dir,
            tags: Vec::new(),
            notes: None,
        }
    }

    /// Set the API key
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Set the entity (username or team name)
    pub fn with_entity(mut self, entity: impl Into<String>) -> Self {
        self.entity = Some(entity.into());
        self
    }

    /// Initialize a new run
    pub fn init(
        &mut self,
        run_name: Option<String>,
        config: Option<HashMap<String, serde_json::Value>>,
        tags: Option<Vec<String>>,
        notes: Option<String>,
    ) -> Result<String, TorshError> {
        let run_id = generate_run_id();
        self.run_id = Some(run_id.clone());
        self.run_name = run_name;

        if let Some(cfg) = config {
            self.config = cfg;
        }

        if let Some(t) = tags {
            self.tags = t;
        }

        self.notes = notes;

        // Create artifact directory
        create_dir_all(&self.artifact_dir).map_err(|e| {
            TorshError::InvalidArgument(format!("Failed to create artifact directory: {}", e))
        })?;

        Ok(run_id)
    }

    /// Log a single metric value
    pub fn log(
        &mut self,
        metrics: HashMap<String, f64>,
        step: Option<usize>,
    ) -> Result<(), TorshError> {
        if self.run_id.is_none() {
            return Err(TorshError::InvalidArgument(
                "No active run. Call init() first.".to_string(),
            ));
        }

        let entry = LogEntry {
            step: step.unwrap_or(self.logged_metrics.len()),
            metrics: metrics.clone(),
            timestamp: current_timestamp(),
        };

        self.logged_metrics.push(entry);

        // Update summary with latest values
        for (key, value) in metrics {
            self.summary.insert(key, value);
        }

        Ok(())
    }

    /// Log multiple metrics at once
    pub fn log_metrics(
        &mut self,
        metrics: HashMap<String, f64>,
        step: Option<usize>,
    ) -> Result<(), TorshError> {
        self.log(metrics, step)
    }

    /// Update the run config
    pub fn config_update(
        &mut self,
        config: HashMap<String, serde_json::Value>,
    ) -> Result<(), TorshError> {
        if self.run_id.is_none() {
            return Err(TorshError::InvalidArgument("No active run".to_string()));
        }

        self.config.extend(config);
        Ok(())
    }

    /// Update the run summary
    pub fn summary_update(&mut self, summary: HashMap<String, f64>) -> Result<(), TorshError> {
        if self.run_id.is_none() {
            return Err(TorshError::InvalidArgument("No active run".to_string()));
        }

        self.summary.extend(summary);
        Ok(())
    }

    /// Log a table of data
    pub fn log_table(
        &mut self,
        table_name: impl Into<String>,
        columns: Vec<String>,
        data: Vec<Vec<String>>,
    ) -> Result<(), TorshError> {
        if self.run_id.is_none() {
            return Err(TorshError::InvalidArgument("No active run".to_string()));
        }

        let table = WandbTable {
            name: table_name.into(),
            columns,
            data,
        };

        self.save_table(&table)?;
        Ok(())
    }

    /// Log a histogram
    pub fn log_histogram(
        &mut self,
        name: impl Into<String>,
        values: &[f64],
        step: Option<usize>,
    ) -> Result<(), TorshError> {
        if self.run_id.is_none() {
            return Err(TorshError::InvalidArgument("No active run".to_string()));
        }

        let histogram = compute_histogram(values, 64);
        let histogram_json = serde_json::to_string(&histogram).map_err(|e| {
            TorshError::InvalidArgument(format!("Failed to serialize histogram: {}", e))
        })?;

        // Save histogram data
        let histogram_path = self.artifact_dir.join(format!(
            "histogram_{}_{}.json",
            name.into(),
            step.unwrap_or(0)
        ));
        let mut file = File::create(histogram_path).map_err(|e| {
            TorshError::InvalidArgument(format!("Failed to create histogram file: {}", e))
        })?;
        file.write_all(histogram_json.as_bytes()).map_err(|e| {
            TorshError::InvalidArgument(format!("Failed to write histogram: {}", e))
        })?;

        Ok(())
    }

    /// Save an artifact (file or directory)
    pub fn save_artifact(
        &mut self,
        artifact_name: impl Into<String>,
        artifact_type: impl Into<String>,
        file_path: impl AsRef<Path>,
    ) -> Result<(), TorshError> {
        if self.run_id.is_none() {
            return Err(TorshError::InvalidArgument("No active run".to_string()));
        }

        let artifact_name = artifact_name.into();
        let _artifact_type = artifact_type.into();
        let file_path = file_path.as_ref();

        // Copy artifact to artifact directory
        let dest_path = self.artifact_dir.join(&artifact_name);
        std::fs::copy(file_path, dest_path)
            .map_err(|e| TorshError::InvalidArgument(format!("Failed to copy artifact: {}", e)))?;

        Ok(())
    }

    /// Add tags to the run
    pub fn add_tags(&mut self, tags: Vec<String>) -> Result<(), TorshError> {
        if self.run_id.is_none() {
            return Err(TorshError::InvalidArgument("No active run".to_string()));
        }

        self.tags.extend(tags);
        Ok(())
    }

    /// Finish the run and save all data
    pub fn finish(&mut self) -> Result<(), TorshError> {
        if self.run_id.is_none() {
            return Err(TorshError::InvalidArgument("No active run".to_string()));
        }

        self.save_run_data()?;
        self.run_id = None;

        Ok(())
    }

    /// Get the current run summary
    pub fn get_summary(&self) -> &HashMap<String, f64> {
        &self.summary
    }

    /// Get the current run config
    pub fn get_config(&self) -> &HashMap<String, serde_json::Value> {
        &self.config
    }

    /// Get all logged metrics
    pub fn get_history(&self) -> &[LogEntry] {
        &self.logged_metrics
    }

    // Private helper methods

    fn save_run_data(&self) -> Result<(), TorshError> {
        let run_data = RunData {
            run_id: self
                .run_id
                .clone()
                .expect("run_id should be set before saving run data"),
            run_name: self.run_name.clone(),
            project_name: self.project_name.clone(),
            entity: self.entity.clone(),
            config: self.config.clone(),
            summary: self.summary.clone(),
            tags: self.tags.clone(),
            notes: self.notes.clone(),
            metrics_count: self.logged_metrics.len(),
        };

        let json = serde_json::to_string_pretty(&run_data).map_err(|e| {
            TorshError::InvalidArgument(format!("Failed to serialize run data: {}", e))
        })?;

        let run_file = self.artifact_dir.join("run_data.json");
        let mut file = File::create(run_file).map_err(|e| {
            TorshError::InvalidArgument(format!("Failed to create run file: {}", e))
        })?;

        file.write_all(json.as_bytes())
            .map_err(|e| TorshError::InvalidArgument(format!("Failed to write run data: {}", e)))?;

        // Save metrics history
        let metrics_json = serde_json::to_string_pretty(&self.logged_metrics).map_err(|e| {
            TorshError::InvalidArgument(format!("Failed to serialize metrics: {}", e))
        })?;

        let metrics_file = self.artifact_dir.join("metrics_history.json");
        let mut file = File::create(metrics_file).map_err(|e| {
            TorshError::InvalidArgument(format!("Failed to create metrics file: {}", e))
        })?;

        file.write_all(metrics_json.as_bytes())
            .map_err(|e| TorshError::InvalidArgument(format!("Failed to write metrics: {}", e)))?;

        Ok(())
    }

    fn save_table(&self, table: &WandbTable) -> Result<(), TorshError> {
        let table_json = serde_json::to_string_pretty(table).map_err(|e| {
            TorshError::InvalidArgument(format!("Failed to serialize table: {}", e))
        })?;

        let table_file = self.artifact_dir.join(format!("table_{}.json", table.name));
        let mut file = File::create(table_file).map_err(|e| {
            TorshError::InvalidArgument(format!("Failed to create table file: {}", e))
        })?;

        file.write_all(table_json.as_bytes())
            .map_err(|e| TorshError::InvalidArgument(format!("Failed to write table: {}", e)))?;

        Ok(())
    }
}

/// A single log entry with metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    pub step: usize,
    pub metrics: HashMap<String, f64>,
    pub timestamp: u64,
}

/// Run data for W&B
#[derive(Debug, Clone, Serialize, Deserialize)]
struct RunData {
    run_id: String,
    run_name: Option<String>,
    project_name: String,
    entity: Option<String>,
    config: HashMap<String, serde_json::Value>,
    summary: HashMap<String, f64>,
    tags: Vec<String>,
    notes: Option<String>,
    metrics_count: usize,
}

/// W&B table structure
#[derive(Debug, Clone, Serialize, Deserialize)]
struct WandbTable {
    name: String,
    columns: Vec<String>,
    data: Vec<Vec<String>>,
}

/// Histogram data for W&B
#[derive(Debug, Clone, Serialize, Deserialize)]
struct HistogramData {
    bins: Vec<f64>,
    counts: Vec<usize>,
}

// Helper functions

fn generate_run_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time should be after UNIX_EPOCH")
        .as_secs();
    format!("run_{}", timestamp)
}

fn current_timestamp() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time should be after UNIX_EPOCH")
        .as_secs()
}

fn compute_histogram(values: &[f64], n_bins: usize) -> HistogramData {
    if values.is_empty() {
        return HistogramData {
            bins: vec![0.0; n_bins + 1],
            counts: vec![0; n_bins],
        };
    }

    let min = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    let bin_width = if max > min {
        (max - min) / n_bins as f64
    } else {
        1.0
    };

    let mut bins = Vec::with_capacity(n_bins + 1);
    for i in 0..=n_bins {
        bins.push(min + i as f64 * bin_width);
    }

    let mut counts = vec![0; n_bins];
    for &value in values {
        let bin_idx = if value >= max {
            n_bins - 1
        } else {
            ((value - min) / bin_width).floor() as usize
        };
        counts[bin_idx.min(n_bins - 1)] += 1;
    }

    HistogramData { bins, counts }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wandb_init() {
        let mut client = WandbClient::new("test_project");
        let run_id = client.init(Some("test_run".to_string()), None, None, None);
        assert!(run_id.is_ok());
        assert!(client.run_id.is_some());
    }

    #[test]
    fn test_wandb_log_metrics() {
        let mut client = WandbClient::new("test_project");
        client
            .init(Some("test_run".to_string()), None, None, None)
            .unwrap();

        let mut metrics = HashMap::new();
        metrics.insert("accuracy".to_string(), 0.95);
        metrics.insert("loss".to_string(), 0.05);

        let result = client.log(metrics.clone(), Some(0));
        assert!(result.is_ok());
        assert_eq!(client.logged_metrics.len(), 1);
        assert_eq!(client.summary.get("accuracy"), Some(&0.95));
    }

    #[test]
    fn test_wandb_config_update() {
        let mut client = WandbClient::new("test_project");
        client
            .init(Some("test_run".to_string()), None, None, None)
            .unwrap();

        let mut config = HashMap::new();
        config.insert("learning_rate".to_string(), serde_json::Value::from(0.001));
        config.insert("batch_size".to_string(), serde_json::Value::from(32));

        let result = client.config_update(config);
        assert!(result.is_ok());
        assert_eq!(client.config.len(), 2);
    }

    #[test]
    fn test_wandb_tags() {
        let mut client = WandbClient::new("test_project");
        client
            .init(Some("test_run".to_string()), None, None, None)
            .unwrap();

        let tags = vec!["baseline".to_string(), "experiment1".to_string()];
        let result = client.add_tags(tags);
        assert!(result.is_ok());
        assert_eq!(client.tags.len(), 2);
    }

    #[test]
    fn test_wandb_finish() {
        let mut client = WandbClient::new("test_project");
        client
            .init(Some("test_run".to_string()), None, None, None)
            .unwrap();

        let mut metrics = HashMap::new();
        metrics.insert("accuracy".to_string(), 0.95);
        client.log(metrics, Some(0)).unwrap();

        let result = client.finish();
        assert!(result.is_ok());
        assert!(client.run_id.is_none());
    }

    #[test]
    fn test_histogram_computation() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let histogram = compute_histogram(&values, 5);

        assert_eq!(histogram.bins.len(), 6);
        assert_eq!(histogram.counts.len(), 5);
        assert_eq!(histogram.counts.iter().sum::<usize>(), 10);
    }

    #[test]
    fn test_log_without_init() {
        let mut client = WandbClient::new("test_project");

        let mut metrics = HashMap::new();
        metrics.insert("accuracy".to_string(), 0.95);

        let result = client.log(metrics, Some(0));
        assert!(result.is_err());
    }
}
