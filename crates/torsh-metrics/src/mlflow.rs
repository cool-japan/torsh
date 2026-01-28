//! MLflow integration for experiment tracking
//!
//! This module provides utilities for logging experiments, metrics, parameters,
//! and artifacts to MLflow tracking server.

use crate::{Metric, MetricCollection};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{create_dir_all, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use torsh_core::error::TorshError;
use torsh_tensor::Tensor;

/// MLflow experiment tracker
pub struct MLflowClient {
    #[allow(dead_code)] // Stored for reference and future HTTP client implementation
    tracking_uri: String,
    experiment_name: String,
    run_id: Option<String>,
    run_name: Option<String>,
    artifact_location: PathBuf,
    logged_metrics: HashMap<String, Vec<MetricPoint>>,
    logged_params: HashMap<String, String>,
    tags: HashMap<String, String>,
}

impl MLflowClient {
    /// Create a new MLflow client
    pub fn new(tracking_uri: impl Into<String>, experiment_name: impl Into<String>) -> Self {
        let experiment_name = experiment_name.into();
        let artifact_location = PathBuf::from("/tmp/mlflow-artifacts")
            .join(&experiment_name)
            .join(format!("run_{}", generate_run_id()));

        Self {
            tracking_uri: tracking_uri.into(),
            experiment_name,
            run_id: None,
            run_name: None,
            artifact_location,
            logged_metrics: HashMap::new(),
            logged_params: HashMap::new(),
            tags: HashMap::new(),
        }
    }

    /// Start a new MLflow run
    pub fn start_run(&mut self, run_name: Option<String>) -> Result<String, TorshError> {
        let run_id = generate_run_id();
        self.run_id = Some(run_id.clone());
        self.run_name = run_name;

        // Create artifact directory
        create_dir_all(&self.artifact_location).map_err(|e| {
            TorshError::InvalidArgument(format!("Failed to create artifact directory: {}", e))
        })?;

        // Set default tags
        let run_name_str = self
            .run_name
            .clone()
            .unwrap_or_else(|| "unnamed_run".to_string());
        self.set_tag("mlflow.runName", run_name_str)?;
        self.set_tag("mlflow.source.type", "LOCAL")?;

        Ok(run_id)
    }

    /// End the current MLflow run
    pub fn end_run(&mut self) -> Result<(), TorshError> {
        if self.run_id.is_none() {
            return Err(TorshError::InvalidArgument("No active run".to_string()));
        }

        // Save run data to file
        self.save_run_data()?;

        self.run_id = None;
        Ok(())
    }

    /// Log a parameter
    pub fn log_param(
        &mut self,
        key: impl Into<String>,
        value: impl Into<String>,
    ) -> Result<(), TorshError> {
        if self.run_id.is_none() {
            return Err(TorshError::InvalidArgument("No active run".to_string()));
        }

        self.logged_params.insert(key.into(), value.into());
        Ok(())
    }

    /// Log multiple parameters
    pub fn log_params(&mut self, params: HashMap<String, String>) -> Result<(), TorshError> {
        for (key, value) in params {
            self.log_param(key, value)?;
        }
        Ok(())
    }

    /// Log a metric value
    pub fn log_metric(
        &mut self,
        key: impl Into<String>,
        value: f64,
        step: Option<u64>,
        timestamp: Option<u64>,
    ) -> Result<(), TorshError> {
        if self.run_id.is_none() {
            return Err(TorshError::InvalidArgument("No active run".to_string()));
        }

        let key = key.into();
        let point = MetricPoint {
            value,
            step: step.unwrap_or(0),
            timestamp: timestamp.unwrap_or_else(|| {
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .expect("system time should be after UNIX_EPOCH")
                    .as_millis() as u64
            }),
        };

        self.logged_metrics
            .entry(key)
            .or_insert_with(Vec::new)
            .push(point);

        Ok(())
    }

    /// Log multiple metrics at once
    pub fn log_metrics(
        &mut self,
        metrics: HashMap<String, f64>,
        step: Option<u64>,
    ) -> Result<(), TorshError> {
        for (key, value) in metrics {
            self.log_metric(key, value, step, None)?;
        }
        Ok(())
    }

    /// Log metrics from a MetricCollection
    pub fn log_metric_collection(
        &mut self,
        collection: &mut MetricCollection,
        predictions: &Tensor,
        targets: &Tensor,
        step: Option<u64>,
    ) -> Result<(), TorshError> {
        let results = collection.compute(predictions, targets);

        for (name, value) in results {
            self.log_metric(name, value, step, None)?;
        }

        Ok(())
    }

    /// Log a single metric computation
    pub fn log_computed_metric<M: Metric>(
        &mut self,
        metric: &M,
        predictions: &Tensor,
        targets: &Tensor,
        step: Option<u64>,
    ) -> Result<(), TorshError> {
        let value = metric.compute(predictions, targets);
        self.log_metric(metric.name(), value, step, None)
    }

    /// Set a tag
    pub fn set_tag(
        &mut self,
        key: impl Into<String>,
        value: impl Into<String>,
    ) -> Result<(), TorshError> {
        if self.run_id.is_none() {
            return Err(TorshError::InvalidArgument("No active run".to_string()));
        }

        self.tags.insert(key.into(), value.into());
        Ok(())
    }

    /// Set multiple tags
    pub fn set_tags(&mut self, tags: HashMap<String, String>) -> Result<(), TorshError> {
        for (key, value) in tags {
            self.set_tag(key, value)?;
        }
        Ok(())
    }

    /// Log an artifact (file)
    pub fn log_artifact(&mut self, local_path: impl AsRef<Path>) -> Result<(), TorshError> {
        if self.run_id.is_none() {
            return Err(TorshError::InvalidArgument("No active run".to_string()));
        }

        let local_path = local_path.as_ref();
        let filename = local_path
            .file_name()
            .ok_or_else(|| TorshError::InvalidArgument("Invalid file path".to_string()))?;

        let dest_path = self.artifact_location.join(filename);

        std::fs::copy(local_path, dest_path)
            .map_err(|e| TorshError::InvalidArgument(format!("Failed to copy artifact: {}", e)))?;

        Ok(())
    }

    /// Log model configuration
    pub fn log_model_config(&mut self, config: HashMap<String, String>) -> Result<(), TorshError> {
        for (key, value) in config {
            self.log_param(format!("model.{}", key), value)?;
        }
        Ok(())
    }

    /// Log training hyperparameters
    pub fn log_hyperparameters(
        &mut self,
        hyperparams: HashMap<String, String>,
    ) -> Result<(), TorshError> {
        for (key, value) in hyperparams {
            self.log_param(format!("hp.{}", key), value)?;
        }
        Ok(())
    }

    /// Get current run ID
    pub fn get_run_id(&self) -> Option<&str> {
        self.run_id.as_deref()
    }

    /// Get artifact location
    pub fn get_artifact_location(&self) -> &Path {
        &self.artifact_location
    }

    /// Save run data to JSON file
    fn save_run_data(&self) -> Result<(), TorshError> {
        let run_data = RunData {
            run_id: self.run_id.clone().unwrap_or_default(),
            run_name: self.run_name.clone(),
            experiment_name: self.experiment_name.clone(),
            metrics: self.logged_metrics.clone(),
            params: self.logged_params.clone(),
            tags: self.tags.clone(),
            artifact_location: self.artifact_location.to_string_lossy().to_string(),
        };

        let json = serde_json::to_string_pretty(&run_data).map_err(|e| {
            TorshError::InvalidArgument(format!("Failed to serialize run data: {}", e))
        })?;

        let run_file = self.artifact_location.join("run_data.json");
        let mut file = File::create(run_file).map_err(|e| {
            TorshError::InvalidArgument(format!("Failed to create run data file: {}", e))
        })?;

        file.write_all(json.as_bytes())
            .map_err(|e| TorshError::InvalidArgument(format!("Failed to write run data: {}", e)))?;

        Ok(())
    }
}

/// Metric point with timestamp
#[derive(Debug, Clone, Serialize, Deserialize)]
struct MetricPoint {
    value: f64,
    step: u64,
    timestamp: u64,
}

/// Run data for serialization
#[derive(Debug, Clone, Serialize, Deserialize)]
struct RunData {
    run_id: String,
    run_name: Option<String>,
    experiment_name: String,
    metrics: HashMap<String, Vec<MetricPoint>>,
    params: HashMap<String, String>,
    tags: HashMap<String, String>,
    artifact_location: String,
}

/// Generate a unique run ID
fn generate_run_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time should be after UNIX_EPOCH")
        .as_millis();
    format!("run_{}", timestamp)
}

/// MLflow run context (RAII pattern for automatic run management)
pub struct MLflowRun<'a> {
    client: &'a mut MLflowClient,
    run_id: String,
}

impl<'a> MLflowRun<'a> {
    /// Create a new MLflow run context
    pub fn new(client: &'a mut MLflowClient, run_name: Option<String>) -> Result<Self, TorshError> {
        let run_id = client.start_run(run_name)?;
        Ok(Self { client, run_id })
    }

    /// Get the client
    pub fn client(&mut self) -> &mut MLflowClient {
        self.client
    }

    /// Get the run ID
    pub fn run_id(&self) -> &str {
        &self.run_id
    }
}

impl<'a> Drop for MLflowRun<'a> {
    fn drop(&mut self) {
        let _ = self.client.end_run();
    }
}

/// Experiment tracker that manages multiple runs
pub struct ExperimentTracker {
    client: MLflowClient,
    auto_log: bool,
    metric_prefix: String,
}

impl ExperimentTracker {
    /// Create a new experiment tracker
    pub fn new(
        tracking_uri: impl Into<String>,
        experiment_name: impl Into<String>,
        metric_prefix: impl Into<String>,
    ) -> Self {
        Self {
            client: MLflowClient::new(tracking_uri, experiment_name),
            auto_log: true,
            metric_prefix: metric_prefix.into(),
        }
    }

    /// Enable or disable auto-logging
    pub fn set_auto_log(&mut self, enabled: bool) {
        self.auto_log = enabled;
    }

    /// Start a new run
    pub fn start_run(&mut self, run_name: Option<String>) -> Result<String, TorshError> {
        self.client.start_run(run_name)
    }

    /// End current run
    pub fn end_run(&mut self) -> Result<(), TorshError> {
        self.client.end_run()
    }

    /// Log training step
    pub fn log_train_step(
        &mut self,
        epoch: u64,
        metrics: HashMap<String, f64>,
    ) -> Result<(), TorshError> {
        if !self.auto_log {
            return Ok(());
        }

        for (name, value) in metrics {
            let key = format!("{}/train/{}", self.metric_prefix, name);
            self.client.log_metric(key, value, Some(epoch), None)?;
        }

        Ok(())
    }

    /// Log validation step
    pub fn log_val_step(
        &mut self,
        epoch: u64,
        metrics: HashMap<String, f64>,
    ) -> Result<(), TorshError> {
        if !self.auto_log {
            return Ok(());
        }

        for (name, value) in metrics {
            let key = format!("{}/val/{}", self.metric_prefix, name);
            self.client.log_metric(key, value, Some(epoch), None)?;
        }

        Ok(())
    }

    /// Log test results
    pub fn log_test_results(&mut self, metrics: HashMap<String, f64>) -> Result<(), TorshError> {
        for (name, value) in metrics {
            let key = format!("{}/test/{}", self.metric_prefix, name);
            self.client.log_metric(key, value, None, None)?;
        }

        Ok(())
    }

    /// Get the underlying client
    pub fn client(&mut self) -> &mut MLflowClient {
        &mut self.client
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mlflow_client_creation() {
        let client = MLflowClient::new("http://localhost:5000", "test_experiment");
        assert!(client.run_id.is_none());
    }

    #[test]
    fn test_start_end_run() {
        let mut client = MLflowClient::new("http://localhost:5000", "test_experiment");

        let run_id = client.start_run(Some("test_run".to_string()));
        assert!(run_id.is_ok());
        assert!(client.get_run_id().is_some());

        let result = client.end_run();
        assert!(result.is_ok());
        assert!(client.get_run_id().is_none());
    }

    #[test]
    fn test_log_param() {
        let mut client = MLflowClient::new("http://localhost:5000", "test_experiment");
        client.start_run(Some("test_run".to_string())).unwrap();

        let result = client.log_param("learning_rate", "0.001");
        assert!(result.is_ok());

        client.end_run().unwrap();
    }

    #[test]
    fn test_log_metric() {
        let mut client = MLflowClient::new("http://localhost:5000", "test_experiment");
        client.start_run(Some("test_run".to_string())).unwrap();

        let result = client.log_metric("accuracy", 0.95, Some(0), None);
        assert!(result.is_ok());

        client.end_run().unwrap();
    }

    #[test]
    fn test_log_metrics() {
        let mut client = MLflowClient::new("http://localhost:5000", "test_experiment");
        client.start_run(Some("test_run".to_string())).unwrap();

        let mut metrics = HashMap::new();
        metrics.insert("accuracy".to_string(), 0.95);
        metrics.insert("precision".to_string(), 0.92);

        let result = client.log_metrics(metrics, Some(0));
        assert!(result.is_ok());

        client.end_run().unwrap();
    }

    #[test]
    fn test_set_tag() {
        let mut client = MLflowClient::new("http://localhost:5000", "test_experiment");
        client.start_run(Some("test_run".to_string())).unwrap();

        let result = client.set_tag("model_type", "ResNet50");
        assert!(result.is_ok());

        client.end_run().unwrap();
    }

    #[test]
    fn test_mlflow_run_context() {
        let mut client = MLflowClient::new("http://localhost:5000", "test_experiment");

        {
            let mut run = MLflowRun::new(&mut client, Some("test_run".to_string())).unwrap();
            assert!(run.client().get_run_id().is_some());
        }

        // Run should be ended after scope
        assert!(client.get_run_id().is_none());
    }

    #[test]
    fn test_experiment_tracker() {
        let mut tracker =
            ExperimentTracker::new("http://localhost:5000", "test_experiment", "model");

        tracker.start_run(Some("test_run".to_string())).unwrap();

        let mut metrics = HashMap::new();
        metrics.insert("accuracy".to_string(), 0.95);

        let result = tracker.log_train_step(0, metrics);
        assert!(result.is_ok());

        tracker.end_run().unwrap();
    }

    #[test]
    fn test_log_hyperparameters() {
        let mut client = MLflowClient::new("http://localhost:5000", "test_experiment");
        client.start_run(Some("test_run".to_string())).unwrap();

        let mut hyperparams = HashMap::new();
        hyperparams.insert("batch_size".to_string(), "32".to_string());
        hyperparams.insert("learning_rate".to_string(), "0.001".to_string());

        let result = client.log_hyperparameters(hyperparams);
        assert!(result.is_ok());

        client.end_run().unwrap();
    }
}
