//! Fault tolerance features for distributed training
//!
//! This module provides comprehensive fault tolerance capabilities including:
//! - Checkpointing system for saving and restoring training state
//! - Elastic training support for dynamic worker scaling
//! - State synchronization during scaling events
//! - Integration with error recovery mechanisms

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use crate::error_recovery::{CircuitBreakerConfig, FailureDetector, RetryConfig};
use crate::{TorshDistributedError, TorshResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::fs;
use torsh_nn::Parameter;
use torsh_tensor::Tensor;
use tracing::{debug, info, warn};

/// Configuration for checkpointing
#[derive(Debug, Clone)]
pub struct CheckpointConfig {
    /// Directory to save checkpoints
    pub checkpoint_dir: PathBuf,
    /// How frequently to save checkpoints (in steps)
    pub checkpoint_frequency: usize,
    /// Maximum number of checkpoints to keep
    pub max_checkpoints: usize,
    /// Whether to save async (non-blocking)
    pub async_save: bool,
    /// Compression level for checkpoint files (0-9)
    pub compression_level: u8,
    /// Whether to verify checkpoints after saving
    pub verify_after_save: bool,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            checkpoint_dir: PathBuf::from("./checkpoints"),
            checkpoint_frequency: 1000,
            max_checkpoints: 5,
            async_save: true,
            compression_level: 3,
            verify_after_save: true,
        }
    }
}

/// Configuration for elastic training
#[derive(Debug, Clone)]
pub struct ElasticConfig {
    /// Minimum number of workers required
    pub min_workers: usize,
    /// Maximum number of workers allowed
    pub max_workers: usize,
    /// How long to wait for workers to join/leave before proceeding
    pub scaling_timeout: Duration,
    /// How frequently to check for scaling events
    pub scaling_check_interval: Duration,
    /// Whether to enable elastic scheduling
    pub enable_elastic_scheduling: bool,
    /// Rendezvous backend for worker coordination
    pub rendezvous_backend: String,
    /// Rendezvous endpoint
    pub rendezvous_endpoint: String,
}

impl Default for ElasticConfig {
    fn default() -> Self {
        Self {
            min_workers: 1,
            max_workers: 64,
            scaling_timeout: Duration::from_secs(300), // 5 minutes
            scaling_check_interval: Duration::from_secs(30),
            enable_elastic_scheduling: true,
            rendezvous_backend: "etcd".to_string(),
            rendezvous_endpoint: "localhost:2379".to_string(),
        }
    }
}

/// Training checkpoint containing all necessary state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingCheckpoint {
    /// Step number when checkpoint was created
    pub step: usize,
    /// Epoch number
    pub epoch: usize,
    /// Model parameters
    pub model_state: HashMap<String, Vec<f32>>,
    /// Optimizer state
    pub optimizer_state: HashMap<String, Vec<f32>>,
    /// Learning rate scheduler state
    pub scheduler_state: HashMap<String, f32>,
    /// Random number generator states
    pub rng_states: HashMap<String, Vec<u8>>,
    /// Loss value at checkpoint
    pub loss: f32,
    /// Metrics at checkpoint
    pub metrics: HashMap<String, f32>,
    /// Training configuration
    pub config: HashMap<String, String>,
    /// Timestamp when checkpoint was created
    pub timestamp: u64,
    /// Version identifier
    pub version: String,
    /// Distributed training metadata
    pub distributed_meta: DistributedMetadata,
}

/// Metadata about the distributed training state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedMetadata {
    /// World size when checkpoint was created
    pub world_size: usize,
    /// Rank of the process that created the checkpoint
    pub rank: usize,
    /// Process group information
    pub process_group_info: HashMap<String, String>,
    /// Data parallel group size
    pub dp_size: usize,
    /// Tensor parallel group size
    pub tp_size: usize,
    /// Pipeline parallel group size
    pub pp_size: usize,
    /// FSDP sharding information
    pub fsdp_sharding: HashMap<String, Vec<usize>>,
}

/// Events that can trigger elastic scaling
#[derive(Debug, Clone, PartialEq)]
pub enum ScalingEvent {
    /// Worker failure detected
    WorkerFailure { failed_ranks: Vec<usize> },
    /// New workers joining
    WorkerJoin { new_ranks: Vec<usize> },
    /// Manual scaling request
    ManualScale { target_workers: usize },
    /// Automatic scaling based on load
    AutoScale {
        target_workers: usize,
        reason: String,
    },
}

/// State during elastic scaling
#[derive(Debug, Clone)]
pub enum ScalingState {
    /// Normal training, no scaling
    Stable,
    /// Scaling in progress
    Scaling {
        event: ScalingEvent,
        start_time: SystemTime,
        expected_workers: usize,
    },
    /// Waiting for workers to synchronize
    Synchronizing {
        current_workers: usize,
        target_workers: usize,
    },
}

/// Checkpoint manager for saving and restoring training state
#[derive(Debug)]
pub struct CheckpointManager {
    config: CheckpointConfig,
    failure_detector: FailureDetector,
    latest_checkpoint: Arc<RwLock<Option<TrainingCheckpoint>>>,
    checkpoint_history: Arc<RwLock<Vec<PathBuf>>>,
}

impl CheckpointManager {
    /// Create a new checkpoint manager
    pub fn new(
        config: CheckpointConfig,
        health_check_interval: Duration,
        health_timeout: Duration,
    ) -> TorshResult<Self> {
        // Ensure checkpoint directory exists
        std::fs::create_dir_all(&config.checkpoint_dir).map_err(|e| {
            TorshDistributedError::backend_error(
                "checkpoint",
                format!("Failed to create checkpoint directory: {}", e),
            )
        })?;

        let retry_config = RetryConfig::default();
        let circuit_breaker_config = CircuitBreakerConfig::default();

        let failure_detector = FailureDetector::new(
            health_check_interval,
            health_timeout,
            retry_config,
            Some(circuit_breaker_config),
        );

        Ok(Self {
            config,
            failure_detector,
            latest_checkpoint: Arc::new(RwLock::new(None)),
            checkpoint_history: Arc::new(RwLock::new(Vec::new())),
        })
    }

    /// Save a training checkpoint
    pub async fn save_checkpoint(
        &self,
        checkpoint: TrainingCheckpoint,
        rank: usize,
    ) -> TorshResult<PathBuf> {
        let checkpoint_path = self.config.checkpoint_dir.join(format!(
            "checkpoint_step_{}_rank_{}.json",
            checkpoint.step, rank
        ));

        info!(
            "Saving checkpoint at step {} to {:?}",
            checkpoint.step, checkpoint_path
        );

        let checkpoint_data = serde_json::to_string_pretty(&checkpoint).map_err(|e| {
            TorshDistributedError::backend_error(
                "checkpoint",
                format!("Failed to serialize checkpoint: {}", e),
            )
        })?;

        // Save with error recovery
        self.failure_detector
            .execute_with_recovery(
                || async {
                    fs::write(&checkpoint_path, &checkpoint_data)
                        .await
                        .map_err(|e| {
                            TorshDistributedError::backend_error(
                                "checkpoint",
                                format!("Failed to write checkpoint: {}", e),
                            )
                        })
                },
                None,
            )
            .await?;

        // Verify checkpoint if configured
        if self.config.verify_after_save {
            self.verify_checkpoint(&checkpoint_path).await?;
        }

        // Store checkpoint step before moving
        let checkpoint_step = checkpoint.step;

        // Update internal state
        {
            let mut latest = self.latest_checkpoint.write().unwrap();
            *latest = Some(checkpoint);
        }

        {
            let mut history = self.checkpoint_history.write().unwrap();
            history.push(checkpoint_path.clone());

            // Clean up old checkpoints
            if history.len() > self.config.max_checkpoints {
                let old_checkpoint = history.remove(0);
                if let Err(e) = std::fs::remove_file(&old_checkpoint) {
                    warn!(
                        "Failed to remove old checkpoint {:?}: {}",
                        old_checkpoint, e
                    );
                }
            }
        }

        info!("Successfully saved checkpoint at step {}", checkpoint_step);
        Ok(checkpoint_path)
    }

    /// Load the latest checkpoint
    pub async fn load_latest_checkpoint(&self) -> TorshResult<Option<TrainingCheckpoint>> {
        let checkpoint_files = self.find_checkpoint_files().await?;

        if checkpoint_files.is_empty() {
            info!("No checkpoints found");
            return Ok(None);
        }

        // Find the latest checkpoint by step number
        let latest_file = checkpoint_files
            .iter()
            .max_by_key(|path| self.extract_step_from_filename(path))
            .unwrap();

        info!("Loading latest checkpoint from {:?}", latest_file);
        self.load_checkpoint(latest_file).await
    }

    /// Load a specific checkpoint
    pub async fn load_checkpoint(
        &self,
        checkpoint_path: &PathBuf,
    ) -> TorshResult<Option<TrainingCheckpoint>> {
        self.failure_detector
            .execute_with_recovery(
                || async {
                    let checkpoint_data =
                        fs::read_to_string(checkpoint_path).await.map_err(|e| {
                            TorshDistributedError::backend_error(
                                "checkpoint",
                                format!("Failed to read checkpoint: {}", e),
                            )
                        })?;

                    let checkpoint: TrainingCheckpoint = serde_json::from_str(&checkpoint_data)
                        .map_err(|e| {
                            TorshDistributedError::backend_error(
                                "checkpoint",
                                format!("Failed to deserialize checkpoint: {}", e),
                            )
                        })?;

                    info!(
                        "Successfully loaded checkpoint from step {}",
                        checkpoint.step
                    );
                    Ok(Some(checkpoint))
                },
                None,
            )
            .await
    }

    /// Verify checkpoint integrity
    async fn verify_checkpoint(&self, checkpoint_path: &PathBuf) -> TorshResult<()> {
        debug!("Verifying checkpoint {:?}", checkpoint_path);

        // Load and parse to ensure it's valid
        let checkpoint = self.load_checkpoint(checkpoint_path).await?;
        if checkpoint.is_none() {
            return Err(TorshDistributedError::backend_error(
                "checkpoint",
                "Checkpoint verification failed: could not load",
            ));
        }

        debug!("Checkpoint verification successful");
        Ok(())
    }

    /// Find all checkpoint files in the directory
    async fn find_checkpoint_files(&self) -> TorshResult<Vec<PathBuf>> {
        let mut checkpoint_files = Vec::new();

        let mut dir_entries = fs::read_dir(&self.config.checkpoint_dir)
            .await
            .map_err(|e| {
                TorshDistributedError::backend_error(
                    "checkpoint",
                    format!("Failed to read checkpoint directory: {}", e),
                )
            })?;

        while let Some(entry) = dir_entries.next_entry().await.map_err(|e| {
            TorshDistributedError::backend_error(
                "checkpoint",
                format!("Failed to read directory entry: {}", e),
            )
        })? {
            let path = entry.path();
            if let Some(filename) = path.file_name() {
                if filename.to_string_lossy().starts_with("checkpoint_")
                    && filename.to_string_lossy().ends_with(".json")
                {
                    checkpoint_files.push(path);
                }
            }
        }

        Ok(checkpoint_files)
    }

    /// Extract step number from checkpoint filename
    fn extract_step_from_filename(&self, path: &Path) -> usize {
        if let Some(filename) = path.file_stem() {
            let filename_str = filename.to_string_lossy();
            if let Some(step_start) = filename_str.find("step_") {
                let step_part = &filename_str[step_start + 5..];
                if let Some(rank_pos) = step_part.find("_rank_") {
                    let step_str = &step_part[..rank_pos];
                    return step_str.parse().unwrap_or(0);
                }
            }
        }
        0
    }

    /// Get the latest checkpoint metadata without loading the full checkpoint
    pub fn get_latest_checkpoint_info(&self) -> Option<TrainingCheckpoint> {
        self.latest_checkpoint.read().unwrap().clone()
    }

    /// Clean up all checkpoints (useful for cleanup)
    pub async fn cleanup_all_checkpoints(&self) -> TorshResult<()> {
        let checkpoint_files = self.find_checkpoint_files().await?;

        for file in checkpoint_files {
            if let Err(e) = fs::remove_file(&file).await {
                warn!("Failed to remove checkpoint file {:?}: {}", file, e);
            }
        }

        {
            let mut history = self.checkpoint_history.write().unwrap();
            history.clear();
        }
        {
            let mut latest = self.latest_checkpoint.write().unwrap();
            *latest = None;
        }

        info!("Cleaned up all checkpoints");
        Ok(())
    }
}

/// Elastic training manager for handling dynamic scaling
#[derive(Debug)]
pub struct ElasticTrainingManager {
    config: ElasticConfig,
    scaling_state: Arc<RwLock<ScalingState>>,
    checkpoint_manager: CheckpointManager,
    current_world_size: Arc<RwLock<usize>>,
    worker_registry: Arc<RwLock<HashMap<usize, SystemTime>>>,
    scaling_events: Arc<Mutex<Vec<ScalingEvent>>>,
}

impl ElasticTrainingManager {
    /// Create a new elastic training manager
    pub fn new(
        config: ElasticConfig,
        checkpoint_config: CheckpointConfig,
        initial_world_size: usize,
    ) -> TorshResult<Self> {
        let checkpoint_manager = CheckpointManager::new(
            checkpoint_config,
            Duration::from_secs(30),  // health check interval
            Duration::from_secs(120), // health timeout
        )?;

        Ok(Self {
            config,
            scaling_state: Arc::new(RwLock::new(ScalingState::Stable)),
            checkpoint_manager,
            current_world_size: Arc::new(RwLock::new(initial_world_size)),
            worker_registry: Arc::new(RwLock::new(HashMap::new())),
            scaling_events: Arc::new(Mutex::new(Vec::new())),
        })
    }

    /// Check if scaling is needed and initiate if necessary
    pub async fn check_scaling_needs(&self) -> TorshResult<Option<ScalingEvent>> {
        let current_state = self.scaling_state.read().unwrap().clone();

        match current_state {
            ScalingState::Stable => {
                // Check for worker failures or new joins
                let current_workers = *self.current_world_size.read().unwrap();

                // Simulate failure detection (in real implementation, this would check actual worker health)
                let failed_workers = self.detect_failed_workers().await?;
                if !failed_workers.is_empty() {
                    let event = ScalingEvent::WorkerFailure {
                        failed_ranks: failed_workers,
                    };
                    info!("Detected worker failures, initiating scaling: {:?}", event);
                    self.initiate_scaling(event.clone()).await?;
                    return Ok(Some(event));
                }

                // Check for new workers
                let new_workers = self.detect_new_workers().await?;
                if !new_workers.is_empty() {
                    let event = ScalingEvent::WorkerJoin {
                        new_ranks: new_workers,
                    };
                    info!("Detected new workers, initiating scaling: {:?}", event);
                    self.initiate_scaling(event.clone()).await?;
                    return Ok(Some(event));
                }

                // Check for automatic scaling based on load (simplified)
                if self.config.enable_elastic_scheduling {
                    if let Some(target) = self.calculate_optimal_workers(current_workers).await? {
                        if target != current_workers {
                            let event = ScalingEvent::AutoScale {
                                target_workers: target,
                                reason: "Load-based scaling".to_string(),
                            };
                            info!("Initiating auto-scaling: {:?}", event);
                            self.initiate_scaling(event.clone()).await?;
                            return Ok(Some(event));
                        }
                    }
                }
            }
            ScalingState::Scaling { .. } => {
                // Check if scaling has completed
                if self.is_scaling_complete().await? {
                    self.complete_scaling().await?;
                }
            }
            ScalingState::Synchronizing { .. } => {
                // Check if synchronization has completed
                if self.is_synchronization_complete().await? {
                    self.complete_synchronization().await?;
                }
            }
        }

        Ok(None)
    }

    /// Initiate scaling process
    async fn initiate_scaling(&self, event: ScalingEvent) -> TorshResult<()> {
        info!("Initiating scaling for event: {:?}", event);

        // Save checkpoint before scaling
        if let Some(checkpoint) = self.checkpoint_manager.get_latest_checkpoint_info() {
            self.checkpoint_manager
                .save_checkpoint(checkpoint, 0)
                .await?;
        }

        let expected_workers = match &event {
            ScalingEvent::WorkerFailure { failed_ranks } => {
                *self.current_world_size.read().unwrap() - failed_ranks.len()
            }
            ScalingEvent::WorkerJoin { new_ranks } => {
                *self.current_world_size.read().unwrap() + new_ranks.len()
            }
            ScalingEvent::ManualScale { target_workers }
            | ScalingEvent::AutoScale { target_workers, .. } => *target_workers,
        };

        // Ensure we stay within bounds
        let expected_workers = expected_workers
            .max(self.config.min_workers)
            .min(self.config.max_workers);

        {
            let mut state = self.scaling_state.write().unwrap();
            *state = ScalingState::Scaling {
                event: event.clone(),
                start_time: SystemTime::now(),
                expected_workers,
            };
        }

        // Add to event history
        {
            let mut events = self.scaling_events.lock().unwrap();
            events.push(event);
            // Keep only recent events
            if events.len() > 100 {
                events.drain(0..50);
            }
        }

        Ok(())
    }

    /// Check if scaling is complete
    async fn is_scaling_complete(&self) -> TorshResult<bool> {
        // Simplified: check if enough time has passed
        if let ScalingState::Scaling { start_time, .. } = *self.scaling_state.read().unwrap() {
            Ok(start_time.elapsed().unwrap_or(Duration::ZERO) >= self.config.scaling_timeout)
        } else {
            Ok(false)
        }
    }

    /// Complete the scaling process
    async fn complete_scaling(&self) -> TorshResult<()> {
        info!("Completing scaling process");

        let expected_workers = if let ScalingState::Scaling {
            expected_workers, ..
        } = *self.scaling_state.read().unwrap()
        {
            expected_workers
        } else {
            return Ok(());
        };

        // Transition to synchronization state
        {
            let mut state = self.scaling_state.write().unwrap();
            *state = ScalingState::Synchronizing {
                current_workers: *self.current_world_size.read().unwrap(),
                target_workers: expected_workers,
            };
        }

        info!("Transitioning to synchronization phase");
        Ok(())
    }

    /// Check if synchronization is complete
    async fn is_synchronization_complete(&self) -> TorshResult<bool> {
        // Simplified: assume synchronization completes quickly
        Ok(true)
    }

    /// Complete the synchronization process
    async fn complete_synchronization(&self) -> TorshResult<()> {
        info!("Completing synchronization process");

        let target_workers = if let ScalingState::Synchronizing { target_workers, .. } =
            *self.scaling_state.read().unwrap()
        {
            target_workers
        } else {
            return Ok(());
        };

        // Update world size
        {
            let mut world_size = self.current_world_size.write().unwrap();
            *world_size = target_workers;
        }

        // Return to stable state
        {
            let mut state = self.scaling_state.write().unwrap();
            *state = ScalingState::Stable;
        }

        info!(
            "Elastic scaling completed, new world size: {}",
            target_workers
        );
        Ok(())
    }

    /// Detect failed workers (simplified implementation)
    async fn detect_failed_workers(&self) -> TorshResult<Vec<usize>> {
        // In a real implementation, this would check actual worker health
        // For now, return empty to simulate no failures
        Ok(Vec::new())
    }

    /// Detect new workers (simplified implementation)
    async fn detect_new_workers(&self) -> TorshResult<Vec<usize>> {
        // In a real implementation, this would check for new worker registrations
        // For now, return empty to simulate no new workers
        Ok(Vec::new())
    }

    /// Calculate optimal number of workers based on current load
    async fn calculate_optimal_workers(
        &self,
        _current_workers: usize,
    ) -> TorshResult<Option<usize>> {
        // Simplified auto-scaling logic
        // In a real implementation, this would consider:
        // - Current throughput
        // - Resource utilization
        // - Queue lengths
        // - Performance metrics

        // For now, maintain current workers
        Ok(None)
    }

    /// Get current scaling state
    pub fn get_scaling_state(&self) -> ScalingState {
        self.scaling_state.read().unwrap().clone()
    }

    /// Get current world size
    pub fn get_world_size(&self) -> usize {
        *self.current_world_size.read().unwrap()
    }

    /// Force manual scaling
    pub async fn scale_to(&self, target_workers: usize) -> TorshResult<()> {
        let event = ScalingEvent::ManualScale { target_workers };
        self.initiate_scaling(event).await
    }

    /// Get scaling event history
    pub fn get_scaling_history(&self) -> Vec<ScalingEvent> {
        self.scaling_events.lock().unwrap().clone()
    }

    /// Check if training can proceed (not currently scaling)
    pub fn can_proceed_training(&self) -> bool {
        matches!(*self.scaling_state.read().unwrap(), ScalingState::Stable)
    }

    /// Get checkpoint manager reference
    pub fn checkpoint_manager(&self) -> &CheckpointManager {
        &self.checkpoint_manager
    }
}

/// Utility functions for creating training checkpoints
pub mod checkpoint_utils {
    use super::*;

    /// Parameters for creating a checkpoint
    #[allow(dead_code)]
    pub struct CheckpointParams {
        pub step: usize,
        pub epoch: usize,
        pub model_params: HashMap<String, Parameter>,
        pub optimizer_state: HashMap<String, Tensor>,
        pub loss: f32,
        pub metrics: HashMap<String, f32>,
        pub world_size: usize,
        pub rank: usize,
    }

    /// Create a checkpoint from model parameters and training state
    pub fn create_checkpoint(params: CheckpointParams) -> TorshResult<TrainingCheckpoint> {
        let CheckpointParams {
            step,
            epoch,
            model_params,
            optimizer_state,
            loss,
            metrics,
            world_size,
            rank,
        } = params;
        // Convert model parameters to serializable format
        let mut model_state = HashMap::new();
        for (name, param) in model_params {
            let tensor = param.tensor();
            let tensor_guard = tensor.read();
            let data = tensor_guard.flatten()?.to_vec()?;
            model_state.insert(name, data);
        }

        // Convert optimizer state to serializable format
        let mut opt_state = HashMap::new();
        for (name, tensor) in optimizer_state {
            let data = tensor.flatten()?.to_vec()?;
            opt_state.insert(name, data);
        }

        let distributed_meta = DistributedMetadata {
            world_size,
            rank,
            process_group_info: HashMap::new(),
            dp_size: world_size, // Simplified
            tp_size: 1,
            pp_size: 1,
            fsdp_sharding: HashMap::new(),
        };

        Ok(TrainingCheckpoint {
            step,
            epoch,
            model_state,
            optimizer_state: opt_state,
            scheduler_state: HashMap::new(),
            rng_states: HashMap::new(),
            loss,
            metrics,
            config: HashMap::new(),
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            version: "1.0.0".to_string(),
            distributed_meta,
        })
    }

    /// Restore model parameters from checkpoint
    pub fn restore_model_from_checkpoint(
        checkpoint: &TrainingCheckpoint,
    ) -> TorshResult<HashMap<String, Tensor>> {
        let mut model_params = HashMap::new();

        for (name, data) in &checkpoint.model_state {
            let shape = vec![data.len()]; // Simplified shape reconstruction
            let tensor = Tensor::from_vec(data.clone(), &shape)?;
            model_params.insert(name.clone(), tensor);
        }

        Ok(model_params)
    }

    /// Restore optimizer state from checkpoint
    pub fn restore_optimizer_from_checkpoint(
        checkpoint: &TrainingCheckpoint,
    ) -> TorshResult<HashMap<String, Tensor>> {
        let mut optimizer_state = HashMap::new();

        for (name, data) in &checkpoint.optimizer_state {
            let shape = vec![data.len()]; // Simplified shape reconstruction
            let tensor = Tensor::from_vec(data.clone(), &shape)?;
            optimizer_state.insert(name.clone(), tensor);
        }

        Ok(optimizer_state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_checkpoint_manager() -> TorshResult<()> {
        let temp_dir = TempDir::new().unwrap();
        let config = CheckpointConfig {
            checkpoint_dir: temp_dir.path().to_path_buf(),
            checkpoint_frequency: 100,
            max_checkpoints: 3,
            ..Default::default()
        };

        let manager = CheckpointManager::new(
            config,
            Duration::from_millis(100),
            Duration::from_millis(200),
        )?;

        // Create a test checkpoint
        let checkpoint = TrainingCheckpoint {
            step: 1000,
            epoch: 10,
            model_state: {
                let mut state = HashMap::new();
                state.insert("weight".to_string(), vec![1.0, 2.0, 3.0]);
                state
            },
            optimizer_state: HashMap::new(),
            scheduler_state: HashMap::new(),
            rng_states: HashMap::new(),
            loss: 0.5,
            metrics: HashMap::new(),
            config: HashMap::new(),
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            version: "1.0.0".to_string(),
            distributed_meta: DistributedMetadata {
                world_size: 4,
                rank: 0,
                process_group_info: HashMap::new(),
                dp_size: 4,
                tp_size: 1,
                pp_size: 1,
                fsdp_sharding: HashMap::new(),
            },
        };

        // Save checkpoint
        let checkpoint_path = manager.save_checkpoint(checkpoint.clone(), 0).await?;
        assert!(checkpoint_path.exists());

        // Load checkpoint
        let loaded = manager.load_latest_checkpoint().await?;
        assert!(loaded.is_some());
        let loaded_checkpoint = loaded.unwrap();
        assert_eq!(loaded_checkpoint.step, checkpoint.step);
        assert_eq!(loaded_checkpoint.loss, checkpoint.loss);

        Ok(())
    }

    #[tokio::test]
    async fn test_elastic_training_manager() -> TorshResult<()> {
        let temp_dir = TempDir::new().unwrap();
        let elastic_config = ElasticConfig {
            min_workers: 2,
            max_workers: 8,
            scaling_timeout: Duration::from_millis(100),
            ..Default::default()
        };
        let checkpoint_config = CheckpointConfig {
            checkpoint_dir: temp_dir.path().to_path_buf(),
            ..Default::default()
        };

        let manager = ElasticTrainingManager::new(
            elastic_config,
            checkpoint_config,
            4, // initial world size
        )?;

        assert_eq!(manager.get_world_size(), 4);
        assert!(manager.can_proceed_training());

        // Test manual scaling
        manager.scale_to(6).await?;

        // Initially should be in scaling state
        match manager.get_scaling_state() {
            ScalingState::Scaling {
                expected_workers, ..
            } => {
                assert_eq!(expected_workers, 6);
            }
            _ => panic!("Expected scaling state"),
        }

        Ok(())
    }

    #[test]
    fn test_checkpoint_config() {
        let config = CheckpointConfig::default();
        assert_eq!(config.checkpoint_frequency, 1000);
        assert_eq!(config.max_checkpoints, 5);
        assert!(config.async_save);
    }

    #[test]
    fn test_elastic_config() {
        let config = ElasticConfig::default();
        assert_eq!(config.min_workers, 1);
        assert_eq!(config.max_workers, 64);
        assert!(config.enable_elastic_scheduling);
    }

    #[test]
    fn test_scaling_events() {
        let event1 = ScalingEvent::WorkerFailure {
            failed_ranks: vec![1, 2],
        };
        let event2 = ScalingEvent::WorkerJoin {
            new_ranks: vec![5, 6],
        };
        let event3 = ScalingEvent::ManualScale { target_workers: 8 };

        assert_ne!(event1, event2);
        assert_ne!(event2, event3);
    }
}
