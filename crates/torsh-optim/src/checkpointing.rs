use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
// use std::sync::Arc;
use crate::{OptimizerError, OptimizerResult};
use std::time::{SystemTime, UNIX_EPOCH};

/// Configuration for gradient checkpointing
#[derive(Debug, Clone)]
pub struct CheckpointConfig {
    /// Directory to store checkpoints
    pub checkpoint_dir: PathBuf,
    /// Maximum number of checkpoints to keep
    pub max_checkpoints: usize,
    /// Frequency of checkpointing (every N steps)
    pub checkpoint_frequency: usize,
    /// Whether to save optimizer state
    pub save_optimizer_state: bool,
    /// Whether to save gradients
    pub save_gradients: bool,
    /// Whether to compress checkpoints
    pub compress: bool,
    /// Async checkpoint saving
    pub async_save: bool,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            checkpoint_dir: PathBuf::from("checkpoints"),
            max_checkpoints: 5,
            checkpoint_frequency: 1000,
            save_optimizer_state: true,
            save_gradients: false,
            compress: true,
            async_save: true,
        }
    }
}

/// Checkpoint metadata
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CheckpointMetadata {
    pub step: usize,
    pub timestamp: u64,
    pub loss: Option<f32>,
    pub learning_rate: f32,
    pub epoch: Option<usize>,
    pub gradient_norm: Option<f32>,
    pub model_hash: Option<String>,
}

/// Checkpoint data structure
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Checkpoint {
    pub metadata: CheckpointMetadata,
    pub optimizer_state: Option<Vec<u8>>,
    pub gradients: Option<HashMap<String, Vec<f32>>>,
    pub model_parameters: Option<HashMap<String, Vec<f32>>>,
}

/// Gradient checkpointing manager
pub struct CheckpointManager {
    config: CheckpointConfig,
    checkpoints: VecDeque<(PathBuf, CheckpointMetadata)>,
    current_step: usize,
    last_checkpoint_step: usize,
}

impl CheckpointManager {
    /// Create a new checkpoint manager
    pub fn new(config: CheckpointConfig) -> Result<Self, OptimizerError> {
        // Create checkpoint directory if it doesn't exist
        std::fs::create_dir_all(&config.checkpoint_dir).map_err(|e| {
            OptimizerError::CheckpointError(format!("Failed to create checkpoint directory: {e}"))
        })?;

        let mut manager = Self {
            config,
            checkpoints: VecDeque::new(),
            current_step: 0,
            last_checkpoint_step: 0,
        };

        // Load existing checkpoints
        manager.load_existing_checkpoints()?;

        Ok(manager)
    }

    /// Check if a checkpoint should be saved at the current step
    pub fn should_checkpoint(&self) -> bool {
        self.current_step > 0
            && (self.current_step - self.last_checkpoint_step) >= self.config.checkpoint_frequency
    }

    /// Save a checkpoint
    pub fn save_checkpoint(
        &mut self,
        optimizer_state: Option<&[u8]>,
        gradients: Option<&HashMap<String, Vec<f32>>>,
        model_parameters: Option<&HashMap<String, Vec<f32>>>,
        loss: Option<f32>,
        learning_rate: f32,
        epoch: Option<usize>,
    ) -> Result<PathBuf, OptimizerError> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| OptimizerError::CheckpointError(format!("Time error: {e}")))?
            .as_secs();

        let gradient_norm = gradients
            .as_ref()
            .map(|grads| grads.values().flatten().map(|&x| x * x).sum::<f32>().sqrt());

        let model_hash = model_parameters.as_ref().map(|params| {
            // Simple hash of model parameters for integrity check
            format!(
                "{:x}",
                params
                    .values()
                    .flatten()
                    .map(|&x| (x * 1000.0) as i32)
                    .fold(0u64, |acc, x| acc.wrapping_add(x as u64))
            )
        });

        let metadata = CheckpointMetadata {
            step: self.current_step,
            timestamp,
            loss,
            learning_rate,
            epoch,
            gradient_norm,
            model_hash,
        };

        let checkpoint = Checkpoint {
            metadata: metadata.clone(),
            optimizer_state: optimizer_state.map(|s| s.to_vec()),
            gradients: gradients.cloned(),
            model_parameters: model_parameters.cloned(),
        };

        let checkpoint_path = self
            .config
            .checkpoint_dir
            .join(format!("checkpoint_step_{}.bin", self.current_step));

        if self.config.async_save {
            self.save_checkpoint_async(checkpoint, checkpoint_path.clone())?;
        } else {
            self.save_checkpoint_sync(&checkpoint, &checkpoint_path)?;
        }

        // Update checkpoint list
        self.checkpoints
            .push_back((checkpoint_path.clone(), metadata));
        self.last_checkpoint_step = self.current_step;

        // Remove old checkpoints if necessary
        self.cleanup_old_checkpoints()?;

        Ok(checkpoint_path)
    }

    /// Load a checkpoint
    pub fn load_checkpoint(&self, checkpoint_path: &Path) -> Result<Checkpoint, OptimizerError> {
        let data = std::fs::read(checkpoint_path).map_err(|e| {
            OptimizerError::CheckpointError(format!("Failed to read checkpoint: {e}"))
        })?;

        let checkpoint: Checkpoint = if self.config.compress {
            // Decompress if needed
            let decompressed = self.decompress_data(&data)?;
                {
                    let (checkpoint, _): (Checkpoint, usize) = bincode::serde::decode_from_slice(&decompressed, bincode::config::standard()).map_err(|e| {
                        OptimizerError::CheckpointError(format!("Failed to deserialize checkpoint: {e}"))
                    })?;
                    checkpoint
                }
        } else {
                let (checkpoint, _): (Checkpoint, usize) = bincode::serde::decode_from_slice(&data, bincode::config::standard()).map_err(|e| {
                    OptimizerError::CheckpointError(format!("Failed to deserialize checkpoint: {e}"))
                })?;
                checkpoint
        };

        Ok(checkpoint)
    }

    /// Get the latest checkpoint path
    pub fn latest_checkpoint(&self) -> Option<&PathBuf> {
        self.checkpoints.back().map(|(path, _)| path)
    }

    /// Get all checkpoint metadata
    pub fn list_checkpoints(&self) -> Vec<&CheckpointMetadata> {
        self.checkpoints
            .iter()
            .map(|(_, metadata)| metadata)
            .collect()
    }

    /// Increment current step
    pub fn step(&mut self) {
        self.current_step += 1;
    }

    /// Get current step
    pub fn current_step(&self) -> usize {
        self.current_step
    }

    /// Set current step
    pub fn set_step(&mut self, step: usize) {
        self.current_step = step;
    }

    /// Resume from the latest checkpoint
    pub fn resume_from_latest(&mut self) -> Result<Option<Checkpoint>, OptimizerError> {
        if let Some((checkpoint_path, metadata)) = self.checkpoints.back() {
            self.current_step = metadata.step;
            self.last_checkpoint_step = metadata.step;
            Ok(Some(self.load_checkpoint(checkpoint_path)?))
        } else {
            Ok(None)
        }
    }

    /// Clean up checkpoints manually
    pub fn cleanup(&mut self) -> Result<(), OptimizerError> {
        self.cleanup_old_checkpoints()
    }

    /// Get checkpoint statistics
    pub fn statistics(&self) -> CheckpointStatistics {
        let total_size = self
            .checkpoints
            .iter()
            .filter_map(|(path, _)| std::fs::metadata(path).ok())
            .map(|metadata| metadata.len())
            .sum();

        CheckpointStatistics {
            total_checkpoints: self.checkpoints.len(),
            total_size_bytes: total_size,
            current_step: self.current_step,
            last_checkpoint_step: self.last_checkpoint_step,
            next_checkpoint_step: self.last_checkpoint_step + self.config.checkpoint_frequency,
        }
    }

    // Private methods

    fn load_existing_checkpoints(&mut self) -> Result<(), OptimizerError> {
        if !self.config.checkpoint_dir.exists() {
            return Ok(());
        }

        let mut checkpoints = Vec::new();

        for entry in std::fs::read_dir(&self.config.checkpoint_dir).map_err(|e| {
            OptimizerError::CheckpointError(format!("Failed to read checkpoint directory: {e}"))
        })? {
            let entry = entry.map_err(|e| {
                OptimizerError::CheckpointError(format!("Failed to read directory entry: {e}"))
            })?;
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) == Some("bin") {
                if let Some(filename) = path.file_stem().and_then(|s| s.to_str()) {
                    if let Some(stripped) = filename.strip_prefix("checkpoint_step_") {
                        if let Ok(step) = stripped.parse::<usize>() {
                            // Try to load metadata from the checkpoint
                            if let Ok(checkpoint) = self.load_checkpoint(&path) {
                                checkpoints.push((path, checkpoint.metadata));
                            }
                        }
                    }
                }
            }
        }

        // Sort by step
        checkpoints.sort_by_key(|(_, metadata)| metadata.step);

        // Keep only the most recent checkpoints
        if checkpoints.len() > self.config.max_checkpoints {
            let to_remove = checkpoints.len() - self.config.max_checkpoints;
            for (path, _) in checkpoints.drain(..to_remove) {
                let _ = std::fs::remove_file(path);
            }
        }

        self.checkpoints = checkpoints.into();

        // Update current step to the latest checkpoint
        if let Some((_, metadata)) = self.checkpoints.back() {
            self.current_step = metadata.step;
            self.last_checkpoint_step = metadata.step;
        }
        Ok(())
    }

    fn save_checkpoint_sync(
        &self,
        checkpoint: &Checkpoint,
        path: &Path,
    ) -> Result<(), OptimizerError> {
        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                OptimizerError::CheckpointError(format!(
                    "Failed to create checkpoint directory: {e}"
                ))
            })?;
        }

        let data = bincode::serde::encode_to_vec(checkpoint, bincode::config::standard()).map_err(|e| {
            OptimizerError::CheckpointError(format!("Failed to serialize checkpoint: {e}"))
        })?;

        let final_data = if self.config.compress {
            self.compress_data(&data)?
        } else {
            data
        };

        std::fs::write(path, final_data).map_err(|e| {
            OptimizerError::CheckpointError(format!("Failed to write checkpoint: {e}"))
        })
    }

    fn save_checkpoint_async(
        &self,
        checkpoint: Checkpoint,
        path: PathBuf,
    ) -> Result<(), OptimizerError> {
        let compress = self.config.compress;

        std::thread::spawn(move || {
            let data = bincode::serde::encode_to_vec(&checkpoint, bincode::config::standard()).unwrap();
            let final_data = if compress {
                // Simple compression implementation
                data // For now, just use the original data
            } else {
                data
            };

            let _ = std::fs::write(path, final_data);
        });

        Ok(())
    }

    fn cleanup_old_checkpoints(&mut self) -> Result<(), OptimizerError> {
        while self.checkpoints.len() > self.config.max_checkpoints {
            if let Some((old_path, _)) = self.checkpoints.pop_front() {
                std::fs::remove_file(old_path).map_err(|e| {
                    OptimizerError::CheckpointError(format!("Failed to remove old checkpoint: {e}"))
                })?;
            }
        }
        Ok(())
    }

    fn compress_data(&self, data: &[u8]) -> Result<Vec<u8>, OptimizerError> {
        // Simple compression - in practice, you'd use a real compression library
        Ok(data.to_vec())
    }

    fn decompress_data(&self, data: &[u8]) -> Result<Vec<u8>, OptimizerError> {
        // Simple decompression - in practice, you'd use a real compression library
        Ok(data.to_vec())
    }
}

/// Checkpoint statistics
#[derive(Debug, Clone)]
pub struct CheckpointStatistics {
    pub total_checkpoints: usize,
    pub total_size_bytes: u64,
    pub current_step: usize,
    pub last_checkpoint_step: usize,
    pub next_checkpoint_step: usize,
}

impl std::fmt::Display for CheckpointStatistics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Checkpoint Statistics:")?;
        writeln!(f, "  Total Checkpoints: {}", self.total_checkpoints)?;
        writeln!(
            f,
            "  Total Size: {:.2} MB",
            self.total_size_bytes as f64 / 1024.0 / 1024.0
        )?;
        writeln!(f, "  Current Step: {}", self.current_step)?;
        writeln!(f, "  Last Checkpoint: {}", self.last_checkpoint_step)?;
        writeln!(f, "  Next Checkpoint: {}", self.next_checkpoint_step)?;
        Ok(())
    }
}

/// Trait for optimizers that support checkpointing
pub trait CheckpointSupport {
    /// Save optimizer state for checkpointing
    fn save_state_for_checkpoint(&self) -> Result<Vec<u8>, OptimizerError>;

    /// Load optimizer state from checkpoint
    fn load_state_from_checkpoint(&mut self, data: &[u8]) -> Result<(), OptimizerError>;

    /// Get current gradients for checkpointing
    fn get_gradients_for_checkpoint(&self) -> Option<HashMap<String, Vec<f32>>>;

    /// Get current model parameters for checkpointing
    fn get_parameters_for_checkpoint(&self) -> Option<HashMap<String, Vec<f32>>>;
}

/// Wrapper optimizer that adds checkpointing support
pub struct CheckpointingOptimizer<T> {
    inner: T,
    checkpoint_manager: CheckpointManager,
    auto_checkpoint: bool,
}

impl<T> CheckpointingOptimizer<T>
where
    T: CheckpointSupport,
{
    /// Create a new checkpointing optimizer wrapper
    pub fn new(inner: T, config: CheckpointConfig) -> Result<Self, OptimizerError> {
        let checkpoint_manager = CheckpointManager::new(config)?;

        Ok(Self {
            inner,
            checkpoint_manager,
            auto_checkpoint: true,
        })
    }

    /// Enable or disable automatic checkpointing
    pub fn set_auto_checkpoint(&mut self, enabled: bool) {
        self.auto_checkpoint = enabled;
    }

    /// Get the inner optimizer
    pub fn inner(&self) -> &T {
        &self.inner
    }

    /// Get the inner optimizer mutably
    pub fn inner_mut(&mut self) -> &mut T {
        &mut self.inner
    }

    /// Get the checkpoint manager
    pub fn checkpoint_manager(&self) -> &CheckpointManager {
        &self.checkpoint_manager
    }

    /// Get the checkpoint manager mutably
    pub fn checkpoint_manager_mut(&mut self) -> &mut CheckpointManager {
        &mut self.checkpoint_manager
    }

    /// Manually save a checkpoint
    pub fn save_checkpoint(
        &mut self,
        loss: Option<f32>,
        learning_rate: f32,
        epoch: Option<usize>,
    ) -> Result<PathBuf, OptimizerError> {
        let optimizer_state = self.inner.save_state_for_checkpoint()?;
        let gradients = self.inner.get_gradients_for_checkpoint();
        let parameters = self.inner.get_parameters_for_checkpoint();

        self.checkpoint_manager.save_checkpoint(
            Some(&optimizer_state),
            gradients.as_ref(),
            parameters.as_ref(),
            loss,
            learning_rate,
            epoch,
        )
    }

    /// Resume from the latest checkpoint
    pub fn resume_from_latest(&mut self) -> Result<bool, OptimizerError> {
        if let Some(checkpoint) = self.checkpoint_manager.resume_from_latest()? {
            if let Some(optimizer_state) = &checkpoint.optimizer_state {
                self.inner.load_state_from_checkpoint(optimizer_state)?;
            }
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Step the optimizer and handle checkpointing
    pub fn step_with_checkpoint(
        &mut self,
        loss: Option<f32>,
        learning_rate: f32,
        epoch: Option<usize>,
    ) -> Result<Option<PathBuf>, OptimizerError> {
        // Step the checkpoint manager
        self.checkpoint_manager.step();

        let checkpoint_path = if self.auto_checkpoint && self.checkpoint_manager.should_checkpoint()
        {
            Some(self.save_checkpoint(loss, learning_rate, epoch)?)
        } else {
            None
        };

        Ok(checkpoint_path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[derive(Debug)]
    struct MockOptimizer {
        state: HashMap<String, f32>,
    }

    impl CheckpointSupport for MockOptimizer {
        fn save_state_for_checkpoint(&self) -> Result<Vec<u8>, OptimizerError> {
            Ok(bincode::serde::encode_to_vec(&self.state, bincode::config::standard()).unwrap())
        }

        fn load_state_from_checkpoint(&mut self, data: &[u8]) -> Result<(), OptimizerError> {
            let (state, _): (HashMap<String, f32>, usize) = bincode::serde::decode_from_slice(data, bincode::config::standard()).unwrap();
            self.state = state;
            Ok(())
        }

        fn get_gradients_for_checkpoint(&self) -> Option<HashMap<String, Vec<f32>>> {
            None
        }

        fn get_parameters_for_checkpoint(&self) -> Option<HashMap<String, Vec<f32>>> {
            None
        }
    }

    #[test]
    fn test_checkpoint_manager() -> OptimizerResult<()> {
        let temp_dir = tempfile::tempdir()?;
        let config = CheckpointConfig {
            checkpoint_dir: temp_dir.path().to_path_buf(),
            max_checkpoints: 3,
            checkpoint_frequency: 2,
            async_save: false, // Use synchronous saving for tests
            ..Default::default()
        };

        let mut manager = CheckpointManager::new(config)?;

        // Test checkpoint frequency
        assert!(!manager.should_checkpoint()); // step 0
        manager.step();
        assert!(!manager.should_checkpoint()); // step 1
        manager.step();
        assert!(manager.should_checkpoint()); // step 2

        // Save a checkpoint
        let checkpoint_path =
            manager.save_checkpoint(None, None, None, Some(0.5), 0.01, Some(1))?;

        assert!(checkpoint_path.exists());

        // Load the checkpoint
        let loaded = manager.load_checkpoint(&checkpoint_path)?;
        assert_eq!(loaded.metadata.step, 2);
        assert_eq!(loaded.metadata.loss, Some(0.5));
        Ok(())
    }

    #[test]
    fn test_checkpointing_optimizer() -> OptimizerResult<()> {
        let temp_dir = tempfile::tempdir()?;
        let config = CheckpointConfig {
            checkpoint_dir: temp_dir.path().to_path_buf(),
            checkpoint_frequency: 1,
            async_save: false, // Use synchronous saving for tests
            ..Default::default()
        };

        let optimizer = MockOptimizer {
            state: [("lr".to_string(), 0.01)].iter().cloned().collect(),
        };

        let mut checkpointing_optimizer = CheckpointingOptimizer::new(optimizer, config)?;

        // Test step with checkpoint
        let checkpoint_path =
            checkpointing_optimizer.step_with_checkpoint(Some(0.5), 0.01, Some(1))?;

        assert!(checkpoint_path.is_some());

        // Test resume
        let resumed = checkpointing_optimizer.resume_from_latest()?;
        assert!(resumed);
        Ok(())
    }
}
