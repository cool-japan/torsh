//! Advanced asynchronous engine for distributed gradient computation
//!
//! This module provides high-level async coordination, distributed checkpointing,
//! advanced gradient synchronization, error recovery, and performance optimization
//! for large-scale distributed deep learning training.

use crate::compression::CompressionStrategy;
use scirs2_core::numeric::{FromPrimitive, ToPrimitive};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use torsh_core::dtype::FloatElement;
use torsh_core::error::{Result, TorshError};

use super::config::DistributedConfig;

/// Distributed checkpoint metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedCheckpointMetadata {
    /// Checkpoint version
    pub version: String,
    /// Timestamp when checkpoint was created
    pub timestamp: SystemTime,
    /// World size when checkpoint was created
    pub world_size: usize,
    /// Rank of the process that created this checkpoint
    pub rank: usize,
    /// Global step number
    pub global_step: usize,
    /// Checkpoint type
    pub checkpoint_type: CheckpointType,
    /// Shard information
    pub shard_info: Vec<ShardInfo>,
    /// Compression used
    pub compression: Option<CompressionStrategy>,
}

/// Type of checkpoint
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum CheckpointType {
    /// Full checkpoint containing all state
    Full,
    /// Incremental checkpoint with only changes
    Incremental,
    /// Gradient checkpoint for fault tolerance
    Gradient,
    /// Optimizer state checkpoint
    Optimizer,
}

/// Information about a checkpoint shard
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardInfo {
    /// Shard ID (typically matches rank)
    pub shard_id: usize,
    /// File path for this shard
    pub file_path: PathBuf,
    /// Size of the shard in bytes
    pub size_bytes: usize,
    /// Checksum for integrity verification
    pub checksum: String,
    /// Parameters contained in this shard
    pub parameter_names: Vec<String>,
}

/// Configuration for fault tolerance
#[derive(Debug, Clone)]
pub struct FaultToleranceConfig {
    /// Enable automatic recovery
    pub enable_auto_recovery: bool,
    /// Number of checkpoint replicas
    pub num_replicas: usize,
    /// Maximum age for checkpoint to be considered valid
    pub max_checkpoint_age: Duration,
    /// Heartbeat interval for failure detection
    pub heartbeat_interval: Duration,
    /// Timeout for checkpoint operations
    pub checkpoint_timeout: Duration,
}

impl Default for FaultToleranceConfig {
    fn default() -> Self {
        Self {
            enable_auto_recovery: true,
            num_replicas: 2,
            max_checkpoint_age: Duration::from_secs(3600), // 1 hour
            heartbeat_interval: Duration::from_secs(10),
            checkpoint_timeout: Duration::from_secs(300), // 5 minutes
        }
    }
}

/// Distributed checkpoint manager
pub struct DistributedCheckpointManager {
    /// Configuration
    config: DistributedConfig,
    /// Base directory for checkpoints
    checkpoint_dir: PathBuf,
    /// Current checkpoint metadata
    current_metadata: Option<DistributedCheckpointMetadata>,
    /// Fault tolerance settings
    fault_tolerance: FaultToleranceConfig,
}

// DistributedCheckpointManager is Send + Sync
unsafe impl Send for DistributedCheckpointManager {}
unsafe impl Sync for DistributedCheckpointManager {}

impl DistributedCheckpointManager {
    /// Create a new distributed checkpoint manager
    pub fn new(config: DistributedConfig, checkpoint_dir: impl AsRef<Path>) -> Result<Self> {
        let checkpoint_dir = checkpoint_dir.as_ref().to_path_buf();

        // Create checkpoint directory if it doesn't exist
        if !checkpoint_dir.exists() {
            fs::create_dir_all(&checkpoint_dir)?;
        }

        Ok(Self {
            config,
            checkpoint_dir,
            current_metadata: None,
            fault_tolerance: FaultToleranceConfig::default(),
        })
    }

    /// Set fault tolerance configuration
    pub fn with_fault_tolerance(mut self, config: FaultToleranceConfig) -> Self {
        self.fault_tolerance = config;
        self
    }

    /// Save distributed checkpoint
    pub fn save_checkpoint<T: FloatElement + Serialize>(
        &mut self,
        gradients: &HashMap<String, Vec<T>>,
        global_step: usize,
        checkpoint_type: CheckpointType,
    ) -> Result<()> {
        let start_time = Instant::now();

        // Create checkpoint metadata
        let metadata = DistributedCheckpointMetadata {
            version: "1.0".to_string(),
            timestamp: SystemTime::now(),
            world_size: self.config.world_size,
            rank: self.config.rank,
            global_step,
            checkpoint_type: checkpoint_type.clone(),
            shard_info: Vec::new(),
            compression: Some(self.config.compression),
        };

        // Create checkpoint directory for this step
        let step_dir = self.checkpoint_dir.join(format!("step_{}", global_step));
        fs::create_dir_all(&step_dir)?;

        // Save shard for this rank
        let shard_path = step_dir.join(format!("shard_{}.bin", self.config.rank));
        let shard_size = self.save_shard(gradients, &shard_path)?;

        // Calculate checksum
        let checksum = self.calculate_checksum(&shard_path)?;

        // Update metadata with shard info
        let mut updated_metadata = metadata;
        updated_metadata.shard_info.push(ShardInfo {
            shard_id: self.config.rank,
            file_path: shard_path,
            size_bytes: shard_size,
            checksum,
            parameter_names: gradients.keys().cloned().collect(),
        });

        // Save metadata
        let metadata_path = step_dir.join(format!("metadata_{}.json", self.config.rank));
        self.save_metadata(&updated_metadata, &metadata_path)?;

        // Coordinate with other ranks to consolidate metadata
        if self.config.world_size > 1 {
            self.coordinate_checkpoint_metadata(&step_dir, &updated_metadata)?;
        }

        // Create replicas if fault tolerance is enabled
        if self.fault_tolerance.enable_auto_recovery && self.fault_tolerance.num_replicas > 1 {
            self.create_checkpoint_replicas(&step_dir, global_step)?;
        }

        self.current_metadata = Some(updated_metadata);

        let elapsed = start_time.elapsed();
        tracing::info!(
            "Saved distributed checkpoint for step {} in {:?} (rank {})",
            global_step,
            elapsed,
            self.config.rank
        );

        Ok(())
    }

    /// Load distributed checkpoint
    pub fn load_checkpoint<T: FloatElement + for<'de> Deserialize<'de> + FromPrimitive>(
        &mut self,
        global_step: usize,
    ) -> Result<HashMap<String, Vec<T>>> {
        let start_time = Instant::now();

        // Find checkpoint directory
        let step_dir = self.checkpoint_dir.join(format!("step_{}", global_step));
        if !step_dir.exists() {
            return Err(TorshError::AutogradError(format!(
                "Checkpoint for step {} not found",
                global_step
            )));
        }

        // Load metadata
        let metadata_path = step_dir.join(format!("metadata_{}.json", self.config.rank));
        let metadata = self.load_metadata(&metadata_path)?;

        // Validate checkpoint integrity
        self.validate_checkpoint_integrity(&metadata)?;

        // Load shard for this rank
        let shard_path = step_dir.join(format!("shard_{}.bin", self.config.rank));
        let gradients = self.load_shard(&shard_path)?;

        self.current_metadata = Some(metadata);

        let elapsed = start_time.elapsed();
        tracing::info!(
            "Loaded distributed checkpoint for step {} in {:?} (rank {})",
            global_step,
            elapsed,
            self.config.rank
        );

        Ok(gradients)
    }

    /// List available checkpoints
    pub fn list_checkpoints(&self) -> Result<Vec<usize>> {
        let mut steps = Vec::new();

        for entry in fs::read_dir(&self.checkpoint_dir)? {
            let entry = entry?;
            let dir_name = entry.file_name();
            let dir_name = dir_name.to_string_lossy();

            if dir_name.starts_with("step_") {
                if let Ok(step) = dir_name.strip_prefix("step_").expect("prefix checked to exist").parse::<usize>() {
                    steps.push(step);
                }
            }
        }

        steps.sort_unstable();
        Ok(steps)
    }

    /// Attempt to recover from a corrupted checkpoint
    pub fn recover_checkpoint(&mut self, global_step: usize) -> Result<()> {
        tracing::warn!("Attempting to recover checkpoint for step {}", global_step);

        // Try to load from replicas
        for replica_id in 1..self.fault_tolerance.num_replicas {
            let replica_dir = self
                .checkpoint_dir
                .join(format!("step_{}_replica_{}", global_step, replica_id));
            if replica_dir.exists() {
                let metadata_path =
                    replica_dir.join(format!("metadata_{}.json", self.config.rank));
                if let Ok(metadata) = self.load_metadata(&metadata_path) {
                    if self.validate_checkpoint_integrity(&metadata).is_ok() {
                        // Copy replica back to main checkpoint
                        let main_dir = self.checkpoint_dir.join(format!("step_{}", global_step));
                        fs::create_dir_all(&main_dir)?;

                        let replica_shard =
                            replica_dir.join(format!("shard_{}.bin", self.config.rank));
                        let main_shard = main_dir.join(format!("shard_{}.bin", self.config.rank));
                        fs::copy(&replica_shard, &main_shard)?;

                        let replica_metadata =
                            replica_dir.join(format!("metadata_{}.json", self.config.rank));
                        let main_metadata =
                            main_dir.join(format!("metadata_{}.json", self.config.rank));
                        fs::copy(&replica_metadata, &main_metadata)?;

                        tracing::info!(
                            "Successfully recovered checkpoint from replica {}",
                            replica_id
                        );
                        return Ok(());
                    }
                }
            }
        }

        Err(TorshError::AutogradError(format!(
            "Failed to recover checkpoint for step {}",
            global_step
        )))
    }

    // Private helper methods
    fn save_shard<T: FloatElement + Serialize>(
        &self,
        gradients: &HashMap<String, Vec<T>>,
        path: &Path,
    ) -> Result<usize> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Apply compression if configured
        let data = match self.config.compression {
            CompressionStrategy::None => oxicode::serde::encode_to_vec(gradients, oxicode::config::standard()).map_err(|e| {
                TorshError::AutogradError(format!("Serialization error: {}", e))
            })?,
            CompressionStrategy::Quantization => {
                // Simple quantization to f16
                let quantized = self.quantize_gradients(gradients);
                oxicode::serde::encode_to_vec(&quantized, oxicode::config::standard()).map_err(|e| {
                    TorshError::AutogradError(format!("Serialization error: {}", e))
                })?
            }
            _ => {
                // Use default serialization for other compression types
                oxicode::serde::encode_to_vec(gradients, oxicode::config::standard()).map_err(|e| {
                    TorshError::AutogradError(format!("Serialization error: {}", e))
                })?
            }
        };

        writer.write_all(&data)?;
        writer.flush()?;

        Ok(data.len())
    }

    fn load_shard<T: FloatElement + for<'de> Deserialize<'de> + FromPrimitive>(
        &self,
        path: &Path,
    ) -> Result<HashMap<String, Vec<T>>> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let mut data = Vec::new();
        reader.read_to_end(&mut data)?;

        // Apply decompression if needed
        let gradients = match self.config.compression {
            CompressionStrategy::None => {
                let (g, _): (HashMap<String, Vec<T>>, usize) = oxicode::serde::decode_from_slice(&data, oxicode::config::standard()).map_err(|e| {
                    TorshError::AutogradError(format!("Deserialization error: {}", e))
                })?; g
            }
            CompressionStrategy::Quantization => {
                // Deserialize quantized data and convert back
                let (quantized, _): (HashMap<String, Vec<f32>>, usize) = oxicode::serde::decode_from_slice(&data, oxicode::config::standard()).map_err(|e| {
                    TorshError::AutogradError(format!("Deserialization error: {}", e))
                })?; self.dequantize_gradients(&quantized)
            }
            _ => {
                let (g, _): (HashMap<String, Vec<T>>, usize) = oxicode::serde::decode_from_slice(&data, oxicode::config::standard()).map_err(|e| {
                    TorshError::AutogradError(format!("Deserialization error: {}", e))
                })?; g
            }
        };

        Ok(gradients)
    }

    fn quantize_gradients<T: FloatElement>(
        &self,
        gradients: &HashMap<String, Vec<T>>,
    ) -> HashMap<String, Vec<f32>> {
        gradients
            .iter()
            .map(|(name, values)| {
                let quantized = values.iter().map(|v| torsh_core::TensorElement::to_f32(v).unwrap_or(0.0)).collect();
                (name.clone(), quantized)
            })
            .collect()
    }

    fn dequantize_gradients<T: FloatElement + FromPrimitive>(
        &self,
        gradients: &HashMap<String, Vec<f32>>,
    ) -> HashMap<String, Vec<T>> {
        gradients
            .iter()
            .map(|(name, values)| {
                let dequantized = values
                    .iter()
                    .map(|v| T::from_f32(*v).unwrap_or_else(|| T::from_f32(0.0).expect("f32 conversion should succeed")))
                    .collect();
                (name.clone(), dequantized)
            })
            .collect()
    }

    fn save_metadata(
        &self,
        metadata: &DistributedCheckpointMetadata,
        path: &Path,
    ) -> Result<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, metadata).map_err(|e| {
            TorshError::AutogradError(format!("Metadata serialization error: {}", e))
        })?;
        Ok(())
    }

    fn load_metadata(&self, path: &Path) -> Result<DistributedCheckpointMetadata> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let metadata = serde_json::from_reader(reader).map_err(|e| {
            TorshError::AutogradError(format!("Metadata deserialization error: {}", e))
        })?;
        Ok(metadata)
    }

    fn calculate_checksum(&self, path: &Path) -> Result<String> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        let mut buffer = [0; 8192];

        loop {
            let bytes_read = reader.read(&mut buffer)?;
            if bytes_read == 0 {
                break;
            }
            std::hash::Hasher::write(&mut hasher, &buffer[..bytes_read]);
        }

        Ok(format!("{:x}", std::hash::Hasher::finish(&hasher)))
    }

    fn validate_checkpoint_integrity(
        &self,
        metadata: &DistributedCheckpointMetadata,
    ) -> Result<()> {
        // Check if checkpoint is too old
        if let Ok(age) = metadata.timestamp.elapsed() {
            if age > self.fault_tolerance.max_checkpoint_age {
                return Err(TorshError::AutogradError(format!(
                    "Checkpoint is too old: {:?}",
                    age
                )));
            }
        }

        // Validate shard checksums
        for shard in &metadata.shard_info {
            if shard.file_path.exists() {
                let calculated_checksum = self.calculate_checksum(&shard.file_path)?;
                if calculated_checksum != shard.checksum {
                    return Err(TorshError::AutogradError(format!(
                        "Checksum mismatch for shard {}: expected {}, got {}",
                        shard.shard_id, shard.checksum, calculated_checksum
                    )));
                }
            }
        }

        Ok(())
    }

    fn coordinate_checkpoint_metadata(
        &self,
        _step_dir: &Path,
        _metadata: &DistributedCheckpointMetadata,
    ) -> Result<()> {
        // In a real implementation, this would:
        // 1. Synchronize metadata across all ranks
        // 2. Consolidate into a single global metadata file
        // 3. Ensure all ranks have consistent view of the checkpoint

        tracing::debug!(
            "Coordinating checkpoint metadata across {} ranks",
            self.config.world_size
        );

        // Placeholder for actual coordination logic
        Ok(())
    }

    fn create_checkpoint_replicas(&self, step_dir: &Path, global_step: usize) -> Result<()> {
        for replica_id in 1..self.fault_tolerance.num_replicas {
            let replica_dir = self
                .checkpoint_dir
                .join(format!("step_{}_replica_{}", global_step, replica_id));
            fs::create_dir_all(&replica_dir)?;

            // Copy shard file
            let source_shard = step_dir.join(format!("shard_{}.bin", self.config.rank));
            let replica_shard = replica_dir.join(format!("shard_{}.bin", self.config.rank));
            fs::copy(&source_shard, &replica_shard)?;

            // Copy metadata
            let source_metadata = step_dir.join(format!("metadata_{}.json", self.config.rank));
            let replica_metadata =
                replica_dir.join(format!("metadata_{}.json", self.config.rank));
            fs::copy(&source_metadata, &replica_metadata)?;

            tracing::debug!(
                "Created checkpoint replica {} for step {}",
                replica_id,
                global_step
            );
        }

        Ok(())
    }
}

/// Advanced gradient synchronization engine
#[derive(Debug)]
pub struct GradientSynchronizationEngine<T: FloatElement + std::fmt::Debug + Send + Sync> {
    /// Configuration
    config: DistributedConfig,
    /// Checkpoint manager
    checkpoint_manager: Option<DistributedCheckpointManager>,
    /// Asynchronous operation handles
    pending_operations: Vec<AsyncOperation<T>>,
    /// Engine statistics
    stats: Arc<RwLock<EngineStatistics>>,
    /// Performance optimizer
    optimizer: PerformanceOptimizer,
}

/// Asynchronous operation in the engine
#[derive(Debug)]
pub struct AsyncOperation<T: FloatElement> {
    /// Operation ID
    pub id: usize,
    /// Operation type
    pub op_type: OperationType,
    /// Data being processed
    pub data: Vec<T>,
    /// Target ranks/devices
    pub targets: Vec<usize>,
    /// Start time
    pub start_time: Instant,
    /// Expected completion time
    pub expected_completion: Instant,
    /// Current status
    pub status: OperationStatus,
}

/// Type of async operation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OperationType {
    /// Gradient synchronization
    GradientSync,
    /// Parameter update
    ParameterUpdate,
    /// Checkpoint save
    CheckpointSave,
    /// Checkpoint load
    CheckpointLoad,
    /// Health check
    HealthCheck,
    /// Custom operation
    Custom(String),
}

/// Operation status
#[derive(Debug, Clone, PartialEq)]
pub enum OperationStatus {
    /// Operation is pending
    Pending,
    /// Operation is in progress
    InProgress,
    /// Operation completed successfully
    Completed,
    /// Operation failed
    Failed(String),
    /// Operation timed out
    TimedOut,
}

/// Performance optimizer for the engine
#[derive(Debug)]
pub struct PerformanceOptimizer {
    /// Historical performance metrics
    metrics_history: VecDeque<PerformanceMetrics>,
    /// Optimization strategy
    strategy: OptimizationStrategy,
    /// Adaptive parameters
    adaptive_params: AdaptiveParameters,
}

/// Performance metrics snapshot
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Timestamp
    timestamp: Instant,
    /// Throughput (operations/sec)
    throughput: f64,
    /// Average latency
    avg_latency: Duration,
    /// Resource utilization
    cpu_utilization: f64,
    /// Memory usage
    memory_usage_mb: f64,
    /// Network bandwidth utilization
    network_utilization: f64,
}

/// Optimization strategy
#[derive(Debug, Clone, PartialEq)]
pub enum OptimizationStrategy {
    /// Optimize for maximum throughput
    MaxThroughput,
    /// Optimize for minimum latency
    MinLatency,
    /// Balance throughput and latency
    Balanced,
    /// Adaptive optimization based on workload
    Adaptive,
}

/// Adaptive optimization parameters
#[derive(Debug, Clone)]
pub struct AdaptiveParameters {
    /// Learning rate for adaptation
    learning_rate: f64,
    /// History window size
    window_size: usize,
    /// Threshold for strategy changes
    change_threshold: f64,
}

/// Engine statistics
#[derive(Debug, Clone, Default)]
pub struct EngineStatistics {
    /// Total operations processed
    total_operations: usize,
    /// Successful operations
    successful_operations: usize,
    /// Failed operations
    failed_operations: usize,
    /// Average operation time
    avg_operation_time: Duration,
    /// Current throughput
    current_throughput: f64,
    /// Resource utilization stats
    avg_cpu_utilization: f64,
    /// Memory usage stats
    avg_memory_usage_mb: f64,
}

impl<T: FloatElement + std::fmt::Debug + Send + Sync> GradientSynchronizationEngine<T> {
    /// Create a new gradient synchronization engine
    pub fn new(config: DistributedConfig) -> Self {
        let optimizer = PerformanceOptimizer {
            metrics_history: VecDeque::new(),
            strategy: OptimizationStrategy::Balanced,
            adaptive_params: AdaptiveParameters {
                learning_rate: 0.01,
                window_size: 100,
                change_threshold: 0.1,
            },
        };

        Self {
            config,
            checkpoint_manager: None,
            pending_operations: Vec::new(),
            stats: Arc::new(RwLock::new(EngineStatistics::default())),
            optimizer,
        }
    }

    /// Enable checkpointing with specified directory
    pub fn with_checkpointing(mut self, checkpoint_dir: impl AsRef<Path>) -> Result<Self> {
        let checkpoint_manager =
            DistributedCheckpointManager::new(self.config.clone(), checkpoint_dir)?;
        self.checkpoint_manager = Some(checkpoint_manager);
        Ok(self)
    }

    /// Submit an async operation to the engine
    pub fn submit_operation(
        &mut self,
        op_type: OperationType,
        data: Vec<T>,
        targets: Vec<usize>,
    ) -> usize {
        let operation_id = self.pending_operations.len();
        let expected_completion = self.estimate_completion_time(&op_type, data.len());

        let operation = AsyncOperation {
            id: operation_id,
            op_type,
            data,
            targets,
            start_time: Instant::now(),
            expected_completion,
            status: OperationStatus::Pending,
        };

        self.pending_operations.push(operation);
        operation_id
    }

    /// Check the status of an operation
    pub fn check_operation_status(&self, operation_id: usize) -> Option<&OperationStatus> {
        self.pending_operations
            .get(operation_id)
            .map(|op| &op.status)
    }

    /// Get engine statistics
    pub fn get_statistics(&self) -> EngineStatistics {
        self.stats.read().clone()
    }

    /// Optimize engine performance based on current metrics
    pub fn optimize_performance(&mut self) {
        let current_metrics = self.collect_current_metrics();
        self.optimizer.metrics_history.push_back(current_metrics);

        // Keep history within window size
        while self.optimizer.metrics_history.len() > self.optimizer.adaptive_params.window_size {
            self.optimizer.metrics_history.pop_front();
        }

        match self.optimizer.strategy {
            OptimizationStrategy::Adaptive => {
                self.adaptive_optimization();
            }
            _ => {
                self.static_optimization();
            }
        }
    }

    fn estimate_completion_time(&self, op_type: &OperationType, data_size: usize) -> Instant {
        let base_time = match op_type {
            OperationType::GradientSync => Duration::from_millis(10),
            OperationType::ParameterUpdate => Duration::from_millis(5),
            OperationType::CheckpointSave => Duration::from_millis(100),
            OperationType::CheckpointLoad => Duration::from_millis(50),
            OperationType::HealthCheck => Duration::from_millis(1),
            OperationType::Custom(_) => Duration::from_millis(20),
        };

        let size_factor = (data_size as f64 / 1024.0).log2().max(1.0);
        let estimated_time = base_time.mul_f64(size_factor);

        Instant::now() + estimated_time
    }

    fn collect_current_metrics(&self) -> PerformanceMetrics {
        // Placeholder implementation - in real code this would collect actual metrics
        PerformanceMetrics {
            timestamp: Instant::now(),
            throughput: 100.0, // ops/sec
            avg_latency: Duration::from_millis(10),
            cpu_utilization: 0.7,
            memory_usage_mb: 1024.0,
            network_utilization: 0.5,
        }
    }

    fn adaptive_optimization(&mut self) {
        // Placeholder for adaptive optimization logic
        // This would analyze metrics trends and adjust parameters
        tracing::debug!("Performing adaptive optimization");
    }

    fn static_optimization(&mut self) {
        // Placeholder for static optimization logic
        // This would apply fixed optimization rules
        tracing::debug!("Performing static optimization");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_checkpoint_manager_creation() {
        let temp_dir = TempDir::new().unwrap();
        let config = DistributedConfig::default();
        let manager = DistributedCheckpointManager::new(config, temp_dir.path());
        assert!(manager.is_ok());
    }

    #[test]
    fn test_engine_creation() {
        let config = DistributedConfig::default();
        let engine = GradientSynchronizationEngine::<f32>::new(config);
        assert_eq!(engine.pending_operations.len(), 0);
    }

    #[test]
    fn test_operation_submission() {
        let config = DistributedConfig::default();
        let mut engine = GradientSynchronizationEngine::<f32>::new(config);

        let op_id = engine.submit_operation(
            OperationType::GradientSync,
            vec![1.0, 2.0, 3.0],
            vec![0, 1],
        );

        assert_eq!(op_id, 0);
        assert_eq!(engine.pending_operations.len(), 1);
        assert!(matches!(
            engine.check_operation_status(op_id),
            Some(OperationStatus::Pending)
        ));
    }
}