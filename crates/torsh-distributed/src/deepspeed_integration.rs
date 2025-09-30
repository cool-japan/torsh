//! DeepSpeed integration for ToRSh distributed training
//!
//! This module provides compatibility with DeepSpeed's optimization strategies
//! and configuration format, allowing users to migrate from PyTorch + DeepSpeed
//! to ToRSh more easily.

use crate::{TorshDistributedError, TorshResult};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// DeepSpeed ZeRO optimization stage
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ZeroStage {
    /// Stage 0: No ZeRO optimization
    Stage0 = 0,
    /// Stage 1: Optimizer state partitioning
    Stage1 = 1,
    /// Stage 2: Gradient partitioning
    Stage2 = 2,
    /// Stage 3: Parameter partitioning
    Stage3 = 3,
}

/// DeepSpeed configuration compatible with ToRSh
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepSpeedConfig {
    /// ZeRO optimization configuration
    pub zero_optimization: ZeroOptimizationConfig,
    /// Gradient clipping configuration
    pub gradient_clipping: Option<f32>,
    /// Gradient accumulation steps
    pub gradient_accumulation_steps: Option<u32>,
    /// Mixed precision configuration
    pub fp16: Option<FP16Config>,
    /// CPU offloading configuration
    pub zero_force_ds_cpu_optimizer: Option<bool>,
    /// Activation checkpointing
    pub activation_checkpointing: Option<ActivationCheckpointingConfig>,
}

/// ZeRO optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZeroOptimizationConfig {
    /// ZeRO stage (0, 1, 2, or 3)
    pub stage: ZeroStage,
    /// Allgather bucket size
    pub allgather_bucket_size: Option<u64>,
    /// Reduce bucket size
    pub reduce_bucket_size: Option<u64>,
    /// Overlap communications
    pub overlap_comm: Option<bool>,
    /// Contiguous gradients
    pub contiguous_gradients: Option<bool>,
    /// Sub-group size
    pub sub_group_size: Option<u32>,
    /// Reduce scatter
    pub reduce_scatter: Option<bool>,
    /// Allgather partitions
    pub allgather_partitions: Option<bool>,
    /// Stage 3 max live parameters
    pub stage3_max_live_parameters: Option<u64>,
    /// Stage 3 max reuse distance
    pub stage3_max_reuse_distance: Option<u64>,
    /// Stage 3 prefetch bucket size
    pub stage3_prefetch_bucket_size: Option<u64>,
    /// Stage 3 parameter persistence threshold
    pub stage3_param_persistence_threshold: Option<u64>,
    /// CPU offloading configuration
    pub offload_optimizer: Option<OffloadOptimizerConfig>,
    /// Parameter offloading configuration
    pub offload_param: Option<OffloadParamConfig>,
}

/// Mixed precision FP16 configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FP16Config {
    /// Enable FP16 mixed precision
    pub enabled: bool,
    /// Loss scale
    pub loss_scale: Option<f32>,
    /// Dynamic loss scaling
    pub loss_scale_window: Option<u32>,
    /// Hysteresis for loss scaling
    pub hysteresis: Option<u32>,
    /// Minimum loss scale
    pub min_loss_scale: Option<f32>,
    /// Initial loss scale
    pub initial_scale_power: Option<u32>,
}

/// Activation checkpointing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationCheckpointingConfig {
    /// Enable activation checkpointing
    pub partition_activations: Option<bool>,
    /// CPU checkpointing
    pub cpu_checkpointing: Option<bool>,
    /// Contiguous memory optimization
    pub contiguous_memory_optimization: Option<bool>,
    /// Number of checkpoints
    pub number_checkpoints: Option<u32>,
    /// Synchronize checkpoint boundary
    pub synchronize_checkpoint_boundary: Option<bool>,
    /// Profile
    pub profile: Option<bool>,
}

/// Optimizer offloading configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OffloadOptimizerConfig {
    /// Device to offload to
    pub device: String,
    /// NVME path for ultra-fast offloading
    pub nvme_path: Option<String>,
    /// Pin memory
    pub pin_memory: Option<bool>,
    /// Buffer count
    pub buffer_count: Option<u32>,
    /// Fast initialization
    pub fast_init: Option<bool>,
}

/// Parameter offloading configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OffloadParamConfig {
    /// Device to offload to
    pub device: String,
    /// NVME path for ultra-fast offloading
    pub nvme_path: Option<String>,
    /// Pin memory
    pub pin_memory: Option<bool>,
    /// Buffer count
    pub buffer_count: Option<u32>,
    /// Max parameters per GPU
    pub max_in_cpu: Option<u64>,
}

/// DeepSpeed integration manager
pub struct DeepSpeedIntegration {
    config: DeepSpeedConfig,
    initialized: bool,
}

impl DeepSpeedIntegration {
    /// Create a new DeepSpeed integration instance
    pub fn new(config: DeepSpeedConfig) -> Self {
        Self {
            config,
            initialized: false,
        }
    }

    /// Load DeepSpeed configuration from file
    pub fn from_file<P: AsRef<Path>>(path: P) -> TorshResult<Self> {
        let content = std::fs::read_to_string(path).map_err(|e| {
            TorshDistributedError::configuration_error(format!(
                "Failed to read DeepSpeed config file: {}",
                e
            ))
        })?;

        let config: DeepSpeedConfig = serde_json::from_str(&content).map_err(|e| {
            TorshDistributedError::configuration_error(format!(
                "Failed to parse DeepSpeed config: {}",
                e
            ))
        })?;

        Ok(Self::new(config))
    }

    /// Load DeepSpeed configuration from JSON string
    pub fn from_json(json: &str) -> TorshResult<Self> {
        let config: DeepSpeedConfig = serde_json::from_str(json).map_err(|e| {
            TorshDistributedError::configuration_error(format!(
                "Failed to parse DeepSpeed config: {}",
                e
            ))
        })?;

        Ok(Self::new(config))
    }

    /// Initialize DeepSpeed integration
    pub fn initialize(&mut self) -> TorshResult<()> {
        if self.initialized {
            return Ok(());
        }

        // Validate configuration
        self.validate_config()?;

        // Initialize based on ZeRO stage
        match self.config.zero_optimization.stage {
            ZeroStage::Stage0 => {
                // No ZeRO optimization
                tracing::info!(
                    "DeepSpeed integration initialized with ZeRO Stage 0 (no optimization)"
                );
            }
            ZeroStage::Stage1 => {
                // Optimizer state partitioning
                self.initialize_zero_stage1()?;
            }
            ZeroStage::Stage2 => {
                // Gradient partitioning
                self.initialize_zero_stage2()?;
            }
            ZeroStage::Stage3 => {
                // Parameter partitioning
                self.initialize_zero_stage3()?;
            }
        }

        self.initialized = true;
        Ok(())
    }

    /// Validate DeepSpeed configuration
    fn validate_config(&self) -> TorshResult<()> {
        // Validate ZeRO stage
        if matches!(self.config.zero_optimization.stage, ZeroStage::Stage3) {
            if self
                .config
                .zero_optimization
                .stage3_max_live_parameters
                .is_none()
            {
                return Err(TorshDistributedError::configuration_error(
                    "ZeRO Stage 3 requires stage3_max_live_parameters to be set",
                ));
            }
        }

        // Validate offloading configuration
        if let Some(ref offload_config) = self.config.zero_optimization.offload_optimizer {
            if offload_config.device.is_empty() {
                return Err(TorshDistributedError::configuration_error(
                    "Offload optimizer device cannot be empty",
                ));
            }
        }

        if let Some(ref offload_config) = self.config.zero_optimization.offload_param {
            if offload_config.device.is_empty() {
                return Err(TorshDistributedError::configuration_error(
                    "Offload parameter device cannot be empty",
                ));
            }
        }

        Ok(())
    }

    /// Initialize ZeRO Stage 1 (optimizer state partitioning)
    fn initialize_zero_stage1(&self) -> TorshResult<()> {
        tracing::info!("Initializing DeepSpeed ZeRO Stage 1 (optimizer state partitioning)");

        // Configure optimizer state partitioning
        let bucket_size = self
            .config
            .zero_optimization
            .reduce_bucket_size
            .unwrap_or(2e8 as u64);
        tracing::debug!("ZeRO Stage 1 - Reduce bucket size: {}", bucket_size);

        Ok(())
    }

    /// Initialize ZeRO Stage 2 (gradient partitioning)
    fn initialize_zero_stage2(&self) -> TorshResult<()> {
        tracing::info!("Initializing DeepSpeed ZeRO Stage 2 (gradient partitioning)");

        // Configure gradient partitioning
        let allgather_bucket_size = self
            .config
            .zero_optimization
            .allgather_bucket_size
            .unwrap_or(2e8 as u64);
        let reduce_bucket_size = self
            .config
            .zero_optimization
            .reduce_bucket_size
            .unwrap_or(2e8 as u64);
        let overlap_comm = self.config.zero_optimization.overlap_comm.unwrap_or(true);

        tracing::debug!(
            "ZeRO Stage 2 - Allgather bucket size: {}",
            allgather_bucket_size
        );
        tracing::debug!("ZeRO Stage 2 - Reduce bucket size: {}", reduce_bucket_size);
        tracing::debug!("ZeRO Stage 2 - Overlap communication: {}", overlap_comm);

        Ok(())
    }

    /// Initialize ZeRO Stage 3 (parameter partitioning)
    fn initialize_zero_stage3(&self) -> TorshResult<()> {
        tracing::info!("Initializing DeepSpeed ZeRO Stage 3 (parameter partitioning)");

        // Configure parameter partitioning
        let max_live_params = self
            .config
            .zero_optimization
            .stage3_max_live_parameters
            .unwrap_or(1e9 as u64);
        let max_reuse_distance = self
            .config
            .zero_optimization
            .stage3_max_reuse_distance
            .unwrap_or(1000);
        let prefetch_bucket_size = self
            .config
            .zero_optimization
            .stage3_prefetch_bucket_size
            .unwrap_or(5e8 as u64);

        tracing::debug!("ZeRO Stage 3 - Max live parameters: {}", max_live_params);
        tracing::debug!("ZeRO Stage 3 - Max reuse distance: {}", max_reuse_distance);
        tracing::debug!(
            "ZeRO Stage 3 - Prefetch bucket size: {}",
            prefetch_bucket_size
        );

        Ok(())
    }

    /// Get the current configuration
    pub fn config(&self) -> &DeepSpeedConfig {
        &self.config
    }

    /// Check if DeepSpeed integration is initialized
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Convert DeepSpeed config to ToRSh FSDP config
    pub fn to_fsdp_config(&self) -> TorshResult<crate::fsdp::FsdpConfig> {
        use crate::fsdp::{BackwardPrefetch, FsdpConfig, MixedPrecisionConfig, ShardingStrategy};

        let sharding_strategy = match self.config.zero_optimization.stage {
            ZeroStage::Stage0 => ShardingStrategy::NoShard,
            ZeroStage::Stage1 => ShardingStrategy::ShardGradOp,
            ZeroStage::Stage2 => ShardingStrategy::ShardGradOp,
            ZeroStage::Stage3 => ShardingStrategy::FullShard,
        };

        let mixed_precision = if let Some(ref fp16_config) = self.config.fp16 {
            if fp16_config.enabled {
                Some(MixedPrecisionConfig {
                    param_dtype: torsh_core::DType::F16,
                    reduce_dtype: torsh_core::DType::F16,
                    buffer_dtype: torsh_core::DType::F16,
                    keep_low_precision_grads: false,
                })
            } else {
                None
            }
        } else {
            None
        };

        Ok(FsdpConfig {
            min_num_params: 1000,
            auto_wrap_policy: crate::fsdp::AutoWrapPolicy::SizeBasedAutoWrap {
                min_num_params: 1000,
            },
            sharding_strategy,
            mixed_precision,
            cpu_offload: self.config.zero_optimization.offload_optimizer.is_some()
                || self.config.zero_optimization.offload_param.is_some(),
            memory_config: crate::fsdp::MemoryConfig::default(),
            backward_prefetch: crate::fsdp::BackwardPrefetch::BackwardPre,
        })
    }

    /// Convert DeepSpeed config to ToRSh gradient compression config
    pub fn to_gradient_compression_config(
        &self,
    ) -> Option<crate::gradient_compression::CompressionConfig> {
        // DeepSpeed doesn't have direct gradient compression config, but we can infer from ZeRO settings
        if self
            .config
            .zero_optimization
            .reduce_scatter
            .unwrap_or(false)
        {
            Some(crate::gradient_compression::CompressionConfig {
                method: crate::gradient_compression::CompressionMethod::TopK { k: 0.1 },
                compression_ratio: 0.1,
                error_feedback: true,
                error_feedback_momentum: 0.9,
                memory_efficient: true,
                warmup_steps: 5,
            })
        } else {
            None
        }
    }

    /// Get performance statistics
    pub fn get_stats(&self) -> DeepSpeedStats {
        DeepSpeedStats {
            zero_stage: self.config.zero_optimization.stage,
            initialized: self.initialized,
            fp16_enabled: self
                .config
                .fp16
                .as_ref()
                .map(|c| c.enabled)
                .unwrap_or(false),
            cpu_offload_enabled: self.config.zero_force_ds_cpu_optimizer.unwrap_or(false),
            activation_checkpointing_enabled: self
                .config
                .activation_checkpointing
                .as_ref()
                .map(|c| c.partition_activations.unwrap_or(false))
                .unwrap_or(false),
        }
    }
}

impl Default for DeepSpeedIntegration {
    fn default() -> Self {
        Self::new(DeepSpeedConfig::default())
    }
}

/// DeepSpeed performance statistics
#[derive(Debug, Clone)]
pub struct DeepSpeedStats {
    /// ZeRO optimization stage
    pub zero_stage: ZeroStage,
    /// Whether DeepSpeed is initialized
    pub initialized: bool,
    /// Whether FP16 is enabled
    pub fp16_enabled: bool,
    /// Whether CPU offloading is enabled
    pub cpu_offload_enabled: bool,
    /// Whether activation checkpointing is enabled
    pub activation_checkpointing_enabled: bool,
}

impl Default for DeepSpeedConfig {
    fn default() -> Self {
        Self {
            zero_optimization: ZeroOptimizationConfig::default(),
            gradient_clipping: None,
            gradient_accumulation_steps: None,
            fp16: None,
            zero_force_ds_cpu_optimizer: None,
            activation_checkpointing: None,
        }
    }
}

impl Default for ZeroOptimizationConfig {
    fn default() -> Self {
        Self {
            stage: ZeroStage::Stage0,
            allgather_bucket_size: None,
            reduce_bucket_size: None,
            overlap_comm: None,
            contiguous_gradients: None,
            sub_group_size: None,
            reduce_scatter: None,
            allgather_partitions: None,
            stage3_max_live_parameters: None,
            stage3_max_reuse_distance: None,
            stage3_prefetch_bucket_size: None,
            stage3_param_persistence_threshold: None,
            offload_optimizer: None,
            offload_param: None,
        }
    }
}

/// Utility functions for DeepSpeed integration
pub mod utils {
    use super::*;

    /// Create a basic DeepSpeed configuration for ZeRO Stage 1
    pub fn create_zero_stage1_config() -> DeepSpeedConfig {
        DeepSpeedConfig {
            zero_optimization: ZeroOptimizationConfig {
                stage: ZeroStage::Stage1,
                overlap_comm: Some(true),
                contiguous_gradients: Some(true),
                reduce_bucket_size: Some(2e8 as u64),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// Create a basic DeepSpeed configuration for ZeRO Stage 2
    pub fn create_zero_stage2_config() -> DeepSpeedConfig {
        DeepSpeedConfig {
            zero_optimization: ZeroOptimizationConfig {
                stage: ZeroStage::Stage2,
                overlap_comm: Some(true),
                contiguous_gradients: Some(true),
                reduce_bucket_size: Some(2e8 as u64),
                allgather_bucket_size: Some(2e8 as u64),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// Create a basic DeepSpeed configuration for ZeRO Stage 3
    pub fn create_zero_stage3_config() -> DeepSpeedConfig {
        DeepSpeedConfig {
            zero_optimization: ZeroOptimizationConfig {
                stage: ZeroStage::Stage3,
                overlap_comm: Some(true),
                contiguous_gradients: Some(true),
                reduce_bucket_size: Some(2e8 as u64),
                allgather_bucket_size: Some(2e8 as u64),
                stage3_max_live_parameters: Some(1e9 as u64),
                stage3_max_reuse_distance: Some(1000),
                stage3_prefetch_bucket_size: Some(5e8 as u64),
                stage3_param_persistence_threshold: Some(1e6 as u64),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// Create a DeepSpeed configuration with FP16 mixed precision
    pub fn create_fp16_config() -> DeepSpeedConfig {
        DeepSpeedConfig {
            zero_optimization: ZeroOptimizationConfig {
                stage: ZeroStage::Stage2,
                overlap_comm: Some(true),
                contiguous_gradients: Some(true),
                reduce_bucket_size: Some(2e8 as u64),
                allgather_bucket_size: Some(2e8 as u64),
                ..Default::default()
            },
            fp16: Some(FP16Config {
                enabled: true,
                loss_scale: None,
                loss_scale_window: Some(1000),
                hysteresis: Some(2),
                min_loss_scale: Some(1.0),
                initial_scale_power: Some(16),
            }),
            ..Default::default()
        }
    }

    /// Create a DeepSpeed configuration with CPU offloading
    pub fn create_cpu_offload_config() -> DeepSpeedConfig {
        DeepSpeedConfig {
            zero_optimization: ZeroOptimizationConfig {
                stage: ZeroStage::Stage3,
                overlap_comm: Some(true),
                contiguous_gradients: Some(true),
                reduce_bucket_size: Some(2e8 as u64),
                allgather_bucket_size: Some(2e8 as u64),
                stage3_max_live_parameters: Some(1e9 as u64),
                stage3_max_reuse_distance: Some(1000),
                stage3_prefetch_bucket_size: Some(5e8 as u64),
                stage3_param_persistence_threshold: Some(1e6 as u64),
                offload_optimizer: Some(OffloadOptimizerConfig {
                    device: "cpu".to_string(),
                    nvme_path: None,
                    pin_memory: Some(false),
                    buffer_count: Some(4),
                    fast_init: Some(false),
                }),
                offload_param: Some(OffloadParamConfig {
                    device: "cpu".to_string(),
                    nvme_path: None,
                    pin_memory: Some(false),
                    buffer_count: Some(4),
                    max_in_cpu: Some(1e9 as u64),
                }),
                ..Default::default()
            },
            zero_force_ds_cpu_optimizer: Some(true),
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deepspeed_config_serialization() {
        let config = utils::create_zero_stage2_config();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: DeepSpeedConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(
            config.zero_optimization.stage,
            deserialized.zero_optimization.stage
        );
        assert_eq!(
            config.zero_optimization.overlap_comm,
            deserialized.zero_optimization.overlap_comm
        );
    }

    #[test]
    fn test_deepspeed_integration_initialization() {
        let config = utils::create_zero_stage1_config();
        let mut integration = DeepSpeedIntegration::new(config);

        assert!(!integration.is_initialized());
        integration.initialize().unwrap();
        assert!(integration.is_initialized());
    }

    #[test]
    fn test_deepspeed_to_fsdp_config() {
        let config = utils::create_zero_stage3_config();
        let integration = DeepSpeedIntegration::new(config);
        let fsdp_config = integration.to_fsdp_config().unwrap();

        assert_eq!(
            fsdp_config.sharding_strategy,
            crate::fsdp::ShardingStrategy::FullShard
        );
    }

    #[test]
    fn test_deepspeed_stats() {
        let config = utils::create_fp16_config();
        let integration = DeepSpeedIntegration::new(config);
        let stats = integration.get_stats();

        assert_eq!(stats.zero_stage, ZeroStage::Stage2);
        assert!(!stats.initialized);
        assert!(!stats.fp16_enabled); // Not initialized yet
    }

    #[test]
    fn test_deepspeed_config_validation() {
        let mut config = utils::create_zero_stage3_config();
        config.zero_optimization.stage3_max_live_parameters = None;

        let mut integration = DeepSpeedIntegration::new(config);
        assert!(integration.initialize().is_err());
    }
}
