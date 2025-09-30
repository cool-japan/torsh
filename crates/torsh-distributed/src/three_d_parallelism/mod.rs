//! Modular 3D Parallelism System
//!
//! This module provides a comprehensive 3D parallelism system that orchestrates
//! data parallelism (DP), tensor parallelism (TP), and pipeline parallelism (PP)
//! for efficient training of ultra-large language models.
//!
//! ## Architecture Overview
//!
//! The system is organized into focused, specialized modules:
//!
//! ### Core Components
//! - **Configuration** (`config.rs`): Type definitions, validation, and ranking systems
//! - **Coordinator** (`coordinator.rs`): Main orchestration engine for forward/backward passes
//! - **Process Groups** (`process_group.rs`): Communication management across 3D dimensions
//! - **Memory Management** (`memory_management.rs`): Activation checkpointing and memory optimization
//! - **Gradient Sync** (`gradient_sync.rs`): Gradient synchronization with compression
//! - **Performance** (`performance.rs`): Comprehensive monitoring and bottleneck analysis
//! - **Model Shards** (`model_shards.rs`): Parameter distribution and sharding logic
//!
//! ## Key Features
//!
//! ### 3D Parallelism Orchestration
//! - **Data Parallelism**: Gradient synchronization across data replicas
//! - **Tensor Parallelism**: Parameter sharding within layers
//! - **Pipeline Parallelism**: Layer distribution across stages with micro-batching
//!
//! ### Advanced Memory Management
//! - Gradient checkpointing with configurable strategies
//! - Activation recomputation for memory efficiency
//! - Disk offloading for extreme memory optimization
//! - Memory pool management for efficient allocation
//!
//! ### Communication Optimization
//! - Multiple all-reduce strategies (Ring, Tree, Hierarchical, Adaptive)
//! - Gradient compression and quantization
//! - Communication-computation overlap
//! - Bucketing for efficient small tensor handling
//!
//! ### Performance Monitoring
//! - Real-time throughput and latency tracking
//! - Bottleneck identification and suggestions
//! - Memory efficiency analysis
//! - Communication overhead monitoring
//!
//! ## Usage Example
//!
//! ```rust,no_run
//! use torsh_distributed::three_d_parallelism::*;
//! use torsh_distributed::{init_process_group, BackendType};
//! use std::sync::Arc;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Initialize distributed environment
//! let pg = init_process_group(BackendType::Nccl, 0, 8, "127.0.0.1", 29500)?;
//!
//! // Configure 3D parallelism (2x2x2 = 8 devices)
//! let config = ThreeDParallelismConfig {
//!     dp_size: 2,
//!     tp_size: 2,
//!     pp_size: 2,
//!     num_layers: 24,
//!     micro_batch_size: 4,
//!     enable_gradient_checkpointing: true,
//!     memory_strategy: MemoryOptimizationStrategy::Aggressive,
//!     comm_strategy: CommunicationStrategy::Adaptive,
//!     ..Default::default()
//! };
//!
//! // Create coordinator
//! let mut coordinator = ThreeDParallelismCoordinator::new(config, Arc::new(pg))?;
//!
//! // Run training step
//! let input = torsh_tensor::tensor_1d(&[0.1, 0.2, 0.3, 0.4])?;
//! let output = coordinator.forward_pass(&input, 0).await?;
//! coordinator.backward_pass(&output, 0).await?;
//!
//! // Get performance report
//! info!("{}", coordinator.get_performance_monitor().generate_report());
//! # Ok(())
//! # }
//! ```

// Module declarations
pub mod config;
pub mod coordinator;
pub mod gradient_sync;
pub mod memory_management;
pub mod model_shards;
pub mod performance;
pub mod process_group;

// Re-export main types for convenient access
pub use config::{
    CommunicationStrategy, MemoryOptimizationStrategy, MemoryRequirements, PipelineSchedule,
    ProcessGroupIds, RankMapping, ThreeDParallelismConfig,
};

pub use coordinator::ThreeDParallelismCoordinator;

pub use process_group::{CommunicationStats, ProcessGroupManager};

pub use memory_management::{MemoryManager, MemoryOptimizationResult, MemoryUsageStats};

pub use gradient_sync::{
    GradientBucketingConfig, GradientCompressionConfig, GradientSynchronizer, SyncStatistics,
};

pub use performance::{
    BottleneckSeverity, CommunicationType, Memory3DStats, Performance3DMonitor, Performance3DStats,
    PerformanceAnalysis, PerformanceBottleneck,
};

pub use model_shards::{
    CommunicationPattern, LayerShard, LayerTensorParallelPlan, LayerType, ModelShard, ModelShards,
    ShardInfo, ShardStrategy, TensorParallelShardingPlan,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{init_process_group, BackendType};
    use std::sync::Arc;
    use torsh_tensor::creation::tensor_1d;

    /// Test 3D parallelism configuration validation
    #[test]
    fn test_3d_config_validation() {
        let config = ThreeDParallelismConfig {
            dp_size: 2,
            tp_size: 2,
            pp_size: 2,
            num_layers: 24,
            ..Default::default()
        };

        // Valid configuration
        assert!(config.validate(8).is_ok()); // 2*2*2 = 8

        // Invalid world size
        assert!(config.validate(16).is_err()); // Mismatch

        // Invalid layer distribution
        let invalid_config = ThreeDParallelismConfig {
            dp_size: 2,
            tp_size: 2,
            pp_size: 3, // 24 layers not divisible by 3 stages
            num_layers: 24,
            ..Default::default()
        };
        assert!(invalid_config.validate(12).is_err());
    }

    /// Test rank mapping calculations
    #[test]
    fn test_rank_mapping() {
        let config = ThreeDParallelismConfig {
            dp_size: 2,
            tp_size: 2,
            pp_size: 2,
            ..Default::default()
        };

        // Test various global ranks
        let mapping_0 = RankMapping::new(&config, 0);
        assert_eq!(
            (mapping_0.dp_rank, mapping_0.tp_rank, mapping_0.pp_rank),
            (0, 0, 0)
        );

        let mapping_3 = RankMapping::new(&config, 3);
        assert_eq!(
            (mapping_3.dp_rank, mapping_3.tp_rank, mapping_3.pp_rank),
            (0, 1, 1)
        );

        let mapping_7 = RankMapping::new(&config, 7);
        assert_eq!(
            (mapping_7.dp_rank, mapping_7.tp_rank, mapping_7.pp_rank),
            (1, 1, 1)
        );

        // Test 3D coordinate conversion
        let reconstructed_rank = RankMapping::from_3d_coords(&config, 1, 1, 1);
        assert_eq!(reconstructed_rank, 7);
    }

    /// Test memory requirements calculation
    #[test]
    fn test_memory_requirements() {
        let config = ThreeDParallelismConfig {
            dp_size: 1,
            tp_size: 2,
            pp_size: 4,
            num_layers: 24,
            memory_strategy: MemoryOptimizationStrategy::Standard,
            ..Default::default()
        };

        let requirements = config.memory_requirements();

        // Should have reasonable memory estimates
        assert!(requirements.model_memory_mb > 0.0);
        assert!(requirements.activation_memory_mb > 0.0);
        assert!(requirements.optimizer_memory_mb > 0.0);
        assert!(requirements.total_memory_mb > requirements.model_memory_mb);

        // Standard strategy should use less memory than basic
        let basic_config = ThreeDParallelismConfig {
            memory_strategy: MemoryOptimizationStrategy::Basic,
            ..config
        };
        let basic_requirements = basic_config.memory_requirements();
        assert!(requirements.activation_memory_mb < basic_requirements.activation_memory_mb);
    }

    /// Test model shards creation
    #[test]
    fn test_model_shards_creation() {
        let config = ThreeDParallelismConfig {
            dp_size: 1,
            tp_size: 2,
            pp_size: 4,
            num_layers: 24,
            ..Default::default()
        };

        let model_shards = ModelShards::new(&config).unwrap();

        // Verify pipeline structure
        assert_eq!(model_shards.pipeline_stages.len(), 4); // pp_size
        assert_eq!(model_shards.pipeline_stages[0].len(), 6); // 24/4 layers per stage

        // Verify parameter counts
        assert!(model_shards.total_parameters > 0);
        assert_eq!(model_shards.parameters_per_stage.len(), 4);

        // Verify shards map
        assert!(!model_shards.shards.is_empty());

        // Test layer access
        let layer_0 = model_shards.get_layer_shard(0);
        assert!(layer_0.is_some());

        let layer_shard = layer_0.unwrap();
        assert_eq!(layer_shard.layer_id, 0);
        assert!(layer_shard.parameter_count() > 0);
    }

    /// Test layer shard parameter counting
    #[test]
    fn test_layer_shard_parameters() {
        let layer = LayerShard::new(0, 4).unwrap(); // TP size 4

        assert_eq!(layer.layer_id, 0);
        assert_eq!(layer.weight.shape().dims()[1], 128); // 512/4 = 128
        assert!(layer.parameter_count() > 0);
        assert!(layer.memory_usage_bytes() > 0);

        // Test gradient initialization
        let mut layer_with_grads = layer;
        layer_with_grads.init_gradients().unwrap();
        assert!(layer_with_grads.grad_weight.is_some());
    }

    /// Test different layer types
    #[test]
    fn test_layer_types() {
        for layer_id in 0..8 {
            let layer = LayerShard::new(layer_id, 2).unwrap();

            // Verify layer type assignment
            let expected_type = match layer_id % 4 {
                0 => LayerType::Embedding,
                1 => LayerType::Attention,
                2 => LayerType::MLP,
                _ => LayerType::Linear,
            };
            assert_eq!(layer.layer_type, expected_type);

            // MLP layers should have down projection
            if matches!(layer.layer_type, LayerType::MLP) {
                assert!(layer.down_projection_weight.is_some());
            }
        }
    }

    /// Test memory optimization strategies
    #[test]
    fn test_memory_optimization_strategies() {
        let config = ThreeDParallelismConfig {
            dp_size: 1,
            tp_size: 1,
            pp_size: 1,
            num_layers: 4,
            ..Default::default()
        };
        let rank_mapping = RankMapping::new(&config, 0);

        // Test different memory strategies
        let strategies = [
            MemoryOptimizationStrategy::Basic,
            MemoryOptimizationStrategy::Standard,
            MemoryOptimizationStrategy::Aggressive,
            MemoryOptimizationStrategy::Extreme,
        ];

        for strategy in strategies {
            let mut test_config = config.clone();
            test_config.memory_strategy = strategy;
            let memory_manager = MemoryManager::new(&test_config, &rank_mapping);
            assert!(memory_manager.is_ok());
        }
    }

    /// Test communication strategies
    #[test]
    fn test_communication_strategies() {
        let strategies = [
            CommunicationStrategy::AllReduce,
            CommunicationStrategy::HierarchicalAllReduce,
            CommunicationStrategy::RingAllReduce,
            CommunicationStrategy::TreeAllReduce,
            CommunicationStrategy::Adaptive,
        ];

        for strategy in strategies {
            let config = ThreeDParallelismConfig {
                dp_size: 2,
                tp_size: 2,
                pp_size: 2,
                comm_strategy: strategy,
                ..Default::default()
            };

            // Should validate successfully
            assert!(config.validate(8).is_ok());
        }
    }

    /// Test performance monitoring
    #[test]
    fn test_performance_monitoring() {
        let config = ThreeDParallelismConfig::default();
        let rank_mapping = RankMapping::new(&config, 0);
        let monitor = Performance3DMonitor::new(&rank_mapping);

        // Initial stats should be empty
        let stats = monitor.get_stats();
        assert_eq!(stats.forward_passes, 0);
        assert_eq!(stats.backward_passes, 0);
        assert_eq!(stats.tokens_per_second, 0.0);

        // Test performance analysis
        let analysis = monitor.get_performance_analysis();
        assert_eq!(analysis.overall_throughput, 0.0);
        assert_eq!(analysis.bottlenecks.len(), 0); // No bottlenecks with no data

        // Test report generation
        let report = monitor.generate_report();
        assert!(report.contains("Performance Report"));
        assert!(report.contains("Overall Performance"));
    }

    /// Test gradient synchronization configuration
    #[test]
    fn test_gradient_synchronization() {
        let config = ThreeDParallelismConfig {
            dp_size: 4,
            tp_size: 2,
            pp_size: 1,
            ..Default::default()
        };
        let rank_mapping = RankMapping::new(&config, 0);

        let gradient_sync = GradientSynchronizer::new(&config, &rank_mapping).unwrap();
        let stats = gradient_sync.get_sync_stats();
        assert_eq!(stats.total_sync_operations, 0);

        // Test compression configuration
        let compression_config = GradientCompressionConfig {
            enable_compression: true,
            compression_ratio: 0.1,
            error_feedback: true,
            quantization_bits: 8,
        };

        // Configuration should be reasonable
        assert!(compression_config.compression_ratio > 0.0);
        assert!(compression_config.compression_ratio < 1.0);
        assert!(compression_config.quantization_bits > 0);
    }

    /// Test pipeline scheduling strategies
    #[test]
    fn test_pipeline_scheduling() {
        let schedules = [
            PipelineSchedule::RoundRobin,
            PipelineSchedule::Interleaved,
            PipelineSchedule::GPipe,
            PipelineSchedule::OneForwardOneBackward,
        ];

        for schedule in schedules {
            let config = ThreeDParallelismConfig {
                dp_size: 1,
                tp_size: 1,
                pp_size: 4,
                num_layers: 16,
                pipeline_schedule: schedule,
                ..Default::default()
            };

            assert!(config.validate(4).is_ok());
            assert_eq!(config.layers_per_stage(), 4);
        }
    }

    /// Test tensor parallel sharding strategies
    #[test]
    fn test_tensor_parallel_sharding() {
        let config = ThreeDParallelismConfig {
            dp_size: 1,
            tp_size: 4,
            pp_size: 1,
            num_layers: 8,
            ..Default::default()
        };

        let model_shards = ModelShards::new(&config).unwrap();
        let sharding_plan = model_shards.create_tp_sharding_plan(config.tp_size);

        // Should have sharding plans for each layer
        let layer_plan = sharding_plan.get_layer_plan(0, 0);
        assert!(layer_plan.is_some());

        let plan = layer_plan.unwrap();
        assert!(!plan.shard_strategies.is_empty());
        assert!(!plan.weight_shape.is_empty());
    }

    /// Test shard strategies for different layer types
    #[test]
    fn test_shard_strategies() {
        let strategies = [
            ShardStrategy::ColumnParallel,
            ShardStrategy::RowParallel,
            ShardStrategy::VocabParallel,
            ShardStrategy::Replicated,
        ];

        // All strategies should be distinct
        for (i, &strategy1) in strategies.iter().enumerate() {
            for &strategy2 in &strategies[i + 1..] {
                assert_ne!(strategy1, strategy2);
            }
        }
    }

    /// Test communication patterns
    #[test]
    fn test_communication_patterns() {
        let patterns = [
            CommunicationPattern::AllReduce,
            CommunicationPattern::AllGatherThenReduceScatter,
            CommunicationPattern::ReduceScatterThenAllGather,
            CommunicationPattern::None,
        ];

        // All patterns should be distinct
        for (i, &pattern1) in patterns.iter().enumerate() {
            for &pattern2 in &patterns[i + 1..] {
                assert_ne!(pattern1, pattern2);
            }
        }
    }

    /// Test bottleneck severity levels
    #[test]
    fn test_bottleneck_severity() {
        let severities = [
            BottleneckSeverity::Low,
            BottleneckSeverity::Medium,
            BottleneckSeverity::High,
            BottleneckSeverity::Critical,
        ];

        // Test string conversion
        for severity in severities {
            let severity_str = severity.as_str();
            assert!(!severity_str.is_empty());
        }

        // Test ordering
        assert_ne!(BottleneckSeverity::Low, BottleneckSeverity::Critical);
    }

    /// Test memory statistics tracking
    #[test]
    fn test_memory_statistics() {
        let mut stats = Memory3DStats::new();

        // Initial values
        assert_eq!(stats.model_memory, 0);
        assert_eq!(stats.total_memory, 0);
        assert_eq!(stats.memory_efficiency, 0.0);

        // Update values
        stats.model_memory = 1000;
        stats.activation_memory = 500;
        stats.total_memory = stats.model_memory + stats.activation_memory;

        assert_eq!(stats.total_memory, 1500);
    }

    /// Test configuration defaults
    #[test]
    fn test_configuration_defaults() {
        let config = ThreeDParallelismConfig::default();

        // Verify reasonable defaults
        assert_eq!(config.dp_size, 1);
        assert_eq!(config.tp_size, 1);
        assert_eq!(config.pp_size, 1);
        assert_eq!(config.num_layers, 24);
        assert_eq!(config.micro_batch_size, 1);
        assert!(!config.enable_gradient_checkpointing);
        assert!(!config.enable_mixed_precision);
        assert!(config.max_memory_per_device > 0.0);
        assert!(config.communication_timeout_ms > 0);
    }

    /// Test rank group calculations
    #[test]
    fn test_rank_groups() {
        let config = ThreeDParallelismConfig {
            dp_size: 2,
            tp_size: 2,
            pp_size: 2,
            ..Default::default()
        };
        let rank_mapping = RankMapping::new(&config, 5); // Rank 5 in 8-device setup

        // Verify DP group
        let dp_group = rank_mapping.dp_group_ranks();
        assert_eq!(dp_group.len(), 2); // DP size
        assert!(dp_group.contains(&5)); // Should contain self

        // Verify TP group
        let tp_group = rank_mapping.tp_group_ranks();
        assert_eq!(tp_group.len(), 2); // TP size
        assert!(tp_group.contains(&5)); // Should contain self

        // Verify PP group
        let pp_group = rank_mapping.pp_group_ranks();
        assert_eq!(pp_group.len(), 2); // PP size
        assert!(pp_group.contains(&5)); // Should contain self

        // Verify next/prev PP ranks
        let next_rank = rank_mapping.next_pp_rank();
        let prev_rank = rank_mapping.prev_pp_rank();

        // Rank 5 has coordinates (1, 0, 1), so PP rank is 1 (middle of pipeline)
        assert!(next_rank.is_none()); // Last in pipeline
        assert!(prev_rank.is_some());
    }

    /// Integration test: Create and validate full 3D parallelism system
    #[tokio::test]
    async fn test_3d_parallelism_integration() {
        // Mock process group for testing
        let pg = init_process_group(BackendType::Nccl, 0, 8, "127.0.0.1", 29500)
            .await
            .unwrap();

        let config = ThreeDParallelismConfig {
            dp_size: 2,
            tp_size: 2,
            pp_size: 2,
            num_layers: 8,
            micro_batch_size: 2,
            enable_gradient_checkpointing: true,
            memory_strategy: MemoryOptimizationStrategy::Standard,
            comm_strategy: CommunicationStrategy::Adaptive,
            ..Default::default()
        };

        // Should create coordinator successfully
        let coordinator = ThreeDParallelismCoordinator::new(config, Arc::new(pg));
        assert!(coordinator.is_ok());

        let coordinator = coordinator.unwrap();

        // Verify configuration
        let retrieved_config = coordinator.get_config();
        assert_eq!(retrieved_config.dp_size, 2);
        assert_eq!(retrieved_config.tp_size, 2);
        assert_eq!(retrieved_config.pp_size, 2);

        // Verify rank mapping
        let rank_mapping = coordinator.get_rank_mapping();
        assert_eq!(rank_mapping.world_size, 8);

        // Verify model shards
        let model_shards = coordinator.get_model_shards();
        assert_eq!(model_shards.pipeline_stages.len(), 2); // PP size
        assert!(model_shards.total_parameters > 0);
    }

    /// Performance test: Verify reasonable performance characteristics
    #[test]
    fn test_performance_characteristics() {
        let config = ThreeDParallelismConfig {
            dp_size: 4,
            tp_size: 4,
            pp_size: 2,
            num_layers: 48,
            ..Default::default()
        };

        // Memory requirements should scale reasonably
        let requirements = config.memory_requirements();
        assert!(requirements.total_memory_mb < 50000.0); // Should be under 50GB
        assert!(requirements.model_memory_mb > 0.0);

        // Model shards should distribute parameters efficiently
        let model_shards = ModelShards::new(&config).unwrap();
        let memory_usage = model_shards.memory_usage_bytes();

        // Should have reasonable memory usage
        assert!(memory_usage > 0);
        assert!(memory_usage < 1_000_000_000); // Under 1GB for test model

        // Parameters should be distributed across stages
        assert!(model_shards
            .parameters_per_stage
            .iter()
            .all(|&count| count > 0));
        let total_from_stages: usize = model_shards.parameters_per_stage.iter().sum();
        assert_eq!(total_from_stages, model_shards.total_parameters);
    }
}
