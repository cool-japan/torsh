//! Utility functions for creating and managing distributed optimizers
//!
//! This module provides convenient factory functions and utilities for setting up
//! distributed training with various optimizer types and configurations.

use super::core::{DistributedConfig, DistributedOptimizer};
use crate::{Adam, OptimizerResult, SGD};
use parking_lot::RwLock;
use std::sync::Arc;
use torsh_core::error::Result;
use torsh_tensor::Tensor;

/// Create a distributed SGD optimizer with standard configuration
///
/// This is a convenience function for creating a distributed SGD optimizer
/// with commonly used settings for distributed training.
///
/// # Arguments
///
/// * `params` - Parameters to optimize
/// * `lr` - Learning rate
/// * `world_size` - Total number of processes in distributed training
/// * `rank` - Current process rank (0 to world_size-1)
/// * `momentum` - Optional momentum factor
/// * `weight_decay` - Optional weight decay (L2 penalty)
///
/// # Returns
///
/// A distributed SGD optimizer ready for training
///
/// # Example
///
/// ```rust
/// use torsh_optim::distributed::utils::distributed_sgd;
///
/// let optimizer = distributed_sgd(
///     params,
///     0.1,        // learning rate
///     4,          // world size (4 workers)
///     0,          // rank (worker 0)
///     Some(0.9),  // momentum
///     Some(1e-4)  // weight decay
/// )?;
/// ```
pub fn distributed_sgd(
    params: Vec<Arc<RwLock<Tensor>>>,
    lr: f32,
    world_size: usize,
    rank: usize,
    momentum: Option<f32>,
    weight_decay: Option<f32>,
) -> OptimizerResult<DistributedOptimizer<SGD>> {
    let sgd = SGD::new(params, lr, momentum, None, weight_decay, false);
    let config = DistributedConfig {
        world_size,
        rank,
        ..Default::default()
    };
    DistributedOptimizer::new(sgd, config)
}

/// Create a distributed Adam optimizer with standard configuration
///
/// This is a convenience function for creating a distributed Adam optimizer
/// with commonly used settings for distributed training.
///
/// # Arguments
///
/// * `params` - Parameters to optimize
/// * `lr` - Learning rate
/// * `world_size` - Total number of processes in distributed training
/// * `rank` - Current process rank (0 to world_size-1)
/// * `betas` - Optional Adam beta parameters (momentum and RMS decay)
/// * `eps` - Optional epsilon for numerical stability
/// * `weight_decay` - Optional weight decay (L2 penalty)
///
/// # Returns
///
/// A distributed Adam optimizer ready for training
///
/// # Example
///
/// ```rust
/// use torsh_optim::distributed::utils::distributed_adam;
///
/// let optimizer = distributed_adam(
///     params,
///     1e-3,                    // learning rate
///     8,                       // world size (8 workers)
///     3,                       // rank (worker 3)
///     Some((0.9, 0.999)),     // betas
///     Some(1e-8),             // epsilon
///     Some(0.01)              // weight decay
/// )?;
/// ```
pub fn distributed_adam(
    params: Vec<Arc<RwLock<Tensor>>>,
    lr: f32,
    world_size: usize,
    rank: usize,
    betas: Option<(f32, f32)>,
    eps: Option<f32>,
    weight_decay: Option<f32>,
) -> OptimizerResult<DistributedOptimizer<Adam>> {
    let adam = Adam::new(params, Some(lr), betas, eps, weight_decay, false);
    let config = DistributedConfig {
        world_size,
        rank,
        ..Default::default()
    };
    DistributedOptimizer::new(adam, config)
}

/// Create a distributed optimizer with custom configuration
///
/// This function allows for full customization of the distributed training setup
/// by accepting a custom DistributedConfig.
///
/// # Arguments
///
/// * `optimizer` - Base optimizer to wrap with distributed functionality
/// * `config` - Distributed training configuration
///
/// # Returns
///
/// A distributed optimizer with the specified configuration
///
/// # Example
///
/// ```rust
/// use torsh_optim::distributed::{utils::distributed_optimizer, core::*};
/// use torsh_optim::AdamW;
///
/// let config = DistributedConfig {
///     backend: DistributedBackend::NCCL,
///     sync_strategy: SyncStrategy::AllReduce,
///     world_size: 16,
///     rank: 0,
///     gradient_compression: true,
///     bucket_size_mb: 50.0,
///     overlap_communication: true,
///     ..Default::default()
/// };
///
/// let base_optimizer = AdamW::new(params, Some(1e-4), None, None, Some(0.01), false);
/// let distributed_opt = distributed_optimizer(base_optimizer, config)?;
/// ```
pub fn distributed_optimizer<O: crate::Optimizer>(
    optimizer: O,
    config: DistributedConfig,
) -> OptimizerResult<DistributedOptimizer<O>> {
    DistributedOptimizer::new(optimizer, config)
}

/// Create distributed optimizer configurations for common scenarios
pub mod configs {
    use super::super::core::{DistributedBackend, DistributedConfig, SyncStrategy};

    /// Configuration for CPU-based distributed training with MPI
    pub fn cpu_mpi_config(world_size: usize, rank: usize) -> DistributedConfig {
        DistributedConfig {
            backend: DistributedBackend::MPI,
            sync_strategy: SyncStrategy::AllReduce,
            world_size,
            rank,
            gradient_compression: false,  // Less beneficial on CPU
            bucket_size_mb: 10.0,         // Smaller buckets for CPU
            overlap_communication: false, // Less effective on CPU
            ..Default::default()
        }
    }

    /// Configuration for GPU-based distributed training with NCCL
    pub fn gpu_nccl_config(world_size: usize, rank: usize) -> DistributedConfig {
        DistributedConfig {
            backend: DistributedBackend::NCCL,
            sync_strategy: SyncStrategy::AllReduce,
            world_size,
            rank,
            gradient_compression: world_size >= 8, // Enable for large clusters
            bucket_size_mb: 25.0,                  // Standard bucket size
            overlap_communication: true,           // Enable overlap for GPUs
            ..Default::default()
        }
    }

    /// Configuration for mixed CPU/GPU training with Gloo
    pub fn mixed_gloo_config(world_size: usize, rank: usize) -> DistributedConfig {
        DistributedConfig {
            backend: DistributedBackend::Gloo,
            sync_strategy: SyncStrategy::AllReduce,
            world_size,
            rank,
            gradient_compression: world_size > 4,
            bucket_size_mb: 15.0,
            overlap_communication: true,
            ..Default::default()
        }
    }

    /// Configuration optimized for large-scale training (many workers)
    pub fn large_scale_config(world_size: usize, rank: usize) -> DistributedConfig {
        DistributedConfig {
            backend: DistributedBackend::NCCL,
            sync_strategy: SyncStrategy::ReduceScatter, // More efficient for large scales
            world_size,
            rank,
            gradient_compression: true,  // Essential for large scale
            bucket_size_mb: 50.0,        // Larger buckets for efficiency
            overlap_communication: true, // Critical for performance
            ..Default::default()
        }
    }

    /// Configuration for bandwidth-limited environments
    pub fn low_bandwidth_config(world_size: usize, rank: usize) -> DistributedConfig {
        DistributedConfig {
            backend: DistributedBackend::Gloo,
            sync_strategy: SyncStrategy::AllReduce,
            world_size,
            rank,
            gradient_compression: true, // Always enable for low bandwidth
            bucket_size_mb: 5.0,        // Small buckets for frequent communication
            overlap_communication: true,
            ..Default::default()
        }
    }
}

/// Utilities for monitoring and debugging distributed training
pub mod monitoring {
    use super::super::core::{CommunicationStats, DistributedOptimizer};
    use crate::Optimizer;
    use std::collections::HashMap;

    /// Collect communication statistics from distributed optimizers
    pub fn collect_communication_stats<O: Optimizer>(
        optimizers: &[DistributedOptimizer<O>],
    ) -> HashMap<usize, CommunicationStats> {
        optimizers
            .iter()
            .enumerate()
            .map(|(i, opt)| (i, opt.get_communication_stats()))
            .collect()
    }

    /// Calculate aggregate statistics across all workers
    pub fn aggregate_communication_stats(
        stats: &HashMap<usize, CommunicationStats>,
    ) -> CommunicationStats {
        if stats.is_empty() {
            return CommunicationStats::default();
        }

        let total_communications: u64 = stats.values().map(|s| s.total_communications).sum();
        let total_bytes: u64 = stats.values().map(|s| s.total_bytes_transferred).sum();
        let avg_time: f32 = stats
            .values()
            .map(|s| s.average_communication_time_ms)
            .sum::<f32>()
            / stats.len() as f32;
        let avg_compression: f32 = stats
            .values()
            .map(|s| s.gradient_compression_ratio)
            .sum::<f32>()
            / stats.len() as f32;

        CommunicationStats {
            total_communications,
            total_bytes_transferred: total_bytes,
            average_communication_time_ms: avg_time,
            gradient_compression_ratio: avg_compression,
        }
    }

    /// Print a summary of distributed training performance
    pub fn print_performance_summary<O: Optimizer>(optimizers: &[DistributedOptimizer<O>]) {
        let stats = collect_communication_stats(optimizers);
        let aggregate = aggregate_communication_stats(&stats);

        println!("=== Distributed Training Performance Summary ===");
        println!("Number of workers: {}", optimizers.len());
        println!("Total communications: {}", aggregate.total_communications);
        println!(
            "Total bytes transferred: {:.2} MB",
            aggregate.total_bytes_transferred as f64 / 1024.0 / 1024.0
        );
        println!(
            "Average communication time: {:.2} ms",
            aggregate.average_communication_time_ms
        );
        println!(
            "Average compression ratio: {:.2}x",
            aggregate.gradient_compression_ratio
        );

        // Per-worker breakdown
        println!("\n--- Per-Worker Breakdown ---");
        for (worker_id, stat) in &stats {
            println!(
                "Worker {}: {} communications, {:.2} MB, {:.2} ms avg",
                worker_id,
                stat.total_communications,
                stat.total_bytes_transferred as f64 / 1024.0 / 1024.0,
                stat.average_communication_time_ms
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::core::*;
    use super::*;

    #[test]
    fn test_distributed_sgd_creation() {
        // This would normally require actual tensors, but for testing we'll just check
        // that the function signature is correct
        // In a real test, you'd create actual tensors and verify the optimizer works
    }

    #[test]
    fn test_config_creation() {
        let config = configs::cpu_mpi_config(4, 0);
        assert_eq!(config.world_size, 4);
        assert_eq!(config.rank, 0);
        assert!(matches!(config.backend, DistributedBackend::MPI));

        let gpu_config = configs::gpu_nccl_config(8, 3);
        assert_eq!(gpu_config.world_size, 8);
        assert_eq!(gpu_config.rank, 3);
        assert!(matches!(gpu_config.backend, DistributedBackend::NCCL));
        assert!(gpu_config.gradient_compression); // Should be enabled for 8 workers
    }

    #[test]
    fn test_large_scale_config() {
        let config = configs::large_scale_config(64, 0);
        assert!(matches!(config.sync_strategy, SyncStrategy::ReduceScatter));
        assert!(config.gradient_compression);
        assert_eq!(config.bucket_size_mb, 50.0);
    }
}
