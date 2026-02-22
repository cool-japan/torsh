//! Distributed optimization for multi-process training
//!
//! This module provides comprehensive support for distributed deep learning training
//! across multiple processes, nodes, or devices. It includes various distributed
//! optimization algorithms and utilities for different distributed training scenarios.
//!
//! # Overview
//!
//! Distributed training allows you to scale deep learning to larger datasets and models
//! by parallelizing the training process across multiple workers. This module supports
//! several distributed training paradigms:
//!
//! ## Synchronous Training
//! - **Data Parallel**: Each worker processes different batches, gradients are synchronized
//! - **All-Reduce**: Efficient gradient averaging across all workers
//! - **Parameter Server**: Central parameter management with worker nodes
//!
//! ## Asynchronous Training
//! - **Async SGD**: Workers update independently without waiting for synchronization
//! - **Bounded Staleness**: Limits how stale parameter updates can be
//! - **Elastic Averaging**: Workers explore different parameter spaces while being pulled toward center
//!
//! # Key Components
//!
//! ## Core Module (`core`)
//! - `DistributedOptimizer<O>`: Wrapper that adds distributed functionality to any optimizer
//! - `DistributedConfig`: Configuration for communication backends and strategies
//! - Communication abstractions for different backends (NCCL, MPI, Gloo)
//!
//! ## Async SGD Module (`async_sgd`)
//! - `AsyncSGD`: Asynchronous stochastic gradient descent
//! - Staleness tracking and adaptive learning rates
//! - Parameter mixing for improved convergence
//!
//! ## Elastic SGD Module (`elastic_sgd`)
//! - `ElasticAveragingSGD`: Elastic averaging SGD (EASGD)
//! - Allows worker exploration while maintaining coordination
//! - Better convergence properties than vanilla distributed SGD
//!
//! ## Utilities Module (`utils`)
//! - Convenient factory functions for common setups
//! - Predefined configurations for different scenarios
//! - Monitoring and debugging utilities
//!
//! # Quick Start
//!
//! ## Basic Distributed Training
//! ```rust
//! # use torsh_tensor::creation::randn;
//! # use torsh_core::error::Result;
//! # use parking_lot::RwLock;
//! # use std::sync::Arc;
//! # fn main() -> Result<()> {
//! use torsh_optim::distributed::{utils, core::*};
//! use torsh_optim::{SGD, Optimizer};
//!
//! // Create some parameters
//! let param1 = Arc::new(RwLock::new(randn::<f32>(&[10, 20])?));
//! let params = vec![param1];
//!
//! // Create distributed SGD optimizer
//! let mut optimizer = utils::distributed_sgd(
//!     params,
//!     0.1,        // learning rate
//!     4,          // world size (4 workers)
//!     0,          // rank (this is worker 0)
//!     Some(0.9),  // momentum
//!     Some(1e-4)  // weight decay
//! )?;
//!
//! // Training loop (simplified)
//! for _batch in 0..10 {
//!     // Forward pass and backward pass would go here
//!     // In real code: loss.backward()?;
//!
//!     // Distributed optimization step (includes gradient synchronization)
//!     optimizer.step()?;
//!     optimizer.zero_grad();
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ## Asynchronous Training
//! ```rust,no_run
//! # use torsh_tensor::creation::randn;
//! # use torsh_core::error::Result;
//! # use parking_lot::RwLock;
//! # use std::sync::Arc;
//! # fn main() -> Result<()> {
//! use torsh_optim::distributed::async_sgd::AsyncSGD;
//!
//! // Create some parameters
//! let param1 = Arc::new(RwLock::new(randn::<f32>(&[10, 20])?));
//! let params = vec![param1];
//!
//! let mut async_optimizer = AsyncSGD::new_async(params, 0.01);
//!
//! // Asynchronous training - workers can update at different rates (conceptual example)
//! // In practice, you would coordinate with a parameter server
//! # Ok(())
//! # }
//! ```
//!
//! ## Elastic Averaging SGD
//! ```rust,no_run
//! # use torsh_tensor::creation::randn;
//! # use torsh_core::error::Result;
//! # use parking_lot::RwLock;
//! # use std::sync::Arc;
//! # fn main() -> Result<()> {
//! use torsh_optim::distributed::elastic_sgd::ElasticAveragingSGD;
//!
//! // Create some parameters
//! let param1 = Arc::new(RwLock::new(randn::<f32>(&[10, 20])?));
//! let params = vec![param1];
//!
//! let mut easgd = ElasticAveragingSGD::new_default(
//!     params,
//!     0.1,    // learning rate
//!     0,      // worker rank
//!     4       // total workers
//! )?;
//!
//! // Training with periodic communication (conceptual example)
//! // In practice, you would implement the full training loop with communication
//! # Ok(())
//! # }
//! ```
//!
//! # Advanced Configurations
//!
//! ## Custom Communication Setup
//! ```rust
//! # use torsh_tensor::creation::randn;
//! # use torsh_core::error::Result;
//! # use parking_lot::RwLock;
//! # use std::sync::Arc;
//! # fn main() -> Result<()> {
//! use torsh_optim::distributed::{core::*, utils};
//! use torsh_optim::AdamW;
//!
//! // Create some parameters
//! let param1 = Arc::new(RwLock::new(randn::<f32>(&[10, 20])?));
//! let params = vec![param1];
//! let base_optimizer = AdamW::new(params, Some(1e-4), None, None, Some(0.01), false);
//!
//! let config = DistributedConfig {
//!     backend: DistributedBackend::NCCL,
//!     sync_strategy: SyncStrategy::AllReduce,
//!     world_size: 8,
//!     rank: 0,
//!     gradient_compression: true,
//!     bucket_size_mb: 25.0,
//!     overlap_communication: true,
//!     ..Default::default()
//! };
//!
//! let optimizer = utils::distributed_optimizer(base_optimizer, config)?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Large Scale Training
//! ```rust
//! # use torsh_tensor::creation::randn;
//! # use torsh_core::error::Result;
//! # use parking_lot::RwLock;
//! # use std::sync::Arc;
//! # fn main() -> Result<()> {
//! use torsh_optim::distributed::utils::{self, configs};
//! use torsh_optim::AdamW;
//!
//! // Create some parameters
//! let param1 = Arc::new(RwLock::new(randn::<f32>(&[10, 20])?));
//! let params = vec![param1];
//! let base_optimizer = AdamW::new(params, Some(1e-4), None, None, Some(0.01), false);
//!
//! let worker_rank = 0;
//! // Optimized for 100+ workers
//! let config = configs::large_scale_config(128, worker_rank);
//! let optimizer = utils::distributed_optimizer(base_optimizer, config)?;
//! # Ok(())
//! # }
//! ```
//!
//! # Communication Backends
//!
//! ## NCCL (Recommended for GPUs)
//! - Highly optimized for NVIDIA GPUs
//! - Supports advanced communication patterns
//! - Best performance for GPU clusters
//!
//! ## MPI (General Purpose)
//! - Works on both CPU and GPU
//! - Widely supported across HPC systems
//! - Good for mixed CPU/GPU environments
//!
//! ## Gloo (Facebook's Backend)
//! - Cross-platform support
//! - Good fallback option
//! - Supports both CPU and GPU
//!
//! # Best Practices
//!
//! ## Choosing the Right Approach
//! - **Synchronous**: Better convergence, easier debugging, may be slower
//! - **Asynchronous**: Faster iteration, handles stragglers, may have convergence issues
//! - **Elastic Averaging**: Best of both worlds, good exploration
//!
//! ## Communication Efficiency
//! - Enable gradient compression for large clusters or limited bandwidth
//! - Use appropriate bucket sizes (smaller for CPU, larger for GPU)
//! - Enable communication overlap when possible
//!
//! ## Monitoring and Debugging
//! ```rust,no_run
//! # use torsh_tensor::creation::randn;
//! # use torsh_core::error::Result;
//! # use parking_lot::RwLock;
//! # use std::sync::Arc;
//! # fn main() -> Result<()> {
//! use torsh_optim::distributed::utils;
//!
//! // Create distributed optimizer
//! let param1 = Arc::new(RwLock::new(randn::<f32>(&[10, 20])?));
//! let params = vec![param1];
//! let optimizer = utils::distributed_adam(params, 0.001, 4, 0, None, None, None)?;
//!
//! // In practice, you would collect performance statistics using distributed monitoring tools
//! # Ok(())
//! # }
//! ```
//!
//! # Performance Tips
//!
//! 1. **Batch Size**: Scale batch size with number of workers
//! 2. **Learning Rate**: May need adjustment for distributed training
//! 3. **Communication Frequency**: Balance between convergence and efficiency
//! 4. **Gradient Compression**: Essential for large clusters
//! 5. **Memory Management**: Monitor memory usage across workers

pub mod async_sgd;
pub mod core;
pub mod elastic_sgd;
pub mod utils;

// Re-export the main types for convenience
pub use async_sgd::{AsyncConfig, AsyncSGD};
pub use core::{
    CommunicationStats, DistributedBackend, DistributedConfig, DistributedOptimizer, OptimizerExt,
    SyncStrategy,
};
pub use elastic_sgd::ElasticAveragingSGD;

// Re-export utilities
pub use utils::{distributed_adam, distributed_optimizer, distributed_sgd};
