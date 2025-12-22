//! Pipeline parallelism implementation for distributed training
//!
//! Pipeline parallelism splits a model into sequential stages placed on different devices/ranks.
//! This enables training of large models that don't fit on a single device by processing
//! multiple micro-batches in a pipelined fashion.

use crate::collectives::{recv, send};
use crate::{process_group::ProcessGroup, TorshDistributedError, TorshResult};
use log::info;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use torsh_core::{error::Result, DeviceType};
use torsh_nn::{Module, Parameter};
use torsh_tensor::Tensor;

/// Pipeline scheduling strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScheduleType {
    /// GPipe: Simple sequential forward and backward passes
    GPipe,
    /// 1F1B: One forward, one backward (most efficient)
    OneFOneBInterleaved,
    /// Interleaved 1F1B: Better load balancing for larger pipelines
    InterleavedOneFOneB,
}

/// Configuration for pipeline parallelism
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Number of micro-batches to split each mini-batch into
    pub num_micro_batches: usize,
    /// Pipeline scheduling strategy
    pub schedule: ScheduleType,
    /// Whether to enable gradient accumulation across micro-batches
    pub accumulate_gradients: bool,
    /// Communication tags for pipeline stages
    pub base_tag: u32,
    /// Whether to use async communication
    pub async_comm: bool,
    /// Timeout for communication operations (in milliseconds)
    pub comm_timeout_ms: u64,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            num_micro_batches: 4,
            schedule: ScheduleType::OneFOneBInterleaved,
            accumulate_gradients: true,
            base_tag: 1000,
            async_comm: true,
            comm_timeout_ms: 30000,
        }
    }
}

/// Represents a single stage in the pipeline
pub struct PipelineStage {
    /// The neural network module for this stage
    module: Box<dyn Module>,
    /// Stage ID (0-indexed from first to last stage)
    stage_id: usize,
    /// Total number of stages in the pipeline
    num_stages: usize,
    /// Rank of the process that handles this stage
    rank: u32,
    /// Whether this stage is the first in the pipeline
    is_first: bool,
    /// Whether this stage is the last in the pipeline  
    is_last: bool,
    /// Device type for this stage
    #[allow(dead_code)]
    device: DeviceType,
}

impl PipelineStage {
    /// Create a new pipeline stage
    pub fn new(
        module: Box<dyn Module>,
        stage_id: usize,
        num_stages: usize,
        rank: u32,
        device: DeviceType,
    ) -> Self {
        let is_first = stage_id == 0;
        let is_last = stage_id == num_stages - 1;

        Self {
            module,
            stage_id,
            num_stages,
            rank,
            is_first,
            is_last,
            device,
        }
    }

    /// Get the rank of the previous stage
    pub fn prev_rank(&self) -> Option<u32> {
        if self.is_first {
            None
        } else {
            Some(self.rank - 1)
        }
    }

    /// Get the rank of the next stage
    pub fn next_rank(&self) -> Option<u32> {
        if self.is_last {
            None
        } else {
            Some(self.rank + 1)
        }
    }
}

impl Module for PipelineStage {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.module.forward(input)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        self.module.parameters()
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.module.named_parameters()
    }

    fn training(&self) -> bool {
        self.module.training()
    }

    fn train(&mut self) {
        self.module.train()
    }

    fn eval(&mut self) {
        self.module.eval()
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.module.to_device(device)
    }
}

/// Main pipeline parallelism wrapper
pub struct PipelineParallel {
    /// The pipeline stage for this rank
    stage: PipelineStage,
    /// Process group for communication
    process_group: Arc<ProcessGroup>,
    /// Pipeline configuration
    config: PipelineConfig,
    /// Activation cache for micro-batches (forward pass)
    activation_cache: VecDeque<Tensor>,
    /// Gradient cache for micro-batches (backward pass)
    gradient_cache: VecDeque<Option<Tensor>>,
    /// Current micro-batch being processed
    current_micro_batch: usize,
    /// Whether we're in training mode
    training: bool,
}

impl PipelineParallel {
    /// Create a new pipeline parallel wrapper
    pub fn new(
        stage: PipelineStage,
        process_group: Arc<ProcessGroup>,
        config: PipelineConfig,
    ) -> TorshResult<Self> {
        let activation_cache = VecDeque::with_capacity(config.num_micro_batches);
        let gradient_cache = VecDeque::with_capacity(config.num_micro_batches);

        Ok(Self {
            stage,
            process_group,
            config,
            activation_cache,
            gradient_cache,
            current_micro_batch: 0,
            training: true,
        })
    }

    /// Process a forward pass for the entire mini-batch
    pub async fn forward(&mut self, input: &Tensor) -> TorshResult<Option<Tensor>> {
        match self.config.schedule {
            ScheduleType::GPipe => self.forward_gpipe(input).await,
            ScheduleType::OneFOneBInterleaved => self.forward_1f1b(input).await,
            ScheduleType::InterleavedOneFOneB => self.forward_interleaved_1f1b(input).await,
        }
    }

    /// Process a backward pass for the entire mini-batch
    pub async fn backward(&mut self, grad_output: Option<&Tensor>) -> TorshResult<()> {
        match self.config.schedule {
            ScheduleType::GPipe => self.backward_gpipe(grad_output).await,
            ScheduleType::OneFOneBInterleaved => self.backward_1f1b(grad_output).await,
            ScheduleType::InterleavedOneFOneB => self.backward_interleaved_1f1b(grad_output).await,
        }
    }

    /// GPipe forward pass: process all micro-batches sequentially
    async fn forward_gpipe(&mut self, input: &Tensor) -> TorshResult<Option<Tensor>> {
        let micro_batches = self.split_mini_batch(input)?;
        let mut final_output: Option<Tensor> = None;

        // Clear caches
        self.activation_cache.clear();
        self.gradient_cache.clear();

        for (micro_batch_id, micro_batch) in micro_batches.iter().enumerate() {
            let mut current_input = micro_batch.clone();

            // Receive input from previous stage if not first stage
            if let Some(prev_rank) = self.stage.prev_rank() {
                let tag = self.config.base_tag + micro_batch_id as u32;
                recv(&mut current_input, prev_rank, tag, &self.process_group).await?;
            }

            // Forward pass through this stage
            let output = self.stage.forward(&current_input)?;

            // Cache activation for backward pass
            self.activation_cache.push_back(current_input);
            self.gradient_cache.push_back(None); // Will be filled during backward

            // Send output to next stage if not last stage
            if let Some(next_rank) = self.stage.next_rank() {
                let tag = self.config.base_tag + micro_batch_id as u32;
                send(&output, next_rank, tag, &self.process_group).await?;
            } else {
                // Last stage: accumulate final output
                final_output = Some(if let Some(ref existing) = final_output {
                    existing.add(&output)?
                } else {
                    output
                });
            }
        }

        // Average the accumulated output if this is the last stage
        if let Some(ref mut output) = final_output {
            let num_micro_batches = self.config.num_micro_batches as f32;
            *output = output.div_scalar(num_micro_batches)?;
        }

        Ok(final_output)
    }

    /// 1F1B forward pass: interleave forward and backward passes for efficiency
    async fn forward_1f1b(&mut self, input: &Tensor) -> TorshResult<Option<Tensor>> {
        let micro_batches = self.split_mini_batch(input)?;
        let mut final_output: Option<Tensor> = None;

        // Clear caches
        self.activation_cache.clear();
        self.gradient_cache.clear();

        // Warmup phase: fill the pipeline
        let warmup_steps = std::cmp::min(self.stage.stage_id + 1, self.config.num_micro_batches);

        for (step, micro_batch) in micro_batches.iter().enumerate().take(warmup_steps) {
            let mut current_input = micro_batch.clone();

            // Receive input from previous stage if not first stage
            if let Some(prev_rank) = self.stage.prev_rank() {
                let tag = self.config.base_tag + step as u32;
                recv(&mut current_input, prev_rank, tag, &self.process_group).await?;
            }

            // Forward pass through this stage
            let output = self.stage.forward(&current_input)?;

            // Cache activation for backward pass
            self.activation_cache.push_back(current_input);
            self.gradient_cache.push_back(None);

            // Send output to next stage if not last stage
            if let Some(next_rank) = self.stage.next_rank() {
                let tag = self.config.base_tag + step as u32;
                send(&output, next_rank, tag, &self.process_group).await?;
            } else {
                // Last stage: accumulate final output
                final_output = Some(if let Some(ref existing) = final_output {
                    existing.add(&output)?
                } else {
                    output
                });
            }
        }

        // 1F1B phase: one forward, one backward
        for step in warmup_steps..self.config.num_micro_batches {
            // Forward pass for new micro-batch
            if step < micro_batches.len() {
                let mut current_input = micro_batches[step].clone();

                if let Some(prev_rank) = self.stage.prev_rank() {
                    let tag = self.config.base_tag + step as u32;
                    recv(&mut current_input, prev_rank, tag, &self.process_group).await?;
                }

                let output = self.stage.forward(&current_input)?;
                self.activation_cache.push_back(current_input);
                self.gradient_cache.push_back(None);

                if let Some(next_rank) = self.stage.next_rank() {
                    let tag = self.config.base_tag + step as u32;
                    send(&output, next_rank, tag, &self.process_group).await?;
                } else {
                    final_output = Some(if let Some(ref existing) = final_output {
                        existing.add(&output)?
                    } else {
                        output
                    });
                }
            }

            // Backward pass for oldest cached micro-batch
            if self.training && !self.activation_cache.is_empty() {
                self.process_backward_micro_batch().await?;
            }
        }

        // Cooldown phase: finish remaining backward passes
        while !self.activation_cache.is_empty() && self.training {
            self.process_backward_micro_batch().await?;
        }

        // Average the accumulated output if this is the last stage
        if let Some(ref mut output) = final_output {
            let num_micro_batches = self.config.num_micro_batches as f32;
            *output = output.div_scalar(num_micro_batches)?;
        }

        Ok(final_output)
    }

    /// Interleaved 1F1B forward pass: more sophisticated load balancing
    async fn forward_interleaved_1f1b(&mut self, input: &Tensor) -> TorshResult<Option<Tensor>> {
        // For simplicity, use regular 1F1B
        // In a full implementation, this would use virtual pipeline stages
        self.forward_1f1b(input).await
    }

    /// Process a single micro-batch backward pass
    async fn process_backward_micro_batch(&mut self) -> TorshResult<()> {
        if let Some(activation) = self.activation_cache.pop_front() {
            let grad_output = if self.stage.is_last {
                // Last stage: create gradient from loss (mock implementation)
                Some(Tensor::ones_like(&activation)?)
            } else {
                // Receive gradient from next stage
                let next_rank = self.stage.next_rank().unwrap();
                let tag = self.config.base_tag + 10000 + self.current_micro_batch as u32;
                let mut grad = Tensor::zeros_like(&activation)?;
                recv(&mut grad, next_rank, tag, &self.process_group).await?;
                Some(grad)
            };

            // Backward pass through this stage (mock implementation)
            let grad_input = grad_output.clone();

            // Send gradient to previous stage if not first stage
            if let Some(prev_rank) = self.stage.prev_rank() {
                if let Some(ref grad) = grad_input {
                    let tag = self.config.base_tag + 10000 + self.current_micro_batch as u32;
                    send(grad, prev_rank, tag, &self.process_group).await?;
                }
            }

            // Store gradient for accumulation
            self.gradient_cache.push_back(grad_input);
            self.current_micro_batch += 1;
        }

        Ok(())
    }

    /// GPipe backward pass: process all micro-batches sequentially (reverse order)
    async fn backward_gpipe(&mut self, _grad_output: Option<&Tensor>) -> TorshResult<()> {
        // Process micro-batches in reverse order
        while !self.activation_cache.is_empty() {
            self.process_backward_micro_batch().await?;
        }

        self.synchronize_gradients().await
    }

    /// 1F1B backward pass: gradients are processed during forward pass
    async fn backward_1f1b(&mut self, _grad_output: Option<&Tensor>) -> TorshResult<()> {
        // Most backward work is done during forward_1f1b
        // Just ensure all gradients are synchronized
        self.synchronize_gradients().await
    }

    /// Interleaved 1F1B backward pass
    async fn backward_interleaved_1f1b(&mut self, grad_output: Option<&Tensor>) -> TorshResult<()> {
        // For simplicity, use regular 1F1B
        self.backward_1f1b(grad_output).await
    }

    /// Split mini-batch into micro-batches
    fn split_mini_batch(&self, input: &Tensor) -> TorshResult<Vec<Tensor>> {
        let batch_size = input.shape().dims()[0];
        let micro_batch_size = batch_size.div_ceil(self.config.num_micro_batches);

        let mut micro_batches = Vec::new();

        for i in 0..self.config.num_micro_batches {
            let start = i * micro_batch_size;
            let end = std::cmp::min(start + micro_batch_size, batch_size);

            if start < batch_size {
                // Create a slice of the input tensor for this micro-batch
                let micro_batch = input.slice(0, start, end)?;
                let tensor = micro_batch.to_tensor()?;
                micro_batches.push(tensor);
            }
        }

        Ok(micro_batches)
    }

    /// Synchronize gradients across all pipeline stages
    async fn synchronize_gradients(&mut self) -> TorshResult<()> {
        if !self.config.accumulate_gradients {
            return Ok(());
        }

        // In a complete implementation, this would:
        // 1. Accumulate gradients from all micro-batches
        // 2. Average gradients by number of micro-batches
        // 3. Apply gradients to parameters

        info!(
            " Synchronizing gradients for stage {} (rank {})",
            self.stage.stage_id, self.stage.rank
        );

        // Clear caches after synchronization
        self.gradient_cache.clear();
        self.current_micro_batch = 0;

        Ok(())
    }

    /// Get pipeline statistics
    pub fn get_pipeline_stats(&self) -> PipelineStats {
        PipelineStats {
            stage_id: self.stage.stage_id,
            num_stages: self.stage.num_stages,
            rank: self.stage.rank,
            num_micro_batches: self.config.num_micro_batches,
            schedule: self.config.schedule,
            cached_activations: self.activation_cache.len(),
            cached_gradients: self.gradient_cache.len(),
            current_micro_batch: self.current_micro_batch,
        }
    }

    /// Clear all caches (useful for memory management)
    pub fn clear_caches(&mut self) {
        self.activation_cache.clear();
        self.gradient_cache.clear();
        self.current_micro_batch = 0;
    }

    /// Check if this stage is the first in the pipeline
    pub fn is_first_stage(&self) -> bool {
        self.stage.is_first
    }

    /// Check if this stage is the last in the pipeline
    pub fn is_last_stage(&self) -> bool {
        self.stage.is_last
    }

    /// Get the stage ID
    pub fn stage_id(&self) -> usize {
        self.stage.stage_id
    }

    /// Get the total number of stages
    pub fn num_stages(&self) -> usize {
        self.stage.num_stages
    }
}

impl Module for PipelineParallel {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Synchronous forward pass (for compatibility)
        // In practice, should use async forward method
        self.stage.forward(input)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        self.stage.parameters()
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.stage.named_parameters()
    }

    fn training(&self) -> bool {
        self.training
    }

    fn train(&mut self) {
        self.training = true;
        self.stage.train();
    }

    fn eval(&mut self) {
        self.training = false;
        self.stage.eval();
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.stage.to_device(device)
    }
}

/// Statistics about pipeline execution
#[derive(Debug, Clone)]
pub struct PipelineStats {
    /// Stage ID in the pipeline
    pub stage_id: usize,
    /// Total number of stages
    pub num_stages: usize,
    /// Rank of this process
    pub rank: u32,
    /// Number of micro-batches
    pub num_micro_batches: usize,
    /// Pipeline scheduling strategy
    pub schedule: ScheduleType,
    /// Number of cached activations
    pub cached_activations: usize,
    /// Number of cached gradients
    pub cached_gradients: usize,
    /// Current micro-batch being processed
    pub current_micro_batch: usize,
}

/// Utility function to create a pipeline from a list of modules
pub fn create_pipeline_stages(
    modules: Vec<Box<dyn Module>>,
    process_group: Arc<ProcessGroup>,
    devices: Vec<DeviceType>,
) -> TorshResult<Vec<PipelineStage>> {
    let num_stages = modules.len();
    let world_size = process_group.world_size() as usize;

    if num_stages != world_size {
        return Err(TorshDistributedError::invalid_argument(
            "num_stages",
            format!(
                "Number of stages ({}) must match world size ({})",
                num_stages, world_size
            ),
            format!("num_stages = world_size = {}", world_size),
        ));
    }

    if devices.len() != num_stages {
        return Err(TorshDistributedError::invalid_argument(
            "devices",
            format!(
                "Number of devices ({}) must match number of stages ({})",
                devices.len(),
                num_stages
            ),
            format!("devices.len() = num_stages = {}", num_stages),
        ));
    }

    let mut stages = Vec::new();

    for (i, (module, device)) in modules.into_iter().zip(devices.into_iter()).enumerate() {
        let stage = PipelineStage::new(module, i, num_stages, i as u32, device);
        stages.push(stage);
    }

    Ok(stages)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::MockBackend;

    use torsh_nn::layers::Linear;

    async fn create_mock_process_group(rank: u32, world_size: u32) -> TorshResult<ProcessGroup> {
        let _backend = Box::new(MockBackend::new(rank, world_size));
        ProcessGroup::new(
            crate::backend::BackendType::Gloo,
            rank,
            world_size,
            "localhost",
            12345,
        )
        .await
    }

    #[tokio::test]
    async fn test_pipeline_config_default() {
        let config = PipelineConfig::default();
        assert_eq!(config.num_micro_batches, 4);
        assert_eq!(config.schedule, ScheduleType::OneFOneBInterleaved);
        assert!(config.accumulate_gradients);
    }

    #[tokio::test]
    async fn test_pipeline_stage_creation() -> TorshResult<()> {
        let linear = Linear::new(10, 5, true);
        let module = Box::new(linear) as Box<dyn Module>;

        let stage = PipelineStage::new(
            module,
            1, // stage_id
            3, // num_stages
            1, // rank
            DeviceType::Cpu,
        );

        assert_eq!(stage.stage_id, 1);
        assert_eq!(stage.num_stages, 3);
        assert_eq!(stage.rank, 1);
        assert!(!stage.is_first);
        assert!(!stage.is_last);
        assert_eq!(stage.prev_rank(), Some(0));
        assert_eq!(stage.next_rank(), Some(2));

        Ok(())
    }

    #[tokio::test]
    async fn test_first_and_last_stage() -> TorshResult<()> {
        let linear = Linear::new(10, 5, true);
        let module = Box::new(linear) as Box<dyn Module>;

        // First stage
        let first_stage = PipelineStage::new(
            module,
            0, // stage_id
            3, // num_stages
            0, // rank
            DeviceType::Cpu,
        );

        assert!(first_stage.is_first);
        assert!(!first_stage.is_last);
        assert_eq!(first_stage.prev_rank(), None);
        assert_eq!(first_stage.next_rank(), Some(1));

        // Last stage
        let linear2 = Linear::new(10, 5, true);
        let module2 = Box::new(linear2) as Box<dyn Module>;

        let last_stage = PipelineStage::new(
            module2,
            2, // stage_id
            3, // num_stages
            2, // rank
            DeviceType::Cpu,
        );

        assert!(!last_stage.is_first);
        assert!(last_stage.is_last);
        assert_eq!(last_stage.prev_rank(), Some(1));
        assert_eq!(last_stage.next_rank(), None);

        Ok(())
    }

    #[tokio::test]
    async fn test_pipeline_parallel_creation() -> TorshResult<()> {
        let linear = Linear::new(10, 5, true);
        let module = Box::new(linear) as Box<dyn Module>;

        let stage = PipelineStage::new(module, 0, 2, 0, DeviceType::Cpu);
        let process_group = Arc::new(create_mock_process_group(0, 2).await?);
        let config = PipelineConfig::default();

        let pipeline = PipelineParallel::new(stage, process_group, config)?;

        assert!(pipeline.is_first_stage());
        assert!(!pipeline.is_last_stage());
        assert_eq!(pipeline.stage_id(), 0);
        assert_eq!(pipeline.num_stages(), 2);

        Ok(())
    }

    #[tokio::test]
    async fn test_micro_batch_splitting() -> TorshResult<()> {
        let linear = Linear::new(10, 5, true);
        let module = Box::new(linear) as Box<dyn Module>;

        let stage = PipelineStage::new(module, 0, 1, 0, DeviceType::Cpu);
        let process_group = Arc::new(create_mock_process_group(0, 1).await?);
        let config = PipelineConfig {
            num_micro_batches: 3,
            ..Default::default()
        };

        let pipeline = PipelineParallel::new(stage, process_group, config)?;

        // Create a batch of size 7
        let input = Tensor::from_vec(vec![0.0; 7 * 10], &[7, 10])?;

        let micro_batches = pipeline.split_mini_batch(&input)?;

        assert_eq!(micro_batches.len(), 3);

        // Check micro-batch sizes: should be [3, 3, 1] for batch size 7 split into 3
        let expected_sizes = [3, 3, 1];
        for (i, micro_batch) in micro_batches.iter().enumerate() {
            assert_eq!(micro_batch.shape().dims()[0], expected_sizes[i]);
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_pipeline_stats() -> TorshResult<()> {
        let linear = Linear::new(10, 5, true);
        let module = Box::new(linear) as Box<dyn Module>;

        let stage = PipelineStage::new(module, 1, 4, 1, DeviceType::Cpu);
        let process_group = Arc::new(create_mock_process_group(1, 4).await?);
        let config = PipelineConfig {
            num_micro_batches: 8,
            schedule: ScheduleType::GPipe,
            ..Default::default()
        };

        let pipeline = PipelineParallel::new(stage, process_group, config)?;
        let stats = pipeline.get_pipeline_stats();

        assert_eq!(stats.stage_id, 1);
        assert_eq!(stats.num_stages, 4);
        assert_eq!(stats.rank, 1);
        assert_eq!(stats.num_micro_batches, 8);
        assert_eq!(stats.schedule, ScheduleType::GPipe);
        assert_eq!(stats.cached_activations, 0);
        assert_eq!(stats.cached_gradients, 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_create_pipeline_stages() -> TorshResult<()> {
        let process_group = Arc::new(create_mock_process_group(0, 3).await?);

        let modules: Vec<Box<dyn Module>> = vec![
            Box::new(Linear::new(10, 8, true)),
            Box::new(Linear::new(8, 6, true)),
            Box::new(Linear::new(6, 4, true)),
        ];

        let devices = vec![DeviceType::Cpu, DeviceType::Cpu, DeviceType::Cpu];

        let stages = create_pipeline_stages(modules, process_group, devices)?;

        assert_eq!(stages.len(), 3);

        for (i, stage) in stages.iter().enumerate() {
            assert_eq!(stage.stage_id, i);
            assert_eq!(stage.num_stages, 3);
            assert_eq!(stage.rank, i as u32);
        }

        assert!(stages[0].is_first);
        assert!(!stages[0].is_last);
        assert!(!stages[1].is_first);
        assert!(!stages[1].is_last);
        assert!(!stages[2].is_first);
        assert!(stages[2].is_last);

        Ok(())
    }

    #[tokio::test]
    async fn test_pipeline_clear_caches() -> TorshResult<()> {
        let linear = Linear::new(10, 5, true);
        let module = Box::new(linear) as Box<dyn Module>;

        let stage = PipelineStage::new(module, 0, 1, 0, DeviceType::Cpu);
        let process_group = Arc::new(create_mock_process_group(0, 1).await?);
        let config = PipelineConfig::default();

        let mut pipeline = PipelineParallel::new(stage, process_group, config)?;

        // Simulate some cached data
        let dummy_tensor = Tensor::from_vec(vec![0.0; 2 * 10], &[2, 10])?;
        pipeline.activation_cache.push_back(dummy_tensor.clone());
        pipeline.gradient_cache.push_back(Some(dummy_tensor));
        pipeline.current_micro_batch = 5;

        assert_eq!(pipeline.activation_cache.len(), 1);
        assert_eq!(pipeline.gradient_cache.len(), 1);
        assert_eq!(pipeline.current_micro_batch, 5);

        pipeline.clear_caches();

        assert_eq!(pipeline.activation_cache.len(), 0);
        assert_eq!(pipeline.gradient_cache.len(), 0);
        assert_eq!(pipeline.current_micro_batch, 0);

        Ok(())
    }
}
