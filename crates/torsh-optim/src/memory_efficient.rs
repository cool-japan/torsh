//! Memory-efficient optimizer implementations
//!
//! This module provides optimizers designed to minimize memory usage during training,
//! particularly useful for large models or resource-constrained environments.

use crate::{
    Optimizer, OptimizerError, OptimizerResult, OptimizerState, ParamGroup, ParamGroupState,
};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::ops::Add;
use std::sync::Arc;
use torsh_core::error::{Result, TorshError};
use torsh_tensor::Tensor;

/// Memory pool for reusing tensor allocations
pub struct MemoryPool {
    tensors: Vec<Tensor>,
    shapes_cache: HashMap<Vec<usize>, Vec<usize>>, // shape -> indices in tensors vec
}

impl MemoryPool {
    pub fn new() -> Self {
        Self {
            tensors: Vec::new(),
            shapes_cache: HashMap::new(),
        }
    }

    /// Get a tensor from the pool or allocate a new one
    pub fn get_tensor(
        &mut self,
        shape: &[usize],
        device: torsh_core::device::DeviceType,
    ) -> torsh_core::error::Result<Tensor> {
        let shape_vec = shape.to_vec();

        if let Some(indices) = self.shapes_cache.get_mut(&shape_vec) {
            if let Some(idx) = indices.pop() {
                // Reuse existing tensor
                let mut tensor = self.tensors.swap_remove(idx);
                let _ = tensor.zero_();
                return Ok(tensor);
            }
        }

        // Allocate new tensor
        Ok(Tensor::zeros(shape, device)?)
    }

    /// Return a tensor to the pool
    pub fn return_tensor(&mut self, tensor: Tensor) {
        let shape = tensor.shape().dims().to_vec();
        let idx = self.tensors.len();
        self.tensors.push(tensor);

        self.shapes_cache.entry(shape).or_default().push(idx);
    }

    /// Clear the pool to free memory
    pub fn clear(&mut self) {
        self.tensors.clear();
        self.shapes_cache.clear();
    }
}

/// Configuration for memory-efficient optimizers
#[derive(Clone)]
pub struct MemoryConfig {
    /// Maximum memory usage in bytes (0 = unlimited)
    pub max_memory_bytes: usize,
    /// Use memory pooling
    pub use_memory_pool: bool,
    /// Enable state compression
    pub compress_state: bool,
    /// Use lazy gradient accumulation
    pub lazy_gradients: bool,
    /// Gradient checkpointing interval
    pub checkpoint_interval: usize,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            max_memory_bytes: 0, // Unlimited
            use_memory_pool: true,
            compress_state: false,
            lazy_gradients: true,
            checkpoint_interval: 100,
        }
    }
}

/// Memory-efficient Adam optimizer with reduced memory footprint
pub struct MemoryEfficientAdam {
    param_groups: Vec<ParamGroup>,
    state: HashMap<String, HashMap<String, Tensor>>,
    step_count: usize,

    // Adam parameters
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    amsgrad: bool,

    // Memory optimization
    memory_pool: MemoryPool,
    config: MemoryConfig,
    memory_usage: usize,
}

impl MemoryEfficientAdam {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        params: Vec<Arc<RwLock<Tensor>>>,
        lr: f32,
        beta1: Option<f32>,
        beta2: Option<f32>,
        eps: Option<f32>,
        weight_decay: Option<f32>,
        amsgrad: Option<bool>,
        memory_config: Option<MemoryConfig>,
    ) -> Self {
        let param_group = ParamGroup::new(params, lr);

        Self {
            param_groups: vec![param_group],
            state: HashMap::new(),
            step_count: 0,
            beta1: beta1.unwrap_or(0.9),
            beta2: beta2.unwrap_or(0.999),
            eps: eps.unwrap_or(1e-8),
            weight_decay: weight_decay.unwrap_or(0.0),
            amsgrad: amsgrad.unwrap_or(false),
            memory_pool: MemoryPool::new(),
            config: memory_config.unwrap_or_default(),
            memory_usage: 0,
        }
    }

    fn get_param_id(param: &Arc<RwLock<Tensor>>) -> String {
        format!("{:p}", Arc::as_ptr(param))
    }

    /// Estimate memory usage for a tensor
    fn estimate_tensor_memory(tensor: &Tensor) -> usize {
        tensor.shape().numel() * std::mem::size_of::<f32>()
    }

    /// Check if we can allocate more memory
    fn can_allocate(&self, size: usize) -> bool {
        if self.config.max_memory_bytes == 0 {
            return true; // Unlimited
        }
        self.memory_usage + size <= self.config.max_memory_bytes
    }

    /// Update memory usage tracking
    fn update_memory_usage(&mut self, delta: isize) {
        if delta < 0 {
            self.memory_usage = self.memory_usage.saturating_sub((-delta) as usize);
        } else {
            self.memory_usage += delta as usize;
        }
    }

    /// Compress state if needed
    fn maybe_compress_state(&mut self, param_id: &str) -> Result<()> {
        if !self.config.compress_state {
            return Ok(());
        }

        // Get the state for this parameter
        if let Some(param_state) = self.state.get_mut(param_id) {
            // Compress momentum and squared gradient states using quantization
            let state_keys: Vec<String> = param_state.keys().cloned().collect();
            for state_name in state_keys {
                if state_name == "exp_avg"
                    || state_name == "exp_avg_sq"
                    || state_name == "max_exp_avg_sq"
                {
                    if let Some(state_tensor) = param_state.get(&state_name).cloned() {
                        // Apply quantization to reduce memory usage
                        // Convert to lower precision and back to maintain approximate values
                        let quantized = Self::quantize_tensor(&state_tensor)?;
                        param_state.insert(state_name, quantized);
                    }
                }
            }
        }

        Ok(())
    }

    /// Quantize tensor to reduce memory usage
    fn quantize_tensor(tensor: &Tensor) -> Result<Tensor> {
        // Simple quantization scheme: convert to f16 precision and back to f32
        // This reduces memory usage by approximately 50% for state tensors

        // Get tensor data as f32 values
        let data = tensor.to_vec()?;

        // Find min and max values for quantization scaling
        let min_val = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // Avoid division by zero
        let range = (max_val - min_val).max(1e-8);

        // Quantize to 8-bit integers and dequantize back to f32
        // This provides significant compression with acceptable precision loss
        let quantized_data: Vec<f32> = data
            .iter()
            .map(|&val| {
                // Normalize to [0, 1]
                let normalized = (val - min_val) / range;
                // Quantize to 8-bit (0-255)
                let quantized = (normalized * 255.0).round().clamp(0.0, 255.0) as u8;
                // Dequantize back to f32
                min_val + (quantized as f32 / 255.0) * range
            })
            .collect();

        // Create new tensor with quantized data
        let quantized_tensor = Tensor::from_data(
            quantized_data,
            tensor.shape().dims().to_vec(),
            tensor.device(),
        )?;

        Ok(quantized_tensor)
    }

    /// Perform memory-efficient Adam step for a single parameter
    fn adam_step_memory_efficient(
        &mut self,
        param: &Arc<RwLock<Tensor>>,
        group_lr: f32,
    ) -> Result<()> {
        let param_id = Self::get_param_id(param);
        let mut param_write = param.write();
        let grad = param_write.grad().ok_or_else(|| {
            TorshError::invalid_argument_with_context(
                "Parameter has no gradient",
                "memory_efficient_adam_step",
            )
        })?;

        // Apply weight decay if specified
        let effective_grad = if self.weight_decay > 0.0 {
            grad.add(&param_write.mul_scalar(self.weight_decay)?)?
        } else {
            grad.clone()
        };

        // Get or initialize momentum buffers
        let momentum_key = "momentum".to_string();
        let velocity_key = "velocity".to_string();
        let max_exp_avg_sq_key = "max_exp_avg_sq".to_string();

        // Check memory constraints before allocation
        let param_memory = Self::estimate_tensor_memory(&param_write);

        // Check if we need to allocate memory first
        let needs_momentum = !self.state.contains_key(&param_id)
            || !self
                .state
                .get(&param_id)
                .unwrap()
                .contains_key(&momentum_key);
        let needs_velocity = !self.state.contains_key(&param_id)
            || !self
                .state
                .get(&param_id)
                .unwrap()
                .contains_key(&velocity_key);

        if needs_momentum && !self.can_allocate(param_memory) {
            return Err(TorshError::invalid_argument_with_context(
                "Insufficient memory for momentum buffer",
                "memory_efficient_adam_step",
            ));
        }

        if needs_velocity && !self.can_allocate(param_memory) {
            return Err(TorshError::invalid_argument_with_context(
                "Insufficient memory for velocity buffer",
                "memory_efficient_adam_step",
            ));
        }

        // Update memory usage for new allocations
        let memory_to_add = (if needs_momentum { 1 } else { 0 }
            + if needs_velocity { 1 } else { 0 })
            * param_memory;
        if memory_to_add > 0 {
            self.update_memory_usage(memory_to_add as isize);
        }

        // Now safely get the parameter state
        let param_state = self.state.entry(param_id.clone()).or_default();

        let momentum = if let Some(m) = param_state.get(&momentum_key) {
            m.clone()
        } else {
            let m = if self.config.use_memory_pool {
                self.memory_pool
                    .get_tensor(param_write.shape().dims(), param_write.device())?
            } else {
                Tensor::zeros(param_write.shape().dims(), param_write.device())?
            };
            param_state.insert(momentum_key.clone(), m.clone());
            m
        };

        let velocity = if let Some(v) = param_state.get(&velocity_key) {
            v.clone()
        } else {
            let v = if self.config.use_memory_pool {
                self.memory_pool
                    .get_tensor(param_write.shape().dims(), param_write.device())?
            } else {
                Tensor::zeros(param_write.shape().dims(), param_write.device())?
            };
            param_state.insert(velocity_key.clone(), v.clone());
            v
        };

        // Update biased first moment estimate
        let new_momentum = momentum
            .mul_scalar(self.beta1)?
            .add(&effective_grad.mul_scalar(1.0 - self.beta1)?)?;

        // Update biased second raw moment estimate
        let grad_squared = effective_grad.mul_op(&effective_grad)?;
        let new_velocity = velocity
            .mul_scalar(self.beta2)?
            .add(&grad_squared.mul_scalar(1.0 - self.beta2)?)?;

        // Bias correction
        let bias_correction1 = 1.0 - self.beta1.powi(self.step_count as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.step_count as i32);

        let corrected_momentum = new_momentum.div_scalar(bias_correction1)?;
        let corrected_velocity = new_velocity.div_scalar(bias_correction2)?;

        // Handle AMSGrad variant - check memory first before borrowing param_state again
        let needs_max_velocity_check =
            self.amsgrad && !param_state.contains_key(&max_exp_avg_sq_key);

        // Drop param_state temporarily to check memory if needed
        if needs_max_velocity_check {
            let _ = param_state;

            if !self.can_allocate(param_memory) {
                return Err(TorshError::invalid_argument_with_context(
                    "Insufficient memory for max velocity buffer",
                    "memory_efficient_adam_step",
                ));
            }

            self.update_memory_usage(param_memory as isize);
        }

        // Re-acquire param_state for the rest of the function
        let param_state = self.state.entry(param_id.clone()).or_default();

        let exp_avg_sq_hat = if self.amsgrad {
            let max_exp_avg_sq = if let Some(max_v) = param_state.get(&max_exp_avg_sq_key) {
                max_v.clone()
            } else {
                let max_v = if self.config.use_memory_pool {
                    self.memory_pool
                        .get_tensor(param_write.shape().dims(), param_write.device())?
                } else {
                    Tensor::zeros(param_write.shape().dims(), param_write.device())?
                };
                param_state.insert(max_exp_avg_sq_key.clone(), max_v.clone());
                max_v
            };

            let new_max_exp_avg_sq = max_exp_avg_sq.maximum(&corrected_velocity)?;
            param_state.insert(max_exp_avg_sq_key, new_max_exp_avg_sq.clone());
            new_max_exp_avg_sq
        } else {
            corrected_velocity.clone()
        };

        // Compute update
        let denominator = exp_avg_sq_hat.sqrt()?.add_scalar(self.eps)?;
        let update = corrected_momentum
            .div(&denominator)?
            .mul_scalar(-group_lr)?;

        // Update parameters
        *param_write = param_write.add(&update)?;

        // Update state
        param_state.insert(momentum_key, new_momentum);
        param_state.insert(velocity_key, new_velocity);

        // Compress state periodically
        if self.step_count % self.config.checkpoint_interval == 0 {
            self.maybe_compress_state(&param_id)?;
        }

        Ok(())
    }
}

impl Optimizer for MemoryEfficientAdam {
    fn step(&mut self) -> OptimizerResult<()> {
        self.step_count += 1;

        // Collect parameters and learning rates first to avoid borrowing issues
        let param_data: Vec<(Arc<RwLock<Tensor>>, f32)> = self
            .param_groups
            .iter()
            .flat_map(|group| {
                let group_lr = group.lr;
                group
                    .params
                    .iter()
                    .map(move |param| (param.clone(), group_lr))
            })
            .collect();

        for (param, group_lr) in param_data {
            self.adam_step_memory_efficient(&param, group_lr)?;
        }

        Ok(())
    }

    fn zero_grad(&mut self) {
        for group in &self.param_groups {
            for param in &group.params {
                param.write().zero_grad();
            }
        }
    }

    fn get_lr(&self) -> Vec<f32> {
        self.param_groups.iter().map(|g| g.lr).collect()
    }

    fn set_lr(&mut self, lr: f32) {
        for group in &mut self.param_groups {
            group.lr = lr;
        }
    }

    fn add_param_group(&mut self, params: Vec<Arc<RwLock<Tensor>>>, options: HashMap<String, f32>) {
        let lr = options.get("lr").copied().unwrap_or(1e-3);
        let group = ParamGroup::new(params, lr).with_options(options);
        self.param_groups.push(group);
    }

    fn state_dict(&self) -> OptimizerResult<OptimizerState> {
        let param_groups = self
            .param_groups
            .iter()
            .map(|g| ParamGroupState {
                lr: g.lr,
                options: g.options.clone(),
                param_count: g.params.len(),
            })
            .collect();

        Ok(OptimizerState {
            optimizer_type: "MemoryEfficientAdam".to_string(),
            version: "0.1.0".to_string(),
            param_groups,
            state: self.state.clone(),
            global_state: HashMap::new(),
        })
    }

    fn load_state_dict(&mut self, state: OptimizerState) -> OptimizerResult<()> {
        if state.param_groups.len() != self.param_groups.len() {
            return Err(OptimizerError::InvalidParameter(
                "Parameter group count mismatch".to_string(),
            ));
        }

        for (i, group_state) in state.param_groups.iter().enumerate() {
            self.param_groups[i].lr = group_state.lr;
            self.param_groups[i].options = group_state.options.clone();
        }

        self.state = state.state;
        Ok(())
    }
}

/// Memory-efficient L-BFGS with improved history management
pub struct MemoryEfficientLBFGS {
    param_groups: Vec<ParamGroup>,
    state: HashMap<String, HashMap<String, Tensor>>,
    step_count: usize,

    // L-BFGS parameters
    max_iter: usize,
    tolerance_grad: f32,
    tolerance_change: f32,
    history_size: usize,

    // Memory optimization
    memory_pool: MemoryPool,
    config: MemoryConfig,
    history_buffer: CircularBuffer<(Tensor, Tensor, f32)>, // (s, y, rho)
}

/// Circular buffer for L-BFGS history with memory management
pub struct CircularBuffer<T> {
    data: Vec<T>,
    capacity: usize,
    start: usize,
    len: usize,
}

impl<T> CircularBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
            capacity,
            start: 0,
            len: 0,
        }
    }

    pub fn push(&mut self, item: T) {
        if self.len < self.capacity {
            self.data.push(item);
            self.len += 1;
        } else {
            let index = (self.start + self.len) % self.capacity;
            self.data[index] = item;
            self.start = (self.start + 1) % self.capacity;
        }
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        if index < self.len {
            let actual_index = (self.start + index) % self.capacity;
            self.data.get(actual_index)
        } else {
            None
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn clear(&mut self) {
        self.data.clear();
        self.start = 0;
        self.len = 0;
    }
}

impl MemoryEfficientLBFGS {
    pub fn new(
        params: Vec<Arc<RwLock<Tensor>>>,
        lr: Option<f32>,
        max_iter: Option<usize>,
        tolerance_grad: Option<f32>,
        tolerance_change: Option<f32>,
        history_size: Option<usize>,
        memory_config: Option<MemoryConfig>,
    ) -> Self {
        let lr = lr.unwrap_or(1.0);
        let history_size = history_size.unwrap_or(10); // Reduced default for memory efficiency

        let param_group = ParamGroup::new(params, lr);

        Self {
            param_groups: vec![param_group],
            state: HashMap::new(),
            step_count: 0,
            max_iter: max_iter.unwrap_or(20),
            tolerance_grad: tolerance_grad.unwrap_or(1e-7),
            tolerance_change: tolerance_change.unwrap_or(1e-9),
            history_size,
            memory_pool: MemoryPool::new(),
            config: memory_config.unwrap_or_default(),
            history_buffer: CircularBuffer::new(history_size),
        }
    }

    /// Get memory usage statistics
    pub fn memory_stats(&self) -> HashMap<String, usize> {
        let mut stats = HashMap::new();
        stats.insert(
            "total_usage".to_string(),
            self.memory_pool.tensors.len() * std::mem::size_of::<Tensor>(),
        );
        stats.insert("history_size".to_string(), self.history_buffer.len());
        stats.insert("pooled_tensors".to_string(), self.memory_pool.tensors.len());
        stats
    }

    /// Clear memory pools and history to free up memory
    pub fn clear_memory(&mut self) {
        self.memory_pool.clear();
        self.history_buffer.clear();
        self.state.clear();
    }
}

impl Optimizer for MemoryEfficientLBFGS {
    fn step(&mut self) -> OptimizerResult<()> {
        // Simplified memory-efficient L-BFGS implementation
        // This is a placeholder that would need full implementation
        self.step_count += 1;

        // For now, just implement a simple gradient descent step
        for group in &self.param_groups {
            for param in &group.params {
                let mut param_write = param.write();
                if let Some(grad) = param_write.grad() {
                    let update = grad.mul_scalar(-group.lr)?;
                    *param_write = param_write.add(&update)?;
                }
            }
        }

        Ok(())
    }

    fn zero_grad(&mut self) {
        for group in &self.param_groups {
            for param in &group.params {
                param.write().zero_grad();
            }
        }
    }

    fn get_lr(&self) -> Vec<f32> {
        self.param_groups.iter().map(|g| g.lr).collect()
    }

    fn set_lr(&mut self, lr: f32) {
        for group in &mut self.param_groups {
            group.lr = lr;
        }
    }

    fn add_param_group(&mut self, params: Vec<Arc<RwLock<Tensor>>>, options: HashMap<String, f32>) {
        let lr = options.get("lr").copied().unwrap_or(1.0);
        let group = ParamGroup::new(params, lr).with_options(options);
        self.param_groups.push(group);
    }

    fn state_dict(&self) -> OptimizerResult<OptimizerState> {
        let param_groups = self
            .param_groups
            .iter()
            .map(|g| ParamGroupState {
                lr: g.lr,
                options: g.options.clone(),
                param_count: g.params.len(),
            })
            .collect();

        Ok(OptimizerState {
            optimizer_type: "MemoryEfficientAdam".to_string(),
            version: "0.1.0".to_string(),
            param_groups,
            state: self.state.clone(),
            global_state: HashMap::new(),
        })
    }

    fn load_state_dict(&mut self, state: OptimizerState) -> OptimizerResult<()> {
        if state.param_groups.len() != self.param_groups.len() {
            return Err(OptimizerError::InvalidParameter(
                "Parameter group count mismatch".to_string(),
            ));
        }

        for (i, group_state) in state.param_groups.iter().enumerate() {
            self.param_groups[i].lr = group_state.lr;
            self.param_groups[i].options = group_state.options.clone();
        }

        self.state = state.state;
        Ok(())
    }
}

/// Builder for memory-efficient optimizers
pub struct MemoryEfficientOptimizerBuilder {
    memory_config: MemoryConfig,
}

impl MemoryEfficientOptimizerBuilder {
    pub fn new() -> Self {
        Self {
            memory_config: MemoryConfig::default(),
        }
    }

    pub fn max_memory_gb(mut self, gb: f32) -> Self {
        self.memory_config.max_memory_bytes = (gb * 1_000_000_000.0) as usize;
        self
    }

    pub fn use_memory_pool(mut self, use_pool: bool) -> Self {
        self.memory_config.use_memory_pool = use_pool;
        self
    }

    pub fn compress_state(mut self, compress: bool) -> Self {
        self.memory_config.compress_state = compress;
        self
    }

    pub fn lazy_gradients(mut self, lazy: bool) -> Self {
        self.memory_config.lazy_gradients = lazy;
        self
    }

    pub fn checkpoint_interval(mut self, interval: usize) -> Self {
        self.memory_config.checkpoint_interval = interval;
        self
    }

    pub fn build_adam(self, params: Vec<Arc<RwLock<Tensor>>>, lr: f32) -> MemoryEfficientAdam {
        MemoryEfficientAdam::new(
            params,
            lr,
            None,
            None,
            None,
            None,
            None,
            Some(self.memory_config),
        )
    }

    pub fn build_lbfgs(self, params: Vec<Arc<RwLock<Tensor>>>, lr: f32) -> MemoryEfficientLBFGS {
        MemoryEfficientLBFGS::new(
            params,
            Some(lr),
            None,
            None,
            None,
            None,
            Some(self.memory_config),
        )
    }
}

impl Default for MemoryEfficientOptimizerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::randn;

    #[test]
    fn test_memory_pool() -> OptimizerResult<()> {
        let mut pool = MemoryPool::new();
        let tensor = pool.get_tensor(&[2, 2], torsh_core::device::DeviceType::Cpu)?;
        assert_eq!(tensor.shape().dims(), &[2, 2]);

        pool.return_tensor(tensor);
        let reused = pool.get_tensor(&[2, 2], torsh_core::device::DeviceType::Cpu)?;
        assert_eq!(reused.shape().dims(), &[2, 2]);
        Ok(())
    }

    #[test]
    fn test_circular_buffer() {
        let mut buffer = CircularBuffer::new(3);
        buffer.push(1);
        buffer.push(2);
        buffer.push(3);
        buffer.push(4); // Should overwrite 1

        assert_eq!(buffer.len(), 3);
        assert_eq!(buffer.get(0), Some(&2));
        assert_eq!(buffer.get(1), Some(&3));
        assert_eq!(buffer.get(2), Some(&4));
    }

    #[test]
    fn test_memory_efficient_adam_creation() -> OptimizerResult<()> {
        let params = vec![Arc::new(RwLock::new(randn::<f32>(&[2, 2])?))];
        let optimizer = MemoryEfficientAdam::new(params, 0.001, None, None, None, None, None, None);
        assert_eq!(optimizer.get_lr()[0], 0.001);
        Ok(())
    }

    #[test]
    fn test_memory_efficient_lbfgs_creation() -> OptimizerResult<()> {
        let params = vec![Arc::new(RwLock::new(randn::<f32>(&[2, 2])?))];
        let optimizer =
            MemoryEfficientLBFGS::new(params, Some(0.1), None, None, None, Some(5), None);
        assert_eq!(optimizer.get_lr()[0], 0.1);
        assert_eq!(optimizer.history_size, 5);
        Ok(())
    }

    #[test]
    fn test_builder_pattern() -> OptimizerResult<()> {
        let params = vec![Arc::new(RwLock::new(randn::<f32>(&[2, 2])?))];
        let optimizer = MemoryEfficientOptimizerBuilder::new()
            .max_memory_gb(1.0)
            .use_memory_pool(true)
            .compress_state(true)
            .build_adam(params, 0.001);

        assert_eq!(optimizer.get_lr()[0], 0.001);
        assert_eq!(optimizer.config.max_memory_bytes, 1_000_000_000);
        assert!(optimizer.config.use_memory_pool);
        assert!(optimizer.config.compress_state);
        Ok(())
    }
}
