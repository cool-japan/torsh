//! Model sharding and layer management for 3D parallelism
//!
//! This module manages model sharding across pipeline, tensor, and data
//! parallelism dimensions, including parameter distribution and gradient handling.

use crate::TorshResult;
use std::collections::HashMap;
use torsh_tensor::Tensor;

use super::config::ThreeDParallelismConfig;

// Helper function to create random tensors (temporary implementation)
fn randn(shape: &[usize]) -> TorshResult<Tensor<f32>> {
    use scirs2_core::random::Random;
    let total_elements: usize = shape.iter().product();
    let mut data = Vec::with_capacity(total_elements);

    let mut random = Random::seed(42); // Fixed seed for reproducibility

    for _ in 0..total_elements {
        data.push(random.gen_range(-1.0..1.0));
    }

    Ok(Tensor::from_data(
        data,
        shape.to_vec(),
        torsh_core::DeviceType::Cpu,
    )?)
}

/// Model shards distributed across 3D parallelism dimensions
#[derive(Debug)]
pub struct ModelShards {
    /// Pipeline stages, each containing multiple layers
    pub pipeline_stages: Vec<Vec<LayerShard>>,
    /// Total number of parameters in the model
    pub total_parameters: usize,
    /// Number of parameters per pipeline stage
    pub parameters_per_stage: Vec<usize>,
    /// Individual parameter shards indexed by name
    pub shards: HashMap<String, ModelShard>,
    /// Layer mapping for efficient access
    layer_mapping: HashMap<usize, (usize, usize)>, // layer_id -> (stage_idx, layer_in_stage_idx)
}

impl ModelShards {
    /// Create new model shards configuration
    pub fn new(config: &ThreeDParallelismConfig) -> TorshResult<Self> {
        let layers_per_stage = config.layers_per_stage();
        let mut pipeline_stages = Vec::new();
        let mut parameters_per_stage = Vec::new();
        let mut shards = HashMap::new();
        let mut layer_mapping = HashMap::new();
        let mut total_parameters = 0;

        // Create pipeline stages
        for stage_idx in 0..config.pp_size {
            let mut stage_layers = Vec::new();
            let mut stage_params = 0;

            for layer_in_stage in 0..layers_per_stage {
                let global_layer_id = stage_idx * layers_per_stage + layer_in_stage;

                // Create layer shard with tensor parallelism
                let layer_shard = LayerShard::new(global_layer_id, config.tp_size)?;
                let param_count = layer_shard.parameter_count();

                stage_params += param_count;
                total_parameters += param_count;

                // Create model shard entry for each parameter tensor
                let layer_name = format!("stage_{}_layer_{}", stage_idx, layer_in_stage);
                let model_shard = ModelShard {
                    parameters: vec![0.0; param_count],
                    gradients: Some(vec![0.0; param_count]),
                    shard_info: ShardInfo {
                        stage_id: stage_idx,
                        layer_id: global_layer_id,
                        tp_rank: 0, // Would be set based on actual TP rank
                        dp_rank: 0, // Would be set based on actual DP rank
                    },
                };

                shards.insert(layer_name, model_shard);
                layer_mapping.insert(global_layer_id, (stage_idx, layer_in_stage));
                stage_layers.push(layer_shard);
            }

            pipeline_stages.push(stage_layers);
            parameters_per_stage.push(stage_params);
        }

        Ok(Self {
            pipeline_stages,
            total_parameters,
            parameters_per_stage,
            shards,
            layer_mapping,
        })
    }

    /// Get layer shard by global layer ID
    pub fn get_layer_shard(&self, layer_id: usize) -> Option<&LayerShard> {
        if let Some(&(stage_idx, layer_in_stage)) = self.layer_mapping.get(&layer_id) {
            self.pipeline_stages
                .get(stage_idx)
                .and_then(|stage| stage.get(layer_in_stage))
        } else {
            None
        }
    }

    /// Get mutable layer shard by global layer ID
    pub fn get_layer_shard_mut(&mut self, layer_id: usize) -> Option<&mut LayerShard> {
        if let Some(&(stage_idx, layer_in_stage)) = self.layer_mapping.get(&layer_id) {
            self.pipeline_stages
                .get_mut(stage_idx)
                .and_then(|stage| stage.get_mut(layer_in_stage))
        } else {
            None
        }
    }

    /// Get all layer shards in a specific pipeline stage
    pub fn get_stage_layers(&self, stage_idx: usize) -> Option<&Vec<LayerShard>> {
        self.pipeline_stages.get(stage_idx)
    }

    /// Get all layer shards in a specific pipeline stage (mutable)
    pub fn get_stage_layers_mut(&mut self, stage_idx: usize) -> Option<&mut Vec<LayerShard>> {
        self.pipeline_stages.get_mut(stage_idx)
    }

    /// Get model shard by name
    pub fn get_model_shard(&self, name: &str) -> Option<&ModelShard> {
        self.shards.get(name)
    }

    /// Get mutable model shard by name
    pub fn get_model_shard_mut(&mut self, name: &str) -> Option<&mut ModelShard> {
        self.shards.get_mut(name)
    }

    /// Update gradients for a specific layer
    pub fn update_layer_gradients(
        &mut self,
        layer_id: usize,
        weight_grad: Option<Tensor<f32>>,
        bias_grad: Option<Tensor<f32>>,
    ) -> TorshResult<()> {
        if let Some(layer) = self.get_layer_shard_mut(layer_id) {
            layer.grad_weight = weight_grad;
            layer.grad_bias = bias_grad;
        }
        Ok(())
    }

    /// Zero all gradients
    pub fn zero_gradients(&mut self) -> TorshResult<()> {
        for stage_layers in &mut self.pipeline_stages {
            for layer in stage_layers {
                layer.zero_gradients()?;
            }
        }
        Ok(())
    }

    /// Get total memory usage in bytes
    pub fn memory_usage_bytes(&self) -> usize {
        let mut total_bytes = 0;

        for stage_layers in &self.pipeline_stages {
            for layer in stage_layers {
                total_bytes += layer.memory_usage_bytes();
            }
        }

        total_bytes
    }

    /// Create sharding plan for tensor parallelism
    pub fn create_tp_sharding_plan(&self, tp_size: usize) -> TensorParallelShardingPlan {
        let mut sharding_plan = TensorParallelShardingPlan::new(tp_size);

        for (stage_idx, stage_layers) in self.pipeline_stages.iter().enumerate() {
            for (layer_idx, layer) in stage_layers.iter().enumerate() {
                let layer_plan = self.create_layer_tp_plan(layer, tp_size);
                sharding_plan.add_layer_plan(stage_idx, layer_idx, layer_plan);
            }
        }

        sharding_plan
    }

    /// Create tensor parallel plan for a single layer
    fn create_layer_tp_plan(&self, layer: &LayerShard, tp_size: usize) -> LayerTensorParallelPlan {
        let binding = layer.weight.shape();
        let weight_dims = binding.dims();
        let shard_strategies = match layer.layer_type {
            LayerType::Linear => {
                // Linear layers: shard along output dimension
                vec![ShardStrategy::ColumnParallel]
            }
            LayerType::Attention => {
                // Attention: shard Q, K, V along head dimension
                vec![ShardStrategy::ColumnParallel, ShardStrategy::RowParallel]
            }
            LayerType::MLP => {
                // MLP: shard first linear along output, second along input
                vec![ShardStrategy::ColumnParallel, ShardStrategy::RowParallel]
            }
            LayerType::Embedding => {
                // Embedding: shard along vocabulary dimension
                vec![ShardStrategy::VocabParallel]
            }
        };

        LayerTensorParallelPlan {
            layer_id: layer.layer_id,
            layer_type: layer.layer_type,
            weight_shape: weight_dims.to_vec(),
            shard_strategies,
            communication_pattern: self.determine_communication_pattern(&layer.layer_type),
        }
    }

    /// Determine communication pattern for layer type
    fn determine_communication_pattern(&self, layer_type: &LayerType) -> CommunicationPattern {
        match layer_type {
            LayerType::Linear => CommunicationPattern::AllReduce,
            LayerType::Attention => CommunicationPattern::AllGatherThenReduceScatter,
            LayerType::MLP => CommunicationPattern::AllGatherThenReduceScatter,
            LayerType::Embedding => CommunicationPattern::AllReduce,
        }
    }

    /// Apply weight updates from optimizer
    pub fn apply_weight_updates(
        &mut self,
        updates: &HashMap<String, Tensor<f32>>,
    ) -> TorshResult<()> {
        for (layer_name, update) in updates {
            if let Some(shard) = self.shards.get_mut(layer_name) {
                // Apply update to parameters (simplified)
                let update_data = update.data()?;
                for (i, &update_val) in update_data.iter().enumerate() {
                    if i < shard.parameters.len() {
                        shard.parameters[i] -= update_val; // SGD-style update
                    }
                }
            }
        }
        Ok(())
    }
}

/// Individual layer shard containing parameters and gradients
#[derive(Debug)]
pub struct LayerShard {
    /// Unique layer identifier
    pub layer_id: usize,
    /// Type of the layer
    pub layer_type: LayerType,
    /// Weight tensor (sharded across TP dimension)
    pub weight: Tensor<f32>,
    /// Bias tensor (optional)
    pub bias: Option<Tensor<f32>>,
    /// Weight gradient tensor
    pub grad_weight: Option<Tensor<f32>>,
    /// Bias gradient tensor
    pub grad_bias: Option<Tensor<f32>>,
    /// Additional projection weight for MLP layers
    pub down_projection_weight: Option<Tensor<f32>>,
    /// Gradient for down projection weight
    pub grad_down_projection: Option<Tensor<f32>>,
}

impl LayerShard {
    /// Create new layer shard
    pub fn new(layer_id: usize, tp_size: usize) -> TorshResult<Self> {
        // Determine layer type based on layer ID (simplified mapping)
        let layer_type = match layer_id % 4 {
            0 => LayerType::Embedding,
            1 => LayerType::Attention,
            2 => LayerType::MLP,
            _ => LayerType::Linear,
        };

        // Create weight tensor sharded across TP dimension
        let hidden_size = 512;
        let shard_size = hidden_size / tp_size;

        let weight = match layer_type {
            LayerType::Linear | LayerType::Embedding => randn(&[hidden_size, shard_size])?,
            LayerType::Attention => {
                // Attention layer has 3x hidden size for Q, K, V
                randn(&[hidden_size, 3 * shard_size])?
            }
            LayerType::MLP => {
                // MLP up-projection
                randn(&[hidden_size, 4 * shard_size])?
            }
        };

        let bias = Some(Tensor::zeros(&[shard_size], torsh_core::DeviceType::Cpu)?);

        // For MLP layers, create down projection weight
        let down_projection_weight = if matches!(layer_type, LayerType::MLP) {
            Some(randn(&[4 * shard_size, hidden_size])?)
        } else {
            None
        };

        Ok(Self {
            layer_id,
            layer_type,
            weight,
            bias,
            grad_weight: None,
            grad_bias: None,
            down_projection_weight,
            grad_down_projection: None,
        })
    }

    /// Get total parameter count for this layer shard
    pub fn parameter_count(&self) -> usize {
        let weight_params = self.weight.numel();
        let bias_params = self.bias.as_ref().map(|b| b.numel()).unwrap_or(0);
        let down_proj_params = self
            .down_projection_weight
            .as_ref()
            .map(|w| w.numel())
            .unwrap_or(0);
        weight_params + bias_params + down_proj_params
    }

    /// Get memory usage in bytes
    pub fn memory_usage_bytes(&self) -> usize {
        let mut bytes = self.weight.numel() * std::mem::size_of::<f32>();

        if let Some(ref bias) = self.bias {
            bytes += bias.numel() * std::mem::size_of::<f32>();
        }

        if let Some(ref down_proj) = self.down_projection_weight {
            bytes += down_proj.numel() * std::mem::size_of::<f32>();
        }

        // Add gradient memory if allocated
        if self.grad_weight.is_some() {
            bytes += self.weight.numel() * std::mem::size_of::<f32>();
        }

        if self.grad_bias.is_some() {
            bytes +=
                self.bias.as_ref().map(|b| b.numel()).unwrap_or(0) * std::mem::size_of::<f32>();
        }

        if self.grad_down_projection.is_some() {
            bytes += self
                .down_projection_weight
                .as_ref()
                .map(|w| w.numel())
                .unwrap_or(0)
                * std::mem::size_of::<f32>();
        }

        bytes
    }

    /// Zero gradients for this layer
    pub fn zero_gradients(&mut self) -> TorshResult<()> {
        if let Some(ref mut grad_weight) = self.grad_weight {
            // Would zero the gradient tensor
            // grad_weight.zero_()?;
        }

        if let Some(ref mut grad_bias) = self.grad_bias {
            // Would zero the gradient tensor
            // grad_bias.zero_()?;
        }

        if let Some(ref mut grad_down_proj) = self.grad_down_projection {
            // Would zero the gradient tensor
            // grad_down_proj.zero_()?;
        }

        Ok(())
    }

    /// Initialize gradients with correct shapes
    pub fn init_gradients(&mut self) -> TorshResult<()> {
        self.grad_weight = Some(Tensor::zeros(
            self.weight.shape().dims(),
            self.weight.device(),
        )?);

        if let Some(ref bias) = self.bias {
            self.grad_bias = Some(Tensor::zeros(bias.shape().dims(), bias.device())?);
        }

        if let Some(ref down_proj) = self.down_projection_weight {
            self.grad_down_projection =
                Some(Tensor::zeros(down_proj.shape().dims(), down_proj.device())?);
        }

        Ok(())
    }
}

/// Layer types supported in the model
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LayerType {
    Linear,
    Attention,
    MLP,
    Embedding,
}

/// Individual model shard containing flattened parameters
#[derive(Debug)]
pub struct ModelShard {
    /// Flattened parameter vector
    pub parameters: Vec<f32>,
    /// Flattened gradient vector
    pub gradients: Option<Vec<f32>>,
    /// Shard metadata
    pub shard_info: ShardInfo,
}

/// Metadata about a model shard
#[derive(Debug, Clone)]
pub struct ShardInfo {
    /// Pipeline stage ID
    pub stage_id: usize,
    /// Layer ID within the stage
    pub layer_id: usize,
    /// Tensor parallel rank
    pub tp_rank: usize,
    /// Data parallel rank
    pub dp_rank: usize,
}

/// Tensor parallel sharding plan
#[derive(Debug)]
pub struct TensorParallelShardingPlan {
    tp_size: usize,
    layer_plans: HashMap<(usize, usize), LayerTensorParallelPlan>, // (stage_idx, layer_idx) -> plan
}

impl TensorParallelShardingPlan {
    fn new(tp_size: usize) -> Self {
        Self {
            tp_size,
            layer_plans: HashMap::new(),
        }
    }

    fn add_layer_plan(
        &mut self,
        stage_idx: usize,
        layer_idx: usize,
        plan: LayerTensorParallelPlan,
    ) {
        self.layer_plans.insert((stage_idx, layer_idx), plan);
    }

    /// Get sharding plan for a specific layer
    pub fn get_layer_plan(
        &self,
        stage_idx: usize,
        layer_idx: usize,
    ) -> Option<&LayerTensorParallelPlan> {
        self.layer_plans.get(&(stage_idx, layer_idx))
    }
}

/// Tensor parallel plan for a single layer
#[derive(Debug, Clone)]
pub struct LayerTensorParallelPlan {
    pub layer_id: usize,
    pub layer_type: LayerType,
    pub weight_shape: Vec<usize>,
    pub shard_strategies: Vec<ShardStrategy>,
    pub communication_pattern: CommunicationPattern,
}

/// Sharding strategies for tensor parallelism
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ShardStrategy {
    /// Shard along output dimension (columns)
    ColumnParallel,
    /// Shard along input dimension (rows)
    RowParallel,
    /// Shard vocabulary dimension
    VocabParallel,
    /// No sharding (replicated)
    Replicated,
}

/// Communication patterns for tensor parallel operations
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CommunicationPattern {
    /// Standard all-reduce
    AllReduce,
    /// All-gather followed by reduce-scatter
    AllGatherThenReduceScatter,
    /// Reduce-scatter followed by all-gather
    ReduceScatterThenAllGather,
    /// No communication required
    None,
}
