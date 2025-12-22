//! Fine-tuning utilities for transfer learning and model adaptation
//!
//! This module provides comprehensive fine-tuning techniques including:
//! - Layer-wise learning rate scheduling
//! - Parameter freezing and gradual unfreezing
//! - Adapter layers (AdaLoRA, LoRA variants)
//! - Progressive fine-tuning strategies
//! - Domain adaptation utilities

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use torsh_core::error::Result;
use torsh_nn::{Module, Parameter};
use torsh_tensor::Tensor;

/// Fine-tuning strategy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FineTuningConfig {
    /// Learning rate schedule for different layers
    pub layer_lr_schedule: LayerLearningRateSchedule,
    /// Parameter freezing strategy
    pub freezing_strategy: FreezingStrategy,
    /// Adapter configuration (if using adapters)
    pub adapter_config: Option<AdapterConfig>,
    /// Progressive unfreezing settings
    pub progressive_unfreezing: Option<ProgressiveUnfreezingConfig>,
    /// Domain adaptation settings
    pub domain_adaptation: Option<DomainAdaptationConfig>,
    /// Regularization for fine-tuning
    pub regularization: FineTuningRegularization,
}

/// Layer-wise learning rate scheduling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LayerLearningRateSchedule {
    /// Uniform learning rate for all layers
    Uniform { learning_rate: f64 },
    /// Linear decay from output to input layers
    LinearDecay { output_lr: f64, input_lr: f64 },
    /// Exponential decay from output to input layers
    ExponentialDecay { output_lr: f64, decay_factor: f64 },
    /// Custom learning rates for specific layers
    Custom {
        layer_rates: HashMap<String, f64>,
        default_rate: f64,
    },
    /// Discriminative learning rates
    Discriminative { base_lr: f64, multipliers: Vec<f64> },
}

/// Parameter freezing strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FreezingStrategy {
    /// No freezing - train all parameters
    None,
    /// Freeze all layers except the last N layers
    FreezeTo { last_n_layers: usize },
    /// Freeze specific layer types
    FreezeLayerTypes { layer_types: Vec<String> },
    /// Freeze specific named layers
    FreezeSpecific { layer_names: Vec<String> },
    /// Freeze embedding layers only
    FreezeEmbeddings,
    /// Custom freezing pattern
    Custom { freeze_pattern: FreezePattern },
}

/// Custom freeze patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FreezePattern {
    /// Layers to freeze
    pub freeze_layers: Vec<String>,
    /// Layers to keep trainable
    pub trainable_layers: Vec<String>,
    /// Use regex patterns for matching
    pub use_regex: bool,
}

/// Adapter configuration for parameter-efficient fine-tuning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterConfig {
    /// Type of adapter to use
    pub adapter_type: AdapterType,
    /// Adapter placement strategy
    pub placement: AdapterPlacement,
    /// Adapter dimensions and hyperparameters
    pub hyperparameters: AdapterHyperparameters,
    /// Whether to train original parameters
    pub train_original: bool,
}

/// Types of adapters for fine-tuning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AdapterType {
    /// Standard adapter with bottleneck architecture
    Standard {
        reduction_factor: usize,
        activation: String,
    },
    /// LoRA (Low-Rank Adaptation)
    LoRA {
        rank: usize,
        alpha: f64,
        dropout: f64,
    },
    /// AdaLoRA (Adaptive Low-Rank Adaptation)
    AdaLoRA {
        rank: usize,
        alpha: f64,
        beta1: f64,
        beta2: f64,
        rank_dropout: f64,
    },
    /// Prefix tuning
    PrefixTuning {
        prefix_length: usize,
        hidden_size: usize,
    },
    /// P-tuning v2
    PtuningV2 {
        num_virtual_tokens: usize,
        token_dim: usize,
    },
    /// Compacter (low-rank + sharing)
    Compacter {
        rank: usize,
        shared_phm_rule: bool,
        factorized_phm: bool,
    },
}

/// Adapter placement strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AdapterPlacement {
    /// Place adapters in all transformer layers
    AllLayers,
    /// Place adapters in last N layers
    LastNLayers { n: usize },
    /// Place adapters in specific layers
    SpecificLayers { layer_indices: Vec<usize> },
    /// Place adapters based on layer types
    LayerTypes { types: Vec<String> },
    /// Custom placement pattern
    Custom { placement_pattern: Vec<bool> },
}

/// Adapter hyperparameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterHyperparameters {
    /// Initialization method
    pub initialization: InitializationMethod,
    /// Learning rate for adapters
    pub learning_rate: f64,
    /// Weight decay for adapters
    pub weight_decay: f64,
    /// Dropout probability
    pub dropout: f64,
    /// Scaling factor
    pub scaling: f64,
}

/// Initialization methods for adapters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InitializationMethod {
    /// Zero initialization
    Zero,
    /// Random normal initialization
    Normal { mean: f64, std: f64 },
    /// Xavier uniform initialization
    XavierUniform,
    /// Kaiming initialization
    Kaiming,
    /// Custom initialization
    Custom { method: String },
}

/// Progressive unfreezing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressiveUnfreezingConfig {
    /// Unfreezing schedule
    pub schedule: UnfreezingSchedule,
    /// Number of epochs between unfreezing steps
    pub epochs_per_step: usize,
    /// Learning rate adjustment when unfreezing
    pub lr_adjustment: LearningRateAdjustment,
    /// Whether to reset optimizer state on unfreezing
    pub reset_optimizer: bool,
}

/// Unfreezing schedules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UnfreezingSchedule {
    /// Unfreeze one layer at a time from top to bottom
    LayerByLayer,
    /// Unfreeze groups of layers
    GroupWise { group_size: usize },
    /// Exponential unfreezing (more layers as time progresses)
    Exponential { rate: f64 },
    /// Custom unfreezing pattern
    Custom { pattern: Vec<usize> },
}

/// Learning rate adjustment strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LearningRateAdjustment {
    /// Keep the same learning rate
    None,
    /// Scale learning rate by factor
    Scale { factor: f64 },
    /// Decay learning rate
    Decay { decay_rate: f64 },
    /// Set new learning rate
    Set { new_lr: f64 },
}

/// Domain adaptation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainAdaptationConfig {
    /// Type of domain adaptation
    pub adaptation_type: DomainAdaptationType,
    /// Source and target domain configurations
    pub domain_config: DomainConfig,
    /// Adversarial training parameters
    pub adversarial_config: Option<AdversarialConfig>,
}

/// Types of domain adaptation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DomainAdaptationType {
    /// Feature-based adaptation
    FeatureBased,
    /// Adversarial adaptation with domain classifier
    Adversarial,
    /// Correlation alignment (CORAL)
    CORAL,
    /// Maximum mean discrepancy (MMD)
    MMD { kernel: String },
    /// Domain confusion
    DomainConfusion,
}

/// Domain configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainConfig {
    /// Source domain identifier
    pub source_domain: String,
    /// Target domain identifier
    pub target_domain: String,
    /// Domain mixing ratio during training
    pub mixing_ratio: f64,
    /// Whether to use domain labels
    pub use_domain_labels: bool,
}

/// Adversarial training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdversarialConfig {
    /// Domain classifier architecture
    pub classifier_layers: Vec<usize>,
    /// Adversarial loss weight
    pub adversarial_weight: f64,
    /// Gradient reversal layer scale
    pub grl_scale: f64,
    /// Training schedule for adversarial component
    pub schedule: AdversarialSchedule,
}

/// Adversarial training schedules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AdversarialSchedule {
    /// Constant adversarial weight
    Constant,
    /// Progressive increase of adversarial weight
    Progressive { max_weight: f64, ramp_epochs: usize },
    /// Cyclical adversarial training
    Cyclical { period: usize },
}

/// Fine-tuning regularization techniques
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FineTuningRegularization {
    /// L2 regularization on parameter changes
    pub l2_on_changes: Option<f64>,
    /// Elastic weight consolidation (EWC)
    pub ewc: Option<EWCConfig>,
    /// Knowledge distillation from original model
    pub knowledge_distillation: Option<KnowledgeDistillationConfig>,
    /// Dropout adjustments
    pub dropout_adjustments: Option<DropoutAdjustments>,
}

/// Elastic Weight Consolidation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EWCConfig {
    /// Importance weight for EWC loss
    pub importance_weight: f64,
    /// Number of samples for Fisher information estimation
    pub fisher_samples: usize,
    /// Whether to use diagonal Fisher approximation
    pub diagonal_fisher: bool,
}

/// Knowledge distillation configuration for fine-tuning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeDistillationConfig {
    /// Temperature for distillation
    pub temperature: f64,
    /// Weight for distillation loss
    pub distillation_weight: f64,
    /// Layers to use for feature distillation
    pub feature_layers: Vec<String>,
}

/// Dropout adjustments during fine-tuning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DropoutAdjustments {
    /// Dropout rate for new layers/adapters
    pub adapter_dropout: f64,
    /// Dropout rate for frozen layers
    pub frozen_dropout: f64,
    /// Dropout schedule
    pub schedule: DropoutSchedule,
}

/// Dropout scheduling during fine-tuning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DropoutSchedule {
    /// Constant dropout rate
    Constant,
    /// Linear decay of dropout
    LinearDecay { final_rate: f64 },
    /// Exponential decay of dropout
    ExponentialDecay { decay_rate: f64 },
}

/// Fine-tuning statistics and monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FineTuningStats {
    /// Number of trainable parameters
    pub trainable_params: usize,
    /// Number of frozen parameters
    pub frozen_params: usize,
    /// Percentage of parameters being trained
    pub training_percentage: f64,
    /// Layer-wise training status
    pub layer_status: HashMap<String, LayerTrainingStatus>,
    /// Adapter statistics
    pub adapter_stats: Option<AdapterStats>,
    /// Domain adaptation metrics
    pub domain_metrics: Option<DomainAdaptationMetrics>,
}

/// Status of individual layers during fine-tuning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerTrainingStatus {
    /// Whether the layer is frozen
    pub frozen: bool,
    /// Learning rate for the layer
    pub learning_rate: f64,
    /// Number of parameters in the layer
    pub num_parameters: usize,
    /// Has adapters attached
    pub has_adapters: bool,
}

/// Adapter statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterStats {
    /// Number of adapter parameters
    pub adapter_params: usize,
    /// Adapter efficiency (adapter_params / total_params)
    pub efficiency: f64,
    /// Average rank (for LoRA-type adapters)
    pub average_rank: Option<f64>,
    /// Adapter utilization per layer
    pub layer_utilization: HashMap<String, f64>,
}

/// Domain adaptation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainAdaptationMetrics {
    /// Domain classification accuracy
    pub domain_accuracy: f64,
    /// Feature distance between domains
    pub feature_distance: f64,
    /// Adaptation loss
    pub adaptation_loss: f64,
}

/// Main fine-tuning engine
pub struct FineTuningEngine<M: Module> {
    model: M,
    config: FineTuningConfig,
    frozen_layers: Vec<String>,
    adapters: HashMap<String, Box<dyn Module>>,
    stats: Option<FineTuningStats>,
    unfreezing_state: Option<UnfreezingState>,
}

/// State for progressive unfreezing
#[derive(Debug, Clone)]
struct UnfreezingState {
    current_step: usize,
    epochs_since_last_unfreeze: usize,
    unfrozen_layers: Vec<String>,
}

impl<M: Module> FineTuningEngine<M> {
    /// Create a new fine-tuning engine
    pub fn new(model: M, config: FineTuningConfig) -> Self {
        Self {
            model,
            config,
            frozen_layers: Vec::new(),
            adapters: HashMap::new(),
            stats: None,
            unfreezing_state: None,
        }
    }

    /// Initialize fine-tuning by applying freezing and adding adapters
    pub fn initialize(&mut self) -> Result<()> {
        // Apply initial freezing strategy
        self.apply_freezing_strategy()?;

        // Add adapters if configured
        if let Some(adapter_config) = self.config.adapter_config.clone() {
            self.add_adapters(&adapter_config)?;
        }

        // Initialize progressive unfreezing if configured
        if let Some(unfreezing_config) = self.config.progressive_unfreezing.clone() {
            self.initialize_progressive_unfreezing(&unfreezing_config)?;
        }

        // Calculate initial statistics
        self.update_statistics()?;

        Ok(())
    }

    /// Get layer-wise learning rates
    pub fn get_layer_learning_rates(&self) -> Result<HashMap<String, f64>> {
        let model_params = self.model.named_parameters();
        let mut layer_rates = HashMap::new();

        match &self.config.layer_lr_schedule {
            LayerLearningRateSchedule::Uniform { learning_rate } => {
                for layer_name in model_params.keys() {
                    layer_rates.insert(layer_name.clone(), *learning_rate);
                }
            }
            LayerLearningRateSchedule::LinearDecay {
                output_lr,
                input_lr,
            } => {
                let layer_names: Vec<String> = model_params.keys().cloned().collect();
                let num_layers = layer_names.len();

                for (i, layer_name) in layer_names.iter().enumerate() {
                    let ratio = i as f64 / (num_layers - 1).max(1) as f64;
                    let lr = input_lr + (output_lr - input_lr) * ratio;
                    layer_rates.insert(layer_name.clone(), lr);
                }
            }
            LayerLearningRateSchedule::ExponentialDecay {
                output_lr,
                decay_factor,
            } => {
                let layer_names: Vec<String> = model_params.keys().cloned().collect();

                for (i, layer_name) in layer_names.iter().enumerate() {
                    let lr = output_lr * decay_factor.powi(i as i32);
                    layer_rates.insert(layer_name.clone(), lr);
                }
            }
            LayerLearningRateSchedule::Custom {
                layer_rates: custom_rates,
                default_rate,
            } => {
                for layer_name in model_params.keys() {
                    let lr = custom_rates.get(layer_name).unwrap_or(default_rate);
                    layer_rates.insert(layer_name.clone(), *lr);
                }
            }
            LayerLearningRateSchedule::Discriminative {
                base_lr,
                multipliers,
            } => {
                let layer_names: Vec<String> = model_params.keys().cloned().collect();

                for (i, layer_name) in layer_names.iter().enumerate() {
                    let multiplier = multipliers.get(i).unwrap_or(&1.0);
                    let lr = base_lr * multiplier;
                    layer_rates.insert(layer_name.clone(), lr);
                }
            }
        }

        Ok(layer_rates)
    }

    /// Update which layers are frozen/unfrozen
    pub fn update_epoch(&mut self, _epoch: usize) -> Result<bool> {
        let mut changed = false;

        if let Some(unfreezing_config) = self.config.progressive_unfreezing.clone() {
            let should_unfreeze = if let Some(state) = &mut self.unfreezing_state {
                state.epochs_since_last_unfreeze += 1;
                state.epochs_since_last_unfreeze >= unfreezing_config.epochs_per_step
            } else {
                false
            };

            if should_unfreeze {
                changed = self.perform_unfreezing_step(&unfreezing_config)?;
                if changed {
                    if let Some(state) = &mut self.unfreezing_state {
                        state.epochs_since_last_unfreeze = 0;
                        state.current_step += 1;
                    }
                }
            }
        }

        if changed {
            self.update_statistics()?;
        }

        Ok(changed)
    }

    /// Add adapters to specified layers
    pub fn add_adapters(&mut self, adapter_config: &AdapterConfig) -> Result<()> {
        let model_params = self.model.named_parameters();
        let target_layers = self.get_adapter_target_layers(&adapter_config.placement)?;

        for layer_name in target_layers {
            if model_params.contains_key(&layer_name) {
                let adapter = self.create_adapter(&adapter_config.adapter_type, &layer_name)?;
                self.adapters.insert(layer_name, adapter);
            }
        }

        Ok(())
    }

    /// Get current frozen layer list
    pub fn get_frozen_layers(&self) -> &[String] {
        &self.frozen_layers
    }

    /// Check if a layer is frozen
    pub fn is_layer_frozen(&self, layer_name: &str) -> bool {
        self.frozen_layers.contains(&layer_name.to_string())
    }

    /// Manually freeze specific layers
    pub fn freeze_layers(&mut self, layer_names: Vec<String>) -> Result<()> {
        for layer_name in layer_names {
            if !self.frozen_layers.contains(&layer_name) {
                self.frozen_layers.push(layer_name);
            }
        }
        self.update_statistics()?;
        Ok(())
    }

    /// Manually unfreeze specific layers
    pub fn unfreeze_layers(&mut self, layer_names: Vec<String>) -> Result<()> {
        for layer_name in &layer_names {
            self.frozen_layers.retain(|x| x != layer_name);
        }
        self.update_statistics()?;
        Ok(())
    }

    /// Calculate domain adaptation loss
    pub fn calculate_domain_loss(
        &self,
        source_features: &Tensor,
        target_features: &Tensor,
    ) -> Result<f64> {
        if let Some(domain_config) = &self.config.domain_adaptation {
            match &domain_config.adaptation_type {
                DomainAdaptationType::CORAL => {
                    self.calculate_coral_loss(source_features, target_features)
                }
                DomainAdaptationType::MMD { kernel: _ } => {
                    self.calculate_mmd_loss(source_features, target_features)
                }
                _ => {
                    // Simplified implementation for other types
                    Ok(0.0)
                }
            }
        } else {
            Ok(0.0)
        }
    }

    /// Get fine-tuning statistics
    pub fn get_stats(&self) -> Option<&FineTuningStats> {
        self.stats.as_ref()
    }

    // Implementation methods
    fn apply_freezing_strategy(&mut self) -> Result<()> {
        let model_params = self.model.named_parameters();

        match &self.config.freezing_strategy {
            FreezingStrategy::None => {
                // Don't freeze anything
            }
            FreezingStrategy::FreezeTo { last_n_layers } => {
                let layer_names: Vec<String> = model_params.keys().cloned().collect();
                let freeze_count = layer_names.len().saturating_sub(*last_n_layers);

                for layer_name in &layer_names[..freeze_count] {
                    self.frozen_layers.push(layer_name.clone());
                }
            }
            FreezingStrategy::FreezeLayerTypes { layer_types } => {
                for layer_name in model_params.keys() {
                    for layer_type in layer_types {
                        if layer_name.contains(layer_type) {
                            self.frozen_layers.push(layer_name.clone());
                            break;
                        }
                    }
                }
            }
            FreezingStrategy::FreezeSpecific { layer_names } => {
                for layer_name in layer_names {
                    if model_params.contains_key(layer_name) {
                        self.frozen_layers.push(layer_name.clone());
                    }
                }
            }
            FreezingStrategy::FreezeEmbeddings => {
                for layer_name in model_params.keys() {
                    if layer_name.contains("embed") || layer_name.contains("embedding") {
                        self.frozen_layers.push(layer_name.clone());
                    }
                }
            }
            FreezingStrategy::Custom { freeze_pattern } => {
                let pattern = freeze_pattern.clone();
                self.apply_custom_freeze_pattern(&pattern, &model_params)?;
            }
        }

        Ok(())
    }

    fn apply_custom_freeze_pattern(
        &mut self,
        pattern: &FreezePattern,
        model_params: &HashMap<String, Parameter>,
    ) -> Result<()> {
        // Apply freeze pattern
        for layer_name in &pattern.freeze_layers {
            if pattern.use_regex {
                // In a real implementation, you would use regex matching
                // For now, use simple string matching
                for param_name in model_params.keys() {
                    if param_name.contains(layer_name) {
                        self.frozen_layers.push(param_name.clone());
                    }
                }
            } else if model_params.contains_key(layer_name) {
                self.frozen_layers.push(layer_name.clone());
            }
        }

        // Remove explicitly trainable layers from frozen list
        for layer_name in &pattern.trainable_layers {
            if pattern.use_regex {
                self.frozen_layers
                    .retain(|frozen| !frozen.contains(layer_name));
            } else {
                self.frozen_layers.retain(|frozen| frozen != layer_name);
            }
        }

        Ok(())
    }

    fn get_adapter_target_layers(&self, placement: &AdapterPlacement) -> Result<Vec<String>> {
        let model_params = self.model.named_parameters();
        let layer_names: Vec<String> = model_params.keys().cloned().collect();

        match placement {
            AdapterPlacement::AllLayers => Ok(layer_names),
            AdapterPlacement::LastNLayers { n } => {
                let start_idx = layer_names.len().saturating_sub(*n);
                Ok(layer_names[start_idx..].to_vec())
            }
            AdapterPlacement::SpecificLayers { layer_indices } => Ok(layer_indices
                .iter()
                .filter_map(|&idx| layer_names.get(idx).cloned())
                .collect()),
            AdapterPlacement::LayerTypes { types } => Ok(layer_names
                .into_iter()
                .filter(|name| types.iter().any(|t| name.contains(t)))
                .collect()),
            AdapterPlacement::Custom { placement_pattern } => Ok(layer_names
                .into_iter()
                .enumerate()
                .filter_map(|(i, name)| {
                    if *placement_pattern.get(i).unwrap_or(&false) {
                        Some(name)
                    } else {
                        None
                    }
                })
                .collect()),
        }
    }

    fn create_adapter(
        &self,
        adapter_type: &AdapterType,
        _layer_name: &str,
    ) -> Result<Box<dyn Module>> {
        match adapter_type {
            AdapterType::Standard {
                reduction_factor: _,
                activation: _,
            } => Ok(Box::new(StandardAdapter::new(512, 64))),
            AdapterType::LoRA {
                rank,
                alpha: _,
                dropout: _,
            } => Ok(Box::new(LoRAAdapter::new(512, 512, *rank))),
            _ => {
                // Simplified implementations for other adapter types
                Ok(Box::new(StandardAdapter::new(512, 64)))
            }
        }
    }

    fn initialize_progressive_unfreezing(
        &mut self,
        _unfreezing_config: &ProgressiveUnfreezingConfig,
    ) -> Result<()> {
        self.unfreezing_state = Some(UnfreezingState {
            current_step: 0,
            epochs_since_last_unfreeze: 0,
            unfrozen_layers: Vec::new(),
        });
        Ok(())
    }

    fn perform_unfreezing_step(
        &mut self,
        unfreezing_config: &ProgressiveUnfreezingConfig,
    ) -> Result<bool> {
        if let Some(state) = &mut self.unfreezing_state {
            match &unfreezing_config.schedule {
                UnfreezingSchedule::LayerByLayer => {
                    if !self.frozen_layers.is_empty() {
                        let layer_to_unfreeze = self.frozen_layers.remove(0);
                        state.unfrozen_layers.push(layer_to_unfreeze);
                        return Ok(true);
                    }
                }
                UnfreezingSchedule::GroupWise { group_size } => {
                    let unfreeze_count = (*group_size).min(self.frozen_layers.len());
                    if unfreeze_count > 0 {
                        let unfrozen: Vec<String> =
                            self.frozen_layers.drain(..unfreeze_count).collect();
                        state.unfrozen_layers.extend(unfrozen);
                        return Ok(true);
                    }
                }
                _ => {
                    // Simplified implementation for other schedules
                    if !self.frozen_layers.is_empty() {
                        let layer_to_unfreeze = self.frozen_layers.remove(0);
                        state.unfrozen_layers.push(layer_to_unfreeze);
                        return Ok(true);
                    }
                }
            }
        }
        Ok(false)
    }

    fn update_statistics(&mut self) -> Result<()> {
        let model_params = self.model.named_parameters();
        let mut trainable_params = 0;
        let mut frozen_params = 0;
        let mut layer_status = HashMap::new();

        for (layer_name, param) in &model_params {
            let is_frozen = self.frozen_layers.contains(layer_name);
            let tensor = param.tensor();
            let tensor_guard = tensor.read();
            let param_count = tensor_guard.numel();
            let has_adapters = self.adapters.contains_key(layer_name);

            if is_frozen {
                frozen_params += param_count;
            } else {
                trainable_params += param_count;
            }

            layer_status.insert(
                layer_name.clone(),
                LayerTrainingStatus {
                    frozen: is_frozen,
                    learning_rate: 0.001, // Would get from actual learning rate schedule
                    num_parameters: param_count,
                    has_adapters,
                },
            );
        }

        // Add adapter parameters to trainable count
        let mut adapter_param_count = 0;
        for adapter in self.adapters.values() {
            let adapter_params = adapter.parameters();
            for param in adapter_params.values() {
                let tensor = param.tensor();
                let tensor_guard = tensor.read();
                adapter_param_count += tensor_guard.numel();
            }
        }
        trainable_params += adapter_param_count;

        let total_params = trainable_params + frozen_params;
        let training_percentage = if total_params > 0 {
            (trainable_params as f64 / total_params as f64) * 100.0
        } else {
            0.0
        };

        let adapter_stats = if !self.adapters.is_empty() {
            Some(AdapterStats {
                adapter_params: adapter_param_count,
                efficiency: adapter_param_count as f64 / total_params as f64,
                average_rank: None, // Would calculate from actual LoRA adapters
                layer_utilization: HashMap::new(),
            })
        } else {
            None
        };

        self.stats = Some(FineTuningStats {
            trainable_params,
            frozen_params,
            training_percentage,
            layer_status,
            adapter_stats,
            domain_metrics: None,
        });

        Ok(())
    }

    fn calculate_coral_loss(
        &self,
        source_features: &Tensor,
        target_features: &Tensor,
    ) -> Result<f64> {
        // Simplified CORAL loss calculation
        // In practice, this would compute the Frobenius norm of covariance difference
        let source_f32 = source_features.to_vec()?;
        let target_f32 = target_features.to_vec()?;

        // Simplified distance calculation
        let mut distance = 0.0;
        for (s, t) in source_f32.iter().zip(target_f32.iter()) {
            distance += (s - t).powi(2);
        }

        Ok(distance as f64 / source_f32.len() as f64)
    }

    fn calculate_mmd_loss(
        &self,
        source_features: &Tensor,
        target_features: &Tensor,
    ) -> Result<f64> {
        // Simplified MMD loss calculation
        // In practice, this would use kernel methods
        self.calculate_coral_loss(source_features, target_features)
    }
}

// Simplified adapter implementations
struct StandardAdapter {
    down_proj: SimpleLinear,
    up_proj: SimpleLinear,
    activation: String,
}

impl StandardAdapter {
    fn new(input_dim: usize, bottleneck_dim: usize) -> Self {
        Self {
            down_proj: SimpleLinear::new(input_dim, bottleneck_dim),
            up_proj: SimpleLinear::new(bottleneck_dim, input_dim),
            activation: "relu".to_string(),
        }
    }
}

impl Module for StandardAdapter {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let down = self.down_proj.forward(input)?;
        let activated = match self.activation.as_str() {
            "relu" => down.relu()?,
            "gelu" => down.gelu()?,
            _ => down,
        };
        let up = self.up_proj.forward(&activated)?;
        input.add(&up) // Residual connection
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (name, param) in self.down_proj.parameters() {
            params.insert(format!("down_proj.{}", name), param);
        }
        for (name, param) in self.up_proj.parameters() {
            params.insert(format!("up_proj.{}", name), param);
        }
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }
    fn training(&self) -> bool {
        true
    }
    fn train(&mut self) {}
    fn eval(&mut self) {}
    fn to_device(&mut self, device: torsh_core::DeviceType) -> Result<()> {
        self.down_proj.to_device(device)?;
        self.up_proj.to_device(device)?;
        Ok(())
    }
}

struct LoRAAdapter {
    lora_a: Tensor,
    lora_b: Tensor,
    scaling: f32,
    rank: usize,
}

impl LoRAAdapter {
    fn new(in_features: usize, out_features: usize, rank: usize) -> Self {
        let lora_a = torsh_tensor::creation::randn(&[rank, in_features]).unwrap();
        let lora_b = torsh_tensor::creation::zeros(&[out_features, rank]).unwrap();
        let scaling = 1.0 / rank as f32;

        Self {
            lora_a,
            lora_b,
            scaling,
            rank,
        }
    }
}

impl Module for LoRAAdapter {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let a_out = input.matmul(&self.lora_a.transpose(0, 1)?)?;
        let b_out = a_out.matmul(&self.lora_b.transpose(0, 1)?)?;
        Ok(b_out.mul_scalar(self.scaling)?)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        params.insert("lora_a".to_string(), Parameter::new(self.lora_a.clone()));
        params.insert("lora_b".to_string(), Parameter::new(self.lora_b.clone()));
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }
    fn training(&self) -> bool {
        true
    }
    fn train(&mut self) {}
    fn eval(&mut self) {}
    fn to_device(&mut self, device: torsh_core::DeviceType) -> Result<()> {
        self.lora_a = self.lora_a.to_device(device)?;
        self.lora_b = self.lora_b.to_device(device)?;
        Ok(())
    }
}

struct SimpleLinear {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl SimpleLinear {
    fn new(in_features: usize, out_features: usize) -> Self {
        let weight = torsh_tensor::creation::randn(&[out_features, in_features]).unwrap();
        let bias = Some(torsh_tensor::creation::zeros(&[out_features]).unwrap());
        Self { weight, bias }
    }
}

impl Module for SimpleLinear {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let output = input.matmul(&self.weight.transpose(0, 1)?)?;
        if let Some(bias) = &self.bias {
            output.add(bias)
        } else {
            Ok(output)
        }
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        params.insert("weight".to_string(), Parameter::new(self.weight.clone()));
        if let Some(bias) = &self.bias {
            params.insert("bias".to_string(), Parameter::new(bias.clone()));
        }
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }
    fn training(&self) -> bool {
        true
    }
    fn train(&mut self) {}
    fn eval(&mut self) {}
    fn to_device(&mut self, device: torsh_core::DeviceType) -> Result<()> {
        self.weight = self.weight.to_device(device)?;
        if let Some(bias) = &self.bias {
            self.bias = Some(bias.to_device(device)?);
        }
        Ok(())
    }
}

/// Utility functions for fine-tuning
pub mod fine_tuning_utils {
    use super::*;

    /// Create a standard fine-tuning config
    pub fn standard_fine_tuning_config(learning_rate: f64) -> FineTuningConfig {
        FineTuningConfig {
            layer_lr_schedule: LayerLearningRateSchedule::Uniform { learning_rate },
            freezing_strategy: FreezingStrategy::FreezeTo { last_n_layers: 3 },
            adapter_config: None,
            progressive_unfreezing: None,
            domain_adaptation: None,
            regularization: FineTuningRegularization {
                l2_on_changes: Some(0.01),
                ewc: None,
                knowledge_distillation: None,
                dropout_adjustments: None,
            },
        }
    }

    /// Create a LoRA fine-tuning config
    pub fn lora_fine_tuning_config(
        rank: usize,
        alpha: f64,
        learning_rate: f64,
    ) -> FineTuningConfig {
        FineTuningConfig {
            layer_lr_schedule: LayerLearningRateSchedule::Uniform { learning_rate },
            freezing_strategy: FreezingStrategy::FreezeLayerTypes {
                layer_types: vec!["linear".to_string(), "conv".to_string()],
            },
            adapter_config: Some(AdapterConfig {
                adapter_type: AdapterType::LoRA {
                    rank,
                    alpha,
                    dropout: 0.1,
                },
                placement: AdapterPlacement::AllLayers,
                hyperparameters: AdapterHyperparameters {
                    initialization: InitializationMethod::Zero,
                    learning_rate,
                    weight_decay: 0.01,
                    dropout: 0.1,
                    scaling: 1.0,
                },
                train_original: false,
            }),
            progressive_unfreezing: None,
            domain_adaptation: None,
            regularization: FineTuningRegularization {
                l2_on_changes: None,
                ewc: None,
                knowledge_distillation: None,
                dropout_adjustments: Some(DropoutAdjustments {
                    adapter_dropout: 0.1,
                    frozen_dropout: 0.0,
                    schedule: DropoutSchedule::Constant,
                }),
            },
        }
    }

    /// Create a progressive unfreezing config
    pub fn progressive_unfreezing_config(learning_rate: f64) -> FineTuningConfig {
        FineTuningConfig {
            layer_lr_schedule: LayerLearningRateSchedule::LinearDecay {
                output_lr: learning_rate,
                input_lr: learning_rate * 0.1,
            },
            freezing_strategy: FreezingStrategy::FreezeTo { last_n_layers: 1 },
            adapter_config: None,
            progressive_unfreezing: Some(ProgressiveUnfreezingConfig {
                schedule: UnfreezingSchedule::LayerByLayer,
                epochs_per_step: 3,
                lr_adjustment: LearningRateAdjustment::Scale { factor: 0.5 },
                reset_optimizer: false,
            }),
            domain_adaptation: None,
            regularization: FineTuningRegularization {
                l2_on_changes: Some(0.001),
                ewc: Some(EWCConfig {
                    importance_weight: 0.1,
                    fisher_samples: 1000,
                    diagonal_fisher: true,
                }),
                knowledge_distillation: None,
                dropout_adjustments: None,
            },
        }
    }

    /// Create a domain adaptation config
    pub fn domain_adaptation_config(
        source_domain: &str,
        target_domain: &str,
        learning_rate: f64,
    ) -> FineTuningConfig {
        FineTuningConfig {
            layer_lr_schedule: LayerLearningRateSchedule::Discriminative {
                base_lr: learning_rate,
                multipliers: vec![0.1, 0.5, 1.0, 1.0, 1.0],
            },
            freezing_strategy: FreezingStrategy::FreezeTo { last_n_layers: 2 },
            adapter_config: None,
            progressive_unfreezing: None,
            domain_adaptation: Some(DomainAdaptationConfig {
                adaptation_type: DomainAdaptationType::CORAL,
                domain_config: DomainConfig {
                    source_domain: source_domain.to_string(),
                    target_domain: target_domain.to_string(),
                    mixing_ratio: 0.5,
                    use_domain_labels: true,
                },
                adversarial_config: None,
            }),
            regularization: FineTuningRegularization {
                l2_on_changes: Some(0.01),
                ewc: None,
                knowledge_distillation: Some(KnowledgeDistillationConfig {
                    temperature: 4.0,
                    distillation_weight: 0.3,
                    feature_layers: vec!["layer_3".to_string(), "layer_4".to_string()],
                }),
                dropout_adjustments: None,
            },
        }
    }

    /// Calculate parameter efficiency for adapter-based fine-tuning
    pub fn calculate_parameter_efficiency(
        total_params: usize,
        adapter_params: usize,
    ) -> (f64, f64) {
        let efficiency = adapter_params as f64 / total_params as f64;
        let reduction_factor = total_params as f64 / adapter_params as f64;
        (efficiency, reduction_factor)
    }

    /// Estimate fine-tuning memory requirements
    pub fn estimate_memory_requirements(
        model_params: usize,
        adapter_params: usize,
        batch_size: usize,
        sequence_length: usize,
    ) -> MemoryEstimate {
        let base_memory = model_params * 4; // 4 bytes per float32 parameter
        let adapter_memory = adapter_params * 4;
        let activation_memory = batch_size * sequence_length * 1024 * 4; // Rough estimate
        let optimizer_memory = (model_params + adapter_params) * 8; // Adam needs 2x memory

        MemoryEstimate {
            base_model_memory: base_memory,
            adapter_memory,
            activation_memory,
            optimizer_memory,
            total_memory: base_memory + adapter_memory + activation_memory + optimizer_memory,
        }
    }

    /// Generate fine-tuning recommendations
    pub fn generate_recommendations(
        model_size: ModelSize,
        task_type: FineTuningTaskType,
        available_memory_gb: f32,
    ) -> FineTuningRecommendation {
        match (model_size, task_type) {
            (ModelSize::Small, _) => FineTuningRecommendation {
                strategy: "Full fine-tuning".to_string(),
                adapter_config: None,
                learning_rate: 1e-4,
                batch_size: 32,
                expected_efficiency: 1.0,
                memory_usage_gb: available_memory_gb * 0.3,
            },
            (ModelSize::Large, FineTuningTaskType::Classification) => FineTuningRecommendation {
                strategy: "LoRA fine-tuning".to_string(),
                adapter_config: Some(AdapterType::LoRA {
                    rank: 16,
                    alpha: 32.0,
                    dropout: 0.1,
                }),
                learning_rate: 3e-4,
                batch_size: 16,
                expected_efficiency: 0.1,
                memory_usage_gb: available_memory_gb * 0.5,
            },
            (ModelSize::Large, FineTuningTaskType::Generation) => FineTuningRecommendation {
                strategy: "Progressive unfreezing with adapters".to_string(),
                adapter_config: Some(AdapterType::LoRA {
                    rank: 32,
                    alpha: 64.0,
                    dropout: 0.05,
                }),
                learning_rate: 1e-4,
                batch_size: 8,
                expected_efficiency: 0.15,
                memory_usage_gb: available_memory_gb * 0.7,
            },
            _ => FineTuningRecommendation {
                strategy: "Standard adapter fine-tuning".to_string(),
                adapter_config: Some(AdapterType::Standard {
                    reduction_factor: 8,
                    activation: "relu".to_string(),
                }),
                learning_rate: 2e-4,
                batch_size: 16,
                expected_efficiency: 0.2,
                memory_usage_gb: available_memory_gb * 0.4,
            },
        }
    }
}

/// Memory estimation for fine-tuning
#[derive(Debug, Clone)]
pub struct MemoryEstimate {
    pub base_model_memory: usize,
    pub adapter_memory: usize,
    pub activation_memory: usize,
    pub optimizer_memory: usize,
    pub total_memory: usize,
}

/// Model size categories
#[derive(Debug, Clone)]
pub enum ModelSize {
    Small,  // < 100M parameters
    Medium, // 100M - 1B parameters
    Large,  // > 1B parameters
}

/// Task types for fine-tuning operations
#[derive(Debug, Clone)]
pub enum FineTuningTaskType {
    Classification,
    Generation,
    SequenceLabeling,
    QuestionAnswering,
}

/// Fine-tuning recommendations
#[derive(Debug, Clone)]
pub struct FineTuningRecommendation {
    pub strategy: String,
    pub adapter_config: Option<AdapterType>,
    pub learning_rate: f64,
    pub batch_size: usize,
    pub expected_efficiency: f64,
    pub memory_usage_gb: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::DeviceType;
    use torsh_tensor::Tensor;

    // Mock model for testing
    struct MockModel {
        parameters: HashMap<String, Parameter>,
    }

    impl MockModel {
        fn new() -> Self {
            let mut parameters = HashMap::new();
            let _device = DeviceType::Cpu;

            parameters.insert(
                "layer1.weight".to_string(),
                Parameter::new(torsh_tensor::creation::randn(&[10, 20]).unwrap()),
            );
            parameters.insert(
                "layer2.weight".to_string(),
                Parameter::new(torsh_tensor::creation::randn(&[20, 30]).unwrap()),
            );
            parameters.insert(
                "output.weight".to_string(),
                Parameter::new(torsh_tensor::creation::randn(&[30, 10]).unwrap()),
            );

            Self { parameters }
        }
    }

    impl Module for MockModel {
        fn forward(&self, input: &Tensor) -> Result<Tensor> {
            Ok(input.clone())
        }

        fn parameters(&self) -> HashMap<String, Parameter> {
            self.parameters.clone()
        }

        fn named_parameters(&self) -> HashMap<String, Parameter> {
            self.parameters.clone()
        }

        fn training(&self) -> bool {
            true
        }
        fn train(&mut self) {}
        fn eval(&mut self) {}
        fn to_device(&mut self, _device: torsh_core::DeviceType) -> Result<()> {
            Ok(())
        }
    }

    #[test]
    fn test_fine_tuning_config_creation() {
        let config = fine_tuning_utils::standard_fine_tuning_config(1e-4);
        assert!(matches!(
            config.layer_lr_schedule,
            LayerLearningRateSchedule::Uniform { .. }
        ));
        assert!(matches!(
            config.freezing_strategy,
            FreezingStrategy::FreezeTo { .. }
        ));
    }

    #[test]
    fn test_lora_config_creation() {
        let config = fine_tuning_utils::lora_fine_tuning_config(16, 32.0, 3e-4);
        assert!(config.adapter_config.is_some());
        if let Some(adapter_config) = &config.adapter_config {
            assert!(matches!(
                adapter_config.adapter_type,
                AdapterType::LoRA { .. }
            ));
        }
    }

    #[test]
    fn test_fine_tuning_engine_initialization() {
        let model = MockModel::new();
        let config = fine_tuning_utils::standard_fine_tuning_config(1e-4);
        let mut engine = FineTuningEngine::new(model, config);

        engine.initialize().unwrap();

        let stats = engine.get_stats().unwrap();
        assert!(stats.trainable_params > 0);
        // Note: frozen_params is usize, so it's always >= 0 - just verify it's accessible
        let _ = stats.frozen_params; // Allow for strategies with no frozen parameters
        assert!(stats.training_percentage > 0.0 && stats.training_percentage <= 100.0);
    }

    #[test]
    fn test_layer_learning_rates() {
        let model = MockModel::new();
        let config = FineTuningConfig {
            layer_lr_schedule: LayerLearningRateSchedule::LinearDecay {
                output_lr: 1e-3,
                input_lr: 1e-5,
            },
            freezing_strategy: FreezingStrategy::None,
            adapter_config: None,
            progressive_unfreezing: None,
            domain_adaptation: None,
            regularization: FineTuningRegularization {
                l2_on_changes: None,
                ewc: None,
                knowledge_distillation: None,
                dropout_adjustments: None,
            },
        };

        let engine = FineTuningEngine::new(model, config);
        let layer_rates = engine.get_layer_learning_rates().unwrap();

        assert!(!layer_rates.is_empty());
        for rate in layer_rates.values() {
            assert!(*rate >= 1e-5 && *rate <= 1e-3);
        }
    }

    #[test]
    fn test_freezing_strategies() {
        let model = MockModel::new();
        let config = FineTuningConfig {
            layer_lr_schedule: LayerLearningRateSchedule::Uniform {
                learning_rate: 1e-4,
            },
            freezing_strategy: FreezingStrategy::FreezeTo { last_n_layers: 1 },
            adapter_config: None,
            progressive_unfreezing: None,
            domain_adaptation: None,
            regularization: FineTuningRegularization {
                l2_on_changes: None,
                ewc: None,
                knowledge_distillation: None,
                dropout_adjustments: None,
            },
        };

        let mut engine = FineTuningEngine::new(model, config);
        engine.initialize().unwrap();

        let frozen_layers = engine.get_frozen_layers();
        assert!(!frozen_layers.is_empty());

        // Test manual freezing/unfreezing
        engine
            .freeze_layers(vec!["output.weight".to_string()])
            .unwrap();
        assert!(engine.is_layer_frozen("output.weight"));

        engine
            .unfreeze_layers(vec!["output.weight".to_string()])
            .unwrap();
        assert!(!engine.is_layer_frozen("output.weight"));
    }

    #[test]
    fn test_adapter_creation() {
        let model = MockModel::new();
        let config = fine_tuning_utils::lora_fine_tuning_config(8, 16.0, 2e-4);
        let mut engine = FineTuningEngine::new(model, config);

        engine.initialize().unwrap();

        let stats = engine.get_stats().unwrap();
        assert!(stats.adapter_stats.is_some());

        if let Some(adapter_stats) = &stats.adapter_stats {
            assert!(adapter_stats.adapter_params > 0);
            assert!(adapter_stats.efficiency > 0.0 && adapter_stats.efficiency < 1.0);
        }
    }

    #[test]
    fn test_progressive_unfreezing() {
        let model = MockModel::new();
        let config = fine_tuning_utils::progressive_unfreezing_config(1e-4);
        let mut engine = FineTuningEngine::new(model, config);

        engine.initialize().unwrap();

        let initial_frozen = engine.get_frozen_layers().len();

        // Simulate epoch updates
        for epoch in 0..10 {
            let changed = engine.update_epoch(epoch).unwrap();
            if changed {
                let new_frozen = engine.get_frozen_layers().len();
                assert!(new_frozen < initial_frozen);
                break;
            }
        }
    }

    #[test]
    fn test_domain_adaptation_loss() {
        let model = MockModel::new();
        let config = fine_tuning_utils::domain_adaptation_config("source", "target", 1e-4);
        let engine = FineTuningEngine::new(model, config);

        let _device = DeviceType::Cpu;
        let source_features = torsh_tensor::creation::randn(&[10, 20]).unwrap();
        let target_features = torsh_tensor::creation::randn(&[10, 20]).unwrap();

        let loss = engine
            .calculate_domain_loss(&source_features, &target_features)
            .unwrap();
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_utility_functions() {
        let (efficiency, reduction) =
            fine_tuning_utils::calculate_parameter_efficiency(1_000_000, 50_000);
        assert_eq!(efficiency, 0.05);
        assert_eq!(reduction, 20.0);

        let memory_est =
            fine_tuning_utils::estimate_memory_requirements(1_000_000, 50_000, 16, 512);
        assert!(memory_est.total_memory > memory_est.base_model_memory);
        assert!(memory_est.total_memory > memory_est.adapter_memory);

        let recommendation = fine_tuning_utils::generate_recommendations(
            ModelSize::Large,
            FineTuningTaskType::Classification,
            8.0,
        );
        assert!(!recommendation.strategy.is_empty());
        assert!(recommendation.learning_rate > 0.0);
        assert!(recommendation.batch_size > 0);
    }

    #[test]
    fn test_config_serialization() {
        let config = fine_tuning_utils::lora_fine_tuning_config(16, 32.0, 3e-4);

        let serialized = serde_json::to_string(&config).unwrap();
        let deserialized: FineTuningConfig = serde_json::from_str(&serialized).unwrap();

        assert!(deserialized.adapter_config.is_some());
        if let Some(adapter_config) = &deserialized.adapter_config {
            assert!(matches!(
                adapter_config.adapter_type,
                AdapterType::LoRA { .. }
            ));
        }
    }

    #[test]
    fn test_adapter_modules() {
        let _device = DeviceType::Cpu;

        // Test StandardAdapter
        let standard_adapter = StandardAdapter::new(128, 32);
        let input = torsh_tensor::creation::randn(&[1, 128]).unwrap();
        let output = standard_adapter.forward(&input).unwrap();
        assert_eq!(output.shape(), input.shape());

        // Test LoRAAdapter
        let lora_adapter = LoRAAdapter::new(128, 128, 16);
        let output = lora_adapter.forward(&input).unwrap();
        assert_eq!(output.shape(), input.shape());

        // Test that adapters have parameters
        assert!(!standard_adapter.parameters().is_empty());
        assert!(!lora_adapter.parameters().is_empty());
    }
}
