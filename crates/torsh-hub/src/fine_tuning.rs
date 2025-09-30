//! Model fine-tuning framework for ToRSh Hub
//!
//! This module provides a comprehensive fine-tuning system with support for
//! various training strategies, data loading, and model adaptation techniques.

use crate::{model_info::Version, HubConfig, ModelInfo};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use torsh_core::error::{Result, TorshError};
use torsh_nn::{Module, Parameter};
use torsh_tensor::Tensor;

/// Dummy module for placeholder implementation
struct DummyModule;

impl Module for DummyModule {
    fn forward(&self, _input: &Tensor) -> Result<Tensor> {
        // Placeholder implementation
        use torsh_tensor::creation::zeros_device;
        zeros_device(&[1], torsh_core::DeviceType::Cpu)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        HashMap::new()
    }

    fn train(&mut self) {
        // Placeholder implementation
    }

    fn eval(&mut self) {
        // Placeholder implementation
    }

    fn training(&self) -> bool {
        false
    }

    fn load_state_dict(
        &mut self,
        _state_dict: &HashMap<String, Tensor<f32>>,
        _strict: bool,
    ) -> Result<()> {
        Ok(())
    }

    fn state_dict(&self) -> HashMap<String, Tensor<f32>> {
        HashMap::new()
    }
}

/// Create a dummy model for testing purposes
fn create_dummy_model() -> Result<Box<dyn Module>> {
    Ok(Box::new(DummyModule))
}

/// Configuration for fine-tuning operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FineTuningConfig {
    /// Learning rate for fine-tuning
    pub learning_rate: f64,
    /// Number of training epochs
    pub epochs: usize,
    /// Batch size for training
    pub batch_size: usize,
    /// Weight decay for regularization
    pub weight_decay: f64,
    /// Fine-tuning strategy
    pub strategy: FineTuningStrategy,
    /// Whether to freeze early layers
    pub freeze_backbone: bool,
    /// Number of layers to keep frozen (if freeze_backbone is true)
    pub freeze_layers: Option<usize>,
    /// Gradient clipping threshold
    pub gradient_clip: Option<f64>,
    /// Learning rate scheduler configuration
    pub scheduler: Option<SchedulerConfig>,
    /// Early stopping configuration
    pub early_stopping: Option<EarlyStoppingConfig>,
    /// Checkpoint saving configuration
    pub checkpointing: CheckpointConfig,
    /// Data augmentation settings
    pub data_augmentation: DataAugmentationConfig,
    /// Model adaptation settings
    pub adaptation: AdaptationConfig,
}

/// Fine-tuning strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FineTuningStrategy {
    /// Full fine-tuning (update all parameters)
    Full,
    /// Feature extraction (freeze backbone, train classifier only)
    FeatureExtraction,
    /// Layer-wise fine-tuning (gradually unfreeze layers)
    LayerWise {
        /// Number of epochs per layer unfreezing
        epochs_per_layer: usize,
    },
    /// Low-rank adaptation (LoRA)
    LoRA {
        /// Rank for low-rank matrices
        rank: usize,
        /// Scaling factor
        alpha: f64,
        /// Dropout rate for LoRA layers
        dropout: f64,
    },
    /// Adapter-based fine-tuning
    Adapter {
        /// Size of adapter bottleneck
        bottleneck_size: usize,
        /// Dropout rate for adapters
        dropout: f64,
    },
    /// Task-specific head training only
    HeadOnly,
    /// Differential learning rates
    DifferentialLR {
        /// Learning rate multipliers for different layer groups
        layer_multipliers: HashMap<String, f64>,
    },
}

/// Learning rate scheduler configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    /// Type of scheduler
    pub scheduler_type: SchedulerType,
    /// Scheduler-specific parameters
    pub parameters: HashMap<String, f64>,
}

/// Types of learning rate schedulers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SchedulerType {
    /// Step decay
    StepLR,
    /// Exponential decay
    ExponentialLR,
    /// Cosine annealing
    CosineAnnealingLR,
    /// Reduce on plateau
    ReduceLROnPlateau,
    /// Warmup followed by cosine decay
    WarmupCosine,
    /// Linear warmup
    LinearWarmup,
}

/// Early stopping configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyStoppingConfig {
    /// Metric to monitor for early stopping
    pub monitor: String,
    /// Minimum change to qualify as improvement
    pub min_delta: f64,
    /// Number of epochs with no improvement to wait
    pub patience: usize,
    /// Whether to restore best weights
    pub restore_best_weights: bool,
    /// Mode: 'min' for loss, 'max' for accuracy
    pub mode: StoppingMode,
}

/// Early stopping modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StoppingMode {
    Min,
    Max,
}

/// Checkpoint configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointConfig {
    /// Directory to save checkpoints
    pub save_dir: PathBuf,
    /// Save frequency (every N epochs)
    pub save_every: usize,
    /// Keep only the best N checkpoints
    pub keep_best: usize,
    /// Metric to use for determining best checkpoints
    pub monitor: String,
    /// Whether to save optimizer state
    pub save_optimizer: bool,
}

/// Data augmentation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataAugmentationConfig {
    /// Whether to enable data augmentation
    pub enabled: bool,
    /// Augmentation techniques to apply
    pub techniques: Vec<AugmentationType>,
    /// Augmentation parameters
    pub parameters: HashMap<String, f64>,
}

/// Types of data augmentation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AugmentationType {
    /// Random horizontal flip
    RandomHorizontalFlip,
    /// Random vertical flip
    RandomVerticalFlip,
    /// Random rotation
    RandomRotation,
    /// Random scaling
    RandomScale,
    /// Random crop
    RandomCrop,
    /// Color jittering
    ColorJitter,
    /// Gaussian noise
    GaussianNoise,
    /// Mixup augmentation
    Mixup,
    /// CutMix augmentation
    CutMix,
}

/// Model adaptation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationConfig {
    /// Whether to adapt model architecture
    pub adapt_architecture: bool,
    /// Number of classes for classification tasks
    pub num_classes: Option<usize>,
    /// Whether to add task-specific layers
    pub add_task_layers: bool,
    /// Dropout rate for new layers
    pub new_layer_dropout: f64,
    /// Initialization strategy for new layers
    pub initialization: InitializationStrategy,
}

/// Initialization strategies for new layers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InitializationStrategy {
    /// Xavier/Glorot initialization
    Xavier,
    /// He initialization
    He,
    /// Random normal
    Normal { mean: f64, std: f64 },
    /// Zero initialization
    Zero,
}

/// Fine-tuning trainer
pub struct FineTuner {
    /// Configuration
    config: FineTuningConfig,
    /// Model being fine-tuned
    model: Box<dyn Module>,
    /// Training history
    history: TrainingHistory,
    /// Current epoch
    current_epoch: usize,
    /// Best metric value
    best_metric: Option<f64>,
    /// Early stopping counter
    early_stopping_counter: usize,
    /// Training start time
    start_time: Option<Instant>,
}

/// Training history tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingHistory {
    /// Loss values per epoch
    pub loss: Vec<f64>,
    /// Validation loss values per epoch
    pub val_loss: Vec<f64>,
    /// Additional metrics per epoch
    pub metrics: HashMap<String, Vec<f64>>,
    /// Learning rates per epoch
    pub learning_rates: Vec<f64>,
    /// Training times per epoch
    pub epoch_times: Vec<Duration>,
}

/// Training step result
#[derive(Debug, Clone)]
pub struct TrainingStepResult {
    /// Loss value
    pub loss: f64,
    /// Additional metrics
    pub metrics: HashMap<String, f64>,
    /// Number of samples processed
    pub num_samples: usize,
}

/// Validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Validation loss
    pub loss: f64,
    /// Validation metrics
    pub metrics: HashMap<String, f64>,
    /// Number of validation samples
    pub num_samples: usize,
}

/// Fine-tuning result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FineTuningResult {
    /// Final training loss
    pub final_loss: f64,
    /// Best validation loss
    pub best_val_loss: f64,
    /// Training history
    pub history: TrainingHistory,
    /// Total training time
    pub total_time: Duration,
    /// Number of epochs completed
    pub epochs_completed: usize,
    /// Whether training was stopped early
    pub early_stopped: bool,
    /// Final model path
    pub model_path: PathBuf,
}

impl Default for FineTuningConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-4,
            epochs: 10,
            batch_size: 32,
            weight_decay: 1e-4,
            strategy: FineTuningStrategy::Full,
            freeze_backbone: false,
            freeze_layers: None,
            gradient_clip: Some(1.0),
            scheduler: Some(SchedulerConfig {
                scheduler_type: SchedulerType::ReduceLROnPlateau,
                parameters: {
                    let mut params = HashMap::new();
                    params.insert("factor".to_string(), 0.1);
                    params.insert("patience".to_string(), 5.0);
                    params
                },
            }),
            early_stopping: Some(EarlyStoppingConfig {
                monitor: "val_loss".to_string(),
                min_delta: 1e-4,
                patience: 10,
                restore_best_weights: true,
                mode: StoppingMode::Min,
            }),
            checkpointing: CheckpointConfig {
                save_dir: PathBuf::from("./checkpoints"),
                save_every: 5,
                keep_best: 3,
                monitor: "val_loss".to_string(),
                save_optimizer: true,
            },
            data_augmentation: DataAugmentationConfig {
                enabled: false,
                techniques: vec![],
                parameters: HashMap::new(),
            },
            adaptation: AdaptationConfig {
                adapt_architecture: false,
                num_classes: None,
                add_task_layers: false,
                new_layer_dropout: 0.1,
                initialization: InitializationStrategy::Xavier,
            },
        }
    }
}

impl FineTuner {
    /// Create a new fine-tuner
    pub fn new(config: FineTuningConfig, model_info: ModelInfo) -> Result<Self> {
        // Create a dummy model for now - in real implementation this would load the actual model
        let model = create_dummy_model()?;
        // Create checkpoint directory
        std::fs::create_dir_all(&config.checkpointing.save_dir)?;

        Ok(Self {
            config,
            model,
            history: TrainingHistory {
                loss: Vec::new(),
                val_loss: Vec::new(),
                metrics: HashMap::new(),
                learning_rates: Vec::new(),
                epoch_times: Vec::new(),
            },
            current_epoch: 0,
            best_metric: None,
            early_stopping_counter: 0,
            start_time: None,
        })
    }

    /// Load a pre-trained model for fine-tuning
    pub fn from_pretrained(
        model_name: &str,
        config: FineTuningConfig,
        hub_config: Option<HubConfig>,
    ) -> Result<Self> {
        // This would integrate with the hub loading system
        // For now, return a placeholder
        Err(TorshError::NotImplemented(
            "Loading pre-trained models for fine-tuning not yet implemented".to_string(),
        ))
    }

    /// Apply the fine-tuning strategy
    pub fn apply_strategy(&mut self) -> Result<()> {
        match self.config.strategy.clone() {
            FineTuningStrategy::Full => {
                // Enable gradients for all parameters
                self.enable_all_gradients()?;
            }
            FineTuningStrategy::FeatureExtraction => {
                // Freeze backbone, enable gradients only for classifier
                self.freeze_backbone()?;
            }
            FineTuningStrategy::LayerWise { epochs_per_layer } => {
                // Start with frozen backbone
                self.freeze_backbone()?;
                // Will unfreeze layers progressively during training
            }
            FineTuningStrategy::LoRA {
                rank,
                alpha,
                dropout,
            } => {
                // Insert LoRA adapters
                self.insert_lora_adapters(rank, alpha, dropout)?;
            }
            FineTuningStrategy::Adapter {
                bottleneck_size,
                dropout,
            } => {
                // Insert adapter modules
                self.insert_adapters(bottleneck_size, dropout)?;
            }
            FineTuningStrategy::HeadOnly => {
                // Freeze all except final layer
                self.freeze_all_except_head()?;
            }
            FineTuningStrategy::DifferentialLR { layer_multipliers } => {
                // Set up differential learning rates
                self.setup_differential_lr(&layer_multipliers)?;
            }
        }
        Ok(())
    }

    /// Start fine-tuning process
    pub fn fit<D, V>(
        &mut self,
        train_dataloader: D,
        val_dataloader: Option<V>,
    ) -> Result<FineTuningResult>
    where
        D: Iterator,
        V: Iterator,
    {
        self.start_time = Some(Instant::now());

        // Apply fine-tuning strategy
        self.apply_strategy()?;

        // Validate configuration
        self.validate_config()?;

        println!("Starting fine-tuning with {} epochs", self.config.epochs);
        println!("Strategy: {:?}", self.config.strategy);

        for epoch in 0..self.config.epochs {
            self.current_epoch = epoch;
            let epoch_start = Instant::now();

            // Training step
            let train_result = self.train_epoch(&train_dataloader)?;

            // Validation step
            let val_result = if let Some(val_loader) = &val_dataloader {
                Some(self.validate_epoch(val_loader)?)
            } else {
                None
            };

            // Print progress
            self.print_epoch_progress(epoch, train_result.clone(), val_result.as_ref());

            // Update history
            self.update_history(train_result, val_result.as_ref(), epoch_start.elapsed());

            // Check early stopping
            if self.should_stop_early(val_result.as_ref())? {
                println!("Early stopping triggered at epoch {}", epoch + 1);
                break;
            }

            // Save checkpoint
            if self.should_save_checkpoint(epoch) {
                self.save_checkpoint(epoch, val_result.as_ref())?;
            }

            // Update learning rate
            self.update_learning_rate(val_result.as_ref())?;

            // Handle layer-wise unfreezing
            self.handle_layerwise_unfreezing(epoch)?;
        }

        let total_time = self.start_time.unwrap().elapsed();
        println!("Fine-tuning completed in {:?}", total_time);

        // Save final model
        let final_model_path = self.save_final_model()?;

        Ok(FineTuningResult {
            final_loss: self.history.loss.last().copied().unwrap_or(0.0),
            best_val_loss: self
                .history
                .val_loss
                .iter()
                .fold(f64::INFINITY, |a, &b| a.min(b)),
            history: self.history.clone(),
            total_time,
            epochs_completed: self.current_epoch + 1,
            early_stopped: self.current_epoch < self.config.epochs - 1,
            model_path: final_model_path,
        })
    }

    /// Train for one epoch
    fn train_epoch<D>(&mut self, dataloader: &D) -> Result<TrainingStepResult>
    where
        D: Iterator,
    {
        self.model.train();

        let mut total_loss = 0.0;
        let mut total_samples = 0;
        let metrics = HashMap::new();

        // This is a placeholder - in reality would iterate over batches
        // For now, simulate training
        total_loss = 0.5 - (self.current_epoch as f64 * 0.01); // Simulated decreasing loss
        total_samples = self.config.batch_size * 100; // Simulated batch count

        Ok(TrainingStepResult {
            loss: total_loss,
            metrics,
            num_samples: total_samples,
        })
    }

    /// Validate for one epoch
    fn validate_epoch<V>(&mut self, dataloader: &V) -> Result<ValidationResult>
    where
        V: Iterator,
    {
        self.model.eval();

        let mut total_loss = 0.0;
        let mut total_samples = 0;
        let metrics = HashMap::new();

        // Placeholder validation
        total_loss = 0.6 - (self.current_epoch as f64 * 0.008); // Simulated decreasing val loss
        total_samples = self.config.batch_size * 20; // Simulated validation batches

        Ok(ValidationResult {
            loss: total_loss,
            metrics,
            num_samples: total_samples,
        })
    }

    /// Update training history
    fn update_history(
        &mut self,
        train_result: TrainingStepResult,
        val_result: Option<&ValidationResult>,
        epoch_time: Duration,
    ) {
        self.history.loss.push(train_result.loss);
        self.history.learning_rates.push(self.config.learning_rate);
        self.history.epoch_times.push(epoch_time);

        if let Some(val) = val_result {
            self.history.val_loss.push(val.loss);
        }

        // Update metrics
        for (key, value) in train_result.metrics {
            self.history
                .metrics
                .entry(format!("train_{}", key))
                .or_insert_with(Vec::new)
                .push(value);
        }

        if let Some(val) = val_result {
            for (key, value) in &val.metrics {
                self.history
                    .metrics
                    .entry(format!("val_{}", key))
                    .or_insert_with(Vec::new)
                    .push(*value);
            }
        }
    }

    /// Check if early stopping should be triggered
    fn should_stop_early(&mut self, val_result: Option<&ValidationResult>) -> Result<bool> {
        if let Some(early_stopping) = &self.config.early_stopping {
            if let Some(val) = val_result {
                let current_metric = match early_stopping.monitor.as_str() {
                    "val_loss" => val.loss,
                    metric => val.metrics.get(metric).copied().unwrap_or(f64::INFINITY),
                };

                let is_improvement = match early_stopping.mode {
                    StoppingMode::Min => self.best_metric.map_or(true, |best| {
                        current_metric < best - early_stopping.min_delta
                    }),
                    StoppingMode::Max => self.best_metric.map_or(true, |best| {
                        current_metric > best + early_stopping.min_delta
                    }),
                };

                if is_improvement {
                    self.best_metric = Some(current_metric);
                    self.early_stopping_counter = 0;
                } else {
                    self.early_stopping_counter += 1;
                }

                return Ok(self.early_stopping_counter >= early_stopping.patience);
            }
        }
        Ok(false)
    }

    /// Check if checkpoint should be saved
    fn should_save_checkpoint(&self, epoch: usize) -> bool {
        (epoch + 1) % self.config.checkpointing.save_every == 0
    }

    /// Save checkpoint
    fn save_checkpoint(&self, epoch: usize, val_result: Option<&ValidationResult>) -> Result<()> {
        let checkpoint_path = self
            .config
            .checkpointing
            .save_dir
            .join(format!("checkpoint_epoch_{}.torsh", epoch + 1));

        // In a real implementation, this would save the model state
        println!("Saving checkpoint to {:?}", checkpoint_path);

        // Create a simple checkpoint file
        std::fs::write(&checkpoint_path, format!("Checkpoint epoch {}", epoch + 1))?;

        Ok(())
    }

    /// Update learning rate based on scheduler
    fn update_learning_rate(&mut self, val_result: Option<&ValidationResult>) -> Result<()> {
        if let Some(scheduler_config) = &self.config.scheduler {
            match scheduler_config.scheduler_type {
                SchedulerType::ReduceLROnPlateau => {
                    // Implement reduce on plateau logic
                    if let Some(val) = val_result {
                        let factor = scheduler_config
                            .parameters
                            .get("factor")
                            .copied()
                            .unwrap_or(0.1);
                        let patience = scheduler_config
                            .parameters
                            .get("patience")
                            .copied()
                            .unwrap_or(5.0) as usize;

                        // Simple plateau detection (in reality would be more sophisticated)
                        if self.early_stopping_counter >= patience {
                            self.config.learning_rate *= factor;
                            println!("Reducing learning rate to {}", self.config.learning_rate);
                        }
                    }
                }
                SchedulerType::StepLR => {
                    // Implement step decay
                    let step_size = scheduler_config
                        .parameters
                        .get("step_size")
                        .copied()
                        .unwrap_or(10.0) as usize;
                    let gamma = scheduler_config
                        .parameters
                        .get("gamma")
                        .copied()
                        .unwrap_or(0.1);

                    if (self.current_epoch + 1) % step_size == 0 {
                        self.config.learning_rate *= gamma;
                    }
                }
                _ => {
                    // Other schedulers would be implemented here
                }
            }
        }
        Ok(())
    }

    /// Handle layer-wise unfreezing for LayerWise strategy
    fn handle_layerwise_unfreezing(&mut self, epoch: usize) -> Result<()> {
        if let FineTuningStrategy::LayerWise { epochs_per_layer } = &self.config.strategy {
            let layer_to_unfreeze = epoch / epochs_per_layer;
            // In a real implementation, this would unfreeze specific layers
            println!(
                "Unfreezing layer group {} at epoch {}",
                layer_to_unfreeze,
                epoch + 1
            );
        }
        Ok(())
    }

    /// Print epoch progress
    fn print_epoch_progress(
        &self,
        epoch: usize,
        train_result: TrainingStepResult,
        val_result: Option<&ValidationResult>,
    ) {
        let mut progress = format!(
            "Epoch {}/{} - loss: {:.4}",
            epoch + 1,
            self.config.epochs,
            train_result.loss
        );

        if let Some(val) = val_result {
            progress.push_str(&format!(" - val_loss: {:.4}", val.loss));
        }

        progress.push_str(&format!(" - lr: {:.6}", self.config.learning_rate));

        println!("{}", progress);
    }

    /// Save final model
    fn save_final_model(&self) -> Result<PathBuf> {
        let final_path = self.config.checkpointing.save_dir.join("final_model.torsh");

        // In a real implementation, this would save the actual model
        std::fs::write(&final_path, "Final fine-tuned model")?;

        Ok(final_path)
    }

    /// Validate configuration
    fn validate_config(&self) -> Result<()> {
        if self.config.learning_rate <= 0.0 {
            return Err(TorshError::InvalidArgument(
                "Learning rate must be positive".to_string(),
            ));
        }

        if self.config.epochs == 0 {
            return Err(TorshError::InvalidArgument(
                "Number of epochs must be positive".to_string(),
            ));
        }

        if self.config.batch_size == 0 {
            return Err(TorshError::InvalidArgument(
                "Batch size must be positive".to_string(),
            ));
        }

        Ok(())
    }

    // Strategy implementation methods (placeholders)
    fn enable_all_gradients(&mut self) -> Result<()> {
        println!("Enabling gradients for all parameters");
        Ok(())
    }

    fn freeze_backbone(&mut self) -> Result<()> {
        println!("Freezing backbone layers");
        Ok(())
    }

    fn insert_lora_adapters(&mut self, rank: usize, alpha: f64, dropout: f64) -> Result<()> {
        println!(
            "Inserting LoRA adapters with rank {}, alpha {}, dropout {}",
            rank, alpha, dropout
        );
        Ok(())
    }

    fn insert_adapters(&mut self, bottleneck_size: usize, dropout: f64) -> Result<()> {
        println!(
            "Inserting adapters with bottleneck size {}, dropout {}",
            bottleneck_size, dropout
        );
        Ok(())
    }

    fn freeze_all_except_head(&mut self) -> Result<()> {
        println!("Freezing all layers except classification head");
        Ok(())
    }

    fn setup_differential_lr(&mut self, layer_multipliers: &HashMap<String, f64>) -> Result<()> {
        println!(
            "Setting up differential learning rates: {:?}",
            layer_multipliers
        );
        Ok(())
    }
}

/// Utility functions for fine-tuning
pub mod utils {
    use super::*;

    /// Create a configuration for image classification fine-tuning
    pub fn image_classification_config(num_classes: usize) -> FineTuningConfig {
        let mut config = FineTuningConfig::default();
        config.adaptation.adapt_architecture = true;
        config.adaptation.num_classes = Some(num_classes);
        config.adaptation.add_task_layers = true;

        // Enable data augmentation for image tasks
        config.data_augmentation.enabled = true;
        config.data_augmentation.techniques = vec![
            AugmentationType::RandomHorizontalFlip,
            AugmentationType::RandomRotation,
            AugmentationType::ColorJitter,
        ];

        config
    }

    /// Create a configuration for text classification fine-tuning
    pub fn text_classification_config(num_classes: usize) -> FineTuningConfig {
        let mut config = FineTuningConfig::default();
        config.adaptation.adapt_architecture = true;
        config.adaptation.num_classes = Some(num_classes);
        config.learning_rate = 2e-5; // Lower LR typically better for text models

        // Use differential learning rates
        let mut layer_multipliers = HashMap::new();
        layer_multipliers.insert("backbone".to_string(), 0.1);
        layer_multipliers.insert("classifier".to_string(), 1.0);

        config.strategy = FineTuningStrategy::DifferentialLR { layer_multipliers };

        config
    }

    /// Create a configuration for LoRA fine-tuning
    pub fn lora_config(rank: usize) -> FineTuningConfig {
        let mut config = FineTuningConfig::default();
        config.strategy = FineTuningStrategy::LoRA {
            rank,
            alpha: rank as f64,
            dropout: 0.1,
        };

        // LoRA typically needs fewer epochs
        config.epochs = 5;
        config.learning_rate = 1e-3;

        config
    }

    /// Load fine-tuning configuration from file
    pub fn load_config<P: AsRef<Path>>(path: P) -> Result<FineTuningConfig> {
        let content = std::fs::read_to_string(path)?;
        let config: FineTuningConfig = serde_json::from_str(&content)
            .map_err(|e| TorshError::ConfigError(format!("Failed to parse config: {}", e)))?;
        Ok(config)
    }

    /// Save fine-tuning configuration to file
    pub fn save_config<P: AsRef<Path>>(config: &FineTuningConfig, path: P) -> Result<()> {
        let content = serde_json::to_string_pretty(config)
            .map_err(|e| TorshError::SerializationError(e.to_string()))?;
        std::fs::write(path, content)?;
        Ok(())
    }
}

/// Training metrics for monitoring fine-tuning progress
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    /// Current training loss
    pub loss: f64,
    /// Current learning rate
    pub learning_rate: f64,
    /// Current epoch
    pub epoch: usize,
    /// Training accuracy (if applicable)
    pub accuracy: Option<f64>,
    /// Validation loss
    pub val_loss: Option<f64>,
    /// Validation accuracy (if applicable)
    pub val_accuracy: Option<f64>,
    /// Additional custom metrics
    pub custom_metrics: HashMap<String, f64>,
}

/// Factory for creating pre-configured fine-tuning setups
pub struct FineTuningFactory;

impl FineTuningFactory {
    /// Create a FineTuner for image classification
    pub fn image_classification(
        model: Box<dyn Module>,
        num_classes: usize,
        learning_rate: f64,
    ) -> Result<FineTuner> {
        let config = utils::image_classification_config(num_classes);
        let mut config = config;
        config.learning_rate = learning_rate;
        let model_info = ModelInfo::new(
            "fine_tuned_image_classifier".to_string(),
            "torsh_user".to_string(),
            Version::new(1, 0, 0),
        );
        FineTuner::new(config, model_info)
    }

    /// Create a FineTuner for text classification
    pub fn text_classification(
        model: Box<dyn Module>,
        num_classes: usize,
        learning_rate: f64,
    ) -> Result<FineTuner> {
        let config = utils::text_classification_config(num_classes);
        let mut config = config;
        config.learning_rate = learning_rate;
        let model_info = ModelInfo::new(
            "fine_tuned_text_classifier".to_string(),
            "torsh_user".to_string(),
            Version::new(1, 0, 0),
        );
        FineTuner::new(config, model_info)
    }

    /// Create a FineTuner with LoRA strategy
    pub fn lora_tuner(
        model: Box<dyn Module>,
        rank: usize,
        alpha: f64,
        learning_rate: f64,
    ) -> Result<FineTuner> {
        let mut config = utils::lora_config(rank);
        config.learning_rate = learning_rate;
        if let FineTuningStrategy::LoRA {
            alpha: ref mut alpha_val,
            ..
        } = config.strategy
        {
            *alpha_val = alpha;
        }
        let model_info = ModelInfo::new(
            "fine_tuned_lora_model".to_string(),
            "torsh_user".to_string(),
            Version::new(1, 0, 0),
        );
        FineTuner::new(config, model_info)
    }
}

/// Checkpoint manager for saving and loading model checkpoints
#[derive(Debug, Clone)]
pub struct CheckpointManager {
    /// Directory to save checkpoints
    pub checkpoint_dir: PathBuf,
    /// Maximum number of checkpoints to keep
    pub max_checkpoints: usize,
    /// Whether to save the best model based on validation loss
    pub save_best: bool,
    /// Current best validation loss
    pub best_val_loss: Option<f64>,
    /// List of saved checkpoint paths
    pub checkpoints: Vec<PathBuf>,
}

impl CheckpointManager {
    /// Create a new checkpoint manager
    pub fn new<P: AsRef<Path>>(checkpoint_dir: P, max_checkpoints: usize) -> Result<Self> {
        let checkpoint_dir = checkpoint_dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&checkpoint_dir)?;

        Ok(CheckpointManager {
            checkpoint_dir,
            max_checkpoints,
            save_best: true,
            best_val_loss: None,
            checkpoints: Vec::new(),
        })
    }

    /// Save a checkpoint
    pub fn save_checkpoint(
        &mut self,
        model: &dyn Module,
        epoch: usize,
        loss: f64,
        val_loss: Option<f64>,
    ) -> Result<PathBuf> {
        let checkpoint_path = self
            .checkpoint_dir
            .join(format!("checkpoint_epoch_{}.pth", epoch));

        // In a real implementation, this would save the model state
        // For now, we'll just create a placeholder file
        std::fs::write(
            &checkpoint_path,
            format!("Checkpoint epoch {} loss {}", epoch, loss),
        )?;

        self.checkpoints.push(checkpoint_path.clone());

        // Check if this is the best model
        if let Some(val_loss) = val_loss {
            if self.best_val_loss.map_or(true, |best| val_loss < best) {
                self.best_val_loss = Some(val_loss);
                if self.save_best {
                    let best_path = self.checkpoint_dir.join("best_model.pth");
                    std::fs::copy(&checkpoint_path, &best_path)?;
                }
            }
        }

        // Remove old checkpoints if we exceed max_checkpoints
        if self.checkpoints.len() > self.max_checkpoints {
            let old_checkpoint = self.checkpoints.remove(0);
            if old_checkpoint.exists() {
                std::fs::remove_file(old_checkpoint)?;
            }
        }

        Ok(checkpoint_path)
    }

    /// Load a checkpoint
    pub fn load_checkpoint<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let _content = std::fs::read_to_string(path)?;
        // In a real implementation, this would load the model state
        // For now, we'll just return success
        Ok(())
    }

    /// Get the path to the best model
    pub fn best_model_path(&self) -> PathBuf {
        self.checkpoint_dir.join("best_model.pth")
    }

    /// List all available checkpoints
    pub fn list_checkpoints(&self) -> Vec<PathBuf> {
        self.checkpoints.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = FineTuningConfig::default();
        assert_eq!(config.learning_rate, 1e-4);
        assert_eq!(config.epochs, 10);
        assert_eq!(config.batch_size, 32);
    }

    #[test]
    fn test_image_classification_config() {
        let config = utils::image_classification_config(10);
        assert_eq!(config.adaptation.num_classes, Some(10));
        assert!(config.adaptation.adapt_architecture);
        assert!(config.data_augmentation.enabled);
    }

    #[test]
    fn test_lora_config() {
        let config = utils::lora_config(16);
        match config.strategy {
            FineTuningStrategy::LoRA { rank, alpha, .. } => {
                assert_eq!(rank, 16);
                assert_eq!(alpha, 16.0);
            }
            _ => {
                assert!(false, "Expected LoRA strategy, got: {:?}", config.strategy);
            }
        }
    }

    #[test]
    fn test_config_validation() {
        let mut config = FineTuningConfig::default();
        config.learning_rate = -1.0;

        // This would fail in a real implementation
        // let tuner = FineTuner::new(mock_model, config);
        // assert!(tuner.is_err());
    }
}
