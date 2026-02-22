//! Training loop with integrated checkpointing
//!
//! This example demonstrates a complete training workflow with automatic
//! checkpointing, training resumption, and best model tracking.

use std::collections::HashMap;
use std::time::Instant;
use torsh_core::device::DeviceType;
use torsh_nn::{
    checkpoint::{CheckpointConfig, CheckpointManager, OptimizerState, TrainingStats},
    Module,
};
use torsh_tensor::creation::*;
use torsh_tensor::Tensor;
use torsh_text::{BertForSequenceClassification, TextModelConfig};

/// Configuration for training with checkpointing
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    pub num_epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub warmup_epochs: usize,
    pub eval_every_n_epochs: usize,
    pub max_sequence_length: usize,
    pub num_classes: usize,
    pub gradient_clip_norm: Option<f32>,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            num_epochs: 10,
            batch_size: 16,
            learning_rate: 2e-5,
            warmup_epochs: 2,
            eval_every_n_epochs: 1,
            max_sequence_length: 128,
            num_classes: 2,
            gradient_clip_norm: Some(1.0),
        }
    }
}

/// Training metrics for a single epoch
#[derive(Debug, Clone)]
pub struct EpochMetrics {
    pub epoch: usize,
    pub train_loss: f64,
    pub val_loss: f64,
    pub train_accuracy: f64,
    pub val_accuracy: f64,
    pub learning_rate: f64,
    pub epoch_duration: f64,
    pub samples_per_second: f64,
}

/// Training state for resumption
#[derive(Debug, Clone)]
pub struct TrainingState {
    pub current_epoch: usize,
    pub global_step: usize,
    pub best_val_accuracy: f64,
    pub metrics_history: Vec<EpochMetrics>,
    pub optimizer_state: Option<OptimizerState>,
    pub random_state: u64,
}

impl Default for TrainingState {
    fn default() -> Self {
        Self {
            current_epoch: 0,
            global_step: 0,
            best_val_accuracy: 0.0,
            metrics_history: Vec::new(),
            optimizer_state: None,
            random_state: 42,
        }
    }
}

/// Main trainer with integrated checkpointing
pub struct CheckpointedTrainer {
    model: BertForSequenceClassification,
    training_config: TrainingConfig,
    checkpoint_manager: CheckpointManager,
    training_state: TrainingState,
    device: DeviceType,
}

impl CheckpointedTrainer {
    /// Create a new trainer with checkpointing
    pub fn new(
        model_config: TextModelConfig,
        training_config: TrainingConfig,
        checkpoint_config: CheckpointConfig,
        device: DeviceType,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let model = BertForSequenceClassification::new(
            model_config,
            training_config.num_classes,
            device,
        )?;

        let checkpoint_manager = CheckpointManager::new(checkpoint_config)?;

        Ok(Self {
            model,
            training_config,
            checkpoint_manager,
            training_state: TrainingState::default(),
            device,
        })
    }

    /// Resume training from a checkpoint
    pub fn resume_from_checkpoint(
        mut self,
        checkpoint_path: &str,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        println!("📂 Resuming training from checkpoint: {}", checkpoint_path);

        let checkpoint = self.checkpoint_manager.load_checkpoint(&mut self.model, checkpoint_path)?;

        // Extract training state from checkpoint metadata
        if let Some(epoch) = checkpoint.metadata.epoch {
            self.training_state.current_epoch = epoch;
        }
        if let Some(step) = checkpoint.metadata.global_step {
            self.training_state.global_step = step;
        }

        // Extract optimizer state
        self.training_state.optimizer_state = checkpoint.optimizer_state;

        // Extract training statistics
        if let Some(stats) = checkpoint.training_stats {
            // Reconstruct metrics history from training stats
            for (i, (&train_loss, &val_loss)) in stats.train_losses.iter().zip(stats.val_losses.iter()).enumerate() {
                let train_acc = stats.train_accuracies.get(i).copied().unwrap_or(0.0);
                let val_acc = stats.val_accuracies.get(i).copied().unwrap_or(0.0);
                let lr = stats.learning_rates.get(i).copied().unwrap_or(self.training_config.learning_rate);
                let duration = stats.epoch_durations.get(i).copied().unwrap_or(0.0);

                let epoch_metrics = EpochMetrics {
                    epoch: i + 1,
                    train_loss,
                    val_loss,
                    train_accuracy: train_acc,
                    val_accuracy: val_acc,
                    learning_rate: lr,
                    epoch_duration: duration,
                    samples_per_second: 0.0, // Cannot reconstruct this
                };

                self.training_state.metrics_history.push(epoch_metrics);

                // Update best validation accuracy
                if val_acc > self.training_state.best_val_accuracy {
                    self.training_state.best_val_accuracy = val_acc;
                }
            }
        }

        println!("✅ Training resumed from epoch {}", self.training_state.current_epoch);
        println!("   Best validation accuracy so far: {:.4}", self.training_state.best_val_accuracy);

        Ok(self)
    }

    /// Run complete training with checkpointing
    pub fn train(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        println!("🚀 Starting training with checkpointing");
        println!("=======================================");
        self.print_training_config();

        let start_epoch = self.training_state.current_epoch + 1;
        let end_epoch = self.training_config.num_epochs;

        for epoch in start_epoch..=end_epoch {
            println!("\n📅 Epoch {}/{}", epoch, self.training_config.num_epochs);
            
            let epoch_start = Instant::now();

            // Training phase
            let (train_loss, train_accuracy) = self.train_epoch(epoch)?;

            // Validation phase
            let (val_loss, val_accuracy) = if epoch % self.training_config.eval_every_n_epochs == 0 {
                self.validate_epoch(epoch)?
            } else {
                // Use previous validation metrics if not evaluating this epoch
                let last_metrics = self.training_state.metrics_history.last();
                (
                    last_metrics.map(|m| m.val_loss).unwrap_or(train_loss),
                    last_metrics.map(|m| m.val_accuracy).unwrap_or(train_accuracy),
                )
            };

            let epoch_duration = epoch_start.elapsed().as_secs_f64();
            let learning_rate = self.get_current_learning_rate(epoch);

            // Calculate samples per second (estimated)
            let samples_per_second = (self.training_config.batch_size * 100) as f64 / epoch_duration;

            // Create epoch metrics
            let epoch_metrics = EpochMetrics {
                epoch,
                train_loss,
                val_loss,
                train_accuracy,
                val_accuracy,
                learning_rate,
                epoch_duration,
                samples_per_second,
            };

            self.training_state.metrics_history.push(epoch_metrics.clone());
            self.training_state.current_epoch = epoch;
            self.training_state.global_step += 100; // Simulate steps per epoch

            // Print epoch results
            self.print_epoch_results(&epoch_metrics);

            // Check if this is the best model
            let is_best = val_accuracy > self.training_state.best_val_accuracy;
            if is_best {
                self.training_state.best_val_accuracy = val_accuracy;
                println!("   🏆 New best validation accuracy: {:.4}", val_accuracy);
            }

            // Save checkpoint
            if epoch % self.training_config.eval_every_n_epochs == 0 {
                self.save_checkpoint(epoch)?;
            }

            // Early stopping check
            if self.should_early_stop() {
                println!("🛑 Early stopping triggered");
                break;
            }
        }

        self.print_training_summary();
        Ok(())
    }

    /// Train for one epoch
    fn train_epoch(&mut self, epoch: usize) -> Result<(f64, f64), Box<dyn std::error::Error>> {
        self.model.train();

        let mut total_loss = 0.0;
        let mut correct_predictions = 0;
        let mut total_samples = 0;
        let num_batches = 100; // Simulate 100 batches per epoch

        for batch_idx in 0..num_batches {
            // Generate synthetic training data
            let input_ids = self.generate_batch_data()?;
            let labels = self.generate_batch_labels()?;

            // Forward pass
            let logits = self.model.forward(&input_ids)?;
            let loss = self.compute_loss(&logits, &labels)?;

            // Simulate backward pass and optimization
            // In a real implementation, this would involve:
            // 1. loss.backward()
            // 2. optimizer.step()
            // 3. optimizer.zero_grad()

            total_loss += loss;

            // Calculate accuracy (simplified)
            let predictions = self.get_predictions(&logits)?;
            correct_predictions += self.count_correct_predictions(&predictions, &labels)?;
            total_samples += self.training_config.batch_size;

            // Print progress occasionally
            if batch_idx % 20 == 0 {
                let current_loss = total_loss / (batch_idx + 1) as f64;
                let current_acc = correct_predictions as f64 / total_samples as f64;
                print!("\r   Training: Batch {}/{}, Loss: {:.4}, Acc: {:.4}", 
                       batch_idx + 1, num_batches, current_loss, current_acc);
            }
        }

        let avg_loss = total_loss / num_batches as f64;
        let accuracy = correct_predictions as f64 / total_samples as f64;

        println!(); // New line after progress updates
        Ok((avg_loss, accuracy))
    }

    /// Validate for one epoch
    fn validate_epoch(&mut self, _epoch: usize) -> Result<(f64, f64), Box<dyn std::error::Error>> {
        self.model.eval();

        let mut total_loss = 0.0;
        let mut correct_predictions = 0;
        let mut total_samples = 0;
        let num_batches = 20; // Simulate 20 validation batches

        for batch_idx in 0..num_batches {
            // Generate synthetic validation data
            let input_ids = self.generate_batch_data()?;
            let labels = self.generate_batch_labels()?;

            // Forward pass (no gradients in validation)
            let logits = self.model.forward(&input_ids)?;
            let loss = self.compute_loss(&logits, &labels)?;

            total_loss += loss;

            // Calculate accuracy
            let predictions = self.get_predictions(&logits)?;
            correct_predictions += self.count_correct_predictions(&predictions, &labels)?;
            total_samples += self.training_config.batch_size;
        }

        let avg_loss = total_loss / num_batches as f64;
        let accuracy = correct_predictions as f64 / total_samples as f64;

        Ok((avg_loss, accuracy))
    }

    /// Save a checkpoint
    fn save_checkpoint(&mut self, epoch: usize) -> Result<(), Box<dyn std::error::Error>> {
        // Create training statistics
        let training_stats = TrainingStats {
            train_losses: self.training_state.metrics_history.iter().map(|m| m.train_loss).collect(),
            val_losses: self.training_state.metrics_history.iter().map(|m| m.val_loss).collect(),
            train_accuracies: self.training_state.metrics_history.iter().map(|m| m.train_accuracy).collect(),
            val_accuracies: self.training_state.metrics_history.iter().map(|m| m.val_accuracy).collect(),
            learning_rates: self.training_state.metrics_history.iter().map(|m| m.learning_rate).collect(),
            epoch_durations: self.training_state.metrics_history.iter().map(|m| m.epoch_duration).collect(),
            custom_metrics: {
                let mut metrics = HashMap::new();
                metrics.insert("val_accuracy".to_string(), 
                              self.training_state.metrics_history.iter().map(|m| m.val_accuracy).collect());
                metrics.insert("samples_per_second".to_string(),
                              self.training_state.metrics_history.iter().map(|m| m.samples_per_second).collect());
                metrics
            },
        };

        // Create optimizer state (simplified)
        let optimizer_state = OptimizerState {
            optimizer_type: "AdamW".to_string(),
            learning_rate: self.get_current_learning_rate(epoch),
            momentum_states: HashMap::new(),
            velocity_states: HashMap::new(),
            step_counts: {
                let mut counts = HashMap::new();
                counts.insert("global".to_string(), self.training_state.global_step);
                counts
            },
            custom_state: HashMap::new(),
        };

        // Create custom metadata
        let custom_metadata = {
            let mut metadata = HashMap::new();
            metadata.insert("best_val_accuracy".to_string(), 
                           format!("{:.6}", self.training_state.best_val_accuracy));
            metadata.insert("random_state".to_string(), 
                           self.training_state.random_state.to_string());
            metadata.insert("batch_size".to_string(), 
                           self.training_config.batch_size.to_string());
            metadata.insert("learning_rate".to_string(), 
                           format!("{:.2e}", self.training_config.learning_rate));
            metadata
        };

        let checkpoint_path = self.checkpoint_manager.save_checkpoint(
            &self.model,
            epoch,
            self.training_state.global_step,
            Some(optimizer_state),
            Some(training_stats),
            Some(custom_metadata),
        )?;

        println!("   💾 Checkpoint saved: {}", checkpoint_path.split('/').last().unwrap_or(""));

        Ok(())
    }

    /// Generate synthetic batch data
    fn generate_batch_data(&self) -> Result<Tensor, Box<dyn std::error::Error>> {
        let input_ids: Tensor<f32> = rand(&[
            self.training_config.batch_size,
            self.training_config.max_sequence_length,
        ]);
        
        // Scale to vocabulary range (simplified)
        let scaled = input_ids.mul_scalar(30522.0)?.floor()?.abs()?; // BERT vocab size
        Ok(scaled)
    }

    /// Generate synthetic labels
    fn generate_batch_labels(&self) -> Result<Tensor, Box<dyn std::error::Error>> {
        let labels: Tensor<f32> = rand(&[self.training_config.batch_size]);
        let scaled = labels.mul_scalar(self.training_config.num_classes as f32)?.floor()?;
        Ok(scaled)
    }

    /// Compute loss (simplified cross-entropy)
    fn compute_loss(&self, logits: &Tensor, _labels: &Tensor) -> Result<f64, Box<dyn std::error::Error>> {
        // Simplified loss computation
        let loss_tensor = logits.sum()?;
        let loss_value = loss_tensor.data()[0] as f64;
        Ok((loss_value / (self.training_config.batch_size as f64)).abs().ln())
    }

    /// Get predictions from logits
    fn get_predictions(&self, logits: &Tensor) -> Result<Tensor, Box<dyn std::error::Error>> {
        // Simplified argmax (take the sign of the sum)
        let predictions = logits.sum_dim(&[-1], false)?;
        Ok(predictions)
    }

    /// Count correct predictions
    fn count_correct_predictions(&self, _predictions: &Tensor, _labels: &Tensor) -> Result<usize, Box<dyn std::error::Error>> {
        // Simulate random accuracy between 0.5 and 0.95 based on epoch
        let base_accuracy = 0.5 + (self.training_state.current_epoch as f64 / self.training_config.num_epochs as f64) * 0.45;
        let noise = (rand::<f32>(&[1]).data()[0] - 0.5) * 0.1;
        let accuracy = (base_accuracy + noise as f64).clamp(0.0, 1.0);
        
        Ok((accuracy * self.training_config.batch_size as f64) as usize)
    }

    /// Get current learning rate (with warmup and decay)
    fn get_current_learning_rate(&self, epoch: usize) -> f64 {
        let base_lr = self.training_config.learning_rate;
        
        if epoch <= self.training_config.warmup_epochs {
            // Linear warmup
            base_lr * (epoch as f64 / self.training_config.warmup_epochs as f64)
        } else {
            // Cosine decay
            let progress = (epoch - self.training_config.warmup_epochs) as f64 
                         / (self.training_config.num_epochs - self.training_config.warmup_epochs) as f64;
            base_lr * (1.0 + (std::f64::consts::PI * progress).cos()) / 2.0
        }
    }

    /// Check if early stopping should be triggered
    fn should_early_stop(&self) -> bool {
        // Simple early stopping: if validation accuracy hasn't improved for 3 epochs
        if self.training_state.metrics_history.len() < 4 {
            return false;
        }

        let recent_accuracies: Vec<f64> = self.training_state.metrics_history
            .iter()
            .rev()
            .take(3)
            .map(|m| m.val_accuracy)
            .collect();

        let best_recent = recent_accuracies.iter().fold(0.0, |a, &b| a.max(b));
        let current = recent_accuracies[0];

        // Stop if current accuracy is significantly worse than best recent
        current < best_recent - 0.01
    }

    /// Print training configuration
    fn print_training_config(&self) {
        println!("📋 Training Configuration:");
        println!("   Epochs: {}", self.training_config.num_epochs);
        println!("   Batch size: {}", self.training_config.batch_size);
        println!("   Learning rate: {:.2e}", self.training_config.learning_rate);
        println!("   Sequence length: {}", self.training_config.max_sequence_length);
        println!("   Number of classes: {}", self.training_config.num_classes);
        println!("   Device: {:?}", self.device);
    }

    /// Print epoch results
    fn print_epoch_results(&self, metrics: &EpochMetrics) {
        println!("   📊 Results:");
        println!("      Train Loss: {:.4} | Train Acc: {:.4}", metrics.train_loss, metrics.train_accuracy);
        println!("      Val Loss:   {:.4} | Val Acc:   {:.4}", metrics.val_loss, metrics.val_accuracy);
        println!("      Learning Rate: {:.2e}", metrics.learning_rate);
        println!("      Duration: {:.1}s | Samples/sec: {:.0}", metrics.epoch_duration, metrics.samples_per_second);
    }

    /// Print training summary
    fn print_training_summary(&self) {
        println!("\n🎯 Training Summary");
        println!("===================");
        println!("✅ Training completed successfully!");
        println!("   Total epochs: {}", self.training_state.current_epoch);
        println!("   Best validation accuracy: {:.4}", self.training_state.best_val_accuracy);
        
        if let Some(best_path) = self.checkpoint_manager.best_model_path() {
            println!("   Best model saved at: {}", best_path.split('/').last().unwrap_or(""));
        }

        // Print learning curve summary
        if !self.training_state.metrics_history.is_empty() {
            let final_metrics = self.training_state.metrics_history.last().unwrap();
            let total_duration: f64 = self.training_state.metrics_history.iter()
                .map(|m| m.epoch_duration)
                .sum();
            
            println!("   Final train accuracy: {:.4}", final_metrics.train_accuracy);
            println!("   Final validation accuracy: {:.4}", final_metrics.val_accuracy);
            println!("   Total training time: {:.1} minutes", total_duration / 60.0);
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎓 Training with Checkpointing Example");
    println!("======================================\n");

    let device = DeviceType::Cpu;

    // Configuration
    let model_config = TextModelConfig::bert_base();
    let training_config = TrainingConfig {
        num_epochs: 5,
        batch_size: 8,
        learning_rate: 2e-5,
        warmup_epochs: 1,
        eval_every_n_epochs: 1,
        max_sequence_length: 64,
        num_classes: 2,
        gradient_clip_norm: Some(1.0),
    };

    let checkpoint_config = CheckpointConfig {
        save_dir: "./checkpoints/training_example".to_string(),
        filename_prefix: "bert_sentiment".to_string(),
        max_checkpoints: 3,
        save_every_n_epochs: 1,
        save_best: true,
        best_metric_name: "val_accuracy".to_string(),
        higher_is_better: true,
        compression_level: 6,
    };

    // Create trainer
    let mut trainer = CheckpointedTrainer::new(
        model_config,
        training_config,
        checkpoint_config,
        device,
    )?;

    // Run training
    trainer.train()?;

    // Demonstrate resumption
    println!("\n🔄 Demonstrating Training Resumption");
    println!("=====================================");

    // Get the last checkpoint
    let checkpoints = trainer.checkpoint_manager.list_checkpoints();
    if let Some(last_checkpoint) = checkpoints.last() {
        println!("📂 Found checkpoint: {}", last_checkpoint.split('/').last().unwrap_or(""));
        
        // Create new trainer and resume
        let resumed_trainer = CheckpointedTrainer::new(
            TextModelConfig::bert_base(),
            TrainingConfig {
                num_epochs: 8, // Train for 3 more epochs
                ..TrainingConfig::default()
            },
            CheckpointConfig {
                save_dir: "./checkpoints/training_example_resumed".to_string(),
                ..CheckpointConfig::default()
            },
            device,
        )?
        .resume_from_checkpoint(last_checkpoint)?;

        println!("✅ Successfully demonstrated checkpoint resumption");
        println!("   Resumed from epoch: {}", resumed_trainer.training_state.current_epoch);
    }

    println!("\n🎉 Training with checkpointing example completed!");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trainer_creation() {
        let device = DeviceType::Cpu;
        let model_config = TextModelConfig::bert_base();
        let training_config = TrainingConfig::default();
        let checkpoint_config = CheckpointConfig::default();

        let trainer = CheckpointedTrainer::new(
            model_config,
            training_config,
            checkpoint_config,
            device,
        );

        assert!(trainer.is_ok());
    }

    #[test]
    fn test_learning_rate_schedule() {
        let trainer = CheckpointedTrainer {
            model: BertForSequenceClassification::new(
                TextModelConfig::bert_base(),
                2,
                DeviceType::Cpu,
            ).unwrap(),
            training_config: TrainingConfig {
                learning_rate: 1e-4,
                warmup_epochs: 2,
                num_epochs: 10,
                ..Default::default()
            },
            checkpoint_manager: CheckpointManager::new(CheckpointConfig::default()).unwrap(),
            training_state: TrainingState::default(),
            device: DeviceType::Cpu,
        };

        // Test warmup
        assert!(trainer.get_current_learning_rate(1) < trainer.training_config.learning_rate);
        assert_eq!(trainer.get_current_learning_rate(2), trainer.training_config.learning_rate);
        
        // Test decay
        assert!(trainer.get_current_learning_rate(10) < trainer.training_config.learning_rate);
    }
}