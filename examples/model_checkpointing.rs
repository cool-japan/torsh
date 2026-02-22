//! Model checkpointing and serialization example
//!
//! This example demonstrates how to save and load transformer models using
//! ToRSh's comprehensive checkpointing system with training resumption capabilities.

use std::collections::HashMap;
use torsh_core::device::DeviceType;
use torsh_nn::{
    checkpoint::{
        CheckpointConfig, CheckpointManager, OptimizerState, TrainingStats,
        utils,
    },
    Module,
};
use torsh_tensor::creation::*;
use torsh_tensor::Tensor;
use torsh_text::{
    BertForSequenceClassification, GPTForCausalLM, T5ForConditionalGeneration,
    TextModelConfig,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("💾 Model Checkpointing and Serialization Demo");
    println!("==============================================\n");

    let device = DeviceType::Cpu;

    // Create checkpoint directory
    std::fs::create_dir_all("./checkpoints")?;

    // Demonstrate checkpointing with different model types
    demonstrate_gpt_checkpointing(device)?;
    demonstrate_bert_checkpointing(device)?;
    demonstrate_t5_checkpointing(device)?;

    // Advanced checkpointing features
    demonstrate_training_resumption(device)?;
    demonstrate_best_model_tracking(device)?;

    // Checkpoint utilities and analysis
    demonstrate_checkpoint_utilities()?;

    println!("✅ Model checkpointing demonstration completed!");
    Ok(())
}

/// Demonstrate checkpointing with GPT models
fn demonstrate_gpt_checkpointing(device: DeviceType) -> Result<(), Box<dyn std::error::Error>> {
    println!("🤖 GPT Model Checkpointing");
    println!("===========================\n");

    // Create a GPT model
    let config = TextModelConfig::gpt2_small();
    let mut model = GPTForCausalLM::new(config.clone());

    println!("📊 Model Information:");
    println!("   Architecture: GPT-2 Small");
    println!("   Parameters: {:.2}M", utils::count_parameters(&model) as f32 / 1_000_000.0);
    println!("   Memory: {:.2} MB", utils::estimate_model_memory(&model) as f32 / (1024.0 * 1024.0));

    // Configure checkpointing
    let checkpoint_config = CheckpointConfig {
        save_dir: "./checkpoints/gpt".to_string(),
        filename_prefix: "gpt2_small".to_string(),
        max_checkpoints: 3,
        save_every_n_epochs: 1,
        save_best: true,
        best_metric_name: "val_loss".to_string(),
        higher_is_better: false,
        compression_level: 6,
    };

    let mut checkpoint_manager = CheckpointManager::new(checkpoint_config)?;

    // Create mock training statistics
    let training_stats = TrainingStats {
        train_losses: vec![2.5, 2.1, 1.8, 1.6],
        val_losses: vec![2.7, 2.3, 2.0, 1.9],
        train_accuracies: vec![0.3, 0.45, 0.6, 0.72],
        val_accuracies: vec![0.28, 0.42, 0.58, 0.68],
        learning_rates: vec![1e-4, 9e-5, 8e-5, 7e-5],
        epoch_durations: vec![120.5, 118.2, 116.8, 115.3],
        custom_metrics: {
            let mut metrics = HashMap::new();
            metrics.insert("val_loss".to_string(), vec![2.7, 2.3, 2.0, 1.9]);
            metrics.insert("perplexity".to_string(), vec![14.8, 9.9, 7.4, 6.7]);
            metrics
        },
    };

    // Create mock optimizer state
    let optimizer_state = OptimizerState {
        optimizer_type: "Adam".to_string(),
        learning_rate: 1e-4,
        momentum_states: HashMap::new(),
        velocity_states: HashMap::new(),
        step_counts: HashMap::new(),
        custom_state: HashMap::new(),
    };

    // Save checkpoint
    println!("💾 Saving GPT checkpoint...");
    let checkpoint_path = checkpoint_manager.save_checkpoint(
        &model,
        4, // epoch
        1000, // global_step
        Some(optimizer_state),
        Some(training_stats),
        Some({
            let mut custom = HashMap::new();
            custom.insert("experiment_name".to_string(), "gpt2_language_modeling".to_string());
            custom.insert("dataset".to_string(), "wikitext-103".to_string());
            custom
        }),
    )?;

    println!("   ✅ Checkpoint saved: {}", checkpoint_path);

    // Test loading
    println!("🔄 Loading GPT checkpoint...");
    let mut loaded_model = GPTForCausalLM::new(config);
    let loaded_checkpoint = checkpoint_manager.load_checkpoint(&mut loaded_model, &checkpoint_path)?;

    println!("   ✅ Checkpoint loaded successfully");
    println!("   📅 Created: {}", loaded_checkpoint.metadata.timestamp);
    println!("   🏆 Best metric: {:?}", loaded_checkpoint.metadata.best_metric);

    // Verify model parameters match
    verify_model_parameters(&model, &loaded_model)?;

    println!("   ✅ Parameter verification passed\n");

    Ok(())
}

/// Demonstrate checkpointing with BERT models
fn demonstrate_bert_checkpointing(device: DeviceType) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 BERT Model Checkpointing");
    println!("============================\n");

    // Create a BERT classification model
    let config = TextModelConfig::bert_base();
    let mut model = BertForSequenceClassification::new(config.clone(), 2, device)?; // Binary classification

    println!("📊 Model Information:");
    println!("   Architecture: BERT Base (Classification)");
    println!("   Parameters: {:.2}M", utils::count_parameters(&model) as f32 / 1_000_000.0);
    println!("   Classes: 2 (binary classification)");

    // Configure checkpointing with different settings
    let checkpoint_config = CheckpointConfig {
        save_dir: "./checkpoints/bert".to_string(),
        filename_prefix: "bert_base_classifier".to_string(),
        max_checkpoints: 5,
        save_best: true,
        best_metric_name: "val_accuracy".to_string(),
        higher_is_better: true, // Higher accuracy is better
        compression_level: 9, // Maximum compression
        ..Default::default()
    };

    let mut checkpoint_manager = CheckpointManager::new(checkpoint_config)?;

    // Create comprehensive training statistics for classification
    let training_stats = TrainingStats {
        train_losses: vec![0.693, 0.421, 0.287, 0.198, 0.156],
        val_losses: vec![0.701, 0.445, 0.312, 0.234, 0.201],
        train_accuracies: vec![0.512, 0.798, 0.876, 0.923, 0.941],
        val_accuracies: vec![0.503, 0.781, 0.854, 0.897, 0.915],
        learning_rates: vec![2e-5, 2e-5, 1.5e-5, 1e-5, 5e-6],
        epoch_durations: vec![180.2, 178.9, 177.1, 176.8, 175.4],
        custom_metrics: {
            let mut metrics = HashMap::new();
            metrics.insert("val_accuracy".to_string(), vec![0.503, 0.781, 0.854, 0.897, 0.915]);
            metrics.insert("f1_score".to_string(), vec![0.485, 0.773, 0.849, 0.891, 0.908]);
            metrics.insert("precision".to_string(), vec![0.492, 0.785, 0.861, 0.904, 0.922]);
            metrics.insert("recall".to_string(), vec![0.478, 0.761, 0.837, 0.878, 0.895]);
            metrics
        },
    };

    // Save multiple checkpoints to demonstrate best model tracking
    for epoch in 1..=5 {
        let step = epoch * 200;
        let checkpoint_path = checkpoint_manager.save_checkpoint(
            &model,
            epoch,
            step,
            None, // No optimizer state for this example
            Some(training_stats.clone()),
            None,
        )?;

        println!("   💾 Epoch {} checkpoint: {}", epoch, checkpoint_path.split('/').last().unwrap_or(""));
    }

    // Check best model
    if let Some(best_path) = checkpoint_manager.best_model_path() {
        println!("   🏆 Best model saved at: {}", best_path.split('/').last().unwrap_or(""));
    }

    // List all checkpoints
    let checkpoints = checkpoint_manager.list_checkpoints();
    println!("   📂 Total checkpoints: {}", checkpoints.len());

    println!("   ✅ BERT checkpointing completed\n");

    Ok(())
}

/// Demonstrate checkpointing with T5 models
fn demonstrate_t5_checkpointing(device: DeviceType) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔄 T5 Model Checkpointing");
    println!("==========================\n");

    // Create a T5 model for conditional generation
    let config = TextModelConfig::t5_small();
    let mut model = T5ForConditionalGeneration::new(config.clone(), device)?;

    println!("📊 Model Information:");
    println!("   Architecture: T5 Small (Conditional Generation)");
    println!("   Parameters: {:.2}M", utils::count_parameters(&model) as f32 / 1_000_000.0);
    println!("   Type: Encoder-Decoder");

    // Configure checkpointing for sequence-to-sequence tasks
    let checkpoint_config = CheckpointConfig {
        save_dir: "./checkpoints/t5".to_string(),
        filename_prefix: "t5_small_translation".to_string(),
        best_metric_name: "bleu_score".to_string(),
        higher_is_better: true, // Higher BLEU is better
        compression_level: 3, // Moderate compression
        ..Default::default()
    };

    let mut checkpoint_manager = CheckpointManager::new(checkpoint_config)?;

    // Create training statistics for sequence-to-sequence tasks
    let training_stats = TrainingStats {
        train_losses: vec![3.2, 2.8, 2.4, 2.1, 1.9],
        val_losses: vec![3.5, 3.0, 2.6, 2.3, 2.1],
        train_accuracies: vec![0.12, 0.23, 0.34, 0.45, 0.52],
        val_accuracies: vec![0.10, 0.21, 0.31, 0.41, 0.48],
        learning_rates: vec![1e-4, 8e-5, 6e-5, 4e-5, 2e-5],
        epoch_durations: vec![240.1, 238.7, 236.2, 234.8, 233.1],
        custom_metrics: {
            let mut metrics = HashMap::new();
            metrics.insert("bleu_score".to_string(), vec![8.2, 15.7, 22.3, 28.1, 32.8]);
            metrics.insert("rouge_l".to_string(), vec![0.15, 0.28, 0.39, 0.47, 0.53]);
            metrics.insert("meteor".to_string(), vec![0.18, 0.31, 0.42, 0.51, 0.57]);
            metrics
        },
    };

    // Create comprehensive optimizer state for T5
    let optimizer_state = OptimizerState {
        optimizer_type: "AdaFactor".to_string(),
        learning_rate: 1e-4,
        momentum_states: {
            let mut states = HashMap::new();
            // Mock some parameter states
            states.insert("encoder.layer_0.self_attn.q_proj.weight".to_string(), vec![0.0; 1000]);
            states.insert("decoder.layer_0.self_attn.q_proj.weight".to_string(), vec![0.0; 1000]);
            states
        },
        velocity_states: HashMap::new(),
        step_counts: {
            let mut counts = HashMap::new();
            counts.insert("global".to_string(), 1000);
            counts
        },
        custom_state: HashMap::new(),
    };

    // Save checkpoint with full state
    let checkpoint_path = checkpoint_manager.save_checkpoint(
        &model,
        5,
        1000,
        Some(optimizer_state),
        Some(training_stats),
        Some({
            let mut custom = HashMap::new();
            custom.insert("task".to_string(), "en_to_de_translation".to_string());
            custom.insert("dataset".to_string(), "wmt14".to_string());
            custom.insert("tokenizer".to_string(), "sentencepiece".to_string());
            custom
        }),
    )?;

    println!("   💾 T5 checkpoint saved: {}", checkpoint_path.split('/').last().unwrap_or(""));

    // Demonstrate checkpoint validation
    let checkpoint = checkpoint_manager.load_checkpoint_from_file(&checkpoint_path)?;
    utils::validate_checkpoint_compatibility(&model, &checkpoint)?;
    println!("   ✅ Checkpoint validation passed");

    println!("   ✅ T5 checkpointing completed\n");

    Ok(())
}

/// Demonstrate training resumption capabilities
fn demonstrate_training_resumption(device: DeviceType) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔄 Training Resumption Demo");
    println!("============================\n");

    // Create models for training resumption simulation
    let config = TextModelConfig::bert_base();
    let mut original_model = BertForSequenceClassification::new(config.clone(), device)?;
    let mut resumed_model = BertForSequenceClassification::new(config, device)?;

    // Configure checkpoint manager
    let checkpoint_config = CheckpointConfig {
        save_dir: "./checkpoints/resumption".to_string(),
        filename_prefix: "training_resumption_test".to_string(),
        ..Default::default()
    };

    let mut checkpoint_manager = CheckpointManager::new(checkpoint_config)?;

    // Simulate training progress
    let mut training_stats = TrainingStats {
        train_losses: vec![0.8, 0.6, 0.4],
        val_losses: vec![0.85, 0.65, 0.45],
        train_accuracies: vec![0.6, 0.75, 0.85],
        val_accuracies: vec![0.58, 0.72, 0.82],
        learning_rates: vec![1e-4, 8e-5, 6e-5],
        epoch_durations: vec![150.0, 148.0, 146.0],
        custom_metrics: HashMap::new(),
    };

    // Save initial checkpoint (epoch 3)
    println!("💾 Saving training checkpoint at epoch 3...");
    let checkpoint_path = checkpoint_manager.save_checkpoint(
        &original_model,
        3,
        600,
        None,
        Some(training_stats.clone()),
        None,
    )?;

    // Simulate training interruption and resumption
    println!("⏸️  Simulating training interruption...");
    
    // Load checkpoint into new model (simulating training resumption)
    println!("▶️  Resuming training from checkpoint...");
    let loaded_checkpoint = checkpoint_manager.load_checkpoint(&mut resumed_model, &checkpoint_path)?;

    // Extract training state for resumption
    if let Some(stats) = loaded_checkpoint.training_stats {
        println!("   📊 Resumed training state:");
        println!("      Last epoch: {}", loaded_checkpoint.metadata.epoch.unwrap_or(0));
        println!("      Last step: {}", loaded_checkpoint.metadata.global_step.unwrap_or(0));
        println!("      Train loss: {:.4}", stats.train_losses.last().unwrap_or(&0.0));
        println!("      Val loss: {:.4}", stats.val_losses.last().unwrap_or(&0.0));
        println!("      Learning rate: {:.2e}", stats.learning_rates.last().unwrap_or(&0.0));
        
        // Simulate continuing training
        training_stats.train_losses.extend(vec![0.3, 0.25]);
        training_stats.val_losses.extend(vec![0.35, 0.3]);
        training_stats.train_accuracies.extend(vec![0.9, 0.92]);
        training_stats.val_accuracies.extend(vec![0.87, 0.89]);
        training_stats.learning_rates.extend(vec![4e-5, 2e-5]);
        training_stats.epoch_durations.extend(vec![144.0, 142.0]);
    }

    // Save continuation checkpoint
    println!("💾 Saving continued training checkpoint...");
    let _continued_checkpoint = checkpoint_manager.save_checkpoint(
        &resumed_model,
        5,
        1000,
        None,
        Some(training_stats),
        Some({
            let mut custom = HashMap::new();
            custom.insert("resumed_from".to_string(), checkpoint_path.clone());
            custom.insert("resumption_step".to_string(), "600".to_string());
            custom
        }),
    )?;

    // Verify models have the same parameters
    verify_model_parameters(&original_model, &resumed_model)?;
    println!("   ✅ Training resumption verified\n");

    Ok(())
}

/// Demonstrate best model tracking
fn demonstrate_best_model_tracking(device: DeviceType) -> Result<(), Box<dyn std::error::Error>> {
    println!("🏆 Best Model Tracking Demo");
    println!("============================\n");

    let config = TextModelConfig::gpt2_small();
    let mut model = GPTForCausalLM::new(config);

    // Configure for best model tracking
    let checkpoint_config = CheckpointConfig {
        save_dir: "./checkpoints/best_tracking".to_string(),
        filename_prefix: "best_model_demo".to_string(),
        save_best: true,
        best_metric_name: "perplexity".to_string(),
        higher_is_better: false, // Lower perplexity is better
        ..Default::default()
    };

    let mut checkpoint_manager = CheckpointManager::new(checkpoint_config)?;

    // Simulate training with varying performance
    let performance_scenarios = vec![
        (1, 45.2, "Initial training"),
        (2, 32.8, "Improving"),
        (3, 28.1, "Best performance"),
        (4, 31.2, "Slight degradation"),
        (5, 26.9, "New best!"),
        (6, 29.4, "Another degradation"),
    ];

    for (epoch, perplexity, description) in performance_scenarios {
        let training_stats = TrainingStats {
            train_losses: vec![perplexity.ln()],
            val_losses: vec![perplexity.ln() + 0.1],
            train_accuracies: vec![0.5],
            val_accuracies: vec![0.48],
            learning_rates: vec![1e-4],
            epoch_durations: vec![120.0],
            custom_metrics: {
                let mut metrics = HashMap::new();
                metrics.insert("perplexity".to_string(), vec![perplexity]);
                metrics
            },
        };

        let checkpoint_path = checkpoint_manager.save_checkpoint(
            &model,
            epoch,
            epoch * 100,
            None,
            Some(training_stats),
            None,
        )?;

        println!("   📊 Epoch {}: {} (perplexity: {:.1})", epoch, description, perplexity);
        
        if let Some(best_path) = checkpoint_manager.best_model_path() {
            if best_path == &checkpoint_path {
                println!("      🏆 New best model saved!");
            }
        }
    }

    // Show final best model
    if let Some(best_path) = checkpoint_manager.best_model_path() {
        println!("\n   🎯 Final best model: {}", best_path.split('/').last().unwrap_or(""));
        
        // Load and verify best model
        let best_checkpoint = checkpoint_manager.load_checkpoint_from_file(best_path)?;
        if let Some(stats) = &best_checkpoint.training_stats {
            if let Some(perplexity) = stats.custom_metrics.get("perplexity").and_then(|v| v.last()) {
                println!("      Best perplexity: {:.1}", perplexity);
            }
        }
    }

    println!("   ✅ Best model tracking completed\n");

    Ok(())
}

/// Demonstrate checkpoint utilities and analysis
fn demonstrate_checkpoint_utilities() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔧 Checkpoint Utilities Demo");
    println!("=============================\n");

    // Analyze all saved checkpoints
    let checkpoint_dirs = vec!["./checkpoints/gpt", "./checkpoints/bert", "./checkpoints/t5"];
    
    for dir in checkpoint_dirs {
        if std::path::Path::new(dir).exists() {
            println!("📂 Analyzing checkpoints in: {}", dir);
            
            if let Ok(entries) = std::fs::read_dir(dir) {
                let mut checkpoints = Vec::new();
                
                for entry in entries.flatten() {
                    if let Some(filename) = entry.file_name().to_str() {
                        if filename.ends_with(".torsh") {
                            checkpoints.push(entry.path());
                        }
                    }
                }
                
                checkpoints.sort();
                
                for checkpoint_path in checkpoints {
                    if let Ok(metadata) = std::fs::metadata(&checkpoint_path) {
                        let size_mb = metadata.len() as f64 / (1024.0 * 1024.0);
                        println!("   📄 {}: {:.2} MB", 
                                checkpoint_path.file_name().unwrap().to_string_lossy(), 
                                size_mb);
                    }
                }
                
                println!();
            }
        }
    }

    // Demonstrate state dict operations
    println!("🔍 State Dictionary Analysis:");
    let config = TextModelConfig::bert_base();
    let model = BertForSequenceClassification::new(config, DeviceType::Cpu)?;
    
    let state_dict = utils::model_to_state_dict(&model);
    println!("   📊 Parameter tensors: {}", state_dict.len());
    
    let total_params = utils::count_parameters(&model);
    println!("   🔢 Total parameters: {:.2}M", total_params as f32 / 1_000_000.0);
    
    let memory_estimate = utils::estimate_model_memory(&model);
    println!("   💾 Memory estimate: {:.2} MB", memory_estimate as f32 / (1024.0 * 1024.0));

    // Show largest parameters
    let mut param_sizes: Vec<_> = state_dict.iter()
        .map(|(name, data)| (name, data.len()))
        .collect();
    param_sizes.sort_by(|a, b| b.1.cmp(&a.1));
    
    println!("\n   🏗️  Largest parameters:");
    for (name, size) in param_sizes.iter().take(5) {
        println!("      {}: {} elements", name, size);
    }

    println!("\n   ✅ Checkpoint utilities demo completed");

    Ok(())
}

/// Verify that two models have identical parameters
fn verify_model_parameters<M1: Module, M2: Module>(
    model1: &M1,
    model2: &M2,
) -> Result<(), Box<dyn std::error::Error>> {
    let params1 = model1.named_parameters();
    let params2 = model2.named_parameters();

    if params1.len() != params2.len() {
        return Err("Parameter count mismatch".into());
    }

    for (name1, param1) in params1 {
        if let Some(param2) = params2.get(&name1) {
            let data1 = param1.data();
            let data2 = param2.data();
            
            if data1.len() != data2.len() {
                return Err(format!("Parameter size mismatch for {}", name1).into());
            }
            
            for (i, (&val1, &val2)) in data1.iter().zip(data2.iter()).enumerate() {
                if (val1 - val2).abs() > 1e-6 {
                    return Err(format!("Parameter value mismatch for {} at index {}", name1, i).into());
                }
            }
        } else {
            return Err(format!("Parameter {} not found in second model", name1).into());
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint_workflow() {
        let device = DeviceType::Cpu;
        let config = TextModelConfig::bert_base();
        let model = BertForSequenceClassification::new(config, device).unwrap();

        // Test state dict extraction
        let state_dict = utils::model_to_state_dict(&model);
        assert!(!state_dict.is_empty());

        // Test parameter counting
        let param_count = utils::count_parameters(&model);
        assert!(param_count > 100_000_000); // BERT Base should have > 100M params

        // Test memory estimation
        let memory = utils::estimate_model_memory(&model);
        assert!(memory > 0);
    }
}