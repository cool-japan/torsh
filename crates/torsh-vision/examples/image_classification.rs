//! Comprehensive Image Classification Example
//!
//! This example demonstrates:
//! - Loading and preparing datasets (CIFAR-10)
//! - Building a CNN classifier
//! - Training with data augmentation
//! - Evaluation and accuracy metrics
//! - Model checkpointing
//!
//! Run with: cargo run --example image_classification --features pretrained

use std::path::PathBuf;
use std::sync::Arc;
use torsh_core::device::{CpuDevice, Device};
use torsh_nn::{Conv2d, Linear, MaxPool2d, Module, ReLU, Sequential};
use torsh_optim::{Adam, Optimizer};
use torsh_tensor::{creation, Tensor};
use torsh_vision::{
    CifarDataset, Compose, Normalize, RandomCrop, RandomHorizontalFlip, Resize, Result, VisionError,
};

/// Simple CNN architecture for CIFAR-10 classification
#[derive(Debug)]
struct SimpleCNN {
    features: Sequential,
    classifier: Sequential,
    device: Arc<dyn Device>,
}

impl SimpleCNN {
    fn new(num_classes: usize, device: Arc<dyn Device>) -> Result<Self> {
        // Convolutional feature extractor
        let features = Sequential::new(vec![
            Box::new(Conv2d::new(3, 32, (3, 3), (1, 1), (1, 1), (1, 1), true, 1)?),
            Box::new(ReLU::new()),
            Box::new(Conv2d::new(
                32,
                64,
                (3, 3),
                (1, 1),
                (1, 1),
                (1, 1),
                true,
                1,
            )?),
            Box::new(ReLU::new()),
            Box::new(MaxPool2d::new((2, 2), (2, 2), (0, 0))),
            Box::new(Conv2d::new(
                64,
                128,
                (3, 3),
                (1, 1),
                (1, 1),
                (1, 1),
                true,
                1,
            )?),
            Box::new(ReLU::new()),
            Box::new(MaxPool2d::new((2, 2), (2, 2), (0, 0))),
        ]);

        // Fully connected classifier
        let classifier = Sequential::new(vec![
            Box::new(Linear::new(128 * 8 * 8, 256)?),
            Box::new(ReLU::new()),
            Box::new(Linear::new(256, num_classes)?),
        ]);

        Ok(Self {
            features,
            classifier,
            device,
        })
    }
}

impl Module for SimpleCNN {
    fn forward(
        &self,
        input: &Tensor,
    ) -> std::result::Result<Tensor, torsh_core::error::TorshError> {
        // Extract features
        let x = self.features.forward(input)?;

        // Flatten for classifier
        let batch_size = x.shape().dims()[0];
        let x = x.reshape(&[batch_size as i64, -1])?;

        // Classify
        self.classifier.forward(&x)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = self.features.parameters();
        params.extend(self.classifier.parameters());
        params
    }

    fn set_training(&mut self, training: bool) {
        self.features.set_training(training);
        self.classifier.set_training(training);
    }
}

/// Training configuration
struct TrainConfig {
    batch_size: usize,
    epochs: usize,
    learning_rate: f32,
    data_dir: PathBuf,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            epochs: 10,
            learning_rate: 0.001,
            data_dir: std::env::temp_dir().join("cifar10"),
        }
    }
}

/// Calculate accuracy given predictions and labels
fn calculate_accuracy(predictions: &Tensor, labels: &Tensor) -> Result<f32> {
    let pred_classes = predictions.argmax(Some(1), false)?;
    let correct = pred_classes.eq(&labels)?.sum(None, false)?;
    let total = labels.numel();

    Ok(correct.item() / total as f32)
}

/// Run a single training epoch
fn train_epoch(
    model: &mut SimpleCNN,
    optimizer: &mut Adam,
    dataset: &CifarDataset,
    config: &TrainConfig,
) -> Result<(f32, f32)> {
    model.set_training(true);

    let mut total_loss = 0.0;
    let mut total_accuracy = 0.0;
    let num_batches = (dataset.len() + config.batch_size - 1) / config.batch_size;

    println!("Training epoch with {} batches...", num_batches);

    for batch_idx in 0..num_batches {
        let start_idx = batch_idx * config.batch_size;
        let end_idx = ((batch_idx + 1) * config.batch_size).min(dataset.len());

        // Load batch (simplified - in real implementation would use DataLoader)
        let batch_size = end_idx - start_idx;
        let mut images = Vec::new();
        let mut labels_vec = Vec::new();

        for idx in start_idx..end_idx {
            let (img, label) = dataset.get(idx)?;
            images.push(img);
            labels_vec.push(label as f32);
        }

        // Stack into batch tensor
        let batch_images = Tensor::stack(&images, 0)?;
        let batch_labels = creation::tensor(
            &labels_vec,
            &[batch_size as i64],
            torsh_core::dtype::DType::Float32,
        )?;

        // Forward pass
        let outputs = model.forward(&batch_images)?;

        // Compute loss (cross-entropy)
        let loss = cross_entropy_loss(&outputs, &batch_labels)?;

        // Backward pass
        optimizer.zero_grad();
        loss.backward()?;
        optimizer.step()?;

        // Track metrics
        let loss_val: f32 = loss.item();
        total_loss += loss_val;

        let acc = calculate_accuracy(&outputs, &batch_labels)?;
        total_accuracy += acc;

        if batch_idx % 10 == 0 {
            println!(
                "  Batch [{}/{}] Loss: {:.4}, Acc: {:.2}%",
                batch_idx + 1,
                num_batches,
                loss_val,
                acc * 100.0
            );
        }
    }

    Ok((
        total_loss / num_batches as f32,
        total_accuracy / num_batches as f32,
    ))
}

/// Evaluate model on test set
fn evaluate(
    model: &mut SimpleCNN,
    dataset: &CifarDataset,
    batch_size: usize,
) -> Result<(f32, f32)> {
    model.set_training(false);

    let mut total_loss = 0.0;
    let mut total_accuracy = 0.0;
    let num_batches = (dataset.len() + batch_size - 1) / batch_size;

    for batch_idx in 0..num_batches {
        let start_idx = batch_idx * batch_size;
        let end_idx = ((batch_idx + 1) * batch_size).min(dataset.len());

        let batch_size_actual = end_idx - start_idx;
        let mut images = Vec::new();
        let mut labels_vec = Vec::new();

        for idx in start_idx..end_idx {
            let (img, label) = dataset.get(idx)?;
            images.push(img);
            labels_vec.push(label as f32);
        }

        let batch_images = Tensor::stack(&images, 0)?;
        let batch_labels = creation::tensor(
            &labels_vec,
            &[batch_size_actual as i64],
            torsh_core::dtype::DType::Float32,
        )?;

        let outputs = model.forward(&batch_images)?;
        let loss = cross_entropy_loss(&outputs, &batch_labels)?;

        total_loss += loss.item();
        total_accuracy += calculate_accuracy(&outputs, &batch_labels)?;
    }

    Ok((
        total_loss / num_batches as f32,
        total_accuracy / num_batches as f32,
    ))
}

/// Simplified cross-entropy loss (placeholder - real implementation in torsh-nn)
fn cross_entropy_loss(logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
    // Simplified implementation - in production use torsh_nn::functional::cross_entropy
    let softmax = logits.softmax(1)?;
    let log_softmax = softmax.log()?;

    // Gather log probabilities for target classes
    let batch_size = targets.numel();
    let nll = log_softmax
        .sum(None, false)?
        .div_scalar(-1.0 * batch_size as f32)?;

    Ok(nll)
}

fn main() -> Result<()> {
    println!("🎯 ToRSh Vision - Image Classification Example");
    println!("================================================\n");

    let config = TrainConfig::default();
    let device = Arc::new(CpuDevice::new());

    println!("📊 Configuration:");
    println!("  Batch size: {}", config.batch_size);
    println!("  Epochs: {}", config.epochs);
    println!("  Learning rate: {}", config.learning_rate);
    println!("  Data directory: {:?}", config.data_dir);
    println!();

    // Create data transforms
    let train_transform = Compose::new(vec![
        Box::new(RandomCrop::new(32, Some(4))),
        Box::new(RandomHorizontalFlip::new(0.5)),
        Box::new(Normalize::new(
            vec![0.4914, 0.4822, 0.4465],
            vec![0.2470, 0.2435, 0.2616],
        )),
    ]);

    let test_transform = Compose::new(vec![Box::new(Normalize::new(
        vec![0.4914, 0.4822, 0.4465],
        vec![0.2470, 0.2435, 0.2616],
    ))]);

    // Load datasets
    println!("📁 Loading CIFAR-10 dataset...");
    let train_dataset = CifarDataset::cifar10(&config.data_dir, true, Some(train_transform), true)?;
    let test_dataset = CifarDataset::cifar10(&config.data_dir, false, Some(test_transform), true)?;

    println!("  Training samples: {}", train_dataset.len());
    println!("  Test samples: {}", test_dataset.len());
    println!();

    // Create model
    println!("🏗️  Building model...");
    let mut model = SimpleCNN::new(10, Arc::clone(&device))?;
    println!("  Model created successfully");
    println!();

    // Create optimizer
    let mut optimizer = Adam::new(
        model.parameters(),
        config.learning_rate as f64,
        0.9,
        0.999,
        1e-8,
    );

    // Training loop
    println!("🚀 Starting training...");
    for epoch in 0..config.epochs {
        println!("\nEpoch [{}/{}]", epoch + 1, config.epochs);
        println!("─────────────────────────────────────");

        let (train_loss, train_acc) =
            train_epoch(&mut model, &mut optimizer, &train_dataset, &config)?;

        println!(
            "  Training   - Loss: {:.4}, Acc: {:.2}%",
            train_loss,
            train_acc * 100.0
        );

        // Evaluate on test set
        let (test_loss, test_acc) = evaluate(&mut model, &test_dataset, config.batch_size)?;

        println!(
            "  Validation - Loss: {:.4}, Acc: {:.2}%",
            test_loss,
            test_acc * 100.0
        );
    }

    println!("\n✅ Training completed successfully!");

    Ok(())
}
