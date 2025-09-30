//! Tests for Distributed Data Parallel

use std::sync::Arc;
use torsh_core::Result;
use torsh_distributed::{backend::BackendType, ddp::DistributedDataParallel, init_process_group};
use torsh_nn::container::Sequential;
use torsh_nn::layers::activation::ReLU;
use torsh_nn::layers::linear::Linear;
use torsh_nn::Module;
use torsh_tensor::creation::randn;
use torsh_tensor::Tensor;

/// Simple test model
struct TestModel {
    fc1: Linear,
    fc2: Linear,
}

impl TestModel {
    fn new() -> Self {
        let mut model = Self {
            fc1: Linear::new(10, 20, true),
            fc2: Linear::new(20, 5, true),
        };

        // Enable gradients for all parameters
        for (_name, param) in model.named_parameters() {
            let tensor = param.tensor();
            let tensor_guard = tensor.read();

            // Create a new tensor with requires_grad=true and replace the parameter
            let new_tensor = tensor_guard.clone().requires_grad_(true);
            drop(tensor_guard);

            let mut tensor_guard = tensor.write();
            *tensor_guard = new_tensor;
        }

        model
    }
}

impl Module for TestModel {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let x = self.fc1.forward(input)?;
        let x = x.relu()?;
        self.fc2.forward(&x)
    }

    fn parameters(&self) -> std::collections::HashMap<String, torsh_nn::Parameter> {
        // Use named_parameters to avoid key conflicts
        self.named_parameters()
    }

    fn named_parameters(&self) -> std::collections::HashMap<String, torsh_nn::Parameter> {
        let mut params = std::collections::HashMap::new();

        // Add fc1 parameters
        for (name, param) in self.fc1.named_parameters() {
            params.insert(format!("fc1.{}", name), param);
        }

        // Add fc2 parameters
        for (name, param) in self.fc2.named_parameters() {
            params.insert(format!("fc2.{}", name), param);
        }

        params
    }

    fn training(&self) -> bool {
        self.fc1.training()
    }

    fn train(&mut self) {
        self.fc1.train();
        self.fc2.train();
    }

    fn eval(&mut self) {
        self.fc1.eval();
        self.fc2.eval();
    }

    fn to_device(&mut self, device: torsh_core::DeviceType) -> Result<()> {
        self.fc1.to_device(device)?;
        self.fc2.to_device(device)?;
        Ok(())
    }
}

#[test]
fn test_ddp_creation() -> Result<()> {
    let pg = Arc::new(init_process_group(
        BackendType::Gloo,
        0,
        4,
        "127.0.0.1",
        29500,
    )?);

    let model = TestModel::new();

    let ddp = DistributedDataParallel::new(
        model,
        pg,
        vec![0], // device_ids
        Some(0), // output_device
        true,    // broadcast_buffers
        25.0,    // bucket_cap_mb
    )?;

    // Verify DDP wrapper preserves module properties
    assert!(ddp.training());
    assert_eq!(ddp.parameters().len(), 4); // 2 weights + 2 biases

    Ok(())
}

#[test]
fn test_ddp_forward() -> Result<()> {
    let pg = Arc::new(init_process_group(
        BackendType::Gloo,
        0,
        4,
        "127.0.0.1",
        29500,
    )?);

    let model = TestModel::new();
    let ddp = DistributedDataParallel::new(model, pg, vec![0], None, false, 25.0)?;

    // Test forward pass
    let input = randn::<f32>(&[32, 10]);
    let output = ddp.forward(&input)?;

    assert_eq!(output.shape().dims(), &[32, 5]);

    Ok(())
}

#[tokio::test]
async fn test_ddp_sync_gradients() -> Result<()> {
    let pg = Arc::new(init_process_group(
        BackendType::Gloo,
        0,
        4,
        "127.0.0.1",
        29500,
    )?);

    let model = TestModel::new();
    let mut ddp = DistributedDataParallel::new(model, pg, vec![0], None, true, 25.0)?;

    // Test that we can sync gradients (even if no gradients are present)
    ddp.sync_gradients().await?;

    // Test gradient statistics
    let stats = ddp.get_sync_stats();
    assert!(stats.total_parameters > 0);

    // Test bucket information
    let bucket_info = ddp.get_bucket_info();
    assert!(!bucket_info.is_empty());

    // Test has_gradients (should be false since we haven't done backward pass)
    assert!(!ddp.has_gradients());

    // Test zero_grad (should not fail even with no gradients)
    ddp.zero_grad()?;

    Ok(())
}

#[tokio::test]
async fn test_ddp_bucket_configuration() -> Result<()> {
    let pg = Arc::new(init_process_group(
        BackendType::Gloo,
        0,
        2,
        "127.0.0.1",
        29500,
    )?);

    let model = TestModel::new();
    let custom_bucket_config = torsh_distributed::BucketConfig {
        max_bucket_size_mb: 10.0,
        enabled: true,
        min_bucket_size_mb: 0.5,
    };

    let ddp = DistributedDataParallel::new_with_bucket_config(
        model,
        pg,
        vec![0],
        None,
        true,
        custom_bucket_config,
    )?;

    // Verify bucket configuration was applied
    let bucket_info = ddp.get_bucket_info();
    for bucket in bucket_info {
        assert!(bucket.size_mb <= 10.0);
        println!(
            "Bucket {}: {:.2} MB with {} parameters",
            bucket.index, bucket.size_mb, bucket.num_parameters
        );
    }

    Ok(())
}

#[tokio::test]
async fn test_ddp_gradient_consistency_check() -> Result<()> {
    let pg = Arc::new(init_process_group(
        BackendType::Gloo,
        0,
        2,
        "127.0.0.1",
        29500,
    )?);

    let model = TestModel::new();
    let ddp = DistributedDataParallel::new(model, pg, vec![0], None, true, 25.0)?;

    // Test gradient consistency check
    let is_consistent = ddp.check_gradient_consistency().await?;
    assert!(is_consistent); // Should always be true for mock implementation

    Ok(())
}

#[test]
fn test_ddp_train_eval_modes() -> Result<()> {
    let pg = Arc::new(init_process_group(
        BackendType::Gloo,
        0,
        4,
        "127.0.0.1",
        29500,
    )?);

    let model = TestModel::new();
    let mut ddp = DistributedDataParallel::new(model, pg, vec![0], None, true, 25.0)?;

    // Test train mode
    ddp.train();
    assert!(ddp.training());

    // Test eval mode
    ddp.eval();
    assert!(!ddp.training());

    Ok(())
}

#[test]
fn test_sequential_model_ddp() -> Result<()> {
    let pg = Arc::new(init_process_group(
        BackendType::Gloo,
        0,
        2,
        "127.0.0.1",
        29500,
    )?);

    // Create a sequential model
    let model = Sequential::new()
        .add(Linear::new(784, 256, true))
        .add(ReLU::new())
        .add(Linear::new(256, 10, true));

    let ddp = DistributedDataParallel::new(model, pg, vec![0], None, true, 25.0)?;

    // Test forward pass
    let input = randn::<f32>(&[16, 784]);
    let output = ddp.forward(&input)?;

    assert_eq!(output.shape().dims(), &[16, 10]);

    Ok(())
}
