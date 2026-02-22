//! Advanced Distributed Training with Gradient Synchronization
//!
//! This example demonstrates the enhanced gradient synchronization capabilities
//! of ToRSh's distributed training system, including:
//! - Proper gradient synchronization (not just parameter synchronization)
//! - Gradient bucketing for communication efficiency
//! - Gradient statistics and monitoring
//! - Gradient consistency checking for debugging

use std::error::Error;
use std::sync::Arc;
use torsh_core::{DeviceType, Result as TorshResult, TorshError};
use torsh_distributed::{
    init_process_group, BackendType, BucketConfig, BucketInfo, DistributedDataParallel,
    GradientSyncStats,
};
use torsh_nn::prelude::{Linear, Module, ReLU, Sequential};
// Note: Optimizer functionality simplified for this demo
use torsh_tensor::{
    creation::{randint, randn},
    Tensor,
};

/// Enhanced training model for gradient synchronization demo
struct AdvancedModel {
    feature_extractor: Sequential,
    classifier: Linear,
}

impl AdvancedModel {
    fn new(input_size: usize, hidden_size: usize, num_classes: usize) -> Self {
        let mut model = Self {
            feature_extractor: Sequential::new()
                .add(Linear::new(input_size, hidden_size, true))
                .add(ReLU::new())
                .add(Linear::new(hidden_size, hidden_size, true))
                .add(ReLU::new())
                .add(Linear::new(hidden_size, hidden_size, true))
                .add(ReLU::new()),
            classifier: Linear::new(hidden_size, num_classes, true),
        };

        // Enable gradients for all parameters
        for (_name, param) in model.named_parameters() {
            let tensor = param.tensor();
            let tensor_guard = tensor.read();
            let new_tensor = tensor_guard.clone().requires_grad_(true);
            drop(tensor_guard);
            let mut tensor_guard = tensor.write();
            *tensor_guard = new_tensor;
        }

        model
    }
}

impl Module for AdvancedModel {
    fn forward(&self, input: &Tensor) -> TorshResult<Tensor> {
        let features = self.feature_extractor.forward(input)?;
        self.classifier.forward(&features)
    }

    fn parameters(&self) -> std::collections::HashMap<String, torsh_nn::Parameter> {
        self.named_parameters()
    }

    fn named_parameters(&self) -> std::collections::HashMap<String, torsh_nn::Parameter> {
        let mut params = std::collections::HashMap::new();

        // Add feature extractor parameters
        for (name, param) in self.feature_extractor.named_parameters() {
            params.insert(format!("feature_extractor.{}", name), param);
        }

        // Add classifier parameters
        for (name, param) in self.classifier.named_parameters() {
            params.insert(format!("classifier.{}", name), param);
        }

        params
    }

    fn training(&self) -> bool {
        self.feature_extractor.training()
    }

    fn train(&mut self) {
        self.feature_extractor.train();
        self.classifier.train();
    }

    fn eval(&mut self) {
        self.feature_extractor.eval();
        self.classifier.eval();
    }

    fn to_device(&mut self, device: torsh_core::DeviceType) -> TorshResult<()> {
        self.feature_extractor.to_device(device)?;
        self.classifier.to_device(device)?;
        Ok(())
    }
}

/// Demonstrates gradient synchronization with custom bucket configuration
async fn demo_gradient_bucketing(rank: usize, world_size: usize) -> Result<(), Box<dyn Error>> {
    println!("\n🪣 Gradient Bucketing Demo (Rank {})", rank);
    println!("========================================");

    // Initialize process group
    let process_group = Arc::new(init_process_group(
        BackendType::Gloo,
        rank as u32,
        world_size as u32,
        "127.0.0.1",
        29500,
    )?);

    // Create model
    let model = AdvancedModel::new(784, 512, 10);

    // Custom bucket configuration for optimal communication
    let bucket_config = BucketConfig {
        max_bucket_size_mb: 15.0, // Smaller buckets for this demo
        enabled: true,
        min_bucket_size_mb: 1.0,
    };

    let mut ddp = DistributedDataParallel::new_with_bucket_config(
        model,
        process_group.clone(),
        vec![rank],
        Some(rank),
        true,
        bucket_config,
    )?;

    // Display bucket information
    let bucket_info = ddp.get_bucket_info();
    println!("📊 Bucket Configuration:");
    for bucket in &bucket_info {
        println!(
            "  Bucket {}: {:.2} MB, {} parameters",
            bucket.index, bucket.size_mb, bucket.num_parameters
        );

        // Show first few parameter names for illustration
        let param_preview: Vec<_> = bucket.parameter_names.iter().take(3).collect();
        println!("    Parameters: {:?}...", param_preview);
    }

    // Get initial gradient statistics
    let initial_stats = ddp.get_sync_stats();
    println!("\n📈 Initial Statistics:");
    println!(
        "  Total parameters requiring gradients: {}",
        initial_stats.total_parameters
    );
    println!(
        "  Parameters with gradients: {}",
        initial_stats.parameters_with_grad
    );
    println!(
        "  Total gradient size: {:.2} MB",
        initial_stats.total_gradient_size_mb
    );
    println!("  Number of buckets: {}", initial_stats.num_buckets);
    println!("  World size: {}", initial_stats.world_size);

    // Simulate forward and backward pass
    println!("\n🔄 Simulating Training Step...");

    // Create dummy input and target
    let input = randn(&[32, 784]);
    let target = randint(0, 10, &[32]);

    // Forward pass
    let output = ddp.forward(&input)?;
    println!(
        "  Forward pass completed: output shape {:?}",
        output.shape().dims()
    );

    // Simulate loss computation (normally done with a loss function)
    let loss = output.mean(None, false)?;
    println!("  Loss: {:.4}", loss.item());

    // TODO: Backward pass would go here when autograd is fully integrated
    // loss.backward()?;

    // For demonstration, let's manually create some fake gradients
    println!("  Creating simulated gradients...");
    for (_name, param) in ddp.parameters() {
        let tensor = param.tensor();
        let tensor_guard = tensor.read();

        if tensor_guard.requires_grad() {
            // Create a fake gradient with some random values
            let fake_grad = randn(tensor_guard.shape().dims());
            tensor_guard.set_grad(Some(fake_grad));
        }
    }

    // Check gradient availability
    let has_grads = ddp.has_gradients();
    println!("  Has gradients: {}", has_grads);

    // Synchronize gradients
    if has_grads {
        println!(
            "  Synchronizing gradients across {} processes...",
            world_size
        );
        ddp.sync_gradients().await?;
        println!("  ✅ Gradient synchronization completed");

        // Check gradient consistency
        let is_consistent = ddp.check_gradient_consistency().await?;
        println!(
            "  Gradient consistency check: {}",
            if is_consistent {
                "✅ PASS"
            } else {
                "❌ FAIL"
            }
        );

        // Get updated statistics
        let final_stats = ddp.get_sync_stats();
        println!("\n📊 Post-Sync Statistics:");
        println!(
            "  Parameters with gradients: {}",
            final_stats.parameters_with_grad
        );
        println!(
            "  Total gradient size: {:.2} MB",
            final_stats.total_gradient_size_mb
        );
    }

    // Zero gradients (cleanup)
    ddp.zero_grad()?;
    println!("  Gradients zeroed for next iteration");

    Ok(())
}

/// Demonstrates different bucket configurations and their impact
async fn compare_bucket_strategies() -> Result<(), Box<dyn Error>> {
    println!("\n⚖️  Bucket Strategy Comparison");
    println!("==============================");

    let process_group = Arc::new(init_process_group(
        BackendType::Gloo,
        0,
        2,
        "127.0.0.1",
        29500,
    )?);

    let model = AdvancedModel::new(784, 256, 10);

    // Strategy 1: Large buckets (better for high bandwidth)
    let large_bucket_config = BucketConfig {
        max_bucket_size_mb: 50.0,
        enabled: true,
        min_bucket_size_mb: 1.0,
    };

    let ddp_large = DistributedDataParallel::new_with_bucket_config(
        AdvancedModel::new(784, 256, 10), // Create new model instance
        process_group.clone(),
        vec![0],
        None,
        true,
        large_bucket_config,
    )?;

    let large_buckets = ddp_large.get_bucket_info();
    println!("🏗️  Large Bucket Strategy:");
    println!("  Number of buckets: {}", large_buckets.len());
    for bucket in &large_buckets {
        println!("    Bucket {}: {:.2} MB", bucket.index, bucket.size_mb);
    }

    // Strategy 2: Small buckets (better for low bandwidth/latency)
    let small_bucket_config = BucketConfig {
        max_bucket_size_mb: 5.0,
        enabled: true,
        min_bucket_size_mb: 0.5,
    };

    let model2 = AdvancedModel::new(784, 256, 10);
    let ddp_small = DistributedDataParallel::new_with_bucket_config(
        model2,
        process_group.clone(),
        vec![0],
        None,
        true,
        small_bucket_config,
    )?;

    let small_buckets = ddp_small.get_bucket_info();
    println!("\n🧱 Small Bucket Strategy:");
    println!("  Number of buckets: {}", small_buckets.len());
    for bucket in &small_buckets {
        println!("    Bucket {}: {:.2} MB", bucket.index, bucket.size_mb);
    }

    // Strategy 3: Disabled bucketing (naive approach)
    let no_bucket_config = BucketConfig {
        max_bucket_size_mb: 1.0,
        enabled: false,
        min_bucket_size_mb: 0.1,
    };

    let model3 = AdvancedModel::new(784, 256, 10);
    let ddp_none = DistributedDataParallel::new_with_bucket_config(
        model3,
        process_group,
        vec![0],
        None,
        true,
        no_bucket_config,
    )?;

    let no_buckets = ddp_none.get_bucket_info();
    println!("\n🚫 No Bucketing Strategy:");
    println!("  Number of buckets: {}", no_buckets.len());
    println!("  Synchronization will be parameter-by-parameter");

    println!("\n💡 Recommendations:");
    println!("  - Large buckets: Use for high-bandwidth networks (InfiniBand, 100GbE)");
    println!("  - Small buckets: Use for commodity networks or high-latency connections");
    println!("  - No bucketing: Use for debugging or very small models");

    Ok(())
}

/// Demonstrates gradient monitoring and statistics
async fn demo_gradient_monitoring(rank: usize) -> Result<(), Box<dyn Error>> {
    println!("\n📊 Gradient Monitoring Demo (Rank {})", rank);
    println!("=====================================");

    let process_group = Arc::new(init_process_group(
        BackendType::Gloo,
        rank as u32,
        4,
        "127.0.0.1",
        29500,
    )?);

    let model = AdvancedModel::new(784, 128, 10);
    let mut ddp = DistributedDataParallel::new(model, process_group, vec![rank], None, true, 25.0)?;

    // Simulate multiple training steps with gradient monitoring
    for step in 1..=3 {
        println!("\n📝 Training Step {}", step);
        println!("  Creating simulated gradients...");

        // Simulate gradients with different characteristics
        for (name, param) in ddp.parameters() {
            let tensor = param.tensor();
            let tensor_guard = tensor.read();

            if tensor_guard.requires_grad() {
                // Create gradient with step-dependent characteristics
                let scale = 0.1 / (step as f32).sqrt(); // Decreasing gradient magnitude
                let grad = randn(tensor_guard.shape().dims()).mul_scalar(scale)?;
                tensor_guard.set_grad(Some(grad));
            }
        }

        // Get statistics before sync
        let pre_sync_stats = ddp.get_sync_stats();
        println!(
            "  Pre-sync: {}/{} parameters have gradients ({:.2} MB total)",
            pre_sync_stats.parameters_with_grad,
            pre_sync_stats.total_parameters,
            pre_sync_stats.total_gradient_size_mb
        );

        // Synchronize gradients
        ddp.sync_gradients().await?;

        // Get statistics after sync
        let post_sync_stats = ddp.get_sync_stats();
        println!(
            "  Post-sync: {}/{} parameters have gradients ({:.2} MB total)",
            post_sync_stats.parameters_with_grad,
            post_sync_stats.total_parameters,
            post_sync_stats.total_gradient_size_mb
        );

        // Simulate optimizer step (zero gradients)
        ddp.zero_grad()?;
        println!("  ✅ Gradients cleared for next step");
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("🚀 ToRSh Advanced Distributed Training Demo");
    println!("==========================================");

    // Demo 1: Gradient bucketing with different configurations
    demo_gradient_bucketing(0, 4).await?;

    // Demo 2: Compare different bucketing strategies
    compare_bucket_strategies().await?;

    // Demo 3: Gradient monitoring across training steps
    demo_gradient_monitoring(0).await?;

    println!("\n🏁 Demo Completed Successfully!");
    println!("\n📚 Key Features Demonstrated:");
    println!("  ✅ Proper gradient synchronization (not just parameters)");
    println!("  ✅ Gradient bucketing for communication efficiency");
    println!("  ✅ Configurable bucket strategies for different network types");
    println!("  ✅ Gradient statistics and monitoring for debugging");
    println!("  ✅ Gradient consistency checking");
    println!("  ✅ Zero-gradient functionality for optimizer integration");

    println!("\n🔄 Next Steps for Production:");
    println!("  - Integrate with actual autograd backward pass");
    println!("  - Add gradient compression for slow networks");
    println!("  - Implement overlap of computation and communication");
    println!("  - Add automatic bucket tuning based on network characteristics");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_advanced_model_creation() -> Result<(), Box<dyn Error>> {
        let model = AdvancedModel::new(784, 128, 10);

        // Verify all parameters require gradients
        let params = model.named_parameters();
        assert!(!params.is_empty());

        for (_name, param) in params {
            let tensor = param.tensor();
            let tensor_guard = tensor.read();
            assert!(tensor_guard.requires_grad());
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_bucket_configuration() -> Result<(), Box<dyn Error>> {
        let process_group = Arc::new(init_process_group(
            BackendType::Gloo,
            0,
            2,
            "127.0.0.1",
            29500,
        )?);

        let model = AdvancedModel::new(100, 50, 5);

        let bucket_config = BucketConfig {
            max_bucket_size_mb: 1.0,
            enabled: true,
            min_bucket_size_mb: 0.1,
        };

        let ddp = DistributedDataParallel::new_with_bucket_config(
            model,
            process_group,
            vec![0],
            None,
            true,
            bucket_config,
        )?;

        let bucket_info = ddp.get_bucket_info();
        assert!(!bucket_info.is_empty());

        // Verify bucket sizes are within limits
        for bucket in bucket_info {
            assert!(bucket.size_mb <= 1.0);
            assert!(bucket.num_parameters > 0);
        }

        Ok(())
    }
}
