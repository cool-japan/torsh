//! Advanced Optimizers and Learning Rate Scheduling Demo
//!
//! This example demonstrates the complete optimizer suite in ToRSh including:
//! - SGD with momentum and Nesterov acceleration
//! - RMSprop with centered and momentum variants
//! - AdaGrad with learning rate decay
//! - Adam and AdamW (already working)
//! - Learning rate scheduling integration
//! - Performance comparison and best practices

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use parking_lot::RwLock;
use torsh_core::device::DeviceType;
use torsh_nn::{layers::Linear, Module};
use torsh_optim::{
    Adam, AdamBuilder, AdaGrad, AdaGradBuilder, RMSprop, RMSpropBuilder, 
    SGD, SGDBuilder, Optimizer, lr_scheduler::*
};
use torsh_tensor::creation::*;
use torsh_tensor::Tensor;

/// Configuration for optimizer comparison experiments
#[derive(Debug, Clone)]
pub struct ExperimentConfig {
    pub input_size: usize,
    pub hidden_size: usize,
    pub output_size: usize,
    pub num_epochs: usize,
    pub batch_size: usize,
    pub num_batches: usize,
    pub learning_rate: f32,
}

impl Default for ExperimentConfig {
    fn default() -> Self {
        Self {
            input_size: 100,
            hidden_size: 50,
            output_size: 10,
            num_epochs: 20,
            batch_size: 32,
            num_batches: 50,
            learning_rate: 0.01,
        }
    }
}

/// Results from running an optimizer experiment
#[derive(Debug, Clone)]
pub struct ExperimentResults {
    pub optimizer_name: String,
    pub final_loss: f64,
    pub losses: Vec<f64>,
    pub learning_rates: Vec<f32>,
    pub total_time: f64,
    pub convergence_epoch: Option<usize>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Advanced Optimizers and Learning Rate Scheduling Demo");
    println!("=========================================================\n");

    let config = ExperimentConfig::default();
    
    // Demonstrate individual optimizers
    demonstrate_sgd_variants(&config)?;
    demonstrate_rmsprop_variants(&config)?;
    demonstrate_adagrad_variants(&config)?;
    demonstrate_adam_variants(&config)?;
    
    // Demonstrate learning rate scheduling
    demonstrate_lr_scheduling(&config)?;
    
    // Performance comparison
    run_optimizer_comparison(&config)?;
    
    // Best practices and recommendations
    demonstrate_best_practices()?;

    println!("\n✅ Advanced optimizers demonstration completed!");
    Ok(())
}

/// Demonstrate SGD with various configurations
fn demonstrate_sgd_variants(config: &ExperimentConfig) -> Result<(), Box<dyn std::error::Error>> {
    println!("📈 SGD Optimizer Variants Demo");
    println!("===============================\n");

    let device = DeviceType::Cpu;

    // Test different SGD configurations
    let sgd_configs = vec![
        ("Standard SGD", SGDBuilder::new(config.learning_rate)),
        ("SGD with Momentum", SGDBuilder::new(config.learning_rate).momentum(0.9)),
        ("SGD with Nesterov", SGDBuilder::new(config.learning_rate).momentum(0.9).nesterov(true)),
        ("SGD with Weight Decay", SGDBuilder::new(config.learning_rate).momentum(0.9).weight_decay(1e-4)),
    ];

    for (name, builder) in sgd_configs {
        println!("🔧 Testing: {}", name);
        let result = run_optimizer_experiment(name, |params| Box::new(builder.clone().build(params)), config)?;
        print_experiment_summary(&result);
        println!();
    }

    Ok(())
}

/// Demonstrate RMSprop with various configurations
fn demonstrate_rmsprop_variants(config: &ExperimentConfig) -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 RMSprop Optimizer Variants Demo");
    println!("===================================\n");

    // Test different RMSprop configurations
    let rmsprop_configs = vec![
        ("Standard RMSprop", RMSpropBuilder::new().lr(config.learning_rate)),
        ("RMSprop with Momentum", RMSpropBuilder::new().lr(config.learning_rate).momentum(0.9)),
        ("Centered RMSprop", RMSpropBuilder::new().lr(config.learning_rate).centered(true)),
        ("RMSprop + Momentum + Centered", RMSpropBuilder::new().lr(config.learning_rate).momentum(0.9).centered(true)),
        ("RMSprop with Weight Decay", RMSpropBuilder::new().lr(config.learning_rate).weight_decay(1e-4)),
    ];

    for (name, builder) in rmsprop_configs {
        println!("🔧 Testing: {}", name);
        let result = run_optimizer_experiment(name, |params| Box::new(builder.clone().build(params)), config)?;
        print_experiment_summary(&result);
        println!();
    }

    Ok(())
}

/// Demonstrate AdaGrad with various configurations
fn demonstrate_adagrad_variants(config: &ExperimentConfig) -> Result<(), Box<dyn std::error::Error>> {
    println!("📉 AdaGrad Optimizer Variants Demo");
    println!("===================================\n");

    // Test different AdaGrad configurations
    let adagrad_configs = vec![
        ("Standard AdaGrad", AdaGradBuilder::new().lr(config.learning_rate)),
        ("AdaGrad with LR Decay", AdaGradBuilder::new().lr(config.learning_rate).lr_decay(1e-4)),
        ("AdaGrad with Weight Decay", AdaGradBuilder::new().lr(config.learning_rate).weight_decay(1e-4)),
        ("AdaGrad + Initial Accumulator", AdaGradBuilder::new().lr(config.learning_rate).initial_accumulator_value(0.1)),
    ];

    for (name, builder) in adagrad_configs {
        println!("🔧 Testing: {}", name);
        let result = run_optimizer_experiment(name, |params| Box::new(builder.clone().build(params)), config)?;
        print_experiment_summary(&result);
        println!();
    }

    Ok(())
}

/// Demonstrate Adam variants
fn demonstrate_adam_variants(config: &ExperimentConfig) -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Adam Optimizer Variants Demo");
    println!("================================\n");

    // Test different Adam configurations
    let adam_configs = vec![
        ("Standard Adam", AdamBuilder::new().lr(config.learning_rate)),
        ("Adam with AMSGrad", AdamBuilder::new().lr(config.learning_rate).amsgrad(true)),
        ("Adam with Weight Decay", AdamBuilder::new().lr(config.learning_rate).weight_decay(1e-4)),
        ("AdamW (Decoupled WD)", AdamBuilder::new().lr(config.learning_rate).weight_decay(0.01)),
    ];

    for (name, builder) in adam_configs {
        println!("🔧 Testing: {}", name);
        let result = run_optimizer_experiment(name, |params| {
            if name.contains("AdamW") {
                Box::new(builder.clone().build_adamw(params))
            } else {
                Box::new(builder.clone().build(params))
            }
        }, config)?;
        print_experiment_summary(&result);
        println!();
    }

    Ok(())
}

/// Demonstrate learning rate scheduling integration
fn demonstrate_lr_scheduling(config: &ExperimentConfig) -> Result<(), Box<dyn std::error::Error>> {
    println!("📅 Learning Rate Scheduling Demo");
    println!("=================================\n");

    let device = DeviceType::Cpu;

    // Create a simple model
    let mut model = Linear::new(config.input_size, config.output_size, true);
    model.to_device(device)?;
    
    let params = collect_parameters(&model);

    // Create SGD optimizer
    let mut optimizer = SGDBuilder::new(config.learning_rate)
        .momentum(0.9)
        .build(params);

    // Test different schedulers
    let scheduler_configs = vec![
        ("Step LR", Box::new(StepLR::new(5, 0.5)) as Box<dyn LRScheduler>),
        ("Exponential LR", Box::new(ExponentialLR::new(0.95)) as Box<dyn LRScheduler>),
        ("Cosine Annealing", Box::new(CosineAnnealingLR::new(config.num_epochs)) as Box<dyn LRScheduler>),
        ("One Cycle", Box::new(OneCycleLR::new(config.learning_rate * 10.0, config.num_epochs)) as Box<dyn LRScheduler>),
    ];

    for (scheduler_name, mut scheduler) in scheduler_configs {
        println!("📊 Testing scheduler: {}", scheduler_name);
        
        let mut lr_history = Vec::new();
        let mut loss_history = Vec::new();

        // Reset optimizer learning rate
        optimizer.set_lr(config.learning_rate);

        for epoch in 0..config.num_epochs {
            // Simulate training step
            let loss = simulate_training_step(&mut model, &mut optimizer, config)?;
            
            let current_lr = optimizer.get_lr()[0];
            lr_history.push(current_lr);
            loss_history.push(loss);
            
            if epoch % 5 == 0 {
                println!("   Epoch {}: LR = {:.6}, Loss = {:.4}", epoch, current_lr, loss);
            }
            
            // Update learning rate
            scheduler.step(&mut optimizer);
        }

        println!("   📈 LR Range: {:.6} -> {:.6}", lr_history[0], lr_history.last().unwrap());
        println!("   📉 Loss: {:.4} -> {:.4}", loss_history[0], loss_history.last().unwrap());
        println!();
    }

    Ok(())
}

/// Run comprehensive optimizer comparison
fn run_optimizer_comparison(config: &ExperimentConfig) -> Result<(), Box<dyn std::error::Error>> {
    println!("🏁 Optimizer Performance Comparison");
    println!("====================================\n");

    let mut results = Vec::new();

    // Test representative optimizers
    let optimizers = vec![
        ("SGD + Momentum", |params| Box::new(SGDBuilder::new(config.learning_rate).momentum(0.9).build(params)) as Box<dyn Optimizer>),
        ("RMSprop", |params| Box::new(RMSpropBuilder::new().lr(config.learning_rate).build(params)) as Box<dyn Optimizer>),
        ("AdaGrad", |params| Box::new(AdaGradBuilder::new().lr(config.learning_rate).build(params)) as Box<dyn Optimizer>),
        ("Adam", |params| Box::new(AdamBuilder::new().lr(config.learning_rate).build(params)) as Box<dyn Optimizer>),
        ("AdamW", |params| Box::new(AdamBuilder::new().lr(config.learning_rate).weight_decay(0.01).build_adamw(params)) as Box<dyn Optimizer>),
    ];

    for (name, optimizer_fn) in optimizers {
        println!("🏃 Running: {}", name);
        let result = run_optimizer_experiment(name, optimizer_fn, config)?;
        results.push(result);
    }

    // Print comparison table
    println!("\n📊 Performance Comparison Table:");
    println!("+-----------------+-------------+----------------+---------------+");
    println!("| Optimizer       | Final Loss  | Convergence    | Time (ms)     |");
    println!("+-----------------+-------------+----------------+---------------+");
    
    for result in &results {
        let convergence = result.convergence_epoch.map_or("N/A".to_string(), |e| e.to_string());
        println!("| {:15} | {:11.4} | {:14} | {:13.1} |", 
                result.optimizer_name, 
                result.final_loss, 
                convergence,
                result.total_time * 1000.0);
    }
    println!("+-----------------+-------------+----------------+---------------+");

    // Find best performer
    let best_result = results.iter().min_by(|a, b| a.final_loss.partial_cmp(&b.final_loss).unwrap()).unwrap();
    println!("\n🏆 Best performing optimizer: {} (Loss: {:.4})", best_result.optimizer_name, best_result.final_loss);

    Ok(())
}

/// Demonstrate best practices for optimizer selection
fn demonstrate_best_practices() -> Result<(), Box<dyn std::error::Error>> {
    println!("💡 Optimizer Selection Best Practices");
    println!("======================================\n");

    println!("📋 General Guidelines:");
    println!("   • Adam/AdamW: Great default choice, works well across domains");
    println!("   • SGD + Momentum: Often achieves best final performance with proper tuning");
    println!("   • RMSprop: Good for RNNs and when dealing with non-stationary objectives");
    println!("   • AdaGrad: Useful for sparse gradients, may decay too aggressively");
    println!();

    println!("🎯 Task-Specific Recommendations:");
    println!("   • Computer Vision: SGD with momentum or AdamW");
    println!("   • NLP/Transformers: AdamW with weight decay");
    println!("   • Reinforcement Learning: Adam or RMSprop");
    println!("   • Sparse Features: AdaGrad or sparse variants");
    println!();

    println!("⚙️ Hyperparameter Tips:");
    println!("   • Learning Rate: Start with 1e-3 for Adam, 1e-2 for SGD");
    println!("   • Momentum: 0.9 is a good default for SGD");
    println!("   • Weight Decay: 1e-4 to 1e-2 for regularization");
    println!("   • Beta2 (Adam): 0.999 for stable training, 0.9-0.95 for faster convergence");
    println!();

    println!("📈 Learning Rate Scheduling:");
    println!("   • Cosine Annealing: Good for long training runs");
    println!("   • Step Decay: Simple and effective");
    println!("   • One Cycle: Achieves fast convergence");
    println!("   • Reduce on Plateau: Adaptive to training progress");

    Ok(())
}

/// Run a single optimizer experiment
fn run_optimizer_experiment<F>(
    name: &str,
    optimizer_fn: F,
    config: &ExperimentConfig,
) -> Result<ExperimentResults, Box<dyn std::error::Error>>
where
    F: FnOnce(Vec<Arc<RwLock<Tensor>>>) -> Box<dyn Optimizer>,
{
    let device = DeviceType::Cpu;
    let start_time = Instant::now();

    // Create model
    let mut model = Linear::new(config.input_size, config.output_size, true);
    model.to_device(device)?;
    
    let params = collect_parameters(&model);
    let mut optimizer = optimizer_fn(params);

    let mut losses = Vec::new();
    let mut learning_rates = Vec::new();
    let mut convergence_epoch = None;
    let convergence_threshold = 0.1;

    // Training loop
    for epoch in 0..config.num_epochs {
        let loss = simulate_training_step(&mut model, &mut *optimizer, config)?;
        let lr = optimizer.get_lr()[0];
        
        losses.push(loss);
        learning_rates.push(lr);

        // Check for convergence
        if convergence_epoch.is_none() && loss < convergence_threshold {
            convergence_epoch = Some(epoch);
        }
    }

    let total_time = start_time.elapsed().as_secs_f64();
    let final_loss = losses.last().copied().unwrap_or(f64::INFINITY);

    Ok(ExperimentResults {
        optimizer_name: name.to_string(),
        final_loss,
        losses,
        learning_rates,
        total_time,
        convergence_epoch,
    })
}

/// Simulate a training step with synthetic data
fn simulate_training_step(
    model: &mut Linear,
    optimizer: &mut dyn Optimizer,
    config: &ExperimentConfig,
) -> Result<f64, Box<dyn std::error::Error>> {
    let mut total_loss = 0.0;

    for _ in 0..config.num_batches {
        // Generate synthetic input and target
        let input = randn(&[config.batch_size, config.input_size]);
        let target = randn(&[config.batch_size, config.output_size]);

        // Forward pass
        let output = model.forward(&input)?;
        
        // Compute simple MSE loss
        let diff = output.sub(&target)?;
        let loss_tensor = diff.mul(&diff)?.mean()?;
        let loss = loss_tensor.to_vec()[0] as f64;
        total_loss += loss;

        // Simulate gradient computation (in real implementation, this would be automatic)
        simulate_gradients(&model.parameters(), &input, &target)?;

        // Optimizer step
        optimizer.step()?;
        optimizer.zero_grad();
    }

    Ok(total_loss / config.num_batches as f64)
}

/// Simulate gradient computation for demonstration
fn simulate_gradients(
    parameters: &HashMap<String, torsh_nn::Parameter>,
    _input: &Tensor,
    _target: &Tensor,
) -> Result<(), Box<dyn std::error::Error>> {
    // In a real implementation, gradients would be computed automatically
    // For this demo, we simulate gradients with small random values
    for param in parameters.values() {
        let param_tensor = param.tensor();
        let param_shape = {
            let tensor = param_tensor.read();
            tensor.shape().dims().to_vec()
        };
        
        // Create mock gradient
        let grad = randn(&param_shape).mul_scalar(0.01)?;
        
        // Set gradient (this is a simplified simulation)
        // In real implementation, gradients would be set by autograd system
        // param_tensor.write().set_grad(grad);
    }
    
    Ok(())
}

/// Collect parameters from a model
fn collect_parameters(model: &Linear) -> Vec<Arc<RwLock<Tensor>>> {
    model.parameters().values().map(|p| p.tensor()).collect()
}

/// Print experiment summary
fn print_experiment_summary(result: &ExperimentResults) {
    println!("   📊 Results for {}:", result.optimizer_name);
    println!("      Final Loss: {:.4}", result.final_loss);
    if let Some(epoch) = result.convergence_epoch {
        println!("      Converged at epoch: {}", epoch);
    } else {
        println!("      Did not converge within {} epochs", result.losses.len());
    }
    println!("      Total time: {:.2}ms", result.total_time * 1000.0);
    println!("      LR trajectory: {:.6} -> {:.6}", 
             result.learning_rates.first().unwrap_or(&0.0), 
             result.learning_rates.last().unwrap_or(&0.0));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimizer_creation() {
        let param = Arc::new(RwLock::new(randn(&[10, 10])));
        let params = vec![param];

        // Test all optimizer builders
        let _sgd = SGDBuilder::new(0.01).momentum(0.9).build(params.clone());
        let _rmsprop = RMSpropBuilder::new().lr(0.01).build(params.clone());
        let _adagrad = AdaGradBuilder::new().lr(0.01).build(params.clone());
        let _adam = AdamBuilder::new().lr(0.01).build(params.clone());
        let _adamw = AdamBuilder::new().lr(0.01).build_adamw(params);
    }

    #[test]
    fn test_experiment_config() {
        let config = ExperimentConfig::default();
        assert_eq!(config.input_size, 100);
        assert_eq!(config.learning_rate, 0.01);
    }
}