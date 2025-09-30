//! # PyTorch to ToRSh Migration Guide
//! 
//! This example shows side-by-side comparisons of PyTorch and ToRSh code
//! to help PyTorch users quickly learn ToRSh patterns and conventions.
//! 
//! ## What you'll learn:
//! - API similarities and differences between PyTorch and ToRSh
//! - How to migrate common PyTorch patterns to ToRSh
//! - ToRSh-specific optimizations and features
//! - Best practices for clean migration
//! 
//! ## Prerequisites:
//! - Familiarity with PyTorch
//! - Basic Rust knowledge
//! 
//! Run with: `cargo run --example pytorch_to_torsh_migration`

use torsh::prelude::*;
use torsh::{Tensor, Device};
use torsh::nn::{Module, Linear, ReLU, MSELoss, CrossEntropyLoss, SGD, Adam};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== PyTorch to ToRSh Migration Guide ===\n");
    
    // 1. Tensor Creation
    println!("1. Tensor Creation");
    println!("==================");
    
    print_comparison(
        "PyTorch",
        r#"
import torch

# From list/array
x = torch.tensor([1, 2, 3])
y = torch.tensor([[1, 2], [3, 4]])

# Special tensors
zeros = torch.zeros(3, 3)
ones = torch.ones(2, 4)
random = torch.randn(3, 3)
arange = torch.arange(0, 10, 1)

# From numpy
import numpy as np
arr = np.array([1, 2, 3])
tensor = torch.from_numpy(arr)
        "#,
        "ToRSh",
        r#"
use torsh::prelude::*;

// From vector
let x = Tensor::from_vec(vec![1, 2, 3], &[3])?;
let y = Tensor::from_vec(vec![1, 2, 3, 4], &[2, 2])?;

// Special tensors
let zeros = Tensor::zeros(&[3, 3])?;
let ones = Tensor::ones(&[2, 4])?;
let random = Tensor::randn(&[3, 3])?;
let arange = Tensor::arange(0.0, 10.0, 1.0)?;

// From slice
let arr = [1.0, 2.0, 3.0];
let tensor = Tensor::from_slice(&arr, &[3])?;
        "#
    );
    
    // Example in action
    let torch_like = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3])?;
    let zeros = Tensor::zeros(&[2, 2])?;
    println!("ToRSh tensor: {:?}", torch_like);
    println!("Zeros tensor: {:?}\n", zeros);
    
    // 2. Basic Operations
    println!("2. Basic Operations");
    println!("===================");
    
    print_comparison(
        "PyTorch",
        r#"
# Element-wise operations
c = a + b
c = a * b
c = a - b
c = a / b

# In-place operations
a += b
a *= 2

# Mathematical functions
sin_a = torch.sin(a)
exp_a = torch.exp(a)
log_a = torch.log(a)

# Reductions
sum_a = torch.sum(a)
mean_a = torch.mean(a)
max_a = torch.max(a)
        "#,
        "ToRSh",
        r#"
// Element-wise operations
let c = &a + &b;
let c = &a * &b;
let c = &a - &b;
let c = &a / &b;

// In-place operations
a.add_(&b)?;
a.mul_scalar_(2.0)?;

// Mathematical functions
let sin_a = a.sin();
let exp_a = a.exp();
let log_a = a.log();

// Reductions
let sum_a = a.sum()?;
let mean_a = a.mean()?;
let max_a = a.max()?;
        "#
    );
    
    // Example in action
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3])?;
    let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], &[3])?;
    let c = &a + &b;
    let sin_a = a.sin();
    println!("a + b = {:?}", c);
    println!("sin(a) = {:?}\n", sin_a);
    
    // 3. Autograd/Gradients
    println!("3. Automatic Differentiation");
    println!("============================");
    
    print_comparison(
        "PyTorch",
        r#"
# Enable gradients
x = torch.tensor([2.0], requires_grad=True)

# Compute function
y = x ** 2

# Backward pass
y.backward()

# Access gradients
print(x.grad)  # tensor([4.0])

# Zero gradients
x.grad.zero_()
        "#,
        "ToRSh",
        r#"
// Enable gradients
let mut x = Tensor::from_vec(vec![2.0], &[1])?;
x.set_requires_grad(true);

// Compute function
let y = &x * &x;

// Backward pass
y.backward()?;

// Access gradients
if let Some(grad) = x.grad() {
    println!("{:?}", grad); // Should be [4.0]
}

// Zero gradients
x.zero_grad();
        "#
    );
    
    // Example in action
    let mut x = Tensor::from_vec(vec![2.0], &[1])?;
    x.set_requires_grad(true);
    let y = &x * &x;
    y.backward()?;
    if let Some(grad) = x.grad() {
        println!("Gradient dy/dx: {:?}\n", grad);
    }
    
    // 4. Neural Network Layers
    println!("4. Neural Network Layers");
    println!("========================");
    
    print_comparison(
        "PyTorch",
        r#"
import torch.nn as nn

# Linear layer
linear = nn.Linear(10, 5)

# Activation functions
relu = nn.ReLU()
sigmoid = nn.Sigmoid()
tanh = nn.Tanh()

# Forward pass
x = torch.randn(32, 10)  # batch_size=32
h = linear(x)
output = relu(h)
        "#,
        "ToRSh",
        r#"
use torsh::nn::*;

// Linear layer
let mut linear = Linear::new(10, 5)?;

// Activation functions
let relu = ReLU::new();
let sigmoid = Sigmoid::new();
let tanh = Tanh::new();

// Forward pass
let x = Tensor::randn(&[32, 10])?; // batch_size=32
let h = linear.forward(&x)?;
let output = relu.forward(&h)?;
        "#
    );
    
    // Example in action
    let mut linear = Linear::new(3, 2)?;
    let relu = ReLU::new();
    let x = Tensor::randn(&[1, 3])?;
    let h = linear.forward(&x)?;
    let output = relu.forward(&h)?;
    println!("Neural network output: {:?}\n", output);
    
    // 5. Loss Functions
    println!("5. Loss Functions");
    println!("=================");
    
    print_comparison(
        "PyTorch",
        r#"
import torch.nn as nn

# Mean Squared Error
mse_loss = nn.MSELoss()
loss = mse_loss(predictions, targets)

# Cross Entropy (for classification)
ce_loss = nn.CrossEntropyLoss()
loss = ce_loss(logits, class_indices)

# Custom loss
def custom_loss(pred, target):
    return torch.mean((pred - target) ** 2)
        "#,
        "ToRSh",
        r#"
use torsh::nn::*;

// Mean Squared Error
let mse_loss = MSELoss::new();
let loss = mse_loss.forward(&predictions, &targets)?;

// Cross Entropy (for classification)
let ce_loss = CrossEntropyLoss::new();
let loss = ce_loss.forward(&logits, &class_indices)?;

// Custom loss
fn custom_loss(pred: &Tensor, target: &Tensor) -> Result<Tensor, TorshError> {
    let diff = pred - target;
    let squared = &diff * &diff;
    squared.mean()
}
        "#
    );
    
    // 6. Optimizers
    println!("6. Optimizers");
    println!("=============");
    
    print_comparison(
        "PyTorch",
        r#"
import torch.optim as optim

# SGD
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Adam
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training step
optimizer.zero_grad()
loss.backward()
optimizer.step()
        "#,
        "ToRSh",
        r#"
use torsh::optim::*;

// SGD
let mut optimizer = SGD::new(model.parameters(), 0.01)?;

// Adam
let mut optimizer = Adam::new(model.parameters(), 0.001)?;

// Training step
optimizer.zero_grad();
loss.backward()?;
optimizer.step()?;
        "#
    );
    
    // 7. Device Management
    println!("7. Device Management");
    println!("====================");
    
    print_comparison(
        "PyTorch",
        r#"
# Check device availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move tensor to device
x = x.to(device)

# Create tensor on device
x = torch.randn(3, 3, device=device)

# Move model to device
model = model.to(device)
        "#,
        "ToRSh",
        r#"
// Check device availability
let device = if Device::Cuda(0).is_available() {
    Device::Cuda(0)
} else {
    Device::Cpu
};

// Move tensor to device
let x = x.to_device(device)?;

// Create tensor on device
let x = Tensor::randn(&[3, 3])?.to_device(device)?;

// Move model to device
model.to_device(device)?;
        "#
    );
    
    // 8. Model Definition Patterns
    println!("8. Model Definition Patterns");
    println!("============================");
    
    print_comparison(
        "PyTorch",
        r#"
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = SimpleNet()
        "#,
        "ToRSh",
        r#"
use torsh::nn::*;

struct SimpleNet {
    fc1: Linear,
    relu: ReLU,
    fc2: Linear,
}

impl SimpleNet {
    fn new() -> Result<Self, TorshError> {
        Ok(Self {
            fc1: Linear::new(784, 128)?,
            relu: ReLU::new(),
            fc2: Linear::new(128, 10)?,
        })
    }
    
    fn forward(&mut self, x: &Tensor) -> Result<Tensor, TorshError> {
        let x = self.fc1.forward(x)?;
        let x = self.relu.forward(&x)?;
        let x = self.fc2.forward(&x)?;
        Ok(x)
    }
}

let model = SimpleNet::new()?;
        "#
    );
    
    // 9. Training Loop Pattern
    println!("9. Training Loop Pattern");
    println!("========================");
    
    print_comparison(
        "PyTorch",
        r#"
# Training loop
model.train()
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch}, Loss: {loss.item():.6f}')
        "#,
        "ToRSh",
        r#"
// Training loop
model.train();
for epoch in 0..num_epochs {
    for (batch_idx, (data, target)) in train_loader.enumerate() {
        optimizer.zero_grad();
        let output = model.forward(&data)?;
        let loss = criterion.forward(&output, &target)?;
        loss.backward()?;
        optimizer.step()?;
        
        if batch_idx % 100 == 0 {
            println!("Epoch: {}, Loss: {:.6}", epoch, loss.item()?);
        }
    }
}
        "#
    );
    
    // 10. Key Differences Summary
    println!("10. Key Differences Summary");
    println!("===========================");
    
    println!("📋 **Memory Management**");
    println!("   PyTorch: Automatic garbage collection");
    println!("   ToRSh:   Rust ownership system (more predictable)\n");
    
    println!("🔧 **Error Handling**");
    println!("   PyTorch: Exceptions");
    println!("   ToRSh:   Result<T, E> pattern (must handle errors)\n");
    
    println!("⚡ **Performance**");
    println!("   PyTorch: Python overhead");
    println!("   ToRSh:   Zero-cost abstractions, no GIL\n");
    
    println!("🔒 **Safety**");
    println!("   PyTorch: Runtime errors possible");
    println!("   ToRSh:   Compile-time safety guarantees\n");
    
    println!("🛠️ **Concurrency**");
    println!("   PyTorch: GIL limitations");
    println!("   ToRSh:   True parallelism with Rust's ownership\n");
    
    // 11. Migration Tips
    println!("11. Migration Tips");
    println!("==================");
    
    println!("✅ **DO:**");
    println!("   • Use ? operator for error handling");
    println!("   • Leverage Rust's type system");
    println!("   • Use references (&) to avoid unnecessary clones");
    println!("   • Take advantage of compile-time optimizations");
    println!("   • Use ToRSh's zero-copy operations");
    
    println!("\n❌ **DON'T:**");
    println!("   • Ignore Result types (handle errors properly)");
    println!("   • Clone tensors unnecessarily");
    println!("   • Fight the borrow checker (redesign if needed)");
    println!("   • Assume Python patterns always translate directly");
    
    println!("\n🎯 **Best Practices:**");
    println!("   • Start with simple examples");
    println!("   • Use ToRSh's builder patterns");
    println!("   • Leverage automatic differentiation");
    println!("   • Profile your code for optimization opportunities");
    println!("   • Take advantage of ToRSh's native performance\n");
    
    // 12. Common Gotchas
    println!("12. Common Migration Gotchas");
    println!("============================");
    
    println!("🐛 **Borrowing Issues:**");
    println!("   Problem:  Multiple mutable borrows");
    println!("   Solution: Use separate scopes or restructure data flow\n");
    
    println!("🐛 **Tensor Shapes:**");
    println!("   Problem:  Shape mismatches at runtime");
    println!("   Solution: Use ToRSh's compile-time shape checking where possible\n");
    
    println!("🐛 **Device Transfers:**");
    println!("   Problem:  Forgetting to move tensors to same device");
    println!("   Solution: Use device-aware tensor creation patterns\n");
    
    println!("🐛 **Gradient Computation:**");
    println!("   Problem:  Forgetting to set requires_grad or call zero_grad()");
    println!("   Solution: Use ToRSh's structured training loop patterns\n");
    
    println!("🎉 Congratulations! You now understand the key differences between PyTorch and ToRSh");
    println!("📚 Next: Explore specific ToRSh examples to see these patterns in action");
    
    Ok(())
}

/// Helper function to print side-by-side code comparisons
fn print_comparison(lang1: &str, code1: &str, lang2: &str, code2: &str) {
    println!("**{}**", lang1);
    println!("```python");
    println!("{}", code1.trim());
    println!("```\n");
    
    println!("**{}**", lang2);
    println!("```rust");
    println!("{}", code2.trim());
    println!("```\n");
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tensor_creation_compatibility() {
        // Test that ToRSh tensors can be created similarly to PyTorch
        let torch_like = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let zeros = Tensor::zeros(&[2, 2]).unwrap();
        let ones = Tensor::ones(&[3, 3]).unwrap();
        
        assert_eq!(torch_like.shape().dims(), &[3]);
        assert_eq!(zeros.shape().dims(), &[2, 2]);
        assert_eq!(ones.shape().dims(), &[3, 3]);
    }
    
    #[test]
    fn test_autograd_compatibility() {
        // Test that ToRSh autograd works similarly to PyTorch
        let mut x = Tensor::from_vec(vec![2.0], &[1]).unwrap();
        x.set_requires_grad(true);
        
        let y = &x * &x; // y = x^2
        y.backward().unwrap();
        
        if let Some(grad) = x.grad() {
            let grad_val = grad.to_vec().unwrap()[0];
            assert!((grad_val - 4.0).abs() < 1e-6); // dy/dx = 2x = 4
        }
    }
    
    #[test]
    fn test_nn_layer_compatibility() {
        // Test that ToRSh neural network layers work similarly to PyTorch
        let mut linear = Linear::new(3, 2).unwrap();
        let relu = ReLU::new();
        
        let x = Tensor::ones(&[1, 3]).unwrap();
        let h = linear.forward(&x).unwrap();
        let output = relu.forward(&h).unwrap();
        
        assert_eq!(output.shape().dims(), &[1, 2]);
    }
}