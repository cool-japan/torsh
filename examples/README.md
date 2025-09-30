# ToRSh Examples & Tutorials

Welcome to ToRSh! This directory contains a comprehensive collection of examples and tutorials to help you learn and master the ToRSh deep learning framework.

## 🎓 Learning Path for Beginners

If you're new to ToRSh or tensor computing, follow this progressive learning path:

### **📚 Tutorial Series (Start Here!)**

1. **[01_tensor_basics.rs](./01_tensor_basics.rs)** - Fundamental tensor operations
   - Creating tensors from data
   - Understanding shapes and dimensions
   - Basic operations (add, multiply, reshape)
   - Device management (CPU/GPU)
   - Indexing and slicing
   - Broadcasting concepts

2. **[02_autograd_basics.rs](./02_autograd_basics.rs)** - Automatic differentiation
   - Enabling gradient computation
   - Computing gradients with backward()
   - Understanding the computational graph
   - Chain rule in action
   - Practical optimization example

3. **[03_neural_networks.rs](./03_neural_networks.rs)** - Building neural networks
   - Creating layers (Linear, Activation)
   - Forward and backward passes
   - Training loops and optimization
   - XOR problem walkthrough

4. **[04_cnn_basics.rs](./04_cnn_basics.rs)** - Convolutional Neural Networks *(Coming Soon)*
   - Convolution operations
   - Pooling layers
   - CNN architecture design
   - Image classification

### **🏃‍♂️ Next Steps**

After completing the tutorial series, explore these examples:

- **[basic_example.rs](./basic_example.rs)** - Core ToRSh functionality showcase
- **[linear_regression.rs](./linear_regression.rs)** - Simple machine learning
- **[neural_network_training.rs](./neural_network_training.rs)** - Basic NN training

## 🚀 Advanced Examples

### **Deep Learning Models**
- **[resnet.rs](./resnet.rs)** - ResNet architecture implementation
- **[transformer_architectures.rs](./transformer_architectures.rs)** - Transformer models
- **[gpt2.rs](./gpt2.rs)** - GPT-2 implementation
- **[vision_models_demo.rs](./vision_models_demo.rs)** - Computer vision models

### **Training & Optimization**
- **[advanced_training.rs](./advanced_training.rs)** - Advanced training techniques
- **[model_checkpointing.rs](./model_checkpointing.rs)** - Save/load models
- **[gradient_checkpointing.rs](./gradient_checkpointing.rs)** - Memory-efficient training
- **[advanced_optimizers_showcase.rs](./advanced_optimizers_showcase.rs)** - Optimizer comparison

### **Distributed Training**
- **[distributed.rs](./distributed.rs)** - Basic distributed training
- **[distributed_comprehensive.rs](./distributed_comprehensive.rs)** - Complete distributed setup
- **[distributed_fsdp.rs](./distributed_fsdp.rs)** - Fully Sharded Data Parallel
- **[distributed_pipeline.rs](./distributed_pipeline.rs)** - Pipeline parallelism

### **Backend-Specific Examples**
- **[metal_demo.rs](./metal_demo.rs)** - Apple Metal backend
- **[webgpu_backend_demo.rs](./webgpu_backend_demo.rs)** - WebGPU backend
- **[cuda_event_coordination.rs](./cuda_event_coordination.rs)** - CUDA optimization

### **Domain-Specific Applications**
- **[image_classification.rs](./image_classification.rs)** - Image classification
- **[nlp_transformer.rs](./nlp_transformer.rs)** - Natural language processing
- **[mnist.rs](./mnist.rs)** - MNIST digit recognition
- **[simple_cnn.rs](./simple_cnn.rs)** - Convolutional neural network

### **Performance & Benchmarking**
- **[performance_benchmark.rs](./performance_benchmark.rs)** - Performance testing
- **[tensor_serialization_demo.rs](./tensor_serialization_demo.rs)** - Serialization performance
- **[broadcasting_demo.rs](./broadcasting_demo.rs)** - Broadcasting mechanics

## 🛠️ How to Run Examples

### **Prerequisites**
```bash
# Ensure you have Rust installed
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Clone the ToRSh repository
git clone https://github.com/your-repo/torsh
cd torsh
```

### **Running Examples**
```bash
# Run a specific example
cargo run --example 01_tensor_basics

# Run with optimizations (recommended for performance testing)
cargo run --release --example performance_benchmark

# Run with specific features
cargo run --example metal_demo --features metal

# Run with CUDA support (if available)
cargo run --example cuda_event_coordination --features cuda
```

### **Running Tests**
```bash
# Run all tests
cargo test

# Run tests for a specific example
cargo test --example 03_neural_networks

# Run with output
cargo test -- --nocapture
```

## 🎯 Examples by Use Case

### **Learning Machine Learning**
1. Start with tensor operations: `01_tensor_basics.rs`
2. Learn gradients: `02_autograd_basics.rs`
3. Build your first network: `03_neural_networks.rs`
4. Try image classification: `mnist.rs`

### **Computer Vision**
- `image_classification.rs` - Image classification basics
- `resnet.rs` - Advanced CNN architecture
- `data_augmentation_demo.rs` - Data preprocessing
- `vision_models_demo.rs` - Pre-trained models

### **Natural Language Processing**
- `nlp_transformer.rs` - Transformer models
- `gpt2.rs` - Language generation
- `text_models_demo.rs` - Text processing
- `transformer_benchmarks.rs` - Performance comparison

### **High-Performance Computing**
- `distributed_comprehensive.rs` - Multi-GPU training
- `distributed_fsdp.rs` - Large model training
- `performance_benchmark.rs` - Performance optimization
- `unified_memory_demo.rs` - Memory management

### **Production Deployment**
- `model_checkpointing.rs` - Model persistence
- `tensor_serialization_demo.rs` - Model serialization
- `webgpu_backend_demo.rs` - Web deployment
- Python examples in `python/` directory

## 🐛 Troubleshooting

### **Common Issues**

1. **Compilation Errors**
   ```bash
   # Make sure all dependencies are up to date
   cargo update
   
   # Clean and rebuild
   cargo clean && cargo build
   ```

2. **CUDA Examples Not Working**
   ```bash
   # Ensure CUDA is installed and ToRSh is built with CUDA support
   cargo build --features cuda
   ```

3. **Memory Issues**
   ```bash
   # Run with memory debugging
   RUST_LOG=debug cargo run --example your_example
   ```

4. **Performance Issues**
   ```bash
   # Always use release mode for performance testing
   cargo run --release --example performance_benchmark
   ```

### **Getting Help**

- 📖 **Documentation**: Check the inline documentation in each example
- 🐛 **Issues**: Report bugs on our GitHub issue tracker
- 💬 **Community**: Join our Discord/Slack for discussions
- 📚 **Guides**: See the main documentation at docs.torsh.ai

## 🤝 Contributing Examples

We welcome contributions! To add a new example:

1. **Follow the naming convention**: `descriptive_name.rs`
2. **Add comprehensive documentation**: Explain what the example does
3. **Include tests**: Add `#[cfg(test)]` module with tests
4. **Update this README**: Add your example to the appropriate section
5. **Ensure it compiles**: Test with `cargo check --example your_example`

### **Example Template**
```rust
//! # Your Example Title
//! 
//! Brief description of what this example demonstrates.
//! 
//! ## What you'll learn:
//! - Key concept 1
//! - Key concept 2
//! 
//! ## Prerequisites:
//! - Required knowledge
//! 
//! Run with: `cargo run --example your_example`

use torsh::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Your Example ===");
    
    // Your example code here
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_your_functionality() {
        // Your tests here
    }
}
```

## 📋 Feature Coverage Matrix

| Feature | Basic | Intermediate | Advanced |
|---------|-------|-------------|----------|
| **Tensors** | ✅ Tutorial 01 | ✅ broadcasting_demo | ✅ unified_memory_demo |
| **Autograd** | ✅ Tutorial 02 | ✅ gradient_checkpointing | ✅ advanced_training |
| **Neural Networks** | ✅ Tutorial 03 | ✅ resnet | ✅ transformer_architectures |
| **Optimization** | ✅ linear_regression | ✅ neural_network_training | ✅ advanced_optimizers |
| **Distributed** | ❌ *Coming Soon* | ✅ distributed | ✅ distributed_fsdp |
| **Vision** | ✅ mnist | ✅ image_classification | ✅ vision_models_demo |
| **NLP** | ❌ *Coming Soon* | ✅ nlp_transformer | ✅ gpt2 |
| **Backends** | ✅ basic_example | ✅ metal_demo | ✅ cuda_event_coordination |

## 🎖️ Difficulty Levels

- 🟢 **Beginner** (Tutorial 01-03, basic_example, linear_regression)
- 🟡 **Intermediate** (mnist, resnet, distributed)
- 🔴 **Advanced** (transformer_architectures, distributed_fsdp, gpt2)
- ⚫ **Expert** (advanced_training, cuda_event_coordination)

---

**Happy learning with ToRSh! 🚀**

*For more information, visit our [documentation](https://docs.torsh.ai) or [GitHub repository](https://github.com/your-repo/torsh).*