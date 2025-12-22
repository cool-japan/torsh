# ToRSh-NN Documentation

Comprehensive documentation for the ToRSh-NN neural network framework.

## 📚 Documentation Index

### Getting Started

#### For New Users
1. **[Custom Module Tutorial](CUSTOM_MODULE_TUTORIAL.md)** - Start here!
   - Step-by-step tutorials for building neural networks
   - Complete working examples (MLP, CNN, Transformer, ResNet)
   - Learn by doing with practical implementations

2. **[Layer Implementation Guide](LAYER_IMPLEMENTATION_GUIDE.md)**
   - How to create custom layers
   - Module trait implementation
   - Parameter management and initialization
   - Testing strategies

### Migrating from PyTorch

3. **[PyTorch Migration Guide](PYTORCH_MIGRATION_GUIDE.md)**
   - Side-by-side API comparisons
   - Layer-by-layer migration guide
   - Training loop conversion
   - Common pitfalls and solutions
   - Quick reference table

### Best Practices

4. **[Best Practices](BEST_PRACTICES.md)**
   - Code organization patterns
   - Memory management strategies
   - Error handling best practices
   - Testing strategies
   - API design principles
   - Common anti-patterns to avoid

### Performance

5. **[Performance Tuning Guide](PERFORMANCE_TUNING.md)**
   - Profiling and measurement
   - Memory optimization
   - Computational optimization
   - Backend selection (CPU, GPU, multi-GPU)
   - Model architecture optimization
   - Training and inference optimization

---

## 🎯 Quick Navigation

### I want to...

#### Learn ToRSh-NN from scratch
→ Start with **[Custom Module Tutorial](CUSTOM_MODULE_TUTORIAL.md)**

#### Implement a custom layer
→ Read **[Layer Implementation Guide](LAYER_IMPLEMENTATION_GUIDE.md)**

#### Migrate from PyTorch
→ Follow **[PyTorch Migration Guide](PYTORCH_MIGRATION_GUIDE.md)**

#### Write better code
→ Study **[Best Practices](BEST_PRACTICES.md)**

#### Optimize performance
→ Read **[Performance Tuning Guide](PERFORMANCE_TUNING.md)**

#### Build a specific architecture
| Architecture | Guide | Section |
|-------------|-------|---------|
| MLP | Custom Module Tutorial | Tutorial 1 |
| CNN | Custom Module Tutorial | Tutorial 2 |
| Transformer | Custom Module Tutorial | Tutorial 3 |
| ResNet | Custom Module Tutorial | Tutorial 4 |
| Custom Loss | Custom Module Tutorial | Tutorial 5 |
| Seq2Seq | Custom Module Tutorial | Tutorial 6 |

---

## 📖 Documentation Overview

### 1. Custom Module Tutorial (800+ lines)
**Level**: Beginner to Advanced

**Contents**:
- 6 Complete tutorials with working code
- MLP, CNN, Transformer, ResNet implementations
- Custom loss functions
- Composite model architectures
- Advanced patterns and techniques

**Who should read**: Everyone! Start here if you're new to ToRSh-NN.

### 2. Layer Implementation Guide (600+ lines)
**Level**: Intermediate

**Contents**:
- Layer structure and conventions
- Module trait implementation
- Parameter management
- Testing strategies
- 3 Complete layer implementations

**Who should read**: Anyone implementing custom layers.

### 3. PyTorch Migration Guide (700+ lines)
**Level**: Intermediate (requires PyTorch knowledge)

**Contents**:
- Quick start comparisons
- Layer-by-layer mapping
- Training loop migration
- API differences
- Migration checklist
- Complete examples

**Who should read**: PyTorch users migrating to ToRSh.

### 4. Best Practices (600+ lines)
**Level**: All Levels

**Contents**:
- Code organization
- Memory management
- Error handling
- Testing strategies
- API design
- DO vs DON'T examples throughout

**Who should read**: Everyone should read this for writing better code.

### 5. Performance Tuning (800+ lines)
**Level**: Advanced

**Contents**:
- Profiling techniques
- Memory optimization
- Computational optimization
- Hardware-specific optimizations
- Model optimization techniques

**Who should read**: Anyone deploying models to production.

---

## 🎓 Learning Paths

### Path 1: Complete Beginner
1. Read **Custom Module Tutorial** (Tutorials 1-3)
2. Study **Layer Implementation Guide** (Basic sections)
3. Review **Best Practices** (Code organization, Error handling)
4. Try implementing your own simple model

### Path 2: PyTorch User
1. Skim **PyTorch Migration Guide** (Quick comparison)
2. Follow **Custom Module Tutorial** (Any relevant tutorial)
3. Read **Best Practices** (Focus on Rust-specific patterns)
4. Check **Performance Tuning** (When needed)

### Path 3: Production Deployment
1. Review **Best Practices** (All sections)
2. Study **Performance Tuning** (Profiling, Optimization)
3. Reference **Layer Implementation Guide** (Testing strategies)
4. Apply optimizations to your models

### Path 4: Research & Development
1. Master **Layer Implementation Guide** (All sections)
2. Study **Custom Module Tutorial** (Advanced patterns)
3. Reference **Best Practices** (API design)
4. Explore **Performance Tuning** (Advanced techniques)

---

## 🔍 Quick Reference

### Common Tasks

#### Create a new layer
```rust
// See: Layer Implementation Guide - Basic Layer Structure
pub struct MyLayer {
    weight: Parameter,
    bias: Option<Parameter>,
    // ...
}

impl Module for MyLayer {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // ... implementation
    }

    fn parameters(&self) -> ParameterCollection {
        // ... parameter registration
    }

    // ... other methods
}
```

#### Build a model
```rust
// See: Custom Module Tutorial - Tutorial 1 (MLP)
let model = MLPClassifier::new(784, 256, 10, 0.5)?;
```

#### Initialize weights
```rust
// See: Layer Implementation Guide - Initialization
use torsh_nn::init::*;

// Xavier
let weight = xavier_uniform(&[out_features, in_features])?;

// Kaiming
let weight = kaiming_normal(&[out_features, in_features], "fan_in")?;

// Using InitMethod
let init = InitMethod::kaiming_normal()
    .with_fan_mode(FanMode::FanOut);
let weight = init.initialize(&[out_features, in_features])?;
```

#### Write tests
```rust
// See: Layer Implementation Guide - Testing Your Layer
#[test]
fn test_forward_shapes() {
    let layer = MyLayer::new(128, 64, true).unwrap();
    let input = randn::<f32>(&[32, 128]).unwrap();
    let output = layer.forward(&input).unwrap();
    assert_eq!(output.shape().dims(), &[32, 64]);
}
```

#### Optimize performance
```rust
// See: Performance Tuning Guide - Profiling
use std::time::Instant;

let start = Instant::now();
let output = model.forward(&input)?;
println!("Forward pass: {:?}", start.elapsed());
```

---

## 💡 Tips for Using Documentation

### Effective Reading
1. **Start with tutorials**: Hands-on learning is fastest
2. **Reference guides**: Use them when implementing
3. **Best practices**: Read regularly to improve code quality
4. **Performance guide**: Reference when optimizing

### Code Examples
- All code examples are tested and working
- Copy-paste into your project and modify
- Refer to comments for explanations

### Cross-References
- Documentation is interconnected
- Follow links between guides
- Build comprehensive understanding

---

## 🎯 Documentation Goals

This documentation aims to:

1. **Lower the barrier to entry**: Make ToRSh-NN accessible to everyone
2. **Provide practical examples**: Working code you can use immediately
3. **Teach best practices**: Write better, more maintainable code
4. **Enable migration**: Smooth transition from PyTorch
5. **Optimize performance**: Deploy fast, efficient models

---

## 📝 Documentation Quality

### Coverage
- ✅ All major topics covered
- ✅ 100+ working code examples
- ✅ Complete API reference
- ✅ Real-world architectures
- ✅ Performance optimizations

### Clarity
- ✅ Step-by-step tutorials
- ✅ Side-by-side comparisons
- ✅ DO vs DON'T examples
- ✅ Common pitfalls identified
- ✅ Clear explanations

### Practicality
- ✅ Copy-paste ready code
- ✅ Production-ready examples
- ✅ Testing strategies
- ✅ Deployment guidance
- ✅ Performance tuning

---

## 🔗 Additional Resources

### Internal Resources
- **API Documentation**: Run `cargo doc --open`
- **Examples Directory**: See `examples/` for complete programs
- **Tests Directory**: See `tests/` for integration tests
- **Source Code**: Read the implementation for details

### External Resources
- **GitHub Repository**: https://github.com/cool-japan/torsh
- **SciRS2 Documentation**: For underlying scientific computing
- **Rust Book**: For Rust language fundamentals
- **PyTorch Documentation**: For API comparisons

---

## 🤝 Contributing

### Documentation Improvements
Found an error or want to improve documentation? Contributions welcome!

1. Fork the repository
2. Make your changes
3. Test code examples
4. Submit a pull request

### Requesting New Content
Need documentation on a specific topic? Open an issue with:
- Topic description
- Use case
- What you'd like to learn

---

## 📊 Documentation Statistics

- **Total Guides**: 5 comprehensive guides
- **Total Lines**: 3,500+ lines of documentation
- **Code Examples**: 100+ working examples
- **Topics Covered**: 50+ major topics
- **Architectures**: MLP, CNN, Transformer, ResNet, Seq2Seq, and more

---

## 🎉 Get Started!

Ready to build neural networks with ToRSh?

**👉 Start with [Custom Module Tutorial](CUSTOM_MODULE_TUTORIAL.md)**

Have questions? Check the other guides or open an issue on GitHub!

---

**Last Updated**: 2025-10-22
**ToRSh-NN Version**: v0.1.0-alpha.2
**Documentation Version**: 1.0.0

For questions, issues, or contributions:
🔗 https://github.com/cool-japan/torsh
