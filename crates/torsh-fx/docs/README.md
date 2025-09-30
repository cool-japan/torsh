# ToRSh FX Documentation

Welcome to the ToRSh FX documentation! This directory contains comprehensive guides and references for using the ToRSh FX functional transformation framework.

## Documentation Overview

### 📚 Getting Started

- **[FX Tutorial](fx_tutorial.md)** - Complete tutorial covering basic usage, optimization, quantization, and advanced features. Start here if you're new to ToRSh FX.

### 🔧 Advanced Usage

- **[Transformation Guide](transformation_guide.md)** - Comprehensive guide to graph transformations, optimization passes, pattern matching, and custom pass development.

- **[Best Practices](best_practices.md)** - Production-ready best practices covering graph design, performance optimization, memory management, error handling, testing, and debugging.

### 📖 Reference Materials

- **[IR Specification](ir_specification.md)** - Technical specification of the ToRSh FX intermediate representation, including node types, edge semantics, serialization formats, and validation rules.

- **[Migration Guide](migration_guide.md)** - Step-by-step migration instructions from other frameworks (PyTorch FX, TensorFlow, ONNX, TorchScript) and between ToRSh FX versions.

## Quick Navigation

### By Use Case

| I want to... | Read this document |
|--------------|-------------------|
| Learn ToRSh FX basics | [FX Tutorial](fx_tutorial.md) |
| Optimize my graphs | [Transformation Guide](transformation_guide.md) |
| Deploy to production | [Best Practices](best_practices.md) |
| Understand the IR format | [IR Specification](ir_specification.md) |
| Migrate from another framework | [Migration Guide](migration_guide.md) |

### By Experience Level

- **Beginner**: Start with [FX Tutorial](fx_tutorial.md)
- **Intermediate**: Read [Transformation Guide](transformation_guide.md) and [Best Practices](best_practices.md)
- **Advanced**: Dive into [IR Specification](ir_specification.md) and [Migration Guide](migration_guide.md)

## Key Features Covered

### Core Functionality
- ✅ Graph construction and manipulation
- ✅ Symbolic tracing and execution
- ✅ Control flow support (conditionals, loops)
- ✅ Serialization (JSON/Binary formats)

### Optimization
- ✅ Operation fusion passes
- ✅ Dead code elimination
- ✅ Constant folding
- ✅ Common subexpression elimination
- ✅ Memory optimization
- ✅ Custom pass development

### Code Generation
- ✅ Python (PyTorch/NumPy) code generation
- ✅ C++ (LibTorch/Standard) code generation
- ✅ Hardware-specific targets (CUDA, TensorRT, XLA)
- ✅ Backend lowering framework

### Advanced Features
- ✅ Dynamic shape support with constraints
- ✅ Quantization (QAT and PTQ)
- ✅ Distributed execution planning
- ✅ ONNX import/export
- ✅ Custom backend integration
- ✅ Graph visualization and debugging

### Production Features
- ✅ Performance profiling and optimization
- ✅ Error handling and validation
- ✅ Memory management and optimization
- ✅ Testing strategies and frameworks
- ✅ Deployment best practices

## Examples and Code Samples

All documentation includes extensive code examples demonstrating:

- **Basic Usage**: Simple graph construction and manipulation
- **Real-world Patterns**: Production-ready code patterns
- **Error Handling**: Robust error handling and recovery
- **Performance**: Optimization techniques and benchmarking
- **Testing**: Comprehensive testing strategies

## Framework Comparisons

The documentation includes comparisons and migration guides for:

- **PyTorch FX**: Concepts, API differences, migration patterns
- **TensorFlow**: Graph execution models, operation mapping
- **ONNX**: Model format conversion and compatibility
- **TorchScript**: Control flow handling and optimization

## Technical Specifications

### Supported Operations
- Arithmetic: `add`, `sub`, `mul`, `div`, `pow`
- Activation: `relu`, `sigmoid`, `tanh`, `gelu`, `silu`
- Linear: `linear`, `conv1d`, `conv2d`, `conv3d`
- Pooling: `max_pool2d`, `avg_pool2d`, `global_avg_pool2d`
- Normalization: `batch_norm`, `layer_norm`, `group_norm`
- Attention: `scaled_dot_product_attention`, `multi_head_attention`
- Tensor: `reshape`, `transpose`, `concat`, `split`, `slice`
- Reduction: `sum`, `mean`, `max`, `min`, `argmax`, `argmin`
- Special: `softmax`, `log_softmax`, `dropout`, `embedding`

### Serialization Formats
- **JSON**: Human-readable, debugging-friendly
- **Binary**: Compact, production-optimized
- **ONNX**: Cross-framework compatibility

### Target Platforms
- **CPU**: Optimized SIMD kernels, multi-threading
- **CUDA**: GPU acceleration, Tensor Core support
- **TensorRT**: High-performance inference optimization
- **XLA**: Google's accelerated linear algebra compiler
- **Custom**: Extensible backend framework

## Contributing

This documentation is part of the ToRSh project. For contributions:

1. **Bug Reports**: Report documentation issues or inaccuracies
2. **Improvements**: Suggest clarifications or additional examples
3. **New Sections**: Propose new topics or use cases
4. **Translations**: Help translate documentation to other languages

## Version Information

- **Documentation Version**: 1.0.0
- **ToRSh FX Version**: 0.1.0-alpha.1
- **Last Updated**: July 2024
- **Compatibility**: Covers all features in ToRSh FX 0.1.0-alpha.1

## Additional Resources

### Community
- **GitHub**: [ToRSh Repository](https://github.com/totem-ml/torsh)
- **Discussions**: Community forums and Q&A
- **Examples**: Additional examples in the repository

### Performance
- **Benchmarks**: Performance comparisons with other frameworks
- **Optimization Tips**: Framework-specific optimization guides
- **Profiling Tools**: Built-in profiling and analysis tools

### Research
- **Papers**: Academic papers using ToRSh FX
- **Case Studies**: Real-world deployment stories
- **Future Roadmap**: Planned features and improvements

---

**Happy learning!** 🚀

For questions or feedback, please refer to the ToRSh project's community channels or file an issue in the main repository.