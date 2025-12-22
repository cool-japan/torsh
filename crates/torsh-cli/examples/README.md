# ToRSh CLI Examples

This directory contains example configurations, datasets, and scripts demonstrating various ToRSh CLI capabilities.

## Directory Structure

```
examples/
├── configs/           # Configuration files
│   ├── train_*.yaml          # Training configurations
│   ├── benchmark_*.yaml      # Benchmarking configurations
│   └── quantize_*.yaml       # Quantization configurations
├── datasets/          # Sample datasets (not in git)
├── scripts/           # Helper scripts
│   ├── train.sh              # Training workflow
│   ├── benchmark.sh          # Benchmarking workflow
│   ├── quantize.sh           # Quantization workflow
│   └── full_pipeline.sh      # Complete ML pipeline
└── README.md          # This file
```

## Quick Start Examples

### 1. Training a Model

**ResNet18 on CIFAR-10 (Quick Test)**
```bash
# Download CIFAR-10 dataset
torsh dataset download cifar10 --output ./examples/datasets/cifar10

# Start training
torsh train start \\
  --config examples/configs/train_resnet18_cifar10.yaml \\
  --data ./examples/datasets/cifar10 \\
  --device cuda

# Monitor training progress
torsh train monitor --run ./runs/run_20240101_120000_abcd --follow
```

**MobileNetV2 on ImageNet (Production)**
```bash
# Note: Requires ImageNet dataset
torsh train start \\
  --config examples/configs/train_mobilenet_imagenet.yaml \\
  --data /path/to/imagenet \\
  --device cuda \\
  --distributed

# Resume from checkpoint
torsh train resume \\
  --checkpoint ./runs/run_xyz/checkpoint_epoch_50.ckpt \\
  --epochs 300
```

### 2. Benchmarking Models

**Single Device Benchmark**
```bash
torsh benchmark \\
  --model ./models/resnet50.torsh \\
  --device cuda:0 \\
  --batch-sizes 1,8,16,32 \\
  --input-shape 3,224,224 \\
  --warmup-iterations 10 \\
  --benchmark-iterations 100
```

**Multi-Device Comparison**
```bash
torsh benchmark \\
  --config examples/configs/benchmark_multi_device.yaml \\
  --output benchmarks/comparison_report.html
```

**Expected Output:**
```
Benchmarking configurations: ████████████ 12/12
╔════════════════════════════════════════════════════════╗
║ Benchmark Results Summary                             ║
╠════════════════════════════════════════════════════════╣
║ Best Throughput: 1234.5 samples/sec (cuda:0, batch=32)║
║ Best Latency:    8.12ms (cuda:0, batch=1)           ║
║ Most Efficient:  cuda:0 (152.1 samples/ms)          ║
╚════════════════════════════════════════════════════════╝

Results saved to: benchmarks/comparison_report.html
```

### 3. Model Quantization

**Dynamic Quantization (Fast)**
```bash
torsh quantize \\
  --input ./models/resnet18.torsh \\
  --output ./models/resnet18_int8_dynamic.torsh \\
  --mode dynamic \\
  --precision int8
```

**Static Quantization (Accurate)**
```bash
torsh quantize \\
  --config examples/configs/quantize_static_int8.yaml

# Expected output:
# Quantization completed:
#   Compression: 4.1x (237MB → 58MB)
#   Original accuracy: 94.23%
#   Quantized accuracy: 93.87%
#   Accuracy degradation: 0.36%
```

**Quantization-Aware Training (Best Accuracy)**
```bash
torsh quantize \\
  --input ./models/resnet18_pretrained.torsh \\
  --output ./models/resnet18_qat_int8.torsh \\
  --mode qat \\
  --precision int8 \\
  --calibration-data ./data/cifar10/train \\
  --epochs 10
```

### 4. Dataset Operations

**Download Public Datasets**
```bash
# CIFAR-10
torsh dataset download cifar10 --output ./examples/datasets/cifar10

# MNIST
torsh dataset download mnist --output ./examples/datasets/mnist

# ImageNet (requires credentials)
torsh dataset download imagenet --output ./data/imagenet
```

**Prepare Custom Dataset**
```bash
# ImageFolder format
torsh dataset prepare \\
  --input ./raw_images \\
  --output ./examples/datasets/my_dataset \\
  --format imagefolder

# CSV format
torsh dataset prepare \\
  --input ./data.csv \\
  --output ./examples/datasets/my_tabular \\
  --format csv
```

**Split Dataset**
```bash
torsh dataset split \\
  --input ./examples/datasets/my_dataset \\
  --output ./examples/datasets/my_dataset_split \\
  --train 0.7 \\
  --val 0.15 \\
  --test 0.15
```

**Calculate Statistics**
```bash
torsh dataset statistics \\
  ./examples/datasets/cifar10/train \\
  --output ./examples/datasets/cifar10_stats.json

# Use stats for normalization in training config
```

**Transform Dataset**
```bash
torsh dataset transform \\
  --input ./examples/datasets/my_dataset \\
  --output ./examples/datasets/my_dataset_normalized \\
  --transformations normalize,standardize
```

## Complete Workflows

### Workflow 1: Train → Benchmark → Quantize

```bash
#!/bin/bash
# Complete ML pipeline

# 1. Download dataset
torsh dataset download cifar10 --output ./data/cifar10

# 2. Train model
torsh train start \\
  --config examples/configs/train_resnet18_cifar10.yaml \\
  --data ./data/cifar10

# 3. Benchmark original model
torsh benchmark \\
  --model ./runs/latest/best_model.ckpt \\
  --output ./benchmarks/original.json

# 4. Quantize model
torsh quantize \\
  --input ./runs/latest/best_model.ckpt \\
  --output ./models/quantized_int8.torsh \\
  --mode static \\
  --precision int8 \\
  --calibration-data ./data/cifar10/val

# 5. Benchmark quantized model
torsh benchmark \\
  --model ./models/quantized_int8.torsh \\
  --output ./benchmarks/quantized.json

# 6. Compare results
echo "Performance Comparison:"
jq -s '
  {
    original_throughput: .[0].summary.best_throughput.metric_value,
    quantized_throughput: .[1].summary.best_throughput.metric_value,
    speedup: (.[1].summary.best_throughput.metric_value / .[0].summary.best_throughput.metric_value)
  }
' ./benchmarks/original.json ./benchmarks/quantized.json
```

### Workflow 2: Distributed Training

```bash
#!/bin/bash
# Distributed training on multiple GPUs

# 1. Prepare environment
export TORSH_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 2. Start distributed training
torsh train start \\
  --config examples/configs/train_mobilenet_imagenet.yaml \\
  --data /data/imagenet \\
  --device cuda \\
  --distributed \\
  --epochs 300

# 3. Monitor from another terminal
torsh train monitor \\
  --run ./runs/latest \\
  --follow
```

### Workflow 3: Hyperparameter Search

```bash
#!/bin/bash
# Grid search over learning rates

for lr in 0.001 0.01 0.1; do
  for wd in 0.0001 0.001 0.01; do
    echo "Training with lr=$lr, wd=$wd"

    torsh train start \\
      --config examples/configs/train_resnet18_cifar10.yaml \\
      --learning-rate $lr \\
      --weight-decay $wd \\
      --epochs 50 \\
      --output-dir ./runs/hp_search_lr${lr}_wd${wd}

  done
done

# Analyze results
python analyze_hp_search.py ./runs/hp_search_*
```

## Configuration File Templates

### Training Configuration Template

```yaml
model:
  name: <model_architecture>
  num_classes: <int>
  pretrained: <bool>

data:
  path: <dataset_path>
  batch_size: <int>

training:
  epochs: <int>
  learning_rate: <float>
  device: <cpu|cuda|metal>
  optimizer: <adam|adamw|sgd|rmsprop>
  scheduler: <constant|step|cosine|exponential>

logging:
  tensorboard: <bool>
  log_dir: <path>
```

### Benchmark Configuration Template

```yaml
model_path: <path>
input_shapes:
  - [channels, height, width]
batch_sizes:
  - <int>
devices:
  - <device_name>
warmup_iterations: <int>
benchmark_iterations: <int>
profile_memory: <bool>
profile_compute: <bool>
output_format: <json|csv|html>
```

### Quantization Configuration Template

```yaml
input_model: <path>
output_model: <path>
mode: <dynamic|static|qat>
precision: <int8|int4|fp16|bf16>
calibration_data: <path>  # For static/qat
calibration_samples: <int>
per_channel: <bool>
symmetric: <bool>
accuracy_threshold: <float>
exclude_layers:
  - <layer_name>
```

## Tips and Best Practices

### Training
1. **Start small**: Test with small models and datasets first
2. **Use mixed precision**: Enable AMP for 2-3x faster training on modern GPUs
3. **Monitor actively**: Use `torsh train monitor --follow` in a separate terminal
4. **Save checkpoints frequently**: Set `save_every` to save every few epochs
5. **Enable early stopping**: Prevent overfitting with patience-based stopping

### Benchmarking
1. **Warm up properly**: Use at least 10 warmup iterations
2. **Test multiple batch sizes**: Find the optimal batch size for your hardware
3. **Compare across devices**: Use multi-device benchmarking for deployment decisions
4. **Profile memory**: Enable memory profiling to identify bottlenecks
5. **Save results**: Keep benchmark results for tracking performance over time

### Quantization
1. **Try dynamic first**: Fastest quantization method, good for CPU deployment
2. **Use static for accuracy**: Better accuracy with calibration dataset
3. **QAT for best results**: Requires more time but best accuracy preservation
4. **Validate accuracy**: Always check accuracy degradation after quantization
5. **Exclude sensitive layers**: First and last layers often benefit from FP32

### Dataset Operations
1. **Validate first**: Always run dataset validation before training
2. **Calculate statistics**: Use dataset stats for proper normalization
3. **Split properly**: Use appropriate train/val/test splits (70/15/15 is common)
4. **Augment carefully**: Too much augmentation can hurt performance
5. **Cache if possible**: Cache preprocessed data for faster training

## Troubleshooting

### Training Issues

**Out of Memory**
```bash
# Reduce batch size
--batch-size 16  # Instead of 32

# Enable gradient accumulation
# In config.yaml:
training:
  accumulation_steps: 4  # Effective batch size = 16 * 4 = 64
```

**Slow Training**
```bash
# Enable mixed precision
--mixed-precision

# Increase num_workers
# In config.yaml:
data:
  num_workers: 8  # More data loading threads
```

### Benchmarking Issues

**Inconsistent Results**
```bash
# Increase iterations
--warmup-iterations 20
--benchmark-iterations 200
```

**CUDA Out of Memory**
```bash
# Test with smaller batch sizes
--batch-sizes 1,2,4,8
```

### Quantization Issues

**Accuracy Drop Too Large**
```bash
# Try per-channel quantization
--per-channel

# Exclude sensitive layers
# In config.yaml:
exclude_layers:
  - first_conv
  - final_fc
```

**Quantization Too Slow**
```bash
# Reduce calibration samples
--calibration-samples 500  # Instead of 1000
```

## Next Steps

1. **Explore advanced features**: Check the full documentation
2. **Customize configurations**: Modify example configs for your use case
3. **Share results**: Contribute your configurations back to the community
4. **Report issues**: Found a bug? Let us know on GitHub

## Additional Resources

- **Documentation**: `/tmp/TORSH_CLI_BETA1_ENHANCEMENTS.md`
- **Quick Start**: `/tmp/TORSH_CLI_BETA1_QUICK_START.md`
- **ToRSh Project**: [GitHub Repository]
- **Community**: [Discord/Forum]

---

**Happy ML workflows with ToRSh! 🚀**
