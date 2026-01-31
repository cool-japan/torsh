# ToRSh Development Roadmap

This document outlines planned features, enhancements, and deferred implementations for future ToRSh releases.

## Version 0.2.0 (Next Release) - Optimizer & State Management Enhancements

### torsh-optim: Advanced Optimizer Improvements

#### Parameter Storage Architecture Redesign
**Priority**: High
**Status**: Deferred from v0.1.0-rc.1
**Affected Components**:
- `AdvancedAdam` optimizer (advanced.rs)
- `LAMB` optimizer (advanced.rs)
- `Lookahead` wrapper (advanced.rs)

**Current Limitation**:
These optimizers currently don't store parameters internally, making full `step()` and `zero_grad()` implementations incomplete.

**Planned Implementation**:
1. Add `params: Vec<Arc<RwLock<Tensor>>>` field to optimizer structs
2. Implement complete `step()` method with:
   - Parameter iteration
   - Gradient computation
   - Adaptive learning rate scheduling
   - Gradient clipping (where applicable)
   - Weight decay application
3. Implement `zero_grad()` to clear gradients for stored parameters
4. Add parameter initialization in constructors

**Expected Outcome**:
- Fully functional advanced optimizers with complete gradient update logic
- Proper integration with torsh-nn models
- Better performance characteristics for large-scale training

---

#### Parameter Group Management
**Priority**: Medium
**Status**: Deferred from v0.1.0-rc.1
**Affected Components**:
- `AdvancedAdam` (advanced.rs lines 112-113)
- `LAMB` (advanced.rs lines 228)
- All advanced optimizers

**Current Limitation**:
`add_param_group()` is a no-op placeholder.

**Planned Implementation**:
1. Add `param_groups: Vec<ParamGroup>` storage
2. Implement per-group hyperparameters (lr, weight_decay, momentum, etc.)
3. Enable different optimization strategies for different parameter groups
4. Support layer-wise learning rate scheduling

**Use Cases**:
- Fine-tuning with different learning rates for different layers
- Discriminative learning rates for transfer learning
- Group-specific regularization strategies

---

#### Enhanced State Persistence
**Priority**: Medium
**Status**: Deferred from v0.1.0-rc.1
**Affected Components**:
- `AdvancedAdam` state_dict/load_state_dict (advanced.rs lines 128-129, 160)
- `LAMB` state_dict/load_state_dict (advanced.rs lines 243-244, 276-277)
- `SAGA` gradient table serialization (online_learning.rs lines 568)

**Current Limitation**:
- Advanced optimizers only save global hyperparameters
- Per-parameter state (momentum buffers, adaptive learning rates) not persisted
- SAGA gradient table not serialized

**Planned Implementation**:

##### For AdvancedAdam & LAMB:
1. Serialize per-parameter state:
   - `exp_avg` (first moment estimates)
   - `exp_avg_sq` (second moment estimates)
   - `max_exp_avg_sq` (for AMSGrad variant)
2. Store parameter groups with options
3. Add version migration support for backward compatibility

##### For SAGA:
1. Serialize complete gradient table:
   - Per-data-point gradients
   - Gradient sum for control variate
2. Efficient binary serialization using `oxicode`
3. Compression options for large gradient tables

**Expected Benefits**:
- Full checkpoint/restore capability
- Training resumption without state loss
- Better distributed training support

---

#### Lookahead Optimizer Enhancements
**Priority**: Low
**Status**: Deferred from v0.1.0-rc.1
**Location**: advanced.rs lines 312-314

**Current Limitation**:
Lookahead only delegates to base optimizer; slow weights update not implemented.

**Planned Implementation**:
1. Store slow weights: `HashMap<String, Tensor>`
2. Implement k-step lookahead logic:
   - Track fast weights (base optimizer parameters)
   - Update slow weights every k steps: `θ_slow = θ_slow + α(θ_fast - θ_slow)`
3. Add configuration for:
   - `k`: lookahead frequency
   - `α`: slow weights step size
4. Proper initialization of slow weights from current parameters

**Reference**:
Zhang et al. (2019) "Lookahead Optimizer: k steps forward, 1 step back"

---

### torsh-text: Advanced NLP Feature Integration

#### scirs2-text v0.2.0+ Integration
**Priority**: High
**Status**: Pending scirs2-text production release
**Affected Components**:
- Embeddings generation (scirs2_text_integration.rs:96)
- Sentiment analysis (scirs2_text_integration.rs:142)
- Named entity recognition (scirs2_text_integration.rs:216)
- Text classification (scirs2_text_integration.rs:269)
- Text summarization (scirs2_text_integration.rs:308)
- Language detection (scirs2_text_integration.rs:356)
- Topic modeling (scirs2_text_integration.rs:552)
- Document clustering (scirs2_text_integration.rs:578)
- Text paraphrasing (scirs2_text_integration.rs:601)

**Current Status**:
All features have placeholder implementations providing basic functionality.

**Planned Implementation**:
Once scirs2-text v0.2.0+ provides production-ready APIs:
1. Replace placeholder logic with scirs2-text native implementations
2. Integrate pre-trained transformer models (BERT, RoBERTa, GPT)
3. Enable GPU-accelerated inference
4. Add caching and optimization layers
5. Comprehensive testing with standard NLP benchmarks

**Expected Benefits**:
- Production-quality NLP processing
- State-of-the-art accuracy
- GPU acceleration for large-scale text processing
- Seamless integration with torsh-nn models

---

## Version 0.3.0+ - Advanced Features

### Distributed Training Support
**Priority**: High
**Components**: torsh-optim, torsh-nn, torsh-data

**Planned Features**:
- Data-parallel training with gradient synchronization
- Model-parallel training for large models
- Pipeline parallelism
- Mixed precision training (FP16/BF16)
- Gradient compression and communication optimization

---

### Model Quantization
**Priority**: Medium
**Components**: torsh-nn, torsh-tensor

**Planned Features**:
- Post-training quantization (INT8/INT4)
- Quantization-aware training
- Dynamic quantization for inference
- Calibration tools and sensitivity analysis

---

### Advanced Data Pipeline
**Priority**: Medium
**Components**: torsh-data, torsh-vision, torsh-text

**Planned Features**:
- Streaming datasets for out-of-core training
- Advanced data augmentation policies (AutoAugment, RandAugment)
- Mixup and CutMix for vision tasks
- Back-translation and synonym replacement for text
- Prefetching and caching optimization

---

### Neural Architecture Search (NAS)
**Priority**: Low
**Components**: New torsh-nas crate

**Planned Features**:
- DARTS (Differentiable Architecture Search)
- ENAS (Efficient Neural Architecture Search)
- ProxylessNAS for mobile deployment
- Hardware-aware NAS with latency constraints

---

## Contributing

This roadmap is subject to change based on:
- Community feedback and feature requests
- scirs2 ecosystem updates
- PyTorch compatibility requirements
- Performance optimization opportunities

For feature requests or to contribute to roadmap items, please:
1. Open an issue on GitHub with the `enhancement` label
2. Reference the roadmap item if applicable
3. Provide use cases and expected benefits

---

## Version History

- **v0.1.0-rc.1** (2026-01-28): Initial roadmap creation with deferred optimizer features
- Future updates will be tracked here

---

## References

### Academic Papers
- **SVRG**: Johnson & Zhang (2013) "Accelerating Stochastic Gradient Descent using Predictive Variance Reduction"
- **SAGA**: Defazio et al. (2014) "SAGA: A Fast Incremental Gradient Method"
- **Lookahead**: Zhang et al. (2019) "Lookahead Optimizer: k steps forward, 1 step back"
- **LAMB**: You et al. (2019) "Large Batch Optimization for Deep Learning"

### Related Projects
- scirs2 ecosystem: https://github.com/cool-japan/scirs
- PyTorch: https://pytorch.org (compatibility target)
- OptiRS: Advanced ML optimization algorithms
