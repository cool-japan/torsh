# ToRSh Metrics - TODO & Enhancement Roadmap

## 🎯 Current Status: NEWLY IMPLEMENTED ⚡
**SciRS2 Integration**: 70% - Evaluation metrics with scirs2-metrics foundation

## 📋 Recently Implemented Features
- ✅ **Classification Metrics** (Accuracy, F1, Precision, Recall, ROC-AUC)
- ✅ **Regression Metrics** (MSE, MAE, R², Explained Variance)
- ✅ **Clustering Metrics** (Silhouette, Calinski-Harabasz, Davies-Bouldin)
- ✅ **Ranking Metrics** (NDCG, MAP, MRR)
- ✅ **MetricCollection** for batch evaluation
- ✅ **Utility functions** for metric computation

## 🚀 High Priority TODOs

### 1. Fix Critical API Compatibility Issues
- [ ] **Resolve tensor operations**
  ```rust
  // Current issues to fix:
  let pred_classes = predictions.argmax(Some(-1)).unwrap();  // ✅ Fixed
  let correct = pred_classes.eq(targets).unwrap().sum_all(); // Need sum_all() method

  // Fix topk operation
  let (_, top_k_indices) = predictions.topk(k, Some(-1), true, true).unwrap(); // ✅ Fixed
  ```
- [ ] **Fix tensor creation APIs**
  ```rust
  // Replace Tensor::zeros with creation functions
  use torsh_tensor::creation::{zeros, ones, from_vec};
  ```
- [ ] **Resolve scirs2_metrics import issues**
  ```rust
  // Remove unavailable imports, implement directly
  // use scirs2_metrics::prelude::*;  // Not available
  ```

### 2. Complete Metric Implementations
- [ ] **Classification Metrics**
  ```rust
  impl Accuracy {
      fn compute(&self, predictions: &Tensor, targets: &Tensor) -> f64 {
          // Fix sum_all() method call
          let correct = pred_classes.eq(targets).unwrap().to_scalar();
          let total = targets.numel() as f64;
          correct / total
      }
  }
  ```
- [ ] **Multi-class and Multi-label Support**
  ```rust
  pub struct MultiClassMetrics {
      pub per_class_precision: Vec<f64>,
      pub per_class_recall: Vec<f64>,
      pub per_class_f1: Vec<f64>,
      pub macro_avg: f64,
      pub weighted_avg: f64,
  }
  ```
- [ ] **Threshold-dependent Metrics**
  ```rust
  pub struct ThresholdMetrics {
      pub optimal_threshold: f64,
      pub precision_recall_curve: (Vec<f64>, Vec<f64>),
      pub roc_curve: (Vec<f64>, Vec<f64>),
  }
  ```

### 3. Advanced Evaluation Metrics
- [ ] **Deep Learning Specific Metrics**
  ```rust
  pub struct DeepLearningMetrics {
      pub perplexity: f64,
      pub bleu_score: f64,
      pub rouge_score: f64,
      pub inception_score: f64,
      pub fid_score: f64,
  }
  ```
- [ ] **Information Retrieval Metrics**
  ```rust
  pub struct IRMetrics {
      pub precision_at_k: Vec<f64>,
      pub recall_at_k: Vec<f64>,
      pub average_precision: f64,
      pub reciprocal_rank: f64,
  }
  ```

### 4. Statistical and Robust Metrics
- [ ] **Bootstrap Confidence Intervals**
  ```rust
  pub struct BootstrapResult {
      pub metric_value: f64,
      pub confidence_interval: (f64, f64),
      pub std_error: f64,
      pub p_value: Option<f64>,
  }
  ```
- [ ] **Cross-Validation Metrics**
  ```rust
  pub struct CVResults {
      pub cv_scores: Vec<f64>,
      pub mean_score: f64,
      pub std_score: f64,
      pub fold_metrics: Vec<HashMap<String, f64>>,
  }
  ```

## 🔬 Research & Development TODOs

### 1. Fairness and Bias Metrics
- [ ] **Algorithmic Fairness Evaluation**
  ```rust
  pub struct FairnessMetrics {
      pub demographic_parity: f64,
      pub equalized_odds: f64,
      pub calibration: f64,
      pub individual_fairness: f64,
  }
  ```
- [ ] **Bias Detection and Measurement**
- [ ] **Group-wise performance analysis**

### 2. Uncertainty Quantification Metrics
- [ ] **Calibration Metrics**
  ```rust
  pub struct CalibrationMetrics {
      pub expected_calibration_error: f64,
      pub maximum_calibration_error: f64,
      pub reliability_diagram: (Vec<f64>, Vec<f64>),
  }
  ```
- [ ] **Prediction Interval Coverage**
- [ ] **Epistemic vs Aleatoric uncertainty separation**

### 3. Advanced ML Metrics
- [ ] **Meta-Learning Metrics**
- [ ] **Few-Shot Learning Evaluation**
- [ ] **Domain Adaptation Metrics**
- [ ] **Continual Learning Evaluation**

## 🛠️ Medium Priority TODOs

### 1. Performance and Scalability
- [ ] **Efficient Metric Computation**
  ```rust
  pub struct StreamingMetrics {
      running_accuracy: RunningAverage,
      confusion_matrix: OnlineConfusionMatrix,
      auroc_calculator: OnlineAUROC,
  }
  ```
- [ ] **GPU-Accelerated Metrics**
  ```rust
  use scirs2_core::gpu::GpuMetrics;
  ```
- [ ] **Parallel Metric Computation**
- [ ] **Memory-Efficient Large Dataset Evaluation**

### 2. Visualization and Reporting
- [ ] **Metric Visualization Tools**
  ```rust
  pub struct MetricVisualizer {
      pub plot_confusion_matrix: fn(&ConfusionMatrix) -> PlotResult,
      pub plot_roc_curve: fn(&ROCCurve) -> PlotResult,
      pub plot_precision_recall: fn(&PRCurve) -> PlotResult,
  }
  ```
- [ ] **Automated Report Generation**
- [ ] **Interactive Metric Dashboards**
- [ ] **LaTeX/PDF Report Export**

### 3. Integration and Compatibility
- [ ] **MLflow Integration**
- [ ] **TensorBoard Logging**
- [ ] **Weights & Biases Support**
- [ ] **scikit-learn Compatibility Layer**

## 🔍 Testing & Quality Assurance

### 1. Comprehensive Test Suite
- [ ] **Unit tests for all metrics**
  ```rust
  #[test]
  fn test_accuracy_metric() {
      let predictions = tensor![[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]];
      let targets = tensor![1, 0, 1];
      let accuracy = Accuracy::new();
      let result = accuracy.compute(&predictions, &targets);
      assert!((result - 0.6667).abs() < 1e-4);
  }
  ```
- [ ] **Edge case testing (empty inputs, NaN values)**
- [ ] **Performance benchmarks vs scikit-learn**
- [ ] **Numerical stability tests**

### 2. Validation Against Reference Implementations
- [ ] **Cross-validate with scikit-learn metrics**
- [ ] **Compare with TorchMetrics results**
- [ ] **Validate statistical properties**
- [ ] **Test on standardized datasets**

## 📦 Dependencies & Integration

### 1. Enhanced SciRS2 Integration
- [ ] **Deep scirs2-metrics integration**
  ```rust
  // When scirs2-metrics API stabilizes
  use scirs2_metrics::{
      classification::*,
      regression::*,
      clustering::*,
      ranking::*,
  };
  ```
- [ ] **Leverage scirs2-stats for statistical tests**
- [ ] **Use scirs2-core for numerical stability**

### 2. Cross-Crate Coordination
- [ ] **Integration with torsh-nn for model evaluation**
- [ ] **Support torsh-data for batch evaluation**
- [ ] **Coordinate with torsh-distributed for large-scale metrics**

## 🎯 Success Metrics
- [ ] **Accuracy**: Results match scikit-learn within 1e-10 tolerance
- [ ] **Performance**: Evaluate 1M+ samples efficiently
- [ ] **Coverage**: Support all standard ML evaluation scenarios
- [ ] **API**: Intuitive interface compatible with PyTorch workflows

## ⚠️ Known Issues
- [ ] **Missing sum_all() method on tensors** (High priority)
- [ ] **to_vec() API changes needed** (Medium priority)
- [ ] **Type mismatches in tensor operations** (Medium priority)
- [ ] **Missing scirs2_metrics imports** (Low priority - implement directly)

## 🔗 Integration Dependencies
- **torsh-tensor**: For tensor operations and data manipulation
- **torsh-nn**: For neural network model evaluation
- **scirs2-metrics**: For advanced statistical metrics
- **scirs2-stats**: For statistical testing and validation

## 📅 Timeline
- **Phase 1** (1 week): Fix tensor API compatibility and basic metrics
- **Phase 2** (2 weeks): Complete all standard ML metrics
- **Phase 3** (1 month): Advanced metrics and statistical validation
- **Phase 4** (2 months): Research metrics and fairness evaluation

---
**Last Updated**: 2025-09-20
**Status**: Framework implemented, needs API compatibility fixes
**Next Milestone**: Fix tensor operations and complete basic metric suite