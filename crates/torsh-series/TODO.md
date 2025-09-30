# ToRSh Series - TODO & Enhancement Roadmap

## 🎯 Current Status: NEWLY IMPLEMENTED ⚡
**SciRS2 Integration**: 80% - Time series analysis with scirs2-series foundation

## 📋 Recently Implemented Features
- ✅ **STL Decomposition** with functional trend extraction and seasonal averaging
- ✅ **Singular Spectrum Analysis (SSA)** with trajectory matrix embedding
- ✅ **ARIMA Models** with seasonal component support
- ✅ **Exponential Smoothing** (Simple, Holt-Winters)
- ✅ **Kalman Filters** for state space modeling
- ✅ **Anomaly Detection** (Statistical, Isolation Forest, LSTM-based)
- ✅ **Time Series Forecasting** framework with multiple model types

## 🚀 High Priority TODOs

### 1. Fix API Compatibility Issues
- [ ] **Resolve tensor slicing operations**
  ```rust
  // Current issue: slice method signature mismatch
  let values = self.values.slice(0, start as i64, end as i64, 1);
  // Should be: slice(dim, start, end)
  ```
- [ ] **Fix tensor creation APIs**
  ```rust
  // Replace Tensor::zeros with creation module functions
  use torsh_tensor::creation::{zeros, ones, randn};
  ```
- [ ] **Resolve import issues for LSTM/GRU**
  ```rust
  // Fix missing neural network imports
  use torsh_nn::{LSTM, GRU, Module};
  ```

### 2. Complete Forecasting Model Implementation
- [ ] **Deep Learning Models**
  ```rust
  pub struct LSTMForecaster {
      lstm_layers: Vec<LSTM>,
      dropout: Dropout,
      output_layer: Linear,
  }

  pub struct TransformerForecaster {
      encoder: TransformerEncoder,
      decoder: TransformerDecoder,
      positional_encoding: PositionalEncoding,
  }
  ```
- [ ] **Advanced ARIMA variants (SARIMA, ARMAX)**
- [ ] **Prophet-style trend and seasonality decomposition**
- [ ] **Neural Prophet hybrid models**

### 3. Enhanced State Space Models
- [ ] **Complete Kalman Filter implementation**
  ```rust
  impl KalmanFilterModel {
      fn predict(&mut self) -> Tensor {
          // Implement actual prediction step
          // x = F @ x
          // P = F @ P @ F.T + Q
      }

      fn update(&mut self, observation: &Tensor) {
          // Implement actual update step
          // K = P @ H.T @ inv(H @ P @ H.T + R)
          // x = x + K @ (z - H @ x)
          // P = (I - K @ H) @ P
      }
  }
  ```
- [ ] **Particle Filter complete implementation**
- [ ] **Unscented Kalman Filter with sigma points**
- [ ] **Extended Kalman Filter for nonlinear systems**

### 4. Deep Integration with scirs2-series
- [ ] **Replace placeholder implementations**
  ```rust
  use scirs2_series::{
      decomposition::{STLDecomposer, X13Decomposer},
      forecasting::{ARIMAModel, ExponentialSmoothing},
      anomaly::{IsolationForest, OneClassSVM},
      preprocessing::{Differencing, BoxCox, Normalization},
  };
  ```
- [ ] **Add wavelet decomposition**
- [ ] **Implement frequency domain analysis**
- [ ] **Add change point detection algorithms**

## 🔬 Research & Development TODOs

### 1. Advanced Deep Learning for Time Series
- [ ] **Attention-based models (Informer, Autoformer)**
  ```rust
  pub struct InformerModel {
      encoder: ProbSparseAttention,
      decoder: DistillingOperation,
      generator: ConvLayer,
  }
  ```
- [ ] **Graph Neural Networks for multivariate time series**
- [ ] **Neural ODEs for continuous time modeling**
- [ ] **Diffusion models for time series generation**

### 2. Multi-Scale and Multi-Resolution Analysis
- [ ] **Wavelet transforms and analysis**
- [ ] **Empirical Mode Decomposition (EMD)**
- [ ] **Variational Mode Decomposition (VMD)**
- [ ] **Multi-resolution forecasting frameworks**

### 3. Causal Inference and Interpretability
- [ ] **Granger causality testing**
- [ ] **SHAP values for time series models**
- [ ] **Feature importance over time**
- [ ] **Counterfactual analysis for forecasting**

## 🛠️ Medium Priority TODOs

### 1. Statistical Methods Enhancement
- [ ] **Complete time series tests**
  ```rust
  pub mod tests {
      pub fn augmented_dickey_fuller(series: &TimeSeries) -> ADFResult;
      pub fn ljung_box_test(residuals: &TimeSeries) -> LjungBoxResult;
      pub fn jarque_bera_test(series: &TimeSeries) -> JarqueBeraResult;
  }
  ```
- [ ] **Cointegration analysis (Johansen test)**
- [ ] **Vector Autoregression (VAR) models**
- [ ] **Error correction models (VEC)**

### 2. Data Preprocessing and Transformation
- [ ] **Robust missing value imputation**
  ```rust
  pub enum ImputationMethod {
      Linear,
      Spline,
      KalmanFilter,
      LOCF,  // Last Observation Carried Forward
      MICE,  // Multiple Imputation by Chained Equations
  }
  ```
- [ ] **Outlier detection and treatment**
- [ ] **Data scaling and normalization**
- [ ] **Feature engineering for time series**

### 3. Multivariate Time Series
- [ ] **Dynamic Factor Models**
- [ ] **Multivariate GARCH models**
- [ ] **Principal Component Analysis for time series**
- [ ] **Cross-correlation analysis**

## 🔍 Testing & Quality Assurance

### 1. Comprehensive Test Suite
- [ ] **Unit tests for all decomposition methods**
  ```rust
  #[test]
  fn test_stl_decomposition() {
      let ts = create_synthetic_series_with_trend_and_seasonality();
      let decomp = STLDecomposer::new(12).decompose(&ts);

      assert!(decomp.trend.variance() > 0.0);
      assert!(decomp.seasonal.mean().abs() < 1e-10);
  }
  ```
- [ ] **Integration tests with real datasets**
- [ ] **Benchmarks against statsmodels/forecasting libraries**
- [ ] **Numerical stability tests**

### 2. Time Series Specific Validation
- [ ] **Test on various data frequencies (daily, hourly, etc.)**
- [ ] **Validate with different trend patterns**
- [ ] **Test seasonal pattern detection**
- [ ] **Cross-validation for time series**

## 📦 Dependencies & Integration

### 1. Enhanced SciRS2 Integration
- [ ] **Complete scirs2-series algorithm adoption**
  ```rust
  use scirs2_series::*;  // Full integration
  ```
- [ ] **Leverage scirs2-signal for frequency analysis**
- [ ] **Use scirs2-stats for statistical tests**
- [ ] **Integrate scirs2-linalg for matrix operations**

### 2. Cross-Crate Coordination
- [ ] **Integration with torsh-nn for deep learning models**
- [ ] **Support torsh-data time series dataloaders**
- [ ] **Coordinate with torsh-optim for model training**

## 🎯 Success Metrics
- [ ] **Accuracy**: Match or exceed statsmodels/scikit-learn benchmarks
- [ ] **Performance**: Handle long time series (>1M points) efficiently
- [ ] **API**: Intuitive interface similar to Prophet/statsmodels
- [ ] **Coverage**: Support major time series analysis workflows

## ⚠️ Known Issues
- [ ] **Tensor slicing API incompatibility** (High priority)
- [ ] **Missing neural network layer imports** (Medium priority)
- [ ] **Incomplete state space model implementations** (Medium priority)
- [ ] **Error handling needs improvement** (Low priority)

## 🔗 Integration Dependencies
- **torsh-nn**: For LSTM, GRU, and other neural network components
- **torsh-tensor**: For efficient tensor operations and time series data
- **scirs2-series**: For advanced time series algorithms
- **scirs2-signal**: For frequency domain analysis

## 📅 Timeline
- **Phase 1** (1 week): Fix API compatibility and tensor operations
- **Phase 2** (2 weeks): Complete statistical models and state space methods
- **Phase 3** (1 month): Advanced deep learning models and scirs2 integration
- **Phase 4** (2 months): Research features and multivariate analysis

---
**Last Updated**: 2025-09-20
**Status**: Core functionality implemented, needs API fixes
**Next Milestone**: Resolve tensor operations and complete statistical tests