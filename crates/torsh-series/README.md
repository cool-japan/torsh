# torsh-series

Time series analysis and forecasting components for ToRSh - powered by SciRS2.

## Overview

This crate provides comprehensive time series analysis, forecasting, and modeling capabilities with PyTorch-compatible neural network integration. It leverages `scirs2-series` for classical time series methods while enabling deep learning approaches through ToRSh's neural network modules.

## Features

- **Forecasting Models**: ARIMA, SARIMA, Prophet, Exponential Smoothing
- **Neural Forecasting**: LSTM, GRU, Temporal CNNs, Transformers, N-BEATS, DeepAR
- **Decomposition**: STL, seasonal decomposition, trend extraction
- **Anomaly Detection**: Isolation forests, statistical methods, autoencoders
- **State Space Models**: Kalman filters, particle filters, Hidden Markov Models
- **Feature Engineering**: Lag features, rolling statistics, fourier features
- **Changepoint Detection**: PELT, Binary segmentation, Bayesian methods
- **Frequency Analysis**: FFT, spectral analysis, wavelets
- **Utilities**: Differencing, transformations, stationarity tests
- **GPU Acceleration**: Neural forecasting models on CUDA

## Usage

### Basic Time Series Operations

```rust
use torsh_series::prelude::*;
use torsh_tensor::prelude::*;

// Create time series data
let data = tensor![1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0];
let timestamps = tensor![0, 1, 2, 3, 4, 5, 6, 7];

// Create TimeSeries object
let ts = TimeSeries::new(data, Some(timestamps), None)?;

println!("Length: {}", ts.len());
println!("Start: {:?}, End: {:?}", ts.start(), ts.end());

// Basic statistics
println!("Mean: {:.4}", ts.mean()?);
println!("Std: {:.4}", ts.std()?);
```

### Time Series Decomposition

#### STL Decomposition (Seasonal-Trend decomposition using Loess)

```rust
use torsh_series::decomposition::*;

let data = generate_seasonal_data(365 * 2, 12, 0.1)?;  // 2 years of monthly data

// Perform STL decomposition
let stl = STL::new()
    .seasonal_window(13)
    .trend_window(51)
    .seasonal_degree(1)
    .trend_degree(1);

let result = stl.fit(&data)?;

let trend = result.trend();
let seasonal = result.seasonal();
let residual = result.residual();

println!("Seasonal strength: {:.4}", result.seasonal_strength()?);
println!("Trend strength: {:.4}", result.trend_strength()?);

// Plot components
plot_decomposition(&result)?;
```

#### Classical Decomposition

```rust
use torsh_series::decomposition::*;

let data = load_ts("airline_passengers.csv")?;

// Additive decomposition: data = trend + seasonal + residual
let decomp_add = seasonal_decompose(&data, 12, "additive")?;

// Multiplicative decomposition: data = trend * seasonal * residual
let decomp_mul = seasonal_decompose(&data, 12, "multiplicative")?;

let trend = decomp_add.trend();
let seasonal = decomp_add.seasonal();
let residual = decomp_add.residual();
```

### ARIMA Models

```rust
use torsh_series::forecast::*;

let data = load_ts("co2_levels.csv")?;

// Auto ARIMA - automatically selects best (p,d,q) parameters
let auto_model = AutoARIMA::new()
    .seasonal(true)
    .m(12)  // Seasonal period
    .max_p(5)
    .max_q(5)
    .max_d(2)
    .information_criterion("aic");

let model = auto_model.fit(&data)?;
println!("Selected model: ARIMA{:?}", model.order());

// Forecast next 24 steps
let forecast = model.predict(24, None)?;
let conf_int = model.predict_interval(24, 0.95)?;  // 95% confidence interval

// Manual ARIMA(2,1,1)
let arima = ARIMA::new(2, 1, 1)?;
arima.fit(&data)?;

let forecast = arima.predict(12, None)?;
```

### SARIMA (Seasonal ARIMA)

```rust
use torsh_series::forecast::*;

let data = load_ts("monthly_sales.csv")?;

// SARIMA(1,1,1)(1,1,1,12)
// (p,d,q) = non-seasonal parameters
// (P,D,Q,m) = seasonal parameters (m=12 for monthly data)
let sarima = SARIMA::new(1, 1, 1, 1, 1, 1, 12)?;

sarima.fit(&data)?;

// Forecast with confidence intervals
let forecast = sarima.predict(24, None)?;
let (lower, upper) = sarima.predict_interval(24, 0.95)?;

// Model diagnostics
let residuals = sarima.residuals()?;
let aic = sarima.aic()?;
let bic = sarima.bic()?;

println!("AIC: {:.4}, BIC: {:.4}", aic, bic);

// Check residuals for white noise
let ljung_box = ljung_box_test(&residuals, 10)?;
println!("Ljung-Box p-value: {:.4}", ljung_box.p_value);
```

### Exponential Smoothing

```rust
use torsh_series::forecast::*;

let data = load_ts("demand.csv")?;

// Simple Exponential Smoothing
let ses = SimpleExpSmoothing::new(0.3)?;  // alpha=0.3
ses.fit(&data)?;
let forecast = ses.predict(12)?;

// Holt's Linear Trend
let holt = Holt::new(0.8, 0.2)?;  // alpha=0.8, beta=0.2
holt.fit(&data)?;
let forecast = holt.predict(12)?;

// Holt-Winters (Triple Exponential Smoothing)
let hw = HoltWinters::new()
    .seasonal_periods(12)
    .trend("add")
    .seasonal("add")
    .alpha(0.8)
    .beta(0.2)
    .gamma(0.3);

hw.fit(&data)?;
let forecast = hw.predict(24)?;

// Auto ETS - automatically selects Error-Trend-Seasonal components
let auto_ets = AutoETS::new()
    .information_criterion("aic")
    .allow_multiplicative_trend(true);

let model = auto_ets.fit(&data)?;
println!("Selected model: {}", model.name());
```

### Neural Forecasting Models

#### LSTM for Time Series

```rust
use torsh_series::forecast::neural::*;
use torsh_nn::prelude::*;

let data = load_ts("stock_prices.csv")?;

// Prepare sequences for LSTM
let (X, y) = create_sequences(&data, window_size=20, horizon=1)?;

// Define LSTM model
let lstm = LSTMForecaster::new()
    .input_size(1)
    .hidden_size(64)
    .num_layers(2)
    .output_size(1)
    .dropout(0.2)
    .bidirectional(false);

// Train model
lstm.fit(&X, &y, epochs=100, batch_size=32, lr=0.001)?;

// Forecast
let forecast = lstm.predict(&data, horizon=10)?;

// Multi-step forecasting
let multi_step_forecast = lstm.predict_multi_step(&data, horizon=30)?;
```

#### Temporal Convolutional Network (TCN)

```rust
use torsh_series::forecast::neural::*;

let tcn = TCN::new()
    .input_channels(1)
    .output_size(1)
    .num_channels(vec![32, 32, 32, 32])
    .kernel_size(3)
    .dropout(0.2);

tcn.fit(&X, &y, epochs=50, batch_size=64)?;
let forecast = tcn.predict(&data, horizon=20)?;
```

#### Transformer for Time Series

```rust
use torsh_series::forecast::neural::*;

let transformer = TimeSeriesTransformer::new()
    .d_model(64)
    .nhead(8)
    .num_encoder_layers(3)
    .num_decoder_layers(3)
    .dim_feedforward(256)
    .dropout(0.1)
    .sequence_length(50)
    .forecast_horizon(10);

transformer.fit(&data, epochs=100, batch_size=32)?;
let forecast = transformer.predict(&data, horizon=10)?;
```

#### N-BEATS (Neural Basis Expansion Analysis)

```rust
use torsh_series::forecast::neural::*;

// N-BEATS: interpretable and accurate forecasting
let nbeats = NBEATS::new()
    .stack_types(vec!["trend", "seasonality", "generic"])
    .num_blocks_per_stack(3)
    .forecast_length(24)
    .backcast_length(96)
    .hidden_layer_units(256)
    .share_weights_in_stack(true);

nbeats.fit(&data, epochs=100)?;
let forecast = nbeats.predict(&data, horizon=24)?;

// Get interpretable components
let (trend, seasonality) = nbeats.decompose(&data)?;
```

#### DeepAR (Probabilistic Forecasting)

```rust
use torsh_series::forecast::neural::*;

// DeepAR provides probabilistic forecasts
let deepar = DeepAR::new()
    .input_size(1)
    .hidden_size(40)
    .num_layers(2)
    .dropout(0.1)
    .context_length(30)
    .prediction_length=10;

deepar.fit(&data, epochs=50)?;

// Get probabilistic forecast with quantiles
let median_forecast = deepar.predict(&data, quantile=0.5)?;
let lower_bound = deepar.predict(&data, quantile=0.1)?;
let upper_bound = deepar.predict(&data, quantile=0.9)?;

// Sample from predictive distribution
let samples = deepar.sample(&data, num_samples=100)?;
```

### Anomaly Detection

```rust
use torsh_series::anomaly::*;

let data = load_ts("sensor_data.csv")?;

// Statistical anomaly detection
let detector = StatisticalAnomalyDetector::new()
    .method("zscore")
    .threshold(3.0)
    .window_size(None);

let anomalies = detector.detect(&data)?;
println!("Found {} anomalies", anomalies.sum()?);

// Isolation Forest
let iforest = IsolationForest::new()
    .n_estimators(100)
    .contamination(0.1)
    .max_samples("auto");

iforest.fit(&data)?;
let anomaly_scores = iforest.score(&data)?;
let anomalies = iforest.predict(&data)?;  // -1 for anomalies, 1 for normal

// LSTM Autoencoder for anomaly detection
let ae = LSTMAutoencoder::new()
    .encoding_dim(16)
    .sequence_length(50)
    .threshold_percentile(95);

ae.fit(&data, epochs=50)?;
let reconstruction_errors = ae.reconstruction_error(&data)?;
let anomalies = ae.detect_anomalies(&data)?;

// Seasonal Hybrid ESD (S-H-ESD) for seasonal data
let shesd = SeasonalESD::new()
    .max_anomalies(10)
    .alpha(0.05)
    .periodicity(24);

let anomalies = shesd.detect(&data)?;
```

### Changepoint Detection

```rust
use torsh_series::changepoint::*;

let data = load_ts("regime_changes.csv")?;

// PELT (Pruned Exact Linear Time)
let pelt = PELT::new()
    .model("rbf")
    .min_size(2)
    .jump(1)
    .penalty(Some(3.0));

let changepoints = pelt.detect(&data)?;
println!("Detected changepoints at: {:?}", changepoints);

// Binary Segmentation
let binseg = BinarySegmentation::new()
    .n_changepoints(5)
    .model("l2");

let changepoints = binseg.detect(&data)?;

// Bayesian Online Changepoint Detection
let bocd = BayesianOnlineChangepointDetection::new()
    .hazard_rate(1.0 / 100.0)
    .delay(15);

let changepoint_probs = bocd.detect_online(&data)?;

// Cumulative Sum (CUSUM)
let cusum = CUSUM::new()
    .threshold(5.0)
    .drift(1.0);

let changepoints = cusum.detect(&data)?;
```

### State Space Models

#### Kalman Filter

```rust
use torsh_series::state_space::*;

// Linear Kalman Filter
let kf = KalmanFilter::new()
    .state_transition(transition_matrix)
    .observation_matrix(observation_matrix)
    .process_noise(process_cov)
    .observation_noise(obs_cov)
    .initial_state(initial_state)
    .initial_covariance(initial_cov);

// Filter (estimate current state)
let filtered_states = kf.filter(&observations)?;

// Smooth (estimate all states given all observations)
let smoothed_states = kf.smooth(&observations)?;

// Predict future states
let predictions = kf.predict(horizon=10)?;
```

#### Particle Filter

```rust
use torsh_series::state_space::*;

// Particle Filter for nonlinear/non-Gaussian systems
let pf = ParticleFilter::new()
    .num_particles(1000)
    .state_transition_fn(transition_fn)
    .observation_fn(observation_fn)
    .resampling_strategy("systematic");

let estimated_states = pf.filter(&observations)?;
```

#### Hidden Markov Model (HMM)

```rust
use torsh_series::state_space::*;

// Discrete HMM
let hmm = HMM::new()
    .n_states(3)
    .n_observations(10);

hmm.fit(&observation_sequences)?;

// Viterbi algorithm - find most likely state sequence
let state_sequence = hmm.viterbi(&observations)?;

// Forward-backward algorithm - compute state probabilities
let state_probs = hmm.forward_backward(&observations)?;

// Predict next observations
let predictions = hmm.predict(&observations, horizon=5)?;
```

### Frequency Domain Analysis

```rust
use torsh_series::frequency::*;

let data = load_ts("signal.csv")?;

// Fast Fourier Transform
let fft_result = fft(&data)?;
let power_spectrum = fft_result.abs()?.pow(2)?;

// Find dominant frequencies
let dominant_freqs = find_dominant_frequencies(&data, top_k=5)?;
println!("Dominant frequencies: {:?}", dominant_freqs);

// Periodogram
let (freqs, psd) = periodogram(&data, window="hann")?;

// Welch's method for power spectral density
let (freqs, psd) = welch(&data, nperseg=256, overlap=128)?;

// Wavelet transform
let wavelet = WaveletTransform::new("morlet");
let coeffs = wavelet.transform(&data, scales)?;

// Spectrogram
let (times, freqs, spec) = spectrogram(&data, window_size=256, overlap=128)?;
```

### Feature Engineering

```rust
use torsh_series::features::*;

let data = load_ts("sales.csv")?;

// Create lag features
let lag_features = create_lags(&data, lags=vec![1, 7, 30])?;

// Rolling statistics
let rolling_mean = rolling_mean(&data, window=7)?;
let rolling_std = rolling_std(&data, window=7)?;
let rolling_min = rolling_min(&data, window=7)?;
let rolling_max = rolling_max(&data, window=7)?;

// Expanding statistics
let expanding_mean = expanding_mean(&data)?;
let expanding_std = expanding_std(&data)?;

// Exponentially weighted statistics
let ewm_mean = ewm_mean(&data, alpha=0.3)?;
let ewm_std = ewm_std(&data, alpha=0.3)?;

// Time-based features
let ts_features = TimeSeriesFeatures::new()
    .add_hour_of_day()
    .add_day_of_week()
    .add_day_of_month()
    .add_month_of_year()
    .add_quarter()
    .add_is_weekend()
    .add_is_holiday(holidays);

let features = ts_features.transform(&timestamps)?;

// Fourier features for seasonality
let fourier = fourier_features(&timestamps, period=365.25, order=10)?;

// Calendar features
let calendar = calendar_features(&dates, country="US")?;
```

### Stationarity and Transformations

```rust
use torsh_series::utils::*;

let data = load_ts("non_stationary.csv")?;

// Test for stationarity
let adf_result = augmented_dickey_fuller(&data, regression="c", lags=None)?;
println!("ADF statistic: {:.4}, p-value: {:.4}", adf_result.statistic, adf_result.p_value);

if adf_result.p_value > 0.05 {
    println!("Series is non-stationary");
}

// KPSS test
let kpss_result = kpss_test(&data, regression="c", lags=None)?;

// Differencing
let diff1 = difference(&data, periods=1)?;
let diff2 = difference(&data, periods=2)?;  // Second-order differencing

// Seasonal differencing
let seasonal_diff = seasonal_difference(&data, period=12)?;

// Log transformation
let log_data = data.log()?;

// Box-Cox transformation
let (transformed, lambda) = boxcox(&data)?;
println!("Optimal lambda: {:.4}", lambda);

// Inverse transform
let original = inv_boxcox(&transformed, lambda)?;

// Detrending
let detrended = detrend(&data, method="linear")?;
```

### Model Selection and Validation

```rust
use torsh_series::validation::*;

let data = load_ts("training_data.csv")?;

// Train-test split
let split = 0.8;
let (train, test) = train_test_split(&data, split)?;

// Time series cross-validation
let tscv = TimeSeriesSplit::new(n_splits=5, gap=0);

for (train_idx, val_idx) in tscv.split(&data) {
    let train_fold = data.index_select(0, &train_idx)?;
    let val_fold = data.index_select(0, &val_idx)?;

    // Train and validate model
    model.fit(&train_fold)?;
    let predictions = model.predict(val_fold.len(), None)?;
    let score = mae(&val_fold, &predictions)?;
}

// Expanding window cross-validation
let expanding_cv = ExpandingWindowSplit::new(
    initial_train_size=100,
    forecast_horizon=10,
    step=10,
);

// Grid search for hyperparameters
let param_grid = vec![
    ("p", vec![0, 1, 2]),
    ("d", vec![0, 1]),
    ("q", vec![0, 1, 2]),
];

let best_params = grid_search_arima(&data, &param_grid, metric="aic")?;
println!("Best parameters: {:?}", best_params);
```

### Evaluation Metrics

```rust
use torsh_series::metrics::*;

let y_true = test_data;
let y_pred = forecasts;

// Mean Absolute Error
let mae = mean_absolute_error(&y_true, &y_pred)?;

// Root Mean Squared Error
let rmse = root_mean_squared_error(&y_true, &y_pred)?;

// Mean Absolute Percentage Error
let mape = mean_absolute_percentage_error(&y_true, &y_pred)?;

// Symmetric MAPE
let smape = symmetric_mape(&y_true, &y_pred)?;

// Mean Absolute Scaled Error
let mase = mean_absolute_scaled_error(&y_true, &y_pred, &y_train, seasonality=1)?;

// Forecast bias
let bias = forecast_bias(&y_true, &y_pred)?;

// Tracking signal
let tracking_signal = tracking_signal(&y_true, &y_pred)?;
```

## Integration with SciRS2

This crate leverages the SciRS2 ecosystem for:

- Classical time series methods through `scirs2-series`
- Signal processing via `scirs2-signal`
- Statistical functions from `scirs2-stats`
- Neural network components via ToRSh modules
- Optimized tensor operations through `scirs2-core`

All implementations follow the [SciRS2 POLICY](https://github.com/cool-japan/scirs/blob/master/SCIRS2_POLICY.md) for consistent APIs and optimal performance.

## Examples

See the `examples/` directory for detailed examples:

- `arima_forecasting.rs` - ARIMA/SARIMA modeling
- `lstm_forecasting.rs` - Neural forecasting with LSTM
- `anomaly_detection.rs` - Time series anomaly detection
- `decomposition.rs` - STL and seasonal decomposition
- `changepoint_detection.rs` - Detecting regime changes
- `kalman_filter.rs` - State space modeling

## Performance Tips

1. **Use GPU acceleration** for neural forecasting models
2. **Apply differencing** to make series stationary before ARIMA
3. **Normalize data** before training neural networks
4. **Use parallel processing** for cross-validation
5. **Cache decomposition results** when used multiple times

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE))
- MIT license ([LICENSE-MIT](../../LICENSE-MIT))

at your option.
