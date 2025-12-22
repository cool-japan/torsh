//! Time series forecasting models
//!
//! This module provides various forecasting approaches:
//! - ARIMA: AutoRegressive Integrated Moving Average models
//! - VAR: Vector Autoregression for multivariate series
//! - Exponential Smoothing: Holt-Winters and variants
//! - Deep Learning: LSTM, GRU, Transformer models

pub mod arima;
pub mod deep;
pub mod smoothing;
pub mod var;

// Re-export main types for easy access
pub use arima::{AutoARIMA, ARIMA, SARIMA};
pub use deep::{CNNForecaster, GRUForecaster, LSTMForecaster, TransformerForecaster};
pub use smoothing::{HoltWinters, SimpleExponentialSmoothing};
pub use var::{GrangerCausality, VAR};
