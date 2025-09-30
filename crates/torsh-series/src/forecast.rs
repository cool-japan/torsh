//! Time series forecasting models
//!
//! This module provides comprehensive forecasting capabilities:
//! - ARIMA: AutoRegressive Integrated Moving Average models
//! - Smoothing: Exponential smoothing methods (Holt-Winters, Simple, Double)
//! - Deep Learning: Neural network-based forecasters (LSTM, GRU, Transformer, CNN)

pub mod arima;
pub mod deep;
pub mod smoothing;

// Re-export main types for easy access
pub use arima::{AutoARIMA, ARIMA, SARIMA};
pub use deep::{CNNForecaster, GRUForecaster, LSTMForecaster, TransformerForecaster};
pub use smoothing::{DoubleExpSmoothing, HoltWinters, SimpleExpSmoothing};
