//! Time series utilities
//!
//! This module provides comprehensive utilities for time series analysis:
//! - Preprocessing: Data cleaning, transformations, and preparation
//! - Features: Statistical and spectral feature extraction
//! - Metrics: Evaluation metrics for forecast accuracy
//! - Validation: Cross-validation and testing methods

pub mod features;
pub mod metrics;
pub mod preprocessing;
pub mod validation;

// Re-export main functionality for easy access
pub use features::{
    autocorrelation, partial_autocorrelation, seasonality_features, spectral_features,
    statistical_features, trend_features, SeasonalityFeatures, SpectralFeatures,
    StatisticalFeatures, TrendFeatures,
};

pub use metrics::{
    directional_accuracy, evaluate_forecast, mae, mape, mase, max_error, mse, r2, rmse, smape,
    theil_u, ForecastMetrics,
};

pub use preprocessing::{
    box_cox, detrend, diff, ema, inv_box_cox, min_max_scale, moving_average, normalize,
    standard_scale,
};

pub use validation::{
    expanding_window_validation, rolling_window_validation, walk_forward_validation,
    BlockedTimeSeriesCV, TimeSeriesCV,
};
