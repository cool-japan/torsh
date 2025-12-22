//! Time series utilities
//!
//! This module provides comprehensive utilities for time series analysis:
//! - Preprocessing: Data cleaning, transformations, and preparation
//! - Features: Statistical and spectral feature extraction
//! - Metrics: Evaluation metrics for forecast accuracy
//! - Validation: Cross-validation and testing methods
//! - Statistical Tests: Hypothesis testing for time series (ADF, KPSS, Ljung-Box, etc.)
//! - Imputation: Missing value handling with advanced methods
//! - Outliers: Detection and treatment of outliers using various methods
//! - Cointegration: Long-run equilibrium relationship testing (Engle-Granger, Johansen)
//! - Transfer Entropy: Information-theoretic causal inference

pub mod cointegration;
pub mod features;
pub mod imputation;
pub mod metrics;
pub mod outliers;
pub mod preprocessing;
pub mod statistical_tests;
pub mod transfer_entropy;
pub mod validation;

// Re-export main functionality for easy access
pub use features::{
    autocorrelation, create_comprehensive_features, create_difference_features,
    create_interaction_features, create_lag_features, partial_autocorrelation, rolling_statistics,
    seasonality_features, spectral_features, statistical_features, trend_features,
    RollingStatistics, SeasonalityFeatures, SpectralFeatures, StatisticalFeatures, TrendFeatures,
};

pub use imputation::{ImputationMethod, MICEImputer, TimeSeriesImputer};

pub use metrics::{
    directional_accuracy, evaluate_forecast, mae, mape, mase, max_error, mse, r2, rmse, smape,
    theil_u, ForecastMetrics,
};

pub use outliers::{
    detect_and_treat_outliers, OutlierDetectionResult, OutlierDetector, OutlierMethod,
    OutlierTreatment,
};

pub use preprocessing::{
    box_cox, detrend, diff, ema, inv_box_cox, min_max_scale, moving_average, normalize,
    standard_scale,
};

pub use statistical_tests::{
    augmented_dickey_fuller_test, jarque_bera, kpss_test, ljung_box, phillips_perron_test,
    ResidualDiagnostics, StationarityTestSuite,
};

pub use validation::{
    expanding_window_validation, rolling_window_validation, walk_forward_validation,
    BlockedTimeSeriesCV, CombinatorialPurgedCV, NestedTimeSeriesCV, PurgedTimeSeriesCV,
    ScoredTimeSeriesCV, TimeSeriesCV,
};

pub use cointegration::{
    engle_granger_test, johansen_test, CriticalValues, EngleGrangerResult, JohansenResult, VECM,
};

pub use transfer_entropy::{
    compute_lagged_transfer_entropy, ConditionalTransferEntropy, TEMethod,
    TransferEntropyEstimator, TransferEntropyResult,
};
