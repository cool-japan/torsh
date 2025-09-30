//! State space models for time series analysis
//!
//! This module provides comprehensive state space filtering capabilities:
//! - Kalman Filter: Linear Gaussian state space models
//! - Extended Kalman Filter: Nonlinear systems with analytical Jacobians
//! - Particle Filter: Non-linear/non-Gaussian systems using Monte Carlo methods
//! - Unscented Kalman Filter: Nonlinear systems using sigma points

pub mod extended;
pub mod kalman;
pub mod particle;
pub mod unscented;

// Re-export main types for easy access
pub use extended::ExtendedKalmanFilter;
pub use kalman::KalmanFilter;
pub use particle::{ParticleFilter, ParticleStats, ResamplingMethod};
pub use unscented::{UKFStats, UnscentedKalmanFilter};

// Backward compatibility aliases
pub use kalman::KalmanFilter as KalmanFilterModel;
pub use particle::ParticleFilter as ParticleFilterModel;
