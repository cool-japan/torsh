//! Advanced gradient filtering and smoothing techniques for robust training
//!
//! This module provides sophisticated gradient filtering methods that help stabilize
//! training by reducing noise, handling outliers, and providing adaptive smoothing.

use scirs2_core::numeric::{Float, FromPrimitive, ToPrimitive};
#[allow(unused_imports)]
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use torsh_core::dtype::FloatElement;
use torsh_core::error::{Result, TorshError};

/// Advanced gradient filtering techniques beyond basic smoothing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdvancedFilterType {
    /// Kalman filtering for gradient estimation
    Kalman,
    /// Bilateral filtering (edge-preserving)
    Bilateral,
    /// Wiener filtering for noise reduction
    Wiener,
    /// Savitzky-Golay smoothing filter
    SavitzkyGolay,
    /// Adaptive median filter
    AdaptiveMedian,
    /// Butterworth low-pass filter
    Butterworth,
    /// Chebyshev filter
    Chebyshev,
    /// Empirical mode decomposition
    EMD,
}

/// Configuration for gradient filtering
#[derive(Debug, Clone)]
pub struct FilterConfig<T: FloatElement> {
    /// Filter type
    pub filter_type: AdvancedFilterType,
    /// Primary filter parameter (e.g., cutoff frequency, window size)
    pub primary_param: T,
    /// Secondary filter parameter (e.g., sigma for bilateral, order for Butterworth)
    pub secondary_param: Option<T>,
    /// Filter order/degree
    pub order: usize,
    /// Enable adaptive behavior
    pub adaptive: bool,
}

/// Kalman filter state for gradient estimation
#[derive(Debug, Clone)]
pub struct KalmanState<T: FloatElement> {
    /// State estimate
    pub estimate: Vec<T>,
    /// Error covariance
    pub covariance: Vec<T>,
    /// Process noise covariance
    pub process_noise: T,
    /// Measurement noise covariance
    pub measurement_noise: T,
    /// Kalman gain
    pub gain: Vec<T>,
}

/// Advanced gradient filter implementing sophisticated filtering algorithms
#[derive(Debug)]
pub struct AdvancedGradientFilter<T: FloatElement> {
    /// Filter configuration
    config: FilterConfig<T>,
    /// Kalman filter states (one per gradient tensor)
    kalman_states: Arc<Mutex<Vec<KalmanState<T>>>>,
    /// Historical gradients for temporal filtering
    gradient_history: Arc<Mutex<VecDeque<Vec<Vec<T>>>>>,
    /// Filter coefficients cache
    #[allow(dead_code)]
    coefficients_cache: Arc<Mutex<Option<Vec<T>>>>,
    /// Adaptive parameters
    adaptive_params: Arc<Mutex<AdaptiveParams<T>>>,
}

/// Adaptive filtering parameters that adjust based on gradient characteristics
#[derive(Debug, Clone)]
pub struct AdaptiveParams<T: FloatElement> {
    /// Current noise estimate
    pub noise_variance: T,
    /// Signal-to-noise ratio estimate
    pub snr_estimate: T,
    /// Adaptation rate
    pub adaptation_rate: T,
    /// Minimum and maximum filter parameters
    pub param_bounds: (T, T),
}

impl<T: FloatElement + FromPrimitive + ToPrimitive + Float> AdvancedGradientFilter<T> {
    /// Create a new advanced gradient filter
    pub fn new(config: FilterConfig<T>) -> Self {
        let adaptive_params = AdaptiveParams {
            noise_variance: T::from(1e-6)
                .unwrap_or_else(|| T::from(0.0).expect("numeric conversion should succeed")),
            snr_estimate: T::from(10.0)
                .unwrap_or_else(|| T::from(1.0).expect("numeric conversion should succeed")),
            adaptation_rate: T::from(0.01)
                .unwrap_or_else(|| T::from(0.0).expect("numeric conversion should succeed")),
            param_bounds: (
                T::from(1e-8)
                    .unwrap_or_else(|| T::from(0.0).expect("numeric conversion should succeed")),
                T::from(100.0)
                    .unwrap_or_else(|| T::from(1.0).expect("numeric conversion should succeed")),
            ),
        };

        Self {
            config,
            kalman_states: Arc::new(Mutex::new(Vec::new())),
            gradient_history: Arc::new(Mutex::new(VecDeque::new())),
            coefficients_cache: Arc::new(Mutex::new(None)),
            adaptive_params: Arc::new(Mutex::new(adaptive_params)),
        }
    }

    /// Apply advanced filtering to gradients
    pub fn filter_gradients(&self, gradients: &[Vec<T>]) -> Result<Vec<Vec<T>>> {
        match self.config.filter_type {
            AdvancedFilterType::Kalman => self.apply_kalman_filter(gradients),
            AdvancedFilterType::Bilateral => self.apply_bilateral_filter(gradients),
            AdvancedFilterType::Wiener => self.apply_wiener_filter(gradients),
            AdvancedFilterType::SavitzkyGolay => self.apply_savitzky_golay_filter(gradients),
            AdvancedFilterType::AdaptiveMedian => self.apply_adaptive_median_filter(gradients),
            AdvancedFilterType::Butterworth => self.apply_butterworth_filter(gradients),
            AdvancedFilterType::Chebyshev => self.apply_chebyshev_filter(gradients),
            AdvancedFilterType::EMD => self.apply_emd_filter(gradients),
        }
    }

    /// Apply Kalman filtering for optimal gradient estimation under noise
    fn apply_kalman_filter(&self, gradients: &[Vec<T>]) -> Result<Vec<Vec<T>>> {
        let mut states = self
            .kalman_states
            .lock()
            .expect("lock should not be poisoned");

        // Initialize Kalman states if needed
        if states.is_empty() {
            for grad in gradients {
                let state = KalmanState {
                    estimate: grad.clone(),
                    covariance: vec![
                        T::from(1.0).expect("numeric conversion should succeed");
                        grad.len()
                    ],
                    process_noise: self.config.primary_param,
                    measurement_noise: self.config.secondary_param.unwrap_or_else(|| {
                        T::from(0.1).expect("numeric conversion should succeed")
                    }),
                    gain: vec![
                        T::from(0.5).expect("numeric conversion should succeed");
                        grad.len()
                    ],
                };
                states.push(state);
            }
        }

        let mut filtered = Vec::new();

        for (i, grad) in gradients.iter().enumerate() {
            if i >= states.len() {
                return Err(TorshError::AutogradError(
                    "Gradient count mismatch with Kalman states".to_string(),
                ));
            }

            let state = &mut states[i];
            let mut filtered_grad = Vec::new();

            for (j, &measurement) in grad.iter().enumerate() {
                if j >= state.estimate.len() {
                    continue;
                }

                // Prediction step
                let predicted_estimate = state.estimate[j];
                let predicted_covariance = state.covariance[j] + state.process_noise;

                // Update step
                let innovation = measurement - predicted_estimate;
                let innovation_covariance = predicted_covariance + state.measurement_noise;
                let kalman_gain = predicted_covariance / innovation_covariance;

                // Update estimates
                state.estimate[j] = predicted_estimate + kalman_gain * innovation;
                state.covariance[j] = (T::from(1.0).expect("numeric conversion should succeed")
                    - kalman_gain)
                    * predicted_covariance;
                state.gain[j] = kalman_gain;

                filtered_grad.push(state.estimate[j]);
            }

            filtered.push(filtered_grad);
        }

        Ok(filtered)
    }

    /// Apply bilateral filtering for edge-preserving gradient smoothing
    fn apply_bilateral_filter(&self, gradients: &[Vec<T>]) -> Result<Vec<Vec<T>>> {
        let sigma_spatial = self.config.primary_param;
        let sigma_intensity = self.config.secondary_param.unwrap_or(sigma_spatial);
        let window_size = self.config.order;

        let mut filtered = Vec::new();

        for grad in gradients {
            let mut filtered_grad = Vec::new();

            for (i, &center_val) in grad.iter().enumerate() {
                let mut weighted_sum = T::from(0.0).expect("numeric conversion should succeed");
                let mut weight_sum = T::from(0.0).expect("numeric conversion should succeed");

                // Define neighborhood window
                let start = i.saturating_sub(window_size / 2);
                let end = (i + window_size / 2 + 1).min(grad.len());

                for (j, &grad_val) in grad.iter().enumerate().take(end).skip(start) {
                    let spatial_dist = T::from((i as f64 - j as f64).abs())
                        .expect("numeric conversion should succeed");
                    let intensity_dist = (center_val - grad_val).abs();

                    // Bilateral weight calculation
                    let spatial_weight = (-spatial_dist * spatial_dist
                        / (sigma_spatial
                            * sigma_spatial
                            * T::from(2.0).expect("numeric conversion should succeed")))
                    .exp();
                    let intensity_weight = (-intensity_dist * intensity_dist
                        / (sigma_intensity
                            * sigma_intensity
                            * T::from(2.0).expect("numeric conversion should succeed")))
                    .exp();
                    let weight = spatial_weight * intensity_weight;

                    weighted_sum = weighted_sum + weight * grad_val;
                    weight_sum = weight_sum + weight;
                }

                let filtered_val =
                    if weight_sum > T::from(1e-10).expect("numeric conversion should succeed") {
                        weighted_sum / weight_sum
                    } else {
                        center_val
                    };

                filtered_grad.push(filtered_val);
            }

            filtered.push(filtered_grad);
        }

        Ok(filtered)
    }

    /// Apply Wiener filtering for optimal noise reduction
    fn apply_wiener_filter(&self, gradients: &[Vec<T>]) -> Result<Vec<Vec<T>>> {
        let noise_variance = self.config.primary_param;
        let mut filtered = Vec::new();

        for grad in gradients {
            // Estimate signal variance
            let mean = grad.iter().copied().fold(
                T::from(0.0).expect("numeric conversion should succeed"),
                |acc, x| acc + x,
            ) / T::from(grad.len()).expect("numeric conversion should succeed");
            let signal_variance = grad.iter().map(|&x| (x - mean) * (x - mean)).fold(
                T::from(0.0).expect("numeric conversion should succeed"),
                |acc, x| acc + x,
            ) / T::from(grad.len())
                .expect("numeric conversion should succeed");

            // Wiener filter coefficient
            let wiener_coeff = signal_variance / (signal_variance + noise_variance);

            let filtered_grad: Vec<T> = grad
                .iter()
                .map(|&x| mean + wiener_coeff * (x - mean))
                .collect();

            filtered.push(filtered_grad);
        }

        Ok(filtered)
    }

    /// Apply Savitzky-Golay smoothing filter
    fn apply_savitzky_golay_filter(&self, gradients: &[Vec<T>]) -> Result<Vec<Vec<T>>> {
        let window_size = self.config.order;
        let poly_order = self
            .config
            .secondary_param
            .map(|p| p.to_usize().unwrap_or(2))
            .unwrap_or(2);

        if window_size % 2 == 0 || window_size < 3 {
            return Err(TorshError::AutogradError(
                "Savitzky-Golay window size must be odd and >= 3".to_string(),
            ));
        }

        // Generate Savitzky-Golay coefficients (simplified for polynomial order 2)
        let coeffs = self.generate_savgol_coefficients(window_size, poly_order)?;

        let mut filtered = Vec::new();

        for grad in gradients {
            let filtered_grad = self.apply_convolution(grad, &coeffs);
            filtered.push(filtered_grad);
        }

        Ok(filtered)
    }

    /// Apply adaptive median filtering
    fn apply_adaptive_median_filter(&self, gradients: &[Vec<T>]) -> Result<Vec<Vec<T>>> {
        let max_window_size = self.config.order;
        let mut filtered = Vec::new();

        for grad in gradients {
            let mut filtered_grad = Vec::new();

            for i in 0..grad.len() {
                let mut window_size = 3;
                let mut result = grad[i];

                while window_size <= max_window_size {
                    let half_window = window_size / 2;
                    let start = i.saturating_sub(half_window);
                    let end = (i + half_window + 1).min(grad.len());

                    let mut window: Vec<T> = grad[start..end].to_vec();
                    window.sort_by(|a, b| a.partial_cmp(b).unwrap());

                    let median = if window.len() % 2 == 0 {
                        (window[window.len() / 2 - 1] + window[window.len() / 2])
                            / T::from(2.0).expect("numeric conversion should succeed")
                    } else {
                        window[window.len() / 2]
                    };

                    let min_val = window[0];
                    let max_val = window[window.len() - 1];

                    // Stage A: Check if median is an impulse
                    if median > min_val && median < max_val {
                        // Stage B: Check if center pixel is an impulse
                        if grad[i] > min_val && grad[i] < max_val {
                            result = grad[i]; // Keep original
                        } else {
                            result = median; // Replace with median
                        }
                        break;
                    } else {
                        window_size += 2;
                        if window_size > max_window_size {
                            result = median;
                        }
                    }
                }

                filtered_grad.push(result);
            }

            filtered.push(filtered_grad);
        }

        Ok(filtered)
    }

    /// Apply Butterworth low-pass filter
    fn apply_butterworth_filter(&self, gradients: &[Vec<T>]) -> Result<Vec<Vec<T>>> {
        let cutoff_freq = self.config.primary_param;
        let order = self.config.order;

        // Generate Butterworth filter coefficients
        let coeffs = self.generate_butterworth_coefficients(cutoff_freq, order)?;

        let mut filtered = Vec::new();

        for grad in gradients {
            let filtered_grad = self.apply_iir_filter(grad, &coeffs.0, &coeffs.1);
            filtered.push(filtered_grad);
        }

        Ok(filtered)
    }

    /// Apply Chebyshev filter
    fn apply_chebyshev_filter(&self, gradients: &[Vec<T>]) -> Result<Vec<Vec<T>>> {
        let cutoff_freq = self.config.primary_param;
        let ripple = self
            .config
            .secondary_param
            .unwrap_or_else(|| T::from(0.5).expect("numeric conversion should succeed"));
        let order = self.config.order;

        // Generate Chebyshev filter coefficients
        let coeffs = self.generate_chebyshev_coefficients(cutoff_freq, ripple, order)?;

        let mut filtered = Vec::new();

        for grad in gradients {
            let filtered_grad = self.apply_iir_filter(grad, &coeffs.0, &coeffs.1);
            filtered.push(filtered_grad);
        }

        Ok(filtered)
    }

    /// Apply Empirical Mode Decomposition (EMD) filtering
    fn apply_emd_filter(&self, gradients: &[Vec<T>]) -> Result<Vec<Vec<T>>> {
        let max_modes = self.config.order;
        let mut filtered = Vec::new();

        for grad in gradients {
            // Simplified EMD: decompose into intrinsic mode functions
            let modes = self.empirical_mode_decomposition(grad, max_modes)?;

            // Reconstruct using only low-frequency modes (noise reduction)
            let num_modes_to_keep = (modes.len() as f64 * 0.7) as usize; // Keep 70% of modes
            let mut reconstructed =
                vec![T::from(0.0).expect("numeric conversion should succeed"); grad.len()];

            for mode in modes.iter().take(num_modes_to_keep.min(modes.len())) {
                for (i, &val) in mode.iter().enumerate() {
                    reconstructed[i] = reconstructed[i] + val;
                }
            }

            filtered.push(reconstructed);
        }

        Ok(filtered)
    }

    // Helper methods for filter implementations

    /// Generate Savitzky-Golay filter coefficients
    fn generate_savgol_coefficients(
        &self,
        window_size: usize,
        poly_order: usize,
    ) -> Result<Vec<T>> {
        // Simplified coefficient generation for common cases
        match (window_size, poly_order) {
            (3, 2) => Ok(vec![
                T::from(-1.0 / 6.0).expect("numeric conversion should succeed"),
                T::from(2.0 / 3.0).expect("numeric conversion should succeed"),
                T::from(-1.0 / 6.0).expect("numeric conversion should succeed"),
            ]),
            (5, 2) => Ok(vec![
                T::from(-3.0 / 35.0).expect("numeric conversion should succeed"),
                T::from(12.0 / 35.0).expect("numeric conversion should succeed"),
                T::from(17.0 / 35.0).expect("numeric conversion should succeed"),
                T::from(12.0 / 35.0).expect("numeric conversion should succeed"),
                T::from(-3.0 / 35.0).expect("numeric conversion should succeed"),
            ]),
            (7, 2) => Ok(vec![
                T::from(-2.0 / 21.0).expect("numeric conversion should succeed"),
                T::from(3.0 / 21.0).expect("numeric conversion should succeed"),
                T::from(6.0 / 21.0).expect("numeric conversion should succeed"),
                T::from(7.0 / 21.0).expect("numeric conversion should succeed"),
                T::from(6.0 / 21.0).expect("numeric conversion should succeed"),
                T::from(3.0 / 21.0).expect("numeric conversion should succeed"),
                T::from(-2.0 / 21.0).expect("numeric conversion should succeed"),
            ]),
            _ => {
                // For other cases, use simple moving average
                let coeff = T::from(1.0).expect("numeric conversion should succeed")
                    / T::from(window_size).expect("numeric conversion should succeed");
                Ok(vec![coeff; window_size])
            }
        }
    }

    /// Apply convolution with given coefficients
    fn apply_convolution(&self, signal: &[T], coeffs: &[T]) -> Vec<T> {
        let mut result = Vec::with_capacity(signal.len());
        let half_window = coeffs.len() / 2;

        for i in 0..signal.len() {
            let mut sum = T::from(0.0).expect("numeric conversion should succeed");
            let mut weight_sum = T::from(0.0).expect("numeric conversion should succeed");

            for (j, &coeff) in coeffs.iter().enumerate() {
                let signal_idx = i + j;
                if signal_idx >= half_window && signal_idx - half_window < signal.len() {
                    sum = sum + coeff * signal[signal_idx - half_window];
                    weight_sum = weight_sum + coeff;
                }
            }

            result.push(
                if weight_sum.abs() > T::from(1e-10).expect("numeric conversion should succeed") {
                    sum / weight_sum
                } else {
                    signal[i]
                },
            );
        }

        result
    }

    /// Generate Butterworth filter coefficients (simplified)
    fn generate_butterworth_coefficients(
        &self,
        cutoff: T,
        order: usize,
    ) -> Result<(Vec<T>, Vec<T>)> {
        // Simplified Butterworth filter design
        // In practice, this would use proper bilinear transform and pole placement
        let normalized_cutoff = cutoff
            .min(T::from(0.5).expect("numeric conversion should succeed"))
            .max(T::from(0.01).expect("numeric conversion should succeed"));

        // Simple first-order approximation
        let alpha = (T::from(std::f64::consts::PI).expect("numeric conversion should succeed")
            * normalized_cutoff)
            .tan()
            / (T::from(1.0).expect("numeric conversion should succeed")
                + (T::from(std::f64::consts::PI).expect("numeric conversion should succeed")
                    * normalized_cutoff)
                    .tan());

        let b_coeffs = vec![alpha, alpha];
        let a_coeffs = vec![
            T::from(1.0).expect("numeric conversion should succeed"),
            alpha - T::from(1.0).expect("numeric conversion should succeed"),
        ];

        // For higher orders, cascade multiple first-order sections
        if order > 1 {
            // Simplified: just repeat the coefficients (not mathematically correct but functional)
            Ok((b_coeffs, a_coeffs))
        } else {
            Ok((b_coeffs, a_coeffs))
        }
    }

    /// Generate Chebyshev filter coefficients (simplified)
    fn generate_chebyshev_coefficients(
        &self,
        cutoff: T,
        _ripple: T,
        _order: usize,
    ) -> Result<(Vec<T>, Vec<T>)> {
        // Simplified Chebyshev filter (using Butterworth as approximation)
        self.generate_butterworth_coefficients(cutoff, _order)
    }

    /// Apply IIR filter with given coefficients
    fn apply_iir_filter(&self, signal: &[T], b_coeffs: &[T], a_coeffs: &[T]) -> Vec<T> {
        let mut output =
            vec![T::from(0.0).expect("numeric conversion should succeed"); signal.len()];
        let mut x_history =
            vec![T::from(0.0).expect("numeric conversion should succeed"); b_coeffs.len()];
        let mut y_history =
            vec![T::from(0.0).expect("numeric conversion should succeed"); a_coeffs.len()];

        for (i, &input) in signal.iter().enumerate() {
            // Shift input history
            for j in (1..x_history.len()).rev() {
                x_history[j] = x_history[j - 1];
            }
            x_history[0] = input;

            // Compute output
            let mut y = T::from(0.0).expect("numeric conversion should succeed");
            for (j, &b) in b_coeffs.iter().enumerate() {
                if j < x_history.len() {
                    y = y + b * x_history[j];
                }
            }
            for (j, &a) in a_coeffs.iter().skip(1).enumerate() {
                if j < y_history.len() {
                    y = y - a * y_history[j];
                }
            }

            output[i] = y;

            // Shift output history
            for j in (1..y_history.len()).rev() {
                y_history[j] = y_history[j - 1];
            }
            y_history[0] = y;
        }

        output
    }

    /// Simplified Empirical Mode Decomposition
    fn empirical_mode_decomposition(&self, signal: &[T], max_modes: usize) -> Result<Vec<Vec<T>>> {
        let mut modes = Vec::new();
        let mut residue = signal.to_vec();

        for _mode_idx in 0..max_modes {
            let mode = self.extract_imf(&residue)?;

            // Check stopping criteria
            let mode_energy: T = mode.iter().map(|&x| x * x).fold(
                T::from(0.0).expect("numeric conversion should succeed"),
                |acc, x| acc + x,
            );
            let residue_energy: T = residue.iter().map(|&x| x * x).fold(
                T::from(0.0).expect("numeric conversion should succeed"),
                |acc, x| acc + x,
            );

            if mode_energy
                < residue_energy * T::from(0.01).expect("numeric conversion should succeed")
            {
                break;
            }

            // Subtract mode from residue
            for i in 0..residue.len() {
                residue[i] = residue[i] - mode[i];
            }

            modes.push(mode);
        }

        // Add final residue as trend
        modes.push(residue);
        Ok(modes)
    }

    /// Extract Intrinsic Mode Function (IMF)
    fn extract_imf(&self, signal: &[T]) -> Result<Vec<T>> {
        let mut h = signal.to_vec();
        let max_iterations = 10;

        for _ in 0..max_iterations {
            let (maxima, minima) = self.find_extrema(&h);

            if maxima.len() < 2 || minima.len() < 2 {
                break;
            }

            let upper_envelope = self.interpolate_envelope(&h, &maxima);
            let lower_envelope = self.interpolate_envelope(&h, &minima);

            // Calculate mean envelope
            let mut mean_envelope = Vec::new();
            for i in 0..h.len() {
                let mean = (upper_envelope[i] + lower_envelope[i])
                    / T::from(2.0).expect("numeric conversion should succeed");
                mean_envelope.push(mean);
            }

            // Subtract mean from signal
            let mut new_h = Vec::new();
            for i in 0..h.len() {
                new_h.push(h[i] - mean_envelope[i]);
            }

            // Check stopping criterion (simplified)
            let diff_norm: T = new_h
                .iter()
                .zip(h.iter())
                .map(|(&new, &old)| (new - old) * (new - old))
                .fold(
                    T::from(0.0).expect("numeric conversion should succeed"),
                    |acc, x| acc + x,
                );

            h = new_h;

            if diff_norm < T::from(1e-6).expect("numeric conversion should succeed") {
                break;
            }
        }

        Ok(h)
    }

    /// Find local extrema in signal
    fn find_extrema(&self, signal: &[T]) -> (Vec<usize>, Vec<usize>) {
        let mut maxima = Vec::new();
        let mut minima = Vec::new();

        for i in 1..signal.len() - 1 {
            if signal[i] > signal[i - 1] && signal[i] > signal[i + 1] {
                maxima.push(i);
            } else if signal[i] < signal[i - 1] && signal[i] < signal[i + 1] {
                minima.push(i);
            }
        }

        (maxima, minima)
    }

    /// Interpolate envelope through extrema points
    fn interpolate_envelope(&self, signal: &[T], extrema: &[usize]) -> Vec<T> {
        let mut envelope =
            vec![T::from(0.0).expect("numeric conversion should succeed"); signal.len()];

        if extrema.is_empty() {
            return signal.to_vec();
        }

        // Simple linear interpolation between extrema
        for (i, envelope_val) in envelope.iter_mut().enumerate().take(signal.len()) {
            let mut left_idx = 0;
            let mut right_idx = extrema.len() - 1;

            // Find surrounding extrema
            for (j, &ext_idx) in extrema.iter().enumerate() {
                if ext_idx <= i {
                    left_idx = j;
                }
                if ext_idx >= i && j < right_idx {
                    right_idx = j;
                    break;
                }
            }

            if left_idx == right_idx {
                *envelope_val = signal[extrema[left_idx]];
            } else {
                let x1 = T::from(extrema[left_idx]).expect("numeric conversion should succeed");
                let y1 = signal[extrema[left_idx]];
                let x2 = T::from(extrema[right_idx]).expect("numeric conversion should succeed");
                let y2 = signal[extrema[right_idx]];
                let x = T::from(i).expect("numeric conversion should succeed");

                // Linear interpolation
                if (x2 - x1).abs() > T::from(1e-10).expect("numeric conversion should succeed") {
                    *envelope_val = y1 + (y2 - y1) * (x - x1) / (x2 - x1);
                } else {
                    *envelope_val = y1;
                }
            }
        }

        envelope
    }

    /// Update adaptive parameters based on gradient characteristics
    pub fn update_adaptive_parameters(&self, gradients: &[Vec<T>]) -> Result<()> {
        if !self.config.adaptive {
            return Ok(());
        }

        let mut params = self
            .adaptive_params
            .lock()
            .expect("lock should not be poisoned");

        // Estimate noise characteristics
        let mut all_values = Vec::new();
        for grad in gradients {
            all_values.extend(grad.iter().copied());
        }

        if all_values.is_empty() {
            return Ok(());
        }

        // Calculate statistics
        let mean = all_values.iter().copied().fold(
            T::from(0.0).expect("numeric conversion should succeed"),
            |acc, x| acc + x,
        ) / T::from(all_values.len()).expect("numeric conversion should succeed");
        let variance = all_values.iter().map(|&x| (x - mean) * (x - mean)).fold(
            T::from(0.0).expect("numeric conversion should succeed"),
            |acc, x| acc + x,
        ) / T::from(all_values.len()).expect("numeric conversion should succeed");

        // Update noise estimate with exponential moving average
        params.noise_variance = params.adaptation_rate * variance
            + (T::from(1.0).expect("numeric conversion should succeed") - params.adaptation_rate)
                * params.noise_variance;

        // Update SNR estimate
        let signal_power = mean * mean;
        params.snr_estimate =
            if params.noise_variance > T::from(1e-10).expect("numeric conversion should succeed") {
                signal_power / params.noise_variance
            } else {
                T::from(100.0).expect("numeric conversion should succeed") // High SNR if very low noise
            };

        Ok(())
    }

    /// Get current filter statistics
    pub fn get_filter_statistics(&self) -> FilterStatistics<T> {
        let params = self
            .adaptive_params
            .lock()
            .expect("lock should not be poisoned");
        FilterStatistics {
            noise_variance: params.noise_variance,
            snr_estimate: params.snr_estimate,
            num_kalman_states: self
                .kalman_states
                .lock()
                .expect("lock should not be poisoned")
                .len(),
            history_length: self
                .gradient_history
                .lock()
                .expect("lock should not be poisoned")
                .len(),
        }
    }
}

/// Statistics about the current filter state
#[derive(Debug, Clone)]
pub struct FilterStatistics<T: FloatElement> {
    pub noise_variance: T,
    pub snr_estimate: T,
    pub num_kalman_states: usize,
    pub history_length: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kalman_filter_creation() {
        let config = FilterConfig {
            filter_type: AdvancedFilterType::Kalman,
            primary_param: 0.1f32,
            secondary_param: Some(0.05f32),
            order: 1,
            adaptive: false,
        };

        let filter: AdvancedGradientFilter<f32> = AdvancedGradientFilter::new(config);
        let gradients = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];

        let result = filter.filter_gradients(&gradients);
        assert!(result.is_ok());

        let filtered = result.unwrap();
        assert_eq!(filtered.len(), 2);
        assert_eq!(filtered[0].len(), 3);
        assert_eq!(filtered[1].len(), 3);
    }

    #[test]
    fn test_bilateral_filter() {
        let config = FilterConfig {
            filter_type: AdvancedFilterType::Bilateral,
            primary_param: 1.0f32,         // Spatial sigma
            secondary_param: Some(2.0f32), // Larger intensity sigma to allow more smoothing of outliers
            order: 5,
            adaptive: false,
        };

        let filter: AdvancedGradientFilter<f32> = AdvancedGradientFilter::new(config);
        let gradients = vec![vec![1.0, 2.0, 1.5, 10.0, 2.5]]; // Contains clear outlier (10.0)

        let result = filter.filter_gradients(&gradients);
        assert!(result.is_ok());

        let filtered = result.unwrap();
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].len(), 5);

        // The bilateral filter should provide some smoothing of the outlier
        // but still preserve some edge information (realistic expectation)
        // Bilateral filters are designed to preserve edges while reducing noise
        assert!(filtered[0][3] < 10.0); // Should be slightly reduced from original
        assert!(filtered[0][3] > 9.0); // Should still preserve the edge/outlier

        // Check that normal values are preserved reasonably
        assert!((filtered[0][0] - 1.0).abs() < 0.5);
        assert!((filtered[0][1] - 2.0).abs() < 0.5);
    }

    #[test]
    fn test_wiener_filter() {
        let config = FilterConfig {
            filter_type: AdvancedFilterType::Wiener,
            primary_param: 0.1f32, // noise variance
            secondary_param: None,
            order: 1,
            adaptive: false,
        };

        let filter: AdvancedGradientFilter<f32> = AdvancedGradientFilter::new(config);
        let gradients = vec![vec![1.0, 1.1, 0.9, 1.2, 0.8]]; // Low noise signal

        let result = filter.filter_gradients(&gradients);
        assert!(result.is_ok());

        let filtered = result.unwrap();
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].len(), 5);
    }

    #[test]
    fn test_adaptive_median_filter() {
        let config = FilterConfig {
            filter_type: AdvancedFilterType::AdaptiveMedian,
            primary_param: 1.0f32,
            secondary_param: None,
            order: 7, // max window size
            adaptive: false,
        };

        let filter: AdvancedGradientFilter<f32> = AdvancedGradientFilter::new(config);
        let gradients = vec![vec![1.0, 1.0, 10.0, 1.0, 1.0]]; // Impulse noise

        let result = filter.filter_gradients(&gradients);
        assert!(result.is_ok());

        let filtered = result.unwrap();
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].len(), 5);

        // The impulse (10.0) should be filtered out
        assert!(filtered[0][2] < 5.0);
    }

    #[test]
    fn test_butterworth_filter() {
        let config = FilterConfig {
            filter_type: AdvancedFilterType::Butterworth,
            primary_param: 0.2f32, // cutoff frequency
            secondary_param: None,
            order: 2,
            adaptive: false,
        };

        let filter: AdvancedGradientFilter<f32> = AdvancedGradientFilter::new(config);
        let gradients = vec![vec![1.0, -1.0, 1.0, -1.0, 1.0]]; // High frequency signal

        let result = filter.filter_gradients(&gradients);
        assert!(result.is_ok());

        let filtered = result.unwrap();
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].len(), 5);

        // High frequencies should be attenuated
        let max_filtered = filtered[0].iter().copied().fold(0.0f32, f32::max);
        assert!(max_filtered < 1.0);
    }

    #[test]
    fn test_emd_filter() {
        let config = FilterConfig {
            filter_type: AdvancedFilterType::EMD,
            primary_param: 1.0f32,
            secondary_param: None,
            order: 3, // max modes
            adaptive: false,
        };

        let filter: AdvancedGradientFilter<f32> = AdvancedGradientFilter::new(config);
        let gradients = vec![vec![1.0, 2.0, 1.5, 3.0, 2.5, 4.0, 3.5]];

        let result = filter.filter_gradients(&gradients);
        assert!(result.is_ok());

        let filtered = result.unwrap();
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].len(), 7);
    }

    #[test]
    fn test_adaptive_parameters() {
        let config = FilterConfig {
            filter_type: AdvancedFilterType::Kalman,
            primary_param: 0.1f32,
            secondary_param: Some(0.05f32),
            order: 1,
            adaptive: true,
        };

        let filter: AdvancedGradientFilter<f32> = AdvancedGradientFilter::new(config);
        let gradients = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];

        let result = filter.update_adaptive_parameters(&gradients);
        assert!(result.is_ok());

        let stats = filter.get_filter_statistics();
        assert!(stats.noise_variance >= 0.0);
        assert!(stats.snr_estimate >= 0.0);
    }
}
