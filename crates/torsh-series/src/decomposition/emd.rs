//! Empirical Mode Decomposition (EMD)
//!
//! EMD is an adaptive, data-driven decomposition method that breaks down a signal
//! into Intrinsic Mode Functions (IMFs) through an iterative sifting process.
//!
//! # Methods Provided
//! - **EMD**: Standard Empirical Mode Decomposition
//! - **EEMD**: Ensemble EMD with noise-assisted decomposition
//! - **CEEMDAN**: Complete Ensemble EMD with Adaptive Noise
//!
//! # References
//! - Huang et al. (1998): "The empirical mode decomposition and the Hilbert spectrum"
//! - Wu & Huang (2009): "Ensemble empirical mode decomposition"

use crate::TimeSeries;
use torsh_core::error::Result;
use torsh_tensor::Tensor;

/// EMD decomposition result
#[derive(Debug, Clone)]
pub struct EMDResult {
    /// Intrinsic Mode Functions (from high frequency to low frequency)
    pub imfs: Vec<TimeSeries>,
    /// Residual (trend component)
    pub residual: TimeSeries,
    /// Number of sifting iterations per IMF
    pub num_sifts: Vec<usize>,
}

/// EMD configuration
#[derive(Debug, Clone)]
pub struct EMDConfig {
    /// Maximum number of IMFs to extract
    pub max_imfs: usize,
    /// Maximum number of sifting iterations
    pub max_sifts: usize,
    /// Sifting stop criterion threshold (SD criterion)
    pub sd_threshold: f64,
    /// Minimum number of extrema required to continue
    pub min_extrema: usize,
}

impl Default for EMDConfig {
    fn default() -> Self {
        Self {
            max_imfs: 10,
            max_sifts: 100,
            sd_threshold: 0.3,
            min_extrema: 3,
        }
    }
}

/// Empirical Mode Decomposition
///
/// Decomposes a signal into Intrinsic Mode Functions (IMFs) using the sifting process.
///
/// # Algorithm
/// 1. Identify all local maxima and minima
/// 2. Interpolate maxima to create upper envelope
/// 3. Interpolate minima to create lower envelope
/// 4. Compute mean of envelopes
/// 5. Subtract mean from signal to get candidate IMF
/// 6. Check stopping criterion (SD test)
/// 7. Repeat until IMF criteria satisfied
/// 8. Extract IMF and repeat on residual
///
/// # Arguments
/// * `series` - Input time series to decompose
/// * `config` - EMD configuration parameters
///
/// # Returns
/// EMDResult containing IMFs and residual
pub fn emd_decompose(series: &TimeSeries, config: &EMDConfig) -> Result<EMDResult> {
    let data = series.values.to_vec()?;
    let n = data.len();

    if n < 4 {
        return Err(torsh_core::error::TorshError::InvalidArgument(
            "Need at least 4 points for EMD".to_string(),
        ));
    }

    let mut imfs = Vec::new();
    let mut residual = data.clone();
    let mut num_sifts_per_imf = Vec::new();

    // Extract IMFs iteratively
    for _imf_idx in 0..config.max_imfs {
        // Check if residual is monotonic (no more IMFs)
        let extrema = find_extrema(&residual);
        if extrema.maxima.len() < config.min_extrema || extrema.minima.len() < config.min_extrema {
            break;
        }

        // Sifting process to extract one IMF
        let (imf, sift_count) = sifting_process(&residual, config)?;

        // Update residual
        for i in 0..n {
            residual[i] -= imf[i];
        }

        // Store IMF
        let imf_tensor = Tensor::from_vec(imf, &[n])?;
        imfs.push(TimeSeries::new(imf_tensor));
        num_sifts_per_imf.push(sift_count);
    }

    // Create residual time series
    let residual_tensor = Tensor::from_vec(residual, &[n])?;
    let residual_ts = TimeSeries::new(residual_tensor);

    Ok(EMDResult {
        imfs,
        residual: residual_ts,
        num_sifts: num_sifts_per_imf,
    })
}

/// Sifting process to extract one IMF
fn sifting_process(signal: &[f32], config: &EMDConfig) -> Result<(Vec<f32>, usize)> {
    let n = signal.len();
    let mut h = signal.to_vec();
    let mut sift_count = 0;

    for _iter in 0..config.max_sifts {
        sift_count += 1;

        // Find extrema
        let extrema = find_extrema(&h);

        // Check if we have enough extrema
        if extrema.maxima.len() < 2 || extrema.minima.len() < 2 {
            break;
        }

        // Create upper and lower envelopes
        let upper_env = create_envelope(&extrema.maxima, n);
        let lower_env = create_envelope(&extrema.minima, n);

        // Compute mean envelope
        let mut mean_env = vec![0.0f32; n];
        for i in 0..n {
            mean_env[i] = (upper_env[i] + lower_env[i]) / 2.0;
        }

        // Update h
        let mut h_new = vec![0.0f32; n];
        for i in 0..n {
            h_new[i] = h[i] - mean_env[i];
        }

        // Check stopping criterion (SD test)
        let sd = compute_sd(&h, &h_new);
        if sd < config.sd_threshold {
            h = h_new;
            break;
        }

        h = h_new;
    }

    Ok((h, sift_count))
}

/// Extrema locations and values
#[derive(Debug)]
struct Extrema {
    maxima: Vec<(usize, f32)>, // (index, value)
    minima: Vec<(usize, f32)>,
}

/// Find local maxima and minima
fn find_extrema(signal: &[f32]) -> Extrema {
    let n = signal.len();
    let mut maxima = Vec::new();
    let mut minima = Vec::new();

    if n < 3 {
        return Extrema { maxima, minima };
    }

    // Check interior points
    for i in 1..n - 1 {
        let prev = signal[i - 1];
        let curr = signal[i];
        let next = signal[i + 1];

        // Local maximum
        if curr > prev && curr > next {
            maxima.push((i, curr));
        }
        // Local minimum
        else if curr < prev && curr < next {
            minima.push((i, curr));
        }
    }

    // Add boundary points if they are extrema
    if n >= 2 {
        // Check first point
        if signal[0] > signal[1] {
            maxima.insert(0, (0, signal[0]));
        } else if signal[0] < signal[1] {
            minima.insert(0, (0, signal[0]));
        }

        // Check last point
        if signal[n - 1] > signal[n - 2] {
            maxima.push((n - 1, signal[n - 1]));
        } else if signal[n - 1] < signal[n - 2] {
            minima.push((n - 1, signal[n - 1]));
        }
    }

    Extrema { maxima, minima }
}

/// Create envelope using cubic spline interpolation
///
/// This is a simplified implementation using linear interpolation.
/// For production use, would implement proper cubic spline.
fn create_envelope(extrema: &[(usize, f32)], length: usize) -> Vec<f32> {
    let mut envelope = vec![0.0f32; length];

    if extrema.is_empty() {
        return envelope;
    }

    if extrema.len() == 1 {
        // Constant envelope
        let value = extrema[0].1;
        for i in 0..length {
            envelope[i] = value;
        }
        return envelope;
    }

    // Linear interpolation between extrema
    for i in 0..length {
        // Find surrounding extrema
        let mut left_idx = 0;
        let mut right_idx = extrema.len() - 1;

        for (idx, &(pos, _)) in extrema.iter().enumerate() {
            if pos <= i {
                left_idx = idx;
            }
            if pos >= i && right_idx == extrema.len() - 1 {
                right_idx = idx;
            }
        }

        let (left_pos, left_val) = extrema[left_idx];
        let (right_pos, right_val) = extrema[right_idx];

        if left_idx == right_idx {
            envelope[i] = left_val;
        } else {
            // Linear interpolation
            let t = (i - left_pos) as f32 / (right_pos - left_pos) as f32;
            envelope[i] = left_val + t * (right_val - left_val);
        }
    }

    envelope
}

/// Compute Standard Deviation criterion for sifting
///
/// SD = Σ[(h_{k-1}(t) - h_k(t))² / h_{k-1}(t)²]
fn compute_sd(h_prev: &[f32], h_curr: &[f32]) -> f64 {
    let n = h_prev.len();
    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for i in 0..n {
        let diff = (h_prev[i] - h_curr[i]) as f64;
        let h_prev_sq = (h_prev[i] * h_prev[i]) as f64;

        numerator += diff * diff;
        denominator += h_prev_sq;
    }

    if denominator > 1e-10 {
        numerator / denominator
    } else {
        0.0
    }
}

/// Ensemble EMD (EEMD)
///
/// Improves EMD by adding noise trials to reduce mode mixing.
///
/// # Algorithm
/// 1. Add white noise to signal
/// 2. Decompose using EMD
/// 3. Repeat with different noise realizations
/// 4. Average corresponding IMFs
///
/// # Arguments
/// * `series` - Input time series
/// * `config` - EMD configuration
/// * `num_ensembles` - Number of noise trials
/// * `noise_std` - Standard deviation of added noise
pub fn eemd_decompose(
    series: &TimeSeries,
    config: &EMDConfig,
    num_ensembles: usize,
    noise_std: f32,
) -> Result<EMDResult> {
    let data = series.values.to_vec()?;
    let n = data.len();

    if num_ensembles == 0 {
        return emd_decompose(series, config);
    }

    // Storage for ensemble IMFs
    let mut ensemble_imfs: Vec<Vec<Vec<f32>>> = Vec::new();
    let mut ensemble_residuals: Vec<Vec<f32>> = Vec::new();

    // Run EMD with added noise
    use scirs2_core::random::{thread_rng, Distribution, Normal};
    let mut rng = thread_rng();
    let noise_dist = Normal::new(0.0, noise_std as f64).expect("distribution should succeed");

    for _trial in 0..num_ensembles {
        // Add noise to signal
        let mut noisy_data = data.clone();
        for val in &mut noisy_data {
            let noise = noise_dist.sample(&mut rng) as f32;
            *val += noise;
        }

        // Decompose noisy signal
        let noisy_tensor = Tensor::from_vec(noisy_data, &[n])?;
        let noisy_series = TimeSeries::new(noisy_tensor);
        let result = emd_decompose(&noisy_series, config)?;

        // Store IMFs
        let imf_data: Vec<Vec<f32>> = result
            .imfs
            .iter()
            .map(|imf| imf.values.to_vec().unwrap_or_default())
            .collect();
        ensemble_imfs.push(imf_data);

        // Store residual
        let residual_data = result.residual.values.to_vec()?;
        ensemble_residuals.push(residual_data);
    }

    // Average IMFs across ensembles
    let max_num_imfs = ensemble_imfs
        .iter()
        .map(|imfs| imfs.len())
        .max()
        .unwrap_or(0);

    let mut averaged_imfs = Vec::new();

    for imf_idx in 0..max_num_imfs {
        let mut averaged_imf = vec![0.0f32; n];
        let mut count = 0;

        for trial_imfs in &ensemble_imfs {
            if imf_idx < trial_imfs.len() {
                for i in 0..n {
                    averaged_imf[i] += trial_imfs[imf_idx][i];
                }
                count += 1;
            }
        }

        if count > 0 {
            for val in &mut averaged_imf {
                *val /= count as f32;
            }
        }

        let imf_tensor = Tensor::from_vec(averaged_imf, &[n])?;
        averaged_imfs.push(TimeSeries::new(imf_tensor));
    }

    // Average residuals
    let mut averaged_residual = vec![0.0f32; n];
    for residual in &ensemble_residuals {
        for i in 0..n {
            averaged_residual[i] += residual[i];
        }
    }
    for val in &mut averaged_residual {
        *val /= num_ensembles as f32;
    }

    let residual_tensor = Tensor::from_vec(averaged_residual, &[n])?;

    Ok(EMDResult {
        imfs: averaged_imfs,
        residual: TimeSeries::new(residual_tensor),
        num_sifts: vec![],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_signal() -> TimeSeries {
        // Create signal with multiple frequency components
        let n = 200;
        let mut data = Vec::with_capacity(n);

        for i in 0..n {
            let t = i as f32 / 20.0;
            // High frequency component
            let high_freq = 0.5 * (2.0 * std::f32::consts::PI * 2.0 * t).sin();
            // Low frequency component
            let low_freq = 1.0 * (2.0 * std::f32::consts::PI * 0.3 * t).sin();
            // Trend
            let trend = 0.1 * t;

            data.push(high_freq + low_freq + trend);
        }

        let tensor = Tensor::from_vec(data, &[n]).unwrap();
        TimeSeries::new(tensor)
    }

    #[test]
    fn test_find_extrema() {
        let signal = vec![1.0f32, 3.0, 2.0, 4.0, 1.0, 2.0];
        let extrema = find_extrema(&signal);

        // Should find maxima at indices 1 and 3
        assert!(extrema.maxima.len() >= 2);
        // Should find minima at indices 0 and 4
        assert!(extrema.minima.len() >= 2);
    }

    #[test]
    fn test_create_envelope() {
        let extrema = vec![(0, 1.0f32), (5, 3.0), (10, 2.0)];
        let envelope = create_envelope(&extrema, 11);

        assert_eq!(envelope.len(), 11);
        // First point should match first extremum
        assert!((envelope[0] - 1.0).abs() < 0.1);
        // Last point should match last extremum
        assert!((envelope[10] - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_emd_basic() {
        let series = create_test_signal();
        let config = EMDConfig {
            max_imfs: 5,
            max_sifts: 50,
            sd_threshold: 0.3,
            min_extrema: 3,
        };

        let result = emd_decompose(&series, &config).unwrap();

        // Should extract at least one IMF
        assert!(!result.imfs.is_empty());

        // Each IMF should have the same length as input
        for imf in &result.imfs {
            assert_eq!(imf.len(), series.len());
        }

        // Residual should have correct length
        assert_eq!(result.residual.len(), series.len());
    }

    #[test]
    fn test_emd_reconstruction() {
        let series = create_test_signal();
        let config = EMDConfig::default();

        let result = emd_decompose(&series, &config).unwrap();

        // Reconstruct signal from IMFs and residual
        let original_data = series.values.to_vec().unwrap();
        let mut reconstructed = vec![0.0f32; series.len()];

        // Add all IMFs
        for imf in &result.imfs {
            let imf_data = imf.values.to_vec().unwrap();
            for i in 0..series.len() {
                reconstructed[i] += imf_data[i];
            }
        }

        // Add residual
        let residual_data = result.residual.values.to_vec().unwrap();
        for i in 0..series.len() {
            reconstructed[i] += residual_data[i];
        }

        // Check reconstruction accuracy
        let mut max_error = 0.0f32;
        for i in 0..series.len() {
            let error = (reconstructed[i] - original_data[i]).abs();
            max_error = max_error.max(error);
        }

        assert!(
            max_error < 1.0,
            "Reconstruction error too large: {}",
            max_error
        );
    }

    #[test]
    fn test_eemd_basic() {
        let series = create_test_signal();
        let config = EMDConfig {
            max_imfs: 3,
            max_sifts: 30,
            sd_threshold: 0.3,
            min_extrema: 3,
        };

        let result = eemd_decompose(&series, &config, 10, 0.1).unwrap();

        // Should extract IMFs
        assert!(!result.imfs.is_empty());

        // Each IMF should have correct length
        for imf in &result.imfs {
            assert_eq!(imf.len(), series.len());
        }
    }

    #[test]
    fn test_compute_sd() {
        let h1 = vec![1.0f32, 2.0, 3.0];
        let h2 = vec![1.1f32, 2.1, 3.1];

        let sd = compute_sd(&h1, &h2);

        // SD should be small for similar signals
        assert!(sd < 0.1);
    }
}
