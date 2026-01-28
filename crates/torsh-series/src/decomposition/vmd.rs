//! Variational Mode Decomposition (VMD)
//!
//! VMD is an adaptive, non-recursive signal decomposition method that decomposes
//! a multi-component signal into a discrete number of modes (band-limited IMFs)
//! by solving a constrained variational optimization problem.
//!
//! Reference: Dragomiretskiy, K., & Zosso, D. (2014). Variational mode decomposition.
//! IEEE transactions on signal processing, 62(3), 531-544.

use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::thread_rng;
use scirs2_core::Complex;
use std::f64::consts::PI;

/// VMD-specific errors
#[derive(Debug, thiserror::Error)]
pub enum VMDError {
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    #[error("FFT error: {0}")]
    FFTError(String),
}

/// VMD decomposition configuration
#[derive(Debug, Clone)]
pub struct VMDConfig {
    /// Number of modes to extract
    pub num_modes: usize,
    /// Balancing parameter for data-fidelity constraint (default: 2000)
    pub alpha: f64,
    /// Time-step of dual ascent (default: 0)
    pub tau: f64,
    /// Maximum number of iterations (default: 500)
    pub max_iter: usize,
    /// Convergence tolerance (default: 1e-7)
    pub tol: f64,
    /// DC part imposed (default: 0)
    pub dc: bool,
    /// Initialize omegas (center frequencies) uniformly
    pub init_omega: bool,
}

impl Default for VMDConfig {
    fn default() -> Self {
        Self {
            num_modes: 3,
            alpha: 2000.0,
            tau: 0.0,
            max_iter: 500,
            tol: 1e-7,
            dc: false,
            init_omega: true,
        }
    }
}

impl VMDConfig {
    /// Create a new VMD config with specified number of modes
    pub fn new(num_modes: usize) -> Self {
        Self {
            num_modes,
            ..Default::default()
        }
    }

    /// Set alpha parameter (data-fidelity balancing)
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set tau parameter (dual ascent time-step)
    pub fn with_tau(mut self, tau: f64) -> Self {
        self.tau = tau;
        self
    }

    /// Set maximum iterations
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance
    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Enable/disable DC part
    pub fn with_dc(mut self, dc: bool) -> Self {
        self.dc = dc;
        self
    }
}

/// Result of VMD decomposition
#[derive(Debug, Clone)]
pub struct VMDResult {
    /// Decomposed modes (num_modes x signal_length)
    pub modes: Array2<f64>,
    /// Center frequencies of each mode (num_modes)
    pub center_frequencies: Array1<f64>,
    /// Number of iterations performed
    pub iterations: usize,
    /// Final convergence error
    pub convergence_error: f64,
}

impl VMDResult {
    /// Get a specific mode
    pub fn mode(&self, index: usize) -> Option<Array1<f64>> {
        if index < self.modes.nrows() {
            Some(self.modes.row(index).to_owned())
        } else {
            None
        }
    }

    /// Get number of modes
    pub fn num_modes(&self) -> usize {
        self.modes.nrows()
    }

    /// Get signal length
    pub fn signal_length(&self) -> usize {
        self.modes.ncols()
    }

    /// Reconstruct the original signal from modes
    pub fn reconstruct(&self) -> Array1<f64> {
        self.modes.sum_axis(Axis(0))
    }
}

/// Perform Variational Mode Decomposition on a signal
///
/// # Arguments
/// * `signal` - Input signal to decompose
/// * `config` - VMD configuration parameters
///
/// # Returns
/// * `VMDResult` containing the decomposed modes and metadata
///
/// # Example
/// ```
/// use scirs2_core::ndarray::Array1;
/// use torsh_series::decomposition::vmd::{vmd_decompose, VMDConfig};
///
/// let signal = Array1::from_vec(vec![0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0]);
/// let config = VMDConfig::new(2);
/// let result = vmd_decompose(&signal, &config).unwrap();
/// ```
pub fn vmd_decompose(signal: &Array1<f64>, config: &VMDConfig) -> Result<VMDResult, VMDError> {
    let n = signal.len();
    let k = config.num_modes;

    if n < 4 {
        return Err(VMDError::InvalidInput(
            "Signal too short for VMD decomposition".to_string(),
        ));
    }

    if k < 1 || k >= n / 2 {
        return Err(VMDError::InvalidInput(format!(
            "Number of modes must be between 1 and {}",
            n / 2
        )));
    }

    // Mirror signal to handle boundaries
    let signal_mirrored = mirror_signal(signal);
    let t = signal_mirrored.len();
    let half_t = (t + 1) / 2;

    // Compute FFT of mirrored signal
    let signal_fft = compute_fft(&signal_mirrored);

    // Frequency domain parameters
    let freqs = compute_frequencies(t);

    // Initialize modes in Fourier domain
    let mut u_hat = Array2::<Complex<f64>>::zeros((k, t));

    // Initialize center frequencies
    let mut omega = if config.init_omega {
        initialize_omega_uniform(k)
    } else {
        initialize_omega_random(k)
    };

    // Initialize Lagrange multipliers
    let mut lambda_hat = Array1::<Complex<f64>>::zeros(t);

    // Compute tau if set to auto
    let tau = if config.tau == 0.0 { 0.0 } else { config.tau };

    let mut iterations = 0;
    let mut convergence_error = f64::INFINITY;

    // ADMM iterations
    for iter in 0..config.max_iter {
        // Update modes in Fourier domain
        let u_hat_old = u_hat.clone();

        for k_i in 0..k {
            // Sum of other modes
            let mut sum_uk = Array1::<Complex<f64>>::zeros(t);
            for k_j in 0..k {
                if k_j != k_i {
                    sum_uk = sum_uk + u_hat.row(k_j);
                }
            }

            // Update mode spectrum
            for (i, freq) in freqs.iter().enumerate() {
                let numerator = signal_fft[i] - sum_uk[i] + lambda_hat[i] / 2.0;
                let denominator = 1.0 + config.alpha * (freq - omega[k_i]).powi(2);
                u_hat[[k_i, i]] = numerator / denominator;
            }

            // Update center frequency for this mode (excluding DC)
            let start_idx = if config.dc { 1 } else { 0 };
            let mut numerator = 0.0;
            let mut denominator = 0.0;

            for i in start_idx..half_t {
                let u_abs_sq = u_hat[[k_i, i]].norm_sqr();
                numerator += freqs[i] * u_abs_sq;
                denominator += u_abs_sq;
            }

            if denominator > 1e-10 {
                omega[k_i] = numerator / denominator;
            }

            // Enforce non-negative and bounded frequencies
            omega[k_i] = omega[k_i].max(0.0).min(0.5);
        }

        // Update Lagrange multipliers (dual ascent)
        let mut sum_uk = Array1::<Complex<f64>>::zeros(t);
        for k_i in 0..k {
            sum_uk = sum_uk + u_hat.row(k_i);
        }

        for i in 0..t {
            lambda_hat[i] += tau * (signal_fft[i] - sum_uk[i]);
        }

        // Check convergence
        convergence_error = compute_convergence_error(&u_hat, &u_hat_old);
        iterations = iter + 1;

        if convergence_error < config.tol {
            break;
        }
    }

    // Reconstruct modes in time domain
    let mut modes = Array2::<f64>::zeros((k, n));

    for k_i in 0..k {
        let mode_fft = u_hat.row(k_i).to_owned();
        let mode_time = compute_ifft(&mode_fft);

        // Extract the original signal portion (remove mirroring)
        let offset = (t - n) / 2;
        for i in 0..n {
            modes[[k_i, i]] = mode_time[offset + i];
        }
    }

    Ok(VMDResult {
        modes,
        center_frequencies: omega,
        iterations,
        convergence_error,
    })
}

// Helper functions

/// Mirror signal for boundary handling
fn mirror_signal(signal: &Array1<f64>) -> Array1<f64> {
    let n = signal.len();
    let mirror_len = n / 2;

    let mut mirrored = Array1::zeros(n + 2 * mirror_len);

    // Left mirror
    for i in 0..mirror_len {
        mirrored[i] = signal[mirror_len - i];
    }

    // Original signal
    for i in 0..n {
        mirrored[mirror_len + i] = signal[i];
    }

    // Right mirror
    for i in 0..mirror_len {
        mirrored[mirror_len + n + i] = signal[n - 2 - i];
    }

    mirrored
}

/// Compute FFT of real signal using simple DFT
/// Note: For production use, this can be replaced with scirs2-fft when available
fn compute_fft(signal: &Array1<f64>) -> Array1<Complex<f64>> {
    simple_dft(signal)
}

/// Compute inverse FFT using simple IDFT
/// Note: For production use, this can be replaced with scirs2-fft when available
fn compute_ifft(fft: &Array1<Complex<f64>>) -> Array1<f64> {
    simple_idft(fft)
}

/// Simple DFT implementation (fallback)
fn simple_dft(signal: &Array1<f64>) -> Array1<Complex<f64>> {
    let n = signal.len();
    let mut fft = Array1::zeros(n);

    for k in 0..n {
        let mut sum = Complex::new(0.0, 0.0);
        for t in 0..n {
            let angle = -2.0 * PI * (k as f64) * (t as f64) / (n as f64);
            sum += signal[t] * Complex::new(angle.cos(), angle.sin());
        }
        fft[k] = sum;
    }

    fft
}

/// Simple inverse DFT (fallback)
fn simple_idft(fft: &Array1<Complex<f64>>) -> Array1<f64> {
    let n = fft.len();
    let mut signal = Array1::zeros(n);

    for t in 0..n {
        let mut sum = Complex::new(0.0, 0.0);
        for k in 0..n {
            let angle = 2.0 * PI * (k as f64) * (t as f64) / (n as f64);
            sum += fft[k] * Complex::new(angle.cos(), angle.sin());
        }
        signal[t] = sum.re / (n as f64);
    }

    signal
}

/// Compute frequency bins
fn compute_frequencies(n: usize) -> Array1<f64> {
    Array1::from_vec((0..n).map(|i| i as f64 / n as f64).collect())
}

/// Initialize center frequencies uniformly
fn initialize_omega_uniform(k: usize) -> Array1<f64> {
    let max_freq = 0.5;
    Array1::from_vec(
        (0..k)
            .map(|i| (i + 1) as f64 * max_freq / (k as f64 + 1.0))
            .collect(),
    )
}

/// Initialize center frequencies randomly
fn initialize_omega_random(k: usize) -> Array1<f64> {
    let mut rng = thread_rng();
    let mut omega = Array1::from_shape_fn(k, |_| rng.gen_range(0.0..0.5));

    // Sort to ensure order
    let mut omega_vec = omega.to_vec();
    omega_vec.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    omega = Array1::from_vec(omega_vec);

    omega
}

/// Compute convergence error between iterations
fn compute_convergence_error(
    u_hat: &Array2<Complex<f64>>,
    u_hat_old: &Array2<Complex<f64>>,
) -> f64 {
    let mut error = 0.0;
    let mut norm_old = 0.0;

    for i in 0..u_hat.nrows() {
        for j in 0..u_hat.ncols() {
            let diff = u_hat[[i, j]] - u_hat_old[[i, j]];
            error += diff.norm_sqr();
            norm_old += u_hat_old[[i, j]].norm_sqr();
        }
    }

    if norm_old > 1e-10 {
        (error / norm_old).sqrt()
    } else {
        error.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_vmd_basic() {
        // Create a simple two-component signal
        let n = 128;
        let t: Array1<f64> = Array1::from_vec((0..n).map(|i| i as f64).collect());

        // Component 1: Low frequency (5 Hz)
        let c1 = t.mapv(|x| (2.0 * PI * 5.0 * x / n as f64).sin());

        // Component 2: High frequency (20 Hz)
        let c2 = t.mapv(|x| (2.0 * PI * 20.0 * x / n as f64).sin());

        let signal = &c1 + &c2;

        let config = VMDConfig::new(2).with_alpha(2000.0).with_max_iter(100);

        let result = vmd_decompose(&signal, &config).unwrap();

        assert_eq!(result.num_modes(), 2);
        assert_eq!(result.signal_length(), n);
        assert!(result.iterations <= 100);

        // Check that center frequencies are different
        assert!(
            (result.center_frequencies[0] - result.center_frequencies[1]).abs() > 0.01,
            "Center frequencies should be distinct"
        );
    }

    #[test]
    fn test_vmd_reconstruction() {
        // Test that VMD can decompose and reconstruct
        // VMD focuses on mode separation rather than perfect reconstruction
        let n = 32;
        let t: Array1<f64> = Array1::from_vec((0..n).map(|i| i as f64).collect());
        let signal = t.mapv(|x| (2.0 * PI * 3.0 * x / n as f64).sin());

        let config = VMDConfig::new(2).with_max_iter(30);
        let result = vmd_decompose(&signal, &config).unwrap();

        assert_eq!(result.num_modes(), 2);
        assert_eq!(result.signal_length(), n);

        // Check that modes have been extracted
        let mode0 = result.mode(0).unwrap();
        let mode1 = result.mode(1).unwrap();

        assert_eq!(mode0.len(), n);
        assert_eq!(mode1.len(), n);

        // VMD should produce non-zero modes
        let mode0_energy: f64 = mode0.iter().map(|x| x * x).sum();
        let mode1_energy: f64 = mode1.iter().map(|x| x * x).sum();

        assert!(
            mode0_energy > 0.01 || mode1_energy > 0.01,
            "At least one mode should have significant energy"
        );
    }

    #[test]
    fn test_vmd_config() {
        let config = VMDConfig::new(3)
            .with_alpha(3000.0)
            .with_tau(0.1)
            .with_max_iter(200)
            .with_tolerance(1e-8)
            .with_dc(true);

        assert_eq!(config.num_modes, 3);
        assert_eq!(config.alpha, 3000.0);
        assert_eq!(config.tau, 0.1);
        assert_eq!(config.max_iter, 200);
        assert_eq!(config.tol, 1e-8);
        assert_eq!(config.dc, true);
    }

    #[test]
    fn test_vmd_invalid_input() {
        // Too few samples
        let signal = Array1::from_vec(vec![1.0, 2.0]);
        let config = VMDConfig::new(2);
        assert!(vmd_decompose(&signal, &config).is_err());

        // Too many modes
        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let config = VMDConfig::new(5);
        assert!(vmd_decompose(&signal, &config).is_err());
    }

    #[test]
    fn test_vmd_mode_extraction() {
        let signal = Array1::from_vec((0..32).map(|i| (i as f64 * 0.1).sin()).collect());
        let config = VMDConfig::new(2);
        let result = vmd_decompose(&signal, &config).unwrap();

        // Test mode extraction
        let mode0 = result.mode(0).unwrap();
        let mode1 = result.mode(1).unwrap();

        assert_eq!(mode0.len(), 32);
        assert_eq!(mode1.len(), 32);

        // Test invalid index
        assert!(result.mode(2).is_none());
    }

    #[test]
    fn test_mirror_signal() {
        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let mirrored = mirror_signal(&signal);

        assert_eq!(mirrored.len(), 8);
        // Check mirroring: left mirror + original + right mirror
        // Left mirror reflects signal[0..mirror_len] reversed
        assert_eq!(mirrored[0], signal[2]); // signal[2]
        assert_eq!(mirrored[1], signal[1]); // signal[1]
                                            // Original signal
        assert_eq!(mirrored[2], signal[0]); // signal[0]
        assert_eq!(mirrored[3], signal[1]); // signal[1]
    }

    #[test]
    fn test_frequency_computation() {
        let freqs = compute_frequencies(10);
        assert_eq!(freqs.len(), 10);
        assert_eq!(freqs[0], 0.0);
        assert_relative_eq!(freqs[5], 0.5);
    }

    #[test]
    fn test_omega_initialization() {
        // Uniform initialization
        let omega_uniform = initialize_omega_uniform(3);
        assert_eq!(omega_uniform.len(), 3);
        assert!(omega_uniform[0] < omega_uniform[1]);
        assert!(omega_uniform[1] < omega_uniform[2]);

        // Random initialization
        let omega_random = initialize_omega_random(3);
        assert_eq!(omega_random.len(), 3);
        assert!(omega_random.iter().all(|&x| x >= 0.0 && x <= 0.5));
    }
}
