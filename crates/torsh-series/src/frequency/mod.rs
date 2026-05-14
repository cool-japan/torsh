//! Frequency domain analysis for time series
//!
//! This module provides comprehensive frequency domain analysis tools:
//! - Fast Fourier Transform (FFT) and inverse FFT
//! - Power Spectral Density (PSD) estimation
//! - Periodogram analysis
//! - Spectral density estimation (Welch's method, multi-taper)
//! - Coherence analysis for multivariate series
//!
//! NOTE: This module provides the structure for frequency analysis.
//! Full implementation will use scirs2-signal when FFT APIs are available.

use crate::TimeSeries;
use torsh_core::error::{Result, TorshError};
use torsh_tensor::Tensor;

/// FFT result containing complex frequency components
#[derive(Debug, Clone)]
pub struct FFTResult {
    /// Real parts of FFT coefficients
    pub real: Vec<f64>,
    /// Imaginary parts of FFT coefficients
    pub imag: Vec<f64>,
    /// Frequencies corresponding to each coefficient
    pub frequencies: Vec<f64>,
    /// Sampling rate used
    pub sampling_rate: f64,
}

impl FFTResult {
    /// Get magnitude spectrum
    pub fn magnitude(&self) -> Vec<f64> {
        self.real
            .iter()
            .zip(self.imag.iter())
            .map(|(r, i)| (r * r + i * i).sqrt())
            .collect()
    }

    /// Get phase spectrum
    pub fn phase(&self) -> Vec<f64> {
        self.real
            .iter()
            .zip(self.imag.iter())
            .map(|(r, i)| i.atan2(*r))
            .collect()
    }

    /// Get power spectrum (magnitude squared)
    pub fn power(&self) -> Vec<f64> {
        self.real
            .iter()
            .zip(self.imag.iter())
            .map(|(r, i)| r * r + i * i)
            .collect()
    }
}

/// Power Spectral Density estimation result
#[derive(Debug, Clone)]
pub struct PSDResult {
    /// Power spectral density values
    pub psd: Vec<f64>,
    /// Frequencies
    pub frequencies: Vec<f64>,
    /// Method used for estimation
    pub method: String,
}

/// Periodogram analysis result
#[derive(Debug, Clone)]
pub struct Periodogram {
    /// Power at each frequency
    pub power: Vec<f64>,
    /// Frequencies
    pub frequencies: Vec<f64>,
    /// Significant peaks (frequency, power)
    pub peaks: Vec<(f64, f64)>,
}

/// FFT analyzer for frequency domain analysis
pub struct FFTAnalyzer {
    sampling_rate: f64,
    window: Option<WindowType>,
}

#[derive(Debug, Clone, Copy)]
pub enum WindowType {
    Hann,
    Hamming,
    Blackman,
    Rectangular,
}

impl FFTAnalyzer {
    /// Create a new FFT analyzer
    pub fn new(sampling_rate: f64) -> Self {
        Self {
            sampling_rate,
            window: None,
        }
    }

    /// Set window function for FFT
    pub fn with_window(mut self, window: WindowType) -> Self {
        self.window = Some(window);
        self
    }

    /// Perform FFT on time series
    pub fn fft(&self, series: &TimeSeries) -> Result<FFTResult> {
        let data = series.values.to_vec()?;
        let n = data.len();

        // Apply window if specified
        let windowed_data = if let Some(window) = self.window {
            self.apply_window(&data, window)
        } else {
            data.clone()
        };

        // TODO: Use scirs2-signal FFT when available
        // For now, implement simplified DFT (slow but functional)
        let (real, imag) = self.naive_dft(&windowed_data);

        // Generate frequency bins
        let frequencies: Vec<f64> = (0..n)
            .map(|k| k as f64 * self.sampling_rate / n as f64)
            .collect();

        Ok(FFTResult {
            real,
            imag,
            frequencies,
            sampling_rate: self.sampling_rate,
        })
    }

    /// Perform inverse FFT
    pub fn ifft(&self, fft_result: &FFTResult) -> Result<TimeSeries> {
        // TODO: Use scirs2-signal IFFT when available
        let n = fft_result.real.len();
        let mut data = vec![0.0f32; n];

        // Simplified inverse DFT
        for t in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                let angle = 2.0 * std::f64::consts::PI * (k * t) as f64 / n as f64;
                sum += fft_result.real[k] * angle.cos() - fft_result.imag[k] * angle.sin();
            }
            data[t] = (sum / n as f64) as f32;
        }

        let tensor = Tensor::from_vec(data, &[n])?;
        Ok(TimeSeries::new(tensor))
    }

    /// Apply window function
    fn apply_window(&self, data: &[f32], window: WindowType) -> Vec<f32> {
        let n = data.len();
        data.iter()
            .enumerate()
            .map(|(i, &x)| x * self.window_coefficient(i, n, window) as f32)
            .collect()
    }

    /// Get window coefficient
    fn window_coefficient(&self, i: usize, n: usize, window: WindowType) -> f64 {
        use std::f64::consts::PI;
        match window {
            WindowType::Hann => 0.5 * (1.0 - (2.0 * PI * i as f64 / (n - 1) as f64).cos()),
            WindowType::Hamming => 0.54 - 0.46 * (2.0 * PI * i as f64 / (n - 1) as f64).cos(),
            WindowType::Blackman => {
                0.42 - 0.5 * (2.0 * PI * i as f64 / (n - 1) as f64).cos()
                    + 0.08 * (4.0 * PI * i as f64 / (n - 1) as f64).cos()
            }
            WindowType::Rectangular => 1.0,
        }
    }

    /// Naive DFT implementation (placeholder for scirs2-signal FFT)
    fn naive_dft(&self, data: &[f32]) -> (Vec<f64>, Vec<f64>) {
        let n = data.len();
        let mut real = vec![0.0; n];
        let mut imag = vec![0.0; n];

        for k in 0..n {
            for t in 0..n {
                let angle = -2.0 * std::f64::consts::PI * (k * t) as f64 / n as f64;
                real[k] += data[t] as f64 * angle.cos();
                imag[k] += data[t] as f64 * angle.sin();
            }
        }

        (real, imag)
    }
}

/// Power Spectral Density estimator
pub struct PSDEstimator {
    method: PSDMethod,
    sampling_rate: f64,
}

#[derive(Debug, Clone, Copy)]
pub enum PSDMethod {
    Periodogram,
    Welch,
    MultitaperThomson,
}

impl PSDEstimator {
    /// Create a new PSD estimator
    pub fn new(method: PSDMethod, sampling_rate: f64) -> Self {
        Self {
            method,
            sampling_rate,
        }
    }

    /// Estimate power spectral density
    pub fn estimate(&self, series: &TimeSeries) -> Result<PSDResult> {
        match self.method {
            PSDMethod::Periodogram => self.periodogram_psd(series),
            PSDMethod::Welch => self.welch_psd(series),
            PSDMethod::MultitaperThomson => self.multitaper_psd(series),
        }
    }

    /// Periodogram-based PSD estimation
    fn periodogram_psd(&self, series: &TimeSeries) -> Result<PSDResult> {
        let fft = FFTAnalyzer::new(self.sampling_rate).fft(series)?;
        let power = fft.power();
        let n = power.len();

        // Normalize by length
        let psd: Vec<f64> = power.iter().map(|&p| p / n as f64).collect();

        Ok(PSDResult {
            psd,
            frequencies: fft.frequencies,
            method: "Periodogram".to_string(),
        })
    }

    /// Welch's method for PSD estimation using 50% overlapping Hann-windowed segments.
    ///
    /// The algorithm:
    /// 1. Split the signal into overlapping segments of length `nperseg`.
    /// 2. Apply a Hann window to each segment.
    /// 3. Compute the periodogram of each segment.
    /// 4. Average all periodograms to reduce variance.
    fn welch_psd(&self, series: &TimeSeries) -> Result<PSDResult> {
        let data = series.values.to_vec()?;
        let n = data.len();

        // Choose segment length: roughly sqrt(n), clamped to [8, 256]
        let nperseg = ((n as f64).sqrt() as usize).max(8).min(256);
        // 50% overlap
        let step = nperseg / 2;

        if n < nperseg {
            // Fall back to periodogram when the signal is too short
            return self.periodogram_psd(series);
        }

        // Hann window coefficients
        let window: Vec<f64> = (0..nperseg)
            .map(|i| {
                0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (nperseg - 1) as f64).cos())
            })
            .collect();

        // Window power normalisation (sum of squared window values)
        let win_power: f64 = window.iter().map(|&w| w * w).sum();

        let mut psd_accum = vec![0.0f64; nperseg];
        let mut num_segments = 0usize;

        let mut start = 0;
        while start + nperseg <= n {
            let segment: Vec<f64> = data[start..start + nperseg]
                .iter()
                .enumerate()
                .map(|(i, &x)| x as f64 * window[i])
                .collect();

            // DFT of this segment
            let (re, im) = self.segment_dft(&segment, nperseg);

            // Accumulate power spectrum
            for k in 0..nperseg {
                psd_accum[k] += (re[k] * re[k] + im[k] * im[k])
                    / (self.sampling_rate * win_power);
            }

            num_segments += 1;
            start += step;
        }

        // Average over segments
        let divisor = num_segments.max(1) as f64;
        let psd: Vec<f64> = psd_accum.iter().map(|&s| s / divisor).collect();

        let frequencies: Vec<f64> = (0..nperseg)
            .map(|k| k as f64 * self.sampling_rate / nperseg as f64)
            .collect();

        Ok(PSDResult {
            psd,
            frequencies,
            method: "Welch".to_string(),
        })
    }

    /// Compute DFT of a single real-valued segment (length `n`).
    fn segment_dft(&self, segment: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
        let mut re = vec![0.0f64; n];
        let mut im = vec![0.0f64; n];

        for k in 0..n {
            for t in 0..n {
                let angle = -2.0 * std::f64::consts::PI * (k * t) as f64 / n as f64;
                re[k] += segment[t] * angle.cos();
                im[k] += segment[t] * angle.sin();
            }
        }

        (re, im)
    }

    /// Thomson's multitaper method (falls back to Welch for now).
    ///
    /// A full multitaper implementation requires discrete prolate spheroidal
    /// sequences (DPSS tapers), which depend on a Laplacian eigensolver not
    /// yet available in the workspace.  Welch's method is a reasonable
    /// substitute that already reduces variance relative to the plain
    /// periodogram.
    fn multitaper_psd(&self, series: &TimeSeries) -> Result<PSDResult> {
        // Use Welch as an approximation until DPSS tapers are available
        self.welch_psd(series)
    }
}

/// Periodogram analyzer
pub struct PeriodogramAnalyzer {
    sampling_rate: f64,
    peak_threshold: f64,
}

impl PeriodogramAnalyzer {
    /// Create a new periodogram analyzer
    pub fn new(sampling_rate: f64) -> Self {
        Self {
            sampling_rate,
            peak_threshold: 0.1,
        }
    }

    /// Set threshold for peak detection (relative to max power)
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.peak_threshold = threshold;
        self
    }

    /// Compute periodogram
    pub fn analyze(&self, series: &TimeSeries) -> Result<Periodogram> {
        let fft = FFTAnalyzer::new(self.sampling_rate).fft(series)?;
        let power = fft.power();

        // Find peaks
        let max_power = power.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let threshold = max_power * self.peak_threshold;

        let mut peaks = Vec::new();
        for i in 1..power.len() - 1 {
            if power[i] > power[i - 1] && power[i] > power[i + 1] && power[i] > threshold {
                peaks.push((fft.frequencies[i], power[i]));
            }
        }

        Ok(Periodogram {
            power,
            frequencies: fft.frequencies,
            peaks,
        })
    }

    /// Find dominant frequency
    pub fn dominant_frequency(&self, series: &TimeSeries) -> Result<f64> {
        let periodogram = self.analyze(series)?;

        if periodogram.peaks.is_empty() {
            // Return frequency with max power
            let max_idx = periodogram
                .power
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            Ok(periodogram.frequencies[max_idx])
        } else {
            // Return peak with highest power
            Ok(periodogram
                .peaks
                .iter()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(freq, _)| *freq)
                .unwrap_or(0.0))
        }
    }
}

/// Spectral coherence analyzer for multivariate time series
pub struct CoherenceAnalyzer {
    sampling_rate: f64,
}

impl CoherenceAnalyzer {
    /// Create a new coherence analyzer
    pub fn new(sampling_rate: f64) -> Self {
        Self { sampling_rate }
    }

    /// Compute coherence between two time series
    pub fn coherence(&self, x: &TimeSeries, y: &TimeSeries) -> Result<Vec<f64>> {
        if x.len() != y.len() {
            return Err(TorshError::InvalidArgument(
                "Time series must have equal length".to_string(),
            ));
        }

        // TODO: Use scirs2-signal cross-spectral density when available
        let fft_x = FFTAnalyzer::new(self.sampling_rate).fft(x)?;
        let fft_y = FFTAnalyzer::new(self.sampling_rate).fft(y)?;

        let n = fft_x.real.len();
        let mut coherence = vec![0.0; n];

        for i in 0..n {
            let pxx = fft_x.real[i] * fft_x.real[i] + fft_x.imag[i] * fft_x.imag[i];
            let pyy = fft_y.real[i] * fft_y.real[i] + fft_y.imag[i] * fft_y.imag[i];
            let pxy = (fft_x.real[i] * fft_y.real[i] + fft_x.imag[i] * fft_y.imag[i]).abs();

            coherence[i] = if pxx > 1e-10 && pyy > 1e-10 {
                (pxy * pxy) / (pxx * pyy)
            } else {
                0.0
            };
        }

        Ok(coherence)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::Tensor;

    fn create_test_series() -> TimeSeries {
        // Create a simple sinusoidal signal
        let mut data = Vec::with_capacity(128);
        for i in 0..128 {
            let t = i as f32 * 0.1;
            data.push((2.0 * std::f32::consts::PI * t).sin());
        }
        let tensor = Tensor::from_vec(data, &[128]).expect("Tensor should succeed");
        TimeSeries::new(tensor)
    }

    #[test]
    fn test_fft_analyzer() {
        let series = create_test_series();
        let analyzer = FFTAnalyzer::new(10.0);
        let result = analyzer
            .fft(&series)
            .expect("FFT computation should succeed");

        assert_eq!(result.real.len(), 128);
        assert_eq!(result.imag.len(), 128);
        assert_eq!(result.frequencies.len(), 128);
    }

    #[test]
    fn test_fft_with_window() {
        let series = create_test_series();
        let analyzer = FFTAnalyzer::new(10.0).with_window(WindowType::Hann);
        let result = analyzer
            .fft(&series)
            .expect("FFT computation should succeed");

        assert_eq!(result.real.len(), 128);
    }

    #[test]
    fn test_fft_magnitude() {
        let series = create_test_series();
        let analyzer = FFTAnalyzer::new(10.0);
        let result = analyzer
            .fft(&series)
            .expect("FFT computation should succeed");
        let magnitude = result.magnitude();

        assert_eq!(magnitude.len(), 128);
        assert!(magnitude.iter().all(|&m| m >= 0.0));
    }

    #[test]
    fn test_ifft() {
        let series = create_test_series();
        let analyzer = FFTAnalyzer::new(10.0);
        let fft_result = analyzer
            .fft(&series)
            .expect("FFT computation should succeed");
        let reconstructed = analyzer
            .ifft(&fft_result)
            .expect("IFFT computation should succeed");

        assert_eq!(reconstructed.len(), series.len());
    }

    #[test]
    fn test_psd_periodogram() {
        let series = create_test_series();
        let estimator = PSDEstimator::new(PSDMethod::Periodogram, 10.0);
        let result = estimator
            .estimate(&series)
            .expect("estimation should succeed");

        assert_eq!(result.psd.len(), 128);
        assert_eq!(result.frequencies.len(), 128);
        assert_eq!(result.method, "Periodogram");
    }

    #[test]
    fn test_periodogram_analyzer() {
        let series = create_test_series();
        let analyzer = PeriodogramAnalyzer::new(10.0);
        let periodogram = analyzer.analyze(&series).expect("analysis should succeed");

        assert_eq!(periodogram.power.len(), 128);
        assert_eq!(periodogram.frequencies.len(), 128);
    }

    #[test]
    fn test_dominant_frequency() {
        let series = create_test_series();
        let analyzer = PeriodogramAnalyzer::new(10.0);
        let freq = analyzer
            .dominant_frequency(&series)
            .expect("dominant frequency estimation should succeed");

        assert!(freq >= 0.0);
    }

    #[test]
    fn test_coherence_analyzer() {
        let series1 = create_test_series();
        let series2 = create_test_series();
        let analyzer = CoherenceAnalyzer::new(10.0);
        let coherence = analyzer
            .coherence(&series1, &series2)
            .expect("coherence computation should succeed");

        assert_eq!(coherence.len(), 128);
        assert!(coherence.iter().all(|&c| c >= 0.0 && c <= 1.0));
    }

    #[test]
    fn test_window_types() {
        let series = create_test_series();

        for window in [
            WindowType::Hann,
            WindowType::Hamming,
            WindowType::Blackman,
            WindowType::Rectangular,
        ] {
            let analyzer = FFTAnalyzer::new(10.0).with_window(window);
            let result = analyzer
                .fft(&series)
                .expect("FFT computation should succeed");
            assert_eq!(result.real.len(), 128);
        }
    }

    /// Welch's method should produce non-negative PSD values and at least one
    /// non-zero bin for a non-trivial sinusoidal signal.
    #[test]
    fn test_psd_welch() {
        let series = create_test_series();
        let estimator = PSDEstimator::new(PSDMethod::Welch, 10.0);
        let result = estimator.estimate(&series).expect("Welch PSD should succeed");

        assert!(!result.psd.is_empty(), "PSD should have entries");
        assert!(!result.frequencies.is_empty(), "Frequencies should have entries");
        assert_eq!(result.method, "Welch");
        assert!(result.psd.iter().all(|&v| v >= 0.0), "PSD values must be non-negative");
        // The signal is sinusoidal, so peak power should be positive
        let max_power = result.psd.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(max_power > 0.0, "Welch PSD should show non-zero power for a sinusoid");
    }

    /// Welch PSD on a very short series (< nperseg minimum) should fall back
    /// gracefully to the periodogram path.
    #[test]
    fn test_psd_welch_short_series() {
        let data = vec![1.0f32, 2.0, 1.0, 0.0, -1.0, -2.0];
        let tensor = Tensor::from_vec(data, &[6]).expect("Tensor should succeed");
        let series = TimeSeries::new(tensor);
        let estimator = PSDEstimator::new(PSDMethod::Welch, 1.0);
        let result = estimator.estimate(&series).expect("Welch PSD on short series should succeed");
        assert!(!result.psd.is_empty());
    }

    /// Multitaper method delegates to Welch and should produce valid output.
    #[test]
    fn test_psd_multitaper() {
        let series = create_test_series();
        let estimator = PSDEstimator::new(PSDMethod::MultitaperThomson, 10.0);
        let result = estimator.estimate(&series).expect("Multitaper PSD should succeed");
        assert!(!result.psd.is_empty());
        assert!(result.psd.iter().all(|&v| v >= 0.0));
    }
}
