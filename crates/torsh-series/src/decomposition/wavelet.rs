//! Wavelet decomposition for time series analysis
//!
//! Wavelets provide a time-frequency representation of signals, making them ideal
//! for analyzing non-stationary time series with features at different scales.
//!
//! This module implements Discrete Wavelet Transform (DWT) using Haar and Daubechies-4
//! filter banks in pure Rust, and a Morlet-based Continuous Wavelet Transform (CWT).

#![allow(dead_code)]
use crate::TimeSeries;
use torsh_core::error::{Result, TorshError};
use torsh_tensor::Tensor;

// ============================================================
// Filter bank coefficients
// ============================================================

/// Low-pass (scaling) and high-pass (wavelet) filter coefficients for each wavelet type.
/// Following the convention: h0 = low-pass, h1 = high-pass (mirror filter).
fn filter_coefficients(wavelet: WaveletType) -> (&'static [f64], &'static [f64]) {
    match wavelet {
        WaveletType::Haar => (
            // Haar low-pass
            &[0.7071067811865476, 0.7071067811865476],
            // Haar high-pass
            &[0.7071067811865476, -0.7071067811865476],
        ),
        WaveletType::Daubechies4 => {
            // Daubechies 4 (db2) scaling/wavelet coefficients
            const DB4_LO: &[f64] = &[
                0.4829629131_445_341,
                0.8365163037_378_079,
                0.2241438680_420_134,
               -0.1294095225_512_604,
            ];
            const DB4_HI: &[f64] = &[
               -0.1294095225_512_604,
               -0.2241438680_420_134,
                0.8365163037_378_079,
               -0.4829629131_445_341,
            ];
            (DB4_LO, DB4_HI)
        }
        WaveletType::Symlet4 => {
            // Symlet 4 (sym4) coefficients
            const SYM4_LO: &[f64] = &[
               -0.075_765_714_789_273_32,
               -0.029_635_527_645_954_27,
                0.497_618_667_632_032_54,
                0.803_738_751_805_916_4,
                0.297_857_795_605_515_27,
               -0.099_219_543_576_847_21,
               -0.012_603_967_262_037_73,
                0.032_223_100_604_042_70,
            ];
            const SYM4_HI: &[f64] = &[
               -0.032_223_100_604_042_70,
               -0.012_603_967_262_037_73,
                0.099_219_543_576_847_21,
                0.297_857_795_605_515_27,
               -0.803_738_751_805_916_4,
                0.497_618_667_632_032_54,
                0.029_635_527_645_954_27,
               -0.075_765_714_789_273_32,
            ];
            (SYM4_LO, SYM4_HI)
        }
        WaveletType::Morlet => {
            // Morlet is continuous – use Haar for discrete approximation
            (
                &[0.7071067811865476, 0.7071067811865476],
                &[0.7071067811865476, -0.7071067811865476],
            )
        }
    }
}

/// Apply a discrete filter with periodic (circular) boundary extension and downsample by 2.
fn convolve_downsample(signal: &[f64], filter: &[f64]) -> Vec<f64> {
    let n = signal.len();
    let _flen = filter.len();
    let out_len = (n + 1) / 2;
    let mut out = vec![0.0f64; out_len];
    for k in 0..out_len {
        let start = 2 * k;
        let mut acc = 0.0;
        for (j, &fj) in filter.iter().enumerate() {
            let idx = (start + j) % n; // periodic boundary
            acc += signal[idx] * fj;
        }
        out[k] = acc;
    }
    out
}

/// Upsample by 2 (insert zeros) and apply synthesis filter (transpose convolution).
fn upsample_convolve(signal: &[f64], filter: &[f64], target_len: usize) -> Vec<f64> {
    let _flen = filter.len();
    let _n = signal.len();
    // Upsampled (insert zeros)
    let mut up = vec![0.0f64; target_len];
    for (k, &sk) in signal.iter().enumerate() {
        if 2 * k < target_len {
            up[2 * k] = sk;
        }
    }
    // Convolve with synthesis filter
    let mut out = vec![0.0f64; target_len];
    for i in 0..target_len {
        let mut acc = 0.0;
        for (j, &fj) in filter.iter().enumerate() {
            // Periodic boundary
            let idx = if i >= j { i - j } else { target_len - (j - i) % target_len };
            acc += up[idx % target_len] * fj;
        }
        out[i] = acc;
    }
    out
}

/// Morlet wavelet at scale `s` and translation `tau` evaluated on the sample at index `t`.
/// ψ(t) = π^{-1/4} * exp(iω₀t) * exp(-t²/2)  — real part only.
fn morlet_real(t_sample: f64, tau: f64, scale: f64, omega0: f64) -> f64 {
    let t = (t_sample - tau) / scale;
    let norm = std::f64::consts::PI.powf(-0.25) / scale.sqrt();
    norm * (omega0 * t).cos() * (-t * t / 2.0).exp()
}

/// Wavelet type enumeration (placeholder until scirs2-signal provides it)
#[derive(Debug, Clone, Copy)]
pub enum WaveletType {
    Haar,
    Daubechies4,
    Symlet4,
    Morlet,
}

/// Wavelet decomposition result
#[derive(Debug, Clone)]
pub struct WaveletDecomposition {
    /// Approximation coefficients (low frequency)
    pub approximation: Tensor,
    /// Detail coefficients (high frequency) for each level
    pub details: Vec<Tensor>,
    /// Wavelet family used
    pub wavelet_family: String,
    /// Decomposition level
    pub level: usize,
}

/// Continuous Wavelet Transform (CWT) result
#[derive(Debug, Clone)]
pub struct CWTResult {
    /// Wavelet coefficients [scale x time]
    pub coefficients: Tensor,
    /// Scales used
    pub scales: Vec<f64>,
    /// Frequencies corresponding to scales
    pub frequencies: Vec<f64>,
}

/// Wavelet-based time series decomposer
pub struct WaveletDecomposer {
    wavelet_type: WaveletType,
    level: Option<usize>,
    mode: String,
}

impl WaveletDecomposer {
    /// Create a new wavelet decomposer
    ///
    /// # Arguments
    /// * `wavelet_type` - Type of wavelet to use (e.g., Haar, Daubechies, Symlet)
    pub fn new(wavelet_type: WaveletType) -> Self {
        Self {
            wavelet_type,
            level: None,
            mode: "symmetric".to_string(),
        }
    }

    /// Set decomposition level
    pub fn with_level(mut self, level: usize) -> Self {
        self.level = Some(level);
        self
    }

    /// Set boundary extension mode
    pub fn with_mode(mut self, mode: &str) -> Self {
        self.mode = mode.to_string();
        self
    }

    /// Perform Discrete Wavelet Transform (DWT) decomposition.
    ///
    /// Uses the filter bank for the selected wavelet type.  Boundary conditions
    /// are handled with periodic extension.
    pub fn decompose(&self, series: &TimeSeries) -> Result<WaveletDecomposition> {
        let raw: Vec<f32> = series.values.to_vec().map_err(|e| {
            TorshError::InvalidArgument(format!("Failed to convert tensor to vec: {}", e))
        })?;
        let mut current: Vec<f64> = raw.iter().map(|&v| v as f64).collect();

        let level = self
            .level
            .unwrap_or_else(|| Self::max_decomposition_level(series.len()));
        let level = level.max(1);

        let (lo, hi) = filter_coefficients(self.wavelet_type);
        let mut details: Vec<Tensor> = Vec::with_capacity(level);
        let mut level_lens: Vec<usize> = Vec::with_capacity(level);

        for _lev in 0..level {
            if current.len() < 2 {
                // Signal too short to decompose further
                let t = Tensor::from_vec(vec![0.0f32], &[1])?;
                details.push(t);
                level_lens.push(current.len());
                break;
            }
            level_lens.push(current.len());
            let approx = convolve_downsample(&current, lo);
            let detail = convolve_downsample(&current, hi);
            let detail_len = detail.len();
            let detail_f32: Vec<f32> = detail.iter().map(|&v| v as f32).collect();
            details.push(Tensor::from_vec(detail_f32, &[detail_len])?);
            current = approx;
        }

        let approx_f32: Vec<f32> = current.iter().map(|&v| v as f32).collect();
        let approx_len = approx_f32.len();
        let approximation = Tensor::from_vec(approx_f32, &[approx_len])?;

        Ok(WaveletDecomposition {
            approximation,
            details,
            wavelet_family: format!("{:?}", self.wavelet_type),
            level,
        })
    }

    /// Reconstruct time series from wavelet decomposition.
    ///
    /// Uses the inverse filter bank (transpose convolution) to reconstruct the signal
    /// from its approximation and detail coefficients.
    pub fn reconstruct(&self, decomposition: &WaveletDecomposition) -> Result<TimeSeries> {
        let (lo, hi) = filter_coefficients(self.wavelet_type);

        // Start from the coarsest approximation
        let mut current: Vec<f64> = decomposition
            .approximation
            .to_vec()
            .map_err(|e| TorshError::InvalidArgument(format!("tensor to_vec: {}", e)))?
            .iter()
            .map(|&v| v as f64)
            .collect();

        // Traverse detail levels from coarsest to finest
        for detail_tensor in decomposition.details.iter().rev() {
            let detail: Vec<f64> = detail_tensor
                .to_vec()
                .map_err(|e| TorshError::InvalidArgument(format!("tensor to_vec: {}", e)))?
                .iter()
                .map(|&v| v as f64)
                .collect();
            let target_len = detail.len() * 2;
            let rec_lo = upsample_convolve(&current, lo, target_len);
            let rec_hi = upsample_convolve(&detail, hi, target_len);
            current = rec_lo
                .iter()
                .zip(rec_hi.iter())
                .map(|(&a, &b)| a + b)
                .collect();
        }

        let out: Vec<f32> = current.iter().map(|&v| v as f32).collect();
        let out_len = out.len();
        let tensor = Tensor::from_vec(out, &[out_len])?;
        Ok(TimeSeries::new(tensor))
    }

    /// Calculate maximum useful decomposition level
    fn max_decomposition_level(n: usize) -> usize {
        // Maximum level is floor(log2(n))
        ((n as f64).log2().floor() as usize).max(1)
    }

    /// Perform single-level DWT using the filter bank for the selected wavelet type.
    pub fn single_level_dwt(&self, series: &TimeSeries) -> Result<(Tensor, Tensor)> {
        let raw: Vec<f32> = series.values.to_vec()?;
        let data: Vec<f64> = raw.iter().map(|&v| v as f64).collect();

        let (lo, hi) = filter_coefficients(self.wavelet_type);
        let approx = convolve_downsample(&data, lo);
        let detail = convolve_downsample(&data, hi);

        let approx_len = approx.len();
        let detail_len = detail.len();
        let approx_f32: Vec<f32> = approx.iter().map(|&v| v as f32).collect();
        let detail_f32: Vec<f32> = detail.iter().map(|&v| v as f32).collect();

        Ok((
            Tensor::from_vec(approx_f32, &[approx_len])?,
            Tensor::from_vec(detail_f32, &[detail_len])?,
        ))
    }

    /// Perform single-level inverse DWT.
    pub fn single_level_idwt(&self, approx: &Tensor, detail: &Tensor) -> Result<TimeSeries> {
        let approx_data: Vec<f64> = approx.to_vec()?.iter().map(|&v: &f32| v as f64).collect();
        let detail_data: Vec<f64> = detail.to_vec()?.iter().map(|&v: &f32| v as f64).collect();

        let (lo, hi) = filter_coefficients(self.wavelet_type);
        let target_len = detail_data.len() * 2;

        let rec_lo = upsample_convolve(&approx_data, lo, target_len);
        let rec_hi = upsample_convolve(&detail_data, hi, target_len);

        let recon: Vec<f32> = rec_lo.iter()
            .zip(rec_hi.iter())
            .map(|(&a, &b)| (a + b) as f32)
            .collect();
        let recon_len = recon.len();
        let tensor = Tensor::from_vec(recon, &[recon_len])?;
        Ok(TimeSeries::new(tensor))
    }
}

/// Continuous Wavelet Transform (CWT) analyzer
pub struct CWTAnalyzer {
    wavelet_type: WaveletType,
    scales: Option<Vec<f64>>,
    sampling_period: f64,
}

impl CWTAnalyzer {
    /// Create a new CWT analyzer
    pub fn new(wavelet_type: WaveletType) -> Self {
        Self {
            wavelet_type,
            scales: None,
            sampling_period: 1.0,
        }
    }

    /// Set scales for CWT
    pub fn with_scales(mut self, scales: Vec<f64>) -> Self {
        self.scales = Some(scales);
        self
    }

    /// Set sampling period
    pub fn with_sampling_period(mut self, period: f64) -> Self {
        self.sampling_period = period;
        self
    }

    /// Perform Continuous Wavelet Transform.
    ///
    /// Uses the real part of the Morlet wavelet ψ(t) = π^{-1/4} cos(ω₀t) e^{-t²/2}
    /// with ω₀ = 6 (the standard central frequency for time-frequency localisation).
    /// For Haar/Daubechies wavelet types the DWT filter bank is used instead.
    pub fn analyze(&self, series: &TimeSeries) -> Result<CWTResult> {
        let raw: Vec<f32> = series.values.to_vec()?;
        let data: Vec<f64> = raw.iter().map(|&v| v as f64).collect();
        let n_time = data.len();

        let scales = self
            .scales
            .clone()
            .unwrap_or_else(|| Self::generate_scales(n_time, 1.0, 64.0, 32));

        let n_scales = scales.len();
        let omega0 = 6.0f64; // Standard Morlet central frequency

        // Compute CWT by direct convolution with real Morlet wavelet at each scale
        let mut coef_data = vec![0.0f32; n_scales * n_time];
        for (si, &scale) in scales.iter().enumerate() {
            for t in 0..n_time {
                let mut sum = 0.0f64;
                for tau_idx in 0..n_time {
                    let morlet = morlet_real(tau_idx as f64, t as f64, scale, omega0);
                    sum += data[tau_idx] * morlet;
                }
                coef_data[si * n_time + t] = sum as f32;
            }
        }

        let coefficients = Tensor::from_vec(coef_data, &[n_scales, n_time])?;
        let frequencies = self.scales_to_frequencies(&scales);

        Ok(CWTResult {
            coefficients,
            scales,
            frequencies,
        })
    }

    /// Generate logarithmically spaced scales
    fn generate_scales(_n: usize, min_scale: f64, max_scale: f64, n_scales: usize) -> Vec<f64> {
        let log_min = min_scale.ln();
        let log_max = max_scale.ln();
        let step = (log_max - log_min) / (n_scales - 1) as f64;

        (0..n_scales)
            .map(|i| (log_min + i as f64 * step).exp())
            .collect()
    }

    /// Convert scales to frequencies
    fn scales_to_frequencies(&self, scales: &[f64]) -> Vec<f64> {
        // For Morlet wavelet, center frequency is typically ~1.0
        // Frequency = center_freq / (scale * sampling_period)
        let center_freq = 1.0;
        scales
            .iter()
            .map(|&scale| center_freq / (scale * self.sampling_period))
            .collect()
    }
}

/// Wavelet packet decomposition for more flexible time-frequency analysis
pub struct WaveletPacketDecomposer {
    wavelet_type: WaveletType,
    level: usize,
}

impl WaveletPacketDecomposer {
    /// Create a new wavelet packet decomposer
    pub fn new(wavelet_type: WaveletType, level: usize) -> Self {
        Self {
            wavelet_type,
            level,
        }
    }

    /// Decompose into wavelet packet tree.
    ///
    /// Each node in the tree is identified by a binary string path (e.g. "ad" means
    /// approximation of the first level, then detail of the second level).
    /// An empty path "" denotes the root (original signal).
    pub fn decompose(&self, series: &TimeSeries) -> Result<WaveletPacketTree> {
        let raw: Vec<f32> = series.values.to_vec()?;
        let root: Vec<f64> = raw.iter().map(|&v| v as f64).collect();

        let (lo, hi) = filter_coefficients(self.wavelet_type);
        let mut nodes: std::collections::HashMap<String, Tensor> =
            std::collections::HashMap::new();

        // BFS expansion of the wavelet packet tree up to `self.level` levels
        // Queue: (path, signal)
        let mut queue: Vec<(String, Vec<f64>)> = vec![("".to_string(), root)];

        while let Some((path, signal)) = queue.pop() {
            let n = signal.len();
            let path_level = path.len();

            // Store this node
            let f32_sig: Vec<f32> = signal.iter().map(|&v| v as f32).collect();
            let t = Tensor::from_vec(f32_sig, &[n])?;
            nodes.insert(path.clone(), t);

            if path_level < self.level && n >= 2 {
                let approx = convolve_downsample(&signal, lo);
                let detail = convolve_downsample(&signal, hi);
                queue.push((format!("{}a", path), approx));
                queue.push((format!("{}d", path), detail));
            }
        }

        Ok(WaveletPacketTree {
            nodes,
            level: self.level,
            wavelet_family: format!("{:?}", self.wavelet_type),
        })
    }
}

/// Wavelet packet tree structure
#[derive(Debug, Clone)]
pub struct WaveletPacketTree {
    /// Nodes indexed by path (e.g., "a" for approximation, "d" for detail)
    pub nodes: std::collections::HashMap<String, Tensor>,
    /// Decomposition level
    pub level: usize,
    /// Wavelet family
    pub wavelet_family: String,
}

impl WaveletPacketTree {
    /// Get node at specific path
    pub fn get_node(&self, path: &str) -> Option<&Tensor> {
        self.nodes.get(path)
    }

    /// Get all leaf nodes at the deepest level
    pub fn leaf_nodes(&self) -> Vec<&Tensor> {
        self.nodes
            .iter()
            .filter(|(path, _)| path.len() == self.level)
            .map(|(_, tensor)| tensor)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::Tensor;

    fn create_test_series() -> TimeSeries {
        // Create a simple time series with multiple frequencies
        let mut data = Vec::with_capacity(128);
        for i in 0..128 {
            let t = i as f32 * 0.1;
            // Low frequency + high frequency components
            let val = (t).sin() + 0.5 * (5.0 * t).sin();
            data.push(val);
        }
        let tensor = Tensor::from_vec(data, &[128]).expect("Tensor should succeed");
        TimeSeries::new(tensor)
    }

    #[test]
    fn test_wavelet_decomposer_creation() {
        let decomposer = WaveletDecomposer::new(WaveletType::Haar);
        assert_eq!(decomposer.mode, "symmetric");
    }

    #[test]
    fn test_wavelet_decomposition() {
        let series = create_test_series();
        let decomposer = WaveletDecomposer::new(WaveletType::Haar).with_level(3);
        let decomposition = decomposer
            .decompose(&series)
            .expect("decomposition should succeed");

        assert_eq!(decomposition.level, 3);
        assert_eq!(decomposition.details.len(), 3);
        assert!(decomposition.approximation.shape().dims()[0] > 0);
    }

    #[test]
    fn test_wavelet_reconstruction() {
        let series = create_test_series();
        let decomposer = WaveletDecomposer::new(WaveletType::Haar).with_level(2);
        let decomposition = decomposer
            .decompose(&series)
            .expect("decomposition should succeed");
        let reconstructed = decomposer
            .reconstruct(&decomposition)
            .expect("reconstruction should succeed");

        // Reconstructed should have similar length to original
        assert!(reconstructed.len() >= series.len() - 10); // Allow small difference due to padding
    }

    #[test]
    fn test_single_level_dwt() {
        let series = create_test_series();
        let decomposer = WaveletDecomposer::new(WaveletType::Haar);
        let (approx, detail) = decomposer
            .single_level_dwt(&series)
            .expect("single-level DWT should succeed");

        assert!(approx.shape().dims()[0] > 0);
        assert!(detail.shape().dims()[0] > 0);
    }

    #[test]
    fn test_cwt_analyzer() {
        let series = create_test_series();
        let analyzer = CWTAnalyzer::new(WaveletType::Morlet).with_sampling_period(0.1);
        let result = analyzer.analyze(&series).expect("analysis should succeed");

        assert!(result.coefficients.shape().dims()[0] > 0); // scales
        assert_eq!(result.coefficients.shape().dims()[1], series.len()); // time
        assert!(!result.scales.is_empty());
        assert!(!result.frequencies.is_empty());
    }

    #[test]
    fn test_max_decomposition_level() {
        assert_eq!(WaveletDecomposer::max_decomposition_level(128), 7);
        assert_eq!(WaveletDecomposer::max_decomposition_level(256), 8);
        assert_eq!(WaveletDecomposer::max_decomposition_level(64), 6);
    }

    #[test]
    fn test_wavelet_packet_decomposer() {
        let series = create_test_series();
        let decomposer = WaveletPacketDecomposer::new(WaveletType::Daubechies4, 2);
        let tree = decomposer
            .decompose(&series)
            .expect("decomposition should succeed");

        assert_eq!(tree.level, 2);
        assert!(!tree.nodes.is_empty());
    }
}
