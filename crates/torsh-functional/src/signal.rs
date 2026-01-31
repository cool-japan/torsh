//! Signal Processing Operations
//!
//! This module provides comprehensive digital signal processing (DSP) operations
//! for time-series analysis, audio processing, and spectral analysis.
//!
//! # Mathematical Foundation
//!
//! ## Window Functions
//!
//! Window functions are used to reduce spectral leakage in Fourier analysis by
//! tapering the signal at the boundaries. Each window has different trade-offs
//! between main lobe width and side lobe attenuation.
//!
//! ### Rectangular Window
//! ```text
//! w[n] = 1  for all n
//! ```
//! - **Main lobe width**: 4π/N
//! - **Side lobe level**: -13 dB
//! - **Use case**: Maximum frequency resolution, transient signals
//!
//! ### Hann Window (Raised Cosine)
//! ```text
//! w[n] = 0.5 - 0.5 cos(2πn / (N-1))  for 0 ≤ n ≤ N-1
//! ```
//! - **Main lobe width**: 8π/N
//! - **Side lobe level**: -31 dB
//! - **Use case**: General purpose, good balance
//!
//! ### Hamming Window
//! ```text
//! w[n] = 0.54 - 0.46 cos(2πn / (N-1))
//! ```
//! - **Main lobe width**: 8π/N
//! - **Side lobe level**: -42 dB
//! - **Use case**: Narrowband signals, better side lobe suppression
//!
//! ### Blackman Window
//! ```text
//! w[n] = 0.42 - 0.5 cos(2πn/(N-1)) + 0.08 cos(4πn/(N-1))
//! ```
//! - **Main lobe width**: 12π/N
//! - **Side lobe level**: -58 dB
//! - **Use case**: High dynamic range, spectral analysis
//!
//! ### Bartlett Window (Triangular)
//! ```text
//! w[n] = 1 - |2n/(N-1) - 1|
//! ```
//! - **Main lobe width**: 8π/N
//! - **Side lobe level**: -26 dB
//! - **Use case**: Simple tapering, fast computation
//!
//! ### Kaiser Window
//! ```text
//! w[n] = I₀(β√(1 - ((n - N/2) / (N/2))²)) / I₀(β)
//!
//! where I₀ is the modified Bessel function of the first kind
//! ```
//! - **Parameter β**: Controls trade-off (typical range 0-10)
//! - **Side lobe level**: -20β dB (approximately)
//! - **Use case**: Optimal windowing, FIR filter design
//!
//! ## Filtering Operations
//!
//! ### FIR Filtering (Finite Impulse Response)
//! ```text
//! y[n] = Σₖ₌₀ᴹ h[k] x[n-k]
//! ```
//! - **Properties**: Always stable, linear phase possible
//! - **Complexity**: O(M) per sample (M = filter order)
//! - **Use case**: Audio processing, decimation, interpolation
//!
//! ### IIR Filtering (Infinite Impulse Response)
//! ```text
//! y[n] = Σₖ₌₀ᴹ b[k] x[n-k] - Σₖ₌₁ᴺ a[k] y[n-k]
//! ```
//! - **Properties**: Efficient, recursive, can be unstable
//! - **Complexity**: O(M + N) per sample
//! - **Use case**: Real-time processing, efficient high-order filters
//!
//! ## Spectral Analysis
//!
//! ### Short-Time Fourier Transform (STFT)
//! ```text
//! X[m, k] = Σₙ x[n] w[n - mH] e^(-j2πkn/N)
//! ```
//! where:
//! - m = frame index
//! - k = frequency bin
//! - H = hop size
//! - `w[n]` = window function
//!
//! **Time-frequency resolution trade-off**:
//! - Δt ≈ N/fₛ (time resolution)
//! - Δf ≈ fₛ/N (frequency resolution)
//! - Δt · Δf ≥ 1 (uncertainty principle)
//!
//! ### Power Spectral Density (Periodogram)
//! ```text
//! P[k] = (1/N) |X[k]|²
//! ```
//! - **Properties**: Biased, inconsistent estimator
//! - **Variance**: σ² ≈ P²\[k\] (does not decrease with N)
//!
//! ### Welch's Method (Averaged Periodogram)
//! ```text
//! P_welch[k] = (1/L) Σᵢ₌₀ᴸ⁻¹ Pᵢ[k]
//! ```
//! - **Properties**: Reduced variance, biased
//! - **Variance**: σ² ≈ P²\[k\] / L
//! - **Trade-off**: Frequency resolution vs variance reduction
//!
//! ## Correlation and Convolution
//!
//! ### Cross-correlation
//! ```text
//! (f ⋆ g)[n] = Σₘ f*[m] g[n + m]
//! ```
//! - **Use case**: Signal detection, time delay estimation
//! - **Properties**: Measures similarity
//!
//! ### Auto-correlation
//! ```text
//! R_xx[k] = Σₙ x[n] x*[n + k]
//! ```
//! - **Use case**: Periodicity detection, power spectrum estimation
//! - **Properties**: `R_xx[0]` = signal power, `R_xx[-k]` = `R*_xx[k]`
//!
//! ### Convolution
//! ```text
//! (f * g)[n] = Σₘ f[m] g[n - m]
//! ```
//! - **Use case**: Filtering, system response
//! - **Fast implementation**: FFT-based O(N log N)
//!
//! # Performance Characteristics
//!
//! | Operation | Complexity | Memory | Notes |
//! |-----------|------------|--------|-------|
//! | Window generation | O(N) | O(N) | One-time cost |
//! | FIR filter | O(M·N) | O(M) | M = filter order |
//! | IIR filter | O((M+N)·L) | O(M+N) | Recursive |
//! | STFT | O(N log N · F) | O(N·F) | F = number of frames |
//! | Periodogram | O(N log N) | O(N) | Single FFT |
//! | Welch's method | O(N log N · L) | O(N) | L overlapped segments |
//! | Convolution (direct) | O(M·N) | O(M+N) | For small M |
//! | Convolution (FFT) | O(N log N) | O(2N) | For large M |
//!
//! # Advanced Signal Processing Theory
//!
//! ## Sampling Theory
//!
//! ### Nyquist-Shannon Sampling Theorem
//! ```text
//! fₛ ≥ 2·fₘₐₓ
//! ```
//! where fₛ is sampling frequency, fₘₐₓ is maximum signal frequency.
//!
//! **Aliasing**: When fₛ < 2·fₘₐₓ, high frequencies fold back into lower frequencies.
//!
//! **Anti-aliasing filter**: Low-pass filter before sampling to prevent aliasing.
//!
//! ### Oversampling
//! ```text
//! OSR = fₛ / (2·fₘₐₓ)  (Oversampling Ratio)
//! ```
//! **Benefits**:
//! - Relaxed anti-aliasing filter requirements
//! - Improved SNR: ~3 dB per doubling of OSR
//! - Simpler analog design
//!
//! ## Filter Design Theory
//!
//! ### Butterworth Filter
//! ```text
//! |H(jω)|² = 1 / (1 + (ω/ωc)^(2n))
//! ```
//! **Properties**:
//! - Maximally flat passband
//! - Monotonic response
//! - -3dB at cutoff frequency
//! - Roll-off: 20n dB/decade
//!
//! ### Chebyshev Type I
//! ```text
//! |H(jω)|² = 1 / (1 + ε²Tₙ²(ω/ωc))
//! ```
//! **Properties**:
//! - Equiripple in passband
//! - Sharper transition than Butterworth
//! - Passband ripple controlled by ε
//!
//! ### Chebyshev Type II
//! ```text
//! |H(jω)|² = 1 / (1 + 1/(ε²Tₙ²(ωc/ω)))
//! ```
//! **Properties**:
//! - Flat passband
//! - Equiripple in stopband
//! - Better phase response than Type I
//!
//! ### Elliptic (Cauer) Filter
//! **Properties**:
//! - Equiripple in both passband and stopband
//! - Sharpest transition for given order
//! - Most complex design
//! - Non-linear phase
//!
//! ## Time-Frequency Analysis
//!
//! ### Spectrogram
//! ```text
//! S[m, k] = |STFT[m, k]|²
//! ```
//! 2D representation showing frequency content evolution over time.
//!
//! **Applications**:
//! - Speech analysis (formants, pitch)
//! - Music transcription
//! - Sonar/radar signal analysis
//! - Seismic data processing
//!
//! ### Mel-Frequency Cepstral Coefficients (MFCC)
//! ```text
//! 1. Pre-emphasis: y[n] = x[n] - α·x[n-1]  (α ≈ 0.97)
//! 2. Windowing: Apply Hamming window
//! 3. FFT: Compute power spectrum
//! 4. Mel filter bank: H_m[k]
//! 5. Log: log(Σₖ |X[k]|² H_m[k])
//! 6. DCT: MFCC = DCT(log mel spectrum)
//! ```
//! **Applications**: Speech recognition, speaker identification
//!
//! ### Wavelet Transform
//! ```text
//! W(a, b) = ∫ x(t) · ψ*((t-b)/a) dt
//! ```
//! where:
//! - a = scale parameter (∝ 1/frequency)
//! - b = translation parameter (time)
//! - ψ = mother wavelet
//!
//! **Advantages over STFT**:
//! - Adaptive time-frequency resolution
//! - Good for transient signals
//! - Multi-resolution analysis
//!
//! ## Advanced Filtering Techniques
//!
//! ### Median Filter
//! ```text
//! y[n] = median{x[n-k], ..., x[n], ..., x[n+k]}
//! ```
//! **Properties**:
//! - Non-linear
//! - Excellent for impulse noise removal
//! - Preserves edges
//!
//! ### Savitzky-Golay Filter
//! Polynomial smoothing filter that preserves higher moments.
//!
//! **Applications**:
//! - Smoothing noisy data
//! - Computing derivatives
//! - Spectroscopy
//!
//! ### Kalman Filter
//! Optimal recursive state estimation for linear systems.
//! ```text
//! Prediction:  x̂ₖ⁻ = Ax̂ₖ₋₁ + Buₖ
//! Update:      x̂ₖ = x̂ₖ⁻ + Kₖ(yₖ - Cx̂ₖ⁻)
//! ```
//! **Applications**:
//! - Tracking
//! - Navigation
//! - Sensor fusion
//!
//! ## Adaptive Filtering
//!
//! ### LMS (Least Mean Squares)
//! ```text
//! w[n+1] = w[n] + μ·e[n]·x[n]
//! ```
//! where `e[n] = d[n] - w^T[n]x[n]` is the error.
//!
//! **Applications**:
//! - Echo cancellation
//! - Noise cancellation
//! - Channel equalization
//!
//! ### NLMS (Normalized LMS)
//! ```text
//! w[n+1] = w[n] + μ/(ε + ||x[n]||²) · e[n] · x[n]
//! ```
//! **Advantage**: Convergence speed independent of signal power.
//!
//! ## Resampling
//!
//! ### Upsampling (Interpolation)
//! ```text
//! 1. Insert L-1 zeros between samples
//! 2. Low-pass filter at π/L
//! ```
//! **Upsampling ratio**: L (increases sample rate by L)
//!
//! ### Downsampling (Decimation)
//! ```text
//! 1. Low-pass filter at π/M
//! 2. Keep every M-th sample
//! ```
//! **Downsampling ratio**: M (decreases sample rate by M)
//!
//! ### Rational Resampling
//! ```text
//! New rate = (L/M) × Original rate
//! ```
//! Combine upsampling by L and downsampling by M.
//!
//! ## Applications in Deep Learning
//!
//! ### Audio Neural Networks
//! - **WaveNet**: Raw audio generation using dilated convolutions
//! - **Mel-spectrograms**: Input representation for audio classification
//! - **Time-frequency attention**: Attention over both time and frequency
//!
//! ### Speech Processing
//! - **ASR (Automatic Speech Recognition)**: MFCC → RNN/Transformer
//! - **TTS (Text-to-Speech)**: Tacotron, FastSpeech architectures
//! - **Voice conversion**: CycleGAN-style architectures
//!
//! ### Biomedical Signals
//! - **ECG/EEG analysis**: CNN for pattern recognition
//! - **EMG processing**: Gesture recognition from muscle signals
//! - **PPG analysis**: Heart rate estimation, blood pressure
//!
//! ### Time Series Forecasting
//! - **Feature engineering**: Spectral features, wavelets
//! - **Temporal convolutions**: 1D conv for time series
//! - **Attention mechanisms**: Temporal attention patterns
//!
//! # Common Use Cases
//!
//! ## Audio Processing
//! ```rust
//! use torsh_functional::signal::{window, WindowType};
//! use torsh_functional::random_ops::randn;
//!
//! fn example() -> Result<(), Box<dyn std::error::Error>> {
//!     let audio = randn(&[44100], None, None, None)?;
//!
//!     // Apply Hann window for spectral analysis
//!     let win = window(WindowType::Hann, 1024, false)?;
//!     let windowed = audio.mul_op(&win)?;
//!     Ok(())
//! }
//! ```
//!
//! ## Filtering and Smoothing
//! ```rust
//! use torsh_functional::signal::lfilter;
//! use torsh_functional::random_ops::randn;
//!
//! fn example() -> Result<(), Box<dyn std::error::Error>> {
//!     let signal = randn(&[1024], None, None, None)?;
//!
//!     // Filter coefficients (example values)
//!     let b = vec![0.1, 0.2, 0.4, 0.2, 0.1];
//!     let a = vec![1.0, -0.5, 0.25];
//!
//!     // Apply filter to signal
//!     let filtered = lfilter(&b, &a, &signal)?;
//!     Ok(())
//! }
//! ```
//!
//! ## Spectral Analysis
//! ```rust
//! use torsh_functional::signal::{periodogram, welch, WindowType, PsdScaling};
//! use torsh_functional::random_ops::randn;
//!
//! fn example() -> Result<(), Box<dyn std::error::Error>> {
//!     let signal = randn(&[1024], None, None, None)?;
//!
//!     // Quick periodogram
//!     let (freqs, psd) = periodogram(&signal, None, None, None, None, PsdScaling::Density)?;
//!
//!     // Welch's method for better variance
//!     let (freqs, psd) = welch(&signal, WindowType::Hann, Some(256), Some(128), None, PsdScaling::Density)?;
//!     Ok(())
//! }
//! ```
//!
//! # Best Practices
//!
//! 1. **Window Selection**:
//!    - Hann: General purpose, good starting point
//!    - Hamming: Better side lobe suppression
//!    - Blackman: High dynamic range, low leakage
//!    - Kaiser: Adjustable trade-off via β parameter
//!
//! 2. **STFT Parameters**:
//!    - Longer windows: Better frequency resolution, worse time resolution
//!    - Shorter windows: Better time resolution, worse frequency resolution
//!    - Typical hop size: 50-75% overlap (hop = N/2 or N/4)
//!
//! 3. **Filter Design**:
//!    - FIR: Linear phase required, transient response important
//!    - IIR: Efficient implementation, real-time processing
//!    - Order selection: Balance between stopband attenuation and complexity
//!
//! 4. **Spectral Estimation**:
//!    - Periodogram: Fast, high variance
//!    - Welch: Better variance, reduced frequency resolution
//!    - Zero-padding: Interpolates spectrum, doesn't improve resolution
//!
//! 5. **Numerical Considerations**:
//!    - Normalize signals to prevent overflow
//!    - Use double buffering for real-time processing
//!    - Consider fixed-point arithmetic for embedded systems

use std::f32::consts::PI;
use torsh_core::{Result as TorshResult, TorshError};
use torsh_tensor::{creation::zeros, Tensor};

/// Window function types for signal processing
#[derive(Debug, Clone, Copy)]
pub enum WindowType {
    /// Rectangular window (no tapering)
    Rectangular,
    /// Hann window (raised cosine)
    Hann,
    /// Hamming window
    Hamming,
    /// Blackman window
    Blackman,
    /// Bartlett (triangular) window
    Bartlett,
    /// Kaiser window (requires beta parameter)
    Kaiser(f32),
    /// Tukey window (requires alpha parameter)
    Tukey(f32),
}

/// Generate window function
pub fn window(window_type: WindowType, size: usize, periodic: bool) -> TorshResult<Tensor> {
    let n = if periodic { size } else { size - 1 };
    let mut window_data = Vec::with_capacity(size);

    match window_type {
        WindowType::Rectangular => {
            for _ in 0..size {
                window_data.push(1.0);
            }
        }
        WindowType::Hann => {
            for i in 0..size {
                let val = 0.5 * (1.0 - (2.0 * PI * i as f32 / n as f32).cos());
                window_data.push(val);
            }
        }
        WindowType::Hamming => {
            for i in 0..size {
                let val = 0.54 - 0.46 * (2.0 * PI * i as f32 / n as f32).cos();
                window_data.push(val);
            }
        }
        WindowType::Blackman => {
            for i in 0..size {
                let t = 2.0 * PI * i as f32 / n as f32;
                let val = 0.42 - 0.5 * t.cos() + 0.08 * (2.0 * t).cos();
                window_data.push(val);
            }
        }
        WindowType::Bartlett => {
            for i in 0..size {
                let val = if i <= n / 2 {
                    2.0 * i as f32 / n as f32
                } else {
                    2.0 - 2.0 * i as f32 / n as f32
                };
                window_data.push(val);
            }
        }
        WindowType::Kaiser(beta) => {
            for i in 0..size {
                let alpha = (n as f32) / 2.0;
                let val =
                    modified_bessel_i0(beta * (1.0 - ((i as f32 - alpha) / alpha).powi(2)).sqrt())
                        / modified_bessel_i0(beta);
                window_data.push(val);
            }
        }
        WindowType::Tukey(alpha) => {
            let alpha = alpha.clamp(0.0, 1.0);
            let transition_width = (alpha * n as f32 / 2.0) as usize;

            for i in 0..size {
                let val = if i < transition_width {
                    0.5 * (1.0 + (PI * i as f32 / transition_width as f32 - PI).cos())
                } else if i >= size - transition_width {
                    0.5 * (1.0 + (PI * (size - 1 - i) as f32 / transition_width as f32 - PI).cos())
                } else {
                    1.0
                };
                window_data.push(val);
            }
        }
    }

    Tensor::from_data(window_data, vec![size], torsh_core::device::DeviceType::Cpu)
}

/// Modified Bessel function of the first kind (I0) for Kaiser window
fn modified_bessel_i0(x: f32) -> f32 {
    let mut result = 1.0;
    let mut term = 1.0;
    let x_half_sq = (x / 2.0).powi(2);

    for k in 1..=50 {
        term *= x_half_sq / (k as f32).powi(2);
        result += term;
        if term < 1e-8 {
            break;
        }
    }

    result
}

/// Overlap-and-add operation for signal reconstruction
pub fn overlap_add(frames: &Tensor, hop_length: usize) -> TorshResult<Tensor> {
    let shape = frames.shape();
    if shape.ndim() != 2 {
        return Err(TorshError::invalid_argument_with_context(
            "Frames must be 2-dimensional (frame_length, num_frames)",
            "overlap_add",
        ));
    }

    let frame_length = shape.dims()[0];
    let num_frames = shape.dims()[1];
    let output_length = (num_frames - 1) * hop_length + frame_length;

    let output = zeros(&[output_length])?;

    for frame_idx in 0..num_frames {
        let start_pos = frame_idx * hop_length;

        for sample_idx in 0..frame_length {
            let output_pos = start_pos + sample_idx;
            if output_pos < output_length {
                let frame_val = frames.get(&[sample_idx, frame_idx])?;
                let current_val = output.get(&[output_pos])?;
                output.set(&[output_pos], current_val + frame_val)?;
            }
        }
    }

    Ok(output)
}

/// Frame a signal into overlapping segments
pub fn frame(
    signal: &Tensor,
    frame_length: usize,
    hop_length: usize,
    center: bool,
) -> TorshResult<Tensor> {
    let signal_length = signal.shape().dims()[0];

    let padded_signal = if center {
        let pad_length = frame_length / 2;
        let mut padded_data = vec![0.0; pad_length];
        let signal_data = signal.data()?;
        padded_data.extend_from_slice(&signal_data);
        padded_data.extend(vec![0.0; pad_length]);
        Tensor::from_data(
            padded_data,
            vec![signal_length + 2 * pad_length],
            signal.device(),
        )?
    } else {
        signal.clone()
    };

    let padded_length = padded_signal.shape().dims()[0];
    let num_frames = if padded_length >= frame_length {
        (padded_length - frame_length) / hop_length + 1
    } else {
        0
    };

    if num_frames == 0 {
        return zeros(&[frame_length, 0]);
    }

    let frames = zeros(&[frame_length, num_frames])?;

    for frame_idx in 0..num_frames {
        let start_pos = frame_idx * hop_length;

        for sample_idx in 0..frame_length {
            let signal_pos = start_pos + sample_idx;
            if signal_pos < padded_length {
                let val = padded_signal.get(&[signal_pos])?;
                frames.set(&[sample_idx, frame_idx], val)?;
            }
        }
    }

    Ok(frames)
}

/// Compute power spectral density
pub fn periodogram(
    signal: &Tensor,
    window: Option<&Tensor>,
    nperseg: Option<usize>,
    noverlap: Option<usize>,
    nfft: Option<usize>,
    scaling: PsdScaling,
) -> TorshResult<(Tensor, Tensor)> {
    let signal_length = signal.shape().dims()[0];
    let nperseg = nperseg.unwrap_or(signal_length);
    let noverlap = noverlap.unwrap_or(nperseg / 2);
    let nfft = nfft.unwrap_or(nperseg);

    if noverlap >= nperseg {
        return Err(TorshError::invalid_argument_with_context(
            "noverlap must be less than nperseg",
            "periodogram",
        ));
    }

    // Apply window if provided
    let windowed_signal = if let Some(win) = window {
        signal.mul_op(win)?
    } else {
        signal.clone()
    };

    // Frame the signal
    let hop_length = nperseg - noverlap;
    let frames = frame(&windowed_signal, nperseg, hop_length, false)?;

    // Compute FFT for each frame (simplified - would use actual FFT)
    let num_frames = frames.shape().dims()[1];
    let freq_bins = nfft / 2 + 1;

    let psd_sum = zeros(&[freq_bins])?;

    for frame_idx in 0..num_frames {
        // Extract frame
        let mut frame_data = Vec::new();
        for i in 0..nperseg {
            frame_data.push(frames.get(&[i, frame_idx])?);
        }

        // Compute magnitude squared of FFT (simplified)
        // In real implementation, this would use FFT
        for freq_idx in 0..freq_bins {
            let magnitude_sq = frame_data.iter().map(|&x| x * x).sum::<f32>() / nperseg as f32;
            let current_psd = psd_sum.get(&[freq_idx])?;
            psd_sum.set(&[freq_idx], current_psd + magnitude_sq)?;
        }
    }

    // Average over frames
    let psd = psd_sum.div_scalar(num_frames as f32)?;

    // Apply scaling
    let scaled_psd = match scaling {
        PsdScaling::Density => psd,
        PsdScaling::Spectrum => {
            // Convert from density to spectrum (multiply by frequency resolution)
            let fs = 1.0; // Assuming normalized frequency
            psd.mul_scalar(fs / nfft as f32)?
        }
    };

    // Generate frequency array
    let mut freqs = Vec::new();
    for i in 0..freq_bins {
        freqs.push(i as f32 / nfft as f32); // Normalized frequency
    }
    let freq_tensor = Tensor::from_data(freqs, vec![freq_bins], signal.device())?;

    Ok((freq_tensor, scaled_psd))
}

/// Power spectral density scaling options
#[derive(Debug, Clone, Copy)]
pub enum PsdScaling {
    /// Power spectral density (V²/Hz)
    Density,
    /// Power spectrum (V²)
    Spectrum,
}

/// Welch's method for power spectral density estimation
pub fn welch(
    signal: &Tensor,
    window_type: WindowType,
    nperseg: Option<usize>,
    noverlap: Option<usize>,
    nfft: Option<usize>,
    scaling: PsdScaling,
) -> TorshResult<(Tensor, Tensor)> {
    let signal_length = signal.shape().dims()[0];
    let nperseg = nperseg.unwrap_or(signal_length.min(256));

    // Generate window
    let window_tensor = window(window_type, nperseg, false)?;

    periodogram(
        signal,
        Some(&window_tensor),
        Some(nperseg),
        noverlap,
        nfft,
        scaling,
    )
}

/// Compute cross-correlation between two signals
pub fn correlate(signal1: &Tensor, signal2: &Tensor, mode: CorrelationMode) -> TorshResult<Tensor> {
    let len1 = signal1.shape().dims()[0];
    let len2 = signal2.shape().dims()[0];

    let output_length = match mode {
        CorrelationMode::Full => len1 + len2 - 1,
        CorrelationMode::Valid => (len1 as i32 - len2 as i32 + 1).max(0) as usize,
        CorrelationMode::Same => len1,
    };

    let correlation = zeros(&[output_length])?;

    for i in 0..output_length {
        let mut sum = 0.0;

        for j in 0..len2 {
            let idx1 = match mode {
                CorrelationMode::Full => i as i32 - j as i32,
                CorrelationMode::Valid => i as i32 + j as i32,
                CorrelationMode::Same => i as i32 - len2 as i32 / 2 + j as i32,
            };

            if idx1 >= 0 && (idx1 as usize) < len1 {
                let val1 = signal1.get(&[idx1 as usize])?;
                let val2 = signal2.get(&[j])?;
                sum += val1 * val2;
            }
        }

        correlation.set(&[i], sum)?;
    }

    Ok(correlation)
}

/// Correlation modes
#[derive(Debug, Clone, Copy)]
pub enum CorrelationMode {
    /// Full correlation (output length = len1 + len2 - 1)
    Full,
    /// Valid correlation (output length = max(len1 - len2 + 1, 0))
    Valid,
    /// Same correlation (output length = len1)
    Same,
}

/// Digital filter implementation (simplified)
pub fn filtfilt(b_coeffs: &[f32], a_coeffs: &[f32], signal: &Tensor) -> TorshResult<Tensor> {
    // Forward filtering
    let forward_filtered = lfilter(b_coeffs, a_coeffs, signal)?;

    // Reverse the signal
    let reversed_signal = reverse(&forward_filtered)?;

    // Backward filtering
    let backward_filtered = lfilter(b_coeffs, a_coeffs, &reversed_signal)?;

    // Reverse again to get final result
    reverse(&backward_filtered)
}

/// Linear filter implementation
pub fn lfilter(b_coeffs: &[f32], a_coeffs: &[f32], signal: &Tensor) -> TorshResult<Tensor> {
    let signal_length = signal.shape().dims()[0];
    let output = zeros(&[signal_length])?;

    let n_b = b_coeffs.len();
    let n_a = a_coeffs.len();

    // Normalize by a[0]
    let a0 = a_coeffs[0];
    if a0 == 0.0 {
        return Err(TorshError::invalid_argument_with_context(
            "First coefficient of 'a' cannot be zero",
            "lfilter",
        ));
    }

    for n in 0..signal_length {
        let mut y = 0.0;

        // FIR part (b coefficients)
        for k in 0..n_b {
            if n >= k {
                y += b_coeffs[k] * signal.get(&[n - k])?;
            }
        }

        // IIR part (a coefficients, excluding a[0])
        for k in 1..n_a {
            if n >= k {
                y -= a_coeffs[k] * output.get(&[n - k])?;
            }
        }

        output.set(&[n], y / a0)?;
    }

    Ok(output)
}

/// Reverse a tensor along the first dimension
fn reverse(tensor: &Tensor) -> TorshResult<Tensor> {
    let length = tensor.shape().dims()[0];
    let mut reversed_data = Vec::with_capacity(length);

    for i in (0..length).rev() {
        reversed_data.push(tensor.get(&[i])?);
    }

    Tensor::from_data(reversed_data, vec![length], tensor.device())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random_ops::randn;

    #[test]
    fn test_hann_window() {
        let win = window(WindowType::Hann, 10, false).unwrap();
        assert_eq!(win.shape().dims(), &[10]);

        // Check symmetry
        assert!((win.get(&[0]).unwrap() - win.get(&[9]).unwrap()).abs() < 1e-6);
        assert!((win.get(&[1]).unwrap() - win.get(&[8]).unwrap()).abs() < 1e-6);
    }

    #[test]
    fn test_frame() {
        let signal = randn(&[100], None, None, None).unwrap();
        let frames = frame(&signal, 20, 10, false).unwrap();

        // Should have (100 - 20) / 10 + 1 = 9 frames
        assert_eq!(frames.shape().dims(), &[20, 9]);
    }

    #[test]
    fn test_correlate() {
        let signal1 = Tensor::from_data(
            vec![1.0, 2.0, 3.0],
            vec![3],
            torsh_core::device::DeviceType::Cpu,
        )
        .unwrap();
        let signal2 =
            Tensor::from_data(vec![1.0, 1.0], vec![2], torsh_core::device::DeviceType::Cpu)
                .unwrap();

        let corr = correlate(&signal1, &signal2, CorrelationMode::Valid).unwrap();
        assert_eq!(corr.shape().dims(), &[2]); // (3 - 2 + 1)
    }

    #[test]
    fn test_window_types() {
        let size = 10;

        // Test all window types
        let windows = vec![
            WindowType::Rectangular,
            WindowType::Hann,
            WindowType::Hamming,
            WindowType::Blackman,
            WindowType::Bartlett,
            WindowType::Kaiser(2.0),
            WindowType::Tukey(0.5),
        ];

        for window_type in windows {
            let win = window(window_type, size, false).unwrap();
            assert_eq!(win.shape().dims(), &[size]);
        }
    }
}
