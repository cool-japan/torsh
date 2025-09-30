//! Signal processing operations

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
