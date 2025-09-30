//! Audio transformation implementations

use super::types::AudioData;
use crate::transforms::Transform;
use torsh_core::error::{Result, TorshError};
use torsh_tensor::Tensor;

/// Transform to convert audio to tensor
pub struct AudioToTensor;

impl Transform<AudioData> for AudioToTensor {
    type Output = Tensor<f32>;

    fn transform(&self, input: AudioData) -> Result<Self::Output> {
        // Convert audio samples to tensor with shape [channels, samples]
        let channels = input.channels;
        let samples_per_channel = input.samples.len() / channels;

        if channels == 1 {
            // Mono audio: shape [1, samples]
            Ok(Tensor::from_data(
                input.samples,
                vec![1, samples_per_channel],
                torsh_core::device::DeviceType::Cpu,
            )?)
        } else {
            // Multi-channel audio: interleaved to channel-first
            let mut channel_data = vec![0.0f32; input.samples.len()];
            for i in 0..samples_per_channel {
                for c in 0..channels {
                    let src_idx = i * channels + c;
                    let dst_idx = c * samples_per_channel + i;
                    channel_data[dst_idx] = input.samples[src_idx];
                }
            }

            Ok(Tensor::from_data(
                channel_data,
                vec![channels, samples_per_channel],
                torsh_core::device::DeviceType::Cpu,
            )?)
        }
    }
}

/// Transform to convert tensor to audio
pub struct TensorToAudio {
    sample_rate: u32,
}

impl TensorToAudio {
    pub fn new(sample_rate: u32) -> Self {
        Self { sample_rate }
    }
}

impl Transform<Tensor<f32>> for TensorToAudio {
    type Output = AudioData;

    fn transform(&self, input: Tensor<f32>) -> Result<Self::Output> {
        let shape = input.shape();
        if shape.ndim() != 2 {
            return Err(TorshError::InvalidShape(
                "Expected 2D tensor (channels, samples)".to_string(),
            ));
        }

        let dims = shape.dims();
        let (channels, samples_per_channel) = (dims[0], dims[1]);
        let data = input.to_vec()?;

        let audio_samples = if channels == 1 {
            // Mono audio
            data
        } else {
            // Multi-channel: convert from channel-first to interleaved
            let mut interleaved = vec![0.0f32; data.len()];
            for i in 0..samples_per_channel {
                for c in 0..channels {
                    let src_idx = c * samples_per_channel + i;
                    let dst_idx = i * channels + c;
                    interleaved[dst_idx] = data[src_idx];
                }
            }
            interleaved
        };

        Ok(AudioData::new(audio_samples, self.sample_rate, channels))
    }
}

/// Common audio transforms
pub mod transforms {
    use super::*;

    /// Resample audio to target sample rate
    pub struct Resample {
        target_sample_rate: u32,
    }

    impl Resample {
        pub fn new(target_sample_rate: u32) -> Self {
            Self { target_sample_rate }
        }
    }

    impl Transform<AudioData> for Resample {
        type Output = AudioData;

        fn transform(&self, input: AudioData) -> Result<Self::Output> {
            if input.sample_rate == self.target_sample_rate {
                return Ok(input);
            }

            // Simple linear resampling (not production quality)
            let ratio = self.target_sample_rate as f32 / input.sample_rate as f32;
            let new_length = (input.samples.len() as f32 * ratio) as usize;
            let mut resampled = Vec::with_capacity(new_length);

            for i in 0..new_length {
                let src_index = i as f32 / ratio;
                let src_index_floor = src_index.floor() as usize;
                let src_index_ceil = (src_index_floor + 1).min(input.samples.len() - 1);
                let fraction = src_index - src_index_floor as f32;

                if src_index_floor < input.samples.len() {
                    let sample = input.samples[src_index_floor] * (1.0 - fraction)
                        + input.samples[src_index_ceil] * fraction;
                    resampled.push(sample);
                }
            }

            Ok(AudioData::new(
                resampled,
                self.target_sample_rate,
                input.channels,
            ))
        }
    }

    /// Trim or pad audio to fixed length
    pub struct FixedLength {
        length: usize,
        pad_value: f32,
    }

    impl FixedLength {
        pub fn new(length: usize) -> Self {
            Self {
                length,
                pad_value: 0.0,
            }
        }

        pub fn with_pad_value(mut self, pad_value: f32) -> Self {
            self.pad_value = pad_value;
            self
        }
    }

    impl Transform<AudioData> for FixedLength {
        type Output = AudioData;

        fn transform(&self, input: AudioData) -> Result<Self::Output> {
            let target_total_length = self.length * input.channels;
            let mut samples = input.samples;

            match samples.len().cmp(&target_total_length) {
                std::cmp::Ordering::Greater => {
                    // Trim
                    samples.truncate(target_total_length);
                }
                std::cmp::Ordering::Less => {
                    // Pad
                    samples.resize(target_total_length, self.pad_value);
                }
                std::cmp::Ordering::Equal => {
                    // No change needed
                }
            }

            Ok(AudioData::new(samples, input.sample_rate, input.channels))
        }
    }

    /// Normalize audio amplitude
    pub struct Normalize {
        target_rms: f32,
    }

    impl Normalize {
        pub fn new(target_rms: f32) -> Self {
            Self { target_rms }
        }
    }

    impl Transform<AudioData> for Normalize {
        type Output = AudioData;

        fn transform(&self, input: AudioData) -> Result<Self::Output> {
            // Calculate RMS
            let rms = (input.samples.iter().map(|&x| x * x).sum::<f32>()
                / input.samples.len() as f32)
                .sqrt();

            if rms == 0.0 {
                return Ok(input); // Avoid division by zero
            }

            let gain = self.target_rms / rms;
            let normalized_samples: Vec<f32> = input.samples.iter().map(|&x| x * gain).collect();

            Ok(AudioData::new(
                normalized_samples,
                input.sample_rate,
                input.channels,
            ))
        }
    }

    /// Add noise to audio
    pub struct AddNoise {
        noise_level: f32,
    }

    impl AddNoise {
        pub fn new(noise_level: f32) -> Self {
            assert!(noise_level >= 0.0, "Noise level must be non-negative");
            Self { noise_level }
        }
    }

    impl Transform<AudioData> for AddNoise {
        type Output = AudioData;

        fn transform(&self, input: AudioData) -> Result<Self::Output> {
            // ✅ SciRS2 Policy Compliant - Using scirs2_core::random instead of direct rand
            use scirs2_core::random::{Random, Rng};
            let mut rng = Random::seed(42);

            let noisy_samples: Vec<f32> = input
                .samples
                .iter()
                .map(|&sample| {
                    let noise = rng.gen_range(-1.0..1.0) * self.noise_level;
                    sample + noise
                })
                .collect();

            Ok(AudioData::new(
                noisy_samples,
                input.sample_rate,
                input.channels,
            ))
        }
    }
}
