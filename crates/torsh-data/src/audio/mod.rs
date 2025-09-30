//! Audio-specific datasets and transformations
//!
//! This module provides comprehensive audio processing capabilities including:
//! - Core audio data types and utilities
//! - Dataset implementations for audio classification and speech tasks
//! - Transform implementations for preprocessing and augmentation
//!
//! # Examples
//!
//! ```rust
//! use torsh_data::audio::{AudioFolder, AudioData, AudioToTensor};
//! use torsh_data::audio::transforms::transforms::*;
//!
//! // Create an audio folder dataset
//! let dataset = AudioFolder::new("/path/to/audio", Some(22050))?;
//!
//! // Apply transforms
//! let normalize = Normalize::new(0.1);
//! let resample = Resample::new(16000);
//! ```

pub mod datasets;
pub mod transforms;
pub mod types;

// Re-export core types for convenience
pub use datasets::{AudioFolder, LibriSpeech};
pub use transforms::{AudioToTensor, TensorToAudio};
pub use types::AudioData;

// Re-export common transforms
pub use transforms::transforms::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::Dataset;
    use crate::transforms::Transform;

    #[test]
    fn test_audio_data_creation() {
        let samples = vec![0.1, -0.2, 0.3, -0.4];
        let audio = AudioData::new(samples.clone(), 44100, 1);

        assert_eq!(audio.samples, samples);
        assert_eq!(audio.sample_rate, 44100);
        assert_eq!(audio.channels, 1);
        assert_eq!(audio.len(), 4);
        assert!(!audio.is_empty());
    }

    #[test]
    fn test_audio_data_duration() {
        let samples = vec![0.0; 44100]; // 1 second at 44100 Hz
        let audio = AudioData::new(samples, 44100, 1);

        assert_eq!(audio.duration(), 1.0);
    }

    #[test]
    fn test_audio_to_tensor_mono() {
        let samples = vec![0.1, -0.2, 0.3, -0.4];
        let audio = AudioData::new(samples, 44100, 1);
        let transform = AudioToTensor;

        let result = transform.transform(audio);
        assert!(result.is_ok());

        let tensor = result.unwrap();
        let shape = tensor.shape();
        assert_eq!(shape.dims(), &[1, 4]); // [channels, samples]
    }

    #[test]
    fn test_audio_to_tensor_stereo() {
        // Interleaved stereo: L, R, L, R
        let samples = vec![0.1, 0.2, -0.3, -0.4];
        let audio = AudioData::new(samples, 44100, 2);
        let transform = AudioToTensor;

        let result = transform.transform(audio);
        assert!(result.is_ok());

        let tensor = result.unwrap();
        let shape = tensor.shape();
        assert_eq!(shape.dims(), &[2, 2]); // [channels, samples_per_channel]
    }

    #[test]
    fn test_tensor_to_audio_mono() {
        use torsh_core::device::DeviceType;
        use torsh_tensor::Tensor;

        let data = vec![0.1, -0.2, 0.3, -0.4];
        let tensor = Tensor::from_data(data.clone(), vec![1, 4], DeviceType::Cpu).unwrap();

        let transform = TensorToAudio::new(44100);
        let result = transform.transform(tensor);
        assert!(result.is_ok());

        let audio = result.unwrap();
        assert_eq!(audio.samples, data);
        assert_eq!(audio.sample_rate, 44100);
        assert_eq!(audio.channels, 1);
    }

    #[test]
    fn test_tensor_to_audio_stereo() {
        use torsh_core::device::DeviceType;
        use torsh_tensor::Tensor;

        // Channel-first format: [ch0_s0, ch0_s1, ch1_s0, ch1_s1]
        let data = vec![0.1, -0.2, 0.3, -0.4];
        let tensor = Tensor::from_data(data, vec![2, 2], DeviceType::Cpu).unwrap();

        let transform = TensorToAudio::new(44100);
        let result = transform.transform(tensor);
        assert!(result.is_ok());

        let audio = result.unwrap();
        // Should be interleaved: [ch0_s0, ch1_s0, ch0_s1, ch1_s1]
        assert_eq!(audio.samples, vec![0.1, 0.3, -0.2, -0.4]);
        assert_eq!(audio.sample_rate, 44100);
        assert_eq!(audio.channels, 2);
    }
}
