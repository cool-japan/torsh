//! Audio-specific datasets and transformations

use torsh_tensor::Tensor;
use torsh_core::{
    error::{Result, TorshError},
    dtype::FloatElement,
};
use crate::{dataset::Dataset, transforms::Transform};

#[cfg(not(feature = "std"))]
use alloc::{vec::Vec, string::String, boxed::Box};
use std::path::{Path, PathBuf};

/// Audio file dataset for loading audio from directories
pub struct AudioFolder {
    root: PathBuf,
    samples: Vec<(PathBuf, usize)>,
    classes: Vec<String>,
    transform: Option<Box<dyn Transform<AudioData, Output = Tensor<f32>>>>,
    sample_rate: Option<u32>,
}

/// Simple audio data container
#[derive(Debug, Clone)]
pub struct AudioData {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: usize,
}

impl AudioData {
    pub fn new(samples: Vec<f32>, sample_rate: u32, channels: usize) -> Self {
        Self {
            samples,
            sample_rate, 
            channels,
        }
    }
    
    pub fn duration(&self) -> f32 {
        self.samples.len() as f32 / (self.sample_rate as f32 * self.channels as f32)
    }
    
    pub fn len(&self) -> usize {
        self.samples.len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

impl AudioFolder {
    /// Create a new audio folder dataset
    pub fn new<P: AsRef<Path>>(root: P, sample_rate: Option<u32>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        
        if !root.exists() {
            return Err(TorshError::IoError(
                format!("Directory does not exist: {:?}", root)
            ));
        }
        
        let mut classes = Vec::new();
        let mut samples = Vec::new();
        
        // Scan subdirectories for classes
        for entry in std::fs::read_dir(&root)
            .map_err(|e| TorshError::IoError(e.to_string()))?
        {
            let entry = entry.map_err(|e| TorshError::IoError(e.to_string()))?;
            let path = entry.path();
            
            if path.is_dir() {
                let class_name = path.file_name()
                    .and_then(|n| n.to_str())
                    .ok_or_else(|| TorshError::IoError("Invalid class directory name".to_string()))?
                    .to_string();
                
                let class_idx = classes.len();
                classes.push(class_name);
                
                // Scan audio files in class directory
                for audio_entry in std::fs::read_dir(&path)
                    .map_err(|e| TorshError::IoError(e.to_string()))?
                {
                    let audio_entry = audio_entry.map_err(|e| TorshError::IoError(e.to_string()))?;
                    let audio_path = audio_entry.path();
                    
                    if Self::is_audio_file(&audio_path) {
                        samples.push((audio_path, class_idx));
                    }
                }
            }
        }
        
        Ok(Self {
            root,
            samples,
            classes,
            transform: None,
            sample_rate,
        })
    }
    
    /// Set transform to apply to audio
    pub fn with_transform<T>(mut self, transform: T) -> Self 
    where
        T: Transform<AudioData, Output = Tensor<f32>> + 'static,
    {
        self.transform = Some(Box::new(transform));
        self
    }
    
    /// Get class names
    pub fn classes(&self) -> &[String] {
        &self.classes
    }
    
    /// Check if file is a supported audio format
    fn is_audio_file(path: &Path) -> bool {
        if let Some(extension) = path.extension().and_then(|ext| ext.to_str()) {
            matches!(extension.to_lowercase().as_str(), 
                "wav" | "mp3" | "flac" | "ogg" | "m4a" | "aac")
        } else {
            false
        }
    }
    
    /// Load audio from path (simplified implementation)
    fn load_audio(&self, path: &Path) -> Result<AudioData> {
        // Placeholder implementation - in a real scenario you'd use a library like:
        // - symphonia for pure Rust audio decoding
        // - rodio for audio playback and simple loading
        // - cpal for low-level audio I/O
        // - hound for WAV files specifically
        
        // For now, return dummy audio data
        let sample_rate = self.sample_rate.unwrap_or(22050);
        let duration_seconds = 3.0; // 3 second dummy audio
        let samples_count = (sample_rate as f32 * duration_seconds) as usize;
        let samples: Vec<f32> = (0..samples_count)
            .map(|i| (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / sample_rate as f32).sin() * 0.1)
            .collect();
        
        Ok(AudioData::new(samples, sample_rate, 1))
    }
}

impl Dataset for AudioFolder {
    type Item = (Tensor<f32>, usize);
    
    fn len(&self) -> usize {
        self.samples.len()
    }
    
    fn get(&self, index: usize) -> Result<Self::Item> {
        if index >= self.samples.len() {
            return Err(TorshError::IndexError {
                index,
                size: self.samples.len(),
            });
        }
        
        let (ref path, class_idx) = self.samples[index];
        let audio = self.load_audio(path)?;
        
        let tensor = if let Some(ref transform) = self.transform {
            transform.transform(audio)?
        } else {
            // Default: convert to tensor
            AudioToTensor.transform(audio)?
        };
        
        Ok((tensor, class_idx))
    }
}

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
                torsh_core::device::DeviceType::Cpu
            ))
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
                torsh_core::device::DeviceType::Cpu
            ))
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
                "Expected 2D tensor (channels, samples)".to_string()
            ));
        }
        
        let dims = shape.dims();
        let (channels, samples_per_channel) = (dims[0], dims[1]);
        let data = input.to_vec();
        
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
    use crate::transforms::Transform;
    
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
                    let sample = input.samples[src_index_floor] * (1.0 - fraction) +
                                input.samples[src_index_ceil] * fraction;
                    resampled.push(sample);
                }
            }
            
            Ok(AudioData::new(resampled, self.target_sample_rate, input.channels))
        }
    }
    
    /// Trim or pad audio to fixed length
    pub struct FixedLength {
        length: usize,
        pad_value: f32,
    }
    
    impl FixedLength {
        pub fn new(length: usize) -> Self {
            Self { length, pad_value: 0.0 }
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
            
            if samples.len() > target_total_length {
                // Trim
                samples.truncate(target_total_length);
            } else if samples.len() < target_total_length {
                // Pad
                samples.resize(target_total_length, self.pad_value);
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
            let rms = (input.samples.iter()
                .map(|&x| x * x)
                .sum::<f32>() / input.samples.len() as f32)
                .sqrt();
            
            if rms == 0.0 {
                return Ok(input); // Avoid division by zero
            }
            
            let gain = self.target_rms / rms;
            let normalized_samples: Vec<f32> = input.samples
                .iter()
                .map(|&x| x * gain)
                .collect();
            
            Ok(AudioData::new(normalized_samples, input.sample_rate, input.channels))
        }
    }
    
    /// Simple mel-scale frequency transform (placeholder)
    pub struct MelSpectrogram {
        n_fft: usize,
        hop_length: usize,
        n_mels: usize,
    }
    
    impl MelSpectrogram {
        pub fn new(n_fft: usize, hop_length: usize, n_mels: usize) -> Self {
            Self { n_fft, hop_length, n_mels }
        }
    }
    
    impl Transform<AudioData> for MelSpectrogram {
        type Output = Tensor<f32>;
        
        fn transform(&self, input: AudioData) -> Result<Self::Output> {
            // Placeholder implementation
            // In a real implementation, you'd use FFT and mel-scale conversion
            let frames = input.samples.len() / self.hop_length;
            let spectrogram_data = vec![0.1f32; self.n_mels * frames];
            
            Ok(Tensor::from_data(
                spectrogram_data,
                vec![self.n_mels, frames],
                torsh_core::device::DeviceType::Cpu
            ))
        }
    }
}

/// Common speech/audio datasets
pub struct LibriSpeech {
    root: PathBuf,
    subset: String,
    transform: Option<Box<dyn Transform<AudioData, Output = Tensor<f32>>>>,
    samples: Vec<(PathBuf, String)>, // (audio_path, transcript)
}

impl LibriSpeech {
    pub fn new<P: AsRef<Path>>(root: P, subset: &str) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        
        // TODO: Implement actual LibriSpeech loading
        // For now, create dummy data
        let samples = vec![
            (root.join("dummy1.wav"), "Hello world".to_string()),
            (root.join("dummy2.wav"), "This is a test".to_string()),
        ];
        
        Ok(Self {
            root,
            subset: subset.to_string(),
            transform: None,
            samples,
        })
    }
    
    pub fn with_transform<T>(mut self, transform: T) -> Self 
    where
        T: Transform<AudioData, Output = Tensor<f32>> + 'static,
    {
        self.transform = Some(Box::new(transform));
        self
    }
}

impl Dataset for LibriSpeech {
    type Item = (Tensor<f32>, String);
    
    fn len(&self) -> usize {
        self.samples.len()
    }
    
    fn get(&self, index: usize) -> Result<Self::Item> {
        if index >= self.samples.len() {
            return Err(TorshError::IndexError {
                index,
                size: self.samples.len(),
            });
        }
        
        let (ref _path, ref transcript) = self.samples[index];
        
        // Generate dummy audio for now
        let dummy_audio = AudioData::new(
            vec![0.1f32; 22050], // 1 second of audio at 22kHz
            22050,
            1
        );
        
        let tensor = if let Some(ref transform) = self.transform {
            transform.transform(dummy_audio)?
        } else {
            AudioToTensor.transform(dummy_audio)?
        };
        
        Ok((tensor, transcript.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_audio_data() {
        let samples = vec![0.1, 0.2, 0.3, 0.4];
        let audio = AudioData::new(samples, 22050, 1);
        
        assert_eq!(audio.sample_rate, 22050);
        assert_eq!(audio.channels, 1);
        assert_eq!(audio.len(), 4);
        assert!(!audio.is_empty());
    }
    
    #[test]
    fn test_audio_to_tensor() {
        let samples = vec![0.1, 0.2, 0.3, 0.4];
        let audio = AudioData::new(samples, 22050, 1);
        
        let transform = AudioToTensor;
        let result = transform.transform(audio);
        assert!(result.is_ok());
        
        let tensor = result.unwrap();
        assert_eq!(tensor.shape().dims(), &[1, 4]);
    }
    
    #[test]
    fn test_stereo_audio_to_tensor() {
        // Interleaved stereo: [L1, R1, L2, R2]
        let samples = vec![0.1, 0.2, 0.3, 0.4];
        let audio = AudioData::new(samples, 22050, 2);
        
        let transform = AudioToTensor;
        let result = transform.transform(audio);
        assert!(result.is_ok());
        
        let tensor = result.unwrap();
        assert_eq!(tensor.shape().dims(), &[2, 2]); // [channels, samples_per_channel]
    }
    
    #[test]
    fn test_resample_transform() {
        let samples = vec![0.1, 0.2, 0.3, 0.4];
        let audio = AudioData::new(samples, 22050, 1);
        
        let resample = transforms::Resample::new(44100);
        let result = resample.transform(audio);
        assert!(result.is_ok());
        
        let resampled = result.unwrap();
        assert_eq!(resampled.sample_rate, 44100);
        assert!(resampled.samples.len() > 4); // Should have more samples at higher rate
    }
    
    #[test]
    fn test_fixed_length_transform() {
        let samples = vec![0.1, 0.2];
        let audio = AudioData::new(samples, 22050, 1);
        
        let fixed_len = transforms::FixedLength::new(5);
        let result = fixed_len.transform(audio);
        assert!(result.is_ok());
        
        let padded = result.unwrap();
        assert_eq!(padded.samples.len(), 5);
    }
    
    #[test]
    fn test_librispeech() {
        let dataset = LibriSpeech::new("/tmp", "test").unwrap();
        assert_eq!(dataset.len(), 2);
        
        let (audio, transcript) = dataset.get(0).unwrap();
        assert_eq!(audio.shape().dims(), &[1, 22050]); // 1 second mono audio
        assert!(!transcript.is_empty());
    }
}