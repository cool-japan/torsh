//! Common types and utilities for audio models

use serde::{Deserialize, Serialize};

/// Audio model architectures supported by the ToRSh framework
///
/// This enum represents the different audio processing model architectures
/// available for speech recognition, representation learning, and audio analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AudioArchitecture {
    /// Wav2Vec2 - Self-supervised speech representation learning
    /// Reference: https://arxiv.org/abs/2006.11477
    Wav2Vec2,

    /// Whisper - Automatic speech recognition
    /// Reference: https://arxiv.org/abs/2212.04356
    Whisper,

    /// HuBERT - Self-supervised speech representation learning
    /// Reference: https://arxiv.org/abs/2106.07447
    HuBERT,

    /// WavLM - Universal speech representation
    /// Reference: https://arxiv.org/abs/2110.13900
    WavLM,
}

impl AudioArchitecture {
    /// Get the architecture name as a string
    pub fn as_str(&self) -> &'static str {
        match self {
            AudioArchitecture::Wav2Vec2 => "Wav2Vec2",
            AudioArchitecture::Whisper => "Whisper",
            AudioArchitecture::HuBERT => "HuBERT",
            AudioArchitecture::WavLM => "WavLM",
        }
    }

    /// Get all available architectures
    pub fn all() -> &'static [AudioArchitecture] {
        &[
            AudioArchitecture::Wav2Vec2,
            AudioArchitecture::Whisper,
            AudioArchitecture::HuBERT,
            AudioArchitecture::WavLM,
        ]
    }

    /// Check if architecture supports self-supervised learning
    pub fn supports_self_supervised(&self) -> bool {
        matches!(
            self,
            AudioArchitecture::Wav2Vec2 | AudioArchitecture::HuBERT | AudioArchitecture::WavLM
        )
    }

    /// Check if architecture supports automatic speech recognition
    pub fn supports_asr(&self) -> bool {
        matches!(
            self,
            AudioArchitecture::Wav2Vec2 | AudioArchitecture::Whisper | AudioArchitecture::WavLM
        )
    }

    /// Check if architecture supports multilingual tasks
    pub fn supports_multilingual(&self) -> bool {
        matches!(self, AudioArchitecture::Whisper | AudioArchitecture::WavLM)
    }

    /// Get the typical input sampling rate for this architecture
    pub fn default_sampling_rate(&self) -> usize {
        match self {
            AudioArchitecture::Wav2Vec2 => 16000,
            AudioArchitecture::Whisper => 16000,
            AudioArchitecture::HuBERT => 16000,
            AudioArchitecture::WavLM => 16000,
        }
    }

    /// Get the typical input audio length in seconds
    pub fn default_audio_length(&self) -> f32 {
        match self {
            AudioArchitecture::Wav2Vec2 => 20.0,
            AudioArchitecture::Whisper => 30.0,
            AudioArchitecture::HuBERT => 20.0,
            AudioArchitecture::WavLM => 20.0,
        }
    }

    /// Get the model's primary use cases
    pub fn use_cases(&self) -> &'static [&'static str] {
        match self {
            AudioArchitecture::Wav2Vec2 => &[
                "Speech Recognition",
                "Speech Representation Learning",
                "Audio Classification",
            ],
            AudioArchitecture::Whisper => &[
                "Automatic Speech Recognition",
                "Speech Translation",
                "Multilingual ASR",
            ],
            AudioArchitecture::HuBERT => &[
                "Speech Representation Learning",
                "Self-supervised Learning",
                "Speech Understanding",
            ],
            AudioArchitecture::WavLM => &[
                "Universal Speech Representation",
                "Speech Recognition",
                "Speaker Verification",
                "Audio Classification",
            ],
        }
    }
}

impl Default for AudioArchitecture {
    fn default() -> Self {
        AudioArchitecture::Wav2Vec2
    }
}

impl std::fmt::Display for AudioArchitecture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl std::str::FromStr for AudioArchitecture {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "wav2vec2" => Ok(AudioArchitecture::Wav2Vec2),
            "whisper" => Ok(AudioArchitecture::Whisper),
            "hubert" => Ok(AudioArchitecture::HuBERT),
            "wavlm" => Ok(AudioArchitecture::WavLM),
            _ => Err(format!("Unknown audio architecture: {}", s)),
        }
    }
}

/// Common audio processing utilities
pub mod utils {
    use torsh_core::error::{Result, TorshError};

    /// Validate audio sampling rate
    pub fn validate_sampling_rate(sampling_rate: usize) -> Result<()> {
        const VALID_RATES: &[usize] = &[8000, 16000, 22050, 44100, 48000];

        if VALID_RATES.contains(&sampling_rate) {
            Ok(())
        } else {
            Err(TorshError::InvalidArgument(format!(
                "Unsupported sampling rate: {}. Supported rates: {:?}",
                sampling_rate, VALID_RATES
            )))
        }
    }

    /// Calculate the number of audio frames for a given duration and sampling rate
    pub fn duration_to_frames(duration_seconds: f32, sampling_rate: usize) -> usize {
        (duration_seconds * sampling_rate as f32) as usize
    }

    /// Calculate the duration in seconds for a given number of frames and sampling rate
    pub fn frames_to_duration(frames: usize, sampling_rate: usize) -> f32 {
        frames as f32 / sampling_rate as f32
    }

    /// Calculate the optimal batch size for audio processing based on memory constraints
    pub fn calculate_optimal_batch_size(
        audio_length_frames: usize,
        feature_dim: usize,
        available_memory_mb: usize,
    ) -> usize {
        const SAFETY_FACTOR: f32 = 0.8; // Use 80% of available memory
        const BYTES_PER_FLOAT: usize = 4;

        let memory_per_sample = audio_length_frames * feature_dim * BYTES_PER_FLOAT;
        let available_memory_bytes = (available_memory_mb * 1024 * 1024) as f32 * SAFETY_FACTOR;

        let max_batch_size = (available_memory_bytes / memory_per_sample as f32) as usize;
        max_batch_size.max(1) // Ensure at least batch size of 1
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_sampling_rate_validation() {
            assert!(validate_sampling_rate(16000).is_ok());
            assert!(validate_sampling_rate(44100).is_ok());
            assert!(validate_sampling_rate(12345).is_err());
        }

        #[test]
        fn test_duration_frame_conversion() {
            assert_eq!(duration_to_frames(1.0, 16000), 16000);
            assert_eq!(frames_to_duration(16000, 16000), 1.0);
            assert_eq!(duration_to_frames(0.5, 44100), 22050);
        }

        #[test]
        fn test_batch_size_calculation() {
            let batch_size = calculate_optimal_batch_size(16000, 768, 1024);
            assert!(batch_size > 0);
            assert!(batch_size <= 100); // Reasonable upper bound
        }
    }
}

/// Audio preprocessing utilities
pub mod preprocessing {
    use torsh_core::error::Result;
    use torsh_tensor::Tensor;

    /// Apply windowing function to audio signal
    pub fn apply_window(audio: &Tensor, window_type: WindowType) -> Result<Tensor> {
        match window_type {
            WindowType::Hann => apply_hann_window(audio),
            WindowType::Hamming => apply_hamming_window(audio),
            WindowType::Blackman => apply_blackman_window(audio),
        }
    }

    /// Window types for audio preprocessing
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum WindowType {
        Hann,
        Hamming,
        Blackman,
    }

    fn apply_hann_window(audio: &Tensor) -> Result<Tensor> {
        // Placeholder implementation
        // In practice, this would apply a Hann window to the audio signal
        Ok(audio.clone())
    }

    fn apply_hamming_window(audio: &Tensor) -> Result<Tensor> {
        // Placeholder implementation
        Ok(audio.clone())
    }

    fn apply_blackman_window(audio: &Tensor) -> Result<Tensor> {
        // Placeholder implementation
        Ok(audio.clone())
    }

    /// Normalize audio signal to [-1, 1] range
    pub fn normalize_audio(audio: &Tensor) -> Result<Tensor> {
        // Simplified implementation for testing: just return a scaled clone
        // In practice, this would find the max absolute value and divide by it
        Ok(audio.clone()) // Placeholder implementation
    }

    /// Apply pre-emphasis filter to audio signal
    pub fn pre_emphasis(audio: &Tensor, coeff: f32) -> Result<Tensor> {
        // Placeholder implementation
        // In practice, this would apply: y[n] = x[n] - coeff * x[n-1]
        Ok(audio.clone())
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use torsh_tensor::creation;

        #[test]
        fn test_window_application() {
            let data: Vec<f32> = (0..1000).map(|i| (i as f32) * 0.001).collect();
            let audio = Tensor::from_vec(data, &[1000]).unwrap();
            let windowed = apply_window(&audio, WindowType::Hann).unwrap();
            assert_eq!(windowed.shape(), audio.shape());
        }

        #[test]
        fn test_audio_normalization() {
            let data: Vec<f32> = (0..1000).map(|i| (i as f32) * 0.001).collect();
            let audio = Tensor::from_vec(data, &[1000]).unwrap();
            let normalized = normalize_audio(&audio).unwrap();
            assert_eq!(normalized.shape(), audio.shape());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_architecture_basic() {
        let arch = AudioArchitecture::Wav2Vec2;
        assert_eq!(arch.as_str(), "Wav2Vec2");
        assert_eq!(format!("{}", arch), "Wav2Vec2");
    }

    #[test]
    fn test_architecture_capabilities() {
        assert!(AudioArchitecture::Wav2Vec2.supports_self_supervised());
        assert!(AudioArchitecture::Whisper.supports_asr());
        assert!(AudioArchitecture::Whisper.supports_multilingual());
        assert!(!AudioArchitecture::HuBERT.supports_multilingual());
    }

    #[test]
    fn test_architecture_defaults() {
        assert_eq!(AudioArchitecture::Wav2Vec2.default_sampling_rate(), 16000);
        assert_eq!(AudioArchitecture::Whisper.default_audio_length(), 30.0);
    }

    #[test]
    fn test_architecture_from_string() {
        assert_eq!(
            "wav2vec2".parse::<AudioArchitecture>().unwrap(),
            AudioArchitecture::Wav2Vec2
        );
        assert_eq!(
            "whisper".parse::<AudioArchitecture>().unwrap(),
            AudioArchitecture::Whisper
        );
        assert!("unknown".parse::<AudioArchitecture>().is_err());
    }

    #[test]
    fn test_all_architectures() {
        let all = AudioArchitecture::all();
        assert_eq!(all.len(), 4);
        assert!(all.contains(&AudioArchitecture::Wav2Vec2));
        assert!(all.contains(&AudioArchitecture::Whisper));
        assert!(all.contains(&AudioArchitecture::HuBERT));
        assert!(all.contains(&AudioArchitecture::WavLM));
    }

    #[test]
    fn test_use_cases() {
        let wav2vec2_cases = AudioArchitecture::Wav2Vec2.use_cases();
        assert!(wav2vec2_cases.contains(&"Speech Recognition"));

        let whisper_cases = AudioArchitecture::Whisper.use_cases();
        assert!(whisper_cases.contains(&"Multilingual ASR"));
    }
}
