//! Audio models for ToRSh deep learning framework
//!
//! This module provides comprehensive audio processing models including:
//! - **Wav2Vec2** for self-supervised speech representation learning
//! - **Whisper** for automatic speech recognition and translation
//! - **HuBERT** for speech representation learning
//! - **WavLM** for universal speech representation
//! - **Classification** models for various audio tasks
//!
//! ## Architecture Overview
//!
//! The audio module is organized into specialized sub-modules:
//!
//! ```
//! audio/
//! ├── common/         # Common types, utilities, and enums
//! ├── wav2vec2/       # Wav2Vec2 model family
//! ├── whisper/        # Whisper model family
//! ├── hubert/         # HuBERT model family
//! ├── wavlm/          # WavLM model family
//! └── classification/ # Audio classification models
//! ```
//!
//! ## Quick Start
//!
//! ```rust
//! use torsh_models::audio::{AudioArchitecture, wav2vec2::Wav2Vec2Config};
//!
//! // Check architecture capabilities
//! let arch = AudioArchitecture::Wav2Vec2;
//! assert!(arch.supports_self_supervised());
//! assert_eq!(arch.default_sampling_rate(), 16000);
//!
//! // Create model configuration
//! let config = Wav2Vec2Config::large();
//! ```

// Sub-modules
pub mod classification;
pub mod common;
pub mod hubert;
pub mod wav2vec2;
pub mod wavlm;
pub mod whisper;

// Re-export common types for convenience
pub use common::{
    preprocessing::{apply_window, normalize_audio, pre_emphasis, WindowType},
    utils::{
        calculate_optimal_batch_size, duration_to_frames, frames_to_duration,
        validate_sampling_rate,
    },
    AudioArchitecture,
};

// Re-export model configurations
pub use classification::{AudioClassifierConfig, AudioClassifierHead};
pub use hubert::HuBERTConfig;
pub use wav2vec2::{Wav2Vec2Config, Wav2Vec2ForCTC};
pub use wavlm::WavLMConfig;
pub use whisper::WhisperConfig;

// Type aliases for backward compatibility
pub type AudioModelConfig = AudioArchitecture;

/// Audio model presets for common use cases
pub struct AudioModelPresets;

impl AudioModelPresets {
    /// Get optimal configuration for speech recognition tasks
    pub fn speech_recognition() -> WhisperConfig {
        WhisperConfig::base()
    }

    /// Get optimal configuration for speech representation learning
    pub fn speech_representation() -> Wav2Vec2Config {
        Wav2Vec2Config::large()
    }

    /// Get optimal configuration for multilingual ASR
    pub fn multilingual_asr() -> WhisperConfig {
        WhisperConfig::large()
    }

    /// Get optimal configuration for emotion recognition
    pub fn emotion_recognition() -> AudioClassifierConfig {
        AudioClassifierConfig::emotion_recognition()
    }

    /// Get optimal configuration for music analysis
    pub fn music_analysis() -> AudioClassifierConfig {
        AudioClassifierConfig::music_genre()
    }
}

/// Audio model factory for creating pre-configured models
pub struct AudioModelFactory;

impl AudioModelFactory {
    /// Create a Wav2Vec2 model for the specified use case
    pub fn wav2vec2(variant: Wav2Vec2Variant) -> Wav2Vec2Config {
        match variant {
            Wav2Vec2Variant::Base => Wav2Vec2Config::base(),
            Wav2Vec2Variant::Large => Wav2Vec2Config::large(),
            Wav2Vec2Variant::Finetuned { vocab_size } => Wav2Vec2Config::for_finetuning(vocab_size),
        }
    }

    /// Create a Whisper model for the specified variant
    pub fn whisper(variant: WhisperVariant) -> WhisperConfig {
        match variant {
            WhisperVariant::Tiny => WhisperConfig::tiny(),
            WhisperVariant::Base => WhisperConfig::base(),
            WhisperVariant::Small => WhisperConfig::small(),
            WhisperVariant::Medium => WhisperConfig::medium(),
            WhisperVariant::Large => WhisperConfig::large(),
        }
    }

    /// Create a HuBERT model for the specified variant
    pub fn hubert(variant: HuBERTVariant) -> HuBERTConfig {
        match variant {
            HuBERTVariant::Base => HuBERTConfig::base(),
            HuBERTVariant::Large => HuBERTConfig::large(),
            HuBERTVariant::XLarge => HuBERTConfig::xlarge(),
        }
    }

    /// Create a WavLM model for the specified variant
    pub fn wavlm(variant: WavLMVariant) -> WavLMConfig {
        match variant {
            WavLMVariant::Base => WavLMConfig::base(),
            WavLMVariant::BasePlus => WavLMConfig::base_plus(),
            WavLMVariant::Large => WavLMConfig::large(),
        }
    }
}

/// Wav2Vec2 model variants
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Wav2Vec2Variant {
    Base,
    Large,
    Finetuned { vocab_size: usize },
}

/// Whisper model variants
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WhisperVariant {
    Tiny,
    Base,
    Small,
    Medium,
    Large,
}

/// HuBERT model variants
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HuBERTVariant {
    Base,
    Large,
    XLarge,
}

/// WavLM model variants
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WavLMVariant {
    Base,
    BasePlus,
    Large,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_architecture_capabilities() {
        let wav2vec2 = AudioArchitecture::Wav2Vec2;
        assert!(wav2vec2.supports_self_supervised());
        assert!(wav2vec2.supports_asr());
        assert!(!wav2vec2.supports_multilingual());

        let whisper = AudioArchitecture::Whisper;
        assert!(!whisper.supports_self_supervised());
        assert!(whisper.supports_asr());
        assert!(whisper.supports_multilingual());
    }

    #[test]
    fn test_model_factory() {
        let config = AudioModelFactory::wav2vec2(Wav2Vec2Variant::Large);
        assert_eq!(config.hidden_size, 1024);

        let config = AudioModelFactory::whisper(WhisperVariant::Large);
        assert_eq!(config.d_model, 1280);
    }

    #[test]
    fn test_audio_presets() {
        let sr_config = AudioModelPresets::speech_recognition();
        assert!(sr_config.is_encoder_decoder);

        let emotion_config = AudioModelPresets::emotion_recognition();
        assert_eq!(emotion_config.num_classes, 7);
    }

    #[test]
    fn test_audio_utils() {
        // Test frame conversion
        assert_eq!(duration_to_frames(1.0, 16000), 16000);
        assert_eq!(frames_to_duration(16000, 16000), 1.0);

        // Test sampling rate validation
        assert!(validate_sampling_rate(16000).is_ok());
        assert!(validate_sampling_rate(12345).is_err());
    }
}
