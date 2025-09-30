//! Prosodic Configuration Module
//!
//! This module provides comprehensive configuration management for prosodic fluency analysis,
//! including hierarchical configuration structures for rhythm, stress, intonation, timing,
//! phonological patterns, and advanced prosodic features.

use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// Comprehensive prosodic analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProsodicConfig {
    /// General prosodic analysis settings
    pub general: GeneralProsodicConfig,
    /// Rhythm analysis configuration
    pub rhythm: RhythmAnalysisConfig,
    /// Stress analysis configuration
    pub stress: StressAnalysisConfig,
    /// Intonation analysis configuration
    pub intonation: IntonationAnalysisConfig,
    /// Timing analysis configuration
    pub timing: TimingAnalysisConfig,
    /// Phonological analysis configuration
    pub phonological: PhonologicalAnalysisConfig,
    /// Advanced prosodic analysis configuration
    pub advanced: AdvancedProsodicConfig,
}

/// General prosodic configuration settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneralProsodicConfig {
    /// Enable prosodic analysis
    pub enabled: bool,
    /// Maximum text length for analysis
    pub max_text_length: usize,
    /// Sentence delimiters for parsing
    pub sentence_delimiters: Vec<char>,
    /// Word delimiters for parsing
    pub word_delimiters: Vec<char>,
    /// Use caching for performance
    pub use_caching: bool,
    /// Cache size limit
    pub cache_size_limit: usize,
}

/// Rhythm analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RhythmAnalysisConfig {
    /// Enable rhythm analysis
    pub enabled: bool,
    /// Weight for rhythmic flow analysis
    pub rhythm_weight: f64,
    /// Rhythm regularity preference
    pub prefer_regular_rhythm: bool,
    /// Enable rhythm template matching
    pub enable_template_matching: bool,
    /// Number of rhythm templates to use
    pub max_rhythm_templates: usize,
    /// Beat detection sensitivity
    pub beat_detection_sensitivity: f64,
    /// Alternation pattern preference
    pub alternation_preference: f64,
    /// Enable rhythm classification
    pub enable_rhythm_classification: bool,
    /// Rhythm complexity threshold
    pub rhythm_complexity_threshold: f64,
    /// Minimum pattern length for analysis
    pub min_pattern_length: usize,
}

/// Stress analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressAnalysisConfig {
    /// Enable stress analysis
    pub enabled: bool,
    /// Weight for stress pattern naturalness
    pub stress_weight: f64,
    /// Enable stress pattern prediction
    pub enable_stress_prediction: bool,
    /// Enable metrical structure analysis
    pub enable_metrical_analysis: bool,
    /// Metrical consistency threshold
    pub metrical_consistency_threshold: f64,
    /// Enable prominence analysis
    pub enable_prominence_analysis: bool,
    /// Foot boundary detection accuracy
    pub foot_boundary_accuracy: f64,
    /// Stress clash detection
    pub detect_stress_clashes: bool,
    /// Primary stress preference
    pub primary_stress_preference: f64,
    /// Enable accent pattern analysis
    pub enable_accent_analysis: bool,
}

/// Intonation analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntonationAnalysisConfig {
    /// Enable intonation analysis
    pub enabled: bool,
    /// Weight for intonation appropriateness
    pub intonation_weight: f64,
    /// Enable pitch contour analysis
    pub enable_pitch_contour: bool,
    /// Enable boundary tone detection
    pub enable_boundary_tone: bool,
    /// Focus pattern detection
    pub detect_focus_patterns: bool,
    /// Sentence type classification
    pub classify_sentence_types: bool,
    /// Pitch range analysis
    pub analyze_pitch_range: bool,
    /// Intonational phrase detection
    pub detect_intonational_phrases: bool,
    /// Contour smoothness preference
    pub contour_smoothness_preference: f64,
    /// Enable tonal accent analysis
    pub enable_tonal_accents: bool,
}

/// Timing analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingAnalysisConfig {
    /// Enable timing analysis
    pub enabled: bool,
    /// Weight for pause placement
    pub pause_weight: f64,
    /// Enable prosodic break detection
    pub enable_break_detection: bool,
    /// Pause placement accuracy threshold
    pub pause_accuracy_threshold: f64,
    /// Enable tempo analysis
    pub enable_tempo_analysis: bool,
    /// Tempo regularity preference
    pub tempo_regularity_preference: f64,
    /// Enable duration analysis
    pub enable_duration_analysis: bool,
    /// Syllable timing analysis
    pub analyze_syllable_timing: bool,
    /// Enable speech rate calculation
    pub calculate_speech_rate: bool,
    /// Timing variability analysis
    pub analyze_timing_variability: bool,
}

/// Phonological analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhonologicalAnalysisConfig {
    /// Enable phonological analysis
    pub enabled: bool,
    /// Enable alliteration detection
    pub detect_alliteration: bool,
    /// Enable assonance detection
    pub detect_assonance: bool,
    /// Enable consonance detection
    pub detect_consonance: bool,
    /// Enable rhyme detection
    pub detect_rhyme: bool,
    /// Phonological pattern sensitivity
    pub pattern_sensitivity: f64,
    /// Sound similarity threshold
    pub sound_similarity_threshold: f64,
    /// Enable phonotactic analysis
    pub enable_phonotactics: bool,
    /// Syllable structure analysis
    pub analyze_syllable_structure: bool,
    /// Enable euphony analysis
    pub analyze_euphony: bool,
}

/// Advanced prosodic analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedProsodicConfig {
    /// Enable advanced analysis
    pub enabled: bool,
    /// Enable prosodic hierarchy analysis
    pub analyze_prosodic_hierarchy: bool,
    /// Enable complexity analysis
    pub analyze_complexity: bool,
    /// Complexity calculation method
    pub complexity_method: ComplexityMethod,
    /// Enable entropy analysis
    pub calculate_entropy: bool,
    /// Enable machine learning features
    pub use_ml_features: bool,
    /// Feature vector dimensions
    pub feature_dimensions: usize,
    /// Enable neural prosodic modeling
    pub use_neural_modeling: bool,
    /// Advanced caching strategy
    pub advanced_caching: bool,
    /// Enable prosodic profiling
    pub enable_profiling: bool,
}

/// Methods for calculating prosodic complexity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplexityMethod {
    /// Information-theoretic entropy
    InformationEntropy,
    /// Pattern variation analysis
    PatternVariation,
    /// Hierarchical complexity
    HierarchicalComplexity,
    /// Combined multi-method approach
    Combined,
}

/// Preset configurations for common use cases
#[derive(Debug, Clone, Copy)]
pub enum ProsodicAnalysisPreset {
    /// Minimal analysis for basic prosodic features
    Minimal,
    /// Comprehensive analysis with all features enabled
    Comprehensive,
    /// Optimized for speech synthesis applications
    SpeechSynthesis,
    /// Optimized for language learning applications
    LanguageLearning,
    /// Optimized for linguistic research
    LinguisticResearch,
    /// Optimized for reading fluency assessment
    ReadingFluency,
}

impl ProsodicConfig {
    /// Create minimal prosodic configuration
    pub fn minimal() -> Self {
        Self {
            general: GeneralProsodicConfig {
                enabled: true,
                max_text_length: 10000,
                sentence_delimiters: vec!['.', '!', '?'],
                word_delimiters: vec![' ', '\t', '\n'],
                use_caching: false,
                cache_size_limit: 1000,
            },
            rhythm: RhythmAnalysisConfig {
                enabled: true,
                rhythm_weight: 0.4,
                prefer_regular_rhythm: true,
                enable_template_matching: false,
                max_rhythm_templates: 10,
                beat_detection_sensitivity: 0.5,
                alternation_preference: 0.6,
                enable_rhythm_classification: false,
                rhythm_complexity_threshold: 0.7,
                min_pattern_length: 3,
            },
            stress: StressAnalysisConfig {
                enabled: true,
                stress_weight: 0.3,
                enable_stress_prediction: false,
                enable_metrical_analysis: false,
                metrical_consistency_threshold: 0.6,
                enable_prominence_analysis: false,
                foot_boundary_accuracy: 0.7,
                detect_stress_clashes: false,
                primary_stress_preference: 0.8,
                enable_accent_analysis: false,
            },
            intonation: IntonationAnalysisConfig {
                enabled: true,
                intonation_weight: 0.3,
                enable_pitch_contour: false,
                enable_boundary_tone: false,
                detect_focus_patterns: false,
                classify_sentence_types: true,
                analyze_pitch_range: false,
                detect_intonational_phrases: false,
                contour_smoothness_preference: 0.7,
                enable_tonal_accents: false,
            },
            timing: TimingAnalysisConfig {
                enabled: false,
                pause_weight: 0.0,
                enable_break_detection: false,
                pause_accuracy_threshold: 0.7,
                enable_tempo_analysis: false,
                tempo_regularity_preference: 0.6,
                enable_duration_analysis: false,
                analyze_syllable_timing: false,
                calculate_speech_rate: false,
                analyze_timing_variability: false,
            },
            phonological: PhonologicalAnalysisConfig {
                enabled: false,
                detect_alliteration: false,
                detect_assonance: false,
                detect_consonance: false,
                detect_rhyme: false,
                pattern_sensitivity: 0.5,
                sound_similarity_threshold: 0.6,
                enable_phonotactics: false,
                analyze_syllable_structure: false,
                analyze_euphony: false,
            },
            advanced: AdvancedProsodicConfig {
                enabled: false,
                analyze_prosodic_hierarchy: false,
                analyze_complexity: false,
                complexity_method: ComplexityMethod::InformationEntropy,
                calculate_entropy: false,
                use_ml_features: false,
                feature_dimensions: 50,
                use_neural_modeling: false,
                advanced_caching: false,
                enable_profiling: false,
            },
        }
    }

    /// Create comprehensive prosodic configuration
    pub fn comprehensive() -> Self {
        Self {
            general: GeneralProsodicConfig {
                enabled: true,
                max_text_length: 50000,
                sentence_delimiters: vec!['.', '!', '?', ';', ':'],
                word_delimiters: vec![' ', '\t', '\n', '-', '_'],
                use_caching: true,
                cache_size_limit: 10000,
            },
            rhythm: RhythmAnalysisConfig {
                enabled: true,
                rhythm_weight: 0.25,
                prefer_regular_rhythm: true,
                enable_template_matching: true,
                max_rhythm_templates: 50,
                beat_detection_sensitivity: 0.7,
                alternation_preference: 0.8,
                enable_rhythm_classification: true,
                rhythm_complexity_threshold: 0.5,
                min_pattern_length: 2,
            },
            stress: StressAnalysisConfig {
                enabled: true,
                stress_weight: 0.20,
                enable_stress_prediction: true,
                enable_metrical_analysis: true,
                metrical_consistency_threshold: 0.8,
                enable_prominence_analysis: true,
                foot_boundary_accuracy: 0.9,
                detect_stress_clashes: true,
                primary_stress_preference: 0.9,
                enable_accent_analysis: true,
            },
            intonation: IntonationAnalysisConfig {
                enabled: true,
                intonation_weight: 0.20,
                enable_pitch_contour: true,
                enable_boundary_tone: true,
                detect_focus_patterns: true,
                classify_sentence_types: true,
                analyze_pitch_range: true,
                detect_intonational_phrases: true,
                contour_smoothness_preference: 0.8,
                enable_tonal_accents: true,
            },
            timing: TimingAnalysisConfig {
                enabled: true,
                pause_weight: 0.15,
                enable_break_detection: true,
                pause_accuracy_threshold: 0.8,
                enable_tempo_analysis: true,
                tempo_regularity_preference: 0.7,
                enable_duration_analysis: true,
                analyze_syllable_timing: true,
                calculate_speech_rate: true,
                analyze_timing_variability: true,
            },
            phonological: PhonologicalAnalysisConfig {
                enabled: true,
                detect_alliteration: true,
                detect_assonance: true,
                detect_consonance: true,
                detect_rhyme: true,
                pattern_sensitivity: 0.7,
                sound_similarity_threshold: 0.8,
                enable_phonotactics: true,
                analyze_syllable_structure: true,
                analyze_euphony: true,
            },
            advanced: AdvancedProsodicConfig {
                enabled: true,
                analyze_prosodic_hierarchy: true,
                analyze_complexity: true,
                complexity_method: ComplexityMethod::Combined,
                calculate_entropy: true,
                use_ml_features: true,
                feature_dimensions: 100,
                use_neural_modeling: true,
                advanced_caching: true,
                enable_profiling: true,
            },
        }
    }

    /// Create configuration for speech synthesis
    pub fn speech_synthesis() -> Self {
        let mut config = Self::comprehensive();

        // Optimize for speech synthesis needs
        config.rhythm.rhythm_weight = 0.35;
        config.stress.stress_weight = 0.30;
        config.intonation.intonation_weight = 0.25;
        config.timing.pause_weight = 0.10;

        // Enable features critical for synthesis
        config.rhythm.enable_rhythm_classification = true;
        config.stress.enable_prominence_analysis = true;
        config.intonation.enable_pitch_contour = true;
        config.timing.enable_tempo_analysis = true;

        config
    }

    /// Create configuration for language learning
    pub fn language_learning() -> Self {
        let mut config = Self::comprehensive();

        // Focus on learning-relevant features
        config.stress.stress_weight = 0.35;
        config.intonation.intonation_weight = 0.30;
        config.rhythm.rhythm_weight = 0.25;
        config.phonological.pattern_sensitivity = 0.8;

        // Enable pronunciation-focused features
        config.stress.detect_stress_clashes = true;
        config.intonation.classify_sentence_types = true;
        config.phonological.analyze_euphony = true;

        config
    }

    /// Create configuration for linguistic research
    pub fn linguistic_research() -> Self {
        let mut config = Self::comprehensive();

        // Enable all advanced features for research
        config.advanced.analyze_prosodic_hierarchy = true;
        config.advanced.calculate_entropy = true;
        config.advanced.use_ml_features = true;
        config.advanced.enable_profiling = true;

        // Maximum detail in all areas
        config.phonological.enable_phonotactics = true;
        config.timing.analyze_timing_variability = true;
        config.stress.enable_metrical_analysis = true;

        config
    }

    /// Create configuration for reading fluency
    pub fn reading_fluency() -> Self {
        let mut config = Self::minimal();

        // Focus on reading-relevant prosodic features
        config.rhythm.enabled = true;
        config.rhythm.rhythm_weight = 0.40;
        config.stress.enabled = true;
        config.stress.stress_weight = 0.30;
        config.timing.enabled = true;
        config.timing.pause_weight = 0.20;
        config.intonation.intonation_weight = 0.10;

        // Enable reading-specific features
        config.timing.enable_break_detection = true;
        config.timing.calculate_speech_rate = true;
        config.rhythm.prefer_regular_rhythm = true;

        config
    }

    /// Get configuration from preset
    pub fn from_preset(preset: ProsodicAnalysisPreset) -> Self {
        match preset {
            ProsodicAnalysisPreset::Minimal => Self::minimal(),
            ProsodicAnalysisPreset::Comprehensive => Self::comprehensive(),
            ProsodicAnalysisPreset::SpeechSynthesis => Self::speech_synthesis(),
            ProsodicAnalysisPreset::LanguageLearning => Self::language_learning(),
            ProsodicAnalysisPreset::LinguisticResearch => Self::linguistic_research(),
            ProsodicAnalysisPreset::ReadingFluency => Self::reading_fluency(),
        }
    }

    /// Validate configuration settings
    pub fn validate(&self) -> Result<(), String> {
        // Validate weights sum to reasonable values
        let total_weight = self.rhythm.rhythm_weight
            + self.stress.stress_weight
            + self.intonation.intonation_weight
            + self.timing.pause_weight;

        if total_weight <= 0.0 {
            return Err("Sum of component weights must be positive".to_string());
        }

        // Validate thresholds are in valid ranges
        if self.stress.metrical_consistency_threshold < 0.0
            || self.stress.metrical_consistency_threshold > 1.0
        {
            return Err("Metrical consistency threshold must be between 0.0 and 1.0".to_string());
        }

        if self.phonological.sound_similarity_threshold < 0.0
            || self.phonological.sound_similarity_threshold > 1.0
        {
            return Err("Sound similarity threshold must be between 0.0 and 1.0".to_string());
        }

        // Validate cache size
        if self.general.cache_size_limit == 0 && self.general.use_caching {
            return Err("Cache size limit must be positive when caching is enabled".to_string());
        }

        Ok(())
    }

    /// Generate cache key for configuration
    pub fn cache_key(&self) -> u64 {
        let mut hasher = DefaultHasher::new();

        // Hash key configuration parameters
        self.general.max_text_length.hash(&mut hasher);
        self.rhythm.enabled.hash(&mut hasher);
        self.stress.enabled.hash(&mut hasher);
        self.intonation.enabled.hash(&mut hasher);
        self.timing.enabled.hash(&mut hasher);
        self.phonological.enabled.hash(&mut hasher);
        self.advanced.enabled.hash(&mut hasher);

        hasher.finish()
    }
}

impl Default for ProsodicConfig {
    fn default() -> Self {
        Self::comprehensive()
    }
}

impl Hash for ProsodicConfig {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash essential configuration parameters
        self.general.enabled.hash(state);
        self.rhythm.enabled.hash(state);
        self.stress.enabled.hash(state);
        self.intonation.enabled.hash(state);
        self.timing.enabled.hash(state);
        self.phonological.enabled.hash(state);
        self.advanced.enabled.hash(state);

        // Hash weights (converted to avoid float precision issues)
        ((self.rhythm.rhythm_weight * 1000.0) as i32).hash(state);
        ((self.stress.stress_weight * 1000.0) as i32).hash(state);
        ((self.intonation.intonation_weight * 1000.0) as i32).hash(state);
        ((self.timing.pause_weight * 1000.0) as i32).hash(state);
    }
}

// Builder pattern support for configuration
pub struct ProsodicConfigBuilder {
    config: ProsodicConfig,
}

impl ProsodicConfigBuilder {
    /// Create new builder with default configuration
    pub fn new() -> Self {
        Self {
            config: ProsodicConfig::default(),
        }
    }

    /// Create builder from preset
    pub fn from_preset(preset: ProsodicAnalysisPreset) -> Self {
        Self {
            config: ProsodicConfig::from_preset(preset),
        }
    }

    /// Enable/disable rhythm analysis
    pub fn rhythm_analysis(mut self, enabled: bool) -> Self {
        self.config.rhythm.enabled = enabled;
        self
    }

    /// Set rhythm weight
    pub fn rhythm_weight(mut self, weight: f64) -> Self {
        self.config.rhythm.rhythm_weight = weight;
        self
    }

    /// Enable/disable stress analysis
    pub fn stress_analysis(mut self, enabled: bool) -> Self {
        self.config.stress.enabled = enabled;
        self
    }

    /// Set stress weight
    pub fn stress_weight(mut self, weight: f64) -> Self {
        self.config.stress.stress_weight = weight;
        self
    }

    /// Enable/disable intonation analysis
    pub fn intonation_analysis(mut self, enabled: bool) -> Self {
        self.config.intonation.enabled = enabled;
        self
    }

    /// Set intonation weight
    pub fn intonation_weight(mut self, weight: f64) -> Self {
        self.config.intonation.intonation_weight = weight;
        self
    }

    /// Enable/disable timing analysis
    pub fn timing_analysis(mut self, enabled: bool) -> Self {
        self.config.timing.enabled = enabled;
        self
    }

    /// Set pause weight
    pub fn pause_weight(mut self, weight: f64) -> Self {
        self.config.timing.pause_weight = weight;
        self
    }

    /// Enable/disable phonological analysis
    pub fn phonological_analysis(mut self, enabled: bool) -> Self {
        self.config.phonological.enabled = enabled;
        self
    }

    /// Enable/disable advanced analysis
    pub fn advanced_analysis(mut self, enabled: bool) -> Self {
        self.config.advanced.enabled = enabled;
        self
    }

    /// Set cache size limit
    pub fn cache_size(mut self, size: usize) -> Self {
        self.config.general.cache_size_limit = size;
        self
    }

    /// Build the configuration
    pub fn build(self) -> Result<ProsodicConfig, String> {
        self.config.validate()?;
        Ok(self.config)
    }
}

impl Default for ProsodicConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}
