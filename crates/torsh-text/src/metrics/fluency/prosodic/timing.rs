//! Timing Analysis Module
//!
//! This module provides comprehensive timing analysis for prosodic fluency,
//! including pause placement evaluation, tempo analysis, duration patterns,
//! speech rate calculation, and timing variability assessment with sophisticated
//! prosodic break detection and syllable timing analysis.

use super::config::TimingAnalysisConfig;
use super::results::{
    BreakType, ChangeDirection, ConsistencyMetrics, DurationAnalysis, DurationPattern,
    IsochronyClass, IsochronyMeasures, LengtheningEffect, LengtheningType, ProsodicBreak,
    SpeechRateAnalysis, SpeechRateChange, SpeechRateClass, SyllableTimingAnalysis,
    SyllableTimingPattern, TempoAnalysis, TempoChange, TempoClass, TimingMetrics,
    TimingVariabilityAnalysis, VariabilityPattern,
};
use crate::error::TextAnalysisError;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum TimingAnalysisError {
    #[error("Invalid timing analysis configuration: {0}")]
    ConfigError(String),
    #[error("Timing calculation failed: {0}")]
    CalculationError(String),
    #[error("Pause analysis error: {0}")]
    PauseError(String),
    #[error("Tempo analysis error: {0}")]
    TempoError(String),
}

pub type TimingResult<T> = Result<T, TimingAnalysisError>;

/// Timing information for syllables and segments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingInfo {
    /// Syllable durations (milliseconds)
    pub syllable_durations: Vec<f64>,
    /// Pause durations between syllables
    pub pause_durations: Vec<f64>,
    /// Speaking rate (syllables per second)
    pub speaking_rate: f64,
    /// Overall timing confidence
    pub confidence: f64,
    /// Timing context information
    pub context: String,
}

impl TimingInfo {
    /// Create new timing information
    pub fn new(syllable_durations: Vec<f64>, speaking_rate: f64) -> Self {
        let pause_durations = vec![0.0; syllable_durations.len().saturating_sub(1)];

        Self {
            syllable_durations,
            pause_durations,
            speaking_rate,
            confidence: 0.8,
            context: String::new(),
        }
    }

    /// Calculate total duration
    pub fn total_duration(&self) -> f64 {
        self.syllable_durations.iter().sum::<f64>() + self.pause_durations.iter().sum::<f64>()
    }

    /// Calculate average syllable duration
    pub fn average_syllable_duration(&self) -> f64 {
        if self.syllable_durations.is_empty() {
            0.0
        } else {
            self.syllable_durations.iter().sum::<f64>() / self.syllable_durations.len() as f64
        }
    }

    /// Calculate speech rate (syllables per second)
    pub fn calculate_speech_rate(&self) -> f64 {
        let total_time = self.total_duration() / 1000.0; // Convert to seconds
        if total_time > 0.0 {
            self.syllable_durations.len() as f64 / total_time
        } else {
            0.0
        }
    }
}

/// Pause placement analyzer for evaluating prosodic breaks
#[derive(Debug, Clone)]
pub struct PausePlacementAnalyzer {
    config: TimingAnalysisConfig,
    pause_detection_params: PauseDetectionParams,
    break_classification_params: BreakClassificationParams,
}

#[derive(Debug, Clone)]
pub struct PauseDetectionParams {
    /// Minimum pause duration threshold (ms)
    pub min_pause_duration: f64,
    /// Silence threshold for pause detection
    pub silence_threshold: f64,
    /// Context window for pause analysis
    pub context_window: usize,
    /// Breathing pause detection sensitivity
    pub breathing_sensitivity: f64,
}

#[derive(Debug, Clone)]
pub struct BreakClassificationParams {
    /// Classification thresholds for break types
    pub minor_break_threshold: f64,
    pub major_break_threshold: f64,
    pub boundary_break_threshold: f64,
    /// Strength calculation weights
    pub duration_weight: f64,
    pub context_weight: f64,
    pub syntactic_weight: f64,
}

impl Default for PauseDetectionParams {
    fn default() -> Self {
        Self {
            min_pause_duration: 50.0,
            silence_threshold: 0.1,
            context_window: 3,
            breathing_sensitivity: 0.7,
        }
    }
}

impl Default for BreakClassificationParams {
    fn default() -> Self {
        Self {
            minor_break_threshold: 100.0,
            major_break_threshold: 200.0,
            boundary_break_threshold: 400.0,
            duration_weight: 0.4,
            context_weight: 0.3,
            syntactic_weight: 0.3,
        }
    }
}

impl PausePlacementAnalyzer {
    /// Create new pause placement analyzer
    pub fn new(config: TimingAnalysisConfig) -> Self {
        Self {
            config,
            pause_detection_params: PauseDetectionParams::default(),
            break_classification_params: BreakClassificationParams::default(),
        }
    }

    /// Analyze pause placement in timing information
    pub fn analyze_pause_placement(
        &self,
        timing_info: &TimingInfo,
        syllables: &[String],
    ) -> TimingResult<Vec<ProsodicBreak>> {
        let mut prosodic_breaks = Vec::new();

        // Analyze explicit pauses
        for (i, &pause_duration) in timing_info.pause_durations.iter().enumerate() {
            if pause_duration >= self.pause_detection_params.min_pause_duration {
                let break_info =
                    self.analyze_pause_at_position(i, pause_duration, timing_info, syllables)?;
                prosodic_breaks.push(break_info);
            }
        }

        // Detect implicit breaks based on duration patterns
        let implicit_breaks = self.detect_implicit_breaks(timing_info, syllables)?;
        prosodic_breaks.extend(implicit_breaks);

        Ok(prosodic_breaks)
    }

    /// Analyze pause at specific position
    fn analyze_pause_at_position(
        &self,
        position: usize,
        duration: f64,
        timing_info: &TimingInfo,
        syllables: &[String],
    ) -> TimingResult<ProsodicBreak> {
        let break_type = self.classify_break_type(duration)?;
        let strength = self.calculate_break_strength(position, duration, timing_info, syllables)?;
        let context = self.build_break_context(position, syllables);

        Ok(ProsodicBreak {
            position,
            break_type,
            strength,
            duration: Some(duration),
            context,
        })
    }

    /// Classify break type based on duration
    fn classify_break_type(&self, duration: f64) -> TimingResult<BreakType> {
        if duration >= self.break_classification_params.boundary_break_threshold {
            Ok(BreakType::Boundary)
        } else if duration >= self.break_classification_params.major_break_threshold {
            Ok(BreakType::Major)
        } else if duration >= self.break_classification_params.minor_break_threshold {
            Ok(BreakType::Minor)
        } else {
            Ok(BreakType::Pause)
        }
    }

    /// Calculate break strength considering multiple factors
    fn calculate_break_strength(
        &self,
        position: usize,
        duration: f64,
        timing_info: &TimingInfo,
        syllables: &[String],
    ) -> TimingResult<f64> {
        // Duration component
        let duration_strength = (duration / 500.0).min(1.0); // Normalize to 500ms max

        // Context component (syllable boundary strength)
        let context_strength = self.calculate_context_strength(position, timing_info, syllables)?;

        // Syntactic component (simplified - would use actual parsing)
        let syntactic_strength = self.estimate_syntactic_strength(position, syllables);

        // Weighted combination
        let combined_strength = duration_strength
            * self.break_classification_params.duration_weight
            + context_strength * self.break_classification_params.context_weight
            + syntactic_strength * self.break_classification_params.syntactic_weight;

        Ok(combined_strength.max(0.0).min(1.0))
    }

    /// Calculate context strength based on surrounding syllables
    fn calculate_context_strength(
        &self,
        position: usize,
        timing_info: &TimingInfo,
        syllables: &[String],
    ) -> TimingResult<f64> {
        if position >= syllables.len() {
            return Ok(0.0);
        }

        let window_size = self.pause_detection_params.context_window;
        let start = position.saturating_sub(window_size);
        let end = (position + window_size + 1).min(syllables.len());

        // Check for natural break points (word boundaries, phrase boundaries)
        let mut context_score = 0.5; // Base score

        // Word boundary detection (simplified)
        if position > 0 && position < syllables.len() {
            let prev_syllable = &syllables[position - 1];
            let next_syllable = &syllables[position];

            // Heuristic: different word if significant syllable difference
            if self.likely_different_words(prev_syllable, next_syllable) {
                context_score += 0.3;
            }
        }

        // Phrase boundary indicators
        if self.likely_phrase_boundary(position, syllables) {
            context_score += 0.4;
        }

        Ok(context_score.min(1.0))
    }

    /// Heuristic to detect different words
    fn likely_different_words(&self, syl1: &str, syl2: &str) -> bool {
        // Simple heuristic based on syllable characteristics
        syl1.len() != syl2.len() || !syl1.chars().next().unwrap_or(' ').is_lowercase()
    }

    /// Heuristic to detect phrase boundaries
    fn likely_phrase_boundary(&self, position: usize, syllables: &[String]) -> bool {
        if position == 0 || position >= syllables.len() - 1 {
            return true;
        }

        // Check for function words that might indicate phrase boundaries
        const FUNCTION_WORDS: &[&str] = &[
            "the", "a", "an", "and", "or", "but", "of", "to", "for", "with",
        ];

        if position < syllables.len() {
            let syllable = syllables[position].to_lowercase();
            FUNCTION_WORDS.iter().any(|&fw| syllable.starts_with(fw))
        } else {
            false
        }
    }

    /// Estimate syntactic strength (simplified)
    fn estimate_syntactic_strength(&self, position: usize, syllables: &[String]) -> f64 {
        // Simplified syntactic boundary detection
        let relative_position = position as f64 / syllables.len() as f64;

        // Stronger breaks at clause/sentence boundaries (beginning and end)
        if relative_position < 0.1 || relative_position > 0.9 {
            0.8
        } else if relative_position > 0.4 && relative_position < 0.6 {
            0.7 // Mid-sentence phrase boundaries
        } else {
            0.4 // Word boundaries
        }
    }

    /// Build context description for break
    fn build_break_context(&self, position: usize, syllables: &[String]) -> String {
        let start = position.saturating_sub(2);
        let end = (position + 3).min(syllables.len());

        if start < end {
            syllables[start..end].join(" ")
        } else {
            "boundary".to_string()
        }
    }

    /// Detect implicit breaks based on duration patterns
    fn detect_implicit_breaks(
        &self,
        timing_info: &TimingInfo,
        syllables: &[String],
    ) -> TimingResult<Vec<ProsodicBreak>> {
        let mut implicit_breaks = Vec::new();

        if timing_info.syllable_durations.len() < 2 {
            return Ok(implicit_breaks);
        }

        // Look for unusually long syllables that might indicate implicit breaks
        let mean_duration = timing_info.average_syllable_duration();
        let threshold = mean_duration * 1.5; // 50% longer than average

        for (i, &duration) in timing_info.syllable_durations.iter().enumerate() {
            if duration >= threshold {
                // Check if this is a natural lengthening position
                if self.is_natural_lengthening_position(i, syllables) {
                    let break_strength =
                        self.calculate_implicit_break_strength(duration, mean_duration);

                    implicit_breaks.push(ProsodicBreak {
                        position: i,
                        break_type: BreakType::Minor,
                        strength: break_strength,
                        duration: Some(duration),
                        context: self.build_break_context(i, syllables),
                    });
                }
            }
        }

        Ok(implicit_breaks)
    }

    /// Check if position is natural for lengthening
    fn is_natural_lengthening_position(&self, position: usize, syllables: &[String]) -> bool {
        // Phrase-final lengthening is common
        let relative_position = position as f64 / syllables.len() as f64;
        relative_position > 0.8 || // Near end
        position == syllables.len() - 1 || // Final syllable
        self.likely_phrase_boundary(position, syllables) // Phrase boundary
    }

    /// Calculate implicit break strength
    fn calculate_implicit_break_strength(&self, duration: f64, mean_duration: f64) -> f64 {
        if mean_duration > 0.0 {
            let lengthening_ratio = duration / mean_duration;
            ((lengthening_ratio - 1.0) / 2.0).min(1.0).max(0.0)
        } else {
            0.0
        }
    }

    /// Calculate overall pause placement accuracy
    pub fn calculate_pause_accuracy(
        &self,
        breaks: &[ProsodicBreak],
        expected_breaks: &[usize],
    ) -> f64 {
        if expected_breaks.is_empty() {
            return if breaks.is_empty() { 1.0 } else { 0.5 };
        }

        let detected_positions: HashSet<usize> = breaks.iter().map(|b| b.position).collect();
        let expected_positions: HashSet<usize> = expected_breaks.iter().cloned().collect();

        let intersection_size = detected_positions.intersection(&expected_positions).count();
        let union_size = detected_positions.union(&expected_positions).count();

        if union_size > 0 {
            intersection_size as f64 / union_size as f64
        } else {
            1.0
        }
    }
}

/// Tempo analyzer for analyzing speech rate and tempo patterns
#[derive(Debug, Clone)]
pub struct TempoAnalyzer {
    config: TimingAnalysisConfig,
    tempo_analysis_params: TempoAnalysisParams,
    rate_classification_params: RateClassificationParams,
}

#[derive(Debug, Clone)]
pub struct TempoAnalysisParams {
    /// Window size for tempo analysis (number of syllables)
    pub analysis_window: usize,
    /// Minimum tempo change threshold (syllables/second)
    pub min_tempo_change: f64,
    /// Smoothing factor for tempo calculation
    pub smoothing_factor: f64,
    /// Tempo regularity analysis depth
    pub regularity_depth: usize,
}

#[derive(Debug, Clone)]
pub struct RateClassificationParams {
    /// Speech rate thresholds (syllables per second)
    pub very_slow_threshold: f64,
    pub slow_threshold: f64,
    pub normal_range: (f64, f64),
    pub fast_threshold: f64,
    pub very_fast_threshold: f64,
}

impl Default for TempoAnalysisParams {
    fn default() -> Self {
        Self {
            analysis_window: 5,
            min_tempo_change: 0.5,
            smoothing_factor: 0.3,
            regularity_depth: 10,
        }
    }
}

impl Default for RateClassificationParams {
    fn default() -> Self {
        Self {
            very_slow_threshold: 2.0,
            slow_threshold: 3.0,
            normal_range: (3.5, 5.5),
            fast_threshold: 6.0,
            very_fast_threshold: 7.5,
        }
    }
}

impl TempoAnalyzer {
    /// Create new tempo analyzer
    pub fn new(config: TimingAnalysisConfig) -> Self {
        Self {
            config,
            tempo_analysis_params: TempoAnalysisParams::default(),
            rate_classification_params: RateClassificationParams::default(),
        }
    }

    /// Analyze tempo patterns in timing information
    pub fn analyze_tempo(&self, timing_info: &TimingInfo) -> TimingResult<TempoAnalysis> {
        let local_tempos = self.calculate_local_tempos(timing_info)?;
        let average_tempo = self.calculate_average_tempo(&local_tempos);
        let tempo_variability = self.calculate_tempo_variability(&local_tempos);
        let tempo_changes = self.detect_tempo_changes(&local_tempos)?;
        let regularity = self.calculate_tempo_regularity(&local_tempos);
        let tempo_class = self.classify_tempo(average_tempo);

        Ok(TempoAnalysis {
            average_tempo,
            tempo_variability,
            tempo_changes,
            regularity,
            tempo_class,
        })
    }

    /// Calculate local tempos using sliding window
    fn calculate_local_tempos(&self, timing_info: &TimingInfo) -> TimingResult<Vec<f64>> {
        let window_size = self.tempo_analysis_params.analysis_window;
        let mut local_tempos = Vec::new();

        if timing_info.syllable_durations.len() < window_size {
            // Not enough data for windowed analysis
            let overall_tempo = timing_info.calculate_speech_rate();
            return Ok(vec![overall_tempo]);
        }

        for i in 0..=timing_info
            .syllable_durations
            .len()
            .saturating_sub(window_size)
        {
            let window_durations = &timing_info.syllable_durations[i..i + window_size];
            let window_total_ms = window_durations.iter().sum::<f64>();
            let window_total_s = window_total_ms / 1000.0;

            let local_tempo = if window_total_s > 0.0 {
                window_size as f64 / window_total_s
            } else {
                0.0
            };

            local_tempos.push(local_tempo);
        }

        // Apply smoothing
        Ok(self.smooth_tempo_sequence(&local_tempos))
    }

    /// Apply smoothing to tempo sequence
    fn smooth_tempo_sequence(&self, tempos: &[f64]) -> Vec<f64> {
        if tempos.len() < 2 {
            return tempos.to_vec();
        }

        let mut smoothed = Vec::with_capacity(tempos.len());
        let alpha = self.tempo_analysis_params.smoothing_factor;

        smoothed.push(tempos[0]);

        for i in 1..tempos.len() {
            let smoothed_value = alpha * tempos[i] + (1.0 - alpha) * smoothed[i - 1];
            smoothed.push(smoothed_value);
        }

        smoothed
    }

    /// Calculate average tempo
    fn calculate_average_tempo(&self, local_tempos: &[f64]) -> f64 {
        if local_tempos.is_empty() {
            0.0
        } else {
            local_tempos.iter().sum::<f64>() / local_tempos.len() as f64
        }
    }

    /// Calculate tempo variability
    fn calculate_tempo_variability(&self, local_tempos: &[f64]) -> f64 {
        if local_tempos.len() < 2 {
            return 0.0;
        }

        let mean = self.calculate_average_tempo(local_tempos);
        let variance = local_tempos
            .iter()
            .map(|&tempo| (tempo - mean).powi(2))
            .sum::<f64>()
            / local_tempos.len() as f64;

        variance.sqrt()
    }

    /// Detect tempo changes
    fn detect_tempo_changes(&self, local_tempos: &[f64]) -> TimingResult<Vec<TempoChange>> {
        let mut tempo_changes = Vec::new();
        let min_change = self.tempo_analysis_params.min_tempo_change;

        for i in 1..local_tempos.len() {
            let prev_tempo = local_tempos[i - 1];
            let curr_tempo = local_tempos[i];
            let change_magnitude = (curr_tempo - prev_tempo).abs();

            if change_magnitude >= min_change {
                let direction = if curr_tempo > prev_tempo {
                    ChangeDirection::Increase
                } else if curr_tempo < prev_tempo {
                    ChangeDirection::Decrease
                } else {
                    ChangeDirection::Stable
                };

                tempo_changes.push(TempoChange {
                    position: i,
                    magnitude: change_magnitude,
                    direction,
                    context: format!(
                        "tempo change from {:.1} to {:.1} syl/s",
                        prev_tempo, curr_tempo
                    ),
                });
            }
        }

        Ok(tempo_changes)
    }

    /// Calculate tempo regularity
    fn calculate_tempo_regularity(&self, local_tempos: &[f64]) -> f64 {
        if local_tempos.len() < 2 {
            return 1.0;
        }

        let variability = self.calculate_tempo_variability(local_tempos);
        let mean_tempo = self.calculate_average_tempo(local_tempos);

        if mean_tempo > 0.0 {
            let coefficient_of_variation = variability / mean_tempo;
            1.0 / (1.0 + coefficient_of_variation)
        } else {
            0.0
        }
    }

    /// Classify tempo based on speech rate
    fn classify_tempo(&self, average_tempo: f64) -> TempoClass {
        let params = &self.rate_classification_params;

        if average_tempo <= params.very_slow_threshold {
            TempoClass::VerySlow
        } else if average_tempo <= params.slow_threshold {
            TempoClass::Slow
        } else if average_tempo >= params.very_fast_threshold {
            TempoClass::VeryFast
        } else if average_tempo >= params.fast_threshold {
            TempoClass::Fast
        } else {
            TempoClass::Moderate
        }
    }
}

/// Duration analyzer for analyzing syllable duration patterns
#[derive(Debug, Clone)]
pub struct DurationAnalyzer {
    config: TimingAnalysisConfig,
    duration_analysis_params: DurationAnalysisParams,
}

#[derive(Debug, Clone)]
pub struct DurationAnalysisParams {
    /// Pattern detection window size
    pub pattern_window: usize,
    /// Minimum pattern strength threshold
    pub pattern_threshold: f64,
    /// Lengthening detection sensitivity
    pub lengthening_sensitivity: f64,
    /// Duration normalization parameters
    pub normalization_params: DurationNormalizationParams,
}

#[derive(Debug, Clone)]
pub struct DurationNormalizationParams {
    /// Minimum expected syllable duration (ms)
    pub min_syllable_duration: f64,
    /// Maximum expected syllable duration (ms)
    pub max_syllable_duration: f64,
    /// Normalization method
    pub normalization_method: NormalizationMethod,
}

#[derive(Debug, Clone)]
pub enum NormalizationMethod {
    ZScore,
    MinMax,
    Quantile,
}

impl Default for DurationAnalysisParams {
    fn default() -> Self {
        Self {
            pattern_window: 4,
            pattern_threshold: 0.6,
            lengthening_sensitivity: 0.3,
            normalization_params: DurationNormalizationParams {
                min_syllable_duration: 50.0,
                max_syllable_duration: 500.0,
                normalization_method: NormalizationMethod::ZScore,
            },
        }
    }
}

impl DurationAnalyzer {
    /// Create new duration analyzer
    pub fn new(config: TimingAnalysisConfig) -> Self {
        Self {
            config,
            duration_analysis_params: DurationAnalysisParams::default(),
        }
    }

    /// Analyze duration patterns
    pub fn analyze_duration_patterns(
        &self,
        timing_info: &TimingInfo,
    ) -> TimingResult<DurationAnalysis> {
        let normalized_durations = self.normalize_durations(&timing_info.syllable_durations)?;

        let average_duration = timing_info.average_syllable_duration();
        let duration_variability = self.calculate_duration_variability(&normalized_durations);
        let duration_patterns = self.detect_duration_patterns(&normalized_durations)?;
        let lengthening_effects =
            self.detect_lengthening_effects(&timing_info.syllable_durations)?;

        Ok(DurationAnalysis {
            average_syllable_duration: average_duration,
            duration_variability,
            duration_patterns,
            lengthening_effects,
        })
    }

    /// Normalize durations for analysis
    fn normalize_durations(&self, durations: &[f64]) -> TimingResult<Vec<f64>> {
        if durations.is_empty() {
            return Ok(Vec::new());
        }

        match self
            .duration_analysis_params
            .normalization_params
            .normalization_method
        {
            NormalizationMethod::ZScore => {
                let mean = durations.iter().sum::<f64>() / durations.len() as f64;
                let variance = durations.iter().map(|&d| (d - mean).powi(2)).sum::<f64>()
                    / durations.len() as f64;
                let std_dev = variance.sqrt();

                if std_dev > 0.0 {
                    Ok(durations.iter().map(|&d| (d - mean) / std_dev).collect())
                } else {
                    Ok(vec![0.0; durations.len()])
                }
            }
            NormalizationMethod::MinMax => {
                let min_duration = durations.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                let max_duration = durations.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let range = max_duration - min_duration;

                if range > 0.0 {
                    Ok(durations
                        .iter()
                        .map(|&d| (d - min_duration) / range)
                        .collect())
                } else {
                    Ok(vec![0.5; durations.len()])
                }
            }
            NormalizationMethod::Quantile => {
                // Simple quantile normalization (median-based)
                let mut sorted_durations = durations.to_vec();
                sorted_durations.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let median = if sorted_durations.len() % 2 == 0 {
                    let mid = sorted_durations.len() / 2;
                    (sorted_durations[mid - 1] + sorted_durations[mid]) / 2.0
                } else {
                    sorted_durations[sorted_durations.len() / 2]
                };

                if median > 0.0 {
                    Ok(durations.iter().map(|&d| d / median).collect())
                } else {
                    Ok(vec![1.0; durations.len()])
                }
            }
        }
    }

    /// Calculate duration variability
    fn calculate_duration_variability(&self, normalized_durations: &[f64]) -> f64 {
        if normalized_durations.len() < 2 {
            return 0.0;
        }

        let mean = normalized_durations.iter().sum::<f64>() / normalized_durations.len() as f64;
        let variance = normalized_durations
            .iter()
            .map(|&d| (d - mean).powi(2))
            .sum::<f64>()
            / normalized_durations.len() as f64;

        variance.sqrt()
    }

    /// Detect duration patterns
    fn detect_duration_patterns(
        &self,
        normalized_durations: &[f64],
    ) -> TimingResult<Vec<DurationPattern>> {
        let mut patterns = Vec::new();
        let window_size = self.duration_analysis_params.pattern_window;

        if normalized_durations.len() < window_size {
            return Ok(patterns);
        }

        // Sliding window pattern detection
        for i in 0..=normalized_durations.len().saturating_sub(window_size) {
            let window = &normalized_durations[i..i + window_size];

            if let Some(pattern) = self.analyze_duration_window(window, i)? {
                patterns.push(pattern);
            }
        }

        // Merge overlapping patterns
        Ok(self.merge_overlapping_patterns(patterns))
    }

    /// Analyze duration window for patterns
    fn analyze_duration_window(
        &self,
        window: &[f64],
        start_position: usize,
    ) -> TimingResult<Option<DurationPattern>> {
        // Check for different pattern types
        if let Some(pattern) = self.detect_alternating_pattern(window, start_position) {
            return Ok(Some(pattern));
        }

        if let Some(pattern) = self.detect_ascending_pattern(window, start_position) {
            return Ok(Some(pattern));
        }

        if let Some(pattern) = self.detect_descending_pattern(window, start_position) {
            return Ok(Some(pattern));
        }

        Ok(None)
    }

    /// Detect alternating duration pattern
    fn detect_alternating_pattern(
        &self,
        window: &[f64],
        start_position: usize,
    ) -> Option<DurationPattern> {
        if window.len() < 4 {
            return None;
        }

        let mut alternations = 0;
        let mut total_comparisons = 0;

        for i in 2..window.len() {
            let trend_prev = window[i - 1] > window[i - 2];
            let trend_curr = window[i] > window[i - 1];

            if trend_prev != trend_curr {
                alternations += 1;
            }
            total_comparisons += 1;
        }

        let alternation_ratio = alternations as f64 / total_comparisons as f64;

        if alternation_ratio >= self.duration_analysis_params.pattern_threshold {
            Some(DurationPattern {
                pattern_type: "alternating".to_string(),
                strength: alternation_ratio,
                positions: (start_position..start_position + window.len()).collect(),
                regularity: self.calculate_pattern_regularity(window),
            })
        } else {
            None
        }
    }

    /// Detect ascending duration pattern
    fn detect_ascending_pattern(
        &self,
        window: &[f64],
        start_position: usize,
    ) -> Option<DurationPattern> {
        let ascending_count = window.windows(2).filter(|pair| pair[1] > pair[0]).count();

        let ascending_ratio = ascending_count as f64 / (window.len() - 1) as f64;

        if ascending_ratio >= self.duration_analysis_params.pattern_threshold {
            Some(DurationPattern {
                pattern_type: "ascending".to_string(),
                strength: ascending_ratio,
                positions: (start_position..start_position + window.len()).collect(),
                regularity: self.calculate_pattern_regularity(window),
            })
        } else {
            None
        }
    }

    /// Detect descending duration pattern
    fn detect_descending_pattern(
        &self,
        window: &[f64],
        start_position: usize,
    ) -> Option<DurationPattern> {
        let descending_count = window.windows(2).filter(|pair| pair[1] < pair[0]).count();

        let descending_ratio = descending_count as f64 / (window.len() - 1) as f64;

        if descending_ratio >= self.duration_analysis_params.pattern_threshold {
            Some(DurationPattern {
                pattern_type: "descending".to_string(),
                strength: descending_ratio,
                positions: (start_position..start_position + window.len()).collect(),
                regularity: self.calculate_pattern_regularity(window),
            })
        } else {
            None
        }
    }

    /// Calculate pattern regularity
    fn calculate_pattern_regularity(&self, window: &[f64]) -> f64 {
        if window.len() < 2 {
            return 1.0;
        }

        let differences: Vec<f64> = window
            .windows(2)
            .map(|pair| (pair[1] - pair[0]).abs())
            .collect();

        let mean_diff = differences.iter().sum::<f64>() / differences.len() as f64;
        let diff_variance = differences
            .iter()
            .map(|&d| (d - mean_diff).powi(2))
            .sum::<f64>()
            / differences.len() as f64;

        if mean_diff > 0.0 {
            1.0 / (1.0 + diff_variance.sqrt() / mean_diff)
        } else {
            1.0
        }
    }

    /// Merge overlapping patterns
    fn merge_overlapping_patterns(&self, patterns: Vec<DurationPattern>) -> Vec<DurationPattern> {
        // Simple non-overlapping selection - would be enhanced for production
        let mut merged = Vec::new();
        let mut last_end_position = 0;

        for pattern in patterns {
            if let Some(&first_pos) = pattern.positions.first() {
                if first_pos >= last_end_position {
                    if let Some(&last_pos) = pattern.positions.last() {
                        last_end_position = last_pos + 1;
                        merged.push(pattern);
                    }
                }
            }
        }

        merged
    }

    /// Detect lengthening effects
    fn detect_lengthening_effects(
        &self,
        durations: &[f64],
    ) -> TimingResult<Vec<LengtheningEffect>> {
        let mut lengthening_effects = Vec::new();

        if durations.is_empty() {
            return Ok(lengthening_effects);
        }

        let mean_duration = durations.iter().sum::<f64>() / durations.len() as f64;
        let lengthening_threshold =
            mean_duration * (1.0 + self.duration_analysis_params.lengthening_sensitivity);

        for (i, &duration) in durations.iter().enumerate() {
            if duration >= lengthening_threshold {
                let lengthening_factor = duration / mean_duration;
                let lengthening_type = self.classify_lengthening_type(i, durations.len());
                let cause = self.determine_lengthening_cause(i, durations);

                lengthening_effects.push(LengtheningEffect {
                    position: i,
                    factor: lengthening_factor,
                    lengthening_type,
                    cause,
                });
            }
        }

        Ok(lengthening_effects)
    }

    /// Classify type of lengthening
    fn classify_lengthening_type(&self, position: usize, total_length: usize) -> LengtheningType {
        let relative_position = position as f64 / total_length as f64;

        if relative_position > 0.8 {
            LengtheningType::PhraseFinal
        } else if relative_position > 0.6 {
            LengtheningType::PreBoundary
        } else {
            // Check context for stress or focus (simplified)
            LengtheningType::Stress
        }
    }

    /// Determine cause of lengthening
    fn determine_lengthening_cause(&self, position: usize, durations: &[f64]) -> String {
        let relative_position = position as f64 / durations.len() as f64;

        if relative_position > 0.9 {
            "phrase-final lengthening".to_string()
        } else if relative_position < 0.1 {
            "phrase-initial emphasis".to_string()
        } else {
            "stress-induced lengthening".to_string()
        }
    }
}

/// Main timing analyzer orchestrating all timing components
#[derive(Debug, Clone)]
pub struct TimingAnalyzer {
    config: TimingAnalysisConfig,
    pause_analyzer: PausePlacementAnalyzer,
    tempo_analyzer: TempoAnalyzer,
    duration_analyzer: DurationAnalyzer,
    analysis_cache: HashMap<u64, TimingMetrics>,
}

impl TimingAnalyzer {
    /// Create new timing analyzer
    pub fn new(config: TimingAnalysisConfig) -> TimingResult<Self> {
        Self::validate_config(&config)?;

        let pause_analyzer = PausePlacementAnalyzer::new(config.clone());
        let tempo_analyzer = TempoAnalyzer::new(config.clone());
        let duration_analyzer = DurationAnalyzer::new(config.clone());

        Ok(Self {
            config,
            pause_analyzer,
            tempo_analyzer,
            duration_analyzer,
            analysis_cache: HashMap::new(),
        })
    }

    /// Analyze timing patterns in text
    pub fn analyze_timing(
        &mut self,
        sentences: &[String],
        timing_data: Option<&TimingInfo>,
    ) -> TimingResult<TimingMetrics> {
        let cache_key = self.generate_cache_key(sentences, timing_data);
        if let Some(cached) = self.analysis_cache.get(&cache_key) {
            return Ok(cached.clone());
        }

        // Extract or simulate timing information
        let timing_info = if let Some(timing) = timing_data {
            timing.clone()
        } else {
            self.simulate_timing_info(sentences)?
        };

        // Overall timing score
        let overall_score = self.calculate_overall_timing_score(&timing_info, sentences)?;

        // Pause placement analysis
        let prosodic_breaks = self
            .pause_analyzer
            .analyze_pause_placement(&timing_info, &self.extract_syllables(sentences))?;
        let expected_breaks = self.estimate_expected_breaks(sentences);
        let pause_accuracy = self
            .pause_analyzer
            .calculate_pause_accuracy(&prosodic_breaks, &expected_breaks);

        // Tempo analysis
        let tempo_analysis = if self.config.enable_tempo_analysis {
            Some(self.tempo_analyzer.analyze_tempo(&timing_info)?)
        } else {
            None
        };

        // Duration analysis
        let duration_analysis = if self.config.enable_duration_analysis {
            Some(
                self.duration_analyzer
                    .analyze_duration_patterns(&timing_info)?,
            )
        } else {
            None
        };

        // Speech rate analysis
        let speech_rate_analysis = if self.config.calculate_speech_rate {
            Some(self.analyze_speech_rate(&timing_info, sentences)?)
        } else {
            None
        };

        // Timing variability analysis
        let variability_analysis = if self.config.analyze_timing_variability {
            Some(self.analyze_timing_variability(&timing_info)?)
        } else {
            None
        };

        // Syllable timing analysis
        let syllable_timing = if self.config.analyze_syllable_timing {
            Some(self.analyze_syllable_timing(&timing_info, sentences)?)
        } else {
            None
        };

        let metrics = TimingMetrics {
            overall_timing_score: overall_score,
            pause_accuracy,
            tempo_analysis,
            duration_analysis,
            speech_rate_analysis,
            variability_analysis,
            break_analysis: prosodic_breaks,
            syllable_timing,
        };

        // Cache results
        self.analysis_cache.insert(cache_key, metrics.clone());

        Ok(metrics)
    }

    // Helper methods for timing analysis

    fn validate_config(config: &TimingAnalysisConfig) -> TimingResult<()> {
        if config.pause_weight < 0.0 {
            return Err(TimingAnalysisError::ConfigError(
                "Pause weight must be non-negative".to_string(),
            ));
        }

        if config.pause_accuracy_threshold < 0.0 || config.pause_accuracy_threshold > 1.0 {
            return Err(TimingAnalysisError::ConfigError(
                "Pause accuracy threshold must be between 0.0 and 1.0".to_string(),
            ));
        }

        Ok(())
    }

    fn generate_cache_key(&self, sentences: &[String], timing_data: Option<&TimingInfo>) -> u64 {
        let mut hasher = DefaultHasher::new();

        for sentence in sentences {
            sentence.hash(&mut hasher);
        }

        if let Some(timing) = timing_data {
            for &duration in &timing.syllable_durations {
                (duration as i64).hash(&mut hasher);
            }
            (timing.speaking_rate as i64).hash(&mut hasher);
        }

        self.config.enabled.hash(&mut hasher);
        hasher.finish()
    }

    fn simulate_timing_info(&self, sentences: &[String]) -> TimingResult<TimingInfo> {
        let syllables = self.extract_syllables(sentences);
        let mut syllable_durations = Vec::new();

        // Simple duration simulation based on syllable characteristics
        for syllable in &syllables {
            let base_duration = 120.0; // ms
            let length_factor = (syllable.len() as f64).min(4.0) / 4.0;
            let vowel_count = syllable
                .chars()
                .filter(|c| "aeiouAEIOU".contains(*c))
                .count();
            let vowel_factor = 1.0 + (vowel_count as f64 * 0.1);

            let duration = base_duration * (0.8 + length_factor * 0.4) * vowel_factor;
            syllable_durations.push(duration);
        }

        let speaking_rate = if syllables.len() > 0 {
            let total_duration_s = syllable_durations.iter().sum::<f64>() / 1000.0;
            syllables.len() as f64 / total_duration_s
        } else {
            4.0 // Default rate
        };

        Ok(TimingInfo::new(syllable_durations, speaking_rate))
    }

    fn extract_syllables(&self, sentences: &[String]) -> Vec<String> {
        let mut syllables = Vec::new();

        for sentence in sentences {
            let words: Vec<&str> = sentence.split_whitespace().collect();
            for word in words {
                let word_syllables = self.extract_word_syllables(word);
                syllables.extend(word_syllables);
            }
        }

        syllables
    }

    fn extract_word_syllables(&self, word: &str) -> Vec<String> {
        // Simple syllable extraction
        let vowels = "aeiouAEIOU";
        let mut syllables = Vec::new();
        let chars: Vec<char> = word.chars().collect();

        let mut current_syllable = String::new();
        let mut has_vowel = false;

        for ch in chars {
            current_syllable.push(ch);

            if vowels.contains(ch) {
                if has_vowel {
                    continue; // Diphthong
                }
                has_vowel = true;
            } else if has_vowel {
                syllables.push(current_syllable.trim().to_string());
                current_syllable = String::new();
                has_vowel = false;
            }
        }

        if !current_syllable.is_empty() {
            syllables.push(current_syllable);
        }

        if syllables.is_empty() {
            vec![word.to_string()]
        } else {
            syllables
        }
    }

    fn calculate_overall_timing_score(
        &self,
        timing_info: &TimingInfo,
        sentences: &[String],
    ) -> TimingResult<f64> {
        let mut score_components = Vec::new();

        // Rate appropriateness
        let rate_score = self.evaluate_rate_appropriateness(timing_info.speaking_rate);
        score_components.push(rate_score);

        // Duration consistency
        if !timing_info.syllable_durations.is_empty() {
            let consistency_score =
                self.evaluate_duration_consistency(&timing_info.syllable_durations);
            score_components.push(consistency_score);
        }

        // Timing naturalness
        let naturalness_score = self.evaluate_timing_naturalness(timing_info, sentences)?;
        score_components.push(naturalness_score);

        Ok(score_components.iter().sum::<f64>() / score_components.len() as f64)
    }

    fn evaluate_rate_appropriateness(&self, speaking_rate: f64) -> f64 {
        // Optimal speaking rate is typically 3.5-5.5 syllables per second
        let optimal_range = (3.5, 5.5);

        if speaking_rate >= optimal_range.0 && speaking_rate <= optimal_range.1 {
            1.0
        } else if speaking_rate < optimal_range.0 {
            // Too slow
            (speaking_rate / optimal_range.0).min(1.0)
        } else {
            // Too fast
            (optimal_range.1 / speaking_rate).min(1.0)
        }
    }

    fn evaluate_duration_consistency(&self, durations: &[f64]) -> f64 {
        if durations.len() < 2 {
            return 1.0;
        }

        let mean_duration = durations.iter().sum::<f64>() / durations.len() as f64;
        let coefficient_of_variation = if mean_duration > 0.0 {
            let variance = durations
                .iter()
                .map(|&d| (d - mean_duration).powi(2))
                .sum::<f64>()
                / durations.len() as f64;
            variance.sqrt() / mean_duration
        } else {
            1.0
        };

        // Lower variability = higher consistency
        1.0 / (1.0 + coefficient_of_variation)
    }

    fn evaluate_timing_naturalness(
        &self,
        timing_info: &TimingInfo,
        sentences: &[String],
    ) -> TimingResult<f64> {
        // Simple naturalness evaluation based on expected patterns
        let mut naturalness_factors = Vec::new();

        // Final lengthening (common in natural speech)
        if !timing_info.syllable_durations.is_empty() {
            let final_duration = timing_info.syllable_durations.last().unwrap();
            let mean_duration = timing_info.average_syllable_duration();

            if mean_duration > 0.0 {
                let final_ratio = final_duration / mean_duration;
                // Final syllables are often 20-50% longer
                let final_lengthening_score = if final_ratio >= 1.2 && final_ratio <= 1.5 {
                    1.0
                } else {
                    0.7
                };
                naturalness_factors.push(final_lengthening_score);
            }
        }

        // Pause appropriateness
        let total_pause_time = timing_info.pause_durations.iter().sum::<f64>();
        let total_speech_time = timing_info.syllable_durations.iter().sum::<f64>();
        let pause_ratio = if total_speech_time > 0.0 {
            total_pause_time / total_speech_time
        } else {
            0.0
        };

        // Natural pause ratio is typically 10-30% of speech time
        let pause_score = if pause_ratio >= 0.1 && pause_ratio <= 0.3 {
            1.0
        } else if pause_ratio < 0.1 {
            pause_ratio / 0.1
        } else {
            0.3 / pause_ratio
        };
        naturalness_factors.push(pause_score);

        Ok(if naturalness_factors.is_empty() {
            0.5
        } else {
            naturalness_factors.iter().sum::<f64>() / naturalness_factors.len() as f64
        })
    }

    fn estimate_expected_breaks(&self, sentences: &[String]) -> Vec<usize> {
        let mut expected_breaks = Vec::new();
        let mut position = 0;

        for (i, sentence) in sentences.iter().enumerate() {
            let words = sentence.split_whitespace().count();

            // Major breaks at sentence boundaries
            if i > 0 {
                expected_breaks.push(position);
            }

            // Minor breaks within long sentences
            if words > 8 {
                let mid_position = position + words / 2;
                expected_breaks.push(mid_position);
            }

            position += words;
        }

        expected_breaks
    }

    fn analyze_speech_rate(
        &self,
        timing_info: &TimingInfo,
        sentences: &[String],
    ) -> TimingResult<SpeechRateAnalysis> {
        let words_per_minute = self.calculate_words_per_minute(timing_info, sentences);
        let syllables_per_second = timing_info.speaking_rate;
        let rate_variability = self.calculate_rate_variability(timing_info)?;
        let rate_changes = self.detect_rate_changes(timing_info)?;
        let rate_class = self.classify_speech_rate(syllables_per_second);

        Ok(SpeechRateAnalysis {
            words_per_minute,
            syllables_per_second,
            rate_variability,
            rate_changes,
            rate_class,
        })
    }

    fn calculate_words_per_minute(&self, timing_info: &TimingInfo, sentences: &[String]) -> f64 {
        let word_count = sentences
            .iter()
            .map(|s| s.split_whitespace().count())
            .sum::<usize>();

        let total_time_minutes = timing_info.total_duration() / 60000.0; // Convert ms to minutes

        if total_time_minutes > 0.0 {
            word_count as f64 / total_time_minutes
        } else {
            0.0
        }
    }

    fn calculate_rate_variability(&self, timing_info: &TimingInfo) -> TimingResult<f64> {
        // Calculate local rates and their variability
        let window_size = 5;
        let mut local_rates = Vec::new();

        for i in 0..=timing_info
            .syllable_durations
            .len()
            .saturating_sub(window_size)
        {
            let window_durations = &timing_info.syllable_durations[i..i + window_size];
            let window_time = window_durations.iter().sum::<f64>() / 1000.0; // Convert to seconds

            if window_time > 0.0 {
                let rate = window_size as f64 / window_time;
                local_rates.push(rate);
            }
        }

        if local_rates.len() < 2 {
            return Ok(0.0);
        }

        let mean_rate = local_rates.iter().sum::<f64>() / local_rates.len() as f64;
        let variance = local_rates
            .iter()
            .map(|&rate| (rate - mean_rate).powi(2))
            .sum::<f64>()
            / local_rates.len() as f64;

        Ok(variance.sqrt())
    }

    fn detect_rate_changes(&self, timing_info: &TimingInfo) -> TimingResult<Vec<SpeechRateChange>> {
        // Simplified rate change detection
        let mut rate_changes = Vec::new();

        // Would implement sliding window rate calculation and change detection
        // For now, return empty vector as placeholder

        Ok(rate_changes)
    }

    fn classify_speech_rate(&self, rate: f64) -> SpeechRateClass {
        if rate < 2.5 {
            SpeechRateClass::VerySlow
        } else if rate < 3.5 {
            SpeechRateClass::Slow
        } else if rate < 5.5 {
            SpeechRateClass::Normal
        } else if rate < 7.0 {
            SpeechRateClass::Fast
        } else {
            SpeechRateClass::VeryFast
        }
    }

    fn analyze_timing_variability(
        &self,
        timing_info: &TimingInfo,
    ) -> TimingResult<TimingVariabilityAnalysis> {
        let overall_variability = self.calculate_overall_variability(timing_info)?;
        let component_variability = self.calculate_component_variability(timing_info)?;
        let variability_patterns = self.detect_variability_patterns(timing_info)?;
        let consistency_metrics = self.calculate_consistency_metrics(timing_info)?;

        Ok(TimingVariabilityAnalysis {
            overall_variability,
            component_variability,
            variability_patterns,
            consistency_metrics,
        })
    }

    fn calculate_overall_variability(&self, timing_info: &TimingInfo) -> TimingResult<f64> {
        self.evaluate_duration_consistency(&timing_info.syllable_durations);
        Ok(1.0 - self.evaluate_duration_consistency(&timing_info.syllable_durations))
    }

    fn calculate_component_variability(
        &self,
        timing_info: &TimingInfo,
    ) -> TimingResult<HashMap<String, f64>> {
        let mut variability = HashMap::new();

        variability.insert(
            "syllable_duration".to_string(),
            1.0 - self.evaluate_duration_consistency(&timing_info.syllable_durations),
        );

        if !timing_info.pause_durations.is_empty() {
            variability.insert(
                "pause_duration".to_string(),
                1.0 - self.evaluate_duration_consistency(&timing_info.pause_durations),
            );
        }

        Ok(variability)
    }

    fn detect_variability_patterns(
        &self,
        timing_info: &TimingInfo,
    ) -> TimingResult<Vec<VariabilityPattern>> {
        // Placeholder for variability pattern detection
        Ok(Vec::new())
    }

    fn calculate_consistency_metrics(
        &self,
        timing_info: &TimingInfo,
    ) -> TimingResult<ConsistencyMetrics> {
        let overall_consistency =
            self.evaluate_duration_consistency(&timing_info.syllable_durations);

        Ok(ConsistencyMetrics {
            beat_consistency: overall_consistency,
            rhythm_consistency: overall_consistency,
            pause_consistency: if timing_info.pause_durations.is_empty() {
                1.0
            } else {
                self.evaluate_duration_consistency(&timing_info.pause_durations)
            },
            overall_consistency,
        })
    }

    fn analyze_syllable_timing(
        &self,
        timing_info: &TimingInfo,
        sentences: &[String],
    ) -> TimingResult<SyllableTimingAnalysis> {
        let average_duration = timing_info.average_syllable_duration();
        let timing_patterns = self.detect_syllable_timing_patterns(timing_info)?;
        let timing_regularity = self.evaluate_duration_consistency(&timing_info.syllable_durations);
        let isochrony_measures = self.analyze_isochrony(timing_info)?;

        Ok(SyllableTimingAnalysis {
            average_duration,
            timing_patterns,
            timing_regularity,
            isochrony_measures,
        })
    }

    fn detect_syllable_timing_patterns(
        &self,
        timing_info: &TimingInfo,
    ) -> TimingResult<Vec<SyllableTimingPattern>> {
        // Placeholder for syllable timing pattern detection
        Ok(Vec::new())
    }

    fn analyze_isochrony(&self, timing_info: &TimingInfo) -> TimingResult<IsochronyMeasures> {
        // Simplified isochrony analysis
        let regularity = self.evaluate_duration_consistency(&timing_info.syllable_durations);

        let classification = if regularity > 0.8 {
            IsochronyClass::StrongSyllableTimed
        } else if regularity > 0.6 {
            IsochronyClass::WeakSyllableTimed
        } else {
            IsochronyClass::Mixed
        };

        Ok(IsochronyMeasures {
            stress_timed: regularity * 0.5, // Simplified
            syllable_timed: regularity,
            mora_timed: Some(regularity * 0.3),
            classification,
        })
    }
}

impl Default for TimingAnalyzer {
    fn default() -> Self {
        Self::new(TimingAnalysisConfig {
            enabled: true,
            pause_weight: 0.15,
            enable_break_detection: true,
            pause_accuracy_threshold: 0.7,
            enable_tempo_analysis: true,
            tempo_regularity_preference: 0.6,
            enable_duration_analysis: true,
            analyze_syllable_timing: true,
            calculate_speech_rate: true,
            analyze_timing_variability: true,
        })
        .unwrap()
    }
}
