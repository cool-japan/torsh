//! Intonation Analysis Module
//!
//! This module provides comprehensive intonation pattern analysis for prosodic fluency,
//! including pitch contour analysis, boundary tone detection, focus pattern recognition,
//! sentence type classification, and intonational phrase structure analysis.

use super::config::IntonationAnalysisConfig;
use super::results::{
    AcousticCorrelates, BoundaryTone, BoundaryType, CompressionTrend, ContourSegment, ContourShape,
    FocusPattern, HierarchyLevel, IntonationMetrics, IntonationalPhraseStructure, LocalPitchRange,
    PeakValleyAnalysis, PhraseHierarchy, PhraseType, PhraseUnit, PitchContourAnalysis, PitchPoint,
    PitchRangeAnalysis, RangeCompressionAnalysis, SentenceType, SentenceTypeClassification,
    ToneType, TrendDirection,
};
use crate::error::TextAnalysisError;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum IntonationAnalysisError {
    #[error("Invalid intonation analysis configuration: {0}")]
    ConfigError(String),
    #[error("Intonation calculation failed: {0}")]
    CalculationError(String),
    #[error("Contour analysis error: {0}")]
    ContourError(String),
    #[error("Pitch analysis error: {0}")]
    PitchError(String),
}

pub type IntonationResult<T> = Result<T, IntonationAnalysisError>;

/// Pitch contour representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PitchContour {
    /// Time points in the contour
    pub time_points: Vec<f64>,
    /// Fundamental frequency values (Hz)
    pub f0_values: Vec<f64>,
    /// Smoothed contour values
    pub smoothed_values: Vec<f64>,
    /// Contour confidence scores
    pub confidence: Vec<f64>,
    /// Syllable alignment information
    pub syllable_alignment: Vec<usize>,
}

impl PitchContour {
    /// Create new pitch contour
    pub fn new(time_points: Vec<f64>, f0_values: Vec<f64>) -> Self {
        let smoothed_values = Self::smooth_contour(&f0_values);
        let confidence = vec![0.8; f0_values.len()]; // Default confidence

        Self {
            time_points,
            f0_values,
            smoothed_values,
            confidence,
            syllable_alignment: Vec::new(),
        }
    }

    /// Apply smoothing to pitch contour
    fn smooth_contour(values: &[f64]) -> Vec<f64> {
        if values.len() < 3 {
            return values.to_vec();
        }

        let mut smoothed = Vec::with_capacity(values.len());

        // Simple moving average smoothing
        smoothed.push(values[0]);

        for i in 1..values.len() - 1 {
            let avg = (values[i - 1] + values[i] + values[i + 1]) / 3.0;
            smoothed.push(avg);
        }

        smoothed.push(values[values.len() - 1]);
        smoothed
    }

    /// Calculate pitch range
    pub fn calculate_range(&self) -> (f64, f64) {
        let valid_values: Vec<f64> = self
            .f0_values
            .iter()
            .filter(|&&x| x > 0.0 && x.is_finite())
            .copied()
            .collect();

        if valid_values.is_empty() {
            return (0.0, 0.0);
        }

        let min_f0 = valid_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_f0 = valid_values
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        (min_f0, max_f0)
    }

    /// Find peaks and valleys in contour
    pub fn find_peaks_valleys(&self, min_prominence: f64) -> PeakValleyAnalysis {
        let values = &self.smoothed_values;
        let mut peaks = Vec::new();
        let mut valleys = Vec::new();

        if values.len() < 3 {
            return PeakValleyAnalysis {
                peaks,
                valleys,
                peak_valley_ratio: 0.0,
                regularity: 0.0,
            };
        }

        // Find local maxima (peaks) and minima (valleys)
        for i in 1..values.len() - 1 {
            let prev = values[i - 1];
            let curr = values[i];
            let next = values[i + 1];

            if curr > prev && curr > next && curr - prev.min(next) >= min_prominence {
                peaks.push(PitchPoint {
                    position: i,
                    pitch: curr,
                    prominence: curr - prev.min(next),
                });
            } else if curr < prev && curr < next && prev.max(next) - curr >= min_prominence {
                valleys.push(PitchPoint {
                    position: i,
                    pitch: curr,
                    prominence: prev.max(next) - curr,
                });
            }
        }

        let peak_valley_ratio = if valleys.len() > 0 {
            peaks.len() as f64 / valleys.len() as f64
        } else {
            peaks.len() as f64
        };

        let regularity = self.calculate_peak_valley_regularity(&peaks, &valleys);

        PeakValleyAnalysis {
            peaks,
            valleys,
            peak_valley_ratio,
            regularity,
        }
    }

    /// Calculate regularity of peaks and valleys
    fn calculate_peak_valley_regularity(
        &self,
        peaks: &[PitchPoint],
        valleys: &[PitchPoint],
    ) -> f64 {
        if peaks.len() < 2 {
            return 1.0;
        }

        // Calculate intervals between peaks
        let peak_intervals: Vec<f64> = peaks
            .windows(2)
            .map(|window| (window[1].position - window[0].position) as f64)
            .collect();

        if peak_intervals.is_empty() {
            return 1.0;
        }

        let mean_interval = peak_intervals.iter().sum::<f64>() / peak_intervals.len() as f64;
        let variance = peak_intervals
            .iter()
            .map(|&x| (x - mean_interval).powi(2))
            .sum::<f64>()
            / peak_intervals.len() as f64;

        let coefficient_of_variation = if mean_interval > 0.0 {
            variance.sqrt() / mean_interval
        } else {
            1.0
        };

        1.0 / (1.0 + coefficient_of_variation)
    }

    /// Segment contour into meaningful parts
    pub fn segment_contour(&self, min_segment_length: usize) -> Vec<ContourSegment> {
        let mut segments = Vec::new();
        let values = &self.smoothed_values;

        if values.len() < min_segment_length {
            return segments;
        }

        let mut start = 0;
        let mut current_trend = self.determine_trend(0, min_segment_length.min(values.len()));

        for i in min_segment_length..values.len() {
            let window_end = (i + min_segment_length).min(values.len());
            let new_trend = self.determine_trend(i, window_end);

            if new_trend != current_trend {
                // End current segment
                let slope = self.calculate_slope(start, i, values);
                let length = (i - start) as f64;

                segments.push(ContourSegment {
                    boundaries: (start, i),
                    trend: current_trend,
                    slope,
                    length,
                });

                start = i;
                current_trend = new_trend;
            }
        }

        // Add final segment
        if start < values.len() {
            let slope = self.calculate_slope(start, values.len(), values);
            let length = (values.len() - start) as f64;

            segments.push(ContourSegment {
                boundaries: (start, values.len()),
                trend: current_trend,
                slope,
                length,
            });
        }

        segments
    }

    /// Determine trend direction in a window
    fn determine_trend(&self, start: usize, end: usize) -> TrendDirection {
        if end <= start + 1 {
            return TrendDirection::Level;
        }

        let values = &self.smoothed_values[start..end];
        let first = values[0];
        let last = values[values.len() - 1];
        let diff = last - first;
        let threshold = 5.0; // Hz threshold for trend detection

        if diff > threshold {
            TrendDirection::Rising
        } else if diff < -threshold {
            TrendDirection::Falling
        } else {
            TrendDirection::Level
        }
    }

    /// Calculate slope of contour segment
    fn calculate_slope(&self, start: usize, end: usize, values: &[f64]) -> f64 {
        if end <= start + 1 {
            return 0.0;
        }

        let segment = &values[start..end];
        let n = segment.len() as f64;
        let x_mean = (n - 1.0) / 2.0;
        let y_mean = segment.iter().sum::<f64>() / n;

        let numerator: f64 = segment
            .iter()
            .enumerate()
            .map(|(i, &y)| (i as f64 - x_mean) * (y - y_mean))
            .sum();

        let denominator: f64 = (0..segment.len())
            .map(|i| (i as f64 - x_mean).powi(2))
            .sum();

        if denominator > 0.0 {
            numerator / denominator
        } else {
            0.0
        }
    }
}

/// Boundary tone analyzer for detecting prosodic boundaries
#[derive(Debug, Clone)]
pub struct BoundaryToneAnalyzer {
    config: IntonationAnalysisConfig,
    tone_detection_params: ToneDetectionParams,
    boundary_detection_params: BoundaryDetectionParams,
}

#[derive(Debug, Clone)]
pub struct ToneDetectionParams {
    /// Minimum tone prominence (Hz)
    pub min_prominence: f64,
    /// Tone classification thresholds
    pub classification_thresholds: ToneClassificationThresholds,
    /// Context window for tone analysis
    pub context_window: usize,
}

#[derive(Debug, Clone)]
pub struct ToneClassificationThresholds {
    /// High tone threshold (relative to range)
    pub high_threshold: f64,
    /// Low tone threshold (relative to range)
    pub low_threshold: f64,
    /// Rising tone threshold (Hz/s)
    pub rising_threshold: f64,
    /// Falling tone threshold (Hz/s)
    pub falling_threshold: f64,
}

#[derive(Debug, Clone)]
pub struct BoundaryDetectionParams {
    /// Minimum boundary strength
    pub min_boundary_strength: f64,
    /// Boundary types to detect
    pub detect_phrase_boundaries: bool,
    pub detect_utterance_boundaries: bool,
    pub detect_intermediate_boundaries: bool,
}

impl Default for ToneDetectionParams {
    fn default() -> Self {
        Self {
            min_prominence: 10.0,
            classification_thresholds: ToneClassificationThresholds {
                high_threshold: 0.75,
                low_threshold: 0.25,
                rising_threshold: 50.0,
                falling_threshold: -50.0,
            },
            context_window: 3,
        }
    }
}

impl Default for BoundaryDetectionParams {
    fn default() -> Self {
        Self {
            min_boundary_strength: 0.6,
            detect_phrase_boundaries: true,
            detect_utterance_boundaries: true,
            detect_intermediate_boundaries: true,
        }
    }
}

impl BoundaryToneAnalyzer {
    /// Create new boundary tone analyzer
    pub fn new(config: IntonationAnalysisConfig) -> Self {
        Self {
            config,
            tone_detection_params: ToneDetectionParams::default(),
            boundary_detection_params: BoundaryDetectionParams::default(),
        }
    }

    /// Detect boundary tones in pitch contour
    pub fn detect_boundary_tones(
        &self,
        contour: &PitchContour,
        syllable_boundaries: &[usize],
    ) -> IntonationResult<Vec<BoundaryTone>> {
        let mut boundary_tones = Vec::new();

        for &boundary_pos in syllable_boundaries {
            if let Some(tone) = self.analyze_boundary_at_position(contour, boundary_pos)? {
                boundary_tones.push(tone);
            }
        }

        Ok(boundary_tones)
    }

    /// Analyze boundary tone at specific position
    fn analyze_boundary_at_position(
        &self,
        contour: &PitchContour,
        position: usize,
    ) -> IntonationResult<Option<BoundaryTone>> {
        if position >= contour.f0_values.len() {
            return Ok(None);
        }

        let boundary_type = self.classify_boundary_type(position, contour)?;
        let tone_type = self.classify_tone_type(position, contour)?;
        let strength = self.calculate_tone_strength(position, contour)?;

        if strength >= self.boundary_detection_params.min_boundary_strength {
            Ok(Some(BoundaryTone {
                position,
                boundary_type,
                tone_type,
                strength,
            }))
        } else {
            Ok(None)
        }
    }

    /// Classify boundary type
    fn classify_boundary_type(
        &self,
        position: usize,
        contour: &PitchContour,
    ) -> IntonationResult<BoundaryType> {
        // Simple heuristic classification based on position and context
        let relative_position = position as f64 / contour.f0_values.len() as f64;

        if relative_position < 0.1 || relative_position > 0.9 {
            Ok(BoundaryType::Utterance)
        } else {
            let boundary_strength = self.calculate_boundary_strength(position, contour)?;

            if boundary_strength > 0.8 {
                Ok(BoundaryType::Major)
            } else if boundary_strength > 0.6 {
                Ok(BoundaryType::Phrase)
            } else {
                Ok(BoundaryType::Intermediate)
            }
        }
    }

    /// Classify tone type
    fn classify_tone_type(
        &self,
        position: usize,
        contour: &PitchContour,
    ) -> IntonationResult<ToneType> {
        let (min_f0, max_f0) = contour.calculate_range();
        let pitch_range = max_f0 - min_f0;

        if pitch_range == 0.0 {
            return Ok(ToneType::Mid);
        }

        let current_pitch = contour.f0_values[position];
        let relative_pitch = (current_pitch - min_f0) / pitch_range;

        // Check for movement (rising/falling)
        let movement = self.calculate_pitch_movement(position, contour)?;

        if movement.abs()
            > self
                .tone_detection_params
                .classification_thresholds
                .rising_threshold
        {
            if movement > 0.0 {
                Ok(ToneType::Rising)
            } else {
                Ok(ToneType::Falling)
            }
        } else if relative_pitch
            >= self
                .tone_detection_params
                .classification_thresholds
                .high_threshold
        {
            Ok(ToneType::High)
        } else if relative_pitch
            <= self
                .tone_detection_params
                .classification_thresholds
                .low_threshold
        {
            Ok(ToneType::Low)
        } else {
            Ok(ToneType::Mid)
        }
    }

    /// Calculate boundary strength
    fn calculate_boundary_strength(
        &self,
        position: usize,
        contour: &PitchContour,
    ) -> IntonationResult<f64> {
        // Multiple factors contribute to boundary strength
        let pitch_discontinuity = self.calculate_pitch_discontinuity(position, contour)?;
        let duration_cues = 0.5; // Placeholder - would use actual duration data
        let pause_likelihood = self.estimate_pause_likelihood(position, contour)?;

        let combined_strength = (pitch_discontinuity + duration_cues + pause_likelihood) / 3.0;
        Ok(combined_strength.max(0.0).min(1.0))
    }

    /// Calculate pitch discontinuity at boundary
    fn calculate_pitch_discontinuity(
        &self,
        position: usize,
        contour: &PitchContour,
    ) -> IntonationResult<f64> {
        let window_size = self.tone_detection_params.context_window;
        let start = position.saturating_sub(window_size);
        let end = (position + window_size).min(contour.f0_values.len());

        if end <= start + 1 {
            return Ok(0.0);
        }

        let values = &contour.f0_values[start..end];
        let variance = self.calculate_variance(values);
        let (min_f0, max_f0) = contour.calculate_range();
        let pitch_range = max_f0 - min_f0;

        if pitch_range > 0.0 {
            Ok((variance.sqrt() / pitch_range).min(1.0))
        } else {
            Ok(0.0)
        }
    }

    /// Calculate variance of pitch values
    fn calculate_variance(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64
    }

    /// Calculate tone strength
    fn calculate_tone_strength(
        &self,
        position: usize,
        contour: &PitchContour,
    ) -> IntonationResult<f64> {
        let prominence = self.calculate_tonal_prominence(position, contour)?;
        let consistency = self.calculate_tonal_consistency(position, contour)?;
        let confidence = if position < contour.confidence.len() {
            contour.confidence[position]
        } else {
            0.5
        };

        Ok((prominence + consistency + confidence) / 3.0)
    }

    /// Calculate tonal prominence
    fn calculate_tonal_prominence(
        &self,
        position: usize,
        contour: &PitchContour,
    ) -> IntonationResult<f64> {
        let window_size = self.tone_detection_params.context_window;
        let start = position.saturating_sub(window_size);
        let end = (position + window_size + 1).min(contour.f0_values.len());

        let context = &contour.f0_values[start..end];
        let current_pitch = contour.f0_values[position];

        if context.is_empty() {
            return Ok(0.0);
        }

        let max_in_context = context.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let min_in_context = context.iter().fold(f64::INFINITY, |a, &b| a.min(b));

        if max_in_context > min_in_context {
            let relative_prominence =
                (current_pitch - min_in_context) / (max_in_context - min_in_context);
            Ok(relative_prominence)
        } else {
            Ok(0.5)
        }
    }

    /// Calculate tonal consistency
    fn calculate_tonal_consistency(
        &self,
        position: usize,
        contour: &PitchContour,
    ) -> IntonationResult<f64> {
        // Check how consistent the tone is with neighboring tones
        let window_size = 2;
        let start = position.saturating_sub(window_size);
        let end = (position + window_size + 1).min(contour.f0_values.len());

        if end <= start + 1 {
            return Ok(1.0);
        }

        let values = &contour.f0_values[start..end];
        let variance = self.calculate_variance(values);
        let mean = values.iter().sum::<f64>() / values.len() as f64;

        if mean > 0.0 {
            let coefficient_of_variation = variance.sqrt() / mean;
            Ok(1.0 / (1.0 + coefficient_of_variation))
        } else {
            Ok(0.5)
        }
    }

    /// Calculate pitch movement (derivative)
    fn calculate_pitch_movement(
        &self,
        position: usize,
        contour: &PitchContour,
    ) -> IntonationResult<f64> {
        if position == 0 || position >= contour.f0_values.len() - 1 {
            return Ok(0.0);
        }

        let prev_pitch = contour.f0_values[position - 1];
        let next_pitch = contour.f0_values[position + 1];
        let movement = (next_pitch - prev_pitch) / 2.0; // Central difference

        Ok(movement)
    }

    /// Estimate pause likelihood at position
    fn estimate_pause_likelihood(
        &self,
        position: usize,
        contour: &PitchContour,
    ) -> IntonationResult<f64> {
        // Simple heuristic based on pitch behavior
        if position == 0 || position >= contour.f0_values.len() - 1 {
            return Ok(0.8);
        }

        let current = contour.f0_values[position];
        let prev = contour.f0_values[position - 1];
        let next = contour.f0_values[position + 1];

        // Check for pitch drops or gaps that might indicate pauses
        let drop_before = prev - current;
        let drop_after = current - next;

        let pause_indicator = if drop_before > 20.0 || drop_after > 20.0 {
            0.8
        } else if drop_before > 10.0 || drop_after > 10.0 {
            0.6
        } else {
            0.3
        };

        Ok(pause_indicator)
    }
}

/// Focus pattern analyzer for detecting prominence and focus
#[derive(Debug, Clone)]
pub struct FocusPatternAnalyzer {
    config: IntonationAnalysisConfig,
    focus_detection_params: FocusDetectionParams,
}

#[derive(Debug, Clone)]
pub struct FocusDetectionParams {
    /// Minimum focus prominence threshold
    pub min_focus_prominence: f64,
    /// Focus detection sensitivity
    pub sensitivity: f64,
    /// Context window for focus analysis
    pub context_window: usize,
    /// Acoustic weight parameters
    pub acoustic_weights: AcousticWeights,
}

#[derive(Debug, Clone)]
pub struct AcousticWeights {
    /// Weight for F0 changes
    pub f0_weight: f64,
    /// Weight for duration changes
    pub duration_weight: f64,
    /// Weight for intensity changes
    pub intensity_weight: f64,
}

impl Default for FocusDetectionParams {
    fn default() -> Self {
        Self {
            min_focus_prominence: 0.7,
            sensitivity: 0.6,
            context_window: 5,
            acoustic_weights: AcousticWeights {
                f0_weight: 0.5,
                duration_weight: 0.3,
                intensity_weight: 0.2,
            },
        }
    }
}

impl FocusPatternAnalyzer {
    /// Create new focus pattern analyzer
    pub fn new(config: IntonationAnalysisConfig) -> Self {
        Self {
            config,
            focus_detection_params: FocusDetectionParams::default(),
        }
    }

    /// Detect focus patterns in pitch contour
    pub fn detect_focus_patterns(
        &self,
        contour: &PitchContour,
        syllable_info: &[(String, usize)],
    ) -> IntonationResult<Vec<FocusPattern>> {
        let mut focus_patterns = Vec::new();

        for (i, (syllable, &position)) in syllable_info.iter().enumerate() {
            if position < contour.f0_values.len() {
                if let Some(focus) =
                    self.analyze_focus_at_syllable(contour, position, i, syllable_info)?
                {
                    focus_patterns.push(focus);
                }
            }
        }

        Ok(focus_patterns)
    }

    /// Analyze focus at specific syllable
    fn analyze_focus_at_syllable(
        &self,
        contour: &PitchContour,
        position: usize,
        syllable_idx: usize,
        syllable_info: &[(String, usize)],
    ) -> IntonationResult<Option<FocusPattern>> {
        let prominence = self.calculate_focus_prominence(contour, position)?;

        if prominence >= self.focus_detection_params.min_focus_prominence {
            let focus_type =
                self.classify_focus_type(contour, position, syllable_idx, syllable_info)?;
            let scope = self.determine_focus_scope(contour, position, syllable_info)?;
            let acoustic_correlates = self.analyze_acoustic_correlates(contour, position)?;

            Ok(Some(FocusPattern {
                position: syllable_idx,
                focus_type,
                scope,
                acoustic_correlates,
            }))
        } else {
            Ok(None)
        }
    }

    /// Calculate focus prominence
    fn calculate_focus_prominence(
        &self,
        contour: &PitchContour,
        position: usize,
    ) -> IntonationResult<f64> {
        let window_size = self.focus_detection_params.context_window;
        let start = position.saturating_sub(window_size);
        let end = (position + window_size + 1).min(contour.f0_values.len());

        let context = &contour.f0_values[start..end];
        let current_pitch = contour.f0_values[position];

        if context.is_empty() {
            return Ok(0.0);
        }

        // Calculate relative prominence
        let context_mean = context.iter().sum::<f64>() / context.len() as f64;
        let context_max = context.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        if context_max > context_mean {
            let prominence = (current_pitch - context_mean) / (context_max - context_mean);
            Ok(prominence.max(0.0))
        } else {
            Ok(0.0)
        }
    }

    /// Classify focus type
    fn classify_focus_type(
        &self,
        contour: &PitchContour,
        position: usize,
        syllable_idx: usize,
        syllable_info: &[(String, usize)],
    ) -> IntonationResult<super::results::FocusType> {
        // Simple classification based on prominence pattern and position
        let prominence = self.calculate_focus_prominence(contour, position)?;

        // Check if it's contrastive (multiple high-prominence syllables)
        let high_prominence_count = self.count_high_prominence_syllables(contour, syllable_info)?;

        if high_prominence_count > 1 {
            Ok(super::results::FocusType::Contrastive)
        } else if prominence > 0.9 {
            Ok(super::results::FocusType::Emphatic)
        } else if syllable_idx < syllable_info.len() / 2 {
            Ok(super::results::FocusType::Information)
        } else {
            Ok(super::results::FocusType::Corrective)
        }
    }

    /// Count high prominence syllables
    fn count_high_prominence_syllables(
        &self,
        contour: &PitchContour,
        syllable_info: &[(String, usize)],
    ) -> IntonationResult<usize> {
        let mut count = 0;

        for (_, &position) in syllable_info {
            if position < contour.f0_values.len() {
                let prominence = self.calculate_focus_prominence(contour, position)?;
                if prominence >= self.focus_detection_params.min_focus_prominence {
                    count += 1;
                }
            }
        }

        Ok(count)
    }

    /// Determine scope of focus
    fn determine_focus_scope(
        &self,
        contour: &PitchContour,
        focus_position: usize,
        syllable_info: &[(String, usize)],
    ) -> IntonationResult<Vec<usize>> {
        let mut scope = vec![focus_position];

        // Find the syllable index for the focus position
        let focus_syllable_idx = syllable_info
            .iter()
            .position(|(_, &pos)| pos == focus_position)
            .unwrap_or(0);

        let focus_prominence = self.calculate_focus_prominence(contour, focus_position)?;
        let scope_threshold = focus_prominence * 0.6;

        // Extend scope to adjacent syllables with sufficient prominence
        for i in 1..=2 {
            // Check before
            if focus_syllable_idx >= i {
                let (_, &pos) = &syllable_info[focus_syllable_idx - i];
                if pos < contour.f0_values.len() {
                    let prominence = self.calculate_focus_prominence(contour, pos)?;
                    if prominence >= scope_threshold {
                        scope.push(focus_syllable_idx - i);
                    }
                }
            }

            // Check after
            if focus_syllable_idx + i < syllable_info.len() {
                let (_, &pos) = &syllable_info[focus_syllable_idx + i];
                if pos < contour.f0_values.len() {
                    let prominence = self.calculate_focus_prominence(contour, pos)?;
                    if prominence >= scope_threshold {
                        scope.push(focus_syllable_idx + i);
                    }
                }
            }
        }

        scope.sort();
        Ok(scope)
    }

    /// Analyze acoustic correlates of focus
    fn analyze_acoustic_correlates(
        &self,
        contour: &PitchContour,
        position: usize,
    ) -> IntonationResult<AcousticCorrelates> {
        // F0 changes
        let f0_changes = self.calculate_f0_changes(contour, position)?;

        // Duration changes (simplified - would use actual duration data)
        let duration_changes = vec![1.2]; // Placeholder for lengthening

        // Intensity changes (placeholder - would use actual intensity data)
        let intensity_changes = vec![1.1]; // Placeholder for intensity increase

        Ok(AcousticCorrelates {
            f0_changes,
            duration_changes,
            intensity_changes,
        })
    }

    /// Calculate F0 changes around focus
    fn calculate_f0_changes(
        &self,
        contour: &PitchContour,
        position: usize,
    ) -> IntonationResult<Vec<f64>> {
        let window = 2;
        let start = position.saturating_sub(window);
        let end = (position + window + 1).min(contour.f0_values.len());

        if end <= start {
            return Ok(vec![]);
        }

        let baseline = if start > 0 {
            contour.f0_values[start - 1]
        } else {
            contour.f0_values[start]
        };

        let changes: Vec<f64> = contour.f0_values[start..end]
            .iter()
            .map(|&f0| if baseline > 0.0 { f0 / baseline } else { 1.0 })
            .collect();

        Ok(changes)
    }
}

/// Main intonation analyzer orchestrating all intonation components
#[derive(Debug, Clone)]
pub struct IntonationAnalyzer {
    config: IntonationAnalysisConfig,
    boundary_analyzer: BoundaryToneAnalyzer,
    focus_analyzer: FocusPatternAnalyzer,
    analysis_cache: HashMap<u64, IntonationMetrics>,
}

impl IntonationAnalyzer {
    /// Create new intonation analyzer
    pub fn new(config: IntonationAnalysisConfig) -> IntonationResult<Self> {
        Self::validate_config(&config)?;

        let boundary_analyzer = BoundaryToneAnalyzer::new(config.clone());
        let focus_analyzer = FocusPatternAnalyzer::new(config.clone());

        Ok(Self {
            config,
            boundary_analyzer,
            focus_analyzer,
            analysis_cache: HashMap::new(),
        })
    }

    /// Analyze intonation patterns in text
    pub fn analyze_intonation(
        &mut self,
        sentences: &[String],
        pitch_data: Option<&[f64]>,
    ) -> IntonationResult<IntonationMetrics> {
        let cache_key = self.generate_cache_key(sentences, pitch_data);
        if let Some(cached) = self.analysis_cache.get(&cache_key) {
            return Ok(cached.clone());
        }

        // Extract or simulate pitch contour
        let contour = if let Some(pitch_values) = pitch_data {
            let time_points: Vec<f64> = (0..pitch_values.len()).map(|i| i as f64 * 0.01).collect();
            PitchContour::new(time_points, pitch_values.to_vec())
        } else {
            self.simulate_pitch_contour(sentences)?
        };

        let syllable_info = self.extract_syllable_info(sentences)?;
        let syllable_boundaries: Vec<usize> = syllable_info.iter().map(|(_, pos)| *pos).collect();

        // Overall appropriateness calculation
        let appropriateness = self.calculate_intonation_appropriateness(sentences, &contour)?;

        // Contour analysis
        let contour_analysis = if self.config.enable_pitch_contour {
            Some(self.analyze_pitch_contour(&contour)?)
        } else {
            None
        };

        // Boundary tone detection
        let boundary_tones = if self.config.enable_boundary_tone {
            Some(
                self.boundary_analyzer
                    .detect_boundary_tones(&contour, &syllable_boundaries)?,
            )
        } else {
            None
        };

        // Focus pattern detection
        let focus_patterns = if self.config.detect_focus_patterns {
            Some(
                self.focus_analyzer
                    .detect_focus_patterns(&contour, &syllable_info)?,
            )
        } else {
            None
        };

        // Sentence type classification
        let sentence_classifications = if self.config.classify_sentence_types {
            self.classify_sentences(sentences, &contour)?
        } else {
            Vec::new()
        };

        // Intonational phrase structure
        let phrase_structure = if self.config.detect_intonational_phrases {
            Some(self.analyze_phrase_structure(sentences, &contour)?)
        } else {
            None
        };

        // Pitch range analysis
        let pitch_range_analysis = if self.config.analyze_pitch_range {
            Some(self.analyze_pitch_range(&contour)?)
        } else {
            None
        };

        // Calculate overall score
        let overall_score = self.calculate_overall_intonation_score(
            appropriateness,
            &contour_analysis,
            &boundary_tones,
            &focus_patterns,
        );

        let metrics = IntonationMetrics {
            overall_intonation_score: overall_score,
            appropriateness,
            contour_analysis,
            boundary_tones,
            focus_patterns,
            sentence_classifications,
            phrase_structure,
            pitch_range_analysis,
        };

        // Cache results
        self.analysis_cache.insert(cache_key, metrics.clone());

        Ok(metrics)
    }

    // Helper methods for intonation analysis

    fn validate_config(config: &IntonationAnalysisConfig) -> IntonationResult<()> {
        if config.intonation_weight < 0.0 {
            return Err(IntonationAnalysisError::ConfigError(
                "Intonation weight must be non-negative".to_string(),
            ));
        }

        if config.contour_smoothness_preference < 0.0 || config.contour_smoothness_preference > 1.0
        {
            return Err(IntonationAnalysisError::ConfigError(
                "Contour smoothness preference must be between 0.0 and 1.0".to_string(),
            ));
        }

        Ok(())
    }

    fn generate_cache_key(&self, sentences: &[String], pitch_data: Option<&[f64]>) -> u64 {
        let mut hasher = DefaultHasher::new();

        for sentence in sentences {
            sentence.hash(&mut hasher);
        }

        if let Some(pitch) = pitch_data {
            for &value in pitch {
                (value as i64).hash(&mut hasher);
            }
        }

        self.config.enabled.hash(&mut hasher);
        hasher.finish()
    }

    fn simulate_pitch_contour(&self, sentences: &[String]) -> IntonationResult<PitchContour> {
        // Simple pitch contour simulation for analysis
        let text_length = sentences.iter().map(|s| s.len()).sum::<usize>();
        let num_points = (text_length / 5).max(10).min(200);

        let time_points: Vec<f64> = (0..num_points).map(|i| i as f64 * 0.01).collect();
        let mut f0_values = Vec::with_capacity(num_points);

        let base_f0 = 150.0; // Hz
        let mut current_f0 = base_f0;

        for (i, sentence) in sentences.iter().enumerate() {
            let sentence_points = num_points / sentences.len().max(1);

            // Simulate sentence-level intonation pattern
            let sentence_type = self.guess_sentence_type(sentence);

            for j in 0..sentence_points {
                let progress = j as f64 / sentence_points as f64;

                let f0 = match sentence_type {
                    SentenceType::Declarative => {
                        base_f0 + 20.0 * (1.0 - progress) // Falling
                    }
                    SentenceType::Interrogative => {
                        base_f0 + 30.0 * progress // Rising
                    }
                    SentenceType::Exclamatory => {
                        base_f0 + 40.0 * (0.5 - (progress - 0.5).abs()) // Peak in middle
                    }
                    _ => base_f0,
                };

                f0_values.push(f0);
            }
        }

        // Pad to correct length
        while f0_values.len() < time_points.len() {
            f0_values.push(base_f0);
        }
        f0_values.truncate(time_points.len());

        Ok(PitchContour::new(time_points, f0_values))
    }

    fn guess_sentence_type(&self, sentence: &str) -> SentenceType {
        if sentence.ends_with('?') {
            SentenceType::Interrogative
        } else if sentence.ends_with('!') {
            SentenceType::Exclamatory
        } else if sentence.to_lowercase().starts_with("please")
            || sentence.to_lowercase().contains("should")
            || sentence.to_lowercase().contains("must")
        {
            SentenceType::Imperative
        } else {
            SentenceType::Declarative
        }
    }

    fn extract_syllable_info(
        &self,
        sentences: &[String],
    ) -> IntonationResult<Vec<(String, usize)>> {
        let mut syllable_info = Vec::new();
        let mut position = 0;

        for sentence in sentences {
            let words: Vec<&str> = sentence.split_whitespace().collect();

            for word in words {
                let syllables = self.extract_syllables(word);
                for syllable in syllables {
                    syllable_info.push((syllable, position));
                    position += 1;
                }
            }
        }

        Ok(syllable_info)
    }

    fn extract_syllables(&self, word: &str) -> Vec<String> {
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

    fn calculate_intonation_appropriateness(
        &self,
        sentences: &[String],
        contour: &PitchContour,
    ) -> IntonationResult<f64> {
        let mut appropriateness_scores = Vec::new();

        for sentence in sentences {
            let expected_pattern = self.get_expected_intonation_pattern(sentence);
            let actual_pattern = self.extract_sentence_pattern(sentence, contour)?;
            let match_score = self.calculate_pattern_match(&expected_pattern, &actual_pattern);

            appropriateness_scores.push(match_score);
        }

        if appropriateness_scores.is_empty() {
            Ok(0.5)
        } else {
            Ok(appropriateness_scores.iter().sum::<f64>() / appropriateness_scores.len() as f64)
        }
    }

    fn get_expected_intonation_pattern(&self, sentence: &str) -> Vec<f64> {
        // Simple expected pattern based on sentence type
        let sentence_type = self.guess_sentence_type(sentence);
        let length = sentence.split_whitespace().count().max(1);

        match sentence_type {
            SentenceType::Declarative => (0..length)
                .map(|i| 1.0 - (i as f64 / length as f64) * 0.5)
                .collect(),
            SentenceType::Interrogative => (0..length)
                .map(|i| 0.7 + (i as f64 / length as f64) * 0.6)
                .collect(),
            SentenceType::Exclamatory => (0..length)
                .map(|i| {
                    let pos = i as f64 / length as f64;
                    1.0 + 0.5 * (1.0 - 2.0 * (pos - 0.5).abs())
                })
                .collect(),
            _ => vec![1.0; length],
        }
    }

    fn extract_sentence_pattern(
        &self,
        sentence: &str,
        contour: &PitchContour,
    ) -> IntonationResult<Vec<f64>> {
        // Extract normalized pattern from contour
        let word_count = sentence.split_whitespace().count();
        if word_count == 0 || contour.f0_values.is_empty() {
            return Ok(vec![]);
        }

        let points_per_word = contour.f0_values.len() / word_count;
        let mut pattern = Vec::new();

        for i in 0..word_count {
            let start_idx = i * points_per_word;
            let end_idx = ((i + 1) * points_per_word).min(contour.f0_values.len());

            if start_idx < end_idx {
                let word_avg = contour.f0_values[start_idx..end_idx].iter().sum::<f64>()
                    / (end_idx - start_idx) as f64;
                pattern.push(word_avg);
            }
        }

        // Normalize pattern
        if let Some(&max_val) = pattern.iter().max_by(|a, b| a.partial_cmp(b).unwrap()) {
            if max_val > 0.0 {
                for val in &mut pattern {
                    *val /= max_val;
                }
            }
        }

        Ok(pattern)
    }

    fn calculate_pattern_match(&self, expected: &[f64], actual: &[f64]) -> f64 {
        if expected.is_empty() || actual.is_empty() {
            return 0.5;
        }

        let min_len = expected.len().min(actual.len());
        let mut total_error = 0.0;

        for i in 0..min_len {
            total_error += (expected[i] - actual[i]).abs();
        }

        let avg_error = total_error / min_len as f64;
        (1.0 - avg_error).max(0.0)
    }

    fn analyze_pitch_contour(
        &self,
        contour: &PitchContour,
    ) -> IntonationResult<PitchContourAnalysis> {
        let smoothness = self.calculate_contour_smoothness(contour)?;
        let complexity = self.calculate_contour_complexity(contour)?;
        let peaks_valleys = contour.find_peaks_valleys(10.0);
        let segments = contour.segment_contour(3);
        let contour_shape = self.classify_overall_contour_shape(contour)?;

        Ok(PitchContourAnalysis {
            smoothness,
            complexity,
            peaks_valleys,
            segments,
            contour_shape,
        })
    }

    fn calculate_contour_smoothness(&self, contour: &PitchContour) -> IntonationResult<f64> {
        if contour.f0_values.len() < 2 {
            return Ok(1.0);
        }

        let mut total_variation = 0.0;
        for i in 1..contour.f0_values.len() {
            total_variation += (contour.f0_values[i] - contour.f0_values[i - 1]).abs();
        }

        let avg_variation = total_variation / (contour.f0_values.len() - 1) as f64;
        let (min_f0, max_f0) = contour.calculate_range();
        let pitch_range = max_f0 - min_f0;

        if pitch_range > 0.0 {
            let relative_variation = avg_variation / pitch_range;
            Ok(1.0 / (1.0 + relative_variation))
        } else {
            Ok(1.0)
        }
    }

    fn calculate_contour_complexity(&self, contour: &PitchContour) -> IntonationResult<f64> {
        let peaks_valleys = contour.find_peaks_valleys(5.0);
        let num_extrema = peaks_valleys.peaks.len() + peaks_valleys.valleys.len();
        let contour_length = contour.f0_values.len();

        if contour_length > 0 {
            let extrema_density = num_extrema as f64 / contour_length as f64;
            Ok(extrema_density.min(1.0))
        } else {
            Ok(0.0)
        }
    }

    fn classify_overall_contour_shape(
        &self,
        contour: &PitchContour,
    ) -> IntonationResult<ContourShape> {
        if contour.f0_values.len() < 3 {
            return Ok(ContourShape::Declarative);
        }

        let first_third = contour.f0_values.len() / 3;
        let last_third = 2 * contour.f0_values.len() / 3;

        let start_avg = contour.f0_values[..first_third].iter().sum::<f64>() / first_third as f64;
        let end_avg = contour.f0_values[last_third..].iter().sum::<f64>()
            / (contour.f0_values.len() - last_third) as f64;

        let overall_change = end_avg - start_avg;
        let threshold = 15.0; // Hz

        if overall_change > threshold {
            Ok(ContourShape::Interrogative)
        } else if overall_change < -threshold {
            Ok(ContourShape::Declarative)
        } else {
            // Check for peak in middle (exclamatory pattern)
            let middle_third = &contour.f0_values[first_third..last_third];
            let middle_max = middle_third
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

            if middle_max > start_avg + 20.0 && middle_max > end_avg + 20.0 {
                Ok(ContourShape::Exclamatory)
            } else {
                Ok(ContourShape::Complex)
            }
        }
    }

    fn classify_sentences(
        &self,
        sentences: &[String],
        contour: &PitchContour,
    ) -> IntonationResult<Vec<SentenceTypeClassification>> {
        let mut classifications = Vec::new();

        for (i, sentence) in sentences.iter().enumerate() {
            let sentence_type = self.guess_sentence_type(sentence);
            let confidence = self.calculate_classification_confidence(sentence, contour, i)?;
            let features = self.extract_intonational_features(sentence, contour, i)?;

            classifications.push(SentenceTypeClassification {
                position: i,
                sentence_type,
                confidence,
                features,
            });
        }

        Ok(classifications)
    }

    fn calculate_classification_confidence(
        &self,
        sentence: &str,
        contour: &PitchContour,
        position: usize,
    ) -> IntonationResult<f64> {
        // Simple confidence based on punctuation match and contour pattern
        let punctuation_confidence =
            if sentence.ends_with('?') || sentence.ends_with('!') || sentence.ends_with('.') {
                0.8
            } else {
                0.5
            };

        let pattern_confidence = 0.7; // Placeholder for pattern analysis

        Ok((punctuation_confidence + pattern_confidence) / 2.0)
    }

    fn extract_intonational_features(
        &self,
        sentence: &str,
        contour: &PitchContour,
        position: usize,
    ) -> IntonationResult<HashMap<String, f64>> {
        let mut features = HashMap::new();

        features.insert(
            "sentence_length".to_string(),
            sentence.split_whitespace().count() as f64,
        );

        // Simple feature extraction
        if sentence.ends_with('?') {
            features.insert("question_marker".to_string(), 1.0);
        }
        if sentence.ends_with('!') {
            features.insert("exclamation_marker".to_string(), 1.0);
        }

        features.insert("pitch_range".to_string(), {
            let (min_f0, max_f0) = contour.calculate_range();
            max_f0 - min_f0
        });

        Ok(features)
    }

    fn analyze_phrase_structure(
        &self,
        sentences: &[String],
        contour: &PitchContour,
    ) -> IntonationResult<IntonationalPhraseStructure> {
        // Simplified phrase structure analysis
        let phrase_boundaries = self.detect_phrase_boundaries(sentences, contour)?;
        let phrase_types = vec![PhraseType::Major; phrase_boundaries.len()];
        let coherence = self.calculate_phrase_coherence(&phrase_boundaries, contour)?;
        let hierarchy = self.build_phrase_hierarchy(&phrase_boundaries, sentences)?;

        Ok(IntonationalPhraseStructure {
            phrase_boundaries,
            phrase_types,
            coherence,
            hierarchy,
        })
    }

    fn detect_phrase_boundaries(
        &self,
        sentences: &[String],
        contour: &PitchContour,
    ) -> IntonationResult<Vec<usize>> {
        // Simple boundary detection based on sentence boundaries and pitch drops
        let mut boundaries = vec![0]; // Start

        let mut position = 0;
        for (i, sentence) in sentences.iter().enumerate() {
            position += sentence.split_whitespace().count();
            if i < sentences.len() - 1 {
                boundaries.push(position);
            }
        }

        Ok(boundaries)
    }

    fn calculate_phrase_coherence(
        &self,
        boundaries: &[usize],
        contour: &PitchContour,
    ) -> IntonationResult<f64> {
        // Placeholder coherence calculation
        Ok(0.7)
    }

    fn build_phrase_hierarchy(
        &self,
        boundaries: &[usize],
        sentences: &[String],
    ) -> IntonationResult<PhraseHierarchy> {
        let mut levels = Vec::new();

        // Simple two-level hierarchy
        let sentence_level = HierarchyLevel {
            level: 0,
            units: sentences
                .iter()
                .enumerate()
                .map(|(i, sentence)| PhraseUnit {
                    boundaries: if i < boundaries.len() - 1 {
                        (boundaries[i], boundaries[i + 1])
                    } else {
                        (boundaries[i], sentence.split_whitespace().count())
                    },
                    unit_type: PhraseType::Major,
                    content: sentence.clone(),
                    properties: HashMap::new(),
                })
                .collect(),
            prominence: 1.0,
        };

        levels.push(sentence_level);

        Ok(PhraseHierarchy {
            levels,
            relationships: HashMap::new(),
            depth: 1,
        })
    }

    fn analyze_pitch_range(&self, contour: &PitchContour) -> IntonationResult<PitchRangeAnalysis> {
        let overall_range = contour.calculate_range();
        let pitch_span = overall_range.1 - overall_range.0;
        let range_utilization = if pitch_span > 0.0 { 0.8 } else { 0.0 }; // Placeholder
        let local_variations = self.analyze_local_pitch_variations(contour)?;
        let compression_analysis = None; // Would be implemented for full analysis

        Ok(PitchRangeAnalysis {
            overall_range,
            pitch_span,
            range_utilization,
            local_variations,
            compression_analysis,
        })
    }

    fn analyze_local_pitch_variations(
        &self,
        contour: &PitchContour,
    ) -> IntonationResult<Vec<LocalPitchRange>> {
        let window_size = 10;
        let mut variations = Vec::new();

        for i in 0..contour.f0_values.len().saturating_sub(window_size) {
            let window = &contour.f0_values[i..i + window_size];
            let local_min = window.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let local_max = window.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let range = (local_min, local_max);
            let width = local_max - local_min;

            variations.push(LocalPitchRange {
                position: i,
                range,
                width,
                variation: 0.0, // Placeholder
            });
        }

        Ok(variations)
    }

    fn calculate_overall_intonation_score(
        &self,
        appropriateness: f64,
        contour_analysis: &Option<PitchContourAnalysis>,
        boundary_tones: &Option<Vec<BoundaryTone>>,
        focus_patterns: &Option<Vec<FocusPattern>>,
    ) -> f64 {
        let mut components = vec![appropriateness];

        if let Some(contour) = contour_analysis {
            components.push(contour.smoothness * self.config.contour_smoothness_preference);
        }

        if let Some(boundaries) = boundary_tones {
            let boundary_quality = if boundaries.is_empty() {
                0.5
            } else {
                boundaries.iter().map(|b| b.strength).sum::<f64>() / boundaries.len() as f64
            };
            components.push(boundary_quality);
        }

        if let Some(focus) = focus_patterns {
            let focus_quality = if focus.is_empty() {
                0.5
            } else {
                0.8 // Placeholder for focus quality
            };
            components.push(focus_quality);
        }

        components.iter().sum::<f64>() / components.len() as f64
    }
}

impl Default for IntonationAnalyzer {
    fn default() -> Self {
        Self::new(IntonationAnalysisConfig {
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
        })
        .unwrap()
    }
}
