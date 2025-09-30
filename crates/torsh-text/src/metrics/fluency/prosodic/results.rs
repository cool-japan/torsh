//! Prosodic Results Module
//!
//! This module provides comprehensive result structures for prosodic fluency analysis,
//! including detailed metrics for rhythm, stress, intonation, timing, phonological patterns,
//! and advanced prosodic features with full serialization support.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Main result structure for prosodic fluency analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProsodicFluencyResult {
    /// Overall prosodic fluency score (0.0 to 1.0)
    pub overall_score: f64,
    /// Rhythmic flow quality assessment
    pub rhythmic_flow: f64,
    /// Stress pattern naturalness evaluation
    pub stress_pattern_naturalness: f64,
    /// Intonation appropriateness assessment
    pub intonation_appropriateness: f64,
    /// Pause placement accuracy evaluation
    pub pause_placement: f64,
    /// Reading ease prosodic component
    pub reading_ease: f64,
    /// Syllable complexity measure
    pub syllable_complexity: f64,
    /// Phonological pattern analysis results
    pub phonological_patterns: HashMap<String, f64>,
    /// Prosodic break analysis
    pub prosodic_breaks: Vec<ProsodicBreak>,
    /// Detailed rhythm analysis breakdown
    pub rhythm_breakdown: Option<RhythmMetrics>,
    /// Detailed stress analysis breakdown
    pub stress_breakdown: Option<StressMetrics>,
    /// Detailed intonation analysis breakdown
    pub intonation_breakdown: Option<IntonationMetrics>,
    /// Detailed timing analysis breakdown
    pub timing_breakdown: Option<TimingMetrics>,
    /// Detailed phonological analysis breakdown
    pub phonological_breakdown: Option<PhonologicalMetrics>,
    /// Advanced prosodic metrics
    pub advanced_breakdown: Option<AdvancedProsodicMetrics>,
    /// Analysis insights and recommendations
    pub insights: Vec<String>,
    /// Performance and quality recommendations
    pub recommendations: Vec<String>,
}

/// Comprehensive rhythm analysis metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RhythmMetrics {
    /// Overall rhythm score
    pub overall_rhythm_score: f64,
    /// Rhythm regularity measure
    pub rhythm_regularity: f64,
    /// Beat pattern consistency
    pub beat_consistency: f64,
    /// Alternation pattern quality
    pub alternation_quality: f64,
    /// Rhythm template matches
    pub template_matches: Vec<RhythmTemplateMatch>,
    /// Beat pattern analysis
    pub beat_patterns: Vec<BeatPattern>,
    /// Rhythm classification results
    pub rhythm_classification: Option<RhythmClassification>,
    /// Timing regularity variance
    pub timing_variance: f64,
    /// Rhythmic complexity score
    pub rhythmic_complexity: f64,
    /// Pattern entropy measure
    pub pattern_entropy: f64,
}

/// Rhythm template matching result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RhythmTemplateMatch {
    /// Template name
    pub template_name: String,
    /// Match confidence score
    pub confidence: f64,
    /// Coverage of text by template
    pub coverage: f64,
    /// Template pattern
    pub pattern: Vec<usize>,
    /// Position in text where match occurs
    pub positions: Vec<usize>,
}

/// Beat pattern analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeatPattern {
    /// Beat type classification
    pub beat_type: BeatType,
    /// Position in text
    pub position: usize,
    /// Strength of beat
    pub strength: f64,
    /// Context information
    pub context: String,
}

/// Types of prosodic beats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BeatType {
    /// Strong prosodic beat
    Strong,
    /// Weak prosodic beat
    Weak,
    /// Intermediate strength beat
    Intermediate,
    /// Silent beat (pause)
    Silent,
}

/// Rhythm classification result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RhythmClassification {
    /// Primary rhythm class
    pub primary_class: RhythmClass,
    /// Secondary rhythm classes with confidence
    pub secondary_classes: HashMap<RhythmClass, f64>,
    /// Classification confidence
    pub confidence: f64,
    /// Classification features
    pub features: HashMap<String, f64>,
}

/// Prosodic rhythm classes
#[derive(Debug, Clone, Serialize, Deserialize, Hash, Eq, PartialEq)]
pub enum RhythmClass {
    /// Stress-timed rhythm
    StressTimed,
    /// Syllable-timed rhythm
    SyllableTimed,
    /// Mora-timed rhythm
    MoraTimed,
    /// Mixed rhythm patterns
    Mixed,
    /// Irregular rhythm
    Irregular,
}

/// Comprehensive stress analysis metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressMetrics {
    /// Overall stress pattern score
    pub overall_stress_score: f64,
    /// Stress pattern naturalness
    pub pattern_naturalness: f64,
    /// Metrical structure quality
    pub metrical_quality: f64,
    /// Stress placement accuracy
    pub placement_accuracy: f64,
    /// Prominence distribution analysis
    pub prominence_analysis: ProminenceAnalysis,
    /// Metrical structure breakdown
    pub metrical_structure: MetricalStructure,
    /// Stress clash detection results
    pub stress_clashes: Vec<StressClash>,
    /// Foot structure analysis
    pub foot_analysis: FootAnalysis,
    /// Accent pattern analysis
    pub accent_patterns: Vec<AccentPattern>,
    /// Stress prediction accuracy
    pub prediction_accuracy: Option<f64>,
}

/// Prominence analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProminenceAnalysis {
    /// Primary stress positions
    pub primary_stress_positions: Vec<usize>,
    /// Secondary stress positions
    pub secondary_stress_positions: Vec<usize>,
    /// Prominence hierarchy
    pub prominence_hierarchy: Vec<ProminenceLevel>,
    /// Focus prominence markers
    pub focus_prominence: Vec<FocusMarker>,
    /// Contrastive stress detection
    pub contrastive_stress: Vec<ContrastiveStress>,
}

/// Prosodic prominence levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProminenceLevel {
    /// Position in text
    pub position: usize,
    /// Prominence strength
    pub strength: f64,
    /// Prominence type
    pub prominence_type: ProminenceType,
    /// Contributing factors
    pub factors: Vec<String>,
}

/// Types of prosodic prominence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProminenceType {
    /// Primary word stress
    Primary,
    /// Secondary word stress
    Secondary,
    /// Phrasal prominence
    Phrasal,
    /// Contrastive prominence
    Contrastive,
    /// Focus prominence
    Focus,
}

/// Focus prominence marker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FocusMarker {
    /// Position in text
    pub position: usize,
    /// Focus type
    pub focus_type: FocusType,
    /// Focus strength
    pub strength: f64,
    /// Scope of focus
    pub scope: Vec<usize>,
}

/// Types of prosodic focus
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FocusType {
    /// Contrastive focus
    Contrastive,
    /// Information focus
    Information,
    /// Corrective focus
    Corrective,
    /// Emphatic focus
    Emphatic,
}

/// Contrastive stress marker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContrastiveStress {
    /// Position in text
    pub position: usize,
    /// Contrast strength
    pub strength: f64,
    /// Contrasted elements
    pub contrasted_elements: Vec<String>,
    /// Contextual information
    pub context: String,
}

/// Metrical structure analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricalStructure {
    /// Metrical feet identified
    pub metrical_feet: Vec<MetricalFoot>,
    /// Overall metrical consistency
    pub consistency_score: f64,
    /// Dominant foot type
    pub dominant_foot_type: FootType,
    /// Metrical pattern regularity
    pub pattern_regularity: f64,
    /// Metrical violations
    pub violations: Vec<MetricalViolation>,
}

/// Metrical foot structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricalFoot {
    /// Foot type classification
    pub foot_type: FootType,
    /// Syllables in foot
    pub syllables: Vec<String>,
    /// Stress pattern within foot
    pub stress_pattern: Vec<bool>,
    /// Foot boundaries
    pub boundaries: (usize, usize),
    /// Foot strength
    pub strength: f64,
}

/// Types of metrical feet
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FootType {
    /// Trochaic foot (strong-weak)
    Trochee,
    /// Iambic foot (weak-strong)
    Iamb,
    /// Dactylic foot (strong-weak-weak)
    Dactyl,
    /// Anapestic foot (weak-weak-strong)
    Anapest,
    /// Spondaic foot (strong-strong)
    Spondee,
    /// Pyrrhic foot (weak-weak)
    Pyrrhic,
}

/// Metrical violation detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricalViolation {
    /// Violation type
    pub violation_type: String,
    /// Position in text
    pub position: usize,
    /// Severity of violation
    pub severity: f64,
    /// Description of violation
    pub description: String,
    /// Suggested correction
    pub correction: Option<String>,
}

/// Stress clash detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressClash {
    /// Positions of clashing stresses
    pub positions: Vec<usize>,
    /// Clash severity
    pub severity: f64,
    /// Context of clash
    pub context: String,
    /// Resolution suggestions
    pub resolutions: Vec<String>,
}

/// Foot structure analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FootAnalysis {
    /// Foot boundary accuracy
    pub boundary_accuracy: f64,
    /// Foot type distribution
    pub type_distribution: HashMap<FootType, f64>,
    /// Average foot length
    pub average_length: f64,
    /// Foot regularity score
    pub regularity_score: f64,
    /// Complex foot patterns
    pub complex_patterns: Vec<ComplexFootPattern>,
}

/// Complex foot pattern analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexFootPattern {
    /// Pattern description
    pub pattern: String,
    /// Occurrence count
    pub count: usize,
    /// Pattern complexity
    pub complexity: f64,
    /// Positions where pattern occurs
    pub positions: Vec<usize>,
}

/// Accent pattern analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccentPattern {
    /// Accent type
    pub accent_type: AccentType,
    /// Position in text
    pub position: usize,
    /// Accent strength
    pub strength: f64,
    /// Tonal properties
    pub tonal_properties: Option<TonalProperties>,
}

/// Types of prosodic accents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AccentType {
    /// High tone accent
    HighTone,
    /// Low tone accent
    LowTone,
    /// Rising tone accent
    Rising,
    /// Falling tone accent
    Falling,
    /// Complex tone accent
    Complex,
}

/// Tonal properties of accents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TonalProperties {
    /// Fundamental frequency values
    pub f0_values: Vec<f64>,
    /// Pitch range
    pub pitch_range: (f64, f64),
    /// Tonal movement
    pub movement: TonalMovement,
    /// Alignment with syllables
    pub alignment: Vec<usize>,
}

/// Types of tonal movement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TonalMovement {
    /// Level tone
    Level,
    /// Rising movement
    Rising,
    /// Falling movement
    Falling,
    /// Rise-fall contour
    RiseFall,
    /// Fall-rise contour
    FallRise,
}

/// Comprehensive intonation analysis metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntonationMetrics {
    /// Overall intonation score
    pub overall_intonation_score: f64,
    /// Intonation appropriateness
    pub appropriateness: f64,
    /// Pitch contour analysis
    pub contour_analysis: Option<PitchContourAnalysis>,
    /// Boundary tone analysis
    pub boundary_tones: Option<Vec<BoundaryTone>>,
    /// Focus pattern analysis
    pub focus_patterns: Option<Vec<FocusPattern>>,
    /// Sentence type classification
    pub sentence_classifications: Vec<SentenceTypeClassification>,
    /// Intonational phrase structure
    pub phrase_structure: Option<IntonationalPhraseStructure>,
    /// Pitch range analysis
    pub pitch_range_analysis: Option<PitchRangeAnalysis>,
}

/// Pitch contour analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PitchContourAnalysis {
    /// Contour smoothness score
    pub smoothness: f64,
    /// Contour complexity
    pub complexity: f64,
    /// Peak and valley analysis
    pub peaks_valleys: PeakValleyAnalysis,
    /// Contour segments
    pub segments: Vec<ContourSegment>,
    /// Overall contour shape
    pub contour_shape: ContourShape,
}

/// Peak and valley analysis in pitch contours
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeakValleyAnalysis {
    /// Peak positions and values
    pub peaks: Vec<PitchPoint>,
    /// Valley positions and values
    pub valleys: Vec<PitchPoint>,
    /// Peak-valley ratio
    pub peak_valley_ratio: f64,
    /// Contour regularity
    pub regularity: f64,
}

/// Pitch point representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PitchPoint {
    /// Position in text
    pub position: usize,
    /// Pitch value
    pub pitch: f64,
    /// Prominence level
    pub prominence: f64,
}

/// Contour segment analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContourSegment {
    /// Segment boundaries
    pub boundaries: (usize, usize),
    /// Segment trend
    pub trend: TrendDirection,
    /// Slope steepness
    pub slope: f64,
    /// Segment length
    pub length: f64,
}

/// Trend directions in pitch contours
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    /// Rising pitch trend
    Rising,
    /// Falling pitch trend
    Falling,
    /// Level pitch trend
    Level,
    /// Complex trend pattern
    Complex,
}

/// Overall contour shape classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContourShape {
    /// Declarative contour
    Declarative,
    /// Interrogative contour
    Interrogative,
    /// Exclamatory contour
    Exclamatory,
    /// Complex contour
    Complex,
}

/// Boundary tone analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundaryTone {
    /// Position in text
    pub position: usize,
    /// Boundary type
    pub boundary_type: BoundaryType,
    /// Tone type
    pub tone_type: ToneType,
    /// Tone strength
    pub strength: f64,
}

/// Types of prosodic boundaries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BoundaryType {
    /// Phrase boundary
    Phrase,
    /// Utterance boundary
    Utterance,
    /// Intermediate boundary
    Intermediate,
    /// Major boundary
    Major,
}

/// Types of boundary tones
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ToneType {
    /// High boundary tone
    High,
    /// Low boundary tone
    Low,
    /// Mid boundary tone
    Mid,
    /// Rising boundary tone
    Rising,
    /// Falling boundary tone
    Falling,
}

/// Focus pattern analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FocusPattern {
    /// Focus position
    pub position: usize,
    /// Focus type
    pub focus_type: FocusType,
    /// Focus scope
    pub scope: Vec<usize>,
    /// Acoustic correlates
    pub acoustic_correlates: AcousticCorrelates,
}

/// Acoustic correlates of focus
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcousticCorrelates {
    /// Fundamental frequency changes
    pub f0_changes: Vec<f64>,
    /// Duration changes
    pub duration_changes: Vec<f64>,
    /// Intensity changes
    pub intensity_changes: Vec<f64>,
}

/// Sentence type classification result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentenceTypeClassification {
    /// Sentence position
    pub position: usize,
    /// Classified sentence type
    pub sentence_type: SentenceType,
    /// Classification confidence
    pub confidence: f64,
    /// Intonational features
    pub features: HashMap<String, f64>,
}

/// Types of sentences based on intonation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SentenceType {
    /// Declarative sentence
    Declarative,
    /// Interrogative sentence
    Interrogative,
    /// Exclamatory sentence
    Exclamatory,
    /// Imperative sentence
    Imperative,
    /// Conditional sentence
    Conditional,
}

/// Intonational phrase structure analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntonationalPhraseStructure {
    /// Phrase boundaries
    pub phrase_boundaries: Vec<usize>,
    /// Phrase types
    pub phrase_types: Vec<PhraseType>,
    /// Phrase coherence
    pub coherence: f64,
    /// Hierarchical structure
    pub hierarchy: PhraseHierarchy,
}

/// Types of intonational phrases
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PhraseType {
    /// Major intonational phrase
    Major,
    /// Minor intonational phrase
    Minor,
    /// Intermediate phrase
    Intermediate,
    /// Accentual phrase
    Accentual,
}

/// Phrase hierarchy representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhraseHierarchy {
    /// Hierarchical levels
    pub levels: Vec<HierarchyLevel>,
    /// Level relationships
    pub relationships: HashMap<usize, Vec<usize>>,
    /// Hierarchy depth
    pub depth: usize,
}

/// Hierarchy level in phrase structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchyLevel {
    /// Level index
    pub level: usize,
    /// Phrase units at this level
    pub units: Vec<PhraseUnit>,
    /// Level prominence
    pub prominence: f64,
}

/// Phrase unit representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhraseUnit {
    /// Unit boundaries
    pub boundaries: (usize, usize),
    /// Unit type
    pub unit_type: PhraseType,
    /// Unit content
    pub content: String,
    /// Prosodic properties
    pub properties: HashMap<String, f64>,
}

/// Pitch range analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PitchRangeAnalysis {
    /// Overall pitch range
    pub overall_range: (f64, f64),
    /// Pitch span
    pub pitch_span: f64,
    /// Range utilization
    pub range_utilization: f64,
    /// Local range variations
    pub local_variations: Vec<LocalPitchRange>,
    /// Range compression analysis
    pub compression_analysis: Option<RangeCompressionAnalysis>,
}

/// Local pitch range variation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalPitchRange {
    /// Position in text
    pub position: usize,
    /// Local range
    pub range: (f64, f64),
    /// Range width
    pub width: f64,
    /// Variation from global range
    pub variation: f64,
}

/// Pitch range compression analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RangeCompressionAnalysis {
    /// Compression ratio
    pub compression_ratio: f64,
    /// Compression points
    pub compression_points: Vec<usize>,
    /// Expansion points
    pub expansion_points: Vec<usize>,
    /// Overall compression trend
    pub trend: CompressionTrend,
}

/// Compression trend types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionTrend {
    /// Increasing compression
    Increasing,
    /// Decreasing compression
    Decreasing,
    /// Stable compression
    Stable,
    /// Variable compression
    Variable,
}

/// Comprehensive timing analysis metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingMetrics {
    /// Overall timing score
    pub overall_timing_score: f64,
    /// Pause placement accuracy
    pub pause_accuracy: f64,
    /// Tempo analysis results
    pub tempo_analysis: Option<TempoAnalysis>,
    /// Duration analysis results
    pub duration_analysis: Option<DurationAnalysis>,
    /// Speech rate analysis
    pub speech_rate_analysis: Option<SpeechRateAnalysis>,
    /// Timing variability analysis
    pub variability_analysis: Option<TimingVariabilityAnalysis>,
    /// Prosodic break analysis
    pub break_analysis: Vec<ProsodicBreak>,
    /// Syllable timing analysis
    pub syllable_timing: Option<SyllableTimingAnalysis>,
}

/// Prosodic break representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProsodicBreak {
    /// Break position
    pub position: usize,
    /// Break type
    pub break_type: BreakType,
    /// Break strength
    pub strength: f64,
    /// Duration of break
    pub duration: Option<f64>,
    /// Context information
    pub context: String,
}

/// Types of prosodic breaks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BreakType {
    /// Minor break
    Minor,
    /// Major break
    Major,
    /// Pause break
    Pause,
    /// Boundary break
    Boundary,
}

/// Tempo analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TempoAnalysis {
    /// Average tempo
    pub average_tempo: f64,
    /// Tempo variability
    pub tempo_variability: f64,
    /// Tempo changes
    pub tempo_changes: Vec<TempoChange>,
    /// Tempo regularity
    pub regularity: f64,
    /// Tempo classification
    pub tempo_class: TempoClass,
}

/// Tempo change detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TempoChange {
    /// Position of change
    pub position: usize,
    /// Change magnitude
    pub magnitude: f64,
    /// Change direction
    pub direction: ChangeDirection,
    /// Context of change
    pub context: String,
}

/// Direction of tempo/duration changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChangeDirection {
    /// Increase in tempo/duration
    Increase,
    /// Decrease in tempo/duration
    Decrease,
    /// Stable tempo/duration
    Stable,
}

/// Tempo classification categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TempoClass {
    /// Very slow tempo
    VerySlow,
    /// Slow tempo
    Slow,
    /// Moderate tempo
    Moderate,
    /// Fast tempo
    Fast,
    /// Very fast tempo
    VeryFast,
}

/// Duration analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DurationAnalysis {
    /// Average syllable duration
    pub average_syllable_duration: f64,
    /// Duration variability
    pub duration_variability: f64,
    /// Duration patterns
    pub duration_patterns: Vec<DurationPattern>,
    /// Lengthening effects
    pub lengthening_effects: Vec<LengtheningEffect>,
}

/// Duration pattern analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DurationPattern {
    /// Pattern type
    pub pattern_type: String,
    /// Pattern strength
    pub strength: f64,
    /// Pattern positions
    pub positions: Vec<usize>,
    /// Pattern regularity
    pub regularity: f64,
}

/// Lengthening effect analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LengtheningEffect {
    /// Position of lengthening
    pub position: usize,
    /// Lengthening factor
    pub factor: f64,
    /// Lengthening type
    pub lengthening_type: LengtheningType,
    /// Contextual cause
    pub cause: String,
}

/// Types of duration lengthening
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LengtheningType {
    /// Pre-boundary lengthening
    PreBoundary,
    /// Focus-induced lengthening
    Focus,
    /// Stress-induced lengthening
    Stress,
    /// Phrase-final lengthening
    PhraseFinal,
}

/// Speech rate analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeechRateAnalysis {
    /// Words per minute
    pub words_per_minute: f64,
    /// Syllables per second
    pub syllables_per_second: f64,
    /// Speech rate variability
    pub rate_variability: f64,
    /// Rate changes over time
    pub rate_changes: Vec<SpeechRateChange>,
    /// Rate classification
    pub rate_class: SpeechRateClass,
}

/// Speech rate change detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeechRateChange {
    /// Position of rate change
    pub position: usize,
    /// New rate value
    pub new_rate: f64,
    /// Change magnitude
    pub magnitude: f64,
    /// Change context
    pub context: String,
}

/// Speech rate classification categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SpeechRateClass {
    /// Very slow speech
    VerySlow,
    /// Slow speech
    Slow,
    /// Normal speech rate
    Normal,
    /// Fast speech
    Fast,
    /// Very fast speech
    VeryFast,
}

/// Timing variability analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingVariabilityAnalysis {
    /// Overall timing variability
    pub overall_variability: f64,
    /// Variability by component
    pub component_variability: HashMap<String, f64>,
    /// Variability patterns
    pub variability_patterns: Vec<VariabilityPattern>,
    /// Consistency metrics
    pub consistency_metrics: ConsistencyMetrics,
}

/// Timing variability pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariabilityPattern {
    /// Pattern description
    pub pattern: String,
    /// Pattern strength
    pub strength: f64,
    /// Pattern positions
    pub positions: Vec<usize>,
    /// Pattern regularity
    pub regularity: f64,
}

/// Timing consistency metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyMetrics {
    /// Beat consistency
    pub beat_consistency: f64,
    /// Rhythm consistency
    pub rhythm_consistency: f64,
    /// Pause consistency
    pub pause_consistency: f64,
    /// Overall consistency
    pub overall_consistency: f64,
}

/// Syllable timing analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyllableTimingAnalysis {
    /// Average syllable duration
    pub average_duration: f64,
    /// Syllable timing patterns
    pub timing_patterns: Vec<SyllableTimingPattern>,
    /// Timing regularity
    pub timing_regularity: f64,
    /// Isochrony measures
    pub isochrony_measures: IsochronyMeasures,
}

/// Syllable timing pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyllableTimingPattern {
    /// Pattern type
    pub pattern_type: String,
    /// Syllable positions
    pub positions: Vec<usize>,
    /// Pattern strength
    pub strength: f64,
    /// Duration values
    pub durations: Vec<f64>,
}

/// Isochrony analysis measures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IsochronyMeasures {
    /// Stress-timed isochrony
    pub stress_timed: f64,
    /// Syllable-timed isochrony
    pub syllable_timed: f64,
    /// Mora-timed isochrony
    pub mora_timed: Option<f64>,
    /// Overall isochrony classification
    pub classification: IsochronyClass,
}

/// Isochrony classification categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IsochronyClass {
    /// Strong stress-timing
    StrongStressTimed,
    /// Weak stress-timing
    WeakStressTimed,
    /// Strong syllable-timing
    StrongSyllableTimed,
    /// Weak syllable-timing
    WeakSyllableTimed,
    /// Mixed timing
    Mixed,
}

/// Comprehensive phonological analysis metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhonologicalMetrics {
    /// Overall phonological score
    pub overall_phonological_score: f64,
    /// Sound pattern analysis
    pub sound_patterns: SoundPatternAnalysis,
    /// Euphony analysis
    pub euphony_analysis: Option<EuphonyAnalysis>,
    /// Phonotactic analysis
    pub phonotactic_analysis: Option<PhonotacticAnalysis>,
    /// Syllable structure analysis
    pub syllable_structure: Option<SyllableStructureAnalysis>,
    /// Alliteration analysis
    pub alliteration: HashMap<String, f64>,
    /// Assonance analysis
    pub assonance: HashMap<String, f64>,
    /// Consonance analysis
    pub consonance: HashMap<String, f64>,
    /// Rhyme analysis
    pub rhyme: HashMap<String, f64>,
}

/// Sound pattern analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoundPatternAnalysis {
    /// Pattern types found
    pub pattern_types: Vec<String>,
    /// Pattern frequencies
    pub frequencies: HashMap<String, usize>,
    /// Pattern strengths
    pub strengths: HashMap<String, f64>,
    /// Pattern positions
    pub positions: HashMap<String, Vec<usize>>,
    /// Overall pattern density
    pub pattern_density: f64,
}

/// Euphony analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EuphonyAnalysis {
    /// Overall euphony score
    pub euphony_score: f64,
    /// Pleasant sound combinations
    pub pleasant_combinations: Vec<SoundCombination>,
    /// Harsh sound combinations
    pub harsh_combinations: Vec<SoundCombination>,
    /// Sound flow quality
    pub flow_quality: f64,
    /// Articulatory ease
    pub articulatory_ease: f64,
}

/// Sound combination analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoundCombination {
    /// Combination sounds
    pub sounds: Vec<String>,
    /// Position in text
    pub position: usize,
    /// Quality score
    pub quality: f64,
    /// Description
    pub description: String,
}

/// Phonotactic analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhonotacticAnalysis {
    /// Legal sound sequences
    pub legal_sequences: Vec<PhoneticSequence>,
    /// Illegal sound sequences
    pub illegal_sequences: Vec<PhoneticSequence>,
    /// Phonotactic probability
    pub probability: f64,
    /// Constraint violations
    pub constraint_violations: Vec<PhonotacticViolation>,
}

/// Phonetic sequence representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhoneticSequence {
    /// Sequence sounds
    pub sounds: Vec<String>,
    /// Position in text
    pub position: usize,
    /// Probability score
    pub probability: f64,
    /// Frequency in language
    pub frequency: Option<f64>,
}

/// Phonotactic constraint violation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhonotacticViolation {
    /// Violation type
    pub violation_type: String,
    /// Position in text
    pub position: usize,
    /// Severity
    pub severity: f64,
    /// Description
    pub description: String,
}

/// Syllable structure analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyllableStructureAnalysis {
    /// Syllable patterns
    pub syllable_patterns: Vec<SyllablePattern>,
    /// Structure complexity
    pub complexity: f64,
    /// Onset analysis
    pub onset_analysis: OnsetAnalysis,
    /// Nucleus analysis
    pub nucleus_analysis: NucleusAnalysis,
    /// Coda analysis
    pub coda_analysis: CodaAnalysis,
}

/// Syllable pattern representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyllablePattern {
    /// Pattern structure (C=consonant, V=vowel)
    pub pattern: String,
    /// Frequency in text
    pub frequency: usize,
    /// Pattern positions
    pub positions: Vec<usize>,
    /// Complexity score
    pub complexity: f64,
}

/// Syllable onset analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnsetAnalysis {
    /// Simple onsets
    pub simple_onsets: Vec<String>,
    /// Complex onsets
    pub complex_onsets: Vec<String>,
    /// Onset complexity distribution
    pub complexity_distribution: HashMap<usize, usize>,
    /// Average onset complexity
    pub average_complexity: f64,
}

/// Syllable nucleus analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NucleusAnalysis {
    /// Vowel nuclei
    pub vowel_nuclei: Vec<String>,
    /// Diphthong nuclei
    pub diphthong_nuclei: Vec<String>,
    /// Nucleus length distribution
    pub length_distribution: HashMap<usize, usize>,
    /// Average nucleus length
    pub average_length: f64,
}

/// Syllable coda analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodaAnalysis {
    /// Simple codas
    pub simple_codas: Vec<String>,
    /// Complex codas
    pub complex_codas: Vec<String>,
    /// Coda presence ratio
    pub presence_ratio: f64,
    /// Average coda complexity
    pub average_complexity: f64,
}

/// Advanced prosodic analysis metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedProsodicMetrics {
    /// Overall advanced score
    pub overall_advanced_score: f64,
    /// Prosodic hierarchy analysis
    pub hierarchy_analysis: Option<ProsodicHierarchyAnalysis>,
    /// Complexity analysis results
    pub complexity_analysis: Option<ProsodicComplexityAnalysis>,
    /// Entropy measures
    pub entropy_measures: Option<ProsodicEntropyMeasures>,
    /// Machine learning features
    pub ml_features: Option<MLProsodicFeatures>,
    /// Neural modeling results
    pub neural_modeling: Option<NeuralProsodicModeling>,
    /// Performance profiling data
    pub profiling_data: Option<ProfilingData>,
}

/// Prosodic hierarchy analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProsodicHierarchyAnalysis {
    /// Hierarchy levels identified
    pub hierarchy_levels: Vec<HierarchyLevel>,
    /// Level relationships
    pub level_relationships: HashMap<usize, Vec<usize>>,
    /// Hierarchy coherence
    pub coherence: f64,
    /// Hierarchy depth
    pub depth: usize,
    /// Hierarchy complexity
    pub complexity: f64,
}

/// Prosodic complexity analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProsodicComplexityAnalysis {
    /// Overall complexity score
    pub overall_complexity: f64,
    /// Rhythmic complexity
    pub rhythmic_complexity: f64,
    /// Stress complexity
    pub stress_complexity: f64,
    /// Intonational complexity
    pub intonational_complexity: f64,
    /// Temporal complexity
    pub temporal_complexity: f64,
    /// Hierarchical complexity
    pub hierarchical_complexity: f64,
    /// Complexity distribution
    pub complexity_distribution: HashMap<String, f64>,
}

/// Prosodic entropy measures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProsodicEntropyMeasures {
    /// Rhythmic entropy
    pub rhythmic_entropy: f64,
    /// Stress pattern entropy
    pub stress_entropy: f64,
    /// Intonational entropy
    pub intonational_entropy: f64,
    /// Temporal entropy
    pub temporal_entropy: f64,
    /// Overall prosodic entropy
    pub overall_entropy: f64,
    /// Entropy trends
    pub entropy_trends: Vec<EntropyTrend>,
}

/// Entropy trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntropyTrend {
    /// Position in text
    pub position: usize,
    /// Entropy value
    pub entropy: f64,
    /// Trend direction
    pub trend: TrendDirection,
    /// Change rate
    pub change_rate: f64,
}

/// Machine learning prosodic features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLProsodicFeatures {
    /// Feature vector
    pub features: Vec<f64>,
    /// Feature names
    pub feature_names: Vec<String>,
    /// Feature importance scores
    pub importance_scores: Vec<f64>,
    /// Dimensionality
    pub dimensionality: usize,
    /// Feature correlations
    pub correlations: Option<Vec<Vec<f64>>>,
}

/// Neural prosodic modeling results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralProsodicModeling {
    /// Model predictions
    pub predictions: Vec<f64>,
    /// Confidence scores
    pub confidence_scores: Vec<f64>,
    /// Attention weights
    pub attention_weights: Option<Vec<Vec<f64>>>,
    /// Hidden states
    pub hidden_states: Option<Vec<Vec<f64>>>,
    /// Model architecture info
    pub architecture_info: ModelArchitectureInfo,
}

/// Neural model architecture information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelArchitectureInfo {
    /// Model type
    pub model_type: String,
    /// Number of layers
    pub num_layers: usize,
    /// Hidden size
    pub hidden_size: usize,
    /// Input size
    pub input_size: usize,
    /// Output size
    pub output_size: usize,
}

/// Performance profiling data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilingData {
    /// Component timing data
    pub component_timings: HashMap<String, f64>,
    /// Memory usage data
    pub memory_usage: HashMap<String, usize>,
    /// Cache statistics
    pub cache_statistics: CacheStatistics,
    /// Performance bottlenecks
    pub bottlenecks: Vec<PerformanceBottleneck>,
}

/// Cache performance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStatistics {
    /// Cache hit rate
    pub hit_rate: f64,
    /// Cache size
    pub cache_size: usize,
    /// Cache usage
    pub cache_usage: f64,
    /// Eviction count
    pub evictions: usize,
}

/// Performance bottleneck identification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBottleneck {
    /// Component name
    pub component: String,
    /// Bottleneck type
    pub bottleneck_type: String,
    /// Severity score
    pub severity: f64,
    /// Description
    pub description: String,
    /// Optimization suggestions
    pub suggestions: Vec<String>,
}

/// Comparison result for prosodic analyses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProsodicComparisonResult {
    /// First text analysis result
    pub text1_result: ProsodicFluencyResult,
    /// Second text analysis result
    pub text2_result: ProsodicFluencyResult,
    /// Overall difference score
    pub overall_difference: f64,
    /// Component-wise differences
    pub rhythm_difference: f64,
    pub stress_difference: f64,
    pub intonation_difference: f64,
    pub timing_difference: f64,
    pub phonological_difference: f64,
    pub advanced_difference: f64,
    /// Similarity score
    pub similarity_score: f64,
    /// Comparative insights
    pub comparative_insights: Vec<String>,
    /// Improvement recommendations
    pub recommendations: Vec<String>,
}

// Default implementations for all result types

impl Default for ProsodicFluencyResult {
    fn default() -> Self {
        Self {
            overall_score: 0.0,
            rhythmic_flow: 0.0,
            stress_pattern_naturalness: 0.0,
            intonation_appropriateness: 0.0,
            pause_placement: 0.0,
            reading_ease: 0.0,
            syllable_complexity: 0.0,
            phonological_patterns: HashMap::new(),
            prosodic_breaks: Vec::new(),
            rhythm_breakdown: None,
            stress_breakdown: None,
            intonation_breakdown: None,
            timing_breakdown: None,
            phonological_breakdown: None,
            advanced_breakdown: None,
            insights: Vec::new(),
            recommendations: Vec::new(),
        }
    }
}

impl Default for RhythmMetrics {
    fn default() -> Self {
        Self {
            overall_rhythm_score: 0.0,
            rhythm_regularity: 0.0,
            beat_consistency: 0.0,
            alternation_quality: 0.0,
            template_matches: Vec::new(),
            beat_patterns: Vec::new(),
            rhythm_classification: None,
            timing_variance: 0.0,
            rhythmic_complexity: 0.0,
            pattern_entropy: 0.0,
        }
    }
}

impl Default for StressMetrics {
    fn default() -> Self {
        Self {
            overall_stress_score: 0.0,
            pattern_naturalness: 0.0,
            metrical_quality: 0.0,
            placement_accuracy: 0.0,
            prominence_analysis: ProminenceAnalysis::default(),
            metrical_structure: MetricalStructure::default(),
            stress_clashes: Vec::new(),
            foot_analysis: FootAnalysis::default(),
            accent_patterns: Vec::new(),
            prediction_accuracy: None,
        }
    }
}

impl Default for ProminenceAnalysis {
    fn default() -> Self {
        Self {
            primary_stress_positions: Vec::new(),
            secondary_stress_positions: Vec::new(),
            prominence_hierarchy: Vec::new(),
            focus_prominence: Vec::new(),
            contrastive_stress: Vec::new(),
        }
    }
}

impl Default for MetricalStructure {
    fn default() -> Self {
        Self {
            metrical_feet: Vec::new(),
            consistency_score: 0.0,
            dominant_foot_type: FootType::Trochee,
            pattern_regularity: 0.0,
            violations: Vec::new(),
        }
    }
}

impl Default for FootAnalysis {
    fn default() -> Self {
        Self {
            boundary_accuracy: 0.0,
            type_distribution: HashMap::new(),
            average_length: 0.0,
            regularity_score: 0.0,
            complex_patterns: Vec::new(),
        }
    }
}

impl Default for IntonationMetrics {
    fn default() -> Self {
        Self {
            overall_intonation_score: 0.0,
            appropriateness: 0.0,
            contour_analysis: None,
            boundary_tones: None,
            focus_patterns: None,
            sentence_classifications: Vec::new(),
            phrase_structure: None,
            pitch_range_analysis: None,
        }
    }
}

impl Default for TimingMetrics {
    fn default() -> Self {
        Self {
            overall_timing_score: 0.0,
            pause_accuracy: 0.0,
            tempo_analysis: None,
            duration_analysis: None,
            speech_rate_analysis: None,
            variability_analysis: None,
            break_analysis: Vec::new(),
            syllable_timing: None,
        }
    }
}

impl Default for PhonologicalMetrics {
    fn default() -> Self {
        Self {
            overall_phonological_score: 0.0,
            sound_patterns: SoundPatternAnalysis::default(),
            euphony_analysis: None,
            phonotactic_analysis: None,
            syllable_structure: None,
            alliteration: HashMap::new(),
            assonance: HashMap::new(),
            consonance: HashMap::new(),
            rhyme: HashMap::new(),
        }
    }
}

impl Default for SoundPatternAnalysis {
    fn default() -> Self {
        Self {
            pattern_types: Vec::new(),
            frequencies: HashMap::new(),
            strengths: HashMap::new(),
            positions: HashMap::new(),
            pattern_density: 0.0,
        }
    }
}

impl Default for AdvancedProsodicMetrics {
    fn default() -> Self {
        Self {
            overall_advanced_score: 0.0,
            hierarchy_analysis: None,
            complexity_analysis: None,
            entropy_measures: None,
            ml_features: None,
            neural_modeling: None,
            profiling_data: None,
        }
    }
}

// Convenience methods for main result

impl ProsodicFluencyResult {
    /// Calculate overall prosodic fluency score with default weights
    pub fn calculate_overall_score(&mut self) -> f64 {
        let weights = [0.25, 0.20, 0.20, 0.15, 0.20]; // Default weights
        let scores = [
            self.rhythmic_flow,
            self.stress_pattern_naturalness,
            self.intonation_appropriateness,
            self.pause_placement,
            self.reading_ease,
        ];

        let weighted_score: f64 = scores
            .iter()
            .zip(weights.iter())
            .map(|(score, weight)| score * weight)
            .sum();

        self.overall_score = weighted_score;
        weighted_score
    }

    /// Get summary of key prosodic metrics
    pub fn get_summary(&self) -> HashMap<String, f64> {
        let mut summary = HashMap::new();
        summary.insert("overall_score".to_string(), self.overall_score);
        summary.insert("rhythmic_flow".to_string(), self.rhythmic_flow);
        summary.insert(
            "stress_naturalness".to_string(),
            self.stress_pattern_naturalness,
        );
        summary.insert(
            "intonation_appropriateness".to_string(),
            self.intonation_appropriateness,
        );
        summary.insert("pause_placement".to_string(), self.pause_placement);
        summary.insert("reading_ease".to_string(), self.reading_ease);
        summary.insert("syllable_complexity".to_string(), self.syllable_complexity);
        summary
    }

    /// Check if analysis is complete (all optional fields populated)
    pub fn is_complete(&self) -> bool {
        self.rhythm_breakdown.is_some()
            && self.stress_breakdown.is_some()
            && self.intonation_breakdown.is_some()
            && self.timing_breakdown.is_some()
            && self.phonological_breakdown.is_some()
            && self.advanced_breakdown.is_some()
    }

    /// Get breakdown by analysis type
    pub fn get_breakdown_summary(&self) -> HashMap<String, bool> {
        let mut breakdown = HashMap::new();
        breakdown.insert("rhythm".to_string(), self.rhythm_breakdown.is_some());
        breakdown.insert("stress".to_string(), self.stress_breakdown.is_some());
        breakdown.insert(
            "intonation".to_string(),
            self.intonation_breakdown.is_some(),
        );
        breakdown.insert("timing".to_string(), self.timing_breakdown.is_some());
        breakdown.insert(
            "phonological".to_string(),
            self.phonological_breakdown.is_some(),
        );
        breakdown.insert("advanced".to_string(), self.advanced_breakdown.is_some());
        breakdown
    }
}
