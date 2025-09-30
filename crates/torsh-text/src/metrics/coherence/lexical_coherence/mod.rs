//! Modular Lexical Coherence Analysis System
//!
//! This module provides a comprehensive, modular approach to lexical coherence analysis,
//! combining lexical chain building, semantic analysis, and cohesion measurement into
//! a unified system with enhanced configurability and specialized analysis capabilities.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

// Re-export all public types for backward compatibility and external usage
pub use chains::*;
pub use cohesion::*;
pub use config::*;
pub use results::*;
pub use semantic::*;

// Module declarations
pub mod chains;
pub mod cohesion;
pub mod config;
pub mod results;
pub mod semantic;

/// Errors that can occur during modular lexical coherence analysis
#[derive(Error, Debug)]
pub enum ModularLexicalCoherenceError {
    #[error("Chain building error: {0}")]
    ChainBuilding(#[from] ChainBuildingError),
    #[error("Semantic analysis error: {0}")]
    SemanticAnalysis(#[from] SemanticAnalysisError),
    #[error("Cohesion analysis error: {0}")]
    CohesionAnalysis(#[from] CohesionAnalysisError),
    #[error("Configuration error: {0}")]
    Configuration(String),
    #[error("Integration error: {0}")]
    Integration(String),
    #[error("Text preprocessing error: {0}")]
    Preprocessing(String),
    #[error("Analysis orchestration error: {0}")]
    Orchestration(String),
}

/// Main orchestrating analyzer for modular lexical coherence analysis
#[derive(Debug)]
pub struct LexicalCoherenceAnalyzer {
    config: LexicalCoherenceConfig,

    // Specialized analyzers
    chain_builder: LexicalChainBuilder,
    semantic_analyzer: SemanticAnalyzer,
    cohesion_analyzer: CohesionAnalyzer,

    // Text processing components
    text_preprocessor: TextPreprocessor,
    sentence_segmenter: SentenceSegmenter,
    lexical_extractor: LexicalItemExtractor,

    // Analysis orchestration
    integration_manager: IntegrationManager,
    result_synthesizer: ResultSynthesizer,

    // Performance optimization
    analysis_cache: HashMap<String, LexicalCoherenceResult>,
    incremental_processor: IncrementalProcessor,
}

/// Text preprocessing component
#[derive(Debug)]
struct TextPreprocessor {
    normalization_enabled: bool,
    case_folding_enabled: bool,
    punctuation_handling: PunctuationHandling,
    whitespace_normalization: bool,
}

/// Sentence segmentation component
#[derive(Debug)]
struct SentenceSegmenter {
    segmentation_rules: Vec<SegmentationRule>,
    abbreviation_list: Vec<String>,
    custom_boundaries: Vec<String>,
}

/// Lexical item extraction component
#[derive(Debug)]
struct LexicalItemExtractor {
    pos_tagging_enabled: bool,
    lemmatization_enabled: bool,
    frequency_calculation: FrequencyCalculationMethod,
    semantic_feature_extraction: bool,
}

/// Analysis integration manager
#[derive(Debug)]
struct IntegrationManager {
    integration_strategies: Vec<IntegrationStrategy>,
    weight_optimization: WeightOptimization,
    conflict_resolution: ConflictResolution,
}

/// Result synthesis component
#[derive(Debug)]
struct ResultSynthesizer {
    synthesis_methods: Vec<SynthesisMethod>,
    aggregation_functions: HashMap<String, AggregationFunction>,
    insight_generators: Vec<InsightGenerator>,
}

/// Incremental processing for large texts
#[derive(Debug)]
struct IncrementalProcessor {
    chunk_size: usize,
    overlap_size: usize,
    merging_strategy: ChunkMergingStrategy,
    progress_tracking: bool,
}

/// Punctuation handling strategy
#[derive(Debug, Clone)]
enum PunctuationHandling {
    Remove,
    Preserve,
    Normalize,
    ContextDependent,
}

/// Segmentation rule for sentence splitting
#[derive(Debug, Clone)]
struct SegmentationRule {
    pattern: String,
    confidence: f64,
    context_requirements: Vec<String>,
}

/// Frequency calculation method
#[derive(Debug, Clone)]
enum FrequencyCalculationMethod {
    Absolute,
    Relative,
    TfIdf,
    LogFrequency,
}

/// Integration strategy for combining analyses
#[derive(Debug, Clone)]
enum IntegrationStrategy {
    WeightedAverage,
    MaximumConfidence,
    ConsensusVoting,
    HierarchicalIntegration,
}

/// Weight optimization approach
#[derive(Debug, Clone)]
enum WeightOptimization {
    Static,
    Dynamic,
    LearningBased,
    ContextAdaptive,
}

/// Conflict resolution strategy
#[derive(Debug, Clone)]
enum ConflictResolution {
    HighestConfidence,
    ConsensusBuilding,
    EvidenceWeighting,
    UserDefined,
}

/// Synthesis method for result combination
#[derive(Debug, Clone)]
enum SynthesisMethod {
    LinearCombination,
    NonLinearFusion,
    EnsembleApproach,
    GraphBasedIntegration,
}

/// Aggregation function type
type AggregationFunction = fn(&[f64]) -> f64;

/// Insight generation component
#[derive(Debug, Clone)]
struct InsightGenerator {
    name: String,
    generation_function: fn(&LexicalCoherenceResult) -> Vec<String>,
    priority: u8,
}

/// Chunk merging strategy for incremental processing
#[derive(Debug, Clone)]
enum ChunkMergingStrategy {
    Overlapping,
    Sequential,
    HierarchicalMerging,
    SemanticBoundaries,
}

impl LexicalCoherenceAnalyzer {
    /// Create a new modular lexical coherence analyzer
    pub fn new() -> Result<Self, ModularLexicalCoherenceError> {
        Self::with_config(LexicalCoherenceConfig::default())
    }

    /// Create analyzer with custom configuration
    pub fn with_config(
        config: LexicalCoherenceConfig,
    ) -> Result<Self, ModularLexicalCoherenceError> {
        let chain_builder = LexicalChainBuilder::new(config.chains.clone())
            .map_err(|e| ModularLexicalCoherenceError::Configuration(e.to_string()))?;

        let semantic_analyzer = SemanticAnalyzer::new(config.semantic.clone())
            .map_err(|e| ModularLexicalCoherenceError::Configuration(e.to_string()))?;

        let cohesion_analyzer = CohesionAnalyzer::new(config.cohesion.clone())
            .map_err(|e| ModularLexicalCoherenceError::Configuration(e.to_string()))?;

        Ok(LexicalCoherenceAnalyzer {
            config: config.clone(),
            chain_builder,
            semantic_analyzer,
            cohesion_analyzer,
            text_preprocessor: TextPreprocessor::new(&config.general)?,
            sentence_segmenter: SentenceSegmenter::new(&config.general)?,
            lexical_extractor: LexicalItemExtractor::new(&config.general)?,
            integration_manager: IntegrationManager::new(&config.advanced)?,
            result_synthesizer: ResultSynthesizer::new(&config.advanced)?,
            analysis_cache: HashMap::new(),
            incremental_processor: IncrementalProcessor::new(&config.advanced)?,
        })
    }

    /// Perform comprehensive lexical coherence analysis
    pub fn analyze_lexical_coherence(
        &mut self,
        text: &str,
    ) -> Result<LexicalCoherenceResult, ModularLexicalCoherenceError> {
        // Check cache first
        let cache_key = self.generate_cache_key(text);
        if let Some(cached_result) = self.analysis_cache.get(&cache_key) {
            return Ok(cached_result.clone());
        }

        // Step 1: Preprocess text
        let preprocessed_text = self.text_preprocessor.preprocess(text)?;

        // Step 2: Segment into sentences
        let sentences = self.sentence_segmenter.segment(&preprocessed_text)?;

        // Step 3: Extract lexical items
        let lexical_items = self.lexical_extractor.extract(&sentences)?;

        // Step 4: Perform specialized analyses
        let chain_analysis = self.perform_chain_analysis(&lexical_items, &sentences)?;
        let semantic_analysis = self.perform_semantic_analysis(&lexical_items, &sentences)?;
        let cohesion_analysis =
            self.perform_cohesion_analysis(&lexical_items, &sentences, &preprocessed_text)?;

        // Step 5: Integrate analysis results
        let integrated_result = self.integration_manager.integrate_analyses(
            &chain_analysis,
            &semantic_analysis,
            &cohesion_analysis,
        )?;

        // Step 6: Synthesize final result
        let final_result = self.result_synthesizer.synthesize_result(
            &integrated_result,
            &lexical_items,
            &sentences,
            &preprocessed_text,
        )?;

        // Step 7: Cache result
        self.analysis_cache.insert(cache_key, final_result.clone());

        Ok(final_result)
    }

    /// Analyze lexical coherence incrementally for large texts
    pub fn analyze_incrementally(
        &mut self,
        text: &str,
    ) -> Result<LexicalCoherenceResult, ModularLexicalCoherenceError> {
        // Extract chunk size to avoid borrowing conflicts
        let chunk_size = self.incremental_processor.chunk_size;

        if text.len() <= chunk_size {
            return self.analyze_lexical_coherence(text);
        }

        let mut results = Vec::new();
        for chunk in text.as_bytes().chunks(chunk_size) {
            let chunk_str = std::str::from_utf8(chunk).map_err(|e| {
                ModularLexicalCoherenceError::ProcessingError {
                    message: format!("UTF-8 error in chunk: {}", e),
                }
            })?;
            let result = self.analyze_lexical_coherence(chunk_str)?;
            results.push(result);
        }

        // Combine results (simplified version)
        if let Some(first_result) = results.first() {
            Ok(first_result.clone())
        } else {
            Err(ModularLexicalCoherenceError::ProcessingError {
                message: "No chunks processed".to_string(),
            })
        }
    }

    /// Get analysis configuration
    pub fn get_config(&self) -> &LexicalCoherenceConfig {
        &self.config
    }

    /// Update analysis configuration
    pub fn update_config(
        &mut self,
        config: LexicalCoherenceConfig,
    ) -> Result<(), ModularLexicalCoherenceError> {
        // Clear cache when configuration changes
        self.analysis_cache.clear();

        // Update component configurations
        self.chain_builder = LexicalChainBuilder::new(config.chains.clone())
            .map_err(|e| ModularLexicalCoherenceError::Configuration(e.to_string()))?;

        self.semantic_analyzer = SemanticAnalyzer::new(config.semantic.clone())
            .map_err(|e| ModularLexicalCoherenceError::Configuration(e.to_string()))?;

        self.cohesion_analyzer = CohesionAnalyzer::new(config.cohesion.clone())
            .map_err(|e| ModularLexicalCoherenceError::Configuration(e.to_string()))?;

        self.config = config;
        Ok(())
    }

    /// Clear analysis cache
    pub fn clear_cache(&mut self) {
        self.analysis_cache.clear();
    }

    /// Get cache statistics
    pub fn get_cache_stats(&self) -> HashMap<String, usize> {
        let mut stats = HashMap::new();
        stats.insert("cache_size".to_string(), self.analysis_cache.len());
        stats
    }

    // Private helper methods

    fn perform_chain_analysis(
        &mut self,
        lexical_items: &[LexicalItem],
        sentences: &[String],
    ) -> Result<ChainAnalysisResult, ModularLexicalCoherenceError> {
        let chains = self.chain_builder.build_chains(sentences)?;

        Ok(ChainAnalysisResult {
            lexical_chains: chains,
            chain_statistics: self.calculate_chain_statistics(&chains),
            chain_coverage: self.calculate_chain_coverage(&chains, sentences),
            chain_connectivity: self.calculate_chain_connectivity(&chains),
        })
    }

    fn perform_semantic_analysis(
        &mut self,
        lexical_items: &[LexicalItem],
        sentences: &[String],
    ) -> Result<SemanticAnalysisResult, ModularLexicalCoherenceError> {
        self.semantic_analyzer
            .analyze_semantic_relationships(lexical_items, sentences)
            .map_err(ModularLexicalCoherenceError::SemanticAnalysis)
    }

    fn perform_cohesion_analysis(
        &mut self,
        lexical_items: &[LexicalItem],
        sentences: &[String],
        text: &str,
    ) -> Result<CohesionAnalysisResult, ModularLexicalCoherenceError> {
        self.cohesion_analyzer
            .analyze_cohesion(lexical_items, sentences, text)
            .map_err(ModularLexicalCoherenceError::CohesionAnalysis)
    }

    fn calculate_chain_statistics(&self, chains: &[LexicalChain]) -> HashMap<String, f64> {
        let mut stats = HashMap::new();

        if chains.is_empty() {
            return stats;
        }

        // Basic statistics
        stats.insert("total_chains".to_string(), chains.len() as f64);

        let total_words: usize = chains.iter().map(|c| c.words.len()).sum();
        stats.insert("total_chain_words".to_string(), total_words as f64);

        let avg_chain_length = total_words as f64 / chains.len() as f64;
        stats.insert("average_chain_length".to_string(), avg_chain_length);

        // Chain type distribution
        let mut type_counts: HashMap<LexicalChainType, usize> = HashMap::new();
        for chain in chains {
            *type_counts.entry(chain.chain_type.clone()).or_insert(0) += 1;
        }

        for (chain_type, count) in type_counts {
            let type_name = format!("{:?}", chain_type);
            stats.insert(format!("{}_chains", type_name), count as f64);
        }

        // Strength statistics
        let strengths: Vec<f64> = chains.iter().map(|c| c.strength).collect();
        if !strengths.is_empty() {
            stats.insert(
                "min_chain_strength".to_string(),
                strengths.iter().fold(f64::INFINITY, |acc, &x| acc.min(x)),
            );
            stats.insert(
                "max_chain_strength".to_string(),
                strengths
                    .iter()
                    .fold(f64::NEG_INFINITY, |acc, &x| acc.max(x)),
            );
            stats.insert(
                "avg_chain_strength".to_string(),
                strengths.iter().sum::<f64>() / strengths.len() as f64,
            );
        }

        stats
    }

    fn calculate_chain_coverage(&self, chains: &[LexicalChain], sentences: &[String]) -> f64 {
        if chains.is_empty() || sentences.is_empty() {
            return 0.0;
        }

        let total_words: usize = sentences.iter().map(|s| s.split_whitespace().count()).sum();

        let covered_words: usize = chains.iter().map(|chain| chain.words.len()).sum();

        if total_words > 0 {
            covered_words as f64 / total_words as f64
        } else {
            0.0
        }
    }

    fn calculate_chain_connectivity(&self, chains: &[LexicalChain]) -> f64 {
        if chains.len() < 2 {
            return 0.0;
        }

        // Simple connectivity based on overlapping words
        let mut total_connections = 0;
        let mut total_pairs = 0;

        for i in 0..chains.len() {
            for j in (i + 1)..chains.len() {
                let chain1_words: std::collections::HashSet<String> = chains[i]
                    .words
                    .iter()
                    .map(|(word, _)| word.clone())
                    .collect();
                let chain2_words: std::collections::HashSet<String> = chains[j]
                    .words
                    .iter()
                    .map(|(word, _)| word.clone())
                    .collect();

                let intersection = chain1_words.intersection(&chain2_words).count();
                if intersection > 0 {
                    total_connections += intersection;
                }
                total_pairs += 1;
            }
        }

        if total_pairs > 0 {
            total_connections as f64 / total_pairs as f64
        } else {
            0.0
        }
    }

    fn generate_cache_key(&self, text: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        self.config.cache_key().hash(&mut hasher);

        format!("{:x}", hasher.finish())
    }
}

impl Default for LexicalCoherenceAnalyzer {
    fn default() -> Self {
        Self::new().expect("Failed to create default LexicalCoherenceAnalyzer")
    }
}

// Component implementations

impl TextPreprocessor {
    fn new(config: &GeneralLexicalConfig) -> Result<Self, ModularLexicalCoherenceError> {
        Ok(TextPreprocessor {
            normalization_enabled: config.enable_normalization,
            case_folding_enabled: config.case_sensitive,
            punctuation_handling: PunctuationHandling::Normalize,
            whitespace_normalization: true,
        })
    }

    fn preprocess(&self, text: &str) -> Result<String, ModularLexicalCoherenceError> {
        let mut processed = text.to_string();

        // Normalize whitespace
        if self.whitespace_normalization {
            processed = processed.split_whitespace().collect::<Vec<_>>().join(" ");
        }

        // Handle punctuation
        match self.punctuation_handling {
            PunctuationHandling::Normalize => {
                processed = processed.replace("...", ".");
                processed = processed.replace("!!", "!");
                processed = processed.replace("??", "?");
            }
            PunctuationHandling::Remove => {
                processed = processed
                    .chars()
                    .filter(|c| c.is_alphabetic() || c.is_whitespace())
                    .collect();
            }
            _ => {} // Other handling methods not implemented
        }

        // Case folding
        if !self.case_folding_enabled {
            processed = processed.to_lowercase();
        }

        Ok(processed)
    }
}

impl SentenceSegmenter {
    fn new(config: &GeneralLexicalConfig) -> Result<Self, ModularLexicalCoherenceError> {
        Ok(SentenceSegmenter {
            segmentation_rules: vec![SegmentationRule {
                pattern: r"[.!?]+\s+".to_string(),
                confidence: 0.9,
                context_requirements: vec![],
            }],
            abbreviation_list: vec![
                "Dr".to_string(),
                "Mr".to_string(),
                "Mrs".to_string(),
                "Ms".to_string(),
                "Prof".to_string(),
                "etc".to_string(),
                "i.e".to_string(),
                "e.g".to_string(),
            ],
            custom_boundaries: vec![],
        })
    }

    fn segment(&self, text: &str) -> Result<Vec<String>, ModularLexicalCoherenceError> {
        // Simple sentence segmentation
        let sentences: Vec<String> = text
            .split(|c| c == '.' || c == '!' || c == '?')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();

        Ok(sentences)
    }
}

impl LexicalItemExtractor {
    fn new(config: &GeneralLexicalConfig) -> Result<Self, ModularLexicalCoherenceError> {
        Ok(LexicalItemExtractor {
            pos_tagging_enabled: false,
            lemmatization_enabled: config.enable_lemmatization,
            frequency_calculation: FrequencyCalculationMethod::Absolute,
            semantic_feature_extraction: config.enable_semantic_features,
        })
    }

    fn extract(
        &self,
        sentences: &[String],
    ) -> Result<Vec<LexicalItem>, ModularLexicalCoherenceError> {
        let mut lexical_items = Vec::new();
        let mut word_counts: HashMap<String, usize> = HashMap::new();

        // First pass: count word frequencies
        for sentence in sentences {
            for word in sentence.split_whitespace() {
                let clean_word = word
                    .trim_matches(|c: char| !c.is_alphabetic())
                    .to_lowercase();
                if !clean_word.is_empty() && clean_word.len() > 2 {
                    *word_counts.entry(clean_word).or_insert(0) += 1;
                }
            }
        }

        // Second pass: create lexical items
        for (sent_idx, sentence) in sentences.iter().enumerate() {
            let mut char_pos = 0;

            for (word_idx, word) in sentence.split_whitespace().enumerate() {
                let clean_word = word
                    .trim_matches(|c: char| !c.is_alphabetic())
                    .to_lowercase();

                if !clean_word.is_empty() && clean_word.len() > 2 {
                    let frequency = *word_counts.get(&clean_word).unwrap_or(&1) as f64;
                    let lemma = if self.lemmatization_enabled {
                        self.simple_lemmatize(&clean_word)
                    } else {
                        clean_word.clone()
                    };

                    let semantic_features = if self.semantic_feature_extraction {
                        self.extract_semantic_features(&clean_word)
                    } else {
                        vec![]
                    };

                    lexical_items.push(LexicalItem {
                        word: clean_word,
                        lemma,
                        positions: vec![(char_pos, char_pos + word.len())],
                        frequency,
                        word_senses: vec![], // Would be populated by semantic analysis
                        semantic_features,
                    });
                }

                char_pos += word.len() + 1; // +1 for space
            }
        }

        Ok(lexical_items)
    }

    fn simple_lemmatize(&self, word: &str) -> String {
        // Very simple lemmatization - remove common suffixes
        let suffixes = vec!["ing", "ed", "er", "est", "ly", "s"];

        for suffix in suffixes {
            if word.ends_with(suffix) && word.len() > suffix.len() + 2 {
                return word[..word.len() - suffix.len()].to_string();
            }
        }

        word.to_string()
    }

    fn extract_semantic_features(&self, word: &str) -> Vec<String> {
        // Simple semantic feature extraction based on word patterns
        let mut features = vec![];

        // Length-based features
        if word.len() > 8 {
            features.push("long_word".to_string());
        } else if word.len() < 4 {
            features.push("short_word".to_string());
        }

        // Pattern-based features
        if word.ends_with("tion") || word.ends_with("sion") {
            features.push("nominalization".to_string());
        }

        if word.ends_with("ly") {
            features.push("adverbial".to_string());
        }

        if word.ends_with("ing") {
            features.push("gerund_or_participle".to_string());
        }

        features
    }
}

impl IntegrationManager {
    fn new(config: &AdvancedLexicalConfig) -> Result<Self, ModularLexicalCoherenceError> {
        Ok(IntegrationManager {
            integration_strategies: vec![IntegrationStrategy::WeightedAverage],
            weight_optimization: WeightOptimization::Static,
            conflict_resolution: ConflictResolution::HighestConfidence,
        })
    }

    fn integrate_analyses(
        &self,
        chain_analysis: &ChainAnalysisResult,
        semantic_analysis: &SemanticAnalysisResult,
        cohesion_analysis: &CohesionAnalysisResult,
    ) -> Result<IntegratedAnalysisResult, ModularLexicalCoherenceError> {
        // Calculate integrated coherence score
        let chain_weight = 0.4;
        let semantic_weight = 0.35;
        let cohesion_weight = 0.25;

        let chain_score = chain_analysis
            .chain_statistics
            .get("avg_chain_strength")
            .unwrap_or(&0.0);
        let semantic_score = semantic_analysis
            .cohesion_scores
            .get("overall_cohesion")
            .unwrap_or(&0.0);
        let cohesion_score = cohesion_analysis.cohesion_metrics.overall_cohesion;

        let integrated_coherence = chain_score * chain_weight
            + semantic_score * semantic_weight
            + cohesion_score * cohesion_weight;

        // Combine insights
        let mut combined_insights = Vec::new();
        combined_insights.extend(chain_analysis.generate_insights());
        combined_insights.extend(semantic_analysis.generate_insights());
        combined_insights.extend(cohesion_analysis.insights.clone());

        Ok(IntegratedAnalysisResult {
            integrated_coherence_score: integrated_coherence,
            component_contributions: HashMap::from([
                ("chains".to_string(), chain_score * chain_weight),
                ("semantic".to_string(), semantic_score * semantic_weight),
                ("cohesion".to_string(), cohesion_score * cohesion_weight),
            ]),
            cross_component_correlations: self.calculate_correlations(
                chain_analysis,
                semantic_analysis,
                cohesion_analysis,
            ),
            combined_insights,
            integration_metadata: self.generate_integration_metadata(),
        })
    }

    fn calculate_correlations(
        &self,
        _chain_analysis: &ChainAnalysisResult,
        _semantic_analysis: &SemanticAnalysisResult,
        _cohesion_analysis: &CohesionAnalysisResult,
    ) -> HashMap<String, f64> {
        // Placeholder for correlation calculations
        HashMap::from([
            ("chains_semantic".to_string(), 0.7),
            ("chains_cohesion".to_string(), 0.6),
            ("semantic_cohesion".to_string(), 0.8),
        ])
    }

    fn generate_integration_metadata(&self) -> HashMap<String, String> {
        HashMap::from([
            ("integration_version".to_string(), "1.0.0".to_string()),
            ("strategy".to_string(), "weighted_average".to_string()),
            ("optimization".to_string(), "static_weights".to_string()),
        ])
    }
}

impl ResultSynthesizer {
    fn new(config: &AdvancedLexicalConfig) -> Result<Self, ModularLexicalCoherenceError> {
        Ok(ResultSynthesizer {
            synthesis_methods: vec![SynthesisMethod::LinearCombination],
            aggregation_functions: HashMap::from([
                ("mean".to_string(), |values: &[f64]| {
                    values.iter().sum::<f64>() / values.len() as f64
                }),
                ("max".to_string(), |values: &[f64]| {
                    values.iter().fold(f64::NEG_INFINITY, |acc, &x| acc.max(x))
                }),
                ("min".to_string(), |values: &[f64]| {
                    values.iter().fold(f64::INFINITY, |acc, &x| acc.min(x))
                }),
            ]),
            insight_generators: vec![],
        })
    }

    fn synthesize_result(
        &self,
        integrated_result: &IntegratedAnalysisResult,
        lexical_items: &[LexicalItem],
        sentences: &[String],
        text: &str,
    ) -> Result<LexicalCoherenceResult, ModularLexicalCoherenceError> {
        // Create comprehensive lexical coherence result
        Ok(LexicalCoherenceResult {
            overall_coherence_score: integrated_result.integrated_coherence_score,
            lexical_diversity: self.calculate_lexical_diversity(lexical_items),
            chain_coherence_score: *integrated_result
                .component_contributions
                .get("chains")
                .unwrap_or(&0.0),
            semantic_coherence_score: *integrated_result
                .component_contributions
                .get("semantic")
                .unwrap_or(&0.0),
            cohesion_coherence_score: *integrated_result
                .component_contributions
                .get("cohesion")
                .unwrap_or(&0.0),
            detailed_metrics: DetailedLexicalMetrics {
                vocabulary_size: lexical_items.len(),
                unique_lemmas: self.count_unique_lemmas(lexical_items),
                average_word_frequency: self.calculate_average_frequency(lexical_items),
                lexical_density: self.calculate_lexical_density(lexical_items, sentences),
                semantic_density: 0.0, // Would be calculated from semantic analysis
                cohesion_density: 0.0, // Would be calculated from cohesion analysis
                repetition_rate: 0.0,  // Would be calculated from chain analysis
                connectivity_strength: integrated_result
                    .cross_component_correlations
                    .values()
                    .sum::<f64>()
                    / integrated_result.cross_component_correlations.len() as f64,
            },
            insights: integrated_result.combined_insights.clone(),
            recommendations: self.generate_recommendations(&integrated_result),
            analysis_metadata: HashMap::from([
                ("analyzer_version".to_string(), "modular_1.0.0".to_string()),
                ("text_length".to_string(), text.len().to_string()),
                ("sentence_count".to_string(), sentences.len().to_string()),
                (
                    "lexical_item_count".to_string(),
                    lexical_items.len().to_string(),
                ),
            ]),
        })
    }

    fn calculate_lexical_diversity(
        &self,
        lexical_items: &[LexicalItem],
    ) -> LexicalDiversityMetrics {
        let total_words = lexical_items.len() as f64;
        let unique_words = lexical_items
            .iter()
            .map(|item| &item.word)
            .collect::<std::collections::HashSet<_>>()
            .len() as f64;

        let type_token_ratio = if total_words > 0.0 {
            unique_words / total_words
        } else {
            0.0
        };

        // Calculate moving average TTR (MATTR)
        let window_size = 50.min(lexical_items.len());
        let mattr = if lexical_items.len() >= window_size {
            let mut ttr_sum = 0.0;
            let mut window_count = 0;

            for i in 0..=(lexical_items.len() - window_size) {
                let window_words: std::collections::HashSet<&String> = lexical_items
                    [i..i + window_size]
                    .iter()
                    .map(|item| &item.word)
                    .collect();

                let window_ttr = window_words.len() as f64 / window_size as f64;
                ttr_sum += window_ttr;
                window_count += 1;
            }

            if window_count > 0 {
                ttr_sum / window_count as f64
            } else {
                type_token_ratio
            }
        } else {
            type_token_ratio
        };

        LexicalDiversityMetrics {
            type_token_ratio,
            moving_average_ttr: mattr,
            hapax_legomena_ratio: self.calculate_hapax_ratio(lexical_items),
            vocabulary_sophistication: self.calculate_sophistication(lexical_items),
        }
    }

    fn count_unique_lemmas(&self, lexical_items: &[LexicalItem]) -> usize {
        lexical_items
            .iter()
            .map(|item| &item.lemma)
            .collect::<std::collections::HashSet<_>>()
            .len()
    }

    fn calculate_average_frequency(&self, lexical_items: &[LexicalItem]) -> f64 {
        if lexical_items.is_empty() {
            0.0
        } else {
            lexical_items.iter().map(|item| item.frequency).sum::<f64>()
                / lexical_items.len() as f64
        }
    }

    fn calculate_lexical_density(
        &self,
        lexical_items: &[LexicalItem],
        sentences: &[String],
    ) -> f64 {
        let total_words: usize = sentences.iter().map(|s| s.split_whitespace().count()).sum();

        if total_words > 0 {
            lexical_items.len() as f64 / total_words as f64
        } else {
            0.0
        }
    }

    fn calculate_hapax_ratio(&self, lexical_items: &[LexicalItem]) -> f64 {
        let hapax_count = lexical_items
            .iter()
            .filter(|item| item.frequency == 1.0)
            .count() as f64;

        if !lexical_items.is_empty() {
            hapax_count / lexical_items.len() as f64
        } else {
            0.0
        }
    }

    fn calculate_sophistication(&self, lexical_items: &[LexicalItem]) -> f64 {
        // Simple sophistication based on word length and semantic features
        if lexical_items.is_empty() {
            return 0.0;
        }

        let avg_length = lexical_items
            .iter()
            .map(|item| item.word.len() as f64)
            .sum::<f64>()
            / lexical_items.len() as f64;

        let feature_count = lexical_items
            .iter()
            .map(|item| item.semantic_features.len() as f64)
            .sum::<f64>();

        (avg_length / 10.0 + feature_count / lexical_items.len() as f64).min(1.0)
    }

    fn generate_recommendations(
        &self,
        integrated_result: &IntegratedAnalysisResult,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        if integrated_result.integrated_coherence_score < 0.4 {
            recommendations
                .push("Consider adding more transitional phrases to improve coherence".to_string());
            recommendations.push("Review sentence connections and logical flow".to_string());
        }

        if integrated_result
            .component_contributions
            .get("chains")
            .unwrap_or(&0.0)
            < &0.3
        {
            recommendations
                .push("Increase lexical repetition and thematic consistency".to_string());
        }

        if integrated_result
            .component_contributions
            .get("semantic")
            .unwrap_or(&0.0)
            < &0.3
        {
            recommendations.push("Strengthen semantic relationships between concepts".to_string());
        }

        if integrated_result
            .component_contributions
            .get("cohesion")
            .unwrap_or(&0.0)
            < &0.3
        {
            recommendations
                .push("Add more cohesive devices (pronouns, connectors, etc.)".to_string());
        }

        recommendations
    }
}

impl IncrementalProcessor {
    fn new(config: &AdvancedLexicalConfig) -> Result<Self, ModularLexicalCoherenceError> {
        Ok(IncrementalProcessor {
            chunk_size: config.chunk_size.unwrap_or(1000),
            overlap_size: config.overlap_size.unwrap_or(200),
            merging_strategy: ChunkMergingStrategy::Overlapping,
            progress_tracking: config.enable_progress_tracking.unwrap_or(false),
        })
    }

    fn process_incrementally<F>(
        &self,
        text: &str,
        mut analyzer_fn: F,
    ) -> Result<LexicalCoherenceResult, ModularLexicalCoherenceError>
    where
        F: FnMut(&str) -> Result<LexicalCoherenceResult, ModularLexicalCoherenceError>,
    {
        if text.len() <= self.chunk_size {
            return analyzer_fn(text);
        }

        let mut chunk_results = Vec::new();
        let mut start = 0;

        while start < text.len() {
            let end = (start + self.chunk_size).min(text.len());
            let chunk = &text[start..end];

            let result = analyzer_fn(chunk)?;
            chunk_results.push(result);

            start += self.chunk_size - self.overlap_size;

            if start >= text.len() - self.overlap_size {
                break;
            }
        }

        self.merge_chunk_results(chunk_results)
    }

    fn merge_chunk_results(
        &self,
        chunk_results: Vec<LexicalCoherenceResult>,
    ) -> Result<LexicalCoherenceResult, ModularLexicalCoherenceError> {
        if chunk_results.is_empty() {
            return Err(ModularLexicalCoherenceError::Orchestration(
                "No chunk results to merge".to_string(),
            ));
        }

        if chunk_results.len() == 1 {
            return Ok(chunk_results.into_iter().next().unwrap());
        }

        // Simple merging strategy - average the scores
        let overall_coherence = chunk_results
            .iter()
            .map(|r| r.overall_coherence_score)
            .sum::<f64>()
            / chunk_results.len() as f64;

        let chain_coherence = chunk_results
            .iter()
            .map(|r| r.chain_coherence_score)
            .sum::<f64>()
            / chunk_results.len() as f64;

        let semantic_coherence = chunk_results
            .iter()
            .map(|r| r.semantic_coherence_score)
            .sum::<f64>()
            / chunk_results.len() as f64;

        let cohesion_coherence = chunk_results
            .iter()
            .map(|r| r.cohesion_coherence_score)
            .sum::<f64>()
            / chunk_results.len() as f64;

        // Combine insights from all chunks
        let combined_insights: Vec<String> = chunk_results
            .iter()
            .flat_map(|r| r.insights.iter().cloned())
            .collect::<std::collections::HashSet<String>>()
            .into_iter()
            .collect();

        // Combine recommendations
        let combined_recommendations: Vec<String> = chunk_results
            .iter()
            .flat_map(|r| r.recommendations.iter().cloned())
            .collect::<std::collections::HashSet<String>>()
            .into_iter()
            .collect();

        // Average lexical diversity metrics
        let lexical_diversity = self.merge_diversity_metrics(
            &chunk_results
                .iter()
                .map(|r| &r.lexical_diversity)
                .collect::<Vec<_>>(),
        );

        // Sum detailed metrics appropriately
        let detailed_metrics = self.merge_detailed_metrics(
            &chunk_results
                .iter()
                .map(|r| &r.detailed_metrics)
                .collect::<Vec<_>>(),
        );

        Ok(LexicalCoherenceResult {
            overall_coherence_score: overall_coherence,
            lexical_diversity,
            chain_coherence_score: chain_coherence,
            semantic_coherence_score: semantic_coherence,
            cohesion_coherence_score: cohesion_coherence,
            detailed_metrics,
            insights: combined_insights,
            recommendations: combined_recommendations,
            analysis_metadata: HashMap::from([
                ("merged_chunks".to_string(), chunk_results.len().to_string()),
                ("merging_strategy".to_string(), "averaging".to_string()),
            ]),
        })
    }

    fn merge_diversity_metrics(
        &self,
        metrics: &[&LexicalDiversityMetrics],
    ) -> LexicalDiversityMetrics {
        if metrics.is_empty() {
            return LexicalDiversityMetrics {
                type_token_ratio: 0.0,
                moving_average_ttr: 0.0,
                hapax_legomena_ratio: 0.0,
                vocabulary_sophistication: 0.0,
            };
        }

        LexicalDiversityMetrics {
            type_token_ratio: metrics.iter().map(|m| m.type_token_ratio).sum::<f64>()
                / metrics.len() as f64,
            moving_average_ttr: metrics.iter().map(|m| m.moving_average_ttr).sum::<f64>()
                / metrics.len() as f64,
            hapax_legomena_ratio: metrics.iter().map(|m| m.hapax_legomena_ratio).sum::<f64>()
                / metrics.len() as f64,
            vocabulary_sophistication: metrics
                .iter()
                .map(|m| m.vocabulary_sophistication)
                .sum::<f64>()
                / metrics.len() as f64,
        }
    }

    fn merge_detailed_metrics(
        &self,
        metrics: &[&DetailedLexicalMetrics],
    ) -> DetailedLexicalMetrics {
        if metrics.is_empty() {
            return DetailedLexicalMetrics {
                vocabulary_size: 0,
                unique_lemmas: 0,
                average_word_frequency: 0.0,
                lexical_density: 0.0,
                semantic_density: 0.0,
                cohesion_density: 0.0,
                repetition_rate: 0.0,
                connectivity_strength: 0.0,
            };
        }

        DetailedLexicalMetrics {
            vocabulary_size: metrics.iter().map(|m| m.vocabulary_size).sum(),
            unique_lemmas: metrics.iter().map(|m| m.unique_lemmas).sum(),
            average_word_frequency: metrics
                .iter()
                .map(|m| m.average_word_frequency)
                .sum::<f64>()
                / metrics.len() as f64,
            lexical_density: metrics.iter().map(|m| m.lexical_density).sum::<f64>()
                / metrics.len() as f64,
            semantic_density: metrics.iter().map(|m| m.semantic_density).sum::<f64>()
                / metrics.len() as f64,
            cohesion_density: metrics.iter().map(|m| m.cohesion_density).sum::<f64>()
                / metrics.len() as f64,
            repetition_rate: metrics.iter().map(|m| m.repetition_rate).sum::<f64>()
                / metrics.len() as f64,
            connectivity_strength: metrics.iter().map(|m| m.connectivity_strength).sum::<f64>()
                / metrics.len() as f64,
        }
    }
}

// Helper result types for internal processing

#[derive(Debug, Clone)]
struct ChainAnalysisResult {
    lexical_chains: Vec<LexicalChain>,
    chain_statistics: HashMap<String, f64>,
    chain_coverage: f64,
    chain_connectivity: f64,
}

impl ChainAnalysisResult {
    fn generate_insights(&self) -> Vec<String> {
        let mut insights = Vec::new();

        if self.chain_coverage > 0.6 {
            insights.push(
                "High lexical chain coverage indicates strong thematic coherence".to_string(),
            );
        } else if self.chain_coverage < 0.3 {
            insights.push("Low chain coverage suggests weak thematic continuity".to_string());
        }

        if let Some(avg_strength) = self.chain_statistics.get("avg_chain_strength") {
            if *avg_strength > 0.7 {
                insights.push("Strong lexical chains enhance text coherence".to_string());
            }
        }

        insights
    }
}

#[derive(Debug, Clone)]
struct IntegratedAnalysisResult {
    integrated_coherence_score: f64,
    component_contributions: HashMap<String, f64>,
    cross_component_correlations: HashMap<String, f64>,
    combined_insights: Vec<String>,
    integration_metadata: HashMap<String, String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_modular_analyzer_creation() {
        let analyzer = LexicalCoherenceAnalyzer::new();
        assert!(analyzer.is_ok());
    }

    #[test]
    fn test_analyzer_with_custom_config() {
        let config = LexicalCoherenceConfig::comprehensive();
        let analyzer = LexicalCoherenceAnalyzer::with_config(config);
        assert!(analyzer.is_ok());
    }

    #[test]
    fn test_text_preprocessing() {
        let config = GeneralLexicalConfig::default();
        let preprocessor = TextPreprocessor::new(&config).unwrap();

        let input = "This   is  a    test...  Text!!";
        let result = preprocessor.preprocess(input).unwrap();

        assert_eq!(result, "this is a test. text!");
    }

    #[test]
    fn test_sentence_segmentation() {
        let config = GeneralLexicalConfig::default();
        let segmenter = SentenceSegmenter::new(&config).unwrap();

        let input = "First sentence. Second sentence! Third sentence?";
        let sentences = segmenter.segment(input).unwrap();

        assert_eq!(sentences.len(), 3);
        assert_eq!(sentences[0], "First sentence");
        assert_eq!(sentences[1], "Second sentence");
        assert_eq!(sentences[2], "Third sentence");
    }

    #[test]
    fn test_lexical_item_extraction() {
        let config = GeneralLexicalConfig::default();
        let extractor = LexicalItemExtractor::new(&config).unwrap();

        let sentences = vec![
            "The cat sat on the mat".to_string(),
            "The cat was happy".to_string(),
        ];

        let items = extractor.extract(&sentences).unwrap();
        assert!(!items.is_empty());

        // Should extract words like "cat", "sat", etc.
        let cat_items: Vec<&LexicalItem> = items.iter().filter(|item| item.word == "cat").collect();
        assert_eq!(cat_items.len(), 2); // "cat" appears twice
    }

    #[test]
    fn test_cache_functionality() {
        let mut analyzer = LexicalCoherenceAnalyzer::new().unwrap();

        let text = "This is a test text for caching.";

        // First analysis
        let result1 = analyzer.analyze_lexical_coherence(text);
        assert!(result1.is_ok());

        // Second analysis should use cache
        let result2 = analyzer.analyze_lexical_coherence(text);
        assert!(result2.is_ok());

        let stats = analyzer.get_cache_stats();
        assert_eq!(stats.get("cache_size").unwrap(), &1);
    }

    #[test]
    fn test_config_updates() {
        let mut analyzer = LexicalCoherenceAnalyzer::new().unwrap();

        let new_config = LexicalCoherenceConfig::minimal();
        let result = analyzer.update_config(new_config);
        assert!(result.is_ok());

        // Cache should be cleared after config update
        let stats = analyzer.get_cache_stats();
        assert_eq!(stats.get("cache_size").unwrap(), &0);
    }

    #[test]
    fn test_incremental_processing() {
        let mut analyzer = LexicalCoherenceAnalyzer::new().unwrap();

        let long_text = "This is a very long text. ".repeat(100);

        let result = analyzer.analyze_incrementally(&long_text);
        assert!(result.is_ok());

        let final_result = result.unwrap();
        assert!(final_result.analysis_metadata.contains_key("merged_chunks"));
    }

    #[test]
    fn test_comprehensive_analysis() {
        let mut analyzer = LexicalCoherenceAnalyzer::new().unwrap();

        let text = "The cat sat on the mat. The cat was comfortable. \
                   The comfortable cat slept peacefully on the soft mat.";

        let result = analyzer.analyze_lexical_coherence(text).unwrap();

        assert!(result.overall_coherence_score >= 0.0);
        assert!(result.overall_coherence_score <= 1.0);
        assert!(!result.insights.is_empty());
        assert!(!result.recommendations.is_empty());
        assert!(result.detailed_metrics.vocabulary_size > 0);
    }
}
