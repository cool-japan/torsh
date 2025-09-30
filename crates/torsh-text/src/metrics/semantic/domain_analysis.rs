//! Advanced semantic domain analysis and classification system
//!
//! This module provides comprehensive domain classification capabilities for text analysis,
//! supporting multiple classification approaches, domain hierarchies, and context-aware analysis.
//! Designed for production use with extensive configuration options and robust error handling.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{rng, Random};
use std::collections::{BTreeMap, HashMap, HashSet};
use thiserror::Error;

/// Errors that can occur during domain analysis
#[derive(Error, Debug)]
pub enum DomainAnalysisError {
    #[error("Invalid domain configuration: {message}")]
    InvalidConfiguration { message: String },

    #[error("Domain classification failed: {reason}")]
    ClassificationFailed { reason: String },

    #[error("Insufficient training data for domain: {domain}")]
    InsufficientTrainingData { domain: String },

    #[error("Domain hierarchy error: {message}")]
    HierarchyError { message: String },

    #[error("Feature extraction failed: {reason}")]
    FeatureExtractionFailed { reason: String },

    #[error("Model training failed: {error}")]
    ModelTrainingFailed { error: String },

    #[error("Context analysis failed: {reason}")]
    ContextAnalysisFailed { reason: String },
}

/// Domain classification approaches
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DomainApproach {
    /// Rule-based classification using keyword patterns and linguistic rules
    RuleBased,
    /// Statistical classification using frequency analysis and feature vectors
    Statistical,
    /// Keyword-based classification with weighted term matching
    KeywordBased,
    /// Context-aware classification considering surrounding text
    Contextual,
    /// Hierarchical classification with domain trees
    Hierarchical,
    /// Multi-modal classification combining multiple approaches
    MultiModal,
}

/// Predefined semantic domains for classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum SemanticDomain {
    Technical,
    Academic,
    Business,
    Literary,
    Legal,
    Medical,
    Scientific,
    Entertainment,
    News,
    Social,
    Educational,
    Financial,
    Political,
    Sports,
    Art,
    Philosophy,
    History,
    Geography,
    Mathematics,
    Unknown,
}

impl SemanticDomain {
    /// Get all available domains
    pub fn all_domains() -> Vec<SemanticDomain> {
        vec![
            SemanticDomain::Technical,
            SemanticDomain::Academic,
            SemanticDomain::Business,
            SemanticDomain::Literary,
            SemanticDomain::Legal,
            SemanticDomain::Medical,
            SemanticDomain::Scientific,
            SemanticDomain::Entertainment,
            SemanticDomain::News,
            SemanticDomain::Social,
            SemanticDomain::Educational,
            SemanticDomain::Financial,
            SemanticDomain::Political,
            SemanticDomain::Sports,
            SemanticDomain::Art,
            SemanticDomain::Philosophy,
            SemanticDomain::History,
            SemanticDomain::Geography,
            SemanticDomain::Mathematics,
        ]
    }
}

impl std::fmt::Display for SemanticDomain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// Domain classification result with confidence and context
#[derive(Debug, Clone)]
pub struct DomainClassification {
    /// Primary domain classification
    pub primary_domain: SemanticDomain,
    /// Confidence score for primary domain (0.0-1.0)
    pub confidence: f64,
    /// Probability distribution over all domains
    pub domain_probabilities: HashMap<SemanticDomain, f64>,
    /// Secondary domains with significant probability
    pub secondary_domains: Vec<(SemanticDomain, f64)>,
    /// Context features that influenced classification
    pub context_features: Vec<String>,
    /// Classification metadata
    pub metadata: DomainMetadata,
}

/// Metadata about domain classification
#[derive(Debug, Clone)]
pub struct DomainMetadata {
    /// Approach used for classification
    pub approach: DomainApproach,
    /// Analysis timestamp
    pub timestamp: std::time::SystemTime,
    /// Processing time in milliseconds
    pub processing_time_ms: u128,
    /// Number of features analyzed
    pub feature_count: usize,
    /// Quality score of classification (0.0-1.0)
    pub quality_score: f64,
    /// Domain hierarchy depth analyzed
    pub hierarchy_depth: usize,
    /// Context window size used
    pub context_window_size: usize,
}

/// Domain hierarchy for hierarchical classification
#[derive(Debug, Clone)]
pub struct DomainHierarchy {
    /// Root domains and their children
    pub hierarchy: HashMap<SemanticDomain, Vec<SemanticDomain>>,
    /// Domain relationships and similarities
    pub relationships: HashMap<(SemanticDomain, SemanticDomain), f64>,
    /// Domain feature weights
    pub feature_weights: HashMap<SemanticDomain, HashMap<String, f64>>,
}

impl DomainHierarchy {
    /// Create a default domain hierarchy
    pub fn default() -> Self {
        let mut hierarchy = HashMap::new();
        let mut relationships = HashMap::new();
        let mut feature_weights = HashMap::new();

        // Build domain hierarchy
        hierarchy.insert(
            SemanticDomain::Academic,
            vec![
                SemanticDomain::Scientific,
                SemanticDomain::Educational,
                SemanticDomain::Philosophy,
                SemanticDomain::History,
            ],
        );

        hierarchy.insert(
            SemanticDomain::Business,
            vec![SemanticDomain::Financial, SemanticDomain::Legal],
        );

        hierarchy.insert(
            SemanticDomain::Technical,
            vec![SemanticDomain::Mathematics, SemanticDomain::Scientific],
        );

        // Add domain relationships (similarity scores)
        relationships.insert((SemanticDomain::Academic, SemanticDomain::Educational), 0.8);
        relationships.insert((SemanticDomain::Scientific, SemanticDomain::Technical), 0.7);
        relationships.insert((SemanticDomain::Business, SemanticDomain::Financial), 0.9);
        relationships.insert((SemanticDomain::Legal, SemanticDomain::Business), 0.6);

        // Initialize feature weights for major domains
        for domain in SemanticDomain::all_domains() {
            feature_weights.insert(domain, HashMap::new());
        }

        Self {
            hierarchy,
            relationships,
            feature_weights,
        }
    }

    /// Get child domains for a parent domain
    pub fn get_children(&self, domain: &SemanticDomain) -> Option<&Vec<SemanticDomain>> {
        self.hierarchy.get(domain)
    }

    /// Get relationship strength between two domains
    pub fn get_relationship(&self, domain1: &SemanticDomain, domain2: &SemanticDomain) -> f64 {
        self.relationships
            .get(&(*domain1, *domain2))
            .or_else(|| self.relationships.get(&(*domain2, *domain1)))
            .copied()
            .unwrap_or(0.0)
    }
}

/// Configuration for domain analysis
#[derive(Debug, Clone)]
pub struct DomainAnalysisConfig {
    /// Primary classification approach
    pub approach: DomainApproach,
    /// Secondary approaches to combine (for MultiModal)
    pub secondary_approaches: Vec<DomainApproach>,
    /// Minimum confidence threshold for classification
    pub confidence_threshold: f64,
    /// Maximum number of secondary domains to report
    pub max_secondary_domains: usize,
    /// Context window size for contextual analysis
    pub context_window_size: usize,
    /// Enable hierarchical domain relationships
    pub use_hierarchy: bool,
    /// Feature extraction depth
    pub feature_depth: usize,
    /// Enable domain adaptation
    pub enable_adaptation: bool,
    /// Custom domain hierarchy
    pub custom_hierarchy: Option<DomainHierarchy>,
}

impl Default for DomainAnalysisConfig {
    fn default() -> Self {
        Self {
            approach: DomainApproach::MultiModal,
            secondary_approaches: vec![
                DomainApproach::Statistical,
                DomainApproach::KeywordBased,
                DomainApproach::Contextual,
            ],
            confidence_threshold: 0.3,
            max_secondary_domains: 3,
            context_window_size: 100,
            use_hierarchy: true,
            feature_depth: 3,
            enable_adaptation: false,
            custom_hierarchy: None,
        }
    }
}

impl DomainAnalysisConfig {
    /// Create a new configuration builder
    pub fn builder() -> DomainAnalysisConfigBuilder {
        DomainAnalysisConfigBuilder::new()
    }
}

/// Builder for domain analysis configuration
pub struct DomainAnalysisConfigBuilder {
    config: DomainAnalysisConfig,
}

impl DomainAnalysisConfigBuilder {
    pub fn new() -> Self {
        Self {
            config: DomainAnalysisConfig::default(),
        }
    }

    pub fn approach(mut self, approach: DomainApproach) -> Self {
        self.config.approach = approach;
        self
    }

    pub fn add_secondary_approach(mut self, approach: DomainApproach) -> Self {
        self.config.secondary_approaches.push(approach);
        self
    }

    pub fn confidence_threshold(mut self, threshold: f64) -> Self {
        self.config.confidence_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    pub fn max_secondary_domains(mut self, max: usize) -> Self {
        self.config.max_secondary_domains = max;
        self
    }

    pub fn context_window_size(mut self, size: usize) -> Self {
        self.config.context_window_size = size;
        self
    }

    pub fn use_hierarchy(mut self, use_hierarchy: bool) -> Self {
        self.config.use_hierarchy = use_hierarchy;
        self
    }

    pub fn feature_depth(mut self, depth: usize) -> Self {
        self.config.feature_depth = depth.max(1);
        self
    }

    pub fn enable_adaptation(mut self, enable: bool) -> Self {
        self.config.enable_adaptation = enable;
        self
    }

    pub fn custom_hierarchy(mut self, hierarchy: DomainHierarchy) -> Self {
        self.config.custom_hierarchy = Some(hierarchy);
        self
    }

    pub fn build(self) -> Result<DomainAnalysisConfig, DomainAnalysisError> {
        if self.config.confidence_threshold < 0.0 || self.config.confidence_threshold > 1.0 {
            return Err(DomainAnalysisError::InvalidConfiguration {
                message: "Confidence threshold must be between 0.0 and 1.0".to_string(),
            });
        }

        if self.config.context_window_size == 0 {
            return Err(DomainAnalysisError::InvalidConfiguration {
                message: "Context window size must be greater than 0".to_string(),
            });
        }

        Ok(self.config)
    }
}

/// Advanced domain analysis engine
pub struct DomainAnalyzer {
    config: DomainAnalysisConfig,
    hierarchy: DomainHierarchy,
    feature_extractors: HashMap<DomainApproach, Box<dyn FeatureExtractor>>,
    domain_models: HashMap<SemanticDomain, DomainModel>,
    adaptation_data: HashMap<SemanticDomain, Vec<String>>,
}

/// Feature extractor trait for different approaches
trait FeatureExtractor: Send + Sync {
    fn extract_features(
        &self,
        text: &str,
        context_size: usize,
    ) -> Result<Vec<String>, DomainAnalysisError>;
}

/// Rule-based feature extractor
struct RuleBasedExtractor;

impl FeatureExtractor for RuleBasedExtractor {
    fn extract_features(
        &self,
        text: &str,
        context_size: usize,
    ) -> Result<Vec<String>, DomainAnalysisError> {
        let mut features = Vec::new();
        let words: Vec<&str> = text.split_whitespace().collect();
        let window_size = context_size.min(words.len());

        // Extract linguistic patterns and keywords
        for window in words.windows(window_size) {
            let context = window.join(" ");

            // Technical domain patterns
            if context.contains("algorithm")
                || context.contains("implementation")
                || context.contains("function")
                || context.contains("variable")
            {
                features.push("technical_pattern".to_string());
            }

            // Academic domain patterns
            if context.contains("research")
                || context.contains("study")
                || context.contains("analysis")
                || context.contains("methodology")
            {
                features.push("academic_pattern".to_string());
            }

            // Business domain patterns
            if context.contains("revenue")
                || context.contains("market")
                || context.contains("strategy")
                || context.contains("customer")
            {
                features.push("business_pattern".to_string());
            }

            // Legal domain patterns
            if context.contains("contract")
                || context.contains("legal")
                || context.contains("court")
                || context.contains("law")
            {
                features.push("legal_pattern".to_string());
            }

            // Medical domain patterns
            if context.contains("patient")
                || context.contains("treatment")
                || context.contains("diagnosis")
                || context.contains("medical")
            {
                features.push("medical_pattern".to_string());
            }
        }

        Ok(features)
    }
}

/// Statistical feature extractor
struct StatisticalExtractor;

impl FeatureExtractor for StatisticalExtractor {
    fn extract_features(
        &self,
        text: &str,
        context_size: usize,
    ) -> Result<Vec<String>, DomainAnalysisError> {
        let mut features = Vec::new();
        let words: Vec<&str> = text.split_whitespace().collect();

        // Calculate statistical features
        let avg_word_length =
            words.iter().map(|w| w.len()).sum::<usize>() as f64 / words.len() as f64;
        let unique_words = words.iter().collect::<HashSet<_>>().len();
        let vocabulary_density = unique_words as f64 / words.len() as f64;

        features.push(format!("avg_word_length_{:.1}", avg_word_length));
        features.push(format!("vocabulary_density_{:.2}", vocabulary_density));
        features.push(format!("text_length_{}", words.len()));

        // N-gram features
        for n in 1..=3 {
            for ngram in words.windows(n) {
                if ngram.len() == n {
                    features.push(format!("ngram_{}_{}", n, ngram.join("_")));
                }
            }
        }

        Ok(features)
    }
}

/// Keyword-based feature extractor
struct KeywordExtractor;

impl FeatureExtractor for KeywordExtractor {
    fn extract_features(
        &self,
        text: &str,
        _context_size: usize,
    ) -> Result<Vec<String>, DomainAnalysisError> {
        let mut features = Vec::new();
        let text_lower = text.to_lowercase();

        // Define domain-specific keywords
        let domain_keywords = vec![
            (
                SemanticDomain::Technical,
                vec!["code", "software", "program", "algorithm", "data", "system"],
            ),
            (
                SemanticDomain::Academic,
                vec![
                    "research",
                    "study",
                    "theory",
                    "hypothesis",
                    "experiment",
                    "scholarly",
                ],
            ),
            (
                SemanticDomain::Business,
                vec![
                    "company", "market", "revenue", "profit", "customer", "strategy",
                ],
            ),
            (
                SemanticDomain::Legal,
                vec!["law", "court", "judge", "contract", "legal", "attorney"],
            ),
            (
                SemanticDomain::Medical,
                vec![
                    "patient",
                    "doctor",
                    "treatment",
                    "disease",
                    "medicine",
                    "health",
                ],
            ),
            (
                SemanticDomain::Scientific,
                vec![
                    "experiment",
                    "hypothesis",
                    "theory",
                    "data",
                    "result",
                    "method",
                ],
            ),
        ];

        for (domain, keywords) in domain_keywords {
            let mut keyword_count = 0;
            for keyword in keywords {
                if text_lower.contains(keyword) {
                    keyword_count += 1;
                    features.push(format!("keyword_{}_{}", domain, keyword));
                }
            }
            if keyword_count > 0 {
                features.push(format!("domain_keyword_count_{}_{}", domain, keyword_count));
            }
        }

        Ok(features)
    }
}

/// Contextual feature extractor
struct ContextualExtractor;

impl FeatureExtractor for ContextualExtractor {
    fn extract_features(
        &self,
        text: &str,
        context_size: usize,
    ) -> Result<Vec<String>, DomainAnalysisError> {
        let mut features = Vec::new();
        let sentences: Vec<&str> = text.split(|c| c == '.' || c == '!' || c == '?').collect();

        // Analyze sentence structure and context
        for sentence in sentences.iter().filter(|s| !s.trim().is_empty()) {
            let words: Vec<&str> = sentence.split_whitespace().collect();
            if words.len() < 3 {
                continue;
            }

            // Sentence complexity
            let complexity = words.len() as f64 / sentence.matches(',').count().max(1) as f64;
            features.push(format!("sentence_complexity_{:.1}", complexity));

            // Context patterns
            let context = words
                .iter()
                .take(context_size)
                .cloned()
                .collect::<Vec<_>>()
                .join(" ");

            // Formal vs informal context
            if context.contains("therefore")
                || context.contains("furthermore")
                || context.contains("consequently")
                || context.contains("moreover")
            {
                features.push("formal_context".to_string());
            }

            // Question patterns
            if sentence.contains('?') {
                features.push("question_context".to_string());
            }

            // Citation patterns
            if sentence.contains("according to")
                || sentence.contains("et al.")
                || sentence.contains("(")
            {
                features.push("citation_context".to_string());
            }
        }

        Ok(features)
    }
}

/// Domain model for classification
#[derive(Debug, Clone)]
struct DomainModel {
    domain: SemanticDomain,
    feature_weights: HashMap<String, f64>,
    bias: f64,
    confidence_adjustments: HashMap<String, f64>,
}

impl DomainModel {
    fn new(domain: SemanticDomain) -> Self {
        Self {
            domain,
            feature_weights: HashMap::new(),
            bias: 0.0,
            confidence_adjustments: HashMap::new(),
        }
    }

    fn predict(&self, features: &[String]) -> f64 {
        let mut score = self.bias;

        for feature in features {
            if let Some(weight) = self.feature_weights.get(feature) {
                score += weight;
            }
        }

        // Apply sigmoid activation
        1.0 / (1.0 + (-score).exp())
    }

    fn get_confidence_adjustment(&self, features: &[String]) -> f64 {
        let mut adjustment = 0.0;

        for feature in features {
            if let Some(adj) = self.confidence_adjustments.get(feature) {
                adjustment += adj;
            }
        }

        adjustment.tanh() // Bounded adjustment
    }
}

impl DomainAnalyzer {
    /// Create a new domain analyzer with the given configuration
    pub fn new(config: DomainAnalysisConfig) -> Result<Self, DomainAnalysisError> {
        let hierarchy = config
            .custom_hierarchy
            .clone()
            .unwrap_or_else(DomainHierarchy::default);

        let mut feature_extractors: HashMap<DomainApproach, Box<dyn FeatureExtractor>> =
            HashMap::new();
        feature_extractors.insert(DomainApproach::RuleBased, Box::new(RuleBasedExtractor));
        feature_extractors.insert(DomainApproach::Statistical, Box::new(StatisticalExtractor));
        feature_extractors.insert(DomainApproach::KeywordBased, Box::new(KeywordExtractor));
        feature_extractors.insert(DomainApproach::Contextual, Box::new(ContextualExtractor));

        // Initialize domain models
        let mut domain_models = HashMap::new();
        for domain in SemanticDomain::all_domains() {
            domain_models.insert(domain.clone(), DomainModel::new(domain));
        }

        Ok(Self {
            config,
            hierarchy,
            feature_extractors,
            domain_models,
            adaptation_data: HashMap::new(),
        })
    }

    /// Create a domain analyzer with default configuration
    pub fn default() -> Result<Self, DomainAnalysisError> {
        Self::new(DomainAnalysisConfig::default())
    }

    /// Classify text into semantic domains
    pub fn classify_domain(
        &mut self,
        text: &str,
    ) -> Result<DomainClassification, DomainAnalysisError> {
        if text.trim().is_empty() {
            return Err(DomainAnalysisError::ClassificationFailed {
                reason: "Empty text provided".to_string(),
            });
        }

        let start_time = std::time::Instant::now();

        // Extract features based on the primary approach
        let features = self.extract_features_for_approach(&self.config.approach, text)?;

        // Get domain probabilities
        let mut domain_probabilities = self.calculate_domain_probabilities(&features)?;

        // Apply hierarchical adjustments if enabled
        if self.config.use_hierarchy {
            domain_probabilities = self.apply_hierarchical_adjustments(domain_probabilities)?;
        }

        // Apply multi-modal combination if configured
        if self.config.approach == DomainApproach::MultiModal {
            domain_probabilities =
                self.apply_multi_modal_combination(text, domain_probabilities)?;
        }

        // Determine primary domain
        let (primary_domain, confidence) = domain_probabilities
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(domain, prob)| (domain.clone(), *prob))
            .unwrap_or((SemanticDomain::Unknown, 0.0));

        // Get secondary domains
        let mut secondary_domains: Vec<(SemanticDomain, f64)> = domain_probabilities
            .iter()
            .filter(|(domain, prob)| {
                **domain != primary_domain && **prob >= self.config.confidence_threshold
            })
            .map(|(domain, prob)| (domain.clone(), *prob))
            .collect();

        secondary_domains.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        secondary_domains.truncate(self.config.max_secondary_domains);

        // Extract context features
        let context_features = self.extract_context_features(text, &features)?;

        // Calculate quality score
        let quality_score = self.calculate_quality_score(&domain_probabilities, &features);

        let processing_time = start_time.elapsed();

        let metadata = DomainMetadata {
            approach: self.config.approach,
            timestamp: std::time::SystemTime::now(),
            processing_time_ms: processing_time.as_millis(),
            feature_count: features.len(),
            quality_score,
            hierarchy_depth: if self.config.use_hierarchy { 3 } else { 1 },
            context_window_size: self.config.context_window_size,
        };

        // Store adaptation data if enabled
        if self.config.enable_adaptation {
            self.store_adaptation_data(&primary_domain, text.to_string());
        }

        Ok(DomainClassification {
            primary_domain,
            confidence,
            domain_probabilities,
            secondary_domains,
            context_features,
            metadata,
        })
    }

    /// Extract features using a specific approach
    fn extract_features_for_approach(
        &self,
        approach: &DomainApproach,
        text: &str,
    ) -> Result<Vec<String>, DomainAnalysisError> {
        if let Some(extractor) = self.feature_extractors.get(approach) {
            extractor.extract_features(text, self.config.context_window_size)
        } else {
            Err(DomainAnalysisError::FeatureExtractionFailed {
                reason: format!(
                    "No feature extractor available for approach: {:?}",
                    approach
                ),
            })
        }
    }

    /// Calculate domain probabilities from features
    fn calculate_domain_probabilities(
        &self,
        features: &[String],
    ) -> Result<HashMap<SemanticDomain, f64>, DomainAnalysisError> {
        let mut probabilities = HashMap::new();

        for (domain, model) in &self.domain_models {
            let base_probability = model.predict(features);
            let confidence_adjustment = model.get_confidence_adjustment(features);
            let final_probability = (base_probability + confidence_adjustment).clamp(0.0, 1.0);

            probabilities.insert(domain.clone(), final_probability);
        }

        // Normalize probabilities
        let total: f64 = probabilities.values().sum();
        if total > 0.0 {
            for prob in probabilities.values_mut() {
                *prob /= total;
            }
        }

        Ok(probabilities)
    }

    /// Apply hierarchical adjustments to domain probabilities
    fn apply_hierarchical_adjustments(
        &self,
        mut probabilities: HashMap<SemanticDomain, f64>,
    ) -> Result<HashMap<SemanticDomain, f64>, DomainAnalysisError> {
        // Boost related domains based on hierarchy relationships
        let relationships: Vec<_> = probabilities.keys().copied().collect();

        for domain1 in &relationships {
            for domain2 in &relationships {
                if domain1 != domain2 {
                    let relationship_strength = self.hierarchy.get_relationship(domain1, domain2);
                    if relationship_strength > 0.5 {
                        let boost = probabilities[domain1] * relationship_strength * 0.1;
                        if let Some(prob) = probabilities.get_mut(domain2) {
                            *prob += boost;
                        }
                    }
                }
            }
        }

        // Re-normalize after adjustments
        let total: f64 = probabilities.values().sum();
        if total > 0.0 {
            for prob in probabilities.values_mut() {
                *prob /= total;
            }
        }

        Ok(probabilities)
    }

    /// Apply multi-modal combination of approaches
    fn apply_multi_modal_combination(
        &mut self,
        text: &str,
        base_probabilities: HashMap<SemanticDomain, f64>,
    ) -> Result<HashMap<SemanticDomain, f64>, DomainAnalysisError> {
        let mut combined_probabilities = base_probabilities.clone();
        let mut total_weight = 1.0;

        // Combine results from secondary approaches
        for approach in &self.config.secondary_approaches {
            if *approach == self.config.approach {
                continue;
            }

            let features = self.extract_features_for_approach(approach, text)?;
            let approach_probabilities = self.calculate_domain_probabilities(&features)?;

            let approach_weight = match approach {
                DomainApproach::Statistical => 0.8,
                DomainApproach::KeywordBased => 0.9,
                DomainApproach::Contextual => 0.7,
                DomainApproach::RuleBased => 0.6,
                _ => 0.5,
            };

            total_weight += approach_weight;

            // Weighted combination
            for (domain, prob) in approach_probabilities {
                if let Some(combined_prob) = combined_probabilities.get_mut(&domain) {
                    *combined_prob += prob * approach_weight;
                }
            }
        }

        // Normalize by total weight
        for prob in combined_probabilities.values_mut() {
            *prob /= total_weight;
        }

        Ok(combined_probabilities)
    }

    /// Extract context features for analysis
    fn extract_context_features(
        &self,
        text: &str,
        features: &[String],
    ) -> Result<Vec<String>, DomainAnalysisError> {
        let mut context_features = Vec::new();

        // Text length category
        let word_count = text.split_whitespace().count();
        context_features.push(match word_count {
            0..=50 => "short_text".to_string(),
            51..=200 => "medium_text".to_string(),
            201..=500 => "long_text".to_string(),
            _ => "very_long_text".to_string(),
        });

        // Feature diversity
        let unique_feature_types = features
            .iter()
            .map(|f| f.split('_').next().unwrap_or(""))
            .collect::<HashSet<_>>()
            .len();

        context_features.push(format!("feature_diversity_{}", unique_feature_types));

        // Formal vs informal indicators
        if text.contains("therefore") || text.contains("consequently") {
            context_features.push("formal_language".to_string());
        }
        if text.contains("gonna") || text.contains("kinda") {
            context_features.push("informal_language".to_string());
        }

        Ok(context_features)
    }

    /// Calculate quality score for the classification
    fn calculate_quality_score(
        &self,
        probabilities: &HashMap<SemanticDomain, f64>,
        features: &[String],
    ) -> f64 {
        // Base quality on confidence distribution
        let max_prob = probabilities.values().copied().fold(0.0f64, f64::max);
        let entropy = -probabilities
            .values()
            .filter(|&&p| p > 0.0)
            .map(|&p| p * p.ln())
            .sum::<f64>();

        let confidence_quality = max_prob;
        let uncertainty_penalty = entropy / (SemanticDomain::all_domains().len() as f64).ln();
        let feature_quality = (features.len() as f64 / 100.0).min(1.0);

        (confidence_quality * 0.5 + (1.0 - uncertainty_penalty) * 0.3 + feature_quality * 0.2)
            .clamp(0.0, 1.0)
    }

    /// Store adaptation data for future learning
    fn store_adaptation_data(&mut self, domain: &SemanticDomain, text: String) {
        self.adaptation_data
            .entry(domain.clone())
            .or_insert_with(Vec::new)
            .push(text);

        // Keep only recent data (max 1000 samples per domain)
        if let Some(samples) = self.adaptation_data.get_mut(domain) {
            if samples.len() > 1000 {
                samples.drain(0..samples.len() - 1000);
            }
        }
    }

    /// Get domain hierarchy information
    pub fn get_hierarchy(&self) -> &DomainHierarchy {
        &self.hierarchy
    }

    /// Update configuration
    pub fn update_config(&mut self, config: DomainAnalysisConfig) {
        self.config = config;
    }

    /// Get adaptation statistics
    pub fn get_adaptation_stats(&self) -> HashMap<SemanticDomain, usize> {
        self.adaptation_data
            .iter()
            .map(|(domain, samples)| (domain.clone(), samples.len()))
            .collect()
    }
}

/// Convenience function for simple domain classification
pub fn classify_text_domain(text: &str) -> Result<DomainClassification, DomainAnalysisError> {
    let mut analyzer = DomainAnalyzer::default()?;
    analyzer.classify_domain(text)
}

/// Convenience function for domain classification with custom config
pub fn classify_text_domain_with_config(
    text: &str,
    config: DomainAnalysisConfig,
) -> Result<DomainClassification, DomainAnalysisError> {
    let mut analyzer = DomainAnalyzer::new(config)?;
    analyzer.classify_domain(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_domain_analyzer_creation() {
        let analyzer = DomainAnalyzer::default();
        assert!(analyzer.is_ok());
    }

    #[test]
    fn test_technical_domain_classification() {
        let mut analyzer = DomainAnalyzer::default().unwrap();
        let text = "This algorithm implements a hash table with linear probing collision resolution. The function takes a key-value pair and stores it in the data structure.";

        let result = analyzer.classify_domain(text).unwrap();
        assert_eq!(result.primary_domain, SemanticDomain::Technical);
        assert!(result.confidence > 0.3);
    }

    #[test]
    fn test_academic_domain_classification() {
        let mut analyzer = DomainAnalyzer::default().unwrap();
        let text = "This research study investigates the hypothesis that cognitive load affects learning outcomes. The methodology involves controlled experiments with participants.";

        let result = analyzer.classify_domain(text).unwrap();
        assert_eq!(result.primary_domain, SemanticDomain::Academic);
        assert!(result.confidence > 0.3);
    }

    #[test]
    fn test_business_domain_classification() {
        let mut analyzer = DomainAnalyzer::default().unwrap();
        let text = "The company's revenue grew by 15% this quarter due to improved customer acquisition strategies and market expansion initiatives.";

        let result = analyzer.classify_domain(text).unwrap();
        assert_eq!(result.primary_domain, SemanticDomain::Business);
        assert!(result.confidence > 0.3);
    }

    #[test]
    fn test_medical_domain_classification() {
        let mut analyzer = DomainAnalyzer::default().unwrap();
        let text = "The patient presented with symptoms of acute myocardial infarction. Treatment included immediate administration of thrombolytic therapy and cardiac monitoring.";

        let result = analyzer.classify_domain(text).unwrap();
        assert_eq!(result.primary_domain, SemanticDomain::Medical);
        assert!(result.confidence > 0.3);
    }

    #[test]
    fn test_empty_text_error() {
        let mut analyzer = DomainAnalyzer::default().unwrap();
        let result = analyzer.classify_domain("");
        assert!(result.is_err());
        match result.unwrap_err() {
            DomainAnalysisError::ClassificationFailed { .. } => {}
            _ => panic!("Expected ClassificationFailed error"),
        }
    }

    #[test]
    fn test_configuration_builder() {
        let config = DomainAnalysisConfig::builder()
            .approach(DomainApproach::Statistical)
            .confidence_threshold(0.5)
            .max_secondary_domains(2)
            .context_window_size(50)
            .use_hierarchy(false)
            .feature_depth(2)
            .enable_adaptation(true)
            .build();

        assert!(config.is_ok());
        let config = config.unwrap();
        assert_eq!(config.approach, DomainApproach::Statistical);
        assert_eq!(config.confidence_threshold, 0.5);
        assert_eq!(config.max_secondary_domains, 2);
    }

    #[test]
    fn test_invalid_configuration() {
        let config = DomainAnalysisConfig::builder()
            .confidence_threshold(1.5) // Invalid: > 1.0
            .build();

        assert!(config.is_err());
        match config.unwrap_err() {
            DomainAnalysisError::InvalidConfiguration { .. } => {}
            _ => panic!("Expected InvalidConfiguration error"),
        }
    }

    #[test]
    fn test_domain_hierarchy() {
        let hierarchy = DomainHierarchy::default();
        let children = hierarchy.get_children(&SemanticDomain::Academic);
        assert!(children.is_some());
        assert!(children.unwrap().contains(&SemanticDomain::Scientific));

        let relationship =
            hierarchy.get_relationship(&SemanticDomain::Academic, &SemanticDomain::Educational);
        assert!(relationship > 0.0);
    }

    #[test]
    fn test_multi_modal_approach() {
        let config = DomainAnalysisConfig::builder()
            .approach(DomainApproach::MultiModal)
            .add_secondary_approach(DomainApproach::Statistical)
            .add_secondary_approach(DomainApproach::KeywordBased)
            .build()
            .unwrap();

        let mut analyzer = DomainAnalyzer::new(config).unwrap();
        let text = "The software engineering process involves systematic design and implementation of complex algorithms.";

        let result = analyzer.classify_domain(text).unwrap();
        assert!(result.quality_score > 0.0);
        assert!(!result.secondary_domains.is_empty());
    }

    #[test]
    fn test_convenience_functions() {
        let text =
            "Legal contract terms and conditions require careful review by qualified attorneys.";

        let result = classify_text_domain(text);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().primary_domain, SemanticDomain::Legal);

        let config = DomainAnalysisConfig::builder()
            .approach(DomainApproach::KeywordBased)
            .build()
            .unwrap();

        let result = classify_text_domain_with_config(text, config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_adaptation_data_storage() {
        let config = DomainAnalysisConfig::builder()
            .enable_adaptation(true)
            .build()
            .unwrap();

        let mut analyzer = DomainAnalyzer::new(config).unwrap();
        let text = "Scientific method involves hypothesis formation and experimental validation.";

        let _result = analyzer.classify_domain(text).unwrap();
        let stats = analyzer.get_adaptation_stats();
        assert!(!stats.is_empty());
    }

    #[test]
    fn test_quality_score_calculation() {
        let mut analyzer = DomainAnalyzer::default().unwrap();
        let high_quality_text = "Advanced machine learning algorithms utilize sophisticated mathematical optimization techniques for training neural networks with backpropagation.";
        let low_quality_text = "The thing is good.";

        let high_quality_result = analyzer.classify_domain(high_quality_text).unwrap();
        let low_quality_result = analyzer.classify_domain(low_quality_text).unwrap();

        assert!(
            high_quality_result.metadata.quality_score > low_quality_result.metadata.quality_score
        );
    }

    #[test]
    fn test_context_features_extraction() {
        let mut analyzer = DomainAnalyzer::default().unwrap();
        let formal_text = "Therefore, we can consequently conclude that the hypothesis is supported by empirical evidence.";

        let result = analyzer.classify_domain(formal_text).unwrap();
        assert!(result
            .context_features
            .contains(&"formal_language".to_string()));
    }
}
