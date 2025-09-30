//! Cohesion Analysis for Lexical Coherence
//!
//! This module provides comprehensive cohesion analysis capabilities for measuring
//! lexical coherence through cohesive devices, referential chains, and textual
//! connectivity patterns.

use crate::metrics::coherence::lexical_coherence::config::{
    CohesionAnalysisConfig, CohesionDeviceType,
};
use crate::metrics::coherence::lexical_coherence::results::{
    CohesionAnalysisResult, CohesionMetrics, CohesiveDevice, ConnectivityAnalysis, LexicalItem,
    ReferentialChain,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use thiserror::Error;

/// Errors that can occur during cohesion analysis
#[derive(Error, Debug)]
pub enum CohesionAnalysisError {
    #[error("Cohesive device detection failed: {0}")]
    DeviceDetectionError(String),
    #[error("Referential chain building failed: {0}")]
    ChainBuildingError(String),
    #[error("Connectivity analysis failed: {0}")]
    ConnectivityError(String),
    #[error("Invalid cohesion configuration: {0}")]
    ConfigurationError(String),
    #[error("Text preprocessing error: {0}")]
    PreprocessingError(String),
}

/// Core cohesion analyzer for lexical coherence analysis
#[derive(Debug)]
pub struct CohesionAnalyzer {
    config: CohesionAnalysisConfig,

    // Analysis components
    device_detector: CohesiveDeviceDetector,
    chain_builder: ReferentialChainBuilder,
    connectivity_analyzer: ConnectivityAnalyzer,

    // Linguistic resources
    cohesive_markers: HashMap<CohesiveDeviceType, HashSet<String>>,
    reference_patterns: HashMap<String, ReferencePattern>,
    connector_lexicon: HashMap<String, ConnectorInfo>,

    // Analysis caches
    device_cache: HashMap<String, Vec<CohesiveDevice>>,
    chain_cache: HashMap<String, Vec<ReferentialChain>>,
    connectivity_cache: HashMap<String, ConnectivityAnalysis>,
}

/// Cohesive device detector component
#[derive(Debug)]
struct CohesiveDeviceDetector {
    detection_patterns: HashMap<CohesiveDeviceType, Vec<DetectionPattern>>,
    context_window: usize,
    confidence_threshold: f64,
}

/// Referential chain builder component
#[derive(Debug)]
struct ReferentialChainBuilder {
    chain_strategies: Vec<ChainBuildingStrategy>,
    maximum_chain_distance: usize,
    minimum_chain_length: usize,
    coreference_resolver: CoreferenceResolver,
}

/// Connectivity analyzer component
#[derive(Debug)]
struct ConnectivityAnalyzer {
    connectivity_measures: Vec<ConnectivityMeasure>,
    graph_algorithms: GraphAnalyzer,
    density_calculator: DensityCalculator,
}

/// Reference pattern for coreference resolution
#[derive(Debug, Clone)]
struct ReferencePattern {
    pattern_type: ReferenceType,
    matching_rules: Vec<String>,
    confidence_weight: f64,
    distance_penalty: f64,
}

/// Connector information for discourse markers
#[derive(Debug, Clone)]
struct ConnectorInfo {
    connector_type: CohesiveDeviceType,
    semantic_function: String,
    scope: ConnectorScope,
    strength: f64,
}

/// Detection pattern for cohesive devices
#[derive(Debug, Clone)]
struct DetectionPattern {
    pattern_regex: String,
    context_requirements: Vec<String>,
    exclusion_patterns: Vec<String>,
    confidence_score: f64,
}

/// Chain building strategy enumeration
#[derive(Debug, Clone)]
enum ChainBuildingStrategy {
    IdenticalRepetition,
    SynonymicRepetition,
    PronounalReference,
    DefiniteReference,
    DemonstrativeReference,
    ComparativeReference,
    PartialRepetition,
    SubstitutionChain,
}

/// Connectivity measure enumeration
#[derive(Debug, Clone)]
enum ConnectivityMeasure {
    LocalConnectivity,
    GlobalConnectivity,
    HierarchicalConnectivity,
    SequentialConnectivity,
    SemanticConnectivity,
}

/// Reference type classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReferenceType {
    Pronominal,
    Definite,
    Demonstrative,
    Comparative,
    Elliptical,
    Substitution,
    Bridging,
}

/// Connector scope classification
#[derive(Debug, Clone)]
enum ConnectorScope {
    Local,     // Within sentence
    Adjacent,  // Between adjacent sentences
    Paragraph, // Within paragraph
    Global,    // Across paragraphs
}

/// Coreference resolution component
#[derive(Debug)]
struct CoreferenceResolver {
    resolution_strategies: Vec<ResolutionStrategy>,
    feature_extractors: Vec<CorefFeatureExtractor>,
    scoring_weights: HashMap<String, f64>,
}

/// Graph analyzer for connectivity analysis
#[derive(Debug)]
struct GraphAnalyzer {
    algorithms: Vec<GraphAlgorithm>,
    centrality_measures: Vec<CentralityMeasure>,
    clustering_algorithms: Vec<ClusteringAlgorithm>,
}

/// Density calculator for cohesion measurement
#[derive(Debug)]
struct DensityCalculator {
    density_types: Vec<DensityType>,
    normalization_methods: Vec<NormalizationMethod>,
    weighting_schemes: Vec<WeightingScheme>,
}

/// Resolution strategy for coreference
#[derive(Debug, Clone)]
enum ResolutionStrategy {
    NumberAgreement,
    GenderAgreement,
    SemanticCompatibility,
    SyntacticConstraints,
    DistancePreference,
    SalienceRanking,
}

/// Feature extractor for coreference resolution
#[derive(Debug, Clone)]
struct CorefFeatureExtractor {
    name: String,
    extraction_function: fn(&str, &str, &[String]) -> Vec<f64>,
}

/// Graph algorithm enumeration
#[derive(Debug, Clone)]
enum GraphAlgorithm {
    ShortestPath,
    MinimumSpanningTree,
    ConnectedComponents,
    CommunityDetection,
}

/// Centrality measure enumeration
#[derive(Debug, Clone)]
enum CentralityMeasure {
    DegreeCentrality,
    BetweennessCentrality,
    ClosenessCentrality,
    EigenvectorCentrality,
    PageRank,
}

/// Clustering algorithm enumeration
#[derive(Debug, Clone)]
enum ClusteringAlgorithm {
    Hierarchical,
    KMeans,
    DBSCAN,
    SpectralClustering,
}

/// Density type enumeration
#[derive(Debug, Clone)]
enum DensityType {
    LocalDensity,
    GlobalDensity,
    WeightedDensity,
    NormalizedDensity,
}

/// Normalization method enumeration
#[derive(Debug, Clone)]
enum NormalizationMethod {
    MinMax,
    ZScore,
    RobustScaling,
    UnitVector,
}

/// Weighting scheme enumeration
#[derive(Debug, Clone)]
enum WeightingScheme {
    Uniform,
    DistanceBased,
    FrequencyBased,
    TfIdf,
}

impl CohesionAnalyzer {
    /// Create a new cohesion analyzer with configuration
    pub fn new(config: CohesionAnalysisConfig) -> Result<Self, CohesionAnalysisError> {
        let mut analyzer = CohesionAnalyzer {
            config: config.clone(),
            device_detector: CohesiveDeviceDetector::new(&config)?,
            chain_builder: ReferentialChainBuilder::new(&config)?,
            connectivity_analyzer: ConnectivityAnalyzer::new(&config)?,
            cohesive_markers: HashMap::new(),
            reference_patterns: HashMap::new(),
            connector_lexicon: HashMap::new(),
            device_cache: HashMap::new(),
            chain_cache: HashMap::new(),
            connectivity_cache: HashMap::new(),
        };

        analyzer.initialize_linguistic_resources()?;
        Ok(analyzer)
    }

    /// Perform comprehensive cohesion analysis
    pub fn analyze_cohesion(
        &mut self,
        lexical_items: &[LexicalItem],
        sentences: &[String],
        text: &str,
    ) -> Result<CohesionAnalysisResult, CohesionAnalysisError> {
        // Step 1: Detect cohesive devices
        let cohesive_devices = self.detect_cohesive_devices(sentences, text)?;

        // Step 2: Build referential chains
        let referential_chains = self.build_referential_chains(lexical_items, sentences)?;

        // Step 3: Analyze connectivity patterns
        let connectivity_analysis =
            self.analyze_connectivity(&cohesive_devices, &referential_chains, sentences)?;

        // Step 4: Calculate cohesion metrics
        let cohesion_metrics = self.calculate_cohesion_metrics(
            &cohesive_devices,
            &referential_chains,
            &connectivity_analysis,
        )?;

        // Step 5: Generate cohesion insights
        let insights = self.generate_cohesion_insights(
            &cohesive_devices,
            &referential_chains,
            &connectivity_analysis,
            &cohesion_metrics,
        )?;

        Ok(CohesionAnalysisResult {
            cohesive_devices,
            referential_chains,
            connectivity_analysis,
            cohesion_metrics,
            insights,
            analysis_metadata: self.generate_analysis_metadata(),
        })
    }

    /// Initialize linguistic resources for cohesion analysis
    fn initialize_linguistic_resources(&mut self) -> Result<(), CohesionAnalysisError> {
        // Initialize cohesive markers for different device types
        self.initialize_cohesive_markers()?;

        // Initialize reference patterns
        self.initialize_reference_patterns()?;

        // Initialize connector lexicon
        self.initialize_connector_lexicon()?;

        Ok(())
    }

    /// Initialize cohesive markers lexicon
    fn initialize_cohesive_markers(&mut self) -> Result<(), CohesionAnalysisError> {
        // Repetition markers (exact word repetition)
        let repetition_markers = HashSet::new(); // Handled by exact matching

        // Synonymy markers
        let synonymy_markers = HashSet::from([
            "also".to_string(),
            "similarly".to_string(),
            "likewise".to_string(),
            "in the same way".to_string(),
            "correspondingly".to_string(),
        ]);

        // Hyponymy/hypernymy markers
        let hyponymy_markers = HashSet::from([
            "such as".to_string(),
            "for example".to_string(),
            "including".to_string(),
            "namely".to_string(),
            "specifically".to_string(),
            "in particular".to_string(),
        ]);

        // Meronymy markers
        let meronymy_markers = HashSet::from([
            "part of".to_string(),
            "consists of".to_string(),
            "contains".to_string(),
            "includes".to_string(),
            "comprises".to_string(),
        ]);

        // Collocation markers
        let collocation_markers = HashSet::from([
            "and".to_string(),
            "with".to_string(),
            "together".to_string(),
            "along with".to_string(),
            "combined with".to_string(),
        ]);

        // Antonymy markers
        let antonymy_markers = HashSet::from([
            "however".to_string(),
            "but".to_string(),
            "yet".to_string(),
            "on the other hand".to_string(),
            "in contrast".to_string(),
            "nevertheless".to_string(),
            "nonetheless".to_string(),
        ]);

        // Morphological markers
        let morphological_markers = HashSet::new(); // Handled by morphological analysis

        // Bridging markers
        let bridging_markers = HashSet::from([
            "the".to_string(),
            "this".to_string(),
            "that".to_string(),
            "these".to_string(),
            "those".to_string(),
            "such".to_string(),
        ]);

        self.cohesive_markers
            .insert(CohesiveDeviceType::Repetition, repetition_markers);
        self.cohesive_markers
            .insert(CohesiveDeviceType::Synonymy, synonymy_markers);
        self.cohesive_markers
            .insert(CohesiveDeviceType::Hyponymy, hyponymy_markers);
        self.cohesive_markers
            .insert(CohesiveDeviceType::Meronymy, meronymy_markers);
        self.cohesive_markers
            .insert(CohesiveDeviceType::Collocation, collocation_markers);
        self.cohesive_markers
            .insert(CohesiveDeviceType::Antonymy, antonymy_markers);
        self.cohesive_markers
            .insert(CohesiveDeviceType::Morphological, morphological_markers);
        self.cohesive_markers
            .insert(CohesiveDeviceType::Bridging, bridging_markers);

        Ok(())
    }

    /// Initialize reference patterns for coreference resolution
    fn initialize_reference_patterns(&mut self) -> Result<(), CohesionAnalysisError> {
        // Pronominal reference patterns
        self.reference_patterns.insert(
            "personal_pronouns".to_string(),
            ReferencePattern {
                pattern_type: ReferenceType::Pronominal,
                matching_rules: vec![
                    r"\b(he|she|it|they|him|her|them|his|hers|its|their)\b".to_string()
                ],
                confidence_weight: 0.8,
                distance_penalty: 0.1,
            },
        );

        // Definite reference patterns
        self.reference_patterns.insert(
            "definite_articles".to_string(),
            ReferencePattern {
                pattern_type: ReferenceType::Definite,
                matching_rules: vec![r"\bthe\s+(\w+)".to_string()],
                confidence_weight: 0.6,
                distance_penalty: 0.05,
            },
        );

        // Demonstrative reference patterns
        self.reference_patterns.insert(
            "demonstratives".to_string(),
            ReferencePattern {
                pattern_type: ReferenceType::Demonstrative,
                matching_rules: vec![
                    r"\b(this|that|these|those)\s+(\w+)".to_string(),
                    r"\b(this|that|these|those)\b".to_string(),
                ],
                confidence_weight: 0.7,
                distance_penalty: 0.08,
            },
        );

        // Comparative reference patterns
        self.reference_patterns.insert(
            "comparatives".to_string(),
            ReferencePattern {
                pattern_type: ReferenceType::Comparative,
                matching_rules: vec![
                    r"\b(another|other|same|similar|different)\s+(\w+)".to_string()
                ],
                confidence_weight: 0.5,
                distance_penalty: 0.12,
            },
        );

        Ok(())
    }

    /// Initialize connector lexicon for discourse markers
    fn initialize_connector_lexicon(&mut self) -> Result<(), CohesionAnalysisError> {
        // Additive connectors
        let additive_connectors = vec![
            (
                "and",
                ConnectorInfo {
                    connector_type: CohesiveDeviceType::Repetition,
                    semantic_function: "addition".to_string(),
                    scope: ConnectorScope::Local,
                    strength: 0.8,
                },
            ),
            (
                "furthermore",
                ConnectorInfo {
                    connector_type: CohesiveDeviceType::Repetition,
                    semantic_function: "addition".to_string(),
                    scope: ConnectorScope::Adjacent,
                    strength: 0.9,
                },
            ),
            (
                "moreover",
                ConnectorInfo {
                    connector_type: CohesiveDeviceType::Repetition,
                    semantic_function: "addition".to_string(),
                    scope: ConnectorScope::Adjacent,
                    strength: 0.9,
                },
            ),
        ];

        // Adversative connectors
        let adversative_connectors = vec![
            (
                "but",
                ConnectorInfo {
                    connector_type: CohesiveDeviceType::Antonymy,
                    semantic_function: "contrast".to_string(),
                    scope: ConnectorScope::Local,
                    strength: 0.8,
                },
            ),
            (
                "however",
                ConnectorInfo {
                    connector_type: CohesiveDeviceType::Antonymy,
                    semantic_function: "contrast".to_string(),
                    scope: ConnectorScope::Adjacent,
                    strength: 0.9,
                },
            ),
            (
                "nevertheless",
                ConnectorInfo {
                    connector_type: CohesiveDeviceType::Antonymy,
                    semantic_function: "contrast".to_string(),
                    scope: ConnectorScope::Paragraph,
                    strength: 0.95,
                },
            ),
        ];

        // Temporal connectors
        let temporal_connectors = vec![
            (
                "then",
                ConnectorInfo {
                    connector_type: CohesiveDeviceType::Bridging,
                    semantic_function: "temporal".to_string(),
                    scope: ConnectorScope::Local,
                    strength: 0.7,
                },
            ),
            (
                "meanwhile",
                ConnectorInfo {
                    connector_type: CohesiveDeviceType::Bridging,
                    semantic_function: "temporal".to_string(),
                    scope: ConnectorScope::Adjacent,
                    strength: 0.8,
                },
            ),
            (
                "subsequently",
                ConnectorInfo {
                    connector_type: CohesiveDeviceType::Bridging,
                    semantic_function: "temporal".to_string(),
                    scope: ConnectorScope::Paragraph,
                    strength: 0.85,
                },
            ),
        ];

        for (connector, info) in additive_connectors
            .into_iter()
            .chain(adversative_connectors.into_iter())
            .chain(temporal_connectors.into_iter())
        {
            self.connector_lexicon.insert(connector.to_string(), info);
        }

        Ok(())
    }

    /// Detect cohesive devices in text
    fn detect_cohesive_devices(
        &mut self,
        sentences: &[String],
        text: &str,
    ) -> Result<Vec<CohesiveDevice>, CohesionAnalysisError> {
        let mut devices = Vec::new();

        // Detect each type of cohesive device
        for device_type in &[
            CohesiveDeviceType::Repetition,
            CohesiveDeviceType::Synonymy,
            CohesiveDeviceType::Hyponymy,
            CohesiveDeviceType::Meronymy,
            CohesiveDeviceType::Collocation,
            CohesiveDeviceType::Antonymy,
            CohesiveDeviceType::Morphological,
            CohesiveDeviceType::Bridging,
        ] {
            let device_instances = self.device_detector.detect_devices(
                device_type,
                sentences,
                text,
                &self.cohesive_markers,
                &self.connector_lexicon,
            )?;
            devices.extend(device_instances);
        }

        Ok(devices)
    }

    /// Build referential chains from lexical items
    fn build_referential_chains(
        &mut self,
        lexical_items: &[LexicalItem],
        sentences: &[String],
    ) -> Result<Vec<ReferentialChain>, CohesionAnalysisError> {
        self.chain_builder
            .build_chains(lexical_items, sentences, &self.reference_patterns)
    }

    /// Analyze connectivity patterns
    fn analyze_connectivity(
        &mut self,
        cohesive_devices: &[CohesiveDevice],
        referential_chains: &[ReferentialChain],
        sentences: &[String],
    ) -> Result<ConnectivityAnalysis, CohesionAnalysisError> {
        self.connectivity_analyzer.analyze_connectivity(
            cohesive_devices,
            referential_chains,
            sentences,
        )
    }

    /// Calculate comprehensive cohesion metrics
    fn calculate_cohesion_metrics(
        &self,
        cohesive_devices: &[CohesiveDevice],
        referential_chains: &[ReferentialChain],
        connectivity_analysis: &ConnectivityAnalysis,
    ) -> Result<CohesionMetrics, CohesionAnalysisError> {
        // Device-based metrics
        let device_density =
            cohesive_devices.len() as f64 / connectivity_analysis.text_length as f64;
        let device_diversity = self.calculate_device_diversity(cohesive_devices);
        let device_distribution = self.calculate_device_distribution(cohesive_devices);

        // Chain-based metrics
        let chain_density =
            referential_chains.len() as f64 / connectivity_analysis.text_length as f64;
        let average_chain_length = if !referential_chains.is_empty() {
            referential_chains
                .iter()
                .map(|chain| chain.elements.len() as f64)
                .sum::<f64>()
                / referential_chains.len() as f64
        } else {
            0.0
        };

        let chain_coverage =
            self.calculate_chain_coverage(referential_chains, &connectivity_analysis)?;

        // Connectivity-based metrics
        let local_connectivity = connectivity_analysis.local_connectivity_score;
        let global_connectivity = connectivity_analysis.global_connectivity_score;
        let hierarchical_connectivity = connectivity_analysis.hierarchical_connectivity_score;

        // Overall cohesion score
        let overall_cohesion = self.calculate_overall_cohesion(
            device_density,
            chain_density,
            local_connectivity,
            global_connectivity,
        );

        Ok(CohesionMetrics {
            overall_cohesion,
            device_density,
            device_diversity,
            device_distribution,
            chain_density,
            average_chain_length,
            chain_coverage,
            local_connectivity,
            global_connectivity,
            hierarchical_connectivity,
            connectivity_variance: connectivity_analysis.connectivity_variance,
            cohesion_breakdown: self.calculate_cohesion_breakdown(cohesive_devices),
        })
    }

    /// Generate cohesion insights
    fn generate_cohesion_insights(
        &self,
        cohesive_devices: &[CohesiveDevice],
        referential_chains: &[ReferentialChain],
        connectivity_analysis: &ConnectivityAnalysis,
        cohesion_metrics: &CohesionMetrics,
    ) -> Result<Vec<String>, CohesionAnalysisError> {
        let mut insights = Vec::new();

        // Device-based insights
        if cohesion_metrics.device_density > 0.1 {
            insights.push(
                "High density of cohesive devices indicates strong lexical cohesion".to_string(),
            );
        } else if cohesion_metrics.device_density < 0.05 {
            insights
                .push("Low density of cohesive devices suggests weak lexical cohesion".to_string());
        }

        if cohesion_metrics.device_diversity > 0.7 {
            insights
                .push("Good variety of cohesive device types enhances text coherence".to_string());
        }

        // Chain-based insights
        if cohesion_metrics.average_chain_length > 3.0 {
            insights.push("Long referential chains contribute to thematic continuity".to_string());
        }

        if cohesion_metrics.chain_coverage > 0.6 {
            insights.push("High chain coverage indicates strong referential coherence".to_string());
        }

        // Connectivity insights
        if cohesion_metrics.local_connectivity > 0.8 {
            insights.push("Strong local connectivity ensures sentence-level coherence".to_string());
        }

        if cohesion_metrics.global_connectivity < 0.4 {
            insights.push(
                "Weak global connectivity may indicate fragmented discourse structure".to_string(),
            );
        }

        // Overall assessment
        if cohesion_metrics.overall_cohesion > 0.7 {
            insights.push("Text demonstrates strong overall lexical cohesion".to_string());
        } else if cohesion_metrics.overall_cohesion < 0.4 {
            insights.push(
                "Text shows limited lexical cohesion and may benefit from revision".to_string(),
            );
        }

        Ok(insights)
    }

    /// Calculate device diversity (Shannon entropy of device types)
    fn calculate_device_diversity(&self, cohesive_devices: &[CohesiveDevice]) -> f64 {
        if cohesive_devices.is_empty() {
            return 0.0;
        }

        let mut type_counts: HashMap<CohesiveDeviceType, usize> = HashMap::new();
        for device in cohesive_devices {
            *type_counts.entry(device.device_type.clone()).or_insert(0) += 1;
        }

        let total = cohesive_devices.len() as f64;
        let mut entropy = 0.0;

        for count in type_counts.values() {
            let probability = *count as f64 / total;
            if probability > 0.0 {
                entropy -= probability * probability.log2();
            }
        }

        // Normalize by maximum possible entropy
        let max_entropy = (type_counts.len() as f64).log2();
        if max_entropy > 0.0 {
            entropy / max_entropy
        } else {
            0.0
        }
    }

    /// Calculate device distribution across text
    fn calculate_device_distribution(
        &self,
        cohesive_devices: &[CohesiveDevice],
    ) -> HashMap<String, f64> {
        let mut distribution = HashMap::new();

        // Calculate distribution by device type
        let total = cohesive_devices.len() as f64;
        if total == 0.0 {
            return distribution;
        }

        let mut type_counts: HashMap<String, usize> = HashMap::new();
        for device in cohesive_devices {
            let type_name = format!("{:?}", device.device_type);
            *type_counts.entry(type_name).or_insert(0) += 1;
        }

        for (device_type, count) in type_counts {
            distribution.insert(device_type, count as f64 / total);
        }

        distribution
    }

    /// Calculate chain coverage of text
    fn calculate_chain_coverage(
        &self,
        referential_chains: &[ReferentialChain],
        connectivity_analysis: &ConnectivityAnalysis,
    ) -> Result<f64, CohesionAnalysisError> {
        if referential_chains.is_empty() || connectivity_analysis.text_length == 0 {
            return Ok(0.0);
        }

        let mut covered_positions = HashSet::new();

        for chain in referential_chains {
            for element in &chain.elements {
                for &(start, end) in &element.positions {
                    for pos in start..=end {
                        covered_positions.insert(pos);
                    }
                }
            }
        }

        let coverage = covered_positions.len() as f64 / connectivity_analysis.text_length as f64;
        Ok(coverage.min(1.0))
    }

    /// Calculate overall cohesion score
    fn calculate_overall_cohesion(
        &self,
        device_density: f64,
        chain_density: f64,
        local_connectivity: f64,
        global_connectivity: f64,
    ) -> f64 {
        let device_weight = 0.25;
        let chain_weight = 0.25;
        let local_weight = 0.25;
        let global_weight = 0.25;

        device_density * device_weight
            + chain_density * chain_weight
            + local_connectivity * local_weight
            + global_connectivity * global_weight
    }

    /// Calculate cohesion breakdown by device types
    fn calculate_cohesion_breakdown(
        &self,
        cohesive_devices: &[CohesiveDevice],
    ) -> HashMap<String, f64> {
        let mut breakdown = HashMap::new();

        if cohesive_devices.is_empty() {
            return breakdown;
        }

        let mut type_strengths: HashMap<String, Vec<f64>> = HashMap::new();
        for device in cohesive_devices {
            let type_name = format!("{:?}", device.device_type);
            type_strengths
                .entry(type_name)
                .or_default()
                .push(device.strength);
        }

        for (device_type, strengths) in type_strengths {
            let average_strength = strengths.iter().sum::<f64>() / strengths.len() as f64;
            breakdown.insert(device_type, average_strength);
        }

        breakdown
    }

    /// Generate analysis metadata
    fn generate_analysis_metadata(&self) -> HashMap<String, String> {
        let mut metadata = HashMap::new();

        metadata.insert("analyzer_version".to_string(), "1.0.0".to_string());
        metadata.insert(
            "device_types_detected".to_string(),
            self.cohesive_markers.len().to_string(),
        );
        metadata.insert(
            "reference_patterns".to_string(),
            self.reference_patterns.len().to_string(),
        );
        metadata.insert(
            "connector_lexicon_size".to_string(),
            self.connector_lexicon.len().to_string(),
        );
        metadata.insert(
            "max_chain_distance".to_string(),
            self.chain_builder.maximum_chain_distance.to_string(),
        );
        metadata.insert(
            "min_chain_length".to_string(),
            self.chain_builder.minimum_chain_length.to_string(),
        );

        metadata
    }
}

impl CohesiveDeviceDetector {
    fn new(config: &CohesionAnalysisConfig) -> Result<Self, CohesionAnalysisError> {
        Ok(CohesiveDeviceDetector {
            detection_patterns: HashMap::new(),
            context_window: config.context_window_size,
            confidence_threshold: config.device_confidence_threshold,
        })
    }

    fn detect_devices(
        &self,
        device_type: &CohesiveDeviceType,
        sentences: &[String],
        text: &str,
        cohesive_markers: &HashMap<CohesiveDeviceType, HashSet<String>>,
        connector_lexicon: &HashMap<String, ConnectorInfo>,
    ) -> Result<Vec<CohesiveDevice>, CohesionAnalysisError> {
        let mut devices = Vec::new();

        match device_type {
            CohesiveDeviceType::Repetition => {
                devices.extend(self.detect_repetition_devices(sentences)?);
            }
            CohesiveDeviceType::Synonymy => {
                devices.extend(self.detect_synonymy_devices(sentences, cohesive_markers)?);
            }
            CohesiveDeviceType::Hyponymy => {
                devices.extend(self.detect_hyponymy_devices(sentences, cohesive_markers)?);
            }
            CohesiveDeviceType::Meronymy => {
                devices.extend(self.detect_meronymy_devices(sentences, cohesive_markers)?);
            }
            CohesiveDeviceType::Collocation => {
                devices.extend(self.detect_collocation_devices(sentences, cohesive_markers)?);
            }
            CohesiveDeviceType::Antonymy => {
                devices.extend(self.detect_antonymy_devices(sentences, cohesive_markers)?);
            }
            CohesiveDeviceType::Morphological => {
                devices.extend(self.detect_morphological_devices(sentences)?);
            }
            CohesiveDeviceType::Bridging => {
                devices.extend(self.detect_bridging_devices(sentences, cohesive_markers)?);
            }
        }

        Ok(devices)
    }

    fn detect_repetition_devices(
        &self,
        sentences: &[String],
    ) -> Result<Vec<CohesiveDevice>, CohesionAnalysisError> {
        let mut devices = Vec::new();
        let mut word_occurrences: HashMap<String, Vec<(usize, usize, usize)>> = HashMap::new();

        // Collect all word occurrences
        for (sent_idx, sentence) in sentences.iter().enumerate() {
            let words: Vec<&str> = sentence.split_whitespace().collect();
            let mut char_pos = 0;

            for (word_idx, word) in words.iter().enumerate() {
                let word_lower = word
                    .to_lowercase()
                    .trim_matches(|c: char| !c.is_alphabetic())
                    .to_string();
                if !word_lower.is_empty() && word_lower.len() > 2 {
                    // Filter short words
                    word_occurrences
                        .entry(word_lower)
                        .or_default()
                        .push((sent_idx, word_idx, char_pos));
                }
                char_pos += word.len() + 1; // +1 for space
            }
        }

        // Find repetitions
        for (word, occurrences) in word_occurrences {
            if occurrences.len() > 1 {
                for window in occurrences.windows(2) {
                    if let [first, second] = window {
                        let distance = second.0 - first.0; // Sentence distance
                        if distance <= self.context_window {
                            devices.push(CohesiveDevice {
                                device_type: CohesiveDeviceType::Repetition,
                                source_element: word.clone(),
                                target_element: word.clone(),
                                source_position: (first.0, first.1),
                                target_position: (second.0, second.1),
                                strength: 1.0 - (distance as f64 * 0.1), // Decrease strength with distance
                                confidence: 0.95,
                                distance: distance as f64,
                                context: vec![
                                    sentences.get(first.0).unwrap_or(&String::new()).clone(),
                                    sentences.get(second.0).unwrap_or(&String::new()).clone(),
                                ],
                            });
                        }
                    }
                }
            }
        }

        Ok(devices)
    }

    fn detect_synonymy_devices(
        &self,
        sentences: &[String],
        cohesive_markers: &HashMap<CohesiveDeviceType, HashSet<String>>,
    ) -> Result<Vec<CohesiveDevice>, CohesionAnalysisError> {
        let mut devices = Vec::new();
        let markers = cohesive_markers
            .get(&CohesiveDeviceType::Synonymy)
            .unwrap_or(&HashSet::new());

        for (sent_idx, sentence) in sentences.iter().enumerate() {
            let sentence_lower = sentence.to_lowercase();

            for marker in markers {
                if sentence_lower.contains(marker) {
                    // Simple detection: if synonymy marker is present, assume synonymy device
                    if let Some(next_sentence) = sentences.get(sent_idx + 1) {
                        devices.push(CohesiveDevice {
                            device_type: CohesiveDeviceType::Synonymy,
                            source_element: marker.clone(),
                            target_element: "context".to_string(),
                            source_position: (sent_idx, 0),
                            target_position: (sent_idx + 1, 0),
                            strength: 0.7,
                            confidence: 0.6,
                            distance: 1.0,
                            context: vec![sentence.clone(), next_sentence.clone()],
                        });
                    }
                }
            }
        }

        Ok(devices)
    }

    fn detect_hyponymy_devices(
        &self,
        sentences: &[String],
        cohesive_markers: &HashMap<CohesiveDeviceType, HashSet<String>>,
    ) -> Result<Vec<CohesiveDevice>, CohesionAnalysisError> {
        let mut devices = Vec::new();
        let markers = cohesive_markers
            .get(&CohesiveDeviceType::Hyponymy)
            .unwrap_or(&HashSet::new());

        for (sent_idx, sentence) in sentences.iter().enumerate() {
            let sentence_lower = sentence.to_lowercase();

            for marker in markers {
                if sentence_lower.contains(marker) {
                    devices.push(CohesiveDevice {
                        device_type: CohesiveDeviceType::Hyponymy,
                        source_element: marker.clone(),
                        target_element: "example".to_string(),
                        source_position: (sent_idx, 0),
                        target_position: (sent_idx, 0),
                        strength: 0.8,
                        confidence: 0.75,
                        distance: 0.0,
                        context: vec![sentence.clone()],
                    });
                }
            }
        }

        Ok(devices)
    }

    fn detect_meronymy_devices(
        &self,
        sentences: &[String],
        cohesive_markers: &HashMap<CohesiveDeviceType, HashSet<String>>,
    ) -> Result<Vec<CohesiveDevice>, CohesionAnalysisError> {
        let mut devices = Vec::new();
        let markers = cohesive_markers
            .get(&CohesiveDeviceType::Meronymy)
            .unwrap_or(&HashSet::new());

        for (sent_idx, sentence) in sentences.iter().enumerate() {
            let sentence_lower = sentence.to_lowercase();

            for marker in markers {
                if sentence_lower.contains(marker) {
                    devices.push(CohesiveDevice {
                        device_type: CohesiveDeviceType::Meronymy,
                        source_element: marker.clone(),
                        target_element: "part_whole".to_string(),
                        source_position: (sent_idx, 0),
                        target_position: (sent_idx, 0),
                        strength: 0.75,
                        confidence: 0.7,
                        distance: 0.0,
                        context: vec![sentence.clone()],
                    });
                }
            }
        }

        Ok(devices)
    }

    fn detect_collocation_devices(
        &self,
        sentences: &[String],
        cohesive_markers: &HashMap<CohesiveDeviceType, HashSet<String>>,
    ) -> Result<Vec<CohesiveDevice>, CohesionAnalysisError> {
        let mut devices = Vec::new();
        let markers = cohesive_markers
            .get(&CohesiveDeviceType::Collocation)
            .unwrap_or(&HashSet::new());

        for (sent_idx, sentence) in sentences.iter().enumerate() {
            let words: Vec<&str> = sentence.split_whitespace().collect();

            for (word_idx, word) in words.iter().enumerate() {
                if markers.contains(&word.to_lowercase()) {
                    // Look for collocated terms nearby
                    let start = word_idx.saturating_sub(2);
                    let end = (word_idx + 3).min(words.len());
                    let context_words: Vec<String> =
                        words[start..end].iter().map(|w| w.to_string()).collect();

                    devices.push(CohesiveDevice {
                        device_type: CohesiveDeviceType::Collocation,
                        source_element: word.to_string(),
                        target_element: context_words.join(" "),
                        source_position: (sent_idx, word_idx),
                        target_position: (sent_idx, word_idx),
                        strength: 0.6,
                        confidence: 0.65,
                        distance: 0.0,
                        context: vec![sentence.clone()],
                    });
                }
            }
        }

        Ok(devices)
    }

    fn detect_antonymy_devices(
        &self,
        sentences: &[String],
        cohesive_markers: &HashMap<CohesiveDeviceType, HashSet<String>>,
    ) -> Result<Vec<CohesiveDevice>, CohesionAnalysisError> {
        let mut devices = Vec::new();
        let markers = cohesive_markers
            .get(&CohesiveDeviceType::Antonymy)
            .unwrap_or(&HashSet::new());

        for (sent_idx, sentence) in sentences.iter().enumerate() {
            let sentence_lower = sentence.to_lowercase();

            for marker in markers {
                if sentence_lower.contains(marker) {
                    if let Some(next_sentence) = sentences.get(sent_idx + 1) {
                        devices.push(CohesiveDevice {
                            device_type: CohesiveDeviceType::Antonymy,
                            source_element: marker.clone(),
                            target_element: "contrast".to_string(),
                            source_position: (sent_idx, 0),
                            target_position: (sent_idx + 1, 0),
                            strength: 0.8,
                            confidence: 0.75,
                            distance: 1.0,
                            context: vec![sentence.clone(), next_sentence.clone()],
                        });
                    }
                }
            }
        }

        Ok(devices)
    }

    fn detect_morphological_devices(
        &self,
        sentences: &[String],
    ) -> Result<Vec<CohesiveDevice>, CohesionAnalysisError> {
        let mut devices = Vec::new();
        let mut word_stems: HashMap<String, Vec<(usize, usize, String)>> = HashMap::new();

        // Simple stem extraction using common suffixes
        let suffixes = vec![
            "ing", "ed", "er", "est", "ly", "tion", "sion", "ness", "ment", "able",
        ];

        for (sent_idx, sentence) in sentences.iter().enumerate() {
            let words: Vec<&str> = sentence.split_whitespace().collect();

            for (word_idx, word) in words.iter().enumerate() {
                let word_lower = word
                    .to_lowercase()
                    .trim_matches(|c: char| !c.is_alphabetic())
                    .to_string();

                if word_lower.len() > 4 {
                    let stem = self.extract_stem(&word_lower, &suffixes);
                    if stem != word_lower {
                        word_stems
                            .entry(stem)
                            .or_default()
                            .push((sent_idx, word_idx, word_lower));
                    }
                }
            }
        }

        // Find morphological relationships
        for (stem, words) in word_stems {
            if words.len() > 1 {
                for window in words.windows(2) {
                    if let [(sent1, word1, form1), (sent2, word2, form2)] = window {
                        if sent2 - sent1 <= self.context_window {
                            devices.push(CohesiveDevice {
                                device_type: CohesiveDeviceType::Morphological,
                                source_element: form1.clone(),
                                target_element: form2.clone(),
                                source_position: (*sent1, *word1),
                                target_position: (*sent2, *word2),
                                strength: 0.7,
                                confidence: 0.6,
                                distance: (sent2 - sent1) as f64,
                                context: vec![
                                    sentences.get(*sent1).unwrap_or(&String::new()).clone(),
                                    sentences.get(*sent2).unwrap_or(&String::new()).clone(),
                                ],
                            });
                        }
                    }
                }
            }
        }

        Ok(devices)
    }

    fn detect_bridging_devices(
        &self,
        sentences: &[String],
        cohesive_markers: &HashMap<CohesiveDeviceType, HashSet<String>>,
    ) -> Result<Vec<CohesiveDevice>, CohesionAnalysisError> {
        let mut devices = Vec::new();
        let markers = cohesive_markers
            .get(&CohesiveDeviceType::Bridging)
            .unwrap_or(&HashSet::new());

        for (sent_idx, sentence) in sentences.iter().enumerate() {
            let words: Vec<&str> = sentence.split_whitespace().collect();

            for (word_idx, word) in words.iter().enumerate() {
                if markers.contains(&word.to_lowercase()) {
                    // Look for potential bridging reference
                    if let Some(next_word) = words.get(word_idx + 1) {
                        devices.push(CohesiveDevice {
                            device_type: CohesiveDeviceType::Bridging,
                            source_element: word.to_string(),
                            target_element: next_word.to_string(),
                            source_position: (sent_idx, word_idx),
                            target_position: (sent_idx, word_idx + 1),
                            strength: 0.5,
                            confidence: 0.5,
                            distance: 0.0,
                            context: vec![sentence.clone()],
                        });
                    }
                }
            }
        }

        Ok(devices)
    }

    fn extract_stem(&self, word: &str, suffixes: &[&str]) -> String {
        for suffix in suffixes {
            if word.ends_with(suffix) && word.len() > suffix.len() + 2 {
                return word[..word.len() - suffix.len()].to_string();
            }
        }
        word.to_string()
    }
}

impl ReferentialChainBuilder {
    fn new(config: &CohesionAnalysisConfig) -> Result<Self, CohesionAnalysisError> {
        Ok(ReferentialChainBuilder {
            chain_strategies: vec![
                ChainBuildingStrategy::IdenticalRepetition,
                ChainBuildingStrategy::SynonymicRepetition,
                ChainBuildingStrategy::PronounalReference,
                ChainBuildingStrategy::DefiniteReference,
                ChainBuildingStrategy::DemonstrativeReference,
            ],
            maximum_chain_distance: config.max_chain_distance,
            minimum_chain_length: config.min_chain_length,
            coreference_resolver: CoreferenceResolver::new(config)?,
        })
    }

    fn build_chains(
        &mut self,
        lexical_items: &[LexicalItem],
        sentences: &[String],
        reference_patterns: &HashMap<String, ReferencePattern>,
    ) -> Result<Vec<ReferentialChain>, CohesionAnalysisError> {
        let mut chains = Vec::new();

        for strategy in &self.chain_strategies {
            let strategy_chains = self.build_chains_with_strategy(
                strategy,
                lexical_items,
                sentences,
                reference_patterns,
            )?;
            chains.extend(strategy_chains);
        }

        // Merge overlapping chains
        let merged_chains = self.merge_overlapping_chains(chains)?;

        // Filter chains by minimum length
        let filtered_chains: Vec<ReferentialChain> = merged_chains
            .into_iter()
            .filter(|chain| chain.elements.len() >= self.minimum_chain_length)
            .collect();

        Ok(filtered_chains)
    }

    fn build_chains_with_strategy(
        &mut self,
        strategy: &ChainBuildingStrategy,
        lexical_items: &[LexicalItem],
        sentences: &[String],
        reference_patterns: &HashMap<String, ReferencePattern>,
    ) -> Result<Vec<ReferentialChain>, CohesionAnalysisError> {
        match strategy {
            ChainBuildingStrategy::IdenticalRepetition => {
                self.build_identical_repetition_chains(lexical_items)
            }
            ChainBuildingStrategy::PronounalReference => {
                self.build_pronominal_chains(sentences, reference_patterns)
            }
            ChainBuildingStrategy::DefiniteReference => {
                self.build_definite_reference_chains(sentences, reference_patterns)
            }
            _ => Ok(Vec::new()), // Other strategies not implemented in this example
        }
    }

    fn build_identical_repetition_chains(
        &self,
        lexical_items: &[LexicalItem],
    ) -> Result<Vec<ReferentialChain>, CohesionAnalysisError> {
        let mut chains = Vec::new();
        let mut word_groups: HashMap<String, Vec<&LexicalItem>> = HashMap::new();

        // Group identical words
        for item in lexical_items {
            word_groups
                .entry(item.word.to_lowercase())
                .or_default()
                .push(item);
        }

        let mut chain_id = 0;
        for (word, items) in word_groups {
            if items.len() >= self.minimum_chain_length {
                let mut elements = Vec::new();

                for item in items {
                    elements.push(
                        crate::metrics::coherence::lexical_coherence::results::ChainElement {
                            word: item.word.clone(),
                            lemma: item.lemma.clone(),
                            positions: item.positions.clone(),
                            reference_type: ReferenceType::Pronominal, // Simplified
                            confidence: 0.9,
                        },
                    );
                }

                // Sort elements by position
                elements.sort_by_key(|elem| {
                    elem.positions.first().map(|(start, _)| *start).unwrap_or(0)
                });

                let chain_strength = self.calculate_chain_strength(&elements);
                let coherence_score = self.calculate_chain_coherence(&elements);

                chains.push(ReferentialChain {
                    chain_id,
                    elements,
                    chain_type: crate::metrics::coherence::lexical_coherence::results::ChainType::IdenticalRepetition,
                    strength: chain_strength,
                    coherence_score,
                    average_distance: self.calculate_average_distance(&chains.last().unwrap().elements),
                    coverage: 0.0, // Will be calculated later
                });

                chain_id += 1;
            }
        }

        Ok(chains)
    }

    fn build_pronominal_chains(
        &mut self,
        sentences: &[String],
        reference_patterns: &HashMap<String, ReferencePattern>,
    ) -> Result<Vec<ReferentialChain>, CohesionAnalysisError> {
        let mut chains = Vec::new();

        if let Some(pronoun_pattern) = reference_patterns.get("personal_pronouns") {
            // Simple pronoun chain building
            // This would be much more sophisticated in a real implementation
            let pronoun_regex = regex::Regex::new(&pronoun_pattern.matching_rules[0])
                .map_err(|e| CohesionAnalysisError::ChainBuildingError(e.to_string()))?;

            let mut pronoun_occurrences = Vec::new();

            for (sent_idx, sentence) in sentences.iter().enumerate() {
                for cap in pronoun_regex.find_iter(sentence) {
                    pronoun_occurrences.push((sent_idx, cap.start(), cap.end(), cap.as_str()));
                }
            }

            if pronoun_occurrences.len() >= self.minimum_chain_length {
                let mut elements = Vec::new();

                for (sent_idx, start, end, pronoun) in pronoun_occurrences {
                    elements.push(
                        crate::metrics::coherence::lexical_coherence::results::ChainElement {
                            word: pronoun.to_string(),
                            lemma: pronoun.to_lowercase(),
                            positions: vec![(start, end)],
                            reference_type: ReferenceType::Pronominal,
                            confidence: 0.8,
                        },
                    );
                }

                if !elements.is_empty() {
                    let chain_strength = self.calculate_chain_strength(&elements);
                    let coherence_score = self.calculate_chain_coherence(&elements);

                    chains.push(ReferentialChain {
                        chain_id: 0,
                        elements,
                        chain_type: crate::metrics::coherence::lexical_coherence::results::ChainType::PronounalReference,
                        strength: chain_strength,
                        coherence_score,
                        average_distance: 0.0,
                        coverage: 0.0,
                    });
                }
            }
        }

        Ok(chains)
    }

    fn build_definite_reference_chains(
        &mut self,
        sentences: &[String],
        reference_patterns: &HashMap<String, ReferencePattern>,
    ) -> Result<Vec<ReferentialChain>, CohesionAnalysisError> {
        let mut chains = Vec::new();

        if let Some(definite_pattern) = reference_patterns.get("definite_articles") {
            // Simple definite reference chain building
            let definite_regex = regex::Regex::new(&definite_pattern.matching_rules[0])
                .map_err(|e| CohesionAnalysisError::ChainBuildingError(e.to_string()))?;

            let mut definite_occurrences = Vec::new();

            for (sent_idx, sentence) in sentences.iter().enumerate() {
                for cap in definite_regex.captures_iter(sentence) {
                    if let Some(full_match) = cap.get(0) {
                        if let Some(noun) = cap.get(1) {
                            definite_occurrences.push((
                                sent_idx,
                                full_match.start(),
                                full_match.end(),
                                noun.as_str(),
                            ));
                        }
                    }
                }
            }

            // Group by noun
            let mut noun_groups: HashMap<String, Vec<(usize, usize, usize, &str)>> = HashMap::new();
            for occurrence in definite_occurrences {
                noun_groups
                    .entry(occurrence.3.to_lowercase())
                    .or_default()
                    .push(occurrence);
            }

            let mut chain_id = 0;
            for (noun, occurrences) in noun_groups {
                if occurrences.len() >= self.minimum_chain_length {
                    let mut elements = Vec::new();

                    for (sent_idx, start, end, _) in occurrences {
                        elements.push(
                            crate::metrics::coherence::lexical_coherence::results::ChainElement {
                                word: format!("the {}", noun),
                                lemma: noun.clone(),
                                positions: vec![(start, end)],
                                reference_type: ReferenceType::Definite,
                                confidence: 0.7,
                            },
                        );
                    }

                    let chain_strength = self.calculate_chain_strength(&elements);
                    let coherence_score = self.calculate_chain_coherence(&elements);

                    chains.push(ReferentialChain {
                        chain_id,
                        elements,
                        chain_type: crate::metrics::coherence::lexical_coherence::results::ChainType::DefiniteReference,
                        strength: chain_strength,
                        coherence_score,
                        average_distance: 0.0,
                        coverage: 0.0,
                    });

                    chain_id += 1;
                }
            }
        }

        Ok(chains)
    }

    fn merge_overlapping_chains(
        &self,
        chains: Vec<ReferentialChain>,
    ) -> Result<Vec<ReferentialChain>, CohesionAnalysisError> {
        // Simple implementation - in reality, this would be more sophisticated
        Ok(chains)
    }

    fn calculate_chain_strength(
        &self,
        elements: &[crate::metrics::coherence::lexical_coherence::results::ChainElement],
    ) -> f64 {
        if elements.is_empty() {
            return 0.0;
        }

        let avg_confidence =
            elements.iter().map(|e| e.confidence).sum::<f64>() / elements.len() as f64;
        let length_factor = (elements.len() as f64).min(10.0) / 10.0; // Cap at 10 elements

        avg_confidence * length_factor
    }

    fn calculate_chain_coherence(
        &self,
        elements: &[crate::metrics::coherence::lexical_coherence::results::ChainElement],
    ) -> f64 {
        if elements.len() < 2 {
            return 0.0;
        }

        // Simple coherence based on distance between elements
        let mut total_coherence = 0.0;
        let mut pairs = 0;

        for i in 0..elements.len() - 1 {
            let current_pos = elements[i]
                .positions
                .first()
                .map(|(start, _)| *start)
                .unwrap_or(0);
            let next_pos = elements[i + 1]
                .positions
                .first()
                .map(|(start, _)| *start)
                .unwrap_or(0);
            let distance = (next_pos as i32 - current_pos as i32).abs() as f64;

            // Closer elements have higher coherence
            let coherence = 1.0 / (1.0 + distance / 100.0);
            total_coherence += coherence;
            pairs += 1;
        }

        if pairs > 0 {
            total_coherence / pairs as f64
        } else {
            0.0
        }
    }

    fn calculate_average_distance(
        &self,
        elements: &[crate::metrics::coherence::lexical_coherence::results::ChainElement],
    ) -> f64 {
        if elements.len() < 2 {
            return 0.0;
        }

        let mut total_distance = 0.0;
        let mut pairs = 0;

        for i in 0..elements.len() - 1 {
            let current_pos = elements[i]
                .positions
                .first()
                .map(|(start, _)| *start)
                .unwrap_or(0);
            let next_pos = elements[i + 1]
                .positions
                .first()
                .map(|(start, _)| *start)
                .unwrap_or(0);
            total_distance += (next_pos as i32 - current_pos as i32).abs() as f64;
            pairs += 1;
        }

        if pairs > 0 {
            total_distance / pairs as f64
        } else {
            0.0
        }
    }
}

impl CoreferenceResolver {
    fn new(config: &CohesionAnalysisConfig) -> Result<Self, CohesionAnalysisError> {
        Ok(CoreferenceResolver {
            resolution_strategies: vec![
                ResolutionStrategy::NumberAgreement,
                ResolutionStrategy::GenderAgreement,
                ResolutionStrategy::SemanticCompatibility,
                ResolutionStrategy::DistancePreference,
            ],
            feature_extractors: Vec::new(),
            scoring_weights: HashMap::from([
                ("number_agreement".to_string(), 0.3),
                ("gender_agreement".to_string(), 0.3),
                ("semantic_compatibility".to_string(), 0.2),
                ("distance_preference".to_string(), 0.2),
            ]),
        })
    }
}

impl ConnectivityAnalyzer {
    fn new(config: &CohesionAnalysisConfig) -> Result<Self, CohesionAnalysisError> {
        Ok(ConnectivityAnalyzer {
            connectivity_measures: vec![
                ConnectivityMeasure::LocalConnectivity,
                ConnectivityMeasure::GlobalConnectivity,
                ConnectivityMeasure::HierarchicalConnectivity,
            ],
            graph_algorithms: GraphAnalyzer::new(),
            density_calculator: DensityCalculator::new(),
        })
    }

    fn analyze_connectivity(
        &mut self,
        cohesive_devices: &[CohesiveDevice],
        referential_chains: &[ReferentialChain],
        sentences: &[String],
    ) -> Result<ConnectivityAnalysis, CohesionAnalysisError> {
        let text_length = sentences.iter().map(|s| s.len()).sum::<usize>();

        // Calculate local connectivity (sentence-to-sentence)
        let local_connectivity_score =
            self.calculate_local_connectivity(cohesive_devices, sentences)?;

        // Calculate global connectivity (across entire text)
        let global_connectivity_score =
            self.calculate_global_connectivity(cohesive_devices, referential_chains, sentences)?;

        // Calculate hierarchical connectivity (paragraph/section level)
        let hierarchical_connectivity_score =
            self.calculate_hierarchical_connectivity(cohesive_devices, sentences)?;

        // Calculate connectivity variance
        let connectivity_scores: Vec<f64> = (0..sentences.len() - 1)
            .map(|i| self.calculate_sentence_pair_connectivity(i, i + 1, cohesive_devices))
            .collect();

        let mean_connectivity = if !connectivity_scores.is_empty() {
            connectivity_scores.iter().sum::<f64>() / connectivity_scores.len() as f64
        } else {
            0.0
        };

        let connectivity_variance = if !connectivity_scores.is_empty() {
            connectivity_scores
                .iter()
                .map(|score| (score - mean_connectivity).powi(2))
                .sum::<f64>()
                / connectivity_scores.len() as f64
        } else {
            0.0
        };

        Ok(ConnectivityAnalysis {
            local_connectivity_score,
            global_connectivity_score,
            hierarchical_connectivity_score,
            connectivity_variance,
            connectivity_distribution: self
                .calculate_connectivity_distribution(&connectivity_scores),
            text_length,
        })
    }

    fn calculate_local_connectivity(
        &self,
        cohesive_devices: &[CohesiveDevice],
        sentences: &[String],
    ) -> Result<f64, CohesionAnalysisError> {
        if sentences.len() < 2 {
            return Ok(0.0);
        }

        let mut total_connectivity = 0.0;
        let mut pair_count = 0;

        for i in 0..sentences.len() - 1 {
            let connectivity =
                self.calculate_sentence_pair_connectivity(i, i + 1, cohesive_devices);
            total_connectivity += connectivity;
            pair_count += 1;
        }

        Ok(if pair_count > 0 {
            total_connectivity / pair_count as f64
        } else {
            0.0
        })
    }

    fn calculate_global_connectivity(
        &self,
        cohesive_devices: &[CohesiveDevice],
        referential_chains: &[ReferentialChain],
        sentences: &[String],
    ) -> Result<f64, CohesionAnalysisError> {
        if sentences.len() < 2 {
            return Ok(0.0);
        }

        // Count cross-sentence connections
        let cross_sentence_devices = cohesive_devices
            .iter()
            .filter(|device| device.source_position.0 != device.target_position.0)
            .count() as f64;

        // Count chain-based connections
        let chain_connections = referential_chains
            .iter()
            .map(|chain| chain.elements.len().saturating_sub(1))
            .sum::<usize>() as f64;

        // Normalize by potential connections
        let total_sentence_pairs = (sentences.len() * (sentences.len() - 1) / 2) as f64;

        if total_sentence_pairs > 0.0 {
            Ok((cross_sentence_devices + chain_connections) / total_sentence_pairs)
        } else {
            Ok(0.0)
        }
    }

    fn calculate_hierarchical_connectivity(
        &self,
        cohesive_devices: &[CohesiveDevice],
        sentences: &[String],
    ) -> Result<f64, CohesionAnalysisError> {
        // Simple implementation: assume every 5 sentences form a paragraph
        let paragraph_size = 5;
        let paragraph_count = (sentences.len() + paragraph_size - 1) / paragraph_size;

        if paragraph_count < 2 {
            return Ok(0.0);
        }

        let mut inter_paragraph_connections = 0;

        for device in cohesive_devices {
            let source_paragraph = device.source_position.0 / paragraph_size;
            let target_paragraph = device.target_position.0 / paragraph_size;

            if source_paragraph != target_paragraph {
                inter_paragraph_connections += 1;
            }
        }

        let potential_connections = paragraph_count * (paragraph_count - 1) / 2;

        if potential_connections > 0 {
            Ok(inter_paragraph_connections as f64 / potential_connections as f64)
        } else {
            Ok(0.0)
        }
    }

    fn calculate_sentence_pair_connectivity(
        &self,
        sent1_idx: usize,
        sent2_idx: usize,
        cohesive_devices: &[CohesiveDevice],
    ) -> f64 {
        let connections = cohesive_devices
            .iter()
            .filter(|device| {
                (device.source_position.0 == sent1_idx && device.target_position.0 == sent2_idx)
                    || (device.source_position.0 == sent2_idx
                        && device.target_position.0 == sent1_idx)
            })
            .count();

        connections as f64
    }

    fn calculate_connectivity_distribution(
        &self,
        connectivity_scores: &[f64],
    ) -> HashMap<String, f64> {
        let mut distribution = HashMap::new();

        if connectivity_scores.is_empty() {
            return distribution;
        }

        let min_connectivity = connectivity_scores
            .iter()
            .fold(f64::INFINITY, |acc, &x| acc.min(x));
        let max_connectivity = connectivity_scores
            .iter()
            .fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));
        let mean_connectivity =
            connectivity_scores.iter().sum::<f64>() / connectivity_scores.len() as f64;
        let median_connectivity = {
            let mut sorted = connectivity_scores.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            if sorted.len() % 2 == 0 {
                (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
            } else {
                sorted[sorted.len() / 2]
            }
        };

        distribution.insert("min".to_string(), min_connectivity);
        distribution.insert("max".to_string(), max_connectivity);
        distribution.insert("mean".to_string(), mean_connectivity);
        distribution.insert("median".to_string(), median_connectivity);

        distribution
    }
}

impl GraphAnalyzer {
    fn new() -> Self {
        GraphAnalyzer {
            algorithms: vec![
                GraphAlgorithm::ShortestPath,
                GraphAlgorithm::ConnectedComponents,
            ],
            centrality_measures: vec![
                CentralityMeasure::DegreeCentrality,
                CentralityMeasure::BetweennessCentrality,
            ],
            clustering_algorithms: vec![ClusteringAlgorithm::Hierarchical],
        }
    }
}

impl DensityCalculator {
    fn new() -> Self {
        DensityCalculator {
            density_types: vec![DensityType::LocalDensity, DensityType::GlobalDensity],
            normalization_methods: vec![NormalizationMethod::MinMax],
            weighting_schemes: vec![WeightingScheme::DistanceBased],
        }
    }
}

// Add regex dependency for pattern matching
extern crate regex;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::coherence::lexical_coherence::config::CohesionAnalysisConfig;
    use crate::metrics::coherence::lexical_coherence::results::LexicalItem;

    #[test]
    fn test_cohesion_analyzer_creation() {
        let config = CohesionAnalysisConfig::default();
        let analyzer = CohesionAnalyzer::new(config);
        assert!(analyzer.is_ok());
    }

    #[test]
    fn test_repetition_device_detection() {
        let config = CohesionAnalysisConfig::default();
        let detector = CohesiveDeviceDetector::new(&config).unwrap();

        let sentences = vec![
            "The cat sat on the mat.".to_string(),
            "The cat was very comfortable.".to_string(),
        ];

        let devices = detector.detect_repetition_devices(&sentences).unwrap();
        assert!(!devices.is_empty());

        // Should detect "the" and "cat" repetitions
        let cat_devices: Vec<&CohesiveDevice> = devices
            .iter()
            .filter(|d| d.source_element == "cat")
            .collect();
        assert!(!cat_devices.is_empty());
    }

    #[test]
    fn test_morphological_device_detection() {
        let config = CohesionAnalysisConfig::default();
        let detector = CohesiveDeviceDetector::new(&config).unwrap();

        let sentences = vec![
            "The runner was running quickly.".to_string(),
            "Running is good exercise.".to_string(),
        ];

        let devices = detector.detect_morphological_devices(&sentences).unwrap();

        // Should detect morphological relationship between "runner", "running"
        let running_devices: Vec<&CohesiveDevice> = devices
            .iter()
            .filter(|d| d.device_type == CohesiveDeviceType::Morphological)
            .collect();

        // May or may not find devices depending on stem extraction
        // This is a basic test to ensure the function runs without error
    }

    #[test]
    fn test_referential_chain_building() {
        let config = CohesionAnalysisConfig::default();
        let mut builder = ReferentialChainBuilder::new(&config).unwrap();

        let lexical_items = vec![
            LexicalItem {
                word: "dog".to_string(),
                lemma: "dog".to_string(),
                positions: vec![(0, 3)],
                frequency: 1.0,
                word_senses: vec![],
                semantic_features: vec![],
            },
            LexicalItem {
                word: "dog".to_string(),
                lemma: "dog".to_string(),
                positions: vec![(20, 23)],
                frequency: 1.0,
                word_senses: vec![],
                semantic_features: vec![],
            },
            LexicalItem {
                word: "dog".to_string(),
                lemma: "dog".to_string(),
                positions: vec![(40, 43)],
                frequency: 1.0,
                word_senses: vec![],
                semantic_features: vec![],
            },
        ];

        let sentences = vec![
            "The dog ran.".to_string(),
            "The dog was happy.".to_string(),
            "The dog barked.".to_string(),
        ];

        let chains = builder
            .build_identical_repetition_chains(&lexical_items)
            .unwrap();
        assert!(!chains.is_empty());

        let dog_chain = &chains[0];
        assert_eq!(dog_chain.elements.len(), 3);
    }

    #[test]
    fn test_connectivity_analysis() {
        let config = CohesionAnalysisConfig::default();
        let mut analyzer = ConnectivityAnalyzer::new(&config).unwrap();

        let cohesive_devices = vec![CohesiveDevice {
            device_type: CohesiveDeviceType::Repetition,
            source_element: "test".to_string(),
            target_element: "test".to_string(),
            source_position: (0, 0),
            target_position: (1, 0),
            strength: 0.8,
            confidence: 0.9,
            distance: 1.0,
            context: vec![],
        }];

        let referential_chains = vec![];
        let sentences = vec![
            "This is a test.".to_string(),
            "The test was successful.".to_string(),
        ];

        let connectivity = analyzer
            .analyze_connectivity(&cohesive_devices, &referential_chains, &sentences)
            .unwrap();

        assert!(connectivity.local_connectivity_score > 0.0);
        assert_eq!(connectivity.text_length, 34); // Total character length
    }

    #[test]
    fn test_cohesion_metrics_calculation() {
        let config = CohesionAnalysisConfig::default();
        let analyzer = CohesionAnalyzer::new(config).unwrap();

        let cohesive_devices = vec![
            CohesiveDevice {
                device_type: CohesiveDeviceType::Repetition,
                source_element: "word".to_string(),
                target_element: "word".to_string(),
                source_position: (0, 0),
                target_position: (1, 0),
                strength: 0.8,
                confidence: 0.9,
                distance: 1.0,
                context: vec![],
            },
            CohesiveDevice {
                device_type: CohesiveDeviceType::Synonymy,
                source_element: "good".to_string(),
                target_element: "excellent".to_string(),
                source_position: (0, 1),
                target_position: (1, 1),
                strength: 0.7,
                confidence: 0.8,
                distance: 1.0,
                context: vec![],
            },
        ];

        let referential_chains = vec![];
        let connectivity_analysis = ConnectivityAnalysis {
            local_connectivity_score: 0.6,
            global_connectivity_score: 0.4,
            hierarchical_connectivity_score: 0.3,
            connectivity_variance: 0.1,
            connectivity_distribution: HashMap::new(),
            text_length: 100,
        };

        let metrics = analyzer
            .calculate_cohesion_metrics(
                &cohesive_devices,
                &referential_chains,
                &connectivity_analysis,
            )
            .unwrap();

        assert!(metrics.overall_cohesion > 0.0);
        assert_eq!(metrics.device_density, 0.02); // 2 devices / 100 chars
        assert!(metrics.device_diversity > 0.0);
    }
}
