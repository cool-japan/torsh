//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::metrics::coherence::lexical_coherence::config::{
    CohesionAnalysisConfig, CohesionDeviceType,
};
use crate::metrics::coherence::lexical_coherence::results::{
    CohesionAnalysisResult, CohesionMetrics, CohesiveDevice, ConnectivityAnalysis,
    LexicalItem, ReferentialChain,
};
use std::collections::{HashMap, HashSet, VecDeque};

use super::types::{CohesionAnalyzer, CohesiveDeviceDetector, ConnectivityAnalyzer, ReferentialChainBuilder};

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
        let detector = CohesiveDeviceDetector::new(&config).expect("Cohesive Device Detector should succeed");
        let sentences = vec![
            "The cat sat on the mat.".to_string(), "The cat was very comfortable."
            .to_string(),
        ];
        let devices = detector.detect_repetition_devices(&sentences).expect("repetition device detection should succeed");
        assert!(! devices.is_empty());
        let cat_devices: Vec<&CohesiveDevice> = devices
            .iter()
            .filter(|d| d.source_element == "cat")
            .collect();
        assert!(! cat_devices.is_empty());
    }
    #[test]
    fn test_morphological_device_detection() {
        let config = CohesionAnalysisConfig::default();
        let detector = CohesiveDeviceDetector::new(&config).expect("Cohesive Device Detector should succeed");
        let sentences = vec![
            "The runner was running quickly.".to_string(), "Running is good exercise."
            .to_string(),
        ];
        let devices = detector.detect_morphological_devices(&sentences).expect("morphological device detection should succeed");
        let running_devices: Vec<&CohesiveDevice> = devices
            .iter()
            .filter(|d| d.device_type == CohesiveDeviceType::Morphological)
            .collect();
    }
    #[test]
    fn test_referential_chain_building() {
        let config = CohesionAnalysisConfig::default();
        let mut builder = ReferentialChainBuilder::new(&config).expect("Referential Chain Builder should succeed");
        let lexical_items = vec![
            LexicalItem { word : "dog".to_string(), lemma : "dog".to_string(), positions
            : vec![(0, 3)], frequency : 1.0, word_senses : vec![], semantic_features :
            vec![], }, LexicalItem { word : "dog".to_string(), lemma : "dog".to_string(),
            positions : vec![(20, 23)], frequency : 1.0, word_senses : vec![],
            semantic_features : vec![], }, LexicalItem { word : "dog".to_string(), lemma
            : "dog".to_string(), positions : vec![(40, 43)], frequency : 1.0, word_senses
            : vec![], semantic_features : vec![], },
        ];
        let sentences = vec![
            "The dog ran.".to_string(), "The dog was happy.".to_string(),
            "The dog barked.".to_string(),
        ];
        let chains = builder.build_identical_repetition_chains(&lexical_items).expect("build identical repetition chains should succeed");
        assert!(! chains.is_empty());
        let dog_chain = &chains[0];
        assert_eq!(dog_chain.elements.len(), 3);
    }
    #[test]
    fn test_connectivity_analysis() {
        let config = CohesionAnalysisConfig::default();
        let mut analyzer = ConnectivityAnalyzer::new(&config).expect("Connectivity Analyzer should succeed");
        let cohesive_devices = vec![
            CohesiveDevice { device_type : CohesiveDeviceType::Repetition, source_element
            : "test".to_string(), target_element : "test".to_string(), source_position :
            (0, 0), target_position : (1, 0), strength : 0.8, confidence : 0.9, distance
            : 1.0, context : vec![], }
        ];
        let referential_chains = vec![];
        let sentences = vec![
            "This is a test.".to_string(), "The test was successful.".to_string(),
        ];
        let connectivity = analyzer
            .analyze_connectivity(&cohesive_devices, &referential_chains, &sentences)
            .expect("operation should succeed");
        assert!(connectivity.local_connectivity_score > 0.0);
        assert_eq!(connectivity.text_length, 34);
    }
    #[test]
    fn test_cohesion_metrics_calculation() {
        let config = CohesionAnalysisConfig::default();
        let analyzer = CohesionAnalyzer::new(config).expect("Cohesion Analyzer should succeed");
        let cohesive_devices = vec![
            CohesiveDevice { device_type : CohesiveDeviceType::Repetition, source_element
            : "word".to_string(), target_element : "word".to_string(), source_position :
            (0, 0), target_position : (1, 0), strength : 0.8, confidence : 0.9, distance
            : 1.0, context : vec![], }, CohesiveDevice { device_type :
            CohesiveDeviceType::Synonymy, source_element : "good".to_string(),
            target_element : "excellent".to_string(), source_position : (0, 1),
            target_position : (1, 1), strength : 0.7, confidence : 0.8, distance : 1.0,
            context : vec![], },
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
            .expect("operation should succeed");
        assert!(metrics.overall_cohesion > 0.0);
        assert_eq!(metrics.device_density, 0.02);
        assert!(metrics.device_diversity > 0.0);
    }
}
