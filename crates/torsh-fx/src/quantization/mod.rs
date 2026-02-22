//! Quantization support framework
//!
//! This module provides comprehensive quantization capabilities for FX graphs,
//! organized into focused sub-modules for better maintainability.

use crate::{FxGraph, TorshResult};
use petgraph::graph::NodeIndex;
use std::collections::HashMap;

// Sub-module declarations
pub mod benchmark;
pub mod context;
pub mod metrics;
pub mod precision;
pub mod ptq;
pub mod qat;
pub mod types;

// Re-export all public types from sub-modules
pub use benchmark::QuantizationBenchmark;
pub use context::QuantizationContext;
pub use metrics::QuantizationMetrics;
pub use precision::{
    apply_automatic_precision, select_automatic_precision, AutomaticPrecisionSelector,
    PrecisionCriteria, PrecisionProfile, PrecisionRecommendation, PrecisionStrategy,
};
pub use ptq::PTQUtils;
pub use qat::QATUtils;
pub use types::{CalibrationData, QuantizationAnnotation, QuantizationParams, QuantizationScheme};

// Convenience functions for quantization
pub fn prepare_graph_for_qat(
    graph: &mut FxGraph,
    scheme: QuantizationScheme,
) -> TorshResult<QuantizationContext> {
    QATUtils::prepare_qat(graph, scheme)
}

pub fn quantize_graph_post_training(
    graph: &mut FxGraph,
    calibration_data: HashMap<NodeIndex, Vec<f32>>,
    scheme: QuantizationScheme,
) -> TorshResult<QuantizationContext> {
    PTQUtils::quantize_post_training(graph, calibration_data, scheme)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tracer::ModuleTracer;
    use crate::Node;
    use petgraph::graph::NodeIndex;
    use petgraph::visit::IntoNodeReferences;

    #[test]
    fn test_quantization_params() {
        let params = QuantizationParams::symmetric(QuantizationScheme::Int8, 0.1);
        assert_eq!(params.scheme, QuantizationScheme::Int8);
        assert_eq!(params.scale, 0.1);
        assert_eq!(params.zero_point, 0);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);
    }

    #[test]
    fn test_calibration_data() {
        let mut data = CalibrationData::new();
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        data.update(&values);

        assert_eq!(data.min_val, 1.0);
        assert_eq!(data.max_val, 5.0);
        assert_eq!(data.sample_count, 5);

        let params = data.compute_params(QuantizationScheme::Int8);
        assert!(params.scale > 0.0);
    }

    #[test]
    fn test_quantization_context() {
        let mut context = QuantizationContext::new(QuantizationScheme::Int8);
        let node_idx = NodeIndex::new(0);

        // Test annotation
        let annotation = QuantizationAnnotation {
            input_params: vec![Some(QuantizationParams::symmetric(
                QuantizationScheme::Int8,
                1.0,
            ))],
            output_params: Some(QuantizationParams::symmetric(QuantizationScheme::Int8, 1.0)),
            calibration_data: None,
        };
        context.annotate_node(node_idx, annotation);

        assert!(context.get_annotation(node_idx).is_some());
    }

    #[test]
    fn test_qat_preparation() {
        let mut tracer = ModuleTracer::new();
        tracer.add_input("x");
        tracer.add_call("linear", vec!["x".to_string()]);
        tracer.add_call("relu", vec!["node_0".to_string()]);
        tracer.add_output("node_1");
        let mut graph = tracer.finalize();

        let context = prepare_graph_for_qat(&mut graph, QuantizationScheme::Int8);
        assert!(context.is_ok());
    }

    #[test]
    fn test_quantization_metrics() {
        let original = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let quantized = vec![1.1, 1.9, 3.1, 3.9, 5.1];

        let metrics = QuantizationBenchmark::measure_accuracy(&original, &quantized);
        assert!(metrics.mean_absolute_error > 0.0);
        assert!(metrics.max_absolute_error > 0.0);
        assert_eq!(metrics.sample_count, 5);
    }

    #[test]
    fn test_automatic_precision_selection() {
        let mut tracer = ModuleTracer::new();
        tracer.add_input("x");
        tracer.add_call("matmul", vec!["x".to_string()]);
        tracer.add_call("relu", vec!["node_0".to_string()]);
        tracer.add_call("attention", vec!["node_1".to_string()]);
        tracer.add_output("node_2");
        let graph = tracer.finalize();

        let selector = AutomaticPrecisionSelector::new(PrecisionCriteria::Balanced);
        let recommendations = selector.analyze_graph(&graph).unwrap();

        // Check that we got recommendations for all operations
        assert!(recommendations.len() >= 3);

        // Find recommendations for specific operations
        let mut matmul_rec = None;
        let mut relu_rec = None;
        let mut attention_rec = None;

        for (node_idx, node) in graph.graph.node_references() {
            if let Node::Call(op_name, _) = node {
                if let Some(rec) = recommendations.get(&node_idx) {
                    match op_name.as_str() {
                        "matmul" => matmul_rec = Some(rec),
                        "relu" => relu_rec = Some(rec),
                        "attention" => attention_rec = Some(rec),
                        _ => {}
                    }
                }
            }
        }

        // MatMul should prefer INT8 (quantization-friendly)
        if let Some(rec) = matmul_rec {
            assert_eq!(rec.scheme, QuantizationScheme::Int8);
            assert!(rec.confidence > 0.7);
        }

        // ReLU should prefer INT8 (very quantization-friendly)
        if let Some(rec) = relu_rec {
            assert_eq!(rec.scheme, QuantizationScheme::Int8);
            assert!(rec.confidence > 0.8);
        }

        // Attention should prefer INT16 (quantization-sensitive)
        if let Some(rec) = attention_rec {
            assert_eq!(rec.scheme, QuantizationScheme::Int16);
            assert!(rec.confidence > 0.6);
        }
    }

    #[test]
    fn test_precision_criteria_performance() {
        let mut tracer = ModuleTracer::new();
        tracer.add_input("x");
        tracer.add_call("conv2d", vec!["x".to_string()]);
        tracer.add_output("node_0");
        let graph = tracer.finalize();

        let selector = AutomaticPrecisionSelector::new(PrecisionCriteria::Performance);
        let recommendations = selector.analyze_graph(&graph).unwrap();

        // Performance-focused should prefer more aggressive quantization
        for (_, rec) in recommendations {
            // Should prefer INT8 for performance
            assert!(matches!(rec.scheme, QuantizationScheme::Int8));
        }
    }

    #[test]
    fn test_precision_criteria_accuracy() {
        let mut tracer = ModuleTracer::new();
        tracer.add_input("x");
        tracer.add_call("attention", vec!["x".to_string()]);
        tracer.add_output("node_0");
        let graph = tracer.finalize();

        let selector = AutomaticPrecisionSelector::new(PrecisionCriteria::Accuracy);
        let recommendations = selector.analyze_graph(&graph).unwrap();

        // Accuracy-focused should prefer more conservative quantization for sensitive operations
        for (_, rec) in recommendations {
            // Should prefer INT16 for accuracy on sensitive operations
            assert!(matches!(rec.scheme, QuantizationScheme::Int16));
        }
    }

    #[test]
    fn test_precision_criteria_custom() {
        let mut tracer = ModuleTracer::new();
        tracer.add_input("x");
        tracer.add_call("matmul", vec!["x".to_string()]);
        tracer.add_output("node_0");
        let graph = tracer.finalize();

        let criteria = PrecisionCriteria::Custom {
            max_accuracy_loss: 0.01,
            min_speedup: 2.0,
        };
        let selector = AutomaticPrecisionSelector::new(criteria);
        let recommendations = selector.analyze_graph(&graph).unwrap();

        // Custom criteria should be respected
        for (_, rec) in recommendations {
            assert!(rec.accuracy_loss <= 0.01);
            assert!(rec.speedup_ratio >= 2.0);
        }
    }

    #[test]
    fn test_apply_automatic_precision() {
        let mut tracer = ModuleTracer::new();
        tracer.add_input("x");
        tracer.add_call("linear", vec!["x".to_string()]);
        tracer.add_call("relu", vec!["node_0".to_string()]);
        tracer.add_output("node_1");
        let mut graph = tracer.finalize();

        let context = apply_automatic_precision(&mut graph, PrecisionCriteria::Balanced).unwrap();

        // Should have annotations for the operations
        assert!(!context.annotations().is_empty());

        // Check that annotations have valid quantization parameters
        for (_, annotation) in context.annotations() {
            assert!(!annotation.input_params.is_empty());
            assert!(annotation.output_params.is_some());
        }
    }

    #[test]
    fn test_precision_strategy_custom() {
        let strategy = PrecisionStrategy {
            int8_priority: 1.0,
            int16_priority: 0.5,
            dynamic_priority: 0.3,
            fp32_priority: 0.1,
            performance_weight: 0.8,
            accuracy_weight: 0.2,
        };

        let mut tracer = ModuleTracer::new();
        tracer.add_input("x");
        tracer.add_call("matmul", vec!["x".to_string()]);
        tracer.add_output("node_0");
        let graph = tracer.finalize();

        let selector =
            AutomaticPrecisionSelector::with_strategy(PrecisionCriteria::Balanced, strategy);
        let recommendations = selector.analyze_graph(&graph).unwrap();

        // Performance-weighted strategy should prefer INT8
        for (_, rec) in recommendations {
            assert_eq!(rec.scheme, QuantizationScheme::Int8);
        }
    }
}
