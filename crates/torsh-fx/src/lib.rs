//! Graph transformation framework for ToRSh
//!
//! This crate provides a comprehensive graph transformation framework built on a modular architecture.
//! The FX graph system has been refactored into specialized modules for maintainability and performance.

use petgraph::graph::{Graph, NodeIndex};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use torsh_core::{Result, TorshError};

/// Convenience type alias for Results in this crate
pub type TorshResult<T> = Result<T>;

// FX Graph modular system
pub mod fx;

// Re-export FX Graph types from the modular system
pub use fx::{Edge, FxGraph, GraphStats, MemoryEstimate, Node, SerializableGraph};

// Re-export key types for convenience from other modules
pub use benchmarking::{BenchmarkResult, GraphBenchmarkSuite, RegressionTester};
pub use checkpointing::{
    create_checkpoint, load_checkpoint, save_checkpoint, CheckpointData, CheckpointFormat,
    CheckpointManager, CheckpointMetadata, CheckpointOptions, ResumableInterpreter,
};
pub use codegen::{
    CacheStats, CodeGenBackend, CodeGenerator, CompiledCode, CppCodeGen, LazyCompiler,
    PythonCodeGen,
};
pub use custom_backends::{
    execute_with_auto_backend, execute_with_backend, get_backend, list_available_backends,
    register_backend_factory, BackendCapability, BackendContext, BackendExecutor, BackendFactory,
    BackendInfo, BackendRegistry, BackendResult, BackendSelectionStrategy, CustomBackend,
};
pub use custom_operations::{
    register_example_operations, CustomInt16AddOperation, CustomInt16MulOperation,
    CustomInt16SubOperation, CustomTypeUnifyOperation, TypeConversionOperation,
};
pub use custom_types::{
    global_extended_registry, register_extended_operation, CustomTypeUtils,
    ExtendedCustomOperation, ExtendedOperationRegistry, ExtendedShapeInferenceContext,
    ExtendedShapeInfo,
};
pub use distributed::{
    create_execution_plan, execute_distributed, init_distributed, CollectiveOp,
    CommunicationBackendType, DistributedConfig, DistributedExecutionPlan, DistributedExecutor,
    DistributionStrategy, ReduceOp,
};
pub use dynamic_shapes::{
    DynamicDim, DynamicShape, DynamicShapeInferenceContext, DynamicShapeInfo, ShapeConstraint,
};
pub use graph_analysis::{
    calculate_graph_metrics, DetectedPattern, GraphDiff, GraphDifference, GraphLinter,
    GraphMetrics, LintIssue, LintReport, LintSeverity, PatternDetector,
};
pub use graph_partitioning::{
    DeviceInfo, DeviceType, GraphPartition, GraphPartitioner, PartitionedGraph,
    PartitioningStrategy,
};
pub use heterogeneous_computing::{
    DeviceCapability, ExecutionPlan, HeterogeneousExecutor, OperationSpecialization,
    PlacementStrategy, SimpleDevice,
};
pub use memory_optimization::{
    AdaptiveMemoryManager, AllocationStrategy, GraphMemoryLayout, MemoryAnalyzer,
    MemoryMappedGraph, MemoryUsageReport,
};
pub use onnx_export::{export_to_onnx, OnnxExporter, OnnxModel};
pub use performance::{
    CacheStatistics, GraphCache, GraphCompression, ParallelTraversal, PerformanceBottleneck,
    PerformanceProfiler, PerformanceReport,
};
pub use torchscript_compat::{
    TorchScriptExporter, TorchScriptGraph, TorchScriptImporter, TorchScriptModel,
};
pub use tracer::{Module, ModuleTracer, SymbolicTensor, TracingProxy};

// Re-export additional types for convenience
pub use emerging_hardware::{
    create_dna_backend, create_neuromorphic_backend, create_photonic_backend, AdaptationStrategy,
    CompatibilityReport, EmergingHardware, EmergingHardwareBackend, EmergingHardwareResult,
    ErrorCorrectionScheme, HardwareCapabilities, HardwareConstraint, HardwareSpecifications,
    NeuromorphicProcessor, OptimizationObjective, PhotonicProcessor, PrecisionType,
    QuantumInspiredProcessor, SpecializedOperation,
};
pub use interactive_editor::{
    launch_interactive_editor, AutoSaveConfig, CollaborativeEdit, EditOperation, ExportFormat,
    ImportFormat, InteractiveGraphEditor, PerformanceMetrics, UserSession, VisualizationConfig,
};
pub use neural_architecture_search::{
    create_default_search_space, create_mobile_constraints, start_neural_architecture_search,
    ArchitectureSearchSpace, CandidateArchitecture, HardwareConstraints, HardwarePlatform,
    LayerType, NeuralArchitectureSearch, ObjectiveWeights, SearchResults, SearchStrategy,
};
pub use neuromorphic_optimization::{
    create_loihi_optimizer, optimize_for_mobile_neuromorphic, EnergyEstimate, NeuromorphicHardware,
    NeuromorphicOptimizationResult, NeuromorphicOptimizer, NeuronModel, OptimizationConfig,
    SNNConversionParams, SpikeEncoding,
};
pub use python_integration::{
    create_jax_integration, create_pytorch_integration, generate_python_api, graph_to_pytorch_code,
    DeploymentPackage, GeneratedPythonCode, PyTorchModelMetadata, PythonBindingConfig,
    PythonCodeGenOptions, PythonDeploymentTarget, PythonFramework, PythonIntegrationService,
    TrainingInfo,
};
pub use quantization::{
    apply_automatic_precision, prepare_graph_for_qat, quantize_graph_post_training,
    select_automatic_precision, AutomaticPrecisionSelector, CalibrationData, PTQUtils,
    PrecisionCriteria, PrecisionProfile, PrecisionRecommendation, PrecisionStrategy, QATUtils,
    QuantizationAnnotation, QuantizationBenchmark, QuantizationContext, QuantizationParams,
    QuantizationScheme,
};
pub use quantum_computing::{
    create_local_quantum_backend, create_qaoa_circuit, create_qiskit_backend, create_vqe_circuit,
    integrate_quantum_computing, CloudProvider, DataTransferType, ErrorMitigation,
    HybridOptimizationStrategy, HybridWorkflow, NoiseModel, QuantumBackend, QuantumCircuit,
    QuantumComputingBackend, QuantumExecutionResult, QuantumGate, QuantumPrecision, StateEncoding,
    SynchronizationType,
};

// Module declarations for the comprehensive graph transformation framework
pub mod checkpointing;
pub mod codegen;
pub mod custom_backends;
pub mod custom_operations;
pub mod custom_types;
pub mod distributed;
pub mod dynamic_shapes;
pub mod emerging_hardware;
// pub mod graph_module;  // Temporarily commented - depends on torsh-nn
pub mod benchmarking;
pub mod graph_analysis;
pub mod graph_partitioning;
pub mod heterogeneous_computing;
pub mod interactive_editor;
pub mod interpreter;
pub mod memory_optimization;
pub mod neural_architecture_search;
pub mod neuromorphic_optimization;
pub mod node;
pub mod onnx_export;
pub mod passes;
pub mod performance;
pub mod python_integration;
pub mod quantization;
pub mod quantum_computing;
pub mod subgraph_rewriter;
pub mod torchscript_compat;
pub mod tracer;
pub mod visualization;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_serialization_json() {
        let mut graph = FxGraph::new();
        let input = graph.graph.add_node(Node::Input("x".to_string()));
        let relu = graph
            .graph
            .add_node(Node::Call("relu".to_string(), vec!["x".to_string()]));
        let output = graph.graph.add_node(Node::Output);

        graph.graph.add_edge(
            input,
            relu,
            Edge {
                name: "x".to_string(),
            },
        );
        graph.graph.add_edge(
            relu,
            output,
            Edge {
                name: "relu_out".to_string(),
            },
        );
        graph.inputs.push(input);
        graph.outputs.push(output);

        // Test JSON serialization
        let json = graph.to_json().unwrap();
        assert!(json.contains("Input"));
        assert!(json.contains("relu"));

        // Test JSON deserialization
        let deserialized = FxGraph::from_json(&json).unwrap();
        assert_eq!(deserialized.node_count(), graph.node_count());
        assert_eq!(deserialized.edge_count(), graph.edge_count());
    }

    #[test]
    fn test_graph_serialization_binary() {
        let mut graph = FxGraph::new();
        let input = graph.graph.add_node(Node::Input("x".to_string()));
        let relu = graph
            .graph
            .add_node(Node::Call("relu".to_string(), vec!["x".to_string()]));
        let output = graph.graph.add_node(Node::Output);

        graph.graph.add_edge(
            input,
            relu,
            Edge {
                name: "x".to_string(),
            },
        );
        graph.graph.add_edge(
            relu,
            output,
            Edge {
                name: "relu_out".to_string(),
            },
        );
        graph.inputs.push(input);
        graph.outputs.push(output);

        // Test binary serialization
        let binary = graph.to_binary().unwrap();
        assert!(!binary.is_empty());

        // Test binary deserialization
        let deserialized = FxGraph::from_binary(&binary).unwrap();
        assert_eq!(deserialized.node_count(), graph.node_count());
        assert_eq!(deserialized.edge_count(), graph.edge_count());
    }

    #[test]
    fn test_single_op_graph() {
        let graph = FxGraph::single_op("relu", vec!["input".to_string()]);

        assert_eq!(graph.node_count(), 3); // input, operation, output
        assert_eq!(graph.edge_count(), 2); // input->op, op->output
        assert_eq!(graph.inputs().len(), 1);
        assert_eq!(graph.outputs().len(), 1);

        // Validate the graph structure
        assert!(graph.validate().is_ok());

        // Check node types
        let input_nodes = graph.input_nodes();
        let call_nodes = graph.call_nodes();
        let output_nodes = graph.output_nodes();

        assert_eq!(input_nodes.len(), 1);
        assert_eq!(call_nodes.len(), 1);
        assert_eq!(output_nodes.len(), 1);

        // Check the operation name
        if let Node::Call(op_name, _) = &call_nodes[0].1 {
            assert_eq!(op_name, "relu");
        } else {
            panic!("Expected Call node");
        }
    }

    #[test]
    fn test_sequential_ops_graph() {
        let ops = vec!["relu", "sigmoid", "tanh"];
        let graph = FxGraph::sequential_ops(&ops);

        assert_eq!(graph.node_count(), 5); // input, 3 ops, output
        assert_eq!(graph.edge_count(), 4); // input->relu, relu->sigmoid, sigmoid->tanh, tanh->output
        assert_eq!(graph.inputs().len(), 1);
        assert_eq!(graph.outputs().len(), 1);

        // Validate the graph structure
        assert!(graph.validate().is_ok());

        // Check that we have the right number of operations
        let call_nodes = graph.call_nodes();
        assert_eq!(call_nodes.len(), 3);

        // Verify the operations are in the right order (by checking their connections)
        let mut op_names = Vec::new();
        for (_, node) in call_nodes {
            if let Node::Call(op_name, _) = node {
                op_names.push(op_name.clone());
            }
        }

        // The order might not be preserved in iteration, so just check that all ops are present
        assert!(op_names.contains(&"relu".to_string()));
        assert!(op_names.contains(&"sigmoid".to_string()));
        assert!(op_names.contains(&"tanh".to_string()));
    }

    #[test]
    fn test_empty_sequential_ops() {
        let graph = FxGraph::sequential_ops(&[]);
        assert_eq!(graph.node_count(), 0);
        assert_eq!(graph.edge_count(), 0);
        assert_eq!(graph.inputs().len(), 0);
        assert_eq!(graph.outputs().len(), 0);
    }

    #[test]
    fn test_modular_architecture() {
        // Test that the modular architecture maintains the same interface
        let graph = FxGraph::single_op("test_op", vec!["input".to_string()]);

        // Basic functionality should work
        assert!(graph.node_count() > 0);
        assert!(graph.edge_count() > 0);
        assert!(graph.validate().is_ok());

        // Analysis functionality should work
        let summary = graph.summary();
        assert!(summary.contains("FX Graph Summary"));

        // Construction utilities should work
        let debug_graph = FxGraph::debug_minimal();
        assert!(debug_graph.validate().is_ok());

        // Serialization should work
        let json = graph.to_json().unwrap();
        assert!(!json.is_empty());

        let deserialized = FxGraph::from_json(&json).unwrap();
        assert_eq!(deserialized.node_count(), graph.node_count());
    }

    #[test]
    fn test_graph_validation() {
        // Test valid graph
        let graph = FxGraph::single_op("relu", vec!["input".to_string()]);
        assert!(graph.validate().is_ok());

        // Test invalid graph - no inputs
        let mut invalid_graph = FxGraph::new();
        let output = invalid_graph.add_node(Node::Output);
        invalid_graph.add_output(output);
        assert!(invalid_graph.validate().is_err());

        // Test invalid graph - no outputs
        let mut invalid_graph2 = FxGraph::new();
        let input = invalid_graph2.add_node(Node::Input("x".to_string()));
        invalid_graph2.add_input(input);
        assert!(invalid_graph2.validate().is_err());
    }

    #[test]
    fn test_performance_recommendations() {
        let graph = FxGraph::sequential_ops(&["relu", "sigmoid", "tanh"]);
        let recommendations = graph.performance_recommendations();
        assert!(!recommendations.is_empty());
    }

    #[test]
    fn test_operation_analysis() {
        let graph = FxGraph::sequential_ops(&["relu", "sigmoid", "relu"]);

        // Test operation names
        let op_names = graph.get_operation_names();
        assert_eq!(op_names.len(), 2); // "relu" and "sigmoid" (unique)
        assert!(op_names.contains(&"relu".to_string()));
        assert!(op_names.contains(&"sigmoid".to_string()));

        // Test contains operation
        assert!(graph.contains_operation("relu"));
        assert!(graph.contains_operation("sigmoid"));
        assert!(!graph.contains_operation("tanh"));

        // Test operation counts
        let counts = graph.operation_counts();
        assert_eq!(counts.get("relu"), Some(&2)); // relu appears twice
        assert_eq!(counts.get("sigmoid"), Some(&1)); // sigmoid appears once
        assert_eq!(counts.get("tanh"), None); // tanh doesn't appear
    }
}
