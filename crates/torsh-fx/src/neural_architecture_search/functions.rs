//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::FxGraph;
use std::collections::HashMap;
use torsh_core::error::Result;

use super::types::{
    ActivationType, ArchitectureSearchSpace, ConnectionPattern, HardwareConstraints,
    HardwareOptimization, HardwarePlatform, LayerType, MobileDeviceType, NeuralArchitectureSearch,
    NormalizationType, ObjectiveWeights, PoolingType, SearchResults, SearchStrategy,
    SelectionStrategy, SkipConnectionType,
};

/// Convenience function to create a default search space
pub fn create_default_search_space() -> ArchitectureSearchSpace {
    ArchitectureSearchSpace {
        layer_types: vec![
            LayerType::Conv2d {
                kernel_sizes: vec![3, 5, 7],
                stride_options: vec![1, 2],
            },
            LayerType::DepthwiseConv2d {
                kernel_sizes: vec![3, 5],
            },
            LayerType::Linear {
                hidden_sizes: vec![64, 128, 256, 512],
            },
            LayerType::Attention {
                head_options: vec![4, 8, 16],
                dim_options: vec![64, 128, 256],
            },
            LayerType::Pooling {
                pool_types: vec![PoolingType::Max, PoolingType::Average],
                kernel_sizes: vec![2, 3],
            },
        ],
        depth_range: (3, 20),
        width_constraints: vec![
            (0, (32, 512)),
            (1, (16, 256)),
            (2, (64, 1024)),
            (3, (64, 512)),
            (4, (2, 8)),
        ],
        connection_patterns: vec![
            ConnectionPattern::Sequential,
            ConnectionPattern::Skip { max_distance: 3 },
            ConnectionPattern::Residual,
        ],
        activation_functions: vec![
            ActivationType::ReLU,
            ActivationType::Swish,
            ActivationType::GELU,
        ],
        normalization_options: vec![
            NormalizationType::BatchNorm,
            NormalizationType::LayerNorm,
            NormalizationType::None,
        ],
        skip_connection_strategies: vec![
            SkipConnectionType::Add,
            SkipConnectionType::Concat,
            SkipConnectionType::None,
        ],
    }
}
/// Convenience function to create mobile-optimized constraints
pub fn create_mobile_constraints() -> HardwareConstraints {
    HardwareConstraints {
        target_platform: HardwarePlatform::Mobile {
            device_type: MobileDeviceType::Smartphone,
        },
        max_latency_ms: Some(50.0),
        max_memory_mb: Some(100.0),
        max_energy_mj: Some(10.0),
        max_model_size_mb: Some(10.0),
        hardware_optimizations: vec![
            HardwareOptimization::QuantizationFriendly,
            HardwareOptimization::MemoryEfficient,
        ],
    }
}
/// Convenience function to start NAS with default settings
pub async fn start_neural_architecture_search(
    _initial_graph: FxGraph,
    target_platform: HardwarePlatform,
) -> Result<SearchResults> {
    let search_space = create_default_search_space();
    let search_strategy = SearchStrategy::Evolutionary {
        population_size: 50,
        mutation_rate: 0.1,
        crossover_rate: 0.8,
        selection_strategy: SelectionStrategy::Tournament { tournament_size: 3 },
    };
    let hardware_constraints = HardwareConstraints {
        target_platform: target_platform.clone(),
        max_latency_ms: Some(100.0),
        max_memory_mb: Some(500.0),
        max_energy_mj: Some(50.0),
        max_model_size_mb: Some(50.0),
        hardware_optimizations: vec![
            HardwareOptimization::QuantizationFriendly,
            HardwareOptimization::ParallelFriendly,
        ],
    };
    let objective_weights = ObjectiveWeights {
        accuracy: 0.6,
        latency: 0.2,
        memory: 0.1,
        energy: 0.05,
        model_size: 0.05,
        custom_objectives: HashMap::new(),
    };
    let nas = NeuralArchitectureSearch::new(
        search_space,
        search_strategy,
        hardware_constraints,
        objective_weights,
    );
    println!("🚀 Starting automated neural architecture search...");
    println!("🎯 Target: Optimal architecture for {:?}", target_platform);
    nas.search(100).await
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::tracer::ModuleTracer;
    #[test]
    fn test_nas_creation() {
        let search_space = create_default_search_space();
        let search_strategy = SearchStrategy::Random { max_iterations: 10 };
        let hardware_constraints = create_mobile_constraints();
        let objective_weights = ObjectiveWeights {
            accuracy: 0.8,
            latency: 0.2,
            memory: 0.0,
            energy: 0.0,
            model_size: 0.0,
            custom_objectives: HashMap::new(),
        };
        let nas = NeuralArchitectureSearch::new(
            search_space,
            search_strategy,
            hardware_constraints,
            objective_weights,
        );
        assert!(nas.search_space.layer_types.len() > 0);
        assert!(nas.search_space.depth_range.0 < nas.search_space.depth_range.1);
    }
    #[tokio::test]
    async fn test_random_search() {
        let search_space = create_default_search_space();
        let search_strategy = SearchStrategy::Random { max_iterations: 5 };
        let hardware_constraints = create_mobile_constraints();
        let objective_weights = ObjectiveWeights {
            accuracy: 1.0,
            latency: 0.0,
            memory: 0.0,
            energy: 0.0,
            model_size: 0.0,
            custom_objectives: HashMap::new(),
        };
        let nas = NeuralArchitectureSearch::new(
            search_space,
            search_strategy,
            hardware_constraints,
            objective_weights,
        );
        let results = nas.search(5).await;
        assert!(results.is_ok());
    }
    #[test]
    fn test_architecture_generation() {
        let search_space = create_default_search_space();
        let search_strategy = SearchStrategy::Random { max_iterations: 10 };
        let hardware_constraints = create_mobile_constraints();
        let objective_weights = ObjectiveWeights {
            accuracy: 1.0,
            latency: 0.0,
            memory: 0.0,
            energy: 0.0,
            model_size: 0.0,
            custom_objectives: HashMap::new(),
        };
        let nas = NeuralArchitectureSearch::new(
            search_space,
            search_strategy,
            hardware_constraints,
            objective_weights,
        );
        let architecture = nas.generate_random_architecture();
        assert!(architecture.is_ok());
        let arch = architecture.unwrap();
        assert!(arch.graph.node_count() >= 2);
        assert!(!arch.id.is_empty());
    }
    #[test]
    fn test_architecture_encoding() {
        let mut tracer = ModuleTracer::new();
        tracer.add_input("x");
        tracer.add_call("conv2d", vec!["x".to_string()]);
        tracer.add_call("relu", vec!["node_0".to_string()]);
        tracer.add_output("node_1");
        let graph = tracer.finalize();
        let search_space = create_default_search_space();
        let search_strategy = SearchStrategy::Random { max_iterations: 10 };
        let hardware_constraints = create_mobile_constraints();
        let objective_weights = ObjectiveWeights {
            accuracy: 1.0,
            latency: 0.0,
            memory: 0.0,
            energy: 0.0,
            model_size: 0.0,
            custom_objectives: HashMap::new(),
        };
        let nas = NeuralArchitectureSearch::new(
            search_space,
            search_strategy,
            hardware_constraints,
            objective_weights,
        );
        let encoding = nas.encode_architecture(&graph);
        assert!(encoding.is_ok());
        let enc = encoding.unwrap();
        assert_eq!(enc.adjacency_matrix.len(), graph.node_count());
        assert_eq!(enc.node_features.len(), graph.node_count());
        assert_eq!(enc.global_features.len(), 4);
    }
}
