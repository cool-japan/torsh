//! Supporting types for neural_architecture_search::types
//!
//! Extracted to keep the parent module under 2000 lines (SplitRS policy).

use super::*;

/// Performance predictor for early architecture evaluation
pub struct PerformancePredictor {
    /// Model type for prediction
    #[allow(dead_code)]
    predictor_type: PredictorType,
    /// Training data for the predictor
    #[allow(dead_code)]
    training_data: Vec<(ArchitectureEncoding, PerformanceMetrics)>,
    /// Prediction accuracy metrics
    #[allow(dead_code)]
    prediction_accuracy: PredictionAccuracy,
    /// Feature extractor for architectures
    #[allow(dead_code)]
    feature_extractor: ArchitectureFeatureExtractor,
}
impl PerformancePredictor {
    pub(super) fn new() -> Self {
        Self {
            predictor_type: PredictorType::MLP {
                hidden_dims: vec![128, 64, 32],
                dropout_rate: 0.1,
            },
            training_data: Vec::new(),
            prediction_accuracy: PredictionAccuracy {
                mae: 0.0,
                rmse: 0.0,
                r2: 0.0,
                kendall_tau: 0.0,
            },
            feature_extractor: ArchitectureFeatureExtractor::new(),
        }
    }
}
/// Pooling operation types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PoolingType {
    Max,
    Average,
    Adaptive,
    GlobalAverage,
    GlobalMax,
}
/// Graph aggregation methods for GNN predictors
#[derive(Debug, Clone)]
pub enum GraphAggregation {
    Mean,
    Sum,
    Max,
    Attention,
    Set2Set,
}
/// Feature extraction methods
#[derive(Debug, Clone)]
pub enum FeatureExtractionMethod {
    /// Graph-based features
    GraphStructural {
        include_centrality: bool,
        include_motifs: bool,
        include_spectral: bool,
    },
    /// Hand-crafted architectural features
    HandCrafted {
        include_operation_counts: bool,
        include_depth_width: bool,
        include_connectivity: bool,
    },
    /// Learned embeddings
    Learned {
        embedding_dim: usize,
        model_path: String,
    },
}
/// Prediction accuracy tracking
#[derive(Debug, Clone)]
pub struct PredictionAccuracy {
    pub mae: f64,
    pub rmse: f64,
    pub r2: f64,
    pub kendall_tau: f64,
}
/// Mutation operations record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutationRecord {
    pub mutation_type: MutationType,
    pub applied_at: std::time::SystemTime,
    pub success: bool,
    pub fitness_change: f64,
}
/// Hardware constraints for architecture search
#[derive(Debug, Clone)]
pub struct HardwareConstraints {
    /// Target hardware platform
    pub target_platform: HardwarePlatform,
    /// Maximum latency constraint (ms)
    pub max_latency_ms: Option<f64>,
    /// Maximum memory constraint (MB)
    pub max_memory_mb: Option<f64>,
    /// Maximum energy constraint (mJ)
    pub max_energy_mj: Option<f64>,
    /// Maximum model size (MB)
    pub max_model_size_mb: Option<f64>,
    /// Hardware-specific optimizations
    pub hardware_optimizations: Vec<HardwareOptimization>,
}
/// Activation function types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ActivationType {
    ReLU,
    LeakyReLU { alpha_options: Vec<f32> },
    ELU { alpha_options: Vec<f32> },
    Swish,
    GELU,
    Mish,
    Sigmoid,
    Tanh,
    PReLU,
    Custom { name: String },
}
/// Skip connection strategies
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SkipConnectionType {
    /// Simple addition
    Add,
    /// Concatenation
    Concat,
    /// Weighted combination
    WeightedSum,
    /// Attention-based gating
    AttentionGated,
    /// None
    None,
}
/// Cloud instance types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CloudInstanceType {
    CPU,
    GPU,
    TPU,
    FPGA,
}
/// Normalization types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NormalizationType {
    BatchNorm,
    LayerNorm,
    GroupNorm { group_options: Vec<usize> },
    InstanceNorm,
    None,
}
/// Microcontroller types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MicrocontrollerType {
    ArmCortexM,
    ESP32,
    Arduino,
    Custom { specs: String },
}
/// Comprehensive search analytics
#[derive(Debug, Clone)]
pub struct SearchAnalytics {
    /// Convergence curve
    pub convergence_curve: Vec<f64>,
    /// Diversity evolution
    pub diversity_evolution: Vec<f64>,
    /// Architecture distribution
    pub architecture_distribution: HashMap<String, usize>,
    /// Performance predictions vs actual
    pub prediction_accuracy: PredictionAccuracy,
}
/// Types of mutations for evolutionary search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MutationType {
    /// Add a new layer
    AddLayer {
        layer_type: LayerType,
        position: usize,
    },
    /// Remove an existing layer
    RemoveLayer { position: usize },
    /// Change layer type
    ChangeLayerType {
        position: usize,
        new_type: LayerType,
    },
    /// Modify layer parameters
    ModifyParameters {
        position: usize,
        parameter_changes: HashMap<String, f64>,
    },
    /// Add skip connection
    AddSkipConnection { from: usize, to: usize },
    /// Remove skip connection
    RemoveSkipConnection { from: usize, to: usize },
    /// Change activation function
    ChangeActivation {
        position: usize,
        new_activation: ActivationType,
    },
    /// Modify network depth
    ChangeDepth { depth_change: i32 },
    /// Modify network width
    ChangeWidth {
        layer_index: usize,
        width_change: i32,
    },
}
/// Hardware-specific optimizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HardwareOptimization {
    /// Quantization-friendly architectures
    QuantizationFriendly,
    /// Pruning-friendly structures
    PruningFriendly,
    /// Low-precision arithmetic
    LowPrecision,
    /// Memory-efficient operations
    MemoryEfficient,
    /// Parallel-friendly structures
    ParallelFriendly,
}
/// Residual block variants
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ResidualBlockType {
    Basic,
    Bottleneck,
    PreActivation,
    SEBlock,
    CBAM,
}
/// Layer types available in search space
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LayerType {
    /// Convolutional layers with various kernel sizes
    Conv2d {
        kernel_sizes: Vec<usize>,
        stride_options: Vec<usize>,
    },
    /// Depthwise separable convolutions
    DepthwiseConv2d { kernel_sizes: Vec<usize> },
    /// Dilated convolutions
    DilatedConv2d {
        kernel_sizes: Vec<usize>,
        dilations: Vec<usize>,
    },
    /// Group convolutions
    GroupConv2d {
        kernel_sizes: Vec<usize>,
        group_options: Vec<usize>,
    },
    /// Linear/Dense layers
    Linear { hidden_sizes: Vec<usize> },
    /// Attention mechanisms
    Attention {
        head_options: Vec<usize>,
        dim_options: Vec<usize>,
    },
    /// Pooling operations
    Pooling {
        pool_types: Vec<PoolingType>,
        kernel_sizes: Vec<usize>,
    },
    /// Residual blocks
    ResidualBlock { block_types: Vec<ResidualBlockType> },
    /// Mobile-optimized blocks
    MobileBlock { expansion_ratios: Vec<f32> },
    /// Transformer blocks
    TransformerBlock {
        head_options: Vec<usize>,
        ff_multipliers: Vec<f32>,
    },
    /// Custom operation
    Custom {
        operation_name: String,
        parameter_ranges: HashMap<String, ParameterRange>,
    },
}
/// Architecture encoding for prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitectureEncoding {
    /// Adjacency matrix representation
    pub adjacency_matrix: Vec<Vec<f64>>,
    /// Node features (operation types, parameters)
    pub node_features: Vec<Vec<f64>>,
    /// Edge features (connection types)
    pub edge_features: Vec<Vec<f64>>,
    /// Global features (depth, width, parameter count)
    pub global_features: Vec<f64>,
}
/// Population management for evolutionary search
pub struct PopulationManager {
    /// Current population
    pub(super) population: Vec<CandidateArchitecture>,
    /// Population size limit
    #[allow(dead_code)]
    pub(super) max_population_size: usize,
    /// Fitness tracking
    #[allow(dead_code)]
    pub(super) fitness_history: Vec<Vec<f64>>,
    /// Diversity metrics
    #[allow(dead_code)]
    pub(super) diversity_metrics: DiversityMetrics,
}
impl PopulationManager {
    pub(super) fn new() -> Self {
        Self {
            population: Vec::new(),
            max_population_size: 50,
            fitness_history: Vec::new(),
            diversity_metrics: DiversityMetrics {
                structural_diversity: 0.0,
                performance_diversity: 0.0,
                functional_diversity: 0.0,
                age_diversity: 0.0,
            },
        }
    }
    pub(super) fn initialize_population(
        &mut self,
        search_space: &ArchitectureSearchSpace,
        size: usize,
    ) -> Result<()> {
        self.population.clear();
        let mut rng = thread_rng();
        for _ in 0..size {
            let mut graph = FxGraph::new();
            let depth = rng.gen_range(search_space.depth_range.0..=search_space.depth_range.1);
            let input_node = graph.add_node(Node::Input("input".to_string()));
            let mut prev_node = input_node;
            for i in 0..depth {
                if search_space.layer_types.is_empty() {
                    break;
                }
                let layer_idx = rng.gen_range(0..search_space.layer_types.len());
                let layer_type = &search_space.layer_types[layer_idx];
                let operation_name = match layer_type {
                    LayerType::Conv2d { .. } => "conv2d",
                    LayerType::DepthwiseConv2d { .. } => "depthwise_conv2d",
                    LayerType::DilatedConv2d { .. } => "dilated_conv2d",
                    LayerType::GroupConv2d { .. } => "group_conv2d",
                    LayerType::Linear { .. } => "linear",
                    LayerType::Attention { .. } => "attention",
                    LayerType::Pooling { .. } => "pooling",
                    LayerType::ResidualBlock { .. } => "residual_block",
                    LayerType::MobileBlock { .. } => "mobile_block",
                    LayerType::TransformerBlock { .. } => "transformer_block",
                    LayerType::Custom { operation_name, .. } => operation_name.as_str(),
                }
                .to_string();
                let layer_node =
                    graph.add_node(Node::Call(operation_name, vec![format!("layer_{}", i)]));
                graph.add_edge(
                    prev_node,
                    layer_node,
                    crate::Edge {
                        name: "data".to_string(),
                    },
                );
                prev_node = layer_node;
            }
            let output_node = graph.add_node(Node::Output);
            graph.add_edge(
                prev_node,
                output_node,
                crate::Edge {
                    name: "output".to_string(),
                },
            );
            let encoding = ArchitectureEncoding {
                adjacency_matrix: vec![vec![0.0; depth + 2]; depth + 2],
                node_features: vec![vec![rng.random::<f64>(); 10]; depth + 2],
                edge_features: vec![vec![1.0]; depth + 1],
                global_features: vec![depth as f64, (depth + 2) as f64, (depth + 1) as f64, 0.0],
            };
            self.population.push(CandidateArchitecture {
                graph,
                encoding,
                fitness: vec![rng.random::<f64>()],
                age: 0,
                parents: vec![],
                id: uuid::Uuid::new_v4().to_string(),
                mutation_history: vec![],
            });
        }
        Ok(())
    }
    pub(super) fn get_best_fitness(&self) -> f64 {
        self.population
            .iter()
            .map(|arch| arch.fitness.iter().fold(0.0f64, |acc, &x| acc.max(x)))
            .fold(f64::NEG_INFINITY, |acc, x| acc.max(x))
    }
}
/// Search progress tracking
#[derive(Debug, Clone)]
pub struct SearchProgressMetrics {
    /// Number of architectures evaluated
    pub architectures_evaluated: usize,
    /// Search time elapsed
    pub search_time: Duration,
    /// Best fitness achieved
    pub best_fitness: f64,
    /// Convergence metrics
    pub convergence_rate: f64,
    /// Diversity evolution
    pub diversity_over_time: Vec<f64>,
    /// Pareto front size evolution
    pub pareto_front_evolution: Vec<usize>,
}
/// Reward function for RL-based search
#[derive(Debug, Clone)]
pub enum RewardFunction {
    Accuracy,
    AccuracyLatencyTradeoff { latency_weight: f64 },
    MultiObjective { weights: ObjectiveWeights },
    Custom { function_name: String },
}
/// Search results containing best architectures and analytics
#[derive(Debug, Clone)]
pub struct SearchResults {
    /// Best architecture found
    pub best_architecture: CandidateArchitecture,
    /// Pareto front of architectures (multi-objective)
    pub pareto_front: Vec<CandidateArchitecture>,
    /// Search analytics
    pub search_analytics: SearchAnalytics,
    /// Total search time
    pub total_search_time: Duration,
}
impl SearchResults {
    pub(super) fn new() -> Self {
        let mut dummy_graph = FxGraph::new();
        let input = dummy_graph.add_node(Node::Input("x".to_string()));
        let output = dummy_graph.add_node(Node::Output);
        dummy_graph.add_edge(
            input,
            output,
            crate::Edge {
                name: "direct".to_string(),
            },
        );
        Self {
            best_architecture: CandidateArchitecture {
                graph: dummy_graph,
                encoding: ArchitectureEncoding {
                    adjacency_matrix: vec![vec![0.0, 1.0], vec![0.0, 0.0]],
                    node_features: vec![vec![1.0; 10]; 2],
                    edge_features: vec![vec![1.0]],
                    global_features: vec![2.0, 1.0, 1.0, 0.0],
                },
                fitness: vec![0.0],
                age: 0,
                parents: vec![],
                id: uuid::Uuid::new_v4().to_string(),
                mutation_history: vec![],
            },
            pareto_front: Vec::new(),
            search_analytics: SearchAnalytics {
                convergence_curve: Vec::new(),
                diversity_evolution: Vec::new(),
                architecture_distribution: HashMap::new(),
                prediction_accuracy: PredictionAccuracy {
                    mae: 0.0,
                    rmse: 0.0,
                    r2: 0.0,
                    kendall_tau: 0.0,
                },
            },
            total_search_time: Duration::from_secs(0),
        }
    }
}
/// Search strategy configuration
#[derive(Debug, Clone)]
pub enum SearchStrategy {
    /// Differentiable Architecture Search (DARTS)
    DARTS {
        learning_rate: f64,
        weight_decay: f64,
        temperature: f64,
        gradient_clip: f64,
    },
    /// Evolutionary Search
    Evolutionary {
        population_size: usize,
        mutation_rate: f64,
        crossover_rate: f64,
        selection_strategy: SelectionStrategy,
    },
    /// Reinforcement Learning based
    ReinforcementLearning {
        controller_type: ControllerType,
        reward_function: RewardFunction,
        exploration_rate: f64,
    },
    /// Random Search (baseline)
    Random { max_iterations: usize },
    /// Progressive Search (starts simple, increases complexity)
    Progressive {
        stages: Vec<ProgressiveStage>,
        complexity_increase_rate: f64,
    },
    /// Bayesian Optimization
    BayesianOptimization {
        acquisition_function: AcquisitionFunction,
        kernel_type: KernelType,
        exploration_exploitation_tradeoff: f64,
    },
}
/// Target hardware platforms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HardwarePlatform {
    /// Mobile devices
    Mobile { device_type: MobileDeviceType },
    /// Edge devices
    Edge {
        compute_capability: EdgeComputeCapability,
    },
    /// Cloud/Server
    Cloud { instance_type: CloudInstanceType },
    /// Embedded systems
    Embedded {
        microcontroller_type: MicrocontrollerType,
    },
    /// Custom hardware
    Custom {
        specifications: HardwareSpecifications,
    },
}
/// Progressive search stages
#[derive(Debug, Clone)]
pub struct ProgressiveStage {
    pub max_depth: usize,
    pub max_width: usize,
    pub allowed_operations: Vec<LayerType>,
    pub duration_epochs: usize,
}
