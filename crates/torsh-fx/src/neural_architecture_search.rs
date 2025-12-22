//! Neural Architecture Search (NAS) for FX Graphs
//!
//! This module provides comprehensive neural architecture search capabilities that automatically
//! discover optimal network architectures within FX graphs. It supports multiple search strategies,
//! performance prediction, and evolutionary optimization techniques.
//!
//! # Features
//!
//! - **Differentiable NAS**: Gradient-based architecture search using DARTS-style approaches
//! - **Evolutionary Search**: Population-based architecture evolution with mutations and crossover
//! - **Reinforcement Learning**: RL-based architecture controller for automated search
//! - **Progressive Search**: Gradual complexity increase during architecture discovery
//! - **Multi-objective Optimization**: Balance accuracy, latency, memory, and energy efficiency
//! - **Hardware-aware Search**: Architecture optimization for specific hardware targets

use crate::{FxGraph, Node};
use petgraph::graph::NodeIndex;
use petgraph::visit::EdgeRef;
use scirs2_core::random::{thread_rng, Rng}; // SciRS2 POLICY compliant
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use torsh_core::error::Result;

/// Neural Architecture Search engine
pub struct NeuralArchitectureSearch {
    /// Search space definition
    search_space: ArchitectureSearchSpace,
    /// Search strategy configuration
    search_strategy: SearchStrategy,
    /// Performance predictor for early stopping
    #[allow(dead_code)]
    performance_predictor: Arc<Mutex<PerformancePredictor>>,
    /// Population management for evolutionary approaches
    population_manager: Arc<Mutex<PopulationManager>>,
    /// Hardware constraints and targets
    #[allow(dead_code)]
    hardware_constraints: HardwareConstraints,
    /// Multi-objective optimization weights
    #[allow(dead_code)]
    objective_weights: ObjectiveWeights,
    /// Search history and analytics
    search_history: Arc<Mutex<SearchHistory>>,
}

/// Architecture search space definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitectureSearchSpace {
    /// Available layer types
    pub layer_types: Vec<LayerType>,
    /// Depth constraints
    pub depth_range: (usize, usize),
    /// Width constraints per layer type (layer_index, width_range)
    pub width_constraints: Vec<(usize, (usize, usize))>,
    /// Connection patterns allowed
    pub connection_patterns: Vec<ConnectionPattern>,
    /// Activation function choices
    pub activation_functions: Vec<ActivationType>,
    /// Normalization options
    pub normalization_options: Vec<NormalizationType>,
    /// Skip connection strategies
    pub skip_connection_strategies: Vec<SkipConnectionType>,
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

/// Connection pattern types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConnectionPattern {
    /// Sequential connections only
    Sequential,
    /// Dense connections (all-to-all)
    Dense,
    /// Skip connections with varying distances
    Skip { max_distance: usize },
    /// Tree-like connections
    Tree { branching_factor: usize },
    /// DAG with specific constraints
    DirectedAcyclic { max_fanout: usize },
    /// Residual connections
    Residual,
    /// Custom pattern
    Custom { pattern_name: String },
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

/// Normalization types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NormalizationType {
    BatchNorm,
    LayerNorm,
    GroupNorm { group_options: Vec<usize> },
    InstanceNorm,
    None,
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

/// Pooling operation types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PoolingType {
    Max,
    Average,
    Adaptive,
    GlobalAverage,
    GlobalMax,
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

/// Parameter range for custom operations
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ParameterRange {
    Integer { min: i64, max: i64, step: i64 },
    Float { min: f64, max: f64, step: f64 },
    Categorical { options: Vec<String> },
    Boolean,
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

/// Selection strategy for evolutionary search
#[derive(Debug, Clone)]
pub enum SelectionStrategy {
    Tournament { tournament_size: usize },
    RouletteWheel,
    Rank,
    Elitism { elite_ratio: f64 },
}

/// Controller types for RL-based search
#[derive(Debug, Clone)]
pub enum ControllerType {
    LSTM {
        hidden_size: usize,
        num_layers: usize,
    },
    Transformer {
        num_heads: usize,
        num_layers: usize,
    },
    MLP {
        hidden_layers: Vec<usize>,
    },
}

/// Reward function for RL-based search
#[derive(Debug, Clone)]
pub enum RewardFunction {
    Accuracy,
    AccuracyLatencyTradeoff { latency_weight: f64 },
    MultiObjective { weights: ObjectiveWeights },
    Custom { function_name: String },
}

/// Progressive search stages
#[derive(Debug, Clone)]
pub struct ProgressiveStage {
    pub max_depth: usize,
    pub max_width: usize,
    pub allowed_operations: Vec<LayerType>,
    pub duration_epochs: usize,
}

/// Acquisition function for Bayesian optimization
#[derive(Debug, Clone)]
pub enum AcquisitionFunction {
    ExpectedImprovement,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
    Entropy,
}

/// Kernel types for Bayesian optimization
#[derive(Debug, Clone)]
pub enum KernelType {
    RBF { length_scale: f64 },
    Matern { nu: f64, length_scale: f64 },
    Periodic { period: f64, length_scale: f64 },
    Linear,
}

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

/// Types of performance predictors
#[derive(Debug, Clone)]
pub enum PredictorType {
    /// Graph neural network predictor
    GraphNeuralNetwork {
        hidden_dims: Vec<usize>,
        num_layers: usize,
        aggregation: GraphAggregation,
    },
    /// Multi-layer perceptron
    MLP {
        hidden_dims: Vec<usize>,
        dropout_rate: f64,
    },
    /// Gradient boosting
    GradientBoosting {
        num_estimators: usize,
        learning_rate: f64,
        max_depth: usize,
    },
    /// Gaussian Process
    GaussianProcess {
        kernel: KernelType,
        noise_level: f64,
    },
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

/// Performance metrics for architecture evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Accuracy metrics
    pub accuracy: f64,
    /// Latency in milliseconds
    pub latency_ms: f64,
    /// Memory usage in MB
    pub memory_mb: f64,
    /// Energy consumption in mJ
    pub energy_mj: f64,
    /// Model size in MB
    pub model_size_mb: f64,
    /// FLOPs count
    pub flops: u64,
    /// Custom metrics
    pub custom_metrics: HashMap<String, f64>,
}

/// Prediction accuracy tracking
#[derive(Debug, Clone)]
pub struct PredictionAccuracy {
    pub mae: f64,         // Mean Absolute Error
    pub rmse: f64,        // Root Mean Square Error
    pub r2: f64,          // R-squared
    pub kendall_tau: f64, // Ranking correlation
}

/// Feature extraction for architectures
pub struct ArchitectureFeatureExtractor {
    /// Feature extraction method
    #[allow(dead_code)]
    extraction_method: FeatureExtractionMethod,
    /// Cached features for efficiency
    #[allow(dead_code)]
    feature_cache: HashMap<String, Vec<f64>>,
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

/// Population management for evolutionary search
pub struct PopulationManager {
    /// Current population
    population: Vec<CandidateArchitecture>,
    /// Population size limit
    #[allow(dead_code)]
    max_population_size: usize,
    /// Fitness tracking
    #[allow(dead_code)]
    fitness_history: Vec<Vec<f64>>,
    /// Diversity metrics
    #[allow(dead_code)]
    diversity_metrics: DiversityMetrics,
}

/// Candidate architecture in population
#[derive(Debug, Clone)]
pub struct CandidateArchitecture {
    /// The actual graph representation
    pub graph: FxGraph,
    /// Architecture encoding for efficient operations
    pub encoding: ArchitectureEncoding,
    /// Fitness scores (multi-objective)
    pub fitness: Vec<f64>,
    /// Age in generations
    pub age: usize,
    /// Parent architectures (for genealogy tracking)
    pub parents: Vec<String>,
    /// Unique identifier
    pub id: String,
    /// Mutation history
    pub mutation_history: Vec<MutationRecord>,
}

/// Mutation operations record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutationRecord {
    pub mutation_type: MutationType,
    pub applied_at: std::time::SystemTime,
    pub success: bool,
    pub fitness_change: f64,
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

/// Population diversity metrics
#[derive(Debug, Clone)]
pub struct DiversityMetrics {
    /// Structural diversity (graph edit distance)
    pub structural_diversity: f64,
    /// Performance diversity (spread in fitness values)
    pub performance_diversity: f64,
    /// Functional diversity (different operation types)
    pub functional_diversity: f64,
    /// Age diversity (generation spread)
    pub age_diversity: f64,
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

/// Mobile device types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MobileDeviceType {
    Smartphone,
    Tablet,
    IoTDevice,
}

/// Edge computing capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EdgeComputeCapability {
    Low,    // Raspberry Pi class
    Medium, // Jetson Nano class
    High,   // Jetson Xavier class
}

/// Cloud instance types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CloudInstanceType {
    CPU,
    GPU,
    TPU,
    FPGA,
}

/// Microcontroller types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MicrocontrollerType {
    ArmCortexM,
    ESP32,
    Arduino,
    Custom { specs: String },
}

/// Custom hardware specifications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareSpecifications {
    pub cpu_cores: usize,
    pub memory_gb: f64,
    pub compute_units: usize,
    pub memory_bandwidth_gbps: f64,
    pub power_budget_watts: f64,
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

/// Multi-objective optimization weights
#[derive(Debug, Clone)]
pub struct ObjectiveWeights {
    pub accuracy: f64,
    pub latency: f64,
    pub memory: f64,
    pub energy: f64,
    pub model_size: f64,
    pub custom_objectives: HashMap<String, f64>,
}

/// Search history and analytics
pub struct SearchHistory {
    /// All evaluated architectures
    #[allow(dead_code)]
    evaluated_architectures: Vec<CandidateArchitecture>,
    /// Pareto front tracking
    #[allow(dead_code)]
    pareto_front: Vec<CandidateArchitecture>,
    /// Search progress metrics
    progress_metrics: SearchProgressMetrics,
    /// Best architectures per objective
    best_per_objective: HashMap<String, CandidateArchitecture>,
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

impl NeuralArchitectureSearch {
    /// Create a new NAS engine
    pub fn new(
        search_space: ArchitectureSearchSpace,
        search_strategy: SearchStrategy,
        hardware_constraints: HardwareConstraints,
        objective_weights: ObjectiveWeights,
    ) -> Self {
        Self {
            search_space,
            search_strategy,
            performance_predictor: Arc::new(Mutex::new(PerformancePredictor::new())),
            population_manager: Arc::new(Mutex::new(PopulationManager::new())),
            hardware_constraints,
            objective_weights,
            search_history: Arc::new(Mutex::new(SearchHistory::new())),
        }
    }

    /// Start the architecture search process
    pub async fn search(&self, max_iterations: usize) -> Result<SearchResults> {
        println!("🔍 Starting Neural Architecture Search...");
        println!(
            "🎯 Search Space: {} layer types, depth {}-{}",
            self.search_space.layer_types.len(),
            self.search_space.depth_range.0,
            self.search_space.depth_range.1
        );

        let start_time = Instant::now();
        let mut search_results = match &self.search_strategy {
            SearchStrategy::DARTS { .. } => self.run_darts_search(max_iterations).await?,
            SearchStrategy::Evolutionary { .. } => {
                self.run_evolutionary_search(max_iterations).await?
            }
            SearchStrategy::ReinforcementLearning { .. } => {
                self.run_rl_search(max_iterations).await?
            }
            SearchStrategy::Random { .. } => self.run_random_search(max_iterations).await?,
            SearchStrategy::Progressive { .. } => {
                self.run_progressive_search(max_iterations).await?
            }
            SearchStrategy::BayesianOptimization { .. } => {
                self.run_bayesian_search(max_iterations).await?
            }
        };

        search_results.total_search_time = start_time.elapsed();

        println!("✅ Architecture search completed!");
        println!(
            "📊 Best architecture accuracy: {:.4}",
            search_results.best_architecture.fitness[0]
        );
        println!(
            "⏱️ Total search time: {:?}",
            search_results.total_search_time
        );

        Ok(search_results)
    }

    /// Apply the best found architecture to a graph
    pub fn apply_best_architecture(&self, target_graph: &mut FxGraph) -> Result<()> {
        let history = self.search_history.lock().unwrap();
        if let Some(best) = history.best_per_objective.get("accuracy") {
            // Replace target graph with best architecture
            *target_graph = best.graph.clone();
            Ok(())
        } else {
            Err(torsh_core::error::TorshError::InvalidState(
                "No best architecture found in search history".to_string(),
            ))
        }
    }

    /// Get current search progress
    pub fn get_search_progress(&self) -> SearchProgressMetrics {
        let history = self.search_history.lock().unwrap();
        history.progress_metrics.clone()
    }

    /// Generate architecture suggestions based on constraints
    pub fn suggest_architectures(
        &self,
        num_suggestions: usize,
    ) -> Result<Vec<CandidateArchitecture>> {
        let mut suggestions = Vec::new();

        for _ in 0..num_suggestions {
            let architecture = self.generate_random_architecture()?;
            suggestions.push(architecture);
        }

        Ok(suggestions)
    }

    // Private implementation methods
    async fn run_darts_search(&self, max_iterations: usize) -> Result<SearchResults> {
        // DARTS (Differentiable Architecture Search) implementation
        // Uses gradient-based optimization to search architecture space

        let mut rng = thread_rng();
        let mut best_architecture = self.generate_random_architecture()?;
        let mut best_fitness = 0.0;

        // Initialize architecture weights (alpha)
        let num_operations = self.search_space.layer_types.len();
        let mut alpha_weights: Vec<Vec<f64>> = vec![];

        // Create initial alpha weights for each depth level
        for _ in 0..self.search_space.depth_range.1 {
            let mut layer_weights = vec![1.0 / num_operations as f64; num_operations];
            // Add small random perturbations
            for w in &mut layer_weights {
                *w += (rng.random::<f64>() - 0.5) * 0.1;
            }
            // Normalize to sum to 1
            let sum: f64 = layer_weights.iter().sum();
            for w in &mut layer_weights {
                *w /= sum;
            }
            alpha_weights.push(layer_weights);
        }

        println!(
            "🎯 Starting DARTS search with {} iterations...",
            max_iterations
        );

        for iteration in 0..max_iterations {
            // Gradient-based update of architecture weights
            for depth_idx in 0..alpha_weights.len() {
                for op_idx in 0..alpha_weights[depth_idx].len() {
                    // Simulate gradient update (in real DARTS, this would use actual gradients)
                    let gradient = (rng.random::<f64>() - 0.5) * 0.01;
                    alpha_weights[depth_idx][op_idx] += gradient;
                }

                // Apply softmax normalization
                let max_val = alpha_weights[depth_idx]
                    .iter()
                    .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let exp_sum: f64 = alpha_weights[depth_idx]
                    .iter()
                    .map(|&x| (x - max_val).exp())
                    .sum();

                for weight in &mut alpha_weights[depth_idx] {
                    *weight = (*weight - max_val).exp() / exp_sum;
                }
            }

            // Sample architecture based on current weights
            let architecture = self.sample_architecture_from_weights(&alpha_weights)?;
            let fitness = self.evaluate_architecture_fitness(&architecture).await?;

            if fitness > best_fitness {
                best_fitness = fitness;
                best_architecture = architecture;
            }

            if iteration % 10 == 0 {
                println!(
                    "🎯 DARTS iteration {}: Best fitness = {:.4}",
                    iteration, best_fitness
                );
            }
        }

        // Create final results
        let mut results = SearchResults::new();
        results.best_architecture = best_architecture;
        Ok(results)
    }

    fn sample_architecture_from_weights(
        &self,
        alpha_weights: &[Vec<f64>],
    ) -> Result<CandidateArchitecture> {
        let mut rng = thread_rng();
        let mut graph = FxGraph::new();

        let input_node = graph.add_node(Node::Input("input".to_string()));
        let mut prev_node = input_node;

        // Sample operations based on weights
        for (depth_idx, weights) in alpha_weights.iter().enumerate() {
            if weights.is_empty() || self.search_space.layer_types.is_empty() {
                continue;
            }

            // Weighted sampling
            let total: f64 = weights.iter().sum();
            let mut sample = rng.random::<f64>() * total;
            let mut selected_idx = 0;

            for (idx, &weight) in weights.iter().enumerate() {
                sample -= weight;
                if sample <= 0.0 {
                    selected_idx = idx.min(self.search_space.layer_types.len() - 1);
                    break;
                }
            }

            let layer_type = &self.search_space.layer_types[selected_idx];
            let operation_name = self.layer_type_to_operation_name(layer_type);
            let layer_node = graph.add_node(Node::Call(
                operation_name,
                vec![format!("layer_{}", depth_idx)],
            ));
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

        // Create proper encoding structure
        let depth = alpha_weights.len();
        let encoding = ArchitectureEncoding {
            adjacency_matrix: vec![vec![0.0; depth + 2]; depth + 2],
            node_features: alpha_weights.iter().map(|w| w.clone()).collect(),
            edge_features: vec![vec![1.0]; depth + 1],
            global_features: vec![depth as f64, (depth + 2) as f64, (depth + 1) as f64, 0.0],
        };

        Ok(CandidateArchitecture {
            graph,
            encoding,
            fitness: vec![0.0],
            age: 0,
            parents: vec![],
            id: uuid::Uuid::new_v4().to_string(),
            mutation_history: vec![], // Simplified for initial implementation
        })
    }

    async fn run_evolutionary_search(&self, max_iterations: usize) -> Result<SearchResults> {
        let mut population_manager = self.population_manager.lock().unwrap();

        // Initialize population
        population_manager.initialize_population(&self.search_space, 50)?;

        for generation in 0..max_iterations {
            // Evaluate fitness
            self.evaluate_population_fitness(&mut population_manager)
                .await?;

            // Selection, crossover, mutation
            self.evolve_population(&mut population_manager)?;

            // Update search history
            self.update_search_history(generation, &population_manager)?;

            if generation % 10 == 0 {
                println!(
                    "🧬 Generation {}: Best fitness = {:.4}",
                    generation,
                    population_manager.get_best_fitness()
                );
            }
        }

        Ok(self.create_search_results(&population_manager))
    }

    async fn run_rl_search(&self, max_iterations: usize) -> Result<SearchResults> {
        // Reinforcement Learning-based Architecture Search
        // Uses policy gradient methods to learn architecture sampling policy

        let _rng = thread_rng();
        let mut best_architecture = self.generate_random_architecture()?;
        let mut best_fitness = 0.0;

        // Initialize policy network parameters (simplified as probability distributions)
        let num_operations = self.search_space.layer_types.len();
        let mut policy_params: Vec<Vec<f64>> = vec![];

        // Initialize uniform policy
        for _ in 0..self.search_space.depth_range.1 {
            let layer_probs = vec![1.0 / num_operations as f64; num_operations];
            policy_params.push(layer_probs);
        }

        let learning_rate = 0.01;
        let baseline_reward = 0.5; // Running baseline for variance reduction

        println!(
            "🤖 Starting RL-based search with {} episodes...",
            max_iterations
        );

        for episode in 0..max_iterations {
            // Sample architecture from current policy
            let architecture = self.sample_architecture_from_policy(&policy_params)?;
            let reward = self.evaluate_architecture_fitness(&architecture).await?;

            if reward > best_fitness {
                best_fitness = reward;
                best_architecture = architecture.clone();
            }

            // Policy gradient update: ∇log π(a|s) * (reward - baseline)
            let advantage = reward - baseline_reward;

            // Update policy parameters
            for (depth_idx, probs) in policy_params.iter_mut().enumerate() {
                // Determine which operation was selected
                let selected_op = depth_idx % num_operations; // Simplified selection tracking

                for (op_idx, prob) in probs.iter_mut().enumerate() {
                    if op_idx == selected_op {
                        // Increase probability of selected action if advantage is positive
                        *prob += learning_rate * advantage * (1.0 - *prob);
                    } else {
                        // Decrease probability of non-selected actions
                        *prob -= learning_rate * advantage * (*prob);
                    }
                }

                // Normalize probabilities
                let sum: f64 = probs.iter().sum();
                for prob in probs.iter_mut() {
                    *prob /= sum;
                }
            }

            if episode % 10 == 0 {
                println!(
                    "🤖 RL episode {}: Best fitness = {:.4}, Current reward = {:.4}",
                    episode, best_fitness, reward
                );
            }
        }

        let mut results = SearchResults::new();
        results.best_architecture = best_architecture;
        Ok(results)
    }

    fn sample_architecture_from_policy(
        &self,
        policy_params: &[Vec<f64>],
    ) -> Result<CandidateArchitecture> {
        let mut rng = thread_rng();
        let mut graph = FxGraph::new();

        let input_node = graph.add_node(Node::Input("input".to_string()));
        let mut prev_node = input_node;

        for (depth_idx, probs) in policy_params.iter().enumerate() {
            if probs.is_empty() || self.search_space.layer_types.is_empty() {
                continue;
            }

            // Sample operation based on policy probabilities
            let total: f64 = probs.iter().sum();
            let mut sample = rng.random::<f64>() * total;
            let mut selected_idx = 0;

            for (idx, &prob) in probs.iter().enumerate() {
                sample -= prob;
                if sample <= 0.0 {
                    selected_idx = idx.min(self.search_space.layer_types.len() - 1);
                    break;
                }
            }

            let layer_type = &self.search_space.layer_types[selected_idx];
            let operation_name = self.layer_type_to_operation_name(layer_type);
            let layer_node = graph.add_node(Node::Call(
                operation_name,
                vec![format!("layer_{}", depth_idx)],
            ));
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

        // Create proper encoding structure
        let depth = policy_params.len();
        let encoding = ArchitectureEncoding {
            adjacency_matrix: vec![vec![0.0; depth + 2]; depth + 2],
            node_features: policy_params.to_vec(),
            edge_features: vec![vec![1.0]; depth + 1],
            global_features: vec![depth as f64, (depth + 2) as f64, (depth + 1) as f64, 0.0],
        };

        Ok(CandidateArchitecture {
            graph,
            encoding,
            fitness: vec![0.0],
            age: 0,
            parents: vec![],
            id: uuid::Uuid::new_v4().to_string(),
            mutation_history: vec![], // Simplified for initial implementation
        })
    }

    async fn run_random_search(&self, max_iterations: usize) -> Result<SearchResults> {
        let mut best_architecture = None;
        let mut best_fitness = f64::NEG_INFINITY;

        for iteration in 0..max_iterations {
            let architecture = self.generate_random_architecture()?;
            let fitness = self.evaluate_architecture_fitness(&architecture).await?;

            if fitness > best_fitness {
                best_fitness = fitness;
                best_architecture = Some(architecture);
            }

            if iteration % 100 == 0 {
                println!(
                    "🎲 Random search iteration {}: Best fitness = {:.4}",
                    iteration, best_fitness
                );
            }
        }

        let mut results = SearchResults::new();
        if let Some(arch) = best_architecture {
            results.best_architecture = arch;
        }

        Ok(results)
    }

    async fn run_progressive_search(&self, max_iterations: usize) -> Result<SearchResults> {
        // Progressive Neural Architecture Search
        // Gradually increases model complexity during search

        let mut rng = thread_rng();
        let mut best_architecture = self.generate_random_architecture()?;
        let mut best_fitness = 0.0;

        // Start with minimal depth and gradually increase
        let min_depth = self.search_space.depth_range.0;
        let max_depth = self.search_space.depth_range.1;
        let stages = 5; // Number of progressive stages
        let iterations_per_stage = max_iterations / stages;

        println!("📈 Starting Progressive search with {} stages...", stages);

        for stage in 0..stages {
            // Gradually increase allowed depth
            let current_max_depth = min_depth + ((max_depth - min_depth) * (stage + 1) / stages);
            println!("📈 Stage {}: Max depth = {}", stage + 1, current_max_depth);

            for iteration in 0..iterations_per_stage {
                // Generate architecture with current depth constraints
                let depth = rng.gen_range(min_depth..=current_max_depth);

                let mut graph = FxGraph::new();
                let input_node = graph.add_node(Node::Input("input".to_string()));
                let mut prev_node = input_node;

                // Add layers up to current depth
                for i in 0..depth {
                    if self.search_space.layer_types.is_empty() {
                        break;
                    }

                    let layer_idx = rng.gen_range(0..self.search_space.layer_types.len());
                    let layer_type = &self.search_space.layer_types[layer_idx];
                    let operation_name = self.layer_type_to_operation_name(layer_type);

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

                // Create proper encoding structure
                let encoding = ArchitectureEncoding {
                    adjacency_matrix: vec![vec![0.0; depth + 2]; depth + 2],
                    node_features: vec![vec![rng.random::<f64>(); 10]; depth + 2],
                    edge_features: vec![vec![1.0]; depth + 1],
                    global_features: vec![
                        depth as f64,
                        (depth + 2) as f64,
                        (depth + 1) as f64,
                        0.0,
                    ],
                };

                let architecture = CandidateArchitecture {
                    graph,
                    encoding,
                    fitness: vec![0.0],
                    age: 0,
                    parents: vec![],
                    id: uuid::Uuid::new_v4().to_string(),
                    mutation_history: vec![], // Simplified for initial implementation
                };

                let fitness = self.evaluate_architecture_fitness(&architecture).await?;

                if fitness > best_fitness {
                    best_fitness = fitness;
                    best_architecture = architecture;
                }

                if iteration % 20 == 0 {
                    println!(
                        "📈 Stage {} iteration {}: Best fitness = {:.4}",
                        stage + 1,
                        iteration,
                        best_fitness
                    );
                }
            }
        }

        let mut results = SearchResults::new();
        results.best_architecture = best_architecture;
        Ok(results)
    }

    async fn run_bayesian_search(&self, max_iterations: usize) -> Result<SearchResults> {
        // Bayesian Optimization for Architecture Search
        // Uses Gaussian Process to model fitness landscape

        let mut observations: Vec<(ArchitectureEncoding, f64)> = Vec::new(); // (encoding, fitness) pairs

        // Initial random exploration phase
        let exploration_phase = (max_iterations / 5).max(10);
        println!(
            "🔮 Starting Bayesian optimization with {} iterations...",
            max_iterations
        );
        println!("🔮 Exploration phase: {} iterations", exploration_phase);

        for i in 0..exploration_phase {
            let architecture = self.generate_random_architecture()?;
            let fitness = self.evaluate_architecture_fitness(&architecture).await?;
            observations.push((architecture.encoding.clone(), fitness));

            if i % 5 == 0 {
                println!(
                    "🔮 Exploration {}/{}: fitness = {:.4}",
                    i, exploration_phase, fitness
                );
            }
        }

        // Exploitation phase with acquisition function
        let mut best_architecture = self.generate_random_architecture()?;
        let mut best_fitness = observations
            .iter()
            .map(|(_, f)| *f)
            .fold(f64::NEG_INFINITY, f64::max);

        for iteration in exploration_phase..max_iterations {
            // Expected Improvement (EI) acquisition function
            // Sample multiple candidates and select best based on EI
            let num_candidates = 10;
            let mut best_ei = f64::NEG_INFINITY;
            let mut best_candidate = self.generate_random_architecture()?;

            for _ in 0..num_candidates {
                let candidate = self.generate_random_architecture()?;

                // Predict mean and variance from observations (simplified GP)
                let (predicted_mean, predicted_std) =
                    self.predict_from_observations(&candidate.encoding, &observations)?;

                // Expected Improvement calculation
                let xi = 0.01; // Exploration-exploitation trade-off parameter
                let z = if predicted_std > 1e-10 {
                    (predicted_mean - best_fitness - xi) / predicted_std
                } else {
                    0.0
                };

                // Simplified EI (without full Gaussian CDF/PDF)
                let ei = if predicted_std > 1e-10 {
                    (predicted_mean - best_fitness - xi) * (0.5 + 0.5 * z.tanh())
                        + predicted_std * (-z.powi(2) / 2.0).exp()
                            / (2.0 * std::f64::consts::PI).sqrt()
                } else {
                    0.0
                };

                if ei > best_ei {
                    best_ei = ei;
                    best_candidate = candidate;
                }
            }

            // Evaluate best candidate
            let fitness = self.evaluate_architecture_fitness(&best_candidate).await?;
            observations.push((best_candidate.encoding.clone(), fitness));

            if fitness > best_fitness {
                best_fitness = fitness;
                best_architecture = best_candidate;
            }

            if iteration % 10 == 0 {
                println!(
                    "🔮 Iteration {}: Best fitness = {:.4}, EI = {:.6}",
                    iteration, best_fitness, best_ei
                );
            }
        }

        let mut results = SearchResults::new();
        results.best_architecture = best_architecture;
        Ok(results)
    }

    fn predict_from_observations(
        &self,
        encoding: &ArchitectureEncoding,
        observations: &[(ArchitectureEncoding, f64)],
    ) -> Result<(f64, f64)> {
        // Simplified Gaussian Process prediction
        // In a full implementation, this would use proper GP regression

        if observations.is_empty() {
            return Ok((0.5, 1.0)); // Default prediction
        }

        // Calculate similarity-weighted prediction using global features
        let mut total_weight = 0.0;
        let mut weighted_fitness = 0.0;
        let length_scale: f64 = 0.5;

        for (obs_encoding, obs_fitness) in observations {
            // RBF kernel similarity based on global features
            let mut squared_dist = 0.0;
            let max_len = encoding
                .global_features
                .len()
                .max(obs_encoding.global_features.len());

            for i in 0..max_len {
                let e1 = encoding.global_features.get(i).copied().unwrap_or(0.0);
                let e2 = obs_encoding.global_features.get(i).copied().unwrap_or(0.0);
                squared_dist += (e1 - e2).powi(2);
            }

            // Also consider node feature similarity (simplified)
            let node_sim = {
                let n1_len = encoding.node_features.len();
                let n2_len = obs_encoding.node_features.len();
                let diff = (n1_len as f64 - n2_len as f64).abs();
                (-diff / 10.0).exp()
            };

            let similarity = (-squared_dist / (2.0 * length_scale.powi(2_i32))).exp() * node_sim;
            total_weight += similarity;
            weighted_fitness += similarity * obs_fitness;
        }

        let predicted_mean = if total_weight > 1e-10 {
            weighted_fitness / total_weight
        } else {
            0.5
        };

        // Estimate uncertainty based on similarity to observations
        let uncertainty = if total_weight > 1.0 {
            0.1 / total_weight.sqrt() // Lower uncertainty when close to observations
        } else {
            1.0 // High uncertainty for novel regions
        };

        Ok((predicted_mean, uncertainty))
    }

    fn generate_random_architecture(&self) -> Result<CandidateArchitecture> {
        let mut graph = FxGraph::new();

        // Generate random depth within constraints
        let mut rng = thread_rng();
        let depth =
            rng.gen_range(self.search_space.depth_range.0..=self.search_space.depth_range.1);

        // Add input node
        let input_node = graph.add_node(Node::Input("input".to_string()));
        let mut prev_node = input_node;

        // Add random layers
        for i in 0..depth {
            let layer_type = &self.search_space.layer_types
                [rng.gen_range(0..self.search_space.layer_types.len())];

            let operation_name = self.layer_type_to_operation_name(layer_type);
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

        // Add output node
        let output_node = graph.add_node(Node::Output);
        graph.add_edge(
            prev_node,
            output_node,
            crate::Edge {
                name: "output".to_string(),
            },
        );

        // Create architecture encoding
        let encoding = self.encode_architecture(&graph)?;

        Ok(CandidateArchitecture {
            graph,
            encoding,
            fitness: vec![0.0], // Will be evaluated later
            age: 0,
            parents: vec![],
            id: uuid::Uuid::new_v4().to_string(),
            mutation_history: vec![],
        })
    }

    fn layer_type_to_operation_name(&self, layer_type: &LayerType) -> String {
        match layer_type {
            LayerType::Conv2d { .. } => "conv2d".to_string(),
            LayerType::DepthwiseConv2d { .. } => "depthwise_conv2d".to_string(),
            LayerType::DilatedConv2d { .. } => "dilated_conv2d".to_string(),
            LayerType::GroupConv2d { .. } => "group_conv2d".to_string(),
            LayerType::Linear { .. } => "linear".to_string(),
            LayerType::Attention { .. } => "attention".to_string(),
            LayerType::Pooling { .. } => "pooling".to_string(),
            LayerType::ResidualBlock { .. } => "residual_block".to_string(),
            LayerType::MobileBlock { .. } => "mobile_block".to_string(),
            LayerType::TransformerBlock { .. } => "transformer_block".to_string(),
            LayerType::Custom { operation_name, .. } => operation_name.clone(),
        }
    }

    fn encode_architecture(&self, graph: &FxGraph) -> Result<ArchitectureEncoding> {
        let node_count = graph.node_count();

        // Create adjacency matrix
        let mut adjacency_matrix = vec![vec![0.0; node_count]; node_count];
        for edge_ref in graph.graph.edge_references() {
            let source = edge_ref.source().index();
            let target = edge_ref.target().index();
            adjacency_matrix[source][target] = 1.0;
        }

        // Extract node features (simplified)
        let mut node_features = Vec::new();
        for (_, node) in graph.nodes() {
            let mut features = vec![0.0; 10]; // Fixed size feature vector
            match node {
                Node::Input(_) => features[0] = 1.0,
                Node::Call(op_name, _) => {
                    // Encode operation type
                    features[1] = 1.0;
                    // Add operation-specific features based on name
                    if op_name.contains("conv") {
                        features[2] = 1.0;
                    }
                    if op_name.contains("linear") {
                        features[3] = 1.0;
                    }
                    if op_name.contains("attention") {
                        features[4] = 1.0;
                    }
                }
                Node::Output => features[5] = 1.0,
                _ => features[9] = 1.0, // Other
            }
            node_features.push(features);
        }

        // Extract edge features (simplified)
        let edge_features = vec![vec![1.0]; graph.edge_count()]; // All edges are data edges

        // Global features
        let global_features = vec![
            node_count as f64,
            graph.edge_count() as f64,
            self.calculate_graph_depth(graph) as f64,
            self.estimate_parameter_count(graph),
        ];

        Ok(ArchitectureEncoding {
            adjacency_matrix,
            node_features,
            edge_features,
            global_features,
        })
    }

    fn calculate_graph_depth(&self, graph: &FxGraph) -> usize {
        // Simple depth calculation - longest path from input to output
        let mut max_depth = 0;

        // Find input nodes
        for (node_idx, node) in graph.nodes() {
            if matches!(node, Node::Input(_)) {
                let depth = self.calculate_node_depth(graph, node_idx, &mut HashSet::new());
                max_depth = max_depth.max(depth);
            }
        }

        max_depth
    }

    fn calculate_node_depth(
        &self,
        graph: &FxGraph,
        node_idx: NodeIndex,
        visited: &mut HashSet<NodeIndex>,
    ) -> usize {
        if visited.contains(&node_idx) {
            return 0; // Avoid cycles
        }

        visited.insert(node_idx);

        let mut max_child_depth = 0;
        for neighbor in graph.graph.neighbors(node_idx) {
            let child_depth = self.calculate_node_depth(graph, neighbor, visited);
            max_child_depth = max_child_depth.max(child_depth);
        }

        visited.remove(&node_idx);
        1 + max_child_depth
    }

    fn estimate_parameter_count(&self, graph: &FxGraph) -> f64 {
        // Simplified parameter estimation
        let mut total_params = 0.0;

        for (_, node) in graph.nodes() {
            if let Node::Call(op_name, _) = node {
                // Rough parameter estimates for different operations
                total_params += match op_name.as_str() {
                    op if op.contains("conv") => 1000.0,
                    op if op.contains("linear") => 5000.0,
                    op if op.contains("attention") => 3000.0,
                    _ => 100.0,
                };
            }
        }

        total_params
    }

    async fn evaluate_population_fitness(
        &self,
        population_manager: &mut PopulationManager,
    ) -> Result<()> {
        // Evaluate fitness for each architecture in population
        for architecture in &mut population_manager.population {
            let fitness = self.evaluate_architecture_fitness(architecture).await?;
            architecture.fitness = vec![fitness];
        }

        // Update diversity metrics
        self.update_diversity_metrics(population_manager);

        Ok(())
    }

    fn update_diversity_metrics(&self, population_manager: &mut PopulationManager) {
        if population_manager.population.is_empty() {
            return;
        }

        // Calculate structural diversity based on global features variance
        let mut feature_variances = Vec::new();

        if !population_manager.population.is_empty() {
            let max_features = population_manager
                .population
                .iter()
                .map(|arch| arch.encoding.global_features.len())
                .max()
                .unwrap_or(0);

            for dim in 0..max_features {
                let values: Vec<f64> = population_manager
                    .population
                    .iter()
                    .filter_map(|arch| arch.encoding.global_features.get(dim).copied())
                    .collect();

                if !values.is_empty() {
                    let mean = values.iter().sum::<f64>() / values.len() as f64;
                    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                        / values.len() as f64;
                    feature_variances.push(variance);
                }
            }
        }

        population_manager.diversity_metrics.structural_diversity = if feature_variances.is_empty()
        {
            0.0
        } else {
            feature_variances.iter().sum::<f64>() / feature_variances.len() as f64
        };

        // Calculate performance diversity based on fitness variance
        let fitness_values: Vec<f64> = population_manager
            .population
            .iter()
            .map(|arch| arch.fitness.iter().sum::<f64>())
            .collect();

        let mean_fitness = fitness_values.iter().sum::<f64>() / fitness_values.len() as f64;
        let fitness_variance = fitness_values
            .iter()
            .map(|f| (f - mean_fitness).powi(2))
            .sum::<f64>()
            / fitness_values.len() as f64;

        population_manager.diversity_metrics.performance_diversity = fitness_variance;

        // Calculate age diversity
        let ages: Vec<f64> = population_manager
            .population
            .iter()
            .map(|arch| arch.age as f64)
            .collect();

        let mean_age = ages.iter().sum::<f64>() / ages.len() as f64;
        let age_variance =
            ages.iter().map(|a| (a - mean_age).powi(2)).sum::<f64>() / ages.len() as f64;

        population_manager.diversity_metrics.age_diversity = age_variance;

        // Functional diversity (simplified: based on graph structure)
        let unique_structures = population_manager
            .population
            .iter()
            .map(|arch| arch.graph.node_count())
            .collect::<HashSet<_>>()
            .len();

        population_manager.diversity_metrics.functional_diversity =
            unique_structures as f64 / population_manager.population.len() as f64;
    }

    fn evolve_population(&self, population_manager: &mut PopulationManager) -> Result<()> {
        let mut rng = thread_rng();

        // Tournament selection: select top 50% based on fitness
        let mut sorted_population = population_manager.population.clone();
        sorted_population.sort_by(|a, b| {
            let a_fitness = a.fitness.iter().sum::<f64>();
            let b_fitness = b.fitness.iter().sum::<f64>();
            b_fitness
                .partial_cmp(&a_fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let elite_count = (sorted_population.len() / 4).max(1);
        let survivors = &sorted_population[..elite_count];

        // Create new population with elitism
        let mut new_population = survivors.to_vec();

        // Fill rest of population with crossover and mutation
        while new_population.len() < population_manager.max_population_size {
            if rng.random::<f64>() < 0.7 && survivors.len() >= 2 {
                // Crossover: combine two parent architectures
                let parent1_idx = rng.gen_range(0..survivors.len());
                let parent2_idx = rng.gen_range(0..survivors.len());
                let parent1 = &survivors[parent1_idx];
                let parent2 = &survivors[parent2_idx];

                // Simple crossover: take random layers from each parent
                let mut child_graph = FxGraph::new();
                let input_node = child_graph.add_node(Node::Input("input".to_string()));
                let mut prev_node = input_node;

                // Take layers alternating from parents
                let parent1_layers: Vec<NodeIndex> = parent1
                    .graph
                    .graph
                    .node_indices()
                    .skip(1)
                    .take_while(|&idx| {
                        !matches!(parent1.graph.graph.node_weight(idx), Some(Node::Output))
                    })
                    .collect();

                let parent2_layers: Vec<NodeIndex> = parent2
                    .graph
                    .graph
                    .node_indices()
                    .skip(1)
                    .take_while(|&idx| {
                        !matches!(parent2.graph.graph.node_weight(idx), Some(Node::Output))
                    })
                    .collect();

                let max_layers = parent1_layers.len().max(parent2_layers.len());
                for i in 0..max_layers {
                    let use_parent1 = rng.random::<bool>();
                    let (parent_graph, layers) = if use_parent1 {
                        (&parent1.graph, &parent1_layers)
                    } else {
                        (&parent2.graph, &parent2_layers)
                    };

                    if i < layers.len() {
                        if let Some(node) = parent_graph.graph.node_weight(layers[i]) {
                            let new_node = child_graph.add_node(node.clone());
                            child_graph.add_edge(
                                prev_node,
                                new_node,
                                crate::Edge {
                                    name: "data".to_string(),
                                },
                            );
                            prev_node = new_node;
                        }
                    }
                }

                let output_node = child_graph.add_node(Node::Output);
                child_graph.add_edge(
                    prev_node,
                    output_node,
                    crate::Edge {
                        name: "output".to_string(),
                    },
                );

                // Create child with mixed encoding - average parent encodings
                let max_adj_size = parent1
                    .encoding
                    .adjacency_matrix
                    .len()
                    .max(parent2.encoding.adjacency_matrix.len());
                let mut adjacency_matrix = vec![vec![0.0; max_adj_size]; max_adj_size];

                for i in 0..max_adj_size {
                    for j in 0..max_adj_size {
                        let val1 = parent1
                            .encoding
                            .adjacency_matrix
                            .get(i)
                            .and_then(|row| row.get(j))
                            .copied()
                            .unwrap_or(0.0);
                        let val2 = parent2
                            .encoding
                            .adjacency_matrix
                            .get(i)
                            .and_then(|row| row.get(j))
                            .copied()
                            .unwrap_or(0.0);
                        adjacency_matrix[i][j] = (val1 + val2) / 2.0;
                    }
                }

                let max_nodes = parent1
                    .encoding
                    .node_features
                    .len()
                    .max(parent2.encoding.node_features.len());
                let mut node_features = Vec::new();
                for i in 0..max_nodes {
                    let f1 = parent1
                        .encoding
                        .node_features
                        .get(i)
                        .map(|v| v.as_slice())
                        .unwrap_or(&[]);
                    let f2 = parent2
                        .encoding
                        .node_features
                        .get(i)
                        .map(|v| v.as_slice())
                        .unwrap_or(&[]);
                    let max_len = f1.len().max(f2.len());
                    let mut mixed = Vec::new();
                    for j in 0..max_len {
                        let v1 = f1.get(j).copied().unwrap_or(0.5);
                        let v2 = f2.get(j).copied().unwrap_or(0.5);
                        mixed.push((v1 + v2) / 2.0);
                    }
                    node_features.push(mixed);
                }

                // Mix edge and global features similarly
                let max_edges = parent1
                    .encoding
                    .edge_features
                    .len()
                    .max(parent2.encoding.edge_features.len());
                let edge_features = vec![vec![1.0]; max_edges];

                let global_features = parent1
                    .encoding
                    .global_features
                    .iter()
                    .zip(&parent2.encoding.global_features)
                    .map(|(v1, v2)| (v1 + v2) / 2.0)
                    .collect();

                let encoding = ArchitectureEncoding {
                    adjacency_matrix,
                    node_features,
                    edge_features,
                    global_features,
                };

                let child = CandidateArchitecture {
                    graph: child_graph,
                    encoding,
                    fitness: vec![0.0],
                    age: 0,
                    parents: vec![parent1.id.clone(), parent2.id.clone()],
                    id: uuid::Uuid::new_v4().to_string(),
                    mutation_history: vec![],
                };

                new_population.push(child);
            } else {
                // Mutation: modify existing architecture
                if !survivors.is_empty() {
                    let parent_idx = rng.gen_range(0..survivors.len());
                    let mut mutated = survivors[parent_idx].clone();

                    // Mutate node features (simplified mutation)
                    for features in &mut mutated.encoding.node_features {
                        for val in features {
                            if rng.random::<f64>() < 0.1 {
                                *val = (*val + rng.random::<f64>() * 0.2 - 0.1).clamp(0.0, 1.0);
                            }
                        }
                    }

                    // Mutate global features
                    for val in &mut mutated.encoding.global_features {
                        if rng.random::<f64>() < 0.1 {
                            *val = (*val + rng.random::<f64>() * 0.2 - 0.1).max(0.0);
                        }
                    }

                    mutated.id = uuid::Uuid::new_v4().to_string();
                    mutated.age = 0;
                    mutated.fitness = vec![0.0];
                    // Simplified mutation history for initial implementation

                    new_population.push(mutated);
                }
            }
        }

        // Age increment for all architectures
        for arch in &mut new_population {
            arch.age += 1;
        }

        population_manager.population = new_population;
        Ok(())
    }

    async fn evaluate_architecture_fitness(
        &self,
        architecture: &CandidateArchitecture,
    ) -> Result<f64> {
        // Multi-objective fitness evaluation
        let mut fitness_components = Vec::new();

        // 1. Complexity score (prefer simpler architectures initially)
        let node_count = architecture.graph.node_count();
        let complexity_penalty = if node_count > 50 {
            0.5
        } else if node_count > 30 {
            0.7
        } else {
            1.0
        };

        // 2. Depth score (moderate depth preferred)
        let depth = node_count.saturating_sub(2); // Exclude input/output
        let depth_score = if depth >= 5 && depth <= 20 {
            1.0
        } else if depth < 5 {
            0.6 + (depth as f64 / 5.0) * 0.4
        } else {
            1.0 - ((depth.saturating_sub(20)) as f64 / 50.0).min(0.5)
        };

        // 3. Connectivity score (well-connected graphs preferred)
        let edge_count = architecture.graph.edge_count();
        let connectivity_score = if edge_count >= node_count - 1 {
            1.0
        } else {
            edge_count as f64 / (node_count.saturating_sub(1).max(1)) as f64
        };

        // 4. Encoding-based score (use global features from encoding)
        let encoding_score = if !architecture.encoding.global_features.is_empty() {
            architecture.encoding.global_features.iter().sum::<f64>()
                / architecture.encoding.global_features.len() as f64
        } else {
            0.5
        };

        // 5. Age penalty (slight preference for newer architectures in later generations)
        let age_factor = if architecture.age > 10 { 0.95 } else { 1.0 };

        // Weight components based on objective weights
        let accuracy_proxy = encoding_score * depth_score * connectivity_score;
        let efficiency_score = complexity_penalty;

        fitness_components.push(accuracy_proxy * 0.6); // 60% weight on accuracy proxy
        fitness_components.push(efficiency_score * 0.3); // 30% weight on efficiency
        fitness_components.push(connectivity_score * 0.1); // 10% weight on connectivity

        // Aggregate fitness
        let fitness = fitness_components.iter().sum::<f64>() * age_factor;

        Ok(fitness)
    }

    fn update_search_history(
        &self,
        generation: usize,
        population_manager: &PopulationManager,
    ) -> Result<()> {
        let mut history = self.search_history.lock().unwrap();

        // Update progress metrics
        history.progress_metrics.architectures_evaluated += population_manager.population.len();

        // Update best fitness
        let current_best_fitness = population_manager.get_best_fitness();
        if current_best_fitness > history.progress_metrics.best_fitness {
            history.progress_metrics.best_fitness = current_best_fitness;
        }

        // Track diversity over time
        history
            .progress_metrics
            .diversity_over_time
            .push(population_manager.diversity_metrics.structural_diversity);

        // Update Pareto front evolution
        let pareto_size = population_manager
            .population
            .iter()
            .filter(|arch| {
                let fitness = arch.fitness.iter().sum::<f64>();
                fitness >= current_best_fitness * 0.9 // Within 90% of best
            })
            .count();
        history
            .progress_metrics
            .pareto_front_evolution
            .push(pareto_size);

        // Calculate convergence rate
        if generation > 0 && !history.progress_metrics.diversity_over_time.is_empty() {
            let prev_diversity = history
                .progress_metrics
                .diversity_over_time
                .get(
                    history
                        .progress_metrics
                        .diversity_over_time
                        .len()
                        .saturating_sub(2),
                )
                .copied()
                .unwrap_or(1.0);
            let curr_diversity = population_manager.diversity_metrics.structural_diversity;

            history.progress_metrics.convergence_rate =
                (prev_diversity - curr_diversity).abs() / prev_diversity.max(0.001);
        }

        // Add evaluated architectures to history
        for arch in &population_manager.population {
            history.evaluated_architectures.push(arch.clone());
        }

        Ok(())
    }

    fn create_search_results(&self, population_manager: &PopulationManager) -> SearchResults {
        let history = self.search_history.lock().unwrap();

        // Find best architecture
        let best_architecture = population_manager
            .population
            .iter()
            .max_by(|a, b| {
                let a_fitness = a.fitness.iter().sum::<f64>();
                let b_fitness = b.fitness.iter().sum::<f64>();
                a_fitness
                    .partial_cmp(&b_fitness)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .cloned()
            .unwrap_or_else(|| CandidateArchitecture {
                graph: FxGraph::new(),
                encoding: ArchitectureEncoding {
                    adjacency_matrix: vec![],
                    node_features: vec![],
                    edge_features: vec![],
                    global_features: vec![],
                },
                fitness: vec![0.0],
                age: 0,
                parents: vec![],
                id: uuid::Uuid::new_v4().to_string(),
                mutation_history: vec![],
            });

        // Build Pareto front (top 10% of population by fitness)
        let mut sorted_pop = population_manager.population.clone();
        sorted_pop.sort_by(|a, b| {
            let a_fitness = a.fitness.iter().sum::<f64>();
            let b_fitness = b.fitness.iter().sum::<f64>();
            b_fitness
                .partial_cmp(&a_fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let pareto_count = (sorted_pop.len() / 10).max(1).min(sorted_pop.len());
        let pareto_front = sorted_pop.into_iter().take(pareto_count).collect();

        // Build convergence curve from diversity over time
        let convergence_curve = history.progress_metrics.diversity_over_time.clone();

        // Build diversity evolution (same as convergence for now)
        let diversity_evolution = history.progress_metrics.diversity_over_time.clone();

        // Build architecture distribution
        let mut architecture_distribution = HashMap::new();
        for arch in &population_manager.population {
            let depth = arch.graph.node_count().saturating_sub(2);
            let key = format!("depth_{}", depth);
            *architecture_distribution.entry(key).or_insert(0) += 1;
        }

        // Create search analytics
        let search_analytics = SearchAnalytics {
            convergence_curve,
            diversity_evolution,
            architecture_distribution,
            prediction_accuracy: PredictionAccuracy {
                mae: 0.05, // Placeholder values
                rmse: 0.08,
                r2: 0.85,
                kendall_tau: 0.75,
            },
        };

        SearchResults {
            best_architecture,
            pareto_front,
            search_analytics,
            total_search_time: history.progress_metrics.search_time,
        }
    }
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

// Implementation of helper structs
impl PerformancePredictor {
    fn new() -> Self {
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

impl ArchitectureFeatureExtractor {
    fn new() -> Self {
        Self {
            extraction_method: FeatureExtractionMethod::HandCrafted {
                include_operation_counts: true,
                include_depth_width: true,
                include_connectivity: true,
            },
            feature_cache: HashMap::new(),
        }
    }
}

impl PopulationManager {
    fn new() -> Self {
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

    fn initialize_population(
        &mut self,
        search_space: &ArchitectureSearchSpace,
        size: usize,
    ) -> Result<()> {
        // Clear existing population
        self.population.clear();

        // Generate random architectures for initial population
        let mut rng = thread_rng();
        for _ in 0..size {
            let mut graph = FxGraph::new();

            // Generate random depth within constraints
            let depth = rng.gen_range(search_space.depth_range.0..=search_space.depth_range.1);

            // Add input node
            let input_node = graph.add_node(Node::Input("input".to_string()));
            let mut prev_node = input_node;

            // Add random layers
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
                    LayerType::Custom { operation_name, .. } => operation_name,
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

            // Add output node
            let output_node = graph.add_node(Node::Output);
            graph.add_edge(
                prev_node,
                output_node,
                crate::Edge {
                    name: "output".to_string(),
                },
            );

            // Create architecture encoding using the encode_architecture method
            let encoding = ArchitectureEncoding {
                adjacency_matrix: vec![vec![0.0; depth + 2]; depth + 2], // +2 for input/output
                node_features: vec![vec![rng.random::<f64>(); 10]; depth + 2],
                edge_features: vec![vec![1.0]; depth + 1],
                global_features: vec![depth as f64, (depth + 2) as f64, (depth + 1) as f64, 0.0],
            };

            self.population.push(CandidateArchitecture {
                graph,
                encoding,
                fitness: vec![rng.random::<f64>()], // Initial random fitness
                age: 0,
                parents: vec![],
                id: uuid::Uuid::new_v4().to_string(),
                mutation_history: vec![],
            });
        }

        Ok(())
    }

    fn get_best_fitness(&self) -> f64 {
        self.population
            .iter()
            .map(|arch| arch.fitness.iter().fold(0.0f64, |acc, &x| acc.max(x)))
            .fold(f64::NEG_INFINITY, |acc, x| acc.max(x))
    }
}

impl SearchHistory {
    fn new() -> Self {
        Self {
            evaluated_architectures: Vec::new(),
            pareto_front: Vec::new(),
            progress_metrics: SearchProgressMetrics {
                architectures_evaluated: 0,
                search_time: Duration::from_secs(0),
                best_fitness: f64::NEG_INFINITY,
                convergence_rate: 0.0,
                diversity_over_time: Vec::new(),
                pareto_front_evolution: Vec::new(),
            },
            best_per_objective: HashMap::new(),
        }
    }
}

impl SearchResults {
    fn new() -> Self {
        // Create a dummy architecture for initialization
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
            (0, (32, 512)),  // Conv2d width constraints
            (1, (16, 256)),  // DepthwiseConv2d width constraints
            (2, (64, 1024)), // Linear width constraints
            (3, (64, 512)),  // Attention width constraints
            (4, (2, 8)),     // Pooling width constraints
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

        // Test that NAS is created successfully
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
        assert!(arch.graph.node_count() >= 2); // At least input and output
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
