//! Configuration types for Expert Parallelism
//!
//! This module defines configuration structures and enums used throughout
//! the expert parallelism system, including sharding strategies, parameters,
//! and optimization settings.

use serde::{Deserialize, Serialize};

/// Expert parallelism configuration
///
/// This structure contains all the configuration parameters needed to set up
/// and run a Mixture of Experts (MoE) model with distributed expert parallelism.
///
/// # Examples
///
/// ```rust
/// use torsh_distributed::expert_parallelism::config::{ExpertParallelismConfig, ExpertShardingStrategy};
///
/// let config = ExpertParallelismConfig {
///     num_experts: 16,
///     num_experts_per_token: 2,
///     capacity_factor: 1.5,
///     sharding_strategy: ExpertShardingStrategy::ModelParallel,
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertParallelismConfig {
    /// Number of experts in the MoE layer
    ///
    /// This determines the total number of expert networks available for routing.
    /// Typical values range from 8 to 1024 depending on model size and requirements.
    pub num_experts: usize,

    /// Number of experts to activate per token (top-k)
    ///
    /// Each token is routed to the top-k experts based on router scores.
    /// Common values are 1, 2, or 4. Higher values increase computational cost
    /// but may improve model quality.
    pub num_experts_per_token: usize,

    /// Expert capacity factor (capacity = tokens_per_expert * capacity_factor)
    ///
    /// This factor determines how many tokens each expert can process.
    /// Values > 1.0 provide buffer capacity to handle load imbalance.
    /// Typical range: 1.0 to 2.0.
    pub capacity_factor: f32,

    /// Load balancing loss coefficient
    ///
    /// Weight for the auxiliary loss that encourages balanced expert utilization.
    /// Higher values enforce stronger load balancing but may hurt model quality.
    /// Typical range: 0.001 to 0.1.
    pub load_balance_loss_coeff: f32,

    /// Router z-loss coefficient (for numerical stability)
    ///
    /// Weight for the z-loss that encourages router logits to stay close to zero,
    /// improving numerical stability. Typical range: 0.0001 to 0.01.
    pub router_z_loss_coeff: f32,

    /// Enable expert dropout during training
    ///
    /// Probability of randomly dropping experts during training to improve
    /// robustness and prevent overfitting. Range: 0.0 to 1.0.
    pub expert_dropout: f32,

    /// Enable load balancing across devices
    ///
    /// When true, the system actively monitors and rebalances expert utilization
    /// across different devices to optimize resource usage.
    pub enable_load_balancing: bool,

    /// Expert sharding strategy
    ///
    /// Determines how experts are distributed across devices and processes.
    pub sharding_strategy: ExpertShardingStrategy,

    /// Maximum batch size for expert processing
    ///
    /// Limits the number of tokens that can be processed by a single expert
    /// in one forward pass. Helps control memory usage.
    pub max_expert_batch_size: Option<usize>,

    /// Enable gradient accumulation across experts
    ///
    /// When true, gradients are accumulated across multiple expert invocations
    /// before updating parameters, which can improve training stability.
    pub enable_gradient_accumulation: bool,

    /// Number of gradient accumulation steps
    ///
    /// Only relevant when gradient accumulation is enabled.
    pub gradient_accumulation_steps: usize,

    /// Expert initialization strategy
    ///
    /// Method used to initialize expert parameters.
    pub initialization_strategy: ExpertInitStrategy,

    /// Enable expert synchronization
    ///
    /// When true, experts synchronize their parameters periodically during training.
    pub enable_expert_sync: bool,

    /// Synchronization frequency (in steps)
    ///
    /// How often to synchronize expert parameters when synchronization is enabled.
    pub sync_frequency: usize,

    /// Gate network configuration
    ///
    /// Optional configuration for hierarchical or advanced gate networks.
    pub gate_network: Option<GateNetworkConfig>,

    /// Load balancing configuration
    ///
    /// Configuration for expert load balancing and migration.
    pub load_balancing: Option<LoadBalancingConfig>,

    /// Migration configuration
    ///
    /// Configuration for expert migration strategies and triggers.
    pub migration: Option<ExpertMigrationConfig>,

    /// Enable expert migration (simplified flag)
    pub enable_expert_migration: bool,

    /// Migration threshold for triggering migrations
    pub migration_threshold: f32,

    /// Memory allocated per expert (in MB)
    pub memory_per_expert_mb: usize,

    /// Enable communication overlap
    pub communication_overlap: bool,

    /// Enable gradient compression
    pub gradient_compression: bool,
}

impl Default for ExpertParallelismConfig {
    fn default() -> Self {
        Self {
            num_experts: 8,
            num_experts_per_token: 2,
            capacity_factor: 1.25,
            load_balance_loss_coeff: 0.01,
            router_z_loss_coeff: 0.001,
            expert_dropout: 0.0,
            enable_load_balancing: true,
            sharding_strategy: ExpertShardingStrategy::ModelParallel,
            max_expert_batch_size: None,
            enable_gradient_accumulation: false,
            gradient_accumulation_steps: 1,
            initialization_strategy: ExpertInitStrategy::Xavier,
            enable_expert_sync: false,
            sync_frequency: 100,
            gate_network: None,
            load_balancing: None,
            migration: None,
            enable_expert_migration: false,
            migration_threshold: 0.3,
            memory_per_expert_mb: 512,
            communication_overlap: true,
            gradient_compression: false,
        }
    }
}

impl ExpertParallelismConfig {
    /// Create a new configuration with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a configuration optimized for small-scale deployment
    ///
    /// # Returns
    ///
    /// A configuration suitable for models with 8-16 experts
    pub fn small_scale() -> Self {
        Self {
            num_experts: 8,
            num_experts_per_token: 2,
            capacity_factor: 1.25,
            load_balance_loss_coeff: 0.01,
            sharding_strategy: ExpertShardingStrategy::DataParallel,
            ..Default::default()
        }
    }

    /// Create a configuration optimized for large-scale deployment
    ///
    /// # Returns
    ///
    /// A configuration suitable for models with 64+ experts
    pub fn large_scale() -> Self {
        Self {
            num_experts: 128,
            num_experts_per_token: 2,
            capacity_factor: 1.5,
            load_balance_loss_coeff: 0.001,
            sharding_strategy: ExpertShardingStrategy::ModelParallel,
            enable_gradient_accumulation: true,
            gradient_accumulation_steps: 4,
            enable_expert_sync: true,
            sync_frequency: 50,
            ..Default::default()
        }
    }

    /// Create a configuration optimized for inference
    ///
    /// # Returns
    ///
    /// A configuration with settings optimized for inference workloads
    pub fn inference() -> Self {
        Self {
            expert_dropout: 0.0,
            enable_load_balancing: false,
            enable_gradient_accumulation: false,
            enable_expert_sync: false,
            ..Default::default()
        }
    }

    /// Validate the configuration parameters
    ///
    /// # Returns
    ///
    /// Result indicating whether the configuration is valid
    pub fn validate(&self) -> Result<(), String> {
        if self.num_experts == 0 {
            return Err("Number of experts must be greater than 0".to_string());
        }

        if self.num_experts_per_token == 0 || self.num_experts_per_token > self.num_experts {
            return Err(
                "Number of experts per token must be between 1 and num_experts".to_string(),
            );
        }

        if self.capacity_factor <= 0.0 {
            return Err("Capacity factor must be positive".to_string());
        }

        if self.load_balance_loss_coeff < 0.0 {
            return Err("Load balance loss coefficient must be non-negative".to_string());
        }

        if self.router_z_loss_coeff < 0.0 {
            return Err("Router z-loss coefficient must be non-negative".to_string());
        }

        if self.expert_dropout < 0.0 || self.expert_dropout > 1.0 {
            return Err("Expert dropout must be between 0.0 and 1.0".to_string());
        }

        if self.gradient_accumulation_steps == 0 {
            return Err("Gradient accumulation steps must be greater than 0".to_string());
        }

        if self.sync_frequency == 0 {
            return Err("Sync frequency must be greater than 0".to_string());
        }

        Ok(())
    }

    /// Calculate the effective expert capacity
    ///
    /// # Arguments
    ///
    /// * `total_tokens` - Total number of tokens in the batch
    ///
    /// # Returns
    ///
    /// The effective capacity per expert
    pub fn calculate_expert_capacity(&self, total_tokens: usize) -> usize {
        let tokens_per_expert = (total_tokens * self.num_experts_per_token) / self.num_experts;
        (tokens_per_expert as f32 * self.capacity_factor).ceil() as usize
    }

    /// Get the recommended number of devices for this configuration
    ///
    /// # Returns
    ///
    /// Recommended number of devices based on the sharding strategy
    pub fn recommended_num_devices(&self) -> usize {
        match self.sharding_strategy {
            ExpertShardingStrategy::DataParallel => 1,
            ExpertShardingStrategy::ModelParallel => self.num_experts.min(64),
            ExpertShardingStrategy::Hybrid => (self.num_experts / 4).clamp(2, 16),
            ExpertShardingStrategy::Dynamic => (self.num_experts / 2).clamp(4, 32),
        }
    }
}

/// Expert sharding strategies
///
/// Defines how experts are distributed across devices and processes
/// in a distributed training setup.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExpertShardingStrategy {
    /// Each device holds all experts (data parallel)
    ///
    /// All experts are replicated on each device. This strategy is suitable
    /// for smaller models or when communication costs are high.
    DataParallel,

    /// Each device holds a subset of experts (model parallel)
    ///
    /// Experts are partitioned across devices. This strategy is suitable
    /// for large models where memory constraints require expert sharding.
    ModelParallel,

    /// Hybrid: some experts replicated, others sharded
    ///
    /// Combines data and model parallelism. Frequently used experts may be
    /// replicated while less common experts are sharded.
    Hybrid,

    /// Dynamic: expert placement adapts to load
    ///
    /// Expert placement is dynamically adjusted based on runtime load patterns.
    /// This strategy requires more sophisticated load monitoring and migration.
    Dynamic,
}

impl ExpertShardingStrategy {
    /// Get a description of the sharding strategy
    ///
    /// # Returns
    ///
    /// A string describing the strategy
    pub fn description(&self) -> &'static str {
        match self {
            Self::DataParallel => "All experts replicated on each device",
            Self::ModelParallel => "Experts partitioned across devices",
            Self::Hybrid => "Mix of replicated and partitioned experts",
            Self::Dynamic => "Dynamic expert placement based on load",
        }
    }

    /// Check if this strategy requires load balancing
    ///
    /// # Returns
    ///
    /// True if the strategy benefits from active load balancing
    pub fn requires_load_balancing(&self) -> bool {
        matches!(self, Self::ModelParallel | Self::Hybrid | Self::Dynamic)
    }

    /// Check if this strategy supports dynamic migration
    ///
    /// # Returns
    ///
    /// True if experts can be migrated between devices
    pub fn supports_migration(&self) -> bool {
        matches!(self, Self::Hybrid | Self::Dynamic)
    }
}

/// Expert parameter configuration
///
/// Defines the architecture parameters for individual expert networks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertParameters {
    /// Input dimension of the expert network
    pub input_dim: usize,

    /// Hidden dimension of the expert network
    ///
    /// Typically 4x the input dimension for transformer-style experts.
    pub hidden_dim: usize,

    /// Output dimension of the expert network
    ///
    /// Usually matches the input dimension for residual connections.
    pub output_dim: usize,

    /// Activation function name
    ///
    /// Common choices: "relu", "gelu", "swish", "tanh"
    pub activation: String,

    /// Number of hidden layers in the expert
    pub num_layers: usize,

    /// Dropout probability for expert layers
    pub dropout: f32,

    /// Whether to use bias in linear layers
    pub use_bias: bool,

    /// Layer normalization configuration
    pub layer_norm_eps: f32,

    /// Weight initialization scale
    pub init_scale: f32,
}

impl Default for ExpertParameters {
    fn default() -> Self {
        Self {
            input_dim: 512,
            hidden_dim: 2048,
            output_dim: 512,
            activation: "relu".to_string(),
            num_layers: 2,
            dropout: 0.1,
            use_bias: true,
            layer_norm_eps: 1e-5,
            init_scale: 0.02,
        }
    }
}

impl ExpertParameters {
    /// Create a new expert parameter configuration
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        Self {
            input_dim,
            hidden_dim,
            output_dim,
            ..Default::default()
        }
    }

    /// Create parameters for a transformer-style expert
    ///
    /// # Arguments
    ///
    /// * `model_dim` - The model dimension (input/output dimension)
    ///
    /// # Returns
    ///
    /// Parameters configured for transformer-style FFN experts
    pub fn transformer_ffn(model_dim: usize) -> Self {
        Self {
            input_dim: model_dim,
            hidden_dim: model_dim * 4,
            output_dim: model_dim,
            activation: "gelu".to_string(),
            ..Default::default()
        }
    }

    /// Create parameters for a lightweight expert
    ///
    /// # Arguments
    ///
    /// * `model_dim` - The model dimension
    ///
    /// # Returns
    ///
    /// Parameters configured for lightweight experts with reduced capacity
    pub fn lightweight(model_dim: usize) -> Self {
        Self {
            input_dim: model_dim,
            hidden_dim: model_dim * 2,
            output_dim: model_dim,
            num_layers: 1,
            dropout: 0.05,
            ..Default::default()
        }
    }

    /// Validate the parameter configuration
    ///
    /// # Returns
    ///
    /// Result indicating whether the parameters are valid
    pub fn validate(&self) -> Result<(), String> {
        if self.input_dim == 0 {
            return Err("Input dimension must be greater than 0".to_string());
        }

        if self.hidden_dim == 0 {
            return Err("Hidden dimension must be greater than 0".to_string());
        }

        if self.output_dim == 0 {
            return Err("Output dimension must be greater than 0".to_string());
        }

        if self.num_layers == 0 {
            return Err("Number of layers must be greater than 0".to_string());
        }

        if self.dropout < 0.0 || self.dropout > 1.0 {
            return Err("Dropout must be between 0.0 and 1.0".to_string());
        }

        if self.layer_norm_eps <= 0.0 {
            return Err("Layer norm epsilon must be positive".to_string());
        }

        if self.init_scale <= 0.0 {
            return Err("Initialization scale must be positive".to_string());
        }

        let valid_activations = ["relu", "gelu", "swish", "tanh", "leaky_relu", "elu"];
        if !valid_activations.contains(&self.activation.as_str()) {
            return Err(format!(
                "Unsupported activation function: {}. Supported: {:?}",
                self.activation, valid_activations
            ));
        }

        Ok(())
    }

    /// Calculate the total number of parameters for this expert configuration
    ///
    /// # Returns
    ///
    /// Estimated number of parameters
    pub fn parameter_count(&self) -> usize {
        if self.num_layers == 1 {
            // Single layer: input -> hidden -> output
            let layer1_params =
                self.input_dim * self.hidden_dim + if self.use_bias { self.hidden_dim } else { 0 };
            let layer2_params =
                self.hidden_dim * self.output_dim + if self.use_bias { self.output_dim } else { 0 };
            layer1_params + layer2_params
        } else {
            // Multiple layers
            let input_layer =
                self.input_dim * self.hidden_dim + if self.use_bias { self.hidden_dim } else { 0 };
            let hidden_layers = (self.num_layers - 2)
                * (self.hidden_dim * self.hidden_dim
                    + if self.use_bias { self.hidden_dim } else { 0 });
            let output_layer =
                self.hidden_dim * self.output_dim + if self.use_bias { self.output_dim } else { 0 };
            input_layer + hidden_layers + output_layer
        }
    }
}

/// Expert initialization strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExpertInitStrategy {
    /// Xavier/Glorot initialization
    Xavier,
    /// Kaiming/He initialization
    Kaiming,
    /// Normal distribution with specified std
    Normal,
    /// Uniform distribution
    Uniform,
    /// Truncated normal distribution
    TruncatedNormal,
}

impl ExpertInitStrategy {
    /// Get a description of the initialization strategy
    pub fn description(&self) -> &'static str {
        match self {
            Self::Xavier => "Xavier/Glorot initialization for balanced gradients",
            Self::Kaiming => "Kaiming/He initialization for ReLU networks",
            Self::Normal => "Standard normal distribution",
            Self::Uniform => "Uniform distribution",
            Self::TruncatedNormal => "Truncated normal distribution",
        }
    }
}

/// Gate network configuration for hierarchical expert routing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateNetworkConfig {
    /// Hierarchical gate configuration
    pub hierarchical: Option<HierarchicalGateConfig>,

    /// Enable learned gate networks
    pub enable_learned_gates: bool,

    /// Gate network dropout
    pub gate_dropout: f32,

    /// Number of gate layers
    pub num_gate_layers: usize,
}

impl Default for GateNetworkConfig {
    fn default() -> Self {
        Self {
            hierarchical: None,
            enable_learned_gates: true,
            gate_dropout: 0.1,
            num_gate_layers: 2,
        }
    }
}

/// Hierarchical gate network configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalGateConfig {
    /// Number of hierarchical levels
    pub levels: usize,

    /// Number of experts per group at each level
    pub experts_per_group: usize,

    /// Hidden dimension for gate networks
    pub gate_hidden_dim: usize,

    /// Enable learned expert grouping
    pub use_learned_grouping: bool,

    /// Group assignment strategy
    pub grouping_strategy: GroupingStrategy,
}

impl Default for HierarchicalGateConfig {
    fn default() -> Self {
        Self {
            levels: 2,
            experts_per_group: 8,
            gate_hidden_dim: 512,
            use_learned_grouping: true,
            grouping_strategy: GroupingStrategy::LoadBased,
        }
    }
}

/// Expert grouping strategies for hierarchical gates
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GroupingStrategy {
    /// Group experts based on current load
    LoadBased,
    /// Group experts based on similarity
    SimilarityBased,
    /// Use static expert grouping
    Static,
    /// Dynamic grouping based on routing patterns
    Dynamic,
}

/// Load balancing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadBalancingConfig {
    /// Enable automatic load balancing
    pub enable_auto_balancing: bool,

    /// Load imbalance threshold for triggering rebalancing
    pub imbalance_threshold: f32,

    /// Frequency of load balancing checks (in steps)
    pub check_frequency: usize,

    /// Maximum number of concurrent migrations
    pub max_concurrent_migrations: usize,

    /// Load smoothing factor for load history
    pub load_smoothing_factor: f32,
}

impl Default for LoadBalancingConfig {
    fn default() -> Self {
        Self {
            enable_auto_balancing: true,
            imbalance_threshold: 0.3,
            check_frequency: 50,
            max_concurrent_migrations: 2,
            load_smoothing_factor: 0.9,
        }
    }
}

/// Expert migration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertMigrationConfig {
    /// Enable expert migration
    pub enable_migration: bool,

    /// Migration trigger conditions
    pub triggers: Vec<MigrationTrigger>,

    /// Migration strategy preferences
    pub preferred_strategies: Vec<MigrationStrategy>,

    /// Migration cooldown period (in steps)
    pub cooldown_period: usize,

    /// Maximum migration distance (number of devices)
    pub max_migration_distance: usize,
}

impl Default for ExpertMigrationConfig {
    fn default() -> Self {
        Self {
            enable_migration: false,
            triggers: vec![MigrationTrigger::LoadImbalance],
            preferred_strategies: vec![MigrationStrategy::GradualMigration],
            cooldown_period: 100,
            max_migration_distance: 1,
        }
    }
}

/// Migration trigger conditions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MigrationTrigger {
    /// Trigger on load imbalance
    LoadImbalance,
    /// Trigger on memory pressure
    MemoryPressure,
    /// Trigger on performance degradation
    PerformanceDegradation,
    /// Trigger at regular intervals
    Periodic,
}

/// Migration strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MigrationStrategy {
    /// Gradual parameter migration
    GradualMigration,
    /// Complete expert migration
    CompleteMigration,
    /// Load redistribution without migration
    LoadRedistribution,
    /// Hybrid approach
    Hybrid,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expert_parallelism_config_default() {
        let config = ExpertParallelismConfig::default();
        assert_eq!(config.num_experts, 8);
        assert_eq!(config.num_experts_per_token, 2);
        assert_eq!(config.capacity_factor, 1.25);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_expert_parallelism_config_validation() {
        let mut config = ExpertParallelismConfig::default();

        // Test invalid num_experts
        config.num_experts = 0;
        assert!(config.validate().is_err());

        // Test invalid num_experts_per_token
        config.num_experts = 8;
        config.num_experts_per_token = 10;
        assert!(config.validate().is_err());

        // Test invalid capacity_factor
        config.num_experts_per_token = 2;
        config.capacity_factor = -1.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_expert_capacity_calculation() {
        let config = ExpertParallelismConfig::default();
        let capacity = config.calculate_expert_capacity(1000);

        // With 8 experts, 2 experts per token, 1000 tokens total
        // tokens_per_expert = (1000 * 2) / 8 = 250
        // capacity = 250 * 1.25 = 312.5 -> 313
        assert_eq!(capacity, 313);
    }

    #[test]
    fn test_sharding_strategy_properties() {
        assert!(ExpertShardingStrategy::ModelParallel.requires_load_balancing());
        assert!(!ExpertShardingStrategy::DataParallel.requires_load_balancing());
        assert!(ExpertShardingStrategy::Dynamic.supports_migration());
        assert!(!ExpertShardingStrategy::DataParallel.supports_migration());
    }

    #[test]
    fn test_expert_parameters_default() {
        let params = ExpertParameters::default();
        assert_eq!(params.input_dim, 512);
        assert_eq!(params.hidden_dim, 2048);
        assert_eq!(params.output_dim, 512);
        assert!(params.validate().is_ok());
    }

    #[test]
    fn test_expert_parameters_transformer_ffn() {
        let params = ExpertParameters::transformer_ffn(768);
        assert_eq!(params.input_dim, 768);
        assert_eq!(params.hidden_dim, 768 * 4);
        assert_eq!(params.output_dim, 768);
        assert_eq!(params.activation, "gelu");
    }

    #[test]
    fn test_expert_parameters_validation() {
        let mut params = ExpertParameters::default();

        // Test invalid dimensions
        params.input_dim = 0;
        assert!(params.validate().is_err());

        // Test invalid dropout
        params.input_dim = 512;
        params.dropout = 1.5;
        assert!(params.validate().is_err());

        // Test invalid activation
        params.dropout = 0.1;
        params.activation = "invalid".to_string();
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_expert_parameters_parameter_count() {
        let params = ExpertParameters::new(100, 200, 100);

        // Single layer case (num_layers = 2 by default)
        // Layer 1: 100 * 200 + 200 = 20200
        // Layer 2: 200 * 100 + 100 = 20100
        // Total: 40300
        let count = params.parameter_count();
        assert_eq!(count, 40300);
    }

    #[test]
    fn test_preset_configs() {
        let small = ExpertParallelismConfig::small_scale();
        assert_eq!(small.num_experts, 8);
        assert_eq!(
            small.sharding_strategy,
            ExpertShardingStrategy::DataParallel
        );

        let large = ExpertParallelismConfig::large_scale();
        assert_eq!(large.num_experts, 128);
        assert!(large.enable_gradient_accumulation);

        let inference = ExpertParallelismConfig::inference();
        assert_eq!(inference.expert_dropout, 0.0);
        assert!(!inference.enable_load_balancing);
    }

    #[test]
    fn test_recommended_num_devices() {
        let config = ExpertParallelismConfig {
            num_experts: 32,
            sharding_strategy: ExpertShardingStrategy::ModelParallel,
            ..Default::default()
        };

        let num_devices = config.recommended_num_devices();
        assert_eq!(num_devices, 32);
    }
}
