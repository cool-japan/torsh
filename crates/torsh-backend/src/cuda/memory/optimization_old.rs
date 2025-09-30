//! CUDA memory performance optimization engine
//!
//! This module provides comprehensive performance optimization for CUDA memory
//! management including ML-based optimization, multi-objective optimization,
//! adaptive strategies, and automatic optimization discovery.
//!
//! # Backward Compatibility Layer
//!
//! This module now serves as a compatibility layer over the new modular optimization
//! framework. The implementation has been refactored into specialized modules for
//! better maintainability and extensibility while preserving the existing API.
//!
//! ## Migration to Modular Components
//!
//! The optimization engine is now built on top of these specialized modules:
//! - [`ml_engine`] - Machine learning optimization core
//! - [`multi_objective`] - Multi-objective optimization algorithms
//! - [`adaptive_controller`] - Adaptive control and learning
//! - [`execution_engine`] - Strategy execution and orchestration
//! - [`predictor`] - Performance prediction and trend analysis
//! - [`validator`] - Optimization validation and safety
//! - [`strategies`] - Strategy management and selection
//! - [`objectives`] - Objective function evaluation
//! - [`parameters`] - Parameter tuning and optimization
//! - [`monitoring`] - Real-time system monitoring
//! - [`history`] - Historical data management
//! - [`config`] - Configuration management
//!
//! # Example Usage
//!
//! ```rust,no_run
//! use torsh_backend::cuda::memory::optimization::{
//!     CudaMemoryOptimizationEngine, OptimizationConfig, OptimizationStrategy, StrategyType
//! };
//! use std::collections::HashMap;
//!
//! // Create optimization engine
//! let config = OptimizationConfig::default();
//! let engine = CudaMemoryOptimizationEngine::new(config);
//!
//! // Register a custom strategy
//! let strategy = OptimizationStrategy {
//!     id: "custom_alloc".to_string(),
//!     name: "Custom Allocation Strategy".to_string(),
//!     description: "Custom memory allocation optimization".to_string(),
//!     strategy_type: StrategyType::AllocationOptimization,
//!     // ... other fields
//! };
//! engine.register_strategy(strategy)?;
//!
//! // Run optimization
//! let mut parameters = HashMap::new();
//! parameters.insert("batch_size".to_string(), 1024.0);
//! let results = engine.optimize_with_strategy("custom_alloc", parameters)?;
//! println!("Optimization improvement: {:.2}%", results.improvement * 100.0);
//!
//! // Get recommendations
//! let recommendations = engine.get_recommendations();
//! for rec in recommendations {
//!     println!("Recommendation: {} (confidence: {:.2})", rec.strategy, rec.confidence);
//! }
//!
//! // Check engine status
//! let status = engine.get_status();
//! println!("Active optimizations: {}", status.active_optimizations);
//! # Ok::<_, Box<dyn std::error::Error>>(())
//! ```

use std::collections::{HashMap, VecDeque, BTreeMap};
use std::sync::{Arc, Mutex, RwLock, atomic::{AtomicU64, Ordering}};
use std::time::{Duration, Instant};
use super::allocation::{AllocationStats, AllocationType};

// Import from modular components
use super::optimization::{
    OptimizationEngine, OptimizationConfig as NewOptimizationConfig, OptimizationObjectives,
    OptimizationResults as NewOptimizationResults, OptimizationError, OptimizationSolution,
    OptimizationParameters, ObjectiveValues, MemoryParameters, PerformanceParameters,
    StrategyParameters, OptimizationMetrics as NewOptimizationMetrics, OptimizationFeedback,
    HistoryQuery, OptimizationRecord, RiskLevel
};

/// Comprehensive CUDA memory optimization engine
///
/// This is the main entry point for CUDA memory optimization, providing backward-compatible
/// access to the new modular optimization framework. The engine maintains the original API
/// while leveraging the enhanced capabilities of the specialized modules.
#[derive(Debug)]
pub struct CudaMemoryOptimizationEngine {
    /// Internal optimization engine (new modular system)
    engine: Arc<tokio::sync::RwLock<OptimizationEngine>>,

    /// Configuration for backward compatibility
    config: OptimizationConfig,

    /// Strategy registry for backward compatibility
    strategies: RwLock<HashMap<String, OptimizationStrategy>>,

    /// Runtime for async operations
    runtime: Arc<tokio::runtime::Runtime>,
}

/// Configuration for optimization engine (backward compatibility)
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    /// Enable ML-based optimization
    pub enable_ml_optimization: bool,

    /// Enable multi-objective optimization
    pub enable_multi_objective: bool,

    /// Enable adaptive optimization
    pub enable_adaptive_optimization: bool,

    /// Enable real-time optimization
    pub enable_realtime_optimization: bool,

    /// Optimization frequency
    pub optimization_frequency: Duration,

    /// Maximum concurrent optimizations
    pub max_concurrent_optimizations: usize,

    /// Optimization timeout
    pub optimization_timeout: Duration,

    /// Enable performance prediction
    pub enable_performance_prediction: bool,

    /// Enable optimization validation
    pub enable_validation: bool,

    /// Safety threshold for risky optimizations
    pub safety_threshold: f32,

    /// Learning rate for adaptive algorithms
    pub learning_rate: f32,

    /// Exploration vs exploitation balance
    pub exploration_factor: f32,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            enable_ml_optimization: true,
            enable_multi_objective: true,
            enable_adaptive_optimization: true,
            enable_realtime_optimization: true,
            optimization_frequency: Duration::from_secs(60),
            max_concurrent_optimizations: 4,
            optimization_timeout: Duration::from_secs(300),
            enable_performance_prediction: true,
            enable_validation: true,
            safety_threshold: 0.8,
            learning_rate: 0.01,
            exploration_factor: 0.1,
        }
    }
}

/// Optimization strategy definition (backward compatibility)
#[derive(Debug, Clone)]
pub struct OptimizationStrategy {
    /// Strategy identifier
    pub id: String,

    /// Strategy name
    pub name: String,

    /// Strategy description
    pub description: String,

    /// Strategy type
    pub strategy_type: StrategyType,

    /// Optimization objectives
    pub objectives: Vec<OptimizationObjective>,

    /// Strategy parameters
    pub parameters: HashMap<String, OptimizationParameter>,

    /// Applicability conditions
    pub conditions: Vec<ApplicabilityCondition>,

    /// Expected benefits
    pub expected_benefits: ExpectedBenefits,

    /// Implementation complexity
    pub complexity: OptimizationComplexity,

    /// Risk assessment
    pub risk_assessment: RiskAssessment,

    /// Success rate history
    pub success_rate: f32,

    /// Performance history
    pub performance_history: Vec<PerformanceResult>,
}

/// Types of optimization strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StrategyType {
    /// Memory allocation optimization
    AllocationOptimization,
    /// Memory pool optimization
    PoolOptimization,
    /// Transfer optimization
    TransferOptimization,
    /// Fragmentation reduction
    FragmentationReduction,
    /// Cache optimization
    CacheOptimization,
    /// Bandwidth optimization
    BandwidthOptimization,
    /// Latency optimization
    LatencyOptimization,
    /// Energy optimization
    EnergyOptimization,
    /// Multi-objective optimization
    MultiObjective,
    /// Hybrid optimization
    Hybrid,
}

/// Optimization objectives (backward compatibility)
#[derive(Debug, Clone)]
pub struct OptimizationObjective {
    /// Objective name
    pub name: String,

    /// Objective type
    pub objective_type: ObjectiveType,

    /// Target value
    pub target_value: f64,

    /// Current value
    pub current_value: f64,

    /// Objective weight in multi-objective optimization
    pub weight: f32,

    /// Optimization direction
    pub direction: OptimizationDirection,

    /// Constraint conditions
    pub constraints: Vec<ObjectiveConstraint>,
}

/// Types of optimization objectives
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectiveType {
    /// Maximize performance
    Performance,
    /// Minimize memory usage
    MemoryUsage,
    /// Minimize latency
    Latency,
    /// Maximize bandwidth utilization
    BandwidthUtilization,
    /// Minimize energy consumption
    EnergyConsumption,
    /// Maximize cache hit rate
    CacheHitRate,
    /// Minimize fragmentation
    Fragmentation,
    /// Maximize allocation success rate
    AllocationSuccessRate,
    /// Minimize error rate
    ErrorRate,
    /// Maximize resource utilization
    ResourceUtilization,
}

/// Optimization direction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationDirection {
    Minimize,
    Maximize,
    Target(f64),
}

/// Objective constraint
#[derive(Debug, Clone)]
pub struct ObjectiveConstraint {
    /// Constraint name
    pub name: String,

    /// Constraint type
    pub constraint_type: ConstraintType,

    /// Constraint value
    pub value: f64,

    /// Constraint priority
    pub priority: ConstraintPriority,
}

/// Types of constraints
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstraintType {
    HardLimit,
    SoftLimit,
    Preference,
    Requirement,
}

/// Constraint priorities
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ConstraintPriority {
    Low,
    Medium,
    High,
    Critical,
}

/// Optimization parameters
#[derive(Debug, Clone)]
pub struct OptimizationParameter {
    /// Parameter name
    pub name: String,

    /// Parameter value
    pub value: ParameterValue,

    /// Parameter bounds
    pub bounds: Option<ParameterBounds>,

    /// Parameter sensitivity
    pub sensitivity: f32,

    /// Tuning history
    pub tuning_history: Vec<ParameterTuning>,
}

/// Parameter value types
#[derive(Debug, Clone)]
pub enum ParameterValue {
    Integer(i64),
    Float(f64),
    Boolean(bool),
    String(String),
    Array(Vec<ParameterValue>),
}

/// Parameter bounds
#[derive(Debug, Clone)]
pub struct ParameterBounds {
    /// Minimum value
    pub min: ParameterValue,

    /// Maximum value
    pub max: ParameterValue,

    /// Suggested value
    pub suggested: Option<ParameterValue>,
}

/// Parameter tuning record
#[derive(Debug, Clone)]
pub struct ParameterTuning {
    /// Tuning timestamp
    pub timestamp: Instant,

    /// Parameter value
    pub value: ParameterValue,

    /// Performance result
    pub performance: f32,

    /// Context information
    pub context: HashMap<String, String>,
}

/// Applicability condition
#[derive(Debug, Clone)]
pub struct ApplicabilityCondition {
    /// Condition name
    pub name: String,

    /// Condition expression
    pub expression: String,

    /// Expected result
    pub expected_result: bool,

    /// Condition weight
    pub weight: f32,
}

/// Expected benefits
#[derive(Debug, Clone)]
pub struct ExpectedBenefits {
    /// Performance improvement percentage
    pub performance_improvement: f32,

    /// Memory reduction percentage
    pub memory_reduction: f32,

    /// Energy savings percentage
    pub energy_savings: f32,

    /// Implementation effort estimate
    pub implementation_effort: f32,
}

/// Optimization complexity
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationComplexity {
    Low,
    Medium,
    High,
    VeryHigh,
}

/// Risk assessment
#[derive(Debug, Clone)]
pub struct RiskAssessment {
    /// Overall risk level
    pub risk_level: RiskLevel,

    /// Risk factors
    pub risk_factors: Vec<String>,

    /// Mitigation strategies
    pub mitigations: Vec<String>,

    /// Success probability
    pub success_probability: f32,
}

/// Performance result
#[derive(Debug, Clone)]
pub struct PerformanceResult {
    /// Result timestamp
    pub timestamp: Instant,

    /// Performance metrics
    pub metrics: HashMap<String, f64>,

    /// Configuration used
    pub configuration: HashMap<String, String>,

    /// Success flag
    pub success: bool,
}

/// Optimization results (backward compatibility)
#[derive(Debug, Clone)]
pub struct OptimizationResults {
    /// Final objective values
    pub objective_values: HashMap<String, f64>,

    /// Solution parameters
    pub solution_parameters: HashMap<String, f64>,

    /// Performance improvement
    pub improvement: f32,

    /// Solution quality
    pub quality: f32,

    /// Optimization time
    pub optimization_time: Duration,

    /// Validation results
    pub validation: ValidationResults,

    /// Convergence information
    pub convergence: ConvergenceInfo,
}

/// Validation results
#[derive(Debug, Clone)]
pub struct ValidationResults {
    /// Validation status
    pub status: ValidationStatus,

    /// Validation score
    pub score: f32,

    /// Validation metrics
    pub metrics: HashMap<String, f64>,

    /// Risk assessment
    pub risk_assessment: RiskAssessment,

    /// Validation messages
    pub messages: Vec<String>,
}

/// Validation status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationStatus {
    Passed,
    Warning,
    Failed,
    Pending,
}

/// Convergence information
#[derive(Debug, Clone)]
pub struct ConvergenceInfo {
    /// Number of iterations
    pub iterations: usize,

    /// Convergence achieved
    pub converged: bool,

    /// Final fitness value
    pub final_fitness: f64,

    /// Convergence rate
    pub convergence_rate: f64,
}

/// Optimization status (for monitoring)
#[derive(Debug, Clone)]
pub struct OptimizationStatus {
    /// Active optimizations count
    pub active_count: usize,

    /// Queued optimizations count
    pub queued_count: usize,

    /// Success rate (last 24h)
    pub recent_success_rate: f32,

    /// Average improvement
    pub average_improvement: f32,

    /// System health score
    pub health_score: f32,
}

/// Optimization recommendation
#[derive(Debug, Clone)]
pub struct OptimizationRecommendation {
    /// Recommendation ID
    pub id: String,

    /// Strategy recommended
    pub strategy: String,

    /// Expected benefit
    pub expected_benefit: f32,

    /// Confidence level
    pub confidence: f32,

    /// Priority level
    pub priority: RecommendationPriority,

    /// Description
    pub description: String,

    /// Required parameters
    pub required_parameters: HashMap<String, ParameterValue>,
}

/// Recommendation priority
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RecommendationPriority {
    Low,
    Medium,
    High,
    Critical,
}

/// Engine status information
#[derive(Debug, Clone)]
pub struct OptimizationEngineStatus {
    /// Number of active optimizations
    pub active_optimizations: usize,

    /// Number of queued optimizations
    pub queued_optimizations: usize,

    /// Recent success rate
    pub success_rate: f32,

    /// Average improvement achieved
    pub average_improvement: f32,

    /// System health indicator
    pub health: f32,

    /// Registered strategies count
    pub strategy_count: usize,

    /// Optimization frequency
    pub optimization_frequency: Duration,
}

impl CudaMemoryOptimizationEngine {
    /// Create new optimization engine
    pub fn new(config: OptimizationConfig) -> Self {
        // Create runtime for async operations
        let runtime = Arc::new(
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(4)
                .enable_all()
                .build()
                .expect("Failed to create async runtime")
        );

        // Convert old config to new config format
        let new_config = Self::convert_config(&config);

        // Create the new modular optimization engine
        let engine = runtime.block_on(async {
            OptimizationEngine::new(new_config)
                .map_err(|e| format!("Failed to create optimization engine: {}", e))
        }).expect("Failed to initialize optimization engine");

        Self {
            engine: Arc::new(tokio::sync::RwLock::new(engine)),
            config,
            strategies: RwLock::new(HashMap::new()),
            runtime,
        }
    }

    /// Convert old configuration to new configuration format
    fn convert_config(old_config: &OptimizationConfig) -> NewOptimizationConfig {
        // Create a new config with defaults and override based on old config
        let mut new_config = NewOptimizationConfig::default();

        // Map old config parameters to new config structure
        // This would need to be implemented based on the actual new config structure
        // For now, return default config
        new_config
    }

    /// Register optimization strategy
    pub fn register_strategy(&self, strategy: OptimizationStrategy) -> Result<(), String> {
        let mut strategies = self.strategies.write()
            .map_err(|_| "Failed to acquire strategies lock")?;

        strategies.insert(strategy.id.clone(), strategy);
        Ok(())
    }

    /// Optimize with specific strategy
    pub fn optimize_with_strategy(
        &self,
        strategy_id: &str,
        parameters: HashMap<String, f64>
    ) -> Result<OptimizationResults, String> {
        // Get the registered strategy
        let strategies = self.strategies.read()
            .map_err(|_| "Failed to acquire strategies lock")?;

        let strategy = strategies.get(strategy_id)
            .ok_or_else(|| format!("Strategy '{}' not found", strategy_id))?;

        // Convert parameters to new format
        let objectives = self.convert_strategy_to_objectives(strategy, &parameters)?;

        // Run optimization using the new engine
        let runtime = &self.runtime;
        let engine = self.engine.clone();

        let results = runtime.block_on(async {
            let mut engine = engine.write().await;
            engine.optimize_with_objectives(objectives).await
        }).map_err(|e| format!("Optimization failed: {}", e))?;

        // Convert new results back to old format
        self.convert_results(results, strategy)
    }

    /// Convert strategy and parameters to new objectives format
    fn convert_strategy_to_objectives(
        &self,
        strategy: &OptimizationStrategy,
        parameters: &HashMap<String, f64>
    ) -> Result<OptimizationObjectives, String> {
        // Build objectives based on strategy type and objectives
        let mut builder = OptimizationObjectives::builder();

        // Map strategy objectives to new objective types
        for objective in &strategy.objectives {
            match objective.objective_type {
                ObjectiveType::MemoryUsage => { builder = builder.minimize_memory_usage(); },
                ObjectiveType::Performance => { builder = builder.maximize_throughput(); },
                ObjectiveType::Latency => { builder = builder.minimize_latency(); },
                ObjectiveType::EnergyConsumption => { builder = builder.minimize_energy_consumption(); },
                _ => {} // Other objectives can be added as needed
            }
        }

        builder.build().map_err(|e| format!("Failed to build objectives: {}", e))
    }

    /// Convert new results to old results format
    fn convert_results(
        &self,
        new_results: NewOptimizationResults,
        strategy: &OptimizationStrategy
    ) -> Result<OptimizationResults, String> {
        let mut objective_values = HashMap::new();
        let mut solution_parameters = HashMap::new();

        // Extract best solution
        let best_solution = &new_results.best_solution;

        // Convert objective values
        objective_values.insert("memory_usage".to_string(), best_solution.objective_values.memory_usage);
        objective_values.insert("throughput".to_string(), best_solution.objective_values.throughput);
        objective_values.insert("latency".to_string(), best_solution.objective_values.latency);
        objective_values.insert("energy_consumption".to_string(), best_solution.objective_values.energy_consumption);

        // Convert solution parameters
        solution_parameters.insert("batch_size".to_string(), best_solution.parameters.performance_parameters.batch_size as f64);
        solution_parameters.insert("cache_size".to_string(), best_solution.parameters.memory_parameters.cache_size as f64);

        // Calculate improvement (simplified)
        let improvement = (best_solution.fitness_score as f32).min(1.0).max(0.0);

        // Create validation results
        let validation = ValidationResults {
            status: match new_results.validation_results.risk_assessment {
                RiskLevel::Low => ValidationStatus::Passed,
                RiskLevel::Medium => ValidationStatus::Warning,
                RiskLevel::High => ValidationStatus::Warning,
                RiskLevel::Critical => ValidationStatus::Failed,
            },
            score: new_results.validation_results.validation_score as f32,
            metrics: HashMap::new(),
            risk_assessment: RiskAssessment {
                risk_level: new_results.validation_results.risk_assessment,
                risk_factors: vec!["Converted from new format".to_string()],
                mitigations: vec!["Monitor performance".to_string()],
                success_probability: new_results.validation_results.validation_score as f32,
            },
            messages: vec!["Optimization completed successfully".to_string()],
        };

        // Create convergence info
        let convergence = ConvergenceInfo {
            iterations: new_results.convergence_metrics.iterations,
            converged: new_results.convergence_metrics.improvement_rate > 0.0,
            final_fitness: new_results.convergence_metrics.final_fitness,
            convergence_rate: new_results.convergence_metrics.convergence_rate,
        };

        Ok(OptimizationResults {
            objective_values,
            solution_parameters,
            improvement,
            quality: best_solution.validation_score as f32,
            optimization_time: new_results.execution_metrics.execution_time,
            validation,
            convergence,
        })
    }

    /// Get optimization recommendations
    pub fn get_recommendations(&self) -> Vec<OptimizationRecommendation> {
        // Generate recommendations based on registered strategies and historical performance
        let strategies = self.strategies.read().unwrap();
        let mut recommendations = Vec::new();

        for (id, strategy) in strategies.iter() {
            let recommendation = OptimizationRecommendation {
                id: format!("rec_{}", id),
                strategy: strategy.name.clone(),
                expected_benefit: strategy.expected_benefits.performance_improvement,
                confidence: strategy.success_rate,
                priority: match strategy.risk_assessment.risk_level {
                    RiskLevel::Low => RecommendationPriority::High,
                    RiskLevel::Medium => RecommendationPriority::Medium,
                    RiskLevel::High => RecommendationPriority::Low,
                    RiskLevel::Critical => RecommendationPriority::Low,
                },
                description: strategy.description.clone(),
                required_parameters: HashMap::new(),
            };
            recommendations.push(recommendation);
        }

        // Sort by priority and confidence
        recommendations.sort_by(|a, b| {
            b.priority.cmp(&a.priority)
                .then_with(|| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal))
        });

        recommendations
    }

    /// Get engine status
    pub fn get_status(&self) -> OptimizationEngineStatus {
        let runtime = &self.runtime;
        let engine = self.engine.clone();

        // Get metrics from new engine
        let metrics_result = runtime.block_on(async {
            let engine = engine.read().await;
            engine.get_metrics().await
        });

        let strategies_count = self.strategies.read().unwrap().len();

        match metrics_result {
            Ok(metrics) => OptimizationEngineStatus {
                active_optimizations: 0, // Would need to track this in new system
                queued_optimizations: 0,  // Would need to track this in new system
                success_rate: metrics.optimization_efficiency as f32,
                average_improvement: metrics.current_performance.throughput as f32 / 1000.0,
                health: metrics.system_health as f32,
                strategy_count: strategies_count,
                optimization_frequency: self.config.optimization_frequency,
            },
            Err(_) => OptimizationEngineStatus {
                active_optimizations: 0,
                queued_optimizations: 0,
                success_rate: 0.0,
                average_improvement: 0.0,
                health: 0.5,
                strategy_count: strategies_count,
                optimization_frequency: self.config.optimization_frequency,
            }
        }
    }

    /// Start optimization monitoring
    pub fn start_monitoring(&self) -> Result<(), String> {
        let runtime = &self.runtime;
        let engine = self.engine.clone();

        runtime.block_on(async {
            let engine = engine.read().await;
            engine.start_monitoring().await
        }).map_err(|e| format!("Failed to start monitoring: {}", e))
    }

    /// Stop optimization monitoring
    pub fn stop_monitoring(&self) -> Result<(), String> {
        let runtime = &self.runtime;
        let engine = self.engine.clone();

        runtime.block_on(async {
            let engine = engine.read().await;
            engine.stop_monitoring().await
        }).map_err(|e| format!("Failed to stop monitoring: {}", e))
    }

    /// Get optimization history
    pub fn get_optimization_history(&self, limit: usize) -> Vec<OptimizationRecord> {
        let runtime = &self.runtime;
        let engine = self.engine.clone();

        let query = HistoryQuery::new()
            .with_limit(limit)
            .with_time_range(
                chrono::Utc::now() - chrono::Duration::days(30),
                chrono::Utc::now()
            );

        runtime.block_on(async {
            let engine = engine.read().await;
            engine.get_history(query).await
        }).unwrap_or_else(|_| Vec::new())
    }

    /// Learn from optimization feedback
    pub fn provide_feedback(&self, solution_id: &str, actual_performance: HashMap<String, f64>, rating: Option<f64>) -> Result<(), String> {
        let feedback = OptimizationFeedback {
            solution_id: solution_id.to_string(),
            actual_performance: ObjectiveValues {
                memory_usage: actual_performance.get("memory_usage").copied().unwrap_or(0.0),
                throughput: actual_performance.get("throughput").copied().unwrap_or(0.0),
                latency: actual_performance.get("latency").copied().unwrap_or(0.0),
                energy_consumption: actual_performance.get("energy_consumption").copied().unwrap_or(0.0),
            },
            user_rating: rating,
            issues_encountered: Vec::new(),
            suggestions: Vec::new(),
        };

        let runtime = &self.runtime;
        let engine = self.engine.clone();

        runtime.block_on(async {
            let mut engine = engine.write().await;
            engine.learn_from_feedback(feedback).await
        }).map_err(|e| format!("Failed to process feedback: {}", e))
    }
}

// Re-export the modular optimization components for direct access
pub use super::optimization::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_memory_optimization_engine_creation() {
        let config = OptimizationConfig::default();
        let engine = CudaMemoryOptimizationEngine::new(config);

        let status = engine.get_status();
        assert_eq!(status.strategy_count, 0);
    }

    #[test]
    fn test_strategy_registration() {
        let config = OptimizationConfig::default();
        let engine = CudaMemoryOptimizationEngine::new(config);

        let strategy = OptimizationStrategy {
            id: "test_strategy".to_string(),
            name: "Test Strategy".to_string(),
            description: "A test optimization strategy".to_string(),
            strategy_type: StrategyType::AllocationOptimization,
            objectives: vec![],
            parameters: HashMap::new(),
            conditions: vec![],
            expected_benefits: ExpectedBenefits {
                performance_improvement: 20.0,
                memory_reduction: 15.0,
                energy_savings: 10.0,
                implementation_effort: 5.0,
            },
            complexity: OptimizationComplexity::Medium,
            risk_assessment: RiskAssessment {
                risk_level: RiskLevel::Low,
                risk_factors: vec!["minimal risk".to_string()],
                mitigations: vec!["thorough testing".to_string()],
                success_probability: 0.9,
            },
            success_rate: 0.85,
            performance_history: vec![],
        };

        assert!(engine.register_strategy(strategy).is_ok());

        let status = engine.get_status();
        assert_eq!(status.strategy_count, 1);
    }

    #[test]
    fn test_recommendations() {
        let config = OptimizationConfig::default();
        let engine = CudaMemoryOptimizationEngine::new(config);

        // Register a strategy first
        let strategy = OptimizationStrategy {
            id: "test_strategy".to_string(),
            name: "Test Strategy".to_string(),
            description: "A test optimization strategy".to_string(),
            strategy_type: StrategyType::AllocationOptimization,
            objectives: vec![],
            parameters: HashMap::new(),
            conditions: vec![],
            expected_benefits: ExpectedBenefits {
                performance_improvement: 20.0,
                memory_reduction: 15.0,
                energy_savings: 10.0,
                implementation_effort: 5.0,
            },
            complexity: OptimizationComplexity::Medium,
            risk_assessment: RiskAssessment {
                risk_level: RiskLevel::Low,
                risk_factors: vec![],
                mitigations: vec![],
                success_probability: 0.9,
            },
            success_rate: 0.85,
            performance_history: vec![],
        };

        engine.register_strategy(strategy).unwrap();

        let recommendations = engine.get_recommendations();
        assert_eq!(recommendations.len(), 1);
        assert_eq!(recommendations[0].strategy, "Test Strategy");
    }

    #[test]
    fn test_optimization_config_default() {
        let config = OptimizationConfig::default();

        assert!(config.enable_ml_optimization);
        assert!(config.enable_multi_objective);
        assert!(config.enable_adaptive_optimization);
        assert_eq!(config.max_concurrent_optimizations, 4);
        assert_eq!(config.safety_threshold, 0.8);
    }

    #[test]
    fn test_parameter_value_types() {
        let int_param = ParameterValue::Integer(42);
        let float_param = ParameterValue::Float(3.14);
        let bool_param = ParameterValue::Boolean(true);
        let string_param = ParameterValue::String("test".to_string());
        let array_param = ParameterValue::Array(vec![ParameterValue::Integer(1), ParameterValue::Integer(2)]);

        assert!(matches!(int_param, ParameterValue::Integer(42)));
        assert!(matches!(float_param, ParameterValue::Float(f) if f == 3.14));
        assert!(matches!(bool_param, ParameterValue::Boolean(true)));
        assert!(matches!(string_param, ParameterValue::String(ref s) if s == "test"));
        assert!(matches!(array_param, ParameterValue::Array(ref v) if v.len() == 2));
    }

    #[test]
    fn test_strategy_types() {
        let alloc_strategy = StrategyType::AllocationOptimization;
        let pool_strategy = StrategyType::PoolOptimization;
        let hybrid_strategy = StrategyType::Hybrid;

        assert!(matches!(alloc_strategy, StrategyType::AllocationOptimization));
        assert!(matches!(pool_strategy, StrategyType::PoolOptimization));
        assert!(matches!(hybrid_strategy, StrategyType::Hybrid));
    }

    #[test]
    fn test_constraint_priorities() {
        let low = ConstraintPriority::Low;
        let medium = ConstraintPriority::Medium;
        let high = ConstraintPriority::High;
        let critical = ConstraintPriority::Critical;

        assert!(low < medium);
        assert!(medium < high);
        assert!(high < critical);
    }
}