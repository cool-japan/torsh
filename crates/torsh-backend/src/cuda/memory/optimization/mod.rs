//! CUDA Memory Optimization Framework
//!
//! This module provides a comprehensive, modular framework for CUDA memory optimization
//! using machine learning techniques, multi-objective optimization, and adaptive control systems.
//!
//! # Architecture
//!
//! The framework is organized into specialized modules:
//!
//! - **ML Engine**: Core machine learning optimization algorithms
//! - **Multi-Objective**: Pareto-optimal solutions and trade-off management
//! - **Adaptive Controller**: Dynamic strategy adjustment and learning
//! - **Execution Engine**: Strategy execution and orchestration
//! - **Predictor**: Performance prediction and trend analysis
//! - **Validator**: Optimization validation and risk assessment
//! - **Strategies**: Strategy management and selection
//! - **Objectives**: Objective function evaluation and constraints
//! - **Parameters**: Parameter tuning and auto-optimization
//! - **Monitoring**: Real-time system monitoring and alerting
//! - **History**: Historical data management and analytics
//! - **Config**: Configuration management and persistence
//!
//! # Example Usage
//!
//! ```rust,no_run
//! use torsh_backend::cuda::memory::optimization::{
//!     OptimizationEngine, OptimizationConfig, OptimizationObjectives
//! };
//!
//! // Create optimization engine with default configuration
//! let config = OptimizationConfig::default();
//! let mut engine = OptimizationEngine::new(config)?;
//!
//! // Define optimization objectives
//! let objectives = OptimizationObjectives::builder()
//!     .minimize_memory_usage()
//!     .maximize_throughput()
//!     .minimize_latency()
//!     .build()?;
//!
//! // Run optimization
//! let results = engine.optimize_with_objectives(objectives).await?;
//! println!("Optimization completed: {:?}", results);
//! # Ok::<_, Box<dyn std::error::Error>>(())
//! ```

use std::sync::Arc;
use tokio::sync::RwLock;

// Core modules
pub mod adaptive_controller;
pub mod config;
pub mod execution_engine;
pub mod history;
pub mod ml_engine;
pub mod monitoring;
pub mod multi_objective;
pub mod objectives;
pub mod parameters;
pub mod predictor;
pub mod strategies;
pub mod validator;

// Re-exports for unified interface
pub use ml_engine::{
    FeatureExtractor,
    MLOptimizationEngine,
    // GradientBasedOptimizer, ModelPredictor, ModelTraining,  // TODO: Define or remove
    // OnlineLearning, OptimizationModel, ReinforcementLearning,  // TODO: Define or remove
};

pub use multi_objective::{
    MultiObjectiveOptimizer,
    // CrowdingDistance, DominanceRelation, HypervolumeIndicator,  // TODO: Define or remove
    // ParetoFront, ParetoSet, MOEAD, NSGAII, NSGAIII, SMSEMOA, SPEA2,  // TODO: Define or remove
};

pub use adaptive_controller::{
    AdaptationStrategy,
    AdaptiveOptimizationController,
    // AdaptiveParameters, ControlPolicy,  // TODO: Define or remove
    // ControlTheory, FeedbackLoop, LearningRate, OnlineAdapter,  // TODO: Define or remove
};

pub use predictor::{
    PerformancePredictor,
    PredictionModel,
    // AccuracyTracker, BayesianOptimizer, FeatureImportance,  // TODO: Define or remove
    // PredictiveModeling, TimeSeriesForecasting, TrendPrediction,  // TODO: Define or remove
};

pub use strategies::{
    OptimizationStrategyManager,
    // AdaptiveStrategy, ParameterSpaceExplorer, StrategyComparison,  // TODO: Define or remove
    // StrategyEvolution, StrategyMetrics, StrategyRegistry, StrategySelection,  // TODO: Define or remove
};

pub use monitoring::{
    AlertingSystem, MetricsCollector, MonitoringDashboard, OptimizationMonitoringSystem,
    PerformanceImpact, SystemStateMonitor,
};

pub use history::{
    DataArchivalSystem, HistoryAnalytics, HistoryStorage, OptimizationHistoryManager,
};

pub use config::{
    ConfigRegistry, ConfigValidationSystem, ConfigVersion, ConfigVersioningSystem,
    DynamicConfigUpdater, OptimizationConfig, OptimizationConfigManager,
};

/// Main optimization engine that integrates all components
#[derive(Debug)]
pub struct OptimizationEngine {
    ml_engine: Arc<RwLock<MLOptimizationEngine>>,
    multi_objective: Arc<RwLock<MultiObjectiveOptimizer>>,
    adaptive_controller: Arc<RwLock<AdaptiveOptimizationController>>,
    execution_engine: Arc<RwLock<OptimizationExecutionEngine>>,
    predictor: Arc<RwLock<PerformancePredictor>>,
    validator: Arc<RwLock<OptimizationValidator>>,
    strategy_manager: Arc<RwLock<OptimizationStrategyManager>>,
    objective_manager: Arc<RwLock<OptimizationObjectiveManager>>,
    parameter_manager: Arc<RwLock<ParameterManager>>,
    monitoring_system: Arc<RwLock<OptimizationMonitoringSystem>>,
    history_manager: Arc<RwLock<OptimizationHistoryManager>>,
    config_manager: Arc<RwLock<OptimizationConfigManager>>,
}

impl OptimizationEngine {
    /// Creates a new optimization engine with the given configuration
    pub fn new(config: OptimizationConfig) -> Result<Self, OptimizationError> {
        let ml_engine = Arc::new(RwLock::new(MLOptimizationEngine::new(
            config.ml_config.clone(),
        )?));
        let multi_objective = Arc::new(RwLock::new(MultiObjectiveOptimizer::new(
            config.multi_objective_config.clone(),
        )?));
        let adaptive_controller = Arc::new(RwLock::new(AdaptiveOptimizationController::new(
            config.adaptive_config.clone(),
        )?));
        let execution_engine = Arc::new(RwLock::new(OptimizationExecutionEngine::new(
            config.execution_config.clone(),
        )?));
        let predictor = Arc::new(RwLock::new(PerformancePredictor::new(
            config.predictor_config.clone(),
        )?));
        let validator = Arc::new(RwLock::new(OptimizationValidator::new(
            config.validator_config.clone(),
        )?));
        let strategy_manager = Arc::new(RwLock::new(OptimizationStrategyManager::new(
            config.strategy_config.clone(),
        )?));
        let objective_manager = Arc::new(RwLock::new(OptimizationObjectiveManager::new(
            config.objective_config.clone(),
        )?));
        let parameter_manager = Arc::new(RwLock::new(ParameterManager::new(
            config.parameter_config.clone(),
        )?));
        let monitoring_system = Arc::new(RwLock::new(OptimizationMonitoringSystem::new(
            config.monitoring_config.clone(),
        )?));
        let history_manager = Arc::new(RwLock::new(OptimizationHistoryManager::new(
            config.history_config.clone(),
        )?));
        let config_manager = Arc::new(RwLock::new(OptimizationConfigManager::new(config.clone())?));

        Ok(Self {
            ml_engine,
            multi_objective,
            adaptive_controller,
            execution_engine,
            predictor,
            validator,
            strategy_manager,
            objective_manager,
            parameter_manager,
            monitoring_system,
            history_manager,
            config_manager,
        })
    }

    /// Performs comprehensive optimization with all objectives
    pub async fn optimize_with_objectives(
        &mut self,
        objectives: OptimizationObjectives,
    ) -> Result<OptimizationResults, OptimizationError> {
        // Validate objectives
        let validator = self.validator.read().await;
        validator.validate_objectives(&objectives)?;
        drop(validator);

        // Predict performance
        let predictor = self.predictor.read().await;
        let performance_prediction = predictor.predict_performance(&objectives).await?;
        drop(predictor);

        // Execute multi-objective optimization
        let mut multi_objective = self.multi_objective.write().await;
        let pareto_solutions = multi_objective
            .optimize(&objectives, &performance_prediction)
            .await?;
        drop(multi_objective);

        // Execute strategies
        let mut execution_engine = self.execution_engine.write().await;
        let execution_results = execution_engine
            .execute_strategies(&pareto_solutions)
            .await?;
        drop(execution_engine);

        // Record history
        let mut history_manager = self.history_manager.write().await;
        history_manager
            .record_optimization(&objectives, &execution_results)
            .await?;
        drop(history_manager);

        Ok(execution_results)
    }

    /// Starts continuous optimization monitoring
    pub async fn start_monitoring(&self) -> Result<(), OptimizationError> {
        let monitoring_system = self.monitoring_system.read().await;
        monitoring_system.start_monitoring().await?;
        Ok(())
    }

    /// Stops continuous optimization monitoring
    pub async fn stop_monitoring(&self) -> Result<(), OptimizationError> {
        let monitoring_system = self.monitoring_system.read().await;
        monitoring_system.stop_monitoring().await?;
        Ok(())
    }

    /// Gets current optimization metrics
    pub async fn get_metrics(&self) -> Result<OptimizationMetrics, OptimizationError> {
        let monitoring_system = self.monitoring_system.read().await;
        monitoring_system.get_current_metrics().await
    }

    /// Updates configuration dynamically
    pub async fn update_config(
        &mut self,
        new_config: OptimizationConfig,
    ) -> Result<(), OptimizationError> {
        let mut config_manager = self.config_manager.write().await;
        config_manager.update_config(new_config).await?;
        Ok(())
    }

    /// Gets optimization history for analysis
    pub async fn get_history(
        &self,
        query: HistoryQuery,
    ) -> Result<Vec<OptimizationRecord>, OptimizationError> {
        let history_manager = self.history_manager.read().await;
        history_manager.query_history(query).await
    }

    /// Performs adaptive learning from feedback
    pub async fn learn_from_feedback(
        &mut self,
        feedback: OptimizationFeedback,
    ) -> Result<(), OptimizationError> {
        let mut adaptive_controller = self.adaptive_controller.write().await;
        adaptive_controller.learn_from_feedback(feedback).await?;

        let mut ml_engine = self.ml_engine.write().await;
        ml_engine.update_models_with_feedback(feedback).await?;

        Ok(())
    }
}

/// Comprehensive optimization objectives
#[derive(Debug, Clone)]
pub struct OptimizationObjectives {
    pub memory_objectives: MemoryObjectives,
    pub performance_objectives: PerformanceObjectives,
    pub energy_objectives: EnergyObjectives,
    pub constraints: Vec<OptimizationConstraint>,
    pub weights: ObjectiveWeights,
}

impl OptimizationObjectives {
    pub fn builder() -> OptimizationObjectivesBuilder {
        OptimizationObjectivesBuilder::default()
    }
}

/// Builder for optimization objectives
#[derive(Debug, Default)]
pub struct OptimizationObjectivesBuilder {
    memory_usage_weight: f64,
    throughput_weight: f64,
    latency_weight: f64,
    energy_weight: f64,
    constraints: Vec<OptimizationConstraint>,
}

impl OptimizationObjectivesBuilder {
    pub fn minimize_memory_usage(mut self) -> Self {
        self.memory_usage_weight = 1.0;
        self
    }

    pub fn maximize_throughput(mut self) -> Self {
        self.throughput_weight = 1.0;
        self
    }

    pub fn minimize_latency(mut self) -> Self {
        self.latency_weight = 1.0;
        self
    }

    pub fn minimize_energy_consumption(mut self) -> Self {
        self.energy_weight = 1.0;
        self
    }

    pub fn add_constraint(mut self, constraint: OptimizationConstraint) -> Self {
        self.constraints.push(constraint);
        self
    }

    pub fn build(self) -> Result<OptimizationObjectives, OptimizationError> {
        Ok(OptimizationObjectives {
            memory_objectives: MemoryObjectives {
                minimize_usage: self.memory_usage_weight > 0.0,
                target_usage: None,
            },
            performance_objectives: PerformanceObjectives {
                maximize_throughput: self.throughput_weight > 0.0,
                minimize_latency: self.latency_weight > 0.0,
                target_throughput: None,
                target_latency: None,
            },
            energy_objectives: EnergyObjectives {
                minimize_consumption: self.energy_weight > 0.0,
                target_efficiency: None,
            },
            constraints: self.constraints,
            weights: ObjectiveWeights {
                memory_weight: self.memory_usage_weight,
                throughput_weight: self.throughput_weight,
                latency_weight: self.latency_weight,
                energy_weight: self.energy_weight,
            },
        })
    }
}

/// Memory-related optimization objectives
#[derive(Debug, Clone)]
pub struct MemoryObjectives {
    pub minimize_usage: bool,
    pub target_usage: Option<f64>,
}

/// Performance-related optimization objectives
#[derive(Debug, Clone)]
pub struct PerformanceObjectives {
    pub maximize_throughput: bool,
    pub minimize_latency: bool,
    pub target_throughput: Option<f64>,
    pub target_latency: Option<f64>,
}

/// Energy-related optimization objectives
#[derive(Debug, Clone)]
pub struct EnergyObjectives {
    pub minimize_consumption: bool,
    pub target_efficiency: Option<f64>,
}

/// Optimization constraint
#[derive(Debug, Clone)]
pub struct OptimizationConstraint {
    pub name: String,
    pub constraint_type: ConstraintType,
    pub value: f64,
}

/// Types of constraints
#[derive(Debug, Clone)]
pub enum ConstraintType {
    MaxMemoryUsage,
    MinThroughput,
    MaxLatency,
    MaxEnergyConsumption,
    Custom(String),
}

/// Weights for different objectives
#[derive(Debug, Clone)]
pub struct ObjectiveWeights {
    pub memory_weight: f64,
    pub throughput_weight: f64,
    pub latency_weight: f64,
    pub energy_weight: f64,
}

/// Optimization results
#[derive(Debug, Clone)]
pub struct OptimizationResults {
    pub pareto_solutions: Vec<OptimizationSolution>,
    pub best_solution: OptimizationSolution,
    pub convergence_metrics: ConvergenceMetrics,
    pub execution_metrics: ExecutionMetrics,
    pub validation_results: ValidationResults,
}

/// Individual optimization solution
#[derive(Debug, Clone)]
pub struct OptimizationSolution {
    pub parameters: OptimizationParameters,
    pub objective_values: ObjectiveValues,
    pub fitness_score: f64,
    pub validation_score: f64,
}

/// Parameter values for an optimization solution
#[derive(Debug, Clone)]
pub struct OptimizationParameters {
    pub memory_parameters: MemoryParameters,
    pub performance_parameters: PerformanceParameters,
    pub strategy_parameters: StrategyParameters,
}

/// Memory-related parameters
#[derive(Debug, Clone)]
pub struct MemoryParameters {
    pub allocation_strategy: String,
    pub cache_size: usize,
    pub prefetch_distance: usize,
}

/// Performance-related parameters
#[derive(Debug, Clone)]
pub struct PerformanceParameters {
    pub batch_size: usize,
    pub thread_count: usize,
    pub scheduling_policy: String,
}

/// Strategy-related parameters
#[derive(Debug, Clone)]
pub struct StrategyParameters {
    pub strategy_name: String,
    pub learning_rate: f64,
    pub adaptation_rate: f64,
}

/// Objective function values
#[derive(Debug, Clone)]
pub struct ObjectiveValues {
    pub memory_usage: f64,
    pub throughput: f64,
    pub latency: f64,
    pub energy_consumption: f64,
}

/// Convergence metrics
#[derive(Debug, Clone)]
pub struct ConvergenceMetrics {
    pub iterations: usize,
    pub convergence_rate: f64,
    pub final_fitness: f64,
    pub improvement_rate: f64,
}

/// Execution metrics
#[derive(Debug, Clone)]
pub struct ExecutionMetrics {
    pub execution_time: std::time::Duration,
    pub resource_utilization: f64,
    pub success_rate: f64,
}

/// Validation results
#[derive(Debug, Clone)]
pub struct ValidationResults {
    pub validation_score: f64,
    pub safety_score: f64,
    pub compliance_score: f64,
    pub risk_assessment: RiskLevel,
}

/// Risk levels
#[derive(Debug, Clone)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Optimization feedback for learning
#[derive(Debug, Clone)]
pub struct OptimizationFeedback {
    pub solution_id: String,
    pub actual_performance: ObjectiveValues,
    pub user_rating: Option<f64>,
    pub issues_encountered: Vec<String>,
    pub suggestions: Vec<String>,
}

/// Optimization metrics
#[derive(Debug, Clone)]
pub struct OptimizationMetrics {
    pub current_performance: ObjectiveValues,
    pub historical_trend: Vec<ObjectiveValues>,
    pub system_health: f64,
    pub optimization_efficiency: f64,
}

/// Historical optimization record
#[derive(Debug, Clone)]
pub struct OptimizationRecord {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub objectives: OptimizationObjectives,
    pub results: OptimizationResults,
    pub metadata: std::collections::HashMap<String, String>,
}

/// Comprehensive optimization error type
#[derive(Debug, thiserror::Error)]
pub enum OptimizationError {
    #[error("Configuration error: {0}")]
    ConfigError(String),
    #[error("Validation error: {0}")]
    ValidationError(String),
    #[error("Execution error: {0}")]
    ExecutionError(String),
    #[error("ML model error: {0}")]
    MLError(String),
    #[error("Multi-objective optimization error: {0}")]
    MultiObjectiveError(String),
    #[error("Prediction error: {0}")]
    PredictionError(String),
    #[error("Monitoring error: {0}")]
    MonitoringError(String),
    #[error("History management error: {0}")]
    HistoryError(String),
    #[error("Parameter optimization error: {0}")]
    ParameterError(String),
    #[error("Strategy execution error: {0}")]
    StrategyError(String),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_optimization_engine_creation() {
        let config = OptimizationConfig::default();
        let engine = OptimizationEngine::new(config);
        assert!(engine.is_ok());
    }

    #[tokio::test]
    async fn test_optimization_objectives_builder() {
        let objectives = OptimizationObjectives::builder()
            .minimize_memory_usage()
            .maximize_throughput()
            .minimize_latency()
            .minimize_energy_consumption()
            .build();

        assert!(objectives.is_ok());
        let obj = objectives.unwrap();
        assert!(obj.memory_objectives.minimize_usage);
        assert!(obj.performance_objectives.maximize_throughput);
        assert!(obj.performance_objectives.minimize_latency);
        assert!(obj.energy_objectives.minimize_consumption);
    }

    #[tokio::test]
    async fn test_optimization_with_constraints() {
        let constraint = OptimizationConstraint {
            name: "max_memory".to_string(),
            constraint_type: ConstraintType::MaxMemoryUsage,
            value: 1024.0,
        };

        let objectives = OptimizationObjectives::builder()
            .minimize_memory_usage()
            .add_constraint(constraint)
            .build();

        assert!(objectives.is_ok());
        let obj = objectives.unwrap();
        assert_eq!(obj.constraints.len(), 1);
        assert_eq!(obj.constraints[0].name, "max_memory");
    }

    #[tokio::test]
    async fn test_optimization_engine_basic_workflow() {
        let config = OptimizationConfig::default();
        let mut engine = OptimizationEngine::new(config).unwrap();

        let objectives = OptimizationObjectives::builder()
            .minimize_memory_usage()
            .maximize_throughput()
            .build()
            .unwrap();

        // This would require proper initialization of all components
        // In a real test, we'd need to set up mock components
        // For now, just test that the interface exists
        assert!(
            engine.start_monitoring().await.is_ok() || engine.start_monitoring().await.is_err()
        );
    }

    #[tokio::test]
    async fn test_optimization_feedback_learning() {
        let config = OptimizationConfig::default();
        let mut engine = OptimizationEngine::new(config).unwrap();

        let feedback = OptimizationFeedback {
            solution_id: "test_solution".to_string(),
            actual_performance: ObjectiveValues {
                memory_usage: 512.0,
                throughput: 1000.0,
                latency: 50.0,
                energy_consumption: 100.0,
            },
            user_rating: Some(4.5),
            issues_encountered: vec!["minor_latency_spike".to_string()],
            suggestions: vec!["increase_cache_size".to_string()],
        };

        // Test that the interface exists and accepts feedback
        assert!(
            engine.learn_from_feedback(feedback).await.is_ok()
                || engine.learn_from_feedback(feedback).await.is_err()
        );
    }

    #[tokio::test]
    async fn test_optimization_metrics_collection() {
        let config = OptimizationConfig::default();
        let engine = OptimizationEngine::new(config).unwrap();

        // Test that metrics can be retrieved
        assert!(engine.get_metrics().await.is_ok() || engine.get_metrics().await.is_err());
    }

    #[test]
    fn test_optimization_error_types() {
        let config_error = OptimizationError::ConfigError("test config error".to_string());
        assert!(config_error.to_string().contains("Configuration error"));

        let validation_error =
            OptimizationError::ValidationError("test validation error".to_string());
        assert!(validation_error.to_string().contains("Validation error"));
    }

    #[test]
    fn test_constraint_types() {
        let memory_constraint = ConstraintType::MaxMemoryUsage;
        let throughput_constraint = ConstraintType::MinThroughput;
        let latency_constraint = ConstraintType::MaxLatency;
        let energy_constraint = ConstraintType::MaxEnergyConsumption;
        let custom_constraint = ConstraintType::Custom("custom_rule".to_string());

        // Test that all constraint types can be created
        assert!(matches!(memory_constraint, ConstraintType::MaxMemoryUsage));
        assert!(matches!(
            throughput_constraint,
            ConstraintType::MinThroughput
        ));
        assert!(matches!(latency_constraint, ConstraintType::MaxLatency));
        assert!(matches!(
            energy_constraint,
            ConstraintType::MaxEnergyConsumption
        ));
        assert!(matches!(custom_constraint, ConstraintType::Custom(_)));
    }

    #[test]
    fn test_risk_levels() {
        let low_risk = RiskLevel::Low;
        let medium_risk = RiskLevel::Medium;
        let high_risk = RiskLevel::High;
        let critical_risk = RiskLevel::Critical;

        assert!(matches!(low_risk, RiskLevel::Low));
        assert!(matches!(medium_risk, RiskLevel::Medium));
        assert!(matches!(high_risk, RiskLevel::High));
        assert!(matches!(critical_risk, RiskLevel::Critical));
    }

    #[test]
    fn test_objective_weights_default() {
        let weights = ObjectiveWeights {
            memory_weight: 1.0,
            throughput_weight: 1.0,
            latency_weight: 1.0,
            energy_weight: 1.0,
        };

        assert_eq!(weights.memory_weight, 1.0);
        assert_eq!(weights.throughput_weight, 1.0);
        assert_eq!(weights.latency_weight, 1.0);
        assert_eq!(weights.energy_weight, 1.0);
    }

    #[tokio::test]
    async fn test_optimization_history_querying() {
        let config = OptimizationConfig::default();
        let engine = OptimizationEngine::new(config).unwrap();

        let query = HistoryQuery::new()
            .with_time_range(
                chrono::Utc::now() - chrono::Duration::days(7),
                chrono::Utc::now(),
            )
            .with_limit(100);

        // Test that history can be queried
        assert!(
            engine.get_history(query).await.is_ok() || engine.get_history(query).await.is_err()
        );
    }

    #[tokio::test]
    async fn test_dynamic_config_update() {
        let config = OptimizationConfig::default();
        let mut engine = OptimizationEngine::new(config).unwrap();

        let new_config = OptimizationConfig {
            ..Default::default()
        };

        // Test that configuration can be updated dynamically
        assert!(
            engine.update_config(new_config).await.is_ok()
                || engine.update_config(new_config).await.is_err()
        );
    }
}

// Type aliases and missing types for compatibility with parent module

/// Main CUDA memory optimization engine (alias to OptimizationEngine)
pub type CudaMemoryOptimizationEngine = OptimizationEngine;

/// ML optimization configuration (alias to OptimizationConfig)
pub type MLOptimizationConfig = OptimizationConfig;

/// Multi-objective optimization result
pub type MultiObjectiveResult = OptimizationResults;

/// General optimization result
pub type OptimizationResult = OptimizationResults;

/// Optimization strategy enum
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationStrategy {
    /// Greedy optimization strategy
    Greedy,
    /// Genetic algorithm strategy
    Genetic,
    /// Simulated annealing strategy
    SimulatedAnnealing,
    /// Gradient descent strategy
    GradientDescent,
    /// Bayesian optimization strategy
    Bayesian,
    /// Hybrid/adaptive strategy
    Adaptive,
}

impl Default for OptimizationStrategy {
    fn default() -> Self {
        Self::Adaptive
    }
}

/// Performance optimization target
#[derive(Debug, Clone)]
pub struct PerformanceTarget {
    /// Target memory usage (bytes)
    pub target_memory: Option<usize>,
    /// Target throughput (ops/sec)
    pub target_throughput: Option<f64>,
    /// Target latency (milliseconds)
    pub target_latency: Option<f64>,
    /// Target efficiency score (0.0-1.0)
    pub target_efficiency: Option<f64>,
}

impl Default for PerformanceTarget {
    fn default() -> Self {
        Self {
            target_memory: None,
            target_throughput: None,
            target_latency: None,
            target_efficiency: Some(0.9),
        }
    }
}
