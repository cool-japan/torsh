/// Change detection algorithm
#[derive(Debug)]
pub struct ChangeDetector {
    /// Detection algorithm type
    pub algorithm_type: ChangeDetectionAlgorithm,
    /// Algorithm parameters
    pub parameters: HashMap<String, f64>,
    /// Detection sensitivity
    pub sensitivity: f32,
    /// Detection history
    pub detection_history: Vec<ChangeDetection>,
    /// Statistical models for comparison
    pub baseline_models: HashMap<String, BaselineModel>,
}
impl ChangeDetector {
    fn new() -> Self {
        Self {
            algorithm_type: ChangeDetectionAlgorithm::CUSUM,
            parameters: HashMap::new(),
            sensitivity: 0.5,
            detection_history: Vec::new(),
            baseline_models: HashMap::new(),
        }
    }
}
/// Advanced control parameters
#[derive(Debug, Clone)]
pub struct AdvancedControlParams {
    /// Adaptive exploration rate
    pub adaptive_exploration: bool,
    /// Dynamic threshold adjustment
    pub dynamic_thresholds: bool,
    /// Multi-objective optimization
    pub multi_objective: bool,
    /// Risk tolerance level
    pub risk_tolerance: f32,
    /// Conservative mode
    pub conservative_mode: bool,
    /// Emergency response threshold
    pub emergency_threshold: f32,
}
/// Impact severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ImpactSeverity {
    Negligible,
    Minor,
    Moderate,
    Major,
    Critical,
}
/// Learning history for specific tasks
#[derive(Debug, Clone)]
pub struct TaskLearningHistory {
    /// Task identifier
    pub task_id: String,
    /// Task description
    pub task_description: String,
    /// Strategies tried
    pub strategies_tried: Vec<StrategyAttempt>,
    /// Best performing strategy
    pub best_strategy: Option<String>,
    /// Task completion time
    pub completion_time: Duration,
    /// Final performance achieved
    pub final_performance: f32,
}
/// System state representation
#[derive(Debug, Clone)]
pub struct SystemState {
    /// Performance metrics
    pub performance_metrics: HashMap<String, f64>,
    /// Resource utilization levels
    pub resource_utilization: HashMap<String, f32>,
    /// Workload characteristics
    pub workload_characteristics: HashMap<String, f64>,
    /// Environmental factors
    pub environmental_factors: HashMap<String, f64>,
    /// State timestamp
    pub timestamp: Instant,
    /// State quality indicators
    pub quality_indicators: StateQualityIndicators,
    /// State confidence score
    pub confidence: f32,
}
/// Threshold directions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThresholdDirection {
    Above,
    Below,
    Equal,
    NotEqual,
}
/// Risk levels for strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RiskLevel {
    VeryLow,
    Low,
    Medium,
    High,
    Critical,
}
/// Types of context changes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextChangeType {
    Gradual,
    Sudden,
    Periodic,
    Anomalous,
    Structural,
}
/// Types of operational environments
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnvironmentType {
    Development,
    Testing,
    Staging,
    Production,
    Emergency,
    Maintenance,
}
/// Pipeline configuration
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Enable feature caching
    pub enable_caching: bool,
    /// Parallel processing
    pub parallel: bool,
    /// Memory limit for pipeline
    pub memory_limit: usize,
    /// Pipeline timeout
    pub timeout: Duration,
}
/// Trigger schedule types
#[derive(Debug, Clone)]
pub enum TriggerSchedule {
    Interval { duration: Duration },
    Cron { expression: String },
    OnEvent { event_type: String },
    Adaptive { base_interval: Duration, adjustment_factor: f32 },
}
/// Context for adaptive experiences
#[derive(Debug, Clone)]
pub struct ExperienceContext {
    /// Environmental conditions
    pub environment: HashMap<String, f64>,
    /// System configuration
    pub configuration: HashMap<String, String>,
    /// Active strategies
    pub active_strategies: Vec<String>,
    /// User preferences
    pub user_preferences: HashMap<String, String>,
}
/// Adaptive learning performance
#[derive(Debug, Clone)]
pub struct AdaptiveLearningPerformance {
    /// Learning accuracy
    pub accuracy: f32,
    /// Adaptation speed
    pub adaptation_speed: f32,
    /// Stability score
    pub stability: f32,
    /// Generalization ability
    pub generalization: f32,
    /// Knowledge retention
    pub retention: f32,
    /// Learning efficiency
    pub efficiency: f32,
    /// Convergence rate
    pub convergence_rate: f32,
}
/// Query processing steps
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryProcessingStep {
    Tokenization,
    Normalization,
    StopwordRemoval,
    Stemming,
    SemanticExpansion,
    VectorEmbedding,
}
/// Training characteristics for anomaly models
#[derive(Debug, Clone)]
pub struct TrainingCharacteristics {
    /// Training dataset size
    pub dataset_size: usize,
    /// Feature dimensionality
    pub feature_dimension: usize,
    /// Training duration
    pub training_duration: Duration,
    /// Data quality score
    pub data_quality: f32,
    /// Training timestamp
    pub trained_at: Instant,
}
/// Trend types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrendType {
    Increasing,
    Decreasing,
    Stable,
    Cyclical,
    Volatile,
    Seasonal,
}
/// Types of baseline models
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BaselineModelType {
    MovingAverage,
    ExponentialSmoothing,
    ARIMA,
    GaussianMixture,
    KernelDensity,
    Custom,
}
/// Configuration change types
#[derive(Debug, Clone)]
pub enum ConfigChange {
    UpdateValue { new_value: String },
    ToggleBoolean,
    ResetToDefault,
    ConditionalUpdate { condition: String, true_value: String, false_value: String },
}
/// Meta-learning system for higher-level learning
#[derive(Debug, Clone)]
pub struct MetaLearningSystem {
    /// Meta-learning algorithms
    pub algorithms: Vec<MetaLearningAlgorithm>,
    /// Learning history across different tasks
    pub learning_history: Vec<TaskLearningHistory>,
    /// Meta-features for tasks
    pub meta_features: HashMap<String, Vec<f64>>,
    /// Performance prediction models
    pub performance_predictors: HashMap<String, PerformancePredictionModel>,
    /// Strategy recommendation system
    pub strategy_recommender: StrategyRecommender,
}
impl MetaLearningSystem {
    fn new() -> Self {
        Self {
            algorithms: vec![
                MetaLearningAlgorithm::MAML, MetaLearningAlgorithm::Reptile
            ],
            learning_history: Vec::new(),
            meta_features: HashMap::new(),
            performance_predictors: HashMap::new(),
            strategy_recommender: StrategyRecommender {
                algorithm: RecommendationAlgorithm::Hybrid,
                recommendation_history: Vec::new(),
                user_preferences: UserPreferenceModel {
                    preference_weights: HashMap::new(),
                    interaction_history: Vec::new(),
                    implicit_preferences: HashMap::new(),
                    explicit_preferences: HashMap::new(),
                },
                context_awareness: true,
            },
        }
    }
}
/// Alert severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AlertSeverity {
    Info,
    Warning,
    Error,
    Critical,
    Emergency,
}
/// Types of action schedules
#[derive(Debug, Clone)]
pub enum ActionScheduleType {
    OneTime { execution_time: Instant },
    Recurring { interval: Duration },
    Cron { expression: String },
    EventDriven { event_type: String },
}
/// Tree learning configuration
#[derive(Debug, Clone)]
pub struct TreeLearningConfig {
    /// Maximum tree depth
    pub max_depth: usize,
    /// Minimum samples per leaf
    pub min_samples_leaf: usize,
    /// Minimum samples for split
    pub min_samples_split: usize,
    /// Pruning strategy
    pub pruning_strategy: PruningStrategy,
    /// Split criterion
    pub split_criterion: SplitCriterion,
}
/// Parameter adjustment types
#[derive(Debug, Clone)]
pub enum ParameterAdjustment {
    Absolute { value: f64 },
    Relative { factor: f64 },
    Increment { step: f64 },
    Adaptive { target: f64, learning_rate: f32 },
}
/// Types of feature extractors
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureExtractorType {
    Statistical,
    Temporal,
    Frequency,
    Wavelet,
    Polynomial,
    Interaction,
    Custom,
}
/// Adaptive knowledge base
#[derive(Debug)]
pub struct AdaptiveKnowledgeBase {
    /// Learned rules and patterns
    pub rules: HashMap<String, AdaptiveRule>,
    /// Experience database
    pub experiences: VecDeque<AdaptiveExperience>,
    /// Knowledge confidence scores
    pub confidence_scores: HashMap<String, f32>,
    /// Knowledge freshness tracking
    pub freshness_scores: HashMap<String, f32>,
    /// Knowledge graph connections
    pub knowledge_graph: KnowledgeGraph,
    /// Semantic search capabilities
    pub semantic_search: SemanticSearch,
}
/// Applicability conditions for strategies
#[derive(Debug, Clone)]
pub enum ApplicabilityCondition {
    SystemLoad { min: f32, max: f32 },
    MemoryPressure { threshold: f32 },
    TimeOfDay { start_hour: u8, end_hour: u8 },
    WorkloadType { workload_pattern: String },
    HistoricalSuccess { min_success_rate: f32 },
    ResourceAvailability { resource: String, min_available: f32 },
    UserPreference { preference: String, value: String },
    ExternalCondition { condition: String, operator: String, value: f64 },
}
/// Types of relationships in knowledge graph
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelationshipType {
    CausedBy,
    LeadsTo,
    SimilarTo,
    PartOf,
    DependsOn,
    Influences,
    Contradicts,
    Supports,
}
/// Adaptive control parameters
#[derive(Debug, Clone)]
pub struct AdaptiveControlParams {
    /// Adaptation sensitivity
    pub sensitivity: f32,
    /// Learning rate
    pub learning_rate: f32,
    /// Forgetting factor
    pub forgetting_factor: f32,
    /// Exploration rate
    pub exploration_rate: f32,
    /// Stability threshold
    pub stability_threshold: f32,
    /// Confidence threshold
    pub confidence_threshold: f32,
    /// Advanced control parameters
    pub advanced_params: AdvancedControlParams,
}
/// False positive tracker
#[derive(Debug, Clone)]
pub struct FalsePositiveTracker {
    /// False positive rate
    pub rate: f32,
    /// Recent false positives
    pub recent_fps: VecDeque<FalsePositive>,
    /// Patterns in false positives
    pub patterns: Vec<FPPattern>,
    /// Correction strategies
    pub correction_strategies: Vec<CorrectionStrategy>,
}
/// Knowledge graph for connecting concepts
#[derive(Debug, Clone)]
pub struct KnowledgeGraph {
    /// Graph nodes (concepts)
    pub nodes: HashMap<String, KnowledgeNode>,
    /// Graph edges (relationships)
    pub edges: Vec<KnowledgeEdge>,
    /// Graph metrics
    pub metrics: GraphMetrics,
}
/// Context change detection
#[derive(Debug, Clone)]
pub struct ContextChangeDetection {
    /// Detection algorithm
    pub algorithm: ContextChangeAlgorithm,
    /// Detection threshold
    pub threshold: f32,
    /// Change history
    pub change_history: Vec<ContextChange>,
    /// Detection sensitivity
    pub sensitivity: f32,
}
/// Emergency severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum EmergencySeverity {
    Low,
    Medium,
    High,
    Critical,
    Catastrophic,
}
/// Strategy complexity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum StrategyComplexity {
    Simple,
    Moderate,
    Complex,
    Advanced,
    Expert,
}
/// Strategy lifecycle management
#[derive(Debug, Clone)]
pub struct StrategyLifecycle {
    /// Creation timestamp
    pub created_at: Instant,
    /// Last updated timestamp
    pub last_updated: Instant,
    /// Last used timestamp
    pub last_used: Option<Instant>,
    /// Number of times used
    pub usage_count: usize,
    /// Current lifecycle stage
    pub stage: LifecycleStage,
    /// Retirement conditions
    pub retirement_conditions: Vec<RetirementCondition>,
}
/// Performance degradation severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DegradationSeverity {
    Minor,
    Moderate,
    Major,
    Severe,
    Critical,
}
/// Change detection result
#[derive(Debug, Clone)]
pub struct ChangeDetection {
    /// Detection timestamp
    pub timestamp: Instant,
    /// Detected change type
    pub change_type: ChangeType,
    /// Change magnitude
    pub magnitude: f32,
    /// Confidence in detection
    pub confidence: f32,
    /// Affected metrics
    pub affected_metrics: Vec<String>,
    /// Change direction
    pub direction: ChangeDirection,
}
/// Lifecycle stages for strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LifecycleStage {
    Experimental,
    Testing,
    Production,
    Mature,
    Deprecated,
    Retired,
}
/// Context requirements for conditions
#[derive(Debug, Clone)]
pub struct ContextRequirement {
    /// Context variable
    pub variable: String,
    /// Required value or range
    pub requirement: ContextRequirementType,
    /// Importance weight
    pub weight: f32,
}
/// False positive pattern
#[derive(Debug, Clone)]
pub struct FPPattern {
    /// Pattern description
    pub description: String,
    /// Pattern frequency
    pub frequency: f32,
    /// Associated conditions
    pub conditions: Vec<String>,
    /// Mitigation strategy
    pub mitigation: String,
}
/// Metadata for adaptation events
#[derive(Debug, Clone)]
pub struct AdaptationMetadata {
    /// System state before adaptation
    pub pre_state: SystemState,
    /// System state after adaptation
    pub post_state: SystemState,
    /// Resource usage during adaptation
    pub resource_usage: ResourceUsage,
    /// Execution duration
    pub duration: Duration,
    /// Confidence in the adaptation
    pub confidence: f32,
}
/// Feature selector
#[derive(Debug, Clone)]
pub struct FeatureSelector {
    /// Selector type
    pub selector_type: FeatureSelectorType,
    /// Selection criteria
    pub criteria: SelectionCriteria,
    /// Selected features
    pub selected_features: Vec<String>,
}
/// Context change record
#[derive(Debug, Clone)]
pub struct ContextChange {
    /// Change timestamp
    pub timestamp: Instant,
    /// Changed factors
    pub changed_factors: Vec<String>,
    /// Change magnitude
    pub magnitude: f32,
    /// Change type
    pub change_type: ContextChangeType,
    /// Impact assessment
    pub impact: ContextChangeImpact,
}
/// Prediction performance metrics
#[derive(Debug, Clone)]
pub struct PredictionPerformance {
    /// Mean absolute error
    pub mae: f32,
    /// Root mean square error
    pub rmse: f32,
    /// Mean absolute percentage error
    pub mape: f32,
    /// R-squared score
    pub r2_score: f32,
    /// Prediction confidence
    pub confidence: f32,
}
/// Context prediction model
#[derive(Debug, Clone)]
pub struct ContextPredictionModel {
    /// Model type
    pub model_type: ContextModelType,
    /// Model parameters
    pub parameters: HashMap<String, f64>,
    /// Training data size
    pub training_data_size: usize,
    /// Model accuracy
    pub accuracy: f32,
    /// Last update timestamp
    pub last_updated: Instant,
}
/// Adaptation outcomes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdaptationOutcome {
    Success,
    PartialSuccess,
    NoEffect,
    Failure,
    Rollback,
    Timeout,
    Cancelled,
}
/// Environmental context awareness
#[derive(Debug, Clone)]
pub struct EnvironmentContext {
    /// Current environment type
    pub environment_type: EnvironmentType,
    /// Environmental factors
    pub factors: HashMap<String, f64>,
    /// Context history
    pub context_history: VecDeque<ContextSnapshot>,
    /// Context change detection
    pub change_detection: ContextChangeDetection,
    /// Context prediction
    pub context_predictor: ContextPredictor,
}
impl EnvironmentContext {
    fn new() -> Self {
        Self {
            environment_type: EnvironmentType::Production,
            factors: HashMap::new(),
            context_history: VecDeque::new(),
            change_detection: ContextChangeDetection {
                algorithm: ContextChangeAlgorithm::Statistical,
                threshold: 0.1,
                change_history: Vec::new(),
                sensitivity: 0.5,
            },
            context_predictor: ContextPredictor {
                models: HashMap::new(),
                prediction_horizon: Duration::from_secs(1800),
                accuracy: 0.5,
                update_frequency: Duration::from_secs(300),
            },
        }
    }
}
/// System state snapshot with metadata
#[derive(Debug, Clone)]
pub struct SystemStateSnapshot {
    /// Snapshot timestamp
    pub timestamp: Instant,
    /// System state at the time
    pub state: SystemState,
    /// State quality score
    pub quality_score: f32,
    /// Stability indicator
    pub stability: f32,
    /// Change magnitude from previous state
    pub change_magnitude: f32,
    /// Anomaly score
    pub anomaly_score: f32,
}
/// Alert configuration
#[derive(Debug, Clone)]
pub struct AlertConfiguration {
    /// Alert message
    pub message: String,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Recipients
    pub recipients: Vec<String>,
    /// Alert channels
    pub channels: Vec<AlertChannel>,
    /// Suppression rules
    pub suppression_rules: Vec<SuppressionRule>,
}
/// Parameter bounds for adjustments
#[derive(Debug, Clone)]
pub struct ParameterBounds {
    pub min: f64,
    pub max: f64,
    pub step_size: Option<f64>,
}
/// Quality metrics for results
#[derive(Debug, Clone)]
pub struct ResultQualityMetrics {
    /// Accuracy of the result
    pub accuracy: f32,
    /// Precision of measurements
    pub precision: f32,
    /// Completeness of data
    pub completeness: f32,
    /// Timeliness of the result
    pub timeliness: f32,
}
/// Types of context requirements
#[derive(Debug, Clone)]
pub enum ContextRequirementType {
    Exact { value: String },
    Range { min: f64, max: f64 },
    OneOf { values: Vec<String> },
    Pattern { regex: String },
}
/// Resource pressure trend types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PressureTrend {
    Increasing,
    Decreasing,
    Stable,
    Volatile,
}
/// Knowledge graph node
#[derive(Debug, Clone)]
pub struct KnowledgeNode {
    /// Node identifier
    pub id: String,
    /// Node type
    pub node_type: NodeType,
    /// Node attributes
    pub attributes: HashMap<String, String>,
    /// Node importance score
    pub importance: f32,
    /// Creation timestamp
    pub created_at: Instant,
}
/// Rule priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RulePriority {
    Low,
    Medium,
    High,
    Critical,
    Emergency,
}
/// User interaction record
#[derive(Debug, Clone)]
pub struct UserInteraction {
    /// Interaction timestamp
    pub timestamp: Instant,
    /// Interaction type
    pub interaction_type: InteractionType,
    /// Strategy involved
    pub strategy: String,
    /// User feedback
    pub feedback: Option<f32>,
    /// Context of interaction
    pub context: HashMap<String, String>,
}
/// Anomaly detection algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnomalyDetectionAlgorithm {
    IsolationForest,
    OneClassSVM,
    LocalOutlierFactor,
    DBSCAN,
    ZScore,
    InterquartileRange,
    LSTM,
    Autoencoder,
    EllipticEnvelope,
    Custom,
}
/// Workload change types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkloadChangeType {
    Increase,
    Decrease,
    ShiftInPattern,
    NewWorkloadType,
    WorkloadRemoval,
}
/// Alert suppression rules
#[derive(Debug, Clone)]
pub struct SuppressionRule {
    /// Rule name
    pub name: String,
    /// Suppression duration
    pub duration: Duration,
    /// Suppression conditions
    pub conditions: Vec<String>,
    /// Max alerts per period
    pub max_alerts_per_period: usize,
}
/// Adaptive learning algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdaptiveLearningAlgorithm {
    OnlineGradientDescent,
    AdaptiveResonanceTheory,
    SelfOrganizingMap,
    NeuroEvolution,
    MetaLearning,
    TransferLearning,
    ReinforcementLearning,
    FederatedLearning,
    ContinualLearning,
    LifelongLearning,
}
/// User feedback sentiment
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeedbackSentiment {
    Positive,
    Negative,
    Neutral,
    Mixed,
}
/// Selection criteria for feature selection
#[derive(Debug, Clone)]
pub struct SelectionCriteria {
    /// Selection method
    pub method: String,
    /// Threshold values
    pub thresholds: HashMap<String, f64>,
    /// Maximum number of features
    pub max_features: Option<usize>,
    /// Minimum score required
    pub min_score: Option<f32>,
}
#[derive(Debug, Clone)]
pub struct OptimizationRecommendation {
    pub id: String,
    pub strategy_name: String,
    pub description: String,
    pub priority: f32,
    pub expected_improvement: f32,
    pub confidence: f32,
    pub resource_requirements: f32,
    pub estimated_duration: Duration,
    pub risk_assessment: f32,
}
/// Impact assessment for context changes
#[derive(Debug, Clone)]
pub struct ContextChangeImpact {
    /// Impact severity
    pub severity: ImpactSeverity,
    /// Affected components
    pub affected_components: Vec<String>,
    /// Recommended actions
    pub recommended_actions: Vec<String>,
    /// Adaptation urgency
    pub urgency: AdaptationUrgency,
}
/// Meta-knowledge base
#[derive(Debug, Clone)]
pub struct MetaKnowledgeBase {
    /// Learning strategies
    pub learning_strategies: HashMap<String, LearningStrategy>,
    /// Problem-solution mappings
    pub problem_solutions: HashMap<String, Vec<String>>,
    /// Performance patterns
    pub performance_patterns: Vec<PerformancePattern>,
    /// Success factors
    pub success_factors: HashMap<String, f32>,
}
/// Adaptation strategy definition
#[derive(Debug, Clone)]
pub struct AdaptationStrategy {
    /// Strategy unique identifier
    pub name: String,
    /// Strategy description
    pub description: String,
    /// Conditions that trigger this strategy
    pub triggers: Vec<AdaptationTrigger>,
    /// Actions performed by this strategy
    pub actions: Vec<AdaptationAction>,
    /// Strategy effectiveness score
    pub effectiveness: f32,
    /// Usage frequency statistics
    pub usage_frequency: f32,
    /// Success rate over time
    pub success_rate: f32,
    /// Strategy complexity level
    pub complexity: StrategyComplexity,
    /// Resource requirements
    pub resource_requirements: StrategyResourceRequirements,
    /// Applicability conditions
    pub applicability_conditions: Vec<ApplicabilityCondition>,
    /// Strategy lifecycle management
    pub lifecycle: StrategyLifecycle,
    /// Learning configuration
    pub learning_config: StrategyLearningConfig,
}
/// Correction strategy for false positives
#[derive(Debug, Clone)]
pub struct CorrectionStrategy {
    /// Strategy name
    pub name: String,
    /// Effectiveness score
    pub effectiveness: f32,
    /// Application conditions
    pub conditions: Vec<String>,
    /// Implementation method
    pub implementation: String,
}
/// Anomaly detector system
#[derive(Debug)]
pub struct AnomalyDetector {
    /// Detection algorithm
    pub algorithm: AnomalyDetectionAlgorithm,
    /// Algorithm parameters
    pub parameters: HashMap<String, f64>,
    /// Detection threshold
    pub threshold: f32,
    /// Historical anomalies
    pub history: Vec<AnomalyDetection>,
    /// Anomaly models
    pub models: HashMap<String, AnomalyModel>,
    /// False positive tracking
    pub false_positive_tracker: FalsePositiveTracker,
}
impl AnomalyDetector {
    fn new() -> Self {
        Self {
            algorithm: AnomalyDetectionAlgorithm::IsolationForest,
            parameters: HashMap::new(),
            threshold: 0.1,
            history: Vec::new(),
            models: HashMap::new(),
            false_positive_tracker: FalsePositiveTracker {
                rate: 0.0,
                recent_fps: VecDeque::new(),
                patterns: Vec::new(),
                correction_strategies: Vec::new(),
            },
        }
    }
}
/// Temporal patterns for time-based conditions
#[derive(Debug, Clone)]
pub struct TemporalPattern {
    /// Pattern type
    pub pattern_type: TemporalPatternType,
    /// Pattern parameters
    pub parameters: HashMap<String, f64>,
    /// Confidence threshold
    pub confidence_threshold: f32,
}
/// Learning configuration for strategies
#[derive(Debug, Clone)]
pub struct StrategyLearningConfig {
    /// Enable continuous learning
    pub enable_learning: bool,
    /// Learning rate for parameter adjustment
    pub learning_rate: f32,
    /// Minimum examples needed for learning
    pub min_examples: usize,
    /// Maximum memory for learning data
    pub max_learning_data: usize,
    /// Learning update frequency
    pub update_frequency: Duration,
    /// Feature extraction configuration
    pub feature_config: FeatureExtractionConfig,
}
/// Types of prediction models
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredictionModelType {
    LinearRegression,
    RandomForest,
    LSTM,
    GRU,
    Transformer,
    ARIMA,
    ProphetModel,
    EnsembleModel,
}
/// Meta-learning algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetaLearningAlgorithm {
    MAML,
    Reptile,
    ModelAgnostic,
    GradientBased,
    MetricBased,
    MemoryBased,
}
/// Context change algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextChangeAlgorithm {
    Statistical,
    MachineLearning,
    RuleBased,
    Hybrid,
}
/// Prediction model
#[derive(Debug, Clone)]
pub struct PredictionModel {
    /// Model type
    pub model_type: PredictionModelType,
    /// Model parameters
    pub parameters: HashMap<String, f64>,
    /// Training history
    pub training_history: Vec<TrainingSession>,
    /// Prediction performance
    pub performance: PredictionPerformance,
}
/// Graph metrics
#[derive(Debug, Clone)]
pub struct GraphMetrics {
    /// Number of nodes
    pub node_count: usize,
    /// Number of edges
    pub edge_count: usize,
    /// Average degree
    pub average_degree: f32,
    /// Clustering coefficient
    pub clustering_coefficient: f32,
    /// Graph density
    pub density: f32,
}
/// Performance patterns for meta-learning
#[derive(Debug, Clone)]
pub struct PerformancePattern {
    /// Pattern identifier
    pub id: String,
    /// Pattern description
    pub description: String,
    /// Pattern conditions
    pub conditions: Vec<String>,
    /// Expected outcomes
    pub expected_outcomes: HashMap<String, f32>,
    /// Confidence in pattern
    pub confidence: f32,
}
/// Alert types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlertType {
    Info,
    Warning,
    Error,
    Critical,
    Performance,
    Security,
}
/// Change detection algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChangeDetectionAlgorithm {
    CUSUM,
    EWMA,
    PageHinkley,
    ADWIN,
    KolmogorovSmirnov,
    MannWhitney,
    Statistical,
    MachineLearningBased,
}
/// Action execution modes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionExecutionMode {
    Sequential,
    Parallel,
    PriorityBased,
    ConditionalSequential,
}
/// Learning strategy definition
#[derive(Debug, Clone)]
pub struct LearningStrategy {
    /// Strategy name
    pub name: String,
    /// Strategy parameters
    pub parameters: HashMap<String, f64>,
    /// Applicable domains
    pub applicable_domains: Vec<String>,
    /// Expected performance
    pub expected_performance: f32,
    /// Resource requirements
    pub resource_requirements: HashMap<String, f32>,
}
/// Anomaly types for detection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnomalyType {
    Statistical,
    Behavioral,
    Performance,
    Security,
    Resource,
    Pattern,
}
/// Resource reallocation types
#[derive(Debug, Clone)]
pub enum ResourceReallocation {
    Increase { amount: f64, unit: String },
    Decrease { amount: f64, unit: String },
    Redistribute { source: String, target: String, amount: f64 },
    Optimize { optimization_goal: String },
}
/// Decision tree node
#[derive(Debug, Clone)]
pub struct DecisionNode {
    /// Node identifier
    pub id: String,
    /// Decision condition
    pub condition: Option<DecisionCondition>,
    /// Left child (true branch)
    pub left: Option<Box<DecisionNode>>,
    /// Right child (false branch)
    pub right: Option<Box<DecisionNode>>,
    /// Action to take (for leaf nodes)
    pub action: Option<DecisionAction>,
    /// Node statistics
    pub stats: NodeStatistics,
}
/// Alternative recommendation
#[derive(Debug, Clone)]
pub struct AlternativeRecommendation {
    /// Alternative strategy
    pub strategy: String,
    /// Confidence in alternative
    pub confidence: f32,
    /// Trade-offs compared to primary recommendation
    pub tradeoffs: Vec<String>,
}
/// Emergency response types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmergencyResponseType {
    SafeShutdown,
    ResourceIsolation,
    Rollback,
    ManualIntervention,
    AutoRecovery,
}
/// Parameter update types
#[derive(Debug, Clone)]
pub enum ParameterUpdate {
    SetValue { value: f64 },
    IncrementBy { amount: f64 },
    MultiplyBy { factor: f64 },
    AdaptiveUpdate { target: f64, learning_rate: f32, bounds: Option<(f64, f64)> },
}
/// Feature extraction configuration
#[derive(Debug, Clone)]
pub struct FeatureExtractionConfig {
    /// Enabled feature types
    pub enabled_features: Vec<FeatureType>,
    /// Feature window size
    pub window_size: usize,
    /// Aggregation methods
    pub aggregation_methods: Vec<AggregationMethod>,
    /// Normalization strategy
    pub normalization: NormalizationStrategy,
}
/// Recommendation algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecommendationAlgorithm {
    CollaborativeFiltering,
    ContentBased,
    Hybrid,
    ReinforcementLearning,
    KnowledgeBased,
}
/// Types of temporal patterns
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemporalPatternType {
    Periodic,
    Seasonal,
    Burst,
    Gradual,
    Sudden,
    Recurring,
}
/// Execution modes for actions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionMode {
    Synchronous,
    Asynchronous,
    Queued,
    Background,
}
/// State quality indicators
#[derive(Debug, Clone)]
pub struct StateQualityIndicators {
    /// Data completeness (0.0 to 1.0)
    pub completeness: f32,
    /// Data freshness (0.0 to 1.0)
    pub freshness: f32,
    /// Measurement accuracy (0.0 to 1.0)
    pub accuracy: f32,
    /// State stability (0.0 to 1.0)
    pub stability: f32,
}
/// Types of knowledge nodes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeType {
    Concept,
    Strategy,
    Parameter,
    Rule,
    Experience,
    Pattern,
}
/// Controller state tracking
#[derive(Debug, Clone)]
pub struct ControllerState {
    /// Current operational mode
    pub mode: ControllerMode,
    /// Active strategies
    pub active_strategies: Vec<String>,
    /// Current performance level
    pub performance_level: PerformanceLevel,
    /// Resource utilization
    pub resource_utilization: f32,
    /// Last adaptation timestamp
    pub last_adaptation: Option<Instant>,
    /// Adaptation frequency
    pub adaptation_frequency: f32,
    /// State confidence
    pub confidence: f32,
}
impl ControllerState {
    fn new() -> Self {
        Self {
            mode: ControllerMode::Learning,
            active_strategies: Vec::new(),
            performance_level: PerformanceLevel::Average,
            resource_utilization: 0.5,
            last_adaptation: None,
            adaptation_frequency: 0.0,
            confidence: 0.5,
        }
    }
}
/// Tree pruning strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PruningStrategy {
    None,
    PrePruning,
    PostPruning,
    MinimalCostComplexity,
}
/// Configuration change types
#[derive(Debug, Clone)]
pub enum ConfigurationChange {
    Update { value: String },
    Toggle { enable: bool },
    Reset { to_default: bool },
    Conditional { condition: String, true_value: String, false_value: String },
}
/// Exploration strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExplorationStrategy {
    EpsilonGreedy,
    Boltzmann,
    UpperConfidenceBound,
    Thompson,
    Adaptive,
}
/// Learning scope for rate adjustments
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LearningScope {
    Global,
    Strategy,
    Parameter,
    Component,
}
/// Feature extractor
#[derive(Debug, Clone)]
pub struct FeatureExtractor {
    /// Extractor type
    pub extractor_type: FeatureExtractorType,
    /// Extraction parameters
    pub parameters: HashMap<String, f64>,
    /// Output feature names
    pub output_features: Vec<String>,
}
/// Pattern shift directions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShiftDirection {
    Upward,
    Downward,
    Lateral,
    Cyclical,
    Chaotic,
}
/// Resource requirements for strategies
#[derive(Debug, Clone)]
pub struct StrategyResourceRequirements {
    /// CPU computation required
    pub cpu_cost: f32,
    /// Memory usage requirement
    pub memory_cost: usize,
    /// Time to execute
    pub execution_time: Duration,
    /// Risk level associated
    pub risk_level: RiskLevel,
    /// Required data availability
    pub data_requirements: Vec<String>,
}
/// Transfer learning configuration
#[derive(Debug, Clone)]
pub struct TransferLearningConfig {
    /// Enable transfer learning
    pub enabled: bool,
    /// Source domains
    pub source_domains: Vec<String>,
    /// Target domain
    pub target_domain: String,
    /// Transfer methods
    pub transfer_methods: Vec<TransferMethod>,
    /// Similarity threshold
    pub similarity_threshold: f32,
}
/// Knowledge graph edge
#[derive(Debug, Clone)]
pub struct KnowledgeEdge {
    /// Source node
    pub source: String,
    /// Target node
    pub target: String,
    /// Relationship type
    pub relationship: RelationshipType,
    /// Edge weight
    pub weight: f32,
    /// Edge attributes
    pub attributes: HashMap<String, String>,
}
/// Adaptation trigger conditions
#[derive(Debug, Clone)]
pub enum AdaptationTrigger {
    /// Performance degradation detected
    PerformanceDegradation {
        threshold: f32,
        duration: Duration,
        severity: DegradationSeverity,
    },
    /// Resource pressure detected
    ResourcePressure { resource: String, threshold: f32, trend: PressureTrend },
    /// Workload change detected
    WorkloadChange { change_magnitude: f32, change_type: WorkloadChangeType },
    /// Pattern shift in system behavior
    PatternShift { pattern: String, confidence: f32, shift_direction: ShiftDirection },
    /// Error rate increase
    ErrorRateIncrease { threshold: f32, error_type: String },
    /// User feedback received
    UserFeedback { feedback_type: String, sentiment: FeedbackSentiment },
    /// Anomaly detection
    AnomalyDetected { anomaly_type: AnomalyType, severity: f32, confidence: f32 },
    /// External event
    ExternalEvent { event_type: String, metadata: HashMap<String, String> },
    /// Scheduled trigger
    ScheduledTrigger { schedule: TriggerSchedule, next_execution: Instant },
    /// Threshold-based trigger
    ThresholdTrigger {
        metric: String,
        operator: ComparisonOperator,
        threshold: f64,
        consecutive_violations: usize,
    },
}
/// False positive record
#[derive(Debug, Clone)]
pub struct FalsePositive {
    /// Timestamp
    pub timestamp: Instant,
    /// Original detection
    pub detection: AnomalyDetection,
    /// Confirmation method
    pub confirmation_method: String,
    /// Root cause
    pub root_cause: Option<String>,
}
/// Adaptation actions that can be performed
#[derive(Debug, Clone)]
pub enum AdaptationAction {
    /// Adjust a parameter value
    ParameterAdjustment {
        parameter: String,
        adjustment: ParameterAdjustment,
        bounds: Option<ParameterBounds>,
    },
    /// Switch optimization strategies
    StrategySwitch {
        from_strategy: String,
        to_strategy: String,
        transition_mode: TransitionMode,
    },
    /// Reallocate system resources
    ResourceReallocation { resource: String, reallocation: ResourceReallocation },
    /// Change configuration settings
    ConfigurationChange { setting: String, change: ConfigurationChange },
    /// Adjust learning parameters
    LearningRateAdjustment { new_rate: f32, scope: LearningScope },
    /// Adjust exploration parameters
    ExplorationAdjustment { new_rate: f32, exploration_strategy: ExplorationStrategy },
    /// Trigger emergency response
    EmergencyResponse {
        response_type: EmergencyResponseType,
        severity: EmergencySeverity,
    },
    /// Generate alert or notification
    AlertGeneration { alert_type: AlertType, message: String, recipients: Vec<String> },
    /// Execute custom action
    CustomAction {
        action_name: String,
        parameters: HashMap<String, f64>,
        execution_mode: ExecutionMode,
    },
    /// Rollback to previous state
    Rollback { target_state: String, rollback_scope: RollbackScope },
}
/// Strategy recommender system
#[derive(Debug, Clone)]
pub struct StrategyRecommender {
    /// Recommendation algorithm
    pub algorithm: RecommendationAlgorithm,
    /// Historical recommendations
    pub recommendation_history: Vec<StrategyRecommendation>,
    /// User preference model
    pub user_preferences: UserPreferenceModel,
    /// Context-aware recommendations
    pub context_awareness: bool,
}
/// Repeat configuration for scheduled actions
#[derive(Debug, Clone)]
pub struct RepeatConfig {
    /// Number of repetitions (-1 for infinite)
    pub repetitions: i32,
    /// Interval between repetitions
    pub interval: Duration,
    /// Maximum duration for repetitions
    pub max_duration: Option<Duration>,
}
/// Adaptation performance metrics
#[derive(Debug, Clone)]
pub struct AdaptationPerformanceMetrics {
    /// Adaptation success rate
    pub success_rate: f32,
    /// Average adaptation time
    pub avg_adaptation_time: Duration,
    /// Performance improvement rate
    pub improvement_rate: f32,
    /// Learning velocity
    pub learning_velocity: f32,
    /// Stability score
    pub stability_score: f32,
    /// Resource efficiency
    pub resource_efficiency: f32,
    /// User satisfaction score
    pub user_satisfaction: f32,
}
/// Query processor for search
#[derive(Debug, Clone)]
pub struct QueryProcessor {
    /// Processing pipeline
    pub pipeline: Vec<QueryProcessingStep>,
    /// Query expansion rules
    pub expansion_rules: Vec<ExpansionRule>,
    /// Result ranking algorithm
    pub ranking_algorithm: RankingAlgorithm,
}
/// Types of features for learning
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureType {
    SystemMetrics,
    PerformanceIndicators,
    ResourceUtilization,
    WorkloadCharacteristics,
    TemporalPatterns,
    UserBehavior,
    Environmental,
    Historical,
}
/// Node statistics for decision tree
#[derive(Debug, Clone)]
pub struct NodeStatistics {
    /// Number of samples that reached this node
    pub sample_count: usize,
    /// Accuracy of decisions made at this node
    pub accuracy: f32,
    /// Information gain at this node
    pub information_gain: f32,
    /// Node impurity
    pub impurity: f32,
}
/// Strategy attempt record
#[derive(Debug, Clone)]
pub struct StrategyAttempt {
    /// Strategy name
    pub strategy: String,
    /// Parameters used
    pub parameters: HashMap<String, f64>,
    /// Performance achieved
    pub performance: f32,
    /// Time to achieve performance
    pub time_to_performance: Duration,
    /// Resource usage
    pub resource_usage: f32,
}
/// Strategy recommendation
#[derive(Debug, Clone)]
pub struct StrategyRecommendation {
    /// Recommended strategy
    pub strategy: String,
    /// Recommendation confidence
    pub confidence: f32,
    /// Expected performance
    pub expected_performance: f32,
    /// Recommendation reasoning
    pub reasoning: Vec<String>,
    /// Alternative recommendations
    pub alternatives: Vec<AlternativeRecommendation>,
}
/// Monitoring configuration
#[derive(Debug, Clone)]
pub struct MonitoringConfig {
    /// Monitoring frequency
    pub frequency: Duration,
    /// Data collection interval
    pub collection_interval: Duration,
    /// Retention period for data
    pub retention_period: Duration,
    /// Enable detailed logging
    pub detailed_logging: bool,
    /// Alert thresholds
    pub alert_thresholds: HashMap<String, f64>,
}
/// Strategy activation modes
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActivationMode {
    Immediate,
    Gradual { transition_duration: Duration },
    Scheduled { schedule_time: Duration },
    Conditional { condition: String },
}
/// Conditions for strategy retirement
#[derive(Debug, Clone)]
pub enum RetirementCondition {
    LowSuccessRate { threshold: f32, duration: Duration },
    LowUsage { max_uses_per_period: usize, period: Duration },
    BetterAlternativeAvailable { alternative_name: String },
    SystemEvolution { incompatible_changes: Vec<String> },
    ManualRetirement { reason: String },
}
/// Feedback on experiences
#[derive(Debug, Clone)]
pub struct ExperienceFeedback {
    /// Feedback source
    pub source: FeedbackSource,
    /// Feedback rating
    pub rating: f32,
    /// Feedback comments
    pub comments: Option<String>,
    /// Feedback timestamp
    pub timestamp: Instant,
}
/// Baseline statistical model
#[derive(Debug, Clone)]
pub struct BaselineModel {
    /// Model type
    pub model_type: BaselineModelType,
    /// Model parameters
    pub parameters: HashMap<String, f64>,
    /// Training data size
    pub training_size: usize,
    /// Model accuracy
    pub accuracy: f32,
    /// Last update timestamp
    pub last_updated: Instant,
}
/// Rule condition types
#[derive(Debug, Clone)]
pub enum RuleCondition {
    StateMatch { state_pattern: String, tolerance: f32 },
    ThresholdExceeded { metric: String, threshold: f64, direction: ThresholdDirection },
    PatternDetected { pattern: String, confidence: f32, window_size: usize },
    TrendObserved { trend: TrendType, duration: Duration, significance: f32 },
    CombinationCondition { conditions: Vec<RuleCondition>, logic: LogicalOperator },
    TemporalCondition { temporal_pattern: TemporalPattern },
    ContextualCondition { context_requirements: Vec<ContextRequirement> },
}
/// Context predictor
#[derive(Debug, Clone)]
pub struct ContextPredictor {
    /// Prediction models
    pub models: HashMap<String, ContextPredictionModel>,
    /// Prediction horizon
    pub prediction_horizon: Duration,
    /// Prediction accuracy
    pub accuracy: f32,
    /// Update frequency
    pub update_frequency: Duration,
}
/// System state monitor for environmental awareness
#[derive(Debug)]
pub struct SystemStateMonitor {
    /// Current system state
    current_state: SystemState,
    /// Historical state snapshots
    state_history: VecDeque<SystemStateSnapshot>,
    /// Change detection algorithms
    change_detector: ChangeDetector,
    /// Anomaly detection system
    anomaly_detector: AnomalyDetector,
    /// State prediction model
    state_predictor: StatePredictor,
    /// Monitoring configuration
    monitoring_config: MonitoringConfig,
}
impl SystemStateMonitor {
    fn new() -> Self {
        Self {
            current_state: SystemState::default(),
            state_history: VecDeque::new(),
            change_detector: ChangeDetector::new(),
            anomaly_detector: AnomalyDetector::new(),
            state_predictor: StatePredictor::new(),
            monitoring_config: MonitoringConfig::default(),
        }
    }
}
/// Resource usage tracking
#[derive(Debug, Clone)]
pub struct ResourceUsage {
    /// CPU utilization
    pub cpu_utilization: f32,
    /// Memory usage in MB
    pub memory_usage_mb: usize,
    /// GPU utilization
    pub gpu_utilization: f32,
    /// Network bandwidth used
    pub network_mbps: f32,
    /// Disk I/O operations
    pub disk_iops: f32,
}
/// Comparison operators for threshold triggers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComparisonOperator {
    GreaterThan,
    LessThan,
    GreaterThanOrEqual,
    LessThanOrEqual,
    Equal,
    NotEqual,
}
/// Feature engineering pipeline
#[derive(Debug, Clone)]
pub struct FeatureEngineeringPipeline {
    /// Feature extractors
    pub extractors: Vec<FeatureExtractor>,
    /// Feature transformers
    pub transformers: Vec<FeatureTransformer>,
    /// Feature selectors
    pub selectors: Vec<FeatureSelector>,
    /// Pipeline configuration
    pub config: PipelineConfig,
}
/// Adaptive rule representation
#[derive(Debug, Clone)]
pub struct AdaptiveRule {
    /// Rule unique identifier
    pub id: String,
    /// Rule condition
    pub condition: RuleCondition,
    /// Rule action
    pub action: RuleAction,
    /// Rule confidence score
    pub confidence: f32,
    /// Number of times rule was used
    pub usage_count: usize,
    /// Success rate of the rule
    pub success_rate: f32,
    /// Rule creation timestamp
    pub created_at: Instant,
    /// Last successful application
    pub last_success: Option<Instant>,
    /// Rule priority
    pub priority: RulePriority,
    /// Rule validity conditions
    pub validity_conditions: Vec<ValidityCondition>,
}
/// Types of user interactions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InteractionType {
    StrategySelection,
    ParameterAdjustment,
    FeedbackProvision,
    Override,
    Approval,
    Rejection,
}
/// Reasons for model updates
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpdateReason {
    PerformanceDegradation,
    DataDrift,
    ConceptDrift,
    ScheduledUpdate,
    ManualUpdate,
    AdaptiveTrigger,
}
/// Online learning configuration
#[derive(Debug, Clone)]
pub struct OnlineLearningConfig {
    /// Enable online learning
    pub enabled: bool,
    /// Learning rate
    pub learning_rate: f32,
    /// Batch size for updates
    pub batch_size: usize,
    /// Update frequency
    pub update_frequency: Duration,
    /// Forgetting factor for old data
    pub forgetting_factor: f32,
    /// Minimum examples before learning
    pub min_examples: usize,
    /// Maximum examples to retain
    pub max_examples: usize,
    /// Adaptive learning rate configuration
    pub adaptive_lr_config: Option<AdaptiveLearningRateConfig>,
}
/// Feature aggregation methods
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregationMethod {
    Mean,
    Median,
    StdDev,
    Min,
    Max,
    Percentile(u8),
    Trend,
    Frequency,
}
/// Feature transformer
#[derive(Debug, Clone)]
pub struct FeatureTransformer {
    /// Transformer type
    pub transformer_type: FeatureTransformerType,
    /// Transformation parameters
    pub parameters: HashMap<String, f64>,
    /// Input features
    pub input_features: Vec<String>,
    /// Output features
    pub output_features: Vec<String>,
}
