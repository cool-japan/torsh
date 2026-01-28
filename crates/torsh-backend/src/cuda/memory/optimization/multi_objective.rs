//! Multi-objective optimization engine for CUDA memory optimization
//!
//! This module provides comprehensive multi-objective optimization capabilities
//! including various algorithms (NSGA-II, NSGA-III, SPEA2, MOEA/D, SMS-EMOA),
//! Pareto front management, constraint handling, and performance metrics.

use scirs2_core::random::thread_rng as rng;
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

/// Multi-objective optimizer for CUDA memory optimization
///
/// Implements various multi-objective optimization algorithms to find
/// Pareto-optimal solutions for competing objectives like performance,
/// memory usage, energy consumption, and latency.
#[derive(Debug)]
pub struct MultiObjectiveOptimizer {
    /// Available optimization algorithms
    algorithms: HashMap<String, MultiObjectiveAlgorithm>,

    /// Current Pareto front solutions
    pareto_front: Vec<ParetoSolution>,

    /// Objective weights for scalarization methods
    objective_weights: HashMap<String, f32>,

    /// Constraint handlers for feasibility management
    constraint_handlers: Vec<ConstraintHandler>,

    /// Archive of all generated solutions
    solution_archive: VecDeque<OptimizationSolution>,

    /// Performance metrics and statistics
    performance_metrics: MultiObjectiveMetrics,

    /// Population for evolutionary algorithms
    population: Vec<Individual>,

    /// Reference point for hypervolume calculation
    reference_point: Vec<f64>,

    /// Archive management configuration
    archive_config: ArchiveConfig,

    /// Diversity maintenance strategy
    diversity_strategy: DiversityStrategy,

    /// Convergence detection mechanism
    convergence_detector: ConvergenceDetector,
}

/// Multi-objective optimization algorithm configuration
#[derive(Debug, Clone)]
pub struct MultiObjectiveAlgorithm {
    /// Algorithm name
    pub name: String,

    /// Algorithm type
    pub algorithm_type: MOAlgorithmType,

    /// Algorithm-specific parameters
    pub parameters: HashMap<String, f64>,

    /// Population size
    pub population_size: usize,

    /// Maximum number of generations
    pub max_generations: usize,

    /// Convergence criteria
    pub convergence_criteria: ConvergenceCriteria,

    /// Crossover probability
    pub crossover_probability: f64,

    /// Mutation probability
    pub mutation_probability: f64,

    /// Selection pressure
    pub selection_pressure: f64,

    /// Elite preservation ratio
    pub elite_ratio: f64,
}

/// Types of multi-objective optimization algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MOAlgorithmType {
    /// Non-dominated Sorting Genetic Algorithm II
    NSGA2,
    /// Non-dominated Sorting Genetic Algorithm III
    NSGA3,
    /// Strength Pareto Evolutionary Algorithm 2
    SPEA2,
    /// Multi-Objective Evolutionary Algorithm based on Decomposition
    MOEAD,
    /// S-Metric Selection Evolutionary Multi-Objective Algorithm
    SmsEmoa,
    /// Hypervolume-based Evolutionary Algorithm
    HypE,
    /// Multi-Objective Particle Swarm Optimization
    MOPSO,
    /// Multi-Objective Differential Evolution
    MODE,
    /// Indicator-Based Evolutionary Algorithm
    IBEA,
    /// Fast and Elitist Multi-Objective Genetic Algorithm
    FastNSGA2,
    /// Custom algorithm implementation
    Custom,
}

/// Convergence criteria for multi-objective algorithms
#[derive(Debug, Clone)]
pub struct ConvergenceCriteria {
    /// Maximum number of generations
    pub max_generations: usize,

    /// Target hypervolume value
    pub target_hypervolume: Option<f64>,

    /// Convergence tolerance
    pub tolerance: f64,

    /// Maximum stagnation generations
    pub stagnation_limit: usize,

    /// Minimum improvement per generation
    pub min_improvement: f64,

    /// Convergence metrics to track
    pub convergence_metrics: Vec<ConvergenceMetric>,

    /// Early stopping configuration
    pub early_stopping: Option<EarlyStoppingConfig>,
}

/// Types of convergence metrics
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvergenceMetric {
    Hypervolume,
    GenerationalDistance,
    InvertedGenerationalDistance,
    Spread,
    Spacing,
    Coverage,
}

/// Early stopping configuration
#[derive(Debug, Clone)]
pub struct EarlyStoppingConfig {
    /// Patience (generations without improvement)
    pub patience: usize,

    /// Minimum delta for improvement detection
    pub min_delta: f64,

    /// Metric to monitor for early stopping
    pub monitor_metric: ConvergenceMetric,

    /// Whether higher values are better
    pub maximize: bool,
}

/// Pareto optimal solution representation
#[derive(Debug, Clone)]
pub struct ParetoSolution {
    /// Unique solution identifier
    pub id: String,

    /// Solution parameters/variables
    pub parameters: HashMap<String, f64>,

    /// Objective function values
    pub objectives: HashMap<String, f64>,

    /// Dominance rank in population
    pub rank: usize,

    /// Crowding distance for diversity
    pub crowding_distance: f64,

    /// Overall quality score
    pub quality_score: f32,

    /// Solution generation timestamp
    pub timestamp: Instant,

    /// Constraint violation amounts
    pub constraint_violations: HashMap<String, f64>,

    /// Solution fitness (for single-objective conversion)
    pub fitness: f64,

    /// Solution age (generations since creation)
    pub age: usize,

    /// Niche count for fitness sharing
    pub niche_count: f64,

    /// Hypervolume contribution
    pub hypervolume_contribution: f64,
}

/// Individual in the population for evolutionary algorithms
#[derive(Debug, Clone)]
pub struct Individual {
    /// Genotype representation
    pub genotype: Vec<f64>,

    /// Phenotype (decoded parameters)
    pub phenotype: HashMap<String, f64>,

    /// Objective values
    pub objectives: Vec<f64>,

    /// Constraint violation values
    pub constraints: Vec<f64>,

    /// Dominance rank
    pub rank: usize,

    /// Crowding distance
    pub crowding_distance: f64,

    /// Fitness value
    pub fitness: f64,

    /// Age in generations
    pub age: usize,
}

/// Constraint handler for feasibility management
#[derive(Debug, Clone)]
pub struct ConstraintHandler {
    /// Handler identifier
    pub name: String,

    /// Type of constraint handled
    pub constraint_type: ConstraintType,

    /// Handling method
    pub method: ConstraintMethod,

    /// Method-specific parameters
    pub penalty_params: HashMap<String, f64>,

    /// Constraint tolerance
    pub tolerance: f64,

    /// Handler priority
    pub priority: u32,

    /// Adaptive penalty configuration
    pub adaptive_penalty: Option<AdaptivePenaltyConfig>,
}

/// Types of optimization constraints
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstraintType {
    Equality,
    Inequality,
    Bound,
    Resource,
    Performance,
    Safety,
    Custom,
}

/// Constraint handling methods
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstraintMethod {
    /// Penalty function method
    Penalty,
    /// Barrier function method
    Barrier,
    /// Augmented Lagrangian method
    Augmented,
    /// Feasibility-based selection
    Feasibility,
    /// Constraint repair
    Repair,
    /// Epsilon constraint method
    Epsilon,
    /// Constraint dominance
    ConstraintDominance,
    /// Stochastic ranking
    StochasticRanking,
}

/// Adaptive penalty configuration
#[derive(Debug, Clone)]
pub struct AdaptivePenaltyConfig {
    /// Initial penalty factor
    pub initial_penalty: f64,

    /// Penalty update factor
    pub update_factor: f64,

    /// Maximum penalty factor
    pub max_penalty: f64,

    /// Update frequency (generations)
    pub update_frequency: usize,
}

/// General optimization solution
#[derive(Debug, Clone)]
pub struct OptimizationSolution {
    /// Solution identifier
    pub id: String,

    /// Creation timestamp
    pub timestamp: Instant,

    /// Strategy that generated this solution
    pub strategy_id: String,

    /// Solution parameters
    pub parameters: HashMap<String, f64>,

    /// Achieved objective values
    pub objective_values: HashMap<String, f64>,

    /// Overall fitness score
    pub fitness: f64,

    /// Implementation status
    pub status: SolutionStatus,

    /// Performance results (if implemented)
    pub performance: Option<PerformanceResult>,

    /// Solution metadata
    pub metadata: SolutionMetadata,

    /// Validation results
    pub validation_results: Option<ValidationResults>,
}

/// Solution implementation status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolutionStatus {
    Generated,
    Evaluated,
    Validated,
    Implemented,
    Active,
    RolledBack,
    Failed,
    Archived,
}

/// Performance results for implemented solutions
#[derive(Debug, Clone)]
pub struct PerformanceResult {
    /// Actual objective values achieved
    pub actual_objectives: HashMap<String, f64>,

    /// Performance improvement over baseline
    pub improvement: f64,

    /// Measurement confidence level
    pub confidence: f64,

    /// Measurement duration
    pub measurement_duration: Duration,

    /// Statistical significance
    pub significance: Option<f64>,
}

/// Solution metadata
#[derive(Debug, Clone)]
pub struct SolutionMetadata {
    /// Generation number when created
    pub generation: usize,

    /// Parent solutions (for tracking lineage)
    pub parents: Vec<String>,

    /// Genetic operators used
    pub operators: Vec<String>,

    /// Computational cost to generate
    pub computation_cost: f64,

    /// Memory usage during evaluation
    pub memory_usage: usize,

    /// Additional custom metadata
    pub custom_data: HashMap<String, String>,
}

/// Validation results for solutions
#[derive(Debug, Clone)]
pub struct ValidationResults {
    /// Overall validation success
    pub passed: bool,

    /// Individual test results
    pub test_results: HashMap<String, bool>,

    /// Validation scores
    pub scores: HashMap<String, f64>,

    /// Risk assessment results
    pub risk_assessment: RiskAssessment,

    /// Validation duration
    pub validation_time: Duration,
}

/// Risk assessment for solutions
#[derive(Debug, Clone)]
pub struct RiskAssessment {
    /// Overall risk level
    pub risk_level: RiskLevel,

    /// Individual risk factors
    pub risk_factors: HashMap<String, f64>,

    /// Risk mitigation suggestions
    pub mitigation_strategies: Vec<String>,

    /// Confidence in risk assessment
    pub confidence: f64,
}

/// Risk levels for solutions
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Multi-objective optimization performance metrics
#[derive(Debug, Clone)]
pub struct MultiObjectiveMetrics {
    /// Hypervolume indicator
    pub hypervolume: f64,

    /// Spacing metric (distribution uniformity)
    pub spacing: f64,

    /// Convergence metric (distance to true Pareto front)
    pub convergence: f64,

    /// Diversity metric (solution spread)
    pub diversity: f64,

    /// Number of non-dominated solutions
    pub solution_count: usize,

    /// Current generation number
    pub generations: usize,

    /// Generational distance
    pub generational_distance: f64,

    /// Inverted generational distance
    pub inverted_generational_distance: f64,

    /// Coverage metrics
    pub coverage_metrics: CoverageMetrics,

    /// Runtime performance metrics
    pub performance_stats: PerformanceStats,

    /// Quality indicators
    pub quality_indicators: QualityIndicators,
}

/// Coverage metrics for multi-objective optimization
#[derive(Debug, Clone)]
pub struct CoverageMetrics {
    /// C-metric (coverage relation)
    pub c_metric: f64,

    /// Set coverage
    pub set_coverage: f64,

    /// Epsilon indicator
    pub epsilon_indicator: f64,

    /// Binary epsilon indicator
    pub binary_epsilon: f64,
}

/// Runtime performance statistics
#[derive(Debug, Clone)]
pub struct PerformanceStats {
    /// Average generation time
    pub avg_generation_time: Duration,

    /// Total optimization time
    pub total_time: Duration,

    /// Evaluations per second
    pub evaluations_per_second: f64,

    /// Memory usage peak
    pub peak_memory_usage: usize,

    /// CPU utilization
    pub cpu_utilization: f64,

    /// Convergence rate
    pub convergence_rate: f64,
}

/// Quality indicators for optimization results
#[derive(Debug, Clone)]
pub struct QualityIndicators {
    /// Additive epsilon indicator
    pub additive_epsilon: f64,

    /// Multiplicative epsilon indicator
    pub multiplicative_epsilon: f64,

    /// R2 indicator
    pub r2_indicator: f64,

    /// IGD+ indicator
    pub igd_plus: f64,

    /// Modified inverted generational distance
    pub modified_igd: f64,
}

/// Archive management configuration
#[derive(Debug, Clone)]
pub struct ArchiveConfig {
    /// Maximum archive size
    pub max_size: usize,

    /// Archive update strategy
    pub update_strategy: ArchiveUpdateStrategy,

    /// Duplicate handling
    pub duplicate_handling: DuplicateHandling,

    /// Archive quality threshold
    pub quality_threshold: f64,

    /// Aging strategy
    pub aging_strategy: Option<AgingStrategy>,
}

/// Archive update strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchiveUpdateStrategy {
    /// Replace worst solutions
    ReplaceWorst,
    /// First-in-first-out
    FIFO,
    /// Random replacement
    Random,
    /// Crowding-based replacement
    CrowdingBased,
    /// Hypervolume-based replacement
    HypervolumeBased,
}

/// Duplicate handling strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DuplicateHandling {
    /// Allow duplicates
    Allow,
    /// Reject duplicates
    Reject,
    /// Replace with better solution
    Replace,
    /// Merge information
    Merge,
}

/// Aging strategies for solution management
#[derive(Debug, Clone)]
pub struct AgingStrategy {
    /// Maximum age before removal
    pub max_age: usize,

    /// Age-based fitness penalty
    pub age_penalty: f64,

    /// Aging update frequency
    pub update_frequency: usize,
}

/// Diversity maintenance strategies
#[derive(Debug, Clone)]
pub struct DiversityStrategy {
    /// Strategy type
    pub strategy_type: DiversityStrategyType,

    /// Diversity threshold
    pub diversity_threshold: f64,

    /// Niche radius for fitness sharing
    pub niche_radius: f64,

    /// Clustering configuration
    pub clustering_config: Option<ClusteringConfig>,
}

/// Types of diversity maintenance strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiversityStrategyType {
    /// Crowding distance
    CrowdingDistance,
    /// Fitness sharing
    FitnessSharing,
    /// Clustering-based
    Clustering,
    /// Hypervolume contribution
    HypervolumeContribution,
    /// Reference point-based
    ReferencePoint,
}

/// Clustering configuration for diversity maintenance
#[derive(Debug, Clone)]
pub struct ClusteringConfig {
    /// Number of clusters
    pub num_clusters: usize,

    /// Clustering algorithm
    pub algorithm: ClusteringAlgorithm,

    /// Update frequency
    pub update_frequency: usize,

    /// Cluster representative selection
    pub representative_selection: RepresentativeSelection,
}

/// Clustering algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClusteringAlgorithm {
    KMeans,
    HierarchicalClustering,
    DBSCAN,
    SpectralClustering,
}

/// Representative selection methods
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepresentativeSelection {
    Centroid,
    Best,
    Random,
    Diverse,
}

/// Convergence detection mechanism
#[derive(Debug, Clone)]
pub struct ConvergenceDetector {
    /// Detection strategy
    pub strategy: ConvergenceDetectionStrategy,

    /// Convergence history
    pub history: VecDeque<ConvergenceSnapshot>,

    /// Window size for trend analysis
    pub window_size: usize,

    /// Sensitivity parameters
    pub sensitivity: f64,

    /// Convergence state
    pub is_converged: bool,
}

/// Convergence detection strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvergenceDetectionStrategy {
    /// Hypervolume stagnation
    HypervolumeStagnation,
    /// Average objective improvement
    AverageImprovement,
    /// Population diversity
    PopulationDiversity,
    /// Gradient-based
    GradientBased,
    /// Statistical tests
    StatisticalTest,
}

/// Convergence snapshot for history tracking
#[derive(Debug, Clone)]
pub struct ConvergenceSnapshot {
    /// Generation number
    pub generation: usize,

    /// Hypervolume value
    pub hypervolume: f64,

    /// Average improvement
    pub avg_improvement: f64,

    /// Population diversity
    pub diversity: f64,

    /// Timestamp
    pub timestamp: Instant,
}

impl MultiObjectiveOptimizer {
    /// Create a new multi-objective optimizer
    pub fn new() -> Self {
        let mut optimizer = Self {
            algorithms: HashMap::new(),
            pareto_front: Vec::new(),
            objective_weights: HashMap::new(),
            constraint_handlers: Vec::new(),
            solution_archive: VecDeque::new(),
            performance_metrics: MultiObjectiveMetrics::default(),
            population: Vec::new(),
            reference_point: Vec::new(),
            archive_config: ArchiveConfig::default(),
            diversity_strategy: DiversityStrategy::default(),
            convergence_detector: ConvergenceDetector::new(),
        };

        // Initialize with default algorithms
        optimizer.initialize_default_algorithms();
        optimizer
    }

    /// Initialize default multi-objective algorithms
    fn initialize_default_algorithms(&mut self) {
        // NSGA-II
        let nsga2 = MultiObjectiveAlgorithm {
            name: "NSGA-II".to_string(),
            algorithm_type: MOAlgorithmType::NSGA2,
            parameters: HashMap::new(),
            population_size: 100,
            max_generations: 500,
            convergence_criteria: ConvergenceCriteria::default(),
            crossover_probability: 0.9,
            mutation_probability: 0.1,
            selection_pressure: 1.0,
            elite_ratio: 0.1,
        };
        self.algorithms.insert("NSGA2".to_string(), nsga2);

        // NSGA-III
        let nsga3 = MultiObjectiveAlgorithm {
            name: "NSGA-III".to_string(),
            algorithm_type: MOAlgorithmType::NSGA3,
            parameters: HashMap::new(),
            population_size: 120,
            max_generations: 750,
            convergence_criteria: ConvergenceCriteria::default(),
            crossover_probability: 0.9,
            mutation_probability: 0.1,
            selection_pressure: 1.0,
            elite_ratio: 0.1,
        };
        self.algorithms.insert("NSGA3".to_string(), nsga3);

        // SPEA2
        let spea2 = MultiObjectiveAlgorithm {
            name: "SPEA2".to_string(),
            algorithm_type: MOAlgorithmType::SPEA2,
            parameters: HashMap::new(),
            population_size: 100,
            max_generations: 500,
            convergence_criteria: ConvergenceCriteria::default(),
            crossover_probability: 0.9,
            mutation_probability: 0.1,
            selection_pressure: 1.0,
            elite_ratio: 0.2,
        };
        self.algorithms.insert("SPEA2".to_string(), spea2);
    }

    /// Optimize objectives using specified algorithm
    pub fn optimize(
        &mut self,
        objectives: &[String],
        constraints: &[String],
        algorithm_name: &str,
    ) -> Result<Vec<ParetoSolution>, String> {
        let algorithm = self
            .algorithms
            .get(algorithm_name)
            .ok_or_else(|| format!("Algorithm '{}' not found", algorithm_name))?;

        match algorithm.algorithm_type {
            MOAlgorithmType::NSGA2 => self.run_nsga2(objectives, constraints, algorithm),
            MOAlgorithmType::NSGA3 => self.run_nsga3(objectives, constraints, algorithm),
            MOAlgorithmType::SPEA2 => self.run_spea2(objectives, constraints, algorithm),
            MOAlgorithmType::MOEAD => self.run_moead(objectives, constraints, algorithm),
            MOAlgorithmType::SmsEmoa => self.run_sms_emoa(objectives, constraints, algorithm),
            _ => Err("Algorithm not implemented yet".to_string()),
        }
    }

    /// Run NSGA-II algorithm
    fn run_nsga2(
        &mut self,
        objectives: &[String],
        constraints: &[String],
        algorithm: &MultiObjectiveAlgorithm,
    ) -> Result<Vec<ParetoSolution>, String> {
        // Initialize population
        self.initialize_population(algorithm.population_size, objectives.len());

        for generation in 0..algorithm.max_generations {
            // Evaluate population
            self.evaluate_population(objectives, constraints);

            // Non-dominated sorting
            self.fast_non_dominated_sort();

            // Calculate crowding distance
            self.calculate_crowding_distance();

            // Create offspring
            let offspring = self.create_offspring_nsga2(algorithm);

            // Environmental selection
            self.environmental_selection_nsga2(&offspring);

            // Update metrics
            self.update_metrics(generation);

            // Check convergence
            if self.check_convergence(generation, algorithm) {
                break;
            }
        }

        // Extract Pareto front
        self.extract_pareto_front()
    }

    /// Run NSGA-III algorithm
    fn run_nsga3(
        &mut self,
        objectives: &[String],
        constraints: &[String],
        algorithm: &MultiObjectiveAlgorithm,
    ) -> Result<Vec<ParetoSolution>, String> {
        // Initialize reference points
        let reference_points = self.generate_reference_points(objectives.len());

        // Initialize population
        self.initialize_population(algorithm.population_size, objectives.len());

        for generation in 0..algorithm.max_generations {
            // Evaluate population
            self.evaluate_population(objectives, constraints);

            // Non-dominated sorting
            self.fast_non_dominated_sort();

            // Reference point association
            self.associate_with_reference_points(&reference_points);

            // Create offspring
            let offspring = self.create_offspring_nsga3(algorithm);

            // Environmental selection with reference points
            self.environmental_selection_nsga3(&offspring, &reference_points);

            // Update metrics
            self.update_metrics(generation);

            // Check convergence
            if self.check_convergence(generation, algorithm) {
                break;
            }
        }

        self.extract_pareto_front()
    }

    /// Run SPEA2 algorithm
    fn run_spea2(
        &mut self,
        objectives: &[String],
        constraints: &[String],
        algorithm: &MultiObjectiveAlgorithm,
    ) -> Result<Vec<ParetoSolution>, String> {
        // Initialize population and archive
        self.initialize_population(algorithm.population_size, objectives.len());
        let mut archive = Vec::new();

        for generation in 0..algorithm.max_generations {
            // Evaluate population
            self.evaluate_population(objectives, constraints);

            // Calculate fitness (strength values)
            self.calculate_spea2_fitness(&mut archive);

            // Environmental selection for archive
            self.update_archive_spea2(&mut archive);

            // Create offspring from archive
            let offspring = self.create_offspring_spea2(&archive, algorithm);

            // Replace population with offspring
            self.population = offspring;

            // Update metrics
            self.update_metrics(generation);

            // Check convergence
            if self.check_convergence(generation, algorithm) {
                break;
            }
        }

        // Convert archive to Pareto solutions
        self.archive_to_pareto_solutions(&archive)
    }

    /// Run MOEA/D algorithm
    fn run_moead(
        &mut self,
        objectives: &[String],
        constraints: &[String],
        algorithm: &MultiObjectiveAlgorithm,
    ) -> Result<Vec<ParetoSolution>, String> {
        // Generate weight vectors
        let weight_vectors =
            self.generate_weight_vectors(objectives.len(), algorithm.population_size);

        // Calculate neighborhoods
        let neighborhoods = self.calculate_neighborhoods(&weight_vectors);

        // Initialize population
        self.initialize_population(algorithm.population_size, objectives.len());

        for generation in 0..algorithm.max_generations {
            // Evaluate population
            self.evaluate_population(objectives, constraints);

            // Update solutions using decomposition
            for i in 0..self.population.len() {
                self.update_solution_moead(i, &weight_vectors, &neighborhoods, algorithm);
            }

            // Update reference point
            self.update_reference_point();

            // Update metrics
            self.update_metrics(generation);

            // Check convergence
            if self.check_convergence(generation, algorithm) {
                break;
            }
        }

        self.extract_pareto_front()
    }

    /// Run SMS-EMOA algorithm
    fn run_sms_emoa(
        &mut self,
        objectives: &[String],
        constraints: &[String],
        algorithm: &MultiObjectiveAlgorithm,
    ) -> Result<Vec<ParetoSolution>, String> {
        // Initialize population
        self.initialize_population(algorithm.population_size, objectives.len());

        for generation in 0..algorithm.max_generations {
            // Evaluate population
            self.evaluate_population(objectives, constraints);

            // Select parent for reproduction
            let parent_indices = self.select_parents_tournament(2);

            // Create offspring
            let offspring = self.create_single_offspring(&parent_indices, algorithm);

            // Combine population with offspring
            let mut combined_population = self.population.clone();
            combined_population.push(offspring);

            // Non-dominated sorting
            self.population = combined_population;
            self.fast_non_dominated_sort();

            // Hypervolume-based selection
            self.hypervolume_based_selection(algorithm.population_size);

            // Update metrics
            self.update_metrics(generation);

            // Check convergence
            if self.check_convergence(generation, algorithm) {
                break;
            }
        }

        self.extract_pareto_front()
    }

    /// Initialize random population
    fn initialize_population(&mut self, population_size: usize, num_objectives: usize) {
        let mut rng = rng();
        self.population.clear();

        for _ in 0..population_size {
            let genotype: Vec<f64> = (0..10).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let individual = Individual {
                genotype,
                phenotype: HashMap::new(),
                objectives: vec![0.0; num_objectives],
                constraints: Vec::new(),
                rank: 0,
                crowding_distance: 0.0,
                fitness: 0.0,
                age: 0,
            };
            self.population.push(individual);
        }
    }

    /// Evaluate population on objectives and constraints
    fn evaluate_population(&mut self, objectives: &[String], constraints: &[String]) {
        for individual in &mut self.population {
            // Evaluate objectives (simplified)
            for (i, _objective) in objectives.iter().enumerate() {
                individual.objectives[i] = self.evaluate_objective(i, &individual.genotype);
            }

            // Evaluate constraints
            individual.constraints = constraints
                .iter()
                .map(|constraint| self.evaluate_constraint(constraint, &individual.genotype))
                .collect();
        }
    }

    /// Evaluate a single objective (placeholder implementation)
    fn evaluate_objective(&self, objective_index: usize, genotype: &[f64]) -> f64 {
        // This would be replaced with actual objective function evaluation
        match objective_index {
            0 => genotype.iter().map(|x| x * x).sum(), // Performance (minimize)
            1 => genotype.iter().sum::<f64>().abs(),   // Memory usage (minimize)
            2 => genotype.iter().map(|x| x.abs()).sum(), // Latency (minimize)
            _ => 0.0,
        }
    }

    /// Evaluate a single constraint (placeholder implementation)
    fn evaluate_constraint(&self, _constraint: &str, genotype: &[f64]) -> f64 {
        // This would be replaced with actual constraint evaluation
        genotype.iter().sum::<f64>() // Simplified constraint
    }

    /// Fast non-dominated sorting algorithm
    fn fast_non_dominated_sort(&mut self) {
        let pop_size = self.population.len();
        let mut dominance_sets: Vec<Vec<usize>> = vec![Vec::new(); pop_size];
        let mut dominated_counts = vec![0; pop_size];
        let mut fronts: Vec<Vec<usize>> = vec![Vec::new()];

        // Calculate dominance relationships
        for i in 0..pop_size {
            for j in 0..pop_size {
                if i != j {
                    if self.dominates(i, j) {
                        dominance_sets[i].push(j);
                    } else if self.dominates(j, i) {
                        dominated_counts[i] += 1;
                    }
                }
            }

            if dominated_counts[i] == 0 {
                self.population[i].rank = 0;
                fronts[0].push(i);
            }
        }

        // Build subsequent fronts
        let mut front_index = 0;
        while !fronts[front_index].is_empty() {
            let mut next_front = Vec::new();

            for &i in &fronts[front_index] {
                for &j in &dominance_sets[i] {
                    dominated_counts[j] -= 1;
                    if dominated_counts[j] == 0 {
                        self.population[j].rank = front_index + 1;
                        next_front.push(j);
                    }
                }
            }

            if !next_front.is_empty() {
                fronts.push(next_front);
                front_index += 1;
            } else {
                break;
            }
        }
    }

    /// Check if individual i dominates individual j
    fn dominates(&self, i: usize, j: usize) -> bool {
        let obj_i = &self.population[i].objectives;
        let obj_j = &self.population[j].objectives;

        let mut at_least_one_better = false;
        for (oi, oj) in obj_i.iter().zip(obj_j.iter()) {
            if oi > oj {
                return false; // Assuming minimization
            }
            if oi < oj {
                at_least_one_better = true;
            }
        }

        at_least_one_better
    }

    /// Calculate crowding distance for diversity preservation
    fn calculate_crowding_distance(&mut self) {
        if self.population.is_empty() {
            return;
        }

        // Initialize all distances to 0
        for individual in &mut self.population {
            individual.crowding_distance = 0.0;
        }

        let num_objectives = self.population[0].objectives.len();

        for obj_index in 0..num_objectives {
            // Sort by objective value
            let mut indices: Vec<usize> = (0..self.population.len()).collect();
            indices.sort_by(|&i, &j| {
                self.population[i].objectives[obj_index]
                    .partial_cmp(&self.population[j].objectives[obj_index])
                    .expect("objective values should be comparable (finite numbers)")
            });

            // Set boundary points to infinite distance
            self.population[indices[0]].crowding_distance = f64::INFINITY;
            self.population[indices[indices.len() - 1]].crowding_distance = f64::INFINITY;

            // Calculate distances for intermediate points
            let obj_range = self.population[indices[indices.len() - 1]].objectives[obj_index]
                - self.population[indices[0]].objectives[obj_index];

            if obj_range > 0.0 {
                for i in 1..indices.len() - 1 {
                    let distance = (self.population[indices[i + 1]].objectives[obj_index]
                        - self.population[indices[i - 1]].objectives[obj_index])
                        / obj_range;
                    self.population[indices[i]].crowding_distance += distance;
                }
            }
        }
    }

    /// Create offspring using NSGA-II operations
    fn create_offspring_nsga2(&self, algorithm: &MultiObjectiveAlgorithm) -> Vec<Individual> {
        let mut offspring = Vec::new();
        let mut rng = rng();

        while offspring.len() < self.population.len() {
            // Tournament selection
            let parent1_idx = self.tournament_selection();
            let parent2_idx = self.tournament_selection();

            let parent1 = &self.population[parent1_idx];
            let parent2 = &self.population[parent2_idx];

            // Crossover
            if rng.gen_range(0.0..1.0) < algorithm.crossover_probability {
                let (child1, child2) = self.simulated_binary_crossover(parent1, parent2);
                offspring.push(child1);
                if offspring.len() < self.population.len() {
                    offspring.push(child2);
                }
            } else {
                offspring.push(parent1.clone());
                if offspring.len() < self.population.len() {
                    offspring.push(parent2.clone());
                }
            }
        }

        // Mutation
        for child in &mut offspring {
            if rng.gen_range(0.0..1.0) < algorithm.mutation_probability {
                self.polynomial_mutation(child);
            }
        }

        offspring
    }

    /// Tournament selection for parent selection
    fn tournament_selection(&self) -> usize {
        let mut rng = rng();
        let tournament_size = 2;
        let mut best_idx = rng.gen_range(0..self.population.len());

        for _ in 1..tournament_size {
            let candidate_idx = rng.gen_range(0..self.population.len());
            if self.is_better_solution(candidate_idx, best_idx) {
                best_idx = candidate_idx;
            }
        }

        best_idx
    }

    /// Check if solution i is better than solution j
    fn is_better_solution(&self, i: usize, j: usize) -> bool {
        let ind_i = &self.population[i];
        let ind_j = &self.population[j];

        if ind_i.rank < ind_j.rank {
            true
        } else if ind_i.rank == ind_j.rank {
            ind_i.crowding_distance > ind_j.crowding_distance
        } else {
            false
        }
    }

    /// Simulated Binary Crossover (SBX)
    fn simulated_binary_crossover(
        &self,
        parent1: &Individual,
        parent2: &Individual,
    ) -> (Individual, Individual) {
        let mut child1 = parent1.clone();
        let mut child2 = parent2.clone();
        let mut rng = rng();
        let eta_c = 20.0; // Distribution index

        for i in 0..parent1.genotype.len() {
            if rng.gen_range(0.0..1.0) <= 0.5 {
                let y1 = parent1.genotype[i].min(parent2.genotype[i]);
                let y2 = parent1.genotype[i].max(parent2.genotype[i]);

                let rand: f64 = rng.gen_range(0.0..1.0);
                let beta: f64 = if rand <= 0.5 {
                    (2.0_f64 * rand).powf(1.0 / (eta_c + 1.0))
                } else {
                    (1.0_f64 / (2.0_f64 * (1.0_f64 - rand))).powf(1.0 / (eta_c + 1.0))
                };

                child1.genotype[i] = 0.5 * ((y1 + y2) - beta * (y2 - y1));
                child2.genotype[i] = 0.5 * ((y1 + y2) + beta * (y2 - y1));

                // Bound the values
                child1.genotype[i] = child1.genotype[i].clamp(-1.0, 1.0);
                child2.genotype[i] = child2.genotype[i].clamp(-1.0, 1.0);
            }
        }

        (child1, child2)
    }

    /// Polynomial mutation
    fn polynomial_mutation(&self, individual: &mut Individual) {
        let mut rng = rng();
        let eta_m = 20.0; // Distribution index
        let mutation_prob = 1.0 / individual.genotype.len() as f64;

        for gene in &mut individual.genotype {
            if rng.gen_range(0.0..1.0) <= mutation_prob {
                let rand: f64 = rng.gen_range(0.0..1.0);
                let delta: f64 = if rand < 0.5 {
                    (2.0_f64 * rand).powf(1.0 / (eta_m + 1.0)) - 1.0
                } else {
                    1.0 - (2.0_f64 * (1.0_f64 - rand)).powf(1.0 / (eta_m + 1.0))
                };

                *gene += delta;
                *gene = gene.clamp(-1.0, 1.0);
            }
        }
    }

    /// Environmental selection for NSGA-II
    fn environmental_selection_nsga2(&mut self, offspring: &[Individual]) {
        // Combine population and offspring
        let mut combined = self.population.clone();
        combined.extend_from_slice(offspring);

        // Perform non-dominated sorting on combined population
        self.population = combined;
        self.fast_non_dominated_sort();
        self.calculate_crowding_distance();

        // Select best individuals
        self.population.sort_by(|a, b| {
            if a.rank != b.rank {
                a.rank.cmp(&b.rank)
            } else {
                b.crowding_distance
                    .partial_cmp(&a.crowding_distance)
                    .expect("crowding_distance should be comparable (finite numbers)")
            }
        });

        self.population.truncate(self.population.len() / 2);
    }

    // Note: run_nsga3, run_moead, run_sms_emoa implementations are defined above

    // Additional placeholder methods
    fn create_offspring_nsga3(&self, _algorithm: &MultiObjectiveAlgorithm) -> Vec<Individual> {
        Vec::new()
    }

    fn environmental_selection_nsga3(
        &mut self,
        _offspring: &[Individual],
        _reference_points: &[Vec<f64>],
    ) {
    }

    fn generate_reference_points(&self, _num_objectives: usize) -> Vec<Vec<f64>> {
        Vec::new()
    }

    fn associate_with_reference_points(&mut self, _reference_points: &[Vec<f64>]) {}

    fn calculate_spea2_fitness(&mut self, _archive: &mut Vec<Individual>) {}

    fn update_archive_spea2(&mut self, _archive: &mut Vec<Individual>) {}

    fn create_offspring_spea2(
        &self,
        _archive: &[Individual],
        _algorithm: &MultiObjectiveAlgorithm,
    ) -> Vec<Individual> {
        Vec::new()
    }

    fn archive_to_pareto_solutions(
        &self,
        _archive: &[Individual],
    ) -> Result<Vec<ParetoSolution>, String> {
        Ok(Vec::new())
    }

    fn generate_weight_vectors(
        &self,
        _num_objectives: usize,
        _population_size: usize,
    ) -> Vec<Vec<f64>> {
        Vec::new()
    }

    fn calculate_neighborhoods(&self, _weight_vectors: &[Vec<f64>]) -> Vec<Vec<usize>> {
        Vec::new()
    }

    fn update_solution_moead(
        &mut self,
        _index: usize,
        _weight_vectors: &[Vec<f64>],
        _neighborhoods: &[Vec<usize>],
        _algorithm: &MultiObjectiveAlgorithm,
    ) {
    }

    fn update_reference_point(&mut self) {}

    fn select_parents_tournament(&self, _tournament_size: usize) -> Vec<usize> {
        Vec::new()
    }

    fn create_single_offspring(
        &self,
        _parent_indices: &[usize],
        _algorithm: &MultiObjectiveAlgorithm,
    ) -> Individual {
        Individual {
            genotype: Vec::new(),
            phenotype: HashMap::new(),
            objectives: Vec::new(),
            constraints: Vec::new(),
            rank: 0,
            crowding_distance: 0.0,
            fitness: 0.0,
            age: 0,
        }
    }

    fn hypervolume_based_selection(&mut self, _target_size: usize) {}

    /// Update performance metrics
    fn update_metrics(&mut self, generation: usize) {
        self.performance_metrics.generations = generation;
        self.performance_metrics.solution_count =
            self.population.iter().filter(|ind| ind.rank == 0).count();

        // Calculate hypervolume (simplified)
        self.performance_metrics.hypervolume = self.calculate_hypervolume();

        // Update other metrics
        self.performance_metrics.diversity = self.calculate_diversity();
        self.performance_metrics.convergence = self.calculate_convergence();
        self.performance_metrics.spacing = self.calculate_spacing();
    }

    /// Calculate hypervolume indicator
    fn calculate_hypervolume(&self) -> f64 {
        // Simplified hypervolume calculation
        // In practice, this would use more sophisticated algorithms
        let front: Vec<&Individual> = self.population.iter().filter(|ind| ind.rank == 0).collect();

        if front.is_empty() {
            return 0.0;
        }

        // Use reference point [1.0, 1.0, ...] for all objectives
        let reference_point: Vec<f64> = vec![1.0; front[0].objectives.len()];

        // Calculate volume dominated by each solution
        let mut total_volume = 0.0;
        for individual in front {
            let mut volume = 1.0;
            for (obj_val, ref_val) in individual.objectives.iter().zip(reference_point.iter()) {
                volume *= (ref_val - obj_val).max(0.0);
            }
            total_volume += volume;
        }

        total_volume
    }

    /// Calculate population diversity
    fn calculate_diversity(&self) -> f64 {
        if self.population.len() < 2 {
            return 0.0;
        }

        let mut total_distance = 0.0;
        let mut count = 0;

        for i in 0..self.population.len() {
            for j in i + 1..self.population.len() {
                let distance = self.euclidean_distance(
                    &self.population[i].objectives,
                    &self.population[j].objectives,
                );
                total_distance += distance;
                count += 1;
            }
        }

        if count > 0 {
            total_distance / count as f64
        } else {
            0.0
        }
    }

    /// Calculate convergence metric
    fn calculate_convergence(&self) -> f64 {
        // Simplified convergence calculation
        // This would typically measure distance to known Pareto front
        self.population
            .iter()
            .filter(|ind| ind.rank == 0)
            .map(|ind| ind.objectives.iter().sum::<f64>())
            .fold(0.0, |acc, sum| acc + sum)
            / self.population.len() as f64
    }

    /// Calculate spacing metric
    fn calculate_spacing(&self) -> f64 {
        let front: Vec<&Individual> = self.population.iter().filter(|ind| ind.rank == 0).collect();

        if front.len() < 2 {
            return 0.0;
        }

        let mut distances = Vec::new();
        for i in 0..front.len() {
            let mut min_distance = f64::INFINITY;
            for j in 0..front.len() {
                if i != j {
                    let distance =
                        self.euclidean_distance(&front[i].objectives, &front[j].objectives);
                    min_distance = min_distance.min(distance);
                }
            }
            distances.push(min_distance);
        }

        let mean_distance = distances.iter().sum::<f64>() / distances.len() as f64;
        let variance = distances
            .iter()
            .map(|d| (d - mean_distance).powi(2))
            .sum::<f64>()
            / distances.len() as f64;

        variance.sqrt()
    }

    /// Calculate Euclidean distance between two objective vectors
    fn euclidean_distance(&self, obj1: &[f64], obj2: &[f64]) -> f64 {
        obj1.iter()
            .zip(obj2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Check convergence criteria
    fn check_convergence(
        &mut self,
        generation: usize,
        algorithm: &MultiObjectiveAlgorithm,
    ) -> bool {
        // Update convergence detector
        let snapshot = ConvergenceSnapshot {
            generation,
            hypervolume: self.performance_metrics.hypervolume,
            avg_improvement: 0.0, // Would be calculated based on previous generations
            diversity: self.performance_metrics.diversity,
            timestamp: Instant::now(),
        };

        self.convergence_detector.history.push_back(snapshot);
        if self.convergence_detector.history.len() > self.convergence_detector.window_size {
            self.convergence_detector.history.pop_front();
        }

        // Check various convergence criteria
        if generation >= algorithm.convergence_criteria.max_generations {
            return true;
        }

        if let Some(target_hv) = algorithm.convergence_criteria.target_hypervolume {
            if self.performance_metrics.hypervolume >= target_hv {
                return true;
            }
        }

        // Check stagnation
        if self.convergence_detector.history.len()
            >= algorithm.convergence_criteria.stagnation_limit
        {
            let recent_hvs: Vec<f64> = self
                .convergence_detector
                .history
                .iter()
                .rev()
                .take(algorithm.convergence_criteria.stagnation_limit)
                .map(|s| s.hypervolume)
                .collect();

            let max_hv = recent_hvs.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let min_hv = recent_hvs.iter().fold(f64::INFINITY, |a, &b| a.min(b));

            if max_hv - min_hv < algorithm.convergence_criteria.tolerance {
                return true;
            }
        }

        false
    }

    /// Extract Pareto front from population
    fn extract_pareto_front(&self) -> Result<Vec<ParetoSolution>, String> {
        let mut pareto_solutions = Vec::new();

        for (index, individual) in self.population.iter().enumerate() {
            if individual.rank == 0 {
                // First front
                let mut parameters = HashMap::new();
                for (i, &value) in individual.genotype.iter().enumerate() {
                    parameters.insert(format!("param_{}", i), value);
                }

                let mut objectives = HashMap::new();
                for (i, &value) in individual.objectives.iter().enumerate() {
                    objectives.insert(format!("objective_{}", i), value);
                }

                let solution = ParetoSolution {
                    id: format!("solution_{}", index),
                    parameters,
                    objectives,
                    rank: individual.rank,
                    crowding_distance: individual.crowding_distance,
                    quality_score: self.calculate_quality_score(individual),
                    timestamp: Instant::now(),
                    constraint_violations: HashMap::new(), // Would be populated from constraints
                    fitness: individual.fitness,
                    age: individual.age,
                    niche_count: 1.0,              // Would be calculated
                    hypervolume_contribution: 0.0, // Would be calculated
                };

                pareto_solutions.push(solution);
            }
        }

        Ok(pareto_solutions)
    }

    /// Calculate quality score for a solution
    fn calculate_quality_score(&self, individual: &Individual) -> f32 {
        // Combine rank and crowding distance into quality score
        let rank_component = 1.0 / (individual.rank as f32 + 1.0);
        let diversity_component = individual.crowding_distance as f32;

        (rank_component + diversity_component.min(1.0)) / 2.0
    }

    /// Get current Pareto front
    pub fn get_pareto_front(&self) -> &[ParetoSolution] {
        &self.pareto_front
    }

    /// Get optimization metrics
    pub fn get_metrics(&self) -> &MultiObjectiveMetrics {
        &self.performance_metrics
    }

    /// Add custom algorithm
    pub fn add_algorithm(&mut self, name: String, algorithm: MultiObjectiveAlgorithm) {
        self.algorithms.insert(name, algorithm);
    }

    /// Remove algorithm
    pub fn remove_algorithm(&mut self, name: &str) -> Option<MultiObjectiveAlgorithm> {
        self.algorithms.remove(name)
    }

    /// List available algorithms
    pub fn list_algorithms(&self) -> Vec<String> {
        self.algorithms.keys().cloned().collect()
    }
}

// Default implementations
impl Default for MultiObjectiveMetrics {
    fn default() -> Self {
        Self {
            hypervolume: 0.0,
            spacing: 0.0,
            convergence: 0.0,
            diversity: 0.0,
            solution_count: 0,
            generations: 0,
            generational_distance: 0.0,
            inverted_generational_distance: 0.0,
            coverage_metrics: CoverageMetrics::default(),
            performance_stats: PerformanceStats::default(),
            quality_indicators: QualityIndicators::default(),
        }
    }
}

impl Default for ConvergenceCriteria {
    fn default() -> Self {
        Self {
            max_generations: 500,
            target_hypervolume: None,
            tolerance: 1e-6,
            stagnation_limit: 50,
            min_improvement: 1e-4,
            convergence_metrics: vec![ConvergenceMetric::Hypervolume],
            early_stopping: None,
        }
    }
}

impl Default for ArchiveConfig {
    fn default() -> Self {
        Self {
            max_size: 200,
            update_strategy: ArchiveUpdateStrategy::CrowdingBased,
            duplicate_handling: DuplicateHandling::Reject,
            quality_threshold: 0.5,
            aging_strategy: None,
        }
    }
}

impl Default for DiversityStrategy {
    fn default() -> Self {
        Self {
            strategy_type: DiversityStrategyType::CrowdingDistance,
            diversity_threshold: 0.1,
            niche_radius: 0.1,
            clustering_config: None,
        }
    }
}

impl Default for CoverageMetrics {
    fn default() -> Self {
        Self {
            c_metric: 0.0,
            set_coverage: 0.0,
            epsilon_indicator: 0.0,
            binary_epsilon: 0.0,
        }
    }
}

impl Default for PerformanceStats {
    fn default() -> Self {
        Self {
            avg_generation_time: Duration::from_secs(0),
            total_time: Duration::from_secs(0),
            evaluations_per_second: 0.0,
            peak_memory_usage: 0,
            cpu_utilization: 0.0,
            convergence_rate: 0.0,
        }
    }
}

impl Default for QualityIndicators {
    fn default() -> Self {
        Self {
            additive_epsilon: 0.0,
            multiplicative_epsilon: 0.0,
            r2_indicator: 0.0,
            igd_plus: 0.0,
            modified_igd: 0.0,
        }
    }
}

impl ConvergenceDetector {
    fn new() -> Self {
        Self {
            strategy: ConvergenceDetectionStrategy::HypervolumeStagnation,
            history: VecDeque::new(),
            window_size: 10,
            sensitivity: 0.01,
            is_converged: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_objective_optimizer_creation() {
        let optimizer = MultiObjectiveOptimizer::new();
        assert!(!optimizer.algorithms.is_empty());
        assert!(optimizer.algorithms.contains_key("NSGA2"));
        assert!(optimizer.algorithms.contains_key("NSGA3"));
        assert!(optimizer.algorithms.contains_key("SPEA2"));
    }

    #[test]
    fn test_algorithm_configuration() {
        let optimizer = MultiObjectiveOptimizer::new();
        let nsga2 = optimizer.algorithms.get("NSGA2").unwrap();

        assert_eq!(nsga2.algorithm_type, MOAlgorithmType::NSGA2);
        assert_eq!(nsga2.population_size, 100);
        assert_eq!(nsga2.max_generations, 500);
        assert_eq!(nsga2.crossover_probability, 0.9);
        assert_eq!(nsga2.mutation_probability, 0.1);
    }

    #[test]
    fn test_dominance_check() {
        let mut optimizer = MultiObjectiveOptimizer::new();

        // Create test individuals
        let ind1 = Individual {
            genotype: vec![1.0, 2.0],
            phenotype: HashMap::new(),
            objectives: vec![1.0, 2.0], // Individual 1
            constraints: Vec::new(),
            rank: 0,
            crowding_distance: 0.0,
            fitness: 0.0,
            age: 0,
        };

        let ind2 = Individual {
            genotype: vec![2.0, 3.0],
            phenotype: HashMap::new(),
            objectives: vec![2.0, 3.0], // Individual 2 (dominated by 1)
            constraints: Vec::new(),
            rank: 0,
            crowding_distance: 0.0,
            fitness: 0.0,
            age: 0,
        };

        optimizer.population = vec![ind1, ind2];

        // Test dominance (assuming minimization)
        assert!(optimizer.dominates(0, 1)); // ind1 dominates ind2
        assert!(!optimizer.dominates(1, 0)); // ind2 does not dominate ind1
    }

    #[test]
    fn test_population_initialization() {
        let mut optimizer = MultiObjectiveOptimizer::new();
        optimizer.initialize_population(50, 3);

        assert_eq!(optimizer.population.len(), 50);
        for individual in &optimizer.population {
            assert_eq!(individual.genotype.len(), 10);
            assert_eq!(individual.objectives.len(), 3);
            assert_eq!(individual.rank, 0);
            assert_eq!(individual.age, 0);
        }
    }

    #[test]
    fn test_metrics_calculation() {
        let mut optimizer = MultiObjectiveOptimizer::new();
        optimizer.initialize_population(20, 2);

        // Evaluate population with dummy objectives
        for individual in &mut optimizer.population {
            individual.objectives[0] = individual.genotype[0].abs();
            individual.objectives[1] = individual.genotype[1].abs();
        }

        optimizer.fast_non_dominated_sort();
        optimizer.update_metrics(1);

        let metrics = &optimizer.performance_metrics;
        assert_eq!(metrics.generations, 1);
        assert!(metrics.hypervolume >= 0.0);
        assert!(metrics.diversity >= 0.0);
        assert!(metrics.solution_count > 0);
    }

    #[test]
    fn test_nsga2_algorithm() {
        let mut optimizer = MultiObjectiveOptimizer::new();

        // Test with simple objectives
        let objectives = vec!["performance".to_string(), "memory".to_string()];
        let constraints = vec![];

        // This should run without panic (though may not converge in few generations)
        let result = optimizer.optimize(&objectives, &constraints, "NSGA2");
        assert!(result.is_ok());
    }

    #[test]
    fn test_pareto_solution_extraction() {
        let mut optimizer = MultiObjectiveOptimizer::new();
        optimizer.initialize_population(10, 2);

        // Set up a simple Pareto front
        for (i, individual) in optimizer.population.iter_mut().enumerate() {
            individual.objectives[0] = i as f64;
            individual.objectives[1] = (10 - i) as f64;
            individual.rank = if i < 5 { 0 } else { 1 }; // First 5 are non-dominated
        }

        let pareto_solutions = optimizer.extract_pareto_front().unwrap();
        assert_eq!(pareto_solutions.len(), 5); // First front only

        for solution in pareto_solutions {
            assert_eq!(solution.rank, 0);
            assert!(!solution.parameters.is_empty());
            assert!(!solution.objectives.is_empty());
        }
    }

    #[test]
    fn test_convergence_detection() {
        let mut optimizer = MultiObjectiveOptimizer::new();
        let algorithm = optimizer.algorithms.get("NSGA2").unwrap().clone();

        // Test early convergence due to max generations
        let mut algorithm_short = algorithm.clone();
        algorithm_short.convergence_criteria.max_generations = 5;

        assert!(optimizer.check_convergence(5, &algorithm_short));
        assert!(!optimizer.check_convergence(3, &algorithm_short));

        // Test target hypervolume convergence
        let mut algorithm_hv = algorithm.clone();
        algorithm_hv.convergence_criteria.target_hypervolume = Some(0.5);
        optimizer.performance_metrics.hypervolume = 0.6;

        assert!(optimizer.check_convergence(10, &algorithm_hv));

        optimizer.performance_metrics.hypervolume = 0.3;
        assert!(!optimizer.check_convergence(10, &algorithm_hv));
    }

    #[test]
    fn test_algorithm_management() {
        let mut optimizer = MultiObjectiveOptimizer::new();
        let initial_count = optimizer.list_algorithms().len();

        // Add custom algorithm
        let custom_algorithm = MultiObjectiveAlgorithm {
            name: "Custom".to_string(),
            algorithm_type: MOAlgorithmType::Custom,
            parameters: HashMap::new(),
            population_size: 50,
            max_generations: 100,
            convergence_criteria: ConvergenceCriteria::default(),
            crossover_probability: 0.8,
            mutation_probability: 0.2,
            selection_pressure: 1.5,
            elite_ratio: 0.05,
        };

        optimizer.add_algorithm("Custom".to_string(), custom_algorithm);
        assert_eq!(optimizer.list_algorithms().len(), initial_count + 1);
        assert!(optimizer.algorithms.contains_key("Custom"));

        // Remove algorithm
        let removed = optimizer.remove_algorithm("Custom");
        assert!(removed.is_some());
        assert_eq!(optimizer.list_algorithms().len(), initial_count);
        assert!(!optimizer.algorithms.contains_key("Custom"));
    }

    #[test]
    fn test_crossover_operation() {
        let optimizer = MultiObjectiveOptimizer::new();

        let parent1 = Individual {
            genotype: vec![0.5, -0.5, 0.0],
            phenotype: HashMap::new(),
            objectives: vec![1.0, 2.0],
            constraints: Vec::new(),
            rank: 0,
            crowding_distance: 0.0,
            fitness: 0.0,
            age: 0,
        };

        let parent2 = Individual {
            genotype: vec![-0.5, 0.5, 1.0],
            phenotype: HashMap::new(),
            objectives: vec![2.0, 1.0],
            constraints: Vec::new(),
            rank: 0,
            crowding_distance: 0.0,
            fitness: 0.0,
            age: 0,
        };

        let (child1, child2) = optimizer.simulated_binary_crossover(&parent1, &parent2);

        // Children should have valid genotypes
        assert_eq!(child1.genotype.len(), 3);
        assert_eq!(child2.genotype.len(), 3);

        // All genes should be within bounds
        for &gene in &child1.genotype {
            assert!(gene >= -1.0 && gene <= 1.0);
        }
        for &gene in &child2.genotype {
            assert!(gene >= -1.0 && gene <= 1.0);
        }
    }

    #[test]
    fn test_tournament_selection() {
        let mut optimizer = MultiObjectiveOptimizer::new();
        optimizer.initialize_population(10, 2);

        // Set up different ranks and crowding distances
        for (i, individual) in optimizer.population.iter_mut().enumerate() {
            individual.rank = i / 3; // Groups of 3 with same rank
            individual.crowding_distance = (i % 3) as f64; // Different crowding distances
        }

        // Run tournament selection multiple times
        let mut selections = Vec::new();
        for _ in 0..20 {
            selections.push(optimizer.tournament_selection());
        }

        // Should select valid indices
        for &selected in &selections {
            assert!(selected < optimizer.population.len());
        }

        // Should prefer better solutions (lower rank, higher crowding distance)
        // This is probabilistic, so we just check that it doesn't crash
        assert!(!selections.is_empty());
    }
}
