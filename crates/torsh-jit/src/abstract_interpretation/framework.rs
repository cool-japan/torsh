//! Core Abstract Interpretation Framework
//!
//! This module contains the main AbstractInterpreter engine that orchestrates
//! abstract interpretation analysis, including forward/backward analysis,
//! invariant detection, and property checking.

use super::domains::{
    AbstractDomain, AbstractDomainFactory, AbstractDomainType, AbstractValue, BinaryAbstractOp,
    ConstantValue as DomainConstantValue, SignValue, UnaryAbstractOp,
};
use crate::graph::operations::{ConstantValue as OpConstantValue, Operation};
use crate::{
    ir::{BasicBlock, BlockId, IrModule, IrOpcode, IrValue},
    ComputationGraph, JitResult, NodeId,
};
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{Duration, Instant};

/// Abstract interpretation configuration
#[derive(Debug, Clone)]
pub struct AbstractInterpretationConfig {
    /// Type of abstract domain to use
    pub domain_type: AbstractDomainType,
    /// Maximum number of fixpoint iterations
    pub max_iterations: usize,
    /// Number of iterations before applying widening
    pub widening_delay: usize,
    /// Enable narrowing after widening
    pub enable_narrowing: bool,
    /// Enable backward analysis in addition to forward analysis
    pub enable_backward_analysis: bool,
    /// Properties to check during analysis
    pub properties: Vec<Property>,
    /// Minimum precision threshold for analysis quality
    pub precision_threshold: f64,
}

impl Default for AbstractInterpretationConfig {
    fn default() -> Self {
        Self {
            domain_type: AbstractDomainType::Intervals,
            max_iterations: 100,
            widening_delay: 3,
            enable_narrowing: true,
            enable_backward_analysis: false,
            properties: Vec::new(),
            precision_threshold: 0.8,
        }
    }
}

/// Properties to verify during abstract interpretation
#[derive(Debug, Clone)]
pub enum Property {
    /// Value is always non-negative
    NonNegative(NodeId),
    /// Value is always positive
    Positive(NodeId),
    /// Value is within specified bounds
    BoundedValue(NodeId, f64, f64),
    /// No division by zero
    NoDivisionByZero(NodeId),
    /// No overflow
    NoOverflow(NodeId),
    /// Custom safety property
    SafetyProperty(String, NodeId),
}

/// Safety check types for verification
#[derive(Debug, Clone)]
pub enum SafetyCheck {
    /// Bounds checking
    BoundsCheck,
    /// Null pointer check
    NullCheck,
    /// Division by zero check
    DivisionByZeroCheck,
    /// Overflow check
    OverflowCheck,
    /// Array access bounds
    ArrayBoundsCheck,
}

/// Result of a safety check
#[derive(Debug, Clone)]
pub enum SafetyCheckResult {
    /// Property definitely holds
    Safe,
    /// Property definitely violated
    Unsafe,
    /// Property may or may not hold (imprecise analysis)
    Unknown,
}

/// Invariant detected during analysis
#[derive(Debug, Clone)]
pub struct Invariant {
    /// Type of invariant
    pub invariant_type: InvariantType,
    /// Human-readable description
    pub description: String,
    /// Confidence level (0.0 to 1.0)
    pub confidence: f64,
    /// Location where invariant holds
    pub location: String,
}

/// Types of invariants that can be detected
#[derive(Debug, Clone)]
pub enum InvariantType {
    /// Value range invariant
    ValueRange,
    /// Loop invariant
    LoopInvariant,
    /// Conditional invariant
    ConditionalInvariant,
    /// Memory safety invariant
    MemorySafety,
    /// Numerical property
    NumericalProperty,
}

/// Function-level invariant
#[derive(Debug, Clone)]
pub struct FunctionInvariant {
    pub invariant_type: InvariantType,
    pub description: String,
    pub confidence: f64,
    pub location: String,
}

/// Module-level invariant
#[derive(Debug, Clone)]
pub struct ModuleInvariant {
    pub invariant_type: InvariantType,
    pub description: String,
    pub confidence: f64,
    pub scope: String,
}

/// Analysis statistics tracking
#[derive(Debug, Clone)]
pub struct AnalysisStatistics {
    /// Total analysis time
    pub analysis_time: Duration,
    /// Number of fixpoint iterations
    pub fixpoint_iterations: usize,
    /// Number of abstract states computed
    pub abstract_states_computed: usize,
    /// Cache hit count
    pub cache_hits: usize,
    /// Cache miss count
    pub cache_misses: usize,
}

/// Forward analysis result
#[derive(Debug, Clone)]
pub struct ForwardAnalysisResult {
    /// Pre-states for each node
    pub pre_states: HashMap<NodeId, AbstractValue>,
    /// Post-states for each node
    pub post_states: HashMap<NodeId, AbstractValue>,
    /// Number of iterations until convergence
    pub iterations: usize,
    /// Whether analysis converged
    pub converged: bool,
}

impl ForwardAnalysisResult {
    pub fn new() -> Self {
        Self {
            pre_states: HashMap::new(),
            post_states: HashMap::new(),
            iterations: 0,
            converged: false,
        }
    }
}

impl Default for ForwardAnalysisResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Backward analysis result
#[derive(Debug, Clone)]
pub struct BackwardAnalysisResult {
    /// Pre-states for each node (computed backwards)
    pub pre_states: HashMap<NodeId, AbstractValue>,
    /// Post-states for each node (computed backwards)
    pub post_states: HashMap<NodeId, AbstractValue>,
    /// Number of iterations until convergence
    pub iterations: usize,
    /// Whether analysis converged
    pub converged: bool,
}

impl BackwardAnalysisResult {
    pub fn new() -> Self {
        Self {
            pre_states: HashMap::new(),
            post_states: HashMap::new(),
            iterations: 0,
            converged: false,
        }
    }
}

impl Default for BackwardAnalysisResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Function-specific analysis results
#[derive(Debug, Clone)]
pub struct AbstractFunctionResult {
    /// Forward analysis result
    pub forward_result: FunctionForwardResult,
    /// Backward analysis result (if enabled)
    pub backward_result: Option<FunctionBackwardResult>,
    /// Function-level invariants
    pub invariants: Vec<FunctionInvariant>,
}

/// Forward analysis result for a function
#[derive(Debug, Clone)]
pub struct FunctionForwardResult {
    /// Whether analysis converged
    pub converged: bool,
    /// Number of iterations
    pub iterations: usize,
    /// Function entry state
    pub entry_state: Option<AbstractValue>,
    /// Function exit state
    pub exit_state: Option<AbstractValue>,
}

impl FunctionForwardResult {
    pub fn new() -> Self {
        Self {
            converged: false,
            iterations: 0,
            entry_state: None,
            exit_state: None,
        }
    }
}

impl Default for FunctionForwardResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Backward analysis result for a function
#[derive(Debug, Clone)]
pub struct FunctionBackwardResult {
    /// Whether analysis converged
    pub converged: bool,
    /// Number of iterations
    pub iterations: usize,
    /// Function entry state (computed backwards)
    pub entry_state: Option<AbstractValue>,
    /// Function exit state (computed backwards)
    pub exit_state: Option<AbstractValue>,
}

impl FunctionBackwardResult {
    pub fn new() -> Self {
        Self {
            converged: false,
            iterations: 0,
            entry_state: None,
            exit_state: None,
        }
    }
}

impl Default for FunctionBackwardResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Interprocedural analysis result
#[derive(Debug, Clone)]
pub struct InterproceduralAnalysisResult {
    /// Call graph representation
    pub call_graph: HashMap<String, Vec<String>>,
    /// Global invariants across functions
    pub global_invariants: Vec<Invariant>,
}

/// Property verification result
#[derive(Debug, Clone)]
pub struct PropertyResult {
    /// The property that was checked
    pub property: Property,
    /// Result of the check
    pub result: SafetyCheckResult,
    /// Confidence in the result
    pub confidence: f64,
    /// Additional details about the check
    pub details: String,
}

/// Precision analysis result
#[derive(Debug, Clone)]
pub struct PrecisionAnalysis {
    /// Overall precision score
    pub overall_precision: f64,
    /// Per-node precision scores
    pub node_precision: HashMap<NodeId, f64>,
    /// Areas where precision could be improved
    pub improvement_suggestions: Vec<String>,
}

/// Performance analysis result
#[derive(Debug, Clone)]
pub struct PerformanceAnalysis {
    /// Overall complexity score
    pub complexity_score: f64,
    /// Detected performance bottlenecks
    pub bottlenecks: Vec<PerformanceBottleneck>,
    /// Optimization opportunities
    pub optimization_opportunities: Vec<OptimizationOpportunity>,
}

/// Performance bottleneck detection
#[derive(Debug, Clone)]
pub struct PerformanceBottleneck {
    /// Type of bottleneck
    pub bottleneck_type: BottleneckType,
    /// Location of the bottleneck
    pub location: NodeId,
    /// Severity score (0.0 to 1.0)
    pub severity: f64,
    /// Description of the issue
    pub description: String,
}

/// Types of performance bottlenecks
#[derive(Debug, Clone)]
pub enum BottleneckType {
    /// Expensive computation
    ComputationIntensive,
    /// Memory bandwidth limited
    MemoryBound,
    /// Loop with high iteration count
    LoopBottleneck,
    /// Frequent function calls
    CallOverhead,
    /// Cache misses
    CacheMisses,
}

/// Optimization opportunity
#[derive(Debug, Clone)]
pub struct OptimizationOpportunity {
    /// Type of optimization
    pub optimization_type: OptimizationType,
    /// Location where optimization applies
    pub location: NodeId,
    /// Expected benefit (0.0 to 1.0)
    pub benefit: f64,
    /// Description of the optimization
    pub description: String,
}

/// Types of optimizations
#[derive(Debug, Clone)]
pub enum OptimizationType {
    /// Loop unrolling
    LoopUnrolling,
    /// Constant folding
    ConstantFolding,
    /// Common subexpression elimination
    CommonSubexpressionElimination,
    /// Dead code elimination
    DeadCodeElimination,
    /// Vectorization
    Vectorization,
}

/// Complexity metrics
#[derive(Debug, Clone)]
pub struct ComplexityMetrics {
    /// Cyclomatic complexity
    pub cyclomatic_complexity: usize,
    /// Number of variables
    pub variable_count: usize,
    /// Nesting depth
    pub nesting_depth: usize,
}

/// Complete abstract analysis result
#[derive(Debug, Clone)]
pub struct AbstractAnalysisResult {
    /// Forward analysis result
    pub forward_result: ForwardAnalysisResult,
    /// Backward analysis result (if performed)
    pub backward_result: Option<BackwardAnalysisResult>,
    /// Detected invariants
    pub invariants: Vec<Invariant>,
    /// Property verification results
    pub property_results: Vec<PropertyResult>,
    /// Precision analysis
    pub precision_analysis: PrecisionAnalysis,
    /// Performance analysis
    pub performance_analysis: PerformanceAnalysis,
    /// Analysis statistics
    pub statistics: AnalysisStatistics,
    /// Abstract values for each node
    pub node_values: HashMap<NodeId, AbstractValue>,
}

/// IR module analysis result
#[derive(Debug, Clone)]
pub struct AbstractIrResult {
    /// Results for individual functions
    pub function_results: HashMap<String, AbstractFunctionResult>,
    /// Interprocedural analysis result
    pub interprocedural_result: InterproceduralAnalysisResult,
    /// Module-level invariants
    pub module_invariants: Vec<ModuleInvariant>,
}

/// Abstract interpretation engine for static analysis
pub struct AbstractInterpreter {
    config: AbstractInterpretationConfig,
    domain_factory: AbstractDomainFactory,
    fixpoint_engine: FixpointEngine,
    invariant_detector: InvariantDetector,
    property_checker: PropertyChecker,
    analysis_cache: AnalysisCache,
}

impl AbstractInterpreter {
    /// Create a new abstract interpreter with specified configuration
    ///
    /// # Arguments
    /// * `config` - Configuration for the abstract interpretation
    ///
    /// # Returns
    /// * `Self` - New AbstractInterpreter instance
    pub fn new(config: AbstractInterpretationConfig) -> Self {
        Self {
            domain_factory: AbstractDomainFactory::new(),
            fixpoint_engine: FixpointEngine::new(config.clone()),
            invariant_detector: InvariantDetector::new(),
            property_checker: PropertyChecker::new(),
            analysis_cache: AnalysisCache::new(),
            config,
        }
    }

    /// Create an interpreter with default configuration
    pub fn with_defaults() -> Self {
        Self::new(AbstractInterpretationConfig::default())
    }

    /// Perform abstract interpretation on a computation graph
    ///
    /// This is the main entry point for analyzing computation graphs.
    /// It orchestrates the entire analysis process including forward/backward
    /// analysis, invariant detection, and property checking.
    ///
    /// # Arguments
    /// * `graph` - The computation graph to analyze
    ///
    /// # Returns
    /// * `JitResult<AbstractAnalysisResult>` - Complete analysis results
    pub fn analyze_graph(&mut self, graph: &ComputationGraph) -> JitResult<AbstractAnalysisResult> {
        let start_time = Instant::now();

        // Create abstract domain for the analysis
        let domain = self.domain_factory.create_domain(&self.config.domain_type);

        // Convert graph to abstract representation
        let abstract_graph = self.convert_to_abstract_graph(graph, domain.as_ref())?;

        // Perform forward analysis
        let forward_result = self.forward_analysis(&abstract_graph, domain.as_ref())?;

        // Perform backward analysis if enabled
        let backward_result = if self.config.enable_backward_analysis {
            Some(self.backward_analysis(&abstract_graph, domain.as_ref())?)
        } else {
            None
        };

        // Detect invariants
        let invariants = self
            .invariant_detector
            .detect_invariants(&forward_result, &backward_result)?;

        // Check properties
        let property_results = self
            .property_checker
            .check_properties(&forward_result, &self.config.properties)?;

        // Analyze precision and performance
        let precision_analysis = self.analyze_precision(&forward_result, &abstract_graph);
        let performance_analysis = self.analyze_performance(&forward_result, graph);

        let total_time = start_time.elapsed();
        let fixpoint_iterations = forward_result.iterations
            + backward_result
                .as_ref()
                .map(|r| r.iterations)
                .unwrap_or(0);
        let abstract_states_computed = forward_result.post_states.len();
        let node_values = forward_result.post_states.clone();

        Ok(AbstractAnalysisResult {
            forward_result,
            backward_result,
            invariants,
            property_results,
            precision_analysis,
            performance_analysis,
            statistics: AnalysisStatistics {
                analysis_time: total_time,
                fixpoint_iterations,
                abstract_states_computed,
                cache_hits: self.analysis_cache.hit_count(),
                cache_misses: self.analysis_cache.miss_count(),
            },
            node_values,
        })
    }

    /// Perform abstract interpretation on an IR module
    ///
    /// Analyzes an intermediate representation module, processing
    /// basic blocks and performing interprocedural analysis.
    ///
    /// # Arguments
    /// * `ir_module` - The IR module to analyze
    ///
    /// # Returns
    /// * `JitResult<AbstractIrResult>` - IR-specific analysis results
    pub fn analyze_ir(&mut self, ir_module: &IrModule) -> JitResult<AbstractIrResult> {
        let mut block_results = HashMap::new();
        let mut module_invariants = Vec::new();

        // Analyze each basic block
        for (block_id, block) in &ir_module.blocks {
            let block_result = self.analyze_block(block)?;
            block_results.insert(*block_id, block_result);
        }

        // Perform interprocedural analysis
        let interprocedural_result = self.interprocedural_analysis_blocks(&block_results)?;

        // Detect module-level invariants
        module_invariants.extend(self.detect_module_invariants_blocks(&block_results)?);

        Ok(AbstractIrResult {
            function_results: HashMap::new(), // No functions in this IR model
            interprocedural_result,
            module_invariants,
        })
    }

    /// Get the current configuration
    pub fn config(&self) -> &AbstractInterpretationConfig {
        &self.config
    }

    /// Update the configuration
    pub fn set_config(&mut self, config: AbstractInterpretationConfig) {
        self.config = config;
        self.fixpoint_engine = FixpointEngine::new(self.config.clone());
    }

    /// Clear the analysis cache
    pub fn clear_cache(&mut self) {
        self.analysis_cache.clear();
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> (usize, usize) {
        (
            self.analysis_cache.hit_count(),
            self.analysis_cache.miss_count(),
        )
    }
}

// Placeholder implementations for engine components
// These would be implemented in separate modules

/// Fixpoint computation engine
pub struct FixpointEngine {
    config: AbstractInterpretationConfig,
}

impl FixpointEngine {
    pub fn new(config: AbstractInterpretationConfig) -> Self {
        Self { config }
    }
}

/// Invariant detection engine.
///
/// Walks the post-states of a forward analysis (and the union with the
/// backward analysis if available) and reports an [`Invariant`] for
/// every node whose abstract value is precise enough to constrain the
/// concrete semantics — singleton intervals, pure signs, exact
/// constants, etc.
pub struct InvariantDetector;

impl InvariantDetector {
    pub fn new() -> Self {
        Self
    }

    /// Detect invariants from forward (and optional backward) results.
    ///
    /// The detector emits:
    /// - `NumericalProperty` for nodes with constant abstract value;
    /// - `ValueRange` for nodes pinned to a finite, non-singleton interval;
    /// - `MemorySafety` for nodes whose sign domain rules out a sign
    ///   (e.g. `NonNegative` rules out negative values).
    pub fn detect_invariants(
        &self,
        forward_result: &ForwardAnalysisResult,
        backward_result: &Option<BackwardAnalysisResult>,
    ) -> JitResult<Vec<Invariant>> {
        let mut invariants = Vec::new();
        for (node_id, value) in &forward_result.post_states {
            if let Some(inv) = describe_invariant(*node_id, value, "forward") {
                invariants.push(inv);
            }
        }
        if let Some(bw) = backward_result {
            for (node_id, value) in &bw.pre_states {
                // Avoid duplicates with forward by tagging the source.
                if let Some(inv) = describe_invariant(*node_id, value, "backward") {
                    invariants.push(inv);
                }
            }
        }
        Ok(invariants)
    }
}

/// Build an [`Invariant`] from a single abstract value, or return
/// `None` if the value is too imprecise to constrain anything.
fn describe_invariant(node_id: NodeId, value: &AbstractValue, source: &str) -> Option<Invariant> {
    match value {
        AbstractValue::Interval { min, max } => {
            if min == max && min.is_finite() {
                Some(Invariant {
                    invariant_type: InvariantType::NumericalProperty,
                    description: format!("node {:?} = {}", node_id, min),
                    confidence: 1.0,
                    location: source.to_string(),
                })
            } else if min.is_finite() && max.is_finite() {
                Some(Invariant {
                    invariant_type: InvariantType::ValueRange,
                    description: format!("node {:?} ∈ [{}, {}]", node_id, min, max),
                    confidence: 0.9,
                    location: source.to_string(),
                })
            } else {
                None
            }
        }
        AbstractValue::Constant(DomainConstantValue::Value(v)) => Some(Invariant {
            invariant_type: InvariantType::NumericalProperty,
            description: format!("node {:?} ≡ {}", node_id, v),
            confidence: 1.0,
            location: source.to_string(),
        }),
        AbstractValue::Sign(SignValue::Zero) => Some(Invariant {
            invariant_type: InvariantType::NumericalProperty,
            description: format!("node {:?} ≡ 0", node_id),
            confidence: 1.0,
            location: source.to_string(),
        }),
        AbstractValue::Sign(SignValue::NonNegative) => Some(Invariant {
            invariant_type: InvariantType::MemorySafety,
            description: format!("node {:?} ≥ 0", node_id),
            confidence: 0.95,
            location: source.to_string(),
        }),
        AbstractValue::Sign(SignValue::Positive) => Some(Invariant {
            invariant_type: InvariantType::MemorySafety,
            description: format!("node {:?} > 0", node_id),
            confidence: 0.95,
            location: source.to_string(),
        }),
        _ => None,
    }
}

impl Default for InvariantDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Property checking engine.
///
/// Verifies safety [`Property`] values against the post-states produced
/// by a forward analysis. Each check returns a [`PropertyResult`]
/// classifying the property as `Safe`, `Unsafe`, or `Unknown` along
/// with an analyst-readable explanation.
pub struct PropertyChecker;

impl PropertyChecker {
    pub fn new() -> Self {
        Self
    }

    /// Check every property against the corresponding node's abstract
    /// post-state.
    ///
    /// The interpretation of "safe" vs. "unsafe" depends on the
    /// property and the abstract domain:
    /// - For interval domains, a property is `Safe` iff every concrete
    ///   value reachable through the abstraction satisfies it.
    /// - For sign domains, signs that strictly imply the property are
    ///   `Safe`; signs that exclude it are `Unsafe`; the rest are
    ///   `Unknown`.
    pub fn check_properties(
        &self,
        forward_result: &ForwardAnalysisResult,
        properties: &[Property],
    ) -> JitResult<Vec<PropertyResult>> {
        let mut results = Vec::with_capacity(properties.len());
        for prop in properties {
            let node = property_node(prop);
            let state = forward_result.post_states.get(&node);
            results.push(check_property(prop, state));
        }
        Ok(results)
    }
}

/// Extract the [`NodeId`] guarded by a [`Property`].
fn property_node(prop: &Property) -> NodeId {
    match prop {
        Property::NonNegative(n)
        | Property::Positive(n)
        | Property::BoundedValue(n, _, _)
        | Property::NoDivisionByZero(n)
        | Property::NoOverflow(n)
        | Property::SafetyProperty(_, n) => *n,
    }
}

/// Decide a single property against its abstract value.
fn check_property(prop: &Property, value: Option<&AbstractValue>) -> PropertyResult {
    let value = match value {
        Some(v) => v,
        None => {
            return PropertyResult {
                property: prop.clone(),
                result: SafetyCheckResult::Unknown,
                confidence: 0.0,
                details: "no abstract value computed for node".to_string(),
            }
        }
    };
    match prop {
        Property::NonNegative(_) => decide_non_negative(prop, value),
        Property::Positive(_) => decide_positive(prop, value),
        Property::BoundedValue(_, lo, hi) => decide_bounded(prop, value, *lo, *hi),
        Property::NoDivisionByZero(_) => decide_nonzero(prop, value),
        Property::NoOverflow(_) => decide_no_overflow(prop, value),
        Property::SafetyProperty(_, _) => PropertyResult {
            property: prop.clone(),
            result: SafetyCheckResult::Unknown,
            confidence: 0.0,
            details: "custom safety property requires explicit verifier".to_string(),
        },
    }
}

fn decide_non_negative(prop: &Property, value: &AbstractValue) -> PropertyResult {
    match value {
        AbstractValue::Interval { min, max } => {
            if *min >= 0.0 {
                PropertyResult {
                    property: prop.clone(),
                    result: SafetyCheckResult::Safe,
                    confidence: 1.0,
                    details: "interval lower bound ≥ 0".to_string(),
                }
            } else if *max < 0.0 {
                PropertyResult {
                    property: prop.clone(),
                    result: SafetyCheckResult::Unsafe,
                    confidence: 1.0,
                    details: "interval upper bound < 0".to_string(),
                }
            } else {
                PropertyResult {
                    property: prop.clone(),
                    result: SafetyCheckResult::Unknown,
                    confidence: 0.5,
                    details: "interval straddles zero".to_string(),
                }
            }
        }
        AbstractValue::Sign(s) => match s {
            SignValue::Zero | SignValue::Positive | SignValue::NonNegative => PropertyResult {
                property: prop.clone(),
                result: SafetyCheckResult::Safe,
                confidence: 1.0,
                details: "sign domain proves non-negativity".to_string(),
            },
            SignValue::Negative => PropertyResult {
                property: prop.clone(),
                result: SafetyCheckResult::Unsafe,
                confidence: 1.0,
                details: "sign domain proves negativity".to_string(),
            },
            _ => PropertyResult {
                property: prop.clone(),
                result: SafetyCheckResult::Unknown,
                confidence: 0.5,
                details: "sign domain too imprecise".to_string(),
            },
        },
        AbstractValue::Constant(DomainConstantValue::Value(v)) => {
            if *v >= 0.0 {
                PropertyResult {
                    property: prop.clone(),
                    result: SafetyCheckResult::Safe,
                    confidence: 1.0,
                    details: format!("constant {} ≥ 0", v),
                }
            } else {
                PropertyResult {
                    property: prop.clone(),
                    result: SafetyCheckResult::Unsafe,
                    confidence: 1.0,
                    details: format!("constant {} < 0", v),
                }
            }
        }
        _ => PropertyResult {
            property: prop.clone(),
            result: SafetyCheckResult::Unknown,
            confidence: 0.0,
            details: "abstract domain insufficient".to_string(),
        },
    }
}

fn decide_positive(prop: &Property, value: &AbstractValue) -> PropertyResult {
    match value {
        AbstractValue::Interval { min, max } => {
            if *min > 0.0 {
                PropertyResult {
                    property: prop.clone(),
                    result: SafetyCheckResult::Safe,
                    confidence: 1.0,
                    details: "interval strictly positive".to_string(),
                }
            } else if *max <= 0.0 {
                PropertyResult {
                    property: prop.clone(),
                    result: SafetyCheckResult::Unsafe,
                    confidence: 1.0,
                    details: "interval entirely ≤ 0".to_string(),
                }
            } else {
                PropertyResult {
                    property: prop.clone(),
                    result: SafetyCheckResult::Unknown,
                    confidence: 0.5,
                    details: "interval includes 0 or negatives".to_string(),
                }
            }
        }
        AbstractValue::Sign(SignValue::Positive) => PropertyResult {
            property: prop.clone(),
            result: SafetyCheckResult::Safe,
            confidence: 1.0,
            details: "sign domain proves positivity".to_string(),
        },
        AbstractValue::Sign(SignValue::Zero) | AbstractValue::Sign(SignValue::Negative)
        | AbstractValue::Sign(SignValue::NonPositive) => PropertyResult {
            property: prop.clone(),
            result: SafetyCheckResult::Unsafe,
            confidence: 1.0,
            details: "sign domain proves non-positivity".to_string(),
        },
        AbstractValue::Constant(DomainConstantValue::Value(v)) => {
            if *v > 0.0 {
                PropertyResult {
                    property: prop.clone(),
                    result: SafetyCheckResult::Safe,
                    confidence: 1.0,
                    details: format!("constant {} > 0", v),
                }
            } else {
                PropertyResult {
                    property: prop.clone(),
                    result: SafetyCheckResult::Unsafe,
                    confidence: 1.0,
                    details: format!("constant {} ≤ 0", v),
                }
            }
        }
        _ => PropertyResult {
            property: prop.clone(),
            result: SafetyCheckResult::Unknown,
            confidence: 0.0,
            details: "abstract domain insufficient".to_string(),
        },
    }
}

fn decide_bounded(prop: &Property, value: &AbstractValue, lo: f64, hi: f64) -> PropertyResult {
    match value {
        AbstractValue::Interval { min, max } => {
            if *min >= lo && *max <= hi {
                PropertyResult {
                    property: prop.clone(),
                    result: SafetyCheckResult::Safe,
                    confidence: 1.0,
                    details: format!("interval [{}, {}] ⊆ [{}, {}]", min, max, lo, hi),
                }
            } else if *max < lo || *min > hi {
                PropertyResult {
                    property: prop.clone(),
                    result: SafetyCheckResult::Unsafe,
                    confidence: 1.0,
                    details: format!("interval disjoint from [{}, {}]", lo, hi),
                }
            } else {
                PropertyResult {
                    property: prop.clone(),
                    result: SafetyCheckResult::Unknown,
                    confidence: 0.5,
                    details: format!(
                        "interval [{}, {}] partially exceeds [{}, {}]",
                        min, max, lo, hi
                    ),
                }
            }
        }
        AbstractValue::Constant(DomainConstantValue::Value(v)) => {
            if *v >= lo && *v <= hi {
                PropertyResult {
                    property: prop.clone(),
                    result: SafetyCheckResult::Safe,
                    confidence: 1.0,
                    details: format!("constant {} ∈ [{}, {}]", v, lo, hi),
                }
            } else {
                PropertyResult {
                    property: prop.clone(),
                    result: SafetyCheckResult::Unsafe,
                    confidence: 1.0,
                    details: format!("constant {} ∉ [{}, {}]", v, lo, hi),
                }
            }
        }
        _ => PropertyResult {
            property: prop.clone(),
            result: SafetyCheckResult::Unknown,
            confidence: 0.0,
            details: "abstract domain insufficient for bounded check".to_string(),
        },
    }
}

fn decide_nonzero(prop: &Property, value: &AbstractValue) -> PropertyResult {
    match value {
        AbstractValue::Interval { min, max } => {
            if *min > 0.0 || *max < 0.0 {
                PropertyResult {
                    property: prop.clone(),
                    result: SafetyCheckResult::Safe,
                    confidence: 1.0,
                    details: "interval excludes zero".to_string(),
                }
            } else if *min == 0.0 && *max == 0.0 {
                PropertyResult {
                    property: prop.clone(),
                    result: SafetyCheckResult::Unsafe,
                    confidence: 1.0,
                    details: "interval is {0}".to_string(),
                }
            } else {
                PropertyResult {
                    property: prop.clone(),
                    result: SafetyCheckResult::Unknown,
                    confidence: 0.5,
                    details: "interval contains zero".to_string(),
                }
            }
        }
        AbstractValue::Sign(SignValue::Zero) => PropertyResult {
            property: prop.clone(),
            result: SafetyCheckResult::Unsafe,
            confidence: 1.0,
            details: "sign domain proves value is zero".to_string(),
        },
        AbstractValue::Sign(SignValue::Positive) | AbstractValue::Sign(SignValue::Negative) => {
            PropertyResult {
                property: prop.clone(),
                result: SafetyCheckResult::Safe,
                confidence: 1.0,
                details: "sign domain proves non-zero".to_string(),
            }
        }
        AbstractValue::Constant(DomainConstantValue::Value(v)) => {
            if *v != 0.0 {
                PropertyResult {
                    property: prop.clone(),
                    result: SafetyCheckResult::Safe,
                    confidence: 1.0,
                    details: format!("constant {} ≠ 0", v),
                }
            } else {
                PropertyResult {
                    property: prop.clone(),
                    result: SafetyCheckResult::Unsafe,
                    confidence: 1.0,
                    details: "constant is zero".to_string(),
                }
            }
        }
        _ => PropertyResult {
            property: prop.clone(),
            result: SafetyCheckResult::Unknown,
            confidence: 0.0,
            details: "abstract domain insufficient".to_string(),
        },
    }
}

fn decide_no_overflow(prop: &Property, value: &AbstractValue) -> PropertyResult {
    match value {
        AbstractValue::Interval { min, max } => {
            if min.is_finite() && max.is_finite() {
                PropertyResult {
                    property: prop.clone(),
                    result: SafetyCheckResult::Safe,
                    confidence: 1.0,
                    details: "interval is finite".to_string(),
                }
            } else {
                PropertyResult {
                    property: prop.clone(),
                    result: SafetyCheckResult::Unknown,
                    confidence: 0.3,
                    details: "interval is unbounded".to_string(),
                }
            }
        }
        AbstractValue::Constant(DomainConstantValue::Value(v)) => {
            if v.is_finite() {
                PropertyResult {
                    property: prop.clone(),
                    result: SafetyCheckResult::Safe,
                    confidence: 1.0,
                    details: format!("constant {} is finite", v),
                }
            } else {
                PropertyResult {
                    property: prop.clone(),
                    result: SafetyCheckResult::Unsafe,
                    confidence: 1.0,
                    details: "constant is not finite".to_string(),
                }
            }
        }
        _ => PropertyResult {
            property: prop.clone(),
            result: SafetyCheckResult::Unknown,
            confidence: 0.3,
            details: "abstract domain insufficient for overflow check".to_string(),
        },
    }
}

impl Default for PropertyChecker {
    fn default() -> Self {
        Self::new()
    }
}

/// Analysis result cache
pub struct AnalysisCache {
    hits: usize,
    misses: usize,
}

impl AnalysisCache {
    pub fn new() -> Self {
        Self { hits: 0, misses: 0 }
    }

    pub fn hit_count(&self) -> usize {
        self.hits
    }

    pub fn miss_count(&self) -> usize {
        self.misses
    }

    pub fn clear(&mut self) {
        self.hits = 0;
        self.misses = 0;
    }
}

impl Default for AnalysisCache {
    fn default() -> Self {
        Self::new()
    }
}

// Real analysis implementations
impl AbstractInterpreter {
    /// Translate a [`ComputationGraph`] into an [`AbstractGraph`] suitable
    /// for fixed-point analysis.
    ///
    /// The translation is a structural pass: every concrete node is
    /// classified into an [`AbstractNodeOp`] using
    /// [`Self::classify_operation`], and successor/predecessor adjacency
    /// is mirrored from the source graph. Entry nodes are nodes with no
    /// predecessors; exit nodes are those with no successors.
    fn convert_to_abstract_graph(
        &self,
        graph: &ComputationGraph,
        _domain: &dyn AbstractDomain,
    ) -> JitResult<AbstractGraph> {
        let mut abstract_graph = AbstractGraph::new();
        for (node_id, node) in graph.nodes() {
            let op = Self::classify_operation(&node.operation);
            abstract_graph.node_ops.insert(node_id, op);
            let preds = graph.get_node_inputs(node_id);
            let succs = graph.get_node_outputs(node_id);
            abstract_graph.predecessors.insert(node_id, preds);
            abstract_graph.successors.insert(node_id, succs);
        }
        for node_id in abstract_graph.node_ops.keys().copied().collect::<Vec<_>>() {
            let preds_empty = abstract_graph
                .predecessors
                .get(&node_id)
                .map(|v| v.is_empty())
                .unwrap_or(true);
            let succs_empty = abstract_graph
                .successors
                .get(&node_id)
                .map(|v| v.is_empty())
                .unwrap_or(true);
            if preds_empty {
                abstract_graph.entry_nodes.push(node_id);
            }
            if succs_empty {
                abstract_graph.exit_nodes.push(node_id);
            }
        }
        Ok(abstract_graph)
    }

    /// Map a concrete [`Operation`] to an [`AbstractNodeOp`].
    ///
    /// Operations without a precise abstract semantics in this framework
    /// fall through to [`AbstractNodeOp::Unknown`], which the transfer
    /// function maps to `domain.top()` — a sound over-approximation.
    fn classify_operation(op: &Operation) -> AbstractNodeOp {
        match op {
            Operation::Input | Operation::Parameter(_) => AbstractNodeOp::Input,
            Operation::Constant(info) => {
                let v = match &info.value {
                    OpConstantValue::Float(f) | OpConstantValue::Scalar(f) => Some(*f),
                    OpConstantValue::Int(i) | OpConstantValue::IntScalar(i) => Some(*i as f64),
                    OpConstantValue::UInt(u) => Some(*u as f64),
                    OpConstantValue::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
                    _ => None,
                };
                match v {
                    Some(value) => AbstractNodeOp::Constant(value),
                    None => AbstractNodeOp::Unknown,
                }
            }
            Operation::Add => AbstractNodeOp::Binary(BinaryAbstractOp::Add),
            Operation::Sub => AbstractNodeOp::Binary(BinaryAbstractOp::Sub),
            Operation::Mul => AbstractNodeOp::Binary(BinaryAbstractOp::Mul),
            Operation::Div => AbstractNodeOp::Binary(BinaryAbstractOp::Div),
            Operation::Neg => AbstractNodeOp::Unary(UnaryAbstractOp::Neg),
            Operation::Abs => AbstractNodeOp::Unary(UnaryAbstractOp::Abs),
            Operation::Sqrt => AbstractNodeOp::Unary(UnaryAbstractOp::Sqrt),
            Operation::Sin => AbstractNodeOp::Unary(UnaryAbstractOp::Sin),
            Operation::Cos => AbstractNodeOp::Unary(UnaryAbstractOp::Cos),
            Operation::Exp => AbstractNodeOp::Unary(UnaryAbstractOp::Exp),
            Operation::Log => AbstractNodeOp::Unary(UnaryAbstractOp::Log),
            _ => AbstractNodeOp::Unknown,
        }
    }

    /// Compute the abstract post-state of `node` by applying its transfer
    /// function over the current `state`.
    ///
    /// This is the abstract-semantics dispatcher used by both the
    /// forward worklist and the per-node propagation logic. Missing
    /// predecessor states default to `domain.bottom()`, which models
    /// "not yet visited" without polluting the join.
    fn transfer_node(
        node: NodeId,
        graph: &AbstractGraph,
        state: &HashMap<NodeId, AbstractValue>,
        domain: &dyn AbstractDomain,
    ) -> JitResult<AbstractValue> {
        let op = match graph.node_ops.get(&node) {
            Some(op) => op,
            None => return Ok(domain.top()),
        };
        match op {
            AbstractNodeOp::Input | AbstractNodeOp::Unknown => Ok(domain.top()),
            AbstractNodeOp::Constant(value) => domain.lift_constant(*value),
            AbstractNodeOp::Binary(bin_op) => {
                let preds = graph.predecessors_of(node);
                if preds.len() < 2 {
                    return Ok(domain.top());
                }
                let left = state
                    .get(&preds[0])
                    .cloned()
                    .unwrap_or_else(|| domain.bottom());
                let right = state
                    .get(&preds[1])
                    .cloned()
                    .unwrap_or_else(|| domain.bottom());
                domain.abstract_binary_op(*bin_op, &left, &right)
            }
            AbstractNodeOp::Unary(un_op) => {
                let preds = graph.predecessors_of(node);
                if preds.is_empty() {
                    return Ok(domain.top());
                }
                let operand = state
                    .get(&preds[0])
                    .cloned()
                    .unwrap_or_else(|| domain.bottom());
                domain.abstract_unary_op(*un_op, &operand)
            }
        }
    }

    /// Test abstract-domain equality via the partial order in both
    /// directions: `a == b ⇔ a ⊑ b ∧ b ⊑ a`.
    ///
    /// Used as the worklist convergence test because [`AbstractValue`]
    /// does not derive [`PartialEq`] for all variants.
    fn values_equal(a: &AbstractValue, b: &AbstractValue, domain: &dyn AbstractDomain) -> bool {
        domain.less_equal(a, b) && domain.less_equal(b, a)
    }

    /// Forward Kildall worklist with widening delay.
    ///
    /// Algorithm:
    /// 1. Initialize each node's pre-state to `domain.bottom()`.
    /// 2. Push all entry nodes onto the worklist.
    /// 3. Pop a node, compute its new post-state via [`Self::transfer_node`],
    ///    join it with the previous post-state, optionally widen if the
    ///    visit count exceeds `widening_delay`, and if the result differs,
    ///    schedule all successors.
    /// 4. Cap iterations at `config.max_iterations`; report convergence.
    fn forward_analysis(
        &mut self,
        graph: &AbstractGraph,
        domain: &dyn AbstractDomain,
    ) -> JitResult<ForwardAnalysisResult> {
        let mut result = ForwardAnalysisResult::new();
        let mut post_states: HashMap<NodeId, AbstractValue> = HashMap::new();
        let mut pre_states: HashMap<NodeId, AbstractValue> = HashMap::new();
        let mut visit_count: HashMap<NodeId, usize> = HashMap::new();

        for node in graph.nodes() {
            post_states.insert(node, domain.bottom());
            pre_states.insert(node, domain.bottom());
        }

        let mut worklist: VecDeque<NodeId> = if graph.entry_nodes.is_empty() {
            graph.nodes().collect()
        } else {
            graph.entry_nodes.iter().copied().collect()
        };
        let mut in_worklist: HashSet<NodeId> = worklist.iter().copied().collect();

        let mut iterations = 0usize;
        let max_iterations = self.config.max_iterations.max(1);
        let widening_delay = self.config.widening_delay;
        let mut converged = false;

        while let Some(node) = worklist.pop_front() {
            in_worklist.remove(&node);
            iterations += 1;
            if iterations > max_iterations {
                break;
            }

            // Pre-state = join of predecessors' post-states.
            let pre = Self::join_predecessor_states(node, graph, &post_states, domain)?;
            pre_states.insert(node, pre);

            // Compute the new post-state via the transfer function.
            let transferred = Self::transfer_node(node, graph, &post_states, domain)?;
            let old_post = post_states
                .get(&node)
                .cloned()
                .unwrap_or_else(|| domain.bottom());

            let visits = visit_count.entry(node).or_insert(0);
            *visits += 1;
            let new_post = if widening_delay > 0 && *visits > widening_delay {
                domain.widen(&old_post, &transferred)?
            } else {
                domain.join(&old_post, &transferred)?
            };

            if !Self::values_equal(&new_post, &old_post, domain) {
                post_states.insert(node, new_post);
                for succ in graph.successors_of(node) {
                    if in_worklist.insert(*succ) {
                        worklist.push_back(*succ);
                    }
                }
            }

            if worklist.is_empty() {
                converged = true;
                break;
            }
        }

        // Optional narrowing pass to refine widened states.
        if self.config.enable_narrowing {
            for node in graph.nodes() {
                let transferred = Self::transfer_node(node, graph, &post_states, domain)?;
                let current = post_states
                    .get(&node)
                    .cloned()
                    .unwrap_or_else(|| domain.bottom());
                let narrowed = domain.narrow(&current, &transferred)?;
                post_states.insert(node, narrowed);
            }
        }

        result.pre_states = pre_states;
        result.post_states = post_states;
        result.iterations = iterations;
        result.converged = converged && iterations <= max_iterations;
        Ok(result)
    }

    /// Compute the join of the post-states of `node`'s predecessors.
    ///
    /// Used to derive a node's pre-state in forward analysis. Returns
    /// `domain.bottom()` for entry nodes (no predecessors).
    fn join_predecessor_states(
        node: NodeId,
        graph: &AbstractGraph,
        post_states: &HashMap<NodeId, AbstractValue>,
        domain: &dyn AbstractDomain,
    ) -> JitResult<AbstractValue> {
        let preds = graph.predecessors_of(node);
        if preds.is_empty() {
            return Ok(domain.bottom());
        }
        let mut acc = domain.bottom();
        for pred in preds {
            let pred_state = post_states
                .get(pred)
                .cloned()
                .unwrap_or_else(|| domain.bottom());
            acc = domain.join(&acc, &pred_state)?;
        }
        Ok(acc)
    }

    /// Backward analysis: same Kildall fixed-point structure as
    /// [`Self::forward_analysis`] but propagating from exits to entries.
    ///
    /// For now this is a structural symmetric pass — each node's
    /// post-state is the join of its successors' pre-states; the
    /// transfer function is the identity (the framework does not yet
    /// model precise backward semantics for arithmetic ops). This is a
    /// sound under-approximation of usable backward information.
    fn backward_analysis(
        &mut self,
        graph: &AbstractGraph,
        domain: &dyn AbstractDomain,
    ) -> JitResult<BackwardAnalysisResult> {
        let mut result = BackwardAnalysisResult::new();
        let mut pre_states: HashMap<NodeId, AbstractValue> = HashMap::new();
        let mut post_states: HashMap<NodeId, AbstractValue> = HashMap::new();
        let mut visit_count: HashMap<NodeId, usize> = HashMap::new();

        for node in graph.nodes() {
            pre_states.insert(node, domain.bottom());
            post_states.insert(node, domain.top());
        }

        let mut worklist: VecDeque<NodeId> = if graph.exit_nodes.is_empty() {
            graph.nodes().collect()
        } else {
            graph.exit_nodes.iter().copied().collect()
        };
        let mut in_worklist: HashSet<NodeId> = worklist.iter().copied().collect();

        let mut iterations = 0usize;
        let max_iterations = self.config.max_iterations.max(1);
        let widening_delay = self.config.widening_delay;
        let mut converged = false;

        while let Some(node) = worklist.pop_front() {
            in_worklist.remove(&node);
            iterations += 1;
            if iterations > max_iterations {
                break;
            }

            // Post-state = join of successors' pre-states.
            let succs = graph.successors_of(node);
            let post = if succs.is_empty() {
                domain.top()
            } else {
                let mut acc = domain.bottom();
                for s in succs {
                    let s_state = pre_states
                        .get(s)
                        .cloned()
                        .unwrap_or_else(|| domain.bottom());
                    acc = domain.join(&acc, &s_state)?;
                }
                acc
            };
            post_states.insert(node, post.clone());

            // Backward transfer: identity (sound).
            let old_pre = pre_states
                .get(&node)
                .cloned()
                .unwrap_or_else(|| domain.bottom());
            let visits = visit_count.entry(node).or_insert(0);
            *visits += 1;
            let new_pre = if widening_delay > 0 && *visits > widening_delay {
                domain.widen(&old_pre, &post)?
            } else {
                domain.join(&old_pre, &post)?
            };

            if !Self::values_equal(&new_pre, &old_pre, domain) {
                pre_states.insert(node, new_pre);
                for pred in graph.predecessors_of(node) {
                    if in_worklist.insert(*pred) {
                        worklist.push_back(*pred);
                    }
                }
            }

            if worklist.is_empty() {
                converged = true;
                break;
            }
        }

        result.pre_states = pre_states;
        result.post_states = post_states;
        result.iterations = iterations;
        result.converged = converged && iterations <= max_iterations;
        Ok(result)
    }

    /// Score the precision of a forward analysis result.
    ///
    /// Per-node precision is taken from [`AbstractValue::precision`].
    /// The overall precision is the unweighted average over all nodes.
    /// Nodes whose precision is below `config.precision_threshold` are
    /// reported as improvement opportunities.
    fn analyze_precision(
        &self,
        forward_result: &ForwardAnalysisResult,
        graph: &AbstractGraph,
    ) -> PrecisionAnalysis {
        let mut node_precision = HashMap::new();
        let mut total = 0.0f64;
        let mut count = 0usize;
        let mut suggestions = Vec::new();

        for node in graph.nodes() {
            let val = forward_result.post_states.get(&node);
            let p = val.map(|v| v.precision()).unwrap_or(0.0);
            node_precision.insert(node, p);
            total += p;
            count += 1;
            if p < self.config.precision_threshold {
                suggestions.push(format!(
                    "node {:?}: low precision ({:.2}); consider richer abstract domain",
                    node, p
                ));
            }
        }
        let overall_precision = if count == 0 { 0.0 } else { total / count as f64 };
        PrecisionAnalysis {
            overall_precision,
            node_precision,
            improvement_suggestions: suggestions,
        }
    }

    /// Heuristic performance analysis driven by node operation
    /// categories and abstract states.
    ///
    /// Bottlenecks are flagged when a node is computation-intensive
    /// (matrix multiply, convolution, reductions) or when its abstract
    /// value is fully unknown (e.g. division by an interval containing
    /// zero), and optimization opportunities surface for nodes with
    /// fully-determined abstract values (constant folding).
    fn analyze_performance(
        &self,
        forward_result: &ForwardAnalysisResult,
        graph: &ComputationGraph,
    ) -> PerformanceAnalysis {
        let mut bottlenecks = Vec::new();
        let mut opportunities = Vec::new();
        let mut weight = 0.0f64;
        let total_nodes = graph.node_count().max(1) as f64;

        for (node_id, node) in graph.nodes() {
            let complexity = node.complexity_estimate() as f64;
            weight += complexity;
            let category = node.operation_category();
            if matches!(
                category,
                crate::graph::core::OperationCategory::LinearAlgebra
                    | crate::graph::core::OperationCategory::NeuralNetwork
                    | crate::graph::core::OperationCategory::Reduction
            ) {
                bottlenecks.push(PerformanceBottleneck {
                    bottleneck_type: BottleneckType::ComputationIntensive,
                    location: node_id,
                    severity: (complexity / 1000.0).min(1.0),
                    description: format!(
                        "{:?} is computation-intensive",
                        node.operation_type()
                    ),
                });
            }

            if let Some(value) = forward_result.post_states.get(&node_id) {
                if value.is_constant() {
                    opportunities.push(OptimizationOpportunity {
                        optimization_type: OptimizationType::ConstantFolding,
                        location: node_id,
                        benefit: 0.9,
                        description: "abstract value is a constant; fold at compile time"
                            .to_string(),
                    });
                }
            }
        }

        let complexity_score = (weight / (total_nodes * 100.0)).min(1.0);
        PerformanceAnalysis {
            complexity_score,
            bottlenecks,
            optimization_opportunities: opportunities,
        }
    }

    /// Analyze a single basic block as a straight-line program.
    ///
    /// Each instruction is processed in order; we maintain a map from
    /// IR values to abstract values, and compute entry/exit states as
    /// the join over all known values at the corresponding boundary.
    /// Convergence is trivial for straight-line code (one pass).
    fn analyze_block(&mut self, block: &BasicBlock) -> JitResult<AbstractFunctionResult> {
        let domain = self.domain_factory.create_domain(&self.config.domain_type);
        let mut value_state: HashMap<IrValue, AbstractValue> = HashMap::new();

        // Block params seed at top (unknown a priori).
        for p in &block.params {
            value_state.insert(*p, domain.top());
        }

        let mut iterations = 0usize;
        for instr in &block.instructions {
            iterations += 1;
            let abstract_value =
                Self::transfer_instruction(instr, &value_state, domain.as_ref())?;
            if let Some(result) = instr.result {
                value_state.insert(result, abstract_value);
            }
        }

        let entry_state = Self::join_value_state(&value_state, domain.as_ref(), true)?;
        let exit_state = Self::join_value_state(&value_state, domain.as_ref(), false)?;

        let invariants = Self::block_invariants(block, &value_state);

        let forward_result = FunctionForwardResult {
            converged: true,
            iterations,
            entry_state: Some(entry_state),
            exit_state: Some(exit_state.clone()),
        };

        let backward_result = if self.config.enable_backward_analysis {
            Some(FunctionBackwardResult {
                converged: true,
                iterations,
                entry_state: Some(domain.top()),
                exit_state: Some(exit_state),
            })
        } else {
            None
        };

        Ok(AbstractFunctionResult {
            forward_result,
            backward_result,
            invariants,
        })
    }

    /// Apply the abstract semantics of a single IR instruction.
    ///
    /// Operands are looked up in `state`; missing operands fall back to
    /// `domain.top()`. Constants are lifted via [`AbstractDomain::lift_constant`];
    /// arithmetic opcodes dispatch to [`AbstractDomain::abstract_binary_op`]
    /// and [`AbstractDomain::abstract_unary_op`]; everything else is
    /// approximated as `top` (sound).
    fn transfer_instruction(
        instr: &crate::ir::Instruction,
        state: &HashMap<IrValue, AbstractValue>,
        domain: &dyn AbstractDomain,
    ) -> JitResult<AbstractValue> {
        let lookup = |v: &IrValue| -> AbstractValue {
            state.get(v).cloned().unwrap_or_else(|| domain.top())
        };

        match &instr.opcode {
            IrOpcode::Const => match instr.attrs.get("value") {
                Some(crate::ir::IrAttribute::Float(value)) => domain.lift_constant(*value),
                Some(crate::ir::IrAttribute::Int(value)) => domain.lift_constant(*value as f64),
                Some(crate::ir::IrAttribute::Bool(value)) => {
                    domain.lift_constant(if *value { 1.0 } else { 0.0 })
                }
                _ => Ok(domain.top()),
            },
            IrOpcode::Add | IrOpcode::Sub | IrOpcode::Mul | IrOpcode::Div => {
                if instr.operands.len() < 2 {
                    return Ok(domain.top());
                }
                let l = lookup(&instr.operands[0]);
                let r = lookup(&instr.operands[1]);
                let op = match instr.opcode {
                    IrOpcode::Add => BinaryAbstractOp::Add,
                    IrOpcode::Sub => BinaryAbstractOp::Sub,
                    IrOpcode::Mul => BinaryAbstractOp::Mul,
                    IrOpcode::Div => BinaryAbstractOp::Div,
                    _ => unreachable!("matched outer opcode"),
                };
                domain.abstract_binary_op(op, &l, &r)
            }
            IrOpcode::Neg | IrOpcode::Abs | IrOpcode::Sqrt | IrOpcode::Sin | IrOpcode::Cos
            | IrOpcode::Exp | IrOpcode::Log => {
                if instr.operands.is_empty() {
                    return Ok(domain.top());
                }
                let operand = lookup(&instr.operands[0]);
                let op = match instr.opcode {
                    IrOpcode::Neg => UnaryAbstractOp::Neg,
                    IrOpcode::Abs => UnaryAbstractOp::Abs,
                    IrOpcode::Sqrt => UnaryAbstractOp::Sqrt,
                    IrOpcode::Sin => UnaryAbstractOp::Sin,
                    IrOpcode::Cos => UnaryAbstractOp::Cos,
                    IrOpcode::Exp => UnaryAbstractOp::Exp,
                    IrOpcode::Log => UnaryAbstractOp::Log,
                    _ => unreachable!("matched outer opcode"),
                };
                domain.abstract_unary_op(op, &operand)
            }
            _ => Ok(domain.top()),
        }
    }

    /// Compute the join (or initial value) over a value-state map.
    ///
    /// `is_entry == true` returns `domain.top()` (no constraints at entry).
    /// `is_entry == false` returns the join of all values, modeling the
    /// post-state of the block.
    fn join_value_state(
        state: &HashMap<IrValue, AbstractValue>,
        domain: &dyn AbstractDomain,
        is_entry: bool,
    ) -> JitResult<AbstractValue> {
        if is_entry || state.is_empty() {
            return Ok(domain.top());
        }
        let mut acc = domain.bottom();
        for v in state.values() {
            acc = domain.join(&acc, v)?;
        }
        Ok(acc)
    }

    /// Detect block-level invariants (constant values).
    ///
    /// For each value whose abstract state is a singleton (interval with
    /// `min == max`, sign `Zero`, or constant `Value`), report a
    /// numerical-property invariant.
    fn block_invariants(
        block: &BasicBlock,
        state: &HashMap<IrValue, AbstractValue>,
    ) -> Vec<FunctionInvariant> {
        let mut invariants = Vec::new();
        for (value, av) in state {
            if av.is_constant() {
                invariants.push(FunctionInvariant {
                    invariant_type: InvariantType::NumericalProperty,
                    description: format!("value {:?} is constant", value),
                    confidence: 1.0,
                    location: format!("block {}", block.id),
                });
            }
        }
        invariants
    }

    /// Build the per-function call graph + global invariants from
    /// per-block analysis results.
    ///
    /// At the IR-module level used here, blocks are not separable
    /// functions, so the call graph is a single synthetic entry "module"
    /// pointing to all analyzed blocks; global invariants are the union
    /// of block invariants whose confidence is high.
    fn interprocedural_analysis_blocks(
        &self,
        block_results: &HashMap<BlockId, AbstractFunctionResult>,
    ) -> JitResult<InterproceduralAnalysisResult> {
        let mut call_graph: HashMap<String, Vec<String>> = HashMap::new();
        let mut global_invariants = Vec::new();
        let block_names: Vec<String> = block_results
            .keys()
            .map(|id| format!("block_{}", id))
            .collect();
        call_graph.insert("module".to_string(), block_names);

        for (block_id, res) in block_results {
            for inv in &res.invariants {
                if inv.confidence >= self.config.precision_threshold {
                    global_invariants.push(Invariant {
                        invariant_type: inv.invariant_type.clone(),
                        description: format!("block {}: {}", block_id, inv.description),
                        confidence: inv.confidence,
                        location: inv.location.clone(),
                    });
                }
            }
        }
        Ok(InterproceduralAnalysisResult {
            call_graph,
            global_invariants,
        })
    }

    /// Lift per-block invariants to module scope.
    ///
    /// A module-level invariant is reported whenever the same kind of
    /// per-block invariant holds in *every* analyzed block — e.g. all
    /// blocks return constants, in which case the module is functionally
    /// constant.
    fn detect_module_invariants_blocks(
        &self,
        block_results: &HashMap<BlockId, AbstractFunctionResult>,
    ) -> JitResult<Vec<ModuleInvariant>> {
        if block_results.is_empty() {
            return Ok(Vec::new());
        }
        let mut all_constant = true;
        for res in block_results.values() {
            let is_block_constant = res
                .forward_result
                .exit_state
                .as_ref()
                .map(AbstractValue::is_constant)
                .unwrap_or(false);
            if !is_block_constant {
                all_constant = false;
                break;
            }
        }
        let mut invariants = Vec::new();
        if all_constant {
            invariants.push(ModuleInvariant {
                invariant_type: InvariantType::NumericalProperty,
                description: "all blocks have constant exit state".to_string(),
                confidence: 1.0,
                scope: "module".to_string(),
            });
        }
        Ok(invariants)
    }
}

/// Abstract operation kind associated with a node, used to dispatch
/// transfer functions over an [`AbstractDomain`].
///
/// This enum decouples the high-level [`Operation`] vocabulary from
/// the abstract semantics, keeping the analysis simple and total: any
/// concrete op that does not have a known abstract semantics is mapped
/// to [`AbstractNodeOp::Unknown`] and conservatively transferred to
/// [`AbstractDomain::top`].
#[derive(Debug, Clone)]
pub enum AbstractNodeOp {
    /// Input or parameter node — its abstract value is the domain top
    /// (no precondition known a priori).
    Input,
    /// A constant node whose concrete value is `value`.
    Constant(f64),
    /// A binary arithmetic/comparison operation.
    Binary(BinaryAbstractOp),
    /// A unary arithmetic/intrinsic operation.
    Unary(UnaryAbstractOp),
    /// Operation with no precise abstract semantics in this framework;
    /// always transferred to top (sound over-approximation).
    Unknown,
}

/// Abstract representation of a [`ComputationGraph`] used by the
/// abstract interpreter.
///
/// The structure stores per-node operation kinds plus pre-computed
/// successor/predecessor adjacency in topological-friendly form so the
/// Kildall fixed-point worklist can iterate without paying for repeated
/// graph queries.
#[derive(Debug, Clone, Default)]
pub struct AbstractGraph {
    /// Operation kind for each node.
    pub node_ops: HashMap<NodeId, AbstractNodeOp>,
    /// Operands (predecessors) of each node, in input order.
    pub predecessors: HashMap<NodeId, Vec<NodeId>>,
    /// Successors of each node.
    pub successors: HashMap<NodeId, Vec<NodeId>>,
    /// Nodes with no predecessors (entries to the analysis).
    pub entry_nodes: Vec<NodeId>,
    /// Nodes with no successors (analysis sinks).
    pub exit_nodes: Vec<NodeId>,
}

impl AbstractGraph {
    /// Create an empty abstract graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Iterate over all nodes in the graph.
    pub fn nodes(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.node_ops.keys().copied()
    }

    /// Number of nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.node_ops.len()
    }

    /// Successors of `node` (empty slice if none).
    pub fn successors_of(&self, node: NodeId) -> &[NodeId] {
        self.successors
            .get(&node)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// Predecessors of `node` (empty slice if none).
    pub fn predecessors_of(&self, node: NodeId) -> &[NodeId] {
        self.predecessors
            .get(&node)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }
}
