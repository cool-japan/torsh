//! ToRSh JIT compilation and kernel fusion module
//!
//! This module provides Just-In-Time (JIT) compilation capabilities for ToRSh,
//! enabling automatic kernel fusion and optimization of computational graphs.
//!
//! # Features
//!
//! - **Kernel Fusion**: Automatically fuses compatible operations to reduce memory bandwidth
//! - **Graph Optimization**: Applies various optimization passes to the computation graph
//! - **Multiple Backends**: Supports Cranelift and (future) MLIR code generation
//! - **TorchScript-like API**: Compatible with PyTorch's JIT compilation model
//!
//! # Example
//!
//! ```rust,ignore
//! use torsh_jit::{jit_compile, FusionStrategy};
//!
//! // Define a model
//! let model = MyModel::new();
//!
//! // JIT compile with fusion enabled
//! let jit_model = jit_compile(model, FusionStrategy::Aggressive)?;
//!
//! // Use the JIT-compiled model
//! let output = jit_model.forward(input);
//! ```

// Note: Some warnings are allowed for experimental/incomplete features
#![allow(dead_code)] // Many public APIs not used internally
#![allow(unused_variables)] // Placeholder parameters in some implementations

use thiserror::Error;
use torsh_core::{DType, DeviceType, TorshError};

pub mod abstract_interpretation;
pub mod adaptive_compilation;
pub mod advisor;
pub mod analysis;
pub mod benchmarking;
pub mod codegen;
pub mod const_eval;
pub mod cranelift_backend;
pub mod custom_ops;
pub mod debug_symbols;
pub mod debugger;
pub mod differentiable_compilation;
pub mod error_diagnostics;
pub mod fusion;
pub mod generics;
pub mod graph;
pub mod hardware_tuning;
pub mod ir;
pub mod llvm_backend;
pub mod lowering;
pub mod metaprogramming;
pub mod mlir_backend;
pub mod neural_compilation;
pub mod optimization_advisor;
pub mod optimizer;
pub mod partial_evaluation;
pub mod pgo;
pub mod plugin_system;
pub mod polyhedral_optimization;
pub mod probabilistic_compilation;
pub mod profiler;
pub mod program_synthesis;
pub mod runtime;
pub mod script;
pub mod specialization;
pub mod speculative_optimization;
pub mod symbolic_execution;
pub mod trace_viz;
pub mod tracing;
pub mod type_inference;

#[cfg(test)]
pub mod compilation_test;

// Re-exports
pub use abstract_interpretation::{
    AbstractAnalysisResult, AbstractDomain, AbstractInterpretationConfig, AbstractInterpreter,
    AbstractValue, ConstantDomain, IntervalDomain, SignDomain,
};
pub use adaptive_compilation::{
    AdaptiveCompiler, AdaptiveConfig, CompilationStrategy, PerformanceMetrics,
};
pub use codegen::CodeGenerator;
pub use const_eval::{ConstEvalConfig, ConstantEvaluator, ConstantValue, EvaluationResult};
pub use custom_ops::{get_custom_op, list_custom_ops, register_custom_op, CustomOpBuilder};
pub use debug_symbols::{DebugSymbolConfig, DebugSymbolManager, SourceLocation, SymbolTable};
pub use debugger::{
    BreakpointLocation, DebugCommand, DebugSession, DebugState, DebugValue, DebuggerConfig,
    ExecutionLocation, InspectionTarget, JitDebugger,
};
pub use differentiable_compilation::{
    CompilationParams, CompilationTrainer, DiffCompilationResult, DifferentiableCompiler,
    GumbelSoftmax, PerformanceMetrics as DiffPerformanceMetrics, SoftDecision,
};
pub use error_diagnostics::{
    DiagnosticError, ErrorCategory, ErrorDiagnosticsManager, ErrorSeverity,
};
pub use fusion::{FusionStrategy, KernelFusion};
pub use generics::{
    create_type_param, shape_constraint, trait_constraint, GenericFunctionManager,
    GenericFunctionTemplate, ParameterKind, TypeConstraint, TypeParameter,
};
pub use graph::{ComputationGraph, Edge, Node, NodeId};
pub use hardware_tuning::{
    Architecture, HardwareInfo, HardwareTuner, HardwareTuningConfig, TuningRecommendation,
};
pub use llvm_backend::{LlvmBackend, LlvmOptimizer};
pub use metaprogramming::{
    CodeTemplate, DynamicCodeGenerator, GeneratedCode, GraphReflection, MacroDefinition,
    MetaprogrammingEngine, RuntimeReflector, TemplateParameters,
};
pub use mlir_backend::{MlirBackend, MlirOptimizer, MlirPass};
pub use neural_compilation::{
    CompilationStrategy as NeuralCompilationStrategy, GraphFeatures, NeuralCompiler,
    NeuralCompilerConfig, OptimizationDecision,
};
pub use optimizer::GraphOptimizer;
pub use partial_evaluation::{
    ConstantFolder, EvaluationStatistics, FunctionSpecializer, OptimizedGraph, OptimizedIrModule,
    PartialEvalConfig, PartialEvaluator,
};
pub use pgo::{
    OptimizationRecommendation as PgoRecommendation, OptimizationType as PgoOptimizationType,
    PgoConfig, ProfileGuidedOptimizer,
};
pub use plugin_system::{
    load_all_plugins, load_plugin, Plugin, PluginCapability, PluginManager, PluginMetadata,
    PluginRegistry,
};
pub use polyhedral_optimization::{
    AffineExpr, AffineSchedule, LoopNest, PolyhedralConfig, PolyhedralOptimizer, Polyhedron,
    TransformationMatrix, TransformationType,
};
pub use probabilistic_compilation::{
    BetaDistribution, MonteCarloResult, NormalDistribution, ProbabilisticCompilationResult,
    ProbabilisticCompiler, ProbabilisticConfig, ProbabilisticPerformance, UncertainDecision,
};
pub use profiler::{PerformanceEvent, ProfilerConfig, ProfilerManager, ProfilingSession};
pub use program_synthesis::{
    ExampleBuilder, ProgramSynthesizer, SynthesisExample, SynthesisResult, SynthesisStrategy,
    SynthesisTemplate, SynthesisValue,
};
pub use runtime::JitRuntime;
pub use script::{export_torchscript, import_torchscript, ScriptCompiler};
pub use specialization::{
    create_specialized_type, SpecializationConfig, SpecializedType, TypeSpecializer,
};
pub use speculative_optimization::{
    DeoptimizationEvent, SpeculationResult, SpeculativeConfig, SpeculativeOptimizer,
};
pub use symbolic_execution::{
    Constraint, ConstraintSet, ExecutionState, SymbolicExecutionConfig, SymbolicExecutionEngine,
    SymbolicExecutionResult, SymbolicGraph, SymbolicValue,
};
pub use trace_viz::{
    TraceEvent, TraceVisualizationManager, VisualizationConfig, VisualizationSession,
};

// Optimization advisor system (new modular architecture)
pub use optimization_advisor::{
    analyze_computation_graph, analyze_with_benchmarks, analyze_with_profiling, create_advisor,
    create_advisor_with_config, create_fast_config, create_production_config,
    create_thorough_config, quick_analyze, AdvisorConfig, AnalysisInput, CostBenefitAnalysis,
    OptimizationAdvisor, OptimizationRecommendation, OptimizationReport, OptimizationType,
    PatternAnalysis, PerformanceAnalysis, SystemConstraints, TargetPlatform, UserPreferences,
};

// Compatibility type aliases for legacy code (temporary until refactoring is complete)
pub type IrFunction = ir::IrModule; // Placeholder: Functions are represented as modules
pub type IrInstruction = ir::Instruction; // Direct alias

/// JIT compilation errors
#[derive(Error, Debug)]
pub enum JitError {
    #[error("Graph construction failed: {0}")]
    GraphError(String),

    #[error("Fusion error: {0}")]
    FusionError(String),

    #[error("Code generation failed: {0}")]
    CodeGenError(String),

    #[error("Optimization error: {0}")]
    OptimizationError(String),

    #[error("Runtime error: {0}")]
    RuntimeError(String),

    #[error("Unsupported operation: {0}")]
    UnsupportedOp(String),

    #[error("Compilation error: {0}")]
    CompilationError(String),

    #[error("Analysis error: {0}")]
    AnalysisError(String),

    #[error("Abstract interpretation error: {0}")]
    AbstractInterpretationError(String),

    #[error("Backend error: {0}")]
    BackendError(#[from] TorshError),
}

impl From<String> for JitError {
    fn from(msg: String) -> Self {
        JitError::RuntimeError(msg)
    }
}

pub type JitResult<T> = Result<T, JitError>;

/// JIT compilation configuration
#[derive(Debug, Clone)]
pub struct JitConfig {
    /// Fusion strategy to use
    pub fusion_strategy: FusionStrategy,

    /// Enable graph optimization passes
    pub enable_optimizations: bool,

    /// Maximum fusion group size
    pub max_fusion_size: usize,

    /// Enable profiling
    pub enable_profiling: bool,

    /// Target device for code generation
    pub target_device: DeviceType,

    /// Cache compiled kernels
    pub enable_caching: bool,

    /// Enable type specialization
    pub enable_specialization: bool,

    /// Type specialization configuration
    pub specialization_config: SpecializationConfig,
}

impl Default for JitConfig {
    fn default() -> Self {
        Self {
            fusion_strategy: FusionStrategy::Default,
            enable_optimizations: true,
            max_fusion_size: 8,
            enable_profiling: false,
            target_device: DeviceType::Cpu,
            enable_caching: true,
            enable_specialization: true,
            specialization_config: SpecializationConfig::default(),
        }
    }
}

/// Main JIT compiler interface
pub struct JitCompiler {
    config: JitConfig,
    runtime: JitRuntime,
    specializer: TypeSpecializer,
    generics: GenericFunctionManager,
    debug_symbols: DebugSymbolManager,
    profiler: ProfilerManager,
    trace_viz: TraceVisualizationManager,
    error_diagnostics: ErrorDiagnosticsManager,
}

impl JitCompiler {
    /// Create a new JIT compiler with the given configuration
    pub fn new(config: JitConfig) -> Self {
        Self {
            runtime: JitRuntime::new(config.clone()),
            specializer: TypeSpecializer::new(config.specialization_config.clone()),
            generics: GenericFunctionManager::with_defaults(),
            debug_symbols: DebugSymbolManager::with_defaults(),
            profiler: ProfilerManager::with_defaults(),
            trace_viz: TraceVisualizationManager::with_defaults(),
            error_diagnostics: ErrorDiagnosticsManager::with_defaults(),
            config,
        }
    }

    /// Compile a computation graph
    pub fn compile(&mut self, graph: ComputationGraph) -> JitResult<CompiledModule> {
        // Validate input graph
        graph
            .validate()
            .map_err(|e| JitError::GraphError(format!("{:?}", e)))?;

        // Apply type and shape inference
        let inferred_graph = self.apply_type_shape_inference(graph)?;

        // Apply optimization passes
        let optimized_graph = if self.config.enable_optimizations {
            let optimizer = GraphOptimizer::new();
            optimizer.optimize(inferred_graph)?
        } else {
            inferred_graph
        };

        // Apply kernel fusion
        let fusion = KernelFusion::new(self.config.fusion_strategy.clone());
        let fused_graph = fusion.apply(optimized_graph)?;

        // Lower to IR
        let ir_module = crate::lowering::lower_graph_to_ir(&fused_graph, "jit_module".to_string())?;

        // Apply IR-level optimizations
        let optimized_ir = self.apply_ir_optimizations(ir_module)?;

        // Generate code
        let compiled_kernels = self.generate_code(&optimized_ir)?;

        // Create compiled module
        Ok(CompiledModule {
            graph: fused_graph,
            kernels: compiled_kernels,
            runtime: self.runtime.clone(),
        })
    }

    /// Apply type and shape inference to the graph
    fn apply_type_shape_inference(
        &self,
        mut graph: ComputationGraph,
    ) -> JitResult<ComputationGraph> {
        use crate::type_inference::{ShapeInference, TypeInference};

        // Perform type inference
        let mut type_inf = TypeInference::new();
        type_inf.infer_types(&graph)?;

        // Perform shape inference
        let mut shape_inf = ShapeInference::new();
        shape_inf.infer_shapes(&graph)?;

        // Update graph with inferred information
        let node_ids: Vec<_> = graph.nodes().map(|(id, _)| id).collect();
        for node_id in node_ids {
            if let Some(inferred_type) = type_inf.get_type(node_id) {
                if let Some(node_mut) = graph.node_mut(node_id) {
                    node_mut.dtype = inferred_type;
                }
            }
            if let Some(inferred_shape) = shape_inf.get_shape(node_id) {
                if let Some(node_mut) = graph.node_mut(node_id) {
                    node_mut.output_shape = inferred_shape.clone();
                }
            }
        }

        Ok(graph)
    }

    /// Apply IR-level optimizations
    fn apply_ir_optimizations(
        &self,
        mut ir_module: crate::ir::IrModule,
    ) -> JitResult<crate::ir::IrModule> {
        use crate::lowering::{IrConstantFolding, IrDeadCodeElimination, IrPass};

        // Apply dead code elimination
        let dce = IrDeadCodeElimination;
        dce.run(&mut ir_module)?;

        // Apply constant folding
        let cf = IrConstantFolding;
        cf.run(&mut ir_module)?;

        // Validate the optimized IR
        ir_module.validate().map_err(JitError::GraphError)?;

        Ok(ir_module)
    }

    /// Generate native code from IR
    fn generate_code(&self, ir_module: &crate::ir::IrModule) -> JitResult<Vec<CompiledKernel>> {
        match self.config.target_device {
            DeviceType::Cpu => {
                #[cfg(feature = "cranelift-backend")]
                {
                    let mut codegen = crate::cranelift_backend::CraneliftCodeGen::new()?;
                    codegen.generate(ir_module)
                }
                #[cfg(not(feature = "cranelift-backend"))]
                {
                    // Fallback to interpreter
                    let codegen = CodeGenerator::new(self.config.target_device.clone());
                    codegen.generate_interpreter(ir_module)
                }
            }
            _ => {
                // Use standard code generator for other devices
                let codegen = CodeGenerator::new(self.config.target_device);
                codegen.generate_from_ir(ir_module)
            }
        }
    }
}

/// A compiled module ready for execution
pub struct CompiledModule {
    graph: ComputationGraph,
    kernels: Vec<CompiledKernel>,
    runtime: JitRuntime,
}

impl CompiledModule {
    /// Execute the compiled module with the given inputs
    pub fn execute(&self, inputs: &[TensorRef]) -> JitResult<Vec<TensorRef>> {
        self.runtime.execute(&self.graph, &self.kernels, inputs)
    }

    /// Get execution statistics
    pub fn stats(&self) -> ExecutionStats {
        self.runtime.stats()
    }
}

/// Compiled kernel representation
pub struct CompiledKernel {
    /// Unique identifier
    pub id: String,

    /// Source nodes that were fused
    pub source_nodes: Vec<NodeId>,

    /// Generated code (backend-specific)
    pub code: Vec<u8>,

    /// Kernel metadata
    pub metadata: KernelMetadata,
}

/// Kernel metadata for runtime execution
#[derive(Debug, Clone)]
pub struct KernelMetadata {
    /// Input tensor descriptions
    pub inputs: Vec<TensorDesc>,

    /// Output tensor descriptions
    pub outputs: Vec<TensorDesc>,

    /// Shared memory requirements
    pub shared_memory: usize,

    /// Thread block configuration
    pub block_size: (usize, usize, usize),

    /// Grid configuration
    pub grid_size: (usize, usize, usize),
}

/// Tensor description for kernel interface
#[derive(Debug, Clone)]
pub struct TensorDesc {
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub offset: usize,
}

/// Execution statistics
#[derive(Debug, Clone, Default)]
pub struct ExecutionStats {
    /// Total execution time in microseconds
    pub total_time_us: u64,

    /// Number of kernel launches
    pub kernel_launches: usize,

    /// Memory transferred in bytes
    pub memory_transferred: usize,

    /// Cache hit rate
    pub cache_hit_rate: f32,
}

/// Placeholder for tensor references (will be properly integrated with torsh-tensor)
#[derive(Clone, Debug)]
pub struct TensorRef {
    /// Placeholder data
    pub data: Vec<f32>,
}

/// JIT trace a function to capture its computation graph
///
/// Traces the execution of a function with example inputs to build a computation graph.
/// The graph can then be optimized and compiled for efficient execution.
///
/// # Arguments
/// * `func` - The function to trace
/// * `example_inputs` - Example tensor inputs for tracing
///
/// # Returns
/// A compiled module ready for execution
///
/// # Example
/// ```rust,ignore
/// use torsh_jit::{trace, TensorRef};
///
/// let example_inputs = vec![/* ... */];
/// let compiled = trace(|inputs| {
///     // Your computation here
///     vec![/* outputs */]
/// }, &example_inputs)?;
/// ```
///
/// # Implementation Status
/// Currently returns a placeholder module. Full tracing implementation requires:
/// - Tensor operation interception
/// - Graph construction from traced operations
/// - Type and shape inference
/// - Integration with autograd for gradient tracking
pub fn trace<F>(_func: F, _example_inputs: &[TensorRef]) -> JitResult<CompiledModule>
where
    F: Fn(&[TensorRef]) -> Vec<TensorRef>,
{
    // Create a placeholder compiled module
    // Full implementation would:
    // 1. Set up tracing context
    // 2. Execute function with traced tensors
    // 3. Build computation graph from traced operations
    // 4. Optimize and compile the graph

    // Return a minimal placeholder module
    // Full implementation would build and compile the traced graph
    Ok(CompiledModule {
        graph: ComputationGraph::new(),
        kernels: Vec::new(),
        runtime: JitRuntime::new(JitConfig::default()),
    })
}

/// JIT script a module
pub fn script<M>(module: M) -> JitResult<CompiledModule>
where
    M: ScriptableModule,
{
    script::script(module)
}

/// Trait for scriptable modules
pub trait ScriptableModule {
    /// Get the computation graph for this module
    fn to_graph(&self) -> JitResult<ComputationGraph>;
}

/// Utility functions for common JIT operations
pub mod utils {
    use super::{graph, ComputationGraph, DType, FusionStrategy, JitConfig};

    /// Estimate compilation time for a graph
    ///
    /// Provides a rough estimate of compilation time based on graph complexity.
    /// Useful for deciding whether to JIT compile or use interpretation.
    ///
    /// # Returns
    /// Estimated compilation time in milliseconds
    #[must_use]
    pub fn estimate_compilation_time(graph: &ComputationGraph) -> u64 {
        let node_count = graph.nodes().count();
        let edge_count = graph.edges().count();

        // Heuristic: ~0.5ms per node + 0.1ms per edge + base overhead
        let base_overhead = 10; // ms
        let node_time = (node_count as f64 * 0.5) as u64;
        let edge_time = (edge_count as f64 * 0.1) as u64;

        base_overhead + node_time + edge_time
    }

    /// Estimate memory usage for a compiled module
    ///
    /// Estimates the memory footprint of a compiled module.
    ///
    /// # Returns
    /// Estimated memory usage in bytes
    #[must_use]
    pub fn estimate_memory_usage(graph: &ComputationGraph) -> usize {
        let mut total_bytes = 0;

        for (_, node) in graph.nodes() {
            let elements: usize = node.output_shape.dims().iter().product();
            let dtype_size = match node.dtype {
                DType::F32 | DType::I32 | DType::U32 | DType::QInt32 => 4,
                DType::F64 | DType::I64 | DType::U64 | DType::C64 => 8,
                DType::F16 | DType::BF16 | DType::I16 => 2,
                DType::I8 | DType::U8 | DType::Bool | DType::QInt8 | DType::QUInt8 => 1,
                DType::C128 => 16,
            };

            total_bytes += elements * dtype_size;
        }

        // Add overhead for graph structure and metadata
        let overhead = graph.nodes().count() * 256; // ~256 bytes per node
        total_bytes + overhead
    }

    /// Check if a graph is amenable to JIT compilation
    ///
    /// Analyzes the graph to determine if JIT compilation would be beneficial.
    ///
    /// # Returns
    /// `true` if JIT compilation is recommended, `false` if interpretation might be better
    #[must_use]
    pub fn should_jit_compile(graph: &ComputationGraph) -> bool {
        let node_count = graph.nodes().count();

        // Too small: interpretation overhead is negligible
        if node_count < 5 {
            return false;
        }

        // Check for fusion opportunities
        let fusion_opportunities = count_fusion_opportunities(graph);
        if fusion_opportunities > 3 {
            return true; // Many fusion opportunities - good for JIT
        }

        // Check for repeated patterns (loops, etc.)
        // For now, simple heuristic: medium-sized graphs benefit from JIT
        node_count >= 10
    }

    /// Count potential fusion opportunities in a graph
    fn count_fusion_opportunities(graph: &ComputationGraph) -> usize {
        let mut opportunities = 0;

        for (node_id, node) in graph.nodes() {
            // Check if this node can be fused with predecessors
            let predecessors = graph.predecessors(node_id).count();

            if predecessors > 0 && is_fusible_op(&node.op) {
                opportunities += 1;
            }
        }

        opportunities
    }

    /// Check if an operation is fusible
    fn is_fusible_op(op: &graph::Operation) -> bool {
        matches!(
            op,
            graph::Operation::Add
                | graph::Operation::Sub
                | graph::Operation::Mul
                | graph::Operation::Div
                | graph::Operation::Relu
                | graph::Operation::Sigmoid
                | graph::Operation::Tanh
                | graph::Operation::Gelu
                | graph::Operation::Exp
                | graph::Operation::Log
                | graph::Operation::Sqrt
                | graph::Operation::Neg
                | graph::Operation::Abs
        )
    }

    /// Get recommended JIT configuration for a graph
    ///
    /// Analyzes the graph and returns optimal JIT configuration settings.
    #[must_use]
    pub fn recommend_config(graph: &ComputationGraph) -> JitConfig {
        let node_count = graph.nodes().count();
        let fusion_ops = count_fusion_opportunities(graph);

        let mut config = JitConfig::default();

        // Adjust fusion strategy based on graph characteristics
        if fusion_ops > 10 {
            config.fusion_strategy = FusionStrategy::Aggressive;
            config.max_fusion_size = 16;
        } else if fusion_ops > 5 {
            config.fusion_strategy = FusionStrategy::Default;
            config.max_fusion_size = 8;
        } else {
            config.fusion_strategy = FusionStrategy::Conservative;
            config.max_fusion_size = 4;
        }

        // Enable optimizations for larger graphs
        config.enable_optimizations = node_count >= 10;

        // Enable profiling for complex graphs
        config.enable_profiling = node_count >= 50;

        config
    }

    /// Calculate the theoretical peak performance (FLOPS) for a graph
    ///
    /// Estimates the floating-point operations required to execute the graph.
    ///
    /// # Returns
    /// Estimated FLOPS (floating-point operations)
    #[must_use]
    pub fn estimate_flops(graph: &ComputationGraph) -> u64 {
        let mut total_flops = 0u64;

        for (_, node) in graph.nodes() {
            let elements: u64 = node.output_shape.dims().iter().product::<usize>() as u64;

            let op_flops = match &node.op {
                graph::Operation::MatMul => {
                    // Matrix multiplication: 2*m*n*k FLOPs
                    // Simplified: assume square matrices
                    let dim = (elements as f64).sqrt() as u64;
                    2 * dim * dim * dim
                }
                graph::Operation::Conv2d { .. } => {
                    // Convolution: very rough estimate
                    elements * 9 // 3x3 kernel approximation
                }
                graph::Operation::Add
                | graph::Operation::Sub
                | graph::Operation::Mul
                | graph::Operation::Div => elements,
                graph::Operation::Relu | graph::Operation::Abs | graph::Operation::Neg => {
                    elements / 2 // Very cheap operations
                }
                graph::Operation::Exp
                | graph::Operation::Log
                | graph::Operation::Sqrt
                | graph::Operation::Sin
                | graph::Operation::Cos => {
                    elements * 10 // Expensive transcendental functions
                }
                graph::Operation::Sigmoid | graph::Operation::Tanh | graph::Operation::Gelu => {
                    elements * 5 // Moderate complexity
                }
                _ => elements, // Default: 1 FLOP per element
            };

            total_flops += op_flops;
        }

        total_flops
    }

    /// Estimate arithmetic intensity (FLOPS/byte) for a graph
    ///
    /// Higher arithmetic intensity indicates compute-bound operations
    /// that benefit more from optimization.
    ///
    /// # Returns
    /// Arithmetic intensity (FLOPS per byte)
    #[must_use]
    pub fn estimate_arithmetic_intensity(graph: &ComputationGraph) -> f64 {
        let flops = estimate_flops(graph) as f64;
        let bytes = estimate_memory_usage(graph) as f64;

        if bytes > 0.0 {
            flops / bytes
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Node, Operation};

    #[test]
    fn test_jit_config_default() {
        let config = JitConfig::default();
        assert!(config.enable_optimizations);
        assert_eq!(config.max_fusion_size, 8);
        assert!(!config.enable_profiling);
    }

    #[test]
    fn test_jit_compiler_creation() {
        let config = JitConfig::default();
        let _compiler = JitCompiler::new(config);
        // Basic creation test
        assert!(true);
    }

    #[test]
    fn test_utils_estimate_compilation_time() {
        let mut graph = ComputationGraph::new();

        // Empty graph should have minimal compilation time
        let time = utils::estimate_compilation_time(&graph);
        assert!(time >= 10); // At least base overhead

        // Add some nodes
        for i in 0..10 {
            let node = Node::new(Operation::Add, format!("node_{}", i))
                .with_output_shapes(vec![Some(crate::graph::shape_from_slice(&[100]))])
                .with_dtypes(vec![DType::F32])
                .with_device(DeviceType::Cpu);
            graph.add_node(node);
        }

        let time_with_nodes = utils::estimate_compilation_time(&graph);
        assert!(time_with_nodes > time); // More nodes = more time
    }

    #[test]
    fn test_utils_estimate_memory_usage() {
        let mut graph = ComputationGraph::new();

        // Add a node with known size
        let node = Node::new(Operation::Add, "test".to_string())
            .with_output_shapes(vec![Some(crate::graph::shape_from_slice(&[100, 100]))])
            .with_dtypes(vec![DType::F32])
            .with_device(DeviceType::Cpu);
        graph.add_node(node);

        let memory = utils::estimate_memory_usage(&graph);

        // 100*100 elements * 4 bytes (F32) + overhead
        let expected_min = 100 * 100 * 4;
        assert!(memory >= expected_min);
    }

    #[test]
    fn test_utils_should_jit_compile() {
        let mut graph = ComputationGraph::new();

        // Very small graph should not JIT compile
        for i in 0..3 {
            let node = Node::new(Operation::Add, format!("node_{}", i))
                .with_output_shapes(vec![Some(crate::graph::shape_from_slice(&[10]))])
                .with_dtypes(vec![DType::F32])
                .with_device(DeviceType::Cpu);
            graph.add_node(node);
        }

        assert!(!utils::should_jit_compile(&graph));

        // Larger graph should JIT compile
        for i in 3..15 {
            let node = Node::new(Operation::Add, format!("node_{}", i))
                .with_output_shapes(vec![Some(crate::graph::shape_from_slice(&[10]))])
                .with_dtypes(vec![DType::F32])
                .with_device(DeviceType::Cpu);
            graph.add_node(node);
        }

        assert!(utils::should_jit_compile(&graph));
    }

    #[test]
    fn test_utils_recommend_config() {
        let mut graph = ComputationGraph::new();

        // Add fusible operations with connections
        let mut prev_nodes = Vec::new();

        for i in 0..15 {
            let op = if i % 3 == 0 {
                Operation::Add
            } else if i % 3 == 1 {
                Operation::Mul
            } else {
                Operation::Relu
            };

            let node = Node::new(op, format!("node_{}", i))
                .with_output_shapes(vec![Some(crate::graph::shape_from_slice(&[100]))])
                .with_dtypes(vec![DType::F32])
                .with_device(DeviceType::Cpu);
            let node_id = graph.add_node(node);

            // Connect to previous node to create fusion opportunities
            if let Some(&prev) = prev_nodes.last() {
                graph.add_edge(
                    prev,
                    node_id,
                    crate::graph::Edge {
                        src_output: 0,
                        dst_input: 0,
                    },
                );
            }

            prev_nodes.push(node_id);
        }

        let config = utils::recommend_config(&graph);

        // Should enable optimizations for larger graphs
        assert!(config.enable_optimizations);
        // Should have reasonable fusion settings
        assert!(config.max_fusion_size >= 4);
    }

    #[test]
    fn test_utils_estimate_flops() {
        let mut graph = ComputationGraph::new();

        // MatMul operation
        let node = Node::new(Operation::MatMul, "matmul".to_string())
            .with_output_shapes(vec![Some(crate::graph::shape_from_slice(&[64, 64]))])
            .with_dtypes(vec![DType::F32])
            .with_device(DeviceType::Cpu);
        graph.add_node(node);

        let flops = utils::estimate_flops(&graph);

        // MatMul of 64x64 should have significant FLOPs
        assert!(flops > 100_000);
    }

    #[test]
    fn test_utils_estimate_arithmetic_intensity() {
        let mut graph = ComputationGraph::new();

        // Add a cheap operation
        let node = Node::new(Operation::Add, "add".to_string())
            .with_output_shapes(vec![Some(crate::graph::shape_from_slice(&[1000]))])
            .with_dtypes(vec![DType::F32])
            .with_device(DeviceType::Cpu);
        graph.add_node(node);

        let intensity = utils::estimate_arithmetic_intensity(&graph);

        // Should have some arithmetic intensity
        assert!(intensity > 0.0);
        assert!(intensity.is_finite());
    }

    #[test]
    fn test_trace_placeholder() {
        // Test that trace returns a valid module (even if placeholder)
        let result = trace(|_inputs| vec![], &[]);
        assert!(result.is_ok());

        let module = result.unwrap();
        assert_eq!(module.kernels.len(), 0); // Placeholder has no kernels
    }

    #[test]
    fn test_jit_error_display() {
        let error = JitError::GraphError("test error".to_string());
        let display = format!("{}", error);
        assert!(display.contains("test error"));
    }

    #[test]
    fn test_jit_config_builder_pattern() {
        let config = JitConfig {
            fusion_strategy: FusionStrategy::Aggressive,
            enable_optimizations: true,
            max_fusion_size: 16,
            enable_profiling: true,
            target_device: DeviceType::Cpu,
            enable_caching: true,
            enable_specialization: false,
            specialization_config: SpecializationConfig::default(),
        };

        assert_eq!(config.fusion_strategy, FusionStrategy::Aggressive);
        assert!(config.enable_optimizations);
        assert_eq!(config.max_fusion_size, 16);
        assert!(config.enable_caching);
    }
}

// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const VERSION_MAJOR: u32 = 0;
pub const VERSION_MINOR: u32 = 1;
pub const VERSION_PATCH: u32 = 0;

/// Prelude module for convenient imports
#[allow(ambiguous_glob_reexports)]
pub mod prelude {
    pub use crate::{
        abstract_interpretation::*, adaptive_compilation::*, codegen::*, const_eval::*,
        custom_ops::*, debug_symbols::*, debugger::*, differentiable_compilation::*,
        error_diagnostics::*, fusion::*, graph::*, optimizer::*, runtime::*, script::*, tracing::*,
    };
}
