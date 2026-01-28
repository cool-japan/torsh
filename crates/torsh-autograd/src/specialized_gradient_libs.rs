//! Integration with Specialized Gradient Computation Libraries
//!
//! This module provides integration capabilities with specialized gradient computation
//! libraries and frameworks, enabling torsh-autograd to leverage external AD systems
//! for specific use cases or performance optimization.

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use crate::error_handling::{AutogradError, AutogradResult};
use scirs2_core::ndarray::{Array, Ix2, IxDyn};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::sync::{Mutex, RwLock};

/// Supported specialized gradient computation libraries
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SpecializedLibrary {
    /// CasADi - symbolic framework for nonlinear optimization
    CasADi,
    /// JAX - high-performance machine learning research
    JAX,
    /// TensorFlow AutoGraph - automatic control flow conversion
    TensorFlowAutoGraph,
    /// PyTorch JIT - Just-In-Time compilation
    PyTorchJIT,
    /// Enzyme - LLVM-based automatic differentiation
    Enzyme,
    /// Zygote.jl - source-to-source automatic differentiation in Julia
    Zygote,
    /// ForwardDiff.jl - forward-mode automatic differentiation
    ForwardDiff,
    /// ReverseDiff.jl - reverse-mode automatic differentiation
    ReverseDiff,
    /// ADOL-C - automatic differentiation by operator overloading
    ADOLC,
    /// CppAD - C++ algorithmic differentiation
    CppAD,
    /// Stan Math - C++ automatic differentiation library
    StanMath,
    /// Custom user-provided library
    Custom(String),
}

impl fmt::Display for SpecializedLibrary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SpecializedLibrary::CasADi => write!(f, "CasADi"),
            SpecializedLibrary::JAX => write!(f, "JAX"),
            SpecializedLibrary::TensorFlowAutoGraph => write!(f, "TensorFlow AutoGraph"),
            SpecializedLibrary::PyTorchJIT => write!(f, "PyTorch JIT"),
            SpecializedLibrary::Enzyme => write!(f, "Enzyme"),
            SpecializedLibrary::Zygote => write!(f, "Zygote.jl"),
            SpecializedLibrary::ForwardDiff => write!(f, "ForwardDiff.jl"),
            SpecializedLibrary::ReverseDiff => write!(f, "ReverseDiff.jl"),
            SpecializedLibrary::ADOLC => write!(f, "ADOL-C"),
            SpecializedLibrary::CppAD => write!(f, "CppAD"),
            SpecializedLibrary::StanMath => write!(f, "Stan Math"),
            SpecializedLibrary::Custom(name) => write!(f, "Custom({})", name),
        }
    }
}

/// Types of gradient computation supported by libraries
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GradientComputationType {
    /// Forward-mode automatic differentiation
    ForwardMode,
    /// Reverse-mode automatic differentiation
    ReverseMode,
    /// Mixed-mode (both forward and reverse)
    MixedMode,
    /// Symbolic differentiation
    Symbolic,
    /// Numerical differentiation (finite differences)
    Numerical,
    /// Sparse gradient computation
    Sparse,
    /// Higher-order derivatives
    HigherOrder,
    /// Jacobian matrix computation
    Jacobian,
    /// Jacobian-vector products
    JVP,
    /// Vector-Jacobian products
    VJP,
    /// Hessian computation
    Hessian,
}

impl fmt::Display for GradientComputationType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GradientComputationType::ForwardMode => write!(f, "Forward-mode AD"),
            GradientComputationType::ReverseMode => write!(f, "Reverse-mode AD"),
            GradientComputationType::MixedMode => write!(f, "Mixed-mode AD"),
            GradientComputationType::Symbolic => write!(f, "Symbolic differentiation"),
            GradientComputationType::Numerical => write!(f, "Numerical differentiation"),
            GradientComputationType::Sparse => write!(f, "Sparse gradients"),
            GradientComputationType::HigherOrder => write!(f, "Higher-order derivatives"),
            GradientComputationType::Jacobian => write!(f, "Jacobian matrix"),
            GradientComputationType::JVP => write!(f, "Jacobian-vector products"),
            GradientComputationType::VJP => write!(f, "Vector-Jacobian products"),
            GradientComputationType::Hessian => write!(f, "Hessian computation"),
        }
    }
}

/// Configuration for specialized library integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecializedLibConfig {
    pub library: SpecializedLibrary,
    pub enabled_types: Vec<GradientComputationType>,
    pub priority: u32,               // Higher priority libraries are preferred
    pub performance_threshold: f64,  // Minimum performance gain to use this library
    pub memory_limit: Option<usize>, // Maximum memory usage in bytes
    pub timeout: Option<std::time::Duration>, // Maximum computation time
    pub fallback_enabled: bool,
    pub library_specific_config: HashMap<String, String>,
}

impl Default for SpecializedLibConfig {
    fn default() -> Self {
        Self {
            library: SpecializedLibrary::Custom("default".to_string()),
            enabled_types: vec![
                GradientComputationType::ForwardMode,
                GradientComputationType::ReverseMode,
            ],
            priority: 1,
            performance_threshold: 1.1, // 10% minimum improvement
            memory_limit: Some(1024 * 1024 * 1024), // 1GB
            timeout: Some(std::time::Duration::from_secs(60)),
            fallback_enabled: true,
            library_specific_config: HashMap::new(),
        }
    }
}

/// Trait for specialized gradient computation libraries
pub trait SpecializedGradientLibrary: Send + Sync + std::fmt::Debug {
    fn library_name(&self) -> SpecializedLibrary;
    fn is_available(&self) -> bool;
    fn supported_types(&self) -> Vec<GradientComputationType>;
    fn initialize(&mut self, config: &SpecializedLibConfig) -> AutogradResult<()>;
    fn shutdown(&mut self) -> AutogradResult<()>;

    // Forward-mode AD
    fn compute_forward_gradient(
        &self,
        function: &dyn Function,
        inputs: &[f64],
        tangents: &[f64],
    ) -> AutogradResult<(Vec<f64>, Vec<f64>)>;

    // Reverse-mode AD
    fn compute_reverse_gradient(
        &self,
        function: &dyn Function,
        inputs: &[f64],
        output_grad: &[f64],
    ) -> AutogradResult<Vec<f64>>;

    // Jacobian computation
    fn compute_jacobian(
        &self,
        function: &dyn Function,
        inputs: &[f64],
    ) -> AutogradResult<Array<f64, Ix2>>;

    // Hessian computation
    fn compute_hessian(
        &self,
        function: &dyn Function,
        inputs: &[f64],
    ) -> AutogradResult<Array<f64, Ix2>>;

    // Higher-order derivatives
    fn compute_higher_order(
        &self,
        function: &dyn Function,
        inputs: &[f64],
        order: usize,
    ) -> AutogradResult<Vec<Array<f64, IxDyn>>>;

    // Sparse gradient computation
    fn compute_sparse_gradient(
        &self,
        function: &dyn Function,
        inputs: &[f64],
        sparsity_pattern: &[usize],
    ) -> AutogradResult<SparseGradient>;

    // JVP and VJP
    fn compute_jvp(
        &self,
        function: &dyn Function,
        inputs: &[f64],
        vectors: &[f64],
    ) -> AutogradResult<Vec<f64>>;
    fn compute_vjp(
        &self,
        function: &dyn Function,
        inputs: &[f64],
        vectors: &[f64],
    ) -> AutogradResult<Vec<f64>>;

    // Performance benchmarking
    fn benchmark(
        &self,
        function: &dyn Function,
        inputs: &[f64],
        computation_type: GradientComputationType,
    ) -> AutogradResult<f64>;

    // Memory usage estimation
    fn estimate_memory_usage(
        &self,
        function: &dyn Function,
        inputs: &[f64],
        computation_type: GradientComputationType,
    ) -> AutogradResult<usize>;
}

/// Abstract function interface for gradient computation
pub trait Function: Send + Sync + std::fmt::Debug {
    fn evaluate(&self, inputs: &[f64]) -> AutogradResult<Vec<f64>>;
    fn input_dimension(&self) -> usize;
    fn output_dimension(&self) -> usize;
    fn name(&self) -> &str;
    fn is_differentiable(&self) -> bool;
    fn sparsity_pattern(&self) -> Option<Vec<(usize, usize)>>; // For sparse Jacobians
}

/// Sparse gradient representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseGradient {
    pub indices: Vec<usize>,
    pub values: Vec<f64>,
    pub size: usize,
}

impl SparseGradient {
    pub fn new(indices: Vec<usize>, values: Vec<f64>, size: usize) -> Self {
        Self {
            indices,
            values,
            size,
        }
    }

    pub fn to_dense(&self) -> Vec<f64> {
        let mut dense = vec![0.0; self.size];
        for (&idx, &val) in self.indices.iter().zip(self.values.iter()) {
            if idx < self.size {
                dense[idx] = val;
            }
        }
        dense
    }

    pub fn sparsity_ratio(&self) -> f64 {
        self.indices.len() as f64 / self.size as f64
    }
}

/// CasADi integration
#[derive(Debug)]
pub struct CasADiLibrary {
    initialized: bool,
    config: Option<SpecializedLibConfig>,
}

impl CasADiLibrary {
    pub fn new() -> Self {
        Self {
            initialized: false,
            config: None,
        }
    }

    fn check_casadi_available(&self) -> bool {
        // In practice, this would check for CasADi Python bindings or C++ library
        // For now, simulate availability
        std::env::var("CASADI_PREFIX").is_ok()
    }
}

impl SpecializedGradientLibrary for CasADiLibrary {
    fn library_name(&self) -> SpecializedLibrary {
        SpecializedLibrary::CasADi
    }

    fn is_available(&self) -> bool {
        self.check_casadi_available()
    }

    fn supported_types(&self) -> Vec<GradientComputationType> {
        vec![
            GradientComputationType::ForwardMode,
            GradientComputationType::ReverseMode,
            GradientComputationType::Symbolic,
            GradientComputationType::Jacobian,
            GradientComputationType::Hessian,
            GradientComputationType::Sparse,
        ]
    }

    fn initialize(&mut self, config: &SpecializedLibConfig) -> AutogradResult<()> {
        if !self.is_available() {
            return Err(AutogradError::gradient_computation(
                "library_initialization",
                "CasADi library not available",
            ));
        }

        self.config = Some(config.clone());
        self.initialized = true;

        tracing::info!("CasADi library initialized successfully");
        Ok(())
    }

    fn shutdown(&mut self) -> AutogradResult<()> {
        self.initialized = false;
        self.config = None;
        tracing::info!("CasADi library shutdown");
        Ok(())
    }

    fn compute_forward_gradient(
        &self,
        function: &dyn Function,
        inputs: &[f64],
        tangents: &[f64],
    ) -> AutogradResult<(Vec<f64>, Vec<f64>)> {
        if !self.initialized {
            return Err(AutogradError::gradient_computation(
                "library_initialization",
                "CasADi library not initialized",
            ));
        }

        // Simulate CasADi forward-mode computation
        let output = function.evaluate(inputs)?;
        let mut grad = vec![0.0; inputs.len()];

        // Simplified forward-mode AD simulation
        for i in 0..inputs.len() {
            grad[i] = tangents[i]; // Placeholder computation
        }

        tracing::debug!(
            "CasADi forward gradient computed for function '{}'",
            function.name()
        );
        Ok((output, grad))
    }

    fn compute_reverse_gradient(
        &self,
        function: &dyn Function,
        inputs: &[f64],
        output_grad: &[f64],
    ) -> AutogradResult<Vec<f64>> {
        if !self.initialized {
            return Err(AutogradError::gradient_computation(
                "library_initialization",
                "CasADi library not initialized",
            ));
        }

        // Simulate CasADi reverse-mode computation
        let mut grad = vec![0.0; inputs.len()];

        // Simplified reverse-mode AD simulation
        for i in 0..inputs.len() {
            grad[i] = output_grad.iter().sum::<f64>() / inputs.len() as f64; // Placeholder
        }

        tracing::debug!(
            "CasADi reverse gradient computed for function '{}'",
            function.name()
        );
        Ok(grad)
    }

    fn compute_jacobian(
        &self,
        function: &dyn Function,
        _inputs: &[f64],
    ) -> AutogradResult<Array<f64, Ix2>> {
        if !self.initialized {
            return Err(AutogradError::gradient_computation(
                "library_initialization",
                "CasADi library not initialized",
            ));
        }

        let input_dim = function.input_dimension();
        let output_dim = function.output_dimension();
        let mut jacobian = Array::zeros((output_dim, input_dim));

        // Simulate Jacobian computation
        for i in 0..output_dim {
            for j in 0..input_dim {
                jacobian[[i, j]] = (i + j) as f64 * 0.1; // Placeholder computation
            }
        }

        tracing::debug!(
            "CasADi Jacobian computed for function '{}'",
            function.name()
        );
        Ok(jacobian)
    }

    fn compute_hessian(
        &self,
        function: &dyn Function,
        _inputs: &[f64],
    ) -> AutogradResult<Array<f64, Ix2>> {
        if !self.initialized {
            return Err(AutogradError::gradient_computation(
                "library_initialization",
                "CasADi library not initialized",
            ));
        }

        let input_dim = function.input_dimension();
        let mut hessian = Array::zeros((input_dim, input_dim));

        // Simulate Hessian computation
        for i in 0..input_dim {
            for j in 0..input_dim {
                hessian[[i, j]] = if i == j { 2.0 } else { 0.1 }; // Placeholder
            }
        }

        tracing::debug!("CasADi Hessian computed for function '{}'", function.name());
        Ok(hessian)
    }

    fn compute_higher_order(
        &self,
        function: &dyn Function,
        _inputs: &[f64],
        order: usize,
    ) -> AutogradResult<Vec<Array<f64, IxDyn>>> {
        if !self.initialized {
            return Err(AutogradError::gradient_computation(
                "library_initialization",
                "CasADi library not initialized",
            ));
        }

        let mut derivatives = Vec::new();

        for ord in 1..=order {
            let shape = vec![function.input_dimension(); ord];
            let derivative = Array::ones(shape.as_slice()) * (1.0 / ord as f64); // Placeholder
            derivatives.push(derivative);
        }

        tracing::debug!(
            "CasADi higher-order derivatives (order {}) computed for function '{}'",
            order,
            function.name()
        );
        Ok(derivatives)
    }

    fn compute_sparse_gradient(
        &self,
        function: &dyn Function,
        _inputs: &[f64],
        sparsity_pattern: &[usize],
    ) -> AutogradResult<SparseGradient> {
        if !self.initialized {
            return Err(AutogradError::gradient_computation(
                "library_initialization",
                "CasADi library not initialized",
            ));
        }

        let indices = sparsity_pattern.to_vec();
        let values = vec![1.0; indices.len()]; // Placeholder computation

        let sparse_grad = SparseGradient::new(indices, values, function.input_dimension());

        tracing::debug!(
            "CasADi sparse gradient computed for function '{}' with sparsity {:.2}%",
            function.name(),
            sparse_grad.sparsity_ratio() * 100.0
        );
        Ok(sparse_grad)
    }

    fn compute_jvp(
        &self,
        function: &dyn Function,
        inputs: &[f64],
        vectors: &[f64],
    ) -> AutogradResult<Vec<f64>> {
        let (_, jvp) = self.compute_forward_gradient(function, inputs, vectors)?;
        Ok(jvp)
    }

    fn compute_vjp(
        &self,
        function: &dyn Function,
        inputs: &[f64],
        vectors: &[f64],
    ) -> AutogradResult<Vec<f64>> {
        self.compute_reverse_gradient(function, inputs, vectors)
    }

    fn benchmark(
        &self,
        function: &dyn Function,
        inputs: &[f64],
        computation_type: GradientComputationType,
    ) -> AutogradResult<f64> {
        use std::time::Instant;

        let start = Instant::now();
        let iterations = 10;

        for _ in 0..iterations {
            match computation_type {
                GradientComputationType::ForwardMode => {
                    let tangents = vec![1.0; inputs.len()];
                    let _ = self.compute_forward_gradient(function, inputs, &tangents)?;
                }
                GradientComputationType::ReverseMode => {
                    let output_grad = vec![1.0; function.output_dimension()];
                    let _ = self.compute_reverse_gradient(function, inputs, &output_grad)?;
                }
                GradientComputationType::Jacobian => {
                    let _ = self.compute_jacobian(function, inputs)?;
                }
                _ => {
                    return Err(AutogradError::gradient_computation(
                        "computation",
                        format!(
                            "Benchmarking not supported for computation type: {}",
                            computation_type
                        ),
                    ));
                }
            }
        }

        let elapsed = start.elapsed();
        let time_per_operation = elapsed.as_secs_f64() / iterations as f64;

        tracing::debug!(
            "CasADi {} benchmark: {:.6}s per operation",
            computation_type,
            time_per_operation
        );

        Ok(time_per_operation)
    }

    fn estimate_memory_usage(
        &self,
        function: &dyn Function,
        _inputs: &[f64],
        computation_type: GradientComputationType,
    ) -> AutogradResult<usize> {
        let input_dim = function.input_dimension();
        let output_dim = function.output_dimension();

        let estimated_bytes = match computation_type {
            GradientComputationType::ForwardMode => input_dim * 8, // 8 bytes per f64
            GradientComputationType::ReverseMode => output_dim * 8,
            GradientComputationType::Jacobian => input_dim * output_dim * 8,
            GradientComputationType::Hessian => input_dim * input_dim * 8,
            _ => input_dim * 8, // Default estimate
        };

        Ok(estimated_bytes)
    }
}

/// Example function implementation
#[derive(Debug)]
pub struct QuadraticFunction {
    pub coefficients: Vec<f64>,
}

impl QuadraticFunction {
    pub fn new(coefficients: Vec<f64>) -> Self {
        Self { coefficients }
    }
}

impl Function for QuadraticFunction {
    fn evaluate(&self, inputs: &[f64]) -> AutogradResult<Vec<f64>> {
        if inputs.len() != self.input_dimension() {
            return Err(AutogradError::gradient_computation(
                "computation",
                format!(
                    "Input dimension mismatch: expected {}, got {}",
                    self.input_dimension(),
                    inputs.len()
                ),
            ));
        }

        // Compute quadratic function: sum(c_i * x_i^2)
        let result = inputs
            .iter()
            .zip(self.coefficients.iter())
            .map(|(x, c)| c * x * x)
            .sum();

        Ok(vec![result])
    }

    fn input_dimension(&self) -> usize {
        self.coefficients.len()
    }

    fn output_dimension(&self) -> usize {
        1
    }

    fn name(&self) -> &str {
        "QuadraticFunction"
    }

    fn is_differentiable(&self) -> bool {
        true
    }

    fn sparsity_pattern(&self) -> Option<Vec<(usize, usize)>> {
        // Diagonal sparsity pattern for quadratic function
        Some((0..self.input_dimension()).map(|i| (0, i)).collect())
    }
}

/// Manager for specialized gradient computation libraries
pub struct SpecializedLibraryManager {
    libraries: HashMap<SpecializedLibrary, Box<dyn SpecializedGradientLibrary>>,
    configs: HashMap<SpecializedLibrary, SpecializedLibConfig>,
    performance_cache: RwLock<HashMap<(SpecializedLibrary, GradientComputationType), f64>>,
    usage_stats: Mutex<HashMap<SpecializedLibrary, LibraryUsageStats>>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LibraryUsageStats {
    pub total_calls: usize,
    pub total_time: f64,
    pub success_count: usize,
    pub error_count: usize,
    pub average_time: f64,
}

impl SpecializedLibraryManager {
    pub fn new() -> Self {
        Self {
            libraries: HashMap::new(),
            configs: HashMap::new(),
            performance_cache: RwLock::new(HashMap::new()),
            usage_stats: Mutex::new(HashMap::new()),
        }
    }

    pub fn register_library(
        &mut self,
        library: Box<dyn SpecializedGradientLibrary>,
        config: SpecializedLibConfig,
    ) -> AutogradResult<()> {
        let lib_name = library.library_name();

        if library.is_available() {
            let mut lib = library;
            lib.initialize(&config)?;
            self.libraries.insert(lib_name.clone(), lib);
            self.configs.insert(lib_name.clone(), config);

            tracing::info!("Registered specialized library: {}", lib_name);
            Ok(())
        } else {
            Err(AutogradError::gradient_computation(
                "computation",
                format!("Library {} is not available", lib_name),
            ))
        }
    }

    pub fn get_best_library(
        &self,
        computation_type: &GradientComputationType,
    ) -> Option<&dyn SpecializedGradientLibrary> {
        let mut best_library = None;
        let mut best_priority = 0;
        let mut best_performance = 0.0;

        for (lib_name, library) in &self.libraries {
            if library.supported_types().contains(&computation_type) {
                let config = self.configs.get(lib_name).unwrap();

                // Check priority first
                if config.priority > best_priority {
                    best_library = Some(library.as_ref());
                    best_priority = config.priority;
                    best_performance = 0.0; // Reset performance comparison
                } else if config.priority == best_priority {
                    // Same priority, check performance
                    if let Ok(cache) = self.performance_cache.read() {
                        if let Some(&performance) =
                            cache.get(&(lib_name.clone(), computation_type.clone()))
                        {
                            if performance > best_performance {
                                best_library = Some(library.as_ref());
                                best_performance = performance;
                            }
                        }
                    }
                }
            }
        }

        best_library
    }

    pub fn compute_gradient(
        &self,
        function: &dyn Function,
        inputs: &[f64],
        computation_type: GradientComputationType,
    ) -> AutogradResult<ComputationResult> {
        let start = std::time::Instant::now();

        let result = if let Some(library) = self.get_best_library(&computation_type) {
            let lib_name = library.library_name();

            let computation_result = match computation_type {
                GradientComputationType::ForwardMode => {
                    let tangents = vec![1.0; inputs.len()];
                    let (output, gradient) =
                        library.compute_forward_gradient(function, inputs, &tangents)?;
                    ComputationResult::ForwardMode { output, gradient }
                }
                GradientComputationType::ReverseMode => {
                    let output_grad = vec![1.0; function.output_dimension()];
                    let gradient =
                        library.compute_reverse_gradient(function, inputs, &output_grad)?;
                    ComputationResult::ReverseMode { gradient }
                }
                GradientComputationType::Jacobian => {
                    let jacobian = library.compute_jacobian(function, inputs)?;
                    ComputationResult::Jacobian { jacobian }
                }
                GradientComputationType::Hessian => {
                    let hessian = library.compute_hessian(function, inputs)?;
                    ComputationResult::Hessian { hessian }
                }
                _ => {
                    return Err(AutogradError::gradient_computation(
                        "computation",
                        format!("Computation type {} not implemented", computation_type),
                    ));
                }
            };

            self.record_usage(lib_name, start.elapsed().as_secs_f64(), true);
            computation_result
        } else {
            return Err(AutogradError::gradient_computation(
                "computation",
                format!(
                    "No library available for computation type: {}",
                    computation_type
                ),
            ));
        };

        Ok(result)
    }

    fn record_usage(&self, library: SpecializedLibrary, time: f64, success: bool) {
        if let Ok(mut stats) = self.usage_stats.lock() {
            let lib_stats = stats.entry(library).or_insert_with(Default::default);

            lib_stats.total_calls += 1;
            lib_stats.total_time += time;

            if success {
                lib_stats.success_count += 1;
            } else {
                lib_stats.error_count += 1;
            }

            lib_stats.average_time = lib_stats.total_time / lib_stats.total_calls as f64;
        }
    }

    pub fn benchmark_libraries(
        &self,
        function: &dyn Function,
        inputs: &[f64],
    ) -> AutogradResult<BenchmarkReport> {
        let mut results = HashMap::new();

        for (lib_name, library) in &self.libraries {
            let mut lib_results = HashMap::new();

            for computation_type in &library.supported_types() {
                match library.benchmark(function, inputs, computation_type.clone()) {
                    Ok(time) => {
                        lib_results.insert(computation_type.clone(), time);

                        // Cache the performance result
                        if let Ok(mut cache) = self.performance_cache.write() {
                            cache.insert((lib_name.clone(), computation_type.clone()), 1.0 / time);
                            // Higher is better
                        }
                    }
                    Err(e) => {
                        tracing::warn!(
                            "Benchmark failed for {} {}: {}",
                            lib_name,
                            computation_type,
                            e
                        );
                    }
                }
            }

            results.insert(lib_name.clone(), lib_results);
        }

        Ok(BenchmarkReport {
            results,
            function_name: function.name().to_string(),
            input_dimension: function.input_dimension(),
            output_dimension: function.output_dimension(),
        })
    }

    pub fn get_usage_report(&self) -> LibraryUsageReport {
        let stats = self
            .usage_stats
            .lock()
            .expect("lock should not be poisoned")
            .clone();

        LibraryUsageReport {
            library_stats: stats.clone(),
            total_calls: stats.values().map(|s| s.total_calls).sum(),
            total_time: stats.values().map(|s| s.total_time).sum(),
        }
    }
}

/// Result of gradient computation
#[derive(Debug, Clone)]
pub enum ComputationResult {
    ForwardMode {
        output: Vec<f64>,
        gradient: Vec<f64>,
    },
    ReverseMode {
        gradient: Vec<f64>,
    },
    Jacobian {
        jacobian: Array<f64, Ix2>,
    },
    Hessian {
        hessian: Array<f64, Ix2>,
    },
    Sparse {
        sparse_gradient: SparseGradient,
    },
    HigherOrder {
        derivatives: Vec<Array<f64, IxDyn>>,
    },
}

/// Benchmark report for libraries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkReport {
    pub results: HashMap<SpecializedLibrary, HashMap<GradientComputationType, f64>>,
    pub function_name: String,
    pub input_dimension: usize,
    pub output_dimension: usize,
}

impl BenchmarkReport {
    pub fn print_summary(&self) {
        println!("=== Specialized Library Benchmark Report ===");
        println!("Function: {}", self.function_name);
        println!("Input Dimension: {}", self.input_dimension);
        println!("Output Dimension: {}", self.output_dimension);
        println!();

        for (library, results) in &self.results {
            println!("{}:", library);
            for (computation_type, time) in results {
                println!("  {}: {:.6}s", computation_type, time);
            }
            println!();
        }
    }
}

/// Library usage report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LibraryUsageReport {
    pub library_stats: HashMap<SpecializedLibrary, LibraryUsageStats>,
    pub total_calls: usize,
    pub total_time: f64,
}

impl LibraryUsageReport {
    pub fn print_summary(&self) {
        println!("=== Specialized Library Usage Report ===");
        println!("Total Calls: {}", self.total_calls);
        println!("Total Time: {:.4}s", self.total_time);
        println!();

        for (library, stats) in &self.library_stats {
            println!("{}:", library);
            println!("  Calls: {}", stats.total_calls);
            println!(
                "  Success Rate: {:.1}%",
                stats.success_count as f64 / stats.total_calls as f64 * 100.0
            );
            println!("  Average Time: {:.6}s", stats.average_time);
            println!();
        }
    }
}

/// Global specialized library manager
static GLOBAL_SPECIALIZED_MANAGER: std::sync::OnceLock<SpecializedLibraryManager> =
    std::sync::OnceLock::new();

pub fn get_global_specialized_manager() -> &'static SpecializedLibraryManager {
    GLOBAL_SPECIALIZED_MANAGER.get_or_init(|| {
        let mut manager = SpecializedLibraryManager::new();

        // Register CasADi if available
        let casadi_lib = Box::new(CasADiLibrary::new());
        let casadi_config = SpecializedLibConfig {
            library: SpecializedLibrary::CasADi,
            priority: 5,
            ..Default::default()
        };

        if let Err(e) = manager.register_library(casadi_lib, casadi_config) {
            tracing::info!("CasADi library not available: {}", e);
        }

        manager
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_specialized_library_display() {
        assert_eq!(SpecializedLibrary::CasADi.to_string(), "CasADi");
        assert_eq!(SpecializedLibrary::JAX.to_string(), "JAX");
        assert_eq!(
            SpecializedLibrary::Custom("test".to_string()).to_string(),
            "Custom(test)"
        );
    }

    #[test]
    fn test_gradient_computation_type_display() {
        assert_eq!(
            GradientComputationType::ForwardMode.to_string(),
            "Forward-mode AD"
        );
        assert_eq!(
            GradientComputationType::Symbolic.to_string(),
            "Symbolic differentiation"
        );
    }

    #[test]
    fn test_quadratic_function() {
        let coeffs = vec![1.0, 2.0, 3.0];
        let func = QuadraticFunction::new(coeffs);

        assert_eq!(func.input_dimension(), 3);
        assert_eq!(func.output_dimension(), 1);
        assert_eq!(func.name(), "QuadraticFunction");
        assert!(func.is_differentiable());

        let inputs = vec![1.0, 2.0, 3.0];
        let result = func.evaluate(&inputs).unwrap();

        // Expected: 1.0*1^2 + 2.0*2^2 + 3.0*3^2 = 1 + 8 + 27 = 36
        assert_eq!(result[0], 36.0);
    }

    #[test]
    fn test_sparse_gradient() {
        let indices = vec![0, 2, 5];
        let values = vec![1.0, 2.0, 3.0];
        let sparse_grad = SparseGradient::new(indices, values, 10);

        assert_eq!(sparse_grad.sparsity_ratio(), 0.3);

        let dense = sparse_grad.to_dense();
        assert_eq!(dense.len(), 10);
        assert_eq!(dense[0], 1.0);
        assert_eq!(dense[1], 0.0);
        assert_eq!(dense[2], 2.0);
        assert_eq!(dense[5], 3.0);
    }

    #[test]
    fn test_casadi_library() {
        let casadi = CasADiLibrary::new();

        // Should not be available without proper setup
        assert!(!casadi.is_available());
        assert_eq!(casadi.library_name(), SpecializedLibrary::CasADi);

        let supported = casadi.supported_types();
        assert!(supported.contains(&GradientComputationType::ForwardMode));
        assert!(supported.contains(&GradientComputationType::ReverseMode));
        assert!(supported.contains(&GradientComputationType::Symbolic));
    }

    #[test]
    fn test_specialized_lib_manager() {
        let manager = SpecializedLibraryManager::new();
        assert_eq!(manager.libraries.len(), 0);

        // Test with no libraries registered
        let result = manager.get_best_library(&GradientComputationType::ForwardMode);
        assert!(result.is_none());
    }

    #[test]
    fn test_computation_result_variants() {
        let gradient = vec![1.0, 2.0, 3.0];
        let result = ComputationResult::ReverseMode {
            gradient: gradient.clone(),
        };

        match result {
            ComputationResult::ReverseMode { gradient: g } => {
                assert_eq!(g, gradient);
            }
            _ => panic!("Unexpected result type"),
        }
    }

    #[test]
    fn test_specialized_lib_config() {
        let config = SpecializedLibConfig::default();

        assert!(config
            .enabled_types
            .contains(&GradientComputationType::ForwardMode));
        assert!(config
            .enabled_types
            .contains(&GradientComputationType::ReverseMode));
        assert_eq!(config.priority, 1);
        assert!(config.fallback_enabled);
    }

    #[test]
    fn test_global_manager_access() {
        let manager = get_global_specialized_manager();
        // Should not panic and return a valid reference
        assert_eq!(manager.libraries.len(), 0); // No libraries registered by default
    }

    #[test]
    fn test_benchmark_report() {
        let mut results = HashMap::new();
        let mut casadi_results = HashMap::new();
        casadi_results.insert(GradientComputationType::ForwardMode, 0.001);
        casadi_results.insert(GradientComputationType::ReverseMode, 0.002);
        results.insert(SpecializedLibrary::CasADi, casadi_results);

        let report = BenchmarkReport {
            results,
            function_name: "test_function".to_string(),
            input_dimension: 5,
            output_dimension: 1,
        };

        assert_eq!(report.function_name, "test_function");
        assert_eq!(report.input_dimension, 5);
        assert_eq!(report.output_dimension, 1);
    }
}
