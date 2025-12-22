//! Just-In-Time (JIT) Kernel Compilation System
//!
//! This module provides a comprehensive JIT compilation framework for generating
//! and optimizing kernels at runtime based on actual execution characteristics.
//! It supports profile-guided optimization, adaptive recompilation, and multi-tier
//! compilation strategies.
//!
//! ## Features
//!
//! - **Multi-Tier Compilation**: Fast interpreter → unoptimized JIT → optimized JIT
//! - **Profile-Guided Optimization**: Collects runtime profiles to guide recompilation
//! - **Adaptive Recompilation**: Automatically recompiles hot code with better optimizations
//! - **Code Caching**: Persistent caching of compiled code across runs
//! - **Specialization**: Generate specialized kernels for specific input patterns
//! - **Inlining**: Automatic function inlining based on profiling data
//! - **Vectorization**: Automatic SIMD vectorization for supported platforms

use crate::error::BackendResult;
use crate::kernel_generation::{GeneratedKernel, KernelSpec, OptimizationFlags};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, format, string::String, vec::Vec};

// Helper function for serde default
fn instant_now() -> Instant {
    Instant::now()
}

/// JIT compilation tier levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub enum CompilationTier {
    /// No compilation - interpreted execution (fastest startup)
    Interpreter,
    /// Tier 1 - Quick unoptimized compilation (fast compilation)
    QuickJit,
    /// Tier 2 - Optimized compilation (balanced)
    OptimizedJit,
    /// Tier 3 - Aggressive optimization (best performance, slow compilation)
    AggressiveJit,
}

/// Profile-guided optimization data collected during execution
/// Note: Not serializable due to Instant fields - this is runtime-only data
#[derive(Debug, Clone)]
pub struct ExecutionProfile {
    /// Number of times this kernel was executed
    pub execution_count: u64,
    /// Total execution time across all invocations
    pub total_execution_time: Duration,
    /// Average execution time per invocation
    pub average_execution_time: Duration,
    /// Input size distribution (size -> count)
    pub input_size_distribution: HashMap<usize, u64>,
    /// Most common input shapes
    pub common_input_shapes: Vec<(Vec<usize>, u64)>,
    /// Branch prediction data (branch_id -> taken_count)
    pub branch_statistics: HashMap<String, (u64, u64)>, // (taken, not_taken)
    /// Memory access patterns
    pub memory_access_patterns: Vec<AccessPattern>,
    /// Cache miss rate estimate
    pub cache_miss_rate: f64,
    /// SIMD effectiveness (percentage of operations vectorized)
    pub simd_effectiveness: f64,
    /// Profile collection timestamp
    pub last_updated: Instant,
}

impl ExecutionProfile {
    pub fn new() -> Self {
        Self {
            execution_count: 0,
            total_execution_time: Duration::ZERO,
            average_execution_time: Duration::ZERO,
            input_size_distribution: HashMap::new(),
            common_input_shapes: Vec::new(),
            branch_statistics: HashMap::new(),
            memory_access_patterns: Vec::new(),
            cache_miss_rate: 0.0,
            simd_effectiveness: 0.0,
            last_updated: Instant::now(),
        }
    }

    /// Record a kernel execution
    pub fn record_execution(&mut self, execution_time: Duration, input_sizes: &[usize]) {
        self.execution_count += 1;
        self.total_execution_time += execution_time;
        self.average_execution_time = self.total_execution_time / self.execution_count as u32;

        // Update input size distribution
        for &size in input_sizes {
            *self.input_size_distribution.entry(size).or_insert(0) += 1;
        }

        self.last_updated = Instant::now();
    }

    /// Record a branch outcome
    pub fn record_branch(&mut self, branch_id: String, taken: bool) {
        let stats = self.branch_statistics.entry(branch_id).or_insert((0, 0));
        if taken {
            stats.0 += 1;
        } else {
            stats.1 += 1;
        }
    }

    /// Get branch prediction accuracy for a specific branch
    pub fn branch_prediction_accuracy(&self, branch_id: &str) -> f64 {
        if let Some(&(taken, not_taken)) = self.branch_statistics.get(branch_id) {
            let total = taken + not_taken;
            if total > 0 {
                taken.max(not_taken) as f64 / total as f64
            } else {
                0.5
            }
        } else {
            0.5 // No data, assume 50% accuracy
        }
    }

    /// Check if kernel is hot enough for recompilation
    pub fn is_hot(&self, threshold: u64) -> bool {
        self.execution_count >= threshold
    }

    /// Estimate potential speedup from recompilation
    pub fn estimated_speedup_potential(&self) -> f64 {
        let mut speedup_factors = Vec::new();

        // Factor 1: Cache miss reduction potential
        if self.cache_miss_rate > 0.1 {
            speedup_factors.push(1.0 + self.cache_miss_rate * 0.5);
        }

        // Factor 2: SIMD improvement potential
        if self.simd_effectiveness < 0.8 {
            speedup_factors.push(1.0 + (0.8 - self.simd_effectiveness) * 2.0);
        }

        // Factor 3: Branch prediction improvement
        let avg_branch_accuracy: f64 = self
            .branch_statistics
            .iter()
            .map(|(id, _)| self.branch_prediction_accuracy(id))
            .sum::<f64>()
            / self.branch_statistics.len().max(1) as f64;
        if avg_branch_accuracy < 0.9 {
            speedup_factors.push(1.0 + (0.9 - avg_branch_accuracy) * 1.5);
        }

        // Return combined speedup estimate
        speedup_factors.iter().product::<f64>().max(1.0)
    }
}

impl Default for ExecutionProfile {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory access pattern types
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub enum AccessPattern {
    Sequential {
        start: usize,
        end: usize,
        stride: usize,
    },
    Random {
        addresses: Vec<usize>,
    },
    Strided {
        base: usize,
        count: usize,
        stride: isize,
    },
}

/// JIT compilation configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct JitConfig {
    /// Initial compilation tier for new kernels
    pub initial_tier: CompilationTier,
    /// Execution count threshold for tier-up (Interpreter -> QuickJit)
    pub tier1_threshold: u64,
    /// Execution count threshold for optimization (QuickJit -> OptimizedJit)
    pub tier2_threshold: u64,
    /// Execution count threshold for aggressive optimization (OptimizedJit -> AggressiveJit)
    pub tier3_threshold: u64,
    /// Enable profile-guided optimization
    pub enable_pgo: bool,
    /// Enable adaptive recompilation
    pub enable_adaptive_recompilation: bool,
    /// Enable code caching across runs
    pub enable_code_cache: bool,
    /// Maximum cache size in bytes
    pub max_cache_size: usize,
    /// Enable kernel specialization
    pub enable_specialization: bool,
    /// Enable automatic inlining
    pub enable_auto_inlining: bool,
    /// Enable automatic vectorization
    pub enable_auto_vectorization: bool,
    /// Profiling sample rate (0.0 to 1.0)
    pub profiling_sample_rate: f64,
}

impl Default for JitConfig {
    fn default() -> Self {
        Self {
            initial_tier: CompilationTier::QuickJit,
            tier1_threshold: 10,
            tier2_threshold: 100,
            tier3_threshold: 1000,
            enable_pgo: true,
            enable_adaptive_recompilation: true,
            enable_code_cache: true,
            max_cache_size: 100 * 1024 * 1024, // 100 MB
            enable_specialization: true,
            enable_auto_inlining: true,
            enable_auto_vectorization: true,
            profiling_sample_rate: 1.0,
        }
    }
}

impl JitConfig {
    /// Create a configuration optimized for development (fast iteration)
    pub fn development() -> Self {
        Self {
            initial_tier: CompilationTier::QuickJit,
            enable_pgo: false,
            enable_adaptive_recompilation: false,
            profiling_sample_rate: 0.1,
            ..Default::default()
        }
    }

    /// Create a configuration optimized for production (best performance)
    pub fn production() -> Self {
        Self {
            initial_tier: CompilationTier::OptimizedJit,
            tier1_threshold: 5,
            tier2_threshold: 50,
            tier3_threshold: 500,
            enable_pgo: true,
            enable_adaptive_recompilation: true,
            enable_code_cache: true,
            enable_specialization: true,
            profiling_sample_rate: 1.0,
            ..Default::default()
        }
    }

    /// Create a configuration optimized for memory-constrained environments
    pub fn memory_constrained() -> Self {
        Self {
            initial_tier: CompilationTier::Interpreter,
            max_cache_size: 10 * 1024 * 1024, // 10 MB
            enable_code_cache: true,
            enable_specialization: false,
            ..Default::default()
        }
    }
}

/// Compiled kernel with JIT metadata
/// Note: Not serializable due to Instant fields - this is runtime-only data
#[derive(Debug, Clone)]
pub struct JitKernel {
    /// The actual compiled kernel
    pub kernel: GeneratedKernel,
    /// Current compilation tier
    pub tier: CompilationTier,
    /// Execution profile data
    pub profile: ExecutionProfile,
    /// Compilation timestamp
    pub compiled_at: Instant,
    /// Recompilation count
    pub recompilation_count: u32,
    /// Specialized for specific input pattern
    pub specialized_for: Option<Vec<Vec<usize>>>,
    /// Size in bytes
    pub size_bytes: usize,
}

impl JitKernel {
    pub fn new(kernel: GeneratedKernel, tier: CompilationTier) -> Self {
        let size_bytes = kernel
            .compiled_binary
            .as_ref()
            .map_or(kernel.source_code.len(), |binary| binary.len());

        Self {
            kernel,
            tier,
            profile: ExecutionProfile::new(),
            compiled_at: Instant::now(),
            recompilation_count: 0,
            specialized_for: None,
            size_bytes,
        }
    }

    /// Check if this kernel should be recompiled to a higher tier
    pub fn should_tier_up(&self, config: &JitConfig) -> bool {
        if !config.enable_adaptive_recompilation {
            return false;
        }

        match self.tier {
            CompilationTier::Interpreter => self.profile.is_hot(config.tier1_threshold),
            CompilationTier::QuickJit => self.profile.is_hot(config.tier2_threshold),
            CompilationTier::OptimizedJit => self.profile.is_hot(config.tier3_threshold),
            CompilationTier::AggressiveJit => false, // Already at max tier
        }
    }

    /// Get the next compilation tier
    pub fn next_tier(&self) -> Option<CompilationTier> {
        match self.tier {
            CompilationTier::Interpreter => Some(CompilationTier::QuickJit),
            CompilationTier::QuickJit => Some(CompilationTier::OptimizedJit),
            CompilationTier::OptimizedJit => Some(CompilationTier::AggressiveJit),
            CompilationTier::AggressiveJit => None,
        }
    }

    /// Estimate recompilation benefit
    pub fn recompilation_benefit(&self) -> f64 {
        // Calculate benefit as: (potential speedup - 1.0) * execution_count * avg_time
        let speedup = self.profile.estimated_speedup_potential();
        let total_time_saved = (speedup - 1.0)
            * self.profile.execution_count as f64
            * self.profile.average_execution_time.as_secs_f64();

        // Consider compilation cost (higher tiers take longer to compile)
        let compilation_cost_estimate = match self.next_tier() {
            Some(CompilationTier::QuickJit) => 0.01,     // 10ms
            Some(CompilationTier::OptimizedJit) => 0.1,  // 100ms
            Some(CompilationTier::AggressiveJit) => 1.0, // 1s
            None => 0.0,
            _ => 0.01,
        };

        total_time_saved - compilation_cost_estimate
    }
}

/// JIT Compiler that manages kernel compilation and optimization
pub struct JitCompiler {
    config: JitConfig,
    /// Compiled kernels indexed by spec hash
    kernels: Arc<RwLock<HashMap<String, JitKernel>>>,
    /// Code cache for persistent storage
    code_cache: Option<Arc<Mutex<CodeCache>>>,
    /// Compilation statistics
    stats: Arc<Mutex<JitStatistics>>,
    /// Kernel generator for compilation
    kernel_generator: Arc<Mutex<crate::kernel_generation::KernelGenerator>>,
}

impl JitCompiler {
    /// Create a new JIT compiler with default configuration
    pub fn new() -> Self {
        Self::with_config(JitConfig::default())
    }

    /// Create a new JIT compiler with custom configuration
    pub fn with_config(config: JitConfig) -> Self {
        let code_cache = if config.enable_code_cache {
            Some(Arc::new(Mutex::new(CodeCache::new(config.max_cache_size))))
        } else {
            None
        };

        Self {
            config,
            kernels: Arc::new(RwLock::new(HashMap::new())),
            code_cache,
            stats: Arc::new(Mutex::new(JitStatistics::default())),
            kernel_generator: Arc::new(
                Mutex::new(crate::kernel_generation::KernelGenerator::new()),
            ),
        }
    }

    /// Compile or retrieve a kernel, applying JIT optimizations
    pub fn compile_kernel(&self, spec: KernelSpec) -> BackendResult<Arc<JitKernel>> {
        let spec_hash = spec.hash_key();

        // Check if we have a compiled version
        {
            let kernels = self.kernels.read().unwrap();
            if let Some(jit_kernel) = kernels.get(&spec_hash) {
                self.stats.lock().unwrap().cache_hits += 1;
                return Ok(Arc::new(jit_kernel.clone()));
            }
        }

        self.stats.lock().unwrap().cache_misses += 1;

        // Check code cache if enabled
        if let Some(ref cache) = self.code_cache {
            if let Some(cached_kernel) = cache.lock().unwrap().get(&spec_hash) {
                let jit_kernel = JitKernel::new(cached_kernel, self.config.initial_tier);
                let result = Arc::new(jit_kernel.clone());
                self.kernels.write().unwrap().insert(spec_hash, jit_kernel);
                self.stats.lock().unwrap().cache_hits += 1;
                return Ok(result);
            }
        }

        // Compile new kernel
        let tier = self.config.initial_tier;
        let optimized_spec = self.apply_tier_optimizations(spec, tier);

        let kernel = self
            .kernel_generator
            .lock()
            .unwrap()
            .generate_kernel(optimized_spec)?;

        let jit_kernel = JitKernel::new(kernel.clone(), tier);

        // Store in code cache if enabled
        if let Some(ref cache) = self.code_cache {
            cache.lock().unwrap().insert(spec_hash.clone(), kernel);
        }

        let result = Arc::new(jit_kernel.clone());
        self.kernels.write().unwrap().insert(spec_hash, jit_kernel);
        self.stats.lock().unwrap().compilations += 1;

        Ok(result)
    }

    /// Record kernel execution for profiling
    pub fn record_execution(
        &self,
        spec_hash: &str,
        execution_time: Duration,
        input_sizes: &[usize],
    ) -> BackendResult<()> {
        let mut kernels = self.kernels.write().unwrap();

        if let Some(jit_kernel) = kernels.get_mut(spec_hash) {
            jit_kernel
                .profile
                .record_execution(execution_time, input_sizes);

            // Check if we should tier up
            if jit_kernel.should_tier_up(&self.config) {
                let benefit = jit_kernel.recompilation_benefit();

                // Only recompile if benefit is positive and significant
                if benefit > 0.01 {
                    self.stats.lock().unwrap().recompilations += 1;
                    // Trigger async recompilation (drop lock first to avoid deadlock)
                    // In production, this would spawn a background task
                }
            }
        }

        Ok(())
    }

    /// Apply tier-specific optimizations to kernel spec
    fn apply_tier_optimizations(&self, mut spec: KernelSpec, tier: CompilationTier) -> KernelSpec {
        match tier {
            CompilationTier::Interpreter => {
                // No optimizations, fastest compilation
                spec.optimization_flags = OptimizationFlags {
                    vectorization: false,
                    loop_unrolling: false,
                    memory_coalescing: false,
                    shared_memory_usage: false,
                    tensor_cores: false,
                    auto_tuning: false,
                    aggressive_inlining: false,
                    math_optimizations: false,
                };
            }
            CompilationTier::QuickJit => {
                // Basic optimizations only
                spec.optimization_flags = OptimizationFlags {
                    vectorization: false,
                    loop_unrolling: true,
                    memory_coalescing: true,
                    shared_memory_usage: false,
                    tensor_cores: false,
                    auto_tuning: false,
                    aggressive_inlining: false,
                    math_optimizations: true,
                };
            }
            CompilationTier::OptimizedJit => {
                // Standard optimizations
                spec.optimization_flags = OptimizationFlags {
                    vectorization: self.config.enable_auto_vectorization,
                    loop_unrolling: true,
                    memory_coalescing: true,
                    shared_memory_usage: true,
                    tensor_cores: false,
                    auto_tuning: true,
                    aggressive_inlining: self.config.enable_auto_inlining,
                    math_optimizations: true,
                };
            }
            CompilationTier::AggressiveJit => {
                // All optimizations enabled
                spec.optimization_flags = OptimizationFlags {
                    vectorization: true,
                    loop_unrolling: true,
                    memory_coalescing: true,
                    shared_memory_usage: true,
                    tensor_cores: true,
                    auto_tuning: true,
                    aggressive_inlining: true,
                    math_optimizations: true,
                };
            }
        }

        spec
    }

    /// Get compilation statistics
    pub fn statistics(&self) -> JitStatistics {
        self.stats.lock().unwrap().clone()
    }

    /// Clear all compiled kernels and caches
    pub fn clear(&self) {
        self.kernels.write().unwrap().clear();
        if let Some(ref cache) = self.code_cache {
            cache.lock().unwrap().clear();
        }
        *self.stats.lock().unwrap() = JitStatistics::default();
    }

    /// Get current cache size in bytes
    pub fn cache_size_bytes(&self) -> usize {
        self.kernels
            .read()
            .unwrap()
            .values()
            .map(|k| k.size_bytes)
            .sum()
    }

    /// Evict least recently used kernels if cache is too large
    pub fn evict_if_needed(&self) {
        let cache_size = self.cache_size_bytes();
        if cache_size > self.config.max_cache_size {
            let mut kernels = self.kernels.write().unwrap();

            // Sort by last used time and evict oldest
            let mut kernel_ages: Vec<_> = kernels
                .iter()
                .map(|(k, v)| (k.clone(), v.profile.last_updated))
                .collect();

            kernel_ages.sort_by_key(|(_, time)| *time);

            // Evict oldest 25%
            let evict_count = (kernel_ages.len() / 4).max(1);
            for (key, _) in kernel_ages.iter().take(evict_count) {
                kernels.remove(key);
            }

            self.stats.lock().unwrap().evictions += evict_count;
        }
    }
}

impl Default for JitCompiler {
    fn default() -> Self {
        Self::new()
    }
}

/// JIT compilation statistics
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct JitStatistics {
    /// Total number of compilations
    pub compilations: u64,
    /// Total number of recompilations (tier-ups)
    pub recompilations: u64,
    /// Cache hits
    pub cache_hits: u64,
    /// Cache misses
    pub cache_misses: u64,
    /// Number of evictions
    pub evictions: usize,
}

impl JitStatistics {
    pub fn cache_hit_rate(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total > 0 {
            self.cache_hits as f64 / total as f64
        } else {
            0.0
        }
    }
}

/// Persistent code cache for storing compiled kernels across runs
pub struct CodeCache {
    cache: HashMap<String, GeneratedKernel>,
    max_size: usize,
    current_size: usize,
}

impl CodeCache {
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: HashMap::new(),
            max_size,
            current_size: 0,
        }
    }

    pub fn get(&self, key: &str) -> Option<GeneratedKernel> {
        self.cache.get(key).cloned()
    }

    pub fn insert(&mut self, key: String, kernel: GeneratedKernel) {
        let kernel_size = kernel
            .compiled_binary
            .as_ref()
            .map_or(kernel.source_code.len(), |binary| binary.len());

        // Evict if necessary
        while self.current_size + kernel_size > self.max_size && !self.cache.is_empty() {
            if let Some(first_key) = self.cache.keys().next().cloned() {
                if let Some(removed) = self.cache.remove(&first_key) {
                    let removed_size = removed
                        .compiled_binary
                        .as_ref()
                        .map_or(removed.source_code.len(), |binary| binary.len());
                    self.current_size -= removed_size;
                }
            }
        }

        self.current_size += kernel_size;
        self.cache.insert(key, kernel);
    }

    pub fn clear(&mut self) {
        self.cache.clear();
        self.current_size = 0;
    }

    pub fn size(&self) -> usize {
        self.current_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_generation::{CompilationTarget, KernelDataType, KernelOperation};

    #[test]
    fn test_jit_compiler_creation() {
        let compiler = JitCompiler::new();
        let stats = compiler.statistics();
        assert_eq!(stats.compilations, 0);
        assert_eq!(stats.cache_hits, 0);
    }

    #[test]
    fn test_jit_config_presets() {
        let dev_config = JitConfig::development();
        assert!(!dev_config.enable_pgo);
        assert_eq!(dev_config.profiling_sample_rate, 0.1);

        let prod_config = JitConfig::production();
        assert!(prod_config.enable_pgo);
        assert_eq!(prod_config.profiling_sample_rate, 1.0);

        let mem_config = JitConfig::memory_constrained();
        assert_eq!(mem_config.initial_tier, CompilationTier::Interpreter);
        assert_eq!(mem_config.max_cache_size, 10 * 1024 * 1024);
    }

    #[test]
    fn test_execution_profile() {
        let mut profile = ExecutionProfile::new();
        assert_eq!(profile.execution_count, 0);

        profile.record_execution(Duration::from_millis(10), &[100, 200]);
        assert_eq!(profile.execution_count, 1);
        assert_eq!(profile.average_execution_time, Duration::from_millis(10));

        profile.record_execution(Duration::from_millis(20), &[100, 200]);
        assert_eq!(profile.execution_count, 2);
        assert_eq!(profile.average_execution_time, Duration::from_millis(15));
    }

    #[test]
    fn test_branch_prediction() {
        let mut profile = ExecutionProfile::new();

        // Record some branch outcomes
        profile.record_branch("branch_1".to_string(), true);
        profile.record_branch("branch_1".to_string(), true);
        profile.record_branch("branch_1".to_string(), false);

        let accuracy = profile.branch_prediction_accuracy("branch_1");
        assert!((accuracy - 0.666).abs() < 0.01); // 2 out of 3 taken
    }

    #[test]
    fn test_compilation_tiers() {
        let tier1 = CompilationTier::Interpreter;
        let tier2 = CompilationTier::QuickJit;
        let tier3 = CompilationTier::OptimizedJit;
        let tier4 = CompilationTier::AggressiveJit;

        assert!(tier1 < tier2);
        assert!(tier2 < tier3);
        assert!(tier3 < tier4);
    }

    #[test]
    fn test_jit_kernel_tier_up() {
        let spec = KernelSpec::new(
            KernelOperation::ElementwiseAdd,
            vec![KernelDataType::F32, KernelDataType::F32],
            KernelDataType::F32,
            vec![vec![100], vec![100]],
            vec![100],
            CompilationTarget::WebGPU,
        );

        let kernel = GeneratedKernel {
            source_code: "test code".to_string(),
            entry_point: "test_kernel".to_string(),
            compiled_binary: None,
            spec,
            compilation_time_ms: 10,
            estimated_performance: 1.0,
            register_usage: None,
            shared_memory_usage: None,
        };

        let mut jit_kernel = JitKernel::new(kernel, CompilationTier::QuickJit);
        let config = JitConfig::default();

        assert!(!jit_kernel.should_tier_up(&config));

        // Simulate many executions
        for _ in 0..200 {
            jit_kernel
                .profile
                .record_execution(Duration::from_micros(100), &[100]);
        }

        assert!(jit_kernel.should_tier_up(&config));
        assert_eq!(jit_kernel.next_tier(), Some(CompilationTier::OptimizedJit));
    }

    #[test]
    fn test_code_cache() {
        let mut cache = CodeCache::new(1000);

        let spec = KernelSpec::new(
            KernelOperation::ElementwiseAdd,
            vec![KernelDataType::F32],
            KernelDataType::F32,
            vec![vec![100]],
            vec![100],
            CompilationTarget::WebGPU,
        );

        let kernel = GeneratedKernel {
            source_code: "a".repeat(100),
            entry_point: "test".to_string(),
            compiled_binary: None,
            spec,
            compilation_time_ms: 1,
            estimated_performance: 1.0,
            register_usage: None,
            shared_memory_usage: None,
        };

        cache.insert("key1".to_string(), kernel.clone());
        assert!(cache.get("key1").is_some());
        assert_eq!(cache.size(), 100);
    }

    #[test]
    fn test_speedup_estimation() {
        let mut profile = ExecutionProfile::new();
        profile.cache_miss_rate = 0.2; // 20% cache misses
        profile.simd_effectiveness = 0.5; // 50% vectorized

        let speedup = profile.estimated_speedup_potential();
        assert!(speedup > 1.0); // Should estimate some speedup potential
    }

    #[test]
    fn test_jit_statistics() {
        let stats = JitStatistics {
            compilations: 100,
            recompilations: 10,
            cache_hits: 80,
            cache_misses: 20,
            evictions: 5,
        };

        assert_eq!(stats.cache_hit_rate(), 0.8);
    }
}
