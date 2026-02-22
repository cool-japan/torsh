//! Function optimization and fusion framework
//!
//! This module provides advanced optimization techniques for autograd functions,
//! including operation fusion, pattern matching, and performance optimization.

use crate::function::FunctionMetadata;
use std::collections::HashMap;
use torsh_core::error::{Result, TorshError};
use tracing::{debug, info};

/// Optimization strategies for autograd functions
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptimizationStrategy {
    /// Fuse sequential operations
    SequentialFusion,
    /// Fuse element-wise operations
    ElementWiseFusion,
    /// Fuse matrix operations
    MatrixFusion,
    /// Eliminate common subexpressions
    CommonSubexpressionElimination,
    /// Dead code elimination
    DeadCodeElimination,
    /// Constant folding
    ConstantFolding,
    /// Memory layout optimization
    MemoryLayoutOptimization,
    /// SIMD vectorization
    SIMDVectorization,
}

/// Configuration for function optimization
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    /// Enabled optimization strategies
    pub enabled_strategies: Vec<OptimizationStrategy>,
    /// Maximum fusion group size
    pub max_fusion_size: usize,
    /// Memory threshold for optimizations
    pub memory_threshold: usize,
    /// Compute threshold for optimizations
    pub compute_threshold: f32,
    /// Enable aggressive optimizations
    pub aggressive_mode: bool,
    /// Profile-guided optimization
    pub enable_pgo: bool,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            enabled_strategies: vec![
                OptimizationStrategy::SequentialFusion,
                OptimizationStrategy::ElementWiseFusion,
                OptimizationStrategy::CommonSubexpressionElimination,
                OptimizationStrategy::DeadCodeElimination,
            ],
            max_fusion_size: 8,
            memory_threshold: 64 * 1024 * 1024, // 64MB
            compute_threshold: 1.5,             // 50% overhead max
            aggressive_mode: false,
            enable_pgo: false,
        }
    }
}

/// Function pattern for optimization matching
#[derive(Debug, Clone)]
pub struct FunctionPattern {
    /// Pattern name
    pub name: String,
    /// Sequence of operation names that match this pattern
    pub operations: Vec<String>,
    /// Required properties for matching
    pub required_properties: Vec<PatternProperty>,
    /// Optimization strategy to apply
    pub optimization: OptimizationStrategy,
    /// Priority for pattern matching
    pub priority: i32,
}

/// Properties required for pattern matching
#[derive(Debug, Clone, PartialEq)]
pub enum PatternProperty {
    /// Operations must be consecutive
    Consecutive,
    /// Operations must have compatible shapes
    CompatibleShapes,
    /// Operations must be element-wise
    ElementWise,
    /// Operations must be commutative
    Commutative,
    /// Operations must have no side effects
    NoSideEffects,
}

/// Fusable operation group
#[derive(Debug, Clone)]
pub struct FusionGroup {
    /// Operations in the fusion group
    pub operations: Vec<FunctionInfo>,
    /// Fusion strategy to apply
    pub strategy: OptimizationStrategy,
    /// Estimated performance gain
    pub performance_gain: f32,
    /// Estimated memory savings
    pub memory_savings: usize,
}

/// Information about a function for optimization
#[derive(Debug, Clone)]
pub struct FunctionInfo {
    /// Function identifier
    pub id: usize,
    /// Function name
    pub name: String,
    /// Function metadata
    pub metadata: FunctionMetadata,
    /// Input shapes (if known)
    pub input_shapes: Vec<Vec<usize>>,
    /// Output shapes (if known)
    pub output_shapes: Vec<Vec<usize>>,
    /// Dependencies
    pub dependencies: Vec<usize>,
    /// Performance profile data
    pub profile_data: Option<ProfileData>,
}

/// Performance profiling data for functions
#[derive(Debug, Clone)]
pub struct ProfileData {
    /// Average execution time (milliseconds)
    pub avg_execution_time: f32,
    /// Memory usage (bytes)
    pub memory_usage: usize,
    /// Cache hit rate
    pub cache_hit_rate: f32,
    /// Number of executions
    pub execution_count: usize,
}

/// Function optimizer and fusion engine
pub struct FunctionOptimizer {
    config: OptimizationConfig,
    patterns: Vec<FunctionPattern>,
    function_registry: HashMap<usize, FunctionInfo>,
    optimization_history: Vec<OptimizationResult>,
    #[allow(dead_code)]
    profile_database: HashMap<String, ProfileData>,
}

/// Result of an optimization pass
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// Strategy that was applied
    pub strategy: OptimizationStrategy,
    /// Functions that were optimized
    pub optimized_functions: Vec<usize>,
    /// Performance improvement estimate
    pub performance_improvement: f32,
    /// Memory savings estimate
    pub memory_savings: usize,
    /// Whether optimization was successful
    pub success: bool,
    /// Error message if optimization failed
    pub error_message: Option<String>,
}

impl FunctionOptimizer {
    /// Create a new function optimizer
    pub fn new(config: OptimizationConfig) -> Self {
        let mut optimizer = Self {
            config,
            patterns: Vec::new(),
            function_registry: HashMap::new(),
            optimization_history: Vec::new(),
            profile_database: HashMap::new(),
        };

        optimizer.initialize_default_patterns();
        optimizer
    }

    /// Initialize default optimization patterns
    fn initialize_default_patterns(&mut self) {
        // Element-wise fusion patterns
        self.patterns.push(FunctionPattern {
            name: "ElementWise_Add_Mul".to_string(),
            operations: vec!["add".to_string(), "mul".to_string()],
            required_properties: vec![
                PatternProperty::Consecutive,
                PatternProperty::ElementWise,
                PatternProperty::CompatibleShapes,
            ],
            optimization: OptimizationStrategy::ElementWiseFusion,
            priority: 100,
        });

        self.patterns.push(FunctionPattern {
            name: "ElementWise_Chain".to_string(),
            operations: vec!["relu".to_string(), "mul".to_string(), "add".to_string()],
            required_properties: vec![PatternProperty::Consecutive, PatternProperty::ElementWise],
            optimization: OptimizationStrategy::ElementWiseFusion,
            priority: 90,
        });

        // Matrix operation fusion patterns
        self.patterns.push(FunctionPattern {
            name: "MatMul_Add".to_string(),
            operations: vec!["matmul".to_string(), "add".to_string()],
            required_properties: vec![
                PatternProperty::Consecutive,
                PatternProperty::CompatibleShapes,
            ],
            optimization: OptimizationStrategy::MatrixFusion,
            priority: 95,
        });

        // Sequential operation patterns
        self.patterns.push(FunctionPattern {
            name: "Sequential_Activation".to_string(),
            operations: vec!["linear".to_string(), "relu".to_string()],
            required_properties: vec![PatternProperty::Consecutive, PatternProperty::NoSideEffects],
            optimization: OptimizationStrategy::SequentialFusion,
            priority: 85,
        });

        info!("Initialized {} optimization patterns", self.patterns.len());
    }

    /// Register a function for optimization tracking
    pub fn register_function(&mut self, function_info: FunctionInfo) {
        self.function_registry
            .insert(function_info.id, function_info);
    }

    /// Optimize a sequence of functions
    pub fn optimize_functions(
        &mut self,
        function_ids: &[usize],
    ) -> Result<Vec<OptimizationResult>> {
        let mut results = Vec::new();

        for &strategy in &self.config.enabled_strategies.clone() {
            if let Ok(result) = self.apply_optimization_strategy(strategy, function_ids) {
                results.push(result);
            }
        }

        // Store optimization history
        self.optimization_history.extend(results.iter().cloned());

        info!(
            "Applied {} optimization strategies to {} functions",
            results.len(),
            function_ids.len()
        );

        Ok(results)
    }

    /// Apply a specific optimization strategy
    fn apply_optimization_strategy(
        &mut self,
        strategy: OptimizationStrategy,
        function_ids: &[usize],
    ) -> Result<OptimizationResult> {
        match strategy {
            OptimizationStrategy::SequentialFusion => self.apply_sequential_fusion(function_ids),
            OptimizationStrategy::ElementWiseFusion => self.apply_element_wise_fusion(function_ids),
            OptimizationStrategy::MatrixFusion => self.apply_matrix_fusion(function_ids),
            OptimizationStrategy::CommonSubexpressionElimination => self.apply_cse(function_ids),
            OptimizationStrategy::DeadCodeElimination => self.apply_dce(function_ids),
            OptimizationStrategy::ConstantFolding => self.apply_constant_folding(function_ids),
            OptimizationStrategy::MemoryLayoutOptimization => {
                self.apply_memory_layout_optimization(function_ids)
            }
            OptimizationStrategy::SIMDVectorization => self.apply_simd_vectorization(function_ids),
        }
    }

    /// Apply sequential fusion optimization
    fn apply_sequential_fusion(&mut self, function_ids: &[usize]) -> Result<OptimizationResult> {
        let fusion_groups = self.find_sequential_fusion_opportunities(function_ids)?;

        let mut optimized_functions = Vec::new();
        let mut total_performance_gain = 0.0;
        let mut total_memory_savings = 0;

        for group in fusion_groups {
            // Apply fusion to the group
            let _fusion_result = self.fuse_sequential_operations(&group)?;

            optimized_functions.extend(group.operations.iter().map(|op| op.id));
            total_performance_gain += group.performance_gain;
            total_memory_savings += group.memory_savings;

            debug!(
                "Fused {} sequential operations with {:.2}% performance gain",
                group.operations.len(),
                group.performance_gain * 100.0
            );
        }

        Ok(OptimizationResult {
            strategy: OptimizationStrategy::SequentialFusion,
            optimized_functions,
            performance_improvement: total_performance_gain,
            memory_savings: total_memory_savings,
            success: true,
            error_message: None,
        })
    }

    /// Apply element-wise fusion optimization
    fn apply_element_wise_fusion(&mut self, function_ids: &[usize]) -> Result<OptimizationResult> {
        let fusion_groups = self.find_element_wise_fusion_opportunities(function_ids)?;

        let mut optimized_functions = Vec::new();
        let mut total_performance_gain = 0.0;
        let mut total_memory_savings = 0;

        for group in fusion_groups {
            optimized_functions.extend(group.operations.iter().map(|op| op.id));
            total_performance_gain += group.performance_gain;
            total_memory_savings += group.memory_savings;

            debug!("Fused {} element-wise operations", group.operations.len());
        }

        Ok(OptimizationResult {
            strategy: OptimizationStrategy::ElementWiseFusion,
            optimized_functions,
            performance_improvement: total_performance_gain,
            memory_savings: total_memory_savings,
            success: true,
            error_message: None,
        })
    }

    /// Apply matrix fusion optimization
    fn apply_matrix_fusion(&mut self, function_ids: &[usize]) -> Result<OptimizationResult> {
        let mut optimized_functions = Vec::new();
        let mut performance_gain = 0.0;
        let mut memory_savings = 0;

        // Look for matrix multiplication followed by addition (GEMM pattern)
        for window in function_ids.windows(2) {
            if let (Some(func1), Some(func2)) = (
                self.function_registry.get(&window[0]),
                self.function_registry.get(&window[1]),
            ) {
                if func1.name == "matmul" && func2.name == "add" {
                    // This is a fusable GEMM pattern
                    optimized_functions.extend_from_slice(window);
                    performance_gain += 0.2; // 20% improvement estimate
                    memory_savings += 1024 * 1024; // 1MB savings estimate

                    debug!("Fused MatMul+Add pattern");
                }
            }
        }

        let success = !optimized_functions.is_empty();
        Ok(OptimizationResult {
            strategy: OptimizationStrategy::MatrixFusion,
            optimized_functions,
            performance_improvement: performance_gain,
            memory_savings,
            success,
            error_message: None,
        })
    }

    /// Apply common subexpression elimination
    fn apply_cse(&mut self, function_ids: &[usize]) -> Result<OptimizationResult> {
        let mut optimized_functions = Vec::new();
        let mut expression_map: HashMap<String, Vec<usize>> = HashMap::new();

        // Group functions by their "signature" (name + input shapes)
        for &func_id in function_ids {
            if let Some(func_info) = self.function_registry.get(&func_id) {
                let signature = format!("{}_{:?}", func_info.name, func_info.input_shapes);
                expression_map.entry(signature).or_default().push(func_id);
            }
        }

        let mut eliminated_expressions = 0;

        // Find common subexpressions
        for (signature, func_ids) in expression_map {
            if func_ids.len() > 1 {
                // Found common subexpression
                optimized_functions.extend(&func_ids[1..]); // Keep first, eliminate rest
                eliminated_expressions += func_ids.len() - 1;

                debug!(
                    "Eliminated {} instances of common subexpression: {}",
                    func_ids.len() - 1,
                    signature
                );
            }
        }

        Ok(OptimizationResult {
            strategy: OptimizationStrategy::CommonSubexpressionElimination,
            optimized_functions,
            performance_improvement: eliminated_expressions as f32 * 0.1, // 10% per elimination
            memory_savings: eliminated_expressions * 1024,                // 1KB per elimination
            success: eliminated_expressions > 0,
            error_message: None,
        })
    }

    /// Apply dead code elimination
    fn apply_dce(&mut self, function_ids: &[usize]) -> Result<OptimizationResult> {
        let mut dead_functions = Vec::new();

        // Simple DCE: find functions with no dependencies in subsequent operations
        for &func_id in function_ids {
            if self.is_function_dead(func_id, function_ids) {
                dead_functions.push(func_id);
            }
        }

        debug!(
            "Found {} dead functions for elimination",
            dead_functions.len()
        );

        Ok(OptimizationResult {
            strategy: OptimizationStrategy::DeadCodeElimination,
            optimized_functions: dead_functions.clone(),
            performance_improvement: dead_functions.len() as f32 * 0.05, // 5% per dead function
            memory_savings: dead_functions.len() * 512,                  // 512B per dead function
            success: !dead_functions.is_empty(),
            error_message: None,
        })
    }

    /// Apply constant folding optimization
    fn apply_constant_folding(&mut self, function_ids: &[usize]) -> Result<OptimizationResult> {
        let mut folded_functions = Vec::new();

        // Look for operations with constant inputs
        for &func_id in function_ids {
            if let Some(func_info) = self.function_registry.get(&func_id) {
                if self.can_constant_fold(&func_info) {
                    folded_functions.push(func_id);
                }
            }
        }

        debug!(
            "Found {} functions for constant folding",
            folded_functions.len()
        );

        Ok(OptimizationResult {
            strategy: OptimizationStrategy::ConstantFolding,
            optimized_functions: folded_functions.clone(),
            performance_improvement: folded_functions.len() as f32 * 0.3, // 30% per folded function
            memory_savings: folded_functions.len() * 256, // 256B per folded function
            success: !folded_functions.is_empty(),
            error_message: None,
        })
    }

    /// Apply memory layout optimization
    fn apply_memory_layout_optimization(
        &mut self,
        function_ids: &[usize],
    ) -> Result<OptimizationResult> {
        let mut optimized_functions = Vec::new();

        // Look for opportunities to reorder operations for better memory access patterns
        for &func_id in function_ids {
            if let Some(func_info) = self.function_registry.get(&func_id) {
                if self.can_optimize_memory_layout(&func_info) {
                    optimized_functions.push(func_id);
                }
            }
        }

        debug!(
            "Found {} functions for memory layout optimization",
            optimized_functions.len()
        );

        Ok(OptimizationResult {
            strategy: OptimizationStrategy::MemoryLayoutOptimization,
            optimized_functions: optimized_functions.clone(),
            performance_improvement: optimized_functions.len() as f32 * 0.15, // 15% per optimization
            memory_savings: 0, // Memory layout doesn't save memory, but improves access patterns
            success: !optimized_functions.is_empty(),
            error_message: None,
        })
    }

    /// Apply SIMD vectorization
    fn apply_simd_vectorization(&mut self, function_ids: &[usize]) -> Result<OptimizationResult> {
        let mut vectorized_functions = Vec::new();

        // Look for element-wise operations that can be vectorized
        for &func_id in function_ids {
            if let Some(func_info) = self.function_registry.get(&func_id) {
                if self.can_vectorize(&func_info) {
                    vectorized_functions.push(func_id);
                }
            }
        }

        debug!(
            "Found {} functions for SIMD vectorization",
            vectorized_functions.len()
        );

        Ok(OptimizationResult {
            strategy: OptimizationStrategy::SIMDVectorization,
            optimized_functions: vectorized_functions.clone(),
            performance_improvement: vectorized_functions.len() as f32 * 0.4, // 40% per vectorized function
            memory_savings: 0, // SIMD doesn't save memory but improves throughput
            success: !vectorized_functions.is_empty(),
            error_message: None,
        })
    }

    /// Find sequential fusion opportunities
    fn find_sequential_fusion_opportunities(
        &self,
        function_ids: &[usize],
    ) -> Result<Vec<FusionGroup>> {
        let mut fusion_groups = Vec::new();

        // Use a sliding window to find consecutive operations that can be fused
        for window_size in (2..=self.config.max_fusion_size).rev() {
            for window in function_ids.windows(window_size) {
                if let Some(group) =
                    self.analyze_fusion_group(window, OptimizationStrategy::SequentialFusion)?
                {
                    fusion_groups.push(group);
                }
            }
        }

        Ok(fusion_groups)
    }

    /// Find element-wise fusion opportunities
    fn find_element_wise_fusion_opportunities(
        &self,
        function_ids: &[usize],
    ) -> Result<Vec<FusionGroup>> {
        let mut fusion_groups = Vec::new();

        // Look for chains of element-wise operations
        let mut current_group = Vec::new();

        for &func_id in function_ids {
            if let Some(func_info) = self.function_registry.get(&func_id) {
                if self.is_element_wise_operation(&func_info) {
                    current_group.push(func_info.clone());
                } else {
                    if current_group.len() >= 2 {
                        if let Some(group) = self.create_fusion_group(
                            current_group.clone(),
                            OptimizationStrategy::ElementWiseFusion,
                        )? {
                            fusion_groups.push(group);
                        }
                    }
                    current_group.clear();
                }
            }
        }

        // Handle remaining group
        if current_group.len() >= 2 {
            if let Some(group) =
                self.create_fusion_group(current_group, OptimizationStrategy::ElementWiseFusion)?
            {
                fusion_groups.push(group);
            }
        }

        Ok(fusion_groups)
    }

    /// Analyze a potential fusion group
    fn analyze_fusion_group(
        &self,
        function_ids: &[usize],
        strategy: OptimizationStrategy,
    ) -> Result<Option<FusionGroup>> {
        let operations: Result<Vec<_>> = function_ids
            .iter()
            .map(|&id| {
                self.function_registry
                    .get(&id)
                    .ok_or_else(|| TorshError::AutogradError(format!("Function {} not found", id)))
                    .map(|f| f.clone())
            })
            .collect();

        let operations = operations?;
        self.create_fusion_group(operations, strategy)
    }

    /// Create a fusion group from operations
    fn create_fusion_group(
        &self,
        operations: Vec<FunctionInfo>,
        strategy: OptimizationStrategy,
    ) -> Result<Option<FusionGroup>> {
        if operations.len() < 2 {
            return Ok(None);
        }

        // Estimate performance gain and memory savings
        let performance_gain = self.estimate_fusion_performance_gain(&operations, strategy);
        let memory_savings = self.estimate_fusion_memory_savings(&operations, strategy);

        // Only create fusion group if it's beneficial
        if performance_gain > 0.05 || memory_savings > 1024 {
            // 5% gain or 1KB savings
            Ok(Some(FusionGroup {
                operations,
                strategy,
                performance_gain,
                memory_savings,
            }))
        } else {
            Ok(None)
        }
    }

    /// Estimate performance gain from fusion
    fn estimate_fusion_performance_gain(
        &self,
        operations: &[FunctionInfo],
        strategy: OptimizationStrategy,
    ) -> f32 {
        let base_gain = match strategy {
            OptimizationStrategy::SequentialFusion => 0.1,
            OptimizationStrategy::ElementWiseFusion => 0.2,
            OptimizationStrategy::MatrixFusion => 0.3,
            _ => 0.05,
        };

        base_gain * (operations.len() - 1) as f32
    }

    /// Estimate memory savings from fusion
    fn estimate_fusion_memory_savings(
        &self,
        operations: &[FunctionInfo],
        _strategy: OptimizationStrategy,
    ) -> usize {
        // Rough estimate: save intermediate results
        (operations.len() - 1) * 1024 // 1KB per eliminated intermediate
    }

    /// Check if a function is element-wise
    fn is_element_wise_operation(&self, func_info: &FunctionInfo) -> bool {
        matches!(
            func_info.name.as_str(),
            "add" | "mul" | "sub" | "div" | "relu" | "sigmoid" | "tanh"
        )
    }

    /// Check if a function is dead (unused)
    fn is_function_dead(&self, func_id: usize, all_functions: &[usize]) -> bool {
        // Simple check: if no other function depends on this one
        for &other_id in all_functions {
            if other_id != func_id {
                if let Some(other_func) = self.function_registry.get(&other_id) {
                    if other_func.dependencies.contains(&func_id) {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Check if a function can be constant folded
    fn can_constant_fold(&self, func_info: &FunctionInfo) -> bool {
        // Simplified check: functions with no dependencies might be constant
        func_info.dependencies.is_empty()
            && matches!(func_info.name.as_str(), "constant" | "zeros" | "ones")
    }

    /// Check if memory layout can be optimized
    fn can_optimize_memory_layout(&self, func_info: &FunctionInfo) -> bool {
        // Look for operations that would benefit from memory reordering
        matches!(func_info.name.as_str(), "transpose" | "reshape" | "permute")
    }

    /// Check if function can be vectorized
    fn can_vectorize(&self, func_info: &FunctionInfo) -> bool {
        // Element-wise operations are good candidates for vectorization
        self.is_element_wise_operation(func_info)
            && func_info
                .input_shapes
                .iter()
                .any(|shape| shape.iter().product::<usize>() > 64)
    }

    /// Actually fuse sequential operations (placeholder)
    fn fuse_sequential_operations(&mut self, _group: &FusionGroup) -> Result<()> {
        // In a real implementation, this would create a new fused function
        // and update the computation graph
        Ok(())
    }

    /// Get optimization statistics
    pub fn get_optimization_stats(&self) -> OptimizationStats {
        let total_optimizations = self.optimization_history.len();
        let successful_optimizations = self
            .optimization_history
            .iter()
            .filter(|r| r.success)
            .count();

        let total_performance_gain: f32 = self
            .optimization_history
            .iter()
            .map(|r| r.performance_improvement)
            .sum();

        let total_memory_savings: usize = self
            .optimization_history
            .iter()
            .map(|r| r.memory_savings)
            .sum();

        OptimizationStats {
            total_optimizations,
            successful_optimizations,
            success_rate: if total_optimizations > 0 {
                successful_optimizations as f32 / total_optimizations as f32
            } else {
                0.0
            },
            total_performance_gain,
            total_memory_savings,
            registered_functions: self.function_registry.len(),
            active_patterns: self.patterns.len(),
        }
    }
}

/// Statistics about optimization performance
#[derive(Debug, Clone)]
pub struct OptimizationStats {
    pub total_optimizations: usize,
    pub successful_optimizations: usize,
    pub success_rate: f32,
    pub total_performance_gain: f32,
    pub total_memory_savings: usize,
    pub registered_functions: usize,
    pub active_patterns: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::function::{ComputationalComplexity, MemoryComplexity};

    #[test]
    fn test_function_optimizer_creation() {
        let config = OptimizationConfig::default();
        let optimizer = FunctionOptimizer::new(config);

        assert!(!optimizer.patterns.is_empty());
        assert!(optimizer.function_registry.is_empty());
    }

    #[test]
    fn test_pattern_matching() {
        let config = OptimizationConfig::default();
        let mut optimizer = FunctionOptimizer::new(config);

        // Register some functions
        let func1 = FunctionInfo {
            id: 1,
            name: "add".to_string(),
            metadata: FunctionMetadata {
                name: "add".to_string(),
                is_differentiable: true,
                memory_complexity: MemoryComplexity::Linear,
                computational_complexity: ComputationalComplexity::Linear,
                is_fusable: true,
                version: "1.0.0".to_string(),
                description: "Element-wise addition".to_string(),
                author: "torsh-autograd".to_string(),
                created_at: "2024-01-01T00:00:00Z".to_string(),
                checksum: "".to_string(),
                dependencies: vec![],
            },
            input_shapes: vec![vec![10, 10]],
            output_shapes: vec![vec![10, 10]],
            dependencies: vec![],
            profile_data: None,
        };

        optimizer.register_function(func1);
        assert_eq!(optimizer.function_registry.len(), 1);
    }

    #[test]
    fn test_optimization_stats() {
        let config = OptimizationConfig::default();
        let optimizer = FunctionOptimizer::new(config);

        let stats = optimizer.get_optimization_stats();
        assert_eq!(stats.total_optimizations, 0);
        assert_eq!(stats.registered_functions, 0);
        assert!(stats.active_patterns > 0);
    }
}
