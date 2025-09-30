//! Type specialization for JIT compilation
//!
//! This module provides type specialization capabilities, allowing the JIT compiler
//! to create optimized versions of functions and kernels for specific types and shapes.

use crate::ir::{IrModule, IrOpcode, TypeKind};
use crate::{JitError, JitResult};
use indexmap::IndexMap;
use torsh_core::{DType, Shape};

/// Type specialization engine
#[derive(Debug, Clone)]
pub struct TypeSpecializer {
    /// Registry of specialized functions
    specializations: IndexMap<SpecializationKey, SpecializedFunction>,

    /// Specialization statistics
    stats: SpecializationStats,

    /// Configuration for specialization
    config: SpecializationConfig,
}

/// Key identifying a specialization
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SpecializationKey {
    /// Original function name
    pub function_name: String,

    /// Specialized parameter types
    pub param_types: Vec<SpecializedType>,

    /// Return type specialization  
    pub return_type: Option<SpecializedType>,
}

/// Specialized type information
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SpecializedType {
    /// Base type
    pub base_type: TypeKind,

    /// Shape specialization (for tensors)
    pub shape: Option<Vec<usize>>,

    /// Constant value (for constant propagation)
    pub constant_value: Option<ConstantValue>,

    /// Memory layout hints
    pub layout_hints: LayoutHints,
}

/// Constant values for specialization
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ConstantValue {
    Int(i64),
    Float(u64), // Stored as bits for hashing
    Bool(bool),
    Shape(Vec<usize>),
}

/// Memory layout optimization hints
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct LayoutHints {
    /// Preferred memory alignment
    pub alignment: Option<usize>,

    /// Whether data is contiguous
    pub contiguous: bool,

    /// Preferred data layout (e.g., row-major, column-major)
    pub layout: Option<DataLayout>,

    /// Cache locality hints
    pub locality: LocalityHint,
}

/// Data layout preferences
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DataLayout {
    RowMajor,
    ColumnMajor,
    Packed,
    Strided { strides: Vec<usize> },
}

/// Cache locality hints
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub enum LocalityHint {
    #[default]
    None,
    Temporal,    // Will be reused soon
    NonTemporal, // Won't be reused
    Streaming,   // Sequential access pattern
}

/// Specialized function implementation
#[derive(Debug, Clone)]
pub struct SpecializedFunction {
    /// Specialization key
    pub key: SpecializationKey,

    /// Specialized IR module
    pub module: IrModule,

    /// Performance characteristics
    pub perf_info: PerformanceInfo,

    /// Usage statistics
    pub usage_count: usize,

    /// Compilation time
    pub compile_time_ns: u64,
}

/// Performance information for specialized functions
#[derive(Debug, Clone, Default)]
pub struct PerformanceInfo {
    /// Estimated execution time in nanoseconds
    pub estimated_exec_time_ns: u64,

    /// Memory bandwidth requirements (bytes/second)
    pub memory_bandwidth: u64,

    /// Arithmetic intensity (ops/byte)
    pub arithmetic_intensity: f64,

    /// Register pressure score (0-100)
    pub register_pressure: u8,

    /// Vectorization factor
    pub vectorization_factor: usize,
}

/// Configuration for type specialization
#[derive(Debug, Clone)]
pub struct SpecializationConfig {
    /// Maximum number of specializations per function
    pub max_specializations_per_function: usize,

    /// Minimum usage count before creating specialization
    pub min_usage_threshold: usize,

    /// Enable shape-based specialization
    pub enable_shape_specialization: bool,

    /// Enable constant propagation specialization
    pub enable_constant_specialization: bool,

    /// Enable layout optimization specialization
    pub enable_layout_specialization: bool,

    /// Performance improvement threshold (speedup ratio)
    pub min_performance_improvement: f64,

    /// Code size increase limit (ratio)
    pub max_code_size_increase: f64,
}

/// Specialization statistics
#[derive(Debug, Clone, Default)]
pub struct SpecializationStats {
    /// Total specializations created
    pub total_specializations: usize,

    /// Cache hits
    pub cache_hits: usize,

    /// Cache misses
    pub cache_misses: usize,

    /// Average compilation time
    pub avg_compilation_time_ns: u64,

    /// Total performance improvement
    pub total_speedup: f64,

    /// Code size overhead
    pub code_size_overhead: f64,
}

impl Default for SpecializationConfig {
    fn default() -> Self {
        Self {
            max_specializations_per_function: 16,
            min_usage_threshold: 3,
            enable_shape_specialization: true,
            enable_constant_specialization: true,
            enable_layout_specialization: true,
            min_performance_improvement: 1.2, // 20% improvement minimum
            max_code_size_increase: 2.0,      // 2x size increase maximum
        }
    }
}

impl TypeSpecializer {
    /// Create a new type specializer
    pub fn new(config: SpecializationConfig) -> Self {
        Self {
            specializations: IndexMap::new(),
            stats: SpecializationStats::default(),
            config,
        }
    }

    /// Create a new type specializer with default configuration
    pub fn with_defaults() -> Self {
        Self::new(SpecializationConfig::default())
    }

    /// Get or create a specialized version of a function
    pub fn specialize_function(
        &mut self,
        function_name: &str,
        param_types: &[SpecializedType],
        return_type: Option<SpecializedType>,
        original_module: &IrModule,
    ) -> JitResult<SpecializedFunction> {
        let key = SpecializationKey {
            function_name: function_name.to_string(),
            param_types: param_types.to_vec(),
            return_type,
        };

        // Check if specialization already exists
        if let Some(specialized) = self.specializations.get_mut(&key) {
            specialized.usage_count += 1;
            self.stats.cache_hits += 1;
            return Ok(specialized.clone());
        }

        self.stats.cache_misses += 1;

        // Check if we should create a new specialization (separate scope to avoid borrow issues)
        let should_specialize = {
            // Count existing specializations for this function
            let existing_count = self
                .specializations
                .keys()
                .filter(|k| k.function_name == key.function_name)
                .count();

            if existing_count >= self.config.max_specializations_per_function {
                false
            } else {
                self.is_specialization_beneficial(&key)
            }
        };

        if !should_specialize {
            return Err(JitError::OptimizationError(
                "Specialization not beneficial".to_string(),
            ));
        }

        // Create the specialized function
        let start_time = std::time::Instant::now();
        let specialized_module = self.create_specialized_module(original_module, &key)?;
        let compile_time = start_time.elapsed().as_nanos() as u64;

        let perf_info = self.estimate_performance(&specialized_module)?;

        let specialized_fn = SpecializedFunction {
            key: key.clone(),
            module: specialized_module,
            perf_info,
            usage_count: 1,
            compile_time_ns: compile_time,
        };

        self.specializations.insert(key, specialized_fn.clone());
        self.stats.total_specializations += 1;
        self.stats.avg_compilation_time_ns = (self.stats.avg_compilation_time_ns
            * (self.stats.total_specializations - 1) as u64
            + compile_time)
            / self.stats.total_specializations as u64;

        Ok(specialized_fn)
    }

    /// Determine if a specialization would be beneficial
    fn is_specialization_beneficial(&self, key: &SpecializationKey) -> bool {
        // Always specialize for constant values
        if self.config.enable_constant_specialization {
            for param_type in &key.param_types {
                if param_type.constant_value.is_some() {
                    return true;
                }
            }
        }

        // Check for beneficial shape specializations
        if self.config.enable_shape_specialization {
            for param_type in &key.param_types {
                if let Some(shape) = &param_type.shape {
                    // Small, fixed shapes are good candidates
                    if shape.iter().product::<usize>() < 1024 {
                        return true;
                    }
                    // Power-of-2 shapes often vectorize well
                    if shape.iter().all(|&dim| dim.is_power_of_two()) {
                        return true;
                    }
                }
            }
        }

        // Check for layout optimizations
        if self.config.enable_layout_specialization {
            for param_type in &key.param_types {
                if param_type.layout_hints.contiguous || param_type.layout_hints.layout.is_some() {
                    return true;
                }
            }
        }

        false
    }

    /// Create a specialized version of the IR module
    fn create_specialized_module(
        &self,
        original: &IrModule,
        key: &SpecializationKey,
    ) -> JitResult<IrModule> {
        let mut specialized = original.clone();
        specialized.name = format!(
            "{}_{}",
            original.name,
            self.generate_specialization_suffix(key)
        );

        // Apply type-specific optimizations
        self.apply_type_optimizations(&mut specialized, key)?;

        // Apply shape-specific optimizations
        self.apply_shape_optimizations(&mut specialized, key)?;

        // Apply constant propagation
        self.apply_constant_propagation(&mut specialized, key)?;

        // Apply layout optimizations
        self.apply_layout_optimizations(&mut specialized, key)?;

        Ok(specialized)
    }

    /// Generate a unique suffix for the specialization
    fn generate_specialization_suffix(&self, key: &SpecializationKey) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }

    /// Apply type-specific optimizations
    fn apply_type_optimizations(
        &self,
        module: &mut IrModule,
        key: &SpecializationKey,
    ) -> JitResult<()> {
        // Replace generic operations with type-specific ones
        for (_, block) in module.blocks.iter_mut() {
            for instruction in &mut block.instructions {
                match instruction.opcode {
                    IrOpcode::Add | IrOpcode::Sub | IrOpcode::Mul | IrOpcode::Div => {
                        // Could specialize to SIMD instructions for specific types
                        if let Some(param_type) = key.param_types.first() {
                            match param_type.base_type {
                                TypeKind::F32 => {
                                    // Could use vectorized f32 operations
                                }
                                TypeKind::F64 => {
                                    // Could use vectorized f64 operations
                                }
                                TypeKind::I32 => {
                                    // Could use integer-specific optimizations
                                }
                                _ => {}
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        Ok(())
    }

    /// Apply shape-specific optimizations
    fn apply_shape_optimizations(
        &self,
        module: &mut IrModule,
        key: &SpecializationKey,
    ) -> JitResult<()> {
        for param_type in &key.param_types {
            if let Some(shape) = &param_type.shape {
                // Unroll loops for small, known shapes
                if shape.iter().product::<usize>() < 64 {
                    self.unroll_small_loops(module, shape)?;
                }

                // Optimize memory access patterns for specific shapes
                self.optimize_memory_access(module, shape)?;
            }
        }

        Ok(())
    }

    /// Apply constant propagation optimizations
    fn apply_constant_propagation(
        &self,
        module: &mut IrModule,
        key: &SpecializationKey,
    ) -> JitResult<()> {
        for param_type in &key.param_types {
            if let Some(const_val) = &param_type.constant_value {
                // Replace parameter with constant throughout the module
                self.propagate_constant(module, const_val)?;
            }
        }

        Ok(())
    }

    /// Apply layout-specific optimizations
    fn apply_layout_optimizations(
        &self,
        module: &mut IrModule,
        key: &SpecializationKey,
    ) -> JitResult<()> {
        for param_type in &key.param_types {
            match &param_type.layout_hints.layout {
                Some(DataLayout::RowMajor) => {
                    // Optimize for row-major access patterns
                    self.optimize_for_row_major(module)?;
                }
                Some(DataLayout::ColumnMajor) => {
                    // Optimize for column-major access patterns
                    self.optimize_for_column_major(module)?;
                }
                Some(DataLayout::Packed) => {
                    // Optimize for packed data
                    self.optimize_for_packed_data(module)?;
                }
                _ => {}
            }
        }

        Ok(())
    }

    /// Unroll loops for small, known iteration counts
    fn unroll_small_loops(&self, _module: &mut IrModule, _shape: &[usize]) -> JitResult<()> {
        // TODO: Implement loop unrolling logic
        Ok(())
    }

    /// Optimize memory access patterns
    fn optimize_memory_access(&self, _module: &mut IrModule, _shape: &[usize]) -> JitResult<()> {
        // TODO: Implement memory access optimization
        Ok(())
    }

    /// Propagate constant values throughout the module
    fn propagate_constant(
        &self,
        _module: &mut IrModule,
        _const_val: &ConstantValue,
    ) -> JitResult<()> {
        // TODO: Implement constant propagation
        Ok(())
    }

    /// Optimize for row-major memory layout
    fn optimize_for_row_major(&self, _module: &mut IrModule) -> JitResult<()> {
        // TODO: Implement row-major optimizations
        Ok(())
    }

    /// Optimize for column-major memory layout
    fn optimize_for_column_major(&self, _module: &mut IrModule) -> JitResult<()> {
        // TODO: Implement column-major optimizations
        Ok(())
    }

    /// Optimize for packed data layout
    fn optimize_for_packed_data(&self, _module: &mut IrModule) -> JitResult<()> {
        // TODO: Implement packed data optimizations
        Ok(())
    }

    /// Estimate performance characteristics of a specialized function
    fn estimate_performance(&self, module: &IrModule) -> JitResult<PerformanceInfo> {
        let mut perf_info = PerformanceInfo::default();

        // Count operations and estimate execution time
        let mut op_count = 0;
        let mut memory_ops = 0;

        for (_, block) in &module.blocks {
            for instruction in &block.instructions {
                op_count += 1;
                match instruction.opcode {
                    IrOpcode::Load | IrOpcode::Store => memory_ops += 1,
                    _ => {}
                }
            }
        }

        // Simple heuristic estimates
        perf_info.estimated_exec_time_ns = op_count * 10; // ~10ns per operation
        perf_info.memory_bandwidth = memory_ops * 64; // ~64 bytes per memory op
        perf_info.arithmetic_intensity = if memory_ops > 0 {
            (op_count - memory_ops) as f64 / memory_ops as f64
        } else {
            f64::INFINITY
        };

        Ok(perf_info)
    }

    /// Get specialization statistics
    pub fn stats(&self) -> &SpecializationStats {
        &self.stats
    }

    /// Clear all specializations (for memory management)
    pub fn clear_cache(&mut self) {
        self.specializations.clear();
        self.stats = SpecializationStats::default();
    }

    /// Get the number of specializations for a function
    pub fn specialization_count(&self, function_name: &str) -> usize {
        self.specializations
            .keys()
            .filter(|k| k.function_name == function_name)
            .count()
    }

    /// List all specialized functions
    pub fn list_specializations(&self) -> Vec<&SpecializationKey> {
        self.specializations.keys().collect()
    }
}

/// Helper function to create specialized type from DType and Shape
pub fn create_specialized_type(dtype: DType, shape: Option<Shape>) -> SpecializedType {
    let base_type = match dtype {
        DType::F16 => TypeKind::F16,
        DType::F32 => TypeKind::F32,
        DType::F64 => TypeKind::F64,
        DType::I8 => TypeKind::I8,
        DType::I16 => TypeKind::I16,
        DType::I32 => TypeKind::I32,
        DType::I64 => TypeKind::I64,
        DType::U8 => TypeKind::U8,
        DType::U32 => TypeKind::U32,
        DType::U64 => TypeKind::U64,
        DType::Bool => TypeKind::Bool,
        DType::BF16 => TypeKind::F16, // Map BF16 to F16 for now
        DType::C64 => TypeKind::C64,
        DType::C128 => TypeKind::C128,
        DType::QInt8 | DType::QUInt8 => TypeKind::I8, // Map quantized types to base types
    };

    let shape_vec = shape.map(|s| s.dims().to_vec());

    SpecializedType {
        base_type,
        shape: shape_vec,
        constant_value: None,
        layout_hints: LayoutHints::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_specialization_key_equality() {
        let key1 = SpecializationKey {
            function_name: "test_fn".to_string(),
            param_types: vec![SpecializedType {
                base_type: TypeKind::F32,
                shape: Some(vec![2, 2]),
                constant_value: None,
                layout_hints: LayoutHints::default(),
            }],
            return_type: None,
        };

        let key2 = key1.clone();
        assert_eq!(key1, key2);
    }

    #[test]
    fn test_specializer_creation() {
        let specializer = TypeSpecializer::with_defaults();
        assert_eq!(specializer.specializations.len(), 0);
        assert_eq!(specializer.stats.total_specializations, 0);
    }

    #[test]
    fn test_create_specialized_type() {
        let dtype = DType::F32;
        let shape = Some(Shape::new(vec![2, 3]));

        let spec_type = create_specialized_type(dtype, shape);
        assert_eq!(spec_type.base_type, TypeKind::F32);
        assert_eq!(spec_type.shape, Some(vec![2, 3]));
    }
}
