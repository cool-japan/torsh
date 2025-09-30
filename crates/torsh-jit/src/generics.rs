//! Generic function support for JIT compilation
//!
//! This module provides generic function capabilities, allowing the JIT compiler
//! to create and instantiate parameterized functions with type parameters.

use crate::ir::{IrModule, TypeKind};
use crate::{JitError, JitResult};
use indexmap::IndexMap;
use std::collections::HashMap;

/// Generic function manager
#[derive(Debug, Clone)]
pub struct GenericFunctionManager {
    /// Registry of generic function templates
    templates: IndexMap<String, GenericFunctionTemplate>,

    /// Instantiated generic functions
    instances: IndexMap<InstantiationKey, InstantiatedFunction>,

    /// Statistics about generic function usage
    stats: GenericFunctionStats,

    /// Configuration for generic functions
    config: GenericFunctionConfig,
}

/// Template for a generic function
#[derive(Debug, Clone)]
pub struct GenericFunctionTemplate {
    /// Function name
    pub name: String,

    /// Generic type parameters
    pub type_params: Vec<TypeParameter>,

    /// Function constraints (bounds on type parameters)
    pub constraints: Vec<TypeConstraint>,

    /// Template IR module (with placeholder types)
    pub template_ir: IrModule,

    /// Default implementations for specific type combinations
    pub default_impls: HashMap<Vec<TypeKind>, IrModule>,

    /// Metadata about the template
    pub metadata: TemplateMetadata,
}

/// Type parameter in a generic function
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TypeParameter {
    /// Parameter name (e.g., "T", "U")
    pub name: String,

    /// Parameter kind (type, shape, constant)
    pub kind: ParameterKind,

    /// Default type (if any)
    pub default: Option<TypeKind>,

    /// Variance (covariant, contravariant, invariant)
    pub variance: Variance,
}

/// Kind of type parameter
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ParameterKind {
    /// Type parameter (e.g., T: Float)
    Type,

    /// Shape parameter (e.g., N: usize)
    Shape,

    /// Constant parameter (e.g., SIZE: usize)
    Constant,

    /// Layout parameter (e.g., L: Layout)
    Layout,
}

/// Variance of type parameters
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Variance {
    /// Covariant (T is subtype of U => F<T> is subtype of F<U>)
    Covariant,

    /// Contravariant (T is subtype of U => F<U> is subtype of F<T>)
    Contravariant,

    /// Invariant (no subtyping relationship)
    Invariant,
}

/// Constraint on type parameters
#[derive(Debug, Clone)]
pub enum TypeConstraint {
    /// Type must implement a trait (e.g., T: Float)
    Trait { param: String, trait_name: String },

    /// Type must be equal to another type (e.g., T = U)
    Equality { param1: String, param2: String },

    /// Type must be a subtype of another type (e.g., T <: U)
    Subtype { subtype: String, supertype: String },

    /// Shape constraint (e.g., N > 0)
    Shape {
        param: String,
        constraint: ShapeConstraint,
    },

    /// Custom constraint with a checker function
    Custom {
        name: String,
        checker: fn(&[TypeKind]) -> bool,
    },
}

/// Shape constraints for shape parameters
#[derive(Debug, Clone)]
pub enum ShapeConstraint {
    /// Must be positive
    Positive,

    /// Must be a power of two
    PowerOfTwo,

    /// Must be equal to another parameter
    Equal(String),

    /// Must be less than a value
    LessThan(usize),

    /// Must be greater than a value
    GreaterThan(usize),

    /// Must be in a range
    InRange(usize, usize),
}

/// Key for function instantiation
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct InstantiationKey {
    /// Template name
    pub template_name: String,

    /// Concrete types for type parameters
    pub type_args: Vec<TypeKind>,

    /// Shape arguments for shape parameters
    pub shape_args: Vec<Vec<usize>>,

    /// Constant arguments
    pub const_args: Vec<ConstantValue>,
}

/// Constant values for generic parameters
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ConstantValue {
    Int(i64),
    UInt(u64),
    Float(u64), // Stored as bits for hashing
    Bool(bool),
    String(String),
}

/// Instantiated function
#[derive(Debug, Clone)]
pub struct InstantiatedFunction {
    /// Instantiation key
    pub key: InstantiationKey,

    /// Concrete IR module
    pub module: IrModule,

    /// Performance characteristics
    pub perf_info: GenericFunctionPerfInfo,

    /// Usage count
    pub usage_count: usize,

    /// Instantiation time
    pub instantiation_time_ns: u64,
}

/// Performance information for instantiated functions
#[derive(Debug, Clone, Default)]
pub struct GenericFunctionPerfInfo {
    /// Estimated execution time
    pub estimated_exec_time_ns: u64,

    /// Code size in bytes
    pub code_size: usize,

    /// Register usage
    pub register_usage: u32,

    /// Memory usage
    pub memory_usage: usize,

    /// Optimization level applied
    pub optimization_level: OptimizationLevel,
}

/// Optimization levels for generic functions
#[derive(Debug, Clone, Default)]
pub enum OptimizationLevel {
    /// No optimization
    None,

    /// Basic optimizations
    #[default]
    Basic,

    /// Aggressive optimizations
    Aggressive,

    /// Size-optimized
    Size,

    /// Speed-optimized
    Speed,
}

/// Template metadata
#[derive(Debug, Clone, Default)]
pub struct TemplateMetadata {
    /// Author of the template
    pub author: Option<String>,

    /// Documentation
    pub documentation: Option<String>,

    /// Version
    pub version: Option<String>,

    /// Tags for categorization
    pub tags: Vec<String>,

    /// Complexity estimate
    pub complexity: ComplexityEstimate,
}

/// Complexity estimate for a template
#[derive(Debug, Clone, Default)]
pub struct ComplexityEstimate {
    /// Time complexity (e.g., O(n), O(n^2))
    pub time_complexity: String,

    /// Space complexity
    pub space_complexity: String,

    /// Instantiation complexity
    pub instantiation_complexity: String,
}

/// Configuration for generic functions
#[derive(Debug, Clone)]
pub struct GenericFunctionConfig {
    /// Maximum number of instantiations per template
    pub max_instantiations_per_template: usize,

    /// Enable automatic monomorphization
    pub enable_monomorphization: bool,

    /// Enable type inference for generic functions
    pub enable_type_inference: bool,

    /// Cache instantiated functions
    pub enable_caching: bool,

    /// Maximum recursion depth for generic instantiation
    pub max_recursion_depth: usize,

    /// Enable generic function profiling
    pub enable_profiling: bool,
}

/// Statistics about generic function usage
#[derive(Debug, Clone, Default)]
pub struct GenericFunctionStats {
    /// Total number of templates
    pub total_templates: usize,

    /// Total number of instantiations
    pub total_instantiations: usize,

    /// Cache hits
    pub cache_hits: usize,

    /// Cache misses
    pub cache_misses: usize,

    /// Average instantiation time
    pub avg_instantiation_time_ns: u64,

    /// Most used templates
    pub most_used_templates: Vec<(String, usize)>,
}

impl Default for GenericFunctionConfig {
    fn default() -> Self {
        Self {
            max_instantiations_per_template: 32,
            enable_monomorphization: true,
            enable_type_inference: true,
            enable_caching: true,
            max_recursion_depth: 16,
            enable_profiling: false,
        }
    }
}

impl GenericFunctionManager {
    /// Create a new generic function manager
    pub fn new(config: GenericFunctionConfig) -> Self {
        Self {
            templates: IndexMap::new(),
            instances: IndexMap::new(),
            stats: GenericFunctionStats::default(),
            config,
        }
    }

    /// Create a new generic function manager with default configuration
    pub fn with_defaults() -> Self {
        Self::new(GenericFunctionConfig::default())
    }

    /// Register a generic function template
    pub fn register_template(&mut self, template: GenericFunctionTemplate) -> JitResult<()> {
        // Validate template
        self.validate_template(&template)?;

        // Store template
        self.templates.insert(template.name.clone(), template);
        self.stats.total_templates += 1;

        Ok(())
    }

    /// Instantiate a generic function with concrete types
    pub fn instantiate(
        &mut self,
        template_name: &str,
        type_args: &[TypeKind],
        shape_args: &[Vec<usize>],
        const_args: &[ConstantValue],
    ) -> JitResult<InstantiatedFunction> {
        let key = InstantiationKey {
            template_name: template_name.to_string(),
            type_args: type_args.to_vec(),
            shape_args: shape_args.to_vec(),
            const_args: const_args.to_vec(),
        };

        // Check if instantiation already exists
        if let Some(instance) = self.instances.get_mut(&key) {
            instance.usage_count += 1;
            self.stats.cache_hits += 1;
            return Ok(instance.clone());
        }

        self.stats.cache_misses += 1;

        // Get template
        let template = self.templates.get(template_name).ok_or_else(|| {
            JitError::CompilationError(format!("Template '{}' not found", template_name))
        })?;

        // Check constraints
        self.check_constraints(template, type_args, shape_args, const_args)?;

        // Perform instantiation
        let start_time = std::time::Instant::now();
        let instantiated_module = self.perform_instantiation(template, &key)?;
        let instantiation_time = start_time.elapsed().as_nanos() as u64;

        // Create instantiated function
        let perf_info = self.estimate_performance(&instantiated_module)?;
        let instance = InstantiatedFunction {
            key: key.clone(),
            module: instantiated_module,
            perf_info,
            usage_count: 1,
            instantiation_time_ns: instantiation_time,
        };

        // Store instance
        self.instances.insert(key, instance.clone());
        self.stats.total_instantiations += 1;
        self.stats.avg_instantiation_time_ns = (self.stats.avg_instantiation_time_ns
            * (self.stats.total_instantiations - 1) as u64
            + instantiation_time)
            / self.stats.total_instantiations as u64;

        Ok(instance)
    }

    /// Validate a template before registration
    fn validate_template(&self, template: &GenericFunctionTemplate) -> JitResult<()> {
        // Check that type parameter names are unique
        let mut param_names = std::collections::HashSet::new();
        for param in &template.type_params {
            if !param_names.insert(&param.name) {
                return Err(JitError::CompilationError(format!(
                    "Duplicate type parameter name: {}",
                    param.name
                )));
            }
        }

        // Validate constraints reference existing parameters
        for constraint in &template.constraints {
            match constraint {
                TypeConstraint::Trait { param, .. } => {
                    if !param_names.contains(param) {
                        return Err(JitError::CompilationError(format!(
                            "Constraint references unknown parameter: {}",
                            param
                        )));
                    }
                }
                TypeConstraint::Equality { param1, param2 } => {
                    if !param_names.contains(param1) || !param_names.contains(param2) {
                        return Err(JitError::CompilationError(
                            "Equality constraint references unknown parameter".to_string(),
                        ));
                    }
                }
                TypeConstraint::Subtype { subtype, supertype } => {
                    if !param_names.contains(subtype) || !param_names.contains(supertype) {
                        return Err(JitError::CompilationError(
                            "Subtype constraint references unknown parameter".to_string(),
                        ));
                    }
                }
                TypeConstraint::Shape { param, .. } => {
                    if !param_names.contains(param) {
                        return Err(JitError::CompilationError(format!(
                            "Shape constraint references unknown parameter: {}",
                            param
                        )));
                    }
                }
                TypeConstraint::Custom { .. } => {
                    // Custom constraints are always valid (checked at runtime)
                }
            }
        }

        Ok(())
    }

    /// Check constraints for a specific instantiation
    fn check_constraints(
        &self,
        template: &GenericFunctionTemplate,
        type_args: &[TypeKind],
        shape_args: &[Vec<usize>],
        _const_args: &[ConstantValue],
    ) -> JitResult<()> {
        if type_args.len() != template.type_params.len() {
            return Err(JitError::CompilationError(
                "Wrong number of type arguments".to_string(),
            ));
        }

        for constraint in &template.constraints {
            match constraint {
                TypeConstraint::Trait { param, trait_name } => {
                    let param_index = template
                        .type_params
                        .iter()
                        .position(|p| p.name == *param)
                        .unwrap();

                    if !self.check_trait_constraint(&type_args[param_index], trait_name) {
                        return Err(JitError::CompilationError(format!(
                            "Type {:?} does not implement trait {}",
                            type_args[param_index], trait_name
                        )));
                    }
                }
                TypeConstraint::Shape {
                    param,
                    constraint: shape_constraint,
                } => {
                    let param_index = template
                        .type_params
                        .iter()
                        .position(|p| p.name == *param)
                        .unwrap();

                    if param_index < shape_args.len() {
                        if !self.check_shape_constraint(&shape_args[param_index], shape_constraint)
                        {
                            return Err(JitError::CompilationError(format!(
                                "Shape constraint violation for parameter {}",
                                param
                            )));
                        }
                    }
                }
                TypeConstraint::Custom { checker, .. } => {
                    if !checker(type_args) {
                        return Err(JitError::CompilationError(
                            "Custom constraint violation".to_string(),
                        ));
                    }
                }
                _ => {
                    // TODO: Implement other constraint checks
                }
            }
        }

        Ok(())
    }

    /// Check if a type satisfies a trait constraint
    fn check_trait_constraint(&self, type_kind: &TypeKind, trait_name: &str) -> bool {
        match trait_name {
            "Float" => matches!(type_kind, TypeKind::F16 | TypeKind::F32 | TypeKind::F64),
            "Integer" => matches!(
                type_kind,
                TypeKind::I8
                    | TypeKind::I16
                    | TypeKind::I32
                    | TypeKind::I64
                    | TypeKind::U8
                    | TypeKind::U16
                    | TypeKind::U32
                    | TypeKind::U64
            ),
            "Numeric" => matches!(
                type_kind,
                TypeKind::I8
                    | TypeKind::I16
                    | TypeKind::I32
                    | TypeKind::I64
                    | TypeKind::U8
                    | TypeKind::U16
                    | TypeKind::U32
                    | TypeKind::U64
                    | TypeKind::F16
                    | TypeKind::F32
                    | TypeKind::F64
            ),
            "Complex" => matches!(type_kind, TypeKind::C64 | TypeKind::C128),
            _ => false, // Unknown trait
        }
    }

    /// Check if a shape satisfies a shape constraint
    fn check_shape_constraint(&self, shape: &[usize], constraint: &ShapeConstraint) -> bool {
        match constraint {
            ShapeConstraint::Positive => shape.iter().all(|&dim| dim > 0),
            ShapeConstraint::PowerOfTwo => shape.iter().all(|&dim| dim.is_power_of_two()),
            ShapeConstraint::LessThan(limit) => shape.iter().all(|&dim| dim < *limit),
            ShapeConstraint::GreaterThan(limit) => shape.iter().all(|&dim| dim > *limit),
            ShapeConstraint::InRange(min, max) => {
                shape.iter().all(|&dim| dim >= *min && dim <= *max)
            }
            _ => true, // TODO: Implement other constraints
        }
    }

    /// Perform the actual instantiation of a template
    fn perform_instantiation(
        &self,
        template: &GenericFunctionTemplate,
        key: &InstantiationKey,
    ) -> JitResult<IrModule> {
        // Start with the template IR
        let mut instantiated = template.template_ir.clone();
        instantiated.name = format!("{}_{}", template.name, self.generate_mangled_name(key));

        // Replace type parameters with concrete types
        self.substitute_types(&mut instantiated, template, key)?;

        // Apply generic-specific optimizations
        self.optimize_instantiated_function(&mut instantiated)?;

        Ok(instantiated)
    }

    /// Generate a mangled name for the instantiation
    fn generate_mangled_name(&self, key: &InstantiationKey) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }

    /// Substitute type parameters with concrete types
    fn substitute_types(
        &self,
        _module: &mut IrModule,
        template: &GenericFunctionTemplate,
        key: &InstantiationKey,
    ) -> JitResult<()> {
        // Create type substitution map
        let mut type_map = HashMap::new();
        for (param, type_arg) in template.type_params.iter().zip(key.type_args.iter()) {
            type_map.insert(param.name.clone(), type_arg.clone());
        }

        // TODO: Implement type substitution throughout the IR
        // This would involve traversing the IR and replacing type parameters
        // with concrete types according to the type_map

        Ok(())
    }

    /// Apply optimizations specific to instantiated generic functions
    fn optimize_instantiated_function(&self, _module: &mut IrModule) -> JitResult<()> {
        // TODO: Implement generic function optimizations
        // - Dead code elimination for unused branches
        // - Constant folding with known type information
        // - Inlining small generic functions
        // - Type-specific optimizations

        Ok(())
    }

    /// Estimate performance characteristics of an instantiated function
    fn estimate_performance(&self, module: &IrModule) -> JitResult<GenericFunctionPerfInfo> {
        let mut perf_info = GenericFunctionPerfInfo::default();

        // Simple heuristics for performance estimation
        let mut instruction_count = 0;
        for (_, block) in &module.blocks {
            instruction_count += block.instructions.len();
        }

        perf_info.estimated_exec_time_ns = instruction_count as u64 * 10; // ~10ns per instruction
        perf_info.code_size = instruction_count * 4; // ~4 bytes per instruction
        perf_info.register_usage = (instruction_count / 4).min(32) as u32; // Estimate register pressure
        perf_info.memory_usage = instruction_count * 8; // Estimate memory usage

        Ok(perf_info)
    }

    /// Get statistics about generic function usage
    pub fn stats(&self) -> &GenericFunctionStats {
        &self.stats
    }

    /// List all registered templates
    pub fn list_templates(&self) -> Vec<&str> {
        self.templates.keys().map(|s| s.as_str()).collect()
    }

    /// Get template by name
    pub fn get_template(&self, name: &str) -> Option<&GenericFunctionTemplate> {
        self.templates.get(name)
    }

    /// Clear all instantiations (for memory management)
    pub fn clear_instances(&mut self) {
        self.instances.clear();
        self.stats.total_instantiations = 0;
        self.stats.cache_hits = 0;
        self.stats.cache_misses = 0;
    }

    /// Get the number of instantiations for a template
    pub fn instantiation_count(&self, template_name: &str) -> usize {
        self.instances
            .keys()
            .filter(|k| k.template_name == template_name)
            .count()
    }
}

/// Helper function to create a simple type parameter
pub fn create_type_param(name: &str, kind: ParameterKind) -> TypeParameter {
    TypeParameter {
        name: name.to_string(),
        kind,
        default: None,
        variance: Variance::Invariant,
    }
}

/// Helper function to create a trait constraint
pub fn trait_constraint(param: &str, trait_name: &str) -> TypeConstraint {
    TypeConstraint::Trait {
        param: param.to_string(),
        trait_name: trait_name.to_string(),
    }
}

/// Helper function to create a shape constraint
pub fn shape_constraint(param: &str, constraint: ShapeConstraint) -> TypeConstraint {
    TypeConstraint::Shape {
        param: param.to_string(),
        constraint,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generic_function_manager_creation() {
        let manager = GenericFunctionManager::with_defaults();
        assert_eq!(manager.templates.len(), 0);
        assert_eq!(manager.instances.len(), 0);
    }

    #[test]
    fn test_type_parameter_creation() {
        let param = create_type_param("T", ParameterKind::Type);
        assert_eq!(param.name, "T");
        assert_eq!(param.kind, ParameterKind::Type);
        assert_eq!(param.variance, Variance::Invariant);
    }

    #[test]
    fn test_trait_constraint_creation() {
        let constraint = trait_constraint("T", "Float");
        match constraint {
            TypeConstraint::Trait { param, trait_name } => {
                assert_eq!(param, "T");
                assert_eq!(trait_name, "Float");
            }
            _ => panic!("Expected trait constraint"),
        }
    }

    #[test]
    fn test_trait_constraint_checking() {
        let manager = GenericFunctionManager::with_defaults();

        // Float types should satisfy Float trait
        assert!(manager.check_trait_constraint(&TypeKind::F32, "Float"));
        assert!(manager.check_trait_constraint(&TypeKind::F64, "Float"));

        // Integer types should not satisfy Float trait
        assert!(!manager.check_trait_constraint(&TypeKind::I32, "Float"));

        // Integer types should satisfy Integer trait
        assert!(manager.check_trait_constraint(&TypeKind::I32, "Integer"));
        assert!(manager.check_trait_constraint(&TypeKind::U64, "Integer"));
    }

    #[test]
    fn test_shape_constraint_checking() {
        let manager = GenericFunctionManager::with_defaults();

        let positive_shape = vec![2, 4, 8];
        let zero_shape = vec![0, 4, 8];
        let power_of_two_shape = vec![2, 4, 8];
        let non_power_of_two_shape = vec![3, 5, 7];

        assert!(manager.check_shape_constraint(&positive_shape, &ShapeConstraint::Positive));
        assert!(!manager.check_shape_constraint(&zero_shape, &ShapeConstraint::Positive));

        assert!(manager.check_shape_constraint(&power_of_two_shape, &ShapeConstraint::PowerOfTwo));
        assert!(
            !manager.check_shape_constraint(&non_power_of_two_shape, &ShapeConstraint::PowerOfTwo)
        );
    }
}
