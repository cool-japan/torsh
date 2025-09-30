//! Metaprogramming support for dynamic code generation and reflection
//!
//! This module provides comprehensive metaprogramming capabilities including:
//! - Dynamic code generation from templates
//! - Runtime reflection and introspection
//! - Compile-time code transformation
//! - Template-based code specialization

use crate::{ir::IrModule, ComputationGraph, JitError, JitResult, NodeId};
use std::collections::HashMap;
use std::fmt::Write;
use torsh_core::{DType, Shape};

/// Metaprogramming engine for dynamic code generation
pub struct MetaprogrammingEngine {
    templates: HashMap<String, CodeTemplate>,
    macros: HashMap<String, MacroDefinition>,
    reflector: RuntimeReflector,
    code_generator: DynamicCodeGenerator,
}

impl MetaprogrammingEngine {
    /// Create a new metaprogramming engine
    pub fn new() -> Self {
        Self {
            templates: HashMap::new(),
            macros: HashMap::new(),
            reflector: RuntimeReflector::new(),
            code_generator: DynamicCodeGenerator::new(),
        }
    }

    /// Register a code template
    pub fn register_template(&mut self, name: String, template: CodeTemplate) {
        self.templates.insert(name, template);
    }

    /// Register a macro definition
    pub fn register_macro(&mut self, name: String, macro_def: MacroDefinition) {
        self.macros.insert(name, macro_def);
    }

    /// Generate code from a template
    pub fn generate_from_template(
        &self,
        template_name: &str,
        parameters: &TemplateParameters,
    ) -> JitResult<GeneratedCode> {
        let template = self.templates.get(template_name).ok_or_else(|| {
            JitError::CompilationError(format!("Template '{}' not found", template_name))
        })?;

        template.instantiate(parameters, &self.code_generator)
    }

    /// Expand a macro with given arguments
    pub fn expand_macro(
        &self,
        macro_name: &str,
        args: &[MacroArgument],
    ) -> JitResult<GeneratedCode> {
        let macro_def = self.macros.get(macro_name).ok_or_else(|| {
            JitError::CompilationError(format!("Macro '{}' not found", macro_name))
        })?;

        macro_def.expand(args, &self.code_generator)
    }

    /// Reflect on a computation graph
    pub fn reflect_graph(&self, graph: &ComputationGraph) -> GraphReflection {
        self.reflector.reflect_graph(graph)
    }

    /// Generate specialized code based on runtime information
    pub fn generate_specialized_code(
        &self,
        base_template: &str,
        specialization_info: &SpecializationInfo,
    ) -> JitResult<GeneratedCode> {
        let mut params = TemplateParameters::new();

        // Add specialization parameters
        for (key, value) in &specialization_info.type_info {
            params.add_type(key.clone(), value.clone());
        }

        for (key, value) in &specialization_info.shape_info {
            params.add_shape(key.clone(), value.clone());
        }

        for (key, value) in &specialization_info.constants {
            params.add_constant(key.clone(), value.clone());
        }

        self.generate_from_template(base_template, &params)
    }
}

/// Code template for dynamic code generation
#[derive(Debug, Clone)]
pub struct CodeTemplate {
    pub name: String,
    pub template_string: String,
    pub parameters: Vec<TemplateParameter>,
    pub constraints: Vec<TemplateConstraint>,
}

impl CodeTemplate {
    /// Create a new code template
    pub fn new(name: String, template_string: String) -> Self {
        Self {
            name,
            template_string,
            parameters: Vec::new(),
            constraints: Vec::new(),
        }
    }

    /// Add a template parameter
    pub fn add_parameter(&mut self, param: TemplateParameter) {
        self.parameters.push(param);
    }

    /// Add a template constraint
    pub fn add_constraint(&mut self, constraint: TemplateConstraint) {
        self.constraints.push(constraint);
    }

    /// Instantiate the template with given parameters
    pub fn instantiate(
        &self,
        parameters: &TemplateParameters,
        generator: &DynamicCodeGenerator,
    ) -> JitResult<GeneratedCode> {
        // Validate constraints
        self.validate_constraints(parameters)?;

        // Perform template substitution
        let mut code = self.template_string.clone();

        // Replace type parameters
        for (name, dtype) in &parameters.types {
            let replacement = generator.format_type(dtype);
            code = code.replace(&format!("${{{}}}", name), &replacement);
        }

        // Replace shape parameters
        for (name, shape) in &parameters.shapes {
            let replacement = generator.format_shape(shape);
            code = code.replace(&format!("$shape{{{}}}", name), &replacement);
        }

        // Replace constant parameters
        for (name, value) in &parameters.constants {
            let replacement = generator.format_constant(value);
            code = code.replace(&format!("$const{{{}}}", name), &replacement);
        }

        // Replace code block parameters
        for (name, block) in &parameters.code_blocks {
            code = code.replace(&format!("$code{{{}}}", name), block);
        }

        Ok(GeneratedCode {
            source: code,
            metadata: CodeMetadata {
                template_name: self.name.clone(),
                parameters: parameters.clone(),
                generated_at: std::time::SystemTime::now(),
            },
        })
    }

    /// Validate template constraints
    fn validate_constraints(&self, parameters: &TemplateParameters) -> JitResult<()> {
        for constraint in &self.constraints {
            match constraint {
                TemplateConstraint::TypeConstraint {
                    param_name,
                    allowed_types,
                } => {
                    if let Some(dtype) = parameters.types.get(param_name) {
                        if !allowed_types.contains(dtype) {
                            return Err(JitError::CompilationError(format!(
                                "Type constraint violated for parameter '{}'",
                                param_name
                            )));
                        }
                    }
                }
                TemplateConstraint::ShapeConstraint {
                    param_name,
                    dimension_count,
                } => {
                    if let Some(shape) = parameters.shapes.get(param_name) {
                        if shape.ndim() != *dimension_count {
                            return Err(JitError::CompilationError(format!(
                                "Shape constraint violated for parameter '{}'",
                                param_name
                            )));
                        }
                    }
                }
                TemplateConstraint::ValueConstraint {
                    param_name,
                    min_value,
                    max_value,
                } => {
                    if let Some(value) = parameters.constants.get(param_name) {
                        if let ConstantValue::Integer(val) = value {
                            if *val < *min_value || *val > *max_value {
                                return Err(JitError::CompilationError(format!(
                                    "Value constraint violated for parameter '{}'",
                                    param_name
                                )));
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// Template parameter definition
#[derive(Debug, Clone)]
pub struct TemplateParameter {
    pub name: String,
    pub param_type: ParameterType,
    pub default_value: Option<String>,
    pub description: String,
}

/// Types of template parameters
#[derive(Debug, Clone)]
pub enum ParameterType {
    Type,
    Shape,
    Constant,
    CodeBlock,
}

/// Template constraints for validation
#[derive(Debug, Clone)]
pub enum TemplateConstraint {
    TypeConstraint {
        param_name: String,
        allowed_types: Vec<DType>,
    },
    ShapeConstraint {
        param_name: String,
        dimension_count: usize,
    },
    ValueConstraint {
        param_name: String,
        min_value: i64,
        max_value: i64,
    },
}

/// Parameters for template instantiation
#[derive(Debug, Clone)]
pub struct TemplateParameters {
    pub types: HashMap<String, DType>,
    pub shapes: HashMap<String, Shape>,
    pub constants: HashMap<String, ConstantValue>,
    pub code_blocks: HashMap<String, String>,
}

impl TemplateParameters {
    /// Create new empty parameters
    pub fn new() -> Self {
        Self {
            types: HashMap::new(),
            shapes: HashMap::new(),
            constants: HashMap::new(),
            code_blocks: HashMap::new(),
        }
    }

    /// Add a type parameter
    pub fn add_type(&mut self, name: String, dtype: DType) {
        self.types.insert(name, dtype);
    }

    /// Add a shape parameter
    pub fn add_shape(&mut self, name: String, shape: Shape) {
        self.shapes.insert(name, shape);
    }

    /// Add a constant parameter
    pub fn add_constant(&mut self, name: String, value: ConstantValue) {
        self.constants.insert(name, value);
    }

    /// Add a code block parameter
    pub fn add_code_block(&mut self, name: String, code: String) {
        self.code_blocks.insert(name, code);
    }
}

/// Constant values for templates
#[derive(Debug, Clone)]
pub enum ConstantValue {
    Integer(i64),
    Float(f64),
    Boolean(bool),
    String(String),
}

/// Macro definition for code expansion
#[derive(Debug, Clone)]
pub struct MacroDefinition {
    pub name: String,
    pub parameters: Vec<String>,
    pub body: String,
    pub expansion_rules: Vec<ExpansionRule>,
}

impl MacroDefinition {
    /// Create a new macro definition
    pub fn new(name: String, parameters: Vec<String>, body: String) -> Self {
        Self {
            name,
            parameters,
            body,
            expansion_rules: Vec::new(),
        }
    }

    /// Add an expansion rule
    pub fn add_rule(&mut self, rule: ExpansionRule) {
        self.expansion_rules.push(rule);
    }

    /// Expand the macro with given arguments
    pub fn expand(
        &self,
        args: &[MacroArgument],
        generator: &DynamicCodeGenerator,
    ) -> JitResult<GeneratedCode> {
        if args.len() != self.parameters.len() {
            return Err(JitError::CompilationError(format!(
                "Macro '{}' expects {} arguments, got {}",
                self.name,
                self.parameters.len(),
                args.len()
            )));
        }

        let mut expanded = self.body.clone();

        // Replace parameters with arguments
        for (param, arg) in self.parameters.iter().zip(args.iter()) {
            let replacement = match arg {
                MacroArgument::Code(code) => code.clone(),
                MacroArgument::Literal(lit) => lit.clone(),
                MacroArgument::Expression(expr) => generator.format_expression(expr),
            };
            expanded = expanded.replace(&format!("${}", param), &replacement);
        }

        // Apply expansion rules
        for rule in &self.expansion_rules {
            expanded = rule.apply(&expanded);
        }

        Ok(GeneratedCode {
            source: expanded,
            metadata: CodeMetadata {
                template_name: self.name.clone(),
                parameters: TemplateParameters::new(), // Macros don't use template parameters
                generated_at: std::time::SystemTime::now(),
            },
        })
    }
}

/// Macro arguments
#[derive(Debug, Clone)]
pub enum MacroArgument {
    Code(String),
    Literal(String),
    Expression(String),
}

/// Rules for macro expansion
#[derive(Debug, Clone)]
pub struct ExpansionRule {
    pub pattern: String,
    pub replacement: String,
}

impl ExpansionRule {
    /// Apply the expansion rule to code
    pub fn apply(&self, code: &str) -> String {
        code.replace(&self.pattern, &self.replacement)
    }
}

/// Runtime reflection capabilities
pub struct RuntimeReflector {
    type_registry: HashMap<String, TypeInfo>,
    operation_registry: HashMap<String, OperationInfo>,
}

impl RuntimeReflector {
    /// Create a new runtime reflector
    pub fn new() -> Self {
        Self {
            type_registry: HashMap::new(),
            operation_registry: HashMap::new(),
        }
    }

    /// Register type information
    pub fn register_type(&mut self, name: String, info: TypeInfo) {
        self.type_registry.insert(name, info);
    }

    /// Register operation information
    pub fn register_operation(&mut self, name: String, info: OperationInfo) {
        self.operation_registry.insert(name, info);
    }

    /// Reflect on a computation graph
    pub fn reflect_graph(&self, graph: &ComputationGraph) -> GraphReflection {
        let mut node_info = HashMap::new();
        let mut edge_info = Vec::new();
        let mut type_analysis = HashMap::new();

        // Analyze nodes
        for (node_id, node) in graph.nodes() {
            let reflection = NodeReflection {
                id: node_id,
                operation: node.operation_type().to_string(),
                input_types: Vec::new(), // TODO: Derive from edges
                output_type: node.dtype,
                output_shape: node.output_shape.clone(),
                metadata: self.get_operation_metadata(&node.operation_type()),
            };
            node_info.insert(node_id, reflection);

            // Analyze types
            type_analysis.insert(
                node_id,
                TypeAnalysis {
                    declared_type: node.dtype,
                    inferred_type: node.dtype, // TODO: Add proper type inference
                    type_constraints: Vec::new(),
                },
            );
        }

        // Analyze edges
        for (from, to, _edge_data) in graph.edges() {
            edge_info.push(EdgeReflection {
                from,
                to,
                data_type: DType::F32,            // TODO: Get actual edge type
                tensor_shape: Shape::new(vec![]), // TODO: Get actual shape
            });
        }

        GraphReflection {
            node_info,
            edge_info,
            type_analysis,
            graph_properties: self.analyze_graph_properties(graph),
        }
    }

    /// Get operation metadata
    fn get_operation_metadata(&self, op_name: &str) -> Option<OperationInfo> {
        self.operation_registry.get(op_name).cloned()
    }

    /// Analyze graph properties
    fn analyze_graph_properties(&self, graph: &ComputationGraph) -> GraphProperties {
        GraphProperties {
            node_count: graph.node_count(),
            edge_count: graph.edge_count(),
            is_acyclic: graph.is_acyclic(),
            has_control_flow: false, // TODO: Implement control flow detection
            complexity_estimate: self.estimate_complexity(graph),
        }
    }

    /// Estimate computational complexity
    fn estimate_complexity(&self, graph: &ComputationGraph) -> ComplexityEstimate {
        let mut total_ops = 0;
        let mut memory_usage = 0;

        for (_, node) in graph.nodes() {
            // Estimate operations based on node type
            let ops = match node.op.as_str() {
                "add" | "sub" | "mul" | "div" => 1,
                "matmul" => node.output_shape.size(0).unwrap_or(1).pow(3), // O(n^3) for matrix multiplication
                "conv2d" => node.output_shape.size(0).unwrap_or(1) * 9,    // Rough estimate
                _ => 1,
            };
            total_ops += ops;

            // Estimate memory usage
            memory_usage += node.output_shape.size(0).unwrap_or(1) * node.dtype.size_bytes();
        }

        ComplexityEstimate {
            operation_count: total_ops,
            memory_usage_bytes: memory_usage,
            estimated_flops: total_ops as f64,
        }
    }
}

/// Information about types
#[derive(Debug, Clone)]
pub struct TypeInfo {
    pub name: String,
    pub size_bytes: usize,
    pub alignment: usize,
    pub is_numeric: bool,
    pub is_floating_point: bool,
}

/// Information about operations
#[derive(Debug, Clone)]
pub struct OperationInfo {
    pub name: String,
    pub input_count: usize,
    pub output_count: usize,
    pub is_commutative: bool,
    pub is_associative: bool,
    pub complexity_class: ComplexityClass,
}

/// Complexity classifications
#[derive(Debug, Clone)]
pub enum ComplexityClass {
    Constant,
    Linear,
    Quadratic,
    Cubic,
    Exponential,
}

/// Dynamic code generator
pub struct DynamicCodeGenerator {
    backend: CodegenBackend,
}

impl DynamicCodeGenerator {
    /// Create a new dynamic code generator
    pub fn new() -> Self {
        Self {
            backend: CodegenBackend::Rust,
        }
    }

    /// Set the code generation backend
    pub fn set_backend(&mut self, backend: CodegenBackend) {
        self.backend = backend;
    }

    /// Format a type for the current backend
    pub fn format_type(&self, dtype: &DType) -> String {
        match self.backend {
            CodegenBackend::Rust => match dtype {
                DType::F32 => "f32".to_string(),
                DType::F64 => "f64".to_string(),
                DType::I32 => "i32".to_string(),
                DType::I64 => "i64".to_string(),
                DType::Bool => "bool".to_string(),
                _ => "f32".to_string(), // Default fallback
            },
            CodegenBackend::C => match dtype {
                DType::F32 => "float".to_string(),
                DType::F64 => "double".to_string(),
                DType::I32 => "int".to_string(),
                DType::I64 => "long".to_string(),
                DType::Bool => "bool".to_string(),
                _ => "float".to_string(),
            },
        }
    }

    /// Format a shape for the current backend
    pub fn format_shape(&self, shape: &Shape) -> String {
        let dims: Vec<String> = shape.dims().iter().map(|d| d.to_string()).collect();
        format!("[{}]", dims.join(", "))
    }

    /// Format a constant for the current backend
    pub fn format_constant(&self, value: &ConstantValue) -> String {
        match value {
            ConstantValue::Integer(i) => i.to_string(),
            ConstantValue::Float(f) => f.to_string(),
            ConstantValue::Boolean(b) => b.to_string(),
            ConstantValue::String(s) => format!("\"{}\"", s),
        }
    }

    /// Format an expression for the current backend
    pub fn format_expression(&self, expr: &str) -> String {
        // For now, just return the expression as-is
        // In a real implementation, this would parse and transform the expression
        expr.to_string()
    }
}

/// Code generation backends
#[derive(Debug, Clone)]
pub enum CodegenBackend {
    Rust,
    C,
}

/// Generated code with metadata
#[derive(Debug, Clone)]
pub struct GeneratedCode {
    pub source: String,
    pub metadata: CodeMetadata,
}

/// Metadata for generated code
#[derive(Debug, Clone)]
pub struct CodeMetadata {
    pub template_name: String,
    pub parameters: TemplateParameters,
    pub generated_at: std::time::SystemTime,
}

/// Specialization information for code generation
#[derive(Debug, Clone)]
pub struct SpecializationInfo {
    pub type_info: HashMap<String, DType>,
    pub shape_info: HashMap<String, Shape>,
    pub constants: HashMap<String, ConstantValue>,
}

/// Graph reflection information
#[derive(Debug)]
pub struct GraphReflection {
    pub node_info: HashMap<NodeId, NodeReflection>,
    pub edge_info: Vec<EdgeReflection>,
    pub type_analysis: HashMap<NodeId, TypeAnalysis>,
    pub graph_properties: GraphProperties,
}

/// Reflection information for a single node
#[derive(Debug)]
pub struct NodeReflection {
    pub id: NodeId,
    pub operation: String,
    pub input_types: Vec<DType>,
    pub output_type: DType,
    pub output_shape: Shape,
    pub metadata: Option<OperationInfo>,
}

/// Reflection information for an edge
#[derive(Debug)]
pub struct EdgeReflection {
    pub from: NodeId,
    pub to: NodeId,
    pub data_type: DType,
    pub tensor_shape: Shape,
}

/// Type analysis information
#[derive(Debug)]
pub struct TypeAnalysis {
    pub declared_type: DType,
    pub inferred_type: DType,
    pub type_constraints: Vec<String>,
}

/// Graph-level properties
#[derive(Debug)]
pub struct GraphProperties {
    pub node_count: usize,
    pub edge_count: usize,
    pub is_acyclic: bool,
    pub has_control_flow: bool,
    pub complexity_estimate: ComplexityEstimate,
}

/// Computational complexity estimate
#[derive(Debug)]
pub struct ComplexityEstimate {
    pub operation_count: usize,
    pub memory_usage_bytes: usize,
    pub estimated_flops: f64,
}

/// Create a standard element-wise operation template
pub fn create_elementwise_template(op_name: &str) -> CodeTemplate {
    let template_string = format!(
        r#"
fn {}(a: &Tensor<${{T}}>, b: &Tensor<${{T}}>) -> Tensor<${{T}}> {{
    let shape = a.shape();
    let mut result = Tensor::zeros(shape.clone());
    
    for i in 0..shape.size(0).unwrap_or(1) {{
        let a_val = a.data()[i];
        let b_val = b.data()[i];
        result.data_mut()[i] = a_val {} b_val;
    }}
    
    result
}}
"#,
        op_name,
        match op_name {
            "add" => "+",
            "sub" => "-",
            "mul" => "*",
            "div" => "/",
            _ => "+",
        }
    );

    let mut template = CodeTemplate::new(format!("{}_template", op_name), template_string);

    template.add_parameter(TemplateParameter {
        name: "T".to_string(),
        param_type: ParameterType::Type,
        default_value: Some("f32".to_string()),
        description: "Element type".to_string(),
    });

    template.add_constraint(TemplateConstraint::TypeConstraint {
        param_name: "T".to_string(),
        allowed_types: vec![DType::F32, DType::F64, DType::I32, DType::I64],
    });

    template
}

/// Create a convolution operation template
pub fn create_conv2d_template() -> CodeTemplate {
    let template_string = r#"
fn conv2d(
    input: &Tensor<${T}>, 
    weight: &Tensor<${T}>,
    stride: $const{stride},
    padding: $const{padding}
) -> Tensor<${T}> {
    let batch_size = input.shape()[0];
    let in_channels = input.shape()[1];
    let in_height = input.shape()[2];
    let in_width = input.shape()[3];
    
    let out_channels = weight.shape()[0];
    let kernel_height = weight.shape()[2];
    let kernel_width = weight.shape()[3];
    
    let out_height = (in_height + 2 * padding - kernel_height) / stride + 1;
    let out_width = (in_width + 2 * padding - kernel_width) / stride + 1;
    
    let output_shape = vec![batch_size, out_channels, out_height, out_width];
    let mut output = Tensor::zeros(output_shape);
    
    $code{convolution_kernel}
    
    output
}
"#
    .to_string();

    let mut template = CodeTemplate::new("conv2d_template".to_string(), template_string);

    template.add_parameter(TemplateParameter {
        name: "T".to_string(),
        param_type: ParameterType::Type,
        default_value: Some("f32".to_string()),
        description: "Element type".to_string(),
    });

    template.add_parameter(TemplateParameter {
        name: "stride".to_string(),
        param_type: ParameterType::Constant,
        default_value: Some("1".to_string()),
        description: "Convolution stride".to_string(),
    });

    template.add_parameter(TemplateParameter {
        name: "padding".to_string(),
        param_type: ParameterType::Constant,
        default_value: Some("0".to_string()),
        description: "Convolution padding".to_string(),
    });

    template.add_parameter(TemplateParameter {
        name: "convolution_kernel".to_string(),
        param_type: ParameterType::CodeBlock,
        default_value: None,
        description: "Convolution kernel implementation".to_string(),
    });

    template
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::Shape;

    #[test]
    fn test_template_creation() {
        let template = create_elementwise_template("add");
        assert_eq!(template.name, "add_template");
        assert_eq!(template.parameters.len(), 1);
        assert_eq!(template.constraints.len(), 1);
    }

    #[test]
    fn test_template_instantiation() {
        let template = create_elementwise_template("add");
        let generator = DynamicCodeGenerator::new();

        let mut params = TemplateParameters::new();
        params.add_type("T".to_string(), DType::F32);

        let result = template.instantiate(&params, &generator);
        assert!(result.is_ok());

        let code = result.unwrap();
        assert!(code.source.contains("f32"));
        assert!(code.source.contains("fn add"));
    }

    #[test]
    fn test_metaprogramming_engine() {
        let mut engine = MetaprogrammingEngine::new();
        let template = create_elementwise_template("mul");
        engine.register_template("mul_op".to_string(), template);

        let mut params = TemplateParameters::new();
        params.add_type("T".to_string(), DType::F64);

        let result = engine.generate_from_template("mul_op", &params);
        assert!(result.is_ok());

        let code = result.unwrap();
        assert!(code.source.contains("f64"));
        assert!(code.source.contains("fn mul"));
    }

    #[test]
    fn test_macro_definition() {
        let macro_def = MacroDefinition::new(
            "BINARY_OP".to_string(),
            vec!["op".to_string(), "T".to_string()],
            "fn $op(a: $T, b: $T) -> $T { a $op b }".to_string(),
        );

        let generator = DynamicCodeGenerator::new();
        let args = vec![
            MacroArgument::Literal("+".to_string()),
            MacroArgument::Literal("f32".to_string()),
        ];

        let result = macro_def.expand(&args, &generator);
        assert!(result.is_ok());

        let code = result.unwrap();
        assert!(code.source.contains("fn +(a: f32, b: f32) -> f32"));
    }

    #[test]
    fn test_constant_value_formatting() {
        let generator = DynamicCodeGenerator::new();

        assert_eq!(generator.format_constant(&ConstantValue::Integer(42)), "42");
        assert_eq!(
            generator.format_constant(&ConstantValue::Float(3.14)),
            "3.14"
        );
        assert_eq!(
            generator.format_constant(&ConstantValue::Boolean(true)),
            "true"
        );
        assert_eq!(
            generator.format_constant(&ConstantValue::String("hello".to_string())),
            "\"hello\""
        );
    }

    #[test]
    fn test_template_constraints() {
        let mut template = CodeTemplate::new("test_template".to_string(), "test ${T}".to_string());

        template.add_constraint(TemplateConstraint::TypeConstraint {
            param_name: "T".to_string(),
            allowed_types: vec![DType::F32],
        });

        let generator = DynamicCodeGenerator::new();

        // Valid parameters
        let mut valid_params = TemplateParameters::new();
        valid_params.add_type("T".to_string(), DType::F32);

        let result = template.instantiate(&valid_params, &generator);
        assert!(result.is_ok());

        // Invalid parameters
        let mut invalid_params = TemplateParameters::new();
        invalid_params.add_type("T".to_string(), DType::I32);

        let result = template.instantiate(&invalid_params, &generator);
        assert!(result.is_err());
    }
}
