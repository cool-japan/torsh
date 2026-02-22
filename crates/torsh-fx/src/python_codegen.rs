//! Python code generation backend for FX graphs
//!
//! This module provides Python and PyTorch code generation capabilities,
//! translating FX graphs into executable Python code with support for
//! PyTorch neural network operations.

use crate::core::CodeGenBackend;
use crate::{FxGraph, Node};
use std::collections::HashSet;
use torsh_core::Result;

/// Python code generation backend
///
/// Generates Python code from FX graphs with optional PyTorch integration.
/// Supports neural network operations, tensor manipulations, and control flow.
#[derive(Debug, Clone)]
pub struct PythonCodeGen {
    /// Whether to use PyTorch for tensor operations
    pub use_torch: bool,
    /// Number of spaces per indentation level
    pub indent_size: usize,
}

impl Default for PythonCodeGen {
    fn default() -> Self {
        Self {
            use_torch: true,
            indent_size: 4,
        }
    }
}

impl PythonCodeGen {
    /// Create a new Python code generator with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Configure whether to use PyTorch for tensor operations
    pub fn with_torch(mut self, use_torch: bool) -> Self {
        self.use_torch = use_torch;
        self
    }

    /// Configure the indentation size
    pub fn with_indent_size(mut self, indent_size: usize) -> Self {
        self.indent_size = indent_size;
        self
    }

    /// Generate indentation string for the given level
    fn indent(&self, level: usize) -> String {
        " ".repeat(level * self.indent_size)
    }

    /// Generate import statements based on graph operations
    fn generate_imports(&self, graph: &FxGraph) -> String {
        let mut imports = HashSet::new();

        if self.use_torch {
            imports.insert("import torch");
            imports.insert("import torch.nn as nn");
            imports.insert("import torch.nn.functional as F");
        }

        // Analyze nodes to determine required imports
        for (_, node) in graph.nodes() {
            if let Node::Call(op_name, _) = node {
                match op_name.as_str() {
                    "numpy" | "np" => {
                        imports.insert("import numpy as np");
                    }
                    op if op.starts_with("torch.") => {
                        if !self.use_torch {
                            imports.insert("import torch");
                        }
                    }
                    _ => {}
                }
            }
        }

        let mut import_vec: Vec<&str> = imports.into_iter().collect();
        import_vec.sort();
        import_vec.join("\n")
    }

    /// Generate function signature from graph inputs
    fn generate_function_signature(&self, graph: &FxGraph) -> String {
        let mut params = Vec::new();

        for (_, node) in graph.nodes() {
            if let Node::Input(input_name) = node {
                params.push(input_name.clone());
            }
        }

        format!("def forward({}):", params.join(", "))
    }

    /// Sanitize variable names for Python identifiers
    fn sanitize_variable_name(&self, name: &str) -> String {
        // Replace invalid Python identifier characters
        name.chars()
            .map(|c| {
                if c.is_alphanumeric() || c == '_' {
                    c
                } else {
                    '_'
                }
            })
            .collect::<String>()
            .trim_start_matches(|c: char| c.is_numeric())
            .to_string()
    }

    /// Generate Python code for a single node
    fn generate_node_code(
        &self,
        node: &Node,
        node_idx: usize,
        graph: &FxGraph,
        indent_level: usize,
    ) -> String {
        let indent = self.indent(indent_level);

        match node {
            Node::Input(input_name) => {
                // Inputs are handled as function parameters
                format!("{indent}# Input: {input_name}")
            }

            Node::Call(op_name, args) => {
                let var_name = self.sanitize_variable_name(&format!("var_{node_idx}"));
                let args_str = args.join(", ");

                match op_name.as_str() {
                    // PyTorch operations
                    "relu" => format!("{indent}{var_name} = F.relu({args_str})"),
                    "sigmoid" => format!("{indent}{var_name} = torch.sigmoid({args_str})"),
                    "tanh" => format!("{indent}{var_name} = torch.tanh({args_str})"),
                    "softmax" => format!("{indent}{var_name} = F.softmax({args_str}, dim=-1)"),
                    "linear" => format!("{indent}{var_name} = F.linear({args_str})"),
                    "conv2d" => format!("{indent}{var_name} = F.conv2d({args_str})"),
                    "max_pool2d" => format!("{indent}{var_name} = F.max_pool2d({args_str})"),
                    "avg_pool2d" => format!("{indent}{var_name} = F.avg_pool2d({args_str})"),
                    "batch_norm" => format!("{indent}{var_name} = F.batch_norm({args_str})"),
                    "dropout" => {
                        format!(
                            "{indent}{var_name} = F.dropout({args_str}, training=self.training)"
                        )
                    }

                    // Math operations
                    "add" => {
                        let arg0 = args.first().cloned().unwrap_or_else(|| "0".to_string());
                        let arg1 = args.get(1).cloned().unwrap_or_else(|| "0".to_string());
                        format!("{indent}{var_name} = {arg0} + {arg1}")
                    }
                    "sub" => {
                        let arg0 = args.first().cloned().unwrap_or_else(|| "0".to_string());
                        let arg1 = args.get(1).cloned().unwrap_or_else(|| "0".to_string());
                        format!("{indent}{var_name} = {arg0} - {arg1}")
                    }
                    "mul" => {
                        let arg0 = args.first().cloned().unwrap_or_else(|| "1".to_string());
                        let arg1 = args.get(1).cloned().unwrap_or_else(|| "1".to_string());
                        format!("{indent}{var_name} = {arg0} * {arg1}")
                    }
                    "div" => {
                        let arg0 = args.first().cloned().unwrap_or_else(|| "1".to_string());
                        let arg1 = args.get(1).cloned().unwrap_or_else(|| "1".to_string());
                        format!("{indent}{var_name} = {arg0} / {arg1}")
                    }
                    "matmul" => format!("{indent}{var_name} = torch.matmul({args_str})"),

                    // Tensor operations
                    "reshape" => format!(
                        "{indent}{var_name} = {}.reshape({})",
                        args.first().unwrap_or(&"tensor".to_string()),
                        args[1..].join(", ")
                    ),
                    "transpose" => format!(
                        "{indent}{var_name} = {}.transpose({})",
                        args.first().unwrap_or(&"tensor".to_string()),
                        args[1..].join(", ")
                    ),
                    "permute" => format!(
                        "{indent}{var_name} = {}.permute({})",
                        args.first().unwrap_or(&"tensor".to_string()),
                        args[1..].join(", ")
                    ),
                    "squeeze" => format!(
                        "{indent}{var_name} = {}.squeeze({})",
                        args.first().unwrap_or(&"tensor".to_string()),
                        args[1..].join(", ")
                    ),
                    "unsqueeze" => format!(
                        "{indent}{var_name} = {}.unsqueeze({})",
                        args.first().unwrap_or(&"tensor".to_string()),
                        args[1..].join(", ")
                    ),

                    // Generic function call
                    _ => format!("{indent}{var_name} = {op_name}({args_str})"),
                }
            }

            Node::Output => {
                // Find the input to this output node
                let predecessors: Vec<_> = graph
                    .graph
                    .edges_directed(
                        petgraph::graph::NodeIndex::new(node_idx),
                        petgraph::Direction::Incoming,
                    )
                    .collect();

                if let Some(edge) = predecessors.first() {
                    let pred_idx = edge.source().index();
                    let var_name = self.sanitize_variable_name(&format!("var_{pred_idx}"));
                    format!("{indent}return {var_name}")
                } else {
                    format!("{indent}return None  # No input to output node")
                }
            }

            Node::Conditional {
                condition,
                then_branch,
                else_branch,
            } => {
                let mut code = format!("{indent}if {condition}:");
                for stmt in then_branch {
                    code.push_str(&format!("\n{}{}", self.indent(indent_level + 1), stmt));
                }
                if !else_branch.is_empty() {
                    code.push_str(&format!("\n{indent}else:"));
                    for stmt in else_branch {
                        code.push_str(&format!("\n{}{}", self.indent(indent_level + 1), stmt));
                    }
                }
                code
            }

            Node::Loop {
                condition,
                body,
                loop_vars,
            } => {
                let mut code = format!("{indent}while {condition}:");
                for var in loop_vars {
                    code.push_str(&format!(
                        "\n{}# Loop variable: {}",
                        self.indent(indent_level + 1),
                        var
                    ));
                }
                for stmt in body {
                    code.push_str(&format!("\n{}{}", self.indent(indent_level + 1), stmt));
                }
                code
            }

            Node::Merge { inputs } => {
                let var_name = self.sanitize_variable_name(&format!("var_{node_idx}"));
                let inputs_str = inputs.join(", ");
                format!("{indent}{var_name} = [{inputs_str}]  # Merge inputs")
            }

            Node::GetAttr { target, attr } => {
                let var_name = self.sanitize_variable_name(&format!("var_{node_idx}"));
                format!("{indent}{var_name} = {target}.{attr}")
            }
        }
    }

    /// Generate PyTorch module class wrapper around the function
    fn generate_class_wrapper(&self, function_code: &str) -> String {
        format!(
            r#"class GeneratedModule(nn.Module):
    def __init__(self):
        super().__init__()
        # Add module parameters and buffers here as needed

{}
"#,
            function_code
                .lines()
                .map(|line| format!("    {line}"))
                .collect::<Vec<_>>()
                .join("\n")
        )
    }
}

impl CodeGenBackend for PythonCodeGen {
    fn generate(&self, graph: &FxGraph) -> Result<String> {
        let mut code = String::new();

        // Add file header
        code.push_str("# Generated code from torsh-fx\n");
        code.push_str("# This file was automatically generated. Do not edit manually.\n\n");

        // Add imports
        let imports = self.generate_imports(graph);
        if !imports.is_empty() {
            code.push_str(&imports);
            code.push_str("\n\n");
        }

        // Generate function body
        let mut function_body = String::new();
        function_body.push_str(&self.generate_function_signature(graph));
        function_body.push('\n');

        // Process nodes in topological order
        let mut visited = HashSet::new();
        let mut node_order = Vec::new();

        // Simple topological sort (assuming the graph is already valid)
        for (idx, _) in graph.nodes() {
            if !visited.contains(&idx) {
                node_order.push(idx);
                visited.insert(idx);
            }
        }

        // Generate code for each node
        for node_idx in node_order {
            if let Some(node) = graph.get_node(node_idx) {
                let node_code = self.generate_node_code(node, node_idx.index(), graph, 1);
                if !node_code.trim().is_empty() {
                    function_body.push_str(&node_code);
                    function_body.push('\n');
                }
            }
        }

        // Wrap in class if using PyTorch
        if self.use_torch {
            code.push_str(&self.generate_class_wrapper(&function_body));
        } else {
            code.push_str(&function_body);
        }

        Ok(code)
    }

    fn file_extension(&self) -> &'static str {
        "py"
    }

    fn language_name(&self) -> &'static str {
        "Python"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_python_codegen_creation() {
        let codegen = PythonCodeGen::new();
        assert!(codegen.use_torch);
        assert_eq!(codegen.indent_size, 4);
    }

    #[test]
    fn test_python_codegen_configuration() {
        let codegen = PythonCodeGen::new().with_torch(false).with_indent_size(2);

        assert!(!codegen.use_torch);
        assert_eq!(codegen.indent_size, 2);
    }

    #[test]
    fn test_indent_generation() {
        let codegen = PythonCodeGen::new();
        assert_eq!(codegen.indent(0), "");
        assert_eq!(codegen.indent(1), "    ");
        assert_eq!(codegen.indent(2), "        ");
    }

    #[test]
    fn test_sanitize_variable_name() {
        let codegen = PythonCodeGen::new();
        assert_eq!(codegen.sanitize_variable_name("valid_name"), "valid_name");
        assert_eq!(
            codegen.sanitize_variable_name("invalid-name!"),
            "invalid_name_"
        );
        assert_eq!(codegen.sanitize_variable_name("123invalid"), "invalid");
    }

    #[test]
    fn test_file_extension() {
        let codegen = PythonCodeGen::new();
        assert_eq!(codegen.file_extension(), "py");
    }

    #[test]
    fn test_language_name() {
        let codegen = PythonCodeGen::new();
        assert_eq!(codegen.language_name(), "Python");
    }

    #[test]
    fn test_generate_class_wrapper() {
        let codegen = PythonCodeGen::new();
        let function_code = "def forward(x):\n    return x";
        let wrapped = codegen.generate_class_wrapper(function_code);

        assert!(wrapped.contains("class GeneratedModule(nn.Module)"));
        assert!(wrapped.contains("def __init__(self)"));
        assert!(wrapped.contains("super().__init__()"));
    }
}
