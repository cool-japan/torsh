//! C++ code generation backend for FX graphs
//!
//! This module provides C++ code generation capabilities with optional LibTorch
//! integration, translating FX graphs into executable C++ code.

use crate::core::CodeGenBackend;
use crate::{FxGraph, Node};
use std::collections::HashSet;
use torsh_core::Result;

/// C++ code generation backend
///
/// Generates C++ code from FX graphs with optional LibTorch integration.
/// Supports tensor operations, mathematical functions, and control flow structures.
#[derive(Debug, Clone)]
pub struct CppCodeGen {
    /// Whether to use LibTorch for tensor operations
    pub use_libtorch: bool,
    /// Number of spaces per indentation level
    pub indent_size: usize,
}

impl Default for CppCodeGen {
    fn default() -> Self {
        Self {
            use_libtorch: true,
            indent_size: 2,
        }
    }
}

impl CppCodeGen {
    /// Create a new C++ code generator with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Configure whether to use LibTorch for tensor operations
    pub fn with_libtorch(mut self, use_libtorch: bool) -> Self {
        self.use_libtorch = use_libtorch;
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

    /// Generate include statements based on configuration
    fn generate_includes(&self) -> String {
        if self.use_libtorch {
            r#"#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <vector>
#include <memory>"#
                .to_string()
        } else {
            r#"#include <iostream>
#include <vector>
#include <memory>
#include <cmath>"#
                .to_string()
        }
    }

    /// Get the appropriate C++ type for an operation
    #[allow(dead_code)]
    fn cpp_type_for_operation(&self, op_name: &str) -> &'static str {
        if self.use_libtorch {
            "torch::Tensor"
        } else {
            match op_name {
                "relu" | "sigmoid" | "tanh" => "double",
                _ => "auto",
            }
        }
    }

    /// Generate C++ code for a single node
    fn generate_node_code(
        &self,
        node: &Node,
        node_idx: usize,
        _graph: &FxGraph,
        indent_level: usize,
    ) -> String {
        let indent = self.indent(indent_level);

        match node {
            Node::Input(input_name) => {
                format!("{indent}// Input: {input_name}")
            }

            Node::Call(op_name, args) => {
                let var_name = format!("var_{node_idx}");
                let args_str = args.join(", ");

                if self.use_libtorch {
                    match op_name.as_str() {
                        "relu" => format!("{indent}auto {var_name} = torch::relu({args_str});"),
                        "sigmoid" => {
                            format!("{indent}auto {var_name} = torch::sigmoid({args_str});")
                        }
                        "tanh" => format!("{indent}auto {var_name} = torch::tanh({args_str});"),
                        "softmax" => {
                            format!("{indent}auto {var_name} = torch::softmax({args_str}, -1);")
                        }
                        "linear" => format!("{indent}auto {var_name} = torch::linear({args_str});"),
                        "conv2d" => format!("{indent}auto {var_name} = torch::conv2d({args_str});"),
                        "max_pool2d" => format!("{indent}auto {var_name} = torch::max_pool2d({args_str});"),
                        "avg_pool2d" => format!("{indent}auto {var_name} = torch::avg_pool2d({args_str});"),
                        "batch_norm" => format!("{indent}auto {var_name} = torch::batch_norm({args_str});"),
                        "dropout" => format!("{indent}auto {var_name} = torch::dropout({args_str});"),

                        // Arithmetic operations
                        "add" => format!(
                            "{}auto {} = {} + {};",
                            indent,
                            var_name,
                            args.first().unwrap_or(&"0".to_string()),
                            args.get(1).unwrap_or(&"0".to_string())
                        ),
                        "sub" => format!(
                            "{}auto {} = {} - {};",
                            indent,
                            var_name,
                            args.first().unwrap_or(&"0".to_string()),
                            args.get(1).unwrap_or(&"0".to_string())
                        ),
                        "mul" => format!(
                            "{}auto {} = {} * {};",
                            indent,
                            var_name,
                            args.first().unwrap_or(&"1".to_string()),
                            args.get(1).unwrap_or(&"1".to_string())
                        ),
                        "div" => format!(
                            "{}auto {} = {} / {};",
                            indent,
                            var_name,
                            args.first().unwrap_or(&"1".to_string()),
                            args.get(1).unwrap_or(&"1".to_string())
                        ),
                        "matmul" => format!("{indent}auto {var_name} = torch::matmul({args_str});"),

                        // Tensor operations
                        "reshape" => format!("{indent}auto {var_name} = {}.reshape({{ {} }});",
                            args.first().unwrap_or(&"tensor".to_string()),
                            args[1..].join(", ")
                        ),
                        "transpose" => format!("{indent}auto {var_name} = {}.transpose({}, {});",
                            args.first().unwrap_or(&"tensor".to_string()),
                            args.get(1).unwrap_or(&"0".to_string()),
                            args.get(2).unwrap_or(&"1".to_string())
                        ),
                        "squeeze" => format!("{indent}auto {var_name} = {}.squeeze({});",
                            args.first().unwrap_or(&"tensor".to_string()),
                            args[1..].join(", ")
                        ),
                        "unsqueeze" => format!("{indent}auto {var_name} = {}.unsqueeze({});",
                            args.first().unwrap_or(&"tensor".to_string()),
                            args.get(1).unwrap_or(&"0".to_string())
                        ),

                        // Generic function call
                        _ => format!("{indent}auto {var_name} = {op_name}({args_str});"),
                    }
                } else {
                    // Standard C++ without LibTorch
                    match op_name.as_str() {
                        "relu" => format!(
                            "{}auto {} = std::max(0.0, {});",
                            indent,
                            var_name,
                            args.first().unwrap_or(&"0.0".to_string())
                        ),
                        "sigmoid" => format!(
                            "{}auto {} = 1.0 / (1.0 + std::exp(-{}));",
                            indent,
                            var_name,
                            args.first().unwrap_or(&"0.0".to_string())
                        ),
                        "tanh" => format!(
                            "{}auto {} = std::tanh({});",
                            indent,
                            var_name,
                            args.first().unwrap_or(&"0.0".to_string())
                        ),
                        "add" => format!(
                            "{}auto {} = {} + {};",
                            indent,
                            var_name,
                            args.first().unwrap_or(&"0.0".to_string()),
                            args.get(1).unwrap_or(&"0.0".to_string())
                        ),
                        "sub" => format!(
                            "{}auto {} = {} - {};",
                            indent,
                            var_name,
                            args.first().unwrap_or(&"0.0".to_string()),
                            args.get(1).unwrap_or(&"0.0".to_string())
                        ),
                        "mul" => format!(
                            "{}auto {} = {} * {};",
                            indent,
                            var_name,
                            args.first().unwrap_or(&"1.0".to_string()),
                            args.get(1).unwrap_or(&"1.0".to_string())
                        ),
                        "div" => format!(
                            "{}auto {} = {} / {};",
                            indent,
                            var_name,
                            args.first().unwrap_or(&"1.0".to_string()),
                            args.get(1).unwrap_or(&"1.0".to_string())
                        ),
                        _ => format!("{indent}auto {var_name} = {op_name}({args_str});"),
                    }
                }
            }

            Node::Output => {
                format!("{}return var_{};", indent, node_idx.saturating_sub(1))
            }

            Node::Conditional {
                condition,
                then_branch,
                else_branch,
            } => {
                let mut code = format!("{indent}if ({condition}) {{");
                for stmt in then_branch {
                    code.push_str(&format!("\n{}{};", self.indent(indent_level + 1), stmt));
                }
                code.push_str(&format!("\n{indent}}}"));
                if !else_branch.is_empty() {
                    code.push_str(" else {");
                    for stmt in else_branch {
                        code.push_str(&format!("\n{}{};", self.indent(indent_level + 1), stmt));
                    }
                    code.push_str(&format!("\n{indent}}}"));
                }
                code
            }

            Node::Loop {
                condition,
                body,
                loop_vars: _,
            } => {
                let mut code = format!("{indent}while ({condition}) {{");
                for stmt in body {
                    code.push_str(&format!("\n{}{};", self.indent(indent_level + 1), stmt));
                }
                code.push_str(&format!("\n{indent}}}"));
                code
            }

            Node::Merge { inputs } => {
                let var_name = format!("var_{node_idx}");
                if self.use_libtorch {
                    format!("{indent}auto {var_name} = torch::cat({{ {} }});", inputs.join(", "))
                } else {
                    format!("{indent}// Merge inputs: {}", inputs.join(", "))
                }
            }

            Node::GetAttr { target, attr } => {
                let var_name = format!("var_{node_idx}");
                format!("{indent}auto {var_name} = {target}.{attr};")
            }
        }
    }
}

impl CodeGenBackend for CppCodeGen {
    fn generate(&self, graph: &FxGraph) -> Result<String> {
        let mut code = String::new();

        // Add file header
        code.push_str("// Generated code from torsh-fx\n");
        code.push_str("// This file was automatically generated. Do not edit manually.\n\n");

        // Add includes
        code.push_str(&self.generate_includes());
        code.push_str("\n\n");

        // Generate function signature
        if self.use_libtorch {
            code.push_str("torch::Tensor forward(");
        } else {
            code.push_str("auto forward(");
        }

        // Collect input parameters
        let mut params = Vec::new();
        for (_, node) in graph.nodes() {
            if let Node::Input(input_name) = node {
                if self.use_libtorch {
                    params.push(format!("const torch::Tensor& {input_name}"));
                } else {
                    params.push(format!("double {input_name}"));
                }
            }
        }

        code.push_str(&params.join(", "));
        code.push_str(") {\n");

        // Generate function body
        let mut visited = HashSet::new();
        let mut node_order = Vec::new();

        // Simple topological sort
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
                    code.push_str(&node_code);
                    code.push('\n');
                }
            }
        }

        code.push_str("}\n");

        Ok(code)
    }

    fn file_extension(&self) -> &'static str {
        "cpp"
    }

    fn language_name(&self) -> &'static str {
        "C++"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpp_codegen_creation() {
        let codegen = CppCodeGen::new();
        assert!(codegen.use_libtorch);
        assert_eq!(codegen.indent_size, 2);
    }

    #[test]
    fn test_cpp_codegen_configuration() {
        let codegen = CppCodeGen::new()
            .with_libtorch(false)
            .with_indent_size(4);

        assert!(!codegen.use_libtorch);
        assert_eq!(codegen.indent_size, 4);
    }

    #[test]
    fn test_indent_generation() {
        let codegen = CppCodeGen::new();
        assert_eq!(codegen.indent(0), "");
        assert_eq!(codegen.indent(1), "  ");
        assert_eq!(codegen.indent(2), "    ");
    }

    #[test]
    fn test_file_extension() {
        let codegen = CppCodeGen::new();
        assert_eq!(codegen.file_extension(), "cpp");
    }

    #[test]
    fn test_language_name() {
        let codegen = CppCodeGen::new();
        assert_eq!(codegen.language_name(), "C++");
    }

    #[test]
    fn test_generate_includes_with_libtorch() {
        let codegen = CppCodeGen::new().with_libtorch(true);
        let includes = codegen.generate_includes();
        assert!(includes.contains("#include <torch/torch.h>"));
        assert!(includes.contains("#include <torch/script.h>"));
    }

    #[test]
    fn test_generate_includes_without_libtorch() {
        let codegen = CppCodeGen::new().with_libtorch(false);
        let includes = codegen.generate_includes();
        assert!(includes.contains("#include <iostream>"));
        assert!(includes.contains("#include <cmath>"));
        assert!(!includes.contains("torch"));
    }

    #[test]
    fn test_cpp_type_for_operation() {
        let codegen_with_libtorch = CppCodeGen::new().with_libtorch(true);
        assert_eq!(codegen_with_libtorch.cpp_type_for_operation("relu"), "torch::Tensor");

        let codegen_without_libtorch = CppCodeGen::new().with_libtorch(false);
        assert_eq!(codegen_without_libtorch.cpp_type_for_operation("relu"), "double");
        assert_eq!(codegen_without_libtorch.cpp_type_for_operation("unknown"), "auto");
    }
}