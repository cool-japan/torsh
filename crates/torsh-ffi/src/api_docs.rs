//! API Documentation Generator for ToRSh FFI
//!
//! This module provides tools to automatically generate comprehensive API documentation
//! for the ToRSh FFI bindings across different programming languages.

#![allow(dead_code)]

use crate::binding_generator::{FunctionSignature, TargetLanguage};
use crate::error::FfiResult;
use std::collections::HashMap;
use std::fmt::Write;

/// Documentation format types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DocFormat {
    Markdown,
    Html,
    RestructuredText,
    ApiDoc,
    Javadoc,
    Sphinx,
    GoDoc,
    SwiftDoc,
    RDoc,
    JuliaDoc,
}

impl DocFormat {
    pub fn file_extension(&self) -> &'static str {
        match self {
            DocFormat::Markdown => "md",
            DocFormat::Html => "html",
            DocFormat::RestructuredText => "rst",
            DocFormat::ApiDoc => "txt",
            DocFormat::Javadoc => "html",
            DocFormat::Sphinx => "rst",
            DocFormat::GoDoc => "md",
            DocFormat::SwiftDoc => "md",
            DocFormat::RDoc => "Rd",
            DocFormat::JuliaDoc => "md",
        }
    }

    pub fn for_language(lang: &TargetLanguage) -> Self {
        match lang {
            TargetLanguage::Java => DocFormat::Javadoc,
            TargetLanguage::Python => DocFormat::Sphinx,
            TargetLanguage::Go => DocFormat::GoDoc,
            TargetLanguage::Swift => DocFormat::SwiftDoc,
            TargetLanguage::R => DocFormat::RDoc,
            TargetLanguage::Julia => DocFormat::JuliaDoc,
            _ => DocFormat::Markdown,
        }
    }
}

/// Function category for organization
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FunctionCategory {
    TensorCreation,
    TensorOperations,
    TensorManipulation,
    NeuralNetworks,
    Optimization,
    Utilities,
    MemoryManagement,
    ErrorHandling,
    Performance,
}

impl FunctionCategory {
    pub fn description(&self) -> &'static str {
        match self {
            FunctionCategory::TensorCreation => "Functions for creating new tensors",
            FunctionCategory::TensorOperations => "Mathematical operations on tensors",
            FunctionCategory::TensorManipulation => {
                "Functions for reshaping, indexing, and manipulating tensors"
            }
            FunctionCategory::NeuralNetworks => "Neural network layers and modules",
            FunctionCategory::Optimization => "Optimizers and training utilities",
            FunctionCategory::Utilities => "Utility and helper functions",
            FunctionCategory::MemoryManagement => "Memory allocation and cleanup functions",
            FunctionCategory::ErrorHandling => "Error management and diagnostics",
            FunctionCategory::Performance => "Performance optimization and profiling",
        }
    }

    pub fn from_function_name(name: &str) -> Self {
        if name.contains("tensor_zeros")
            || name.contains("tensor_ones")
            || name.contains("tensor_rand")
            || name.contains("tensor_new")
        {
            FunctionCategory::TensorCreation
        } else if name.contains("tensor_add")
            || name.contains("tensor_mul")
            || name.contains("tensor_matmul")
            || name.contains("tensor_sub")
        {
            FunctionCategory::TensorOperations
        } else if name.contains("tensor_reshape")
            || name.contains("tensor_transpose")
            || name.contains("tensor_view")
        {
            FunctionCategory::TensorManipulation
        } else if name.contains("linear")
            || name.contains("conv")
            || name.contains("relu")
            || name.contains("module")
        {
            FunctionCategory::NeuralNetworks
        } else if name.contains("sgd") || name.contains("adam") || name.contains("optimizer") {
            FunctionCategory::Optimization
        } else if name.contains("free") || name.contains("cleanup") {
            FunctionCategory::MemoryManagement
        } else if name.contains("error") || name.contains("clear") {
            FunctionCategory::ErrorHandling
        } else if name.contains("batch") || name.contains("performance") || name.contains("stats") {
            FunctionCategory::Performance
        } else {
            FunctionCategory::Utilities
        }
    }
}

/// API documentation entry
#[derive(Debug, Clone)]
pub struct ApiDocEntry {
    pub function: FunctionSignature,
    pub category: FunctionCategory,
    pub examples: Vec<String>,
    pub notes: Vec<String>,
    pub see_also: Vec<String>,
    pub since_version: Option<String>,
}

impl ApiDocEntry {
    pub fn new(function: FunctionSignature) -> Self {
        let category = FunctionCategory::from_function_name(&function.name);
        Self {
            function,
            category,
            examples: Vec::new(),
            notes: Vec::new(),
            see_also: Vec::new(),
            since_version: None,
        }
    }

    pub fn with_example(mut self, example: String) -> Self {
        self.examples.push(example);
        self
    }

    pub fn with_note(mut self, note: String) -> Self {
        self.notes.push(note);
        self
    }

    pub fn with_see_also(mut self, reference: String) -> Self {
        self.see_also.push(reference);
        self
    }

    pub fn with_version(mut self, version: String) -> Self {
        self.since_version = Some(version);
        self
    }
}

/// API documentation generator
pub struct ApiDocGenerator {
    target_language: TargetLanguage,
    format: DocFormat,
    entries: Vec<ApiDocEntry>,
    metadata: HashMap<String, String>,
}

impl ApiDocGenerator {
    pub fn new(target_language: TargetLanguage) -> Self {
        let format = DocFormat::for_language(&target_language);
        Self {
            target_language,
            format,
            entries: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    pub fn with_format(mut self, format: DocFormat) -> Self {
        self.format = format;
        self
    }

    pub fn add_entry(&mut self, entry: ApiDocEntry) {
        self.entries.push(entry);
    }

    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }

    pub fn load_standard_functions(&mut self) {
        // Core tensor functions with examples
        let tensor_zeros = FunctionSignature {
            name: "torsh_tensor_zeros".to_string(),
            return_type: "*mut TorshTensor".to_string(),
            parameters: vec![
                ("shape".to_string(), "*const c_int".to_string()),
                ("shape_len".to_string(), "c_int".to_string()),
            ],
            description: "Create a tensor filled with zeros".to_string(),
            is_unsafe: true,
        };

        let entry = ApiDocEntry::new(tensor_zeros)
            .with_example(self.generate_example_for_function("torsh_tensor_zeros"))
            .with_note("The shape array must contain positive integers".to_string())
            .with_see_also("torsh_tensor_ones".to_string())
            .with_version("0.1.0".to_string());
        self.add_entry(entry);

        let tensor_add = FunctionSignature {
            name: "torsh_tensor_add".to_string(),
            return_type: "*mut TorshTensor".to_string(),
            parameters: vec![
                ("a".to_string(), "*mut TorshTensor".to_string()),
                ("b".to_string(), "*mut TorshTensor".to_string()),
            ],
            description: "Add two tensors element-wise".to_string(),
            is_unsafe: true,
        };

        let entry = ApiDocEntry::new(tensor_add)
            .with_example(self.generate_example_for_function("torsh_tensor_add"))
            .with_note("Both tensors must have compatible shapes for broadcasting".to_string())
            .with_see_also("torsh_tensor_sub".to_string())
            .with_see_also("torsh_tensor_mul".to_string())
            .with_version("0.1.0".to_string());
        self.add_entry(entry);

        let linear_create = FunctionSignature {
            name: "torsh_linear_create".to_string(),
            return_type: "*mut TorshModule".to_string(),
            parameters: vec![
                ("in_features".to_string(), "c_int".to_string()),
                ("out_features".to_string(), "c_int".to_string()),
                ("bias".to_string(), "bool".to_string()),
            ],
            description: "Create a linear (fully connected) layer".to_string(),
            is_unsafe: true,
        };

        let entry = ApiDocEntry::new(linear_create)
            .with_example(self.generate_example_for_function("torsh_linear_create"))
            .with_note("The layer weights are initialized randomly".to_string())
            .with_see_also("torsh_linear_forward".to_string())
            .with_version("0.1.0".to_string());
        self.add_entry(entry);

        // Add more standard functions...
        self.add_optimizer_functions();
        self.add_utility_functions();
    }

    fn add_optimizer_functions(&mut self) {
        let sgd_create = FunctionSignature {
            name: "torsh_sgd_create".to_string(),
            return_type: "*mut TorshOptimizer".to_string(),
            parameters: vec![("learning_rate".to_string(), "c_float".to_string())],
            description: "Create a Stochastic Gradient Descent optimizer".to_string(),
            is_unsafe: true,
        };

        let entry = ApiDocEntry::new(sgd_create)
            .with_example(self.generate_example_for_function("torsh_sgd_create"))
            .with_note(
                "Learning rate should be positive and typically between 0.001 and 0.1".to_string(),
            )
            .with_see_also("torsh_adam_create".to_string())
            .with_version("0.1.0".to_string());
        self.add_entry(entry);
    }

    fn add_utility_functions(&mut self) {
        let get_error = FunctionSignature {
            name: "torsh_get_last_error".to_string(),
            return_type: "c_int".to_string(),
            parameters: vec![
                ("buffer".to_string(), "*mut c_char".to_string()),
                ("buffer_size".to_string(), "c_int".to_string()),
            ],
            description: "Get the last error message".to_string(),
            is_unsafe: true,
        };

        let entry = ApiDocEntry::new(get_error)
            .with_example(self.generate_example_for_function("torsh_get_last_error"))
            .with_note("Buffer should be large enough to hold the error message".to_string())
            .with_see_also("torsh_clear_last_error".to_string())
            .with_version("0.1.0".to_string());
        self.add_entry(entry);
    }

    fn generate_example_for_function(&self, function_name: &str) -> String {
        match self.target_language {
            TargetLanguage::Python => self.generate_python_example(function_name),
            TargetLanguage::Java => self.generate_java_example(function_name),
            TargetLanguage::Go => self.generate_go_example(function_name),
            TargetLanguage::CSharp => self.generate_csharp_example(function_name),
            TargetLanguage::Swift => self.generate_swift_example(function_name),
            TargetLanguage::R => self.generate_r_example(function_name),
            TargetLanguage::Julia => self.generate_julia_example(function_name),
            _ => self.generate_c_example(function_name),
        }
    }

    fn generate_python_example(&self, function_name: &str) -> String {
        match function_name {
            "torsh_tensor_zeros" => r#"```python
import torsh_ffi as torsh

# Create a 2x3 tensor filled with zeros
shape = [2, 3]
tensor = torsh.tensor_zeros(shape)
print(f"Created tensor with shape: {tensor.shape()}")
```"#
                .to_string(),
            "torsh_tensor_add" => r#"```python
import torsh_ffi as torsh

# Create two tensors and add them
a = torsh.tensor_ones([2, 3])
b = torsh.tensor_ones([2, 3])
result = torsh.tensor_add(a, b)
print("Added two tensors successfully")
```"#
                .to_string(),
            "torsh_linear_create" => r#"```python
import torsh_ffi as torsh

# Create a linear layer with 10 input features and 5 output features
layer = torsh.linear_create(in_features=10, out_features=5, bias=True)
print("Created linear layer")
```"#
                .to_string(),
            _ => "```python\n# Example not available\n```".to_string(),
        }
    }

    fn generate_java_example(&self, function_name: &str) -> String {
        match function_name {
            "torsh_tensor_zeros" => r#"```java
import com.torsh.ffi.TorshBindings;

// Create a 2x3 tensor filled with zeros
int[] shape = {2, 3};
TensorHandle tensor = TorshBindings.tensorZeros(shape);
System.out.println("Created tensor with zeros");
```"#
                .to_string(),
            "torsh_tensor_add" => r#"```java
import com.torsh.ffi.TorshBindings;

// Create two tensors and add them
int[] shape = {2, 3};
TensorHandle a = TorshBindings.tensorOnes(shape);
TensorHandle b = TorshBindings.tensorOnes(shape);
TensorHandle result = TorshBindings.tensorAdd(a, b);
System.out.println("Added two tensors successfully");
```"#
                .to_string(),
            _ => "```java\n// Example not available\n```".to_string(),
        }
    }

    fn generate_go_example(&self, function_name: &str) -> String {
        match function_name {
            "torsh_tensor_zeros" => r#"```go
package main

import (
    "fmt"
    "github.com/torsh/go-bindings/torsh"
)

func main() {
    // Create a 2x3 tensor filled with zeros
    shape := []int32{2, 3}
    tensor := torsh.TensorZeros(shape)
    fmt.Println("Created tensor with zeros")
}
```"#
                .to_string(),
            _ => "```go\n// Example not available\n```".to_string(),
        }
    }

    fn generate_csharp_example(&self, function_name: &str) -> String {
        match function_name {
            "torsh_tensor_zeros" => r#"```csharp
using TorshBindings;

// Create a 2x3 tensor filled with zeros
int[] shape = {2, 3};
IntPtr tensor = TorshAPI.TensorZeros(shape, shape.Length);
Console.WriteLine("Created tensor with zeros");
```"#
                .to_string(),
            _ => "```csharp\n// Example not available\n```".to_string(),
        }
    }

    fn generate_swift_example(&self, function_name: &str) -> String {
        match function_name {
            "torsh_tensor_zeros" => r#"```swift
import TorshBindings

// Create a 2x3 tensor filled with zeros
let shape: [Int32] = [2, 3]
let tensor = tensorZeros(shape: shape)
print("Created tensor with zeros")
```"#
                .to_string(),
            _ => "```swift\n// Example not available\n```".to_string(),
        }
    }

    fn generate_r_example(&self, function_name: &str) -> String {
        match function_name {
            "torsh_tensor_zeros" => r#"```r
library(torsh)

# Create a 2x3 tensor filled with zeros
shape <- c(2L, 3L)
tensor <- r_tensor_zeros(shape)
cat("Created tensor with zeros\n")
```"#
                .to_string(),
            _ => "```r\n# Example not available\n```".to_string(),
        }
    }

    fn generate_julia_example(&self, function_name: &str) -> String {
        match function_name {
            "torsh_tensor_zeros" => r#"```julia
using TorshBindings

# Create a 2x3 tensor filled with zeros
shape = Int32[2, 3]
tensor = jl_tensor_zeros(shape)
println("Created tensor with zeros")
```"#
                .to_string(),
            _ => "```julia\n# Example not available\n```".to_string(),
        }
    }

    fn generate_c_example(&self, function_name: &str) -> String {
        match function_name {
            "torsh_tensor_zeros" => r#"```c
#include "torsh_ffi.h"

// Create a 2x3 tensor filled with zeros
int shape[] = {2, 3};
TorshTensor* tensor = torsh_tensor_zeros(shape, 2);
printf("Created tensor with zeros\n");
```"#
                .to_string(),
            _ => "```c\n// Example not available\n```".to_string(),
        }
    }

    pub fn generate_documentation(&self) -> FfiResult<String> {
        match self.format {
            DocFormat::Markdown => self.generate_markdown(),
            DocFormat::Html => self.generate_html(),
            DocFormat::RestructuredText => self.generate_rst(),
            DocFormat::Sphinx => self.generate_sphinx(),
            DocFormat::Javadoc => self.generate_javadoc(),
            _ => self.generate_markdown(), // Default to markdown
        }
    }

    fn generate_markdown(&self) -> FfiResult<String> {
        let mut output = String::new();

        // Header
        writeln!(
            output,
            "# ToRSh {} API Documentation",
            format!("{:?}", self.target_language)
        )?;
        writeln!(output)?;

        // Metadata
        if let Some(version) = self.metadata.get("version") {
            writeln!(output, "**Version:** {}", version)?;
        }
        if let Some(generated) = self.metadata.get("generated_at") {
            writeln!(output, "**Generated:** {}", generated)?;
        }
        writeln!(output)?;

        // Table of Contents
        writeln!(output, "## Table of Contents")?;
        writeln!(output)?;

        let mut categories: HashMap<FunctionCategory, Vec<&ApiDocEntry>> = HashMap::new();
        for entry in &self.entries {
            categories
                .entry(entry.category.clone())
                .or_default()
                .push(entry);
        }

        for (category, _) in &categories {
            writeln!(
                output,
                "- [{}](#{:?})",
                category.description(),
                format!("{:?}", category).to_lowercase().replace(' ', "-")
            )?;
        }
        writeln!(output)?;

        // Function documentation by category
        for (category, entries) in categories {
            writeln!(output, "## {}", category.description())?;
            writeln!(output)?;

            for entry in entries {
                self.write_function_markdown(&mut output, entry)?;
            }
        }

        Ok(output)
    }

    fn write_function_markdown(&self, output: &mut String, entry: &ApiDocEntry) -> FfiResult<()> {
        let func = &entry.function;

        // Function signature
        writeln!(output, "### `{}`", func.name)?;
        writeln!(output)?;
        writeln!(output, "{}", func.description)?;
        writeln!(output)?;

        // Parameters
        if !func.parameters.is_empty() {
            writeln!(output, "**Parameters:**")?;
            writeln!(output)?;
            for (name, param_type) in &func.parameters {
                writeln!(
                    output,
                    "- `{}`: {} - Parameter description",
                    name, param_type
                )?;
            }
            writeln!(output)?;
        }

        // Return value
        writeln!(output, "**Returns:** `{}`", func.return_type)?;
        writeln!(output)?;

        // Examples
        if !entry.examples.is_empty() {
            writeln!(output, "**Example:**")?;
            writeln!(output)?;
            for example in &entry.examples {
                writeln!(output, "{}", example)?;
                writeln!(output)?;
            }
        }

        // Notes
        if !entry.notes.is_empty() {
            writeln!(output, "**Notes:**")?;
            writeln!(output)?;
            for note in &entry.notes {
                writeln!(output, "- {}", note)?;
            }
            writeln!(output)?;
        }

        // See also
        if !entry.see_also.is_empty() {
            writeln!(output, "**See also:** {}", entry.see_also.join(", "))?;
            writeln!(output)?;
        }

        // Version
        if let Some(version) = &entry.since_version {
            writeln!(output, "**Since version:** {}", version)?;
            writeln!(output)?;
        }

        writeln!(output, "---")?;
        writeln!(output)?;

        Ok(())
    }

    fn generate_html(&self) -> FfiResult<String> {
        let mut output = String::new();

        writeln!(output, "<!DOCTYPE html>")?;
        writeln!(output, "<html>")?;
        writeln!(output, "<head>")?;
        writeln!(
            output,
            "    <title>ToRSh {} API Documentation</title>",
            format!("{:?}", self.target_language)
        )?;
        writeln!(output, "    <style>")?;
        writeln!(
            output,
            "        body {{ font-family: Arial, sans-serif; margin: 40px; }}"
        )?;
        writeln!(
            output,
            "        .function {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}"
        )?;
        writeln!(
            output,
            "        .signature {{ font-family: monospace; background: #f5f5f5; padding: 10px; }}"
        )?;
        writeln!(
            output,
            "        .example {{ background: #f9f9f9; padding: 10px; margin: 10px 0; }}"
        )?;
        writeln!(output, "    </style>")?;
        writeln!(output, "</head>")?;
        writeln!(output, "<body>")?;

        writeln!(
            output,
            "<h1>ToRSh {} API Documentation</h1>",
            format!("{:?}", self.target_language)
        )?;

        // Group by category
        let mut categories: HashMap<FunctionCategory, Vec<&ApiDocEntry>> = HashMap::new();
        for entry in &self.entries {
            categories
                .entry(entry.category.clone())
                .or_default()
                .push(entry);
        }

        for (category, entries) in categories {
            writeln!(output, "<h2>{}</h2>", category.description())?;

            for entry in entries {
                writeln!(output, "<div class=\"function\">")?;
                writeln!(output, "<h3>{}</h3>", entry.function.name)?;
                writeln!(output, "<p>{}</p>", entry.function.description)?;

                if !entry.examples.is_empty() {
                    writeln!(output, "<div class=\"example\">")?;
                    writeln!(output, "<h4>Example:</h4>")?;
                    for example in &entry.examples {
                        writeln!(output, "<pre>{}</pre>", example)?;
                    }
                    writeln!(output, "</div>")?;
                }

                writeln!(output, "</div>")?;
            }
        }

        writeln!(output, "</body>")?;
        writeln!(output, "</html>")?;

        Ok(output)
    }

    fn generate_rst(&self) -> FfiResult<String> {
        let mut output = String::new();

        writeln!(
            output,
            "ToRSh {} API Documentation",
            format!("{:?}", self.target_language)
        )?;
        writeln!(output, "{}", "=".repeat(50))?;
        writeln!(output)?;

        // Group by category and generate RST
        let mut categories: HashMap<FunctionCategory, Vec<&ApiDocEntry>> = HashMap::new();
        for entry in &self.entries {
            categories
                .entry(entry.category.clone())
                .or_default()
                .push(entry);
        }

        for (category, entries) in categories {
            writeln!(output, "{}", category.description())?;
            writeln!(output, "{}", "-".repeat(category.description().len()))?;
            writeln!(output)?;

            for entry in entries {
                writeln!(output, "{}", entry.function.name)?;
                writeln!(output, "{}", "~".repeat(entry.function.name.len()))?;
                writeln!(output)?;
                writeln!(output, "{}", entry.function.description)?;
                writeln!(output)?;

                if !entry.examples.is_empty() {
                    writeln!(output, ".. code-block::")?;
                    writeln!(output)?;
                    for example in &entry.examples {
                        for line in example.lines() {
                            writeln!(output, "   {}", line)?;
                        }
                    }
                    writeln!(output)?;
                }
            }
        }

        Ok(output)
    }

    fn generate_sphinx(&self) -> FfiResult<String> {
        // Similar to RST but with Sphinx-specific directives
        self.generate_rst()
    }

    fn generate_javadoc(&self) -> FfiResult<String> {
        let mut output = String::new();

        writeln!(output, "/**")?;
        writeln!(output, " * ToRSh Java API Documentation")?;
        writeln!(
            output,
            " * Auto-generated documentation for ToRSh Java bindings"
        )?;
        writeln!(output, " */")?;

        // Generate Javadoc-style documentation
        for entry in &self.entries {
            writeln!(output, "/**")?;
            writeln!(output, " * {}", entry.function.description)?;
            writeln!(output, " *")?;

            for (name, _) in &entry.function.parameters {
                writeln!(output, " * @param {} parameter description", name)?;
            }

            writeln!(output, " * @return {}", entry.function.return_type)?;

            if let Some(version) = &entry.since_version {
                writeln!(output, " * @since {}", version)?;
            }

            writeln!(output, " */")?;
            writeln!(output)?;
        }

        Ok(output)
    }
}

/// Generate API documentation for a specific language and format
pub fn generate_api_docs(target_language: TargetLanguage, format: DocFormat) -> FfiResult<String> {
    let mut generator = ApiDocGenerator::new(target_language).with_format(format);

    // Add metadata
    generator.add_metadata("version".to_string(), "0.1.0-alpha.2".to_string());
    generator.add_metadata("generated_at".to_string(), chrono::Utc::now().to_rfc3339());

    // Load standard functions
    generator.load_standard_functions();

    generator.generate_documentation()
}

/// Generate documentation for all supported languages
pub fn generate_all_api_docs() -> FfiResult<HashMap<TargetLanguage, String>> {
    let languages = vec![
        TargetLanguage::Python,
        TargetLanguage::Java,
        TargetLanguage::Go,
        TargetLanguage::CSharp,
        TargetLanguage::Swift,
        TargetLanguage::R,
        TargetLanguage::Julia,
    ];

    let mut results = HashMap::new();

    for lang in languages {
        let docs = generate_api_docs(lang.clone(), DocFormat::for_language(&lang))?;
        results.insert(lang, docs);
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_doc_format_for_language() {
        assert_eq!(
            DocFormat::for_language(&TargetLanguage::Java),
            DocFormat::Javadoc
        );
        assert_eq!(
            DocFormat::for_language(&TargetLanguage::Python),
            DocFormat::Sphinx
        );
        assert_eq!(
            DocFormat::for_language(&TargetLanguage::Go),
            DocFormat::GoDoc
        );
    }

    #[test]
    fn test_function_category_from_name() {
        assert_eq!(
            FunctionCategory::from_function_name("torsh_tensor_zeros"),
            FunctionCategory::TensorCreation
        );
        assert_eq!(
            FunctionCategory::from_function_name("torsh_tensor_add"),
            FunctionCategory::TensorOperations
        );
        assert_eq!(
            FunctionCategory::from_function_name("torsh_linear_create"),
            FunctionCategory::NeuralNetworks
        );
        assert_eq!(
            FunctionCategory::from_function_name("torsh_sgd_create"),
            FunctionCategory::Optimization
        );
    }

    #[test]
    fn test_api_doc_entry_creation() {
        let func = FunctionSignature {
            name: "test_function".to_string(),
            return_type: "void".to_string(),
            parameters: vec![],
            description: "Test function".to_string(),
            is_unsafe: false,
        };

        let entry = ApiDocEntry::new(func)
            .with_example("Example code".to_string())
            .with_note("Important note".to_string())
            .with_version("1.0.0".to_string());

        assert_eq!(entry.examples.len(), 1);
        assert_eq!(entry.notes.len(), 1);
        assert_eq!(entry.since_version, Some("1.0.0".to_string()));
    }

    #[test]
    fn test_doc_generator_creation() {
        let generator = ApiDocGenerator::new(TargetLanguage::Python);
        assert_eq!(generator.target_language, TargetLanguage::Python);
        assert_eq!(generator.format, DocFormat::Sphinx);
    }

    #[test]
    fn test_markdown_generation() {
        let mut generator =
            ApiDocGenerator::new(TargetLanguage::Python).with_format(DocFormat::Markdown);
        generator.load_standard_functions();

        let docs = generator.generate_documentation();
        assert!(docs.is_ok());

        let content = docs.unwrap();
        assert!(content.contains("# ToRSh Python API Documentation"));
        assert!(content.contains("## Table of Contents"));
    }
}
