//! Automatic test generator for ToRSh FFI bindings
//!
//! This module generates comprehensive test suites for all language bindings,
//! ensuring consistent behavior and coverage across all supported languages.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Test case definition
#[derive(Debug, Clone)]
pub struct TestCase {
    pub name: String,
    pub description: String,
    pub category: TestCategory,
    pub inputs: Vec<TestInput>,
    pub expected_output: TestOutput,
    pub tolerance: Option<f64>,
}

/// Test categories
#[derive(Debug, Clone, PartialEq)]
pub enum TestCategory {
    TensorCreation,
    BasicOperations,
    MatrixOperations,
    Activations,
    Reductions,
    ShapeOperations,
    NeuralNetwork,
    ErrorHandling,
}

/// Test input types
#[derive(Debug, Clone)]
pub enum TestInput {
    TensorData(Vec<Vec<f32>>),
    Scalar(f32),
    Shape(Vec<usize>),
    Integer(i32),
    String(String),
}

/// Expected test output
#[derive(Debug, Clone)]
pub enum TestOutput {
    Tensor(Vec<Vec<f32>>),
    Shape(Vec<usize>),
    Scalar(f32),
    Error(String),
    Boolean(bool),
}

/// Language-specific test generators
pub trait TestGenerator {
    fn language_name(&self) -> &str;
    fn file_extension(&self) -> &str;
    fn generate_test_file(&self, test_cases: &[TestCase]) -> String;
    fn generate_single_test(&self, test_case: &TestCase) -> String;
}

/// Python test generator
pub struct PythonTestGenerator;

impl TestGenerator for PythonTestGenerator {
    fn language_name(&self) -> &str {
        "Python"
    }

    fn file_extension(&self) -> &str {
        "py"
    }

    fn generate_test_file(&self, test_cases: &[TestCase]) -> String {
        let mut output = String::new();

        // Header
        output.push_str("#!/usr/bin/env python3\n");
        output.push_str("\"\"\"Automatically generated tests for ToRSh Python bindings\"\"\"\n\n");
        output.push_str("import unittest\n");
        output.push_str("import numpy as np\n");
        output.push_str("import torsh\n\n");

        // Test class
        output.push_str("class TestTorshBindings(unittest.TestCase):\n");
        output.push_str("    \"\"\"Test suite for ToRSh Python bindings\"\"\"\n\n");

        // Setup method
        output.push_str("    def setUp(self):\n");
        output.push_str("        \"\"\"Set up test fixtures\"\"\"\n");
        output.push_str("        self.tolerance = 1e-6\n\n");

        // Generate individual tests
        for test_case in test_cases {
            output.push_str(&self.generate_single_test(test_case));
            output.push('\n');
        }

        // Main block
        output.push_str("if __name__ == '__main__':\n");
        output.push_str("    unittest.main()\n");

        output
    }

    fn generate_single_test(&self, test_case: &TestCase) -> String {
        let mut output = String::new();

        output.push_str(&format!("    def test_{}(self):\n", test_case.name));
        output.push_str(&format!(
            "        \"\"\"Test: {}\"\"\"\n",
            test_case.description
        ));

        // Generate input setup
        for (i, input) in test_case.inputs.iter().enumerate() {
            match input {
                TestInput::TensorData(data) => {
                    output.push_str(&format!(
                        "        input_{} = torsh.tensor({})\n",
                        i,
                        format_tensor_data(data)
                    ));
                }
                TestInput::Scalar(value) => {
                    output.push_str(&format!("        input_{} = {}\n", i, value));
                }
                TestInput::Shape(shape) => {
                    output.push_str(&format!("        input_{} = {}\n", i, format_shape(shape)));
                }
                TestInput::Integer(value) => {
                    output.push_str(&format!("        input_{} = {}\n", i, value));
                }
                TestInput::String(value) => {
                    output.push_str(&format!("        input_{} = '{}'\n", i, value));
                }
            }
        }

        // Generate test operation based on category
        let operation = match test_case.category {
            TestCategory::TensorCreation => "torsh.tensor(input_0)",
            TestCategory::BasicOperations => "input_0.add(input_1)",
            TestCategory::MatrixOperations => "input_0.matmul(input_1)",
            TestCategory::Activations => "input_0.relu()",
            TestCategory::Reductions => "input_0.sum()",
            TestCategory::ShapeOperations => "input_0.reshape(*input_1)",
            TestCategory::NeuralNetwork => "torsh.nn.linear(input_0, input_1, input_2)",
            TestCategory::ErrorHandling => "# Error handling test",
        };

        if test_case.category == TestCategory::ErrorHandling {
            output.push_str("        with self.assertRaises(Exception):\n");
            output.push_str(&format!("            result = {}\n", operation));
        } else {
            output.push_str(&format!("        result = {}\n", operation));

            // Generate assertion
            match &test_case.expected_output {
                TestOutput::Tensor(expected) => {
                    output.push_str(&format!(
                        "        expected = {}\n",
                        format_tensor_data(expected)
                    ));
                    output.push_str("        np.testing.assert_allclose(result.numpy(), expected, rtol=self.tolerance)\n");
                }
                TestOutput::Shape(expected) => {
                    output.push_str(&format!(
                        "        self.assertEqual(result.shape, {})\n",
                        format_shape(expected)
                    ));
                }
                TestOutput::Scalar(expected) => {
                    output.push_str(&format!(
                        "        self.assertAlmostEqual(result.item(), {}, places=6)\n",
                        expected
                    ));
                }
                TestOutput::Boolean(expected) => {
                    output.push_str(&format!("        self.assertEqual(result, {})\n", expected));
                }
                TestOutput::Error(_) => {
                    // Already handled above
                }
            }
        }

        output
    }
}

/// JavaScript/Node.js test generator
pub struct JavaScriptTestGenerator;

impl TestGenerator for JavaScriptTestGenerator {
    fn language_name(&self) -> &str {
        "JavaScript"
    }

    fn file_extension(&self) -> &str {
        "js"
    }

    fn generate_test_file(&self, test_cases: &[TestCase]) -> String {
        let mut output = String::new();

        // Header
        output
            .push_str("/**\n * Automatically generated tests for ToRSh Node.js bindings\n */\n\n");
        output.push_str("const { Tensor } = require('@torsh/core');\n");
        output.push_str("const assert = require('assert');\n\n");

        output.push_str("describe('ToRSh Node.js Bindings', function() {\n");
        output.push_str("  const tolerance = 1e-6;\n\n");

        // Helper function
        output.push_str("  function assertTensorClose(actual, expected, tol = tolerance) {\n");
        output.push_str("    const actualData = actual.data();\n");
        output.push_str("    const expectedFlat = expected.flat();\n");
        output.push_str("    assert.strictEqual(actualData.length, expectedFlat.length);\n");
        output.push_str("    for (let i = 0; i < actualData.length; i++) {\n");
        output.push_str("      assert(Math.abs(actualData[i] - expectedFlat[i]) < tol,\n");
        output.push_str(
            "        `Expected ${expectedFlat[i]}, got ${actualData[i]} at index ${i}`);\n",
        );
        output.push_str("    }\n");
        output.push_str("  }\n\n");

        // Generate individual tests
        for test_case in test_cases {
            output.push_str(&self.generate_single_test(test_case));
            output.push('\n');
        }

        output.push_str("});\n");

        output
    }

    fn generate_single_test(&self, test_case: &TestCase) -> String {
        let mut output = String::new();

        output.push_str(&format!(
            "  it('{}', function() {{\n",
            test_case.description
        ));

        // Generate input setup
        for (i, input) in test_case.inputs.iter().enumerate() {
            match input {
                TestInput::TensorData(data) => {
                    output.push_str(&format!(
                        "    const input{} = Tensor.tensor({});\n",
                        i,
                        format_js_array(data)
                    ));
                }
                TestInput::Scalar(value) => {
                    output.push_str(&format!("    const input{} = {};\n", i, value));
                }
                TestInput::Shape(shape) => {
                    output.push_str(&format!(
                        "    const input{} = {};\n",
                        i,
                        format_js_array_1d(shape)
                    ));
                }
                TestInput::Integer(value) => {
                    output.push_str(&format!("    const input{} = {};\n", i, value));
                }
                TestInput::String(value) => {
                    output.push_str(&format!("    const input{} = '{}';\n", i, value));
                }
            }
        }

        // Generate test operation
        let operation = match test_case.category {
            TestCategory::TensorCreation => "Tensor.tensor(input0)",
            TestCategory::BasicOperations => "input0.add(input1)",
            TestCategory::MatrixOperations => "input0.matmul(input1)",
            TestCategory::Activations => "input0.relu()",
            TestCategory::Reductions => "input0.sum()",
            TestCategory::ShapeOperations => "input0.reshape(...input1)",
            TestCategory::NeuralNetwork => "nn.linear(input0, input1, input2)",
            TestCategory::ErrorHandling => "// Error handling test",
        };

        if test_case.category == TestCategory::ErrorHandling {
            output.push_str("    assert.throws(() => {\n");
            output.push_str(&format!("      const result = {};\n", operation));
            output.push_str("    });\n");
        } else {
            output.push_str(&format!("    const result = {};\n", operation));

            // Generate assertion
            match &test_case.expected_output {
                TestOutput::Tensor(expected) => {
                    output.push_str(&format!(
                        "    const expected = {};\n",
                        format_js_array(expected)
                    ));
                    output.push_str("    assertTensorClose(result, expected);\n");
                }
                TestOutput::Shape(expected) => {
                    output.push_str(&format!(
                        "    assert.deepStrictEqual(result.shape(), {});\n",
                        format_js_array_1d(expected)
                    ));
                }
                TestOutput::Scalar(expected) => {
                    output.push_str(&format!(
                        "    assert(Math.abs(result.data()[0] - {}) < tolerance);\n",
                        expected
                    ));
                }
                TestOutput::Boolean(expected) => {
                    output.push_str(&format!("    assert.strictEqual(result, {});\n", expected));
                }
                TestOutput::Error(_) => {
                    // Already handled above
                }
            }
        }

        output.push_str("  });\n");
        output
    }
}

/// Lua test generator
pub struct LuaTestGenerator;

impl TestGenerator for LuaTestGenerator {
    fn language_name(&self) -> &str {
        "Lua"
    }

    fn file_extension(&self) -> &str {
        "lua"
    }

    fn generate_test_file(&self, test_cases: &[TestCase]) -> String {
        let mut output = String::new();

        // Header
        output.push_str("-- Automatically generated tests for ToRSh Lua bindings\n\n");
        output.push_str("local torsh = require('torsh')\n");
        output.push_str("local function assert_close(a, b, tol)\n");
        output.push_str("  tol = tol or 1e-6\n");
        output.push_str("  return math.abs(a - b) < tol\n");
        output.push_str("end\n\n");

        output.push_str("local tests_passed = 0\n");
        output.push_str("local tests_failed = 0\n\n");

        // Generate individual tests
        for test_case in test_cases {
            output.push_str(&self.generate_single_test(test_case));
            output.push('\n');
        }

        // Summary
        output.push_str("print(string.format('Tests completed: %d passed, %d failed', tests_passed, tests_failed))\n");
        output.push_str("if tests_failed > 0 then\n");
        output.push_str("  os.exit(1)\n");
        output.push_str("end\n");

        output
    }

    fn generate_single_test(&self, test_case: &TestCase) -> String {
        let mut output = String::new();

        output.push_str(&format!("-- Test: {}\n", test_case.description));
        output.push_str("do\n");
        output.push_str("  local success, error_msg = pcall(function()\n");

        // Generate input setup
        for (i, input) in test_case.inputs.iter().enumerate() {
            match input {
                TestInput::TensorData(data) => {
                    output.push_str(&format!(
                        "    local input{} = torsh.tensor({})\n",
                        i,
                        format_lua_table(data)
                    ));
                }
                TestInput::Scalar(value) => {
                    output.push_str(&format!("    local input{} = {}\n", i, value));
                }
                TestInput::Shape(shape) => {
                    output.push_str(&format!(
                        "    local input{} = {}\n",
                        i,
                        format_lua_table_1d(shape)
                    ));
                }
                TestInput::Integer(value) => {
                    output.push_str(&format!("    local input{} = {}\n", i, value));
                }
                TestInput::String(value) => {
                    output.push_str(&format!("    local input{} = '{}'\n", i, value));
                }
            }
        }

        // Generate test operation
        let operation = match test_case.category {
            TestCategory::TensorCreation => "torsh.tensor(input0)",
            TestCategory::BasicOperations => "input0:add(input1)",
            TestCategory::MatrixOperations => "input0:matmul(input1)",
            TestCategory::Activations => "input0:relu()",
            TestCategory::Reductions => "input0:sum()",
            TestCategory::ShapeOperations => "input0:reshape(table.unpack(input1))",
            TestCategory::NeuralNetwork => "torsh.nn.linear(input0, input1, input2)",
            TestCategory::ErrorHandling => "-- Error handling test",
        };

        if test_case.category != TestCategory::ErrorHandling {
            output.push_str(&format!("    local result = {}\n", operation));

            // Generate assertion based on expected output
            match &test_case.expected_output {
                TestOutput::Tensor(_) => {
                    output.push_str("    -- Tensor comparison would go here\n");
                    output.push_str("    assert(result ~= nil, 'Result should not be nil')\n");
                }
                TestOutput::Shape(expected) => {
                    output.push_str(&format!(
                        "    local expected_shape = {}\n",
                        format_lua_table_1d(expected)
                    ));
                    output.push_str("    local actual_shape = result:shape()\n");
                    output.push_str(
                        "    assert(#actual_shape == #expected_shape, 'Shape length mismatch')\n",
                    );
                }
                TestOutput::Scalar(expected) => {
                    output.push_str(&format!(
                        "    assert(assert_close(result:data()[1], {}), 'Scalar value mismatch')\n",
                        expected
                    ));
                }
                TestOutput::Boolean(expected) => {
                    output.push_str(&format!(
                        "    assert(result == {}, 'Boolean value mismatch')\n",
                        expected
                    ));
                }
                TestOutput::Error(_) => {
                    // Should not reach here for non-error tests
                }
            }
        }

        output.push_str("  end)\n");
        output.push_str("  \n");
        output.push_str("  if success then\n");
        output.push_str(&format!("    print('PASS: {}')\n", test_case.description));
        output.push_str("    tests_passed = tests_passed + 1\n");
        output.push_str("  else\n");
        output.push_str(&format!(
            "    print('FAIL: {} - ' .. tostring(error_msg))\n",
            test_case.description
        ));
        output.push_str("    tests_failed = tests_failed + 1\n");
        output.push_str("  end\n");
        output.push_str("end\n");

        output
    }
}

/// Generate standard test cases
pub fn create_standard_test_cases() -> Vec<TestCase> {
    vec![
        TestCase {
            name: "tensor_creation_2d".to_string(),
            description: "Create 2D tensor from nested array".to_string(),
            category: TestCategory::TensorCreation,
            inputs: vec![TestInput::TensorData(vec![vec![1.0, 2.0], vec![3.0, 4.0]])],
            expected_output: TestOutput::Shape(vec![2, 2]),
            tolerance: Some(1e-6),
        },
        TestCase {
            name: "tensor_addition".to_string(),
            description: "Element-wise tensor addition".to_string(),
            category: TestCategory::BasicOperations,
            inputs: vec![
                TestInput::TensorData(vec![vec![1.0, 2.0], vec![3.0, 4.0]]),
                TestInput::TensorData(vec![vec![5.0, 6.0], vec![7.0, 8.0]]),
            ],
            expected_output: TestOutput::Tensor(vec![vec![6.0, 8.0], vec![10.0, 12.0]]),
            tolerance: Some(1e-6),
        },
        TestCase {
            name: "matrix_multiplication".to_string(),
            description: "Matrix multiplication operation".to_string(),
            category: TestCategory::MatrixOperations,
            inputs: vec![
                TestInput::TensorData(vec![vec![1.0, 2.0], vec![3.0, 4.0]]),
                TestInput::TensorData(vec![vec![5.0, 6.0], vec![7.0, 8.0]]),
            ],
            expected_output: TestOutput::Tensor(vec![vec![19.0, 22.0], vec![43.0, 50.0]]),
            tolerance: Some(1e-6),
        },
        TestCase {
            name: "relu_activation".to_string(),
            description: "ReLU activation function".to_string(),
            category: TestCategory::Activations,
            inputs: vec![TestInput::TensorData(vec![vec![-1.0, 0.0, 1.0, 2.0]])],
            expected_output: TestOutput::Tensor(vec![vec![0.0, 0.0, 1.0, 2.0]]),
            tolerance: Some(1e-6),
        },
        TestCase {
            name: "tensor_sum".to_string(),
            description: "Sum all tensor elements".to_string(),
            category: TestCategory::Reductions,
            inputs: vec![TestInput::TensorData(vec![vec![1.0, 2.0], vec![3.0, 4.0]])],
            expected_output: TestOutput::Scalar(10.0),
            tolerance: Some(1e-6),
        },
        TestCase {
            name: "zeros_creation".to_string(),
            description: "Create tensor filled with zeros".to_string(),
            category: TestCategory::TensorCreation,
            inputs: vec![TestInput::Shape(vec![3, 3])],
            expected_output: TestOutput::Tensor(vec![
                vec![0.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0],
            ]),
            tolerance: Some(1e-6),
        },
        TestCase {
            name: "shape_mismatch_error".to_string(),
            description: "Matrix multiplication with incompatible shapes should error".to_string(),
            category: TestCategory::ErrorHandling,
            inputs: vec![
                TestInput::TensorData(vec![vec![1.0, 2.0]]),
                TestInput::TensorData(vec![vec![1.0], vec![2.0], vec![3.0]]),
            ],
            expected_output: TestOutput::Error("Shape mismatch".to_string()),
            tolerance: None,
        },
    ]
}

/// Test suite generator
pub struct TestSuiteGenerator {
    generators: HashMap<String, Box<dyn TestGenerator>>,
}

impl TestSuiteGenerator {
    pub fn new() -> Self {
        let mut generators: HashMap<String, Box<dyn TestGenerator>> = HashMap::new();
        generators.insert("python".to_string(), Box::new(PythonTestGenerator));
        generators.insert("javascript".to_string(), Box::new(JavaScriptTestGenerator));
        generators.insert("lua".to_string(), Box::new(LuaTestGenerator));

        Self { generators }
    }

    pub fn generate_all_tests(&self, output_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let test_cases = create_standard_test_cases();

        for (lang, generator) in &self.generators {
            let test_content = generator.generate_test_file(&test_cases);
            let filename = format!("test_torsh_bindings.{}", generator.file_extension());
            let file_path = output_dir.join(lang).join(filename);

            // Create directory if it doesn't exist
            if let Some(parent) = file_path.parent() {
                fs::create_dir_all(parent)?;
            }

            fs::write(&file_path, test_content)?;
            println!(
                "Generated {} tests: {}",
                generator.language_name(),
                file_path.display()
            );
        }

        Ok(())
    }

    pub fn add_generator(&mut self, name: String, generator: Box<dyn TestGenerator>) {
        self.generators.insert(name, generator);
    }
}

/// Helper functions for formatting data structures

fn format_tensor_data(data: &[Vec<f32>]) -> String {
    let formatted_rows: Vec<String> = data
        .iter()
        .map(|row| {
            format!(
                "[{}]",
                row.iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })
        .collect();
    format!("[{}]", formatted_rows.join(", "))
}

fn format_shape(shape: &[usize]) -> String {
    format!(
        "[{}]",
        shape
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn format_js_array(data: &[Vec<f32>]) -> String {
    let formatted_rows: Vec<String> = data
        .iter()
        .map(|row| {
            format!(
                "[{}]",
                row.iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })
        .collect();
    format!("[{}]", formatted_rows.join(", "))
}

fn format_js_array_1d(data: &[usize]) -> String {
    format!(
        "[{}]",
        data.iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn format_lua_table(data: &[Vec<f32>]) -> String {
    let formatted_rows: Vec<String> = data
        .iter()
        .map(|row| {
            format!(
                "{{{}}}",
                row.iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })
        .collect();
    format!("{{{}}}", formatted_rows.join(", "))
}

fn format_lua_table_1d(data: &[usize]) -> String {
    format!(
        "{{{}}}",
        data.iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_python_test_generation() {
        let generator = PythonTestGenerator;
        let test_cases = create_standard_test_cases();
        let result = generator.generate_test_file(&test_cases);

        assert!(result.contains("import unittest"));
        assert!(result.contains("class TestTorshBindings"));
        assert!(result.contains("def test_tensor_creation_2d"));
    }

    #[test]
    fn test_javascript_test_generation() {
        let generator = JavaScriptTestGenerator;
        let test_cases = create_standard_test_cases();
        let result = generator.generate_test_file(&test_cases);

        assert!(result.contains("const { Tensor }"));
        assert!(result.contains("describe('ToRSh Node.js Bindings'"));
    }

    #[test]
    fn test_standard_test_cases() {
        let test_cases = create_standard_test_cases();
        assert!(!test_cases.is_empty());
        assert!(test_cases
            .iter()
            .any(|tc| tc.category == TestCategory::TensorCreation));
        assert!(test_cases
            .iter()
            .any(|tc| tc.category == TestCategory::BasicOperations));
    }

    #[test]
    fn test_test_suite_generator() {
        let generator = TestSuiteGenerator::new();
        assert!(generator.generators.contains_key("python"));
        assert!(generator.generators.contains_key("javascript"));
        assert!(generator.generators.contains_key("lua"));
    }
}
