//! Interactive Tensor Operations Notebook
//!
//! This example serves as an interactive notebook for learning ToRSh tensor operations.
//! It demonstrates basic tensor creation, manipulation, and operations in a step-by-step manner.

use std::result::Result as StdResult;
use torsh::prelude::*;

/// Interactive notebook for tensor operations
pub struct TensorNotebook {
    examples: Vec<NotebookExample>,
    current_example: usize,
}

/// A single notebook example with code and explanation
pub struct NotebookExample {
    title: String,
    description: String,
    code: Box<dyn Fn() -> StdResult<(), TorshError>>,
    expected_output: String,
}

impl TensorNotebook {
    /// Create a new interactive tensor notebook
    pub fn new() -> Self {
        let mut notebook = Self {
            examples: Vec::new(),
            current_example: 0,
        };

        notebook.add_basic_examples();
        notebook.add_arithmetic_examples();
        notebook.add_reshaping_examples();
        notebook.add_advanced_examples();

        notebook
    }

    /// Run all examples in the notebook
    pub fn run_all(&self) -> StdResult<(), TorshError> {
        println!("🚀 ToRSh Interactive Tensor Notebook");
        println!("=====================================\n");

        for (i, example) in self.examples.iter().enumerate() {
            println!("📝 Example {}: {}", i + 1, example.title);
            println!("{}", example.description);
            println!("\n💻 Running code...");

            match (example.code)() {
                Ok(()) => {
                    println!("✅ Example completed successfully!");
                    println!("📄 Expected output: {}\n", example.expected_output);
                }
                Err(e) => {
                    println!("❌ Example failed: {:?}\n", e);
                }
            }

            println!("---\n");
        }

        Ok(())
    }

    /// Run a specific example by index
    pub fn run_example(&self, index: usize) -> StdResult<(), TorshError> {
        if let Some(example) = self.examples.get(index) {
            println!("📝 {}", example.title);
            println!("{}", example.description);
            (example.code)()?;
            println!("📄 Expected output: {}", example.expected_output);
            Ok(())
        } else {
            Err(TorshError::InvalidArgument(format!(
                "Example {} not found",
                index
            )))
        }
    }

    /// Get the number of examples
    pub fn len(&self) -> usize {
        self.examples.len()
    }

    /// Add basic tensor creation examples
    fn add_basic_examples(&mut self) {
        // Example 1: Basic tensor creation
        self.examples.push(NotebookExample {
            title: "Basic Tensor Creation".to_string(),
            description: "Learn how to create tensors with different initialization methods."
                .to_string(),
            code: Box::new(|| {
                println!("Creating tensors with different methods:");

                // Create from data
                let data = vec![1.0f32, 2.0, 3.0, 4.0];
                let tensor1 = Tensor::from_data(data, vec![2, 2], DeviceType::Cpu)?;
                println!("From data: {:?}", tensor1);

                // Create zeros tensor
                let zeros_tensor = zeros::<f32>(&[3, 3])?;
                println!("Zeros tensor: {:?}", zeros_tensor);

                // Create ones tensor
                let ones_tensor = ones::<f32>(&[2, 3])?;
                println!("Ones tensor: {:?}", ones_tensor);

                // Create random tensor
                let rand_tensor = randn::<f32>(&[2, 2])?;
                println!("Random tensor: {:?}", rand_tensor);

                Ok(())
            }),
            expected_output: "Various tensor creation methods demonstrated".to_string(),
        });

        // Example 2: Tensor properties
        self.examples.push(NotebookExample {
            title: "Tensor Properties and Information".to_string(),
            description: "Explore tensor properties like shape, dtype, and device.".to_string(),
            code: Box::new(|| {
                let tensor = randn::<f32>(&[3, 4, 2])?;

                println!("Tensor properties:");
                println!("Shape: {:?}", tensor.shape());
                println!("Device: {:?}", tensor.device());
                println!("Number of elements: {}", tensor.numel());
                println!("Number of dimensions: {}", tensor.ndim());

                Ok(())
            }),
            expected_output: "Tensor shape, device, and dimension information".to_string(),
        });
    }

    /// Add arithmetic operation examples
    fn add_arithmetic_examples(&mut self) {
        // Example 3: Basic arithmetic
        self.examples.push(NotebookExample {
            title: "Basic Arithmetic Operations".to_string(),
            description: "Perform element-wise arithmetic operations between tensors.".to_string(),
            code: Box::new(|| {
                let a =
                    Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2], DeviceType::Cpu)?;
                let b =
                    Tensor::from_data(vec![2.0f32, 3.0, 4.0, 5.0], vec![2, 2], DeviceType::Cpu)?;

                println!("Tensor A: {:?}", a);
                println!("Tensor B: {:?}", b);

                let sum = a.add(&b)?;
                println!("A + B = {:?}", sum);

                let diff = a.sub(&b)?;
                println!("A - B = {:?}", diff);

                let product = a.mul(&b)?;
                println!("A * B = {:?}", product);

                let quotient = a.div(&b)?;
                println!("A / B = {:?}", quotient);

                Ok(())
            }),
            expected_output: "Element-wise arithmetic results".to_string(),
        });

        // Example 4: Broadcasting
        self.examples.push(NotebookExample {
            title: "Broadcasting Operations".to_string(),
            description: "Understand how broadcasting works with different tensor shapes."
                .to_string(),
            code: Box::new(|| {
                let large = randn::<f32>(&[3, 4])?;
                let small = randn::<f32>(&[1, 4])?;
                let scalar = Tensor::from_data(vec![2.0f32], vec![], DeviceType::Cpu)?;

                println!("Large tensor shape: {:?}", large.shape());
                println!("Small tensor shape: {:?}", small.shape());
                println!("Scalar shape: {:?}", scalar.shape());

                let broadcast_result = large.add(&small)?;
                println!(
                    "Broadcast addition result shape: {:?}",
                    broadcast_result.shape()
                );

                let scalar_result = large.mul(&scalar)?;
                println!(
                    "Scalar multiplication result shape: {:?}",
                    scalar_result.shape()
                );

                Ok(())
            }),
            expected_output: "Broadcasting operations with different shapes".to_string(),
        });
    }

    /// Add reshaping and manipulation examples
    fn add_reshaping_examples(&mut self) {
        // Example 5: Reshaping and views
        self.examples.push(NotebookExample {
            title: "Tensor Reshaping and Views".to_string(),
            description: "Learn how to reshape tensors and create views.".to_string(),
            code: Box::new(|| {
                let original = randn::<f32>(&[2, 6])?;
                println!("Original shape: {:?}", original.shape());

                let reshaped = original.reshape(&[3, 4])?;
                println!("Reshaped to [3, 4]: {:?}", reshaped.shape());

                let flattened = original.reshape(&[12])?;
                println!("Flattened to [12]: {:?}", flattened.shape());

                let expanded = original.unsqueeze(0)?;
                println!("Unsqueezed at dim 0: {:?}", expanded.shape());

                let squeezed = expanded.squeeze(0)?;
                println!("Squeezed at dim 0: {:?}", squeezed.shape());

                Ok(())
            }),
            expected_output: "Various tensor shape transformations".to_string(),
        });

        // Example 6: Indexing and slicing
        self.examples.push(NotebookExample {
            title: "Tensor Indexing and Slicing".to_string(),
            description: "Access and modify tensor elements and sub-tensors.".to_string(),
            code: Box::new(|| {
                let tensor = Tensor::from_data(
                    (0..12).map(|x| x as f32).collect::<Vec<f32>>(),
                    vec![3, 4],
                    DeviceType::Cpu,
                )?;

                println!("Original tensor: {:?}", tensor);

                // Note: Actual indexing implementation would depend on the tensor API
                println!("Tensor indexing operations would be demonstrated here");
                println!("This includes accessing rows, columns, and sub-tensors");

                Ok(())
            }),
            expected_output: "Tensor indexing and slicing demonstrations".to_string(),
        });
    }

    /// Add advanced operation examples
    fn add_advanced_examples(&mut self) {
        // Example 7: Reduction operations
        self.examples.push(NotebookExample {
            title: "Reduction Operations".to_string(),
            description: "Perform reduction operations like sum, mean, max, min across dimensions."
                .to_string(),
            code: Box::new(|| {
                let tensor = randn::<f32>(&[3, 4])?;
                println!("Original tensor: {:?}", tensor);

                let sum_all = tensor.sum()?;
                println!("Sum of all elements: {:?}", sum_all);

                let mean_all = tensor.mean()?;
                println!("Mean of all elements: {:?}", mean_all);

                // Note: Dimension-specific reductions would depend on API
                println!("Dimension-specific reductions would be shown here");

                Ok(())
            }),
            expected_output: "Various reduction operation results".to_string(),
        });

        // Example 8: Linear algebra operations
        self.examples.push(NotebookExample {
            title: "Linear Algebra Operations".to_string(),
            description: "Perform matrix multiplication and other linear algebra operations."
                .to_string(),
            code: Box::new(|| {
                let a = randn::<f32>(&[3, 4])?;
                let b = randn::<f32>(&[4, 2])?;

                println!("Matrix A shape: {:?}", a.shape());
                println!("Matrix B shape: {:?}", b.shape());

                let result = a.matmul(&b)?;
                println!("Matrix multiplication result shape: {:?}", result.shape());

                // Transpose
                let a_t = a.transpose(0, 1)?;
                println!("A transposed shape: {:?}", a_t.shape());

                Ok(())
            }),
            expected_output: "Linear algebra operation results".to_string(),
        });

        // Example 9: Advanced functions
        self.examples.push(NotebookExample {
            title: "Advanced Mathematical Functions".to_string(),
            description: "Apply mathematical functions like exp, log, trigonometric functions."
                .to_string(),
            code: Box::new(|| {
                let tensor =
                    Tensor::from_data(vec![0.0f32, 1.0, 2.0, 3.0], vec![2, 2], DeviceType::Cpu)?;
                println!("Original tensor: {:?}", tensor);

                let exp_result = tensor.exp()?;
                println!("Exponential: {:?}", exp_result);

                let positive_tensor = tensor.abs()?.add_scalar(1.0)?;
                let log_result = positive_tensor.log()?;
                println!("Natural logarithm: {:?}", log_result);

                let sin_result = tensor.sin()?;
                println!("Sine: {:?}", sin_result);

                Ok(())
            }),
            expected_output: "Results of various mathematical functions".to_string(),
        });
    }
}

/// Interactive notebook runner
fn main() -> StdResult<(), TorshError> {
    let notebook = TensorNotebook::new();

    println!("🎓 Welcome to the ToRSh Interactive Tensor Notebook!");
    println!(
        "This notebook contains {} examples to help you learn tensor operations.\n",
        notebook.len()
    );

    // Run all examples
    notebook.run_all()?;

    println!("🎉 Congratulations! You've completed the interactive tensor notebook.");
    println!("Try modifying the examples or creating your own to explore more ToRSh features!");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_notebook_creation() {
        let notebook = TensorNotebook::new();
        assert!(notebook.len() > 0);
    }

    #[test]
    fn test_example_execution() -> StdResult<(), TorshError> {
        let notebook = TensorNotebook::new();

        // Test that we can run individual examples without panicking
        if notebook.len() > 0 {
            // Just test that the function doesn't panic
            let _ = notebook.run_example(0);
        }

        Ok(())
    }
}
