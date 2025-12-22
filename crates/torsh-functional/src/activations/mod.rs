//! # Activation Functions for Neural Networks
//!
//! This module provides a comprehensive collection of activation functions organized into
//! focused sub-modules for better maintainability and discoverability.
//!
//! ## Mathematical Foundation
//!
//! Activation functions introduce **non-linearity** into neural networks, enabling them to
//! learn complex patterns beyond linear transformations. Without activation functions, a
//! multi-layer network would be equivalent to a single-layer linear transformation.
//!
//! ### Role in Neural Networks
//! ```text
//! Layer output: y = σ(Wx + b)
//! ```
//! where:
//! - `W` is the weight matrix
//! - `x` is the input
//! - `b` is the bias
//! - `σ` is the activation function
//!
//! ### Key Properties
//!
//! #### Non-linearity
//! - **Essential**: Linear compositions remain linear: f(g(x)) = linear
//! - **Non-linear**: Enables learning of complex decision boundaries
//!
//! #### Differentiability
//! - Required for backpropagation and gradient descent
//! - Sub-differentiable at isolated points (e.g., ReLU at 0) is acceptable
//!
//! #### Range and Saturation
//! - **Unbounded** (ReLU): Can cause exploding activations
//! - **Bounded** (Sigmoid, Tanh): May cause vanishing gradients in deep networks
//!
//! #### Zero-Centered
//! - **Zero-centered** (Tanh): Gradients can be positive or negative, faster convergence
//! - **Not zero-centered** (ReLU, Sigmoid): Can cause zig-zagging dynamics
//!
//! ## Activation Function Families
//!
//! ### ReLU Family (Piecewise Linear)
//! ```text
//! ReLU(x) = max(0, x)
//! LeakyReLU(x) = max(αx, x)  where α ∈ (0, 1)
//! ELU(x) = { x if x > 0, α(exp(x) - 1) if x ≤ 0 }
//! ```
//! **Best for**: Hidden layers in deep networks, CNNs
//! **Advantages**: Computationally efficient, sparse activation, no vanishing gradient for x > 0
//! **Disadvantages**: "Dying ReLU" problem, not zero-centered
//!
//! ### Sigmoid Family (Smooth Bounded)
//! ```text
//! Sigmoid(x) = 1 / (1 + exp(-x))  ∈ (0, 1)
//! SiLU(x) = x · Sigmoid(x)  (Swish)
//! ```
//! **Best for**: Binary classification (output layer), gates in LSTMs
//! **Advantages**: Smooth, interpretable as probability
//! **Disadvantages**: Vanishing gradient problem, not zero-centered, expensive exp()
//!
//! ### Tanh Family (Zero-Centered Bounded)
//! ```text
//! Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))  ∈ (-1, 1)
//! ```
//! **Best for**: Hidden layers when zero-centered output desired, RNNs
//! **Advantages**: Zero-centered, stronger gradients than sigmoid
//! **Disadvantages**: Still suffers from vanishing gradient, expensive computation
//!
//! ### Softmax Family (Normalization)
//! ```text
//! Softmax(x_i) = exp(x_i) / Σ_j exp(x_j)
//! ```
//! **Best for**: Multi-class classification (output layer)
//! **Advantages**: Probabilistic interpretation, differentiable
//! **Disadvantages**: Only for output layer, sensitive to outliers
//!
//! ### Advanced Functions (Modern)
//! ```text
//! GELU(x) = x · Φ(x)  where Φ is Gaussian CDF
//! Mish(x) = x · tanh(softplus(x))
//! ```
//! **Best for**: Transformers (GELU), general deep learning (Mish)
//! **Advantages**: Smooth, non-monotonic, state-of-the-art performance
//! **Disadvantages**: More expensive computation
//!
//! ## Performance Characteristics
//!
//! ### Computational Complexity (per element)
//! - **ReLU family**: O(1) - simple comparison/multiplication
//! - **Sigmoid/Tanh**: O(1) but requires exp() - ~10-100x slower than ReLU
//! - **Softmax**: O(n) where n is the number of classes - reduction operation
//! - **GELU/Mish**: O(1) but more complex than ReLU - ~2-5x slower
//!
//! ### Memory Usage
//! - **Standard operations**: O(n) - same as input
//! - **In-place operations**: O(1) - modify input directly
//! - **Softmax**: O(n) - requires temporary storage for normalization
//!
//! ### Gradient Computation
//! - **ReLU**: Fastest - binary gradient (0 or 1)
//! - **Sigmoid/Tanh**: Moderate - requires output value
//! - **Softmax**: Expensive - requires full Jacobian for multi-dimensional
//!
//! ## Choosing the Right Activation
//!
//! ### Decision Tree
//! 1. **Output Layer?**
//!    - Binary classification → **Sigmoid**
//!    - Multi-class classification → **Softmax**
//!    - Regression → **None** or **ReLU** (for non-negative)
//!
//! 2. **Hidden Layer in CNN?**
//!    - Default → **ReLU**
//!    - Want smoothness → **GELU**
//!    - Concerned about dying ReLU → **Leaky ReLU** or **ELU**
//!
//! 3. **Hidden Layer in Transformer?**
//!    - Modern standard → **GELU**
//!    - Alternative → **SiLU/Swish**
//!
//! 4. **Hidden Layer in RNN/LSTM?**
//!    - Gates → **Sigmoid** (by design)
//!    - Hidden state → **Tanh** (by design)
//!
//! 5. **Memory Constrained?**
//!    - Use **in-place variants** (relu_, sigmoid_, etc.)
//!
//! ## Common Use Cases
//!
//! ### Convolutional Neural Network
//! ```rust,no_run
//! # use torsh_functional::activations::{relu, softmax};
//! # use torsh_functional::random_ops::randn;
//! # fn example() -> torsh_core::Result<()> {
//! // Conv → ReLU → Pool → Conv → ReLU → Pool → FC → Softmax
//! let conv1_out = randn(&[32, 64, 28, 28], None, None, None)?;
//! let relu1_out = relu(&conv1_out, false)?;
//!
//! // ... more layers ...
//!
//! let logits = randn(&[32, 10], None, None, None)?;
//! let predictions = softmax(&logits, 1, None)?;
//! # Ok(())
//! # }
//! ```
//!
//! ### Transformer Architecture
//! ```rust,no_run
//! # use torsh_functional::activations::gelu;
//! # use torsh_functional::random_ops::randn;
//! # fn example() -> torsh_core::Result<()> {
//! // Attention → LayerNorm → FFN(GELU) → LayerNorm
//! let ffn_input = randn(&[32, 512, 768], None, None, None)?;
//! let weights1 = randn(&[768, 3072], None, None, None)?;
//! let hidden = ffn_input.matmul(&weights1)?;
//! let activated = gelu(&hidden)?;
//! let weights2 = randn(&[3072, 768], None, None, None)?;
//! let output = activated.matmul(&weights2)?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Organization
//!
//! The activation functions are organized into the following sub-modules:
//!
//! - [`relu_family`]: ReLU and related variants (ReLU, Leaky ReLU, ELU, SELU, etc.)
//! - [`sigmoid_family`]: Sigmoid and related variants (Sigmoid, Hard Sigmoid, SiLU, etc.)
//! - [`tanh_family`]: Tanh and related variants (Tanh, Softsign, Hardtanh, etc.)
//! - [`softmax_family`]: Softmax and related variants (Softmax, Log Softmax, Softmin, etc.)
//! - [`advanced`]: Advanced functions (GELU, GLU, Scaled Dot-Product Attention, etc.)
//! - [`inplace`]: In-place variants for memory-efficient operations
//!
//! ## Quick Reference
//!
//! | Function | Range | Zero-Centered | Computation | Best For |
//! |----------|-------|---------------|-------------|----------|
//! | ReLU | [0, ∞) | No | Fast | CNNs, hidden layers |
//! | Leaky ReLU | (-∞, ∞) | No | Fast | Avoid dying ReLU |
//! | ELU | (-α, ∞) | Nearly | Moderate | Smoother gradients |
//! | Sigmoid | (0, 1) | No | Slow | Binary output |
//! | Tanh | (-1, 1) | Yes | Slow | RNN hidden layers |
//! | Softmax | (0, 1), sum=1 | No | Moderate | Multi-class output |
//! | GELU | (-∞, ∞) | Nearly | Moderate | Transformers |
//! | SiLU/Swish | (-∞, ∞) | No | Moderate | General deep learning |

// Sub-modules
pub mod advanced;
pub mod inplace;
pub mod relu_family;
pub mod sigmoid_family;
pub mod softmax_family;
pub mod tanh_family;

// Helper functions for reducing code duplication
use torsh_core::dtype::FloatElement;
use torsh_core::Result as TorshResult;
use torsh_tensor::Tensor;

/// Generic element-wise activation function helper
///
/// This function eliminates code duplication across activation functions by providing
/// a common pattern for element-wise transformations.
///
/// # Parameters
/// - `input`: Input tensor
/// - `operation`: Closure that transforms each element
///
/// # Returns
/// New tensor with the same shape and device as input, with transformed elements
pub fn apply_elementwise<T, F>(input: &Tensor<T>, operation: F) -> TorshResult<Tensor<T>>
where
    T: FloatElement + Copy,
    F: Fn(T) -> T,
{
    let data = input.data()?;
    let result_data: Vec<T> = data.iter().map(|&x| operation(x)).collect();
    Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())
}

/// Generic element-wise activation function with inplace support
///
/// This function provides a common pattern for activation functions that support
/// both inplace and non-inplace operations.
///
/// # Parameters
/// - `input`: Input tensor
/// - `inplace`: Whether to perform operation in-place (currently creates new tensor regardless)
/// - `operation`: Closure that transforms each element
///
/// # Returns
/// New tensor with transformed elements
pub fn apply_elementwise_inplace<T, F>(
    input: &Tensor<T>,
    _inplace: bool,
    operation: F,
) -> TorshResult<Tensor<T>>
where
    T: FloatElement + Copy,
    F: Fn(T) -> T,
{
    // Note: Currently always creates new tensor for simplicity
    // True inplace operations would require mutable tensor interface
    apply_elementwise(input, operation)
}

// Re-export all functions from sub-modules for backward compatibility and convenience

// ReLU family functions
pub use relu_family::{
    celu, elu, hardshrink, leaky_relu, prelu, relu, relu6, rrelu, selu, softshrink, threshold,
};

// Sigmoid family functions
pub use sigmoid_family::{
    hardsigmoid, hardsigmoid_v2, hardswish, log_sigmoid, mish, sigmoid, silu, softplus, swish,
};

// Tanh family functions
pub use tanh_family::{hardtanh, softsign, tanh, tanhshrink};

// Softmax family functions
pub use softmax_family::{gumbel_softmax, log_softmax, softmax, softmin};

// Advanced functions
pub use advanced::{gelu, glu, local_response_norm, scaled_dot_product_attention};

// In-place functions
pub use inplace::{gelu_, leaky_relu_, relu_, sigmoid_, silu_, tanh_};

// All functions are already re-exported above, no need for additional aliases

#[cfg(test)]
mod integration_tests {
    use super::*;
    use torsh_core::device::DeviceType;
    use torsh_tensor::creation::from_vec;

    /// Integration test to ensure all activation functions work together
    #[test]
    fn test_activation_functions_integration() -> torsh_core::Result<()> {
        let device = DeviceType::Cpu;

        // Test input
        let input = from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5], device)?;

        // Test ReLU family
        let _relu_out = relu(&input, false)?;
        let _leaky_relu_out = leaky_relu(&input, 0.1, false)?;
        let _elu_out = elu(&input, 1.0, false)?;
        let _selu_out = selu(&input, false)?;

        // Test sigmoid family
        let _sigmoid_out = sigmoid(&input)?;
        let _silu_out = silu(&input, false)?;
        let _mish_out = mish(&input, false)?;

        // Test tanh family
        let _tanh_out = tanh(&input)?;
        let _softsign_out = softsign(&input)?;
        let _hardtanh_out = hardtanh(&input, -1.0, 1.0)?;

        // Test advanced functions
        let _gelu_out = gelu(&input)?;

        // Test softmax with different input
        let logits = from_vec(vec![1.0, 2.0, 3.0], &[3], device)?;
        let _softmax_out = softmax(&logits, 0, None)?;
        let _log_softmax_out = log_softmax(&logits, 0, None)?;

        Ok(())
    }

    /// Test that all activation functions produce finite, non-NaN values
    #[test]
    fn test_activation_functions_numerical_stability() -> torsh_core::Result<()> {
        let device = DeviceType::Cpu;

        // Test with edge cases
        let extreme_input = from_vec(vec![-100.0, -1e-8, 0.0, 1e-8, 100.0], &[5], device)?;

        // Test functions that should handle extreme values
        let sigmoid_out = sigmoid(&extreme_input)?;
        let sigmoid_data = sigmoid_out.data()?;
        for &val in sigmoid_data.iter() {
            let val: f32 = val;
            assert!(
                val.is_finite() && !val.is_nan(),
                "Sigmoid produced invalid value: {}",
                val
            );
            assert!(
                val >= 0.0 && val <= 1.0,
                "Sigmoid value {} not in [0,1]",
                val
            );
        }

        let tanh_out = tanh(&extreme_input)?;
        let tanh_data = tanh_out.data()?;
        for &val in tanh_data.iter() {
            let val: f32 = val;
            assert!(
                val.is_finite() && !val.is_nan(),
                "Tanh produced invalid value: {}",
                val
            );
            assert!(
                val >= -1.0 && val <= 1.0,
                "Tanh value {} not in [-1,1]",
                val
            );
        }

        let gelu_out = gelu(&extreme_input)?;
        let gelu_data = gelu_out.data()?;
        for &val in gelu_data.iter() {
            let val: f32 = val;
            assert!(
                val.is_finite() && !val.is_nan(),
                "GELU produced invalid value: {}",
                val
            );
        }

        Ok(())
    }

    /// Test in-place operations
    #[test]
    fn test_inplace_operations() -> torsh_core::Result<()> {
        let device = DeviceType::Cpu;

        // Test in-place ReLU
        let mut input = from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5], device)?;
        relu_(&mut input)?;
        let data = input.data()?;

        // Verify ReLU behavior
        assert_eq!(data[0], 0.0); // -2.0 -> 0.0
        assert_eq!(data[1], 0.0); // -1.0 -> 0.0
        assert_eq!(data[2], 0.0); // 0.0 -> 0.0
        assert_eq!(data[3], 1.0); // 1.0 -> 1.0
        assert_eq!(data[4], 2.0); // 2.0 -> 2.0

        // Test in-place sigmoid
        let mut input2 = from_vec(vec![0.0], &[1], device)?;
        sigmoid_(&mut input2)?;
        let data2 = input2.data()?;
        assert!((data2[0] - 0.5_f32).abs() < 1e-6); // sigmoid(0) = 0.5

        Ok(())
    }

    /// Test that different activation families produce expected output ranges
    #[test]
    fn test_activation_output_ranges() -> torsh_core::Result<()> {
        let device = DeviceType::Cpu;
        let input = from_vec(vec![-5.0, -1.0, 0.0, 1.0, 5.0], &[5], device)?;

        // ReLU family - non-negative outputs
        let relu_out = relu(&input, false)?;
        let relu_data = relu_out.data()?;
        for &val in relu_data.iter() {
            assert!(val >= 0.0, "ReLU output {} should be non-negative", val);
        }

        // Sigmoid family - (0, 1) range
        let sigmoid_out = sigmoid(&input)?;
        let sigmoid_data = sigmoid_out.data()?;
        for &val in sigmoid_data.iter() {
            assert!(
                val > 0.0 && val < 1.0,
                "Sigmoid output {} not in (0,1)",
                val
            );
        }

        // Tanh family - (-1, 1) range
        let tanh_out = tanh(&input)?;
        let tanh_data = tanh_out.data()?;
        for &val in tanh_data.iter() {
            assert!(val > -1.0 && val < 1.0, "Tanh output {} not in (-1,1)", val);
        }

        // Softmax - probability distribution
        let logits = from_vec(vec![1.0, 2.0, 3.0], &[3], device)?;
        let softmax_out = softmax(&logits, 0, None)?;
        let softmax_data = softmax_out.data()?;

        // Check probability properties
        let sum: f32 = softmax_data.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Softmax should sum to 1, got {}",
            sum
        );
        for &val in softmax_data.iter() {
            assert!(
                val >= 0.0 && val <= 1.0,
                "Softmax output {} not in [0,1]",
                val
            );
        }

        Ok(())
    }

    /// Test monotonicity properties where applicable
    #[test]
    fn test_activation_monotonicity() -> torsh_core::Result<()> {
        let device = DeviceType::Cpu;
        let input = from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5], device)?;

        // Test monotonic functions (note: SILU/Swish is NOT monotonic, it has a minimum around x ≈ -1.278)
        let monotonic_functions = vec![
            ("relu", relu(&input, false)?),
            ("sigmoid", sigmoid(&input)?),
            ("tanh", tanh(&input)?),
        ];

        for (name, output) in monotonic_functions {
            let data = output.data()?;
            // Check that outputs are non-decreasing (monotonic)
            for i in 1..data.len() {
                assert!(
                    data[i] >= data[i - 1],
                    "{} should be monotonic: {} < {} at indices {}, {}",
                    name,
                    data[i],
                    data[i - 1],
                    i,
                    i - 1
                );
            }
        }

        Ok(())
    }
}
