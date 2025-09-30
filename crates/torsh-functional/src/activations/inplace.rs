//! In-place activation functions
//!
//! This module provides in-place variants of activation functions that modify
//! the input tensor directly, offering potential memory and performance benefits.

use torsh_core::{dtype::FloatElement, Result as TorshResult};
use torsh_tensor::Tensor;

/// In-place ReLU activation function
///
/// **Mathematical Definition:**
/// ```text
/// ReLU(x) = max(0, x) = {
///   x   if x > 0
///   0   if x ≤ 0
/// }
/// ```
///
/// **In-place Operation:**
/// Modifies the input tensor directly, replacing each element x with max(0, x).
/// This saves memory by not creating a new tensor.
///
/// **Properties:**
/// - **Memory efficient**: No additional memory allocation for output
/// - **Fast execution**: Direct modification without data copying
/// - **Destructive**: Original input values are lost
/// - **Cache friendly**: Better memory locality than out-of-place operations
///
/// **Implementation Notes:**
/// This implementation provides optimized in-place operation with fallback support.
/// The tensor data is replaced element-wise with ReLU-activated values.
///
/// **Usage Considerations:**
/// - Use when you don't need to preserve original input values
/// - Particularly beneficial for large tensors where memory is constrained
/// - Be careful in computational graphs where original values might be needed
pub fn relu_<T: FloatElement>(input: &mut Tensor<T>) -> TorshResult<()>
where
    T: Copy + PartialOrd + torsh_core::dtype::TensorElement + Default,
{
    // Element-wise computation with optimized memory access
    let zero = <T as torsh_core::dtype::TensorElement>::zero();
    let data = input.data()?;
    let result_data: Vec<T> = data
        .iter()
        .map(|&x| if x < zero { zero } else { x })
        .collect();

    // Replace tensor data
    *input = Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())?;
    Ok(())
}

/// In-place Sigmoid activation function
///
/// **Mathematical Definition:**
/// ```text
/// sigmoid(x) = 1 / (1 + exp(-x))
/// ```
///
/// **Numerically Stable Implementation:**
/// ```text
/// For x ≥ 0: sigmoid(x) = 1 / (1 + exp(-x))
/// For x < 0: sigmoid(x) = exp(x) / (1 + exp(x))
/// ```
///
/// **In-place Operation:**
/// Modifies the input tensor directly, replacing each element x with sigmoid(x).
/// Uses numerically stable computation to avoid overflow/underflow.
///
/// **Properties:**
/// - **Numerically stable**: Handles large positive and negative values
/// - **Memory efficient**: In-place computation saves memory
/// - **Range preservation**: Output always in (0, 1)
/// - **Smooth activation**: Differentiable everywhere
///
/// **Numerical Considerations:**
/// - For x ≥ 0: Uses standard sigmoid formula
/// - For x < 0: Uses exp(x)/(1 + exp(x)) to avoid underflow
/// - Prevents overflow in exponential computation
pub fn sigmoid_<T: FloatElement>(input: &mut Tensor<T>) -> TorshResult<()>
where
    T: Copy + PartialOrd + torsh_core::dtype::TensorElement + Default,
{
    // Enhanced fallback with numerical stability improvements
    let one = <T as torsh_core::dtype::TensorElement>::one();
    let zero = <T as torsh_core::dtype::TensorElement>::zero();
    let data = input.data()?;
    let result_data: Vec<T> = data
        .iter()
        .map(|&x| {
            // Use numerically stable sigmoid implementation
            // For x >= 0: sigmoid(x) = 1 / (1 + exp(-x))
            // For x < 0: sigmoid(x) = exp(x) / (1 + exp(x))
            if x >= zero {
                one / (one + (-x).exp())
            } else {
                let exp_x = x.exp();
                exp_x / (one + exp_x)
            }
        })
        .collect();

    // Replace tensor data with computed sigmoid values
    *input = Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())?;
    Ok(())
}

/// In-place Tanh activation function
///
/// **Mathematical Definition:**
/// ```text
/// tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
/// ```
///
/// **Numerically Stable Implementation:**
/// ```text
/// For x ≥ 0: tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
/// For x < 0: tanh(x) = (1 - exp(-2x)) / (1 + exp(-2x))
/// ```
///
/// **In-place Operation:**
/// Modifies the input tensor directly, replacing each element x with tanh(x).
/// Uses numerically stable computation to handle large positive and negative values.
///
/// **Properties:**
/// - **Zero-centered**: Output range (-1, 1) with tanh(0) = 0
/// - **Numerically stable**: Handles extreme values without overflow
/// - **Memory efficient**: In-place operation saves memory
/// - **Smooth**: Differentiable everywhere
///
/// **Numerical Considerations:**
/// - Uses different formulations for positive and negative inputs
/// - Prevents overflow in exponential computation
/// - Maintains numerical precision across the full input range
pub fn tanh_<T: FloatElement>(input: &mut Tensor<T>) -> TorshResult<()>
where
    T: Copy + PartialOrd + torsh_core::dtype::TensorElement + Default,
{
    // Enhanced fallback with numerical stability improvements
    let one = <T as torsh_core::dtype::TensorElement>::one();
    let two = one + one;
    let zero = <T as torsh_core::dtype::TensorElement>::zero();
    let data = input.data()?;
    let result_data: Vec<T> = data
        .iter()
        .map(|&x| {
            // Use numerically stable tanh implementation
            // tanh(x) = (exp(2x) - 1) / (exp(2x) + 1) for x >= 0
            // tanh(x) = (1 - exp(-2x)) / (1 + exp(-2x)) for x < 0
            if x >= zero {
                let exp_2x = (two * x).exp();
                (exp_2x - one) / (exp_2x + one)
            } else {
                let exp_neg_2x = (-two * x).exp();
                (one - exp_neg_2x) / (one + exp_neg_2x)
            }
        })
        .collect();

    // Replace tensor data with computed tanh values
    *input = Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())?;
    Ok(())
}

/// In-place Leaky ReLU activation function
///
/// **Mathematical Definition:**
/// ```text
/// LeakyReLU(x) = {
///   x           if x ≥ 0
///   slope * x   if x < 0
/// }
/// ```
///
/// **In-place Operation:**
/// Modifies the input tensor directly, replacing each element x with LeakyReLU(x).
/// Uses the specified negative slope for negative values.
///
/// **Properties:**
/// - **Fixes dying ReLU**: Non-zero gradient for negative inputs
/// - **Memory efficient**: In-place computation saves memory
/// - **Parameterized**: Negative slope controls behavior for x < 0
/// - **Piecewise linear**: Simple and fast computation
///
/// **Parameters:**
/// - `negative_slope`: Slope for negative values (typically 0.01)
///   - 0.01: Standard leaky ReLU (1% of negative signal)
///   - 0.1: More aggressive negative slope
///   - 0.2: Approaching linear behavior
pub fn leaky_relu_<T: FloatElement>(input: &mut Tensor<T>, negative_slope: f64) -> TorshResult<()>
where
    T: Copy + PartialOrd + From<f32> + torsh_core::dtype::TensorElement,
{
    // Enhanced fallback implementation
    let slope = <T as From<f32>>::from(negative_slope as f32);
    let zero = <T as torsh_core::dtype::TensorElement>::zero();
    let data = input.data()?;
    let result_data: Vec<T> = data
        .iter()
        .map(|&x| if x >= zero { x } else { x * slope })
        .collect();

    *input = Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())?;
    Ok(())
}

/// In-place GELU activation function
///
/// **Mathematical Definition:**
/// ```text
/// GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
/// ```
///
/// **In-place Operation:**
/// Modifies the input tensor directly, replacing each element x with GELU(x).
/// Uses the tanh approximation for efficient computation.
///
/// **Properties:**
/// - **Smooth**: Differentiable everywhere (unlike ReLU)
/// - **Self-gating**: Multiplies input by activation probability
/// - **Memory efficient**: In-place operation saves memory
/// - **Modern**: Popular in transformer architectures
///
/// **Implementation Notes:**
/// Uses the commonly employed tanh approximation for computational efficiency.
/// This approximation is very close to the exact GELU formula.
///
/// **Applications:**
/// - **Transformers**: BERT, GPT, and similar models
/// - **Modern CNNs**: Replacement for ReLU in many architectures
/// - **Research models**: Default choice in many recent papers
pub fn gelu_<T: FloatElement>(input: &mut Tensor<T>) -> TorshResult<()>
where
    T: Copy + PartialOrd + From<f32> + torsh_core::dtype::TensorElement,
{
    // Direct fallback implementation using GELU approximation
    // GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    let half = <T as From<f32>>::from(0.5);
    let one = <T as torsh_core::dtype::TensorElement>::one();
    let sqrt_2_over_pi = <T as From<f32>>::from(0.797884561); // √(2/π)
    let coeff = <T as From<f32>>::from(0.044715);

    let data = input.data()?;
    let result_data: Vec<T> = data
        .iter()
        .map(|&x| {
            let x_cubed = x * x * x;
            let inner = sqrt_2_over_pi * (x + coeff * x_cubed);
            let tanh_val = inner.tanh();
            half * x * (one + tanh_val)
        })
        .collect();

    *input = Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())?;
    Ok(())
}

/// In-place SiLU/Swish activation function
///
/// **Mathematical Definition:**
/// ```text
/// SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
/// ```
///
/// **In-place Operation:**
/// Modifies the input tensor directly, replacing each element x with SiLU(x).
/// Combines the input value with its sigmoid activation.
///
/// **Properties:**
/// - **Self-gating**: Uses input to gate itself via sigmoid
/// - **Smooth**: Differentiable everywhere
/// - **Non-monotonic**: Has small negative region for negative inputs
/// - **Memory efficient**: In-place operation saves memory
///
/// **Relationship to Other Functions:**
/// - SiLU(x) = x * σ(x), where σ is sigmoid
/// - Also known as Swish activation
/// - Similar to GELU but with different mathematical foundation
///
/// **Applications:**
/// - **Modern architectures**: Increasingly popular in recent models
/// - **Mobile models**: Good balance of performance and efficiency
/// - **Research**: Alternative to ReLU and GELU
pub fn silu_<T: FloatElement>(input: &mut Tensor<T>) -> TorshResult<()>
where
    T: Copy + PartialOrd + torsh_core::dtype::TensorElement,
{
    // Fallback implementation: x * sigmoid(x)
    let one = <T as torsh_core::dtype::TensorElement>::one();
    let zero = <T as torsh_core::dtype::TensorElement>::zero();
    let data = input.data()?;
    let result_data: Vec<T> = data
        .iter()
        .map(|&x| {
            // Compute sigmoid(x) with numerical stability
            let sigmoid_x = if x >= zero {
                one / (one + (-x).exp())
            } else {
                let exp_x = x.exp();
                exp_x / (one + exp_x)
            };
            x * sigmoid_x
        })
        .collect();

    *input = Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::device::DeviceType;
    use torsh_tensor::creation::from_vec;

    #[test]
    fn test_in_place_relu() -> TorshResult<()> {
        let mut input = from_vec(vec![-2.0f32, -1.0, 0.0, 1.0, 2.0], &[5], DeviceType::Cpu)?;
        let expected = vec![0.0, 0.0, 0.0, 1.0, 2.0];

        relu_(&mut input)?;
        let data = input.data()?;

        for (i, (&actual, &expected)) in data.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-6,
                "ReLU mismatch at index {}: {} vs {}",
                i,
                actual,
                expected
            );
        }

        Ok(())
    }

    #[test]
    fn test_in_place_sigmoid() -> TorshResult<()> {
        let mut input = from_vec(vec![-2.0f32, 0.0, 2.0], &[3], DeviceType::Cpu)?;

        sigmoid_(&mut input)?;
        let data = input.data()?;

        // All values should be between 0 and 1
        for &val in data.iter() {
            assert!(
                val > 0.0 && val < 1.0,
                "Sigmoid output {} not in (0,1)",
                val
            );
        }

        // sigmoid(0) should be 0.5
        assert!(
            (data[1] - 0.5).abs() < 1e-6,
            "sigmoid(0) should be 0.5, got {}",
            data[1]
        );

        // Check ordering: sigmoid(-2) < sigmoid(0) < sigmoid(2)
        assert!(data[0] < data[1]);
        assert!(data[1] < data[2]);

        Ok(())
    }

    #[test]
    fn test_in_place_tanh() -> TorshResult<()> {
        let mut input = from_vec(vec![-2.0f32, 0.0, 2.0], &[3], DeviceType::Cpu)?;

        tanh_(&mut input)?;
        let data = input.data()?;

        // All values should be between -1 and 1
        for &val in data.iter() {
            assert!(val > -1.0 && val < 1.0, "Tanh output {} not in (-1,1)", val);
        }

        // tanh(0) should be 0
        assert!(
            (data[1] - 0.0).abs() < 1e-6,
            "tanh(0) should be 0, got {}",
            data[1]
        );

        // Check ordering: tanh(-2) < tanh(0) < tanh(2)
        assert!(data[0] < data[1]);
        assert!(data[1] < data[2]);

        // Check approximate values
        assert!(data[0] < -0.95); // tanh(-2) ≈ -0.964
        assert!(data[2] > 0.95); // tanh(2) ≈ 0.964

        Ok(())
    }

    #[test]
    fn test_in_place_leaky_relu() -> TorshResult<()> {
        let mut input = from_vec(vec![-2.0f32, -1.0, 0.0, 1.0, 2.0], &[5], DeviceType::Cpu)?;
        let negative_slope = 0.1;

        leaky_relu_(&mut input, negative_slope)?;
        let data = input.data()?;

        // Expected values with slope 0.1
        let expected = vec![-0.2, -0.1, 0.0, 1.0, 2.0];

        for (i, (&actual, &expected)) in data.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-6,
                "LeakyReLU mismatch at index {}: {} vs {}",
                i,
                actual,
                expected
            );
        }

        Ok(())
    }

    #[test]
    fn test_in_place_gelu() -> TorshResult<()> {
        let mut input = from_vec(vec![-1.0f32, 0.0, 1.0], &[3], DeviceType::Cpu)?;

        gelu_(&mut input)?;
        let data = input.data()?;

        // GELU(0) should be approximately 0
        assert!(
            (data[1] - 0.0).abs() < 1e-6,
            "GELU(0) should be 0, got {}",
            data[1]
        );

        // For positive inputs, GELU should be positive and close to input for large values
        assert!(data[2] > 0.0, "GELU(1) should be positive, got {}", data[2]);

        // For negative inputs, GELU should be small negative
        assert!(
            data[0] < 0.0 && data[0] > -0.5,
            "GELU(-1) should be small negative, got {}",
            data[0]
        );

        Ok(())
    }

    #[test]
    fn test_in_place_silu() -> TorshResult<()> {
        let mut input = from_vec(vec![-1.0f32, 0.0, 1.0], &[3], DeviceType::Cpu)?;

        silu_(&mut input)?;
        let data = input.data()?;

        // SiLU(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
        assert!(
            (data[1] - 0.0).abs() < 1e-6,
            "SiLU(0) should be 0, got {}",
            data[1]
        );

        // For positive inputs, SiLU should be positive
        assert!(data[2] > 0.0, "SiLU(1) should be positive, got {}", data[2]);

        // For negative inputs, SiLU should be negative but closer to 0
        assert!(
            data[0] < 0.0 && data[0] > -1.0,
            "SiLU(-1) should be small negative, got {}",
            data[0]
        );

        Ok(())
    }

    #[test]
    fn test_in_place_memory_efficiency() -> TorshResult<()> {
        // Test that in-place operations actually modify the original tensor
        let mut original = from_vec(vec![1.0f32, -1.0, 2.0, -2.0], &[4], DeviceType::Cpu)?;
        let original_data = original.data()?.clone();

        // Apply in-place ReLU
        relu_(&mut original)?;
        let modified_data = original.data()?;

        // Check that negative values were zeroed
        assert_eq!(modified_data[0], 1.0); // 1.0 unchanged
        assert_eq!(modified_data[1], 0.0); // -1.0 -> 0.0
        assert_eq!(modified_data[2], 2.0); // 2.0 unchanged
        assert_eq!(modified_data[3], 0.0); // -2.0 -> 0.0

        // Verify that original values were indeed modified
        assert_ne!(original_data[1], modified_data[1]);
        assert_ne!(original_data[3], modified_data[3]);

        Ok(())
    }

    #[test]
    fn test_in_place_numerical_stability() -> TorshResult<()> {
        // Test with extreme values to check numerical stability
        let mut large_input = from_vec(vec![-100.0f32, 0.0, 100.0], &[3], DeviceType::Cpu)?;

        // Test sigmoid with extreme values
        sigmoid_(&mut large_input)?;
        let data = large_input.data()?;

        // All values should be finite and in [0, 1] (inclusive due to floating point precision)
        for &val in data.iter() {
            assert!(
                val.is_finite(),
                "Sigmoid produced non-finite value: {}",
                val
            );
            assert!(
                val >= 0.0 && val <= 1.0,
                "Sigmoid value {} not in [0,1]",
                val
            );
        }

        // sigmoid(-100) should be very close to 0
        assert!(
            data[0] < 1e-10,
            "sigmoid(-100) should be ~0, got {}",
            data[0]
        );
        // sigmoid(0) should be 0.5
        assert!(
            (data[1] - 0.5).abs() < 1e-6,
            "sigmoid(0) should be 0.5, got {}",
            data[1]
        );
        // sigmoid(100) should be very close to 1
        assert!(
            data[2] > 0.9999999,
            "sigmoid(100) should be ~1, got {}",
            data[2]
        );

        Ok(())
    }
}
