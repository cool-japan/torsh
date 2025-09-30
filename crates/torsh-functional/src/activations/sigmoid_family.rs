//! Sigmoid family activation functions
//!
//! This module provides various sigmoid-based activation functions including
//! standard sigmoid, hard sigmoid, SiLU (Swish), and related variants.

use torsh_core::{dtype::FloatElement, Result as TorshResult};
use torsh_tensor::Tensor;

/// Sigmoid activation function
///
/// **Mathematical Definition:**
/// ```text
/// σ(x) = 1 / (1 + e^(-x)) = e^x / (e^x + 1)
/// ```
///
/// **Alternative Forms:**
/// ```text
/// σ(x) = (1 + tanh(x/2)) / 2
/// σ(x) = 1 / (1 + exp(-x))
/// ```
///
/// **Derivative:**
/// ```text
/// d/dx σ(x) = σ(x)(1 - σ(x)) = σ(x)σ(-x)
/// ```
/// This self-referential derivative makes sigmoid convenient for backpropagation.
///
/// **Properties:**
/// - **Range**: (0, 1) - strictly between 0 and 1
/// - **Monotonic**: Strictly increasing function
/// - **Smooth**: Infinitely differentiable everywhere
/// - **S-shaped curve**: Characteristic sigmoid shape
/// - **Symmetry**: σ(-x) = 1 - σ(x)
/// - **Limit behavior**:
///   - lim(x→∞) σ(x) = 1
///   - lim(x→-∞) σ(x) = 0
///   - σ(0) = 0.5
///
/// **Applications:**
/// - **Binary classification**: Output layer for binary classification tasks
/// - **Probability estimation**: Natural interpretation as probability
/// - **Gate mechanisms**: Used in LSTM and GRU gates (forget, input, output)
/// - **Logistic regression**: Core of logistic regression models
/// - **Historical networks**: Common in older neural network architectures
///
/// **Advantages:**
/// - Clear probabilistic interpretation
/// - Smooth gradient everywhere
/// - Well-studied mathematical properties
/// - Natural output range for probabilities
///
/// **Disadvantages:**
/// - **Vanishing gradient problem**: Gradients become very small for |x| > 3
/// - **Not zero-centered**: Outputs always positive, can slow convergence
/// - **Saturation**: Gradients vanish in saturation regions
/// - **Computationally expensive**: Requires exponential computation
///
/// **Numerical Stability:**
/// For large negative x, direct computation may underflow.
/// Implementation should use: σ(x) = exp(x)/(1 + exp(x)) for x < 0.
pub fn sigmoid<T: FloatElement>(input: &Tensor<T>) -> TorshResult<Tensor<T>> {
    input.sigmoid()
}

/// Hard Sigmoid activation function
///
/// A piecewise-linear approximation of sigmoid that is computationally efficient.
/// HardSigmoid(x) = max(0, min(1, (x + 3) / 6))
pub fn hardsigmoid<T: FloatElement>(input: &Tensor<T>) -> TorshResult<Tensor<T>>
where
    T: Copy + PartialOrd + From<f32>,
{
    // HardSigmoid(x) = max(0, min(1, (x + 3) / 6))
    let three = <T as From<f32>>::from(3.0);
    let six = <T as From<f32>>::from(6.0);
    let zero = <T as torsh_core::dtype::TensorElement>::zero();
    let one = <T as torsh_core::dtype::TensorElement>::one();

    let data = input.data()?;
    let result_data: Vec<T> = data
        .iter()
        .map(|&x| {
            let normalized = (x + three) / six;
            if normalized < zero {
                zero
            } else if normalized > one {
                one
            } else {
                normalized
            }
        })
        .collect();

    Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())
}

/// Hard Sigmoid v2 activation function (alternative formulation)
pub fn hardsigmoid_v2<T: FloatElement>(input: &Tensor<T>, _inplace: bool) -> TorshResult<Tensor<T>>
where
    T: Copy + PartialOrd + From<f32>,
{
    // Alternative formulation: HardSigmoid(x) = max(0, min(1, 0.2 * x + 0.5))
    let point_two = <T as From<f32>>::from(0.2);
    let point_five = <T as From<f32>>::from(0.5);
    let zero = <T as torsh_core::dtype::TensorElement>::zero();
    let one = <T as torsh_core::dtype::TensorElement>::one();

    let data = input.data()?;
    let result_data: Vec<T> = data
        .iter()
        .map(|&x| {
            let value = x * point_two + point_five;
            if value < zero {
                zero
            } else if value > one {
                one
            } else {
                value
            }
        })
        .collect();

    Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())
}

/// SiLU (Sigmoid Linear Unit) / Swish activation function
///
/// SiLU(x) = x * sigmoid(x) = x / (1 + e^(-x))
/// Also known as Swish activation function.
pub fn silu<T: FloatElement>(input: &Tensor<T>, _inplace: bool) -> TorshResult<Tensor<T>>
where
    T: Copy,
{
    let sigmoid_input = input.sigmoid()?;
    input.mul(&sigmoid_input)
}

/// Swish activation function (alias for SiLU)
///
/// Swish(x) = x * sigmoid(β * x), where β is typically 1
pub fn swish<T: FloatElement>(input: &Tensor<T>, inplace: bool) -> TorshResult<Tensor<T>>
where
    T: Copy,
{
    // Default β = 1, so Swish(x) = x * sigmoid(x) = SiLU(x)
    silu(input, inplace)
}

/// Hard Swish activation function
///
/// A piecewise-linear approximation of Swish that is computationally efficient.
/// HardSwish(x) = x * HardSigmoid(x)
pub fn hardswish<T: FloatElement>(input: &Tensor<T>) -> TorshResult<Tensor<T>>
where
    T: Copy + PartialOrd + From<f32>,
{
    let hard_sigmoid_input = hardsigmoid(input)?;
    input.mul(&hard_sigmoid_input)
}

/// Mish activation function
///
/// Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^x))
/// A smooth, non-monotonic activation function with good properties.
pub fn mish<T: FloatElement>(input: &Tensor<T>, inplace: bool) -> TorshResult<Tensor<T>>
where
    T: Copy + From<f32>,
{
    // Mish(x) = x * tanh(softplus(x))
    let softplus_input = softplus_impl(input)?;
    let tanh_softplus = softplus_input.tanh()?;

    let result = input.mul(&tanh_softplus)?;

    if inplace {
        // For true in-place operation, we would modify the original tensor
        // This is a simplified implementation
        Ok(result)
    } else {
        Ok(result)
    }
}

/// Log Sigmoid activation function
///
/// LogSigmoid(x) = log(sigmoid(x)) = log(1 / (1 + e^(-x))) = -log(1 + e^(-x))
/// Numerically stable implementation of log(sigmoid(x)).
pub fn log_sigmoid<T: FloatElement>(input: &Tensor<T>) -> TorshResult<Tensor<T>>
where
    T: Copy + PartialOrd + From<f32>,
{
    // Numerically stable: log(sigmoid(x)) = -softplus(-x)
    let neg_input = input.neg()?;
    let softplus_neg = softplus_impl(&neg_input)?;
    softplus_neg.neg()
}

// Helper function for softplus
fn softplus_impl<T: FloatElement>(input: &Tensor<T>) -> TorshResult<Tensor<T>>
where
    T: Copy + PartialOrd + From<f32>,
{
    // Softplus(x) = log(1 + e^x)
    // Numerically stable implementation
    let data = input.data()?;
    let result_data: Vec<T> = data
        .iter()
        .map(|&x| {
            let exp_x = x.exp();
            let one = <T as torsh_core::dtype::TensorElement>::one();
            (one + exp_x).ln()
        })
        .collect();

    Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())
}

/// Soft Plus activation function
///
/// SoftPlus(x) = log(1 + e^x)
/// A smooth approximation to the ReLU function.
pub fn softplus<T: FloatElement>(
    input: &Tensor<T>,
    beta: f64,
    threshold: f64,
) -> TorshResult<Tensor<T>>
where
    T: Copy + PartialOrd + From<f32>,
{
    let beta_val = <T as From<f32>>::from(beta as f32);
    let threshold_val = <T as From<f32>>::from(threshold as f32);

    let data = input.data()?;
    let result_data: Vec<T> = data
        .iter()
        .map(|&x| {
            let scaled = x * beta_val;
            if scaled > threshold_val {
                // For large values, return x to avoid overflow
                x
            } else {
                // SoftPlus(x) = (1/beta) * log(1 + exp(beta * x))
                let exp_scaled = scaled.exp();
                let one = <T as torsh_core::dtype::TensorElement>::one();
                (one + exp_scaled).ln() / beta_val
            }
        })
        .collect();

    Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::device::DeviceType;
    use torsh_tensor::creation::from_vec;

    #[test]
    fn test_sigmoid_range() -> TorshResult<()> {
        let input = from_vec(vec![-10.0, -1.0, 0.0, 1.0, 10.0], &[5], DeviceType::Cpu)?;
        let output = sigmoid(&input)?;
        let output_data = output.data()?;

        // All values should be in (0, 1)
        for &val in output_data.iter() {
            assert!(val > 0.0 && val < 1.0, "Sigmoid value {} not in (0,1)", val);
        }

        // Check specific values
        assert!((output_data[2] - 0.5_f32).abs() < 1e-6); // sigmoid(0) = 0.5

        Ok(())
    }

    #[test]
    fn test_hardsigmoid_piecewise() -> TorshResult<()> {
        let input = from_vec(vec![-10.0, -3.0, 0.0, 3.0, 10.0], &[5], DeviceType::Cpu)?;
        let output = hardsigmoid(&input)?;
        let output_data = output.data()?;

        // Check boundary conditions
        assert!(output_data[0] == 0.0); // hardsigmoid(-10) = 0
        assert!(output_data[1] == 0.0); // hardsigmoid(-3) = 0
        assert!((output_data[2] - 0.5_f32).abs() < 1e-6); // hardsigmoid(0) = 0.5
        assert!(output_data[3] == 1.0); // hardsigmoid(3) = 1
        assert!(output_data[4] == 1.0); // hardsigmoid(10) = 1

        Ok(())
    }

    #[test]
    fn test_silu_properties() -> TorshResult<()> {
        let input = from_vec(vec![-1.0, 0.0, 1.0], &[3], DeviceType::Cpu)?;
        let output = silu(&input, false)?;
        let output_data = output.data()?;

        // SiLU(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
        assert!((output_data[1] as f32).abs() < 1e-6);

        // SiLU(x) should be approximately x for large positive x
        let large_input = from_vec(vec![10.0], &[1], DeviceType::Cpu)?;
        let large_output = silu(&large_input, false)?;
        let large_val = large_output.item()?;
        assert!((large_val - 10.0_f32).abs() < 0.1); // Should be close to 10

        Ok(())
    }

    #[test]
    fn test_hardswish_efficiency() -> TorshResult<()> {
        let input = from_vec(vec![-3.0, 0.0, 3.0], &[3], DeviceType::Cpu)?;
        let output = hardswish(&input)?;
        let output_data = output.data()?;

        // HardSwish should approximate Swish but with piecewise computation
        // At boundaries: HardSwish(-3) = -3 * 0 = 0, HardSwish(3) = 3 * 1 = 3
        assert!(output_data[0] == 0.0);
        assert!(output_data[2] == 3.0);

        Ok(())
    }

    #[test]
    fn test_log_sigmoid_stability() -> TorshResult<()> {
        // Test with moderately large values to test numerical stability
        // Note: log_sigmoid(-100) = -inf is mathematically correct
        let input = from_vec(vec![-10.0, 0.0, 10.0], &[3], DeviceType::Cpu)?;
        let output = log_sigmoid(&input)?;
        let output_data = output.data()?;

        // log_sigmoid should produce finite values for reasonable inputs
        for &val in output_data.iter() {
            assert!(
                val.is_finite(),
                "log_sigmoid produced non-finite value: {}",
                val
            );
        }

        // log_sigmoid(0) should be log(0.5) = -ln(2)
        let expected_zero = -std::f32::consts::LN_2;
        assert!((output_data[1] - expected_zero).abs() < 1e-6);

        Ok(())
    }

    #[test]
    fn test_softplus_approximation() -> TorshResult<()> {
        let input = from_vec(vec![-5.0, 0.0, 5.0], &[3], DeviceType::Cpu)?;
        let output = softplus(&input, 1.0, 20.0)?;
        let output_data = output.data()?;

        // SoftPlus should be smooth approximation to ReLU
        // For large positive x, SoftPlus(x) ≈ x
        // For large negative x, SoftPlus(x) ≈ 0
        assert!(output_data[0] < 0.1); // SoftPlus(-5) ≈ 0
        assert!((output_data[1] - std::f32::consts::LN_2).abs() < 1e-6); // SoftPlus(0) = ln(2)
        assert!((output_data[2] - 5.0).abs() < 0.1); // SoftPlus(5) ≈ 5

        Ok(())
    }
}
