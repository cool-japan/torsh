//! ReLU family activation functions
//!
//! This module provides various ReLU-based activation functions including
//! standard ReLU, Leaky ReLU, ELU, SELU, and other related variants.

use torsh_core::{dtype::FloatElement, Result as TorshResult};
use torsh_tensor::Tensor;

/// Rectified Linear Unit (ReLU) activation function
///
/// **Mathematical Definition:**
/// ```text
/// ReLU(x) = max(0, x) = {
///   x   if x > 0
///   0   if x ≤ 0
/// }
/// ```
///
/// **Derivative:**
/// ```text
/// d/dx ReLU(x) = {
///   1   if x > 0
///   0   if x ≤ 0
///   undefined at x = 0
/// }
/// ```
///
/// **Properties:**
/// - **Non-saturating**: Does not saturate for positive values, avoiding vanishing gradient problem
/// - **Sparse activation**: Sets negative values to zero, leading to sparse representations
/// - **Computationally efficient**: Simple thresholding operation
/// - **Non-differentiable at zero**: Though in practice treated as 0 or 1
/// - **Unbounded above**: Can produce arbitrarily large positive outputs
///
/// **Applications:**
/// - Most common activation function in deep learning (especially CNNs)
/// - Hidden layers in feedforward networks
/// - Convolutional layers in computer vision models
/// - Default choice for most neural network architectures
///
/// **Advantages:**
/// - Solves vanishing gradient problem for positive inputs
/// - Computationally inexpensive
/// - Induces sparsity in hidden units
/// - Empirically works well in practice
///
/// **Disadvantages:**
/// - "Dying ReLU" problem: neurons can get stuck at zero
/// - Not zero-centered (outputs only positive values)
/// - Unbounded activation can lead to exploding activations
///
/// **Variants:**
/// - Leaky ReLU: Allows small negative values
/// - ELU: Smooth alternative with negative saturation
/// - ReLU6: Bounded version with upper limit at 6
pub fn relu<T: FloatElement>(input: &Tensor<T>, inplace: bool) -> TorshResult<Tensor<T>>
where
    T: Copy + PartialOrd,
{
    if inplace {
        // For in-place operation, use generic helper
        use crate::activations::apply_elementwise_inplace;
        let zero = <T as torsh_core::dtype::TensorElement>::zero();
        apply_elementwise_inplace(input, inplace, |x| if x < zero { zero } else { x })
    } else {
        input.relu()
    }
}

/// Leaky ReLU activation function
///
/// **Mathematical Definition:**
/// ```text
/// LeakyReLU(x) = {
///   x                    if x > 0
///   α * x               if x ≤ 0
/// }
/// ```
/// where α is the negative slope parameter (typically 0.01).
///
/// **Alternative Form:**
/// ```text
/// LeakyReLU(x) = max(0, x) + α * min(0, x)
/// ```
///
/// **Derivative:**
/// ```text
/// d/dx LeakyReLU(x) = {
///   1   if x > 0
///   α   if x ≤ 0
///   undefined at x = 0
/// }
/// ```
///
/// **Properties:**
/// - **Fixes dying ReLU**: Small gradient for negative values prevents dead neurons
/// - **Non-saturating**: Like ReLU, unbounded for positive values
/// - **Sparse activation**: Reduces but doesn't eliminate sparsity
/// - **Parameter dependent**: Behavior controlled by negative slope α
/// - **Piecewise linear**: Simple computation, easy optimization
///
/// **Parameter Guidelines:**
/// - **α = 0.01**: Default value, allows 1% of negative signal through
/// - **α = 0.1**: More aggressive, allows 10% of negative signal
/// - **α = 0.2**: Approaching linear behavior
/// - **Parametric ReLU**: α becomes learnable parameter
///
/// **Applications:**
/// - **Deep networks**: Where dying ReLU is problematic
/// - **GANs**: Often used in generator networks
/// - **CNNs**: Alternative to ReLU in convolutional layers
/// - **Autoencoders**: Helps preserve information flow
///
/// **Advantages:**
/// - **Solves dying ReLU**: Neurons can recover from negative weights
/// - **Computationally efficient**: Simple linear operations
/// - **Small memory footprint**: Minimal additional computation
/// - **Empirically effective**: Often matches or exceeds ReLU performance
///
/// **Disadvantages:**
/// - **Not zero-centered**: Still biased towards positive outputs
/// - **Parameter tuning**: Requires choosing appropriate α value
/// - **Inconsistent results**: Performance varies across different tasks
/// - **Linear negative region**: May be too simple for complex patterns
///
/// **Variants:**
/// - **Parametric ReLU (PReLU)**: Learnable α parameter
/// - **Randomized ReLU (RReLU)**: Random α during training
/// - **ELU**: Smooth alternative with exponential negative region
pub fn leaky_relu<T: FloatElement>(
    input: &Tensor<T>,
    negative_slope: f64,
    inplace: bool,
) -> TorshResult<Tensor<T>>
where
    T: Copy + PartialOrd + From<f32>,
{
    if inplace {
        // For in-place operation, create element-wise implementation
        let data = input.data()?;
        let slope = <T as From<f32>>::from(negative_slope as f32);
        let zero = <T as torsh_core::dtype::TensorElement>::zero();
        let result_data: Vec<T> = data
            .iter()
            .map(|&x| if x < zero { x * slope } else { x })
            .collect();

        Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())
    } else {
        input.leaky_relu(<T as From<f32>>::from(negative_slope as f32))
    }
}

/// Exponential Linear Unit (ELU) activation function
pub fn elu<T: FloatElement>(input: &Tensor<T>, alpha: f64, _inplace: bool) -> TorshResult<Tensor<T>>
where
    T: Copy + PartialOrd + From<f32>,
{
    // ELU(x) = x if x > 0, alpha * (exp(x) - 1) if x <= 0
    // For now, use element-wise implementation for both in-place and out-of-place
    let data = input.data()?;
    let alpha_val = <T as From<f32>>::from(alpha as f32);
    let zero = <T as torsh_core::dtype::TensorElement>::zero();
    let one = <T as torsh_core::dtype::TensorElement>::one();
    let result_data: Vec<T> = data
        .iter()
        .map(|&x| {
            if x > zero {
                x
            } else {
                alpha_val * (x.exp() - one)
            }
        })
        .collect();

    Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())
}

/// Scaled Exponential Linear Unit (SELU) activation function
pub fn selu<T: FloatElement>(input: &Tensor<T>, _inplace: bool) -> TorshResult<Tensor<T>>
where
    T: Copy + PartialOrd + From<f32> + Default,
{
    // SELU(x) = scale * (max(0, x) + min(0, alpha * (exp(x) - 1)))
    // where alpha = 1.6732632423543772848170429916717
    // and scale = 1.0507009873554804934193349852946
    let alpha = 1.673_263_242_354_377_2;
    let scale = 1.050_700_987_355_480_5;

    let elu_result = elu(input, alpha, false)?;
    elu_result.mul_scalar(num_traits::cast(scale).unwrap())
}

/// ReLU6 activation function
///
/// Bounded ReLU that caps outputs at 6: ReLU6(x) = min(max(0, x), 6)
pub fn relu6<T: FloatElement>(input: &Tensor<T>, inplace: bool) -> TorshResult<Tensor<T>>
where
    T: Copy + PartialOrd + From<f32>,
{
    use crate::activations::apply_elementwise_inplace;

    let zero = <T as torsh_core::dtype::TensorElement>::zero();
    let six = <T as From<f32>>::from(6.0);

    apply_elementwise_inplace(input, inplace, |x| {
        if x < zero {
            zero
        } else if x > six {
            six
        } else {
            x
        }
    })
}

/// Parametric ReLU activation function
///
/// PReLU(x) = max(0, x) + a * min(0, x), where 'a' is learned per channel
pub fn prelu<T: FloatElement>(input: &Tensor<T>, weight: &Tensor<T>) -> TorshResult<Tensor<T>>
where
    T: Copy + PartialOrd,
{
    // PReLU: x if x > 0, weight * x if x <= 0
    let zero = Tensor::zeros_like(input)?;
    let positive_mask = input.gt(&zero)?;
    let negative_part = input.mul(weight)?;
    input.where_tensor(&positive_mask, &negative_part)
}

/// Continuously differentiable exponential linear unit (CELU)
pub fn celu<T: FloatElement>(input: &Tensor<T>, alpha: f64) -> TorshResult<Tensor<T>>
where
    T: Copy + PartialOrd + From<f32>,
{
    // CELU(x) = max(0, x) + min(0, alpha * (exp(x/alpha) - 1))
    let alpha_val = <T as From<f32>>::from(alpha as f32);
    let zero = <T as torsh_core::dtype::TensorElement>::zero();
    let one = <T as torsh_core::dtype::TensorElement>::one();

    let data = input.data()?;
    let result_data: Vec<T> = data
        .iter()
        .map(|&x| {
            if x > zero {
                x
            } else {
                alpha_val * ((x / alpha_val).exp() - one)
            }
        })
        .collect();

    Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())
}

/// Randomized ReLU (RReLU)
///
/// Uses random slope for negative values during training, fixed slope during inference
pub fn rrelu<T: FloatElement>(
    input: &Tensor<T>,
    lower: f64,
    upper: f64,
    training: bool,
    inplace: bool,
) -> TorshResult<Tensor<T>>
where
    T: Copy + PartialOrd + From<f32>,
{
    let slope = if training {
        // Random slope between lower and upper during training
        use scirs2_core::random::thread_rng;
        let mut rng = thread_rng();
        rng.gen_range(lower..upper)
    } else {
        // Fixed slope (average) during inference
        (lower + upper) / 2.0
    };

    leaky_relu(input, slope, inplace)
}

/// Hard shrinkage function
pub fn hardshrink<T: FloatElement>(input: &Tensor<T>, lambd: f64) -> TorshResult<Tensor<T>>
where
    T: Copy + PartialOrd + From<f32>,
{
    let lambda_val = <T as From<f32>>::from(lambd as f32);
    let zero = <T as torsh_core::dtype::TensorElement>::zero();

    let data = input.data()?;
    let result_data: Vec<T> = data
        .iter()
        .map(|&x| if x.abs() > lambda_val { x } else { zero })
        .collect();

    Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())
}

/// Soft shrinkage function
pub fn softshrink<T: FloatElement>(input: &Tensor<T>, lambd: f64) -> TorshResult<Tensor<T>>
where
    T: Copy + PartialOrd + From<f32>,
{
    let lambda_val = <T as From<f32>>::from(lambd as f32);
    let zero = <T as torsh_core::dtype::TensorElement>::zero();

    let data = input.data()?;
    let result_data: Vec<T> = data
        .iter()
        .map(|&x| {
            if x > lambda_val {
                x - lambda_val
            } else if x < -lambda_val {
                x + lambda_val
            } else {
                zero
            }
        })
        .collect();

    Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())
}

/// Threshold activation function
pub fn threshold<T: FloatElement>(
    input: &Tensor<T>,
    threshold: f64,
    value: f64,
    inplace: bool,
) -> TorshResult<Tensor<T>>
where
    T: Copy + PartialOrd + From<f32>,
{
    let threshold_val = <T as From<f32>>::from(threshold as f32);
    let replace_val = <T as From<f32>>::from(value as f32);

    let data = input.data()?;
    let result_data: Vec<T> = data
        .iter()
        .map(|&x| if x > threshold_val { x } else { replace_val })
        .collect();

    let result = Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())?;

    if inplace {
        // For inplace, we would need to modify the original tensor
        // This is a simplified implementation
        Ok(result)
    } else {
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::device::DeviceType;
    use torsh_tensor::creation::from_vec;

    #[test]
    fn test_relu_basic() -> TorshResult<()> {
        let input = from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5], DeviceType::Cpu)?;
        let output = relu(&input, false)?;
        let expected = from_vec(vec![0.0, 0.0, 0.0, 1.0, 2.0], &[5], DeviceType::Cpu)?;

        let output_data = output.data()?;
        let expected_data = expected.data()?;

        for (i, (&out, &exp)) in output_data.iter().zip(expected_data.iter()).enumerate() {
            assert!(
                ((out - exp) as f32).abs() < 1e-6,
                "Mismatch at index {}: {} vs {}",
                i,
                out,
                exp
            );
        }
        Ok(())
    }

    #[test]
    fn test_leaky_relu_basic() -> TorshResult<()> {
        let input = from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5], DeviceType::Cpu)?;
        let output = leaky_relu(&input, 0.1, false)?;

        let output_data = output.data()?;
        // Expected: [-0.2, -0.1, 0.0, 1.0, 2.0]
        let expected = vec![-0.2, -0.1, 0.0, 1.0, 2.0];

        for (i, (&out, &exp)) in output_data.iter().zip(expected.iter()).enumerate() {
            assert!(
                ((out - exp) as f32).abs() < 1e-6,
                "Mismatch at index {}: {} vs {}",
                i,
                out,
                exp
            );
        }
        Ok(())
    }

    #[test]
    fn test_relu6_clipping() -> TorshResult<()> {
        let input = from_vec(vec![-1.0, 3.0, 8.0], &[3], DeviceType::Cpu)?;
        let output = relu6(&input, false)?;

        let output_data = output.data()?;
        // Expected: [0.0, 3.0, 6.0]
        let expected = vec![0.0, 3.0, 6.0];

        for (i, (&out, &exp)) in output_data.iter().zip(expected.iter()).enumerate() {
            assert!(
                ((out - exp) as f32).abs() < 1e-6,
                "Mismatch at index {}: {} vs {}",
                i,
                out,
                exp
            );
        }
        Ok(())
    }

    #[test]
    fn test_elu_properties() -> TorshResult<()> {
        let input = from_vec(vec![-1.0, 0.0, 1.0], &[3], DeviceType::Cpu)?;
        let output = elu(&input, 1.0, false)?;

        let output_data = output.data()?;

        // For x >= 0, ELU(x) = x
        assert!((output_data[1] - 0.0_f32).abs() < 1e-6); // ELU(0) = 0
        assert!((output_data[2] - 1.0_f32).abs() < 1e-6); // ELU(1) = 1

        // For x < 0, ELU(x) = alpha * (exp(x) - 1)
        // ELU(-1) = 1 * (exp(-1) - 1) = e^(-1) - 1 ≈ 0.368 - 1 = -0.632
        assert!((output_data[0] + 0.632_f32).abs() < 0.01); // Approximate check

        Ok(())
    }
}
