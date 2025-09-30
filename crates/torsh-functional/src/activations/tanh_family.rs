//! Hyperbolic tangent family activation functions
//!
//! This module provides various tanh-based activation functions including
//! standard tanh, softsign, hardtanh, and tanhshrink variants.

use torsh_core::{dtype::FloatElement, ComplexElement, Result as TorshResult};
use torsh_tensor::Tensor;

/// Hyperbolic tangent activation function
///
/// **Mathematical Definition:**
/// ```text
/// tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x)) = (e^(2x) - 1) / (e^(2x) + 1)
/// ```
///
/// **Relationship to Sigmoid:**
/// ```text
/// tanh(x) = 2σ(2x) - 1, where σ is the sigmoid function
/// σ(x) = (1 + tanh(x/2)) / 2
/// ```
///
/// **Derivative:**
/// ```text
/// d/dx tanh(x) = 1 - tanh²(x) = sech²(x)
/// ```
///
/// **Properties:**
/// - **Range**: (-1, 1) - strictly between -1 and 1
/// - **Zero-centered**: Unlike sigmoid, outputs are centered around zero
/// - **Odd function**: tanh(-x) = -tanh(x)
/// - **Monotonic**: Strictly increasing function
/// - **Smooth**: Infinitely differentiable everywhere
/// - **Limit behavior**:
///   - lim(x→∞) tanh(x) = 1
///   - lim(x→-∞) tanh(x) = -1
///   - tanh(0) = 0
///
/// **Applications:**
/// - **Hidden layers**: Better than sigmoid due to zero-centered outputs
/// - **RNNs**: Traditional activation for vanilla RNNs
/// - **LSTM gates**: Used in some LSTM gate implementations
/// - **Normalization**: When you need outputs in [-1, 1] range
/// - **Control systems**: Natural for systems requiring bipolar outputs
///
/// **Advantages:**
/// - **Zero-centered**: Helps with optimization convergence
/// - **Stronger gradients**: Less gradient vanishing compared to sigmoid
/// - **Bounded output**: Prevents exploding activations
/// - **Smooth everywhere**: Better optimization properties than ReLU
///
/// **Disadvantages:**
/// - **Still saturates**: Vanishing gradient problem for |x| > 2.5
/// - **Computationally expensive**: Requires exponential calculations
/// - **Gradient magnitude**: Max gradient is 1, can slow learning
///
/// **Numerical Stability:**
/// For large |x|, direct computation may overflow. Implementation should use:
/// ```text
/// tanh(x) = sign(x) * (1 - 2/(exp(2|x|) + 1)) for |x| > threshold
/// ```
pub fn tanh<T: FloatElement>(input: &Tensor<T>) -> TorshResult<Tensor<T>> {
    input.tanh()
}

/// Softsign activation function
///
/// **Mathematical Definition:**
/// ```text
/// softsign(x) = x / (1 + |x|)
/// ```
///
/// **Alternative Form:**
/// ```text
/// softsign(x) = x / (1 + abs(x))
/// ```
///
/// **Derivative:**
/// ```text
/// d/dx softsign(x) = 1 / (1 + |x|)²
/// ```
///
/// **Properties:**
/// - **Range**: (-1, 1) - strictly between -1 and 1
/// - **Zero-centered**: Outputs are centered around zero
/// - **Odd function**: softsign(-x) = -softsign(x)
/// - **Monotonic**: Strictly increasing function
/// - **Smooth**: Differentiable everywhere except at x = 0 (but continuous)
/// - **Limit behavior**:
///   - lim(x→∞) softsign(x) = 1
///   - lim(x→-∞) softsign(x) = -1
///   - softsign(0) = 0
///
/// **Comparison to Tanh:**
/// - **Softer saturation**: Slower approach to asymptotes than tanh
/// - **Polynomial denominator**: Uses 1 + |x| instead of exponential
/// - **Gentler gradients**: More gradual gradient decay
/// - **Computational efficiency**: No exponential calculations required
///
/// **Applications:**
/// - **Alternative to tanh**: When you want slower saturation
/// - **Normalization layers**: Simple bounded activation
/// - **Gradient flow**: When you need gentler gradient behavior
/// - **Resource-constrained models**: More efficient than tanh
///
/// **Advantages:**
/// - **Computationally efficient**: No exponential operations
/// - **Gentler saturation**: Slower decay than tanh
/// - **Zero-centered**: Good for optimization
/// - **Smooth gradients**: No abrupt changes in gradient
///
/// **Disadvantages:**
/// - **Still saturates**: Though more slowly than tanh
/// - **Less common**: Fewer empirical studies compared to tanh/sigmoid
/// - **Non-differentiable at zero**: Though continuous and has one-sided derivatives
pub fn softsign<T: FloatElement>(input: &Tensor<T>) -> TorshResult<Tensor<T>>
where
    T: Default + ComplexElement<Real = T>,
{
    let abs_input = input.abs()?;
    let ones = input.ones_like()?;
    let denominator = ones.add_op(&abs_input)?;
    input.div(&denominator)
}

/// Hardtanh activation function
///
/// **Mathematical Definition:**
/// ```text
/// hardtanh(x) = max(min_val, min(max_val, x))
/// ```
/// Typically with min_val = -1 and max_val = 1:
/// ```text
/// hardtanh(x) = {
///   -1  if x < -1
///    x  if -1 ≤ x ≤ 1
///    1  if x > 1
/// }
/// ```
///
/// **Derivative:**
/// ```text
/// d/dx hardtanh(x) = {
///   0   if x < min_val
///   1   if min_val ≤ x ≤ max_val
///   0   if x > max_val
/// }
/// ```
///
/// **Properties:**
/// - **Piecewise linear**: Three linear segments
/// - **Bounded output**: Strictly bounded between min_val and max_val
/// - **Zero-centered**: When min_val = -max_val
/// - **Computationally efficient**: Simple clipping operation
/// - **Non-differentiable**: At boundary points (min_val, max_val)
///
/// **Comparison to Tanh:**
/// - **Linear center**: Identity function in the middle region
/// - **Hard boundaries**: Abrupt clipping vs smooth asymptotes
/// - **Faster computation**: No exponential calculations
/// - **Sparse gradients**: Gradients are 0 outside the linear region
///
/// **Applications:**
/// - **Mobile/edge computing**: Efficient alternative to tanh
/// - **Quantized networks**: Works well with low-precision arithmetic
/// - **ReLU alternative**: When you need bounded activation
/// - **Hardware implementations**: Easy to implement in fixed-point
///
/// **Advantages:**
/// - **Extremely fast**: Simple min/max operations
/// - **Memory efficient**: No intermediate exponential computations
/// - **Bounded**: Prevents exploding activations
/// - **Hardware friendly**: Easy to implement in specialized hardware
///
/// **Disadvantages:**
/// - **Non-smooth**: Discontinuous derivatives at boundaries
/// - **Sparse gradients**: Zero gradients in saturation regions
/// - **Less expressive**: Linear center may be limiting
/// - **Gradient flow**: Can block gradients completely in saturation
pub fn hardtanh<T: FloatElement>(
    input: &Tensor<T>,
    min_val: f64,
    max_val: f64,
) -> TorshResult<Tensor<T>>
where
    T: From<f32> + Default,
{
    let min_tensor = input
        .ones_like()?
        .mul_scalar(num_traits::cast(min_val as f32).unwrap())?;
    let max_tensor = input
        .ones_like()?
        .mul_scalar(num_traits::cast(max_val as f32).unwrap())?;

    // Clamp between min_val and max_val
    let clamped_min = input.maximum(&min_tensor)?;
    clamped_min.minimum(&max_tensor)
}

/// Tanhshrink activation function
///
/// **Mathematical Definition:**
/// ```text
/// tanhshrink(x) = x - tanh(x)
/// ```
///
/// **Derivative:**
/// ```text
/// d/dx tanhshrink(x) = 1 - (1 - tanh²(x)) = tanh²(x)
/// ```
///
/// **Properties:**
/// - **Range**: (-∞, ∞) - unbounded output
/// - **Zero at origin**: tanhshrink(0) = 0 - tanh(0) = 0
/// - **Odd function**: tanhshrink(-x) = -tanhshrink(x)
/// - **Monotonic**: Strictly increasing function
/// - **Asymptotic behavior**:
///   - For large positive x: tanhshrink(x) ≈ x - 1
///   - For large negative x: tanhshrink(x) ≈ x + 1
///   - For small x: tanhshrink(x) ≈ x³/3 (Taylor expansion)
///
/// **Physical Interpretation:**
/// Represents the "excess" of the input beyond what tanh can represent.
/// It's the residual after tanh normalization.
///
/// **Gradient Properties:**
/// - **Never zero**: Gradient is always tanh²(x) ≥ 0
/// - **Small near origin**: Gradient approaches 0 as x → 0
/// - **Approaches 1**: Gradient approaches 1 as |x| → ∞
/// - **Self-regularizing**: Automatically reduces gradients near zero
///
/// **Applications:**
/// - **Residual connections**: Where you want to preserve input information
/// - **Attention mechanisms**: As a gating function
/// - **Normalization**: When you need to remove bounded components
/// - **Research models**: Specialized architectures requiring unbounded outputs
///
/// **Advantages:**
/// - **Unbounded**: No saturation issues for large inputs
/// - **Identity-like**: Becomes nearly linear for large |x|
/// - **Smooth gradients**: No abrupt gradient changes
/// - **Self-normalizing**: Automatically handles different input scales
///
/// **Disadvantages:**
/// - **Vanishing gradients**: Very small gradients near zero
/// - **Computational cost**: Requires tanh computation
/// - **Less studied**: Limited empirical validation
/// - **Complex behavior**: Non-intuitive activation pattern
pub fn tanhshrink<T: FloatElement>(input: &Tensor<T>) -> TorshResult<Tensor<T>>
where
    T: Default,
{
    let tanh_result = input.tanh()?;
    input.sub(&tanh_result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::device::DeviceType;
    use torsh_tensor::creation::from_vec;

    #[test]
    fn test_tanh_range() -> TorshResult<()> {
        let input = from_vec(vec![-10.0, -1.0, 0.0, 1.0, 10.0], &[5], DeviceType::Cpu)?;
        let output = tanh(&input)?;
        let output_data = output.data()?;

        // All values should be in (-1, 1)
        for &val in output_data.iter() {
            assert!(val > -1.0 && val < 1.0, "Tanh value {} not in (-1,1)", val);
        }

        // Check specific values
        assert!((output_data[2] - 0.0).abs() < 1e-6); // tanh(0) = 0

        // Check asymptotic behavior
        assert!(output_data[0] < -0.99); // tanh(-10) ≈ -1
        assert!(output_data[4] > 0.99); // tanh(10) ≈ 1

        Ok(())
    }

    #[test]
    fn test_softsign_properties() -> TorshResult<()> {
        let input = from_vec(vec![-10.0, -1.0, 0.0, 1.0, 10.0], &[5], DeviceType::Cpu)?;
        let output = softsign(&input)?;
        let output_data = output.data()?;

        // All values should be in (-1, 1)
        for &val in output_data.iter() {
            assert!(
                val > -1.0 && val < 1.0,
                "Softsign value {} not in (-1,1)",
                val
            );
        }

        // softsign(0) = 0
        assert!((output_data[2] - 0.0).abs() < 1e-6);

        // Check specific values
        // softsign(1) = 1/(1+1) = 0.5
        assert!((output_data[3] - 0.5).abs() < 1e-6);
        // softsign(-1) = -1/(1+1) = -0.5
        assert!((output_data[1] + 0.5).abs() < 1e-6);

        Ok(())
    }

    #[test]
    fn test_hardtanh_clipping() -> TorshResult<()> {
        let input = from_vec(vec![-5.0, -1.0, 0.0, 1.0, 5.0], &[5], DeviceType::Cpu)?;
        let output = hardtanh(&input, -1.0, 1.0)?;
        let output_data = output.data()?;

        // Check clipping behavior
        assert!((output_data[0] + 1.0).abs() < 1e-6); // hardtanh(-5) = -1
        assert!((output_data[1] + 1.0).abs() < 1e-6); // hardtanh(-1) = -1
        assert!((output_data[2] - 0.0).abs() < 1e-6); // hardtanh(0) = 0
        assert!((output_data[3] - 1.0).abs() < 1e-6); // hardtanh(1) = 1
        assert!((output_data[4] - 1.0).abs() < 1e-6); // hardtanh(5) = 1

        Ok(())
    }

    #[test]
    fn test_tanhshrink_properties() -> TorshResult<()> {
        let input = from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5], DeviceType::Cpu)?;
        let output = tanhshrink(&input)?;
        let output_data = output.data()?;

        // tanhshrink(0) = 0 - tanh(0) = 0 - 0 = 0
        assert!((output_data[2] - 0.0).abs() < 1e-6);

        // For positive x: tanhshrink(x) should be positive but smaller than x
        // since tanh(x) < x for positive x
        assert!(output_data[3] > 0.0 && output_data[3] < 1.0);
        assert!(output_data[4] > 0.0 && output_data[4] < 2.0);

        // For negative x: tanhshrink(x) should be negative
        assert!(output_data[0] < 0.0);
        assert!(output_data[1] < 0.0);

        // For large |x|, tanhshrink(x) ≈ x - tanh(x)
        assert!((output_data[4] - (2.0 - (2.0_f32).tanh())).abs() < 0.1);

        Ok(())
    }

    #[test]
    fn test_softsign_vs_tanh() -> TorshResult<()> {
        // Compare softsign and tanh behavior
        let input = from_vec(vec![0.5, 1.0, 2.0, 5.0], &[4], DeviceType::Cpu)?;
        let softsign_out = softsign(&input)?;
        let tanh_out = tanh(&input)?;

        let softsign_data = softsign_out.data()?;
        let tanh_data = tanh_out.data()?;

        // For small inputs, softsign and tanh should be similar (but not identical)
        assert!((softsign_data[0] - tanh_data[0]).abs() < 0.2);

        // For larger inputs, tanh saturates faster than softsign
        assert!(tanh_data[3] > softsign_data[3]); // tanh(5) > softsign(5)

        Ok(())
    }

    #[test]
    fn test_hardtanh_custom_bounds() -> TorshResult<()> {
        let input = from_vec(vec![-3.0, 0.0, 3.0], &[3], DeviceType::Cpu)?;
        let output = hardtanh(&input, -2.0, 2.0)?;
        let output_data = output.data()?;

        // Check custom bounds
        assert!((output_data[0] + 2.0).abs() < 1e-6); // clamp(-3, -2, 2) = -2
        assert!((output_data[1] - 0.0).abs() < 1e-6); // clamp(0, -2, 2) = 0
        assert!((output_data[2] - 2.0).abs() < 1e-6); // clamp(3, -2, 2) = 2

        Ok(())
    }
}
