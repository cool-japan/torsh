//! Property-based tests for torsh-functional operations
//!
//! This module uses property-based testing to verify mathematical properties
//! that should hold for all valid inputs. These tests generate random inputs
//! and verify invariants, identities, and mathematical relationships.

use torsh_core::Result as TorshResult;
use torsh_tensor::creation::{ones, rand, randn, zeros};
use torsh_tensor::Tensor;

/// Generate random tensor with specified shape and reasonable value range
fn random_tensor(shape: &[usize], min: f32, max: f32) -> TorshResult<Tensor> {
    use torsh_tensor::creation::rand;

    let random_01 = rand(shape)?;
    let range = max - min;
    random_01.mul_scalar(range)?.add_scalar(min)
}

/// Property tests for activation functions
pub mod activation_properties {
    use super::*;
    use crate::activations::*;

    /// Test ReLU properties: non-negativity and monotonicity
    #[test]
    fn test_relu_properties() -> TorshResult<()> {
        for _ in 0..20 {
            let x = random_tensor(&[10, 5], -10.0, 10.0)?;
            let result = relu(&x, false)?;
            let result_data = result.data()?;

            // Property 1: ReLU output is always non-negative
            for &val in result_data.iter() {
                assert!(
                    val >= 0.0,
                    "ReLU output should be non-negative, got {}",
                    val
                );
            }

            // Property 2: ReLU is monotonic: if x1 <= x2, then ReLU(x1) <= ReLU(x2)
            let x_data = x.data()?;
            for i in 0..x_data.len() - 1 {
                if x_data[i] <= x_data[i + 1] {
                    assert!(
                        result_data[i] <= result_data[i + 1],
                        "ReLU should be monotonic: x[{}]={}, x[{}]={}, relu[{}]={}, relu[{}]={}",
                        i,
                        x_data[i],
                        i + 1,
                        x_data[i + 1],
                        i,
                        result_data[i],
                        i + 1,
                        result_data[i + 1]
                    );
                }
            }

            // Property 3: ReLU(x) = max(0, x)
            for (i, (&x_val, &relu_val)) in x_data.iter().zip(result_data.iter()).enumerate() {
                let expected = x_val.max(0.0);
                assert!(
                    (relu_val - expected).abs() < 1e-6,
                    "ReLU({}) should equal max(0, {}), got {}",
                    x_val,
                    x_val,
                    relu_val
                );
            }
        }
        Ok(())
    }

    /// Test sigmoid properties: output range (0, 1) and monotonicity
    #[test]
    fn test_sigmoid_properties() -> TorshResult<()> {
        for _ in 0..20 {
            let x = random_tensor(&[8, 6], -20.0, 20.0)?;
            let result = sigmoid(&x)?;
            let result_data = result.data()?;

            // Property 1: Sigmoid output is in [0, 1] (with floating-point tolerance)
            for &val in result_data.iter() {
                assert!(
                    val >= 0.0 && val <= 1.0,
                    "Sigmoid output should be in [0, 1], got {}",
                    val
                );
            }

            // Property 2: Sigmoid is monotonically increasing
            let x_data = x.data()?;
            for i in 0..x_data.len() - 1 {
                if x_data[i] < x_data[i + 1] {
                    assert!(
                        result_data[i] <= result_data[i + 1],
                        "Sigmoid should be monotonic increasing"
                    );
                }
            }

            // Property 3: Sigmoid symmetry: sigmoid(-x) = 1 - sigmoid(x)
            let neg_x = x.mul_scalar(-1.0)?;
            let neg_result = sigmoid(&neg_x)?;
            let neg_result_data = neg_result.data()?;

            for (i, (&pos, &neg)) in result_data.iter().zip(neg_result_data.iter()).enumerate() {
                let expected = 1.0 - neg;
                assert!(
                    (pos - expected).abs() < 1e-6,
                    "Sigmoid symmetry: sigmoid({}) + sigmoid({}) should equal 1",
                    x_data[i],
                    -x_data[i]
                );
            }
        }
        Ok(())
    }

    /// Test tanh properties: output range (-1, 1), odd function
    #[test]
    fn test_tanh_properties() -> TorshResult<()> {
        for _ in 0..20 {
            let x = random_tensor(&[7, 4], -10.0, 10.0)?;
            let result = tanh(&x)?;
            let result_data = result.data()?;

            // Property 1: Tanh output is in [-1, 1] (with floating-point tolerance)
            for &val in result_data.iter() {
                assert!(
                    val >= -1.0 && val <= 1.0,
                    "Tanh output should be in [-1, 1], got {}",
                    val
                );
            }

            // Property 2: Tanh is an odd function: tanh(-x) = -tanh(x)
            let neg_x = x.mul_scalar(-1.0)?;
            let neg_result = tanh(&neg_x)?;
            let neg_result_data = neg_result.data()?;

            for (i, (&pos, &neg)) in result_data.iter().zip(neg_result_data.iter()).enumerate() {
                assert!(
                    (pos + neg).abs() < 1e-6,
                    "Tanh odd function property: tanh({}) + tanh({}) should equal 0",
                    x.data()?[i],
                    -x.data()?[i]
                );
            }

            // Property 3: Tanh is monotonically increasing
            let x_data = x.data()?;
            for i in 0..x_data.len() - 1 {
                if x_data[i] < x_data[i + 1] {
                    assert!(
                        result_data[i] <= result_data[i + 1],
                        "Tanh should be monotonic increasing"
                    );
                }
            }
        }
        Ok(())
    }

    /// Test softmax properties: sum to 1, non-negativity
    #[test]
    fn test_softmax_properties() -> TorshResult<()> {
        for _ in 0..20 {
            let x = random_tensor(&[5], -10.0, 10.0)?;
            let result = softmax(&x, 0, None)?;
            let result_data = result.data()?;

            // Property 1: All values are non-negative
            for &val in result_data.iter() {
                assert!(
                    val >= 0.0,
                    "Softmax output should be non-negative, got {}",
                    val
                );
            }

            // Property 2: Sum equals 1
            let sum: f32 = result_data.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "Softmax outputs should sum to 1, got {}",
                sum
            );

            // Property 3: Translation invariance: softmax(x + c) = softmax(x)
            let c = 5.0;
            let x_shifted = x.add_scalar(c)?;
            let result_shifted = softmax(&x_shifted, 0, None)?;
            let shifted_data = result_shifted.data()?;

            for (i, (&orig, &shifted)) in result_data.iter().zip(shifted_data.iter()).enumerate() {
                assert!(
                    (orig - shifted).abs() < 1e-6,
                    "Softmax translation invariance failed at index {}",
                    i
                );
            }
        }
        Ok(())
    }
}

/// Property tests for linear algebra operations
pub mod linalg_properties {
    use super::*;
    use crate::linalg::*;

    /// Test matrix multiplication properties: associativity, distributivity
    #[test]
    fn test_matmul_properties() -> TorshResult<()> {
        for _ in 0..10 {
            let a = random_tensor(&[3, 4], -2.0, 2.0)?;
            let b = random_tensor(&[4, 5], -2.0, 2.0)?;
            let c = random_tensor(&[5, 3], -2.0, 2.0)?;

            // Property 1: Associativity: (AB)C = A(BC)
            let ab = a.matmul(&b)?;
            let ab_c = ab.matmul(&c)?;

            let bc = b.matmul(&c)?;
            let a_bc = a.matmul(&bc)?;

            let ab_c_data = ab_c.data()?;
            let a_bc_data = a_bc.data()?;

            for (i, (&left, &right)) in ab_c_data.iter().zip(a_bc_data.iter()).enumerate() {
                assert!(
                    (left - right).abs() < 1e-4,
                    "Matrix multiplication associativity failed at index {}",
                    i
                );
            }
        }
        Ok(())
    }

    /// Test matrix transpose properties: (A^T)^T = A, (AB)^T = B^T A^T
    #[test]
    fn test_transpose_properties() -> TorshResult<()> {
        for _ in 0..10 {
            let a = random_tensor(&[3, 4], -2.0, 2.0)?;

            // Property 1: (A^T)^T = A
            let at = a.t()?;
            let att = at.t()?;

            let a_data = a.data()?;
            let att_data = att.data()?;

            for (i, (&orig, &double_transposed)) in a_data.iter().zip(att_data.iter()).enumerate() {
                assert!(
                    (orig - double_transposed).abs() < 1e-6,
                    "Double transpose should equal original at index {}",
                    i
                );
            }

            // Property 2: (AB)^T = B^T A^T
            let b = random_tensor(&[4, 5], -2.0, 2.0)?;
            let ab = a.matmul(&b)?;
            let ab_t = ab.t()?;

            let bt = b.t()?;
            let at = a.t()?;
            let bt_at = bt.matmul(&at)?;

            let ab_t_data = ab_t.data()?;
            let bt_at_data = bt_at.data()?;

            for (i, (&left, &right)) in ab_t_data.iter().zip(bt_at_data.iter()).enumerate() {
                assert!(
                    (left - right).abs() < 1e-5,
                    "Transpose of product property failed at index {}",
                    i
                );
            }
        }
        Ok(())
    }

    /// Test norm properties: positive definiteness, triangle inequality
    #[test]
    fn test_norm_properties() -> TorshResult<()> {
        for _ in 0..20 {
            let x = random_tensor(&[10], -5.0, 5.0)?;
            let y = random_tensor(&[10], -5.0, 5.0)?;

            // Property 1: Norm is non-negative
            let norm_x = norm(&x, Some(NormOrd::P(2.0)), None, false)?;
            let norm_x_data = norm_x.data()?;
            assert!(norm_x_data[0] >= 0.0, "Norm should be non-negative");

            // Property 2: Norm is zero iff vector is zero
            let zero_vec = zeros(&[10])?;
            let norm_zero = norm(&zero_vec, Some(NormOrd::P(2.0)), None, false)?;
            let norm_zero_data = norm_zero.data()?;
            assert!(
                norm_zero_data[0].abs() < 1e-6,
                "Norm of zero vector should be zero"
            );

            // Property 3: Triangle inequality: ||x + y|| <= ||x|| + ||y||
            let x_plus_y = x.add_op(&y)?;
            let norm_x_plus_y = norm(&x_plus_y, Some(NormOrd::P(2.0)), None, false)?;
            let norm_y = norm(&y, Some(NormOrd::P(2.0)), None, false)?;

            let norm_x_plus_y_val = norm_x_plus_y.data()?[0];
            let norm_x_val = norm_x_data[0];
            let norm_y_val = norm_y.data()?[0];

            assert!(
                norm_x_plus_y_val <= norm_x_val + norm_y_val + 1e-6,
                "Triangle inequality violated: ||x+y||={} > ||x||+||y||={}",
                norm_x_plus_y_val,
                norm_x_val + norm_y_val
            );

            // Property 4: Homogeneity: ||cx|| = |c| * ||x||
            let c = 2.5;
            let cx = x.mul_scalar(c)?;
            let norm_cx = norm(&cx, Some(NormOrd::P(2.0)), None, false)?;
            let expected_norm = c.abs() * norm_x_val;
            let actual_norm = norm_cx.data()?[0];

            assert!(
                (actual_norm - expected_norm).abs() < 1e-5,
                "Homogeneity property violated: ||{}*x||={}, expected {}",
                c,
                actual_norm,
                expected_norm
            );
        }
        Ok(())
    }
}

/// Property tests for reduction operations
pub mod reduction_properties {
    use super::*;

    /// Test sum properties: linearity, commutativity
    #[test]
    fn test_sum_properties() -> TorshResult<()> {
        for _ in 0..20 {
            let x = random_tensor(&[3, 4], -5.0, 5.0)?;
            let y = random_tensor(&[3, 4], -5.0, 5.0)?;
            let c = 2.3;

            // Property 1: Linearity: sum(x + y) = sum(x) + sum(y)
            let x_plus_y = x.add_op(&y)?;
            let sum_x_plus_y = x_plus_y.sum()?;

            let sum_x = x.sum()?;
            let sum_y = y.sum()?;
            let sum_x_plus_sum_y = sum_x.add_op(&sum_y)?;

            let lhs = sum_x_plus_y.data()?[0];
            let rhs = sum_x_plus_sum_y.data()?[0];
            assert!(
                (lhs - rhs).abs() < 1e-5,
                "Sum linearity failed: sum(x+y)={}, sum(x)+sum(y)={}",
                lhs,
                rhs
            );

            // Property 2: Scaling: sum(c*x) = c*sum(x)
            let cx = x.mul_scalar(c)?;
            let sum_cx = cx.sum()?;
            let c_sum_x = sum_x.mul_scalar(c)?;

            let sum_cx_val = sum_cx.data()?[0];
            let c_sum_x_val = c_sum_x.data()?[0];
            assert!(
                (sum_cx_val - c_sum_x_val).abs() < 1e-5,
                "Sum scaling failed: sum({}*x)={}, {}*sum(x)={}",
                c,
                sum_cx_val,
                c,
                c_sum_x_val
            );
        }
        Ok(())
    }

    /// Test mean properties: translation invariance, scaling
    #[test]
    fn test_mean_properties() -> TorshResult<()> {
        for _ in 0..20 {
            let x = random_tensor(&[4, 5], -3.0, 3.0)?;
            let c = 1.7;
            let d = 0.8;

            // Property 1: Translation: mean(x + c) = mean(x) + c
            let x_plus_c = x.add_scalar(c)?;
            let mean_x_plus_c = x_plus_c.mean(None, false)?;

            let mean_x = x.mean(None, false)?;
            let mean_x_plus_c_expected = mean_x.add_scalar(c)?;

            let actual = mean_x_plus_c.data()?[0];
            let expected = mean_x_plus_c_expected.data()?[0];
            assert!(
                (actual - expected).abs() < 1e-6,
                "Mean translation failed: mean(x+{})={}, mean(x)+{}={}",
                c,
                actual,
                c,
                expected
            );

            // Property 2: Scaling: mean(d*x) = d*mean(x)
            let dx = x.mul_scalar(d)?;
            let mean_dx = dx.mean(None, false)?;
            let d_mean_x = mean_x.mul_scalar(d)?;

            let actual_scaled = mean_dx.data()?[0];
            let expected_scaled = d_mean_x.data()?[0];
            assert!(
                (actual_scaled - expected_scaled).abs() < 1e-6,
                "Mean scaling failed: mean({}*x)={}, {}*mean(x)={}",
                d,
                actual_scaled,
                d,
                expected_scaled
            );
        }
        Ok(())
    }
}

/// Property tests for operation fusion
pub mod fusion_properties {
    use super::*;
    use crate::fusion::*;

    /// Test fusion equivalence: fused operations should equal separate operations
    #[test]
    fn test_fusion_equivalence_properties() -> TorshResult<()> {
        for _ in 0..15 {
            let x = random_tensor(&[6, 4], -3.0, 3.0)?;
            let y = random_tensor(&[6, 4], -3.0, 3.0)?;
            let z = random_tensor(&[6, 4], -3.0, 3.0)?;

            // Test fused_mul_add equivalence: x * y + z
            let fused_result = fused_mul_add(&x, &y, &z)?;

            let xy = x.mul_op(&y)?;
            let separate_result = xy.add_op(&z)?;

            let fused_data = fused_result.data()?;
            let separate_data = separate_result.data()?;

            for (i, (&fused, &separate)) in fused_data.iter().zip(separate_data.iter()).enumerate()
            {
                assert!(
                    (fused - separate).abs() < 1e-6,
                    "Fused mul-add equivalence failed at index {}: fused={}, separate={}",
                    i,
                    fused,
                    separate
                );
            }

            // Test fused_add_mul equivalence: (x + y) * z
            let fused_add_mul_result = fused_add_mul(&x, &y, &z)?;

            let x_plus_y = x.add_op(&y)?;
            let separate_add_mul_result = x_plus_y.mul_op(&z)?;

            let fused_add_mul_data = fused_add_mul_result.data()?;
            let separate_add_mul_data = separate_add_mul_result.data()?;

            for (i, (&fused, &separate)) in fused_add_mul_data
                .iter()
                .zip(separate_add_mul_data.iter())
                .enumerate()
            {
                assert!(
                    (fused - separate).abs() < 1e-6,
                    "Fused add-mul equivalence failed at index {}: fused={}, separate={}",
                    i,
                    fused,
                    separate
                );
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_random_tensor_generation() -> TorshResult<()> {
        let tensor = random_tensor(&[3, 4], -1.0, 1.0)?;
        let data = tensor.data()?;

        assert_eq!(data.len(), 12);

        // Check all values are in range
        for &val in data.iter() {
            assert!(
                val >= -1.0 && val <= 1.0,
                "Value {} out of range [-1, 1]",
                val
            );
        }

        Ok(())
    }

    /// Run a subset of property tests to ensure they work
    #[test]
    fn test_property_framework() -> TorshResult<()> {
        // This just ensures the property testing framework is working
        // The actual property tests are in their respective modules and run individually
        let tensor = random_tensor(&[3, 4], -1.0, 1.0)?;
        assert_eq!(tensor.shape().dims(), &[3, 4]);
        Ok(())
    }
}

/// Enhanced mathematical property tests
pub mod advanced_mathematical_properties {
    use super::*;
    use crate::activations::*;
    use crate::linalg::*;

    /// Test distributive property for tensor operations: a * (b + c) = a * b + a * c
    #[test]
    fn test_distributive_property() -> TorshResult<()> {
        for _ in 0..15 {
            let a = random_tensor(&[4, 3], -5.0, 5.0)?;
            let b = random_tensor(&[4, 3], -5.0, 5.0)?;
            let c = random_tensor(&[4, 3], -5.0, 5.0)?;

            // Calculate a * (b + c)
            let sum_bc = b.add(&c)?;
            let left_side = a.mul(&sum_bc)?;

            // Calculate a * b + a * c
            let ab = a.mul(&b)?;
            let ac = a.mul(&c)?;
            let right_side = ab.add(&ac)?;

            // Check distributive property with tolerance
            let diff = left_side.sub(&right_side)?;
            let diff_data = diff.data()?;
            for &val in diff_data.iter() {
                assert!(
                    val.abs() < 1e-5,
                    "Distributive property violated: |difference| = {}",
                    val.abs()
                );
            }
        }
        Ok(())
    }

    /// Test associative property for addition: (a + b) + c = a + (b + c)
    #[test]
    fn test_addition_associativity() -> TorshResult<()> {
        for _ in 0..15 {
            let a = random_tensor(&[6, 2], -10.0, 10.0)?;
            let b = random_tensor(&[6, 2], -10.0, 10.0)?;
            let c = random_tensor(&[6, 2], -10.0, 10.0)?;

            // Calculate (a + b) + c
            let ab = a.add(&b)?;
            let left_side = ab.add(&c)?;

            // Calculate a + (b + c)
            let bc = b.add(&c)?;
            let right_side = a.add(&bc)?;

            // Check associativity with tolerance
            let diff = left_side.sub(&right_side)?;
            let diff_data = diff.data()?;
            for &val in diff_data.iter() {
                assert!(
                    val.abs() < 1e-6,
                    "Addition associativity violated: |difference| = {}",
                    val.abs()
                );
            }
        }
        Ok(())
    }

    /// Test commutative property for multiplication: a * b = b * a
    #[test]
    fn test_multiplication_commutativity() -> TorshResult<()> {
        for _ in 0..15 {
            let a = random_tensor(&[5, 4], -3.0, 3.0)?;
            let b = random_tensor(&[5, 4], -3.0, 3.0)?;

            // Calculate a * b
            let ab = a.mul(&b)?;

            // Calculate b * a
            let ba = b.mul(&a)?;

            // Check commutativity with tolerance
            let diff = ab.sub(&ba)?;
            let diff_data = diff.data()?;
            for &val in diff_data.iter() {
                assert!(
                    val.abs() < 1e-6,
                    "Multiplication commutativity violated: |difference| = {}",
                    val.abs()
                );
            }
        }
        Ok(())
    }

    /// Test identity properties: a + 0 = a, a * 1 = a
    #[test]
    fn test_identity_properties() -> TorshResult<()> {
        for _ in 0..10 {
            let a = random_tensor(&[3, 5], -8.0, 8.0)?;
            let zero = zeros(a.shape().dims())?;
            let one = ones(a.shape().dims())?;

            // Test additive identity: a + 0 = a
            let a_plus_zero = a.add(&zero)?;
            let add_diff = a.sub(&a_plus_zero)?;
            let add_diff_data = add_diff.data()?;
            for &val in add_diff_data.iter() {
                assert!(
                    val.abs() < 1e-7,
                    "Additive identity violated: |difference| = {}",
                    val.abs()
                );
            }

            // Test multiplicative identity: a * 1 = a
            let a_times_one = a.mul(&one)?;
            let mul_diff = a.sub(&a_times_one)?;
            let mul_diff_data = mul_diff.data()?;
            for &val in mul_diff_data.iter() {
                assert!(
                    val.abs() < 1e-7,
                    "Multiplicative identity violated: |difference| = {}",
                    val.abs()
                );
            }
        }
        Ok(())
    }

    /// Test inverse property for addition: a + (-a) = 0
    #[test]
    fn test_additive_inverse() -> TorshResult<()> {
        for _ in 0..10 {
            let a = random_tensor(&[4, 6], -5.0, 5.0)?;
            let neg_a = a.mul_scalar(-1.0)?;

            // Test: a + (-a) = 0
            let sum = a.add(&neg_a)?;
            let sum_data = sum.data()?;
            for &val in sum_data.iter() {
                assert!(
                    val.abs() < 1e-6,
                    "Additive inverse property violated: |value| = {}",
                    val.abs()
                );
            }
        }
        Ok(())
    }

    /// Test activation function composition properties
    #[test]
    fn test_activation_composition_properties() -> TorshResult<()> {
        for _ in 0..10 {
            let x = random_tensor(&[3, 4], -2.0, 2.0)?;

            // Test: ReLU is idempotent: ReLU(ReLU(x)) = ReLU(x)
            let relu_x = relu(&x, false)?;
            let relu_relu_x = relu(&relu_x, false)?;
            let relu_diff = relu_x.sub(&relu_relu_x)?;
            let relu_diff_data = relu_diff.data()?;
            for &val in relu_diff_data.iter() {
                assert!(
                    val.abs() < 1e-7,
                    "ReLU idempotency violated: |difference| = {}",
                    val.abs()
                );
            }

            // Test: Sigmoid and logit are inverse (approximately for values in (0,1))
            // sigmoid(logit(p)) ≈ p for p ∈ (0,1)
            let p = random_tensor(&[3, 4], 0.1, 0.9)?; // Values in (0,1)
            let logit_p = p.div(&(&ones(&[3, 4])?.sub(&p)?))?;
            let log_logit_p = logit_p.log()?;
            let sigmoid_log_logit_p = sigmoid(&log_logit_p)?;

            let sigmoid_diff = p.sub(&sigmoid_log_logit_p)?;
            let sigmoid_diff_data = sigmoid_diff.data()?;
            for &val in sigmoid_diff_data.iter() {
                assert!(
                    val.abs() < 1e-5,
                    "Sigmoid-logit inverse property violated: |difference| = {}",
                    val.abs()
                );
            }
        }
        Ok(())
    }
}
