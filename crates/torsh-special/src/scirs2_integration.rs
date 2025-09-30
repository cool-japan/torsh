//! SciRS2 Special Functions Integration
//!
//! This module provides PyTorch-compatible wrappers around SciRS2 special functions.

use crate::TorshResult;
use torsh_core::TorshError;
use torsh_tensor::Tensor;

/// Wrapper for the gamma function using SciRS2
///
/// Computes the gamma function Γ(x) for each element in the input tensor.
/// The gamma function is defined as Γ(x) = ∫₀^∞ t^(x-1) e^(-t) dt for x > 0.
pub fn gamma(input: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let data = input.data()?;
    let result_data: Vec<f32> = data
        .iter()
        .map(|&x| {
            // Use SciRS2 gamma function
            scirs2_special::gamma(x as f64) as f32
        })
        .collect();

    Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())
}

/// Wrapper for the log gamma function using SciRS2
///
/// Computes the natural logarithm of the gamma function ln(Γ(x)) for each element.
/// This is more numerically stable than computing log(gamma(x)) directly.
pub fn lgamma(input: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let data = input.data()?;
    let result_data: Vec<f32> = data
        .iter()
        .map(|&x| scirs2_special::loggamma(x as f64) as f32)
        .collect();

    Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())
}

/// Wrapper for the digamma function using SciRS2
///
/// Computes the digamma function ψ(x) = d/dx ln(Γ(x)) = Γ'(x)/Γ(x).
pub fn digamma(input: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let data = input.data()?;
    let result_data: Vec<f32> = data
        .iter()
        .map(|&x| scirs2_special::digamma(x as f64) as f32)
        .collect();

    Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())
}

/// Wrapper for the polygamma function using SciRS2
///
/// Computes the polygamma function ψ^(m)(x) = d^(m+1)/dx^(m+1) ln(Γ(x)).
pub fn polygamma(m: i32, input: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let data = input.data()?;
    let result_data: Vec<f32> = data
        .iter()
        .map(|&x| {
            // Note: polygamma not available in scirs2-special, using digamma for m=0
            if m == 0 {
                scirs2_special::digamma(x as f64) as f32
            } else {
                // Use the polygamma implementation from gamma.rs for higher orders
                crate::gamma::polygamma_scalar(m, x as f64) as f32
            }
        })
        .collect();

    Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())
}

/// Wrapper for the beta function using SciRS2
///
/// Computes the beta function B(a, b) = Γ(a)Γ(b)/Γ(a+b).
pub fn beta(a: &Tensor<f32>, b: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let a_data = a.data()?;
    let b_data = b.data()?;

    if a_data.len() != b_data.len() {
        return Err(TorshError::shape_mismatch(
            a.shape().dims(),
            b.shape().dims(),
        ));
    }

    let result_data: Vec<f32> = a_data
        .iter()
        .zip(b_data.iter())
        .map(|(&a_val, &b_val)| scirs2_special::beta(a_val as f64, b_val as f64) as f32)
        .collect();

    Tensor::from_data(result_data, a.shape().dims().to_vec(), a.device())
}

/// Wrapper for the error function using SciRS2
///
/// Computes the error function erf(x) = (2/√π) ∫₀^x e^(-t²) dt.
pub fn erf(input: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let data = input.data()?;
    let result_data: Vec<f32> = data
        .iter()
        .map(|&x| scirs2_special::erf(x as f64) as f32)
        .collect();

    Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())
}

/// Wrapper for the complementary error function using SciRS2
///
/// Computes the complementary error function erfc(x) = 1 - erf(x).
pub fn erfc(input: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let data = input.data()?;
    let result_data: Vec<f32> = data
        .iter()
        .map(|&x| scirs2_special::erfc(x as f64) as f32)
        .collect();

    Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())
}

/// Wrapper for the scaled complementary error function using SciRS2
///
/// Computes erfcx(x) = exp(x²) * erfc(x).
pub fn erfcx(input: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let data = input.data()?;
    let result_data: Vec<f32> = data
        .iter()
        .map(|&x| {
            // For erfcx, we need exp(x²) * erfc(x)
            let x_f64 = x as f64;
            ((x_f64 * x_f64).exp() * scirs2_special::erfc(x_f64)) as f32
        })
        .collect();

    Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())
}

/// Wrapper for the inverse error function using SciRS2
///
/// Computes the inverse error function erf⁻¹(x).
pub fn erfinv(input: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let data = input.data()?;
    let result_data: Vec<f32> = data
        .iter()
        .map(|&x| scirs2_special::erfinv(x as f64) as f32)
        .collect();

    Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())
}

/// Wrapper for Bessel function J₀ using SciRS2
///
/// Computes the Bessel function of the first kind of order 0.
pub fn bessel_j0_scirs2(input: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    crate::bessel::bessel_j0(input)
}

/// Wrapper for Bessel function J₁ using SciRS2
///
/// Computes the Bessel function of the first kind of order 1.
pub fn bessel_j1_scirs2(input: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    crate::bessel::bessel_j1(input)
}

/// Wrapper for Bessel function Jₙ using SciRS2
///
/// Computes the Bessel function of the first kind of order n.
pub fn bessel_jn_scirs2(n: i32, input: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    crate::bessel::bessel_jn(n, input)
}

/// Wrapper for Bessel function Y₀ using SciRS2
///
/// Computes the Bessel function of the second kind of order 0.
pub fn bessel_y0_scirs2(input: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    crate::bessel::bessel_y0(input)
}

/// Wrapper for Bessel function Y₁ using SciRS2
///
/// Computes the Bessel function of the second kind of order 1.
pub fn bessel_y1_scirs2(input: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    crate::bessel::bessel_y1(input)
}

/// Wrapper for Bessel function Yₙ using SciRS2
///
/// Computes the Bessel function of the second kind of order n.
pub fn bessel_yn_scirs2(n: i32, input: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    // Now use the general bessel_yn function for all orders
    crate::bessel::bessel_yn(n, input)
}

/// Wrapper for modified Bessel function I₀ using SciRS2
///
/// Computes the modified Bessel function of the first kind of order 0.
pub fn bessel_i0_scirs2(input: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    crate::bessel::bessel_i0(input)
}

/// Wrapper for modified Bessel function I₁ using SciRS2
///
/// Computes the modified Bessel function of the first kind of order 1.
pub fn bessel_i1_scirs2(input: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    crate::bessel::bessel_i1(input)
}

/// Wrapper for modified Bessel function K₀ using SciRS2
///
/// Computes the modified Bessel function of the second kind of order 0.
pub fn bessel_k0_scirs2(input: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    crate::bessel::bessel_k0(input)
}

/// Wrapper for modified Bessel function K₁ using SciRS2
///
/// Computes the modified Bessel function of the second kind of order 1.
pub fn bessel_k1_scirs2(input: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    crate::bessel::bessel_k1(input)
}

/// Wrapper for the sinc function using SciRS2
///
/// Computes the normalized sinc function sinc(x) = sin(πx)/(πx).
pub fn sinc(input: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let data = input.data()?;
    let result_data: Vec<f32> = data
        .iter()
        .map(|&x| scirs2_special::sinc(x as f64) as f32)
        .collect();

    Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())
}

/// Wrapper for Fresnel sine integral using SciRS2
///
/// Computes the Fresnel sine integral S(x) = ∫₀^x sin(πt²/2) dt.
pub fn fresnel_s(input: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let data = input.data()?;
    let result_data: Vec<f32> = data
        .iter()
        .map(|&x| {
            let (s, _c) = scirs2_special::fresnel(x as f64).unwrap_or((0.0, 0.0));
            s as f32
        })
        .collect();

    Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())
}

/// Wrapper for Fresnel cosine integral using SciRS2
///
/// Computes the Fresnel cosine integral C(x) = ∫₀^x cos(πt²/2) dt.
pub fn fresnel_c(input: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let data = input.data()?;
    let result_data: Vec<f32> = data
        .iter()
        .map(|&x| {
            let (_s, c) = scirs2_special::fresnel(x as f64).unwrap_or((0.0, 0.0));
            c as f32
        })
        .collect();

    Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())
}

/// Compute both Fresnel integrals S(x) and C(x) simultaneously
///
/// Returns a tuple (S(x), C(x)) where:
/// - S(x) = ∫₀^x sin(πt²/2) dt (Fresnel sine integral)
/// - C(x) = ∫₀^x cos(πt²/2) dt (Fresnel cosine integral)
pub fn fresnel(input: &Tensor<f32>) -> TorshResult<(Tensor<f32>, Tensor<f32>)> {
    let data = input.data()?;
    let mut s_data = Vec::with_capacity(data.len());
    let mut c_data = Vec::with_capacity(data.len());

    for &x in data.iter() {
        let (s, c) = scirs2_special::fresnel(x as f64).unwrap_or((0.0, 0.0));
        s_data.push(s as f32);
        c_data.push(c as f32);
    }

    let s_tensor = Tensor::from_data(s_data, input.shape().dims().to_vec(), input.device())?;
    let c_tensor = Tensor::from_data(c_data, input.shape().dims().to_vec(), input.device())?;

    Ok((s_tensor, c_tensor))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use torsh_tensor::tensor;

    #[test]
    fn test_gamma_function() -> TorshResult<()> {
        let input = tensor![1.0f32, 2.0, 3.0]?;
        let result = gamma(&input)?;
        let data = result.data()?;

        // Γ(1) = 1, Γ(2) = 1, Γ(3) = 2
        assert_relative_eq!(data[0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(data[1], 1.0, epsilon = 1e-6);
        assert_relative_eq!(data[2], 2.0, epsilon = 1e-6);
        Ok(())
    }

    #[test]
    fn test_error_function() -> TorshResult<()> {
        let input = tensor![0.0f32, 1.0, -1.0]?;
        let result = erf(&input)?;
        let data = result.data()?;

        // erf(0) = 0, erf(1) ≈ 0.8427, erf(-1) ≈ -0.8427
        assert_relative_eq!(data[0], 0.0, epsilon = 1e-6);
        assert_relative_eq!(data[1], 0.8427007, epsilon = 1e-6);
        assert_relative_eq!(data[2], -0.8427007, epsilon = 1e-6);
        Ok(())
    }

    #[test]
    fn test_bessel_j0() -> TorshResult<()> {
        let input = tensor![0.0f32, 1.0]?;
        let result = bessel_j0_scirs2(&input)?;
        let data = result.data()?;

        // J₀(0) = 1, J₀(1) ≈ 0.7652
        assert_relative_eq!(data[0], 1.0, epsilon = 1e-3);
        assert_relative_eq!(data[1], 0.7652, epsilon = 1e-3);
        Ok(())
    }

    #[test]
    fn test_fresnel_integrals() -> TorshResult<()> {
        let input = tensor![0.0f32, 1.0]?;
        let (s_result, c_result) = fresnel(&input)?;
        let s_data = s_result.data()?;
        let c_data = c_result.data()?;

        // S(0) = 0, C(0) = 0
        assert_relative_eq!(s_data[0], 0.0, epsilon = 1e-6);
        assert_relative_eq!(c_data[0], 0.0, epsilon = 1e-6);

        // S(1) and C(1) have known values
        assert!(s_data[1] > 0.0);
        assert!(c_data[1] > 0.0);
        Ok(())
    }
}
