//! Error functions

use crate::TorshResult;
use torsh_tensor::Tensor;

/// Error function
pub fn erf(x: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let data = x.data()?;
    let result_data: Vec<f32> = data
        .iter()
        .map(|&val| erf_scalar(val as f64) as f32)
        .collect();

    Tensor::from_data(result_data, x.shape().dims().to_vec(), x.device())
}

/// Complementary error function (1 - erf(x))
pub fn erfc(x: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let data = x.data()?;
    let result_data: Vec<f32> = data
        .iter()
        .map(|&val| erfc_scalar(val as f64) as f32)
        .collect();

    Tensor::from_data(result_data, x.shape().dims().to_vec(), x.device())
}

/// Inverse error function
pub fn erfinv(x: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let data = x.data()?;
    let result_data: Vec<f32> = data
        .iter()
        .map(|&val| erfinv_scalar(val as f64) as f32)
        .collect();

    Tensor::from_data(result_data, x.shape().dims().to_vec(), x.device())
}

fn erf_scalar(x: f64) -> f64 {
    // Abramowitz & Stegun approximation
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

fn erfc_scalar(x: f64) -> f64 {
    1.0 - erf_scalar(x)
}

/// Scaled complementary error function (erfcx(x) = exp(x^2) * erfc(x))
pub fn erfcx(x: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let data = x.data()?;
    let result_data: Vec<f32> = data
        .iter()
        .map(|&val| erfcx_scalar(val as f64) as f32)
        .collect();

    Tensor::from_data(result_data, x.shape().dims().to_vec(), x.device())
}

/// Fresnel sine integral
pub fn fresnel_s(x: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let data = x.data()?;
    let result_data: Vec<f32> = data
        .iter()
        .map(|&val| fresnel_s_scalar(val as f64) as f32)
        .collect();

    Tensor::from_data(result_data, x.shape().dims().to_vec(), x.device())
}

/// Fresnel cosine integral
pub fn fresnel_c(x: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let data = x.data()?;
    let result_data: Vec<f32> = data
        .iter()
        .map(|&val| fresnel_c_scalar(val as f64) as f32)
        .collect();

    Tensor::from_data(result_data, x.shape().dims().to_vec(), x.device())
}

fn erfinv_scalar(x: f64) -> f64 {
    if x.abs() >= 1.0 {
        return if x > 0.0 {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        };
    }
    if x == 0.0 {
        return 0.0;
    }

    // Simple approximation for now
    let a = 0.147;
    let ln_term = (1.0 - x * x).ln();
    let first_term = 2.0 / (std::f64::consts::PI * a) + ln_term / 2.0;
    let second_term = ln_term / a;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    sign * ((first_term * first_term - second_term).sqrt() - first_term)
}

fn erfcx_scalar(x: f64) -> f64 {
    // erfcx(x) = exp(x^2) * erfc(x)
    if x >= 0.0 {
        // For positive x, use the continued fraction expansion
        if x < 1.0 {
            (x * x).exp() * erfc_scalar(x)
        } else {
            // Asymptotic expansion for large x
            let x2 = x * x;
            let sqrt_pi = std::f64::consts::PI.sqrt();
            1.0 / (sqrt_pi * x) * (1.0 - 1.0 / (2.0 * x2) + 3.0 / (4.0 * x2 * x2))
        }
    } else {
        // For negative x
        let pos_result = erfcx_scalar(-x);
        2.0 * (x * x).exp() - pos_result
    }
}

fn fresnel_s_scalar(x: f64) -> f64 {
    // S(x) = integral from 0 to x of sin(π*t²/2) dt
    let abs_x = x.abs();

    if abs_x < 1e-10 {
        return 0.0;
    }

    // Use series expansion for small x
    if abs_x < 1.0 {
        let x3 = x * x * x;
        let pi_2 = std::f64::consts::PI / 2.0;
        let term1 = x3 * pi_2 / 6.0;
        let term2 = x3 * x3 * x * (pi_2).powi(3) / (336.0 * 6.0);
        let result = term1 - term2;
        return if x >= 0.0 { result } else { -result };
    }

    // For larger x, use asymptotic expansion
    let pi_x2_2 = std::f64::consts::PI * x * x / 2.0;
    let cos_term = pi_x2_2.cos();
    let sin_term = pi_x2_2.sin();
    let pi_x = std::f64::consts::PI * abs_x;

    let result = 0.5 + sin_term / pi_x - cos_term / (std::f64::consts::PI * abs_x);
    if x >= 0.0 {
        result
    } else {
        -result
    }
}

fn fresnel_c_scalar(x: f64) -> f64 {
    // C(x) = integral from 0 to x of cos(π*t²/2) dt
    let abs_x = x.abs();

    if abs_x < 1e-10 {
        return 0.0;
    }

    // Use series expansion for small x
    if abs_x < 1.0 {
        let x2 = x * x;
        let pi_2 = std::f64::consts::PI / 2.0;
        let term1 = x - x2 * x2 * x * pi_2 / 40.0;
        return if x >= 0.0 { term1 } else { -term1 };
    }

    // For larger x, use asymptotic expansion
    let pi_x2_2 = std::f64::consts::PI * x * x / 2.0;
    let cos_term = pi_x2_2.cos();
    let sin_term = pi_x2_2.sin();
    let pi_x = std::f64::consts::PI * abs_x;

    let result = 0.5 + cos_term / pi_x + sin_term / (std::f64::consts::PI * abs_x);
    if x >= 0.0 {
        result
    } else {
        -result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use torsh_core::device::DeviceType;
    use torsh_tensor::Tensor;

    #[test]
    fn test_erf() -> TorshResult<()> {
        let device = DeviceType::Cpu;
        let x = Tensor::from_data(vec![0.0, 1.0, -1.0], vec![3], device)?;
        let result = erf(&x)?;
        let data = result.data()?;

        // Known values: erf(0) = 0, erf(1) ≈ 0.8427, erf(-1) ≈ -0.8427
        assert_relative_eq!(data[0], 0.0, epsilon = 1e-4);
        assert_relative_eq!(data[1], 0.8427, epsilon = 1e-3);
        assert_relative_eq!(data[2], -0.8427, epsilon = 1e-3);
        Ok(())
    }

    #[test]
    fn test_erfc() -> TorshResult<()> {
        let device = DeviceType::Cpu;
        let x = Tensor::from_data(vec![0.0, 1.0, -1.0], vec![3], device)?;
        let result = erfc(&x)?;
        let data = result.data()?;

        // Known values: erfc(0) = 1, erfc(1) ≈ 0.1573, erfc(-1) ≈ 1.8427
        assert_relative_eq!(data[0], 1.0, epsilon = 1e-4);
        assert_relative_eq!(data[1], 0.1573, epsilon = 1e-3);
        assert_relative_eq!(data[2], 1.8427, epsilon = 1e-3);
        Ok(())
    }

    #[test]
    #[allow(dead_code)]
    fn test_erfcx() -> TorshResult<()> {
        let device = DeviceType::Cpu;
        let x = Tensor::from_data(vec![0.0, 1.0, 2.0], vec![3], device)?;
        let result = erfcx(&x)?;
        let data = result.data()?;

        // erfcx(0) = 1, erfcx should be positive and decreasing for positive x
        assert_relative_eq!(data[0], 1.0, epsilon = 1e-4);
        assert!(data[1] > 0.0);
        assert!(data[2] > 0.0);
        assert!(data[1] > data[2]); // Should be decreasing
        Ok(())
    }

    #[test]
    #[allow(dead_code)]
    fn test_erfinv() -> TorshResult<()> {
        let device = DeviceType::Cpu;
        let x = Tensor::from_data(vec![0.0, 0.5, -0.5], vec![3], device)?;
        let result = erfinv(&x)?;
        let data = result.data()?;

        // erfinv(0) = 0, erfinv should be antisymmetric
        assert_relative_eq!(data[0], 0.0, epsilon = 1e-4);
        assert!(data[1] > 0.0);
        assert!(data[2] < 0.0);
        assert_relative_eq!(data[1], -data[2], epsilon = 1e-4);
        Ok(())
    }

    #[test]
    #[allow(dead_code)]
    fn test_fresnel_integrals() -> TorshResult<()> {
        let device = DeviceType::Cpu;
        let x = Tensor::from_data(vec![0.0, 1.0, -1.0], vec![3], device)?;

        let s_result = fresnel_s(&x)?;
        let c_result = fresnel_c(&x)?;

        let s_data = s_result.data()?;
        let c_data = c_result.data()?;

        // Fresnel integrals at x=0 should be 0
        assert_relative_eq!(s_data[0], 0.0, epsilon = 1e-6);
        assert_relative_eq!(c_data[0], 0.0, epsilon = 1e-6);

        // Fresnel integrals should be antisymmetric
        assert_relative_eq!(s_data[1], -s_data[2], epsilon = 1e-4);
        assert_relative_eq!(c_data[1], -c_data[2], epsilon = 1e-4);
        Ok(())
    }

    #[test]
    #[allow(dead_code)]
    fn test_erf_erfc_complement() -> TorshResult<()> {
        let device = DeviceType::Cpu;
        let x = Tensor::from_data(vec![0.5, 1.0, 1.5], vec![3], device)?;

        let erf_result = erf(&x)?;
        let erfc_result = erfc(&x)?;

        let erf_data = erf_result.data()?;
        let erfc_data = erfc_result.data()?;

        // Test that erf(x) + erfc(x) = 1
        for i in 0..erf_data.len() {
            assert_relative_eq!(erf_data[i] + erfc_data[i], 1.0, epsilon = 1e-6);
        }
        Ok(())
    }
}
