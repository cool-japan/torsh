//! Complex Bessel functions
//!
//! This module provides implementations of complex Bessel functions of the
//! first and second kind with proper handling of branch cuts and numerical stability.

use scirs2_core::ComplexFloat; // SciRS2 POLICY compliant (for Complex trait methods)
use std::f64::consts::PI;
use torsh_core::dtype::{Complex32, Complex64};
use torsh_tensor::Tensor;

use super::gamma::lanczos_gamma;
use crate::TorshResult;

/// Complex Bessel function of the first kind J_ν(z)
///
/// Uses series expansion for small |z| and asymptotic expansion for large |z|
pub fn complex_bessel_j_c64(nu: f64, input: &Tensor<Complex64>) -> TorshResult<Tensor<Complex64>> {
    let data = input.data()?;
    let result_data: Vec<Complex64> = data.iter().map(|&z| complex_bessel_j_main(nu, z)).collect();

    Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())
}

/// Complex Bessel function for Complex32
pub fn complex_bessel_j_c32(nu: f32, input: &Tensor<Complex32>) -> TorshResult<Tensor<Complex32>> {
    let data = input.data()?;
    let c64_data: Vec<Complex64> = data
        .iter()
        .map(|&z| Complex64::new(z.re as f64, z.im as f64))
        .collect();

    let c64_tensor = Tensor::from_data(c64_data, input.shape().dims().to_vec(), input.device())?;

    let result_c64 = complex_bessel_j_c64(nu as f64, &c64_tensor)?;
    let result_data = result_c64.data()?;
    let result_c32: Vec<Complex32> = result_data
        .iter()
        .map(|&z| Complex32::new(z.re as f32, z.im as f32))
        .collect();

    Tensor::from_data(result_c32, input.shape().dims().to_vec(), input.device())
}

/// Complex Bessel function of the second kind Y_ν(z)
pub fn complex_bessel_y_c64(nu: f64, input: &Tensor<Complex64>) -> TorshResult<Tensor<Complex64>> {
    let data = input.data()?;
    let result_data: Vec<Complex64> = data.iter().map(|&z| complex_bessel_y_main(nu, z)).collect();

    Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())
}

/// Complex Bessel function Y for Complex32
pub fn complex_bessel_y_c32(nu: f32, input: &Tensor<Complex32>) -> TorshResult<Tensor<Complex32>> {
    let data = input.data()?;
    let c64_data: Vec<Complex64> = data
        .iter()
        .map(|&z| Complex64::new(z.re as f64, z.im as f64))
        .collect();

    let c64_tensor = Tensor::from_data(c64_data, input.shape().dims().to_vec(), input.device())?;

    let result_c64 = complex_bessel_y_c64(nu as f64, &c64_tensor)?;
    let result_data = result_c64.data()?;
    let result_c32: Vec<Complex32> = result_data
        .iter()
        .map(|&z| Complex32::new(z.re as f32, z.im as f32))
        .collect();

    Tensor::from_data(result_c32, input.shape().dims().to_vec(), input.device())
}

/// Helper function for complex Bessel J implementation
fn complex_bessel_j_main(nu: f64, z: Complex64) -> Complex64 {
    let abs_z = z.norm();

    // Handle the special case z = 0
    if abs_z == 0.0 {
        return if nu == 0.0 {
            Complex64::new(1.0, 0.0)
        } else {
            Complex64::new(0.0, 0.0)
        };
    }

    // For real inputs with zero imaginary part, use the accurate real implementation
    if z.im.abs() < 1e-14 && nu.fract() == 0.0 && (0.0..100.0).contains(&nu) {
        let real_val = z.re;
        let nu_int = nu as i32;
        let result_real = if nu_int == 0 {
            j0_scalar_accurate(real_val)
        } else if nu_int == 1 {
            j1_scalar_accurate(real_val)
        } else {
            jn_scalar_accurate(nu_int, real_val)
        };
        return Complex64::new(result_real, 0.0);
    }

    if abs_z < 10.0 {
        // Series expansion: J_ν(z) = (z/2)^ν * Σ[k=0 to ∞] (-1)^k / (k! * Γ(ν+k+1)) * (z/2)^(2k)
        let z_half = z / Complex64::new(2.0, 0.0);
        let z_half_nu = z_half.powc(Complex64::new(nu, 0.0));
        let z_half_squared = z_half * z_half;

        let mut sum = Complex64::new(1.0, 0.0);
        let mut term = Complex64::new(1.0, 0.0);
        let mut factorial = 1.0;

        for k in 1..50 {
            factorial *= k as f64;
            term = -term * z_half_squared / Complex64::new(factorial, 0.0);
            let gamma_term = lanczos_gamma(Complex64::new(nu + k as f64 + 1.0, 0.0));
            let series_term = term / gamma_term * lanczos_gamma(Complex64::new(nu + 1.0, 0.0));
            sum += series_term;

            if series_term.norm() < 1e-15 {
                break;
            }
        }

        z_half_nu * sum / lanczos_gamma(Complex64::new(nu + 1.0, 0.0))
    } else {
        // Asymptotic expansion for large |z|
        let sqrt_2_pi_z = (Complex64::new(2.0 / PI, 0.0) / z).sqrt();
        let phase = z - Complex64::new(nu * PI / 2.0 + PI / 4.0, 0.0);
        sqrt_2_pi_z * (Complex64::new(0.0, 1.0) * phase).exp()
    }
}

// Accurate real Bessel functions (from the main bessel module)
fn j0_scalar_accurate(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 8.0 {
        let y = x * x;
        let ans1 = 57568490574.0
            + y * (-13362590354.0
                + y * (651619640.7 + y * (-11214424.18 + y * (77392.33017 + y * (-184.9052456)))));
        let ans2 = 57568490411.0
            + y * (1029532985.0 + y * (9494680.718 + y * (59272.64853 + y * (267.8532712 + y))));
        ans1 / ans2
    } else {
        let z = 8.0 / ax;
        let y = z * z;
        let xx = ax - 0.785398164;
        let ans1 = 1.0
            + y * (-0.1098628627e-2
                + y * (0.2734510407e-4 + y * (-0.2073370639e-5 + y * 0.2093887211e-6)));
        let ans2 = -0.1562499995e-1
            + y * (0.1430488765e-3
                + y * (-0.6911147651e-5 + y * (0.7621095161e-6 - y * 0.934945152e-7)));
        (2.0 / PI / ax).sqrt() * (ans1 * xx.cos() - z * ans2 * xx.sin())
    }
}

fn j1_scalar_accurate(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 8.0 {
        let y = x * x;
        let ans1 = x
            * (72362614232.0
                + y * (-7895059235.0
                    + y * (242396853.1
                        + y * (-2972611.439 + y * (15704.48260 + y * (-30.16036606))))));
        let ans2 = 144725228442.0
            + y * (2300535178.0 + y * (18583304.74 + y * (99447.43394 + y * (376.9991397 + y))));
        ans1 / ans2
    } else {
        let z = 8.0 / ax;
        let y = z * z;
        let xx = ax - 2.356194491;
        let ans1 = 1.0
            + y * (0.183105e-2
                + y * (-0.3516396496e-4 + y * (0.2457520174e-5 + y * (-0.240337019e-6))));
        let ans2 = 0.04687499995
            + y * (-0.2002690873e-3
                + y * (0.8449199096e-5 + y * (-0.88228987e-6 + y * 0.105787412e-6)));
        let result = (2.0 / PI / ax).sqrt() * (ans1 * xx.cos() - z * ans2 * xx.sin());
        if x < 0.0 {
            -result
        } else {
            result
        }
    }
}

fn jn_scalar_accurate(n: i32, x: f64) -> f64 {
    if n == 0 {
        return j0_scalar_accurate(x);
    }
    if n == 1 {
        return j1_scalar_accurate(x);
    }

    let ax = x.abs();
    if ax == 0.0 {
        return 0.0;
    }

    if n < 0 {
        let result = jn_scalar_accurate(-n, x);
        return if (-n) % 2 == 0 { result } else { -result };
    }

    if ax < n as f64 {
        // Use series expansion for small arguments
        let half_x = 0.5 * x;
        let mut sum = 0.0;
        let mut term = half_x.powi(n) / factorial(n as u32);

        sum += term;
        for k in 1..50 {
            term *= -(half_x * half_x) / ((k * (n + k)) as f64);
            sum += term;
            if term.abs() < 1e-15 {
                break;
            }
        }
        sum
    } else {
        // Use recurrence relation for larger arguments
        let mut jnm1 = j0_scalar_accurate(ax);
        let mut jn = j1_scalar_accurate(ax);

        for i in 2..=n {
            let current_n = (i - 1) as f64;
            let jnp1 = (2.0 * current_n / ax) * jn - jnm1;
            jnm1 = jn;
            jn = jnp1;
        }

        if x < 0.0 && n % 2 == 1 {
            -jn
        } else {
            jn
        }
    }
}

fn factorial(n: u32) -> f64 {
    if n == 0 || n == 1 {
        1.0
    } else {
        (2..=n).map(|i| i as f64).product()
    }
}

/// Helper function for complex Bessel Y implementation
fn complex_bessel_y_main(nu: f64, z: Complex64) -> Complex64 {
    let abs_z = z.norm();

    // Handle the special case z = 0 (Y functions have singularity at zero)
    if abs_z == 0.0 {
        return Complex64::new(f64::NEG_INFINITY, 0.0);
    }

    // For real inputs with zero imaginary part, use the accurate real implementation
    if z.im.abs() < 1e-14 && nu.fract() == 0.0 && (0.0..100.0).contains(&nu) && z.re > 0.0 {
        let real_val = z.re;
        let nu_int = nu as i32;
        let result_real = if nu_int == 0 {
            y0_scalar_accurate(real_val)
        } else if nu_int == 1 {
            y1_scalar_accurate(real_val)
        } else {
            yn_scalar_accurate(nu_int, real_val)
        };
        return Complex64::new(result_real, 0.0);
    }

    // For integer nu, use the special formula to avoid division by zero
    if nu.fract() == 0.0 {
        let n = nu as i32;
        if n >= 0 {
            // Use the limit formula for integer orders
            complex_bessel_y_integer(n, z)
        } else {
            // Y_{-n} = (-1)^n * Y_n for integer n
            let result = complex_bessel_y_integer(-n, z);
            if (-n) % 2 == 0 {
                result
            } else {
                -result
            }
        }
    } else {
        // Non-integer case: Y_ν(z) = [J_ν(z) * cos(νπ) - J_{-ν}(z)] / sin(νπ)
        let j_nu = complex_bessel_j_main(nu, z);
        let j_minus_nu = complex_bessel_j_main(-nu, z);
        let cos_nu_pi = Complex64::new((nu * PI).cos(), 0.0);
        let sin_nu_pi = Complex64::new((nu * PI).sin(), 0.0);

        (j_nu * cos_nu_pi - j_minus_nu) / sin_nu_pi
    }
}

/// Bessel function of second kind for integer orders (using limit formula)
fn complex_bessel_y_integer(n: i32, z: Complex64) -> Complex64 {
    if n == 0 {
        return complex_bessel_y0(z);
    }
    if n == 1 {
        return complex_bessel_y1(z);
    }

    // For n >= 2, use recurrence relation: Y_{n+1}(z) = (2n/z) * Y_n(z) - Y_{n-1}(z)
    let mut yn_minus_1 = complex_bessel_y0(z);
    let mut yn = complex_bessel_y1(z);

    for i in 2..=n {
        let current_n = (i - 1) as f64;
        let yn_plus_1 = Complex64::new(2.0 * current_n, 0.0) * yn / z - yn_minus_1;
        yn_minus_1 = yn;
        yn = yn_plus_1;
    }
    yn
}

/// Complex Y_0 using logarithmic derivative approach
fn complex_bessel_y0(z: Complex64) -> Complex64 {
    let j0 = complex_bessel_j_main(0.0, z);

    // Y_0(z) = (2/π) * [γ + ln(z/2)] * J_0(z) + series...
    // Simplified implementation using relation with Hankel functions
    let euler_gamma = 0.5772156649015329;
    let ln_z_half = (z / Complex64::new(2.0, 0.0)).ln();
    let log_term = Complex64::new(euler_gamma, 0.0) + ln_z_half;

    Complex64::new(2.0 / PI, 0.0) * log_term * j0
        - Complex64::new(2.0 / PI, 0.0) * bessel_y0_series(z)
}

/// Complex Y_1
fn complex_bessel_y1(z: Complex64) -> Complex64 {
    let j1 = complex_bessel_j_main(1.0, z);

    // Simplified Y_1 implementation
    let euler_gamma = 0.5772156649015329;
    let ln_z_half = (z / Complex64::new(2.0, 0.0)).ln();
    let log_term = Complex64::new(euler_gamma, 0.0) + ln_z_half - Complex64::new(0.5, 0.0);

    Complex64::new(2.0 / PI, 0.0) * log_term * j1
        - Complex64::new(2.0 / PI, 0.0) * z.recip()
        - Complex64::new(2.0 / PI, 0.0) * bessel_y1_series(z)
}

/// Series part for Y_0
fn bessel_y0_series(z: Complex64) -> Complex64 {
    let z_half = z / Complex64::new(2.0, 0.0);
    let z_half_squared = z_half * z_half;

    let mut sum = Complex64::new(0.0, 0.0);
    let mut term = z_half_squared / Complex64::new(4.0, 0.0); // k=1 term

    for k in 1..20 {
        let harmonic_k = (1..=k).map(|i| 1.0 / i as f64).sum::<f64>();
        let k_fact = factorial(k as u32);

        sum += term * Complex64::new(harmonic_k, 0.0) / Complex64::new(k_fact, 0.0);
        term *= -z_half_squared / Complex64::new((k + 1) as f64, 0.0);

        if term.norm() < 1e-15 {
            break;
        }
    }
    sum
}

/// Series part for Y_1
fn bessel_y1_series(z: Complex64) -> Complex64 {
    let z_half = z / Complex64::new(2.0, 0.0);
    let z_half_squared = z_half * z_half;

    let mut sum = Complex64::new(0.0, 0.0);
    let mut term = z_half_squared / Complex64::new(8.0, 0.0); // k=1 term, adjusted for Y_1

    for k in 1..20 {
        let harmonic_k = (1..=k).map(|i| 1.0 / i as f64).sum::<f64>();
        let harmonic_k_plus_1 = harmonic_k + 1.0 / ((k + 1) as f64);
        let k_fact = factorial(k as u32);

        sum += term * Complex64::new(harmonic_k + harmonic_k_plus_1, 0.0)
            / Complex64::new(k_fact * (k + 1) as f64, 0.0);
        term *= -z_half_squared / Complex64::new((k + 1) as f64, 0.0);

        if term.norm() < 1e-15 {
            break;
        }
    }
    sum * z_half
}

// Real Y functions for accurate fallback
fn y0_scalar_accurate(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NAN;
    }

    let ax = x.abs();
    if ax < 8.0 {
        let y = x * x;
        let ans1 = -2957821389.0
            + y * (7062834065.0
                + y * (-512359803.6 + y * (10879881.29 + y * (-86327.92757 + y * 228.4622733))));
        let ans2 = 40076544269.0
            + y * (745249964.8 + y * (7189466.438 + y * (47447.26470 + y * (226.1030244 + y))));
        ans1 / ans2 + (2.0 / PI) * j0_scalar_accurate(x) * x.ln()
    } else {
        let z = 8.0 / ax;
        let y = z * z;
        let xx = ax - 0.785398164;
        let ans1 = 1.0
            + y * (-0.1098628627e-2
                + y * (0.2734510407e-4 + y * (-0.2073370639e-5 + y * 0.2093887211e-6)));
        let ans2 = -0.1562499995e-1
            + y * (0.1430488765e-3
                + y * (-0.6911147651e-5 + y * (0.7621095161e-6 - y * 0.934945152e-7)));
        (2.0 / PI / ax).sqrt() * (ans1 * xx.sin() + z * ans2 * xx.cos())
    }
}

fn y1_scalar_accurate(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NAN;
    }

    let ax = x.abs();
    if ax < 8.0 {
        let y = x * x;
        let ans1 = x
            * (-0.4900604943e13
                + y * (0.1275274390e13
                    + y * (-0.5153438139e11
                        + y * (0.7349264551e9 + y * (-0.4237922726e7 + y * 0.8511937935e4)))));
        let ans2 = 0.2499580570e14
            + y * (0.4244419664e12
                + y * (0.3733650367e10
                    + y * (0.2245904002e8 + y * (0.1020426050e6 + y * (0.3549632885e3 + y)))));
        ans1 / ans2 + (2.0 / PI) * (j1_scalar_accurate(x) * x.ln() - 1.0 / x)
    } else {
        let z = 8.0 / ax;
        let y = z * z;
        let xx = ax - 2.356194491;
        let ans1 = 1.0
            + y * (0.183105e-2
                + y * (-0.3516396496e-4 + y * (0.2457520174e-5 + y * (-0.240337019e-6))));
        let ans2 = 0.04687499995
            + y * (-0.2002690873e-3
                + y * (0.8449199096e-5 + y * (-0.88228987e-6 + y * 0.105787412e-6)));
        (2.0 / PI / ax).sqrt() * (ans1 * xx.sin() - z * ans2 * xx.cos())
    }
}

fn yn_scalar_accurate(n: i32, x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NAN;
    }

    if n == 0 {
        return y0_scalar_accurate(x);
    }
    if n == 1 {
        return y1_scalar_accurate(x);
    }

    if n < 0 {
        // Y_{-n}(x) = (-1)^n * Y_n(x)
        let result = yn_scalar_accurate(-n, x);
        if (-n) % 2 == 0 {
            result
        } else {
            -result
        }
    } else {
        // Forward recurrence: Y_{n+1}(x) = (2n/x) * Y_n(x) - Y_{n-1}(x)
        let mut ynm1 = y0_scalar_accurate(x);
        let mut yn = y1_scalar_accurate(x);

        for i in 2..=n {
            let current_n = (i - 1) as f64;
            let ynp1 = (2.0 * current_n / x) * yn - ynm1;
            ynm1 = yn;
            yn = ynp1;
        }
        yn
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use torsh_core::device::DeviceType;

    #[test]
    fn test_complex_bessel_j0_c64() {
        // Test J_0(0) = 1
        let input_data = vec![Complex64::new(0.0, 0.0)];
        let input = Tensor::from_data(input_data, vec![1], DeviceType::Cpu).unwrap();
        let result = complex_bessel_j_c64(0.0, &input).unwrap();
        let data = result.data().unwrap();

        assert_relative_eq!(data[0].re, 1.0, max_relative = 1e-6);
        assert_relative_eq!(data[0].im, 0.0, max_relative = 1e-6);
    }

    #[test]
    fn test_complex_bessel_j1_c64() {
        // Test J_1(0) = 0
        let input_data = vec![Complex64::new(0.0, 0.0)];
        let input = Tensor::from_data(input_data, vec![1], DeviceType::Cpu).unwrap();
        let result = complex_bessel_j_c64(1.0, &input).unwrap();
        let data = result.data().unwrap();

        assert_relative_eq!(data[0].re, 0.0, max_relative = 1e-6);
        assert_relative_eq!(data[0].im, 0.0, max_relative = 1e-6);
    }

    #[test]
    fn test_complex_bessel_j_c32() {
        let input_data = vec![Complex32::new(1.0, 0.0)];
        let input = Tensor::from_data(input_data, vec![1], DeviceType::Cpu).unwrap();
        let result = complex_bessel_j_c32(0.0, &input).unwrap();
        let data = result.data().unwrap();

        // J_0(1) ≈ 0.7652
        assert_relative_eq!(data[0].re, 0.7652, max_relative = 1e-2);
    }

    #[test]
    fn test_complex_bessel_y_c64() {
        // Test that Y functions don't crash on basic input
        let input_data = vec![Complex64::new(1.0, 0.0)];
        let input = Tensor::from_data(input_data, vec![1], DeviceType::Cpu).unwrap();
        let result = complex_bessel_y_c64(0.0, &input).unwrap();
        let data = result.data().unwrap();

        // Y_0(1) ≈ 0.0883 (just check it's finite and reasonable)
        assert!(data[0].re.is_finite());
        assert!(data[0].im.is_finite());
    }
}
