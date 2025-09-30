//! Special trigonometric functions

use crate::TorshResult;
use std::f64::consts::PI;
use torsh_tensor::Tensor;

/// Normalized sinc function: sinc(x) = sin(π*x)/(π*x)
pub fn sinc(x: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let data = x.data()?;
    let result_data: Vec<f32> = data
        .iter()
        .map(|&val| sinc_scalar(val as f64) as f32)
        .collect();

    Tensor::from_data(result_data, x.shape().dims().to_vec(), x.device())
}

/// Unnormalized sinc function: sinc(x) = sin(x)/x
pub fn sinc_unnormalized(x: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let data = x.data()?;
    let result_data: Vec<f32> = data
        .iter()
        .map(|&val| {
            let x_val = val as f64;
            let result = if x_val.abs() < 1e-10 {
                1.0
            } else {
                x_val.sin() / x_val
            };
            result as f32
        })
        .collect();

    Tensor::from_data(result_data, x.shape().dims().to_vec(), x.device())
}

/// Spherical Bessel function of the first kind, order 0
pub fn spherical_j0(x: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let data = x.data()?;
    let result_data: Vec<f32> = data
        .iter()
        .map(|&val| {
            let x_val = val as f64;
            let result = if x_val.abs() < 1e-10 {
                1.0
            } else {
                x_val.sin() / x_val
            };
            result as f32
        })
        .collect();

    Tensor::from_data(result_data, x.shape().dims().to_vec(), x.device())
}

/// Spherical Bessel function of the first kind, order 1
pub fn spherical_j1(x: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let data = x.data()?;
    let result_data: Vec<f32> = data
        .iter()
        .map(|&val| {
            let x_val = val as f64;
            spherical_j1_scalar(x_val) as f32
        })
        .collect();

    Tensor::from_data(result_data, x.shape().dims().to_vec(), x.device())
}

/// Spherical Bessel function of the first kind, order n
pub fn spherical_jn(n: i32, x: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let data = x.data()?;
    let result_data: Vec<f32> = data
        .iter()
        .map(|&val| {
            let x_val = val as f64;
            spherical_jn_scalar(n, x_val) as f32
        })
        .collect();

    Tensor::from_data(result_data, x.shape().dims().to_vec(), x.device())
}

/// Spherical Bessel function of the second kind, order 0
pub fn spherical_y0(x: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let data = x.data()?;
    let result_data: Vec<f32> = data
        .iter()
        .map(|&val| {
            let x_val = val as f64;
            spherical_y0_scalar(x_val) as f32
        })
        .collect();

    Tensor::from_data(result_data, x.shape().dims().to_vec(), x.device())
}

/// Spherical Bessel function of the second kind, order 1
pub fn spherical_y1(x: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let data = x.data()?;
    let result_data: Vec<f32> = data
        .iter()
        .map(|&val| {
            let x_val = val as f64;
            spherical_y1_scalar(x_val) as f32
        })
        .collect();

    Tensor::from_data(result_data, x.shape().dims().to_vec(), x.device())
}

/// Spherical Bessel function of the second kind, order n
pub fn spherical_yn(n: i32, x: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let data = x.data()?;
    let result_data: Vec<f32> = data
        .iter()
        .map(|&val| {
            let x_val = val as f64;
            spherical_yn_scalar(n, x_val) as f32
        })
        .collect();

    Tensor::from_data(result_data, x.shape().dims().to_vec(), x.device())
}

fn sinc_scalar(x: f64) -> f64 {
    let pi_x = PI * x;
    if pi_x.abs() < 1e-10 {
        1.0
    } else {
        pi_x.sin() / pi_x
    }
}

fn spherical_j1_scalar(x: f64) -> f64 {
    if x.abs() < 1e-10 {
        0.0
    } else {
        (x.sin() / (x * x)) - (x.cos() / x)
    }
}

fn spherical_jn_scalar(n: i32, x: f64) -> f64 {
    if n == 0 {
        return if x.abs() < 1e-10 { 1.0 } else { x.sin() / x };
    }
    if n == 1 {
        return spherical_j1_scalar(x);
    }
    if x.abs() < 1e-10 {
        return 0.0;
    }

    // Use recurrence relation: j_{n+1}(x) = (2n+1)/x * j_n(x) - j_{n-1}(x)
    let mut jnm1 = x.sin() / x; // j_0
    let mut jn = spherical_j1_scalar(x); // j_1

    for k in 1..n {
        let jnp1 = ((2 * k + 1) as f64 / x) * jn - jnm1;
        jnm1 = jn;
        jn = jnp1;
    }
    jn
}

fn spherical_y0_scalar(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NAN;
    }
    -x.cos() / x
}

fn spherical_y1_scalar(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NAN;
    }
    -(x.cos() / (x * x)) - (x.sin() / x)
}

fn spherical_yn_scalar(n: i32, x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NAN;
    }
    if n == 0 {
        return spherical_y0_scalar(x);
    }
    if n == 1 {
        return spherical_y1_scalar(x);
    }

    // Use recurrence relation: y_{n+1}(x) = (2n+1)/x * y_n(x) - y_{n-1}(x)
    let mut ynm1 = spherical_y0_scalar(x); // y_0
    let mut yn = spherical_y1_scalar(x); // y_1

    for k in 1..n {
        let ynp1 = ((2 * k + 1) as f64 / x) * yn - ynm1;
        ynm1 = yn;
        yn = ynp1;
    }
    yn
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use torsh_core::device::DeviceType;
    use torsh_tensor::Tensor;

    #[test]
    fn test_sinc() -> TorshResult<()> {
        let device = DeviceType::Cpu;
        let x = Tensor::from_data(vec![0.0, 1.0, -1.0], vec![3], device)?;
        let result = sinc(&x)?;
        let data = result.data()?;

        // sinc(0) = 1, sinc(1) = sin(π)/π ≈ 0, sinc(-1) = sinc(1)
        assert_relative_eq!(data[0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(data[1], 0.0, epsilon = 1e-3);
        assert_relative_eq!(data[2], data[1], epsilon = 1e-6);
        Ok(())
    }

    #[test]
    fn test_sinc_unnormalized() -> TorshResult<()> {
        let device = DeviceType::Cpu;
        let x = Tensor::from_data(
            vec![0.0, std::f32::consts::PI, -std::f32::consts::PI],
            vec![3],
            device,
        )?;
        let result = sinc_unnormalized(&x)?;
        let data = result.data()?;

        // sinc(0) = 1, sinc(π) = 0, sinc(-π) = 0
        assert_relative_eq!(data[0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(data[1], 0.0, epsilon = 1e-3);
        assert_relative_eq!(data[2], 0.0, epsilon = 1e-3);
        Ok(())
    }

    #[test]
    fn test_spherical_j0() -> TorshResult<()> {
        let device = DeviceType::Cpu;
        let x = Tensor::from_data(vec![0.0, 1.0, std::f32::consts::PI], vec![3], device)?;
        let result = spherical_j0(&x).unwrap();
        let data = result.data()?;

        // j_0(0) = 1, j_0(π) = 0
        assert_relative_eq!(data[0], 1.0, epsilon = 1e-6);
        assert!(data[1] > 0.0); // j_0(1) should be positive
        assert_relative_eq!(data[2], 0.0, epsilon = 1e-3);
        Ok(())
    }

    #[test]
    #[allow(dead_code)]
    fn test_spherical_j1() -> TorshResult<()> {
        let device = DeviceType::Cpu;
        let x = Tensor::from_data(vec![0.0, 1.0, 2.0], vec![3], device)?;
        let result = spherical_j1(&x).unwrap();
        let data = result.data()?;

        // j_1(0) = 0
        assert_relative_eq!(data[0], 0.0, epsilon = 1e-6);
        assert!(data[1] > 0.0); // j_1(1) should be positive
        Ok(())
    }

    #[test]
    #[allow(dead_code)]
    fn test_spherical_y0() -> TorshResult<()> {
        let device = DeviceType::Cpu;
        let x = Tensor::from_data(vec![1.0, 2.0, std::f32::consts::PI], vec![3], device)?;
        let result = spherical_y0(&x).unwrap();
        let data = result.data()?;

        // y_0(x) = -cos(x)/x, so y_0(π/2) should be 0
        assert!(data[0] < 0.0); // y_0(1) = -cos(1) < 0
        assert!(data[1] > 0.0); // y_0(2) = -cos(2)/2 > 0 since cos(2) < 0
        Ok(())
    }

    #[test]
    #[allow(dead_code)]
    fn test_spherical_y1() -> TorshResult<()> {
        let device = DeviceType::Cpu;
        let x = Tensor::from_data(vec![1.0, 2.0], vec![2], device)?;
        let result = spherical_y1(&x).unwrap();
        let data = result.data()?;

        // y_1(x) = -(cos(x)/x^2 + sin(x)/x)
        // Both should be finite and real
        assert!(data[0].is_finite());
        assert!(data[1].is_finite());
        Ok(())
    }

    #[test]
    #[allow(dead_code)]
    fn test_spherical_jn() -> TorshResult<()> {
        let device = DeviceType::Cpu;
        let x = Tensor::from_data(vec![1.0, 2.0], vec![2], device)?;

        // Test that j_0 from spherical_jn matches spherical_j0
        let j0_direct = spherical_j0(&x).unwrap();
        let j0_from_jn = spherical_jn(0, &x).unwrap();

        let j0_direct_data = j0_direct.data()?;
        let j0_from_jn_data = j0_from_jn.data()?;

        for i in 0..j0_direct_data.len() {
            assert_relative_eq!(j0_direct_data[i], j0_from_jn_data[i], epsilon = 1e-6);
        }

        // Test that j_1 from spherical_jn matches spherical_j1
        let j1_direct = spherical_j1(&x).unwrap();
        let j1_from_jn = spherical_jn(1, &x).unwrap();

        let j1_direct_data = j1_direct.data()?;
        let j1_from_jn_data = j1_from_jn.data()?;

        for i in 0..j1_direct_data.len() {
            assert_relative_eq!(j1_direct_data[i], j1_from_jn_data[i], epsilon = 1e-6);
        }
        Ok(())
    }

    #[test]
    #[allow(dead_code)]
    fn test_spherical_yn() -> TorshResult<()> {
        let device = DeviceType::Cpu;
        let x = Tensor::from_data(vec![1.0, 2.0], vec![2], device)?;

        // Test that y_0 from spherical_yn matches spherical_y0
        let y0_direct = spherical_y0(&x)?;
        let y0_from_yn = spherical_yn(0, &x)?;

        let y0_direct_data = y0_direct.data()?;
        let y0_from_yn_data = y0_from_yn.data()?;

        for i in 0..y0_direct_data.len() {
            assert_relative_eq!(y0_direct_data[i], y0_from_yn_data[i], epsilon = 1e-6);
        }

        // Test that y_1 from spherical_yn matches spherical_y1
        let y1_direct = spherical_y1(&x)?;
        let y1_from_yn = spherical_yn(1, &x)?;

        let y1_direct_data = y1_direct.data()?;
        let y1_from_yn_data = y1_from_yn.data()?;

        for i in 0..y1_direct_data.len() {
            assert_relative_eq!(y1_direct_data[i], y1_from_yn_data[i], epsilon = 1e-6);
        }

        Ok(())
    }
}
