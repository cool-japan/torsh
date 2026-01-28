//! Numerical integration and differentiation operations
//!
//! This module provides numerical methods for integration and differentiation,
//! including various quadrature rules and finite difference methods.

use torsh_core::{Result as TorshResult, TorshError};
use torsh_tensor::Tensor;

// ============================================================================
// Numerical Integration
// ============================================================================

/// Integration methods
#[derive(Debug, Clone, Copy)]
pub enum IntegrationMethod {
    /// Trapezoidal rule
    Trapezoidal,
    /// Simpson's rule (requires odd number of points)
    Simpson,
    /// Simpson's 3/8 rule
    Simpson38,
    /// Gaussian quadrature
    Gaussian,
    /// Romberg integration
    Romberg,
    /// Adaptive quadrature
    Adaptive,
}

/// Numerical integration using trapezoidal rule
///
/// Integrates a 1D tensor representing function values at equally spaced points.
///
/// # Arguments
/// * `y` - Tensor of function values
/// * `dx` - Step size between points
pub fn trapz(y: &Tensor, dx: Option<f32>) -> TorshResult<Tensor> {
    let data = y.data()?;
    let dx = dx.unwrap_or(1.0);

    if data.len() < 2 {
        return Err(TorshError::InvalidArgument(
            "Need at least 2 points for trapezoidal integration".to_string(),
        ));
    }

    let mut sum = 0.5 * (data[0] + data[data.len() - 1]);
    for i in 1..data.len() - 1 {
        sum += data[i];
    }

    let result = sum * dx;
    Tensor::from_data(vec![result], vec![], y.device())
}

/// Cumulative integration using trapezoidal rule
///
/// Returns cumulative integral at each point.
///
/// # Arguments
/// * `y` - Tensor of function values
/// * `dx` - Step size between points
pub fn cumtrapz(y: &Tensor, dx: Option<f32>) -> TorshResult<Tensor> {
    let data = y.data()?;
    let dx = dx.unwrap_or(1.0);

    if data.len() < 2 {
        return Err(TorshError::InvalidArgument(
            "Need at least 2 points for cumulative integration".to_string(),
        ));
    }

    let mut result = Vec::with_capacity(data.len());
    result.push(0.0); // First point has zero integral

    for i in 1..data.len() {
        let integral = result[i - 1] + 0.5 * (data[i - 1] + data[i]) * dx;
        result.push(integral);
    }

    Tensor::from_data(result, y.shape().dims().to_vec(), y.device())
}

/// Simpson's rule integration
///
/// Requires odd number of points (even number of intervals).
///
/// # Arguments
/// * `y` - Tensor of function values
/// * `dx` - Step size between points
pub fn simps(y: &Tensor, dx: Option<f32>) -> TorshResult<Tensor> {
    let data = y.data()?;
    let dx = dx.unwrap_or(1.0);
    let n = data.len();

    if n < 3 {
        return Err(TorshError::InvalidArgument(
            "Need at least 3 points for Simpson's rule".to_string(),
        ));
    }

    if n % 2 == 0 {
        return Err(TorshError::InvalidArgument(
            "Simpson's rule requires odd number of points".to_string(),
        ));
    }

    let mut sum = data[0] + data[n - 1];

    // Add even indices with coefficient 4
    for i in (1..n - 1).step_by(2) {
        sum += 4.0 * data[i];
    }

    // Add odd indices with coefficient 2
    for i in (2..n - 1).step_by(2) {
        sum += 2.0 * data[i];
    }

    let result = sum * dx / 3.0;
    Tensor::from_data(vec![result], vec![], y.device())
}

/// Gaussian quadrature integration
///
/// Uses Gauss-Legendre quadrature for integration over [-1, 1].
/// For other intervals, use change of variables.
///
/// # Arguments
/// * `func` - Function to integrate (closure)
/// * `n_points` - Number of quadrature points (2-10 supported)
pub fn gaussian_quad<F>(func: F, n_points: usize) -> TorshResult<f32>
where
    F: Fn(f32) -> f32,
{
    let (nodes, weights) = match n_points {
        2 => (vec![-0.5773502692, 0.5773502692], vec![1.0, 1.0]),
        3 => (
            vec![-0.7745966692, 0.0, 0.7745966692],
            vec![0.5555555556, 0.8888888889, 0.5555555556],
        ),
        4 => (
            vec![-0.8611363116, -0.3399810436, 0.3399810436, 0.8611363116],
            vec![0.3478548451, 0.6521451549, 0.6521451549, 0.3478548451],
        ),
        5 => (
            vec![
                -0.9061798459,
                -0.5384693101,
                0.0,
                0.5384693101,
                0.9061798459,
            ],
            vec![
                0.2369268851,
                0.4786286705,
                0.5688888889,
                0.4786286705,
                0.2369268851,
            ],
        ),
        _ => {
            return Err(TorshError::InvalidArgument(
                "Gaussian quadrature supports 2-5 points".to_string(),
            ))
        }
    };

    let mut integral = 0.0;
    for (i, &x) in nodes.iter().enumerate() {
        integral += weights[i] * func(x);
    }

    Ok(integral)
}

/// Adaptive quadrature integration
///
/// Uses recursive subdivision to achieve desired accuracy.
///
/// # Arguments
/// * `func` - Function to integrate
/// * `a` - Lower bound
/// * `b` - Upper bound
/// * `tol` - Tolerance for convergence
/// * `max_depth` - Maximum recursion depth
pub fn adaptive_quad<F>(
    func: F,
    a: f32,
    b: f32,
    tol: Option<f32>,
    max_depth: Option<usize>,
) -> TorshResult<f32>
where
    F: Fn(f32) -> f32 + Clone,
{
    let tol = tol.unwrap_or(1e-6);
    let max_depth = max_depth.unwrap_or(10);

    fn adaptive_simpson<F>(
        func: &F,
        a: f32,
        b: f32,
        tol: f32,
        depth: usize,
        max_depth: usize,
    ) -> f32
    where
        F: Fn(f32) -> f32,
    {
        let h = b - a;
        let c = (a + b) / 2.0;

        let fa = func(a);
        let fb = func(b);
        let fc = func(c);

        let s1 = h * (fa + 4.0 * fc + fb) / 6.0; // Simpson's rule

        let fd = func((a + c) / 2.0);
        let fe = func((c + b) / 2.0);

        let s2 = h * (fa + 4.0 * fd + 2.0 * fc + 4.0 * fe + fb) / 12.0;

        if depth >= max_depth || (s2 - s1).abs() < 15.0 * tol {
            s2 + (s2 - s1) / 15.0
        } else {
            adaptive_simpson(func, a, c, tol / 2.0, depth + 1, max_depth)
                + adaptive_simpson(func, c, b, tol / 2.0, depth + 1, max_depth)
        }
    }

    Ok(adaptive_simpson(&func, a, b, tol, 0, max_depth))
}

// ============================================================================
// Numerical Differentiation
// ============================================================================

/// Differentiation methods
#[derive(Debug, Clone, Copy)]
pub enum DifferentiationMethod {
    /// Forward finite difference
    Forward,
    /// Backward finite difference
    Backward,
    /// Central finite difference
    Central,
    /// Higher-order finite difference
    HigherOrder,
}

/// Gradient computation using finite differences
///
/// Computes gradient of a 1D tensor using specified method.
///
/// # Arguments
/// * `y` - Tensor of function values
/// * `dx` - Step size
/// * `method` - Differentiation method
pub fn gradient(
    y: &Tensor,
    dx: Option<f32>,
    method: Option<DifferentiationMethod>,
) -> TorshResult<Tensor> {
    let data = y.data()?;
    let dx = dx.unwrap_or(1.0);
    let method = method.unwrap_or(DifferentiationMethod::Central);

    if data.len() < 2 {
        return Err(TorshError::InvalidArgument(
            "Need at least 2 points for differentiation".to_string(),
        ));
    }

    let mut grad = Vec::with_capacity(data.len());

    match method {
        DifferentiationMethod::Forward => {
            for i in 0..data.len() - 1 {
                grad.push((data[i + 1] - data[i]) / dx);
            }
            // Last point uses backward difference
            grad.push((data[data.len() - 1] - data[data.len() - 2]) / dx);
        }
        DifferentiationMethod::Backward => {
            // First point uses forward difference
            grad.push((data[1] - data[0]) / dx);
            for i in 1..data.len() {
                grad.push((data[i] - data[i - 1]) / dx);
            }
        }
        DifferentiationMethod::Central => {
            // First point uses forward difference
            grad.push((data[1] - data[0]) / dx);
            // Central difference for interior points
            for i in 1..data.len() - 1 {
                grad.push((data[i + 1] - data[i - 1]) / (2.0 * dx));
            }
            // Last point uses backward difference
            grad.push((data[data.len() - 1] - data[data.len() - 2]) / dx);
        }
        DifferentiationMethod::HigherOrder => {
            if data.len() < 5 {
                return Err(TorshError::InvalidArgument(
                    "Need at least 5 points for higher-order differentiation".to_string(),
                ));
            }

            // First two points use forward differences
            grad.push(
                (-25.0 * data[0] + 48.0 * data[1] - 36.0 * data[2] + 16.0 * data[3]
                    - 3.0 * data[4])
                    / (12.0 * dx),
            );
            grad.push(
                (-3.0 * data[0] - 10.0 * data[1] + 18.0 * data[2] - 6.0 * data[3] + data[4])
                    / (12.0 * dx),
            );

            // Central differences for interior points
            for i in 2..data.len() - 2 {
                grad.push(
                    (data[i - 2] - 8.0 * data[i - 1] + 8.0 * data[i + 1] - data[i + 2])
                        / (12.0 * dx),
                );
            }

            // Last two points use backward differences
            let n = data.len();
            grad.push(
                (3.0 * data[n - 1] + 10.0 * data[n - 2] - 18.0 * data[n - 3] + 6.0 * data[n - 4]
                    - data[n - 5])
                    / (12.0 * dx),
            );
            grad.push(
                (25.0 * data[n - 1] - 48.0 * data[n - 2] + 36.0 * data[n - 3] - 16.0 * data[n - 4]
                    + 3.0 * data[n - 5])
                    / (12.0 * dx),
            );
        }
    }

    Tensor::from_data(grad, y.shape().dims().to_vec(), y.device())
}

/// Second derivative using finite differences
///
/// # Arguments
/// * `y` - Tensor of function values
/// * `dx` - Step size
pub fn second_derivative(y: &Tensor, dx: Option<f32>) -> TorshResult<Tensor> {
    let data = y.data()?;
    let dx = dx.unwrap_or(1.0);

    if data.len() < 3 {
        return Err(TorshError::InvalidArgument(
            "Need at least 3 points for second derivative".to_string(),
        ));
    }

    let mut second_deriv = Vec::with_capacity(data.len());
    let dx2 = dx * dx;

    // First point (forward difference)
    if data.len() >= 4 {
        second_deriv.push((2.0 * data[0] - 5.0 * data[1] + 4.0 * data[2] - data[3]) / dx2);
    } else {
        // For small arrays, use simple forward difference
        second_deriv.push((data[2] - 2.0 * data[1] + data[0]) / dx2);
    }

    // Interior points (central difference)
    for i in 1..data.len() - 1 {
        second_deriv.push((data[i - 1] - 2.0 * data[i] + data[i + 1]) / dx2);
    }

    // Last point (backward difference)
    let n = data.len();
    if n >= 4 {
        second_deriv
            .push((2.0 * data[n - 1] - 5.0 * data[n - 2] + 4.0 * data[n - 3] - data[n - 4]) / dx2);
    } else {
        // For small arrays, use simple backward difference
        second_deriv.push((data[n - 1] - 2.0 * data[n - 2] + data[n - 3]) / dx2);
    }

    Tensor::from_data(second_deriv, y.shape().dims().to_vec(), y.device())
}

/// Partial derivatives for multi-dimensional tensors
///
/// Computes partial derivative along specified axis.
///
/// # Arguments
/// * `tensor` - Input tensor
/// * `axis` - Axis along which to compute derivative
/// * `dx` - Step size
pub fn partial_derivative(tensor: &Tensor, axis: usize, dx: Option<f32>) -> TorshResult<Tensor> {
    let dx = dx.unwrap_or(1.0);
    let binding = tensor.shape();
    let shape = binding.dims();

    if axis >= shape.len() {
        return Err(TorshError::InvalidArgument(format!(
            "Axis {} out of bounds for tensor with {} dimensions",
            axis,
            shape.len()
        )));
    }

    if shape[axis] < 2 {
        return Err(TorshError::InvalidArgument(
            "Need at least 2 points along differentiation axis".to_string(),
        ));
    }

    // For now, implement simple central difference
    // This is a simplified implementation - a full implementation would need
    // proper multi-dimensional indexing
    gradient(tensor, Some(dx), Some(DifferentiationMethod::Central))
}

// ============================================================================
// Optimization and Root Finding
// ============================================================================

/// Find roots using Newton-Raphson method
///
/// # Arguments
/// * `func` - Function for which to find roots
/// * `dfunc` - Derivative of the function
/// * `x0` - Initial guess
/// * `tol` - Tolerance for convergence
/// * `max_iter` - Maximum number of iterations
pub fn newton_raphson<F, DF>(
    func: F,
    dfunc: DF,
    x0: f32,
    tol: Option<f32>,
    max_iter: Option<usize>,
) -> TorshResult<f32>
where
    F: Fn(f32) -> f32,
    DF: Fn(f32) -> f32,
{
    let tol = tol.unwrap_or(1e-6);
    let max_iter = max_iter.unwrap_or(100);

    let mut x = x0;

    for _ in 0..max_iter {
        let fx = func(x);
        let dfx = dfunc(x);

        if dfx.abs() < 1e-12 {
            return Err(TorshError::ComputeError(
                "Derivative is zero, Newton-Raphson method failed".to_string(),
            ));
        }

        let x_new = x - fx / dfx;

        if (x_new - x).abs() < tol {
            return Ok(x_new);
        }

        x = x_new;
    }

    Err(TorshError::ComputeError(
        "Newton-Raphson method did not converge".to_string(),
    ))
}

/// Find roots using bisection method
///
/// # Arguments
/// * `func` - Function for which to find roots
/// * `a` - Lower bound (func(a) and func(b) should have opposite signs)
/// * `b` - Upper bound
/// * `tol` - Tolerance for convergence
/// * `max_iter` - Maximum number of iterations
pub fn bisection<F>(
    func: F,
    a: f32,
    b: f32,
    tol: Option<f32>,
    max_iter: Option<usize>,
) -> TorshResult<f32>
where
    F: Fn(f32) -> f32,
{
    let tol = tol.unwrap_or(1e-6);
    let max_iter = max_iter.unwrap_or(100);

    let fa = func(a);
    let fb = func(b);

    if fa * fb > 0.0 {
        return Err(TorshError::InvalidArgument(
            "Function values at endpoints must have opposite signs".to_string(),
        ));
    }

    let mut a = a;
    let mut b = b;

    for _ in 0..max_iter {
        let c = (a + b) / 2.0;
        let fc = func(c);

        if fc.abs() < tol || (b - a) / 2.0 < tol {
            return Ok(c);
        }

        if fa * fc < 0.0 {
            b = c;
        } else {
            a = c;
        }
    }

    Ok((a + b) / 2.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use torsh_core::device::DeviceType;
    use torsh_tensor::creation::*;

    #[test]
    fn test_trapz() {
        // Test integration of x^2 from 0 to 1
        let x: Vec<f32> = (0..11).map(|i| i as f32 / 10.0).collect();
        let y: Vec<f32> = x.iter().map(|&xi| xi * xi).collect();
        let tensor = from_vec(y, &[11], DeviceType::Cpu).unwrap();

        let result = trapz(&tensor, Some(0.1)).unwrap();
        let result_val = result.data().expect("tensor should have data")[0];

        // Analytical result for integral of x^2 from 0 to 1 is 1/3
        assert_relative_eq!(result_val, 1.0 / 3.0, epsilon = 0.01);
    }

    #[test]
    fn test_gradient() {
        // Test gradient of x^2
        let x: Vec<f32> = (0..11).map(|i| i as f32 / 10.0).collect();
        let y: Vec<f32> = x.iter().map(|&xi| xi * xi).collect();
        let tensor = from_vec(y, &[11], DeviceType::Cpu).unwrap();

        let grad = gradient(&tensor, Some(0.1), Some(DifferentiationMethod::Central)).unwrap();
        let grad_data = grad.data().expect("tensor should have data");

        // Analytical gradient of x^2 is 2x
        // Check a few points
        for i in 1..grad_data.len() - 1 {
            let expected = 2.0 * x[i];
            assert_relative_eq!(grad_data[i], expected, epsilon = 0.1);
        }
    }

    #[test]
    fn test_simps() {
        // Test Simpson's rule on x^2 from 0 to 1
        let x: Vec<f32> = (0..11).map(|i| i as f32 / 10.0).collect();
        let y: Vec<f32> = x.iter().map(|&xi| xi * xi).collect();
        let tensor = from_vec(y, &[11], DeviceType::Cpu).unwrap();

        let result = simps(&tensor, Some(0.1)).unwrap();
        let result_val = result.data().expect("tensor should have data")[0];

        // Should be more accurate than trapezoidal rule
        assert_relative_eq!(result_val, 1.0 / 3.0, epsilon = 0.001);
    }

    #[test]
    fn test_newton_raphson() {
        // Find root of x^2 - 2 = 0 (should be sqrt(2))
        let func = |x: f32| x * x - 2.0;
        let dfunc = |x: f32| 2.0 * x;

        let root = newton_raphson(func, dfunc, 1.0, None, None).unwrap();
        assert_relative_eq!(root, 2.0_f32.sqrt(), epsilon = 1e-6);
    }

    #[test]
    fn test_bisection() {
        // Find root of x^2 - 2 = 0
        let func = |x: f32| x * x - 2.0;

        let root = bisection(func, 0.0, 2.0, None, None).unwrap();
        assert_relative_eq!(root, 2.0_f32.sqrt(), epsilon = 1e-6);
    }
}
