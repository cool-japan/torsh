//! Standard gradient function implementations for common operations

use super::core::GradientFunction;
use torsh_core::error::Result;

/// Addition gradient function: d/dx(x + y) = 1, d/dy(x + y) = 1
#[derive(Debug)]
pub struct AddGradient;

impl GradientFunction for AddGradient {
    fn backward(&self, grad_output: &[f32]) -> Result<Vec<Vec<f32>>> {
        // For addition, both inputs get the full gradient
        Ok(vec![grad_output.to_vec(), grad_output.to_vec()])
    }

    fn name(&self) -> &str {
        "Add"
    }
}

/// Multiplication gradient function: d/dx(x * y) = y, d/dy(x * y) = x
#[derive(Debug)]
pub struct MulGradient {
    pub x_values: Vec<f32>,
    pub y_values: Vec<f32>,
}

impl GradientFunction for MulGradient {
    fn backward(&self, grad_output: &[f32]) -> Result<Vec<Vec<f32>>> {
        let grad_x: Vec<f32> = grad_output
            .iter()
            .zip(&self.y_values)
            .map(|(&go, &y)| go * y)
            .collect();
        let grad_y: Vec<f32> = grad_output
            .iter()
            .zip(&self.x_values)
            .map(|(&go, &x)| go * x)
            .collect();
        Ok(vec![grad_x, grad_y])
    }

    fn name(&self) -> &str {
        "Mul"
    }
}

/// Square gradient function: d/dx(x²) = 2x
#[derive(Debug)]
pub struct SquareGradient {
    pub input_values: Vec<f32>,
}

impl GradientFunction for SquareGradient {
    fn backward(&self, grad_output: &[f32]) -> Result<Vec<Vec<f32>>> {
        let grad_input: Vec<f32> = grad_output
            .iter()
            .zip(&self.input_values)
            .map(|(&go, &x)| go * 2.0 * x)
            .collect();
        Ok(vec![grad_input])
    }

    fn name(&self) -> &str {
        "Square"
    }
}

/// ReLU gradient function: d/dx(ReLU(x)) = x > 0 ? 1 : 0
#[derive(Debug)]
pub struct ReLUGradient {
    pub input_values: Vec<f32>,
}

impl GradientFunction for ReLUGradient {
    fn backward(&self, grad_output: &[f32]) -> Result<Vec<Vec<f32>>> {
        let grad_input: Vec<f32> = grad_output
            .iter()
            .zip(&self.input_values)
            .map(|(&go, &x)| if x > 0.0 { go } else { 0.0 })
            .collect();
        Ok(vec![grad_input])
    }

    fn name(&self) -> &str {
        "ReLU"
    }
}

/// Sigmoid gradient function: d/dx(sigmoid(x)) = sigmoid(x) * (1 - sigmoid(x))
#[derive(Debug)]
pub struct SigmoidGradient {
    pub output_values: Vec<f32>, // sigmoid(x) values
}

impl GradientFunction for SigmoidGradient {
    fn backward(&self, grad_output: &[f32]) -> Result<Vec<Vec<f32>>> {
        let grad_input: Vec<f32> = grad_output
            .iter()
            .zip(&self.output_values)
            .map(|(&go, &s)| go * s * (1.0 - s))
            .collect();
        Ok(vec![grad_input])
    }

    fn name(&self) -> &str {
        "Sigmoid"
    }
}

/// Matrix multiplication gradient function
#[derive(Debug)]
pub struct MatMulGradient {
    pub lhs_shape: Vec<usize>,
    pub rhs_shape: Vec<usize>,
    pub lhs_values: Vec<f32>,
    pub rhs_values: Vec<f32>,
}

impl GradientFunction for MatMulGradient {
    fn backward(&self, grad_output: &[f32]) -> Result<Vec<Vec<f32>>> {
        // For matrix multiplication C = A @ B:
        // dA = grad_output @ B^T
        // dB = A^T @ grad_output

        // This is a simplified implementation
        // In practice, we'd need proper matrix operations
        let grad_lhs = vec![0.0; self.lhs_values.len()];
        let grad_rhs = vec![0.0; self.rhs_values.len()];

        // Suppress unused warnings for now
        let _ = grad_output;

        Ok(vec![grad_lhs, grad_rhs])
    }

    fn name(&self) -> &str {
        "MatMul"
    }
}

/// Sum gradient function: distributes gradient to all inputs
#[derive(Debug)]
pub struct SumGradient {
    pub num_inputs: usize,
    pub input_size: usize,
}

impl GradientFunction for SumGradient {
    fn backward(&self, grad_output: &[f32]) -> Result<Vec<Vec<f32>>> {
        // For sum operation, each input gets the full gradient
        let grad_input = grad_output.to_vec();
        Ok(vec![grad_input; self.num_inputs])
    }

    fn name(&self) -> &str {
        "Sum"
    }
}

/// Mean gradient function: divides gradient by number of elements
#[derive(Debug)]
pub struct MeanGradient {
    pub num_elements: usize,
}

impl GradientFunction for MeanGradient {
    fn backward(&self, grad_output: &[f32]) -> Result<Vec<Vec<f32>>> {
        let scale = 1.0 / self.num_elements as f32;
        let grad_input: Vec<f32> = grad_output.iter().map(|&go| go * scale).collect();
        Ok(vec![grad_input])
    }

    fn name(&self) -> &str {
        "Mean"
    }
}

/// Subtraction gradient function: d/dx(x - y) = 1, d/dy(x - y) = -1
#[derive(Debug)]
pub struct SubGradient;

impl GradientFunction for SubGradient {
    fn backward(&self, grad_output: &[f32]) -> Result<Vec<Vec<f32>>> {
        // For subtraction, first input gets positive gradient, second gets negative
        let grad_x = grad_output.to_vec();
        let grad_y: Vec<f32> = grad_output.iter().map(|&go| -go).collect();
        Ok(vec![grad_x, grad_y])
    }

    fn name(&self) -> &str {
        "Sub"
    }
}

/// Division gradient function: d/dx(x / y) = 1/y, d/dy(x / y) = -x/y²
#[derive(Debug)]
pub struct DivGradient {
    pub x_values: Vec<f32>,
    pub y_values: Vec<f32>,
}

impl GradientFunction for DivGradient {
    fn backward(&self, grad_output: &[f32]) -> Result<Vec<Vec<f32>>> {
        let grad_x: Vec<f32> = grad_output
            .iter()
            .zip(&self.y_values)
            .map(|(&go, &y)| go / y)
            .collect();
        let grad_y: Vec<f32> = grad_output
            .iter()
            .zip(self.x_values.iter().zip(&self.y_values))
            .map(|(&go, (&x, &y))| -go * x / (y * y))
            .collect();
        Ok(vec![grad_x, grad_y])
    }

    fn name(&self) -> &str {
        "Div"
    }
}

/// Power gradient function: d/dx(x^n) = n * x^(n-1)
#[derive(Debug)]
pub struct PowGradient {
    pub base_values: Vec<f32>,
    pub exponent: f32,
}

impl GradientFunction for PowGradient {
    fn backward(&self, grad_output: &[f32]) -> Result<Vec<Vec<f32>>> {
        let grad_input: Vec<f32> = grad_output
            .iter()
            .zip(&self.base_values)
            .map(|(&go, &x)| go * self.exponent * x.powf(self.exponent - 1.0))
            .collect();
        Ok(vec![grad_input])
    }

    fn name(&self) -> &str {
        "Pow"
    }
}

/// Exponential gradient function: d/dx(exp(x)) = exp(x)
#[derive(Debug)]
pub struct ExpGradient {
    pub output_values: Vec<f32>, // exp(x) values
}

impl GradientFunction for ExpGradient {
    fn backward(&self, grad_output: &[f32]) -> Result<Vec<Vec<f32>>> {
        let grad_input: Vec<f32> = grad_output
            .iter()
            .zip(&self.output_values)
            .map(|(&go, &exp_x)| go * exp_x)
            .collect();
        Ok(vec![grad_input])
    }

    fn name(&self) -> &str {
        "Exp"
    }
}

/// Natural logarithm gradient function: d/dx(ln(x)) = 1/x
#[derive(Debug)]
pub struct LogGradient {
    pub input_values: Vec<f32>,
}

impl GradientFunction for LogGradient {
    fn backward(&self, grad_output: &[f32]) -> Result<Vec<Vec<f32>>> {
        let grad_input: Vec<f32> = grad_output
            .iter()
            .zip(&self.input_values)
            .map(|(&go, &x)| go / x)
            .collect();
        Ok(vec![grad_input])
    }

    fn name(&self) -> &str {
        "Log"
    }
}

/// Tanh gradient function: d/dx(tanh(x)) = 1 - tanh²(x)
#[derive(Debug)]
pub struct TanhGradient {
    pub output_values: Vec<f32>, // tanh(x) values
}

impl GradientFunction for TanhGradient {
    fn backward(&self, grad_output: &[f32]) -> Result<Vec<Vec<f32>>> {
        let grad_input: Vec<f32> = grad_output
            .iter()
            .zip(&self.output_values)
            .map(|(&go, &tanh_x)| go * (1.0 - tanh_x * tanh_x))
            .collect();
        Ok(vec![grad_input])
    }

    fn name(&self) -> &str {
        "Tanh"
    }
}

/// Absolute value gradient function: d/dx(|x|) = sign(x)
#[derive(Debug)]
pub struct AbsGradient {
    pub input_values: Vec<f32>,
}

impl GradientFunction for AbsGradient {
    fn backward(&self, grad_output: &[f32]) -> Result<Vec<Vec<f32>>> {
        let grad_input: Vec<f32> = grad_output
            .iter()
            .zip(&self.input_values)
            .map(|(&go, &x)| {
                if x > 0.0 {
                    go
                } else if x < 0.0 {
                    -go
                } else {
                    0.0 // Undefined at x=0, use 0
                }
            })
            .collect();
        Ok(vec![grad_input])
    }

    fn name(&self) -> &str {
        "Abs"
    }
}
