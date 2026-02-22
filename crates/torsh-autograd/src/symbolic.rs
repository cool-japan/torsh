//! Symbolic differentiation for simple expressions
//!
//! This module provides symbolic differentiation capabilities for simple mathematical
//! expressions, complementing the automatic differentiation system with compile-time
//! derivative computation for known analytical forms.

use std::collections::HashMap;
use std::fmt;
use torsh_core::error::{Result, TorshError};

/// Symbolic expression representation
#[derive(Debug, Clone, PartialEq)]
pub enum SymbolicExpr {
    /// Constant value
    Constant(f64),
    /// Variable (identified by name)
    Variable(String),
    /// Addition of two expressions
    Add(Box<SymbolicExpr>, Box<SymbolicExpr>),
    /// Subtraction of two expressions
    Sub(Box<SymbolicExpr>, Box<SymbolicExpr>),
    /// Multiplication of two expressions
    Mul(Box<SymbolicExpr>, Box<SymbolicExpr>),
    /// Division of two expressions
    Div(Box<SymbolicExpr>, Box<SymbolicExpr>),
    /// Power operation (base^exponent)
    Pow(Box<SymbolicExpr>, Box<SymbolicExpr>),
    /// Natural exponential function
    Exp(Box<SymbolicExpr>),
    /// Natural logarithm
    Ln(Box<SymbolicExpr>),
    /// Sine function
    Sin(Box<SymbolicExpr>),
    /// Cosine function
    Cos(Box<SymbolicExpr>),
    /// Tangent function
    Tan(Box<SymbolicExpr>),
    /// Hyperbolic sine
    Sinh(Box<SymbolicExpr>),
    /// Hyperbolic cosine
    Cosh(Box<SymbolicExpr>),
    /// Hyperbolic tangent
    Tanh(Box<SymbolicExpr>),
}

impl fmt::Display for SymbolicExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SymbolicExpr::Constant(c) => write!(f, "{}", c),
            SymbolicExpr::Variable(v) => write!(f, "{}", v),
            SymbolicExpr::Add(a, b) => write!(f, "({} + {})", a, b),
            SymbolicExpr::Sub(a, b) => write!(f, "({} - {})", a, b),
            SymbolicExpr::Mul(a, b) => write!(f, "({} * {})", a, b),
            SymbolicExpr::Div(a, b) => write!(f, "({} / {})", a, b),
            SymbolicExpr::Pow(a, b) => write!(f, "({} ^ {})", a, b),
            SymbolicExpr::Exp(a) => write!(f, "exp({})", a),
            SymbolicExpr::Ln(a) => write!(f, "ln({})", a),
            SymbolicExpr::Sin(a) => write!(f, "sin({})", a),
            SymbolicExpr::Cos(a) => write!(f, "cos({})", a),
            SymbolicExpr::Tan(a) => write!(f, "tan({})", a),
            SymbolicExpr::Sinh(a) => write!(f, "sinh({})", a),
            SymbolicExpr::Cosh(a) => write!(f, "cosh({})", a),
            SymbolicExpr::Tanh(a) => write!(f, "tanh({})", a),
        }
    }
}

impl SymbolicExpr {
    /// Create a constant expression
    pub fn constant(value: f64) -> Self {
        SymbolicExpr::Constant(value)
    }

    /// Create a variable expression
    pub fn variable<S: Into<String>>(name: S) -> Self {
        SymbolicExpr::Variable(name.into())
    }

    /// Create an addition expression
    pub fn add(left: SymbolicExpr, right: SymbolicExpr) -> Self {
        SymbolicExpr::Add(Box::new(left), Box::new(right))
    }

    /// Create a subtraction expression
    pub fn sub(left: SymbolicExpr, right: SymbolicExpr) -> Self {
        SymbolicExpr::Sub(Box::new(left), Box::new(right))
    }

    /// Create a multiplication expression
    pub fn mul(left: SymbolicExpr, right: SymbolicExpr) -> Self {
        SymbolicExpr::Mul(Box::new(left), Box::new(right))
    }

    /// Create a division expression
    pub fn div(left: SymbolicExpr, right: SymbolicExpr) -> Self {
        SymbolicExpr::Div(Box::new(left), Box::new(right))
    }

    /// Create a power expression
    pub fn pow(base: SymbolicExpr, exponent: SymbolicExpr) -> Self {
        SymbolicExpr::Pow(Box::new(base), Box::new(exponent))
    }

    /// Create an exponential expression
    pub fn exp(expr: SymbolicExpr) -> Self {
        SymbolicExpr::Exp(Box::new(expr))
    }

    /// Create a natural logarithm expression
    pub fn ln(expr: SymbolicExpr) -> Self {
        SymbolicExpr::Ln(Box::new(expr))
    }

    /// Create a sine expression
    pub fn sin(expr: SymbolicExpr) -> Self {
        SymbolicExpr::Sin(Box::new(expr))
    }

    /// Create a cosine expression
    pub fn cos(expr: SymbolicExpr) -> Self {
        SymbolicExpr::Cos(Box::new(expr))
    }

    /// Create a tangent expression
    pub fn tan(expr: SymbolicExpr) -> Self {
        SymbolicExpr::Tan(Box::new(expr))
    }

    /// Create a hyperbolic sine expression
    pub fn sinh(expr: SymbolicExpr) -> Self {
        SymbolicExpr::Sinh(Box::new(expr))
    }

    /// Create a hyperbolic cosine expression
    pub fn cosh(expr: SymbolicExpr) -> Self {
        SymbolicExpr::Cosh(Box::new(expr))
    }

    /// Create a hyperbolic tangent expression
    pub fn tanh(expr: SymbolicExpr) -> Self {
        SymbolicExpr::Tanh(Box::new(expr))
    }

    /// Evaluate the expression with given variable values
    pub fn evaluate(&self, variables: &HashMap<String, f64>) -> Result<f64> {
        match self {
            SymbolicExpr::Constant(c) => Ok(*c),
            SymbolicExpr::Variable(v) => variables
                .get(v)
                .copied()
                .ok_or_else(|| TorshError::InvalidArgument(format!("Variable '{}' not found", v))),
            SymbolicExpr::Add(a, b) => Ok(a.evaluate(variables)? + b.evaluate(variables)?),
            SymbolicExpr::Sub(a, b) => Ok(a.evaluate(variables)? - b.evaluate(variables)?),
            SymbolicExpr::Mul(a, b) => Ok(a.evaluate(variables)? * b.evaluate(variables)?),
            SymbolicExpr::Div(a, b) => {
                let denominator = b.evaluate(variables)?;
                if denominator.abs() < f64::EPSILON {
                    return Err(TorshError::InvalidArgument("Division by zero".to_string()));
                }
                Ok(a.evaluate(variables)? / denominator)
            }
            SymbolicExpr::Pow(a, b) => Ok(a.evaluate(variables)?.powf(b.evaluate(variables)?)),
            SymbolicExpr::Exp(a) => Ok(a.evaluate(variables)?.exp()),
            SymbolicExpr::Ln(a) => {
                let val = a.evaluate(variables)?;
                if val <= 0.0 {
                    return Err(TorshError::InvalidArgument(
                        "Logarithm of non-positive number".to_string(),
                    ));
                }
                Ok(val.ln())
            }
            SymbolicExpr::Sin(a) => Ok(a.evaluate(variables)?.sin()),
            SymbolicExpr::Cos(a) => Ok(a.evaluate(variables)?.cos()),
            SymbolicExpr::Tan(a) => Ok(a.evaluate(variables)?.tan()),
            SymbolicExpr::Sinh(a) => Ok(a.evaluate(variables)?.sinh()),
            SymbolicExpr::Cosh(a) => Ok(a.evaluate(variables)?.cosh()),
            SymbolicExpr::Tanh(a) => Ok(a.evaluate(variables)?.tanh()),
        }
    }

    /// Compute the symbolic derivative with respect to a variable
    pub fn differentiate(&self, var: &str) -> SymbolicExpr {
        match self {
            // d/dx (c) = 0
            SymbolicExpr::Constant(_) => SymbolicExpr::Constant(0.0),

            // d/dx (x) = 1, d/dx (y) = 0 if x != y
            SymbolicExpr::Variable(v) => {
                if v == var {
                    SymbolicExpr::Constant(1.0)
                } else {
                    SymbolicExpr::Constant(0.0)
                }
            }

            // d/dx (f + g) = f' + g'
            SymbolicExpr::Add(f, g) => SymbolicExpr::Add(
                Box::new(f.differentiate(var)),
                Box::new(g.differentiate(var)),
            ),

            // d/dx (f - g) = f' - g'
            SymbolicExpr::Sub(f, g) => SymbolicExpr::Sub(
                Box::new(f.differentiate(var)),
                Box::new(g.differentiate(var)),
            ),

            // d/dx (f * g) = f' * g + f * g' (product rule)
            SymbolicExpr::Mul(f, g) => {
                let f_prime = f.differentiate(var);
                let g_prime = g.differentiate(var);
                SymbolicExpr::Add(
                    Box::new(SymbolicExpr::Mul(Box::new(f_prime), g.clone())),
                    Box::new(SymbolicExpr::Mul(f.clone(), Box::new(g_prime))),
                )
            }

            // d/dx (f / g) = (f' * g - f * g') / g^2 (quotient rule)
            SymbolicExpr::Div(f, g) => {
                let f_prime = f.differentiate(var);
                let g_prime = g.differentiate(var);
                let numerator = SymbolicExpr::Sub(
                    Box::new(SymbolicExpr::Mul(Box::new(f_prime), g.clone())),
                    Box::new(SymbolicExpr::Mul(f.clone(), Box::new(g_prime))),
                );
                let denominator =
                    SymbolicExpr::Pow(g.clone(), Box::new(SymbolicExpr::Constant(2.0)));
                SymbolicExpr::Div(Box::new(numerator), Box::new(denominator))
            }

            // d/dx (f^g) = f^g * (g' * ln(f) + g * f'/f) (generalized power rule)
            // But use simpler power rule when g is constant: d/dx(f^c) = c * f^(c-1) * f'
            SymbolicExpr::Pow(f, g) => {
                let f_prime = f.differentiate(var);
                let g_prime = g.differentiate(var);

                // Check if g is constant (g' = 0)
                if matches!(g_prime, SymbolicExpr::Constant(0.0)) {
                    // Simple power rule: d/dx(f^c) = c * f^(c-1) * f'
                    SymbolicExpr::Mul(
                        Box::new(SymbolicExpr::Mul(
                            g.clone(),
                            Box::new(SymbolicExpr::Pow(
                                f.clone(),
                                Box::new(SymbolicExpr::Sub(
                                    g.clone(),
                                    Box::new(SymbolicExpr::Constant(1.0)),
                                )),
                            )),
                        )),
                        Box::new(f_prime),
                    )
                } else {
                    // Generalized power rule for variable exponents
                    let term1 =
                        SymbolicExpr::Mul(Box::new(g_prime), Box::new(SymbolicExpr::Ln(f.clone())));
                    let term2 = SymbolicExpr::Mul(
                        g.clone(),
                        Box::new(SymbolicExpr::Div(Box::new(f_prime), f.clone())),
                    );
                    let derivative_ln = SymbolicExpr::Add(Box::new(term1), Box::new(term2));

                    SymbolicExpr::Mul(
                        Box::new(SymbolicExpr::Pow(f.clone(), g.clone())),
                        Box::new(derivative_ln),
                    )
                }
            }

            // d/dx (exp(f)) = exp(f) * f'
            SymbolicExpr::Exp(f) => {
                let f_prime = f.differentiate(var);
                SymbolicExpr::Mul(Box::new(SymbolicExpr::Exp(f.clone())), Box::new(f_prime))
            }

            // d/dx (ln(f)) = f' / f
            SymbolicExpr::Ln(f) => {
                let f_prime = f.differentiate(var);
                SymbolicExpr::Div(Box::new(f_prime), f.clone())
            }

            // d/dx (sin(f)) = cos(f) * f'
            SymbolicExpr::Sin(f) => {
                let f_prime = f.differentiate(var);
                SymbolicExpr::Mul(Box::new(SymbolicExpr::Cos(f.clone())), Box::new(f_prime))
            }

            // d/dx (cos(f)) = -sin(f) * f'
            SymbolicExpr::Cos(f) => {
                let f_prime = f.differentiate(var);
                SymbolicExpr::Mul(
                    Box::new(SymbolicExpr::Mul(
                        Box::new(SymbolicExpr::Constant(-1.0)),
                        Box::new(SymbolicExpr::Sin(f.clone())),
                    )),
                    Box::new(f_prime),
                )
            }

            // d/dx (tan(f)) = sec^2(f) * f' = (1 + tan^2(f)) * f'
            SymbolicExpr::Tan(f) => {
                let f_prime = f.differentiate(var);
                let sec_squared = SymbolicExpr::Add(
                    Box::new(SymbolicExpr::Constant(1.0)),
                    Box::new(SymbolicExpr::Pow(
                        Box::new(SymbolicExpr::Tan(f.clone())),
                        Box::new(SymbolicExpr::Constant(2.0)),
                    )),
                );
                SymbolicExpr::Mul(Box::new(sec_squared), Box::new(f_prime))
            }

            // d/dx (sinh(f)) = cosh(f) * f'
            SymbolicExpr::Sinh(f) => {
                let f_prime = f.differentiate(var);
                SymbolicExpr::Mul(Box::new(SymbolicExpr::Cosh(f.clone())), Box::new(f_prime))
            }

            // d/dx (cosh(f)) = sinh(f) * f'
            SymbolicExpr::Cosh(f) => {
                let f_prime = f.differentiate(var);
                SymbolicExpr::Mul(Box::new(SymbolicExpr::Sinh(f.clone())), Box::new(f_prime))
            }

            // d/dx (tanh(f)) = sech^2(f) * f' = (1 - tanh^2(f)) * f'
            SymbolicExpr::Tanh(f) => {
                let f_prime = f.differentiate(var);
                let sech_squared = SymbolicExpr::Sub(
                    Box::new(SymbolicExpr::Constant(1.0)),
                    Box::new(SymbolicExpr::Pow(
                        Box::new(SymbolicExpr::Tanh(f.clone())),
                        Box::new(SymbolicExpr::Constant(2.0)),
                    )),
                );
                SymbolicExpr::Mul(Box::new(sech_squared), Box::new(f_prime))
            }
        }
    }

    /// Simplify the expression using basic algebraic rules
    pub fn simplify(self) -> SymbolicExpr {
        match self {
            // Addition simplifications
            SymbolicExpr::Add(a, b) => {
                let a_simplified = a.simplify();
                let b_simplified = b.simplify();

                match (&a_simplified, &b_simplified) {
                    // 0 + x = x, x + 0 = x
                    (SymbolicExpr::Constant(0.0), _) => b_simplified,
                    (_, SymbolicExpr::Constant(0.0)) => a_simplified,
                    // c1 + c2 = c3
                    (SymbolicExpr::Constant(c1), SymbolicExpr::Constant(c2)) => {
                        SymbolicExpr::Constant(c1 + c2)
                    }
                    _ => SymbolicExpr::Add(Box::new(a_simplified), Box::new(b_simplified)),
                }
            }

            // Subtraction simplifications
            SymbolicExpr::Sub(a, b) => {
                let a_simplified = a.simplify();
                let b_simplified = b.simplify();

                match (&a_simplified, &b_simplified) {
                    // x - 0 = x
                    (_, SymbolicExpr::Constant(0.0)) => a_simplified,
                    // 0 - x = -x
                    (SymbolicExpr::Constant(0.0), _) => SymbolicExpr::Mul(
                        Box::new(SymbolicExpr::Constant(-1.0)),
                        Box::new(b_simplified),
                    ),
                    // c1 - c2 = c3
                    (SymbolicExpr::Constant(c1), SymbolicExpr::Constant(c2)) => {
                        SymbolicExpr::Constant(c1 - c2)
                    }
                    // x - x = 0
                    _ if a_simplified == b_simplified => SymbolicExpr::Constant(0.0),
                    _ => SymbolicExpr::Sub(Box::new(a_simplified), Box::new(b_simplified)),
                }
            }

            // Multiplication simplifications
            SymbolicExpr::Mul(a, b) => {
                let a_simplified = a.simplify();
                let b_simplified = b.simplify();

                match (&a_simplified, &b_simplified) {
                    // 0 * x = 0, x * 0 = 0
                    (SymbolicExpr::Constant(0.0), _) | (_, SymbolicExpr::Constant(0.0)) => {
                        SymbolicExpr::Constant(0.0)
                    }
                    // 1 * x = x, x * 1 = x
                    (SymbolicExpr::Constant(1.0), _) => b_simplified,
                    (_, SymbolicExpr::Constant(1.0)) => a_simplified,
                    // c1 * c2 = c3
                    (SymbolicExpr::Constant(c1), SymbolicExpr::Constant(c2)) => {
                        SymbolicExpr::Constant(c1 * c2)
                    }
                    _ => SymbolicExpr::Mul(Box::new(a_simplified), Box::new(b_simplified)),
                }
            }

            // Division simplifications
            SymbolicExpr::Div(a, b) => {
                let a_simplified = a.simplify();
                let b_simplified = b.simplify();

                match (&a_simplified, &b_simplified) {
                    // 0 / x = 0 (x != 0)
                    (SymbolicExpr::Constant(0.0), _) => SymbolicExpr::Constant(0.0),
                    // x / 1 = x
                    (_, SymbolicExpr::Constant(1.0)) => a_simplified,
                    // c1 / c2 = c3
                    (SymbolicExpr::Constant(c1), SymbolicExpr::Constant(c2)) if *c2 != 0.0 => {
                        SymbolicExpr::Constant(c1 / c2)
                    }
                    // x / x = 1
                    _ if a_simplified == b_simplified => SymbolicExpr::Constant(1.0),
                    _ => SymbolicExpr::Div(Box::new(a_simplified), Box::new(b_simplified)),
                }
            }

            // Power simplifications
            SymbolicExpr::Pow(a, b) => {
                let a_simplified = a.simplify();
                let b_simplified = b.simplify();

                match (&a_simplified, &b_simplified) {
                    // x^0 = 1
                    (_, SymbolicExpr::Constant(0.0)) => SymbolicExpr::Constant(1.0),
                    // x^1 = x
                    (_, SymbolicExpr::Constant(1.0)) => a_simplified,
                    // 0^x = 0 (x > 0)
                    (SymbolicExpr::Constant(0.0), SymbolicExpr::Constant(exp)) if *exp > 0.0 => {
                        SymbolicExpr::Constant(0.0)
                    }
                    // 1^x = 1
                    (SymbolicExpr::Constant(1.0), _) => SymbolicExpr::Constant(1.0),
                    // c1^c2 = c3
                    (SymbolicExpr::Constant(c1), SymbolicExpr::Constant(c2)) => {
                        SymbolicExpr::Constant(c1.powf(*c2))
                    }
                    _ => SymbolicExpr::Pow(Box::new(a_simplified), Box::new(b_simplified)),
                }
            }

            // Recursively simplify other expressions
            SymbolicExpr::Exp(a) => {
                let a_simplified = a.simplify();
                match &a_simplified {
                    SymbolicExpr::Constant(0.0) => SymbolicExpr::Constant(1.0),
                    SymbolicExpr::Constant(c) => SymbolicExpr::Constant(c.exp()),
                    _ => SymbolicExpr::Exp(Box::new(a_simplified)),
                }
            }

            SymbolicExpr::Ln(a) => {
                let a_simplified = a.simplify();
                match &a_simplified {
                    SymbolicExpr::Constant(1.0) => SymbolicExpr::Constant(0.0),
                    SymbolicExpr::Constant(c) if *c > 0.0 => SymbolicExpr::Constant(c.ln()),
                    _ => SymbolicExpr::Ln(Box::new(a_simplified)),
                }
            }

            SymbolicExpr::Sin(a) => {
                let a_simplified = a.simplify();
                match &a_simplified {
                    SymbolicExpr::Constant(0.0) => SymbolicExpr::Constant(0.0),
                    SymbolicExpr::Constant(c) => SymbolicExpr::Constant(c.sin()),
                    _ => SymbolicExpr::Sin(Box::new(a_simplified)),
                }
            }

            SymbolicExpr::Cos(a) => {
                let a_simplified = a.simplify();
                match &a_simplified {
                    SymbolicExpr::Constant(0.0) => SymbolicExpr::Constant(1.0),
                    SymbolicExpr::Constant(c) => SymbolicExpr::Constant(c.cos()),
                    _ => SymbolicExpr::Cos(Box::new(a_simplified)),
                }
            }

            SymbolicExpr::Tan(a) => {
                let a_simplified = a.simplify();
                match &a_simplified {
                    SymbolicExpr::Constant(0.0) => SymbolicExpr::Constant(0.0),
                    SymbolicExpr::Constant(c) => SymbolicExpr::Constant(c.tan()),
                    _ => SymbolicExpr::Tan(Box::new(a_simplified)),
                }
            }

            SymbolicExpr::Sinh(a) => {
                let a_simplified = a.simplify();
                match &a_simplified {
                    SymbolicExpr::Constant(0.0) => SymbolicExpr::Constant(0.0),
                    SymbolicExpr::Constant(c) => SymbolicExpr::Constant(c.sinh()),
                    _ => SymbolicExpr::Sinh(Box::new(a_simplified)),
                }
            }

            SymbolicExpr::Cosh(a) => {
                let a_simplified = a.simplify();
                match &a_simplified {
                    SymbolicExpr::Constant(0.0) => SymbolicExpr::Constant(1.0),
                    SymbolicExpr::Constant(c) => SymbolicExpr::Constant(c.cosh()),
                    _ => SymbolicExpr::Cosh(Box::new(a_simplified)),
                }
            }

            SymbolicExpr::Tanh(a) => {
                let a_simplified = a.simplify();
                match &a_simplified {
                    SymbolicExpr::Constant(0.0) => SymbolicExpr::Constant(0.0),
                    SymbolicExpr::Constant(c) => SymbolicExpr::Constant(c.tanh()),
                    _ => SymbolicExpr::Tanh(Box::new(a_simplified)),
                }
            }

            // Base cases
            expr => expr,
        }
    }

    /// Compute higher-order derivatives
    pub fn nth_derivative(&self, var: &str, n: usize) -> SymbolicExpr {
        if n == 0 {
            self.clone()
        } else {
            let first_derivative = self.differentiate(var);
            first_derivative.nth_derivative(var, n - 1)
        }
    }

    /// Get all variables used in this expression
    pub fn variables(&self) -> std::collections::HashSet<String> {
        let mut vars = std::collections::HashSet::new();
        self.collect_variables(&mut vars);
        vars
    }

    fn collect_variables(&self, vars: &mut std::collections::HashSet<String>) {
        match self {
            SymbolicExpr::Variable(v) => {
                vars.insert(v.clone());
            }
            SymbolicExpr::Add(a, b)
            | SymbolicExpr::Sub(a, b)
            | SymbolicExpr::Mul(a, b)
            | SymbolicExpr::Div(a, b)
            | SymbolicExpr::Pow(a, b) => {
                a.collect_variables(vars);
                b.collect_variables(vars);
            }
            SymbolicExpr::Exp(a)
            | SymbolicExpr::Ln(a)
            | SymbolicExpr::Sin(a)
            | SymbolicExpr::Cos(a)
            | SymbolicExpr::Tan(a)
            | SymbolicExpr::Sinh(a)
            | SymbolicExpr::Cosh(a)
            | SymbolicExpr::Tanh(a) => {
                a.collect_variables(vars);
            }
            SymbolicExpr::Constant(_) => {}
        }
    }
}

/// Symbolic differentiation engine
pub struct SymbolicDifferentiator {
    /// Cache for computed derivatives
    derivative_cache: std::collections::HashMap<(String, String), SymbolicExpr>,
}

impl SymbolicDifferentiator {
    /// Create a new symbolic differentiator
    pub fn new() -> Self {
        Self {
            derivative_cache: std::collections::HashMap::new(),
        }
    }

    /// Differentiate an expression with caching
    pub fn differentiate(&mut self, expr: &SymbolicExpr, var: &str) -> SymbolicExpr {
        let key = (format!("{}", expr), var.to_string());

        if let Some(cached) = self.derivative_cache.get(&key) {
            return cached.clone();
        }

        let derivative = expr.differentiate(var).simplify();
        self.derivative_cache.insert(key, derivative.clone());
        derivative
    }

    /// Compute partial derivatives for all variables
    pub fn gradient(&mut self, expr: &SymbolicExpr) -> HashMap<String, SymbolicExpr> {
        let variables = expr.variables();
        let mut gradient = HashMap::new();

        for var in variables {
            let partial = self.differentiate(expr, &var);
            gradient.insert(var, partial);
        }

        gradient
    }

    /// Compute the Hessian matrix (second-order partial derivatives)
    pub fn hessian(&mut self, expr: &SymbolicExpr) -> HashMap<(String, String), SymbolicExpr> {
        let variables: Vec<String> = expr.variables().into_iter().collect();
        let mut hessian = HashMap::new();

        for var1 in &variables {
            let first_partial = self.differentiate(expr, var1);
            for var2 in &variables {
                let second_partial = self.differentiate(&first_partial, var2);
                hessian.insert((var1.clone(), var2.clone()), second_partial);
            }
        }

        hessian
    }

    /// Clear the derivative cache
    pub fn clear_cache(&mut self) {
        self.derivative_cache.clear();
    }

    /// Get cache statistics
    pub fn cache_size(&self) -> usize {
        self.derivative_cache.len()
    }
}

impl Default for SymbolicDifferentiator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_differentiation() {
        // d/dx (x) = 1
        let x = SymbolicExpr::variable("x");
        let derivative = x.differentiate("x");
        assert_eq!(derivative, SymbolicExpr::Constant(1.0));

        // d/dx (5) = 0
        let constant = SymbolicExpr::constant(5.0);
        let derivative = constant.differentiate("x");
        assert_eq!(derivative, SymbolicExpr::Constant(0.0));
    }

    #[test]
    fn test_arithmetic_differentiation() {
        // d/dx (x + y) = 1 + 0 = 1 (w.r.t. x)
        let x = SymbolicExpr::variable("x");
        let y = SymbolicExpr::variable("y");
        let sum = SymbolicExpr::add(x, y);
        let derivative = sum.differentiate("x").simplify();
        assert_eq!(derivative, SymbolicExpr::Constant(1.0));

        // d/dx (x * x) = 2x
        let x = SymbolicExpr::variable("x");
        let x_squared = SymbolicExpr::mul(x.clone(), x.clone());
        let derivative = x_squared.differentiate("x").simplify();

        // The result should be 2x (simplified from x*1 + 1*x = x + x = 2x)
        let _expected = SymbolicExpr::mul(SymbolicExpr::constant(2.0), SymbolicExpr::variable("x"));
        // Note: The exact form might vary due to simplification, so let's test evaluation
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 3.0);
        assert!((derivative.evaluate(&vars).unwrap() - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_trigonometric_differentiation() {
        // d/dx (sin(x)) = cos(x)
        let x = SymbolicExpr::variable("x");
        let sin_x = SymbolicExpr::sin(x.clone());
        let derivative = sin_x.differentiate("x");

        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 0.0);
        assert!((derivative.evaluate(&vars).unwrap() - 1.0).abs() < 1e-10);

        // d/dx (cos(x)) = -sin(x)
        let cos_x = SymbolicExpr::cos(x.clone());
        let derivative = cos_x.differentiate("x");

        vars.insert("x".to_string(), 0.0);
        assert!((derivative.evaluate(&vars).unwrap() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_exponential_differentiation() {
        // d/dx (exp(x)) = exp(x)
        let x = SymbolicExpr::variable("x");
        let exp_x = SymbolicExpr::exp(x.clone());
        let derivative = exp_x.differentiate("x");

        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 0.0);
        assert!((derivative.evaluate(&vars).unwrap() - 1.0).abs() < 1e-10);

        // d/dx (ln(x)) = 1/x
        let ln_x = SymbolicExpr::ln(x.clone());
        let derivative = ln_x.differentiate("x");

        vars.insert("x".to_string(), 1.0);
        assert!((derivative.evaluate(&vars).unwrap() - 1.0).abs() < 1e-10);

        vars.insert("x".to_string(), 2.0);
        assert!((derivative.evaluate(&vars).unwrap() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_power_rule() {
        // d/dx (x^3) = 3x^2
        let x = SymbolicExpr::variable("x");
        let x_cubed = SymbolicExpr::pow(x.clone(), SymbolicExpr::constant(3.0));
        let derivative = x_cubed.differentiate("x");

        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 2.0);
        // At x=2, derivative should be 3*4 = 12
        assert!((derivative.evaluate(&vars).unwrap() - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_chain_rule() {
        // d/dx (sin(x^2)) = cos(x^2) * 2x
        let x = SymbolicExpr::variable("x");
        let x_squared = SymbolicExpr::pow(x.clone(), SymbolicExpr::constant(2.0));
        let sin_x_squared = SymbolicExpr::sin(x_squared);
        let derivative = sin_x_squared.differentiate("x");

        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 0.0);
        // At x=0, derivative should be cos(0) * 0 = 0
        assert!((derivative.evaluate(&vars).unwrap() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_simplification() {
        // 0 + x should simplify to x
        let zero = SymbolicExpr::constant(0.0);
        let x = SymbolicExpr::variable("x");
        let sum = SymbolicExpr::add(zero, x.clone());
        let simplified = sum.simplify();
        assert_eq!(simplified, x);

        // x * 1 should simplify to x
        let one = SymbolicExpr::constant(1.0);
        let product = SymbolicExpr::mul(x.clone(), one);
        let simplified = product.simplify();
        assert_eq!(simplified, x);

        // x - x should simplify to 0
        let difference = SymbolicExpr::sub(x.clone(), x.clone());
        let simplified = difference.simplify();
        assert_eq!(simplified, SymbolicExpr::constant(0.0));
    }

    #[test]
    fn test_evaluation() {
        // Evaluate x^2 + 2x + 1 at x = 3 (should be 16)
        let x = SymbolicExpr::variable("x");
        let x_squared = SymbolicExpr::pow(x.clone(), SymbolicExpr::constant(2.0));
        let two_x = SymbolicExpr::mul(SymbolicExpr::constant(2.0), x.clone());
        let expr = SymbolicExpr::add(
            SymbolicExpr::add(x_squared, two_x),
            SymbolicExpr::constant(1.0),
        );

        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 3.0);
        let result = expr.evaluate(&vars).unwrap();
        assert!((result - 16.0).abs() < 1e-10);
    }

    #[test]
    fn test_differentiator_with_cache() {
        let mut diff = SymbolicDifferentiator::new();
        let x = SymbolicExpr::variable("x");
        let expr = SymbolicExpr::mul(x.clone(), x.clone());

        // First differentiation
        let derivative1 = diff.differentiate(&expr, "x");
        assert_eq!(diff.cache_size(), 1);

        // Second differentiation (should use cache)
        let derivative2 = diff.differentiate(&expr, "x");
        assert_eq!(diff.cache_size(), 1);
        assert_eq!(derivative1, derivative2);
    }

    #[test]
    fn test_gradient_computation() {
        let mut diff = SymbolicDifferentiator::new();

        // f(x, y) = x^2 + y^2
        let x = SymbolicExpr::variable("x");
        let y = SymbolicExpr::variable("y");
        let x_squared = SymbolicExpr::pow(x.clone(), SymbolicExpr::constant(2.0));
        let y_squared = SymbolicExpr::pow(y.clone(), SymbolicExpr::constant(2.0));
        let expr = SymbolicExpr::add(x_squared, y_squared);

        let gradient = diff.gradient(&expr);
        assert_eq!(gradient.len(), 2);
        assert!(gradient.contains_key("x"));
        assert!(gradient.contains_key("y"));

        // Evaluate gradient at (1, 2)
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 1.0);
        vars.insert("y".to_string(), 2.0);

        let df_dx = gradient["x"].evaluate(&vars).unwrap();
        let df_dy = gradient["y"].evaluate(&vars).unwrap();

        // Should be 2x = 2, 2y = 4
        assert!((df_dx - 2.0).abs() < 1e-10);
        assert!((df_dy - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_higher_order_derivatives() {
        // f(x) = x^4, f''(x) = 12x^2
        let x = SymbolicExpr::variable("x");
        let x_fourth = SymbolicExpr::pow(x.clone(), SymbolicExpr::constant(4.0));
        let second_derivative = x_fourth.nth_derivative("x", 2);

        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 2.0);
        let result = second_derivative.evaluate(&vars).unwrap();

        // At x=2, second derivative should be 12*4 = 48
        assert!((result - 48.0).abs() < 1e-10);
    }

    #[test]
    fn test_variables_collection() {
        let x = SymbolicExpr::variable("x");
        let y = SymbolicExpr::variable("y");
        let z = SymbolicExpr::variable("z");

        let expr = SymbolicExpr::add(SymbolicExpr::mul(x, y), SymbolicExpr::sin(z));

        let vars = expr.variables();
        assert_eq!(vars.len(), 3);
        assert!(vars.contains("x"));
        assert!(vars.contains("y"));
        assert!(vars.contains("z"));
    }
}
