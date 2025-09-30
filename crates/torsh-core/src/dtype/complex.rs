// Data Types: Complex Number Extensions and Operations
//
// This module provides comprehensive support for complex numbers in the ToRSh
// tensor framework. It includes specialized traits for complex operations,
// implementations for standard complex types, and utilities for complex
// number computations in scientific and engineering applications.

use num_complex::Complex;
use num_traits::Float;

use crate::dtype::traits::{FloatElement, TensorElement};

/// Type aliases for convenience and clarity
pub type Complex32 = Complex<f32>;
pub type Complex64 = Complex<f64>;

/// Trait for complex tensor elements with advanced operations
///
/// This trait extends the basic TensorElement functionality with complex-specific
/// operations such as conjugation, phase calculations, and component access.
/// It provides a unified interface for working with complex numbers in tensors.
pub trait ComplexElement: TensorElement {
    /// The underlying real number type (f32 for Complex32, f64 for Complex64)
    type Real: FloatElement;

    /// Get the real part of the complex number
    fn real(&self) -> Self::Real;

    /// Get the imaginary part of the complex number
    fn imag(&self) -> Self::Real;

    /// Create a complex number from real and imaginary parts
    fn new(real: Self::Real, imag: Self::Real) -> Self;

    /// Create a complex number from a real value (imaginary part = 0)
    fn from_real(real: Self::Real) -> Self {
        Self::new(real, Self::Real::zero())
    }

    /// Create a complex number from an imaginary value (real part = 0)
    fn from_imag(imag: Self::Real) -> Self {
        Self::new(Self::Real::zero(), imag)
    }

    /// Get the magnitude (absolute value) of the complex number
    ///
    /// Computes sqrt(real^2 + imag^2)
    fn abs(&self) -> Self::Real;

    /// Get the phase (argument) of the complex number in radians
    ///
    /// Returns the angle θ such that the complex number can be written as
    /// r·e^(iθ) where r is the magnitude.
    fn arg(&self) -> Self::Real;

    /// Get the complex conjugate
    ///
    /// Returns a + bi → a - bi
    fn conj(&self) -> Self;

    /// Get the squared magnitude (norm) of the complex number
    ///
    /// Computes real^2 + imag^2 (more efficient than abs() when you don't need the square root)
    fn norm_sqr(&self) -> Self::Real {
        let r = self.real();
        let i = self.imag();
        r * r + i * i
    }

    /// Create a complex number in polar form (magnitude, phase)
    fn from_polar(magnitude: Self::Real, phase: Self::Real) -> Self {
        Self::new(magnitude * phase.cos(), magnitude * phase.sin())
    }

    /// Check if the complex number is real (imaginary part is zero)
    fn is_real(&self) -> bool {
        self.imag() == Self::Real::zero()
    }

    /// Check if the complex number is purely imaginary (real part is zero)
    fn is_imaginary(&self) -> bool {
        self.real() == Self::Real::zero()
    }

    /// Check if the complex number is finite (both parts are finite)
    fn is_finite(&self) -> bool {
        self.real().is_finite() && self.imag().is_finite()
    }

    /// Check if the complex number is infinite (either part is infinite)
    fn is_infinite(&self) -> bool {
        self.real().is_infinite() || self.imag().is_infinite()
    }

    /// Check if the complex number is NaN (either part is NaN)
    fn is_nan(&self) -> bool {
        self.real().is_nan() || self.imag().is_nan()
    }

    /// Compute the reciprocal (1/z) of the complex number
    fn recip(&self) -> Self {
        let norm_sqr = self.norm_sqr();
        let conj = self.conj();
        Self::new(conj.real() / norm_sqr, conj.imag() / norm_sqr)
    }

    /// Raise the complex number to a real power
    fn powf(&self, exp: Self::Real) -> Self {
        let r = self.abs();
        let theta = self.arg();
        let new_r = r.powf(exp);
        let new_theta = theta * exp;
        Self::from_polar(new_r, new_theta)
    }

    /// Complex exponential function
    fn exp(&self) -> Self {
        let exp_real = self.real().exp();
        Self::new(exp_real * self.imag().cos(), exp_real * self.imag().sin())
    }

    /// Complex natural logarithm
    fn ln(&self) -> Self {
        Self::new(self.abs().ln(), self.arg())
    }

    /// Complex square root
    fn sqrt(&self) -> Self {
        let r = self.abs();
        let theta = self.arg();
        Self::from_polar(r.sqrt(), theta / (Self::Real::one() + Self::Real::one()))
    }
}

// Note: TensorElement implementations for Complex32 and Complex64
// are provided elsewhere in the codebase to avoid conflicts

// Implement ComplexElement for Complex32
impl ComplexElement for Complex32 {
    type Real = f32;

    fn real(&self) -> Self::Real {
        self.re
    }

    fn imag(&self) -> Self::Real {
        self.im
    }

    fn new(real: Self::Real, imag: Self::Real) -> Self {
        Complex32::new(real, imag)
    }

    fn abs(&self) -> Self::Real {
        self.norm()
    }

    fn arg(&self) -> Self::Real {
        self.im.atan2(self.re)
    }

    fn conj(&self) -> Self {
        Complex32::new(self.re, -self.im)
    }
}

// Implement ComplexElement for Complex64
impl ComplexElement for Complex64 {
    type Real = f64;

    fn real(&self) -> Self::Real {
        self.re
    }

    fn imag(&self) -> Self::Real {
        self.im
    }

    fn new(real: Self::Real, imag: Self::Real) -> Self {
        Complex64::new(real, imag)
    }

    fn abs(&self) -> Self::Real {
        self.norm()
    }

    fn arg(&self) -> Self::Real {
        self.im.atan2(self.re)
    }

    fn conj(&self) -> Self {
        Complex64::new(self.re, -self.im)
    }
}

// Implement ComplexElement for real floating-point types
// Real numbers are treated as complex numbers with zero imaginary part

impl ComplexElement for f32 {
    type Real = f32;

    fn real(&self) -> Self::Real {
        *self
    }

    fn imag(&self) -> Self::Real {
        0.0
    }

    fn new(real: Self::Real, _imag: Self::Real) -> Self {
        real
    }

    fn abs(&self) -> Self::Real {
        (*self).abs()
    }

    fn arg(&self) -> Self::Real {
        if *self >= 0.0 {
            0.0
        } else {
            std::f32::consts::PI
        }
    }

    fn conj(&self) -> Self {
        *self
    }
}

impl ComplexElement for f64 {
    type Real = f64;

    fn real(&self) -> Self::Real {
        *self
    }

    fn imag(&self) -> Self::Real {
        0.0
    }

    fn new(real: Self::Real, _imag: Self::Real) -> Self {
        real
    }

    fn abs(&self) -> Self::Real {
        (*self).abs()
    }

    fn arg(&self) -> Self::Real {
        if *self >= 0.0 {
            0.0
        } else {
            std::f64::consts::PI
        }
    }

    fn conj(&self) -> Self {
        *self
    }
}

/// Constants for complex number operations
pub mod constants {
    use super::*;

    /// Complex unit i (0 + 1i) for f32
    pub const I_F32: Complex32 = Complex32::new(0.0, 1.0);

    /// Complex unit i (0 + 1i) for f64
    pub const I_F64: Complex64 = Complex64::new(0.0, 1.0);

    /// Complex one (1 + 0i) for f32
    pub const ONE_F32: Complex32 = Complex32::new(1.0, 0.0);

    /// Complex one (1 + 0i) for f64
    pub const ONE_F64: Complex64 = Complex64::new(1.0, 0.0);

    /// Complex zero (0 + 0i) for f32
    pub const ZERO_F32: Complex32 = Complex32::new(0.0, 0.0);

    /// Complex zero (0 + 0i) for f64
    pub const ZERO_F64: Complex64 = Complex64::new(0.0, 0.0);
}

#[cfg(test)]
mod tests {
    use super::constants::*;
    use super::*;
    use crate::dtype::core::DType;

    #[test]
    fn test_complex_element_basic_operations() {
        let c32 = Complex32::new(3.0, 4.0);
        assert_eq!(c32.real(), 3.0);
        assert_eq!(c32.imag(), 4.0);
        assert_eq!(c32.abs(), 5.0); // sqrt(3^2 + 4^2) = 5

        let c64 = Complex64::new(1.0, 1.0);
        assert_eq!(c64.real(), 1.0);
        assert_eq!(c64.imag(), 1.0);
        assert!((c64.abs() - std::f64::consts::SQRT_2).abs() < f64::EPSILON);
    }

    #[test]
    fn test_complex_conjugate() {
        let c = Complex32::new(3.0, 4.0);
        let conj = c.conj();
        assert_eq!(conj.real(), 3.0);
        assert_eq!(conj.imag(), -4.0);
    }

    #[test]
    fn test_tensor_element_implementations() {
        // Test Complex32
        assert_eq!(Complex32::dtype(), DType::C64);
        let zero_c32 = Complex32::zero();
        assert_eq!(zero_c32.real(), 0.0);
        assert_eq!(zero_c32.imag(), 0.0);

        let one_c32 = Complex32::one();
        assert_eq!(one_c32.real(), 1.0);
        assert_eq!(one_c32.imag(), 0.0);

        // Test Complex64
        assert_eq!(Complex64::dtype(), DType::C128);
        let zero_c64 = Complex64::zero();
        assert_eq!(zero_c64.real(), 0.0);
        assert_eq!(zero_c64.imag(), 0.0);

        let one_c64 = Complex64::one();
        assert_eq!(one_c64.real(), 1.0);
        assert_eq!(one_c64.imag(), 0.0);
    }

    #[test]
    fn test_real_as_complex() {
        let real: f32 = 5.0;
        assert_eq!(real.real(), 5.0);
        assert_eq!(real.imag(), 0.0);
        assert!(real.is_real());
        assert!(!real.is_imaginary());
        assert_eq!(real.conj(), 5.0);
    }

    #[test]
    fn test_complex_constants() {
        // Test imaginary unit
        assert_eq!(I_F32.real(), 0.0);
        assert_eq!(I_F32.imag(), 1.0);

        // Test zero and one
        assert_eq!(ZERO_F32.real(), 0.0);
        assert_eq!(ZERO_F32.imag(), 0.0);
        assert_eq!(ONE_F32.real(), 1.0);
        assert_eq!(ONE_F32.imag(), 0.0);
    }
}
