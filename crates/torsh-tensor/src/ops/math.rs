//! Mathematical functions for tensors
//!
//! This module provides element-wise mathematical operations for tensors,
//! including trigonometric, exponential, logarithmic, and other common functions.
//!
//! # Examples
//!
//! ```rust
//! use torsh_tensor::Tensor;
//!
//! # fn main() -> torsh_core::error::Result<()> {
//! // Create a tensor with values
//! let x = Tensor::<f32>::from_vec(vec![0.0, 1.0, 2.0], &[3])?;
//!
//! // Apply mathematical operations
//! let sin_x = x.sin();
//! let exp_x = x.exp();
//! let sqrt_x = x.sqrt();
//!
//! // Chain operations
//! let result = x.abs().log1p().tanh();
//! # Ok(())
//! # }
//! ```

use crate::{Tensor, TensorElement, FloatElement};
use torsh_core::Device;
use torsh_core::error::{Result, TorshError};
use scirs2_core::numeric::{Float, Zero, One, cast::ToPrimitive};
use std::f64::consts;

impl<T: FloatElement + Default> Tensor<T> {
    /// Applies sin function element-wise
    ///
    /// # Examples
    ///
    /// ```rust
    /// use torsh_tensor::Tensor;
    /// use std::f32::consts::PI;
    ///
    /// # fn main() -> torsh_core::error::Result<()> {
    /// let x = Tensor::<f32>::from_vec(vec![0.0, PI / 2.0, PI], &[3])?;
    /// let y = x.sin();
    /// // y ≈ [0.0, 1.0, 0.0]
    /// # Ok(())
    /// # }
    /// ```
    pub fn sin(&self) -> Self {
        let mut result = self.clone();
        for i in 0..result.numel() {
            if let Ok(val) = result.get_item_flat(i) {
                if let Some(val_f64) = <T as TensorElement>::to_f64(&val) {
                    let sin_val = T::from(val_f64.sin()).unwrap_or(val);
                    let _ = result.set_item_flat(i, sin_val);
                }
            }
        }
        result
    }

    /// Applies cos function element-wise
    ///
    /// # Examples
    ///
    /// ```rust
    /// use torsh_tensor::Tensor;
    /// use std::f32::consts::PI;
    ///
    /// # fn main() -> torsh_core::error::Result<()> {
    /// let x = Tensor::<f32>::from_vec(vec![0.0, PI / 2.0, PI], &[3])?;
    /// let y = x.cos();
    /// // y ≈ [1.0, 0.0, -1.0]
    /// # Ok(())
    /// # }
    /// ```
    pub fn cos(&self) -> Self {
        let mut result = self.clone();
        for i in 0..result.numel() {
            if let Ok(val) = result.get_item_flat(i) {
                if let Some(val_f64) = <T as TensorElement>::to_f64(&val) {
                    let cos_val = T::from(val_f64.cos()).unwrap_or(val);
                    let _ = result.set_item_flat(i, cos_val);
                }
            }
        }
        result
    }

    /// Applies tan function element-wise
    pub fn tan(&self) -> Self {
        let mut result = self.clone();
        for i in 0..result.numel() {
            if let Ok(val) = result.get_item_flat(i) {
                if let Some(val_f64) = <T as TensorElement>::to_f64(&val) {
                    let tan_val = T::from(val_f64.tan()).unwrap_or(val);
                    let _ = result.set_item_flat(i, tan_val);
                }
            }
        }
        result
    }

    /// Applies arcsin function element-wise
    pub fn asin(&self) -> Self {
        let mut result = self.clone();
        for i in 0..result.numel() {
            if let Ok(val) = result.get_item_flat(i) {
                if let Some(val_f64) = <T as TensorElement>::to_f64(&val) {
                    let asin_val = T::from(val_f64.asin()).unwrap_or(val);
                    let _ = result.set_item_flat(i, asin_val);
                }
            }
        }
        result
    }

    /// Applies arccos function element-wise
    pub fn acos(&self) -> Self {
        let mut result = self.clone();
        for i in 0..result.numel() {
            if let Ok(val) = result.get_item_flat(i) {
                if let Some(val_f64) = <T as TensorElement>::to_f64(&val) {
                    let acos_val = T::from(val_f64.acos()).unwrap_or(val);
                    let _ = result.set_item_flat(i, acos_val);
                }
            }
        }
        result
    }

    /// Applies arctan function element-wise
    pub fn atan(&self) -> Self {
        let mut result = self.clone();
        for i in 0..result.numel() {
            if let Ok(val) = result.get_item_flat(i) {
                if let Some(val_f64) = <T as TensorElement>::to_f64(&val) {
                    let atan_val = T::from(val_f64.atan()).unwrap_or(val);
                    let _ = result.set_item_flat(i, atan_val);
                }
            }
        }
        result
    }

    /// Applies atan2 function element-wise
    pub fn atan2(&self, other: &Self) -> Result<Self> {
        let result = self.broadcast_binary_op(other, |y, x| {
            if let (Some(y_f64), Some(x_f64)) = (<T as TensorElement>::to_f64(&y), <T as TensorElement>::to_f64(&x)) {
                T::from(y_f64.atan2(x_f64)).unwrap_or(y)
            } else {
                y
            }
        })?;
        Ok(result)
    }

    /// Applies sinh function element-wise
    pub fn sinh(&self) -> Self {
        let mut result = self.clone();
        for i in 0..result.numel() {
            if let Ok(val) = result.get_item_flat(i) {
                if let Some(val_f64) = <T as TensorElement>::to_f64(&val) {
                    let sinh_val = T::from(val_f64.sinh()).unwrap_or(val);
                    let _ = result.set_item_flat(i, sinh_val);
                }
            }
        }
        result
    }

    /// Applies cosh function element-wise
    pub fn cosh(&self) -> Self {
        let mut result = self.clone();
        for i in 0..result.numel() {
            if let Ok(val) = result.get_item_flat(i) {
                if let Some(val_f64) = <T as TensorElement>::to_f64(&val) {
                    let cosh_val = T::from(val_f64.cosh()).unwrap_or(val);
                    let _ = result.set_item_flat(i, cosh_val);
                }
            }
        }
        result
    }


    /// Applies exp function element-wise (e^x)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use torsh_tensor::Tensor;
    ///
    /// # fn main() -> torsh_core::error::Result<()> {
    /// let x = Tensor::<f32>::from_vec(vec![0.0, 1.0, 2.0], &[3])?;
    /// let y = x.exp();
    /// // y ≈ [1.0, 2.718, 7.389]
    /// # Ok(())
    /// # }
    /// ```
    pub fn exp(&self) -> Self {
        let mut result = self.clone();
        for i in 0..result.numel() {
            if let Ok(val) = result.get_item_flat(i) {
                if let Some(val_f64) = <T as TensorElement>::to_f64(&val) {
                    let exp_val = T::from(val_f64.exp()).unwrap_or(val);
                    let _ = result.set_item_flat(i, exp_val);
                }
            }
        }
        result
    }

    /// Applies exp2 (2^x) function element-wise
    pub fn exp2(&self) -> Self {
        let mut result = self.clone();
        for i in 0..result.numel() {
            if let Ok(val) = result.get_item_flat(i) {
                if let Some(val_f64) = <T as TensorElement>::to_f64(&val) {
                    let exp2_val = T::from(val_f64.exp2()).unwrap_or(val);
                    let _ = result.set_item_flat(i, exp2_val);
                }
            }
        }
        result
    }

    /// Applies expm1 (exp(x) - 1) function element-wise
    pub fn expm1(&self) -> Self {
        let mut result = self.clone();
        for i in 0..result.numel() {
            if let Ok(val) = result.get_item_flat(i) {
                if let Some(val_f64) = <T as TensorElement>::to_f64(&val) {
                    let expm1_val = T::from(val_f64.exp_m1()).unwrap_or(val);
                    let _ = result.set_item_flat(i, expm1_val);
                }
            }
        }
        result
    }

    /// Applies natural logarithm function element-wise (ln(x))
    ///
    /// # Examples
    ///
    /// ```rust
    /// use torsh_tensor::Tensor;
    ///
    /// # fn main() -> torsh_core::error::Result<()> {
    /// let x = Tensor::<f32>::from_vec(vec![1.0, std::f32::consts::E, 10.0], &[3])?;
    /// let y = x.log();
    /// // y ≈ [0.0, 1.0, 2.303]
    /// # Ok(())
    /// # }
    /// ```
    pub fn log(&self) -> Self {
        let mut result = self.clone();
        for i in 0..result.numel() {
            if let Ok(val) = result.get_item_flat(i) {
                if let Some(val_f64) = <T as TensorElement>::to_f64(&val) {
                    let log_val = T::from(val_f64.ln()).unwrap_or(val);
                    let _ = result.set_item_flat(i, log_val);
                }
            }
        }
        result
    }

    /// Applies log2 function element-wise
    pub fn log2(&self) -> Self {
        let mut result = self.clone();
        for i in 0..result.numel() {
            if let Ok(val) = result.get_item_flat(i) {
                if let Some(val_f64) = <T as TensorElement>::to_f64(&val) {
                    let log2_val = T::from(val_f64.log2()).unwrap_or(val);
                    let _ = result.set_item_flat(i, log2_val);
                }
            }
        }
        result
    }

    /// Applies log10 function element-wise
    pub fn log10(&self) -> Self {
        let mut result = self.clone();
        for i in 0..result.numel() {
            if let Ok(val) = result.get_item_flat(i) {
                if let Some(val_f64) = <T as TensorElement>::to_f64(&val) {
                    let log10_val = T::from(val_f64.log10()).unwrap_or(val);
                    let _ = result.set_item_flat(i, log10_val);
                }
            }
        }
        result
    }

    /// Applies ln(1 + x) function element-wise
    pub fn log1p(&self) -> Self {
        let mut result = self.clone();
        for i in 0..result.numel() {
            if let Ok(val) = result.get_item_flat(i) {
                if let Some(val_f64) = <T as TensorElement>::to_f64(&val) {
                    let log1p_val = T::from(val_f64.ln_1p()).unwrap_or(val);
                    let _ = result.set_item_flat(i, log1p_val);
                }
            }
        }
        result
    }

    /// Applies square root function element-wise
    ///
    /// # Examples
    ///
    /// ```rust
    /// use torsh_tensor::Tensor;
    ///
    /// # fn main() -> torsh_core::error::Result<()> {
    /// let x = Tensor::<f32>::from_vec(vec![1.0, 4.0, 9.0, 16.0], &[4])?;
    /// let y = x.sqrt();
    /// // y = [1.0, 2.0, 3.0, 4.0]
    /// # Ok(())
    /// # }
    /// ```
    pub fn sqrt(&self) -> Self {
        let mut result = self.clone();
        for i in 0..result.numel() {
            if let Ok(val) = result.get_item_flat(i) {
                if let Some(val_f64) = <T as TensorElement>::to_f64(&val) {
                    let sqrt_val = T::from(val_f64.sqrt()).unwrap_or(val);
                    let _ = result.set_item_flat(i, sqrt_val);
                }
            }
        }
        result
    }

    /// Applies cube root function element-wise
    pub fn cbrt(&self) -> Self {
        let mut result = self.clone();
        for i in 0..result.numel() {
            if let Ok(val) = result.get_item_flat(i) {
                if let Some(val_f64) = <T as TensorElement>::to_f64(&val) {
                    let cbrt_val = T::from(val_f64.cbrt()).unwrap_or(val);
                    let _ = result.set_item_flat(i, cbrt_val);
                }
            }
        }
        result
    }

    /// Raises each element of `self` to the power of corresponding element in `exponent`.
    ///
    /// Computes `self[i]^exponent[i]` for each element. If the tensors have
    /// different shapes, broadcasting rules are applied to make them compatible.
    ///
    /// # Broadcasting Rules
    /// - Dimensions are aligned from right to left
    /// - Dimension of size 1 can broadcast to any size
    /// - Missing dimensions are treated as size 1
    ///
    /// Examples of valid broadcasts:
    /// - `[3, 4]` ^ `[3, 4]` → `[3, 4]` (same shape)
    /// - `[3, 4]` ^ `[4]` → `[3, 4]` (broadcast exponent to each row)
    /// - `[3, 1]` ^ `[1, 4]` → `[3, 4]` (broadcast both)
    ///
    /// # Special Cases
    /// Following IEEE 754 standard:
    /// - `x^0` = 1 (for any x, including 0)
    /// - `0^x` = 0 (for x > 0)
    /// - `0^0` = 1 (by convention)
    /// - `x^1` = x
    /// - Negative base with non-integer exponent may produce NaN
    ///
    /// # Arguments
    /// * `exponent` - The tensor containing exponent values
    ///
    /// # Returns
    /// A new tensor containing element-wise powers
    ///
    /// # Errors
    /// Returns error if the shapes are not compatible for broadcasting
    ///
    /// # Examples
    /// ```rust,no_run
    /// # use torsh::Tensor;
    /// # use torsh_core::device::DeviceType;
    /// // Same shape: compute squares and cubes
    /// let base = Tensor::from_data(vec![2.0, 3.0, 4.0], vec![3], DeviceType::Cpu)?;
    /// let exp = Tensor::from_data(vec![2.0, 3.0, 2.0], vec![3], DeviceType::Cpu)?;
    /// let result = base.pow(&exp)?;
    /// assert_eq!(result.data()?, vec![4.0, 27.0, 16.0]); // [2^2, 3^3, 4^2]
    ///
    /// // Broadcasting: raise matrix to per-column powers
    /// let base = Tensor::from_data(
    ///     vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    ///     vec![2, 3],
    ///     DeviceType::Cpu
    /// )?;
    /// let exp = Tensor::from_data(vec![1.0, 2.0, 3.0], vec![3], DeviceType::Cpu)?;
    /// let result = base.pow(&exp)?;  // Each column raised to its exponent
    /// assert_eq!(result.data()?, vec![2.0, 9.0, 64.0, 5.0, 36.0, 343.0]);
    ///
    /// // Polynomial activation: x + x^2 + x^3
    /// let x = Tensor::randn(&[64, 128], DeviceType::Cpu)?;
    /// let two = Tensor::full(&[128], 2.0, DeviceType::Cpu)?;
    /// let three = Tensor::full(&[128], 3.0, DeviceType::Cpu)?;
    /// let x2 = x.pow(&two)?;
    /// let x3 = x.pow(&three)?;
    /// let result = x.add(&x2)?.add(&x3)?;
    ///
    /// // Feature interactions
    /// let features = Tensor::randn(&[32, 10], DeviceType::Cpu)?;
    /// let powers = Tensor::from_data(vec![1.0, 2.0, 1.0, 3.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0], vec![10], DeviceType::Cpu)?;
    /// let interactions = features.pow(&powers)?;  // Different power per feature
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # PyTorch Compatibility
    /// Equivalent to `torch.pow(base, exponent)` or `base ** exponent`
    ///
    /// For scalar exponents, consider using [`Self::powf`] for better performance.
    ///
    /// See also: [`Self::powf`], [`Self::sqrt`], [`Self::exp`], [`Self::mul`]
    pub fn pow(&self, exponent: &Self) -> Result<Self> {
        let result = self.broadcast_binary_op(exponent, |base, exp| {
            if let (Some(base_f64), Some(exp_f64)) = (<T as TensorElement>::to_f64(&base), <T as TensorElement>::to_f64(&exp)) {
                T::from(base_f64.powf(exp_f64)).unwrap_or(base)
            } else {
                base
            }
        })?;
        Ok(result)
    }

    /// Applies power function with scalar exponent
    pub fn powf(&self, exponent: f64) -> Self {
        let mut result = self.clone();
        for i in 0..result.numel() {
            if let Ok(val) = result.get_item_flat(i) {
                if let Some(val_f64) = <T as TensorElement>::to_f64(&val) {
                    let pow_val = T::from(val_f64.powf(exponent)).unwrap_or(val);
                    let _ = result.set_item_flat(i, pow_val);
                }
            }
        }
        result
    }

    /// Applies absolute value function element-wise
    ///
    /// # Examples
    ///
    /// ```rust
    /// use torsh_tensor::Tensor;
    ///
    /// # fn main() -> torsh_core::error::Result<()> {
    /// let x = Tensor::<f32>::from_vec(vec![-3.0, -1.0, 0.0, 2.0], &[4])?;
    /// let y = x.abs();
    /// // y = [3.0, 1.0, 0.0, 2.0]
    /// # Ok(())
    /// # }
    /// ```
    pub fn abs(&self) -> Self {
        let mut result = self.clone();
        for i in 0..result.numel() {
            if let Ok(val) = result.get_item_flat(i) {
                if let Some(val_f64) = <T as TensorElement>::to_f64(&val) {
                    let abs_val = T::from(val_f64.abs()).unwrap_or(val);
                    let _ = result.set_item_flat(i, abs_val);
                }
            }
        }
        result
    }

    /// Applies sign function element-wise (-1, 0, or 1)
    pub fn sign(&self) -> Self
    where
        T: Zero + One + PartialOrd
    {
        let mut result = self.clone();
        for i in 0..result.numel() {
            if let Ok(val) = result.get_item_flat(i) {
                let sign_val = if val > <T as TensorElement>::zero() {
                    <T as TensorElement>::one()
                } else if val < <T as TensorElement>::zero() {
                    <T as TensorElement>::zero() - <T as TensorElement>::one() // -1
                } else {
                    <T as TensorElement>::zero()
                };
                let _ = result.set_item_flat(i, sign_val);
            }
        }
        result
    }

    /// Applies ceiling function element-wise
    pub fn ceil(&self) -> Self {
        let mut result = self.clone();
        for i in 0..result.numel() {
            if let Ok(val) = result.get_item_flat(i) {
                if let Some(val_f64) = <T as TensorElement>::to_f64(&val) {
                    let ceil_val = T::from(val_f64.ceil()).unwrap_or(val);
                    let _ = result.set_item_flat(i, ceil_val);
                }
            }
        }
        result
    }

    /// Applies floor function element-wise
    pub fn floor(&self) -> Self {
        let mut result = self.clone();
        for i in 0..result.numel() {
            if let Ok(val) = result.get_item_flat(i) {
                if let Some(val_f64) = <T as TensorElement>::to_f64(&val) {
                    let floor_val = T::from(val_f64.floor()).unwrap_or(val);
                    let _ = result.set_item_flat(i, floor_val);
                }
            }
        }
        result
    }

    /// Applies round function element-wise
    pub fn round(&self) -> Self {
        let mut result = self.clone();
        for i in 0..result.numel() {
            if let Ok(val) = result.get_item_flat(i) {
                if let Some(val_f64) = <T as TensorElement>::to_f64(&val) {
                    let round_val = T::from(val_f64.round()).unwrap_or(val);
                    let _ = result.set_item_flat(i, round_val);
                }
            }
        }
        result
    }

    /// Applies truncate function element-wise
    pub fn trunc(&self) -> Self {
        let mut result = self.clone();
        for i in 0..result.numel() {
            if let Ok(val) = result.get_item_flat(i) {
                if let Some(val_f64) = <T as TensorElement>::to_f64(&val) {
                    let trunc_val = T::from(val_f64.trunc()).unwrap_or(val);
                    let _ = result.set_item_flat(i, trunc_val);
                }
            }
        }
        result
    }

    /// Applies fractional part function element-wise
    pub fn fract(&self) -> Self {
        let mut result = self.clone();
        for i in 0..result.numel() {
            if let Ok(val) = result.get_item_flat(i) {
                if let Some(val_f64) = <T as TensorElement>::to_f64(&val) {
                    let fract_val = T::from(val_f64.fract()).unwrap_or(val);
                    let _ = result.set_item_flat(i, fract_val);
                }
            }
        }
        result
    }

    /// Reciprocal function element-wise (1/x)
    pub fn reciprocal(&self) -> Self
    where
        T: One + std::ops::Div<Output = T>
    {
        let mut result = self.clone();
        for i in 0..result.numel() {
            if let Ok(val) = result.get_item_flat(i) {
                let recip_val = <T as TensorElement>::one() / val;
                let _ = result.set_item_flat(i, recip_val);
            }
        }
        result
    }

    /// Negative function element-wise (-x)
    pub fn neg(&self) -> Self
    where
        T: std::ops::Neg<Output = T> + Copy
    {
        let mut result = self.clone();
        for i in 0..result.numel() {
            if let Ok(val) = result.get_item_flat(i) {
                let neg_val = -val;
                let _ = result.set_item_flat(i, neg_val);
            }
        }
        result
    }

    /// Clamps values to range [min, max]
    ///
    /// # Examples
    ///
    /// ```rust
    /// use torsh_tensor::Tensor;
    ///
    /// # fn main() -> torsh_core::error::Result<()> {
    /// let x = Tensor::<f32>::from_vec(vec![-5.0, 0.5, 2.0, 10.0], &[4])?;
    /// let y = x.clamp(0.0, 1.0);
    /// // y = [0.0, 0.5, 1.0, 1.0]
    /// # Ok(())
    /// # }
    /// ```
    pub fn clamp(&self, min_val: T, max_val: T) -> Self
    where
        T: PartialOrd + Copy
    {
        let mut result = self.clone();
        for i in 0..result.numel() {
            if let Ok(val) = result.get_item_flat(i) {
                let clamped_val = if val < min_val {
                    min_val
                } else if val > max_val {
                    max_val
                } else {
                    val
                };
                let _ = result.set_item_flat(i, clamped_val);
            }
        }
        result
    }

    /// Linear interpolation between two tensors
    pub fn lerp(&self, end: &Self, weight: T) -> Result<Self>
    where
        T: std::ops::Sub<Output = T> + std::ops::Mul<Output = T> + std::ops::Add<Output = T> + One + Copy
    {
        // lerp(start, end, weight) = start + weight * (end - start)
        let diff = end.sub(self)?;
        let weighted_diff = diff.mul_scalar(weight)?;
        self.add(&weighted_diff)
    }

    /// Gaussian error function
    pub fn erf(&self) -> Self {
        let mut result = self.clone();
        for i in 0..result.numel() {
            if let Ok(val) = result.get_item_flat(i) {
                if let Some(val_f64) = <T as TensorElement>::to_f64(&val) {
                    // Approximation of error function using built-in functions
                    let erf_val = if val_f64 >= 0.0 {
                        1.0 - (1.0 + 0.0498673470 * val_f64 + 0.0211410061 * val_f64 * val_f64 +
                               0.0032776263 * val_f64.powi(3) + 0.0000380036 * val_f64.powi(4) +
                               0.0000488906 * val_f64.powi(5) + 0.0000053830 * val_f64.powi(6)).powf(-16.0)
                    } else {
                        -((1.0 + 0.0498673470 * (-val_f64) + 0.0211410061 * val_f64 * val_f64 +
                               0.0032776263 * (-val_f64).powi(3) + 0.0000380036 * val_f64.powi(4) +
                               0.0000488906 * (-val_f64).powi(5) + 0.0000053830 * val_f64.powi(6)).powf(-16.0) - 1.0)
                    };
                    let result_val = T::from(erf_val).unwrap_or(val);
                    let _ = result.set_item_flat(i, result_val);
                }
            }
        }
        result
    }

    /// Complementary error function (1 - erf(x))
    pub fn erfc(&self) -> Self
    where
        T: One + std::ops::Sub<Output = T>
    {
        let erf_result = self.erf();
        let ones = Self::ones(&self.shape().dims(), self.device()).expect("ones tensor creation should succeed");
        ones.sub(&erf_result).expect("erfc subtraction should succeed")
    }

    /// Gamma function using Lanczos approximation
    pub fn gamma(&self) -> Self {
        let mut result = self.clone();
        for i in 0..result.numel() {
            if let Ok(val) = result.get_item_flat(i) {
                if let Some(val_f64) = <T as TensorElement>::to_f64(&val) {
                    let gamma_val_f64 = lanczos_gamma(val_f64);
                    if let Some(gamma_val) = <T as TensorElement>::from_f64(gamma_val_f64) {
                        let _ = result.set_item_flat(i, gamma_val);
                    }
                }
            }
        }
        result
    }

    /// Natural logarithm of gamma function using Lanczos approximation
    pub fn lgamma(&self) -> Self {
        let mut result = self.clone();
        for i in 0..result.numel() {
            if let Ok(val) = result.get_item_flat(i) {
                if let Some(val_f64) = <T as TensorElement>::to_f64(&val) {
                    let lgamma_val_f64 = lanczos_lgamma(val_f64);
                    if let Some(lgamma_val) = <T as TensorElement>::from_f64(lgamma_val_f64) {
                        let _ = result.set_item_flat(i, lgamma_val);
                    }
                }
            }
        }
        result
    }

    /// Determines if values are finite (not infinite or NaN)
    pub fn isfinite(&self) -> Tensor<bool> {
        let mut result = Tensor::<bool>::zeros(&self.shape().dims(), self.device()).expect("zeros tensor creation should succeed for isfinite");
        for i in 0..self.numel() {
            if let Ok(val) = self.get_item_flat(i) {
                let is_finite = if let Some(val_f64) = <T as TensorElement>::to_f64(&val) {
                    val_f64.is_finite()
                } else {
                    true
                };
                let _ = result.set_item_flat(i, is_finite);
            }
        }
        result
    }

    /// Determines if values are infinite
    pub fn isinf(&self) -> Tensor<bool> {
        let mut result = Tensor::<bool>::zeros(&self.shape().dims(), self.device()).expect("zeros tensor creation should succeed for isinf");
        for i in 0..self.numel() {
            if let Ok(val) = self.get_item_flat(i) {
                let is_inf = if let Some(val_f64) = <T as TensorElement>::to_f64(&val) {
                    val_f64.is_infinite()
                } else {
                    false
                };
                let _ = result.set_item_flat(i, is_inf);
            }
        }
        result
    }

    /// Determines if values are NaN
    pub fn isnan(&self) -> Tensor<bool> {
        let mut result = Tensor::<bool>::zeros(&self.shape().dims(), self.device()).expect("zeros tensor creation should succeed for isnan");
        for i in 0..self.numel() {
            if let Ok(val) = self.get_item_flat(i) {
                let is_nan = if let Some(val_f64) = <T as TensorElement>::to_f64(&val) {
                    val_f64.is_nan()
                } else {
                    false
                };
                let _ = result.set_item_flat(i, is_nan);
            }
        }
        result
    }

    /// Replace NaN values with specified value
    pub fn nan_to_num(&self, nan_val: T, posinf_val: Option<T>, neginf_val: Option<T>) -> Self
    where
        T: Copy
    {
        let mut result = self.clone();
        for i in 0..result.numel() {
            if let Ok(val) = result.get_item_flat(i) {
                let new_val = if let Some(val_f64) = <T as TensorElement>::to_f64(&val) {
                    if val_f64.is_nan() {
                        nan_val
                    } else if val_f64.is_infinite() && val_f64 > 0.0 {
                        posinf_val.unwrap_or(val)
                    } else if val_f64.is_infinite() && val_f64 < 0.0 {
                        neginf_val.unwrap_or(val)
                    } else {
                        val
                    }
                } else {
                    val
                };
                let _ = result.set_item_flat(i, new_val);
            }
        }
        result
    }

    /// Create a tensor of uniformly spaced values
    pub fn linspace(start: T, end: T, steps: usize, device: &dyn torsh_core::Device) -> Result<Self>
    where
        T: std::ops::Sub<Output = T> + std::ops::Div<f64, Output = T> + std::ops::Add<Output = T> + Copy + ToPrimitive
    {
        if steps == 0 {
            return Err(TorshError::Other("Number of steps must be greater than 0".to_string()));
        }

        let mut result = Self::zeros(&[steps], device.device_type())?;

        if steps == 1 {
            result.set_item(&[0], start)?;
            return Ok(result);
        }

        let step_size = (end - start) / T::from((steps - 1) as f64).unwrap_or(end);

        for i in 0..steps {
            let i_val = T::from(i as f64).unwrap_or(<T as TensorElement>::zero());
            let val = start + step_size * i_val;
            result.set_item(&[i], val)?;
        }

        Ok(result)
    }

    /// Create a tensor of logarithmically spaced values
    pub fn logspace(start: T, end: T, steps: usize, base: f64, device: &dyn torsh_core::Device) -> Result<Self>
    where
        T: Copy + ToPrimitive + std::ops::Sub<Output = T> + std::ops::Div<f64, Output = T> + std::ops::Add<Output = T>
    {
        let linear = Self::linspace(start, end, steps, device)?;

        // Apply base^x to each element
        let mut result = linear;
        for i in 0..result.numel() {
            if let Ok(val) = result.get_item_flat(i) {
                if let Some(val_f64) = <T as TensorElement>::to_f64(&val) {
                    let log_val = T::from(base.powf(val_f64)).unwrap_or(val);
                    let _ = result.set_item_flat(i, log_val);
                }
            }
        }

        Ok(result)
    }
}

// =============================================================================
// Helper functions for special mathematical functions
// =============================================================================

/// Lanczos approximation for the gamma function
/// Uses g=7 with 9 coefficients for good accuracy
fn lanczos_gamma(z: f64) -> f64 {
    // Lanczos coefficients for g=7
    const LANCZOS_G: f64 = 7.0;
    const LANCZOS_COEFF: [f64; 9] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];

    if z < 0.5 {
        // Use reflection formula: Γ(z)Γ(1-z) = π/sin(πz)
        std::f64::consts::PI / ((std::f64::consts::PI * z).sin() * lanczos_gamma(1.0 - z))
    } else {
        let z = z - 1.0;
        let mut x = LANCZOS_COEFF[0];
        for i in 1..9 {
            x += LANCZOS_COEFF[i] / (z + i as f64);
        }
        let t = z + LANCZOS_G + 0.5;
        let sqrt_2pi = (2.0 * std::f64::consts::PI).sqrt();
        sqrt_2pi * t.powf(z + 0.5) * (-t).exp() * x
    }
}

/// Natural logarithm of gamma function using Lanczos approximation
fn lanczos_lgamma(z: f64) -> f64 {
    // Lanczos coefficients for g=7
    const LANCZOS_G: f64 = 7.0;
    const LANCZOS_COEFF: [f64; 9] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];

    if z < 0.5 {
        // Use reflection formula for log: ln(Γ(z)) = ln(π/sin(πz)) - ln(Γ(1-z))
        let sin_pi_z = (std::f64::consts::PI * z).sin();
        (std::f64::consts::PI / sin_pi_z).ln() - lanczos_lgamma(1.0 - z)
    } else {
        let z = z - 1.0;
        let mut x = LANCZOS_COEFF[0];
        for i in 1..9 {
            x += LANCZOS_COEFF[i] / (z + i as f64);
        }
        let t = z + LANCZOS_G + 0.5;
        let log_sqrt_2pi = 0.5 * (2.0 * std::f64::consts::PI).ln();
        log_sqrt_2pi + (z + 0.5) * t.ln() - t + x.ln()
    }
}

// ✅ NaN/Inf detection operations for PyTorch compatibility
impl<T: TensorElement + Copy> Tensor<T> {
    /// Check if elements are NaN
    ///
    /// # PyTorch Compatibility
    /// Equivalent to `torch.isnan(tensor)`
    pub fn isnan(&self) -> Result<Tensor<bool>>
    where
        T: FloatElement,
    {
        let data = self.data()?;
        let result: Vec<bool> = data.iter().map(|&x| x.is_nan()).collect();
        Tensor::<bool>::from_data(result, self.shape().to_vec(), self.device())
    }

    /// Check if elements are infinite
    ///
    /// # PyTorch Compatibility
    /// Equivalent to `torch.isinf(tensor)`
    pub fn isinf(&self) -> Result<Tensor<bool>>
    where
        T: FloatElement,
    {
        let data = self.data()?;
        let result: Vec<bool> = data.iter().map(|&x| x.is_infinite()).collect();
        Tensor::<bool>::from_data(result, self.shape().to_vec(), self.device())
    }

    /// Check if elements are finite (not NaN and not infinite)
    ///
    /// # PyTorch Compatibility
    /// Equivalent to `torch.isfinite(tensor)`
    pub fn isfinite(&self) -> Result<Tensor<bool>>
    where
        T: FloatElement,
    {
        let data = self.data()?;
        let result: Vec<bool> = data.iter().map(|&x| x.is_finite()).collect();
        Tensor::<bool>::from_data(result, self.shape().to_vec(), self.device())
    }

    /// Check if two tensors are element-wise equal within tolerance
    ///
    /// # PyTorch Compatibility
    /// Equivalent to `torch.allclose(tensor, other, rtol, atol)`
    pub fn allclose(&self, other: &Self, rtol: f64, atol: f64) -> Result<bool>
    where
        T: FloatElement,
    {
        if self.shape() != other.shape() {
            return Ok(false);
        }

        let self_data = self.data()?;
        let other_data = other.data()?;

        for (&a, &b) in self_data.iter().zip(other_data.iter()) {
            let a_f64 = a.to_f64().ok_or_else(|| {
                TorshError::ConversionError("Cannot convert to f64".to_string())
            })?;
            let b_f64 = b.to_f64().ok_or_else(|| {
                TorshError::ConversionError("Cannot convert to f64".to_string())
            })?;

            let diff = (a_f64 - b_f64).abs();
            let threshold = atol + rtol * b_f64.abs();

            if diff > threshold {
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Check if two tensors are element-wise close within tolerance
    ///
    /// # PyTorch Compatibility
    /// Equivalent to `torch.isclose(tensor, other, rtol, atol)`
    pub fn isclose(&self, other: &Self, rtol: f64, atol: f64) -> Result<Tensor<bool>>
    where
        T: FloatElement,
    {
        if self.shape() != other.shape() {
            return Err(TorshError::ShapeMismatch {
                expected: self.shape().to_vec(),
                got: other.shape().to_vec(),
            });
        }

        let self_data = self.data()?;
        let other_data = other.data()?;

        let result: Vec<bool> = self_data
            .iter()
            .zip(other_data.iter())
            .map(|(&a, &b)| {
                let a_f64 = a.to_f64().unwrap_or(0.0);
                let b_f64 = b.to_f64().unwrap_or(0.0);

                let diff = (a_f64 - b_f64).abs();
                let threshold = atol + rtol * b_f64.abs();

                diff <= threshold
            })
            .collect();

        Tensor::<bool>::from_data(result, self.shape().to_vec(), self.device())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::device::DeviceType;

    #[test]
    fn test_isnan_with_nan() {
        let tensor = Tensor::from_data(
            vec![1.0f32, f32::NAN, 3.0, f32::NAN],
            vec![4],
            DeviceType::Cpu,
        ).unwrap();

        let result = tensor.isnan().unwrap();
        let data = result.data().unwrap();

        assert_eq!(data, vec![false, true, false, true]);
    }

    #[test]
    fn test_isnan_no_nan() {
        let tensor = Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], DeviceType::Cpu).unwrap();

        let result = tensor.isnan().unwrap();
        let data = result.data().unwrap();

        assert_eq!(data, vec![false, false, false]);
    }

    #[test]
    fn test_isinf_with_inf() {
        let tensor = Tensor::from_data(
            vec![1.0f32, f32::INFINITY, -f32::INFINITY, 3.0],
            vec![4],
            DeviceType::Cpu,
        ).unwrap();

        let result = tensor.isinf().unwrap();
        let data = result.data().unwrap();

        assert_eq!(data, vec![false, true, true, false]);
    }

    #[test]
    fn test_isinf_no_inf() {
        let tensor = Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], DeviceType::Cpu).unwrap();

        let result = tensor.isinf().unwrap();
        let data = result.data().unwrap();

        assert_eq!(data, vec![false, false, false]);
    }

    #[test]
    fn test_isfinite_mixed() {
        let tensor = Tensor::from_data(
            vec![1.0f32, f32::NAN, f32::INFINITY, 3.0],
            vec![4],
            DeviceType::Cpu,
        ).unwrap();

        let result = tensor.isfinite().unwrap();
        let data = result.data().unwrap();

        assert_eq!(data, vec![true, false, false, true]);
    }

    #[test]
    fn test_isfinite_all_finite() {
        let tensor = Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], DeviceType::Cpu).unwrap();

        let result = tensor.isfinite().unwrap();
        let data = result.data().unwrap();

        assert_eq!(data, vec![true, true, true]);
    }

    #[test]
    fn test_allclose_identical() {
        let a = Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], DeviceType::Cpu).unwrap();
        let b = Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], DeviceType::Cpu).unwrap();

        let result = a.allclose(&b, 1e-5, 1e-8).unwrap();

        assert!(result);
    }

    #[test]
    fn test_allclose_within_tolerance() {
        let a = Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], DeviceType::Cpu).unwrap();
        let b = Tensor::from_data(vec![1.00001f32, 2.00001, 3.00001], vec![3], DeviceType::Cpu).unwrap();

        let result = a.allclose(&b, 1e-3, 1e-3).unwrap();

        assert!(result);
    }

    #[test]
    fn test_allclose_exceeds_tolerance() {
        let a = Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], DeviceType::Cpu).unwrap();
        let b = Tensor::from_data(vec![1.1f32, 2.1, 3.1], vec![3], DeviceType::Cpu).unwrap();

        let result = a.allclose(&b, 1e-5, 1e-5).unwrap();

        assert!(!result);
    }

    #[test]
    fn test_allclose_shape_mismatch() {
        let a = Tensor::from_data(vec![1.0f32, 2.0], vec![2], DeviceType::Cpu).unwrap();
        let b = Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], DeviceType::Cpu).unwrap();

        let result = a.allclose(&b, 1e-5, 1e-8).unwrap();

        assert!(!result); // Shape mismatch should return false
    }

    #[test]
    fn test_isclose_identical() {
        let a = Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], DeviceType::Cpu).unwrap();
        let b = Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], DeviceType::Cpu).unwrap();

        let result = a.isclose(&b, 1e-5, 1e-8).unwrap();
        let data = result.data().unwrap();

        assert_eq!(data, vec![true, true, true]);
    }

    #[test]
    fn test_isclose_mixed() {
        let a = Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], DeviceType::Cpu).unwrap();
        let b = Tensor::from_data(vec![1.00001f32, 2.1, 3.00001], vec![3], DeviceType::Cpu).unwrap();

        let result = a.isclose(&b, 1e-3, 1e-3).unwrap();
        let data = result.data().unwrap();

        // First and third are close, middle is not
        assert_eq!(data, vec![true, false, true]);
    }

    #[test]
    fn test_isclose_shape_mismatch() {
        let a = Tensor::from_data(vec![1.0f32, 2.0], vec![2], DeviceType::Cpu).unwrap();
        let b = Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], DeviceType::Cpu).unwrap();

        let result = a.isclose(&b, 1e-5, 1e-8);

        assert!(result.is_err()); // Shape mismatch should return error
    }
}