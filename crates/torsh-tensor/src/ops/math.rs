//! Mathematical functions for tensors

use crate::{Tensor, TensorElement, FloatElement};
use torsh_core::Device;
use torsh_core::error::{Result, TorshError};
use num_traits::{Float, Zero, One, cast::ToPrimitive};
use std::f64::consts;

impl<T: FloatElement + Default> Tensor<T> {
    /// Applies sin function element-wise
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


    /// Applies exp function element-wise
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

    /// Applies natural logarithm function element-wise
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

    /// Applies power function element-wise (self^exponent)
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
        let ones = Self::ones(&self.shape().dims(), self.device()).unwrap();
        ones.sub(&erf_result).unwrap()
    }

    /// Gamma function
    pub fn gamma(&self) -> Self {
        let mut result = self.clone();
        for i in 0..result.numel() {
            if let Ok(val) = result.get_item_flat(i) {
                if let Some(val_f64) = <T as TensorElement>::to_f64(&val) {
                    // TODO: Implement gamma function (currently unstable in std)
                    let gamma_val = val; // Placeholder
                    let _ = result.set_item_flat(i, gamma_val);
                }
            }
        }
        result
    }

    /// Natural logarithm of gamma function
    pub fn lgamma(&self) -> Self {
        let mut result = self.clone();
        for i in 0..result.numel() {
            if let Ok(val) = result.get_item_flat(i) {
                if let Some(val_f64) = <T as TensorElement>::to_f64(&val) {
                    // TODO: Implement ln_gamma function (currently unstable in std)
                    let lgamma_val = val; // Placeholder
                    let _ = result.set_item_flat(i, lgamma_val);
                }
            }
        }
        result
    }

    /// Determines if values are finite (not infinite or NaN)
    pub fn isfinite(&self) -> Tensor<bool> {
        let mut result = Tensor::<bool>::zeros(&self.shape().dims(), self.device()).unwrap();
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
        let mut result = Tensor::<bool>::zeros(&self.shape().dims(), self.device()).unwrap();
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
        let mut result = Tensor::<bool>::zeros(&self.shape().dims(), self.device()).unwrap();
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