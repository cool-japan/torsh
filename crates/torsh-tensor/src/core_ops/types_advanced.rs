/// Advanced operations for Tensor: comparisons, shape manipulation, logical ops, and Operation enum.
///
/// This module is included by `types.rs` via `#[path = "types_advanced.rs"]`.
use super::*;

/// Comparison operations for tensors
impl<T: TensorElement + PartialOrd + Copy> Tensor<T> {
    /// Element-wise greater than comparison
    pub fn gt(&self, other: &Self) -> Result<Tensor<bool>> {
        if self.shape != other.shape {
            return Err(TorshError::ShapeMismatch {
                expected: self.shape.dims().to_vec(),
                got: other.shape.dims().to_vec(),
            });
        }
        let self_data = self.to_vec()?;
        let other_data = other.to_vec()?;
        let result_data: Vec<bool> = self_data
            .iter()
            .zip(other_data.iter())
            .map(|(&a, &b)| a > b)
            .collect();
        Tensor::from_data(result_data, self.shape.dims().to_vec(), self.device)
    }
    /// Element-wise less than comparison
    pub fn lt(&self, other: &Self) -> Result<Tensor<bool>> {
        if self.shape != other.shape {
            return Err(TorshError::ShapeMismatch {
                expected: self.shape.dims().to_vec(),
                got: other.shape.dims().to_vec(),
            });
        }
        let self_data = self.to_vec()?;
        let other_data = other.to_vec()?;
        let result_data: Vec<bool> = self_data
            .iter()
            .zip(other_data.iter())
            .map(|(&a, &b)| a < b)
            .collect();
        Tensor::from_data(result_data, self.shape.dims().to_vec(), self.device)
    }
    /// Element-wise greater than or equal comparison
    pub fn ge(&self, other: &Self) -> Result<Tensor<bool>> {
        if self.shape != other.shape {
            return Err(TorshError::ShapeMismatch {
                expected: self.shape.dims().to_vec(),
                got: other.shape.dims().to_vec(),
            });
        }
        let self_data = self.to_vec()?;
        let other_data = other.to_vec()?;
        let result_data: Vec<bool> = self_data
            .iter()
            .zip(other_data.iter())
            .map(|(&a, &b)| a >= b)
            .collect();
        Tensor::from_data(result_data, self.shape.dims().to_vec(), self.device)
    }
    /// Element-wise less than or equal comparison
    pub fn le(&self, other: &Self) -> Result<Tensor<bool>> {
        if self.shape != other.shape {
            return Err(TorshError::ShapeMismatch {
                expected: self.shape.dims().to_vec(),
                got: other.shape.dims().to_vec(),
            });
        }
        let self_data = self.to_vec()?;
        let other_data = other.to_vec()?;
        let result_data: Vec<bool> = self_data
            .iter()
            .zip(other_data.iter())
            .map(|(&a, &b)| a <= b)
            .collect();
        Tensor::from_data(result_data, self.shape.dims().to_vec(), self.device)
    }
    /// Element-wise equality comparison
    pub fn eq(&self, other: &Self) -> Result<Tensor<bool>> {
        if self.shape != other.shape {
            return Err(TorshError::ShapeMismatch {
                expected: self.shape.dims().to_vec(),
                got: other.shape.dims().to_vec(),
            });
        }
        let self_data = self.to_vec()?;
        let other_data = other.to_vec()?;
        let result_data: Vec<bool> = self_data
            .iter()
            .zip(other_data.iter())
            .map(|(&a, &b)| a == b)
            .collect();
        Tensor::from_data(result_data, self.shape.dims().to_vec(), self.device)
    }
    /// Element-wise inequality comparison
    pub fn ne(&self, other: &Self) -> Result<Tensor<bool>> {
        if self.shape != other.shape {
            return Err(TorshError::ShapeMismatch {
                expected: self.shape.dims().to_vec(),
                got: other.shape.dims().to_vec(),
            });
        }
        let self_data = self.to_vec()?;
        let other_data = other.to_vec()?;
        let result_data: Vec<bool> = self_data
            .iter()
            .zip(other_data.iter())
            .map(|(&a, &b)| a != b)
            .collect();
        Tensor::from_data(result_data, self.shape.dims().to_vec(), self.device)
    }
    /// Scalar comparison methods
    /// Element-wise equality comparison with scalar
    pub fn eq_scalar(&self, value: T) -> Result<Tensor<bool>>
    where
        T: PartialEq + Copy,
    {
        let self_data = self.to_vec()?;
        let result_data: Vec<bool> = self_data.iter().map(|&a| a == value).collect();
        Tensor::from_data(result_data, self.shape.dims().to_vec(), self.device)
    }
    /// Element-wise inequality comparison with scalar
    pub fn ne_scalar(&self, value: T) -> Result<Tensor<bool>>
    where
        T: PartialEq + Copy,
    {
        let self_data = self.to_vec()?;
        let result_data: Vec<bool> = self_data.iter().map(|&a| a != value).collect();
        Tensor::from_data(result_data, self.shape.dims().to_vec(), self.device)
    }
    /// Element-wise greater than comparison with scalar
    pub fn gt_scalar(&self, value: T) -> Result<Tensor<bool>>
    where
        T: PartialOrd + Copy,
    {
        let self_data = self.to_vec()?;
        let result_data: Vec<bool> = self_data.iter().map(|&a| a > value).collect();
        Tensor::from_data(result_data, self.shape.dims().to_vec(), self.device)
    }
    /// Element-wise less than comparison with scalar
    pub fn lt_scalar(&self, value: T) -> Result<Tensor<bool>>
    where
        T: PartialOrd + Copy,
    {
        let self_data = self.to_vec()?;
        let result_data: Vec<bool> = self_data.iter().map(|&a| a < value).collect();
        Tensor::from_data(result_data, self.shape.dims().to_vec(), self.device)
    }
    /// Element-wise less than or equal comparison with scalar
    pub fn le_scalar(&self, value: T) -> Result<Tensor<bool>>
    where
        T: PartialOrd + Copy,
    {
        let self_data = self.to_vec()?;
        let result_data: Vec<bool> = self_data.iter().map(|&a| a <= value).collect();
        Tensor::from_data(result_data, self.shape.dims().to_vec(), self.device)
    }
    /// Element-wise greater than or equal comparison with scalar
    pub fn ge_scalar(&self, value: T) -> Result<Tensor<bool>>
    where
        T: PartialOrd + Copy,
    {
        let self_data = self.to_vec()?;
        let result_data: Vec<bool> = self_data.iter().map(|&a| a >= value).collect();
        Tensor::from_data(result_data, self.shape.dims().to_vec(), self.device)
    }
}
/// Shape manipulation operations for tensors
impl<T: TensorElement> Tensor<T> {
    /// Flatten tensor to 1D
    pub fn flatten(&self) -> Result<Self> {
        let total_elements = self.numel();
        self.view(&[total_elements as i32])
    }
    /// Conditional tensor selection - where condition is true, select from self, otherwise from other
    pub fn where_tensor(&self, condition: &Tensor<bool>, other: &Self) -> Result<Self> {
        if self.shape != condition.shape || self.shape != other.shape {
            return Err(TorshError::ShapeMismatch {
                expected: self.shape.dims().to_vec(),
                got: condition.shape.dims().to_vec(),
            });
        }
        let self_data = self.to_vec()?;
        let condition_data = condition.to_vec()?;
        let other_data = other.to_vec()?;
        let result_data: Vec<T> = self_data
            .iter()
            .zip(condition_data.iter())
            .zip(other_data.iter())
            .map(
                |((&self_val, &cond), &other_val)| {
                    if cond {
                        self_val
                    } else {
                        other_val
                    }
                },
            )
            .collect();
        Tensor::from_data(result_data, self.shape.dims().to_vec(), self.device)
    }
    /// Add bias vector to tensor (element-wise addition)
    pub fn add_bias(&self, bias: &Self) -> Result<Self>
    where
        T: std::ops::Add<Output = T>,
    {
        self.add(bias)
    }
}
/// Logical operations for boolean tensors
impl Tensor<bool> {
    /// Element-wise logical AND operation
    pub fn logical_and(&self, other: &Self) -> Result<Self> {
        if self.shape != other.shape {
            return Err(TorshError::ShapeMismatch {
                expected: self.shape.dims().to_vec(),
                got: other.shape.dims().to_vec(),
            });
        }
        let self_data = self.to_vec()?;
        let other_data = other.to_vec()?;
        let result_data: Vec<bool> = self_data
            .iter()
            .zip(other_data.iter())
            .map(|(&a, &b)| a && b)
            .collect();
        Tensor::from_data(result_data, self.shape.dims().to_vec(), self.device)
    }
    /// Element-wise logical OR operation
    pub fn logical_or(&self, other: &Self) -> Result<Self> {
        if self.shape != other.shape {
            return Err(TorshError::ShapeMismatch {
                expected: self.shape.dims().to_vec(),
                got: other.shape.dims().to_vec(),
            });
        }
        let self_data = self.to_vec()?;
        let other_data = other.to_vec()?;
        let result_data: Vec<bool> = self_data
            .iter()
            .zip(other_data.iter())
            .map(|(&a, &b)| a || b)
            .collect();
        Tensor::from_data(result_data, self.shape.dims().to_vec(), self.device)
    }
    /// Element-wise logical XOR operation
    pub fn logical_xor(&self, other: &Self) -> Result<Self> {
        if self.shape != other.shape {
            return Err(TorshError::ShapeMismatch {
                expected: self.shape.dims().to_vec(),
                got: other.shape.dims().to_vec(),
            });
        }
        let self_data = self.to_vec()?;
        let other_data = other.to_vec()?;
        let result_data: Vec<bool> = self_data
            .iter()
            .zip(other_data.iter())
            .map(|(&a, &b)| a ^ b)
            .collect();
        Tensor::from_data(result_data, self.shape.dims().to_vec(), self.device)
    }
}
/// Operation type for gradient computation
#[derive(Debug, Clone)]
pub enum Operation<T: TensorElement> {
    /// Leaf node (no operation)
    Leaf,
    /// Power operation: x^n
    Power {
        input: Arc<Tensor<T>>,
        exponent: f32,
    },
    /// Addition operation: a + b
    Add {
        lhs: Arc<Tensor<T>>,
        rhs: Arc<Tensor<T>>,
    },
    /// Subtraction operation: a - b
    Sub {
        lhs: Arc<Tensor<T>>,
        rhs: Arc<Tensor<T>>,
    },
    /// Multiplication operation: a * b
    Mul {
        lhs: Arc<Tensor<T>>,
        rhs: Arc<Tensor<T>>,
    },
    /// Mean reduction operation: mean(input)
    Mean {
        /// The input tensor before reduction
        input: Arc<Tensor<T>>,
        /// Number of elements that were averaged
        count: f64,
    },
    /// Custom operation with name and inputs
    Custom(String, Vec<Weak<Tensor<T>>>),
}
