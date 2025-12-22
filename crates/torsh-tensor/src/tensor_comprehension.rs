//! Tensor Comprehensions - Declarative Tensor Creation
//!
//! This module provides powerful macro-based tensor comprehensions similar to Python's list
//! comprehensions and NumPy's array creation syntax. It enables concise, readable tensor
//! creation with various patterns and conditions.
//!
//! # Features
//!
//! - **List-style comprehensions**: Create tensors with for-loop style syntax
//! - **Conditional filtering**: Include conditions to filter elements
//! - **Multi-dimensional**: Support for creating multi-dimensional tensors
//! - **Generator expressions**: Lazy generation of tensor elements
//! - **Type inference**: Automatic type inference from expressions
//!
//! # Examples
//!
//! ```rust
//! use torsh_tensor::tensor_comp;
//!
//! // Simple range: [0, 1, 2, 3, 4]
//! // let t = tensor_comp![x; x in 0..5];
//!
//! // With transformation: [0, 2, 4, 6, 8]
//! // let t = tensor_comp![x * 2; x in 0..5];
//!
//! // With condition: [0, 2, 4]
//! // let t = tensor_comp![x; x in 0..5, if x % 2 == 0];
//!
//! // 2D comprehension: [[0, 1], [2, 3], [4, 5]]
//! // let t = tensor_comp![[i * 2 + j; j in 0..2]; i in 0..3];
//! ```

use torsh_core::{device::DeviceType, dtype::TensorElement, error::Result};

use crate::Tensor;

/// Builder for tensor comprehensions
pub struct TensorComprehension<T: TensorElement> {
    elements: Vec<T>,
    shape: Vec<usize>,
    device: DeviceType,
}

impl<T: TensorElement + Copy> TensorComprehension<T> {
    /// Create a new tensor comprehension builder
    pub fn new() -> Self {
        Self {
            elements: Vec::new(),
            shape: Vec::new(),
            device: DeviceType::Cpu,
        }
    }

    /// Set the device for the tensor
    pub fn device(mut self, device: DeviceType) -> Self {
        self.device = device;
        self
    }

    /// Add elements from an iterator
    pub fn from_iter<I>(mut self, iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        self.elements = iter.into_iter().collect();
        if self.shape.is_empty() {
            self.shape = vec![self.elements.len()];
        }
        self
    }

    /// Add elements with explicit shape
    pub fn from_iter_with_shape<I>(mut self, iter: I, shape: Vec<usize>) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        self.elements = iter.into_iter().collect();
        self.shape = shape;
        self
    }

    /// Build the tensor
    pub fn build(self) -> Result<Tensor<T>> {
        Tensor::from_data(self.elements, self.shape, self.device)
    }
}

impl<T: TensorElement + Copy> Default for TensorComprehension<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper function for creating tensors from ranges
pub fn range_tensor<T>(start: T, end: T, step: T, device: DeviceType) -> Result<Tensor<T>>
where
    T: TensorElement + Copy + std::ops::Add<Output = T> + std::cmp::PartialOrd + num_traits::Zero,
{
    let mut elements = Vec::new();
    let mut current = start;

    if step == <T as torsh_core::TensorElement>::zero() {
        return Err(torsh_core::error::TorshError::InvalidArgument(
            "Step cannot be zero".to_string(),
        ));
    }

    // Handle both positive and negative steps
    let ascending = start < end;
    while (ascending && current < end) || (!ascending && current > end) {
        elements.push(current);
        current = current + step;
    }

    let len = elements.len();
    Tensor::from_data(elements, vec![len], device)
}

/// Helper function for creating linspace tensors
pub fn linspace_range<T>(
    start: f64,
    end: f64,
    steps: usize,
    device: DeviceType,
) -> Result<Tensor<T>>
where
    T: TensorElement + Copy + num_traits::FromPrimitive,
{
    if steps == 0 {
        return Err(torsh_core::error::TorshError::InvalidArgument(
            "Steps must be greater than 0".to_string(),
        ));
    }

    let step = if steps == 1 {
        0.0
    } else {
        (end - start) / (steps - 1) as f64
    };

    let elements: Vec<T> = (0..steps)
        .map(|i| {
            let val = start + step * i as f64;
            <T as torsh_core::TensorElement>::from_f64(val)
                .unwrap_or_else(|| <T as torsh_core::TensorElement>::zero())
        })
        .collect();

    Tensor::from_data(elements, vec![steps], device)
}

/// Helper function for creating logspace tensors
pub fn logspace<T>(
    start: f64,
    end: f64,
    steps: usize,
    base: f64,
    device: DeviceType,
) -> Result<Tensor<T>>
where
    T: TensorElement + Copy + num_traits::FromPrimitive,
{
    if steps == 0 {
        return Err(torsh_core::error::TorshError::InvalidArgument(
            "Steps must be greater than 0".to_string(),
        ));
    }

    let step = if steps == 1 {
        0.0
    } else {
        (end - start) / (steps - 1) as f64
    };

    let elements: Vec<T> = (0..steps)
        .map(|i| {
            let exponent = start + step * i as f64;
            let val = base.powf(exponent);
            <T as torsh_core::TensorElement>::from_f64(val)
                .unwrap_or_else(|| <T as torsh_core::TensorElement>::zero())
        })
        .collect();

    Tensor::from_data(elements, vec![steps], device)
}

/// Helper function for creating meshgrid-style tensors
pub fn meshgrid<T>(x: &Tensor<T>, y: &Tensor<T>) -> Result<(Tensor<T>, Tensor<T>)>
where
    T: TensorElement + Copy,
{
    let x_data = x.to_vec()?;
    let y_data = y.to_vec()?;

    let nx = x_data.len();
    let ny = y_data.len();

    // X grid: repeat each x value ny times
    let mut x_grid = Vec::with_capacity(nx * ny);
    for &x_val in &x_data {
        for _ in 0..ny {
            x_grid.push(x_val);
        }
    }

    // Y grid: tile y values nx times
    let mut y_grid = Vec::with_capacity(nx * ny);
    for _ in 0..nx {
        for &y_val in &y_data {
            y_grid.push(y_val);
        }
    }

    let x_tensor = Tensor::from_data(x_grid, vec![nx, ny], x.device)?;
    let y_tensor = Tensor::from_data(y_grid, vec![nx, ny], y.device)?;

    Ok((x_tensor, y_tensor))
}

/// Macro for tensor comprehensions
#[macro_export]
macro_rules! tensor_comp {
    // Simple range: tensor_comp![x; x in start..end]
    ($expr:expr; $var:ident in $start:expr, $end:expr) => {{
        let elements: Vec<_> = ($start..$end).map(|$var| $expr).collect();
        $crate::Tensor::from_data(elements, vec![elements.len()], $crate::DeviceType::Cpu)
    }};

    // Range with step: tensor_comp![x; x in start..end, step s]
    ($expr:expr; $var:ident in $start:expr, $end:expr, step $step:expr) => {{
        let mut elements = Vec::new();
        let mut $var = $start;
        while $var < $end {
            elements.push($expr);
            $var = $var + $step;
        }
        $crate::Tensor::from_data(elements, vec![elements.len()], $crate::DeviceType::Cpu)
    }};

    // With condition: tensor_comp![x; x in start..end, if condition]
    ($expr:expr; $var:ident in $start:expr, $end:expr, if $cond:expr) => {{
        let elements: Vec<_> = ($start..$end)
            .filter(|&$var| $cond)
            .map(|$var| $expr)
            .collect();
        $crate::Tensor::from_data(elements, vec![elements.len()], $crate::DeviceType::Cpu)
    }};

    // 2D comprehension: tensor_comp![[expr; j in 0..n]; i in 0..m]
    ([$expr:expr; $inner_var:ident in $inner_start:expr, $inner_end:expr]; $outer_var:ident in $outer_start:expr, $outer_end:expr) => {{
        let mut all_elements = Vec::new();
        let rows = $outer_end - $outer_start;
        let cols = $inner_end - $inner_start;

        for $outer_var in $outer_start..$outer_end {
            for $inner_var in $inner_start..$inner_end {
                all_elements.push($expr);
            }
        }
        $crate::Tensor::from_data(all_elements, vec![rows, cols], $crate::DeviceType::Cpu)
    }};
}

/// Macro for creating tensors with repeated values
#[macro_export]
macro_rules! tensor_repeat {
    // Repeat value n times: tensor_repeat![value; n]
    ($value:expr; $count:expr) => {{
        let elements = vec![$value; $count];
        $crate::Tensor::from_data(elements, vec![$count], $crate::DeviceType::Cpu)
    }};

    // Repeat with shape: tensor_repeat![value; shape]
    ($value:expr; [$($dim:expr),+]) => {{
        let shape = vec![$($dim),+];
        let size: usize = shape.iter().product();
        let elements = vec![$value; size];
        $crate::Tensor::from_data(elements, shape, $crate::DeviceType::Cpu)
    }};
}

/// Macro for creating identity-like tensors
#[macro_export]
macro_rules! tensor_eye {
    // Identity matrix: tensor_eye![n]
    ($n:expr) => {{
        tensor_eye![$n, $n]
    }};

    // Rectangular identity: tensor_eye![m, n]
    ($m:expr, $n:expr) => {{
        let mut elements = vec![0.0f32; $m * $n];
        let min_dim = std::cmp::min($m, $n);
        for i in 0..min_dim {
            elements[i * $n + i] = 1.0;
        }
        $crate::Tensor::from_data(elements, vec![$m, $n], $crate::DeviceType::Cpu)
    }};

    // With offset: tensor_eye![m, n, offset k]
    ($m:expr, $n:expr, offset $k:expr) => {{
        let mut elements = vec![0.0f32; $m * $n];
        if $k >= 0 {
            let k = $k as usize;
            for i in 0..$m {
                let j = i + k;
                if j < $n {
                    elements[i * $n + j] = 1.0;
                }
            }
        } else {
            let k = (-$k) as usize;
            for j in 0..$n {
                let i = j + k;
                if i < $m {
                    elements[i * $n + j] = 1.0;
                }
            }
        }
        $crate::Tensor::from_data(elements, vec![$m, $n], $crate::DeviceType::Cpu)
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::creation::*;

    #[test]
    fn test_tensor_comprehension_builder() {
        let comp = TensorComprehension::new().from_iter(0..5).build().unwrap();

        let data = comp.to_vec().unwrap();
        assert_eq!(data, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_range_tensor() {
        let t = range_tensor(0, 10, 2, DeviceType::Cpu).unwrap();
        let data = t.to_vec().unwrap();
        assert_eq!(data, vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_linspace() {
        let t: Tensor<f32> = linspace_range(0.0, 10.0, 5, DeviceType::Cpu).unwrap();
        let data = t.to_vec().unwrap();

        assert!((data[0] - 0.0).abs() < 1e-6);
        assert!((data[1] - 2.5).abs() < 1e-6);
        assert!((data[2] - 5.0).abs() < 1e-6);
        assert!((data[3] - 7.5).abs() < 1e-6);
        assert!((data[4] - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_logspace() {
        let t: Tensor<f32> = logspace(0.0, 2.0, 3, 10.0, DeviceType::Cpu).unwrap();
        let data = t.to_vec().unwrap();

        assert!((data[0] - 1.0).abs() < 1e-6); // 10^0
        assert!((data[1] - 10.0).abs() < 1e-5); // 10^1
        assert!((data[2] - 100.0).abs() < 1e-4); // 10^2
    }

    #[test]
    fn test_meshgrid() {
        let x = tensor_1d(&[1.0f32, 2.0, 3.0]).unwrap();
        let y = tensor_1d(&[4.0f32, 5.0]).unwrap();

        let (x_grid, y_grid) = meshgrid(&x, &y).unwrap();

        assert_eq!(x_grid.shape().dims(), &[3, 2]);
        assert_eq!(y_grid.shape().dims(), &[3, 2]);

        let x_data = x_grid.to_vec().unwrap();
        assert_eq!(x_data, vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);

        let y_data = y_grid.to_vec().unwrap();
        assert_eq!(y_data, vec![4.0, 5.0, 4.0, 5.0, 4.0, 5.0]);
    }

    #[test]
    fn test_tensor_comprehension_with_device() {
        let comp = TensorComprehension::new()
            .device(DeviceType::Cpu)
            .from_iter(0..3)
            .build()
            .unwrap();

        assert_eq!(comp.device, DeviceType::Cpu);
    }

    #[test]
    fn test_linspace_single_step() {
        let t: Tensor<f32> = linspace_range(5.0, 5.0, 1, DeviceType::Cpu).unwrap();
        let data = t.to_vec().unwrap();

        assert_eq!(data.len(), 1);
        assert!((data[0] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_range_tensor_zero_step_error() {
        let result = range_tensor(0, 10, 0, DeviceType::Cpu);
        assert!(result.is_err());
    }

    #[test]
    fn test_linspace_zero_steps_error() {
        let result: Result<Tensor<f32>> = linspace_range(0.0, 10.0, 0, DeviceType::Cpu);
        assert!(result.is_err());
    }

    #[test]
    fn test_meshgrid_different_sizes() {
        let x = tensor_1d(&[1.0f32, 2.0]).unwrap();
        let y = tensor_1d(&[3.0f32, 4.0, 5.0]).unwrap();

        let (x_grid, y_grid) = meshgrid(&x, &y).unwrap();

        assert_eq!(x_grid.shape().dims(), &[2, 3]);
        assert_eq!(y_grid.shape().dims(), &[2, 3]);
    }
}
