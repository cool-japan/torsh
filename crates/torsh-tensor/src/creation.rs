//! Tensor creation functions

use crate::{FloatElement, Tensor, TensorElement};
use rand::{
    distributions::{Distribution, Standard, Uniform},
    Rng,
};
use rand_distr::StandardNormal;
use torsh_core::{device::DeviceType, error::Result};

/// Create a tensor from a scalar value
pub fn tensor_scalar<T: TensorElement>(value: T) -> Tensor<T> {
    Tensor::from_data(vec![value], vec![], DeviceType::Cpu)
}

/// Create a 1D tensor from a slice
pub fn tensor_1d<T: TensorElement>(data: &[T]) -> Tensor<T> {
    Tensor::from_data(data.to_vec(), vec![data.len()], DeviceType::Cpu)
}

/// Create a 2D tensor from nested slices
pub fn tensor_2d<T: TensorElement>(data: &[&[T]]) -> Tensor<T> {
    let rows = data.len();
    let cols = if rows > 0 { data[0].len() } else { 0 };

    let mut flat_data = Vec::with_capacity(rows * cols);
    for row in data {
        flat_data.extend_from_slice(row);
    }

    Tensor::from_data(flat_data, vec![rows, cols], DeviceType::Cpu)
}

/// Create a 2D tensor from nested arrays (for macro use)
pub fn tensor_2d_arrays<T: TensorElement, const M: usize, const N: usize>(
    data: &[[T; N]; M],
) -> Tensor<T> {
    let rows = M;
    let cols = N;

    let mut flat_data = Vec::with_capacity(rows * cols);
    for row in data {
        flat_data.extend_from_slice(row);
    }

    Tensor::from_data(flat_data, vec![rows, cols], DeviceType::Cpu)
}

/// Create a tensor of zeros
pub fn zeros<T: TensorElement>(shape: &[usize]) -> Tensor<T> {
    let size = shape.iter().product();
    let data = vec![T::zero(); size];

    Tensor::from_data(data, shape.to_vec(), DeviceType::Cpu)
}

/// Create a tensor of ones
pub fn ones<T: TensorElement>(shape: &[usize]) -> Tensor<T> {
    let size = shape.iter().product();
    let data = vec![T::one(); size];

    Tensor::from_data(data, shape.to_vec(), DeviceType::Cpu)
}

/// Create a tensor filled with a specific value
pub fn full<T: TensorElement>(shape: &[usize], value: T) -> Tensor<T> {
    let size = shape.iter().product();
    let data = vec![value; size];

    Tensor::from_data(data, shape.to_vec(), DeviceType::Cpu)
}

/// Create an identity matrix
pub fn eye<T: TensorElement>(n: usize) -> Tensor<T> {
    let mut data = vec![T::zero(); n * n];
    for i in 0..n {
        data[i * n + i] = T::one();
    }

    Tensor::from_data(data, vec![n, n], DeviceType::Cpu)
}

/// Create a tensor with values from a range
pub fn arange<T: TensorElement + std::cmp::PartialOrd + std::ops::Add<Output = T> + Copy>(
    start: T,
    end: T,
    step: T,
) -> Result<Tensor<T>> {
    let mut values = Vec::new();
    let mut current = start;

    while current < end {
        values.push(current);
        current = current + step;
    }

    let len = values.len();
    Ok(Tensor::from_data(values, vec![len], DeviceType::Cpu))
}

/// Create a tensor with linearly spaced values
pub fn linspace<T: FloatElement>(start: T, end: T, steps: usize) -> Tensor<T> {
    if steps == 0 {
        return zeros(&[0]);
    }

    if steps == 1 {
        return tensor_scalar(start);
    }

    let mut values = Vec::with_capacity(steps);
    let step_size = (end - start) / T::from(steps - 1).unwrap();

    for i in 0..steps {
        let value = start + step_size * T::from(i).unwrap();
        values.push(value);
    }

    Tensor::from_data(values, vec![steps], DeviceType::Cpu)
}

/// Create a tensor with random values from uniform distribution [0, 1)
pub fn rand<T: FloatElement>(shape: &[usize]) -> Tensor<T>
where
    Standard: Distribution<T>,
{
    let size = shape.iter().product();
    let mut rng = rand::thread_rng();
    let values: Vec<T> = (0..size).map(|_| rng.gen()).collect();

    Tensor::from_data(values, shape.to_vec(), DeviceType::Cpu)
}

/// Create a tensor with random values from standard normal distribution
pub fn randn<T: FloatElement>(shape: &[usize]) -> Tensor<T>
where
    StandardNormal: Distribution<T>,
{
    let size = shape.iter().product();
    let mut rng = rand::thread_rng();
    let normal = StandardNormal;
    let values: Vec<T> = (0..size).map(|_| normal.sample(&mut rng)).collect();

    Tensor::from_data(values, shape.to_vec(), DeviceType::Cpu)
}

/// Create a tensor with random integers
pub fn randint(low: i32, high: i32, shape: &[usize]) -> Tensor<i32> {
    let size = shape.iter().product();
    let mut rng = rand::thread_rng();
    let uniform = Uniform::new(low, high);
    let values: Vec<i32> = (0..size).map(|_| uniform.sample(&mut rng)).collect();

    Tensor::from_data(values, shape.to_vec(), DeviceType::Cpu)
}

/// Create a tensor of zeros with the same shape as another tensor
pub fn zeros_like<T: TensorElement>(tensor: &Tensor<T>) -> Tensor<T> {
    zeros(tensor.shape().dims())
}

/// Create a tensor of ones with the same shape as another tensor
pub fn ones_like<T: TensorElement>(tensor: &Tensor<T>) -> Tensor<T> {
    ones(tensor.shape().dims())
}

/// Create a tensor with random values with the same shape as another tensor
pub fn rand_like<T: FloatElement>(tensor: &Tensor<T>) -> Tensor<T>
where
    Standard: Distribution<T>,
{
    rand(tensor.shape().dims())
}

/// Create a tensor with random normal values with the same shape as another tensor
pub fn randn_like<T: FloatElement>(tensor: &Tensor<T>) -> Tensor<T>
where
    StandardNormal: Distribution<T>,
{
    randn(tensor.shape().dims())
}
