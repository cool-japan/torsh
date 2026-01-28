//! Tensor creation functions

use crate::{FloatElement, Tensor, TensorElement};
// ✅ SciRS2 Policy Compliant - Using scirs2_core::random instead of direct rand
use scirs2_core::random::{Random, Rng};
use torsh_core::{
    device::DeviceType,
    dtype::{Complex32, Complex64, ComplexElement},
    error::{Result, TorshError},
};

/// Create a tensor from a scalar value
pub fn tensor_scalar<T: TensorElement>(value: T) -> Result<Tensor<T>> {
    Tensor::from_data(vec![value], vec![], DeviceType::Cpu)
}

/// Create a 1D tensor from a slice
pub fn tensor_1d<T: TensorElement>(data: &[T]) -> Result<Tensor<T>> {
    Tensor::from_data(data.to_vec(), vec![data.len()], DeviceType::Cpu)
}

/// Create a 2D tensor from nested slices
pub fn tensor_2d<T: TensorElement>(data: &[&[T]]) -> Result<Tensor<T>> {
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
) -> Result<Tensor<T>> {
    let rows = M;
    let cols = N;

    let mut flat_data = Vec::with_capacity(rows * cols);
    for row in data {
        flat_data.extend_from_slice(row);
    }

    Tensor::from_data(flat_data, vec![rows, cols], DeviceType::Cpu)
}

/// Create a tensor of zeros
pub fn zeros<T: TensorElement>(shape: &[usize]) -> Result<Tensor<T>> {
    let size = shape.iter().product();
    let data = vec![T::zero(); size];

    Tensor::from_data(data, shape.to_vec(), DeviceType::Cpu)
}

/// Create a mutable tensor of zeros (uses InMemory storage for mutability)
///
/// This is useful for operations that need to write to the tensor element-by-element,
/// as it bypasses SimdOptimized storage which is immutable.
pub fn zeros_mut<T: TensorElement>(shape: &[usize]) -> Tensor<T> {
    let size = shape.iter().product();
    let data = vec![T::zero(); size];

    Tensor::from_data_fast(data, shape.to_vec(), DeviceType::Cpu)
}

/// Create a tensor of zeros on a specific device
pub fn zeros_device<T: TensorElement>(shape: &[usize], device: DeviceType) -> Result<Tensor<T>> {
    let size = shape.iter().product();
    let data = vec![T::zero(); size];

    Tensor::from_data(data, shape.to_vec(), device)
}

/// Create a tensor of ones
pub fn ones<T: TensorElement>(shape: &[usize]) -> Result<Tensor<T>> {
    let size = shape.iter().product();
    let data = vec![T::one(); size];

    Tensor::from_data(data, shape.to_vec(), DeviceType::Cpu)
}

/// Create a tensor of ones on a specific device  
pub fn ones_device<T: TensorElement>(shape: &[usize], device: DeviceType) -> Result<Tensor<T>> {
    let size = shape.iter().product();
    let data = vec![T::one(); size];

    Tensor::from_data(data, shape.to_vec(), device)
}

/// Create a tensor filled with a specific value
pub fn full<T: TensorElement>(shape: &[usize], value: T) -> Result<Tensor<T>> {
    let size = shape.iter().product();
    let data = vec![value; size];

    Tensor::from_data(data, shape.to_vec(), DeviceType::Cpu)
}

/// Create an identity matrix
pub fn eye<T: TensorElement>(n: usize) -> Result<Tensor<T>> {
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
    Tensor::from_data(values, vec![len], DeviceType::Cpu)
}

/// Create a tensor with linearly spaced values
pub fn linspace<T: FloatElement>(start: T, end: T, steps: usize) -> Result<Tensor<T>> {
    if steps == 0 {
        return zeros(&[0]);
    }

    if steps == 1 {
        return tensor_scalar(start);
    }

    let mut values = Vec::with_capacity(steps);
    let step_size = (end - start) / T::from(steps - 1).expect("numeric conversion should succeed");

    for i in 0..steps {
        let value = start + step_size * T::from(i).expect("numeric conversion should succeed");
        values.push(value);
    }

    Tensor::from_data(values, vec![steps], DeviceType::Cpu)
}

/// Create a tensor with random values from uniform distribution [0, 1)
pub fn rand<T: FloatElement>(shape: &[usize]) -> Result<Tensor<T>>
where
    T: From<f32>,
{
    let size = shape.iter().product();
    let mut rng = Random::seed(42); // Deterministic seed for reproducibility
    let values: Vec<T> = (0..size)
        .map(|_| <T as From<f32>>::from(rng.random::<f32>()))
        .collect();

    Tensor::from_data(values, shape.to_vec(), DeviceType::Cpu)
}

/// Create a tensor with random values from standard normal distribution
pub fn randn<T: FloatElement>(shape: &[usize]) -> Result<Tensor<T>> {
    let size = shape.iter().product();
    let mut rng = Random::seed(42); // Deterministic seed for reproducibility

    let values: Vec<T> = if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        // Box-Muller for f32
        (0..size)
            .map(|_| {
                let u1: f32 = rng.gen_range(0.0..1.0);
                let u2: f32 = rng.gen_range(0.0..1.0);
                let normal =
                    (-2.0_f32 * u1.ln()).sqrt() * (2.0_f32 * std::f32::consts::PI * u2).cos();
                unsafe { std::mem::transmute_copy(&normal) }
            })
            .collect()
    } else {
        // Box-Muller for f64
        (0..size)
            .map(|_| {
                let u1: f64 = rng.gen_range(0.0..1.0);
                let u2: f64 = rng.gen_range(0.0..1.0);
                let normal =
                    (-2.0_f64 * u1.ln()).sqrt() * (2.0_f64 * std::f64::consts::PI * u2).cos();
                unsafe { std::mem::transmute_copy(&normal) }
            })
            .collect()
    };

    Tensor::from_data(values, shape.to_vec(), DeviceType::Cpu)
}

/// Create a tensor with random integers
pub fn randint(low: i32, high: i32, shape: &[usize]) -> Result<Tensor<i32>> {
    let size = shape.iter().product();
    let mut rng = Random::seed(42); // Deterministic seed for reproducibility
    use scirs2_core::random::Uniform;
    let dist = Uniform::new(low, high)
        .map_err(|e| TorshError::InvalidArgument(format!("Invalid range for randint: {}", e)))?;
    let values: Vec<i32> = (0..size).map(|_| rng.sample(&dist)).collect();

    Tensor::from_data(values, shape.to_vec(), DeviceType::Cpu)
}

/// Create a tensor of zeros with the same shape as another tensor
pub fn zeros_like<T: TensorElement>(tensor: &Tensor<T>) -> Result<Tensor<T>> {
    zeros(tensor.shape().dims())
}

/// Create a tensor of ones with the same shape as another tensor
pub fn ones_like<T: TensorElement>(tensor: &Tensor<T>) -> Result<Tensor<T>> {
    ones(tensor.shape().dims())
}

/// Create a tensor filled with a value with the same shape as another tensor
pub fn full_like<T: TensorElement>(tensor: &Tensor<T>, value: T) -> Result<Tensor<T>> {
    full(tensor.shape().dims(), value)
}

/// Create a tensor with random values with the same shape as another tensor
pub fn rand_like<T: FloatElement>(tensor: &Tensor<T>) -> Result<Tensor<T>>
where
    T: From<f32>,
{
    rand(tensor.shape().dims())
}

/// Create a tensor with random normal values with the same shape as another tensor
pub fn randn_like<T: FloatElement>(tensor: &Tensor<T>) -> Result<Tensor<T>> {
    randn(tensor.shape().dims())
}

// Complex tensor creation functions

/// Create a complex tensor from real and imaginary parts
pub fn complex_from_parts<T, C>(real: &Tensor<T>, imag: &Tensor<T>) -> Result<Tensor<C>>
where
    T: FloatElement,
    C: ComplexElement<Real = T> + TensorElement,
{
    if real.shape() != imag.shape() {
        return Err(torsh_core::error::TorshError::InvalidArgument(
            "Real and imaginary parts must have the same shape".to_string(),
        ));
    }

    let real_data = real.to_vec()?;
    let imag_data = imag.to_vec()?;

    let complex_data: Vec<C> = real_data
        .iter()
        .zip(imag_data.iter())
        .map(|(&r, &i)| C::new(r, i))
        .collect();

    Tensor::from_data(complex_data, real.shape().dims().to_vec(), real.device())
}

/// Create a complex tensor of zeros
pub fn complex_zeros<T, C>(shape: &[usize]) -> Result<Tensor<C>>
where
    T: FloatElement,
    C: ComplexElement<Real = T> + TensorElement,
{
    let size = shape.iter().product();
    let data = vec![C::new(<T as TensorElement>::zero(), <T as TensorElement>::zero()); size];

    Tensor::from_data(data, shape.to_vec(), DeviceType::Cpu)
}

/// Create a complex tensor of ones
pub fn complex_ones<T, C>(shape: &[usize]) -> Result<Tensor<C>>
where
    T: FloatElement,
    C: ComplexElement<Real = T> + TensorElement,
{
    let size = shape.iter().product();
    let data = vec![C::new(<T as TensorElement>::one(), <T as TensorElement>::zero()); size];

    Tensor::from_data(data, shape.to_vec(), DeviceType::Cpu)
}

/// Create a complex tensor filled with a specific value
pub fn complex_full<C: TensorElement>(shape: &[usize], value: C) -> Result<Tensor<C>> {
    let size = shape.iter().product();
    let data = vec![value; size];

    Tensor::from_data(data, shape.to_vec(), DeviceType::Cpu)
}

/// Create a complex tensor with random values (uniform distribution)
/// Real and imaginary parts are independently sampled from [0, 1)
pub fn complex_rand<T, C>(shape: &[usize]) -> Result<Tensor<C>>
where
    T: FloatElement + From<f32>,
    C: ComplexElement<Real = T> + TensorElement,
{
    let size = shape.iter().product();
    let mut rng = Random::seed(42); // Deterministic seed for reproducibility
    let values: Vec<C> = (0..size)
        .map(|_| {
            C::new(
                <T as From<f32>>::from(rng.random::<f32>()),
                <T as From<f32>>::from(rng.random::<f32>()),
            )
        })
        .collect();

    Tensor::from_data(values, shape.to_vec(), DeviceType::Cpu)
}

/// Create a complex tensor with random values from standard normal distribution
/// Real and imaginary parts are independently sampled from N(0, 1)
pub fn complex_randn<T, C>(shape: &[usize]) -> Result<Tensor<C>>
where
    T: FloatElement,
    C: ComplexElement<Real = T> + TensorElement,
{
    let size = shape.iter().product();
    let mut rng = Random::seed(42); // Deterministic seed for reproducibility

    let values: Vec<C> = if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        // Box-Muller for f32
        (0..size)
            .map(|_| {
                let u1: f32 = rng.gen_range(0.0..1.0);
                let u2: f32 = rng.gen_range(0.0..1.0);
                let normal1 =
                    (-2.0_f32 * u1.ln()).sqrt() * (2.0_f32 * std::f32::consts::PI * u2).cos();
                let normal2 =
                    (-2.0_f32 * u1.ln()).sqrt() * (2.0_f32 * std::f32::consts::PI * u2).sin();
                let real: T = unsafe { std::mem::transmute_copy(&normal1) };
                let imag: T = unsafe { std::mem::transmute_copy(&normal2) };
                C::new(real, imag)
            })
            .collect()
    } else {
        // Box-Muller for f64
        (0..size)
            .map(|_| {
                let u1: f64 = rng.gen_range(0.0..1.0);
                let u2: f64 = rng.gen_range(0.0..1.0);
                let normal1 =
                    (-2.0_f64 * u1.ln()).sqrt() * (2.0_f64 * std::f64::consts::PI * u2).cos();
                let normal2 =
                    (-2.0_f64 * u1.ln()).sqrt() * (2.0_f64 * std::f64::consts::PI * u2).sin();
                let real: T = unsafe { std::mem::transmute_copy(&normal1) };
                let imag: T = unsafe { std::mem::transmute_copy(&normal2) };
                C::new(real, imag)
            })
            .collect()
    };

    Tensor::from_data(values, shape.to_vec(), DeviceType::Cpu)
}

/// Create complex tensor with the same shape as another tensor, filled with zeros
pub fn complex_zeros_like<T, C>(tensor: &Tensor<T>) -> Result<Tensor<C>>
where
    T: FloatElement,
    C: ComplexElement<Real = T> + TensorElement,
{
    complex_zeros(tensor.shape().dims())
}

/// Create complex tensor with the same shape as another tensor, filled with ones
pub fn complex_ones_like<T, C>(tensor: &Tensor<T>) -> Result<Tensor<C>>
where
    T: FloatElement,
    C: ComplexElement<Real = T> + TensorElement,
{
    complex_ones(tensor.shape().dims())
}

/// Create complex tensor with random values with the same shape as another tensor
pub fn complex_rand_like<T, C>(tensor: &Tensor<T>) -> Result<Tensor<C>>
where
    T: FloatElement + From<f32>,
    C: ComplexElement<Real = T> + TensorElement,
{
    complex_rand(tensor.shape().dims())
}

/// Create complex tensor with random normal values with the same shape as another tensor
pub fn complex_randn_like<T, C>(tensor: &Tensor<T>) -> Result<Tensor<C>>
where
    T: FloatElement,
    C: ComplexElement<Real = T> + TensorElement,
{
    complex_randn(tensor.shape().dims())
}

// Convenience functions for specific complex types

/// Create a Complex32 tensor of zeros
pub fn complex32_zeros(shape: &[usize]) -> Result<Tensor<Complex32>> {
    complex_zeros::<f32, Complex32>(shape)
}

/// Create a Complex32 tensor of ones
pub fn complex32_ones(shape: &[usize]) -> Result<Tensor<Complex32>> {
    complex_ones::<f32, Complex32>(shape)
}

/// Create a Complex32 tensor with random values
pub fn complex32_rand(shape: &[usize]) -> Result<Tensor<Complex32>> {
    complex_rand::<f32, Complex32>(shape)
}

/// Create a Complex32 tensor with random normal values
pub fn complex32_randn(shape: &[usize]) -> Result<Tensor<Complex32>> {
    complex_randn::<f32, Complex32>(shape)
}

/// Create a Complex64 tensor of zeros
pub fn complex64_zeros(shape: &[usize]) -> Result<Tensor<Complex64>> {
    complex_zeros::<f64, Complex64>(shape)
}

/// Create a Complex64 tensor of ones
pub fn complex64_ones(shape: &[usize]) -> Result<Tensor<Complex64>> {
    complex_ones::<f64, Complex64>(shape)
}

/// Create a Complex64 tensor with random values
pub fn complex64_rand(shape: &[usize]) -> Result<Tensor<Complex64>> {
    complex_rand::<f64, Complex64>(shape)
}

/// Create a Complex64 tensor with random normal values
pub fn complex64_randn(shape: &[usize]) -> Result<Tensor<Complex64>> {
    complex_randn::<f64, Complex64>(shape)
}

#[cfg(test)]
mod complex_tests {
    use super::*;
    use crate::tensor;

    #[test]
    fn test_complex_zeros() {
        let tensor = complex32_zeros(&[2, 3]).unwrap();
        assert_eq!(tensor.shape().dims(), &[2, 3]);
        assert_eq!(tensor.numel(), 6);

        let data = tensor.to_vec().unwrap();
        for &val in &data {
            assert_eq!(val.re, 0.0);
            assert_eq!(val.im, 0.0);
        }
    }

    #[test]
    fn test_complex_ones() {
        let tensor = complex32_ones(&[2, 2]).unwrap();
        assert_eq!(tensor.shape().dims(), &[2, 2]);

        let data = tensor.to_vec().unwrap();
        for &val in &data {
            assert_eq!(val.re, 1.0);
            assert_eq!(val.im, 0.0);
        }
    }

    #[test]
    fn test_complex_from_parts() {
        let real = tensor![1.0f32, 2.0, 3.0].unwrap();
        let imag = tensor![4.0f32, 5.0, 6.0].unwrap();

        let complex_tensor: Tensor<Complex32> = complex_from_parts(&real, &imag).unwrap();
        assert_eq!(complex_tensor.shape().dims(), &[3]);

        let data = complex_tensor.to_vec().unwrap();
        assert_eq!(data[0].re, 1.0);
        assert_eq!(data[0].im, 4.0);
        assert_eq!(data[1].re, 2.0);
        assert_eq!(data[1].im, 5.0);
        assert_eq!(data[2].re, 3.0);
        assert_eq!(data[2].im, 6.0);
    }

    #[test]
    fn test_complex_from_parts_shape_mismatch() {
        let real = tensor![1.0f32, 2.0].unwrap();
        let imag = tensor![4.0f32, 5.0, 6.0].unwrap();

        let result: Result<Tensor<Complex32>> = complex_from_parts(&real, &imag);
        assert!(result.is_err());
    }

    #[test]
    fn test_complex_rand() {
        let tensor = complex32_rand(&[10]).unwrap();
        assert_eq!(tensor.shape().dims(), &[10]);

        let data = tensor.to_vec().unwrap();
        // Check that we have some variation in real and imaginary parts
        let all_same_real = data.iter().all(|&c| c.re == data[0].re);
        let all_same_imag = data.iter().all(|&c| c.im == data[0].im);

        // With random data, this should be extremely unlikely
        assert!(!all_same_real || !all_same_imag);
    }

    #[test]
    fn test_complex_randn() {
        let tensor = complex64_randn(&[5, 5]).unwrap();
        assert_eq!(tensor.shape().dims(), &[5, 5]);
        assert_eq!(tensor.numel(), 25);

        // Test that values are reasonably distributed (not all zeros)
        let data = tensor.to_vec().unwrap();
        let has_nonzero_real = data.iter().any(|&c| c.re.abs() > 0.01);
        let has_nonzero_imag = data.iter().any(|&c| c.im.abs() > 0.01);

        assert!(has_nonzero_real);
        assert!(has_nonzero_imag);
    }

    #[test]
    fn test_complex_like_functions() {
        let base = tensor![1.0f32, 2.0, 3.0].unwrap();

        let zeros: Tensor<Complex32> = complex_zeros_like(&base).unwrap();
        assert_eq!(zeros.shape().dims(), base.shape().dims());

        let ones: Tensor<Complex32> = complex_ones_like(&base).unwrap();
        assert_eq!(ones.shape().dims(), base.shape().dims());

        let random: Tensor<Complex32> = complex_rand_like(&base).unwrap();
        assert_eq!(random.shape().dims(), base.shape().dims());

        let normal: Tensor<Complex32> = complex_randn_like(&base).unwrap();
        assert_eq!(normal.shape().dims(), base.shape().dims());
    }
}

/// Create a tensor from a vector and shape
pub fn from_vec<T: TensorElement>(
    data: Vec<T>,
    shape: &[usize],
    device: DeviceType,
) -> Result<Tensor<T>> {
    Tensor::from_data(data, shape.to_vec(), device)
}
