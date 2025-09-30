//! Metal buffer management and tensor storage

use metal::Buffer;
// use torsh_backends::BackendStorage; // TODO: This trait doesn't exist in current API
use torsh_core::{DType, Shape, TensorElement};

use crate::{
    metal::device::MetalDevice,
    metal::error::{MetalError, Result},
};

/// Metal buffer for tensor storage
#[derive(Clone)]
pub struct MetalBuffer {
    /// The underlying Metal buffer
    buffer: Buffer,
    /// Shape of the tensor
    shape: Shape,
    /// Data type
    dtype: DType,
    /// Number of elements
    numel: usize,
    /// Device reference
    device: MetalDevice,
}

impl MetalBuffer {
    /// Create a new Metal buffer from a slice
    pub fn from_slice<T: TensorElement>(
        data: &[T],
        shape: &Shape,
        device: &MetalDevice,
    ) -> Result<Self> {
        let dtype = T::dtype();
        let numel = shape.numel();

        if data.len() != numel {
            return Err(MetalError::ShapeMismatch {
                expected: vec![numel],
                got: vec![data.len()],
            });
        }

        let byte_size = numel * dtype.size();
        let options = device.resource_options();

        let buffer = device.device().new_buffer_with_data(
            data.as_ptr() as *const _,
            byte_size as u64,
            options,
        );

        Ok(Self {
            buffer,
            shape: shape.clone(),
            dtype,
            numel,
            device: device.clone(),
        })
    }

    /// Create a new Metal buffer filled with zeros
    pub fn zeros(shape: &Shape, dtype: &DType, device: &MetalDevice) -> Result<Self> {
        let numel = shape.numel();
        let byte_size = numel * dtype.size();

        let buffer = device
            .device()
            .new_buffer(byte_size as u64, device.resource_options());

        // Initialize to zeros
        let buffer_clone = buffer.clone();
        unsafe {
            let ptr = buffer_clone.contents() as *mut u8;
            std::ptr::write_bytes(ptr, 0, byte_size);
        }

        Ok(Self {
            buffer,
            shape: shape.clone(),
            dtype: *dtype,
            numel,
            device: device.clone(),
        })
    }

    /// Create a new Metal buffer filled with ones
    pub fn ones(shape: &Shape, dtype: &DType, device: &MetalDevice) -> Result<Self> {
        let buffer = Self::zeros(shape, dtype, device)?;
        buffer.fill_with_scalar(1.0)?;
        Ok(buffer)
    }

    /// Create a new Metal buffer with random values
    pub fn rand(shape: &Shape, dtype: &DType, device: &MetalDevice) -> Result<Self> {
        use scirs2_core::random::{Random, Rng};
        let mut rng = Random::seed(42);

        let numel = shape.numel();
        let buffer = Self::zeros(shape, dtype, device)?;

        unsafe {
            match dtype {
                DType::F32 => {
                    let ptr = buffer.buffer.contents() as *mut f32;
                    for i in 0..numel {
                        *ptr.add(i) = rng.gen_range(0.0..1.0);
                    }
                }
                DType::F64 => {
                    let ptr = buffer.buffer.contents() as *mut f64;
                    for i in 0..numel {
                        *ptr.add(i) = rng.gen_range(0.0..1.0);
                    }
                }
                _ => return Err(MetalError::UnsupportedOperation {
                    op: "fill_with_scalar".to_string(),
                    dtype: format!("{:?}", dtype)
                }),
            }
        }

        Ok(buffer)
    }

    /// Create a new Metal buffer with random normal values
    pub fn randn(shape: &Shape, dtype: &DType, device: &MetalDevice) -> Result<Self> {
        use scirs2_core::random::{Random, Rng};
        let mut rng = Random::seed(42);

        // Box-Muller transform for normal distribution
        let mut normal_gen = || -> f64 {
            let u1: f64 = rng.gen_range(0.0..1.0);
            let u2: f64 = rng.gen_range(0.0..1.0);
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        };

        let numel = shape.numel();
        let buffer = Self::zeros(shape, dtype, device)?;

        unsafe {
            match dtype {
                DType::F32 => {
                    let ptr = buffer.buffer.contents() as *mut f32;
                    for i in 0..numel {
                        *ptr.add(i) = normal_gen() as f32;
                    }
                }
                DType::F64 => {
                    let ptr = buffer.buffer.contents() as *mut f64;
                    for i in 0..numel {
                        *ptr.add(i) = normal_gen();
                    }
                }
                _ => return Err(MetalError::UnsupportedOperation {
                    op: "fill_with_scalar".to_string(),
                    dtype: format!("{:?}", dtype)
                }),
            }
        }

        Ok(buffer)
    }

    /// Fill buffer with a scalar value
    pub fn fill_with_scalar(&self, value: f64) -> Result<()> {
        unsafe {
            match self.dtype {
                DType::F32 => {
                    let ptr = self.buffer.contents() as *mut f32;
                    let val = value as f32;
                    for i in 0..self.numel {
                        *ptr.add(i) = val;
                    }
                }
                DType::F64 => {
                    let ptr = self.buffer.contents() as *mut f64;
                    for i in 0..self.numel {
                        *ptr.add(i) = value;
                    }
                }
                DType::I32 => {
                    let ptr = self.buffer.contents() as *mut i32;
                    let val = value as i32;
                    for i in 0..self.numel {
                        *ptr.add(i) = val;
                    }
                }
                DType::I64 => {
                    let ptr = self.buffer.contents() as *mut i64;
                    let val = value as i64;
                    for i in 0..self.numel {
                        *ptr.add(i) = val;
                    }
                }
                _ => return Err(MetalError::UnsupportedOperation {
                    op: "get_data".to_string(),
                    dtype: format!("{:?}", self.dtype)
                }),
            }
        }
        Ok(())
    }

    /// Get the underlying Metal buffer
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// Get the device
    pub fn device(&self) -> &MetalDevice {
        &self.device
    }

    /// Get the shape
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Get the dtype
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Copy data from the buffer to a vector
    pub fn to_vec<T: TensorElement>(&self) -> Result<Vec<T>> {
        if T::dtype() != self.dtype {
            return Err(MetalError::ConversionError(format!(
                "Type mismatch: expected {:?}, got {:?}",
                T::dtype(),
                self.dtype
            )));
        }

        let mut result = Vec::with_capacity(self.numel);
        unsafe {
            let ptr = self.buffer.contents() as *const T;
            for i in 0..self.numel {
                result.push(ptr.add(i).read());
            }
        }
        Ok(result)
    }

    /// Create a view with a different shape (must have same number of elements)
    pub fn view(&self, new_shape: &Shape) -> Result<Self> {
        if new_shape.numel() != self.numel {
            return Err(MetalError::ShapeMismatch {
                expected: vec![self.numel],
                got: vec![new_shape.numel()],
            });
        }

        Ok(Self {
            buffer: self.buffer.clone(),
            shape: new_shape.clone(),
            dtype: self.dtype,
            numel: self.numel,
            device: self.device.clone(),
        })
    }

    /// Create a new uninitialized buffer with the given size
    pub fn new(size: usize, device: &MetalDevice) -> Result<Self> {
        let buffer = device
            .device()
            .new_buffer(size as u64, device.resource_options());

        Ok(Self {
            buffer,
            shape: Shape::from(vec![size]),
            dtype: DType::U8,
            numel: size,
            device: device.clone(),
        })
    }

    /// Create a buffer from raw bytes
    pub fn from_data(data: &[u8], device: &MetalDevice) -> Result<Self> {
        let size = data.len();
        let buffer = device.device().new_buffer_with_data(
            data.as_ptr() as *const _,
            size as u64,
            device.resource_options(),
        );

        Ok(Self {
            buffer,
            shape: Shape::from(vec![size]),
            dtype: DType::U8,
            numel: size,
            device: device.clone(),
        })
    }

    /// Get the raw pointer to the buffer
    pub fn as_ptr(&self) -> *const u8 {
        self.buffer.contents() as *const u8
    }

    /// Get the size in bytes
    pub fn size_bytes(&self) -> usize {
        self.numel * self.dtype.size()
    }
}

// TODO: BackendStorage trait doesn't exist in current API
// impl BackendStorage for MetalBuffer {
//     fn shape(&self) -> &Shape {
//         &self.shape
//     }

//     fn dtype(&self) -> DType {
//         self.dtype
//     }

//     fn device(&self) -> torsh_core::Device {
//         torsh_core::Device::Metal(0)
//     }

//     fn to_vec<T: TensorElement>(&self) -> anyhow::Result<Vec<T>> {
//         self.to_vec::<T>().map_err(Into::into)
//     }

//     fn from_vec<T: TensorElement>(data: Vec<T>, shape: &Shape) -> anyhow::Result<Self> {
//         // This is a bit of a hack since we don't have the device here
//         // In practice, this should be called through the backend
//         Err(anyhow::anyhow!("Use MetalBuffer::from_slice instead"))
//     }

//     fn clone_storage(&self) -> Box<dyn BackendStorage> {
//         Box::new(self.clone())
//     }
// }

// SciRS2 dependencies for random number generation
use scirs2_core::random;

impl std::fmt::Debug for MetalBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalBuffer")
            .field("shape", &self.shape)
            .field("dtype", &self.dtype)
            .field("numel", &self.numel)
            .field("device", &self.device.info().name)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_creation() {
        if let Ok(device) = MetalDevice::new() {
            let shape = Shape::from(vec![2, 3]);
            let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];

            let buffer = MetalBuffer::from_slice(&data, &shape, &device);
            assert!(buffer.is_ok());

            let buffer = buffer.unwrap();
            assert_eq!(buffer.shape(), &shape);
            assert_eq!(buffer.dtype(), DType::F32);
        }
    }

    #[test]
    fn test_buffer_zeros() {
        if let Ok(device) = MetalDevice::new() {
            let shape = Shape::from(vec![3, 4]);
            let buffer = MetalBuffer::zeros(&shape, &DType::F32, &device);
            assert!(buffer.is_ok());

            let buffer = buffer.unwrap();
            let data = buffer.to_vec::<f32>().unwrap();
            assert_eq!(data, vec![0.0f32; 12]);
        }
    }
}
