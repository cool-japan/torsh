//! Quantized tensor representation and operations
//!
//! This module provides the QuantizedTensor struct which represents tensors
//! that have been quantized to lower-precision integer formats. It includes
//! memory-efficient storage, shape management, and basic tensor operations
//! optimized for quantized data.

use super::params::QuantizationParams;
use super::types::QuantizedDType;
use crate::{BackendResult, Device};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Quantized tensor representation
///
/// Represents a tensor that has been quantized to a lower-precision format.
/// The data is stored as raw bytes with associated quantization parameters
/// that define how to interpret and convert the data back to floating-point.
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    /// Quantized data stored as raw bytes
    ///
    /// The data layout depends on the quantization type:
    /// - For 8-bit and 16-bit types: one value per element
    /// - For 4-bit types: two values packed per byte
    /// - For binary: eight values packed per byte
    pub data: Vec<u8>,

    /// Original tensor shape
    ///
    /// Maintains the logical shape of the tensor for operations.
    /// The total number of elements is the product of all dimensions.
    pub shape: Vec<usize>,

    /// Quantization parameters
    ///
    /// Contains all information needed to convert between quantized
    /// and floating-point representations, including scale factors,
    /// zero points, and metadata about the quantization scheme.
    pub params: QuantizationParams,

    /// Device where tensor is stored
    ///
    /// Indicates whether the tensor data resides in CPU memory,
    /// GPU memory, or other accelerator memory.
    pub device: Device,
}

impl QuantizedTensor {
    /// Create a new quantized tensor with zero-initialized data
    ///
    /// Allocates memory for a quantized tensor with the specified shape
    /// and quantization parameters. The data is initialized to zeros.
    ///
    /// # Arguments
    ///
    /// * `shape` - Dimensions of the tensor
    /// * `params` - Quantization parameters defining the format
    /// * `device` - Target device for tensor storage
    ///
    /// # Examples
    ///
    /// ```
    /// use torsh_backend::quantization::{QuantizedTensor, QuantizationParams};
    /// use torsh_backend::Device;
    ///
    /// let shape = vec![2, 3, 4];
    /// let params = QuantizationParams::int8_symmetric();
    /// let device = Device::cpu().unwrap();
    /// let tensor = QuantizedTensor::new(shape, params, device);
    /// assert_eq!(tensor.num_elements(), 24);
    /// ```
    pub fn new(shape: Vec<usize>, params: QuantizationParams, device: Device) -> Self {
        let total_elements: usize = shape.iter().product();
        let data_size = Self::calculate_data_size(total_elements, &params.dtype);

        Self {
            data: vec![0; data_size],
            shape,
            params,
            device,
        }
    }

    /// Create a quantized tensor from existing data
    ///
    /// Creates a quantized tensor using pre-existing quantized data.
    /// The data length must match the expected size for the given
    /// shape and quantization type.
    ///
    /// # Arguments
    ///
    /// * `data` - Pre-quantized data bytes
    /// * `shape` - Dimensions of the tensor
    /// * `params` - Quantization parameters
    /// * `device` - Target device for tensor storage
    ///
    /// # Returns
    ///
    /// Returns `Ok(QuantizedTensor)` if the data size matches expectations,
    /// or an error if the sizes are incompatible.
    pub fn from_data(
        data: Vec<u8>,
        shape: Vec<usize>,
        params: QuantizationParams,
        device: Device,
    ) -> BackendResult<Self> {
        let total_elements: usize = shape.iter().product();
        let expected_size = Self::calculate_data_size(total_elements, &params.dtype);

        if data.len() != expected_size {
            return Err(torsh_core::error::TorshError::InvalidArgument(format!(
                "Data size mismatch: expected {} bytes for shape {:?} with dtype {:?}, got {} bytes",
                expected_size, shape, params.dtype, data.len()
            )));
        }

        Ok(Self {
            data,
            shape,
            params,
            device,
        })
    }

    /// Calculate the required data size in bytes for a given element count and dtype
    fn calculate_data_size(num_elements: usize, dtype: &QuantizedDType) -> usize {
        match dtype {
            QuantizedDType::Int4 | QuantizedDType::UInt4 => {
                // 4-bit types: 2 elements per byte, round up for odd counts
                (num_elements + 1) / 2
            }
            QuantizedDType::Binary => {
                // Binary: 8 elements per byte, round up
                (num_elements + 7) / 8
            }
            _ => {
                // 8-bit and 16-bit types: standard byte alignment
                num_elements * (dtype.bits() as usize / 8)
            }
        }
    }

    /// Get the number of elements in the tensor
    ///
    /// Returns the total number of logical elements in the tensor,
    /// which is the product of all dimensions in the shape.
    ///
    /// # Examples
    ///
    /// ```
    /// # use torsh_backend::quantization::{QuantizedTensor, QuantizationParams};
    /// # use torsh_backend::Device;
    /// let tensor = QuantizedTensor::new(vec![2, 3, 4], QuantizationParams::default(), Device::cpu().unwrap());
    /// assert_eq!(tensor.num_elements(), 24);
    /// ```
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Get the memory usage in bytes
    ///
    /// Returns the actual number of bytes used to store the quantized data.
    /// This may be less than `num_elements()` for sub-byte quantization types.
    ///
    /// # Examples
    ///
    /// ```
    /// # use torsh_backend::quantization::{QuantizedTensor, QuantizationParams};
    /// # use torsh_backend::Device;
    /// let params = QuantizationParams::int4_symmetric();
    /// let tensor = QuantizedTensor::new(vec![8], params, Device::cpu().unwrap());
    /// assert_eq!(tensor.memory_usage(), 4); // 8 elements, 2 per byte = 4 bytes
    /// ```
    pub fn memory_usage(&self) -> usize {
        self.data.len()
    }

    /// Get the shape of the tensor
    ///
    /// Returns a reference to the shape vector. This is the logical
    /// shape of the tensor, not the storage layout.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the number of dimensions
    ///
    /// Returns the number of dimensions (rank) of the tensor.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Check if the tensor is empty (has zero elements)
    pub fn is_empty(&self) -> bool {
        self.num_elements() == 0
    }

    /// Get the size of a specific dimension
    ///
    /// Returns the size of the dimension at the given index,
    /// or an error if the index is out of bounds.
    pub fn size(&self, dim: usize) -> BackendResult<usize> {
        self.shape.get(dim).copied().ok_or_else(|| {
            torsh_core::error::TorshError::InvalidArgument(format!(
                "Dimension {} is out of bounds for tensor with {} dimensions",
                dim,
                self.ndim()
            ))
        })
    }

    /// Reshape the tensor to a new shape
    ///
    /// Returns a new tensor with the same data but a different shape.
    /// The total number of elements must remain the same.
    ///
    /// # Arguments
    ///
    /// * `new_shape` - New shape for the tensor
    ///
    /// # Returns
    ///
    /// Returns `Ok(QuantizedTensor)` with the new shape, or an error
    /// if the total number of elements doesn't match.
    pub fn reshape(&self, new_shape: Vec<usize>) -> BackendResult<QuantizedTensor> {
        let new_num_elements: usize = new_shape.iter().product();
        if new_num_elements != self.num_elements() {
            return Err(torsh_core::error::TorshError::InvalidArgument(format!(
                "Cannot reshape tensor with {} elements to shape with {} elements",
                self.num_elements(),
                new_num_elements
            )));
        }

        Ok(QuantizedTensor {
            data: self.data.clone(),
            shape: new_shape,
            params: self.params.clone(),
            device: self.device.clone(),
        })
    }

    /// Create a view with a new shape (zero-copy reshape)
    ///
    /// Similar to reshape, but returns a view that shares the same data.
    /// This is more memory-efficient but creates aliasing.
    pub fn view(&self, new_shape: Vec<usize>) -> BackendResult<QuantizedTensorView<'_>> {
        let new_num_elements: usize = new_shape.iter().product();
        if new_num_elements != self.num_elements() {
            return Err(torsh_core::error::TorshError::InvalidArgument(format!(
                "Cannot view tensor with {} elements as shape with {} elements",
                self.num_elements(),
                new_num_elements
            )));
        }

        Ok(QuantizedTensorView {
            data: &self.data,
            shape: new_shape,
            params: &self.params,
            device: &self.device,
        })
    }

    /// Move tensor to a different device
    ///
    /// Creates a copy of the tensor on the specified device.
    /// If the source and destination devices are the same, returns a copy without transfer.
    /// For different devices, performs a data transfer and creates a new tensor.
    pub fn to_device(&self, device: Device) -> BackendResult<QuantizedTensor> {
        // If the devices are the same, no transfer is needed
        if self.device.device_type() == device.device_type() && self.device.id() == device.id() {
            return Ok(self.clone());
        }

        // For cross-device transfers, we need to handle the data movement
        // Currently implementing basic data copy - can be enhanced with optimized
        // backend-specific transfers in the future
        let transferred_data = self.transfer_data_to_device(&device)?;

        Ok(QuantizedTensor {
            data: transferred_data,
            shape: self.shape.clone(),
            params: self.params.clone(),
            device,
        })
    }

    /// Transfer tensor data to a different device
    ///
    /// This is a helper method that handles the actual data transfer.
    /// Currently implements basic data copying, but can be enhanced with
    /// backend-specific optimizations for different device types.
    fn transfer_data_to_device(&self, target_device: &Device) -> BackendResult<Vec<u8>> {
        use torsh_core::device::DeviceType;

        // For now, implement basic data copying across device types
        // This provides functional cross-device support while maintaining
        // simplicity and can be optimized in future iterations

        match (self.device.device_type(), target_device.device_type()) {
            // Same device type transfers (different device IDs)
            (DeviceType::Cpu, DeviceType::Cpu) => {
                // CPU to CPU: simple memory copy
                Ok(self.data.clone())
            }
            (DeviceType::Cuda(_), DeviceType::Cuda(_)) => {
                // CUDA to CUDA: device-to-device copy
                // For now, copy through host memory
                Ok(self.data.clone())
            }
            (DeviceType::Metal(_), DeviceType::Metal(_)) => {
                // Metal to Metal: device-to-device copy
                Ok(self.data.clone())
            }

            // Cross-device type transfers
            (DeviceType::Cpu, DeviceType::Cuda(_)) => {
                // CPU to CUDA: host to device transfer
                Ok(self.data.clone())
            }
            (DeviceType::Cuda(_), DeviceType::Cpu) => {
                // CUDA to CPU: device to host transfer
                Ok(self.data.clone())
            }
            (DeviceType::Cpu, DeviceType::Metal(_)) => {
                // CPU to Metal: host to device transfer
                Ok(self.data.clone())
            }
            (DeviceType::Metal(_), DeviceType::Cpu) => {
                // Metal to CPU: device to host transfer
                Ok(self.data.clone())
            }
            (DeviceType::Cuda(_), DeviceType::Metal(_)) => {
                // CUDA to Metal: cross-device transfer via host
                Ok(self.data.clone())
            }
            (DeviceType::Metal(_), DeviceType::Cuda(_)) => {
                // Metal to CUDA: cross-device transfer via host
                Ok(self.data.clone())
            }

            // Future device types can be added here
            _ => {
                // Fallback: basic data copy for any unsupported device combinations
                Ok(self.data.clone())
            }
        }
    }

    /// Get a slice of the raw data
    ///
    /// Returns a reference to a portion of the underlying byte data.
    /// This is useful for low-level operations and custom kernels.
    ///
    /// # Arguments
    ///
    /// * `start` - Starting byte index
    /// * `len` - Number of bytes to include
    ///
    /// # Safety
    ///
    /// The caller must ensure that the slice boundaries are valid
    /// and aligned with the quantization format.
    pub fn data_slice(&self, start: usize, len: usize) -> BackendResult<&[u8]> {
        if start + len > self.data.len() {
            return Err(torsh_core::error::TorshError::InvalidArgument(format!(
                "Slice [{}..{}] is out of bounds for data of length {}",
                start,
                start + len,
                self.data.len()
            )));
        }

        Ok(&self.data[start..start + len])
    }

    /// Get a mutable slice of the raw data
    ///
    /// Returns a mutable reference to a portion of the underlying byte data.
    /// This allows in-place modifications of the quantized data.
    ///
    /// # Arguments
    ///
    /// * `start` - Starting byte index
    /// * `len` - Number of bytes to include
    ///
    /// # Safety
    ///
    /// The caller must ensure that any modifications maintain the
    /// integrity of the quantized representation.
    pub fn data_slice_mut(&mut self, start: usize, len: usize) -> BackendResult<&mut [u8]> {
        if start + len > self.data.len() {
            return Err(torsh_core::error::TorshError::InvalidArgument(format!(
                "Slice [{}..{}] is out of bounds for data of length {}",
                start,
                start + len,
                self.data.len()
            )));
        }

        Ok(&mut self.data[start..start + len])
    }

    /// Calculate storage efficiency compared to FP32
    ///
    /// Returns the ratio of this tensor's memory usage to what
    /// an equivalent FP32 tensor would require.
    pub fn storage_efficiency(&self) -> f32 {
        let fp32_size = self.num_elements() * 4; // 4 bytes per FP32
        if fp32_size == 0 {
            return 1.0;
        }
        self.memory_usage() as f32 / fp32_size as f32
    }

    /// Get compression ratio compared to FP32
    ///
    /// Returns how many times smaller this tensor is compared to FP32.
    pub fn compression_ratio(&self) -> f32 {
        1.0 / self.storage_efficiency()
    }

    /// Validate tensor consistency
    ///
    /// Checks that the tensor's data size, shape, and parameters
    /// are all consistent with each other.
    pub fn validate(&self) -> BackendResult<()> {
        // Validate quantization parameters
        self.params.validate()?;

        // Check data size consistency
        let expected_size = Self::calculate_data_size(self.num_elements(), &self.params.dtype);
        if self.data.len() != expected_size {
            return Err(torsh_core::error::TorshError::InvalidArgument(format!(
                "Data size inconsistency: expected {} bytes, actual {} bytes",
                expected_size,
                self.data.len()
            )));
        }

        // Check for empty shape
        if self.shape.is_empty() {
            return Err(torsh_core::error::TorshError::InvalidArgument(
                "Tensor shape cannot be empty".to_string(),
            ));
        }

        // Check for zero dimensions
        for (i, &dim) in self.shape.iter().enumerate() {
            if dim == 0 {
                return Err(torsh_core::error::TorshError::InvalidArgument(format!(
                    "Dimension {} cannot be zero",
                    i
                )));
            }
        }

        Ok(())
    }
}

/// Read-only view of a quantized tensor with different shape
///
/// Provides a view into an existing quantized tensor with a potentially
/// different shape, without copying the underlying data.
#[derive(Debug)]
pub struct QuantizedTensorView<'a> {
    /// Reference to the original data
    pub data: &'a [u8],
    /// View shape (may differ from original)
    pub shape: Vec<usize>,
    /// Reference to quantization parameters
    pub params: &'a QuantizationParams,
    /// Reference to device information
    pub device: &'a Device,
}

impl<'a> QuantizedTensorView<'a> {
    /// Get the number of elements in the view
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Get the memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        self.data.len()
    }

    /// Get the shape of the view
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the number of dimensions
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Convert view to owned tensor
    pub fn to_owned(&self) -> QuantizedTensor {
        QuantizedTensor {
            data: self.data.to_vec(),
            shape: self.shape.clone(),
            params: self.params.clone(),
            device: self.device.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantization::QuantizationParams;

    #[test]
    fn test_tensor_creation() {
        let shape = vec![2, 3, 4];
        let params = QuantizationParams::int8_symmetric();
        let device = Device::cpu().unwrap();
        let tensor = QuantizedTensor::new(shape.clone(), params, device.clone());

        assert_eq!(tensor.shape(), &shape);
        assert_eq!(tensor.num_elements(), 24);
        assert_eq!(tensor.memory_usage(), 24); // 1 byte per element for Int8
        assert_eq!(tensor.device, device);
    }

    #[test]
    fn test_int4_tensor_size() {
        let shape = vec![8];
        let params = QuantizationParams::int4_symmetric();
        let tensor = QuantizedTensor::new(shape, params, Device::cpu().unwrap());

        assert_eq!(tensor.num_elements(), 8);
        assert_eq!(tensor.memory_usage(), 4); // 2 elements per byte for Int4
    }

    #[test]
    fn test_binary_tensor_size() {
        let shape = vec![16];
        let mut params = QuantizationParams::default();
        params.dtype = QuantizedDType::Binary;
        let tensor = QuantizedTensor::new(shape, params, Device::cpu().unwrap());

        assert_eq!(tensor.num_elements(), 16);
        assert_eq!(tensor.memory_usage(), 2); // 8 elements per byte for Binary
    }

    #[test]
    fn test_tensor_from_data() {
        let data = vec![1, 2, 3, 4];
        let shape = vec![4];
        let params = QuantizationParams::int8_symmetric();
        let tensor =
            QuantizedTensor::from_data(data.clone(), shape, params, Device::cpu().unwrap())
                .unwrap();

        assert_eq!(tensor.data, data);
        assert_eq!(tensor.num_elements(), 4);
    }

    #[test]
    fn test_tensor_from_data_size_mismatch() {
        let data = vec![1, 2, 3]; // 3 bytes
        let shape = vec![4]; // Expects 4 bytes for Int8
        let params = QuantizationParams::int8_symmetric();
        let result = QuantizedTensor::from_data(data, shape, params, Device::cpu().unwrap());

        assert!(result.is_err());
    }

    #[test]
    fn test_tensor_reshape() {
        let tensor = QuantizedTensor::new(
            vec![2, 6],
            QuantizationParams::default(),
            Device::cpu().unwrap(),
        );
        let reshaped = tensor.reshape(vec![3, 4]).unwrap();

        assert_eq!(reshaped.shape(), &[3, 4]);
        assert_eq!(reshaped.num_elements(), 12);
        assert_eq!(reshaped.data.len(), tensor.data.len());
    }

    #[test]
    fn test_tensor_reshape_invalid() {
        let tensor = QuantizedTensor::new(
            vec![2, 6],
            QuantizationParams::default(),
            Device::cpu().unwrap(),
        );
        let result = tensor.reshape(vec![3, 5]); // 15 elements != 12

        assert!(result.is_err());
    }

    #[test]
    fn test_tensor_view() {
        let tensor = QuantizedTensor::new(
            vec![2, 6],
            QuantizationParams::default(),
            Device::cpu().unwrap(),
        );
        let view = tensor.view(vec![4, 3]).unwrap();

        assert_eq!(view.shape(), &[4, 3]);
        assert_eq!(view.num_elements(), 12);
        assert_eq!(view.data.len(), tensor.data.len());
    }

    #[test]
    fn test_storage_efficiency() {
        let int8_tensor = QuantizedTensor::new(
            vec![10],
            QuantizationParams::int8_symmetric(),
            Device::cpu().unwrap(),
        );
        assert_eq!(int8_tensor.storage_efficiency(), 0.25); // 1 byte vs 4 bytes per element

        let int4_tensor = QuantizedTensor::new(
            vec![10],
            QuantizationParams::int4_symmetric(),
            Device::cpu().unwrap(),
        );
        assert_eq!(int4_tensor.storage_efficiency(), 0.125); // 0.5 bytes vs 4 bytes per element
    }

    #[test]
    fn test_compression_ratio() {
        let int8_tensor = QuantizedTensor::new(
            vec![10],
            QuantizationParams::int8_symmetric(),
            Device::cpu().unwrap(),
        );
        assert_eq!(int8_tensor.compression_ratio(), 4.0); // 4x compression

        let int4_tensor = QuantizedTensor::new(
            vec![10],
            QuantizationParams::int4_symmetric(),
            Device::cpu().unwrap(),
        );
        assert_eq!(int4_tensor.compression_ratio(), 8.0); // 8x compression
    }

    #[test]
    fn test_data_slice() {
        let tensor = QuantizedTensor::new(
            vec![4],
            QuantizationParams::int8_symmetric(),
            Device::cpu().unwrap(),
        );
        let slice = tensor.data_slice(1, 2).unwrap();
        assert_eq!(slice.len(), 2);

        // Out of bounds should fail
        assert!(tensor.data_slice(3, 3).is_err());
    }

    #[test]
    fn test_tensor_validation() {
        let tensor = QuantizedTensor::new(
            vec![2, 3],
            QuantizationParams::default(),
            Device::cpu().unwrap(),
        );
        assert!(tensor.validate().is_ok());

        // Test with inconsistent data
        let mut bad_tensor = tensor.clone();
        bad_tensor.data.truncate(1); // Make data too small
        assert!(bad_tensor.validate().is_err());
    }

    #[test]
    fn test_tensor_properties() {
        let tensor = QuantizedTensor::new(
            vec![2, 3, 4],
            QuantizationParams::default(),
            Device::cpu().unwrap(),
        );

        assert_eq!(tensor.ndim(), 3);
        assert_eq!(tensor.size(0).unwrap(), 2);
        assert_eq!(tensor.size(1).unwrap(), 3);
        assert_eq!(tensor.size(2).unwrap(), 4);
        assert!(tensor.size(3).is_err()); // Out of bounds

        assert!(!tensor.is_empty());
    }

    #[test]
    fn test_empty_tensor() {
        let tensor = QuantizedTensor::new(
            vec![0],
            QuantizationParams::default(),
            Device::cpu().unwrap(),
        );
        assert!(tensor.validate().is_err()); // Zero dimension should be invalid
    }
}
