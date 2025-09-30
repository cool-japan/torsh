//! Buffer management and memory operations

use crate::Device;
use torsh_core::{
    dtype::DType,
    error::{Result, TorshError},
    shape::Shape,
};

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, string::String, vec::Vec};

#[cfg(not(feature = "std"))]
use core::sync::atomic::{AtomicUsize, Ordering};
#[cfg(feature = "std")]
use std::sync::atomic::{AtomicUsize, Ordering};

/// Global buffer ID generator
static BUFFER_ID_COUNTER: AtomicUsize = AtomicUsize::new(1);

/// Generate a unique buffer ID
pub fn generate_buffer_id() -> usize {
    BUFFER_ID_COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// Buffer handle representing device memory
#[derive(Debug, Clone)]
pub struct Buffer {
    /// Unique buffer ID
    pub id: usize,

    /// Device this buffer belongs to
    pub device: Device,

    /// Buffer size in bytes
    pub size: usize,

    /// Buffer usage flags
    pub usage: BufferUsage,

    /// Buffer descriptor used for creation
    pub descriptor: BufferDescriptor,

    /// Backend-specific handle (opaque)
    pub handle: BufferHandle,
}

impl Buffer {
    /// Create a new buffer
    pub fn new(
        id: usize,
        device: Device,
        size: usize,
        usage: BufferUsage,
        descriptor: BufferDescriptor,
        handle: BufferHandle,
    ) -> Self {
        Self {
            id,
            device,
            size,
            usage,
            descriptor,
            handle,
        }
    }

    /// Get buffer ID
    pub fn id(&self) -> usize {
        self.id
    }

    /// Get the device this buffer belongs to
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get buffer size in bytes
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get buffer usage flags
    pub fn usage(&self) -> BufferUsage {
        self.usage
    }

    /// Get the backend-specific handle
    pub fn handle(&self) -> &BufferHandle {
        &self.handle
    }

    /// Check if buffer can be used for the given usage
    pub fn supports_usage(&self, usage: BufferUsage) -> bool {
        self.usage.contains(usage)
    }
}

/// Buffer descriptor for creation
#[derive(Debug, Clone, PartialEq)]
pub struct BufferDescriptor {
    /// Buffer size in bytes
    pub size: usize,

    /// Buffer usage flags
    pub usage: BufferUsage,

    /// Memory location hint
    pub location: MemoryLocation,

    /// Data type stored in buffer (for type safety)
    pub dtype: Option<DType>,

    /// Shape of data in buffer (for tensor operations)
    pub shape: Option<Shape>,

    /// Initial data to copy to buffer
    pub initial_data: Option<Vec<u8>>,

    /// Memory alignment requirement
    pub alignment: Option<usize>,

    /// Whether buffer should be zero-initialized
    pub zero_init: bool,
}

impl BufferDescriptor {
    /// Create a new buffer descriptor
    pub fn new(size: usize, usage: BufferUsage) -> Self {
        Self {
            size,
            usage,
            location: MemoryLocation::Device,
            dtype: None,
            shape: None,
            initial_data: None,
            alignment: None,
            zero_init: false,
        }
    }

    /// Set memory location
    pub fn with_location(mut self, location: MemoryLocation) -> Self {
        self.location = location;
        self
    }

    /// Set data type
    pub fn with_dtype(mut self, dtype: DType) -> Self {
        self.dtype = Some(dtype);
        self
    }

    /// Set shape
    pub fn with_shape(mut self, shape: Shape) -> Self {
        self.shape = Some(shape);
        self
    }

    /// Set initial data
    pub fn with_initial_data(mut self, data: Vec<u8>) -> Self {
        self.initial_data = Some(data);
        self
    }

    /// Set alignment requirement
    pub fn with_alignment(mut self, alignment: usize) -> Self {
        self.alignment = Some(alignment);
        self
    }

    /// Enable zero initialization
    pub fn with_zero_init(mut self) -> Self {
        self.zero_init = true;
        self
    }
}

/// Buffer usage flags
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferUsage {
    bits: u32,
}

impl BufferUsage {
    /// Empty usage flags
    pub const NONE: Self = Self { bits: 0 };

    /// Buffer can be read from
    pub const READ: Self = Self { bits: 1 << 0 };

    /// Buffer can be written to
    pub const WRITE: Self = Self { bits: 1 << 1 };

    /// Buffer can be used as storage (compute shader)
    pub const STORAGE: Self = Self { bits: 1 << 2 };

    /// Buffer can be used as uniform data
    pub const UNIFORM: Self = Self { bits: 1 << 3 };

    /// Buffer can be used as vertex data
    pub const VERTEX: Self = Self { bits: 1 << 4 };

    /// Buffer can be used as index data
    pub const INDEX: Self = Self { bits: 1 << 5 };

    /// Buffer can be copied from
    pub const COPY_SRC: Self = Self { bits: 1 << 6 };

    /// Buffer can be copied to
    pub const COPY_DST: Self = Self { bits: 1 << 7 };

    /// Buffer can be mapped for host access
    pub const MAP_READ: Self = Self { bits: 1 << 8 };

    /// Buffer can be mapped for host writing
    pub const MAP_WRITE: Self = Self { bits: 1 << 9 };

    /// Commonly used combinations
    pub const READ_WRITE: Self = Self {
        bits: Self::READ.bits | Self::WRITE.bits,
    };
    pub const STORAGE_READ_WRITE: Self = Self {
        bits: Self::STORAGE.bits | Self::READ.bits | Self::WRITE.bits,
    };

    /// Create new usage flags
    pub const fn new(bits: u32) -> Self {
        Self { bits }
    }

    /// Check if usage contains the given flag
    pub const fn contains(self, other: Self) -> bool {
        (self.bits & other.bits) == other.bits
    }

    /// Combine with another usage flag
    pub const fn union(self, other: Self) -> Self {
        Self {
            bits: self.bits | other.bits,
        }
    }

    /// Remove a usage flag
    pub const fn difference(self, other: Self) -> Self {
        Self {
            bits: self.bits & !other.bits,
        }
    }

    /// Get the raw bits
    pub const fn bits(self) -> u32 {
        self.bits
    }
}

impl std::ops::BitOr for BufferUsage {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        self.union(rhs)
    }
}

impl std::ops::BitOrAssign for BufferUsage {
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs;
    }
}

/// Memory location hint
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MemoryLocation {
    /// Device memory (GPU VRAM, etc.)
    #[default]
    Device,

    /// Host memory (system RAM)
    Host,

    /// Unified memory (accessible from both host and device)
    Unified,

    /// Host memory that is cached by device
    HostCached,

    /// Device memory that is visible to host
    DeviceHost,
}

/// Backend-specific buffer handle
#[derive(Debug)]
pub enum BufferHandle {
    /// CPU buffer (raw pointer)
    Cpu { ptr: *mut u8, size: usize },

    /// CUDA buffer
    #[cfg(feature = "cuda")]
    Cuda { device_ptr: u64, size: usize },

    /// Metal buffer
    #[cfg(feature = "metal")]
    Metal { buffer_id: u64, size: usize },

    /// WebGPU buffer
    #[cfg(feature = "webgpu")]
    WebGpu { buffer_ptr: u64, size: usize },

    /// Generic handle for custom backends
    Generic {
        handle: Box<dyn std::any::Any + Send + Sync>,
        size: usize,
    },
}

impl Clone for BufferHandle {
    fn clone(&self) -> Self {
        match self {
            BufferHandle::Cpu { ptr, size } => BufferHandle::Cpu { ptr: *ptr, size: *size },
            #[cfg(feature = "cuda")]
            BufferHandle::Cuda { device_ptr, size } => BufferHandle::Cuda {
                device_ptr: *device_ptr,
                size: *size
            },
            #[cfg(feature = "metal")]
            BufferHandle::Metal { buffer_id, size } => BufferHandle::Metal {
                buffer_id: *buffer_id,
                size: *size
            },
            #[cfg(feature = "webgpu")]
            BufferHandle::WebGpu { buffer_ptr, size } => BufferHandle::WebGpu {
                buffer_ptr: *buffer_ptr,
                size: *size
            },
            BufferHandle::Generic { size, .. } => {
                // For Generic handles, we can't actually clone the Box<dyn Any>
                // This is a limitation - in practice, backends should avoid using Generic handles
                // for buffers that need to be cloned
                panic!("Cannot clone Generic buffer handles")
            }
        }
    }
}

impl BufferHandle {
    /// Get the size of the buffer
    pub fn size(&self) -> usize {
        match self {
            BufferHandle::Cpu { size, .. } => *size,
            #[cfg(feature = "cuda")]
            BufferHandle::Cuda { size, .. } => *size,
            #[cfg(feature = "metal")]
            BufferHandle::Metal { size, .. } => *size,
            #[cfg(feature = "webgpu")]
            BufferHandle::WebGpu { size, .. } => *size,
            BufferHandle::Generic { size, .. } => *size,
        }
    }

    /// Get a unique identifier for this buffer handle
    pub fn id(&self) -> usize {
        match self {
            BufferHandle::Cpu { ptr, .. } => *ptr as usize,
            #[cfg(feature = "cuda")]
            BufferHandle::Cuda { device_ptr, .. } => *device_ptr as usize,
            #[cfg(feature = "metal")]
            BufferHandle::Metal { buffer_id, .. } => *buffer_id as usize,
            #[cfg(feature = "webgpu")]
            BufferHandle::WebGpu { buffer_ptr, .. } => *buffer_ptr as usize,
            BufferHandle::Generic { .. } => 0, // Generic handles don't have meaningful IDs
        }
    }

    /// Check if handle is valid
    pub fn is_valid(&self) -> bool {
        match self {
            BufferHandle::Cpu { ptr, size } => !ptr.is_null() && *size > 0,
            #[cfg(feature = "cuda")]
            BufferHandle::Cuda { device_ptr, size } => *device_ptr != 0 && *size > 0,
            #[cfg(feature = "metal")]
            BufferHandle::Metal { buffer_id, size } => *buffer_id != 0 && *size > 0,
            #[cfg(feature = "webgpu")]
            BufferHandle::WebGpu { buffer_ptr, size } => *buffer_ptr != 0 && *size > 0,
            BufferHandle::Generic { size, .. } => *size > 0,
        }
    }
}

// Note: BufferHandle should not implement Send/Sync automatically due to raw pointers
// Individual backends should ensure thread safety
unsafe impl Send for BufferHandle {}
unsafe impl Sync for BufferHandle {}

impl PartialEq for BufferHandle {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (BufferHandle::Cpu { ptr: ptr1, size: size1 }, BufferHandle::Cpu { ptr: ptr2, size: size2 }) => {
                ptr1 == ptr2 && size1 == size2
            },
            #[cfg(feature = "cuda")]
            (BufferHandle::Cuda { device_ptr: ptr1, size: size1 }, BufferHandle::Cuda { device_ptr: ptr2, size: size2 }) => {
                ptr1 == ptr2 && size1 == size2
            },
            #[cfg(feature = "metal")]
            (BufferHandle::Metal { buffer_id: id1, size: size1 }, BufferHandle::Metal { buffer_id: id2, size: size2 }) => {
                id1 == id2 && size1 == size2
            },
            #[cfg(feature = "webgpu")]
            (BufferHandle::WebGpu { buffer_ptr: ptr1, size: size1 }, BufferHandle::WebGpu { buffer_ptr: ptr2, size: size2 }) => {
                ptr1 == ptr2 && size1 == size2
            },
            (BufferHandle::Generic { size: size1, .. }, BufferHandle::Generic { size: size2, .. }) => {
                // For Generic handles, we can only compare sizes
                size1 == size2
            },
            _ => false,
        }
    }
}

impl Eq for BufferHandle {}

impl std::hash::Hash for BufferHandle {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            BufferHandle::Cpu { ptr, size } => {
                0u8.hash(state); // discriminant
                (*ptr as usize).hash(state);
                size.hash(state);
            },
            #[cfg(feature = "cuda")]
            BufferHandle::Cuda { device_ptr, size } => {
                1u8.hash(state); // discriminant
                device_ptr.hash(state);
                size.hash(state);
            },
            #[cfg(feature = "metal")]
            BufferHandle::Metal { buffer_id, size } => {
                2u8.hash(state); // discriminant
                buffer_id.hash(state);
                size.hash(state);
            },
            #[cfg(feature = "webgpu")]
            BufferHandle::WebGpu { buffer_ptr, size } => {
                3u8.hash(state); // discriminant
                buffer_ptr.hash(state);
                size.hash(state);
            },
            BufferHandle::Generic { size, .. } => {
                4u8.hash(state); // discriminant
                size.hash(state);
            },
        }
    }
}

/// Buffer view for sub-buffer operations
#[derive(Debug)]
pub struct BufferView {
    /// Parent buffer
    pub buffer: Buffer,

    /// Offset in bytes from start of buffer
    pub offset: usize,

    /// Size of the view in bytes
    pub size: usize,

    /// Data type for typed views
    pub dtype: Option<DType>,

    /// Shape for tensor views
    pub shape: Option<Shape>,
}

impl BufferView {
    /// Create a new buffer view
    pub fn new(buffer: Buffer, offset: usize, size: usize) -> Result<Self> {
        if offset + size > buffer.size {
            return Err(TorshError::InvalidArgument(
                "Buffer view exceeds buffer bounds".to_string(),
            ));
        }

        Ok(Self {
            buffer,
            offset,
            size,
            dtype: None,
            shape: None,
        })
    }

    /// Create a typed buffer view
    pub fn typed(mut self, dtype: DType) -> Self {
        self.dtype = Some(dtype);
        self
    }

    /// Create a tensor buffer view
    pub fn shaped(mut self, shape: Shape) -> Self {
        self.shape = Some(shape);
        self
    }

    /// Get the underlying buffer
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// Get the offset
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Get the size
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get the end offset
    pub fn end_offset(&self) -> usize {
        self.offset + self.size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::{Device, DeviceInfo};
    use torsh_core::{device::DeviceType, dtype::DType, shape::Shape};

    fn create_test_device() -> Device {
        let info = DeviceInfo::default();
        Device::new(0, DeviceType::Cpu, "Test CPU".to_string(), info)
    }

    #[test]
    fn test_buffer_descriptor_creation() {
        let desc = BufferDescriptor::new(1024, BufferUsage::READ_WRITE);

        assert_eq!(desc.size, 1024);
        assert_eq!(desc.usage, BufferUsage::READ_WRITE);
        assert_eq!(desc.location, MemoryLocation::Device);
        assert_eq!(desc.dtype, None);
        assert_eq!(desc.shape, None);
        assert_eq!(desc.initial_data, None);
        assert_eq!(desc.alignment, None);
        assert!(!desc.zero_init);
    }

    #[test]
    fn test_buffer_descriptor_builder() {
        let desc = BufferDescriptor::new(2048, BufferUsage::STORAGE)
            .with_location(MemoryLocation::Host)
            .with_dtype(DType::F32)
            .with_shape(Shape::new(vec![64, 32]))
            .with_alignment(64)
            .with_zero_init();

        assert_eq!(desc.size, 2048);
        assert_eq!(desc.usage, BufferUsage::STORAGE);
        assert_eq!(desc.location, MemoryLocation::Host);
        assert_eq!(desc.dtype, Some(DType::F32));
        assert!(desc.shape.is_some());
        assert_eq!(desc.alignment, Some(64));
        assert!(desc.zero_init);
    }

    #[test]
    fn test_buffer_usage_flags() {
        let usage = BufferUsage::READ | BufferUsage::WRITE;
        assert!(usage.contains(BufferUsage::READ));
        assert!(usage.contains(BufferUsage::WRITE));
        assert!(!usage.contains(BufferUsage::STORAGE));

        let combined = BufferUsage::STORAGE_READ_WRITE;
        assert!(combined.contains(BufferUsage::STORAGE));
        assert!(combined.contains(BufferUsage::READ));
        assert!(combined.contains(BufferUsage::WRITE));
    }

    #[test]
    fn test_buffer_handle_validation() {
        let handle_valid = BufferHandle::Cpu {
            ptr: 0x1000 as *mut u8,
            size: 1024,
        };
        assert!(handle_valid.is_valid());
        assert_eq!(handle_valid.size(), 1024);

        let handle_invalid = BufferHandle::Cpu {
            ptr: std::ptr::null_mut(),
            size: 1024,
        };
        assert!(!handle_invalid.is_valid());
    }

    #[test]
    fn test_buffer_creation() {
        let device = create_test_device();
        let desc = BufferDescriptor::new(512, BufferUsage::READ_WRITE);
        let handle = BufferHandle::Cpu {
            ptr: 0x2000 as *mut u8,
            size: 512,
        };

        let buffer = Buffer::new(
            1,
            device.clone(),
            512,
            BufferUsage::READ_WRITE,
            desc.clone(),
            handle,
        );

        assert_eq!(buffer.id(), 1);
        assert_eq!(buffer.size(), 512);
        assert_eq!(buffer.usage(), BufferUsage::READ_WRITE);
        assert_eq!(buffer.device().id(), device.id());
        assert!(buffer.supports_usage(BufferUsage::READ));
        assert!(buffer.supports_usage(BufferUsage::WRITE));
        assert!(!buffer.supports_usage(BufferUsage::STORAGE));
    }

    #[test]
    fn test_buffer_view_creation() {
        let device = create_test_device();
        let desc = BufferDescriptor::new(1024, BufferUsage::READ_WRITE);
        let handle = BufferHandle::Cpu {
            ptr: 0x3000 as *mut u8,
            size: 1024,
        };

        let buffer = Buffer::new(1, device, 1024, BufferUsage::READ_WRITE, desc, handle);

        // Valid buffer view
        let view = BufferView::new(buffer, 256, 512).unwrap();
        assert_eq!(view.offset(), 256);
        assert_eq!(view.size(), 512);
        assert_eq!(view.end_offset(), 768);

        // Test with a new buffer for invalid case
        let device2 = create_test_device();
        let desc2 = BufferDescriptor::new(1024, BufferUsage::READ_WRITE);
        let handle2 = BufferHandle::Cpu {
            ptr: 0x3001 as *mut u8,
            size: 1024,
        };
        let buffer2 = Buffer::new(2, device2, 1024, BufferUsage::READ_WRITE, desc2, handle2);
        let invalid_view = BufferView::new(buffer2, 800, 512);
        assert!(invalid_view.is_err());
    }

    #[test]
    fn test_buffer_view_typed() {
        let device = create_test_device();
        let desc = BufferDescriptor::new(1024, BufferUsage::READ_WRITE);
        let handle = BufferHandle::Cpu {
            ptr: 0x4000 as *mut u8,
            size: 1024,
        };

        let buffer = Buffer::new(1, device, 1024, BufferUsage::READ_WRITE, desc, handle);
        let view = BufferView::new(buffer, 0, 1024)
            .unwrap()
            .typed(DType::F32)
            .shaped(Shape::new(vec![256])); // 256 f32 values = 1024 bytes

        assert_eq!(view.dtype, Some(DType::F32));
        assert!(view.shape.is_some());
    }

    #[test]
    fn test_memory_location_variants() {
        assert_eq!(MemoryLocation::default(), MemoryLocation::Device);

        let locations = [
            MemoryLocation::Device,
            MemoryLocation::Host,
            MemoryLocation::Unified,
            MemoryLocation::HostCached,
            MemoryLocation::DeviceHost,
        ];

        for location in locations {
            let desc = BufferDescriptor::new(1024, BufferUsage::READ_WRITE).with_location(location);
            assert_eq!(desc.location, location);
        }
    }
}
