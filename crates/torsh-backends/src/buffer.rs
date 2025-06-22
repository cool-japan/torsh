//! Buffer management and memory operations

use crate::Device;
use torsh_core::{
    dtype::DType,
    error::{Result, TorshError},
    shape::Shape,
};

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, string::String, vec::Vec};

/// Buffer handle representing device memory
#[derive(Debug)]
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
    WebGpu { buffer_id: String, size: usize },

    /// Generic handle for custom backends
    Generic {
        handle: Box<dyn std::any::Any + Send + Sync>,
        size: usize,
    },
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

    /// Check if handle is valid
    pub fn is_valid(&self) -> bool {
        match self {
            BufferHandle::Cpu { ptr, size } => !ptr.is_null() && *size > 0,
            #[cfg(feature = "cuda")]
            BufferHandle::Cuda { device_ptr, size } => *device_ptr != 0 && *size > 0,
            #[cfg(feature = "metal")]
            BufferHandle::Metal { buffer_id, size } => *buffer_id != 0 && *size > 0,
            #[cfg(feature = "webgpu")]
            BufferHandle::WebGpu { buffer_id, size } => !buffer_id.is_empty() && *size > 0,
            BufferHandle::Generic { size, .. } => *size > 0,
        }
    }
}

// Note: BufferHandle should not implement Send/Sync automatically due to raw pointers
// Individual backends should ensure thread safety
unsafe impl Send for BufferHandle {}
unsafe impl Sync for BufferHandle {}

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
