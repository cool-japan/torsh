//! HDF5 metadata support for scientific computing workflows
//!
//! This module provides comprehensive HDF5-compatible metadata structures for tensor storage,
//! enabling seamless integration with scientific computing tools and libraries that use HDF5.
//!
//! # Features
//! - Dataset metadata with attributes and chunking information
//! - Group hierarchy support for organizing tensors
//! - Compression and filter metadata
//! - Dimension scales and coordinate systems
//! - Attribute storage for custom metadata

use crate::dtype::DType;
use crate::shape::Shape;

#[cfg(not(feature = "std"))]
use alloc::{
    string::{String, ToString},
    vec::Vec,
};

#[cfg(feature = "std")]
use std::collections::HashMap;

#[cfg(not(feature = "std"))]
use alloc::collections::BTreeMap as HashMap;

/// HDF5 datatype class
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Hdf5TypeClass {
    /// Integer types
    Integer,
    /// Floating-point types
    Float,
    /// String types
    String,
    /// Bitfield
    Bitfield,
    /// Opaque
    Opaque,
    /// Compound (struct)
    Compound,
    /// Reference
    Reference,
    /// Enum
    Enum,
    /// Variable-length
    VarLen,
    /// Array
    Array,
}

/// HDF5 byte order
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Hdf5ByteOrder {
    /// Little-endian
    LittleEndian,
    /// Big-endian
    BigEndian,
    /// Native (system-dependent)
    Native,
}

/// HDF5 datatype information
#[derive(Debug, Clone)]
pub struct Hdf5Datatype {
    /// Type class
    pub class: Hdf5TypeClass,

    /// Size in bytes
    pub size: usize,

    /// Byte order
    pub byte_order: Hdf5ByteOrder,

    /// Sign (for integers: true = signed, false = unsigned)
    pub sign: Option<bool>,

    /// Precision (number of significant bits)
    pub precision: Option<usize>,

    /// Offset (bit offset of the first significant bit)
    pub offset: Option<usize>,
}

impl Hdf5Datatype {
    /// Create HDF5 datatype from ToRSh DType
    pub fn from_dtype(dtype: DType) -> Self {
        match dtype {
            DType::F32 => Self {
                class: Hdf5TypeClass::Float,
                size: 4,
                byte_order: Hdf5ByteOrder::Native,
                sign: None,
                precision: Some(23),
                offset: Some(0),
            },
            DType::F64 => Self {
                class: Hdf5TypeClass::Float,
                size: 8,
                byte_order: Hdf5ByteOrder::Native,
                sign: None,
                precision: Some(52),
                offset: Some(0),
            },
            DType::I8 => Self {
                class: Hdf5TypeClass::Integer,
                size: 1,
                byte_order: Hdf5ByteOrder::Native,
                sign: Some(true),
                precision: Some(8),
                offset: Some(0),
            },
            DType::I16 => Self {
                class: Hdf5TypeClass::Integer,
                size: 2,
                byte_order: Hdf5ByteOrder::Native,
                sign: Some(true),
                precision: Some(16),
                offset: Some(0),
            },
            DType::I32 => Self {
                class: Hdf5TypeClass::Integer,
                size: 4,
                byte_order: Hdf5ByteOrder::Native,
                sign: Some(true),
                precision: Some(32),
                offset: Some(0),
            },
            DType::I64 => Self {
                class: Hdf5TypeClass::Integer,
                size: 8,
                byte_order: Hdf5ByteOrder::Native,
                sign: Some(true),
                precision: Some(64),
                offset: Some(0),
            },
            DType::U8 => Self {
                class: Hdf5TypeClass::Integer,
                size: 1,
                byte_order: Hdf5ByteOrder::Native,
                sign: Some(false),
                precision: Some(8),
                offset: Some(0),
            },
            DType::Bool => Self {
                class: Hdf5TypeClass::Integer,
                size: 1,
                byte_order: Hdf5ByteOrder::Native,
                sign: Some(false),
                precision: Some(1),
                offset: Some(0),
            },
            _ => Self {
                class: Hdf5TypeClass::Opaque,
                size: dtype.size(),
                byte_order: Hdf5ByteOrder::Native,
                sign: None,
                precision: None,
                offset: None,
            },
        }
    }

    /// Check if this is a floating-point type
    pub fn is_float(&self) -> bool {
        matches!(self.class, Hdf5TypeClass::Float)
    }

    /// Check if this is an integer type
    pub fn is_integer(&self) -> bool {
        matches!(self.class, Hdf5TypeClass::Integer)
    }

    /// Check if this is a signed type
    pub fn is_signed(&self) -> bool {
        self.sign.unwrap_or(false)
    }
}

/// HDF5 compression filter types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Hdf5Filter {
    /// No compression
    None,

    /// GZIP/DEFLATE compression (levels 0-9)
    Gzip { level: u8 },

    /// SZIP compression
    Szip,

    /// LZF compression (fast)
    Lzf,

    /// Shuffle filter (improves compression)
    Shuffle,

    /// Fletcher32 checksum
    Fletcher32,

    /// BZIP2 compression
    Bzip2 { level: u8 },

    /// LZ4 compression (very fast)
    Lz4,

    /// Blosc compression (multi-threaded)
    Blosc {
        compressor: BloscCompressor,
        level: u8,
        shuffle: BloscShuffle,
    },
}

/// Blosc compressor types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BloscCompressor {
    /// BLOSCLZ (default)
    BloscLz,
    /// LZ4
    Lz4,
    /// LZ4HC (high compression)
    Lz4Hc,
    /// Snappy
    Snappy,
    /// Zlib
    Zlib,
    /// Zstd
    Zstd,
}

/// Blosc shuffle options
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BloscShuffle {
    /// No shuffle
    None,
    /// Byte shuffle
    Byte,
    /// Bit shuffle
    Bit,
}

impl Hdf5Filter {
    /// Get compression ratio estimate (1.0 = no compression)
    pub fn compression_ratio_estimate(&self) -> f32 {
        match self {
            Self::None | Self::Shuffle | Self::Fletcher32 => 1.0,
            Self::Gzip { level } => 2.0 + (*level as f32 / 9.0) * 3.0, // 2-5x
            Self::Szip => 2.5,
            Self::Lzf => 1.8,
            Self::Bzip2 { level } => 2.5 + (*level as f32 / 9.0) * 2.5, // 2.5-5x
            Self::Lz4 => 1.5,
            Self::Blosc { level, .. } => 2.0 + (*level as f32 / 9.0) * 3.0, // 2-5x
        }
    }

    /// Check if filter is lossy
    pub fn is_lossy(&self) -> bool {
        false // All current filters are lossless
    }
}

/// HDF5 chunking strategy
#[derive(Debug, Clone)]
pub struct Hdf5Chunking {
    /// Chunk dimensions (must match dataset rank)
    pub chunk_dims: Vec<usize>,

    /// Cache size in bytes
    pub cache_size: Option<usize>,

    /// Number of slots in chunk cache
    pub cache_slots: Option<usize>,

    /// Chunk cache preemption policy (0.0-1.0)
    pub cache_w0: Option<f32>,
}

impl Hdf5Chunking {
    /// Create chunking configuration
    pub fn new(chunk_dims: Vec<usize>) -> Self {
        Self {
            chunk_dims,
            cache_size: None,
            cache_slots: None,
            cache_w0: None,
        }
    }

    /// Set cache size in bytes
    pub fn with_cache_size(mut self, size: usize) -> Self {
        self.cache_size = Some(size);
        self
    }

    /// Set number of cache slots
    pub fn with_cache_slots(mut self, slots: usize) -> Self {
        self.cache_slots = Some(slots);
        self
    }

    /// Set cache preemption policy
    pub fn with_cache_w0(mut self, w0: f32) -> Self {
        self.cache_w0 = Some(w0.clamp(0.0, 1.0));
        self
    }

    /// Calculate chunk size in bytes
    pub fn chunk_size_bytes(&self, dtype: &Hdf5Datatype) -> usize {
        let elements: usize = self.chunk_dims.iter().product();
        elements * dtype.size
    }

    /// Validate chunk dimensions against dataset shape
    pub fn validate(&self, shape: &Shape) -> bool {
        if self.chunk_dims.len() != shape.ndim() {
            return false;
        }

        for (chunk_dim, shape_dim) in self.chunk_dims.iter().zip(shape.dims()) {
            if *chunk_dim > *shape_dim || *chunk_dim == 0 {
                return false;
            }
        }

        true
    }

    /// Estimate number of chunks for a dataset
    pub fn estimate_num_chunks(&self, shape: &Shape) -> usize {
        if !self.validate(shape) {
            return 0;
        }

        shape
            .dims()
            .iter()
            .zip(self.chunk_dims.iter())
            .map(|(dim, chunk)| (dim + chunk - 1) / chunk)
            .product()
    }
}

/// HDF5 attribute value
#[derive(Debug, Clone)]
pub enum Hdf5AttributeValue {
    /// String value
    String(String),
    /// Integer value
    Int(i64),
    /// Float value
    Float(f64),
    /// Array of integers
    IntArray(Vec<i64>),
    /// Array of floats
    FloatArray(Vec<f64>),
    /// Array of strings
    StringArray(Vec<String>),
}

impl Hdf5AttributeValue {
    /// Get attribute type name
    pub fn type_name(&self) -> &str {
        match self {
            Self::String(_) => "string",
            Self::Int(_) => "int",
            Self::Float(_) => "float",
            Self::IntArray(_) => "int_array",
            Self::FloatArray(_) => "float_array",
            Self::StringArray(_) => "string_array",
        }
    }

    /// Get byte size estimate
    pub fn size_bytes(&self) -> usize {
        match self {
            Self::String(s) => s.len(),
            Self::Int(_) => 8,
            Self::Float(_) => 8,
            Self::IntArray(arr) => arr.len() * 8,
            Self::FloatArray(arr) => arr.len() * 8,
            Self::StringArray(arr) => arr.iter().map(|s| s.len()).sum::<usize>(),
        }
    }
}

/// HDF5 dataset metadata
#[derive(Debug, Clone)]
pub struct Hdf5DatasetMetadata {
    /// Dataset name
    pub name: String,

    /// Dataset shape
    pub shape: Shape,

    /// Data type
    pub dtype: Hdf5Datatype,

    /// Chunking configuration
    pub chunking: Option<Hdf5Chunking>,

    /// Compression filters
    pub filters: Vec<Hdf5Filter>,

    /// Dataset attributes
    pub attributes: HashMap<String, Hdf5AttributeValue>,

    /// Fill value (for uninitialized chunks)
    pub fill_value: Option<Vec<u8>>,

    /// Track times (creation, modification)
    pub track_times: bool,
}

impl Hdf5DatasetMetadata {
    /// Create new dataset metadata
    pub fn new(name: String, shape: Shape, dtype: DType) -> Self {
        Self {
            name,
            shape,
            dtype: Hdf5Datatype::from_dtype(dtype),
            chunking: None,
            filters: Vec::new(),
            attributes: HashMap::new(),
            fill_value: None,
            track_times: true,
        }
    }

    /// Set chunking configuration
    pub fn with_chunking(mut self, chunking: Hdf5Chunking) -> Self {
        self.chunking = Some(chunking);
        self
    }

    /// Add compression filter
    pub fn with_filter(mut self, filter: Hdf5Filter) -> Self {
        self.filters.push(filter);
        self
    }

    /// Add attribute
    pub fn with_attribute(mut self, key: String, value: Hdf5AttributeValue) -> Self {
        self.attributes.insert(key, value);
        self
    }

    /// Set fill value
    pub fn with_fill_value(mut self, value: Vec<u8>) -> Self {
        self.fill_value = Some(value);
        self
    }

    /// Enable/disable time tracking
    pub fn with_track_times(mut self, track: bool) -> Self {
        self.track_times = track;
        self
    }

    /// Calculate total dataset size in bytes
    pub fn dataset_size_bytes(&self) -> usize {
        self.shape.numel() * self.dtype.size
    }

    /// Calculate compressed size estimate
    pub fn compressed_size_estimate(&self) -> usize {
        let base_size = self.dataset_size_bytes();

        if self.filters.is_empty() {
            return base_size;
        }

        let compression_ratio: f32 = self
            .filters
            .iter()
            .map(|f| f.compression_ratio_estimate())
            .product();

        (base_size as f32 / compression_ratio) as usize
    }

    /// Get metadata size in bytes
    pub fn metadata_size_bytes(&self) -> usize {
        let mut size = 0;

        // Name
        size += self.name.len();

        // Shape and dtype
        size += self.shape.ndim() * 8 + 16;

        // Chunking
        if let Some(ref chunking) = self.chunking {
            size += chunking.chunk_dims.len() * 8 + 16;
        }

        // Filters
        size += self.filters.len() * 4;

        // Attributes
        for (key, value) in &self.attributes {
            size += key.len() + value.size_bytes();
        }

        size
    }
}

/// HDF5 dimension scale (coordinate system)
#[derive(Debug, Clone)]
pub struct Hdf5DimensionScale {
    /// Dimension name
    pub name: String,

    /// Scale values (coordinates)
    pub values: Vec<f64>,

    /// Scale label
    pub label: Option<String>,

    /// Units
    pub units: Option<String>,
}

impl Hdf5DimensionScale {
    /// Create new dimension scale
    pub fn new(name: String, values: Vec<f64>) -> Self {
        Self {
            name,
            values,
            label: None,
            units: None,
        }
    }

    /// Set label
    pub fn with_label(mut self, label: String) -> Self {
        self.label = Some(label);
        self
    }

    /// Set units
    pub fn with_units(mut self, units: String) -> Self {
        self.units = Some(units);
        self
    }

    /// Check if scale is uniform (evenly spaced)
    pub fn is_uniform(&self) -> bool {
        if self.values.len() < 2 {
            return true;
        }

        let delta = self.values[1] - self.values[0];
        let epsilon = delta.abs() * 1e-6;

        for i in 2..self.values.len() {
            let current_delta = self.values[i] - self.values[i - 1];
            if (current_delta - delta).abs() > epsilon {
                return false;
            }
        }

        true
    }

    /// Get scale step (for uniform scales)
    pub fn step(&self) -> Option<f64> {
        if self.values.len() < 2 {
            return None;
        }

        if self.is_uniform() {
            Some(self.values[1] - self.values[0])
        } else {
            None
        }
    }
}

/// HDF5 group metadata (for organizing datasets hierarchically)
#[derive(Debug, Clone)]
pub struct Hdf5GroupMetadata {
    /// Group name (path)
    pub name: String,

    /// Child datasets
    pub datasets: Vec<String>,

    /// Child groups
    pub groups: Vec<String>,

    /// Group attributes
    pub attributes: HashMap<String, Hdf5AttributeValue>,
}

impl Hdf5GroupMetadata {
    /// Create new group metadata
    pub fn new(name: String) -> Self {
        Self {
            name,
            datasets: Vec::new(),
            groups: Vec::new(),
            attributes: HashMap::new(),
        }
    }

    /// Add dataset
    pub fn add_dataset(mut self, dataset: String) -> Self {
        self.datasets.push(dataset);
        self
    }

    /// Add child group
    pub fn add_group(mut self, group: String) -> Self {
        self.groups.push(group);
        self
    }

    /// Add attribute
    pub fn with_attribute(mut self, key: String, value: Hdf5AttributeValue) -> Self {
        self.attributes.insert(key, value);
        self
    }

    /// Get total number of items
    pub fn num_items(&self) -> usize {
        self.datasets.len() + self.groups.len()
    }
}

/// HDF5 file metadata
#[derive(Debug, Clone)]
pub struct Hdf5FileMetadata {
    /// File format version
    pub version: (u8, u8),

    /// Root group
    pub root: Hdf5GroupMetadata,

    /// User block size (bytes before HDF5 data)
    pub user_block_size: Option<usize>,

    /// File creation properties
    pub creation_properties: HashMap<String, String>,
}

impl Hdf5FileMetadata {
    /// Create new file metadata
    pub fn new() -> Self {
        Self {
            version: (1, 10), // HDF5 1.10
            root: Hdf5GroupMetadata::new("/".to_string()),
            user_block_size: None,
            creation_properties: HashMap::new(),
        }
    }

    /// Set HDF5 version
    pub fn with_version(mut self, major: u8, minor: u8) -> Self {
        self.version = (major, minor);
        self
    }

    /// Set user block size
    pub fn with_user_block(mut self, size: usize) -> Self {
        self.user_block_size = Some(size);
        self
    }

    /// Add creation property
    pub fn with_property(mut self, key: String, value: String) -> Self {
        self.creation_properties.insert(key, value);
        self
    }

    /// Get version string
    pub fn version_string(&self) -> String {
        format!("{}.{}", self.version.0, self.version.1)
    }
}

impl Default for Hdf5FileMetadata {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hdf5_datatype_from_dtype() {
        let dtype = Hdf5Datatype::from_dtype(DType::F32);
        assert_eq!(dtype.class, Hdf5TypeClass::Float);
        assert_eq!(dtype.size, 4);
        assert!(dtype.is_float());
        assert!(!dtype.is_integer());

        let dtype = Hdf5Datatype::from_dtype(DType::I32);
        assert_eq!(dtype.class, Hdf5TypeClass::Integer);
        assert_eq!(dtype.size, 4);
        assert!(dtype.is_integer());
        assert!(dtype.is_signed());
    }

    #[test]
    fn test_hdf5_filter_compression_ratio() {
        let gzip = Hdf5Filter::Gzip { level: 9 };
        assert!(gzip.compression_ratio_estimate() > 2.0);

        let lz4 = Hdf5Filter::Lz4;
        assert_eq!(lz4.compression_ratio_estimate(), 1.5);

        assert!(!gzip.is_lossy());
    }

    #[test]
    fn test_hdf5_chunking() {
        let shape = Shape::new(vec![100, 200, 300]);
        let chunking = Hdf5Chunking::new(vec![10, 20, 30])
            .with_cache_size(1024 * 1024)
            .with_cache_slots(521);

        assert!(chunking.validate(&shape));
        assert_eq!(chunking.estimate_num_chunks(&shape), 10 * 10 * 10);

        let dtype = Hdf5Datatype::from_dtype(DType::F32);
        assert_eq!(chunking.chunk_size_bytes(&dtype), 10 * 20 * 30 * 4);
    }

    #[test]
    fn test_hdf5_dataset_metadata() {
        let shape = Shape::new(vec![100, 200]);
        let metadata = Hdf5DatasetMetadata::new("test_dataset".to_string(), shape, DType::F32)
            .with_chunking(Hdf5Chunking::new(vec![10, 20]))
            .with_filter(Hdf5Filter::Gzip { level: 6 })
            .with_attribute(
                "description".to_string(),
                Hdf5AttributeValue::String("Test dataset".to_string()),
            );

        assert_eq!(metadata.dataset_size_bytes(), 100 * 200 * 4);
        assert!(metadata.compressed_size_estimate() < metadata.dataset_size_bytes());
    }

    #[test]
    fn test_dimension_scale() {
        let scale = Hdf5DimensionScale::new("time".to_string(), vec![0.0, 1.0, 2.0, 3.0, 4.0])
            .with_units("seconds".to_string());

        assert!(scale.is_uniform());
        assert_eq!(scale.step(), Some(1.0));

        let non_uniform = Hdf5DimensionScale::new("x".to_string(), vec![0.0, 1.0, 3.0, 6.0]);
        assert!(!non_uniform.is_uniform());
        assert_eq!(non_uniform.step(), None);
    }

    #[test]
    fn test_hdf5_group_metadata() {
        let group = Hdf5GroupMetadata::new("/data".to_string())
            .add_dataset("tensor1".to_string())
            .add_dataset("tensor2".to_string())
            .add_group("subgroup".to_string())
            .with_attribute(
                "created".to_string(),
                Hdf5AttributeValue::String("2025-10-04".to_string()),
            );

        assert_eq!(group.num_items(), 3);
        assert_eq!(group.datasets.len(), 2);
        assert_eq!(group.groups.len(), 1);
    }

    #[test]
    fn test_hdf5_file_metadata() {
        let file = Hdf5FileMetadata::new()
            .with_version(1, 12)
            .with_user_block(512)
            .with_property("library".to_string(), "torsh".to_string());

        assert_eq!(file.version_string(), "1.12");
        assert_eq!(file.user_block_size, Some(512));
    }
}
