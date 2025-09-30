//! Data loading and preprocessing utilities for ToRSh
//!
//! This crate provides PyTorch-compatible data loading functionality,
//! including datasets, data loaders, and common transformations.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

pub mod builtin;
pub mod collate;
pub mod dataloader;
pub mod dataset;
pub mod error;
pub mod sampler;
pub mod transforms;
pub mod utils;

// TODO: Re-enable when async_dataloader module is implemented
// #[cfg(feature = "async-support")]
// pub use dataloader::async_dataloader::{
//     async_dataloader, AsyncDataLoader, AsyncDataLoaderBuilder, AsyncDataLoaderStream,
// };

// #[cfg(feature = "async-support")]
// pub mod async_utils {
//     pub use crate::dataloader::async_dataloader::async_utils::*;
// }

#[cfg(feature = "gpu-acceleration")]
pub mod gpu_acceleration;

// #[cfg(feature = "arrow")]
pub mod arrow_integration;

#[cfg(feature = "hdf5-support")]
pub mod hdf5_integration;

#[cfg(feature = "parquet-support")]
pub mod parquet_integration;

pub mod tfrecord_integration;

pub mod database_integration;

#[cfg(feature = "image-support")]
pub mod vision;

#[cfg(feature = "image-support")]
pub use vision::{
    Compose, ImageFolder, ImageNet, ImageToTensor, RandomHorizontalFlip, RandomRotation,
    RandomVerticalFlip, TensorToImage, TensorToVideo, VideoFolder, VideoFrames, VideoToTensor,
    CIFAR10, MNIST,
};

#[cfg(feature = "dataframe")]
pub mod tabular;

#[cfg(feature = "audio-support")]
pub mod audio;

pub mod augmentation_pipeline;
pub mod online_transforms;
pub mod tensor_transforms;
pub mod text;
pub mod text_processing;

#[cfg(feature = "privacy")]
pub mod privacy;

#[cfg(feature = "federated")]
pub mod federated;

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub mod wasm;

pub use collate::{
    collate_fn, AdaptiveBatchSampler, BucketBatchSampler, CachedCollate, Collate,
    DynamicBatchCollate,
};

#[cfg(feature = "std")]
pub use collate::{optimized_collate_fn, OptimizedCollate};

#[cfg(feature = "sparse")]
pub use collate::{MixedCollate, SparseCollate};
pub use dataloader::{simple_random_dataloader, DataLoader, DataLoaderBuilder, DataLoaderTrait};
pub use dataset::{
    random_split, BufferedStreamingDataset, CachedDataset, ChainDataset, ConcatDataset,
    DataPipeline, Dataset, DatasetToStreaming, InfiniteDataset, IterableDataset,
    PipelineStreamingDataset, RealTimeDataset, StreamingDataset, Subset, TensorDataset,
};

#[cfg(feature = "std")]
pub use dataset::SharedMemoryDataset;

#[cfg(all(feature = "std", feature = "mmap-support"))]
pub use dataset::MemoryMappedDataset;
pub use sampler::{
    AcquisitionStrategy, ActiveLearningSampler, AdaptiveSampler, AdaptiveStrategy, BatchSampler,
    BatchingSampler, CurriculumSampler, CurriculumStrategy, DistributedSampler, GroupedSampler,
    ImportanceSampler, RandomSampler, Sampler, SequentialSampler, StratifiedSampler,
    SubsetRandomSampler, WeightedRandomSampler,
};

#[cfg(feature = "privacy")]
pub use privacy::{
    dp_utils, CompositionType, DPMechanism, GaussianNoise, LaplaceNoise, NoiseGenerator,
    PrivacyBudget, PrivacyBuilder, PrivateDataset, PrivateSampler,
};

#[cfg(feature = "federated")]
pub use federated::{
    federated_utils, AggregationStrategy, ClientCapabilities, ClientId, ClientInfo,
    ClientSelectionStrategy, ComputePower, FederatedConfig, FederatedDataset,
    FederatedDatasetBuilder, FederatedSampler, NetworkBandwidth,
};

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use wasm::{optimization, wasm_utils, StreamingDataset, WasmDataLoader, WasmDataset};

pub use text::{
    TextClassificationDataset, TextFileDataset, TextSequence, TokenIdsToTensor, Vocabulary,
};

pub use error::{
    diagnostics, patterns, recovery, BatchInfo, CollationErrorKind, ConfigErrorKind, DataError,
    DataLoaderErrorKind, DatasetErrorKind, ErrorContext, ErrorSeverity, IoErrorKind,
    ResourceErrorKind, Result, SamplerErrorKind, TransformErrorKind, WithContext,
};

pub use transforms::{
    // TODO: Re-enable when modules are implemented
    // augmentation_pipeline,
    // // Specialized modules
    // core_framework,
    // lambda,
    // normalize,
    // online_transforms,
    // tensor_transforms,
    // text_processing,
    // to_type,
    // zero_copy,
    Chain,
    Compose as TransformCompose,
    Conditional,
    // Common types
    Lambda,
    Normalize,
    ToType,
    // Core framework
    Transform,
    TransformBuilder,
    TransformExt,
};

pub use utils::{
    batch, concurrent, create_size_tuple, errors, memory, performance, validate_dataset_path,
    validate_file_extension, validate_not_empty, validate_positive, validate_probability,
    validate_range, validate_same_length, validate_tensor_shape, Cacheable, Configurable,
    ProgressTracker, Resettable,
};

pub use builtin::{
    load_builtin_dataset, make_blobs, make_classification, make_regression, BuiltinDataset,
    ClassificationConfig, ClusteringConfig, DatasetRegistry, DatasetResult, RegressionConfig,
    ScalingMethod, SyntheticDataConfig,
};

// #[cfg(feature = "arrow")]
pub use arrow_integration::{arrow_utils, ArrowDataset};

#[cfg(feature = "hdf5-support")]
pub use hdf5_integration::{hdf5_utils, HDF5DatasetBuilder, HDF5Metadata, HDF5TensorDataset};

#[cfg(feature = "parquet-support")]
pub use parquet_integration::{parquet_utils, ParquetDataset, ParquetDatasetBuilder, ParquetError};

pub use tfrecord_integration::{
    tfrecord_utils, Example, FeatureValue, TFRecordDataset, TFRecordDatasetBuilder, TFRecordError,
    TFRecordReader,
};

pub use database_integration::{
    database_utils, DatabaseBackend, DatabaseConfig, DatabaseConnection, DatabaseDataset,
    DatabaseDatasetBuilder, DatabaseError, DatabaseRow, DatabaseValue,
};

// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const VERSION_MAJOR: u32 = 0;
pub const VERSION_MINOR: u32 = 1;
pub const VERSION_PATCH: u32 = 0;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::builtin::*;
    pub use crate::collate::*;
    pub use crate::dataloader::*;
    pub use crate::dataset::*;
    pub use crate::error::{DataError, ErrorContext, Result, WithContext};
    pub use crate::sampler::*;
    pub use crate::text::*;
    pub use crate::utils::{
        batch, concurrent, memory, performance, validate_not_empty, validate_positive,
        validate_probability, validate_range, Cacheable, Configurable, ProgressTracker, Resettable,
    };

    #[cfg(feature = "std")]
    pub use crate::dataset::SharedMemoryDataset;

    #[cfg(all(feature = "std", feature = "mmap-support"))]
    pub use crate::dataset::MemoryMappedDataset;

    #[cfg(feature = "parquet-support")]
    pub use crate::parquet_integration::{ParquetDataset, ParquetDatasetBuilder};

    pub use crate::tfrecord_integration::{
        TFRecordDataset, TFRecordDatasetBuilder, TFRecordReader,
    };

    pub use crate::database_integration::{
        DatabaseConfig, DatabaseDataset, DatabaseDatasetBuilder,
    };

    #[cfg(feature = "privacy")]
    pub use crate::privacy::{
        DPMechanism, PrivacyBudget, PrivacyBuilder, PrivateDataset, PrivateSampler,
    };

    #[cfg(feature = "federated")]
    pub use crate::federated::{
        ClientSelectionStrategy, FederatedConfig, FederatedDataset, FederatedDatasetBuilder,
    };

    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    pub use crate::wasm::{wasm_utils, WasmDataLoader, WasmDataset};

    // TODO: Re-enable when async_dataloader module is implemented
    // #[cfg(feature = "async-support")]
    // pub use crate::dataloader::async_dataloader::{
    //     async_dataloader, AsyncDataLoader, AsyncDataLoaderBuilder,
    // };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_imports() {
        // Basic smoke test
        let _ = SequentialSampler::new(10);
    }
}
