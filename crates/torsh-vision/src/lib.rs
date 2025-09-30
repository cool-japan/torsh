#![allow(clippy::all)]
#![allow(dead_code)]
#![allow(unused_imports)]

pub mod advanced_transforms;
pub mod benchmarks;
pub mod datasets;
pub mod datasets_impl;
pub mod error_handling;
pub mod examples;
pub mod hardware;
pub mod interactive;
pub mod io;
pub mod memory;
pub mod models;
pub mod ops;
pub mod optimized_impl;
pub mod prelude;
pub mod scirs2_integration;
pub mod spatial;
pub mod transforms;
pub mod unified_transforms;
pub mod utils;
pub mod video;
pub mod viz3d;

pub use advanced_transforms::*;
pub use datasets::{DatasetConfig, DatasetError, DatasetStats};
pub use datasets_impl::{CifarDataset, CocoDataset, ImageFolder, MnistDataset, VocDataset};
pub use error_handling::*;
pub use examples::*;
pub use hardware::*;
pub use interactive::*;
pub use io::*;
pub use memory::*;
pub use models::*;
pub use ops::{
    adjust_brightness, adjust_contrast, adjust_hue, adjust_saturation, calculate_iou, center_crop,
    edge_detection, gaussian_blur, generate_anchors, histogram_equalization, horizontal_flip,
    morphological_operation, nms, normalize, random_crop, resize, rgb_to_grayscale, rotate,
    sobel_edge_detection, vertical_flip,
};
pub use spatial::{
    distance::PatchMatcher,
    interpolation::{ImageWarper, OpticalFlowInterpolator, SpatialInterpolator},
    matching::{Feature, FeatureMatcher, Keypoint, TemplateMatcher},
    structures::{BoundingBox, PointCloudProcessor, SpatialObjectTracker},
    transforms::{GeometricProcessor, ImageRegistrar, PoseEstimator},
    FeatureMatch, SpatialConfig, SpatialPoint, SpatialProcessor, TransformResult,
};
pub use transforms::*;
pub use unified_transforms::*;
pub use utils::*;
pub use video::*;
pub use viz3d::*;

// Comprehensive scirs2-vision integration
pub use scirs2_integration::{
    ContrastMethod, CornerPoint, DenoiseMethod, DisparityMap, EdgeDetectionMethod,
    Keypoint as SciKeypoint, MemoryStrategy, OpticalFlow, OrbFeatures, QualityLevel,
    SciRS2VisionProcessor, SiftFeatures, SimdLevel, SurfFeatures, VisionConfig,
};

// Comprehensive benchmarking suite
pub use benchmarks::{
    run_full_benchmark_suite, run_quick_benchmark, AccuracyMetrics, BenchmarkConfig,
    BenchmarkResult, VisionBenchmarkSuite,
};

#[derive(Debug, thiserror::Error)]
pub enum VisionError {
    #[error("Image processing error: {0}")]
    ImageError(#[from] image::ImageError),

    #[error("Model error: {0}")]
    ModelError(String),

    #[error("Transform error: {0}")]
    TransformError(String),

    #[error("Invalid shape: {0}")]
    InvalidShape(String),

    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Tensor error: {0}")]
    TensorError(#[from] torsh_core::error::TorshError),

    #[cfg(feature = "pretrained")]
    #[error("Request error: {0}")]
    RequestError(#[from] reqwest::Error),

    #[error("Other error: {0}")]
    Other(#[from] anyhow::Error),
}

pub type Result<T> = std::result::Result<T, VisionError>;

// Conversion from VisionError to TorshError for Module trait compatibility
impl From<VisionError> for torsh_core::error::TorshError {
    fn from(err: VisionError) -> Self {
        torsh_core::error::TorshError::Other(err.to_string())
    }
}
