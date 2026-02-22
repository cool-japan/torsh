//! Vision utilities module
//!
//! This module provides a comprehensive suite of computer vision utilities organized into
//! specialized submodules for maintainability and ease of use. The utilities are designed
//! to work seamlessly with ToRSh tensors and provide PyTorch-compatible functionality.
//!
//! ## Module Organization
//!
//! The vision utilities are organized into 9 specialized modules:
//!
//! ### Core Data Operations
//! - [`image_conversion`] - Tensor/image conversion with normalization support
//! - [`image_io`] - File I/O operations for images and datasets
//! - [`image_processing`] - Basic image processing operations (resize, normalize, etc.)
//!
//! ### Analysis and Metrics
//! - [`statistics`] - Statistical analysis and image quality metrics (PSNR, SSIM, etc.)
//!
//! ### Rendering and Visualization
//! - [`text_rendering`] - Text rendering system with built-in fonts
//! - [`visualization`] - 2D visualization utilities (grids, bounding boxes)
//! - [`visualization_3d`] - 3D visualization engine for tensors and activations
//!
//! ### Performance and Interaction
//! - [`performance`] - Caching, prefetching, and performance optimization
//! - [`interactive`] - Interactive viewers and HTML generation
//!
//! ## Usage Examples
//!
//! ```rust
//! use torsh_vision::utils::{
//!     // Image conversion
//!     tensor_to_image, image_to_tensor,
//!     // Visualization
//!     make_grid, draw_bounding_boxes,
//!     // Statistics
//!     psnr, ssim,
//!     // Interactive tools
//!     create_interactive_viewer,
//! };
//! ```
//!
//! ## Backward Compatibility
//!
//! All existing imports and function calls remain fully compatible. The modular
//! organization enhances maintainability while preserving the existing API.

// ================================
// Module Declarations
// ================================

pub mod image_conversion;
pub mod image_io;
pub mod image_processing;
pub mod interactive;
pub mod performance;
pub mod statistics;
pub mod text_rendering;
pub mod visualization;
pub mod visualization_3d;

// ================================
// Core Data Operations
// ================================

/// Image/tensor conversion utilities with normalization support
pub use image_conversion::{denormalize, image_to_tensor, tensor_to_image};

/// File I/O operations for images and datasets
pub use image_io::{load_images_from_dir, save_tensor_as_image};

/// Basic image processing operations (resize, normalize, etc.)
pub use image_processing::{
    clamp_tensor, denormalize_tensor, normalize_tensor, resize_image, rgb_to_grayscale,
    validate_image_tensor_shape,
};

// ================================
// Analysis and Metrics
// ================================

/// Statistical analysis and image quality metrics
pub use statistics::{calculate_stats, mae, mse, psnr, ssim};

// ================================
// Rendering and Visualization
// ================================

/// Text rendering system with built-in fonts
pub use text_rendering::{
    calculate_text_height, calculate_text_width, draw_character, draw_simple_text,
    draw_text_with_background, get_simple_font_map, is_character_supported, UNKNOWN_CHAR_BITMAP,
};

/// 2D visualization utilities (grids, bounding boxes, etc.)
pub use visualization::{draw_bounding_boxes, make_grid};

/// 3D visualization engine for tensors and activations
pub use visualization_3d::{
    create_3d_visualizer, visualize_activations_3d, Mesh3D, Point3D, Visualizer3D, VoxelData,
};

// ================================
// Performance and Interaction
// ================================

/// Performance optimization infrastructure (caching, prefetching, etc.)
pub use performance::{
    BatchImageLoader, CacheEntry, CacheStats, ImageCache, ImagePrefetcher, LoadingMetrics,
    MemoryMappedLoader,
};

/// Interactive viewers and HTML generation
pub use interactive::{
    create_interactive_viewer, Annotation, AnnotationType, InteractiveViewer, Parameter,
    TransformOp,
};
