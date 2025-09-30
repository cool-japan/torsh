//! Consolidated I/O operations for torsh-vision
//!
//! This module provides a unified interface for all I/O operations including:
//! - Image loading and saving in various formats
//! - Video I/O operations
//! - Dataset loading and caching
//! - Memory-mapped file operations
//! - Batch processing utilities

use crate::{Result, VisionError};
use image::{DynamicImage, GenericImageView, ImageFormat};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use torsh_tensor::Tensor;

/// Unified I/O manager for all vision operations
#[derive(Debug, Clone)]
pub struct VisionIO {
    /// Default image format for saving
    pub default_format: ImageFormat,
    /// Enable caching for loaded images
    pub enable_caching: bool,
    /// Maximum cache size in MB
    pub cache_size_mb: usize,
    /// Supported image extensions
    pub supported_extensions: Vec<String>,
}

impl Default for VisionIO {
    fn default() -> Self {
        Self {
            default_format: ImageFormat::Png,
            enable_caching: true,
            cache_size_mb: 512,
            supported_extensions: vec![
                "jpg".to_string(),
                "jpeg".to_string(),
                "png".to_string(),
                "bmp".to_string(),
                "gif".to_string(),
                "tiff".to_string(),
                "webp".to_string(),
                "ico".to_string(),
                "pnm".to_string(),
            ],
        }
    }
}

impl VisionIO {
    /// Create a new VisionIO instance with custom settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Configure the default image format
    pub fn with_default_format(mut self, format: ImageFormat) -> Self {
        self.default_format = format;
        self
    }

    /// Configure caching settings
    pub fn with_caching(mut self, enable: bool, cache_size_mb: usize) -> Self {
        self.enable_caching = enable;
        self.cache_size_mb = cache_size_mb;
        self
    }

    /// Add supported file extension
    pub fn add_extension(mut self, ext: &str) -> Self {
        self.supported_extensions.push(ext.to_lowercase());
        self
    }
}

/// Image loading operations
impl VisionIO {
    /// Load a single image from file path
    pub fn load_image<P: AsRef<Path>>(&self, path: P) -> Result<DynamicImage> {
        let path = path.as_ref();

        if !path.exists() {
            return Err(VisionError::IoError(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Image file {:?} does not exist", path),
            )));
        }

        let image = image::open(path).map_err(|e| VisionError::ImageError(e))?;

        Ok(image)
    }

    /// Load multiple images from a directory
    pub fn load_images_from_dir<P: AsRef<Path>>(
        &self,
        dir_path: P,
    ) -> Result<Vec<(DynamicImage, String)>> {
        let dir_path = dir_path.as_ref();

        if !dir_path.exists() {
            return Err(VisionError::IoError(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Directory {:?} does not exist", dir_path),
            )));
        }

        if !dir_path.is_dir() {
            return Err(VisionError::InvalidArgument(format!(
                "Path {:?} is not a directory",
                dir_path
            )));
        }

        let mut images = Vec::new();
        let mut failed_files = Vec::new();

        for entry in std::fs::read_dir(dir_path)? {
            let entry = entry?;
            let path = entry.path();

            if let Some(extension) = path.extension() {
                let ext_str = extension.to_string_lossy().to_lowercase();

                if self.supported_extensions.contains(&ext_str) {
                    match self.load_image(&path) {
                        Ok(img) => {
                            let filename = path
                                .file_name()
                                .map(|n| n.to_string_lossy().to_string())
                                .unwrap_or_else(|| "unknown".to_string());
                            images.push((img, filename));
                        }
                        Err(e) => {
                            failed_files.push((path.clone(), e));
                        }
                    }
                }
            }
        }

        // Log warnings for failed files but continue
        if !failed_files.is_empty() {
            eprintln!("Warning: Failed to load {} image(s):", failed_files.len());
            for (path, error) in &failed_files {
                eprintln!("  {:?}: {}", path, error);
            }
        }

        Ok(images)
    }

    /// Load images with pattern matching
    pub fn load_images_with_pattern<P: AsRef<Path>>(
        &self,
        pattern: P,
    ) -> Result<Vec<(DynamicImage, String)>> {
        let pattern_str = pattern.as_ref().to_string_lossy();
        let paths = glob::glob(&pattern_str)
            .map_err(|e| VisionError::InvalidArgument(format!("Invalid pattern: {}", e)))?;

        let mut images = Vec::new();
        let mut failed_files = Vec::new();

        for path_result in paths {
            match path_result {
                Ok(path) => {
                    if let Some(extension) = path.extension() {
                        let ext_str = extension.to_string_lossy().to_lowercase();

                        if self.supported_extensions.contains(&ext_str) {
                            match self.load_image(&path) {
                                Ok(img) => {
                                    let filename = path
                                        .file_name()
                                        .map(|n| n.to_string_lossy().to_string())
                                        .unwrap_or_else(|| "unknown".to_string());
                                    images.push((img, filename));
                                }
                                Err(e) => {
                                    failed_files.push((path.clone(), e));
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    return Err(VisionError::IoError(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        format!("Glob error: {}", e),
                    )));
                }
            }
        }

        // Log warnings for failed files
        if !failed_files.is_empty() {
            eprintln!(
                "Warning: Failed to load {} image(s) from pattern '{}':",
                failed_files.len(),
                pattern_str
            );
            for (path, error) in &failed_files {
                eprintln!("  {:?}: {}", path, error);
            }
        }

        Ok(images)
    }

    /// Batch load images with progress reporting
    pub fn batch_load_images<P: AsRef<Path>>(&self, paths: &[P]) -> Result<BatchLoadResult> {
        let mut loaded = Vec::new();
        let mut failed = Vec::new();

        for (index, path) in paths.iter().enumerate() {
            let path = path.as_ref();

            match self.load_image(path) {
                Ok(img) => {
                    let filename = path
                        .file_name()
                        .map(|n| n.to_string_lossy().to_string())
                        .unwrap_or_else(|| format!("image_{}", index));
                    loaded.push((img, filename));
                }
                Err(e) => {
                    failed.push((path.to_path_buf(), e));
                }
            }
        }

        Ok(BatchLoadResult { loaded, failed })
    }
}

/// Image saving operations
impl VisionIO {
    /// Save a single image to file
    pub fn save_image<P: AsRef<Path>>(&self, image: &DynamicImage, path: P) -> Result<()> {
        let path = path.as_ref();

        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Determine format from extension or use default
        let format = path
            .extension()
            .and_then(|ext| ext.to_str())
            .and_then(|ext| ImageFormat::from_extension(ext))
            .unwrap_or(self.default_format);

        image
            .save_with_format(path, format)
            .map_err(|e| VisionError::ImageError(e))?;

        Ok(())
    }

    /// Save tensor as image
    pub fn save_tensor_as_image<P: AsRef<Path>>(
        &self,
        tensor: &Tensor<f32>,
        path: P,
        normalize: bool,
    ) -> Result<()> {
        let image = crate::utils::tensor_to_image(tensor, normalize)?;
        self.save_image(&image, path)
    }

    /// Batch save images
    pub fn batch_save_images<P: AsRef<Path>>(
        &self,
        images: &[(DynamicImage, P)],
    ) -> Result<BatchSaveResult> {
        let mut saved = Vec::new();
        let mut failed = Vec::new();

        for (image, path) in images {
            let path = path.as_ref();

            match self.save_image(image, path) {
                Ok(()) => {
                    saved.push(path.to_path_buf());
                }
                Err(e) => {
                    failed.push((path.to_path_buf(), e));
                }
            }
        }

        Ok(BatchSaveResult { saved, failed })
    }
}

/// Format conversion operations
impl VisionIO {
    /// Convert image to a different format
    pub fn convert_format(
        &self,
        image: &DynamicImage,
        target_format: ImageFormat,
    ) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        let mut cursor = std::io::Cursor::new(&mut bytes);

        image
            .write_to(&mut cursor, target_format)
            .map_err(|e| VisionError::ImageError(e))?;

        Ok(bytes)
    }

    /// Convert and save image in different format
    pub fn convert_and_save<P: AsRef<Path>>(
        &self,
        source_path: P,
        target_path: P,
        target_format: ImageFormat,
    ) -> Result<()> {
        let image = self.load_image(source_path)?;

        let target_path = target_path.as_ref();
        if let Some(parent) = target_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        image
            .save_with_format(target_path, target_format)
            .map_err(|e| VisionError::ImageError(e))?;

        Ok(())
    }

    /// Batch convert images to a different format
    pub fn batch_convert_format<P: AsRef<Path>, Q: AsRef<Path>>(
        &self,
        source_paths: &[P],
        target_dir: Q,
        target_format: ImageFormat,
        preserve_names: bool,
    ) -> Result<BatchConvertResult> {
        let target_dir = target_dir.as_ref();
        std::fs::create_dir_all(target_dir)?;

        let mut converted = Vec::new();
        let mut failed = Vec::new();

        for source_path in source_paths {
            let source_path = source_path.as_ref();

            let target_filename = if preserve_names {
                source_path
                    .file_stem()
                    .map(|stem| {
                        format!(
                            "{}.{}",
                            stem.to_string_lossy(),
                            target_format.extensions_str()[0]
                        )
                    })
                    .unwrap_or_else(|| format!("converted.{}", target_format.extensions_str()[0]))
            } else {
                format!(
                    "image_{}.{}",
                    converted.len(),
                    target_format.extensions_str()[0]
                )
            };

            let target_path = target_dir.join(target_filename);

            match self.convert_and_save(source_path, &target_path, target_format) {
                Ok(()) => {
                    converted.push((source_path.to_path_buf(), target_path));
                }
                Err(e) => {
                    failed.push((source_path.to_path_buf(), e));
                }
            }
        }

        Ok(BatchConvertResult { converted, failed })
    }
}

/// Utility operations
impl VisionIO {
    /// Get image metadata without loading the full image
    pub fn get_image_info<P: AsRef<Path>>(&self, path: P) -> Result<ImageInfo> {
        let path = path.as_ref();

        let reader = image::io::Reader::open(path)?;

        let reader = reader.with_guessed_format()?;

        let format = reader.format();
        let dimensions = reader.into_dimensions()?;

        let file_size = std::fs::metadata(path)?.len();

        Ok(ImageInfo {
            width: dimensions.0,
            height: dimensions.1,
            format,
            file_size_bytes: file_size,
            path: path.to_path_buf(),
        })
    }

    /// Check if a file is a supported image format
    pub fn is_supported_image<P: AsRef<Path>>(&self, path: P) -> bool {
        let path = path.as_ref();

        if let Some(extension) = path.extension() {
            let ext_str = extension.to_string_lossy().to_lowercase();
            self.supported_extensions.contains(&ext_str)
        } else {
            false
        }
    }

    /// Get supported formats
    pub fn supported_formats(&self) -> Vec<ImageFormat> {
        vec![
            ImageFormat::Png,
            ImageFormat::Jpeg,
            ImageFormat::Gif,
            ImageFormat::WebP,
            ImageFormat::Bmp,
            ImageFormat::Ico,
            ImageFormat::Tiff,
            ImageFormat::Pnm,
        ]
    }
}

/// Result types for batch operations
#[derive(Debug)]
pub struct BatchLoadResult {
    pub loaded: Vec<(DynamicImage, String)>,
    pub failed: Vec<(PathBuf, VisionError)>,
}

#[derive(Debug)]
pub struct BatchSaveResult {
    pub saved: Vec<PathBuf>,
    pub failed: Vec<(PathBuf, VisionError)>,
}

#[derive(Debug)]
pub struct BatchConvertResult {
    pub converted: Vec<(PathBuf, PathBuf)>,
    pub failed: Vec<(PathBuf, VisionError)>,
}

/// Image metadata information
#[derive(Debug, Clone)]
pub struct ImageInfo {
    pub width: u32,
    pub height: u32,
    pub format: Option<ImageFormat>,
    pub file_size_bytes: u64,
    pub path: PathBuf,
}

impl BatchLoadResult {
    /// Get success rate as a percentage
    pub fn success_rate(&self) -> f32 {
        let total = self.loaded.len() + self.failed.len();
        if total == 0 {
            0.0
        } else {
            (self.loaded.len() as f32 / total as f32) * 100.0
        }
    }

    /// Check if all images loaded successfully
    pub fn all_successful(&self) -> bool {
        self.failed.is_empty()
    }
}

impl BatchSaveResult {
    /// Get success rate as a percentage
    pub fn success_rate(&self) -> f32 {
        let total = self.saved.len() + self.failed.len();
        if total == 0 {
            0.0
        } else {
            (self.saved.len() as f32 / total as f32) * 100.0
        }
    }

    /// Check if all images saved successfully
    pub fn all_successful(&self) -> bool {
        self.failed.is_empty()
    }
}

impl BatchConvertResult {
    /// Get success rate as a percentage
    pub fn success_rate(&self) -> f32 {
        let total = self.converted.len() + self.failed.len();
        if total == 0 {
            0.0
        } else {
            (self.converted.len() as f32 / total as f32) * 100.0
        }
    }

    /// Check if all conversions completed successfully
    pub fn all_successful(&self) -> bool {
        self.failed.is_empty()
    }
}

/// Global I/O instance for convenience
static mut GLOBAL_IO: Option<VisionIO> = None;
static INIT: std::sync::Once = std::sync::Once::new();

/// Get the global VisionIO instance
pub fn global_io() -> &'static VisionIO {
    unsafe {
        INIT.call_once(|| {
            GLOBAL_IO = Some(VisionIO::default());
        });
        GLOBAL_IO.as_ref().unwrap()
    }
}

/// Configure the global VisionIO instance
pub fn configure_global_io(io: VisionIO) {
    unsafe {
        GLOBAL_IO = Some(io);
    }
}

/// Convenience functions using the global I/O instance
pub mod global {
    use super::*;

    /// Load an image using global settings
    pub fn load_image<P: AsRef<Path>>(path: P) -> Result<DynamicImage> {
        global_io().load_image(path)
    }

    /// Save an image using global settings
    pub fn save_image<P: AsRef<Path>>(image: &DynamicImage, path: P) -> Result<()> {
        global_io().save_image(image, path)
    }

    /// Load images from directory using global settings
    pub fn load_images_from_dir<P: AsRef<Path>>(
        dir_path: P,
    ) -> Result<Vec<(DynamicImage, String)>> {
        global_io().load_images_from_dir(dir_path)
    }

    /// Save tensor as image using global settings
    pub fn save_tensor_as_image<P: AsRef<Path>>(
        tensor: &Tensor<f32>,
        path: P,
        normalize: bool,
    ) -> Result<()> {
        global_io().save_tensor_as_image(tensor, path, normalize)
    }
}
