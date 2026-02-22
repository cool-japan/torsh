//! Vision-specific datasets and transformations
//!
//! This module provides comprehensive computer vision support including:
//! - Image datasets and transformations
//! - Standard vision datasets (MNIST, CIFAR-10, ImageNet)
//! - Video processing components
//!
//! The module is organized into focused submodules:
//! - `image`: Image datasets and transformations
//! - `datasets`: Standard vision datasets
//! - `video`: Video processing (extracted inline for now)

pub mod datasets;
pub mod image;

// Re-export for backward compatibility
pub use datasets::{ImageNet, CIFAR10, MNIST};
pub use image::transforms::transforms::{CenterCrop, Normalize, Resize};
pub use image::{
    Compose, ImageFolder, ImageToTensor, RandomHorizontalFlip, RandomRotation, RandomVerticalFlip,
    TensorToImage,
};

// Inline video components for now (extracted from original)
use crate::{dataset::Dataset, transforms::Transform};
use torsh_core::error::{Result, TorshError};
use torsh_tensor::Tensor;

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, string::String, vec::Vec};
use std::path::{Path, PathBuf};

/// Video frame container
#[derive(Debug, Clone)]
pub struct VideoFrames {
    frames: Vec<Tensor<f32>>,
    fps: f32,
    duration: f32,
}

impl VideoFrames {
    pub fn new(frames: Vec<Tensor<f32>>, fps: f32) -> Self {
        let duration = frames.len() as f32 / fps;
        Self {
            frames,
            fps,
            duration,
        }
    }

    pub fn frames(&self) -> &[Tensor<f32>] {
        &self.frames
    }

    pub fn fps(&self) -> f32 {
        self.fps
    }

    pub fn duration(&self) -> f32 {
        self.duration
    }

    pub fn num_frames(&self) -> usize {
        self.frames.len()
    }
}

/// Video dataset for loading video files from directories
pub struct VideoFolder {
    root: PathBuf,
    samples: Vec<(PathBuf, usize)>,
    classes: Vec<String>,
    transform: Option<Box<dyn Transform<VideoFrames, Output = Tensor<f32>>>>,
    max_frames: usize,
    frame_rate: Option<f32>,
}

impl VideoFolder {
    /// Create a new video folder dataset
    pub fn new<P: AsRef<Path>>(root: P) -> Result<Self> {
        let root = root.as_ref().to_path_buf();

        if !root.exists() {
            return Err(TorshError::IoError(format!(
                "Directory does not exist: {root:?}"
            )));
        }

        let mut classes = Vec::new();
        let mut samples = Vec::new();

        // Scan subdirectories for classes
        for entry in std::fs::read_dir(&root).map_err(|e| TorshError::IoError(e.to_string()))? {
            let entry = entry.map_err(|e| TorshError::IoError(e.to_string()))?;
            let path = entry.path();

            if path.is_dir() {
                let class_name = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .ok_or_else(|| TorshError::IoError("Invalid class directory name".to_string()))?
                    .to_string();

                let class_idx = classes.len();
                classes.push(class_name);

                // Scan videos in class directory
                for video_entry in
                    std::fs::read_dir(&path).map_err(|e| TorshError::IoError(e.to_string()))?
                {
                    let video_entry =
                        video_entry.map_err(|e| TorshError::IoError(e.to_string()))?;
                    let video_path = video_entry.path();

                    if Self::is_video_file(&video_path) {
                        samples.push((video_path, class_idx));
                    }
                }
            }
        }

        Ok(Self {
            root,
            samples,
            classes,
            transform: None,
            max_frames: 32, // Default to 32 frames
            frame_rate: None,
        })
    }

    /// Set maximum number of frames to extract
    pub fn with_max_frames(mut self, max_frames: usize) -> Self {
        self.max_frames = max_frames;
        self
    }

    /// Set target frame rate for extraction
    pub fn with_frame_rate(mut self, fps: f32) -> Self {
        self.frame_rate = Some(fps);
        self
    }

    /// Set transform to apply to video frames
    pub fn with_transform<T>(mut self, transform: T) -> Self
    where
        T: Transform<VideoFrames, Output = Tensor<f32>> + 'static,
    {
        self.transform = Some(Box::new(transform));
        self
    }

    /// Get class names
    pub fn classes(&self) -> &[String] {
        &self.classes
    }

    /// Get the root directory path
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Get the number of samples
    pub fn num_samples(&self) -> usize {
        self.samples.len()
    }

    /// Check if file is a supported video format
    fn is_video_file(path: &Path) -> bool {
        if let Some(extension) = path.extension().and_then(|ext| ext.to_str()) {
            matches!(
                extension.to_lowercase().as_str(),
                "mp4" | "avi" | "mov" | "mkv" | "wmv" | "flv" | "webm"
            )
        } else {
            false
        }
    }

    /// Load video frames (simplified implementation)
    fn load_video(&self, _path: &Path) -> Result<VideoFrames> {
        // In a real implementation, this would use ffmpeg or similar to extract frames
        // For now, create dummy video data
        let mut frames = Vec::new();
        for _i in 0..self.max_frames {
            // Create dummy frame (3 channels, 224x224 - typical video frame size)
            let frame = torsh_tensor::creation::rand::<f32>(&[3, 224, 224])?;
            frames.push(frame);
        }

        let fps = self.frame_rate.unwrap_or(30.0);
        Ok(VideoFrames::new(frames, fps))
    }
}

impl Dataset for VideoFolder {
    type Item = (Tensor<f32>, usize);

    fn len(&self) -> usize {
        self.samples.len()
    }

    fn get(&self, index: usize) -> Result<Self::Item> {
        if index >= self.samples.len() {
            return Err(TorshError::IndexError {
                index,
                size: self.samples.len(),
            });
        }

        let (ref path, class_idx) = self.samples[index];
        let video_frames = self.load_video(path)?;

        let tensor = if let Some(ref transform) = self.transform {
            transform.transform(video_frames)?
        } else {
            // Default: convert to tensor (concatenate frames along batch dimension)
            VideoToTensor.transform(video_frames)?
        };

        Ok((tensor, class_idx))
    }
}

/// Transform to convert video frames to tensor
pub struct VideoToTensor;

impl Transform<VideoFrames> for VideoToTensor {
    type Output = Tensor<f32>;

    fn transform(&self, input: VideoFrames) -> Result<Self::Output> {
        let frames = input.frames();
        if frames.is_empty() {
            return Err(TorshError::InvalidArgument(
                "VideoFrames cannot be empty".to_string(),
            ));
        }

        // Get frame dimensions
        let frame_shape = frames[0].shape();
        let dims = frame_shape.dims();

        if dims.len() != 3 {
            return Err(TorshError::InvalidShape(
                "Expected 3D frame tensors (C, H, W)".to_string(),
            ));
        }

        let (channels, height, width) = (dims[0], dims[1], dims[2]);
        let num_frames = frames.len();

        // Concatenate frames into a single tensor: (T, C, H, W)
        let mut video_data = Vec::with_capacity(num_frames * channels * height * width);

        for frame in frames {
            let frame_data = frame.to_vec()?;
            video_data.extend(frame_data);
        }

        Tensor::from_data(
            video_data,
            vec![num_frames, channels, height, width],
            torsh_core::device::DeviceType::Cpu,
        )
    }
}

/// Transform to convert tensor to video frames
pub struct TensorToVideo {
    fps: f32,
}

impl TensorToVideo {
    pub fn new(fps: f32) -> Self {
        Self { fps }
    }
}

impl Transform<Tensor<f32>> for TensorToVideo {
    type Output = VideoFrames;

    fn transform(&self, input: Tensor<f32>) -> Result<Self::Output> {
        let shape = input.shape();
        let dims = shape.dims();

        if dims.len() != 4 {
            return Err(TorshError::InvalidShape(
                "Expected 4D tensor (T, C, H, W)".to_string(),
            ));
        }

        let (num_frames, channels, height, width) = (dims[0], dims[1], dims[2], dims[3]);
        let frame_size = channels * height * width;

        let data = input.to_vec()?;
        let mut frames = Vec::with_capacity(num_frames);

        for t in 0..num_frames {
            let start_idx = t * frame_size;
            let end_idx = start_idx + frame_size;
            let frame_data = data[start_idx..end_idx].to_vec();

            let frame = Tensor::from_data(
                frame_data,
                vec![channels, height, width],
                torsh_core::device::DeviceType::Cpu,
            )?;

            frames.push(frame);
        }

        Ok(VideoFrames::new(frames, self.fps))
    }
}
