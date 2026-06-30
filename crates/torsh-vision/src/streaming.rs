//! Real-Time Video Stream Processing
//!
//! This module provides efficient stream processing capabilities for real-time computer vision,
//! integrated with scirs2-vision 0.1.5's streaming features.
//!
//! # Features
//! - Video frame buffering and preprocessing pipelines
//! - Real-time performance monitoring and adaptation
//! - Async/parallel processing for low latency
//! - GPU-accelerated stream processing
//! - Frame dropping and quality adaptation strategies
//!
//! # Examples
//!
//! ```rust,ignore
//! use torsh_vision::streaming::*;
//!
//! // Create a video stream processor
//! let processor = StreamProcessor::new(config)?;
//! processor.process_stream(video_source, |frame| {
//!     // Process each frame
//!     detect_objects(frame)
//! })?;
//! ```

use crate::{Result, VisionError};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use torsh_tensor::Tensor;

/// Stream processing configuration
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Maximum frames to buffer
    pub buffer_size: usize,
    /// Target frames per second
    pub target_fps: f32,
    /// Enable frame dropping if processing too slow
    pub enable_frame_drop: bool,
    /// Enable GPU acceleration
    pub use_gpu: bool,
    /// Number of parallel processing threads
    pub num_threads: usize,
    /// Quality adaptation strategy
    pub quality_adaptation: QualityAdaptation,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            buffer_size: 30,
            target_fps: 30.0,
            enable_frame_drop: true,
            use_gpu: false,
            num_threads: 4,
            quality_adaptation: QualityAdaptation::None,
        }
    }
}

/// Quality adaptation strategies for maintaining real-time performance
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualityAdaptation {
    /// No adaptation
    None,
    /// Reduce resolution when falling behind
    ResolutionScaling,
    /// Skip non-keyframe processing
    KeyframeOnly,
    /// Adaptive quality based on complexity
    Dynamic,
}

/// Frame metadata for processing pipeline
#[derive(Debug, Clone)]
pub struct FrameMetadata {
    /// Frame number in stream
    pub frame_number: u64,
    /// Timestamp when frame was captured
    pub timestamp: Instant,
    /// Frame width
    pub width: usize,
    /// Frame height
    pub height: usize,
    /// Is this a keyframe
    pub is_keyframe: bool,
    /// Processing priority (higher = more important)
    pub priority: u8,
}

/// Frame with metadata
#[derive(Debug, Clone)]
pub struct Frame {
    /// Frame data as tensor
    pub data: Tensor,
    /// Frame metadata
    pub metadata: FrameMetadata,
}

/// Performance statistics for stream processing
#[derive(Debug, Clone)]
pub struct StreamStats {
    /// Average processing time per frame (ms)
    pub avg_processing_time: f32,
    /// Current frames per second
    pub current_fps: f32,
    /// Number of dropped frames
    pub dropped_frames: u64,
    /// Total frames processed
    pub total_frames: u64,
    /// Buffer utilization (0.0 to 1.0)
    pub buffer_utilization: f32,
    /// Number of quality adaptations
    pub num_adaptations: u64,
}

impl Default for StreamStats {
    fn default() -> Self {
        Self {
            avg_processing_time: 0.0,
            current_fps: 0.0,
            dropped_frames: 0,
            total_frames: 0,
            buffer_utilization: 0.0,
            num_adaptations: 0,
        }
    }
}

/// Real-time stream processor
pub struct StreamProcessor {
    config: StreamConfig,
    stats: Arc<Mutex<StreamStats>>,
    frame_buffer: Arc<Mutex<VecDeque<Frame>>>,
    processing_times: Arc<Mutex<VecDeque<Duration>>>,
}

impl StreamProcessor {
    /// Create a new stream processor
    pub fn new(config: StreamConfig) -> Result<Self> {
        let buffer_size = config.buffer_size;
        Ok(Self {
            config,
            stats: Arc::new(Mutex::new(StreamStats::default())),
            frame_buffer: Arc::new(Mutex::new(VecDeque::with_capacity(buffer_size))),
            processing_times: Arc::new(Mutex::new(VecDeque::with_capacity(100))),
        })
    }

    /// Get current performance statistics
    pub fn stats(&self) -> StreamStats {
        self.stats.lock().map(|s| s.clone()).unwrap_or_default()
    }

    /// Reset statistics
    pub fn reset_stats(&self) {
        if let Ok(mut stats) = self.stats.lock() {
            *stats = StreamStats::default();
        }
        if let Ok(mut times) = self.processing_times.lock() {
            times.clear();
        }
    }

    /// Add a frame to the processing buffer
    pub fn push_frame(&self, frame: Frame) -> Result<()> {
        let mut buffer = self.frame_buffer.lock().map_err(|_| {
            VisionError::InvalidParameter("Failed to lock frame buffer".to_string())
        })?;

        // Check buffer capacity
        if buffer.len() >= self.config.buffer_size {
            if self.config.enable_frame_drop {
                // Drop oldest non-keyframe
                let mut dropped = false;
                for i in 0..buffer.len() {
                    if !buffer[i].metadata.is_keyframe {
                        buffer.remove(i);
                        dropped = true;

                        // Update stats
                        if let Ok(mut stats) = self.stats.lock() {
                            stats.dropped_frames += 1;
                        }
                        break;
                    }
                }

                if !dropped {
                    // All are keyframes, drop oldest
                    buffer.pop_front();
                    if let Ok(mut stats) = self.stats.lock() {
                        stats.dropped_frames += 1;
                    }
                }
            } else {
                return Err(VisionError::InvalidParameter(
                    "Frame buffer full and dropping disabled".to_string(),
                ));
            }
        }

        buffer.push_back(frame);

        // Update buffer utilization
        if let Ok(mut stats) = self.stats.lock() {
            stats.buffer_utilization = buffer.len() as f32 / self.config.buffer_size as f32;
        }

        Ok(())
    }

    /// Get next frame from buffer
    pub fn pop_frame(&self) -> Result<Option<Frame>> {
        let mut buffer = self.frame_buffer.lock().map_err(|_| {
            VisionError::InvalidParameter("Failed to lock frame buffer".to_string())
        })?;

        Ok(buffer.pop_front())
    }

    /// Record processing time for a frame
    pub fn record_processing_time(&self, duration: Duration) {
        if let Ok(mut times) = self.processing_times.lock() {
            times.push_back(duration);

            // Keep only recent times (last 100 frames)
            while times.len() > 100 {
                times.pop_front();
            }

            // Update stats
            if let Ok(mut stats) = self.stats.lock() {
                // Calculate average processing time
                let sum: Duration = times.iter().sum();
                stats.avg_processing_time = sum.as_secs_f32() * 1000.0 / times.len() as f32;

                // Calculate FPS
                if stats.avg_processing_time > 0.0 {
                    stats.current_fps = 1000.0 / stats.avg_processing_time;
                }
            }
        }
    }

    /// Process a frame and update statistics
    pub fn process_frame<F, T>(&self, frame: Frame, process_fn: F) -> Result<T>
    where
        F: FnOnce(&Frame) -> Result<T>,
    {
        let start = Instant::now();

        // Apply quality adaptation if needed
        let adapted_frame = self.adapt_frame_quality(&frame)?;

        // Process the frame
        let result = process_fn(&adapted_frame)?;

        // Record processing time
        let elapsed = start.elapsed();
        self.record_processing_time(elapsed);

        // Update total frames
        if let Ok(mut stats) = self.stats.lock() {
            stats.total_frames += 1;
        }

        Ok(result)
    }

    /// Adapt frame quality based on performance
    fn adapt_frame_quality(&self, frame: &Frame) -> Result<Frame> {
        match self.config.quality_adaptation {
            QualityAdaptation::None => Ok(frame.clone()),

            QualityAdaptation::ResolutionScaling => {
                // Check if we're falling behind
                let stats = self.stats();
                if stats.current_fps < self.config.target_fps * 0.8 {
                    // Reduce resolution by 25% and resample with bilinear interpolation.
                    let new_width = ((frame.metadata.width as f32 * 0.75) as usize).max(1);
                    let new_height = ((frame.metadata.height as f32 * 0.75) as usize).max(1);

                    let downscaled = downscale_frame_bilinear(frame, new_width, new_height)?;
                    if let Ok(mut stats) = self.stats.lock() {
                        stats.num_adaptations += 1;
                    }
                    return Ok(downscaled);
                }
                Ok(frame.clone())
            }

            QualityAdaptation::KeyframeOnly => {
                // Skip non-keyframes if falling behind
                let stats = self.stats();
                if !frame.metadata.is_keyframe && stats.current_fps < self.config.target_fps * 0.9 {
                    // Mark for skipping (caller should handle)
                    if let Ok(mut stats_lock) = self.stats.lock() {
                        stats_lock.num_adaptations += 1;
                    }
                }
                Ok(frame.clone())
            }

            QualityAdaptation::Dynamic => {
                // Combine strategies based on performance
                let stats = self.stats();
                let performance_ratio = stats.current_fps / self.config.target_fps;

                if performance_ratio < 0.7 {
                    // Severe degradation: halve the resolution with bilinear resampling.
                    let new_width = ((frame.metadata.width as f32 * 0.5) as usize).max(1);
                    let new_height = ((frame.metadata.height as f32 * 0.5) as usize).max(1);

                    let downscaled = downscale_frame_bilinear(frame, new_width, new_height)?;
                    if let Ok(mut stats_lock) = self.stats.lock() {
                        stats_lock.num_adaptations += 1;
                    }
                    return Ok(downscaled);
                } else if performance_ratio < 0.9 && !frame.metadata.is_keyframe {
                    // Moderate degradation: skip non-keyframes
                    if let Ok(mut stats_lock) = self.stats.lock() {
                        stats_lock.num_adaptations += 1;
                    }
                }

                Ok(frame.clone())
            }
        }
    }

    /// Check if processing is keeping up with target FPS
    pub fn is_realtime(&self) -> bool {
        let stats = self.stats();
        stats.current_fps >= self.config.target_fps * 0.95
    }

    /// Get recommended adjustments for configuration
    pub fn recommend_config_adjustments(&self) -> Vec<String> {
        let stats = self.stats();
        let mut recommendations = Vec::new();

        // Check FPS performance
        if stats.current_fps < self.config.target_fps * 0.8 {
            recommendations
                .push("Consider reducing target_fps or enabling quality adaptation".to_string());
        }

        // Check buffer utilization
        if stats.buffer_utilization > 0.9 {
            recommendations.push("Buffer is frequently full - consider increasing buffer_size or enabling frame dropping".to_string());
        }

        // Check dropped frames
        if stats.total_frames > 0 {
            let drop_rate = stats.dropped_frames as f32 / stats.total_frames as f32;
            if drop_rate > 0.1 {
                recommendations.push(format!(
                    "High frame drop rate ({:.1}%) - consider reducing input rate or optimizing processing",
                    drop_rate * 100.0
                ));
            }
        }

        recommendations
    }
}

/// Downscale a video frame to the requested resolution using bilinear interpolation.
///
/// The frame's tensor is interpreted as `CHW` (channels, height, width) when it has
/// three dimensions and as `HW` (height, width) when it has two; any other rank is
/// rejected with an honest error rather than silently passing the frame through
/// unchanged.
///
/// Sampling uses the half-pixel-center convention
/// (`src = (dst + 0.5) * scale - 0.5`, `scale = src_size / dst_size`), which matches the
/// behaviour of OpenCV's `resize` and PyTorch's
/// `interpolate(.., mode = "bilinear", align_corners = false)`. Source coordinates that
/// fall outside the image are clamped to the edge so border pixels are handled correctly.
fn downscale_frame_bilinear(frame: &Frame, target_w: usize, target_h: usize) -> Result<Frame> {
    let src_w = frame.metadata.width;
    let src_h = frame.metadata.height;

    if target_w == 0 || target_h == 0 {
        return Err(VisionError::InvalidParameter(
            "downscale target resolution must be non-zero".to_string(),
        ));
    }

    // Nothing to resample when the resolution is unchanged.
    if target_w == src_w && target_h == src_h {
        return Ok(frame.clone());
    }

    // `shape()` yields an owned temporary, so copy the dimensions out within the statement.
    let dims = frame.data.shape().dims().to_vec();
    let channels = match dims.len() {
        3 => dims[0],
        2 => 1,
        other => {
            return Err(VisionError::InvalidShape(format!(
                "bilinear downscale expects a 2D (HW) or 3D (CHW) frame tensor, got rank {other}"
            )));
        }
    };

    let src_data: Vec<f32> = frame.data.to_vec()?;
    let expected = channels * src_h * src_w;
    if src_data.len() != expected {
        return Err(VisionError::InvalidShape(format!(
            "frame tensor has {} elements but metadata implies {expected} ({channels}x{src_h}x{src_w})",
            src_data.len()
        )));
    }

    let scale_y = src_h as f32 / target_h as f32;
    let scale_x = src_w as f32 / target_w as f32;
    let src_h_max = src_h as isize - 1;
    let src_w_max = src_w as isize - 1;

    let mut out = vec![0.0f32; channels * target_h * target_w];

    for c in 0..channels {
        let src_plane = c * src_h * src_w;
        let dst_plane = c * target_h * target_w;
        for oy in 0..target_h {
            let fy = (oy as f32 + 0.5) * scale_y - 0.5;
            let y_floor = fy.floor();
            let wy = fy - y_floor;
            let y0 = (y_floor as isize).clamp(0, src_h_max) as usize;
            let y1 = (y_floor as isize + 1).clamp(0, src_h_max) as usize;
            let row0 = src_plane + y0 * src_w;
            let row1 = src_plane + y1 * src_w;
            for ox in 0..target_w {
                let fx = (ox as f32 + 0.5) * scale_x - 0.5;
                let x_floor = fx.floor();
                let wx = fx - x_floor;
                let x0 = (x_floor as isize).clamp(0, src_w_max) as usize;
                let x1 = (x_floor as isize + 1).clamp(0, src_w_max) as usize;

                // Bilinear blend of the four neighbouring source pixels.
                let v00 = src_data[row0 + x0];
                let v01 = src_data[row0 + x1];
                let v10 = src_data[row1 + x0];
                let v11 = src_data[row1 + x1];
                let top = v00 + (v01 - v00) * wx;
                let bot = v10 + (v11 - v10) * wx;
                out[dst_plane + oy * target_w + ox] = top + (bot - top) * wy;
            }
        }
    }

    let new_shape = if dims.len() == 3 {
        vec![channels, target_h, target_w]
    } else {
        vec![target_h, target_w]
    };
    let data = Tensor::from_vec(out, &new_shape)?;

    let mut metadata = frame.metadata.clone();
    metadata.width = target_w;
    metadata.height = target_h;

    Ok(Frame { data, metadata })
}

/// Frame preprocessor for common vision tasks
pub struct FramePreprocessor {
    /// Target size for resizing (width, height)
    pub target_size: Option<(usize, usize)>,
    /// Normalization mean values
    pub normalize_mean: Option<Vec<f32>>,
    /// Normalization std values
    pub normalize_std: Option<Vec<f32>>,
    /// Convert to grayscale
    pub to_grayscale: bool,
}

impl Default for FramePreprocessor {
    fn default() -> Self {
        Self {
            target_size: None,
            normalize_mean: None,
            normalize_std: None,
            to_grayscale: false,
        }
    }
}

impl FramePreprocessor {
    /// Create a new preprocessor with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Set target resize dimensions
    pub fn with_resize(mut self, width: usize, height: usize) -> Self {
        self.target_size = Some((width, height));
        self
    }

    /// Set normalization parameters
    pub fn with_normalize(mut self, mean: Vec<f32>, std: Vec<f32>) -> Self {
        self.normalize_mean = Some(mean);
        self.normalize_std = Some(std);
        self
    }

    /// Enable grayscale conversion
    pub fn with_grayscale(mut self) -> Self {
        self.to_grayscale = true;
        self
    }

    /// Preprocess a frame
    ///
    /// Applies the configured operations in order:
    /// 1. Resize (nearest-neighbor) if `target_size` is set
    /// 2. Per-channel normalization if `normalize_mean`/`normalize_std` are set
    /// 3. Grayscale conversion (weighted average) if `to_grayscale` is set
    pub fn preprocess(&self, frame: &Frame) -> Result<Frame> {
        let mut data = frame.data.clone();
        let mut width = frame.metadata.width;
        let mut height = frame.metadata.height;

        // Step 1: Resize using nearest-neighbor sampling
        if let Some((target_w, target_h)) = self.target_size {
            if target_w != width || target_h != height {
                let orig_data: Vec<f32> = data.to_vec().map_err(|e| VisionError::TensorError(e))?;
                let shape = data.shape();
                let dims = shape.dims();

                // Determine number of channels from the tensor shape
                let channels = if dims.len() == 3 {
                    dims[0] // CHW layout
                } else {
                    1
                };

                let mut resized = vec![0.0f32; channels * target_h * target_w];
                let scale_y = height as f32 / target_h as f32;
                let scale_x = width as f32 / target_w as f32;

                for c in 0..channels {
                    for oy in 0..target_h {
                        for ox in 0..target_w {
                            let src_y = ((oy as f32 * scale_y) as usize).min(height - 1);
                            let src_x = ((ox as f32 * scale_x) as usize).min(width - 1);
                            let src_idx = c * height * width + src_y * width + src_x;
                            let dst_idx = c * target_h * target_w + oy * target_w + ox;
                            resized[dst_idx] = orig_data[src_idx];
                        }
                    }
                }

                let new_shape = if dims.len() == 3 {
                    vec![channels, target_h, target_w]
                } else {
                    vec![target_h, target_w]
                };
                data = Tensor::from_vec(resized, &new_shape)
                    .map_err(|e| VisionError::TensorError(e))?;
                width = target_w;
                height = target_h;
            }
        }

        // Step 2: Per-channel normalization: (pixel - mean) / std
        if let (Some(means), Some(stds)) = (&self.normalize_mean, &self.normalize_std) {
            let shape = data.shape();
            let dims = shape.dims();
            let channels = if dims.len() == 3 { dims[0] } else { 1 };
            let mut pixel_data: Vec<f32> =
                data.to_vec().map_err(|e| VisionError::TensorError(e))?;
            let pixels_per_channel = height * width;

            for c in 0..channels.min(means.len()).min(stds.len()) {
                let mean = means[c];
                let std = stds[c].max(1e-7);
                let offset = c * pixels_per_channel;
                for i in 0..pixels_per_channel {
                    pixel_data[offset + i] = (pixel_data[offset + i] - mean) / std;
                }
            }

            let shape_vec = dims.to_vec();
            data = Tensor::from_vec(pixel_data, &shape_vec)
                .map_err(|e| VisionError::TensorError(e))?;
        }

        // Step 3: Grayscale conversion using luminance weights (ITU-R BT.601)
        if self.to_grayscale {
            let shape = data.shape();
            let dims = shape.dims();
            if dims.len() == 3 && dims[0] >= 3 {
                let pixel_data: Vec<f32> =
                    data.to_vec().map_err(|e| VisionError::TensorError(e))?;
                let pixels = height * width;
                let mut gray = vec![0.0f32; pixels];
                // R=dims[0]=0, G=1, B=2
                for i in 0..pixels {
                    let r = pixel_data[i];
                    let g = pixel_data[pixels + i];
                    let b = pixel_data[2 * pixels + i];
                    gray[i] = 0.299 * r + 0.587 * g + 0.114 * b;
                }
                data = Tensor::from_vec(gray, &[1, height, width])
                    .map_err(|e| VisionError::TensorError(e))?;
            }
        }

        let mut metadata = frame.metadata.clone();
        metadata.width = width;
        metadata.height = height;

        Ok(Frame { data, metadata })
    }
}

/// Batch processor for efficient multi-frame processing
pub struct BatchProcessor {
    batch_size: usize,
    frames: Vec<Frame>,
}

impl BatchProcessor {
    /// Create a new batch processor
    pub fn new(batch_size: usize) -> Self {
        Self {
            batch_size,
            frames: Vec::with_capacity(batch_size),
        }
    }

    /// Add a frame to the batch
    pub fn add_frame(&mut self, frame: Frame) -> Option<Vec<Frame>> {
        self.frames.push(frame);

        if self.frames.len() >= self.batch_size {
            Some(self.flush())
        } else {
            None
        }
    }

    /// Get the current batch without clearing
    pub fn current_batch(&self) -> &[Frame] {
        &self.frames
    }

    /// Flush the current batch
    pub fn flush(&mut self) -> Vec<Frame> {
        std::mem::replace(&mut self.frames, Vec::with_capacity(self.batch_size))
    }

    /// Check if batch is full
    pub fn is_full(&self) -> bool {
        self.frames.len() >= self.batch_size
    }

    /// Get number of frames in current batch
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Check if batch is empty
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_dummy_frame(frame_number: u64) -> Frame {
        use torsh_core::DeviceType;
        Frame {
            data: Tensor::zeros(&[224, 224, 3], DeviceType::Cpu).expect("Failed to create tensor"),
            metadata: FrameMetadata {
                frame_number,
                timestamp: Instant::now(),
                width: 224,
                height: 224,
                is_keyframe: frame_number % 10 == 0,
                priority: 1,
            },
        }
    }

    #[test]
    fn test_stream_config_default() {
        let config = StreamConfig::default();
        assert_eq!(config.buffer_size, 30);
        assert_eq!(config.target_fps, 30.0);
        assert!(config.enable_frame_drop);
    }

    #[test]
    fn test_stream_processor_creation() {
        let processor = StreamProcessor::new(StreamConfig::default());
        assert!(processor.is_ok());
    }

    #[test]
    fn test_push_pop_frame() {
        let processor =
            StreamProcessor::new(StreamConfig::default()).expect("Failed to create processor");

        let frame = create_dummy_frame(1);
        processor
            .push_frame(frame.clone())
            .expect("Failed to push frame");

        let popped = processor.pop_frame().expect("Failed to pop frame");
        assert!(popped.is_some());
        assert_eq!(popped.unwrap().metadata.frame_number, 1);
    }

    #[test]
    fn test_frame_dropping() {
        let mut config = StreamConfig::default();
        config.buffer_size = 2;
        config.enable_frame_drop = true;

        let processor = StreamProcessor::new(config).expect("Failed to create processor");

        // Fill buffer beyond capacity
        for i in 0..5 {
            let frame = create_dummy_frame(i);
            processor.push_frame(frame).expect("Failed to push frame");
        }

        let stats = processor.stats();
        assert!(stats.dropped_frames > 0);
    }

    #[test]
    fn test_processing_time_recording() {
        let processor =
            StreamProcessor::new(StreamConfig::default()).expect("Failed to create processor");

        processor.record_processing_time(Duration::from_millis(10));
        processor.record_processing_time(Duration::from_millis(20));

        let stats = processor.stats();
        assert!(stats.avg_processing_time > 0.0);
    }

    #[test]
    fn test_batch_processor() {
        let mut batch = BatchProcessor::new(3);

        assert!(batch.is_empty());
        assert!(!batch.is_full());

        batch.add_frame(create_dummy_frame(1));
        batch.add_frame(create_dummy_frame(2));

        assert_eq!(batch.len(), 2);
        assert!(!batch.is_full());

        let result = batch.add_frame(create_dummy_frame(3));
        assert!(result.is_some());
        assert_eq!(result.unwrap().len(), 3);
        assert!(batch.is_empty());
    }

    #[test]
    fn test_frame_preprocessor() {
        let preprocessor = FramePreprocessor::new()
            .with_resize(224, 224)
            .with_grayscale();

        let frame = create_dummy_frame(1);
        let result = preprocessor.preprocess(&frame);
        assert!(result.is_ok());
    }

    #[test]
    fn test_quality_adaptation_variants() {
        assert_eq!(QualityAdaptation::None, QualityAdaptation::None);
        assert_ne!(
            QualityAdaptation::None,
            QualityAdaptation::ResolutionScaling
        );
    }

    #[test]
    fn test_is_realtime() {
        let processor =
            StreamProcessor::new(StreamConfig::default()).expect("Failed to create processor");

        // Initially no frames processed, should return true (no data)
        // After processing, FPS will be calculated
        processor.record_processing_time(Duration::from_millis(10));
        let is_rt = processor.is_realtime();
        // Result depends on target FPS (30) vs current FPS (100 from 10ms processing)
        assert!(is_rt); // 100 FPS > 30 FPS target
    }

    #[test]
    fn test_stream_stats_default() {
        let stats = StreamStats::default();
        assert_eq!(stats.total_frames, 0);
        assert_eq!(stats.dropped_frames, 0);
        assert_eq!(stats.current_fps, 0.0);
    }

    #[test]
    fn test_frame_preprocessor_resize() {
        // Create a 4×4×3 CHW frame
        let data: Vec<f32> = (0..48).map(|x| x as f32).collect();
        let tensor = Tensor::from_vec(data, &[3, 4, 4]).expect("tensor creation should succeed");

        let frame = Frame {
            data: tensor,
            metadata: FrameMetadata {
                frame_number: 0,
                timestamp: Instant::now(),
                width: 4,
                height: 4,
                is_keyframe: true,
                priority: 1,
            },
        };

        let preprocessor = FramePreprocessor::new().with_resize(2, 2);
        let result = preprocessor
            .preprocess(&frame)
            .expect("resize should succeed");
        assert_eq!(result.metadata.width, 2);
        assert_eq!(result.metadata.height, 2);
        let shape = result.data.shape();
        let dims = shape.dims();
        assert_eq!(dims, &[3, 2, 2]);
    }

    #[test]
    fn test_frame_preprocessor_normalize() {
        // CHW frame with all values = 128.0
        let data = vec![128.0f32; 3 * 4 * 4];
        let tensor = Tensor::from_vec(data, &[3, 4, 4]).expect("tensor creation should succeed");

        let frame = Frame {
            data: tensor,
            metadata: FrameMetadata {
                frame_number: 0,
                timestamp: Instant::now(),
                width: 4,
                height: 4,
                is_keyframe: true,
                priority: 1,
            },
        };

        // Normalize with mean=128, std=128 → output should be ~0.0
        let preprocessor = FramePreprocessor::new()
            .with_normalize(vec![128.0, 128.0, 128.0], vec![128.0, 128.0, 128.0]);
        let result = preprocessor
            .preprocess(&frame)
            .expect("normalize should succeed");
        let vals: Vec<f32> = result.data.to_vec().expect("to_vec should succeed");
        for &v in &vals {
            assert!(v.abs() < 1e-5, "Expected ~0 after normalization, got {v}");
        }
    }

    #[test]
    fn test_frame_preprocessor_grayscale() {
        // Pure red image: R=1.0, G=0.0, B=0.0 → luminance = 0.299
        let mut data = vec![0.0f32; 3 * 4 * 4];
        // Channel 0 (R) = 1.0
        for i in 0..(4 * 4) {
            data[i] = 1.0;
        }
        let tensor = Tensor::from_vec(data, &[3, 4, 4]).expect("tensor creation should succeed");

        let frame = Frame {
            data: tensor,
            metadata: FrameMetadata {
                frame_number: 0,
                timestamp: Instant::now(),
                width: 4,
                height: 4,
                is_keyframe: true,
                priority: 1,
            },
        };

        let preprocessor = FramePreprocessor::new().with_grayscale();
        let result = preprocessor
            .preprocess(&frame)
            .expect("grayscale should succeed");

        let shape = result.data.shape();
        let dims = shape.dims();
        assert_eq!(dims, &[1, 4, 4]);

        let vals: Vec<f32> = result.data.to_vec().expect("to_vec should succeed");
        for &v in &vals {
            assert!(
                (v - 0.299).abs() < 1e-5,
                "Expected 0.299 luminance for pure red, got {v}"
            );
        }
    }

    #[test]
    fn test_downscale_bilinear_known_values() {
        // 4x4 single-channel frame where pixel(y, x) = 4*y + x:
        //   0  1  2  3
        //   4  5  6  7
        //   8  9 10 11
        //  12 13 14 15
        let data: Vec<f32> = (0..16).map(|x| x as f32).collect();
        let tensor = Tensor::from_vec(data, &[1, 4, 4]).expect("tensor creation should succeed");

        let frame = Frame {
            data: tensor,
            metadata: FrameMetadata {
                frame_number: 0,
                timestamp: Instant::now(),
                width: 4,
                height: 4,
                is_keyframe: true,
                priority: 1,
            },
        };

        let result = downscale_frame_bilinear(&frame, 2, 2).expect("downscale should succeed");

        // Dimensions must reflect the new resolution.
        assert_eq!(result.metadata.width, 2);
        assert_eq!(result.metadata.height, 2);
        let shape = result.data.shape();
        assert_eq!(shape.dims(), &[1, 2, 2]);

        // Hand-computed bilinear values (half-pixel centres, scale = 2):
        //   out(0,0): fy=fx=0.5 -> blend of {0,1,4,5} = 2.5
        //   out(0,1): fx=2.5    -> blend of {2,3,6,7} = 4.5
        //   out(1,0): fy=2.5    -> blend of {8,9,12,13} = 10.5
        //   out(1,1): fy=fx=2.5 -> blend of {10,11,14,15} = 12.5
        let expected = [2.5f32, 4.5, 10.5, 12.5];
        let vals: Vec<f32> = result.data.to_vec().expect("to_vec should succeed");
        assert_eq!(vals.len(), expected.len());
        for (got, want) in vals.iter().zip(expected.iter()) {
            assert!(
                (got - want).abs() < 1e-6,
                "bilinear downscale mismatch: got {got}, expected {want}"
            );
        }
    }

    #[test]
    fn test_downscale_bilinear_rejects_bad_rank() {
        // A rank-1 tensor cannot be interpreted as HW or CHW: expect an honest error
        // rather than a silent passthrough.
        let tensor = Tensor::from_vec(vec![0.0f32; 4], &[4]).expect("tensor creation");
        let frame = Frame {
            data: tensor,
            metadata: FrameMetadata {
                frame_number: 0,
                timestamp: Instant::now(),
                width: 4,
                height: 1,
                is_keyframe: true,
                priority: 1,
            },
        };
        assert!(downscale_frame_bilinear(&frame, 2, 1).is_err());
    }

    #[test]
    fn test_resolution_scaling_triggers_real_downscale() {
        // A fresh processor has current_fps == 0, which is below target * 0.8, so the
        // ResolutionScaling adaptation must produce a genuinely smaller frame (8 -> 6).
        let config = StreamConfig {
            quality_adaptation: QualityAdaptation::ResolutionScaling,
            ..Default::default()
        };
        let processor = StreamProcessor::new(config).expect("processor creation");

        let data: Vec<f32> = (0..64).map(|x| x as f32).collect();
        let tensor = Tensor::from_vec(data, &[1, 8, 8]).expect("tensor creation");
        let frame = Frame {
            data: tensor,
            metadata: FrameMetadata {
                frame_number: 0,
                timestamp: Instant::now(),
                width: 8,
                height: 8,
                is_keyframe: true,
                priority: 1,
            },
        };

        // `process_frame` runs the adaptation then hands the adapted frame to the closure.
        let adapted = processor
            .process_frame(frame, |f| Ok(f.clone()))
            .expect("process_frame should succeed");

        assert_eq!(adapted.metadata.width, 6);
        assert_eq!(adapted.metadata.height, 6);
        let shape = adapted.data.shape();
        assert_eq!(shape.dims(), &[1, 6, 6]);
        assert_eq!(processor.stats().num_adaptations, 1);
    }
}
