//! Interactive visualization tools for torsh-vision
//!
//! This module provides interactive visualization capabilities for exploring images,
//! annotations, and computer vision results. It includes tools for interactive
//! exploration, annotation editing, and real-time visualization.

use crate::{Result, VisionError};
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use torsh_core::dtype::DType;
use torsh_core::{Device, DeviceType};
use torsh_tensor::Tensor;

/// Interactive image viewer with annotation support
#[derive(Clone)]
pub struct InteractiveViewer {
    /// Current image being displayed
    current_image: Option<Tensor>,
    /// Annotations overlaid on the image
    annotations: Vec<Annotation>,
    /// Viewer configuration
    config: ViewerConfig,
    /// Event handlers
    event_handlers: Arc<Mutex<HashMap<String, Box<dyn Fn(&ViewerEvent) + Send + Sync>>>>,
}

impl std::fmt::Debug for InteractiveViewer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InteractiveViewer")
            .field("current_image", &self.current_image)
            .field("annotations", &self.annotations)
            .field("config", &self.config)
            .field("event_handlers", &"<event_handlers>")
            .finish()
    }
}

/// Configuration for the interactive viewer
#[derive(Debug, Clone)]
pub struct ViewerConfig {
    /// Window width
    pub width: u32,
    /// Window height  
    pub height: u32,
    /// Whether to show zoom controls
    pub show_zoom_controls: bool,
    /// Whether to show annotation tools
    pub show_annotation_tools: bool,
    /// Default annotation color
    pub default_annotation_color: [u8; 3],
    /// Background color
    pub background_color: [u8; 3],
}

impl Default for ViewerConfig {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            show_zoom_controls: true,
            show_annotation_tools: true,
            default_annotation_color: [255, 0, 0], // Red
            background_color: [240, 240, 240],     // Light gray
        }
    }
}

/// Annotation types for interactive visualization
#[derive(Debug, Clone)]
pub enum Annotation {
    /// Bounding box annotation
    BoundingBox {
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        label: String,
        color: [u8; 3],
        confidence: Option<f32>,
    },
    /// Point annotation
    Point {
        x: f32,
        y: f32,
        label: String,
        color: [u8; 3],
        radius: f32,
    },
    /// Polygon annotation
    Polygon {
        points: Vec<(f32, f32)>,
        label: String,
        color: [u8; 3],
        filled: bool,
    },
    /// Text annotation
    Text {
        x: f32,
        y: f32,
        text: String,
        color: [u8; 3],
        font_size: f32,
    },
    /// Segmentation mask
    Mask {
        mask: Tensor,
        color: [u8; 3],
        alpha: f32,
        label: String,
    },
}

/// Events that can occur in the interactive viewer
#[derive(Debug, Clone)]
pub enum ViewerEvent {
    /// Mouse click event
    MouseClick { x: f32, y: f32, button: MouseButton },
    /// Mouse move event
    MouseMove { x: f32, y: f32 },
    /// Key press event
    KeyPress { key: String },
    /// Annotation created
    AnnotationCreated { annotation: Annotation },
    /// Annotation selected
    AnnotationSelected { index: usize },
    /// Annotation modified
    AnnotationModified {
        index: usize,
        annotation: Annotation,
    },
    /// Annotation deleted
    AnnotationDeleted { index: usize },
    /// Zoom changed
    ZoomChanged { zoom_level: f32 },
    /// Image changed
    ImageChanged { image: Tensor },
}

/// Mouse button types
#[derive(Debug, Clone, PartialEq)]
pub enum MouseButton {
    Left,
    Right,
    Middle,
}

impl InteractiveViewer {
    /// Create a new interactive viewer
    pub fn new() -> Self {
        Self {
            current_image: None,
            annotations: Vec::new(),
            config: ViewerConfig::default(),
            event_handlers: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Create a new interactive viewer with custom configuration
    pub fn with_config(config: ViewerConfig) -> Self {
        Self {
            current_image: None,
            annotations: Vec::new(),
            config,
            event_handlers: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Load an image for viewing
    pub fn load_image(&mut self, image: Tensor) -> Result<()> {
        // Validate image tensor
        if image.ndim() < 2 || image.ndim() > 3 {
            return Err(VisionError::InvalidInput(
                "Image must be 2D (grayscale) or 3D (color)".to_string(),
            ));
        }

        self.current_image = Some(image.clone());

        // Trigger image changed event
        let event = ViewerEvent::ImageChanged { image };
        self.emit_event(event);

        Ok(())
    }

    /// Load an image from file path
    pub fn load_image_from_path<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        use crate::io::{global_io, VisionIO};
        use crate::utils::image_to_tensor;
        let dynamic_image = global_io().load_image(path)?;
        let tensor = image_to_tensor(&dynamic_image)?;
        self.load_image(tensor)
    }

    /// Add an annotation to the viewer
    pub fn add_annotation(&mut self, annotation: Annotation) {
        self.annotations.push(annotation.clone());

        // Trigger annotation created event
        let event = ViewerEvent::AnnotationCreated { annotation };
        self.emit_event(event);
    }

    /// Remove an annotation by index
    pub fn remove_annotation(&mut self, index: usize) -> Result<()> {
        if index >= self.annotations.len() {
            return Err(VisionError::InvalidArgument(format!(
                "Annotation index {} out of bounds",
                index
            )));
        }

        self.annotations.remove(index);

        // Trigger annotation deleted event
        let event = ViewerEvent::AnnotationDeleted { index };
        self.emit_event(event);

        Ok(())
    }

    /// Update an annotation at the given index
    pub fn update_annotation(&mut self, index: usize, annotation: Annotation) -> Result<()> {
        if index >= self.annotations.len() {
            return Err(VisionError::InvalidArgument(format!(
                "Annotation index {} out of bounds",
                index
            )));
        }

        self.annotations[index] = annotation.clone();

        // Trigger annotation modified event
        let event = ViewerEvent::AnnotationModified { index, annotation };
        self.emit_event(event);

        Ok(())
    }

    /// Get all annotations
    pub fn annotations(&self) -> &[Annotation] {
        &self.annotations
    }

    /// Clear all annotations
    pub fn clear_annotations(&mut self) {
        self.annotations.clear();
    }

    /// Get current image
    pub fn current_image(&self) -> Option<&Tensor> {
        self.current_image.as_ref()
    }

    /// Register an event handler
    pub fn on_event<F>(&mut self, event_name: String, handler: F)
    where
        F: Fn(&ViewerEvent) + Send + Sync + 'static,
    {
        let mut handlers = self.event_handlers.lock().unwrap();
        handlers.insert(event_name, Box::new(handler));
    }

    /// Emit an event to all registered handlers
    fn emit_event(&self, event: ViewerEvent) {
        let handlers = self.event_handlers.lock().unwrap();
        for handler in handlers.values() {
            handler(&event);
        }
    }

    /// Handle mouse click
    pub fn handle_mouse_click(&mut self, x: f32, y: f32, button: MouseButton) {
        let event = ViewerEvent::MouseClick { x, y, button };
        self.emit_event(event);
    }

    /// Handle mouse move
    pub fn handle_mouse_move(&mut self, x: f32, y: f32) {
        let event = ViewerEvent::MouseMove { x, y };
        self.emit_event(event);
    }

    /// Handle key press
    pub fn handle_key_press(&mut self, key: String) {
        let event = ViewerEvent::KeyPress { key };
        self.emit_event(event);
    }

    /// Export annotations to JSON format
    pub fn export_annotations(&self) -> Result<String> {
        use serde_json::json;

        let annotations_json: Vec<serde_json::Value> = self
            .annotations
            .iter()
            .map(|ann| match ann {
                Annotation::BoundingBox {
                    x,
                    y,
                    width,
                    height,
                    label,
                    color,
                    confidence,
                } => {
                    json!({
                        "type": "bounding_box",
                        "x": x,
                        "y": y,
                        "width": width,
                        "height": height,
                        "label": label,
                        "color": color,
                        "confidence": confidence
                    })
                }
                Annotation::Point {
                    x,
                    y,
                    label,
                    color,
                    radius,
                } => {
                    json!({
                        "type": "point",
                        "x": x,
                        "y": y,
                        "label": label,
                        "color": color,
                        "radius": radius
                    })
                }
                Annotation::Polygon {
                    points,
                    label,
                    color,
                    filled,
                } => {
                    json!({
                        "type": "polygon",
                        "points": points,
                        "label": label,
                        "color": color,
                        "filled": filled
                    })
                }
                Annotation::Text {
                    x,
                    y,
                    text,
                    color,
                    font_size,
                } => {
                    json!({
                        "type": "text",
                        "x": x,
                        "y": y,
                        "text": text,
                        "color": color,
                        "font_size": font_size
                    })
                }
                Annotation::Mask {
                    label,
                    color,
                    alpha,
                    ..
                } => {
                    json!({
                        "type": "mask",
                        "label": label,
                        "color": color,
                        "alpha": alpha
                    })
                }
            })
            .collect();

        let export = json!({
            "annotations": annotations_json,
            "config": {
                "width": self.config.width,
                "height": self.config.height
            }
        });

        Ok(serde_json::to_string_pretty(&export).map_err(|e| {
            VisionError::InvalidArgument(format!("JSON serialization error: {}", e))
        })?)
    }

    /// Import annotations from JSON format
    pub fn import_annotations(&mut self, json_str: &str) -> Result<()> {
        let data: serde_json::Value = serde_json::from_str(json_str)
            .map_err(|e| VisionError::InvalidArgument(format!("JSON parsing error: {}", e)))?;

        if let Some(annotations) = data["annotations"].as_array() {
            self.annotations.clear();

            for ann_json in annotations {
                let annotation = self.parse_annotation_from_json(ann_json)?;
                self.annotations.push(annotation);
            }
        }

        Ok(())
    }

    /// Parse a single annotation from JSON
    fn parse_annotation_from_json(&self, json: &serde_json::Value) -> Result<Annotation> {
        let ann_type = json["type"]
            .as_str()
            .ok_or_else(|| VisionError::InvalidInput("Missing annotation type".to_string()))?;

        match ann_type {
            "bounding_box" => Ok(Annotation::BoundingBox {
                x: json["x"].as_f64().unwrap_or(0.0) as f32,
                y: json["y"].as_f64().unwrap_or(0.0) as f32,
                width: json["width"].as_f64().unwrap_or(0.0) as f32,
                height: json["height"].as_f64().unwrap_or(0.0) as f32,
                label: json["label"].as_str().unwrap_or("").to_string(),
                color: [
                    json["color"][0].as_u64().unwrap_or(255) as u8,
                    json["color"][1].as_u64().unwrap_or(0) as u8,
                    json["color"][2].as_u64().unwrap_or(0) as u8,
                ],
                confidence: json["confidence"].as_f64().map(|v| v as f32),
            }),
            "point" => Ok(Annotation::Point {
                x: json["x"].as_f64().unwrap_or(0.0) as f32,
                y: json["y"].as_f64().unwrap_or(0.0) as f32,
                label: json["label"].as_str().unwrap_or("").to_string(),
                color: [
                    json["color"][0].as_u64().unwrap_or(255) as u8,
                    json["color"][1].as_u64().unwrap_or(0) as u8,
                    json["color"][2].as_u64().unwrap_or(0) as u8,
                ],
                radius: json["radius"].as_f64().unwrap_or(3.0) as f32,
            }),
            "polygon" => {
                let points = json["points"]
                    .as_array()
                    .ok_or_else(|| VisionError::InvalidInput("Missing polygon points".to_string()))?
                    .iter()
                    .map(|p| {
                        (
                            p[0].as_f64().unwrap_or(0.0) as f32,
                            p[1].as_f64().unwrap_or(0.0) as f32,
                        )
                    })
                    .collect();

                Ok(Annotation::Polygon {
                    points,
                    label: json["label"].as_str().unwrap_or("").to_string(),
                    color: [
                        json["color"][0].as_u64().unwrap_or(255) as u8,
                        json["color"][1].as_u64().unwrap_or(0) as u8,
                        json["color"][2].as_u64().unwrap_or(0) as u8,
                    ],
                    filled: json["filled"].as_bool().unwrap_or(false),
                })
            }
            "text" => Ok(Annotation::Text {
                x: json["x"].as_f64().unwrap_or(0.0) as f32,
                y: json["y"].as_f64().unwrap_or(0.0) as f32,
                text: json["text"].as_str().unwrap_or("").to_string(),
                color: [
                    json["color"][0].as_u64().unwrap_or(255) as u8,
                    json["color"][1].as_u64().unwrap_or(0) as u8,
                    json["color"][2].as_u64().unwrap_or(0) as u8,
                ],
                font_size: json["font_size"].as_f64().unwrap_or(12.0) as f32,
            }),
            _ => Err(VisionError::InvalidInput(format!(
                "Unknown annotation type: {}",
                ann_type
            ))),
        }
    }
}

/// Interactive image gallery for browsing multiple images
#[derive(Debug)]
pub struct InteractiveGallery {
    /// Images in the gallery
    images: Vec<(String, Tensor)>, // (name, tensor)
    /// Current image index
    current_index: usize,
    /// Gallery configuration
    config: GalleryConfig,
    /// Per-image annotations
    annotations: HashMap<String, Vec<Annotation>>,
}

/// Configuration for the interactive gallery
#[derive(Debug, Clone)]
pub struct GalleryConfig {
    /// Thumbnail size
    pub thumbnail_size: (u32, u32),
    /// Images per row in grid view
    pub images_per_row: usize,
    /// Whether to show image names
    pub show_names: bool,
    /// Whether to show navigation controls
    pub show_navigation: bool,
}

impl Default for GalleryConfig {
    fn default() -> Self {
        Self {
            thumbnail_size: (150, 150),
            images_per_row: 4,
            show_names: true,
            show_navigation: true,
        }
    }
}

impl InteractiveGallery {
    /// Create a new interactive gallery
    pub fn new() -> Self {
        Self {
            images: Vec::new(),
            current_index: 0,
            config: GalleryConfig::default(),
            annotations: HashMap::new(),
        }
    }

    /// Create a new gallery with custom configuration
    pub fn with_config(config: GalleryConfig) -> Self {
        Self {
            images: Vec::new(),
            current_index: 0,
            config,
            annotations: HashMap::new(),
        }
    }

    /// Add an image to the gallery
    pub fn add_image(&mut self, name: String, image: Tensor) -> Result<()> {
        // Validate image tensor
        if image.ndim() < 2 || image.ndim() > 3 {
            return Err(VisionError::InvalidInput(
                "Image must be 2D (grayscale) or 3D (color)".to_string(),
            ));
        }

        self.images.push((name.clone(), image));
        self.annotations.insert(name, Vec::new());
        Ok(())
    }

    /// Load images from a directory
    pub fn load_from_directory<P: AsRef<Path>>(&mut self, dir_path: P) -> Result<()> {
        use crate::io::{global_io, VisionIO};
        use crate::utils::image_to_tensor;
        use std::fs;

        let dir = fs::read_dir(dir_path)?;

        for entry in dir {
            let entry = entry?;
            let path = entry.path();

            if let Some(extension) = path.extension() {
                let ext_str = extension.to_str().unwrap_or("").to_lowercase();
                if matches!(
                    ext_str.as_str(),
                    "jpg" | "jpeg" | "png" | "bmp" | "tiff" | "webp"
                ) {
                    let name = path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("unknown")
                        .to_string();

                    let dynamic_image = global_io().load_image(&path)?;
                    let tensor = image_to_tensor(&dynamic_image)?;
                    self.add_image(name, tensor)?;
                }
            }
        }

        Ok(())
    }

    /// Get current image
    pub fn current_image(&self) -> Option<&(String, Tensor)> {
        self.images.get(self.current_index)
    }

    /// Navigate to next image
    pub fn next_image(&mut self) -> Result<()> {
        if self.images.is_empty() {
            return Err(VisionError::InvalidInput(
                "No images in gallery".to_string(),
            ));
        }

        self.current_index = (self.current_index + 1) % self.images.len();
        Ok(())
    }

    /// Navigate to previous image
    pub fn previous_image(&mut self) -> Result<()> {
        if self.images.is_empty() {
            return Err(VisionError::InvalidInput(
                "No images in gallery".to_string(),
            ));
        }

        if self.current_index == 0 {
            self.current_index = self.images.len() - 1;
        } else {
            self.current_index -= 1;
        }
        Ok(())
    }

    /// Navigate to specific image by index
    pub fn goto_image(&mut self, index: usize) -> Result<()> {
        if index >= self.images.len() {
            return Err(VisionError::InvalidArgument(format!(
                "Image index {} out of bounds",
                index
            )));
        }

        self.current_index = index;
        Ok(())
    }

    /// Get all image names
    pub fn image_names(&self) -> Vec<&String> {
        self.images.iter().map(|(name, _)| name).collect()
    }

    /// Get number of images
    pub fn len(&self) -> usize {
        self.images.len()
    }

    /// Check if gallery is empty
    pub fn is_empty(&self) -> bool {
        self.images.is_empty()
    }

    /// Add annotation to current image
    pub fn add_annotation_to_current(&mut self, annotation: Annotation) -> Result<()> {
        let current_name = self
            .current_image()
            .ok_or_else(|| VisionError::InvalidInput("No current image".to_string()))?
            .0
            .clone();

        self.annotations
            .entry(current_name)
            .or_default()
            .push(annotation);
        Ok(())
    }

    /// Get annotations for current image
    pub fn current_annotations(&self) -> Option<&Vec<Annotation>> {
        let current_name = &self.current_image()?.0;
        self.annotations.get(current_name)
    }

    /// Clear annotations for current image
    pub fn clear_current_annotations(&mut self) -> Result<()> {
        let current_name = self
            .current_image()
            .ok_or_else(|| VisionError::InvalidInput("No current image".to_string()))?
            .0
            .clone();

        self.annotations.insert(current_name, Vec::new());
        Ok(())
    }
}

/// Live visualization for real-time computer vision pipelines
#[derive(Debug)]
pub struct LiveVisualization {
    /// Current frame being displayed
    current_frame: Option<Tensor>,
    /// Frame buffer for smooth playback
    frame_buffer: std::collections::VecDeque<Tensor>,
    /// Buffer size limit
    buffer_size: usize,
    /// FPS counter
    fps_counter: FpsCounter,
    /// Live visualization config
    config: LiveConfig,
}

/// Configuration for live visualization
#[derive(Debug, Clone)]
pub struct LiveConfig {
    /// Target FPS for display
    pub target_fps: f32,
    /// Buffer size for frames
    pub buffer_size: usize,
    /// Whether to show FPS counter
    pub show_fps: bool,
    /// Whether to show performance metrics
    pub show_metrics: bool,
}

impl Default for LiveConfig {
    fn default() -> Self {
        Self {
            target_fps: 30.0,
            buffer_size: 10,
            show_fps: true,
            show_metrics: false,
        }
    }
}

/// FPS counter for performance monitoring
#[derive(Debug)]
pub struct FpsCounter {
    frame_times: std::collections::VecDeque<std::time::Instant>,
    window_size: usize,
}

impl FpsCounter {
    fn new(window_size: usize) -> Self {
        Self {
            frame_times: std::collections::VecDeque::new(),
            window_size,
        }
    }

    fn update(&mut self) {
        let now = std::time::Instant::now();
        self.frame_times.push_back(now);

        if self.frame_times.len() > self.window_size {
            self.frame_times.pop_front();
        }
    }

    fn current_fps(&self) -> f32 {
        if self.frame_times.len() < 2 {
            return 0.0;
        }

        let elapsed = self
            .frame_times
            .back()
            .unwrap()
            .duration_since(*self.frame_times.front().unwrap());

        let num_frames = self.frame_times.len() - 1;
        num_frames as f32 / elapsed.as_secs_f32()
    }
}

impl LiveVisualization {
    /// Create a new live visualization
    pub fn new() -> Self {
        Self::with_config(LiveConfig::default())
    }

    /// Create a new live visualization with custom configuration
    pub fn with_config(config: LiveConfig) -> Self {
        Self {
            current_frame: None,
            frame_buffer: std::collections::VecDeque::with_capacity(config.buffer_size),
            buffer_size: config.buffer_size,
            fps_counter: FpsCounter::new(30),
            config,
        }
    }

    /// Add a new frame to the visualization
    pub fn add_frame(&mut self, frame: Tensor) -> Result<()> {
        // Validate frame tensor
        if frame.ndim() < 2 || frame.ndim() > 3 {
            return Err(VisionError::InvalidInput(
                "Frame must be 2D (grayscale) or 3D (color)".to_string(),
            ));
        }

        // Add to buffer
        if self.frame_buffer.len() >= self.buffer_size {
            self.frame_buffer.pop_front();
        }
        self.frame_buffer.push_back(frame.clone());

        // Update current frame
        self.current_frame = Some(frame);

        // Update FPS counter
        self.fps_counter.update();

        Ok(())
    }

    /// Get current frame
    pub fn current_frame(&self) -> Option<&Tensor> {
        self.current_frame.as_ref()
    }

    /// Get current FPS
    pub fn current_fps(&self) -> f32 {
        self.fps_counter.current_fps()
    }

    /// Get frame buffer size
    pub fn buffer_len(&self) -> usize {
        self.frame_buffer.len()
    }

    /// Clear frame buffer
    pub fn clear_buffer(&mut self) {
        self.frame_buffer.clear();
        self.current_frame = None;
    }
}

impl Default for InteractiveViewer {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for InteractiveGallery {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for LiveVisualization {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::{DType, Device};

    #[test]
    fn test_interactive_viewer_creation() {
        let viewer = InteractiveViewer::new();
        assert!(viewer.current_image().is_none());
        assert_eq!(viewer.annotations().len(), 0);
    }

    #[test]
    fn test_interactive_viewer_load_image() {
        let mut viewer = InteractiveViewer::new();
        let image = Tensor::zeros(&[3, 224, 224], DeviceType::Cpu).unwrap();

        viewer.load_image(image).unwrap();
        assert!(viewer.current_image().is_some());
    }

    #[test]
    fn test_annotation_management() {
        let mut viewer = InteractiveViewer::new();

        let annotation = Annotation::BoundingBox {
            x: 10.0,
            y: 20.0,
            width: 50.0,
            height: 30.0,
            label: "test".to_string(),
            color: [255, 0, 0],
            confidence: Some(0.95),
        };

        viewer.add_annotation(annotation);
        assert_eq!(viewer.annotations().len(), 1);

        viewer.remove_annotation(0).unwrap();
        assert_eq!(viewer.annotations().len(), 0);
    }

    #[test]
    fn test_interactive_gallery() {
        let mut gallery = InteractiveGallery::new();

        let image1 = Tensor::zeros(&[3, 224, 224], DeviceType::Cpu).unwrap();
        let image2 = Tensor::ones(&[3, 224, 224], DeviceType::Cpu).unwrap();

        gallery.add_image("image1".to_string(), image1).unwrap();
        gallery.add_image("image2".to_string(), image2).unwrap();

        assert_eq!(gallery.len(), 2);
        assert!(!gallery.is_empty());

        gallery.next_image().unwrap();
        let (current_name, _) = gallery.current_image().unwrap();
        assert_eq!(current_name, "image2");
    }

    #[test]
    fn test_live_visualization() {
        let mut live_viz = LiveVisualization::new();

        let frame = Tensor::zeros(&[3, 480, 640], DeviceType::Cpu).unwrap();
        live_viz.add_frame(frame).unwrap();

        assert!(live_viz.current_frame().is_some());
        assert_eq!(live_viz.buffer_len(), 1);
    }

    #[test]
    fn test_annotation_export_import() {
        let mut viewer = InteractiveViewer::new();

        let annotation = Annotation::Point {
            x: 100.0,
            y: 200.0,
            label: "landmark".to_string(),
            color: [0, 255, 0],
            radius: 5.0,
        };

        viewer.add_annotation(annotation);

        let exported = viewer.export_annotations().unwrap();
        assert!(exported.contains("landmark"));

        let mut new_viewer = InteractiveViewer::new();
        new_viewer.import_annotations(&exported).unwrap();
        assert_eq!(new_viewer.annotations().len(), 1);
    }
}
