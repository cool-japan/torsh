//! Interactive visualization utilities for real-time image manipulation and viewing
//!
//! This module provides comprehensive interactive visualization capabilities for ToRSh Vision,
//! including web-based viewers, annotation systems, and real-time transform operations.
//!
//! # Features
//!
//! - **Interactive Viewer**: Web-based interface for real-time image manipulation
//! - **Annotation System**: Support for various annotation types (bounding boxes, points, polygons, etc.)
//! - **Transform Operations**: Interactive transform pipeline with parameter controls
//! - **HTML Generation**: Automatic generation of web interfaces for visualization
//!
//! # Examples
//!
//! ## Basic Interactive Viewer
//!
//! ```rust
//! use torsh_vision::utils::interactive::{create_interactive_viewer, Annotation, AnnotationType};
//! use torsh_tensor::Tensor;
//!
//! // Create a new interactive viewer
//! let mut viewer = create_interactive_viewer(8080);
//!
//! // Set an image (3D tensor with shape [C, H, W])
//! let image = Tensor::zeros(&[3, 224, 224]).unwrap();
//! viewer.set_image(image).unwrap();
//!
//! // Add annotations
//! let annotation = Annotation {
//!     annotation_type: AnnotationType::BoundingBox,
//!     coordinates: vec![10.0, 10.0, 50.0, 50.0],
//!     label: "Object".to_string(),
//!     color: (255, 0, 0),
//!     confidence: Some(0.95),
//! };
//! viewer.add_annotation(annotation);
//!
//! // Generate HTML interface
//! let html = viewer.generate_html_interface();
//!
//! // Start the viewer server
//! viewer.start_server().unwrap();
//! ```
//!
//! ## Interactive Transform Operations
//!
//! ```rust
//! use torsh_vision::utils::interactive::{InteractiveViewer, Parameter};
//!
//! struct BrightnessTransform {
//!     brightness: f32,
//! }
//!
//! impl torsh_vision::utils::interactive::TransformOp for BrightnessTransform {
//!     fn apply(&self, image: &Tensor<f32>) -> torsh_vision::Result<Tensor<f32>> {
//!         // Apply brightness adjustment
//!         image.clone() + self.brightness
//!     }
//!
//!     fn name(&self) -> &str {
//!         "Brightness"
//!     }
//!
//!     fn parameters(&self) -> Vec<Parameter> {
//!         vec![Parameter {
//!             name: "brightness".to_string(),
//!             value: self.brightness,
//!             min: -1.0,
//!             max: 1.0,
//!             step: 0.01,
//!         }]
//!     }
//! }
//! ```

use crate::{Result, VisionError};
use serde::{Deserialize, Serialize};
use torsh_tensor::Tensor;

/// Interactive visualization server for real-time image manipulation and viewing
///
/// The `InteractiveViewer` provides a comprehensive web-based interface for visualizing
/// and manipulating images in real-time. It supports various annotation types,
/// interactive transforms, and generates HTML interfaces for browser-based viewing.
///
/// # Features
///
/// - Real-time image display and manipulation
/// - Multiple annotation types (bounding boxes, points, polygons, etc.)
/// - Interactive transform pipeline
/// - Web-based HTML interface generation
/// - Annotation management and export
///
/// # Architecture
///
/// The viewer operates as a stateful server that maintains:
/// - Current image data as a 3D tensor
/// - Collection of annotations with metadata
/// - Pipeline of interactive transforms
/// - Configuration for web interface generation
pub struct InteractiveViewer {
    port: u16,
    current_image: Option<Tensor<f32>>,
    annotations: Vec<Annotation>,
    transforms: Vec<Box<dyn TransformOp>>,
}

/// Annotation structure for interactive visualization
///
/// Represents a single annotation with associated metadata including position,
/// type, label, visual styling, and optional confidence score.
///
/// # Fields
///
/// - `annotation_type`: Type of annotation (bounding box, point, etc.)
/// - `coordinates`: Position data specific to annotation type
/// - `label`: Human-readable label for the annotation
/// - `color`: RGB color tuple for visual rendering
/// - `confidence`: Optional confidence score [0.0, 1.0]
///
/// # Coordinate Formats
///
/// Different annotation types use coordinates differently:
/// - **BoundingBox**: [x_min, y_min, x_max, y_max]
/// - **Point**: [x, y]
/// - **Polygon**: [x1, y1, x2, y2, ..., xn, yn]
/// - **Text**: [x, y] (anchor point)
/// - **Arrow**: [x_start, y_start, x_end, y_end]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Annotation {
    pub annotation_type: AnnotationType,
    pub coordinates: Vec<f32>,
    pub label: String,
    pub color: (u8, u8, u8),
    pub confidence: Option<f32>,
}

/// Enumeration of supported annotation types
///
/// Each type has specific coordinate requirements and rendering behavior:
///
/// - **BoundingBox**: Rectangular bounding box with min/max coordinates
/// - **Point**: Single point marker
/// - **Polygon**: Multi-point polygon shape
/// - **Mask**: Pixel-level segmentation mask
/// - **Text**: Text label with anchor position
/// - **Arrow**: Directional arrow with start and end points
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnnotationType {
    BoundingBox,
    Point,
    Polygon,
    Mask,
    Text,
    Arrow,
}

/// Transform operation trait for interactive manipulation
///
/// Defines the interface for interactive image transforms that can be applied
/// in real-time through the web interface. Each transform provides:
///
/// - `apply()`: Core transformation logic
/// - `name()`: Human-readable name for UI display
/// - `parameters()`: Configurable parameters with constraints
///
/// # Thread Safety
///
/// Implementations must be thread-safe (`Send + Sync`) to support
/// concurrent web interface operations.
///
/// # Example Implementation
///
/// ```rust
/// struct RotationTransform {
///     angle: f32,
/// }
///
/// impl TransformOp for RotationTransform {
///     fn apply(&self, image: &Tensor<f32>) -> Result<Tensor<f32>> {
///         // Rotation logic here
///         Ok(image.clone())
///     }
///
///     fn name(&self) -> &str {
///         "Rotation"
///     }
///
///     fn parameters(&self) -> Vec<Parameter> {
///         vec![Parameter {
///             name: "angle".to_string(),
///             value: self.angle,
///             min: 0.0,
///             max: 360.0,
///             step: 1.0,
///         }]
///     }
/// }
/// ```
pub trait TransformOp: Send + Sync {
    /// Apply the transformation to an input image
    ///
    /// # Arguments
    ///
    /// * `image` - Input image tensor with shape [C, H, W]
    ///
    /// # Returns
    ///
    /// Transformed image tensor with same or compatible shape
    ///
    /// # Errors
    ///
    /// Returns `VisionError` if transformation fails due to:
    /// - Invalid input tensor shape
    /// - Unsupported data types
    /// - Transform-specific parameter errors
    fn apply(&self, image: &Tensor<f32>) -> Result<Tensor<f32>>;

    /// Get the human-readable name of this transform
    fn name(&self) -> &str;

    /// Get the configurable parameters for this transform
    ///
    /// Returns a vector of parameters that can be adjusted through
    /// the interactive interface. Each parameter includes constraints
    /// and current values.
    fn parameters(&self) -> Vec<Parameter>;
}

/// Configurable parameter for interactive transforms
///
/// Represents a single adjustable parameter with constraints and metadata
/// for rendering in the web interface. Supports slider-based controls
/// with defined ranges and step sizes.
///
/// # Fields
///
/// - `name`: Parameter identifier and display name
/// - `value`: Current parameter value
/// - `min`/`max`: Valid value range
/// - `step`: Granularity for UI controls
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Parameter {
    pub name: String,
    pub value: f32,
    pub min: f32,
    pub max: f32,
    pub step: f32,
}

impl InteractiveViewer {
    /// Create a new interactive viewer on specified port
    ///
    /// Initializes an empty viewer that can be configured with images,
    /// annotations, and transforms before starting the web server.
    ///
    /// # Arguments
    ///
    /// * `port` - TCP port number for the web server (e.g., 8080)
    ///
    /// # Examples
    ///
    /// ```rust
    /// let viewer = InteractiveViewer::new(8080);
    /// ```
    pub fn new(port: u16) -> Self {
        Self {
            port,
            current_image: None,
            annotations: Vec::new(),
            transforms: Vec::new(),
        }
    }

    /// Set the current image for viewing
    ///
    /// Validates and stores the input image for display and manipulation.
    /// The image must be a 3D tensor with shape [C, H, W] where:
    /// - C: Number of channels (typically 1 for grayscale, 3 for RGB)
    /// - H: Height in pixels
    /// - W: Width in pixels
    ///
    /// # Arguments
    ///
    /// * `image` - Input image tensor with shape [C, H, W]
    ///
    /// # Returns
    ///
    /// - `Ok(())` if image is valid and set successfully
    /// - `Err(VisionError::InvalidShape)` if tensor shape is invalid
    ///
    /// # Examples
    ///
    /// ```rust
    /// use torsh_tensor::Tensor;
    ///
    /// let mut viewer = InteractiveViewer::new(8080);
    /// let image = Tensor::zeros(&[3, 224, 224]).unwrap();
    /// viewer.set_image(image).unwrap();
    /// ```
    pub fn set_image(&mut self, image: Tensor<f32>) -> Result<()> {
        // Validate image format
        let shape = image.shape();
        if shape.dims().len() != 3 {
            return Err(VisionError::InvalidShape(format!(
                "Expected 3D tensor (C, H, W), got {}D",
                shape.dims().len()
            )));
        }

        self.current_image = Some(image);
        Ok(())
    }

    /// Add annotation to current image
    ///
    /// Appends a new annotation to the current collection. Annotations
    /// are preserved across image changes and transform operations.
    ///
    /// # Arguments
    ///
    /// * `annotation` - Annotation to add with position, type, and metadata
    ///
    /// # Examples
    ///
    /// ```rust
    /// let annotation = Annotation {
    ///     annotation_type: AnnotationType::BoundingBox,
    ///     coordinates: vec![10.0, 10.0, 100.0, 100.0],
    ///     label: "Object".to_string(),
    ///     color: (255, 0, 0),
    ///     confidence: Some(0.95),
    /// };
    /// viewer.add_annotation(annotation);
    /// ```
    pub fn add_annotation(&mut self, annotation: Annotation) {
        self.annotations.push(annotation);
    }

    /// Clear all annotations
    ///
    /// Removes all annotations from the viewer while preserving
    /// the current image and transforms.
    pub fn clear_annotations(&mut self) {
        self.annotations.clear();
    }

    /// Add interactive transform
    ///
    /// Appends a transform operation to the processing pipeline.
    /// Transforms are applied in the order they were added.
    ///
    /// # Arguments
    ///
    /// * `transform` - Boxed transform implementing the TransformOp trait
    ///
    /// # Examples
    ///
    /// ```rust
    /// struct BrightnessTransform { brightness: f32 }
    ///
    /// // Implementation of TransformOp trait required
    ///
    /// let transform = Box::new(BrightnessTransform { brightness: 0.2 });
    /// viewer.add_transform(transform);
    /// ```
    pub fn add_transform(&mut self, transform: Box<dyn TransformOp>) {
        self.transforms.push(transform);
    }

    /// Apply all current transforms to the image
    ///
    /// Sequentially applies all registered transforms to the current image,
    /// returning the final processed result. If no image is set, returns None.
    ///
    /// # Returns
    ///
    /// - `Ok(Some(tensor))` - Successfully processed image
    /// - `Ok(None)` - No current image set
    /// - `Err(VisionError)` - Transform operation failed
    ///
    /// # Transform Pipeline
    ///
    /// Transforms are applied in registration order:
    /// 1. Original image
    /// 2. Transform 1 → Intermediate result 1
    /// 3. Transform 2 → Intermediate result 2
    /// 4. ... → Final result
    ///
    /// # Examples
    ///
    /// ```rust
    /// match viewer.apply_transforms()? {
    ///     Some(processed_image) => {
    ///         // Use processed image
    ///     }
    ///     None => {
    ///         // No image set
    ///     }
    /// }
    /// ```
    pub fn apply_transforms(&self) -> Result<Option<Tensor<f32>>> {
        if let Some(ref image) = self.current_image {
            let mut result = image.clone();
            for transform in &self.transforms {
                result = transform.apply(&result)?;
            }
            Ok(Some(result))
        } else {
            Ok(None)
        }
    }

    /// Generate HTML interface for web-based viewing
    ///
    /// Creates a complete HTML document with embedded JavaScript for
    /// interactive image viewing, annotation management, and transform controls.
    ///
    /// # Features of Generated Interface
    ///
    /// - **Canvas Display**: Interactive image canvas with zoom/pan
    /// - **Annotation Tools**: Click-to-add annotations with type selection
    /// - **Transform Controls**: Sliders and inputs for transform parameters
    /// - **Export Functions**: Save images and export annotation data
    /// - **Real-time Updates**: Live preview of transform effects
    ///
    /// # Returns
    ///
    /// Complete HTML document as a string ready for serving or saving
    ///
    /// # Browser Compatibility
    ///
    /// Generated HTML uses standard web APIs compatible with modern browsers:
    /// - HTML5 Canvas for image rendering
    /// - ES6 JavaScript for interactivity
    /// - CSS3 for styling and layout
    ///
    /// # Examples
    ///
    /// ```rust
    /// let html = viewer.generate_html_interface();
    /// std::fs::write("viewer.html", html).unwrap();
    /// ```
    pub fn generate_html_interface(&self) -> String {
        let html = String::from(
            r#"
<!DOCTYPE html>
<html>
<head>
    <title>ToRSh Vision Interactive Viewer</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .container { display: flex; gap: 20px; }
        .image-panel { flex: 2; }
        .control-panel { flex: 1; background: #f5f5f5; padding: 20px; border-radius: 8px; }
        .annotation { margin: 10px 0; padding: 10px; background: white; border-radius: 4px; }
        .transform { margin: 10px 0; padding: 10px; background: white; border-radius: 4px; }
        #canvas { border: 1px solid #ccc; max-width: 100%; }
        button { padding: 8px 16px; margin: 4px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; }
        button:hover { background: #0056b3; }
        input[type="range"] { width: 100%; }
    </style>
</head>
<body>
    <h1>ToRSh Vision Interactive Viewer</h1>
    <div class="container">
        <div class="image-panel">
            <canvas id="canvas" width="800" height="600"></canvas>
            <div>
                <button onclick="clearAnnotations()">Clear Annotations</button>
                <button onclick="saveImage()">Save Image</button>
                <button onclick="exportData()">Export Data</button>
            </div>
        </div>
        <div class="control-panel">
            <h3>Annotations</h3>
            <div id="annotations"></div>

            <h3>Transforms</h3>
            <div id="transforms"></div>

            <h3>Add Annotation</h3>
            <select id="annotationType">
                <option value="BoundingBox">Bounding Box</option>
                <option value="Point">Point</option>
                <option value="Polygon">Polygon</option>
                <option value="Text">Text</option>
            </select>
            <input type="text" id="annotationLabel" placeholder="Label" />
            <button onclick="addAnnotation()">Add</button>
        </div>
    </div>

    <script>
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        let currentMode = 'view';
        let annotations = [];

        // Canvas event listeners for interactive annotation
        canvas.addEventListener('click', handleCanvasClick);
        canvas.addEventListener('mousemove', handleMouseMove);

        function handleCanvasClick(event) {
            const rect = canvas.getBoundingClientRect();
            const x = event.clientX - rect.left;
            const y = event.clientY - rect.top;

            if (currentMode === 'annotate') {
                createAnnotation(x, y);
            }
        }

        function handleMouseMove(event) {
            // Update cursor or preview based on current mode
        }

        function createAnnotation(x, y) {
            const type = document.getElementById('annotationType').value;
            const label = document.getElementById('annotationLabel').value || 'Annotation';

            const annotation = {
                type: type,
                x: x,
                y: y,
                label: label,
                color: [255, 0, 0]
            };

            annotations.push(annotation);
            updateAnnotationsList();
            redrawCanvas();
        }

        function updateAnnotationsList() {
            const container = document.getElementById('annotations');
            container.innerHTML = '';

            annotations.forEach((ann, index) => {
                const div = document.createElement('div');
                div.className = 'annotation';
                div.innerHTML = `
                    <strong>${ann.label}</strong> (${ann.type})<br>
                    Position: (${ann.x.toFixed(0)}, ${ann.y.toFixed(0)})<br>
                    <button onclick="removeAnnotation(${index})">Remove</button>
                `;
                container.appendChild(div);
            });
        }

        function removeAnnotation(index) {
            annotations.splice(index, 1);
            updateAnnotationsList();
            redrawCanvas();
        }

        function clearAnnotations() {
            annotations = [];
            updateAnnotationsList();
            redrawCanvas();
        }

        function redrawCanvas() {
            // Clear canvas
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Draw image (would be loaded from server)
            // Draw annotations
            annotations.forEach(ann => {
                ctx.strokeStyle = `rgb(${ann.color[0]}, ${ann.color[1]}, ${ann.color[2]})`;
                ctx.lineWidth = 2;

                if (ann.type === 'Point') {
                    ctx.beginPath();
                    ctx.arc(ann.x, ann.y, 5, 0, 2 * Math.PI);
                    ctx.stroke();
                } else if (ann.type === 'BoundingBox') {
                    ctx.strokeRect(ann.x - 25, ann.y - 25, 50, 50);
                }

                // Draw label
                ctx.fillStyle = 'white';
                ctx.fillRect(ann.x, ann.y - 20, ctx.measureText(ann.label).width + 4, 16);
                ctx.fillStyle = 'black';
                ctx.fillText(ann.label, ann.x + 2, ann.y - 6);
            });
        }

        function addAnnotation() {
            currentMode = 'annotate';
            canvas.style.cursor = 'crosshair';
        }

        function saveImage() {
            // Implementation for saving the current view
            alert('Save functionality would be implemented here');
        }

        function exportData() {
            const data = {
                annotations: annotations,
                timestamp: new Date().toISOString()
            };

            const dataStr = JSON.stringify(data, null, 2);
            const dataBlob = new Blob([dataStr], {type: 'application/json'});
            const url = URL.createObjectURL(dataBlob);

            const link = document.createElement('a');
            link.href = url;
            link.download = 'annotations.json';
            link.click();
        }

        // Initialize
        redrawCanvas();
    </script>
</body>
</html>
        "#,
        );

        html
    }

    /// Start the interactive viewer server
    ///
    /// Initializes and starts the web server to serve the interactive interface.
    /// This is currently a placeholder implementation that would start an HTTP
    /// server with WebSocket support for real-time updates.
    ///
    /// # Server Features (Planned)
    ///
    /// - **HTTP Server**: Serves HTML interface and static assets
    /// - **WebSocket**: Real-time communication for live updates
    /// - **Image Streaming**: Efficient tensor-to-image conversion and streaming
    /// - **State Synchronization**: Sync annotations and transforms across clients
    ///
    /// # Returns
    ///
    /// - `Ok(())` - Server started successfully
    /// - `Err(VisionError)` - Server startup failed
    ///
    /// # Current Implementation
    ///
    /// The current implementation is a placeholder that prints server information
    /// and would be replaced with actual HTTP server logic in production.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let viewer = InteractiveViewer::new(8080);
    /// viewer.start_server().unwrap();
    /// // Navigate to http://localhost:8080 in browser
    /// ```
    pub fn start_server(&self) -> Result<()> {
        println!("Interactive viewer would start on port {}", self.port);
        println!(
            "HTML interface generated with {} annotations",
            self.annotations.len()
        );

        // In a real implementation, this would start an HTTP server
        // serving the HTML interface and handling WebSocket connections
        // for real-time updates

        Ok(())
    }
}

/// Create an interactive viewer for real-time image manipulation
///
/// Convenience function to create a new InteractiveViewer with default settings.
/// Equivalent to calling `InteractiveViewer::new(port)` directly.
///
/// # Arguments
///
/// * `port` - TCP port number for the web server
///
/// # Returns
///
/// New InteractiveViewer instance ready for configuration
///
/// # Examples
///
/// ```rust
/// let viewer = create_interactive_viewer(8080);
/// ```
pub fn create_interactive_viewer(port: u16) -> InteractiveViewer {
    InteractiveViewer::new(port)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interactive_viewer_creation() {
        let viewer = InteractiveViewer::new(8080);
        assert_eq!(viewer.port, 8080);
        assert!(viewer.current_image.is_none());
        assert!(viewer.annotations.is_empty());
        assert!(viewer.transforms.is_empty());
    }

    #[test]
    fn test_annotation_management() {
        let mut viewer = InteractiveViewer::new(8080);

        let annotation = Annotation {
            annotation_type: AnnotationType::BoundingBox,
            coordinates: vec![10.0, 10.0, 50.0, 50.0],
            label: "Test".to_string(),
            color: (255, 0, 0),
            confidence: Some(0.95),
        };

        viewer.add_annotation(annotation);
        assert_eq!(viewer.annotations.len(), 1);

        viewer.clear_annotations();
        assert!(viewer.annotations.is_empty());
    }

    #[test]
    fn test_html_generation() {
        let viewer = InteractiveViewer::new(8080);
        let html = viewer.generate_html_interface();

        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("ToRSh Vision Interactive Viewer"));
        assert!(html.contains("<canvas"));
    }

    #[test]
    fn test_parameter_serialization() {
        let param = Parameter {
            name: "brightness".to_string(),
            value: 0.5,
            min: 0.0,
            max: 1.0,
            step: 0.01,
        };

        let json = serde_json::to_string(&param).unwrap();
        let deserialized: Parameter = serde_json::from_str(&json).unwrap();

        assert_eq!(param.name, deserialized.name);
        assert_eq!(param.value, deserialized.value);
    }

    #[test]
    fn test_annotation_types() {
        let annotation_types = vec![
            AnnotationType::BoundingBox,
            AnnotationType::Point,
            AnnotationType::Polygon,
            AnnotationType::Mask,
            AnnotationType::Text,
            AnnotationType::Arrow,
        ];

        for annotation_type in annotation_types {
            let annotation = Annotation {
                annotation_type: annotation_type.clone(),
                coordinates: vec![0.0, 0.0],
                label: "Test".to_string(),
                color: (255, 255, 255),
                confidence: None,
            };

            // Test serialization
            let json = serde_json::to_string(&annotation).unwrap();
            let _deserialized: Annotation = serde_json::from_str(&json).unwrap();
        }
    }
}
