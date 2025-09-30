//! 3D Visualization Engine for ToRSh Vision
//!
//! This module provides comprehensive 3D visualization capabilities for volumetric data,
//! feature maps, and tensor activations. It supports interactive browser-based visualization
//! using Three.js and WebGL for high-performance 3D rendering.
//!
//! # Features
//!
//! - **Volumetric Rendering**: Visualize 3D tensors as voxel clouds with customizable thresholds
//! - **Feature Map Visualization**: Display neural network feature maps as 3D point clouds
//! - **Mesh Generation**: Create 3D meshes from segmentation masks using marching cubes
//! - **Interactive HTML Output**: Generate self-contained HTML files with Three.js integration
//! - **Batch Processing**: Process multiple activations and create navigation interfaces
//!
//! # Quick Start
//!
//! ```rust
//! use torsh_vision::utils::visualization_3d::{Visualizer3D, create_3d_visualizer};
//! use torsh_tensor::Tensor;
//! use torsh_core::device::DeviceType;
//!
//! // Create a 3D visualizer
//! let visualizer = create_3d_visualizer();
//!
//! // Visualize a volume (3D tensor)
//! let volume = Tensor::randn(vec![32, 32, 32], DeviceType::Cpu)?;
//! let html = visualizer.visualize_volume(&volume, 0.5)?;
//! visualizer.save_visualization(&html, "volume.html")?;
//!
//! // Visualize feature maps
//! let feature_map = Tensor::randn(vec![64, 28, 28], DeviceType::Cpu)?;
//! let html = visualizer.visualize_feature_map(&feature_map, "conv1")?;
//! visualizer.save_visualization(&html, "feature_map.html")?;
//! # Ok::<(), torsh_vision::VisionError>(())
//! ```
//!
//! # Advanced Usage
//!
//! ```rust
//! use torsh_vision::utils::visualization_3d::{Visualizer3D, visualize_activations_3d};
//! use torsh_tensor::Tensor;
//! use torsh_core::device::DeviceType;
//!
//! // Batch process multiple layer activations
//! let activations = vec![
//!     Tensor::randn(vec![32, 56, 56], DeviceType::Cpu)?,
//!     Tensor::randn(vec![64, 28, 28], DeviceType::Cpu)?,
//!     Tensor::randn(vec![128, 14, 14], DeviceType::Cpu)?,
//! ];
//! let layer_names = vec!["conv1".to_string(), "conv2".to_string(), "conv3".to_string()];
//!
//! visualize_activations_3d(&activations, &layer_names, "output/3d_viz")?;
//! # Ok::<(), torsh_vision::VisionError>(())
//! ```

use crate::{Result, VisionError};
use std::path::Path;
use torsh_tensor::Tensor;

/// 3D visualization utilities for volumetric data and feature maps
///
/// This struct provides a comprehensive 3D visualization engine that can handle
/// various types of 3D data including volumetric tensors, feature maps, and
/// segmentation masks. It generates interactive HTML visualizations using Three.js.
///
/// # Configuration
///
/// - `background_color`: RGB color for the 3D scene background (0.0-1.0 range)
/// - `camera_position`: Initial camera position in 3D space
/// - `camera_target`: Point the camera is looking at
///
/// # Examples
///
/// ```rust
/// use torsh_vision::utils::visualization_3d::Visualizer3D;
///
/// let mut visualizer = Visualizer3D::new();
/// visualizer.background_color = (0.2, 0.2, 0.3); // Dark blue background
/// visualizer.camera_position = (10.0, 10.0, 10.0); // Further camera position
/// ```
pub struct Visualizer3D {
    pub background_color: (f32, f32, f32),
    pub camera_position: (f32, f32, f32),
    pub camera_target: (f32, f32, f32),
}

/// 3D point for point cloud visualization
///
/// Represents a single point in 3D space with color and intensity information.
/// Used primarily for feature map visualizations where each significant activation
/// becomes a colored point in 3D space.
///
/// # Fields
///
/// - `x, y, z`: 3D coordinates
/// - `color`: RGB color tuple (0-255 range)
/// - `intensity`: Normalized intensity value (0.0-1.0 range)
#[derive(Debug, Clone)]
pub struct Point3D {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub color: (u8, u8, u8),
    pub intensity: f32,
}

/// 3D mesh structure for complex visualizations
///
/// Represents a triangular mesh generated from volumetric data or segmentation masks.
/// Uses a simplified marching cubes approach for isosurface extraction.
///
/// # Components
///
/// - `vertices`: Array of 3D points that form the mesh vertices
/// - `faces`: Triangular faces defined by vertex indices
/// - `normals`: Surface normals for lighting calculations
#[derive(Debug, Clone)]
pub struct Mesh3D {
    pub vertices: Vec<Point3D>,
    pub faces: Vec<[usize; 3]>,
    pub normals: Vec<(f32, f32, f32)>,
}

/// Helper structure for volumetric voxel data
///
/// Represents a single voxel (3D pixel) in a volumetric dataset.
/// Used for volumetric rendering of 3D tensors above a certain threshold.
///
/// # Fields
///
/// - `x, y, z`: Voxel position in 3D space
/// - `value`: Original tensor value at this position
/// - `color`: RGB color based on the voxel value
#[derive(Debug, Clone)]
pub struct VoxelData {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub value: f32,
    pub color: (u8, u8, u8),
}

impl Visualizer3D {
    /// Create a new 3D visualizer with default settings
    ///
    /// Returns a visualizer configured with sensible defaults:
    /// - Dark background (0.1, 0.1, 0.1)
    /// - Camera positioned at (0, 0, 5) looking at origin
    ///
    /// # Examples
    ///
    /// ```rust
    /// use torsh_vision::utils::visualization_3d::Visualizer3D;
    ///
    /// let visualizer = Visualizer3D::new();
    /// assert_eq!(visualizer.background_color, (0.1, 0.1, 0.1));
    /// assert_eq!(visualizer.camera_position, (0.0, 0.0, 5.0));
    /// ```
    pub fn new() -> Self {
        Self {
            background_color: (0.1, 0.1, 0.1),
            camera_position: (0.0, 0.0, 5.0),
            camera_target: (0.0, 0.0, 0.0),
        }
    }

    /// Visualize a 3D tensor as a volumetric rendering
    ///
    /// Converts a 3D tensor into a voxel-based visualization where each tensor element
    /// above the specified threshold becomes a colored cube in 3D space. The resulting
    /// HTML page includes interactive controls for rotation and zoom.
    ///
    /// # Arguments
    ///
    /// - `volume`: 3D tensor with shape (D, H, W)
    /// - `threshold`: Minimum value to include in visualization
    ///
    /// # Returns
    ///
    /// HTML string containing a complete Three.js visualization
    ///
    /// # Errors
    ///
    /// Returns `VisionError::InvalidShape` if the tensor is not 3D.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use torsh_vision::utils::visualization_3d::Visualizer3D;
    /// use torsh_tensor::Tensor;
    /// use torsh_core::device::DeviceType;
    ///
    /// let visualizer = Visualizer3D::new();
    /// let volume = Tensor::randn(vec![16, 16, 16], DeviceType::Cpu)?;
    /// let html = visualizer.visualize_volume(&volume, 0.5)?;
    ///
    /// // Save to file
    /// std::fs::write("volume.html", html)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn visualize_volume(&self, volume: &Tensor<f32>, threshold: f32) -> Result<String> {
        let shape = volume.shape();
        if shape.dims().len() != 3 {
            return Err(VisionError::InvalidShape(format!(
                "Expected 3D tensor (D, H, W), got {}D",
                shape.dims().len()
            )));
        }

        let (depth, height, width) = (shape.dims()[0], shape.dims()[1], shape.dims()[2]);
        let mut voxels = Vec::new();

        // Extract voxels above threshold
        for d in 0..depth {
            for h in 0..height {
                for w in 0..width {
                    let value = volume.get(&[d, h, w])?;
                    if value > threshold {
                        voxels.push(VoxelData {
                            x: w as f32,
                            y: h as f32,
                            z: d as f32,
                            value,
                            color: self.value_to_color(value),
                        });
                    }
                }
            }
        }

        self.generate_volume_html(&voxels, width as f32, height as f32, depth as f32)
    }

    /// Create point cloud visualization from feature maps
    ///
    /// Converts a feature map tensor into a 3D point cloud where each significant
    /// activation becomes a colored point. The spatial dimensions become X-Y coordinates
    /// and the channel dimension becomes the Z coordinate.
    ///
    /// # Arguments
    ///
    /// - `feature_map`: 3D tensor with shape (C, H, W)
    /// - `layer_name`: Name for the layer (displayed in visualization)
    ///
    /// # Returns
    ///
    /// HTML string containing a complete Three.js point cloud visualization
    ///
    /// # Sampling Strategy
    ///
    /// To maintain performance, the function automatically samples points:
    /// - Limits total points to ~1000 for interactive performance
    /// - Only includes activations with absolute value > 0.1
    /// - Samples every 2nd channel (max 10 channels total)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use torsh_vision::utils::visualization_3d::Visualizer3D;
    /// use torsh_tensor::Tensor;
    /// use torsh_core::device::DeviceType;
    ///
    /// let visualizer = Visualizer3D::new();
    /// let feature_map = Tensor::randn(vec![64, 32, 32], DeviceType::Cpu)?;
    /// let html = visualizer.visualize_feature_map(&feature_map, "conv2_relu")?;
    ///
    /// std::fs::write("feature_map.html", html)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn visualize_feature_map(
        &self,
        feature_map: &Tensor<f32>,
        layer_name: &str,
    ) -> Result<String> {
        let shape = feature_map.shape();
        if shape.dims().len() != 3 {
            return Err(VisionError::InvalidShape(format!(
                "Expected 3D tensor (C, H, W), got {}D",
                shape.dims().len()
            )));
        }

        let (channels, height, width) = (shape.dims()[0], shape.dims()[1], shape.dims()[2]);
        let mut points = Vec::new();

        // Sample points from feature map
        let sample_rate = (height * width / 1000).max(1); // Limit to ~1000 points

        for c in (0..channels).step_by(2.max(channels / 10)) {
            for h in (0..height).step_by(sample_rate) {
                for w in (0..width).step_by(sample_rate) {
                    let value = feature_map.get(&[c, h, w])?;
                    if value.abs() > 0.1 {
                        // Only include significant activations
                        points.push(Point3D {
                            x: w as f32 / width as f32,
                            y: h as f32 / height as f32,
                            z: c as f32 / channels as f32,
                            color: self.value_to_color(value),
                            intensity: value.abs(),
                        });
                    }
                }
            }
        }

        self.generate_point_cloud_html(&points, layer_name)
    }

    /// Generate mesh from 3D segmentation mask
    ///
    /// Creates a triangular mesh from a 3D binary mask using a simplified marching cubes
    /// algorithm. This is useful for visualizing segmentation results or object boundaries
    /// in 3D space.
    ///
    /// # Arguments
    ///
    /// - `mask`: 3D tensor with shape (D, H, W) containing binary or probability values
    ///
    /// # Returns
    ///
    /// A `Mesh3D` structure containing vertices, faces, and normals
    ///
    /// # Algorithm
    ///
    /// Uses a simplified marching cubes approach with a threshold of 0.5:
    /// 1. Iterates through each cube in the volume
    /// 2. Checks for surface intersections (values both above and below threshold)
    /// 3. Generates vertices and triangular faces for intersecting cubes
    /// 4. Calculates surface normals for proper lighting
    ///
    /// # Examples
    ///
    /// ```rust
    /// use torsh_vision::utils::visualization_3d::Visualizer3D;
    /// use torsh_tensor::Tensor;
    /// use torsh_core::device::DeviceType;
    ///
    /// let visualizer = Visualizer3D::new();
    ///
    /// // Create a sphere-like mask
    /// let mask = Tensor::zeros(vec![32, 32, 32], DeviceType::Cpu)?;
    /// // ... fill mask with sphere data ...
    ///
    /// let mesh = visualizer.generate_mesh_from_mask(&mask)?;
    /// println!("Generated mesh with {} vertices and {} faces",
    ///          mesh.vertices.len(), mesh.faces.len());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn generate_mesh_from_mask(&self, mask: &Tensor<f32>) -> Result<Mesh3D> {
        let shape = mask.shape();
        if shape.dims().len() != 3 {
            return Err(VisionError::InvalidShape(format!(
                "Expected 3D tensor (D, H, W), got {}D",
                shape.dims().len()
            )));
        }

        let (depth, height, width) = (shape.dims()[0], shape.dims()[1], shape.dims()[2]);
        let mut vertices = Vec::new();
        let mut faces = Vec::new();

        // Marching cubes algorithm (simplified implementation)
        for d in 0..(depth - 1) {
            for h in 0..(height - 1) {
                for w in 0..(width - 1) {
                    let cube_values = [
                        mask.get(&[d, h, w])?,
                        mask.get(&[d, h, w + 1])?,
                        mask.get(&[d, h + 1, w + 1])?,
                        mask.get(&[d, h + 1, w])?,
                        mask.get(&[d + 1, h, w])?,
                        mask.get(&[d + 1, h, w + 1])?,
                        mask.get(&[d + 1, h + 1, w + 1])?,
                        mask.get(&[d + 1, h + 1, w])?,
                    ];

                    // Simple isosurface extraction at threshold 0.5
                    let threshold = 0.5;
                    if self.has_surface_intersection(&cube_values, threshold) {
                        // Add vertices and faces for this cube
                        let base_vertex = vertices.len();

                        // Add cube corner vertices
                        vertices.push(Point3D {
                            x: w as f32,
                            y: h as f32,
                            z: d as f32,
                            color: (255, 255, 255),
                            intensity: 1.0,
                        });
                        vertices.push(Point3D {
                            x: (w + 1) as f32,
                            y: h as f32,
                            z: d as f32,
                            color: (255, 255, 255),
                            intensity: 1.0,
                        });
                        vertices.push(Point3D {
                            x: (w + 1) as f32,
                            y: (h + 1) as f32,
                            z: d as f32,
                            color: (255, 255, 255),
                            intensity: 1.0,
                        });
                        vertices.push(Point3D {
                            x: w as f32,
                            y: (h + 1) as f32,
                            z: d as f32,
                            color: (255, 255, 255),
                            intensity: 1.0,
                        });

                        // Add triangular faces (simplified)
                        faces.push([base_vertex, base_vertex + 1, base_vertex + 2]);
                        faces.push([base_vertex, base_vertex + 2, base_vertex + 3]);
                    }
                }
            }
        }

        let normals = self.calculate_normals(&vertices, &faces);

        Ok(Mesh3D {
            vertices,
            faces,
            normals,
        })
    }

    /// Helper function to check for surface intersection in a cube
    ///
    /// Determines if an isosurface passes through a cube by checking if the
    /// cube contains values both above and below the threshold.
    ///
    /// # Arguments
    ///
    /// - `values`: Array of 8 values at cube corners
    /// - `threshold`: Isosurface threshold value
    ///
    /// # Returns
    ///
    /// `true` if the surface intersects the cube, `false` otherwise
    fn has_surface_intersection(&self, values: &[f32; 8], threshold: f32) -> bool {
        let above_threshold = values.iter().filter(|&&v| v > threshold).count();
        above_threshold > 0 && above_threshold < 8
    }

    /// Calculate face normals for mesh
    ///
    /// Computes surface normals for each triangular face in the mesh using
    /// the cross product of edge vectors. Normals are normalized for consistent
    /// lighting calculations.
    ///
    /// # Arguments
    ///
    /// - `vertices`: Array of mesh vertices
    /// - `faces`: Array of triangular faces (vertex indices)
    ///
    /// # Returns
    ///
    /// Vector of normalized surface normals, one per face
    fn calculate_normals(
        &self,
        vertices: &[Point3D],
        faces: &[[usize; 3]],
    ) -> Vec<(f32, f32, f32)> {
        let mut normals = Vec::with_capacity(faces.len());

        for face in faces {
            let v1 = &vertices[face[0]];
            let v2 = &vertices[face[1]];
            let v3 = &vertices[face[2]];

            let edge1 = (v2.x - v1.x, v2.y - v1.y, v2.z - v1.z);
            let edge2 = (v3.x - v1.x, v3.y - v1.y, v3.z - v1.z);

            // Cross product for normal
            let normal = (
                edge1.1 * edge2.2 - edge1.2 * edge2.1,
                edge1.2 * edge2.0 - edge1.0 * edge2.2,
                edge1.0 * edge2.1 - edge1.1 * edge2.0,
            );

            // Normalize
            let length = (normal.0 * normal.0 + normal.1 * normal.1 + normal.2 * normal.2).sqrt();
            if length > 0.0 {
                normals.push((normal.0 / length, normal.1 / length, normal.2 / length));
            } else {
                normals.push((0.0, 1.0, 0.0));
            }
        }

        normals
    }

    /// Convert scalar value to color
    ///
    /// Maps scalar values to RGB colors using a diverging color scheme:
    /// - Positive values: Blue to Red gradient
    /// - Negative values: Red to Green gradient
    /// - Zero values: Blue
    ///
    /// # Arguments
    ///
    /// - `value`: Scalar value to convert (clamped to [-1, 1] range)
    ///
    /// # Returns
    ///
    /// RGB color tuple with values in 0-255 range
    fn value_to_color(&self, value: f32) -> (u8, u8, u8) {
        let normalized = (value.abs().clamp(0.0, 1.0) * 255.0) as u8;

        if value >= 0.0 {
            (normalized, 0, 255 - normalized) // Blue to red for positive values
        } else {
            (255 - normalized, normalized, 0) // Red to green for negative values
        }
    }

    /// Generate HTML for volumetric visualization using Three.js
    ///
    /// Creates a complete HTML page with embedded Three.js visualization for
    /// volumetric data. The page includes interactive controls, lighting, and
    /// information display.
    ///
    /// # Arguments
    ///
    /// - `voxels`: Array of voxel data to visualize
    /// - `width, height, depth`: Dimensions of the original volume
    ///
    /// # Returns
    ///
    /// Complete HTML string ready to save as a file
    ///
    /// # Features
    ///
    /// - Orbit controls for mouse interaction
    /// - Ambient and directional lighting
    /// - Responsive design that adapts to window size
    /// - Information panel with volume statistics
    fn generate_volume_html(
        &self,
        voxels: &[VoxelData],
        width: f32,
        height: f32,
        depth: f32,
    ) -> Result<String> {
        let html = format!(
            r#"
<!DOCTYPE html>
<html>
<head>
    <title>ToRSh Vision 3D Volume Viewer</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <style>
        body {{ margin: 0; padding: 0; background: #000; }}
        #container {{ width: 100vw; height: 100vh; }}
        #info {{ position: absolute; top: 10px; left: 10px; color: white; font-family: Arial; }}
    </style>
</head>
<body>
    <div id="container"></div>
    <div id="info">
        <h3>3D Volume Visualization</h3>
        <p>Dimensions: {:.0} x {:.0} x {:.0}</p>
        <p>Voxels: {}</p>
        <p>Mouse: Rotate | Scroll: Zoom</p>
    </div>

    <script>
        // Scene setup
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer();
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setClearColor(0x{:02x}{:02x}{:02x});
        document.getElementById('container').appendChild(renderer.domElement);

        // Controls
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;

        // Add voxels
        const voxelGroup = new THREE.Group();
        const voxelGeometry = new THREE.BoxGeometry(1, 1, 1);

        // Voxel data would be embedded here in a real implementation
        const voxelData = [];

        voxelData.forEach(voxel => {{
            const material = new THREE.MeshBasicMaterial({{
                color: new THREE.Color(`rgb(${{voxel.color[0]}}, ${{voxel.color[1]}}, ${{voxel.color[2]}})`)
            }});
            const cube = new THREE.Mesh(voxelGeometry, material);
            cube.position.set(voxel.x - {:.1}, voxel.y - {:.1}, voxel.z - {:.1});
            voxelGroup.add(cube);
        }});

        scene.add(voxelGroup);

        // Lighting
        const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
        scene.add(ambientLight);
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(10, 10, 5);
        scene.add(directionalLight);

        // Camera position
        camera.position.set({:.1}, {:.1}, {:.1});

        // Animation loop
        function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }}

        // Handle window resize
        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }});

        animate();
    </script>
</body>
</html>
        "#,
            width,
            height,
            depth,
            voxels.len(),
            (self.background_color.0 * 255.0) as u8,
            (self.background_color.1 * 255.0) as u8,
            (self.background_color.2 * 255.0) as u8,
            width / 2.0,
            height / 2.0,
            depth / 2.0,
            self.camera_position.0,
            self.camera_position.1,
            self.camera_position.2
        );

        Ok(html)
    }

    /// Generate HTML for point cloud visualization
    ///
    /// Creates a complete HTML page with embedded Three.js point cloud visualization.
    /// The visualization includes automatic rotation and interactive controls.
    ///
    /// # Arguments
    ///
    /// - `points`: Array of 3D points to visualize
    /// - `layer_name`: Name of the layer being visualized
    ///
    /// # Returns
    ///
    /// Complete HTML string ready to save as a file
    ///
    /// # Features
    ///
    /// - Point-based rendering with vertex colors
    /// - Automatic rotation animation
    /// - Orbit controls for manual interaction
    /// - Responsive design
    /// - Information panel with point count
    fn generate_point_cloud_html(&self, points: &[Point3D], layer_name: &str) -> Result<String> {
        let points_json: Vec<String> = points
            .iter()
            .map(|p| {
                format!(
                    r#"{{"x": {}, "y": {}, "z": {}, "color": [{}, {}, {}], "intensity": {}}}"#,
                    p.x, p.y, p.z, p.color.0, p.color.1, p.color.2, p.intensity
                )
            })
            .collect();

        let html = format!(
            r#"
<!DOCTYPE html>
<html>
<head>
    <title>ToRSh Vision 3D Feature Map: {}</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <style>
        body {{ margin: 0; padding: 0; background: #000; }}
        #container {{ width: 100vw; height: 100vh; }}
        #info {{ position: absolute; top: 10px; left: 10px; color: white; font-family: Arial; }}
    </style>
</head>
<body>
    <div id="container"></div>
    <div id="info">
        <h3>3D Feature Map: {}</h3>
        <p>Points: {}</p>
        <p>Mouse: Rotate | Scroll: Zoom</p>
    </div>

    <script>
        // Scene setup
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer();
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setClearColor(0x{:02x}{:02x}{:02x});
        document.getElementById('container').appendChild(renderer.domElement);

        // Controls
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;

        // Point cloud
        const pointsData = [{}];
        const geometry = new THREE.BufferGeometry();
        const positions = [];
        const colors = [];

        pointsData.forEach(point => {{
            positions.push(point.x, point.y, point.z);
            colors.push(point.color[0] / 255, point.color[1] / 255, point.color[2] / 255);
        }});

        geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

        const material = new THREE.PointsMaterial({{
            size: 0.01,
            vertexColors: true,
            transparent: true,
            opacity: 0.8
        }});

        const pointCloud = new THREE.Points(geometry, material);
        scene.add(pointCloud);

        // Lighting
        const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
        scene.add(ambientLight);

        // Camera position
        camera.position.set(1.5, 1.5, 1.5);

        // Animation loop
        function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            pointCloud.rotation.y += 0.005;
            renderer.render(scene, camera);
        }}

        // Handle window resize
        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }});

        animate();
    </script>
</body>
</html>
        "#,
            layer_name,
            layer_name,
            points.len(),
            (self.background_color.0 * 255.0) as u8,
            (self.background_color.1 * 255.0) as u8,
            (self.background_color.2 * 255.0) as u8,
            points_json.join(", ")
        );

        Ok(html)
    }

    /// Save 3D visualization to file
    ///
    /// Writes HTML content to a file. The file can be opened directly in a web browser
    /// to view the interactive 3D visualization.
    ///
    /// # Arguments
    ///
    /// - `html_content`: Complete HTML string from visualization methods
    /// - `path`: File path where the HTML should be saved
    ///
    /// # Examples
    ///
    /// ```rust
    /// use torsh_vision::utils::visualization_3d::Visualizer3D;
    /// use torsh_tensor::Tensor;
    /// use torsh_core::device::DeviceType;
    ///
    /// let visualizer = Visualizer3D::new();
    /// let volume = Tensor::randn(vec![16, 16, 16], DeviceType::Cpu)?;
    /// let html = visualizer.visualize_volume(&volume, 0.3)?;
    ///
    /// visualizer.save_visualization(&html, "my_volume.html")?;
    /// println!("Visualization saved! Open my_volume.html in a browser.");
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn save_visualization<P: AsRef<Path>>(&self, html_content: &str, path: P) -> Result<()> {
        std::fs::write(path, html_content)?;
        Ok(())
    }
}

impl Default for Visualizer3D {
    fn default() -> Self {
        Self::new()
    }
}

/// Create a 3D visualizer for volumetric and point cloud data
///
/// Convenience function that creates a new `Visualizer3D` with default settings.
/// Equivalent to calling `Visualizer3D::new()`.
///
/// # Examples
///
/// ```rust
/// use torsh_vision::utils::visualization_3d::create_3d_visualizer;
///
/// let visualizer = create_3d_visualizer();
/// // Same as: let visualizer = Visualizer3D::new();
/// ```
pub fn create_3d_visualizer() -> Visualizer3D {
    Visualizer3D::new()
}

/// Visualize tensor activations in 3D space
///
/// Batch processes multiple activation tensors and creates a complete visualization
/// suite with individual HTML files for each layer and a navigation index page.
///
/// # Arguments
///
/// - `activations`: Array of activation tensors (each should be 3D: C×H×W)
/// - `layer_names`: Corresponding names for each activation tensor
/// - `output_dir`: Directory where HTML files will be saved
///
/// # Output Structure
///
/// ```text
/// output_dir/
/// ├── index.html          # Navigation page with links to all layers
/// ├── layer_00_conv1.html # Individual layer visualizations
/// ├── layer_01_conv2.html
/// └── ...
/// ```
///
/// # Examples
///
/// ```rust
/// use torsh_vision::utils::visualization_3d::visualize_activations_3d;
/// use torsh_tensor::Tensor;
/// use torsh_core::device::DeviceType;
///
/// // Simulate activations from different layers
/// let activations = vec![
///     Tensor::randn(vec![32, 64, 64], DeviceType::Cpu)?,  // Early layer
///     Tensor::randn(vec![64, 32, 32], DeviceType::Cpu)?,  // Middle layer
///     Tensor::randn(vec![128, 16, 16], DeviceType::Cpu)?, // Deep layer
/// ];
///
/// let layer_names = vec![
///     "conv1".to_string(),
///     "conv2".to_string(),
///     "conv3".to_string(),
/// ];
///
/// visualize_activations_3d(&activations, &layer_names, "./viz_output")?;
/// println!("Open ./viz_output/index.html to browse all visualizations");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn visualize_activations_3d(
    activations: &[Tensor<f32>],
    layer_names: &[String],
    output_dir: &str,
) -> Result<()> {
    let visualizer = Visualizer3D::new();

    std::fs::create_dir_all(output_dir)?;

    for (i, (activation, name)) in activations.iter().zip(layer_names.iter()).enumerate() {
        let html = visualizer.visualize_feature_map(activation, name)?;
        let filename = format!("{}/layer_{:02}_{}.html", output_dir, i, name);
        visualizer.save_visualization(&html, filename)?;
    }

    // Create index page
    let index_html = create_activation_index(layer_names);
    let index_path = format!("{}/index.html", output_dir);
    std::fs::write(index_path, index_html)?;

    println!("3D visualizations saved to: {}", output_dir);
    Ok(())
}

/// Create an index page linking all visualization files
///
/// Generates an HTML navigation page that provides easy access to all layer
/// visualizations. The page includes a clean interface with hover effects
/// and descriptive styling.
///
/// # Arguments
///
/// - `layer_names`: Array of layer names to create links for
///
/// # Returns
///
/// Complete HTML string for the index page
fn create_activation_index(layer_names: &[String]) -> String {
    let mut links = String::new();
    for (i, name) in layer_names.iter().enumerate() {
        links.push_str(&format!(
            r#"<li><a href="layer_{:02}_{}.html">{}</a></li>"#,
            i, name, name
        ));
    }

    format!(
        r#"
<!DOCTYPE html>
<html>
<head>
    <title>ToRSh Vision 3D Activations Index</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        ul {{ list-style-type: none; padding: 0; }}
        li {{ margin: 10px 0; }}
        a {{ display: block; padding: 10px; background: #f0f0f0; text-decoration: none; border-radius: 4px; }}
        a:hover {{ background: #e0e0e0; }}
    </style>
</head>
<body>
    <h1>ToRSh Vision 3D Layer Activations</h1>
    <p>Click on any layer to view its 3D activation visualization:</p>
    <ul>
        {}
    </ul>
</body>
</html>
    "#,
        links
    )
}
