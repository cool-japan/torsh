//! 3D visualization tools for torsh-vision
//!
//! This module provides 3D visualization capabilities including point clouds,
//! mesh rendering, volumetric visualization, and 3D object detection visualization.

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use crate::{Result, VisionError};
use std::collections::HashMap;
use torsh_core::dtype::DType;
use torsh_core::DeviceType;
use torsh_tensor::Tensor;

/// 3D point in space with optional color and normal
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point3D {
    /// X coordinate
    pub x: f32,
    /// Y coordinate
    pub y: f32,
    /// Z coordinate
    pub z: f32,
    /// RGB color (optional)
    pub color: Option<[u8; 3]>,
    /// Surface normal (optional)
    pub normal: Option<[f32; 3]>,
}

impl Point3D {
    /// Create a new 3D point
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self {
            x,
            y,
            z,
            color: None,
            normal: None,
        }
    }

    /// Create a new 3D point with color
    pub fn with_color(x: f32, y: f32, z: f32, color: [u8; 3]) -> Self {
        Self {
            x,
            y,
            z,
            color: Some(color),
            normal: None,
        }
    }

    /// Create a new 3D point with color and normal
    pub fn with_color_and_normal(x: f32, y: f32, z: f32, color: [u8; 3], normal: [f32; 3]) -> Self {
        Self {
            x,
            y,
            z,
            color: Some(color),
            normal: Some(normal),
        }
    }

    /// Get position as array
    pub fn position(&self) -> [f32; 3] {
        [self.x, self.y, self.z]
    }

    /// Calculate distance to another point
    pub fn distance_to(&self, other: &Point3D) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

/// 3D triangle face for mesh rendering
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Triangle3D {
    /// Vertex indices
    pub vertices: [usize; 3],
    /// Face normal (optional)
    pub normal: Option<[f32; 3]>,
    /// Material ID (optional)
    pub material_id: Option<u32>,
}

impl Triangle3D {
    /// Create a new triangle
    pub fn new(v0: usize, v1: usize, v2: usize) -> Self {
        Self {
            vertices: [v0, v1, v2],
            normal: None,
            material_id: None,
        }
    }

    /// Create a triangle with normal
    pub fn with_normal(v0: usize, v1: usize, v2: usize, normal: [f32; 3]) -> Self {
        Self {
            vertices: [v0, v1, v2],
            normal: Some(normal),
            material_id: None,
        }
    }
}

/// 3D bounding box for object detection visualization
#[derive(Debug, Clone, PartialEq)]
pub struct BoundingBox3D {
    /// Center position
    pub center: [f32; 3],
    /// Dimensions (width, height, depth)
    pub dimensions: [f32; 3],
    /// Rotation angles (roll, pitch, yaw)
    pub rotation: [f32; 3],
    /// Label
    pub label: String,
    /// Confidence score
    pub confidence: f32,
    /// Color
    pub color: [u8; 3],
}

impl BoundingBox3D {
    /// Create a new 3D bounding box
    pub fn new(
        center: [f32; 3],
        dimensions: [f32; 3],
        rotation: [f32; 3],
        label: String,
        confidence: f32,
    ) -> Self {
        Self {
            center,
            dimensions,
            rotation,
            label,
            confidence,
            color: [255, 0, 0], // Default red
        }
    }

    /// Set color
    pub fn with_color(mut self, color: [u8; 3]) -> Self {
        self.color = color;
        self
    }

    /// Get the 8 corner points of the bounding box
    pub fn corners(&self) -> [Point3D; 8] {
        let [cx, cy, cz] = self.center;
        let [w, h, d] = self.dimensions;
        let [roll, pitch, yaw] = self.rotation;

        // Half dimensions
        let hw = w / 2.0;
        let hh = h / 2.0;
        let hd = d / 2.0;

        // Local corners (before rotation)
        let local_corners = [
            [-hw, -hh, -hd],
            [hw, -hh, -hd],
            [hw, hh, -hd],
            [-hw, hh, -hd],
            [-hw, -hh, hd],
            [hw, -hh, hd],
            [hw, hh, hd],
            [-hw, hh, hd],
        ];

        // Apply rotation and translation
        let mut corners = [Point3D::new(0.0, 0.0, 0.0); 8];
        for (i, &[x, y, z]) in local_corners.iter().enumerate() {
            let rotated = rotate_point([x, y, z], roll, pitch, yaw);
            corners[i] = Point3D::with_color(
                rotated[0] + cx,
                rotated[1] + cy,
                rotated[2] + cz,
                self.color,
            );
        }

        corners
    }

    /// Get volume
    pub fn volume(&self) -> f32 {
        self.dimensions[0] * self.dimensions[1] * self.dimensions[2]
    }

    /// Check if point is inside the bounding box
    pub fn contains_point(&self, point: Point3D) -> bool {
        // Transform point to local coordinates
        let local_point = [
            point.x - self.center[0],
            point.y - self.center[1],
            point.z - self.center[2],
        ];

        // Apply inverse rotation
        let local_rotated = rotate_point(
            local_point,
            -self.rotation[0],
            -self.rotation[1],
            -self.rotation[2],
        );

        // Check bounds
        let [hw, hh, hd] = [
            self.dimensions[0] / 2.0,
            self.dimensions[1] / 2.0,
            self.dimensions[2] / 2.0,
        ];

        local_rotated[0].abs() <= hw && local_rotated[1].abs() <= hh && local_rotated[2].abs() <= hd
    }
}

/// 3D point cloud for visualization
#[derive(Debug, Clone)]
pub struct PointCloud3D {
    /// Points in the cloud
    pub points: Vec<Point3D>,
    /// Point cloud metadata
    pub metadata: PointCloudMetadata,
}

/// Metadata for point clouds
#[derive(Debug, Clone)]
pub struct PointCloudMetadata {
    /// Number of points
    pub num_points: usize,
    /// Bounding box of the point cloud
    pub bounds: Option<BoundingBox3D>,
    /// Point cloud source
    pub source: String,
    /// Creation timestamp
    pub created_at: std::time::SystemTime,
}

impl PointCloud3D {
    /// Create a new point cloud
    pub fn new(points: Vec<Point3D>) -> Self {
        let num_points = points.len();
        let bounds = if !points.is_empty() {
            Some(Self::calculate_bounds(&points))
        } else {
            None
        };

        Self {
            points,
            metadata: PointCloudMetadata {
                num_points,
                bounds,
                source: "unknown".to_string(),
                created_at: std::time::SystemTime::now(),
            },
        }
    }

    /// Create point cloud from tensor
    pub fn from_tensor(tensor: &Tensor) -> Result<Self> {
        // Expect tensor of shape [N, 3] or [N, 6] (XYZ or XYZRGB)
        if tensor.ndim() != 2 {
            return Err(VisionError::InvalidShape(
                "Point cloud tensor must be 2D".to_string(),
            ));
        }

        let shape = tensor.shape();
        let num_points = shape.dims()[0];
        let num_features = shape.dims()[1];

        if num_features != 3 && num_features != 6 {
            return Err(VisionError::InvalidShape(
                "Point cloud tensor must have 3 or 6 features (XYZ or XYZRGB)".to_string(),
            ));
        }

        let mut points = Vec::with_capacity(num_points);

        // Convert tensor data to points
        // This is a simplified conversion - in reality you'd extract the actual tensor data
        for _i in 0..num_points {
            let x = 0.0; // Extract from tensor[i, 0]
            let y = 0.0; // Extract from tensor[i, 1]
            let z = 0.0; // Extract from tensor[i, 2]

            let point = if num_features == 6 {
                // Has color information
                let r = 0u8; // Extract from tensor[i, 3]
                let g = 0u8; // Extract from tensor[i, 4]
                let b = 0u8; // Extract from tensor[i, 5]
                Point3D::with_color(x, y, z, [r, g, b])
            } else {
                Point3D::new(x, y, z)
            };

            points.push(point);
        }

        Ok(Self::new(points))
    }

    /// Convert point cloud to tensor
    pub fn to_tensor(&self) -> Result<Tensor> {
        let num_points = self.points.len();
        let has_color = self.points.iter().any(|p| p.color.is_some());
        let num_features = if has_color { 6 } else { 3 };

        // Create tensor data
        let mut data = Vec::with_capacity(num_points * num_features);

        for point in &self.points {
            data.push(point.x);
            data.push(point.y);
            data.push(point.z);

            if has_color {
                if let Some(color) = point.color {
                    data.push(color[0] as f32 / 255.0);
                    data.push(color[1] as f32 / 255.0);
                    data.push(color[2] as f32 / 255.0);
                } else {
                    data.push(0.0);
                    data.push(0.0);
                    data.push(0.0);
                }
            }
        }

        Ok(Tensor::from_data(
            data,
            vec![num_points, num_features],
            DeviceType::Cpu,
        )?)
    }

    /// Calculate bounding box of points
    fn calculate_bounds(points: &[Point3D]) -> BoundingBox3D {
        if points.is_empty() {
            return BoundingBox3D::new([0.0; 3], [0.0; 3], [0.0; 3], "empty".to_string(), 1.0);
        }

        let mut min_x = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_y = f32::NEG_INFINITY;
        let mut min_z = f32::INFINITY;
        let mut max_z = f32::NEG_INFINITY;

        for point in points {
            min_x = min_x.min(point.x);
            max_x = max_x.max(point.x);
            min_y = min_y.min(point.y);
            max_y = max_y.max(point.y);
            min_z = min_z.min(point.z);
            max_z = max_z.max(point.z);
        }

        let center = [
            (min_x + max_x) / 2.0,
            (min_y + max_y) / 2.0,
            (min_z + max_z) / 2.0,
        ];

        let dimensions = [max_x - min_x, max_y - min_y, max_z - min_z];

        BoundingBox3D::new(center, dimensions, [0.0; 3], "bounds".to_string(), 1.0)
    }

    /// Add point to cloud
    pub fn add_point(&mut self, point: Point3D) {
        self.points.push(point);
        self.metadata.num_points = self.points.len();
        self.metadata.bounds = Some(Self::calculate_bounds(&self.points));
    }

    /// Remove point by index
    pub fn remove_point(&mut self, index: usize) -> Result<Point3D> {
        if index >= self.points.len() {
            return Err(VisionError::InvalidArgument(format!(
                "Point index {} out of bounds",
                index
            )));
        }

        let point = self.points.remove(index);
        self.metadata.num_points = self.points.len();
        if !self.points.is_empty() {
            self.metadata.bounds = Some(Self::calculate_bounds(&self.points));
        } else {
            self.metadata.bounds = None;
        }

        Ok(point)
    }

    /// Get number of points
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Check if point cloud is empty
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Filter points by distance from center
    pub fn filter_by_distance(&self, center: Point3D, max_distance: f32) -> Self {
        let filtered_points: Vec<Point3D> = self
            .points
            .iter()
            .filter(|&point| point.distance_to(&center) <= max_distance)
            .cloned()
            .collect();

        Self::new(filtered_points)
    }

    /// Downsample point cloud using voxel grid
    pub fn voxel_downsample(&self, voxel_size: f32) -> Self {
        let mut voxel_map: HashMap<(i32, i32, i32), Vec<Point3D>> = HashMap::new();

        // Group points by voxel
        for &point in &self.points {
            let voxel_x = (point.x / voxel_size).floor() as i32;
            let voxel_y = (point.y / voxel_size).floor() as i32;
            let voxel_z = (point.z / voxel_size).floor() as i32;

            voxel_map
                .entry((voxel_x, voxel_y, voxel_z))
                .or_default()
                .push(point);
        }

        // Average points in each voxel
        let mut downsampled_points = Vec::new();
        for voxel_points in voxel_map.values() {
            if voxel_points.is_empty() {
                continue;
            }

            let avg_x = voxel_points.iter().map(|p| p.x).sum::<f32>() / voxel_points.len() as f32;
            let avg_y = voxel_points.iter().map(|p| p.y).sum::<f32>() / voxel_points.len() as f32;
            let avg_z = voxel_points.iter().map(|p| p.z).sum::<f32>() / voxel_points.len() as f32;

            // Average color if available
            let avg_color = if voxel_points.iter().any(|p| p.color.is_some()) {
                let colors: Vec<[u8; 3]> = voxel_points.iter().filter_map(|p| p.color).collect();

                if !colors.is_empty() {
                    let avg_r =
                        colors.iter().map(|c| c[0] as u32).sum::<u32>() / colors.len() as u32;
                    let avg_g =
                        colors.iter().map(|c| c[1] as u32).sum::<u32>() / colors.len() as u32;
                    let avg_b =
                        colors.iter().map(|c| c[2] as u32).sum::<u32>() / colors.len() as u32;
                    Some([avg_r as u8, avg_g as u8, avg_b as u8])
                } else {
                    None
                }
            } else {
                None
            };

            let point = if let Some(color) = avg_color {
                Point3D::with_color(avg_x, avg_y, avg_z, color)
            } else {
                Point3D::new(avg_x, avg_y, avg_z)
            };

            downsampled_points.push(point);
        }

        Self::new(downsampled_points)
    }
}

/// 3D mesh for surface visualization
#[derive(Debug, Clone)]
pub struct Mesh3D {
    /// Vertices of the mesh
    pub vertices: Vec<Point3D>,
    /// Triangular faces
    pub faces: Vec<Triangle3D>,
    /// Mesh metadata
    pub metadata: MeshMetadata,
}

/// Metadata for 3D meshes
#[derive(Debug, Clone)]
pub struct MeshMetadata {
    /// Number of vertices
    pub num_vertices: usize,
    /// Number of faces
    pub num_faces: usize,
    /// Mesh name
    pub name: String,
    /// Whether normals are computed
    pub has_normals: bool,
    /// Whether texture coordinates exist
    pub has_texture: bool,
}

impl Mesh3D {
    /// Create a new mesh
    pub fn new(vertices: Vec<Point3D>, faces: Vec<Triangle3D>) -> Self {
        let has_normals = vertices.iter().any(|v| v.normal.is_some());

        Self {
            metadata: MeshMetadata {
                num_vertices: vertices.len(),
                num_faces: faces.len(),
                name: "mesh".to_string(),
                has_normals,
                has_texture: false,
            },
            vertices,
            faces,
        }
    }

    /// Compute face normals
    pub fn compute_face_normals(&mut self) {
        for face in &mut self.faces {
            let v0 = self.vertices[face.vertices[0]];
            let v1 = self.vertices[face.vertices[1]];
            let v2 = self.vertices[face.vertices[2]];

            let edge1 = [v1.x - v0.x, v1.y - v0.y, v1.z - v0.z];
            let edge2 = [v2.x - v0.x, v2.y - v0.y, v2.z - v0.z];

            let normal = cross_product(edge1, edge2);
            let length =
                (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();

            if length > 0.0 {
                face.normal = Some([normal[0] / length, normal[1] / length, normal[2] / length]);
            }
        }
    }

    /// Compute vertex normals from face normals
    pub fn compute_vertex_normals(&mut self) {
        if self.faces.iter().any(|f| f.normal.is_none()) {
            self.compute_face_normals();
        }

        let mut vertex_normals = vec![[0.0f32; 3]; self.vertices.len()];
        let mut vertex_counts = vec![0u32; self.vertices.len()];

        // Accumulate face normals at vertices
        for face in &self.faces {
            if let Some(face_normal) = face.normal {
                for &vertex_idx in &face.vertices {
                    vertex_normals[vertex_idx][0] += face_normal[0];
                    vertex_normals[vertex_idx][1] += face_normal[1];
                    vertex_normals[vertex_idx][2] += face_normal[2];
                    vertex_counts[vertex_idx] += 1;
                }
            }
        }

        // Average and normalize
        for (i, vertex) in self.vertices.iter_mut().enumerate() {
            if vertex_counts[i] > 0 {
                let count = vertex_counts[i] as f32;
                let mut normal = [
                    vertex_normals[i][0] / count,
                    vertex_normals[i][1] / count,
                    vertex_normals[i][2] / count,
                ];

                let length =
                    (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
                if length > 0.0 {
                    normal[0] /= length;
                    normal[1] /= length;
                    normal[2] /= length;
                    vertex.normal = Some(normal);
                }
            }
        }

        self.metadata.has_normals = true;
    }

    /// Create a sphere mesh
    pub fn create_sphere(center: Point3D, radius: f32, rings: usize, sectors: usize) -> Self {
        let mut vertices = Vec::new();
        let mut faces = Vec::new();

        // Generate vertices
        for i in 0..=rings {
            let lat_angle = std::f32::consts::PI * i as f32 / rings as f32;
            let y = radius * lat_angle.cos();
            let ring_radius = radius * lat_angle.sin();

            for j in 0..=sectors {
                let lon_angle = 2.0 * std::f32::consts::PI * j as f32 / sectors as f32;
                let x = ring_radius * lon_angle.cos();
                let z = ring_radius * lon_angle.sin();

                let vertex = Point3D::new(center.x + x, center.y + y, center.z + z);
                vertices.push(vertex);
            }
        }

        // Generate faces
        for i in 0..rings {
            for j in 0..sectors {
                let current = i * (sectors + 1) + j;
                let next = current + sectors + 1;

                // Two triangles per quad
                faces.push(Triangle3D::new(current, next, current + 1));
                faces.push(Triangle3D::new(current + 1, next, next + 1));
            }
        }

        let mut mesh = Self::new(vertices, faces);
        mesh.compute_vertex_normals();
        mesh.metadata.name = "sphere".to_string();
        mesh
    }

    /// Create a cube mesh
    pub fn create_cube(center: Point3D, size: f32) -> Self {
        let half_size = size / 2.0;

        let vertices = vec![
            Point3D::new(
                center.x - half_size,
                center.y - half_size,
                center.z - half_size,
            ), // 0
            Point3D::new(
                center.x + half_size,
                center.y - half_size,
                center.z - half_size,
            ), // 1
            Point3D::new(
                center.x + half_size,
                center.y + half_size,
                center.z - half_size,
            ), // 2
            Point3D::new(
                center.x - half_size,
                center.y + half_size,
                center.z - half_size,
            ), // 3
            Point3D::new(
                center.x - half_size,
                center.y - half_size,
                center.z + half_size,
            ), // 4
            Point3D::new(
                center.x + half_size,
                center.y - half_size,
                center.z + half_size,
            ), // 5
            Point3D::new(
                center.x + half_size,
                center.y + half_size,
                center.z + half_size,
            ), // 6
            Point3D::new(
                center.x - half_size,
                center.y + half_size,
                center.z + half_size,
            ), // 7
        ];

        let faces = vec![
            // Front face
            Triangle3D::new(0, 1, 2),
            Triangle3D::new(0, 2, 3),
            // Back face
            Triangle3D::new(4, 6, 5),
            Triangle3D::new(4, 7, 6),
            // Left face
            Triangle3D::new(0, 3, 7),
            Triangle3D::new(0, 7, 4),
            // Right face
            Triangle3D::new(1, 5, 6),
            Triangle3D::new(1, 6, 2),
            // Top face
            Triangle3D::new(3, 2, 6),
            Triangle3D::new(3, 6, 7),
            // Bottom face
            Triangle3D::new(0, 4, 5),
            Triangle3D::new(0, 5, 1),
        ];

        let mut mesh = Self::new(vertices, faces);
        mesh.compute_vertex_normals();
        mesh.metadata.name = "cube".to_string();
        mesh
    }
}

/// 3D visualization scene containing multiple objects
#[derive(Debug)]
pub struct Scene3D {
    /// Point clouds in the scene
    pub point_clouds: Vec<PointCloud3D>,
    /// Meshes in the scene
    pub meshes: Vec<Mesh3D>,
    /// 3D bounding boxes
    pub bounding_boxes: Vec<BoundingBox3D>,
    /// Scene metadata
    pub metadata: SceneMetadata,
}

/// Metadata for 3D scenes
#[derive(Debug, Clone)]
pub struct SceneMetadata {
    /// Scene name
    pub name: String,
    /// Scene bounds
    pub bounds: Option<BoundingBox3D>,
    /// Number of objects
    pub num_objects: usize,
    /// Creation time
    pub created_at: std::time::SystemTime,
}

impl Scene3D {
    /// Create a new 3D scene
    pub fn new(name: String) -> Self {
        Self {
            point_clouds: Vec::new(),
            meshes: Vec::new(),
            bounding_boxes: Vec::new(),
            metadata: SceneMetadata {
                name,
                bounds: None,
                num_objects: 0,
                created_at: std::time::SystemTime::now(),
            },
        }
    }

    /// Add point cloud to scene
    pub fn add_point_cloud(&mut self, point_cloud: PointCloud3D) {
        self.point_clouds.push(point_cloud);
        self.update_metadata();
    }

    /// Add mesh to scene
    pub fn add_mesh(&mut self, mesh: Mesh3D) {
        self.meshes.push(mesh);
        self.update_metadata();
    }

    /// Add bounding box to scene
    pub fn add_bounding_box(&mut self, bbox: BoundingBox3D) {
        self.bounding_boxes.push(bbox);
        self.update_metadata();
    }

    /// Update scene metadata
    fn update_metadata(&mut self) {
        self.metadata.num_objects =
            self.point_clouds.len() + self.meshes.len() + self.bounding_boxes.len();

        // Calculate scene bounds
        let mut all_points = Vec::new();

        // Add points from point clouds
        for pc in &self.point_clouds {
            all_points.extend_from_slice(&pc.points);
        }

        // Add vertices from meshes
        for mesh in &self.meshes {
            all_points.extend_from_slice(&mesh.vertices);
        }

        // Add corners from bounding boxes
        for bbox in &self.bounding_boxes {
            all_points.extend_from_slice(&bbox.corners());
        }

        if !all_points.is_empty() {
            self.metadata.bounds = Some(PointCloud3D::calculate_bounds(&all_points));
        }
    }

    /// Clear all objects from scene
    pub fn clear(&mut self) {
        self.point_clouds.clear();
        self.meshes.clear();
        self.bounding_boxes.clear();
        self.metadata.num_objects = 0;
        self.metadata.bounds = None;
    }

    /// Get total number of objects
    pub fn num_objects(&self) -> usize {
        self.metadata.num_objects
    }

    /// Export scene to basic format
    pub fn export_summary(&self) -> String {
        format!(
            "Scene: {}\nPoint Clouds: {}\nMeshes: {}\nBounding Boxes: {}\nTotal Objects: {}",
            self.metadata.name,
            self.point_clouds.len(),
            self.meshes.len(),
            self.bounding_boxes.len(),
            self.num_objects()
        )
    }
}

impl Default for Scene3D {
    fn default() -> Self {
        Self::new("default_scene".to_string())
    }
}

/// Utility function to rotate a point around origin
fn rotate_point(point: [f32; 3], roll: f32, pitch: f32, yaw: f32) -> [f32; 3] {
    let [x, y, z] = point;

    // Rotation matrices
    let cos_roll = roll.cos();
    let sin_roll = roll.sin();
    let cos_pitch = pitch.cos();
    let sin_pitch = pitch.sin();
    let cos_yaw = yaw.cos();
    let sin_yaw = yaw.sin();

    // Apply rotations in order: yaw (Z), pitch (Y), roll (X)
    // First yaw
    let x1 = cos_yaw * x - sin_yaw * y;
    let y1 = sin_yaw * x + cos_yaw * y;
    let z1 = z;

    // Then pitch
    let x2 = cos_pitch * x1 + sin_pitch * z1;
    let y2 = y1;
    let z2 = -sin_pitch * x1 + cos_pitch * z1;

    // Finally roll
    let x3 = x2;
    let y3 = cos_roll * y2 - sin_roll * z2;
    let z3 = sin_roll * y2 + cos_roll * z2;

    [x3, y3, z3]
}

/// Utility function to compute cross product
fn cross_product(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point3d_creation() {
        let point = Point3D::new(1.0, 2.0, 3.0);
        assert_eq!(point.position(), [1.0, 2.0, 3.0]);
        assert!(point.color.is_none());
        assert!(point.normal.is_none());

        let colored_point = Point3D::with_color(1.0, 2.0, 3.0, [255, 0, 0]);
        assert_eq!(colored_point.color, Some([255, 0, 0]));
    }

    #[test]
    fn test_point3d_distance() {
        let p1 = Point3D::new(0.0, 0.0, 0.0);
        let p2 = Point3D::new(3.0, 4.0, 0.0);
        assert_eq!(p1.distance_to(&p2), 5.0);
    }

    #[test]
    fn test_bounding_box3d() {
        let bbox = BoundingBox3D::new(
            [0.0, 0.0, 0.0],
            [2.0, 2.0, 2.0],
            [0.0, 0.0, 0.0],
            "test".to_string(),
            0.95,
        );

        assert_eq!(bbox.volume(), 8.0);

        let point_inside = Point3D::new(0.5, 0.5, 0.5);
        let point_outside = Point3D::new(2.0, 2.0, 2.0);

        assert!(bbox.contains_point(point_inside));
        assert!(!bbox.contains_point(point_outside));
    }

    #[test]
    fn test_point_cloud() {
        let points = vec![
            Point3D::new(0.0, 0.0, 0.0),
            Point3D::new(1.0, 1.0, 1.0),
            Point3D::with_color(2.0, 2.0, 2.0, [255, 0, 0]),
        ];

        let mut cloud = PointCloud3D::new(points);
        assert_eq!(cloud.len(), 3);
        assert!(!cloud.is_empty());

        cloud.add_point(Point3D::new(3.0, 3.0, 3.0));
        assert_eq!(cloud.len(), 4);

        let removed = cloud.remove_point(0).unwrap();
        assert_eq!(removed.position(), [0.0, 0.0, 0.0]);
        assert_eq!(cloud.len(), 3);
    }

    #[test]
    fn test_mesh_creation() {
        let vertices = vec![
            Point3D::new(0.0, 0.0, 0.0),
            Point3D::new(1.0, 0.0, 0.0),
            Point3D::new(0.5, 1.0, 0.0),
        ];

        let faces = vec![Triangle3D::new(0, 1, 2)];

        let mut mesh = Mesh3D::new(vertices, faces);
        assert_eq!(mesh.metadata.num_vertices, 3);
        assert_eq!(mesh.metadata.num_faces, 1);

        mesh.compute_face_normals();
        assert!(mesh.faces[0].normal.is_some());
    }

    #[test]
    fn test_sphere_mesh() {
        let center = Point3D::new(0.0, 0.0, 0.0);
        let mesh = Mesh3D::create_sphere(center, 1.0, 10, 10);

        assert!(mesh.metadata.num_vertices > 0);
        assert!(mesh.metadata.num_faces > 0);
        assert_eq!(mesh.metadata.name, "sphere");
    }

    #[test]
    fn test_cube_mesh() {
        let center = Point3D::new(0.0, 0.0, 0.0);
        let mesh = Mesh3D::create_cube(center, 2.0);

        assert_eq!(mesh.metadata.num_vertices, 8);
        assert_eq!(mesh.metadata.num_faces, 12);
        assert_eq!(mesh.metadata.name, "cube");
    }

    #[test]
    fn test_scene3d() {
        let mut scene = Scene3D::new("test_scene".to_string());

        let points = vec![Point3D::new(0.0, 0.0, 0.0)];
        let cloud = PointCloud3D::new(points);
        scene.add_point_cloud(cloud);

        let center = Point3D::new(0.0, 0.0, 0.0);
        let mesh = Mesh3D::create_cube(center, 1.0);
        scene.add_mesh(mesh);

        assert_eq!(scene.num_objects(), 2);
        assert!(scene.metadata.bounds.is_some());
    }

    #[test]
    fn test_voxel_downsampling() {
        let points = vec![
            Point3D::new(0.1, 0.1, 0.1),
            Point3D::new(0.2, 0.2, 0.2),
            Point3D::new(1.1, 1.1, 1.1),
            Point3D::new(1.2, 1.2, 1.2),
        ];

        let cloud = PointCloud3D::new(points);
        let downsampled = cloud.voxel_downsample(1.0);

        // Should have 2 points after downsampling (one per voxel)
        assert_eq!(downsampled.len(), 2);
    }
}
