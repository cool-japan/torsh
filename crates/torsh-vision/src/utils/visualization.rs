//! 2D Visualization Functions for Image Processing
//!
//! This module provides visualization utilities for creating image grids, drawing annotations,
//! and rendering bounding boxes with labels. It focuses on 2D visualization tasks commonly
//! used in computer vision applications.

use crate::{Result, VisionError};
use image::{DynamicImage, GenericImageView};
use torsh_tensor::{creation, creation::zeros_mut, Tensor};

// Import text rendering functions from the parent utils module
use super::text_rendering::draw_simple_text;

/// Make a grid of images from tensor data
///
/// Creates a grid layout from a collection of image tensors, arranging them in rows and columns
/// with optional padding between images. This is useful for creating visualizations of image
/// batches, model outputs, or dataset samples.
///
/// # Arguments
/// * `tensors` - Slice of image tensors to arrange in a grid (each tensor should be 3D: C×H×W)
/// * `nrow` - Number of images per row in the grid
/// * `padding` - Padding in pixels between adjacent images
///
/// # Returns
/// A single tensor representing the image grid with shape (C, grid_height, grid_width)
///
/// # Example
/// ```
/// use torsh_vision::utils::visualization::make_grid;
/// use torsh_tensor::creation;
///
/// // Create sample tensors (3 channels, 64x64 each)
/// let tensors: Vec<_> = (0..8).map(|_| creation::rand(&[3, 64, 64]).unwrap()).collect();
///
/// // Create a 3×3 grid with 2 pixels padding
/// let grid = make_grid(&tensors, 3, 2)?;
/// println!("Grid shape: {:?}", grid.shape().dims());
/// ```
///
/// # Errors
/// Returns `VisionError` if:
/// - No tensors are provided
/// - Tensors have different shapes
/// - Tensors are not 3D (C×H×W format)
/// - Memory allocation fails during grid creation
pub fn make_grid(tensors: &[Tensor<f32>], nrow: usize, padding: usize) -> Result<Tensor<f32>> {
    if tensors.is_empty() {
        return Err(VisionError::TransformError(
            "No tensors provided".to_string(),
        ));
    }

    // Validate all tensors have the same shape
    let first_shape = tensors[0].shape();
    for tensor in tensors.iter().skip(1) {
        if tensor.shape() != first_shape {
            return Err(VisionError::InvalidArgument(
                "All tensors must have the same shape".to_string(),
            ));
        }
    }

    if first_shape.dims().len() != 3 {
        return Err(VisionError::InvalidShape(format!(
            "Expected 3D tensors (C, H, W), got {}D",
            first_shape.dims().len()
        )));
    }

    let (channels, height, width) = (
        first_shape.dims()[0],
        first_shape.dims()[1],
        first_shape.dims()[2],
    );
    let num_images = tensors.len();
    let ncol = (num_images + nrow - 1) / nrow; // Ceiling division
    let actual_nrow = (num_images + ncol - 1) / ncol; // Actual number of rows

    // Calculate grid dimensions
    let grid_height = actual_nrow * height + (actual_nrow - 1) * padding;
    let grid_width = ncol * width + (ncol - 1) * padding;

    // Create output grid tensor (mutable storage for element-wise writes)
    let grid = zeros_mut(&[channels, grid_height, grid_width]);

    // Fill the grid
    for (idx, tensor) in tensors.iter().enumerate() {
        let row = idx / ncol;
        let col = idx % ncol;

        let start_y = row * (height + padding);
        let start_x = col * (width + padding);

        // Copy tensor data to grid
        for c in 0..channels {
            for y in 0..height {
                for x in 0..width {
                    let pixel_val = tensor.get(&[c, y, x])?;
                    grid.set(&[c, start_y + y, start_x + x], pixel_val)?;
                }
            }
        }
    }

    Ok(grid)
}

/// Draw bounding boxes with labels on an image
///
/// Renders bounding boxes with optional labels and confidence scores on a mutable image.
/// This function is commonly used for visualizing object detection results, annotation overlays,
/// or region proposals.
///
/// # Arguments
/// * `image` - Mutable reference to the image to draw on
/// * `boxes` - Tensor containing bounding box coordinates in format [N, 4] where each box is [x1, y1, x2, y2]
/// * `labels` - Optional slice of string labels for each bounding box
/// * `scores` - Optional tensor of confidence scores for each bounding box \[N\]
/// * `colors` - Optional slice of RGB color tuples for each box. Cycles through default colors if None
///
/// # Returns
/// `Result<()>` indicating success or failure
///
/// # Box Format
/// Bounding boxes should be in (x1, y1, x2, y2) format where:
/// - (x1, y1) is the top-left corner
/// - (x2, y2) is the bottom-right corner
/// - Coordinates are in image pixel space
///
/// # Example
/// ```
/// use torsh_vision::utils::visualization::draw_bounding_boxes;
/// use torsh_tensor::creation;
/// use image::DynamicImage;
///
/// let mut image: DynamicImage = load_test_image();
/// let boxes = creation::tensor_from_vec(vec![10.0, 10.0, 100.0, 100.0], &[1, 4]).unwrap();
/// let labels = vec!["object".to_string()];
/// let colors = vec![(255, 0, 0)]; // Red box
///
/// draw_bounding_boxes(&mut image, &boxes, Some(&labels), None, Some(&colors))?;
/// ```
///
/// # Errors
/// Returns `VisionError` if:
/// - Boxes tensor is not 2D with shape [N, 4]
/// - Tensor access fails during coordinate retrieval
/// - Image conversion or pixel manipulation fails
pub fn draw_bounding_boxes(
    image: &mut DynamicImage,
    boxes: &Tensor<f32>,
    labels: Option<&[String]>,
    scores: Option<&Tensor<f32>>,
    colors: Option<&[(u8, u8, u8)]>,
) -> Result<()> {
    let box_shape = boxes.shape();
    if box_shape.dims().len() != 2 || box_shape.dims()[1] != 4 {
        return Err(VisionError::InvalidShape(format!(
            "Expected boxes tensor of shape [N, 4], got [{}, {}]",
            box_shape.dims()[0],
            box_shape.dims()[1]
        )));
    }

    let num_boxes = box_shape.dims()[0];
    let default_colors = vec![
        (255, 0, 0),   // Red
        (0, 255, 0),   // Green
        (0, 0, 255),   // Blue
        (255, 255, 0), // Yellow
        (255, 0, 255), // Magenta
        (0, 255, 255), // Cyan
    ];
    let box_colors = colors.unwrap_or(&default_colors);

    // Convert to RGB8 for drawing
    let mut rgb_image = image.to_rgb8();

    for i in 0..num_boxes {
        // Get box coordinates [x1, y1, x2, y2]
        let x1 = boxes.get(&[i, 0])? as u32;
        let y1 = boxes.get(&[i, 1])? as u32;
        let x2 = boxes.get(&[i, 2])? as u32;
        let y2 = boxes.get(&[i, 3])? as u32;

        let color = box_colors[i % box_colors.len()];
        let rgb_color = image::Rgb([color.0, color.1, color.2]);

        // Draw bounding box edges (simplified implementation)
        // Top and bottom edges
        for x in x1..=x2 {
            if y1 < rgb_image.height() && x < rgb_image.width() {
                rgb_image.put_pixel(x, y1, rgb_color);
            }
            if y2 < rgb_image.height() && x < rgb_image.width() {
                rgb_image.put_pixel(x, y2, rgb_color);
            }
        }

        // Left and right edges
        for y in y1..=y2 {
            if x1 < rgb_image.width() && y < rgb_image.height() {
                rgb_image.put_pixel(x1, y, rgb_color);
            }
            if x2 < rgb_image.width() && y < rgb_image.height() {
                rgb_image.put_pixel(x2, y, rgb_color);
            }
        }

        // Draw text labels and scores (basic implementation)
        if let Some(label_texts) = labels {
            if i < label_texts.len() {
                let label_text = &label_texts[i];
                let score_text = if let Some(score_tensor) = scores {
                    if i < score_tensor.shape().dims()[0] {
                        let score = score_tensor.get(&[i])?;
                        format!("{}: {:.2}", label_text, score)
                    } else {
                        label_text.clone()
                    }
                } else {
                    label_text.clone()
                };

                // Basic text rendering using pixels (simple approach)
                // This is a minimal implementation - for production use, consider imageproc or similar crates
                draw_simple_text(
                    &mut rgb_image,
                    &score_text,
                    x1 + 2,
                    y1.saturating_sub(15),
                    rgb_color,
                );
            }
        }
    }

    *image = DynamicImage::ImageRgb8(rgb_image);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, RgbImage};
    use torsh_tensor::creation;

    #[test]
    fn test_make_grid_single_tensor() {
        let tensor = creation::ones(&[3, 32, 32]).unwrap();
        let tensors = vec![tensor];
        let grid = make_grid(&tensors, 1, 0).unwrap();

        assert_eq!(grid.shape().dims(), &[3, 32, 32]);
    }

    #[test]
    fn test_make_grid_multiple_tensors() {
        let tensors: Vec<_> = (0..4)
            .map(|_| creation::ones(&[3, 16, 16]).unwrap())
            .collect();
        let grid = make_grid(&tensors, 2, 2).unwrap();

        // 2x2 grid with 2 pixel padding: (2*16 + 1*2) x (2*16 + 1*2) = 34 x 34
        assert_eq!(grid.shape().dims(), &[3, 34, 34]);
    }

    #[test]
    fn test_make_grid_empty_tensors() {
        let tensors: Vec<Tensor<f32>> = vec![];
        let result = make_grid(&tensors, 1, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_make_grid_mismatched_shapes() {
        let tensor1 = creation::ones(&[3, 32, 32]).unwrap();
        let tensor2 = creation::ones(&[3, 16, 16]).unwrap();
        let tensors = vec![tensor1, tensor2];
        let result = make_grid(&tensors, 1, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_make_grid_invalid_dimensions() {
        let tensor = creation::ones(&[32, 32]).unwrap(); // 2D instead of 3D
        let tensors = vec![tensor];
        let result = make_grid(&tensors, 1, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_draw_bounding_boxes_valid_input() -> Result<()> {
        let mut image = DynamicImage::ImageRgb8(RgbImage::new(100, 100));
        let boxes_data = vec![10.0, 10.0, 50.0, 50.0, 60.0, 60.0, 90.0, 90.0];
        let boxes = creation::tensor_1d(&boxes_data)?.reshape(&[2, 4])?;

        let labels = vec!["box1".to_string(), "box2".to_string()];
        let result = draw_bounding_boxes(&mut image, &boxes, Some(&labels), None, None);
        assert!(result.is_ok());
        Ok(())
    }

    #[test]
    fn test_draw_bounding_boxes_invalid_shape() {
        let mut image = DynamicImage::ImageRgb8(RgbImage::new(100, 100));
        let boxes = creation::ones(&[2, 3]).unwrap(); // Wrong shape, should be [N, 4]

        let result = draw_bounding_boxes(&mut image, &boxes, None, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_draw_bounding_boxes_with_scores() -> Result<()> {
        let mut image = DynamicImage::ImageRgb8(RgbImage::new(100, 100));
        let boxes = creation::tensor_1d(&[10.0, 10.0, 50.0, 50.0])?.reshape(&[1, 4])?;
        let scores = creation::tensor_1d(&[0.95])?;
        let labels = vec!["object".to_string()];

        let result = draw_bounding_boxes(&mut image, &boxes, Some(&labels), Some(&scores), None);
        assert!(result.is_ok());
        Ok(())
    }

    #[test]
    fn test_draw_bounding_boxes_custom_colors() -> Result<()> {
        let mut image = DynamicImage::ImageRgb8(RgbImage::new(100, 100));
        let boxes = creation::tensor_1d(&[10.0, 10.0, 50.0, 50.0])?.reshape(&[1, 4])?;
        let colors = vec![(128, 64, 192)];

        let result = draw_bounding_boxes(&mut image, &boxes, None, None, Some(&colors));
        assert!(result.is_ok());
        Ok(())
    }
}
