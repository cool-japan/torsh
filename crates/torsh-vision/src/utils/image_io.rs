//! Image I/O utilities module
//!
//! This module provides clean delegation functions for image I/O operations,
//! forwarding calls to the unified I/O system in `crate::io::global`.

use crate::Result;
use image::DynamicImage;
use std::path::Path;
use torsh_tensor::Tensor;

/// Save a tensor as an image using the unified I/O system
///
/// This function delegates to the global I/O system to save a tensor as an image file.
/// The tensor is expected to be in (C, H, W) format where C is channels (1 or 3),
/// H is height, and W is width.
///
/// # Arguments
/// * `tensor` - The tensor to save as an image (C, H, W format)
/// * `path` - Output file path
/// * `normalize` - Whether to normalize the tensor values to [0, 1] range
///
/// # Returns
/// * `Ok(())` if the image was saved successfully
/// * `Err(VisionError)` if there was an error during saving
///
/// # Example
/// ```rust
/// use torsh_vision::utils::image_io::save_tensor_as_image;
/// use torsh_tensor::creation;
///
/// let tensor = creation::rand(&[3, 224, 224]).unwrap();
/// save_tensor_as_image(&tensor, "output.png", true)?;
/// ```
pub fn save_tensor_as_image<P: AsRef<Path>>(
    tensor: &Tensor<f32>,
    path: P,
    normalize: bool,
) -> Result<()> {
    crate::io::global::save_tensor_as_image(tensor, path, normalize)
}

/// Load multiple images from a directory using the unified I/O system
///
/// This function delegates to the global I/O system to load all images from a directory.
/// Returns a vector of tuples containing the loaded image and its filename.
///
/// # Arguments
/// * `dir_path` - Path to the directory containing images
///
/// # Returns
/// * `Ok(Vec<(DynamicImage, String)>)` - Vector of (image, filename) pairs
/// * `Err(VisionError)` if there was an error during loading
///
/// # Example
/// ```rust
/// use torsh_vision::utils::image_io::load_images_from_dir;
///
/// let images = load_images_from_dir("./images")?;
/// println!("Loaded {} images", images.len());
/// ```
pub fn load_images_from_dir<P: AsRef<Path>>(dir_path: P) -> Result<Vec<(DynamicImage, String)>> {
    crate::io::global::load_images_from_dir(dir_path)
}
