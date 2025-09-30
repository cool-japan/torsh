//! Image Segmentation Example using Spectral Clustering
//!
//! This example demonstrates how to use spectral clustering for image segmentation,
//! a common computer vision task where we want to partition an image into meaningful regions.

use torsh_cluster::{
    algorithms::SpectralClustering, evaluation::metrics::silhouette_score, traits::Fit,
};
use torsh_tensor::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🖼️  Image Segmentation with Spectral Clustering");
    println!("===============================================");

    // Simulate image data (in practice, this would come from actual image pixels)
    let (image_features, image_info) = create_synthetic_image_data()?;

    println!("📊 Image Information:");
    println!(
        "  • Image size: {}x{} pixels",
        image_info.width, image_info.height
    );
    println!("  • Total pixels: {}", image_info.width * image_info.height);
    println!(
        "  • Feature dimensions: {}",
        image_features.shape().dims()[1]
    );

    // Demonstrate different numbers of segments
    let segment_counts = [3, 5, 8];

    for &n_segments in &segment_counts {
        segment_image(&image_features, &image_info, n_segments)?;
    }

    println!("\n✅ Image segmentation demonstration completed!");
    println!("💡 In practice, you would:");
    println!("   • Load actual image data (RGB values, gradients, textures)");
    println!("   • Extract meaningful features (SIFT, color histograms, etc.)");
    println!("   • Apply post-processing to smooth segment boundaries");
    println!("   • Visualize the segmented image");

    Ok(())
}

/// Perform image segmentation using spectral clustering
fn segment_image(
    features: &Tensor,
    info: &ImageInfo,
    n_segments: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🎯 Segmenting image into {} regions", n_segments);
    println!(
        "{}{}",
        "=".repeat(35),
        "=".repeat(n_segments.to_string().len())
    );

    let spectral = SpectralClustering::new(n_segments)
        .gamma(1.0) // RBF kernel parameter
        .eigen_tolerance(1e-4)
        .random_state(42);

    let start = std::time::Instant::now();
    let result = spectral.fit(features)?;
    let duration = start.elapsed();

    // Evaluate segmentation quality
    let silhouette = silhouette_score(features, &result.labels)?;

    println!("  ✓ Segmentation completed");
    println!("  ✓ Embedding Success: {}", result.embedding_success);
    println!("  ✓ Silhouette Score: {:.4}", silhouette);
    println!("  ⏱️  Processing Time: {:?}", duration);

    // Analyze segment statistics
    analyze_segments(&result.labels, info)?;

    Ok(())
}

/// Analyze the generated segments
fn analyze_segments(labels: &Tensor, info: &ImageInfo) -> Result<(), Box<dyn std::error::Error>> {
    let labels_vec = labels
        .to_vec()
        .map_err(|e| format!("Failed to convert labels: {:?}", e))?;

    // Count pixels in each segment
    let mut segment_counts = std::collections::HashMap::new();
    for &label in &labels_vec {
        *segment_counts.entry(label as i32).or_insert(0) += 1;
    }

    println!("  📊 Segment Analysis:");
    for (segment_id, count) in segment_counts.iter() {
        let percentage = (*count as f64 / labels_vec.len() as f64) * 100.0;
        println!(
            "     Segment {}: {} pixels ({:.1}%)",
            segment_id, count, percentage
        );
    }

    // Simulate segment properties (in practice, these would be computed from actual image data)
    println!("  🎨 Segment Properties:");
    for segment_id in segment_counts.keys() {
        let avg_intensity = simulate_segment_intensity(*segment_id);
        let compactness = simulate_segment_compactness(*segment_id, info);
        println!(
            "     Segment {}: Avg Intensity={:.2}, Compactness={:.3}",
            segment_id, avg_intensity, compactness
        );
    }

    Ok(())
}

/// Create synthetic image data with different regions
fn create_synthetic_image_data() -> Result<(Tensor, ImageInfo), Box<dyn std::error::Error>> {
    let width = 64;
    let height = 64;
    let total_pixels = width * height;

    let mut features = Vec::with_capacity(total_pixels * 5); // 5 features per pixel

    for y in 0..height {
        for x in 0..width {
            // Feature 1: X coordinate (normalized)
            let x_norm = x as f32 / width as f32;

            // Feature 2: Y coordinate (normalized)
            let y_norm = y as f32 / height as f32;

            // Feature 3: Distance from center
            let center_x = width as f32 / 2.0;
            let center_y = height as f32 / 2.0;
            let dist_from_center = ((x as f32 - center_x).powi(2) + (y as f32 - center_y).powi(2))
                .sqrt()
                / (width as f32 / 2.0);

            // Feature 4: Simulated intensity (checkerboard pattern)
            let intensity = if (x / 8 + y / 8) % 2 == 0 { 0.8 } else { 0.3 };

            // Feature 5: Simulated texture (gradient)
            let texture = (x_norm + y_norm) / 2.0;

            features.extend_from_slice(&[x_norm, y_norm, dist_from_center, intensity, texture]);
        }
    }

    let features_tensor = Tensor::from_vec(features, &[total_pixels, 5])?;
    let image_info = ImageInfo { width, height };

    Ok((features_tensor, image_info))
}

/// Simulate average intensity for a segment
fn simulate_segment_intensity(segment_id: i32) -> f32 {
    // Simulate different intensities for different segments
    match segment_id % 5 {
        0 => 0.2,  // Dark region
        1 => 0.5,  // Medium region
        2 => 0.8,  // Bright region
        3 => 0.35, // Medium-dark region
        _ => 0.65, // Medium-bright region
    }
}

/// Simulate compactness measure for a segment
fn simulate_segment_compactness(segment_id: i32, info: &ImageInfo) -> f32 {
    // Simulate different compactness values
    let base_compactness = 0.7;
    let variation = (segment_id as f32 * 0.1) % 0.3;
    (base_compactness + variation).min(1.0)
}

/// Information about the synthetic image
#[derive(Debug)]
struct ImageInfo {
    width: usize,
    height: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthetic_image_creation() -> Result<(), Box<dyn std::error::Error>> {
        let (features, info) = create_synthetic_image_data()?;

        assert_eq!(features.shape().dims()[0], info.width * info.height);
        assert_eq!(features.shape().dims()[1], 5); // 5 features per pixel

        Ok(())
    }

    #[test]
    fn test_image_segmentation() -> Result<(), Box<dyn std::error::Error>> {
        let (features, info) = create_synthetic_image_data()?;

        let spectral = SpectralClustering::new(3).random_state(42);
        let result = spectral.fit(&features)?;

        // Check that we get the right number of labels
        assert_eq!(result.labels.shape().dims()[0], info.width * info.height);

        // Verify labels are in valid range
        let labels_vec = result.labels.to_vec()?;
        for &label in &labels_vec {
            assert!(label >= 0.0 && label < 3.0);
        }

        Ok(())
    }
}
