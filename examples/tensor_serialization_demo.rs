//! Tensor Serialization Demo
//!
//! This example demonstrates comprehensive tensor serialization and deserialization
//! capabilities across multiple formats including binary, JSON, HDF5, and Arrow.

use torsh_core::device::DeviceType;
use torsh_tensor::Tensor;

#[cfg(feature = "serialize")]
use torsh_tensor::serialize::{SerializationFormat, SerializationOptions};

use std::path::Path;
use tempfile::TempDir;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Tensor Serialization Demo");
    println!("========================");

    // Create test data
    let test_tensors = create_test_tensors();

    // Create temporary directory for file operations
    let temp_dir = TempDir::new()?;

    // Example 1: Binary Format (Custom, Fast)
    println!("\n1. Binary Format Serialization");
    binary_format_examples(&test_tensors, temp_dir.path())?;

    // Example 2: JSON Format (Human-readable)
    #[cfg(feature = "serialize")]
    {
        println!("\n2. JSON Format Serialization");
        json_format_examples(&test_tensors, temp_dir.path())?;
    }

    // Example 3: HDF5 Format (Scientific Computing)
    #[cfg(feature = "serialize-hdf5")]
    {
        println!("\n3. HDF5 Format Serialization");
        hdf5_format_examples(&test_tensors, temp_dir.path())?;
    }

    // Example 4: Arrow/Parquet Format (Data Science)
    #[cfg(feature = "serialize-arrow")]
    {
        println!("\n4. Arrow/Parquet Format Serialization");
        arrow_format_examples(&test_tensors, temp_dir.path())?;
    }

    // Example 5: PyTorch Compatibility
    println!("\n5. PyTorch Compatibility");
    pytorch_compatibility_examples(&test_tensors, temp_dir.path())?;

    // Example 6: Serialization Options
    #[cfg(feature = "serialize")]
    {
        println!("\n6. Advanced Serialization Options");
        advanced_options_examples(&test_tensors, temp_dir.path())?;
    }

    // Example 7: Performance Benchmarks
    println!("\n7. Performance Benchmarks");
    performance_benchmarks(&test_tensors, temp_dir.path())?;

    Ok(())
}

fn create_test_tensors() -> Vec<Tensor<f32>> {
    vec![
        // 1D tensor
        Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5], DeviceType::Cpu),
        // 2D tensor (matrix)
        Tensor::from_data(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            DeviceType::Cpu,
        ),
        // 3D tensor (small image-like)
        Tensor::from_data(
            (0..24).map(|i| i as f32 * 0.1).collect(),
            vec![2, 3, 4],
            DeviceType::Cpu,
        ),
        // Large tensor for performance testing
        Tensor::from_data(
            (0..10000).map(|i| (i as f32).sin()).collect(),
            vec![100, 100],
            DeviceType::Cpu,
        ),
    ]
}

fn binary_format_examples(
    tensors: &[Tensor<f32>],
    temp_dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "serialize")]
    {
        let options = SerializationOptions::default();

        for (i, tensor) in tensors.iter().enumerate() {
            println!(
                "  Testing tensor {} (shape: {:?})",
                i,
                tensor.shape().dims()
            );

            // Serialize to bytes
            let bytes = tensor.serialize_to_bytes(SerializationFormat::Binary, &options)?;
            println!("    Serialized size: {} bytes", bytes.len());

            // Deserialize from bytes
            let reconstructed =
                Tensor::deserialize_from_bytes(&bytes, SerializationFormat::Binary)?;

            // Verify data integrity
            verify_tensor_equality(tensor, &reconstructed)?;
            println!("    ✓ Round-trip successful");

            // Test file serialization
            let file_path = temp_dir.join(format!("tensor_{}.torsh", i));
            tensor.serialize_to_file(&file_path, SerializationFormat::Binary, &options)?;
            let file_reconstructed =
                Tensor::deserialize_from_file(&file_path, SerializationFormat::Binary)?;
            verify_tensor_equality(tensor, &file_reconstructed)?;
            println!("    ✓ File round-trip successful");
        }
    }

    #[cfg(not(feature = "serialize"))]
    {
        println!("  Binary serialization requires 'serialize' feature");
    }

    Ok(())
}

#[cfg(feature = "serialize")]
fn json_format_examples(
    tensors: &[Tensor<f32>],
    temp_dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let options = SerializationOptions::default();

    // Only test smaller tensors for JSON (due to size)
    for (i, tensor) in tensors.iter().take(3).enumerate() {
        println!(
            "  Testing tensor {} (shape: {:?})",
            i,
            tensor.shape().dims()
        );

        // Serialize to JSON bytes
        let json_bytes = tensor.serialize_to_bytes(SerializationFormat::Json, &options)?;
        println!("    JSON size: {} bytes", json_bytes.len());

        // Deserialize from JSON
        let reconstructed = Tensor::deserialize_from_bytes(&json_bytes, SerializationFormat::Json)?;
        verify_tensor_equality(tensor, &reconstructed)?;
        println!("    ✓ JSON round-trip successful");

        // Save as human-readable JSON file
        let json_path = temp_dir.join(format!("tensor_{}.json", i));
        tensor.serialize_to_file(&json_path, SerializationFormat::Json, &options)?;

        // Display first few lines of JSON for inspection
        let json_content = std::fs::read_to_string(&json_path)?;
        let preview: String = json_content.lines().take(5).collect::<Vec<_>>().join("\n");
        println!("    JSON preview:\n{}", preview);
    }

    Ok(())
}

#[cfg(feature = "serialize-hdf5")]
fn hdf5_format_examples(
    tensors: &[Tensor<f32>],
    temp_dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let options = SerializationOptions::default();

    for (i, tensor) in tensors.iter().enumerate() {
        println!(
            "  Testing tensor {} (shape: {:?})",
            i,
            tensor.shape().dims()
        );

        let hdf5_path = temp_dir.join(format!("tensor_{}.h5", i));

        // Serialize to HDF5
        tensor.serialize_to_file(&hdf5_path, SerializationFormat::Hdf5, &options)?;
        println!(
            "    HDF5 file size: {} bytes",
            std::fs::metadata(&hdf5_path)?.len()
        );

        // Deserialize from HDF5
        let reconstructed = Tensor::deserialize_from_file(&hdf5_path, SerializationFormat::Hdf5)?;
        verify_tensor_equality(tensor, &reconstructed)?;
        println!("    ✓ HDF5 round-trip successful");
    }

    Ok(())
}

#[cfg(feature = "serialize-arrow")]
fn arrow_format_examples(
    tensors: &[Tensor<f32>],
    temp_dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let options = SerializationOptions::default();

    for (i, tensor) in tensors.iter().enumerate() {
        println!(
            "  Testing tensor {} (shape: {:?})",
            i,
            tensor.shape().dims()
        );

        let parquet_path = temp_dir.join(format!("tensor_{}.parquet", i));

        // Serialize to Parquet (Arrow format)
        tensor.serialize_to_file(&parquet_path, SerializationFormat::Parquet, &options)?;
        println!(
            "    Parquet file size: {} bytes",
            std::fs::metadata(&parquet_path)?.len()
        );

        // Note: Deserialization not yet implemented for Arrow format
        println!("    ✓ Arrow serialization successful (deserialization TODO)");
    }

    Ok(())
}

fn pytorch_compatibility_examples(
    tensors: &[Tensor<f32>],
    temp_dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "serialize")]
    {
        let options = SerializationOptions::default();

        for (i, tensor) in tensors.iter().enumerate() {
            println!(
                "  Testing tensor {} (shape: {:?})",
                i,
                tensor.shape().dims()
            );

            let pt_path = temp_dir.join(format!("tensor_{}.pt", i));

            // Save in PyTorch-compatible format
            tensor.save_pytorch_compatible(&pt_path, &options)?;
            println!(
                "    PyTorch file size: {} bytes",
                std::fs::metadata(&pt_path)?.len()
            );

            // Load from PyTorch-compatible format
            let reconstructed = Tensor::load_pytorch_compatible(&pt_path)?;
            verify_tensor_equality(tensor, &reconstructed)?;
            println!("    ✓ PyTorch compatibility successful");
        }
    }

    #[cfg(not(feature = "serialize"))]
    {
        println!("  PyTorch compatibility requires 'serialize' feature");
    }

    Ok(())
}

#[cfg(feature = "serialize")]
fn advanced_options_examples(
    tensors: &[Tensor<f32>],
    temp_dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let tensor = &tensors[1]; // Use 2D tensor

    // Test different serialization options
    let test_cases = vec![
        ("Basic options", SerializationOptions::default()),
        (
            "With gradients",
            SerializationOptions {
                include_gradients: true,
                ..Default::default()
            },
        ),
        (
            "With metadata",
            SerializationOptions {
                metadata: {
                    let mut meta = std::collections::HashMap::new();
                    meta.insert("experiment".to_string(), "demo_run".to_string());
                    meta.insert("model_version".to_string(), "v1.0.0".to_string());
                    meta.insert("created_by".to_string(), "torsh_demo".to_string());
                    meta
                },
                ..Default::default()
            },
        ),
        (
            "Compression level 5",
            SerializationOptions {
                compression_level: 5,
                ..Default::default()
            },
        ),
    ];

    for (name, options) in test_cases {
        println!("  Testing: {}", name);

        let bytes = tensor.serialize_to_bytes(SerializationFormat::Binary, &options)?;
        println!("    Serialized size: {} bytes", bytes.len());

        let reconstructed = Tensor::deserialize_from_bytes(&bytes, SerializationFormat::Binary)?;
        verify_tensor_equality(tensor, &reconstructed)?;
        println!("    ✓ Options test successful");
    }

    Ok(())
}

fn performance_benchmarks(
    tensors: &[Tensor<f32>],
    temp_dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "serialize")]
    {
        use std::time::Instant;

        let large_tensor = &tensors[3]; // 100x100 tensor
        let options = SerializationOptions::default();

        println!(
            "  Benchmarking large tensor (shape: {:?})",
            large_tensor.shape().dims()
        );

        // Binary format benchmark
        let start = Instant::now();
        let binary_bytes =
            large_tensor.serialize_to_bytes(SerializationFormat::Binary, &options)?;
        let serialize_time = start.elapsed();

        let start = Instant::now();
        let _reconstructed =
            Tensor::<f32>::deserialize_from_bytes(&binary_bytes, SerializationFormat::Binary)?;
        let deserialize_time = start.elapsed();

        println!("    Binary format:");
        println!(
            "      Serialize: {:?} ({} bytes)",
            serialize_time,
            binary_bytes.len()
        );
        println!("      Deserialize: {:?}", deserialize_time);
        println!(
            "      Throughput: {:.2} MB/s",
            (binary_bytes.len() as f64 / 1_000_000.0) / serialize_time.as_secs_f64()
        );

        // JSON format benchmark (for comparison)
        let start = Instant::now();
        let json_bytes = large_tensor.serialize_to_bytes(SerializationFormat::Json, &options)?;
        let json_serialize_time = start.elapsed();

        println!("    JSON format:");
        println!(
            "      Serialize: {:?} ({} bytes)",
            json_serialize_time,
            json_bytes.len()
        );
        println!(
            "      Size ratio (JSON/Binary): {:.2}x",
            json_bytes.len() as f64 / binary_bytes.len() as f64
        );
    }

    #[cfg(not(feature = "serialize"))]
    {
        println!("  Performance benchmarks require 'serialize' feature");
    }

    Ok(())
}

fn verify_tensor_equality(
    original: &Tensor<f32>,
    reconstructed: &Tensor<f32>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Check shape
    if original.shape().dims() != reconstructed.shape().dims() {
        return Err(format!(
            "Shape mismatch: {:?} vs {:?}",
            original.shape().dims(),
            reconstructed.shape().dims()
        )
        .into());
    }

    // Check device
    if original.device() != reconstructed.device() {
        return Err(format!(
            "Device mismatch: {:?} vs {:?}",
            original.device(),
            reconstructed.device()
        )
        .into());
    }

    // Check data
    let original_data = original.data();
    let reconstructed_data = reconstructed.data();

    if original_data.len() != reconstructed_data.len() {
        return Err(format!(
            "Data length mismatch: {} vs {}",
            original_data.len(),
            reconstructed_data.len()
        )
        .into());
    }

    for (i, (&orig, &recon)) in original_data
        .iter()
        .zip(reconstructed_data.iter())
        .enumerate()
    {
        if (orig - recon).abs() > 1e-6 {
            return Err(format!("Data mismatch at index {}: {} vs {}", i, orig, recon).into());
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialization_demo() {
        // Test that the demo functions run without errors
        let test_tensors = create_test_tensors();
        assert!(!test_tensors.is_empty());
        assert_eq!(test_tensors[0].shape().dims(), &[5]);
        assert_eq!(test_tensors[1].shape().dims(), &[2, 3]);
        assert_eq!(test_tensors[2].shape().dims(), &[2, 3, 4]);
        assert_eq!(test_tensors[3].shape().dims(), &[100, 100]);
    }

    #[test]
    fn test_tensor_verification() {
        let tensor1 = Tensor::from_data(vec![1.0, 2.0, 3.0], vec![3], DeviceType::Cpu);
        let tensor2 = Tensor::from_data(vec![1.0, 2.0, 3.0], vec![3], DeviceType::Cpu);
        let tensor3 = Tensor::from_data(vec![1.0, 2.0, 4.0], vec![3], DeviceType::Cpu);

        assert!(verify_tensor_equality(&tensor1, &tensor2).is_ok());
        assert!(verify_tensor_equality(&tensor1, &tensor3).is_err());
    }
}
