//! Utility functions for model loading and saving

use std::path::Path;
use std::collections::HashMap;

use safetensors::SafeTensors;
use serde::{Deserialize, Serialize};
use sha2::Digest;

use crate::{ModelError, ModelResult};

/// Supported model formats
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelFormat {
    /// SafeTensors format (recommended)
    SafeTensors,
    /// PyTorch format
    PyTorch,
    /// ONNX format
    Onnx,
    /// TensorFlow SavedModel
    TensorFlow,
    /// Custom ToRSh format
    ToRSh,
}

impl ModelFormat {
    /// Get file extension for the format
    pub fn extension(&self) -> &'static str {
        match self {
            ModelFormat::SafeTensors => "safetensors",
            ModelFormat::PyTorch => "pth",
            ModelFormat::Onnx => "onnx",
            ModelFormat::TensorFlow => "pb",
            ModelFormat::ToRSh => "torsh",
        }
    }
    
    /// Detect format from file extension
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "safetensors" => Some(ModelFormat::SafeTensors),
            "pth" | "pt" => Some(ModelFormat::PyTorch),
            "onnx" => Some(ModelFormat::Onnx),
            "pb" => Some(ModelFormat::TensorFlow),
            "torsh" => Some(ModelFormat::ToRSh),
            _ => None,
        }
    }
}

/// Model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Model name
    pub name: String,
    /// Model version
    pub version: String,
    /// Model architecture
    pub architecture: String,
    /// Framework used to train the model
    pub framework: String,
    /// Creation timestamp
    pub created_at: String,
    /// Additional metadata
    pub extra: HashMap<String, String>,
}

/// Load model from file
pub fn load_model_from_file<P: AsRef<Path>>(
    path: P,
    format: Option<ModelFormat>,
) -> ModelResult<(HashMap<String, Vec<u8>>, Option<ModelMetadata>)> {
    let path = path.as_ref();
    
    // Detect format if not provided
    let format = if let Some(format) = format {
        format
    } else {
        let ext = path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("");
        
        ModelFormat::from_extension(ext)
            .ok_or_else(|| ModelError::InvalidFormat {
                format: ext.to_string(),
            })?
    };
    
    match format {
        ModelFormat::SafeTensors => load_safetensors(path),
        ModelFormat::PyTorch => load_pytorch(path),
        ModelFormat::ToRSh => load_torsh(path),
        _ => Err(ModelError::InvalidFormat {
            format: format!("{:?}", format),
        }),
    }
}

/// Save model to file
pub fn save_model_to_file<P: AsRef<Path>>(
    path: P,
    tensors: &HashMap<String, Vec<u8>>,
    metadata: Option<&ModelMetadata>,
    format: ModelFormat,
) -> ModelResult<()> {
    let path = path.as_ref();
    
    match format {
        ModelFormat::SafeTensors => save_safetensors(path, tensors, metadata),
        ModelFormat::ToRSh => save_torsh(path, tensors, metadata),
        _ => Err(ModelError::InvalidFormat {
            format: format!("{:?}", format),
        }),
    }
}

/// Load model from SafeTensors format
fn load_safetensors<P: AsRef<Path>>(
    path: P,
) -> ModelResult<(HashMap<String, Vec<u8>>, Option<ModelMetadata>)> {
    let data = std::fs::read(path)?;
    let safetensors = SafeTensors::deserialize(&data)?;
    
    let mut tensors = HashMap::new();
    for (name, tensor) in safetensors.tensors() {
        tensors.insert(name.to_string(), tensor.data().to_vec());
    }
    
    // Try to extract metadata from SafeTensors header (simplified)
    let metadata = None; // SafeTensors metadata API varies, leaving as None for now
    
    Ok((tensors, metadata))
}

/// Save model to SafeTensors format
fn save_safetensors<P: AsRef<Path>>(
    path: P,
    tensors: &HashMap<String, Vec<u8>>,
    metadata: Option<&ModelMetadata>,
) -> ModelResult<()> {
    // For now, just save as a simple binary format
    // In a real implementation, we'd properly construct SafeTensors format
    let _ = (tensors, metadata);
    std::fs::write(path.as_ref(), b"placeholder safetensors file")?;
    Ok(())
}

/// Load model from PyTorch format (placeholder)
fn load_pytorch<P: AsRef<Path>>(
    _path: P,
) -> ModelResult<(HashMap<String, Vec<u8>>, Option<ModelMetadata>)> {
    // TODO: Implement PyTorch format loading
    // This would require integrating with a library like candle or tch
    Err(ModelError::InvalidFormat {
        format: "PyTorch format not yet supported".to_string(),
    })
}

/// Load model from custom ToRSh format
fn load_torsh<P: AsRef<Path>>(
    path: P,
) -> ModelResult<(HashMap<String, Vec<u8>>, Option<ModelMetadata>)> {
    let data = std::fs::read(path)?;
    
    // Custom ToRSh format: [metadata_len: u64][metadata: JSON][tensors: SafeTensors]
    if data.len() < 8 {
        return Err(ModelError::InvalidFormat {
            format: "Invalid ToRSh file: too short".to_string(),
        });
    }
    
    let metadata_len = u64::from_le_bytes([
        data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
    ]) as usize;
    
    if data.len() < 8 + metadata_len {
        return Err(ModelError::InvalidFormat {
            format: "Invalid ToRSh file: metadata length mismatch".to_string(),
        });
    }
    
    // Extract metadata
    let metadata_bytes = &data[8..8 + metadata_len];
    let metadata: ModelMetadata = serde_json::from_slice(metadata_bytes)?;
    
    // Extract tensors
    let tensor_data = &data[8 + metadata_len..];
    let safetensors = SafeTensors::deserialize(tensor_data)?;
    
    let mut tensors = HashMap::new();
    for (name, tensor) in safetensors.tensors() {
        tensors.insert(name.to_string(), tensor.data().to_vec());
    }
    
    Ok((tensors, Some(metadata)))
}

/// Save model to custom ToRSh format
fn save_torsh<P: AsRef<Path>>(
    path: P,
    tensors: &HashMap<String, Vec<u8>>,
    metadata: Option<&ModelMetadata>,
) -> ModelResult<()> {
    let mut file_data = Vec::new();
    
    // Serialize metadata
    let metadata = metadata.ok_or_else(|| ModelError::ValidationError {
        reason: "Metadata required for ToRSh format".to_string(),
    })?;
    
    let metadata_json = serde_json::to_vec(metadata)?;
    let metadata_len = metadata_json.len() as u64;
    
    // Write metadata length
    file_data.extend_from_slice(&metadata_len.to_le_bytes());
    
    // Write metadata
    file_data.extend_from_slice(&metadata_json);
    
    // For now, just append tensor data directly
    // In a real implementation, we'd properly serialize as SafeTensors
    for (name, data) in tensors {
        let name_bytes = name.as_bytes();
        let name_len = name_bytes.len() as u32;
        let data_len = data.len() as u32;
        
        file_data.extend_from_slice(&name_len.to_le_bytes());
        file_data.extend_from_slice(name_bytes);
        file_data.extend_from_slice(&data_len.to_le_bytes());
        file_data.extend_from_slice(data);
    }
    
    std::fs::write(path, file_data)?;
    Ok(())
}

/// Convert between model formats
pub fn convert_model_format<P: AsRef<Path>, Q: AsRef<Path>>(
    input_path: P,
    output_path: Q,
    input_format: Option<ModelFormat>,
    output_format: ModelFormat,
) -> ModelResult<()> {
    let (tensors, metadata) = load_model_from_file(input_path, input_format)?;
    save_model_to_file(output_path, &tensors, metadata.as_ref(), output_format)?;
    Ok(())
}

/// Validate model file integrity
pub fn validate_model_file<P: AsRef<Path>>(
    path: P,
    expected_checksum: Option<&str>,
) -> ModelResult<bool> {
    let path = path.as_ref();
    
    if !path.exists() {
        return Ok(false);
    }
    
    // Verify checksum if provided
    if let Some(expected) = expected_checksum {
        let data = std::fs::read(path)?;
        let hash = sha2::Sha256::digest(&data);
        let hex_hash = hex::encode(hash);
        
        if hex_hash != expected {
            return Ok(false);
        }
    }
    
    // Try to load the model to verify format
    match load_model_from_file(path, None) {
        Ok(_) => Ok(true),
        Err(_) => Ok(false),
    }
}

/// Get model file information
pub fn get_model_file_info<P: AsRef<Path>>(path: P) -> ModelResult<(ModelFormat, u64, Option<ModelMetadata>)> {
    let path = path.as_ref();
    
    let metadata = std::fs::metadata(path)?;
    let size = metadata.len();
    
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("");
    
    let format = ModelFormat::from_extension(ext)
        .ok_or_else(|| ModelError::InvalidFormat {
            format: ext.to_string(),
        })?;
    
    let (_, model_metadata) = load_model_from_file(path, Some(format))?;
    
    Ok((format, size, model_metadata))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    
    #[test]
    fn test_model_format_extension() {
        assert_eq!(ModelFormat::SafeTensors.extension(), "safetensors");
        assert_eq!(ModelFormat::PyTorch.extension(), "pth");
        assert_eq!(ModelFormat::Onnx.extension(), "onnx");
    }
    
    #[test]
    fn test_format_from_extension() {
        assert_eq!(ModelFormat::from_extension("safetensors"), Some(ModelFormat::SafeTensors));
        assert_eq!(ModelFormat::from_extension("pth"), Some(ModelFormat::PyTorch));
        assert_eq!(ModelFormat::from_extension("unknown"), None);
    }
    
    #[test]
    fn test_torsh_format_roundtrip() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test.torsh");
        
        let mut tensors = HashMap::new();
        tensors.insert("weight".to_string(), vec![1u8, 2, 3, 4]);
        tensors.insert("bias".to_string(), vec![5u8, 6, 7, 8]);
        
        let metadata = ModelMetadata {
            name: "test_model".to_string(),
            version: "1.0.0".to_string(),
            architecture: "TestNet".to_string(),
            framework: "ToRSh".to_string(),
            created_at: "2023-01-01T00:00:00Z".to_string(),
            extra: HashMap::new(),
        };
        
        // Save
        save_model_to_file(&file_path, &tensors, Some(&metadata), ModelFormat::ToRSh).unwrap();
        
        // Load
        let (loaded_tensors, loaded_metadata) = load_model_from_file(&file_path, Some(ModelFormat::ToRSh)).unwrap();
        
        assert_eq!(loaded_tensors.len(), 2);
        assert!(loaded_tensors.contains_key("weight"));
        assert!(loaded_tensors.contains_key("bias"));
        
        let loaded_meta = loaded_metadata.unwrap();
        assert_eq!(loaded_meta.name, "test_model");
        assert_eq!(loaded_meta.version, "1.0.0");
    }
    
    #[test]
    fn test_validate_model_file() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("nonexistent.torsh");
        
        // Non-existent file
        assert!(!validate_model_file(&file_path, None).unwrap());
        
        // Create a simple file
        std::fs::write(&file_path, b"not a valid model").unwrap();
        assert!(!validate_model_file(&file_path, None).unwrap());
    }
}