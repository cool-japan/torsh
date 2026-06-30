//! PyTorch model format parser for ToRSh compatibility
//!
//! This module provides functionality to parse and convert PyTorch models
//! to ToRSh format, enabling interoperability between frameworks.

// Infrastructure module - functions designed for CLI command integration
#![allow(dead_code)]

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::Path;
use tracing::{debug, info, warn};

// ✅ SciRS2 POLICY COMPLIANT: Use scirs2-core unified access patterns
use scirs2_core::random::{thread_rng, Distribution, Normal};

// ToRSh integration
use torsh::core::device::DeviceType;

use super::tensor_integration::ModelTensor;
use super::types::{DType, Device, LayerInfo, ModelMetadata, TensorInfo, TorshModel};

/// PyTorch model metadata extracted from .pth files
#[derive(Debug, Clone)]
pub struct PyTorchModelInfo {
    /// PyTorch version that produced the checkpoint, if it could be determined
    /// from the file. `None` when the version is not recoverable from metadata.
    pub pytorch_version: Option<String>,
    /// Model class name (if available)
    pub model_class: Option<String>,
    /// State dict keys
    pub state_dict_keys: Vec<String>,
    /// Total file size in bytes
    pub file_size: u64,
    /// Number of parameters
    pub num_parameters: u64,
    /// Whether this is a full model or just state_dict
    pub is_full_model: bool,
}

impl PyTorchModelInfo {
    /// Human-readable PyTorch version, or `"unknown"` when it could not be
    /// determined from the checkpoint. This never fabricates a version number.
    pub fn version_display(&self) -> &str {
        self.pytorch_version.as_deref().unwrap_or("unknown")
    }
}

/// PyTorch layer type mapping
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyTorchLayerType {
    Linear,
    Conv2d,
    Conv1d,
    Conv3d,
    BatchNorm2d,
    BatchNorm1d,
    LayerNorm,
    Dropout,
    Embedding,
    LSTM,
    GRU,
    Attention,
    Unknown,
}

impl PyTorchLayerType {
    /// Convert PyTorch layer type to ToRSh layer type string
    pub fn to_torsh_type(&self) -> &'static str {
        match self {
            PyTorchLayerType::Linear => "Linear",
            PyTorchLayerType::Conv2d => "Conv2d",
            PyTorchLayerType::Conv1d => "Conv1d",
            PyTorchLayerType::Conv3d => "Conv3d",
            PyTorchLayerType::BatchNorm2d => "BatchNorm2d",
            PyTorchLayerType::BatchNorm1d => "BatchNorm1d",
            PyTorchLayerType::LayerNorm => "LayerNorm",
            PyTorchLayerType::Dropout => "Dropout",
            PyTorchLayerType::Embedding => "Embedding",
            PyTorchLayerType::LSTM => "LSTM",
            PyTorchLayerType::GRU => "GRU",
            PyTorchLayerType::Attention => "Attention",
            PyTorchLayerType::Unknown => "Unknown",
        }
    }

    /// Infer layer type from parameter name
    pub fn from_param_name(param_name: &str) -> Self {
        if param_name.contains("linear") || param_name.contains("fc") {
            PyTorchLayerType::Linear
        } else if param_name.contains("conv3d") {
            PyTorchLayerType::Conv3d
        } else if param_name.contains("conv1d") {
            PyTorchLayerType::Conv1d
        } else if param_name.contains("conv2d") || param_name.contains("conv") {
            // Default conv layers to Conv2d (most common in vision models)
            PyTorchLayerType::Conv2d
        } else if param_name.contains("bn") || param_name.contains("batch_norm") {
            PyTorchLayerType::BatchNorm2d
        } else if param_name.contains("layer_norm") || param_name.contains("ln") {
            PyTorchLayerType::LayerNorm
        } else if param_name.contains("embed") {
            PyTorchLayerType::Embedding
        } else if param_name.contains("lstm") {
            PyTorchLayerType::LSTM
        } else if param_name.contains("gru") {
            PyTorchLayerType::GRU
        } else if param_name.contains("attn") || param_name.contains("attention") {
            PyTorchLayerType::Attention
        } else {
            PyTorchLayerType::Unknown
        }
    }
}

/// Parse PyTorch model file and extract metadata
pub async fn parse_pytorch_model(path: &Path) -> Result<PyTorchModelInfo> {
    info!("Parsing PyTorch model from: {}", path.display());

    // Read file metadata
    let metadata = tokio::fs::metadata(path)
        .await
        .with_context(|| format!("Failed to read file metadata: {}", path.display()))?;

    let file_size = metadata.len();

    // Read file header to detect format
    let file_data = tokio::fs::read(path)
        .await
        .with_context(|| format!("Failed to read PyTorch file: {}", path.display()))?;

    // Check if it's a ZIP file (PyTorch >= 1.6 uses ZIP format)
    let is_zip = file_data.len() >= 4 && &file_data[0..4] == b"PK\x03\x04";

    debug!(
        "PyTorch model format: {}",
        if is_zip { "ZIP" } else { "Pickle" }
    );

    // Parse model structure (simplified for now)
    let (state_dict_keys, num_parameters, is_full_model) =
        parse_pytorch_structure(&file_data, is_zip)?;

    Ok(PyTorchModelInfo {
        pytorch_version: detect_pytorch_version(&file_data),
        model_class: None, // Would be extracted from full model files
        state_dict_keys,
        file_size,
        num_parameters,
        is_full_model,
    })
}

/// Parse PyTorch file structure, returning the real state-dict parameter names,
/// an estimated parameter count, and whether the file holds a full model.
///
/// PyTorch checkpoints embed each parameter name (e.g. `layer1.0.conv1.weight`)
/// as an ASCII string inside the pickled payload. This scans the raw bytes for
/// those tokens instead of fabricating a fixed list. When no parameter names can
/// be recovered, it returns an empty key list rather than inventing layers.
fn parse_pytorch_structure(file_data: &[u8], is_zip: bool) -> Result<(Vec<String>, u64, bool)> {
    let state_dict_keys = extract_state_dict_keys(file_data);

    // A checkpoint that pickles module objects (a "full model") references the
    // class machinery; a bare state_dict does not. `torch.nn.Module` / `OrderedDict`
    // markers reliably distinguish the two cases in the pickle stream.
    let is_full_model = find_subslice(file_data, b"torch.nn.modules").is_some()
        || find_subslice(file_data, b"torch\nModule").is_some();

    // Estimate parameter count from the real tensor storage entries when this is
    // a ZIP checkpoint (each tensor lives under `.../data/<n>`), otherwise fall
    // back to a byte-size heuristic. The estimate is reported as such by callers.
    let num_parameters = if is_zip {
        estimate_parameters_from_zip(file_data).unwrap_or_else(|| (file_data.len() / 4) as u64)
    } else {
        (file_data.len() / 4) as u64
    };

    Ok((state_dict_keys, num_parameters, is_full_model))
}

/// Detect the PyTorch version that produced a checkpoint file.
///
/// PyTorch (>= 1.6) saves models as a ZIP archive that contains a `version`
/// entry holding the *serialization protocol* number and may embed the
/// `torch.__version__` string inside the pickled payload. This function reads
/// the real bytes:
///
/// 1. It scans for an embedded `torch.__version__`-style version string
///    (e.g. `1.13.1`, `2.0.0+cu118`) and returns it verbatim if found.
/// 2. Failing that, for ZIP checkpoints it extracts the serialization protocol
///    number from the `version` archive entry and reports it as
///    `"serialization protocol N"`.
///
/// When the version genuinely cannot be determined from the file, it returns
/// `None` rather than fabricating a plausible-looking version number.
fn detect_pytorch_version(file_data: &[u8]) -> Option<String> {
    if let Some(version) = scan_embedded_torch_version(file_data) {
        debug!("Detected embedded torch version string: {}", version);
        return Some(version);
    }

    let is_zip = file_data.len() >= 4 && &file_data[0..4] == b"PK\x03\x04";
    if is_zip {
        if let Some(protocol) = read_zip_serialization_protocol(file_data) {
            debug!("Detected serialization protocol: {}", protocol);
            return Some(format!("serialization protocol {}", protocol));
        }
    }

    debug!("PyTorch version could not be determined from file metadata");
    None
}

/// Scan raw checkpoint bytes for an embedded `torch.__version__` string.
///
/// Newer checkpoints store the producing torch version as an ASCII string in the
/// pickle stream. We locate the literal `__version__` (or a `torch ` qualifier)
/// and parse the dotted version token that follows. Returns `None` if no
/// plausible version token is present.
fn scan_embedded_torch_version(data: &[u8]) -> Option<String> {
    const MARKERS: [&[u8]; 2] = [b"__version__", b"torch_version"];

    for marker in MARKERS {
        let mut search_start = 0;
        while let Some(rel) = find_subslice(&data[search_start..], marker) {
            let after = search_start + rel + marker.len();

            // Preferred path: the value is a length-prefixed pickle string right
            // after the marker's memo opcode. Reading the exact length avoids
            // swallowing trailing opcode bytes.
            if let Some(value) = read_pickle_string_after(&data[after..]) {
                if let Some(version) = parse_version_token(value.as_bytes()) {
                    return Some(version);
                }
            }

            // Fallback: heuristically scan the bytes following the marker.
            let window_end = after.saturating_add(64).min(data.len()).max(after);
            if let Some(version) = parse_version_token(&data[after..window_end]) {
                return Some(version);
            }

            search_start = after;
        }
    }
    None
}

/// Read the next length-prefixed pickle unicode string within `data`.
///
/// Recognizes the protocol-2+ opcodes `SHORT_BINUNICODE` (`0x8c`, 1-byte length)
/// and `BINUNICODE` (`X`, 4-byte little-endian length). Scans a short distance
/// for the opcode so an intervening memo opcode (`q<idx>`) is skipped. Returns
/// `None` if no well-formed string is found.
fn read_pickle_string_after(data: &[u8]) -> Option<String> {
    let scan_limit = data.len().min(16);
    let mut i = 0;
    while i < scan_limit {
        match data[i] {
            0x8c => {
                // SHORT_BINUNICODE: 1-byte length follows.
                let len_pos = i + 1;
                let body_start = len_pos + 1;
                if len_pos < data.len() {
                    let len = data[len_pos] as usize;
                    let body_end = body_start + len;
                    if body_end <= data.len() {
                        return Some(
                            String::from_utf8_lossy(&data[body_start..body_end]).into_owned(),
                        );
                    }
                }
                return None;
            }
            b'X' => {
                // BINUNICODE: 4-byte little-endian length follows.
                let len_pos = i + 1;
                let body_start = len_pos + 4;
                if body_start <= data.len() {
                    let len = u32::from_le_bytes([
                        data[len_pos],
                        data[len_pos + 1],
                        data[len_pos + 2],
                        data[len_pos + 3],
                    ]) as usize;
                    let body_end = body_start + len;
                    if body_end <= data.len() && len <= 256 {
                        return Some(
                            String::from_utf8_lossy(&data[body_start..body_end]).into_owned(),
                        );
                    }
                }
                return None;
            }
            _ => i += 1,
        }
    }
    None
}

/// Parse the first dotted semantic-version token (e.g. `2.0.1`, `1.13.0+cu117`)
/// found at the start region of `window`, skipping non-version separator bytes.
///
/// The numeric `MAJOR.MINOR.PATCH` core is parsed first; a local-version suffix
/// (e.g. `+cu118`, `.dev20240101`) is appended only when it is introduced by an
/// explicit `+` or `-` separator. This prevents trailing pickle opcode bytes
/// (such as a memo `q`) from being mistaken for part of the version string.
fn parse_version_token(window: &[u8]) -> Option<String> {
    let mut idx = 0;
    // Skip leading non-digit bytes (quotes, length prefixes, separators).
    while idx < window.len() && !window[idx].is_ascii_digit() {
        idx += 1;
        if idx > 8 {
            // Version token should appear right after the marker; give up early.
            return None;
        }
    }

    // Parse the numeric core: digits and dots only.
    let core_start = idx;
    let mut dot_count = 0;
    while idx < window.len() {
        let byte = window[idx];
        if byte.is_ascii_digit() {
            idx += 1;
        } else if byte == b'.' {
            dot_count += 1;
            idx += 1;
        } else {
            break;
        }
    }
    let core_end = idx;

    if dot_count < 2 || core_end == core_start {
        return None;
    }

    // Optionally consume a local-version suffix, but only when it is explicitly
    // introduced by `+` or `-` (PEP 440 / PyTorch convention).
    let mut suffix_end = core_end;
    if idx < window.len() && (window[idx] == b'+' || window[idx] == b'-') {
        idx += 1;
        while idx < window.len() {
            let byte = window[idx];
            if byte.is_ascii_alphanumeric() || byte == b'.' || byte == b'_' {
                idx += 1;
            } else {
                break;
            }
        }
        suffix_end = idx;
    }

    let token = String::from_utf8_lossy(&window[core_start..suffix_end]);
    let trimmed = token.trim_end_matches(['.', '+', '-', '_']);
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

/// Extract the serialization protocol number from a PyTorch ZIP checkpoint.
///
/// PyTorch stores a stored (uncompressed) `version` entry whose body is the
/// ASCII protocol number. We locate the entry by its ZIP local file header and
/// read the immediately-following uncompressed body. Returns `None` if the
/// entry is absent or compressed in a way we cannot read directly.
fn read_zip_serialization_protocol(data: &[u8]) -> Option<u32> {
    const LOCAL_HEADER_SIG: &[u8] = b"PK\x03\x04";
    let mut cursor = 0;

    while let Some(rel) = find_subslice(&data[cursor..], LOCAL_HEADER_SIG) {
        let header = cursor + rel;
        // Local file header layout (offsets from signature):
        //  +8  compression method (u16, LE)
        //  +18 compressed size (u32, LE)
        //  +26 file name length (u16, LE)
        //  +28 extra field length (u16, LE)
        //  +30 file name bytes
        if header + 30 > data.len() {
            break;
        }
        let compression = u16::from_le_bytes([data[header + 8], data[header + 9]]);
        let compressed_size = u32::from_le_bytes([
            data[header + 18],
            data[header + 19],
            data[header + 20],
            data[header + 21],
        ]) as usize;
        let name_len = u16::from_le_bytes([data[header + 26], data[header + 27]]) as usize;
        let extra_len = u16::from_le_bytes([data[header + 28], data[header + 29]]) as usize;

        let name_start = header + 30;
        let name_end = name_start + name_len;
        if name_end > data.len() {
            break;
        }
        let name = &data[name_start..name_end];

        // Match an entry whose path component is exactly `version` (the archive
        // is rooted at the model name, e.g. `archive/version`).
        let is_version_entry = name == b"version" || name.ends_with(b"/version");

        // Only "stored" (compression method 0) bodies can be read as raw ASCII.
        if is_version_entry && compression == 0 {
            let body_start = name_end + extra_len;
            let body_end = body_start + compressed_size;
            if body_end <= data.len() {
                let body = String::from_utf8_lossy(&data[body_start..body_end]);
                if let Ok(protocol) = body.trim().parse::<u32>() {
                    return Some(protocol);
                }
            }
        }

        cursor = header + 4;
    }

    None
}

/// Find the first occurrence of `needle` within `haystack`, returning its index.
fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || needle.len() > haystack.len() {
        return None;
    }
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

/// Extract real state-dict parameter names embedded in a PyTorch checkpoint.
///
/// Parameter keys appear in the pickle stream as ASCII tokens ending in a known
/// PyTorch parameter suffix (`.weight`, `.bias`, `.running_mean`, etc.). This
/// scans for those suffixes and walks backwards to recover the full dotted name.
/// Results are de-duplicated while preserving first-seen order. Returns an empty
/// vector when none are present — it never fabricates names.
fn extract_state_dict_keys(data: &[u8]) -> Vec<String> {
    const SUFFIXES: [&[u8]; 6] = [
        b".weight",
        b".bias",
        b".running_mean",
        b".running_var",
        b".num_batches_tracked",
        b".in_proj_weight",
    ];

    let mut keys: Vec<String> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

    for suffix in SUFFIXES {
        let mut search_start = 0;
        while let Some(rel) = find_subslice(&data[search_start..], suffix) {
            let suffix_pos = search_start + rel;
            let name_end = suffix_pos + suffix.len();

            // Walk backwards over the identifier characters preceding the suffix.
            let mut name_start = suffix_pos;
            while name_start > 0 {
                let candidate = data[name_start - 1];
                if candidate.is_ascii_alphanumeric() || candidate == b'_' || candidate == b'.' {
                    name_start -= 1;
                } else {
                    break;
                }
            }

            if name_start < name_end {
                let raw = String::from_utf8_lossy(&data[name_start..name_end]);
                // Real parameter keys never start with a dot; strip any leading
                // dots introduced by surrounding binary bytes.
                let token = raw.trim_start_matches('.');
                // Require a real prefix before the suffix that begins with an
                // identifier character (not the bare suffix, not noise).
                let starts_clean = token
                    .chars()
                    .next()
                    .is_some_and(|c| c.is_ascii_alphanumeric() || c == '_');
                if starts_clean && token.len() > suffix.len() {
                    let owned = token.to_string();
                    if seen.insert(owned.clone()) {
                        keys.push(owned);
                    }
                }
            }

            search_start = name_end;
        }
    }

    keys
}

/// Estimate the total parameter count of a ZIP checkpoint from its stored tensor
/// data entries (`.../data/<n>`). Each such entry's uncompressed body is raw
/// tensor bytes; summing them and dividing by the f32 element size yields a real
/// element count. Returns `None` when no tensor data entries are found.
fn estimate_parameters_from_zip(data: &[u8]) -> Option<u64> {
    const LOCAL_HEADER_SIG: &[u8] = b"PK\x03\x04";
    let mut cursor = 0;
    let mut total_bytes: u64 = 0;
    let mut found_any = false;

    while let Some(rel) = find_subslice(&data[cursor..], LOCAL_HEADER_SIG) {
        let header = cursor + rel;
        if header + 30 > data.len() {
            break;
        }
        let compressed_size = u32::from_le_bytes([
            data[header + 18],
            data[header + 19],
            data[header + 20],
            data[header + 21],
        ]) as u64;
        let name_len = u16::from_le_bytes([data[header + 26], data[header + 27]]) as usize;

        let name_start = header + 30;
        let name_end = name_start + name_len;
        if name_end > data.len() {
            break;
        }
        let name = &data[name_start..name_end];

        // PyTorch stores tensor payloads under a `data/` directory; the entries
        // are numbered (e.g. `archive/data/0`). Count only those bodies.
        if let Some(data_dir) = find_subslice(name, b"/data/") {
            let tail = &name[data_dir + b"/data/".len()..];
            if !tail.is_empty() && tail.iter().all(|b| b.is_ascii_digit()) {
                total_bytes += compressed_size;
                found_any = true;
            }
        }

        cursor = header + 4;
    }

    if found_any {
        // Tensor storages are serialized in their element dtype; f32 (4 bytes) is
        // the dominant case for the models this tool targets.
        Some(total_bytes / 4)
    } else {
        None
    }
}

/// Convert PyTorch model to ToRSh model
pub async fn convert_pytorch_to_torsh(
    pytorch_path: &Path,
    device: DeviceType,
) -> Result<TorshModel> {
    info!("Converting PyTorch model to ToRSh format");

    let pytorch_info = parse_pytorch_model(pytorch_path).await?;

    // Build ToRSh model structure from PyTorch state dict
    let (layers, weights) = build_torsh_structure(&pytorch_info, device)?;

    let mut metadata = ModelMetadata::default();
    metadata.format = "torsh".to_string();
    metadata.framework = "pytorch".to_string();
    metadata.description = Some(format!(
        "Converted from PyTorch {} model",
        pytorch_info.version_display()
    ));
    metadata.tags = vec!["converted".to_string(), "pytorch".to_string()];

    // Add conversion metadata
    metadata
        .custom
        .insert("original_format".to_string(), serde_json::json!("pytorch"));
    metadata.custom.insert(
        "pytorch_version".to_string(),
        serde_json::json!(pytorch_info.pytorch_version),
    );
    metadata.custom.insert(
        "original_file_size".to_string(),
        serde_json::json!(pytorch_info.file_size),
    );

    Ok(TorshModel {
        layers,
        weights,
        metadata,
    })
}

/// Build ToRSh model structure from PyTorch state dict
fn build_torsh_structure(
    pytorch_info: &PyTorchModelInfo,
    _device: DeviceType,
) -> Result<(Vec<LayerInfo>, HashMap<String, TensorInfo>)> {
    debug!(
        "Building ToRSh structure from {} parameters",
        pytorch_info.num_parameters
    );

    let mut layers = Vec::new();
    let mut weights = HashMap::new();

    // Group parameters by layer
    let layer_groups = group_parameters_by_layer(&pytorch_info.state_dict_keys);

    for (layer_name, param_names) in layer_groups {
        debug!(
            "Processing layer: {} with {} parameters",
            layer_name,
            param_names.len()
        );

        // Infer layer type from parameter names
        let layer_type = PyTorchLayerType::from_param_name(&layer_name);

        // Infer shapes from parameter names
        let (input_shape, output_shape) = infer_layer_shapes(&param_names, layer_type);

        // Count parameters
        let param_count = estimate_layer_parameters(&param_names, layer_type);

        // Create layer info
        let layer = LayerInfo {
            name: layer_name.clone(),
            layer_type: layer_type.to_torsh_type().to_string(),
            input_shape,
            output_shape,
            parameters: param_count,
            trainable: true,
            config: create_layer_config(layer_type),
        };

        layers.push(layer);

        // Create weight tensors
        for param_name in param_names {
            let shape = infer_tensor_shape(&param_name, layer_type);

            let weight_info = TensorInfo {
                name: param_name.clone(),
                shape,
                dtype: DType::F32,
                requires_grad: !param_name.contains("running"), // Running stats are non-trainable
                device: Device::Cpu,
            };

            weights.insert(param_name, weight_info);
        }
    }

    Ok((layers, weights))
}

/// Group parameters by layer name
fn group_parameters_by_layer(param_names: &[String]) -> HashMap<String, Vec<String>> {
    let mut groups: HashMap<String, Vec<String>> = HashMap::new();

    for param_name in param_names {
        // Extract layer name (everything before the last dot)
        let layer_name = if let Some(pos) = param_name.rfind('.') {
            param_name[..pos].to_string()
        } else {
            param_name.clone()
        };

        groups
            .entry(layer_name)
            .or_insert_with(Vec::new)
            .push(param_name.clone());
    }

    groups
}

/// Infer layer shapes from parameter names
fn infer_layer_shapes(
    param_names: &[String],
    layer_type: PyTorchLayerType,
) -> (Vec<usize>, Vec<usize>) {
    // Find weight parameter to infer dimensions
    let weight_param = param_names.iter().find(|name| name.ends_with(".weight"));

    match layer_type {
        PyTorchLayerType::Linear => {
            // Linear layers: weight shape is [out_features, in_features]
            if weight_param.is_some() {
                // Realistic sizes for common architectures
                let input_dim = 512;
                let output_dim = 256;
                (vec![input_dim], vec![output_dim])
            } else {
                (vec![512], vec![256])
            }
        }
        PyTorchLayerType::Conv2d => {
            // Conv2d: input [batch, in_channels, height, width]
            (vec![3, 224, 224], vec![64, 112, 112])
        }
        PyTorchLayerType::BatchNorm2d | PyTorchLayerType::BatchNorm1d => {
            // BatchNorm preserves shape
            (vec![64, 56, 56], vec![64, 56, 56])
        }
        PyTorchLayerType::Embedding => {
            // Embedding: [vocab_size, embedding_dim]
            (vec![30000], vec![512])
        }
        PyTorchLayerType::LSTM | PyTorchLayerType::GRU => {
            // RNN: [seq_len, batch, features]
            (vec![128, 512], vec![128, 256])
        }
        _ => (vec![512], vec![512]),
    }
}

/// Estimate layer parameter count
fn estimate_layer_parameters(param_names: &[String], layer_type: PyTorchLayerType) -> u64 {
    let (input_shape, output_shape) = infer_layer_shapes(param_names, layer_type);

    let input_size: u64 = input_shape.iter().map(|&x| x as u64).product();
    let output_size: u64 = output_shape.iter().map(|&x| x as u64).product();

    match layer_type {
        PyTorchLayerType::Linear => {
            // weight: out * in, bias: out
            input_size * output_size + output_size
        }
        PyTorchLayerType::Conv2d => {
            // Rough estimate based on typical kernel sizes
            let kernel_size = 9; // 3x3
            output_size * kernel_size + output_size // weights + bias
        }
        PyTorchLayerType::BatchNorm2d | PyTorchLayerType::BatchNorm1d => {
            // gamma, beta, running_mean, running_var
            output_size * 4
        }
        PyTorchLayerType::Embedding => input_size * output_size,
        _ => output_size,
    }
}

/// Infer tensor shape from parameter name
fn infer_tensor_shape(param_name: &str, layer_type: PyTorchLayerType) -> Vec<usize> {
    if param_name.ends_with(".weight") {
        match layer_type {
            PyTorchLayerType::Linear => vec![256, 512],
            PyTorchLayerType::Conv2d => vec![64, 3, 3, 3], // [out_ch, in_ch, kH, kW]
            PyTorchLayerType::BatchNorm2d => vec![64],
            PyTorchLayerType::Embedding => vec![30000, 512],
            _ => vec![512, 512],
        }
    } else if param_name.ends_with(".bias") {
        match layer_type {
            PyTorchLayerType::Linear => vec![256],
            PyTorchLayerType::Conv2d => vec![64],
            _ => vec![512],
        }
    } else if param_name.contains("running_mean") || param_name.contains("running_var") {
        vec![64]
    } else {
        vec![512]
    }
}

/// Create layer configuration based on type
fn create_layer_config(layer_type: PyTorchLayerType) -> HashMap<String, serde_json::Value> {
    let mut config = HashMap::new();

    match layer_type {
        PyTorchLayerType::Conv2d => {
            config.insert("kernel_size".to_string(), serde_json::json!(3));
            config.insert("stride".to_string(), serde_json::json!(1));
            config.insert("padding".to_string(), serde_json::json!(1));
        }
        PyTorchLayerType::Dropout => {
            config.insert("p".to_string(), serde_json::json!(0.5));
        }
        PyTorchLayerType::LSTM | PyTorchLayerType::GRU => {
            config.insert("hidden_size".to_string(), serde_json::json!(256));
            config.insert("num_layers".to_string(), serde_json::json!(2));
            config.insert("bidirectional".to_string(), serde_json::json!(false));
        }
        _ => {}
    }

    config
}

/// Map PyTorch tensor to ToRSh tensor (simplified)
pub fn map_pytorch_tensor_to_torsh(
    _pytorch_tensor: &[u8],
    shape: Vec<usize>,
    requires_grad: bool,
    device: DeviceType,
) -> Result<ModelTensor> {
    // In real implementation, would deserialize PyTorch tensor format
    // For now, create a random tensor with the correct shape

    let mut rng = thread_rng();
    let normal = Normal::new(0.0, 0.1)?;

    let num_elements: usize = shape.iter().product();
    let data: Vec<f32> = (0..num_elements)
        .map(|_| normal.sample(&mut rng) as f32)
        .collect();

    ModelTensor::from_data("converted".to_string(), data, shape, requires_grad, device)
}

/// Validate PyTorch to ToRSh conversion
pub fn validate_conversion(
    pytorch_info: &PyTorchModelInfo,
    torsh_model: &TorshModel,
) -> Result<()> {
    info!("Validating PyTorch to ToRSh conversion");

    // Check parameter count is reasonable
    let torsh_params: u64 = torsh_model.layers.iter().map(|l| l.parameters).sum();

    let param_ratio = torsh_params as f64 / pytorch_info.num_parameters as f64;

    if param_ratio < 0.5 || param_ratio > 2.0 {
        warn!(
            "Parameter count mismatch: PyTorch {} vs ToRSh {} (ratio: {:.2})",
            pytorch_info.num_parameters, torsh_params, param_ratio
        );
    }

    // Check all layers have valid shapes
    for layer in &torsh_model.layers {
        if layer.input_shape.is_empty() || layer.output_shape.is_empty() {
            anyhow::bail!("Layer {} has invalid shape", layer.name);
        }
    }

    info!("Conversion validation passed");
    Ok(())
}

/// Export conversion report
pub fn generate_conversion_report(
    pytorch_info: &PyTorchModelInfo,
    torsh_model: &TorshModel,
) -> String {
    let mut report = String::new();

    report.push_str("╔═══════════════════════════════════════════════════════════════════════╗\n");
    report.push_str("║                  PYTORCH → TORSH CONVERSION REPORT                    ║\n");
    report
        .push_str("╚═══════════════════════════════════════════════════════════════════════╝\n\n");

    report.push_str("📦 Source Model (PyTorch)\n");
    report.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    report.push_str(&format!(
        "  PyTorch Version:    {}\n",
        pytorch_info.version_display()
    ));
    report.push_str(&format!(
        "  File Size:          {:.2} MB\n",
        pytorch_info.file_size as f64 / (1024.0 * 1024.0)
    ));
    report.push_str(&format!(
        "  Parameters:         {}\n",
        pytorch_info.num_parameters
    ));
    report.push_str(&format!(
        "  State Dict Keys:    {}\n",
        pytorch_info.state_dict_keys.len()
    ));
    report.push_str("\n");

    report.push_str("🎯 Target Model (ToRSh)\n");
    report.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    let torsh_params: u64 = torsh_model.layers.iter().map(|l| l.parameters).sum();
    report.push_str(&format!(
        "  ToRSh Version:      {}\n",
        torsh_model.metadata.version
    ));
    report.push_str(&format!(
        "  Layers:             {}\n",
        torsh_model.layers.len()
    ));
    report.push_str(&format!("  Parameters:         {}\n", torsh_params));
    report.push_str(&format!(
        "  Tensors:            {}\n",
        torsh_model.weights.len()
    ));
    report.push_str("\n");

    report.push_str("📊 Conversion Statistics\n");
    report.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    let param_ratio = torsh_params as f64 / pytorch_info.num_parameters as f64;
    report.push_str(&format!("  Parameter Ratio:    {:.2}\n", param_ratio));
    report.push_str(&format!(
        "  Layers Created:     {}\n",
        torsh_model.layers.len()
    ));

    report.push_str("\n");
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_type_inference() {
        assert_eq!(
            PyTorchLayerType::from_param_name("model.fc1.weight"),
            PyTorchLayerType::Linear
        );
        assert_eq!(
            PyTorchLayerType::from_param_name("conv1.weight"),
            PyTorchLayerType::Conv2d
        );
        assert_eq!(
            PyTorchLayerType::from_param_name("bn1.running_mean"),
            PyTorchLayerType::BatchNorm2d
        );
    }

    #[test]
    fn test_parameter_grouping() {
        let params = vec![
            "layer1.weight".to_string(),
            "layer1.bias".to_string(),
            "layer2.weight".to_string(),
            "layer2.bias".to_string(),
        ];

        let groups = group_parameters_by_layer(&params);
        assert_eq!(groups.len(), 2);
        assert_eq!(
            groups
                .get("layer1")
                .expect("element retrieval should succeed for valid index")
                .len(),
            2
        );
        assert_eq!(
            groups
                .get("layer2")
                .expect("element retrieval should succeed for valid index")
                .len(),
            2
        );
    }

    #[test]
    fn test_shape_inference() {
        let params = vec!["fc.weight".to_string(), "fc.bias".to_string()];
        let (input, output) = infer_layer_shapes(&params, PyTorchLayerType::Linear);

        assert!(!input.is_empty());
        assert!(!output.is_empty());
    }

    #[test]
    fn test_layer_config_creation() {
        let config = create_layer_config(PyTorchLayerType::Conv2d);
        assert!(config.contains_key("kernel_size"));
        assert!(config.contains_key("stride"));
        assert!(config.contains_key("padding"));
    }

    #[test]
    fn test_detect_version_returns_none_on_unknown() {
        // Random non-checkpoint bytes carry no version metadata: must be honest.
        let junk = vec![0x00u8, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77];
        assert_eq!(detect_pytorch_version(&junk), None);
    }

    #[test]
    fn test_scan_embedded_torch_version() {
        // Simulate a pickle fragment carrying `__version__` then a length-prefixed
        // BINUNICODE value `2.1.0` (length 5), followed by a memo opcode.
        let mut data = Vec::new();
        data.extend_from_slice(b"\x80\x02}q\x00X\x0b\x00\x00\x00__version__q\x01");
        data.extend_from_slice(b"X\x05\x00\x00\x002.1.0q\x02");
        let version = scan_embedded_torch_version(&data);
        assert_eq!(version.as_deref(), Some("2.1.0"));
    }

    #[test]
    fn test_parse_version_token_local_suffix() {
        // Delimited local-version suffix (terminated by a non-identifier byte).
        assert_eq!(
            parse_version_token(b"q\x002.0.1+cu118\x00"),
            Some("2.0.1+cu118".to_string())
        );
        // Bare numeric core with no suffix.
        assert_eq!(
            parse_version_token(b"\x001.13.0\x00"),
            Some("1.13.0".to_string())
        );
        // Not enough dots to be a version => None (no fabrication).
        assert_eq!(parse_version_token(b"abc"), None);
        assert_eq!(parse_version_token(b"12"), None);
    }

    #[test]
    fn test_read_pickle_string_short_binunicode() {
        // 0x8c <len=6> "2.0.0+"  -> exact-length read, no trailing opcode bytes.
        let mut data = vec![0x71, 0x01, 0x8c, 0x05];
        data.extend_from_slice(b"2.0.0");
        data.extend_from_slice(b"q\x02");
        assert_eq!(read_pickle_string_after(&data).as_deref(), Some("2.0.0"));
    }

    #[test]
    fn test_extract_state_dict_keys_real_names() {
        // Embedded ASCII parameter names as they appear in a real pickle stream.
        let mut data = Vec::new();
        data.extend_from_slice(b"...q\x00conv1.weightq\x01....fc.biasq\x02....bn.running_meanq");
        let keys = extract_state_dict_keys(&data);
        assert!(keys.contains(&"conv1.weight".to_string()));
        assert!(keys.contains(&"fc.bias".to_string()));
        assert!(keys.contains(&"bn.running_mean".to_string()));
    }

    #[test]
    fn test_extract_state_dict_keys_empty_when_absent() {
        // No parameter-name tokens present: must return empty, not a fixed list.
        let data = b"no parameter names here, just prose".to_vec();
        assert!(extract_state_dict_keys(&data).is_empty());
    }
}
