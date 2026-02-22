use crate::OptimizerError;
use std::collections::HashMap;
use std::fmt;

/// Trait for types that can be converted to/from low-precision representation
pub trait LowPrecisionConvertible: Clone + fmt::Debug {
    /// Convert to low-precision representation
    fn to_low_precision(&self) -> LowPrecisionState;

    /// Convert from low-precision representation
    fn from_low_precision(state: &LowPrecisionState) -> Result<Self, OptimizerError>;
}

/// Low-precision state representation for memory-efficient optimizer states
#[derive(Clone, Debug)]
pub enum LowPrecisionState {
    /// 16-bit float representation
    F16(Vec<half::f16>),
    /// 16-bit brain float representation
    BF16(Vec<half::bf16>),
    /// 8-bit integer representation with scale factor
    I8 { values: Vec<i8>, scale: f32 },
    /// 4-bit integer representation with scale factor
    I4 { values: Vec<u8>, scale: f32 }, // packed 2 values per byte
    /// Sparse representation for mostly-zero states
    Sparse {
        indices: Vec<usize>,
        values: Vec<f32>,
    },
}

impl LowPrecisionState {
    /// Get the memory footprint in bytes
    pub fn memory_footprint(&self) -> usize {
        match self {
            LowPrecisionState::F16(values) => values.len() * 2,
            LowPrecisionState::BF16(values) => values.len() * 2,
            LowPrecisionState::I8 { values, .. } => values.len() + 4, // +4 for scale
            LowPrecisionState::I4 { values, .. } => (values.len() + 1) / 2 + 4, // packed + scale
            LowPrecisionState::Sparse { indices, values } => {
                indices.len() * 8 + values.len() * 4 // usize + f32
            }
        }
    }

    /// Convert to full precision f32 vector
    pub fn to_f32(&self) -> Vec<f32> {
        match self {
            LowPrecisionState::F16(values) => values.iter().map(|&x| x.to_f32()).collect(),
            LowPrecisionState::BF16(values) => values.iter().map(|&x| x.to_f32()).collect(),
            LowPrecisionState::I8 { values, scale } => {
                values.iter().map(|&x| (x as f32) * scale).collect()
            }
            LowPrecisionState::I4 { values, scale } => {
                let mut result = Vec::with_capacity(values.len() * 2);
                for &packed in values {
                    let low = ((packed & 0x0F) as i8 - 8) as f32 * scale;
                    let high = (((packed >> 4) as i8) - 8) as f32 * scale;
                    result.push(low);
                    result.push(high);
                }
                result
            }
            LowPrecisionState::Sparse { indices, values } => {
                let max_idx = indices.iter().max().copied().unwrap_or(0);
                let mut result = vec![0.0f32; max_idx + 1];
                for (&idx, &val) in indices.iter().zip(values.iter()) {
                    result[idx] = val;
                }
                result
            }
        }
    }

    /// Create from f32 vector with specified precision
    pub fn from_f32(values: &[f32], precision: PrecisionType) -> Self {
        match precision {
            PrecisionType::F16 => {
                let converted: Vec<half::f16> =
                    values.iter().map(|&x| half::f16::from_f32(x)).collect();
                LowPrecisionState::F16(converted)
            }
            PrecisionType::BF16 => {
                let converted: Vec<half::bf16> =
                    values.iter().map(|&x| half::bf16::from_f32(x)).collect();
                LowPrecisionState::BF16(converted)
            }
            PrecisionType::I8 => {
                let max_val = values.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
                let scale = max_val / 127.0;
                let converted: Vec<i8> =
                    values.iter().map(|&x| (x / scale).round() as i8).collect();
                LowPrecisionState::I8 {
                    values: converted,
                    scale,
                }
            }
            PrecisionType::I4 => {
                let max_val = values.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
                let scale = max_val / 7.0;
                let mut converted = Vec::with_capacity((values.len() + 1) / 2);

                for chunk in values.chunks(2) {
                    let low = ((chunk[0] / scale).round() as i8 + 8) as u8 & 0x0F;
                    let high = if chunk.len() > 1 {
                        (((chunk[1] / scale).round() as i8 + 8) as u8 & 0x0F) << 4
                    } else {
                        0
                    };
                    converted.push(low | high);
                }

                LowPrecisionState::I4 {
                    values: converted,
                    scale,
                }
            }
            PrecisionType::Sparse(threshold) => {
                let mut indices = Vec::new();
                let mut sparse_values = Vec::new();

                for (i, &val) in values.iter().enumerate() {
                    if val.abs() > threshold {
                        indices.push(i);
                        sparse_values.push(val);
                    }
                }

                LowPrecisionState::Sparse {
                    indices,
                    values: sparse_values,
                }
            }
        }
    }
}

/// Precision type for low-precision states
#[derive(Clone, Debug)]
pub enum PrecisionType {
    /// 16-bit float
    F16,
    /// 16-bit brain float
    BF16,
    /// 8-bit integer with scale
    I8,
    /// 4-bit integer with scale
    I4,
    /// Sparse representation with threshold
    Sparse(f32),
}

/// Low-precision optimizer wrapper
pub struct LowPrecisionOptimizer<T> {
    inner: T,
    precision: PrecisionType,
    state_cache: HashMap<String, LowPrecisionState>,
}

impl<T> LowPrecisionOptimizer<T> {
    /// Create a new low-precision optimizer wrapper
    pub fn new(inner: T, precision: PrecisionType) -> Self {
        Self {
            inner,
            precision,
            state_cache: HashMap::new(),
        }
    }

    /// Get the precision type
    pub fn precision(&self) -> &PrecisionType {
        &self.precision
    }

    /// Get the inner optimizer
    pub fn inner(&self) -> &T {
        &self.inner
    }

    /// Get the inner optimizer mutably
    pub fn inner_mut(&mut self) -> &mut T {
        &mut self.inner
    }

    /// Store state in low precision
    pub fn store_state(&mut self, key: String, values: &[f32]) {
        let low_precision_state = LowPrecisionState::from_f32(values, self.precision.clone());
        self.state_cache.insert(key, low_precision_state);
    }

    /// Load state from low precision
    pub fn load_state(&self, key: &str) -> Option<Vec<f32>> {
        self.state_cache.get(key).map(|state| state.to_f32())
    }

    /// Get total memory footprint of stored states
    pub fn memory_footprint(&self) -> usize {
        self.state_cache
            .values()
            .map(|state| state.memory_footprint())
            .sum()
    }

    /// Get compression ratio compared to full precision
    pub fn compression_ratio(&self) -> f32 {
        let total_elements: usize = self
            .state_cache
            .values()
            .map(|state| state.to_f32().len())
            .sum();

        if total_elements == 0 {
            return 1.0;
        }

        let full_precision_size = total_elements * 4; // 4 bytes per f32
        let compressed_size = self.memory_footprint();

        full_precision_size as f32 / compressed_size as f32
    }

    /// Clear all cached states
    pub fn clear_cache(&mut self) {
        self.state_cache.clear();
    }

    /// Get statistics about the stored states
    pub fn state_statistics(&self) -> StateStatistics {
        let total_states = self.state_cache.len();
        let total_memory = self.memory_footprint();
        let compression_ratio = self.compression_ratio();

        let precision_breakdown =
            self.state_cache
                .values()
                .fold(HashMap::new(), |mut acc, state| {
                    let precision_name = match state {
                        LowPrecisionState::F16(_) => "F16",
                        LowPrecisionState::BF16(_) => "BF16",
                        LowPrecisionState::I8 { .. } => "I8",
                        LowPrecisionState::I4 { .. } => "I4",
                        LowPrecisionState::Sparse { .. } => "Sparse",
                    };
                    *acc.entry(precision_name.to_string()).or_insert(0) += 1;
                    acc
                });

        StateStatistics {
            total_states,
            total_memory,
            compression_ratio,
            precision_breakdown,
        }
    }
}

/// Statistics about low-precision states
#[derive(Debug, Clone)]
pub struct StateStatistics {
    pub total_states: usize,
    pub total_memory: usize,
    pub compression_ratio: f32,
    pub precision_breakdown: HashMap<String, usize>,
}

impl fmt::Display for StateStatistics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Low-Precision State Statistics:")?;
        writeln!(f, "  Total States: {}", self.total_states)?;
        writeln!(f, "  Total Memory: {} bytes", self.total_memory)?;
        writeln!(f, "  Compression Ratio: {:.2}x", self.compression_ratio)?;
        writeln!(f, "  Precision Breakdown:")?;
        for (precision, count) in &self.precision_breakdown {
            writeln!(f, "    {}: {} states", precision, count)?;
        }
        Ok(())
    }
}

// Implement for f32 vectors (common optimizer state)
impl LowPrecisionConvertible for Vec<f32> {
    fn to_low_precision(&self) -> LowPrecisionState {
        LowPrecisionState::from_f32(self, PrecisionType::F16)
    }

    fn from_low_precision(state: &LowPrecisionState) -> Result<Self, OptimizerError> {
        Ok(state.to_f32())
    }
}

// Implement for HashMap<String, f32> (common optimizer state)
impl LowPrecisionConvertible for HashMap<String, f32> {
    fn to_low_precision(&self) -> LowPrecisionState {
        let values: Vec<f32> = self.values().copied().collect();
        LowPrecisionState::from_f32(&values, PrecisionType::F16)
    }

    fn from_low_precision(state: &LowPrecisionState) -> Result<Self, OptimizerError> {
        let values = state.to_f32();
        // This is a simplified implementation - in practice, you'd need to store keys separately
        let mut result = HashMap::new();
        for (i, value) in values.into_iter().enumerate() {
            result.insert(format!("param_{}", i), value);
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f16_conversion() {
        let values = vec![1.0, 2.5, -3.7, 0.0, 1000.0];
        let state = LowPrecisionState::from_f32(&values, PrecisionType::F16);
        let recovered = state.to_f32();

        // Check that values are approximately equal (f16 has limited precision)
        for (original, recovered) in values.iter().zip(recovered.iter()) {
            assert!(
                (original - recovered).abs() < 0.01,
                "Original: {}, Recovered: {}",
                original,
                recovered
            );
        }
    }

    #[test]
    fn test_i8_conversion() {
        let values = vec![1.0, 2.5, -3.7, 0.0, 10.0];
        let state = LowPrecisionState::from_f32(&values, PrecisionType::I8);
        let recovered = state.to_f32();

        // I8 should have reasonable precision for small values
        for (original, recovered) in values.iter().zip(recovered.iter()) {
            assert!(
                (original - recovered).abs() < 0.5,
                "Original: {}, Recovered: {}",
                original,
                recovered
            );
        }
    }

    #[test]
    fn test_sparse_conversion() {
        let values = vec![0.0, 2.5, 0.0, 0.0, 10.0, 0.0];
        let state = LowPrecisionState::from_f32(&values, PrecisionType::Sparse(1.0));
        let recovered = state.to_f32();

        // Sparse should preserve non-zero values exactly
        assert_eq!(recovered.len(), 5); // max index + 1
        assert_eq!(recovered[1], 2.5);
        assert_eq!(recovered[4], 10.0);
        for i in [0, 2, 3] {
            assert_eq!(recovered[i], 0.0);
        }
    }

    #[test]
    fn test_memory_footprint() {
        let values = vec![1.0; 1000];

        let f16_state = LowPrecisionState::from_f32(&values, PrecisionType::F16);
        let i8_state = LowPrecisionState::from_f32(&values, PrecisionType::I8);
        let sparse_state = LowPrecisionState::from_f32(&values, PrecisionType::Sparse(2.0));

        let full_size = values.len() * 4; // 4 bytes per f32

        assert!(f16_state.memory_footprint() < full_size);
        assert!(i8_state.memory_footprint() < full_size);
        assert!(sparse_state.memory_footprint() < full_size);
    }

    #[test]
    fn test_low_precision_optimizer() {
        let mut optimizer =
            LowPrecisionOptimizer::new("dummy_optimizer".to_string(), PrecisionType::F16);

        let values = vec![1.0, 2.0, 3.0, 4.0];
        optimizer.store_state("momentum".to_string(), &values);

        let recovered = optimizer.load_state("momentum").unwrap();

        // Check approximate equality
        for (original, recovered) in values.iter().zip(recovered.iter()) {
            assert!((original - recovered).abs() < 0.01);
        }

        // Check compression ratio
        assert!(optimizer.compression_ratio() > 1.0);
    }
}
