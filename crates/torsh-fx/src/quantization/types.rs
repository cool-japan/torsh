//! Core quantization types and data structures

/// Quantization scheme
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantizationScheme {
    /// 8-bit integer quantization
    Int8,
    /// 16-bit integer quantization
    Int16,
    /// Dynamic quantization
    Dynamic,
    /// Fake quantization for QAT
    Fake,
}

/// Quantization parameters
#[derive(Debug, Clone)]
pub struct QuantizationParams {
    pub scheme: QuantizationScheme,
    pub scale: f32,
    pub zero_point: i32,
    pub qmin: i32,
    pub qmax: i32,
}

impl QuantizationParams {
    /// Create new quantization parameters
    pub fn new(scheme: QuantizationScheme, scale: f32, zero_point: i32) -> Self {
        let (qmin, qmax) = match scheme {
            QuantizationScheme::Int8 => (-128, 127),
            QuantizationScheme::Int16 => (-32768, 32767),
            _ => (0, 255), // Default for dynamic/fake
        };

        Self {
            scheme,
            scale,
            zero_point,
            qmin,
            qmax,
        }
    }

    /// Create symmetric quantization parameters
    pub fn symmetric(scheme: QuantizationScheme, scale: f32) -> Self {
        Self::new(scheme, scale, 0)
    }

    /// Create asymmetric quantization parameters
    pub fn asymmetric(scheme: QuantizationScheme, scale: f32, zero_point: i32) -> Self {
        Self::new(scheme, scale, zero_point)
    }
}

/// Quantization annotation for nodes
#[derive(Debug, Clone)]
pub struct QuantizationAnnotation {
    pub input_params: Vec<Option<QuantizationParams>>,
    pub output_params: Option<QuantizationParams>,
    pub calibration_data: Option<CalibrationData>,
}

/// Calibration data for quantization
#[derive(Debug, Clone)]
pub struct CalibrationData {
    pub min_val: f32,
    pub max_val: f32,
    pub histogram: Vec<u64>,
    pub sample_count: usize,
}

impl CalibrationData {
    pub fn new() -> Self {
        Self {
            min_val: f32::INFINITY,
            max_val: f32::NEG_INFINITY,
            histogram: vec![0; 256],
            sample_count: 0,
        }
    }

    /// Update calibration data with new values
    pub fn update(&mut self, values: &[f32]) {
        for &val in values {
            self.min_val = self.min_val.min(val);
            self.max_val = self.max_val.max(val);

            // Update histogram
            let bin = ((val - self.min_val) / (self.max_val - self.min_val) * 255.0)
                .clamp(0.0, 255.0) as usize;
            if bin < self.histogram.len() {
                self.histogram[bin] += 1;
            }
        }
        self.sample_count += values.len();
    }

    /// Compute optimal quantization parameters
    pub fn compute_params(&self, scheme: QuantizationScheme) -> QuantizationParams {
        let range = self.max_val - self.min_val;
        let scale = match scheme {
            QuantizationScheme::Int8 => range / 255.0,
            QuantizationScheme::Int16 => range / 65535.0,
            _ => range / 255.0,
        };

        let zero_point = (-self.min_val / scale).round() as i32;
        QuantizationParams::new(scheme, scale, zero_point)
    }
}

impl Default for CalibrationData {
    fn default() -> Self {
        Self::new()
    }
}
