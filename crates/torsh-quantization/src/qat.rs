//! Quantization-aware training

use crate::{FakeQuantize, Observer, QuantConfig, TorshResult};
use std::collections::HashMap;
use torsh_core::{DType, TorshError};
// use torsh_nn::Module; // Temporarily disabled due to autograd compilation issues
use torsh_tensor::Tensor;

// Temporary placeholder trait for Module (to allow compilation)
#[allow(dead_code)]
pub trait Module {
    fn forward(&self, input: &Tensor) -> TorshResult<Tensor>;
    fn parameters(&self) -> Vec<&Tensor>;
    fn parameters_mut(&mut self) -> Vec<&mut Tensor>;
    fn named_parameters(&self) -> Vec<(String, &Tensor)>;
    fn train(&mut self, mode: bool);
    fn eval(&mut self) {
        self.train(false);
    }
}

/// QAT layer state
#[derive(Debug)]
pub struct QATLayerState {
    pub fake_quant: FakeQuantize,
    pub observer: Observer,
    pub layer_name: String,
    pub enabled: bool,
    pub num_updates: usize,
}

impl QATLayerState {
    pub fn new(layer_name: String, config: &QuantConfig) -> Self {
        let (_qmin, _qmax) = config.get_qint_range();
        let fake_quant = match config.dtype {
            DType::I8 => FakeQuantize::int8(1.0, 0),
            DType::U8 => FakeQuantize::uint8(1.0, 0),
            _ => FakeQuantize::int8(1.0, 0),
        };

        Self {
            fake_quant,
            observer: Observer::new(config.observer_type),
            layer_name,
            enabled: config.enable_fake_quant,
            num_updates: 0,
        }
    }

    pub fn update_and_quantize(&mut self, input: &Tensor) -> TorshResult<Tensor> {
        if !self.enabled {
            return Ok(input.clone());
        }

        // Update observer with input statistics
        self.observer.update(input)?;
        self.num_updates += 1;

        // Update fake quantization parameters every N steps
        if self.num_updates % 100 == 0 {
            self.update_fake_quant_params()?;
        }

        // Apply fake quantization
        self.fake_quant.forward(input)
    }

    fn update_fake_quant_params(&mut self) -> TorshResult<()> {
        // Calculate new quantization parameters from observer
        let (scale, zero_point) = self.observer.calculate_qparams(DType::I8)?;
        self.fake_quant.update_params(scale, zero_point);
        Ok(())
    }

    pub fn enable(&mut self) {
        self.enabled = true;
        self.fake_quant.enable();
    }

    pub fn disable(&mut self) {
        self.enabled = false;
        self.fake_quant.disable();
    }
}

/// QAT state for tracking quantization parameters during training
#[derive(Debug)]
pub struct QATState {
    pub config: QuantConfig,
    pub layer_states: HashMap<String, QATLayerState>,
    pub enabled: bool,
    pub training_step: usize,
    pub warmup_steps: usize,
}

impl QATState {
    pub fn new(config: QuantConfig) -> Self {
        Self {
            warmup_steps: 1000, // Default warmup steps
            config,
            layer_states: HashMap::new(),
            enabled: true,
            training_step: 0,
        }
    }

    pub fn with_warmup_steps(mut self, warmup_steps: usize) -> Self {
        self.warmup_steps = warmup_steps;
        self
    }

    pub fn add_layer(&mut self, layer_name: String) {
        let layer_state = QATLayerState::new(layer_name.clone(), &self.config);
        self.layer_states.insert(layer_name, layer_state);
    }

    pub fn get_layer_state(&self, layer_name: &str) -> Option<&QATLayerState> {
        self.layer_states.get(layer_name)
    }

    pub fn get_layer_state_mut(&mut self, layer_name: &str) -> Option<&mut QATLayerState> {
        self.layer_states.get_mut(layer_name)
    }

    pub fn step(&mut self) {
        self.training_step += 1;

        // Enable fake quantization after warmup
        if self.training_step == self.warmup_steps {
            self.enable_fake_quantization();
        }
    }

    pub fn enable_fake_quantization(&mut self) {
        self.enabled = true;
        for layer_state in self.layer_states.values_mut() {
            layer_state.enable();
        }
    }

    pub fn disable_fake_quantization(&mut self) {
        self.enabled = false;
        for layer_state in self.layer_states.values_mut() {
            layer_state.disable();
        }
    }

    pub fn is_in_warmup(&self) -> bool {
        self.training_step < self.warmup_steps
    }

    pub fn get_quantization_stats(&self) -> HashMap<String, (f32, i32, usize)> {
        let mut stats = HashMap::new();

        for (layer_name, layer_state) in &self.layer_states {
            if let Ok((scale, zero_point)) =
                layer_state.observer.calculate_qparams(self.config.dtype)
            {
                stats.insert(
                    layer_name.clone(),
                    (scale, zero_point, layer_state.num_updates),
                );
            }
        }

        stats
    }
}

/// QAT preparation result
#[derive(Debug)]
pub struct QATPreparationResult {
    pub qat_state: QATState,
    pub quantized_layers: Vec<String>,
    pub total_parameters: usize,
}

/// Prepare a module for quantization-aware training
///
/// This function:
/// 1. Inserts fake quantization operations into the model
/// 2. Attaches observers to track activation ranges
/// 3. Prepares the model for quantization-aware training
pub fn prepare_qat(module: &mut dyn Module) -> TorshResult<QATPreparationResult> {
    prepare_qat_with_config(module, QuantConfig::qat())
}

/// Prepare QAT with custom configuration
pub fn prepare_qat_with_config(
    module: &mut dyn Module,
    config: QuantConfig,
) -> TorshResult<QATPreparationResult> {
    // Validate configuration
    config.validate()?;

    // Get all parameters in the model
    let params = module.named_parameters();
    let total_parameters = params.len();

    // Initialize QAT state
    let mut qat_state = QATState::new(config);
    let mut quantized_layers = Vec::new();

    // For each quantizable layer (Conv2d, Linear, etc.)
    for (name, _param) in params {
        // Check if this parameter belongs to a quantizable layer
        if is_quantizable_parameter(&name) {
            let layer_name = extract_layer_name(&name);

            // Add layer to QAT state if not already added
            if !qat_state.layer_states.contains_key(&layer_name) {
                qat_state.add_layer(layer_name.clone());
                quantized_layers.push(layer_name);
            }
        }
    }

    // Enable fake quantization for training
    enable_fake_quantization(module)?;

    Ok(QATPreparationResult {
        qat_state,
        quantized_layers,
        total_parameters,
    })
}

/// Extract layer name from parameter name
fn extract_layer_name(param_name: &str) -> String {
    // Simple heuristic: take everything before ".weight" or ".bias"
    param_name
        .split(".weight")
        .next()
        .unwrap_or(param_name)
        .split(".bias")
        .next()
        .unwrap_or(param_name)
        .to_string()
}

/// Check if a parameter belongs to a quantizable layer
fn is_quantizable_parameter(param_name: &str) -> bool {
    // Check if parameter name contains quantizable layer types
    let lower_name = param_name.to_lowercase();
    lower_name.contains("linear")
        || lower_name.contains("conv")
        || lower_name.contains("dense")
        || lower_name.contains("fc")
        || lower_name.contains("embedding")
        || (lower_name.contains("weight") && !lower_name.contains("norm"))
}

/// Enable fake quantization for training
fn enable_fake_quantization(module: &mut dyn Module) -> TorshResult<()> {
    // In a real implementation, this would:
    // 1. Wrap quantizable layers with FakeQuantize modules
    // 2. Insert fake quantization ops after activations
    // 3. Configure observers for calibration

    // For now, we mark the module as being in QAT mode
    module.train(true);

    Ok(())
}

/// QAT training step - updates observers and fake quantization
pub fn qat_training_step(
    qat_state: &mut QATState,
    layer_inputs: &HashMap<String, Tensor>,
) -> TorshResult<HashMap<String, Tensor>> {
    let mut outputs = HashMap::new();

    // Step the QAT state
    qat_state.step();

    // Process each layer
    for (layer_name, input) in layer_inputs {
        if let Some(layer_state) = qat_state.get_layer_state_mut(layer_name) {
            let output = layer_state.update_and_quantize(input)?;
            outputs.insert(layer_name.clone(), output);
        } else {
            // Pass through for non-quantized layers
            outputs.insert(layer_name.clone(), input.clone());
        }
    }

    Ok(outputs)
}

/// Disable fake quantization (for evaluation)
pub fn disable_fake_quantization(module: &mut dyn Module) -> TorshResult<()> {
    module.eval();
    Ok(())
}

/// Disable fake quantization in QAT state
pub fn disable_qat_fake_quantization(qat_state: &mut QATState) {
    qat_state.disable_fake_quantization();
}

/// Enable fake quantization in QAT state
pub fn enable_qat_fake_quantization(qat_state: &mut QATState) {
    qat_state.enable_fake_quantization();
}

/// QAT conversion result
#[derive(Debug)]
pub struct QATConversionResult {
    pub quantized_params: HashMap<String, (f32, i32)>, // scale, zero_point
    pub conversion_stats: HashMap<String, usize>,      // layer -> num_updates
}

/// Convert QAT model to quantized model
pub fn convert_qat(_module: &dyn Module, qat_state: &QATState) -> TorshResult<QATConversionResult> {
    let mut quantized_params = HashMap::new();
    let mut conversion_stats = HashMap::new();

    // Extract quantization parameters from QAT state
    for (layer_name, layer_state) in &qat_state.layer_states {
        if let Ok((scale, zero_point)) = layer_state
            .observer
            .calculate_qparams(qat_state.config.dtype)
        {
            quantized_params.insert(layer_name.clone(), (scale, zero_point));
            conversion_stats.insert(layer_name.clone(), layer_state.num_updates);
        }
    }

    if quantized_params.is_empty() {
        return Err(TorshError::Other(
            "No quantization parameters found in QAT state".to_string(),
        ));
    }

    // In a real implementation, this would:
    // 1. Remove fake quantization ops
    // 2. Replace floating-point ops with quantized ops
    // 3. Use observed statistics to set quantization parameters

    Ok(QATConversionResult {
        quantized_params,
        conversion_stats,
    })
}

/// Complete QAT pipeline
pub fn qat_pipeline(
    module: &mut dyn Module,
    config: QuantConfig,
    num_training_steps: usize,
) -> TorshResult<QATConversionResult> {
    // Prepare model for QAT
    let mut result = prepare_qat_with_config(module, config)?;

    // Simulate training for the specified number of steps
    for step in 0..num_training_steps {
        result.qat_state.step();

        // In a real implementation, this would be integrated with
        // the actual training loop
        if step % 1000 == 0 {
            println!(
                "QAT step {}, warmup: {}",
                step,
                result.qat_state.is_in_warmup()
            );
        }
    }

    // Convert to quantized model
    convert_qat(module, &result.qat_state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ObserverType, QuantBackend};
    use torsh_tensor::creation::tensor_1d;

    // Mock module for testing
    struct MockModule {
        layer1_weight: Tensor,
        layer2_linear_weight: Tensor,
        norm_weight: Tensor, // Should not be quantized
    }

    impl MockModule {
        fn new() -> Self {
            Self {
                layer1_weight: tensor_1d(&[1.0, 2.0]).unwrap(),
                layer2_linear_weight: tensor_1d(&[3.0, 4.0]).unwrap(),
                norm_weight: tensor_1d(&[0.5]).unwrap(),
            }
        }
    }

    impl Module for MockModule {
        fn forward(&self, input: &Tensor) -> TorshResult<Tensor> {
            Ok(input.clone())
        }

        fn parameters(&self) -> Vec<&Tensor> {
            vec![
                &self.layer1_weight,
                &self.layer2_linear_weight,
                &self.norm_weight,
            ]
        }

        fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
            vec![
                &mut self.layer1_weight,
                &mut self.layer2_linear_weight,
                &mut self.norm_weight,
            ]
        }

        fn named_parameters(&self) -> Vec<(String, &Tensor)> {
            vec![
                ("layer1.weight".to_string(), &self.layer1_weight),
                (
                    "layer2.linear.weight".to_string(),
                    &self.layer2_linear_weight,
                ),
                ("norm.weight".to_string(), &self.norm_weight), // This should not be quantized
            ]
        }

        fn train(&mut self, _mode: bool) {}
    }

    #[test]
    fn test_qat_layer_state() {
        let config = QuantConfig::qat();
        let mut layer_state = QATLayerState::new("test_layer".to_string(), &config);

        assert!(layer_state.enabled);
        assert_eq!(layer_state.num_updates, 0);
        assert_eq!(layer_state.layer_name, "test_layer");

        // Test update and quantize
        let input = tensor_1d(&[1.0, 2.0, 3.0]).unwrap();
        let output = layer_state.update_and_quantize(&input).unwrap();
        assert_eq!(output.shape().dims(), input.shape().dims());
        assert_eq!(layer_state.num_updates, 1);

        // Test enable/disable
        layer_state.disable();
        assert!(!layer_state.enabled);

        layer_state.enable();
        assert!(layer_state.enabled);
    }

    #[test]
    fn test_qat_state() {
        let config = QuantConfig::qat();
        let mut qat_state = QATState::new(config).with_warmup_steps(10);

        assert_eq!(qat_state.warmup_steps, 10);
        assert_eq!(qat_state.training_step, 0);
        assert!(qat_state.is_in_warmup());

        // Add a layer
        qat_state.add_layer("test_layer".to_string());
        assert!(qat_state.get_layer_state("test_layer").is_some());

        // Test stepping
        for _ in 0..5 {
            qat_state.step();
        }
        assert_eq!(qat_state.training_step, 5);
        assert!(qat_state.is_in_warmup());

        // Step past warmup
        for _ in 0..10 {
            qat_state.step();
        }
        assert!(!qat_state.is_in_warmup());
    }

    #[test]
    fn test_is_quantizable_parameter() {
        assert!(is_quantizable_parameter("layer1.linear.weight"));
        assert!(is_quantizable_parameter("conv2d.weight"));
        assert!(is_quantizable_parameter("dense.weight"));
        assert!(is_quantizable_parameter("fc.weight"));
        assert!(is_quantizable_parameter("embedding.weight"));

        // Should not quantize normalization layers
        assert!(!is_quantizable_parameter("batch_norm.weight"));
        assert!(!is_quantizable_parameter("layer_norm.weight"));
    }

    #[test]
    fn test_extract_layer_name() {
        assert_eq!(extract_layer_name("layer1.weight"), "layer1");
        assert_eq!(extract_layer_name("model.linear.bias"), "model.linear");
        assert_eq!(extract_layer_name("conv2d"), "conv2d");
    }

    #[test]
    fn test_prepare_qat() {
        let mut module = MockModule::new();
        let result = prepare_qat(&mut module).unwrap();

        // Should have found 2 quantizable layers (layer1 and layer2.linear)
        assert_eq!(result.quantized_layers.len(), 2);
        assert_eq!(result.total_parameters, 3);
        assert!(result.quantized_layers.contains(&"layer1".to_string()));
        assert!(result
            .quantized_layers
            .contains(&"layer2.linear".to_string()));

        // Should not include normalization layer
        assert!(!result.quantized_layers.contains(&"norm".to_string()));
    }

    #[test]
    fn test_qat_preparation_result() {
        let config = QuantConfig::qat()
            .with_backend(QuantBackend::Native)
            .with_observer(ObserverType::MovingAverage);

        let mut module = MockModule::new();
        let result = prepare_qat_with_config(&mut module, config).unwrap();

        assert!(result.qat_state.enabled);
        assert_eq!(result.qat_state.config.backend, QuantBackend::Native);
        assert_eq!(
            result.qat_state.config.observer_type,
            ObserverType::MovingAverage
        );
    }

    #[test]
    fn test_qat_training_step() {
        let config = QuantConfig::qat();
        let mut qat_state = QATState::new(config);
        qat_state.add_layer("layer1".to_string());

        let mut layer_inputs = HashMap::new();
        layer_inputs.insert("layer1".to_string(), tensor_1d(&[1.0, 2.0, 3.0]).unwrap());
        layer_inputs.insert("unknown_layer".to_string(), tensor_1d(&[4.0, 5.0]).unwrap());

        let outputs = qat_training_step(&mut qat_state, &layer_inputs).unwrap();

        assert_eq!(outputs.len(), 2);
        assert!(outputs.contains_key("layer1"));
        assert!(outputs.contains_key("unknown_layer"));
        assert_eq!(qat_state.training_step, 1);
    }

    #[test]
    fn test_qat_conversion() {
        let config = QuantConfig::qat();
        let mut qat_state = QATState::new(config);
        qat_state.add_layer("layer1".to_string());

        // Update layer state with some data
        let input = tensor_1d(&[1.0, 2.0, 3.0]).unwrap();
        if let Some(layer_state) = qat_state.get_layer_state_mut("layer1") {
            layer_state.update_and_quantize(&input).unwrap();
        }

        let module = MockModule::new();
        let result = convert_qat(&module, &qat_state).unwrap();

        assert!(result.quantized_params.contains_key("layer1"));
        assert!(result.conversion_stats.contains_key("layer1"));

        let (scale, zero_point) = result.quantized_params["layer1"];
        assert!(scale > 0.0);
        assert!((-128..=127).contains(&zero_point));
    }

    #[test]
    fn test_qat_enable_disable() {
        let config = QuantConfig::qat();
        let mut qat_state = QATState::new(config);
        qat_state.add_layer("layer1".to_string());

        // Test disabling
        disable_qat_fake_quantization(&mut qat_state);
        assert!(!qat_state.enabled);

        // Test enabling
        enable_qat_fake_quantization(&mut qat_state);
        assert!(qat_state.enabled);
    }

    #[test]
    fn test_qat_quantization_stats() {
        let config = QuantConfig::qat();
        let mut qat_state = QATState::new(config);
        qat_state.add_layer("layer1".to_string());

        // Update with some data
        let input = tensor_1d(&[1.0, 2.0, 3.0]).unwrap();
        if let Some(layer_state) = qat_state.get_layer_state_mut("layer1") {
            layer_state.update_and_quantize(&input).unwrap();
        }

        let stats = qat_state.get_quantization_stats();
        assert!(stats.contains_key("layer1"));

        let (scale, zero_point, num_updates) = stats["layer1"];
        assert!(scale > 0.0);
        assert!((-128..=127).contains(&zero_point));
        assert_eq!(num_updates, 1);
    }
}
