use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// SciRS2 POLICY: Use scirs2_core::random::quick instead of direct rand
use scirs2_core::random::quick::random_int;

use crate::grad_mode;

thread_local! {
    static EAGER_CONTEXT: RefCell<TensorFlowEagerContext> = RefCell::new(TensorFlowEagerContext::new());
}

#[derive(Debug, Clone)]
pub struct TensorFlowEagerContext {
    pub device_policy: DevicePolicy,
    pub execution_mode: ExecutionMode,
    pub gradient_tape: Option<Arc<Mutex<GradientTape>>>,
    pub function_cache: HashMap<String, CompiledFunction>,
    pub eager_execution_enabled: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DevicePolicy {
    Explicit,
    Warn,
    Silent,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExecutionMode {
    Eager,
    Graph,
    Mixed,
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct GradientTape {
    watched_tensors: Vec<TensorId>,
    operations: Vec<Operation>,
    persistent: bool,
    recording: bool,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct TensorId(pub u64);

#[derive(Debug, Clone)]
pub struct Operation {
    pub op_type: String,
    pub inputs: Vec<TensorId>,
    pub outputs: Vec<TensorId>,
    pub attrs: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct CompiledFunction {
    pub name: String,
    pub input_shapes: Vec<Vec<i64>>,
    pub output_shapes: Vec<Vec<i64>>,
    pub bytecode: Vec<u8>,
}

impl TensorFlowEagerContext {
    pub fn new() -> Self {
        Self {
            device_policy: DevicePolicy::Warn,
            execution_mode: ExecutionMode::Eager,
            gradient_tape: None,
            function_cache: HashMap::new(),
            eager_execution_enabled: true,
        }
    }

    pub fn enable_eager_execution(&mut self) {
        self.eager_execution_enabled = true;
        self.execution_mode = ExecutionMode::Eager;
    }

    pub fn disable_eager_execution(&mut self) {
        self.eager_execution_enabled = false;
        self.execution_mode = ExecutionMode::Graph;
    }

    pub fn set_device_policy(&mut self, policy: DevicePolicy) {
        self.device_policy = policy;
    }

    pub fn is_eager_execution_enabled(&self) -> bool {
        self.eager_execution_enabled
    }
}

impl Default for TensorFlowEagerContext {
    fn default() -> Self {
        Self::new()
    }
}

impl GradientTape {
    pub fn new(persistent: bool) -> Self {
        Self {
            watched_tensors: Vec::new(),
            operations: Vec::new(),
            persistent,
            recording: true,
        }
    }

    pub fn watch(&mut self, tensor_id: TensorId) {
        if !self.watched_tensors.contains(&tensor_id) {
            self.watched_tensors.push(tensor_id);
        }
    }

    pub fn record_operation(&mut self, op: Operation) {
        if self.recording {
            self.operations.push(op);
        }
    }

    pub fn stop_recording(&mut self) {
        self.recording = false;
    }

    pub fn start_recording(&mut self) {
        self.recording = true;
    }

    pub fn reset(&mut self) {
        self.watched_tensors.clear();
        self.operations.clear();
        self.recording = true;
    }

    pub fn gradient(
        &self,
        target: &[TensorId],
        sources: &[TensorId],
    ) -> Result<Vec<Option<TensorId>>, TensorFlowError> {
        let mut gradients = vec![None; sources.len()];

        for (i, source) in sources.iter().enumerate() {
            if self.watched_tensors.contains(source) {
                let grad = self.compute_gradient_for_source(target, source)?;
                gradients[i] = grad;
            }
        }

        Ok(gradients)
    }

    fn compute_gradient_for_source(
        &self,
        targets: &[TensorId],
        source: &TensorId,
    ) -> Result<Option<TensorId>, TensorFlowError> {
        if targets.is_empty() {
            return Ok(None);
        }

        for op in self.operations.iter().rev() {
            if op.outputs.iter().any(|id| targets.contains(id)) {
                if op.inputs.contains(source) {
                    return Ok(Some(TensorId(source.0 + 1000000)));
                }
            }
        }

        Ok(None)
    }
}

#[derive(Debug, Clone)]
pub enum TensorFlowError {
    InvalidOperation(String),
    GradientNotAvailable(String),
    ExecutionError(String),
    DeviceError(String),
}

impl std::fmt::Display for TensorFlowError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TensorFlowError::InvalidOperation(msg) => write!(f, "Invalid operation: {}", msg),
            TensorFlowError::GradientNotAvailable(msg) => {
                write!(f, "Gradient not available: {}", msg)
            }
            TensorFlowError::ExecutionError(msg) => write!(f, "Execution error: {}", msg),
            TensorFlowError::DeviceError(msg) => write!(f, "Device error: {}", msg),
        }
    }
}

impl std::error::Error for TensorFlowError {}

pub fn enable_eager_execution() {
    EAGER_CONTEXT.with(|ctx| {
        ctx.borrow_mut().enable_eager_execution();
    });
}

pub fn disable_eager_execution() {
    EAGER_CONTEXT.with(|ctx| {
        ctx.borrow_mut().disable_eager_execution();
    });
}

pub fn is_eager_execution_enabled() -> bool {
    EAGER_CONTEXT.with(|ctx| ctx.borrow().is_eager_execution_enabled())
}

pub fn set_device_policy(policy: DevicePolicy) {
    EAGER_CONTEXT.with(|ctx| {
        ctx.borrow_mut().set_device_policy(policy);
    });
}

pub struct GradientTapeContext {
    tape: Arc<Mutex<GradientTape>>,
}

impl GradientTapeContext {
    pub fn new(persistent: bool) -> Self {
        let tape = Arc::new(Mutex::new(GradientTape::new(persistent)));

        EAGER_CONTEXT.with(|ctx| {
            ctx.borrow_mut().gradient_tape = Some(tape.clone());
        });

        Self { tape }
    }

    pub fn watch(&self, tensor_id: TensorId) {
        if let Ok(mut tape) = self.tape.lock() {
            tape.watch(tensor_id);
        }
    }

    pub fn gradient(
        &self,
        target: &[TensorId],
        sources: &[TensorId],
    ) -> Result<Vec<Option<TensorId>>, TensorFlowError> {
        if let Ok(tape) = self.tape.lock() {
            tape.gradient(target, sources)
        } else {
            Err(TensorFlowError::ExecutionError(
                "Failed to acquire tape lock".to_string(),
            ))
        }
    }

    pub fn stop_recording(&self) {
        if let Ok(mut tape) = self.tape.lock() {
            tape.stop_recording();
        }
    }

    pub fn start_recording(&self) {
        if let Ok(mut tape) = self.tape.lock() {
            tape.start_recording();
        }
    }

    pub fn reset(&self) {
        if let Ok(mut tape) = self.tape.lock() {
            tape.reset();
        }
    }
}

impl Drop for GradientTapeContext {
    fn drop(&mut self) {
        EAGER_CONTEXT.with(|ctx| {
            ctx.borrow_mut().gradient_tape = None;
        });
    }
}

pub fn execute_eagerly<F, T>(f: F) -> Result<T, TensorFlowError>
where
    F: FnOnce() -> Result<T, TensorFlowError>,
{
    if !is_eager_execution_enabled() {
        return Err(TensorFlowError::ExecutionError(
            "Eager execution is not enabled".to_string(),
        ));
    }

    let _guard = grad_mode::InferenceModeGuard::new();
    f()
}

pub fn function(
    f: impl Fn(&[TensorId]) -> Result<Vec<TensorId>, TensorFlowError> + Send + Sync + 'static,
) -> TensorFlowFunction {
    TensorFlowFunction {
        inner: Arc::new(f),
        compiled: None,
    }
}

pub struct TensorFlowFunction {
    inner: Arc<dyn Fn(&[TensorId]) -> Result<Vec<TensorId>, TensorFlowError> + Send + Sync>,
    compiled: Option<CompiledFunction>,
}

impl TensorFlowFunction {
    pub fn call(&self, inputs: &[TensorId]) -> Result<Vec<TensorId>, TensorFlowError> {
        (self.inner)(inputs)
    }

    pub fn get_concrete_function(
        &mut self,
        input_shapes: &[Vec<i64>],
    ) -> Result<&CompiledFunction, TensorFlowError> {
        if self.compiled.is_none() {
            let compiled = CompiledFunction {
                name: format!("function_{}", random_int(0, i64::MAX) as u64),
                input_shapes: input_shapes.to_vec(),
                output_shapes: vec![vec![1]],
                bytecode: vec![0u8; 1024],
            };
            self.compiled = Some(compiled);
        }

        Ok(self.compiled.as_ref().unwrap())
    }
}

pub fn with_tape<F, T>(persistent: bool, f: F) -> Result<T, TensorFlowError>
where
    F: FnOnce(&GradientTapeContext) -> Result<T, TensorFlowError>,
{
    let tape = GradientTapeContext::new(persistent);
    let result = f(&tape)?;
    Ok(result)
}

pub fn record_operation(op_type: &str, inputs: &[TensorId], outputs: &[TensorId]) {
    let operation = Operation {
        op_type: op_type.to_string(),
        inputs: inputs.to_vec(),
        outputs: outputs.to_vec(),
        attrs: HashMap::new(),
    };

    EAGER_CONTEXT.with(|ctx| {
        if let Some(tape) = &ctx.borrow().gradient_tape {
            if let Ok(mut tape) = tape.lock() {
                tape.record_operation(operation);
            }
        }
    });
}

pub fn convert_to_tensor(_value: &[f64]) -> TensorId {
    TensorId(random_int(0, i64::MAX) as u64)
}

pub fn convert_to_value(tensor_id: &TensorId) -> Vec<f64> {
    vec![tensor_id.0 as f64]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eager_execution_enabled_by_default() {
        assert!(is_eager_execution_enabled());
    }

    #[test]
    fn test_disable_enable_eager_execution() {
        disable_eager_execution();
        assert!(!is_eager_execution_enabled());

        enable_eager_execution();
        assert!(is_eager_execution_enabled());
    }

    #[test]
    fn test_gradient_tape_context() {
        let tape = GradientTapeContext::new(false);
        let tensor_id = TensorId(42);

        tape.watch(tensor_id.clone());
        let result = tape.gradient(&[tensor_id.clone()], &[tensor_id]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_tensorflow_function() {
        let func = function(|inputs| {
            let mut outputs = Vec::new();
            for input in inputs {
                outputs.push(TensorId(input.0 + 1));
            }
            Ok(outputs)
        });

        let inputs = vec![TensorId(1), TensorId(2)];
        let result = func.call(&inputs);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 2);
    }

    #[test]
    fn test_with_tape() {
        let result = with_tape(false, |tape| {
            let tensor_id = TensorId(100);
            tape.watch(tensor_id.clone());

            tape.gradient(&[tensor_id.clone()], &[tensor_id])?;
            Ok(42)
        });

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
    }

    #[test]
    fn test_record_operation() {
        let inputs = vec![TensorId(1), TensorId(2)];
        let outputs = vec![TensorId(3)];

        record_operation("Add", &inputs, &outputs);
    }

    #[test]
    fn test_execute_eagerly() {
        let result = execute_eagerly(|| Ok(42));

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
    }

    #[test]
    fn test_device_policy() {
        set_device_policy(DevicePolicy::Silent);
        EAGER_CONTEXT.with(|ctx| {
            assert_eq!(ctx.borrow().device_policy, DevicePolicy::Silent);
        });
    }

    #[test]
    fn test_gradient_tape_persistent() {
        let tape = GradientTape::new(true);
        assert!(tape.persistent);

        let tape = GradientTape::new(false);
        assert!(!tape.persistent);
    }

    #[test]
    fn test_gradient_tape_recording() {
        let mut tape = GradientTape::new(false);
        assert!(tape.recording);

        tape.stop_recording();
        assert!(!tape.recording);

        tape.start_recording();
        assert!(tape.recording);
    }

    #[test]
    fn test_convert_tensor() {
        let values = vec![1.0, 2.0, 3.0];
        let tensor_id = convert_to_tensor(&values);
        let converted_values = convert_to_value(&tensor_id);
        assert!(!converted_values.is_empty());
    }
}
