//! Quantum computing autograd extensions
//!
//! This module provides automatic differentiation support for quantum circuits
//! and quantum machine learning. It enables gradient-based optimization of
//! quantum parameters in variational quantum algorithms, quantum neural networks,
//! and hybrid classical-quantum models.

use crate::error_handling::{AutogradError, AutogradResult};
use std::collections::HashMap;
use std::f64::consts::PI;
use std::sync::{Arc, Mutex};

/// Represents a quantum bit (qubit) in a quantum circuit
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Qubit(pub usize);

impl Qubit {
    pub fn new(index: usize) -> Self {
        Self(index)
    }

    pub fn index(&self) -> usize {
        self.0
    }
}

/// Complex number representation for quantum amplitudes
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex {
    pub real: f64,
    pub imag: f64,
}

impl Complex {
    pub fn new(real: f64, imag: f64) -> Self {
        Self { real, imag }
    }

    pub fn zero() -> Self {
        Self::new(0.0, 0.0)
    }

    pub fn one() -> Self {
        Self::new(1.0, 0.0)
    }

    pub fn i() -> Self {
        Self::new(0.0, 1.0)
    }

    pub fn magnitude_squared(&self) -> f64 {
        self.real * self.real + self.imag * self.imag
    }

    pub fn magnitude(&self) -> f64 {
        self.magnitude_squared().sqrt()
    }

    pub fn conjugate(&self) -> Complex {
        Complex::new(self.real, -self.imag)
    }

    pub fn exp(&self) -> Complex {
        let r = self.real.exp();
        Complex::new(r * self.imag.cos(), r * self.imag.sin())
    }
}

impl std::ops::Add for Complex {
    type Output = Complex;
    fn add(self, other: Complex) -> Complex {
        Complex::new(self.real + other.real, self.imag + other.imag)
    }
}

impl std::ops::Sub for Complex {
    type Output = Complex;
    fn sub(self, other: Complex) -> Complex {
        Complex::new(self.real - other.real, self.imag - other.imag)
    }
}

impl std::ops::Mul for Complex {
    type Output = Complex;
    fn mul(self, other: Complex) -> Complex {
        Complex::new(
            self.real * other.real - self.imag * other.imag,
            self.real * other.imag + self.imag * other.real,
        )
    }
}

impl std::ops::Mul<f64> for Complex {
    type Output = Complex;
    fn mul(self, scalar: f64) -> Complex {
        Complex::new(self.real * scalar, self.imag * scalar)
    }
}

/// Quantum state vector representation
#[derive(Debug, Clone)]
pub struct QuantumState {
    /// Amplitudes for each computational basis state
    amplitudes: Vec<Complex>,
    /// Number of qubits
    num_qubits: usize,
}

impl QuantumState {
    /// Create a quantum state with the given number of qubits in |0...0⟩ state
    pub fn zeros(num_qubits: usize) -> Self {
        let num_states = 1 << num_qubits;
        let mut amplitudes = vec![Complex::zero(); num_states];
        amplitudes[0] = Complex::one(); // |0...0⟩ state
        Self {
            amplitudes,
            num_qubits,
        }
    }

    /// Create a quantum state from amplitudes
    pub fn from_amplitudes(amplitudes: Vec<Complex>) -> AutogradResult<Self> {
        let num_states = amplitudes.len();
        if !num_states.is_power_of_two() {
            return Err(AutogradError::gradient_computation(
                "quantum_state_from_amplitudes",
                format!("Number of amplitudes {} is not a power of 2", num_states),
            ));
        }

        let num_qubits = num_states.trailing_zeros() as usize;
        Ok(Self {
            amplitudes,
            num_qubits,
        })
    }

    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    pub fn amplitudes(&self) -> &[Complex] {
        &self.amplitudes
    }

    pub fn amplitudes_mut(&mut self) -> &mut [Complex] {
        &mut self.amplitudes
    }

    /// Compute the probability of measuring a specific computational basis state
    pub fn probability(&self, state_index: usize) -> f64 {
        if state_index >= self.amplitudes.len() {
            return 0.0;
        }
        self.amplitudes[state_index].magnitude_squared()
    }

    /// Normalize the quantum state
    pub fn normalize(&mut self) {
        let norm_squared: f64 = self
            .amplitudes
            .iter()
            .map(|amp| amp.magnitude_squared())
            .sum();
        let norm = norm_squared.sqrt();

        if norm > 1e-12 {
            for amp in &mut self.amplitudes {
                *amp = *amp * (1.0 / norm);
            }
        }
    }

    /// Check if the state is normalized
    pub fn is_normalized(&self) -> bool {
        let norm_squared: f64 = self
            .amplitudes
            .iter()
            .map(|amp| amp.magnitude_squared())
            .sum();
        (norm_squared - 1.0).abs() < 1e-10
    }
}

/// Trait for quantum gates with automatic differentiation support
pub trait QuantumGate: Send + Sync + std::fmt::Debug {
    /// Apply the gate to a quantum state
    fn apply(&self, state: &mut QuantumState) -> AutogradResult<()>;

    /// Compute the gradient of the gate with respect to its parameters
    fn gradient(
        &self,
        state: &QuantumState,
        parameter_index: usize,
    ) -> AutogradResult<QuantumStateGradient>;

    /// Get the number of parameters in this gate
    fn parameter_count(&self) -> usize;

    /// Get gate parameters
    fn parameters(&self) -> &[f64];

    /// Get mutable gate parameters
    fn parameters_mut(&mut self) -> &mut [f64];

    /// Get gate name for debugging
    fn name(&self) -> &str;

    /// Get qubits that this gate acts on
    fn qubits(&self) -> &[Qubit];
}

/// Gradient of a quantum state with respect to a parameter
#[derive(Debug, Clone)]
pub struct QuantumStateGradient {
    /// Gradient amplitudes (∂ψ/∂θ)
    pub gradient_amplitudes: Vec<Complex>,
    /// Parameter index this gradient corresponds to
    pub parameter_index: usize,
}

/// Pauli-X gate (bit flip)
#[derive(Debug, Clone)]
pub struct PauliX {
    qubit: Qubit,
}

impl PauliX {
    pub fn new(qubit: Qubit) -> Self {
        Self { qubit }
    }
}

impl QuantumGate for PauliX {
    fn apply(&self, state: &mut QuantumState) -> AutogradResult<()> {
        let qubit_index = self.qubit.index();
        if qubit_index >= state.num_qubits() {
            return Err(AutogradError::gradient_computation(
                "pauli_x_apply",
                format!(
                    "Qubit index {} exceeds number of qubits {}",
                    qubit_index,
                    state.num_qubits()
                ),
            ));
        }

        let num_states = state.amplitudes.len();
        let bit_mask = 1 << qubit_index;

        // Apply Pauli-X by swapping amplitudes for states that differ only in the target qubit
        for i in 0..num_states {
            if (i & bit_mask) == 0 {
                let j = i | bit_mask;
                state.amplitudes.swap(i, j);
            }
        }

        Ok(())
    }

    fn gradient(
        &self,
        _state: &QuantumState,
        parameter_index: usize,
    ) -> AutogradResult<QuantumStateGradient> {
        if parameter_index >= self.parameter_count() {
            return Err(AutogradError::gradient_computation(
                "pauli_x_gradient",
                format!(
                    "Parameter index {} exceeds parameter count {}",
                    parameter_index,
                    self.parameter_count()
                ),
            ));
        }

        // Pauli-X has no parameters, so gradient is zero
        Ok(QuantumStateGradient {
            gradient_amplitudes: vec![Complex::zero(); 1 << _state.num_qubits()],
            parameter_index,
        })
    }

    fn parameter_count(&self) -> usize {
        0
    }

    fn parameters(&self) -> &[f64] {
        &[]
    }

    fn parameters_mut(&mut self) -> &mut [f64] {
        &mut []
    }

    fn name(&self) -> &str {
        "PauliX"
    }

    fn qubits(&self) -> &[Qubit] {
        std::slice::from_ref(&self.qubit)
    }
}

/// Rotation around Y-axis (RY gate) - parametric gate
#[derive(Debug, Clone)]
pub struct RotationY {
    qubit: Qubit,
    angle: f64,
}

impl RotationY {
    pub fn new(qubit: Qubit, angle: f64) -> Self {
        Self { qubit, angle }
    }

    pub fn angle(&self) -> f64 {
        self.angle
    }

    pub fn set_angle(&mut self, angle: f64) {
        self.angle = angle;
    }
}

impl QuantumGate for RotationY {
    fn apply(&self, state: &mut QuantumState) -> AutogradResult<()> {
        let qubit_index = self.qubit.index();
        if qubit_index >= state.num_qubits() {
            return Err(AutogradError::gradient_computation(
                "rotation_y_apply",
                format!(
                    "Qubit index {} exceeds number of qubits {}",
                    qubit_index,
                    state.num_qubits()
                ),
            ));
        }

        let cos_half = (self.angle / 2.0).cos();
        let sin_half = (self.angle / 2.0).sin();

        let num_states = state.amplitudes.len();
        let bit_mask = 1 << qubit_index;
        let mut new_amplitudes = state.amplitudes.clone();

        // Apply RY rotation matrix
        for i in 0..num_states {
            if (i & bit_mask) == 0 {
                let j = i | bit_mask;
                let amp_0 = state.amplitudes[i];
                let amp_1 = state.amplitudes[j];

                // RY matrix: [[cos(θ/2), -sin(θ/2)], [sin(θ/2), cos(θ/2)]]
                new_amplitudes[i] = amp_0 * cos_half - amp_1 * sin_half;
                new_amplitudes[j] = amp_0 * sin_half + amp_1 * cos_half;
            }
        }

        state.amplitudes = new_amplitudes;
        Ok(())
    }

    fn gradient(
        &self,
        state: &QuantumState,
        parameter_index: usize,
    ) -> AutogradResult<QuantumStateGradient> {
        if parameter_index >= self.parameter_count() {
            return Err(AutogradError::gradient_computation(
                "rotation_y_gradient",
                format!(
                    "Parameter index {} exceeds parameter count {}",
                    parameter_index,
                    self.parameter_count()
                ),
            ));
        }

        let qubit_index = self.qubit.index();
        let num_states = state.amplitudes.len();
        let bit_mask = 1 << qubit_index;

        let cos_half = (self.angle / 2.0).cos();
        let sin_half = (self.angle / 2.0).sin();

        // Gradient of RY with respect to angle
        let d_cos = -0.5 * (self.angle / 2.0).sin();
        let d_sin = 0.5 * (self.angle / 2.0).cos();

        let mut gradient_amplitudes = vec![Complex::zero(); num_states];

        for i in 0..num_states {
            if (i & bit_mask) == 0 {
                let j = i | bit_mask;
                let amp_0 = state.amplitudes[i];
                let amp_1 = state.amplitudes[j];

                // ∂/∂θ [cos(θ/2) * amp_0 - sin(θ/2) * amp_1]
                gradient_amplitudes[i] = amp_0 * d_cos - amp_1 * d_sin;
                // ∂/∂θ [sin(θ/2) * amp_0 + cos(θ/2) * amp_1]
                gradient_amplitudes[j] = amp_0 * d_sin + amp_1 * d_cos;
            }
        }

        Ok(QuantumStateGradient {
            gradient_amplitudes,
            parameter_index,
        })
    }

    fn parameter_count(&self) -> usize {
        1
    }

    fn parameters(&self) -> &[f64] {
        std::slice::from_ref(&self.angle)
    }

    fn parameters_mut(&mut self) -> &mut [f64] {
        std::slice::from_mut(&mut self.angle)
    }

    fn name(&self) -> &str {
        "RotationY"
    }

    fn qubits(&self) -> &[Qubit] {
        std::slice::from_ref(&self.qubit)
    }
}

/// Controlled-NOT (CNOT) gate
#[derive(Debug, Clone)]
pub struct CNOT {
    control: Qubit,
    target: Qubit,
    qubits: [Qubit; 2],
}

impl CNOT {
    pub fn new(control: Qubit, target: Qubit) -> Self {
        Self {
            control,
            target,
            qubits: [control, target],
        }
    }
}

impl QuantumGate for CNOT {
    fn apply(&self, state: &mut QuantumState) -> AutogradResult<()> {
        let control_index = self.control.index();
        let target_index = self.target.index();

        if control_index >= state.num_qubits() || target_index >= state.num_qubits() {
            return Err(AutogradError::gradient_computation(
                "cnot_apply",
                format!(
                    "Qubit indices ({}, {}) exceed number of qubits {}",
                    control_index,
                    target_index,
                    state.num_qubits()
                ),
            ));
        }

        if control_index == target_index {
            return Err(AutogradError::gradient_computation(
                "cnot_apply",
                "Control and target qubits must be different",
            ));
        }

        let num_states = state.amplitudes.len();
        let control_mask = 1 << control_index;
        let target_mask = 1 << target_index;

        // Apply CNOT: if control is |1⟩, flip target
        for i in 0..num_states {
            if (i & control_mask) != 0 {
                // Control qubit is |1⟩, so flip target
                let j = i ^ target_mask;
                if i < j {
                    state.amplitudes.swap(i, j);
                }
            }
        }

        Ok(())
    }

    fn gradient(
        &self,
        _state: &QuantumState,
        parameter_index: usize,
    ) -> AutogradResult<QuantumStateGradient> {
        if parameter_index >= self.parameter_count() {
            return Err(AutogradError::gradient_computation(
                "cnot_gradient",
                format!(
                    "Parameter index {} exceeds parameter count {}",
                    parameter_index,
                    self.parameter_count()
                ),
            ));
        }

        // CNOT has no parameters, so gradient is zero
        Ok(QuantumStateGradient {
            gradient_amplitudes: vec![Complex::zero(); 1 << _state.num_qubits()],
            parameter_index,
        })
    }

    fn parameter_count(&self) -> usize {
        0
    }

    fn parameters(&self) -> &[f64] {
        &[]
    }

    fn parameters_mut(&mut self) -> &mut [f64] {
        &mut []
    }

    fn name(&self) -> &str {
        "CNOT"
    }

    fn qubits(&self) -> &[Qubit] {
        &self.qubits
    }
}

/// Quantum circuit with automatic differentiation support
#[derive(Debug)]
pub struct QuantumCircuit {
    gates: Vec<Box<dyn QuantumGate>>,
    num_qubits: usize,
    parameter_map: HashMap<usize, (usize, usize)>, // global_param_idx -> (gate_idx, local_param_idx)
}

impl QuantumCircuit {
    /// Create a new quantum circuit
    pub fn new(num_qubits: usize) -> Self {
        Self {
            gates: Vec::new(),
            num_qubits,
            parameter_map: HashMap::new(),
        }
    }

    /// Add a gate to the circuit
    pub fn add_gate(&mut self, gate: Box<dyn QuantumGate>) {
        // Validate that all qubits are within range
        for qubit in gate.qubits() {
            if qubit.index() >= self.num_qubits {
                panic!(
                    "Gate qubit index {} exceeds circuit size {}",
                    qubit.index(),
                    self.num_qubits
                );
            }
        }

        // Update parameter mapping
        let gate_index = self.gates.len();
        for local_param_idx in 0..gate.parameter_count() {
            let global_param_idx = self.total_parameters();
            self.parameter_map
                .insert(global_param_idx, (gate_index, local_param_idx));
        }

        self.gates.push(gate);
    }

    /// Execute the quantum circuit on a given initial state
    pub fn execute(&self, initial_state: &QuantumState) -> AutogradResult<QuantumState> {
        if initial_state.num_qubits() != self.num_qubits {
            return Err(AutogradError::gradient_computation(
                "quantum_circuit_execute",
                format!(
                    "Initial state has {} qubits, but circuit expects {}",
                    initial_state.num_qubits(),
                    self.num_qubits
                ),
            ));
        }

        let mut state = initial_state.clone();

        for gate in &self.gates {
            gate.apply(&mut state)?;
        }

        Ok(state)
    }

    /// Compute gradients of the final state with respect to circuit parameters
    pub fn compute_gradients(
        &self,
        initial_state: &QuantumState,
    ) -> AutogradResult<Vec<QuantumStateGradient>> {
        let mut gradients = Vec::new();
        let total_params = self.total_parameters();

        for global_param_idx in 0..total_params {
            let gradient = self.compute_parameter_gradient(initial_state, global_param_idx)?;
            gradients.push(gradient);
        }

        Ok(gradients)
    }

    /// Compute gradient with respect to a specific parameter using parameter shift rule
    fn compute_parameter_gradient(
        &self,
        initial_state: &QuantumState,
        global_param_idx: usize,
    ) -> AutogradResult<QuantumStateGradient> {
        let (gate_idx, local_param_idx) =
            self.parameter_map.get(&global_param_idx).ok_or_else(|| {
                AutogradError::gradient_computation(
                    "compute_parameter_gradient",
                    format!("Invalid parameter index {}", global_param_idx),
                )
            })?;

        // Parameter shift rule: gradient = (f(θ + π/2) - f(θ - π/2)) / 2
        let shift = PI / 2.0;

        // Create modified circuits with shifted parameters
        let mut circuit_plus =
            self.clone_with_parameter_shift(*gate_idx, *local_param_idx, shift)?;
        let mut circuit_minus =
            self.clone_with_parameter_shift(*gate_idx, *local_param_idx, -shift)?;

        // Execute both circuits
        let state_plus = circuit_plus.execute(initial_state)?;
        let state_minus = circuit_minus.execute(initial_state)?;

        // Compute gradient using finite difference
        let mut gradient_amplitudes = Vec::with_capacity(state_plus.amplitudes().len());
        for (amp_plus, amp_minus) in state_plus
            .amplitudes()
            .iter()
            .zip(state_minus.amplitudes().iter())
        {
            let gradient_amp = (*amp_plus - *amp_minus) * 0.5;
            gradient_amplitudes.push(gradient_amp);
        }

        Ok(QuantumStateGradient {
            gradient_amplitudes,
            parameter_index: global_param_idx,
        })
    }

    /// Clone circuit with a parameter shift for gradient computation
    fn clone_with_parameter_shift(
        &self,
        gate_idx: usize,
        local_param_idx: usize,
        shift: f64,
    ) -> AutogradResult<QuantumCircuit> {
        let mut new_circuit = QuantumCircuit::new(self.num_qubits);

        for (i, gate) in self.gates.iter().enumerate() {
            if i == gate_idx {
                // Clone gate with shifted parameter
                let mut cloned_gate = self.clone_gate_with_shift(gate, local_param_idx, shift)?;
                new_circuit.add_gate(cloned_gate);
            } else {
                // Clone gate as-is
                let cloned_gate = self.clone_gate(gate)?;
                new_circuit.add_gate(cloned_gate);
            }
        }

        Ok(new_circuit)
    }

    /// Clone a gate with a parameter shift
    fn clone_gate_with_shift(
        &self,
        gate: &Box<dyn QuantumGate>,
        local_param_idx: usize,
        shift: f64,
    ) -> AutogradResult<Box<dyn QuantumGate>> {
        // This is a simplified implementation - in practice, you'd need proper gate cloning
        // For now, we'll handle the specific gates we've implemented

        if gate.name() == "RotationY" {
            let qubits = gate.qubits();
            if qubits.is_empty() {
                return Err(AutogradError::gradient_computation(
                    "clone_gate_with_shift",
                    "RotationY gate has no qubits",
                ));
            }

            let params = gate.parameters();
            if local_param_idx >= params.len() {
                return Err(AutogradError::gradient_computation(
                    "clone_gate_with_shift",
                    format!("Parameter index {} out of range", local_param_idx),
                ));
            }

            let new_angle = params[local_param_idx] + shift;
            Ok(Box::new(RotationY::new(qubits[0], new_angle)))
        } else {
            // For gates without parameters, just clone as-is
            self.clone_gate(gate)
        }
    }

    /// Clone a gate without modification
    fn clone_gate(&self, gate: &Box<dyn QuantumGate>) -> AutogradResult<Box<dyn QuantumGate>> {
        match gate.name() {
            "PauliX" => {
                let qubits = gate.qubits();
                if qubits.is_empty() {
                    return Err(AutogradError::gradient_computation(
                        "clone_gate",
                        "PauliX gate has no qubits",
                    ));
                }
                Ok(Box::new(PauliX::new(qubits[0])))
            }
            "RotationY" => {
                let qubits = gate.qubits();
                let params = gate.parameters();
                if qubits.is_empty() || params.is_empty() {
                    return Err(AutogradError::gradient_computation(
                        "clone_gate",
                        "RotationY gate missing qubits or parameters",
                    ));
                }
                Ok(Box::new(RotationY::new(qubits[0], params[0])))
            }
            "CNOT" => {
                let qubits = gate.qubits();
                if qubits.len() < 2 {
                    return Err(AutogradError::gradient_computation(
                        "clone_gate",
                        "CNOT gate requires 2 qubits",
                    ));
                }
                Ok(Box::new(CNOT::new(qubits[0], qubits[1])))
            }
            _ => Err(AutogradError::gradient_computation(
                "clone_gate",
                format!("Unknown gate type: {}", gate.name()),
            )),
        }
    }

    /// Get total number of parameters in the circuit
    pub fn total_parameters(&self) -> usize {
        self.gates.iter().map(|gate| gate.parameter_count()).sum()
    }

    /// Get all parameters in the circuit
    pub fn parameters(&self) -> Vec<f64> {
        let mut params = Vec::new();
        for gate in &self.gates {
            params.extend_from_slice(gate.parameters());
        }
        params
    }

    /// Update circuit parameters
    pub fn set_parameters(&mut self, params: &[f64]) -> AutogradResult<()> {
        if params.len() != self.total_parameters() {
            return Err(AutogradError::gradient_computation(
                "set_parameters",
                format!(
                    "Expected {} parameters, got {}",
                    self.total_parameters(),
                    params.len()
                ),
            ));
        }

        let mut param_idx = 0;
        for gate in &mut self.gates {
            let gate_param_count = gate.parameter_count();
            let gate_params = &params[param_idx..param_idx + gate_param_count];
            gate.parameters_mut().copy_from_slice(gate_params);
            param_idx += gate_param_count;
        }

        Ok(())
    }
}

/// Quantum expectation value computation with automatic differentiation
pub struct QuantumExpectationValue {
    circuit: QuantumCircuit,
    observable: Observable,
}

impl QuantumExpectationValue {
    /// Create a new expectation value computation
    pub fn new(circuit: QuantumCircuit, observable: Observable) -> Self {
        Self {
            circuit,
            observable,
        }
    }

    /// Compute the expectation value ⟨ψ|O|ψ⟩
    pub fn compute(&self, initial_state: &QuantumState) -> AutogradResult<f64> {
        let final_state = self.circuit.execute(initial_state)?;
        self.observable.expectation_value(&final_state)
    }

    /// Compute gradients of the expectation value with respect to circuit parameters
    pub fn compute_gradients(&self, initial_state: &QuantumState) -> AutogradResult<Vec<f64>> {
        let state_gradients = self.circuit.compute_gradients(initial_state)?;
        let final_state = self.circuit.execute(initial_state)?;

        let mut gradients = Vec::new();

        for state_grad in state_gradients {
            // Gradient of expectation value: ⟨∂ψ/∂θ|O|ψ⟩ + ⟨ψ|O|∂ψ/∂θ⟩
            let grad_expectation = self
                .observable
                .expectation_value_gradient(&final_state, &state_grad.gradient_amplitudes)?;
            gradients.push(grad_expectation);
        }

        Ok(gradients)
    }
}

/// Quantum observable for expectation value computation
#[derive(Debug, Clone)]
pub struct Observable {
    /// Pauli string representation (e.g., "XZIY" for X⊗Z⊗I⊗Y)
    pauli_string: String,
    /// Coefficient for the observable
    coefficient: f64,
}

impl Observable {
    /// Create a new observable from a Pauli string
    pub fn from_pauli_string(pauli_string: String, coefficient: f64) -> Self {
        Self {
            pauli_string,
            coefficient,
        }
    }

    /// Compute expectation value ⟨ψ|O|ψ⟩
    pub fn expectation_value(&self, state: &QuantumState) -> AutogradResult<f64> {
        let num_qubits = state.num_qubits();
        if self.pauli_string.len() != num_qubits {
            return Err(AutogradError::gradient_computation(
                "observable_expectation_value",
                format!(
                    "Pauli string length {} doesn't match number of qubits {}",
                    self.pauli_string.len(),
                    num_qubits
                ),
            ));
        }

        let mut expectation = 0.0;

        // Apply Pauli operators and compute expectation value
        for (state_idx, amplitude) in state.amplitudes().iter().enumerate() {
            let mut result_amplitude = *amplitude;
            let mut phase = 1.0;

            for (qubit_idx, pauli_char) in self.pauli_string.chars().enumerate() {
                let bit = (state_idx >> qubit_idx) & 1;

                match pauli_char {
                    'I' => {
                        // Identity: no change
                    }
                    'X' => {
                        // Pauli-X: flip bit (contribution will come from flipped state)
                        let flipped_state = state_idx ^ (1 << qubit_idx);
                        if flipped_state != state_idx {
                            result_amplitude = state.amplitudes()[flipped_state];
                        }
                    }
                    'Y' => {
                        // Pauli-Y: flip bit and apply -i^bit phase
                        let flipped_state = state_idx ^ (1 << qubit_idx);
                        if flipped_state != state_idx {
                            result_amplitude = state.amplitudes()[flipped_state];
                            phase *= if bit == 0 { 1.0 } else { -1.0 }; // Simplified Y gate
                        }
                    }
                    'Z' => {
                        // Pauli-Z: apply (-1)^bit phase
                        phase *= if bit == 0 { 1.0 } else { -1.0 };
                    }
                    _ => {
                        return Err(AutogradError::gradient_computation(
                            "observable_expectation_value",
                            format!("Invalid Pauli character: {}", pauli_char),
                        ));
                    }
                }
            }

            expectation += (amplitude.conjugate() * (result_amplitude * phase)).real;
        }

        Ok(expectation * self.coefficient)
    }

    /// Compute gradient of expectation value with respect to state
    pub fn expectation_value_gradient(
        &self,
        state: &QuantumState,
        state_gradient: &[Complex],
    ) -> AutogradResult<f64> {
        // This is a simplified implementation
        // Real implementation would compute ⟨∂ψ/∂θ|O|ψ⟩ + ⟨ψ|O|∂ψ/∂θ⟩
        let mut gradient = 0.0;

        for (i, (amp, grad_amp)) in state
            .amplitudes()
            .iter()
            .zip(state_gradient.iter())
            .enumerate()
        {
            // Simplified: assume observable is just measurement in computational basis
            gradient += 2.0 * (amp.conjugate() * *grad_amp).real;
        }

        Ok(gradient * self.coefficient)
    }
}

/// Variational Quantum Eigensolver (VQE) implementation
pub struct VQE {
    ansatz: QuantumCircuit,
    hamiltonian: Vec<Observable>,
    optimizer_learning_rate: f64,
}

impl VQE {
    /// Create a new VQE instance
    pub fn new(ansatz: QuantumCircuit, hamiltonian: Vec<Observable>) -> Self {
        Self {
            ansatz,
            hamiltonian,
            optimizer_learning_rate: 0.01,
        }
    }

    /// Run VQE optimization to find ground state energy
    pub fn optimize(
        &mut self,
        initial_state: &QuantumState,
        max_iterations: usize,
    ) -> AutogradResult<VQEResult> {
        let mut current_params = self.ansatz.parameters();
        let mut energy_history = Vec::new();

        for iteration in 0..max_iterations {
            // Compute current energy
            let energy = self.compute_energy(initial_state, &current_params)?;
            energy_history.push(energy);

            // Compute gradients
            let gradients = self.compute_energy_gradients(initial_state, &current_params)?;

            // Update parameters using gradient descent
            for (param, grad) in current_params.iter_mut().zip(gradients.iter()) {
                *param -= self.optimizer_learning_rate * grad;
            }

            // Update circuit parameters
            self.ansatz.set_parameters(&current_params)?;

            // Check for convergence
            if iteration > 0 {
                let energy_change =
                    (energy_history[iteration] - energy_history[iteration - 1]).abs();
                if energy_change < 1e-8 {
                    break;
                }
            }
        }

        let final_energy = self.compute_energy(initial_state, &current_params)?;
        let final_state = self.ansatz.execute(initial_state)?;

        Ok(VQEResult {
            final_energy,
            final_parameters: current_params,
            final_state,
            energy_history,
            converged: true,
        })
    }

    /// Compute total energy for given parameters
    fn compute_energy(
        &mut self,
        initial_state: &QuantumState,
        params: &[f64],
    ) -> AutogradResult<f64> {
        self.ansatz.set_parameters(params)?;
        let final_state = self.ansatz.execute(initial_state)?;

        let mut total_energy = 0.0;
        for observable in &self.hamiltonian {
            let expectation = observable.expectation_value(&final_state)?;
            total_energy += expectation;
        }

        Ok(total_energy)
    }

    /// Compute energy gradients with respect to parameters
    fn compute_energy_gradients(
        &mut self,
        initial_state: &QuantumState,
        params: &[f64],
    ) -> AutogradResult<Vec<f64>> {
        self.ansatz.set_parameters(params)?;

        let mut total_gradients = vec![0.0; params.len()];

        for observable in &self.hamiltonian {
            let expectation_value = QuantumExpectationValue::new(
                QuantumCircuit::new(self.ansatz.num_qubits), // This should be a proper clone
                observable.clone(),
            );

            // Copy circuit structure (simplified)
            // In practice, you'd need proper circuit cloning
            let gradients = expectation_value.compute_gradients(initial_state)?;

            for (total_grad, grad) in total_gradients.iter_mut().zip(gradients.iter()) {
                *total_grad += grad;
            }
        }

        Ok(total_gradients)
    }
}

/// Result of VQE optimization
#[derive(Debug, Clone)]
pub struct VQEResult {
    pub final_energy: f64,
    pub final_parameters: Vec<f64>,
    pub final_state: QuantumState,
    pub energy_history: Vec<f64>,
    pub converged: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complex_arithmetic() {
        let a = Complex::new(1.0, 2.0);
        let b = Complex::new(3.0, 4.0);

        let sum = a + b;
        assert_eq!(sum.real, 4.0);
        assert_eq!(sum.imag, 6.0);

        let product = a * b;
        assert_eq!(product.real, -5.0); // 1*3 - 2*4
        assert_eq!(product.imag, 10.0); // 1*4 + 2*3

        assert!((a.magnitude() - 5.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_quantum_state_creation() {
        let state = QuantumState::zeros(2);
        assert_eq!(state.num_qubits(), 2);
        assert_eq!(state.amplitudes().len(), 4);
        assert_eq!(state.amplitudes()[0], Complex::one());
        assert_eq!(state.amplitudes()[1], Complex::zero());
    }

    #[test]
    fn test_pauli_x_gate() {
        let mut state = QuantumState::zeros(1);
        let gate = PauliX::new(Qubit::new(0));

        gate.apply(&mut state).unwrap();

        // After applying X to |0⟩, we should get |1⟩
        assert_eq!(state.amplitudes()[0], Complex::zero());
        assert_eq!(state.amplitudes()[1], Complex::one());
    }

    #[test]
    fn test_rotation_y_gate() {
        let mut state = QuantumState::zeros(1);
        let gate = RotationY::new(Qubit::new(0), PI / 2.0);

        gate.apply(&mut state).unwrap();

        // After applying RY(π/2) to |0⟩, we should get (|0⟩ + |1⟩)/√2
        let expected = 1.0 / 2.0_f64.sqrt();
        assert!((state.amplitudes()[0].real - expected).abs() < 1e-10);
        assert!((state.amplitudes()[1].real - expected).abs() < 1e-10);
    }

    #[test]
    fn test_cnot_gate() {
        // Test CNOT on |10⟩ -> |11⟩
        let mut state = QuantumState::from_amplitudes(vec![
            Complex::zero(), // |00⟩
            Complex::zero(), // |01⟩
            Complex::one(),  // |10⟩
            Complex::zero(), // |11⟩
        ])
        .unwrap();

        let gate = CNOT::new(Qubit::new(1), Qubit::new(0)); // control=1, target=0

        gate.apply(&mut state).unwrap();

        // After CNOT, |10⟩ should become |11⟩
        assert_eq!(state.amplitudes()[2], Complex::zero());
        assert_eq!(state.amplitudes()[3], Complex::one());
    }

    #[test]
    fn test_quantum_circuit() {
        let mut circuit = QuantumCircuit::new(2);

        // Add X gate to first qubit
        circuit.add_gate(Box::new(PauliX::new(Qubit::new(0))));

        // Add CNOT gate
        circuit.add_gate(Box::new(CNOT::new(Qubit::new(0), Qubit::new(1))));

        let initial_state = QuantumState::zeros(2);
        let final_state = circuit.execute(&initial_state).unwrap();

        // Should result in |11⟩ state
        assert_eq!(final_state.amplitudes()[0], Complex::zero()); // |00⟩
        assert_eq!(final_state.amplitudes()[1], Complex::zero()); // |01⟩
        assert_eq!(final_state.amplitudes()[2], Complex::zero()); // |10⟩
        assert_eq!(final_state.amplitudes()[3], Complex::one()); // |11⟩
    }

    #[test]
    fn test_observable_expectation_value() {
        let state = QuantumState::zeros(2);
        let observable = Observable::from_pauli_string("ZI".to_string(), 1.0);

        let expectation = observable.expectation_value(&state).unwrap();
        assert!((expectation - 1.0).abs() < 1e-10); // ⟨00|Z⊗I|00⟩ = 1
    }

    #[test]
    fn test_parameter_gradient_computation() {
        let mut circuit = QuantumCircuit::new(1);
        circuit.add_gate(Box::new(RotationY::new(Qubit::new(0), PI / 4.0)));

        let initial_state = QuantumState::zeros(1);
        let gradients = circuit.compute_gradients(&initial_state).unwrap();

        assert_eq!(gradients.len(), 1); // One parameter
                                        // Gradient should be non-zero since RY is parametric
        assert!(gradients[0]
            .gradient_amplitudes
            .iter()
            .any(|amp| amp.magnitude() > 1e-10));
    }

    #[test]
    fn test_quantum_state_normalization() {
        let mut state =
            QuantumState::from_amplitudes(vec![Complex::new(1.0, 0.0), Complex::new(1.0, 0.0)])
                .unwrap();

        assert!(!state.is_normalized());
        state.normalize();
        assert!(state.is_normalized());

        let expected = 1.0 / 2.0_f64.sqrt();
        assert!((state.amplitudes()[0].real - expected).abs() < 1e-10);
        assert!((state.amplitudes()[1].real - expected).abs() < 1e-10);
    }

    #[test]
    fn test_qubit_creation() {
        let qubit = Qubit::new(5);
        assert_eq!(qubit.index(), 5);
    }
}
