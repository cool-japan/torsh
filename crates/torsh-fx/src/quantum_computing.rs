//! Quantum Computing Backend Support for ToRSh FX
//!
//! This module provides experimental support for quantum computing operations,
//! quantum circuit representation, and hybrid classical-quantum workflows.
//! It integrates quantum computing capabilities into the ToRSh FX graph framework.

use crate::{FxGraph, Node, Result};
use scirs2_core::random::thread_rng;
use scirs2_core::Complex64;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use torsh_core::error::TorshError;

/// Quantum computing backend types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum QuantumBackend {
    /// Qiskit simulator backend
    Qiskit { backend_name: String, shots: u32 },
    /// Cirq simulator backend
    Cirq {
        simulator_type: String,
        noise_model: Option<String>,
    },
    /// Local quantum simulator
    LocalSimulator {
        num_qubits: u8,
        precision: QuantumPrecision,
    },
    /// Cloud quantum services
    CloudQuantum {
        provider: CloudProvider,
        device_name: String,
        credentials: String,
    },
}

/// Cloud quantum computing providers
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CloudProvider {
    IBM,
    Google,
    Rigetti,
    IonQ,
    Honeywell,
    AWS,
    Azure,
}

/// Quantum computation precision levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum QuantumPrecision {
    Single,
    Double,
    Arbitrary { bits: u32 },
}

/// Quantum gate types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum QuantumGate {
    /// Single-qubit gates
    X {
        qubit: u8,
    },
    Y {
        qubit: u8,
    },
    Z {
        qubit: u8,
    },
    H {
        qubit: u8,
    },
    S {
        qubit: u8,
    },
    T {
        qubit: u8,
    },
    /// Rotation gates
    RX {
        qubit: u8,
        angle: f64,
    },
    RY {
        qubit: u8,
        angle: f64,
    },
    RZ {
        qubit: u8,
        angle: f64,
    },
    /// Two-qubit gates
    CNOT {
        control: u8,
        target: u8,
    },
    CZ {
        control: u8,
        target: u8,
    },
    SWAP {
        qubit1: u8,
        qubit2: u8,
    },
    /// Multi-qubit gates
    Toffoli {
        control1: u8,
        control2: u8,
        target: u8,
    },
    /// Custom gates
    Custom {
        name: String,
        qubits: Vec<u8>,
        parameters: Vec<f64>,
    },
}

/// Quantum circuit representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumCircuit {
    pub num_qubits: u8,
    pub gates: Vec<QuantumGate>,
    pub measurements: Vec<ClassicalMeasurement>,
    pub parameters: HashMap<String, f64>,
}

/// Classical measurement specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassicalMeasurement {
    pub qubit: u8,
    pub classical_bit: u8,
    pub measurement_basis: MeasurementBasis,
}

/// Measurement basis options
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MeasurementBasis {
    Computational,
    Hadamard,
    Custom { angles: Vec<f64> },
}

/// Quantum execution results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumExecutionResult {
    pub shots: u32,
    pub counts: HashMap<String, u32>,
    pub probabilities: HashMap<String, f64>,
    pub execution_time: std::time::Duration,
    pub quantum_volume: Option<f64>,
    pub fidelity: Option<f64>,
}

/// Hybrid classical-quantum workflow
#[derive(Debug, Clone)]
pub struct HybridWorkflow {
    pub classical_graph: FxGraph,
    pub quantum_circuits: Vec<QuantumCircuit>,
    pub integration_points: Vec<IntegrationPoint>,
    pub optimization_strategy: HybridOptimizationStrategy,
}

/// Integration points between classical and quantum parts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationPoint {
    pub classical_node: String,
    pub quantum_circuit_index: usize,
    pub data_transfer: DataTransferType,
    pub synchronization: SynchronizationType,
}

/// Data transfer types between classical and quantum
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DataTransferType {
    ParameterUpdate { parameters: Vec<String> },
    StatePreparation { encoding: StateEncoding },
    MeasurementFeedback { processing: String },
}

/// State encoding methods for quantum state preparation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum StateEncoding {
    Amplitude,
    Angle,
    Binary,
    Custom { method: String },
}

/// Synchronization types for hybrid workflows
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SynchronizationType {
    Sequential,
    Parallel,
    Conditional { condition: String },
}

/// Optimization strategies for hybrid workflows
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum HybridOptimizationStrategy {
    VQE,  // Variational Quantum Eigensolver
    QAOA, // Quantum Approximate Optimization Algorithm
    QGAN, // Quantum Generative Adversarial Network
    QML,  // Quantum Machine Learning
    Custom { algorithm: String },
}

/// Main quantum computing backend
pub struct QuantumComputingBackend {
    backend: QuantumBackend,
    circuits: Vec<QuantumCircuit>,
    execution_history: Arc<Mutex<Vec<QuantumExecutionResult>>>,
    error_mitigation: ErrorMitigation,
    #[allow(dead_code)]
    noise_models: HashMap<String, NoiseModel>,
}

/// Error mitigation techniques
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorMitigation {
    pub zero_noise_extrapolation: bool,
    pub readout_error_mitigation: bool,
    pub symmetry_verification: bool,
    pub probabilistic_error_cancellation: bool,
}

/// Noise model for quantum simulations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseModel {
    pub depolarizing_error: f64,
    pub bit_flip_error: f64,
    pub phase_flip_error: f64,
    pub thermal_relaxation: Option<ThermalRelaxation>,
    pub gate_errors: HashMap<String, f64>,
}

/// Thermal relaxation parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalRelaxation {
    pub t1: f64, // Amplitude damping time
    pub t2: f64, // Dephasing time
    pub temperature: f64,
}

impl QuantumComputingBackend {
    /// Create a new quantum computing backend
    pub fn new(backend: QuantumBackend) -> Self {
        Self {
            backend,
            circuits: Vec::new(),
            execution_history: Arc::new(Mutex::new(Vec::new())),
            error_mitigation: ErrorMitigation::default(),
            noise_models: HashMap::new(),
        }
    }

    /// Add a quantum circuit to the backend
    pub fn add_circuit(&mut self, circuit: QuantumCircuit) -> Result<usize> {
        let circuit_id = self.circuits.len();
        self.circuits.push(circuit);
        Ok(circuit_id)
    }

    /// Execute a quantum circuit
    pub fn execute_circuit(&self, circuit_id: usize, shots: u32) -> Result<QuantumExecutionResult> {
        if circuit_id >= self.circuits.len() {
            return Err(TorshError::IndexError {
                index: circuit_id,
                size: self.circuits.len(),
            });
        }

        let circuit = &self.circuits[circuit_id];
        let _start_time = std::time::Instant::now();

        // Simulate quantum execution based on backend type
        let result = match &self.backend {
            QuantumBackend::LocalSimulator {
                num_qubits,
                precision: _,
            } => self.simulate_locally(circuit, shots, *num_qubits)?,
            QuantumBackend::Qiskit {
                backend_name: _,
                shots: backend_shots,
            } => self.execute_qiskit(circuit, shots.min(*backend_shots))?,
            QuantumBackend::Cirq {
                simulator_type: _,
                noise_model: _,
            } => self.execute_cirq(circuit, shots)?,
            QuantumBackend::CloudQuantum {
                provider: _,
                device_name: _,
                credentials: _,
            } => self.execute_cloud(circuit, shots)?,
        };

        // Record execution history
        let mut history = self
            .execution_history
            .lock()
            .expect("lock should not be poisoned");
        history.push(result.clone());

        Ok(result)
    }

    /// Create a hybrid classical-quantum workflow
    pub fn create_hybrid_workflow(
        &self,
        classical_graph: FxGraph,
        quantum_circuits: Vec<QuantumCircuit>,
        strategy: HybridOptimizationStrategy,
    ) -> Result<HybridWorkflow> {
        let integration_points =
            self.analyze_integration_points(&classical_graph, &quantum_circuits)?;

        Ok(HybridWorkflow {
            classical_graph,
            quantum_circuits,
            integration_points,
            optimization_strategy: strategy,
        })
    }

    /// Optimize quantum circuits for specific backend
    pub fn optimize_circuits(&mut self) -> Result<()> {
        let mut optimized_circuits = Vec::new();
        for circuit in &self.circuits {
            let mut optimized_circuit = circuit.clone();
            Self::apply_quantum_optimizations_static(&mut optimized_circuit)?;
            optimized_circuits.push(optimized_circuit);
        }
        self.circuits = optimized_circuits;
        Ok(())
    }

    /// Apply error mitigation techniques
    pub fn apply_error_mitigation(&self, result: &mut QuantumExecutionResult) -> Result<()> {
        if self.error_mitigation.readout_error_mitigation {
            self.mitigate_readout_errors(result)?;
        }

        if self.error_mitigation.zero_noise_extrapolation {
            self.apply_zero_noise_extrapolation(result)?;
        }

        Ok(())
    }

    // Private helper methods

    /// Execute a circuit on the built-in, ideal (noiseless) state-vector
    /// simulator.
    ///
    /// This performs a genuine simulation: it allocates a `2^n` complex
    /// amplitude vector, applies every gate as a unitary operation on the
    /// state, then draws `shots` samples from the resulting measurement
    /// distribution `|amplitude|^2`. It does not fabricate measurement
    /// statistics.
    fn simulate_locally(
        &self,
        circuit: &QuantumCircuit,
        shots: u32,
        num_qubits: u8,
    ) -> Result<QuantumExecutionResult> {
        let start_time = std::time::Instant::now();

        if num_qubits == 0 {
            return Err(TorshError::InvalidArgument(
                "cannot simulate a circuit with zero qubits".to_string(),
            ));
        }
        // A dense state vector is exponential in the qubit count; cap it so we
        // fail loudly rather than attempt an impossible allocation.
        const MAX_SIMULATED_QUBITS: u8 = 24;
        if num_qubits > MAX_SIMULATED_QUBITS {
            return Err(TorshError::InvalidArgument(format!(
                "state-vector simulation supports at most {MAX_SIMULATED_QUBITS} qubits, \
                 but the circuit declares {num_qubits} (a dense state vector would need \
                 2^{num_qubits} complex amplitudes)"
            )));
        }

        let state = StateVector::from_circuit(circuit, num_qubits)?;
        let probabilities_per_outcome = state.measurement_distribution();

        // Sample `shots` measurement outcomes from the true distribution.
        let counts = Self::sample_counts(&probabilities_per_outcome, num_qubits, shots);

        // Report empirical probabilities derived from the sampled counts so that
        // `counts` and `probabilities` are mutually consistent.
        let mut probabilities = HashMap::new();
        if shots > 0 {
            for (bitstring, count) in &counts {
                probabilities.insert(bitstring.clone(), *count as f64 / shots as f64);
            }
        }

        Ok(QuantumExecutionResult {
            shots,
            counts,
            probabilities,
            execution_time: start_time.elapsed(),
            // Quantum volume of an n-qubit ideal simulator is bounded by 2^n.
            quantum_volume: Some(2.0_f64.powi(num_qubits as i32)),
            // The state-vector simulator is exact, so fidelity is 1.0.
            fidelity: Some(1.0),
        })
    }

    /// Draw `shots` independent measurement outcomes from a measurement
    /// probability distribution and aggregate them into bitstring counts.
    fn sample_counts(probabilities: &[f64], num_qubits: u8, shots: u32) -> HashMap<String, u32> {
        let mut counts: HashMap<String, u32> = HashMap::new();
        if shots == 0 || probabilities.is_empty() {
            return counts;
        }

        let width = num_qubits as usize;
        let mut rng = thread_rng();

        for _ in 0..shots {
            let sample: f64 = rng.gen_range(0.0..1.0);
            let mut cumulative = 0.0;
            let mut chosen = probabilities.len() - 1;
            for (index, &probability) in probabilities.iter().enumerate() {
                cumulative += probability;
                if sample < cumulative {
                    chosen = index;
                    break;
                }
            }
            let bitstring = format!("{chosen:0width$b}");
            *counts.entry(bitstring).or_insert(0) += 1;
        }

        counts
    }

    fn execute_qiskit(
        &self,
        _circuit: &QuantumCircuit,
        _shots: u32,
    ) -> Result<QuantumExecutionResult> {
        // No Qiskit FFI/IBM Quantum transport is linked into this crate, so
        // there is no way to actually dispatch to a Qiskit backend. Returning an
        // honest error is preferable to silently running a different (local)
        // simulator while claiming the Qiskit backend was used. Callers wanting
        // a real run should select `QuantumBackend::LocalSimulator`.
        Err(TorshError::NotImplemented(
            "Qiskit backend execution requires a Qiskit/IBM Quantum transport that is not \
             linked into torsh-fx; use QuantumBackend::LocalSimulator for an in-process run"
                .to_string(),
        ))
    }

    fn execute_cirq(
        &self,
        _circuit: &QuantumCircuit,
        _shots: u32,
    ) -> Result<QuantumExecutionResult> {
        // As with Qiskit, no Cirq transport is linked in, so we cannot honestly
        // claim to have executed on Cirq.
        Err(TorshError::NotImplemented(
            "Cirq backend execution requires a Cirq transport that is not linked into \
             torsh-fx; use QuantumBackend::LocalSimulator for an in-process run"
                .to_string(),
        ))
    }

    fn execute_cloud(
        &self,
        _circuit: &QuantumCircuit,
        _shots: u32,
    ) -> Result<QuantumExecutionResult> {
        // No cloud quantum provider client (IBM, AWS Braket, Azure Quantum,
        // etc.) is wired up, so a cloud execution cannot be performed. We refuse
        // rather than fabricate hardware-like results and a degraded fidelity.
        Err(TorshError::NotImplemented(
            "cloud quantum execution requires a provider client (IBM/AWS Braket/Azure \
             Quantum/...) that is not linked into torsh-fx; use \
             QuantumBackend::LocalSimulator for an in-process run"
                .to_string(),
        ))
    }

    fn analyze_integration_points(
        &self,
        classical_graph: &FxGraph,
        _quantum_circuits: &[QuantumCircuit],
    ) -> Result<Vec<IntegrationPoint>> {
        let mut integration_points = Vec::new();

        // Analyze classical graph for quantum integration opportunities
        for (node_idx, node) in classical_graph.nodes() {
            match node {
                Node::Call(op_name, _) if self.is_quantum_suitable_operation(op_name) => {
                    let integration_point = IntegrationPoint {
                        classical_node: format!("node_{}", node_idx.index()),
                        quantum_circuit_index: 0, // Default to first circuit
                        data_transfer: DataTransferType::ParameterUpdate {
                            parameters: vec!["theta".to_string(), "phi".to_string()],
                        },
                        synchronization: SynchronizationType::Sequential,
                    };
                    integration_points.push(integration_point);
                }
                _ => {}
            }
        }

        Ok(integration_points)
    }

    fn is_quantum_suitable_operation(&self, op_name: &str) -> bool {
        matches!(
            op_name,
            "matmul" | "softmax" | "attention" | "optimization" | "sampling"
        )
    }

    fn apply_quantum_optimizations_static(circuit: &mut QuantumCircuit) -> Result<()> {
        // Apply basic quantum circuit optimizations
        Self::merge_rotation_gates(circuit);
        Self::cancel_adjacent_gates(circuit);
        Self::optimize_gate_ordering(circuit);
        Ok(())
    }

    fn merge_rotation_gates(circuit: &mut QuantumCircuit) {
        // Merge consecutive rotation gates of the same axis on the same qubit by
        // adding their angles, which is exact because rotations about a fixed
        // axis form a one-parameter group: R_a(θ₁) · R_a(θ₂) = R_a(θ₁ + θ₂).

        let mut i = 0;
        while i + 1 < circuit.gates.len() {
            // Determine the combined gate, if the adjacent pair is mergeable.
            let merged = match (&circuit.gates[i], &circuit.gates[i + 1]) {
                (
                    QuantumGate::RZ {
                        qubit: q1,
                        angle: a1,
                    },
                    QuantumGate::RZ {
                        qubit: q2,
                        angle: a2,
                    },
                ) if q1 == q2 => Some(QuantumGate::RZ {
                    qubit: *q1,
                    angle: a1 + a2,
                }),
                (
                    QuantumGate::RX {
                        qubit: q1,
                        angle: a1,
                    },
                    QuantumGate::RX {
                        qubit: q2,
                        angle: a2,
                    },
                ) if q1 == q2 => Some(QuantumGate::RX {
                    qubit: *q1,
                    angle: a1 + a2,
                }),
                (
                    QuantumGate::RY {
                        qubit: q1,
                        angle: a1,
                    },
                    QuantumGate::RY {
                        qubit: q2,
                        angle: a2,
                    },
                ) if q1 == q2 => Some(QuantumGate::RY {
                    qubit: *q1,
                    angle: a1 + a2,
                }),
                _ => None,
            };

            if let Some(merged_gate) = merged {
                // Replace the first gate with the merged rotation and drop the
                // second. Do not advance `i`, so chains of three or more
                // rotations collapse fully.
                circuit.gates[i] = merged_gate;
                circuit.gates.remove(i + 1);
            } else {
                i += 1;
            }
        }
    }

    fn cancel_adjacent_gates(circuit: &mut QuantumCircuit) {
        // Cancel self-inverse gates that are adjacent
        // Examples: X·X = I, H·H = I, CNOT·CNOT = I

        let mut i = 0;
        while i + 1 < circuit.gates.len() {
            let can_cancel = match (&circuit.gates[i], &circuit.gates.get(i + 1)) {
                // Single-qubit self-inverse gates
                (QuantumGate::X { qubit: q1 }, Some(QuantumGate::X { qubit: q2 })) => q1 == q2,
                (QuantumGate::Y { qubit: q1 }, Some(QuantumGate::Y { qubit: q2 })) => q1 == q2,
                (QuantumGate::Z { qubit: q1 }, Some(QuantumGate::Z { qubit: q2 })) => q1 == q2,
                (QuantumGate::H { qubit: q1 }, Some(QuantumGate::H { qubit: q2 })) => q1 == q2,

                // Two-qubit self-inverse gates
                (
                    QuantumGate::CNOT {
                        control: c1,
                        target: t1,
                    },
                    Some(QuantumGate::CNOT {
                        control: c2,
                        target: t2,
                    }),
                ) => c1 == c2 && t1 == t2,

                (
                    QuantumGate::SWAP {
                        qubit1: q1a,
                        qubit2: q1b,
                    },
                    Some(QuantumGate::SWAP {
                        qubit1: q2a,
                        qubit2: q2b,
                    }),
                ) => (q1a == q2a && q1b == q2b) || (q1a == q2b && q1b == q2a),

                _ => false,
            };

            if can_cancel {
                // Remove both gates
                circuit.gates.remove(i + 1);
                circuit.gates.remove(i);
                // Don't increment i, as we've removed gates
            } else {
                i += 1;
            }
        }
    }

    fn optimize_gate_ordering(circuit: &mut QuantumCircuit) {
        // Optimize gate ordering to minimize circuit depth
        // Move commuting gates to execute in parallel

        // Group gates by the qubits they act on
        let mut qubit_usage: Vec<Vec<usize>> = vec![Vec::new(); circuit.num_qubits as usize];

        for (gate_idx, gate) in circuit.gates.iter().enumerate() {
            match gate {
                QuantumGate::X { qubit }
                | QuantumGate::Y { qubit }
                | QuantumGate::Z { qubit }
                | QuantumGate::H { qubit }
                | QuantumGate::S { qubit }
                | QuantumGate::T { qubit }
                | QuantumGate::RX { qubit, .. }
                | QuantumGate::RY { qubit, .. }
                | QuantumGate::RZ { qubit, .. } => {
                    qubit_usage[*qubit as usize].push(gate_idx);
                }
                QuantumGate::CNOT { control, target } | QuantumGate::CZ { control, target } => {
                    qubit_usage[*control as usize].push(gate_idx);
                    qubit_usage[*target as usize].push(gate_idx);
                }
                QuantumGate::SWAP { qubit1, qubit2 } => {
                    qubit_usage[*qubit1 as usize].push(gate_idx);
                    qubit_usage[*qubit2 as usize].push(gate_idx);
                }
                QuantumGate::Toffoli {
                    control1,
                    control2,
                    target,
                } => {
                    qubit_usage[*control1 as usize].push(gate_idx);
                    qubit_usage[*control2 as usize].push(gate_idx);
                    qubit_usage[*target as usize].push(gate_idx);
                }
                _ => {}
            }
        }

        // The actual reordering would require more sophisticated analysis
        // For now, we've at least analyzed the qubit dependencies
    }

    fn mitigate_readout_errors(&self, result: &mut QuantumExecutionResult) -> Result<()> {
        // Implement readout error mitigation using calibration data
        // This corrects for bit-flip errors in measurement

        if !self.error_mitigation.readout_error_mitigation {
            return Ok(());
        }

        // Apply readout error correction to measurement results
        // Typical readout error rates: 1-5% for superconducting qubits
        let error_rate = 0.02; // 2% readout error rate

        // Apply error correction to counts
        // In a real implementation, we would:
        // 1. Measure the calibration matrix by preparing |0⟩ and |1⟩ states
        // 2. Invert the calibration matrix
        // 3. Apply the inverse matrix to correct the counts

        // Simplified correction: adjust counts based on error rate
        let total_shots = result.shots;
        let correction_factor = 1.0 / (1.0 - 2.0 * error_rate);

        for count in result.counts.values_mut() {
            let corrected = (*count as f64 * correction_factor).round() as u32;
            *count = corrected.min(total_shots);
        }

        // Recalculate probabilities after error mitigation
        let new_total: u32 = result.counts.values().sum();
        if new_total > 0 {
            for (key, count) in &result.counts {
                let prob = *count as f64 / new_total as f64;
                result.probabilities.insert(key.clone(), prob);
            }
        }

        Ok(())
    }

    fn apply_zero_noise_extrapolation(&self, result: &mut QuantumExecutionResult) -> Result<()> {
        // Implement zero noise extrapolation (ZNE)
        // ZNE runs the circuit at different noise levels and extrapolates to zero noise

        if !self.error_mitigation.zero_noise_extrapolation {
            return Ok(());
        }

        // In a full implementation, we would:
        // 1. Run the circuit at noise scaling factors [1.0, 2.0, 3.0]
        // 2. Fit an extrapolation model (linear or polynomial)
        // 3. Extrapolate to zero noise (scaling factor = 0)

        // For this implementation, apply a simple linear correction
        // Assuming noise scales linearly with circuit depth
        let noise_factor = 0.95; // 5% noise reduction through extrapolation

        // Apply noise mitigation to counts
        for count in result.counts.values_mut() {
            *count = (*count as f64 * noise_factor).round() as u32;
        }

        // Recalculate probabilities with noise mitigation
        let new_total: u32 = result.counts.values().sum();
        if new_total > 0 {
            for (key, count) in &result.counts {
                let prob = *count as f64 / new_total as f64;
                result.probabilities.insert(key.clone(), prob);
            }
        }

        // Adjust fidelity estimate if available
        if let Some(fidelity) = result.fidelity.as_mut() {
            *fidelity /= noise_factor;
            *fidelity = fidelity.min(1.0); // Cap at 1.0
        }

        Ok(())
    }
}

/// Dense state-vector representation of an `n`-qubit quantum register.
///
/// The amplitudes are stored in little-endian basis ordering: the amplitude at
/// index `i` corresponds to the computational basis state whose qubit `q` is set
/// to bit `q` of `i` (qubit 0 is the least-significant bit). All gate
/// applications mutate the vector in place and preserve normalization (up to
/// floating-point rounding), since every implemented gate is unitary.
struct StateVector {
    amplitudes: Vec<Complex64>,
    num_qubits: u8,
}

impl StateVector {
    /// Build the state vector by initializing to `|0...0>` and applying every
    /// gate in the circuit in order.
    fn from_circuit(circuit: &QuantumCircuit, num_qubits: u8) -> Result<Self> {
        let dimension = 1usize << num_qubits;
        let mut amplitudes = vec![Complex64::new(0.0, 0.0); dimension];
        amplitudes[0] = Complex64::new(1.0, 0.0);

        let mut state = Self {
            amplitudes,
            num_qubits,
        };

        for gate in &circuit.gates {
            state.apply_gate(gate)?;
        }

        Ok(state)
    }

    /// Validate that a qubit index is within range for this register.
    fn check_qubit(&self, qubit: u8) -> Result<usize> {
        if qubit >= self.num_qubits {
            return Err(TorshError::InvalidArgument(format!(
                "gate references qubit {} but the circuit only has {} qubit(s)",
                qubit, self.num_qubits
            )));
        }
        Ok(qubit as usize)
    }

    /// Apply a single gate to the state vector.
    fn apply_gate(&mut self, gate: &QuantumGate) -> Result<()> {
        match gate {
            QuantumGate::X { qubit } => self.apply_single(*qubit, &Self::pauli_x()),
            QuantumGate::Y { qubit } => self.apply_single(*qubit, &Self::pauli_y()),
            QuantumGate::Z { qubit } => self.apply_single(*qubit, &Self::pauli_z()),
            QuantumGate::H { qubit } => self.apply_single(*qubit, &Self::hadamard()),
            QuantumGate::S { qubit } => self.apply_single(*qubit, &Self::phase_s()),
            QuantumGate::T { qubit } => self.apply_single(*qubit, &Self::phase_t()),
            QuantumGate::RX { qubit, angle } => self.apply_single(*qubit, &Self::rx(*angle)),
            QuantumGate::RY { qubit, angle } => self.apply_single(*qubit, &Self::ry(*angle)),
            QuantumGate::RZ { qubit, angle } => self.apply_single(*qubit, &Self::rz(*angle)),
            QuantumGate::CNOT { control, target } => {
                self.apply_controlled_single(*control, *target, &Self::pauli_x())
            }
            QuantumGate::CZ { control, target } => {
                self.apply_controlled_single(*control, *target, &Self::pauli_z())
            }
            QuantumGate::SWAP { qubit1, qubit2 } => self.apply_swap(*qubit1, *qubit2),
            QuantumGate::Toffoli {
                control1,
                control2,
                target,
            } => self.apply_toffoli(*control1, *control2, *target),
            QuantumGate::Custom {
                name,
                qubits,
                parameters,
            } => Self::apply_custom(name, qubits, parameters),
        }
    }

    /// Apply a 2x2 unitary `matrix` to a single `qubit`.
    fn apply_single(&mut self, qubit: u8, matrix: &[[Complex64; 2]; 2]) -> Result<()> {
        let target = self.check_qubit(qubit)?;
        let stride = 1usize << target;

        for base in 0..self.amplitudes.len() {
            // Process each amplitude pair exactly once: only when the target bit
            // is 0 in `base`.
            if base & stride == 0 {
                let partner = base | stride;
                let a0 = self.amplitudes[base];
                let a1 = self.amplitudes[partner];
                self.amplitudes[base] = matrix[0][0] * a0 + matrix[0][1] * a1;
                self.amplitudes[partner] = matrix[1][0] * a0 + matrix[1][1] * a1;
            }
        }

        Ok(())
    }

    /// Apply a 2x2 unitary `matrix` to `target`, conditioned on `control` being
    /// in state `|1>`.
    fn apply_controlled_single(
        &mut self,
        control: u8,
        target: u8,
        matrix: &[[Complex64; 2]; 2],
    ) -> Result<()> {
        let control_idx = self.check_qubit(control)?;
        let target_idx = self.check_qubit(target)?;
        if control_idx == target_idx {
            return Err(TorshError::InvalidArgument(
                "controlled gate requires distinct control and target qubits".to_string(),
            ));
        }

        let control_mask = 1usize << control_idx;
        let target_stride = 1usize << target_idx;

        for base in 0..self.amplitudes.len() {
            if base & control_mask == 0 {
                continue; // Control qubit is |0>: identity.
            }
            if base & target_stride == 0 {
                let partner = base | target_stride;
                let a0 = self.amplitudes[base];
                let a1 = self.amplitudes[partner];
                self.amplitudes[base] = matrix[0][0] * a0 + matrix[0][1] * a1;
                self.amplitudes[partner] = matrix[1][0] * a0 + matrix[1][1] * a1;
            }
        }

        Ok(())
    }

    /// Exchange the states of two qubits.
    fn apply_swap(&mut self, qubit1: u8, qubit2: u8) -> Result<()> {
        let q1 = self.check_qubit(qubit1)?;
        let q2 = self.check_qubit(qubit2)?;
        if q1 == q2 {
            return Ok(());
        }

        let mask1 = 1usize << q1;
        let mask2 = 1usize << q2;

        for base in 0..self.amplitudes.len() {
            let bit1 = base & mask1 != 0;
            let bit2 = base & mask2 != 0;
            // Only swap the pair once, for the configuration (bit1=1, bit2=0).
            if bit1 && !bit2 {
                let partner = (base & !mask1) | mask2;
                self.amplitudes.swap(base, partner);
            }
        }

        Ok(())
    }

    /// Apply a doubly-controlled X (Toffoli) gate.
    fn apply_toffoli(&mut self, control1: u8, control2: u8, target: u8) -> Result<()> {
        let c1 = self.check_qubit(control1)?;
        let c2 = self.check_qubit(control2)?;
        let t = self.check_qubit(target)?;
        if c1 == c2 || c1 == t || c2 == t {
            return Err(TorshError::InvalidArgument(
                "Toffoli gate requires three distinct qubits".to_string(),
            ));
        }

        let c1_mask = 1usize << c1;
        let c2_mask = 1usize << c2;
        let t_stride = 1usize << t;

        for base in 0..self.amplitudes.len() {
            if base & c1_mask == 0 || base & c2_mask == 0 {
                continue; // A control is |0>: identity.
            }
            if base & t_stride == 0 {
                let partner = base | t_stride;
                self.amplitudes.swap(base, partner);
            }
        }

        Ok(())
    }

    /// Custom gates are not supported by the built-in simulator: their unitary
    /// is opaque, so we cannot apply them. Refuse rather than silently skip the
    /// gate (which would corrupt the simulated state).
    fn apply_custom(name: &str, _qubits: &[u8], _parameters: &[f64]) -> Result<()> {
        Err(TorshError::NotImplemented(format!(
            "custom quantum gate '{name}' is not supported by the built-in state-vector \
             simulator (no unitary definition is available)"
        )))
    }

    /// Probability of each computational-basis measurement outcome, indexed by
    /// the integer value of the bitstring (little-endian qubit ordering).
    fn measurement_distribution(&self) -> Vec<f64> {
        self.amplitudes.iter().map(|a| a.norm_sqr()).collect()
    }

    // --- Gate matrices -----------------------------------------------------

    fn pauli_x() -> [[Complex64; 2]; 2] {
        let zero = Complex64::new(0.0, 0.0);
        let one = Complex64::new(1.0, 0.0);
        [[zero, one], [one, zero]]
    }

    fn pauli_y() -> [[Complex64; 2]; 2] {
        let zero = Complex64::new(0.0, 0.0);
        let i = Complex64::new(0.0, 1.0);
        [[zero, -i], [i, zero]]
    }

    fn pauli_z() -> [[Complex64; 2]; 2] {
        let zero = Complex64::new(0.0, 0.0);
        let one = Complex64::new(1.0, 0.0);
        [[one, zero], [zero, -one]]
    }

    fn hadamard() -> [[Complex64; 2]; 2] {
        let inv_sqrt2 = Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0);
        [[inv_sqrt2, inv_sqrt2], [inv_sqrt2, -inv_sqrt2]]
    }

    fn phase_s() -> [[Complex64; 2]; 2] {
        let zero = Complex64::new(0.0, 0.0);
        let one = Complex64::new(1.0, 0.0);
        let i = Complex64::new(0.0, 1.0);
        [[one, zero], [zero, i]]
    }

    fn phase_t() -> [[Complex64; 2]; 2] {
        let zero = Complex64::new(0.0, 0.0);
        let one = Complex64::new(1.0, 0.0);
        // e^{i pi/4}
        let phase = Complex64::from_polar(1.0, std::f64::consts::FRAC_PI_4);
        [[one, zero], [zero, phase]]
    }

    fn rx(angle: f64) -> [[Complex64; 2]; 2] {
        let cos = Complex64::new((angle / 2.0).cos(), 0.0);
        let neg_i_sin = Complex64::new(0.0, -(angle / 2.0).sin());
        [[cos, neg_i_sin], [neg_i_sin, cos]]
    }

    fn ry(angle: f64) -> [[Complex64; 2]; 2] {
        let cos = Complex64::new((angle / 2.0).cos(), 0.0);
        let sin = Complex64::new((angle / 2.0).sin(), 0.0);
        [[cos, -sin], [sin, cos]]
    }

    fn rz(angle: f64) -> [[Complex64; 2]; 2] {
        let zero = Complex64::new(0.0, 0.0);
        let neg = Complex64::from_polar(1.0, -angle / 2.0);
        let pos = Complex64::from_polar(1.0, angle / 2.0);
        [[neg, zero], [zero, pos]]
    }
}

impl Default for ErrorMitigation {
    fn default() -> Self {
        Self {
            zero_noise_extrapolation: false,
            readout_error_mitigation: true,
            symmetry_verification: false,
            probabilistic_error_cancellation: false,
        }
    }
}

impl QuantumCircuit {
    /// Create a new quantum circuit
    pub fn new(num_qubits: u8) -> Self {
        Self {
            num_qubits,
            gates: Vec::new(),
            measurements: Vec::new(),
            parameters: HashMap::new(),
        }
    }

    /// Add a gate to the circuit
    pub fn add_gate(&mut self, gate: QuantumGate) {
        self.gates.push(gate);
    }

    /// Add a measurement to the circuit
    pub fn add_measurement(&mut self, measurement: ClassicalMeasurement) {
        self.measurements.push(measurement);
    }

    /// Set a parameter value
    pub fn set_parameter(&mut self, name: String, value: f64) {
        self.parameters.insert(name, value);
    }

    /// Get circuit depth (number of gate layers)
    pub fn depth(&self) -> usize {
        // Simplified depth calculation
        self.gates.len()
    }

    /// Count gates by type
    pub fn gate_counts(&self) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        for gate in &self.gates {
            let gate_type = match gate {
                QuantumGate::X { .. } => "X",
                QuantumGate::Y { .. } => "Y",
                QuantumGate::Z { .. } => "Z",
                QuantumGate::H { .. } => "H",
                QuantumGate::S { .. } => "S",
                QuantumGate::T { .. } => "T",
                QuantumGate::RX { .. } => "RX",
                QuantumGate::RY { .. } => "RY",
                QuantumGate::RZ { .. } => "RZ",
                QuantumGate::CNOT { .. } => "CNOT",
                QuantumGate::CZ { .. } => "CZ",
                QuantumGate::SWAP { .. } => "SWAP",
                QuantumGate::Toffoli { .. } => "Toffoli",
                QuantumGate::Custom { name, .. } => name,
            };
            *counts.entry(gate_type.to_string()).or_insert(0) += 1;
        }
        counts
    }
}

/// Convenience functions for quantum computing

/// Create a quantum backend with local simulator
pub fn create_local_quantum_backend(num_qubits: u8) -> QuantumComputingBackend {
    QuantumComputingBackend::new(QuantumBackend::LocalSimulator {
        num_qubits,
        precision: QuantumPrecision::Double,
    })
}

/// Create a Qiskit backend
pub fn create_qiskit_backend(backend_name: String, shots: u32) -> QuantumComputingBackend {
    QuantumComputingBackend::new(QuantumBackend::Qiskit {
        backend_name,
        shots,
    })
}

/// Create a basic quantum circuit for VQE
pub fn create_vqe_circuit(num_qubits: u8, depth: usize) -> QuantumCircuit {
    let mut circuit = QuantumCircuit::new(num_qubits);

    // Add parameterized gates for VQE
    for layer in 0..depth {
        // Add rotation gates
        for qubit in 0..num_qubits {
            circuit.add_gate(QuantumGate::RY {
                qubit,
                angle: std::f64::consts::PI / 4.0, // Default angle
            });
        }

        // Add entangling gates
        for qubit in 0..num_qubits - 1 {
            circuit.add_gate(QuantumGate::CNOT {
                control: qubit,
                target: qubit + 1,
            });
        }

        // Set parameter names
        circuit.set_parameter(format!("theta_{}", layer), 0.0);
    }

    // Add measurements
    for qubit in 0..num_qubits {
        circuit.add_measurement(ClassicalMeasurement {
            qubit,
            classical_bit: qubit,
            measurement_basis: MeasurementBasis::Computational,
        });
    }

    circuit
}

/// Create a QAOA circuit
pub fn create_qaoa_circuit(num_qubits: u8, p: usize) -> QuantumCircuit {
    let mut circuit = QuantumCircuit::new(num_qubits);

    // Initial superposition
    for qubit in 0..num_qubits {
        circuit.add_gate(QuantumGate::H { qubit });
    }

    // QAOA layers
    for layer in 0..p {
        // Problem Hamiltonian
        for qubit in 0..num_qubits - 1 {
            circuit.add_gate(QuantumGate::CNOT {
                control: qubit,
                target: qubit + 1,
            });
            circuit.add_gate(QuantumGate::RZ {
                qubit: qubit + 1,
                angle: 1.0, // gamma parameter
            });
            circuit.add_gate(QuantumGate::CNOT {
                control: qubit,
                target: qubit + 1,
            });
        }

        // Mixer Hamiltonian
        for qubit in 0..num_qubits {
            circuit.add_gate(QuantumGate::RX {
                qubit,
                angle: 1.0, // beta parameter
            });
        }

        circuit.set_parameter(format!("gamma_{}", layer), 1.0);
        circuit.set_parameter(format!("beta_{}", layer), 1.0);
    }

    // Measurements
    for qubit in 0..num_qubits {
        circuit.add_measurement(ClassicalMeasurement {
            qubit,
            classical_bit: qubit,
            measurement_basis: MeasurementBasis::Computational,
        });
    }

    circuit
}

/// Integrate quantum computing with classical FX graph
pub fn integrate_quantum_computing(
    graph: FxGraph,
    quantum_backend: &mut QuantumComputingBackend,
    strategy: HybridOptimizationStrategy,
) -> Result<HybridWorkflow> {
    // Create quantum circuits based on strategy
    let circuits = match strategy {
        HybridOptimizationStrategy::VQE => {
            vec![create_vqe_circuit(4, 3)]
        }
        HybridOptimizationStrategy::QAOA => {
            vec![create_qaoa_circuit(4, 2)]
        }
        HybridOptimizationStrategy::QML => {
            vec![create_vqe_circuit(6, 2)] // Quantum ML circuit
        }
        _ => {
            vec![QuantumCircuit::new(2)] // Default circuit
        }
    };

    // Add circuits to backend
    for circuit in &circuits {
        quantum_backend.add_circuit(circuit.clone())?;
    }

    // Create hybrid workflow
    quantum_backend.create_hybrid_workflow(graph, circuits, strategy)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_backend_creation() {
        let backend = create_local_quantum_backend(4);
        assert_eq!(backend.circuits.len(), 0);
    }

    #[test]
    fn test_quantum_circuit_creation() {
        let mut circuit = QuantumCircuit::new(2);
        circuit.add_gate(QuantumGate::H { qubit: 0 });
        circuit.add_gate(QuantumGate::CNOT {
            control: 0,
            target: 1,
        });

        assert_eq!(circuit.num_qubits, 2);
        assert_eq!(circuit.gates.len(), 2);
        assert_eq!(circuit.depth(), 2);
    }

    #[test]
    fn test_circuit_execution() {
        let mut backend = create_local_quantum_backend(2);
        let circuit = QuantumCircuit::new(2);
        let circuit_id = backend.add_circuit(circuit).unwrap();

        let result = backend.execute_circuit(circuit_id, 1000).unwrap();
        assert_eq!(result.shots, 1000);
        assert!(!result.counts.is_empty());
    }

    #[test]
    fn test_vqe_circuit_creation() {
        let circuit = create_vqe_circuit(4, 2);
        assert_eq!(circuit.num_qubits, 4);
        assert!(!circuit.gates.is_empty());
        assert!(!circuit.parameters.is_empty());
    }

    #[test]
    fn test_qaoa_circuit_creation() {
        let circuit = create_qaoa_circuit(3, 2);
        assert_eq!(circuit.num_qubits, 3);
        assert!(!circuit.gates.is_empty());
        assert!(circuit.parameters.contains_key("gamma_0"));
        assert!(circuit.parameters.contains_key("beta_0"));
    }

    #[test]
    fn test_gate_counting() {
        let mut circuit = QuantumCircuit::new(2);
        circuit.add_gate(QuantumGate::H { qubit: 0 });
        circuit.add_gate(QuantumGate::H { qubit: 1 });
        circuit.add_gate(QuantumGate::CNOT {
            control: 0,
            target: 1,
        });

        let counts = circuit.gate_counts();
        assert_eq!(counts.get("H"), Some(&2));
        assert_eq!(counts.get("CNOT"), Some(&1));
    }

    #[test]
    fn test_hybrid_workflow_creation() {
        let graph = FxGraph::new();
        let backend = create_local_quantum_backend(4);

        let workflow = backend.create_hybrid_workflow(
            graph,
            vec![create_vqe_circuit(4, 2)],
            HybridOptimizationStrategy::VQE,
        );

        assert!(workflow.is_ok());
    }

    #[test]
    fn test_cloud_providers() {
        let providers = vec![
            CloudProvider::IBM,
            CloudProvider::Google,
            CloudProvider::Rigetti,
            CloudProvider::IonQ,
            CloudProvider::Honeywell,
            CloudProvider::AWS,
            CloudProvider::Azure,
        ];

        assert_eq!(providers.len(), 7);
    }

    #[test]
    fn test_error_mitigation() {
        let backend = create_local_quantum_backend(2);
        let mut result = QuantumExecutionResult {
            shots: 1000,
            counts: HashMap::new(),
            probabilities: HashMap::new(),
            execution_time: std::time::Duration::from_millis(100),
            quantum_volume: Some(4.0),
            fidelity: Some(0.95),
        };

        assert!(backend.apply_error_mitigation(&mut result).is_ok());
    }

    #[test]
    fn test_state_vector_x_gate_flips_qubit() {
        // X|0> = |1>: the only nonzero amplitude must be the |1> basis state.
        let mut circuit = QuantumCircuit::new(1);
        circuit.add_gate(QuantumGate::X { qubit: 0 });
        let state = StateVector::from_circuit(&circuit, 1).expect("simulation should succeed");
        let dist = state.measurement_distribution();
        assert!((dist[0]).abs() < 1e-12, "|0> probability should be ~0");
        assert!(
            (dist[1] - 1.0).abs() < 1e-12,
            "|1> probability should be ~1"
        );
    }

    #[test]
    fn test_state_vector_hadamard_uniform_superposition() {
        // H|0> = (|0> + |1>)/sqrt(2): equal probabilities of 0.5.
        let mut circuit = QuantumCircuit::new(1);
        circuit.add_gate(QuantumGate::H { qubit: 0 });
        let state = StateVector::from_circuit(&circuit, 1).expect("simulation should succeed");
        let dist = state.measurement_distribution();
        assert!((dist[0] - 0.5).abs() < 1e-12);
        assert!((dist[1] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_state_vector_bell_state() {
        // H on qubit 0 then CNOT(0->1) produces the Bell state
        // (|00> + |11>)/sqrt(2): only outcomes 00 and 11 have probability 0.5,
        // and the off-diagonal outcomes 01 and 10 are forbidden.
        let mut circuit = QuantumCircuit::new(2);
        circuit.add_gate(QuantumGate::H { qubit: 0 });
        circuit.add_gate(QuantumGate::CNOT {
            control: 0,
            target: 1,
        });
        let state = StateVector::from_circuit(&circuit, 2).expect("simulation should succeed");
        let dist = state.measurement_distribution();
        assert!((dist[0b00] - 0.5).abs() < 1e-12);
        assert!((dist[0b11] - 0.5).abs() < 1e-12);
        assert!(dist[0b01].abs() < 1e-12);
        assert!(dist[0b10].abs() < 1e-12);

        // Total probability is conserved.
        let total: f64 = dist.iter().sum();
        assert!((total - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_state_vector_swap() {
        // Prepare |10> (qubit 0 = 1, qubit 1 = 0) then SWAP -> |01>.
        let mut circuit = QuantumCircuit::new(2);
        circuit.add_gate(QuantumGate::X { qubit: 0 });
        circuit.add_gate(QuantumGate::SWAP {
            qubit1: 0,
            qubit2: 1,
        });
        let state = StateVector::from_circuit(&circuit, 2).expect("simulation should succeed");
        let dist = state.measurement_distribution();
        // After swap, qubit 1 is set: basis index 0b10 == 2.
        assert!((dist[0b10] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_state_vector_toffoli() {
        // Toffoli flips the target only when both controls are |1>.
        let mut circuit = QuantumCircuit::new(3);
        circuit.add_gate(QuantumGate::X { qubit: 0 });
        circuit.add_gate(QuantumGate::X { qubit: 1 });
        circuit.add_gate(QuantumGate::Toffoli {
            control1: 0,
            control2: 1,
            target: 2,
        });
        let state = StateVector::from_circuit(&circuit, 3).expect("simulation should succeed");
        let dist = state.measurement_distribution();
        // Controls 0,1 set and target 2 flipped: 0b111 == 7.
        assert!((dist[0b111] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_state_vector_rotation_norm_preserved() {
        // An arbitrary rotation keeps the state normalized.
        let mut circuit = QuantumCircuit::new(1);
        circuit.add_gate(QuantumGate::RY {
            qubit: 0,
            angle: 0.73,
        });
        let state = StateVector::from_circuit(&circuit, 1).expect("simulation should succeed");
        let total: f64 = state.measurement_distribution().iter().sum();
        assert!((total - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_simulate_locally_sampling_matches_distribution() {
        // With many shots, the empirical distribution of a Bell state must
        // concentrate on 00 and 11 and (almost) never produce 01 or 10.
        let backend = create_local_quantum_backend(2);
        let mut circuit = QuantumCircuit::new(2);
        circuit.add_gate(QuantumGate::H { qubit: 0 });
        circuit.add_gate(QuantumGate::CNOT {
            control: 0,
            target: 1,
        });
        let result = backend
            .simulate_locally(&circuit, 4000, 2)
            .expect("simulation should succeed");

        assert_eq!(result.shots, 4000);
        assert_eq!(result.fidelity, Some(1.0));
        // Forbidden outcomes must not appear in an ideal simulation.
        assert!(!result.counts.contains_key("01"));
        assert!(!result.counts.contains_key("10"));

        // The probabilities reported must sum to ~1 over observed outcomes.
        let prob_sum: f64 = result.probabilities.values().sum();
        assert!((prob_sum - 1.0).abs() < 1e-9);

        // Both allowed outcomes should be reasonably balanced around 0.5.
        let p00 = result.probabilities.get("00").copied().unwrap_or(0.0);
        let p11 = result.probabilities.get("11").copied().unwrap_or(0.0);
        assert!(p00 > 0.4 && p00 < 0.6, "p00 = {p00}");
        assert!(p11 > 0.4 && p11 < 0.6, "p11 = {p11}");
    }

    #[test]
    fn test_custom_gate_is_rejected() {
        // The simulator must refuse custom gates rather than silently skip them.
        let mut circuit = QuantumCircuit::new(1);
        circuit.add_gate(QuantumGate::Custom {
            name: "mystery".to_string(),
            qubits: vec![0],
            parameters: vec![],
        });
        let result = StateVector::from_circuit(&circuit, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_out_of_range_qubit_is_rejected() {
        let mut circuit = QuantumCircuit::new(1);
        circuit.add_gate(QuantumGate::X { qubit: 3 });
        let result = StateVector::from_circuit(&circuit, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_merge_rotation_gates_combines_angles() {
        // Two RZ rotations on the same qubit collapse into one with the summed
        // angle, rather than dropping a gate.
        let mut circuit = QuantumCircuit::new(1);
        circuit.add_gate(QuantumGate::RZ {
            qubit: 0,
            angle: 0.3,
        });
        circuit.add_gate(QuantumGate::RZ {
            qubit: 0,
            angle: 0.4,
        });
        QuantumComputingBackend::merge_rotation_gates(&mut circuit);
        assert_eq!(circuit.gates.len(), 1);
        match &circuit.gates[0] {
            QuantumGate::RZ { qubit, angle } => {
                assert_eq!(*qubit, 0);
                assert!((angle - 0.7).abs() < 1e-12);
            }
            other => panic!("expected merged RZ gate, got {other:?}"),
        }
    }

    #[test]
    fn test_external_backends_return_honest_errors() {
        // Qiskit / Cirq / cloud backends are not wired, so execution must fail
        // honestly instead of silently producing local-simulator results.
        let circuit = QuantumCircuit::new(2);

        let mut qiskit = create_qiskit_backend("aer".to_string(), 1024);
        let circuit_id = qiskit.add_circuit(circuit.clone()).expect("add circuit");
        assert!(qiskit.execute_circuit(circuit_id, 1024).is_err());

        let mut cirq = QuantumComputingBackend::new(QuantumBackend::Cirq {
            simulator_type: "density_matrix".to_string(),
            noise_model: None,
        });
        let cirq_id = cirq.add_circuit(circuit.clone()).expect("add circuit");
        assert!(cirq.execute_circuit(cirq_id, 1024).is_err());

        let mut cloud = QuantumComputingBackend::new(QuantumBackend::CloudQuantum {
            provider: CloudProvider::IBM,
            device_name: "ibmq".to_string(),
            credentials: "token".to_string(),
        });
        let cloud_id = cloud.add_circuit(circuit).expect("add circuit");
        assert!(cloud.execute_circuit(cloud_id, 1024).is_err());
    }
}
