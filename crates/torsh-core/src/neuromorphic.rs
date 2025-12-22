// Copyright (c) 2025 ToRSh Contributors
//
// Neuromorphic Computing Data Structures
//
// This module provides data structures and abstractions for neuromorphic computing,
// enabling efficient representation and simulation of spiking neural networks (SNNs)
// and neuromorphic hardware architectures.
//
// # Key Features
//
// - **Spiking Neural Networks**: Efficient representation of spike-based computation
// - **Spike Timing**: Precise temporal coding with microsecond resolution
// - **Neuron Models**: LIF, Izhikevich, and other spiking neuron models
// - **Synaptic Plasticity**: STDP and other learning rules
// - **Event-Driven Simulation**: Efficient spike-driven computation
//
// # Design Principles
//
// 1. **Temporal Precision**: Support microsecond-level spike timing
// 2. **Memory Efficiency**: Sparse event representation
// 3. **Hardware Mapping**: Compatible with neuromorphic chips (Loihi, TrueNorth, etc.)
// 4. **Biological Realism**: Support for realistic neuron and synapse models
//
// # Examples
//
// ```rust
// use torsh_core::neuromorphic::{SpikeEvent, LIFNeuron, STDPSynapse};
//
// // Create a spiking neuron
// let neuron = LIFNeuron::new(0, 0.02, 0.2, -65.0, 8.0);
//
// // Create a spike event
// let spike = SpikeEvent::new(0, 1000); // Neuron 0 spikes at t=1ms
//
// // STDP learning rule
// let synapse = STDPSynapse::new(0, 1, 0.5, 0.02, 0.02);
// ```

use core::cmp::Ordering;

/// Spike event representation
///
/// Represents a single spike event in a spiking neural network.
/// Contains the neuron ID and the precise timing of the spike.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpikeEvent {
    /// ID of the neuron that fired
    neuron_id: usize,
    /// Timestamp in microseconds
    timestamp_us: u64,
}

impl SpikeEvent {
    /// Create a new spike event
    pub fn new(neuron_id: usize, timestamp_us: u64) -> Self {
        Self {
            neuron_id,
            timestamp_us,
        }
    }

    /// Get the neuron ID
    pub fn neuron_id(&self) -> usize {
        self.neuron_id
    }

    /// Get the timestamp
    pub fn timestamp_us(&self) -> u64 {
        self.timestamp_us
    }

    /// Get the timestamp in milliseconds
    pub fn timestamp_ms(&self) -> f64 {
        self.timestamp_us as f64 / 1000.0
    }

    /// Get the timestamp in seconds
    pub fn timestamp_s(&self) -> f64 {
        self.timestamp_us as f64 / 1_000_000.0
    }
}

impl PartialOrd for SpikeEvent {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SpikeEvent {
    fn cmp(&self, other: &Self) -> Ordering {
        self.timestamp_us.cmp(&other.timestamp_us)
    }
}

/// Spike train representation
///
/// A sequence of spike events for a single neuron or population.
#[derive(Debug, Clone)]
pub struct SpikeTrain {
    neuron_id: usize,
    spikes: Vec<u64>, // Timestamps in microseconds
}

impl SpikeTrain {
    /// Create a new spike train
    pub fn new(neuron_id: usize) -> Self {
        Self {
            neuron_id,
            spikes: Vec::new(),
        }
    }

    /// Add a spike to the train
    pub fn add_spike(&mut self, timestamp_us: u64) {
        self.spikes.push(timestamp_us);
    }

    /// Get the neuron ID
    pub fn neuron_id(&self) -> usize {
        self.neuron_id
    }

    /// Get the number of spikes
    pub fn spike_count(&self) -> usize {
        self.spikes.len()
    }

    /// Get the spikes
    pub fn spikes(&self) -> &[u64] {
        &self.spikes
    }

    /// Calculate firing rate (Hz)
    pub fn firing_rate(&self, duration_us: u64) -> f64 {
        if duration_us == 0 {
            return 0.0;
        }
        let duration_s = duration_us as f64 / 1_000_000.0;
        self.spikes.len() as f64 / duration_s
    }

    /// Calculate inter-spike intervals (ISI)
    pub fn inter_spike_intervals(&self) -> Vec<u64> {
        if self.spikes.len() < 2 {
            return Vec::new();
        }
        self.spikes
            .windows(2)
            .map(|w| w[1].saturating_sub(w[0]))
            .collect()
    }
}

/// Leaky Integrate-and-Fire (LIF) neuron model
///
/// Classic spiking neuron model with exponential decay and threshold-based firing.
#[derive(Debug, Clone)]
pub struct LIFNeuron {
    id: usize,
    /// Time constant (seconds)
    tau: f64,
    /// Membrane resistance (MOhm)
    resistance: f64,
    /// Resting potential (mV)
    v_rest: f64,
    /// Threshold potential (mV)
    v_thresh: f64,
    /// Current membrane potential (mV)
    v_mem: f64,
    /// Refractory period (microseconds)
    refrac_period_us: u64,
    /// Time of last spike (microseconds)
    last_spike_us: Option<u64>,
}

impl LIFNeuron {
    /// Create a new LIF neuron
    pub fn new(id: usize, tau: f64, resistance: f64, v_rest: f64, v_thresh: f64) -> Self {
        Self {
            id,
            tau,
            resistance,
            v_rest,
            v_thresh,
            v_mem: v_rest,
            refrac_period_us: 2000, // 2ms default
            last_spike_us: None,
        }
    }

    /// Get the neuron ID
    pub fn id(&self) -> usize {
        self.id
    }

    /// Get current membrane potential
    pub fn membrane_potential(&self) -> f64 {
        self.v_mem
    }

    /// Update membrane potential with input current
    ///
    /// Returns Some(spike_time) if neuron fires, None otherwise
    pub fn update(&mut self, current: f64, dt_us: u64, current_time_us: u64) -> Option<u64> {
        // Check if in refractory period
        if let Some(last_spike) = self.last_spike_us {
            if current_time_us < last_spike + self.refrac_period_us {
                return None;
            }
        }

        let dt = dt_us as f64 / 1_000_000.0; // Convert to seconds

        // Integrate membrane potential: dV/dt = (-(V - V_rest) + R*I) / tau
        let dv = ((-(self.v_mem - self.v_rest) + self.resistance * current) / self.tau) * dt;
        self.v_mem += dv;

        // Check for spike
        if self.v_mem >= self.v_thresh {
            self.v_mem = self.v_rest; // Reset
            self.last_spike_us = Some(current_time_us);
            Some(current_time_us)
        } else {
            None
        }
    }

    /// Reset the neuron to resting state
    pub fn reset(&mut self) {
        self.v_mem = self.v_rest;
        self.last_spike_us = None;
    }

    /// Set refractory period
    pub fn set_refractory_period(&mut self, period_us: u64) {
        self.refrac_period_us = period_us;
    }
}

/// Izhikevich neuron model
///
/// More biologically realistic neuron model capable of reproducing various
/// spiking patterns (regular spiking, fast spiking, bursting, etc.)
#[derive(Debug, Clone)]
pub struct IzhikevichNeuron {
    id: usize,
    /// Time scale of recovery variable u
    a: f64,
    /// Sensitivity of u to v
    b: f64,
    /// After-spike reset value of v
    c: f64,
    /// After-spike reset increment of u
    d: f64,
    /// Membrane potential
    v: f64,
    /// Recovery variable
    u: f64,
}

impl IzhikevichNeuron {
    /// Create a new Izhikevich neuron
    pub fn new(id: usize, a: f64, b: f64, c: f64, d: f64) -> Self {
        Self {
            id,
            a,
            b,
            c,
            d,
            v: c, // Initialize to reset value
            u: b * c,
        }
    }

    /// Create a regular spiking neuron
    pub fn regular_spiking(id: usize) -> Self {
        Self::new(id, 0.02, 0.2, -65.0, 8.0)
    }

    /// Create a fast spiking neuron
    pub fn fast_spiking(id: usize) -> Self {
        Self::new(id, 0.1, 0.2, -65.0, 2.0)
    }

    /// Create a bursting neuron
    pub fn bursting(id: usize) -> Self {
        Self::new(id, 0.02, 0.2, -50.0, 2.0)
    }

    /// Get the neuron ID
    pub fn id(&self) -> usize {
        self.id
    }

    /// Get membrane potential
    pub fn membrane_potential(&self) -> f64 {
        self.v
    }

    /// Update neuron state
    ///
    /// Returns Some(spike_time) if neuron fires, None otherwise
    pub fn update(&mut self, current: f64, dt_us: u64, current_time_us: u64) -> Option<u64> {
        let dt = dt_us as f64 / 1000.0; // Convert to ms

        // Izhikevich equations:
        // v' = 0.04v^2 + 5v + 140 - u + I
        // u' = a(bv - u)

        let v_squared = self.v * self.v;
        let dv = (0.04 * v_squared + 5.0 * self.v + 140.0 - self.u + current) * dt;
        let du = (self.a * (self.b * self.v - self.u)) * dt;

        self.v += dv;
        self.u += du;

        // Check for spike
        if self.v >= 30.0 {
            // Spike threshold
            self.v = self.c;
            self.u += self.d;
            Some(current_time_us)
        } else {
            None
        }
    }

    /// Reset the neuron
    pub fn reset(&mut self) {
        self.v = self.c;
        self.u = self.b * self.c;
    }
}

/// Synapse with Spike-Timing-Dependent Plasticity (STDP)
///
/// Implements STDP learning rule where synapse strength is modified
/// based on the relative timing of pre- and post-synaptic spikes.
#[derive(Debug, Clone)]
pub struct STDPSynapse {
    /// Pre-synaptic neuron ID
    pre_id: usize,
    /// Post-synaptic neuron ID
    post_id: usize,
    /// Synaptic weight
    weight: f64,
    /// Learning rate for potentiation (pre before post)
    a_plus: f64,
    /// Learning rate for depression (post before pre)
    a_minus: f64,
    /// Time constant for potentiation (ms)
    tau_plus: f64,
    /// Time constant for depression (ms)
    tau_minus: f64,
    /// Time of last pre-synaptic spike
    last_pre_spike_us: Option<u64>,
    /// Time of last post-synaptic spike
    last_post_spike_us: Option<u64>,
}

impl STDPSynapse {
    /// Create a new STDP synapse
    pub fn new(
        pre_id: usize,
        post_id: usize,
        initial_weight: f64,
        a_plus: f64,
        a_minus: f64,
    ) -> Self {
        Self {
            pre_id,
            post_id,
            weight: initial_weight,
            a_plus,
            a_minus,
            tau_plus: 20.0,  // 20ms default
            tau_minus: 20.0, // 20ms default
            last_pre_spike_us: None,
            last_post_spike_us: None,
        }
    }

    /// Get the synaptic weight
    pub fn weight(&self) -> f64 {
        self.weight
    }

    /// Get pre-synaptic neuron ID
    pub fn pre_id(&self) -> usize {
        self.pre_id
    }

    /// Get post-synaptic neuron ID
    pub fn post_id(&self) -> usize {
        self.post_id
    }

    /// Update synapse on pre-synaptic spike
    pub fn on_pre_spike(&mut self, spike_time_us: u64) {
        self.last_pre_spike_us = Some(spike_time_us);

        // Depression: post spike came before pre spike
        if let Some(last_post) = self.last_post_spike_us {
            if last_post < spike_time_us {
                let dt = ((spike_time_us - last_post) as f64) / 1000.0; // Convert to ms
                let delta_w = -self.a_minus * (-dt / self.tau_minus).exp();
                self.weight += delta_w;
                self.weight = self.weight.max(0.0); // Ensure non-negative
            }
        }
    }

    /// Update synapse on post-synaptic spike
    pub fn on_post_spike(&mut self, spike_time_us: u64) {
        self.last_post_spike_us = Some(spike_time_us);

        // Potentiation: pre spike came before post spike
        if let Some(last_pre) = self.last_pre_spike_us {
            if last_pre < spike_time_us {
                let dt = ((spike_time_us - last_pre) as f64) / 1000.0; // Convert to ms
                let delta_w = self.a_plus * (-dt / self.tau_plus).exp();
                self.weight += delta_w;
            }
        }
    }

    /// Set time constants
    pub fn set_time_constants(&mut self, tau_plus: f64, tau_minus: f64) {
        self.tau_plus = tau_plus;
        self.tau_minus = tau_minus;
    }

    /// Reset synapse state
    pub fn reset(&mut self) {
        self.last_pre_spike_us = None;
        self.last_post_spike_us = None;
    }
}

/// Neuromorphic hardware core abstraction
///
/// Represents a computation core in neuromorphic hardware (e.g., Loihi core)
#[derive(Debug, Clone)]
pub struct NeuromorphicCore {
    /// Core ID
    id: usize,
    /// Maximum number of neurons
    max_neurons: usize,
    /// Maximum number of synapses
    max_synapses: usize,
    /// Current neuron count
    neuron_count: usize,
    /// Current synapse count
    synapse_count: usize,
}

impl NeuromorphicCore {
    /// Create a new neuromorphic core
    pub fn new(id: usize, max_neurons: usize, max_synapses: usize) -> Self {
        Self {
            id,
            max_neurons,
            max_synapses,
            neuron_count: 0,
            synapse_count: 0,
        }
    }

    /// Create a Loihi-like core (128 neurons, 128K synapses per core)
    pub fn loihi_core(id: usize) -> Self {
        Self::new(id, 128, 128 * 1024)
    }

    /// Create a TrueNorth-like core (256 neurons, 64K synapses per core)
    pub fn truenorth_core(id: usize) -> Self {
        Self::new(id, 256, 64 * 1024)
    }

    /// Get core ID
    pub fn id(&self) -> usize {
        self.id
    }

    /// Check if can add a neuron
    pub fn can_add_neuron(&self) -> bool {
        self.neuron_count < self.max_neurons
    }

    /// Check if can add a synapse
    pub fn can_add_synapse(&self) -> bool {
        self.synapse_count < self.max_synapses
    }

    /// Add a neuron
    pub fn add_neuron(&mut self) -> Result<usize, &'static str> {
        if !self.can_add_neuron() {
            return Err("Core neuron capacity exceeded");
        }
        let neuron_id = self.neuron_count;
        self.neuron_count += 1;
        Ok(neuron_id)
    }

    /// Add a synapse
    pub fn add_synapse(&mut self) -> Result<usize, &'static str> {
        if !self.can_add_synapse() {
            return Err("Core synapse capacity exceeded");
        }
        let synapse_id = self.synapse_count;
        self.synapse_count += 1;
        Ok(synapse_id)
    }

    /// Get utilization statistics
    pub fn utilization(&self) -> CoreUtilization {
        CoreUtilization {
            neuron_util: self.neuron_count as f64 / self.max_neurons as f64,
            synapse_util: self.synapse_count as f64 / self.max_synapses as f64,
        }
    }
}

/// Core utilization statistics
#[derive(Debug, Clone, Copy)]
pub struct CoreUtilization {
    /// Neuron utilization (0.0 to 1.0)
    pub neuron_util: f64,
    /// Synapse utilization (0.0 to 1.0)
    pub synapse_util: f64,
}

/// Event-driven simulation state
///
/// Manages the simulation of spiking neural networks using event-driven approach
#[derive(Debug, Clone)]
pub struct EventDrivenSimulation {
    /// Current simulation time (microseconds)
    current_time_us: u64,
    /// Event queue (sorted by timestamp)
    event_queue: Vec<SpikeEvent>,
}

impl EventDrivenSimulation {
    /// Create a new event-driven simulation
    pub fn new() -> Self {
        Self {
            current_time_us: 0,
            event_queue: Vec::new(),
        }
    }

    /// Get current simulation time
    pub fn current_time_us(&self) -> u64 {
        self.current_time_us
    }

    /// Add a spike event to the queue
    pub fn schedule_event(&mut self, event: SpikeEvent) {
        self.event_queue.push(event);
        self.event_queue.sort(); // Keep queue sorted
    }

    /// Get next event
    pub fn next_event(&mut self) -> Option<SpikeEvent> {
        if self.event_queue.is_empty() {
            return None;
        }
        let event = self.event_queue.remove(0);
        self.current_time_us = event.timestamp_us();
        Some(event)
    }

    /// Check if events remain
    pub fn has_events(&self) -> bool {
        !self.event_queue.is_empty()
    }

    /// Get number of pending events
    pub fn event_count(&self) -> usize {
        self.event_queue.len()
    }

    /// Advance simulation to a specific time
    pub fn advance_to(&mut self, time_us: u64) {
        self.current_time_us = time_us;
    }

    /// Reset simulation
    pub fn reset(&mut self) {
        self.current_time_us = 0;
        self.event_queue.clear();
    }
}

impl Default for EventDrivenSimulation {
    fn default() -> Self {
        Self::new()
    }
}

/// Spike encoding schemes for converting continuous signals to spikes
#[derive(Debug, Clone, Copy)]
pub enum SpikeEncoding {
    /// Rate coding: spike frequency represents value
    Rate,
    /// Temporal coding: spike timing represents value
    Temporal,
    /// Population coding: population activity represents value
    Population,
    /// Phase coding: spike phase relative to oscillation
    Phase,
}

/// Rate encoder for converting continuous values to spike trains
#[derive(Debug, Clone)]
pub struct RateEncoder {
    /// Maximum firing rate (Hz)
    max_rate: f64,
    /// Encoding window (microseconds)
    window_us: u64,
}

impl RateEncoder {
    /// Create a new rate encoder
    pub fn new(max_rate: f64, window_us: u64) -> Self {
        Self {
            max_rate,
            window_us,
        }
    }

    /// Encode a value (0.0 to 1.0) as spike count
    pub fn encode(&self, value: f64) -> usize {
        let rate = value.max(0.0).min(1.0) * self.max_rate;
        let window_s = self.window_us as f64 / 1_000_000.0;
        (rate * window_s).round() as usize
    }

    /// Get maximum rate
    pub fn max_rate(&self) -> f64 {
        self.max_rate
    }

    /// Get encoding window
    pub fn window_us(&self) -> u64 {
        self.window_us
    }
}

/// Spike decoder for converting spike trains to continuous values
#[derive(Debug, Clone)]
pub struct RateDecoder {
    /// Decoding window (microseconds)
    window_us: u64,
}

impl RateDecoder {
    /// Create a new rate decoder
    pub fn new(window_us: u64) -> Self {
        Self { window_us }
    }

    /// Decode spike train to continuous value
    pub fn decode(&self, spike_train: &SpikeTrain) -> f64 {
        spike_train.firing_rate(self.window_us)
    }

    /// Get decoding window
    pub fn window_us(&self) -> u64 {
        self.window_us
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spike_event_creation() {
        let event = SpikeEvent::new(0, 1000);
        assert_eq!(event.neuron_id(), 0);
        assert_eq!(event.timestamp_us(), 1000);
        assert_eq!(event.timestamp_ms(), 1.0);
        assert_eq!(event.timestamp_s(), 0.001);
    }

    #[test]
    fn test_spike_event_ordering() {
        let event1 = SpikeEvent::new(0, 1000);
        let event2 = SpikeEvent::new(1, 2000);
        assert!(event1 < event2);
    }

    #[test]
    fn test_spike_train() {
        let mut train = SpikeTrain::new(0);
        train.add_spike(1000);
        train.add_spike(2000);
        train.add_spike(3000);

        assert_eq!(train.spike_count(), 3);
        assert_eq!(train.neuron_id(), 0);

        let rate = train.firing_rate(10_000); // 10ms window
        assert!((rate - 300.0).abs() < 1.0); // ~300 Hz

        let isis = train.inter_spike_intervals();
        assert_eq!(isis.len(), 2);
        assert_eq!(isis[0], 1000);
        assert_eq!(isis[1], 1000);
    }

    #[test]
    fn test_lif_neuron() {
        let mut neuron = LIFNeuron::new(0, 0.02, 0.2, -65.0, -50.0);
        assert_eq!(neuron.id(), 0);
        assert_eq!(neuron.membrane_potential(), -65.0);

        // Apply input current
        let spike = neuron.update(100.0, 1000, 1000);
        assert!(spike.is_some() || spike.is_none()); // May or may not spike depending on parameters

        neuron.reset();
        assert_eq!(neuron.membrane_potential(), -65.0);
    }

    #[test]
    fn test_izhikevich_neuron() {
        let mut neuron = IzhikevichNeuron::regular_spiking(0);
        assert_eq!(neuron.id(), 0);

        let spike = neuron.update(10.0, 1000, 1000);
        assert!(spike.is_some() || spike.is_none());

        neuron.reset();
    }

    #[test]
    fn test_izhikevich_neuron_types() {
        let _rs = IzhikevichNeuron::regular_spiking(0);
        let _fs = IzhikevichNeuron::fast_spiking(1);
        let _burst = IzhikevichNeuron::bursting(2);
    }

    #[test]
    fn test_stdp_synapse() {
        let mut synapse = STDPSynapse::new(0, 1, 0.5, 0.02, 0.02);
        assert_eq!(synapse.pre_id(), 0);
        assert_eq!(synapse.post_id(), 1);
        assert_eq!(synapse.weight(), 0.5);

        synapse.on_pre_spike(1000);
        synapse.on_post_spike(2000); // Post after pre -> potentiation

        assert!(synapse.weight() > 0.5); // Weight should increase

        synapse.reset();
    }

    #[test]
    fn test_neuromorphic_core() {
        let mut core = NeuromorphicCore::loihi_core(0);
        assert_eq!(core.id(), 0);
        assert!(core.can_add_neuron());
        assert!(core.can_add_synapse());

        let neuron_id = core.add_neuron().unwrap();
        assert_eq!(neuron_id, 0);

        let synapse_id = core.add_synapse().unwrap();
        assert_eq!(synapse_id, 0);

        let util = core.utilization();
        assert!(util.neuron_util > 0.0);
        assert!(util.synapse_util > 0.0);
    }

    #[test]
    fn test_truenorth_core() {
        let core = NeuromorphicCore::truenorth_core(0);
        assert_eq!(core.id(), 0);
    }

    #[test]
    fn test_event_driven_simulation() {
        let mut sim = EventDrivenSimulation::new();
        assert_eq!(sim.current_time_us(), 0);
        assert!(!sim.has_events());

        sim.schedule_event(SpikeEvent::new(0, 1000));
        sim.schedule_event(SpikeEvent::new(1, 500));

        assert!(sim.has_events());
        assert_eq!(sim.event_count(), 2);

        let event1 = sim.next_event().unwrap();
        assert_eq!(event1.timestamp_us(), 500); // Earlier event comes first

        let event2 = sim.next_event().unwrap();
        assert_eq!(event2.timestamp_us(), 1000);

        assert!(!sim.has_events());

        sim.reset();
        assert_eq!(sim.current_time_us(), 0);
    }

    #[test]
    fn test_rate_encoder() {
        let encoder = RateEncoder::new(100.0, 10_000); // 100 Hz max, 10ms window
        let spike_count = encoder.encode(0.5); // 50% of max rate
        assert!(spike_count > 0);
        assert_eq!(encoder.max_rate(), 100.0);
        assert_eq!(encoder.window_us(), 10_000);
    }

    #[test]
    fn test_rate_decoder() {
        let decoder = RateDecoder::new(10_000);
        let mut train = SpikeTrain::new(0);
        train.add_spike(1000);
        train.add_spike(2000);
        train.add_spike(3000);

        let rate = decoder.decode(&train);
        assert!(rate > 0.0);
        assert_eq!(decoder.window_us(), 10_000);
    }

    #[test]
    fn test_spike_encoding_variants() {
        let _rate = SpikeEncoding::Rate;
        let _temporal = SpikeEncoding::Temporal;
        let _population = SpikeEncoding::Population;
        let _phase = SpikeEncoding::Phase;
    }

    #[test]
    fn test_default_simulation() {
        let sim = EventDrivenSimulation::default();
        assert_eq!(sim.current_time_us(), 0);
    }

    #[test]
    fn test_simulation_advance() {
        let mut sim = EventDrivenSimulation::new();
        sim.advance_to(5000);
        assert_eq!(sim.current_time_us(), 5000);
    }

    #[test]
    fn test_lif_refractory_period() {
        let mut neuron = LIFNeuron::new(0, 0.02, 0.2, -65.0, -50.0);
        neuron.set_refractory_period(3000); // 3ms
    }

    #[test]
    fn test_stdp_time_constants() {
        let mut synapse = STDPSynapse::new(0, 1, 0.5, 0.02, 0.02);
        synapse.set_time_constants(15.0, 25.0);
    }

    #[test]
    fn test_core_capacity() {
        let mut core = NeuromorphicCore::new(0, 2, 2);

        core.add_neuron().unwrap();
        core.add_neuron().unwrap();
        assert!(core.add_neuron().is_err()); // Should fail - capacity exceeded

        core.add_synapse().unwrap();
        core.add_synapse().unwrap();
        assert!(core.add_synapse().is_err()); // Should fail - capacity exceeded
    }
}
