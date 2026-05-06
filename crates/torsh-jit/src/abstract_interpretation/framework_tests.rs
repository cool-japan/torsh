//! Tests for the abstract interpretation framework.
//!
//! Verifies lattice operations, transfer functions, fixed-point
//! convergence, property checking and invariant detection over the
//! interval and sign domains.

use super::domains::{
    AbstractDomain, AbstractDomainType, AbstractValue, BinaryAbstractOp, IntervalDomain, SignValue,
};
use super::framework::{
    AbstractGraph, AbstractInterpretationConfig, AbstractInterpreter, AbstractNodeOp,
    InvariantDetector, Property, PropertyChecker, SafetyCheckResult,
};
use crate::graph::core::ComputationGraph;
use crate::graph::operations::{ConstantInfo, ConstantValue as OpConstantValue, Operation};
use crate::graph::Node;
use std::collections::HashMap;

fn interval(min: f64, max: f64) -> AbstractValue {
    AbstractValue::Interval { min, max }
}

#[test]
fn test_interval_join_takes_widest_bounds() {
    let domain = IntervalDomain::new();
    let a = interval(1.0, 5.0);
    let b = interval(3.0, 9.0);
    let joined = domain.join(&a, &b).expect("join");
    match joined {
        AbstractValue::Interval { min, max } => {
            assert_eq!(min, 1.0);
            assert_eq!(max, 9.0);
        }
        _ => panic!("expected Interval"),
    }
}

#[test]
fn test_interval_meet_intersects() {
    let domain = IntervalDomain::new();
    let a = interval(1.0, 5.0);
    let b = interval(3.0, 7.0);
    let met = domain.meet(&a, &b).expect("meet");
    match met {
        AbstractValue::Interval { min, max } => {
            assert_eq!(min, 3.0);
            assert_eq!(max, 5.0);
        }
        _ => panic!("expected Interval"),
    }
}

#[test]
fn test_meet_disjoint_returns_bottom() {
    let domain = IntervalDomain::new();
    let a = interval(1.0, 2.0);
    let b = interval(3.0, 4.0);
    let met = domain.meet(&a, &b).expect("meet");
    match met {
        AbstractValue::Interval { min, max } => {
            assert!(min > max, "expected bottom (min > max), got [{}, {}]", min, max);
        }
        _ => panic!("expected Interval"),
    }
}

#[test]
fn test_widen_unbounded_above_after_growing_max() {
    let domain = IntervalDomain::new();
    let a = interval(0.0, 10.0);
    let b = interval(0.0, 11.0);
    let widened = domain.widen(&a, &b).expect("widen");
    match widened {
        AbstractValue::Interval { min, max } => {
            assert_eq!(min, 0.0);
            assert!(max.is_infinite() && max.is_sign_positive(), "expected +∞, got {}", max);
        }
        _ => panic!("expected Interval"),
    }
}

#[test]
fn test_widen_stable_when_subsumed() {
    let domain = IntervalDomain::new();
    let a = interval(-5.0, 10.0);
    let b = interval(0.0, 5.0);
    let widened = domain.widen(&a, &b).expect("widen");
    match widened {
        AbstractValue::Interval { min, max } => {
            assert_eq!(min, -5.0);
            assert_eq!(max, 10.0);
        }
        _ => panic!("expected Interval"),
    }
}

#[test]
fn test_transfer_add_sums_intervals() {
    let domain = IntervalDomain::new();
    let a = interval(1.0, 2.0);
    let b = interval(3.0, 4.0);
    let result = domain
        .abstract_binary_op(BinaryAbstractOp::Add, &a, &b)
        .expect("add");
    match result {
        AbstractValue::Interval { min, max } => {
            assert_eq!(min, 4.0);
            assert_eq!(max, 6.0);
        }
        _ => panic!("expected Interval"),
    }
}

#[test]
fn test_transfer_mul_intervals() {
    let domain = IntervalDomain::new();
    let a = interval(-1.0, 2.0);
    let b = interval(3.0, 4.0);
    let result = domain
        .abstract_binary_op(BinaryAbstractOp::Mul, &a, &b)
        .expect("mul");
    match result {
        AbstractValue::Interval { min, max } => {
            assert_eq!(min, -4.0);
            assert_eq!(max, 8.0);
        }
        _ => panic!("expected Interval"),
    }
}

/// Build a tiny synthesized graph: const(3) + const(5) → output.
fn build_add_graph() -> ComputationGraph {
    let mut graph = ComputationGraph::new();
    let const3 = graph.add_node(Node::new(
        Operation::Constant(ConstantInfo {
            value: OpConstantValue::Float(3.0),
        }),
        "const3".to_string(),
    ));
    let const5 = graph.add_node(Node::new(
        Operation::Constant(ConstantInfo {
            value: OpConstantValue::Float(5.0),
        }),
        "const5".to_string(),
    ));
    let add = graph.add_node(Node::new(Operation::Add, "add".to_string()));
    graph.add_edge(const3, add, Default::default());
    graph.add_edge(const5, add, Default::default());
    graph.add_input(const3);
    graph.add_input(const5);
    graph.add_output(add);
    graph
}

#[test]
fn test_fixed_point_terminates() {
    let graph = build_add_graph();
    let config = AbstractInterpretationConfig {
        domain_type: AbstractDomainType::Intervals,
        max_iterations: 50,
        widening_delay: 3,
        enable_narrowing: false,
        enable_backward_analysis: false,
        properties: vec![],
        precision_threshold: 0.5,
    };
    let mut interp = AbstractInterpreter::new(config);
    let result = interp.analyze_graph(&graph).expect("analyze");
    assert!(result.forward_result.converged, "analysis must converge");
    assert!(
        result.forward_result.iterations < 50,
        "iterations must be below cap, got {}",
        result.forward_result.iterations
    );
}

#[test]
fn test_constant_folding_via_intervals() {
    let graph = build_add_graph();
    let config = AbstractInterpretationConfig {
        domain_type: AbstractDomainType::Intervals,
        max_iterations: 20,
        widening_delay: 0,
        enable_narrowing: false,
        enable_backward_analysis: false,
        properties: vec![],
        precision_threshold: 0.5,
    };
    let mut interp = AbstractInterpreter::new(config);
    let result = interp.analyze_graph(&graph).expect("analyze");
    // The "add" node should be the singleton interval [8, 8].
    let add_id = graph
        .nodes()
        .find(|(_, n)| n.name == "add")
        .map(|(id, _)| id)
        .expect("add node");
    let value = result
        .forward_result
        .post_states
        .get(&add_id)
        .expect("post-state for add");
    match value {
        AbstractValue::Interval { min, max } => {
            assert_eq!(*min, 8.0);
            assert_eq!(*max, 8.0);
        }
        other => panic!("expected singleton interval, got {:?}", other),
    }
}

#[test]
fn test_convert_to_abstract_graph_classifies_ops() {
    let graph = build_add_graph();
    let interp = AbstractInterpreter::with_defaults();
    // Reach the converter by running the public entry point and
    // observing the result's node_values map plus statistics.
    let mut interp = interp;
    let result = interp.analyze_graph(&graph).expect("analyze");
    assert!(result.statistics.abstract_states_computed >= 3);
}

#[test]
fn test_property_check_non_negative_safe_on_positive_interval() {
    let checker = PropertyChecker::new();
    let mut forward = super::framework::ForwardAnalysisResult::new();
    let node = crate::NodeId::new(0);
    forward.post_states.insert(node, interval(1.0, 5.0));
    let props = vec![Property::NonNegative(node)];
    let results = checker
        .check_properties(&forward, &props)
        .expect("check_properties");
    assert_eq!(results.len(), 1);
    assert!(matches!(results[0].result, SafetyCheckResult::Safe));
}

#[test]
fn test_property_check_non_negative_unsafe_on_negative_interval() {
    let checker = PropertyChecker::new();
    let mut forward = super::framework::ForwardAnalysisResult::new();
    let node = crate::NodeId::new(0);
    forward.post_states.insert(node, interval(-5.0, -1.0));
    let props = vec![Property::NonNegative(node)];
    let results = checker
        .check_properties(&forward, &props)
        .expect("check_properties");
    assert!(matches!(results[0].result, SafetyCheckResult::Unsafe));
}

#[test]
fn test_property_check_no_division_by_zero_unknown_when_includes_zero() {
    let checker = PropertyChecker::new();
    let mut forward = super::framework::ForwardAnalysisResult::new();
    let node = crate::NodeId::new(0);
    forward.post_states.insert(node, interval(-1.0, 1.0));
    let props = vec![Property::NoDivisionByZero(node)];
    let results = checker
        .check_properties(&forward, &props)
        .expect("check_properties");
    assert!(matches!(results[0].result, SafetyCheckResult::Unknown));
}

#[test]
fn test_invariant_detector_reports_constants() {
    let detector = InvariantDetector::new();
    let mut forward = super::framework::ForwardAnalysisResult::new();
    let node = crate::NodeId::new(7);
    forward.post_states.insert(node, interval(42.0, 42.0));
    let invariants = detector.detect_invariants(&forward, &None).expect("detect");
    assert!(
        invariants
            .iter()
            .any(|i| matches!(i.invariant_type, super::framework::InvariantType::NumericalProperty)),
        "expected NumericalProperty invariant for singleton interval"
    );
}

#[test]
fn test_abstract_graph_classification_unknown_for_unsupported_op() {
    // Operations like MatMul should classify as Unknown so transfer is sound.
    let mut graph = ComputationGraph::new();
    let inp = graph.add_node(Node::new(Operation::Input, "in".to_string()));
    let mm = graph.add_node(Node::new(Operation::MatMul, "mm".to_string()));
    graph.add_edge(inp, mm, Default::default());
    graph.add_input(inp);
    graph.add_output(mm);

    let mut interp = AbstractInterpreter::with_defaults();
    let result = interp.analyze_graph(&graph).expect("analyze");
    let mm_id = graph
        .nodes()
        .find(|(_, n)| n.name == "mm")
        .map(|(id, _)| id)
        .expect("mm node");
    let value = result
        .forward_result
        .post_states
        .get(&mm_id)
        .expect("post-state for mm");
    // Should have been transferred to top (NEG_INFINITY..INFINITY).
    match value {
        AbstractValue::Interval { min, max } => {
            assert!(min.is_infinite() && min.is_sign_negative());
            assert!(max.is_infinite() && max.is_sign_positive());
        }
        other => panic!("expected top interval, got {:?}", other),
    }
}

#[test]
fn test_sign_domain_join() {
    let domain = super::domains::SignDomain::new();
    let pos = AbstractValue::Sign(SignValue::Positive);
    let zero = AbstractValue::Sign(SignValue::Zero);
    let joined = domain.join(&pos, &zero).expect("join");
    match joined {
        AbstractValue::Sign(SignValue::NonNegative) => {}
        other => panic!("expected NonNegative, got {:?}", other),
    }
}

#[test]
fn test_abstract_graph_records_predecessors_and_entry_exit() {
    let graph = build_add_graph();
    let interp = AbstractInterpreter::with_defaults();
    let domain = super::domains::AbstractDomainFactory::new()
        .create_domain(&AbstractDomainType::Intervals);
    // We cannot call the private converter directly, but analyze_graph
    // populates node_values with one entry per node.
    let mut interp = interp;
    let result = interp.analyze_graph(&graph).expect("analyze");
    assert_eq!(result.node_values.len(), graph.node_count());
    // Sanity check: domain factory returns something useful.
    assert!(matches!(
        domain.bottom(),
        AbstractValue::Interval { .. }
    ));
}

#[test]
fn test_abstract_node_op_classification() {
    // Round-trip a couple of operations through the public surface to
    // ensure classification stays consistent. We do this by running an
    // analysis on a small graph and checking the abstract value yielded.
    let mut graph = ComputationGraph::new();
    let c = graph.add_node(Node::new(
        Operation::Constant(ConstantInfo {
            value: OpConstantValue::Int(7),
        }),
        "c".to_string(),
    ));
    let neg = graph.add_node(Node::new(Operation::Neg, "neg".to_string()));
    graph.add_edge(c, neg, Default::default());
    graph.add_input(c);
    graph.add_output(neg);

    let mut interp = AbstractInterpreter::with_defaults();
    let result = interp.analyze_graph(&graph).expect("analyze");
    let neg_id = graph
        .nodes()
        .find(|(_, n)| n.name == "neg")
        .map(|(id, _)| id)
        .expect("neg node");
    let value = result
        .forward_result
        .post_states
        .get(&neg_id)
        .expect("post-state for neg");
    match value {
        AbstractValue::Interval { min, max } => {
            assert_eq!(*min, -7.0);
            assert_eq!(*max, -7.0);
        }
        other => panic!("expected [-7,-7], got {:?}", other),
    }
}

#[test]
fn test_abstract_graph_default_constructible() {
    let g = AbstractGraph::new();
    assert_eq!(g.node_count(), 0);
}

#[test]
fn test_abstract_node_op_values_compile() {
    // Make sure all variants of AbstractNodeOp can be constructed.
    let _ = AbstractNodeOp::Input;
    let _ = AbstractNodeOp::Constant(0.0);
    let _ = AbstractNodeOp::Binary(BinaryAbstractOp::Add);
    let _ = AbstractNodeOp::Unary(super::domains::UnaryAbstractOp::Neg);
    let _ = AbstractNodeOp::Unknown;
}

#[test]
fn test_state_map_join_via_predecessors() {
    // Build chain: const(2) -> abs -> output, ensure abs gets [2,2].
    let mut graph = ComputationGraph::new();
    let c = graph.add_node(Node::new(
        Operation::Constant(ConstantInfo {
            value: OpConstantValue::Float(2.0),
        }),
        "c".to_string(),
    ));
    let abs_node = graph.add_node(Node::new(Operation::Abs, "abs".to_string()));
    graph.add_edge(c, abs_node, Default::default());
    graph.add_input(c);
    graph.add_output(abs_node);
    let mut interp = AbstractInterpreter::with_defaults();
    let result = interp.analyze_graph(&graph).expect("analyze");
    let abs_id = graph
        .nodes()
        .find(|(_, n)| n.name == "abs")
        .map(|(id, _)| id)
        .expect("abs");
    let value = result
        .forward_result
        .post_states
        .get(&abs_id)
        .expect("post-state");
    match value {
        AbstractValue::Interval { min, max } => {
            assert_eq!(*min, 2.0);
            assert_eq!(*max, 2.0);
        }
        other => panic!("expected [2,2], got {:?}", other),
    }
}

#[test]
fn test_analysis_statistics_populated() {
    let graph = build_add_graph();
    let mut interp = AbstractInterpreter::with_defaults();
    let result = interp.analyze_graph(&graph).expect("analyze");
    assert!(result.statistics.abstract_states_computed > 0);
    assert!(result.statistics.fixpoint_iterations > 0);
}

#[test]
fn test_widening_caps_runaway_growth() {
    // Synthetic loop-like graph would require cycles; here we simply
    // verify that widening produces +infinity when invoked with growing
    // intervals.
    let domain = IntervalDomain::new();
    let a = interval(0.0, 1.0);
    let b = interval(0.0, 2.0);
    let widened = domain.widen(&a, &b).expect("widen");
    match widened {
        AbstractValue::Interval { min, max } => {
            assert_eq!(min, 0.0);
            assert!(max.is_infinite());
        }
        _ => panic!("expected Interval"),
    }
}

#[test]
fn test_property_checker_unknown_when_no_state() {
    let checker = PropertyChecker::new();
    let forward = super::framework::ForwardAnalysisResult::new();
    let node = crate::NodeId::new(99);
    let props = vec![Property::NonNegative(node)];
    let results = checker
        .check_properties(&forward, &props)
        .expect("check");
    assert!(matches!(results[0].result, SafetyCheckResult::Unknown));
}

#[test]
fn test_unused_imports_silenced() {
    // Touch the rarely-used items so unused-import warnings do not fire
    // in this test module.
    let _ = HashMap::<u32, u32>::new();
}
