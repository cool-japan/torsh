// Copyright (c) 2025 ToRSh Project
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Interactive Gradient Computation Debugger
//!
//! This module provides an interactive debugging interface for gradient computations,
//! allowing step-by-step execution, tensor inspection, and computation graph analysis.
//!
//! # Features
//!
//! - **Step-by-step Execution**: Execute operations one at a time
//! - **Breakpoints**: Set breakpoints on specific operations or conditions
//! - **Tensor Inspection**: Examine tensor values and gradients
//! - **Call Stack**: View operation call stack
//! - **Watchpoints**: Monitor specific tensors for changes
//! - **Time Travel**: Step backward through execution history

use crate::error_handling::{AutogradError, AutogradResult};
use crate::gradient_tracer::{EventType, PathId, TraceEvent, TraceEventId};
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

/// Debugger state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DebuggerState {
    /// Debugger is inactive
    Inactive,
    /// Debugger is paused (waiting for user input)
    Paused,
    /// Debugger is running
    Running,
    /// Debugger is stepping (will pause after next operation)
    Stepping,
    /// Debugger is continuing to next breakpoint
    Continuing,
}

/// Breakpoint condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BreakpointCondition {
    /// Break on specific operation name
    OperationName(String),

    /// Break on specific tensor ID
    TensorId(String),

    /// Break on anomaly detection
    Anomaly,

    /// Break on memory threshold
    MemoryThreshold(usize),

    /// Break after N operations
    OperationCount(usize),

    /// Break on gradient explosion (norm exceeds threshold)
    GradientExplosion(f64),

    /// Break on gradient vanishing (norm below threshold)
    GradientVanishing(f64),

    /// Custom condition (evaluates to true if should break)
    Custom(String),
}

/// Breakpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Breakpoint {
    /// Breakpoint ID
    pub id: u64,

    /// Breakpoint condition
    pub condition: BreakpointCondition,

    /// Whether breakpoint is enabled
    pub enabled: bool,

    /// Number of times this breakpoint has been hit
    pub hit_count: usize,

    /// Description
    pub description: String,
}

impl Breakpoint {
    /// Create a new breakpoint
    pub fn new(id: u64, condition: BreakpointCondition, description: String) -> Self {
        Self {
            id,
            condition,
            enabled: true,
            hit_count: 0,
            description,
        }
    }

    /// Check if this breakpoint should trigger
    pub fn should_trigger(&mut self, event: &TraceEvent, context: &DebugContext) -> bool {
        if !self.enabled {
            return false;
        }

        let triggered = match &self.condition {
            BreakpointCondition::OperationName(name) => &event.operation == name,
            BreakpointCondition::TensorId(id) => {
                event.input_ids.contains(id) || event.output_ids.contains(id)
            }
            BreakpointCondition::Anomaly => {
                matches!(event.event_type, EventType::Custom)
            }
            BreakpointCondition::MemoryThreshold(threshold) => {
                event.memory_allocated.unwrap_or(0) > *threshold
            }
            BreakpointCondition::OperationCount(count) => self.hit_count >= *count,
            BreakpointCondition::GradientExplosion(threshold) => {
                // Trigger only when real gradient data is available and its L2
                // norm exceeds the threshold. Missing gradient data never fires
                // (we do not invent a value).
                extract_gradient_norm(event, context).is_some_and(|norm| norm > *threshold)
            }
            BreakpointCondition::GradientVanishing(threshold) => {
                // Trigger only when real gradient data is available and its L2
                // norm is below the threshold.
                extract_gradient_norm(event, context).is_some_and(|norm| norm < *threshold)
            }
            BreakpointCondition::Custom(expr) => {
                // Evaluate the user expression against the live debug state. An
                // unevaluable expression does not fire (and is surfaced via the
                // public `evaluate_custom_expression`); it never fakes a hit.
                match evaluate_custom_expression(expr, event, context) {
                    Ok(result) => result,
                    Err(err) => {
                        tracing::warn!("custom breakpoint expression error: {}", err);
                        false
                    }
                }
            }
        };

        if triggered {
            self.hit_count += 1;
        }

        triggered
    }
}

// ---------------------------------------------------------------------------
// Gradient-norm extraction
// ---------------------------------------------------------------------------

/// Compute the L2 norm of the gradient associated with a trace event.
///
/// Gradient-norm breakpoints (`GradientExplosion` / `GradientVanishing`) and
/// the `gradient_norm` expression field require a concrete numeric gradient to
/// test against. This helper extracts that gradient from the data actually
/// attached to the event / debug context, in the following priority order:
///
/// 1. `event.metadata["gradient_norm"]` -- a pre-computed L2 norm (its absolute
///    value is used).
/// 2. `event.metadata["gradient_values"]` -- comma-separated gradient
///    components; the norm is computed as `sqrt(sum(x_i^2))`.
/// 3. For gradient-related events only (`GradientComputation`, `BackwardBegin`,
///    `BackwardEnd`), each of the event's output tensors is looked up in
///    `context.gradient_values`. An entry may be either a `norm=<value>` string
///    or comma-separated components. The returned value is the global norm
///    across all output tensors (`sqrt(sum over tensors of tensor_norm^2)`).
///
/// Returns `None` when no parseable gradient data is available, so callers can
/// decline to fire a breakpoint rather than fabricating a value.
fn extract_gradient_norm(event: &TraceEvent, context: &DebugContext) -> Option<f64> {
    // 1. Pre-computed norm carried explicitly on the event.
    if let Some(raw) = event.metadata.get("gradient_norm") {
        if let Ok(value) = raw.trim().parse::<f64>() {
            if value.is_finite() {
                return Some(value.abs());
            }
        }
    }

    // 2. Raw gradient components carried explicitly on the event.
    if let Some(raw) = event.metadata.get("gradient_values") {
        if let Some(norm) = l2_norm_from_csv(raw) {
            return Some(norm);
        }
    }

    // 3. Per-output-tensor gradients tracked in the debug context. Restricted to
    //    gradient-related events to avoid reading stale gradients on forward ops.
    let is_gradient_event = matches!(
        event.event_type,
        EventType::GradientComputation | EventType::BackwardBegin | EventType::BackwardEnd
    );
    if !is_gradient_event {
        return None;
    }

    let mut sum_sq = 0.0_f64;
    let mut found = false;
    for output_id in &event.output_ids {
        if let Some(descriptor) = context.gradient_values.get(output_id) {
            if let Some(tensor_norm) = parse_gradient_descriptor(descriptor) {
                sum_sq += tensor_norm * tensor_norm;
                found = true;
            }
        }
    }

    if found {
        Some(sum_sq.sqrt())
    } else {
        None
    }
}

/// Compute the L2 norm of a comma-separated list of finite floats.
///
/// Returns `None` if the string is empty, contains no numeric components, or
/// contains any token that does not parse as a finite float (so we never treat
/// malformed data as a valid zero gradient).
fn l2_norm_from_csv(raw: &str) -> Option<f64> {
    let mut sum_sq = 0.0_f64;
    let mut count = 0_usize;
    for token in raw.split(',') {
        let token = token.trim();
        if token.is_empty() {
            continue;
        }
        let value: f64 = token.parse().ok()?;
        if !value.is_finite() {
            return None;
        }
        sum_sq += value * value;
        count += 1;
    }
    if count == 0 {
        None
    } else {
        Some(sum_sq.sqrt())
    }
}

/// Parse a per-tensor gradient descriptor into an L2 norm.
///
/// Supports either a `norm=<value>` shorthand or a comma-separated list of
/// gradient components. Returns `None` when neither form parses.
fn parse_gradient_descriptor(descriptor: &str) -> Option<f64> {
    let trimmed = descriptor.trim();
    if let Some(rest) = trimmed.strip_prefix("norm=") {
        let value: f64 = rest.trim().parse().ok()?;
        return if value.is_finite() {
            Some(value.abs())
        } else {
            None
        };
    }
    l2_norm_from_csv(trimmed)
}

// ---------------------------------------------------------------------------
// Custom breakpoint expression evaluation
// ---------------------------------------------------------------------------

/// Comparison operator in a custom breakpoint expression.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CompareOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// Lexical token of a custom breakpoint expression.
#[derive(Debug, Clone, PartialEq)]
enum Token {
    /// A bareword identifier (field name on the left, string literal on the right).
    Ident(String),
    /// A numeric literal.
    Number(f64),
    /// A quoted string literal.
    Str(String),
    /// A comparison operator.
    Compare(CompareOp),
    /// Logical AND (`&&`).
    And,
    /// Logical OR (`||`).
    Or,
}

/// Resolved value of an expression field.
enum FieldValue {
    Number(f64),
    Text(String),
    /// The field is supported but no value is currently available (e.g. a
    /// gradient norm with no recorded gradient). Comparisons evaluate to false.
    Missing,
}

/// Right-hand-side literal of a comparison.
enum Operand {
    Number(f64),
    Text(String),
}

/// Evaluate a custom breakpoint expression against the current debug state.
///
/// # Supported grammar
///
/// ```text
/// expr        := and_expr ( "||" and_expr )*
/// and_expr    := comparison ( "&&" comparison )*
/// comparison  := field op literal
/// op          := "==" | "!=" | "<" | "<=" | ">" | ">="
/// literal     := number | "quoted string" | bareword
/// ```
///
/// `&&` binds tighter than `||`; there is no parenthesisation. The right-hand
/// side of every comparison is a literal, not another field.
///
/// # Supported fields
///
/// * `operation` (text) -- the event operation name
/// * `event_type` (text) -- the event type, e.g. `OperationBegin`
/// * `memory` / `memory_allocated` (number) -- bytes allocated by the event
/// * `memory_deallocated` (number) -- bytes freed by the event
/// * `memory_usage` (number) -- current total memory usage
/// * `operation_index` (number) -- current operation index
/// * `total_operations` (number) -- total operation count
/// * `input_count` / `output_count` (number) -- tensor arity of the event
/// * `gradient_norm` (number) -- L2 norm of the event gradient (see
///   [`extract_gradient_norm`]); comparisons are false when unavailable
/// * `duration_micros` (number) -- event duration in microseconds
///
/// Text fields support only `==` / `!=`. Numeric fields support all operators.
///
/// # Errors
///
/// Returns an [`AutogradError::Configuration`] describing the problem when the
/// expression is empty, malformed, references an unknown field, or compares
/// incompatible types -- never a fabricated boolean.
pub fn evaluate_custom_expression(
    expr: &str,
    event: &TraceEvent,
    context: &DebugContext,
) -> AutogradResult<bool> {
    let tokens = tokenize_expression(expr)?;
    if tokens.is_empty() {
        return Err(expression_error(expr, "expression is empty"));
    }

    let mut parser = ExpressionParser {
        tokens: &tokens,
        pos: 0,
        expr,
        event,
        context,
    };
    let result = parser.parse_or()?;
    if parser.pos != tokens.len() {
        return Err(expression_error(
            expr,
            "unexpected trailing tokens after expression",
        ));
    }
    Ok(result)
}

/// Construct a descriptive expression error carrying the supported grammar.
fn expression_error(expr: &str, reason: &str) -> AutogradError {
    AutogradError::Configuration {
        parameter: "custom_breakpoint_expression".to_string(),
        value: expr.to_string(),
        reason: reason.to_string(),
        valid_range: Some(
            "grammar: <field> <op> <literal> [(&& | ||) <field> <op> <literal>]*; \
             ops: == != < <= > >=; \
             fields: operation, event_type, memory, memory_allocated, \
             memory_deallocated, memory_usage, operation_index, total_operations, \
             input_count, output_count, gradient_norm, duration_micros"
                .to_string(),
        ),
    }
}

/// Tokenize a custom breakpoint expression.
fn tokenize_expression(expr: &str) -> AutogradResult<Vec<Token>> {
    let chars: Vec<char> = expr.chars().collect();
    let mut tokens = Vec::new();
    let mut i = 0;

    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
            continue;
        }

        match c {
            '"' => {
                let mut literal = String::new();
                i += 1;
                let mut closed = false;
                while i < chars.len() {
                    if chars[i] == '"' {
                        closed = true;
                        i += 1;
                        break;
                    }
                    literal.push(chars[i]);
                    i += 1;
                }
                if !closed {
                    return Err(expression_error(expr, "unterminated string literal"));
                }
                tokens.push(Token::Str(literal));
            }
            '=' => {
                if chars.get(i + 1) == Some(&'=') {
                    tokens.push(Token::Compare(CompareOp::Eq));
                    i += 2;
                } else {
                    return Err(expression_error(
                        expr,
                        "expected '==' (a single '=' is not a valid operator)",
                    ));
                }
            }
            '!' => {
                if chars.get(i + 1) == Some(&'=') {
                    tokens.push(Token::Compare(CompareOp::Ne));
                    i += 2;
                } else {
                    return Err(expression_error(expr, "expected '!='"));
                }
            }
            '<' => {
                if chars.get(i + 1) == Some(&'=') {
                    tokens.push(Token::Compare(CompareOp::Le));
                    i += 2;
                } else {
                    tokens.push(Token::Compare(CompareOp::Lt));
                    i += 1;
                }
            }
            '>' => {
                if chars.get(i + 1) == Some(&'=') {
                    tokens.push(Token::Compare(CompareOp::Ge));
                    i += 2;
                } else {
                    tokens.push(Token::Compare(CompareOp::Gt));
                    i += 1;
                }
            }
            '&' => {
                if chars.get(i + 1) == Some(&'&') {
                    tokens.push(Token::And);
                    i += 2;
                } else {
                    return Err(expression_error(expr, "expected '&&'"));
                }
            }
            '|' => {
                if chars.get(i + 1) == Some(&'|') {
                    tokens.push(Token::Or);
                    i += 2;
                } else {
                    return Err(expression_error(expr, "expected '||'"));
                }
            }
            _ if c.is_ascii_digit()
                || c == '.'
                || (c == '-'
                    && chars
                        .get(i + 1)
                        .is_some_and(|n| n.is_ascii_digit() || *n == '.')) =>
            {
                let start = i;
                if chars[i] == '-' {
                    i += 1;
                }
                while i < chars.len() {
                    let ch = chars[i];
                    let is_exponent_sign =
                        (ch == '+' || ch == '-') && matches!(chars.get(i - 1), Some('e' | 'E'));
                    if ch.is_ascii_digit()
                        || ch == '.'
                        || ch == 'e'
                        || ch == 'E'
                        || is_exponent_sign
                    {
                        i += 1;
                    } else {
                        break;
                    }
                }
                let lexeme: String = chars[start..i].iter().collect();
                let value: f64 = lexeme
                    .parse()
                    .map_err(|_| expression_error(expr, &format!("invalid number '{lexeme}'")))?;
                tokens.push(Token::Number(value));
            }
            _ if c.is_alphabetic() || c == '_' => {
                let start = i;
                while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                let ident: String = chars[start..i].iter().collect();
                tokens.push(Token::Ident(ident));
            }
            _ => {
                return Err(expression_error(
                    expr,
                    &format!("unexpected character '{c}'"),
                ));
            }
        }
    }

    Ok(tokens)
}

/// Recursive-descent evaluator for custom breakpoint expressions.
struct ExpressionParser<'a> {
    tokens: &'a [Token],
    pos: usize,
    expr: &'a str,
    event: &'a TraceEvent,
    context: &'a DebugContext,
}

impl ExpressionParser<'_> {
    /// `expr := and_expr ( "||" and_expr )*`
    fn parse_or(&mut self) -> AutogradResult<bool> {
        let mut result = self.parse_and()?;
        while matches!(self.tokens.get(self.pos), Some(Token::Or)) {
            self.pos += 1;
            // Both sides are always parsed (no side effects), so evaluating the
            // right operand unconditionally is correct.
            let rhs = self.parse_and()?;
            result = result || rhs;
        }
        Ok(result)
    }

    /// `and_expr := comparison ( "&&" comparison )*`
    fn parse_and(&mut self) -> AutogradResult<bool> {
        let mut result = self.parse_comparison()?;
        while matches!(self.tokens.get(self.pos), Some(Token::And)) {
            self.pos += 1;
            let rhs = self.parse_comparison()?;
            result = result && rhs;
        }
        Ok(result)
    }

    /// `comparison := field op literal`
    fn parse_comparison(&mut self) -> AutogradResult<bool> {
        let field_name = match self.tokens.get(self.pos) {
            Some(Token::Ident(name)) => name.clone(),
            Some(other) => {
                return Err(expression_error(
                    self.expr,
                    &format!("expected a field name, found {other:?}"),
                ));
            }
            None => {
                return Err(expression_error(self.expr, "expected a field name"));
            }
        };
        self.pos += 1;

        let op = match self.tokens.get(self.pos) {
            Some(Token::Compare(op)) => *op,
            _ => {
                return Err(expression_error(
                    self.expr,
                    &format!("expected a comparison operator after field '{field_name}'"),
                ));
            }
        };
        self.pos += 1;

        let operand = match self.tokens.get(self.pos) {
            Some(Token::Number(value)) => Operand::Number(*value),
            Some(Token::Str(text)) => Operand::Text(text.clone()),
            Some(Token::Ident(text)) => Operand::Text(text.clone()),
            _ => {
                return Err(expression_error(
                    self.expr,
                    &format!("expected a literal value after operator for field '{field_name}'"),
                ));
            }
        };
        self.pos += 1;

        let field_value = self.resolve_field(&field_name)?;
        self.compare(field_value, op, operand)
    }

    /// Resolve a field name to its current value.
    fn resolve_field(&self, name: &str) -> AutogradResult<FieldValue> {
        let value = match name {
            "operation" => FieldValue::Text(self.event.operation.clone()),
            "event_type" => FieldValue::Text(format!("{:?}", self.event.event_type)),
            "memory" | "memory_allocated" => {
                FieldValue::Number(self.event.memory_allocated.unwrap_or(0) as f64)
            }
            "memory_deallocated" => {
                FieldValue::Number(self.event.memory_deallocated.unwrap_or(0) as f64)
            }
            "memory_usage" => FieldValue::Number(self.context.memory_usage as f64),
            "operation_index" => FieldValue::Number(self.context.operation_index as f64),
            "total_operations" => FieldValue::Number(self.context.total_operations as f64),
            "input_count" => FieldValue::Number(self.event.input_ids.len() as f64),
            "output_count" => FieldValue::Number(self.event.output_ids.len() as f64),
            "gradient_norm" => match extract_gradient_norm(self.event, self.context) {
                Some(norm) => FieldValue::Number(norm),
                None => FieldValue::Missing,
            },
            "duration_micros" => match self.event.duration {
                Some(duration) => FieldValue::Number(duration.as_micros() as f64),
                None => FieldValue::Missing,
            },
            other => {
                return Err(expression_error(
                    self.expr,
                    &format!("unknown field '{other}'"),
                ));
            }
        };
        Ok(value)
    }

    /// Evaluate a single comparison.
    fn compare(&self, field: FieldValue, op: CompareOp, operand: Operand) -> AutogradResult<bool> {
        match field {
            // Supported field, but no data available: the comparison is simply
            // not satisfied (this is honest, not a fabricated value).
            FieldValue::Missing => Ok(false),
            FieldValue::Number(lhs) => {
                let rhs = match operand {
                    Operand::Number(value) => value,
                    Operand::Text(text) => text.trim().parse::<f64>().map_err(|_| {
                        expression_error(
                            self.expr,
                            &format!(
                                "type mismatch: numeric field compared with non-numeric value '{text}'"
                            ),
                        )
                    })?,
                };
                Ok(apply_numeric_compare(lhs, op, rhs))
            }
            FieldValue::Text(lhs) => match op {
                CompareOp::Eq | CompareOp::Ne => {
                    let rhs = match operand {
                        Operand::Text(text) => text,
                        Operand::Number(_) => {
                            return Err(expression_error(
                                self.expr,
                                "type mismatch: text field requires a string operand",
                            ));
                        }
                    };
                    Ok(if op == CompareOp::Eq {
                        lhs == rhs
                    } else {
                        lhs != rhs
                    })
                }
                _ => Err(expression_error(
                    self.expr,
                    "ordering comparison is not supported for text fields (use == or !=)",
                )),
            },
        }
    }
}

/// Apply a comparison operator to two floats (NaN-safe; NaN comparisons are
/// false). Uses `partial_cmp` to avoid direct float equality.
fn apply_numeric_compare(lhs: f64, op: CompareOp, rhs: f64) -> bool {
    match lhs.partial_cmp(&rhs) {
        None => false,
        Some(ordering) => match op {
            CompareOp::Eq => ordering == Ordering::Equal,
            CompareOp::Ne => ordering != Ordering::Equal,
            CompareOp::Lt => ordering == Ordering::Less,
            CompareOp::Le => ordering != Ordering::Greater,
            CompareOp::Gt => ordering == Ordering::Greater,
            CompareOp::Ge => ordering != Ordering::Less,
        },
    }
}

// ---------------------------------------------------------------------------
// Recorded-graph navigation (step over / step out)
// ---------------------------------------------------------------------------

/// Lightweight snapshot of a recorded event used for navigation.
///
/// Captures only the parent/child relationship and the operation name so the
/// recorded execution tree can be traversed without holding the history lock.
#[derive(Debug, Clone)]
struct HistoryNode {
    id: TraceEventId,
    parent_id: Option<TraceEventId>,
    operation: String,
}

/// Return whether `node` is a (transitive) descendant of `ancestor` in the
/// recorded event tree. A node is never a descendant of itself.
fn is_descendant(
    node: TraceEventId,
    ancestor: TraceEventId,
    parent_map: &HashMap<TraceEventId, Option<TraceEventId>>,
) -> bool {
    let mut current = node;
    let mut guard = 0_usize;
    while let Some(Some(parent)) = parent_map.get(&current) {
        if *parent == ancestor {
            return true;
        }
        current = *parent;
        guard += 1;
        // Defensive bound against a malformed (cyclic) parent chain.
        if guard > parent_map.len() {
            break;
        }
    }
    false
}

/// Reconstruct the call stack (root-first list of operation names) for the
/// event at `index` by walking its ancestor chain in the recorded tree.
fn build_call_stack(nodes: &[HistoryNode], index: usize) -> Vec<String> {
    let by_id: HashMap<TraceEventId, &HistoryNode> = nodes.iter().map(|n| (n.id, n)).collect();
    let mut stack = Vec::new();
    let mut current = Some(nodes[index].id);
    let mut guard = 0_usize;
    while let Some(id) = current {
        match by_id.get(&id) {
            Some(node) => {
                stack.push(node.operation.clone());
                current = node.parent_id;
            }
            None => break,
        }
        guard += 1;
        if guard > nodes.len() {
            break;
        }
    }
    stack.reverse();
    stack
}

/// Watchpoint for monitoring tensor changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Watchpoint {
    /// Watchpoint ID
    pub id: u64,

    /// Tensor ID to watch
    pub tensor_id: String,

    /// Whether to break on read
    pub break_on_read: bool,

    /// Whether to break on write
    pub break_on_write: bool,

    /// Whether to break on gradient update
    pub break_on_gradient: bool,

    /// Number of times this watchpoint has been triggered
    pub trigger_count: usize,
}

/// Debug context containing current execution state
#[derive(Debug, Clone)]
pub struct DebugContext {
    /// Current operation index
    pub operation_index: usize,

    /// Total operations
    pub total_operations: usize,

    /// Current call stack
    pub call_stack: Vec<String>,

    /// Current memory usage
    pub memory_usage: usize,

    /// Tensor values (tensor_id -> description)
    pub tensor_values: HashMap<String, String>,

    /// Gradient values (tensor_id -> gradient_description)
    pub gradient_values: HashMap<String, String>,
}

impl DebugContext {
    /// Create a new debug context
    pub fn new() -> Self {
        Self {
            operation_index: 0,
            total_operations: 0,
            call_stack: Vec::new(),
            memory_usage: 0,
            tensor_values: HashMap::new(),
            gradient_values: HashMap::new(),
        }
    }
}

impl Default for DebugContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Interactive debugger for gradient computations
pub struct InteractiveDebugger {
    /// Debugger state
    state: Arc<RwLock<DebuggerState>>,

    /// Breakpoints
    breakpoints: Arc<Mutex<HashMap<u64, Breakpoint>>>,

    /// Watchpoints
    watchpoints: Arc<Mutex<HashMap<u64, Watchpoint>>>,

    /// Debug context
    context: Arc<Mutex<DebugContext>>,

    /// Execution history
    history: Arc<Mutex<VecDeque<TraceEvent>>>,

    /// Next breakpoint ID
    next_breakpoint_id: Arc<Mutex<u64>>,

    /// Next watchpoint ID
    next_watchpoint_id: Arc<Mutex<u64>>,

    /// Current path being debugged
    current_path: Arc<Mutex<Option<PathId>>>,

    /// Maximum history size
    max_history_size: usize,

    /// Command queue (for programmatic control)
    #[allow(dead_code)]
    command_queue: Arc<Mutex<VecDeque<DebugCommand>>>,
}

/// Debug command
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DebugCommand {
    /// Step to next operation
    Step,

    /// Continue execution
    Continue,

    /// Step over (skip into children)
    StepOver,

    /// Step out (return to parent)
    StepOut,

    /// Run to completion
    Run,

    /// Pause execution
    Pause,

    /// Restart from beginning
    Restart,

    /// Inspect tensor
    InspectTensor(String),

    /// Inspect gradient
    InspectGradient(String),

    /// Show call stack
    ShowCallStack,

    /// Show memory usage
    ShowMemory,

    /// List breakpoints
    ListBreakpoints,

    /// List watchpoints
    ListWatchpoints,
}

impl InteractiveDebugger {
    /// Create a new interactive debugger
    pub fn new() -> Self {
        Self {
            state: Arc::new(RwLock::new(DebuggerState::Inactive)),
            breakpoints: Arc::new(Mutex::new(HashMap::new())),
            watchpoints: Arc::new(Mutex::new(HashMap::new())),
            context: Arc::new(Mutex::new(DebugContext::new())),
            history: Arc::new(Mutex::new(VecDeque::new())),
            next_breakpoint_id: Arc::new(Mutex::new(1)), // Start IDs from 1
            next_watchpoint_id: Arc::new(Mutex::new(1)), // Start IDs from 1
            current_path: Arc::new(Mutex::new(None)),
            max_history_size: 1000,
            command_queue: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    /// Start debugging a gradient path
    pub fn start_debugging(&self, path_id: PathId) -> AutogradResult<()> {
        *self.state.write() = DebuggerState::Paused;
        *self.current_path.lock() = Some(path_id);
        self.context.lock().operation_index = 0;

        Ok(())
    }

    /// Stop debugging
    pub fn stop_debugging(&self) {
        *self.state.write() = DebuggerState::Inactive;
        *self.current_path.lock() = None;
        self.history.lock().clear();
    }

    /// Get current state
    pub fn state(&self) -> DebuggerState {
        *self.state.read()
    }

    /// Add a breakpoint
    pub fn add_breakpoint(&self, condition: BreakpointCondition, description: String) -> u64 {
        let id = {
            let mut next_id = self.next_breakpoint_id.lock();
            let id = *next_id;
            *next_id += 1;
            id
        };

        let breakpoint = Breakpoint::new(id, condition, description);
        self.breakpoints.lock().insert(id, breakpoint);

        id
    }

    /// Remove a breakpoint
    pub fn remove_breakpoint(&self, id: u64) -> AutogradResult<()> {
        self.breakpoints.lock().remove(&id);
        Ok(())
    }

    /// Enable a breakpoint
    pub fn enable_breakpoint(&self, id: u64) -> AutogradResult<()> {
        if let Some(bp) = self.breakpoints.lock().get_mut(&id) {
            bp.enabled = true;
            Ok(())
        } else {
            Err(AutogradError::Configuration {
                parameter: "breakpoint_id".to_string(),
                value: id.to_string(),
                reason: "Breakpoint not found".to_string(),
                valid_range: None,
            })
        }
    }

    /// Disable a breakpoint
    pub fn disable_breakpoint(&self, id: u64) -> AutogradResult<()> {
        if let Some(bp) = self.breakpoints.lock().get_mut(&id) {
            bp.enabled = false;
            Ok(())
        } else {
            Err(AutogradError::Configuration {
                parameter: "breakpoint_id".to_string(),
                value: id.to_string(),
                reason: "Breakpoint not found".to_string(),
                valid_range: None,
            })
        }
    }

    /// List all breakpoints
    pub fn list_breakpoints(&self) -> Vec<Breakpoint> {
        self.breakpoints.lock().values().cloned().collect()
    }

    /// Add a watchpoint
    pub fn add_watchpoint(&self, tensor_id: String) -> u64 {
        let id = {
            let mut next_id = self.next_watchpoint_id.lock();
            let id = *next_id;
            *next_id += 1;
            id
        };

        let watchpoint = Watchpoint {
            id,
            tensor_id,
            break_on_read: true,
            break_on_write: true,
            break_on_gradient: true,
            trigger_count: 0,
        };

        self.watchpoints.lock().insert(id, watchpoint);

        id
    }

    /// Remove a watchpoint
    pub fn remove_watchpoint(&self, id: u64) -> AutogradResult<()> {
        self.watchpoints.lock().remove(&id);
        Ok(())
    }

    /// List all watchpoints
    pub fn list_watchpoints(&self) -> Vec<Watchpoint> {
        self.watchpoints.lock().values().cloned().collect()
    }

    /// Process a trace event (called during execution)
    pub fn process_event(&self, event: &TraceEvent) -> AutogradResult<bool> {
        // Add to history
        {
            let mut history = self.history.lock();
            history.push_back(event.clone());

            while history.len() > self.max_history_size {
                history.pop_front();
            }
        }

        // Update context
        {
            let mut context = self.context.lock();
            context.operation_index += 1;

            if let Some(mem) = event.memory_allocated {
                context.memory_usage += mem;
            }

            if let Some(mem) = event.memory_deallocated {
                context.memory_usage = context.memory_usage.saturating_sub(mem);
            }
        }

        // Check breakpoints
        let should_break = {
            let mut breakpoints = self.breakpoints.lock();
            let context = self.context.lock();

            breakpoints
                .values_mut()
                .any(|bp| bp.should_trigger(event, &context))
        };

        if should_break {
            *self.state.write() = DebuggerState::Paused;
            return Ok(true);
        }

        // Check state
        match *self.state.read() {
            DebuggerState::Stepping => {
                *self.state.write() = DebuggerState::Paused;
                Ok(true)
            }
            DebuggerState::Paused => Ok(true),
            _ => Ok(false),
        }
    }

    /// Execute a debug command
    pub fn execute_command(&self, command: DebugCommand) -> AutogradResult<String> {
        match command {
            DebugCommand::Step => {
                *self.state.write() = DebuggerState::Stepping;
                Ok("Stepping to next operation...".to_string())
            }

            DebugCommand::Continue => {
                *self.state.write() = DebuggerState::Continuing;
                Ok("Continuing execution...".to_string())
            }

            DebugCommand::Run => {
                *self.state.write() = DebuggerState::Running;
                Ok("Running to completion...".to_string())
            }

            DebugCommand::Pause => {
                *self.state.write() = DebuggerState::Paused;
                Ok("Paused".to_string())
            }

            DebugCommand::ShowCallStack => {
                let context = self.context.lock();
                let mut output = String::from("Call Stack:\n");

                for (i, op) in context.call_stack.iter().enumerate() {
                    output.push_str(&format!("  #{}: {}\n", i, op));
                }

                Ok(output)
            }

            DebugCommand::ShowMemory => {
                let context = self.context.lock();
                Ok(format!(
                    "Current memory usage: {} bytes",
                    context.memory_usage
                ))
            }

            DebugCommand::InspectTensor(tensor_id) => {
                let context = self.context.lock();
                if let Some(desc) = context.tensor_values.get(&tensor_id) {
                    Ok(format!("Tensor {}: {}", tensor_id, desc))
                } else {
                    Ok(format!("Tensor {} not found in current context", tensor_id))
                }
            }

            DebugCommand::InspectGradient(tensor_id) => {
                let context = self.context.lock();
                if let Some(desc) = context.gradient_values.get(&tensor_id) {
                    Ok(format!("Gradient for {}: {}", tensor_id, desc))
                } else {
                    Ok(format!("Gradient for {} not found", tensor_id))
                }
            }

            DebugCommand::ListBreakpoints => {
                let breakpoints = self.list_breakpoints();
                let mut output = String::from("Breakpoints:\n");

                for bp in breakpoints {
                    output.push_str(&format!(
                        "  #{}: {} [{}] (hits: {})\n",
                        bp.id,
                        bp.description,
                        if bp.enabled { "enabled" } else { "disabled" },
                        bp.hit_count
                    ));
                }

                Ok(output)
            }

            DebugCommand::ListWatchpoints => {
                let watchpoints = self.list_watchpoints();
                let mut output = String::from("Watchpoints:\n");

                for wp in watchpoints {
                    output.push_str(&format!(
                        "  #{}: {} (triggers: {})\n",
                        wp.id, wp.tensor_id, wp.trigger_count
                    ));
                }

                Ok(output)
            }

            DebugCommand::Restart => {
                self.stop_debugging();
                Ok("Debugger restarted".to_string())
            }

            DebugCommand::StepOver => self.step_over(),

            DebugCommand::StepOut => self.step_out(),
        }
    }

    /// Snapshot the recorded execution history for navigation.
    fn snapshot_history(&self) -> Vec<HistoryNode> {
        self.history
            .lock()
            .iter()
            .map(|event| HistoryNode {
                id: event.id,
                parent_id: event.parent_id,
                operation: event.operation.clone(),
            })
            .collect()
    }

    /// Move the navigation cursor to a specific recorded-event index.
    ///
    /// The cursor (stored as `DebugContext::operation_index`) identifies the
    /// "current" event that [`DebugCommand::StepOver`] and
    /// [`DebugCommand::StepOut`] navigate relative to. The debug context's call
    /// stack is rebuilt from the recorded graph to reflect the new position.
    ///
    /// # Errors
    ///
    /// Returns an error when `index` is out of range for the recorded history.
    pub fn seek(&self, index: usize) -> AutogradResult<()> {
        let nodes = self.snapshot_history();
        if index >= nodes.len() {
            return Err(AutogradError::Configuration {
                parameter: "seek_index".to_string(),
                value: index.to_string(),
                reason: format!(
                    "index out of range: {} recorded event(s) available",
                    nodes.len()
                ),
                valid_range: if nodes.is_empty() {
                    Some("no recorded events".to_string())
                } else {
                    Some(format!("0..{}", nodes.len()))
                },
            });
        }

        let call_stack = build_call_stack(&nodes, index);
        let mut context = self.context.lock();
        context.operation_index = index;
        context.call_stack = call_stack;
        Ok(())
    }

    /// Step over the current event: advance the cursor past the current node's
    /// entire subtree, landing on the next sibling or ancestor continuation.
    ///
    /// Operates on the recorded execution graph. Returns a description of the
    /// resulting position. When there is no current event (empty history or the
    /// cursor is already at the end) an honest message is returned rather than
    /// moving an imaginary cursor.
    fn step_over(&self) -> AutogradResult<String> {
        let nodes = self.snapshot_history();
        if nodes.is_empty() {
            return Ok("No recorded execution to navigate".to_string());
        }
        let cur = self.context.lock().operation_index;
        if cur >= nodes.len() {
            return Ok(format!(
                "Already at end of recorded execution (index {cur}); nothing to step over"
            ));
        }

        let parent_map: HashMap<TraceEventId, Option<TraceEventId>> =
            nodes.iter().map(|n| (n.id, n.parent_id)).collect();
        let current_id = nodes[cur].id;
        let new_pos = ((cur + 1)..nodes.len())
            .find(|&j| !is_descendant(nodes[j].id, current_id, &parent_map))
            .unwrap_or(nodes.len());

        Ok(self.commit_navigation(&nodes, cur, new_pos, "Stepped over"))
    }

    /// Step out of the current frame: advance the cursor past the remainder of
    /// the current node's parent subtree, returning to the caller frame.
    ///
    /// Operates on the recorded execution graph. A top-level event (no parent)
    /// has no caller, so the cursor runs to the end of the recorded execution.
    fn step_out(&self) -> AutogradResult<String> {
        let nodes = self.snapshot_history();
        if nodes.is_empty() {
            return Ok("No recorded execution to navigate".to_string());
        }
        let cur = self.context.lock().operation_index;
        if cur >= nodes.len() {
            return Ok(format!(
                "Already at end of recorded execution (index {cur}); nothing to step out of"
            ));
        }

        match nodes[cur].parent_id {
            Some(parent_id) => {
                let parent_map: HashMap<TraceEventId, Option<TraceEventId>> =
                    nodes.iter().map(|n| (n.id, n.parent_id)).collect();
                let new_pos = ((cur + 1)..nodes.len())
                    .find(|&j| !is_descendant(nodes[j].id, parent_id, &parent_map))
                    .unwrap_or(nodes.len());
                Ok(self.commit_navigation(&nodes, cur, new_pos, "Stepped out to"))
            }
            None => {
                // A top-level operation has no enclosing frame to return to.
                let new_pos = nodes.len();
                let message = self.commit_navigation(&nodes, cur, new_pos, "Stepped out of");
                Ok(format!(
                    "{message} (no enclosing frame: '{}' is a top-level operation)",
                    nodes[cur].operation
                ))
            }
        }
    }

    /// Apply a navigation result: update the cursor, rebuild the call stack,
    /// pause the debugger, and produce a human-readable description.
    fn commit_navigation(
        &self,
        nodes: &[HistoryNode],
        cur: usize,
        new_pos: usize,
        verb: &str,
    ) -> String {
        let call_stack = if new_pos < nodes.len() {
            build_call_stack(nodes, new_pos)
        } else {
            Vec::new()
        };

        {
            let mut context = self.context.lock();
            context.operation_index = new_pos;
            context.call_stack = call_stack;
        }
        *self.state.write() = DebuggerState::Paused;

        if new_pos < nodes.len() {
            format!(
                "{verb} '{}' (event #{}) -> now at '{}' (event #{}, index {new_pos})",
                nodes[cur].operation, nodes[cur].id, nodes[new_pos].operation, nodes[new_pos].id
            )
        } else {
            format!(
                "{verb} '{}' (event #{}) -> reached end of recorded execution (index {new_pos})",
                nodes[cur].operation, nodes[cur].id
            )
        }
    }

    /// Get debug context
    pub fn context(&self) -> DebugContext {
        self.context.lock().clone()
    }

    /// Get execution history
    pub fn history(&self) -> Vec<TraceEvent> {
        self.history.lock().iter().cloned().collect()
    }

    /// Generate debug summary
    pub fn summary(&self) -> String {
        let context = self.context.lock();
        let breakpoints = self.breakpoints.lock();
        let watchpoints = self.watchpoints.lock();

        let mut output = String::new();

        output.push_str("=== Interactive Debugger Summary ===\n\n");
        output.push_str(&format!("State: {:?}\n", *self.state.read()));
        output.push_str(&format!(
            "Progress: {}/{} operations\n",
            context.operation_index, context.total_operations
        ));
        output.push_str(&format!("Memory usage: {} bytes\n", context.memory_usage));
        output.push_str(&format!(
            "Breakpoints: {} ({} enabled)\n",
            breakpoints.len(),
            breakpoints.values().filter(|b| b.enabled).count()
        ));
        output.push_str(&format!("Watchpoints: {}\n", watchpoints.len()));

        output
    }
}

impl Default for InteractiveDebugger {
    fn default() -> Self {
        Self::new()
    }
}

/// Global interactive debugger instance
static GLOBAL_DEBUGGER: once_cell::sync::Lazy<InteractiveDebugger> =
    once_cell::sync::Lazy::new(InteractiveDebugger::new);

/// Get the global debugger
pub fn global_debugger() -> &'static InteractiveDebugger {
    &GLOBAL_DEBUGGER
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_debugger_creation() {
        let debugger = InteractiveDebugger::new();
        assert_eq!(debugger.state(), DebuggerState::Inactive);
    }

    #[test]
    fn test_breakpoint_management() {
        let debugger = InteractiveDebugger::new();

        let bp_id = debugger.add_breakpoint(
            BreakpointCondition::OperationName("matmul".to_string()),
            "Break on matmul".to_string(),
        );

        assert!(bp_id > 0);

        let breakpoints = debugger.list_breakpoints();
        assert_eq!(breakpoints.len(), 1);

        debugger.disable_breakpoint(bp_id).unwrap();

        let breakpoints = debugger.list_breakpoints();
        assert!(!breakpoints[0].enabled);

        debugger.remove_breakpoint(bp_id).unwrap();

        let breakpoints = debugger.list_breakpoints();
        assert_eq!(breakpoints.len(), 0);
    }

    #[test]
    fn test_watchpoint_management() {
        let debugger = InteractiveDebugger::new();

        let wp_id = debugger.add_watchpoint("tensor_1".to_string());
        assert!(wp_id > 0);

        let watchpoints = debugger.list_watchpoints();
        assert_eq!(watchpoints.len(), 1);

        debugger.remove_watchpoint(wp_id).unwrap();

        let watchpoints = debugger.list_watchpoints();
        assert_eq!(watchpoints.len(), 0);
    }

    #[test]
    fn test_command_execution() {
        let debugger = InteractiveDebugger::new();

        let result = debugger.execute_command(DebugCommand::ShowMemory);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert!(output.contains("memory usage"));
    }

    #[test]
    fn test_state_transitions() {
        let debugger = InteractiveDebugger::new();

        debugger.execute_command(DebugCommand::Step).unwrap();
        assert_eq!(debugger.state(), DebuggerState::Stepping);

        debugger.execute_command(DebugCommand::Continue).unwrap();
        assert_eq!(debugger.state(), DebuggerState::Continuing);

        debugger.execute_command(DebugCommand::Pause).unwrap();
        assert_eq!(debugger.state(), DebuggerState::Paused);
    }

    // -- test helpers -------------------------------------------------------

    fn make_event(
        id: TraceEventId,
        parent_id: Option<TraceEventId>,
        operation: &str,
        event_type: EventType,
    ) -> TraceEvent {
        TraceEvent {
            id,
            parent_id,
            path_id: 1,
            event_type,
            operation: operation.to_string(),
            timestamp: chrono::Utc::now(),
            duration: None,
            memory_allocated: None,
            memory_deallocated: None,
            input_ids: Vec::new(),
            output_ids: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    // -- (a) gradient-norm checking ----------------------------------------

    #[test]
    fn test_l2_norm_from_csv_is_correct() {
        // 3-4-5 right triangle: ||(3, 4)|| == 5.
        let norm = l2_norm_from_csv("3, 4").expect("should parse");
        assert!((norm - 5.0).abs() < 1e-9, "got {norm}");

        // ||(1, 2, 2)|| == 3.
        let norm = l2_norm_from_csv("1,2,2").expect("should parse");
        assert!((norm - 3.0).abs() < 1e-9, "got {norm}");

        // Malformed data is rejected rather than treated as zero.
        assert!(l2_norm_from_csv("3, oops").is_none());
        assert!(l2_norm_from_csv("").is_none());
    }

    #[test]
    fn test_extract_gradient_norm_from_metadata_and_context() {
        let context = DebugContext::new();

        // Pre-computed norm wins and is taken as absolute value.
        let mut event = make_event(1, None, "backward", EventType::GradientComputation);
        event
            .metadata
            .insert("gradient_norm".to_string(), "-7.5".to_string());
        let norm = extract_gradient_norm(&event, &context).expect("norm available");
        assert!((norm - 7.5).abs() < 1e-9, "got {norm}");

        // Raw components are reduced to their L2 norm.
        let mut event = make_event(2, None, "backward", EventType::GradientComputation);
        event
            .metadata
            .insert("gradient_values".to_string(), "6,8".to_string());
        let norm = extract_gradient_norm(&event, &context).expect("norm available");
        assert!((norm - 10.0).abs() < 1e-9, "got {norm}");

        // Per-output-tensor gradients are aggregated into a global norm
        // (sqrt(5^2 + 12^2) == 13) for gradient events only.
        let mut context = DebugContext::new();
        context
            .gradient_values
            .insert("out_a".to_string(), "3,4".to_string());
        context
            .gradient_values
            .insert("out_b".to_string(), "norm=12".to_string());
        let mut event = make_event(3, None, "backward", EventType::BackwardEnd);
        event.output_ids = vec!["out_a".to_string(), "out_b".to_string()];
        let norm = extract_gradient_norm(&event, &context).expect("norm available");
        assert!((norm - 13.0).abs() < 1e-9, "got {norm}");

        // A non-gradient event does not read context gradients (no fabrication).
        let mut forward = make_event(4, None, "forward", EventType::OperationBegin);
        forward.output_ids = vec!["out_a".to_string()];
        assert!(extract_gradient_norm(&forward, &context).is_none());

        // No data anywhere -> None.
        let bare = make_event(5, None, "backward", EventType::GradientComputation);
        assert!(extract_gradient_norm(&bare, &context).is_none());
    }

    #[test]
    fn test_gradient_explosion_breakpoint_threshold_flag() {
        let context = DebugContext::new();

        let mut bp = Breakpoint::new(
            1,
            BreakpointCondition::GradientExplosion(10.0),
            "explosion".to_string(),
        );

        // norm == 5 (from 3,4): below threshold 10 -> no trigger.
        let mut small = make_event(1, None, "backward", EventType::GradientComputation);
        small
            .metadata
            .insert("gradient_values".to_string(), "3,4".to_string());
        assert!(!bp.should_trigger(&small, &context));
        assert_eq!(bp.hit_count, 0);

        // norm == 50 (from 30,40): above threshold 10 -> trigger and count.
        let mut big = make_event(2, None, "backward", EventType::GradientComputation);
        big.metadata
            .insert("gradient_values".to_string(), "30,40".to_string());
        assert!(bp.should_trigger(&big, &context));
        assert_eq!(bp.hit_count, 1);

        // No gradient data -> never fires (does not fabricate a norm).
        let bare = make_event(3, None, "backward", EventType::GradientComputation);
        assert!(!bp.should_trigger(&bare, &context));
        assert_eq!(bp.hit_count, 1);
    }

    #[test]
    fn test_gradient_vanishing_breakpoint_threshold_flag() {
        let context = DebugContext::new();

        let mut bp = Breakpoint::new(
            1,
            BreakpointCondition::GradientVanishing(1e-3),
            "vanishing".to_string(),
        );

        // norm == 5: above threshold -> no trigger.
        let mut healthy = make_event(1, None, "backward", EventType::GradientComputation);
        healthy
            .metadata
            .insert("gradient_values".to_string(), "3,4".to_string());
        assert!(!bp.should_trigger(&healthy, &context));

        // norm == 1e-4 (below 1e-3) -> trigger.
        let mut tiny = make_event(2, None, "backward", EventType::GradientComputation);
        tiny.metadata
            .insert("gradient_norm".to_string(), "0.0001".to_string());
        assert!(bp.should_trigger(&tiny, &context));
        assert_eq!(bp.hit_count, 1);
    }

    // -- (b) custom expression evaluation -----------------------------------

    #[test]
    fn test_custom_expression_evaluates_known_values() {
        let mut event = make_event(1, None, "matmul", EventType::OperationBegin);
        event.memory_allocated = Some(2048);
        event.input_ids = vec!["a".to_string(), "b".to_string()];
        event.output_ids = vec!["c".to_string()];

        let mut context = DebugContext::new();
        context.operation_index = 5;
        context.memory_usage = 1000;

        // String equality, both quoted and bareword forms.
        assert!(evaluate_custom_expression("operation == matmul", &event, &context).unwrap());
        assert!(evaluate_custom_expression("operation == \"matmul\"", &event, &context).unwrap());
        assert!(!evaluate_custom_expression("operation == \"add\"", &event, &context).unwrap());
        assert!(evaluate_custom_expression("operation != add", &event, &context).unwrap());

        // Numeric ordering.
        assert!(evaluate_custom_expression("operation_index >= 5", &event, &context).unwrap());
        assert!(!evaluate_custom_expression("operation_index > 5", &event, &context).unwrap());
        assert!(evaluate_custom_expression("memory > 1000", &event, &context).unwrap());
        assert!(evaluate_custom_expression("input_count == 2", &event, &context).unwrap());
        assert!(evaluate_custom_expression("output_count == 1", &event, &context).unwrap());

        // event_type as a bareword.
        assert!(
            evaluate_custom_expression("event_type == OperationBegin", &event, &context).unwrap()
        );

        // Logical AND / OR with correct precedence (&& binds tighter than ||).
        assert!(evaluate_custom_expression(
            "memory > 1000 && operation == matmul",
            &event,
            &context
        )
        .unwrap());
        assert!(!evaluate_custom_expression(
            "memory < 1000 && operation == matmul",
            &event,
            &context
        )
        .unwrap());
        assert!(evaluate_custom_expression(
            "memory < 1000 || operation_index == 5",
            &event,
            &context
        )
        .unwrap());
        // false && true || true == (false && true) || true == true
        assert!(evaluate_custom_expression(
            "operation == add && memory > 1000 || output_count == 1",
            &event,
            &context
        )
        .unwrap());
    }

    #[test]
    fn test_custom_expression_gradient_norm_field() {
        let mut event = make_event(1, None, "backward", EventType::GradientComputation);
        event
            .metadata
            .insert("gradient_norm".to_string(), "50".to_string());
        let context = DebugContext::new();

        assert!(evaluate_custom_expression("gradient_norm > 10", &event, &context).unwrap());
        assert!(!evaluate_custom_expression("gradient_norm < 10", &event, &context).unwrap());

        // Missing gradient data: the field is supported but evaluates false,
        // it does not error and does not fabricate a value.
        let bare = make_event(2, None, "backward", EventType::GradientComputation);
        assert!(!evaluate_custom_expression("gradient_norm > 10", &bare, &context).unwrap());
        assert!(!evaluate_custom_expression("gradient_norm < 10", &bare, &context).unwrap());
    }

    #[test]
    fn test_custom_expression_unsupported_inputs_error() {
        let event = make_event(1, None, "matmul", EventType::OperationBegin);
        let context = DebugContext::new();

        // Unknown field.
        assert!(evaluate_custom_expression("frobnicate == 5", &event, &context).is_err());
        // Empty expression.
        assert!(evaluate_custom_expression("   ", &event, &context).is_err());
        // Single '=' is not a valid operator.
        assert!(evaluate_custom_expression("operation = matmul", &event, &context).is_err());
        // Missing operand.
        assert!(evaluate_custom_expression("operation ==", &event, &context).is_err());
        // Type mismatch: numeric field vs non-numeric literal.
        assert!(evaluate_custom_expression("operation_index < matmul", &event, &context).is_err());
        // Ordering on a text field is unsupported.
        assert!(evaluate_custom_expression("operation < matmul", &event, &context).is_err());
        // Trailing garbage.
        assert!(evaluate_custom_expression("operation == matmul extra", &event, &context).is_err());
    }

    #[test]
    fn test_custom_breakpoint_should_trigger() {
        let context = DebugContext::new();
        let mut bp = Breakpoint::new(
            1,
            BreakpointCondition::Custom("operation == matmul".to_string()),
            "custom".to_string(),
        );

        let matmul = make_event(1, None, "matmul", EventType::OperationBegin);
        assert!(bp.should_trigger(&matmul, &context));
        assert_eq!(bp.hit_count, 1);

        let add = make_event(2, None, "add", EventType::OperationBegin);
        assert!(!bp.should_trigger(&add, &context));
        assert_eq!(bp.hit_count, 1);

        // A malformed custom expression must never fire (no fabricated hit).
        let mut broken = Breakpoint::new(
            2,
            BreakpointCondition::Custom("frobnicate == 5".to_string()),
            "broken".to_string(),
        );
        assert!(!broken.should_trigger(&matmul, &context));
        assert_eq!(broken.hit_count, 0);
    }

    // -- (c) step-over / step-out navigation --------------------------------

    /// Build a debugger with a known recorded execution tree:
    /// ```text
    /// 1 forward            (root,  index 0)
    /// 2  layer1   parent 1 (index 1)
    /// 3   matmul  parent 2 (index 2)
    /// 4   add     parent 2 (index 3)
    /// 5  layer2   parent 1 (index 4)
    /// 6   matmul  parent 5 (index 5)
    /// 7 backward           (root,  index 6)
    /// ```
    fn debugger_with_tree() -> InteractiveDebugger {
        let debugger = InteractiveDebugger::new();
        let events = [
            make_event(1, None, "forward", EventType::OperationBegin),
            make_event(2, Some(1), "layer1", EventType::OperationBegin),
            make_event(3, Some(2), "matmul", EventType::OperationBegin),
            make_event(4, Some(2), "add", EventType::OperationBegin),
            make_event(5, Some(1), "layer2", EventType::OperationBegin),
            make_event(6, Some(5), "matmul", EventType::OperationBegin),
            make_event(7, None, "backward", EventType::BackwardBegin),
        ];
        for event in &events {
            debugger.process_event(event).unwrap();
        }
        debugger
    }

    #[test]
    fn test_step_over_skips_subtree() {
        let debugger = debugger_with_tree();

        // Positioned at "layer1" (index 1); its subtree is {matmul, add}.
        debugger.seek(1).unwrap();
        let msg = debugger.execute_command(DebugCommand::StepOver).unwrap();
        // Should land on "layer2" (index 4), skipping the subtree.
        assert_eq!(debugger.context().operation_index, 4);
        assert_eq!(debugger.state(), DebuggerState::Paused);
        assert!(msg.contains("layer2"), "message was: {msg}");

        // From "layer2" (index 4), step over skips {matmul} -> "backward" (6).
        debugger.seek(4).unwrap();
        debugger.execute_command(DebugCommand::StepOver).unwrap();
        assert_eq!(debugger.context().operation_index, 6);
    }

    #[test]
    fn test_step_over_at_end_runs_to_end() {
        let debugger = debugger_with_tree();
        // Last event (index 6) has no following events.
        debugger.seek(6).unwrap();
        debugger.execute_command(DebugCommand::StepOver).unwrap();
        assert_eq!(debugger.context().operation_index, 7); // == history length
    }

    #[test]
    fn test_step_out_returns_to_caller_frame() {
        let debugger = debugger_with_tree();

        // Positioned at "matmul" (index 2) inside "layer1"; stepping out should
        // finish layer1's frame and land on its sibling "layer2" (index 4).
        debugger.seek(2).unwrap();
        let msg = debugger.execute_command(DebugCommand::StepOut).unwrap();
        assert_eq!(debugger.context().operation_index, 4);
        assert_eq!(debugger.state(), DebuggerState::Paused);
        assert!(msg.contains("layer2"), "message was: {msg}");

        // From "add" (index 3), also inside layer1 -> step out to "layer2" (4).
        debugger.seek(3).unwrap();
        debugger.execute_command(DebugCommand::StepOut).unwrap();
        assert_eq!(debugger.context().operation_index, 4);
    }

    #[test]
    fn test_step_out_of_top_level_runs_to_end() {
        let debugger = debugger_with_tree();
        // "forward" (index 0) is a top-level frame: no caller to return to.
        debugger.seek(0).unwrap();
        let msg = debugger.execute_command(DebugCommand::StepOut).unwrap();
        assert_eq!(debugger.context().operation_index, 7); // ran to end
        assert!(
            msg.contains("top-level") || msg.contains("end of recorded execution"),
            "message was: {msg}"
        );
    }

    #[test]
    fn test_seek_rebuilds_call_stack_and_validates_range() {
        let debugger = debugger_with_tree();

        // matmul (index 2) is forward -> layer1 -> matmul.
        debugger.seek(2).unwrap();
        assert_eq!(
            debugger.context().call_stack,
            vec![
                "forward".to_string(),
                "layer1".to_string(),
                "matmul".to_string()
            ]
        );

        // Out-of-range seek is an honest error.
        assert!(debugger.seek(99).is_err());
    }

    #[test]
    fn test_step_commands_on_empty_history() {
        let debugger = InteractiveDebugger::new();
        let over = debugger.execute_command(DebugCommand::StepOver).unwrap();
        let out = debugger.execute_command(DebugCommand::StepOut).unwrap();
        assert!(over.contains("No recorded execution"));
        assert!(out.contains("No recorded execution"));
    }
}
