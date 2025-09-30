# ToRSh FX Intermediate Representation (IR) Specification

This document specifies the intermediate representation used by ToRSh FX for graph-based computation and transformation.

## Table of Contents

1. [Overview](#overview)
2. [Graph Structure](#graph-structure)
3. [Node Types](#node-types)
4. [Edge Semantics](#edge-semantics)
5. [Data Types](#data-types)
6. [Control Flow](#control-flow)
7. [Serialization Format](#serialization-format)
8. [Validation Rules](#validation-rules)
9. [Extensions](#extensions)

## Overview

The ToRSh FX IR is a graph-based intermediate representation designed for:

- **Functional Programming**: Immutable data flow with explicit dependencies
- **Optimization**: Transparent representation for transformation passes
- **Portability**: Serializable format for cross-platform deployment
- **Analysis**: Rich metadata for static analysis and optimization

### Design Principles

1. **Explicit Data Flow**: All data dependencies are represented as edges
2. **Type Safety**: Strong typing with explicit type information
3. **Composability**: Graphs can be composed and decomposed
4. **Determinism**: Execution order is fully determined by the graph structure

## Graph Structure

### FxGraph

The root structure representing a computational graph:

```rust
pub struct FxGraph {
    graph: Graph<Node, Edge>,     // Directed graph using petgraph
    inputs: Vec<NodeIndex>,       // Input nodes
    outputs: Vec<NodeIndex>,      // Output nodes
}
```

**Properties:**
- **Directed Acyclic Graph (DAG)**: No cycles allowed except for loop constructs
- **Single Static Assignment (SSA)**: Each value is assigned exactly once
- **Explicit I/O**: Clear input and output boundaries

### Graph Invariants

1. **Topological Order**: Nodes can be executed in topological order
2. **Reachability**: All output nodes must be reachable from input nodes
3. **Type Consistency**: Connected nodes must have compatible types
4. **Control Flow Integrity**: Control flow constructs must be well-formed

## Node Types

### Input Node

Represents an input to the computation graph.

```rust
Node::Input(String)
```

**Semantics:**
- No predecessors
- Exactly one output value
- Must be connected to downstream nodes

**Example:**
```rust
Node::Input("x".to_string())
```

**Properties:**
- `name`: Unique identifier for the input
- `type`: Data type (inferred or explicit)
- `shape`: Tensor shape (may be dynamic)

### Call Node

Represents a function call or operation.

```rust
Node::Call(String, Vec<String>)
```

**Semantics:**
- Zero or more inputs (arguments)
- Exactly one output value
- Operation semantics defined by the operation name

**Example:**
```rust
Node::Call("relu".to_string(), vec!["x".to_string()])
```

**Properties:**
- `operation`: Name of the operation to execute
- `arguments`: List of input value names
- `attributes`: Optional operation-specific parameters

### Output Node

Represents an output from the computation graph.

```rust
Node::Output
```

**Semantics:**
- Exactly one input
- No successors
- Defines the result of the computation

**Example:**
```rust
Node::Output
```

**Properties:**
- Connected to exactly one predecessor
- May have explicit type and shape annotations

### GetAttr Node

Represents attribute access on objects.

```rust
Node::GetAttr {
    target: String,
    attr: String,
}
```

**Semantics:**
- Accesses a named attribute of a target object
- Used for parameter access in modules

**Example:**
```rust
Node::GetAttr {
    target: "module".to_string(),
    attr: "weight".to_string(),
}
```

**Properties:**
- `target`: Name of the object to access
- `attr`: Name of the attribute to retrieve

### Conditional Node

Represents conditional execution (if-then-else).

```rust
Node::Conditional {
    condition: String,
    then_branch: Vec<String>,
    else_branch: Vec<String>,
}
```

**Semantics:**
- Conditional execution based on a boolean condition
- Exactly one of the branches is executed
- Both branches must produce compatible output types

**Example:**
```rust
Node::Conditional {
    condition: "cond".to_string(),
    then_branch: vec!["true_value".to_string()],
    else_branch: vec!["false_value".to_string()],
}
```

**Properties:**
- `condition`: Boolean value determining which branch to execute
- `then_branch`: Values produced if condition is true
- `else_branch`: Values produced if condition is false

### Loop Node

Represents iterative execution.

```rust
Node::Loop {
    condition: String,
    body: Vec<String>,
    loop_vars: Vec<String>,
}
```

**Semantics:**
- Executes body while condition is true
- Loop variables are updated each iteration
- Must terminate (no infinite loops in static analysis)

**Example:**
```rust
Node::Loop {
    condition: "i < 10".to_string(),
    body: vec!["body_result".to_string()],
    loop_vars: vec!["i".to_string()],
}
```

**Properties:**
- `condition`: Boolean condition for loop continuation
- `body`: Computations performed in each iteration
- `loop_vars`: Variables that change each iteration

### Merge Node

Represents convergence of control flow.

```rust
Node::Merge {
    inputs: Vec<String>,
}
```

**Semantics:**
- Combines values from different control flow paths
- All inputs must have compatible types
- Typically follows conditional or loop constructs

**Example:**
```rust
Node::Merge {
    inputs: vec!["path1_result".to_string(), "path2_result".to_string()],
}
```

**Properties:**
- `inputs`: List of values to merge
- Output type is the unified type of all inputs

## Edge Semantics

### Edge Structure

```rust
pub struct Edge {
    pub name: String,
}
```

**Properties:**
- `name`: Identifier for the data flowing through this edge
- Represents a value dependency between nodes

### Edge Types

1. **Data Edges**: Carry tensor values between operations
2. **Control Edges**: Represent control dependencies (implicit)
3. **Attribute Edges**: Connect attribute accesses to their targets

### Edge Constraints

- **Single Producer**: Each edge has exactly one source node
- **Multiple Consumers**: Edges can have multiple target nodes
- **Type Compatibility**: Source and target must have compatible types
- **Execution Order**: Edges define partial execution order

## Data Types

### Primitive Types

```rust
pub enum DType {
    Float32,
    Float64,
    Int32,
    Int64,
    Bool,
    Complex32,
    Complex64,
    String,
}
```

### Tensor Types

```rust
pub struct TensorType {
    pub dtype: DType,
    pub shape: Shape,
    pub device: Device,
}
```

**Properties:**
- `dtype`: Element data type
- `shape`: Tensor dimensions (may include dynamic dimensions)
- `device`: Target device for computation

### Dynamic Types

```rust
pub struct DynamicType {
    pub base_type: TensorType,
    pub constraints: Vec<ShapeConstraint>,
}
```

**Shape Constraints:**
- `Equal(dim1, dim2)`: Two dimensions must be equal
- `Divisible(dim, factor)`: Dimension must be divisible by factor
- `Range(dim, min, max)`: Dimension must be within range

### Function Types

```rust
pub struct FunctionType {
    pub inputs: Vec<TensorType>,
    pub outputs: Vec<TensorType>,
    pub attributes: HashMap<String, AttributeType>,
}
```

## Control Flow

### Structured Control Flow

ToRSh FX uses structured control flow with explicit entry and exit points:

1. **Conditional**: Single entry, two paths, single exit (merge)
2. **Loop**: Single entry, iterative body, single exit
3. **Sequential**: Linear execution order

### Control Flow Graph Properties

- **Reducible**: All control flow can be structured
- **Single Entry/Exit**: Each control construct has clear boundaries
- **Nested Structure**: Control flow constructs can be nested arbitrarily

### Example: Conditional Execution

```
Input(x) -> Condition(x > 0) -> Conditional {
                                   then: ReLU(x)
                                   else: Sigmoid(x)
                                } -> Merge -> Output
```

### Example: Loop Execution

```
Input(x), Input(i) -> Loop {
                         condition: i < 10
                         body: x = x + 1, i = i + 1
                         vars: [x, i]
                      } -> Output(x)
```

## Serialization Format

### JSON Representation

```json
{
  "nodes": [
    {
      "id": 0,
      "type": "Input",
      "data": "x"
    },
    {
      "id": 1,
      "type": "Call",
      "data": {
        "operation": "relu",
        "arguments": ["x"]
      }
    },
    {
      "id": 2,
      "type": "Output",
      "data": null
    }
  ],
  "edges": [
    {
      "source": 0,
      "target": 1,
      "name": "x"
    },
    {
      "source": 1,
      "target": 2,
      "name": "relu_output"
    }
  ],
  "inputs": [0],
  "outputs": [2],
  "metadata": {
    "version": "1.0",
    "created": "2024-01-01T00:00:00Z"
  }
}
```

### Binary Representation

The binary format uses MessagePack for efficient serialization:

```rust
pub struct SerializableGraph {
    nodes: Vec<(usize, Node)>,
    edges: Vec<(usize, usize, Edge)>,
    inputs: Vec<usize>,
    outputs: Vec<usize>,
    metadata: HashMap<String, String>,
}
```

**Properties:**
- Compact representation for large graphs
- Fast serialization/deserialization
- Preserves all graph structure and metadata

## Validation Rules

### Structural Validation

1. **Graph Connectivity**:
   - All output nodes reachable from inputs
   - No unreachable nodes (except during construction)
   - Proper edge connectivity

2. **Node Constraints**:
   - Input nodes have no predecessors
   - Output nodes have no successors
   - Call nodes have proper argument count

3. **Control Flow Validation**:
   - Conditional nodes have exactly two branches
   - Loop nodes have proper condition and body structure
   - Merge nodes have compatible input types

### Semantic Validation

1. **Type Checking**:
   - All operations have compatible input types
   - Edge types match connected node types
   - Return types match expected output types

2. **Shape Validation**:
   - Broadcasting rules are followed
   - Dynamic shapes have consistent constraints
   - Operations have valid input/output shapes

3. **Control Flow Semantics**:
   - Conditional branches have compatible output types
   - Loop variables are properly updated
   - No infinite loops in static analysis

### Implementation

```rust
pub struct GraphValidator {
    rules: Vec<Box<dyn ValidationRule>>,
}

pub trait ValidationRule {
    fn validate(&self, graph: &FxGraph) -> Result<(), ValidationError>;
    fn name(&self) -> &str;
}

impl GraphValidator {
    pub fn validate(&self, graph: &FxGraph) -> Result<(), Vec<ValidationError>> {
        let mut errors = Vec::new();
        
        for rule in &self.rules {
            if let Err(error) = rule.validate(graph) {
                errors.push(error);
            }
        }
        
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}
```

## Extensions

### Custom Operations

```rust
pub struct CustomOperation {
    pub name: String,
    pub input_types: Vec<TensorType>,
    pub output_types: Vec<TensorType>,
    pub attributes: HashMap<String, AttributeType>,
    pub implementation: OperationImpl,
}

pub trait OperationImpl {
    fn execute(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>, ExecutionError>;
    fn infer_shapes(&self, input_shapes: &[Shape]) -> Result<Vec<Shape>, ShapeError>;
}
```

### Metadata Annotations

```rust
pub struct NodeMetadata {
    pub debug_info: Option<DebugInfo>,
    pub performance_hints: Vec<PerformanceHint>,
    pub device_constraints: Vec<DeviceConstraint>,
    pub custom_attributes: HashMap<String, serde_json::Value>,
}

pub struct DebugInfo {
    pub source_location: SourceLocation,
    pub original_name: Option<String>,
    pub documentation: Option<String>,
}
```

### Quantization Annotations

```rust
pub struct QuantizationInfo {
    pub scheme: QuantizationScheme,
    pub parameters: QuantizationParams,
    pub calibration_data: Option<CalibrationData>,
}

pub enum QuantizationScheme {
    Symmetric,
    Asymmetric,
    Dynamic,
}
```

### Dynamic Shape Extensions

```rust
pub struct DynamicShapeInfo {
    pub symbolic_dimensions: HashMap<String, DynamicDim>,
    pub constraints: Vec<ShapeConstraint>,
    pub inference_context: ShapeInferenceContext,
}

pub struct DynamicDim {
    pub name: String,
    pub min_value: Option<usize>,
    pub max_value: Option<usize>,
    pub divisibility_constraints: Vec<usize>,
}
```

## Version Compatibility

### Version 1.0 Features

- Basic node types (Input, Call, Output)
- Simple control flow (Conditional, Loop)
- Static shapes and types
- JSON/Binary serialization

### Future Extensions

- Advanced control flow constructs
- First-class function support
- Gradual typing system
- Hardware-specific annotations
- Distributed execution metadata

### Backward Compatibility

ToRSh FX maintains backward compatibility through:

1. **Versioned Serialization**: Each graph includes version information
2. **Progressive Enhancement**: New features are additive
3. **Migration Tools**: Automatic conversion between versions
4. **Deprecation Policy**: Gradual removal of obsolete features

## Best Practices

### Graph Construction

1. **Minimize Graph Complexity**: Keep graphs as simple as possible
2. **Use Structured Control Flow**: Prefer structured constructs over arbitrary control flow
3. **Explicit Dependencies**: Make all data dependencies explicit through edges
4. **Type Annotations**: Include type information for better optimization

### Performance Considerations

1. **Node Granularity**: Balance between fine-grained and coarse-grained operations
2. **Memory Layout**: Consider data layout for efficient execution
3. **Device Placement**: Annotate preferred device placement
4. **Batching**: Design for efficient batch processing

### Debugging and Maintenance

1. **Debug Information**: Include source location and naming information
2. **Graph Visualization**: Ensure graphs can be visualized effectively
3. **Validation**: Run validation checks during development
4. **Testing**: Include both positive and negative test cases

This IR specification provides the foundation for all ToRSh FX functionality and ensures consistency across different components of the framework.