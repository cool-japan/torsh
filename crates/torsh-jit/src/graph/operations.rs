//! Operation definitions and related structures

use crate::graph::core::NodeId;

/// Parameter information for parameter nodes
#[derive(Debug, Clone, PartialEq)]
pub struct ParameterInfo {
    pub name: String,
    pub trainable: bool,
}

/// Constant information for constant nodes
#[derive(Debug, Clone, PartialEq)]
pub struct ConstantInfo {
    pub value: ConstantValue,
}

/// Constant values that can be stored in constant nodes
#[derive(Debug, Clone, PartialEq)]
pub enum ConstantValue {
    Bool(bool),
    Int(i64),
    UInt(u64),
    Float(f64),
    String(String),
    FloatArray(Vec<f32>),
    IntArray(Vec<i64>),
    Array(Vec<ConstantValue>),
    Tensor {
        shape: Vec<usize>,
        data: Vec<f64>,
        dtype: String,
    },
    Complex {
        real: f64,
        imag: f64,
    },
    None,
    Undefined,

    // Legacy aliases for backward compatibility
    Scalar(f64),    // Alias for Float
    IntScalar(i64), // Alias for Int
}

/// Convolution 2D operation information
#[derive(Debug, Clone, PartialEq)]
pub struct Conv2dInfo {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: (usize, usize),
    pub stride: (usize, usize),
    pub padding: (usize, usize),
    pub dilation: (usize, usize),
    pub groups: usize,
}

/// Linear layer operation information
#[derive(Debug, Clone, PartialEq)]
pub struct LinearInfo {
    pub in_features: usize,
    pub out_features: usize,
    pub bias: bool,
}

/// Batch normalization operation information
#[derive(Debug, Clone, PartialEq)]
pub struct BatchNormInfo {
    pub num_features: usize,
    pub eps: f32,
    pub momentum: f32,
    pub affine: bool,
    pub track_running_stats: bool,
}

/// Pooling operation information
#[derive(Debug, Clone, PartialEq)]
pub struct PoolInfo {
    pub kernel_size: (usize, usize),
    pub stride: Option<(usize, usize)>,
    pub padding: (usize, usize),
    pub ceil_mode: bool,
}

/// Slice range for slice operations
#[derive(Debug, Clone, PartialEq)]
pub struct SliceRange {
    pub start: Option<isize>,
    pub end: Option<isize>,
    pub step: Option<isize>,
}

/// If operation information for conditional execution
#[derive(Debug, Clone, PartialEq)]
pub struct IfInfo {
    pub condition: NodeId,
    pub then_block: NodeId,
    pub else_block: Option<NodeId>,
    pub merge_point: Option<NodeId>,
}

/// While loop operation information
#[derive(Debug, Clone, PartialEq)]
pub struct WhileInfo {
    pub condition: NodeId,
    pub body: NodeId,
    pub max_iterations: Option<usize>,
}

/// For loop operation information
#[derive(Debug, Clone, PartialEq)]
pub struct ForInfo {
    pub start: NodeId,
    pub end: NodeId,
    pub step: Option<NodeId>,
    pub body: NodeId,
    pub index_var: String,
}

/// Block operation information for grouping operations
#[derive(Debug, Clone, PartialEq)]
pub struct BlockInfo {
    pub block_type: BlockType,
    pub operations: Vec<NodeId>,
    pub name: Option<String>,
}

/// Types of execution blocks
#[derive(Debug, Clone, PartialEq)]
pub enum BlockType {
    Sequential,  // Operations execute in sequence
    Parallel,    // Operations can execute in parallel
    Conditional, // Block executes conditionally
    Loop,        // Block is a loop body
    Function,    // Block is a function body
}

/// Merge operation information for combining control flow
#[derive(Debug, Clone, PartialEq)]
pub struct MergeInfo {
    pub inputs: Vec<NodeId>,
    pub strategy: MergeStrategy,
}

/// Strategies for merging control flow
#[derive(Debug, Clone, PartialEq)]
pub enum MergeStrategy {
    Select, // Select one input based on condition
    Phi,    // PHI node (select based on predecessor)
    Stack,  // Stack multiple values
    Concat, // Concatenate tensors
}

/// Node attributes for additional metadata
#[derive(Debug, Clone, PartialEq)]
pub enum Attribute {
    String(String),
    Int(i64), // Renamed from Integer for consistency
    Float(f64),
    Bool(bool), // Renamed from Boolean for consistency
    IntegerList(Vec<i64>),
    FloatList(Vec<f64>),

    // Legacy aliases for backward compatibility
    Integer(i64),      // Alias for Int
    Boolean(bool),     // Alias for Bool
    IntList(Vec<i64>), // Alias for IntegerList
}

/// Operation types
#[derive(Debug, Clone, PartialEq)]
pub enum Operation {
    // Input/Output
    Input,
    Parameter(ParameterInfo),
    Constant(ConstantInfo),

    // Element-wise unary
    Neg,
    Abs,
    Exp,
    Log,
    Sqrt,
    Sin,
    Cos,
    Tanh,
    Sigmoid,
    Relu,
    Gelu,
    Silu,
    Softmax {
        dim: Option<usize>,
    },
    LogSoftmax {
        dim: Option<usize>,
    },

    // Element-wise binary
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Maximum,
    Minimum,

    // Reduction
    Sum {
        dims: Vec<usize>,
        keepdim: bool,
    },
    Mean {
        dims: Vec<usize>,
        keepdim: bool,
    },
    Max {
        dims: Vec<usize>,
        keepdim: bool,
    },
    Min {
        dims: Vec<usize>,
        keepdim: bool,
    },

    // Matrix operations
    MatMul,
    BatchMatMul,

    // Shape operations
    Reshape {
        shape: Vec<isize>,
    },
    Transpose {
        dims: Vec<usize>,
    },
    Squeeze {
        dims: Vec<usize>,
    },
    Unsqueeze {
        dims: Vec<usize>,
    },
    Slice {
        ranges: Vec<SliceRange>,
    },
    Concat {
        axis: usize,
    },

    // Neural network operations
    Conv2d(Conv2dInfo),
    Linear(LinearInfo),
    BatchNorm2d(BatchNormInfo),
    Dropout {
        p: f32,
    },
    MaxPool2d(PoolInfo),
    AvgPool2d(PoolInfo),

    // Control flow operations
    If(IfInfo),
    While(WhileInfo),
    For(ForInfo),
    Break,
    Continue,
    Return(Option<NodeId>),

    // Control flow blocks
    Block(BlockInfo),
    Merge(MergeInfo),

    // Custom operations
    Custom(String),

    // Indexing and data manipulation operations
    Split {
        split_size_or_sections: Vec<usize>,
        dim: usize,
    },
    Gather {
        dim: usize,
        index_shape: Vec<usize>,
    },
    Scatter {
        dim: usize,
        index_shape: Vec<usize>,
    },

    // Additional normalization operations (alias for consistency)
    BatchNorm,
    LayerNorm,

    // Fused operations for kernel fusion optimization
    FusedKernel {
        name: String,
        ops: Vec<Operation>,
        input_count: usize,
        output_count: usize,
    },

    // Loss functions
    CrossEntropy {
        reduction: Option<String>,
        ignore_index: Option<i64>,
    },
    MSELoss,
    BCELoss,

    // Utility operations
    Nop,
}

impl Operation {
    /// Return string representation of the operation
    pub fn as_str(&self) -> &str {
        match self {
            Operation::Input => "input",
            Operation::Parameter(_) => "parameter",
            Operation::Constant(_) => "constant",
            Operation::Neg => "neg",
            Operation::Abs => "abs",
            Operation::Exp => "exp",
            Operation::Log => "log",
            Operation::Sqrt => "sqrt",
            Operation::Sin => "sin",
            Operation::Cos => "cos",
            Operation::Tanh => "tanh",
            Operation::Sigmoid => "sigmoid",
            Operation::Relu => "relu",
            Operation::Gelu => "gelu",
            Operation::Silu => "silu",
            Operation::Softmax { .. } => "softmax",
            Operation::LogSoftmax { .. } => "log_softmax",
            Operation::Add => "add",
            Operation::Sub => "sub",
            Operation::Mul => "mul",
            Operation::Div => "div",
            Operation::Pow => "pow",
            Operation::Maximum => "maximum",
            Operation::Minimum => "minimum",
            Operation::Sum { .. } => "sum",
            Operation::Mean { .. } => "mean",
            Operation::Max { .. } => "max",
            Operation::Min { .. } => "min",
            Operation::MatMul => "matmul",
            Operation::BatchMatMul => "batch_matmul",
            Operation::Reshape { .. } => "reshape",
            Operation::Transpose { .. } => "transpose",
            Operation::Squeeze { .. } => "squeeze",
            Operation::Unsqueeze { .. } => "unsqueeze",
            Operation::Slice { .. } => "slice",
            Operation::Concat { .. } => "concat",
            Operation::Conv2d(_) => "conv2d",
            Operation::Linear(_) => "linear",
            Operation::BatchNorm2d(_) => "batch_norm2d",
            Operation::Dropout { .. } => "dropout",
            Operation::MaxPool2d(_) => "max_pool2d",
            Operation::AvgPool2d(_) => "avg_pool2d",
            Operation::If(_) => "if",
            Operation::While(_) => "while",
            Operation::For(_) => "for",
            Operation::Break => "break",
            Operation::Continue => "continue",
            Operation::Return(_) => "return",
            Operation::Block(_) => "block",
            Operation::Merge(_) => "merge",
            Operation::Custom(name) => name,
            Operation::Split { .. } => "split",
            Operation::Gather { .. } => "gather",
            Operation::Scatter { .. } => "scatter",
            Operation::BatchNorm => "batch_norm",
            Operation::LayerNorm => "layer_norm",
            Operation::FusedKernel { name, .. } => name,
            Operation::CrossEntropy { .. } => "cross_entropy",
            Operation::MSELoss => "mse_loss",
            Operation::BCELoss => "bce_loss",
            Operation::Nop => "nop",
        }
    }

    /// Check if this is an element-wise operation
    pub fn is_elementwise(&self) -> bool {
        matches!(
            self,
            Operation::Neg
                | Operation::Abs
                | Operation::Exp
                | Operation::Log
                | Operation::Sqrt
                | Operation::Sin
                | Operation::Cos
                | Operation::Tanh
                | Operation::Sigmoid
                | Operation::Relu
                | Operation::Gelu
                | Operation::Silu
                | Operation::Add
                | Operation::Sub
                | Operation::Mul
                | Operation::Div
                | Operation::Pow
                | Operation::Maximum
                | Operation::Minimum
        )
    }

    /// Check if this is a reduction operation
    pub fn is_reduction(&self) -> bool {
        matches!(
            self,
            Operation::Sum { .. }
                | Operation::Mean { .. }
                | Operation::Max { .. }
                | Operation::Min { .. }
        )
    }

    /// Check if this operation modifies tensor shape
    pub fn modifies_shape(&self) -> bool {
        matches!(
            self,
            Operation::Reshape { .. }
                | Operation::Transpose { .. }
                | Operation::Squeeze { .. }
                | Operation::Unsqueeze { .. }
                | Operation::Slice { .. }
                | Operation::Concat { .. }
                | Operation::Split { .. }
        )
    }

    /// Check if this operation can be fused with others
    pub fn is_fusible(&self) -> bool {
        self.is_elementwise()
            || matches!(self, Operation::Relu | Operation::Tanh | Operation::Sigmoid)
    }

    /// Check if this operation requires gradients during training
    pub fn requires_gradients(&self) -> bool {
        !matches!(
            self,
            Operation::Constant(_) | Operation::Dropout { .. } | Operation::Nop
        )
    }

    /// Get the expected number of inputs for this operation
    pub fn expected_inputs(&self) -> usize {
        match self {
            Operation::Input | Operation::Parameter(_) | Operation::Constant(_) => 0,
            Operation::Neg
            | Operation::Abs
            | Operation::Exp
            | Operation::Log
            | Operation::Sqrt
            | Operation::Sin
            | Operation::Cos
            | Operation::Tanh
            | Operation::Sigmoid
            | Operation::Relu
            | Operation::Gelu
            | Operation::Silu
            | Operation::Softmax { .. }
            | Operation::LogSoftmax { .. }
            | Operation::Reshape { .. }
            | Operation::Transpose { .. }
            | Operation::Squeeze { .. }
            | Operation::Unsqueeze { .. }
            | Operation::Slice { .. }
            | Operation::Sum { .. }
            | Operation::Mean { .. }
            | Operation::Max { .. }
            | Operation::Min { .. }
            | Operation::Dropout { .. } => 1,
            Operation::Add
            | Operation::Sub
            | Operation::Mul
            | Operation::Div
            | Operation::Pow
            | Operation::Maximum
            | Operation::Minimum
            | Operation::MatMul
            | Operation::BatchMatMul => 2,
            Operation::Conv2d(_) => 2, // input + weight, bias is optional
            Operation::Linear(_) => 2, // input + weight, bias is optional
            Operation::BatchNorm2d(_) => 1, // input, parameters are stored separately
            Operation::MaxPool2d(_) | Operation::AvgPool2d(_) => 1, // pooling operations take one input
            Operation::Concat { .. } => usize::MAX,                 // Variable number of inputs
            Operation::Split { .. } => 1,
            Operation::Gather { .. } | Operation::Scatter { .. } => 2,
            Operation::If(_) => 3,    // condition + then_value + else_value
            Operation::While(_) => 2, // condition + body
            Operation::For(_) => 4,   // start + end + step + body
            Operation::Block(info) => info.operations.len(),
            Operation::Merge(info) => info.inputs.len(),
            Operation::FusedKernel { input_count, .. } => *input_count,
            Operation::CrossEntropy { .. } => 2, // predictions + targets
            Operation::MSELoss | Operation::BCELoss => 2, // predictions + targets
            Operation::Custom(_) => usize::MAX,  // Unknown
            Operation::BatchNorm | Operation::LayerNorm => 1,
            Operation::Break | Operation::Continue | Operation::Return(_) | Operation::Nop => 0,
        }
    }

    /// Get the expected number of outputs for this operation
    pub fn expected_outputs(&self) -> usize {
        match self {
            Operation::Split {
                split_size_or_sections,
                ..
            } => split_size_or_sections.len(),
            Operation::FusedKernel { output_count, .. } => *output_count,
            _ => 1, // Most operations produce one output
        }
    }
}
