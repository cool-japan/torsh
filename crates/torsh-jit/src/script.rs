//! Script mode for JIT compilation
//!
//! This module implements TorchScript-style script mode compilation,
//! allowing models to be exported and optimized without tracing.

use crate::{
    CompiledModule, ComputationGraph, JitCompiler, JitConfig, JitError, JitResult, Node, NodeId,
    ScriptableModule,
};
use petgraph::visit::EdgeRef;
use std::collections::HashMap;
use torsh_core::{DType, Shape};

/// Script compiler for converting modules to JIT-compiled form
pub struct ScriptCompiler {
    jit_compiler: JitCompiler,
    type_annotations: HashMap<String, TypeAnnotation>,
}

impl ScriptCompiler {
    /// Create a new script compiler
    pub fn new(config: JitConfig) -> Self {
        Self {
            jit_compiler: JitCompiler::new(config),
            type_annotations: HashMap::new(),
        }
    }

    /// Script a module into a compiled module
    pub fn script<M: ScriptableModule>(&mut self, module: M) -> JitResult<CompiledModule> {
        // Convert module to computation graph
        let graph = module.to_graph()?;

        // Apply type annotations if available
        let annotated_graph = self.apply_type_annotations(graph)?;

        // Compile the graph
        self.jit_compiler.compile(annotated_graph)
    }

    /// Add type annotation for a parameter or variable
    pub fn add_type_annotation(&mut self, name: String, annotation: TypeAnnotation) {
        self.type_annotations.insert(name, annotation);
    }

    /// Apply type annotations to the graph
    fn apply_type_annotations(&self, mut graph: ComputationGraph) -> JitResult<ComputationGraph> {
        // Apply annotations to nodes by name
        let node_ids: Vec<_> = graph.nodes().map(|(id, _)| id).collect();
        for node_id in node_ids {
            if let Some(node) = graph.node(node_id) {
                let node_name = node.name.clone();
                if let Some(annotation) = self.type_annotations.get(&node_name) {
                    if let Some(node_mut) = graph.node_mut(node_id) {
                        match annotation {
                            TypeAnnotation::Tensor { dtype, shape } => {
                                node_mut.dtype = *dtype;
                                node_mut.output_shape = Shape::new(shape.clone());
                            }
                            TypeAnnotation::Scalar(dtype) => {
                                node_mut.dtype = *dtype;
                                node_mut.output_shape = Shape::new(vec![1]);
                            }
                            TypeAnnotation::List { element_type, size } => {
                                // Handle list types via attributes
                                node_mut.attrs.insert(
                                    "list_element_type".to_string(),
                                    crate::graph::Attribute::String(format!("{:?}", element_type)),
                                );
                                node_mut.attrs.insert(
                                    "list_size".to_string(),
                                    crate::graph::Attribute::Int(*size as i64),
                                );
                            }
                        }
                    }
                }
            }
        }

        Ok(graph)
    }
}

/// Type annotation for script mode
#[derive(Debug, Clone)]
pub enum TypeAnnotation {
    /// Tensor with specific dtype and shape
    Tensor { dtype: DType, shape: Vec<usize> },
    /// Scalar value
    Scalar(DType),
    /// List of elements
    List {
        element_type: Box<TypeAnnotation>,
        size: usize,
    },
}

/// Script AST node for parsing script code
#[derive(Debug, Clone)]
pub enum ScriptAst {
    /// Function definition
    Function {
        name: String,
        params: Vec<Parameter>,
        return_type: Option<TypeAnnotation>,
        body: Box<ScriptAst>,
    },
    /// Variable declaration
    Let {
        name: String,
        type_ann: Option<TypeAnnotation>,
        value: Box<ScriptAst>,
    },
    /// Binary operation
    BinOp {
        op: BinaryOp,
        left: Box<ScriptAst>,
        right: Box<ScriptAst>,
    },
    /// Unary operation
    UnaryOp {
        op: UnaryOp,
        operand: Box<ScriptAst>,
    },
    /// Function call
    Call { func: String, args: Vec<ScriptAst> },
    /// Conditional
    If {
        condition: Box<ScriptAst>,
        then_branch: Box<ScriptAst>,
        else_branch: Option<Box<ScriptAst>>,
    },
    /// Loop
    For {
        var: String,
        iter: Box<ScriptAst>,
        body: Box<ScriptAst>,
    },
    /// Block of statements
    Block(Vec<ScriptAst>),
    /// Variable reference
    Var(String),
    /// Literal value
    Literal(LiteralValue),
    /// Return statement
    Return(Box<ScriptAst>),
}

/// Function parameter
#[derive(Debug, Clone)]
pub struct Parameter {
    pub name: String,
    pub type_ann: TypeAnnotation,
}

/// Binary operators
#[derive(Debug, Clone)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
}

/// Unary operators
#[derive(Debug, Clone)]
pub enum UnaryOp {
    Neg,
    Not,
}

/// Literal values
#[derive(Debug, Clone)]
pub enum LiteralValue {
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
}

/// Script parser
pub struct ScriptParser;

impl ScriptParser {
    /// Parse script code into AST
    pub fn parse(code: &str) -> JitResult<ScriptAst> {
        let mut parser = PythonParser::new(code);
        parser.parse()
    }
}

/// Python subset parser for JIT compilation
pub struct PythonParser {
    tokens: Vec<Token>,
    current: usize,
}

/// Token types for Python parsing
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // Literals
    Integer(i64),
    Float(f64),
    Boolean(bool),
    String(String),

    // Identifiers
    Identifier(String),

    // Keywords
    Def,
    If,
    Else,
    For,
    In,
    Return,
    True,
    False,

    // Operators
    Plus,
    Minus,
    Star,
    Slash,
    DoubleStar,
    Equal,
    EqualEqual,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    And,
    Or,
    Not,

    // Punctuation
    LeftParen,
    RightParen,
    LeftBracket,
    RightBracket,
    Comma,
    Colon,
    Arrow,

    // Special
    Newline,
    Indent,
    Dedent,
    Eof,
}

impl PythonParser {
    /// Create a new Python parser
    pub fn new(code: &str) -> Self {
        let tokens = Self::tokenize(code);
        Self { tokens, current: 0 }
    }

    /// Simple tokenizer for Python subset
    fn tokenize(code: &str) -> Vec<Token> {
        let mut tokens = Vec::new();
        let mut chars = code.chars().peekable();

        while let Some(&ch) = chars.peek() {
            match ch {
                ' ' | '\t' => {
                    chars.next();
                }
                '\n' => {
                    chars.next();
                    tokens.push(Token::Newline);
                }
                '(' => {
                    chars.next();
                    tokens.push(Token::LeftParen);
                }
                ')' => {
                    chars.next();
                    tokens.push(Token::RightParen);
                }
                '[' => {
                    chars.next();
                    tokens.push(Token::LeftBracket);
                }
                ']' => {
                    chars.next();
                    tokens.push(Token::RightBracket);
                }
                ',' => {
                    chars.next();
                    tokens.push(Token::Comma);
                }
                ':' => {
                    chars.next();
                    tokens.push(Token::Colon);
                }
                '+' => {
                    chars.next();
                    tokens.push(Token::Plus);
                }
                '-' => {
                    chars.next();
                    if chars.peek() == Some(&'>') {
                        chars.next();
                        tokens.push(Token::Arrow);
                    } else {
                        tokens.push(Token::Minus);
                    }
                }
                '*' => {
                    chars.next();
                    if chars.peek() == Some(&'*') {
                        chars.next();
                        tokens.push(Token::DoubleStar);
                    } else {
                        tokens.push(Token::Star);
                    }
                }
                '/' => {
                    chars.next();
                    tokens.push(Token::Slash);
                }
                '=' => {
                    chars.next();
                    if chars.peek() == Some(&'=') {
                        chars.next();
                        tokens.push(Token::EqualEqual);
                    } else {
                        tokens.push(Token::Equal);
                    }
                }
                '!' => {
                    chars.next();
                    if chars.peek() == Some(&'=') {
                        chars.next();
                        tokens.push(Token::NotEqual);
                    }
                }
                '<' => {
                    chars.next();
                    if chars.peek() == Some(&'=') {
                        chars.next();
                        tokens.push(Token::LessEqual);
                    } else {
                        tokens.push(Token::Less);
                    }
                }
                '>' => {
                    chars.next();
                    if chars.peek() == Some(&'=') {
                        chars.next();
                        tokens.push(Token::GreaterEqual);
                    } else {
                        tokens.push(Token::Greater);
                    }
                }
                '"' => {
                    chars.next();
                    let mut string_val = String::new();
                    while let Some(&ch) = chars.peek() {
                        if ch == '"' {
                            chars.next();
                            break;
                        }
                        string_val.push(ch);
                        chars.next();
                    }
                    tokens.push(Token::String(string_val));
                }
                c if c.is_ascii_digit() => {
                    let mut number = String::new();
                    let mut is_float = false;
                    while let Some(&ch) = chars.peek() {
                        if ch.is_ascii_digit() {
                            number.push(ch);
                            chars.next();
                        } else if ch == '.' && !is_float {
                            is_float = true;
                            number.push(ch);
                            chars.next();
                        } else {
                            break;
                        }
                    }

                    if is_float {
                        if let Ok(val) = number.parse::<f64>() {
                            tokens.push(Token::Float(val));
                        }
                    } else if let Ok(val) = number.parse::<i64>() {
                        tokens.push(Token::Integer(val));
                    }
                }
                c if c.is_ascii_alphabetic() || c == '_' => {
                    let mut ident = String::new();
                    while let Some(&ch) = chars.peek() {
                        if ch.is_ascii_alphanumeric() || ch == '_' {
                            ident.push(ch);
                            chars.next();
                        } else {
                            break;
                        }
                    }

                    let token = match ident.as_str() {
                        "def" => Token::Def,
                        "if" => Token::If,
                        "else" => Token::Else,
                        "for" => Token::For,
                        "in" => Token::In,
                        "return" => Token::Return,
                        "True" => Token::Boolean(true),
                        "False" => Token::Boolean(false),
                        "and" => Token::And,
                        "or" => Token::Or,
                        "not" => Token::Not,
                        _ => Token::Identifier(ident),
                    };
                    tokens.push(token);
                }
                _ => {
                    chars.next(); // Skip unknown characters
                }
            }
        }

        tokens.push(Token::Eof);
        tokens
    }

    /// Parse tokens into AST
    pub fn parse(&mut self) -> JitResult<ScriptAst> {
        let mut statements = Vec::new();

        while !self.is_at_end() {
            if self.match_token(&Token::Newline) {
                continue;
            }
            statements.push(self.parse_statement()?);
        }

        Ok(ScriptAst::Block(statements))
    }

    /// Parse a statement
    fn parse_statement(&mut self) -> JitResult<ScriptAst> {
        if self.match_token(&Token::Def) {
            self.parse_function()
        } else if self.match_token(&Token::Return) {
            let expr = self.parse_expression()?;
            Ok(ScriptAst::Return(Box::new(expr)))
        } else if self.match_token(&Token::If) {
            self.parse_if()
        } else if self.match_token(&Token::For) {
            self.parse_for()
        } else {
            // Assignment or expression statement
            let expr = self.parse_expression()?;
            if self.match_token(&Token::Equal) {
                if let ScriptAst::Var(name) = expr {
                    let value = self.parse_expression()?;
                    Ok(ScriptAst::Let {
                        name,
                        type_ann: None,
                        value: Box::new(value),
                    })
                } else {
                    Err(JitError::CompilationError(
                        "Invalid assignment target".to_string(),
                    ))
                }
            } else {
                Ok(expr)
            }
        }
    }

    /// Parse function definition
    fn parse_function(&mut self) -> JitResult<ScriptAst> {
        let name = if let Some(Token::Identifier(name)) = self.advance() {
            name.clone()
        } else {
            return Err(JitError::CompilationError(
                "Expected function name".to_string(),
            ));
        };

        self.consume(&Token::LeftParen, "Expected '(' after function name")?;

        let mut params = Vec::new();
        while !self.check(&Token::RightParen) && !self.is_at_end() {
            if let Some(Token::Identifier(param_name)) = self.advance() {
                // For now, default all parameters to float tensors
                params.push(Parameter {
                    name: param_name.clone(),
                    type_ann: TypeAnnotation::Tensor {
                        dtype: DType::F32,
                        shape: vec![], // Will be inferred
                    },
                });

                if !self.check(&Token::RightParen) {
                    self.consume(&Token::Comma, "Expected ',' between parameters")?;
                }
            }
        }

        self.consume(&Token::RightParen, "Expected ')' after parameters")?;
        self.consume(&Token::Colon, "Expected ':' after function signature")?;

        let body = self.parse_block()?;

        Ok(ScriptAst::Function {
            name,
            params,
            return_type: None,
            body: Box::new(body),
        })
    }

    /// Parse if statement
    fn parse_if(&mut self) -> JitResult<ScriptAst> {
        let condition = self.parse_expression()?;
        self.consume(&Token::Colon, "Expected ':' after if condition")?;

        let then_branch = self.parse_block()?;

        let else_branch = if self.match_token(&Token::Else) {
            self.consume(&Token::Colon, "Expected ':' after else")?;
            Some(Box::new(self.parse_block()?))
        } else {
            None
        };

        Ok(ScriptAst::If {
            condition: Box::new(condition),
            then_branch: Box::new(then_branch),
            else_branch,
        })
    }

    /// Parse for loop
    fn parse_for(&mut self) -> JitResult<ScriptAst> {
        let var = if let Some(Token::Identifier(name)) = self.advance() {
            name.clone()
        } else {
            return Err(JitError::CompilationError(
                "Expected variable name in for loop".to_string(),
            ));
        };

        self.consume(&Token::In, "Expected 'in' in for loop")?;
        let iter = self.parse_expression()?;
        self.consume(&Token::Colon, "Expected ':' after for loop header")?;

        let body = self.parse_block()?;

        Ok(ScriptAst::For {
            var,
            iter: Box::new(iter),
            body: Box::new(body),
        })
    }

    /// Parse a block of statements
    fn parse_block(&mut self) -> JitResult<ScriptAst> {
        let mut statements = Vec::new();

        // Simple block parsing - in a real implementation this would handle indentation
        while !self.is_at_end() && !self.check(&Token::Else) && !self.check(&Token::Def) {
            if self.match_token(&Token::Newline) {
                continue;
            }
            statements.push(self.parse_statement()?);
            break; // For simplicity, just parse one statement per block
        }

        Ok(ScriptAst::Block(statements))
    }

    /// Parse expression
    fn parse_expression(&mut self) -> JitResult<ScriptAst> {
        self.parse_or()
    }

    /// Parse logical OR
    fn parse_or(&mut self) -> JitResult<ScriptAst> {
        let mut expr = self.parse_and()?;

        while self.match_token(&Token::Or) {
            let right = self.parse_and()?;
            expr = ScriptAst::BinOp {
                op: BinaryOp::Or,
                left: Box::new(expr),
                right: Box::new(right),
            };
        }

        Ok(expr)
    }

    /// Parse logical AND
    fn parse_and(&mut self) -> JitResult<ScriptAst> {
        let mut expr = self.parse_equality()?;

        while self.match_token(&Token::And) {
            let right = self.parse_equality()?;
            expr = ScriptAst::BinOp {
                op: BinaryOp::And,
                left: Box::new(expr),
                right: Box::new(right),
            };
        }

        Ok(expr)
    }

    /// Parse equality operations
    fn parse_equality(&mut self) -> JitResult<ScriptAst> {
        let mut expr = self.parse_comparison()?;

        while let Some(op) = self.match_equality_op() {
            let right = self.parse_comparison()?;
            expr = ScriptAst::BinOp {
                op,
                left: Box::new(expr),
                right: Box::new(right),
            };
        }

        Ok(expr)
    }

    /// Parse comparison operations
    fn parse_comparison(&mut self) -> JitResult<ScriptAst> {
        let mut expr = self.parse_term()?;

        while let Some(op) = self.match_comparison_op() {
            let right = self.parse_term()?;
            expr = ScriptAst::BinOp {
                op,
                left: Box::new(expr),
                right: Box::new(right),
            };
        }

        Ok(expr)
    }

    /// Parse addition and subtraction
    fn parse_term(&mut self) -> JitResult<ScriptAst> {
        let mut expr = self.parse_factor()?;

        while self.check(&Token::Plus) || self.check(&Token::Minus) {
            let op = if self.match_token(&Token::Plus) {
                BinaryOp::Add
            } else {
                self.advance();
                BinaryOp::Sub
            };

            let right = self.parse_factor()?;
            expr = ScriptAst::BinOp {
                op,
                left: Box::new(expr),
                right: Box::new(right),
            };
        }

        Ok(expr)
    }

    /// Parse multiplication, division, and power
    fn parse_factor(&mut self) -> JitResult<ScriptAst> {
        let mut expr = self.parse_unary()?;

        while self.check(&Token::Star)
            || self.check(&Token::Slash)
            || self.check(&Token::DoubleStar)
        {
            let op = if self.match_token(&Token::Star) {
                BinaryOp::Mul
            } else if self.match_token(&Token::Slash) {
                BinaryOp::Div
            } else {
                self.advance();
                BinaryOp::Pow
            };

            let right = self.parse_unary()?;
            expr = ScriptAst::BinOp {
                op,
                left: Box::new(expr),
                right: Box::new(right),
            };
        }

        Ok(expr)
    }

    /// Parse unary operations
    fn parse_unary(&mut self) -> JitResult<ScriptAst> {
        if self.match_token(&Token::Not) {
            let operand = self.parse_unary()?;
            Ok(ScriptAst::UnaryOp {
                op: UnaryOp::Not,
                operand: Box::new(operand),
            })
        } else if self.match_token(&Token::Minus) {
            let operand = self.parse_unary()?;
            Ok(ScriptAst::UnaryOp {
                op: UnaryOp::Neg,
                operand: Box::new(operand),
            })
        } else {
            self.parse_call()
        }
    }

    /// Parse function calls
    fn parse_call(&mut self) -> JitResult<ScriptAst> {
        let mut expr = self.parse_primary()?;

        while self.match_token(&Token::LeftParen) {
            let mut args = Vec::new();
            while !self.check(&Token::RightParen) && !self.is_at_end() {
                args.push(self.parse_expression()?);
                if !self.check(&Token::RightParen) {
                    self.consume(&Token::Comma, "Expected ',' between arguments")?;
                }
            }
            self.consume(&Token::RightParen, "Expected ')' after arguments")?;

            if let ScriptAst::Var(func_name) = expr {
                expr = ScriptAst::Call {
                    func: func_name,
                    args,
                };
            }
        }

        Ok(expr)
    }

    /// Parse primary expressions
    fn parse_primary(&mut self) -> JitResult<ScriptAst> {
        if let Some(token) = self.advance() {
            match token {
                Token::Integer(val) => Ok(ScriptAst::Literal(LiteralValue::Int(*val))),
                Token::Float(val) => Ok(ScriptAst::Literal(LiteralValue::Float(*val))),
                Token::Boolean(val) => Ok(ScriptAst::Literal(LiteralValue::Bool(*val))),
                Token::String(val) => Ok(ScriptAst::Literal(LiteralValue::String(val.clone()))),
                Token::Identifier(name) => Ok(ScriptAst::Var(name.clone())),
                Token::LeftParen => {
                    let expr = self.parse_expression()?;
                    self.consume(&Token::RightParen, "Expected ')' after expression")?;
                    Ok(expr)
                }
                _ => Err(JitError::CompilationError(
                    "Unexpected token in expression".to_string(),
                )),
            }
        } else {
            Err(JitError::CompilationError(
                "Unexpected end of input".to_string(),
            ))
        }
    }

    /// Match equality operators
    fn match_equality_op(&mut self) -> Option<BinaryOp> {
        if self.match_token(&Token::EqualEqual) {
            Some(BinaryOp::Eq)
        } else if self.match_token(&Token::NotEqual) {
            Some(BinaryOp::Ne)
        } else {
            None
        }
    }

    /// Match comparison operators
    fn match_comparison_op(&mut self) -> Option<BinaryOp> {
        if self.match_token(&Token::Greater) {
            Some(BinaryOp::Gt)
        } else if self.match_token(&Token::GreaterEqual) {
            Some(BinaryOp::Ge)
        } else if self.match_token(&Token::Less) {
            Some(BinaryOp::Lt)
        } else if self.match_token(&Token::LessEqual) {
            Some(BinaryOp::Le)
        } else {
            None
        }
    }

    /// Helper methods for parsing
    fn match_token(&mut self, expected: &Token) -> bool {
        if self.check(expected) {
            self.advance();
            true
        } else {
            false
        }
    }

    fn check(&self, expected: &Token) -> bool {
        if self.is_at_end() {
            false
        } else {
            std::mem::discriminant(&self.tokens[self.current]) == std::mem::discriminant(expected)
        }
    }

    fn advance(&mut self) -> Option<&Token> {
        if !self.is_at_end() {
            self.current += 1;
        }
        self.previous()
    }

    fn is_at_end(&self) -> bool {
        self.current >= self.tokens.len()
            || matches!(self.tokens.get(self.current), Some(Token::Eof))
    }

    fn previous(&self) -> Option<&Token> {
        self.tokens.get(self.current.saturating_sub(1))
    }

    fn consume(&mut self, expected: &Token, message: &str) -> JitResult<()> {
        if self.check(expected) {
            self.advance();
            Ok(())
        } else {
            Err(JitError::CompilationError(message.to_string()))
        }
    }
}

/// Convert script AST to computation graph
pub struct AstToGraphConverter {
    graph: ComputationGraph,
    var_map: HashMap<String, NodeId>,
    next_id: usize,
}

impl Default for AstToGraphConverter {
    fn default() -> Self {
        Self::new()
    }
}

impl AstToGraphConverter {
    /// Create a new converter
    pub fn new() -> Self {
        Self {
            graph: ComputationGraph::new(),
            var_map: HashMap::new(),
            next_id: 0,
        }
    }

    /// Convert AST to computation graph
    pub fn convert(&mut self, ast: ScriptAst) -> JitResult<ComputationGraph> {
        self.convert_ast(ast)?;
        Ok(self.graph.clone())
    }

    /// Convert an AST node
    fn convert_ast(&mut self, ast: ScriptAst) -> JitResult<NodeId> {
        match ast {
            ScriptAst::BinOp { op, left, right } => {
                let left_id = self.convert_ast(*left)?;
                let right_id = self.convert_ast(*right)?;
                self.create_binop_node(op, left_id, right_id)
            }
            ScriptAst::UnaryOp { op, operand } => {
                let operand_id = self.convert_ast(*operand)?;
                self.create_unaryop_node(op, operand_id)
            }
            ScriptAst::Call { func, args } => {
                let arg_ids: Vec<_> = args
                    .into_iter()
                    .map(|arg| self.convert_ast(arg))
                    .collect::<JitResult<Vec<_>>>()?;
                self.create_call_node(func, arg_ids)
            }
            ScriptAst::Var(name) => self
                .var_map
                .get(&name)
                .copied()
                .ok_or_else(|| JitError::GraphError(format!("Undefined variable: {}", name))),
            ScriptAst::Literal(lit) => self.create_literal_node(lit),
            ScriptAst::Let { name, value, .. } => {
                let value_id = self.convert_ast(*value)?;
                self.var_map.insert(name, value_id);
                Ok(value_id)
            }
            ScriptAst::Block(stmts) => {
                let mut last_id = None;
                for stmt in stmts {
                    last_id = Some(self.convert_ast(stmt)?);
                }
                last_id.ok_or_else(|| JitError::GraphError("Empty block".to_string()))
            }
            _ => Err(JitError::GraphError("Unsupported AST node".to_string())),
        }
    }

    /// Create a binary operation node
    fn create_binop_node(
        &mut self,
        op: BinaryOp,
        left: NodeId,
        right: NodeId,
    ) -> JitResult<NodeId> {
        use crate::graph::{Edge, Operation};
        use torsh_core::DeviceType;

        let operation = match op {
            BinaryOp::Add => Operation::Add,
            BinaryOp::Sub => Operation::Sub,
            BinaryOp::Mul => Operation::Mul,
            BinaryOp::Div => Operation::Div,
            _ => return Err(JitError::UnsupportedOp(format!("{:?}", op))),
        };

        let mut node = Node::new(operation, format!("binop_{}", self.next_id));
        node.device = DeviceType::Cpu;
        node.inputs = vec![];
        node.is_output = false;

        let node_id = self.graph.add_node(node);
        self.graph.add_edge(left, node_id, Edge::default());
        self.graph.add_edge(right, node_id, Edge::default());
        self.next_id += 1;
        Ok(node_id)
    }

    /// Create a unary operation node
    fn create_unaryop_node(&mut self, op: UnaryOp, operand: NodeId) -> JitResult<NodeId> {
        use crate::graph::{Edge, Operation};
        use torsh_core::DeviceType;

        let operation = match op {
            UnaryOp::Neg => Operation::Neg,
            _ => return Err(JitError::UnsupportedOp(format!("{:?}", op))),
        };

        let mut node = Node::new(operation, format!("unaryop_{}", self.next_id));
        node.device = DeviceType::Cpu;
        node.inputs = vec![];
        node.is_output = false;

        let node_id = self.graph.add_node(node);
        self.graph.add_edge(operand, node_id, Edge::default());
        self.next_id += 1;
        Ok(node_id)
    }

    /// Create a function call node
    fn create_call_node(&mut self, func: String, args: Vec<NodeId>) -> JitResult<NodeId> {
        use crate::graph::{Edge, Operation};
        use torsh_core::DeviceType;

        let operation = match func.as_str() {
            "relu" => Operation::Relu,
            "sigmoid" => Operation::Sigmoid,
            "tanh" => Operation::Tanh,
            "matmul" => Operation::MatMul,
            _ => Operation::Custom(func),
        };

        let mut node = Node::new(operation, format!("call_{}", self.next_id));
        node.device = DeviceType::Cpu;
        node.inputs = vec![];
        node.is_output = false;

        let node_id = self.graph.add_node(node);
        for (i, arg_id) in args.iter().enumerate() {
            let edge = Edge {
                src_output: 0,
                dst_input: i,
            };
            self.graph.add_edge(*arg_id, node_id, edge);
        }
        self.next_id += 1;
        Ok(node_id)
    }

    /// Create a literal node
    fn create_literal_node(&mut self, lit: LiteralValue) -> JitResult<NodeId> {
        use crate::graph::{Attribute, ConstantInfo, ConstantValue, Operation};
        use torsh_core::DeviceType;

        let (dtype, constant_value) = match lit {
            LiteralValue::Int(v) => (DType::I64, ConstantValue::IntScalar(v)),
            LiteralValue::Float(v) => (DType::F32, ConstantValue::Scalar(v)),
            LiteralValue::Bool(v) => (DType::Bool, ConstantValue::IntScalar(if v { 1 } else { 0 })),
            LiteralValue::String(v) => {
                // String literals need special handling
                let mut node = Node::new(
                    Operation::Custom("string_literal".to_string()),
                    format!("string_literal_{}", self.next_id),
                );
                node.device = DeviceType::Cpu;
                node.attrs.insert("value".to_string(), Attribute::String(v));
                node.inputs = vec![];
                node.is_output = false;
                let node_id = self.graph.add_node(node);
                self.next_id += 1;
                return Ok(node_id);
            }
        };

        let mut node = Node::new(
            Operation::Constant(ConstantInfo {
                value: constant_value,
            }),
            format!("constant_{}", self.next_id),
        );
        node.device = DeviceType::Cpu;
        node.inputs = vec![];
        node.is_output = false;

        let node_id = self.graph.add_node(node);
        self.next_id += 1;
        Ok(node_id)
    }
}

/// Export a compiled module to TorchScript format
pub fn export_torchscript(module: &CompiledModule, path: &str) -> JitResult<()> {
    use std::fs::File;
    use std::io::Write;

    // Create TorchScript representation
    let ts_repr = TorchScriptModule {
        version: 1,
        graph: module.graph.clone(),
        constants: extract_constants_from_graph(&module.graph),
        metadata: create_metadata_from_module(module),
    };

    // Convert to TorchScript IR format
    let torchscript_ir = generate_torchscript_ir(&ts_repr)?;

    // Write to file
    let mut file = File::create(path)
        .map_err(|e| JitError::RuntimeError(format!("Failed to create file {}: {}", path, e)))?;

    file.write_all(torchscript_ir.as_bytes())
        .map_err(|e| JitError::RuntimeError(format!("Failed to write file {}: {}", path, e)))?;

    Ok(())
}

/// Import a module from TorchScript format
pub fn import_torchscript(path: &str, config: JitConfig) -> JitResult<CompiledModule> {
    use std::fs::File;
    use std::io::Read;

    // Read file
    let mut file = File::open(path)
        .map_err(|e| JitError::RuntimeError(format!("Failed to open file {}: {}", path, e)))?;

    let mut contents = String::new();
    file.read_to_string(&mut contents)
        .map_err(|e| JitError::RuntimeError(format!("Failed to read file {}: {}", path, e)))?;

    // Parse TorchScript IR
    let ts_module = parse_torchscript_ir(&contents)?;

    // Convert to our internal representation
    let mut jit_compiler = JitCompiler::new(config);
    let compiled_module = jit_compiler.compile(ts_module.graph)?;

    Ok(compiled_module)
}

/// TorchScript module representation for serialization
#[derive(Debug, Clone)]
struct TorchScriptModule {
    version: u32,
    graph: ComputationGraph,
    constants: HashMap<String, Vec<f32>>,
    metadata: HashMap<String, String>,
}

/// Extract constants from computation graph
fn extract_constants_from_graph(graph: &ComputationGraph) -> HashMap<String, Vec<f32>> {
    use crate::graph::{ConstantValue, Operation};

    let mut constants = HashMap::new();

    for (node_id, node) in graph.nodes() {
        if let Operation::Constant(ref const_info) = node.op {
            let const_name = format!("const_{:?}", node_id);
            match &const_info.value {
                ConstantValue::Scalar(val) => {
                    constants.insert(const_name, vec![*val as f32]);
                }
                ConstantValue::IntScalar(val) => {
                    constants.insert(const_name, vec![*val as f32]);
                }
                ConstantValue::Tensor {
                    shape: _,
                    data,
                    dtype: _,
                } => {
                    constants.insert(const_name, data.iter().map(|&x| x as f32).collect());
                }
                ConstantValue::Bool(val) => {
                    constants.insert(const_name, vec![if *val { 1.0 } else { 0.0 }]);
                }
                ConstantValue::Int(val) => {
                    constants.insert(const_name, vec![*val as f32]);
                }
                ConstantValue::UInt(val) => {
                    constants.insert(const_name, vec![*val as f32]);
                }
                ConstantValue::Float(val) => {
                    constants.insert(const_name, vec![*val as f32]);
                }
                ConstantValue::String(_) => {
                    constants.insert(const_name, vec![0.0]); // String as placeholder
                }
                ConstantValue::FloatArray(arr) => {
                    constants.insert(const_name, arr.clone());
                }
                ConstantValue::IntArray(arr) => {
                    constants.insert(const_name, arr.iter().map(|&x| x as f32).collect());
                }
                ConstantValue::Array(arr) => {
                    // Convert array of values to f32 - simplified
                    constants.insert(const_name, vec![arr.len() as f32]);
                }
                ConstantValue::Complex { real, imag: _ } => {
                    constants.insert(const_name, vec![*real as f32]);
                }
                ConstantValue::None => {
                    constants.insert(const_name, vec![0.0]);
                }
                ConstantValue::Undefined => {
                    constants.insert(const_name, vec![0.0]);
                }
            }
        }
    }

    constants
}

/// Create metadata from compiled module
fn create_metadata_from_module(module: &CompiledModule) -> HashMap<String, String> {
    let mut metadata = HashMap::new();

    metadata.insert("producer".to_string(), "torsh-jit".to_string());
    metadata.insert("producer_version".to_string(), "0.1.0".to_string());
    metadata.insert("graph_name".to_string(), "main".to_string());
    metadata.insert(
        "node_count".to_string(),
        module.graph.node_count().to_string(),
    );
    metadata.insert(
        "edge_count".to_string(),
        module.graph.edge_count().to_string(),
    );

    metadata
}

/// Generate TorchScript IR from TorchScript module
fn generate_torchscript_ir(ts_module: &TorchScriptModule) -> JitResult<String> {
    use crate::graph::{ConstantValue, Operation};

    let mut ir = String::new();

    // Header
    ir.push_str(&format!("graph():\n"));

    // Constants section
    for (name, values) in &ts_module.constants {
        ir.push_str(&format!(
            "  %{} : Float({}) = prim::Constant[value={}]()\n",
            name,
            values.len(),
            format_tensor_values(values)
        ));
    }

    // Nodes section
    let mut output_counter = 0;
    for (node_id, node) in ts_module.graph.nodes() {
        match &node.op {
            Operation::Add => {
                let inputs = get_node_inputs(&ts_module.graph, node_id);
                ir.push_str(&format!(
                    "  %{} : Float = aten::add({}, {})\n",
                    output_counter, inputs[0], inputs[1]
                ));
            }
            Operation::Mul => {
                let inputs = get_node_inputs(&ts_module.graph, node_id);
                ir.push_str(&format!(
                    "  %{} : Float = aten::mul({}, {})\n",
                    output_counter, inputs[0], inputs[1]
                ));
            }
            Operation::MatMul => {
                let inputs = get_node_inputs(&ts_module.graph, node_id);
                ir.push_str(&format!(
                    "  %{} : Float = aten::mm({}, {})\n",
                    output_counter, inputs[0], inputs[1]
                ));
            }
            Operation::Relu => {
                let inputs = get_node_inputs(&ts_module.graph, node_id);
                ir.push_str(&format!(
                    "  %{} : Float = aten::relu({})\n",
                    output_counter, inputs[0]
                ));
            }
            Operation::Sigmoid => {
                let inputs = get_node_inputs(&ts_module.graph, node_id);
                ir.push_str(&format!(
                    "  %{} : Float = aten::sigmoid({})\n",
                    output_counter, inputs[0]
                ));
            }
            Operation::Constant(const_info) => match &const_info.value {
                ConstantValue::Scalar(val) => {
                    ir.push_str(&format!(
                        "  %{} : Float = prim::Constant[value={}]()\n",
                        output_counter, val
                    ));
                }
                ConstantValue::IntScalar(val) => {
                    ir.push_str(&format!(
                        "  %{} : int = prim::Constant[value={}]()\n",
                        output_counter, val
                    ));
                }
                ConstantValue::Tensor {
                    shape: _,
                    data,
                    dtype: _,
                } => {
                    let data_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();
                    ir.push_str(&format!(
                        "  %{} : Float = prim::Constant[value={}]()\n",
                        output_counter,
                        format_tensor_values(&data_f32)
                    ));
                }
                ConstantValue::Bool(val) => {
                    ir.push_str(&format!(
                        "  %{} : bool = prim::Constant[value={}]()\n",
                        output_counter, val
                    ));
                }
                ConstantValue::Int(val) => {
                    ir.push_str(&format!(
                        "  %{} : int = prim::Constant[value={}]()\n",
                        output_counter, val
                    ));
                }
                ConstantValue::UInt(val) => {
                    ir.push_str(&format!(
                        "  %{} : int = prim::Constant[value={}]()\n",
                        output_counter, val
                    ));
                }
                ConstantValue::Float(val) => {
                    ir.push_str(&format!(
                        "  %{} : Float = prim::Constant[value={}]()\n",
                        output_counter, val
                    ));
                }
                ConstantValue::String(val) => {
                    ir.push_str(&format!(
                        "  %{} : str = prim::Constant[value=\"{}\"]()\n",
                        output_counter, val
                    ));
                }
                ConstantValue::FloatArray(arr) => {
                    ir.push_str(&format!(
                        "  %{} : Float[] = prim::Constant[value={}]()\n",
                        output_counter,
                        format_tensor_values(arr)
                    ));
                }
                ConstantValue::IntArray(arr) => {
                    let arr_str = arr
                        .iter()
                        .map(|x| x.to_string())
                        .collect::<Vec<_>>()
                        .join(", ");
                    ir.push_str(&format!(
                        "  %{} : int[] = prim::Constant[value=[{}]]()\n",
                        output_counter, arr_str
                    ));
                }
                ConstantValue::Array(_) => {
                    ir.push_str(&format!(
                        "  %{} : Tensor = prim::Constant[value=<complex_array>]()\n",
                        output_counter
                    ));
                }
                ConstantValue::Complex { real, imag } => {
                    ir.push_str(&format!(
                        "  %{} : complex = prim::Constant[value={}+{}i]()\n",
                        output_counter, real, imag
                    ));
                }
                ConstantValue::None => {
                    ir.push_str(&format!(
                        "  %{} : NoneType = prim::Constant[value=None]()\n",
                        output_counter
                    ));
                }
                ConstantValue::Undefined => {
                    ir.push_str(&format!(
                        "  %{} : Tensor = prim::Constant[value=<undefined>]()\n",
                        output_counter
                    ));
                }
            },
            Operation::Custom(name) => {
                let inputs = get_node_inputs(&ts_module.graph, node_id);
                let input_str = inputs.join(", ");
                ir.push_str(&format!(
                    "  %{} : Float = custom::{}({})\n",
                    output_counter, name, input_str
                ));
            }
            _ => {
                // Generic operation handling
                let inputs = get_node_inputs(&ts_module.graph, node_id);
                let input_str = inputs.join(", ");
                ir.push_str(&format!(
                    "  %{} : Float = aten::{:?}({})\n",
                    output_counter, node.op, input_str
                ));
            }
        }
        output_counter += 1;
    }

    // Return the last output
    if output_counter > 0 {
        ir.push_str(&format!("  return (%{})\n", output_counter - 1));
    } else {
        ir.push_str("  return ()\n");
    }

    Ok(ir)
}

/// Parse TorchScript IR into TorchScript module
fn parse_torchscript_ir(ir: &str) -> JitResult<TorchScriptModule> {
    let mut graph = ComputationGraph::new();
    let mut constants = HashMap::new();
    let mut metadata = HashMap::new();

    // Simple line-based parser for TorchScript IR
    let lines: Vec<&str> = ir.lines().collect();
    let mut node_counter = 0;

    for line in lines {
        let line = line.trim();

        if line.starts_with('%') && line.contains("prim::Constant") {
            // Parse constant
            if let Some(value_start) = line.find("value=") {
                let value_part = &line[value_start + 6..];
                if let Some(value_end) = value_part.find(']') {
                    let value_str = &value_part[..value_end];
                    if let Ok(val) = value_str.parse::<f32>() {
                        let const_name = format!("const_{}", node_counter);
                        constants.insert(const_name, vec![val]);

                        // Add constant node to graph
                        add_constant_node_to_graph(&mut graph, val, node_counter);
                        node_counter += 1;
                    }
                }
            }
        } else if line.starts_with('%') && line.contains("aten::") {
            // Parse operation
            parse_aten_operation(&mut graph, line, node_counter)?;
            node_counter += 1;
        }
    }

    // Add default metadata
    metadata.insert("producer".to_string(), "torchscript".to_string());
    metadata.insert("version".to_string(), "1.0".to_string());

    Ok(TorchScriptModule {
        version: 1,
        graph,
        constants,
        metadata,
    })
}

/// Helper function to format tensor values for TorchScript IR
fn format_tensor_values(values: &[f32]) -> String {
    if values.len() == 1 {
        values[0].to_string()
    } else {
        format!(
            "[{}]",
            values
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

/// Helper function to get node inputs as strings
fn get_node_inputs(graph: &ComputationGraph, node_id: NodeId) -> Vec<String> {
    let mut inputs = Vec::new();

    for edge in graph.edges_directed(node_id, petgraph::Direction::Incoming) {
        let src_id = edge.source();
        inputs.push(format!("%{:?}", src_id));
    }

    // If no inputs, assume it's an input node
    if inputs.is_empty() {
        inputs.push(format!("%input_{:?}", node_id));
    }

    inputs
}

/// Add constant node to computation graph
fn add_constant_node_to_graph(graph: &mut ComputationGraph, value: f32, node_id: usize) {
    use crate::graph::{ConstantInfo, ConstantValue, Operation};
    use torsh_core::DeviceType;

    let mut node = Node::new(
        Operation::Constant(ConstantInfo {
            value: ConstantValue::Scalar(value as f64),
        }),
        format!("const_{}", node_id),
    );
    node = node
        .with_output_shapes(vec![Some(Shape::new(vec![1]))])
        .with_dtypes(vec![DType::F32])
        .with_device(DeviceType::Cpu);
    node.inputs = vec![];
    node.is_output = false;

    graph.add_node(node);
}

/// Parse aten operation from TorchScript IR line
fn parse_aten_operation(graph: &mut ComputationGraph, line: &str, node_id: usize) -> JitResult<()> {
    use crate::graph::Operation;
    use torsh_core::DeviceType;

    let operation = if line.contains("aten::add") {
        Operation::Add
    } else if line.contains("aten::mul") {
        Operation::Mul
    } else if line.contains("aten::mm") {
        Operation::MatMul
    } else if line.contains("aten::relu") {
        Operation::Relu
    } else if line.contains("aten::sigmoid") {
        Operation::Sigmoid
    } else {
        // Extract operation name
        if let Some(op_start) = line.find("aten::") {
            let op_part = &line[op_start + 6..];
            if let Some(op_end) = op_part.find('(') {
                let op_name = &op_part[..op_end];
                Operation::Custom(op_name.to_string())
            } else {
                Operation::Custom("unknown".to_string())
            }
        } else {
            Operation::Custom("unknown".to_string())
        }
    };

    let mut node = Node::new(operation, format!("op_{}", node_id));
    node = node
        .with_output_shapes(vec![Some(Shape::new(vec![]))]) // Will be inferred
        .with_dtypes(vec![DType::F32])
        .with_device(DeviceType::Cpu);
    node.inputs = vec![];
    node.is_output = false;

    graph.add_node(node);
    Ok(())
}

/// Implementation of script function
pub fn script<M: ScriptableModule>(module: M) -> JitResult<CompiledModule> {
    let config = JitConfig::default();
    let mut compiler = ScriptCompiler::new(config);
    compiler.script(module)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_annotation() {
        let tensor_ann = TypeAnnotation::Tensor {
            dtype: DType::F32,
            shape: vec![10, 20],
        };

        match tensor_ann {
            TypeAnnotation::Tensor { dtype, shape } => {
                assert_eq!(dtype, DType::F32);
                assert_eq!(shape, vec![10, 20]);
            }
            _ => panic!("Wrong type annotation"),
        }
    }

    #[test]
    fn test_ast_to_graph_converter() {
        let mut converter = AstToGraphConverter::new();

        // Test literal conversion
        let lit_ast = ScriptAst::Literal(LiteralValue::Float(3.14));
        let result = converter.convert(lit_ast);
        assert!(result.is_ok());
    }

    #[test]
    fn test_script_compiler_creation() {
        let config = JitConfig::default();
        let compiler = ScriptCompiler::new(config);
        assert!(compiler.type_annotations.is_empty());
    }
}
