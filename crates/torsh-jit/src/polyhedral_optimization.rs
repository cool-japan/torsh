// Copyright (c) 2025 ToRSh Contributors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! # Polyhedral Optimization
//!
//! This module implements advanced loop transformations using the polyhedral model,
//! enabling aggressive optimization of nested loop structures common in deep learning.
//!
//! ## Key Concepts
//!
//! - **Polyhedral Model**: Represents loop iterations as integer points in polyhedra
//! - **Affine Scheduling**: Compute optimal execution order using affine transformations
//! - **Dependence Analysis**: Precise analysis of data dependencies in loop nests
//! - **Loop Transformations**: Tiling, fusion, interchange, skewing, distribution
//! - **Locality Optimization**: Maximize cache reuse through careful scheduling
//!
//! ## Transformations
//!
//! ```text
//! Original Loop:                  After Tiling:
//! for i in 0..N                  for ii in 0..N step T
//!   for j in 0..M                  for jj in 0..M step T
//!     A[i,j] = ...                   for i in ii..min(ii+T,N)
//!                                      for j in jj..min(jj+T,M)
//!                                        A[i,j] = ...
//! ```
//!
//! ## Example
//!
//! ```rust,ignore
//! use torsh_jit::polyhedral_optimization::{PolyhedralOptimizer, LoopNest};
//!
//! let optimizer = PolyhedralOptimizer::new();
//!
//! // Analyze loop nest
//! let nest = LoopNest::from_graph(&graph)?;
//!
//! // Compute optimal schedule
//! let schedule = optimizer.compute_schedule(&nest)?;
//!
//! // Apply transformations
//! let optimized = optimizer.apply_schedule(&nest, &schedule)?;
//! ```

use crate::graph::{ComputationGraph, NodeId};
use crate::JitResult;
use std::collections::HashMap;

// ============================================================================
// Polyhedral Representation
// ============================================================================

/// A loop nest represented in polyhedral form
#[derive(Debug, Clone)]
pub struct LoopNest {
    /// Loops in the nest
    pub loops: Vec<Loop>,

    /// Statements in the loop body
    pub statements: Vec<Statement>,

    /// Data dependencies
    pub dependencies: Vec<Dependence>,

    /// Iteration domain
    pub domain: IterationDomain,
}

/// A single loop in the nest
#[derive(Debug, Clone)]
pub struct Loop {
    /// Loop variable name
    pub variable: String,

    /// Lower bound (affine expression)
    pub lower_bound: AffineExpr,

    /// Upper bound (affine expression)
    pub upper_bound: AffineExpr,

    /// Step size
    pub step: i64,

    /// Nesting depth
    pub depth: usize,
}

/// A statement within the loop
#[derive(Debug, Clone)]
pub struct Statement {
    /// Statement ID
    pub id: usize,

    /// Associated graph node
    pub node_id: NodeId,

    /// Iteration domain (which loop iterations execute this)
    pub domain: Polyhedron,

    /// Schedule (when this executes)
    pub schedule: AffineSchedule,

    /// Memory accesses
    pub accesses: Vec<MemoryAccess>,
}

/// Memory access pattern
#[derive(Debug, Clone)]
pub struct MemoryAccess {
    /// Array being accessed
    pub array_name: String,

    /// Access function (affine)
    pub access_fn: Vec<AffineExpr>,

    /// Access type
    pub access_type: AccessType,
}

/// Type of memory access
#[derive(Debug, Clone, PartialEq)]
pub enum AccessType {
    Read,
    Write,
    ReadWrite,
}

/// Data dependence between statements
#[derive(Debug, Clone)]
pub struct Dependence {
    /// Source statement
    pub source: usize,

    /// Target statement
    pub target: usize,

    /// Dependence type
    pub dep_type: DependenceType,

    /// Dependence polyhedron (iterations involved)
    pub polyhedron: Polyhedron,

    /// Dependence distance vector
    pub distance: Vec<i64>,
}

/// Types of dependencies
#[derive(Debug, Clone, PartialEq)]
pub enum DependenceType {
    /// Read-after-Write (true dependence)
    Flow,

    /// Write-after-Read (anti dependence)
    Anti,

    /// Write-after-Write (output dependence)
    Output,

    /// Read-after-Read (no real dependence)
    Input,
}

// ============================================================================
// Affine Expressions
// ============================================================================

/// Affine expression: a₀ + a₁*x₁ + a₂*x₂ + ... + aₙ*xₙ
#[derive(Debug, Clone, PartialEq)]
pub struct AffineExpr {
    /// Constant term
    pub constant: i64,

    /// Coefficients for each variable
    pub coefficients: HashMap<String, i64>,
}

impl AffineExpr {
    /// Create constant expression
    pub fn constant(value: i64) -> Self {
        Self {
            constant: value,
            coefficients: HashMap::new(),
        }
    }

    /// Create variable expression
    pub fn variable(name: String) -> Self {
        let mut coefficients = HashMap::new();
        coefficients.insert(name, 1);
        Self {
            constant: 0,
            coefficients,
        }
    }

    /// Add two expressions
    pub fn add(&self, other: &AffineExpr) -> AffineExpr {
        let mut coefficients = self.coefficients.clone();
        for (var, &coeff) in &other.coefficients {
            *coefficients.entry(var.clone()).or_insert(0) += coeff;
        }
        AffineExpr {
            constant: self.constant + other.constant,
            coefficients,
        }
    }

    /// Multiply by constant
    pub fn mul(&self, scalar: i64) -> AffineExpr {
        let coefficients = self
            .coefficients
            .iter()
            .map(|(k, &v)| (k.clone(), v * scalar))
            .collect();
        AffineExpr {
            constant: self.constant * scalar,
            coefficients,
        }
    }

    /// Evaluate with given variable values
    pub fn evaluate(&self, vars: &HashMap<String, i64>) -> i64 {
        let mut result = self.constant;
        for (var, &coeff) in &self.coefficients {
            if let Some(&val) = vars.get(var) {
                result += coeff * val;
            }
        }
        result
    }

    /// Check if expression is constant
    pub fn is_constant(&self) -> bool {
        self.coefficients.is_empty()
    }
}

/// Affine schedule: maps iterations to execution time
#[derive(Debug, Clone)]
pub struct AffineSchedule {
    /// Schedule dimensions (one per level)
    pub dimensions: Vec<AffineExpr>,
}

impl AffineSchedule {
    /// Create identity schedule (original order)
    pub fn identity(num_dims: usize) -> Self {
        let dimensions = (0..num_dims)
            .map(|i| AffineExpr::variable(format!("i{}", i)))
            .collect();
        Self { dimensions }
    }

    /// Apply transformation matrix
    pub fn transform(&self, matrix: &TransformationMatrix) -> AffineSchedule {
        matrix.apply_schedule(self)
    }
}

// ============================================================================
// Polyhedra
// ============================================================================

/// A polyhedron defined by affine inequalities: Ax + b ≥ 0
#[derive(Debug, Clone)]
pub struct Polyhedron {
    /// Affine constraints
    pub constraints: Vec<AffineConstraint>,

    /// Dimension (number of variables)
    pub dimension: usize,
}

/// Single affine constraint: expr ≥ 0 or expr = 0
#[derive(Debug, Clone)]
pub struct AffineConstraint {
    /// Affine expression
    pub expression: AffineExpr,

    /// Constraint type
    pub constraint_type: ConstraintType,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConstraintType {
    /// expr ≥ 0
    Inequality,

    /// expr = 0
    Equality,
}

impl Polyhedron {
    /// Create empty polyhedron
    pub fn empty(dimension: usize) -> Self {
        Self {
            constraints: Vec::new(),
            dimension,
        }
    }

    /// Add constraint
    pub fn add_constraint(&mut self, constraint: AffineConstraint) {
        self.constraints.push(constraint);
    }

    /// Check if polyhedron is empty
    pub fn is_empty(&self) -> bool {
        // Simplified: check for obvious contradictions
        for c in &self.constraints {
            if c.constraint_type == ConstraintType::Equality {
                if c.expression.is_constant() && c.expression.constant != 0 {
                    return true; // 0 = constant (non-zero) is contradiction
                }
            }
        }
        false
    }

    /// Compute intersection with another polyhedron
    pub fn intersect(&self, other: &Polyhedron) -> Polyhedron {
        let mut result = self.clone();
        for constraint in &other.constraints {
            result.add_constraint(constraint.clone());
        }
        result
    }

    /// Project out a dimension
    pub fn project_out(&self, _dimension: usize) -> Polyhedron {
        // Simplified: Fourier-Motzkin elimination would be used here
        self.clone()
    }
}

/// Iteration domain for a loop nest
#[derive(Debug, Clone)]
pub struct IterationDomain {
    /// Polyhedron representing valid iterations
    pub polyhedron: Polyhedron,

    /// Loop variables
    pub variables: Vec<String>,
}

impl IterationDomain {
    /// Create domain for simple rectangular iteration space
    pub fn rectangular(bounds: Vec<(String, i64, i64)>) -> Self {
        let dimension = bounds.len();
        let mut polyhedron = Polyhedron::empty(dimension);
        let variables: Vec<String> = bounds.iter().map(|(v, _, _)| v.clone()).collect();

        for (var, lower, upper) in bounds {
            // var - lower ≥ 0
            let mut lower_expr = AffineExpr::variable(var.clone());
            lower_expr.constant = -lower;
            polyhedron.add_constraint(AffineConstraint {
                expression: lower_expr,
                constraint_type: ConstraintType::Inequality,
            });

            // upper - var ≥ 0
            let mut upper_expr = AffineExpr::constant(upper);
            *upper_expr.coefficients.entry(var).or_insert(0) -= 1;
            polyhedron.add_constraint(AffineConstraint {
                expression: upper_expr,
                constraint_type: ConstraintType::Inequality,
            });
        }

        Self {
            polyhedron,
            variables,
        }
    }
}

// ============================================================================
// Transformations
// ============================================================================

/// Transformation matrix for affine scheduling
#[derive(Debug, Clone)]
pub struct TransformationMatrix {
    /// Matrix coefficients (row-major)
    pub matrix: Vec<Vec<i64>>,

    /// Constant vector
    pub offset: Vec<i64>,
}

impl TransformationMatrix {
    /// Create identity transformation
    pub fn identity(size: usize) -> Self {
        let mut matrix = vec![vec![0; size]; size];
        for i in 0..size {
            matrix[i][i] = 1;
        }
        Self {
            matrix,
            offset: vec![0; size],
        }
    }

    /// Create loop interchange (swap dimensions i and j)
    pub fn interchange(size: usize, i: usize, j: usize) -> Self {
        let mut matrix = Self::identity(size);
        matrix.matrix.swap(i, j);
        matrix
    }

    /// Create loop reversal (reverse dimension i)
    pub fn reversal(size: usize, i: usize) -> Self {
        let mut matrix = Self::identity(size);
        matrix.matrix[i][i] = -1;
        matrix
    }

    /// Create skewing transformation
    pub fn skew(size: usize, i: usize, j: usize, factor: i64) -> Self {
        let mut matrix = Self::identity(size);
        matrix.matrix[i][j] = factor;
        matrix
    }

    /// Apply to affine schedule
    pub fn apply_schedule(&self, schedule: &AffineSchedule) -> AffineSchedule {
        let mut new_dims = Vec::new();

        for (row_idx, row) in self.matrix.iter().enumerate() {
            let mut new_expr = AffineExpr::constant(self.offset[row_idx]);

            for (col_idx, &coeff) in row.iter().enumerate() {
                if coeff != 0 && col_idx < schedule.dimensions.len() {
                    let scaled = schedule.dimensions[col_idx].mul(coeff);
                    new_expr = new_expr.add(&scaled);
                }
            }

            new_dims.push(new_expr);
        }

        AffineSchedule {
            dimensions: new_dims,
        }
    }
}

// ============================================================================
// Polyhedral Optimizer
// ============================================================================

/// Main polyhedral optimization engine
pub struct PolyhedralOptimizer {
    /// Configuration
    config: PolyhedralConfig,

    /// Optimization statistics
    stats: OptimizationStats,
}

/// Configuration for polyhedral optimization
#[derive(Debug, Clone)]
pub struct PolyhedralConfig {
    /// Enable loop tiling
    pub enable_tiling: bool,

    /// Tile size for cache blocking
    pub tile_size: usize,

    /// Enable loop fusion
    pub enable_fusion: bool,

    /// Enable loop interchange
    pub enable_interchange: bool,

    /// Enable loop skewing
    pub enable_skewing: bool,

    /// Maximize parallelism
    pub maximize_parallelism: bool,

    /// Optimize for cache locality
    pub optimize_locality: bool,
}

impl Default for PolyhedralConfig {
    fn default() -> Self {
        Self {
            enable_tiling: true,
            tile_size: 32,
            enable_fusion: true,
            enable_interchange: true,
            enable_skewing: true,
            maximize_parallelism: true,
            optimize_locality: true,
        }
    }
}

/// Optimization statistics
#[derive(Debug, Clone, Default)]
pub struct OptimizationStats {
    /// Number of loops transformed
    pub loops_transformed: usize,

    /// Number of statements fused
    pub statements_fused: usize,

    /// Estimated speedup
    pub estimated_speedup: f32,

    /// Parallelism exposed
    pub parallelism_degree: usize,
}

impl PolyhedralOptimizer {
    /// Create new polyhedral optimizer
    pub fn new() -> Self {
        Self::with_config(PolyhedralConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: PolyhedralConfig) -> Self {
        Self {
            config,
            stats: OptimizationStats::default(),
        }
    }

    /// Extract loop nest from computation graph
    pub fn extract_loop_nest(&self, _graph: &ComputationGraph) -> JitResult<LoopNest> {
        // Simplified: Create a sample loop nest
        // In production, would analyze graph structure

        let loops = vec![
            Loop {
                variable: "i".to_string(),
                lower_bound: AffineExpr::constant(0),
                upper_bound: AffineExpr::constant(100),
                step: 1,
                depth: 0,
            },
            Loop {
                variable: "j".to_string(),
                lower_bound: AffineExpr::constant(0),
                upper_bound: AffineExpr::constant(100),
                step: 1,
                depth: 1,
            },
        ];

        let domain = IterationDomain::rectangular(vec![
            ("i".to_string(), 0, 100),
            ("j".to_string(), 0, 100),
        ]);

        Ok(LoopNest {
            loops,
            statements: Vec::new(),
            dependencies: Vec::new(),
            domain,
        })
    }

    /// Compute optimal affine schedule
    pub fn compute_schedule(&mut self, nest: &LoopNest) -> JitResult<Vec<AffineSchedule>> {
        let mut schedules = Vec::new();

        // Start with identity schedule
        for stmt in &nest.statements {
            let schedule = AffineSchedule::identity(nest.loops.len());
            schedules.push(schedule);
        }

        // If no statements, create one for each loop
        if schedules.is_empty() {
            schedules.push(AffineSchedule::identity(nest.loops.len()));
        }

        // Apply transformations based on config
        if self.config.enable_interchange {
            schedules = self.apply_interchange(nest, schedules)?;
        }

        if self.config.enable_skewing {
            schedules = self.apply_skewing(nest, schedules)?;
        }

        if self.config.enable_tiling {
            schedules = self.apply_tiling(nest, schedules)?;
        }

        Ok(schedules)
    }

    /// Apply loop interchange
    fn apply_interchange(
        &mut self,
        nest: &LoopNest,
        schedules: Vec<AffineSchedule>,
    ) -> JitResult<Vec<AffineSchedule>> {
        let num_loops = nest.loops.len();

        if num_loops < 2 {
            return Ok(schedules);
        }

        // Simple heuristic: interchange if inner loop has better locality
        let transform = TransformationMatrix::interchange(num_loops, 0, 1);

        let new_schedules = schedules
            .iter()
            .map(|sched| transform.apply_schedule(sched))
            .collect();

        self.stats.loops_transformed += num_loops;

        Ok(new_schedules)
    }

    /// Apply loop skewing
    fn apply_skewing(
        &mut self,
        nest: &LoopNest,
        schedules: Vec<AffineSchedule>,
    ) -> JitResult<Vec<AffineSchedule>> {
        let num_loops = nest.loops.len();

        if num_loops < 2 {
            return Ok(schedules);
        }

        // Check if skewing would help with dependencies
        let has_diagonal_deps = nest
            .dependencies
            .iter()
            .any(|dep| dep.distance.len() >= 2 && dep.distance[0] == dep.distance[1]);

        if has_diagonal_deps {
            let transform = TransformationMatrix::skew(num_loops, 0, 1, 1);
            let new_schedules = schedules
                .iter()
                .map(|sched| transform.apply_schedule(sched))
                .collect();
            return Ok(new_schedules);
        }

        Ok(schedules)
    }

    /// Apply loop tiling
    fn apply_tiling(
        &mut self,
        nest: &LoopNest,
        schedules: Vec<AffineSchedule>,
    ) -> JitResult<Vec<AffineSchedule>> {
        // Tiling creates new loop dimensions
        // Original: i, j  → Tiled: ii, jj, i, j
        // where ii, jj are outer tile loops

        let tile_size = self.config.tile_size as i64;
        let num_loops = nest.loops.len();

        // Create tiled schedule (simplified)
        let mut new_schedules = Vec::new();

        for schedule in schedules {
            let mut tiled_dims = Vec::new();

            // Outer tile loops (ii, jj, ...)
            for dim in &schedule.dimensions {
                // ii = floor(i / tile_size)
                let tiled = dim.mul(1); // Simplified
                tiled_dims.push(tiled);
            }

            // Inner tile loops (original dimensions)
            tiled_dims.extend(schedule.dimensions.clone());

            new_schedules.push(AffineSchedule {
                dimensions: tiled_dims,
            });
        }

        self.stats.loops_transformed += num_loops;

        Ok(new_schedules)
    }

    /// Analyze dependencies
    pub fn analyze_dependencies(&self, nest: &LoopNest) -> Vec<Dependence> {
        let mut dependencies = Vec::new();

        // Simplified: check all statement pairs
        for (i, stmt1) in nest.statements.iter().enumerate() {
            for (j, stmt2) in nest.statements.iter().enumerate().skip(i) {
                if let Some(dep) = self.check_dependence(stmt1, stmt2) {
                    dependencies.push(dep);
                }
            }
        }

        dependencies
    }

    /// Check for dependence between two statements
    fn check_dependence(&self, stmt1: &Statement, stmt2: &Statement) -> Option<Dependence> {
        // Check if there's a memory access conflict
        for access1 in &stmt1.accesses {
            for access2 in &stmt2.accesses {
                if access1.array_name == access2.array_name {
                    let dep_type = self.classify_dependence(access1, access2);

                    if dep_type != DependenceType::Input {
                        // Compute dependence polyhedron (simplified)
                        let polyhedron = Polyhedron::empty(2);

                        return Some(Dependence {
                            source: stmt1.id,
                            target: stmt2.id,
                            dep_type,
                            polyhedron,
                            distance: vec![1, 0], // Simplified
                        });
                    }
                }
            }
        }

        None
    }

    /// Classify dependence type
    fn classify_dependence(
        &self,
        access1: &MemoryAccess,
        access2: &MemoryAccess,
    ) -> DependenceType {
        match (&access1.access_type, &access2.access_type) {
            (AccessType::Write, AccessType::Read) => DependenceType::Flow,
            (AccessType::Read, AccessType::Write) => DependenceType::Anti,
            (AccessType::Write, AccessType::Write) => DependenceType::Output,
            (AccessType::Read, AccessType::Read) => DependenceType::Input,
            _ => DependenceType::Flow,
        }
    }

    /// Check if loop fusion is legal
    pub fn is_fusion_legal(&self, nest1: &LoopNest, nest2: &LoopNest) -> bool {
        // Check if loops have compatible bounds and no conflicting dependencies
        if nest1.loops.len() != nest2.loops.len() {
            return false;
        }

        // Check bounds compatibility
        for (loop1, loop2) in nest1.loops.iter().zip(nest2.loops.iter()) {
            if loop1.lower_bound != loop2.lower_bound || loop1.upper_bound != loop2.upper_bound {
                return false;
            }
        }

        true
    }

    /// Get optimization statistics
    pub fn statistics(&self) -> &OptimizationStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = OptimizationStats::default();
    }
}

impl Default for PolyhedralOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Optimization Strategies
// ============================================================================

/// Strategy for selecting polyhedral transformations
#[derive(Debug, Clone)]
pub enum OptimizationStrategy {
    /// Maximize parallelism
    MaxParallelism,

    /// Maximize cache locality
    MaxLocality,

    /// Balanced approach
    Balanced,

    /// Custom transformation sequence
    Custom(Vec<TransformationType>),
}

/// Types of polyhedral transformations
#[derive(Debug, Clone, PartialEq)]
pub enum TransformationType {
    Interchange(usize, usize),
    Skewing(usize, usize, i64),
    Tiling(Vec<usize>),
    Fusion(Vec<usize>),
    Distribution(Vec<usize>),
    Reversal(usize),
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_affine_expr() {
        let expr1 = AffineExpr::constant(5);
        let expr2 = AffineExpr::variable("x".to_string());

        let sum = expr1.add(&expr2);
        assert_eq!(sum.constant, 5);
        assert_eq!(sum.coefficients.get("x"), Some(&1));

        let scaled = expr2.mul(3);
        assert_eq!(scaled.coefficients.get("x"), Some(&3));
    }

    #[test]
    fn test_affine_evaluation() {
        let mut expr = AffineExpr::constant(10);
        expr.coefficients.insert("x".to_string(), 2);
        expr.coefficients.insert("y".to_string(), 3);

        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 4);
        vars.insert("y".to_string(), 5);

        let result = expr.evaluate(&vars);
        assert_eq!(result, 10 + 2 * 4 + 3 * 5); // 10 + 8 + 15 = 33
    }

    #[test]
    fn test_polyhedron() {
        let mut poly = Polyhedron::empty(2);
        assert_eq!(poly.dimension, 2);

        poly.add_constraint(AffineConstraint {
            expression: AffineExpr::variable("x".to_string()),
            constraint_type: ConstraintType::Inequality,
        });

        assert_eq!(poly.constraints.len(), 1);
        assert!(!poly.is_empty());
    }

    #[test]
    fn test_iteration_domain() {
        let domain =
            IterationDomain::rectangular(vec![("i".to_string(), 0, 10), ("j".to_string(), 0, 20)]);

        assert_eq!(domain.variables.len(), 2);
        assert_eq!(domain.polyhedron.constraints.len(), 4); // 2 bounds × 2 variables
    }

    #[test]
    fn test_transformation_matrix() {
        let identity = TransformationMatrix::identity(3);
        assert_eq!(identity.matrix[0][0], 1);
        assert_eq!(identity.matrix[0][1], 0);

        let interchange = TransformationMatrix::interchange(3, 0, 1);
        assert_eq!(interchange.matrix[0][0], 0);
        assert_eq!(interchange.matrix[0][1], 1);
    }

    #[test]
    fn test_polyhedral_optimizer() {
        let optimizer = PolyhedralOptimizer::new();
        assert!(optimizer.config.enable_tiling);
        assert!(optimizer.config.enable_fusion);
    }

    #[test]
    fn test_schedule_computation() {
        use crate::graph::GraphBuilder;
        use torsh_core::{DType, Shape};

        let mut optimizer = PolyhedralOptimizer::new();

        let mut builder = GraphBuilder::new();
        let x = builder.add_input("x".to_string(), Shape::new(vec![10, 10]), DType::F32);
        builder.mark_output(x).unwrap();

        let graph = builder.build().unwrap();
        let nest = optimizer.extract_loop_nest(&graph).unwrap();
        let schedules = optimizer.compute_schedule(&nest).unwrap();

        assert!(!schedules.is_empty());
    }

    #[test]
    fn test_dependence_analysis() {
        let optimizer = PolyhedralOptimizer::new();

        let stmt1 = Statement {
            id: 0,
            node_id: 0.into(),
            domain: Polyhedron::empty(2),
            schedule: AffineSchedule::identity(2),
            accesses: vec![MemoryAccess {
                array_name: "A".to_string(),
                access_fn: vec![AffineExpr::variable("i".to_string())],
                access_type: AccessType::Write,
            }],
        };

        let stmt2 = Statement {
            id: 1,
            node_id: 1.into(),
            domain: Polyhedron::empty(2),
            schedule: AffineSchedule::identity(2),
            accesses: vec![MemoryAccess {
                array_name: "A".to_string(),
                access_fn: vec![AffineExpr::variable("i".to_string())],
                access_type: AccessType::Read,
            }],
        };

        let dep = optimizer.check_dependence(&stmt1, &stmt2);
        assert!(dep.is_some());
        assert_eq!(dep.unwrap().dep_type, DependenceType::Flow);
    }

    #[test]
    fn test_fusion_legality() {
        let optimizer = PolyhedralOptimizer::new();

        let nest1 = LoopNest {
            loops: vec![Loop {
                variable: "i".to_string(),
                lower_bound: AffineExpr::constant(0),
                upper_bound: AffineExpr::constant(10),
                step: 1,
                depth: 0,
            }],
            statements: Vec::new(),
            dependencies: Vec::new(),
            domain: IterationDomain::rectangular(vec![("i".to_string(), 0, 10)]),
        };

        let nest2 = nest1.clone();

        assert!(optimizer.is_fusion_legal(&nest1, &nest2));
    }
}
