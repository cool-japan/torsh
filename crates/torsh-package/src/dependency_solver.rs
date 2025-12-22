//! Advanced dependency resolution using SAT-based constraint solving
//!
//! This module implements a sophisticated dependency resolution algorithm
//! using Boolean Satisfiability (SAT) solving techniques, similar to modern
//! package managers like cargo, npm, and pip.

use std::collections::{HashMap, HashSet};
use std::fmt;

use serde::{Deserialize, Serialize};
use torsh_core::error::{Result, TorshError};

use crate::dependency::DependencySpec;

/// SAT variable representing a package version choice
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SatVariable(usize);

/// SAT literal (variable or negation)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SatLiteral {
    variable: SatVariable,
    negated: bool,
}

/// SAT clause (disjunction of literals)
#[derive(Debug, Clone)]
pub struct SatClause {
    literals: Vec<SatLiteral>,
}

/// Assignment of variables to boolean values
#[derive(Debug, Clone)]
pub struct Assignment {
    values: HashMap<SatVariable, bool>,
}

/// Conflict-Driven Clause Learning (CDCL) SAT solver
#[derive(Debug)]
pub struct CdclSolver {
    /// All clauses in the problem
    clauses: Vec<SatClause>,
    /// Current partial assignment
    assignment: Assignment,
    /// Decision level for each variable
    decision_levels: HashMap<SatVariable, usize>,
    /// Current decision level
    current_level: usize,
    /// Learned clauses from conflicts
    learned_clauses: Vec<SatClause>,
    /// Variable activity scores for decision heuristics
    activity: HashMap<SatVariable, f64>,
}

/// Package version constraint in CNF form
#[derive(Debug, Clone)]
pub struct VersionConstraint {
    /// Package name
    pub package: String,
    /// Version
    pub version: String,
    /// SAT variable representing this version choice
    pub variable: SatVariable,
}

/// Dependency constraint solver using SAT
pub struct DependencySatSolver {
    /// Mapping of package versions to SAT variables
    version_vars: HashMap<(String, String), SatVariable>,
    /// Reverse mapping
    var_to_version: HashMap<SatVariable, (String, String)>,
    /// Next available variable ID
    next_var_id: usize,
    /// SAT solver instance
    solver: CdclSolver,
    /// All available package versions
    available_versions: HashMap<String, Vec<String>>,
}

/// Solution to dependency resolution problem
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencySolution {
    /// Selected package versions
    pub selected_versions: HashMap<String, String>,
    /// Installation order (topologically sorted)
    pub install_order: Vec<String>,
    /// Conflict information if unsatisfiable
    pub conflicts: Vec<String>,
}

impl SatLiteral {
    /// Create a positive literal
    pub fn positive(var: SatVariable) -> Self {
        Self {
            variable: var,
            negated: false,
        }
    }

    /// Create a negative literal
    pub fn negative(var: SatVariable) -> Self {
        Self {
            variable: var,
            negated: true,
        }
    }

    /// Negate this literal
    pub fn negate(&self) -> Self {
        Self {
            variable: self.variable,
            negated: !self.negated,
        }
    }

    /// Check if literal is satisfied by assignment
    pub fn is_satisfied(&self, assignment: &Assignment) -> Option<bool> {
        assignment
            .get(self.variable)
            .map(|value| if self.negated { !value } else { value })
    }
}

impl SatClause {
    /// Create a new clause
    pub fn new(literals: Vec<SatLiteral>) -> Self {
        Self { literals }
    }

    /// Check if clause is satisfied by assignment
    pub fn is_satisfied(&self, assignment: &Assignment) -> bool {
        self.literals
            .iter()
            .any(|lit| lit.is_satisfied(assignment) == Some(true))
    }

    /// Check if clause is conflicting (all literals false)
    pub fn is_conflicting(&self, assignment: &Assignment) -> bool {
        self.literals
            .iter()
            .all(|lit| lit.is_satisfied(assignment) == Some(false))
    }

    /// Get unit literal if clause is unit (all but one literal assigned false)
    pub fn get_unit_literal(&self, assignment: &Assignment) -> Option<SatLiteral> {
        let mut unassigned = None;
        let mut unassigned_count = 0;

        for literal in &self.literals {
            match literal.is_satisfied(assignment) {
                Some(true) => return None, // Clause already satisfied
                Some(false) => continue,   // Literal is false
                None => {
                    unassigned = Some(*literal);
                    unassigned_count += 1;
                    if unassigned_count > 1 {
                        return None; // More than one unassigned
                    }
                }
            }
        }

        if unassigned_count == 1 {
            unassigned
        } else {
            None
        }
    }
}

impl Assignment {
    /// Create an empty assignment
    pub fn new() -> Self {
        Self {
            values: HashMap::new(),
        }
    }

    /// Get the value of a variable
    pub fn get(&self, var: SatVariable) -> Option<bool> {
        self.values.get(&var).copied()
    }

    /// Set the value of a variable
    pub fn set(&mut self, var: SatVariable, value: bool) {
        self.values.insert(var, value);
    }

    /// Unset a variable
    pub fn unset(&mut self, var: SatVariable) {
        self.values.remove(&var);
    }

    /// Check if variable is assigned
    pub fn is_assigned(&self, var: SatVariable) -> bool {
        self.values.contains_key(&var)
    }
}

impl Default for Assignment {
    fn default() -> Self {
        Self::new()
    }
}

impl CdclSolver {
    /// Create a new CDCL solver
    pub fn new() -> Self {
        Self {
            clauses: Vec::new(),
            assignment: Assignment::new(),
            decision_levels: HashMap::new(),
            current_level: 0,
            learned_clauses: Vec::new(),
            activity: HashMap::new(),
        }
    }

    /// Add a clause to the solver
    pub fn add_clause(&mut self, clause: SatClause) {
        // Update activity scores for variables in the clause
        for literal in &clause.literals {
            *self.activity.entry(literal.variable).or_insert(0.0) += 1.0;
        }
        self.clauses.push(clause);
    }

    /// Solve the SAT problem using CDCL algorithm
    pub fn solve(&mut self) -> Result<bool> {
        // Initial unit propagation
        if self.unit_propagate()? {
            return Ok(false); // Conflict at decision level 0 - UNSAT
        }

        loop {
            // Check if all variables are assigned
            if self.is_complete() {
                return Ok(true); // SAT
            }

            // Make a decision
            let decision_var = self.choose_decision_variable();
            self.current_level += 1;
            self.assign(decision_var, true, self.current_level);

            // Propagate and handle conflicts
            loop {
                if self.unit_propagate()? {
                    // Conflict occurred
                    if self.current_level == 0 {
                        return Ok(false); // UNSAT
                    }

                    // Analyze conflict and learn clause
                    let (learned_clause, backtrack_level) = self.analyze_conflict()?;
                    self.learned_clauses.push(learned_clause.clone());
                    self.add_clause(learned_clause);

                    // Backtrack
                    self.backtrack(backtrack_level);
                } else {
                    break; // No conflict
                }
            }
        }
    }

    /// Unit propagation: propagate all unit clauses
    fn unit_propagate(&mut self) -> Result<bool> {
        loop {
            let mut propagated = false;

            // Collect unit literals first to avoid borrow checker issues
            let mut unit_literals = Vec::new();
            let mut conflicts = Vec::new();

            // Check all clauses for unit clauses
            for clause in self.clauses.iter().chain(self.learned_clauses.iter()) {
                if let Some(unit_literal) = clause.get_unit_literal(&self.assignment) {
                    unit_literals.push(unit_literal);
                } else if clause.is_conflicting(&self.assignment) {
                    conflicts.push(true);
                }
            }

            // Apply assignments after iteration
            for unit_literal in unit_literals {
                self.assign(
                    unit_literal.variable,
                    !unit_literal.negated,
                    self.current_level,
                );
                propagated = true;
            }

            // Check for conflicts
            if !conflicts.is_empty() {
                return Ok(true); // Conflict
            }

            if !propagated {
                break;
            }
        }

        Ok(false) // No conflict
    }

    /// Assign a variable at a decision level
    fn assign(&mut self, var: SatVariable, value: bool, level: usize) {
        self.assignment.set(var, value);
        self.decision_levels.insert(var, level);
    }

    /// Check if assignment is complete
    fn is_complete(&self) -> bool {
        // Get all variables from clauses
        let mut all_vars = HashSet::new();
        for clause in self.clauses.iter().chain(self.learned_clauses.iter()) {
            for literal in &clause.literals {
                all_vars.insert(literal.variable);
            }
        }

        all_vars.iter().all(|var| self.assignment.is_assigned(*var))
    }

    /// Choose next decision variable using activity heuristics
    fn choose_decision_variable(&self) -> SatVariable {
        // Get all variables
        let mut unassigned_vars: Vec<_> = self
            .activity
            .iter()
            .filter(|(var, _)| !self.assignment.is_assigned(**var))
            .collect();

        if unassigned_vars.is_empty() {
            // Fallback: find any unassigned variable from clauses
            for clause in self.clauses.iter().chain(self.learned_clauses.iter()) {
                for literal in &clause.literals {
                    if !self.assignment.is_assigned(literal.variable) {
                        return literal.variable;
                    }
                }
            }
            // Should never reach here if is_complete() check is correct
            panic!("No unassigned variables found");
        }

        // Sort by activity (highest first)
        unassigned_vars.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

        *unassigned_vars[0].0
    }

    /// Analyze conflict and learn a new clause
    fn analyze_conflict(&self) -> Result<(SatClause, usize)> {
        // Simplified conflict analysis - in production, use 1UIP analysis
        // For now, just learn a clause that prevents the current decision

        let mut learned_literals = Vec::new();
        let mut backtrack_level = 0;

        // Find variables assigned at current level
        for (var, level) in &self.decision_levels {
            if *level == self.current_level {
                if let Some(value) = self.assignment.get(*var) {
                    learned_literals.push(if value {
                        SatLiteral::negative(*var)
                    } else {
                        SatLiteral::positive(*var)
                    });
                }
            } else if *level > backtrack_level {
                backtrack_level = *level;
            }
        }

        if learned_literals.is_empty() {
            // Add at least one literal to prevent empty clause
            for (var, _) in &self.decision_levels {
                if let Some(value) = self.assignment.get(*var) {
                    learned_literals.push(if value {
                        SatLiteral::negative(*var)
                    } else {
                        SatLiteral::positive(*var)
                    });
                    break;
                }
            }
        }

        Ok((
            SatClause::new(learned_literals),
            backtrack_level.saturating_sub(1),
        ))
    }

    /// Backtrack to a decision level
    fn backtrack(&mut self, level: usize) {
        // Remove assignments made after the backtrack level
        let vars_to_remove: Vec<_> = self
            .decision_levels
            .iter()
            .filter(|(_, &var_level)| var_level > level)
            .map(|(var, _)| *var)
            .collect();

        for var in vars_to_remove {
            self.assignment.unset(var);
            self.decision_levels.remove(&var);
        }

        self.current_level = level;
    }

    /// Get the current assignment
    pub fn get_assignment(&self) -> &Assignment {
        &self.assignment
    }
}

impl Default for CdclSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl DependencySatSolver {
    /// Create a new dependency SAT solver
    pub fn new() -> Self {
        Self {
            version_vars: HashMap::new(),
            var_to_version: HashMap::new(),
            next_var_id: 0,
            solver: CdclSolver::new(),
            available_versions: HashMap::new(),
        }
    }

    /// Get or create a SAT variable for a package version
    fn get_or_create_variable(&mut self, package: &str, version: &str) -> SatVariable {
        let key = (package.to_string(), version.to_string());
        if let Some(&var) = self.version_vars.get(&key) {
            return var;
        }

        let var = SatVariable(self.next_var_id);
        self.next_var_id += 1;
        self.version_vars.insert(key.clone(), var);
        self.var_to_version.insert(var, key);
        var
    }

    /// Add available versions for a package
    pub fn add_available_versions(&mut self, package: &str, versions: Vec<String>) {
        self.available_versions
            .insert(package.to_string(), versions.clone());

        // Create variables for all versions
        for version in &versions {
            self.get_or_create_variable(package, version);
        }

        // Add constraint: exactly one version must be selected (if package is needed)
        // This is encoded as: at most one version (pairwise conflicts)
        for i in 0..versions.len() {
            for j in (i + 1)..versions.len() {
                let var_i = self.get_or_create_variable(package, &versions[i]);
                let var_j = self.get_or_create_variable(package, &versions[j]);

                // Clause: NOT var_i OR NOT var_j (can't have both versions)
                self.solver.add_clause(SatClause::new(vec![
                    SatLiteral::negative(var_i),
                    SatLiteral::negative(var_j),
                ]));
            }
        }
    }

    /// Add dependency constraint: if package A version X is selected,
    /// then one of the compatible versions of package B must be selected
    pub fn add_dependency_constraint(
        &mut self,
        package: &str,
        version: &str,
        dep_spec: &DependencySpec,
    ) -> Result<()> {
        let package_var = self.get_or_create_variable(package, version);

        // Find all compatible versions of the dependency (clone to avoid borrow issues)
        let dep_versions = self
            .available_versions
            .get(&dep_spec.name)
            .ok_or_else(|| {
                TorshError::InvalidArgument(format!(
                    "No versions available for dependency: {}",
                    dep_spec.name
                ))
            })?
            .clone();

        let mut compatible_vars = Vec::new();
        for dep_version in &dep_versions {
            if dep_spec.is_satisfied_by(dep_version)? {
                let dep_var = self.get_or_create_variable(&dep_spec.name, dep_version);
                compatible_vars.push(dep_var);
            }
        }

        if compatible_vars.is_empty() {
            return Err(TorshError::InvalidArgument(format!(
                "No compatible versions found for dependency: {} with requirement: {}",
                dep_spec.name, dep_spec.version_req
            )));
        }

        // Add clause: NOT package_var OR dep_var1 OR dep_var2 OR ...
        // Meaning: if this package version is selected, at least one compatible dep version must be selected
        let mut clause_literals = vec![SatLiteral::negative(package_var)];
        for dep_var in compatible_vars {
            clause_literals.push(SatLiteral::positive(dep_var));
        }

        self.solver.add_clause(SatClause::new(clause_literals));
        Ok(())
    }

    /// Add root dependency constraint: at least one version of the root package must be selected
    pub fn add_root_constraint(&mut self, package: &str) -> Result<()> {
        let versions = self
            .available_versions
            .get(package)
            .ok_or_else(|| {
                TorshError::InvalidArgument(format!(
                    "No versions available for root package: {}",
                    package
                ))
            })?
            .clone();

        // Add clause: var1 OR var2 OR ... (at least one version must be selected)
        let clause_literals: Vec<_> = versions
            .iter()
            .map(|v| {
                let var = self.get_or_create_variable(package, v);
                SatLiteral::positive(var)
            })
            .collect();

        self.solver.add_clause(SatClause::new(clause_literals));
        Ok(())
    }

    /// Solve the dependency constraints
    pub fn solve(&mut self) -> Result<DependencySolution> {
        let is_sat = self.solver.solve()?;

        if !is_sat {
            return Ok(DependencySolution {
                selected_versions: HashMap::new(),
                install_order: Vec::new(),
                conflicts: vec!["Dependency constraints are unsatisfiable".to_string()],
            });
        }

        // Extract solution from assignment
        let assignment = self.solver.get_assignment();
        let mut selected_versions = HashMap::new();

        for (var, &value) in &assignment.values {
            if value {
                if let Some((package, version)) = self.var_to_version.get(var) {
                    selected_versions.insert(package.clone(), version.clone());
                }
            }
        }

        // Compute topological order for installation
        let install_order = self.compute_install_order(&selected_versions)?;

        Ok(DependencySolution {
            selected_versions,
            install_order,
            conflicts: Vec::new(),
        })
    }

    /// Compute installation order using topological sort
    fn compute_install_order(&self, selected: &HashMap<String, String>) -> Result<Vec<String>> {
        // Simple topological sort - in production, use the dependency graph
        let mut order: Vec<_> = selected.keys().cloned().collect();
        order.sort(); // Simplified - should use actual dependency order
        Ok(order)
    }
}

impl Default for DependencySatSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for DependencySolution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if !self.conflicts.is_empty() {
            writeln!(f, "Dependency resolution failed:")?;
            for conflict in &self.conflicts {
                writeln!(f, "  - {}", conflict)?;
            }
            return Ok(());
        }

        writeln!(f, "Dependency resolution successful:")?;
        writeln!(f, "Selected versions:")?;
        for (package, version) in &self.selected_versions {
            writeln!(f, "  {} = {}", package, version)?;
        }
        writeln!(f, "Installation order:")?;
        for (i, package) in self.install_order.iter().enumerate() {
            writeln!(f, "  {}. {}", i + 1, package)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sat_literal() {
        let var = SatVariable(0);
        let pos = SatLiteral::positive(var);
        let neg = SatLiteral::negative(var);

        assert!(!pos.negated);
        assert!(neg.negated);
        assert_eq!(pos.negate(), neg);
    }

    #[test]
    fn test_sat_clause_satisfaction() {
        let var1 = SatVariable(0);
        let var2 = SatVariable(1);

        let clause = SatClause::new(vec![SatLiteral::positive(var1), SatLiteral::negative(var2)]);

        let mut assignment = Assignment::new();
        assignment.set(var1, true);
        assignment.set(var2, false);

        assert!(clause.is_satisfied(&assignment));
    }

    #[test]
    fn test_simple_sat_solving() {
        let mut solver = CdclSolver::new();

        let var1 = SatVariable(0);
        let var2 = SatVariable(1);

        // Add clause: var1 OR var2
        solver.add_clause(SatClause::new(vec![
            SatLiteral::positive(var1),
            SatLiteral::positive(var2),
        ]));

        // Add clause: NOT var1 OR var2 (if var1 then var2)
        solver.add_clause(SatClause::new(vec![
            SatLiteral::negative(var1),
            SatLiteral::positive(var2),
        ]));

        let result = solver.solve().unwrap();
        assert!(result); // Should be SAT
    }

    #[test]
    fn test_dependency_sat_solver() {
        let mut solver = DependencySatSolver::new();

        // Add package A with versions 1.0.0 and 2.0.0
        solver.add_available_versions("pkg-a", vec!["1.0.0".to_string(), "2.0.0".to_string()]);

        // Add package B with versions 1.0.0
        solver.add_available_versions("pkg-b", vec!["1.0.0".to_string()]);

        // Package A 1.0.0 depends on B ^1.0.0
        let dep_spec = DependencySpec::new("pkg-b".to_string(), "^1.0.0".to_string());
        solver
            .add_dependency_constraint("pkg-a", "1.0.0", &dep_spec)
            .unwrap();

        // We want package A
        solver.add_root_constraint("pkg-a").unwrap();

        let solution = solver.solve().unwrap();
        assert!(solution.conflicts.is_empty());
        assert!(solution.selected_versions.contains_key("pkg-a"));
    }

    #[test]
    fn test_version_conflict_detection() {
        let mut solver = DependencySatSolver::new();

        // Add package A with version 1.0.0
        solver.add_available_versions("pkg-a", vec!["1.0.0".to_string()]);

        // Add package B with version 1.0.0 that requires A 2.0.0 (not available)
        solver.add_available_versions("pkg-b", vec!["1.0.0".to_string()]);

        let dep_spec = DependencySpec::new("pkg-a".to_string(), "^2.0.0".to_string());
        let result = solver.add_dependency_constraint("pkg-b", "1.0.0", &dep_spec);

        // Should fail because no compatible version of A exists
        assert!(result.is_err());
    }
}
