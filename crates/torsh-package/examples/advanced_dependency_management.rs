//! Advanced Dependency Management Example
//!
//! This example demonstrates the advanced dependency management features:
//! - SAT-based dependency resolution
//! - Lockfile generation for reproducible builds
//! - Parallel dependency installation
//! - Sandboxed package execution
//! - Role-based access control (RBAC)

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use torsh_package::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Advanced Dependency Management Demo ===\n");

    // 1. Demonstrate SAT-based dependency resolution
    demonstrate_sat_solver()?;

    // 2. Demonstrate lockfile generation and validation
    demonstrate_lockfile()?;

    // 3. Demonstrate parallel dependency installation
    demonstrate_parallel_installation()?;

    // 4. Demonstrate sandboxed execution
    demonstrate_sandbox()?;

    // 5. Demonstrate access control (RBAC)
    demonstrate_rbac()?;

    println!("\n=== All demonstrations completed successfully! ===");
    Ok(())
}

/// Demonstrate SAT-based dependency resolution
fn demonstrate_sat_solver() -> Result<(), Box<dyn std::error::Error>> {
    println!("1. SAT-Based Dependency Resolution");
    println!("-----------------------------------");

    let mut solver = DependencySatSolver::new();

    // Add available package versions
    solver.add_available_versions(
        "torch",
        vec![
            "2.0.0".to_string(),
            "2.1.0".to_string(),
            "2.2.0".to_string(),
        ],
    );

    solver.add_available_versions("numpy", vec!["1.24.0".to_string(), "1.25.0".to_string()]);

    solver.add_available_versions("scipy", vec!["1.11.0".to_string()]);

    // Add dependency constraints
    // torch 2.1.0 depends on numpy ^1.24
    let numpy_dep = DependencySpec::new("numpy".to_string(), "^1.24.0".to_string());
    solver.add_dependency_constraint("torch", "2.1.0", &numpy_dep)?;

    // scipy 1.11.0 depends on numpy ^1.24
    let numpy_dep2 = DependencySpec::new("numpy".to_string(), "^1.24.0".to_string());
    solver.add_dependency_constraint("scipy", "1.11.0", &numpy_dep2)?;

    // We want torch as a root dependency
    solver.add_root_constraint("torch")?;

    // Solve the constraints
    println!("Solving dependency constraints...");
    let solution = solver.solve()?;

    println!("\nSolution:");
    if solution.conflicts.is_empty() {
        println!("✓ Dependencies resolved successfully!");
        for (pkg, version) in &solution.selected_versions {
            println!("  {} = {}", pkg, version);
        }
        println!("\nInstallation order:");
        for (i, pkg) in solution.install_order.iter().enumerate() {
            println!("  {}. {}", i + 1, pkg);
        }
    } else {
        println!("✗ Conflicts detected:");
        for conflict in &solution.conflicts {
            println!("  - {}", conflict);
        }
    }

    println!();
    Ok(())
}

/// Demonstrate lockfile generation and validation
fn demonstrate_lockfile() -> Result<(), Box<dyn std::error::Error>> {
    println!("2. Lockfile Generation and Validation");
    println!("--------------------------------------");

    // Create a lockfile generator
    let generator = LockfileGenerator::new()
        .with_optional(true)
        .with_platform_specific(true);

    // Create some dependency specs
    let deps = vec![
        DependencySpec::new("torch".to_string(), "^2.0.0".to_string()),
        DependencySpec::new("numpy".to_string(), "^1.24.0".to_string()),
        DependencySpec::new("scipy".to_string(), "^1.11.0".to_string()),
    ];

    // Generate lockfile
    println!("Generating lockfile...");
    let lockfile =
        generator.generate_from_specs("my-ml-project".to_string(), "1.0.0".to_string(), &deps)?;

    println!(
        "✓ Lockfile generated with {} dependencies",
        lockfile.dependencies.len()
    );
    println!("  Generated at: {}", lockfile.generated_at);
    println!("  Platform: {}", lockfile.metadata.platform);

    // Get statistics
    let stats = lockfile.get_statistics();
    println!("\nLockfile statistics:");
    println!("  Total dependencies: {}", stats.total_dependencies);
    println!("  Optional dependencies: {}", stats.optional_dependencies);
    println!(
        "  Platform-specific: {}",
        stats.platform_specific_dependencies
    );

    // Validate the lockfile
    let validator = LockfileValidator::new()
        .with_allow_missing_optional(true)
        .with_strict_integrity(false);

    println!("\nValidating lockfile...");
    let report = validator.validate(&lockfile)?;

    if report.is_valid() {
        println!("✓ Lockfile is valid!");
    } else {
        println!("✗ Validation errors:");
        for error in &report.errors {
            println!("  - {}", error);
        }
    }

    if !report.warnings.is_empty() {
        println!("\nWarnings:");
        for warning in &report.warnings {
            println!("  - {}", warning);
        }
    }

    // Save lockfile to temp directory
    let temp_dir = std::env::temp_dir();
    let lockfile_path = temp_dir.join("my-project.lock");
    lockfile.save(&lockfile_path)?;
    println!("\n✓ Lockfile saved to: {:?}", lockfile_path);

    // Load and compare
    let loaded_lockfile = PackageLockfile::load(&lockfile_path)?;
    println!("✓ Lockfile loaded successfully");

    // Compare versions
    let diff = validator.compare_lockfiles(&lockfile, &loaded_lockfile);
    if diff.has_changes() {
        println!("\nChanges detected: {}", diff.summary());
    } else {
        println!("✓ No changes detected (lockfiles are identical)");
    }

    println!();
    Ok(())
}

/// Demonstrate parallel dependency installation
fn demonstrate_parallel_installation() -> Result<(), Box<dyn std::error::Error>> {
    println!("3. Parallel Dependency Installation");
    println!("------------------------------------");

    // Create download options
    let options = DownloadOptions::new()
        .with_max_parallel(4)
        .with_timeout(60)
        .with_max_retries(3)
        .with_verify_integrity(true);

    println!("Download options:");
    println!("  Max parallel: {}", options.max_parallel);
    println!("  Timeout: {} seconds", options.timeout_secs);
    println!("  Max retries: {}", options.max_retries);

    // Create a mock registry
    let registry: Arc<dyn PackageRegistry> = Arc::new(LocalPackageRegistry::new());

    // Create installer
    let temp_dir = std::env::temp_dir().join("torsh-packages");
    std::fs::create_dir_all(&temp_dir)?;

    let _installer = ParallelDependencyInstaller::new(registry, temp_dir.clone(), options);

    // Create an installation plan
    let mut plan = InstallationPlan::new();

    // Add packages in dependency order
    plan.add_package(PlannedPackage {
        name: "numpy".to_string(),
        version: "1.24.0".to_string(),
        priority: 0,
        depends_on: vec![],
        size: 50 * 1024 * 1024, // 50 MB
    });

    plan.add_package(PlannedPackage {
        name: "scipy".to_string(),
        version: "1.11.0".to_string(),
        priority: 1,
        depends_on: vec!["numpy".to_string()],
        size: 30 * 1024 * 1024, // 30 MB
    });

    plan.add_package(PlannedPackage {
        name: "torch".to_string(),
        version: "2.1.0".to_string(),
        priority: 1,
        depends_on: vec!["numpy".to_string()],
        size: 800 * 1024 * 1024, // 800 MB
    });

    println!("\nInstallation plan:");
    println!("  Total packages: {}", plan.packages.len());
    println!("  Total size: {} MB", plan.total_size / (1024 * 1024));
    println!("  Estimated time: {} seconds", plan.estimated_time);

    // Sort by dependencies
    plan.sort_by_dependencies()?;
    println!("\n✓ Packages sorted by dependency order");

    println!("\nInstallation would proceed in this order:");
    for (i, pkg) in plan.packages.iter().enumerate() {
        println!(
            "  {}. {} v{} ({} MB)",
            i + 1,
            pkg.name,
            pkg.version,
            pkg.size / (1024 * 1024)
        );
    }

    // Note: Actual installation skipped in demo to avoid network calls
    println!("\n✓ Installation plan created successfully");
    println!("  (Actual installation skipped in demo)");

    println!();
    Ok(())
}

/// Demonstrate sandboxed execution
fn demonstrate_sandbox() -> Result<(), Box<dyn std::error::Error>> {
    println!("4. Sandboxed Package Execution");
    println!("-------------------------------");

    // Create a restrictive sandbox configuration
    let config = SandboxConfig::restrictive();

    println!("Sandbox configuration:");
    println!("  Max CPU: {}%", config.limits.max_cpu_percent);
    println!(
        "  Max memory: {} MB",
        config.limits.max_memory_bytes / (1024 * 1024)
    );
    println!("  Max execution time: {:?}", config.max_execution_time);
    println!("  Network access: {}", config.network.allowed);
    println!(
        "  Filesystem: {} readonly paths, {} forbidden paths",
        config.filesystem.readonly_paths.len(),
        config.filesystem.forbidden_paths.len()
    );

    // Create sandbox
    let mut sandbox = Sandbox::new(config)?;
    println!("\n✓ Sandbox created with ID: {}", sandbox.id());

    // Execute a function in the sandbox
    println!("\nExecuting function in sandbox...");
    let result = sandbox.execute(|| {
        // Simulate some computation
        let sum: u64 = (0..1000).sum();
        Ok(sum)
    });

    match result.result {
        Ok(value) => {
            println!("✓ Execution completed successfully");
            println!("  Result: {}", value);
        }
        Err(e) => {
            println!("✗ Execution failed: {}", e);
        }
    }

    // Display resource usage
    println!("\nResource usage:");
    println!("  Peak CPU: {:.2}%", result.resource_usage.peak_cpu);
    println!("  Peak memory: {} bytes", result.resource_usage.peak_memory);
    println!(
        "  Execution time: {:?}",
        result.resource_usage.execution_time
    );

    // Check for violations
    if result.violations.is_empty() {
        println!("  ✓ No security violations detected");
    } else {
        println!("  ✗ {} violations detected:", result.violations.len());
        for violation in &result.violations {
            println!(
                "    - {:?}: {}",
                violation.violation_type, violation.description
            );
        }
    }

    // Test timeout violation
    println!("\nTesting timeout violation...");
    let mut timeout_config = SandboxConfig::restrictive();
    timeout_config.max_execution_time = Duration::from_millis(1);
    let mut timeout_sandbox = Sandbox::new(timeout_config)?;

    let timeout_result = timeout_sandbox.execute(|| {
        std::thread::sleep(Duration::from_millis(100));
        Ok(())
    });

    if !timeout_result.violations.is_empty() {
        println!("✓ Timeout violation correctly detected");
        for violation in &timeout_result.violations {
            println!("  - {}", violation.description);
        }
    }

    println!();
    Ok(())
}

/// Demonstrate role-based access control
fn demonstrate_rbac() -> Result<(), Box<dyn std::error::Error>> {
    println!("5. Role-Based Access Control (RBAC)");
    println!("------------------------------------");

    // Create access control manager
    let mut acl = AccessControlManager::new();

    // Create users
    println!("Creating users...");
    let admin = acl.create_user(
        "user1".to_string(),
        "alice".to_string(),
        "alice@example.com".to_string(),
    )?;
    println!("✓ Created admin user: {}", admin.username);

    let contributor = acl.create_user(
        "user2".to_string(),
        "bob".to_string(),
        "bob@example.com".to_string(),
    )?;
    println!("✓ Created contributor user: {}", contributor.username);

    // Grant roles
    println!("\nGranting roles...");
    acl.grant_role("user1", "admin")?;
    println!("✓ Granted 'admin' role to alice");

    acl.grant_role("user2", "contributor")?;
    println!("✓ Granted 'contributor' role to bob");

    // Create an organization
    println!("\nCreating organization...");
    let org = acl.create_organization("org1".to_string(), "ML Research Lab".to_string())?;
    println!("✓ Created organization: {}", org.name);

    // Add users to organization
    let org_roles = HashSet::from(["maintainer".to_string()]);
    acl.add_to_organization("user1", "org1", org_roles)?;
    println!("✓ Added alice to organization as maintainer");

    // Set up package ownership
    let mut ownership = PackageOwnership::new("ml-models".to_string());
    ownership.add_owner("user1".to_string());
    ownership.public_access = AccessLevel::Public;
    acl.set_package_ownership(ownership);
    println!("\n✓ Set up package 'ml-models' with alice as owner");

    // Test access control
    println!("\nTesting access control...");

    // Test 1: Admin can publish
    let publish_result = acl.check_access("user1", "ml-models", &Permission::Publish);
    println!("\nAlice (admin) publishing 'ml-models':");
    println!(
        "  Result: {}",
        if publish_result.granted {
            "✓ GRANTED"
        } else {
            "✗ DENIED"
        }
    );
    if let Some(reason) = &publish_result.denial_reason {
        println!("  Reason: {}", reason);
    }

    // Test 2: Contributor cannot publish
    let contrib_publish = acl.check_access("user2", "ml-models", &Permission::Publish);
    println!("\nBob (contributor) publishing 'ml-models':");
    println!(
        "  Result: {}",
        if contrib_publish.granted {
            "✓ GRANTED"
        } else {
            "✗ DENIED"
        }
    );
    if let Some(reason) = &contrib_publish.denial_reason {
        println!("  Reason: {}", reason);
    }

    // Test 3: Anyone can download public package
    let download_result = acl.check_access("user2", "ml-models", &Permission::Download);
    println!("\nBob downloading 'ml-models' (public package):");
    println!(
        "  Result: {}",
        if download_result.granted {
            "✓ GRANTED"
        } else {
            "✗ DENIED"
        }
    );

    // Test 4: Test deletion permission
    let delete_result = acl.check_access("user1", "ml-models", &Permission::Delete);
    println!("\nAlice (owner) deleting 'ml-models':");
    println!(
        "  Result: {}",
        if delete_result.granted {
            "✓ GRANTED"
        } else {
            "✗ DENIED"
        }
    );

    // Display role permissions
    println!("\nBuilt-in roles and their permissions:");
    let admin_role = Role::admin();
    println!("\n  Admin role:");
    println!("    Total permissions: {}", admin_role.permissions.len());

    let maintainer_role = Role::maintainer();
    println!("\n  Maintainer role:");
    println!(
        "    Can publish: {}",
        maintainer_role.has_permission(&Permission::Publish)
    );
    println!(
        "    Can delete: {}",
        maintainer_role.has_permission(&Permission::Delete)
    );

    let viewer_role = Role::viewer();
    println!("\n  Viewer role:");
    println!(
        "    Can read: {}",
        viewer_role.has_permission(&Permission::ReadMetadata)
    );
    println!(
        "    Can publish: {}",
        viewer_role.has_permission(&Permission::Publish)
    );

    println!();
    Ok(())
}
