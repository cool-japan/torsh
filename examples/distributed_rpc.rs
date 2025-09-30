//! Advanced Distributed RPC Framework Example
//!
//! This example demonstrates the RPC (Remote Procedure Call) framework
//! capabilities of ToRSh's distributed training system, including:
//! - Remote function calls across workers
//! - Remote references (RRef) for distributed objects
//! - Function registration and execution
//! - Error handling and timeouts

use serde::{Deserialize, Serialize};
use std::error::Error;
use std::time::Duration;
use torsh_distributed::{
    delete_rref, get_worker_rank, get_world_size, init_rpc, is_initialized as rpc_initialized,
    register_function, remote, rpc_async, shutdown as rpc_shutdown, RRef, RpcBackendOptions,
};

/// Input arguments for mathematical operations
#[derive(Debug, Clone, Serialize, Deserialize)]
struct MathArgs {
    a: f64,
    b: f64,
}

/// Result of mathematical operations
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct MathResult {
    value: f64,
    operation: String,
}

/// Arguments for matrix operations
#[derive(Debug, Clone, Serialize, Deserialize)]
struct MatrixArgs {
    rows: usize,
    cols: usize,
    fill_value: f64,
}

/// Matrix data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl Matrix {
    fn new(rows: usize, cols: usize, fill_value: f64) -> Self {
        Self {
            rows,
            cols,
            data: vec![fill_value; rows * cols],
        }
    }

    fn transpose(&self) -> Self {
        let mut transposed_data = vec![0.0; self.data.len()];
        for i in 0..self.rows {
            for j in 0..self.cols {
                transposed_data[j * self.rows + i] = self.data[i * self.cols + j];
            }
        }
        Self {
            rows: self.cols,
            cols: self.rows,
            data: transposed_data,
        }
    }

    fn scale(&mut self, factor: f64) {
        for value in &mut self.data {
            *value *= factor;
        }
    }
}

/// Remote function implementations
fn add_numbers(args: MathArgs) -> Result<MathResult, String> {
    Ok(MathResult {
        value: args.a + args.b,
        operation: format!("{} + {} = {}", args.a, args.b, args.a + args.b),
    })
}

fn multiply_numbers(args: MathArgs) -> Result<MathResult, String> {
    Ok(MathResult {
        value: args.a * args.b,
        operation: format!("{} * {} = {}", args.a, args.b, args.a * args.b),
    })
}

fn power_function(args: MathArgs) -> Result<MathResult, String> {
    let result = args.a.powf(args.b);
    Ok(MathResult {
        value: result,
        operation: format!("{}^{} = {}", args.a, args.b, result),
    })
}

fn create_matrix(args: MatrixArgs) -> Result<Matrix, String> {
    if args.rows == 0 || args.cols == 0 {
        return Err("Matrix dimensions must be positive".to_string());
    }

    if args.rows > 1000 || args.cols > 1000 {
        return Err("Matrix too large (max 1000x1000)".to_string());
    }

    Ok(Matrix::new(args.rows, args.cols, args.fill_value))
}

fn factorial(n: u64) -> Result<u64, String> {
    if n > 20 {
        return Err("Factorial too large (max 20!)".to_string());
    }

    let mut result = 1;
    for i in 1..=n {
        result *= i;
    }
    Ok(result)
}

/// Demonstrate basic RPC function calls
async fn demo_basic_rpc_calls(rank: u32, world_size: u32) -> Result<(), Box<dyn Error>> {
    println!("\n🔧 Basic RPC Function Calls Demo (Worker {})", rank);
    println!("===========================================");

    if rank == 0 {
        // Worker 0 calls functions on other workers
        for target_rank in 1..world_size {
            println!("\n📞 Calling functions on worker {}", target_rank);

            // Test addition
            let add_args = MathArgs { a: 10.5, b: 5.3 };
            match rpc_async::<MathArgs, MathResult>(target_rank, "add", add_args).await {
                Ok(result) => {
                    println!("  ✅ Addition: {}", result.operation);
                    assert_eq!(result.value, 15.8);
                }
                Err(e) => {
                    eprintln!("  ❌ Addition failed: {}", e);
                }
            }

            // Test multiplication
            let mul_args = MathArgs { a: 3.0, b: 7.0 };
            match rpc_async::<MathArgs, MathResult>(target_rank, "multiply", mul_args).await {
                Ok(result) => {
                    println!("  ✅ Multiplication: {}", result.operation);
                    assert_eq!(result.value, 21.0);
                }
                Err(e) => {
                    eprintln!("  ❌ Multiplication failed: {}", e);
                }
            }

            // Test power function
            let pow_args = MathArgs { a: 2.0, b: 8.0 };
            match rpc_async::<MathArgs, MathResult>(target_rank, "power", pow_args).await {
                Ok(result) => {
                    println!("  ✅ Power: {}", result.operation);
                    assert_eq!(result.value, 256.0);
                }
                Err(e) => {
                    eprintln!("  ❌ Power failed: {}", e);
                }
            }

            // Test factorial
            match rpc_async::<u64, u64>(target_rank, "factorial", 5).await {
                Ok(result) => {
                    println!("  ✅ Factorial: 5! = {}", result);
                    assert_eq!(result, 120);
                }
                Err(e) => {
                    eprintln!("  ❌ Factorial failed: {}", e);
                }
            }
        }
    } else {
        println!("  Worker {} ready to handle RPC calls", rank);
    }

    Ok(())
}

/// Demonstrate remote references (RRef)
async fn demo_remote_references(rank: u32, world_size: u32) -> Result<(), Box<dyn Error>> {
    println!("\n🔗 Remote References (RRef) Demo (Worker {})", rank);
    println!("========================================");

    if rank == 0 && world_size > 1 {
        let target_rank = 1;
        println!("\\n📋 Creating remote matrices on worker {}", target_rank);

        // Create a remote matrix
        let matrix_args = MatrixArgs {
            rows: 3,
            cols: 4,
            fill_value: 2.5,
        };

        match remote::<MatrixArgs, Matrix>(target_rank, "create_matrix", matrix_args).await {
            Ok(matrix_rref) => {
                println!(
                    "  ✅ Created remote matrix with RRef ID: {}",
                    matrix_rref.id()
                );
                println!("  📍 Matrix located on worker {}", matrix_rref.owner_rank());

                // Create another remote matrix
                let matrix_args2 = MatrixArgs {
                    rows: 2,
                    cols: 2,
                    fill_value: 1.0,
                };

                match remote::<MatrixArgs, Matrix>(target_rank, "create_matrix", matrix_args2).await
                {
                    Ok(matrix_rref2) => {
                        println!(
                            "  ✅ Created second remote matrix with RRef ID: {}",
                            matrix_rref2.id()
                        );

                        // Clean up remote references
                        println!("\\n🗑️  Cleaning up remote references...");
                        delete_rref(matrix_rref).await?;
                        delete_rref(matrix_rref2).await?;
                        println!("  ✅ Remote references deleted");
                    }
                    Err(e) => {
                        eprintln!("  ❌ Failed to create second matrix: {}", e);
                    }
                }
            }
            Err(e) => {
                eprintln!("  ❌ Failed to create remote matrix: {}", e);
            }
        }
    } else if rank == 1 {
        println!("  Worker {} ready to create remote objects", rank);
    }

    Ok(())
}

/// Demonstrate error handling and edge cases
async fn demo_error_handling(rank: u32, world_size: u32) -> Result<(), Box<dyn Error>> {
    println!("\\n⚠️  Error Handling Demo (Worker {})", rank);
    println!("===========================");

    if rank == 0 && world_size > 1 {
        let target_rank = 1;

        println!("\\n🧪 Testing error conditions...");

        // Test calling non-existent function
        match rpc_async::<MathArgs, MathResult>(
            target_rank,
            "nonexistent",
            MathArgs { a: 1.0, b: 2.0 },
        )
        .await
        {
            Ok(_) => {
                eprintln!("  ❌ Should have failed for non-existent function");
            }
            Err(e) => {
                println!(
                    "  ✅ Correctly caught error for non-existent function: {}",
                    e
                );
            }
        }

        // Test invalid matrix creation
        let invalid_matrix_args = MatrixArgs {
            rows: 0,
            cols: 5,
            fill_value: 1.0,
        };

        match rpc_async::<MatrixArgs, Matrix>(target_rank, "create_matrix", invalid_matrix_args)
            .await
        {
            Ok(_) => {
                eprintln!("  ❌ Should have failed for invalid matrix dimensions");
            }
            Err(e) => {
                println!("  ✅ Correctly caught error for invalid matrix: {}", e);
            }
        }

        // Test factorial with too large number
        match rpc_async::<u64, u64>(target_rank, "factorial", 25).await {
            Ok(_) => {
                eprintln!("  ❌ Should have failed for large factorial");
            }
            Err(e) => {
                println!("  ✅ Correctly caught error for large factorial: {}", e);
            }
        }

        // Test calling invalid worker rank
        match rpc_async::<MathArgs, MathResult>(999, "add", MathArgs { a: 1.0, b: 2.0 }).await {
            Ok(_) => {
                eprintln!("  ❌ Should have failed for invalid worker rank");
            }
            Err(e) => {
                println!("  ✅ Correctly caught error for invalid worker rank: {}", e);
            }
        }
    }

    Ok(())
}

/// Demonstrate concurrent RPC calls
async fn demo_concurrent_calls(rank: u32, world_size: u32) -> Result<(), Box<dyn Error>> {
    println!("\\n🚀 Concurrent RPC Calls Demo (Worker {})", rank);
    println!("====================================");

    if rank == 0 && world_size > 1 {
        println!("\\n⚡ Making concurrent calls to multiple workers...");

        let mut handles = Vec::new();

        // Create concurrent RPC calls
        for target_rank in 1..world_size {
            for i in 0..3 {
                let args = MathArgs {
                    a: (target_rank * 10 + i) as f64,
                    b: (i + 1) as f64,
                };

                let handle = tokio::spawn(async move {
                    let result = rpc_async::<MathArgs, MathResult>(target_rank, "add", args).await;
                    (target_rank, i, result)
                });

                handles.push(handle);
            }
        }

        // Wait for all calls to complete
        let start_time = std::time::Instant::now();
        for handle in handles {
            match handle.await? {
                (target_rank, call_id, Ok(result)) => {
                    println!(
                        "  ✅ Worker {} call {}: {}",
                        target_rank, call_id, result.operation
                    );
                }
                (target_rank, call_id, Err(e)) => {
                    eprintln!("  ❌ Worker {} call {} failed: {}", target_rank, call_id, e);
                }
            }
        }

        let elapsed = start_time.elapsed();
        println!("  ⏱️  All concurrent calls completed in {:?}", elapsed);
    }

    Ok(())
}

/// Initialize RPC worker
async fn init_worker(rank: u32, world_size: u32) -> Result<(), Box<dyn Error>> {
    let worker_name = format!("rpc_worker_{}", rank);

    let options = RpcBackendOptions {
        num_worker_threads: 4,
        rpc_timeout: Duration::from_secs(30),
        init_method: "tcp://".to_string(),
        buffer_size: 8192,
        max_connections: 50,
    };

    println!("[Worker {}] Initializing RPC framework...", rank);
    init_rpc(&worker_name, rank, world_size, options).await?;

    // Register functions
    register_function("add", add_numbers).await?;
    register_function("multiply", multiply_numbers).await?;
    register_function("power", power_function).await?;
    register_function("create_matrix", create_matrix).await?;
    register_function("factorial", factorial).await?;

    println!("[Worker {}] RPC framework initialized successfully", rank);
    println!(
        "[Worker {}] Registered functions: add, multiply, power, create_matrix, factorial",
        rank
    );

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("🌐 ToRSh Distributed RPC Framework Demo");
    println!("======================================");

    // For demo purposes, simulate different ranks
    let world_size = 3;
    let mut worker_handles = Vec::new();

    // Start multiple workers
    for rank in 0..world_size {
        let handle = tokio::spawn(async move {
            if let Err(e) = run_worker(rank, world_size).await {
                eprintln!("[Worker {}] Error: {}", rank, e);
            }
        });
        worker_handles.push(handle);
    }

    // Wait a moment for all workers to initialize
    tokio::time::sleep(Duration::from_millis(1000)).await;

    // Wait for all workers to complete
    for handle in worker_handles {
        handle.await?;
    }

    println!("\\n🏁 RPC Demo Completed Successfully!");
    println!("\\n📚 Key Features Demonstrated:");
    println!("  ✅ Remote function calls with type safety");
    println!("  ✅ Remote references (RRef) for distributed objects");
    println!("  ✅ Function registration and dynamic dispatch");
    println!("  ✅ Error handling and validation");
    println!("  ✅ Concurrent RPC calls");
    println!("  ✅ Worker discovery and communication");
    println!("  ✅ Proper cleanup and shutdown");

    println!("\\n🔄 Production Use Cases:");
    println!("  - Distributed model parameter servers");
    println!("  - Remote dataset loading and preprocessing");
    println!("  - Distributed checkpointing and state management");
    println!("  - Cross-worker synchronization and coordination");
    println!("  - Remote function execution for specialized hardware");

    Ok(())
}

/// Run a single worker
async fn run_worker(rank: u32, world_size: u32) -> Result<(), Box<dyn Error>> {
    // Initialize RPC framework
    init_worker(rank, world_size).await?;

    // Verify initialization
    assert!(rpc_initialized());
    assert_eq!(get_worker_rank()?, rank);
    assert_eq!(get_world_size()?, world_size);

    // Wait for all workers to be ready
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Run demos (only worker 0 initiates calls to avoid conflicts)
    demo_basic_rpc_calls(rank, world_size).await?;

    tokio::time::sleep(Duration::from_millis(200)).await;
    demo_remote_references(rank, world_size).await?;

    tokio::time::sleep(Duration::from_millis(200)).await;
    demo_error_handling(rank, world_size).await?;

    tokio::time::sleep(Duration::from_millis(200)).await;
    demo_concurrent_calls(rank, world_size).await?;

    // Wait a bit before shutdown
    tokio::time::sleep(Duration::from_millis(300)).await;

    // Shutdown RPC framework
    println!("[Worker {}] Shutting down RPC framework...", rank);
    rpc_shutdown().await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_math_operations() {
        let args = MathArgs { a: 5.0, b: 3.0 };

        let add_result = add_numbers(args.clone()).unwrap();
        assert_eq!(add_result.value, 8.0);

        let mul_result = multiply_numbers(args.clone()).unwrap();
        assert_eq!(mul_result.value, 15.0);

        let pow_result = power_function(args).unwrap();
        assert_eq!(pow_result.value, 125.0);
    }

    #[test]
    fn test_matrix_operations() {
        let args = MatrixArgs {
            rows: 2,
            cols: 3,
            fill_value: 1.5,
        };

        let matrix = create_matrix(args).unwrap();
        assert_eq!(matrix.rows, 2);
        assert_eq!(matrix.cols, 3);
        assert_eq!(matrix.data.len(), 6);
        assert!(matrix.data.iter().all(|&x| x == 1.5));

        let transposed = matrix.transpose();
        assert_eq!(transposed.rows, 3);
        assert_eq!(transposed.cols, 2);
    }

    #[test]
    fn test_factorial() {
        assert_eq!(factorial(0).unwrap(), 1);
        assert_eq!(factorial(1).unwrap(), 1);
        assert_eq!(factorial(5).unwrap(), 120);
        assert_eq!(factorial(10).unwrap(), 3628800);

        // Test error case
        assert!(factorial(25).is_err());
    }

    #[test]
    fn test_error_cases() {
        // Invalid matrix dimensions
        let invalid_args = MatrixArgs {
            rows: 0,
            cols: 5,
            fill_value: 1.0,
        };
        assert!(create_matrix(invalid_args).is_err());

        // Too large matrix
        let large_args = MatrixArgs {
            rows: 2000,
            cols: 2000,
            fill_value: 1.0,
        };
        assert!(create_matrix(large_args).is_err());
    }
}
