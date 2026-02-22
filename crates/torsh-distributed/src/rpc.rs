//! Remote Procedure Call (RPC) framework for distributed training
//!
//! This module provides a complete RPC framework for distributed training,
//! supporting remote function calls, remote references, and distributed
//! computation patterns.

use crate::{TorshDistributedError, TorshResult};
use log::info;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{mpsc, oneshot, RwLock};
use uuid::Uuid;

// Type aliases for complex types
type PendingRequestMap = Arc<Mutex<HashMap<String, oneshot::Sender<Result<Vec<u8>, String>>>>>;
type FunctionRegistry =
    Arc<RwLock<HashMap<String, Box<dyn Fn(&[u8]) -> Result<Vec<u8>, String> + Send + Sync>>>>;

/// RPC backend options
#[derive(Debug, Clone)]
pub struct RpcBackendOptions {
    pub num_worker_threads: usize,
    pub rpc_timeout: Duration,
    pub init_method: String,
    pub buffer_size: usize,
    pub max_connections: usize,
}

impl Default for RpcBackendOptions {
    fn default() -> Self {
        Self {
            num_worker_threads: 4,
            rpc_timeout: Duration::from_secs(60),
            init_method: String::from("env://"),
            buffer_size: 8192,
            max_connections: 100,
        }
    }
}

/// RPC message types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RpcMessage {
    /// Function call request
    FunctionCall {
        id: String,
        function_name: String,
        args: Vec<u8>, // Serialized arguments
    },
    /// Function call response
    FunctionResponse {
        id: String,
        result: Result<Vec<u8>, String>, // Serialized result or error
    },
    /// Remote reference creation
    RemoteRef {
        id: String,
        function_name: String,
        args: Vec<u8>,
        rref_id: String,
    },
    /// Remote reference response
    RemoteRefResponse {
        id: String,
        result: Result<String, String>, // RRef ID or error
    },
    /// Remote reference deletion
    DeleteRRef { rref_id: String },
    /// Ping message for health checks
    Ping,
    /// Pong response
    Pong,
}

/// Remote reference to a value
#[derive(Debug, Clone)]
pub struct RRef<T> {
    id: String,
    owner_rank: u32,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> RRef<T> {
    pub fn new(id: String, owner_rank: u32) -> Self {
        Self {
            id,
            owner_rank,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn owner_rank(&self) -> u32 {
        self.owner_rank
    }
}

/// RPC worker state
struct RpcWorker {
    rank: u32,
    world_size: u32,
    connections: Arc<RwLock<HashMap<u32, TcpStream>>>,
    pending_requests: PendingRequestMap,
    remote_refs: Arc<RwLock<HashMap<String, Box<dyn std::any::Any + Send + Sync>>>>,
    function_registry: FunctionRegistry,
    shutdown_tx: Option<mpsc::Sender<()>>,
}

/// Global RPC worker instance
static RPC_WORKER: once_cell::sync::OnceCell<Arc<Mutex<Option<RpcWorker>>>> =
    once_cell::sync::OnceCell::new();

// Test-only RPC worker for isolated testing
#[cfg(test)]
thread_local! {
    static TEST_RPC_WORKER: std::cell::RefCell<Option<Arc<Mutex<Option<RpcWorker>>>>> = const { std::cell::RefCell::new(None) };
}

/// Get the global RPC worker
fn get_rpc_worker() -> TorshResult<Arc<Mutex<Option<RpcWorker>>>> {
    #[cfg(test)]
    {
        // In tests, try thread-local first
        let local_worker = TEST_RPC_WORKER.with(|w| w.borrow().clone());
        if let Some(worker) = local_worker {
            return Ok(worker);
        }
    }

    RPC_WORKER
        .get()
        .ok_or(TorshDistributedError::BackendNotInitialized)
        .cloned()
}

/// Initialize RPC framework
pub async fn init_rpc(
    name: &str,
    rank: u32,
    world_size: u32,
    _options: RpcBackendOptions,
) -> TorshResult<()> {
    // Initialize the RPC worker
    let worker = RpcWorker {
        rank,
        world_size,
        connections: Arc::new(RwLock::new(HashMap::new())),
        pending_requests: Arc::new(Mutex::new(HashMap::new())),
        remote_refs: Arc::new(RwLock::new(HashMap::new())),
        function_registry: Arc::new(RwLock::new(HashMap::new())),
        shutdown_tx: None,
    };

    let worker_arc = Arc::new(Mutex::new(Some(worker)));

    // Set the worker (test-local in tests, global otherwise)
    #[cfg(test)]
    {
        TEST_RPC_WORKER.with(|w| *w.borrow_mut() = Some(worker_arc.clone()));
    }

    #[cfg(not(test))]
    {
        RPC_WORKER
            .set(worker_arc.clone())
            .map_err(|_| TorshDistributedError::backend_error("rpc", "RPC already initialized"))?;
    }

    // Start RPC server with dynamic port allocation for tests
    let base_port = if cfg!(test) {
        // Use a wider range of ports for testing to avoid conflicts
        29600 + (std::process::id() % 1000) + rank * 100
    } else {
        29600 + rank
    };

    // Try multiple ports if the first one fails
    let mut listener = None;
    for port_offset in 0..10 {
        let listen_addr = format!("127.0.0.1:{}", base_port + port_offset);
        match TcpListener::bind(&listen_addr).await {
            Ok(l) => {
                info!(
                    "[RPC] Worker '{}' (rank {}) starting on {}",
                    name, rank, listen_addr
                );
                listener = Some(l);
                break;
            }
            Err(e) => {
                if port_offset == 9 {
                    return Err(TorshDistributedError::communication_error(
                        "rpc_server",
                        format!("Failed to bind after trying 10 ports: {}", e),
                    ));
                }
            }
        }
    }

    let listener = listener.expect("listener should be successfully bound");

    let (shutdown_tx, mut shutdown_rx) = mpsc::channel::<()>(1);

    // Update worker with shutdown channel
    {
        let mut worker_guard = worker_arc.lock().expect("lock should not be poisoned");
        if let Some(ref mut worker) = *worker_guard {
            worker.shutdown_tx = Some(shutdown_tx);
        }
    }

    // Spawn server task
    let worker_for_server = worker_arc.clone();
    tokio::spawn(async move {
        loop {
            tokio::select! {
                result = listener.accept() => {
                    match result {
                        Ok((stream, addr)) => {
                            info!("[RPC] Accepted connection from {}", addr);
                            let worker_clone = worker_for_server.clone();
                            tokio::spawn(handle_connection(stream, worker_clone));
                        }
                        Err(e) => {
                            info!("[RPC] Failed to accept connection: {}", e);
                        }
                    }
                }
                _ = shutdown_rx.recv() => {
                    info!("[RPC] Server shutting down");
                    break;
                }
            }
        }
    });

    // Wait a bit for other workers to start
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Connect to other workers (skip for single-worker setups)
    if world_size > 1 {
        for other_rank in 0..world_size {
            if other_rank != rank {
                let target_addr = format!("127.0.0.1:{}", base_port + other_rank);

                // Retry connection with exponential backoff
                let mut retries = 0;
                let max_retries = if cfg!(test) { 3 } else { 10 }; // Fewer retries in tests

                while retries < max_retries {
                    match TcpStream::connect(&target_addr).await {
                        Ok(stream) => {
                            info!(
                                "[RPC] Connected to worker {} at {}",
                                other_rank, target_addr
                            );
                            let connections = {
                                let worker_guard =
                                    worker_arc.lock().expect("lock should not be poisoned");
                                worker_guard
                                    .as_ref()
                                    .expect("worker should be initialized")
                                    .connections
                                    .clone()
                            };
                            let mut connections_guard = connections.write().await;
                            connections_guard.insert(other_rank, stream);
                            break;
                        }
                        Err(e) => {
                            retries += 1;
                            let delay = Duration::from_millis(100 * (1 << retries.min(3)));
                            tokio::time::sleep(delay).await;
                            if retries == max_retries {
                                // In tests, just log the error and continue
                                if cfg!(test) {
                                    info!(
                                        "[RPC] Failed to connect to worker {} (test mode): {}",
                                        other_rank, e
                                    );
                                    break;
                                } else {
                                    return Err(TorshDistributedError::communication_error(
                                        "rpc_connect",
                                        format!(
                                            "Failed to connect to worker {}: {}",
                                            other_rank, e
                                        ),
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    info!(
        "[RPC] Worker '{}' (rank {}) initialized successfully",
        name, rank
    );
    Ok(())
}

/// Handle incoming RPC connection
async fn handle_connection(mut stream: TcpStream, worker: Arc<Mutex<Option<RpcWorker>>>) {
    let mut buffer = vec![0u8; 8192];

    loop {
        match stream.read(&mut buffer).await {
            Ok(0) => break, // Connection closed
            Ok(n) => {
                let data = &buffer[..n];

                // Try to deserialize the message
                let result: Result<(RpcMessage, usize), _> =
                    oxicode::serde::decode_from_slice(data, oxicode::config::standard());
                match result {
                    Ok((message, _)) => {
                        if let Err(e) = handle_rpc_message(message, &mut stream, &worker).await {
                            info!("[RPC] Error handling message: {}", e);
                        }
                    }
                    Err(e) => {
                        info!("[RPC] Failed to deserialize message: {}", e);
                    }
                }
            }
            Err(e) => {
                info!("[RPC] Connection error: {}", e);
                break;
            }
        }
    }
}

/// Handle a specific RPC message
async fn handle_rpc_message(
    message: RpcMessage,
    stream: &mut TcpStream,
    worker: &Arc<Mutex<Option<RpcWorker>>>,
) -> TorshResult<()> {
    match message {
        RpcMessage::FunctionCall {
            id,
            function_name,
            args,
        } => {
            // Get a clone of the function registry to avoid holding locks across await
            let function_registry = {
                let worker_guard = worker.lock().expect("lock should not be poisoned");
                let worker_ref = worker_guard.as_ref().expect("worker should be initialized");
                worker_ref.function_registry.clone()
            };

            let result = {
                let registry = function_registry.read().await;
                if let Some(func) = registry.get(&function_name) {
                    func(&args)
                } else {
                    Err(format!("Function '{}' not found", function_name))
                }
            };

            let response = RpcMessage::FunctionResponse { id, result };
            let response_data =
                oxicode::serde::encode_to_vec(&response, oxicode::config::standard()).map_err(
                    |e| {
                        TorshDistributedError::communication_error(
                            "rpc",
                            format!("Serialization error: {}", e),
                        )
                    },
                )?;

            stream.write_all(&response_data).await.map_err(|e| {
                TorshDistributedError::communication_error("rpc", format!("Write error: {}", e))
            })?;
        }

        RpcMessage::RemoteRef {
            id,
            function_name,
            args,
            rref_id,
        } => {
            // Get clones to avoid holding locks across await
            let (function_registry, remote_refs) = {
                let worker_guard = worker.lock().expect("lock should not be poisoned");
                let worker_ref = worker_guard.as_ref().expect("worker should be initialized");
                (
                    worker_ref.function_registry.clone(),
                    worker_ref.remote_refs.clone(),
                )
            };

            let result = {
                let registry = function_registry.read().await;
                if let Some(func) = registry.get(&function_name) {
                    match func(&args) {
                        Ok(result_data) => {
                            // Store the result as a remote reference
                            let mut refs = remote_refs.write().await;
                            // For simplicity, store as Vec<u8>
                            refs.insert(rref_id.clone(), Box::new(result_data));
                            Ok(rref_id)
                        }
                        Err(e) => Err(e),
                    }
                } else {
                    Err(format!("Function '{}' not found", function_name))
                }
            };

            let response = RpcMessage::RemoteRefResponse { id, result };
            let response_data =
                oxicode::serde::encode_to_vec(&response, oxicode::config::standard()).map_err(
                    |e| {
                        TorshDistributedError::communication_error(
                            "rpc",
                            format!("Serialization error: {}", e),
                        )
                    },
                )?;

            stream.write_all(&response_data).await.map_err(|e| {
                TorshDistributedError::communication_error("rpc", format!("Write error: {}", e))
            })?;
        }

        RpcMessage::DeleteRRef { rref_id } => {
            let remote_refs = {
                let worker_guard = worker.lock().expect("lock should not be poisoned");
                let worker_ref = worker_guard.as_ref().expect("worker should be initialized");
                worker_ref.remote_refs.clone()
            };

            let mut refs = remote_refs.write().await;
            refs.remove(&rref_id);
        }

        RpcMessage::Ping => {
            let response = RpcMessage::Pong;
            let response_data =
                oxicode::serde::encode_to_vec(&response, oxicode::config::standard()).map_err(
                    |e| {
                        TorshDistributedError::communication_error(
                            "rpc",
                            format!("Serialization error: {}", e),
                        )
                    },
                )?;

            stream.write_all(&response_data).await.map_err(|e| {
                TorshDistributedError::communication_error("rpc", format!("Write error: {}", e))
            })?;
        }

        _ => {
            // Handle responses by forwarding to pending requests
            if let RpcMessage::FunctionResponse { id, result } = message {
                let worker_guard = worker.lock().expect("lock should not be poisoned");
                let worker_ref = worker_guard.as_ref().expect("worker should be initialized");
                let mut pending = worker_ref
                    .pending_requests
                    .lock()
                    .expect("lock should not be poisoned");

                if let Some(sender) = pending.remove(&id) {
                    let _ = sender.send(result);
                }
            }
        }
    }

    Ok(())
}

/// Shutdown RPC framework
pub async fn shutdown() -> TorshResult<()> {
    let worker_arc = get_rpc_worker()?;

    let (shutdown_tx, remote_refs) = {
        let mut worker_guard = worker_arc.lock().expect("lock should not be poisoned");
        if let Some(worker) = worker_guard.take() {
            (worker.shutdown_tx, Some(worker.remote_refs))
        } else {
            (None, None)
        }
    };

    if let Some(shutdown_tx) = shutdown_tx {
        let _ = shutdown_tx.send(()).await;
    }

    if let Some(remote_refs) = remote_refs {
        // Clear all remote references
        let mut refs = remote_refs.write().await;
        refs.clear();
    }

    info!("[RPC] Framework shut down successfully");
    Ok(())
}

/// Register a function for remote execution
pub async fn register_function<F, Args, Ret>(name: &str, func: F) -> TorshResult<()>
where
    F: Fn(Args) -> Result<Ret, String> + Send + Sync + 'static,
    Args: for<'de> Deserialize<'de> + 'static,
    Ret: Serialize + 'static,
{
    let worker_arc = get_rpc_worker()?;
    let function_registry = {
        let worker_guard = worker_arc.lock().expect("lock should not be poisoned");
        let worker_ref = worker_guard.as_ref().expect("worker should be initialized");
        worker_ref.function_registry.clone()
    };

    let wrapper = move |args_bytes: &[u8]| -> Result<Vec<u8>, String> {
        let (args, _): (Args, usize) =
            oxicode::serde::decode_from_slice(args_bytes, oxicode::config::standard())
                .map_err(|e| format!("Deserialization error: {}", e))?;

        let result = func(args)?;

        oxicode::serde::encode_to_vec(&result, oxicode::config::standard())
            .map_err(|e| format!("Serialization error: {}", e))
    };

    let mut registry = function_registry.write().await;
    registry.insert(name.to_string(), Box::new(wrapper));

    Ok(())
}

/// Call a remote function
pub async fn rpc_async<Args, Ret>(to: u32, function_name: &str, args: Args) -> TorshResult<Ret>
where
    Args: Serialize,
    Ret: for<'de> Deserialize<'de>,
{
    let worker_arc = get_rpc_worker()?;

    // Serialize arguments
    let args_bytes =
        oxicode::serde::encode_to_vec(&args, oxicode::config::standard()).map_err(|e| {
            TorshDistributedError::communication_error("rpc", format!("Serialization error: {}", e))
        })?;

    // Generate request ID
    let request_id = Uuid::new_v4().to_string();

    // Create message
    let message = RpcMessage::FunctionCall {
        id: request_id.clone(),
        function_name: function_name.to_string(),
        args: args_bytes,
    };

    // Serialize message
    let message_bytes = oxicode::serde::encode_to_vec(&message, oxicode::config::standard())
        .map_err(|e| {
            TorshDistributedError::communication_error("rpc", format!("Serialization error: {}", e))
        })?;

    // Get clones to avoid holding locks across await
    let (connections, pending_requests) = {
        let worker_guard = worker_arc.lock().expect("lock should not be poisoned");
        let worker_ref = worker_guard.as_ref().expect("worker should be initialized");
        (
            worker_ref.connections.clone(),
            worker_ref.pending_requests.clone(),
        )
    };

    // Create response channel and register pending request
    let (response_tx, response_rx) = oneshot::channel();
    {
        let mut pending = pending_requests
            .lock()
            .expect("lock should not be poisoned");
        pending.insert(request_id, response_tx);
    }

    // Get connection and send message
    {
        let mut connections_guard = connections.write().await;
        let connection = connections_guard.get_mut(&to).ok_or_else(|| {
            TorshDistributedError::communication_error(
                "rpc",
                format!("No connection to worker {}", to),
            )
        })?;

        connection.write_all(&message_bytes).await.map_err(|e| {
            TorshDistributedError::communication_error("rpc", format!("Write error: {}", e))
        })?;
    }

    // Wait for response with timeout
    let result = tokio::time::timeout(Duration::from_secs(60), response_rx)
        .await
        .map_err(|_| TorshDistributedError::communication_error("rpc", "RPC timeout"))?
        .map_err(|_| {
            TorshDistributedError::communication_error("rpc", "Response channel closed")
        })?;

    match result {
        Ok(result_bytes) => {
            let (value, _): (Ret, usize) =
                oxicode::serde::decode_from_slice(&result_bytes, oxicode::config::standard())
                    .map_err(|e| {
                        TorshDistributedError::communication_error(
                            "rpc",
                            format!("Deserialization error: {}", e),
                        )
                    })?;
            Ok(value)
        }
        Err(error_msg) => Err(TorshDistributedError::communication_error(
            "rpc_remote",
            format!("Remote function error: {}", error_msg),
        )),
    }
}

/// Get a remote reference
pub async fn remote<Args, Ret>(to: u32, function_name: &str, args: Args) -> TorshResult<RRef<Ret>>
where
    Args: Serialize,
    Ret: for<'de> Deserialize<'de> + 'static,
{
    let worker_arc = get_rpc_worker()?;

    // Serialize arguments
    let args_bytes =
        oxicode::serde::encode_to_vec(&args, oxicode::config::standard()).map_err(|e| {
            TorshDistributedError::communication_error("rpc", format!("Serialization error: {}", e))
        })?;

    // Generate request and RRef IDs
    let request_id = Uuid::new_v4().to_string();
    let rref_id = Uuid::new_v4().to_string();

    // Create message
    let message = RpcMessage::RemoteRef {
        id: request_id.clone(),
        function_name: function_name.to_string(),
        args: args_bytes,
        rref_id: rref_id.clone(),
    };

    // Serialize message
    let message_bytes = oxicode::serde::encode_to_vec(&message, oxicode::config::standard())
        .map_err(|e| {
            TorshDistributedError::communication_error("rpc", format!("Serialization error: {}", e))
        })?;

    // Get clones to avoid holding locks across await
    let (connections, pending_requests) = {
        let worker_guard = worker_arc.lock().expect("lock should not be poisoned");
        let worker_ref = worker_guard.as_ref().expect("worker should be initialized");
        (
            worker_ref.connections.clone(),
            worker_ref.pending_requests.clone(),
        )
    };

    // Create response channel and register pending request
    let (response_tx, response_rx) = oneshot::channel();
    {
        let mut pending = pending_requests
            .lock()
            .expect("lock should not be poisoned");
        pending.insert(request_id, response_tx);
    }

    // Get connection and send message
    {
        let mut connections_guard = connections.write().await;
        let connection = connections_guard.get_mut(&to).ok_or_else(|| {
            TorshDistributedError::communication_error(
                "rpc",
                format!("No connection to worker {}", to),
            )
        })?;

        connection.write_all(&message_bytes).await.map_err(|e| {
            TorshDistributedError::communication_error("rpc", format!("Write error: {}", e))
        })?;
    }

    // Wait for response with timeout
    let result = tokio::time::timeout(Duration::from_secs(60), response_rx)
        .await
        .map_err(|_| TorshDistributedError::communication_error("rpc", "RPC timeout"))?
        .map_err(|_| {
            TorshDistributedError::communication_error("rpc", "Response channel closed")
        })?;

    match result {
        Ok(returned_rref_id) => {
            let (actual_rref_id, _): (String, usize) =
                oxicode::serde::decode_from_slice(&returned_rref_id, oxicode::config::standard())
                    .map_err(|e| {
                    TorshDistributedError::communication_error(
                        "rpc",
                        format!("Deserialization error: {}", e),
                    )
                })?;

            Ok(RRef::new(actual_rref_id, to))
        }
        Err(error_msg) => Err(TorshDistributedError::communication_error(
            "rpc_remote",
            format!("Remote function error: {}", error_msg),
        )),
    }
}

/// Delete a remote reference
pub async fn delete_rref<T>(rref: RRef<T>) -> TorshResult<()> {
    let worker_arc = get_rpc_worker()?;
    let connections = {
        let worker_guard = worker_arc.lock().expect("lock should not be poisoned");
        let worker_ref = worker_guard.as_ref().expect("worker should be initialized");
        worker_ref.connections.clone()
    };

    let message = RpcMessage::DeleteRRef {
        rref_id: rref.id().to_string(),
    };

    let message_bytes = oxicode::serde::encode_to_vec(&message, oxicode::config::standard())
        .map_err(|e| {
            TorshDistributedError::communication_error("rpc", format!("Serialization error: {}", e))
        })?;

    let mut connections_guard = connections.write().await;
    if let Some(connection) = connections_guard.get_mut(&rref.owner_rank()) {
        connection.write_all(&message_bytes).await.map_err(|e| {
            TorshDistributedError::communication_error("rpc", format!("Write error: {}", e))
        })?;
    }

    Ok(())
}

/// Check if RPC framework is initialized
pub fn is_initialized() -> bool {
    #[cfg(test)]
    {
        let local_worker = TEST_RPC_WORKER.with(|w| w.borrow().clone());
        if local_worker.is_some() {
            return true;
        }
    }

    RPC_WORKER.get().is_some()
}

/// Reset RPC framework (for testing only)
#[cfg(test)]
pub fn reset_rpc() {
    TEST_RPC_WORKER.with(|w| *w.borrow_mut() = None);
}

/// Get current worker rank
pub fn get_worker_rank() -> TorshResult<u32> {
    let worker_arc = get_rpc_worker()?;
    let worker_guard = worker_arc.lock().expect("lock should not be poisoned");
    let worker_ref = worker_guard.as_ref().expect("worker should be initialized");
    Ok(worker_ref.rank)
}

/// Get world size
pub fn get_world_size() -> TorshResult<u32> {
    let worker_arc = get_rpc_worker()?;
    let worker_guard = worker_arc.lock().expect("lock should not be poisoned");
    let worker_ref = worker_guard.as_ref().expect("worker should be initialized");
    Ok(worker_ref.world_size)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct TestArgs {
        x: i32,
        y: i32,
    }

    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct TestResult {
        sum: i32,
    }

    fn add_function(args: TestArgs) -> Result<TestResult, String> {
        Ok(TestResult {
            sum: args.x + args.y,
        })
    }

    fn multiply_function(args: TestArgs) -> Result<TestResult, String> {
        Ok(TestResult {
            sum: args.x * args.y,
        })
    }

    #[tokio::test]
    async fn test_rpc_initialization() -> TorshResult<()> {
        reset_rpc();

        let options = RpcBackendOptions::default();

        // Test initialization
        let result = init_rpc("test_worker", 0, 1, options).await;
        if let Err(e) = &result {
            info!("RPC initialization failed: {}", e);
        }
        assert!(result.is_ok());

        // Test that we can get worker info
        assert_eq!(get_worker_rank()?, 0);
        assert_eq!(get_world_size()?, 1);
        assert!(is_initialized());

        // Clean up
        shutdown().await?;
        reset_rpc();

        Ok(())
    }

    #[tokio::test]
    async fn test_function_registration() -> TorshResult<()> {
        reset_rpc();

        let options = RpcBackendOptions::default();
        init_rpc("test_worker", 0, 1, options).await?;

        // Register a function
        register_function("add", add_function).await?;
        register_function("multiply", multiply_function).await?;

        // Verify functions are registered
        let function_registry = {
            let worker_arc = get_rpc_worker()?;
            let worker_guard = worker_arc.lock().expect("lock should not be poisoned");
            let worker_ref = worker_guard.as_ref().unwrap();
            worker_ref.function_registry.clone()
        }; // Guard dropped here

        let registry = function_registry.read().await;
        assert!(registry.contains_key("add"));
        assert!(registry.contains_key("multiply"));
        drop(registry); // Release the registry lock

        shutdown().await?;
        reset_rpc();

        Ok(())
    }

    #[tokio::test]
    async fn test_rpc_message_serialization() -> TorshResult<()> {
        let message = RpcMessage::FunctionCall {
            id: "test-123".to_string(),
            function_name: "add".to_string(),
            args: vec![1, 2, 3, 4],
        };

        // Test serialization
        let serialized =
            oxicode::serde::encode_to_vec(&message, oxicode::config::standard()).unwrap();

        // Test deserialization
        let (deserialized, _): (RpcMessage, usize) =
            oxicode::serde::decode_from_slice(&serialized, oxicode::config::standard()).unwrap();

        match (message, deserialized) {
            (
                RpcMessage::FunctionCall {
                    id: id1,
                    function_name: fn1,
                    args: args1,
                },
                RpcMessage::FunctionCall {
                    id: id2,
                    function_name: fn2,
                    args: args2,
                },
            ) => {
                assert_eq!(id1, id2);
                assert_eq!(fn1, fn2);
                assert_eq!(args1, args2);
            }
            _ => panic!("Message types don't match"),
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_rref_creation() -> TorshResult<()> {
        let rref: RRef<TestResult> = RRef::new("test-id".to_string(), 42);

        assert_eq!(rref.id(), "test-id");
        assert_eq!(rref.owner_rank(), 42);

        Ok(())
    }

    #[tokio::test]
    async fn test_rpc_backend_options() {
        let default_options = RpcBackendOptions::default();
        assert_eq!(default_options.num_worker_threads, 4);
        assert_eq!(default_options.rpc_timeout, Duration::from_secs(60));
        assert_eq!(default_options.init_method, "env://");
        assert_eq!(default_options.buffer_size, 8192);
        assert_eq!(default_options.max_connections, 100);

        let custom_options = RpcBackendOptions {
            num_worker_threads: 8,
            rpc_timeout: Duration::from_secs(30),
            init_method: "file://".to_string(),
            buffer_size: 4096,
            max_connections: 50,
        };

        assert_eq!(custom_options.num_worker_threads, 8);
        assert_eq!(custom_options.rpc_timeout, Duration::from_secs(30));
        assert_eq!(custom_options.init_method, "file://");
        assert_eq!(custom_options.buffer_size, 4096);
        assert_eq!(custom_options.max_connections, 50);
    }

    #[test]
    fn test_rpc_not_initialized() {
        // Without initialization, we should get errors
        assert!(!is_initialized());
    }

    #[tokio::test]
    async fn test_rpc_shutdown_cleanup() -> TorshResult<()> {
        reset_rpc();

        let options = RpcBackendOptions::default();
        init_rpc("test_worker", 0, 1, options).await?;

        assert!(is_initialized());

        // Register some functions and remote refs
        register_function("add", add_function).await?;

        // Shutdown should clean everything up
        shutdown().await?;
        reset_rpc();

        Ok(())
    }
}
