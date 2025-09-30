//! Chrome tracing format export

use crate::{ProfileEvent, TorshResult};
use serde_json::{json, Value};
use std::fs::File;
use std::io::Write;
use torsh_core::TorshError;

/// Export events to Chrome tracing format
pub fn export(events: &[ProfileEvent], path: &str) -> TorshResult<()> {
    let file = File::create(path)
        .map_err(|e| TorshError::InvalidArgument(format!("Failed to create file {path}: {e}")))?;

    export_to_writer(events, file)
}

/// Export events to a writer in Chrome tracing format
pub fn export_to_writer<W: Write>(events: &[ProfileEvent], mut writer: W) -> TorshResult<()> {
    let chrome_events: Vec<Value> = events
        .iter()
        .map(|event| {
            let mut args = serde_json::Map::new();

            if let Some(ops) = event.operation_count {
                args.insert("operations".to_string(), json!(ops));
            }

            if let Some(flops) = event.flops {
                args.insert("flops".to_string(), json!(flops));
                args.insert("gflops".to_string(), json!(flops as f64 / 1_000_000_000.0));
            }

            if let Some(bytes) = event.bytes_transferred {
                args.insert("bytes".to_string(), json!(bytes));
                args.insert("gb".to_string(), json!(bytes as f64 / 1_073_741_824.0));
            }

            json!({
                "name": event.name,
                "cat": event.category,
                "ph": "X", // Complete event (duration event)
                "ts": event.start_us,
                "dur": event.duration_us,
                "pid": 1, // Process ID (hardcoded for simplicity)
                "tid": event.thread_id,
                "args": args
            })
        })
        .collect();

    let trace_data = json!({
        "traceEvents": chrome_events,
        "displayTimeUnit": "ms",
        "systemTraceEvents": "SystemTraceData",
        "otherData": {
            "version": "torsh-profiler 0.1.0-alpha.1"
        }
    });

    serde_json::to_writer_pretty(&mut writer, &trace_data)
        .map_err(|e| TorshError::InvalidArgument(format!("Failed to write JSON: {e}")))?;

    Ok(())
}

/// Create a Chrome trace event
#[allow(clippy::too_many_arguments)]
pub fn create_chrome_event(
    name: &str,
    category: &str,
    phase: &str,
    timestamp_us: u64,
    duration_us: Option<u64>,
    process_id: u32,
    thread_id: u32,
    args: Option<Value>,
) -> Value {
    let mut event = json!({
        "name": name,
        "cat": category,
        "ph": phase,
        "ts": timestamp_us,
        "pid": process_id,
        "tid": thread_id
    });

    if let Some(dur) = duration_us {
        event["dur"] = json!(dur);
    }

    if let Some(args) = args {
        event["args"] = args;
    }

    event
}

/// Chrome trace phases
pub mod phases {
    pub const DURATION_BEGIN: &str = "B";
    pub const DURATION_END: &str = "E";
    pub const COMPLETE: &str = "X";
    pub const INSTANT: &str = "i";
    pub const COUNTER: &str = "C";
    pub const ASYNC_START: &str = "b";
    pub const ASYNC_INSTANT: &str = "n";
    pub const ASYNC_END: &str = "e";
    pub const FLOW_START: &str = "s";
    pub const FLOW_STEP: &str = "t";
    pub const FLOW_END: &str = "f";
    pub const SAMPLE: &str = "P";
    pub const OBJECT_CREATED: &str = "N";
    pub const OBJECT_SNAPSHOT: &str = "O";
    pub const OBJECT_DESTROYED: &str = "D";
    pub const METADATA: &str = "M";
    pub const MEMORY_DUMP_GLOBAL: &str = "V";
    pub const MEMORY_DUMP_PROCESS: &str = "v";
    pub const MARK: &str = "R";
    pub const CLOCK_SYNC: &str = "c";
}

/// Chrome trace scopes
pub mod scopes {
    pub const GLOBAL: &str = "g";
    pub const PROCESS: &str = "p";
    pub const THREAD: &str = "t";
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_chrome_trace_export_empty() {
        let events = vec![];
        let mut buffer = Vec::new();
        let cursor = Cursor::new(&mut buffer);

        export_to_writer(&events, cursor).unwrap();

        let output = String::from_utf8(buffer).unwrap();
        assert!(output.contains("\"traceEvents\": []"));
    }

    #[test]
    fn test_chrome_trace_export_single_event() {
        let events = vec![ProfileEvent {
            name: "test_event".to_string(),
            category: "test".to_string(),
            start_us: 1000,
            duration_us: 500,
            thread_id: 123,
            operation_count: None,
            flops: None,
            bytes_transferred: None,
            stack_trace: None,
        }];

        let mut buffer = Vec::new();
        let cursor = Cursor::new(&mut buffer);

        export_to_writer(&events, cursor).unwrap();

        let output = String::from_utf8(buffer).unwrap();
        assert!(output.contains("\"name\": \"test_event\""));
        assert!(output.contains("\"cat\": \"test\""));
        assert!(output.contains("\"ts\": 1000"));
        assert!(output.contains("\"dur\": 500"));
        assert!(output.contains("\"tid\": 123"));
    }

    #[test]
    fn test_create_chrome_event() {
        let event = create_chrome_event(
            "my_function",
            "compute",
            phases::COMPLETE,
            5000,
            Some(1000),
            1,
            42,
            Some(json!({"param": "value"})),
        );

        assert_eq!(event["name"], "my_function");
        assert_eq!(event["cat"], "compute");
        assert_eq!(event["ph"], "X");
        assert_eq!(event["ts"], 5000);
        assert_eq!(event["dur"], 1000);
        assert_eq!(event["pid"], 1);
        assert_eq!(event["tid"], 42);
        assert_eq!(event["args"]["param"], "value");
    }

    #[test]
    fn test_chrome_trace_multiple_events() {
        let events = vec![
            ProfileEvent {
                name: "matrix_mul".to_string(),
                category: "compute".to_string(),
                start_us: 1000,
                duration_us: 2000,
                thread_id: 1,
                operation_count: Some(1000000),
                flops: Some(2000000000),
                bytes_transferred: Some(4096),
                stack_trace: None,
            },
            ProfileEvent {
                name: "conv2d".to_string(),
                category: "neural_net".to_string(),
                start_us: 3000,
                duration_us: 1500,
                thread_id: 2,
                operation_count: None,
                flops: None,
                bytes_transferred: None,
                stack_trace: None,
            },
        ];

        let mut buffer = Vec::new();
        let cursor = Cursor::new(&mut buffer);

        export_to_writer(&events, cursor).unwrap();

        let output = String::from_utf8(buffer).unwrap();
        assert!(output.contains("matrix_mul"));
        assert!(output.contains("conv2d"));
        assert!(output.contains("compute"));
        assert!(output.contains("neural_net"));
    }
}
