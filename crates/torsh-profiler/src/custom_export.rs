//! Custom export formats for profiling data

use crate::{ProfileEvent, TorshResult};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use torsh_core::TorshError;

/// Custom export format configuration
#[derive(Debug, Clone)]
pub struct CustomExportFormat {
    pub name: String,
    pub description: String,
    pub file_extension: String,
    pub schema: ExportSchema,
}

/// Export schema definition
#[derive(Debug, Clone)]
pub enum ExportSchema {
    /// JSON with custom field mapping
    Json {
        field_mapping: HashMap<String, String>,
        include_metadata: bool,
        pretty_print: bool,
    },
    /// CSV with custom column configuration
    Csv {
        columns: Vec<CsvColumn>,
        delimiter: char,
        include_header: bool,
    },
    /// XML format
    Xml {
        root_element: String,
        event_element: String,
        field_mapping: HashMap<String, String>,
    },
    /// Custom text format with template
    Text { template: String, separator: String },
}

/// CSV column configuration
#[derive(Debug, Clone)]
pub struct CsvColumn {
    pub name: String,
    pub field: String,
    pub formatter: Option<CsvFormatter>,
}

/// CSV field formatters
#[derive(Debug, Clone)]
pub enum CsvFormatter {
    Duration(DurationFormat),
    Memory(MemoryFormat),
    Number(NumberFormat),
    Text(TextFormat),
}

#[derive(Debug, Clone)]
pub enum DurationFormat {
    Microseconds,
    Milliseconds,
    Seconds,
    HumanReadable,
}

#[derive(Debug, Clone)]
pub enum MemoryFormat {
    Bytes,
    Kilobytes,
    Megabytes,
    Gigabytes,
    HumanReadable,
}

#[derive(Debug, Clone)]
pub enum NumberFormat {
    Default,
    Scientific,
    Percentage,
    WithCommas,
}

#[derive(Debug, Clone)]
pub enum TextFormat {
    Default,
    Uppercase,
    Lowercase,
    Truncate(usize),
}

/// Custom export engine
#[derive(Debug, Clone)]
pub struct CustomExporter {
    formats: HashMap<String, CustomExportFormat>,
}

impl CustomExporter {
    /// Create a new custom exporter
    pub fn new() -> Self {
        let mut exporter = Self {
            formats: HashMap::new(),
        };

        // Register default custom formats
        exporter.register_default_formats();
        exporter
    }

    /// Register a custom export format
    pub fn register_format(&mut self, format: CustomExportFormat) {
        self.formats.insert(format.name.clone(), format);
    }

    /// Get available format names
    pub fn get_format_names(&self) -> Vec<String> {
        self.formats.keys().cloned().collect()
    }

    /// Export events using a registered format
    pub fn export(
        &self,
        events: &[ProfileEvent],
        format_name: &str,
        path: &str,
    ) -> TorshResult<()> {
        let format = self.formats.get(format_name).ok_or_else(|| {
            TorshError::InvalidArgument(format!("Unknown export format: {format_name}"))
        })?;

        match &format.schema {
            ExportSchema::Json {
                field_mapping,
                include_metadata,
                pretty_print,
            } => self.export_json(
                events,
                field_mapping,
                *include_metadata,
                *pretty_print,
                path,
            ),
            ExportSchema::Csv {
                columns,
                delimiter,
                include_header,
            } => self.export_csv(events, columns, *delimiter, *include_header, path),
            ExportSchema::Xml {
                root_element,
                event_element,
                field_mapping,
            } => self.export_xml(events, root_element, event_element, field_mapping, path),
            ExportSchema::Text {
                template,
                separator,
            } => self.export_text(events, template, separator, path),
        }
    }

    /// Register default custom formats
    fn register_default_formats(&mut self) {
        // Compact JSON format
        let compact_json = CustomExportFormat {
            name: "compact_json".to_string(),
            description: "Compact JSON format with minimal fields".to_string(),
            file_extension: "json".to_string(),
            schema: ExportSchema::Json {
                field_mapping: [
                    ("name".to_string(), "n".to_string()),
                    ("duration_us".to_string(), "d".to_string()),
                    ("category".to_string(), "c".to_string()),
                ]
                .iter()
                .cloned()
                .collect(),
                include_metadata: false,
                pretty_print: false,
            },
        };
        self.register_format(compact_json);

        // Performance-focused CSV
        let perf_csv = CustomExportFormat {
            name: "performance_csv".to_string(),
            description: "CSV focused on performance metrics".to_string(),
            file_extension: "csv".to_string(),
            schema: ExportSchema::Csv {
                columns: vec![
                    CsvColumn {
                        name: "Event".to_string(),
                        field: "name".to_string(),
                        formatter: None,
                    },
                    CsvColumn {
                        name: "Duration (ms)".to_string(),
                        field: "duration_us".to_string(),
                        formatter: Some(CsvFormatter::Duration(DurationFormat::Milliseconds)),
                    },
                    CsvColumn {
                        name: "FLOPS".to_string(),
                        field: "flops".to_string(),
                        formatter: Some(CsvFormatter::Number(NumberFormat::WithCommas)),
                    },
                    CsvColumn {
                        name: "Bandwidth (MB)".to_string(),
                        field: "bytes_transferred".to_string(),
                        formatter: Some(CsvFormatter::Memory(MemoryFormat::Megabytes)),
                    },
                ],
                delimiter: ',',
                include_header: true,
            },
        };
        self.register_format(perf_csv);

        // Simple text format
        let simple_text = CustomExportFormat {
            name: "simple_text".to_string(),
            description: "Simple text format for quick viewing".to_string(),
            file_extension: "txt".to_string(),
            schema: ExportSchema::Text {
                template: "{name}: {duration_us}μs ({category})".to_string(),
                separator: "\n".to_string(),
            },
        };
        self.register_format(simple_text);
    }

    /// Export as custom JSON
    fn export_json(
        &self,
        events: &[ProfileEvent],
        field_mapping: &HashMap<String, String>,
        include_metadata: bool,
        pretty_print: bool,
        path: &str,
    ) -> TorshResult<()> {
        let mut mapped_events = Vec::new();

        for event in events {
            let mut mapped_event = serde_json::Map::new();

            // Map fields according to configuration
            self.map_field(&mut mapped_event, "name", &event.name, field_mapping);
            self.map_field(
                &mut mapped_event,
                "category",
                &event.category,
                field_mapping,
            );
            self.map_field(
                &mut mapped_event,
                "start_us",
                &event.start_us,
                field_mapping,
            );
            self.map_field(
                &mut mapped_event,
                "duration_us",
                &event.duration_us,
                field_mapping,
            );
            self.map_field(
                &mut mapped_event,
                "thread_id",
                &event.thread_id,
                field_mapping,
            );

            if let Some(ops) = event.operation_count {
                self.map_field(&mut mapped_event, "operation_count", &ops, field_mapping);
            }
            if let Some(flops) = event.flops {
                self.map_field(&mut mapped_event, "flops", &flops, field_mapping);
            }
            if let Some(bytes) = event.bytes_transferred {
                self.map_field(
                    &mut mapped_event,
                    "bytes_transferred",
                    &bytes,
                    field_mapping,
                );
            }
            if let Some(ref stack_trace) = event.stack_trace {
                self.map_field(&mut mapped_event, "stack_trace", stack_trace, field_mapping);
            }

            mapped_events.push(Value::Object(mapped_event));
        }

        let mut output = serde_json::Map::new();
        output.insert("events".to_string(), Value::Array(mapped_events));

        if include_metadata {
            let metadata = json!({
                "export_timestamp": chrono::Utc::now().to_rfc3339(),
                "event_count": events.len(),
                "format": "custom_json"
            });
            output.insert("metadata".to_string(), metadata);
        }

        let json_output = Value::Object(output);
        let mut file = File::create(path).map_err(|e| {
            TorshError::InvalidArgument(format!("Failed to create file {path}: {e}"))
        })?;

        let json_string = if pretty_print {
            serde_json::to_string_pretty(&json_output)
        } else {
            serde_json::to_string(&json_output)
        }
        .map_err(|e| TorshError::InvalidArgument(format!("Failed to serialize JSON: {e}")))?;

        file.write_all(json_string.as_bytes())
            .map_err(|e| TorshError::InvalidArgument(format!("Failed to write file: {e}")))?;

        Ok(())
    }

    /// Export as custom CSV
    fn export_csv(
        &self,
        events: &[ProfileEvent],
        columns: &[CsvColumn],
        delimiter: char,
        include_header: bool,
        path: &str,
    ) -> TorshResult<()> {
        let mut file = File::create(path).map_err(|e| {
            TorshError::InvalidArgument(format!("Failed to create file {path}: {e}"))
        })?;

        // Write header
        if include_header {
            let header: Vec<String> = columns.iter().map(|col| col.name.clone()).collect();
            writeln!(file, "{}", header.join(&delimiter.to_string()))
                .map_err(|e| TorshError::InvalidArgument(format!("Failed to write header: {e}")))?;
        }

        // Write events
        for event in events {
            let row: Vec<String> = columns
                .iter()
                .map(|col| self.format_field_value(event, &col.field, &col.formatter))
                .collect();
            writeln!(file, "{}", row.join(&delimiter.to_string()))
                .map_err(|e| TorshError::InvalidArgument(format!("Failed to write row: {e}")))?;
        }

        Ok(())
    }

    /// Export as XML
    fn export_xml(
        &self,
        events: &[ProfileEvent],
        root_element: &str,
        event_element: &str,
        field_mapping: &HashMap<String, String>,
        path: &str,
    ) -> TorshResult<()> {
        let mut file = File::create(path).map_err(|e| {
            TorshError::InvalidArgument(format!("Failed to create file {path}: {e}"))
        })?;

        writeln!(file, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>")
            .map_err(|e| TorshError::InvalidArgument(format!("Failed to write XML header: {e}")))?;

        writeln!(file, "<{root_element}>").map_err(|e| {
            TorshError::InvalidArgument(format!("Failed to write root element: {e}"))
        })?;

        for event in events {
            writeln!(file, "  <{event_element}>").map_err(|e| {
                TorshError::InvalidArgument(format!("Failed to write event element: {e}"))
            })?;

            self.write_xml_field(&mut file, "name", &event.name, field_mapping)?;
            self.write_xml_field(&mut file, "category", &event.category, field_mapping)?;
            self.write_xml_field(
                &mut file,
                "start_us",
                &event.start_us.to_string(),
                field_mapping,
            )?;
            self.write_xml_field(
                &mut file,
                "duration_us",
                &event.duration_us.to_string(),
                field_mapping,
            )?;
            self.write_xml_field(
                &mut file,
                "thread_id",
                &event.thread_id.to_string(),
                field_mapping,
            )?;

            if let Some(ops) = event.operation_count {
                self.write_xml_field(
                    &mut file,
                    "operation_count",
                    &ops.to_string(),
                    field_mapping,
                )?;
            }
            if let Some(flops) = event.flops {
                self.write_xml_field(&mut file, "flops", &flops.to_string(), field_mapping)?;
            }
            if let Some(bytes) = event.bytes_transferred {
                self.write_xml_field(
                    &mut file,
                    "bytes_transferred",
                    &bytes.to_string(),
                    field_mapping,
                )?;
            }

            writeln!(file, "  </{event_element}>").map_err(|e| {
                TorshError::InvalidArgument(format!("Failed to write closing event element: {e}"))
            })?;
        }

        writeln!(file, "</{root_element}>").map_err(|e| {
            TorshError::InvalidArgument(format!("Failed to write closing root element: {e}"))
        })?;

        Ok(())
    }

    /// Export as custom text
    fn export_text(
        &self,
        events: &[ProfileEvent],
        template: &str,
        separator: &str,
        path: &str,
    ) -> TorshResult<()> {
        let mut file = File::create(path).map_err(|e| {
            TorshError::InvalidArgument(format!("Failed to create file {path}: {e}"))
        })?;

        for (i, event) in events.iter().enumerate() {
            let formatted = self.format_template(template, event);
            write!(file, "{formatted}")
                .map_err(|e| TorshError::InvalidArgument(format!("Failed to write event: {e}")))?;

            if i < events.len() - 1 {
                write!(file, "{separator}").map_err(|e| {
                    TorshError::InvalidArgument(format!("Failed to write separator: {e}"))
                })?;
            }
        }

        Ok(())
    }

    /// Helper function to map JSON fields
    fn map_field<T: serde::Serialize>(
        &self,
        object: &mut serde_json::Map<String, Value>,
        original_field: &str,
        value: &T,
        field_mapping: &HashMap<String, String>,
    ) {
        let field_name = field_mapping
            .get(original_field)
            .unwrap_or(&original_field.to_string())
            .clone();
        object.insert(field_name, json!(value));
    }

    /// Helper function to write XML fields
    fn write_xml_field(
        &self,
        file: &mut File,
        original_field: &str,
        value: &str,
        field_mapping: &HashMap<String, String>,
    ) -> TorshResult<()> {
        let default_field = original_field.to_string();
        let field_name = field_mapping.get(original_field).unwrap_or(&default_field);

        writeln!(file, "    <{field_name}>{value}</{field_name}>")
            .map_err(|e| TorshError::InvalidArgument(format!("Failed to write XML field: {e}")))
    }

    /// Format field value according to formatter
    fn format_field_value(
        &self,
        event: &ProfileEvent,
        field: &str,
        formatter: &Option<CsvFormatter>,
    ) -> String {
        let raw_value = match field {
            "name" => event.name.clone(),
            "category" => event.category.clone(),
            "start_us" => event.start_us.to_string(),
            "duration_us" => event.duration_us.to_string(),
            "thread_id" => event.thread_id.to_string(),
            "operation_count" => event
                .operation_count
                .map_or("".to_string(), |v| v.to_string()),
            "flops" => event.flops.map_or("".to_string(), |v| v.to_string()),
            "bytes_transferred" => event
                .bytes_transferred
                .map_or("".to_string(), |v| v.to_string()),
            "stack_trace" => event.stack_trace.as_deref().unwrap_or("").to_string(),
            _ => "".to_string(),
        };

        if let Some(formatter) = formatter {
            self.apply_formatter(&raw_value, field, formatter)
        } else {
            raw_value
        }
    }

    /// Apply formatter to value
    fn apply_formatter(&self, value: &str, _field: &str, formatter: &CsvFormatter) -> String {
        match formatter {
            CsvFormatter::Duration(format) => {
                if let Ok(duration_us) = value.parse::<u64>() {
                    match format {
                        DurationFormat::Microseconds => format!("{duration_us}μs"),
                        DurationFormat::Milliseconds => {
                            format!("{:.3}ms", duration_us as f64 / 1000.0)
                        }
                        DurationFormat::Seconds => {
                            format!("{:.6}s", duration_us as f64 / 1_000_000.0)
                        }
                        DurationFormat::HumanReadable => {
                            if duration_us < 1000 {
                                format!("{duration_us}μs")
                            } else if duration_us < 1_000_000 {
                                format!("{:.2}ms", duration_us as f64 / 1000.0)
                            } else {
                                format!("{:.3}s", duration_us as f64 / 1_000_000.0)
                            }
                        }
                    }
                } else {
                    value.to_string()
                }
            }
            CsvFormatter::Memory(format) => {
                if let Ok(bytes) = value.parse::<u64>() {
                    match format {
                        MemoryFormat::Bytes => format!("{bytes}B"),
                        MemoryFormat::Kilobytes => format!("{:.2}KB", bytes as f64 / 1024.0),
                        MemoryFormat::Megabytes => format!("{:.2}MB", bytes as f64 / 1_048_576.0),
                        MemoryFormat::Gigabytes => {
                            format!("{:.2}GB", bytes as f64 / 1_073_741_824.0)
                        }
                        MemoryFormat::HumanReadable => {
                            if bytes < 1024 {
                                format!("{bytes}B")
                            } else if bytes < 1_048_576 {
                                format!("{:.2}KB", bytes as f64 / 1024.0)
                            } else if bytes < 1_073_741_824 {
                                format!("{:.2}MB", bytes as f64 / 1_048_576.0)
                            } else {
                                format!("{:.2}GB", bytes as f64 / 1_073_741_824.0)
                            }
                        }
                    }
                } else {
                    value.to_string()
                }
            }
            CsvFormatter::Number(format) => {
                if let Ok(num) = value.parse::<f64>() {
                    match format {
                        NumberFormat::Default => value.to_string(),
                        NumberFormat::Scientific => format!("{num:.2e}"),
                        NumberFormat::Percentage => format!("{:.2}%", num * 100.0),
                        NumberFormat::WithCommas => {
                            let parts: Vec<&str> = value.split('.').collect();
                            let integer_part = parts[0];
                            let mut formatted = String::new();
                            for (i, c) in integer_part.chars().rev().enumerate() {
                                if i > 0 && i % 3 == 0 {
                                    formatted.insert(0, ',');
                                }
                                formatted.insert(0, c);
                            }
                            if parts.len() > 1 {
                                formatted.push('.');
                                formatted.push_str(parts[1]);
                            }
                            formatted
                        }
                    }
                } else {
                    value.to_string()
                }
            }
            CsvFormatter::Text(format) => match format {
                TextFormat::Default => value.to_string(),
                TextFormat::Uppercase => value.to_uppercase(),
                TextFormat::Lowercase => value.to_lowercase(),
                TextFormat::Truncate(len) => {
                    if value.len() > *len {
                        format!("{}...", &value[..*len])
                    } else {
                        value.to_string()
                    }
                }
            },
        }
    }

    /// Format template string with event data
    fn format_template(&self, template: &str, event: &ProfileEvent) -> String {
        template
            .replace("{name}", &event.name)
            .replace("{category}", &event.category)
            .replace("{start_us}", &event.start_us.to_string())
            .replace("{duration_us}", &event.duration_us.to_string())
            .replace("{thread_id}", &event.thread_id.to_string())
            .replace(
                "{operation_count}",
                &event
                    .operation_count
                    .map_or("".to_string(), |v| v.to_string()),
            )
            .replace(
                "{flops}",
                &event.flops.map_or("".to_string(), |v| v.to_string()),
            )
            .replace(
                "{bytes_transferred}",
                &event
                    .bytes_transferred
                    .map_or("".to_string(), |v| v.to_string()),
            )
            .replace("{stack_trace}", event.stack_trace.as_deref().unwrap_or(""))
    }
}

impl Default for CustomExporter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_event() -> ProfileEvent {
        ProfileEvent {
            name: "test_event".to_string(),
            category: "test".to_string(),
            start_us: 1000,
            duration_us: 5000,
            thread_id: 123,
            operation_count: Some(100),
            flops: Some(1000000),
            bytes_transferred: Some(1024),
            stack_trace: None,
        }
    }

    #[test]
    fn test_custom_exporter_creation() {
        let exporter = CustomExporter::new();
        let formats = exporter.get_format_names();

        assert!(formats.contains(&"compact_json".to_string()));
        assert!(formats.contains(&"performance_csv".to_string()));
        assert!(formats.contains(&"simple_text".to_string()));
    }

    #[test]
    fn test_custom_json_export() {
        let exporter = CustomExporter::new();
        let events = vec![create_test_event()];

        let result = exporter.export(&events, "compact_json", "/tmp/test_compact.json");
        assert!(result.is_ok());

        // Clean up
        let _ = std::fs::remove_file("/tmp/test_compact.json");
    }

    #[test]
    fn test_custom_csv_export() {
        let exporter = CustomExporter::new();
        let events = vec![create_test_event()];

        let result = exporter.export(&events, "performance_csv", "/tmp/test_perf.csv");
        assert!(result.is_ok());

        // Clean up
        let _ = std::fs::remove_file("/tmp/test_perf.csv");
    }

    #[test]
    fn test_custom_text_export() {
        let exporter = CustomExporter::new();
        let events = vec![create_test_event()];

        let result = exporter.export(&events, "simple_text", "/tmp/test_simple.txt");
        assert!(result.is_ok());

        // Clean up
        let _ = std::fs::remove_file("/tmp/test_simple.txt");
    }

    #[test]
    fn test_duration_formatting() {
        let exporter = CustomExporter::new();

        let microseconds = exporter.apply_formatter(
            "5000",
            "duration_us",
            &CsvFormatter::Duration(DurationFormat::Microseconds),
        );
        assert_eq!(microseconds, "5000μs");

        let milliseconds = exporter.apply_formatter(
            "5000",
            "duration_us",
            &CsvFormatter::Duration(DurationFormat::Milliseconds),
        );
        assert_eq!(milliseconds, "5.000ms");

        let human_readable = exporter.apply_formatter(
            "5000",
            "duration_us",
            &CsvFormatter::Duration(DurationFormat::HumanReadable),
        );
        assert_eq!(human_readable, "5.00ms");
    }

    #[test]
    fn test_memory_formatting() {
        let exporter = CustomExporter::new();

        let bytes = exporter.apply_formatter(
            "1024",
            "bytes_transferred",
            &CsvFormatter::Memory(MemoryFormat::Bytes),
        );
        assert_eq!(bytes, "1024B");

        let kilobytes = exporter.apply_formatter(
            "1024",
            "bytes_transferred",
            &CsvFormatter::Memory(MemoryFormat::Kilobytes),
        );
        assert_eq!(kilobytes, "1.00KB");

        let human_readable = exporter.apply_formatter(
            "1048576",
            "bytes_transferred",
            &CsvFormatter::Memory(MemoryFormat::HumanReadable),
        );
        assert_eq!(human_readable, "1.00MB");
    }

    #[test]
    fn test_register_custom_format() {
        let mut exporter = CustomExporter::new();

        let custom_format = CustomExportFormat {
            name: "test_format".to_string(),
            description: "Test format".to_string(),
            file_extension: "test".to_string(),
            schema: ExportSchema::Text {
                template: "Event: {name}".to_string(),
                separator: " | ".to_string(),
            },
        };

        exporter.register_format(custom_format);
        let formats = exporter.get_format_names();
        assert!(formats.contains(&"test_format".to_string()));
    }
}
