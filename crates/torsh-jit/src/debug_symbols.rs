//! Debug symbol generation for JIT compilation
//!
//! This module provides comprehensive debugging symbol infrastructure for JIT-compiled
//! code, including DWARF debug information, symbol tables, and source mapping.

use crate::ir::{IrModule, IrValue, TypeKind};
use crate::JitResult;
use indexmap::IndexMap;
use std::collections::HashMap;

/// Debug symbol manager for JIT compilation
#[derive(Debug, Clone)]
pub struct DebugSymbolManager {
    /// Symbol tables indexed by module name
    symbol_tables: IndexMap<String, SymbolTable>,

    /// Source mappings for compiled modules
    source_mappings: IndexMap<String, SourceMapping>,

    /// DWARF debug information
    dwarf_info: DwarfDebugInfo,

    /// Configuration for debug symbol generation
    config: DebugSymbolConfig,

    /// Statistics about debug symbols
    stats: DebugSymbolStats,
}

/// Symbol table containing debug symbols for a module
#[derive(Debug, Clone)]
pub struct SymbolTable {
    /// Module name
    pub module_name: String,

    /// Function symbols
    pub functions: IndexMap<String, FunctionSymbol>,

    /// Variable symbols
    pub variables: IndexMap<String, VariableSymbol>,

    /// Type symbols
    pub types: IndexMap<String, TypeSymbol>,

    /// Address ranges for symbols
    pub address_ranges: Vec<AddressRange>,

    /// Line number information
    pub line_info: LineNumberTable,
}

/// Function symbol information
#[derive(Debug, Clone)]
pub struct FunctionSymbol {
    /// Function name
    pub name: String,

    /// Mangled name (if different from name)
    pub mangled_name: Option<String>,

    /// Function start address
    pub start_address: u64,

    /// Function end address
    pub end_address: u64,

    /// Function size in bytes
    pub size: usize,

    /// Return type
    pub return_type: TypeSymbol,

    /// Parameter symbols
    pub parameters: Vec<ParameterSymbol>,

    /// Local variable symbols
    pub locals: Vec<LocalVariableSymbol>,

    /// Source location
    pub source_location: SourceLocation,

    /// Inlined functions (for optimization tracking)
    pub inlined_functions: Vec<InlinedFunction>,
}

/// Variable symbol information
#[derive(Debug, Clone)]
pub struct VariableSymbol {
    /// Variable name
    pub name: String,

    /// Variable type
    pub var_type: TypeSymbol,

    /// Storage location
    pub location: VariableLocation,

    /// Source location where declared
    pub declaration_location: SourceLocation,

    /// Scope information
    pub scope: Scope,

    /// Live ranges (where variable is valid)
    pub live_ranges: Vec<LiveRange>,
}

/// Type symbol information
#[derive(Debug, Clone)]
pub struct TypeSymbol {
    /// Type name
    pub name: String,

    /// Type kind
    pub kind: TypeKind,

    /// Size in bytes
    pub size: usize,

    /// Alignment requirements
    pub alignment: usize,

    /// For composite types, member information
    pub members: Vec<TypeMember>,

    /// Source location where type is defined
    pub definition_location: Option<SourceLocation>,
}

/// Parameter symbol information
#[derive(Debug, Clone)]
pub struct ParameterSymbol {
    /// Parameter name
    pub name: String,

    /// Parameter type
    pub param_type: TypeSymbol,

    /// Parameter index
    pub index: usize,

    /// Storage location
    pub location: VariableLocation,
}

/// Local variable symbol information
#[derive(Debug, Clone)]
pub struct LocalVariableSymbol {
    /// Variable name
    pub name: String,

    /// Variable type
    pub var_type: TypeSymbol,

    /// Storage location
    pub location: VariableLocation,

    /// Scope information
    pub scope: Scope,

    /// Live ranges
    pub live_ranges: Vec<LiveRange>,
}

/// Variable storage location
#[derive(Debug, Clone)]
pub enum VariableLocation {
    /// Stored in a register
    Register { register: RegisterId },

    /// Stored on the stack
    Stack { offset: i32 },

    /// Stored in memory
    Memory { address: u64 },

    /// Constant value
    Constant { value: ConstantValue },

    /// Composite location (e.g., split across registers)
    Composite { locations: Vec<VariableLocation> },

    /// Location unknown or optimized away
    Unknown,
}

/// Register identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RegisterId {
    /// X86-64 registers
    X64(X64Register),

    /// ARM64 registers
    Arm64(Arm64Register),

    /// Generic register by number
    Generic(u32),
}

/// X86-64 register names
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum X64Register {
    RAX,
    RBX,
    RCX,
    RDX,
    RSI,
    RDI,
    RBP,
    RSP,
    R8,
    R9,
    R10,
    R11,
    R12,
    R13,
    R14,
    R15,
    XMM0,
    XMM1,
    XMM2,
    XMM3,
    XMM4,
    XMM5,
    XMM6,
    XMM7,
    XMM8,
    XMM9,
    XMM10,
    XMM11,
    XMM12,
    XMM13,
    XMM14,
    XMM15,
}

/// ARM64 register names
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Arm64Register {
    X0,
    X1,
    X2,
    X3,
    X4,
    X5,
    X6,
    X7,
    X8,
    X9,
    X10,
    X11,
    X12,
    X13,
    X14,
    X15,
    X16,
    X17,
    X18,
    X19,
    X20,
    X21,
    X22,
    X23,
    X24,
    X25,
    X26,
    X27,
    X28,
    X29,
    X30,
    SP,
    V0,
    V1,
    V2,
    V3,
    V4,
    V5,
    V6,
    V7,
    V8,
    V9,
    V10,
    V11,
    V12,
    V13,
    V14,
    V15,
    V16,
    V17,
    V18,
    V19,
    V20,
    V21,
    V22,
    V23,
    V24,
    V25,
    V26,
    V27,
    V28,
    V29,
    V30,
    V31,
}

/// Constant value for debugging
#[derive(Debug, Clone)]
pub enum ConstantValue {
    Int(i64),
    UInt(u64),
    Float(f64),
    Bool(bool),
    String(String),
    Null,
}

/// Type member information
#[derive(Debug, Clone)]
pub struct TypeMember {
    /// Member name
    pub name: String,

    /// Member type
    pub member_type: TypeSymbol,

    /// Offset within the type
    pub offset: usize,

    /// Size of the member
    pub size: usize,
}

/// Variable scope information
#[derive(Debug, Clone)]
pub struct Scope {
    /// Scope start address
    pub start_address: u64,

    /// Scope end address
    pub end_address: u64,

    /// Parent scope (if any)
    pub parent: Option<Box<Scope>>,

    /// Scope kind
    pub kind: ScopeKind,
}

/// Kind of scope
#[derive(Debug, Clone)]
pub enum ScopeKind {
    /// Function scope
    Function,

    /// Block scope
    Block,

    /// Try/catch scope
    Exception,

    /// Loop scope
    Loop,

    /// Conditional scope
    Conditional,
}

/// Variable live range
#[derive(Debug, Clone)]
pub struct LiveRange {
    /// Start address
    pub start: u64,

    /// End address
    pub end: u64,

    /// Location during this range
    pub location: VariableLocation,
}

/// Address range with metadata
#[derive(Debug, Clone)]
pub struct AddressRange {
    /// Start address
    pub start: u64,

    /// End address
    pub end: u64,

    /// Symbol associated with this range
    pub symbol: String,

    /// Additional attributes
    pub attributes: HashMap<String, String>,
}

/// Line number table for source mapping
#[derive(Debug, Clone)]
pub struct LineNumberTable {
    /// Entries mapping addresses to line numbers
    pub entries: Vec<LineNumberEntry>,

    /// Source file information
    pub source_files: Vec<SourceFile>,
}

/// Line number table entry
#[derive(Debug, Clone)]
pub struct LineNumberEntry {
    /// Address
    pub address: u64,

    /// Source file index
    pub file_index: usize,

    /// Line number
    pub line: u32,

    /// Column number
    pub column: u32,

    /// Whether this is a statement boundary
    pub is_statement: bool,

    /// Whether this is a basic block boundary
    pub is_basic_block: bool,
}

/// Source file information
#[derive(Debug, Clone)]
pub struct SourceFile {
    /// File path
    pub path: String,

    /// File size
    pub size: usize,

    /// Modification time
    pub mtime: Option<u64>,

    /// MD5 hash of file contents
    pub md5_hash: Option<[u8; 16]>,
}

/// Source location information
#[derive(Debug, Clone)]
pub struct SourceLocation {
    /// File path
    pub file: String,

    /// Line number (1-based)
    pub line: u32,

    /// Column number (1-based)
    pub column: u32,

    /// Length of the construct
    pub length: Option<u32>,
}

/// Inlined function information
#[derive(Debug, Clone)]
pub struct InlinedFunction {
    /// Original function name
    pub original_name: String,

    /// Inlined at location
    pub inlined_at: SourceLocation,

    /// Address ranges where inlined
    pub address_ranges: Vec<AddressRange>,

    /// Call site information
    pub call_site: SourceLocation,
}

/// Source mapping for a compiled module
#[derive(Debug, Clone)]
pub struct SourceMapping {
    /// Module name
    pub module_name: String,

    /// Address to source location mapping
    pub address_to_source: HashMap<u64, SourceLocation>,

    /// Source location to address mapping
    pub source_to_address: HashMap<(String, u32, u32), Vec<u64>>,

    /// Inline stack information
    pub inline_stacks: HashMap<u64, Vec<InlinedFunction>>,
}

/// DWARF debug information
#[derive(Debug, Clone, Default)]
pub struct DwarfDebugInfo {
    /// Compilation units
    pub compilation_units: Vec<CompilationUnit>,

    /// Debug sections
    pub debug_sections: HashMap<String, Vec<u8>>,

    /// String table
    pub string_table: Vec<String>,

    /// Abbreviation tables
    pub abbreviation_tables: Vec<AbbreviationTable>,
}

/// DWARF compilation unit
#[derive(Debug, Clone)]
pub struct CompilationUnit {
    /// Unit offset
    pub offset: u64,

    /// Unit length
    pub length: u64,

    /// DWARF version
    pub version: u16,

    /// Producer (compiler) information
    pub producer: String,

    /// Language code
    pub language: u32,

    /// Low PC (start address)
    pub low_pc: u64,

    /// High PC (end address)
    pub high_pc: u64,

    /// Debug information entries
    pub entries: Vec<DebugInfoEntry>,
}

/// DWARF debug information entry
#[derive(Debug, Clone)]
pub struct DebugInfoEntry {
    /// Entry offset
    pub offset: u64,

    /// Tag (what kind of entity this describes)
    pub tag: DwarfTag,

    /// Attributes
    pub attributes: HashMap<DwarfAttribute, DwarfValue>,

    /// Child entries
    pub children: Vec<DebugInfoEntry>,
}

/// DWARF tags
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DwarfTag {
    CompileUnit,
    Subprogram,
    Variable,
    Parameter,
    BaseType,
    PointerType,
    ArrayType,
    StructureType,
    UnionType,
    EnumerationType,
    LexicalBlock,
    InlinedSubroutine,
}

/// DWARF attributes
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DwarfAttribute {
    Name,
    Type,
    Location,
    LowPc,
    HighPc,
    FrameBase,
    ByteSize,
    Encoding,
    DeclarationFile,
    DeclarationLine,
    CallFile,
    CallLine,
    InlineStatus,
}

/// DWARF attribute values
#[derive(Debug, Clone)]
pub enum DwarfValue {
    String(String),
    Address(u64),
    Constant(u64),
    Block(Vec<u8>),
    Reference(u64),
    Flag(bool),
}

/// DWARF abbreviation table
#[derive(Debug, Clone)]
pub struct AbbreviationTable {
    /// Table offset
    pub offset: u64,

    /// Abbreviation entries
    pub entries: HashMap<u64, AbbreviationEntry>,
}

/// DWARF abbreviation entry
#[derive(Debug, Clone)]
pub struct AbbreviationEntry {
    /// Abbreviation code
    pub code: u64,

    /// Tag
    pub tag: DwarfTag,

    /// Whether entry has children
    pub has_children: bool,

    /// Attribute specifications
    pub attributes: Vec<(DwarfAttribute, DwarfForm)>,
}

/// DWARF attribute forms
#[derive(Debug, Clone)]
pub enum DwarfForm {
    Addr,
    Block1,
    Block2,
    Block4,
    Data1,
    Data2,
    Data4,
    Data8,
    String,
    Strp,
    Ref1,
    Ref2,
    Ref4,
    Ref8,
    RefAddr,
    Flag,
    FlagPresent,
}

/// Configuration for debug symbol generation
#[derive(Debug, Clone)]
pub struct DebugSymbolConfig {
    /// Enable DWARF debug information generation
    pub enable_dwarf: bool,

    /// Enable source mapping
    pub enable_source_mapping: bool,

    /// Include variable location information
    pub include_variable_locations: bool,

    /// Include inlined function information
    pub include_inline_info: bool,

    /// Debug information level (0-3)
    pub debug_level: u8,

    /// Compress debug sections
    pub compress_debug_sections: bool,

    /// Include optimization remarks
    pub include_optimization_remarks: bool,
}

/// Statistics about debug symbol generation
#[derive(Debug, Clone, Default)]
pub struct DebugSymbolStats {
    /// Total number of symbols generated
    pub total_symbols: usize,

    /// Total debug information size in bytes
    pub debug_info_size: usize,

    /// Number of source mappings
    pub source_mappings: usize,

    /// Generation time in nanoseconds
    pub generation_time_ns: u64,

    /// Compression ratio (if enabled)
    pub compression_ratio: f32,
}

impl Default for DebugSymbolConfig {
    fn default() -> Self {
        Self {
            enable_dwarf: true,
            enable_source_mapping: true,
            include_variable_locations: true,
            include_inline_info: true,
            debug_level: 2,
            compress_debug_sections: false,
            include_optimization_remarks: false,
        }
    }
}

impl DebugSymbolManager {
    /// Create a new debug symbol manager
    pub fn new(config: DebugSymbolConfig) -> Self {
        Self {
            symbol_tables: IndexMap::new(),
            source_mappings: IndexMap::new(),
            dwarf_info: DwarfDebugInfo::default(),
            config,
            stats: DebugSymbolStats::default(),
        }
    }

    /// Create a new debug symbol manager with default configuration
    pub fn with_defaults() -> Self {
        Self::new(DebugSymbolConfig::default())
    }

    /// Generate debug symbols for a compiled module
    pub fn generate_symbols(
        &mut self,
        module: &IrModule,
        code_address: u64,
        code_size: usize,
    ) -> JitResult<()> {
        let start_time = std::time::Instant::now();

        // Create symbol table for the module
        let symbol_table = self.create_symbol_table(module, code_address, code_size)?;

        // Generate source mapping
        let source_mapping = self.create_source_mapping(module, code_address)?;

        // Generate DWARF debug information
        if self.config.enable_dwarf {
            self.generate_dwarf_info(module, &symbol_table)?;
        }

        // Store the generated information
        self.symbol_tables.insert(module.name.clone(), symbol_table);
        self.source_mappings
            .insert(module.name.clone(), source_mapping);

        // Update statistics
        let generation_time = start_time.elapsed().as_nanos() as u64;
        self.stats.generation_time_ns += generation_time;
        self.stats.total_symbols += 1;

        Ok(())
    }

    /// Create symbol table for a module
    fn create_symbol_table(
        &self,
        module: &IrModule,
        code_address: u64,
        code_size: usize,
    ) -> JitResult<SymbolTable> {
        let mut symbol_table = SymbolTable {
            module_name: module.name.clone(),
            functions: IndexMap::new(),
            variables: IndexMap::new(),
            types: IndexMap::new(),
            address_ranges: Vec::new(),
            line_info: LineNumberTable {
                entries: Vec::new(),
                source_files: Vec::new(),
            },
        };

        // Add module-level function symbol
        let function_symbol = FunctionSymbol {
            name: module.name.clone(),
            mangled_name: None,
            start_address: code_address,
            end_address: code_address + code_size as u64,
            size: code_size,
            return_type: TypeSymbol::void_type(),
            parameters: Vec::new(),
            locals: Vec::new(),
            source_location: SourceLocation {
                file: "<generated>".to_string(),
                line: 1,
                column: 1,
                length: None,
            },
            inlined_functions: Vec::new(),
        };

        symbol_table
            .functions
            .insert(module.name.clone(), function_symbol);

        // Add address range for the entire module
        symbol_table.address_ranges.push(AddressRange {
            start: code_address,
            end: code_address + code_size as u64,
            symbol: module.name.clone(),
            attributes: HashMap::new(),
        });

        // Extract type information from IR
        for (type_id, type_def) in &module.types {
            let type_symbol = self.create_type_symbol(type_id, type_def)?;
            symbol_table
                .types
                .insert(format!("type_{}", type_id.0), type_symbol);
        }

        // Extract variable information from IR values
        for (value_id, value_def) in &module.values {
            if let Some(variable_symbol) = self.create_variable_symbol(value_id, value_def)? {
                symbol_table
                    .variables
                    .insert(format!("var_{}", value_id.0), variable_symbol);
            }
        }

        Ok(symbol_table)
    }

    /// Create source mapping for a module
    fn create_source_mapping(
        &self,
        module: &IrModule,
        code_address: u64,
    ) -> JitResult<SourceMapping> {
        let mut source_mapping = SourceMapping {
            module_name: module.name.clone(),
            address_to_source: HashMap::new(),
            source_to_address: HashMap::new(),
            inline_stacks: HashMap::new(),
        };

        // Map the entry point
        let entry_location = SourceLocation {
            file: "<generated>".to_string(),
            line: 1,
            column: 1,
            length: None,
        };

        source_mapping
            .address_to_source
            .insert(code_address, entry_location.clone());

        let key = (
            entry_location.file.clone(),
            entry_location.line,
            entry_location.column,
        );
        source_mapping
            .source_to_address
            .entry(key)
            .or_default()
            .push(code_address);

        Ok(source_mapping)
    }

    /// Generate DWARF debug information
    fn generate_dwarf_info(
        &mut self,
        _module: &IrModule,
        symbol_table: &SymbolTable,
    ) -> JitResult<()> {
        // Create compilation unit
        let compilation_unit = CompilationUnit {
            offset: 0,
            length: 0, // Will be filled in later
            version: 4,
            producer: "ToRSh JIT Compiler".to_string(),
            language: 0x8001, // DW_LANG_Rust (unofficial)
            low_pc: symbol_table
                .address_ranges
                .first()
                .map(|r| r.start)
                .unwrap_or(0),
            high_pc: symbol_table
                .address_ranges
                .last()
                .map(|r| r.end)
                .unwrap_or(0),
            entries: self.create_debug_entries(symbol_table)?,
        };

        self.dwarf_info.compilation_units.push(compilation_unit);

        Ok(())
    }

    /// Create debug information entries from symbol table
    fn create_debug_entries(&self, symbol_table: &SymbolTable) -> JitResult<Vec<DebugInfoEntry>> {
        let mut entries = Vec::new();

        // Create entries for functions
        for (name, function) in &symbol_table.functions {
            let mut attributes = HashMap::new();
            attributes.insert(DwarfAttribute::Name, DwarfValue::String(name.clone()));
            attributes.insert(
                DwarfAttribute::LowPc,
                DwarfValue::Address(function.start_address),
            );
            attributes.insert(
                DwarfAttribute::HighPc,
                DwarfValue::Address(function.end_address),
            );

            let entry = DebugInfoEntry {
                offset: 0,
                tag: DwarfTag::Subprogram,
                attributes,
                children: Vec::new(),
            };

            entries.push(entry);
        }

        // Create entries for types
        for (name, type_symbol) in &symbol_table.types {
            let mut attributes = HashMap::new();
            attributes.insert(DwarfAttribute::Name, DwarfValue::String(name.clone()));
            attributes.insert(
                DwarfAttribute::ByteSize,
                DwarfValue::Constant(type_symbol.size as u64),
            );

            let entry = DebugInfoEntry {
                offset: 0,
                tag: DwarfTag::BaseType,
                attributes,
                children: Vec::new(),
            };

            entries.push(entry);
        }

        Ok(entries)
    }

    /// Create type symbol from IR type definition
    fn create_type_symbol(
        &self,
        _type_id: &crate::ir::IrType,
        type_def: &crate::ir::TypeDef,
    ) -> JitResult<TypeSymbol> {
        Ok(TypeSymbol {
            name: format!("{:?}", type_def.kind),
            kind: type_def.kind.clone(),
            size: type_def.size.unwrap_or(0),
            alignment: type_def.align.unwrap_or(1),
            members: Vec::new(),
            definition_location: None,
        })
    }

    /// Create variable symbol from IR value definition
    fn create_variable_symbol(
        &self,
        _value_id: &IrValue,
        _value_def: &crate::ir::ValueDef,
    ) -> JitResult<Option<VariableSymbol>> {
        // For now, only create symbols for certain kinds of values
        // TODO: Implement based on value kind
        Ok(None)
    }

    /// Get symbol table for a module
    pub fn get_symbol_table(&self, module_name: &str) -> Option<&SymbolTable> {
        self.symbol_tables.get(module_name)
    }

    /// Get source mapping for a module
    pub fn get_source_mapping(&self, module_name: &str) -> Option<&SourceMapping> {
        self.source_mappings.get(module_name)
    }

    /// Lookup source location for an address
    pub fn lookup_source_location(&self, address: u64) -> Option<SourceLocation> {
        for source_mapping in self.source_mappings.values() {
            if let Some(location) = source_mapping.address_to_source.get(&address) {
                return Some(location.clone());
            }
        }
        None
    }

    /// Lookup addresses for a source location
    pub fn lookup_addresses(&self, file: &str, line: u32, column: u32) -> Vec<u64> {
        let mut addresses = Vec::new();
        let key = (file.to_string(), line, column);

        for source_mapping in self.source_mappings.values() {
            if let Some(addrs) = source_mapping.source_to_address.get(&key) {
                addresses.extend(addrs);
            }
        }

        addresses
    }

    /// Get DWARF debug information
    pub fn get_dwarf_info(&self) -> &DwarfDebugInfo {
        &self.dwarf_info
    }

    /// Get debug symbol statistics
    pub fn stats(&self) -> &DebugSymbolStats {
        &self.stats
    }

    /// Clear all debug symbols (for memory management)
    pub fn clear(&mut self) {
        self.symbol_tables.clear();
        self.source_mappings.clear();
        self.dwarf_info = DwarfDebugInfo::default();
        self.stats = DebugSymbolStats::default();
    }
}

impl TypeSymbol {
    /// Create a void type symbol
    pub fn void_type() -> Self {
        Self {
            name: "void".to_string(),
            kind: TypeKind::Void,
            size: 0,
            alignment: 1,
            members: Vec::new(),
            definition_location: None,
        }
    }

    /// Create a basic type symbol
    pub fn basic_type(kind: TypeKind) -> Self {
        let (name, size, alignment) = match kind {
            TypeKind::Bool => ("bool".to_string(), 1, 1),
            TypeKind::I8 => ("i8".to_string(), 1, 1),
            TypeKind::I16 => ("i16".to_string(), 2, 2),
            TypeKind::I32 => ("i32".to_string(), 4, 4),
            TypeKind::I64 => ("i64".to_string(), 8, 8),
            TypeKind::U8 => ("u8".to_string(), 1, 1),
            TypeKind::U16 => ("u16".to_string(), 2, 2),
            TypeKind::U32 => ("u32".to_string(), 4, 4),
            TypeKind::U64 => ("u64".to_string(), 8, 8),
            TypeKind::F16 => ("f16".to_string(), 2, 2),
            TypeKind::F32 => ("f32".to_string(), 4, 4),
            TypeKind::F64 => ("f64".to_string(), 8, 8),
            TypeKind::C64 => ("c64".to_string(), 8, 4),
            TypeKind::C128 => ("c128".to_string(), 16, 8),
            TypeKind::Void => ("void".to_string(), 0, 1),
            _ => ("unknown".to_string(), 0, 1),
        };

        Self {
            name,
            kind,
            size,
            alignment,
            members: Vec::new(),
            definition_location: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_debug_symbol_manager_creation() {
        let manager = DebugSymbolManager::with_defaults();
        assert_eq!(manager.symbol_tables.len(), 0);
        assert_eq!(manager.source_mappings.len(), 0);
    }

    #[test]
    fn test_type_symbol_creation() {
        let void_type = TypeSymbol::void_type();
        assert_eq!(void_type.name, "void");
        assert_eq!(void_type.size, 0);

        let i32_type = TypeSymbol::basic_type(TypeKind::I32);
        assert_eq!(i32_type.name, "i32");
        assert_eq!(i32_type.size, 4);
        assert_eq!(i32_type.alignment, 4);
    }

    #[test]
    fn test_register_identification() {
        let x64_reg = RegisterId::X64(X64Register::RAX);
        let arm64_reg = RegisterId::Arm64(Arm64Register::X0);
        let generic_reg = RegisterId::Generic(0);

        assert_ne!(x64_reg, arm64_reg);
        assert_ne!(arm64_reg, generic_reg);
    }

    #[test]
    fn test_source_location() {
        let location = SourceLocation {
            file: "test.rs".to_string(),
            line: 42,
            column: 10,
            length: Some(15),
        };

        assert_eq!(location.file, "test.rs");
        assert_eq!(location.line, 42);
        assert_eq!(location.column, 10);
    }
}
