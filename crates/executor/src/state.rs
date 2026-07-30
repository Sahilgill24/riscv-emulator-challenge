use std::{
    fs::File,
    io::{Seek, Write},
};

use hashbrown::HashMap;
use serde::{Deserialize, Serialize};

use crate::{events::MemoryRecord, syscalls::SyscallCode, ExecutorMode};

/// Holds data describing the current state of a program's execution.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[repr(C)]
pub struct ExecutionState {
    /// The program counter.
    pub pc: u32,

    /// The shard clock keeps track of how many shards have been executed.
    pub current_shard: u32,

    /// The memory which instructions operate over. Values contain the memory value and last shard
    /// + timestamp that each memory address was accessed.
    pub memory: HashMap<u32, MemoryRecord>,

    /// The global clock keeps track of how many instructions have been executed through all shards.
    pub global_clk: u64,

    /// The clock increments by 4 (possibly more in syscalls) for each instruction that has been
    /// executed in this shard.
    pub clk: u32,

    /// Fast register file: stores values for x0–x31 directly, bypassing the memory HashMap.
    pub registers: [u32; 32],

    /// Base address of the flat value array (`mem_values`). Addresses below this, or above the
    /// array's covered span, fall through to `mem_overflow`.
    pub mem_base: u32,

    /// Flat, direct-indexed memory value store: `mem_values[(addr - mem_base) >> 2]`. This is the
    /// single source of truth for memory *values* on the hot path — no hashing, no `Option`, no
    /// per-access record. Allocated lazily (zero-backed pages) in `initialize`.
    #[serde(skip)]
    pub mem_values: Vec<u32>,

    /// Safety net for addresses outside the dense `mem_values` span (rare: high stack / mmap).
    pub mem_overflow: HashMap<u32, u32>,

    /// Uninitialized memory addresses that have a specific value they should be initialized with.
    /// `SyscallHintRead` uses this to write hint data into uninitialized memory.
    pub uninitialized_memory: HashMap<u32, u32>,

    /// A stream of input values (global to the entire program).
    pub input_stream: Vec<Vec<u8>>,

    /// A ptr to the current position in the input stream incremented by `HINT_READ` opcode.
    pub input_stream_ptr: usize,

    /// A ptr to the current position in the proof stream, incremented after verifying a proof.
    pub proof_stream_ptr: usize,

    /// A stream of public values from the program (global to entire program).
    pub public_values_stream: Vec<u8>,

    /// A ptr to the current position in the public values stream, incremented when reading from
    /// `public_values_stream`.
    pub public_values_stream_ptr: usize,

    /// Keeps track of how many times a certain syscall has been called.
    pub syscall_counts: HashMap<SyscallCode, u64>,
}

impl ExecutionState {
    #[must_use]
    /// Create a new [`ExecutionState`].
    pub fn new(pc_start: u32) -> Self {
        Self {
            global_clk: 0,
            // Start at shard 1 since shard 0 is reserved for memory initialization.
            current_shard: 1,
            clk: 0,
            pc: pc_start,
            memory: HashMap::new(),
            uninitialized_memory: HashMap::new(),
            input_stream: Vec::new(),
            input_stream_ptr: 0,
            public_values_stream: Vec::new(),
            public_values_stream_ptr: 0,
            proof_stream_ptr: 0,
            syscall_counts: HashMap::new(),
            registers: [0u32; 32],
            mem_base: 0x1_0000,
            mem_values: Vec::new(),
            mem_overflow: HashMap::new(),
        }
    }

    /// Read a memory value from the flat store, falling back to the overflow map.
    #[inline(always)]
    #[must_use]
    pub fn mem_get(&self, addr: u32) -> u32 {
        let idx = (addr.wrapping_sub(self.mem_base) >> 2) as usize;
        if idx < self.mem_values.len() {
            // SAFETY: bounds checked above.
            unsafe { *self.mem_values.get_unchecked(idx) }
        } else {
            self.mem_overflow.get(&addr).copied().unwrap_or(0)
        }
    }

    /// Write a memory value to the flat store, falling back to the overflow map.
    #[inline(always)]
    pub fn mem_set(&mut self, addr: u32, value: u32) {
        let idx = (addr.wrapping_sub(self.mem_base) >> 2) as usize;
        if idx < self.mem_values.len() {
            // SAFETY: bounds checked above.
            unsafe { *self.mem_values.get_unchecked_mut(idx) = value; }
        } else {
            self.mem_overflow.insert(addr, value);
        }
    }
}

/// Holds data to track changes made to the runtime since a fork point.
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct ForkState {
    /// The `global_clk` value at the fork point.
    pub global_clk: u64,
    /// The original `clk` value at the fork point.
    pub clk: u32,
    /// The original `pc` value at the fork point.
    pub pc: u32,
    /// Original memory *values* for every address touched since the fork point (first-touch wins),
    /// used to roll `mem_values` back on unconstrained exit.
    pub memory_diff: HashMap<u32, u32>,
    // /// The original memory access record at the fork point.
    // pub op_record: MemoryAccessRecord,
    // /// The original execution record at the fork point.
    // pub record: ExecutionRecord,
    /// Whether `emit_events` was enabled at the fork point.
    pub executor_mode: ExecutorMode,
    /// Snapshot of the register file at the fork point, for restoration on unconstrained exit.
    pub registers: [u32; 32],
}

impl ExecutionState {
    /// Save the execution state to a file.
    pub fn save(&self, file: &mut File) -> std::io::Result<()> {
        let mut writer = std::io::BufWriter::new(file);
        bincode::serialize_into(&mut writer, self).unwrap();
        writer.flush()?;
        writer.seek(std::io::SeekFrom::Start(0))?;
        Ok(())
    }
}
