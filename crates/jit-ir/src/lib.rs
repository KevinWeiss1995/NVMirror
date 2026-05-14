//! # jit-ir
//!
//! Typed intermediate representation for the eBPF JIT compiler.
//!
//! The IR sits between the decoded/verified eBPF instructions and the
//! target-specific code emitters. It operates on *physical* registers
//! (after register allocation) and includes explicit spill/fill nodes.
//!
//! Branch targets are represented as `BasicBlockId` — an index into
//! a verified basic-block array — making out-of-bounds jumps
//! unrepresentable by construction.

#![no_std]
#![forbid(unsafe_code)]

extern crate alloc;
use alloc::vec::Vec;

use ebpf_core::{AluOp, JmpOp, MemWidth, AtomicOp, EndianWidth};

/// Opaque basic-block identifier — an index into the program's BB array.
/// Using a newtype prevents accidental use of raw `usize` as a branch target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BasicBlockId(pub u32);

/// Physical register on the target ISA.
/// The numeric value is target-dependent (e.g., 0..30 for AArch64 X-regs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PhysReg(pub u8);

/// Stack slot for spilled registers.
/// Offset is relative to the JIT frame pointer, always negative (grows down).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpillSlot {
    pub offset: i16,
}

/// Source operand in the lowered IR: either a physical register or an immediate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operand {
    Reg(PhysReg),
    Imm(i64),
}

/// Branch condition for conditional jumps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BranchCond {
    pub op: JmpOp,
    pub lhs: PhysReg,
    pub rhs: Operand,
    pub is_32bit: bool,
}

/// A single lowered IR instruction, ready for target-specific emission.
///
/// Every variant maps to a small, bounded number of target instructions.
/// The worst-case expansion ratio per node determines the dry-run buffer
/// size calculation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IrNode {
    /// dst = dst OP src (64-bit)
    Alu64 { op: AluOp, dst: PhysReg, src: Operand },

    /// dst = (u32)(dst OP src) (32-bit, zero-extends result)
    Alu32 { op: AluOp, dst: PhysReg, src: Operand },

    /// dst = *(width*)(base + off)
    Load { width: MemWidth, dst: PhysReg, base: PhysReg, off: i16 },

    /// *(width*)(base + off) = src
    Store { width: MemWidth, base: PhysReg, src: PhysReg, off: i16 },

    /// *(width*)(base + off) = imm
    StoreImm { width: MemWidth, base: PhysReg, off: i16, imm: i32 },

    /// dst = imm64 (load wide immediate)
    LoadImm64 { dst: PhysReg, imm: u64 },

    /// Conditional branch to target BB.
    Branch { cond: BranchCond, target: BasicBlockId },

    /// Unconditional jump to target BB.
    Jump { target: BasicBlockId },

    /// Call helper function by ID.
    Call { func_id: u32 },

    /// Return from JIT'd program.
    Ret,

    /// Spill a register to the stack.
    Spill { reg: PhysReg, slot: SpillSlot },

    /// Fill (reload) a register from the stack.
    Fill { reg: PhysReg, slot: SpillSlot },

    /// Atomic memory operation.
    Atomic { width: MemWidth, op: AtomicOp, base: PhysReg, src: PhysReg, off: i16 },

    /// Byte-swap endianness.
    Endian { to_be: bool, width: EndianWidth, dst: PhysReg },

    /// Label: marks the start of a basic block (pseudo-instruction).
    Label { bb: BasicBlockId },

    /// Prologue: set up JIT stack frame.
    Prologue { frame_size: u16 },

    /// Epilogue: tear down JIT stack frame (before Ret).
    Epilogue { frame_size: u16 },
}

/// A basic block in the lowered IR.
#[derive(Debug, Clone)]
pub struct IrBasicBlock {
    pub id: BasicBlockId,
    pub nodes: Vec<IrNode>,
}

/// A complete lowered program, ready for code emission.
#[derive(Debug, Clone)]
pub struct IrProgram {
    pub blocks: Vec<IrBasicBlock>,
    pub frame_size: u16,
}

/// Trait that code emitters must implement.
///
/// Each method emits target-specific machine code for one IR node.
/// The emitter maintains an internal buffer and a relocation table
/// for branch fixups.
pub trait CodeEmitter {
    type Error: core::fmt::Debug;

    fn emit_node(&mut self, node: &IrNode) -> Result<(), Self::Error>;

    fn emit_program(&mut self, program: &IrProgram) -> Result<(), Self::Error> {
        for block in &program.blocks {
            self.emit_node(&IrNode::Label { bb: block.id })?;
            for node in &block.nodes {
                self.emit_node(node)?;
            }
        }
        Ok(())
    }

    /// Finalize: resolve relocations and return the executable byte buffer.
    fn finalize(&mut self) -> Result<Vec<u8>, Self::Error>;

    /// Dry-run: calculate the exact output size without emitting.
    /// O(n) where n = number of IR nodes.
    fn calculate_size(&self, program: &IrProgram) -> usize;
}
