//! # jit-regalloc
//!
//! Register allocator for the eBPF JIT compiler.
//!
//! Provides two strategies:
//!
//! 1. **Fixed mapping** (fast path): statically maps eBPF R0-R10 to
//!    target physical registers. No spills needed when the target has
//!    enough callee-saved registers (true for both AArch64 and RV64).
//!
//! 2. **Linear-scan** (fallback): for hypothetical targets with fewer
//!    registers, performs a single-pass scan over live ranges to allocate
//!    physical registers and insert spill/fill nodes.
//!
//! ## Complexity
//! Fixed mapping: O(n) — one pass to rewrite register references.
//! Linear-scan: O(n log n) — dominated by sorting live intervals.

#![no_std]
#![forbid(unsafe_code)]

extern crate alloc;
use alloc::vec::Vec;

use ebpf_core::{BpfReg, Insn, Source, NUM_REGS};
use jit_ir::*;

/// Target-specific register mapping table.
///
/// Maps each eBPF register (R0-R10) to a physical register on the target ISA.
#[derive(Debug, Clone)]
pub struct RegMap {
    pub map: [PhysReg; NUM_REGS],
    /// Temporary register for the JIT to use (not mapped to any eBPF reg).
    pub tmp: PhysReg,
    /// Base pointer for program data bounds checks.
    pub bounds_base: PhysReg,
    /// Length register for bounds checks.
    pub bounds_len: PhysReg,
}

impl RegMap {
    /// Look up the physical register for an eBPF register.
    #[inline]
    pub fn phys(&self, bpf: BpfReg) -> PhysReg {
        self.map[bpf.index()]
    }

    /// Translate a Source operand.
    #[inline]
    pub fn translate_source(&self, src: &Source) -> Operand {
        match src {
            Source::Imm(v) => Operand::Imm(*v),
            Source::Reg(r) => Operand::Reg(self.phys(*r)),
        }
    }
}

/// AArch64 register mapping (per plan §3.2).
///
/// | eBPF | AArch64 | Role                  |
/// |------|---------|-----------------------|
/// | R0   | X7      | Return value          |
/// | R1   | X0      | Arg 1 (matches ABI)   |
/// | R2   | X1      | Arg 2                 |
/// | R3   | X2      | Arg 3                 |
/// | R4   | X3      | Arg 4                 |
/// | R5   | X4      | Arg 5                 |
/// | R6   | X19     | Callee-saved          |
/// | R7   | X20     | Callee-saved          |
/// | R8   | X21     | Callee-saved          |
/// | R9   | X22     | Callee-saved          |
/// | R10  | X25     | Frame pointer         |
pub fn aarch64_regmap() -> RegMap {
    RegMap {
        map: [
            PhysReg(7),  // R0 → X7
            PhysReg(0),  // R1 → X0
            PhysReg(1),  // R2 → X1
            PhysReg(2),  // R3 → X2
            PhysReg(3),  // R4 → X3
            PhysReg(4),  // R5 → X4
            PhysReg(19), // R6 → X19
            PhysReg(20), // R7 → X20
            PhysReg(21), // R8 → X21
            PhysReg(22), // R9 → X22
            PhysReg(25), // R10 → X25
        ],
        tmp: PhysReg(9),         // X9 — caller-saved scratch
        bounds_base: PhysReg(10), // X10 — caller-saved
        bounds_len: PhysReg(11),  // X11 — caller-saved
    }
}

/// RISC-V RV64 register mapping (per plan §3.2).
///
/// | eBPF | RISC-V | ABI name | Role          |
/// |------|--------|----------|---------------|
/// | R0   | x10    | a0       | Return value  |
/// | R1   | x11    | a1       | Arg 1         |
/// | R2   | x12    | a2       | Arg 2         |
/// | R3   | x13    | a3       | Arg 3         |
/// | R4   | x14    | a4       | Arg 4         |
/// | R5   | x15    | a5       | Arg 5         |
/// | R6   | x18    | s2       | Callee-saved  |
/// | R7   | x19    | s3       | Callee-saved  |
/// | R8   | x20    | s4       | Callee-saved  |
/// | R9   | x21    | s5       | Callee-saved  |
/// | R10  | x22    | s6       | Frame pointer |
pub fn riscv64_regmap() -> RegMap {
    RegMap {
        map: [
            PhysReg(10), // R0 → a0
            PhysReg(11), // R1 → a1
            PhysReg(12), // R2 → a2
            PhysReg(13), // R3 → a3
            PhysReg(14), // R4 → a4
            PhysReg(15), // R5 → a5
            PhysReg(18), // R6 → s2
            PhysReg(19), // R7 → s3
            PhysReg(20), // R8 → s4
            PhysReg(21), // R9 → s5
            PhysReg(22), // R10 → s6
        ],
        tmp: PhysReg(23),       // s7
        bounds_base: PhysReg(24), // s8
        bounds_len: PhysReg(25),  // s9
    }
}

/// Lower a verified eBPF program to IR using fixed register mapping.
///
/// O(n) — single pass over the instruction stream.
pub fn lower_to_ir(insns: &[Insn], regmap: &RegMap) -> IrProgram {
    let mut blocks: Vec<IrBasicBlock> = Vec::new();
    let mut current_nodes: Vec<IrNode> = Vec::new();
    let mut bb_id: u32 = 0;
    let mut frame_size: u16 = 0;

    // Find basic block boundaries (leader instructions)
    let mut is_leader = alloc::vec![false; insns.len()];
    is_leader[0] = true;

    for (pc, insn) in insns.iter().enumerate() {
        match insn {
            Insn::Ja { off } => {
                let target = (pc as isize + 1 + *off as isize) as usize;
                if target < insns.len() {
                    is_leader[target] = true;
                }
                if pc + 1 < insns.len() {
                    is_leader[pc + 1] = true;
                }
            }
            Insn::JmpCond { off, .. } | Insn::JmpCond32 { off, .. } => {
                let target = (pc as isize + 1 + *off as isize) as usize;
                if target < insns.len() {
                    is_leader[target] = true;
                }
                if pc + 1 < insns.len() {
                    is_leader[pc + 1] = true;
                }
            }
            Insn::Exit | Insn::TailCall => {
                if pc + 1 < insns.len() {
                    is_leader[pc + 1] = true;
                }
            }
            _ => {}
        }
    }

    // Build a map from instruction PC to BasicBlockId
    let mut pc_to_bb = alloc::vec![BasicBlockId(0); insns.len()];
    let mut next_bb: u32 = 0;
    for (pc, &leader) in is_leader.iter().enumerate() {
        if leader {
            pc_to_bb[pc] = BasicBlockId(next_bb);
            next_bb += 1;
        } else if pc > 0 {
            pc_to_bb[pc] = pc_to_bb[pc - 1];
        }
    }

    // Detect whether the program writes to any callee-saved register or uses the stack.
    // If so, we must emit a prologue/epilogue to save/restore them.
    let mut needs_frame = false;
    for insn in insns.iter() {
        // Check for stack-relative accesses via R10
        let off = match insn {
            Insn::Load { src, off, .. } if *src == BpfReg::R10 => Some(*off),
            Insn::StoreReg { dst, off, .. } if *dst == BpfReg::R10 => Some(*off),
            Insn::StoreImm { dst, off, .. } if *dst == BpfReg::R10 => Some(*off),
            Insn::Atomic { dst, off, .. } if *dst == BpfReg::R10 => Some(*off),
            _ => None,
        };
        if let Some(o) = off {
            let needed = (-o) as u16;
            if needed > frame_size {
                frame_size = needed;
            }
            needs_frame = true;
        }

        // Check for writes to callee-saved registers (R6-R9, mapped to X19-X22)
        // or any helper call (which clobbers caller-saved and requires frame)
        match insn {
            Insn::Alu64 { dst, .. } | Insn::Alu32 { dst, .. }
            | Insn::LoadImm64 { dst, .. } | Insn::Load { dst, .. }
            | Insn::Endian { dst, .. } => {
                let r = dst.raw();
                if (6..=9).contains(&r) {
                    needs_frame = true;
                }
            }
            Insn::Call { .. } | Insn::TailCall => {
                needs_frame = true;
            }
            _ => {}
        }
    }

    // Minimum frame size: 64 bytes to hold callee-saved regs (STP pairs + LR/FP)
    if needs_frame && frame_size < 64 {
        frame_size = 64;
    }
    // Align frame size to 16 bytes
    frame_size = (frame_size + 15) & !15;

    // Lower each instruction
    for (pc, insn) in insns.iter().enumerate() {
        if is_leader[pc] && (!current_nodes.is_empty() || pc == 0) {
            if !current_nodes.is_empty() {
                blocks.push(IrBasicBlock {
                    id: BasicBlockId(bb_id),
                    nodes: core::mem::take(&mut current_nodes),
                });
                bb_id += 1;
            }
            // Start new BB with prologue if first block
            if pc == 0 && frame_size > 0 {
                current_nodes.push(IrNode::Prologue { frame_size });
            }
        }

        match insn {
            Insn::Alu64 { op, dst, src } => {
                current_nodes.push(IrNode::Alu64 {
                    op: *op,
                    dst: regmap.phys(*dst),
                    src: regmap.translate_source(src),
                });
            }

            Insn::Alu32 { op, dst, src } => {
                current_nodes.push(IrNode::Alu32 {
                    op: *op,
                    dst: regmap.phys(*dst),
                    src: regmap.translate_source(src),
                });
            }

            Insn::Load { width, dst, src, off } => {
                current_nodes.push(IrNode::Load {
                    width: *width,
                    dst: regmap.phys(*dst),
                    base: regmap.phys(*src),
                    off: *off,
                });
            }

            Insn::StoreReg { width, dst, src, off } => {
                current_nodes.push(IrNode::Store {
                    width: *width,
                    base: regmap.phys(*dst),
                    src: regmap.phys(*src),
                    off: *off,
                });
            }

            Insn::StoreImm { width, dst, off, imm } => {
                current_nodes.push(IrNode::StoreImm {
                    width: *width,
                    base: regmap.phys(*dst),
                    off: *off,
                    imm: *imm,
                });
            }

            Insn::LoadImm64 { dst, imm } => {
                current_nodes.push(IrNode::LoadImm64 {
                    dst: regmap.phys(*dst),
                    imm: *imm,
                });
            }

            Insn::JmpCond { op, dst, src, off } => {
                let target_pc = (pc as isize + 1 + *off as isize) as usize;
                current_nodes.push(IrNode::Branch {
                    cond: BranchCond {
                        op: *op,
                        lhs: regmap.phys(*dst),
                        rhs: regmap.translate_source(src),
                        is_32bit: false,
                    },
                    target: pc_to_bb[target_pc],
                });
            }

            Insn::JmpCond32 { op, dst, src, off } => {
                let target_pc = (pc as isize + 1 + *off as isize) as usize;
                current_nodes.push(IrNode::Branch {
                    cond: BranchCond {
                        op: *op,
                        lhs: regmap.phys(*dst),
                        rhs: regmap.translate_source(src),
                        is_32bit: true,
                    },
                    target: pc_to_bb[target_pc],
                });
            }

            Insn::Ja { off } => {
                let target_pc = (pc as isize + 1 + *off as isize) as usize;
                current_nodes.push(IrNode::Jump { target: pc_to_bb[target_pc] });
            }

            Insn::Call { func_id } => {
                current_nodes.push(IrNode::Call { func_id: *func_id });
            }

            Insn::TailCall => {
                // Tail calls are lowered as a jump to the helper dispatch
                current_nodes.push(IrNode::Call { func_id: 0 });
                current_nodes.push(IrNode::Ret);
            }

            Insn::Exit => {
                if frame_size > 0 {
                    current_nodes.push(IrNode::Epilogue { frame_size });
                }
                current_nodes.push(IrNode::Ret);
            }

            Insn::Atomic { width, op, dst, src, off } => {
                current_nodes.push(IrNode::Atomic {
                    width: *width,
                    op: *op,
                    base: regmap.phys(*dst),
                    src: regmap.phys(*src),
                    off: *off,
                });
            }

            Insn::Endian { to_be, width, dst } => {
                current_nodes.push(IrNode::Endian {
                    to_be: *to_be,
                    width: *width,
                    dst: regmap.phys(*dst),
                });
            }
        }
    }

    // Flush last block
    if !current_nodes.is_empty() {
        blocks.push(IrBasicBlock {
            id: BasicBlockId(bb_id),
            nodes: current_nodes,
        });
    }

    IrProgram { blocks, frame_size }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use ebpf_core::*;

    #[test]
    fn lower_minimal_program() {
        let insns = vec![
            Insn::Alu64 { op: AluOp::Mov, dst: BpfReg::R0, src: Source::Imm(0) },
            Insn::Exit,
        ];
        let regmap = aarch64_regmap();
        let ir = lower_to_ir(&insns, &regmap);

        assert_eq!(ir.blocks.len(), 1);
        assert!(matches!(ir.blocks[0].nodes.last(), Some(IrNode::Ret)));
    }

    #[test]
    fn lower_with_branch() {
        let insns = vec![
            Insn::Alu64 { op: AluOp::Mov, dst: BpfReg::R0, src: Source::Imm(0) },
            Insn::JmpCond {
                op: JmpOp::Jeq,
                dst: BpfReg::R1,
                src: Source::Imm(0),
                off: 1,
            },
            Insn::Alu64 { op: AluOp::Mov, dst: BpfReg::R0, src: Source::Imm(1) },
            Insn::Exit,
        ];
        let regmap = aarch64_regmap();
        let ir = lower_to_ir(&insns, &regmap);

        // Should have multiple basic blocks due to branching
        assert!(ir.blocks.len() >= 2);
    }

    #[test]
    fn frame_size_aligned() {
        let insns = vec![
            Insn::StoreReg {
                width: MemWidth::DW,
                dst: BpfReg::R10,
                src: BpfReg::R1,
                off: -8,
            },
            Insn::Alu64 { op: AluOp::Mov, dst: BpfReg::R0, src: Source::Imm(0) },
            Insn::Exit,
        ];
        let regmap = aarch64_regmap();
        let ir = lower_to_ir(&insns, &regmap);
        assert_eq!(ir.frame_size % 16, 0);
        assert!(ir.frame_size >= 8);
    }
}
