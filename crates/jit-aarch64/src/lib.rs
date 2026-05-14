//! # jit-aarch64
//!
//! AArch64 (ARMv8-A) machine code emitter for the eBPF JIT compiler.
//!
//! Targets: Apple M-series, NVIDIA Jetson Orin, ARM Cortex-R82.
//!
//! All instruction encodings follow the ARM Architecture Reference Manual
//! (ARMv8-A profile). We intentionally avoid NEON/SVE to maintain
//! compatibility with Cortex-R82 (which lacks SIMD).
//!
//! Each AArch64 instruction is exactly 4 bytes (fixed-width encoding).

#![no_std]
#![forbid(unsafe_code)]

extern crate alloc;
use alloc::vec::Vec;

use ebpf_core::{AluOp, JmpOp, MemWidth, AtomicOp, EndianWidth};
use jit_ir::*;

pub mod encode;

use encode::*;

/// Maximum number of target instructions per IR node.
/// Used for dry-run buffer size calculation.
const MAX_EXPANSION_RATIO: usize = 8;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AArch64EmitError {
    BufferOverflow,
    UnsupportedNode,
    RelocationOutOfRange { bb: BasicBlockId },
    InvalidImmediate { value: i64 },
}

/// Pending branch relocation.
#[derive(Debug, Clone)]
struct Relocation {
    /// Offset in the output buffer where the branch instruction lives.
    offset: usize,
    /// Target basic block.
    target: BasicBlockId,
    /// Is this a conditional branch (B.cond) or unconditional (B)?
    is_conditional: bool,
}

/// AArch64 code emitter.
pub struct AArch64Emitter {
    buf: Vec<u8>,
    /// Map from BasicBlockId to byte offset in buf.
    bb_offsets: Vec<Option<usize>>,
    relocations: Vec<Relocation>,
}

impl AArch64Emitter {
    pub fn new(estimated_size: usize) -> Self {
        Self {
            buf: Vec::with_capacity(estimated_size),
            bb_offsets: Vec::new(),
            relocations: Vec::new(),
        }
    }

    fn emit32(&mut self, insn: u32) {
        self.buf.extend_from_slice(&insn.to_le_bytes());
    }

    fn current_offset(&self) -> usize {
        self.buf.len()
    }

    fn ensure_bb_map(&mut self, bb: BasicBlockId) {
        let idx = bb.0 as usize;
        if self.bb_offsets.len() <= idx {
            self.bb_offsets.resize(idx + 1, None);
        }
    }

    /// Emit a 64-bit immediate into a register using MOV/MOVK sequence.
    /// Worst case: 4 instructions (one per 16-bit chunk).
    /// O(1).
    fn emit_load_imm64(&mut self, dst: PhysReg, imm: u64) {
        let chunks = [
            (imm & 0xFFFF) as u16,
            ((imm >> 16) & 0xFFFF) as u16,
            ((imm >> 32) & 0xFFFF) as u16,
            ((imm >> 48) & 0xFFFF) as u16,
        ];

        // Find highest non-zero chunk for minimal instruction count
        let top = chunks.iter().rposition(|&c| c != 0).unwrap_or(0);

        // MOVZ for the lowest chunk
        self.emit32(movz_x(dst.0, chunks[0], 0));

        // MOVK for remaining non-zero chunks
        for (shift, &chunk) in chunks.iter().enumerate().skip(1) {
            if shift <= top {
                if chunk != 0 {
                    self.emit32(movk_x(dst.0, chunk, shift as u8));
                }
            }
        }
    }

    /// Emit an immediate into the tmp register if needed, returning
    /// the register containing the value.
    fn materialize_imm(&mut self, val: i64, tmp: PhysReg) -> PhysReg {
        self.emit_load_imm64(tmp, val as u64);
        tmp
    }

    fn emit_alu64_node(&mut self, op: &AluOp, dst: PhysReg, src: &Operand) {
        let tmp = PhysReg(9); // X9 — caller-saved scratch

        match (op, src) {
            // --- Register-register forms ---
            (AluOp::Add, Operand::Reg(rs)) => {
                // ADD Xd, Xd, Xs
                self.emit32(add_x_reg(dst.0, dst.0, rs.0));
            }
            (AluOp::Sub, Operand::Reg(rs)) => {
                // SUB Xd, Xd, Xs
                self.emit32(sub_x_reg(dst.0, dst.0, rs.0));
            }
            (AluOp::Mul, Operand::Reg(rs)) => {
                // MUL Xd, Xd, Xs  (alias: MADD Xd, Xd, Xs, XZR)
                self.emit32(mul_x(dst.0, dst.0, rs.0));
            }
            (AluOp::Div, Operand::Reg(rs)) => {
                // UDIV Xd, Xd, Xs
                self.emit32(udiv_x(dst.0, dst.0, rs.0));
            }
            (AluOp::Mod, Operand::Reg(rs)) => {
                // Use X12 as secondary scratch to avoid clobbering src if src==tmp.
                // UDIV X12, Xd, Xs; MSUB Xd, X12, Xs, Xd → Xd = Xd - (Xd/Xs)*Xs
                self.emit32(udiv_x(12, dst.0, rs.0));
                self.emit32(msub_x(dst.0, 12, rs.0, dst.0));
            }
            (AluOp::Or, Operand::Reg(rs)) => {
                // ORR Xd, Xd, Xs
                self.emit32(orr_x_reg(dst.0, dst.0, rs.0));
            }
            (AluOp::And, Operand::Reg(rs)) => {
                // AND Xd, Xd, Xs
                self.emit32(and_x_reg(dst.0, dst.0, rs.0));
            }
            (AluOp::Xor, Operand::Reg(rs)) => {
                // EOR Xd, Xd, Xs
                self.emit32(eor_x_reg(dst.0, dst.0, rs.0));
            }
            (AluOp::Lsh, Operand::Reg(rs)) => {
                // LSL Xd, Xd, Xs (alias: LSLV)
                self.emit32(lslv_x(dst.0, dst.0, rs.0));
            }
            (AluOp::Rsh, Operand::Reg(rs)) => {
                // LSR Xd, Xd, Xs (alias: LSRV)
                self.emit32(lsrv_x(dst.0, dst.0, rs.0));
            }
            (AluOp::Arsh, Operand::Reg(rs)) => {
                // ASR Xd, Xd, Xs (alias: ASRV)
                self.emit32(asrv_x(dst.0, dst.0, rs.0));
            }
            (AluOp::Mov, Operand::Reg(rs)) => {
                // MOV Xd, Xs (alias: ORR Xd, XZR, Xs)
                self.emit32(mov_x_reg(dst.0, rs.0));
            }
            (AluOp::Neg, _) => {
                // NEG Xd, Xd (alias: SUB Xd, XZR, Xd)
                self.emit32(sub_x_reg(dst.0, 31, dst.0));
            }

            // --- Register-immediate forms ---
            (AluOp::Add, Operand::Imm(v)) if *v >= 0 && *v < 4096 => {
                self.emit32(add_x_imm(dst.0, dst.0, *v as u16));
            }
            (AluOp::Sub, Operand::Imm(v)) if *v >= 0 && *v < 4096 => {
                self.emit32(sub_x_imm(dst.0, dst.0, *v as u16));
            }
            (AluOp::Mov, Operand::Imm(v)) => {
                self.emit_load_imm64(dst, *v as u64);
            }
            (AluOp::Lsh, Operand::Imm(v)) => {
                // UBFM alias for LSL with immediate
                let shift = (*v & 63) as u8;
                self.emit32(lsl_x_imm(dst.0, dst.0, shift));
            }
            (AluOp::Rsh, Operand::Imm(v)) => {
                let shift = (*v & 63) as u8;
                self.emit32(lsr_x_imm(dst.0, dst.0, shift));
            }
            (AluOp::Arsh, Operand::Imm(v)) => {
                let shift = (*v & 63) as u8;
                self.emit32(asr_x_imm(dst.0, dst.0, shift));
            }

            // Fallback: materialize immediate into tmp, then reg-reg
            (_, Operand::Imm(v)) => {
                let rs = self.materialize_imm(*v, tmp);
                self.emit_alu64_node(op, dst, &Operand::Reg(rs));
            }
        }
    }

    fn emit_alu32_node(&mut self, op: &AluOp, dst: PhysReg, src: &Operand) {
        let tmp = PhysReg(9);

        match (op, src) {
            (AluOp::Add, Operand::Reg(rs)) => self.emit32(add_w_reg(dst.0, dst.0, rs.0)),
            (AluOp::Sub, Operand::Reg(rs)) => self.emit32(sub_w_reg(dst.0, dst.0, rs.0)),
            (AluOp::Mul, Operand::Reg(rs)) => self.emit32(mul_w(dst.0, dst.0, rs.0)),
            (AluOp::Div, Operand::Reg(rs)) => self.emit32(udiv_w(dst.0, dst.0, rs.0)),
            (AluOp::Or, Operand::Reg(rs)) => self.emit32(orr_w_reg(dst.0, dst.0, rs.0)),
            (AluOp::And, Operand::Reg(rs)) => self.emit32(and_w_reg(dst.0, dst.0, rs.0)),
            (AluOp::Xor, Operand::Reg(rs)) => self.emit32(eor_w_reg(dst.0, dst.0, rs.0)),
            (AluOp::Lsh, Operand::Reg(rs)) => self.emit32(lslv_w(dst.0, dst.0, rs.0)),
            (AluOp::Rsh, Operand::Reg(rs)) => self.emit32(lsrv_w(dst.0, dst.0, rs.0)),
            (AluOp::Arsh, Operand::Reg(rs)) => self.emit32(asrv_w(dst.0, dst.0, rs.0)),
            (AluOp::Mov, Operand::Reg(rs)) => self.emit32(mov_w_reg(dst.0, rs.0)),
            (AluOp::Neg, _) => self.emit32(sub_w_reg(dst.0, 31, dst.0)),
            (AluOp::Mod, Operand::Reg(rs)) => {
                self.emit32(udiv_w(12, dst.0, rs.0));
                self.emit32(msub_w(dst.0, 12, rs.0, dst.0));
            }
            (AluOp::Mov, Operand::Imm(v)) => {
                self.emit32(movz_w(dst.0, *v as u16, 0));
                if *v as u64 > 0xFFFF {
                    self.emit32(movk_w(dst.0, ((*v as u64) >> 16) as u16, 1));
                }
            }
            (AluOp::Add, Operand::Imm(v)) if *v >= 0 && *v < 4096 => {
                self.emit32(add_w_imm(dst.0, dst.0, *v as u16));
            }
            (AluOp::Sub, Operand::Imm(v)) if *v >= 0 && *v < 4096 => {
                self.emit32(sub_w_imm(dst.0, dst.0, *v as u16));
            }
            (_, Operand::Imm(v)) => {
                let rs = self.materialize_imm(*v, tmp);
                self.emit_alu32_node(op, dst, &Operand::Reg(rs));
            }
        }
    }

    fn emit_load_node(&mut self, width: &MemWidth, dst: PhysReg, base: PhysReg, off: i16) {
        match width {
            MemWidth::B => self.emit32(ldrb_imm(dst.0, base.0, off)),
            MemWidth::H => self.emit32(ldrh_imm(dst.0, base.0, off)),
            MemWidth::W => self.emit32(ldr_w_imm(dst.0, base.0, off)),
            MemWidth::DW => self.emit32(ldr_x_imm(dst.0, base.0, off)),
        }
    }

    fn emit_store_node(&mut self, width: &MemWidth, base: PhysReg, src: PhysReg, off: i16) {
        match width {
            MemWidth::B => self.emit32(strb_imm(src.0, base.0, off)),
            MemWidth::H => self.emit32(strh_imm(src.0, base.0, off)),
            MemWidth::W => self.emit32(str_w_imm(src.0, base.0, off)),
            MemWidth::DW => self.emit32(str_x_imm(src.0, base.0, off)),
        }
    }

    fn emit_branch_node(&mut self, cond: &BranchCond, target: BasicBlockId) {
        let tmp = PhysReg(9);

        // Materialize RHS if immediate
        let rhs_reg = match &cond.rhs {
            Operand::Reg(r) => *r,
            Operand::Imm(v) => {
                self.materialize_imm(*v, tmp);
                tmp
            }
        };

        // CMP (SUBS XZR, Xn, Xm)
        if cond.is_32bit {
            self.emit32(cmp_w_reg(cond.lhs.0, rhs_reg.0));
        } else {
            self.emit32(cmp_x_reg(cond.lhs.0, rhs_reg.0));
        }

        // B.cond <target> — offset will be patched during finalization
        let cc = jmpop_to_condition_code(&cond.op);
        let reloc_offset = self.current_offset();
        self.emit32(b_cond(cc, 0)); // placeholder offset

        self.relocations.push(Relocation {
            offset: reloc_offset,
            target,
            is_conditional: true,
        });
    }

    fn emit_atomic_node(
        &mut self,
        _width: &MemWidth,
        op: &AtomicOp,
        base: PhysReg,
        src: PhysReg,
        off: i16,
    ) {
        let tmp = PhysReg(9);

        // Add offset to base in tmp
        if off != 0 {
            self.emit_load_imm64(tmp, off as i64 as u64);
            self.emit32(add_x_reg(tmp.0, base.0, tmp.0));
        } else {
            self.emit32(mov_x_reg(tmp.0, base.0));
        }

        // DMB ISH — full barrier (conservative, sequentially consistent)
        self.emit32(dmb_ish());

        match op {
            AtomicOp::Add | AtomicOp::FetchAdd => {
                // LDAXR Xtmp2, [Xtmp]; ADD Xtmp2, Xtmp2, Xsrc; STLXR W-status, Xtmp2, [Xtmp]
                // Simplified: use LDADD (LSE atomics, ARMv8.1+)
                self.emit32(ldadd_x(src.0, 31, tmp.0)); // XZR as old-value discard
            }
            AtomicOp::Or | AtomicOp::FetchOr => {
                self.emit32(ldset_x(src.0, 31, tmp.0));
            }
            AtomicOp::And | AtomicOp::FetchAnd => {
                self.emit32(ldclr_x(src.0, 31, tmp.0));
            }
            AtomicOp::Xor | AtomicOp::FetchXor => {
                self.emit32(ldeor_x(src.0, 31, tmp.0));
            }
            AtomicOp::Xchg => {
                self.emit32(swp_x(src.0, 31, tmp.0));
            }
            AtomicOp::CmpXchg => {
                // CAS: compare R0, swap with src at [tmp]
                self.emit32(cas_x(0, src.0, tmp.0)); // R0 = BPF_R0 mapped register
            }
        }

        // DMB ISH — trailing barrier
        self.emit32(dmb_ish());
    }

    fn emit_endian_node(&mut self, to_be: bool, width: &EndianWidth, dst: PhysReg) {
        // On little-endian AArch64, to_be means byte-swap, to_le is a no-op
        if to_be {
            match width {
                EndianWidth::Bits16 => {
                    // REV16 Wd, Wd; AND Xd, Xd, #0xFFFF
                    self.emit32(rev16_w(dst.0, dst.0));
                    self.emit32(and_x_imm_mask(dst.0, dst.0, 0xFFFF));
                }
                EndianWidth::Bits32 => {
                    // REV Wd, Wd (32-bit byte reversal)
                    self.emit32(rev_w(dst.0, dst.0));
                }
                EndianWidth::Bits64 => {
                    // REV Xd, Xd (64-bit byte reversal)
                    self.emit32(rev_x(dst.0, dst.0));
                }
            }
        }
        // LE on LE hardware = no-op
    }
}

impl CodeEmitter for AArch64Emitter {
    type Error = AArch64EmitError;

    fn emit_node(&mut self, node: &IrNode) -> Result<(), Self::Error> {
        match node {
            IrNode::Alu64 { op, dst, src } => {
                self.emit_alu64_node(op, *dst, src);
            }
            IrNode::Alu32 { op, dst, src } => {
                self.emit_alu32_node(op, *dst, src);
            }
            IrNode::Load { width, dst, base, off } => {
                self.emit_load_node(width, *dst, *base, *off);
            }
            IrNode::Store { width, base, src, off } => {
                self.emit_store_node(width, *base, *src, *off);
            }
            IrNode::StoreImm { width, base, off, imm } => {
                let tmp = PhysReg(9);
                self.emit_load_imm64(tmp, *imm as i64 as u64);
                self.emit_store_node(width, *base, tmp, *off);
            }
            IrNode::LoadImm64 { dst, imm } => {
                self.emit_load_imm64(*dst, *imm);
            }
            IrNode::Branch { cond, target } => {
                self.emit_branch_node(cond, *target);
            }
            IrNode::Jump { target } => {
                let reloc_offset = self.current_offset();
                self.emit32(b_imm(0)); // placeholder
                self.relocations.push(Relocation {
                    offset: reloc_offset,
                    target: *target,
                    is_conditional: false,
                });
            }
            IrNode::Call { func_id: _ } => {
                // BLR X9 — caller must have loaded helper address into tmp.
                // In practice, the engine pre-populates a jump table.
                self.emit32(blr(9));
            }
            IrNode::Ret => {
                // Move eBPF R0 (X7) to ABI return register (X0) before returning.
                // This is specific to our AArch64 register mapping where eBPF R0 → X7
                // but the C calling convention expects the return value in X0.
                self.emit32(mov_x_reg(0, 7));
                self.emit32(ret());
            }
            IrNode::Spill { reg, slot } => {
                // STR Xreg, [SP, #slot.offset]
                self.emit32(str_x_imm(reg.0, 31, slot.offset));
            }
            IrNode::Fill { reg, slot } => {
                // LDR Xreg, [SP, #slot.offset]
                self.emit32(ldr_x_imm(reg.0, 31, slot.offset));
            }
            IrNode::Atomic { width, op, base, src, off } => {
                self.emit_atomic_node(width, op, *base, *src, *off);
            }
            IrNode::Endian { to_be, width, dst } => {
                self.emit_endian_node(*to_be, width, *dst);
            }
            IrNode::Label { bb } => {
                self.ensure_bb_map(*bb);
                self.bb_offsets[bb.0 as usize] = Some(self.current_offset());
            }
            IrNode::Prologue { frame_size } => {
                // STP X29, X30, [SP, #-frame_size]!
                self.emit32(stp_pre(29, 30, 31, -(*frame_size as i16)));
                // MOV X29, SP
                self.emit32(mov_x_reg(29, 31));
                // Save callee-saved registers that we use (X19-X22, X25)
                self.emit32(stp_offset(19, 20, 31, 16));
                self.emit32(stp_offset(21, 22, 31, 32));
                self.emit32(str_x_imm(25, 31, 48));
            }
            IrNode::Epilogue { frame_size } => {
                // Restore callee-saved registers
                self.emit32(ldp_offset(19, 20, 31, 16));
                self.emit32(ldp_offset(21, 22, 31, 32));
                self.emit32(ldr_x_imm(25, 31, 48));
                // LDP X29, X30, [SP], #frame_size
                self.emit32(ldp_post(29, 30, 31, *frame_size as i16));
            }
        }
        Ok(())
    }

    fn finalize(&mut self) -> Result<Vec<u8>, Self::Error> {
        // Resolve all branch relocations
        for reloc in &self.relocations {
            let target_offset = self.bb_offsets
                .get(reloc.target.0 as usize)
                .and_then(|o| *o)
                .ok_or(AArch64EmitError::RelocationOutOfRange { bb: reloc.target })?;

            let branch_offset = target_offset as isize - reloc.offset as isize;
            let imm = branch_offset / 4; // AArch64 branches are in units of 4 bytes

            if reloc.is_conditional {
                // B.cond: 19-bit signed offset
                if imm < -(1 << 18) || imm >= (1 << 18) {
                    return Err(AArch64EmitError::RelocationOutOfRange { bb: reloc.target });
                }
                let existing = u32::from_le_bytes([
                    self.buf[reloc.offset],
                    self.buf[reloc.offset + 1],
                    self.buf[reloc.offset + 2],
                    self.buf[reloc.offset + 3],
                ]);
                let patched = (existing & 0xFF00001F) | (((imm as u32) & 0x7FFFF) << 5);
                self.buf[reloc.offset..reloc.offset + 4]
                    .copy_from_slice(&patched.to_le_bytes());
            } else {
                // B: 26-bit signed offset
                if imm < -(1 << 25) || imm >= (1 << 25) {
                    return Err(AArch64EmitError::RelocationOutOfRange { bb: reloc.target });
                }
                let patched = 0x14000000u32 | ((imm as u32) & 0x03FFFFFF);
                self.buf[reloc.offset..reloc.offset + 4]
                    .copy_from_slice(&patched.to_le_bytes());
            }
        }

        Ok(self.buf.clone())
    }

    fn calculate_size(&self, program: &IrProgram) -> usize {
        let mut count = 0;
        for block in &program.blocks {
            count += 1; // label (zero-width, but count for safety)
            count += block.nodes.len() * MAX_EXPANSION_RATIO;
        }
        count * 4 // 4 bytes per instruction
    }
}

/// Map eBPF JmpOp to AArch64 condition code.
fn jmpop_to_condition_code(op: &JmpOp) -> u8 {
    match op {
        JmpOp::Jeq => CC_EQ,
        JmpOp::Jne => CC_NE,
        JmpOp::Jgt => CC_HI,   // unsigned greater than
        JmpOp::Jge => CC_HS,   // unsigned greater or equal (carry set)
        JmpOp::Jlt => CC_LO,   // unsigned less than (carry clear)
        JmpOp::Jle => CC_LS,   // unsigned less or equal
        JmpOp::Jsgt => CC_GT,  // signed greater than
        JmpOp::Jsge => CC_GE,  // signed greater or equal
        JmpOp::Jslt => CC_LT,  // signed less than
        JmpOp::Jsle => CC_LE,  // signed less or equal
        JmpOp::Jset => CC_NE,  // TST + BNE (handled separately)
        JmpOp::Ja => CC_AL,    // always (shouldn't reach here)
    }
}
