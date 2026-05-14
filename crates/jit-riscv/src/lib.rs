//! # jit-riscv
//!
//! RISC-V RV64IMC machine code emitter for the eBPF JIT compiler.
//!
//! Targets: RISC-V based NVMe controllers (e.g., ScaleFlux, future designs).
//!
//! Instruction encoding follows the RISC-V Unprivileged ISA Specification
//! (Volume I, Version 20191213). We use the RV64I base + M (multiply/divide)
//! + A (atomics) extensions. Compressed (C) instructions are not used in
//! the emitter to simplify relocation handling.
//!
//! RISC-V instructions are 4 bytes (32-bit) in the base ISA.

#![no_std]
#![forbid(unsafe_code)]

extern crate alloc;
use alloc::vec::Vec;

use ebpf_core::{AluOp, JmpOp, MemWidth, AtomicOp, EndianWidth};
use jit_ir::*;

pub mod encode;

use encode::*;

const MAX_EXPANSION_RATIO: usize = 10;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RiscVEmitError {
    BufferOverflow,
    UnsupportedNode,
    RelocationOutOfRange { bb: BasicBlockId },
    InvalidImmediate { value: i64 },
}

#[derive(Debug, Clone)]
struct Relocation {
    offset: usize,
    target: BasicBlockId,
    kind: RelocKind,
}

#[derive(Debug, Clone)]
enum RelocKind {
    /// JAL (J-type): 20-bit signed offset
    Jal,
    /// Branch (B-type): 12-bit signed offset
    Branch,
}

/// RISC-V RV64 code emitter.
pub struct RiscVEmitter {
    buf: Vec<u8>,
    bb_offsets: Vec<Option<usize>>,
    relocations: Vec<Relocation>,
}

impl RiscVEmitter {
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

    /// Load a 64-bit immediate into a register.
    /// Uses LUI + ADDI sequences for up to 64-bit values.
    /// Worst case: 6 instructions.
    fn emit_load_imm64(&mut self, rd: u8, imm: u64) {
        if imm == 0 {
            // ADDI rd, x0, 0
            self.emit32(addi(rd, 0, 0));
            return;
        }

        let val = imm as i64;

        // Check if fits in 12-bit signed immediate
        if val >= -2048 && val < 2048 {
            self.emit32(addi(rd, 0, val as i32));
            return;
        }

        // Check if fits in 32-bit (LUI + ADDI)
        if val >= -2147483648 && val < 2147483648 {
            let hi = ((val + 0x800) >> 12) as i32;
            let lo = (val as i32) & 0xFFF;
            self.emit32(lui(rd, hi));
            if lo != 0 {
                self.emit32(addi(rd, rd, sign_extend_12(lo)));
            }
            return;
        }

        // Full 64-bit: load upper 32 bits, shift, add lower 32 bits
        let upper = (imm >> 32) as i64;
        let lower = (imm & 0xFFFFFFFF) as i64;

        // Load upper 32 bits
        let hi_upper = ((upper + 0x800) >> 12) as i32;
        let lo_upper = (upper as i32) & 0xFFF;
        self.emit32(lui(rd, hi_upper));
        if lo_upper != 0 {
            self.emit32(addi(rd, rd, sign_extend_12(lo_upper)));
        }

        // Shift left 32
        self.emit32(slli(rd, rd, 32));

        // Load and add lower 32 bits
        let tmp = 23u8; // s7 — JIT temp register
        let hi_lower = ((lower + 0x800) >> 12) as i32;
        let lo_lower = (lower as i32) & 0xFFF;
        if hi_lower != 0 {
            self.emit32(lui(tmp, hi_lower));
            if lo_lower != 0 {
                self.emit32(addi(tmp, tmp, sign_extend_12(lo_lower)));
            }
            self.emit32(add(rd, rd, tmp));
        } else if lo_lower != 0 {
            self.emit32(addi(tmp, 0, sign_extend_12(lo_lower)));
            self.emit32(add(rd, rd, tmp));
        }
    }

    fn materialize_imm(&mut self, val: i64, tmp: u8) -> u8 {
        self.emit_load_imm64(tmp, val as u64);
        tmp
    }

    fn emit_alu64_node(&mut self, op: &AluOp, dst: PhysReg, src: &Operand) {
        let tmp = 23u8; // s7

        match (op, src) {
            (AluOp::Add, Operand::Reg(rs)) => self.emit32(add(dst.0, dst.0, rs.0)),
            (AluOp::Sub, Operand::Reg(rs)) => self.emit32(sub(dst.0, dst.0, rs.0)),
            (AluOp::Mul, Operand::Reg(rs)) => self.emit32(mul(dst.0, dst.0, rs.0)),
            (AluOp::Div, Operand::Reg(rs)) => self.emit32(divu(dst.0, dst.0, rs.0)),
            (AluOp::Mod, Operand::Reg(rs)) => self.emit32(remu(dst.0, dst.0, rs.0)),
            (AluOp::Or, Operand::Reg(rs)) => self.emit32(or(dst.0, dst.0, rs.0)),
            (AluOp::And, Operand::Reg(rs)) => self.emit32(and(dst.0, dst.0, rs.0)),
            (AluOp::Xor, Operand::Reg(rs)) => self.emit32(xor(dst.0, dst.0, rs.0)),
            (AluOp::Lsh, Operand::Reg(rs)) => self.emit32(sll(dst.0, dst.0, rs.0)),
            (AluOp::Rsh, Operand::Reg(rs)) => self.emit32(srl(dst.0, dst.0, rs.0)),
            (AluOp::Arsh, Operand::Reg(rs)) => self.emit32(sra(dst.0, dst.0, rs.0)),
            (AluOp::Mov, Operand::Reg(rs)) => self.emit32(addi(dst.0, rs.0, 0)), // MV pseudo
            (AluOp::Neg, _) => self.emit32(sub(dst.0, 0, dst.0)),

            (AluOp::Add, Operand::Imm(v)) if *v >= -2048 && *v < 2048 => {
                self.emit32(addi(dst.0, dst.0, *v as i32));
            }
            (AluOp::Mov, Operand::Imm(v)) => {
                self.emit_load_imm64(dst.0, *v as u64);
            }
            (AluOp::Lsh, Operand::Imm(v)) => {
                self.emit32(slli(dst.0, dst.0, (*v & 63) as u32));
            }
            (AluOp::Rsh, Operand::Imm(v)) => {
                self.emit32(srli(dst.0, dst.0, (*v & 63) as u32));
            }
            (AluOp::Arsh, Operand::Imm(v)) => {
                self.emit32(srai(dst.0, dst.0, (*v & 63) as u32));
            }
            (AluOp::And, Operand::Imm(v)) if *v >= -2048 && *v < 2048 => {
                self.emit32(andi(dst.0, dst.0, *v as i32));
            }
            (AluOp::Or, Operand::Imm(v)) if *v >= -2048 && *v < 2048 => {
                self.emit32(ori(dst.0, dst.0, *v as i32));
            }
            (AluOp::Xor, Operand::Imm(v)) if *v >= -2048 && *v < 2048 => {
                self.emit32(xori(dst.0, dst.0, *v as i32));
            }

            (_, Operand::Imm(v)) => {
                let rs = self.materialize_imm(*v, tmp);
                self.emit_alu64_node(op, dst, &Operand::Reg(PhysReg(rs)));
            }
        }
    }

    fn emit_alu32_node(&mut self, op: &AluOp, dst: PhysReg, src: &Operand) {
        let tmp = 23u8;

        match (op, src) {
            (AluOp::Add, Operand::Reg(rs)) => self.emit32(addw(dst.0, dst.0, rs.0)),
            (AluOp::Sub, Operand::Reg(rs)) => self.emit32(subw(dst.0, dst.0, rs.0)),
            (AluOp::Mul, Operand::Reg(rs)) => self.emit32(mulw(dst.0, dst.0, rs.0)),
            (AluOp::Div, Operand::Reg(rs)) => self.emit32(divuw(dst.0, dst.0, rs.0)),
            (AluOp::Mod, Operand::Reg(rs)) => self.emit32(remuw(dst.0, dst.0, rs.0)),
            (AluOp::Lsh, Operand::Reg(rs)) => self.emit32(sllw(dst.0, dst.0, rs.0)),
            (AluOp::Rsh, Operand::Reg(rs)) => self.emit32(srlw(dst.0, dst.0, rs.0)),
            (AluOp::Arsh, Operand::Reg(rs)) => self.emit32(sraw(dst.0, dst.0, rs.0)),
            (AluOp::Mov, Operand::Reg(rs)) => {
                // Zero-extend 32-bit: SLLI + SRLI
                self.emit32(addi(dst.0, rs.0, 0));
                self.emit32(slli(dst.0, dst.0, 32));
                self.emit32(srli(dst.0, dst.0, 32));
            }
            (AluOp::Or, Operand::Reg(rs)) => {
                self.emit32(or(dst.0, dst.0, rs.0));
                self.emit32(slli(dst.0, dst.0, 32));
                self.emit32(srli(dst.0, dst.0, 32));
            }
            (AluOp::And, Operand::Reg(rs)) => {
                self.emit32(and(dst.0, dst.0, rs.0));
                self.emit32(slli(dst.0, dst.0, 32));
                self.emit32(srli(dst.0, dst.0, 32));
            }
            (AluOp::Xor, Operand::Reg(rs)) => {
                self.emit32(xor(dst.0, dst.0, rs.0));
                self.emit32(slli(dst.0, dst.0, 32));
                self.emit32(srli(dst.0, dst.0, 32));
            }
            (AluOp::Neg, _) => {
                self.emit32(subw(dst.0, 0, dst.0));
            }
            (_, Operand::Imm(v)) => {
                let rs = self.materialize_imm(*v, tmp);
                self.emit_alu32_node(op, dst, &Operand::Reg(PhysReg(rs)));
            }
        }
    }

    fn emit_load_node(&mut self, width: &MemWidth, dst: PhysReg, base: PhysReg, off: i16) {
        match width {
            MemWidth::B => self.emit32(lbu(dst.0, base.0, off as i32)),
            MemWidth::H => self.emit32(lhu(dst.0, base.0, off as i32)),
            MemWidth::W => self.emit32(lwu(dst.0, base.0, off as i32)),
            MemWidth::DW => self.emit32(ld(dst.0, base.0, off as i32)),
        }
    }

    fn emit_store_node(&mut self, width: &MemWidth, base: PhysReg, src: PhysReg, off: i16) {
        match width {
            MemWidth::B => self.emit32(sb(src.0, base.0, off as i32)),
            MemWidth::H => self.emit32(sh(src.0, base.0, off as i32)),
            MemWidth::W => self.emit32(sw(src.0, base.0, off as i32)),
            MemWidth::DW => self.emit32(sd(src.0, base.0, off as i32)),
        }
    }

    fn emit_branch_node(&mut self, cond: &BranchCond, target: BasicBlockId) {
        let tmp = 23u8;

        let rhs_reg = match &cond.rhs {
            Operand::Reg(r) => r.0,
            Operand::Imm(v) => {
                self.materialize_imm(*v, tmp);
                tmp
            }
        };

        let reloc_offset = self.current_offset();

        // Map JmpOp to RISC-V branch instruction
        let insn = match cond.op {
            JmpOp::Jeq => beq(cond.lhs.0, rhs_reg, 0),
            JmpOp::Jne => bne(cond.lhs.0, rhs_reg, 0),
            JmpOp::Jgt => bltu(rhs_reg, cond.lhs.0, 0),  // unsigned: a > b ↔ b < a
            JmpOp::Jge => bgeu(cond.lhs.0, rhs_reg, 0),
            JmpOp::Jlt => bltu(cond.lhs.0, rhs_reg, 0),
            JmpOp::Jle => bgeu(rhs_reg, cond.lhs.0, 0),   // unsigned: a <= b ↔ b >= a
            JmpOp::Jsgt => blt(rhs_reg, cond.lhs.0, 0),
            JmpOp::Jsge => bge(cond.lhs.0, rhs_reg, 0),
            JmpOp::Jslt => blt(cond.lhs.0, rhs_reg, 0),
            JmpOp::Jsle => bge(rhs_reg, cond.lhs.0, 0),
            JmpOp::Jset => {
                // AND tmp, lhs, rhs; BNE tmp, x0, target
                self.emit32(and(tmp, cond.lhs.0, rhs_reg));
                bne(tmp, 0, 0)
            }
            JmpOp::Ja => jal(0, 0), // shouldn't happen for conditional
        };

        self.emit32(insn);
        self.relocations.push(Relocation {
            offset: reloc_offset + if matches!(cond.op, JmpOp::Jset) { 4 } else { 0 },
            target,
            kind: RelocKind::Branch,
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
        let tmp = 23u8; // s7

        // Add offset to base if nonzero
        if off != 0 {
            self.emit32(addi(tmp, base.0, off as i32));
        } else {
            self.emit32(addi(tmp, base.0, 0)); // MV
        }

        // FENCE rw, rw — full barrier
        self.emit32(fence_rw_rw());

        match op {
            AtomicOp::Add | AtomicOp::FetchAdd => {
                self.emit32(amoadd_d(0, src.0, tmp)); // rd=x0 discards old value
            }
            AtomicOp::Or | AtomicOp::FetchOr => {
                self.emit32(amoor_d(0, src.0, tmp));
            }
            AtomicOp::And | AtomicOp::FetchAnd => {
                self.emit32(amoand_d(0, src.0, tmp));
            }
            AtomicOp::Xor | AtomicOp::FetchXor => {
                self.emit32(amoxor_d(0, src.0, tmp));
            }
            AtomicOp::Xchg => {
                self.emit32(amoswap_d(0, src.0, tmp));
            }
            AtomicOp::CmpXchg => {
                // LR/SC loop for CAS
                let loop_start = self.current_offset();
                self.emit32(lr_d(tmp, tmp));            // LR.D tmp2, (addr)
                // BNE tmp2, expected, done
                self.emit32(bne(tmp, 10, 12));          // Skip SC if mismatch (a0 = R0)
                self.emit32(sc_d(tmp, src.0, tmp));     // SC.D tmp, new, (addr)
                self.emit32(bne(tmp, 0, loop_start as i32 - self.current_offset() as i32));
            }
        }

        self.emit32(fence_rw_rw());
    }

    fn emit_endian_node(&mut self, _to_be: bool, _width: &EndianWidth, dst: PhysReg) {
        // RISC-V is typically little-endian. For byte-swap (to_be on LE),
        // we need a manual byte reversal sequence.
        // For now, emit a no-op placeholder — real implementation would use
        // Zbb REV8 if available, or a shift-and-mask sequence.
        self.emit32(addi(dst.0, dst.0, 0)); // NOP placeholder
    }
}

impl CodeEmitter for RiscVEmitter {
    type Error = RiscVEmitError;

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
                let tmp = 23u8;
                self.emit_load_imm64(tmp, *imm as i64 as u64);
                self.emit_store_node(width, *base, PhysReg(tmp), *off);
            }
            IrNode::LoadImm64 { dst, imm } => {
                self.emit_load_imm64(dst.0, *imm);
            }
            IrNode::Branch { cond, target } => {
                self.emit_branch_node(cond, *target);
            }
            IrNode::Jump { target } => {
                let reloc_offset = self.current_offset();
                self.emit32(jal(0, 0)); // JAL x0, offset (unconditional jump, no link)
                self.relocations.push(Relocation {
                    offset: reloc_offset,
                    target: *target,
                    kind: RelocKind::Jal,
                });
            }
            IrNode::Call { func_id: _ } => {
                // JALR ra, 0(s7) — helper address pre-loaded in s7
                self.emit32(jalr(1, 23, 0));
            }
            IrNode::Ret => {
                self.emit32(jalr(0, 1, 0)); // JR ra
            }
            IrNode::Spill { reg, slot } => {
                self.emit32(sd(reg.0, 2, slot.offset as i32)); // SD reg, offset(sp)
            }
            IrNode::Fill { reg, slot } => {
                self.emit32(ld(reg.0, 2, slot.offset as i32)); // LD reg, offset(sp)
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
                // ADDI sp, sp, -frame_size
                self.emit32(addi(2, 2, -(*frame_size as i32)));
                // SD ra, frame_size-8(sp)
                self.emit32(sd(1, 2, *frame_size as i32 - 8));
                // SD s0/fp, frame_size-16(sp)
                self.emit32(sd(8, 2, *frame_size as i32 - 16));
                // Save callee-saved: s2-s6 (eBPF R6-R10)
                self.emit32(sd(18, 2, *frame_size as i32 - 24)); // s2
                self.emit32(sd(19, 2, *frame_size as i32 - 32)); // s3
                self.emit32(sd(20, 2, *frame_size as i32 - 40)); // s4
                self.emit32(sd(21, 2, *frame_size as i32 - 48)); // s5
                self.emit32(sd(22, 2, *frame_size as i32 - 56)); // s6
                // ADDI s0, sp, frame_size
                self.emit32(addi(8, 2, *frame_size as i32));
            }
            IrNode::Epilogue { frame_size } => {
                // Restore callee-saved
                self.emit32(ld(22, 2, *frame_size as i32 - 56));
                self.emit32(ld(21, 2, *frame_size as i32 - 48));
                self.emit32(ld(20, 2, *frame_size as i32 - 40));
                self.emit32(ld(19, 2, *frame_size as i32 - 32));
                self.emit32(ld(18, 2, *frame_size as i32 - 24));
                self.emit32(ld(8, 2, *frame_size as i32 - 16));
                self.emit32(ld(1, 2, *frame_size as i32 - 8));
                // ADDI sp, sp, frame_size
                self.emit32(addi(2, 2, *frame_size as i32));
            }
        }
        Ok(())
    }

    fn finalize(&mut self) -> Result<Vec<u8>, Self::Error> {
        for reloc in &self.relocations {
            let target_offset = self.bb_offsets
                .get(reloc.target.0 as usize)
                .and_then(|o| *o)
                .ok_or(RiscVEmitError::RelocationOutOfRange { bb: reloc.target })?;

            let offset = target_offset as isize - reloc.offset as isize;

            match reloc.kind {
                RelocKind::Jal => {
                    if offset < -(1 << 20) || offset >= (1 << 20) {
                        return Err(RiscVEmitError::RelocationOutOfRange { bb: reloc.target });
                    }
                    let existing = u32::from_le_bytes([
                        self.buf[reloc.offset],
                        self.buf[reloc.offset + 1],
                        self.buf[reloc.offset + 2],
                        self.buf[reloc.offset + 3],
                    ]);
                    let rd = existing & 0xF80;
                    let patched = encode_j_type_imm(offset as i32) | 0x6F | rd;
                    self.buf[reloc.offset..reloc.offset + 4]
                        .copy_from_slice(&patched.to_le_bytes());
                }
                RelocKind::Branch => {
                    if offset < -(1 << 12) || offset >= (1 << 12) {
                        return Err(RiscVEmitError::RelocationOutOfRange { bb: reloc.target });
                    }
                    let existing = u32::from_le_bytes([
                        self.buf[reloc.offset],
                        self.buf[reloc.offset + 1],
                        self.buf[reloc.offset + 2],
                        self.buf[reloc.offset + 3],
                    ]);
                    let opcode_funct3_rs = existing & 0x01FFF07F;
                    let patched = encode_b_type_imm(offset as i32) | opcode_funct3_rs;
                    self.buf[reloc.offset..reloc.offset + 4]
                        .copy_from_slice(&patched.to_le_bytes());
                }
            }
        }

        Ok(self.buf.clone())
    }

    fn calculate_size(&self, program: &IrProgram) -> usize {
        let mut count = 0;
        for block in &program.blocks {
            count += 1;
            count += block.nodes.len() * MAX_EXPANSION_RATIO;
        }
        count * 4
    }
}
