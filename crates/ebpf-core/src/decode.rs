//! eBPF bytecode decoder.
//!
//! Translates a slice of `RawInsn` into a sequence of typed `Insn` values.
//! This pass performs *syntactic* validation only — semantic verification
//! (reachability, register liveness, bounds) is handled by [`crate::verify`].
//!
//! ## Complexity
//! O(n) in the number of raw instructions — single linear scan.

use alloc::vec::Vec;
use crate::isa::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeError {
    UnknownOpcode { pc: usize, opcode: u8 },
    InvalidRegister { pc: usize, raw: u8 },
    TruncatedWideInsn { pc: usize },
    ProgramTooLarge { count: usize },
    EmptyProgram,
    InvalidAtomicOp { pc: usize, imm: u32 },
    InvalidEndianWidth { pc: usize, imm: i32 },
}

/// Decode a slice of raw eBPF instructions into typed instructions.
///
/// Wide instructions (LDDW) consume two slots; the returned Vec may
/// be shorter than the input slice.
///
/// O(n) — single pass, no allocations beyond the output vector.
pub fn decode_program(raw: &[RawInsn]) -> Result<Vec<Insn>, DecodeError> {
    if raw.is_empty() {
        return Err(DecodeError::EmptyProgram);
    }
    if raw.len() > MAX_INSNS {
        return Err(DecodeError::ProgramTooLarge { count: raw.len() });
    }

    let mut insns = Vec::with_capacity(raw.len());
    let mut pc = 0usize;

    while pc < raw.len() {
        let r = &raw[pc];
        let class = r.insn_class();

        let insn = match class {
            BPF_ALU64 => decode_alu(r, pc, true)?,
            BPF_ALU => decode_alu(r, pc, false)?,
            BPF_JMP => decode_jmp(r, pc)?,
            BPF_JMP32 => decode_jmp32(r, pc)?,
            BPF_LDX => decode_ldx(r, pc)?,
            BPF_STX => decode_stx(r, pc)?,
            BPF_ST => decode_st(r, pc)?,
            BPF_LD => {
                let mode = r.opcode & 0xe0;
                if mode == BPF_IMM && (r.opcode & 0x18) == BPF_DW {
                    if pc + 1 >= raw.len() {
                        return Err(DecodeError::TruncatedWideInsn { pc });
                    }
                    let next = &raw[pc + 1];
                    let imm_lo = r.imm as u32 as u64;
                    let imm_hi = (next.imm as u32 as u64) << 32;
                    let dst = reg(r.dst_reg(), pc)?;
                    pc += 2;
                    insns.push(Insn::LoadImm64 { dst, imm: imm_lo | imm_hi });
                    continue;
                }
                return Err(DecodeError::UnknownOpcode { pc, opcode: r.opcode });
            }
            _ => return Err(DecodeError::UnknownOpcode { pc, opcode: r.opcode }),
        };

        insns.push(insn);
        pc += 1;
    }

    Ok(insns)
}

fn reg(raw: u8, pc: usize) -> Result<BpfReg, DecodeError> {
    BpfReg::from_raw(raw).ok_or(DecodeError::InvalidRegister { pc, raw })
}

fn source(r: &RawInsn, pc: usize) -> Result<Source, DecodeError> {
    if r.opcode & BPF_X != 0 {
        Ok(Source::Reg(reg(r.src_reg(), pc)?))
    } else {
        Ok(Source::Imm(r.imm as i64))
    }
}

fn decode_alu(r: &RawInsn, pc: usize, is_64: bool) -> Result<Insn, DecodeError> {
    let op_bits = r.opcode & 0xf0;
    let dst = reg(r.dst_reg(), pc)?;

    if op_bits == BPF_END {
        let to_be = (r.opcode & BPF_X) != 0;
        let width = match r.imm {
            16 => EndianWidth::Bits16,
            32 => EndianWidth::Bits32,
            64 => EndianWidth::Bits64,
            _ => return Err(DecodeError::InvalidEndianWidth { pc, imm: r.imm }),
        };
        return Ok(Insn::Endian { to_be, width, dst });
    }

    let op = AluOp::from_opcode_bits(r.opcode)
        .ok_or(DecodeError::UnknownOpcode { pc, opcode: r.opcode })?;
    let src = source(r, pc)?;

    if is_64 {
        Ok(Insn::Alu64 { op, dst, src })
    } else {
        Ok(Insn::Alu32 { op, dst, src })
    }
}

fn decode_jmp(r: &RawInsn, pc: usize) -> Result<Insn, DecodeError> {
    let op_bits = r.opcode & 0xf0;

    match op_bits {
        BPF_JA => Ok(Insn::Ja { off: r.off }),
        BPF_CALL => Ok(Insn::Call { func_id: r.imm as u32 }),
        BPF_EXIT => Ok(Insn::Exit),
        _ => {
            let op = JmpOp::from_opcode_bits(r.opcode)
                .ok_or(DecodeError::UnknownOpcode { pc, opcode: r.opcode })?;
            let dst = reg(r.dst_reg(), pc)?;
            let src = source(r, pc)?;
            Ok(Insn::JmpCond { op, dst, src, off: r.off })
        }
    }
}

fn decode_jmp32(r: &RawInsn, pc: usize) -> Result<Insn, DecodeError> {
    let op = JmpOp::from_opcode_bits(r.opcode)
        .ok_or(DecodeError::UnknownOpcode { pc, opcode: r.opcode })?;
    let dst = reg(r.dst_reg(), pc)?;
    let src = source(r, pc)?;
    Ok(Insn::JmpCond32 { op, dst, src, off: r.off })
}

fn decode_ldx(r: &RawInsn, pc: usize) -> Result<Insn, DecodeError> {
    let width = MemWidth::from_size_code(r.opcode)
        .ok_or(DecodeError::UnknownOpcode { pc, opcode: r.opcode })?;
    let dst = reg(r.dst_reg(), pc)?;
    let src = reg(r.src_reg(), pc)?;
    Ok(Insn::Load { width, dst, src, off: r.off })
}

fn decode_stx(r: &RawInsn, pc: usize) -> Result<Insn, DecodeError> {
    let mode = r.opcode & 0xe0;
    let width = MemWidth::from_size_code(r.opcode)
        .ok_or(DecodeError::UnknownOpcode { pc, opcode: r.opcode })?;
    let dst = reg(r.dst_reg(), pc)?;
    let src = reg(r.src_reg(), pc)?;

    if mode == BPF_ATOMIC {
        let atomic_op = decode_atomic_op(r.imm as u32, pc)?;
        Ok(Insn::Atomic { width, op: atomic_op, dst, src, off: r.off })
    } else {
        Ok(Insn::StoreReg { width, dst, src, off: r.off })
    }
}

fn decode_st(r: &RawInsn, pc: usize) -> Result<Insn, DecodeError> {
    let width = MemWidth::from_size_code(r.opcode)
        .ok_or(DecodeError::UnknownOpcode { pc, opcode: r.opcode })?;
    let dst = reg(r.dst_reg(), pc)?;
    Ok(Insn::StoreImm { width, dst, off: r.off, imm: r.imm })
}

fn decode_atomic_op(imm: u32, pc: usize) -> Result<AtomicOp, DecodeError> {
    let fetch = imm & (BPF_ATOMIC_FETCH);
    let base = imm & !BPF_ATOMIC_FETCH;

    match (base, fetch != 0) {
        (BPF_ATOMIC_ADD, false) => Ok(AtomicOp::Add),
        (BPF_ATOMIC_ADD, true) => Ok(AtomicOp::FetchAdd),
        (BPF_ATOMIC_OR, false) => Ok(AtomicOp::Or),
        (BPF_ATOMIC_OR, true) => Ok(AtomicOp::FetchOr),
        (BPF_ATOMIC_AND, false) => Ok(AtomicOp::And),
        (BPF_ATOMIC_AND, true) => Ok(AtomicOp::FetchAnd),
        (BPF_ATOMIC_XOR, false) => Ok(AtomicOp::Xor),
        (BPF_ATOMIC_XOR, true) => Ok(AtomicOp::FetchXor),
        _ if imm == BPF_ATOMIC_XCHG => Ok(AtomicOp::Xchg),
        _ if imm == BPF_ATOMIC_CMPXCHG => Ok(AtomicOp::CmpXchg),
        _ => Err(DecodeError::InvalidAtomicOp { pc, imm }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_raw(opcode: u8, dst: u8, src: u8, off: i16, imm: i32) -> RawInsn {
        RawInsn { opcode, regs: (src << 4) | dst, off, imm }
    }

    #[test]
    fn decode_mov64_imm() {
        // MOV64 R1, 42
        let raw = [make_raw(BPF_ALU64 | BPF_MOV | BPF_K, 1, 0, 0, 42)];
        let insns = decode_program(&raw).unwrap();
        assert_eq!(insns.len(), 1);
        assert_eq!(
            insns[0],
            Insn::Alu64 { op: AluOp::Mov, dst: BpfReg::R1, src: Source::Imm(42) }
        );
    }

    #[test]
    fn decode_add64_reg() {
        // ADD64 R0, R1
        let raw = [make_raw(BPF_ALU64 | BPF_ADD | BPF_X, 0, 1, 0, 0)];
        let insns = decode_program(&raw).unwrap();
        assert_eq!(
            insns[0],
            Insn::Alu64 { op: AluOp::Add, dst: BpfReg::R0, src: Source::Reg(BpfReg::R1) }
        );
    }

    #[test]
    fn decode_exit() {
        let raw = [make_raw(BPF_JMP | BPF_EXIT, 0, 0, 0, 0)];
        let insns = decode_program(&raw).unwrap();
        assert_eq!(insns[0], Insn::Exit);
    }

    #[test]
    fn decode_lddw() {
        // LDDW R3, 0x0000_0002_0000_0001
        let raw = [
            make_raw(BPF_LD | BPF_IMM | BPF_DW, 3, 0, 0, 1),
            make_raw(0, 0, 0, 0, 2),
        ];
        let insns = decode_program(&raw).unwrap();
        assert_eq!(insns.len(), 1);
        assert_eq!(
            insns[0],
            Insn::LoadImm64 { dst: BpfReg::R3, imm: 0x0000_0002_0000_0001 }
        );
    }

    #[test]
    fn decode_jeq_imm() {
        // JEQ R2, 10, +3
        let raw = [make_raw(BPF_JMP | BPF_JEQ | BPF_K, 2, 0, 3, 10)];
        let insns = decode_program(&raw).unwrap();
        assert_eq!(
            insns[0],
            Insn::JmpCond {
                op: JmpOp::Jeq,
                dst: BpfReg::R2,
                src: Source::Imm(10),
                off: 3,
            }
        );
    }

    #[test]
    fn decode_store_load() {
        // STX [R10-8], R1 (64-bit)
        let raw_stx = [make_raw(BPF_STX | BPF_MEM | BPF_DW, 10, 1, -8, 0)];
        let insns = decode_program(&raw_stx).unwrap();
        assert_eq!(
            insns[0],
            Insn::StoreReg {
                width: MemWidth::DW,
                dst: BpfReg::R10,
                src: BpfReg::R1,
                off: -8,
            }
        );

        // LDX R2, [R10-8] (64-bit)
        let raw_ldx = [make_raw(BPF_LDX | BPF_MEM | BPF_DW, 2, 10, -8, 0)];
        let insns = decode_program(&raw_ldx).unwrap();
        assert_eq!(
            insns[0],
            Insn::Load {
                width: MemWidth::DW,
                dst: BpfReg::R2,
                src: BpfReg::R10,
                off: -8,
            }
        );
    }

    #[test]
    fn reject_empty() {
        assert_eq!(decode_program(&[]), Err(DecodeError::EmptyProgram));
    }

    #[test]
    fn reject_invalid_register() {
        let raw = [make_raw(BPF_ALU64 | BPF_MOV | BPF_K, 15, 0, 0, 0)];
        assert!(matches!(
            decode_program(&raw),
            Err(DecodeError::InvalidRegister { pc: 0, raw: 15 })
        ));
    }
}
