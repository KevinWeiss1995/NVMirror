//! # jit-engine
//!
//! Orchestrator for the eBPF JIT pipeline:
//! `decode → verify → lower → allocate → emit → finalize`
//!
//! This crate ties together all phases and provides a single entry point
//! for compiling eBPF bytecode to target-specific machine code.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::vec::Vec;

use ebpf_core::{RawInsn, decode_program, verify_program, DecodeError, VerifyError};
use jit_ir::{CodeEmitter, IrProgram};
use jit_regalloc::{lower_to_ir, RegMap};

/// Target architecture selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Target {
    AArch64,
    RiscV64,
}

/// Unified JIT compilation error.
#[derive(Debug)]
pub enum JitError {
    Decode(DecodeError),
    Verify(VerifyError),
    #[cfg(feature = "aarch64")]
    EmitAArch64(jit_aarch64::AArch64EmitError),
    #[cfg(feature = "riscv")]
    EmitRiscV(jit_riscv::RiscVEmitError),
    TargetNotEnabled,
}

impl From<DecodeError> for JitError {
    fn from(e: DecodeError) -> Self { Self::Decode(e) }
}

impl From<VerifyError> for JitError {
    fn from(e: VerifyError) -> Self { Self::Verify(e) }
}

/// Result of a successful JIT compilation.
pub struct JitOutput {
    /// The emitted machine code bytes.
    pub code: Vec<u8>,
    /// The lowered IR (useful for debugging/disassembly).
    pub ir: IrProgram,
    /// Number of basic blocks.
    pub num_blocks: usize,
}

/// Compile eBPF raw bytecode to native machine code.
///
/// This is the main entry point for the JIT engine. It performs
/// the full pipeline: decode → verify → lower → emit → finalize.
///
/// ## Complexity
/// O(n) overall where n = instruction count. Each phase is linear
/// or near-linear. The emission phase is O(n * k) where k is the
/// maximum expansion ratio per instruction (bounded constant).
pub fn compile(raw_insns: &[RawInsn], target: Target) -> Result<JitOutput, JitError> {
    // Phase 1: Decode
    let insns = decode_program(raw_insns)?;

    // Phase 2: Verify
    let num_blocks = verify_program(&insns)?;

    // Phase 3: Lower (register allocation + IR generation)
    let regmap = regmap_for_target(target)?;
    let ir = lower_to_ir(&insns, &regmap);

    // Phase 4: Emit + Finalize
    let code = emit_for_target(&ir, target)?;

    Ok(JitOutput { code, ir, num_blocks })
}

/// Two-pass compilation: first compute size, then emit.
/// This is the firmware-friendly path that avoids reallocation.
pub fn compile_sized(raw_insns: &[RawInsn], target: Target) -> Result<JitOutput, JitError> {
    let insns = decode_program(raw_insns)?;
    let num_blocks = verify_program(&insns)?;
    let regmap = regmap_for_target(target)?;
    let ir = lower_to_ir(&insns, &regmap);

    // Dry-run: calculate exact buffer size
    let size = dry_run_size(&ir, target)?;

    // Emit with pre-allocated buffer
    let code = emit_for_target_with_size(&ir, target, size)?;

    Ok(JitOutput { code, ir, num_blocks })
}

fn regmap_for_target(target: Target) -> Result<RegMap, JitError> {
    match target {
        Target::AArch64 => Ok(jit_regalloc::aarch64_regmap()),
        Target::RiscV64 => Ok(jit_regalloc::riscv64_regmap()),
    }
}

fn dry_run_size(ir: &IrProgram, target: Target) -> Result<usize, JitError> {
    match target {
        #[cfg(feature = "aarch64")]
        Target::AArch64 => {
            let emitter = jit_aarch64::AArch64Emitter::new(0);
            Ok(emitter.calculate_size(ir))
        }
        #[cfg(feature = "riscv")]
        Target::RiscV64 => {
            let emitter = jit_riscv::RiscVEmitter::new(0);
            Ok(emitter.calculate_size(ir))
        }
        #[allow(unreachable_patterns)]
        _ => Err(JitError::TargetNotEnabled),
    }
}

fn emit_for_target(ir: &IrProgram, target: Target) -> Result<Vec<u8>, JitError> {
    match target {
        #[cfg(feature = "aarch64")]
        Target::AArch64 => {
            let mut emitter = jit_aarch64::AArch64Emitter::new(1024);
            emitter.emit_program(ir).map_err(JitError::EmitAArch64)?;
            emitter.finalize().map_err(JitError::EmitAArch64)
        }
        #[cfg(feature = "riscv")]
        Target::RiscV64 => {
            let mut emitter = jit_riscv::RiscVEmitter::new(1024);
            emitter.emit_program(ir).map_err(JitError::EmitRiscV)?;
            emitter.finalize().map_err(JitError::EmitRiscV)
        }
        #[allow(unreachable_patterns)]
        _ => Err(JitError::TargetNotEnabled),
    }
}

fn emit_for_target_with_size(
    ir: &IrProgram,
    target: Target,
    size: usize,
) -> Result<Vec<u8>, JitError> {
    match target {
        #[cfg(feature = "aarch64")]
        Target::AArch64 => {
            let mut emitter = jit_aarch64::AArch64Emitter::new(size);
            emitter.emit_program(ir).map_err(JitError::EmitAArch64)?;
            emitter.finalize().map_err(JitError::EmitAArch64)
        }
        #[cfg(feature = "riscv")]
        Target::RiscV64 => {
            let mut emitter = jit_riscv::RiscVEmitter::new(size);
            emitter.emit_program(ir).map_err(JitError::EmitRiscV)?;
            emitter.finalize().map_err(JitError::EmitRiscV)
        }
        #[allow(unreachable_patterns)]
        _ => Err(JitError::TargetNotEnabled),
    }
}

/// Reference interpreter for eBPF programs.
/// Used in differential testing to validate JIT output.
pub mod interpreter {
    use ebpf_core::{Insn, BpfReg, Source, AluOp, JmpOp, MemWidth, NUM_REGS, STACK_SIZE};

    pub struct InterpreterState {
        pub regs: [u64; NUM_REGS],
        pub stack: [u8; STACK_SIZE],
        pub pc: usize,
        pub halted: bool,
    }

    impl InterpreterState {
        pub fn new() -> Self {
            Self {
                regs: [0; NUM_REGS],
                stack: [0; STACK_SIZE],
                pc: 0,
                halted: false,
            }
        }

        /// Execute one instruction. Returns true if the program should continue.
        /// O(1) per instruction.
        pub fn step(&mut self, insns: &[Insn]) -> bool {
            if self.halted || self.pc >= insns.len() {
                return false;
            }

            let insn = &insns[self.pc];
            match insn {
                Insn::Alu64 { op, dst, src } => {
                    let s = self.resolve_source(src);
                    let d = self.regs[dst.index()];
                    self.regs[dst.index()] = eval_alu64(op, d, s);
                    self.pc += 1;
                }
                Insn::Alu32 { op, dst, src } => {
                    let s = self.resolve_source(src) as u32;
                    let d = self.regs[dst.index()] as u32;
                    self.regs[dst.index()] = eval_alu32(op, d, s) as u64;
                    self.pc += 1;
                }
                Insn::LoadImm64 { dst, imm } => {
                    self.regs[dst.index()] = *imm;
                    self.pc += 1;
                }
                Insn::Load { width, dst, src, off } => {
                    let addr = (self.regs[src.index()] as i64 + *off as i64) as usize;
                    let stack_base = self.stack.as_ptr() as usize + STACK_SIZE;
                    let rel = addr.wrapping_sub(stack_base.wrapping_sub(STACK_SIZE));
                    let val = read_stack(&self.stack, rel, *width);
                    self.regs[dst.index()] = val;
                    self.pc += 1;
                }
                Insn::StoreReg { width, dst, src, off } => {
                    let addr = (self.regs[dst.index()] as i64 + *off as i64) as usize;
                    let stack_base = self.stack.as_ptr() as usize + STACK_SIZE;
                    let rel = addr.wrapping_sub(stack_base.wrapping_sub(STACK_SIZE));
                    let val = self.regs[src.index()];
                    write_stack(&mut self.stack, rel, *width, val);
                    self.pc += 1;
                }
                Insn::StoreImm { width, dst, off, imm } => {
                    let addr = (self.regs[dst.index()] as i64 + *off as i64) as usize;
                    let stack_base = self.stack.as_ptr() as usize + STACK_SIZE;
                    let rel = addr.wrapping_sub(stack_base.wrapping_sub(STACK_SIZE));
                    write_stack(&mut self.stack, rel, *width, *imm as u64);
                    self.pc += 1;
                }
                Insn::Ja { off } => {
                    self.pc = (self.pc as isize + 1 + *off as isize) as usize;
                }
                Insn::JmpCond { op, dst, src, off } => {
                    let a = self.regs[dst.index()];
                    let b = self.resolve_source(src);
                    if eval_jmp(op, a, b) {
                        self.pc = (self.pc as isize + 1 + *off as isize) as usize;
                    } else {
                        self.pc += 1;
                    }
                }
                Insn::JmpCond32 { op, dst, src, off } => {
                    let a = self.regs[dst.index()] as u32 as u64;
                    let b = self.resolve_source(src) as u32 as u64;
                    if eval_jmp(op, a, b) {
                        self.pc = (self.pc as isize + 1 + *off as isize) as usize;
                    } else {
                        self.pc += 1;
                    }
                }
                Insn::Call { func_id: _ } => {
                    // Stub: for testing, calls are no-ops that set R0 = 0
                    self.regs[BpfReg::R0.index()] = 0;
                    self.pc += 1;
                }
                Insn::Exit => {
                    self.halted = true;
                    return false;
                }
                Insn::TailCall => {
                    self.halted = true;
                    return false;
                }
                Insn::Atomic { .. } => {
                    // Simplified: treat as no-op for interpreter
                    self.pc += 1;
                }
                Insn::Endian { to_be, width, dst } => {
                    let val = self.regs[dst.index()];
                    self.regs[dst.index()] = eval_endian(*to_be, width, val);
                    self.pc += 1;
                }
            }
            true
        }

        /// Run until exit or instruction limit.
        pub fn run(&mut self, insns: &[Insn], max_steps: usize) -> u64 {
            for _ in 0..max_steps {
                if !self.step(insns) {
                    break;
                }
            }
            self.regs[BpfReg::R0.index()]
        }

        fn resolve_source(&self, src: &Source) -> u64 {
            match src {
                Source::Imm(v) => *v as u64,
                Source::Reg(r) => self.regs[r.index()],
            }
        }
    }

    fn eval_alu64(op: &AluOp, dst: u64, src: u64) -> u64 {
        match op {
            AluOp::Add => dst.wrapping_add(src),
            AluOp::Sub => dst.wrapping_sub(src),
            AluOp::Mul => dst.wrapping_mul(src),
            AluOp::Div => if src == 0 { 0 } else { dst / src },
            AluOp::Mod => if src == 0 { dst } else { dst % src },
            AluOp::Or => dst | src,
            AluOp::And => dst & src,
            AluOp::Xor => dst ^ src,
            AluOp::Lsh => dst << (src & 63),
            AluOp::Rsh => dst >> (src & 63),
            AluOp::Arsh => ((dst as i64) >> (src & 63)) as u64,
            AluOp::Mov => src,
            AluOp::Neg => (-(dst as i64)) as u64,
        }
    }

    fn eval_alu32(op: &AluOp, dst: u32, src: u32) -> u32 {
        match op {
            AluOp::Add => dst.wrapping_add(src),
            AluOp::Sub => dst.wrapping_sub(src),
            AluOp::Mul => dst.wrapping_mul(src),
            AluOp::Div => if src == 0 { 0 } else { dst / src },
            AluOp::Mod => if src == 0 { dst } else { dst % src },
            AluOp::Or => dst | src,
            AluOp::And => dst & src,
            AluOp::Xor => dst ^ src,
            AluOp::Lsh => dst << (src & 31),
            AluOp::Rsh => dst >> (src & 31),
            AluOp::Arsh => ((dst as i32) >> (src & 31)) as u32,
            AluOp::Mov => src,
            AluOp::Neg => (-(dst as i32)) as u32,
        }
    }

    fn eval_jmp(op: &JmpOp, a: u64, b: u64) -> bool {
        match op {
            JmpOp::Jeq => a == b,
            JmpOp::Jne => a != b,
            JmpOp::Jgt => a > b,
            JmpOp::Jge => a >= b,
            JmpOp::Jlt => a < b,
            JmpOp::Jle => a <= b,
            JmpOp::Jsgt => (a as i64) > (b as i64),
            JmpOp::Jsge => (a as i64) >= (b as i64),
            JmpOp::Jslt => (a as i64) < (b as i64),
            JmpOp::Jsle => (a as i64) <= (b as i64),
            JmpOp::Jset => (a & b) != 0,
            JmpOp::Ja => true,
        }
    }

    fn eval_endian(
        to_be: bool,
        width: &ebpf_core::EndianWidth,
        val: u64,
    ) -> u64 {
        use ebpf_core::EndianWidth;
        if to_be {
            match width {
                EndianWidth::Bits16 => (val as u16).swap_bytes() as u64,
                EndianWidth::Bits32 => (val as u32).swap_bytes() as u64,
                EndianWidth::Bits64 => val.swap_bytes(),
            }
        } else {
            match width {
                EndianWidth::Bits16 => (val as u16).to_le() as u64,
                EndianWidth::Bits32 => (val as u32).to_le() as u64,
                EndianWidth::Bits64 => val.to_le(),
            }
        }
    }

    fn read_stack(stack: &[u8; STACK_SIZE], offset: usize, width: MemWidth) -> u64 {
        if offset + width.byte_len() as usize > STACK_SIZE {
            return 0;
        }
        let s = &stack[offset..];
        match width {
            MemWidth::B => s[0] as u64,
            MemWidth::H => u16::from_le_bytes([s[0], s[1]]) as u64,
            MemWidth::W => u32::from_le_bytes([s[0], s[1], s[2], s[3]]) as u64,
            MemWidth::DW => u64::from_le_bytes([s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]]),
        }
    }

    fn write_stack(stack: &mut [u8; STACK_SIZE], offset: usize, width: MemWidth, val: u64) {
        if offset + width.byte_len() as usize > STACK_SIZE {
            return;
        }
        let s = &mut stack[offset..];
        match width {
            MemWidth::B => s[0] = val as u8,
            MemWidth::H => s[..2].copy_from_slice(&(val as u16).to_le_bytes()),
            MemWidth::W => s[..4].copy_from_slice(&(val as u32).to_le_bytes()),
            MemWidth::DW => s[..8].copy_from_slice(&val.to_le_bytes()),
        }
    }

    impl Default for InterpreterState {
        fn default() -> Self {
            Self::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ebpf_core::isa::*;

    fn make_raw(opcode: u8, dst: u8, src: u8, off: i16, imm: i32) -> RawInsn {
        RawInsn { opcode, regs: (src << 4) | dst, off, imm }
    }

    #[test]
    fn compile_minimal_aarch64() {
        // MOV64 R0, 42; EXIT
        let raw = [
            make_raw(BPF_ALU64 | BPF_MOV | BPF_K, 0, 0, 0, 42),
            make_raw(BPF_JMP | BPF_EXIT, 0, 0, 0, 0),
        ];

        let output = compile(&raw, Target::AArch64).unwrap();
        assert!(!output.code.is_empty());
        assert_eq!(output.code.len() % 4, 0); // AArch64 instructions are 4-byte aligned
    }

    #[test]
    fn compile_minimal_riscv() {
        let raw = [
            make_raw(BPF_ALU64 | BPF_MOV | BPF_K, 0, 0, 0, 42),
            make_raw(BPF_JMP | BPF_EXIT, 0, 0, 0, 0),
        ];

        let output = compile(&raw, Target::RiscV64).unwrap();
        assert!(!output.code.is_empty());
        assert_eq!(output.code.len() % 4, 0);
    }

    #[test]
    fn compile_with_branch() {
        // R0 = 0; if R1 == 0 goto +1; R0 = 1; EXIT
        let raw = [
            make_raw(BPF_ALU64 | BPF_MOV | BPF_K, 0, 0, 0, 0),
            make_raw(BPF_JMP | BPF_JEQ | BPF_K, 1, 0, 1, 0),
            make_raw(BPF_ALU64 | BPF_MOV | BPF_K, 0, 0, 0, 1),
            make_raw(BPF_JMP | BPF_EXIT, 0, 0, 0, 0),
        ];

        let output = compile(&raw, Target::AArch64).unwrap();
        assert!(!output.code.is_empty());
    }

    #[test]
    fn interpreter_basic() {
        use interpreter::InterpreterState;
        use ebpf_core::{Insn, AluOp, Source, BpfReg};

        let insns = vec![
            Insn::Alu64 { op: AluOp::Mov, dst: BpfReg::R0, src: Source::Imm(42) },
            Insn::Exit,
        ];

        let mut state = InterpreterState::new();
        let result = state.run(&insns, 100);
        assert_eq!(result, 42);
    }

    #[test]
    fn interpreter_branch() {
        use interpreter::InterpreterState;
        use ebpf_core::{Insn, AluOp, JmpOp, Source, BpfReg};

        let insns = vec![
            Insn::Alu64 { op: AluOp::Mov, dst: BpfReg::R0, src: Source::Imm(0) },
            Insn::Alu64 { op: AluOp::Mov, dst: BpfReg::R2, src: Source::Imm(5) },
            Insn::JmpCond {
                op: JmpOp::Jeq,
                dst: BpfReg::R2,
                src: Source::Imm(5),
                off: 1,
            },
            Insn::Alu64 { op: AluOp::Mov, dst: BpfReg::R0, src: Source::Imm(99) },
            Insn::Alu64 { op: AluOp::Mov, dst: BpfReg::R0, src: Source::Imm(42) },
            Insn::Exit,
        ];

        let mut state = InterpreterState::new();
        let result = state.run(&insns, 100);
        assert_eq!(result, 42); // Should skip R0=99 and set R0=42
    }

    #[test]
    fn two_pass_compilation() {
        let raw = [
            make_raw(BPF_ALU64 | BPF_MOV | BPF_K, 0, 0, 0, 42),
            make_raw(BPF_JMP | BPF_EXIT, 0, 0, 0, 0),
        ];

        let output = compile_sized(&raw, Target::AArch64).unwrap();
        assert!(!output.code.is_empty());
    }
}
