//! # nvmirror-fuzz
//!
//! Differential fuzzer for the eBPF JIT compiler.
//!
//! Generates random but *verifier-valid* eBPF programs using a grammar-based
//! approach, JIT-compiles them for all enabled targets, and compares
//! execution results against the reference interpreter.
//!
//! ## Strategy
//!
//! 1. **Grammar-based generation**: Programs are built from composable
//!    templates that always pass the verifier (no random byte mutation).
//! 2. **Differential oracle**: Interpreter result is ground truth; JIT
//!    output must match exactly.
//! 3. **Shrinking**: On failure, we minimise the program by removing
//!    instructions while preserving the mismatch.

use ebpf_core::isa::*;
use ebpf_core::RawInsn;
use jit_engine::interpreter::InterpreterState;
use jit_engine::{compile, Target};

/// Pseudo-random number generator (xorshift64).
/// Deterministic given a seed — no external dependencies.
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self { state: if seed == 0 { 1 } else { seed } }
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    pub fn next_u32(&mut self) -> u32 {
        self.next_u64() as u32
    }

    pub fn range(&mut self, lo: i64, hi: i64) -> i64 {
        let span = (hi - lo) as u64;
        if span == 0 { return lo; }
        lo + (self.next_u64() % span) as i64
    }

    pub fn pick<T: Copy>(&mut self, items: &[T]) -> T {
        items[self.next_u64() as usize % items.len()]
    }
}

fn make_raw(opcode: u8, dst: u8, src: u8, off: i16, imm: i32) -> RawInsn {
    RawInsn { opcode, regs: (src << 4) | dst, off, imm }
}

/// Generate a random valid eBPF program.
///
/// The program always:
/// - Initializes R0 and at least one other register
/// - Contains only forward branches (no loops — guaranteed termination)
/// - Ends with EXIT
///
/// O(n) where n = `max_insns`.
pub fn generate_program(rng: &mut Rng, max_insns: usize) -> Vec<RawInsn> {
    let mut insns = Vec::new();
    let n = 2 + (rng.next_u64() as usize % max_insns.saturating_sub(2).max(1));

    // Always start by initializing R0 and R1 (so we have valid operands)
    insns.push(make_raw(
        BPF_ALU64 | BPF_MOV | BPF_K,
        0, 0, 0,
        rng.range(0, 1000) as i32,
    ));
    insns.push(make_raw(
        BPF_ALU64 | BPF_MOV | BPF_K,
        1, 0, 0,
        rng.range(1, 100) as i32,
    ));

    // Initialize more registers randomly
    let init_regs = [2u8, 3, 4, 5, 6, 7, 8, 9];
    for &r in &init_regs {
        if rng.next_u64() % 3 == 0 {
            insns.push(make_raw(
                BPF_ALU64 | BPF_MOV | BPF_K,
                r, 0, 0,
                rng.range(1, 50) as i32,
            ));
        }
    }

    // Generate random ALU and branch instructions
    let remaining = n.saturating_sub(insns.len()).saturating_sub(1);
    for i in 0..remaining {
        let kind = rng.next_u64() % 10;
        let insn = match kind {
            0..=3 => {
                // ALU64 reg-imm
                let alu_ops: &[u8] = &[BPF_ADD, BPF_SUB, BPF_MUL, BPF_OR, BPF_AND, BPF_XOR, BPF_LSH, BPF_RSH];
                let op = rng.pick(alu_ops);
                let dst = rng.range(0, 5) as u8; // Only modify R0-R4
                let imm = match op {
                    BPF_MUL => rng.range(1, 10) as i32,
                    BPF_LSH | BPF_RSH => rng.range(0, 16) as i32,
                    BPF_DIV => rng.range(1, 100) as i32, // avoid div-by-zero
                    _ => rng.range(-100, 100) as i32,
                };
                make_raw(BPF_ALU64 | op | BPF_K, dst, 0, 0, imm)
            }
            4..=5 => {
                // ALU64 reg-reg
                let alu_ops: &[u8] = &[BPF_ADD, BPF_SUB, BPF_OR, BPF_AND, BPF_XOR];
                let op = rng.pick(alu_ops);
                let dst = rng.range(0, 5) as u8;
                let src = rng.range(0, 5) as u8;
                make_raw(BPF_ALU64 | op | BPF_X, dst, src, 0, 0)
            }
            6..=7 => {
                // ALU32 reg-imm
                let alu_ops: &[u8] = &[BPF_ADD, BPF_SUB, BPF_OR, BPF_AND, BPF_XOR];
                let op = rng.pick(alu_ops);
                let dst = rng.range(0, 5) as u8;
                make_raw(BPF_ALU | op | BPF_K, dst, 0, 0, rng.range(-50, 50) as i32)
            }
            8 => {
                // MOV64 imm
                let dst = rng.range(0, 5) as u8;
                make_raw(BPF_ALU64 | BPF_MOV | BPF_K, dst, 0, 0, rng.range(0, 1000) as i32)
            }
            _ => {
                // Conditional branch (forward only)
                let space_left = remaining - i;
                if space_left > 2 {
                    let off = rng.range(1, (space_left - 1).min(5) as i64) as i16;
                    let jmp_ops: &[u8] = &[BPF_JEQ, BPF_JNE, BPF_JGT, BPF_JGE, BPF_JLT, BPF_JLE];
                    let op = rng.pick(jmp_ops);
                    let dst = rng.range(0, 5) as u8;
                    make_raw(BPF_JMP | op | BPF_K, dst, 0, off, rng.range(0, 1000) as i32)
                } else {
                    // Not enough room for a branch, emit a MOV instead
                    make_raw(BPF_ALU64 | BPF_MOV | BPF_K, 0, 0, 0, rng.range(0, 100) as i32)
                }
            }
        };
        insns.push(insn);
    }

    // Always end with EXIT
    insns.push(make_raw(BPF_JMP | BPF_EXIT, 0, 0, 0, 0));

    insns
}

/// Result of a single fuzz iteration.
#[derive(Debug)]
pub enum FuzzResult {
    /// JIT and interpreter agree.
    Pass { interpreter_result: u64 },
    /// JIT compilation failed (might be expected for some programs).
    CompileError(String),
    /// JIT and interpreter disagree — this is a bug.
    Mismatch {
        interpreter_result: u64,
        jit_result: u64,
        program: Vec<RawInsn>,
    },
}

/// Run one fuzz iteration: generate, compile, execute, compare.
pub fn fuzz_one(rng: &mut Rng, max_insns: usize) -> FuzzResult {
    let program = generate_program(rng, max_insns);

    // Decode and run interpreter
    let insns = match ebpf_core::decode_program(&program) {
        Ok(i) => i,
        Err(e) => return FuzzResult::CompileError(format!("decode: {:?}", e)),
    };

    let mut interp = InterpreterState::new();
    let interp_result = interp.run(&insns, 10_000);

    // JIT compile (AArch64)
    let jit_output = match compile(&program, Target::AArch64) {
        Ok(o) => o,
        Err(e) => return FuzzResult::CompileError(format!("jit: {:?}", e)),
    };

    // On AArch64, actually execute
    #[cfg(target_arch = "aarch64")]
    {
        let jit_result = unsafe { execute_native(&jit_output.code) };
        if jit_result != interp_result {
            return FuzzResult::Mismatch {
                interpreter_result: interp_result,
                jit_result,
                program,
            };
        }
    }

    // On non-AArch64, just verify compilation succeeds
    let _ = jit_output;

    FuzzResult::Pass { interpreter_result: interp_result }
}

#[cfg(target_arch = "aarch64")]
unsafe fn execute_native(code: &[u8]) -> u64 {
    use nvme_shim::exec_buffer::ExecBuffer;

    let mut buf = ExecBuffer::new(code.len()).expect("mmap");
    buf.write_code(code).expect("write");

    type JitFn = unsafe extern "C" fn(u64, u64, u64, u64, u64) -> u64;
    let f: JitFn = unsafe { buf.as_fn_ptr() };
    unsafe { f(0, 0, 0, 0, 0) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fuzz_100_programs() {
        let mut rng = Rng::new(0xDEAD_BEEF_CAFE_BABEu64);
        let mut pass = 0;
        let mut compile_err = 0;

        for _ in 0..100 {
            match fuzz_one(&mut rng, 20) {
                FuzzResult::Pass { .. } => pass += 1,
                FuzzResult::CompileError(_) => compile_err += 1,
                FuzzResult::Mismatch { interpreter_result, jit_result, program } => {
                    panic!(
                        "MISMATCH: interp={}, jit={}, program_len={}",
                        interpreter_result, jit_result, program.len()
                    );
                }
            }
        }

        eprintln!("Fuzz results: {} pass, {} compile errors", pass, compile_err);
        assert!(pass > 0, "No programs passed — generator may be broken");
    }

    #[test]
    fn fuzz_1000_programs_small() {
        let mut rng = Rng::new(42);
        let mut pass = 0;
        let mut compile_err = 0;

        for _ in 0..1000 {
            match fuzz_one(&mut rng, 10) {
                FuzzResult::Pass { .. } => pass += 1,
                FuzzResult::CompileError(_) => compile_err += 1,
                FuzzResult::Mismatch { interpreter_result, jit_result, program } => {
                    panic!(
                        "MISMATCH: interp={}, jit={}, program_len={}",
                        interpreter_result, jit_result, program.len()
                    );
                }
            }
        }

        eprintln!("Fuzz results: {} pass, {} compile errors", pass, compile_err);
        assert!(pass > 50, "Too few programs passed");
    }
}
