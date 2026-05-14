//! eBPF conformance test suite.
//!
//! Test vectors derived from the Linux kernel's BPF test suite
//! (tools/testing/selftests/bpf/test_verifier.c and
//!  lib/test_bpf.c). Each test specifies:
//!
//! - Raw eBPF bytecode
//! - Expected return value
//! - Optional initial register state (R1 context pointer)
//!
//! Tests validate both the interpreter and JIT backends.

use ebpf_core::isa::*;
use ebpf_core::{RawInsn, decode_program};
use jit_engine::interpreter::InterpreterState;
use jit_engine::{compile, Target};

fn make_raw(opcode: u8, dst: u8, src: u8, off: i16, imm: i32) -> RawInsn {
    RawInsn { opcode, regs: (src << 4) | dst, off, imm }
}

/// Run a program through the interpreter and return R0.
fn interp_run(raw: &[RawInsn]) -> u64 {
    let insns = decode_program(raw).unwrap();
    let mut state = InterpreterState::new();
    state.run(&insns, 10_000)
}

/// Compile for AArch64 and verify it produces valid code.
fn jit_compile_aarch64(raw: &[RawInsn]) -> Vec<u8> {
    let output = compile(raw, Target::AArch64).unwrap();
    assert_eq!(output.code.len() % 4, 0);
    output.code
}

/// Execute JIT'd code natively on AArch64.
#[cfg(target_arch = "aarch64")]
unsafe fn exec_native(code: &[u8]) -> u64 {
    use nvme_shim::exec_buffer::ExecBuffer;

    let mut buf = ExecBuffer::new(code.len()).expect("mmap");
    buf.write_code(code).expect("write");

    type JitFn = unsafe extern "C" fn(u64, u64, u64, u64, u64) -> u64;
    let f: JitFn = unsafe { buf.as_fn_ptr() };
    unsafe { f(0, 0, 0, 0, 0) }
}

/// Verify interpreter AND JIT both produce `expected`.
fn check(raw: &[RawInsn], expected: u64) {
    let interp = interp_run(raw);
    assert_eq!(interp, expected, "Interpreter mismatch");

    let code = jit_compile_aarch64(raw);
    #[cfg(target_arch = "aarch64")]
    {
        let jit = unsafe { exec_native(&code) };
        assert_eq!(jit, expected, "JIT mismatch (AArch64 native)");
    }
    let _ = code;

    // Also verify RISC-V compilation succeeds
    let rv_output = compile(raw, Target::RiscV64).unwrap();
    assert_eq!(rv_output.code.len() % 4, 0);
}

// =========================================================================
// ALU64 — Immediate operations
// =========================================================================

#[test]
fn alu64_mov_imm() {
    check(&[
        make_raw(BPF_ALU64 | BPF_MOV | BPF_K, 0, 0, 0, 1),
        make_raw(BPF_JMP | BPF_EXIT, 0, 0, 0, 0),
    ], 1);
}

#[test]
fn alu64_mov_large_imm() {
    check(&[
        make_raw(BPF_ALU64 | BPF_MOV | BPF_K, 0, 0, 0, 0x7FFFFFFF),
        make_raw(BPF_JMP | BPF_EXIT, 0, 0, 0, 0),
    ], 0x7FFFFFFF);
}

#[test]
fn alu64_mov_negative_imm() {
    // MOV64 R0, -1  → R0 should be 0xFFFFFFFFFFFFFFFF (sign-extended)
    check(&[
        make_raw(BPF_ALU64 | BPF_MOV | BPF_K, 0, 0, 0, -1),
        make_raw(BPF_JMP | BPF_EXIT, 0, 0, 0, 0),
    ], u64::MAX);
}

#[test]
fn alu64_add_imm() {
    check(&[
        make_raw(BPF_ALU64 | BPF_MOV | BPF_K, 0, 0, 0, 10),
        make_raw(BPF_ALU64 | BPF_ADD | BPF_K, 0, 0, 0, 20),
        make_raw(BPF_JMP | BPF_EXIT, 0, 0, 0, 0),
    ], 30);
}

#[test]
fn alu64_sub_imm() {
    check(&[
        make_raw(BPF_ALU64 | BPF_MOV | BPF_K, 0, 0, 0, 100),
        make_raw(BPF_ALU64 | BPF_SUB | BPF_K, 0, 0, 0, 37),
        make_raw(BPF_JMP | BPF_EXIT, 0, 0, 0, 0),
    ], 63);
}

#[test]
fn alu64_mul_imm() {
    check(&[
        make_raw(BPF_ALU64 | BPF_MOV | BPF_K, 0, 0, 0, 7),
        make_raw(BPF_ALU64 | BPF_MUL | BPF_K, 0, 0, 0, 6),
        make_raw(BPF_JMP | BPF_EXIT, 0, 0, 0, 0),
    ], 42);
}

#[test]
fn alu64_div_imm() {
    check(&[
        make_raw(BPF_ALU64 | BPF_MOV | BPF_K, 0, 0, 0, 42),
        make_raw(BPF_ALU64 | BPF_DIV | BPF_K, 0, 0, 0, 7),
        make_raw(BPF_JMP | BPF_EXIT, 0, 0, 0, 0),
    ], 6);
}

#[test]
fn alu64_mod_imm() {
    check(&[
        make_raw(BPF_ALU64 | BPF_MOV | BPF_K, 0, 0, 0, 17),
        make_raw(BPF_ALU64 | BPF_MOD | BPF_K, 0, 0, 0, 5),
        make_raw(BPF_JMP | BPF_EXIT, 0, 0, 0, 0),
    ], 2);
}

#[test]
fn alu64_or_imm() {
    check(&[
        make_raw(BPF_ALU64 | BPF_MOV | BPF_K, 0, 0, 0, 0x0F),
        make_raw(BPF_ALU64 | BPF_OR | BPF_K, 0, 0, 0, 0xF0u8 as i32),
        make_raw(BPF_JMP | BPF_EXIT, 0, 0, 0, 0),
    ], 0xFF);
}

#[test]
fn alu64_and_imm() {
    check(&[
        make_raw(BPF_ALU64 | BPF_MOV | BPF_K, 0, 0, 0, 0xFF),
        make_raw(BPF_ALU64 | BPF_AND | BPF_K, 0, 0, 0, 0x0F),
        make_raw(BPF_JMP | BPF_EXIT, 0, 0, 0, 0),
    ], 0x0F);
}

#[test]
fn alu64_xor_imm() {
    check(&[
        make_raw(BPF_ALU64 | BPF_MOV | BPF_K, 0, 0, 0, 0xFF),
        make_raw(BPF_ALU64 | BPF_XOR | BPF_K, 0, 0, 0, 0x0F),
        make_raw(BPF_JMP | BPF_EXIT, 0, 0, 0, 0),
    ], 0xF0);
}

#[test]
fn alu64_lsh_imm() {
    check(&[
        make_raw(BPF_ALU64 | BPF_MOV | BPF_K, 0, 0, 0, 1),
        make_raw(BPF_ALU64 | BPF_LSH | BPF_K, 0, 0, 0, 8),
        make_raw(BPF_JMP | BPF_EXIT, 0, 0, 0, 0),
    ], 256);
}

#[test]
fn alu64_rsh_imm() {
    check(&[
        make_raw(BPF_ALU64 | BPF_MOV | BPF_K, 0, 0, 0, 256),
        make_raw(BPF_ALU64 | BPF_RSH | BPF_K, 0, 0, 0, 4),
        make_raw(BPF_JMP | BPF_EXIT, 0, 0, 0, 0),
    ], 16);
}

#[test]
fn alu64_arsh_imm() {
    // ARSH of -16 >> 2 → -4 → 0xFFFFFFFFFFFFFFFC
    check(&[
        make_raw(BPF_ALU64 | BPF_MOV | BPF_K, 0, 0, 0, -16),
        make_raw(BPF_ALU64 | BPF_ARSH | BPF_K, 0, 0, 0, 2),
        make_raw(BPF_JMP | BPF_EXIT, 0, 0, 0, 0),
    ], (-4i64) as u64);
}

// =========================================================================
// ALU64 — Register operations
// =========================================================================

#[test]
fn alu64_add_reg() {
    check(&[
        make_raw(BPF_ALU64 | BPF_MOV | BPF_K, 0, 0, 0, 10),
        make_raw(BPF_ALU64 | BPF_MOV | BPF_K, 1, 0, 0, 20),
        make_raw(BPF_ALU64 | BPF_ADD | BPF_X, 0, 1, 0, 0),
        make_raw(BPF_JMP | BPF_EXIT, 0, 0, 0, 0),
    ], 30);
}

#[test]
fn alu64_mov_reg() {
    check(&[
        make_raw(BPF_ALU64 | BPF_MOV | BPF_K, 0, 0, 0, 0),
        make_raw(BPF_ALU64 | BPF_MOV | BPF_K, 2, 0, 0, 99),
        make_raw(BPF_ALU64 | BPF_MOV | BPF_X, 0, 2, 0, 0),
        make_raw(BPF_JMP | BPF_EXIT, 0, 0, 0, 0),
    ], 99);
}

// =========================================================================
// ALU32 operations (result zero-extended to 64-bit)
// =========================================================================

#[test]
fn alu32_add_imm() {
    check(&[
        make_raw(BPF_ALU64 | BPF_MOV | BPF_K, 0, 0, 0, 10),
        make_raw(BPF_ALU | BPF_ADD | BPF_K, 0, 0, 0, 20),
        make_raw(BPF_JMP | BPF_EXIT, 0, 0, 0, 0),
    ], 30);
}

// =========================================================================
// LDDW — 64-bit immediate load
// =========================================================================

#[test]
fn lddw_basic() {
    check(&[
        make_raw(BPF_LD | BPF_IMM | BPF_DW, 0, 0, 0, 0x1234),
        make_raw(0, 0, 0, 0, 0x5678),
        make_raw(BPF_JMP | BPF_EXIT, 0, 0, 0, 0),
    ], 0x0000_5678_0000_1234);
}

#[test]
fn lddw_full_64bit() {
    check(&[
        make_raw(BPF_LD | BPF_IMM | BPF_DW, 0, 0, 0, -1),  // lo = 0xFFFFFFFF
        make_raw(0, 0, 0, 0, -1),                             // hi = 0xFFFFFFFF
        make_raw(BPF_JMP | BPF_EXIT, 0, 0, 0, 0),
    ], u64::MAX);
}

// =========================================================================
// Jumps — conditional branches
// =========================================================================

#[test]
fn jmp_jeq_taken() {
    // R0=0; R2=5; if R2==5 goto +1; R0=99; R0=42; EXIT → 42
    check(&[
        make_raw(BPF_ALU64 | BPF_MOV | BPF_K, 0, 0, 0, 0),
        make_raw(BPF_ALU64 | BPF_MOV | BPF_K, 2, 0, 0, 5),
        make_raw(BPF_JMP | BPF_JEQ | BPF_K, 2, 0, 1, 5),
        make_raw(BPF_ALU64 | BPF_MOV | BPF_K, 0, 0, 0, 99),
        make_raw(BPF_ALU64 | BPF_MOV | BPF_K, 0, 0, 0, 42),
        make_raw(BPF_JMP | BPF_EXIT, 0, 0, 0, 0),
    ], 42);
}

#[test]
fn jmp_jeq_not_taken() {
    // R0=0; R2=5; if R2==10 goto +1; R0=77; EXIT → 77
    check(&[
        make_raw(BPF_ALU64 | BPF_MOV | BPF_K, 0, 0, 0, 0),
        make_raw(BPF_ALU64 | BPF_MOV | BPF_K, 2, 0, 0, 5),
        make_raw(BPF_JMP | BPF_JEQ | BPF_K, 2, 0, 1, 10),
        make_raw(BPF_ALU64 | BPF_MOV | BPF_K, 0, 0, 0, 77),
        make_raw(BPF_JMP | BPF_EXIT, 0, 0, 0, 0),
    ], 77);
}

#[test]
fn jmp_jgt() {
    // R2=10; if R2 > 5 goto +1; R0=0; R0=1; EXIT → 1
    check(&[
        make_raw(BPF_ALU64 | BPF_MOV | BPF_K, 0, 0, 0, 0),
        make_raw(BPF_ALU64 | BPF_MOV | BPF_K, 2, 0, 0, 10),
        make_raw(BPF_JMP | BPF_JGT | BPF_K, 2, 0, 1, 5),
        make_raw(BPF_ALU64 | BPF_MOV | BPF_K, 0, 0, 0, 0),
        make_raw(BPF_ALU64 | BPF_MOV | BPF_K, 0, 0, 0, 1),
        make_raw(BPF_JMP | BPF_EXIT, 0, 0, 0, 0),
    ], 1);
}

#[test]
fn jmp_jne() {
    check(&[
        make_raw(BPF_ALU64 | BPF_MOV | BPF_K, 0, 0, 0, 0),
        make_raw(BPF_ALU64 | BPF_MOV | BPF_K, 2, 0, 0, 5),
        make_raw(BPF_JMP | BPF_JNE | BPF_K, 2, 0, 1, 5),
        make_raw(BPF_ALU64 | BPF_MOV | BPF_K, 0, 0, 0, 88),
        make_raw(BPF_ALU64 | BPF_MOV | BPF_K, 0, 0, 0, 1),
        make_raw(BPF_JMP | BPF_EXIT, 0, 0, 0, 0),
    ], 1);
}

#[test]
fn jmp_ja_unconditional() {
    // JA skips one instruction. Both paths must be reachable for the verifier.
    // Use a conditional branch that's always-true in practice to make both paths reachable.
    // R2=1; if R2 != 0 goto +1; R0=99; EXIT — the JA-like effect is via JNE.
    check(&[
        make_raw(BPF_ALU64 | BPF_MOV | BPF_K, 0, 0, 0, 42),
        make_raw(BPF_ALU64 | BPF_MOV | BPF_K, 2, 0, 0, 1),
        make_raw(BPF_JMP | BPF_JNE | BPF_K, 2, 0, 1, 0),    // if R2 != 0 skip
        make_raw(BPF_ALU64 | BPF_MOV | BPF_K, 0, 0, 0, 99),  // skipped
        make_raw(BPF_JMP | BPF_EXIT, 0, 0, 0, 0),
    ], 42);
}

// =========================================================================
// Callee-saved registers (R6-R9)
// =========================================================================

#[test]
fn callee_saved_r6() {
    check(&[
        make_raw(BPF_ALU64 | BPF_MOV | BPF_K, 6, 0, 0, 42),
        make_raw(BPF_ALU64 | BPF_MOV | BPF_X, 0, 6, 0, 0),
        make_raw(BPF_JMP | BPF_EXIT, 0, 0, 0, 0),
    ], 42);
}

#[test]
fn callee_saved_r9() {
    check(&[
        make_raw(BPF_ALU64 | BPF_MOV | BPF_K, 9, 0, 0, 123),
        make_raw(BPF_ALU64 | BPF_MOV | BPF_X, 0, 9, 0, 0),
        make_raw(BPF_JMP | BPF_EXIT, 0, 0, 0, 0),
    ], 123);
}

// =========================================================================
// Multi-operation sequences
// =========================================================================

#[test]
fn fibonacci_like() {
    // Compute fib(7) = 13 iteratively using R6, R7
    // R6=0, R7=1, loop 7 times: tmp=R7, R7=R6+R7, R6=tmp
    // Unrolled for verifier (no loops allowed)
    check(&[
        make_raw(BPF_ALU64 | BPF_MOV | BPF_K, 6, 0, 0, 0),   // R6 = 0 (a)
        make_raw(BPF_ALU64 | BPF_MOV | BPF_K, 7, 0, 0, 1),   // R7 = 1 (b)
        // iter 1
        make_raw(BPF_ALU64 | BPF_MOV | BPF_X, 8, 7, 0, 0),   // R8 = R7
        make_raw(BPF_ALU64 | BPF_ADD | BPF_X, 7, 6, 0, 0),   // R7 = R7 + R6
        make_raw(BPF_ALU64 | BPF_MOV | BPF_X, 6, 8, 0, 0),   // R6 = R8
        // iter 2
        make_raw(BPF_ALU64 | BPF_MOV | BPF_X, 8, 7, 0, 0),
        make_raw(BPF_ALU64 | BPF_ADD | BPF_X, 7, 6, 0, 0),
        make_raw(BPF_ALU64 | BPF_MOV | BPF_X, 6, 8, 0, 0),
        // iter 3
        make_raw(BPF_ALU64 | BPF_MOV | BPF_X, 8, 7, 0, 0),
        make_raw(BPF_ALU64 | BPF_ADD | BPF_X, 7, 6, 0, 0),
        make_raw(BPF_ALU64 | BPF_MOV | BPF_X, 6, 8, 0, 0),
        // iter 4
        make_raw(BPF_ALU64 | BPF_MOV | BPF_X, 8, 7, 0, 0),
        make_raw(BPF_ALU64 | BPF_ADD | BPF_X, 7, 6, 0, 0),
        make_raw(BPF_ALU64 | BPF_MOV | BPF_X, 6, 8, 0, 0),
        // iter 5
        make_raw(BPF_ALU64 | BPF_MOV | BPF_X, 8, 7, 0, 0),
        make_raw(BPF_ALU64 | BPF_ADD | BPF_X, 7, 6, 0, 0),
        make_raw(BPF_ALU64 | BPF_MOV | BPF_X, 6, 8, 0, 0),
        // iter 6
        make_raw(BPF_ALU64 | BPF_MOV | BPF_X, 8, 7, 0, 0),
        make_raw(BPF_ALU64 | BPF_ADD | BPF_X, 7, 6, 0, 0),
        make_raw(BPF_ALU64 | BPF_MOV | BPF_X, 6, 8, 0, 0),
        // R0 = R7 (fib(7) = 13)
        make_raw(BPF_ALU64 | BPF_MOV | BPF_X, 0, 7, 0, 0),
        make_raw(BPF_JMP | BPF_EXIT, 0, 0, 0, 0),
    ], 13);
}

#[test]
fn chained_alu_ops() {
    // R0 = ((10 + 5) * 3 - 15) / 5 = 6
    check(&[
        make_raw(BPF_ALU64 | BPF_MOV | BPF_K, 0, 0, 0, 10),
        make_raw(BPF_ALU64 | BPF_ADD | BPF_K, 0, 0, 0, 5),
        make_raw(BPF_ALU64 | BPF_MUL | BPF_K, 0, 0, 0, 3),
        make_raw(BPF_ALU64 | BPF_SUB | BPF_K, 0, 0, 0, 15),
        make_raw(BPF_ALU64 | BPF_DIV | BPF_K, 0, 0, 0, 5),
        make_raw(BPF_JMP | BPF_EXIT, 0, 0, 0, 0),
    ], 6);
}

// =========================================================================
// Verifier rejection tests
// =========================================================================

#[test]
fn verifier_rejects_uninit_read() {
    let raw = [
        make_raw(BPF_ALU64 | BPF_ADD | BPF_X, 0, 2, 0, 0), // R0 += R2 (R2 uninit)
        make_raw(BPF_JMP | BPF_EXIT, 0, 0, 0, 0),
    ];
    assert!(compile(&raw, Target::AArch64).is_err());
}

#[test]
fn verifier_rejects_no_exit() {
    let raw = [
        make_raw(BPF_ALU64 | BPF_MOV | BPF_K, 0, 0, 0, 0),
        make_raw(BPF_JMP | BPF_JA, 0, 0, -1, 0), // infinite loop
    ];
    assert!(compile(&raw, Target::AArch64).is_err());
}

#[test]
fn verifier_rejects_write_to_r10() {
    let raw = [
        make_raw(BPF_ALU64 | BPF_MOV | BPF_K, 10, 0, 0, 0),
        make_raw(BPF_JMP | BPF_EXIT, 0, 0, 0, 0),
    ];
    assert!(compile(&raw, Target::AArch64).is_err());
}

#[test]
fn verifier_rejects_oob_branch() {
    let raw = [
        make_raw(BPF_ALU64 | BPF_MOV | BPF_K, 0, 0, 0, 0),
        make_raw(BPF_JMP | BPF_JA, 0, 0, 100, 0),
        make_raw(BPF_JMP | BPF_EXIT, 0, 0, 0, 0),
    ];
    assert!(compile(&raw, Target::AArch64).is_err());
}
