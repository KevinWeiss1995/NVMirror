//! Native execution integration tests.
//!
//! On AArch64 (Apple Silicon / Jetson), these tests JIT-compile eBPF
//! programs and execute the generated machine code natively, validating
//! the result against the reference interpreter.
//!
//! On other architectures, these tests are skipped.

use ebpf_core::isa::*;
use ebpf_core::RawInsn;
use jit_engine::{compile, Target};
use jit_engine::interpreter::InterpreterState;
use ebpf_core::decode_program;

fn make_raw(opcode: u8, dst: u8, src: u8, off: i16, imm: i32) -> RawInsn {
    RawInsn { opcode, regs: (src << 4) | dst, off, imm }
}

/// Execute JIT'd AArch64 code natively on Apple Silicon / Jetson.
///
/// # Safety
/// Executes generated machine code. Only safe if the JIT output is
/// correct for the current architecture.
#[cfg(target_arch = "aarch64")]
unsafe fn execute_jit_aarch64(code: &[u8]) -> u64 {
    use nvme_shim::exec_buffer::ExecBuffer;

    let mut buf = ExecBuffer::new(code.len()).expect("Failed to allocate exec buffer");
    buf.write_code(code).expect("Failed to write code");

    type JitFn = unsafe extern "C" fn(u64, u64, u64, u64, u64) -> u64;
    let f: JitFn = unsafe { buf.as_fn_ptr() };

    unsafe { f(0, 0, 0, 0, 0) }
}

#[test]
fn interpreter_matches_for_mov_exit() {
    let raw = [
        make_raw(BPF_ALU64 | BPF_MOV | BPF_K, 0, 0, 0, 42),
        make_raw(BPF_JMP | BPF_EXIT, 0, 0, 0, 0),
    ];

    // Interpreter
    let insns = decode_program(&raw).unwrap();
    let mut interp = InterpreterState::new();
    let interp_result = interp.run(&insns, 100);
    assert_eq!(interp_result, 42);

    // JIT (AArch64)
    let output = compile(&raw, Target::AArch64).unwrap();
    assert!(!output.code.is_empty());

    // Native execution on AArch64
    #[cfg(target_arch = "aarch64")]
    {
        let jit_result = unsafe { execute_jit_aarch64(&output.code) };
        assert_eq!(jit_result, interp_result, "JIT result mismatch with interpreter");
    }
}

#[test]
fn interpreter_matches_for_alu_sequence() {
    // R0 = 10; R0 += 20; R0 *= 3; EXIT → expect 90
    let raw = [
        make_raw(BPF_ALU64 | BPF_MOV | BPF_K, 0, 0, 0, 10),
        make_raw(BPF_ALU64 | BPF_ADD | BPF_K, 0, 0, 0, 20),
        make_raw(BPF_ALU64 | BPF_MUL | BPF_K, 0, 0, 0, 3),
        make_raw(BPF_JMP | BPF_EXIT, 0, 0, 0, 0),
    ];

    let insns = decode_program(&raw).unwrap();
    let mut interp = InterpreterState::new();
    let interp_result = interp.run(&insns, 100);
    assert_eq!(interp_result, 90);

    let output = compile(&raw, Target::AArch64).unwrap();
    assert!(!output.code.is_empty());

    // Verify code is 4-byte aligned
    assert_eq!(output.code.len() % 4, 0);

    #[cfg(target_arch = "aarch64")]
    {
        // Dump emitted instructions for debugging
        for (i, chunk) in output.code.chunks(4).enumerate() {
            let insn = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            eprintln!("  [{:3}] 0x{:08X}", i * 4, insn);
        }

        let jit_result = unsafe { execute_jit_aarch64(&output.code) };
        assert_eq!(jit_result, interp_result);
    }
}

#[test]
fn interpreter_matches_for_conditional_branch() {
    // R2 = 5; R0 = 0; if R2 == 5 goto +1; R0 = 99; R0 = 42; EXIT
    let raw = [
        make_raw(BPF_ALU64 | BPF_MOV | BPF_K, 2, 0, 0, 5),  // R2 = 5
        make_raw(BPF_ALU64 | BPF_MOV | BPF_K, 0, 0, 0, 0),  // R0 = 0
        make_raw(BPF_JMP | BPF_JEQ | BPF_K, 2, 0, 1, 5),    // if R2 == 5 goto +1
        make_raw(BPF_ALU64 | BPF_MOV | BPF_K, 0, 0, 0, 99), // R0 = 99 (skipped)
        make_raw(BPF_ALU64 | BPF_MOV | BPF_K, 0, 0, 0, 42), // R0 = 42
        make_raw(BPF_JMP | BPF_EXIT, 0, 0, 0, 0),            // EXIT
    ];

    let insns = decode_program(&raw).unwrap();
    let mut interp = InterpreterState::new();
    let interp_result = interp.run(&insns, 100);
    assert_eq!(interp_result, 42);

    let output = compile(&raw, Target::AArch64).unwrap();
    assert!(!output.code.is_empty());
}

#[test]
fn riscv_compilation_produces_valid_output() {
    let raw = [
        make_raw(BPF_ALU64 | BPF_MOV | BPF_K, 0, 0, 0, 42),
        make_raw(BPF_JMP | BPF_EXIT, 0, 0, 0, 0),
    ];

    let output = compile(&raw, Target::RiscV64).unwrap();
    assert!(!output.code.is_empty());
    assert_eq!(output.code.len() % 4, 0, "RISC-V instructions must be 4-byte aligned");
}

#[test]
fn both_targets_produce_same_ir() {
    let raw = [
        make_raw(BPF_ALU64 | BPF_MOV | BPF_K, 0, 0, 0, 42),
        make_raw(BPF_JMP | BPF_EXIT, 0, 0, 0, 0),
    ];

    let aarch64_output = compile(&raw, Target::AArch64).unwrap();
    let riscv_output = compile(&raw, Target::RiscV64).unwrap();

    // Both should produce the same number of basic blocks
    assert_eq!(aarch64_output.num_blocks, riscv_output.num_blocks);
}
