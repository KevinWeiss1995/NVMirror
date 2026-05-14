//! NVMirror Pipeline Visualizer
//!
//! Shows the full JIT compilation pipeline stage-by-stage:
//!   Raw bytecode → Decode → Verify → Lower (IR) → Emit (machine code) → Execute
//!
//! Run: `cargo run -p nvmirror-disasm`
//! Or with a specific example: `cargo run -p nvmirror-disasm -- --example fibonacci`

use ebpf_core::isa::*;
use ebpf_core::{RawInsn, decode_program, verify_program};
use jit_engine::{compile, Target};
use jit_engine::interpreter::InterpreterState;
use jit_regalloc::{lower_to_ir, aarch64_regmap};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let example = args.iter()
        .position(|a| a == "--example")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("all");

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║          NVMirror — eBPF JIT Pipeline Visualizer             ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    match example {
        "fibonacci" => run_example("Fibonacci (R6/R7 iterative)", fibonacci_program(), Some(13)),
        "arithmetic" => run_example("Chained Arithmetic", arithmetic_program(), Some(6)),
        "branch" => run_example("Conditional Branch", branch_program(), Some(42)),
        "minimal" => run_example("Minimal (return 42)", minimal_program(), Some(42)),
        "bitwise" => run_example("Bitwise Operations", bitwise_program(), Some(0x9E)),
        "all" | _ => {
            run_example("Minimal (return 42)", minimal_program(), Some(42));
            println!("\n{}\n", "─".repeat(70));
            run_example("Chained Arithmetic: ((10+5)*3-15)/5", arithmetic_program(), Some(6));
            println!("\n{}\n", "─".repeat(70));
            run_example("Conditional Branch", branch_program(), Some(42));
            println!("\n{}\n", "─".repeat(70));
            run_example("Fibonacci(7) via R6/R7", fibonacci_program(), Some(13));
            println!("\n{}\n", "─".repeat(70));
            run_example("Bitwise Operations", bitwise_program(), Some(0x9E));
        }
    }
}

fn run_example(name: &str, raw: Vec<RawInsn>, expected: Option<u64>) {
    println!("┌─── Example: {} ───", name);
    println!("│");

    // Stage 1: Raw bytecode
    println!("│  ┌── Stage 1: RAW BYTECODE ({} instructions, {} bytes)", raw.len(), raw.len() * 8);
    for (i, r) in raw.iter().enumerate() {
        println!("│  │  [{:3}] op=0x{:02X} dst=r{} src=r{} off={:+} imm={}",
            i, r.opcode, r.dst_reg(), r.src_reg(), r.off, r.imm);
    }
    println!("│  └──");
    println!("│");

    // Stage 2: Decode
    let insns = match decode_program(&raw) {
        Ok(insns) => {
            println!("│  ┌── Stage 2: DECODED ({} typed instructions)", insns.len());
            for (i, insn) in insns.iter().enumerate() {
                println!("│  │  [{:3}] {:?}", i, insn);
            }
            println!("│  └──");
            insns
        }
        Err(e) => {
            println!("│  ✗ DECODE FAILED: {:?}", e);
            return;
        }
    };
    println!("│");

    // Stage 3: Verify
    match verify_program(&insns) {
        Ok(num_bbs) => {
            println!("│  ┌── Stage 3: VERIFIED ✓ ({} basic blocks)", num_bbs);
            println!("│  │  • No uninitialized register reads");
            println!("│  │  • All branches in-bounds");
            println!("│  │  • Stack accesses within 512-byte frame");
            println!("│  │  • All paths reach EXIT");
            println!("│  └──");
        }
        Err(e) => {
            println!("│  ✗ VERIFY FAILED: {:?}", e);
            return;
        }
    }
    println!("│");

    // Stage 4: Lower to IR (AArch64)
    let regmap = aarch64_regmap();
    let ir = lower_to_ir(&insns, &regmap);
    println!("│  ┌── Stage 4: LOWERED IR ({} blocks, frame_size={}B)", ir.blocks.len(), ir.frame_size);
    for block in &ir.blocks {
        println!("│  │  BB{}:", block.id.0);
        for node in &block.nodes {
            println!("│  │    {:?}", node);
        }
    }
    println!("│  └──");
    println!("│");

    // Stage 5: Emit AArch64
    let output = match compile(&raw, Target::AArch64) {
        Ok(o) => o,
        Err(e) => {
            println!("│  ✗ EMIT FAILED: {:?}", e);
            return;
        }
    };
    println!("│  ┌── Stage 5: AArch64 MACHINE CODE ({} bytes, {} instructions)",
        output.code.len(), output.code.len() / 4);
    for (i, chunk) in output.code.chunks(4).enumerate() {
        let word = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let disasm = disassemble_aarch64(word);
        println!("│  │  {:04X}: {:08X}  {}", i * 4, word, disasm);
    }
    println!("│  └──");
    println!("│");

    // Stage 5b: Also show RISC-V
    if let Ok(rv_output) = compile(&raw, Target::RiscV64) {
        println!("│  ┌── Stage 5b: RISC-V MACHINE CODE ({} bytes, {} instructions)",
            rv_output.code.len(), rv_output.code.len() / 4);
        for (i, chunk) in rv_output.code.chunks(4).enumerate() {
            let word = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            println!("│  │  {:04X}: {:08X}", i * 4, word);
        }
        println!("│  └──");
        println!("│");
    }

    // Stage 6: Execute
    println!("│  ┌── Stage 6: EXECUTION");

    // Interpreter
    let mut interp = InterpreterState::new();
    let interp_result = interp.run(&insns, 10_000);
    println!("│  │  Interpreter result: R0 = {} (0x{:X})", interp_result, interp_result);

    // Native execution on AArch64
    #[cfg(target_arch = "aarch64")]
    {
        let native_result = unsafe { execute_native(&output.code) };
        println!("│  │  Native AArch64:    R0 = {} (0x{:X})", native_result, native_result);

        if native_result == interp_result {
            println!("│  │  ✓ JIT matches interpreter");
        } else {
            println!("│  │  ✗ MISMATCH! JIT={} vs Interp={}", native_result, interp_result);
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        println!("│  │  (Native execution skipped — not on AArch64)");
    }

    if let Some(exp) = expected {
        if interp_result == exp {
            println!("│  │  ✓ Expected value: {}", exp);
        } else {
            println!("│  │  ✗ UNEXPECTED: got {} expected {}", interp_result, exp);
        }
    }

    println!("│  └──");
    println!("└───");
}

#[cfg(target_arch = "aarch64")]
unsafe fn execute_native(code: &[u8]) -> u64 {
    use nvme_shim::exec_buffer::ExecBuffer;

    let mut buf = ExecBuffer::new(code.len()).expect("mmap failed");
    buf.write_code(code).expect("write_code failed");

    type JitFn = unsafe extern "C" fn(u64, u64, u64, u64, u64) -> u64;
    let f: JitFn = unsafe { buf.as_fn_ptr() };
    unsafe { f(0, 0, 0, 0, 0) }
}

/// Rudimentary AArch64 disassembler — decodes common instruction patterns.
fn disassemble_aarch64(insn: u32) -> String {
    let rd = (insn & 0x1F) as u8;
    let rn = ((insn >> 5) & 0x1F) as u8;
    let rm = ((insn >> 16) & 0x1F) as u8;

    // RET
    if insn == 0xD65F03C0 {
        return "RET".into();
    }

    // DMB ISH
    if insn == 0xD5033BBF {
        return "DMB ISH".into();
    }

    // MOV Xd, Xm (ORR Xd, XZR, Xm)
    if insn & 0xFFE0FFE0 == 0xAA0003E0 {
        return format!("MOV X{}, X{}", rd, rm);
    }

    // MOVZ Xd, #imm16, LSL #(hw*16)
    if insn & 0xFF800000 == 0xD2800000 {
        let hw = ((insn >> 21) & 3) as u32;
        let imm16 = ((insn >> 5) & 0xFFFF) as u64;
        let val = imm16 << (hw * 16);
        return format!("MOVZ X{}, #0x{:X} (={})", rd, val, val);
    }

    // MOVK Xd, #imm16, LSL #(hw*16)
    if insn & 0xFF800000 == 0xF2800000 {
        let hw = ((insn >> 21) & 3) as u32;
        let imm16 = (insn >> 5) & 0xFFFF;
        return format!("MOVK X{}, #0x{:X}, LSL #{}", rd, imm16, hw * 16);
    }

    // MOVZ Wd
    if insn & 0xFF800000 == 0x52800000 {
        let imm16 = (insn >> 5) & 0xFFFF;
        return format!("MOVZ W{}, #0x{:X}", rd, imm16);
    }

    // ADD Xd, Xn, Xm
    if insn & 0xFFE00000 == 0x8B000000 {
        return format!("ADD X{}, X{}, X{}", rd, rn, rm);
    }

    // SUB Xd, Xn, Xm
    if insn & 0xFFE00000 == 0xCB000000 {
        return format!("SUB X{}, X{}, X{}", rd, rn, rm);
    }

    // ADD Xd, Xn, #imm12
    if insn & 0xFFC00000 == 0x91000000 {
        let imm12 = (insn >> 10) & 0xFFF;
        return format!("ADD X{}, X{}, #{}", rd, rn, imm12);
    }

    // SUB Xd, Xn, #imm12
    if insn & 0xFFC00000 == 0xD1000000 {
        let imm12 = (insn >> 10) & 0xFFF;
        return format!("SUB X{}, X{}, #{}", rd, rn, imm12);
    }

    // MUL Xd, Xn, Xm (MADD Xd, Xn, Xm, XZR)
    if insn & 0xFFE08000 == 0x9B007C00 {
        return format!("MUL X{}, X{}, X{}", rd, rn, rm);
    }

    // UDIV Xd, Xn, Xm
    if insn & 0xFFE0FC00 == 0x9AC00800 {
        return format!("UDIV X{}, X{}, X{}", rd, rn, rm);
    }

    // MSUB Xd, Xn, Xm, Xa
    if insn & 0xFFE08000 == 0x9B008000 {
        let ra = ((insn >> 10) & 0x1F) as u8;
        return format!("MSUB X{}, X{}, X{}, X{}", rd, rn, rm, ra);
    }

    // ORR Xd, Xn, Xm
    if insn & 0xFFE00000 == 0xAA000000 {
        return format!("ORR X{}, X{}, X{}", rd, rn, rm);
    }

    // AND Xd, Xn, Xm
    if insn & 0xFFE00000 == 0x8A000000 {
        return format!("AND X{}, X{}, X{}", rd, rn, rm);
    }

    // EOR Xd, Xn, Xm
    if insn & 0xFFE00000 == 0xCA000000 {
        return format!("EOR X{}, X{}, X{}", rd, rn, rm);
    }

    // LSLV Xd, Xn, Xm
    if insn & 0xFFE0FC00 == 0x9AC02000 {
        return format!("LSL X{}, X{}, X{}", rd, rn, rm);
    }

    // LSRV Xd, Xn, Xm
    if insn & 0xFFE0FC00 == 0x9AC02400 {
        return format!("LSR X{}, X{}, X{}", rd, rn, rm);
    }

    // ASRV Xd, Xn, Xm
    if insn & 0xFFE0FC00 == 0x9AC02800 {
        return format!("ASR X{}, X{}, X{}", rd, rn, rm);
    }

    // CMP Xn, Xm (SUBS XZR, Xn, Xm)
    if insn & 0xFFE00000 == 0xEB000000 && rd == 31 {
        return format!("CMP X{}, X{}", rn, rm);
    }

    // B.cond
    if insn & 0xFF000010 == 0x54000000 {
        let cond = insn & 0xF;
        let imm19 = ((insn >> 5) & 0x7FFFF) as i32;
        let offset = (imm19 << 13) >> 13; // sign extend
        let cc = match cond {
            0 => "EQ", 1 => "NE", 2 => "HS", 3 => "LO",
            8 => "HI", 9 => "LS", 0xA => "GE", 0xB => "LT",
            0xC => "GT", 0xD => "LE", _ => "??",
        };
        return format!("B.{} {:+} (pc{:+})", cc, offset * 4, offset * 4);
    }

    // B (unconditional)
    if insn & 0xFC000000 == 0x14000000 {
        let imm26 = (insn & 0x03FFFFFF) as i32;
        let offset = (imm26 << 6) >> 6; // sign extend
        return format!("B {:+} (pc{:+})", offset * 4, offset * 4);
    }

    // BLR Xn
    if insn & 0xFFFFFC1F == 0xD63F0000 {
        return format!("BLR X{}", rn);
    }

    // STP (pre-index, post-index, offset)
    if insn & 0xFFC00000 == 0xA9800000 {
        return format!("STP X{}, X{}, [X{}]!", rd, ((insn >> 10) & 0x1F), rn);
    }
    if insn & 0xFFC00000 == 0xA9000000 {
        let rt2 = (insn >> 10) & 0x1F;
        let imm7 = ((insn >> 15) & 0x7F) as i8;
        let off = (imm7 as i16) * 8;
        return format!("STP X{}, X{}, [X{}, #{}]", rd, rt2, rn, off);
    }

    // LDP
    if insn & 0xFFC00000 == 0xA9400000 {
        let rt2 = (insn >> 10) & 0x1F;
        let imm7 = ((insn >> 15) & 0x7F) as i8;
        let off = (imm7 as i16) * 8;
        return format!("LDP X{}, X{}, [X{}, #{}]", rd, rt2, rn, off);
    }
    if insn & 0xFFC00000 == 0xA8C00000 {
        return format!("LDP X{}, X{}, [X{}], #post", rd, ((insn >> 10) & 0x1F), rn);
    }

    // LDR Xt, [Xn, #off] (64-bit unsigned offset)
    if insn & 0xFFC00000 == 0xF9400000 {
        let uoff = ((insn >> 10) & 0xFFF) * 8;
        return format!("LDR X{}, [X{}, #{}]", rd, rn, uoff);
    }

    // STR Xt, [Xn, #off] (64-bit unsigned offset)
    if insn & 0xFFC00000 == 0xF9000000 {
        let uoff = ((insn >> 10) & 0xFFF) * 8;
        return format!("STR X{}, [X{}, #{}]", rd, rn, uoff);
    }

    // Shift immediate (UBFM/SBFM patterns)
    if insn & 0xFFC00000 == 0xD3400000 {
        let immr = (insn >> 16) & 0x3F;
        let imms = (insn >> 10) & 0x3F;
        if imms == 63 - immr + (64 - immr) - 1 || true {
            return format!("LSL X{}, X{}, #{}", rd, rn, (64 - immr) & 63);
        }
    }
    if insn & 0xFFC00000 == 0xD340FC00 & 0xFFC00000 {
        if (insn >> 10) & 0x3F == 63 {
            let shift = (insn >> 16) & 0x3F;
            return format!("LSR X{}, X{}, #{}", rd, rn, shift);
        }
    }

    format!("??? (0x{:08X})", insn)
}

// ─── Example Programs ───────────────────────────────────────────────────────

fn make_raw(opcode: u8, dst: u8, src: u8, off: i16, imm: i32) -> RawInsn {
    RawInsn { opcode, regs: (src << 4) | dst, off, imm }
}

fn minimal_program() -> Vec<RawInsn> {
    vec![
        make_raw(BPF_ALU64 | BPF_MOV | BPF_K, 0, 0, 0, 42),
        make_raw(BPF_JMP | BPF_EXIT, 0, 0, 0, 0),
    ]
}

fn arithmetic_program() -> Vec<RawInsn> {
    // ((10 + 5) * 3 - 15) / 5 = 6
    vec![
        make_raw(BPF_ALU64 | BPF_MOV | BPF_K, 0, 0, 0, 10),
        make_raw(BPF_ALU64 | BPF_ADD | BPF_K, 0, 0, 0, 5),
        make_raw(BPF_ALU64 | BPF_MUL | BPF_K, 0, 0, 0, 3),
        make_raw(BPF_ALU64 | BPF_SUB | BPF_K, 0, 0, 0, 15),
        make_raw(BPF_ALU64 | BPF_DIV | BPF_K, 0, 0, 0, 5),
        make_raw(BPF_JMP | BPF_EXIT, 0, 0, 0, 0),
    ]
}

fn branch_program() -> Vec<RawInsn> {
    // R2=5; R0=0; if R2==5 goto +1; R0=99; R0=42; EXIT
    vec![
        make_raw(BPF_ALU64 | BPF_MOV | BPF_K, 2, 0, 0, 5),
        make_raw(BPF_ALU64 | BPF_MOV | BPF_K, 0, 0, 0, 0),
        make_raw(BPF_JMP | BPF_JEQ | BPF_K, 2, 0, 1, 5),
        make_raw(BPF_ALU64 | BPF_MOV | BPF_K, 0, 0, 0, 99),
        make_raw(BPF_ALU64 | BPF_MOV | BPF_K, 0, 0, 0, 42),
        make_raw(BPF_JMP | BPF_EXIT, 0, 0, 0, 0),
    ]
}

fn fibonacci_program() -> Vec<RawInsn> {
    // fib(7) = 13 using R6, R7, R8 (callee-saved)
    vec![
        make_raw(BPF_ALU64 | BPF_MOV | BPF_K, 6, 0, 0, 0),   // R6 = 0
        make_raw(BPF_ALU64 | BPF_MOV | BPF_K, 7, 0, 0, 1),   // R7 = 1
        // 6 iterations of: tmp=R7; R7+=R6; R6=tmp
        make_raw(BPF_ALU64 | BPF_MOV | BPF_X, 8, 7, 0, 0),
        make_raw(BPF_ALU64 | BPF_ADD | BPF_X, 7, 6, 0, 0),
        make_raw(BPF_ALU64 | BPF_MOV | BPF_X, 6, 8, 0, 0),
        make_raw(BPF_ALU64 | BPF_MOV | BPF_X, 8, 7, 0, 0),
        make_raw(BPF_ALU64 | BPF_ADD | BPF_X, 7, 6, 0, 0),
        make_raw(BPF_ALU64 | BPF_MOV | BPF_X, 6, 8, 0, 0),
        make_raw(BPF_ALU64 | BPF_MOV | BPF_X, 8, 7, 0, 0),
        make_raw(BPF_ALU64 | BPF_ADD | BPF_X, 7, 6, 0, 0),
        make_raw(BPF_ALU64 | BPF_MOV | BPF_X, 6, 8, 0, 0),
        make_raw(BPF_ALU64 | BPF_MOV | BPF_X, 8, 7, 0, 0),
        make_raw(BPF_ALU64 | BPF_ADD | BPF_X, 7, 6, 0, 0),
        make_raw(BPF_ALU64 | BPF_MOV | BPF_X, 6, 8, 0, 0),
        make_raw(BPF_ALU64 | BPF_MOV | BPF_X, 8, 7, 0, 0),
        make_raw(BPF_ALU64 | BPF_ADD | BPF_X, 7, 6, 0, 0),
        make_raw(BPF_ALU64 | BPF_MOV | BPF_X, 6, 8, 0, 0),
        make_raw(BPF_ALU64 | BPF_MOV | BPF_X, 8, 7, 0, 0),
        make_raw(BPF_ALU64 | BPF_ADD | BPF_X, 7, 6, 0, 0),
        make_raw(BPF_ALU64 | BPF_MOV | BPF_X, 6, 8, 0, 0),
        // R0 = R7
        make_raw(BPF_ALU64 | BPF_MOV | BPF_X, 0, 7, 0, 0),
        make_raw(BPF_JMP | BPF_EXIT, 0, 0, 0, 0),
    ]
}

fn bitwise_program() -> Vec<RawInsn> {
    // R0 = 0xFF; R0 &= 0xFE; R0 |= 0x40; R0 ^= 0x60 → 0xDE
    vec![
        make_raw(BPF_ALU64 | BPF_MOV | BPF_K, 0, 0, 0, 0xFF),
        make_raw(BPF_ALU64 | BPF_AND | BPF_K, 0, 0, 0, 0xFEu8 as i32),
        make_raw(BPF_ALU64 | BPF_OR  | BPF_K, 0, 0, 0, 0x40),
        make_raw(BPF_ALU64 | BPF_XOR | BPF_K, 0, 0, 0, 0x60),
        make_raw(BPF_JMP | BPF_EXIT, 0, 0, 0, 0),
    ]
}
