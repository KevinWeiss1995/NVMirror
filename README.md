# NVMirror

A provably correct eBPF JIT compiler targeting NVMe controller processors.

## Architecture

```
eBPF Bytecode → Decoder → Verifier → IR Lowering → Register Allocation → Code Emission → Executable Buffer
                (ebpf-core)           (jit-ir)       (jit-regalloc)      (jit-aarch64 / jit-riscv)  (nvme-shim)
```

**Targets:** AArch64 (Apple M-series, Jetson Orin, Cortex-R82) and RISC-V RV64IMC.

## Crates

| Crate | Description |
|-------|-------------|
| `ebpf-core` | eBPF ISA definitions, bytecode decoder, static verifier |
| `jit-ir` | Typed intermediate representation with `CodeEmitter` trait |
| `jit-regalloc` | Fixed-mapping register allocator (eBPF R0-R10 → physical regs) |
| `jit-aarch64` | AArch64 instruction encoder and emitter |
| `jit-riscv` | RISC-V RV64IMC instruction encoder and emitter |
| `jit-engine` | Pipeline orchestrator + reference interpreter |
| `nvme-shim` | Executable buffer management (MAP_JIT on macOS, mmap on Linux) |

All crates are `#![no_std]` compatible for firmware deployment.

## Building

```bash
cargo build
cargo test
```

## Testing

- **Unit tests:** Per-crate tests for decoder, verifier, instruction encoders
- **Native execution:** JIT'd AArch64 code runs natively on Apple Silicon / Jetson
- **Conformance suite:** 32 test vectors covering ALU, branches, LDDW, callee-saved regs
- **Differential fuzzer:** 1000+ random programs validated against the reference interpreter

```bash
cargo test                           # all tests
cargo test -p nvmirror-fuzz          # fuzzer only
cargo test -p jit-engine --test conformance  # conformance suite
```

## Correct-by-Construction

- Bounded `BpfReg(0..10)` enum — invalid register references are unrepresentable
- Branch targets are `BasicBlockId` indices — no raw offset arithmetic in the backend
- Verifier enforces: no uninit reads, no R10 writes, stack bounds, reachability
- Two-pass compilation: dry-run calculates exact buffer size before emission
- W^X enforcement via MAP_JIT + `pthread_jit_write_protect_np` on Apple Silicon

## License

MIT
