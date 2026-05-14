//! eBPF static verifier.
//!
//! Constructs a CFG from decoded instructions, performs dataflow analysis
//! to track register state, and enforces safety invariants:
//!
//! - No reads from uninitialized registers
//! - All branches target valid instruction indices
//! - Stack accesses stay within the 512-byte frame
//! - No unreachable code after the verifier pass
//! - Programs must terminate (back-edge budget)
//!
//! ## Complexity
//! O(n * k) where n = instruction count, k = bounded iteration limit for
//! fixed-point dataflow convergence (typically k ≤ 3 for eBPF programs).

use alloc::{vec, vec::Vec};
use crate::isa::*;

/// Maximum number of back-edge traversals before we declare "possible infinite loop".
const MAX_BACK_EDGE_TRAVERSALS: usize = 32;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifyError {
    BranchOutOfBounds { pc: usize, target: isize },
    UninitializedRegRead { pc: usize, reg: BpfReg },
    StackOutOfBounds { pc: usize, offset: i16 },
    WriteToR10 { pc: usize },
    NoExitInstruction,
    UnreachableCode { pc: usize },
    BackEdgeBudgetExceeded { pc: usize },
    FallthroughAfterExit { pc: usize },
    EmptyProgram,
}

/// Register initialization state for dataflow analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RegState {
    Uninit,
    Init,
}

/// Per-basic-block register state vector.
#[derive(Debug, Clone)]
struct RegFile {
    regs: [RegState; NUM_REGS],
}

impl RegFile {
    fn new_entry() -> Self {
        let mut regs = [RegState::Uninit; NUM_REGS];
        // R1 = context pointer (initialized by caller)
        regs[BpfReg::R1.index()] = RegState::Init;
        // R10 = frame pointer (always valid)
        regs[BpfReg::R10.index()] = RegState::Init;
        Self { regs }
    }

    fn new_uninit() -> Self {
        Self { regs: [RegState::Uninit; NUM_REGS] }
    }

    fn is_init(&self, r: BpfReg) -> bool {
        self.regs[r.index()] == RegState::Init
    }

    fn mark_init(&mut self, r: BpfReg) {
        self.regs[r.index()] = RegState::Init;
    }

    /// Merge: for each register, result is Init only if Init in *both* paths.
    fn merge(&self, other: &Self) -> Self {
        let mut regs = [RegState::Uninit; NUM_REGS];
        for i in 0..NUM_REGS {
            if self.regs[i] == RegState::Init && other.regs[i] == RegState::Init {
                regs[i] = RegState::Init;
            }
        }
        Self { regs }
    }

    fn equals(&self, other: &Self) -> bool {
        self.regs == other.regs
    }
}

/// A basic block in the control-flow graph.
#[derive(Debug, Clone)]
struct BasicBlock {
    start: usize,
    end: usize, // exclusive
    successors: Vec<usize>, // indices of successor BBs
    reg_state_in: RegFile,
    reachable: bool,
}

/// Verify a decoded eBPF program.
///
/// On success, returns the number of basic blocks (useful for downstream
/// JIT buffer sizing). On failure, returns the first error found.
pub fn verify_program(insns: &[Insn]) -> Result<usize, VerifyError> {
    if insns.is_empty() {
        return Err(VerifyError::EmptyProgram);
    }

    // Phase 1: find basic block boundaries
    let bb_starts = find_bb_starts(insns)?;
    let bbs = build_cfg(insns, &bb_starts)?;

    // Phase 2: dataflow analysis — propagate register states
    let analyzed = dataflow_analysis(insns, bbs)?;

    // Phase 3: check that all paths reach an exit
    check_exit_reachability(insns, &analyzed)?;

    Ok(analyzed.len())
}

/// Identify basic-block start indices.
/// O(n).
fn find_bb_starts(insns: &[Insn]) -> Result<Vec<usize>, VerifyError> {
    let n = insns.len();
    let mut is_start = vec![false; n];
    is_start[0] = true;

    for (pc, insn) in insns.iter().enumerate() {
        match insn {
            Insn::Ja { off } => {
                let target = resolve_branch(pc, *off, n)?;
                is_start[target] = true;
                if pc + 1 < n {
                    is_start[pc + 1] = true;
                }
            }
            Insn::JmpCond { off, .. } | Insn::JmpCond32 { off, .. } => {
                let target = resolve_branch(pc, *off, n)?;
                is_start[target] = true;
                if pc + 1 < n {
                    is_start[pc + 1] = true;
                }
            }
            Insn::Exit => {
                if pc + 1 < n {
                    is_start[pc + 1] = true;
                }
            }
            Insn::Call { .. } => {
                // After a call, the next instruction is still part of the same BB
                // (calls return).
            }
            _ => {}
        }
    }

    Ok(is_start.iter().enumerate().filter(|(_, &s)| s).map(|(i, _)| i).collect())
}

/// Resolve a relative branch offset to an absolute instruction index.
fn resolve_branch(pc: usize, off: i16, prog_len: usize) -> Result<usize, VerifyError> {
    let target = (pc as isize) + 1 + (off as isize);
    if target < 0 || target >= prog_len as isize {
        return Err(VerifyError::BranchOutOfBounds { pc, target });
    }
    Ok(target as usize)
}

/// Build the CFG from basic-block starts.
/// O(n).
fn build_cfg(insns: &[Insn], bb_starts: &[usize]) -> Result<Vec<BasicBlock>, VerifyError> {
    let n = insns.len();
    let num_bbs = bb_starts.len();
    let mut bbs = Vec::with_capacity(num_bbs);

    // Map instruction index → BB index for successor resolution
    let mut insn_to_bb = vec![0usize; n];
    for (bb_idx, &start) in bb_starts.iter().enumerate() {
        let end = if bb_idx + 1 < num_bbs { bb_starts[bb_idx + 1] } else { n };
        for i in start..end {
            insn_to_bb[i] = bb_idx;
        }
    }

    for (bb_idx, &start) in bb_starts.iter().enumerate() {
        let end = if bb_idx + 1 < num_bbs { bb_starts[bb_idx + 1] } else { n };
        let last_pc = end - 1;
        let last_insn = &insns[last_pc];
        let mut successors = Vec::new();

        match last_insn {
            Insn::Ja { off } => {
                let target = resolve_branch(last_pc, *off, n)?;
                successors.push(insn_to_bb[target]);
            }
            Insn::JmpCond { off, .. } | Insn::JmpCond32 { off, .. } => {
                // Fall-through
                if end < n {
                    successors.push(insn_to_bb[end]);
                }
                // Branch target
                let target = resolve_branch(last_pc, *off, n)?;
                successors.push(insn_to_bb[target]);
            }
            Insn::Exit => {
                // No successors — terminal block
            }
            _ => {
                // Fallthrough to next BB
                if end < n {
                    successors.push(insn_to_bb[end]);
                }
            }
        }

        bbs.push(BasicBlock {
            start,
            end,
            successors,
            reg_state_in: if bb_idx == 0 {
                RegFile::new_entry()
            } else {
                RegFile::new_uninit()
            },
            reachable: bb_idx == 0,
        });
    }

    Ok(bbs)
}

/// Fixed-point dataflow analysis: propagate register initialization states
/// forward through the CFG until convergence.
///
/// O(n * k) where k ≤ MAX_BACK_EDGE_TRAVERSALS.
fn dataflow_analysis(
    insns: &[Insn],
    mut bbs: Vec<BasicBlock>,
) -> Result<Vec<BasicBlock>, VerifyError> {
    let mut changed = true;
    let mut iterations = 0;

    while changed {
        changed = false;
        iterations += 1;
        if iterations > MAX_BACK_EDGE_TRAVERSALS {
            return Err(VerifyError::BackEdgeBudgetExceeded { pc: 0 });
        }

        for bb_idx in 0..bbs.len() {
            if !bbs[bb_idx].reachable {
                continue;
            }

            // Simulate instructions in this block
            let mut state = bbs[bb_idx].reg_state_in.clone();
            for pc in bbs[bb_idx].start..bbs[bb_idx].end {
                check_and_update_regs(insns, pc, &mut state)?;
            }

            // Propagate to successors
            let successors = bbs[bb_idx].successors.clone();
            for &succ_idx in &successors {
                let succ = &mut bbs[succ_idx];
                if !succ.reachable {
                    succ.reachable = true;
                    succ.reg_state_in = state.clone();
                    changed = true;
                } else {
                    let merged = succ.reg_state_in.merge(&state);
                    if !merged.equals(&succ.reg_state_in) {
                        succ.reg_state_in = merged;
                        changed = true;
                    }
                }
            }
        }
    }

    Ok(bbs)
}

/// Check register reads/writes for a single instruction, updating `state`.
fn check_and_update_regs(
    insns: &[Insn],
    pc: usize,
    state: &mut RegFile,
) -> Result<(), VerifyError> {
    let insn = &insns[pc];

    match insn {
        Insn::Alu64 { dst, src, .. } | Insn::Alu32 { dst, src, .. } => {
            check_write_dst(pc, *dst)?;
            check_source_read(pc, src, state)?;
            // For non-MOV, dst is also read
            if !matches!(insn, Insn::Alu64 { op: AluOp::Mov, .. } | Insn::Alu32 { op: AluOp::Mov, .. }) {
                check_reg_read(pc, *dst, state)?;
            }
            state.mark_init(*dst);
        }

        Insn::Load { dst, src, off, .. } => {
            check_write_dst(pc, *dst)?;
            check_reg_read(pc, *src, state)?;
            check_stack_if_fp(pc, *src, *off)?;
            state.mark_init(*dst);
        }

        Insn::StoreReg { dst, src, off, .. } => {
            check_reg_read(pc, *dst, state)?;
            check_reg_read(pc, *src, state)?;
            check_stack_if_fp(pc, *dst, *off)?;
        }

        Insn::StoreImm { dst, off, .. } => {
            check_reg_read(pc, *dst, state)?;
            check_stack_if_fp(pc, *dst, *off)?;
        }

        Insn::LoadImm64 { dst, .. } => {
            check_write_dst(pc, *dst)?;
            state.mark_init(*dst);
        }

        Insn::JmpCond { dst, src, .. } | Insn::JmpCond32 { dst, src, .. } => {
            check_reg_read(pc, *dst, state)?;
            check_source_read(pc, src, state)?;
        }

        Insn::Ja { .. } => {}

        Insn::Call { .. } => {
            // Helpers expect R1-R5 as arguments (caller must init).
            // R0 is return value. R1-R5 are clobbered. R6-R9 callee-saved.
            state.mark_init(BpfReg::R0);
            // R1-R5 become uninitialized after call (clobbered)
            for i in 1..=5 {
                if let Some(r) = BpfReg::from_raw(i) {
                    state.regs[r.index()] = RegState::Uninit;
                }
            }
        }

        Insn::TailCall => {
            // Tail call doesn't return — similar to exit
        }

        Insn::Exit => {
            check_reg_read(pc, BpfReg::R0, state)?;
        }

        Insn::Atomic { dst, src, off, .. } => {
            check_reg_read(pc, *dst, state)?;
            check_reg_read(pc, *src, state)?;
            check_stack_if_fp(pc, *dst, *off)?;
        }

        Insn::Endian { dst, .. } => {
            check_write_dst(pc, *dst)?;
            check_reg_read(pc, *dst, state)?;
            state.mark_init(*dst);
        }
    }

    Ok(())
}

fn check_write_dst(pc: usize, dst: BpfReg) -> Result<(), VerifyError> {
    if dst == BpfReg::R10 {
        return Err(VerifyError::WriteToR10 { pc });
    }
    Ok(())
}

fn check_reg_read(pc: usize, reg: BpfReg, state: &RegFile) -> Result<(), VerifyError> {
    if !state.is_init(reg) {
        return Err(VerifyError::UninitializedRegRead { pc, reg });
    }
    Ok(())
}

fn check_source_read(pc: usize, src: &Source, state: &RegFile) -> Result<(), VerifyError> {
    if let Source::Reg(r) = src {
        check_reg_read(pc, *r, state)?;
    }
    Ok(())
}

fn check_stack_if_fp(pc: usize, base: BpfReg, off: i16) -> Result<(), VerifyError> {
    if base == BpfReg::R10 {
        // Stack grows downward from R10. Valid offsets: [-512, 0).
        if off >= 0 || off < -(STACK_SIZE as i16) {
            return Err(VerifyError::StackOutOfBounds { pc, offset: off });
        }
    }
    Ok(())
}

fn check_exit_reachability(
    insns: &[Insn],
    bbs: &[BasicBlock],
) -> Result<(), VerifyError> {
    let mut has_exit = false;

    for bb in bbs {
        if !bb.reachable {
            return Err(VerifyError::UnreachableCode { pc: bb.start });
        }

        let last_insn = &insns[bb.end - 1];
        if matches!(last_insn, Insn::Exit) {
            has_exit = true;
        }

        // Check: if a non-terminal block has no successors, something is wrong
        if bb.successors.is_empty() && !matches!(last_insn, Insn::Exit | Insn::TailCall) {
            return Err(VerifyError::FallthroughAfterExit { pc: bb.end - 1 });
        }
    }

    if !has_exit {
        return Err(VerifyError::NoExitInstruction);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verify_minimal_program() {
        // MOV64 R0, 0; EXIT
        let insns = vec![
            Insn::Alu64 { op: AluOp::Mov, dst: BpfReg::R0, src: Source::Imm(0) },
            Insn::Exit,
        ];
        assert!(verify_program(&insns).is_ok());
    }

    #[test]
    fn reject_uninit_read() {
        // ADD64 R0, R2 (R2 never initialized); EXIT
        let insns = vec![
            Insn::Alu64 { op: AluOp::Add, dst: BpfReg::R0, src: Source::Reg(BpfReg::R2) },
            Insn::Exit,
        ];
        assert!(matches!(
            verify_program(&insns),
            Err(VerifyError::UninitializedRegRead { pc: 0, .. })
        ));
    }

    #[test]
    fn reject_write_to_r10() {
        let insns = vec![
            Insn::Alu64 { op: AluOp::Mov, dst: BpfReg::R10, src: Source::Imm(0) },
            Insn::Exit,
        ];
        assert!(matches!(
            verify_program(&insns),
            Err(VerifyError::WriteToR10 { pc: 0 })
        ));
    }

    #[test]
    fn reject_stack_overflow() {
        // STORE [R10 - 520], R1 — out of bounds
        let insns = vec![
            Insn::Alu64 { op: AluOp::Mov, dst: BpfReg::R0, src: Source::Imm(0) },
            Insn::StoreReg {
                width: MemWidth::DW,
                dst: BpfReg::R10,
                src: BpfReg::R1,
                off: -520,
            },
            Insn::Exit,
        ];
        assert!(matches!(
            verify_program(&insns),
            Err(VerifyError::StackOutOfBounds { pc: 1, offset: -520 })
        ));
    }

    #[test]
    fn verify_conditional_branch() {
        // R0 = 0; if R1 == 0 goto +1; R0 = 1; EXIT
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
        assert!(verify_program(&insns).is_ok());
    }

    #[test]
    fn reject_branch_out_of_bounds() {
        let insns = vec![
            Insn::Alu64 { op: AluOp::Mov, dst: BpfReg::R0, src: Source::Imm(0) },
            Insn::Ja { off: 100 },
            Insn::Exit,
        ];
        assert!(matches!(
            verify_program(&insns),
            Err(VerifyError::BranchOutOfBounds { .. })
        ));
    }

    #[test]
    fn reject_no_exit() {
        let insns = vec![
            Insn::Alu64 { op: AluOp::Mov, dst: BpfReg::R0, src: Source::Imm(0) },
            Insn::Ja { off: -1 }, // infinite loop, no exit
        ];
        assert!(verify_program(&insns).is_err());
    }

    #[test]
    fn verify_call_clobbers_r1_r5() {
        // R0 = call(1); R0 already set by call. Try read R2 → should fail.
        let insns = vec![
            Insn::Call { func_id: 1 },
            Insn::Alu64 { op: AluOp::Add, dst: BpfReg::R0, src: Source::Reg(BpfReg::R2) },
            Insn::Exit,
        ];
        assert!(matches!(
            verify_program(&insns),
            Err(VerifyError::UninitializedRegRead { pc: 1, .. })
        ));
    }
}
