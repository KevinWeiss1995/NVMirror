//! AArch64 instruction encoding primitives.
//!
//! Each function returns a 32-bit little-endian encoded AArch64 instruction.
//! Naming follows the ARM Architecture Reference Manual (ARMv8-A).
//!
//! Register arguments are raw register numbers (0-31, where 31 = SP/ZR
//! depending on context).

// ---------------------------------------------------------------------------
// Condition codes (4-bit, used in B.cond and CSEL)
// ---------------------------------------------------------------------------

pub const CC_EQ: u8 = 0x0;
pub const CC_NE: u8 = 0x1;
pub const CC_HS: u8 = 0x2; // (CS) carry set / unsigned >=
pub const CC_LO: u8 = 0x3; // (CC) carry clear / unsigned <
pub const CC_HI: u8 = 0x8;
pub const CC_LS: u8 = 0x9;
pub const CC_GE: u8 = 0xA;
pub const CC_LT: u8 = 0xB;
pub const CC_GT: u8 = 0xC;
pub const CC_LE: u8 = 0xD;
pub const CC_AL: u8 = 0xE;

// ---------------------------------------------------------------------------
// Data processing — register (64-bit, X-form)
// ---------------------------------------------------------------------------

/// ADD Xd, Xn, Xm
#[inline]
pub const fn add_x_reg(rd: u8, rn: u8, rm: u8) -> u32 {
    0x8B000000 | ((rm as u32) << 16) | ((rn as u32) << 5) | (rd as u32)
}

/// SUB Xd, Xn, Xm
#[inline]
pub const fn sub_x_reg(rd: u8, rn: u8, rm: u8) -> u32 {
    0xCB000000 | ((rm as u32) << 16) | ((rn as u32) << 5) | (rd as u32)
}

/// MUL Xd, Xn, Xm (alias: MADD Xd, Xn, Xm, XZR)
#[inline]
pub const fn mul_x(rd: u8, rn: u8, rm: u8) -> u32 {
    0x9B007C00 | ((rm as u32) << 16) | ((rn as u32) << 5) | (rd as u32)
}

/// UDIV Xd, Xn, Xm
#[inline]
pub const fn udiv_x(rd: u8, rn: u8, rm: u8) -> u32 {
    0x9AC00800 | ((rm as u32) << 16) | ((rn as u32) << 5) | (rd as u32)
}

/// MSUB Xd, Xn, Xm, Xa — Xd = Xa - (Xn * Xm)
#[inline]
pub const fn msub_x(rd: u8, rn: u8, rm: u8, ra: u8) -> u32 {
    0x9B008000 | ((rm as u32) << 16) | ((ra as u32) << 10) | ((rn as u32) << 5) | (rd as u32)
}

/// ORR Xd, Xn, Xm
#[inline]
pub const fn orr_x_reg(rd: u8, rn: u8, rm: u8) -> u32 {
    0xAA000000 | ((rm as u32) << 16) | ((rn as u32) << 5) | (rd as u32)
}

/// AND Xd, Xn, Xm
#[inline]
pub const fn and_x_reg(rd: u8, rn: u8, rm: u8) -> u32 {
    0x8A000000 | ((rm as u32) << 16) | ((rn as u32) << 5) | (rd as u32)
}

/// EOR Xd, Xn, Xm
#[inline]
pub const fn eor_x_reg(rd: u8, rn: u8, rm: u8) -> u32 {
    0xCA000000 | ((rm as u32) << 16) | ((rn as u32) << 5) | (rd as u32)
}

/// LSLV Xd, Xn, Xm
#[inline]
pub const fn lslv_x(rd: u8, rn: u8, rm: u8) -> u32 {
    0x9AC02000 | ((rm as u32) << 16) | ((rn as u32) << 5) | (rd as u32)
}

/// LSRV Xd, Xn, Xm
#[inline]
pub const fn lsrv_x(rd: u8, rn: u8, rm: u8) -> u32 {
    0x9AC02400 | ((rm as u32) << 16) | ((rn as u32) << 5) | (rd as u32)
}

/// ASRV Xd, Xn, Xm
#[inline]
pub const fn asrv_x(rd: u8, rn: u8, rm: u8) -> u32 {
    0x9AC02800 | ((rm as u32) << 16) | ((rn as u32) << 5) | (rd as u32)
}

/// MOV Xd, Xm (alias: ORR Xd, XZR, Xm)
#[inline]
pub const fn mov_x_reg(rd: u8, rm: u8) -> u32 {
    orr_x_reg(rd, 31, rm)
}

// ---------------------------------------------------------------------------
// Data processing — register (32-bit, W-form)
// ---------------------------------------------------------------------------

#[inline]
pub const fn add_w_reg(rd: u8, rn: u8, rm: u8) -> u32 {
    0x0B000000 | ((rm as u32) << 16) | ((rn as u32) << 5) | (rd as u32)
}

#[inline]
pub const fn sub_w_reg(rd: u8, rn: u8, rm: u8) -> u32 {
    0x4B000000 | ((rm as u32) << 16) | ((rn as u32) << 5) | (rd as u32)
}

#[inline]
pub const fn mul_w(rd: u8, rn: u8, rm: u8) -> u32 {
    0x1B007C00 | ((rm as u32) << 16) | ((rn as u32) << 5) | (rd as u32)
}

#[inline]
pub const fn udiv_w(rd: u8, rn: u8, rm: u8) -> u32 {
    0x1AC00800 | ((rm as u32) << 16) | ((rn as u32) << 5) | (rd as u32)
}

#[inline]
pub const fn msub_w(rd: u8, rn: u8, rm: u8, ra: u8) -> u32 {
    0x1B008000 | ((rm as u32) << 16) | ((ra as u32) << 10) | ((rn as u32) << 5) | (rd as u32)
}

#[inline]
pub const fn orr_w_reg(rd: u8, rn: u8, rm: u8) -> u32 {
    0x2A000000 | ((rm as u32) << 16) | ((rn as u32) << 5) | (rd as u32)
}

#[inline]
pub const fn and_w_reg(rd: u8, rn: u8, rm: u8) -> u32 {
    0x0A000000 | ((rm as u32) << 16) | ((rn as u32) << 5) | (rd as u32)
}

#[inline]
pub const fn eor_w_reg(rd: u8, rn: u8, rm: u8) -> u32 {
    0x4A000000 | ((rm as u32) << 16) | ((rn as u32) << 5) | (rd as u32)
}

#[inline]
pub const fn lslv_w(rd: u8, rn: u8, rm: u8) -> u32 {
    0x1AC02000 | ((rm as u32) << 16) | ((rn as u32) << 5) | (rd as u32)
}

#[inline]
pub const fn lsrv_w(rd: u8, rn: u8, rm: u8) -> u32 {
    0x1AC02400 | ((rm as u32) << 16) | ((rn as u32) << 5) | (rd as u32)
}

#[inline]
pub const fn asrv_w(rd: u8, rn: u8, rm: u8) -> u32 {
    0x1AC02800 | ((rm as u32) << 16) | ((rn as u32) << 5) | (rd as u32)
}

#[inline]
pub const fn mov_w_reg(rd: u8, rm: u8) -> u32 {
    orr_w_reg(rd, 31, rm)
}

// ---------------------------------------------------------------------------
// Data processing — immediate
// ---------------------------------------------------------------------------

/// ADD Xd, Xn, #imm12
#[inline]
pub const fn add_x_imm(rd: u8, rn: u8, imm12: u16) -> u32 {
    0x91000000 | ((imm12 as u32 & 0xFFF) << 10) | ((rn as u32) << 5) | (rd as u32)
}

/// SUB Xd, Xn, #imm12
#[inline]
pub const fn sub_x_imm(rd: u8, rn: u8, imm12: u16) -> u32 {
    0xD1000000 | ((imm12 as u32 & 0xFFF) << 10) | ((rn as u32) << 5) | (rd as u32)
}

/// ADD Wd, Wn, #imm12
#[inline]
pub const fn add_w_imm(rd: u8, rn: u8, imm12: u16) -> u32 {
    0x11000000 | ((imm12 as u32 & 0xFFF) << 10) | ((rn as u32) << 5) | (rd as u32)
}

/// SUB Wd, Wn, #imm12
#[inline]
pub const fn sub_w_imm(rd: u8, rn: u8, imm12: u16) -> u32 {
    0x51000000 | ((imm12 as u32 & 0xFFF) << 10) | ((rn as u32) << 5) | (rd as u32)
}

/// MOVZ Xd, #imm16, LSL #(hw*16)
#[inline]
pub const fn movz_x(rd: u8, imm16: u16, hw: u8) -> u32 {
    0xD2800000 | ((hw as u32 & 3) << 21) | ((imm16 as u32) << 5) | (rd as u32)
}

/// MOVK Xd, #imm16, LSL #(hw*16)
#[inline]
pub const fn movk_x(rd: u8, imm16: u16, hw: u8) -> u32 {
    0xF2800000 | ((hw as u32 & 3) << 21) | ((imm16 as u32) << 5) | (rd as u32)
}

/// MOVZ Wd, #imm16, LSL #(hw*16)
#[inline]
pub const fn movz_w(rd: u8, imm16: u16, hw: u8) -> u32 {
    0x52800000 | ((hw as u32 & 1) << 21) | ((imm16 as u32) << 5) | (rd as u32)
}

/// MOVK Wd, #imm16, LSL #(hw*16)
#[inline]
pub const fn movk_w(rd: u8, imm16: u16, hw: u8) -> u32 {
    0x72800000 | ((hw as u32 & 1) << 21) | ((imm16 as u32) << 5) | (rd as u32)
}

// ---------------------------------------------------------------------------
// Shifts (immediate forms via bitfield instructions)
// ---------------------------------------------------------------------------

/// LSL Xd, Xn, #shift (alias: UBFM Xd, Xn, #(64-shift), #(63-shift))
#[inline]
pub const fn lsl_x_imm(rd: u8, rn: u8, shift: u8) -> u32 {
    let immr = (64 - shift) & 63;
    let imms = (63 - shift) & 63;
    0xD3400000 | ((immr as u32) << 16) | ((imms as u32) << 10) | ((rn as u32) << 5) | (rd as u32)
}

/// LSR Xd, Xn, #shift (alias: UBFM Xd, Xn, #shift, #63)
#[inline]
pub const fn lsr_x_imm(rd: u8, rn: u8, shift: u8) -> u32 {
    0xD340FC00 | (((shift & 63) as u32) << 16) | ((rn as u32) << 5) | (rd as u32)
}

/// ASR Xd, Xn, #shift (alias: SBFM Xd, Xn, #shift, #63)
#[inline]
pub const fn asr_x_imm(rd: u8, rn: u8, shift: u8) -> u32 {
    0x9340FC00 | (((shift & 63) as u32) << 16) | ((rn as u32) << 5) | (rd as u32)
}

// ---------------------------------------------------------------------------
// Comparison
// ---------------------------------------------------------------------------

/// CMP Xn, Xm (alias: SUBS XZR, Xn, Xm)
#[inline]
pub const fn cmp_x_reg(rn: u8, rm: u8) -> u32 {
    0xEB000000 | ((rm as u32) << 16) | ((rn as u32) << 5) | 31 // Rd = XZR
}

/// CMP Wn, Wm (alias: SUBS WZR, Wn, Wm)
#[inline]
pub const fn cmp_w_reg(rn: u8, rm: u8) -> u32 {
    0x6B000000 | ((rm as u32) << 16) | ((rn as u32) << 5) | 31
}

// ---------------------------------------------------------------------------
// Branches
// ---------------------------------------------------------------------------

/// B.cond: conditional branch with 19-bit signed offset (in instructions).
#[inline]
pub const fn b_cond(cond: u8, imm19: i32) -> u32 {
    0x54000000 | (((imm19 as u32) & 0x7FFFF) << 5) | (cond as u32)
}

/// B: unconditional branch with 26-bit signed offset (in instructions).
#[inline]
pub const fn b_imm(imm26: i32) -> u32 {
    0x14000000 | ((imm26 as u32) & 0x03FFFFFF)
}

/// BLR Xn: branch with link to register.
#[inline]
pub const fn blr(rn: u8) -> u32 {
    0xD63F0000 | ((rn as u32) << 5)
}

/// RET (alias: RET X30)
#[inline]
pub const fn ret() -> u32 {
    0xD65F03C0
}

// ---------------------------------------------------------------------------
// Load/Store — unsigned offset
// ---------------------------------------------------------------------------

/// LDR Xt, [Xn, #off] (64-bit)
#[inline]
pub const fn ldr_x_imm(rt: u8, rn: u8, off: i16) -> u32 {
    let uoff = ((off as u16) >> 3) as u32; // scaled by 8
    0xF9400000 | (uoff << 10) | ((rn as u32) << 5) | (rt as u32)
}

/// LDR Wt, [Xn, #off] (32-bit)
#[inline]
pub const fn ldr_w_imm(rt: u8, rn: u8, off: i16) -> u32 {
    let uoff = ((off as u16) >> 2) as u32; // scaled by 4
    0xB9400000 | (uoff << 10) | ((rn as u32) << 5) | (rt as u32)
}

/// LDRH Wt, [Xn, #off] (16-bit, zero-extend)
#[inline]
pub const fn ldrh_imm(rt: u8, rn: u8, off: i16) -> u32 {
    let uoff = ((off as u16) >> 1) as u32; // scaled by 2
    0x79400000 | (uoff << 10) | ((rn as u32) << 5) | (rt as u32)
}

/// LDRB Wt, [Xn, #off] (8-bit, zero-extend)
#[inline]
pub const fn ldrb_imm(rt: u8, rn: u8, off: i16) -> u32 {
    let uoff = off as u16 as u32; // no scaling
    0x39400000 | (uoff << 10) | ((rn as u32) << 5) | (rt as u32)
}

/// STR Xt, [Xn, #off] (64-bit)
#[inline]
pub const fn str_x_imm(rt: u8, rn: u8, off: i16) -> u32 {
    let uoff = ((off as u16) >> 3) as u32;
    0xF9000000 | (uoff << 10) | ((rn as u32) << 5) | (rt as u32)
}

/// STR Wt, [Xn, #off] (32-bit)
#[inline]
pub const fn str_w_imm(rt: u8, rn: u8, off: i16) -> u32 {
    let uoff = ((off as u16) >> 2) as u32;
    0xB9000000 | (uoff << 10) | ((rn as u32) << 5) | (rt as u32)
}

/// STRH Wt, [Xn, #off] (16-bit)
#[inline]
pub const fn strh_imm(rt: u8, rn: u8, off: i16) -> u32 {
    let uoff = ((off as u16) >> 1) as u32;
    0x79000000 | (uoff << 10) | ((rn as u32) << 5) | (rt as u32)
}

/// STRB Wt, [Xn, #off] (8-bit)
#[inline]
pub const fn strb_imm(rt: u8, rn: u8, off: i16) -> u32 {
    let uoff = off as u16 as u32;
    0x39000000 | (uoff << 10) | ((rn as u32) << 5) | (rt as u32)
}

// ---------------------------------------------------------------------------
// Load/Store pair (for prologue/epilogue)
// ---------------------------------------------------------------------------

/// STP Xt1, Xt2, [Xn, #off]! (pre-index)
#[inline]
pub const fn stp_pre(rt1: u8, rt2: u8, rn: u8, off: i16) -> u32 {
    let imm7 = ((off / 8) as u32) & 0x7F;
    0xA9800000 | (imm7 << 15) | ((rt2 as u32) << 10) | ((rn as u32) << 5) | (rt1 as u32)
}

/// LDP Xt1, Xt2, [Xn], #off (post-index)
#[inline]
pub const fn ldp_post(rt1: u8, rt2: u8, rn: u8, off: i16) -> u32 {
    let imm7 = ((off / 8) as u32) & 0x7F;
    0xA8C00000 | (imm7 << 15) | ((rt2 as u32) << 10) | ((rn as u32) << 5) | (rt1 as u32)
}

/// STP Xt1, Xt2, [Xn, #off] (signed offset)
#[inline]
pub const fn stp_offset(rt1: u8, rt2: u8, rn: u8, off: i16) -> u32 {
    let imm7 = ((off / 8) as u32) & 0x7F;
    0xA9000000 | (imm7 << 15) | ((rt2 as u32) << 10) | ((rn as u32) << 5) | (rt1 as u32)
}

/// LDP Xt1, Xt2, [Xn, #off] (signed offset)
#[inline]
pub const fn ldp_offset(rt1: u8, rt2: u8, rn: u8, off: i16) -> u32 {
    let imm7 = ((off / 8) as u32) & 0x7F;
    0xA9400000 | (imm7 << 15) | ((rt2 as u32) << 10) | ((rn as u32) << 5) | (rt1 as u32)
}

// ---------------------------------------------------------------------------
// Atomics (ARMv8.1 LSE extensions)
// ---------------------------------------------------------------------------

/// LDADD Xs, Xt, [Xn] — atomic add, acquire+release
#[inline]
pub const fn ldadd_x(rs: u8, rt: u8, rn: u8) -> u32 {
    0xF8E00000 | ((rs as u32) << 16) | ((rn as u32) << 5) | (rt as u32)
}

/// LDSET Xs, Xt, [Xn] — atomic OR
#[inline]
pub const fn ldset_x(rs: u8, rt: u8, rn: u8) -> u32 {
    0xF8E03000 | ((rs as u32) << 16) | ((rn as u32) << 5) | (rt as u32)
}

/// LDCLR Xs, Xt, [Xn] — atomic AND-NOT (clear bits)
#[inline]
pub const fn ldclr_x(rs: u8, rt: u8, rn: u8) -> u32 {
    0xF8E01000 | ((rs as u32) << 16) | ((rn as u32) << 5) | (rt as u32)
}

/// LDEOR Xs, Xt, [Xn] — atomic XOR
#[inline]
pub const fn ldeor_x(rs: u8, rt: u8, rn: u8) -> u32 {
    0xF8E02000 | ((rs as u32) << 16) | ((rn as u32) << 5) | (rt as u32)
}

/// SWP Xs, Xt, [Xn] — atomic swap
#[inline]
pub const fn swp_x(rs: u8, rt: u8, rn: u8) -> u32 {
    0xF8E08000 | ((rs as u32) << 16) | ((rn as u32) << 5) | (rt as u32)
}

/// CAS Xs, Xt, [Xn] — compare-and-swap
#[inline]
pub const fn cas_x(rs: u8, rt: u8, rn: u8) -> u32 {
    0xC8E07C00 | ((rs as u32) << 16) | ((rn as u32) << 5) | (rt as u32)
}

// ---------------------------------------------------------------------------
// Barriers
// ---------------------------------------------------------------------------

/// DMB ISH — data memory barrier, inner shareable
#[inline]
pub const fn dmb_ish() -> u32 {
    0xD5033BBF
}

// ---------------------------------------------------------------------------
// Byte reversal
// ---------------------------------------------------------------------------

/// REV16 Wd, Wn — reverse bytes in each 16-bit halfword
#[inline]
pub const fn rev16_w(rd: u8, rn: u8) -> u32 {
    0x5AC00400 | ((rn as u32) << 5) | (rd as u32)
}

/// REV Wd, Wn — reverse bytes in 32-bit word
#[inline]
pub const fn rev_w(rd: u8, rn: u8) -> u32 {
    0x5AC00800 | ((rn as u32) << 5) | (rd as u32)
}

/// REV Xd, Xn — reverse bytes in 64-bit doubleword
#[inline]
pub const fn rev_x(rd: u8, rn: u8) -> u32 {
    0xDAC00C00 | ((rn as u32) << 5) | (rd as u32)
}

/// AND Xd, Xn, #mask — for 16-bit mask (0xFFFF)
/// Encoded using logical immediate encoding for the specific case of 0xFFFF.
#[inline]
pub const fn and_x_imm_mask(rd: u8, rn: u8, _mask: u16) -> u32 {
    // 0xFFFF: N=1, immr=0, imms=0b001111 (16 bits set)
    0x92400F00 | ((rn as u32) << 5) | (rd as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_add_x0_x1_x2() {
        // ADD X0, X1, X2 → 0x8B020020
        assert_eq!(add_x_reg(0, 1, 2), 0x8B020020);
    }

    #[test]
    fn encode_ret() {
        assert_eq!(ret(), 0xD65F03C0);
    }

    #[test]
    fn encode_movz() {
        // MOVZ X0, #42
        let insn = movz_x(0, 42, 0);
        assert_eq!(insn & 0xFF800000, 0xD2800000); // opcode
        assert_eq!((insn >> 5) & 0xFFFF, 42); // immediate
        assert_eq!(insn & 0x1F, 0); // Rd = X0
    }

    #[test]
    fn encode_b_cond() {
        // B.EQ with offset 0 → should encode condition in low nibble
        let insn = b_cond(CC_EQ, 0);
        assert_eq!(insn & 0xF, 0); // EQ = 0
        assert_eq!(insn & 0xFF000000, 0x54000000); // B.cond opcode
    }

    #[test]
    fn encode_dmb_ish() {
        assert_eq!(dmb_ish(), 0xD5033BBF);
    }
}
