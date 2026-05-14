//! RISC-V RV64 instruction encoding primitives.
//!
//! Each function returns a 32-bit little-endian encoded RISC-V instruction.
//! Register arguments are raw register numbers (0-31, where 0 = x0/zero).
//!
//! Encoding formats follow the RISC-V Unprivileged ISA spec:
//! - R-type: funct7[6:0] | rs2[4:0] | rs1[4:0] | funct3[2:0] | rd[4:0] | opcode[6:0]
//! - I-type: imm[11:0] | rs1[4:0] | funct3[2:0] | rd[4:0] | opcode[6:0]
//! - S-type: imm[11:5] | rs2[4:0] | rs1[4:0] | funct3[2:0] | imm[4:0] | opcode[6:0]
//! - B-type: imm[12|10:5] | rs2[4:0] | rs1[4:0] | funct3[2:0] | imm[4:1|11] | opcode[6:0]
//! - U-type: imm[31:12] | rd[4:0] | opcode[6:0]
//! - J-type: imm[20|10:1|11|19:12] | rd[4:0] | opcode[6:0]

/// Sign-extend a 12-bit value.
#[inline]
pub const fn sign_extend_12(val: i32) -> i32 {
    ((val & 0xFFF) << 20) >> 20
}

// ---------------------------------------------------------------------------
// R-type: arithmetic register-register
// ---------------------------------------------------------------------------

fn r_type(funct7: u32, rs2: u8, rs1: u8, funct3: u32, rd: u8, opcode: u32) -> u32 {
    (funct7 << 25)
        | ((rs2 as u32) << 20)
        | ((rs1 as u32) << 15)
        | (funct3 << 12)
        | ((rd as u32) << 7)
        | opcode
}

/// ADD rd, rs1, rs2
pub fn add(rd: u8, rs1: u8, rs2: u8) -> u32 { r_type(0, rs2, rs1, 0, rd, 0x33) }

/// SUB rd, rs1, rs2
pub fn sub(rd: u8, rs1: u8, rs2: u8) -> u32 { r_type(0x20, rs2, rs1, 0, rd, 0x33) }

/// SLL rd, rs1, rs2
pub fn sll(rd: u8, rs1: u8, rs2: u8) -> u32 { r_type(0, rs2, rs1, 1, rd, 0x33) }

/// SRL rd, rs1, rs2
pub fn srl(rd: u8, rs1: u8, rs2: u8) -> u32 { r_type(0, rs2, rs1, 5, rd, 0x33) }

/// SRA rd, rs1, rs2
pub fn sra(rd: u8, rs1: u8, rs2: u8) -> u32 { r_type(0x20, rs2, rs1, 5, rd, 0x33) }

/// OR rd, rs1, rs2
pub fn or(rd: u8, rs1: u8, rs2: u8) -> u32 { r_type(0, rs2, rs1, 6, rd, 0x33) }

/// AND rd, rs1, rs2
pub fn and(rd: u8, rs1: u8, rs2: u8) -> u32 { r_type(0, rs2, rs1, 7, rd, 0x33) }

/// XOR rd, rs1, rs2
pub fn xor(rd: u8, rs1: u8, rs2: u8) -> u32 { r_type(0, rs2, rs1, 4, rd, 0x33) }

// M-extension (multiply/divide)

/// MUL rd, rs1, rs2
pub fn mul(rd: u8, rs1: u8, rs2: u8) -> u32 { r_type(1, rs2, rs1, 0, rd, 0x33) }

/// DIVU rd, rs1, rs2
pub fn divu(rd: u8, rs1: u8, rs2: u8) -> u32 { r_type(1, rs2, rs1, 5, rd, 0x33) }

/// REMU rd, rs1, rs2
pub fn remu(rd: u8, rs1: u8, rs2: u8) -> u32 { r_type(1, rs2, rs1, 7, rd, 0x33) }

// RV64 word-width (32-bit) ops — opcode 0x3B

/// ADDW rd, rs1, rs2
pub fn addw(rd: u8, rs1: u8, rs2: u8) -> u32 { r_type(0, rs2, rs1, 0, rd, 0x3B) }

/// SUBW rd, rs1, rs2
pub fn subw(rd: u8, rs1: u8, rs2: u8) -> u32 { r_type(0x20, rs2, rs1, 0, rd, 0x3B) }

/// SLLW rd, rs1, rs2
pub fn sllw(rd: u8, rs1: u8, rs2: u8) -> u32 { r_type(0, rs2, rs1, 1, rd, 0x3B) }

/// SRLW rd, rs1, rs2
pub fn srlw(rd: u8, rs1: u8, rs2: u8) -> u32 { r_type(0, rs2, rs1, 5, rd, 0x3B) }

/// SRAW rd, rs1, rs2
pub fn sraw(rd: u8, rs1: u8, rs2: u8) -> u32 { r_type(0x20, rs2, rs1, 5, rd, 0x3B) }

/// MULW rd, rs1, rs2
pub fn mulw(rd: u8, rs1: u8, rs2: u8) -> u32 { r_type(1, rs2, rs1, 0, rd, 0x3B) }

/// DIVUW rd, rs1, rs2
pub fn divuw(rd: u8, rs1: u8, rs2: u8) -> u32 { r_type(1, rs2, rs1, 5, rd, 0x3B) }

/// REMUW rd, rs1, rs2
pub fn remuw(rd: u8, rs1: u8, rs2: u8) -> u32 { r_type(1, rs2, rs1, 7, rd, 0x3B) }

// ---------------------------------------------------------------------------
// I-type: arithmetic immediate, loads, JALR
// ---------------------------------------------------------------------------

fn i_type(imm: i32, rs1: u8, funct3: u32, rd: u8, opcode: u32) -> u32 {
    (((imm & 0xFFF) as u32) << 20)
        | ((rs1 as u32) << 15)
        | (funct3 << 12)
        | ((rd as u32) << 7)
        | opcode
}

/// ADDI rd, rs1, imm
pub fn addi(rd: u8, rs1: u8, imm: i32) -> u32 { i_type(imm, rs1, 0, rd, 0x13) }

/// ANDI rd, rs1, imm
pub fn andi(rd: u8, rs1: u8, imm: i32) -> u32 { i_type(imm, rs1, 7, rd, 0x13) }

/// ORI rd, rs1, imm
pub fn ori(rd: u8, rs1: u8, imm: i32) -> u32 { i_type(imm, rs1, 6, rd, 0x13) }

/// XORI rd, rs1, imm
pub fn xori(rd: u8, rs1: u8, imm: i32) -> u32 { i_type(imm, rs1, 4, rd, 0x13) }

/// SLLI rd, rs1, shamt (RV64: 6-bit shift amount)
pub fn slli(rd: u8, rs1: u8, shamt: u32) -> u32 {
    ((shamt & 0x3F) << 20) | ((rs1 as u32) << 15) | (1 << 12) | ((rd as u32) << 7) | 0x13
}

/// SRLI rd, rs1, shamt
pub fn srli(rd: u8, rs1: u8, shamt: u32) -> u32 {
    ((shamt & 0x3F) << 20) | ((rs1 as u32) << 15) | (5 << 12) | ((rd as u32) << 7) | 0x13
}

/// SRAI rd, rs1, shamt
pub fn srai(rd: u8, rs1: u8, shamt: u32) -> u32 {
    (0x400 << 20) | ((shamt & 0x3F) << 20) | ((rs1 as u32) << 15) | (5 << 12) | ((rd as u32) << 7) | 0x13
}

/// JALR rd, rs1, imm
pub fn jalr(rd: u8, rs1: u8, imm: i32) -> u32 { i_type(imm, rs1, 0, rd, 0x67) }

// Loads

/// LB rd, imm(rs1) — load byte, sign-extend
pub fn lb(rd: u8, rs1: u8, imm: i32) -> u32 { i_type(imm, rs1, 0, rd, 0x03) }

/// LBU rd, imm(rs1) — load byte, zero-extend
pub fn lbu(rd: u8, rs1: u8, imm: i32) -> u32 { i_type(imm, rs1, 4, rd, 0x03) }

/// LH rd, imm(rs1) — load halfword, sign-extend
pub fn lh(rd: u8, rs1: u8, imm: i32) -> u32 { i_type(imm, rs1, 1, rd, 0x03) }

/// LHU rd, imm(rs1) — load halfword, zero-extend
pub fn lhu(rd: u8, rs1: u8, imm: i32) -> u32 { i_type(imm, rs1, 5, rd, 0x03) }

/// LW rd, imm(rs1) — load word, sign-extend
pub fn lw(rd: u8, rs1: u8, imm: i32) -> u32 { i_type(imm, rs1, 2, rd, 0x03) }

/// LWU rd, imm(rs1) — load word, zero-extend (RV64)
pub fn lwu(rd: u8, rs1: u8, imm: i32) -> u32 { i_type(imm, rs1, 6, rd, 0x03) }

/// LD rd, imm(rs1) — load doubleword (RV64)
pub fn ld(rd: u8, rs1: u8, imm: i32) -> u32 { i_type(imm, rs1, 3, rd, 0x03) }

// ---------------------------------------------------------------------------
// S-type: stores
// ---------------------------------------------------------------------------

fn s_type(imm: i32, rs2: u8, rs1: u8, funct3: u32, opcode: u32) -> u32 {
    let imm = imm & 0xFFF;
    let imm_hi = ((imm >> 5) & 0x7F) as u32;
    let imm_lo = (imm & 0x1F) as u32;
    (imm_hi << 25)
        | ((rs2 as u32) << 20)
        | ((rs1 as u32) << 15)
        | (funct3 << 12)
        | (imm_lo << 7)
        | opcode
}

/// SB rs2, imm(rs1) — store byte
pub fn sb(rs2: u8, rs1: u8, imm: i32) -> u32 { s_type(imm, rs2, rs1, 0, 0x23) }

/// SH rs2, imm(rs1) — store halfword
pub fn sh(rs2: u8, rs1: u8, imm: i32) -> u32 { s_type(imm, rs2, rs1, 1, 0x23) }

/// SW rs2, imm(rs1) — store word
pub fn sw(rs2: u8, rs1: u8, imm: i32) -> u32 { s_type(imm, rs2, rs1, 2, 0x23) }

/// SD rs2, imm(rs1) — store doubleword (RV64)
pub fn sd(rs2: u8, rs1: u8, imm: i32) -> u32 { s_type(imm, rs2, rs1, 3, 0x23) }

// ---------------------------------------------------------------------------
// B-type: branches
// ---------------------------------------------------------------------------

/// Encode B-type immediate field (12-bit signed, byte-addressed).
pub fn encode_b_type_imm(imm: i32) -> u32 {
    let imm = imm & 0x1FFE; // bits [12:1], bit 0 always 0
    let bit12 = ((imm >> 12) & 1) as u32;
    let bit11 = ((imm >> 11) & 1) as u32;
    let bits10_5 = ((imm >> 5) & 0x3F) as u32;
    let bits4_1 = ((imm >> 1) & 0xF) as u32;
    (bit12 << 31) | (bits10_5 << 25) | (bits4_1 << 8) | (bit11 << 7)
}

/// BEQ rs1, rs2, offset
pub fn beq(rs1: u8, rs2: u8, imm: i32) -> u32 {
    encode_b_type_imm(imm) | ((rs2 as u32) << 20) | ((rs1 as u32) << 15) | (0 << 12) | 0x63
}

/// BNE rs1, rs2, offset
pub fn bne(rs1: u8, rs2: u8, imm: i32) -> u32 {
    encode_b_type_imm(imm) | ((rs2 as u32) << 20) | ((rs1 as u32) << 15) | (1 << 12) | 0x63
}

/// BLT rs1, rs2, offset (signed)
pub fn blt(rs1: u8, rs2: u8, imm: i32) -> u32 {
    encode_b_type_imm(imm) | ((rs2 as u32) << 20) | ((rs1 as u32) << 15) | (4 << 12) | 0x63
}

/// BGE rs1, rs2, offset (signed)
pub fn bge(rs1: u8, rs2: u8, imm: i32) -> u32 {
    encode_b_type_imm(imm) | ((rs2 as u32) << 20) | ((rs1 as u32) << 15) | (5 << 12) | 0x63
}

/// BLTU rs1, rs2, offset (unsigned)
pub fn bltu(rs1: u8, rs2: u8, imm: i32) -> u32 {
    encode_b_type_imm(imm) | ((rs2 as u32) << 20) | ((rs1 as u32) << 15) | (6 << 12) | 0x63
}

/// BGEU rs1, rs2, offset (unsigned)
pub fn bgeu(rs1: u8, rs2: u8, imm: i32) -> u32 {
    encode_b_type_imm(imm) | ((rs2 as u32) << 20) | ((rs1 as u32) << 15) | (7 << 12) | 0x63
}

// ---------------------------------------------------------------------------
// U-type: LUI, AUIPC
// ---------------------------------------------------------------------------

/// LUI rd, imm — load upper immediate (bits [31:12])
pub fn lui(rd: u8, imm: i32) -> u32 {
    ((imm as u32) & 0xFFFFF000) | ((rd as u32) << 7) | 0x37
}

// ---------------------------------------------------------------------------
// J-type: JAL
// ---------------------------------------------------------------------------

/// Encode J-type immediate field (20-bit signed, byte-addressed).
pub fn encode_j_type_imm(imm: i32) -> u32 {
    let bit20 = ((imm >> 20) & 1) as u32;
    let bits10_1 = ((imm >> 1) & 0x3FF) as u32;
    let bit11 = ((imm >> 11) & 1) as u32;
    let bits19_12 = ((imm >> 12) & 0xFF) as u32;
    (bit20 << 31) | (bits10_1 << 21) | (bit11 << 20) | (bits19_12 << 12)
}

/// JAL rd, offset
pub fn jal(rd: u8, imm: i32) -> u32 {
    encode_j_type_imm(imm) | ((rd as u32) << 7) | 0x6F
}

// ---------------------------------------------------------------------------
// Atomics (A-extension)
// ---------------------------------------------------------------------------

fn amo(funct5: u32, aq: bool, rl: bool, rs2: u8, rs1: u8, funct3: u32, rd: u8) -> u32 {
    let aq_bit = if aq { 1u32 } else { 0 };
    let rl_bit = if rl { 1u32 } else { 0 };
    (funct5 << 27)
        | (aq_bit << 26)
        | (rl_bit << 25)
        | ((rs2 as u32) << 20)
        | ((rs1 as u32) << 15)
        | (funct3 << 12)
        | ((rd as u32) << 7)
        | 0x2F
}

/// AMOADD.D rd, rs2, (rs1) — atomic add doubleword
pub fn amoadd_d(rd: u8, rs2: u8, rs1: u8) -> u32 { amo(0x00, true, true, rs2, rs1, 3, rd) }

/// AMOOR.D rd, rs2, (rs1)
pub fn amoor_d(rd: u8, rs2: u8, rs1: u8) -> u32 { amo(0x08, true, true, rs2, rs1, 3, rd) }

/// AMOAND.D rd, rs2, (rs1)
pub fn amoand_d(rd: u8, rs2: u8, rs1: u8) -> u32 { amo(0x0C, true, true, rs2, rs1, 3, rd) }

/// AMOXOR.D rd, rs2, (rs1)
pub fn amoxor_d(rd: u8, rs2: u8, rs1: u8) -> u32 { amo(0x04, true, true, rs2, rs1, 3, rd) }

/// AMOSWAP.D rd, rs2, (rs1)
pub fn amoswap_d(rd: u8, rs2: u8, rs1: u8) -> u32 { amo(0x01, true, true, rs2, rs1, 3, rd) }

/// LR.D rd, (rs1) — load-reserved doubleword
pub fn lr_d(rd: u8, rs1: u8) -> u32 { amo(0x02, true, true, 0, rs1, 3, rd) }

/// SC.D rd, rs2, (rs1) — store-conditional doubleword
pub fn sc_d(rd: u8, rs2: u8, rs1: u8) -> u32 { amo(0x03, true, true, rs2, rs1, 3, rd) }

// ---------------------------------------------------------------------------
// Fences
// ---------------------------------------------------------------------------

/// FENCE rw, rw — full memory fence
pub fn fence_rw_rw() -> u32 {
    // pred=rw (0b0011), succ=rw (0b0011)
    0x0330000F
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_add_x1_x2_x3() {
        // ADD x1, x2, x3 should encode as R-type
        let insn = add(1, 2, 3);
        assert_eq!(insn & 0x7F, 0x33); // opcode = OP
        assert_eq!((insn >> 7) & 0x1F, 1); // rd = x1
        assert_eq!((insn >> 15) & 0x1F, 2); // rs1 = x2
        assert_eq!((insn >> 20) & 0x1F, 3); // rs2 = x3
        assert_eq!((insn >> 25) & 0x7F, 0); // funct7 = 0
    }

    #[test]
    fn encode_addi_x1_x0_42() {
        let insn = addi(1, 0, 42);
        assert_eq!(insn & 0x7F, 0x13); // opcode = OP-IMM
        assert_eq!((insn >> 7) & 0x1F, 1); // rd = x1
        assert_eq!((insn >> 15) & 0x1F, 0); // rs1 = x0
        assert_eq!((insn >> 20) & 0xFFF, 42); // imm = 42
    }

    #[test]
    fn encode_sd() {
        let insn = sd(1, 2, 8);
        assert_eq!(insn & 0x7F, 0x23); // opcode = STORE
        // funct3 = 3 (doubleword)
        assert_eq!((insn >> 12) & 0x7, 3);
    }

    #[test]
    fn encode_beq() {
        let insn = beq(1, 2, 0);
        assert_eq!(insn & 0x7F, 0x63); // opcode = BRANCH
        assert_eq!((insn >> 12) & 0x7, 0); // funct3 = BEQ
    }

    #[test]
    fn encode_jal_zero_offset() {
        let insn = jal(1, 0);
        assert_eq!(insn & 0x7F, 0x6F); // opcode = JAL
        assert_eq!((insn >> 7) & 0x1F, 1); // rd = x1
    }

    #[test]
    fn encode_fence() {
        assert_eq!(fence_rw_rw(), 0x0330000F);
    }
}
