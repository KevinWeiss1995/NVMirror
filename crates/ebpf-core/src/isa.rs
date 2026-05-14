//! eBPF Instruction Set Architecture definitions.
//!
//! All opcodes, register identifiers, and instruction formats are defined
//! here as strongly-typed enums to make invalid encodings unrepresentable.

/// Maximum number of eBPF instructions per program (kernel default: 1M).
pub const MAX_INSNS: usize = 1_000_000;

/// eBPF stack size in bytes (per spec).
pub const STACK_SIZE: usize = 512;

/// Number of eBPF registers (R0..R10).
pub const NUM_REGS: usize = 11;

// ---------------------------------------------------------------------------
// Instruction classes (3-bit field, bits [2:0] of opcode)
// ---------------------------------------------------------------------------

pub const BPF_LD: u8 = 0x00;
pub const BPF_LDX: u8 = 0x01;
pub const BPF_ST: u8 = 0x02;
pub const BPF_STX: u8 = 0x03;
pub const BPF_ALU: u8 = 0x04;
pub const BPF_JMP: u8 = 0x05;
pub const BPF_JMP32: u8 = 0x06;
pub const BPF_ALU64: u8 = 0x07;

// ---------------------------------------------------------------------------
// ALU operation codes (4-bit field, bits [7:4] of opcode)
// ---------------------------------------------------------------------------

pub const BPF_ADD: u8 = 0x00;
pub const BPF_SUB: u8 = 0x10;
pub const BPF_MUL: u8 = 0x20;
pub const BPF_DIV: u8 = 0x30;
pub const BPF_OR: u8 = 0x40;
pub const BPF_AND: u8 = 0x50;
pub const BPF_LSH: u8 = 0x60;
pub const BPF_RSH: u8 = 0x70;
pub const BPF_NEG: u8 = 0x80;
pub const BPF_MOD: u8 = 0x90;
pub const BPF_XOR: u8 = 0xa0;
pub const BPF_MOV: u8 = 0xb0;
pub const BPF_ARSH: u8 = 0xc0;
pub const BPF_END: u8 = 0xd0;

// ---------------------------------------------------------------------------
// Jump operation codes (4-bit field, bits [7:4] of opcode)
// ---------------------------------------------------------------------------

pub const BPF_JA: u8 = 0x00;
pub const BPF_JEQ: u8 = 0x10;
pub const BPF_JGT: u8 = 0x20;
pub const BPF_JGE: u8 = 0x30;
pub const BPF_JSET: u8 = 0x40;
pub const BPF_JNE: u8 = 0x50;
pub const BPF_JSGT: u8 = 0x60;
pub const BPF_JSGE: u8 = 0x70;
pub const BPF_CALL: u8 = 0x80;
pub const BPF_EXIT: u8 = 0x90;
pub const BPF_JLT: u8 = 0xa0;
pub const BPF_JLE: u8 = 0xb0;
pub const BPF_JSLT: u8 = 0xc0;
pub const BPF_JSLE: u8 = 0xd0;

// ---------------------------------------------------------------------------
// Source modifier (1-bit, bit [3] of opcode)
// ---------------------------------------------------------------------------

pub const BPF_K: u8 = 0x00;
pub const BPF_X: u8 = 0x08;

// ---------------------------------------------------------------------------
// Memory access size (2-bit field, bits [4:3] of opcode for LD/ST)
// ---------------------------------------------------------------------------

pub const BPF_W: u8 = 0x00;   // 32-bit
pub const BPF_H: u8 = 0x08;   // 16-bit
pub const BPF_B: u8 = 0x10;   // 8-bit
pub const BPF_DW: u8 = 0x18;  // 64-bit

// ---------------------------------------------------------------------------
// Memory access mode (3-bit field, bits [7:5] of opcode for LD/ST)
// ---------------------------------------------------------------------------

pub const BPF_IMM: u8 = 0x00;
pub const BPF_ABS: u8 = 0x20;
pub const BPF_IND: u8 = 0x40;
pub const BPF_MEM: u8 = 0x60;
pub const BPF_ATOMIC: u8 = 0xc0;

// ---------------------------------------------------------------------------
// Atomic operations (encoded in imm field when BPF_ATOMIC)
// ---------------------------------------------------------------------------

pub const BPF_ATOMIC_ADD: u32 = 0x00;
pub const BPF_ATOMIC_OR: u32 = 0x40;
pub const BPF_ATOMIC_AND: u32 = 0x50;
pub const BPF_ATOMIC_XOR: u32 = 0xa0;
pub const BPF_ATOMIC_XCHG: u32 = 0xe0 | 0x01;
pub const BPF_ATOMIC_CMPXCHG: u32 = 0xf0 | 0x01;
pub const BPF_ATOMIC_FETCH: u32 = 0x01;

// ---------------------------------------------------------------------------
// Endianness for BPF_END
// ---------------------------------------------------------------------------

pub const BPF_TO_LE: u8 = 0x00;
pub const BPF_TO_BE: u8 = 0x08;

/// Bounded eBPF register identifier: R0..R10.
///
/// Using a newtype with a validity check makes out-of-range register
/// references unrepresentable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BpfReg(u8);

impl BpfReg {
    pub const R0: Self = Self(0);
    pub const R1: Self = Self(1);
    pub const R2: Self = Self(2);
    pub const R3: Self = Self(3);
    pub const R4: Self = Self(4);
    pub const R5: Self = Self(5);
    pub const R6: Self = Self(6);
    pub const R7: Self = Self(7);
    pub const R8: Self = Self(8);
    pub const R9: Self = Self(9);
    pub const R10: Self = Self(10); // Frame pointer (read-only)

    /// Construct from a raw 4-bit register field.
    /// Returns `None` if `raw > 10`.
    #[inline]
    pub const fn from_raw(raw: u8) -> Option<Self> {
        if raw <= 10 {
            Some(Self(raw))
        } else {
            None
        }
    }

    #[inline]
    pub const fn raw(self) -> u8 {
        self.0
    }

    #[inline]
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// Memory access width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemWidth {
    B,  // 8-bit
    H,  // 16-bit
    W,  // 32-bit
    DW, // 64-bit
}

impl MemWidth {
    pub const fn from_size_code(code: u8) -> Option<Self> {
        match code & 0x18 {
            BPF_B => Some(Self::B),
            BPF_H => Some(Self::H),
            BPF_W => Some(Self::W),
            BPF_DW => Some(Self::DW),
            _ => None,
        }
    }

    pub const fn byte_len(self) -> u8 {
        match self {
            Self::B => 1,
            Self::H => 2,
            Self::W => 4,
            Self::DW => 8,
        }
    }
}

/// Source operand: either an immediate constant or a register.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    Imm(i64),
    Reg(BpfReg),
}

/// ALU operation kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AluOp {
    Add,
    Sub,
    Mul,
    Div,
    Or,
    And,
    Lsh,
    Rsh,
    Neg,
    Mod,
    Xor,
    Mov,
    Arsh,
}

impl AluOp {
    pub const fn from_opcode_bits(bits: u8) -> Option<Self> {
        match bits & 0xf0 {
            BPF_ADD => Some(Self::Add),
            BPF_SUB => Some(Self::Sub),
            BPF_MUL => Some(Self::Mul),
            BPF_DIV => Some(Self::Div),
            BPF_OR => Some(Self::Or),
            BPF_AND => Some(Self::And),
            BPF_LSH => Some(Self::Lsh),
            BPF_RSH => Some(Self::Rsh),
            BPF_NEG => Some(Self::Neg),
            BPF_MOD => Some(Self::Mod),
            BPF_XOR => Some(Self::Xor),
            BPF_MOV => Some(Self::Mov),
            BPF_ARSH => Some(Self::Arsh),
            _ => None,
        }
    }
}

/// Branch condition kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JmpOp {
    Ja,    // unconditional
    Jeq,
    Jgt,
    Jge,
    Jset,
    Jne,
    Jsgt,
    Jsge,
    Jlt,
    Jle,
    Jslt,
    Jsle,
}

impl JmpOp {
    pub const fn from_opcode_bits(bits: u8) -> Option<Self> {
        match bits & 0xf0 {
            BPF_JA => Some(Self::Ja),
            BPF_JEQ => Some(Self::Jeq),
            BPF_JGT => Some(Self::Jgt),
            BPF_JGE => Some(Self::Jge),
            BPF_JSET => Some(Self::Jset),
            BPF_JNE => Some(Self::Jne),
            BPF_JSGT => Some(Self::Jsgt),
            BPF_JSGE => Some(Self::Jsge),
            BPF_JLT => Some(Self::Jlt),
            BPF_JLE => Some(Self::Jle),
            BPF_JSLT => Some(Self::Jslt),
            BPF_JSLE => Some(Self::Jsle),
            _ => None,
        }
    }
}

/// Atomic memory operation kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtomicOp {
    Add,
    Or,
    And,
    Xor,
    Xchg,
    CmpXchg,
    FetchAdd,
    FetchOr,
    FetchAnd,
    FetchXor,
}

/// Endianness conversion width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndianWidth {
    Bits16,
    Bits32,
    Bits64,
}

/// A single decoded eBPF instruction in typed form.
///
/// This enum makes invalid instruction encodings unrepresentable.
/// Every variant carries exactly the operands required by the eBPF spec.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Insn {
    /// 64-bit ALU: dst = dst OP src
    Alu64 { op: AluOp, dst: BpfReg, src: Source },

    /// 32-bit ALU: dst = (u32)(dst OP src)
    Alu32 { op: AluOp, dst: BpfReg, src: Source },

    /// Load from memory: dst = *(width*)(src + off)
    Load { width: MemWidth, dst: BpfReg, src: BpfReg, off: i16 },

    /// Store register to memory: *(width*)(dst + off) = src
    StoreReg { width: MemWidth, dst: BpfReg, src: BpfReg, off: i16 },

    /// Store immediate to memory: *(width*)(dst + off) = imm
    StoreImm { width: MemWidth, dst: BpfReg, off: i16, imm: i32 },

    /// Load 64-bit immediate (wide instruction, consumes 2 slots).
    LoadImm64 { dst: BpfReg, imm: u64 },

    /// Conditional jump: if (dst OP src) goto pc + off
    JmpCond { op: JmpOp, dst: BpfReg, src: Source, off: i16 },

    /// Conditional jump (32-bit comparison): if ((u32)dst OP (u32)src) goto pc + off
    JmpCond32 { op: JmpOp, dst: BpfReg, src: Source, off: i16 },

    /// Unconditional jump: goto pc + off
    Ja { off: i16 },

    /// Call helper function by ID.
    Call { func_id: u32 },

    /// Tail call (program-to-program).
    TailCall,

    /// Exit program, return value in R0.
    Exit,

    /// Atomic memory operation.
    Atomic { width: MemWidth, op: AtomicOp, dst: BpfReg, src: BpfReg, off: i16 },

    /// Byte-swap endianness.
    Endian { to_be: bool, width: EndianWidth, dst: BpfReg },
}

/// Raw 64-bit eBPF instruction encoding (network/wire format).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct RawInsn {
    pub opcode: u8,
    pub regs: u8,   // dst_reg:4 | src_reg:4
    pub off: i16,
    pub imm: i32,
}

impl RawInsn {
    #[inline]
    pub const fn dst_reg(&self) -> u8 {
        self.regs & 0x0f
    }

    #[inline]
    pub const fn src_reg(&self) -> u8 {
        (self.regs >> 4) & 0x0f
    }

    #[inline]
    pub const fn insn_class(&self) -> u8 {
        self.opcode & 0x07
    }

    /// Deserialize from a little-endian 8-byte slice.
    pub fn from_le_bytes(bytes: &[u8; 8]) -> Self {
        Self {
            opcode: bytes[0],
            regs: bytes[1],
            off: i16::from_le_bytes([bytes[2], bytes[3]]),
            imm: i32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
        }
    }
}
