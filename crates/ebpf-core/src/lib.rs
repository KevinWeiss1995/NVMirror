//! # ebpf-core
//!
//! eBPF ISA definitions, bytecode decoder, and static verifier.
//!
//! This crate is `no_std` by default for firmware portability.
//! Enable the `std` feature for userspace testing.

#![no_std]
#![forbid(unsafe_code)]

extern crate alloc;

pub mod isa;
pub mod decode;
pub mod verify;

pub use isa::*;
pub use decode::{decode_program, DecodeError};
pub use verify::{verify_program, VerifyError};
