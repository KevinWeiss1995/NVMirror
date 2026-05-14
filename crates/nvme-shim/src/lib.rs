//! # nvme-shim
//!
//! NVMe controller memory model, DMA stubs, and W^X buffer management.
//!
//! In firmware mode (`no_std`), this provides abstractions over the
//! controller's SRAM regions, MPU configuration, and DMA engines.
//!
//! In userspace mode (`std` + `userspace` features), it provides a
//! `mmap`-based executable buffer for testing on macOS/Linux.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "std")]
extern crate alloc;

/// Maximum JIT buffer size (64KB).
pub const MAX_JIT_BUFFER_SIZE: usize = 64 * 1024;

/// Guard page size (4KB).
pub const GUARD_PAGE_SIZE: usize = 4096;

/// Memory region descriptor for the NVMe controller.
#[derive(Debug, Clone)]
pub struct MemoryRegion {
    pub base: usize,
    pub size: usize,
    pub executable: bool,
    pub writable: bool,
}

/// Helper function table — maps eBPF helper IDs to function pointers.
///
/// In firmware, these are resolved at link time to NVMe-specific helpers
/// (e.g., DMA transfer, completion queue manipulation).
/// In userspace testing, they're stubs that record calls.
#[derive(Debug, Clone)]
pub struct HelperTable {
    entries: [Option<usize>; 256],
}

impl HelperTable {
    pub const fn new() -> Self {
        Self { entries: [None; 256] }
    }

    pub fn register(&mut self, id: u32, addr: usize) {
        if (id as usize) < self.entries.len() {
            self.entries[id as usize] = Some(addr);
        }
    }

    pub fn resolve(&self, id: u32) -> Option<usize> {
        self.entries.get(id as usize).copied().flatten()
    }
}

/// Executable buffer abstraction.
///
/// On firmware: wraps a pointer to SRAM with W^X enforcement.
/// On userspace: wraps an mmap'd region with platform-specific JIT support.
#[cfg(feature = "userspace")]
pub mod exec_buffer {
    use super::*;

    /// Platform-specific executable buffer for JIT'd code.
    pub struct ExecBuffer {
        ptr: *mut u8,
        size: usize,
        code_len: usize,
    }

    impl ExecBuffer {
        /// Allocate a new executable buffer.
        ///
        /// On macOS (Apple Silicon), uses MAP_JIT and
        /// `pthread_jit_write_protect_np` for W^X compliance.
        ///
        /// O(1) — single mmap syscall.
        pub fn new(size: usize) -> Result<Self, &'static str> {
            let size = if size > MAX_JIT_BUFFER_SIZE {
                return Err("JIT buffer exceeds maximum size");
            } else {
                // Round up to page boundary
                (size + GUARD_PAGE_SIZE - 1) & !(GUARD_PAGE_SIZE - 1)
            };

            let ptr = unsafe { alloc_executable(size)? };

            Ok(Self { ptr, size, code_len: 0 })
        }

        /// Write JIT'd code into the buffer.
        /// Buffer must be in writable mode.
        pub fn write_code(&mut self, code: &[u8]) -> Result<(), &'static str> {
            if code.len() > self.size {
                return Err("Code exceeds buffer capacity");
            }

            unsafe {
                set_writable(true);
                core::ptr::copy_nonoverlapping(code.as_ptr(), self.ptr, code.len());
                invalidate_icache(self.ptr, code.len());
                set_writable(false);
            }

            self.code_len = code.len();
            Ok(())
        }

        /// Get a function pointer to the JIT'd code.
        ///
        /// # Safety
        /// The caller must ensure the buffer contains valid machine code
        /// for the current architecture.
        pub unsafe fn as_fn_ptr<F>(&self) -> F
        where
            F: Copy,
        {
            unsafe { core::mem::transmute_copy(&self.ptr) }
        }

        pub fn code_len(&self) -> usize {
            self.code_len
        }

        pub fn capacity(&self) -> usize {
            self.size
        }

        pub fn as_ptr(&self) -> *const u8 {
            self.ptr
        }
    }

    impl Drop for ExecBuffer {
        fn drop(&mut self) {
            unsafe {
                dealloc_executable(self.ptr, self.size);
            }
        }
    }

    // -- Platform-specific allocation --

    #[cfg(target_os = "macos")]
    unsafe fn alloc_executable(size: usize) -> Result<*mut u8, &'static str> {
        use core::ptr;

        extern "C" {
            fn mmap(
                addr: *mut u8,
                len: usize,
                prot: i32,
                flags: i32,
                fd: i32,
                offset: i64,
            ) -> *mut u8;
        }

        const PROT_READ: i32 = 0x01;
        const PROT_WRITE: i32 = 0x02;
        const PROT_EXEC: i32 = 0x04;
        const MAP_PRIVATE: i32 = 0x02;
        const MAP_ANONYMOUS: i32 = 0x1000;
        const MAP_JIT: i32 = 0x0800;
        const MAP_FAILED: *mut u8 = !0usize as *mut u8;

        let ptr = unsafe {
            mmap(
                ptr::null_mut(),
                size,
                PROT_READ | PROT_WRITE | PROT_EXEC,
                MAP_PRIVATE | MAP_ANONYMOUS | MAP_JIT,
                -1,
                0,
            )
        };

        if ptr == MAP_FAILED {
            Err("mmap failed")
        } else {
            Ok(ptr)
        }
    }

    #[cfg(target_os = "linux")]
    unsafe fn alloc_executable(size: usize) -> Result<*mut u8, &'static str> {
        use core::ptr;

        extern "C" {
            fn mmap(
                addr: *mut u8,
                len: usize,
                prot: i32,
                flags: i32,
                fd: i32,
                offset: i64,
            ) -> *mut u8;
        }

        const PROT_READ: i32 = 0x01;
        const PROT_WRITE: i32 = 0x02;
        const PROT_EXEC: i32 = 0x04;
        const MAP_PRIVATE: i32 = 0x02;
        const MAP_ANONYMOUS: i32 = 0x20;
        const MAP_FAILED: *mut u8 = !0usize as *mut u8;

        let ptr = unsafe {
            mmap(
                ptr::null_mut(),
                size,
                PROT_READ | PROT_WRITE | PROT_EXEC,
                MAP_PRIVATE | MAP_ANONYMOUS,
                -1,
                0,
            )
        };

        if ptr == MAP_FAILED {
            Err("mmap failed")
        } else {
            Ok(ptr)
        }
    }

    #[cfg(target_os = "macos")]
    unsafe fn invalidate_icache(ptr: *mut u8, len: usize) {
        extern "C" {
            fn sys_icache_invalidate(start: *mut u8, len: usize);
        }
        unsafe { sys_icache_invalidate(ptr, len) };
    }

    #[cfg(target_os = "linux")]
    unsafe fn invalidate_icache(ptr: *mut u8, len: usize) {
        extern "C" {
            fn __clear_cache(start: *mut u8, end: *mut u8);
        }
        unsafe { __clear_cache(ptr, ptr.add(len)) };
    }

    #[cfg(target_os = "macos")]
    unsafe fn set_writable(writable: bool) {
        extern "C" {
            fn pthread_jit_write_protect_np(enabled: i32);
        }
        // On macOS, 0 = writable, 1 = executable
        unsafe { pthread_jit_write_protect_np(if writable { 0 } else { 1 }) };
    }

    #[cfg(target_os = "linux")]
    unsafe fn set_writable(_writable: bool) {
        // On Linux, RWX pages don't need toggling (no hardware W^X enforcement).
        // For production firmware, we'd use mprotect here.
    }

    #[cfg(target_os = "macos")]
    unsafe fn dealloc_executable(ptr: *mut u8, size: usize) {
        extern "C" {
            fn munmap(addr: *mut u8, len: usize) -> i32;
        }
        unsafe { munmap(ptr, size) };
    }

    #[cfg(target_os = "linux")]
    unsafe fn dealloc_executable(ptr: *mut u8, size: usize) {
        extern "C" {
            fn munmap(addr: *mut u8, len: usize) -> i32;
        }
        unsafe { munmap(ptr, size) };
    }
}

/// Firmware-mode (no_std) buffer — just a static array.
#[cfg(not(feature = "userspace"))]
pub mod exec_buffer {
    use super::MAX_JIT_BUFFER_SIZE;

    pub struct ExecBuffer {
        buf: [u8; MAX_JIT_BUFFER_SIZE],
        code_len: usize,
    }

    impl ExecBuffer {
        pub fn new(_size: usize) -> Result<Self, &'static str> {
            Ok(Self {
                buf: [0u8; MAX_JIT_BUFFER_SIZE],
                code_len: 0,
            })
        }

        pub fn write_code(&mut self, code: &[u8]) -> Result<(), &'static str> {
            if code.len() > MAX_JIT_BUFFER_SIZE {
                return Err("Code exceeds buffer capacity");
            }
            self.buf[..code.len()].copy_from_slice(code);
            self.code_len = code.len();
            Ok(())
        }

        pub fn code_len(&self) -> usize {
            self.code_len
        }

        pub fn capacity(&self) -> usize {
            MAX_JIT_BUFFER_SIZE
        }

        pub fn as_ptr(&self) -> *const u8 {
            self.buf.as_ptr()
        }
    }
}
