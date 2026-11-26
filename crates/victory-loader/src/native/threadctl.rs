//! Stopping a title thread and reading its registers: `SuspendThread` and
//! `GetThreadContext` on a Mach thread port. Under Rosetta the thread state
//! is the x86-64 guest's (`x86_THREAD_STATE64`, `x86_FLOAT_STATE64`). On the arm64 host the
//! registers of the title are the lifted code's `Cpu`, not the thread's: stopping works, reading does not.

use super::seh::context::Context;
#[cfg(target_arch = "x86_64")]
use super::seh::context::CONTEXT_FULL;

unsafe extern "C" {
    fn mach_thread_self() -> u32;
    fn thread_suspend(thread: u32) -> i32;
    fn thread_resume(thread: u32) -> i32;
    #[cfg(target_arch = "x86_64")]
    fn thread_get_state(thread: u32, flavor: i32, state: *mut u32, count: *mut u32) -> i32;
}

#[cfg(target_arch = "x86_64")]
const X86_THREAD_STATE64: i32 = 4;
#[cfg(target_arch = "x86_64")]
const X86_FLOAT_STATE64: i32 = 5;
/// Words of `x86_thread_state64_t` and `x86_float_state64_t`.
#[cfg(target_arch = "x86_64")]
const THREAD_STATE_COUNT: u32 = 42;
#[cfg(target_arch = "x86_64")]
const FLOAT_STATE_COUNT: u32 = 131;

/// The Mach port of a thread.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ThreadPort(u32);

impl ThreadPort {
    /// The calling thread's port (a send right; kept for the process's life).
    pub fn current() -> ThreadPort {
        // SAFETY: no arguments.
        ThreadPort(unsafe { mach_thread_self() })
    }

    pub fn suspend(&self) -> bool {
        // SAFETY: a thread port of this process.
        unsafe { thread_suspend(self.0) == 0 }
    }

    pub fn resume(&self) -> bool {
        // SAFETY: as above.
        unsafe { thread_resume(self.0) == 0 }
    }

    #[cfg(target_arch = "x86_64")]
    fn thread_state(&self) -> Option<[u64; 21]> {
        let mut words = [0u64; 21];
        let mut count = THREAD_STATE_COUNT;
        // SAFETY: room for `count` words.
        let r = unsafe {
            thread_get_state(self.0, X86_THREAD_STATE64, words.as_mut_ptr().cast(), &mut count)
        };
        (r == 0).then_some(words)
    }

    /// `rip` of a suspended thread.
    #[cfg(target_arch = "x86_64")]
    pub fn rip(&self) -> Option<u64> {
        self.thread_state().map(|w| w[16])
    }

    /// There is no guest `rip` to read on the arm64 host.
    #[cfg(not(target_arch = "x86_64"))]
    pub fn rip(&self) -> Option<u64> {
        None
    }

    /// The registers of a suspended thread, as a `CONTEXT` (not available on the arm64 host).
    #[cfg(not(target_arch = "x86_64"))]
    pub fn context(&self, _ctx: &mut Context) -> bool {
        false
    }

    /// The registers of a suspended thread, as a `CONTEXT`.
    #[cfg(target_arch = "x86_64")]
    pub fn context(&self, ctx: &mut Context) -> bool {
        let Some(words) = self.thread_state() else {
            return false;
        };
        *ctx = Context::zeroed();
        ctx.context_flags = CONTEXT_FULL;
        ctx.set_from_thread_state(&words);
        let mut float = [0u32; FLOAT_STATE_COUNT as usize];
        let mut count = FLOAT_STATE_COUNT;
        // SAFETY: room for `count` words.
        let r = unsafe {
            thread_get_state(self.0, X86_FLOAT_STATE64, float.as_mut_ptr(), &mut count)
        };
        if r == 0 {
            // SAFETY: the 512 bytes after the 8-byte header are the FXSAVE image.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    float.as_ptr().cast::<u8>().add(8),
                    ctx.flt.as_mut_ptr().cast::<u8>(),
                    512,
                )
            };
            ctx.mx_csr = u32::from_le_bytes([ctx.flt[1][8], ctx.flt[1][9], ctx.flt[1][10], ctx.flt[1][11]]);
        }
        true
    }
}
