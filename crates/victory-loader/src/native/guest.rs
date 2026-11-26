//! Calls into title code on the arm64 host. The title is recompiled (`victory-rt`), not run by the CPU, so
//! there is no bridge: a call is `rt_call_guest`. The runtime lives in a crate this one cannot depend on
//! (it would pull the recompiler in), so the host installs the caller once, before any title code runs.

use std::sync::OnceLock;

/// Calls the Windows x64 function at `f` with up to 12 integer or pointer arguments, returning `rax`.
pub type GuestCall = unsafe fn(f: usize, args: &[u64]) -> u64;

static CALL: OnceLock<GuestCall> = OnceLock::new();

/// Installs the function that enters recompiled code. The first call wins.
pub fn install(call: GuestCall) {
    let _ = CALL.set(call);
}

/// Continues title code at a context (the registers an unwinder computed), dropping the frames of whatever was
/// running below it. The arm64 host's runtime does it (`victory_rt::resume`).
pub type GuestResume = unsafe fn(&super::seh::context::Context) -> !;

static RESUME: OnceLock<GuestResume> = OnceLock::new();

/// Installs the function that continues title code at a context. The first call wins.
pub fn install_resume(resume: GuestResume) {
    let _ = RESUME.set(resume);
}

/// Continues at `ctx`.
///
/// # Safety
/// `ctx` is an unwound context of this thread's guest stack, at a landing pad.
pub(crate) unsafe fn resume(ctx: *const super::seh::context::Context) -> ! {
    let resume = RESUME
        .get()
        .expect("no guest resume installed (`native::guest::install_resume`)");
    // SAFETY: forwarded.
    unsafe { resume(&*ctx) }
}

/// Calls the Windows-ABI function `f` of a title image.
///
/// # Safety
/// `f` is a function of the title taking `args.len()` integer or pointer arguments, in an image whose
/// imports are bound.
pub unsafe fn call_win64(f: usize, args: &[u64]) -> u64 {
    let call = CALL
        .get()
        .expect("no guest runtime installed (`native::guest::install`)");
    // SAFETY: forwarded.
    unsafe { call(f, args) }
}

/// Whether `address` is code of a mapped title image, as opposed to a function of ours.
pub fn is_title_code(address: usize) -> bool {
    super::seh::unwind::image_base_of(address).is_some()
}
