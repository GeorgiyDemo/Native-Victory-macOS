//! Exceptions on the arm64 host: the unwind and dispatch functions the loader's `native::seh` serves on the
//! x86-64 host, as plain C functions the runtime's thunks can call (`extern "efiapi"` is the C ABI here, and
//! gives each a signature string). The two that read the caller's registers (`RtlCaptureContext`,
//! `RaiseException`) are `Raw` thunks over the lifted `Cpu` instead: see `title::arm64`.

use std::ffi::c_void;

use victory_loader::native::seh::api;
use victory_loader::native::seh::context::{Context, ExceptionRecord};
use victory_loader::native::seh::unwind::RuntimeFunction;

pub extern "efiapi" fn rtl_lookup_function_entry(
    pc: usize,
    image_base: *mut usize,
    history: *mut c_void,
) -> *const RuntimeFunction {
    api::lookup_function_entry(pc, image_base, history)
}

/// # Safety
/// The title's own call: `ctx` and the out-pointers are its.
pub unsafe extern "efiapi" fn rtl_virtual_unwind(
    kind: u32,
    base: usize,
    pc: usize,
    entry: *const RuntimeFunction,
    ctx: *mut Context,
    handler_data: *mut usize,
    frame: *mut u64,
    pointers: *mut c_void,
) -> usize {
    // SAFETY: forwarded.
    unsafe { api::virtual_unwind(kind, base, pc, entry, ctx, handler_data, frame, pointers) }
}

pub extern "efiapi" fn rtl_add_function_table(entries: usize, count: u32, base: usize) -> i32 {
    api::add_function_table(entries, count, base)
}

pub extern "efiapi" fn rtl_delete_function_table(entries: usize) -> i32 {
    api::delete_function_table(entries)
}

pub extern "efiapi" fn rtl_pc_to_file_header(pc: usize, base: *mut usize) -> usize {
    api::pc_to_file_header(pc, base)
}

pub extern "efiapi" fn add_vectored_exception_handler(first: u32, handler: usize) -> usize {
    api::add_vectored(first, handler)
}

pub extern "efiapi" fn remove_vectored_exception_handler(handle: usize) -> u32 {
    api::remove_vectored(handle)
}

pub extern "efiapi" fn set_unhandled_exception_filter(filter: usize) -> usize {
    api::set_unhandled(filter)
}

/// # Safety
/// `ctx` is a context of this thread's guest stack at a landing pad.
pub unsafe extern "efiapi" fn rtl_restore_context(ctx: *const Context, rec: *mut ExceptionRecord) {
    // SAFETY: forwarded.
    unsafe { api::restore_context(ctx, rec) }
}

/// # Safety
/// Called with a record and context of the title's.
pub unsafe extern "efiapi" fn rtl_unwind_ex(
    target_frame: u64,
    target_ip: usize,
    rec: *mut ExceptionRecord,
    retval: u64,
    orig: *mut Context,
    history: *mut c_void,
) {
    // SAFETY: forwarded.
    unsafe { api::unwind_ex(target_frame, target_ip, rec, retval, orig, history) }
}

/// # Safety
/// Called by the dispatcher or the unwinder, with their records.
pub unsafe extern "efiapi" fn c_specific_handler(
    rec: *mut ExceptionRecord,
    frame: u64,
    ctx: *mut Context,
    dc: *mut victory_loader::native::seh::context::DispatcherContext,
) -> u32 {
    // SAFETY: forwarded.
    unsafe { api::c_specific_handler(rec, frame, ctx, dc) }
}
