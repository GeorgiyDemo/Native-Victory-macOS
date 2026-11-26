//! A title DLL on the arm64 host: mapped by the loader's `map_title_dll` (which gives its static TLS an index),
//! bound through the HLE binder, started with `attach_dll` (TLS callbacks, then `DllMain`) through the
//! arm64 caller, told of threads (`DLL_THREAD_ATTACH`/`DETACH`) by `spawn_title_thread`, and run as the main
//! image by `run_entry_with` (its TLS callbacks, then the entry point, then detach).
//!
//! One lifted image per process (the runtime has one image), so the DLL is also the "main" image of the last step.
#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

mod common;

use std::sync::mpsc;

use victory_hle::{HleResolver, HostPolicy, arm64};
use victory_loader::native::dll::{WithTitleDlls, map_title_dll};
use victory_loader::native::{attach_dll, register_dll_thread_notices, run_entry_with, spawn_title_thread};
use victory_loader::{DLL_PROCESS_ATTACH, DLL_PROCESS_DETACH, UnimplementedPolicy};

#[test]
fn a_title_dll_starts_with_tls_callbacks_and_dllmain() {
    let (pe, dylib) = common::build("dllprobe", "dllprobe.c", &[], &["-fno-emulated-tls"]);
    victory_rt::open(&dylib).expect("open");
    // The loader's own path: found in the title's folder, mapped, its TLS registered.
    let dir = dylib.parent().unwrap();
    let mapped = map_title_dll("dllprobe.dll", dir, &|_| true).expect("map");
    let dll = *mapped.last().expect("mapped");
    let image = &dll.image;
    assert!(image.tls().is_some(), "the fixture has a TLS directory");
    // SAFETY: the image stays mapped for the process.
    unsafe { victory_rt::attach_image(image.base() as *mut u8, image.len()).unwrap() };
    let hle = HleResolver::new(HostPolicy::default());
    let report = arm64::bind_imports(image, &WithTitleDlls { inner: &hle }, UnimplementedPolicy::ReturnZero)
        .expect("bind");
    assert!(report.missing.is_empty());
    arm64::install();
    let export = |name| image.export(name).unwrap();
    let events = || {
        let mut out = [0u32; 32];
        // SAFETY: an export taking a pointer to 32 words.
        let n = unsafe { arm64::call_win64(export("probe_events"), &[out.as_mut_ptr() as u64]) } as usize;
        out[..n].to_vec()
    };
    // SAFETY (for every call below): exports of the lifted DLL with the arguments they take.
    let call = |name, args: &[u64]| unsafe { arm64::call_win64(export(name), args) } as u32;

    // DLL_PROCESS_ATTACH: TLS callbacks first, then DllMain.
    assert!(unsafe { attach_dll(image, DLL_PROCESS_ATTACH, 1) });
    assert_eq!(events(), [0x101, 0x201]);

    // Static TLS: this thread's block starts from the template (40) and is its own.
    assert_eq!((call("probe_tls_bump", &[]), call("probe_tls_bump", &[])), (41, 42));

    // A new thread: hears DLL_THREAD_ATTACH (callback, DllMain) before its start routine and
    // DLL_THREAD_DETACH after, starts from the template again, and its start routine's result is its exit code.
    register_dll_thread_notices(image);
    let (tx, rx) = mpsc::channel();
    // SAFETY: `probe_thread_main` is a one-argument function of the bound image.
    let thread = unsafe {
        spawn_title_thread("probe", 1 << 20, export("probe_thread_main"), 100, move |code| {
            tx.send(code).unwrap()
        })
    }
    .expect("spawn");
    thread.join();
    assert_eq!(rx.recv().unwrap(), 141);
    assert_eq!(events()[2..], [0x102, 0x202, 0x103, 0x203]);
    // ... and this thread's counter is untouched by it.
    assert_eq!(call("probe_tls_bump", &[]), 43);

    // As the main image: process attach (its TLS callback), the entry point (DllMain, reason 0 for lack of
    // arguments), process detach.
    // SAFETY: the image is bound; it lives for the process (leaked: `run_entry_with` wants 'static).
    let main: &'static _ = unsafe { &*std::ptr::from_ref(image) };
    assert_eq!(unsafe { run_entry_with(main, &[]) }.expect("run"), 1);
    assert_eq!(events()[6..], [0x101, 0x200, 0x100]);
    assert_eq!(DLL_PROCESS_DETACH, 0);
}
