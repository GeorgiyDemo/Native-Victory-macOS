//! Structured exception handling over lifted frames on arm64: a C fixture with `__try`/`__except`/`__finally`
//! (built with `-fms-extensions` for the MinGW target, so its `.pdata`/`.xdata` carry C scope tables and the
//! personality is the imported `__C_specific_handler`). Faults in lifted code arrive as signals, become
//! `EXCEPTION_RECORD`s, and are dispatched as on the Rosetta host: vectored handlers, then the frames'
//! language handlers over the image's unwind data (`RtlVirtualUnwind`), the filters run as guest code, and
//! the unwinder resumes at the `__except` block of the frame that took it, dropping the host frames below.
#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

mod common;

use victory_hle::{HleResolver, HostPolicy, arm64};
use victory_loader::UnimplementedPolicy;
use victory_loader::native::NativeImage;

#[test]
fn faults_and_raised_exceptions_are_handled_by_the_images_try_blocks() {
    let (pe, dylib) = common::build("sehprobe", "sehprobe.c", &["ntdll_seh"], &["-fms-extensions"]);
    victory_rt::open(&dylib).expect("open");
    let image = NativeImage::map(&pe, None).expect("map");
    // SAFETY: the image stays mapped for the process.
    unsafe { victory_rt::attach_image(image.base() as *mut u8, image.len()).unwrap() };
    let report = arm64::bind_imports(
        &image,
        &HleResolver::new(HostPolicy::default()),
        UnimplementedPolicy::Trap,
    )
    .expect("bind");
    assert!(report.missing.is_empty(), "every exception function is served: {:?}", report.missing);
    arm64::install();

    // SAFETY (for every call): exports of the lifted image with the arguments they take; a null or
    // small pointer is only ever stored to inside a `__try`.
    let call = |name: &str, args: &[u64]| unsafe { arm64::call_win64(image.export(name).unwrap(), args) };
    let mut ok = 0i32;
    let valid = &raw mut ok as u64;

    // Without a fault nothing is caught.
    assert_eq!(call("probe_fault_in_callee", &[valid]) as u32, 1);
    assert_eq!(ok, 1);
    // The unwind functions, called directly: capture, look up, unwind one frame.
    assert_eq!(call("probe_unwind", &[]) as u32, 1);

    assert_eq!(call("probe_vectored_install", &[]) as u32, 1);
    // A store through a null pointer in a callee: the caller's `__except` takes it.
    assert_eq!(call("probe_fault_in_callee", &[0]) as u32, 2);
    // ... in the guarded code itself.
    assert_eq!(call("probe_fault_inline", &[0]) as u32, 12);
    // The filter sees an access violation and the address.
    let mut code = 0u32;
    assert_eq!(call("probe_filter", &[8, &raw mut code as u64]), 20 << 32 | 8);
    // A `__finally` between the fault and the handler runs while unwinding.
    assert_eq!(call("probe_finally", &[0]) as u32, 42);
    // An inner filter that declines: the outer function takes it.
    assert_eq!(call("probe_outer", &[0]) as u32, 62);
    // A software exception raised by the image.
    assert_eq!(call("probe_raise", &[]) as u32, 32);

    // The vectored handler saw each of the six first, and declined every time.
    let mut last = 0u32;
    assert_eq!(call("probe_vectored_seen", &[&raw mut last as u64]) as u32, 6);
    assert_eq!(last, 0xE000_0001, "the raise came last");

    // And the thread is intact: a normal call afterwards, and a fault again.
    assert_eq!(call("probe_fault_inline", &[valid]) as u32, 11);
    assert_eq!(call("probe_fault_in_callee", &[0]) as u32, 2);
}
