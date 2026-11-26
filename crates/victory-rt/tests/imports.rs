//! Binding a lifted image the way the loader binds a native one: `NativeImage::map` places and relocates it,
//! the HLE binder fills its IAT, and what the HLE does not serve (a name, an ordinal) is stubbed to log and
//! return 0 (`UnimplementedPolicy::ReturnZero`) rather than refused.
#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

mod common;

use victory_hle::{HleResolver, HostPolicy, arm64};
use victory_loader::UnimplementedPolicy;
use victory_loader::native::NativeImage;

#[test]
fn a_native_image_binds_with_stubs_for_the_unserved() {
    let (pe, dylib) = common::build("bindprobe", "bindprobe.c", &["kernel32", "notserved"], &[]);
    victory_rt::open(&dylib).expect("open");
    let image = NativeImage::map(&pe, None).expect("map");
    // SAFETY: the image stays mapped for the process.
    unsafe { victory_rt::attach_image(image.base() as *mut u8, image.len()).unwrap() };
    let report = arm64::bind_imports(
        &image,
        &HleResolver::new(HostPolicy::default()),
        UnimplementedPolicy::ReturnZero,
    )
    .expect("bind");
    assert_eq!(report.hle + report.host, 1, "{report:?}");
    let mut missing: Vec<_> = report.missing.iter().map(|m| (m.dll.as_str(), m.symbol.as_str())).collect();
    missing.sort();
    assert_eq!(missing.len(), 2, "{missing:?}");
    assert!(missing.iter().any(|m| m.1 == "NotServedByName"), "{missing:?}");
    assert!(missing.iter().any(|m| m.1.contains('9')), "ordinal 9: {missing:?}");

    arm64::install();
    // SAFETY: exports of the lifted image taking no arguments.
    let call = |name: &str| unsafe { arm64::call_win64(image.export(name).unwrap(), &[]) };
    // The served import runs the real HLE: the loader's thread id, the same one the host reports.
    let tid = call("probe_served") as u32;
    assert_eq!(tid, victory_platform::thread::current_id());
    // The unserved ones returned 0 (the constants added after are all that is left), and the second call
    // does not log again or fail.
    for _ in 0..2 {
        assert_eq!(call("probe_unserved"), 7 | 11 << 32);
    }
}
