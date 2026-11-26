//! Exceptions on arm64, second round: integer divide errors raise `EXCEPTION_INT_DIVIDE_BY_ZERO` (and
//! `EXCEPTION_INT_OVERFLOW`) instead of trapping; a filter that answers `EXCEPTION_CONTINUE_EXECUTION` resumes at the
//! faulting instruction with the context it modified; and an exception raised in a guest callback that a host
//! function made is caught by the guest frame above the host function. The callback itself is a frameless leaf that
//! only `lea` reaches, which discovery seeds.
#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

mod common;

use victory_hle::{HleResolver, HostPolicy, abi, arm64};
use victory_loader::native::NativeImage;
use victory_loader::{ImportResolver, Provenance, Resolution, UnimplementedPolicy};
use victory_pe::ImportSymbol;

/// A function of the host that calls a guest function back: what `EnumWindows` or a `qsort` comparator is.
extern "C" fn host_apply(cb: u64, arg: u64) -> u64 {
    victory_rt::call_guest(cb, &[arg], 0).0
}

struct WithHostApply<'a>(&'a dyn ImportResolver);

impl ImportResolver for WithHostApply<'_> {
    fn resolve(&self, dll: &str, symbol: &ImportSymbol) -> Resolution {
        match symbol {
            ImportSymbol::Name { name, .. } if name == "host_apply" => {
                let address = host_apply as *const () as usize;
                abi::record(address, "ii>i");
                Resolution::Found { address, provenance: Provenance::Host }
            }
            _ => self.0.resolve(dll, symbol),
        }
    }
}

#[test]
fn divide_errors_continued_faults_and_exceptions_across_a_host_function() {
    let (pe, dylib) = common::build("sehprobe2", "sehprobe2.c", &["ntdll_seh", "host2"], &["-fms-extensions"]);
    victory_rt::open(&dylib).expect("open");
    let image = NativeImage::map(&pe, None).expect("map");
    // SAFETY: the image stays mapped for the process.
    unsafe { victory_rt::attach_image(image.base() as *mut u8, image.len()).unwrap() };
    let hle = HleResolver::new(HostPolicy::default());
    let report = arm64::bind_imports(&image, &WithHostApply(&hle), UnimplementedPolicy::Trap).expect("bind");
    assert!(report.missing.is_empty(), "{:?}", report.missing);
    arm64::install();

    // SAFETY (for every call): exports of the lifted image with the arguments they take; the pointers are valid or
    // null where a fault is the point.
    let call = |name: &str, args: &[u64]| unsafe { arm64::call_win64(image.export(name).unwrap(), args) };

    // --- integer divide errors ---
    assert_eq!(call("probe_div", &[5]) as u32, 20, "no fault when the divisor is not zero");
    assert_eq!(call("probe_div", &[0]) as u32, 7777, "a divide by zero is caught by the caller's __except");
    assert_eq!(call("probe_div_code", &[]) as u32, 0xC000_0094, "EXCEPTION_INT_DIVIDE_BY_ZERO");
    let (min, minus_one) = (i32::MIN, -1i32);
    assert_eq!(call("probe_idiv_overflow", &[&raw const min as u64, &raw const minus_one as u64]) as u32, 8888);
    assert_eq!(call("probe_div_code", &[]) as u32, 0xC000_0095, "EXCEPTION_INT_OVERFLOW");

    // --- continuing execution ---
    // The divide by zero runs again with the divisor the filter put in rcx (and the dividend it still had).
    assert_eq!(call("probe_div_continue", &[0]) as u32, 20);
    assert_eq!(call("probe_div_continue", &[4]) as u32, 25, "a divisor that is fine is not touched");
    // An access violation: the store runs again on the slot the filter chose.
    assert_eq!(call("probe_av_continue", &[0]) as u32, 5);

    // --- exceptions across a host function ---
    assert_eq!(victory_rt::is_loaded(), true);
    let mut slot = 0i32;
    assert_eq!(call("probe_callback_plain", &[&raw mut slot as u64]) as u32, 3, "no exception: the result comes back");
    assert_eq!(slot, 1);
    // A fault in the callback, taken by the __except above the host function.
    assert_eq!(call("probe_callback_fault", &[0]) as u32, 70);
    // A raised exception, with a __finally in the caller that runs on the way.
    assert_eq!(call("probe_callback_raise", &[]) as u32, 72);
    // The thread is intact afterwards: plain callbacks and faults again.
    assert_eq!(call("probe_callback_plain", &[&raw mut slot as u64]) as u32, 3);
    assert_eq!(call("probe_callback_fault", &[0]) as u32, 70);
    assert_eq!(call("probe_div", &[0]) as u32, 7777);
}
