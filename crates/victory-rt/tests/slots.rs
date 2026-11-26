//! Host functions the title calls through pointers it was handed (method-table slots) have no recorded
//! signature, so they are called with `abi::METHOD_SIG`: sixteen integers, which carries the stack arguments
//! past the eighth. A twelve-argument host function, reached by pointer from lifted code, sees all twelve.
#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

mod common;

use victory_hle::arm64;
use victory_loader::native::NativeImage;

/// Weighted so a wrong, missing or swapped argument changes the result.
extern "C" fn twelve(a: u64, b: u64, c: u64, d: u64, e: u64, f: u64, g: u64, h: u64, i: u64, j: u64, k: u64, l: u64) -> u64 {
    [a, b, c, d, e, f, g, h, i, j, k, l]
        .iter()
        .enumerate()
        .map(|(n, v)| (n as u64 + 1) * v)
        .sum()
}

#[test]
fn a_method_slot_gets_stack_arguments_past_the_eighth() {
    let (pe, dylib) = common::build("slotsprobe", "slotsprobe.c", &[], &[]);
    victory_rt::open(&dylib).expect("open");
    let image = NativeImage::map(&pe, None).expect("map");
    // SAFETY: the image stays mapped for the process.
    unsafe { victory_rt::attach_image(image.base() as *mut u8, image.len()).unwrap() };
    arm64::install();
    // No signature recorded for it, as for a method table's slot: the default applies.
    let thunk = arm64::stub_for(twelve as usize).expect("thunk");
    // SAFETY: an export taking one function pointer.
    let got = unsafe { arm64::call_win64(image.export("probe_call12").unwrap(), &[thunk as u64]) };
    assert_eq!(got, (1..=12u64).map(|n| n * n).sum::<u64>());
}
