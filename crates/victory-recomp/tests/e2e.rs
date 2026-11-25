//! End to end: build a small x86-64 Windows DLL with the gnullvm toolchain,
//! statically recompile it to arm64, and check the native result prints exactly
//! what the original x86-64 code prints when run under Rosetta 2.
#![cfg(target_os = "macos")]

mod common;

use std::process::Command;

use common::*;
use victory_recomp::{image::Image, lift_image};

#[test]
fn testpe_recompiled_output_matches_x86() {
    if !have_tools() {
        eprintln!("skipped: needs Rosetta 2 and clang");
        return;
    }
    let root = repo_root();
    let dir = workdir("e2e");
    let src = root.join("crates/victory-recomp/tests/fixtures");
    let dll = dir.join("testpe.dll");
    // `-funwind-tables` so every function gets .pdata like a MSVC build would.
    run(Command::new(root.join("tools/macos/gnullvm-cc.sh")).args([
        "-O1", "-funwind-tables", "-fno-vectorize", "-fno-slp-vectorize", "-fno-builtin", "-fno-math-errno",
        "-ffreestanding", "-nostdlib", "-shared", "-Wl,--entry=DllMain", "-o",
    ]).arg(&dll).arg(src.join("testpe.c")));

    let data = std::fs::read(&dll).unwrap();
    let img = Image::parse(&data).unwrap();
    let unit = lift_image(&img).unwrap();
    assert!(unit.failed.is_empty(), "unlifted functions: {:?}", unit.failed);
    assert!(unit.lifted.len() >= 18);
    std::fs::write(dir.join("unit.c"), &unit.c).unwrap();

    // Flat image (what the host loader would map) and the export table.
    let mut flat = vec![0u8; img.info.size_of_image as usize];
    for s in &img.info.sections {
        let raw = &data[s.file_offset as usize..(s.file_offset + s.file_size) as usize];
        flat[s.rva as usize..s.rva as usize + raw.len()].copy_from_slice(raw);
    }
    std::fs::write(dir.join("image.bin"), flat).unwrap();
    let exports: String = img.info.exports.iter().filter_map(|e| Some(format!("{} {:#x}\n", e.name.as_ref()?, e.rva?))).collect();
    std::fs::write(dir.join("exports.txt"), exports).unwrap();
    std::fs::copy(src.join("e2e_driver.c"), dir.join("e2e_driver.c")).unwrap();

    let (x86, arm) = run_both(&dir, "e2e_driver.c", "unit.c", &["image.bin", "exports.txt"]);
    assert!(x86.lines().count() > 7000);
    assert_eq!(x86, arm, "lifted arm64 output differs from the x86-64 original");
}
