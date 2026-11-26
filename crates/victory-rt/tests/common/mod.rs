//! Shared by the arm64 host tests: build a fixture into a Windows DLL, lift it, build lifted code and
//! runtime into a dylib. The runtime keeps its state in the dylib, so each test file is one host per process.
#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::process::Command;

use victory_recomp::{image::Image, lift_image};

pub fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

pub fn fixtures() -> PathBuf {
    root().join("crates/victory-recomp/tests/fixtures")
}

pub fn run(cmd: &mut Command) {
    let out = cmd
        .output()
        .unwrap_or_else(|e| panic!("spawn {cmd:?}: {e}"));
    assert!(
        out.status.success(),
        "{cmd:?} failed:\n{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}

/// Builds `fixtures/<source>` (a C file; `flags` are extra compiler/linker flags) into a DLL linked against
/// import libraries made from `fixtures/<lib>.def` for each of `libs`, then lifts it and builds lifted code
/// and runtime into a dylib. Returns the PE and the dylib's path.
pub fn build(name: &str, source: &str, libs: &[&str], flags: &[&str]) -> (Vec<u8>, PathBuf) {
    let root = root();
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for lib in libs {
        run(Command::new("/opt/homebrew/opt/llvm/bin/llvm-dlltool")
            .args(["-m", "i386:x86-64", "-d"])
            .arg(fixtures().join(format!("{lib}.def")))
            .arg("-l")
            .arg(dir.join(format!("lib{lib}.a"))));
    }
    let dll = dir.join(format!("{name}.dll"));
    run(Command::new(root.join("tools/macos/gnullvm-cc.sh"))
        .args([
            "-O1",
            "-funwind-tables",
            "-fno-vectorize",
            "-fno-slp-vectorize",
            "-fno-builtin",
            "-ffreestanding",
            "-nostdlib",
            "-shared",
            "-Wl,--entry=DllMain",
            // Relocatable: 0x180000000, the default base, is the dyld shared cache on arm64 macOS.
            "-Wl,--dynamicbase",
        ])
        .args(flags)
        .arg("-o")
        .arg(&dll)
        .arg(fixtures().join(source))
        .arg(fixtures().join("reloc.c"))
        .arg(format!("-L{}", dir.display()))
        .args(libs.iter().map(|l| format!("-l{l}"))));
    let pe = std::fs::read(&dll).unwrap();

    let unit = lift_image(&Image::parse(&pe).unwrap()).unwrap();
    assert!(unit.failed.is_empty(), "unlifted: {:?}", unit.failed);
    std::fs::write(dir.join("unit.c"), &unit.c).unwrap();
    std::fs::write(dir.join("rt.h"), victory_recomp::lift::RT_HEADER).unwrap();
    std::fs::write(dir.join("rt_host.c"), victory_recomp::RT_HOST_C).unwrap();
    run(Command::new("clang")
        .args([
            "-arch",
            "arm64",
            "-O2",
            "-ffp-contract=off",
            "-Wno-unused",
            "-dynamiclib",
            "-I",
        ])
        .arg(&dir)
        .arg(dir.join("unit.c"))
        .arg(dir.join("rt_host.c"))
        .arg("-o")
        .arg(dir.join("libtitle.dylib")));
    (pe, dir.join("libtitle.dylib"))
}
