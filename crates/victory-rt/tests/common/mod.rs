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
    let dir = scratch(name);
    let pe = build_pe(&dir, name, source, libs, flags);
    lift_into(&dir, &pe);
    run(Command::new("clang")
        .args(["-arch", "arm64", "-O2", "-ffp-contract=off", "-Wno-unused", "-dynamiclib", "-I"])
        .arg(&dir)
        .arg(dir.join("unit.c"))
        .arg(dir.join("rt_host.c"))
        .arg("-o")
        .arg(dir.join("libtitle.dylib")));
    (pe, dir.join("libtitle.dylib"))
}

/// The runtime alone, built for several lifted images (`-DRT_MULTI`): the dylib to `victory_rt::open`.
pub fn runtime_dylib() -> PathBuf {
    let dir = scratch("rt_multi");
    std::fs::write(dir.join("rt.h"), victory_recomp::lift::RT_HEADER).unwrap();
    std::fs::write(dir.join("rt_host.c"), victory_recomp::RT_HOST_C).unwrap();
    run(Command::new("clang")
        .args(["-arch", "arm64", "-O2", "-Wno-unused", "-DRT_MULTI", "-dynamiclib", "-I"])
        .arg(&dir)
        .arg(dir.join("rt_host.c"))
        .arg("-o")
        .arg(dir.join("librt.dylib")));
    dir.join("librt.dylib")
}

/// As [`build`], for an image that shares the process with others: the lifted code and `rt_unit.c` go into a
/// dylib of their own, which binds the runtime ([`runtime_dylib`]) when loaded (`victory_rt::attach_unit`).
/// The DLL is left at `<scratch>/<name>/<name>.dll`: the returned folder is where the loader finds it.
pub fn build_unit(name: &str, source: &str, libs: &[&str], flags: &[&str]) -> (Vec<u8>, PathBuf) {
    let dir = scratch(name);
    let pe = build_pe(&dir, name, source, libs, flags);
    lift_into(&dir, &pe);
    std::fs::write(dir.join("rt_unit.c"), victory_recomp::RT_UNIT_C).unwrap();
    let out = dir.join(format!("lib{name}.dylib"));
    run(Command::new("clang")
        .args([
            "-arch",
            "arm64",
            "-O2",
            "-ffp-contract=off",
            "-Wno-unused",
            "-dynamiclib",
            "-Wl,-undefined,dynamic_lookup",
            "-I",
        ])
        .arg(&dir)
        .arg(dir.join("unit.c"))
        .arg(dir.join("rt_unit.c"))
        .arg("-o")
        .arg(&out));
    (pe, out)
}

fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// As [`build_unit`], for a DLL built for the MSVC target (`x86_64-pc-windows-msvc`: what the title is) from `sources`
/// (C, or C++ with exceptions on, by extension), linked by `lld-link` with no runtime of its own: import libraries
/// come from `fixtures/<lib>.def`, exports from `__declspec(dllexport)`.
pub fn build_msvc_unit(name: &str, sources: &[&str], libs: &[&str]) -> (Vec<u8>, PathBuf) {
    let dir = scratch(name);
    for lib in libs {
        run(Command::new("/opt/homebrew/opt/llvm/bin/llvm-dlltool")
            .args(["-m", "i386:x86-64", "-d"])
            .arg(fixtures().join(format!("{lib}.def")))
            .arg("-l")
            .arg(dir.join(format!("lib{lib}.a"))));
    }
    let mut objects = vec![];
    for (n, source) in sources.iter().chain(&["reloc.c"]).enumerate() {
        let object = dir.join(format!("{n}.obj"));
        let mut cc = Command::new("clang");
        cc.args([
            "--target=x86_64-pc-windows-msvc",
            "-O1",
            "-funwind-tables",
            "-fno-vectorize",
            "-fno-slp-vectorize",
            "-fno-builtin",
            "-ffreestanding",
            "-c",
        ]);
        if source.ends_with(".cpp") {
            cc.args([
                "-fexceptions",
                "-fcxx-exceptions",
                "-fno-rtti",
                "-nostdinc++",
                "-fms-extensions",
                "-fno-threadsafe-statics",
            ]);
        }
        run(cc.arg(fixtures().join(source)).arg("-o").arg(&object));
        objects.push(object);
    }
    let dll = dir.join(format!("{name}.dll"));
    run(Command::new("/opt/homebrew/bin/lld-link")
        .args(["/dll", "/nodefaultlib", "/entry:DllMain", "/dynamicbase"])
        .arg(format!("/out:{}", dll.display()))
        .args(&objects)
        .args(libs.iter().map(|l| dir.join(format!("lib{l}.a")))));
    let pe = std::fs::read(&dll).unwrap();
    lift_into(&dir, &pe);
    std::fs::write(dir.join("rt_unit.c"), victory_recomp::RT_UNIT_C).unwrap();
    let out = dir.join(format!("lib{name}.dylib"));
    run(Command::new("clang")
        .args([
            "-arch",
            "arm64",
            "-O2",
            "-ffp-contract=off",
            "-Wno-unused",
            "-dynamiclib",
            "-Wl,-undefined,dynamic_lookup",
            "-I",
        ])
        .arg(&dir)
        .arg(dir.join("unit.c"))
        .arg(dir.join("rt_unit.c"))
        .arg("-o")
        .arg(&out));
    (pe, out)
}

/// The DLL as a PE, written to `dir/<name>.dll`.
fn build_pe(dir: &Path, name: &str, source: &str, libs: &[&str], flags: &[&str]) -> Vec<u8> {
    let root = root();
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
    std::fs::read(&dll).unwrap()
}

fn lift_into(dir: &Path, pe: &[u8]) {
    let unit = lift_image(&Image::parse(pe).unwrap()).unwrap();
    assert!(unit.failed.is_empty(), "unlifted: {:?}", unit.failed);
    std::fs::write(dir.join("unit.c"), &unit.c).unwrap();
    std::fs::write(dir.join("rt.h"), victory_recomp::lift::RT_HEADER).unwrap();
    std::fs::write(dir.join("rt_host.c"), victory_recomp::RT_HOST_C).unwrap();
}
