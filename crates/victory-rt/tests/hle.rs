//! Lifted title code on an arm64 host calling the real `victory-hle`: build `hleprobe.c` into a Windows DLL
//! whose import table names `kernel32.dll` and `xgameruntime.dll`, lift it, load it into this (arm64) test
//! process, and bind every import through the HLE resolver and its signature strings. Nothing of the host's
//! is mocked: the calls land in `host_posix` (events, tick count, last error, feature bits) and in the
//! console runtime (`QueryApiImpl`, then a method through the object's method table, thunked per slot).
//!
//! One test: the runtime keeps its state in the dylib, so one host per process.
#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use std::path::{Path, PathBuf};
use std::process::Command;

use victory_hle::{HleResolver, HostPolicy, arm64};
use victory_loader::native::teb;
use victory_recomp::{image::Image, lift_image};
use victory_rt::Host;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn run(cmd: &mut Command) {
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

/// Builds `hleprobe.c` and its import libraries, lifts it, and builds lifted code and runtime into a
/// dylib. Returns the PE and the dylib's path.
fn build() -> (Vec<u8>, PathBuf) {
    let root = root();
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("hle");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let fixtures = root.join("crates/victory-recomp/tests/fixtures");
    for lib in ["kernel32", "xgameruntime"] {
        run(Command::new("/opt/homebrew/opt/llvm/bin/llvm-dlltool")
            .args(["-m", "i386:x86-64", "-d"])
            .arg(fixtures.join(format!("{lib}.def")))
            .arg("-l")
            .arg(dir.join(format!("lib{lib}.a"))));
    }
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
            "-o",
        ])
        .arg(dir.join("hleprobe.dll"))
        .arg(fixtures.join("hleprobe.c"))
        .arg(format!("-L{}", dir.display()))
        .args(["-lkernel32", "-lxgameruntime"]));
    let pe = std::fs::read(dir.join("hleprobe.dll")).unwrap();

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

#[test]
fn lifted_code_calls_the_real_hle_on_arm64() {
    let (pe, dylib) = build();

    // The imports as the Rosetta host's binder resolves them, with their derived signatures.
    let image = Image::parse(&pe).unwrap();
    let resolver = HleResolver::new(HostPolicy::default());
    let (imports, missing) = arm64::rt_imports(&resolver, &image.info.imports);
    assert!(missing.is_empty(), "not served: {missing:?}");
    assert_eq!(imports.len(), 12);

    let host = Host::load(&dylib, &pe, &imports).expect("load");
    // The loader's TEB (static TLS, last error) is the one `gs:` reads, from here on.
    arm64::install();
    arm64::attach_thread();
    let call = |name: &str, args: &[u64]| host.call(name, args).unwrap();
    assert_eq!(call("guest_teb_self", &[]), teb::current() as u64);

    // kernel32: GetTickCount64 returns a u64 in x0, and the clock does not go back.
    let mut ticks = [0u64; 2];
    assert_eq!(
        call("guest_ticks", &[ticks.as_mut_ptr() as u64, ticks[1..].as_mut_ptr() as u64]),
        1
    );
    assert!(ticks[0] <= ticks[1]);

    // CreateEventW / WaitForSingleObject / SetEvent: handles, 32-bit results and a timeout.
    let packed = call("guest_event", &[]);
    assert_eq!((packed >> 32, packed as u32), (0x102, 0), "{packed:#x}");

    // SetLastError / GetLastError: the TEB the HLE writes is the TEB the lifted code reads.
    let last = call("guest_last_error", &[0x4d2]);
    assert_eq!((last >> 32, last as u32), (0x4d2, 0x4d2), "{last:#x}");
    // QueryPerformanceFrequency writes through a pointer on the guest stack.
    assert_eq!(call("guest_qpf", &[]), 1_000_000_000);
    // AVX and AVX2 are present (feature codes 39 and 40).
    assert_eq!(call("guest_has_avx", &[]), 3);

    // xgameruntime: an object from QueryApiImpl, then a method through its method table (a thunk per slot).
    let mut id = 0u32;
    assert_eq!(call("guest_title_id", &[&raw mut id as u64]), 0);
    assert_eq!(id, victory_hle::runtime::title().id);

    // Another thread: its own TEB, its own last error, its own id, through the loader's thread start.
    let main_thread = call("guest_thread_id", &[]);
    let export = |name| host.export(name).unwrap() as usize;
    let (last_error, teb_self, thread_id) =
        (export("guest_last_error"), export("guest_teb_self"), export("guest_thread_id"));
    let (other_thread, other_last) = std::thread::spawn(move || {
        // SAFETY (for each): an export of the lifted image taking these integer arguments.
        let last = unsafe { arm64::call_win64(last_error, &[99]) };
        assert_eq!(unsafe { arm64::call_win64(teb_self, &[]) }, teb::current() as u64);
        (unsafe { arm64::call_win64(thread_id, &[]) }, last)
    })
    .join()
    .unwrap();
    assert_ne!(main_thread, other_thread);
    assert_eq!((other_last >> 32, other_last as u32), (99, 99));
    // ... and the first thread's error is still its own.
    assert_eq!(call("guest_last_error", &[0x4d2]) >> 32, 0x4d2);
}
