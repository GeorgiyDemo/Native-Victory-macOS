//! End to end on an arm64 host with no Rosetta: build `hostprobe.c` into a Windows DLL (with a real import
//! table for `host.dll`), lift it, build the lifted code and the runtime into a dylib, load it into this very
//! (arm64) test process, bind the imports to Rust functions and run exports.
//!
//! One test: the runtime keeps its state in the dylib, so one host per process.
#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use std::ffi::{CStr, c_char};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;

use victory_recomp::{image::Image, lift_image};
use victory_rt::{Binding, Cpu, Host, Import, call_guest, reg};

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

// ---- the host's side of the imports: plain `extern "C"` Rust ----

static PUTS: Mutex<Vec<String>> = Mutex::new(Vec::new());
static FALLBACK_HITS: Mutex<Vec<u64>> = Mutex::new(Vec::new());

extern "C" fn host_add3(a: u64, b: u64, c: u64) -> u64 {
    a + b + c
}
extern "C" fn host_mix(a: u64, b: f64, c: u64, d: f32, e: u64, f: f64) -> f64 {
    (a + c + e) as f64 + b * 2.0 + d as f64 + f * 0.5
}
extern "C" fn host_sum8(a: u64, b: u64, c: u64, d: u64, e: u64, f: u64, g: u64, h: u64) -> u64 {
    a + 2 * b + 3 * c + 4 * d + 5 * e + 6 * f + 7 * g + 8 * h
}
/// Calls a guest function pointer: the host-to-lifted callback path.
extern "C" fn host_callback(cb: u64, x: u64) -> u64 {
    call_guest(cb, &[x, 2.5f64.to_bits()], 0b10).0
}
extern "C" fn host_puts(s: *const c_char) -> i32 {
    // SAFETY: the lifted code passes a NUL-terminated string in the image.
    let s = unsafe { CStr::from_ptr(s) }.to_string_lossy().into_owned();
    let n = s.len() as i32;
    PUTS.lock().unwrap().push(s);
    n
}
/// A fallback for dispatch targets the lifter has no function for: acts like a callee returning 0x1000.
unsafe extern "C" fn fallback(c: *mut Cpu, target: u64) {
    FALLBACK_HITS.lock().unwrap().push(target);
    // SAFETY: the runtime passes the live Cpu of the calling lifted frame.
    unsafe {
        (*c).r[reg::RAX] = 0x1000;
        (*c).r[reg::RSP] += 8; // the callee's `ret`
    }
}

fn imports() -> Vec<Import> {
    let typed = |name, f: *const std::ffi::c_void, sig| Import {
        name,
        binding: Binding::Typed(f, sig),
    };
    vec![
        typed("host_add3", host_add3 as *const _, "iii>i"),
        typed("host_mix", host_mix as *const _, "idifid>d"),
        typed("host_sum8", host_sum8 as *const _, "iiiiiiii>i"),
        typed("host_callback", host_callback as *const _, "ii>i"),
        typed("host_puts", host_puts as *const _, "i>w"),
    ]
}

#[test]
fn lifted_title_code_runs_natively_with_imports_callbacks_gs_and_cpuid() {
    let root = root();
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("native");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let fixtures = root.join("crates/victory-recomp/tests/fixtures");

    // The title-shaped input: a Windows DLL importing host.dll.
    run(Command::new("/opt/homebrew/opt/llvm/bin/llvm-dlltool")
        .args(["-m", "i386:x86-64", "-d"])
        .arg(fixtures.join("host.def"))
        .arg("-l")
        .arg(dir.join("libhost.a")));
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
        .arg(dir.join("hostprobe.dll"))
        .arg(fixtures.join("hostprobe.c"))
        .arg(format!("-L{}", dir.display()))
        .arg("-lhost"));
    let pe = std::fs::read(dir.join("hostprobe.dll")).unwrap();

    // Lift, and build lifted code + runtime into a dylib.
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

    let host = Host::load(&dir.join("libtitle.dylib"), &pe, &imports()).expect("load");

    // ---- imports: argument marshalling (ints, doubles, a float, stack arguments), results, callbacks ----
    let n = 5u64;
    let want =
        (3 * n + 3) + ((3.0 * n as f64 + 21.25) * 1000.0) as u64 + (140 + 8 * n) + (n * n + 2);
    assert_eq!(host.call("guest_run", &[n]).unwrap(), want);
    assert_eq!(PUTS.lock().unwrap().as_slice(), ["hello from lifted code"]);
    // The callback ran on a fresh Cpu below the lifted frame, with its double argument in xmm1.
    let (rax, _) = host
        .call_ex("guest_square_plus", &[7, 3.75f64.to_bits()], 0b10)
        .unwrap();
    assert_eq!(rax, 49 + 3);

    // ---- gs: is the Cpu's TEB pointer ----
    assert_eq!(host.call("guest_teb_self", &[]).unwrap(), host.teb as u64);
    // SAFETY: the TEB was filled by init_thread.
    let stack_base = unsafe { host.teb.add(8).cast::<u64>().read() };
    assert_eq!(host.call("guest_teb_stack_base", &[]).unwrap(), stack_base);

    // ---- CPUID, XGETBV, RDTSC answer like the console's CPU ----
    let mut out = [0u32; 4];
    for (leaf, sub) in [(0, 0), (1, 0), (7, 0), (0x80000001, 0), (0xD, 1)] {
        host.call(
            "guest_cpuid",
            &[leaf as u64, sub as u64, out.as_mut_ptr() as u64],
        )
        .unwrap();
        assert_eq!(out, host.cpuid(leaf, sub), "leaf {leaf:#x}");
    }
    let l0 = host.cpuid(0, 0);
    let vendor: Vec<u8> = [l0[1], l0[3], l0[2]]
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect();
    assert_eq!(vendor, b"AuthenticAMD");
    let bit = |w: u32, b: u32| w >> b & 1 == 1;
    let l1 = host.cpuid(1, 0);
    // SSE3 PCLMUL SSSE3 FMA CX16 SSE4.1 SSE4.2 MOVBE POPCNT AES XSAVE OSXSAVE AVX F16C RDRAND; not a hypervisor.
    for b in [0, 1, 9, 12, 13, 19, 20, 22, 23, 25, 26, 27, 28, 29, 30] {
        assert!(bit(l1[2], b), "leaf 1 ecx bit {b}");
    }
    assert!(!bit(l1[2], 31), "hypervisor bit");
    assert!(bit(l1[3], 25) && bit(l1[3], 26), "SSE and SSE2");
    let l7 = host.cpuid(7, 0);
    for b in [0, 3, 5, 8, 18, 19, 29] {
        assert!(bit(l7[1], b), "leaf 7 ebx bit {b}"); // BMI1 AVX2 BMI2 RDSEED ADX SHA
    }
    assert!(!bit(l7[1], 16), "no AVX-512");
    let x1 = host.cpuid(0x80000001, 0);
    assert!(bit(x1[2], 5) && bit(x1[2], 6), "ABM (lzcnt) and SSE4a");
    assert_eq!(host.call("guest_xgetbv", &[]).unwrap(), 7);
    let d = host.call("guest_tsc_delta", &[]).unwrap();
    assert!(d < 1_000_000, "rdtsc went backwards or stalled: {d}");

    // ---- dispatch of addresses the lifter has no function for goes to the logged, pluggable fallback ----
    host.set_fallback(fallback);
    assert_eq!(
        host.call("guest_indirect", &[0xdead_0000_0000, 7]).unwrap(),
        0x1001
    );
    let (img, _) = host.image();
    // Inside the image but not a function start (one byte past the export).
    let odd = host.export("guest_square_plus").unwrap() + 1;
    assert_eq!(host.call("guest_indirect", &[odd, 7]).unwrap(), 0x1001);
    assert_eq!(
        FALLBACK_HITS.lock().unwrap().as_slice(),
        [0xdead_0000_0000, odd]
    );
    assert_eq!(host.unresolved_count(), 2);
    assert!(odd > img as u64);
}
