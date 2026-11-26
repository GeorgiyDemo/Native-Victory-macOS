//! The recompiler's existing end-to-end input (`tests/fixtures/testpe.c`, 18 exports) run natively in an arm64
//! process through the host: the lifted code in a dylib, the image mapped by `victory-rt`, no Rosetta. The
//! expected values are computed here in Rust (the recompiler's own `tests/e2e.rs` already proves the lifted
//! code equal to the x86-64 original under Rosetta for 7,200 random calls).
#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use std::path::{Path, PathBuf};
use std::process::Command;

use victory_recomp::{image::Image, lift_image};
use victory_rt::Host;

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

fn helper(x: u64) -> u64 {
    x.wrapping_mul(3).wrapping_add(x >> 2)
}

#[test]
fn testpe_exports_run_natively() {
    let root: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("testpe-native");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let src = root.join("crates/victory-recomp/tests/fixtures/testpe.c");
    run(Command::new(root.join("tools/macos/gnullvm-cc.sh"))
        .args([
            "-O1",
            "-funwind-tables",
            "-fno-vectorize",
            "-fno-slp-vectorize",
            "-fno-builtin",
            "-fno-math-errno",
            "-ffreestanding",
            "-nostdlib",
            "-shared",
            "-Wl,--entry=DllMain",
            "-o",
        ])
        .arg(dir.join("testpe.dll"))
        .arg(&src));
    let pe = std::fs::read(dir.join("testpe.dll")).unwrap();
    let unit = lift_image(&Image::parse(&pe).unwrap()).unwrap();
    assert!(unit.failed.is_empty());
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

    let host = Host::load(&dir.join("libtitle.dylib"), &pe, &[]).unwrap();
    assert_eq!(host.call("t_add3", &[1, 2, 3]).unwrap(), 6);
    assert_eq!(host.call("t_fib", &[50]).unwrap(), 12_586_269_025);
    assert_eq!(host.call("t_gcd", &[1071, 462]).unwrap(), 21);
    assert_eq!(host.call("t_fact", &[10]).unwrap(), 3_628_800);
    assert_eq!(
        host.call("t_calls", &[7, 9]).unwrap(),
        helper(7) + helper(9) * 5
    );
    for x in 0..12 {
        let want = match x {
            0 => 11,
            1 => 107,
            2 => 2 ^ 0x55,
            3 => helper(3),
            4 => 4444,
            5 => 5 << 9,
            6 => !6u64,
            7 => 77,
            _ => x * 2,
        };
        assert_eq!(host.call("t_switch", &[x]).unwrap(), want, "t_switch({x})");
    }
    // Pointers into host memory work as they are (guest pointers are host pointers).
    let bytes: Vec<u8> = (1..=40).collect();
    let want: u64 = bytes
        .iter()
        .enumerate()
        .map(|(i, &b)| b as u64 * (i as u64 + 1))
        .sum();
    assert_eq!(
        host.call("t_sum_bytes", &[bytes.as_ptr() as u64, 40])
            .unwrap(),
        want
    );
    let (a, b): (Vec<f64>, Vec<f64>) = (
        (1..=8).map(f64::from).collect(),
        (1..=8).map(|x| f64::from(x) * 0.5).collect(),
    );
    let (_, x0) = host
        .call_ex("t_dot", &[a.as_ptr() as u64, b.as_ptr() as u64, 8], 0)
        .unwrap();
    assert_eq!(
        f64::from_bits(x0),
        a.iter().zip(&b).map(|(x, y)| x * y).sum::<f64>()
    );
    // Double arguments arrive in xmm0 and xmm1.
    let (_, x0) = host
        .call_ex("t_hyp", &[3.0f64.to_bits(), 4.0f64.to_bits()], 0b11)
        .unwrap();
    assert_eq!(f64::from_bits(x0), 5.0);
    assert_eq!(
        host.call_ex("t_itof", &[(-7i64) as u64], 0).unwrap().1,
        (-7.0f64).to_bits()
    );
    let (rax, _) = host
        .call_ex("t_ftoi", &[(-9.75f64).to_bits()], 0b1)
        .unwrap();
    assert_eq!(rax as i64, -9);
}
