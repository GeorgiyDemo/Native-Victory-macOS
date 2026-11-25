//! Shared helpers for the macOS-only tests: build C with the system clang, run
//! the x86-64 flavour under Rosetta 2 and the lifted arm64 flavour natively.
#![cfg(target_os = "macos")]
#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::process::Command;

pub fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

pub fn workdir(name: &str) -> PathBuf {
    let d = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Runs a command, panicking with its output on failure.
pub fn run(cmd: &mut Command) -> String {
    let out = cmd.output().unwrap_or_else(|e| panic!("spawn {cmd:?}: {e}"));
    assert!(
        out.status.success(),
        "{cmd:?} failed ({}):\n{}{}",
        out.status,
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// True when the toolchain pieces the tests need are present (Rosetta, clang).
pub fn have_tools() -> bool {
    Command::new("arch").args(["-x86_64", "/usr/bin/true"]).status().is_ok_and(|s| s.success())
        && Command::new("clang").arg("--version").output().is_ok()
}

pub fn write_runtime(dir: &Path) {
    std::fs::write(dir.join("rt.h"), victory_recomp::lift::RT_HEADER).unwrap();
    std::fs::write(dir.join("rt_host.c"), victory_recomp::RT_HOST_C).unwrap();
    std::fs::write(dir.join("harness.h"), victory_recomp::HARNESS_H).unwrap();
}

/// Builds `driver.c` twice (x86-64 original, arm64 + lifted unit) and returns both outputs.
pub fn run_both(dir: &Path, driver: &str, unit: &str, args: &[&str]) -> (String, String) {
    write_runtime(dir);
    let (drv, unit_c) = (dir.join(driver), dir.join(unit));
    run(Command::new("clang").args(["-arch", "x86_64", "-O1", "-I"]).arg(dir).arg(&drv).arg("-o").arg(dir.join("drv_x86")));
    run(Command::new("clang")
        .args(["-arch", "arm64", "-O2", "-ffp-contract=off", "-Wno-unused", "-DLIFTED", "-I"])
        .arg(dir)
        .arg(&drv)
        .arg(&unit_c)
        .arg(dir.join("rt_host.c"))
        .arg("-o")
        .arg(dir.join("drv_arm")));
    let x86 = run(Command::new(dir.join("drv_x86")).args(args).current_dir(dir));
    let arm = run(Command::new(dir.join("drv_arm")).args(args).current_dir(dir));
    (x86, arm)
}
