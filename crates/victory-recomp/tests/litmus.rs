//! Memory-ordering litmus tests (message passing, store buffering, load buffering, IRIW) for lifted plain
//! loads and stores on arm64's weak model, against the x86-64 original under Rosetta 2 (hardware TSO), and a
//! micro-benchmark of what ordering the lifted accesses costs. See `docs/recompiler.md`, "Memory ordering".
//!
//! Builds compared: x86 under Rosetta; lifted plain (the default); lifted `-DRT_TSO` (shared accesses are
//! acquire loads and release stores; stack and `gs:` accesses stay plain); lifted `-DRT_TSO_ALL` (every access).
//! Forbidden outcomes must never appear on x86 and in the two TSO builds. Whether the plain build shows them
//! depends on the core and the timing, so the test reports counts and only fails on a forbidden outcome
//! where TSO is promised.
#![cfg(target_os = "macos")]
#![allow(unused_variables)]

#[macro_use]
mod common;

use std::collections::BTreeMap;

use common::*;
use iced_x86::code_asm::*;

fn lit(
    name: &'static str,
    f: impl Fn(&mut CodeAssembler) -> Result<(), IcedError> + 'static,
) -> Case {
    case("litmus", name, Kind::Int, move |a, _| {
        f(a)?;
        a.ret()
    })
}

fn cases() -> Vec<Case> {
    let mut v = vec![
        // Message passing: the writer publishes data (X) then a flag (Y); the reader loads the flag, then the data.
        lit("MP_W", |a| {
            a.mov(dword_ptr(rcx), 1)?;
            a.mov(dword_ptr(rdx), 1)
        }),
        lit("MP_R", |a| {
            a.mov(eax, dword_ptr(rdx))?;
            a.mov(r9d, dword_ptr(rcx))?;
            a.mov(dword_ptr(r8), eax)?;
            a.mov(dword_ptr(r8 + 4), r9d)
        }),
        // Store buffering: each thread stores to its variable then loads the other's.
        lit("SB_0", |a| {
            a.mov(dword_ptr(rcx), 1)?;
            a.mov(eax, dword_ptr(rdx))?;
            a.mov(dword_ptr(r8), eax)
        }),
        lit("SB_1", |a| {
            a.mov(dword_ptr(rdx), 1)?;
            a.mov(eax, dword_ptr(rcx))?;
            a.mov(dword_ptr(r8), eax)
        }),
        lit("SBF_0", |a| {
            a.mov(dword_ptr(rcx), 1)?;
            a.mfence()?;
            a.mov(eax, dword_ptr(rdx))?;
            a.mov(dword_ptr(r8), eax)
        }),
        lit("SBF_1", |a| {
            a.mov(dword_ptr(rdx), 1)?;
            a.mfence()?;
            a.mov(eax, dword_ptr(rcx))?;
            a.mov(dword_ptr(r8), eax)
        }),
        // Load buffering: load the other thread's variable, then store to ours.
        lit("LB_0", |a| {
            a.mov(eax, dword_ptr(rcx))?;
            a.mov(dword_ptr(rdx), 1)?;
            a.mov(dword_ptr(r8), eax)
        }),
        lit("LB_1", |a| {
            a.mov(eax, dword_ptr(rdx))?;
            a.mov(dword_ptr(rcx), 1)?;
            a.mov(dword_ptr(r8), eax)
        }),
        // Independent reads of independent writes: two readers must agree on the order of the two writes.
        lit("IRIW_W0", |a| a.mov(dword_ptr(rcx), 1)),
        lit("IRIW_W1", |a| a.mov(dword_ptr(rdx), 1)),
        lit("IRIW_R0", |a| {
            a.mov(eax, dword_ptr(rcx))?;
            a.mov(r9d, dword_ptr(rdx))?;
            a.mov(dword_ptr(r8), eax)?;
            a.mov(dword_ptr(r8 + 4), r9d)
        }),
        lit("IRIW_R1", |a| {
            a.mov(eax, dword_ptr(rdx))?;
            a.mov(r9d, dword_ptr(rcx))?;
            a.mov(dword_ptr(r8), eax)?;
            a.mov(dword_ptr(r8 + 4), r9d)
        }),
        // Benchmark kernels (rcx, rdx, r8 = a, b, n).
        lit("BENCH_COPY", |a| {
            let mut l = a.create_label();
            a.xor(r9d, r9d)?;
            a.set_label(&mut l)?;
            a.mov(eax, dword_ptr(rcx + r9 * 4))?;
            a.add(eax, 1)?;
            a.mov(dword_ptr(rdx + r9 * 4), eax)?;
            a.inc(r9)?;
            a.cmp(r9, r8)?;
            a.jb(l)
        }),
        lit("BENCH_STACK", |a| {
            let mut l = a.create_label();
            a.xor(r9d, r9d)?;
            a.set_label(&mut l)?;
            a.mov(qword_ptr(rsp - 16), r9)?;
            a.mov(rax, qword_ptr(rsp - 16))?;
            a.add(rax, 3)?;
            a.mov(qword_ptr(rsp - 24), rax)?;
            a.inc(r9)?;
            a.cmp(r9, r8)?;
            a.jb(l)
        }),
        lit("BENCH_CHASE", |a| {
            let mut l = a.create_label();
            a.nop()?;
            a.set_label(&mut l)?;
            a.mov(rcx, qword_ptr(rcx))?;
            a.dec(r8)?;
            a.jne(l)
        }),
        lit("BENCH_STORE", |a| {
            let mut l = a.create_label();
            a.xor(r9d, r9d)?;
            a.set_label(&mut l)?;
            a.mov(dword_ptr(rcx + r9 * 4), r9d)?;
            a.inc(r9)?;
            a.cmp(r9, r8)?;
            a.jb(l)
        }),
    ];
    v.shrink_to_fit();
    v
}

/// `name weak=<n> ...` lines as (name, weak count).
fn parse(out: &str) -> BTreeMap<String, u64> {
    out.lines()
        .filter_map(|l| {
            let (name, rest) = l.split_once(" weak=")?;
            Some((
                name.to_string(),
                rest.split_whitespace().next()?.parse().ok()?,
            ))
        })
        .collect()
}

fn clang_arm(dir: &std::path::Path, drv: &str, out: &str, flags: &[&str]) {
    run(std::process::Command::new("clang")
        .args([
            "-arch",
            "arm64",
            "-O2",
            "-ffp-contract=off",
            "-Wno-unused",
            "-DLIFTED",
            "-pthread",
        ])
        .args(flags)
        .arg("-I")
        .arg(dir)
        .arg(dir.join(drv))
        .arg(dir.join("unit.c"))
        .arg(dir.join("rt_host.c"))
        .arg("-o")
        .arg(dir.join(out)));
}

#[test]
fn litmus_tests_and_ordering_cost() {
    if !have_tools() {
        eprintln!("skipped: needs Rosetta 2 and clang");
        return;
    }
    let all = cases();
    let names: Vec<_> = all.iter().map(|c| c.name.clone()).collect();
    let (dir, spans) = prepare("litmus", &all, 0);
    write_runtime(&dir);
    let rvas: String = names
        .iter()
        .zip(&spans)
        .map(|(n, s)| format!("#define {n} {:#x}\n", s.0))
        .collect();
    std::fs::write(dir.join("rvas.h"), rvas).unwrap();
    let fixtures = repo_root().join("crates/victory-recomp/tests/fixtures");
    for f in ["litmus_driver.c", "bench_driver.c"] {
        std::fs::copy(fixtures.join(f), dir.join(f)).unwrap();
    }
    let iters = std::env::var("LITMUS_ITERS").unwrap_or_else(|_| "300000".into());

    run(std::process::Command::new("clang")
        .args(["-arch", "x86_64", "-O1", "-pthread", "-I"])
        .arg(&dir)
        .arg(dir.join("litmus_driver.c"))
        .arg("-o")
        .arg(dir.join("lit_x86")));
    clang_arm(&dir, "litmus_driver.c", "lit_plain", &[]);
    clang_arm(&dir, "litmus_driver.c", "lit_tso", &["-DRT_TSO"]);
    clang_arm(&dir, "litmus_driver.c", "lit_tso_all", &["-DRT_TSO_ALL"]);

    let go = |bin: &str| {
        let out = run(std::process::Command::new(dir.join(bin))
            .arg("image.bin")
            .arg(&iters)
            .current_dir(&dir));
        eprintln!("--- {bin} ({iters} iterations each)\n{out}");
        parse(&out)
    };
    let x86 = go("lit_x86");
    let plain = go("lit_plain");
    let tso = go("lit_tso");
    let tso_all = go("lit_tso_all");

    // The compiled code really differs: release stores / acquire loads only in the TSO builds.
    let asm = |flags: &[&str]| {
        run(std::process::Command::new("clang")
            .args([
                "-arch",
                "arm64",
                "-O2",
                "-ffp-contract=off",
                "-Wno-unused",
                "-S",
                "-o",
                "-",
            ])
            .args(flags)
            .arg("-I")
            .arg(&dir)
            .arg(dir.join("unit.c")))
    };
    let count = |s: &str, pat: &[&str]| {
        s.lines()
            .filter(|l| pat.iter().any(|p| l.trim_start().starts_with(p)))
            .count()
    };
    let (a_plain, a_tso) = (asm(&[]), asm(&["-DRT_TSO"]));
    assert_eq!(
        count(&a_plain, &["stlr", "ldar", "ldapr"]),
        0,
        "plain build has ordered accesses"
    );
    assert!(
        count(&a_tso, &["stlr"]) >= 8 && count(&a_tso, &["ldar", "ldapr"]) >= 8,
        "TSO build lacks ordered accesses"
    );

    // x86 under Rosetta and the TSO builds: the forbidden outcomes never happen.
    for (build, r) in [
        ("x86 (Rosetta)", &x86),
        ("lifted RT_TSO", &tso),
        ("lifted RT_TSO_ALL", &tso_all),
    ] {
        for t in ["MP", "SB+mfence", "LB", "IRIW"] {
            assert_eq!(
                r.get(t).copied(),
                Some(0),
                "{build}: forbidden outcome of {t} observed"
            );
        }
    }
    eprintln!("plain lifted weak outcomes: {plain:?}");
    eprintln!(
        "store buffering (allowed on TSO): x86 {:?}, TSO build {:?}",
        x86.get("SB"),
        tso.get("SB")
    );

    // Cost: dependent loads, stores, copy and stack traffic under the three builds.
    let bench = |flags: &[&str], bin: &str| {
        clang_arm(&dir, "bench_driver.c", bin, flags);
        let out = run(std::process::Command::new(dir.join(bin))
            .arg("image.bin")
            .current_dir(&dir));
        eprintln!("--- {bin}\n{out}");
        out
    };
    bench(&[], "bench_plain");
    bench(&["-DRT_TSO"], "bench_tso");
    bench(&["-DRT_TSO_ALL"], "bench_tso_all");
}
