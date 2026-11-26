//! Differential-test machinery shared by `diff.rs` and friends: assemble snippets,
//! lift them, run the x86-64 original under Rosetta 2 and the lifted arm64 build,
//! and compare every output line.
#![allow(dead_code, unused_variables)]

use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;

use super::*;
use iced_x86::code_asm::*;
use iced_x86::{BlockEncoderOptions, Decoder, DecoderOptions};
use victory_recomp::analyze::find_jump_tables_with;
use victory_recomp::image::Func;
use victory_recomp::lift::{Ctx, FuncIn, lift_unit};

pub const BASE: u64 = 0x1_4000_0000;

#[derive(Clone, Copy, PartialEq)]
pub enum Kind {
    Int,
    Fp,
    /// Integer-register inputs that are packed random/special float32 pairs.
    Fl,
    /// Integer-register inputs that are random/special float64 bit patterns.
    Db,
    /// Only called from other snippets.
    Helper,
}

/// Labels shared between the case bodies and the table data.
pub struct Env {
    pub helper: CodeLabel,
    pub table: CodeLabel,
    pub jt: Vec<CodeLabel>,
}

pub type Build = Box<dyn Fn(&mut CodeAssembler, &mut Env) -> Result<(), IcedError>>;
pub struct Case {
    pub name: String,
    pub kind: Kind,
    pub build: Build,
    pub class: &'static str,
}

pub fn case(
    class: &'static str,
    name: impl Into<String>,
    kind: Kind,
    f: impl Fn(&mut CodeAssembler, &mut Env) -> Result<(), IcedError> + 'static,
) -> Case {
    Case {
        name: name.into(),
        kind,
        build: Box::new(f),
        class,
    }
}

pub fn fl(a: &mut CodeAssembler, m: &str) -> Result<(), IcedError> {
    for c in m.chars() {
        match c {
            'c' => a.setb(byte_ptr(r8))?,
            'z' => a.sete(byte_ptr(r8 + 1))?,
            's' => a.sets(byte_ptr(r8 + 2))?,
            'o' => a.seto(byte_ptr(r8 + 3))?,
            'p' => a.setp(byte_ptr(r8 + 4))?,
            _ => unreachable!(),
        }
    }
    Ok(())
}

pub fn done(a: &mut CodeAssembler) -> Result<(), IcedError> {
    a.mov(qword_ptr(r8 + 8), rax)?;
    a.ret()
}

#[macro_export]
macro_rules! sizes {
    ($v:expr, $class:expr, $nm:expr, [$($bits:literal $x:ident $y:ident $z:ident $mp:ident),*], |$a:ident, $ra:ident, $rc:ident, $rd:ident, $mp_:ident| $body:expr) => {
        $( {
            let f = move |$a: &mut CodeAssembler, _e: &mut Env| -> Result<(), IcedError> {
                let ($ra, $rc, $rd) = ($x, $y, $z);
                let $mp_: fn(AsmMemoryOperand) -> AsmMemoryOperand = $mp;
                $body
            };
            $v.push(case($class, format!("{}{}", $nm, $bits), Kind::Int, f));
        } )*
    };
}
#[macro_export]
macro_rules! all_sizes {
    ($v:expr, $class:expr, $nm:expr, |$a:ident, $ra:ident, $rc:ident, $rd:ident, $mp:ident| $body:expr) => {
        sizes!($v, $class, $nm, [8 al cl dl byte_ptr, 16 ax cx dx word_ptr, 32 eax ecx edx dword_ptr, 64 rax rcx rdx qword_ptr], |$a, $ra, $rc, $rd, $mp| $body)
    };
}
#[macro_export]
macro_rules! big_sizes {
    ($v:expr, $class:expr, $nm:expr, |$a:ident, $ra:ident, $rc:ident, $rd:ident, $mp:ident| $body:expr) => {
        sizes!($v, $class, $nm, [32 eax ecx edx dword_ptr, 64 rax rcx rdx qword_ptr], |$a, $ra, $rc, $rd, $mp| $body)
    };
}

pub fn assemble(cases: &[Case]) -> (Vec<u8>, Vec<(u32, u32)>) {
    let mut table_vals: Vec<u32> = vec![0; 6];
    for pass in 0..2 {
        let mut a = CodeAssembler::new(64).unwrap();
        let mut env = Env {
            helper: a.create_label(),
            table: a.create_label(),
            jt: (0..6).map(|_| a.create_label()).collect(),
        };
        let mut spans = vec![];
        for c in cases {
            let mut start = if c.kind == Kind::Helper {
                env.helper
            } else {
                a.create_label()
            };
            let mut end = a.create_label();
            a.set_label(&mut start).unwrap();
            (c.build)(&mut a, &mut env).unwrap_or_else(|e| panic!("assembling {}: {e}", c.name));
            a.set_label(&mut end).unwrap();
            a.int3().unwrap();
            a.nop().unwrap();
            spans.push((start, end));
        }
        if !cases.iter().any(|c| c.kind == Kind::Helper) {
            a.set_label(&mut env.helper).unwrap();
            a.nop().unwrap();
        }
        // Table data lives after all code, like .rdata, outside every function.
        a.set_label(&mut env.table).unwrap();
        a.dd(&table_vals).unwrap();
        let res = a
            .assemble_options(BASE, BlockEncoderOptions::RETURN_NEW_INSTRUCTION_OFFSETS)
            .unwrap();
        let ip = |l: &CodeLabel| res.label_ip(l).unwrap();
        if pass == 0 {
            let t = ip(&env.table);
            table_vals = env
                .jt
                .iter()
                .map(|l| res.label_ip(l).map_or(0, |x| x.wrapping_sub(t)) as u32)
                .collect();
        } else {
            let spans = spans
                .iter()
                .map(|(s, e)| ((ip(s) - BASE) as u32, (ip(e) - BASE) as u32))
                .collect();
            return (res.inner.code_buffer, spans);
        }
    }
    unreachable!()
}

pub fn driver_c(calls: &[(String, u32, Kind)]) -> String {
    let mut s = String::from(
        r#"#include "harness.h"
static u64 ig(void) {
  switch (rnd() % 10) {
  case 0: return 0;
  case 1: return rnd() % 4;
  case 2: return ~0ull - rnd() % 3;
  case 3: return 0x8000000000000000ull + rnd() % 3 - 1;
  case 4: return 0x7fffffffull + rnd() % 3 - 1;
  case 5: return 0x80000000ull + rnd() % 3 - 1;
  case 6: return rnd() & 0xff;
  case 7: return rnd() & 0xffff;
  default: return rnd();
  }
}
static double dg(void) {
  static const u64 sp[] = { 0, 0x8000000000000000ull, 0x3ff0000000000000ull, 0xbff0000000000000ull, 0x7ff0000000000000ull, 0xfff0000000000000ull,
    0x7ff8000000000000ull, 0xfff8000000000000ull, 1, 0x000fffffffffffffull, 0x7fefffffffffffffull, 0x3fe0000000000000ull, 0x4340000000000000ull,
    0xc3e0000000000000ull, 0x41dfffffffc00000ull, 0xc1e0000000000000ull };
  u64 b;
  switch (rnd() % 4) {
  case 0: b = sp[rnd() % 16]; break;
  case 1: { double d = (double)(s64)(rnd() % 2001) - 1000; memcpy(&b, &d, 8); break; }
  case 2: { double d = (double)(s64)rnd() / (double)(1 + rnd() % 100000); memcpy(&b, &d, 8); break; }
  default: b = rnd();
  }
  double d; memcpy(&d, &b, 8);
  if (d != d) { b = (b & 0x8000000000000000ull) | 0x7ff8000000000000ull; memcpy(&d, &b, 8); } /* no signalling NaNs: ARM and x86 order them differently */
  return d;
}
static u32 fg32(void) {
  static const u32 sp[] = { 0, 0x80000000u, 0x3f800000u, 0xbf800000u, 0x7f800000u, 0xff800000u, 0x7fc00000u, 0xffc00000u, 1, 0x007fffffu,
    0x7f7fffffu, 0x3f000000u, 0x4b800000u, 0xcf000000u, 0x4effffffu, 0x5a000000u, 0x00800000u, 0x80000001u, 0x3fc00000u, 0x40490fdbu };
  u32 b;
  switch (rnd() % 4) {
  case 0: b = sp[rnd() % 20]; break;
  case 1: { float f = (float)((s64)(rnd() % 2001) - 1000); memcpy(&b, &f, 4); break; }
  case 2: { float f = (float)(s64)rnd() / (float)(1 + rnd() % 100000); memcpy(&b, &f, 4); break; }
  default: b = (u32)rnd();
  }
  float f; memcpy(&f, &b, 4);
  if (f != f) b = (b & 0x80000000u) | 0x7fc00000u; /* no signalling NaNs */
  return b;
}
static u64 fg(void) { u64 lo = fg32(); return lo | (u64)fg32() << 32; }
static u64 dgbits(void) { double d = dg(); u64 b; memcpy(&b, &d, 8); return b; }
static void show(const char *n, const u8 *o) {
  u64 w[32]; memcpy(w, o, 256);
  printf("%s", n); for (int k = 0; k < 32; k++) printf(" %llx", (unsigned long long)w[k]); printf("\n");
}
int main(int argc, char **argv) {
  (void)argc; setvbuf(stdout, 0, _IOLBF, 0); load_image(argv[1]);
  u8 out[256];
  for (int it = 0; it < 300; it++) {
    u64 a = ig(), b = ig(); if (rnd() % 5 == 0) b = a; else if (rnd() % 9 == 0) b = a + 1;
    double x = dg(), y = dg(); if (rnd() % 6 == 0) y = x;
    u64 fa = fg(), fb = fg(); if (rnd() % 6 == 0) fb = fa; u64 da = dgbits(), db = dgbits(); if (rnd() % 6 == 0) db = da;
    (void)a; (void)b; (void)x; (void)y; (void)fa; (void)fb; (void)da; (void)db;
"#,
    );
    for (name, rva, kind) in calls {
        match kind {
            Kind::Int => writeln!(s, "    memset(out, 0, 256); fi({rva:#x}, a, b, out); printf(\"%llx %llx \", (unsigned long long)a, (unsigned long long)b); show(\"{name}\", out);").unwrap(),
            Kind::Fl => writeln!(s, "    memset(out, 0, 256); fi({rva:#x}, fa, fb, out); printf(\"%llx %llx \", (unsigned long long)fa, (unsigned long long)fb); show(\"{name}\", out);").unwrap(),
            Kind::Db => writeln!(s, "    memset(out, 0, 256); fi({rva:#x}, da, db, out); printf(\"%llx %llx \", (unsigned long long)da, (unsigned long long)db); show(\"{name}\", out);").unwrap(),
            Kind::Fp => writeln!(s, "    memset(out, 0, 256); fd({rva:#x}, x, y, out); printf(\"%016llx %016llx \", *(unsigned long long *)&x, *(unsigned long long *)&y); show(\"{name}\", out);").unwrap(),
            Kind::Helper => {}
        }
    }
    s.push_str("  }\n  return 0;\n}\n");
    s
}

/// Assembles, lifts and runs `cases` as x86-64 (Rosetta 2) and arm64, and compares every output line.
/// `tables` is the number of jump tables the cases contain.
pub fn run_cases(label: &str, cases: Vec<Case>, tables: usize) {
    if !have_tools() {
        eprintln!("skipped: needs Rosetta 2 and clang");
        return;
    }
    let (dir, spans) = prepare(label, &cases, tables);
    let (x86, arm) = run_both(&dir, "driver.c", "unit.c", &["image.bin"]);
    compare(label, &cases, &x86, &arm);
}

/// One result line of a snippet: the two inputs and the 256 output bytes as 32 qwords.
pub struct Line {
    pub name: String,
    pub a: u64,
    pub b: u64,
    pub w: [u64; 32],
}

/// Lifts `cases` and runs only the arm64 build (for instructions Rosetta 2 cannot run), 300 inputs each.
pub fn run_arm_only(label: &str, cases: Vec<Case>) -> Vec<Line> {
    let (dir, _) = prepare(label, &cases, 0);
    write_runtime(&dir);
    run(Command::new("clang")
        .args([
            "-arch",
            "arm64",
            "-O2",
            "-ffp-contract=off",
            "-Wno-unused",
            "-DLIFTED",
            "-I",
        ])
        .arg(&dir)
        .arg(dir.join("driver.c"))
        .arg(dir.join("unit.c"))
        .arg(dir.join("rt_host.c"))
        .arg("-o")
        .arg(dir.join("drv_arm")));
    let out = run(Command::new(dir.join("drv_arm"))
        .arg("image.bin")
        .current_dir(&dir));
    out.lines()
        .map(|l| {
            let t: Vec<&str> = l.split_whitespace().collect();
            let h = |s: &str| u64::from_str_radix(s, 16).unwrap();
            let mut w = [0u64; 32];
            for (k, x) in t[3..35].iter().enumerate() {
                w[k] = h(x);
            }
            Line {
                name: t[2].to_string(),
                a: h(t[0]),
                b: h(t[1]),
                w,
            }
        })
        .collect()
}

/// Assembles and lifts `cases` into `target/tmp/<label>`; returns the directory and snippet spans.
pub fn prepare(
    label: &str,
    cases: &[Case],
    tables: usize,
) -> (std::path::PathBuf, Vec<(u32, u32)>) {
    let (blob, spans) = assemble(cases);

    // Decode, find jump tables, lift.
    let read_u32 = |rva: u32| {
        blob.get(rva as usize..rva as usize + 4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
    };
    let decoded: Vec<_> = spans
        .iter()
        .map(|&(b, e)| {
            let mut d = Decoder::with_ip(
                64,
                &blob[b as usize..e as usize],
                BASE + b as u64,
                DecoderOptions::NONE,
            );
            let ins: Vec<_> = d.iter().collect();
            assert!(ins.iter().all(|i| !i.is_invalid()));
            let f = Func {
                begin: b,
                end: e,
                unwind: 0,
                chained: false,
            };
            let tables: BTreeMap<u64, Vec<u64>> = find_jump_tables_with(BASE, &read_u32, &f, &ins)
                .iter()
                .map(|t| {
                    (
                        BASE + t.jmp_rva as u64,
                        t.targets.iter().map(|&r| BASE + r as u64).collect(),
                    )
                })
                .collect();
            (b, e, ins, tables)
        })
        .collect();
    assert_eq!(
        decoded.iter().map(|d| d.3.len()).sum::<usize>(),
        tables,
        "unexpected number of recovered jump tables"
    );
    let starts: std::collections::HashSet<u64> = spans.iter().map(|s| BASE + s.0 as u64).collect();
    let is_func = |va: u64| starts.contains(&va);
    let fins: Vec<FuncIn> = decoded
        .iter()
        .map(|(b, e, ins, t)| FuncIn {
            begin_va: BASE + *b as u64,
            end_va: BASE + *e as u64,
            ins,
            tables: t.clone(),
            extra: vec![],
            landing_pads: vec![],
        })
        .collect();
    let unit = lift_unit(
        &Ctx {
            image_base: BASE,
            is_func: &is_func,
        },
        &fins,
    );
    let names: HashMap<u64, &str> = cases
        .iter()
        .zip(&spans)
        .map(|(c, s)| (BASE + s.0 as u64, c.name.as_str()))
        .collect();
    let failed: Vec<_> = unit
        .failed
        .iter()
        .map(|(va, e)| format!("{}: {e}", names[va]))
        .collect();
    assert!(
        failed.is_empty(),
        "unlifted snippets:\n{}",
        failed.join("\n")
    );

    let dir = workdir(label);
    std::fs::write(dir.join("image.bin"), &blob).unwrap();
    std::fs::write(dir.join("unit.c"), &unit.c).unwrap();
    let calls: Vec<_> = cases
        .iter()
        .zip(&spans)
        .map(|(c, s)| (c.name.clone(), s.0, c.kind))
        .collect();
    std::fs::write(dir.join("driver.c"), driver_c(&calls)).unwrap();
    (dir, spans)
}

fn compare(label: &str, cases: &[Case], x86: &str, arm: &str) {
    // Report per class so a failure names the instruction.
    let class_of: HashMap<&str, &str> = cases.iter().map(|c| (c.name.as_str(), c.class)).collect();
    let mut bad: BTreeMap<String, (usize, String, String)> = BTreeMap::new();
    for (l, r) in x86.lines().zip(arm.lines()) {
        if l != r {
            let name = l.split_whitespace().nth(2).unwrap_or("?").to_string();
            let e = bad.entry(name).or_insert((0, l.to_string(), r.to_string()));
            e.0 += 1;
        }
    }
    assert_eq!(x86.lines().count(), arm.lines().count());
    let report: Vec<String> = bad
        .iter()
        .map(|(n, (c, l, r))| {
            format!(
                "[{}] {n}: {c} mismatches\n  x86: {l}\n  arm: {r}",
                class_of.get(n.as_str()).unwrap_or(&"?")
            )
        })
        .collect();
    assert!(
        bad.is_empty(),
        "lifted output differs from x86:\n{}",
        report.join("\n")
    );
    let classes: std::collections::BTreeSet<_> = cases.iter().map(|c| c.class).collect();
    eprintln!(
        "{label}: {} snippets in {} classes x 300 random inputs matched: {classes:?}",
        cases.len(),
        classes.len()
    );
}

// ---- the ymm prelude shared by the vector diff tests ----
#[derive(Clone, Copy)]
pub enum Pre {
    /// Integer inputs: four qwords per buffer derived from a and b.
    Int,
    /// Doubles: the inputs are random and special float64 bit patterns.
    FpD,
    /// Floats: the inputs are pairs of random and special float32 bit patterns.
    FpS,
}

pub fn pre(a: &mut CodeAssembler, p: Pre) -> Result<(), IcedError> {
    match p {
        Pre::FpS | Pre::FpD => {
            // Lanes are the inputs and their half-swapped forms, so every lane stays a "real" float.
            a.mov(r9, rcx)?;
            a.ror(r9, 32)?;
            a.mov(r10, rdx)?;
            a.ror(r10, 32)?;
            for (off, q) in [
                (128, [rcx, rdx, r9, r10]),
                (160, [rdx, rcx, r10, r9]),
                (192, [r9, rdx, rcx, r10]),
            ] {
                for (l, r) in q.iter().enumerate() {
                    a.mov(qword_ptr(r8 + off + 8 * l as i32), *r)?;
                }
            }
        }
        Pre::Int => {
            a.mov(qword_ptr(r8 + 128), rcx)?;
            a.mov(qword_ptr(r8 + 136), rdx)?;
            a.lea(rax, ptr(rcx + rdx * 4 + 3))?;
            a.mov(qword_ptr(r8 + 144), rax)?;
            a.mov(rax, rcx)?;
            a.xor(rax, rdx)?;
            a.ror(rax, 13)?;
            a.mov(qword_ptr(r8 + 152), rax)?;
            a.mov(qword_ptr(r8 + 160), rdx)?;
            a.mov(qword_ptr(r8 + 168), rcx)?;
            a.mov(rax, rcx)?;
            a.xor(rax, rdx)?;
            a.not(rax)?;
            a.mov(qword_ptr(r8 + 176), rax)?;
            a.mov(rax, rcx)?;
            a.ror(rax, 19)?;
            a.sub(rax, rdx)?;
            a.mov(qword_ptr(r8 + 184), rax)?;
            a.mov(rax, rdx)?;
            a.rol(rax, 9)?;
            a.xor(rax, rcx)?;
            a.mov(qword_ptr(r8 + 192), rax)?;
            a.mov(rax, rcx)?;
            a.sub(rax, rdx)?;
            a.mov(qword_ptr(r8 + 200), rax)?;
            a.mov(rax, qword_ptr(r8 + 152))?;
            a.mov(qword_ptr(r8 + 208), rax)?;
            a.mov(rax, rdx)?;
            a.ror(rax, 11)?;
            a.mov(qword_ptr(r8 + 216), rax)?;
        }
    }
    a.vmovups(ymm1, ymmword_ptr(r8 + 128))?;
    a.vmovups(ymm2, ymmword_ptr(r8 + 160))?;
    a.vmovups(ymm3, ymmword_ptr(r8 + 192))?;
    a.vmovups(ymm0, ymmword_ptr(r8 + 192))?;
    a.vmovups(ymm4, ymmword_ptr(r8 + 192))?;
    a.vmovups(ymm5, ymmword_ptr(r8 + 160))?;
    a.xor(eax, eax)?;
    Ok(())
}

pub fn vcase(
    v: &mut Vec<Case>,
    class: &'static str,
    name: String,
    p: Pre,
    body: impl Fn(&mut CodeAssembler) -> Result<(), IcedError> + 'static,
) {
    let kind = match p {
        Pre::Int => Kind::Int,
        Pre::FpS => Kind::Fl,
        Pre::FpD => Kind::Db,
    };
    v.push(case(class, name, kind, move |a, _| {
        pre(a, p)?;
        body(a)?;
        a.vmovups(ymmword_ptr(r8 + 64), ymm0)?;
        a.vmovups(ymmword_ptr(r8 + 96), ymm4)?;
        a.mov(qword_ptr(r8 + 8), rax)?;
        a.ret()
    }));
}
