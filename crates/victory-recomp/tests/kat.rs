//! Known-answer tests for instructions Rosetta 2 cannot run (so `diff*.rs` cannot compare them):
//! the 256-bit VAES and VPCLMULQDQ forms, SSE4a `insertq`/`extrq`, and `adcx`/`adox` (SIGILL under
//! Rosetta). Only the lifted arm64 build runs. Oracles: FIPS-197 appendix C.1 for AES-128 (key
//! schedule with `aeskeygenassist`, encrypt, `aesimc` decrypt), a bitwise carry-less multiply and the
//! SDM/AMD definitions written out in Rust, and, for the ymm forms, the already differentially tested
//! xmm forms applied to each half.
#![cfg(target_os = "macos")]
#![allow(unused_variables)]

#[macro_use]
mod common;

use common::*;
use iced_x86::code_asm::*;

fn clmul(a: u64, b: u64) -> u128 {
    (0..64)
        .filter(|i| b >> i & 1 == 1)
        .fold(0u128, |r, i| r ^ (a as u128) << i)
}

fn store_u64s(a: &mut CodeAssembler, off: i32, q: [u64; 2]) -> Result<(), IcedError> {
    for (k, v) in q.iter().enumerate() {
        a.mov(rax, *v)?;
        a.mov(qword_ptr(r8 + off + 8 * k as i32), rax)?;
    }
    Ok(())
}

/// AES-128 as in Intel's AES-NI white paper: expand the key into xmm4..xmm14, encrypt xmm0, store the
/// ciphertext at out+64, decrypt it back with `aesimc` + `aesdec` and store the result at out+96.
fn aes128(a: &mut CodeAssembler, vex: bool) -> Result<(), IcedError> {
    store_u64s(a, 128, [0x0706050403020100, 0x0f0e0d0c0b0a0908])?;
    store_u64s(a, 144, [0x7766554433221100, 0xffeeddccbbaa9988])?;
    a.movdqu(xmm1, xmmword_ptr(r8 + 128))?;
    a.movdqu(xmm0, xmmword_ptr(r8 + 144))?;
    let rk = [
        xmm4, xmm5, xmm6, xmm7, xmm8, xmm9, xmm10, xmm11, xmm12, xmm13, xmm14,
    ];
    a.movdqa(rk[0], xmm1)?;
    macro_rules! step {
        ($rcon:expr) => {
            if vex {
                a.vaeskeygenassist(xmm2, xmm1, $rcon)?;
                a.vpshufd(xmm2, xmm2, 0xff)?;
                a.vpslldq(xmm3, xmm1, 4)?;
                a.vpxor(xmm1, xmm1, xmm3)?;
                a.vpslldq(xmm3, xmm3, 4)?;
                a.vpxor(xmm1, xmm1, xmm3)?;
                a.vpslldq(xmm3, xmm3, 4)?;
                a.vpxor(xmm1, xmm1, xmm3)?;
                a.vpxor(xmm1, xmm1, xmm2)?;
            } else {
                a.aeskeygenassist(xmm2, xmm1, $rcon)?;
                a.pshufd(xmm2, xmm2, 0xff)?;
                a.movdqa(xmm3, xmm1)?;
                a.pslldq(xmm3, 4)?;
                a.pxor(xmm1, xmm3)?;
                a.pslldq(xmm3, 4)?;
                a.pxor(xmm1, xmm3)?;
                a.pslldq(xmm3, 4)?;
                a.pxor(xmm1, xmm3)?;
                a.pxor(xmm1, xmm2)?;
            }
        };
    }
    for (r, rc) in [1, 2, 4, 8, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36]
        .into_iter()
        .enumerate()
    {
        step!(rc);
        a.movdqa(rk[r + 1], xmm1)?;
    }
    a.pxor(xmm0, rk[0])?;
    for k in &rk[1..10] {
        if vex {
            a.vaesenc(xmm0, xmm0, *k)?
        } else {
            a.aesenc(xmm0, *k)?
        }
    }
    if vex {
        a.vaesenclast(xmm0, xmm0, rk[10])?
    } else {
        a.aesenclast(xmm0, rk[10])?
    }
    a.movdqu(xmmword_ptr(r8 + 64), xmm0)?;
    a.pxor(xmm0, rk[10])?;
    for k in rk[1..10].iter().rev() {
        if vex {
            a.vaesimc(xmm1, *k)?;
            a.vaesdec(xmm0, xmm0, xmm1)?;
        } else {
            a.aesimc(xmm1, *k)?;
            a.aesdec(xmm0, xmm1)?;
        }
    }
    if vex {
        a.vaesdeclast(xmm0, xmm0, rk[0])?
    } else {
        a.aesdeclast(xmm0, rk[0])?
    }
    a.movdqu(xmmword_ptr(r8 + 96), xmm0)?;
    a.ret()
}

#[test]
fn aes128_fips197_known_answer() {
    if !have_tools() {
        return;
    }
    let cases = vec![
        case("kat", "aes_legacy", Kind::Int, |a, _| aes128(a, false)),
        case("kat", "aes_vex", Kind::Int, |a, _| aes128(a, true)),
    ];
    let lines = run_arm_only("kat_aes", cases);
    assert_eq!(lines.len(), 600);
    for l in &lines {
        assert_eq!(
            [l.w[8], l.w[9]],
            [0x30047b6ad8e0c469, 0x5ac5b47080b7cdd8],
            "{} ciphertext",
            l.name
        );
        assert_eq!(
            [l.w[12], l.w[13]],
            [0x7766554433221100, 0xffeeddccbbaa9988],
            "{} decrypted",
            l.name
        );
    }
}

/// A ymm instruction against the same xmm instruction on each half.
#[test]
fn vaes_and_vpclmulqdq_ymm_equal_two_xmm() {
    if !have_tools() {
        return;
    }
    let mut v = vec![];
    macro_rules! y {
        ($($op:ident),*) => { $(
            vcase(&mut v, "kat", stringify!($op).to_string(), Pre::Int, |a| {
                a.$op(ymm0, ymm1, ymm2)?;
                a.$op(xmm6, xmm1, xmm2)?;
                a.vextracti128(xmm7, ymm1, 1)?;
                a.vextracti128(xmm8, ymm2, 1)?;
                a.$op(xmm7, xmm7, xmm8)?;
                a.vinserti128(ymm4, ymm6, xmm7, 1)
            });
        )* };
    }
    y!(vaesenc, vaesenclast, vaesdec, vaesdeclast);
    for imm in [0x00, 0x01, 0x10, 0x11] {
        vcase(
            &mut v,
            "kat",
            format!("vpclmulqdq{imm}"),
            Pre::Int,
            move |a| {
                a.vpclmulqdq(ymm0, ymm1, ymm2, imm)?;
                a.vpclmulqdq(xmm6, xmm1, xmm2, imm)?;
                a.vextracti128(xmm7, ymm1, 1)?;
                a.vextracti128(xmm8, ymm2, 1)?;
                a.vpclmulqdq(xmm7, xmm7, xmm8, imm)?;
                a.vinserti128(ymm4, ymm6, xmm7, 1)
            },
        );
    }
    let mut nonzero = 0;
    for l in run_arm_only("kat_ymm", v) {
        // ymm0 is stored at out+64 (w[8..12]), the two-xmm composition (ymm4) at out+96 (w[12..16]).
        assert_eq!(
            l.w[8..12],
            l.w[12..16],
            "{} (inputs {:#x} {:#x})",
            l.name,
            l.a,
            l.b
        );
        nonzero += l.w[8..12].iter().any(|&x| x != 0) as usize;
    }
    assert!(nonzero > 2000);
}

#[test]
fn pclmulqdq_matches_bitwise_reference() {
    if !have_tools() {
        return;
    }
    let case_for = |name: &'static str, imm: i32| {
        case("kat", name, Kind::Int, move |a, _| {
            a.movq(xmm0, rcx)?;
            a.pinsrq(xmm0, rdx, 1)?;
            a.movq(xmm1, rdx)?;
            a.pinsrq(xmm1, rcx, 1)?;
            a.pclmulqdq(xmm0, xmm1, imm)?;
            a.movdqu(xmmword_ptr(r8 + 64), xmm0)?;
            a.ret()
        })
    };
    let cases = vec![
        case_for("c00", 0x00),
        case_for("c01", 0x01),
        case_for("c10", 0x10),
        case_for("c11", 0x11),
    ];
    for l in run_arm_only("kat_clmul", cases) {
        // xmm0 = (a, b), xmm1 = (b, a): imm bit 0 picks xmm0's qword, bit 4 xmm1's.
        let (x, y) = match l.name.as_str() {
            "c00" => (l.a, l.b),
            "c01" => (l.b, l.b),
            "c10" => (l.a, l.a),
            _ => (l.b, l.a),
        };
        let want = clmul(x, y);
        assert_eq!(
            [l.w[8], l.w[9]],
            [want as u64, (want >> 64) as u64],
            "{} {:#x} {:#x}",
            l.name,
            l.a,
            l.b
        );
    }
}

#[test]
fn sse4a_insertq_extrq_known_answers() {
    if !have_tools() {
        return;
    }
    let mut cases = vec![];
    // (len, idx) immediates; len 0 means 64.
    let shapes = [
        (8u8, 4u8),
        (16, 8),
        (1, 63),
        (32, 0),
        (0, 0),
        (63, 1),
        (5, 17),
    ];
    for (k, &(len, idx)) in shapes.iter().enumerate() {
        cases.push(case(
            "kat",
            format!("insq_imm{k}"),
            Kind::Int,
            move |a, _| {
                a.movq(xmm0, rcx)?;
                a.movq(xmm1, rdx)?;
                a.insertq_4(xmm0, xmm1, len as i32, idx as i32)?;
                a.movq(qword_ptr(r8 + 64), xmm0)?;
                a.ret()
            },
        ));
        cases.push(case(
            "kat",
            format!("insq_reg{k}"),
            Kind::Int,
            move |a, _| {
                a.movq(xmm0, rcx)?;
                a.movq(xmm1, rdx)?;
                a.mov(rax, (len as u64) | (idx as u64) << 8)?;
                a.pinsrq(xmm1, rax, 1)?;
                a.insertq(xmm0, xmm1)?;
                a.movq(qword_ptr(r8 + 64), xmm0)?;
                a.ret()
            },
        ));
        cases.push(case(
            "kat",
            format!("extq_imm{k}"),
            Kind::Int,
            move |a, _| {
                a.movq(xmm0, rcx)?;
                a.extrq_3(xmm0, len as i32, idx as i32)?;
                a.movq(qword_ptr(r8 + 64), xmm0)?;
                a.ret()
            },
        ));
        cases.push(case(
            "kat",
            format!("extq_reg{k}"),
            Kind::Int,
            move |a, _| {
                a.movq(xmm0, rcx)?;
                a.mov(rax, (len as u64) | (idx as u64) << 8)?;
                a.movq(xmm1, rax)?;
                a.extrq(xmm0, xmm1)?;
                a.movq(qword_ptr(r8 + 64), xmm0)?;
                a.ret()
            },
        ));
    }
    for l in run_arm_only("kat_sse4a", cases) {
        let k: usize = l.name[l.name.len() - 1..].parse().unwrap();
        let (len, idx) = shapes[k];
        let width = if len == 0 { 64 } else { len as u32 };
        let low = |n: u32| if n >= 64 { !0u64 } else { (1u64 << n) - 1 };
        let want = if l.name.starts_with("insq") {
            let m = low(width) << idx;
            (l.a & !m) | ((l.b << idx) & m)
        } else {
            (l.a >> idx) & low(width)
        };
        assert_eq!(l.w[8], want, "{} {:#x} {:#x}", l.name, l.a, l.b);
    }
    // Two hand-checked vectors, independent of the formula above.
    let insert = |dst: u64, src: u64, len: u32, idx: u32| {
        let m = ((1u64 << len) - 1) << idx;
        (dst & !m) | ((src << idx) & m)
    };
    assert_eq!(insert(!0, 0, 8, 4), 0xffff_ffff_ffff_f00f);
    assert_eq!((0x1234_5678_9abc_def0u64 >> 8) & 0xffff, 0xbcde);
}

#[test]
fn adcx_adox_known_answers() {
    if !have_tools() {
        return;
    }
    // cf/of input chosen by the snippet name: cmp sets CF=1 (0 < 1) or CF=0; `mov r9, -1; add r9, r9` sets CF.
    let mut cases = vec![];
    for (name, adox) in [("adcx", false), ("adox", true)] {
        for cin in [0, 1] {
            for bits in [32, 64] {
                cases.push(case(
                    "kat",
                    format!("{name}_{cin}_{bits}"),
                    Kind::Int,
                    move |a, _| {
                        // Put `cin` in CF (adcx) or OF (adox) without disturbing the other flag:
                        // CF: cmp eax,ebx; OF: add of two large positives.
                        a.xor(eax, eax)?;
                        if adox {
                            a.mov(r9, 0x7fff_ffff_ffff_ffffu64)?;
                            if cin == 1 {
                                a.add(r9, r9)?
                            } else {
                                a.add(r9, 0)?
                            }
                        } else if cin == 1 {
                            a.cmp(eax, 1)?;
                        } else {
                            a.cmp(eax, 0)?;
                        }
                        a.mov(rax, rcx)?;
                        match (adox, bits) {
                            (false, 64) => a.adcx(rax, rdx)?,
                            (false, _) => a.adcx(eax, edx)?,
                            (true, 64) => a.adox(rax, rdx)?,
                            (true, _) => a.adox(eax, edx)?,
                        }
                        fl(a, "co")?;
                        done(a)
                    },
                ));
            }
        }
    }
    for l in run_arm_only("kat_adx", cases) {
        let p: Vec<&str> = l.name.split('_').collect();
        let (adox, cin, bits): (bool, u64, u32) =
            (p[0] == "adox", p[1].parse().unwrap(), p[2].parse().unwrap());
        let (a, b) = if bits == 64 {
            (l.a as u128, l.b as u128)
        } else {
            (l.a as u32 as u128, l.b as u32 as u128)
        };
        let s = a + b + cin as u128;
        let (res, cout) = if bits == 64 {
            (s as u64, (s >> 64) as u8)
        } else {
            (s as u32 as u64, (s >> 32) as u8)
        };
        assert_eq!(l.w[1], res, "{} {:#x} {:#x}", l.name, l.a, l.b);
        let flags = l.w[0].to_le_bytes();
        // adcx writes CF only, adox OF only: the other flag keeps its setup value.
        let (cf, of) = (flags[0], flags[3]);
        if adox {
            assert_eq!(of, cout, "{} OF", l.name);
        } else {
            assert_eq!(cf, cout, "{} CF", l.name);
        }
    }
}
