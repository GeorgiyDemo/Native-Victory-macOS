//! Differential tests for the legacy (non-VEX) SSE integer forms, AES-NI, PCLMULQDQ and
//! `vphminposuw`: x86-64 under Rosetta 2 against the lifted arm64 build, 300 random inputs each.
//! The operand buffers and result registers are those of `diff_vex.rs` (`pre`/`vcase`); the result
//! ymm0 is stored whole, so a legacy form that wrongly touched the upper half would show.
//! The ymm forms (VAES, VPCLMULQDQ), SSE4a and `adcx`/`adox` raise SIGILL under Rosetta 2; `kat.rs` covers them.
#![cfg(target_os = "macos")]
#![allow(unused_variables)]

#[macro_use]
mod common;

use common::*;
use iced_x86::code_asm::*;

/// `op xmm0, xmm2` and `op xmm0, [mem]`.
macro_rules! l2 {
    ($v:expr, $pre:expr, $($op:ident),*) => { $(
        vcase($v, "legacy-bin", format!("{}_r", stringify!($op)), $pre, |a| a.$op(xmm0, xmm2));
        vcase($v, "legacy-bin", format!("{}_m", stringify!($op)), $pre, |a| a.$op(xmm0, xmmword_ptr(r8 + 160)));
    )* };
}
macro_rules! l2i {
    ($v:expr, $cls:expr, $imm:expr, $($op:ident),*) => { $(
        for imm in $imm {
            vcase($v, $cls, format!("{}_r{imm}", stringify!($op)), Pre::Int, move |a| a.$op(xmm0, xmm2, imm));
            vcase($v, $cls, format!("{}_m{imm}", stringify!($op)), Pre::Int, move |a| a.$op(xmm0, xmmword_ptr(r8 + 160), imm));
        }
    )* };
}
macro_rules! l1i {
    ($v:expr, $cls:expr, $imm:expr, $($op:ident),*) => { $(
        for imm in $imm {
            vcase($v, $cls, format!("{}_r{imm}", stringify!($op)), Pre::Int, move |a| a.$op(xmm0, xmm2, imm));
            vcase($v, $cls, format!("{}_m{imm}", stringify!($op)), Pre::Int, move |a| a.$op(xmm0, xmmword_ptr(r8 + 160), imm));
        }
    )* };
}

const IMMS: [i32; 6] = [0, 1, 0x1b, 0x4e, 0xe4, 0xff];

#[test]
fn legacy_sse_integer_forms_match_x86() {
    let mut v = vec![];
    l2!(
        &mut v,
        Pre::Int,
        paddb,
        paddw,
        paddd,
        paddq,
        psubb,
        psubw,
        psubd,
        psubq,
        paddsb,
        paddsw,
        paddusb,
        paddusw,
        psubsb,
        psubsw,
        psubusb,
        psubusw,
        pmullw,
        pmulld,
        pmulhw,
        pmulhuw,
        pmulhrsw,
        pmuludq,
        pmuldq,
        pmaddwd,
        pmaddubsw,
        pavgb,
        pavgw,
        pminsb,
        pminsw,
        pminsd,
        pminub,
        pminuw,
        pminud,
        pmaxsb,
        pmaxsw,
        pmaxsd,
        pmaxub,
        pmaxuw,
        pmaxud,
        pcmpeqb,
        pcmpeqw,
        pcmpeqd,
        pcmpeqq,
        pcmpgtb,
        pcmpgtw,
        pcmpgtd,
        pcmpgtq,
        psignb,
        psignw,
        psignd,
        psadbw,
        phaddw,
        phaddd,
        phsubw,
        phsubd,
        phaddsw,
        phsubsw,
        packsswb,
        packuswb,
        packssdw,
        packusdw,
        punpcklbw,
        punpckhbw,
        punpcklwd,
        punpckhwd,
        punpckldq,
        punpckhdq,
        punpcklqdq,
        punpckhqdq,
        pshufb,
        unpcklps,
        unpckhps,
        unpcklpd,
        unpckhpd
    );
    vcase(&mut v, "legacy-bin", "movlhps_r".into(), Pre::Int, |a| {
        a.movlhps(xmm0, xmm2)
    });
    vcase(&mut v, "legacy-bin", "movhlps_r".into(), Pre::Int, |a| {
        a.movhlps(xmm0, xmm2)
    });
    l2!(&mut v, Pre::FpS, haddps, hsubps, addsubps);
    l2!(&mut v, Pre::FpD, haddpd, hsubpd, addsubpd);
    l2i!(&mut v, "legacy-imm", IMMS, palignr, pblendw, shufps, shufpd);
    l2i!(&mut v, "legacy-imm", [0, 5, 0xf], blendps);
    l2i!(&mut v, "legacy-imm", [0, 1, 3], blendpd);
    l1i!(&mut v, "legacy-un-imm", IMMS, pshufd, pshufhw, pshuflw);
    for op in 0..3 {
        vcase(
            &mut v,
            "legacy-un",
            format!("pabs{op}"),
            Pre::Int,
            move |a| match op {
                0 => a.pabsb(xmm0, xmm2),
                1 => a.pabsw(xmm0, xmm2),
                _ => a.pabsd(xmm0, xmmword_ptr(r8 + 160)),
            },
        );
    }
    vcase(&mut v, "legacy-un", "phminposuw".into(), Pre::Int, |a| {
        a.phminposuw(xmm0, xmm2)
    });
    vcase(&mut v, "legacy-un", "vphminposuw".into(), Pre::Int, |a| {
        a.vphminposuw(xmm0, xmm2)
    });
    vcase(&mut v, "legacy-un", "vphminposuw_m".into(), Pre::Int, |a| {
        a.vphminposuw(xmm0, xmmword_ptr(r8 + 160))
    });
    // Shifts: immediate, xmm count, memory count.
    macro_rules! sh {
        ($($op:ident),*) => { $(
            for imm in [0, 1, 7, 15, 16, 31, 32, 63, 64, 255] {
                vcase(&mut v, "legacy-shift", format!("{}_i{imm}", stringify!($op)), Pre::Int, move |a| a.$op(xmm0, imm));
            }
            vcase(&mut v, "legacy-shift", format!("{}_r", stringify!($op)), Pre::Int, |a| a.$op(xmm0, xmm2));
            vcase(&mut v, "legacy-shift", format!("{}_m", stringify!($op)), Pre::Int, |a| a.$op(xmm0, xmmword_ptr(r8 + 160)));
        )* };
    }
    sh!(psllw, pslld, psllq, psrlw, psrld, psrlq, psraw, psrad);
    for imm in [0, 1, 4, 8, 15, 16, 20] {
        vcase(
            &mut v,
            "legacy-shift",
            format!("pslldq{imm}"),
            Pre::Int,
            move |a| a.pslldq(xmm0, imm),
        );
        vcase(
            &mut v,
            "legacy-shift",
            format!("psrldq{imm}"),
            Pre::Int,
            move |a| a.psrldq(xmm0, imm),
        );
    }
    // Widening converts.
    macro_rules! ext {
        ($($op:ident),*) => { $(
            vcase(&mut v, "legacy-ext", format!("{}_r", stringify!($op)), Pre::Int, |a| a.$op(xmm0, xmm2));
            vcase(&mut v, "legacy-ext", format!("{}_m", stringify!($op)), Pre::Int, |a| a.$op(xmm0, xmmword_ptr(r8 + 160)));
        )* };
    }
    ext!(
        pmovsxbw, pmovsxbd, pmovsxbq, pmovsxwd, pmovsxwq, pmovsxdq, pmovzxbw, pmovzxbd, pmovzxbq,
        pmovzxwd, pmovzxwq, pmovzxdq
    );
    vcase(&mut v, "legacy-ext", "cvtdq2pd_r".into(), Pre::Int, |a| {
        a.cvtdq2pd(xmm0, xmm2)
    });
    vcase(&mut v, "legacy-ext", "cvtdq2pd_m".into(), Pre::Int, |a| {
        a.cvtdq2pd(xmm0, qword_ptr(r8 + 160))
    });
    // Element insert/extract and masks.
    for imm in 0..16 {
        vcase(
            &mut v,
            "legacy-elem",
            format!("pinsrb{imm}"),
            Pre::Int,
            move |a| a.pinsrb(xmm0, ecx, imm),
        );
        vcase(
            &mut v,
            "legacy-elem",
            format!("pextrb{imm}"),
            Pre::Int,
            move |a| a.pextrb(eax, xmm1, imm),
        );
    }
    for imm in 0..8 {
        vcase(
            &mut v,
            "legacy-elem",
            format!("pinsrw{imm}"),
            Pre::Int,
            move |a| a.pinsrw(xmm0, edx, imm),
        );
        vcase(
            &mut v,
            "legacy-elem",
            format!("pextrw{imm}"),
            Pre::Int,
            move |a| a.pextrw(eax, xmm1, imm),
        );
    }
    for imm in 0..4 {
        vcase(
            &mut v,
            "legacy-elem",
            format!("pinsrd{imm}"),
            Pre::Int,
            move |a| a.pinsrd(xmm0, edx, imm),
        );
        vcase(
            &mut v,
            "legacy-elem",
            format!("pextrd{imm}"),
            Pre::Int,
            move |a| a.pextrd(eax, xmm1, imm),
        );
    }
    for imm in 0..2 {
        vcase(
            &mut v,
            "legacy-elem",
            format!("pinsrq{imm}"),
            Pre::Int,
            move |a| a.pinsrq(xmm0, rcx, imm),
        );
        vcase(
            &mut v,
            "legacy-elem",
            format!("pextrq{imm}"),
            Pre::Int,
            move |a| a.pextrq(rax, xmm1, imm),
        );
        vcase(
            &mut v,
            "legacy-elem",
            format!("pextrd_m{imm}"),
            Pre::Int,
            move |a| {
                a.pextrd(dword_ptr(r8 + 224), xmm2, imm)?;
                a.mov(eax, dword_ptr(r8 + 224))
            },
        );
    }
    vcase(&mut v, "legacy-elem", "pmovmskb".into(), Pre::Int, |a| {
        a.pmovmskb(eax, xmm1)
    });
    vcase(&mut v, "legacy-elem", "movmskps".into(), Pre::Int, |a| {
        a.movmskps(eax, xmm1)
    });
    vcase(&mut v, "legacy-elem", "movmskpd".into(), Pre::Int, |a| {
        a.movmskpd(eax, xmm1)
    });
    // The result lives in out+64; an extra 32-bit store to out+224 must not leak into it.
    run_cases("diff_legacy", v, 0);
}

#[test]
fn aes_and_clmul_match_x86() {
    let mut v = vec![];
    for op in ["aesenc", "aesenclast", "aesdec", "aesdeclast"] {
        for form in 0..3 {
            vcase(
                &mut v,
                "aes",
                format!("{op}_{form}"),
                Pre::Int,
                move |a| match (op, form) {
                    ("aesenc", 0) => a.aesenc(xmm0, xmm2),
                    ("aesenc", 1) => a.aesenc(xmm0, xmmword_ptr(r8 + 160)),
                    ("aesenc", _) => a.vaesenc(xmm0, xmm1, xmm2),
                    ("aesenclast", 0) => a.aesenclast(xmm0, xmm2),
                    ("aesenclast", 1) => a.aesenclast(xmm0, xmmword_ptr(r8 + 160)),
                    ("aesenclast", _) => a.vaesenclast(xmm0, xmm1, xmm2),
                    ("aesdec", 0) => a.aesdec(xmm0, xmm2),
                    ("aesdec", 1) => a.aesdec(xmm0, xmmword_ptr(r8 + 160)),
                    ("aesdec", _) => a.vaesdec(xmm0, xmm1, xmm2),
                    ("aesdeclast", 0) => a.aesdeclast(xmm0, xmm2),
                    ("aesdeclast", 1) => a.aesdeclast(xmm0, xmmword_ptr(r8 + 160)),
                    _ => a.vaesdeclast(xmm0, xmm1, xmm2),
                },
            );
        }
    }
    vcase(&mut v, "aes", "aesimc".into(), Pre::Int, |a| {
        a.aesimc(xmm0, xmm2)
    });
    vcase(&mut v, "aes", "vaesimc".into(), Pre::Int, |a| {
        a.vaesimc(xmm0, xmm2)
    });
    for rc in [0, 1, 0x1b, 0x80, 0xff] {
        vcase(
            &mut v,
            "aes",
            format!("aeskeygenassist{rc}"),
            Pre::Int,
            move |a| a.aeskeygenassist(xmm0, xmm2, rc),
        );
        vcase(
            &mut v,
            "aes",
            format!("vaeskeygenassist_m{rc}"),
            Pre::Int,
            move |a| a.vaeskeygenassist(xmm0, xmmword_ptr(r8 + 160), rc),
        );
    }
    for imm in [0x00, 0x01, 0x10, 0x11] {
        vcase(
            &mut v,
            "clmul",
            format!("pclmulqdq{imm}"),
            Pre::Int,
            move |a| a.pclmulqdq(xmm0, xmm2, imm),
        );
        vcase(
            &mut v,
            "clmul",
            format!("pclmulqdq_m{imm}"),
            Pre::Int,
            move |a| a.pclmulqdq(xmm0, xmmword_ptr(r8 + 160), imm),
        );
        vcase(
            &mut v,
            "clmul",
            format!("vpclmulqdq{imm}"),
            Pre::Int,
            move |a| a.vpclmulqdq(xmm0, xmm1, xmm2, imm),
        );
    }
    run_cases("diff_aes", v, 0);
}
