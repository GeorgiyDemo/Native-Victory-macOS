//! Differential tests for the VEX forms (AVX, AVX2, FMA, F16C): each instruction is
//! assembled in 128-bit, 256-bit and memory-operand forms, run as x86-64 under
//! Rosetta 2 and as lifted arm64, and the full 256-bit result must match bit for bit.
//!
//! Every snippet starts with a prelude that fills three 32-byte operand buffers
//! from its inputs (`out+128` A, `out+160` B, `out+192` C) and loads them into
//! ymm1, ymm2, ymm3 (ymm0 and ymm4 get C, so 128-bit forms also prove the upper
//! half is zeroed). The result ymm0 goes to `out+64`, ymm4 to `out+96`.
#![cfg(target_os = "macos")]
#![allow(unused_variables)]

#[macro_use]
mod common;

use common::*;
use iced_x86::code_asm::*;

#[derive(Clone, Copy)]
enum Pre {
    /// Integer inputs: four qwords per buffer derived from a and b.
    Int,
    /// Doubles: the inputs are random and special float64 bit patterns.
    FpD,
    /// Floats: the inputs are pairs of random and special float32 bit patterns.
    FpS,
}

fn pre(a: &mut CodeAssembler, p: Pre) -> Result<(), IcedError> {
    match p {
        Pre::FpS | Pre::FpD => {
            // Lanes are the inputs and their half-swapped forms, so every lane stays a "real" float.
            a.mov(r9, rcx)?;
            a.ror(r9, 32)?;
            a.mov(r10, rdx)?;
            a.ror(r10, 32)?;
            for (off, q) in [(128, [rcx, rdx, r9, r10]), (160, [rdx, rcx, r10, r9]), (192, [r9, rdx, rcx, r10])] {
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

fn vcase(v: &mut Vec<Case>, class: &'static str, name: String, p: Pre, body: impl Fn(&mut CodeAssembler) -> Result<(), IcedError> + 'static) {
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

/// dst, src1, src2 in xmm, ymm, xmm+mem and ymm+mem forms.
macro_rules! v3 {
    ($v:expr, $cls:expr, $pre:expr, $($op:ident),*) => { $(
        vcase($v, $cls, format!("{}_x", stringify!($op)), $pre, |a| a.$op(xmm0, xmm1, xmm2));
        vcase($v, $cls, format!("{}_y", stringify!($op)), $pre, |a| a.$op(ymm0, ymm1, ymm2));
        vcase($v, $cls, format!("{}_xm", stringify!($op)), $pre, |a| a.$op(xmm0, xmm1, xmmword_ptr(r8 + 160)));
        vcase($v, $cls, format!("{}_ym", stringify!($op)), $pre, |a| a.$op(ymm0, ymm1, ymmword_ptr(r8 + 160)));
    )* };
}
/// With an immediate.
macro_rules! v3i {
    ($v:expr, $cls:expr, $pre:expr, $imm:expr, $($op:ident),*) => { $(
        for imm in $imm {
            vcase($v, $cls, format!("{}_x{imm}", stringify!($op)), $pre, move |a| a.$op(xmm0, xmm1, xmm2, imm));
            vcase($v, $cls, format!("{}_y{imm}", stringify!($op)), $pre, move |a| a.$op(ymm0, ymm1, ymm2, imm));
            vcase($v, $cls, format!("{}_ym{imm}", stringify!($op)), $pre, move |a| a.$op(ymm0, ymm1, ymmword_ptr(r8 + 160), imm));
        }
    )* };
}
/// dst, src.
macro_rules! v2 {
    ($v:expr, $cls:expr, $pre:expr, $($op:ident),*) => { $(
        vcase($v, $cls, format!("{}_x", stringify!($op)), $pre, |a| a.$op(xmm0, xmm2));
        vcase($v, $cls, format!("{}_y", stringify!($op)), $pre, |a| a.$op(ymm0, ymm2));
        vcase($v, $cls, format!("{}_ym", stringify!($op)), $pre, |a| a.$op(ymm0, ymmword_ptr(r8 + 160)));
    )* };
}
macro_rules! v2i {
    ($v:expr, $cls:expr, $pre:expr, $imm:expr, $($op:ident),*) => { $(
        for imm in $imm {
            vcase($v, $cls, format!("{}_x{imm}", stringify!($op)), $pre, move |a| a.$op(xmm0, xmm2, imm));
            vcase($v, $cls, format!("{}_y{imm}", stringify!($op)), $pre, move |a| a.$op(ymm0, ymm2, imm));
            vcase($v, $cls, format!("{}_ym{imm}", stringify!($op)), $pre, move |a| a.$op(ymm0, ymmword_ptr(r8 + 160), imm));
        }
    )* };
}
/// Scalar three-operand (xmm only), register and memory source.
macro_rules! s3 {
    ($v:expr, $cls:expr, $pre:expr, $mem:ident, $($nm:ident = $op:ident),*) => { $(
        vcase($v, $cls, format!("{}_r", stringify!($nm)), $pre, |a| a.$op(xmm0, xmm1, xmm2));
        vcase($v, $cls, format!("{}_d", stringify!($nm)), $pre, |a| { a.vmovups(ymm0, ymm1)?; a.$op(xmm0, xmm0, xmm2) });
        vcase($v, $cls, format!("{}_o", stringify!($nm)), $pre, |a| { a.$op(xmm5, xmm1, xmm2)?; a.vmovups(ymm0, ymm5) });
        vcase($v, $cls, format!("{}_m", stringify!($nm)), $pre, |a| a.$op(xmm0, xmm1, $mem(r8 + 160)));
    )* };
}

const IMM8: [i32; 6] = [0x1b, 0xe4, 0x4e, 0xb1, 0x93, 0x00];

fn arith_cases(v: &mut Vec<Case>) {
    // float arithmetic and logic
    v3!(v, "avx-float", Pre::FpS, vaddps, vsubps, vmulps, vdivps, vminps, vmaxps, vhaddps, vhsubps, vaddsubps, vunpcklps, vunpckhps, vpermilps);
    v3!(v, "avx-float", Pre::FpD, vaddpd, vsubpd, vmulpd, vdivpd, vminpd, vmaxpd, vhaddpd, vhsubpd, vaddsubpd, vunpcklpd, vunpckhpd);
    v3!(v, "avx-logic", Pre::Int, vxorps, vxorpd, vandps, vandpd, vorps, vorpd, vandnps, vandnpd, vpxor, vpand, vpor, vpandn);
    s3!(v, "avx-scalar", Pre::FpS, dword_ptr, vaddss = vaddss, vsubss = vsubss, vmulss = vmulss, vdivss = vdivss, vminss = vminss, vmaxss = vmaxss, vsqrtss = vsqrtss, vrcpss = vrcpss, vrsqrtss = vrsqrtss);
    s3!(v, "avx-scalar-int", Pre::Int, dword_ptr, vaddss_i = vaddss, vmulss_i = vmulss, vsqrtss_i = vsqrtss);
    s3!(v, "avx-scalar", Pre::FpD, qword_ptr, vaddsd = vaddsd, vsubsd = vsubsd, vmulsd = vmulsd, vdivsd = vdivsd, vminsd = vminsd, vmaxsd = vmaxsd, vsqrtsd = vsqrtsd);
    v2!(v, "avx-float", Pre::FpS, vsqrtps, vrcpps, vrsqrtps, vcvtdq2ps, vcvttps2dq, vcvtps2dq, vmovshdup, vmovsldup);
    v2!(v, "avx-float", Pre::FpD, vsqrtpd, vmovddup);
    // The integer view of the same lanes reaches the NaN/denormal and large-value paths of the conversions.
    v2!(v, "avx-int", Pre::Int, vcvtdq2ps, vcvttps2dq, vcvtps2dq, vrcpps, vrsqrtps);
    v2i!(v, "avx-float", Pre::FpS, [0, 1, 2, 3, 4, 9, 12], vroundps);
    v2i!(v, "avx-float", Pre::FpD, [0, 1, 2, 3, 4, 10], vroundpd);
    // lane-wise integer
    // Rosetta 2 truncates the count of vpsllvq/vpsrlvq to 32 bits, hardware does not: keep counts below 128.
    for (nm, f) in [("vpsllvq", 0), ("vpsrlvq", 1)] {
        for (sfx, wide) in [("x", false), ("y", true)] {
            vcase(v, "avx-int", format!("{nm}_{sfx}"), Pre::Int, move |a| {
                a.vpsrlq(ymm2, ymm2, 57)?;
                match (f, wide) {
                    (0, false) => a.vpsllvq(xmm0, xmm1, xmm2),
                    (0, true) => a.vpsllvq(ymm0, ymm1, ymm2),
                    (_, false) => a.vpsrlvq(xmm0, xmm1, xmm2),
                    (_, true) => a.vpsrlvq(ymm0, ymm1, ymm2),
                }
            });
        }
    }
    v3!(v, "avx-int", Pre::Int,
        vpaddb, vpaddw, vpaddd, vpaddq, vpsubb, vpsubw, vpsubd, vpsubq, vpaddsb, vpaddsw, vpaddusb, vpaddusw,
        vpsubsb, vpsubsw, vpsubusb, vpsubusw, vpmullw, vpmulld, vpmulhw, vpmulhuw, vpmulhrsw, vpmuludq, vpmuldq,
        vpmaddwd, vpmaddubsw, vpavgb, vpavgw, vpminsb, vpminsw, vpminsd, vpminub, vpminuw, vpminud, vpmaxsb,
        vpmaxsw, vpmaxsd, vpmaxub, vpmaxuw, vpmaxud, vpcmpeqb, vpcmpeqw, vpcmpeqd, vpcmpeqq, vpcmpgtb, vpcmpgtw,
        vpcmpgtd, vpcmpgtq, vpsignb, vpsignw, vpsignd, vpsadbw, vphaddw, vphaddd, vphsubw, vphsubd, vphaddsw,
        vphsubsw, vpacksswb, vpackuswb, vpackssdw, vpackusdw, vpunpcklbw, vpunpckhbw, vpunpcklwd, vpunpckhwd,
        vpunpckldq, vpunpckhdq, vpunpcklqdq, vpunpckhqdq, vpshufb, vpsllvd, vpsrlvd, vpsravd);
    v2!(v, "avx-int", Pre::Int, vpabsb, vpabsw, vpabsd);
}

fn imm_cases(v: &mut Vec<Case>) {
    v3i!(v, "avx-imm", Pre::FpS, IMM8, vshufps, vblendps);
    v3i!(v, "avx-imm", Pre::FpD, [0, 1, 2, 3, 5, 10, 15], vshufpd, vblendpd);
    v3i!(v, "avx-imm", Pre::Int, [1, 3, 8, 15, 17], vpalignr);
    v3i!(v, "avx-imm", Pre::Int, [0x1b, 0xa5, 0x5a, 0xff, 0x01], vpblendw, vpblendd);
    v2i!(v, "avx-imm", Pre::Int, IMM8, vpshufd, vpshufhw, vpshuflw, vpermilps);
    v2i!(v, "avx-imm", Pre::FpD, [0, 1, 2, 3, 5, 10, 15], vpermilpd);
    // vcmpps/vcmppd/vcmpss/vcmpsd: every predicate.
    for p in 0..32 {
        vcase(v, "avx-cmp", format!("vcmpps_x{p}"), Pre::FpS, move |a| a.vcmpps(xmm0, xmm1, xmm2, p));
        vcase(v, "avx-cmp", format!("vcmpps_y{p}"), Pre::FpS, move |a| a.vcmpps(ymm0, ymm1, ymm2, p));
        vcase(v, "avx-cmp", format!("vcmppd_y{p}"), Pre::FpD, move |a| a.vcmppd(ymm0, ymm1, ymm2, p));
        vcase(v, "avx-cmp", format!("vcmpss_{p}"), Pre::FpS, move |a| a.vcmpss(xmm0, xmm1, xmm2, p));
        vcase(v, "avx-cmp", format!("vcmpsd_m{p}"), Pre::FpD, move |a| a.vcmpsd(xmm0, xmm1, qword_ptr(r8 + 160), p));
    }
    for m in [0, 1, 2, 3, 4, 9] {
        vcase(v, "avx-float", format!("vroundss_{m}"), Pre::FpS, move |a| a.vroundss(xmm0, xmm1, xmm2, m));
        vcase(v, "avx-float", format!("vroundsd_{m}"), Pre::FpD, move |a| a.vroundsd(xmm0, xmm1, xmm2, m));
    }
}

fn check(label: &str, build: fn(&mut Vec<Case>)) {
    if !have_tools() {
        eprintln!("skipped: needs Rosetta 2 and clang");
        return;
    }
    let mut cases = vec![];
    build(&mut cases);
    run_cases(label, cases, 0);
}

#[test]
fn vex_arithmetic_and_integer_match_x86() {
    check("vex_arith", arith_cases);
}

#[test]
fn vex_immediate_forms_match_x86() {
    check("vex_imm", imm_cases);
}
