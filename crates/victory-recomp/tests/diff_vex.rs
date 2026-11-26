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

/// Scratch area (`out+224`): store here, then read the 32 bytes back into ymm0.
const SCRATCH: i32 = 224;

fn readback(a: &mut CodeAssembler) -> Result<(), IcedError> {
    a.vmovups(ymm0, ymmword_ptr(r8 + SCRATCH))
}

fn move_cases(v: &mut Vec<Case>) {
    macro_rules! mv {
        ($($op:ident),*) => { $(
            vcase(v, "avx-move", format!("{}_ld_x", stringify!($op)), Pre::Int, |a| a.$op(xmm0, xmmword_ptr(r8 + 160)));
            vcase(v, "avx-move", format!("{}_ld_y", stringify!($op)), Pre::Int, |a| a.$op(ymm0, ymmword_ptr(r8 + 160)));
            vcase(v, "avx-move", format!("{}_rr_x", stringify!($op)), Pre::Int, |a| a.$op(xmm0, xmm2));
            vcase(v, "avx-move", format!("{}_rr_y", stringify!($op)), Pre::Int, |a| a.$op(ymm0, ymm2));
            vcase(v, "avx-move", format!("{}_st_x", stringify!($op)), Pre::Int, |a| { a.$op(xmmword_ptr(r8 + SCRATCH), xmm1)?; readback(a) });
            vcase(v, "avx-move", format!("{}_st_y", stringify!($op)), Pre::Int, |a| { a.$op(ymmword_ptr(r8 + SCRATCH), ymm1)?; readback(a) });
        )* };
    }
    mv!(vmovaps, vmovups, vmovapd, vmovupd, vmovdqa, vmovdqu);
    vcase(v, "avx-move", "vmovss_rrr".into(), Pre::FpS, |a| a.vmovss_3(xmm0, xmm1, xmm2));
    vcase(v, "avx-move", "vmovsd_rrr".into(), Pre::FpD, |a| a.vmovsd_3(xmm0, xmm1, xmm2));
    vcase(v, "avx-move", "vmovss_ld".into(), Pre::FpS, |a| a.vmovss(xmm0, dword_ptr(r8 + 160)));
    vcase(v, "avx-move", "vmovsd_ld".into(), Pre::FpD, |a| a.vmovsd(xmm0, qword_ptr(r8 + 160)));
    vcase(v, "avx-move", "vmovss_st".into(), Pre::FpS, |a| { a.vmovss(dword_ptr(r8 + SCRATCH), xmm1)?; readback(a) });
    vcase(v, "avx-move", "vmovsd_st".into(), Pre::FpD, |a| { a.vmovsd(qword_ptr(r8 + SCRATCH), xmm1)?; readback(a) });
    vcase(v, "avx-move", "vmovd_to".into(), Pre::Int, |a| a.vmovd(xmm0, ecx));
    vcase(v, "avx-move", "vmovq_to".into(), Pre::Int, |a| a.vmovq(xmm0, rcx));
    vcase(v, "avx-move", "vmovd_from".into(), Pre::Int, |a| a.vmovd(eax, xmm1));
    vcase(v, "avx-move", "vmovq_from".into(), Pre::Int, |a| a.vmovq(rax, xmm1));
    vcase(v, "avx-move", "vmovd_ld".into(), Pre::Int, |a| a.vmovd(xmm0, dword_ptr(r8 + 160)));
    vcase(v, "avx-move", "vmovq_ld".into(), Pre::Int, |a| a.vmovq(xmm0, qword_ptr(r8 + 160)));
    vcase(v, "avx-move", "vmovq_rr".into(), Pre::Int, |a| a.vmovq(xmm0, xmm2));
    vcase(v, "avx-move", "vmovd_st".into(), Pre::Int, |a| { a.vmovd(dword_ptr(r8 + SCRATCH), xmm1)?; readback(a) });
    vcase(v, "avx-move", "vmovq_st".into(), Pre::Int, |a| { a.vmovq(qword_ptr(r8 + SCRATCH), xmm1)?; readback(a) });
    vcase(v, "avx-move", "vmovlps_ld".into(), Pre::Int, |a| a.vmovlps_3(xmm0, xmm1, qword_ptr(r8 + 160)));
    vcase(v, "avx-move", "vmovhps_ld".into(), Pre::Int, |a| a.vmovhps_3(xmm0, xmm1, qword_ptr(r8 + 160)));
    vcase(v, "avx-move", "vmovlpd_ld".into(), Pre::Int, |a| a.vmovlpd_3(xmm0, xmm1, qword_ptr(r8 + 160)));
    vcase(v, "avx-move", "vmovhpd_ld".into(), Pre::Int, |a| a.vmovhpd_3(xmm0, xmm1, qword_ptr(r8 + 160)));
    vcase(v, "avx-move", "vmovlps_st".into(), Pre::Int, |a| { a.vmovlps(qword_ptr(r8 + SCRATCH), xmm1)?; readback(a) });
    vcase(v, "avx-move", "vmovhps_st".into(), Pre::Int, |a| { a.vmovhps(qword_ptr(r8 + SCRATCH), xmm1)?; readback(a) });
    vcase(v, "avx-move", "vmovhpd_st".into(), Pre::Int, |a| { a.vmovhpd(qword_ptr(r8 + SCRATCH), xmm1)?; readback(a) });
    vcase(v, "avx-move", "vmovlhps".into(), Pre::Int, |a| a.vmovlhps(xmm0, xmm1, xmm2));
    vcase(v, "avx-move", "vmovhlps".into(), Pre::Int, |a| a.vmovhlps(xmm0, xmm1, xmm2));
    vcase(v, "avx-move", "vmovmskps_x".into(), Pre::Int, |a| a.vmovmskps(eax, xmm1));
    vcase(v, "avx-move", "vmovmskps_y".into(), Pre::Int, |a| a.vmovmskps(eax, ymm1));
    vcase(v, "avx-move", "vmovmskpd_x".into(), Pre::Int, |a| a.vmovmskpd(eax, xmm1));
    vcase(v, "avx-move", "vmovmskpd_y".into(), Pre::Int, |a| a.vmovmskpd(eax, ymm1));
    vcase(v, "avx-move", "vpmovmskb_x".into(), Pre::Int, |a| a.vpmovmskb(eax, xmm1));
    vcase(v, "avx-move", "vpmovmskb_y".into(), Pre::Int, |a| a.vpmovmskb(eax, ymm1));
    vcase(v, "avx-bcast", "vbroadcastss_xm".into(), Pre::Int, |a| a.vbroadcastss(xmm0, dword_ptr(r8 + 164)));
    vcase(v, "avx-bcast", "vbroadcastss_ym".into(), Pre::Int, |a| a.vbroadcastss(ymm0, dword_ptr(r8 + 164)));
    vcase(v, "avx-bcast", "vbroadcastss_xx".into(), Pre::Int, |a| a.vbroadcastss(xmm0, xmm2));
    vcase(v, "avx-bcast", "vbroadcastss_yx".into(), Pre::Int, |a| a.vbroadcastss(ymm0, xmm2));
    vcase(v, "avx-bcast", "vbroadcastsd_ym".into(), Pre::Int, |a| a.vbroadcastsd(ymm0, qword_ptr(r8 + 168)));
    vcase(v, "avx-bcast", "vbroadcastsd_yx".into(), Pre::Int, |a| a.vbroadcastsd(ymm0, xmm2));
    vcase(v, "avx-bcast", "vbroadcastf128".into(), Pre::Int, |a| a.vbroadcastf128(ymm0, xmmword_ptr(r8 + 168)));
    vcase(v, "avx-bcast", "vbroadcasti128".into(), Pre::Int, |a| a.vbroadcasti128(ymm0, xmmword_ptr(r8 + 168)));
    macro_rules! pb {
        ($($op:ident $mem:ident),*) => { $(
            vcase(v, "avx-bcast", format!("{}_xm", stringify!($op)), Pre::Int, |a| a.$op(xmm0, $mem(r8 + 165)));
            vcase(v, "avx-bcast", format!("{}_ym", stringify!($op)), Pre::Int, |a| a.$op(ymm0, $mem(r8 + 165)));
            vcase(v, "avx-bcast", format!("{}_xx", stringify!($op)), Pre::Int, |a| a.$op(xmm0, xmm2));
            vcase(v, "avx-bcast", format!("{}_yx", stringify!($op)), Pre::Int, |a| a.$op(ymm0, xmm2));
        )* };
    }
    pb!(vpbroadcastb byte_ptr, vpbroadcastw word_ptr, vpbroadcastd dword_ptr, vpbroadcastq qword_ptr);
}

fn lane_cases(v: &mut Vec<Case>) {
    for i in 0..2 {
        vcase(v, "avx-lane", format!("vextractf128_r{i}"), Pre::Int, move |a| a.vextractf128(xmm0, ymm1, i));
        vcase(v, "avx-lane", format!("vextracti128_r{i}"), Pre::Int, move |a| a.vextracti128(xmm0, ymm1, i));
        vcase(v, "avx-lane", format!("vextractf128_m{i}"), Pre::Int, move |a| { a.vextractf128(xmmword_ptr(r8 + SCRATCH), ymm1, i)?; readback(a) });
        vcase(v, "avx-lane", format!("vinsertf128_r{i}"), Pre::Int, move |a| a.vinsertf128(ymm0, ymm1, xmm2, i));
        vcase(v, "avx-lane", format!("vinserti128_r{i}"), Pre::Int, move |a| a.vinserti128(ymm0, ymm1, xmm2, i));
        vcase(v, "avx-lane", format!("vinsertf128_m{i}"), Pre::Int, move |a| a.vinsertf128(ymm0, ymm1, xmmword_ptr(r8 + 160), i));
    }
    for m in [0x20, 0x31, 0x02, 0x13, 0x83, 0x28, 0x80, 0x88, 0x01, 0x30] {
        vcase(v, "avx-lane", format!("vperm2f128_{m:x}"), Pre::Int, move |a| a.vperm2f128(ymm0, ymm1, ymm2, m));
        vcase(v, "avx-lane", format!("vperm2i128_m{m:x}"), Pre::Int, move |a| a.vperm2i128(ymm0, ymm1, ymmword_ptr(r8 + 160), m));
    }
    for m in [0x1b, 0xe4, 0x4e, 0xb1, 0x93, 0x00, 0xff, 0x39] {
        vcase(v, "avx-lane", format!("vpermq_{m:x}"), Pre::Int, move |a| a.vpermq(ymm0, ymm1, m));
        vcase(v, "avx-lane", format!("vpermpd_m{m:x}"), Pre::Int, move |a| a.vpermpd(ymm0, ymmword_ptr(r8 + 160), m));
    }
    vcase(v, "avx-lane", "vpermd".into(), Pre::Int, |a| a.vpermd(ymm0, ymm1, ymm2));
    vcase(v, "avx-lane", "vpermps_m".into(), Pre::Int, |a| a.vpermps(ymm0, ymm1, ymmword_ptr(r8 + 160)));
    for i in 0..16 {
        vcase(v, "avx-elem", format!("vpextrb_{i}"), Pre::Int, move |a| a.vpextrb(eax, xmm1, i));
        vcase(v, "avx-elem", format!("vpinsrb_{i}"), Pre::Int, move |a| a.vpinsrb(xmm0, xmm1, ecx, i));
    }
    for i in 0..8 {
        vcase(v, "avx-elem", format!("vpextrw_{i}"), Pre::Int, move |a| a.vpextrw(eax, xmm1, i));
        vcase(v, "avx-elem", format!("vpinsrw_{i}"), Pre::Int, move |a| a.vpinsrw(xmm0, xmm1, ecx, i));
    }
    for i in 0..4 {
        vcase(v, "avx-elem", format!("vpextrd_{i}"), Pre::Int, move |a| a.vpextrd(eax, xmm1, i));
        vcase(v, "avx-elem", format!("vpinsrd_{i}"), Pre::Int, move |a| a.vpinsrd(xmm0, xmm1, ecx, i));
        vcase(v, "avx-elem", format!("vextractps_{i}"), Pre::Int, move |a| a.vextractps(eax, xmm1, i));
    }
    for i in 0..2 {
        vcase(v, "avx-elem", format!("vpextrq_{i}"), Pre::Int, move |a| a.vpextrq(rax, xmm1, i));
        vcase(v, "avx-elem", format!("vpinsrq_{i}"), Pre::Int, move |a| a.vpinsrq(xmm0, xmm1, rcx, i));
    }
    vcase(v, "avx-elem", "vpextrb_m".into(), Pre::Int, |a| { a.vpextrb(byte_ptr(r8 + SCRATCH), xmm1, 5)?; readback(a) });
    vcase(v, "avx-elem", "vpextrw_m".into(), Pre::Int, |a| { a.vpextrw(word_ptr(r8 + SCRATCH), xmm1, 3)?; readback(a) });
    vcase(v, "avx-elem", "vpextrd_m".into(), Pre::Int, |a| { a.vpextrd(dword_ptr(r8 + SCRATCH), xmm1, 2)?; readback(a) });
    vcase(v, "avx-elem", "vpextrq_m".into(), Pre::Int, |a| { a.vpextrq(qword_ptr(r8 + SCRATCH), xmm1, 1)?; readback(a) });
    vcase(v, "avx-elem", "vpinsrb_m".into(), Pre::Int, |a| a.vpinsrb(xmm0, xmm1, byte_ptr(r8 + 161), 9));
    vcase(v, "avx-elem", "vpinsrd_m".into(), Pre::Int, |a| a.vpinsrd(xmm0, xmm1, dword_ptr(r8 + 164), 1));
    for m in [0x00, 0x10, 0x4e, 0x9c, 0xff, 0x35, 0xc5] {
        vcase(v, "avx-elem", format!("vinsertps_r{m:x}"), Pre::Int, move |a| a.vinsertps(xmm0, xmm1, xmm2, m));
        vcase(v, "avx-elem", format!("vinsertps_m{m:x}"), Pre::Int, move |a| a.vinsertps(xmm0, xmm1, dword_ptr(r8 + 168), m));
    }
    macro_rules! ext {
        ($($op:ident $x:ident $y:ident),*) => { $(
            vcase(v, "avx-widen", format!("{}_x", stringify!($op)), Pre::Int, |a| a.$op(xmm0, xmm2));
            vcase(v, "avx-widen", format!("{}_y", stringify!($op)), Pre::Int, |a| a.$op(ymm0, xmm2));
            vcase(v, "avx-widen", format!("{}_xm", stringify!($op)), Pre::Int, |a| a.$op(xmm0, $x(r8 + 161)));
            vcase(v, "avx-widen", format!("{}_ym", stringify!($op)), Pre::Int, |a| a.$op(ymm0, $y(r8 + 161)));
        )* };
    }
    ext!(vpmovsxbw qword_ptr xmmword_ptr, vpmovsxbd dword_ptr qword_ptr, vpmovsxbq word_ptr dword_ptr, vpmovsxwd qword_ptr xmmword_ptr,
         vpmovsxwq dword_ptr qword_ptr, vpmovsxdq qword_ptr xmmword_ptr, vpmovzxbw qword_ptr xmmword_ptr, vpmovzxbd dword_ptr qword_ptr,
         vpmovzxbq word_ptr dword_ptr, vpmovzxwd qword_ptr xmmword_ptr, vpmovzxwq dword_ptr qword_ptr, vpmovzxdq qword_ptr xmmword_ptr);
    vcase(v, "avx-lane", "vzeroupper".into(), Pre::Int, |a| { a.vzeroupper()?; a.vmovups(ymm0, ymm1) });
    vcase(v, "avx-lane", "vzeroupper2".into(), Pre::Int, |a| { a.vzeroupper()?; a.vmovups(ymm4, ymm1)?; a.vmovups(ymm0, ymm2) });
}

fn shift_cases(v: &mut Vec<Case>) {
    macro_rules! sh {
        ($($op:ident),*) => { $(
            for c in [0, 1, 7, 15, 16, 31, 32, 33, 63, 64, 200] {
                vcase(v, "avx-shift", format!("{}_x{c}", stringify!($op)), Pre::Int, move |a| a.$op(xmm0, xmm1, c));
                vcase(v, "avx-shift", format!("{}_y{c}", stringify!($op)), Pre::Int, move |a| a.$op(ymm0, ymm1, c));
            }
            vcase(v, "avx-shift", format!("{}_xr", stringify!($op)), Pre::Int, |a| a.$op(xmm0, xmm1, xmm2));
            vcase(v, "avx-shift", format!("{}_yr", stringify!($op)), Pre::Int, |a| a.$op(ymm0, ymm1, xmm2));
            vcase(v, "avx-shift", format!("{}_ys", stringify!($op)), Pre::Int, |a| { a.vpsrlq(xmm5, xmm2, 58)?; a.$op(ymm0, ymm1, xmm5) });
            vcase(v, "avx-shift", format!("{}_ym", stringify!($op)), Pre::Int, |a| a.$op(ymm0, ymm1, xmmword_ptr(r8 + 160)));
        )* };
    }
    sh!(vpsllw, vpsrlw, vpsraw, vpslld, vpsrld, vpsrad, vpsllq, vpsrlq);
    for c in [0, 1, 5, 15, 16, 17, 255] {
        vcase(v, "avx-shift", format!("vpslldq_x{c}"), Pre::Int, move |a| a.vpslldq(xmm0, xmm1, c));
        vcase(v, "avx-shift", format!("vpslldq_y{c}"), Pre::Int, move |a| a.vpslldq(ymm0, ymm1, c));
        vcase(v, "avx-shift", format!("vpsrldq_x{c}"), Pre::Int, move |a| a.vpsrldq(xmm0, xmm1, c));
        vcase(v, "avx-shift", format!("vpsrldq_y{c}"), Pre::Int, move |a| a.vpsrldq(ymm0, ymm1, c));
    }
    vcase(v, "avx-blend", "vblendvps_x".into(), Pre::Int, |a| a.vblendvps(xmm0, xmm1, xmm2, xmm3));
    vcase(v, "avx-blend", "vblendvps_y".into(), Pre::Int, |a| a.vblendvps(ymm0, ymm1, ymm2, ymm3));
    vcase(v, "avx-blend", "vblendvps_ym".into(), Pre::Int, |a| a.vblendvps(ymm0, ymm1, ymmword_ptr(r8 + 160), ymm3));
    vcase(v, "avx-blend", "vblendvpd_x".into(), Pre::Int, |a| a.vblendvpd(xmm0, xmm1, xmm2, xmm3));
    vcase(v, "avx-blend", "vblendvpd_y".into(), Pre::Int, |a| a.vblendvpd(ymm0, ymm1, ymm2, ymm3));
    vcase(v, "avx-blend", "vpblendvb_x".into(), Pre::Int, |a| a.vpblendvb(xmm0, xmm1, xmm2, xmm3));
    vcase(v, "avx-blend", "vpblendvb_y".into(), Pre::Int, |a| a.vpblendvb(ymm0, ymm1, ymm2, ymm3));
    vcase(v, "avx-blend", "vpblendvb_ym".into(), Pre::Int, |a| a.vpblendvb(ymm0, ymm1, ymmword_ptr(r8 + 160), ymm3));
    vcase(v, "avx-mask", "vmaskmovps_ld_x".into(), Pre::Int, |a| a.vmaskmovps(xmm0, xmm3, xmmword_ptr(r8 + 160)));
    vcase(v, "avx-mask", "vmaskmovps_ld_y".into(), Pre::Int, |a| a.vmaskmovps(ymm0, ymm3, ymmword_ptr(r8 + 160)));
    vcase(v, "avx-mask", "vmaskmovpd_ld_y".into(), Pre::Int, |a| a.vmaskmovpd(ymm0, ymm3, ymmword_ptr(r8 + 160)));
    vcase(v, "avx-mask", "vpmaskmovd_ld_y".into(), Pre::Int, |a| a.vpmaskmovd(ymm0, ymm3, ymmword_ptr(r8 + 160)));
    vcase(v, "avx-mask", "vpmaskmovq_ld_x".into(), Pre::Int, |a| a.vpmaskmovq(xmm0, xmm3, xmmword_ptr(r8 + 160)));
    vcase(v, "avx-mask", "vmaskmovps_st_y".into(), Pre::Int, |a| { a.vmaskmovps(ymmword_ptr(r8 + SCRATCH), ymm3, ymm1)?; readback(a) });
    vcase(v, "avx-mask", "vmaskmovpd_st_x".into(), Pre::Int, |a| { a.vmaskmovpd(xmmword_ptr(r8 + SCRATCH), xmm3, xmm1)?; readback(a) });
    vcase(v, "avx-mask", "vpmaskmovd_st_y".into(), Pre::Int, |a| { a.vpmaskmovd(ymmword_ptr(r8 + SCRATCH), ymm3, ymm1)?; readback(a) });
    vcase(v, "avx-mask", "vpmaskmovq_st_y".into(), Pre::Int, |a| { a.vpmaskmovq(ymmword_ptr(r8 + SCRATCH), ymm3, ymm1)?; readback(a) });
}

fn flag_cases(v: &mut Vec<Case>) {
    macro_rules! fl_op {
        ($pre:expr, $flags:literal, $($op:ident),*) => { $(
            vcase(v, "avx-flags", format!("{}_x", stringify!($op)), $pre, |a| { a.$op(xmm1, xmm2)?; fl(a, $flags) });
        )* };
    }
    macro_rules! fl_op_y {
        ($($op:ident),*) => { $(
            vcase(v, "avx-flags", format!("{}_y", stringify!($op)), Pre::Int, |a| { a.$op(ymm1, ymm2)?; fl(a, "cz") });
            vcase(v, "avx-flags", format!("{}_ym", stringify!($op)), Pre::Int, |a| { a.$op(ymm1, ymmword_ptr(r8 + 160))?; fl(a, "cz") });
        )* };
    }
    // Rosetta 2 reports PF=1 after ptest/vtestps/vtestpd; the SDM says PF, SF, OF are cleared. Compare CF and ZF only.
    fl_op!(Pre::Int, "cz", vptest, vtestps, vtestpd);
    fl_op_y!(vptest, vtestps, vtestpd);
    vcase(v, "avx-flags", "vptest_same".into(), Pre::Int, |a| { a.vptest(ymm1, ymm1)?; fl(a, "cz") });
    vcase(v, "avx-flags", "vtestps_same".into(), Pre::Int, |a| { a.vtestps(ymm2, ymm2)?; fl(a, "cz") });
    fl_op!(Pre::FpS, "czsop", vucomiss, vcomiss);
    fl_op!(Pre::FpD, "czsop", vucomisd, vcomisd);
    vcase(v, "avx-flags", "vucomiss_m".into(), Pre::FpS, |a| { a.vucomiss(xmm1, dword_ptr(r8 + 160))?; fl(a, "czsop") });
    vcase(v, "avx-flags", "vcomisd_m".into(), Pre::FpD, |a| { a.vcomisd(xmm1, qword_ptr(r8 + 160))?; fl(a, "czsop") });
}

fn convert_cases(v: &mut Vec<Case>) {
    vcase(v, "avx-cvt", "vcvtsi2ss_32".into(), Pre::Int, |a| a.vcvtsi2ss(xmm0, xmm1, ecx));
    vcase(v, "avx-cvt", "vcvtsi2ss_64".into(), Pre::Int, |a| a.vcvtsi2ss(xmm0, xmm1, rcx));
    vcase(v, "avx-cvt", "vcvtsi2sd_32".into(), Pre::Int, |a| a.vcvtsi2sd(xmm0, xmm1, ecx));
    vcase(v, "avx-cvt", "vcvtsi2sd_64".into(), Pre::Int, |a| a.vcvtsi2sd(xmm0, xmm1, rcx));
    vcase(v, "avx-cvt", "vcvtsi2ss_m32".into(), Pre::Int, |a| a.vcvtsi2ss(xmm0, xmm1, dword_ptr(r8 + 160)));
    vcase(v, "avx-cvt", "vcvtsi2sd_m64".into(), Pre::Int, |a| a.vcvtsi2sd(xmm0, xmm1, qword_ptr(r8 + 160)));
    vcase(v, "avx-cvt", "vcvttss2si_32".into(), Pre::FpS, |a| a.vcvttss2si(eax, xmm1));
    vcase(v, "avx-cvt", "vcvttss2si_64".into(), Pre::FpS, |a| a.vcvttss2si(rax, xmm1));
    vcase(v, "avx-cvt", "vcvttsd2si_32".into(), Pre::FpD, |a| a.vcvttsd2si(eax, xmm1));
    vcase(v, "avx-cvt", "vcvttsd2si_64".into(), Pre::FpD, |a| a.vcvttsd2si(rax, xmm1));
    vcase(v, "avx-cvt", "vcvtss2si_32".into(), Pre::FpS, |a| a.vcvtss2si(eax, xmm1));
    vcase(v, "avx-cvt", "vcvtss2si_64".into(), Pre::FpS, |a| a.vcvtss2si(rax, xmm1));
    vcase(v, "avx-cvt", "vcvtsd2si_32".into(), Pre::FpD, |a| a.vcvtsd2si(eax, xmm1));
    vcase(v, "avx-cvt", "vcvtsd2si_64".into(), Pre::FpD, |a| a.vcvtsd2si(rax, xmm1));
    vcase(v, "avx-cvt", "vcvttss2si_m".into(), Pre::FpS, |a| a.vcvttss2si(eax, dword_ptr(r8 + 160)));
    vcase(v, "avx-cvt", "vcvtss2sd".into(), Pre::FpS, |a| a.vcvtss2sd(xmm0, xmm1, xmm2));
    vcase(v, "avx-cvt", "vcvtsd2ss".into(), Pre::FpD, |a| a.vcvtsd2ss(xmm0, xmm1, xmm2));
    vcase(v, "avx-cvt", "vcvtss2sd_m".into(), Pre::FpS, |a| a.vcvtss2sd(xmm0, xmm1, dword_ptr(r8 + 160)));
    vcase(v, "avx-cvt", "vcvtsd2ss_m".into(), Pre::FpD, |a| a.vcvtsd2ss(xmm0, xmm1, qword_ptr(r8 + 160)));
    vcase(v, "avx-cvt", "vcvtps2pd_x".into(), Pre::FpS, |a| a.vcvtps2pd(xmm0, xmm2));
    vcase(v, "avx-cvt", "vcvtps2pd_y".into(), Pre::FpS, |a| a.vcvtps2pd(ymm0, xmm2));
    vcase(v, "avx-cvt", "vcvtps2pd_xm".into(), Pre::FpS, |a| a.vcvtps2pd(xmm0, qword_ptr(r8 + 160)));
    vcase(v, "avx-cvt", "vcvtps2pd_ym".into(), Pre::FpS, |a| a.vcvtps2pd(ymm0, xmmword_ptr(r8 + 160)));
    vcase(v, "avx-cvt", "vcvtdq2pd_x".into(), Pre::Int, |a| a.vcvtdq2pd(xmm0, xmm2));
    vcase(v, "avx-cvt", "vcvtdq2pd_y".into(), Pre::Int, |a| a.vcvtdq2pd(ymm0, xmm2));
    vcase(v, "avx-cvt", "vcvtdq2pd_ym".into(), Pre::Int, |a| a.vcvtdq2pd(ymm0, xmmword_ptr(r8 + 160)));
    vcase(v, "avx-cvt", "vcvtpd2ps_x".into(), Pre::FpD, |a| a.vcvtpd2ps(xmm0, xmm2));
    vcase(v, "avx-cvt", "vcvtpd2ps_y".into(), Pre::FpD, |a| a.vcvtpd2ps(xmm0, ymm2));
    vcase(v, "avx-cvt", "vcvtpd2ps_xm".into(), Pre::FpD, |a| a.vcvtpd2ps(xmm0, xmmword_ptr(r8 + 160)));
    vcase(v, "avx-cvt", "vcvtpd2ps_ym".into(), Pre::FpD, |a| a.vcvtpd2ps(xmm0, ymmword_ptr(r8 + 160)));
    vcase(v, "avx-cvt", "vcvttpd2dq_y".into(), Pre::FpD, |a| a.vcvttpd2dq(xmm0, ymm2));
    vcase(v, "avx-cvt", "vcvttpd2dq_x".into(), Pre::FpD, |a| a.vcvttpd2dq(xmm0, xmm2));
    vcase(v, "avx-cvt", "vcvtpd2dq_y".into(), Pre::FpD, |a| a.vcvtpd2dq(xmm0, ymm2));
    vcase(v, "avx-cvt", "vcvtpd2dq_x".into(), Pre::FpD, |a| a.vcvtpd2dq(xmm0, xmm2));
    vcase(v, "avx-f16", "vcvtph2ps_x".into(), Pre::Int, |a| a.vcvtph2ps(xmm0, xmm2));
    vcase(v, "avx-f16", "vcvtph2ps_y".into(), Pre::Int, |a| a.vcvtph2ps(ymm0, xmm2));
    vcase(v, "avx-f16", "vcvtph2ps_xm".into(), Pre::Int, |a| a.vcvtph2ps(xmm0, qword_ptr(r8 + 160)));
    vcase(v, "avx-f16", "vcvtph2ps_ym".into(), Pre::Int, |a| a.vcvtph2ps(ymm0, xmmword_ptr(r8 + 160)));
    for m in [0, 1, 2, 3, 4] {
        vcase(v, "avx-f16", format!("vcvtps2ph_x{m}"), Pre::FpS, move |a| a.vcvtps2ph(xmm0, xmm2, m));
        vcase(v, "avx-f16", format!("vcvtps2ph_y{m}"), Pre::FpS, move |a| a.vcvtps2ph(xmm0, ymm2, m));
        vcase(v, "avx-f16", format!("vcvtps2ph_ms{m}"), Pre::FpS, move |a| { a.vcvtps2ph(qword_ptr(r8 + SCRATCH), xmm2, m)?; readback(a) });
        vcase(v, "avx-f16", format!("vcvtps2ph_my{m}"), Pre::FpS, move |a| { a.vcvtps2ph(xmmword_ptr(r8 + SCRATCH), ymm2, m)?; readback(a) });
    }
}

fn fma_cases(v: &mut Vec<Case>) {
    macro_rules! fma {
        ($pre:expr, $($op:ident),*) => { $(
            vcase(v, "avx-fma", format!("{}_x", stringify!($op)), $pre, |a| a.$op(xmm0, xmm1, xmm2));
            vcase(v, "avx-fma", format!("{}_y", stringify!($op)), $pre, |a| a.$op(ymm0, ymm1, ymm2));
            vcase(v, "avx-fma", format!("{}_ym", stringify!($op)), $pre, |a| a.$op(ymm0, ymm1, ymmword_ptr(r8 + 160)));
        )* };
    }
    macro_rules! fma_s {
        ($pre:expr, $mem:ident, $($op:ident),*) => { $(
            vcase(v, "avx-fma", format!("{}_r", stringify!($op)), $pre, |a| a.$op(xmm0, xmm1, xmm2));
            vcase(v, "avx-fma", format!("{}_m", stringify!($op)), $pre, |a| a.$op(xmm0, xmm1, $mem(r8 + 160)));
        )* };
    }
    fma!(Pre::FpS, vfmadd132ps, vfmadd213ps, vfmadd231ps, vfmsub132ps, vfmsub213ps, vfmsub231ps,
         vfnmadd132ps, vfnmadd213ps, vfnmadd231ps, vfnmsub132ps, vfnmsub213ps, vfnmsub231ps,
         vfmaddsub132ps, vfmaddsub213ps, vfmaddsub231ps, vfmsubadd132ps, vfmsubadd213ps, vfmsubadd231ps);
    fma!(Pre::FpD, vfmadd132pd, vfmadd213pd, vfmadd231pd, vfmsub132pd, vfmsub213pd, vfmsub231pd,
         vfnmadd132pd, vfnmadd213pd, vfnmadd231pd, vfnmsub132pd, vfnmsub213pd, vfnmsub231pd,
         vfmaddsub132pd, vfmaddsub213pd, vfmaddsub231pd, vfmsubadd132pd, vfmsubadd213pd, vfmsubadd231pd);
    fma_s!(Pre::FpS, dword_ptr, vfmadd132ss, vfmadd213ss, vfmadd231ss, vfmsub132ss, vfmsub213ss, vfmsub231ss,
           vfnmadd132ss, vfnmadd213ss, vfnmadd231ss, vfnmsub132ss, vfnmsub213ss, vfnmsub231ss);
    fma_s!(Pre::FpD, qword_ptr, vfmadd132sd, vfmadd213sd, vfmadd231sd, vfmsub132sd, vfmsub213sd, vfmsub231sd,
           vfnmadd132sd, vfnmadd213sd, vfnmadd231sd, vfnmsub132sd, vfnmsub213sd, vfnmsub231sd);
}

#[test]
fn vex_moves_broadcasts_and_lane_ops_match_x86() {
    check("vex_move", |v| {
        move_cases(v);
        lane_cases(v);
    });
}

#[test]
fn vex_shifts_blends_masks_and_flags_match_x86() {
    check("vex_shift", |v| {
        shift_cases(v);
        flag_cases(v);
    });
}

#[test]
fn vex_conversions_and_fma_match_x86() {
    check("vex_cvt", |v| {
        convert_cases(v);
        fma_cases(v);
    });
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
