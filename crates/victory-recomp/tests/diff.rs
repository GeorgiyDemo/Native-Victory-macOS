//! Differential tests: every instruction class the lifter supports is
//! assembled into a tiny function, run as original x86-64 (Rosetta 2) and as
//! lifted arm64 on the same pseudo-random inputs, and the outputs (result,
//! captured flags, xmm lanes) must match bit for bit.
//!
//! Snippet ABI (Windows x64): `fn(rcx = a, rdx = b, r8 = out)` for integer
//! cases and `fn(xmm0 = a, xmm1 = b, r8 = out)` for floating point ones.
//! `out[0..5]` = CF ZF SF OF PF bytes, `out[8..16]` = result, `out[16..32]` =
//! extra result, `out[32..64]` is scratch. Only flags the architecture defines
//! for the instruction are captured.
#![cfg(target_os = "macos")]
#![allow(unused_variables)]

#[macro_use]
mod common;

use common::*;
use iced_x86::code_asm::*;


fn int_cases(v: &mut Vec<Case>) {
    // --- ALU, register and memory forms ---
    macro_rules! alu {
        ($op:ident, $pre_cf:expr) => {
            all_sizes!(v, "alu-reg", stringify!($op), |a, ra, rc, rd, _mp| {
                if $pre_cf {
                    a.cmp(rcx, rdx)?;
                }
                a.mov(rax, rcx)?;
                a.$op(ra, rd)?;
                fl(a, "czsop")?;
                done(a)
            });
            all_sizes!(
                v,
                "alu-mem-src",
                concat!(stringify!($op), "_ms"),
                |a, ra, rc, rd, mp| {
                    if $pre_cf {
                        a.cmp(rcx, rdx)?;
                    }
                    a.mov(rax, rcx)?;
                    a.mov(qword_ptr(r8 + 16), rdx)?;
                    a.$op(ra, mp(r8 + 16))?;
                    fl(a, "czsop")?;
                    done(a)
                }
            );
            all_sizes!(
                v,
                "alu-mem-dst",
                concat!(stringify!($op), "_md"),
                |a, ra, rc, rd, mp| {
                    if $pre_cf {
                        a.cmp(rcx, rdx)?;
                    }
                    a.mov(qword_ptr(r8 + 16), rcx)?;
                    a.$op(mp(r8 + 16), rd)?;
                    fl(a, "czsop")?;
                    a.mov(rax, qword_ptr(r8 + 16))?;
                    done(a)
                }
            );
        };
    }
    alu!(add, false);
    alu!(sub, false);
    alu!(adc, true);
    alu!(sbb, true);
    alu!(and, false);
    alu!(or, false);
    alu!(xor, false);
    // cmp/test write nothing but flags.
    all_sizes!(v, "alu-reg", "cmp", |a, ra, rc, rd, _mp| {
        a.mov(rax, rcx)?;
        a.cmp(ra, rd)?;
        fl(a, "czsop")?;
        done(a)
    });
    all_sizes!(v, "alu-reg", "test", |a, ra, rc, rd, _mp| {
        a.mov(rax, rcx)?;
        a.test(ra, rd)?;
        fl(a, "czsop")?;
        done(a)
    });
    big_sizes!(v, "alu-imm", "add_i", |a, ra, rc, rd, _mp| {
        a.mov(rax, rcx)?;
        a.add(ra, -0x1234_5678)?;
        fl(a, "czsop")?;
        done(a)
    });
    big_sizes!(v, "alu-imm", "sub_i8", |a, ra, rc, rd, _mp| {
        a.mov(rax, rcx)?;
        a.sub(ra, 0x12)?;
        fl(a, "czsop")?;
        done(a)
    });
    big_sizes!(v, "alu-imm", "cmp_i", |a, ra, rc, rd, _mp| {
        a.mov(rax, rcx)?;
        a.cmp(ra, 0x7fff_ffff)?;
        fl(a, "czsop")?;
        done(a)
    });
    big_sizes!(v, "alu-imm", "and_i", |a, ra, rc, rd, _mp| {
        a.mov(rax, rcx)?;
        a.and(ra, 0xff00_ff0f_u32 as i32)?;
        fl(a, "czsop")?;
        done(a)
    });
    // inc/dec keep CF, not changes no flags, neg is a subtraction from zero.
    all_sizes!(v, "incdec", "inc", |a, ra, rc, rd, _mp| {
        a.cmp(rcx, rdx)?;
        a.mov(rax, rcx)?;
        a.inc(ra)?;
        fl(a, "czsop")?;
        done(a)
    });
    all_sizes!(v, "incdec", "dec", |a, ra, rc, rd, _mp| {
        a.cmp(rcx, rdx)?;
        a.mov(rax, rcx)?;
        a.dec(ra)?;
        fl(a, "czsop")?;
        done(a)
    });
    all_sizes!(v, "incdec", "neg", |a, ra, rc, rd, _mp| {
        a.mov(rax, rcx)?;
        a.neg(ra)?;
        fl(a, "czsop")?;
        done(a)
    });
    all_sizes!(v, "incdec", "not", |a, ra, rc, rd, _mp| {
        a.cmp(rcx, rdx)?;
        a.mov(rax, rcx)?;
        a.not(ra)?;
        fl(a, "czsop")?;
        done(a)
    });
    all_sizes!(v, "incdec", "inc_mem", |a, ra, rc, rd, mp| {
        a.cmp(rcx, rdx)?;
        a.mov(qword_ptr(r8 + 16), rcx)?;
        a.inc(mp(r8 + 16))?;
        fl(a, "czsop")?;
        a.mov(rax, qword_ptr(r8 + 16))?;
        done(a)
    });

    // --- shifts and rotates ---
    macro_rules! shifts {
        ($op:ident) => {
            all_sizes!(
                v,
                "shift",
                concat!(stringify!($op), "_i5"),
                |a, ra, rc, rd, _mp| {
                    a.cmp(rcx, rdx)?;
                    a.mov(rax, rcx)?;
                    a.$op(ra, 5)?;
                    fl(a, "czsp")?;
                    done(a)
                }
            );
            all_sizes!(
                v,
                "shift",
                concat!(stringify!($op), "_1"),
                |a, ra, rc, rd, _mp| {
                    a.cmp(rcx, rdx)?;
                    a.mov(rax, rcx)?;
                    a.$op(ra, 1)?;
                    fl(a, "czsop")?;
                    done(a)
                }
            );
            big_sizes!(
                v,
                "shift",
                concat!(stringify!($op), "_cl"),
                |a, ra, rc, rd, _mp| {
                    a.cmp(rcx, rdx)?;
                    a.mov(rax, rcx)?;
                    a.mov(rcx, rdx)?;
                    a.$op(ra, cl)?;
                    fl(a, "czsp")?;
                    done(a)
                }
            );
        };
    }
    shifts!(shl);
    shifts!(shr);
    shifts!(sar);
    shifts!(rol);
    shifts!(ror);

    // --- multiply / divide ---
    sizes!(v, "mul", "imul2_", [16 ax cx dx word_ptr, 32 eax ecx edx dword_ptr, 64 rax rcx rdx qword_ptr], |a, ra, rc, rd, _mp| { a.mov(rax, rcx)?; a.imul_2(ra, rd)?; fl(a, "co")?; done(a) });
    sizes!(v, "mul", "imul3_", [16 ax cx dx word_ptr, 32 eax ecx edx dword_ptr, 64 rax rcx rdx qword_ptr], |a, ra, rc, rd, _mp| { a.mov(rax, rcx)?; a.imul_3(ra, rd, 0x1234)?; fl(a, "co")?; done(a) });
    all_sizes!(v, "mul", "mul1_", |a, ra, rc, rd, _mp| {
        a.mov(rax, rcx)?;
        a.mul(rd)?;
        fl(a, "co")?;
        a.mov(qword_ptr(r8 + 16), rdx)?;
        done(a)
    });
    all_sizes!(v, "mul", "imul1_", |a, ra, rc, rd, _mp| {
        a.mov(rax, rcx)?;
        a.imul(rd)?;
        fl(a, "co")?;
        a.mov(qword_ptr(r8 + 16), rdx)?;
        done(a)
    });
    // Divisors are forced positive and odd; the dividend's high half is the zero/sign extension.
    let pos = |a: &mut CodeAssembler| -> Result<(), IcedError> {
        a.mov(r9, rdx)?;
        a.or(r9, 1)?;
        a.shr(r9, 1)?;
        a.and(r9d, 0x7fff_ffff)?;
        a.or(r9, 1)?;
        a.mov(rax, rcx)
    };
    v.push(case("div", "div32", Kind::Int, move |a, _| {
        pos(a)?;
        a.xor(edx, edx)?;
        a.div(r9d)?;
        a.mov(qword_ptr(r8 + 16), rdx)?;
        done(a)
    }));
    v.push(case("div", "div64", Kind::Int, move |a, _| {
        pos(a)?;
        a.xor(edx, edx)?;
        a.div(r9)?;
        a.mov(qword_ptr(r8 + 16), rdx)?;
        done(a)
    }));
    v.push(case("div", "idiv32", Kind::Int, move |a, _| {
        pos(a)?;
        a.cdq()?;
        a.idiv(r9d)?;
        a.mov(qword_ptr(r8 + 16), rdx)?;
        done(a)
    }));
    v.push(case("div", "idiv64", Kind::Int, move |a, _| {
        pos(a)?;
        a.cqo()?;
        a.idiv(r9)?;
        a.mov(qword_ptr(r8 + 16), rdx)?;
        done(a)
    }));

    // --- moves, extension, lea ---
    let mv = |v: &mut Vec<Case>, n: &str, f: fn(&mut CodeAssembler) -> Result<(), IcedError>| {
        v.push(case("mov-ext", n, Kind::Int, move |a, _| {
            a.mov(rax, rdx)?;
            f(a)?;
            done(a)
        }))
    };
    mv(v, "movzx_r32_r8", |a| a.movzx(eax, cl));
    mv(v, "movzx_r32_r16", |a| a.movzx(eax, cx));
    mv(v, "movzx_r64_r8", |a| a.movzx(rax, cl));
    mv(v, "movsx_r32_r8", |a| a.movsx(eax, cl));
    mv(v, "movsx_r64_r16", |a| a.movsx(rax, cx));
    mv(v, "movsx_r16_r8", |a| a.movsx(ax, cl));
    mv(v, "movsxd", |a| a.movsxd(rax, ecx));
    mv(v, "mov_r32", |a| a.mov(eax, ecx));
    mv(v, "mov_r16", |a| a.mov(ax, cx));
    mv(v, "mov_r8", |a| a.mov(al, cl));
    mv(v, "mov_high8", |a| {
        a.mov(ah, cl)?;
        a.add(ah, dl)
    });
    mv(v, "lea_sib", |a| a.lea(rax, ptr(rcx + rdx * 4 + 0x10)));
    mv(v, "lea_r32", |a| a.lea(eax, ptr(rcx + rdx * 8 - 3)));
    mv(v, "xchg", |a| a.xchg(rax, rcx));
    mv(v, "movabs", |a| a.mov(rax, 0x1122_3344_5566_7788_u64));
    mv(v, "cdq", |a| {
        a.mov(eax, ecx)?;
        a.cdq()
    });
    mv(v, "cqo", |a| {
        a.mov(rax, rcx)?;
        a.cqo()?;
        a.mov(rax, rdx)
    });
    mv(v, "cdqe", |a| {
        a.mov(eax, ecx)?;
        a.cdqe()
    });
    mv(v, "cwde", |a| {
        a.mov(ax, cx)?;
        a.cwde()
    });
    big_sizes!(v, "bitscan", "bsf", |a, ra, rc, rd, _mp| {
        a.mov(rax, rdx)?;
        a.bsf(ra, rc)?;
        fl(a, "z")?;
        done(a)
    });
    big_sizes!(v, "bitscan", "bsr", |a, ra, rc, rd, _mp| {
        a.mov(rax, rdx)?;
        a.bsr(ra, rc)?;
        fl(a, "z")?;
        done(a)
    });
    big_sizes!(v, "bitscan", "popcnt", |a, ra, rc, rd, _mp| {
        a.popcnt(ra, rc)?;
        fl(a, "czsop")?;
        done(a)
    });

    // --- condition codes: setcc, cmovcc and jcc after a compare ---
    macro_rules! cc {
        ($($n:literal $set:ident $cmov:ident $j:ident),*) => { $(
            v.push(case("setcc", concat!("set", $n), Kind::Int, |a, _| { a.cmp(rcx, rdx)?; a.xor(eax, eax)?; a.$set(byte_ptr(r8 + 16))?; a.$set(al)?; a.cmp(rcx, rdx)?; done(a) }));
            v.push(case("cmov", concat!("cmov", $n), Kind::Int, |a, _| { a.cmp(rcx, rdx)?; a.mov(rax, rcx)?; a.mov(r9, rdx)?; a.$cmov(rax, r9)?; a.mov(r10d, 0x7777_7777)?; a.mov(r11, rcx)?; a.cmp(rcx, rdx)?; a.$cmov(r10d, r11d)?; a.mov(qword_ptr(r8 + 16), r10)?; done(a) }));
            v.push(case("jcc", concat!("j", $n), Kind::Int, |a, _| {
                let (mut taken, mut end) = (a.create_label(), a.create_label());
                a.cmp(rcx, rdx)?; a.$j(taken)?; a.mov(eax, 10)?; a.jmp(end)?;
                a.set_label(&mut taken)?; a.mov(eax, 20)?;
                a.set_label(&mut end)?; done(a)
            }));
        )* };
    }
    cc!("o" seto cmovo jo, "no" setno cmovno jno, "b" setb cmovb jb, "ae" setae cmovae jae, "e" sete cmove je, "ne" setne cmovne jne,
        "be" setbe cmovbe jbe, "a" seta cmova ja, "s" sets cmovs js, "ns" setns cmovns jns, "p" setp cmovp jp, "np" setnp cmovnp jnp,
        "l" setl cmovl jl, "ge" setge cmovge jge, "le" setle cmovle jle, "g" setg cmovg jg);

    // --- stack, direct and indirect calls, jump table ---
    v.push(case("helper", "helper", Kind::Helper, |a, _| {
        a.lea(rax, ptr(rcx + rdx * 2))?;
        a.ret()
    }));
    v.push(case("stack", "push_pop", Kind::Int, |a, _| {
        a.push(rcx)?;
        a.push(rdx)?;
        a.pop(rax)?;
        a.pop(r9)?;
        a.add(rax, r9)?;
        a.mov(qword_ptr(r8 + 16), r9)?;
        done(a)
    }));
    v.push(case("call", "call_direct", Kind::Int, |a, e| {
        a.push(r8)?;
        a.sub(rsp, 32)?;
        a.call(e.helper)?;
        a.add(rsp, 32)?;
        a.pop(r8)?;
        done(a)
    }));
    v.push(case("call", "call_indirect", Kind::Int, |a, e| {
        a.push(r8)?;
        a.sub(rsp, 32)?;
        a.lea(r9, ptr(e.helper))?;
        a.call(r9)?;
        a.add(rsp, 32)?;
        a.pop(r8)?;
        done(a)
    }));
    v.push(case("jmp", "jump_table", Kind::Int, |a, e| {
        let mut def = a.create_label();
        let mut fin = a.create_label();
        a.and(ecx, 7)?;
        a.cmp(rcx, 5)?;
        a.ja(def)?;
        a.lea(rdx, ptr(e.table))?;
        a.movsxd(rax, dword_ptr(rdx + rcx * 4))?;
        a.add(rax, rdx)?;
        a.jmp(rax)?;
        for i in 0..6 {
            a.set_label(&mut e.jt[i])?;
            a.mov(eax, 100 + i as i32 * 7)?;
            a.jmp(fin)?;
        }
        a.set_label(&mut def)?;
        a.mov(eax, 999)?;
        a.set_label(&mut fin)?;
        done(a)
    }));

    // --- sse integer/gpr moves and bitwise ---
    v.push(case("sse-mov", "movq_gpr", Kind::Int, |a, _| {
        a.pxor(xmm0, xmm0)?;
        a.movq(xmm0, rcx)?;
        a.movq(rax, xmm0)?;
        a.movups(xmmword_ptr(r8 + 16), xmm0)?;
        done(a)
    }));
    v.push(case("sse-mov", "movd_gpr", Kind::Int, |a, _| {
        a.pxor(xmm0, xmm0)?;
        a.movd(xmm0, ecx)?;
        a.movd(eax, xmm0)?;
        a.movups(xmmword_ptr(r8 + 16), xmm0)?;
        done(a)
    }));
    v.push(case("sse-mov", "movq_mem", Kind::Int, |a, _| {
        a.movq(qword_ptr(r8 + 32), xmm0)?;
        a.mov(qword_ptr(r8 + 32), rcx)?;
        a.movq(xmm1, qword_ptr(r8 + 32))?;
        a.movups(xmmword_ptr(r8 + 16), xmm1)?;
        a.mov(rax, rdx)?;
        done(a)
    }));
    v.push(case("sse-mov", "movaps_ld_st", Kind::Int, |a, _| {
        a.mov(qword_ptr(r8 + 32), rcx)?;
        a.mov(qword_ptr(r8 + 40), rdx)?;
        a.movups(xmm2, xmmword_ptr(r8 + 32))?;
        a.movaps(xmm3, xmm2)?;
        a.movdqa(xmm4, xmm3)?;
        a.movdqu(xmmword_ptr(r8 + 48), xmm4)?;
        a.mov(rax, qword_ptr(r8 + 56))?;
        done(a)
    }));
    for (n, f) in BITW {
        v.push(case("sse-bitwise", *n, Kind::Int, move |a, _| {
            a.movq(xmm0, rcx)?;
            a.movq(xmm1, rdx)?;
            f(a)?;
            a.movups(xmmword_ptr(r8 + 16), xmm0)?;
            a.mov(rax, rcx)?;
            done(a)
        }));
    }
    v.push(case("cvt", "cvtsi2sd64", Kind::Int, |a, _| {
        a.xor(eax, eax)?;
        a.pxor(xmm0, xmm0)?;
        a.cvtsi2sd(xmm0, rcx)?;
        a.movups(xmmword_ptr(r8 + 16), xmm0)?;
        done(a)
    }));
    v.push(case("cvt", "cvtsi2sd32", Kind::Int, |a, _| {
        a.xor(eax, eax)?;
        a.pxor(xmm0, xmm0)?;
        a.cvtsi2sd(xmm0, ecx)?;
        a.movups(xmmword_ptr(r8 + 16), xmm0)?;
        done(a)
    }));
    v.push(case("cvt", "cvtsi2ss64", Kind::Int, |a, _| {
        a.xor(eax, eax)?;
        a.pxor(xmm0, xmm0)?;
        a.cvtsi2ss(xmm0, rcx)?;
        a.movups(xmmword_ptr(r8 + 16), xmm0)?;
        done(a)
    }));
    v.push(case("cvt", "cvtsi2ss32", Kind::Int, |a, _| {
        a.xor(eax, eax)?;
        a.pxor(xmm0, xmm0)?;
        a.cvtsi2ss(xmm0, ecx)?;
        a.movups(xmmword_ptr(r8 + 16), xmm0)?;
        done(a)
    }));
}

type Emit = fn(&mut CodeAssembler) -> Result<(), IcedError>;
const BITW: &[(&str, Emit)] = &[
    ("pxor", |a| a.pxor(xmm0, xmm1)),
    ("xorps", |a| a.xorps(xmm0, xmm1)),
    ("xorpd", |a| a.xorpd(xmm0, xmm1)),
    ("pand", |a| a.pand(xmm0, xmm1)),
    ("andps", |a| a.andps(xmm0, xmm1)),
    ("andpd", |a| a.andpd(xmm0, xmm1)),
    ("pandn", |a| a.pandn(xmm0, xmm1)),
    ("andnps", |a| a.andnps(xmm0, xmm1)),
    ("por", |a| a.por(xmm0, xmm1)),
    ("orps", |a| a.orps(xmm0, xmm1)),
    ("orpd", |a| a.orpd(xmm0, xmm1)),
];

fn fp_cases(v: &mut Vec<Case>) {
    // Inputs arrive as doubles; single-precision cases convert first, which also tests cvtsd2ss.
    fn pre(a: &mut CodeAssembler) -> Result<(), IcedError> {
        a.movq(xmm0, xmm0)?;
        a.movq(xmm1, xmm1)
    }
    macro_rules! sd {
        ($($op:ident),*) => { $(
            v.push(case("sse-scalar", stringify!($op), Kind::Fp, move |a, _| { pre(a)?; a.$op(xmm0, xmm1)?; a.movsd_2(qword_ptr(r8 + 8), xmm0)?; a.movups(xmmword_ptr(r8 + 16), xmm0)?; a.ret() }));
        )* };
    }
    sd!(addsd, subsd, mulsd, divsd, minsd, maxsd, sqrtsd);
    macro_rules! ss {
        ($($op:ident),*) => { $(
            v.push(case("sse-scalar", stringify!($op), Kind::Fp, move |a, _| { pre(a)?; a.cvtsd2ss(xmm0, xmm0)?; a.cvtsd2ss(xmm1, xmm1)?; a.$op(xmm0, xmm1)?; a.movss(dword_ptr(r8 + 8), xmm0)?; a.movups(xmmword_ptr(r8 + 16), xmm0)?; a.ret() }));
        )* };
    }
    ss!(addss, subss, mulss, divss, minss, maxss, sqrtss);
    // Packed: build (a,b) / (b,a) from scratch memory so no unsupported shuffles are needed.
    macro_rules! pd {
        ($($op:ident),*) => { $(
            v.push(case("sse-packed", stringify!($op), Kind::Fp, move |a, _| {
                pre(a)?; a.movsd_2(qword_ptr(r8 + 32), xmm0)?; a.movsd_2(qword_ptr(r8 + 40), xmm1)?; a.movsd_2(qword_ptr(r8 + 48), xmm1)?; a.movsd_2(qword_ptr(r8 + 56), xmm0)?;
                a.movups(xmm2, xmmword_ptr(r8 + 32))?; a.movups(xmm3, xmmword_ptr(r8 + 48))?; a.$op(xmm2, xmm3)?; a.movups(xmmword_ptr(r8 + 16), xmm2)?; a.ret()
            }));
        )* };
    }
    pd!(addpd, subpd, mulpd, divpd);
    macro_rules! ps {
        ($($op:ident),*) => { $(
            v.push(case("sse-packed", stringify!($op), Kind::Fp, move |a, _| {
                pre(a)?; a.cvtsd2ss(xmm0, xmm0)?; a.cvtsd2ss(xmm1, xmm1)?;
                a.movss(dword_ptr(r8 + 32), xmm0)?; a.movss(dword_ptr(r8 + 36), xmm1)?; a.movss(dword_ptr(r8 + 40), xmm1)?; a.movss(dword_ptr(r8 + 44), xmm0)?;
                a.movss(dword_ptr(r8 + 48), xmm1)?; a.movss(dword_ptr(r8 + 52), xmm0)?; a.movss(dword_ptr(r8 + 56), xmm0)?; a.movss(dword_ptr(r8 + 60), xmm1)?;
                a.movups(xmm2, xmmword_ptr(r8 + 32))?; a.movups(xmm3, xmmword_ptr(r8 + 48))?; a.$op(xmm2, xmm3)?; a.movups(xmmword_ptr(r8 + 16), xmm2)?; a.ret()
            }));
        )* };
    }
    ps!(addps, subps, mulps, divps);
    macro_rules! cmpf {
        ($($op:ident $sz:literal),*) => { $(
            v.push(case("sse-compare", stringify!($op), Kind::Fp, move |a, _| {
                pre(a)?; if $sz == 32 { a.cvtsd2ss(xmm0, xmm0)?; a.cvtsd2ss(xmm1, xmm1)?; } a.$op(xmm0, xmm1)?; fl(a, "czsop")?; a.ret()
            }));
        )* };
    }
    cmpf!(ucomisd 64, comisd 64, ucomiss 32, comiss 32);
    v.push(case("cvt", "cvttsd2si64", Kind::Fp, |a, _| {
        a.cvttsd2si(rax, xmm0)?;
        a.mov(qword_ptr(r8 + 8), rax)?;
        a.ret()
    }));
    v.push(case("cvt", "cvttsd2si32", Kind::Fp, |a, _| {
        a.cvttsd2si(eax, xmm0)?;
        a.mov(qword_ptr(r8 + 8), rax)?;
        a.ret()
    }));
    v.push(case("cvt", "cvttss2si64", Kind::Fp, |a, _| {
        a.cvtsd2ss(xmm0, xmm0)?;
        a.cvttss2si(rax, xmm0)?;
        a.mov(qword_ptr(r8 + 8), rax)?;
        a.ret()
    }));
    v.push(case("cvt", "cvttss2si32", Kind::Fp, |a, _| {
        a.cvtsd2ss(xmm0, xmm0)?;
        a.cvttss2si(eax, xmm0)?;
        a.mov(qword_ptr(r8 + 8), rax)?;
        a.ret()
    }));
    v.push(case("cvt", "cvtss2sd", Kind::Fp, |a, _| {
        pre(a)?;
        a.cvtsd2ss(xmm0, xmm0)?;
        a.cvtss2sd(xmm1, xmm0)?;
        a.movups(xmmword_ptr(r8 + 16), xmm1)?;
        a.ret()
    }));
    v.push(case("sse-mov", "movsd_reg_mem", Kind::Fp, |a, _| {
        pre(a)?;
        a.movsd_2(qword_ptr(r8 + 32), xmm0)?;
        a.movups(xmm2, xmmword_ptr(r8 + 48))?;
        a.movsd_2(xmm2, xmm1)?;
        a.movups(xmmword_ptr(r8 + 16), xmm2)?;
        a.movsd_2(xmm3, qword_ptr(r8 + 32))?;
        a.movups(xmmword_ptr(r8 + 48), xmm3)?;
        a.ret()
    }));
    v.push(case("sse-mov", "movss_reg_mem", Kind::Fp, |a, _| {
        pre(a)?;
        a.cvtsd2ss(xmm0, xmm0)?;
        a.movss(dword_ptr(r8 + 32), xmm0)?;
        a.movss(xmm2, xmm1)?;
        a.movups(xmmword_ptr(r8 + 16), xmm2)?;
        a.movss(xmm3, dword_ptr(r8 + 32))?;
        a.movups(xmmword_ptr(r8 + 48), xmm3)?;
        a.ret()
    }));
}

/// Assembles every case into one blob; returns (bytes, per-case (begin, end) offsets).

#[test]
fn lifted_instruction_classes_match_x86() {
    if !have_tools() {
        eprintln!("skipped: needs Rosetta 2 and clang");
        return;
    }
    let mut cases = vec![];
    int_cases(&mut cases);
    fp_cases(&mut cases);
    run_cases("diff", cases, 1);
}
