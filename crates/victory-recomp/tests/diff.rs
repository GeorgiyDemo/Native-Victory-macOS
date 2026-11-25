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

mod common;

use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;

use common::*;
use iced_x86::code_asm::*;
use iced_x86::{BlockEncoderOptions, Decoder, DecoderOptions};
use victory_recomp::analyze::find_jump_tables_with;
use victory_recomp::image::Func;
use victory_recomp::lift::{Ctx, FuncIn, lift_unit};

const BASE: u64 = 0x1_4000_0000;

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Int,
    Fp,
    /// Only called from other snippets.
    Helper,
}

/// Labels shared between the case bodies and the table data.
struct Env {
    helper: CodeLabel,
    table: CodeLabel,
    jt: Vec<CodeLabel>,
}

type Build = Box<dyn Fn(&mut CodeAssembler, &mut Env) -> Result<(), IcedError>>;
struct Case {
    name: String,
    kind: Kind,
    build: Build,
    class: &'static str,
}

fn case(
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

fn fl(a: &mut CodeAssembler, m: &str) -> Result<(), IcedError> {
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

fn done(a: &mut CodeAssembler) -> Result<(), IcedError> {
    a.mov(qword_ptr(r8 + 8), rax)?;
    a.ret()
}

/// Registers for one operand size: (a, c, d, memory-pointer constructor).
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
macro_rules! all_sizes {
    ($v:expr, $class:expr, $nm:expr, |$a:ident, $ra:ident, $rc:ident, $rd:ident, $mp:ident| $body:expr) => {
        sizes!($v, $class, $nm, [8 al cl dl byte_ptr, 16 ax cx dx word_ptr, 32 eax ecx edx dword_ptr, 64 rax rcx rdx qword_ptr], |$a, $ra, $rc, $rd, $mp| $body)
    };
}
macro_rules! big_sizes {
    ($v:expr, $class:expr, $nm:expr, |$a:ident, $ra:ident, $rc:ident, $rd:ident, $mp:ident| $body:expr) => {
        sizes!($v, $class, $nm, [32 eax ecx edx dword_ptr, 64 rax rcx rdx qword_ptr], |$a, $ra, $rc, $rd, $mp| $body)
    };
}

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
fn assemble(cases: &[Case]) -> (Vec<u8>, Vec<(u32, u32)>) {
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
                .map(|l| ip(l).wrapping_sub(t) as u32)
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

fn driver_c(calls: &[(String, u32, Kind)]) -> String {
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
static void show(const char *n, const u8 *o) {
  u64 w[8]; memcpy(w, o, 64);
  printf("%s %llx %llx %llx %llx %llx %llx %llx %llx\n", n, (unsigned long long)w[0], (unsigned long long)w[1], (unsigned long long)w[2], (unsigned long long)w[3], (unsigned long long)w[4], (unsigned long long)w[5], (unsigned long long)w[6], (unsigned long long)w[7]);
}
int main(int argc, char **argv) {
  (void)argc; setvbuf(stdout, 0, _IOLBF, 0); load_image(argv[1]);
  u8 out[64];
  for (int it = 0; it < 300; it++) {
    u64 a = ig(), b = ig(); if (rnd() % 5 == 0) b = a; else if (rnd() % 9 == 0) b = a + 1;
    double x = dg(), y = dg(); if (rnd() % 6 == 0) y = x;
    (void)a; (void)b; (void)x; (void)y;
"#,
    );
    for (name, rva, kind) in calls {
        match kind {
            Kind::Int => writeln!(s, "    memset(out, 0, 64); fi({rva:#x}, a, b, out); printf(\"%llx %llx \", (unsigned long long)a, (unsigned long long)b); show(\"{name}\", out);").unwrap(),
            Kind::Fp => writeln!(s, "    memset(out, 0, 64); fd({rva:#x}, x, y, out); printf(\"%016llx %016llx \", *(unsigned long long *)&x, *(unsigned long long *)&y); show(\"{name}\", out);").unwrap(),
            Kind::Helper => {}
        }
    }
    s.push_str("  }\n  return 0;\n}\n");
    s
}

#[test]
fn lifted_instruction_classes_match_x86() {
    if !have_tools() {
        eprintln!("skipped: needs Rosetta 2 and clang");
        return;
    }
    let mut cases = vec![];
    int_cases(&mut cases);
    fp_cases(&mut cases);
    let (blob, spans) = assemble(&cases);

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
        1,
        "exactly one jump table must be recovered"
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

    let dir = workdir("diff");
    std::fs::write(dir.join("image.bin"), &blob).unwrap();
    std::fs::write(dir.join("unit.c"), &unit.c).unwrap();
    let calls: Vec<_> = cases
        .iter()
        .zip(&spans)
        .map(|(c, s)| (c.name.clone(), s.0, c.kind))
        .collect();
    std::fs::write(dir.join("driver.c"), driver_c(&calls)).unwrap();
    let (x86, arm) = run_both(&dir, "driver.c", "unit.c", &["image.bin"]);

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
        "{} snippets in {} classes x 300 random inputs matched: {classes:?}",
        cases.len(),
        classes.len()
    );
}
