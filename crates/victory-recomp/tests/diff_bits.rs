//! Differential tests for the prefixed and bit-twiddling instructions: `lock` RMW,
//! `xchg`/`xadd`/`cmpxchg`, `cmpxchg16b`, `rep` string ops, `bt*`, BMI1/BMI2,
//! shld/shrd, flag moves (`pushfq`/`popfq`/`lahf`/`sahf`) and the small leftovers.
//! Same snippet ABI as `diff.rs`. Atomicity itself cannot be told apart in a
//! single-threaded run; these tests pin the values and flags.
#![cfg(target_os = "macos")]
#![allow(unused_variables)]

#[macro_use]
mod common;

use common::*;
use iced_x86::code_asm::*;

fn finish(a: &mut CodeAssembler, flags: &str) -> Result<(), IcedError> {
    fl(a, flags)?;
    a.mov(rax, qword_ptr(r8 + 16))?;
    done(a)
}

fn lock_cases(v: &mut Vec<Case>) {
    macro_rules! lock_rmw {
        ($($op:ident),*) => { $(
            all_sizes!(v, "lock-rmw", concat!("lock_", stringify!($op)), |a, ra, rc, rd, mp| {
                a.mov(qword_ptr(r8 + 16), rcx)?;
                a.lock().$op(mp(r8 + 16), rd)?;
                finish(a, "czsop")
            });
            all_sizes!(v, "lock-rmw", concat!("plain_", stringify!($op)), |a, ra, rc, rd, mp| {
                a.mov(qword_ptr(r8 + 16), rcx)?;
                a.$op(mp(r8 + 16), rd)?;
                finish(a, "czsop")
            });
        )* };
    }
    lock_rmw!(add, sub, and, or, xor);
    macro_rules! lock_un {
        ($($op:ident $fl:literal),*) => { $(
            all_sizes!(v, "lock-rmw", concat!("lock_", stringify!($op)), |a, ra, rc, rd, mp| {
                a.cmp(rcx, rdx)?;
                a.mov(qword_ptr(r8 + 16), rcx)?;
                a.lock().$op(mp(r8 + 16))?;
                finish(a, $fl)
            });
        )* };
    }
    lock_un!(inc "czsop", dec "czsop", neg "czsop", not "czsop");
    all_sizes!(v, "lock-rmw", "xchg_mem", |a, ra, rc, rd, mp| {
        a.mov(qword_ptr(r8 + 16), rcx)?;
        a.xchg(mp(r8 + 16), rd)?;
        a.mov(qword_ptr(r8 + 24), rdx)?;
        a.mov(rax, qword_ptr(r8 + 16))?;
        done(a)
    });
    all_sizes!(v, "lock-rmw", "lock_xadd", |a, ra, rc, rd, mp| {
        a.mov(qword_ptr(r8 + 16), rcx)?;
        a.lock().xadd(mp(r8 + 16), rd)?;
        fl(a, "czsop")?;
        a.mov(qword_ptr(r8 + 24), rdx)?;
        a.mov(rax, qword_ptr(r8 + 16))?;
        done(a)
    });
    // Rosetta 2 computes the other flags as dest - accumulator; the lifter uses accumulator - dest (QEMU's model), so only ZF is compared.
    all_sizes!(v, "lock-rmw", "lock_cmpxchg_hit", |a, ra, rc, rd, mp| {
        a.mov(qword_ptr(r8 + 16), rcx)?;
        a.mov(rax, rcx)?;
        a.lock().cmpxchg(mp(r8 + 16), rd)?;
        fl(a, "z")?;
        a.mov(qword_ptr(r8 + 24), rax)?;
        a.mov(rax, qword_ptr(r8 + 16))?;
        done(a)
    });
    all_sizes!(v, "lock-rmw", "lock_cmpxchg_any", |a, ra, rc, rd, mp| {
        a.mov(qword_ptr(r8 + 16), rcx)?;
        a.mov(rax, rdx)?;
        a.lock().cmpxchg(mp(r8 + 16), rd)?;
        fl(a, "z")?;
        a.mov(qword_ptr(r8 + 24), rax)?;
        a.mov(rax, qword_ptr(r8 + 16))?;
        done(a)
    });
    all_sizes!(v, "lock-rmw", "cmpxchg_reg", |a, ra, rc, rd, mp| {
        a.mov(rax, rdx)?;
        a.mov(r9, rcx)?;
        a.cmpxchg(rc, rd)?;
        fl(a, "z")?;
        a.mov(qword_ptr(r8 + 24), rcx)?;
        done(a)
    });
    for (n, hit) in [("hit", true), ("miss", false)] {
        v.push(case("lock-rmw", format!("cmpxchg16b_{n}"), Kind::Int, move |a, _| {
            a.push(rbx)?;
            a.push(rsi)?;
            a.lea(rsi, ptr(r8 + 79))?;
            a.and(rsi, -16)?;
            a.mov(qword_ptr(rsi), rcx)?;
            a.mov(qword_ptr(rsi + 8), rdx)?;
            a.mov(r9, rdx)?;
            a.mov(r10, rcx)?;
            a.mov(rax, rcx)?;
            if hit {
                a.mov(rdx, r9)?;
            } else {
                a.lea(rdx, ptr(r9 + 1))?;
            }
            a.mov(rbx, 0x1111_2222_3333_4444_u64)?;
            a.mov(rcx, 0x5555_6666_7777_8888_u64)?;
            a.lock().cmpxchg16b(xmmword_ptr(rsi))?;
            fl(a, "z")?;
            a.mov(r11, qword_ptr(rsi))?;
            a.mov(qword_ptr(r8 + 16), r11)?;
            a.mov(r11, qword_ptr(rsi + 8))?;
            a.mov(qword_ptr(r8 + 24), r11)?;
            a.pop(rsi)?;
            a.pop(rbx)?;
            done(a)
        }));
    }
}

fn bt_cases(v: &mut Vec<Case>) {
    macro_rules! bt {
        ($($op:ident),*) => { $(
            sizes!(v, "bittest", concat!(stringify!($op), "_rr"), [16 ax cx dx word_ptr, 32 eax ecx edx dword_ptr, 64 rax rcx rdx qword_ptr], |a, ra, rc, rd, mp| {
                a.mov(rax, rcx)?;
                a.$op(ra, rd)?;
                fl(a, "c")?;
                done(a)
            });
            sizes!(v, "bittest", concat!(stringify!($op), "_ri"), [16 ax cx dx word_ptr, 32 eax ecx edx dword_ptr, 64 rax rcx rdx qword_ptr], |a, ra, rc, rd, mp| {
                a.mov(rax, rcx)?;
                a.$op(ra, 13)?;
                fl(a, "c")?;
                done(a)
            });
            sizes!(v, "bittest", concat!(stringify!($op), "_mi"), [16 ax cx dx word_ptr, 32 eax ecx edx dword_ptr, 64 rax rcx rdx qword_ptr], |a, ra, rc, rd, mp| {
                a.mov(qword_ptr(r8 + 16), rcx)?;
                a.$op(mp(r8 + 16), 37)?;
                fl(a, "c")?;
                a.mov(rax, qword_ptr(r8 + 16))?;
                done(a)
            });
            // Register bit offset into a bit string around out+128 (about +-32 bytes either way).
            sizes!(v, "bittest", concat!(stringify!($op), "_mr"), [16 ax cx dx word_ptr, 32 eax ecx edx dword_ptr, 64 rax rcx rdx qword_ptr], |a, ra, rc, rd, mp| {
                a.mov(qword_ptr(r8 + 120), rcx)?;
                a.mov(qword_ptr(r8 + 128), rdx)?;
                a.mov(qword_ptr(r8 + 136), rcx)?;
                a.mov(qword_ptr(r8 + 144), rdx)?;
                a.mov(rax, rdx)?;
                a.and(rax, 0x1ff)?;
                a.sub(rax, 0x100)?;
                a.$op(mp(r8 + 128), ra)?;
                fl(a, "c")?;
                a.mov(rax, qword_ptr(r8 + 128))?;
                a.mov(qword_ptr(r8 + 16), rax)?;
                a.mov(rax, qword_ptr(r8 + 120))?;
                done(a)
            });
        )* };
    }
    bt!(bt, bts, btr, btc);
    macro_rules! lbt {
        ($($op:ident),*) => { $(
            sizes!(v, "bittest", concat!("lock_", stringify!($op), "_mi"), [16 ax cx dx word_ptr, 32 eax ecx edx dword_ptr, 64 rax rcx rdx qword_ptr], |a, ra, rc, rd, mp| {
                a.mov(qword_ptr(r8 + 16), rcx)?;
                a.lock().$op(mp(r8 + 16), 9)?;
                fl(a, "c")?;
                a.mov(rax, qword_ptr(r8 + 16))?;
                done(a)
            });
            sizes!(v, "bittest", concat!("lock_", stringify!($op), "_mr"), [16 ax cx dx word_ptr, 32 eax ecx edx dword_ptr, 64 rax rcx rdx qword_ptr], |a, ra, rc, rd, mp| {
                a.mov(qword_ptr(r8 + 120), rcx)?;
                a.mov(qword_ptr(r8 + 128), rdx)?;
                a.mov(rax, rdx)?;
                a.and(rax, 0xff)?;
                a.sub(rax, 0x80)?;
                a.lock().$op(mp(r8 + 128), ra)?;
                fl(a, "c")?;
                a.mov(rax, qword_ptr(r8 + 128))?;
                a.mov(qword_ptr(r8 + 16), rax)?;
                a.mov(rax, qword_ptr(r8 + 120))?;
                done(a)
            });
        )* };
    }
    lbt!(bts, btr, btc);
}

fn bmi_cases(v: &mut Vec<Case>) {
    // andn dst, src1, src2 / bextr dst, src, ctl / bzhi / shifts.
    big_sizes!(v, "bmi", "andn", |a, ra, rc, rd, mp| { a.mov(rax, rdx)?; a.andn(ra, rc, rd)?; fl(a, "czso")?; done(a) });
    big_sizes!(v, "bmi", "andn_m", |a, ra, rc, rd, mp| { a.mov(qword_ptr(r8 + 16), rdx)?; a.andn(ra, rc, mp(r8 + 16))?; fl(a, "czso")?; done(a) });
    big_sizes!(v, "bmi", "bextr", |a, ra, rc, rd, mp| { a.mov(rax, rdx)?; a.bextr(ra, rc, rd)?; fl(a, "czo")?; done(a) });
    big_sizes!(v, "bmi", "bextr_m", |a, ra, rc, rd, mp| { a.mov(qword_ptr(r8 + 16), rcx)?; a.bextr(ra, mp(r8 + 16), rd)?; fl(a, "czo")?; done(a) });
    big_sizes!(v, "bmi", "bextr_small", |a, ra, rc, rd, mp| { a.mov(rax, rdx)?; a.and(edx, 0x1f1f)?; a.bextr(ra, rc, rd)?; fl(a, "czo")?; done(a) });
    big_sizes!(v, "bmi", "bzhi", |a, ra, rc, rd, mp| { a.mov(rax, rdx)?; a.bzhi(ra, rc, rd)?; fl(a, "czso")?; done(a) });
    big_sizes!(v, "bmi", "bzhi_small", |a, ra, rc, rd, mp| { a.and(edx, 0x7f)?; a.bzhi(ra, rc, rd)?; fl(a, "czso")?; done(a) });
    big_sizes!(v, "bmi", "pdep", |a, ra, rc, rd, mp| { a.pdep(ra, rc, rd)?; done(a) });
    big_sizes!(v, "bmi", "pext", |a, ra, rc, rd, mp| { a.pext(ra, rc, rd)?; done(a) });
    big_sizes!(v, "bmi", "shlx", |a, ra, rc, rd, mp| { a.shlx(ra, rc, rd)?; done(a) });
    big_sizes!(v, "bmi", "shrx", |a, ra, rc, rd, mp| { a.shrx(ra, rc, rd)?; done(a) });
    big_sizes!(v, "bmi", "sarx", |a, ra, rc, rd, mp| { a.sarx(ra, rc, rd)?; done(a) });
    big_sizes!(v, "bmi", "shlx_m", |a, ra, rc, rd, mp| { a.mov(qword_ptr(r8 + 16), rcx)?; a.shlx(ra, mp(r8 + 16), rd)?; done(a) });
    big_sizes!(v, "bmi", "rorx", |a, ra, rc, rd, mp| { a.rorx(ra, rc, 7)?; done(a) });
    big_sizes!(v, "bmi", "rorx0", |a, ra, rc, rd, mp| { a.rorx(ra, rc, 0)?; done(a) });
    macro_rules! bmi2 {
        ($fl:literal, $($op:ident),*) => { $(
            big_sizes!(v, "bmi", stringify!($op), |a, ra, rc, rd, mp| { a.mov(rax, rdx)?; a.$op(ra, rc)?; fl(a, $fl)?; done(a) });
            big_sizes!(v, "bmi", concat!(stringify!($op), "_m"), |a, ra, rc, rd, mp| { a.mov(qword_ptr(r8 + 16), rcx)?; a.$op(ra, mp(r8 + 16))?; fl(a, $fl)?; done(a) });
        )* };
    }
    bmi2!("czso", blsi, blsr, blsmsk);
    bmi2!("cz", tzcnt, lzcnt);
    sizes!(v, "bmi", "tzcnt16", [16 ax cx dx word_ptr], |a, ra, rc, rd, mp| { a.mov(rax, rdx)?; a.tzcnt(ra, rc)?; fl(a, "cz")?; done(a) });
    sizes!(v, "bmi", "lzcnt16", [16 ax cx dx word_ptr], |a, ra, rc, rd, mp| { a.mov(rax, rdx)?; a.lzcnt(ra, rc)?; fl(a, "cz")?; done(a) });
    // mulx hi, lo, src (rdx implicit)
    big_sizes!(v, "bmi", "mulx", |a, ra, rc, rd, mp| { a.cmp(rcx, rdx)?; a.mulx(ra, rc, rd)?; a.mov(qword_ptr(r8 + 16), rcx)?; fl(a, "czsop")?; done(a) });
    big_sizes!(v, "bmi", "mulx_m", |a, ra, rc, rd, mp| { a.mov(qword_ptr(r8 + 24), rcx)?; a.mulx(ra, rc, mp(r8 + 24))?; a.mov(qword_ptr(r8 + 16), rcx)?; done(a) });
    // bswap, movbe
    big_sizes!(v, "bmi", "bswap", |a, ra, rc, rd, mp| { a.mov(rax, rcx)?; a.bswap(ra)?; done(a) });
    sizes!(v, "bmi", "movbe_ld", [16 ax cx dx word_ptr, 32 eax ecx edx dword_ptr, 64 rax rcx rdx qword_ptr], |a, ra, rc, rd, mp| { a.mov(qword_ptr(r8 + 16), rcx)?; a.mov(rax, rdx)?; a.movbe(ra, mp(r8 + 16))?; done(a) });
    sizes!(v, "bmi", "movbe_st", [16 ax cx dx word_ptr, 32 eax ecx edx dword_ptr, 64 rax rcx rdx qword_ptr], |a, ra, rc, rd, mp| { a.mov(qword_ptr(r8 + 16), rdx)?; a.movbe(mp(r8 + 16), rc)?; a.mov(rax, qword_ptr(r8 + 16))?; done(a) });
    // adcx / adox read and write one flag each
    // adcx/adox are not tested: Rosetta 2 raises SIGILL on ADX.
}

fn misc_cases(v: &mut Vec<Case>) {
    for (n, cnt) in [("i5", 5), ("i1", 1), ("i0", 0), ("i31", 31)] {
        big_sizes!(v, "shld", &format!("shld_{n}"), |a, ra, rc, rd, mp| { a.cmp(rcx, rdx)?; a.mov(rax, rcx)?; a.shld(ra, rd, cnt)?; fl(a, "czsp")?; done(a) });
        big_sizes!(v, "shld", &format!("shrd_{n}"), |a, ra, rc, rd, mp| { a.cmp(rcx, rdx)?; a.mov(rax, rcx)?; a.shrd(ra, rd, cnt)?; fl(a, "czsp")?; done(a) });
    }
    big_sizes!(v, "shld", "shld_1o", |a, ra, rc, rd, mp| { a.cmp(rcx, rdx)?; a.mov(rax, rcx)?; a.shld(ra, rd, 1)?; fl(a, "czspo")?; done(a) });
    big_sizes!(v, "shld", "shrd_1o", |a, ra, rc, rd, mp| { a.cmp(rcx, rdx)?; a.mov(rax, rcx)?; a.shrd(ra, rd, 1)?; fl(a, "czspo")?; done(a) });
    big_sizes!(v, "shld", "shld_cl", |a, ra, rc, rd, mp| { a.cmp(rcx, rdx)?; a.mov(rax, rcx)?; a.mov(rcx, rdx)?; a.shld(ra, rd, cl)?; fl(a, "czsp")?; done(a) });
    big_sizes!(v, "shld", "shrd_cl", |a, ra, rc, rd, mp| { a.cmp(rcx, rdx)?; a.mov(rax, rcx)?; a.mov(rcx, rdx)?; a.shrd(ra, rd, cl)?; fl(a, "czsp")?; done(a) });
    v.push(case("misc", "cwd", Kind::Int, |a, _| { a.mov(eax, ecx)?; a.mov(rdx, 0x1234_5678_9abc_def0_u64)?; a.cwd()?; a.mov(qword_ptr(r8 + 16), rdx)?; done(a) }));
    // 8- and 16-bit division with a quotient that fits.
    let pos8 = |a: &mut CodeAssembler| -> Result<(), IcedError> { a.mov(r9, rdx)?; a.or(r9d, 8)?; a.and(r9d, 0x7f)?; a.mov(eax, ecx)?; a.and(eax, 0x3ff) };
    v.push(case("misc", "div8", Kind::Int, move |a, _| { pos8(a)?; a.mov(eax, ecx)?; a.and(eax, 0x7ff)?; a.div(r9b)?; a.mov(qword_ptr(r8 + 16), rdx)?; done(a) }));
    v.push(case("misc", "idiv8", Kind::Int, move |a, _| { pos8(a)?; a.idiv(r9b)?; a.mov(qword_ptr(r8 + 16), rdx)?; done(a) }));
    v.push(case("misc", "div16", Kind::Int, |a, _| { a.mov(r9, rdx)?; a.or(r9d, 5)?; a.and(r9d, 0x7fff)?; a.mov(eax, ecx)?; a.xor(edx, edx)?; a.div(r9w)?; a.mov(qword_ptr(r8 + 16), rdx)?; done(a) }));
    v.push(case("misc", "idiv16", Kind::Int, |a, _| { a.mov(r9, rdx)?; a.or(r9d, 5)?; a.and(r9d, 0x7fff)?; a.mov(eax, ecx)?; a.cwd()?; a.idiv(r9w)?; a.mov(qword_ptr(r8 + 16), rdx)?; done(a) }));
    // flag moves
    v.push(case("flags", "pushfq", Kind::Int, |a, _| { a.cmp(rcx, rdx)?; a.pushfq()?; a.pop(rax)?; a.and(eax, 0x8c5)?; done(a) }));
    v.push(case("flags", "pushfq_add", Kind::Int, |a, _| { a.mov(rax, rcx)?; a.add(rax, rdx)?; a.pushfq()?; a.pop(rax)?; a.and(eax, 0x8c5)?; done(a) }));
    v.push(case("flags", "popfq", Kind::Int, |a, _| { a.mov(rax, rcx)?; a.and(eax, 0x8c5)?; a.or(eax, 2)?; a.push(rax)?; a.popfq()?; fl(a, "czsop")?; a.mov(eax, 0)?; done(a) }));
    v.push(case("flags", "lahf", Kind::Int, |a, _| { a.cmp(rcx, rdx)?; a.lahf()?; a.movzx(eax, ah)?; a.and(eax, 0xc7)?; done(a) }));
    v.push(case("flags", "sahf", Kind::Int, |a, _| { a.mov(eax, ecx)?; a.shl(eax, 8)?; a.sahf()?; fl(a, "czsp")?; a.mov(eax, 0)?; done(a) }));
    // string ops (rsi/rdi are callee-saved)
    macro_rules! rep {
        ($($nm:literal $sz:literal $op:ident),*) => { $(
            v.push(case("string", $nm, Kind::Int, |a, _| {
                a.push(rsi)?;
                a.push(rdi)?;
                a.mov(r9, rcx)?;
                a.mov(r10, rdx)?;
                for k in 0..8 { a.mov(qword_ptr(r8 + 96 + 8 * k), if k % 2 == 0 { r9 } else { r10 })?; }
                a.lea(rsi, ptr(r8 + 96))?;
                a.lea(rdi, ptr(r8 + 160))?;
                a.mov(rax, r9)?;
                a.mov(rcx, r10)?;
                a.and(ecx, 7)?;
                a.rep().$op()?;
                a.sub(rsi, r8)?;
                a.sub(rdi, r8)?;
                a.mov(qword_ptr(r8 + 16), rsi)?;
                a.mov(qword_ptr(r8 + 24), rdi)?;
                a.mov(qword_ptr(r8 + 32), rcx)?;
                a.pop(rdi)?;
                a.pop(rsi)?;
                done(a)
            }));
        )* };
    }
    rep!("rep_movsb" 1 movsb, "rep_movsw" 2 movsw, "rep_movsd" 4 movsd, "rep_movsq" 8 movsq,
         "rep_stosb" 1 stosb, "rep_stosw" 2 stosw, "rep_stosd" 4 stosd, "rep_stosq" 8 stosq,
         "rep_lodsb" 1 lodsb, "rep_lodsw" 2 lodsw, "rep_lodsd" 4 lodsd, "rep_lodsq" 8 lodsq);
    macro_rules! cmp_s {
        ($($nm:literal $op:ident $pre:ident $acc:ident),*) => { $(
            v.push(case("string", $nm, Kind::Int, |a, _| {
                a.push(rsi)?;
                a.push(rdi)?;
                a.mov(r9, rcx)?;
                a.mov(r10, rdx)?;
                for k in 0..8 { a.mov(qword_ptr(r8 + 96 + 8 * k), r9)?; a.mov(qword_ptr(r8 + 160 + 8 * k), r9)?; }
                // make the buffers differ at a position taken from b
                a.mov(rax, r10)?;
                a.and(eax, 31)?;
                a.xor(byte_ptr(r8 + rax + 160), 0x5a)?;
                a.lea(rsi, ptr(r8 + 96))?;
                a.lea(rdi, ptr(r8 + 160))?;
                a.mov(rax, r9)?;
                a.mov(ecx, 40)?;
                a.$pre().$op()?;
                fl(a, "czsop")?;
                a.sub(rsi, r8)?;
                a.sub(rdi, r8)?;
                a.mov(qword_ptr(r8 + 16), rsi)?;
                a.mov(qword_ptr(r8 + 24), rdi)?;
                a.mov(qword_ptr(r8 + 32), rcx)?;
                a.pop(rdi)?;
                a.pop(rsi)?;
                done(a)
            }));
        )* };
    }
    cmp_s!("repe_cmpsb" cmpsb repe rax, "repe_cmpsw" cmpsw repe rax, "repe_cmpsd" cmpsd repe rax, "repe_cmpsq" cmpsq repe rax,
           "repne_cmpsb" cmpsb repne rax);
    v.push(case("string", "repne_scasb", Kind::Int, |a, _| {
        a.push(rdi)?;
        a.mov(r9, rcx)?;
        a.mov(r10, rdx)?;
        for k in 0..8 { a.mov(qword_ptr(r8 + 96 + 8 * k), r9)?; }
        a.lea(rdi, ptr(r8 + 96))?;
        a.mov(rax, r10)?;
        a.mov(ecx, 64)?;
        a.repne().scasb()?;
        fl(a, "czsop")?;
        a.sub(rdi, r8)?;
        a.mov(qword_ptr(r8 + 16), rdi)?;
        a.mov(qword_ptr(r8 + 32), rcx)?;
        a.pop(rdi)?;
        done(a)
    }));
    v.push(case("string", "repe_scasq", Kind::Int, |a, _| {
        a.push(rdi)?;
        a.mov(r9, rcx)?;
        for k in 0..8 { a.mov(qword_ptr(r8 + 96 + 8 * k), r9)?; }
        a.mov(rax, r9)?;
        a.mov(rcx, rdx)?;
        a.and(ecx, 1)?;
        a.mov(rdx, rcx)?;
        a.mov(ecx, 8)?;
        a.lea(rdi, ptr(r8 + 96))?;
        a.xor(qword_ptr(rdi + rdx * 8 + 24), 1)?;
        a.repe().scasq()?;
        fl(a, "czsop")?;
        a.sub(rdi, r8)?;
        a.mov(qword_ptr(r8 + 16), rdi)?;
        a.mov(qword_ptr(r8 + 32), rcx)?;
        a.pop(rdi)?;
        done(a)
    }));
}

fn check(label: &str, tables: usize, build: fn(&mut Vec<Case>)) {
    if !have_tools() {
        eprintln!("skipped: needs Rosetta 2 and clang");
        return;
    }
    let mut cases = vec![];
    build(&mut cases);
    run_cases(label, cases, tables);
}

#[test]
fn lock_xchg_cmpxchg_match_x86() {
    check("bits_lock", 0, lock_cases);
}

#[test]
fn bit_tests_match_x86() {
    check("bits_bt", 0, bt_cases);
}

#[test]
fn bmi_match_x86() {
    check("bits_bmi", 0, bmi_cases);
}

#[test]
fn string_flags_and_misc_match_x86() {
    check("bits_misc", 0, misc_cases);
}
