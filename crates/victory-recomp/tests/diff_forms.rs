//! Instruction-form differential test. Every x86-64 form (an iced `Code`: mnemonic, operand kinds, prefixes) that the
//! lifter supports is encoded synthetically with register, immediate and memory operands (no bytes of any title),
//! wrapped in a snippet that loads the whole register file, the flags and a memory block from pseudo-random inputs,
//! runs the instruction, and dumps all of it. The x86-64 original runs under Rosetta 2 and the lifted C on arm64, and
//! every difference in a GPR, a defined flag, a ymm register or the memory block is reported with its encoding.
//!
//! `VICTORY_FORMS` picks forms by mnemonic substring (comma separated, e.g. `vpermps,shl`), `VICTORY_FORMS_BATCH`
//! sets the snippets per build (default 400), `VICTORY_FORMS_LIST=<file>` restricts to the forms named in a file of
//! `Code` names (as `victory-recomp forms <pe>` prints them). The report goes to stderr and to
//! `<target tmp>/diff_forms/report.txt`.
#![cfg(target_os = "macos")]
#![allow(unused_variables)]

mod common;

use std::collections::{BTreeMap, BTreeSet};
use std::process::Command;

use common::*;
use iced_x86::code_asm::*;
use iced_x86::{
    Code, Decoder, DecoderOptions, Encoder, FlowControl, Formatter, Instruction, InstructionInfoFactory, IntelFormatter,
    OpCodeOperandKind, OpKind, Register,
};

const A: i64 = 0x40; // aligned displacement from r14
const U: i64 = 0x43; // misaligned

/// Differences with the x86 oracle (Rosetta 2), not with the manuals: prefix and reason.
const WAIVERS: &[(&str, &str)] = &[
    ("Shld_rm16", "count above 16 is undefined for 16-bit operands; Rosetta, Intel and AMD differ"),
    ("Shrd_rm16", "count above 16 is undefined for 16-bit operands; Rosetta, Intel and AMD differ"),
    ("Movsxd_r32_rm32", "Rosetta sign-extends the 32-bit-destination form into 64 bits; the manual says a plain 32-bit move"),
    ("Movsxd_r16_rm16", "Rosetta sign-extends the 16-bit-destination form; the manual says a plain 16-bit move"),
    ("Lea_", "the result is a host address, which differs between the two processes"),
    ("VEX_Vpsllvq", "Rosetta truncates the per-lane count to 32 bits"),
    ("VEX_Vpsrlvq", "Rosetta truncates the per-lane count to 32 bits"),
    ("VEX_Vcvtps2ph", "Rosetta reads the reserved bits of the immediate"),
];

/// Forms Rosetta 2 cannot run at all.
const ROSETTA_LACKS: &[&str] = &["Adcx_", "Adox_", "VEX_Vaes", "VEX_Vpclmulqdq_ymm"];

/// One synthetic test: the encoded instruction and what to mask.
struct Form {
    name: String,
    code: Code,
    bytes: Vec<u8>,
    /// Flags the instruction leaves defined (compared).
    flag_mask: u32,
    kind: Kind,
}

fn skip_mnemonic(m: &str) -> bool {
    const EXACT: &[&str] = &[
        "div", "idiv", "cpuid", "rdtsc", "rdtscp", "xgetbv", "xsetbv", "syscall", "sysret", "sysenter", "sysexit", "int",
        "into", "int3", "int1", "ud0", "ud1", "ud2", "hlt", "rdrand", "rdseed", "rdpid", "ldmxcsr", "stmxcsr", "vldmxcsr",
        "vstmxcsr", "wait", "emms", "femms", "monitor", "mwait", "monitorx", "mwaitx", "clzero", "xlatb", "enter", "leave",
        "cmpxchg16b", "cmpxchg8b", "clflush", "clflushopt", "clwb", "maskmovdqu", "vmaskmovdqu", "maskmovq", "lfs", "lgs",
        "lss", "pause", "insertq", "extrq", "lahf", "sahf", "pushf", "popf", "cli", "sti", "std", "cld", "xend", "xabort",
        "xbegin", "xtest", "rdpmc", "rdmsr", "wrmsr", "in", "out", "iret", "lar", "lsl", "verr", "verw", "arpl", "bound",
        "invlpg", "invd", "wbinvd", "swapgs", "rdfsbase", "rdgsbase", "wrfsbase", "wrgsbase", "rdsspd", "rdsspq", "incsspd",
        "incsspq", "xchg", "nop", "fnstcw", "fldcw", "vzeroall", "mfence", "lfence", "sfence", "serialize", "ldtilecfg",
        "sttilecfg", "movdir64b", "movdiri",
    ];
    const PREFIX: &[&str] = &[
        "movs", "cmps", "scas", "lods", "stos", "ins", "outs", "prefetch", "f", "xsave", "xrstor", "fxsave", "fxrstor",
        "vgather", "vpgather", "vscatter", "vpscatter", "jmp", "call", "ret", "loop", "push", "pop", "cmov_",
    ];
    // `movsb/movsw/movsd/movsq` with no operands are the string forms; `movsd xmm, xmm` is the scalar double: the
    // string ones are dropped by operand kind (es_rDI) below.
    EXACT.contains(&m) || PREFIX.iter().any(|p| m.starts_with(p) && !matches!(m, "movsxd" | "movsx" | "movss" | "movsldup" | "movshdup" | "movsd" | "insertps" | "insertq" | "pinsrb" | "pinsrw" | "pinsrd" | "pinsrq" | "lddqu" | "popcnt" | "stosd_" ) && !(*p == "ins" && (m.starts_with("insert") || m == "insertps")) && !(*p == "f" && false))
}

/// Registers an operand kind can take, by a pool index.
fn reg_for(kind: OpCodeOperandKind, idx: usize) -> Option<Register> {
    use OpCodeOperandKind as K;
    // r14 is the memory base and r15 the output block; rsp is not touched.
    const R8: [Register; 12] = [
        Register::AL, Register::CL, Register::DL, Register::BL, Register::SIL, Register::DIL, Register::BPL, Register::R8L,
        Register::R9L, Register::R10L, Register::R11L, Register::R12L,
    ];
    const R16: [Register; 13] = [
        Register::AX, Register::CX, Register::DX, Register::BX, Register::BP, Register::SI, Register::DI, Register::R8W,
        Register::R9W, Register::R10W, Register::R11W, Register::R12W, Register::R13W,
    ];
    const R32: [Register; 13] = [
        Register::EAX, Register::ECX, Register::EDX, Register::EBX, Register::EBP, Register::ESI, Register::EDI,
        Register::R8D, Register::R9D, Register::R10D, Register::R11D, Register::R12D, Register::R13D,
    ];
    const R64: [Register; 13] = [
        Register::RAX, Register::RCX, Register::RDX, Register::RBX, Register::RBP, Register::RSI, Register::RDI,
        Register::R8, Register::R9, Register::R10, Register::R11, Register::R12, Register::R13,
    ];
    let xmm = |i: usize| Register::XMM0 + (i % 16) as u32;
    let ymm = |i: usize| Register::YMM0 + (i % 16) as u32;
    Some(match kind {
        K::r8_or_mem | K::r8_reg | K::r8_opcode => R8[idx % R8.len()],
        K::r16_or_mem | K::r16_reg | K::r16_reg_mem | K::r16_rm | K::r16_opcode => R16[idx % R16.len()],
        K::r32_or_mem | K::r32_reg | K::r32_reg_mem | K::r32_rm | K::r32_opcode | K::r32_vvvv => R32[idx % R32.len()],
        K::r64_or_mem | K::r64_reg | K::r64_reg_mem | K::r64_rm | K::r64_opcode | K::r64_vvvv => R64[idx % R64.len()],
        K::xmm_or_mem | K::xmm_reg | K::xmm_rm | K::xmm_vvvv | K::xmm_is4 | K::xmm_is5 | K::xmmp3_vvvv => xmm(idx),
        K::ymm_or_mem | K::ymm_reg | K::ymm_rm | K::ymm_vvvv | K::ymm_is4 | K::ymm_is5 => ymm(idx),
        K::al => Register::AL,
        K::cl => Register::CL,
        K::ax => Register::AX,
        K::dx => Register::DX,
        K::eax => Register::EAX,
        K::rax => Register::RAX,
        _ => return None,
    })
}

fn has_mem_alt(kind: OpCodeOperandKind) -> bool {
    use OpCodeOperandKind as K;
    matches!(
        kind,
        K::r8_or_mem | K::r16_or_mem | K::r32_or_mem | K::r64_or_mem | K::xmm_or_mem | K::ymm_or_mem | K::mem
    )
}

fn is_imm(kind: OpCodeOperandKind) -> bool {
    use OpCodeOperandKind as K;
    matches!(
        kind,
        K::imm8 | K::imm8_const_1 | K::imm8sex16 | K::imm8sex32 | K::imm8sex64 | K::imm16 | K::imm32 | K::imm32sex64 | K::imm64
    )
}

fn supported_kind(kind: OpCodeOperandKind) -> bool {
    use OpCodeOperandKind as K;
    reg_for(kind, 0).is_some() || has_mem_alt(kind) || is_imm(kind) || kind == K::None
}

#[derive(Clone, Copy, PartialEq)]
enum Variant {
    Reg,
    Alias,
    Mem(i64),
}

fn build_instruction(code: Code, var: Variant, imm_sel: usize, seed: usize) -> Option<Instruction> {
    let oc = code.op_code();
    let mut ins = Instruction::default();
    ins.set_code(code);
    let n = oc.op_count();
    let imm8 = [0x03u64, 0x1f, 0x41, 0xa5][imm_sel % 4];
    let imm16 = [0x1234u64, 0x00ff, 0x8001, 0x0020][imm_sel % 4];
    let imm32 = [0x1234_5678u64, 0xffff_ff00, 0x8000_0001, 0x0000_0020][imm_sel % 4];
    let imm64 = [0x0123_4567_89ab_cdefu64, 0x8000_0000_0000_0000, 0xffff_ffff_ffff_ff80, 0x20][imm_sel % 4];
    let mut reg_k = 0usize;
    for i in 0..n {
        let kind = oc.op_kind(i);
        use OpCodeOperandKind as K;
        let mem = matches!(var, Variant::Mem(_)) && has_mem_alt(kind);
        if mem {
            let Variant::Mem(disp) = var else { unreachable!() };
            ins.set_op_kind(i, OpKind::Memory);
            ins.set_memory_base(Register::R14);
            ins.set_memory_index(Register::None);
            ins.set_memory_index_scale(1);
            ins.set_memory_displacement64(disp as u64);
            ins.set_memory_displ_size(1);
        } else if let Some(r) = reg_for(kind, if var == Variant::Alias { seed } else { seed + reg_k * 3 }) {
            ins.set_op_kind(i, OpKind::Register);
            ins.set_op_register(i, r);
            if !matches!(kind, K::al | K::cl | K::ax | K::dx | K::eax | K::rax) {
                reg_k += 1;
            }
        } else if is_imm(kind) {
            match kind {
                K::imm8_const_1 => {
                    ins.set_op_kind(i, OpKind::Immediate8);
                    ins.set_immediate8(1);
                }
                K::imm8 => {
                    ins.set_op_kind(i, OpKind::Immediate8);
                    ins.set_immediate8(imm8 as u8);
                }
                K::imm8sex16 => {
                    ins.set_op_kind(i, OpKind::Immediate8to16);
                    ins.set_immediate8to16(imm8 as u8 as i8 as i16);
                }
                K::imm8sex32 => {
                    ins.set_op_kind(i, OpKind::Immediate8to32);
                    ins.set_immediate8to32(imm8 as u8 as i8 as i32);
                }
                K::imm8sex64 => {
                    ins.set_op_kind(i, OpKind::Immediate8to64);
                    ins.set_immediate8to64(imm8 as u8 as i8 as i64);
                }
                K::imm16 => {
                    ins.set_op_kind(i, OpKind::Immediate16);
                    ins.set_immediate16(imm16 as u16);
                }
                K::imm32 => {
                    ins.set_op_kind(i, OpKind::Immediate32);
                    ins.set_immediate32(imm32 as u32);
                }
                K::imm32sex64 => {
                    ins.set_op_kind(i, OpKind::Immediate32to64);
                    ins.set_immediate32to64(imm32 as u32 as i32 as i64);
                }
                _ => {
                    ins.set_op_kind(i, OpKind::Immediate64);
                    ins.set_immediate64(imm64);
                }
            }
        } else {
            return None;
        }
    }
    Some(ins)
}

fn aligned_only(code: Code, mnemonic: &str) -> bool {
    let size = code.op_code().memory_size().size();
    let vex = matches!(code.encoding(), iced_x86::EncodingKind::VEX);
    mnemonic.contains("movap")
        || mnemonic.contains("movdqa")
        || mnemonic.contains("movnt")
        || (!vex && size >= 16 && !matches!(mnemonic, "movups" | "movdqu" | "lddqu" | "movupd"))
}

/// All the forms to test for one `Code`.
fn forms_of(code: Code) -> Vec<(Variant, usize, Instruction)> {
    let oc = code.op_code();
    if !oc.mode64() || !matches!(code.encoding(), iced_x86::EncodingKind::Legacy | iced_x86::EncodingKind::VEX) {
        return vec![];
    }
    if oc.is_privileged() || code.flow_control() != FlowControl::Next || oc.is_stack_instruction() {
        return vec![];
    }
    if code.cpuid_features().iter().any(|f| format!("{f:?}").starts_with("FPU") || format!("{f:?}").contains("MMX") || format!("{f:?}") == "D3NOW") {
        return vec![];
    }
    let mnemonic = format!("{:?}", code.mnemonic()).to_lowercase();
    if skip_mnemonic(&mnemonic) {
        return vec![];
    }
    let n = oc.op_count();
    let kinds: Vec<_> = (0..n).map(|i| oc.op_kind(i)).collect();
    if !kinds.iter().all(|&k| supported_kind(k)) {
        return vec![];
    }
    let has_mem = kinds.iter().any(|&k| has_mem_alt(k));
    let has_imm = kinds.iter().any(|&k| is_imm(k) && k != OpCodeOperandKind::imm8_const_1);
    let seed = (code as usize).wrapping_mul(2654435761) >> 4;
    let mut plan: Vec<(Variant, usize)> = vec![(Variant::Reg, 0)];
    // The legacy compares take a predicate 0..7 only; other values are not defined.
    let legacy_cmp = matches!(code.encoding(), iced_x86::EncodingKind::Legacy) && mnemonic.starts_with("cmp") && n == 3;
    if has_imm && !legacy_cmp {
        plan.extend([(Variant::Reg, 1), (Variant::Reg, 2), (Variant::Reg, 3)]);
    }
    if kinds.iter().filter(|&&k| reg_for(k, 0).is_some()).count() >= 2 {
        plan.push((Variant::Alias, 0));
    }
    if has_mem && !mnemonic.starts_with("bt") {
        plan.push((Variant::Mem(A), 0));
        if has_imm && !legacy_cmp {
            plan.push((Variant::Mem(A), 2));
        }
        if !aligned_only(code, &mnemonic) {
            plan.push((Variant::Mem(U), 1));
        }
    }
    plan.into_iter()
        .filter_map(|(v, imm)| build_instruction(code, v, imm, seed).map(|i| (v, imm, i)))
        .collect()
}

fn encode(ins: &Instruction) -> Option<Vec<u8>> {
    let mut e = Encoder::new(64);
    e.encode(ins, 0x1000).ok()?;
    Some(e.take_buffer())
}

fn kind_of(mnemonic: &str) -> Kind {
    let ends = |s: &[&str]| s.iter().any(|x| mnemonic.ends_with(x));
    if ends(&["ss", "ps"]) && !mnemonic.contains("vpcmpestr") {
        Kind::WideFl
    } else if ends(&["sd", "pd"]) {
        Kind::WideDb
    } else {
        Kind::Wide
    }
}

/// The snippet: load every register from the inputs, run `bytes`, dump everything.
fn snippet(bytes: Vec<u8>, flag_mask: u32) -> impl Fn(&mut CodeAssembler, &mut Env) -> Result<(), IcedError> {
    move |a, _env| {
        let g = [rax, rcx, rdx, rbx, rsp, rbp, rsi, rdi, r8, r9, r10, r11, r12, r13, r14, r15];
        let xm = [xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8, xmm9, xmm10, xmm11, xmm12, xmm13, xmm14, xmm15];
        let ym = [ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7, ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15];
        for r in [rbx, rbp, rdi, rsi, r12, r13, r14, r15] {
            a.push(r)?;
        }
        a.sub(rsp, 0xc8)?;
        for i in 6..16 {
            a.vmovups(xmmword_ptr(rsp + 0x20 + 16 * (i - 6)), xm[i])?;
        }
        a.mov(r15, r8)?;
        // Inputs: slot j mixes a (rcx) and b (rdx); the first four are the values themselves, so edge values survive.
        let mix = |a: &mut CodeAssembler, j: i32| -> Result<(), IcedError> {
            let rot = if j < 4 { 0 } else { (j * 7) % 63 + 1 };
            match j % 4 {
                0 => {
                    a.mov(rax, rcx)?;
                    if rot != 0 {
                        a.rol(rax, rot)?;
                    }
                }
                1 => {
                    a.mov(rax, rdx)?;
                    if rot != 0 {
                        a.ror(rax, rot)?;
                    }
                }
                2 => {
                    a.mov(rax, rcx)?;
                    a.xor(rax, rdx)?;
                    if rot != 0 {
                        a.rol(rax, rot)?;
                    }
                }
                _ => {
                    a.mov(rax, rcx)?;
                    a.add(rax, rdx)?;
                    if rot != 0 {
                        a.ror(rax, rot)?;
                    }
                }
            }
            Ok(())
        };
        for j in 0..16 {
            mix(a, j)?;
            a.mov(qword_ptr(r15 + 0x400 + 8 * j), rax)?;
        }
        for j in 0..64 {
            mix(a, j + 16)?;
            a.mov(qword_ptr(r15 + 0x500 + 8 * j), rax)?;
        }
        for j in 0..16 {
            mix(a, j + 90)?;
            a.mov(qword_ptr(r15 + 0x340 + 8 * j), rax)?;
        }
        a.mov(rax, rcx)?;
        a.xor(rax, rdx)?;
        a.and(eax, 0x8d5)?;
        a.or(eax, 2)?;
        a.mov(qword_ptr(r15 + 0x3c8), rax)?;
        a.mov(rax, qword_ptr(r15 + 0x3c8))?;
        a.push(rax)?;
        a.popfq()?;
        for i in 0..16 {
            a.vmovups(ym[i], ymmword_ptr(r15 + 0x500 + 32 * i as i32))?;
        }
        a.lea(r14, qword_ptr(r15 + 0x340))?;
        for i in [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13] {
            a.mov(g[i], qword_ptr(r15 + 0x400 + 8 * i as i32))?;
        }
        a.db(&bytes)?;
        a.pushfq()?;
        a.pop(qword_ptr(r15 + 0x80))?;
        a.and(qword_ptr(r15 + 0x80), flag_mask as i32)?;
        for i in [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13] {
            a.mov(qword_ptr(r15 + 8 * i as i32), g[i])?;
        }
        for i in 0..16 {
            a.vmovups(ymmword_ptr(r15 + 0x100 + 32 * i as i32), ym[i])?;
        }
        for i in 6..16 {
            a.vmovups(xm[i], xmmword_ptr(rsp + 0x20 + 16 * (i as i32 - 6)))?;
        }
        a.add(rsp, 0xc8)?;
        for r in [r15, r14, r13, r12, rsi, rdi, rbp, rbx] {
            a.pop(r)?;
        }
        a.ret()
    }
}

struct Mismatch {
    form: String,
    bytes: String,
    asm: String,
    inputs: (u64, u64),
    diffs: Vec<String>,
    count: usize,
}

fn slot_name(k: usize) -> String {
    const G: [&str; 16] = ["rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15"];
    match k {
        0..=15 => G[k].to_string(),
        16 => "flags".to_string(),
        32..=95 => format!("ymm{}.q{}", (k - 32) / 4, (k - 32) % 4),
        104..=119 => format!("mem+{:#x}", (k - 104) * 8),
        _ => format!("out[{k}]"),
    }
}

fn parse(l: &str) -> Option<(String, u64, u64, Vec<u64>)> {
    let t: Vec<&str> = l.split_whitespace().collect();
    if t.len() < 3 + 128 {
        return None;
    }
    let h = |s: &str| u64::from_str_radix(s, 16).unwrap_or(0);
    Some((t[2].to_string(), h(t[0]), h(t[1]), t[3..131].iter().map(|x| h(x)).collect()))
}

/// Runs one batch; `Err` has the names of snippets that crashed one of the two builds.
fn run_batch(label: &str, forms: &[Form]) -> (Vec<Mismatch>, Vec<String>) {
    let cases: Vec<Case> = forms
        .iter()
        .map(|f| case("form", f.name.clone(), f.kind, snippet(f.bytes.clone(), f.flag_mask)))
        .collect();
    let (dir, _) = prepare(label, &cases, 0);
    write_runtime(&dir);
    let cc = |args: &[&str], out: &str, srcs: &[&str]| {
        let mut c = Command::new("clang");
        c.args(args).arg("-I").arg(&dir);
        for s in srcs {
            c.arg(dir.join(s));
        }
        run(c.arg("-o").arg(dir.join(out)));
    };
    cc(&["-arch", "x86_64", "-O1"], "drv_x86", &["driver.c"]);
    cc(
        &["-arch", "arm64", "-O2", "-ffp-contract=off", "-Wno-unused", "-DLIFTED"],
        "drv_arm",
        &["driver.c", "unit.c", "rt_host.c"],
    );
    let exec = |bin: &str, sel: Option<usize>| -> Option<String> {
        let mut c = Command::new(dir.join(bin));
        c.arg("image.bin").current_dir(&dir);
        if let Some(s) = sel {
            c.arg(s.to_string());
        }
        let o = c.output().ok()?;
        o.status.success().then(|| String::from_utf8_lossy(&o.stdout).into_owned())
    };
    let mut crashed = vec![];
    let (x86, arm) = match (exec("drv_x86", None), exec("drv_arm", None)) {
        (Some(x), Some(a)) => (x, a),
        _ => {
            // One snippet crashes a build: find which, run the others one by one.
            let (mut x, mut a) = (String::new(), String::new());
            for (n, f) in forms.iter().enumerate() {
                match (exec("drv_x86", Some(n)), exec("drv_arm", Some(n))) {
                    (Some(lx), Some(la)) => {
                        x.push_str(&lx);
                        a.push_str(&la);
                    }
                    (lx, la) => crashed.push(format!("{} (x86 {}, arm64 {})", f.name, if lx.is_some() { "ok" } else { "crash" }, if la.is_some() { "ok" } else { "crash" })),
                }
            }
            (x, a)
        }
    };
    let by_name: BTreeMap<&str, &Form> = forms.iter().map(|f| (f.name.as_str(), f)).collect();
    let mut bad: BTreeMap<String, Mismatch> = BTreeMap::new();
    let mut fmt = IntelFormatter::new();
    for (l, r) in x86.lines().zip(arm.lines()) {
        if l == r {
            continue;
        }
        let (Some((name, a, b, mut xw)), Some((_, _, _, mut aw))) = (parse(l), parse(r)) else { continue };
        // Signalling NaNs are ordered differently by x86 and ARM (documented, not fixed): where both sides produced a NaN
        // of the same sign in a float lane, the payload is not compared.
        let kind = by_name[name.as_str()].kind;
        if matches!(kind, Kind::WideFl | Kind::WideDb) {
            for k in (32..96).chain(104..120) {
                let (w, nan): (u32, &dyn Fn(u64) -> bool) = if kind == Kind::WideFl { (32, &|h| (h >> 23) & 0xff == 0xff && h & 0x7f_ffff != 0) } else { (64, &|h| (h >> 52) & 0x7ff == 0x7ff && h & 0xf_ffff_ffff_ffff != 0) };
                for half in 0..(64 / w) {
                    let sh = half * w;
                    let mask = if w == 32 { 0xffff_ffffu64 } else { u64::MAX };
                    let (hx, ha) = ((xw[k] >> sh) & mask, (aw[k] >> sh) & mask);
                    let sign = 1u64 << (w - 1);
                    if nan(hx) && nan(ha) && (hx & sign) == (ha & sign) {
                        xw[k] = (xw[k] & !(mask << sh)) | ((hx & sign) << 0 | 0x7fc0_0000u64 >> (32 - w).min(31) & mask) << sh;
                        aw[k] = (aw[k] & !(mask << sh)) | ((ha & sign) << 0 | 0x7fc0_0000u64 >> (32 - w).min(31) & mask) << sh;
                    }
                }
            }
            if xw == aw {
                continue;
            }
        }
        let e = bad.entry(name.clone()).or_insert_with(|| {
            let f = by_name[name.as_str()];
            let mut dec = Decoder::with_ip(64, &f.bytes, 0, DecoderOptions::NONE);
            let mut text = String::new();
            fmt.format(&dec.decode(), &mut text);
            let diffs = (0..128)
                .filter(|&k| xw[k] != aw[k])
                .map(|k| format!("{}: x86 {:#x} arm64 {:#x}", slot_name(k), xw[k], aw[k]))
                .collect();
            Mismatch {
                form: name,
                bytes: f.bytes.iter().map(|x| format!("{x:02x}")).collect::<Vec<_>>().join(" "),
                asm: text,
                inputs: (a, b),
                diffs,
                count: 0,
            }
        });
        e.count += 1;
    }
    (bad.into_values().collect(), crashed)
}

#[test]
fn every_supported_instruction_form_matches_x86() {
    if !have_tools() {
        eprintln!("skipped: needs Rosetta 2 and clang");
        return;
    }
    let pick: Vec<String> = std::env::var("VICTORY_FORMS")
        .map(|s| s.split(',').map(|x| x.trim().to_lowercase()).collect())
        .unwrap_or_default();
    let list: Option<BTreeSet<String>> = std::env::var("VICTORY_FORMS_LIST")
        .ok()
        .map(|p| std::fs::read_to_string(p).unwrap().split_whitespace().map(str::to_owned).collect());
    let batch: usize = std::env::var("VICTORY_FORMS_BATCH").ok().and_then(|s| s.parse().ok()).unwrap_or(400);

    // Enumerate, encode, and keep what the lifter takes.
    let mut forms: Vec<Form> = vec![];
    let mut unsupported: BTreeMap<String, usize> = BTreeMap::new();
    let mut tried_codes = 0usize;
    let mut info = InstructionInfoFactory::new();
    let ctx_funcs = |bytes: &[u8]| -> Result<(), String> {
        // Lift the snippet alone: `lift_function` says whether the form is supported.
        let sn = snippet(bytes.to_vec(), 0);
        let (blob, spans) = assemble(&[case("form", "x", Kind::Wide, sn)]);
        let (b, e) = spans[0];
        let ins: Vec<_> = Decoder::with_ip(64, &blob[b as usize..e as usize], BASE + b as u64, DecoderOptions::NONE).iter().collect();
        let f = victory_recomp::lift::FuncIn {
            begin_va: BASE + b as u64,
            end_va: BASE + e as u64,
            ins: &ins,
            tables: Default::default(),
            extra: vec![],
            landing_pads: vec![],
        };
        let starts: std::collections::HashSet<u64> = [BASE + b as u64].into();
        let is_func = |va: u64| starts.contains(&va);
        victory_recomp::lift::lift_function(&victory_recomp::lift::Ctx { image_base: BASE, is_func: &is_func }, &f).map(|_| ())
    };
    for code in Code::values() {
        let mnemonic = format!("{:?}", code.mnemonic()).to_lowercase();
        if !pick.is_empty() && !pick.iter().any(|p| mnemonic.contains(p.as_str())) {
            continue;
        }
        if let Some(l) = &list {
            if !l.contains(&format!("{code:?}")) {
                continue;
            }
        }
        for (v, imm, ins) in forms_of(code) {
            tried_codes += 1;
            let Some(bytes) = encode(&ins) else { continue };
            let undefined = ins.rflags_undefined();
            let mut mask = 0x8d5u32 & !0x10; // OF SF ZF CF PF, never AF
            for (bit, flag) in [(1u32, 0x800u32), (2, 0x80), (4, 0x40), (0x10, 0x1), (0x20, 0x4)] {
                if undefined & bit != 0 {
                    mask &= !flag;
                }
            }
            // Rosetta clears OF/SF/PF differently after ptest and flags cmpxchg as dest-minus-accumulator: compare what
            // the manuals say is defined.
            if mnemonic.contains("ptest") || mnemonic.starts_with("vtestp") {
                mask &= 0x41;
            }
            if mnemonic.starts_with("cmpxchg") {
                mask = 0;
            }
            let name = format!("{code:?}_{}", forms.len());
            if let Err(e) = ctx_funcs(&bytes) {
                *unsupported.entry(format!("{code:?}: {e}")).or_default() += 1;
                continue;
            }
            forms.push(Form { name, code, bytes, flag_mask: mask, kind: kind_of(&mnemonic) });
        }
    }
    eprintln!(
        "{} variants of {} forms to run, {} variants the lifter does not support ({} forms)",
        forms.len(),
        forms.iter().map(|f| f.code).collect::<BTreeSet<_>>().len(),
        unsupported.values().sum::<usize>(),
        unsupported.len()
    );
    let mut report = String::new();
    if let Some(l) = &list {
        // What the title uses and this test did not run (an operand kind it does not build, or a form the lifter lacks).
        let ran: BTreeSet<String> = forms.iter().map(|f| format!("{:?}", f.code)).collect();
        let missing: Vec<&String> = l.iter().filter(|n| !ran.contains(*n)).collect();
        eprintln!("{} of {} listed forms were not run", missing.len(), l.len());
        report.push_str(&format!("listed forms not run ({}): {}\n", missing.len(), missing.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(" ")));
    }
    let mut mismatches = vec![];
    let mut crashes = vec![];
    for (n, chunk) in forms.chunks(batch).enumerate() {
        let (m, c) = run_batch(&format!("diff_forms_{n}"), chunk);
        eprintln!("batch {n}: {} snippets, {} mismatching, {} crashing", chunk.len(), m.len(), c.len());
        mismatches.extend(m);
        crashes.extend(c);
    }
    let waived: Vec<String> = mismatches
        .iter()
        .filter_map(|m| WAIVERS.iter().find(|(p, _)| m.form.starts_with(p)).map(|(_, why)| format!("{} ({why})", m.form)))
        .collect();
    mismatches.retain(|m| !WAIVERS.iter().any(|(p, _)| m.form.starts_with(p)));
    // Rosetta 2 raises SIGILL on these (the KAT tests run them on arm64 only).
    crashes.retain(|c| !(c.contains("x86 crash, arm64 ok") && ROSETTA_LACKS.iter().any(|p| c.starts_with(p))));
    use std::fmt::Write as _;
    let _ = writeln!(report, "forms run: {} variants; unsupported by the lifter: {} forms", forms.len(), unsupported.len());
    for (k, v) in &unsupported {
        let _ = writeln!(report, "  unsupported x{v}: {k}");
    }
    for w in &waived {
        let _ = writeln!(report, "waived {w}");
    }
    for c in &crashes {
        let _ = writeln!(report, "CRASH {c}");
    }
    for m in &mismatches {
        let _ = writeln!(
            report,
            "MISMATCH {} ({} of 300 inputs)\n  asm: {}\n  bytes: {}\n  inputs a={:#x} b={:#x}\n  {}",
            m.form,
            m.count,
            m.asm,
            m.bytes,
            m.inputs.0,
            m.inputs.1,
            m.diffs.iter().take(8).cloned().collect::<Vec<_>>().join("\n  ")
        );
    }
    let out = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("diff_forms");
    std::fs::create_dir_all(&out).unwrap();
    std::fs::write(out.join("report.txt"), &report).unwrap();
    eprintln!("{report}");
    assert!(crashes.is_empty(), "snippets crashed: {crashes:?}");
    assert!(mismatches.is_empty(), "{} instruction forms differ from x86 (see the report)", mismatches.len());
}
