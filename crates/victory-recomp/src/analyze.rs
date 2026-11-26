//! Whole-image analysis: disassemble every `.pdata` function and gather the
//! statistics that drive the recompiler design (extensions used, indirect
//! control flow, jump tables, JIT / self-modifying-code signs).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Write as _;

use anyhow::Result;
use iced_x86::{
    Code, CpuidFeature, Decoder, DecoderOptions, FlowControl, Instruction, Mnemonic, OpKind,
    Register,
};

use crate::image::{Func, Image, SCN_EXEC, SCN_WRITE};

/// A jump table recovered from a `jmp reg` dispatch sequence.
#[derive(Debug, Clone)]
pub struct JumpTable {
    /// RVA of the `jmp reg` instruction.
    pub jmp_rva: u32,
    pub table_rva: u32,
    /// Target RVAs, all inside the owning function.
    pub targets: Vec<u32>,
}

/// Decodes one function. Returns instructions and whether decoding hit garbage.
pub fn decode_function(img: &Image, f: &Func) -> (Vec<Instruction>, bool) {
    let Some(code) = img.bytes(f.begin, (f.end - f.begin) as usize) else {
        return (vec![], true);
    };
    let ip = img.info.image_base + f.begin as u64;
    let mut dec = Decoder::with_ip(64, code, ip, DecoderOptions::NONE);
    let mut out = Vec::with_capacity(code.len() / 4);
    let mut bad = false;
    while dec.can_decode() {
        let i = dec.decode();
        if i.is_invalid() {
            bad = true;
            break;
        }
        out.push(i);
    }
    (out, bad)
}

fn is_cmov_free_load(code: Code) -> bool {
    matches!(
        code,
        Code::Movsxd_r64_rm32
            | Code::Mov_r32_rm32
            | Code::Mov_r64_rm64
            | Code::Movzx_r32_rm16
            | Code::Movzx_r64_rm16
    )
}

/// Recognises both table styles: clang/gcc (`lea base,[rip+table]` then
/// `movsxd r,[base+idx*4]`) and MSVC (`lea base,[__ImageBase]` then
/// `mov r32,[base+idx*4+table_rva]`), each followed by `add r,base; jmp r`.
pub fn find_jump_tables(img: &Image, f: &Func, ins: &[Instruction]) -> Vec<JumpTable> {
    find_jump_tables_with(img.info.image_base, &|rva| img.read_u32(rva), f, ins)
}

/// Same, over any RVA-addressed `u32` reader (used by tests on raw blobs).
pub fn find_jump_tables_with(
    base: u64,
    read_u32: &dyn Fn(u32) -> Option<u32>,
    f: &Func,
    ins: &[Instruction],
) -> Vec<JumpTable> {
    let mut out = vec![];
    for (n, j) in ins.iter().enumerate() {
        if j.code() != Code::Jmp_rm64 || j.op0_kind() != OpKind::Register {
            continue;
        }
        let win = &ins[n.saturating_sub(40)..n];
        // The table base register: a rip-relative lea in the window.
        let Some(lea) = win
            .iter()
            .rev()
            .find(|i| i.code() == Code::Lea_r64_m && i.is_ip_rel_memory_operand())
        else {
            continue;
        };
        let lea_rva = (lea.ip_rel_memory_address().wrapping_sub(base)) as u32;
        let base_reg = lea.op0_register();
        // The scaled load from the table.
        let Some(ld) = win.iter().rev().find(|i| {
            is_cmov_free_load(i.code())
                && i.op1_kind() == OpKind::Memory
                && i.memory_base() == base_reg
                && i.memory_index_scale() == 4
        }) else {
            continue;
        };
        let table_rva = lea_rva.wrapping_add(ld.memory_displacement64() as u32);
        // Bound from `cmp idx, N` / `ja default` when present.
        let bound = win
            .iter()
            .rev()
            .find(|i| {
                matches!(
                    i.code(),
                    Code::Cmp_rm32_imm8
                        | Code::Cmp_rm32_imm32
                        | Code::Cmp_rm64_imm8
                        | Code::Cmp_rm64_imm32
                        | Code::Cmp_rm8_imm8
                )
            })
            .map(|i| i.immediate(1) as usize + 1)
            .filter(|b| (1..=65536).contains(b));
        let mut targets = vec![];
        for k in 0..bound.unwrap_or(4096) {
            let Some(e) = read_u32(table_rva + 4 * k as u32) else {
                break;
            };
            let t = if ld.code() == Code::Mov_r32_rm32 || ld.code() == Code::Mov_r64_rm64 {
                lea_rva.wrapping_add(e)
            } else {
                lea_rva.wrapping_add(e as i32 as u32)
            };
            if t < f.begin || t >= f.end {
                break;
            }
            targets.push(t);
        }
        if targets.len() >= 2 || (bound.is_some() && !targets.is_empty()) {
            out.push(JumpTable {
                jmp_rva: (j.ip() - base) as u32,
                table_rva,
                targets,
            });
        }
    }
    out
}

#[derive(Default, Clone)]
pub struct Stats {
    pub functions: u64,
    pub chained: u64,
    pub decode_failed_funcs: u64,
    pub code_bytes: u64,
    pub instructions: u64,
    pub max_func_instrs: u64,
    pub func_sizes: Vec<u32>,
    pub features: BTreeMap<String, u64>,
    pub mnemonics: HashMap<String, u64>,
    pub call_direct: u64,
    pub call_indirect_reg: u64,
    pub call_indirect_mem_rip: u64,
    pub call_indirect_mem_other: u64,
    pub jmp_direct: u64,
    pub jmp_cond: u64,
    pub jmp_indirect_reg: u64,
    pub jmp_indirect_mem: u64,
    pub tail_jmp_out_of_func: u64,
    pub rets: u64,
    pub jump_tables: u64,
    pub jump_table_entries: u64,
    pub jmp_reg_without_table: u64,
    pub lock_prefix: u64,
    pub rep_string: u64,
    pub gs_access: u64,
    pub fs_access: u64,
    pub syscall_like: u64,
    pub cpuid_rdtsc: u64,
    pub store_into_exec: u64,
    pub x87: u64,
    pub xchg_mem: u64,
    pub xmm_ins: u64,
    pub ymm_ins: u64,
    pub zmm_ins: u64,
    pub call_targets_non_pdata: u64,
    pub call_targets_into_exec_gaps: u64,
    /// Distinct non-.pdata direct call / out-of-function jmp targets (leaf functions, thunks).
    pub gap_targets: HashSet<u32>,
}

impl Stats {
    fn merge(&mut self, o: Stats) {
        macro_rules! add { ($($f:ident),*) => { $( self.$f += o.$f; )* } }
        add!(
            functions,
            chained,
            decode_failed_funcs,
            code_bytes,
            instructions,
            call_direct,
            call_indirect_reg,
            call_indirect_mem_rip,
            call_indirect_mem_other,
            jmp_direct,
            jmp_cond,
            jmp_indirect_reg,
            jmp_indirect_mem,
            tail_jmp_out_of_func,
            rets,
            jump_tables,
            jump_table_entries,
            jmp_reg_without_table,
            lock_prefix,
            rep_string,
            gs_access,
            fs_access,
            syscall_like,
            cpuid_rdtsc,
            store_into_exec,
            x87,
            xchg_mem,
            xmm_ins,
            ymm_ins,
            zmm_ins,
            call_targets_non_pdata,
            call_targets_into_exec_gaps
        );
        self.gap_targets.extend(o.gap_targets);
        self.max_func_instrs = self.max_func_instrs.max(o.max_func_instrs);
        self.func_sizes.extend(o.func_sizes);
        for (k, v) in o.features {
            *self.features.entry(k).or_default() += v;
        }
        for (k, v) in o.mnemonics {
            *self.mnemonics.entry(k).or_default() += v;
        }
    }
}

/// Image-level facts that are not per-instruction.
#[derive(Default)]
pub struct ImageFacts {
    pub image_bytes_in_exec_sections: u64,
    pub exec_sections: Vec<(String, u32, u32, bool)>,
    pub wx_sections: Vec<String>,
    pub pdata_covered_bytes: u64,
    pub exports: usize,
    pub exports_not_in_pdata: usize,
    pub entry_in_pdata: bool,
    pub tls_callbacks: usize,
    pub tls_callbacks_not_in_pdata: usize,
    pub relocs_dir64: u64,
    pub relocs_into_exec: u64,
    pub relocs_to_func_start: u64,
    pub relocs_into_exec_not_func_start: u64,
    pub suspicious_imports: Vec<String>,
    pub reloc_gap_targets: usize,
    pub gap_targets_union: usize,
}

pub struct Report {
    pub stats: Stats,
    pub facts: ImageFacts,
}

fn analyze_chunk(
    img: &Image,
    funcs: &[Func],
    starts: &HashSet<u32>,
    exec_ranges: &[(u32, u32)],
) -> Stats {
    let base = img.info.image_base;
    let mut st = Stats::default();
    let in_exec = |rva: u32| exec_ranges.iter().any(|&(a, b)| rva >= a && rva < b);
    for f in funcs {
        st.functions += 1;
        st.chained += f.chained as u64;
        st.code_bytes += (f.end - f.begin) as u64;
        st.func_sizes.push(f.end - f.begin);
        let (ins, bad) = decode_function(img, f);
        st.decode_failed_funcs += bad as u64;
        st.instructions += ins.len() as u64;
        st.max_func_instrs = st.max_func_instrs.max(ins.len() as u64);
        let tables = find_jump_tables(img, f, &ins);
        st.jump_tables += tables.len() as u64;
        st.jump_table_entries += tables.iter().map(|t| t.targets.len() as u64).sum::<u64>();
        let table_jmps: HashSet<u32> = tables.iter().map(|t| t.jmp_rva).collect();
        for i in &ins {
            *st.mnemonics
                .entry(format!("{:?}", i.mnemonic()))
                .or_default() += 1;
            for &ft in i.cpuid_features() {
                *st.features.entry(format!("{ft:?}")).or_default() += 1;
            }
            if i.cpuid_features().contains(&CpuidFeature::FPU) {
                st.x87 += 1;
            }
            if i.has_lock_prefix() {
                st.lock_prefix += 1;
            }
            let rep = i.has_rep_prefix() || i.has_repe_prefix() || i.has_repne_prefix();
            if rep
                && matches!(
                    i.mnemonic(),
                    Mnemonic::Movsb
                        | Mnemonic::Movsw
                        | Mnemonic::Movsd
                        | Mnemonic::Movsq
                        | Mnemonic::Stosb
                        | Mnemonic::Stosw
                        | Mnemonic::Stosd
                        | Mnemonic::Stosq
                        | Mnemonic::Cmpsb
                        | Mnemonic::Scasb
                        | Mnemonic::Cmpsq
                        | Mnemonic::Scasq
                        | Mnemonic::Lodsb
                )
            {
                st.rep_string += 1;
            }
            match i.segment_prefix() {
                Register::GS => st.gs_access += 1,
                Register::FS => st.fs_access += 1,
                _ => {}
            }
            match i.mnemonic() {
                Mnemonic::Syscall | Mnemonic::Sysenter | Mnemonic::Int | Mnemonic::Int1 => {
                    st.syscall_like += 1
                }
                Mnemonic::Cpuid | Mnemonic::Rdtsc | Mnemonic::Rdtscp | Mnemonic::Xgetbv => {
                    st.cpuid_rdtsc += 1
                }
                Mnemonic::Xchg
                    if i.op0_kind() == OpKind::Memory || i.op1_kind() == OpKind::Memory =>
                {
                    st.xchg_mem += 1
                }
                _ => {}
            }
            let (mut x, mut y, mut z) = (false, false, false);
            for k in 0..i.op_count() {
                if i.op_kind(k) == OpKind::Register {
                    let r = i.op_register(k);
                    x |= r.is_xmm();
                    y |= r.is_ymm();
                    z |= r.is_zmm();
                }
            }
            st.xmm_ins += (x && !y && !z) as u64;
            st.ymm_ins += y as u64;
            st.zmm_ins += z as u64;
            // Memory stores into executable sections: self-modifying-code sign.
            if i.is_ip_rel_memory_operand()
                && i.op0_kind() == OpKind::Memory
                && !matches!(
                    i.mnemonic(),
                    Mnemonic::Cmp
                        | Mnemonic::Test
                        | Mnemonic::Lea
                        | Mnemonic::Jmp
                        | Mnemonic::Call
                        | Mnemonic::Nop
                        | Mnemonic::Prefetcht0
                        | Mnemonic::Prefetchw
                )
                && in_exec(i.ip_rel_memory_address().wrapping_sub(base) as u32)
            {
                st.store_into_exec += 1;
            }
            match i.flow_control() {
                FlowControl::Call => {
                    st.call_direct += 1;
                    let t = (i.near_branch_target().wrapping_sub(base)) as u32;
                    if !starts.contains(&t) {
                        st.call_targets_non_pdata += 1;
                        if in_exec(t) {
                            st.call_targets_into_exec_gaps += 1;
                            st.gap_targets.insert(t);
                        }
                    }
                }
                FlowControl::IndirectCall => match i.op0_kind() {
                    OpKind::Register => st.call_indirect_reg += 1,
                    _ if i.is_ip_rel_memory_operand() => st.call_indirect_mem_rip += 1,
                    _ => st.call_indirect_mem_other += 1,
                },
                FlowControl::UnconditionalBranch => {
                    st.jmp_direct += 1;
                    let t = (i.near_branch_target().wrapping_sub(base)) as u32;
                    if t < f.begin || t >= f.end {
                        st.tail_jmp_out_of_func += 1;
                        if !starts.contains(&t) && in_exec(t) {
                            st.gap_targets.insert(t);
                        }
                    }
                }
                FlowControl::ConditionalBranch => st.jmp_cond += 1,
                FlowControl::IndirectBranch => {
                    if i.op0_kind() == OpKind::Register {
                        st.jmp_indirect_reg += 1;
                        if !table_jmps.contains(&((i.ip() - base) as u32)) {
                            st.jmp_reg_without_table += 1;
                        }
                    } else {
                        st.jmp_indirect_mem += 1;
                    }
                }
                FlowControl::Return => st.rets += 1,
                _ => {}
            }
        }
    }
    st
}

pub fn analyze(img: &Image) -> Result<Report> {
    let funcs = img.functions()?;
    let starts: HashSet<u32> = funcs.iter().map(|f| f.begin).collect();
    let exec_ranges: Vec<(u32, u32)> = img
        .info
        .sections
        .iter()
        .filter(|s| s.characteristics & SCN_EXEC != 0)
        .map(|s| (s.rva, s.rva + s.virtual_size.max(s.file_size)))
        .collect();
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let chunk = funcs.len().div_ceil(threads).max(1);
    let mut stats = Stats::default();
    std::thread::scope(|sc| {
        let hs: Vec<_> = funcs
            .chunks(chunk)
            .map(|c| sc.spawn(|| analyze_chunk(img, c, &starts, &exec_ranges)))
            .collect();
        for h in hs {
            stats.merge(h.join().unwrap());
        }
    });

    let mut facts = ImageFacts::default();
    for s in &img.info.sections {
        if s.characteristics & SCN_EXEC != 0 {
            facts.image_bytes_in_exec_sections += s.virtual_size as u64;
            facts.exec_sections.push((
                s.name.clone(),
                s.rva,
                s.virtual_size,
                s.characteristics & SCN_WRITE != 0,
            ));
            if s.characteristics & SCN_WRITE != 0 {
                facts.wx_sections.push(s.name.clone());
            }
        }
    }
    facts.pdata_covered_bytes = stats.code_bytes;
    facts.exports = img.info.exports.len();
    facts.exports_not_in_pdata = img
        .info
        .exports
        .iter()
        .filter(|e| {
            e.rva
                .is_some_and(|r| img.is_exec(r) && !starts.contains(&r))
        })
        .count();
    facts.entry_in_pdata = starts.contains(&img.info.entry_rva);
    if let Some(t) = &img.info.tls {
        facts.tls_callbacks = t.callback_rvas.len();
        facts.tls_callbacks_not_in_pdata = t
            .callback_rvas
            .iter()
            .filter(|r| !starts.contains(r))
            .count();
    }
    let relocs = img.dir64_relocs();
    facts.relocs_dir64 = relocs.len() as u64;
    let mut reloc_gap: HashSet<u32> = HashSet::new();
    for r in relocs {
        let Some(b) = img.bytes(r, 8) else { continue };
        let v = u64::from_le_bytes(b.try_into().unwrap());
        let t = v.wrapping_sub(img.info.image_base);
        if t > u32::MAX as u64 {
            continue;
        }
        let t = t as u32;
        if img.is_exec(t) {
            facts.relocs_into_exec += 1;
            if starts.contains(&t) {
                facts.relocs_to_func_start += 1;
            } else {
                facts.relocs_into_exec_not_func_start += 1;
                reloc_gap.insert(t);
            }
        }
    }
    facts.reloc_gap_targets = reloc_gap.len();
    facts.gap_targets_union = reloc_gap.union(&stats.gap_targets).count();
    const SUS: &[&str] = &[
        "VirtualProtect",
        "VirtualAlloc",
        "VirtualAllocEx",
        "FlushInstructionCache",
        "WriteProcessMemory",
        "NtProtectVirtualMemory",
        "VirtualProtectFromApp",
        "VirtualAllocFromApp",
        "MapViewOfFile3",
        "MapViewOfFileFromApp",
        "CreateFileMappingFromApp",
    ];
    for m in &img.info.imports {
        for e in &m.entries {
            let n = e.symbol.to_string();
            if SUS.iter().any(|s| n == *s) {
                facts.suspicious_imports.push(format!("{}!{}", m.dll, n));
            }
        }
    }
    Ok(Report { stats, facts })
}

impl Report {
    pub fn render(&self) -> String {
        let (s, f) = (&self.stats, &self.facts);
        let mut o = String::new();
        let pct = |a: u64, b: u64| {
            if b == 0 {
                0.0
            } else {
                a as f64 * 100.0 / b as f64
            }
        };
        let mut sizes = s.func_sizes.clone();
        sizes.sort_unstable();
        let med = sizes.get(sizes.len() / 2).copied().unwrap_or(0);
        let p99 = sizes.get(sizes.len() * 99 / 100).copied().unwrap_or(0);
        let _ = writeln!(
            o,
            "functions (.pdata entries): {}  (chained/cold parts: {})",
            s.functions, s.chained
        );
        let _ = writeln!(
            o,
            "functions with a decode failure: {}",
            s.decode_failed_funcs
        );
        let _ = writeln!(
            o,
            "code bytes covered by .pdata: {}  (exec section virtual bytes: {}, {:.1}% covered)",
            s.code_bytes,
            f.image_bytes_in_exec_sections,
            pct(s.code_bytes, f.image_bytes_in_exec_sections)
        );
        let _ = writeln!(
            o,
            "instructions: {}  (avg {:.2} bytes)",
            s.instructions,
            s.code_bytes as f64 / s.instructions.max(1) as f64
        );
        let _ = writeln!(
            o,
            "function size bytes: median {med}, p99 {p99}, max {}; largest function {} instructions",
            sizes.last().copied().unwrap_or(0),
            s.max_func_instrs
        );
        let _ = writeln!(o, "\ncontrol flow");
        let _ = writeln!(
            o,
            "  direct calls {}  (to non-.pdata targets: {}, of which inside exec sections: {})",
            s.call_direct, s.call_targets_non_pdata, s.call_targets_into_exec_gaps
        );
        let _ = writeln!(
            o,
            "  indirect calls: reg {}  mem[rip] {}  mem[other] {}",
            s.call_indirect_reg, s.call_indirect_mem_rip, s.call_indirect_mem_other
        );
        let _ = writeln!(
            o,
            "  direct jmp {}  (leaving the function: {})  conditional {}",
            s.jmp_direct, s.tail_jmp_out_of_func, s.jmp_cond
        );
        let _ = writeln!(
            o,
            "  indirect jmp: reg {}  mem {}  ret {}",
            s.jmp_indirect_reg, s.jmp_indirect_mem, s.rets
        );
        let _ = writeln!(
            o,
            "  recovered jump tables {} ({} entries); jmp reg with no recovered table: {}",
            s.jump_tables, s.jump_table_entries, s.jmp_reg_without_table
        );
        let _ = writeln!(
            o,
            "\nextensions (instruction counts by CPUID feature; an instruction can list several)"
        );
        let mut feats: Vec<_> = s.features.iter().collect();
        feats.sort_by(|a, b| b.1.cmp(a.1));
        for (k, v) in feats {
            let _ = writeln!(o, "  {k:<16} {v}");
        }
        let _ = writeln!(o, "  x87 (FPU) instructions: {}", s.x87);
        let _ = writeln!(
            o,
            "  touching xmm only: {}  ymm: {}  zmm: {}",
            s.xmm_ins, s.ymm_ins, s.zmm_ins
        );
        let _ = writeln!(o, "\nother");
        let _ = writeln!(
            o,
            "  lock prefix {}  xchg-mem {}  rep string ops {}",
            s.lock_prefix, s.xchg_mem, s.rep_string
        );
        let _ = writeln!(
            o,
            "  gs-prefixed {}  fs-prefixed {}  syscall/int {}  cpuid/rdtsc/xgetbv {}",
            s.gs_access, s.fs_access, s.syscall_like, s.cpuid_rdtsc
        );
        let _ = writeln!(o, "\nimage / discovery");
        for (n, rva, sz, w) in &f.exec_sections {
            let _ = writeln!(
                o,
                "  exec section {n} rva {rva:#x} size {sz:#x}{}",
                if *w { " WRITABLE" } else { "" }
            );
        }
        let _ = writeln!(
            o,
            "  exports {} (not in .pdata: {})  entry in .pdata: {}",
            f.exports, f.exports_not_in_pdata, f.entry_in_pdata
        );
        let _ = writeln!(
            o,
            "  tls callbacks {} (not in .pdata: {})",
            f.tls_callbacks, f.tls_callbacks_not_in_pdata
        );
        let _ = writeln!(
            o,
            "  DIR64 relocations {}  into exec {} (at .pdata function starts {}, elsewhere {})",
            f.relocs_dir64,
            f.relocs_into_exec,
            f.relocs_to_func_start,
            f.relocs_into_exec_not_func_start
        );
        let _ = writeln!(
            o,
            "  distinct non-.pdata targets: from calls/tail jmps {}, from relocs {}, union {} (leaf functions and thunks the .pdata seed misses)",
            s.gap_targets.len(),
            f.reloc_gap_targets,
            f.gap_targets_union
        );
        let _ = writeln!(o, "\nself-modifying / JIT signs");
        let _ = writeln!(o, "  W+X sections: {:?}", f.wx_sections);
        let _ = writeln!(
            o,
            "  rip-relative stores into exec sections: {}",
            s.store_into_exec
        );
        let _ = writeln!(o, "  suspicious imports: {:?}", f.suspicious_imports);
        let _ = writeln!(o, "\ntop mnemonics");
        let mut m: Vec<_> = s.mnemonics.iter().collect();
        m.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
        for (k, v) in m.iter().take(40) {
            let _ = writeln!(o, "  {k:<12} {v}  ({:.2}%)", pct(**v, s.instructions));
        }
        o
    }
}
