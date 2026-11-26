//! Static x86-64 -> ARM64 recompiler foundation. See `docs/recompiler.md`.

pub mod analyze;
pub mod discover;
pub mod image;
pub mod lift;

/// Files a host needs next to a lifted unit: the runtime header and prototype host.
pub const RT_HOST_C: &str = include_str!("rt_host.c");
/// For an image built as a dylib of its own (several images in one process): its `IMG` and unit record.
pub const RT_UNIT_C: &str = include_str!("rt_unit.c");
pub const HARNESS_H: &str = include_str!("harness.h");

use std::collections::{BTreeMap, HashSet};

use anyhow::Result;
use image::{Func, Image};

type Tables = BTreeMap<u64, Vec<u64>>;

fn tables_for(img: &Image, f: &Func, ins: &[iced_x86::Instruction]) -> Tables {
    let base = img.info.image_base;
    analyze::find_jump_tables(img, f, ins)
        .iter()
        .map(|t| {
            (
                base + t.jmp_rva as u64,
                t.targets.iter().map(|&r| base + r as u64).collect(),
            )
        })
        .collect()
}

/// Entry-point functions with their chained `.pdata` parts attached (cold blocks and funclets
/// continue a parent through the unwind info's chain pointer). A part whose parent is unknown is dropped.
pub fn group_chained(img: &Image, all: Vec<Func>) -> Vec<(Func, Vec<Func>)> {
    let mut parts: std::collections::HashMap<u32, Vec<Func>> = Default::default();
    let by_begin: std::collections::HashMap<u32, Func> = all.iter().map(|f| (f.begin, *f)).collect();
    for f in all.iter().filter(|f| f.chained) {
        let mut root = *f;
        for _ in 0..8 {
            match img.chain_parent(&root).and_then(|p| by_begin.get(&p)) {
                Some(p) => root = *p,
                None => break,
            }
            if !root.chained {
                break;
            }
        }
        if !root.chained {
            parts.entry(root.begin).or_default().push(*f);
        }
    }
    all.into_iter()
        .filter(|f| !f.chained)
        .map(|f| {
            let mut p = parts.remove(&f.begin).unwrap_or_default();
            p.sort_by_key(|x| x.begin);
            (f, p)
        })
        .collect()
}

/// Decodes a function and its merged parts into one instruction list (parent first).
fn decode_group(img: &Image, f: &Func, parts: &[Func]) -> (Vec<iced_x86::Instruction>, bool) {
    let (mut ins, mut bad) = analyze::decode_function(img, f);
    for p in parts {
        let (i2, b2) = analyze::decode_function(img, p);
        ins.extend(i2);
        bad |= b2;
    }
    (ins, bad)
}

fn extra_ranges(img: &Image, parts: &[Func]) -> Vec<(u64, u64)> {
    let base = img.info.image_base;
    parts.iter().map(|p| (base + p.begin as u64, base + p.end as u64)).collect()
}

fn landing_pads(img: &Image, f: &Func) -> Vec<u64> {
    img.landing_pads(f).iter().map(|&r| img.info.image_base + r as u64).collect()
}

/// Lifts every non-chained `.pdata` function of `img` into one C unit.
pub fn lift_image(img: &Image) -> Result<lift::Unit> {
    let funcs = group_chained(img, discover::discover(img)?);
    let base = img.info.image_base;
    let starts: HashSet<u64> = funcs.iter().map(|(f, _)| base + f.begin as u64).collect();
    let decoded: Vec<_> = funcs
        .iter()
        .map(|(f, parts)| {
            let (ins, _) = decode_group(img, f, parts);
            let tables = tables_for(img, f, &ins);
            (f, parts, ins, tables)
        })
        .collect();
    let fins: Vec<lift::FuncIn> = decoded
        .iter()
        .map(|(f, parts, ins, tables)| lift::FuncIn {
            begin_va: base + f.begin as u64,
            end_va: base + f.end as u64,
            ins,
            tables: tables.clone(),
            extra: extra_ranges(img, parts),
            landing_pads: landing_pads(img, f),
        })
        .collect();
    let is_func = |va: u64| starts.contains(&va);
    Ok(lift::lift_unit(
        &lift::Ctx {
            image_base: base,
            is_func: &is_func,
        },
        &fins,
    ))
}

/// Largest amount of x86 code (bytes) one split unit may hold.
pub const UNIT_BYTES: usize = 600_000;

/// Writes the whole image as `tu_*.c` files of `per_unit` functions, plus `table.c`, `rt.h` and
/// `rt_host.c`, into `dir`. Returns (functions, lifted, stubbed, units).
pub fn lift_image_split(
    img: &Image,
    dir: &std::path::Path,
    per_unit: usize,
) -> Result<(usize, usize, usize, usize)> {
    use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
    let funcs = group_chained(img, discover::discover(img)?);
    let base = img.info.image_base;
    let starts: HashSet<u64> = funcs.iter().map(|(f, _)| base + f.begin as u64).collect();
    let is_func = |va: u64| starts.contains(&va);
    let ctx = lift::Ctx {
        image_base: base,
        is_func: &is_func,
    };
    std::fs::create_dir_all(dir)?;
    std::fs::write(dir.join("rt.h"), lift::RT_HEADER)?;
    std::fs::write(dir.join("rt_host.c"), RT_HOST_C)?;
    std::fs::write(dir.join("rt_unit.c"), RT_UNIT_C)?;
    let mut all: Vec<u64> = starts.iter().copied().collect();
    all.sort_unstable();
    let pads: Vec<(u64, u64)> = funcs
        .iter()
        .flat_map(|(f, _)| landing_pads(img, f).into_iter().map(move |p| (base + f.begin as u64, p)))
        .collect();
    std::fs::write(dir.join("table.c"), lift::table_unit(base, &all, &pads))?;
    // Units are `per_unit` functions, further cut so no unit holds more than UNIT_BYTES of x86 code
    // (a few giant functions would otherwise make one unit take far longer than the rest).
    let mut chunks: Vec<&[(Func, Vec<Func>)]> = vec![];
    for part in funcs.chunks(per_unit) {
        let (mut from, mut bytes) = (0, 0usize);
        for (i, (f, parts)) in part.iter().enumerate() {
            let sz = (f.end - f.begin) as usize + parts.iter().map(|p| (p.end - p.begin) as usize).sum::<usize>();
            if i > from && bytes + sz > UNIT_BYTES {
                chunks.push(&part[from..i]);
                (from, bytes) = (i, 0);
            }
            bytes += sz;
        }
        chunks.push(&part[from..]);
    }
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let (next, lifted, stubbed) = (AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0));
    std::thread::scope(|sc| -> Result<()> {
        let hs: Vec<_> = (0..threads)
            .map(|_| {
                sc.spawn(|| -> Result<()> {
                    loop {
                        let n = next.fetch_add(1, Relaxed);
                        let Some(part) = chunks.get(n) else {
                            return Ok(());
                        };
                        let decoded: Vec<_> = part
                            .iter()
                            .map(|(f, parts)| {
                                let (ins, _) = decode_group(img, f, parts);
                                let tables = tables_for(img, f, &ins);
                                (f, parts, ins, tables)
                            })
                            .collect();
                        let fins: Vec<lift::FuncIn> = decoded
                            .iter()
                            .map(|(f, parts, ins, tables)| lift::FuncIn {
                                begin_va: base + f.begin as u64,
                                end_va: base + f.end as u64,
                                ins,
                                tables: tables.clone(),
                                extra: extra_ranges(img, parts),
                                landing_pads: landing_pads(img, f),
                            })
                            .collect();
                        let u = lift::lift_part(&ctx, &fins);
                        lifted.fetch_add(u.lifted.len(), Relaxed);
                        stubbed.fetch_add(u.failed.len(), Relaxed);
                        std::fs::write(dir.join(format!("tu_{n:05}.c")), u.c)?;
                    }
                })
            })
            .collect();
        for h in hs {
            h.join().map_err(|_| anyhow::anyhow!("lift thread panicked"))??;
        }
        Ok(())
    })?;
    Ok((funcs.len(), lifted.into_inner(), stubbed.into_inner(), chunks.len()))
}

/// How much of an image the current lifter handles, by first blocking reason.
#[derive(Default)]
pub struct Coverage {
    pub total: u64,
    pub lifted: u64,
    pub instructions_in_lifted: u64,
    /// `jmp reg` sites with no recovered table (lifted as dispatcher tail calls) and the functions holding them.
    pub untabled_jmps: u64,
    pub funcs_with_untabled_jmp: u64,
    /// Chained `.pdata` parts merged into their parents.
    pub merged_parts: u64,
    pub blockers: BTreeMap<String, u64>,
    /// Every unsupported instruction reason: (instructions, functions containing it).
    pub unsupported: BTreeMap<String, (u64, u64)>,
}

pub fn coverage(img: &Image) -> Result<Coverage> {
    let funcs = group_chained(img, discover::discover(img)?);
    let base = img.info.image_base;
    let starts: HashSet<u64> = funcs.iter().map(|(f, _)| base + f.begin as u64).collect();
    let is_func = |va: u64| starts.contains(&va);
    let ctx = lift::Ctx {
        image_base: base,
        is_func: &is_func,
    };
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let chunk = funcs.len().div_ceil(threads).max(1);
    let mut cov = Coverage::default();
    std::thread::scope(|sc| {
        let hs: Vec<_> = funcs
            .chunks(chunk)
            .map(|part| {
                let ctx = &ctx;
                sc.spawn(move || {
                    let mut c = Coverage::default();
                    for (f, parts) in part {
                        let (ins, bad) = decode_group(img, f, parts);
                        c.total += 1;
                        let tables = tables_for(img, f, &ins);
                        let untabled = ins
                            .iter()
                            .filter(|i| {
                                i.code() == iced_x86::Code::Jmp_rm64
                                    && i.op0_kind() == iced_x86::OpKind::Register
                                    && !tables.contains_key(&i.ip())
                            })
                            .count() as u64;
                        c.untabled_jmps += untabled;
                        c.funcs_with_untabled_jmp += (untabled > 0) as u64;
                        c.merged_parts += parts.len() as u64;
                        let fi = lift::FuncIn {
                            begin_va: base + f.begin as u64,
                            end_va: base + f.end as u64,
                            ins: &ins,
                            tables,
                            extra: extra_ranges(img, parts),
                            landing_pads: landing_pads(img, f),
                        };
                        match lift::lift_function(ctx, &fi) {
                            Ok(_) if !bad => {
                                c.lifted += 1;
                                c.instructions_in_lifted += ins.len() as u64;
                            }
                            Ok(_) => *c.blockers.entry("decode failure".into()).or_default() += 1,
                            Err(e) => {
                                let mut seen = HashSet::new();
                                for u in lift::unsupported(ctx, &fi) {
                                    let r = u.split_once(": ").map_or(u.as_str(), |x| x.1);
                                    let e = c.unsupported.entry(r.to_string()).or_default();
                                    e.0 += 1;
                                    if seen.insert(r.to_string()) {
                                        e.1 += 1;
                                    }
                                }
                                // "<rva>: <reason>"; bucket by reason.
                                let r = e.split_once(": ").map_or(e.as_str(), |x| x.1);
                                *c.blockers.entry(r.to_string()).or_default() += 1;
                            }
                        }
                    }
                    c
                })
            })
            .collect();
        for h in hs {
            let c = h.join().unwrap();
            cov.total += c.total;
            cov.lifted += c.lifted;
            cov.instructions_in_lifted += c.instructions_in_lifted;
            cov.untabled_jmps += c.untabled_jmps;
            cov.funcs_with_untabled_jmp += c.funcs_with_untabled_jmp;
            cov.merged_parts += c.merged_parts;
            for (k, v) in c.blockers {
                *cov.blockers.entry(k).or_default() += v;
            }
            for (k, v) in c.unsupported {
                let e = cov.unsupported.entry(k).or_default();
                e.0 += v.0;
                e.1 += v.1;
            }
        }
    });
    Ok(cov)
}

/// Shapes of the instruction windows before every `jmp reg` that has no recovered table
/// (mnemonic + first operand kind of the previous 6 instructions), most common first.
pub fn unresolved_jmp_shapes(img: &Image) -> Result<Vec<(String, u64)>> {
    use iced_x86::Code;
    let funcs: Vec<Func> = discover::discover(img)?.into_iter().filter(|f| !f.chained).collect();
    let mut m: BTreeMap<String, u64> = BTreeMap::new();
    for f in &funcs {
        let (ins, _) = analyze::decode_function(img, f);
        let tables = tables_for(img, f, &ins);
        for (n, j) in ins.iter().enumerate() {
            if j.code() != Code::Jmp_rm64 || j.op0_kind() != iced_x86::OpKind::Register {
                continue;
            }
            if tables.contains_key(&j.ip()) {
                continue;
            }
            let sig: Vec<String> = ins[n.saturating_sub(6)..n]
                .iter()
                .map(|i| format!("{:?}", i.mnemonic()))
                .collect();
            *m.entry(sig.join(" ")).or_default() += 1;
        }
    }
    let mut v: Vec<_> = m.into_iter().collect();
    v.sort_by_key(|a| std::cmp::Reverse(a.1));
    Ok(v)
}
