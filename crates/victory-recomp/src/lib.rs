//! Static x86-64 -> ARM64 recompiler foundation. See `docs/recompiler.md`.

pub mod analyze;
pub mod discover;
pub mod image;
pub mod lift;

/// Files a host needs next to a lifted unit: the runtime header and prototype host.
pub const RT_HOST_C: &str = include_str!("rt_host.c");
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

/// Lifts every non-chained `.pdata` function of `img` into one C unit.
pub fn lift_image(img: &Image) -> Result<lift::Unit> {
    let funcs: Vec<Func> = discover::discover(img)?
        .into_iter()
        .filter(|f| !f.chained)
        .collect();
    let base = img.info.image_base;
    let starts: HashSet<u64> = funcs.iter().map(|f| base + f.begin as u64).collect();
    let decoded: Vec<_> = funcs
        .iter()
        .map(|f| {
            let (ins, _) = analyze::decode_function(img, f);
            let tables = tables_for(img, f, &ins);
            (f, ins, tables)
        })
        .collect();
    let fins: Vec<lift::FuncIn> = decoded
        .iter()
        .map(|(f, ins, tables)| lift::FuncIn {
            begin_va: base + f.begin as u64,
            end_va: base + f.end as u64,
            ins,
            tables: tables.clone(),
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

/// How much of an image the current lifter handles, by first blocking reason.
#[derive(Default)]
pub struct Coverage {
    pub total: u64,
    pub lifted: u64,
    pub instructions_in_lifted: u64,
    pub blockers: BTreeMap<String, u64>,
}

pub fn coverage(img: &Image) -> Result<Coverage> {
    let funcs: Vec<Func> = discover::discover(img)?
        .into_iter()
        .filter(|f| !f.chained)
        .collect();
    let base = img.info.image_base;
    let starts: HashSet<u64> = funcs.iter().map(|f| base + f.begin as u64).collect();
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
                    for f in part {
                        let (ins, bad) = analyze::decode_function(img, f);
                        c.total += 1;
                        let tables = tables_for(img, f, &ins);
                        let fi = lift::FuncIn {
                            begin_va: base + f.begin as u64,
                            end_va: base + f.end as u64,
                            ins: &ins,
                            tables,
                        };
                        match lift::lift_function(ctx, &fi) {
                            Ok(_) if !bad => {
                                c.lifted += 1;
                                c.instructions_in_lifted += ins.len() as u64;
                            }
                            Ok(_) => *c.blockers.entry("decode failure".into()).or_default() += 1,
                            Err(e) => {
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
            for (k, v) in c.blockers {
                *cov.blockers.entry(k).or_default() += v;
            }
        }
    });
    Ok(cov)
}
