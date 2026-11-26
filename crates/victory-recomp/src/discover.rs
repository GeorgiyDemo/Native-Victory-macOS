//! Code discovery beyond `.pdata`. x64 compilers emit unwind info only for
//! functions that need it, so leaf functions (no stack use, no calls) are
//! missing from `.pdata`; on the title that is ~105k extra function starts.
//! Seeds: exports, entry, TLS callbacks, DIR64 relocation targets (vtables,
//! function-pointer tables) and direct call / tail-jump targets, iterated to a
//! fixpoint. Gap-function extents come from a linear walk that stops at the
//! first terminator past every forward branch.

use std::collections::BTreeSet;

use iced_x86::{Decoder, DecoderOptions, FlowControl, Mnemonic, OpKind};

use crate::image::{Func, Image};

fn covering(sorted: &[Func], rva: u32) -> bool {
    let i = sorted.partition_point(|f| f.begin <= rva);
    i > 0 && rva < sorted[i - 1].end
}

fn section_end(img: &Image, rva: u32) -> u32 {
    img.section_of(rva)
        .map_or(rva, |s| s.rva + s.virtual_size.min(s.file_size.max(1)))
}

/// Walks from `start` (not past `limit`) and returns the end of the function and
/// the direct branch/call targets seen.
fn walk(img: &Image, start: u32, limit: u32) -> (u32, Vec<u32>) {
    let base = img.info.image_base;
    let Some(code) = img.bytes(start, (limit - start) as usize) else {
        return (start, vec![]);
    };
    let mut dec = Decoder::with_ip(64, code, base + start as u64, DecoderOptions::NONE);
    let mut max_fwd = start;
    let mut targets = vec![];
    let mut end = start;
    while dec.can_decode() {
        let i = dec.decode();
        if i.is_invalid() {
            break;
        }
        end = (i.next_ip() - base) as u32;
        let direct = matches!(i.op0_kind(), OpKind::NearBranch64);
        // `lea reg, [rip + f]`: the address of code taken as a value, a callback or a pointer in a table built at run
        // time. A leaf function (no `.pdata`) that is only reached this way has no other seed. Targets inside the
        // function being walked are its own labels; those inside a function with `.pdata` are dropped by the caller.
        if i.mnemonic() == Mnemonic::Lea && i.is_ip_rel_memory_operand() {
            let t = i.ip_rel_memory_address().wrapping_sub(base);
            if t < u32::MAX as u64 && !(t as u32 >= start && (t as u32) < limit) {
                targets.push(t as u32);
            }
        }
        match i.flow_control() {
            FlowControl::Call if direct => targets.push((i.near_branch_target() - base) as u32),
            FlowControl::ConditionalBranch | FlowControl::UnconditionalBranch if direct => {
                let t = (i.near_branch_target() - base) as u32;
                if t > start && t < limit {
                    max_fwd = max_fwd.max(t);
                } else {
                    targets.push(t);
                }
            }
            _ => {}
        }
        let terminal = matches!(
            i.flow_control(),
            FlowControl::Return | FlowControl::UnconditionalBranch | FlowControl::IndirectBranch
        ) || matches!(i.mnemonic(), Mnemonic::Int3 | Mnemonic::Ud2);
        if terminal && end > max_fwd {
            break;
        }
    }
    (end, targets)
}

/// All functions: `.pdata` entries plus discovered leaf functions and thunks.
pub fn discover(img: &Image) -> anyhow::Result<Vec<Func>> {
    discover_with(img, &[])
}

/// [`discover`] with more function starts the caller knows of (targets a run found that no static pass can: pointers
/// kept as image-relative offsets, say).
pub fn discover_with(img: &Image, extra: &[u32]) -> anyhow::Result<Vec<Func>> {
    let mut known = img.functions()?;
    let pdata_n = known.len();
    // Targets of pdata functions: every instruction must be scanned once.
    let mut pending: Vec<u32> = vec![];
    for e in &img.info.exports {
        pending.extend(e.rva);
    }
    pending.push(img.info.entry_rva);
    pending.extend(extra);
    if let Some(t) = &img.info.tls {
        pending.extend(&t.callback_rvas);
    }
    for r in img.dir64_relocs() {
        if let Some(b) = img.bytes(r, 8) {
            let t = u64::from_le_bytes(b.try_into().unwrap()).wrapping_sub(img.info.image_base);
            if t <= u32::MAX as u64 {
                pending.push(t as u32);
            }
        }
    }
    for f in known.iter().take(pdata_n) {
        pending.extend(img.handler_seeds(f));
        pending.extend(walk(img, f.begin, f.end).1);
    }
    let mut starts: BTreeSet<u32> = BTreeSet::new();
    let mut gap: Vec<u32> = vec![];
    while let Some(t) = pending.pop() {
        if !img.is_exec(t) || covering(&known, t) || !starts.insert(t) {
            continue;
        }
        let limit = starts
            .range(t + 1..)
            .next()
            .copied()
            .unwrap_or(u32::MAX)
            .min(section_end(img, t));
        let next_known = known[known.partition_point(|f| f.begin <= t)..]
            .first()
            .map_or(u32::MAX, |f| f.begin);
        let (_, targets) = walk(img, t, limit.min(next_known));
        pending.extend(targets);
        gap.push(t);
    }
    // Final extents with every start known.
    let all: BTreeSet<u32> = known
        .iter()
        .map(|f| f.begin)
        .chain(gap.iter().copied())
        .collect();
    for &t in &gap {
        let next = all.range(t + 1..).next().copied().unwrap_or(u32::MAX);
        let (end, _) = walk(img, t, next.min(section_end(img, t)));
        if end > t {
            known.push(Func {
                begin: t,
                end,
                unwind: 0,
                chained: false,
            });
        }
    }
    known.sort_by_key(|f| f.begin);
    Ok(known)
}
