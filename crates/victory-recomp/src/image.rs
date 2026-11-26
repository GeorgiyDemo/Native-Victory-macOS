//! A read-only view of a PE image with RVA addressing, plus the code-discovery
//! seeds (.pdata, exports, relocations) the recompiler starts from.

use anyhow::{Result, bail};
use iced_x86::{Decoder, DecoderOptions, Mnemonic};
use victory_pe::layout::{self, dir};
use victory_pe::{ImageInfo, SectionInfo};

pub const SCN_EXEC: u32 = 0x2000_0000;
pub const SCN_WRITE: u32 = 0x8000_0000;

/// One `RUNTIME_FUNCTION` from the exception directory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Func {
    pub begin: u32,
    pub end: u32,
    pub unwind: u32,
    /// `UNW_FLAG_CHAININFO`: a continuation of another function (cold part or
    /// funclet), not an entry point of its own.
    pub chained: bool,
}

pub struct Image<'a> {
    pub data: &'a [u8],
    pub info: ImageInfo,
}

impl<'a> Image<'a> {
    pub fn parse(data: &'a [u8]) -> Result<Self> {
        Ok(Self {
            data,
            info: ImageInfo::parse(data)?,
        })
    }

    pub fn section_of(&self, rva: u32) -> Option<&SectionInfo> {
        self.info.sections.iter().find(|s| {
            rva.checked_sub(s.rva)
                .is_some_and(|d| d < s.virtual_size.max(s.file_size))
        })
    }

    pub fn is_exec(&self, rva: u32) -> bool {
        self.section_of(rva)
            .is_some_and(|s| s.characteristics & SCN_EXEC != 0)
    }

    /// File-backed bytes at `rva`, up to `len` (clamped to the section's raw data).
    pub fn bytes(&self, rva: u32, len: usize) -> Option<&'a [u8]> {
        let off = self.info.rva_to_file_offset(rva)? as usize;
        let s = self.section_of(rva)?;
        let room = (s.file_offset as usize + s.file_size as usize).saturating_sub(off);
        self.data.get(off..off + len.min(room))
    }

    pub fn read_u32(&self, rva: u32) -> Option<u32> {
        Some(u32::from_le_bytes(self.bytes(rva, 4)?.try_into().ok()?))
    }

    pub fn dir(&self, index: usize) -> Option<(u32, u32)> {
        let at = layout::data_directory_offset(self.data, index)?;
        let rva = layout::read_u32(self.data, at)?;
        let size = layout::read_u32(self.data, at + 4)?;
        (rva != 0 && size != 0).then_some((rva, size))
    }

    /// The `.pdata` table. On x64 every non-leaf function is listed, which
    /// makes it the best function-start oracle available.
    pub fn functions(&self) -> Result<Vec<Func>> {
        let Some((rva, size)) = self.dir(dir::EXCEPTION) else {
            return Ok(vec![]);
        };
        let Some(raw) = self.bytes(rva, size as usize) else {
            bail!("exception directory is not file-backed")
        };
        let mut out = Vec::with_capacity(raw.len() / 12);
        for e in raw.as_chunks::<12>().0 {
            let w = |i: usize| u32::from_le_bytes(e[i..i + 4].try_into().unwrap());
            let (begin, end, unwind) = (w(0), w(4), w(8));
            if begin == 0 && end == 0 {
                continue;
            }
            let chained = self.unwind_flags(unwind & !1).is_some_and(|f| f & 4 != 0);
            out.push(Func {
                begin,
                end,
                unwind,
                chained,
            });
        }
        out.sort_by_key(|f| f.begin);
        Ok(out)
    }

    /// The parent `RUNTIME_FUNCTION` a chained entry continues (UNWIND_INFO ends with the parent's
    /// 12-byte entry when CHAININFO is set); returns the parent's begin RVA.
    pub fn chain_parent(&self, f: &Func) -> Option<u32> {
        if !f.chained {
            return None;
        }
        let u = f.unwind & !1;
        let hdr = self.bytes(u, 4)?;
        let codes = hdr[2] as u32;
        let at = u + 4 + ((codes + 1) & !1) * 2;
        self.read_u32(at)
    }

    /// `UNWIND_INFO.Flags` (EHANDLER=1, UHANDLER=2, CHAININFO=4).
    pub fn unwind_flags(&self, unwind_rva: u32) -> Option<u8> {
        Some(self.bytes(unwind_rva, 1)?[0] >> 3)
    }

    /// The language handler of `f` (the personality routine) and the C scope table behind it (`__try`:
    /// begin, end, handler, jump target), if its unwind info has an exception or termination handler and the
    /// handler data looks like such a table (a C++ frame handler's data starts differently, so it is not
    /// read). shortcut: the handler is not checked to be `__C_specific_handler`.
    fn scope_table(&self, f: &Func) -> Option<(u32, Vec<[u32; 4]>)> {
        let u = f.unwind & !1;
        let h = self.bytes(u, 4)?;
        if f.chained || (h[0] >> 3) & 3 == 0 {
            return None;
        }
        let at = u + 4 + ((h[2] as u32 + 1) & !1) * 2;
        let handler = self.read_u32(at)?;
        let count = self.read_u32(at + 4).filter(|c| (1..=512).contains(c))?;
        let mut scopes = vec![];
        for i in 0..count {
            let e = at + 8 + i * 16;
            let s = [self.read_u32(e)?, self.read_u32(e + 4)?, self.read_u32(e + 8)?, self.read_u32(e + 12)?];
            if s[0] >= s[1] || !self.is_exec(s[0]) {
                return None;
            }
            scopes.push(s);
        }
        Some((handler, scopes))
    }

    /// The landing pads of `f`: the addresses an unwinder (or the C++ runtime that drives it) resumes at inside it.
    /// - the `JumpTarget`s of its C scope table (`__except` blocks);
    /// - where its C++ catch blocks continue: the catch funclets named by its `FuncInfo` (`__CxxFrameHandler3`, or
    ///   the compressed FH4 form) return the address of the code after the `try`, which the C++ runtime then jumps to
    ///   with the frame's registers; the funclet loads it with `lea rax, [rip + L]` (FH3), and FH4 also lists it;
    /// - in a function that calls `RtlUnwindEx` (the runtime itself), the labels it passes as the target.
    /// Only targets inside the function.
    pub fn landing_pads(&self, f: &Func) -> Vec<u32> {
        let mut pads: Vec<u32> = self
            .scope_table(f)
            .into_iter()
            .flat_map(|(_, scopes)| scopes)
            .map(|s| s[3])
            .filter(|j| (f.begin..f.end).contains(j))
            .collect();
        pads.extend(self.cxx_pads(f));
        pads.extend(self.unwind_target_pads(f));
        pads.sort_unstable();
        pads.dedup();
        pads
    }

    /// The language handler of `f` and where its handler data starts.
    fn handler_data(&self, f: &Func) -> Option<(u32, u32)> {
        let u = f.unwind & !1;
        let h = self.bytes(u, 4)?;
        if f.chained || (h[0] >> 3) & 3 == 0 {
            return None;
        }
        let at = u + 4 + ((h[2] as u32 + 1) & !1) * 2;
        Some((self.read_u32(at)?, at + 4))
    }

    /// The name of the import a one-instruction thunk `jmp qword ptr [rip + slot]` at `rva` goes through (how a
    /// frame handler in another DLL is reached: the unwind info holds an RVA inside the image).
    fn thunk_import(&self, rva: u32) -> Option<String> {
        let b = self.bytes(rva, 6)?;
        if b[0] != 0xFF || b[1] != 0x25 {
            return None;
        }
        let slot = (rva as i64 + 6 + i32::from_le_bytes(b[2..6].try_into().ok()?) as i64) as u32;
        self.import_at(slot)
    }

    fn import_at(&self, slot: u32) -> Option<String> {
        self.info
            .imports
            .iter()
            .flat_map(|m| &m.entries)
            .find(|e| e.iat_slot_rva == slot)
            .and_then(|e| match &e.symbol {
                victory_pe::ImportSymbol::Name { name, .. } => Some(name.clone()),
                _ => None,
            })
    }

    /// The C++ exception data behind `f`'s language handler: which form (by the handler's import name, else by the
    /// FH3 magic number) and the RVA of the `FuncInfo`.
    fn cxx_info(&self, f: &Func) -> Option<(bool, u32)> {
        let (handler, data) = self.handler_data(f)?;
        let info = self.read_u32(data)?;
        if let Some(name) = self.thunk_import(handler) {
            if name.contains("CxxFrameHandler4") || name.contains("GSHandlerCheck_EH4") {
                return Some((true, info));
            }
            if name.contains("CxxFrameHandler") || name.contains("GSHandlerCheck_EH") {
                return Some((false, info));
            }
            return None;
        }
        // A handler in the image itself (a static runtime): only FH3 is told by its data.
        let magic = self.read_u32(info)?;
        (0x1993_0520..=0x1993_0522).contains(&(magic & 0x1fff_ffff)).then_some((false, info))
    }

    /// The funclets `f`'s C++ data names, `(unwind actions, catch handlers, continuation addresses FH4 lists)`.
    /// shortcut: the FH4 layout is implemented from memory of ehdata4; no FH4 binary was at hand to check it against.
    fn cxx_funclets(&self, f: &Func) -> Option<(Vec<u32>, Vec<u32>, Vec<u32>)> {
        let (fh4, info) = self.cxx_info(f)?;
        let (mut actions, mut handlers, mut conts) = (vec![], vec![], vec![]);
        if !fh4 {
            // FuncInfo: magic, maxState, dispUnwindMap, nTryBlocks, dispTryBlockMap, ...
            let (max_state, unwind, tries, try_map) =
                (self.read_u32(info + 4)?, self.read_u32(info + 8)?, self.read_u32(info + 12)?, self.read_u32(info + 16)?);
            if max_state > 4096 || tries > 1024 {
                return None;
            }
            if unwind != 0 {
                for k in 0..max_state {
                    // UnwindMapEntry: toState, action (RVA, or 0)
                    if let Some(a) = self.read_u32(unwind + 8 * k + 4).filter(|&a| a != 0 && self.is_exec(a)) {
                        actions.push(a);
                    }
                }
            }
            for t in 0..tries {
                // TryBlockMapEntry: tryLow, tryHigh, catchHigh, nCatches, dispHandlerArray
                let tb = try_map + 20 * t;
                let (n, array) = (self.read_u32(tb + 12)?, self.read_u32(tb + 16)?);
                if n > 256 {
                    return None;
                }
                for h in 0..n {
                    // HandlerType: adjectives, dispType, dispCatchObj, dispOfHandler, dispFrame
                    if let Some(funclet) = self.read_u32(array + 20 * h + 12).filter(|&x| self.is_exec(x)) {
                        handlers.push(funclet);
                    }
                }
            }
        } else {
            let mut c = Cursor { img: self, at: info };
            let header = c.u8()?;
            let (catch, bbt, unwind_map, try_map) = (header & 1 != 0, header & 4 != 0, header & 8 != 0, header & 16 != 0);
            let _ = catch;
            if bbt {
                c.varint()?;
            }
            if unwind_map {
                c.i32()?;
            }
            if !try_map {
                return Some((actions, handlers, conts));
            }
            let map = c.i32()? as u32;
            let mut m = Cursor { img: self, at: map };
            let tries = m.varint()?;
            if tries > 1024 {
                return None;
            }
            for _ in 0..tries {
                // tryLow, tryHigh (as a delta), catchHigh (as a delta), dispHandlerArray
                m.varint()?;
                m.varint()?;
                m.varint()?;
                let array = m.i32()? as u32;
                let mut h = Cursor { img: self, at: array };
                let n = h.varint()?;
                if n > 256 {
                    return None;
                }
                for _ in 0..n {
                    // header: adjectives, dispType, dispCatchObj present; continuations are function-relative;
                    // how many continuation addresses (0, 1 or 2)
                    let flags = h.u8()?;
                    if flags & 1 != 0 {
                        h.varint()?;
                    }
                    if flags & 2 != 0 {
                        h.i32()?;
                    }
                    if flags & 4 != 0 {
                        h.varint()?;
                    }
                    let funclet = h.i32()? as u32;
                    if self.is_exec(funclet) {
                        handlers.push(funclet);
                    }
                    for _ in 0..(flags >> 4) & 3 {
                        let rel = flags & 8 != 0;
                        let v = if rel { f.begin.wrapping_add(h.varint()?) } else { h.i32()? as u32 };
                        conts.push(v);
                    }
                }
            }
        }
        Some((actions, handlers, conts))
    }

    /// Where `f`'s catch blocks continue (see [`Self::landing_pads`]).
    fn cxx_pads(&self, f: &Func) -> Vec<u32> {
        let Some((_, handlers, conts)) = self.cxx_funclets(f) else { return vec![] };
        let base = self.info.image_base;
        let inside = |v: u32| (f.begin..f.end).contains(&v);
        let mut pads: Vec<u32> = conts.into_iter().filter(|&v| inside(v)).collect();
        for funclet in handlers {
            // A funclet is outside the function it belongs to; its `lea rax, [rip + L]` names a label in the parent.
            if inside(funclet) {
                continue;
            }
            let Some(code) = self.bytes(funclet, 0x400) else { continue };
            let mut dec = Decoder::with_ip(64, code, base + funclet as u64, DecoderOptions::NONE);
            while dec.can_decode() {
                let i = dec.decode();
                if i.is_invalid() || i.mnemonic() == Mnemonic::Int3 {
                    break;
                }
                if i.mnemonic() == Mnemonic::Lea && i.is_ip_rel_memory_operand() {
                    let t = i.ip_rel_memory_address().wrapping_sub(base);
                    if t < u32::MAX as u64 && inside(t as u32) {
                        pads.push(t as u32);
                    }
                }
            }
        }
        pads
    }

    /// In a function that calls `RtlUnwindEx` (or `RtlRestoreContext`): the labels of the function whose address it
    /// takes, which it can hand over as the place to continue at. This is how a C++ runtime that is itself lifted
    /// resumes after unwinding; they are landing pads like a scope table's.
    fn unwind_target_pads(&self, f: &Func) -> Vec<u32> {
        let slots: Vec<u32> = self
            .info
            .imports
            .iter()
            .flat_map(|m| &m.entries)
            .filter(|e| matches!(&e.symbol, victory_pe::ImportSymbol::Name { name, .. } if matches!(name.as_str(), "RtlUnwindEx" | "RtlUnwind" | "RtlRestoreContext")))
            .map(|e| e.iat_slot_rva)
            .collect();
        if slots.is_empty() || f.chained {
            return vec![];
        }
        let base = self.info.image_base;
        let (ins, _) = crate::analyze::decode_function(self, f);
        let rva = |a: u64| a.wrapping_sub(base);
        let calls_unwind = ins.iter().any(|i| {
            i.is_ip_rel_memory_operand()
                && matches!(i.mnemonic(), Mnemonic::Call | Mnemonic::Jmp)
                && slots.contains(&(rva(i.ip_rel_memory_address()) as u32))
        });
        if !calls_unwind {
            return vec![];
        }
        ins.iter()
            .filter(|i| i.mnemonic() == Mnemonic::Lea && i.is_ip_rel_memory_operand())
            .map(|i| rva(i.ip_rel_memory_address()))
            .filter(|&t| t < u32::MAX as u64 && (f.begin..f.end).contains(&(t as u32)) && t as u32 != f.begin)
            .map(|t| t as u32)
            .collect()
    }

    /// Code the unwinder calls that nothing in the image calls by name: the language handler of `f` (often a
    /// thunk to an import) and the filter and termination funclets of its scope table (frameless leaves have
    /// no `.pdata`), and the funclets of its C++ data (destructor actions, catch blocks). Seeds for code discovery.
    pub fn handler_seeds(&self, f: &Func) -> Vec<u32> {
        let mut seeds = vec![];
        if let Some((handler, _)) = self.handler_data(f) {
            seeds.push(handler);
        }
        if let Some((handler, scopes)) = self.scope_table(f) {
            seeds.push(handler);
            seeds.extend(scopes.iter().map(|s| s[2]).filter(|&h| h > 1));
        }
        if let Some((actions, handlers, _)) = self.cxx_funclets(f) {
            seeds.extend(actions);
            seeds.extend(handlers);
        }
        seeds
    }

    /// RVAs of every `IMAGE_REL_BASED_DIR64` relocation slot.
    pub fn dir64_relocs(&self) -> Vec<u32> {
        let mut out = vec![];
        let Some((rva, size)) = self.dir(dir::BASERELOC) else {
            return out;
        };
        let Some(raw) = self.bytes(rva, size as usize) else {
            return out;
        };
        let mut p = 0;
        while p + 8 <= raw.len() {
            let page = u32::from_le_bytes(raw[p..p + 4].try_into().unwrap());
            let blk = u32::from_le_bytes(raw[p + 4..p + 8].try_into().unwrap()) as usize;
            if blk < 8 {
                break;
            }
            for e in raw[p + 8..(p + blk).min(raw.len())].as_chunks::<2>().0 {
                let v = u16::from_le_bytes([e[0], e[1]]);
                if v >> 12 == 10 {
                    out.push(page + (v & 0xfff) as u32);
                }
            }
            p += blk;
        }
        out
    }
}

/// A reader of the compressed data FH4 exception tables use.
struct Cursor<'a, 'b> {
    img: &'b Image<'a>,
    at: u32,
}

impl Cursor<'_, '_> {
    fn u8(&mut self) -> Option<u8> {
        let b = self.img.bytes(self.at, 1)?[0];
        self.at += 1;
        Some(b)
    }

    fn i32(&mut self) -> Option<i32> {
        let v = self.img.read_u32(self.at)?;
        self.at += 4;
        Some(v as i32)
    }

    /// FH4's variable-length unsigned integer, see [`read_varint`].
    fn varint(&mut self) -> Option<u32> {
        let bytes = self.img.bytes(self.at, 5).or_else(|| self.img.bytes(self.at, 1))?;
        let (v, n) = read_varint(bytes)?;
        self.at += n as u32;
        Some(v)
    }
}

/// FH4's variable-length unsigned integer: the number of trailing one bits of the first byte (up to four) gives the
/// number of bytes that follow, which carry the higher bits; four ones and a whole 32-bit word after the byte is the
/// longest form. Returns the value and the bytes used.
fn read_varint(b: &[u8]) -> Option<(u32, usize)> {
    let first = *b.first()? as u32;
    let extra = first.trailing_ones().min(4) as usize;
    if extra == 4 {
        return Some((u32::from_le_bytes(b.get(1..5)?.try_into().ok()?), 5));
    }
    let mut v = first >> (extra + 1);
    for k in 0..extra {
        v |= (*b.get(1 + k)? as u32) << (7 - extra + 8 * k);
    }
    Some((v, 1 + extra))
}

#[cfg(test)]
mod tests {
    use super::read_varint;

    #[test]
    fn fh4_integers_decode_in_every_length() {
        assert_eq!(read_varint(&[0x0A]), Some((5, 1)));
        assert_eq!(read_varint(&[0x00]), Some((0, 1)));
        // 2 bytes: low bits 01, then 6 bits + 8 more
        assert_eq!(read_varint(&[0x15, 0x01]), Some((5 | 1 << 6, 2)));
        // 3 bytes: low bits 011
        assert_eq!(read_varint(&[0x2B, 0x01, 0x02]), Some((5 | 1 << 5 | 2 << 13, 3)));
        // 4 bytes: low bits 0111
        assert_eq!(read_varint(&[0x57, 0x01, 0x02, 0x03]), Some((5 | 1 << 4 | 2 << 12 | 3 << 20, 4)));
        // 5 bytes: low bits 1111, then the word
        assert_eq!(read_varint(&[0x0F, 0x78, 0x56, 0x34, 0x12]), Some((0x1234_5678, 5)));
        // a truncated one is nothing
        assert_eq!(read_varint(&[0x15]), None);
    }
}
