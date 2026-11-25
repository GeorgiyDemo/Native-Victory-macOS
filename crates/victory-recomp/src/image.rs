//! A read-only view of a PE image with RVA addressing, plus the code-discovery
//! seeds (.pdata, exports, relocations) the recompiler starts from.

use anyhow::{Result, bail};
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

    /// `UNWIND_INFO.Flags` (EHANDLER=1, UHANDLER=2, CHAININFO=4).
    pub fn unwind_flags(&self, unwind_rva: u32) -> Option<u8> {
        Some(self.bytes(unwind_rva, 1)?[0] >> 3)
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
