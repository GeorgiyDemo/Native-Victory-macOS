//! Legacy (non-VEX) SSE integer forms, AES-NI, PCLMULQDQ, SSE4a and the odd system
//! instructions. A legacy form is its VEX helper applied to the destination in place:
//! two operands, destination is also source 1, and the upper ymm half is left alone
//! (so unlike VEX it must not be zeroed).

use iced_x86::{Mnemonic as M, OpKind};

use super::{Instruction, L, R};

/// `dst = h_v<name>(dst, src)`.
const BIN: &[&str] = &[
    "Paddb",
    "Paddw",
    "Paddd",
    "Paddq",
    "Psubb",
    "Psubw",
    "Psubd",
    "Psubq",
    "Paddsb",
    "Paddsw",
    "Paddusb",
    "Paddusw",
    "Psubsb",
    "Psubsw",
    "Psubusb",
    "Psubusw",
    "Pmullw",
    "Pmulld",
    "Pmulhw",
    "Pmulhuw",
    "Pmulhrsw",
    "Pmuludq",
    "Pmuldq",
    "Pmaddwd",
    "Pmaddubsw",
    "Pavgb",
    "Pavgw",
    "Pminsb",
    "Pminsw",
    "Pminsd",
    "Pminub",
    "Pminuw",
    "Pminud",
    "Pmaxsb",
    "Pmaxsw",
    "Pmaxsd",
    "Pmaxub",
    "Pmaxuw",
    "Pmaxud",
    "Pcmpeqb",
    "Pcmpeqw",
    "Pcmpeqd",
    "Pcmpeqq",
    "Pcmpgtb",
    "Pcmpgtw",
    "Pcmpgtd",
    "Pcmpgtq",
    "Psignb",
    "Psignw",
    "Psignd",
    "Psadbw",
    "Phaddw",
    "Phaddd",
    "Phsubw",
    "Phsubd",
    "Phaddsw",
    "Phsubsw",
    "Packsswb",
    "Packuswb",
    "Packssdw",
    "Packusdw",
    "Unpcklps",
    "Unpckhps",
    "Unpcklpd",
    "Unpckhpd",
    "Punpcklbw",
    "Punpckhbw",
    "Punpcklwd",
    "Punpckhwd",
    "Punpckldq",
    "Punpckhdq",
    "Punpcklqdq",
    "Punpckhqdq",
    "Pshufb",
    "Haddps",
    "Haddpd",
    "Hsubps",
    "Hsubpd",
    "Addsubps",
    "Addsubpd",
    "Movlhps",
    "Movhlps",
    "Minps",
    "Minpd",
    "Maxps",
    "Maxpd",
    "Aesenc",
    "Aesenclast",
    "Aesdec",
    "Aesdeclast",
];
/// `dst = h_v<name>(dst, src, imm)`.
const BIN_IMM: &[&str] = &[
    "Palignr",
    "Pblendw",
    "Blendps",
    "Blendpd",
    "Shufps",
    "Shufpd",
    "Pclmulqdq",
    "Cmpps",
    "Cmppd",
    "Dpps",
    "Dppd",
];
/// `dst = h_v<name>(src)`.
const UN: &[&str] = &["Pabsb", "Pabsw", "Pabsd", "Phminposuw", "Aesimc", "Cvtdq2ps", "Cvttps2dq", "Cvtps2dq"];
/// `dst = h_v<name>(src, imm)`.
const UN_IMM: &[&str] = &["Pshufd", "Pshufhw", "Pshuflw", "Aeskeygenassist"];
/// `dst = h_v<name>(dst, count)`; the count is an immediate or an xmm/m128 low qword.
const SHIFT: &[&str] = &[
    "Psllw", "Pslld", "Psllq", "Psrlw", "Psrld", "Psrlq", "Psraw", "Psrad", "Pslldq", "Psrldq",
];
/// Sign/zero-extending converts: (name, source bytes).
const EXT: &[(&str, u32)] = &[
    ("Pmovsxbw", 8),
    ("Pmovsxbd", 4),
    ("Pmovsxbq", 2),
    ("Pmovsxwd", 8),
    ("Pmovsxwq", 4),
    ("Pmovsxdq", 8),
    ("Pmovzxbw", 8),
    ("Pmovzxbd", 4),
    ("Pmovzxbq", 2),
    ("Pmovzxwd", 8),
    ("Pmovzxwq", 4),
    ("Pmovzxdq", 8),
    ("Cvtdq2pd", 8),
    ("Cvtps2pd", 8),
];

fn vh(name: &str) -> String {
    format!("h_v{}", name.to_lowercase())
}

impl L<'_> {
    fn xdst(i: &Instruction) -> R<usize> {
        let r = i.op0_register();
        if i.op0_kind() == OpKind::Register && r.is_xmm() {
            Ok(r.number())
        } else {
            Err(format!("xmm operand {r:?}"))
        }
    }

    pub(super) fn legacy(&mut self, i: &Instruction) -> R<bool> {
        if i.code().encoding() == iced_x86::EncodingKind::VEX {
            return Ok(false);
        }
        let m = i.mnemonic();
        let name = format!("{m:?}");
        let n = name.as_str();
        if BIN.contains(&n) && i.op_count() == 2 {
            let (d, s) = (Self::xdst(i)?, self.x128(i, 1)?);
            self.emit(&format!("x{d} = {}(x{d}, {s});", vh(n)));
        } else if BIN_IMM.contains(&n) {
            let (d, s) = (Self::xdst(i)?, self.x128(i, 1)?);
            let v = i.immediate(2) as u8 as u32;
            let v = match m {
                M::Blendps => v & 15,
                M::Blendpd | M::Shufpd => v & 3,
                _ => v,
            };
            self.emit(&format!("x{d} = {}(x{d}, {s}, {v});", vh(n)));
        } else if matches!(m, M::Blendvps | M::Blendvpd | M::Pblendvb) {
            let (d, s) = (Self::xdst(i)?, self.x128(i, 1)?);
            self.emit(&format!("x{d} = {}(x{d}, {s}, x0);", vh(&format!("{m:?}"))));
        } else if UN.contains(&n) {
            let (d, s) = (Self::xdst(i)?, self.x128(i, 1)?);
            self.emit(&format!("x{d} = {}({s});", vh(n)));
        } else if UN_IMM.contains(&n) {
            let (d, s) = (Self::xdst(i)?, self.x128(i, 1)?);
            self.emit(&format!("x{d} = {}({s}, {});", vh(n), i.immediate(2) as u8));
        } else if SHIFT.contains(&n) {
            let d = Self::xdst(i)?;
            let cnt = match i.op_kind(1) {
                OpKind::Register | OpKind::Memory => self.x128(i, 1)?,
                _ => format!("(X128){{ .q = {{ {:#x}ull, 0 }} }}", i.immediate(1) as u8),
            };
            self.emit(&format!("x{d} = {}(x{d}, {cnt});", vh(n)));
        } else if let Some(&(_, bytes)) = EXT.iter().find(|e| e.0 == n) {
            let d = Self::xdst(i)?;
            let s = match i.op_kind(1) {
                OpKind::Register => format!("x{}", i.op1_register().number()),
                _ => format!("ldx{}({})", bytes * 8, self.addr(i)?),
            };
            self.emit(&format!("x{d} = {}({s}, 0);", vh(n)));
        } else {
            return self.legacy_misc(i);
        }
        Ok(true)
    }

    fn legacy_misc(&mut self, i: &Instruction) -> R<bool> {
        let m = i.mnemonic();
        match m {
            M::Pinsrb | M::Pinsrw | M::Pinsrd | M::Pinsrq => {
                let d = Self::xdst(i)?;
                let (bits, l, n) = match m {
                    M::Pinsrb => (8, "b", 15),
                    M::Pinsrw => (16, "w", 7),
                    M::Pinsrd => (32, "d", 3),
                    _ => (64, "q", 1),
                };
                let src_bits = if i.op1_kind() == OpKind::Register {
                    64
                } else {
                    bits
                };
                let v = self.rd(i, 1, src_bits)?;
                self.emit(&format!(
                    "x{d}.{l}[{}] = (u{bits})({v});",
                    i.immediate(2) as u32 & n
                ));
            }
            M::Pextrb | M::Pextrw | M::Pextrd | M::Pextrq | M::Extractps => {
                let k = i.op1_register().number();
                let v = i.immediate(2) as u32;
                let e = match m {
                    M::Pextrb => format!("x{k}.b[{}]", v & 15),
                    M::Pextrw => format!("x{k}.w[{}]", v & 7),
                    M::Pextrq => format!("x{k}.q[{}]", v & 1),
                    _ => format!("x{k}.d[{}]", v & 3),
                };
                let w = self.wr(i, 0, &format!("(u64)({e})"))?;
                self.emit(&w);
            }
            M::Pmovmskb | M::Movmskps | M::Movmskpd => {
                let k = i.op1_register().number();
                let f = match m {
                    M::Pmovmskb => "mskb",
                    M::Movmskps => "mskps",
                    _ => "mskpd",
                };
                let w = self.wr(i, 0, &format!("(u64){f}(x{k})"))?;
                self.emit(&w);
            }
            M::Insertq => {
                let d = Self::xdst(i)?;
                let s = i.op1_register().number();
                if i.op_count() == 4 {
                    self.emit(&format!(
                        "x{d} = h_insertq(x{d}, x{s}, {}, {});",
                        i.immediate(2) as u8,
                        i.immediate(3) as u8
                    ));
                } else {
                    self.emit(&format!(
                        "x{d} = h_insertq(x{d}, x{s}, x{s}.q[1] & 63, (x{s}.q[1] >> 8) & 63);"
                    ));
                }
            }
            M::Extrq => {
                let d = Self::xdst(i)?;
                if i.op_count() == 3 {
                    self.emit(&format!(
                        "x{d} = h_extrq(x{d}, {}, {});",
                        i.immediate(1) as u8,
                        i.immediate(2) as u8
                    ));
                } else {
                    let s = i.op1_register().number();
                    self.emit(&format!(
                        "x{d} = h_extrq(x{d}, x{s}.q[0] & 63, (x{s}.q[0] >> 8) & 63);"
                    ));
                }
            }
            // Scalar compare with a predicate; the string instruction of the same name has no xmm operand.
            M::Cmpss | M::Cmpsd if i.op_count() == 3 => {
                let dbl = m == M::Cmpsd;
                let d = Self::xdst(i)?;
                let s = self.xscalar(i, 1, dbl)?;
                let p = i.immediate(2) as u8 & 7;
                if dbl {
                    self.emit(&format!("x{d}.q[0] = cmp_pred(x{d}.g[0], {s}, {p}) ? ~0ull : 0;"));
                } else {
                    self.emit(&format!("x{d}.d[0] = cmp_pred(x{d}.f[0], {s}, {p}) ? ~0u : 0;"));
                }
            }
            M::Ptest => {
                let (d, s) = (Self::xdst(i)?, self.x128(i, 1)?);
                self.emit(&format!(
                    "{{ X128 pa = x{d}, pb = {s}; fl_set_test(&F, !h_pt_and(pa, pb, ~0ull), !h_pt_andn(pa, pb, ~0ull)); }}"
                ));
            }
            // SSE4.2 string compare: the helper sets the flags and the mask, returns the index.
            M::Pcmpistri | M::Pcmpestri | M::Pcmpistrm | M::Pcmpestrm => {
                let (d, s) = (Self::xdst(i)?, self.x128(i, 1)?);
                let imm = i.immediate(2) as u8;
                let explicit = matches!(m, M::Pcmpestri | M::Pcmpestrm);
                let (la, lb) = if explicit { ("(s64)(s32)r0", "(s64)(s32)r2") } else { ("(-0x7fffffffffffffffll-1)", "(-0x7fffffffffffffffll-1)") };
                let out = if matches!(m, M::Pcmpistri | M::Pcmpestri) { "r1 = ix;" } else { "x0 = mk;" };
                self.emit(&format!(
                    "{{ X128 mk; u32 ix = h_pcmpstr(x{d}, {s}, {la}, {lb}, {imm}, &F, &mk); {out} }}"
                ));
            }
            M::Crc32 => {
                let bits = Self::bits0(i)?;
                let sb = match i.op1_kind() {
                    OpKind::Register => i.op1_register().size() as u32 * 8,
                    _ => i.memory_size().size() as u32 * 8,
                };
                let (d, s) = (self.rd(i, 0, bits)?, self.rd(i, 1, sb)?);
                let w = self.wr(i, 0, &format!("h_crc32c((u32)({d}), {s}, {})", sb / 8))?;
                self.emit(&w);
            }
            // Control-flow shadow stack and the x87 control word: no CET, default x87 state.
            M::Rdsspq | M::Rdsspd => {}
            M::Fnstcw => {
                let a = self.addr(i)?;
                self.emit(&format!("st16({a}, 0x37f);"));
            }
            M::Fldcw | M::Wait | M::Fninit | M::Fnclex => {}
            // ---- odd system instructions ----
            M::Int | M::Int1 => {
                let n = if m == M::Int { i.immediate(0) as u8 } else { 1 };
                // int 0x29 is the Windows fast-fail trap; everything else is a debugger or kernel call.
                self.emit(&format!("rt_trap(\"int {n:#x}\");"));
            }
            M::Hlt => self.emit("rt_trap(\"hlt\");"),
            M::Monitorx | M::Mwaitx | M::Monitor | M::Mwait => {}
            M::Clzero => self.emit("memset((u8*)(r0 & ~63ull), 0, 64);"),
            M::Enter => {
                let size = i.immediate(0) as u16;
                let lvl = i.immediate(1) as u32 & 31;
                // Nesting copies the enclosing frame pointers, then pushes the new frame pointer.
                let copy = if lvl > 1 {
                    format!(
                        "for (u32 k = 1; k < {lvl}; k++) {{ r5 -= 8; r4 -= 8; st64((u8*)r4, ld64((u8*)r5)); }} "
                    )
                } else {
                    String::new()
                };
                let push_ft = if lvl > 0 {
                    "r4 -= 8; st64((u8*)r4, ft); "
                } else {
                    ""
                };
                self.emit(&format!("{{ r4 -= 8; st64((u8*)r4, r5); u64 ft = r4; {copy}{push_ft}r5 = ft; r4 -= {size}; }}"));
            }
            _ => return Ok(false),
        }
        Ok(true)
    }
}
