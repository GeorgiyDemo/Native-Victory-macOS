//! Atomics (`lock`, `xchg`, `cmpxchg`), string ops, bit tests, BMI1/BMI2, flag moves
//! and the odd integer instructions.
//!
//! Memory model: every `lock`-prefixed or implicitly locked instruction becomes a
//! `__atomic` builtin with `__ATOMIC_SEQ_CST` (x86 locked operations are full
//! barriers). Plain loads and stores stay plain; see docs/recompiler.md.

use iced_x86::Mnemonic as M;

use super::{Instruction, L, OpKind, R, gpr, mask};

const SC: &str = "__ATOMIC_SEQ_CST";

impl L<'_> {
    /// `(type, bits)` of the operand-0 integer size.
    fn ity(i: &Instruction) -> R<(String, u32)> {
        let b = Self::bits0(i)?;
        Ok((format!("u{b}"), b))
    }

    /// `lock`-prefixed read-modify-write on memory, `xchg`, `xadd` and `cmpxchg`.
    fn atomic(&mut self, i: &Instruction) -> R<bool> {
        let m = i.mnemonic();
        if i.op_count() == 0 || (i.op0_kind() != OpKind::Memory && !(m == M::Xchg && i.op1_kind() == OpKind::Memory)) {
            return Ok(false);
        }
        if !matches!(
            m,
            M::Add | M::Sub | M::And | M::Or | M::Xor | M::Inc | M::Dec | M::Neg | M::Not | M::Xchg | M::Xadd | M::Cmpxchg
        ) {
            return Ok(false);
        }
        let (t, bits) = Self::ity(i)?;
        let a = self.addr(i)?;
        let mk = mask(bits);
        let p = format!("{t} *p = ({t}*)({a});");
        match m {
            M::Add | M::Sub | M::And | M::Or | M::Xor => {
                let (f, op, kind) = match m {
                    M::Add => ("add", "+", "FK_ADD"),
                    M::Sub => ("sub", "-", "FK_SUB"),
                    M::And => ("and", "&", "FK_LOGIC"),
                    M::Or => ("or", "|", "FK_LOGIC"),
                    _ => ("xor", "^", "FK_LOGIC"),
                };
                let b = self.rd(i, 1, bits)?;
                self.emit(&format!(
                    "{{ {p} u64 b = {b}; u64 a = (u64)__atomic_fetch_{f}(p, ({t})b, {SC}); u64 r = (a {op} b) & {mk}; F.k = {kind}; F.sz = {bits}; F.a = a; F.b = b; F.r = r; F.c = 0; }}"
                ));
            }
            M::Inc | M::Dec => {
                let (f, op, kind) = if m == M::Inc { ("add", "+", "FK_INC") } else { ("sub", "-", "FK_DEC") };
                self.emit(&format!(
                    "{{ {p} u64 cf = fl_cf(&F); u64 a = (u64)__atomic_fetch_{f}(p, ({t})1, {SC}); u64 r = (a {op} 1) & {mk}; F.k = {kind}; F.sz = {bits}; F.r = r; F.c = cf; }}"
                ));
            }
            M::Neg | M::Not => {
                let (nw, fl) = if m == M::Neg {
                    ("(u64)(0 - old) & MK", "F.k = FK_SUB; F.sz = BITS; F.a = 0; F.b = old; F.r = nw; F.c = 0;")
                } else {
                    ("~old & MK", "")
                };
                let nw = nw.replace("MK", &mk);
                let fl = fl.replace("BITS", &bits.to_string());
                self.emit(&format!(
                    "{{ {p} {t} old = __atomic_load_n(p, __ATOMIC_RELAXED); u64 nw; do {{ nw = {nw}; }} while (!__atomic_compare_exchange_n(p, &old, ({t})nw, 0, {SC}, {SC})); {fl} }}"
                ));
            }
            M::Xchg => {
                let r = if i.op0_kind() == OpKind::Memory { 1 } else { 0 };
                let v = self.rd(i, r, bits)?;
                let w = self.wr(i, r, "old")?;
                self.emit(&format!("{{ {p} u64 old = (u64)__atomic_exchange_n(p, ({t})({v}), {SC}); {w} }}"));
            }
            M::Xadd => {
                let b = self.rd(i, 1, bits)?;
                let w = self.wr(i, 1, "a")?;
                self.emit(&format!(
                    "{{ {p} u64 b = {b}; u64 a = (u64)__atomic_fetch_add(p, ({t})b, {SC}); u64 r = (a + b) & {mk}; {w} F.k = FK_ADD; F.sz = {bits}; F.a = a; F.b = b; F.r = r; F.c = 0; }}"
                ));
            }
            M::Cmpxchg => {
                let src = self.rd(i, 1, bits)?;
                let ax = match bits {
                    8 => "(u8)r0".to_string(),
                    _ => format!("(u{bits})r0"),
                };
                let setax = match bits {
                    8 => "r0 = (r0 & ~0xffull) | old;".to_string(),
                    16 => "r0 = (r0 & ~0xffffull) | old;".to_string(),
                    _ => "r0 = old;".to_string(),
                };
                self.emit(&format!(
                    "{{ {p} {t} exp = {ax}; {t} want = ({t})({src}); int ok = __atomic_compare_exchange_n(p, &exp, want, 0, {SC}, {SC}); u64 old = exp; u64 a = {ax}; u64 r = (a - old) & {mk}; {setax} F.k = FK_SUB; F.sz = {bits}; F.a = a; F.b = old; F.r = r; F.c = 0; }}"
                ));
            }
            _ => unreachable!(),
        }
        Ok(true)
    }

    /// Instruction with a `lock`/`rep` prefix: what the title uses, or an error.
    pub(super) fn prefixed(&mut self, i: &Instruction) -> R<()> {
        use iced_x86::Mnemonic as M;
        if i.has_lock_prefix() {
            if self.atomic(i)? {
                return Ok(());
            }
            if matches!(i.mnemonic(), M::Bts | M::Btr | M::Btc) && self.bit_test(i)? {
                return Ok(());
            }
            if i.mnemonic() == M::Cmpxchg16b {
                return self.cmpxchg16b(i);
            }
            return Err(format!("lock {:?}", i.mnemonic()));
        }
        self.string_op(i)
    }

    pub(super) fn cmpxchg16b(&mut self, i: &Instruction) -> R<()> {
        let a = self.addr(i)?;
        self.emit(&format!(
            "{{ unsigned __int128 *p = (unsigned __int128*)({a}); unsigned __int128 exp = ((unsigned __int128)r2 << 64) | r0; unsigned __int128 want = ((unsigned __int128)r1 << 64) | r3; int ok = __atomic_compare_exchange_n(p, &exp, want, 0, {SC}, {SC}); if (!ok) {{ r0 = (u64)exp; r2 = (u64)(exp >> 64); }} fl_set_zf(&F, ok); }}"
        ));
        Ok(())
    }

    /// `rep movs/stos/lods` and `repe/repne cmps/scas` (direction flag assumed clear).
    fn string_op(&mut self, i: &Instruction) -> R<()> {
        let (name, sz) = match i.mnemonic() {
            M::Movsb => ("movs", 1),
            M::Movsw => ("movs", 2),
            M::Movsd if i.op_count() == 2 && i.op0_kind() == OpKind::MemoryESRDI => ("movs", 4),
            M::Movsq => ("movs", 8),
            M::Stosb => ("stos", 1),
            M::Stosw => ("stos", 2),
            M::Stosd => ("stos", 4),
            M::Stosq => ("stos", 8),
            M::Lodsb => ("lods", 1),
            M::Lodsw => ("lods", 2),
            M::Lodsd => ("lods", 4),
            M::Lodsq => ("lods", 8),
            M::Cmpsb => ("cmps", 1),
            M::Cmpsw => ("cmps", 2),
            M::Cmpsd => ("cmps", 4),
            M::Cmpsq => ("cmps", 8),
            M::Scasb => ("scas", 1),
            M::Scasw => ("scas", 2),
            M::Scasd => ("scas", 4),
            M::Scasq => ("scas", 8),
            m => return Err(format!("{m:?} with lock/rep prefix")),
        };
        let bits = sz * 8;
        let t = format!("u{bits}");
        let ax = format!("({t})r0");
        let step = format!("r6 += {sz}; r7 += {sz};");
        let one = match name {
            "movs" => format!("{t} v; memcpy(&v, (u8*)r6, {sz}); memcpy((u8*)r7, &v, {sz}); {step}"),
            "stos" => format!("{t} v = {ax}; memcpy((u8*)r7, &v, {sz}); r7 += {sz};"),
            "lods" => format!(
                "{t} v; memcpy(&v, (u8*)r6, {sz}); {} r6 += {sz};",
                match sz {
                    1 => "r0 = (r0 & ~0xffull) | v;",
                    2 => "r0 = (r0 & ~0xffffull) | v;",
                    _ => "r0 = v;",
                }
            ),
            "cmps" => format!(
                "{t} x, y; memcpy(&x, (u8*)r6, {sz}); memcpy(&y, (u8*)r7, {sz}); F.k = FK_SUB; F.sz = {bits}; F.a = x; F.b = y; F.r = (u64)(({t})(x - y)); F.c = 0; {step}"
            ),
            _ => format!(
                "{t} y; memcpy(&y, (u8*)r7, {sz}); {t} x = {ax}; F.k = FK_SUB; F.sz = {bits}; F.a = x; F.b = y; F.r = (u64)(({t})(x - y)); F.c = 0; r7 += {sz};"
            ),
        };
        let cond = match name {
            "cmps" | "scas" => {
                if i.has_repne_prefix() {
                    "if (fl_zf(&F)) break;"
                } else {
                    "if (!fl_zf(&F)) break;"
                }
            }
            _ => "",
        };
        if i.has_rep_prefix() || i.has_repe_prefix() || i.has_repne_prefix() {
            self.emit(&format!("while (r1) {{ {{ {one} }} r1--; {cond} }}"));
        } else {
            self.emit(&format!("{{ {one} }}"));
        }
        Ok(())
    }

    /// bt / bts / btr / btc, including the bit-string addressing of a memory operand with a register offset.
    pub(super) fn bit_test(&mut self, i: &Instruction) -> R<bool> {
        let m = i.mnemonic();
        let (_, bits) = Self::ity(i)?;
        let lg = bits.trailing_zeros();
        let imm = i.op1_kind() != OpKind::Register;
        let off = if imm {
            format!("{}ull", i.immediate(1) as u8)
        } else {
            format!("(u64)(s64)sx({}, {bits})", self.rd(i, 1, bits)?)
        };
        let (apply, atomic) = match m {
            M::Bt => ("v", ""),
            M::Bts => ("v | bm", "or"),
            M::Btr => ("v & ~bm", "and"),
            _ => ("v ^ bm", "xor"),
        };
        if i.op0_kind() == OpKind::Register {
            let a = self.rd(i, 0, bits)?;
            let w = if m == M::Bt { String::new() } else { self.wr(i, 0, "nv")? };
            self.emit(&format!(
                "{{ u64 bit = {off} & {}; u64 v = {a}; u64 bm = 1ull << bit; u64 nv = ({apply}) & {}; {w} fl_set_cf(&F, (v >> bit) & 1); }}",
                bits - 1,
                mask(bits)
            ));
            return Ok(true);
        }
        let t = format!("u{bits}");
        let a = self.addr(i)?;
        let loc = if imm {
            format!("u64 bit = {off} & {}; {t} *p = ({t}*)({a});", bits - 1)
        } else {
            format!(
                "s64 o = (s64){off}; u64 bit = (u64)o & {}; {t} *p = ({t}*)((u8*)({a}) + (o >> {lg}) * {});",
                bits - 1,
                bits / 8
            )
        };
        let body = if m == M::Bt {
            format!("{t} v; memcpy(&v, p, {}); u64 old = v;", bits / 8)
        } else if i.has_lock_prefix() {
            format!("u64 bm = 1ull << bit; u64 old = (u64)__atomic_fetch_{atomic}(p, ({t})({}), {SC});", if m == M::Btr { "~bm" } else { "bm" })
        } else {
            format!("u64 bm = 1ull << bit; {t} v; memcpy(&v, p, {}); u64 old = v; {t} nv = ({t})({apply}); memcpy(p, &nv, {});", bits / 8, bits / 8)
        };
        self.emit(&format!("{{ {loc} {body} fl_set_cf(&F, (old >> bit) & 1); }}"));
        Ok(true)
    }

    pub(super) fn bits(&mut self, i: &Instruction) -> R<bool> {
        let m = i.mnemonic();
        match m {
            M::Xchg if i.op0_kind() == OpKind::Memory || i.op1_kind() == OpKind::Memory => return self.atomic(i),
            M::Xadd | M::Cmpxchg if i.op0_kind() == OpKind::Memory => return self.atomic(i),
            M::Cmpxchg if i.op0_kind() == OpKind::Register => {
                let (_, bits) = Self::ity(i)?;
                let (d, s) = (self.rd(i, 0, bits)?, self.rd(i, 1, bits)?);
                let (wd, wa) = (self.wr(i, 0, &s)?, self.wr_ax(bits)?);
                let wd2 = self.wr(i, 0, "d")?;
                let mk = mask(bits);
                self.emit(&format!(
                    "{{ u64 a = r0 & {mk}; u64 d = {d}; u64 r = (a - d) & {mk}; u64 old = d; {wa} if (a == d) {{ {wd} }} else {{ {wd2} }} F.k = FK_SUB; F.sz = {bits}; F.a = a; F.b = d; F.r = r; F.c = 0; }}"
                ));
            }
            M::Xadd => {
                let (_, bits) = Self::ity(i)?;
                let (a, b) = (self.rd(i, 0, bits)?, self.rd(i, 1, bits)?);
                let (w0, w1) = (self.wr(i, 0, "r")?, self.wr(i, 1, "a")?);
                self.emit(&format!(
                    "{{ u64 a = {a}, b = {b}; u64 r = (a + b) & {}; {w1} {w0} F.k = FK_ADD; F.sz = {bits}; F.a = a; F.b = b; F.r = r; F.c = 0; }}",
                    mask(bits)
                ));
            }
            M::Bt | M::Bts | M::Btr | M::Btc => return self.bit_test(i),
            M::Cmpxchg16b => self.cmpxchg16b(i)?,
            M::Andn | M::Bextr | M::Blsi | M::Blsr | M::Blsmsk | M::Tzcnt | M::Lzcnt | M::Bzhi | M::Pdep | M::Pext
            | M::Shlx | M::Shrx | M::Sarx | M::Rorx | M::Mulx => self.bmi(i)?,
            M::Movbe => {
                let bits = Self::bits0(i)?;
                let sw = match bits {
                    16 => "__builtin_bswap16",
                    32 => "__builtin_bswap32",
                    _ => "__builtin_bswap64",
                };
                let v = self.rd(i, 1, bits)?;
                let w = self.wr(i, 0, &format!("(u64){sw}((u{bits})({v}))"))?;
                self.emit(&w);
            }
            M::Bswap => {
                let bits = Self::bits0(i)?;
                let v = self.rd(i, 0, bits)?;
                let sw = if bits == 32 { "__builtin_bswap32" } else { "__builtin_bswap64" };
                let w = self.wr(i, 0, &format!("(u64){sw}((u{bits})({v}))"))?;
                self.emit(&w);
            }
            M::Adcx | M::Adox => {
                let bits = Self::bits0(i)?;
                let (a, b) = (self.rd(i, 0, bits)?, self.rd(i, 1, bits)?);
                let w = self.wr(i, 0, "r")?;
                let (rd, set) = if m == M::Adcx { ("fl_cf", "fl_set_cf") } else { ("fl_of", "fl_set_of") };
                self.emit(&format!(
                    "{{ u64 a = {a}, b = {b}, cin = {rd}(&F); u64 r = (a + b + cin) & {m_}; int c = r < a || (cin && r == a); {w} {set}(&F, c); }}",
                    m_ = mask(bits)
                ));
            }
            M::Shld | M::Shrd => {
                let bits = Self::bits0(i)?;
                let (d, s) = (self.rd(i, 0, bits)?, self.rd(i, 1, bits)?);
                let cnt = if i.op2_kind() == OpKind::Register {
                    "(r1 & 255)".to_string()
                } else {
                    format!("{}", i.immediate(2) as u8)
                };
                let w = self.wr(i, 0, "r")?;
                let mk = mask(bits);
                let cmask = if bits == 64 { 63 } else { 31 };
                let body = if m == M::Shld {
                    format!("r = ((d << c) | (s >> ({bits} - c))) & {mk}; int cf = (d >> ({bits} - c)) & 1;")
                } else {
                    format!("r = ((d >> c) | (s << ({bits} - c))) & {mk}; int cf = (d >> (c - 1)) & 1;")
                };
                self.emit(&format!(
                    "{{ u64 d = {d}, s = {s}; u64 c = (u64)({cnt}) & {cmask}; u64 r = d; if (c && c < {bits}) {{ {body} int of = (int)(((r ^ d) >> {}) & 1); fl_set_raw(&F, {bits}, r, cf, of); }} {w} }}",
                    bits - 1
                ));
            }
            M::Cwd => self.emit("r2 = (r2 & ~0xffffull) | ((u64)(u16)((s16)r0 >> 15));"),
            M::Pushfq => self.emit("{ r4 -= 8; st64((u8*)r4, fl_pack(&F)); }"),
            M::Popfq => self.emit("{ u64 v = ld64((u8*)r4); r4 += 8; fl_unpack(&F, v); }"),
            M::Lahf => self.emit("r0 = (r0 & ~0xff00ull) | (((fl_cf(&F) | 2 | fl_pf(&F) << 2 | fl_zf(&F) << 6 | fl_sf(&F) << 7) & 0xffull) << 8);"),
            M::Sahf => self.emit("{ u64 h = (r0 >> 8) & 0xff; u64 of = fl_of(&F); F.k = FK_RAW; F.c = (h & 1) | (h >> 6 & 1) << 1 | (h >> 7 & 1) << 2 | of << 3 | (h >> 2 & 1) << 4; }"),
            M::Cld => {}
            M::Prefetcht0 | M::Prefetcht1 | M::Prefetcht2 | M::Prefetchnta | M::Prefetchw | M::Clwb | M::Clflush | M::Clflushopt => {}
            M::Cpuid => self.emit("{ u32 a, b, c_, d; rt_cpuid((u32)r0, (u32)r1, &a, &b, &c_, &d); r0 = a; r3 = b; r1 = c_; r2 = d; }"),
            M::Xgetbv => self.emit("{ u64 v = rt_xgetbv((u32)r1); r0 = (u32)v; r2 = v >> 32; }"),
            M::Rdtsc => self.emit("{ u64 v = rt_rdtsc(); r0 = (u32)v; r2 = v >> 32; }"),
            M::Vstmxcsr | M::Stmxcsr => {
                let a = self.addr(i)?;
                self.emit(&format!("st32({a}, 0x1f80);"));
            }
            M::Vldmxcsr | M::Ldmxcsr => {}
            M::Div | M::Idiv if Self::bits0(i)? <= 16 => {
                let bits = Self::bits0(i)?;
                let s = self.rd(i, 0, bits)?;
                let sgn = m == M::Idiv;
                let body = match (bits, sgn) {
                    (8, false) => format!("u64 n = r0 & 0xffff; u64 d = {s}; if (!d || n / d > 0xff) rt_trap(\"#DE\"); r0 = (r0 & ~0xffffull) | ((n % d) << 8) | (n / d);"),
                    (8, true) => format!("s64 n = (s16)r0; s64 d = (s8){s}; if (!d || n / d > 127 || n / d < -128) rt_trap(\"#DE\"); r0 = (r0 & ~0xffffull) | (((u64)(n % d) & 0xff) << 8) | ((u64)(n / d) & 0xff);"),
                    (_, false) => format!("u64 n = ((r2 & 0xffff) << 16) | (r0 & 0xffff); u64 d = {s}; if (!d || n / d > 0xffff) rt_trap(\"#DE\"); r0 = (r0 & ~0xffffull) | (n / d); r2 = (r2 & ~0xffffull) | (n % d);"),
                    (_, true) => format!("s64 n = (s32)(((r2 & 0xffff) << 16) | (r0 & 0xffff)); s64 d = (s16){s}; if (!d || n / d > 32767 || n / d < -32768) rt_trap(\"#DE\"); r0 = (r0 & ~0xffffull) | ((u64)(n / d) & 0xffff); r2 = (r2 & ~0xffffull) | ((u64)(n % d) & 0xffff);"),
                };
                self.emit(&format!("{{ {body} }}"));
            }
            _ => return Ok(false),
        }
        Ok(true)
    }

    fn wr_ax(&self, bits: u32) -> R<String> {
        Ok(match bits {
            8 => "r0 = (r0 & ~0xffull) | (old & 0xff);".to_string(),
            16 => "r0 = (r0 & ~0xffffull) | (old & 0xffff);".to_string(),
            32 => "r0 = (u32)old;".to_string(),
            _ => "r0 = old;".to_string(),
        })
    }

    fn bmi(&mut self, i: &Instruction) -> R<()> {
        use iced_x86::Mnemonic as M;
        let m = i.mnemonic();
        let bits = Self::bits0(i)?;
        let mk = mask(bits);
        let w = self.wr(i, 0, "r")?;
        let s1 = self.rd(i, 1, bits)?;
        let st = match m {
            M::Andn => {
                let s2 = self.rd(i, 2, bits)?;
                format!("u64 r = ~{s1} & {s2} & {mk}; {w} fl_set_raw(&F, {bits}, r, 0, 0);")
            }
            M::Bextr => {
                let c = self.rd(i, 2, bits)?;
                format!("u64 s = {s1}, c = {c}; u64 st = c & 0xff, len = (c >> 8) & 0xff; u64 r = st >= {bits} ? 0 : (s >> st); if (len < {bits}) r &= (1ull << len) - 1; r &= {mk}; {w} fl_set_raw(&F, {bits}, r, 0, 0);")
            }
            M::Blsi => format!("u64 s = {s1}; u64 r = s & (0 - s) & {mk}; {w} fl_set_raw(&F, {bits}, r, s != 0, 0);"),
            M::Blsr => format!("u64 s = {s1}; u64 r = s & (s - 1) & {mk}; {w} fl_set_raw(&F, {bits}, r, s == 0, 0);"),
            M::Blsmsk => format!("u64 s = {s1}; u64 r = (s ^ (s - 1)) & {mk}; {w} fl_set_raw(&F, {bits}, r, s == 0, 0); fl_set_zf(&F, 0);"),
            M::Tzcnt => format!("u64 s = {s1}; u64 r = s ? (u64)__builtin_ctzll(s) : {bits}; {w} F.k = FK_RAW; F.c = (u64)(s == 0) | (u64)(r == 0) << 1;"),
            M::Lzcnt => format!("u64 s = {s1}; u64 r = s ? (u64)__builtin_clzll(s) - {} : {bits}; {w} F.k = FK_RAW; F.c = (u64)(s == 0) | (u64)(r == 0) << 1;", 64 - bits),
            M::Bzhi => {
                let c = self.rd(i, 2, bits)?;
                format!("u64 s = {s1}; u64 n = ({c}) & 0xff; u64 r = n >= {bits} ? s : (s & ((1ull << n) - 1)); {w} fl_set_raw(&F, {bits}, r, n >= {bits}, 0);")
            }
            M::Pdep | M::Pext => {
                let s2 = self.rd(i, 2, bits)?;
                let f = if m == M::Pdep { "pdep64" } else { "pext64" };
                format!("u64 r = {f}({s1}, {s2}) & {mk}; {w}")
            }
            M::Shlx | M::Shrx | M::Sarx => {
                let c = self.rd(i, 2, bits)?;
                let n = bits - 1;
                let e = match m {
                    M::Shlx => format!("({s1} << ({c} & {n})) & {mk}"),
                    M::Shrx => format!("{s1} >> ({c} & {n})"),
                    _ => format!("(u64)((s64)sx({s1}, {bits}) >> ({c} & {n})) & {mk}"),
                };
                format!("u64 r = {e}; {w}")
            }
            M::Rorx => {
                let k = (i.immediate(2) as u32) & (bits - 1);
                if k == 0 {
                    format!("u64 r = {s1}; {w}")
                } else {
                    format!("u64 s = {s1}; u64 r = ((s >> {k}) | (s << ({bits} - {k}))) & {mk}; {w}")
                }
            }
            M::Mulx => {
                // mulx hi, lo, src: lo = op1, hi = op0 (hi wins when they are the same register).
                let s2 = self.rd(i, 2, bits)?;
                let wlo = self.wr(i, 1, "lo")?;
                let whi = self.wr(i, 0, "hi")?;
                let (ty, d) = if bits == 64 { ("unsigned __int128", "r2") } else { ("u64", "(r2 & 0xffffffffull)") };
                return {
                    self.emit(&format!(
                        "{{ {ty} p = ({ty}){d} * ({ty})({s2}); u64 lo = (u64)p & {mk}; u64 hi = (u64)(p >> {bits}) & {mk}; {wlo} {whi} }}"
                    ));
                    Ok(())
                };
            }
            _ => unreachable!(),
        };
        self.emit(&format!("{{ {st} }}"));
        let _ = gpr;
        Ok(())
    }
}
