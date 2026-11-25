//! VEX-encoded instructions (AVX, AVX2, FMA, F16C). A ymm register is two 128-bit
//! halves: `x<n>` (low) and `xh<n>` (high), so every 256-bit instruction is the same
//! helper applied to both halves; only the cross-lane ones (`vperm*`, `vinsert*`,
//! `vextract*`) mix halves, through `Y`. Why halves and not clang vector extensions:
//! NEON is 128 bits wide, so a 256-bit vector type is split into two q registers by
//! clang anyway; with halves the lowering is explicit, the existing `X128` lane
//! model and `Cpu` layout are reused, and each helper is checked in isolation.
//! Every VEX write to an xmm register zeroes bits 128..255.

use iced_x86::{Mnemonic as M, Register};

use super::{Instruction, L, OpKind, R, gpr, mask};

fn vreg(r: Register) -> R<(usize, bool)> {
    if r.is_xmm() {
        Ok((r.number(), false))
    } else if r.is_ymm() {
        Ok((r.number(), true))
    } else {
        Err(format!("vector register {r:?}"))
    }
}

/// `h_<mnemonic>` in the runtime header.
fn h(m: M) -> String {
    format!("h_{}", format!("{m:?}").to_lowercase())
}

fn imm8(i: &Instruction) -> u32 {
    i.immediate(i.op_count() - 1) as u8 as u32
}

impl L<'_> {
    /// Operand `n` as (low, high) 128-bit expressions; `high` is only meaningful when `wide`.
    fn vsrc(&self, i: &Instruction, n: u32, wide: bool) -> R<(String, String)> {
        match i.op_kind(n) {
            OpKind::Register => {
                let (k, _) = vreg(i.op_register(n))?;
                Ok((format!("x{k}"), format!("xh{k}")))
            }
            OpKind::Memory => {
                let a = self.addr(i)?;
                let hi = if wide {
                    format!("ld128((u8*)({a}) + 16)")
                } else {
                    String::new()
                };
                Ok((format!("ld128({a})"), hi))
            }
            k => Err(format!("operand kind {k:?}")),
        }
    }

    /// A scalar source as an X128 with the value in lane 0: a register, or memory of `bytes`.
    fn vscalar(&self, i: &Instruction, n: u32, bytes: u32) -> R<String> {
        match i.op_kind(n) {
            OpKind::Register => Ok(format!("x{}", vreg(i.op_register(n))?.0)),
            OpKind::Memory => Ok(format!("ldx{}({})", bytes * 8, self.addr(i)?)),
            k => Err(format!("operand kind {k:?}")),
        }
    }

    fn vset(d: usize, wide: bool, lo: &str, hi: &str) -> String {
        if wide {
            format!("x{d} = {lo}; xh{d} = {hi};")
        } else {
            format!("x{d} = {lo}; xh{d} = xz();")
        }
    }

    /// dst = helper(srcs.., extra..) on each half.
    fn lane(
        &mut self,
        i: &Instruction,
        helper: &str,
        srcs: &[u32],
        extra: &[String],
        hi_extra: Option<&[String]>,
    ) -> R<()> {
        let (d, wide) = vreg(i.op0_register())?;
        let mut lo = vec![];
        let mut hi = vec![];
        for &n in srcs {
            let (l, h_) = self.vsrc(i, n, wide)?;
            lo.push(l);
            hi.push(h_);
        }
        lo.extend(extra.iter().cloned());
        hi.extend(hi_extra.unwrap_or(extra).iter().cloned());
        let s = Self::vset(
            d,
            wide,
            &format!("{helper}({})", lo.join(", ")),
            &format!("{helper}({})", hi.join(", ")),
        );
        self.emit(&s);
        Ok(())
    }

    /// Scalar op: dst = helper(src1, src2[, imm]), result in lane 0, upper from src1, ymm upper zero.
    fn scalar3(&mut self, i: &Instruction, helper: &str, bytes: u32, extra: &[String]) -> R<()> {
        let (d, _) = vreg(i.op0_register())?;
        let a = self.vscalar(i, 1, 16)?;
        let b = self.vscalar(i, 2, bytes)?;
        let mut args = vec![a, b];
        args.extend(extra.iter().cloned());
        self.emit(&format!("x{d} = {helper}({}); xh{d} = xz();", args.join(", ")));
        Ok(())
    }

    fn count_arg(&self, i: &Instruction, n: u32) -> R<String> {
        Ok(match i.op_kind(n) {
            OpKind::Register | OpKind::Memory => self.vsrc(i, n, false)?.0,
            _ => format!("(X128){{ .q = {{ {:#x}ull, 0 }} }}", i.immediate(n) as u8),
        })
    }

    fn fma(&mut self, i: &Instruction) -> R<()> {
        let name = format!("{:?}", i.mnemonic()).to_lowercase(); // vfmadd132ps
        let body = &name[2..];
        let split = body.find(|c: char| c.is_ascii_digit()).ok_or("fma name")?;
        let (op, rest) = body.split_at(split);
        let (order, ty) = rest.split_at(3);
        let scalar = ty.starts_with('s') && ty != "ps";
        let dbl = ty.ends_with('d');
        let (d, wide) = vreg(i.op0_register())?;
        // dst = X, op1 = Y, op2 = Z.
        let pick = |x: &str, y: &str, z: &str| match order {
            "132" => (x.to_string(), z.to_string(), y.to_string()),
            "213" => (y.to_string(), x.to_string(), z.to_string()),
            _ => (y.to_string(), z.to_string(), x.to_string()),
        };
        let bytes = if dbl { 8 } else { 4 };
        if scalar {
            let (a, b, c) = pick(
                &format!("x{d}"),
                &self.vscalar(i, 1, 16)?,
                &self.vscalar(i, 2, bytes)?,
            );
            let (ng, sc) = (op.starts_with("nm") as u32, op.ends_with("sub") as u32);
            self.emit(&format!(
                "x{d} = h_fma_s{}(x{d}, {a}, {b}, {c}, {ng}, {sc}); xh{d} = xz();",
                if dbl { 'd' } else { 's' }
            ));
            return Ok(());
        }
        let p = if dbl { "pd" } else { "ps" };
        let z = self.vsrc(i, 2, wide)?;
        let y = self.vsrc(i, 1, wide)?;
        let (alo, blo, clo) = pick(&format!("x{d}"), &y.0, &z.0);
        let (ahi, bhi, chi) = pick(&format!("xh{d}"), &y.1, &z.1);
        let call = |a: &str, b: &str, c: &str| -> R<String> {
            Ok(if op == "maddsub" || op == "msubadd" {
                format!("h_fmas_{p}({a}, {b}, {c}, {})", (op == "maddsub") as u32)
            } else {
                let (ng, sc) = (op.starts_with("nm") as u32, op.ends_with("sub") as u32);
                format!("h_fma_{p}({a}, {b}, {c}, {ng}, {sc})")
            })
        };
        let s = Self::vset(d, wide, &call(&alo, &blo, &clo)?, &call(&ahi, &bhi, &chi)?);
        self.emit(&s);
        Ok(())
    }

    pub(super) fn vex(&mut self, i: &Instruction) -> R<bool> {
        if i.code().encoding() != iced_x86::EncodingKind::VEX {
            return Ok(false);
        }
        let m = i.mnemonic();
        let hn = h(m);
        let imm = || imm8(i);
        match m {
            // ---- three-operand, per 128-bit half ----
            M::Vaddps | M::Vaddpd | M::Vsubps | M::Vsubpd | M::Vmulps | M::Vmulpd | M::Vdivps
            | M::Vdivpd | M::Vminps | M::Vminpd | M::Vmaxps | M::Vmaxpd | M::Vxorps | M::Vxorpd
            | M::Vandps | M::Vandpd | M::Vorps | M::Vorpd | M::Vandnps | M::Vandnpd | M::Vpxor
            | M::Vpand | M::Vpor | M::Vpandn | M::Vhaddps | M::Vhaddpd | M::Vhsubps | M::Vhsubpd
            | M::Vaddsubps | M::Vaddsubpd | M::Vpaddb | M::Vpaddw | M::Vpaddd | M::Vpaddq
            | M::Vpsubb | M::Vpsubw | M::Vpsubd | M::Vpsubq | M::Vpaddsb | M::Vpaddsw
            | M::Vpaddusb | M::Vpaddusw | M::Vpsubsb | M::Vpsubsw | M::Vpsubusb | M::Vpsubusw
            | M::Vpmullw | M::Vpmulld | M::Vpmulhw | M::Vpmulhuw | M::Vpmulhrsw | M::Vpmuludq
            | M::Vpmuldq | M::Vpmaddwd | M::Vpmaddubsw | M::Vpavgb | M::Vpavgw | M::Vpminsb
            | M::Vpminsw | M::Vpminsd | M::Vpminub | M::Vpminuw | M::Vpminud | M::Vpmaxsb
            | M::Vpmaxsw | M::Vpmaxsd | M::Vpmaxub | M::Vpmaxuw | M::Vpmaxud | M::Vpcmpeqb
            | M::Vpcmpeqw | M::Vpcmpeqd | M::Vpcmpeqq | M::Vpcmpgtb | M::Vpcmpgtw | M::Vpcmpgtd
            | M::Vpcmpgtq | M::Vpsignb | M::Vpsignw | M::Vpsignd | M::Vpsadbw | M::Vphaddw
            | M::Vphaddd | M::Vphsubw | M::Vphsubd | M::Vphaddsw | M::Vphsubsw | M::Vpacksswb
            | M::Vpackuswb | M::Vpackssdw | M::Vpackusdw | M::Vunpcklps | M::Vunpckhps
            | M::Vunpcklpd | M::Vunpckhpd | M::Vpunpcklbw | M::Vpunpckhbw | M::Vpunpcklwd
            | M::Vpunpckhwd | M::Vpunpckldq | M::Vpunpckhdq | M::Vpunpcklqdq | M::Vpunpckhqdq
            | M::Vpshufb | M::Vpsllvd | M::Vpsrlvd | M::Vpsravd | M::Vpsllvq | M::Vpsrlvq
            | M::Vmovlhps | M::Vmovhlps
                if i.op_count() == 3 && i.op_kind(1) == OpKind::Register =>
            {
                self.lane(i, &hn, &[1, 2], &[], None)?
            }
            // ---- three operands plus an immediate ----
            M::Vshufps | M::Vpalignr | M::Vcmpps | M::Vcmppd | M::Vpblendw | M::Vblendps
            | M::Vpblendd | M::Vblendpd | M::Vshufpd => {
                let v = imm();
                let (helper, lo, hi) = match m {
                    M::Vblendps | M::Vpblendd => ("h_vblendps".to_string(), v & 15, v >> 4),
                    M::Vblendpd => (hn, v & 3, v >> 2),
                    M::Vshufpd => (hn, v & 3, v >> 2),
                    _ => (hn, v, v),
                };
                self.lane(i, &helper, &[1, 2], &[lo.to_string()], Some(&[hi.to_string()]))?
            }
            // ---- one source ----
            M::Vsqrtps | M::Vsqrtpd | M::Vrcpps | M::Vrsqrtps | M::Vpabsb | M::Vpabsw
            | M::Vpabsd | M::Vcvtdq2ps | M::Vcvttps2dq | M::Vcvtps2dq | M::Vmovshdup
            | M::Vmovsldup | M::Vmovddup
                if i.op_count() == 2 =>
            {
                self.lane(i, &hn, &[1], &[], None)?
            }
            M::Vroundps | M::Vroundpd | M::Vpshufd | M::Vpshufhw | M::Vpshuflw => {
                let v = imm().to_string();
                self.lane(i, &hn, &[1], &[v], None)?
            }
            M::Vpermilps | M::Vpermilpd => {
                let dbl = m == M::Vpermilpd;
                if i.op_kind(2) == OpKind::Register || i.op_kind(2) == OpKind::Memory {
                    self.lane(i, &format!("{hn}v"), &[1, 2], &[], None)?
                } else {
                    let v = imm();
                    let (lo, hi) = if dbl { (v & 3, v >> 2) } else { (v, v) };
                    self.lane(i, &hn, &[1], &[lo.to_string()], Some(&[hi.to_string()]))?
                }
            }
            M::Vblendvps | M::Vblendvpd | M::Vpblendvb => {
                let (d, wide) = vreg(i.op0_register())?;
                let a = self.vsrc(i, 1, wide)?;
                let b = self.vsrc(i, 2, wide)?;
                let k = vreg(i.op3_register())?.0;
                let s = Self::vset(
                    d,
                    wide,
                    &format!("{hn}({}, {}, x{k})", a.0, b.0),
                    &format!("{hn}({}, {}, xh{k})", a.1, b.1),
                );
                self.emit(&s);
            }
            M::Vinsertps => {
                let v = imm();
                let (d, _) = vreg(i.op0_register())?;
                let a = self.vsrc(i, 1, false)?.0;
                let (b, v) = if i.op_kind(2) == OpKind::Memory {
                    (self.vscalar(i, 2, 4)?, v & 0x3f)
                } else {
                    (self.vscalar(i, 2, 16)?, v)
                };
                self.emit(&format!("x{d} = {hn}({a}, {b}, {v}); xh{d} = xz();"));
            }
            // ---- scalar float ----
            M::Vaddss | M::Vsubss | M::Vmulss | M::Vdivss | M::Vminss | M::Vmaxss | M::Vsqrtss
            | M::Vrcpss | M::Vrsqrtss => self.scalar3(i, &hn, 4, &[])?,
            M::Vaddsd | M::Vsubsd | M::Vmulsd | M::Vdivsd | M::Vminsd | M::Vmaxsd | M::Vsqrtsd => {
                self.scalar3(i, &hn, 8, &[])?
            }
            M::Vcmpss | M::Vroundss => self.scalar3(i, &hn, 4, &[imm().to_string()])?,
            M::Vcmpsd | M::Vroundsd => self.scalar3(i, &hn, 8, &[imm().to_string()])?,
            M::Vcvtss2sd => {
                let (d, _) = vreg(i.op0_register())?;
                let (a, b) = (self.vscalar(i, 1, 16)?, self.vscalar(i, 2, 4)?);
                self.emit(&format!(
                    "{{ X128 t = {a}; t.g[0] = (double)({b}).f[0]; x{d} = t; xh{d} = xz(); }}"
                ));
            }
            M::Vcvtsd2ss => {
                let (d, _) = vreg(i.op0_register())?;
                let (a, b) = (self.vscalar(i, 1, 16)?, self.vscalar(i, 2, 8)?);
                self.emit(&format!(
                    "{{ X128 t = {a}; t.f[0] = (float)({b}).g[0]; x{d} = t; xh{d} = xz(); }}"
                ));
            }
            M::Vcvtsi2ss | M::Vcvtsi2sd => {
                let (d, _) = vreg(i.op0_register())?;
                let a = self.vscalar(i, 1, 16)?;
                let bits = if i.op2_kind() == OpKind::Register {
                    gpr(i.op2_register())?.1
                } else {
                    Self::mem_bits(i)
                };
                let v = self.rd(i, 2, bits)?;
                let (l, t) = if m == M::Vcvtsi2sd {
                    ("g", "double")
                } else {
                    ("f", "float")
                };
                self.emit(&format!(
                    "{{ X128 t = {a}; t.{l}[0] = ({t})(s64)sx({v}, {bits}); x{d} = t; xh{d} = xz(); }}"
                ));
            }
            M::Vcvttss2si | M::Vcvttsd2si | M::Vcvtss2si | M::Vcvtsd2si => {
                let dbl = matches!(m, M::Vcvttsd2si | M::Vcvtsd2si);
                let bits = gpr(i.op0_register())?.1;
                let s = self.vscalar(i, 1, if dbl { 8 } else { 4 })?;
                let f = if matches!(m, M::Vcvttss2si | M::Vcvttsd2si) {
                    format!("cvtt{bits}")
                } else {
                    format!("cvt{bits}")
                };
                let v = format!("({s}).{}[0]", if dbl { "g" } else { "f" });
                let w = self.wr(i, 0, &format!("(u64){f}((double){v})"))?;
                self.emit(&w);
            }
            M::Vucomiss | M::Vcomiss | M::Vucomisd | M::Vcomisd => {
                let dbl = matches!(m, M::Vucomisd | M::Vcomisd);
                let l = if dbl { "g" } else { "f" };
                let sz = if dbl { 8 } else { 4 };
                let (a, b) = (self.vscalar(i, 0, sz)?, self.vscalar(i, 1, sz)?);
                self.emit(&format!(
                    "fl_set_ucomi(&F, (double)({a}).{l}[0], (double)({b}).{l}[0]);"
                ));
            }
            // ---- FMA ----
            _ if format!("{m:?}").starts_with("Vf")
                && (format!("{m:?}").contains("132")
                    || format!("{m:?}").contains("213")
                    || format!("{m:?}").contains("231")) =>
            {
                self.fma(i)?
            }
            // ---- moves ----
            M::Vmovaps | M::Vmovups | M::Vmovapd | M::Vmovupd | M::Vmovdqa | M::Vmovdqu
            | M::Vlddqu | M::Vmovntps | M::Vmovntpd | M::Vmovntdq | M::Vmovntdqa => {
                if i.op0_kind() == OpKind::Memory {
                    let (k, wide) = vreg(i.op1_register())?;
                    let a = self.addr(i)?;
                    self.emit(&format!("st128({a}, x{k});"));
                    if wide {
                        self.emit(&format!("st128((u8*)({a}) + 16, xh{k});"));
                    }
                } else {
                    let (d, wide) = vreg(i.op0_register())?;
                    let (lo, hi) = self.vsrc(i, 1, wide)?;
                    let s = Self::vset(d, wide, &lo, &hi);
                    self.emit(&s);
                }
            }
            M::Vmovss | M::Vmovsd => {
                let dbl = m == M::Vmovsd;
                let (bytes, l) = if dbl { (8, "q") } else { (4, "d") };
                if i.op0_kind() == OpKind::Memory {
                    let k = vreg(i.op1_register())?.0;
                    let a = self.addr(i)?;
                    self.emit(&format!("st{}({a}, x{k}.{l}[0]);", bytes * 8));
                } else {
                    let d = vreg(i.op0_register())?.0;
                    if i.op1_kind() == OpKind::Memory {
                        let a = self.addr(i)?;
                        self.emit(&format!("x{d} = ldx{}({a}); xh{d} = xz();", bytes * 8));
                    } else {
                        let (a, b) = (self.vscalar(i, 1, 16)?, self.vscalar(i, 2, 16)?);
                        self.emit(&format!(
                            "{{ X128 t = {a}; t.{l}[0] = ({b}).{l}[0]; x{d} = t; xh{d} = xz(); }}"
                        ));
                    }
                }
            }
            M::Vmovd | M::Vmovq => {
                let bits = if m == M::Vmovd { 32 } else { 64 };
                let to_x = i.op0_kind() == OpKind::Register && i.op0_register().is_xmm();
                if to_x {
                    let d = vreg(i.op0_register())?.0;
                    let v = if i.op1_kind() == OpKind::Register && i.op1_register().is_xmm() {
                        format!("x{}.q[0]", vreg(i.op1_register())?.0)
                    } else {
                        self.rd(i, 1, bits)?
                    };
                    self.emit(&format!(
                        "x{d}.q[0] = {v} & {}; x{d}.q[1] = 0; xh{d} = xz();",
                        mask(bits)
                    ));
                } else {
                    let s = vreg(i.op1_register())?.0;
                    let w = self.wr(i, 0, &format!("x{s}.q[0] & {}", mask(bits)))?;
                    self.emit(&w);
                }
            }
            M::Vmovlps | M::Vmovhps | M::Vmovlpd | M::Vmovhpd => {
                let hi = matches!(m, M::Vmovhps | M::Vmovhpd);
                if i.op0_kind() == OpKind::Memory {
                    let k = vreg(i.op1_register())?.0;
                    let a = self.addr(i)?;
                    self.emit(&format!("st64({a}, x{k}.q[{}]);", hi as u32));
                } else {
                    let d = vreg(i.op0_register())?.0;
                    let a = self.vsrc(i, 1, false)?.0;
                    let b = self.vscalar(i, 2, 8)?;
                    let helper = if hi { "h_vmovhps" } else { "h_vmovlps" };
                    self.emit(&format!("x{d} = {helper}({a}, {b}); xh{d} = xz();"));
                }
            }
            M::Vmovmskps | M::Vmovmskpd | M::Vpmovmskb => {
                let (k, wide) = vreg(i.op1_register())?;
                let (f, n) = match m {
                    M::Vmovmskps => ("mskps", 4),
                    M::Vmovmskpd => ("mskpd", 2),
                    _ => ("mskb", 16),
                };
                let v = if wide {
                    format!("{f}(x{k}) | {f}(xh{k}) << {n}")
                } else {
                    format!("{f}(x{k})")
                };
                let w = self.wr(i, 0, &format!("(u64)({v})"))?;
                self.emit(&w);
            }
            M::Vzeroupper | M::Vzeroall => {
                let mut s = String::new();
                for k in 0..16 {
                    s.push_str(&format!("xh{k} = xz(); "));
                    if m == M::Vzeroall {
                        s.push_str(&format!("x{k} = xz(); "));
                    }
                }
                self.emit(&s);
            }
            // ---- broadcasts ----
            M::Vbroadcastss | M::Vbroadcastsd | M::Vpbroadcastb | M::Vpbroadcastw
            | M::Vpbroadcastd | M::Vpbroadcastq => {
                let (d, wide) = vreg(i.op0_register())?;
                let (bytes, f) = match m {
                    M::Vpbroadcastb => (1, "h_bcb"),
                    M::Vpbroadcastw => (2, "h_bcw"),
                    M::Vbroadcastss | M::Vpbroadcastd => (4, "h_bcd"),
                    _ => (8, "h_bcq"),
                };
                let s = self.vscalar(i, 1, bytes)?;
                self.emit(&format!(
                    "{{ X128 t = {f}({s}); x{d} = t; xh{d} = {}; }}",
                    if wide { "t" } else { "xz()" }
                ));
            }
            M::Vbroadcastf128 | M::Vbroadcasti128 => {
                let d = vreg(i.op0_register())?.0;
                let s = self.vsrc(i, 1, false)?.0;
                self.emit(&format!("{{ X128 t = {s}; x{d} = t; xh{d} = t; }}"));
            }
            // ---- 128-bit lane moves ----
            M::Vextractf128 | M::Vextracti128 => {
                let (s, _) = vreg(i.op1_register())?;
                let v = if imm() & 1 == 1 {
                    format!("xh{s}")
                } else {
                    format!("x{s}")
                };
                if i.op0_kind() == OpKind::Memory {
                    let a = self.addr(i)?;
                    self.emit(&format!("st128({a}, {v});"));
                } else {
                    let d = vreg(i.op0_register())?.0;
                    self.emit(&format!("x{d} = {v}; xh{d} = xz();"));
                }
            }
            M::Vinsertf128 | M::Vinserti128 => {
                let d = vreg(i.op0_register())?.0;
                let a = self.vsrc(i, 1, true)?;
                let b = self.vsrc(i, 2, false)?.0;
                let s = if imm() & 1 == 1 {
                    format!("{{ X128 t = {b}; X128 l = {}; x{d} = l; xh{d} = t; }}", a.0)
                } else {
                    format!("{{ X128 t = {b}; X128 u = {}; x{d} = t; xh{d} = u; }}", a.1)
                };
                self.emit(&s);
            }
            M::Vperm2f128 | M::Vperm2i128 => {
                let d = vreg(i.op0_register())?.0;
                let a = self.vsrc(i, 1, true)?;
                let b = self.vsrc(i, 2, true)?;
                self.emit(&format!(
                    "{{ Y t = h_vperm2f128((Y){{ {}, {} }}, (Y){{ {}, {} }}, {}); x{d} = t.l; xh{d} = t.h; }}",
                    a.0, a.1, b.0, b.1, imm()
                ));
            }
            M::Vpermq | M::Vpermpd => {
                let d = vreg(i.op0_register())?.0;
                let a = self.vsrc(i, 1, true)?;
                self.emit(&format!(
                    "{{ Y t = h_vpermq((Y){{ {}, {} }}, {}); x{d} = t.l; xh{d} = t.h; }}",
                    a.0, a.1, imm()
                ));
            }
            M::Vpermd | M::Vpermps => {
                let d = vreg(i.op0_register())?.0;
                let a = self.vsrc(i, 1, true)?;
                let b = self.vsrc(i, 2, true)?;
                self.emit(&format!(
                    "{{ Y t = h_vpermd((Y){{ {}, {} }}, (Y){{ {}, {} }}); x{d} = t.l; xh{d} = t.h; }}",
                    a.0, a.1, b.0, b.1
                ));
            }
            // ---- masked moves ----
            M::Vmaskmovps | M::Vmaskmovpd | M::Vpmaskmovd | M::Vpmaskmovq => {
                let sfx = if matches!(m, M::Vmaskmovps | M::Vpmaskmovd) { "d" } else { "q" };
                if i.op0_kind() == OpKind::Memory {
                    let (k, wide) = vreg(i.op1_register())?;
                    let v = vreg(i.op2_register())?.0;
                    let a = self.addr(i)?;
                    self.emit(&format!("h_maskst_{sfx}({a}, x{k}, x{v});"));
                    if wide {
                        self.emit(&format!("h_maskst_{sfx}((u8*)({a}) + 16, xh{k}, xh{v});"));
                    }
                } else {
                    let (d, wide) = vreg(i.op0_register())?;
                    let k = vreg(i.op1_register())?.0;
                    let a = self.addr(i)?;
                    let hi = format!("h_maskld_{sfx}((u8*)({a}) + 16, xh{k})");
                    let s = Self::vset(d, wide, &format!("h_maskld_{sfx}({a}, x{k})"), &hi);
                    self.emit(&s);
                }
            }
            // ---- shifts: immediate or xmm count ----
            M::Vpsllw | M::Vpsrlw | M::Vpsraw | M::Vpslld | M::Vpsrld | M::Vpsrad | M::Vpsllq
            | M::Vpsrlq | M::Vpslldq | M::Vpsrldq => {
                let cnt_n = if i.op_count() == 3 { 2 } else { 1 };
                let cnt = self.count_arg(i, cnt_n)?;
                self.lane(i, &hn, &[1], &[cnt], None)?
            }
            // ---- element insert/extract ----
            M::Vpextrb | M::Vpextrw | M::Vpextrd | M::Vpextrq | M::Vextractps => {
                let (k, _) = vreg(i.op1_register())?;
                let v = imm();
                let e = match m {
                    M::Vpextrb => format!("x{k}.b[{}]", v & 15),
                    M::Vpextrw => format!("x{k}.w[{}]", v & 7),
                    M::Vpextrq => format!("x{k}.q[{}]", v & 1),
                    _ => format!("x{k}.d[{}]", v & 3),
                };
                let w = self.wr(i, 0, &format!("(u64)({e})"))?;
                self.emit(&w);
            }
            M::Vpinsrb | M::Vpinsrw | M::Vpinsrd | M::Vpinsrq => {
                let d = vreg(i.op0_register())?.0;
                let a = self.vsrc(i, 1, false)?.0;
                let (bits, l, n) = match m {
                    M::Vpinsrb => (8, "b", 15),
                    M::Vpinsrw => (16, "w", 7),
                    M::Vpinsrd => (32, "d", 3),
                    _ => (64, "q", 1),
                };
                let src_bits = if i.op2_kind() == OpKind::Register { 64 } else { bits };
                let v = self.rd(i, 2, src_bits)?;
                self.emit(&format!(
                    "{{ X128 t = {a}; t.{l}[{}] = (u{bits})({v}); x{d} = t; xh{d} = xz(); }}",
                    imm() & n
                ));
            }
            M::Vpmovsxbw | M::Vpmovsxbd | M::Vpmovsxbq | M::Vpmovsxwd | M::Vpmovsxwq
            | M::Vpmovsxdq | M::Vpmovzxbw | M::Vpmovzxbd | M::Vpmovzxbq | M::Vpmovzxwd
            | M::Vpmovzxwq | M::Vpmovzxdq | M::Vcvtps2pd | M::Vcvtdq2pd | M::Vcvtph2ps => {
                let (d, wide) = vreg(i.op0_register())?;
                // Source elements consumed by one destination half.
                let per_half = match m {
                    M::Vpmovsxbw | M::Vpmovzxbw => 8,
                    M::Vpmovsxbd | M::Vpmovzxbd | M::Vpmovsxwd | M::Vpmovzxwd | M::Vcvtph2ps => 4,
                    _ => 2,
                };
                let elem = match m {
                    M::Vpmovsxbw | M::Vpmovsxbd | M::Vpmovsxbq | M::Vpmovzxbw | M::Vpmovzxbd
                    | M::Vpmovzxbq => 1,
                    M::Vpmovsxwd | M::Vpmovsxwq | M::Vpmovzxwd | M::Vpmovzxwq | M::Vcvtph2ps => 2,
                    _ => 4,
                };
                let bytes = per_half * elem * if wide { 2 } else { 1 };
                let s = match i.op1_kind() {
                    OpKind::Register => format!("x{}", vreg(i.op1_register())?.0),
                    _ if bytes >= 16 => self.vsrc(i, 1, false)?.0,
                    _ => self.vscalar(i, 1, bytes)?,
                };
                let hi = if wide {
                    format!("{hn}(t, {per_half})")
                } else {
                    "xz()".into()
                };
                self.emit(&format!(
                    "{{ X128 t = {s}; x{d} = {hn}(t, 0); xh{d} = {hi}; }}"
                ));
            }
            M::Vcvtpd2ps | M::Vcvttpd2dq | M::Vcvtpd2dq => {
                let d = vreg(i.op0_register())?.0;
                let wide = match i.op1_kind() {
                    OpKind::Register => vreg(i.op1_register())?.1,
                    _ => Self::mem_bits(i) == 256,
                };
                let (lo, hi) = self.vsrc(i, 1, wide)?;
                let v = if wide {
                    format!("{{ X128 l = {hn}({lo}), u = {hn}({hi}); x{d}.q[0] = l.q[0]; x{d}.q[1] = u.q[0]; }}")
                } else {
                    format!("x{d} = {hn}({lo});")
                };
                self.emit(&format!("{{ {v} xh{d} = xz(); }}"));
            }
            M::Vcvtps2ph => {
                let (k, wide) = vreg(i.op1_register())?;
                let v = imm();
                if v & 4 == 0 && v & 3 != 0 {
                    return Err("vcvtps2ph with a directed rounding mode".into());
                }
                let t = if wide {
                    format!("{{ X128 l = h_vcvtps2ph(x{k}), u = h_vcvtps2ph(xh{k}); X128 t; t.q[0] = l.q[0]; t.q[1] = u.q[0];")
                } else {
                    format!("{{ X128 t = h_vcvtps2ph(x{k});")
                };
                if i.op0_kind() == OpKind::Memory {
                    let a = self.addr(i)?;
                    let st = if wide {
                        format!("st128({a}, t);")
                    } else {
                        format!("st64({a}, t.q[0]);")
                    };
                    self.emit(&format!("{t} {st} }}"));
                } else {
                    let d = vreg(i.op0_register())?.0;
                    self.emit(&format!("{t} x{d} = t; xh{d} = xz(); }}"));
                }
            }
            M::Vptest | M::Vtestps | M::Vtestpd => {
                let wide = vreg(i.op0_register())?.1;
                let msk = match m {
                    M::Vptest => "~0ull",
                    M::Vtestps => "0x8000000080000000ull",
                    _ => "0x8000000000000000ull",
                };
                let a = self.vsrc(i, 0, wide)?;
                let b = self.vsrc(i, 1, wide)?;
                let hi_and = if wide { format!(" | h_pt_and({}, {}, {msk})", a.1, b.1) } else { String::new() };
                let hi_andn = if wide { format!(" | h_pt_andn({}, {}, {msk})", a.1, b.1) } else { String::new() };
                self.emit(&format!(
                    "{{ X128 pa = {}, pb = {}; u64 z = h_pt_and(pa, pb, {msk}){hi_and}; u64 cc = h_pt_andn(pa, pb, {msk}){hi_andn}; fl_set_test(&F, z == 0, cc == 0); }}",
                    a.0, b.0
                ));
            }
            _ => return Ok(false),
        }
        Ok(true)
    }
}
