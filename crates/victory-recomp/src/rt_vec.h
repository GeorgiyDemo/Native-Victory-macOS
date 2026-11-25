/* VEX/AVX helpers. Every helper works on one 128-bit half (X128); a 256-bit
 * instruction applies it to the low and the high half, except the cross-lane
 * ones, which take and return Y. The name is `h_` plus the lower-case mnemonic.
 * Semantics follow the Intel SDM; they are checked against Rosetta 2 by tests/diff.rs. */
typedef struct { X128 l, h; } Y;
#define LP(n) for (int k = 0; k < (n); k++)

static inline X128 xz(void) { X128 r; r.q[0] = 0; r.q[1] = 0; return r; }
static inline X128 ldx8(const u8 *p) { X128 r = xz(); r.b[0] = *p; return r; }
static inline X128 ldx16(const u8 *p) { X128 r = xz(); r.w[0] = ld16(p); return r; }
static inline X128 ldx32(const u8 *p) { X128 r = xz(); r.d[0] = ld32(p); return r; }
static inline X128 ldx64(const u8 *p) { X128 r = xz(); r.q[0] = ld64(p); return r; }

/* ---- floating point, lane-wise ---- */
#define FB(nm, op) \
  static inline X128 h_v##nm##ps(X128 a, X128 b) { X128 r; LP(4) r.f[k] = nanfix_f(a.f[k] op b.f[k], a.f[k], b.f[k]); return r; } \
  static inline X128 h_v##nm##pd(X128 a, X128 b) { X128 r; LP(2) r.g[k] = nanfix_d(a.g[k] op b.g[k], a.g[k], b.g[k]); return r; } \
  static inline X128 h_v##nm##ss(X128 a, X128 b) { a.f[0] = nanfix_f(a.f[0] op b.f[0], a.f[0], b.f[0]); return a; } \
  static inline X128 h_v##nm##sd(X128 a, X128 b) { a.g[0] = nanfix_d(a.g[0] op b.g[0], a.g[0], b.g[0]); return a; }
FB(add, +) FB(sub, -) FB(mul, *) FB(div, /)
/* x86 min/max return the second operand when either is NaN or both are equal. */
#define FM(nm, op) \
  static inline X128 h_v##nm##ps(X128 a, X128 b) { X128 r; LP(4) r.f[k] = a.f[k] op b.f[k] ? a.f[k] : b.f[k]; return r; } \
  static inline X128 h_v##nm##pd(X128 a, X128 b) { X128 r; LP(2) r.g[k] = a.g[k] op b.g[k] ? a.g[k] : b.g[k]; return r; } \
  static inline X128 h_v##nm##ss(X128 a, X128 b) { a.f[0] = a.f[0] op b.f[0] ? a.f[0] : b.f[0]; return a; } \
  static inline X128 h_v##nm##sd(X128 a, X128 b) { a.g[0] = a.g[0] op b.g[0] ? a.g[0] : b.g[0]; return a; }
FM(min, <) FM(max, >)

static inline X128 h_vsqrtps(X128 a) { X128 r; LP(4) r.f[k] = nanfix_f(sqrtf(a.f[k]), a.f[k], a.f[k]); return r; }
static inline X128 h_vsqrtpd(X128 a) { X128 r; LP(2) r.g[k] = nanfix_d(sqrt(a.g[k]), a.g[k], a.g[k]); return r; }
static inline X128 h_vsqrtss(X128 a, X128 b) { a.f[0] = nanfix_f(sqrtf(b.f[0]), b.f[0], b.f[0]); return a; }
static inline X128 h_vsqrtsd(X128 a, X128 b) { a.g[0] = nanfix_d(sqrt(b.g[0]), b.g[0], b.g[0]); return a; }

/* rcpps / rsqrtps: Intel's 12-bit approximations, reproduced from their observable table
 * (index = top 11 / 10 mantissa bits, midpoint, round to nearest on a 2^-13 grid). */
static inline u32 rcp_bits(u32 x) {
  u32 sgn = x & 0x80000000u, e = (x >> 23) & 0xff, m = x & 0x7fffff;
  if (e == 255) return m ? x | 0x400000u : sgn;                 /* NaN stays, inf -> 0 */
  if (e == 0) return sgn | 0x7f800000u;                          /* zero and denormals -> inf */
  if (e >= 253) return sgn;                                      /* result would be denormal */
  u32 i = m >> 12;
  u32 v = (u32)__builtin_roundeven(8192.0 / (1.0 + ((double)i + 0.5) / 2048.0));
  return sgn | ((253 - e) << 23) | ((v - 4096) << 11);
}
static inline u32 rsqrt_bits(u32 x) {
  u32 sgn = x & 0x80000000u, e = (x >> 23) & 0xff, m = x & 0x7fffff;
  if (e == 255) return m ? x | 0x400000u : (sgn ? 0xffc00000u : 0);
  if (e == 0) return sgn | 0x7f800000u;
  if (sgn) return 0xffc00000u;
  u32 i = m >> 13;
  double xo = 1.0 + ((double)i + 0.5) / 1024.0;
  if (e & 1) { u32 v = (u32)__builtin_roundeven(8192.0 / sqrt(xo)); return ((126 + (127 - e) / 2) << 23) | ((v - 4096) << 11); }
  u32 v = (u32)__builtin_roundeven(4096.0 * 1.4142135623730951 / sqrt(xo));
  return ((127 + (126 - (s32)e) / 2) << 23) | ((v - 4096) << 11);
}
static inline X128 h_vrcpps(X128 a) { X128 r; LP(4) r.d[k] = rcp_bits(a.d[k]); return r; }
static inline X128 h_vrsqrtps(X128 a) { X128 r; LP(4) r.d[k] = rsqrt_bits(a.d[k]); return r; }
static inline X128 h_vrcpss(X128 a, X128 b) { a.d[0] = rcp_bits(b.d[0]); return a; }
static inline X128 h_vrsqrtss(X128 a, X128 b) { a.d[0] = rsqrt_bits(b.d[0]); return a; }

static inline double rnd_d(double x, int m) {
  switch ((m & 4) ? 0 : (m & 3)) {
  case 0: return __builtin_roundeven(x);
  case 1: return __builtin_floor(x);
  case 2: return __builtin_ceil(x);
  default: return __builtin_trunc(x);
  }
}
static inline float rnd_f(float x, int m) {
  switch ((m & 4) ? 0 : (m & 3)) {
  case 0: return __builtin_roundevenf(x);
  case 1: return __builtin_floorf(x);
  case 2: return __builtin_ceilf(x);
  default: return __builtin_truncf(x);
  }
}
static inline X128 h_vroundps(X128 a, int m) { X128 r; LP(4) r.f[k] = rnd_f(a.f[k], m); return r; }
static inline X128 h_vroundpd(X128 a, int m) { X128 r; LP(2) r.g[k] = rnd_d(a.g[k], m); return r; }
static inline X128 h_vroundss(X128 a, X128 b, int m) { a.f[0] = rnd_f(b.f[0], m); return a; }
static inline X128 h_vroundsd(X128 a, X128 b, int m) { a.g[0] = rnd_d(b.g[0], m); return a; }

static inline X128 h_vhaddps(X128 a, X128 b) { X128 r; r.f[0] = nanfix_f(a.f[0] + a.f[1], a.f[0], a.f[1]); r.f[1] = nanfix_f(a.f[2] + a.f[3], a.f[2], a.f[3]); r.f[2] = nanfix_f(b.f[0] + b.f[1], b.f[0], b.f[1]); r.f[3] = nanfix_f(b.f[2] + b.f[3], b.f[2], b.f[3]); return r; }
static inline X128 h_vhsubps(X128 a, X128 b) { X128 r; r.f[0] = nanfix_f(a.f[0] - a.f[1], a.f[0], a.f[1]); r.f[1] = nanfix_f(a.f[2] - a.f[3], a.f[2], a.f[3]); r.f[2] = nanfix_f(b.f[0] - b.f[1], b.f[0], b.f[1]); r.f[3] = nanfix_f(b.f[2] - b.f[3], b.f[2], b.f[3]); return r; }
static inline X128 h_vhaddpd(X128 a, X128 b) { X128 r; r.g[0] = nanfix_d(a.g[0] + a.g[1], a.g[0], a.g[1]); r.g[1] = nanfix_d(b.g[0] + b.g[1], b.g[0], b.g[1]); return r; }
static inline X128 h_vhsubpd(X128 a, X128 b) { X128 r; r.g[0] = nanfix_d(a.g[0] - a.g[1], a.g[0], a.g[1]); r.g[1] = nanfix_d(b.g[0] - b.g[1], b.g[0], b.g[1]); return r; }
static inline X128 h_vaddsubps(X128 a, X128 b) { X128 r; LP(4) r.f[k] = (k & 1) ? nanfix_f(a.f[k] + b.f[k], a.f[k], b.f[k]) : nanfix_f(a.f[k] - b.f[k], a.f[k], b.f[k]); return r; }
static inline X128 h_vaddsubpd(X128 a, X128 b) { X128 r; LP(2) r.g[k] = (k & 1) ? nanfix_d(a.g[k] + b.g[k], a.g[k], b.g[k]) : nanfix_d(a.g[k] - b.g[k], a.g[k], b.g[k]); return r; }

/* Compare predicates 0..31 (quiet/signalling variants behave the same without exceptions). */
static inline int cmp_pred(double a, double b, int p) {
  int un = a != a || b != b;
  switch (p & 15) {
  case 0: return !un && a == b;
  case 1: return !un && a < b;
  case 2: return !un && a <= b;
  case 3: return un;
  case 4: return un || a != b;
  case 5: return un || !(a < b);
  case 6: return un || !(a <= b);
  case 7: return !un;
  case 8: return un || a == b;
  case 9: return un || !(a >= b);
  case 10: return un || !(a > b);
  case 11: return 0;
  case 12: return !un && a != b;
  case 13: return !un && a >= b;
  case 14: return !un && a > b;
  default: return 1;
  }
}
static inline X128 h_vcmpps(X128 a, X128 b, int p) { X128 r; LP(4) r.d[k] = cmp_pred(a.f[k], b.f[k], p) ? ~0u : 0; return r; }
static inline X128 h_vcmppd(X128 a, X128 b, int p) { X128 r; LP(2) r.q[k] = cmp_pred(a.g[k], b.g[k], p) ? ~0ull : 0; return r; }
static inline X128 h_vcmpss(X128 a, X128 b, int p) { a.d[0] = cmp_pred(a.f[0], b.f[0], p) ? ~0u : 0; return a; }
static inline X128 h_vcmpsd(X128 a, X128 b, int p) { a.q[0] = cmp_pred(a.g[0], b.g[0], p) ? ~0ull : 0; return a; }

/* ---- bitwise ---- */
#define BW(nm, e) static inline X128 h_##nm(X128 a, X128 b) { X128 r; LP(2) r.q[k] = (e); return r; }
BW(vxorps, a.q[k] ^ b.q[k]) BW(vxorpd, a.q[k] ^ b.q[k]) BW(vpxor, a.q[k] ^ b.q[k])
BW(vandps, a.q[k] & b.q[k]) BW(vandpd, a.q[k] & b.q[k]) BW(vpand, a.q[k] & b.q[k])
BW(vorps, a.q[k] | b.q[k]) BW(vorpd, a.q[k] | b.q[k]) BW(vpor, a.q[k] | b.q[k])
BW(vandnps, ~a.q[k] & b.q[k]) BW(vandnpd, ~a.q[k] & b.q[k]) BW(vpandn, ~a.q[k] & b.q[k])

/* ---- integer lane-wise ---- */
static inline int sat_s8(int v) { return v < -128 ? -128 : v > 127 ? 127 : v; }
static inline int sat_u8(int v) { return v < 0 ? 0 : v > 255 ? 255 : v; }
static inline int sat_s16(s64 v) { return v < -32768 ? -32768 : v > 32767 ? 32767 : (int)v; }
static inline int sat_u16(s64 v) { return v < 0 ? 0 : v > 65535 ? 65535 : (int)v; }
#define LB(nm, N, e) static inline X128 h_##nm(X128 a, X128 b) { X128 r; LP(N) { e; } return r; }
LB(vpaddb, 16, r.b[k] = a.b[k] + b.b[k]) LB(vpaddw, 8, r.w[k] = a.w[k] + b.w[k])
LB(vpaddd, 4, r.d[k] = a.d[k] + b.d[k]) LB(vpaddq, 2, r.q[k] = a.q[k] + b.q[k])
LB(vpsubb, 16, r.b[k] = a.b[k] - b.b[k]) LB(vpsubw, 8, r.w[k] = a.w[k] - b.w[k])
LB(vpsubd, 4, r.d[k] = a.d[k] - b.d[k]) LB(vpsubq, 2, r.q[k] = a.q[k] - b.q[k])
LB(vpaddsb, 16, r.sb[k] = sat_s8(a.sb[k] + b.sb[k])) LB(vpaddsw, 8, r.sw[k] = sat_s16(a.sw[k] + b.sw[k]))
LB(vpaddusb, 16, r.b[k] = sat_u8(a.b[k] + b.b[k])) LB(vpaddusw, 8, r.w[k] = sat_u16(a.w[k] + b.w[k]))
LB(vpsubsb, 16, r.sb[k] = sat_s8(a.sb[k] - b.sb[k])) LB(vpsubsw, 8, r.sw[k] = sat_s16(a.sw[k] - b.sw[k]))
LB(vpsubusb, 16, r.b[k] = sat_u8(a.b[k] - b.b[k])) LB(vpsubusw, 8, r.w[k] = sat_u16(a.w[k] - b.w[k]))
LB(vpmullw, 8, r.w[k] = a.w[k] * b.w[k]) LB(vpmulld, 4, r.d[k] = a.d[k] * b.d[k])
LB(vpmulhw, 8, r.sw[k] = (s16)(((s32)a.sw[k] * b.sw[k]) >> 16)) LB(vpmulhuw, 8, r.w[k] = (u16)(((u32)a.w[k] * b.w[k]) >> 16))
LB(vpmulhrsw, 8, r.sw[k] = (s16)((((s32)a.sw[k] * b.sw[k] >> 14) + 1) >> 1))
LB(vpmuludq, 2, r.q[k] = (u64)a.d[2 * k] * b.d[2 * k]) LB(vpmuldq, 2, r.sq[k] = (s64)a.sd[2 * k] * b.sd[2 * k])
LB(vpmaddwd, 4, r.d[k] = (u32)(a.sw[2 * k] * b.sw[2 * k]) + (u32)(a.sw[2 * k + 1] * b.sw[2 * k + 1]))
LB(vpmaddubsw, 8, r.sw[k] = sat_s16(a.b[2 * k] * b.sb[2 * k] + a.b[2 * k + 1] * b.sb[2 * k + 1]))
LB(vpavgb, 16, r.b[k] = (a.b[k] + b.b[k] + 1) >> 1) LB(vpavgw, 8, r.w[k] = (a.w[k] + b.w[k] + 1) >> 1)
LB(vpminsb, 16, r.sb[k] = a.sb[k] < b.sb[k] ? a.sb[k] : b.sb[k]) LB(vpminsw, 8, r.sw[k] = a.sw[k] < b.sw[k] ? a.sw[k] : b.sw[k])
LB(vpminsd, 4, r.sd[k] = a.sd[k] < b.sd[k] ? a.sd[k] : b.sd[k]) LB(vpminub, 16, r.b[k] = a.b[k] < b.b[k] ? a.b[k] : b.b[k])
LB(vpminuw, 8, r.w[k] = a.w[k] < b.w[k] ? a.w[k] : b.w[k]) LB(vpminud, 4, r.d[k] = a.d[k] < b.d[k] ? a.d[k] : b.d[k])
LB(vpmaxsb, 16, r.sb[k] = a.sb[k] > b.sb[k] ? a.sb[k] : b.sb[k]) LB(vpmaxsw, 8, r.sw[k] = a.sw[k] > b.sw[k] ? a.sw[k] : b.sw[k])
LB(vpmaxsd, 4, r.sd[k] = a.sd[k] > b.sd[k] ? a.sd[k] : b.sd[k]) LB(vpmaxub, 16, r.b[k] = a.b[k] > b.b[k] ? a.b[k] : b.b[k])
LB(vpmaxuw, 8, r.w[k] = a.w[k] > b.w[k] ? a.w[k] : b.w[k]) LB(vpmaxud, 4, r.d[k] = a.d[k] > b.d[k] ? a.d[k] : b.d[k])
LB(vpcmpeqb, 16, r.b[k] = a.b[k] == b.b[k] ? 0xff : 0) LB(vpcmpeqw, 8, r.w[k] = a.w[k] == b.w[k] ? 0xffff : 0)
LB(vpcmpeqd, 4, r.d[k] = a.d[k] == b.d[k] ? ~0u : 0) LB(vpcmpeqq, 2, r.q[k] = a.q[k] == b.q[k] ? ~0ull : 0)
LB(vpcmpgtb, 16, r.b[k] = a.sb[k] > b.sb[k] ? 0xff : 0) LB(vpcmpgtw, 8, r.w[k] = a.sw[k] > b.sw[k] ? 0xffff : 0)
LB(vpcmpgtd, 4, r.d[k] = a.sd[k] > b.sd[k] ? ~0u : 0) LB(vpcmpgtq, 2, r.q[k] = a.sq[k] > b.sq[k] ? ~0ull : 0)
LB(vpsignb, 16, r.sb[k] = b.sb[k] < 0 ? -a.sb[k] : b.sb[k] == 0 ? 0 : a.sb[k])
LB(vpsignw, 8, r.sw[k] = b.sw[k] < 0 ? -a.sw[k] : b.sw[k] == 0 ? 0 : a.sw[k])
LB(vpsignd, 4, r.sd[k] = b.sd[k] < 0 ? (s32)(0u - a.d[k]) : b.sd[k] == 0 ? 0 : a.sd[k])
static inline X128 h_vpabsb(X128 a) { X128 r; LP(16) r.b[k] = a.sb[k] < 0 ? -a.sb[k] : a.sb[k]; return r; }
static inline X128 h_vpabsw(X128 a) { X128 r; LP(8) r.w[k] = a.sw[k] < 0 ? -a.sw[k] : a.sw[k]; return r; }
static inline X128 h_vpabsd(X128 a) { X128 r; LP(4) r.d[k] = a.sd[k] < 0 ? 0u - a.d[k] : a.d[k]; return r; }
static inline X128 h_vpsadbw(X128 a, X128 b) {
  X128 r; LP(2) { u32 s = 0; for (int j = 0; j < 8; j++) { int d = (int)a.b[8 * k + j] - (int)b.b[8 * k + j]; s += d < 0 ? -d : d; } r.q[k] = s; }
  return r;
}
/* Horizontal integer add/sub within each 128-bit lane. */
static inline X128 h_vphaddw(X128 a, X128 b) { X128 r; LP(4) { r.w[k] = a.w[2 * k] + a.w[2 * k + 1]; r.w[k + 4] = b.w[2 * k] + b.w[2 * k + 1]; } return r; }
static inline X128 h_vphaddd(X128 a, X128 b) { X128 r; LP(2) { r.d[k] = a.d[2 * k] + a.d[2 * k + 1]; r.d[k + 2] = b.d[2 * k] + b.d[2 * k + 1]; } return r; }
static inline X128 h_vphsubw(X128 a, X128 b) { X128 r; LP(4) { r.w[k] = a.w[2 * k] - a.w[2 * k + 1]; r.w[k + 4] = b.w[2 * k] - b.w[2 * k + 1]; } return r; }
static inline X128 h_vphsubd(X128 a, X128 b) { X128 r; LP(2) { r.d[k] = a.d[2 * k] - a.d[2 * k + 1]; r.d[k + 2] = b.d[2 * k] - b.d[2 * k + 1]; } return r; }
static inline X128 h_vphaddsw(X128 a, X128 b) { X128 r; LP(4) { r.sw[k] = sat_s16(a.sw[2 * k] + a.sw[2 * k + 1]); r.sw[k + 4] = sat_s16(b.sw[2 * k] + b.sw[2 * k + 1]); } return r; }
static inline X128 h_vphsubsw(X128 a, X128 b) { X128 r; LP(4) { r.sw[k] = sat_s16(a.sw[2 * k] - a.sw[2 * k + 1]); r.sw[k + 4] = sat_s16(b.sw[2 * k] - b.sw[2 * k + 1]); } return r; }

/* Packs. */
static inline X128 h_vpacksswb(X128 a, X128 b) { X128 r; LP(8) { r.sb[k] = sat_s8(a.sw[k]); r.sb[k + 8] = sat_s8(b.sw[k]); } return r; }
static inline X128 h_vpackuswb(X128 a, X128 b) { X128 r; LP(8) { r.b[k] = sat_u8(a.sw[k]); r.b[k + 8] = sat_u8(b.sw[k]); } return r; }
static inline X128 h_vpackssdw(X128 a, X128 b) { X128 r; LP(4) { r.sw[k] = sat_s16(a.sd[k]); r.sw[k + 4] = sat_s16(b.sd[k]); } return r; }
static inline X128 h_vpackusdw(X128 a, X128 b) { X128 r; LP(4) { r.w[k] = sat_u16(a.sd[k]); r.w[k + 4] = sat_u16(b.sd[k]); } return r; }

/* Shifts: the count is the low qword of `cnt` (immediates are passed as a qword). */
#define SHF(nm, L, W, N, e) static inline X128 h_##nm(X128 a, X128 cnt) { X128 r; u64 c = cnt.q[0]; LP(N) { e; } return r; }
SHF(vpsllw, w, 16, 8, r.w[k] = c > 15 ? 0 : (u16)(a.w[k] << c)) SHF(vpsrlw, w, 16, 8, r.w[k] = c > 15 ? 0 : a.w[k] >> c)
SHF(vpsraw, w, 16, 8, r.sw[k] = a.sw[k] >> (c > 15 ? 15 : c))
SHF(vpslld, d, 32, 4, r.d[k] = c > 31 ? 0 : a.d[k] << c) SHF(vpsrld, d, 32, 4, r.d[k] = c > 31 ? 0 : a.d[k] >> c)
SHF(vpsrad, d, 32, 4, r.sd[k] = a.sd[k] >> (c > 31 ? 31 : c))
SHF(vpsllq, q, 64, 2, r.q[k] = c > 63 ? 0 : a.q[k] << c) SHF(vpsrlq, q, 64, 2, r.q[k] = c > 63 ? 0 : a.q[k] >> c)
static inline X128 h_vpslldq(X128 a, X128 cnt) { X128 r; u64 n = cnt.q[0] > 16 ? 16 : cnt.q[0]; LP(16) r.b[k] = k >= (int)n ? a.b[k - n] : 0; return r; }
static inline X128 h_vpsrldq(X128 a, X128 cnt) { X128 r; u64 n = cnt.q[0] > 16 ? 16 : cnt.q[0]; LP(16) r.b[k] = k + (int)n < 16 ? a.b[k + n] : 0; return r; }
/* Per-lane variable shifts (AVX2). */
LB(vpsllvd, 4, r.d[k] = b.d[k] > 31 ? 0 : a.d[k] << b.d[k]) LB(vpsrlvd, 4, r.d[k] = b.d[k] > 31 ? 0 : a.d[k] >> b.d[k])
LB(vpsravd, 4, r.sd[k] = a.sd[k] >> (b.d[k] > 31 ? 31 : b.d[k]))
LB(vpsllvq, 2, r.q[k] = b.q[k] > 63 ? 0 : a.q[k] << b.q[k]) LB(vpsrlvq, 2, r.q[k] = b.q[k] > 63 ? 0 : a.q[k] >> b.q[k])

/* ---- shuffles, within a 128-bit lane ---- */
static inline X128 h_vshufps(X128 a, X128 b, int m) { X128 r; r.f[0] = a.f[m & 3]; r.f[1] = a.f[(m >> 2) & 3]; r.f[2] = b.f[(m >> 4) & 3]; r.f[3] = b.f[(m >> 6) & 3]; return r; }
static inline X128 h_vshufpd(X128 a, X128 b, int m) { X128 r; r.g[0] = a.g[m & 1]; r.g[1] = b.g[(m >> 1) & 1]; return r; }
static inline X128 h_vunpcklps(X128 a, X128 b) { X128 r; r.f[0] = a.f[0]; r.f[1] = b.f[0]; r.f[2] = a.f[1]; r.f[3] = b.f[1]; return r; }
static inline X128 h_vunpckhps(X128 a, X128 b) { X128 r; r.f[0] = a.f[2]; r.f[1] = b.f[2]; r.f[2] = a.f[3]; r.f[3] = b.f[3]; return r; }
static inline X128 h_vunpcklpd(X128 a, X128 b) { X128 r; r.q[0] = a.q[0]; r.q[1] = b.q[0]; return r; }
static inline X128 h_vunpckhpd(X128 a, X128 b) { X128 r; r.q[0] = a.q[1]; r.q[1] = b.q[1]; return r; }
#define UNP(nm, L, N, off) static inline X128 h_##nm(X128 a, X128 b) { X128 r; LP(N / 2) { r.L[2 * k] = a.L[off + k]; r.L[2 * k + 1] = b.L[off + k]; } return r; }
UNP(vpunpcklbw, b, 16, 0) UNP(vpunpckhbw, b, 16, 8) UNP(vpunpcklwd, w, 8, 0) UNP(vpunpckhwd, w, 8, 4)
UNP(vpunpckldq, d, 4, 0) UNP(vpunpckhdq, d, 4, 2) UNP(vpunpcklqdq, q, 2, 0) UNP(vpunpckhqdq, q, 2, 1)
static inline X128 h_vpshufd(X128 a, int m) { X128 r; LP(4) r.d[k] = a.d[(m >> (2 * k)) & 3]; return r; }
static inline X128 h_vpshufhw(X128 a, int m) { X128 r = a; LP(4) r.w[4 + k] = a.w[4 + ((m >> (2 * k)) & 3)]; return r; }
static inline X128 h_vpshuflw(X128 a, int m) { X128 r = a; LP(4) r.w[k] = a.w[(m >> (2 * k)) & 3]; return r; }
static inline X128 h_vpshufb(X128 a, X128 b) { X128 r; LP(16) r.b[k] = (b.b[k] & 0x80) ? 0 : a.b[b.b[k] & 15]; return r; }
static inline X128 h_vpalignr(X128 a, X128 b, int n) {
  X128 r; u8 t[32]; memcpy(t, b.b, 16); memcpy(t + 16, a.b, 16);
  LP(16) r.b[k] = (k + n) < 32 ? t[k + n] : 0;
  return r;
}
static inline X128 h_vpermilps(X128 a, int m) { X128 r; LP(4) r.d[k] = a.d[(m >> (2 * k)) & 3]; return r; }
static inline X128 h_vpermilpsv(X128 a, X128 c) { X128 r; LP(4) r.d[k] = a.d[c.d[k] & 3]; return r; }
static inline X128 h_vpermilpd(X128 a, int m) { X128 r; LP(2) r.q[k] = a.q[(m >> k) & 1]; return r; }
static inline X128 h_vpermilpdv(X128 a, X128 c) { X128 r; LP(2) r.q[k] = a.q[(c.q[k] >> 1) & 1]; return r; }
static inline X128 h_vmovshdup(X128 a) { X128 r; r.f[0] = a.f[1]; r.f[1] = a.f[1]; r.f[2] = a.f[3]; r.f[3] = a.f[3]; return r; }
static inline X128 h_vmovsldup(X128 a) { X128 r; r.f[0] = a.f[0]; r.f[1] = a.f[0]; r.f[2] = a.f[2]; r.f[3] = a.f[2]; return r; }
static inline X128 h_vmovddup(X128 a) { X128 r; r.q[0] = a.q[0]; r.q[1] = a.q[0]; return r; }
static inline X128 h_vmovlhps(X128 a, X128 b) { X128 r; r.q[0] = a.q[0]; r.q[1] = b.q[0]; return r; }
static inline X128 h_vmovhlps(X128 a, X128 b) { X128 r; r.q[0] = b.q[1]; r.q[1] = a.q[1]; return r; }
/* b is the source (register, or a qword loaded into lane 0). */
static inline X128 h_vmovlps(X128 a, X128 b) { a.q[0] = b.q[0]; return a; }
static inline X128 h_vmovhps(X128 a, X128 b) { a.q[1] = b.q[0]; return a; }
static inline X128 h_vinsertps(X128 a, X128 b, int m) {
  X128 r = a; u32 v = b.d[(m >> 6) & 3]; r.d[(m >> 4) & 3] = v;
  LP(4) if ((m >> k) & 1) r.d[k] = 0;
  return r;
}
/* Blends: imm selects b per lane; the mask forms use the sign bit. */
static inline X128 h_vblendps(X128 a, X128 b, int m) { X128 r; LP(4) r.d[k] = ((m >> k) & 1) ? b.d[k] : a.d[k]; return r; }
static inline X128 h_vblendpd(X128 a, X128 b, int m) { X128 r; LP(2) r.q[k] = ((m >> k) & 1) ? b.q[k] : a.q[k]; return r; }
static inline X128 h_vpblendw(X128 a, X128 b, int m) { X128 r; LP(8) r.w[k] = ((m >> k) & 1) ? b.w[k] : a.w[k]; return r; }
static inline X128 h_vblendvps(X128 a, X128 b, X128 m) { X128 r; LP(4) r.d[k] = (m.d[k] >> 31) ? b.d[k] : a.d[k]; return r; }
static inline X128 h_vblendvpd(X128 a, X128 b, X128 m) { X128 r; LP(2) r.q[k] = (m.q[k] >> 63) ? b.q[k] : a.q[k]; return r; }
static inline X128 h_vpblendvb(X128 a, X128 b, X128 m) { X128 r; LP(16) r.b[k] = (m.b[k] >> 7) ? b.b[k] : a.b[k]; return r; }

/* ---- element moves, widening and narrowing ---- */
/* Widening moves read 128-bit source elements starting at `off`. */
#define PMV(nm, S, T, N, ST) static inline X128 h_##nm(X128 a, int off) { X128 r; LP(N) r.T[k] = (ST)a.S[off + k]; return r; }
PMV(vpmovsxbw, sb, sw, 8, s16) PMV(vpmovsxbd, sb, sd, 4, s32) PMV(vpmovsxbq, sb, sq, 2, s64)
PMV(vpmovsxwd, sw, sd, 4, s32) PMV(vpmovsxwq, sw, sq, 2, s64) PMV(vpmovsxdq, sd, sq, 2, s64)
PMV(vpmovzxbw, b, w, 8, u16) PMV(vpmovzxbd, b, d, 4, u32) PMV(vpmovzxbq, b, q, 2, u64)
PMV(vpmovzxwd, w, d, 4, u32) PMV(vpmovzxwq, w, q, 2, u64) PMV(vpmovzxdq, d, q, 2, u64)

static inline u32 mskps(X128 a) { return (a.d[0] >> 31) | (a.d[1] >> 31) << 1 | (a.d[2] >> 31) << 2 | (a.d[3] >> 31) << 3; }
static inline u32 mskpd(X128 a) { return (u32)(a.q[0] >> 63) | (u32)(a.q[1] >> 63) << 1; }
static inline u32 mskb(X128 a) { u32 m = 0; LP(16) m |= (u32)(a.b[k] >> 7) << k; return m; }

/* Conversions. */
static inline s32 cvt32(double v) { v = __builtin_roundeven(v); return (v != v || v >= 2147483648.0 || v < -2147483648.0) ? INT32_MIN : (s32)v; }
static inline s64 cvt64(double v) { v = __builtin_roundeven(v); return (v != v || v >= 9223372036854775808.0 || v < -9223372036854775808.0) ? INT64_MIN : (s64)v; }
static inline X128 h_vcvtdq2ps(X128 a) { X128 r; LP(4) r.f[k] = (float)a.sd[k]; return r; }
static inline X128 h_vcvttps2dq(X128 a) { X128 r; LP(4) r.sd[k] = cvtt32((double)a.f[k]); return r; }
static inline X128 h_vcvtps2dq(X128 a) { X128 r; LP(4) r.sd[k] = cvt32((double)a.f[k]); return r; }
/* Narrowing pd conversions: two doubles per call, results in lanes 0..1 (upper lanes zero). */
static inline X128 h_vcvtpd2ps(X128 a) { X128 r = xz(); LP(2) r.f[k] = (float)a.g[k]; return r; }
static inline X128 h_vcvttpd2dq(X128 a) { X128 r = xz(); LP(2) r.sd[k] = cvtt32(a.g[k]); return r; }
static inline X128 h_vcvtpd2dq(X128 a) { X128 r = xz(); LP(2) r.sd[k] = cvt32(a.g[k]); return r; }
/* Widening: `off` selects which two source elements. */
static inline X128 h_vcvtps2pd(X128 a, int off) { X128 r; LP(2) r.g[k] = (double)a.f[off + k]; return r; }
static inline X128 h_vcvtdq2pd(X128 a, int off) { X128 r; LP(2) r.g[k] = (double)a.sd[off + k]; return r; }
static inline X128 h_vcvtph2ps(X128 a, int off) { X128 r; LP(4) { u16 h = a.w[off + k]; _Float16 t; memcpy(&t, &h, 2); r.f[k] = (float)t; } return r; }
/* Narrowing to half: four floats to the low four halves (upper zero). Rounding is nearest-even. */
static inline X128 h_vcvtps2ph(X128 a) { X128 r = xz(); LP(4) { _Float16 t = (_Float16)a.f[k]; u16 h; memcpy(&h, &t, 2); r.w[k] = h; } return r; }

static inline X128 h_bcb(X128 s) { X128 r; LP(16) r.b[k] = s.b[0]; return r; }
static inline X128 h_bcw(X128 s) { X128 r; LP(8) r.w[k] = s.w[0]; return r; }
static inline X128 h_bcd(X128 s) { X128 r; LP(4) r.d[k] = s.d[0]; return r; }
static inline X128 h_bcq(X128 s) { X128 r; LP(2) r.q[k] = s.q[0]; return r; }
/* vptest / vtestps / vtestpd: ZF and CF from (a & b) and (~a & b), restricted to the bits of `m`. */
static inline u64 h_pt_and(X128 a, X128 b, u64 m) { return ((a.q[0] & b.q[0]) | (a.q[1] & b.q[1])) & m; }
static inline u64 h_pt_andn(X128 a, X128 b, u64 m) { return ((~a.q[0] & b.q[0]) | (~a.q[1] & b.q[1])) & m; }
static inline void fl_set_test(Fl *f, int zf, int cf) { f->k = FK_RAW; f->c = (u64)cf | (u64)zf << 1; }

/* ---- cross-lane (256-bit) ---- */
static inline X128 y_sel(Y a, Y b, int s) { switch (s & 3) { case 0: return a.l; case 1: return a.h; case 2: return b.l; default: return b.h; } }
static inline Y h_vperm2f128(Y a, Y b, int m) { Y r; r.l = (m & 8) ? xz() : y_sel(a, b, m); r.h = (m & 0x80) ? xz() : y_sel(a, b, m >> 4); return r; }
static inline Y h_vpermq(Y a, int m) {
  u64 q[4] = { a.l.q[0], a.l.q[1], a.h.q[0], a.h.q[1] }; Y r;
  r.l.q[0] = q[m & 3]; r.l.q[1] = q[(m >> 2) & 3]; r.h.q[0] = q[(m >> 4) & 3]; r.h.q[1] = q[(m >> 6) & 3];
  return r;
}
/* vpermd/vpermps: idx first, then the table. */
static inline Y h_vpermd(Y idx, Y t) {
  u32 d[8]; memcpy(d, &t.l, 16); memcpy(d + 4, &t.h, 16); Y r;
  LP(4) { r.l.d[k] = d[idx.l.d[k] & 7]; r.h.d[k] = d[idx.h.d[k] & 7]; }
  return r;
}

/* Masked moves: lanes whose mask sign bit is clear are neither read nor written. */
static inline X128 h_maskld_d(const u8 *p, X128 m) { X128 r; LP(4) r.d[k] = (m.d[k] >> 31) ? ld32(p + 4 * k) : 0; return r; }
static inline X128 h_maskld_q(const u8 *p, X128 m) { X128 r; LP(2) r.q[k] = (m.q[k] >> 63) ? ld64(p + 8 * k) : 0; return r; }
static inline void h_maskst_d(u8 *p, X128 m, X128 v) { LP(4) if (m.d[k] >> 31) st32(p + 4 * k, v.d[k]); }
static inline void h_maskst_q(u8 *p, X128 m, X128 v) { LP(2) if (m.q[k] >> 63) st64(p + 8 * k, v.q[k]); }

/* ---- FMA: a*b + c with one rounding. `ng` negates the product, `sc` the addend. ---- */
/* The x86 default NaN (negative) when the result is NaN and no input was. */
static inline float nan3f(float r, float a, float b, float c) {
  u32 br, ba, bb, bc; memcpy(&br, &r, 4); memcpy(&ba, &a, 4); memcpy(&bb, &b, 4); memcpy(&bc, &c, 4);
  if ((br << 1) > 0xFF000000u && (ba << 1) <= 0xFF000000u && (bb << 1) <= 0xFF000000u && (bc << 1) <= 0xFF000000u) br = 0xFFC00000u;
  memcpy(&r, &br, 4); return r;
}
static inline double nan3d(double r, double a, double b, double c) {
  u64 br, ba, bb, bc; memcpy(&br, &r, 8); memcpy(&ba, &a, 8); memcpy(&bb, &b, 8); memcpy(&bc, &c, 8);
  if ((br << 1) > 0xFFE0000000000000ull && (ba << 1) <= 0xFFE0000000000000ull && (bb << 1) <= 0xFFE0000000000000ull && (bc << 1) <= 0xFFE0000000000000ull) br = 0xFFF8000000000000ull;
  memcpy(&r, &br, 8); return r;
}
static inline float fma1f(float a, float b, float c, int ng, int sc) { return nan3f(fmaf(ng ? -a : a, b, sc ? -c : c), a, b, c); }
static inline double fma1d(double a, double b, double c, int ng, int sc) { return nan3d(fma(ng ? -a : a, b, sc ? -c : c), a, b, c); }
static inline X128 h_fma_ps(X128 a, X128 b, X128 c, int ng, int sc) { X128 r; LP(4) r.f[k] = fma1f(a.f[k], b.f[k], c.f[k], ng, sc); return r; }
static inline X128 h_fma_pd(X128 a, X128 b, X128 c, int ng, int sc) { X128 r; LP(2) r.g[k] = fma1d(a.g[k], b.g[k], c.g[k], ng, sc); return r; }
/* Scalar: lane 0 from the operation, the rest from `d` (the destination register). */
static inline X128 h_fma_ss(X128 d, X128 a, X128 b, X128 c, int ng, int sc) { d.f[0] = fma1f(a.f[0], b.f[0], c.f[0], ng, sc); return d; }
static inline X128 h_fma_sd(X128 d, X128 a, X128 b, X128 c, int ng, int sc) { d.g[0] = fma1d(a.g[0], b.g[0], c.g[0], ng, sc); return d; }
/* fmaddsub (sub = 1: even lanes subtract c) and fmsubadd (sub = 0: even lanes add). */
static inline X128 h_fmas_ps(X128 a, X128 b, X128 c, int sub) { X128 r; LP(4) r.f[k] = fma1f(a.f[k], b.f[k], c.f[k], 0, (k & 1) != sub); return r; }
static inline X128 h_fmas_pd(X128 a, X128 b, X128 c, int sub) { X128 r; LP(2) r.g[k] = fma1d(a.g[k], b.g[k], c.g[k], 0, (k & 1) != sub); return r; }
