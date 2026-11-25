/* victory-recomp runtime: the contract between lifted code and the host.
 * Guest pointers are host pointers. The image is mapped at `IMG` (host) with
 * relocations already applied, so guest address == IMG + rva. */
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
typedef uint8_t u8; typedef uint16_t u16; typedef uint32_t u32; typedef uint64_t u64;
typedef int8_t s8; typedef int16_t s16; typedef int32_t s32; typedef int64_t s64;
typedef union { u8 b[16]; u16 w[8]; u32 d[4]; u64 q[2]; float f[4]; double g[2]; } X128;

/* Lazy EFLAGS. FK_RAW holds explicit bits in `c`: CF=1 ZF=2 SF=4 OF=8 PF=16. */
enum { FK_RAW, FK_ADD, FK_ADC, FK_SUB, FK_SBB, FK_LOGIC, FK_INC, FK_DEC };
typedef struct { u8 k, sz; u64 a, b, r, c; } Fl;

/* Architectural state shared across lifted function boundaries. */
typedef struct { u64 r[16]; X128 x[16]; } Cpu;

extern u8 *IMG;      /* host address of RVA 0 */
extern u8 *rt_gs;    /* GS base: TEB (or the per-thread block the host chooses) */
#define H(va) ((u8 *)(IMG + ((va) - IMAGE_BASE)))

/* Host hooks. rt_dispatch resolves a guest address to lifted code; anything it
 * cannot resolve (HLE import, JIT, unknown) goes to rt_unresolved. */
void rt_dispatch(Cpu *c, u64 target);
void rt_unresolved(Cpu *c, u64 target);
void rt_trap(const char *why);

static inline u16 ld16(const u8 *p) { u16 v; memcpy(&v, p, 2); return v; }
static inline u32 ld32(const u8 *p) { u32 v; memcpy(&v, p, 4); return v; }
static inline u64 ld64(const u8 *p) { u64 v; memcpy(&v, p, 8); return v; }
static inline void st16(u8 *p, u64 v) { u16 t = (u16)v; memcpy(p, &t, 2); }
static inline void st32(u8 *p, u64 v) { u32 t = (u32)v; memcpy(p, &t, 4); }
static inline void st64(u8 *p, u64 v) { memcpy(p, &v, 8); }
static inline X128 ld128(const u8 *p) { X128 v; memcpy(&v, p, 16); return v; }
static inline void st128(u8 *p, X128 v) { memcpy(p, &v, 16); }
static inline float ldf(const u8 *p) { float v; memcpy(&v, p, 4); return v; }
static inline double ldg(const u8 *p) { double v; memcpy(&v, p, 8); return v; }
static inline void stf(u8 *p, float v) { memcpy(p, &v, 4); }
static inline void stg(u8 *p, double v) { memcpy(p, &v, 8); }

#define MASK(n) ((n) == 64 ? ~0ull : ((1ull << (n)) - 1))
static inline u64 sx(u64 v, int bits) { return (u64)((s64)(v << (64 - bits)) >> (64 - bits)); }

static inline int fl_cf(const Fl *f) {
  switch (f->k) {
  case FK_ADD: return f->r < f->a;
  case FK_ADC: return f->c ? f->r <= f->a : f->r < f->a;
  case FK_SUB: return f->a < f->b;
  case FK_SBB: return f->c ? f->a <= f->b : f->a < f->b;
  case FK_LOGIC: return 0;
  default: return f->c & 1; /* RAW, INC, DEC keep CF in c */
  }
}
static inline int fl_zf(const Fl *f) { return f->k == FK_RAW ? (f->c >> 1) & 1 : f->r == 0; }
static inline int fl_sf(const Fl *f) { return f->k == FK_RAW ? (f->c >> 2) & 1 : (f->r >> (f->sz - 1)) & 1; }
static inline int fl_pf(const Fl *f) { return f->k == FK_RAW ? (f->c >> 4) & 1 : !__builtin_parityll(f->r & 0xff); }
static inline int fl_of(const Fl *f) {
  int s = f->sz - 1;
  switch (f->k) {
  case FK_ADD: case FK_ADC: return (((f->a ^ f->r) & (f->b ^ f->r)) >> s) & 1;
  case FK_SUB: case FK_SBB: return (((f->a ^ f->b) & (f->a ^ f->r)) >> s) & 1;
  case FK_INC: return f->r == (1ull << s);
  case FK_DEC: return f->r == (1ull << s) - 1;
  case FK_LOGIC: return 0;
  default: return (f->c >> 3) & 1;
  }
}
/* Explicit-flag state from a result and chosen CF/OF. */
static inline void fl_set_raw(Fl *f, int sz, u64 r, int cf, int of) {
  r &= MASK(sz);
  f->k = FK_RAW; f->sz = sz;
  f->c = (u64)cf | (r == 0) << 1 | (u64)((r >> (sz - 1)) & 1) << 2 | (u64)of << 3 | (u64)!__builtin_parityll(r & 0xff) << 4;
}
/* Rotates leave ZF/SF/PF alone. */
static inline void fl_set_cfof(Fl *f, int cf, int of) {
  u64 keep = (u64)fl_zf(f) << 1 | (u64)fl_sf(f) << 2 | (u64)fl_pf(f) << 4;
  f->k = FK_RAW; f->c = keep | (u64)cf | (u64)of << 3;
}
static inline void fl_set_ucomi(Fl *f, double a, double b) {
  int un = a != a || b != b;
  int cf = un || a < b, zf = un || a == b, pf = un;
  f->k = FK_RAW; f->c = (u64)cf | (u64)zf << 1 | (u64)pf << 4;
}

/* x86 returns the negative "real indefinite" QNaN for invalid operations; ARM returns a positive one. */
/* Done on the bit patterns: LLVM treats every NaN as equivalent and would delete a float-typed fix-up. */
static inline double nanfix_d(double r, double a, double b) {
  u64 br, ba, bb; memcpy(&br, &r, 8); memcpy(&ba, &a, 8); memcpy(&bb, &b, 8);
  if ((br << 1) > 0xFFE0000000000000ull && (ba << 1) <= 0xFFE0000000000000ull && (bb << 1) <= 0xFFE0000000000000ull) br = 0xFFF8000000000000ull;
  memcpy(&r, &br, 8); return r;
}
static inline float nanfix_f(float r, float a, float b) {
  u32 br, ba, bb; memcpy(&br, &r, 4); memcpy(&ba, &a, 4); memcpy(&bb, &b, 4);
  if ((br << 1) > 0xFF000000u && (ba << 1) <= 0xFF000000u && (bb << 1) <= 0xFF000000u) br = 0xFFC00000u;
  memcpy(&r, &br, 4); return r;
}

static inline s64 cvtt64(double v) { return (v != v || v >= 9223372036854775808.0 || v < -9223372036854775808.0) ? INT64_MIN : (s64)v; }
static inline s32 cvtt32(double v) { return (v != v || v >= 2147483648.0 || v <= -2147483649.0) ? INT32_MIN : (s32)v; }

#define SPILL() do { c->r[0]=r0;c->r[1]=r1;c->r[2]=r2;c->r[3]=r3;c->r[4]=r4;c->r[5]=r5;c->r[6]=r6;c->r[7]=r7; \
  c->r[8]=r8;c->r[9]=r9;c->r[10]=r10;c->r[11]=r11;c->r[12]=r12;c->r[13]=r13;c->r[14]=r14;c->r[15]=r15; \
  c->x[0]=x0;c->x[1]=x1;c->x[2]=x2;c->x[3]=x3;c->x[4]=x4;c->x[5]=x5;c->x[6]=x6;c->x[7]=x7; \
  c->x[8]=x8;c->x[9]=x9;c->x[10]=x10;c->x[11]=x11;c->x[12]=x12;c->x[13]=x13;c->x[14]=x14;c->x[15]=x15; } while (0)
#define RELOAD() do { r0=c->r[0];r1=c->r[1];r2=c->r[2];r3=c->r[3];r4=c->r[4];r5=c->r[5];r6=c->r[6];r7=c->r[7]; \
  r8=c->r[8];r9=c->r[9];r10=c->r[10];r11=c->r[11];r12=c->r[12];r13=c->r[13];r14=c->r[14];r15=c->r[15]; \
  x0=c->x[0];x1=c->x[1];x2=c->x[2];x3=c->x[3];x4=c->x[4];x5=c->x[5];x6=c->x[6];x7=c->x[7]; \
  x8=c->x[8];x9=c->x[9];x10=c->x[10];x11=c->x[11];x12=c->x[12];x13=c->x[13];x14=c->x[14];x15=c->x[15]; } while (0)
#define FUNC_PROLOGUE() \
  u64 r0=c->r[0],r1=c->r[1],r2=c->r[2],r3=c->r[3],r4=c->r[4],r5=c->r[5],r6=c->r[6],r7=c->r[7], \
      r8=c->r[8],r9=c->r[9],r10=c->r[10],r11=c->r[11],r12=c->r[12],r13=c->r[13],r14=c->r[14],r15=c->r[15]; \
  X128 x0=c->x[0],x1=c->x[1],x2=c->x[2],x3=c->x[3],x4=c->x[4],x5=c->x[5],x6=c->x[6],x7=c->x[7], \
       x8=c->x[8],x9=c->x[9],x10=c->x[10],x11=c->x[11],x12=c->x[12],x13=c->x[13],x14=c->x[14],x15=c->x[15]; \
  Fl F = {0}; (void)F; (void)rt_gs

typedef void (*RtFn)(Cpu *);
typedef struct { u32 rva; RtFn fn; } RtEntry;
extern const RtEntry rt_table[];
extern const unsigned rt_table_n;

/* Calls a lifted function by RVA with the Windows x64 convention (tests, host entry). */
static inline u64 rt_call_win64(u32 rva, u64 a, u64 b, u64 cc, u64 d, double da, double db, double dc, X128 *xmm0) {
  static _Alignas(16) u8 stack[1 << 20];
  Cpu c; memset(&c, 0, sizeof c);
  u64 top = (u64)(stack + sizeof stack) - 0x48; /* rsp = 8 mod 16 at entry, 32 bytes shadow above */
  st64((u8 *)top, 0);
  c.r[4] = top; c.r[1] = a; c.r[2] = b; c.r[8] = cc; c.r[9] = d;
  c.x[0].g[0] = da; c.x[1].g[0] = db; c.x[2].g[0] = dc;
  rt_dispatch(&c, (u64)(IMG + rva));
  if (xmm0) *xmm0 = c.x[0];
  return c.r[0];
}
