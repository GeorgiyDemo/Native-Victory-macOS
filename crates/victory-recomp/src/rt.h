/* victory-recomp runtime: the contract between lifted code and the host.
 * Guest pointers are host pointers. The image is mapped at `IMG` (host) with
 * relocations already applied, so guest address == IMG + rva. */
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <setjmp.h>
typedef uint8_t u8; typedef uint16_t u16; typedef uint32_t u32; typedef uint64_t u64;
typedef int8_t s8; typedef int16_t s16; typedef int32_t s32; typedef int64_t s64;
typedef union { u8 b[16]; u16 w[8]; u32 d[4]; u64 q[2]; float f[4]; double g[2]; s8 sb[16]; s16 sw[8]; s32 sd[4]; s64 sq[2]; } X128;

/* Lazy EFLAGS. FK_RAW holds explicit bits in `c`: CF=1 ZF=2 SF=4 OF=8 PF=16. */
enum { FK_RAW, FK_ADD, FK_ADC, FK_SUB, FK_SBB, FK_LOGIC, FK_INC, FK_DEC };
typedef struct { u8 k, sz; u64 a, b, r, c; } Fl;

/* Architectural state shared across lifted function boundaries. */
/* x = low 128 bits of ymm0..15, xh = high 128 bits. Win64 treats the upper halves as volatile, but vectorcall can pass ymm. */
typedef struct {
  u64 r[16]; X128 x[16]; X128 xh[16];
  /* What `gs:` addresses: the thread's TEB. The lifter reads `c->gs_base + offset` for a gs: access, so the TEB
   * travels with the Cpu (an arm64 process has no free segment register). Set once per guest thread, never spilled. */
  u64 gs_base;
  /* The guest address of the last instruction that may fault, in functions with landing pads (RT_MARK in the lifter). */
  u64 rip;
} Cpu;

extern u8 *IMG;      /* host address of RVA 0 */
#define H(va) ((u8 *)(IMG + ((va) - IMAGE_BASE)))

/* ---- Host runtime (rt_host.c) ---- */
typedef void (*RtFn)(Cpu *);
/* rt_dispatch resolves a guest address to lifted code (hash over the generated rt_table), to an import
 * thunk (an IAT slot value from rt_import_bind), or to rt_unresolved, which logs the address and hands it
 * to the pluggable fallback (interpreter, JIT helper) or aborts. */
void rt_dispatch(Cpu *c, u64 target);
void rt_unresolved(Cpu *c, u64 target);
void rt_trap(const char *why);
typedef void (*RtFallback)(Cpu *c, u64 target);
void rt_set_fallback(RtFallback f);
u64 rt_unresolved_count(void); /* distinct unresolved targets seen so far */
/* Maps the table over an image already placed at `img` (size bytes): builds the lookup hash. */
void rt_init(u8 *img, u64 size);

/* Imports. A lifted call through an IAT slot lands in rt_dispatch with the slot's value, which is the thunk
 * address rt_import_bind returned. The host function is called directly: arguments are taken from the Cpu by the
 * Windows x64 rules (rcx rdx r8 r9 / xmm0-3 by position, then the stack after the return address and 32 bytes of
 * shadow space) and passed by the native (AAPCS64) rules, so it can be a plain C or Rust `extern "C"` function.
 * At most 16 integer and 8 floating-point arguments (those past the eighth integer go on the native stack as
 * 64-bit slots, so the function must declare them 64-bit). `raw`, if set, is called with the Cpu instead. */
enum { RT_RET_I64, RT_RET_I32, RT_RET_F64, RT_RET_F32, RT_RET_VOID };
typedef struct {
  const char *name;
  void *fn;           /* typed host function */
  RtFn raw;           /* or: void f(Cpu *) */
  u8 argc;
  u16 fmask, smask;   /* bit i: argument i is a double / a float */
  u8 ret;
  u16 bmask, hmask;   /* bit i: integer argument i is 8 / 16 bits wide (the rest of its register is undefined) */
} RtImport;
u64 rt_import_bind(const RtImport *imp);
const char *rt_import_name(u64 thunk);
/* The name of the import this thread's lifted code is inside (for a raw handler shared by several), or null. */
const char *rt_current_import(void);

/* Callbacks (host to guest). Sets up a Cpu from Windows x64 arguments (bit i of fmask: argument i is a double,
 * given as its bit pattern), runs the guest function at `target` on the thread's guest stack (below the
 * innermost running lifted frame if called from an import) and returns rax; xmm0's low qword goes to *xmm0. */
u64 rt_call_guest(u64 target, const u64 *args, u32 n, u32 fmask, u64 *xmm0);
/* Exceptions (docs/recompiler.md, "Exceptions"). A lifted function with landing pads (an `__except` target) is a
 * wrapper that records a frame (rt_frame_enter) and `_setjmp`s, around the body `fb_<rva>(c, resume)`. To resume at a
 * pad the host calls rt_resume with the unwound registers: it finds the frame, loads the registers into its Cpu, sets
 * rt_resume_va and longjmps; the wrapper then runs the body again with resume set, which jumps to the pad. */
typedef struct RtFrame {
  jmp_buf jb; struct RtFrame *prev; Cpu *c; u64 entry_rsp; Cpu *cur; const void *cur_imp; const void *cb;
} RtFrame;
extern __thread u64 rt_resume_va;
void rt_frame_enter(RtFrame *fr, Cpu *c);
void rt_frame_leave(RtFrame *fr);
/* Continues at the landing pad `rip` of the innermost frame whose entry rsp is above gpr[4], with the registers of
 * a Windows CONTEXT (rax rcx rdx rbx rsp rbp rsi rdi r8..r15; xmm0..15). Does not return. */
void rt_resume(const u64 *gpr, const X128 *xmm, u64 rip, u64 ret) __attribute__((noreturn));
/* `rip` is a landing pad or resume point inside the frame's function (the wrapper runs the body again with `resume`
 * set), or the start of a function: then the frame's function "calls" it, with `ret` as the return address, on the
 * frame's stack as the registers leave it (what RtlUnwindEx to a TargetIp that is a function does). */
extern __thread u64 rt_resume_ret_va;
extern __thread int rt_resume_call;
void rt_resume_enter(Cpu *c) __attribute__((noreturn));
/* An integer divide error (#DE) in lifted code: the registers are spilled and `ip` is the guest address of the
 * instruction. Raises EXCEPTION_INT_DIVIDE_BY_ZERO (or EXCEPTION_INT_OVERFLOW when the divisor was not zero) through the
 * fault hook; without a hook it traps. A handler that continues execution re-runs the instruction. */
void rt_de(Cpu *c, u64 ip, int overflow) __attribute__((noreturn));
#define RT_DE(ip, ovf) do { SPILL(); rt_de(c, (ip) - IMAGE_BASE + (u64)IMG, (ovf)); } while (0)
/* The guest frame that called a host function which called back into the guest (rt_call_guest): an unwind that
 * reaches the callback's bottom (a zero return address at `slot`) continues in the frame that made the host call,
 * with the registers it had then. Fills `regs` and the return address and stack pointer after that call; 0 if the
 * callback was not made from guest code. */
int rt_callback_context(u64 slot, Cpu *regs, u64 *rip, u64 *rsp);
/* A fault in lifted code: what the signal handler saw, and the thread's Cpu (the state at its last spill). */
typedef struct { u32 signo, code, esr; u64 pc, addr; Cpu *run; } RtFault;
typedef void (*RtFaultHook)(const RtFault *);
/* Installs the signal handlers; the hook runs on the faulting thread, on its stack below the fault, and must not
 * return (it dispatches the exception and ends in rt_resume, or aborts). */
void rt_fault_install(RtFaultHook hook);
/* Per-thread guest state: the TEB `gs:` addresses and the guest stack [lo, hi). A thread that never calls this
 * gets a zeroed TEB and a 1 MiB stack on first use. */
void rt_thread_init(u8 *gs_base, u8 *stack_lo, u8 *stack_hi);

/* Thread-local accesses (the guest stack, the TEB) and the building blocks of the ordered ones: plain. */
static inline u8 ld8(const u8 *p) { return *p; }
static inline void st8(u8 *p, u64 v) { *p = (u8)v; }
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

/* Memory ordering (docs/recompiler.md, "Memory ordering"). The lifter emits `sld*`/`sst*` (shared) for every access
 * that is not provably thread-local (anything not based on rsp or gs:), and the plain `ld*`/`st*` above for the rest.
 *   default:      shared accesses are plain: ARM's weak model, fastest, wrong for lock-free code that relies on TSO.
 *   -DRT_TSO:     shared loads are acquire (LDAR/LDAPR) and shared stores release (STLR), which gives x86's
 *                 load-load, load-store and store-store ordering; a store may still pass a later load, as on x86.
 *   -DRT_TSO_ALL: every access, stack included (to measure what the thread-local test saves).
 * Misaligned accesses cannot be LDAR/STLR (alignment fault); they take a plain access plus a barrier. */
#if defined(RT_TSO) || defined(RT_TSO_ALL)
#define RT_ORD(T, N, BITS) \
  static inline T o_ld##BITS(const u8 *p) { \
    if (__builtin_expect(((uintptr_t)p & (N - 1)) == 0, 1)) return __atomic_load_n((const T *)p, __ATOMIC_ACQUIRE); \
    T v; memcpy(&v, p, N); __atomic_thread_fence(__ATOMIC_ACQUIRE); return v; } \
  static inline void o_st##BITS(u8 *p, u64 v) { \
    T t = (T)v; \
    if (__builtin_expect(((uintptr_t)p & (N - 1)) == 0, 1)) { __atomic_store_n((T *)p, t, __ATOMIC_RELEASE); return; } \
    __atomic_thread_fence(__ATOMIC_RELEASE); memcpy(p, &t, N); }
RT_ORD(u16, 2, 16) RT_ORD(u32, 4, 32) RT_ORD(u64, 8, 64)
static inline u8 o_ld8(const u8 *p) { return __atomic_load_n(p, __ATOMIC_ACQUIRE); }
static inline void o_st8(u8 *p, u64 v) { __atomic_store_n(p, (u8)v, __ATOMIC_RELEASE); }
/* 128-bit accesses have no LDAR/STLR: a barrier around the plain access. */
static inline X128 o_ld128(const u8 *p) { X128 v = ld128(p); __atomic_thread_fence(__ATOMIC_ACQUIRE); return v; }
static inline void o_st128(u8 *p, X128 v) { __atomic_thread_fence(__ATOMIC_RELEASE); st128(p, v); }
static inline float o_ldf(const u8 *p) { u32 v = o_ld32(p); float f; memcpy(&f, &v, 4); return f; }
static inline double o_ldg(const u8 *p) { u64 v = o_ld64(p); double f; memcpy(&f, &v, 8); return f; }
static inline void o_stf(u8 *p, float f) { u32 v; memcpy(&v, &f, 4); o_st32(p, v); }
static inline void o_stg(u8 *p, double f) { u64 v; memcpy(&v, &f, 8); o_st64(p, v); }
static inline X128 o_ldx8(const u8 *p) { X128 r; r.q[0] = r.q[1] = 0; r.b[0] = o_ld8(p); return r; }
static inline X128 o_ldx16(const u8 *p) { X128 r; r.q[0] = r.q[1] = 0; r.w[0] = o_ld16(p); return r; }
static inline X128 o_ldx32(const u8 *p) { X128 r; r.q[0] = r.q[1] = 0; r.d[0] = o_ld32(p); return r; }
static inline X128 o_ldx64(const u8 *p) { X128 r; r.q[0] = o_ld64(p); r.q[1] = 0; return r; }
#define sld8 o_ld8
#define sst8 o_st8
#define sld16 o_ld16
#define sld32 o_ld32
#define sld64 o_ld64
#define sst16 o_st16
#define sst32 o_st32
#define sst64 o_st64
#define sld128 o_ld128
#define sst128 o_st128
#define sldf o_ldf
#define sldg o_ldg
#define sstf o_stf
#define sstg o_stg
#define sldx8 o_ldx8
#define sldx16 o_ldx16
#define sldx32 o_ldx32
#define sldx64 o_ldx64
#else
#define sld8 ld8
#define sst8 st8
#define sld16 ld16
#define sld32 ld32
#define sld64 ld64
#define sst16 st16
#define sst32 st32
#define sst64 st64
#define sld128 ld128
#define sst128 st128
#define sldf ldf
#define sldg ldg
#define sstf stf
#define sstg stg
#define sldx8 ldx8
#define sldx16 ldx16
#define sldx32 ldx32
#define sldx64 ldx64
#endif

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

/* Update one flag, keep the rest. */
static inline void fl_set_flag(Fl *f, int bit, int v) {
  u64 c = (u64)fl_cf(f) | (u64)fl_zf(f) << 1 | (u64)fl_sf(f) << 2 | (u64)fl_of(f) << 3 | (u64)fl_pf(f) << 4;
  c = (c & ~(1ull << bit)) | (u64)(v != 0) << bit;
  f->k = FK_RAW; f->c = c;
}
#define fl_set_cf(f, v) fl_set_flag(f, 0, v)
#define fl_set_zf(f, v) fl_set_flag(f, 1, v)
#define fl_set_of(f, v) fl_set_flag(f, 3, v)
/* RFLAGS image (AF not modelled, IF set as in user mode) and back. */
static inline u64 fl_pack(const Fl *f) {
  return (u64)fl_cf(f) | 2 | (u64)fl_pf(f) << 2 | (u64)fl_zf(f) << 6 | (u64)fl_sf(f) << 7 | 0x200 | (u64)fl_of(f) << 11;
}
static inline void fl_unpack(Fl *f, u64 v) {
  f->k = FK_RAW; f->c = (v & 1) | (v >> 6 & 1) << 1 | (v >> 7 & 1) << 2 | (v >> 11 & 1) << 3 | (v >> 2 & 1) << 4;
}
static inline u64 pdep64(u64 src, u64 m) { u64 r = 0; for (u64 b = 1; m; b <<= 1) { u64 low = m & -m; if (src & b) r |= low; m &= m - 1; } return r; }
static inline u64 pext64(u64 src, u64 m) { u64 r = 0; for (u64 b = 1; m; b <<= 1) { u64 low = m & -m; if (src & low) r |= b; m &= m - 1; } return r; }
/* Host services for instructions the lifted code cannot do itself. */
void rt_cpuid(u32 leaf, u32 sub, u32 *a, u32 *b, u32 *c, u32 *d);
u64 rt_xgetbv(u32 index);
u64 rt_rdtsc(void);

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
/* Functions that touch ymm/VEX carry the upper halves too. */
#define SPILLY() do { SPILL(); \
  c->xh[0]=xh0;c->xh[1]=xh1;c->xh[2]=xh2;c->xh[3]=xh3;c->xh[4]=xh4;c->xh[5]=xh5;c->xh[6]=xh6;c->xh[7]=xh7; \
  c->xh[8]=xh8;c->xh[9]=xh9;c->xh[10]=xh10;c->xh[11]=xh11;c->xh[12]=xh12;c->xh[13]=xh13;c->xh[14]=xh14;c->xh[15]=xh15; } while (0)
#define RELOADY() do { RELOAD(); \
  xh0=c->xh[0];xh1=c->xh[1];xh2=c->xh[2];xh3=c->xh[3];xh4=c->xh[4];xh5=c->xh[5];xh6=c->xh[6];xh7=c->xh[7]; \
  xh8=c->xh[8];xh9=c->xh[9];xh10=c->xh[10];xh11=c->xh[11];xh12=c->xh[12];xh13=c->xh[13];xh14=c->xh[14];xh15=c->xh[15]; } while (0)
#define FUNC_PROLOGUE_Y() FUNC_PROLOGUE(); \
  X128 xh0=c->xh[0],xh1=c->xh[1],xh2=c->xh[2],xh3=c->xh[3],xh4=c->xh[4],xh5=c->xh[5],xh6=c->xh[6],xh7=c->xh[7], \
       xh8=c->xh[8],xh9=c->xh[9],xh10=c->xh[10],xh11=c->xh[11],xh12=c->xh[12],xh13=c->xh[13],xh14=c->xh[14],xh15=c->xh[15]
#define FUNC_PROLOGUE() \
  u64 r0=c->r[0],r1=c->r[1],r2=c->r[2],r3=c->r[3],r4=c->r[4],r5=c->r[5],r6=c->r[6],r7=c->r[7], \
      r8=c->r[8],r9=c->r[9],r10=c->r[10],r11=c->r[11],r12=c->r[12],r13=c->r[13],r14=c->r[14],r15=c->r[15]; \
  X128 x0=c->x[0],x1=c->x[1],x2=c->x[2],x3=c->x[3],x4=c->x[4],x5=c->x[5],x6=c->x[6],x7=c->x[7], \
       x8=c->x[8],x9=c->x[9],x10=c->x[10],x11=c->x[11],x12=c->x[12],x13=c->x[13],x14=c->x[14],x15=c->x[15]; \
  Fl F = {0}; (void)F

/* `pad`: a landing pad of some function (a jump to it resumes in that function's frame), not a function start. */
typedef struct { u32 rva; RtFn fn; u32 pad; } RtEntry;
extern const RtEntry rt_table[];
extern const unsigned rt_table_n;
/* A lifted image built as a dylib of its own (rt_unit.c, beside the generated tables) hands the runtime this: the
 * variable that holds its IMG and its dispatch table. rt_image_add registers it for the image mapped at `base`
 * (size bytes, relocated) and sets its IMG; the runtime then dispatches into every registered image. */
typedef struct { u8 **img; const RtEntry *tab; const unsigned *n; } RtUnit;
void rt_image_add(u8 *base, u64 size, const RtUnit *unit);
/* The base of the registered image whose lifted code contains the host instruction `pc`, or 0. */
u64 rt_image_of_pc(u64 pc);

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
