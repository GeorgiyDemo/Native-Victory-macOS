/* Runtime for lifted code: dispatch, imports, callbacks, CPUID. See docs/recompiler.md ("Runtime").
 * Prototype host and the real one share this file; the pluggable parts are rt_set_fallback (unresolved
 * targets) and the imports the host binds. */
#include "rt.h"
#include <pthread.h>
#include <sys/mman.h>
#include <time.h>

u8 *IMG;
static u64 img_size;

void rt_trap(const char *why) { fprintf(stderr, "rt_trap: %s\n", why); abort(); }

/* ---- dispatch ---- */
static u32 *hash_slot;   /* index + 1 into rt_table, 0 = empty */
static u32 hash_shift;

static inline u32 hash_of(u32 rva) { return (rva * 0x9E3779B1u) >> hash_shift; }

void rt_init(u8 *img, u64 size) {
  IMG = img; img_size = size;
  u32 bits = 4;
  while ((1u << bits) < rt_table_n * 2u) bits++;
  hash_shift = 32 - bits;
  free(hash_slot);
  hash_slot = calloc((size_t)1 << bits, sizeof(u32));
  if (!hash_slot) rt_trap("out of memory building the dispatch hash");
  for (unsigned i = 0; i < rt_table_n; i++) {
    u32 h = hash_of(rt_table[i].rva);
    while (hash_slot[h]) h = (h + 1) & ((1u << bits) - 1);
    hash_slot[h] = i + 1;
  }
}

static inline RtFn lookup(u32 rva) {
  if (__builtin_expect(hash_slot != 0, 1)) {
    u32 mask = (1u << (32 - hash_shift)) - 1;
    for (u32 h = hash_of(rva); hash_slot[h]; h = (h + 1) & mask)
      if (rt_table[hash_slot[h] - 1].rva == rva) return rt_table[hash_slot[h] - 1].fn;
    return 0;
  }
  unsigned lo = 0, hi = rt_table_n; /* before rt_init: binary search */
  while (lo < hi) {
    unsigned mid = (lo + hi) / 2;
    if (rt_table[mid].rva < rva) lo = mid + 1; else hi = mid;
  }
  return lo < rt_table_n && rt_table[lo].rva == rva ? rt_table[lo].fn : 0;
}

/* ---- imports ---- */
#define MAX_IMPORTS 4096
static _Alignas(16) u8 imp_area[MAX_IMPORTS * 16];
static RtImport imps[MAX_IMPORTS];
static unsigned n_imps;
static __thread Cpu *rt_cur; /* innermost Cpu of this thread's lifted code that entered the host */

u64 rt_import_bind(const RtImport *m) {
  if (n_imps == MAX_IMPORTS) rt_trap("too many imports");
  unsigned i = n_imps++;
  imps[i] = *m;
  imps[i].name = strdup(m->name ? m->name : "?");
  return (u64)(imp_area + (size_t)i * 16);
}
const char *rt_import_name(u64 t) {
  return t >= (u64)imp_area && t < (u64)(imp_area + sizeof imp_area) ? imps[(t - (u64)imp_area) / 16].name : 0;
}

typedef u64 (*FnI)(u64, u64, u64, u64, u64, u64, u64, u64, double, double, double, double, double, double, double, double);
typedef double (*FnD)(u64, u64, u64, u64, u64, u64, u64, u64, double, double, double, double, double, double, double, double);
typedef float (*FnF)(u64, u64, u64, u64, u64, u64, u64, u64, double, double, double, double, double, double, double, double);

static void import_call(Cpu *c, const RtImport *m) {
  Cpu *saved = rt_cur;
  rt_cur = c;
  if (m->raw) {
    m->raw(c);
  } else {
    static const u8 reg[4] = { 1, 2, 8, 9 }; /* rcx rdx r8 r9 */
    u64 ia[8] = {0}; double da[8] = {0};
    unsigned ni = 0, nf = 0;
    for (unsigned i = 0; i < m->argc; i++) {
      int isf = (m->fmask | m->smask) >> i & 1;
      u64 v = i < 4 ? (isf ? c->x[i].q[0] : c->r[reg[i]]) : ld64((u8 *)(c->r[4] + 8 + 8 * (u64)i));
      if (isf) { if (nf == 8) rt_trap("more than 8 floating-point import arguments"); memcpy(&da[nf++], &v, 8); }
      else { if (ni == 8) rt_trap("more than 8 integer import arguments"); ia[ni++] = v; }
    }
    switch (m->ret) {
    case RT_RET_F64: c->x[0].g[0] = ((FnD)m->fn)(ia[0], ia[1], ia[2], ia[3], ia[4], ia[5], ia[6], ia[7], da[0], da[1], da[2], da[3], da[4], da[5], da[6], da[7]); break;
    case RT_RET_F32: c->x[0].f[0] = ((FnF)m->fn)(ia[0], ia[1], ia[2], ia[3], ia[4], ia[5], ia[6], ia[7], da[0], da[1], da[2], da[3], da[4], da[5], da[6], da[7]); break;
    default: {
      u64 r = ((FnI)m->fn)(ia[0], ia[1], ia[2], ia[3], ia[4], ia[5], ia[6], ia[7], da[0], da[1], da[2], da[3], da[4], da[5], da[6], da[7]);
      /* A 32-bit result leaves the upper half of x0 undefined on arm64; Windows code reads eax. */
      c->r[0] = m->ret == RT_RET_I32 ? (u32)r : m->ret == RT_RET_VOID ? c->r[0] : r;
    }
    }
  }
  rt_cur = saved;
  c->r[4] += 8; /* the import's `ret`: pop the return address the call pushed */
}

/* ---- unresolved targets ---- */
static RtFallback fallback;
static u64 seen[2048];
static unsigned n_seen;
static pthread_mutex_t seen_lock = PTHREAD_MUTEX_INITIALIZER;
void rt_set_fallback(RtFallback f) { fallback = f; }
u64 rt_unresolved_count(void) { return n_seen; }

void rt_unresolved(Cpu *c, u64 t) {
  pthread_mutex_lock(&seen_lock);
  int fresh = 1;
  for (unsigned i = 0; i < n_seen; i++) if (seen[i] == t) fresh = 0;
  if (fresh && n_seen < 2048) seen[n_seen++] = t;
  pthread_mutex_unlock(&seen_lock);
  if (fresh)
    fprintf(stderr, "rt: unresolved guest target %#llx (%s, rva %#llx)\n", (unsigned long long)t,
            t - (u64)IMG < img_size ? "inside the image, not a lifted function start" : "outside the image",
            (unsigned long long)(t - (u64)IMG));
  if (!fallback) { fprintf(stderr, "rt: no fallback installed\n"); abort(); }
  fallback(c, t);
}

void rt_dispatch(Cpu *c, u64 target) {
  u64 rva = target - (u64)IMG;
  if (__builtin_expect(rva < img_size, 1)) {
    RtFn f = lookup((u32)rva);
    if (__builtin_expect(f != 0, 1)) { f(c); return; }
  } else if (target - (u64)imp_area < sizeof imp_area) {
    unsigned i = (unsigned)((target - (u64)imp_area) / 16);
    if (i < n_imps) { import_call(c, &imps[i]); return; }
  }
  rt_unresolved(c, target);
}

/* ---- guest threads and callbacks ---- */
static __thread u8 *tl_gs, *tl_lo, *tl_hi;

void rt_thread_init(u8 *gs, u8 *lo, u8 *hi) { tl_gs = gs; tl_lo = lo; tl_hi = hi; }

static void thread_defaults(void) {
  if (!tl_hi) {
    size_t n = 1 << 20;
    tl_lo = mmap(0, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (tl_lo == MAP_FAILED) rt_trap("cannot allocate a guest stack");
    tl_hi = tl_lo + n;
  }
  if (!tl_gs) {
    tl_gs = calloc(1, 0x2000);
    st64(tl_gs + 0x30, (u64)tl_gs); st64(tl_gs + 0x08, (u64)tl_hi); st64(tl_gs + 0x10, (u64)tl_lo);
  }
}

u64 rt_call_guest(u64 target, const u64 *args, u32 n, u32 fmask, u64 *xmm0) {
  static const u8 reg[4] = { 1, 2, 8, 9 };
  thread_defaults();
  Cpu c; memset(&c, 0, sizeof c);
  c.gs_base = (u64)tl_gs;
  /* Below the innermost running frame (minus a margin for what the host call itself may have used), else the stack top. */
  u64 top = rt_cur ? rt_cur->r[4] - 0x200 : (u64)tl_hi;
  top &= ~15ull;
  u64 need = (0x28 + 8 * (n > 4 ? n - 4 : 0) + 15) & ~15ull;
  u64 sp = top - need - 8; /* rsp = 8 mod 16 after the (fake) call */
  if (sp < (u64)tl_lo + 0x1000) rt_trap("guest stack overflow in a callback");
  st64((u8 *)sp, 0);
  for (u32 i = 0; i < n; i++) {
    if (i < 4) { if (fmask >> i & 1) c.x[i].q[0] = args[i]; else c.r[reg[i]] = args[i]; }
    else st64((u8 *)(sp + 8 + 8 * (u64)i), args[i]);
  }
  c.r[4] = sp;
  Cpu *saved = rt_cur;
  rt_cur = 0; /* the callee is the new innermost frame; an import it calls sets rt_cur itself */
  rt_dispatch(&c, target);
  rt_cur = saved;
  if (xmm0) *xmm0 = c.x[0].q[0];
  return c.r[0];
}

/* ---- CPUID, XGETBV, RDTSC: the console's CPU ----
 * The Xbox Series X is a custom Zen 2 part (8 cores, 16 threads, 3.8 GHz, AVX2/FMA/BMI2/AES/PCLMUL/SHA/ADX, no
 * AVX-512). What victory-hle reports (IsProcessorFeaturePresent: SSE..SSE4.2, RDTSC, NX, CX16, XSAVE, RDRAND) is a
 * subset of this. The family/model and brand string are those of a retail Zen 2 (family 17h, model 71h) with the
 * console's core count; the title's own CPU checks are the only consumer, so only feature bits matter. */
#define B(n) (1u << (n))
static const u32 F1_ECX = B(0) | B(1) | B(3) | B(9) | B(12) | B(13) | B(19) | B(20) | B(22) | B(23) | B(25) | B(26) | B(27) | B(28) | B(29) | B(30);
/* SSE3 PCLMUL MONITOR SSSE3 FMA CX16 SSE4.1 SSE4.2 MOVBE POPCNT AES XSAVE OSXSAVE AVX F16C RDRAND (no hypervisor bit) */
static const u32 F1_EDX = 0x178BFBFF; /* FPU .. SSE2, HTT, as every Zen */
/* leaf 7: FSGSBASE BMI1 AVX2 SMEP BMI2 INVPCID RDSEED ADX SMAP CLFLUSHOPT CLWB SHA */
static const u32 F7_EBX = B(0) | B(3) | B(5) | B(7) | B(8) | B(10) | B(18) | B(19) | B(20) | B(23) | B(24) | B(29);
/* 0x80000001 ecx: LAHF, CMP_LEGACY, ABM (lzcnt), SSE4A, MISALIGNSSE, 3DNOWPREFETCH, OSVW, TOPOEXT ... */
static const u32 FX1_ECX = 0x75C237FB;
static const u32 FX1_EDX = 0x2FD3FBFF; /* SYSCALL, NX, MMXEXT, FXSR_OPT, 1G pages, RDTSCP, LM */

void rt_cpuid(u32 leaf, u32 sub, u32 *a, u32 *b, u32 *c, u32 *d) {
  static const char brand[48] = "AMD Custom APU (Xbox Series X, Zen 2 emulated)";
  *a = *b = *c = *d = 0;
  switch (leaf) {
  case 0: *a = 0x10; *b = 0x68747541; *d = 0x69746e65; *c = 0x444d4163; break; /* "AuthenticAMD" in ebx, edx, ecx */
  case 1: *a = 0x00870F10; *b = 16u << 16 | 8u << 8; *c = F1_ECX; *d = F1_EDX; break;
  case 6: *a = 4; break; /* ARAT */
  case 7: if (sub == 0) { *b = F7_EBX; *c = B(2) | B(22); } break; /* UMIP, RDPID */
  case 0xB: /* x2APIC topology: SMT level (2 threads), then core level (16 logical) */
    if (sub == 0) { *a = 1; *b = 2; *c = 0x100; } else if (sub == 1) { *a = 4; *b = 16; *c = 0x201; }
    break;
  case 0xD:
    if (sub == 0) { *a = 7; *b = *c = 0x340; } else if (sub == 1) *a = 0xF; /* x87+SSE+AVX state; XSAVEOPT/C, XGETBV1 */
    else if (sub == 2) { *a = 0x100; *b = 0x240; } /* AVX high halves */
    break;
  case 0x80000000: *a = 0x8000001F; *b = 0x68747541; *d = 0x69746e65; *c = 0x444d4163; break;
  case 0x80000001: *a = 0x00870F10; *c = FX1_ECX; *d = FX1_EDX; break;
  case 0x80000002: case 0x80000003: case 0x80000004: {
    const char *p = brand + 16 * (leaf - 0x80000002);
    memcpy(a, p, 4); memcpy(b, p + 4, 4); memcpy(c, p + 8, 4); memcpy(d, p + 12, 4);
    break;
  }
  case 0x80000007: *d = B(8); break; /* invariant TSC */
  case 0x80000008: *a = 0x3030; *c = 0x400F; break; /* 48-bit physical and virtual, 16 threads */
  default: break;
  }
}
u64 rt_xgetbv(u32 index) { return index == 0 ? 7 : 0; } /* x87 | SSE | AVX */
/* One tick per nanosecond, the same 1 GHz the HLE reports for the performance counter. */
u64 rt_rdtsc(void) { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }
