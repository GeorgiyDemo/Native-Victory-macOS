/* Runtime for lifted code: dispatch, imports, callbacks, CPUID. See docs/recompiler.md ("Runtime").
 * Prototype host and the real one share this file; the pluggable parts are rt_set_fallback (unresolved
 * targets) and the imports the host binds. */
#include "rt.h"
#include <pthread.h>
#include <sys/mman.h>
#include <time.h>
#include <signal.h>
#include <dlfcn.h>
#include <sys/ucontext.h>

#ifndef RT_MULTI
u8 *IMG; /* one image, compiled into the same dylib as the runtime */
#endif

void rt_trap(const char *why) { fprintf(stderr, "rt_trap: %s\n", why); abort(); }

/* ---- dispatch: one table per lifted image ----
 * A process can hold several images (the exe and the DLLs it ships), each a lifted unit of its own (a dylib that
 * defines IMG, rt_table and rt_unit; see rt_unit.c) registered by rt_image_add. An address belongs to the image whose
 * mapped range holds it. Built with -DRT_MULTI the runtime has no unit of its own; without it, a single image
 * compiled in beside the runtime is registered by rt_init(img, size). */
typedef struct {
  u8 *base; u64 size;
  const RtEntry *tab; unsigned n;
  u32 *hash;    /* index + 1 into tab, 0 = empty */
  u32 shift;
  const void *dylib; /* base of the dylib holding the lifted code, to tell a fault in it from one in the host */
} RtImg;
#define MAX_IMAGES 32
static RtImg images[MAX_IMAGES];
static unsigned n_images;

static inline u32 hash_of(const RtImg *m, u32 rva) { return (rva * 0x9E3779B1u) >> m->shift; }

static unsigned image_add(u8 *base, u64 size, const RtEntry *tab, unsigned n, u8 **slot) {
  if (n_images == MAX_IMAGES) rt_trap("too many lifted images");
  RtImg *m = &images[n_images];
  memset(m, 0, sizeof *m);
  m->base = base; m->size = size; m->tab = tab; m->n = n;
  u32 bits = 4;
  while ((1u << bits) < n * 2u) bits++;
  m->shift = 32 - bits;
  m->hash = calloc((size_t)1 << bits, sizeof(u32));
  if (!m->hash) rt_trap("out of memory building the dispatch hash");
  for (unsigned i = 0; i < n; i++) {
    u32 h = hash_of(m, tab[i].rva);
    while (m->hash[h]) h = (h + 1) & ((1u << bits) - 1);
    m->hash[h] = i + 1;
  }
  Dl_info di;
  m->dylib = dladdr(tab, &di) ? di.dli_fbase : 0;
  if (slot) *slot = base;
  __atomic_store_n(&n_images, n_images + 1, __ATOMIC_RELEASE); /* published once filled: other threads dispatch meanwhile */
  return n_images - 1;
}

void rt_image_add(u8 *base, u64 size, const RtUnit *u) { image_add(base, size, u->tab, *u->n, u->img); }

#ifdef RT_MULTI
void rt_init(u8 *img, u64 size) { (void)img; (void)size; rt_trap("rt_init: this runtime has no unit of its own (rt_image_add)"); }
#else
void rt_init(u8 *img, u64 size) {
  if (n_images) rt_trap("rt_init: an image is already attached");
  image_add(img, size, rt_table, rt_table_n, &IMG);
}
#endif

static inline RtFn lookup(const RtImg *m, u32 rva) {
  u32 mask = (1u << (32 - m->shift)) - 1;
  for (u32 h = hash_of(m, rva); m->hash[h]; h = (h + 1) & mask)
    if (m->tab[m->hash[h] - 1].rva == rva) return m->tab[m->hash[h] - 1].fn;
  return 0;
}

/* The image holding the host instruction `pc`, by the dylib it is in: its base address, or 0. */
u64 rt_image_of_pc(u64 pc) {
  Dl_info di;
  if (!dladdr((void *)pc, &di)) return 0;
  for (unsigned i = 0; i < n_images; i++) if (images[i].dylib == di.dli_fbase) return (u64)images[i].base;
  return 0;
}

/* ---- imports ---- */
#define MAX_IMPORTS 16384
static _Alignas(16) u8 imp_area[MAX_IMPORTS * 16];
static RtImport imps[MAX_IMPORTS];
static unsigned n_imps;
static __thread const RtImport *rt_cur_imp; /* the import this thread is inside */
static __thread Cpu *rt_cur; /* innermost Cpu of this thread's lifted code that entered the host */
static __thread Cpu *rt_run; /* the Cpu of this thread's innermost rt_call_guest: what lifted code is running on */

u64 rt_import_bind(const RtImport *m) {
  if (n_imps == MAX_IMPORTS) rt_trap("too many imports");
  unsigned i = n_imps++;
  imps[i] = *m;
  imps[i].name = strdup(m->name ? m->name : "?");
  return (u64)(imp_area + (size_t)i * 16);
}
const char *rt_current_import(void) { return rt_cur_imp ? rt_cur_imp->name : 0; }
const char *rt_import_name(u64 t) {
  return t >= (u64)imp_area && t < (u64)(imp_area + sizeof imp_area) ? imps[(t - (u64)imp_area) / 16].name : 0;
}

typedef u64 (*FnI)(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, double, double, double, double, double, double, double, double);
typedef double (*FnD)(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, double, double, double, double, double, double, double, double);
typedef float (*FnF)(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, double, double, double, double, double, double, double, double);

static void import_call(Cpu *c, const RtImport *m) {
  Cpu *saved = rt_cur;
  const RtImport *saved_imp = rt_cur_imp;
  rt_cur = c;
  rt_cur_imp = m;
  if (m->raw) {
    m->raw(c);
  } else {
    static const u8 reg[4] = { 1, 2, 8, 9 }; /* rcx rdx r8 r9 */
    u64 ia[16] = {0}; double da[8] = {0};
    unsigned ni = 0, nf = 0;
    for (unsigned i = 0; i < m->argc; i++) {
      int isf = (m->fmask | m->smask) >> i & 1;
      u64 v = i < 4 ? (isf ? c->x[i].q[0] : c->r[reg[i]]) : ld64((u8 *)(c->r[4] + 8 + 8 * (u64)i));
      if (isf) { if (nf == 8) rt_trap("more than 8 floating-point import arguments"); memcpy(&da[nf++], &v, 8); }
      else { if (ni == 16) rt_trap("more than 16 integer import arguments"); ia[ni++] = v; }
    }
    switch (m->ret) {
    case RT_RET_F64: c->x[0].g[0] = ((FnD)m->fn)(ia[0], ia[1], ia[2], ia[3], ia[4], ia[5], ia[6], ia[7], ia[8], ia[9], ia[10], ia[11], ia[12], ia[13], ia[14], ia[15], da[0], da[1], da[2], da[3], da[4], da[5], da[6], da[7]); break;
    case RT_RET_F32: c->x[0].f[0] = ((FnF)m->fn)(ia[0], ia[1], ia[2], ia[3], ia[4], ia[5], ia[6], ia[7], ia[8], ia[9], ia[10], ia[11], ia[12], ia[13], ia[14], ia[15], da[0], da[1], da[2], da[3], da[4], da[5], da[6], da[7]); break;
    default: {
      u64 r = ((FnI)m->fn)(ia[0], ia[1], ia[2], ia[3], ia[4], ia[5], ia[6], ia[7], ia[8], ia[9], ia[10], ia[11], ia[12], ia[13], ia[14], ia[15], da[0], da[1], da[2], da[3], da[4], da[5], da[6], da[7]);
      /* A 32-bit result leaves the upper half of x0 undefined on arm64; Windows code reads eax. */
      c->r[0] = m->ret == RT_RET_I32 ? (u32)r : m->ret == RT_RET_VOID ? c->r[0] : r;
    }
    }
  }
  rt_cur = saved;
  rt_cur_imp = saved_imp;
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
  if (fresh) {
    const RtImg *in = 0;
    for (unsigned i = 0; i < n_images; i++) if (t - (u64)images[i].base < images[i].size) in = &images[i];
    fprintf(stderr, "rt: unresolved guest target %#llx (%s, rva %#llx)\n", (unsigned long long)t,
            in ? "inside an image, not a lifted function start" : "outside every image",
            (unsigned long long)(in ? t - (u64)in->base : t));
  }
  if (!fallback) { fprintf(stderr, "rt: no fallback installed\n"); abort(); }
  fallback(c, t);
}

void rt_dispatch(Cpu *c, u64 target) {
  for (unsigned i = 0, n = __atomic_load_n(&n_images, __ATOMIC_ACQUIRE); i < n; i++) {
    const RtImg *m = &images[i];
    u64 rva = target - (u64)m->base;
    if (__builtin_expect(rva < m->size, 1)) {
      RtFn f = lookup(m, (u32)rva);
      if (__builtin_expect(f != 0, 1)) { f(c); return; }
      rt_unresolved(c, target);
      return;
    }
  }
  if (target - (u64)imp_area < sizeof imp_area) {
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
  /* On the stack's top, a margin: the callee may read stack arguments of an import call past what the caller wrote. */
  u64 top = rt_cur ? rt_cur->r[4] - 0x200 : (u64)tl_hi - 0x100;
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
  Cpu *saved = rt_cur, *saved_run = rt_run;
  rt_cur = 0; rt_run = &c; /* the callee is the new innermost frame; an import it calls sets rt_cur itself */
  rt_dispatch(&c, target);
  rt_cur = saved; rt_run = saved_run;
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

/* ---- exceptions: frames, resuming at a landing pad, faults in lifted code ---- */
static __thread RtFrame *rt_frames;
__thread u64 rt_resume_va;

void rt_frame_enter(RtFrame *fr, Cpu *c) {
  fr->prev = rt_frames; fr->c = c; fr->entry_rsp = c->r[4]; fr->cur = rt_cur; fr->cur_imp = rt_cur_imp;
  rt_frames = fr;
}
void rt_frame_leave(RtFrame *fr) { rt_frames = fr->prev; }

void rt_resume(const u64 *gpr, const X128 *xmm, u64 rip) {
  /* Frames of deeper functions have their entry rsp at or below the unwound rsp; the target is the first above. */
  RtFrame *fr = rt_frames;
  while (fr && fr->entry_rsp <= gpr[4]) fr = fr->prev;
  if (!fr) rt_trap("no frame to resume in");
  Cpu *c = fr->c;
  memcpy(c->r, gpr, sizeof c->r);
  memcpy(c->x, xmm, sizeof c->x);
  rt_frames = fr; rt_cur = fr->cur; rt_cur_imp = (const RtImport *)fr->cur_imp; rt_run = c;
  rt_resume_va = rip;
  longjmp(fr->jb, 1);
}

static RtFaultHook fault_hook;
static __thread RtFault fault_info;
static struct sigaction old_action[32];
static int is_unit_dylib(const void *base) {
  for (unsigned i = 0; i < n_images; i++) if (images[i].dylib == base) return 1;
  return 0;
}

/* Runs on the faulting thread after the signal handler returned into it: the stack is the thread's own, below the
 * fault. The guest's rsp is what the last spill says; frames for callbacks go well below it. */
static void fault_entry(void) {
  static __thread Cpu below;
  below = *fault_info.run;
  below.r[4] -= 0x10000;
  rt_cur = &below; rt_cur_imp = 0;
  fault_hook(&fault_info);
  fprintf(stderr, "rt: the fault hook returned\n");
  abort();
}

static void on_fault(int sig, siginfo_t *si, void *uctx) {
  ucontext_t *uc = uctx;
  u64 pc = uc->uc_mcontext->__ss.__pc, sp = uc->uc_mcontext->__ss.__sp;
  Dl_info di;
  /* Ours only if lifted code (this dylib) was running on a guest call. */
  if (rt_run && fault_hook && dladdr((void *)pc, &di) && is_unit_dylib(di.dli_fbase)) {
    fault_info = (RtFault){ (u32)sig, (u32)si->si_code, (u32)uc->uc_mcontext->__es.__esr, pc, (u64)si->si_addr, rt_run };
    uc->uc_mcontext->__ss.__sp = (sp - 256) & ~15ull; /* below the red zone */
    uc->uc_mcontext->__ss.__pc = (u64)fault_entry;
    return;
  }
  /* Not ours: the previous action takes the fault again once we return (or immediately, if it was a handler). */
  sigaction(sig, &old_action[sig], 0);
}

void rt_fault_install(RtFaultHook hook) {
  fault_hook = hook;
  struct sigaction sa;
  memset(&sa, 0, sizeof sa);
  sa.sa_sigaction = on_fault;
  sa.sa_flags = SA_SIGINFO;
  sigemptyset(&sa.sa_mask);
  static const int sigs[] = { SIGSEGV, SIGBUS, SIGILL, SIGFPE, SIGTRAP };
  for (unsigned i = 0; i < sizeof sigs / sizeof *sigs; i++) sigaction(sigs[i], &sa, &old_action[sigs[i]]);
}
