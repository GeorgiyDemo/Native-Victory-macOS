/* Shared by the "original" (x86-64, run under Rosetta 2) and "lifted" (arm64)
 * test drivers, so both execute the identical call sequence.
 * Build with -DLIFTED for the recompiled image; otherwise the x86 bytes in the
 * flat image are executed directly. */
#include "rt.h"
#include <sys/mman.h>

#ifdef LIFTED
#define MS
#else
#define MS __attribute__((ms_abi))
#endif

#ifndef LIFTED
u8 *IMG;
u8 *rt_gs;
#endif

/* Loads a flat image (RVA-addressed) to IMG. */
static void load_image(const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f) { perror(path); exit(2); }
  fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
  size_t sz = ((size_t)n + 0x4000) & ~(size_t)0x3fff;
  int prot = PROT_READ | PROT_WRITE;
#ifndef LIFTED
  prot |= PROT_EXEC;
#endif
  IMG = mmap(0, sz, prot, MAP_PRIVATE | MAP_ANON, -1, 0);
  if (IMG == MAP_FAILED) { perror("mmap"); exit(2); }
  if (fread(IMG, 1, (size_t)n, f) != (size_t)n) exit(2);
  fclose(f);
}

/* integer args/result */
static u64 ci(u32 rva, u64 a, u64 b, u64 c, u64 d) {
#ifdef LIFTED
  return rt_call_win64(rva, a, b, c, d, 0, 0, 0, 0);
#else
  return ((MS u64 (*)(u64, u64, u64, u64))(IMG + rva))(a, b, c, d);
#endif
}
/* double args and result */
static double cd(u32 rva, double a, double b, double c) {
#ifdef LIFTED
  X128 x; rt_call_win64(rva, 0, 0, 0, 0, a, b, c, &x); return x.g[0];
#else
  return ((MS double (*)(double, double, double))(IMG + rva))(a, b, c);
#endif
}
/* double arg, integer result */
static u64 cdi(u32 rva, double a) {
#ifdef LIFTED
  return rt_call_win64(rva, 0, 0, 0, 0, a, 0, 0, 0);
#else
  return ((MS u64 (*)(double))(IMG + rva))(a);
#endif
}
/* integer arg, double result */
static double cid(u32 rva, u64 a) {
#ifdef LIFTED
  X128 x; rt_call_win64(rva, a, 0, 0, 0, 0, 0, 0, &x); return x.g[0];
#else
  return ((MS double (*)(u64))(IMG + rva))(a);
#endif
}
/* (u64, u64, out*) -> void and (double, double, out*) -> void: the per-instruction snippets. */
static void fi(u32 rva, u64 a, u64 b, u8 *out) {
#ifdef LIFTED
  rt_call_win64(rva, a, b, (u64)out, 0, 0, 0, 0, 0);
#else
  ((MS void (*)(u64, u64, u8 *))(IMG + rva))(a, b, out);
#endif
}
static void fd(u32 rva, double a, double b, u8 *out) {
#ifdef LIFTED
  rt_call_win64(rva, 0, 0, (u64)out, 0, a, b, 0, 0);
#else
  ((MS void (*)(double, double, u8 *))(IMG + rva))(a, b, out);
#endif
}

static u64 rng_state = 0x243F6A8885A308D3ull;
static u64 rnd(void) { rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7; rng_state ^= rng_state << 17; return rng_state; }
