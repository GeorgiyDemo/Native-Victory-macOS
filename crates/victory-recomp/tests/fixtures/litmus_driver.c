/* Memory-ordering litmus driver. The same source runs the x86-64 original under Rosetta 2 (hardware TSO) and the
 * lifted arm64 build. Thread functions come from the generated rvas.h; each takes (X, Y, out) in rcx, rdx, r8.
 * Usage: litmus_driver image.bin iterations */
#include "harness.h"
#include "rvas.h"
#include <pthread.h>

typedef struct { const char *name; int nthreads; u32 rva[4]; int (*weak)(const u32 (*out)[2]); const char *meaning; } Test;

static int weak_mp(const u32 (*o)[2]) { return o[1][0] == 1 && o[1][1] == 0; }   /* saw y but not x */
static int weak_sb(const u32 (*o)[2]) { return o[0][0] == 0 && o[1][0] == 0; }   /* both loads missed the other's store */
static int weak_lb(const u32 (*o)[2]) { return o[0][0] == 1 && o[1][0] == 1; }   /* each load saw the later store */
static int weak_iriw(const u32 (*o)[2]) { return o[2][0] == 1 && o[2][1] == 0 && o[3][0] == 1 && o[3][1] == 0; }

static const Test tests[] = {
  { "MP",      2, { MP_W, MP_R, 0, 0 },                 weak_mp,   "forbidden on TSO" },
  { "SB",      2, { SB_0, SB_1, 0, 0 },                 weak_sb,   "ALLOWED on TSO (store buffer)" },
  { "SB+mfence", 2, { SBF_0, SBF_1, 0, 0 },             weak_sb,   "forbidden on TSO" },
  { "LB",      2, { LB_0, LB_1, 0, 0 },                 weak_lb,   "forbidden on TSO" },
  { "IRIW",    4, { IRIW_W0, IRIW_W1, IRIW_R0, IRIW_R1 }, weak_iriw, "forbidden on TSO" },
};

static _Alignas(256) u8 mem[4096];
#define X (mem + 0)
#define Y (mem + 512)
#define OUT(t) ((u32 *)(mem + 1024 + 128 * (t)))

static struct { _Alignas(128) int count; _Alignas(128) int sense; } bar;
static long iters, weak_count;
static const Test *cur;

static void barrier(int *local) {
  *local = !*local;
  if (__atomic_add_fetch(&bar.count, 1, __ATOMIC_SEQ_CST) == cur->nthreads) {
    __atomic_store_n(&bar.count, 0, __ATOMIC_RELAXED);
    __atomic_store_n(&bar.sense, *local, __ATOMIC_RELEASE);
  } else {
    while (__atomic_load_n(&bar.sense, __ATOMIC_ACQUIRE) != *local) {}
  }
}

static void call3(u32 rva, u64 a, u64 b, u64 c) {
#ifdef LIFTED
  u64 args[3] = { a, b, c };
  rt_call_guest((u64)(IMG + rva), args, 3, 0, 0);
#else
  ((MS void (*)(u64, u64, u64))(IMG + rva))(a, b, c);
#endif
}

static void *worker(void *arg) {
  int t = (int)(long)arg, local = 0;
  u64 s = 0x9E3779B97F4A7C15ull * (t + 1);
  for (long it = 0; it < iters; it++) {
    barrier(&local);
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    for (u64 k = s % 48; k; k--) __asm__ volatile("" ::: "memory");
    call3(cur->rva[t], (u64)X, (u64)Y, (u64)OUT(t));
    barrier(&local);
    if (t == 0) {
      u32 o[4][2];
      for (int k = 0; k < cur->nthreads; k++) { o[k][0] = OUT(k)[0]; o[k][1] = OUT(k)[1]; }
      weak_count += cur->weak((const u32 (*)[2])o);
      memset(mem, 0, sizeof mem);
      __atomic_thread_fence(__ATOMIC_SEQ_CST);
    }
  }
  return 0;
}

int main(int argc, char **argv) {
  if (argc < 3) return 2;
  load_image(argv[1]);
  iters = atol(argv[2]);
  for (unsigned k = 0; k < sizeof tests / sizeof *tests; k++) {
    cur = &tests[k]; weak_count = 0; bar.count = 0; bar.sense = 0;
    memset(mem, 0, sizeof mem);
    pthread_t th[4];
    for (int t = 0; t < cur->nthreads; t++) pthread_create(&th[t], 0, worker, (void *)(long)t);
    for (int t = 0; t < cur->nthreads; t++) pthread_join(th[t], 0);
    printf("%s weak=%ld iterations=%ld %s\n", cur->name, weak_count, iters, cur->meaning);
    fflush(stdout);
  }
  return 0;
}
