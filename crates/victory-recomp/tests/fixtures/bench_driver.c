/* Micro-benchmark of the lifted memory accesses under the three ordering builds (lifted arm64 only).
 * Kernels come from the generated rvas.h. Usage: bench_driver image.bin */
#include "harness.h"
#include "rvas.h"
#include <time.h>

static double now(void) { return (double)clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }

static void call(u32 rva, u64 a, u64 b, u64 c) {
  u64 args[3] = { a, b, c };
  rt_call_guest((u64)(IMG + rva), args, 3, 0, 0);
}

int main(int argc, char **argv) {
  (void)argc;
  load_image(argv[1]);
  enum { N = 1 << 12, REPS = 4000 };
  u32 *src = aligned_alloc(64, N * 4), *dst = aligned_alloc(64, N * 4);
  for (int i = 0; i < N; i++) { src[i] = i; dst[i] = 0; }
  /* a ring of pointers for the dependent-load chase */
  u64 *ring = aligned_alloc(64, N * 8);
  for (int i = 0; i < N; i++) ring[i] = (u64)&ring[(i * 977 + 1) % N];
  const struct { const char *name; u32 rva; u64 a, b; } k[] = {
    { "copy+1 (shared load and store)", BENCH_COPY, (u64)src, (u64)dst },
    { "stack spill/reload (thread-local)", BENCH_STACK, 0, 0 },
    { "pointer chase (shared dependent loads)", BENCH_CHASE, (u64)ring, 0 },
    { "store stream (shared stores)", BENCH_STORE, (u64)dst, 0 },
  };
  for (unsigned i = 0; i < sizeof k / sizeof *k; i++) {
    call(k[i].rva, k[i].a, k[i].b, N); /* warm up */
    double best = 1e30;
    for (int rep = 0; rep < 5; rep++) {
      double t0 = now();
      for (int r = 0; r < REPS; r++) call(k[i].rva, k[i].a, k[i].b, N);
      double dt = now() - t0;
      if (dt < best) best = dt;
    }
    printf("%s %.4f ns/iter\n", k[i].name, best / ((double)N * REPS));
  }
  return 0;
}
