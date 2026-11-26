/* Calls every export of testpe.dll with the same deterministic inputs and prints
 * the results. argv: image.bin exports.txt ("name 0xrva" per line). */
#include "harness.h"
#include <string.h>

static u32 rva_of(const char *want, const char *table) {
  const char *p = table;
  size_t n = strlen(want);
  while (*p) {
    if (!strncmp(p, want, n) && p[n] == ' ') return (u32)strtoul(p + n + 1, 0, 16);
    while (*p && *p != '\n') p++;
    if (*p) p++;
  }
  fprintf(stderr, "no export %s\n", want); exit(2);
}

#define F(n) u32 n = rva_of(#n, ex)
int main(int argc, char **argv) {
  if (argc < 3) return 2;
  load_image(argv[1]);
  static char ex[1 << 14];
  FILE *f = fopen(argv[2], "r"); size_t k = fread(ex, 1, sizeof ex - 1, f); ex[k] = 0; fclose(f);
  F(t_add3); F(t_mix); F(t_fib); F(t_gcd); F(t_sdiv); F(t_collatz); F(t_sum_bytes); F(t_fill); F(t_dot);
  F(t_lerp); F(t_hyp); F(t_ftoi); F(t_itof); F(t_min); F(t_narrow); F(t_calls); F(t_fact); F(t_switch);

  static u8 buf[512]; static u32 words[64]; static double da[32], db[32];
  for (int round = 0; round < 400; round++) {
    u64 a = rnd(), b = rnd(), c = rnd();
    if (round % 5 == 0) { a >>= 40; b >>= 50; }
    printf("add3 %llx\n", (unsigned long long)ci(t_add3, a, b, c, 0));
    printf("mix %llx\n", (unsigned long long)ci(t_mix, a, b, 0, 0));
    printf("fib %llx\n", (unsigned long long)ci(t_fib, a & 63, 0, 0, 0));
    printf("gcd %llx\n", (unsigned long long)ci(t_gcd, a, b, 0, 0));
    printf("sdiv %lld\n", (long long)ci(t_sdiv, a, b >> (round % 40), 0, 0));
    printf("collatz %llx\n", (unsigned long long)ci(t_collatz, a, 0, 0, 0));
    for (int i = 0; i < 512; i++) buf[i] = (u8)rnd();
    printf("sum_bytes %llx\n", (unsigned long long)ci(t_sum_bytes, (u64)buf, rnd() % 512, 0, 0));
    memset(words, 0, sizeof words);
    ci(t_fill, (u64)words, rnd() % 64, (u32)rnd(), 0);
    u64 h = 0; for (int i = 0; i < 64; i++) h = h * 1000003 + words[i];
    printf("fill %llx\n", (unsigned long long)h);
    for (int i = 0; i < 32; i++) { da[i] = (double)(s64)rnd() / 1e12; db[i] = (double)(s32)rnd() / 1e3; }
    u64 nn = rnd() % 32;
#ifdef LIFTED
    X128 xr; rt_call_win64(t_dot, (u64)da, (u64)db, nn, 0, 0, 0, 0, &xr); double dd = xr.g[0];
#else
    double dd = ((MS double (*)(double *, double *, u64))(IMG + t_dot))(da, db, nn);
#endif
    printf("dot %016llx\n", (unsigned long long)*(u64 *)&dd);
    double x = da[0], y = db[1], t = (double)(rnd() % 1000) / 1000.0;
    double v;
    v = cd(t_lerp, x, y, t); printf("lerp %016llx\n", (unsigned long long)*(u64 *)&v);
    v = cd(t_hyp, x, y, 0); printf("hyp %016llx\n", (unsigned long long)*(u64 *)&v);
    printf("ftoi %lld\n", (long long)cdi(t_ftoi, x));
    v = cid(t_itof, (u64)(s64)a); printf("itof %016llx\n", (unsigned long long)*(u64 *)&v);
    printf("min %lld\n", (long long)ci(t_min, a, b, 0, 0));
    printf("narrow %x\n", (u32)ci(t_narrow, a, 0, 0, 0));
    printf("calls %llx\n", (unsigned long long)ci(t_calls, a, b, 0, 0));
    printf("fact %llx\n", (unsigned long long)ci(t_fact, a % 25, 0, 0, 0));
    printf("switch %llx\n", (unsigned long long)ci(t_switch, a % 12, 0, 0, 0));
  }
  return 0;
}
