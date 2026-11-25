/* A tiny freestanding Windows x64 DLL used as the recompiler's end-to-end
 * input. Own code; built by tools/macos/gnullvm-cc.sh (see tests/e2e.rs). */
#define X __declspec(dllexport)
typedef unsigned long long u64;
typedef long long s64;
typedef unsigned u32;

__attribute__((noinline)) static u64 helper(u64 x) { return x * 3 + (x >> 2); }

X u64 t_add3(u64 a, u64 b, u64 c) { return a + b + c; }
X u64 t_mix(u64 a, u64 b) {
  a ^= a >> 13;
  a *= 0x9E3779B97F4A7C15ull;
  a = (a << 7) | (a >> 57);
  return a + b * 31 - (a & b);
}
X u64 t_fib(u64 n) { u64 a = 0, b = 1; while (n--) { u64 t = a + b; a = b; b = t; } return a; }
X u64 t_gcd(u64 a, u64 b) { while (b) { u64 t = a % b; a = b; b = t; } return a; }
X s64 t_sdiv(s64 a, s64 b) { if (b == 0 || b == -1) return 0; return a / b + a % b; }
X u64 t_collatz(u64 n) { u64 s = 0; n = n % 100000 + 1; while (n != 1) { n = (n & 1) ? 3 * n + 1 : n >> 1; s++; } return s; }
X u64 t_sum_bytes(const unsigned char *p, u64 n) { u64 s = 0; for (u64 i = 0; i < n; i++) s += p[i] * (i + 1); return s; }
X void t_fill(u32 *p, u64 n, u32 v) { for (u64 i = 0; i < n; i++) p[i] = v + (u32)i * 3; }
X double t_dot(const double *a, const double *b, u64 n) { double s = 0; for (u64 i = 0; i < n; i++) s += a[i] * b[i]; return s; }
X double t_lerp(double a, double b, double t) { float fa = (float)a, fb = (float)b, ft = (float)t; return (double)(fa + (fb - fa) * ft); }
X double t_hyp(double x, double y) { return __builtin_sqrt(x * x + y * y); }
X s64 t_ftoi(double x) { return (s64)x; }
X double t_itof(s64 x) { return (double)x; }
X u64 t_min(s64 a, s64 b) { return (u64)(a < b ? a : b); }
X u32 t_narrow(u32 a) { unsigned short s = (unsigned short)a; s = (unsigned short)(s * 3 + 1); signed char c = (signed char)a; return s + c; }
X u64 t_calls(u64 a, u64 b) { return helper(a) + helper(b) * 5; }
X u64 t_fact(u64 n) { return n <= 1 ? 1 : n * t_fact(n - 1); }
X u64 t_switch(u64 x) {
  switch (x) {
  case 0: return 11;
  case 1: return x * 100 + 7;
  case 2: return x ^ 0x55;
  case 3: return helper(x);
  case 4: return 4444;
  case 5: return x << 9;
  case 6: return ~x;
  case 7: return x + 70;
  default: return x * 2;
  }
}

int DllMain(void *h, u32 r, void *x) { (void)h; (void)r; (void)x; return 1; }
