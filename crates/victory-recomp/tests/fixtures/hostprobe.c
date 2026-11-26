/* A freestanding Windows x64 DLL that calls into a host through a real import table (host.dll, import
 * library made from host.def) and exports functions the host calls back. Own code; built like testpe.c. */
#define X __declspec(dllexport)
#define I __declspec(dllimport)
typedef unsigned long long u64;
typedef unsigned u32;

I u64 host_add3(u64 a, u64 b, u64 c);
/* a rcx, b xmm1, c r8, d xmm3, then e and f on the stack. */
I double host_mix(u64 a, double b, u64 c, float d, u64 e, double f);
I u64 host_sum8(u64 a, u64 b, u64 c, u64 d, u64 e, u64 f, u64 g, u64 h);
I u64 host_callback(u64 (*cb)(u64, double), u64 x);
I int host_puts(const char *s);

X u64 guest_square_plus(u64 x, double y) { return x * x + (u64)y; }

X u64 guest_run(u64 n) {
  u64 s = host_add3(n, n + 1, n + 2);
  double m = host_mix(n, 1.5, n + 7, 0.25f, n + 9, 4.0);
  s += (u64)(m * 1000.0);
  s += host_sum8(1, 2, 3, 4, 5, 6, 7, n);
  s += host_callback(guest_square_plus, n);
  host_puts("hello from lifted code");
  return s;
}

/* A call through a pointer the host chooses: an address the lifter never saw goes to the dispatcher. */
X u64 guest_indirect(u64 (*f)(u64), u64 x) { return f(x) + 1; }

/* gs: is the TEB: self pointer (0x30) and stack base (0x08). */
X u64 guest_teb_self(void) { u64 v; __asm__ volatile("movq %%gs:0x30, %0" : "=r"(v)); return v; }
X u64 guest_teb_stack_base(void) { u64 v; __asm__ volatile("movq %%gs:0x08, %0" : "=r"(v)); return v; }

X void guest_cpuid(u32 leaf, u32 sub, u32 *out) {
  __asm__ volatile("cpuid" : "=a"(out[0]), "=b"(out[1]), "=c"(out[2]), "=d"(out[3]) : "a"(leaf), "c"(sub));
}
X u64 guest_xgetbv(void) { u32 a, d; __asm__ volatile("xgetbv" : "=a"(a), "=d"(d) : "c"(0)); return a | (u64)d << 32; }
X u64 guest_tsc_delta(void) { u64 a = __builtin_ia32_rdtsc(); u64 b = __builtin_ia32_rdtsc(); return b - a; }

int DllMain(void *h, u32 r, void *x) { (void)h; (void)r; (void)x; return 1; }
