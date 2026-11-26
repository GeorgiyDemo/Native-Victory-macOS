/* Calls a host function through a pointer (as through a method-table slot) with twelve integer arguments: the
 * Windows x64 convention puts the first four in registers and the other eight on the stack. Own code. */
#define X __declspec(dllexport)
typedef unsigned long long u64;
typedef unsigned u32;
typedef u64 (*Twelve)(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64);

X u64 probe_call12(Twelve f) { return f(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12); }

int DllMain(void *h, u32 r, void *x) {
  (void)h;
  (void)r;
  (void)x;
  return 1;
}
