/* The "exe" of a three-image process (pa.dll, pb.dll, pc.dll: own code, built like hleprobe.c): imports pb.dll,
 * hands pb pointers to its own functions, and looks pc.dll up with LoadLibraryA and GetProcAddress. */
#define X __declspec(dllexport)
#define I __declspec(dllimport)
typedef unsigned long long u64;
typedef unsigned u32;

I u64 pb_scale(u64 x);                      /* x * 3 */
I u64 pb_apply(u64 (*fp)(u64), u64 x);      /* fp(x) + 1 */
I void *LoadLibraryA(const char *name);
I void *GetProcAddress(void *module, const char *name);

typedef u64 (*Fn1)(u64);
typedef u64 (*Fn2)(Fn1, u64);

/* A function of this image that another image calls through a pointer. */
static u64 exe_triple_plus_7(u64 x) { return pb_scale(x) + 7; }

/* exe -> pb through the import table */
X u64 probe_import(u64 x) { return pb_scale(x); }
/* exe -> pb -> (pointer) exe -> pb: (3x + 7) + 1 */
X u64 probe_via_pointer(u64 x) { return pb_apply(exe_triple_plus_7, x); }
/* a pointer to another image's function, taken here and called here */
X u64 probe_pointer_to_import(u64 x) {
  Fn1 f = pb_scale;
  return f(x) + 1;
}
/* exe -> pc by LoadLibraryA + GetProcAddress; pc imports pb */
X u64 probe_getproc(u64 x) {
  void *h = LoadLibraryA("pc.dll");
  if (!h) return ~0ull;
  Fn1 f = (Fn1)GetProcAddress(h, "pc_nest");
  return f ? f(x) : ~1ull;
}
/* exe -> pc (pointer) -> exe (pointer given) -> pb */
X u64 probe_getproc_callback(u64 x) {
  void *h = LoadLibraryA("pc.dll");
  Fn2 f = h ? (Fn2)GetProcAddress(h, "pc_call") : 0;
  return f ? f(exe_triple_plus_7, x) : ~1ull;
}
/* The same function by two routes: GetProcAddress and the import table agree. */
X u64 probe_same_address(void) {
  void *h = LoadLibraryA("pb.dll");
  return h && GetProcAddress(h, "pb_scale") == (void *)pb_scale;
}
int DllMain(void *h, u32 r, void *x) { (void)h; (void)r; (void)x; return 1; }
