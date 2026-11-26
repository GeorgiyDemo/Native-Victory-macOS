/* Third image: imports pb.dll, calls back through pointers it is given. */
#define X __declspec(dllexport)
#define I __declspec(dllimport)
typedef unsigned long long u64;
typedef unsigned u32;
I u64 pb_scale(u64 x);
X u64 pc_inc(u64 x) { return x + 1000; }
/* pc -> pb */
X u64 pc_nest(u64 x) { return pb_scale(pc_inc(x)); }
/* pc -> (pointer given by the caller's image) */
X u64 pc_call(u64 (*fp)(u64), u64 x) { return fp(x) * 2; }
int DllMain(void *h, u32 r, void *x) { (void)h; (void)r; (void)x; return 1; }
