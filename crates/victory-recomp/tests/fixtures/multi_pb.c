/* Second image of the three-image process: plain exports, one of which calls through a pointer. */
#define X __declspec(dllexport)
typedef unsigned long long u64;
typedef unsigned u32;
X u64 pb_scale(u64 x) { return x * 3; }
X u64 pb_apply(u64 (*fp)(u64), u64 x) { return fp(x) + 1; }
int DllMain(void *h, u32 r, void *x) { (void)h; (void)r; (void)x; return 1; }
