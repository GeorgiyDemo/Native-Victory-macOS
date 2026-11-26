/* Imports the HLE serves (kernel32), one it does not by name and one by ordinal (notserved.dll, made from
 * notserved.def): the binder must stub the last two, not refuse the image. Own code; built like hleprobe.c. */
#define X __declspec(dllexport)
#define I __declspec(dllimport)
typedef unsigned long long u64;
typedef unsigned u32;

I u32 GetCurrentThreadId(void);
I u64 NotServedByName(u64 arg);
I u64 NotServedByOrdinal(u64 arg);

X u32 probe_served(void) { return GetCurrentThreadId(); }
/* A stub returning 0 shows as exactly the constants. */
X u64 probe_unserved(void) { return NotServedByName(0x1234) + 7 | (NotServedByOrdinal(0x5678) + 11) << 32; }

int DllMain(void *h, u32 r, void *x) {
  (void)h;
  (void)r;
  (void)x;
  return 1;
}
