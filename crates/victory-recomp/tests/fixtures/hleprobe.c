/* A freestanding Windows x64 DLL that calls real exports of victory-hle through a real import table
 * (kernel32.dll and xgameruntime.dll, import libraries made from kernel32.def and xgameruntime.def).
 * Own code; built like hostprobe.c. */
#define X __declspec(dllexport)
#define I __declspec(dllimport)
typedef unsigned long long u64;
typedef unsigned u32;
typedef void *HANDLE;

I u64 GetTickCount64(void);
I u32 GetCurrentThreadId(void);
I HANDLE CreateEventW(void *attributes, int manual_reset, int initial, const unsigned short *name);
I int SetEvent(HANDLE h);
I u32 WaitForSingleObject(HANDLE h, u32 milliseconds);
I int CloseHandle(HANDLE h);
I void SetLastError(u32 code);
I u32 GetLastError(void);
I int IsProcessorFeaturePresent(u32 feature);
I int QueryPerformanceFrequency(long long *out);
I int XGameRuntimeInitialize(void);
I int QueryApiImpl(const void *provider, const void *iid, void **out);

/* Two reads of the clock, a busy loop apart: never backwards. */
X u64 guest_ticks(u64 *first, u64 *second) {
  *first = GetTickCount64();
  for (volatile int i = 0; i < 100000; i++) {}
  *second = GetTickCount64();
  return *second >= *first;
}

/* An event: unsignalled it times out (0x102), signalled it is taken at once (0); both results, packed. */
X u64 guest_event(void) {
  HANDLE e = CreateEventW(0, 1, 0, 0);
  if (!e) return ~0ull;
  u32 before = WaitForSingleObject(e, 0);
  SetEvent(e);
  u32 after = WaitForSingleObject(e, 1000);
  CloseHandle(e);
  return (u64)before << 32 | after;
}

/* The last error lives in the thread's TEB, which the lifted code reads at gs:0x68 too. */
X u64 guest_last_error(u32 code) {
  SetLastError(code);
  u64 teb;
  __asm__ volatile("movq %%gs:0x30, %0" : "=r"(teb));
  u32 seen_through_gs = *(u32 *)(teb + 0x68);
  return (u64)GetLastError() << 32 | seen_through_gs;
}

X u64 guest_teb_self(void) {
  u64 v;
  __asm__ volatile("movq %%gs:0x30, %0" : "=r"(v));
  return v;
}
X u32 guest_thread_id(void) { return GetCurrentThreadId(); }
X u64 guest_has_avx(void) { return (u64)IsProcessorFeaturePresent(39) << 1 | (u64)IsProcessorFeaturePresent(40); }
X long long guest_qpf(void) {
  long long f = 0;
  return QueryPerformanceFrequency(&f) ? f : -1;
}

/* The console runtime: XGameRuntimeInitialize, then an object by (provider, interface id), and a method
 * through the object's table: `GetTitleId(this, &id)` is slot 3. */
typedef struct {
  u32 d1;
  unsigned short d2, d3;
  unsigned char d4[8];
} Guid;
typedef struct {
  void **vtbl;
} Object;
X u64 guest_title_id(u32 *id) {
  static const Guid provider = {0x973a344e, 0x24bf, 0x4d0f, {0x84, 0x57, 0x56, 0xc5, 0x34, 0x89, 0x2b, 0x29}};
  static const Guid iid = {0x2549f142, 0x6419, 0x4a06, {0x97, 0xb5, 0x93, 0x1a, 0xab, 0x7c, 0x2f, 0x34}};
  int hr = XGameRuntimeInitialize();
  if (hr) return (u64)(u32)hr;
  Object *title = 0;
  hr = QueryApiImpl(&provider, &iid, (void **)&title);
  if (hr || !title) return (u64)(u32)hr | 1ull << 40;
  typedef int (*GetTitleId)(Object *, u32 *);
  return (u64)(u32)((GetTitleId)title->vtbl[3])(title, id);
}

int DllMain(void *h, u32 r, void *x) {
  (void)h;
  (void)r;
  (void)x;
  return 1;
}
