/* A title DLL for the arm64 host: a TLS directory (hand-made, as the build has no CRT) with a callback, a
 * static thread-local variable with an initial value, and a DllMain; every callback logs its reason.
 * Own code; built like hleprobe.c. */
#define X __declspec(dllexport)
typedef unsigned long long u64;
typedef unsigned u32;

/* What the CRT's tlssup.c provides: the index word and the start and end of the TLS template. */
int _tls_index;
__attribute__((section(".tls"))) char _tls_start = 0;
__attribute__((section(".tls$ZZZ"))) char _tls_end = 0;

static volatile u32 events[32];
static volatile u32 n_events;
static void log_event(u32 what) { events[n_events++] = what; }

__thread u32 tls_counter = 40;

static void tls_callback(void *h, u32 reason, void *reserved) {
  (void)h;
  (void)reserved;
  log_event(0x100 | reason);
}
void (*const tls_callbacks[])(void *, u32, void *) = {tls_callback, 0};

typedef struct {
  u64 start, end, index, callbacks;
  u32 zero_fill, characteristics;
} TlsDirectory;
const TlsDirectory _tls_used = {(u64)&_tls_start, (u64)&_tls_end, (u64)&_tls_index, (u64)tls_callbacks, 0, 0};

int DllMain(void *h, u32 reason, void *reserved) {
  (void)h;
  (void)reserved;
  log_event(0x200 | reason);
  return 1;
}

/* This thread's counter, one more each call: 41 on a thread that has not called it before. */
X u32 probe_tls_bump(void) { return ++tls_counter; }
/* A thread start routine: bumps this thread's counter and adds the parameter. */
X u32 probe_thread_main(u64 param) { return probe_tls_bump() + (u32)param; }
/* The log so far: its length, and the entries copied out. */
X u32 probe_events(u32 *out) {
  for (u32 i = 0; i < n_events; i++) out[i] = events[i];
  return n_events;
}
