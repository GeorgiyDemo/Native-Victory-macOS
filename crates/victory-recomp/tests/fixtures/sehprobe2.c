/* Exceptions in lifted code, second round (own code, built like sehprobe.c with -fms-extensions):
 * integer divides that fault, handlers that continue execution, and exceptions that cross a host function (the
 * image calls `host_apply`, a function of the host, which calls a guest function back). */
#define X __declspec(dllexport)
#define I __declspec(dllimport)
typedef unsigned long long u64;
typedef unsigned u32;

I int __C_specific_handler(void *, u64, void *, void *);
I void RaiseException(u32 code, u32 flags, u32 count, const u64 *args);
/* The host calls cb(arg) as a callback and returns its result. */
I u64 host_apply(u64 (*cb)(u64), u64 arg);

static volatile u32 seen_code;
static volatile u32 finally_runs;
volatile int av_scratch;

/* --- integer divide errors --- */
__attribute__((noinline)) static u32 divide(u32 a, u32 b) { return a / b; }
X u32 probe_div(u32 b) {
  u32 r = 0;
  __try {
    r = divide(100, b);
  } __except (seen_code = _exception_code(), 1) {
    r = 7777;
  }
  return r;
}
X u32 probe_div_code(void) { return seen_code; }
__attribute__((noinline)) static int sdivide(int a, int b) { return a / b; }
X int probe_idiv_overflow(volatile int *a, volatile int *b) {
  int r = 0;
  __try {
    r = sdivide(*a, *b);
  } __except (seen_code = _exception_code(), 1) {
    r = 8888;
  }
  return r;
}

/* --- continuing execution: written in assembly with the scope tables by hand (clang drops a `__try` body
 * without a call, and the filters here work on the CONTEXT) --- */
__attribute__((used)) static int fix_divisor(void **info, u64 frame) {
  (void)frame;
  u64 *ctx = info[1];
  ctx[0x80 / 8] = 5; /* rcx */
  return -1;         /* EXCEPTION_CONTINUE_EXECUTION */
}
__attribute__((used)) static int fix_pointer(void **info, u64 frame) {
  (void)frame;
  u64 *ctx = info[1];
  ctx[0x80 / 8] = (u64)&av_scratch; /* rcx */
  return -1;
}
/* 100 / ecx, with a filter that makes ecx 5: 20. */
__asm__(
    ".intel_syntax noprefix\n"
    ".text\n"
    ".globl probe_div_continue\n"
    ".seh_proc probe_div_continue\n"
    "probe_div_continue:\n"
    ".seh_handler __C_specific_handler, @except\n"
    "sub rsp, 40\n"
    ".seh_stackalloc 40\n"
    ".seh_endprologue\n"
    "div_try:\n"
    "mov eax, 100\n"
    "xor edx, edx\n"
    "div ecx\n"
    "div_end:\n"
    "jmp div_done\n"
    "div_pad:\n"
    "mov eax, 4444\n"
    "div_done:\n"
    "add rsp, 40\n"
    "ret\n"
    ".seh_handlerdata\n"
    ".long 1\n"
    ".long div_try@IMGREL, div_end@IMGREL, fix_divisor@IMGREL, div_pad@IMGREL\n"
    ".text\n"
    ".seh_endproc\n"
    /* *rcx = 5, with a filter that points rcx at a valid slot: then the slot is returned. */
    ".globl probe_av_continue\n"
    ".seh_proc probe_av_continue\n"
    "probe_av_continue:\n"
    ".seh_handler __C_specific_handler, @except\n"
    "sub rsp, 40\n"
    ".seh_stackalloc 40\n"
    ".seh_endprologue\n"
    "av_try:\n"
    "mov dword ptr [rcx], 5\n"
    "av_end:\n"
    "mov eax, dword ptr [rip + av_scratch]\n"
    "jmp av_done\n"
    "av_pad:\n"
    "mov eax, 5555\n"
    "av_done:\n"
    "add rsp, 40\n"
    "ret\n"
    ".seh_handlerdata\n"
    ".long 1\n"
    ".long av_try@IMGREL, av_end@IMGREL, fix_pointer@IMGREL, av_pad@IMGREL\n"
    ".text\n"
    ".seh_endproc\n"
    ".section .drectve\n"
    ".ascii \" -export:probe_div_continue -export:probe_av_continue\"\n"
    ".text\n"
    ".att_syntax\n");

/* --- exceptions across a host function --- */
/* A frameless leaf that nothing calls by name: only `lea rdx, [rip + cb_store]` reaches it. */
__attribute__((noinline)) static u64 cb_store(u64 p) {
  *(volatile int *)p = 1;
  return 3;
}
/* The callback faults; the caller above the host function takes it. */
X u32 probe_callback_fault(u64 p) {
  u32 r = 0;
  __try {
    r = (u32)host_apply(cb_store, p);
  } __except (1) {
    r = 70;
  }
  return r;
}
static u64 cb_raise(u64 code) {
  u64 args[1] = {0x55};
  RaiseException((u32)code, 0, 1, args);
  return 4;
}
/* A software exception raised in the callback, with a termination handler in the caller between. */
X u32 probe_callback_raise(void) {
  u32 r = 0;
  __try {
    __try {
      r = (u32)host_apply(cb_raise, 0xE0000003);
    } __finally {
      finally_runs++;
    }
  } __except (_exception_code() == 0xE0000003) {
    r = 71 + finally_runs;
  }
  return r;
}
/* No exception: the callback's result comes back through the host function. */
X u32 probe_callback_plain(u64 p) { return (u32)host_apply(cb_store, p); }

int DllMain(void *h, u32 r, void *x) {
  (void)h;
  (void)r;
  (void)x;
  return 1;
}
