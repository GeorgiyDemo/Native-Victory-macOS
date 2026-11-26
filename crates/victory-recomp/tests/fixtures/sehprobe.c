/* Structured exception handling in a lifted image: `__try`/`__except`/`__finally` (a language handler,
 * `__C_specific_handler`, imported by the image and served by the host), a vectored handler, a software
 * exception, and the unwind functions called directly. Own code; built like hleprobe.c with `-fms-extensions`. */
#define X __declspec(dllexport)
#define I __declspec(dllimport)
typedef unsigned long long u64;
typedef unsigned u32;

/* The first import of the image is this one: the personality routine of every `__try` in it. */
I int __C_specific_handler(void *, u64, void *, void *);
I void *AddVectoredExceptionHandler(u32 first, long (*handler)(void *));
I void RaiseException(u32 code, u32 flags, u32 count, const u64 *args);
I void RtlCaptureContext(void *context);
I void *RtlLookupFunctionEntry(u64 pc, u64 *image_base, void *history);
I void *RtlVirtualUnwind(u32 kind, u64 image_base, u64 pc, void *entry, void *context, void **handler_data, u64 *frame, void *pointers);

static volatile u32 vectored_calls;
static volatile u32 vectored_code;
static volatile u64 filter_address;
static volatile u32 finally_runs;

/* A vectored handler sees the exception first and declines (EXCEPTION_CONTINUE_SEARCH). `optnone` keeps a frame,
 * so it has `.pdata`: the lifter finds functions by it, and a frameless leaf only passed by address
 * (`lea rdx, [rip+f]`) is not a seed. */
__attribute__((optnone, noinline)) static long vectored(void *pointers) {
  vectored_calls++;
  vectored_code = **(u32 **)pointers;
  return 0;
}
X u32 probe_vectored_install(void) { return AddVectoredExceptionHandler(1, vectored) != 0; }
X u32 probe_vectored_seen(u32 *code) {
  *code = vectored_code;
  return vectored_calls;
}

__attribute__((noinline)) static void store_through(volatile int *p) { *p = 1; }

/* Fault in a callee that has no handler: the frame of this function takes it. */
X u32 probe_fault_in_callee(volatile int *p) {
  u32 r = 0;
  __try {
    store_through(p);
    r = 1;
  } __except (1) {
    r = 2;
  }
  return r;
}

/* Fault in the guarded code itself. Written in assembly with the unwind data by hand: clang drops a `__try`
 * body that has no call in it, as it takes a plain store not to throw (MSVC's /EHa does not). The scope table
 * is the one `__try { *p = 5; } __except (1) { 12 }` has: begin, end, filter (1: execute), landing pad. */
__asm__(
    ".intel_syntax noprefix\n"
    ".text\n"
    ".globl probe_fault_inline\n"
    ".seh_proc probe_fault_inline\n"
    "probe_fault_inline:\n"
    ".seh_handler __C_specific_handler, @except\n"
    "sub rsp, 40\n"
    ".seh_stackalloc 40\n"
    ".seh_endprologue\n"
    "inline_try:\n"
    "mov dword ptr [rcx], 5\n"
    "inline_end:\n"
    "mov eax, 11\n"
    "jmp inline_done\n"
    "inline_pad:\n"
    "mov eax, 12\n"
    "inline_done:\n"
    "add rsp, 40\n"
    "ret\n"
    ".seh_handlerdata\n"
    ".long 1\n"
    ".long inline_try@IMGREL, inline_end@IMGREL, 1, inline_pad@IMGREL\n"
    ".text\n"
    ".seh_endproc\n"
    ".section .drectve\n"
    ".ascii \" -export:probe_fault_inline\"\n"
    ".text\n"
    ".att_syntax\n");

/* The filter sees the exception: its code, and for an access violation the address it touched. */
static int filter(unsigned code, void **info) {
  void *record = info[0];
  filter_address = ((u64 *)record)[0x28 / 8]; /* ExceptionInformation[1] */
  return code == 0xC0000005 ? 1 : 0;
}
X u64 probe_filter(volatile int *p, u32 *code) {
  u64 r = 0;
  __try {
    store_through(p);
  } __except (filter(_exception_code(), (void **)_exception_info())) {
    r = 20;
  }
  *code = 0xC0000005;
  return r << 32 | (filter_address & 0xFFFFFFFF);
}

/* A software exception: RaiseException, with a code and a parameter. */
X u32 probe_raise(void) {
  u32 r = 30;
  __try {
    u64 args[1] = {0x1234};
    RaiseException(0xE0000001, 0, 1, args);
    r = 31;
  } __except (_exception_code() == 0xE0000001) {
    r = 32;
  }
  return r;
}

/* A termination handler between the fault and the handler: it runs as the stack unwinds. */
X u32 probe_finally(volatile int *p) {
  u32 r = 40;
  __try {
    __try {
      store_through(p);
    } __finally {
      finally_runs++;
    }
  } __except (1) {
    r = 41 + finally_runs;
  }
  return r;
}

/* Not handled by this function's filter (it declines), so an outer function's takes it. */
static u32 inner_declines(volatile int *p) {
  u32 r = 50;
  __try {
    store_through(p);
  } __except (0) {
    r = 51;
  }
  return r;
}
X u32 probe_outer(volatile int *p) {
  u32 r = 60;
  __try {
    r = inner_declines(p);
  } __except (1) {
    r = 62;
  }
  return r;
}

/* The unwind functions, called directly: capture, find the function's entry, unwind one frame. */
__attribute__((noinline)) static u32 unwind_one(void) {
  /* A CONTEXT (0x4D0 bytes): the registers start at 0x78 (rax, rcx, rdx, rbx, rsp, ...) and rip is at 0xF8. */
  u64 ctx[0x4D0 / 8] __attribute__((aligned(16)));
  RtlCaptureContext(ctx);
  u64 pc = ctx[0xF8 / 8], rsp = ctx[0x78 / 8 + 4], base = 0, frame = 0;
  void *entry = RtlLookupFunctionEntry(pc, &base, 0);
  if (!entry || base == 0) return 0;
  void *data = 0;
  RtlVirtualUnwind(0, base, pc, entry, ctx, &data, &frame, 0);
  /* The caller's context: its pc is outside this function and its stack above ours. */
  return ctx[0xF8 / 8] != pc && ctx[0x78 / 8 + 4] > rsp ? 1 : 2;
}
X u32 probe_unwind(void) { return unwind_one(); }

int DllMain(void *h, u32 r, void *x) {
  (void)h;
  (void)r;
  (void)x;
  return 1;
}
