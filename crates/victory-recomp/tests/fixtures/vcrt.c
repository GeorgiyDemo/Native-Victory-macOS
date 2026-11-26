/* A stand-in for vcruntime140.dll: the pieces of the MSVC C++ exception machinery a lifted image needs, written as the
 * real ones behave on x64 (own code; built for the x86_64-pc-windows-msvc target, like the C++ images that use it):
 *   _CxxThrowException(object, ThrowInfo*)       raises 0xE06D7363 with the throw's four parameters; `throw;` re-raises
 *   __CxxFrameHandler3(record, frame, ctx, dc)   the language handler of every C++ function: `FuncInfo` from the
 *                                                handler data, the state from the IP-to-state map, the try block and the
 *                                                catch that match by type name
 * The flow is the real one: in the search phase a frame with a matching catch asks RtlUnwindEx to unwind to it, with
 * `catch_entry` as the target. The unwind runs the destructor funclets of the frames below (this handler again, with
 * EXCEPTION_UNWINDING) and then enters `catch_entry` as a call made from the catching frame, which calls the catch
 * funclet, takes the continuation address it returns and has RtlRestoreContext continue there with the frame's
 * registers. Not supported (the real one does these): a try inside a catch block, copy constructors and destructors of
 * thrown objects, catch by base class, noexcept checks. */
#define X __declspec(dllexport)
#define I __declspec(dllimport)
typedef unsigned long long u64;
typedef unsigned u32;
typedef int i32;
typedef unsigned char u8;

I void RaiseException(u32 code, u32 flags, u32 count, const u64 *args);
I void RtlUnwindEx(void *frame, void *target_ip, void *record, void *value, void *context, void *history);
I void RtlRestoreContext(void *context, void *record);
I void *RtlPcToFileHeader(void *pc, void **base);

/* Called when an exception leaves a destructor during unwinding. */
X __declspec(noreturn) void __std_terminate(void) { __builtin_trap(); }

/* A log the images share, so a test can see in what order constructors, destructors and catch blocks ran. */
static int log_buf[64];
static int log_n;
X void vc_log(int v) {
  if (log_n < 64) log_buf[log_n++] = v;
}
X int vc_log_get(int *out) {
  for (int i = 0; i < log_n; i++) out[i] = log_buf[i];
  return log_n;
}
X void vc_log_reset(void) { log_n = 0; }

#define CODE_CXX 0xE06D7363u
#define EH_UNWINDING 0x02u
#define EH_TARGET_UNWIND 0x20u

typedef struct ER {
  u32 code, flags;
  struct ER *chain;
  void *address;
  u32 n, pad;
  u64 info[15];
} ER;
typedef struct {
  u64 control_pc, image_base;
  struct { u32 begin, end, unwind; } *entry;
  u64 frame, target_ip;
  void *ctx, *lang, *handler_data, *history;
  u32 scope, fill;
} DC;

typedef struct { u32 magic; i32 max_state, unwind; u32 n_try; i32 try_map; u32 n_ip; i32 ip_map, unwind_help, es, flags; } FuncInfo;
typedef struct { i32 to_state, action; } UnwindEntry;
typedef struct { i32 low, high, catch_high, n_catch, handlers; } TryBlock;
typedef struct { u32 adj; i32 type, obj, funclet, frame; } Catch;
typedef struct { i32 ip, state; } IpState;
typedef struct { u32 attrs; i32 unwind, fwd, catchables; } ThrowInfo;
typedef struct { i32 n; i32 type[1]; } CatchableArray;
typedef struct { u32 props; i32 type, mdisp, pdisp, vdisp, size, copy; } Catchable;
typedef struct { const void *vft; void *spare; char name[1]; } TypeDesc;

static void copy(void *dst, const void *src, u64 n) {
  for (u64 i = 0; i < n; i++) ((volatile u8 *)dst)[i] = ((const u8 *)src)[i];
}
static int same(const char *a, const char *b) {
  while (*a && *a == *b) a++, b++;
  return *a == *b;
}

/* The catches in progress: a catch block is running for `frame`, which took try block [low, high]. An exception
 * thrown from the block (or `throw;`) meets that frame again at its original call site, which is inside the try
 * block as far as the state map goes; the record remaps it to the catch block's state. */
typedef struct { u64 frame; i32 low, high, catch_high; ER rec; } CatchRec;
static CatchRec catches[8];
static int n_catches;
static CatchRec *find(u64 frame) {
  for (int i = n_catches - 1; i >= 0; i--)
    if (catches[i].frame == frame) return &catches[i];
  return 0;
}
static void drop(CatchRec *r) {
  while (n_catches && &catches[n_catches - 1] >= r) n_catches--;
}

/* What the handler hands `catch_entry`: everything the catch needs, in static storage (the handler's frame and
 * the dispatcher's are gone by then). */
typedef struct {
  void *funclet;
  u64 frame;
  void *object;
  ER rec;
  u8 ctx[1232] __attribute__((aligned(16)));
  i32 low, high, catch_high;
} Pending;
static Pending pending;

static i32 ip_state(u64 base, const FuncInfo *fi, u32 ip) {
  const IpState *m = (const IpState *)(base + fi->ip_map);
  i32 s = -1;
  for (u32 i = 0; i < fi->n_ip && (u32)m[i].ip <= ip; i++) s = m[i].state;
  return s;
}

/* Runs the destructor actions from `state` down to `target`. */
static void unwind_to(u64 base, const FuncInfo *fi, u64 frame, i32 state, i32 target) {
  const UnwindEntry *um = (const UnwindEntry *)(base + fi->unwind);
  for (int guard = 0; state != target && state >= 0 && guard < 256; guard++) {
    if (um[state].action) ((void (*)(void *, u64))(base + um[state].action))((void *)1, frame);
    state = um[state].to_state;
  }
}

/* Entered by the unwind as a call from the catching frame, rax = &pending. */
void catch_body(Pending *p) {
  CatchRec *r = &catches[n_catches++];
  r->frame = p->frame;
  r->low = p->low;
  r->high = p->high;
  r->catch_high = p->catch_high;
  copy(&r->rec, &p->rec, sizeof p->rec);
  u64 cont = ((u64 (*)(void *, u64))p->funclet)(p->object, p->frame);
  drop(r);
  *(u64 *)(p->ctx + 0xF8) = cont; /* CONTEXT.Rip */
  RtlRestoreContext(p->ctx, 0);
}
__asm__(
    ".text\n"
    ".globl catch_entry\n"
    ".seh_proc catch_entry\n"
    "catch_entry:\n"
    "subq $40, %rsp\n"
    ".seh_stackalloc 40\n"
    ".seh_endprologue\n"
    "movq %rax, %rcx\n"
    "callq catch_body\n"
    "int3\n"
    ".seh_endproc\n");
void catch_entry(void);

X int __CxxFrameHandler3(ER *rec, u64 frame, void *ctx, DC *dc) {
  if (rec->code != CODE_CXX) return 1; /* ExceptionContinueSearch */
  u64 base = dc->image_base;
  const FuncInfo *fi = (const FuncInfo *)(base + *(u32 *)dc->handler_data);
  u32 begin = dc->entry->begin;
  /* A catch funclet's own frame: the function's frame is in the funclet's home slot for rdx. */
  const TryBlock *in_try = 0;
  u64 pframe = frame;
  for (u32 t = 0; t < fi->n_try; t++) {
    const TryBlock *tb = (const TryBlock *)(base + fi->try_map) + t;
    const Catch *cs = (const Catch *)(base + tb->handlers);
    for (i32 k = 0; k < tb->n_catch; k++)
      if ((u32)cs[k].funclet == begin) {
        in_try = tb;
        pframe = *(u64 *)(frame + cs[k].frame);
      }
  }
  i32 s = ip_state(base, fi, (u32)(dc->control_pc - base));
  if (in_try) {
    /* Unwinding out of a catch block: what was constructed in it goes; the function's own frame is handled when the
     * unwind reaches it. A try inside a catch block is not supported: nothing here takes an exception. */
    if (rec->flags & EH_UNWINDING) unwind_to(base, fi, pframe, s, in_try->catch_high);
    return 1;
  }
  CatchRec *r = find(frame);
  if (r && s >= r->low && s <= r->high) s = r->catch_high;
  if (rec->flags & EH_UNWINDING) {
    unwind_to(base, fi, frame, s, (rec->flags & EH_TARGET_UNWIND) ? pending.low : -1);
    if (r) drop(r);
    return 1;
  }
  const ThrowInfo *ti = (const ThrowInfo *)rec->info[2];
  u64 tbase = rec->info[3];
  for (u32 t = 0; t < fi->n_try; t++) {
    const TryBlock *tb = (const TryBlock *)(base + fi->try_map) + t;
    if (s < tb->low || s > tb->high) continue;
    const Catch *cs = (const Catch *)(base + tb->handlers);
    for (i32 k = 0; k < tb->n_catch; k++) {
      const Catchable *hit = 0;
      if (cs[k].type == 0) {
        hit = (const Catchable *)1; /* catch (...) */
      } else {
        const CatchableArray *ca = (const CatchableArray *)(tbase + ti->catchables);
        for (i32 j = 0; j < ca->n; j++) {
          const Catchable *ct = (const Catchable *)(tbase + ca->type[j]);
          if (same(((const TypeDesc *)(base + cs[k].type))->name, ((const TypeDesc *)(tbase + ct->type))->name)) {
            hit = ct;
            break;
          }
        }
      }
      if (!hit) continue;
      /* The exception object goes where the catch variable lives, before anything is unwound. */
      if (cs[k].type != 0 && cs[k].obj) {
        if (cs[k].adj & 8) *(u64 *)(frame + cs[k].obj) = rec->info[1]; /* by reference: the thrown copy */
        else copy((void *)(frame + cs[k].obj), (void *)rec->info[1], hit->size);
      }
      pending.funclet = (void *)(base + cs[k].funclet);
      pending.frame = frame;
      pending.object = (void *)rec->info[1];
      copy(&pending.rec, rec, sizeof *rec);
      copy(pending.ctx, ctx, sizeof pending.ctx);
      pending.low = tb->low;
      pending.high = tb->high;
      pending.catch_high = tb->catch_high;
      RtlUnwindEx((void *)frame, (void *)catch_entry, &pending.rec, &pending, pending.ctx, dc->history);
      return 1;
    }
  }
  return 1;
}

/* `throw v`: the object is copied to storage that outlives the frames the unwind removes. */
static u8 pool[16][64];
static int pool_next;
X __declspec(noreturn) void _CxxThrowException(void *object, ThrowInfo *ti) {
  u64 info[4];
  if (!object && !ti) {
    /* `throw;`: the exception being handled by the innermost catch block. */
    if (!n_catches) RaiseException(CODE_CXX, 1, 0, 0);
    const ER *r = &catches[n_catches - 1].rec;
    RaiseException(r->code, 1, r->n, r->info);
  } else {
    void *base = 0;
    RtlPcToFileHeader(ti, &base);
    const CatchableArray *ca = (const CatchableArray *)((u64)base + ti->catchables);
    const Catchable *first = (const Catchable *)((u64)base + ca->type[0]);
    void *obj = pool[pool_next++ & 15];
    copy(obj, object, first->size < 64 ? first->size : 64);
    info[0] = 0x19930520;
    info[1] = (u64)obj;
    info[2] = (u64)ti;
    info[3] = (u64)base;
    RaiseException(CODE_CXX, 1, 4, info);
  }
  for (;;) {
  }
}

int DllMain(void *h, u32 r, void *x) {
  (void)h;
  (void)r;
  (void)x;
  return 1;
}
