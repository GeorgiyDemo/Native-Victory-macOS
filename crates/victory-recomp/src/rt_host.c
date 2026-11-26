/* Prototype host for lifted code: dispatch by RVA over the generated table.
 * The real host (native macOS agent) replaces rt_unresolved with HLE / fallback. */
#include "rt.h"

u8 *IMG;
u8 *rt_gs;

void rt_trap(const char *why) { fprintf(stderr, "rt_trap: %s\n", why); abort(); }
void rt_unresolved(Cpu *c, u64 t) { (void)c; fprintf(stderr, "unresolved guest target %#llx (rva %#llx)\n", (unsigned long long)t, (unsigned long long)(t - (u64)IMG)); abort(); }

void rt_dispatch(Cpu *c, u64 target) {
  u64 rva = target - (u64)IMG;
  unsigned lo = 0, hi = rt_table_n;
  while (lo < hi) {
    unsigned mid = (lo + hi) / 2;
    if (rt_table[mid].rva < rva) lo = mid + 1; else hi = mid;
  }
  if (lo < rt_table_n && rt_table[lo].rva == rva) { rt_table[lo].fn(c); return; }
  rt_unresolved(c, target);
}

/* Prototype answers: no CPU features, XCR0 = x87+SSE+AVX, a monotonic counter. */
void rt_cpuid(u32 leaf, u32 sub, u32 *a, u32 *b, u32 *c, u32 *d) { (void)leaf; (void)sub; *a = *b = *c = *d = 0; }
u64 rt_xgetbv(u32 index) { (void)index; return 7; }
u64 rt_rdtsc(void) { static u64 t; return t += 100; }
