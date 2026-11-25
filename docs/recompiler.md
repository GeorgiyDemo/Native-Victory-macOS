# Static recompiler: x86-64 title to native ARM64

This is the design for `crates/victory-recomp`, the foundation of the native macOS port. The title's own code is x86-64 and runs under Rosetta 2 today. The recompiler turns it into ARM64 ahead of time, so a native host (memory, threads, loader, HLE, no Wine) and a Metal backend can run it directly. No title code or bytes are in the repo. Everything below that mentions the title is a count.

## What exists now

| Piece | Where | State |
|---|---|---|
| Analysis pass (`victory-recomp analyze <pe>`) | `src/analyze.rs`, `src/image.rs` | Works on the title in about 0.6 s. |
| Function discovery beyond .pdata | `src/discover.rs` | Works. Adds leaf functions from call targets, relocations and exports. |
| Lifter, x86-64 to C | `src/lift.rs`, `src/rt.h` | Integer ALU, shifts, rotates, mul/div, loads/stores, push/pop, call/ret/jmp/jcc, jump tables, setcc/cmovcc, bsf/bsr/popcnt, SSE scalar and 128-bit packed float/logic, conversions, compares. |
| Prototype host | `src/rt_host.c`, `src/harness.h` | Dispatch by RVA over a generated table. A stand-in for the native host. |
| Coverage report (`victory-recomp coverage <pe>`) | `src/lib.rs` | Counts functions the lifter accepts. |
| Tests | `tests/e2e.rs`, `tests/diff.rs` | See "Test results". |

CLI: `victory-recomp analyze|coverage|lift|flatten|exports`. `lift <pe> <out.c>` writes one C unit for the whole image.

## Title statistics (game_scarlett_master.exe, 120 MB, x86-64 PE32+)

Measured with `victory-recomp analyze` and `coverage`. The decoder is iced-x86 1.22.0 (fetched from crates.io, it was not in the offline registry; the network worked).

Functions and code
- 299,257 `.pdata` entries, of which 515 are chained cold parts or funclets (UNW_FLAG_CHAININFO).
- 6 functions hit a decode failure.
- `.pdata` covers 92.35 MB of the 99.88 MB `.text` (92.5%). The rest is leaf functions with no unwind info, thunks and padding.
- Discovery finds 104,873 more function starts that `.pdata` misses (60,229 from direct calls and tail jumps, 46,023 from relocations, with overlap). Total after discovery: 403,953 functions, not counting chained parts.
- 22,435,434 instructions in `.pdata` functions (average 4.12 bytes). Median function is 117 bytes, p99 is 2,829 bytes, the largest is 129,632 bytes (28,252 instructions).
- One exec section, `.text`, not writable. No W+X section.

Control flow (`.pdata` functions)
- Direct calls 1,441,955. 653,669 of them (45%) target a non-.pdata address, all inside `.text`. That is where the leaf functions are.
- Indirect calls: 16,795 through a register, 5,037 through a rip-relative slot (IAT or a global), 87,326 through other memory (vtables and function-pointer tables). Total 109,158.
- Direct jmp 584,187, of which 64,178 leave the function (tail calls and cold-part jumps). Conditional jumps 1,974,994. `ret` 271,528.
- Indirect jmp: 7,028 through a register, 1,205 through memory.
- Jump tables recovered: 2,641 tables with 26,611 entries. 4,387 `jmp reg` have no recovered table. Some of those are tail calls through a register. The rest need better recovery (see below).
- Relocations: 598,282 DIR64. 529,045 point into `.text`; 160,693 at a `.pdata` function start and 368,352 elsewhere (leaf functions, switch cases, vtable entries). So 88% of the pointer data in the image is code pointers.
- 2 TLS callbacks, 1 export, entry point is in `.pdata`.

Instruction set (instruction counts)
- The code is almost entirely VEX-encoded. Legacy SSE is nearly absent: SSE 1,793, SSE2 448.
- AVX 2,839,856 (12.7%). AVX2 113,317. FMA 200,924. F16C 2,966. BMI1 58,398. BMI2 12,403. POPCNT 442. LZCNT 294. MOVBE 195. AES 13. CMPXCHG16B 56. SSE4A 17.
- 256-bit (ymm) instructions: 245,113. 128-bit and scalar xmm: 2,863,879.
- AVX-512: 0. x87: 0. MMX: none seen.
- Top mnemonics: mov 29.8%, call 6.9%, lea 5.3%, test 4.6%, je 4.5%, pop 4.2%, cmp 4.0%, push 3.7%, vmovaps 3.1%, add 3.0%.
- `lock` prefix: 51,251. `xchg` with memory: 4,067. `rep` string ops: 6. `syscall`/`int`: 3. `cpuid`/`rdtsc`/`xgetbv`: 11.
- GS-prefixed accesses: 53,055 (the TEB and TLS). No FS.

Self-modifying code and JIT
- No writable executable section. No rip-relative store into an executable section (0).
- The only suspicious import is `VirtualAlloc`. No `VirtualProtect`, `FlushInstructionCache` or `WriteProcessMemory`. The title does not look like it generates code at run time. Treat any code the title maps at run time as unknown until the host logs it (see indirect calls).

Lifter coverage today (`coverage`): 257,261 of 403,953 functions (63.7%) contain only instructions the lifter accepts. This is "accepted", not "proven". The first blocker per function, top entries: `vmovaps` 49,886; `vxorps` 18,876; `vmovss` 15,098; lock/rep prefix 11,182; `vmovups` 9,280; `andn` 7,191; `bt` 5,370; indirect `jmp reg` without a table 4,803; `bextr` 4,635; `vmovdqa` 3,086; `vmovd` 2,993. So the next gains are almost all AVX forms (VEX scalar and 128/256-bit moves and logic), BMI, `bt`, and atomics.

## Design and hard problems

### Output form: C compiled by clang, not LLVM IR, not direct ARM64

Choice: generate C and compile with clang (`-O2`, `-ffp-contract=off`). The lifter emits one `void f_<rva>(Cpu *c)` per guest function.

Reasons:
- Clang gives register allocation, scheduling and dead flag elimination for free. Direct ARM64 emission would mean writing all of that, and the title has 404k functions and 22M instructions.
- C is readable and diffable. A wrong lift can be read, and the differential tests run the same C. LLVM IR text works equally well for optimisation, but remill/McSema-style IR makes every instruction a call to a semantics function plus a large state struct. That compiles slowly and needs heavy optimisation passes to recover registers. Plain C with locals avoids that.
- C compile time is the known cost. 22M instructions will give several million lines of C. Plan: split into translation units of about 2,000 functions, compile in parallel, link with ThinLTO off. If that is still too slow, the same lifter can print LLVM IR directly with `inkwell` or textual IR. The `Cpu`/flag model and runtime ABI do not change, so this is a back-end swap, not a redesign.
- Direct ARM64 stays open as a later optimisation for hot functions only.

Risks of C: undefined behaviour must be avoided (shifts, signed overflow; the lifter masks and uses unsigned arithmetic), and LLVM may delete NaN fix-ups written as floating-point selects. The runtime does those on bit patterns (see SSE).

Model: guest registers are C locals (`r0..r15` as `u64`, `x0..x15` as 128-bit unions). Locals are loaded at entry and spilled around calls and returns, so a callee sees the caller's registers through `Cpu`. Guest pointers are host pointers: the image is mapped at `IMG` with relocations applied, and rip-relative operands become `IMG + rva`. The guest stack is a real memory region. `call` pushes a return address, so `[rsp+n]` argument accesses and unwinding still work. `ret` pops it and returns from the C function.

### Code discovery

Seeds, in order of reliability:
1. `.pdata` (RUNTIME_FUNCTION). On x64 it lists every function that has unwind info. It gave 299,257 entries on the title. Chained entries (UNW_FLAG_CHAININFO) are parts of a parent function. The prototype skips them as entry points, so a jump into a cold part currently goes to the dispatcher. Next step: merge them into the parent so the jump becomes a `goto`.
2. Exports, the entry point and TLS callbacks.
3. DIR64 relocations that point into code. These are vtables, callback tables and switch targets. On the title that is 529k pointers.
4. Direct call and tail-jump targets. `.pdata` misses leaf functions (no stack, no calls), and on the title those are 45% of call targets.
5. Jump tables. Recognised at `jmp reg`: a rip-relative `lea` of the base, a scaled 4-byte `movsxd`/`mov` load, then `add base; jmp`. The table size comes from the preceding `cmp idx, N; ja default`. Both clang style (table-relative) and MSVC style (image-relative with the displacement in the load) are handled. Targets must lie in the same function.

Extents for functions not in `.pdata` come from a linear walk that stops at the first terminator (`ret`, `jmp`, `int3`, `ud2`) that is past every forward branch target, and never crosses the next known start.

Known gaps (the 4,387 `jmp reg` with no table, and the 11% of relocations that land mid-function): switch tables with 8- or 16-bit entries, tables with a bias, tables loaded in a different block than the `jmp`, and `jmp reg` used as a computed tail call. Approach: a backward slice per `jmp reg` over the CFG (not a fixed window), and fall back to the dispatcher for the rest. Anything not resolved statically goes through the runtime dispatcher.

### Indirect calls and jumps

All indirect transfers call `rt_dispatch(Cpu*, u64 target)`. The host owns the table: a sorted `(rva, fn)` array generated with the unit, searched by binary search. For hot sites a per-site inline cache (last target and function) comes first. A hash map is the alternative if profiling shows the search is hot.

An address the table does not know goes to `rt_unresolved`. Ordered fallbacks:
1. Address outside the image: an HLE import. The host recognises the IAT slot value and runs the HLE function with the calling convention below.
2. Address inside `.text` but not a known start (a missed function, or a jump into the middle of one): run it through the interpreter fallback. For the prototype this aborts and logs the address. The log is the discovery feed: re-run the lifter with these as extra seeds.
3. Address in memory the title mapped at run time (JIT): Rosetta fallback only if a separate x86-64 helper process is accepted, otherwise the interpreter. The static analysis shows no JIT signs, so this should stay unused.

The fallback is the safety net that makes partial coverage runnable: unlifted functions can run in the interpreter while the lifted ones run natively.

### Lazy EFLAGS

Flags are not computed at each instruction. A flag-setting instruction records `Fl {kind, size, a, b, r, c}` in a local. A consumer (`jcc`, `setcc`, `cmovcc`, `adc`, `sbb`, `inc/dec` carry preservation) derives CF, ZF, SF, OF or PF from that record. With the kind known at compile time, clang folds each use to a single compare. Kinds: ADD, ADC, SUB, SBB, LOGIC, INC, DEC, and RAW. RAW holds explicit bits for shifts, rotates, mul, `ucomis*`, `bsf`. Rotates keep ZF/SF/PF of the previous state. Across a join where kinds differ, clang emits a switch on the kind, which is correct and still cheap.

Not modelled: AF (used only by BCD instructions, and `pushf`/`lahf` are not lifted yet), and the undefined flags of shifts, rotates and mul. The tests compare only the defined flags. `pushf`, `popf` and `lahf` need a function that materialises the full record into a flags word.

### Memory ordering: x86 TSO versus ARM

x86 is TSO: loads are not reordered with loads, stores are not reordered with stores, and a store is not reordered with an earlier load. ARM64 is weak. Lifted code that uses plain `ldr`/`str` for guest memory can see reordering the x86 code never expected, for example a lock-free queue or a flag-then-data publish. The title has 51k `lock` instructions and 4k `xchg mem`, so it uses atomics, and plain loads and stores used with them as ordering carry the same assumption.

Options:
1. Apple's hardware TSO mode. Apple silicon implements it, and the kernel turns it on per thread for Rosetta-translated processes. A study cited below measured it within about 3% of Arm mode on most applications. I found no public macOS API for a native arm64 process to turn it on. On Linux there was a 2024 `prctl` patch proposal for per-process TSO, and Linux maintainers objected. I could not confirm a macOS entitlement, a mach call or a sysctl for native processes, and I did not try private interfaces. Treat this as unavailable until proven. If it exists, it is the best option: zero change to the generated code, and it costs only what the hardware charges. Risks: it is probably private or entitlement-gated, so it may break on an OS update or block distribution; it is per thread, so every thread, including ones the host creates, must opt in; and its exact behaviour is not documented.
2. LDAR/STLR (acquire loads and release stores) on every guest memory access. This gives load-load, load-store and store-store ordering, which is all TSO guarantees (x86 lets a store pass a later load, and so does LDAR/STLR, so it matches). Cost: ARM's acquire/release loads on Apple silicon are cheap relative to a full fence, but they block some addressing modes (no register offset, no pair loads) and stop the compiler merging or reordering accesses. Expect a large slowdown on memory-heavy code (about 22M instructions, 30% of them `mov`). Risk: clang treats `ldar`/`stlr` as atomics, which defeats vectorisation and hoisting.
3. Plain accesses plus fences only around atomics and at specific points. `lock`-prefixed instructions become `__atomic_*` with seq_cst, and `xchg mem` is implicitly locked. Cost: near zero for ordinary code. Risk: wrong for code that relies on TSO ordering between plain accesses without a `lock` (a spinlock released with a plain `mov`, a Dekker-style publish, `volatile` flags). MSVC's `volatile` on x64 is acquire/release, so plain `mov` carries ordering in the title's own synchronisation primitives.

Decision: build with option 3 as the baseline, because it matches the C11-style code a game uses. Make the memory-access emission a single function in the lifter with a mode switch, so any of the three can be selected per build. Run with option 1 if a way to enable it is found (ask the native-host agent to probe it; the answer changes the plan). Add option 2 only for functions identified as synchronisation code (those that contain `lock`/`xchg`/`pause` or sit in a spin loop, and the functions that call them), not globally. This needs a test: a multi-threaded stress test (message passing and store buffering litmus tests) run lifted versus under Rosetta. That test is not written yet.

### SSE, AVX and AVX2 to NEON

The title is VEX-first: 2.84M AVX instructions, 245k of them 256-bit, 200k FMA. Plan:
- 128-bit and scalar ops map to NEON one for one (`addps` to `fadd v.4s`, `fmadd`, `fsqrt`, `fcvtzs` with the x86 "integer indefinite" fix-up).
- 256-bit ops (ymm) are two NEON q registers. Model `ymm` as a pair of 128-bit halves in `Cpu`. Every VEX 128-bit write zeroes the upper half, and `vzeroupper` (49,563 uses) zeroes all upper halves. Many can be dropped when the upper halves are provably unused.
- Shuffles, blends and horizontal ops (`vshufps`, `vblendps`, `vhaddps`, `vpermilps`, `vperm2f128`, `vbroadcastss`) have no direct NEON form for all immediates. Use `tbl`/`ext`/`zip`/`uzp` sequences, or the C compiler's `__builtin_shufflevector` with the immediate known at compile time. The generated C should use clang vector extensions (`__attribute__((vector_size))`), not intrinsics, so clang picks the NEON sequence.
- Comparisons: `vcmpps` predicates and `vmovmskps` need a movemask sequence (shift and `addv`), which is more expensive on NEON.
- F16C is `fcvtl`/`fcvtn`. BMI1/2 (`andn`, `bextr`, `shlx`, `rorx`, `mulx`, `bzhi`) are short integer sequences, or `bic`, `ubfx`, `lslv`, `rorv`, and `umulh`.
- Floating-point details that already matter, and are handled in `rt.h` and tested: x86 returns the "real indefinite" negative NaN (`0xFFF8...`) for invalid operations, ARM returns a positive one, so scalar and packed `add/sub/mul/div/sqrt` are fixed up on the bit pattern; `min/max` return the second operand on NaN or equal; float-to-int conversions return the "integer indefinite" value on overflow where ARM saturates.
- Known divergence, not fixed: signalling NaNs are ordered differently by x86 and ARM when both inputs are NaN, and the differential tests do not generate them. Denormals and rounding match by default (the title does not set MXCSR flush modes in the lifted subset). MXCSR itself is not modelled; `ldmxcsr`/`stmxcsr` will need the ARM FPCR mapping.

### x87

0 x87 instructions in the title. Do not implement x87 now. If a dependency DLL needs it, use `long double` helpers (ARM `long double` is 128-bit, not 80-bit) or the interpreter fallback.

### SEH and unwinding via .pdata/.xdata

The title is built with unwind info for every non-leaf function (299k RUNTIME_FUNCTION entries), so C++ exceptions and SEH are possible, but the lifted code has no x64 unwind data to walk. Options and decision:
- Keep the guest stack faithful (real return addresses and frames, as now). Then the host can implement `RtlVirtualUnwind` and `RtlLookupFunctionEntry` itself by reading the original `.pdata`/`.xdata` against guest addresses, and emulate `RtlUnwindEx`, `RtlRaiseException` and the language handlers (`__C_specific_handler`, `__CxxFrameHandler`) as host code that resumes by calling a lifted continuation. The return addresses on the guest stack are the RVAs of the original call sites, so a lookup works unchanged.
- To resume at a handler or a catch funclet, the lifter needs an entry point at each landing pad. Make every catch and `__except` target a function start in the table (funclets are chained or have their own `.pdata`) and call it with the frame pointer, as the Windows ABI does.
- A lifted C function cannot be unwound by `longjmp` through its locals safely, since locals are registers saved in `Cpu` only at calls. At every call site registers are already spilled, so unwinding through a call boundary and resuming in a handler with `Cpu` set from the unwound context is consistent. This is why spilling at calls matters.
- Hardware faults (`#DE`, access violations) become `SIGFPE`/`SIGSEGV`/`SIGBUS` handlers on the host. The handler maps the faulting host PC back to a guest RVA (a table from the lifter, or only at function granularity, with registers from the last spill) and raises the guest exception. Precise per-instruction state at a fault is not available in the C model. Compromise: only fault-prone instructions (`div`, memory accesses with guard pages) are decorated, and exact state is guaranteed only at function granularity. Most titles only need access violations for stack growth and for crash handling.

Not yet written: any of this. The prototype traps on `#DE` with a message.

### Calling convention into the host's HLE

HLE functions are host functions behind the title's import slots, and the title also calls the kernel and system DLL exports through them. Proposed contract (the native host agent should confirm or counter-propose):
- Lifted code and the host share one `Cpu` struct, and every transition passes `Cpu *c`. An HLE entry has the signature `void hle_fn(Cpu *c)`. It reads Windows x64 arguments from `rcx, rdx, r8, r9` and `[rsp+0x28...]` (the stack after the return address and 32 bytes of shadow space), `xmm0..3` for floating-point arguments by position, and writes the result to `rax` or `xmm0`. This is exactly what a lifted callee sees, so lifted code and HLE are interchangeable and a function can be switched between them per symbol.
- A typed fast path can be added later: the host registers `u64 (*)(u64,u64,u64,u64)` for a function with at most four integer arguments, and the lifter emits a direct native call at that site, skipping `Cpu`. The generator needs a signature table for that.
- The host fills the IAT with the address of a thunk per import, `IMG`-relative so a call through the slot goes to `rt_dispatch`, which sees an address outside lifted code and finds the HLE entry in the host's table.
- Callbacks from the host into the title (window procedures, thread start routines, completion callbacks) go through `rt_call_win64(rva, a, b, c, d, ...)`, which sets up a fresh or the current guest stack and runs the dispatcher.
- Callee-saved registers (`rbx, rbp, rdi, rsi, r12-r15, xmm6-15`) are preserved by the lifted code naturally, because lifted functions restore them from the stack like the originals. HLE host code must preserve them in `Cpu`.

### GS-based TEB and TLS

53,055 instructions use a `gs:` prefix (the title reads the TEB: `gs:[0x30]` self pointer, `gs:[0x58]` TLS array pointer, `gs:[0x60]` PEB, `gs:[0x8]`/`[0x10]` stack bounds, thread id, last error). The lifter emits `(u8*)rt_gs + offset`. `rt_gs` is a host-provided per-thread pointer to a TEB-shaped block. Because lifted code runs on ordinary host threads, `rt_gs` must be a thread-local variable (`__thread`), loaded once per function (it is not re-read inside one, so a function must not migrate threads, which holds). The prototype uses a plain global.

Needed in the TEB block: `ExceptionList` unused on x64, `StackBase`/`StackLimit` (stack probes and stack-size queries), `Self` at `0x30`, `ProcessEnvironmentBlock` at `0x60`, `ThreadLocalStoragePointer` at `0x58` (implicit TLS: `_tls_index` and the per-module TLS array, filled from the image's TLS directory), `LastErrorValue` at `0x68`, and the Xbox-specific fields the title reads. The analysis shows 2 TLS callbacks, which the host runs at thread creation.

### What I need from the native host's ABI

1. `rt_dispatch(Cpu*, u64 target)` implementation, with the unresolved fallbacks above, and `rt_trap`.
2. `IMG`: the image mapped at one base, relocations applied by the host loader (598k DIR64 entries; 529k point into code). The lifter emits `IMG + rva` for rip-relative operands, so the image must keep its section layout. The lifted code does not need the image to be executable (it never executes the bytes) but needs it readable and writable (data, vtables).
3. IAT fill: each import slot holds an address the dispatcher recognises as an HLE entry. Imports called through `call [rip+slot]` are 5,037 sites.
4. HLE entry point shape: `void hle(Cpu *c)`, Windows x64 convention as above. Optional typed fast path.
5. TEB/TLS: a thread-local `rt_gs` per guest thread, with the fields listed; thread start must set up `rsp`, push a return address and call through `rt_dispatch`.
6. Exceptions: `RtlLookupFunctionEntry`/`RtlVirtualUnwind`/`RtlUnwindEx` as host code over the original `.pdata`; fault handlers translating host signals to guest exceptions; a way to call a landing pad with a given `Cpu`.
7. Threads and atomics: guest threads are host threads; keep the memory-model choice (TSO mode if any exists) a host decision per thread. Report if the host finds an API for hardware TSO.
8. A stack with at least 1 MB, 16-byte aligned, 8 bytes below alignment at function entry (as after a `call`) plus 32 bytes of shadow space above it, as `rt_call_win64` does.
9. The host should log unresolved targets; the recompiler consumes that log as extra function seeds.
10. Guest address space: guest pointers are host pointers (64-bit). Anything the title stores as an address of its own memory works only if the host allocates at addresses that fit what the title expects (the title uses 64-bit pointers, but may assume high-bit patterns, for example a console memory map). Check this before loading real data.

## Test results

Run with `cargo test -p victory-recomp`. They run on macOS with Rosetta 2 and clang and skip with a message otherwise. They are compiled but not run by the Windows clippy gate.

- `tests/e2e.rs::testpe_recompiled_output_matches_x86`: builds `tests/fixtures/testpe.c` into a Windows DLL with `tools/macos/gnullvm-cc.sh` (clang to `x86_64-w64-windows-gnu`, lld), reads its `.pdata` and exports, lifts all 20 functions (0 stubbed, jump table and tail call included) and compiles the C for arm64. `tests/fixtures/e2e_driver.c` runs 400 rounds over 18 exported functions (7,200 output lines). The x86-64 flavour executes the DLL's own bytes under Rosetta 2, and the arm64 flavour runs the lifted code. The outputs are byte-identical.
- `tests/diff.rs::lifted_instruction_classes_match_x86`: 318 assembled snippets in 23 classes (alu-reg, alu-mem-src, alu-mem-dst, alu-imm, incdec, shift, mul, div, mov-ext, bitscan, setcc, cmov, jcc, jmp (jump table), call (direct and indirect), stack, helper, sse-scalar, sse-packed, sse-compare, sse-mov, sse-bitwise, cvt). Each runs 300 pseudo-random inputs (zeros, boundaries, equal pairs, NaN, infinity, denormals) as x86-64 under Rosetta 2 and as lifted arm64, and the result, defined flags and xmm lanes must match bit for bit. Integer snippets cover 8/16/32/64-bit sizes where valid.

Bugs these tests found while building the lifter, now fixed: a double pop of the return address, 32-bit shifts with count 0 not zero-extending, `bsf`/`bsr` with zero source writing the destination, LLVM deleting the negative-NaN fix-up, and function end detection stopping before a forward branch target.

Not tested: AVX/VEX, `lock`, GS access (no x86 oracle for `gs:` on a Mac), threads, exceptions, `pushf/popf/lahf`, 8/16-bit `div`, and floating-point signalling NaNs.

## Next steps, in order

1. VEX scalar and 128-bit forms with the `Cpu` ymm halves (`vmovaps`, `vxorps`, `vmovss`, `vmovups`, `vmovdqa`, `vmovd/q`, `vbroadcastss`, `vzeroupper`, scalar arithmetic and compares, `vcvt*`). The first-blocker list shows this is about 100k of the 147k unlifted functions.
2. Atomics and prefixes: `lock add/sub/inc/dec/cmpxchg/xadd`, `xchg mem`, `rep movs/stos`, `bt/bts/btr`, `andn/bextr/shlx/shrx/mulx/bzhi/rorx/tzcnt/lzcnt/movbe`, then `pushf/popf/lahf`.
3. Chained `.pdata` merge, better `jmp reg` recovery, and a function-extent check against `.pdata` ends.
4. Split output into parallel translation units; measure clang time and binary size on the full title. This tells whether C output stays viable or the LLVM IR back-end is needed.
5. Hook up the native host: real `rt_dispatch` and HLE table, loader with relocations, `rt_gs` TEB, IAT fill. Run the first lifted title function end to end.
6. Memory model: ask the host agent to probe for hardware TSO on native macOS; write the litmus tests; implement option 3 as a mode switch plus option 2 for synchronisation functions.
7. 256-bit and AVX2 via clang vector extensions, FMA, shuffle/blend/permute, F16C.
8. Exceptions: host `RtlVirtualUnwind` over `.pdata`, landing-pad entry points, fault signal mapping.
9. Interpreter fallback for unresolved targets, so partial lifts run.
10. Differential testing against Rosetta on whole real functions: run the same function with random register state in both and compare, once the title's code is allowed on this machine (outside the repo).

Sources for the TSO notes: [Apple silicon TSO study (ASPLOS 2026)](https://www.cs.tau.ac.il/~mad/publications/asplos2026-TSO.pdf) and [the 2024 arm64 TSO prctl thread](https://lists.infradead.org/pipermail/linux-arm-kernel/2024-May/926327.html). Neither says whether macOS lets a native process enable it.
