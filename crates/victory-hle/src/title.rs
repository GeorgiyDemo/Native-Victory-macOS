//! The two crossings between our runtime and title code that need care on
//! the native macOS path (title code is Windows x64 with GS = the TEB; ours
//! is System V with GS = the host's TSD). On Windows and Wine both are
//! plain calls and nothing here does anything.
//!
//! * [`expose`]: an object (first word = method table) is being handed to
//!   the title. On macOS x86-64 its table word is replaced by a bridged copy
//!   (every slot a bridge stub). `this` stays the object's address, so our
//!   methods find their state as before; the cost is that Rust code must
//!   not call the object's methods through the table word afterwards (ours
//!   never does outside tests, which never expose).
//! * [`call`]: calls a callback pointer the title gave us. If it points into
//!   a mapped title image it goes through `call_win64`; a function of ours
//!   (a Rust callback, as tests and internal completions use) is called
//!   directly.
//!
//! On macOS two hosts implement them: under Rosetta (x86-64) the GS-swapping
//! bridge of `victory_loader::native`; in an arm64 process (the recompiled
//! title, `victory-rt`) the `arm64` module below, where title code is lifted
//! C and a crossing is a runtime import thunk or `rt_call_guest`.

/// Gives the object at `this`, whose first word is a method table of `len`
/// slots, a table the title can call. Idempotent.
///
/// # Safety
/// `this` is a live object of ours, writable, with such a first word.
#[inline]
pub unsafe fn expose<T>(this: *const T, len: usize, area: &'static str) -> *const T {
    #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
    // SAFETY: forwarded.
    unsafe {
        victory_loader::native::expose_object(this.cast(), len, area)
    };
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    // SAFETY: forwarded.
    unsafe {
        arm64::expose_object(this.cast(), len, area)
    };
    #[cfg(not(target_os = "macos"))]
    let _ = (len, area);
    this
}

/// Calls the `extern "system"` function at `f` with integer or pointer
/// `args` (up to 6), returning `rax`. Narrow results (`HRESULT`, `bool`)
/// are the caller's to truncate.
///
/// # Safety
/// `f` is a function of that signature, ours or the title's.
pub unsafe fn call(f: usize, args: &[u64]) -> u64 {
    #[cfg(target_os = "macos")]
    if is_title_code(f) {
        // SAFETY: forwarded.
        return unsafe { call_win64(f, args) };
    }
    let a = |i: usize| args.get(i).copied().unwrap_or(0);
    // SAFETY: forwarded; unused trailing arguments are ignored by the callee.
    unsafe {
        let f: unsafe extern "system" fn(u64, u64, u64, u64, u64, u64) -> u64 = std::mem::transmute::<
            usize,
            unsafe extern "system" fn(u64, u64, u64, u64, u64, u64) -> u64,
        >(f);
        f(a(0), a(1), a(2), a(3), a(4), a(5))
    }
}

// The crossings, one name per operation, implemented by the host the process is.
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
pub(crate) use victory_loader::native::{
    call_win64, declare_direct, declare_win64, host_callbacks, is_title_code,
};
/// A callable entry for the host function at `address`, as the binder would put in an IAT slot.
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
pub(crate) fn stub_for(address: usize) -> Option<usize> {
    victory_loader::native::bridge::stub_for(address).ok()
}
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) use arm64::{call_win64, declare_win64, host_callbacks, is_title_code, stub_for};

/// The arm64 host: title code is recompiled (`victory-rt`) and runs in this process. There is no GS
/// to swap and no argument remapping in a bridge stub: the runtime marshals a lifted call into one of
/// our functions from its signature string ([`crate::abi`]), and enters lifted code with `rt_call_guest`.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub mod arm64 {
    use std::cell::Cell;
    use std::collections::{HashMap, HashSet};
    use std::sync::Mutex;

    use victory_loader::native::teb;

    use crate::abi;

    fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
        m.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Makes the calling thread's TEB (the loader's, with its static TLS) the one `gs:` reads in lifted
    /// code, and the thread's guest stack the TEB's. Runs once per thread, before its first guest call.
    pub fn attach_thread() {
        thread_local!(static ATTACHED: Cell<bool> = const { Cell::new(false) });
        if ATTACHED.get() {
            return;
        }
        let block = teb::current();
        // SAFETY: the TEB block, whose stack base and limit the loader set.
        let (hi, lo) = unsafe {
            (
                block.add(teb::OFF_STACK_BASE).cast::<usize>().read(),
                block.add(teb::OFF_STACK_LIMIT).cast::<usize>().read(),
            )
        };
        // SAFETY: the loader's TEB and its leaked guest stack live as long as the thread.
        unsafe { victory_rt::thread_init(block, lo as *mut u8, hi as *mut u8) };
        ATTACHED.set(true);
    }

    /// Calls the Windows x64 function `f` of the title.
    ///
    /// # Safety
    /// `f` is a lifted function taking `args.len()` integer or pointer arguments.
    pub unsafe fn call_win64(f: usize, args: &[u64]) -> u64 {
        attach_thread();
        victory_rt::call_guest(f as u64, args, 0).0
    }

    /// Lets the loader (thread starts, TLS callbacks, `DllMain`) enter recompiled code. Call once, after the
    /// runtime is loaded.
    pub fn install() {
        victory_loader::native::guest::install(call_win64);
        victory_loader::native::guest::install_resume(resume_context);
        victory_loader::native::guest::install_callback_frame(callback_frame);
        victory_rt::install_fault_hook(on_fault);
    }

    // ---- exceptions: the Windows CONTEXT over the lifted `Cpu`, faults, and the two raw imports ----

    use victory_loader::native::seh::context::{self as seh_context, Context, ExceptionRecord};
    use victory_rt::{Cpu, reg};

    /// Raw thunks (functions of the lifted `Cpu`): their addresses, to tell `stub_for`.
    static RAW: Mutex<Option<HashSet<usize>>> = Mutex::new(None);

    /// The address to serve for an import that reads the caller's registers: `stub_for` makes it a `Raw` thunk.
    pub fn raw_export(f: unsafe extern "C" fn(*mut Cpu)) -> usize {
        let address = f as usize;
        lock(&RAW).get_or_insert_default().insert(address);
        address
    }

    /// A context from the lifted registers, with the `rip` and `rsp` the caller gives.
    fn context_of(cpu: &Cpu, rip: u64, rsp: u64) -> Context {
        let mut c = Context::zeroed();
        c.context_flags = seh_context::CONTEXT_FULL;
        c.gpr = cpu.r;
        c.gpr[seh_context::RSP] = rsp;
        c.rip = rip;
        for (i, x) in cpu.x.iter().enumerate() {
            *c.xmm_mut(i) = *x;
        }
        c.set_mxcsr(0x1F80);
        c.eflags = 0x246;
        (c.seg_cs, c.seg_ds, c.seg_es, c.seg_fs, c.seg_gs, c.seg_ss) = (0x33, 0x2B, 0x2B, 0x53, 0x2B, 0x2B);
        c
    }

    /// `RtlCaptureContext(rcx: CONTEXT *)`: the caller's registers as of the call (lifted code spills at
    /// every call), `rip` the return address and `rsp` as after the return.
    ///
    /// # Safety
    /// Called by the runtime with the `Cpu` of a lifted call.
    pub unsafe extern "C" fn capture_context(cpu: *mut Cpu) {
        // SAFETY: the Cpu of the call; `rcx` is the title's CONTEXT.
        unsafe {
            let c = &*cpu;
            let rsp = c.r[reg::RSP];
            let ctx = context_of(c, (rsp as *const u64).read_unaligned(), rsp + 8);
            (c.r[reg::RCX] as *mut Context).write_unaligned(ctx);
        }
    }

    /// `RaiseException(rcx: code, edx: flags, r8d: count, r9: arguments)`: dispatches the exception from the
    /// caller's context. Ends in a landing pad or the unhandled path; a handler that continues execution of
    /// a software exception is not supported (the dispatcher resumes at landing pads only).
    ///
    /// # Safety
    /// Called by the runtime with the `Cpu` of a lifted call.
    pub unsafe extern "C" fn raise_exception(cpu: *mut Cpu) {
        // SAFETY: the Cpu of the call; the arguments are the title's.
        unsafe {
            let c = &*cpu;
            let rsp = c.r[reg::RSP];
            let ret = (rsp as *const u64).read_unaligned();
            let mut ctx = context_of(c, ret, rsp + 8);
            let mut rec = ExceptionRecord::new(c.r[reg::RCX] as u32, c.r[reg::RDX] as u32, ret as usize);
            let count = (c.r[reg::R8] as usize).min(15);
            rec.number_parameters = count as u32;
            let args = c.r[reg::R9] as *const usize;
            if !args.is_null() {
                std::ptr::copy_nonoverlapping(args, rec.information.as_mut_ptr(), count);
            }
            victory_loader::native::seh::raise_record(&raw mut ctx, &raw mut rec)
        }
    }

    /// Continues at an unwound context: its registers into the runtime's frame for the landing pad.
    unsafe fn resume_context(ctx: &Context, ret: u64) -> ! {
        let mut xmm = [[0u8; 16]; 16];
        for (i, x) in xmm.iter_mut().enumerate() {
            *x = *ctx.xmm(i);
        }
        // SAFETY: an unwound context at a landing pad, per the loader.
        unsafe { victory_rt::resume(&ctx.gpr, &xmm, ctx.rip, ret) }
    }

    /// The guest frame a callback was made from (`rt_callback_context`), for the unwinder.
    fn callback_frame(slot: u64) -> Option<victory_loader::native::guest::CallbackFrame> {
        let f = victory_rt::callback_frame(slot)?;
        Some(victory_loader::native::guest::CallbackFrame { gpr: f.gpr, xmm: f.xmm, rip: f.rip, rsp: f.rsp })
    }

    #[repr(C)]
    struct DlInfo {
        file: *const std::ffi::c_char,
        base: *mut std::ffi::c_void,
        symbol: *const std::ffi::c_char,
        address: *mut std::ffi::c_void,
    }

    unsafe extern "C" {
        fn dladdr(address: *const std::ffi::c_void, info: *mut DlInfo) -> i32;
    }

    /// The RVA of the lifted function (`f_<rva>`, or the body `fb_<rva>` of one with landing pads) holding
    /// the host instruction at `pc`.
    fn lifted_function_at(pc: u64) -> Option<u32> {
        let mut info = DlInfo {
            file: std::ptr::null(),
            base: std::ptr::null_mut(),
            symbol: std::ptr::null(),
            address: std::ptr::null_mut(),
        };
        // SAFETY: a valid out pointer; the name is a NUL-terminated string owned by dyld.
        let name = unsafe {
            if dladdr(pc as *const _, &mut info) == 0 || info.symbol.is_null() {
                return None;
            }
            std::ffi::CStr::from_ptr(info.symbol).to_str().ok()?
        };
        let name = name.trim_start_matches('_');
        let hex = name.strip_prefix("fb_").or_else(|| name.strip_prefix("f_"))?;
        u32::from_str_radix(hex, 16).ok()
    }

    /// A fault in lifted code (`victory_rt::install_fault_hook`): an exception in the faulting function.
    ///
    /// Lifted code keeps its registers in host registers between calls, so what is exact at a fault is the
    /// last spill: the function's state at entry, or at its latest call (the return address on its stack is
    /// in the function), or, in a function with landing pads, at the last instruction that could fault
    /// (the lifter records it in `Cpu::rip`). The context is that state. A function that faults after its
    /// prologue and before any call unwinds as if at entry.
    unsafe extern "C" fn on_fault(f: *const victory_rt::Fault) {
        // SAFETY: the runtime's record, and the Cpu it names.
        let (fault, run) = unsafe { (&*f, &*(*f).run) };
        // An integer divide error the lifted code found itself (`rt_de`: SIGFPE with FPE_INTDIV or FPE_INTOVF): the
        // registers were spilled right before and `addr` is the instruction, so the context is exact.
        let divide = fault.signo == 8 && matches!(fault.code, 1 | 2);
        let (rip, sp) = if divide {
            (fault.addr, run.r[reg::RSP])
        } else {
            let Some((base, begin)) = victory_rt::image_of_pc(fault.pc)
                .and_then(|base| Some((base, base + lifted_function_at(fault.pc)? as usize)))
            else {
                eprintln!("VIctory: fault at host {:#x} outside any lifted function", fault.pc);
                std::process::abort();
            };
            let end = victory_loader::native::seh::unwind::lookup(begin).map_or(begin, |(_, e)| {
                // SAFETY: an entry of an image we mapped.
                base + unsafe { (*e).end } as usize
            });
            let rsp = run.r[reg::RSP];
            // SAFETY: the guest stack as of the last spill: the return address (or entry state) is at rsp.
            let word = unsafe { (rsp as *const u64).read_unaligned() } as usize;
            if (begin..end).contains(&(run.rip as usize)) {
                (run.rip, rsp)
            } else if (begin..end).contains(&word) {
                (word as u64, rsp + 8)
            } else {
                (begin as u64, rsp)
            }
        };
        let mut ctx = context_of(run, rip, sp);
        let (code, write) = match fault.signo {
            4 => (seh_context::EXCEPTION_ILLEGAL_INSTRUCTION, 0),
            5 => (seh_context::EXCEPTION_BREAKPOINT, 0),
            8 if fault.code == 1 => (seh_context::EXCEPTION_INT_DIVIDE_BY_ZERO, 0),
            8 if fault.code == 2 => (seh_context::EXCEPTION_INT_OVERFLOW, 0),
            8 => (seh_context::EXCEPTION_FLT_DIVIDE_BY_ZERO, 0),
            // A data abort says in bit 6 of its syndrome whether it was a write.
            _ => (seh_context::EXCEPTION_ACCESS_VIOLATION, usize::from(fault.esr & 0x40 != 0)),
        };
        let mut rec = ExceptionRecord::new(code, 0, rip as usize);
        if code == seh_context::EXCEPTION_ACCESS_VIOLATION {
            rec.number_parameters = 2;
            rec.information[..2].copy_from_slice(&[write, fault.addr as usize]);
        }
        // SAFETY: a context on this thread's guest stack, valid until the thread resumes.
        unsafe { victory_loader::native::seh::raise_record(&raw mut ctx, &raw mut rec) }
    }

    /// Whether `address` is code of the title (the recompiled image, or an image the loader mapped).
    pub fn is_title_code(address: usize) -> bool {
        victory_rt::contains(address as u64)
            || victory_loader::native::guest::is_title_code(address)
    }

    /// Floats need no declaring here: the signature says where they are.
    pub fn declare_win64(_address: usize) {}

    /// The default signature of a host function with none recorded: the Windows integer arguments.
    const DEFAULT_SIG: &str = abi::METHOD_SIG;

    /// The thunk by which lifted code calls the host function at `address`, made on first use and kept for
    /// the life of the process: the same stub for every caller. The signature is the one recorded for the
    /// address ([`abi::sig_of`]), else the one a method owner noted, else eight integers.
    pub fn stub_for(address: usize) -> Option<usize> {
        // Nothing to hand it to (unit tests of the runtime call functions directly).
        if !victory_rt::is_loaded() {
            return Some(address);
        }
        static THUNKS: Mutex<Option<HashMap<usize, usize>>> = Mutex::new(None);
        if let Some(&t) = lock(&THUNKS).get_or_insert_default().get(&address) {
            return Some(t);
        }
        let sig = known_sig(address).unwrap_or(DEFAULT_SIG);
        let name = format!("host@{address:#x}");
        let binding = if lock(&RAW).get_or_insert_default().contains(&address) {
            // SAFETY: registered by `raw_export` as a function of the lifted `Cpu`.
            victory_rt::Binding::Raw(unsafe {
                std::mem::transmute::<usize, unsafe extern "C" fn(*mut victory_rt::Cpu)>(address)
            })
        } else {
            victory_rt::Binding::Typed(address as *const _, sig)
        };
        let thunk = match victory_rt::import_thunk(&name, binding) {
            Ok(t) => t as usize,
            Err(e) => {
                tracing::error!(
                    address = format_args!("{address:#x}"),
                    sig,
                    "cannot make an import thunk: {e:#}"
                );
                return None;
            }
        };
        lock(&THUNKS).get_or_insert_default().insert(address, thunk);
        Some(thunk)
    }

    /// The imports of a title image as `victory_rt::Host::load` takes them: each (DLL, name) resolved as
    /// the Rosetta host's binder does (`HleResolver`: our registry, then the POSIX host), bound to the
    /// function's address and its signature string. With the names nobody serves, as `dll!name`. Imports
    /// by ordinal are reported unserved (the runtime binds by name).
    pub fn rt_imports<'a>(
        resolver: &crate::HleResolver,
        modules: impl IntoIterator<Item = &'a victory_pe::ImportModule>,
    ) -> (Vec<victory_rt::Import>, Vec<String>) {
        use victory_loader::{ImportResolver, Resolution};
        let (mut found, mut missing) = (Vec::<victory_rt::Import>::new(), vec![]);
        for module in modules {
            for entry in &module.entries {
                let victory_pe::ImportSymbol::Name { name, .. } = &entry.symbol else {
                    missing.push(format!("{}!{}", module.dll, entry.symbol));
                    continue;
                };
                if found.iter().any(|i| i.name == name) {
                    continue;
                }
                match resolver.resolve(&module.dll, &entry.symbol) {
                    Resolution::Found { address, .. } => found.push(victory_rt::Import {
                        name: Box::leak(name.clone().into_boxed_str()),
                        binding: victory_rt::Binding::Typed(
                            address as *const _,
                            known_sig(address).unwrap_or(DEFAULT_SIG),
                        ),
                    }),
                    Resolution::Missing => missing.push(format!("{}!{name}", module.dll)),
                }
            }
        }
        (found, missing)
    }

    /// What an unbound import's thunk does, per `dll!symbol`: the policy and a call count.
    static UNSERVED: Mutex<Option<HashMap<String, (victory_loader::UnimplementedPolicy, u64)>>> =
        Mutex::new(None);

    /// The `Raw` handler behind every unserved import: it asks the runtime which import it is in.
    unsafe extern "C" fn unserved(cpu: *mut victory_rt::Cpu) {
        let name = victory_rt::current_import().unwrap_or_default();
        let (policy, calls) = {
            let mut sites = lock(&UNSERVED);
            let site = sites
                .get_or_insert_default()
                .entry(name.clone())
                .or_insert((victory_loader::UnimplementedPolicy::ReturnZero, 0));
            site.1 += 1;
            *site
        };
        match policy {
            victory_loader::UnimplementedPolicy::ReturnZero => {
                if calls == 1 {
                    tracing::warn!(import = %name, "unimplemented import called, returning 0");
                }
                // SAFETY: the Cpu of the call, handed to us by the runtime.
                unsafe { (*cpu).r[victory_rt::reg::RAX] = 0 };
            }
            victory_loader::UnimplementedPolicy::Trap => {
                eprintln!("VIctory: unimplemented import {name} called");
                std::process::abort();
            }
        }
    }

    /// Fills every IAT slot of `image` (delay-load slots too), as the Rosetta host's `bind_imports` does:
    /// what `resolver` serves gets the runtime's thunk for it (our functions, by their signatures; a title
    /// DLL's export is the lifted function's own address), and what it cannot, by name or by ordinal, a
    /// thunk that logs and acts on `policy` (`ReturnZero`: `rax = 0`; `Trap`: abort). Needs the runtime
    /// loaded and the image attached (`victory_rt::open`, `attach_image`).
    pub fn bind_imports(
        image: &victory_loader::native::NativeImage,
        resolver: &dyn victory_loader::ImportResolver,
        policy: victory_loader::UnimplementedPolicy,
    ) -> anyhow::Result<victory_loader::BindReport> {
        use victory_loader::{MissingImport, Provenance, Resolution};
        let info = image.info();
        let mut report = victory_loader::BindReport::default();
        let mut slots = Vec::with_capacity(info.import_count());
        for module in info.imports.iter().chain(&info.delay_imports) {
            for entry in &module.entries {
                let address = match resolver.resolve(&module.dll, &entry.symbol) {
                    Resolution::Found {
                        address,
                        provenance: Provenance::Title,
                    } => {
                        report.title += 1;
                        address
                    }
                    Resolution::Found {
                        address,
                        provenance,
                    } => {
                        match provenance {
                            Provenance::Hle => report.hle += 1,
                            _ => report.host += 1,
                        }
                        stub_for(address).ok_or_else(|| {
                            anyhow::anyhow!("no thunk for {}!{}", module.dll, entry.symbol)
                        })?
                    }
                    Resolution::Missing => {
                        let name = format!("{}!{}", module.dll, entry.symbol);
                        lock(&UNSERVED)
                            .get_or_insert_default()
                            .insert(name.clone(), (policy, 0));
                        report.missing.push(MissingImport {
                            dll: module.dll.clone(),
                            symbol: entry.symbol.to_string(),
                        });
                        victory_rt::import_thunk(&name, victory_rt::Binding::Raw(unserved))? as usize
                    }
                };
                slots.push((entry.iat_slot_rva, address));
            }
        }
        image.write_slots(&slots)?;
        tracing::info!(
            hle = report.hle,
            host = report.host,
            missing = report.missing.len(),
            "native imports bound"
        );
        Ok(report)
    }

    /// Entry points for host functions the title calls through pointers it was handed (the methods of a
    /// COM object), one per target in the same order. Same thunks as everywhere else.
    pub fn host_callbacks(targets: &[(usize, bool)]) -> std::io::Result<Vec<usize>> {
        targets
            .iter()
            .map(|&(target, _)| {
                stub_for(target)
                    .ok_or_else(|| std::io::Error::other("no import thunk for a method"))
            })
            .collect()
    }

    /// The recorded signature of the function at `address`. Method-table slots have none of their own; the
    /// owners of the few that take a float by value (the default, eight integers, gets those wrong) say so.
    fn known_sig(address: usize) -> Option<&'static str> {
        static NOTED: std::sync::Once = std::sync::Once::new();
        NOTED.call_once(|| {
            crate::modules::note_methods();
            for (address, sig) in victory_gpu::object::method_sigs() {
                abi::record(address, sig);
            }
        });
        abi::sig_of(address)
    }

    /// Extra slots after the last real one of an exposed table: a title that calls a method we never listed
    /// would read past the table. These answer `E_NOTIMPL` and log.
    const SPARE_SLOTS: usize = 64;

    extern "system" fn spare_slot() -> i32 {
        tracing::warn!("title called a method past the end of one of our tables");
        0x8000_4001_u32 as i32 // E_NOTIMPL
    }

    /// Raw table to its copy, and the set of copies.
    static TABLES: Mutex<Option<(HashMap<usize, usize>, HashSet<usize>)>> = Mutex::new(None);

    /// The copy of the method table at `raw` that has thunks for slots, made on first use: `len` slots then
    /// [`SPARE_SLOTS`] spare ones. The copy is leaked.
    ///
    /// # Safety
    /// `raw` is a table of `len` function addresses.
    pub unsafe fn exposed_table(raw: *const *const (), len: usize) -> Option<*const *const ()> {
        if let Some(&c) = lock(&TABLES).get_or_insert_default().0.get(&(raw as usize)) {
            return Some(c as *const *const ());
        }
        let spare = stub_for(crate::fn_addr!(spare_slot))?;
        let mut copy = Vec::with_capacity(len + SPARE_SLOTS);
        for i in 0..len {
            // SAFETY: slot `i` of a table of `len`.
            let method = unsafe { raw.add(i).read() } as usize;
            copy.push(stub_for(method)?);
        }
        copy.resize(len + SPARE_SLOTS, spare);
        let copy = Box::leak(copy.into_boxed_slice())
            .as_ptr()
            .cast::<*const ()>();
        let mut tables = lock(&TABLES);
        let (by_raw, copies) = tables.get_or_insert_default();
        by_raw.insert(raw as usize, copy as usize);
        copies.insert(copy as usize);
        Some(copy)
    }

    /// Whether the method table at `table` is one of our copies.
    pub fn is_bridged(table: *const *const ()) -> bool {
        lock(&TABLES)
            .get_or_insert_default()
            .1
            .contains(&(table as usize))
    }

    /// The copy of the method table at `raw`, if it has one.
    pub fn bridged_copy_of(raw: *const *const ()) -> Option<*const *const ()> {
        lock(&TABLES)
            .get_or_insert_default()
            .0
            .get(&(raw as usize))
            .map(|&c| c as *const *const ())
    }

    /// Gives the object at `this` (first word: its method table of `len` slots) a table the title can call,
    /// in place. `this` stays the object's address, so our methods find their state as before. Idempotent;
    /// does nothing before an image is loaded.
    ///
    /// # Safety
    /// `this` is a live object of ours whose first word is such a table pointer, in writable memory.
    pub unsafe fn expose_object(this: *const (), len: usize, area: &'static str) {
        use std::sync::atomic::{AtomicPtr, Ordering};
        if !victory_rt::is_loaded() {
            return;
        }
        // SAFETY: the first word is a pointer-sized table pointer.
        let word = unsafe { &*this.cast::<AtomicPtr<*const ()>>() };
        let raw = word.load(Ordering::Acquire).cast_const();
        if is_bridged(raw) {
            return;
        }
        // SAFETY: forwarded.
        match unsafe { exposed_table(raw, len) } {
            Some(copy) => word.store(copy.cast_mut(), Ordering::Release),
            None => tracing::error!(area, "cannot make a method table callable"),
        }
    }

    /// The GPU crate's hooks into the above, for `victory_gpu::object::set_bridge`.
    pub fn gpu_bridge() -> victory_gpu::object::Bridge {
        victory_gpu::object::Bridge {
            is_bridged,
            copy_of: bridged_copy_of,
            expose: expose_object,
            declare_win64,
        }
    }
}
