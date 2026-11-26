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
        victory_rt::thread_init(block, lo as *mut u8, hi as *mut u8);
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
        let binding = victory_rt::Binding::Typed(address as *const _, sig);
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
