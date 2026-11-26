//! A minimal native arm64 host for statically recompiled title code (`victory-recomp`).
//!
//! The lifted C units plus the runtime (`rt_host.c`) are built into a dylib. This crate loads that dylib
//! into the current process, maps the PE image where the lifted code expects it (`IMG`), applies the base
//! relocations, writes an import thunk into every IAT slot, and runs exports natively on the calling
//! thread. There is no Rosetta, no `gs:` segment and no calling-convention bridge: a lifted call through an
//! IAT slot lands in the runtime's dispatcher, which marshals the Windows x64 arguments out of the lifted
//! `Cpu` struct and calls an ordinary `extern "C"` Rust function. See `docs/recompiler.md`, "Native arm64 host".
//!
//! One host per process: the runtime keeps its state in the dylib's globals, and import functions reach
//! the callback entry point through [`call_guest`].

use std::alloc::{Layout, alloc_zeroed};
use std::collections::HashMap;
use std::ffi::{CString, c_char, c_void};
use std::path::Path;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Context, Result, bail};
use victory_recomp::image::Image;

unsafe extern "C" {
    fn dlopen(path: *const c_char, mode: i32) -> *mut c_void;
    fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
    fn dlerror() -> *const c_char;
}

/// Mirror of the runtime's `Cpu` (rt.h): the architectural state lifted functions share.
#[repr(C)]
#[derive(Clone)]
pub struct Cpu {
    pub r: [u64; 16],
    pub x: [[u8; 16]; 16],
    pub xh: [[u8; 16]; 16],
    /// What `gs:` addresses: the thread's TEB.
    pub gs_base: u64,
}

/// Index of a general register in `Cpu::r`.
pub mod reg {
    pub const RAX: usize = 0;
    pub const RCX: usize = 1;
    pub const RDX: usize = 2;
    pub const RSP: usize = 4;
    pub const R8: usize = 8;
    pub const R9: usize = 9;
}

type RtFn = Option<unsafe extern "C" fn(*mut Cpu)>;

/// Mirror of `RtImport` (rt.h).
#[repr(C)]
struct RtImport {
    name: *const c_char,
    func: *const c_void,
    raw: RtFn,
    argc: u8,
    fmask: u16,
    smask: u16,
    ret: u8,
}

/// How an import is called.
#[derive(Clone, Copy)]
pub enum Binding {
    /// A native function with arguments by the native (AAPCS64) rules, described by a signature string:
    /// one letter per argument (`i` integer or pointer, `d` double, `f` float), then `>` and the result
    /// (`i` 64-bit, `w` 32-bit, `d`, `f`, `v` none). `"iid>d"` is `f64 (u64, u64, f64)`.
    Typed(*const c_void, &'static str),
    /// A function that takes the lifted `Cpu` and reads its own arguments.
    Raw(unsafe extern "C" fn(*mut Cpu)),
}

/// An import the host provides, by name (the DLL name is not checked in this prototype).
pub struct Import {
    pub name: &'static str,
    pub binding: Binding,
}

fn parse_sig(sig: &str) -> Result<(u8, u16, u16, u8)> {
    let (args, ret) = sig
        .split_once('>')
        .with_context(|| format!("signature {sig:?} has no `>`"))?;
    let (mut fmask, mut smask) = (0u16, 0u16);
    for (i, c) in args.chars().enumerate() {
        match c {
            'i' => {}
            'd' => fmask |= 1 << i,
            'f' => smask |= 1 << i,
            _ => bail!("bad argument letter {c:?} in {sig:?}"),
        }
    }
    let ret = match ret {
        "i" => 0,
        "w" => 1,
        "d" => 2,
        "f" => 3,
        "v" => 4,
        _ => bail!("bad result {ret:?} in {sig:?}"),
    };
    Ok((args.len() as u8, fmask, smask, ret))
}

/// Entry points of the runtime inside the dylib.
#[derive(Clone, Copy)]
struct Rt {
    call_guest: CallGuest,
    thread_init: ThreadInit,
    set_fallback: SetFallback,
    unresolved_count: UnresolvedCount,
    cpuid: Cpuid,
    bind: Bind,
}
type Bind = unsafe extern "C" fn(*const RtImport) -> u64;
type CallGuest = unsafe extern "C" fn(u64, *const u64, u32, u32, *mut u64) -> u64;
type ThreadInit = unsafe extern "C" fn(*mut u8, *mut u8, *mut u8);
type SetFallback = unsafe extern "C" fn(unsafe extern "C" fn(*mut Cpu, u64));
type UnresolvedCount = unsafe extern "C" fn() -> u64;
type Cpuid = unsafe extern "C" fn(u32, u32, *mut u32, *mut u32, *mut u32, *mut u32);

static RT: OnceLock<Rt> = OnceLock::new();

/// Calls a lifted guest function (a callback pointer the title gave us) with Windows x64 arguments; bit `i`
/// of `fmask` says argument `i` is a double given as its bit pattern. Returns `rax` and the low qword of `xmm0`.
/// Usable from import functions: the call runs below the lifted frame that called the import.
pub fn call_guest(target: u64, args: &[u64], fmask: u32) -> (u64, u64) {
    let rt = RT.get().expect("no host loaded");
    let mut x0 = 0u64;
    // SAFETY: `target` is a guest function address; the runtime resolves it or reports it unresolved.
    let rax = unsafe { (rt.call_guest)(target, args.as_ptr(), args.len() as u32, fmask, &mut x0) };
    (rax, x0)
}

static IMAGE: AtomicUsize = AtomicUsize::new(0);
static IMAGE_END: AtomicUsize = AtomicUsize::new(0);

/// Whether an image is loaded: until then there is no guest to hand a function or an object to.
pub fn is_loaded() -> bool {
    IMAGE.load(Ordering::Acquire) != 0
}

/// Whether `addr` is inside the loaded guest image: a callback pointer there is lifted code and must be
/// called through [`call_guest`]; any other address is a function of ours.
pub fn contains(addr: u64) -> bool {
    (IMAGE.load(Ordering::Acquire) as u64..IMAGE_END.load(Ordering::Acquire) as u64).contains(&addr)
}

/// The runtime's thunk for a host function: the address lifted code calls (through an IAT slot, or from a
/// pointer the title got from `GetProcAddress` or a method table). Needs a loaded host.
pub fn import_thunk(name: &str, binding: Binding) -> Result<u64> {
    let rt = RT.get().context("no host loaded")?;
    let cname = CString::new(name)?;
    let (func, raw, sig) = match binding {
        Binding::Typed(f, s) => (f, None, parse_sig(s)?),
        Binding::Raw(f) => (std::ptr::null(), Some(f), (0, 0, 0, 4)),
    };
    let desc = RtImport {
        name: cname.as_ptr(),
        func,
        raw,
        argc: sig.0,
        fmask: sig.1,
        smask: sig.2,
        ret: sig.3,
    };
    // SAFETY: desc is valid for the call; the runtime copies it and the name.
    Ok(unsafe { (rt.bind)(&desc) })
}

/// Gives the calling thread its TEB and guest stack (`stack_lo..stack_hi`): what `gs:` reads in lifted code.
/// A thread that never calls it gets a private TEB on its first guest call. Needs a loaded host.
///
/// # Safety
/// `teb` is a TEB block (self pointer at 0x30) and `stack_lo..stack_hi` a stack, both outliving the thread's
/// guest calls.
pub unsafe fn thread_init(teb: *mut u8, stack_lo: *mut u8, stack_hi: *mut u8) {
    let rt = RT.get().expect("no host loaded");
    // SAFETY: forwarded.
    unsafe { (rt.thread_init)(teb, stack_lo, stack_hi) };
}

/// A mapped image with its runtime.
pub struct Host {
    rt: Rt,
    img: *mut u8,
    size: usize,
    exports: HashMap<String, u32>,
    /// The thread environment block the calling thread's `gs:` accesses read.
    pub teb: *mut u8,
}

impl Host {
    /// Loads `dylib` (lifted units and runtime, built by the caller), maps `pe` into it and binds `imports`.
    /// Fails if the image imports something not in `imports`.
    pub fn load(dylib: &Path, pe: &[u8], imports: &[Import]) -> Result<Host> {
        let path = CString::new(dylib.to_str().context("dylib path")?)?;
        // SAFETY: plain dlopen of a library we just built.
        let lib = unsafe { dlopen(path.as_ptr(), 2) };
        if lib.is_null() {
            // SAFETY: dlerror returns a NUL-terminated string after a failed dlopen.
            let e = unsafe { std::ffi::CStr::from_ptr(dlerror()) };
            bail!("dlopen {}: {}", dylib.display(), e.to_string_lossy());
        }
        // SAFETY (for every use): each symbol is declared with exactly this signature in rt.h, and a function
        // pointer is the size of the `*mut c_void` dlsym returns.
        let sym = |n: &str| -> Result<*mut c_void> {
            let c = CString::new(n)?;
            // SAFETY: lib is a live handle.
            let p = unsafe { dlsym(lib, c.as_ptr()) };
            if p.is_null() {
                bail!("{n} not found in {}", dylib.display());
            }
            Ok(p)
        };
        // SAFETY: see above.
        let rt = unsafe {
            Rt {
                call_guest: std::mem::transmute::<*mut c_void, CallGuest>(sym("rt_call_guest")?),
                thread_init: std::mem::transmute::<*mut c_void, ThreadInit>(sym("rt_thread_init")?),
                set_fallback: std::mem::transmute::<*mut c_void, SetFallback>(sym(
                    "rt_set_fallback",
                )?),
                unresolved_count: std::mem::transmute::<*mut c_void, UnresolvedCount>(sym(
                    "rt_unresolved_count",
                )?),
                cpuid: std::mem::transmute::<*mut c_void, Cpuid>(sym("rt_cpuid")?),
                bind: std::mem::transmute::<*mut c_void, Bind>(sym("rt_import_bind")?),
            }
        };
        // SAFETY: see above.
        let init = unsafe {
            std::mem::transmute::<*mut c_void, unsafe extern "C" fn(*mut u8, u64)>(sym("rt_init")?)
        };
        if RT.set(rt).is_err() {
            bail!("a host is already loaded in this process");
        }

        // Map the image: sections at their RVAs, then base relocations.
        let image = Image::parse(pe)?;
        let size = (image.info.size_of_image as usize).next_multiple_of(0x4000);
        // SAFETY: non-zero size, 16 KiB alignment (the arm64 page size).
        let img = unsafe { alloc_zeroed(Layout::from_size_align(size, 0x4000)?) };
        if img.is_null() {
            bail!("cannot allocate {size} bytes for the image");
        }
        for s in &image.info.sections {
            let raw = pe
                .get(s.file_offset as usize..(s.file_offset + s.file_size) as usize)
                .context("section outside the file")?;
            // SAFETY: the section lies inside size_of_image.
            unsafe {
                std::ptr::copy_nonoverlapping(raw.as_ptr(), img.add(s.rva as usize), raw.len())
            };
        }
        let delta = (img as u64).wrapping_sub(image.info.image_base);
        for rva in image.dir64_relocs() {
            // SAFETY: a relocation slot inside the image.
            unsafe {
                let p = img.add(rva as usize).cast::<u64>();
                p.write_unaligned(p.read_unaligned().wrapping_add(delta));
            }
        }
        // SAFETY: img/size describe the mapping just made.
        unsafe { init(img, size as u64) };
        IMAGE.store(img as usize, Ordering::Release);
        IMAGE_END.store(img as usize + size, Ordering::Release);

        // Imports: one runtime thunk per host function, written into the IAT slots.
        let by_name: HashMap<&str, &Import> = imports.iter().map(|i| (i.name, i)).collect();
        let mut thunks: HashMap<&str, u64> = HashMap::new();
        let mut missing = vec![];
        for m in &image.info.imports {
            for e in &m.entries {
                let name = match &e.symbol {
                    victory_pe::ImportSymbol::Name { name, .. } => name.as_str(),
                    victory_pe::ImportSymbol::Ordinal(_) => {
                        missing.push(format!("{}!{}", m.dll, e.symbol));
                        continue;
                    }
                };
                let Some(imp) = by_name.get(name) else {
                    missing.push(format!("{}!{name}", m.dll));
                    continue;
                };
                let thunk = match thunks.get(name) {
                    Some(t) => *t,
                    None => {
                        let t = import_thunk(name, imp.binding)?;
                        thunks.insert(name, t);
                        t
                    }
                };
                // SAFETY: the IAT slot lies inside the image.
                unsafe {
                    img.add(e.iat_slot_rva as usize)
                        .cast::<u64>()
                        .write_unaligned(thunk)
                };
            }
        }
        if !missing.is_empty() {
            bail!("unbound imports: {}", missing.join(", "));
        }
        let exports = image
            .info
            .exports
            .iter()
            .filter_map(|e| Some((e.name.clone()?, e.rva?)))
            .collect();
        let mut host = Host {
            rt,
            img,
            size,
            exports,
            teb: std::ptr::null_mut(),
        };
        host.init_thread();
        Ok(host)
    }

    /// Gives the calling thread a TEB (self pointer at 0x30, stack bounds at 0x08/0x10) and a 1 MiB guest stack.
    /// The memory is leaked: a prototype has no thread teardown.
    pub fn init_thread(&mut self) {
        const STACK: usize = 1 << 20;
        // SAFETY: fresh zeroed allocations of non-zero size.
        let (teb, lo) = unsafe {
            (
                alloc_zeroed(Layout::from_size_align(0x2000, 0x1000).unwrap()),
                alloc_zeroed(Layout::from_size_align(STACK, 0x1000).unwrap()),
            )
        };
        // SAFETY: both allocations are large enough for these fields.
        unsafe {
            let hi = lo.add(STACK);
            teb.add(0x30).cast::<u64>().write(teb as u64);
            teb.add(0x08).cast::<u64>().write(hi as u64);
            teb.add(0x10).cast::<u64>().write(lo as u64);
            (self.rt.thread_init)(teb, lo, hi);
        }
        self.teb = teb;
    }

    /// Guest address of an export.
    pub fn export(&self, name: &str) -> Result<u64> {
        let rva = *self
            .exports
            .get(name)
            .with_context(|| format!("no export {name}"))?;
        Ok(self.img as u64 + rva as u64)
    }

    /// Calls an export with integer arguments; returns `rax`.
    pub fn call(&self, name: &str, args: &[u64]) -> Result<u64> {
        Ok(call_guest(self.export(name)?, args, 0).0)
    }

    /// Calls an export with floating-point arguments (bit `i` of `fmask`); returns `rax` and `xmm0`'s low qword.
    pub fn call_ex(&self, name: &str, args: &[u64], fmask: u32) -> Result<(u64, u64)> {
        Ok(call_guest(self.export(name)?, args, fmask))
    }

    /// Installs the handler for dispatch targets the lifter has no function for. It must behave like the
    /// callee: set `rax` and pop the return address (`rsp += 8`).
    pub fn set_fallback(&self, f: unsafe extern "C" fn(*mut Cpu, u64)) {
        // SAFETY: f has the RtFallback signature.
        unsafe { (self.rt.set_fallback)(f) };
    }

    pub fn unresolved_count(&self) -> u64 {
        // SAFETY: plain getter.
        unsafe { (self.rt.unresolved_count)() }
    }

    /// The runtime's CPUID answer.
    pub fn cpuid(&self, leaf: u32, sub: u32) -> [u32; 4] {
        let mut r = [0u32; 4];
        let [a, b, c, d] = &mut r;
        // SAFETY: four valid out pointers.
        unsafe { (self.rt.cpuid)(leaf, sub, a, b, c, d) };
        r
    }

    pub fn image(&self) -> (*mut u8, usize) {
        (self.img, self.size)
    }
}
