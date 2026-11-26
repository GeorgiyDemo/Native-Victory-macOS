//! Signatures of our host functions, for the arm64 host.
//!
//! Recompiled title code calls an import with the Windows x64 argument registers in its register file, and
//! the runtime (`victory-rt`) needs to know, per function, which arguments are integers and which are
//! floats to call it as a plain AAPCS64 function. The string is `victory_rt::Binding::Typed`'s: one letter
//! per argument (`i` integer or pointer, `d` double, `f` float), `>`, then the result (`i` 64-bit, `w`
//! 32-bit, `d`, `f`, `v` none).
//!
//! Nothing is kept by hand. `build.rs` reads the declaration of every `extern "system"` and
//! `extern "efiapi"` function in `src/` and writes the strings into a table keyed by source file and
//! function name; [`fn_addr!`] is what the lookup tables write instead of `f as *const () as usize`, and
//! records the string under the address ([`sig_of`]). The same functions are `extern "efiapi"` where floats
//! must arrive in the Windows registers on x86-64 (Rosetta host) and plain C on arm64, where the signature
//! string does the remapping.
//!
//! shortcut: only functions that appear in a lookup table (`fn_addr!`) get a signature. Method tables are
//! static arrays of bare addresses; their slots are called with [`METHOD_SIG`] (sixteen integers, an integer
//! result: `this` and the Windows arguments, four in registers and the rest on the stack), right for every
//! method without a float argument; the modules that own the few with one say so (`note_methods`). A method
//! with more than eight integer arguments must declare those past the eighth as 64-bit (AAPCS64 on Apple
//! packs smaller ones into the stack slot); one with more than sixteen is a trap.

#[cfg(target_arch = "aarch64")]
mod derived {
    use std::collections::HashMap;
    use std::sync::{LazyLock, Mutex};

    include!(concat!(env!("OUT_DIR"), "/sigs.rs"));

    static BY_NAME: LazyLock<HashMap<&'static str, Vec<(&'static str, &'static str)>>> = LazyLock::new(|| {
        let mut m: HashMap<_, Vec<_>> = HashMap::new();
        for &(file, name, sig) in SIGS {
            m.entry(name).or_default().push((file, sig));
        }
        m
    });

    /// The signature string of the function `name` written in `file` (a path as `file!()` gives it).
    /// Another file's function of that name stands in when this one's declaration was not found and the
    /// name is unambiguous.
    pub fn declared(file: &str, name: &str) -> Option<&'static str> {
        // `device::configure_virtual_memory`, `api::lookup_function_entry`: the function is the last segment.
        let name = name.rsplit("::").next().unwrap_or(name);
        let name = name.trim();
        let rel = file.rsplit_once("src/").map_or(file, |(_, r)| r);
        let candidates = BY_NAME.get(name)?;
        candidates
            .iter()
            .find(|(f, _)| *f == rel)
            .or_else(|| (candidates.len() == 1).then(|| &candidates[0]))
            .map(|&(_, sig)| sig)
    }

    static NOTED: LazyLock<Mutex<HashMap<usize, &'static str>>> = LazyLock::new(Default::default);

    /// Records `sig` for the function at `address`.
    pub fn record(address: usize, sig: &'static str) {
        NOTED.lock().unwrap_or_else(|e| e.into_inner()).insert(address, sig);
    }

    /// The recorded signature of the function at `address`.
    pub fn sig_of(address: usize) -> Option<&'static str> {
        NOTED.lock().unwrap_or_else(|e| e.into_inner()).get(&address).copied()
    }

    /// Records the declared signature of function `name` of `file` under `address`, and returns the address.
    pub fn note(file: &str, name: &str, address: usize) -> usize {
        if let Some(sig) = declared(file, name) {
            record(address, sig);
        }
        address
    }
}

#[cfg(target_arch = "aarch64")]
pub use derived::{declared, note, record, sig_of};

/// What a method table's slots are called with when nothing says otherwise: the Windows x64 integer
/// arguments (four registers, then the stack), an integer result.
#[cfg(target_arch = "aarch64")]
pub const METHOD_SIG: &str = "iiiiiiiiiiiiiiii>i";

/// Elsewhere the address is all there is (x86-64 hosts need no signature).
#[cfg(not(target_arch = "aarch64"))]
#[inline]
pub fn note(_file: &str, _name: &str, address: usize) -> usize {
    address
}

/// The address of a host function, recording its signature for the arm64 host (a plain cast elsewhere).
#[macro_export]
macro_rules! fn_addr {
    ($f:expr) => {
        $crate::abi::note(file!(), stringify!($f), $f as *const () as usize)
    };
}

#[cfg(all(test, target_arch = "aarch64"))]
mod tests {
    use super::*;

    #[test]
    fn declared_signatures_follow_the_rust_types() {
        // Integers and pointers, a 32-bit and a 64-bit result, no result.
        assert_eq!(declared("src/host_posix.rs", "GetTickCount64"), Some(">i"));
        assert_eq!(declared("src/host_posix.rs", "GetCurrentThreadId"), Some(">w"));
        assert_eq!(declared("src/host_posix.rs", "QueryPerformanceCounter"), Some("i>w"));
        assert_eq!(declared("src/host_posix.rs", "Sleep"), Some("i>v"));
        // Floats where the Windows ABI has them: `double sin(double)`, `ldexp(double, int)`.
        assert_eq!(declared("src/host_posix/ucrt/math.rs", "sin"), Some("d>d"));
        assert_eq!(declared("src/host_posix/ucrt/math.rs", "ldexp"), Some("di>d"));
        assert_eq!(declared("src/host_posix/ucrt/math.rs", "sinf"), Some("f>f"));
    }

    #[test]
    fn a_noted_address_has_its_signature() {
        extern "efiapi" fn probe(_: f64, _: *const u8, _: u32) -> u32 {
            0
        }
        // Not in a table of ours: the signature comes from nothing and the address is untouched.
        let at = fn_addr!(probe);
        assert_eq!(at, probe as *const () as usize);
        assert_eq!(sig_of(at), declared(file!(), "probe"));
        assert_eq!(declared(file!(), "probe"), Some("dii>w"));
    }

    #[test]
    fn every_registry_export_has_a_signature() {
        let unsigned: Vec<_> = crate::registry()
            .iter()
            .flat_map(|m| m.exports.iter().map(move |e| (m.name, e)))
            // `IID_*` are data, not functions.
            .filter(|(_, e)| !e.name.starts_with("IID_") && sig_of(e.address).is_none())
            .map(|(m, e)| format!("{m}::{}", e.name))
            .collect();
        assert!(unsigned.is_empty(), "no signature: {unsigned:?}");
    }

    /// With `VICTORY_IMPORTS` naming a `--list-imports` dry-run output, every import in it that we serve
    /// (the HLE registry or the POSIX host) has a signature.
    #[test]
    fn every_served_import_of_a_dry_run_has_a_signature() {
        use victory_loader::{ImportResolver, Resolution};
        let Some(list) = std::env::var_os("VICTORY_IMPORTS") else {
            return;
        };
        let text = std::fs::read_to_string(list).unwrap();
        let resolver = crate::HleResolver::new(crate::HostPolicy::default());
        let (mut served, mut unsigned, mut missing) = (0, vec![], 0);
        for item in text.lines().filter_map(|l| l.split_whitespace().last()) {
            let Some((dll, name)) = item.split_once('!') else {
                continue;
            };
            let symbol = victory_pe::ImportSymbol::Name { hint: 0, name: name.into() };
            match resolver.resolve(dll, &symbol) {
                Resolution::Found { address, .. } => {
                    served += 1;
                    if sig_of(address).is_none() {
                        unsigned.push(item.to_string());
                    }
                }
                Resolution::Missing => missing += 1,
            }
        }
        eprintln!("{served} served, {missing} not served, {} unsigned", unsigned.len());
        assert!(served > 50, "only {served} imports served");
        assert!(unsigned.is_empty(), "no signature: {unsigned:?}");
    }
}
