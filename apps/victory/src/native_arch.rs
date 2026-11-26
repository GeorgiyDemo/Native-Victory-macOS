//! What differs between the two macOS hosts of `victory run`: under Rosetta (x86-64) the title's own
//! code runs as it is behind a GS-swapping bridge; in an arm64 process it runs recompiled
//! (`victory-recomp`, built outside the repo) behind `victory-rt`. Everything else in `native_run` is shared.

use anyhow::Result;
use victory_loader::native::NativeImage;
use victory_loader::{BindReport, ImportResolver, UnimplementedPolicy};

#[cfg(target_arch = "x86_64")]
mod imp {
    use super::*;

    pub fn has_unit(_dll: &str) -> bool {
        true
    }

    pub fn start() -> Result<()> {
        Ok(())
    }

    pub fn attach(_name: &str, _image: &NativeImage) -> Result<()> {
        Ok(())
    }

    pub fn bind(
        image: &NativeImage,
        resolver: &dyn ImportResolver,
        policy: UnimplementedPolicy,
    ) -> Result<BindReport> {
        let report = victory_loader::native::bind_imports(image, resolver, policy)?;
        image.finalize()?;
        Ok(report)
    }

    pub fn gpu_bridge() -> victory_gpu::object::Bridge {
        use victory_loader::native as n;
        victory_gpu::object::Bridge {
            is_bridged: n::is_bridged,
            copy_of: n::bridged_copy_of,
            expose: |this, len, area| {
                // SAFETY: forwarded: a live object of ours whose first word is such a table.
                unsafe { n::expose_object(this, len, area) }
            },
            declare_win64: n::declare_win64,
        }
    }
}

/// The recompiled title lives in `$VICTORY_LIFT_DIR`: `rt/librt.dylib` (the runtime, built `-DRT_MULTI`) and, per image,
/// `<name>/lib<name>.dylib` where `<name>` is the file name in lower case without `.exe` or `.dll`.
#[cfg(target_arch = "aarch64")]
mod imp {
    use std::path::PathBuf;

    use anyhow::Context;

    use super::*;

    fn dir() -> Result<PathBuf> {
        std::env::var_os("VICTORY_LIFT_DIR")
            .map(PathBuf::from)
            .context("VICTORY_LIFT_DIR names the folder of the recompiled title (see docs/recompiler.md)")
    }

    fn unit_of(file: &str) -> Option<PathBuf> {
        let lower = file.to_ascii_lowercase();
        let stem = lower
            .strip_suffix(".dll")
            .or_else(|| lower.strip_suffix(".exe"))
            .unwrap_or(&lower);
        let unit = dir().ok()?.join(stem).join(format!("lib{stem}.dylib"));
        unit.is_file().then_some(unit)
    }

    pub fn has_unit(dll: &str) -> bool {
        unit_of(dll).is_some()
    }

    /// Loads the runtime and gives the loader its way into recompiled code. Once.
    pub fn start() -> Result<()> {
        victory_rt::open(&dir()?.join("rt/librt.dylib"))?;
        victory_hle::arm64::install();
        Ok(())
    }

    /// Attaches the recompiled code of the mapped image `file`. The first image attached is the main one.
    pub fn attach(file: &str, image: &NativeImage) -> Result<()> {
        let unit = unit_of(file).with_context(|| format!("no recompiled unit for {file} in VICTORY_LIFT_DIR"))?;
        // SAFETY: the image stays mapped for the life of the process, and is the one the unit was lifted from
        // (the caller maps the file the unit was made from).
        unsafe { victory_rt::attach_unit(&unit, image.base() as *mut u8, image.len()) }
    }

    /// The image stays writable (lifted code does not execute it, but writes its data).
    pub fn bind(
        image: &NativeImage,
        resolver: &dyn ImportResolver,
        policy: UnimplementedPolicy,
    ) -> Result<BindReport> {
        victory_hle::arm64::bind_imports(image, resolver, policy)
    }

    pub fn gpu_bridge() -> victory_gpu::object::Bridge {
        victory_hle::arm64::gpu_bridge()
    }
}

pub use imp::*;
