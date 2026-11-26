//! MSVC C++ exceptions against a real `vcruntime140.dll`, lifted like the title's: `VICTORY_REAL_VCRT` names the file
//! (it belongs to the title's folder and is never copied into the repository; the test skips without it). Own C++
//! (`cxx_c.cpp`, `/EHsc` tables, `__CxxFrameHandler3`) throws and catches across frames: destructors of several frames,
//! a try inside a catch, a throw from a catch, rethrow, catch by base class and a thrown object with a copy constructor.
#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

mod common;

use victory_hle::{HleResolver, HostPolicy, arm64, runtime};
use victory_loader::UnimplementedPolicy;
use victory_loader::native::NativeImage;
use victory_loader::native::dll::{WithTitleDlls, map_title_dll};

#[test]
fn real_vcruntime140_throws_and_catches() {
    let Some(real) = std::env::var_os("VICTORY_REAL_VCRT") else {
        eprintln!("skipped: VICTORY_REAL_VCRT names the title's vcruntime140.dll");
        return;
    };
    let real_pe = std::fs::read(&real).expect("read vcruntime140.dll");
    victory_rt::open(&common::runtime_dylib()).expect("open");
    let unit_r = common::unit_of_pe("realvcrt", "vcruntime140.dll", &real_pe);
    let (pe_l, unit_l) = common::build_msvc_unit("vclog", &["vclog.c"], &[]);
    let (pe_c, unit_c) = common::build_msvc_unit("cxxc", &["cxx_c.cpp", "typeinfo.c"], &["vcruntime140", "vclog"]);
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("cxx_real_title");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("vcruntime140.dll"), &real_pe).unwrap();
    std::fs::write(dir.join("vclog.dll"), &pe_l).unwrap();

    let main = NativeImage::map(&pe_c, None).expect("map");
    let r = map_title_dll("vcruntime140.dll", &dir, &|_| true).unwrap();
    let l = map_title_dll("vclog.dll", &dir, &|_| true).unwrap();
    let (r, l) = (&r[0].image, &l[0].image);
    for (unit, image) in [(&unit_c, &main), (&unit_r, r), (&unit_l, l)] {
        // SAFETY: each image stays mapped for the process.
        unsafe { victory_rt::attach_unit(unit, image.base() as *mut u8, image.len()).expect("attach") };
    }
    runtime::set_main_image(main.base());
    eprintln!("bases: exe {:#x} vcruntime {:#x} log {:#x}", main.base(), r.base(), l.base());
    let hle = HleResolver::new(HostPolicy::default());
    let resolver = WithTitleDlls { inner: &hle };
    for image in [&main, r, l] {
        let report = arm64::bind_imports(image, &resolver, UnimplementedPolicy::Trap).expect("bind");
        assert!(report.missing.is_empty(), "{:?}", report.missing);
    }
    arm64::install();
    // The runtime's DllMain sets up its per-thread data (FlsAlloc) before anything throws.
    // SAFETY: bound images.
    assert!(unsafe { victory_loader::native::attach_dll(r, 1, 1) });

    // SAFETY (for every call): exports of the test images with the arguments they take.
    let call = |name: &str, args: &[u64]| unsafe { arm64::call_win64(main.export(name).unwrap(), args) } as u32 as i32;
    let log_of = |name: &str, args: &[u64]| -> (i32, Vec<i32>) {
        let (reset, get) = (l.export("vc_log_reset").unwrap(), l.export("vc_log_get").unwrap());
        unsafe {
            arm64::call_win64(reset, &[]);
            let r = call(name, args);
            let mut out = [0i32; 64];
            let n = arm64::call_win64(get, &[out.as_mut_ptr() as u64]) as usize;
            (r, out[..n].to_vec())
        }
    };
    for round in 0..2 {
        assert_eq!(log_of("c_unwind", &[5]), (5, vec![30, 20, 10, 110, 120, 130, 900]), "unwind, round {round}");
        assert_eq!(log_of("c_nested", &[4]), (6, vec![910, 911, 912]), "nested, round {round}");
        assert_eq!(log_of("c_throw_in_catch", &[8]), (9, vec![40, 140, 920]), "throw in catch, round {round}");
        assert_eq!(log_of("c_rethrow", &[11]), (22, vec![930, 931]), "rethrow, round {round}");
        assert_eq!(call("c_base", &[3]), 1003, "base class, round {round}");
        let (v, log) = log_of("c_object", &[6]);
        assert_eq!(v, 6, "object, round {round}");
        assert!(log.contains(&940) && log.contains(&506), "object log {log:?}");
    }
}
