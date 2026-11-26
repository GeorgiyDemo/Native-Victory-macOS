//! MSVC C++ exceptions over lifted images. Three DLLs built for the MSVC target, as the title's are: `vcrt.dll`, a
//! stand-in for vcruntime140 (`_CxxThrowException`, `__CxxFrameHandler3`: the handler is lifted code, not served by us),
//! `cxxb.dll`, and `cxxa.dll`, which imports both. A throw is `RaiseException(0xE06D7363)`; the host dispatcher calls
//! each frame's lifted handler, which finds the catch from `FuncInfo`, has `RtlUnwindEx` run the destructor funclets
//! of the frames below and enter the catch as a call from the catching frame, and resumes after the `try` at the
//! continuation address the catch funclet returns: a landing pad the recompiler found from `FuncInfo` and the funclet.
#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

mod common;

use victory_hle::{HleResolver, HostPolicy, arm64, runtime};
use victory_loader::UnimplementedPolicy;
use victory_loader::native::NativeImage;
use victory_loader::native::dll::{WithTitleDlls, map_title_dll};

#[test]
fn cxx_throw_catch_rethrow_and_destructors_across_lifted_images() {
    victory_rt::open(&common::runtime_dylib()).expect("open");
    let (pe_v, unit_v) = common::build_msvc_unit("vcrt", &["vcrt.c"], &["ntdll_cxx"]);
    let (pe_b, unit_b) = common::build_msvc_unit("cxxb", &["cxx_b.cpp", "typeinfo.c"], &["vcrt"]);
    let (pe_a, unit_a) = common::build_msvc_unit("cxxa", &["cxx_a.cpp", "typeinfo.c"], &["vcrt", "cxxb"]);
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("cxx_title");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for (name, pe) in [("vcrt.dll", &pe_v), ("cxxb.dll", &pe_b), ("cxxa.dll", &pe_a)] {
        std::fs::write(dir.join(name), pe).unwrap();
    }

    // What the recompiler read from the C++ data: the catch's continuation is a landing pad of the function (found from
    // `FuncInfo` and the catch funclet), and the handler, destructor and catch funclets are seeds.
    {
        let image = victory_recomp::image::Image::parse(&pe_a).unwrap();
        let rva = image.info.exports.iter().find(|e| e.name.as_deref() == Some("a_dtors")).unwrap().rva.unwrap();
        let func = *image.functions().unwrap().iter().find(|f| f.begin == rva).unwrap();
        let pads = image.landing_pads(&func);
        assert!(!pads.is_empty() && pads.iter().all(|p| (func.begin..func.end).contains(p)), "{pads:x?}");
        assert!(image.handler_seeds(&func).len() >= 3, "handler thunk, destructor funclet, catch funclet");
    }

    let main = NativeImage::map(&pe_a, None).expect("map");
    let dlls = [map_title_dll("vcrt.dll", &dir, &|_| true).unwrap(), map_title_dll("cxxb.dll", &dir, &|_| true).unwrap()];
    let (v, b) = (&dlls[0][0].image, &dlls[1][0].image);
    for (unit, image) in [(&unit_a, &main), (&unit_v, v), (&unit_b, b)] {
        // SAFETY: each image stays mapped for the process.
        unsafe { victory_rt::attach_unit(unit, image.base() as *mut u8, image.len()).expect("attach") };
    }
    runtime::set_main_image(main.base());
    let hle = HleResolver::new(HostPolicy::default());
    let resolver = WithTitleDlls { inner: &hle };
    for image in [&main, v, b] {
        let report = arm64::bind_imports(image, &resolver, UnimplementedPolicy::Trap).expect("bind");
        assert!(report.missing.is_empty(), "{:?}", report.missing);
    }
    arm64::install();

    // SAFETY (for every call): exports of the exe with the arguments they take.
    let call = |name: &str, args: &[u64]| unsafe { arm64::call_win64(main.export(name).unwrap(), args) } as u32 as i32;
    let log_of = |name: &str, args: &[u64]| -> (i32, Vec<i32>) {
        let (reset, get) = (v.export("vc_log_reset").unwrap(), v.export("vc_log_get").unwrap());
        // SAFETY: the stand-in runtime's exports.
        unsafe {
            arm64::call_win64(reset, &[]);
            let r = call(name, args);
            let mut out = [0i32; 64];
            let n = arm64::call_win64(get, &[out.as_mut_ptr() as u64]) as usize;
            (r, out[..n].to_vec())
        }
    };

    for round in 0..2 {
        // throw in one image, catch by type in the other
        assert_eq!(call("a_catch_int", &[7]), 8, "round {round}");
        // destructors of both frames run, the nearer the throw first, before the catch block
        assert_eq!(log_of("a_dtors", &[7]), (7, vec![5, 1, 101, 105, 777]), "round {round}");
        // catch (...)
        assert_eq!(log_of("a_catch_all", &[9]), (42, vec![1, 101, 888]), "round {round}");
        // caught in the DLL, logged, thrown again, caught in the exe: the DLL frame's guard is gone before the catch ran
        assert_eq!(log_of("a_rethrown", &[3]), (1003, vec![2, 102, 50]), "round {round}");
        // a struct, caught by reference
        assert_eq!(call("a_thing", &[4]), 45, "round {round}");
        // the inner catch (int) does not take a double; the outer catch (...) does
        assert_eq!(call("a_mismatch", &[]), 5, "round {round}");
        // thrown in the exe, caught in the DLL, through a function pointer
        assert_eq!(call("a_through_b", &[6]), 14, "round {round}");
        // a frame with a destructor and no handler between the throw and the catch
        assert_eq!(log_of("a_nested", &[8]), (2008, vec![3, 103]), "round {round}");
    }
}
