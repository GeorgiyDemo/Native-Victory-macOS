//! Several lifted images in one process: an exe (`pa.dll`) and two DLLs (`pb.dll`, `pc.dll`), each lifted into a
//! dylib of its own over one runtime. Calls cross the images through import tables, through function pointers
//! the images give each other, and through `GetProcAddress` of a lifted DLL.
#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

mod common;

use victory_hle::{HleResolver, HostPolicy, arm64, runtime};
use victory_loader::UnimplementedPolicy;
use victory_loader::native::NativeImage;
use victory_loader::native::dll::{WithTitleDlls, map_title_dll};

#[test]
fn images_call_each_other_by_import_pointer_and_getprocaddress() {
    victory_rt::open(&common::runtime_dylib()).expect("open");
    let (pe_a, unit_a) = common::build_unit("pa", "multi_pa.c", &["pb", "kernel32_dl"], &[]);
    let (pe_b, unit_b) = common::build_unit("pb", "multi_pb.c", &[], &[]);
    let (pe_c, unit_c) = common::build_unit("pc", "multi_pc.c", &["pb"], &[]);

    // The title's folder, as the loader sees it.
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("multi_title");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for (name, pe) in [("pa.dll", &pe_a), ("pb.dll", &pe_b), ("pc.dll", &pe_c)] {
        std::fs::write(dir.join(name), pe).unwrap();
    }

    let main = NativeImage::map(&pe_a, None).expect("map exe");
    let b = map_title_dll("pb.dll", &dir, &|_| true).expect("map pb");
    let c = map_title_dll("pc.dll", &dir, &|_| true).expect("map pc");
    assert_eq!((b.len(), c.len()), (1, 1), "pb is mapped once, pc finds it already there");
    let (b, c) = (&b[0].image, &c[0].image);
    assert!(main.base() != b.base() && b.base() != c.base());
    for (unit, image) in [(&unit_a, &main), (&unit_b, b), (&unit_c, c)] {
        // SAFETY: each image stays mapped for the process.
        unsafe { victory_rt::attach_unit(unit, image.base() as *mut u8, image.len()).expect("attach") };
    }
    for (name, image) in [("pa.dll", &main), ("pb.dll", b), ("pc.dll", c)] {
        runtime::set_image_path(image.base(), &dir.join(name));
    }
    runtime::set_main_image(main.base());

    let hle = HleResolver::new(HostPolicy::default());
    let resolver = WithTitleDlls { inner: &hle };
    for image in [&main, b, c] {
        let report = arm64::bind_imports(image, &resolver, UnimplementedPolicy::Trap).expect("bind");
        assert!(report.missing.is_empty(), "{:?}", report.missing);
    }
    // The exe's imports of pb.dll are pb's own lifted functions, not thunks.
    assert!(report_title_imports(&main) > 0);
    arm64::install();

    // SAFETY (for every call): exports of the exe with the arguments they take.
    let call = |name: &str, x: u64| unsafe { arm64::call_win64(main.export(name).unwrap(), &[x]) };
    // exe -> pb by import table.
    assert_eq!(call("probe_import", 5), 15);
    // exe -> pb -> (pointer) exe -> pb: pb_apply calls back into the exe, which calls pb again.
    assert_eq!(call("probe_via_pointer", 5), (3 * 5 + 7) + 1);
    // A pointer to another image's function, taken by the exe from its IAT.
    assert_eq!(call("probe_pointer_to_import", 5), 16);
    // exe -> pc found by LoadLibraryA and GetProcAddress; pc -> pb by import.
    assert_eq!(call("probe_getproc", 5), 3 * (5 + 1000));
    // exe -> pc (looked up) -> exe (pointer) -> pb.
    assert_eq!(call("probe_getproc_callback", 5), 2 * (3 * 5 + 7));
    // GetProcAddress and the import table give the same address.
    assert_eq!(call("probe_same_address", 0), 1);
    assert_eq!(
        victory_rt::image_base(),
        main.base(),
        "the first image attached is the main one"
    );
    for image in [&main, b, c] {
        assert!(victory_rt::contains(image.base() as u64 + 0x1000));
    }
}

fn report_title_imports(image: &NativeImage) -> usize {
    image.info().imports.iter().filter(|m| m.dll.eq_ignore_ascii_case("pb.dll")).map(|m| m.entries.len()).sum()
}
