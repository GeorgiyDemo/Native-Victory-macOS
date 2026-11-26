//! Writes `$OUT_DIR/sigs.rs`: the signature string (`src/abi.rs`) of every `extern "system"` and
//! `extern "efiapi"` function in `src/`, read from its Rust declaration, so the arm64 host never keeps a
//! table by hand. Rust cannot turn a fn item with such an ABI into its signature (it implements none of the
//! `Fn` traits), so the declaration is read as text: the parameter list, split at top-level commas, and the
//! result. An argument is a float only if its type is `f32`, `f64` or an alias of them; anything else
//! (pointers, handles, structs by value, enums) is an integer.

use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            rust_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

/// Index just past the bracket that closes the one at `open`.
fn close_of(s: &[u8], open: usize) -> usize {
    let (o, c) = match s[open] {
        b'(' => (b'(', b')'),
        b'<' => (b'<', b'>'),
        _ => unreachable!(),
    };
    let mut depth = 0;
    let mut i = open;
    while i < s.len() {
        // `->` inside a fn-pointer type is not a closing angle bracket.
        if s[i] == b'-' && s.get(i + 1) == Some(&b'>') {
            i += 2;
            continue;
        }
        if s[i] == o {
            depth += 1;
        } else if s[i] == c {
            depth -= 1;
            if depth == 0 {
                return i + 1;
            }
        }
        i += 1;
    }
    s.len()
}

/// The top-level comma-separated pieces of `list`.
fn split_top(list: &str) -> Vec<&str> {
    let b = list.as_bytes();
    let (mut depth, mut start, mut parts) = (0i32, 0, vec![]);
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'-' if b.get(i + 1) == Some(&b'>') => i += 1,
            b'(' | b'<' | b'[' => depth += 1,
            b')' | b'>' | b']' => depth -= 1,
            b',' if depth == 0 => {
                parts.push(&list[start..i]);
                start = i + 1;
            }
            _ => {}
        }
        i += 1;
    }
    parts.push(&list[start..]);
    parts.into_iter().filter(|p| !p.trim().is_empty()).collect()
}

fn float_letter(ty: &str, aliases: &HashMap<String, String>) -> Option<char> {
    let ty = ty.trim();
    let ty = aliases.get(ty).map_or(ty, String::as_str);
    match ty {
        "f64" => Some('d'),
        "f32" => Some('f'),
        _ => None,
    }
}

/// 8- and 16-bit integer parameters: Apple's arm64 ABI lets the callee assume the caller extended them.
fn narrow_letter(ty: &str, aliases: &HashMap<String, String>) -> char {
    let ty = ty.trim();
    let ty = aliases.get(ty).map_or(ty, String::as_str);
    match ty {
        "u8" | "i8" | "bool" => 'b',
        "u16" | "i16" => 'h',
        _ => 'i',
    }
}

fn result_letter(ty: &str, aliases: &HashMap<String, String>) -> char {
    let ty = ty.trim();
    let ty = aliases.get(ty).map_or(ty, String::as_str);
    match ty {
        "" | "()" | "!" => 'v',
        "f64" => 'd',
        "f32" => 'f',
        // 32 bits or fewer: the upper half of the register is not defined, so the runtime zero-extends.
        "i32" | "u32" | "i16" | "u16" | "i8" | "u8" | "bool" | "c_int" | "c_uint" | "HRESULT" | "BOOL"
        | "NTSTATUS" | "DWORD" | "Bool" => 'w',
        _ => 'i',
    }
}

/// The functions a macro defines, `macro! { a, b => ...; c, d => ...; }` for a
/// `macro_rules! macro { ($($x:ident, $y:ident => ...;)*) => { fn $x ... fn $y ... } }`: the identifiers of
/// each entry before `=>` stand for the matcher's `ident` variables in order. Macros whose invocations are
/// shaped otherwise define no signature (their functions are called with integers).
fn expand_templates(
    rel: &str,
    text: &str,
    templated: &[(String, String, usize)],
    sigs: &mut BTreeMap<(String, String), String>,
) {
    if templated.is_empty() {
        return;
    }
    let b = text.as_bytes();
    let mut from = 0;
    while let Some(at) = text[from..].find("macro_rules! ") {
        let start = from + at + "macro_rules! ".len();
        from = start;
        let name_end = text[start..]
            .find(char::is_whitespace)
            .map_or(text.len(), |i| start + i);
        let name = &text[start..name_end];
        let Some(open) = text[name_end..].find('{').map(|i| name_end + i) else {
            continue;
        };
        let close = close_brace(b, open);
        let body = &text[open..close];
        let mut vars = vec![];
        let mut rest = &body[..body.find("=>").unwrap_or(0)];
        while let Some(i) = rest.find('$') {
            rest = &rest[i + 1..];
            let end = rest
                .find(|c: char| !(c.is_alphanumeric() || c == '_'))
                .unwrap_or(rest.len());
            if rest[end..].starts_with(":ident") {
                vars.push(rest[..end].to_string());
            }
        }
        let mine: Vec<(&String, &String)> = templated
            .iter()
            .filter(|(v, _, at)| vars.contains(v) && (open..close).contains(at))
            .map(|(v, sig, _)| (v, sig))
            .collect();
        if vars.is_empty() || mine.is_empty() {
            continue;
        }
        // Invocations: `name! { ... }` or `name!(...)`.
        let call = format!("{name}!");
        let mut at = 0;
        while let Some(i) = text[at..].find(&call) {
            let p = at + i + call.len();
            let is_definition = text[..p - call.len()].ends_with("macro_rules! ");
            at = p;
            let q = p + (text[p..].len() - text[p..].trim_start().len());
            if is_definition || !matches!(b.get(q), Some(b'{' | b'(')) {
                continue;
            }
            let end = if b[q] == b'(' { close_of(b, q) } else { close_brace(b, q) };
            for entry in text[q + 1..end - 1].split(';') {
                let lhs = entry.split_once("=>").map_or(entry, |(l, _)| l);
                let idents: Vec<&str> = lhs.split(',').map(str::trim).collect();
                let plain = |i: &&str| !i.is_empty() && i.chars().all(|c| c.is_alphanumeric() || c == '_');
                // The identifiers come first; other variables (`$ty`, `$kind`) follow and say nothing of the
                // signature's integers and pointers.
                if idents.len() < vars.len() || !idents[..vars.len()].iter().all(plain) {
                    continue;
                }
                for (var, sig) in &mine {
                    if let Some(k) = vars.iter().position(|v| v == *var) {
                        sigs.insert((rel.to_string(), idents[k].to_string()), (*sig).clone());
                    }
                }
            }
        }
    }
}

/// Index just past the `}` closing the `{` at `open`.
fn close_brace(s: &[u8], open: usize) -> usize {
    let mut depth = 0;
    for (i, &c) in s.iter().enumerate().skip(open) {
        match c {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return i + 1;
                }
            }
            _ => {}
        }
    }
    s.len()
}

fn main() {
    let src = Path::new("src");
    println!("cargo:rerun-if-changed=src");
    // Only the arm64 host reads the table.
    if std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() != Ok("aarch64") {
        let dir = PathBuf::from(std::env::var_os("OUT_DIR").unwrap());
        std::fs::write(dir.join("sigs.rs"), "pub(crate) static SIGS: &[(&str, &str, &str)] = &[];\n").unwrap();
        return;
    }
    // Our own sources, and the GPU crate's, whose functions `modules::d3d12x` serves.
    let gpu = Path::new("../victory-gpu/src");
    println!("cargo:rerun-if-changed=../victory-gpu/src");
    let mut files = vec![];
    rust_files(src, &mut files);
    rust_files(gpu, &mut files);
    files.sort();

    // `type Name = f32;` and the 32-bit result aliases, across the crate.
    let mut aliases: HashMap<String, String> = HashMap::new();
    let texts: Vec<(String, String)> = files
        .iter()
        .map(|p| {
            let rel = match p.strip_prefix(src) {
                Ok(r) => r.to_string_lossy().replace('\\', "/"),
                Err(_) => format!("gpu/{}", p.strip_prefix(gpu).unwrap().to_string_lossy().replace('\\', "/")),
            };
            (rel, std::fs::read_to_string(p).unwrap_or_default())
        })
        .collect();
    for (_, text) in &texts {
        for line in text.lines() {
            let line = line.trim().trim_start_matches("pub(crate) ").trim_start_matches("pub ");
            if let Some(rest) = line.strip_prefix("type ")
                && let Some((name, ty)) = rest.split_once('=')
            {
                let (name, ty) = (name.trim(), ty.trim().trim_end_matches(';').trim());
                if !name.contains('<') {
                    aliases.insert(name.to_string(), ty.to_string());
                }
            }
        }
    }

    // (file, function) -> signature.
    let mut sigs: BTreeMap<(String, String), String> = BTreeMap::new();
    for (rel, text) in &texts {
        let s = text.as_bytes();
        let mut templated: Vec<(String, String, usize)> = vec![];
        let mut from = 0;
        while let Some(at) = text[from..].find("extern \"") {
            let start = from + at;
            from = start + 8;
            let rest = &text[start + 8..];
            let Some(q) = rest.find('"') else { continue };
            if !matches!(&rest[..q], "system" | "efiapi" | "C" | "win64") {
                continue;
            }
            let after = rest[q + 1..].trim_start();
            let Some(after) = after.strip_prefix("fn ") else {
                continue;
            };
            // `fn $name(...)` inside a macro: a template, named by the invocations below.
            let templ = after.starts_with('$');
            let name_end = after
                .char_indices()
                .skip(usize::from(templ))
                .find(|&(_, c)| !(c.is_alphanumeric() || c == '_'))
                .map_or(after.len(), |(i, _)| i);
            let name = &after[..name_end];
            if name.is_empty() || name == "$" {
                continue;
            }
            // The offset of the name in `text`, to find the parameter list.
            let name_at = text.len() - after.len();
            let mut p = name_at + name_end;
            while text.as_bytes()[p].is_ascii_whitespace() {
                p += 1;
            }
            if s[p] == b'<' {
                p = close_of(s, p);
                while s[p].is_ascii_whitespace() {
                    p += 1;
                }
            }
            if s[p] != b'(' {
                continue;
            }
            let end = close_of(s, p);
            let params = &text[p + 1..end - 1];
            let tail = text[end..].trim_start();
            let ret = match tail.strip_prefix("->") {
                Some(r) => {
                    let stop = r.find(['{', ';']).unwrap_or(r.len());
                    r[..stop].trim().trim_end_matches("where").trim()
                }
                None => "",
            };
            let mut sig = String::new();
            for param in split_top(params) {
                // `name: Type`; a pattern with a colon in a path (`std::x`) cannot be a parameter name.
                let ty = param.split_once(':').map_or(param, |(_, t)| t);
                sig.push(float_letter(ty, &aliases).unwrap_or_else(|| narrow_letter(ty, &aliases)));
            }
            sig.push('>');
            sig.push(result_letter(ret, &aliases));
            if templ {
                templated.push((name[1..].to_string(), sig, start));
            } else {
                sigs.insert((rel.clone(), name.to_string()), sig);
            }
        }
        expand_templates(rel, text, &templated, &mut sigs);
    }

    let mut out = String::from("pub(crate) static SIGS: &[(&str, &str, &str)] = &[\n");
    for ((file, name), sig) in &sigs {
        writeln!(out, "    ({file:?}, {name:?}, {sig:?}),").unwrap();
    }
    out.push_str("];\n");
    let dir = PathBuf::from(std::env::var_os("OUT_DIR").unwrap());
    std::fs::write(dir.join("sigs.rs"), out).unwrap();
}
