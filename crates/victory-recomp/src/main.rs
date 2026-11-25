use anyhow::{Context, Result, bail};
use victory_recomp::{analyze, image::Image};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("analyze") => {
            let path = args.get(2).context("usage: victory-recomp analyze <pe>")?;
            let data = std::fs::read(path)?;
            let img = Image::parse(&data)?;
            print!("{}", analyze::analyze(&img)?.render());
            Ok(())
        }
        _ => bail!("usage: victory-recomp analyze <pe>"),
    }
}
