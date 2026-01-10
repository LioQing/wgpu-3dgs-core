macro_rules! cargo {
    ($arg:literal $(, $fmt_args:expr),* $(,)?) => {{
        let cmd_arg = format!($arg, $($fmt_args),*);
        let args = cmd_arg.split_whitespace();

        println!("cargo {}", cmd_arg);

        let status = std::process::Command::new("cargo")
            .args(args)
            .status()
            .expect("failed to execute process");

        assert!(status.success(), "command 'cargo {}' failed", cmd_arg);
    }};
}

fn main() {
    let exe_path = std::env::current_exe().expect("current exe");

    let mut manifest_path = exe_path.parent().expect("exe parent").to_path_buf();
    while std::fs::read_dir(&manifest_path)
        .expect("read dir")
        .find(|entry| entry.as_ref().expect("entry").file_name() == "Cargo.toml")
        .is_none()
    {
        manifest_path = manifest_path.parent().expect("parent").to_path_buf();
    }

    let coverage_path = manifest_path.join("coverage");
    let examples_path = manifest_path.join("examples");
    let lcov_path = coverage_path.join("lcov.info");
    let lcov_path_str = lcov_path.to_str().expect("lcov path");
    let badge_path = coverage_path.join("badge.json");
    let model_ply_path = examples_path.join("model.ply");
    let model_ply_path_str = model_ply_path.to_str().expect("model ply path");
    let model_spz_path = examples_path.join("model.spz");
    let model_spz_path_str = model_spz_path.to_str().expect("model spz path");
    let output_ply_path = coverage_path.join("output.ply");
    let output_ply_path_str = output_ply_path.to_str().expect("output ply path");
    let output_spz_path = coverage_path.join("output.spz");
    let output_spz_path_str = output_spz_path.to_str().expect("output spz path");

    println!("Running coverage...");

    cargo!("llvm-cov clean --workspace");

    println!("Running 'write-ply' example");
    cargo!("llvm-cov run --example write-ply -- {output_ply_path_str}");

    println!("Running 'read-ply' example");
    cargo!("llvm-cov run --example read-ply -- {model_ply_path_str}");

    println!("Running 'write-spz' example");
    cargo!("llvm-cov run --example write-spz -- {output_spz_path_str}");

    println!("Running 'read-spz' example");
    cargo!("llvm-cov run --example read-spz -- {model_spz_path_str}");

    println!("Running doctests");
    // `--doctests` flag is currently unstable
    // cargo!("llvm-cov --no-report --doctests");
    cargo!("test --doc");

    println!("Running tests");
    cargo!("llvm-cov --no-report nextest");

    println!("Generating coverage report");
    cargo!("llvm-cov report --lcov --output-path {lcov_path_str}");

    println!("Generating badge");

    let lcov = std::fs::read_to_string(&lcov_path).expect("read lcov.info");
    let mut total: u64 = 0;
    let mut covered: u64 = 0;

    for line in lcov.lines() {
        if !line.starts_with("DA:") {
            continue;
        }

        let mut parts = line[3..].split(',');
        let _line_number = parts.next();
        let hits_str = parts.next();

        let Some(hits_str) = hits_str else {
            continue;
        };
        let Ok(hits) = hits_str.parse::<u64>() else {
            continue;
        };

        total += 1;
        if hits != 0 {
            covered += 1;
        }
    }

    let badge_percentage: u64 = if total == 0 {
        100
    } else {
        ((covered as f32 / total as f32) * 100.0).round() as u64
    };

    let badge_color = if badge_percentage >= 80 {
        "brightgreen"
    } else if badge_percentage >= 50 {
        "yellow"
    } else {
        "red"
    };

    let badge_json = format!(
        r#"
{{
    "schemaVersion": 1,
    "label": "coverage",
    "message": "{badge_percentage}%",
    "color": "{badge_color}"
}}
        "#
    );
    std::fs::write(&badge_path, badge_json.trim().to_owned() + "\n").expect("write badge.json");

    println!("Cleaning up");

    std::fs::remove_file(&output_ply_path).expect("remove output ply");
    std::fs::remove_file(&output_spz_path).expect("remove output spz");

    println!("Done");
}
